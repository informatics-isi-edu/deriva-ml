# Denormalization — Architecture & Contract

**Status:** Authoritative as of 2026-05-21.
**Scope:** The full denormalization pipeline — planner, source-mode
selection, row population, INSERT semantics, and the cross-call/
cross-session caching behavior. This is the single design reference
for engineers working on `deriva_ml.local_db.denormalize`,
`local_db.denormalizer`, `local_db.paged_fetcher`, and
`model.denormalize_planner`.

**Companion docs:**

- `docs/superpowers/specs/2026-04-17-denormalization-semantics-design.md` —
  the eight semantic Rules (auto-inference of `row_per`,
  downstream-leaf rejection, ambiguity detection, orphan emission,
  etc.). Still authoritative for planner-level semantics.
- `docs/superpowers/specs/2026-04-15-unified-local-db-design.md` and
  `2026-04-15-unified-local-db-phase2-design.md` — the per-catalog
  workspace layout (ATTACH'd schema files, `main.db` registries) and
  the original Phase 2 cutover from two engines to one. Still
  authoritative for storage; the present doc supersedes their
  fetch/insert-side claims.
- `docs/concepts/` user-facing guides (`features.md`, `datasets.md`).
- `src/deriva_ml/local_db/README.md` — short orienting README in the
  source tree.

**What this doc supersedes:** the implementation plans under
`docs/superpowers/plans/` for Phases 1 & 2 of the unified local DB,
and the older multi-hop-FK / join-tree-refactor / duplicate-association
plans. Those documents were operational; they recorded
how the code *got* into its current shape, not what shape it has.
The plans live on under `docs/superpowers/plans/archive/` for
historical context; do not cite them from new code or as the
"current architecture."

---

## 1. What denormalization is, in one paragraph

A user has a `Dataset` (live, on the catalog) or a `DatasetBag`
(downloaded snapshot). They ask: "give me a wide table joining these
N tables in the dataset's scope, with one row per X." The
denormalization pipeline plans the join (which FK chains to follow,
which table is the "leaf" producing one row per output row, what
columns each table contributes), populates the local SQLite cache
from whichever source applies (live catalog, downloaded bag, or
"local" for tests), runs the join in SQL against the local cache,
and returns the rows. The user sees a pandas DataFrame or a row
iterator.

---

## 2. Pipeline architecture

```
┌────────────────────────────────────────────────────────────────────┐
│ Caller                                                             │
├────────────────────────────────────────────────────────────────────┤
│ Dataset.get_denormalized_as_dataframe(...)        ← live catalog   │
│ DatasetBag.get_denormalized_as_dataframe(...)     ← downloaded bag │
└────────────────────────────────────────────────────────────────────┘
              │                                           │
              ▼                                           ▼
       Denormalizer(dataset_like)              Denormalizer(bag)
       source = "catalog"                      source = "local"
       paged_client = ErmrestPagedClient                  │
              │                                           │
              └──────────┬────────────────────────────────┘
                         ▼
                   _denormalize_impl(...)                          ← src/deriva_ml/local_db/denormalize.py
                         │
                         ├─ _prepare_wide_table(...)               ← src/deriva_ml/model/denormalize_planner.py
                         │    pure model code; no I/O
                         │    returns:  join_tables, column_specs, multi_schema
                         │
                         ├─ if source == "catalog":
                         │    _populate_from_catalog(...)
                         │      └─ new PagedFetcher(client, engine)         ← single-use; one per call
                         │         └─ _populate_from_catalog_inner(...)
                         │            ├─ fetch_by_rids("deriva-ml:Dataset", rids=[dataset_rids], rid_column="RID")
                         │            └─ for each (path, conditions) in join_tables:
                         │                 for each table_name in path[1:]:
                         │                   _collect_fk_values(...) → (rids_to_fetch, fk_column_on_target)
                         │                   fetcher.fetch_by_rids(qualified, rids_to_fetch,
                         │                                          target_table=target_orm.__table__,
                         │                                          rid_column=fk_column_on_target)
                         │
                         ├─ Step 4: build SQLAlchemy SELECT per element path
                         ├─ Step 5: UNION → execute against engine → rows
                         └─ return DenormalizeResult
```

Three source modes are supported by `_denormalize_impl`:

- **`source="catalog"`** — the live path. A `PagedClient`
  (`ErmrestPagedClient` by default) fetches rows from the
  ERMrest catalog into the workspace's local SQLite. Used by
  `Dataset.get_denormalized_as_dataframe`.
- **`source="local"`** — the bag / unit-test path. Rows are
  already in the engine (a downloaded bag's SQLite files, or a
  test fixture's pre-populated tables). No HTTP. Used by
  `DatasetBag.get_denormalized_as_dataframe`.
- **`source="slice"`** — a slice (a subset of a workspace) has
  been ATTACH'd into the engine and rows are visible that way.
  Used by some bag-style flows.

The SQL emission code (Steps 4 & 5) is identical across the three
modes; only the row-population side differs.

---

## 3. State, ownership, and lifetime

Read this table carefully. Every denormalize bug we have shipped has
been a state-ownership confusion.

| State | Type | Owner | Lifetime | Visible to |
|---|---|---|---|---|
| `model` (catalog schema + FK graph) | in-memory | `DerivaML` / `DatasetBag` | session | all calls in the session |
| `engine` (SQLAlchemy `Engine` over a per-catalog SQLite directory) | sqlite-backed | `Workspace` | **persists to disk; survives the process** | every call against the same catalog, across processes and sessions |
| `orm_resolver` (table-name → SQLAlchemy ORM class) | in-memory | `LocalSchema` | session | all calls |
| `PagedClient` (`ErmrestPagedClient`) | in-memory | `Denormalizer` instance | one Denormalizer | one denormalize call |
| `PagedFetcher` instance | in-memory | `_populate_from_catalog` | **one call to `_denormalize_impl`** | only that fetcher's own methods |
| Local SQLite rows (`Image`, `Dataset`, `Execution_Image_*`, …) | sqlite-backed | engine | **persists to disk; survives the process** | every later call, every later session |

### The two truths to remember

1. **`PagedFetcher` is single-use.** A fresh one is constructed every
   time `_populate_from_catalog` is called. Any cache that lives on
   a `PagedFetcher` instance is empty at the start of each
   denormalize call. **In-memory state on a `PagedFetcher` cannot
   carry information across denormalize calls.**

2. **The local SQLite is the only durable, shared state.** It
   accumulates rows from every prior fetch. It persists across
   processes, across DerivaML instances, across days. A denormalize
   call doesn't start from a blank engine; it starts from whatever
   the engine has accumulated.

These two facts together generate every subtle denormalize bug.

---

## 4. The fetcher contract

`PagedFetcher` is a **thin transport adapter** with three jobs:

1. **Fetch.** Given `(table, rids, rid_column)`, issue HTTP requests
   to ERMrest and return rows. The `rid_column` may be a primary key
   (most often `"RID"`) *or* a foreign-key column (e.g.
   `"Image"` on `Execution_Image_<Feature>`). When `rid_column` is
   an FK, the server may return many rows per filter value;
   `PagedFetcher` must not assume one-to-one.
2. **Insert.** Write rows into the engine's local SQLite via
   `_insert_rows`. Handle conflicts cleanly: a row whose RID already
   exists must not crash the insert and must not be overwritten.
3. **Count.** Memoise per-table row counts so cardinality heuristics
   don't re-query.

### What it does NOT do

- **Network dedup based on engine state.** Do not consult the local
  SQLite to decide whether to issue a fetch. The local cache is a
  write-through history of past fetches, not an authoritative answer
  to "is the server's row for this value already represented here?"
  This is the bug-class that produced finding **A01** (2026-05-21):
  v1.37.2 of this library hydrated a "seen" set from the engine
  keyed by the caller's `rid_column`, and skipped fetches whenever
  the FK values appeared in the engine — silently dropping all but
  one feature row per anchor.
- **Make idempotency claims via row dedup.** Use the database's
  built-in UNIQUE constraint as the authority. Conflict handling
  belongs at the INSERT, not at the fetch.
- **Outlive a single denormalize call.** No long-lived
  `PagedFetcher` instances exist or are intended to exist.

### Public API summary

```python
PagedFetcher(client: PagedClient, engine: Engine)

  .fetch_by_rids(table, rids, target_table, rid_column="RID",
                 batch_size=DEFAULT_BATCH_SIZE,
                 max_url_bytes=DEFAULT_MAX_URL_BYTES) -> int
  .fetch_predicate(table, predicate, target_table, sort=("RID",),
                   page_size=DEFAULT_PAGE_SIZE) -> int
  .fetch_by_rids_or_predicate(...)             # cardinality heuristic
  .fetched_rids(table, target_table=None) -> set[str]
```

Both `fetch_*` methods return the number of rows actually inserted
(after conflict skipping). `fetched_rids` returns the set of RIDs in
the target_table after the call.

---

## 5. The INSERT contract

`_insert_rows(target_table, rows)` is **the only place that mutates
engine state**. Its contract:

- **Input.** A list of dicts. Each dict has at least the columns
  the target_table declares; extras are silently dropped. Each dict
  has a `"RID"` key (every Deriva-managed table has `RID` as its
  primary key — this is invariant across the Deriva catalog
  schema).
- **Behavior.** For each row: if a row with that RID is already in
  the target_table, **skip** (do not update, do not crash). Otherwise
  insert. Use dialect-native upsert when available
  (`sqlite_insert(...).on_conflict_do_nothing(index_elements=["RID"])`
  for SQLite; equivalent for any other engine added later).
- **Output.** Number of rows actually written (skipped rows do not
  count).
- **Invariant after call.** Every RID from `rows` appears in
  `target_table` exactly once.
- **Idempotent.** Calling twice with the same `rows` yields the
  same engine state and second call returns `0`.

This is **stateless** — no cache to hydrate, no per-table set to
track. The database's UNIQUE constraint is the authority. This is
the only sustainable design: any client-side cache that mirrors the
engine's content will eventually disagree with it, and that
disagreement is what we keep tripping on.

---

## 6. The denormalize contract

`Denormalizer(dataset).as_dataframe(include_tables, *, row_per, via,
ignore_unrelated_anchors)`:

- **Input.** A `DatasetLike` (live `Dataset`, `DatasetBag`, or test
  fixture), a list of table names to include, and the four
  semantic-rule knobs documented in the semantics spec.
- **Behavior.**
  1. Resolves source mode from the dataset type.
  2. Plans the join via `_prepare_wide_table` (planner is pure;
     no I/O).
  3. If `source="catalog"`: populates the local SQLite via
     `_populate_from_catalog`. This step may be a no-op for tables
     whose rows are already present, but **must always re-issue the
     fetch for every (table, rid_column, rid) tuple in the plan** —
     because the server is the only authority for "what rows match
     this query right now."
  4. Emits SQL against the local SQLite, runs it, materialises rows.
  5. Returns a `DenormalizeResult` (rows + column metadata).
- **Output row count.** Determined by the eight Rules in the
  semantics spec. The pipeline is *correct* iff that row count
  matches the server's reality, modulo the freshness caveat below.

### Freshness caveat

The local SQLite is a **write-through cache of past fetches**, not a
live view of the server. If the catalog mutates between two
denormalize calls in the same process, the engine still holds the
old rows in addition to whatever the new fetch adds. Specifically:

- **Insertions on the server are observed correctly.** Step 3's
  fetch picks up new rows and `_insert_rows` adds them.
- **Deletions on the server are NOT observed.** A row deleted
  server-side after we cached it remains in the engine and
  participates in subsequent JOINs.
- **Updates on the server are NOT observed.** Same reason.

This is a real design limitation, not a bug. It is acceptable
because (a) Deriva data is largely append-only in practice, (b) the
cache is per-catalog-snapshot conceptually (callers who need a
snapshot should `download_dataset_bag(version=...)`), and (c) the
fix is invasive (we'd need invalidation hooks tied to every server
mutation).

Documented; tracked as a known limitation. Tests assert the positive
behavior (new server rows show up) but `xfail` the
deletion/update freshness cases until we decide to fix.

---

## 7. Fragility map — known bug patterns and how the contract prevents them

| # | Pattern | Cause | Status |
|---|---|---|---|
| **F1** | Re-denormalize crashes with `UNIQUE constraint failed: Dataset.RID` | Plain `INSERT` against engine that has prior rows. | Fixed by the INSERT-OR-IGNORE contract in §5. Originally surfaced as 2026-05-21 finding 05 in the model-template e2e run. |
| **F2** | Re-denormalize silently drops rows | `_get_seen` (v1.37.2) hydrated a dedup map from the engine keyed by the caller's `rid_column`. For FK columns with N rows per value, this collapsed the fetch to one row per FK. | Fixed by removing the engine-hydrated seen-set entirely. The fetcher does not dedup based on engine state; conflict handling belongs at INSERT. Originally surfaced as 2026-05-21 finding A01. |
| F3 | Stale local data when server mutates between calls (deletes, updates) | Cache is write-through, never invalidated. | Documented as a known limitation. See §6. Test cases marked `xfail`. |
| F4 | `_collect_fk_values` walks "values currently present in the engine" to decide what to fetch from the server. If a parent table's membership was updated server-side after the engine cached it, downstream fetches use the stale parent set. | Same root cause as F3. | Same status — known limitation, tracked. |
| F5 | Path-walk order silently determines which rows get loaded when two element tables share intermediate tables | `_populate_from_catalog_inner` walks `join_tables` in dict iteration order; the order isn't part of the documented contract. | Not currently a bug, because every fetch must visit each (table, rid_column, rid) tuple at least once. Pin to the test matrix in §8 so a regression here would be caught. |

---

## 8. Test matrix

The matrix below pins the contracts in §4–§6 against concrete pytest
cases. Each row is one test. Cases against a live catalog are
marked `live`; cases against an in-memory engine + `FakePagedClient`
are marked `unit`.

### Layer A — `_insert_rows` (the heart of the new contract)

| # | Scenario | Kind | Assert |
|---|---|---|---|
| A.1 | Insert N rows into empty target | unit | engine has N rows; returned count == N |
| A.2 | Insert overlapping rows twice into one target | unit | engine has N rows; 2nd call returns 0 |
| A.3 | Insert rows where some RIDs already exist | unit | engine = pre ∪ new; returned count == \|new only\| |
| A.4 | Insert rows missing RID — pin behavior | unit | raise (rows without RID are a programming error) |
| A.5 | Insert rows with extra columns | unit | extras silently dropped (existing contract) |
| A.6 | Two `PagedFetcher` instances inserting overlapping rows into one engine | unit | no crash; engine = union |
| A.7 | Fresh fetcher against pre-populated engine (cross-session simulation) | unit | no crash; existing rows preserved |
| A.8 | Engine has rows with `RID=NULL` (defensive) | unit | treated as never-seen; insert proceeds (or raise — pin one) |

### Layer B — `fetch_by_rids` / `fetch_predicate`

| # | Scenario | Kind | Assert |
|---|---|---|---|
| B.1 | `fetch_by_rids(rid_column="RID")`, fresh engine | unit | all rows fetched and inserted |
| B.2 | Same call repeated within session | unit | no crash; engine final state correct |
| B.3 | Same call against pre-populated engine | unit | no crash; new RIDs inserted; existing untouched |
| B.4 | **`fetch_by_rids(rid_column=<FK>)`, multiple rows per FK value** | unit | **all matching rows fetched and inserted (A01)** |
| B.5 | `fetch_by_rids(rid_column=<FK>)`, partial population (some FK values have rows; server has more) | unit | all server rows fetched; engine = union |
| B.6 | `fetch_predicate` over populated engine | unit | no crash; new rows inserted; existing untouched |
| B.7 | Empty input list | unit | returns 0; no HTTP requests |
| B.8 | `fetch_by_rids` with `rid_column` not in target table | unit | clear error |

### Layer C — End-to-end through the full stack

| # | Scenario | Kind | Assert |
|---|---|---|---|
| C.1 | Live denormalize on a fresh workspace | live | correct row count |
| C.2 | Live denormalize twice in one session | live | both succeed; same result both times |
| C.3 | Live denormalize across sessions (two processes, same workspace dir) | live | both succeed; same result both times |
| C.4 | **Live denormalize with N feature rows per anchor (A01 reproduction)** | live | N × anchor-count rows; all N executions present |
| C.5 | Mutation → re-denormalize. **Insertion** server-side between calls. | live | second call observes the new row |
| C.5x | Mutation → re-denormalize. **Deletion or update** server-side between calls. | live, `xfail` | freshness limitation — see §6 |
| C.6 | `split_dataset` then live denormalize of the parent | live | parent's feature rows visible |
| C.7 | Live denormalize a Split parent containing children | live | members from children appear |

### Layer D — Cross-channel parity

| # | Scenario | Kind | Assert |
|---|---|---|---|
| D.1 | Same dataset, same include_tables/row_per, fetched via DatasetBag (source="local") AND Dataset (source="catalog") | live | identical row count and column set |
| D.2 | Same as D.1 but with multi-feature-per-anchor data (A01 shape) | live | identical row count |

Coverage status as of 2026-05-21:

- Layer A: covered by `tests/local_db/test_paged_fetcher.py` (A.1–A.7 in
  `TestFetchByRids`; A.8 is the new defensive case).
- Layer B: B.1, B.2, B.7 covered. **B.3, B.4, B.5, B.6, B.8 are the
  fix-pass test additions.**
- Layer C: C.1, C.2, C.6 covered by `tests/dataset/test_split.py` and
  `test_denormalize.py`. **C.3, C.4, C.5, C.7 are fix-pass additions.**
- Layer D: not currently covered; new additions.

The fix-pass for A01 must close out every case in Layers A and B,
plus C.4 and C.5 (positive insertion case). C.5x stays `xfail`.

---

## 9. Process commitments

Future engineering on this subsystem should respect:

1. **Reference this doc**, not the implementation plans. Plans are
   point-in-time. This doc is current. If a piece of behavior is
   not in this doc, it is not part of the contract.
2. **Update this doc** when changing any of: the `_insert_rows`
   semantics, the `PagedFetcher` lifetime/contract, the source-mode
   selection in `Denormalizer.__init__`, or the freshness model in
   §6.
3. **Add a test row** to §8 when fixing any denormalize bug. The
   minimum-failing repro becomes the regression test; the contract
   it exposed gets restated in §4–§6.
4. **The Rules (1–8) live in the semantics spec.** This doc does
   not duplicate them. If you change a Rule's semantics, update
   `docs/superpowers/specs/2026-04-17-denormalization-semantics-design.md`
   and link the change from here.
5. **Plans go to `docs/superpowers/plans/archive/` when superseded.**
   Don't delete them — they're historical context — but don't
   leave them in `plans/` where future readers will mistake them for
   current design.
