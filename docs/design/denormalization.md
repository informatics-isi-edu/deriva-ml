# Denormalization — Architecture & Contract

**Status:** Authoritative as of 2026-05-26.
**Scope:** The full denormalization pipeline — planner, source-mode
selection, row population, INSERT semantics, the cross-call/
cross-session caching behavior, and the public surface on the
`Denormalizer` class. This is the single design reference for
engineers working on `deriva_ml.local_db.denormalize`,
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
- `docs/audits/2026-05-26-denormalize-audit.md` — the 2026-05-26
  audit that surfaced the gaps closed by this revision. Findings
  SC-01 through SC-08, TC-01 through TC-10, and RB-01 through RB-10
  are referenced inline where they motivated a contract.

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
                         │    FK-graph walking delegated to
                         │      deriva.bag.path_walker.SchemaPathWalker
                         │      (shared with deriva-py's CatalogBagBuilder;
                         │       DerivaML layers skip-tables + nested-dataset
                         │       loopback as an edge_filter hook)
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

`_denormalize_impl` validates the `source` / `paged_client`
combination at entry: `source="catalog"` with `paged_client=None`
raises `ValueError` before any work is done.

### What "planner is pure; no I/O" actually means

`_prepare_wide_table` takes a `DatasetLike` and a `dataset_rid` as
parameters but **does not call any methods on them** in the current
implementation — anchor enumeration (`list_dataset_members`) and
nested-dataset enumeration (`list_dataset_children`) happen in the
`Denormalizer` layer above the planner (see §6). "Pure; no I/O"
therefore means:

1. **No HTTP**: the planner does not consult ERMrest. It walks the
   in-memory `DerivaModel` only (FK graph, table existence,
   sink/path enumeration).
2. **No engine reads**: the planner does not read from the local
   SQLite engine. It plans purely against schema.
3. **No dataset I/O**: today the planner accepts the `dataset`
   parameter past type-checking but does not invoke methods on it.
   If a future change adds dataset reads (e.g. to specialize the
   plan on dataset state), the planner becomes impure and this
   contract has to be revisited.

The takeaway: callers may safely invoke `_prepare_wide_table` from
a `describe`-style dry-run path on an offline catalog handle, and
the only failure modes are model-level (table not found, ambiguous
path, no sink). SC-08 in the 2026-05-26 audit named the
"no I/O" claim as ambiguous; this section is the tightened wording.

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
| `PagedFetcher._counts` (per-table row count memo) | in-memory | `PagedFetcher` instance | one fetcher lifetime (== one `_denormalize_impl` call) | only that fetcher's own cardinality heuristic |
| Local SQLite rows (`Image`, `Dataset`, `Execution_Image_*`, …) | sqlite-backed | engine | **persists to disk; survives the process** | every later call, every later session |

### The two truths to remember

1. **`PagedFetcher` is single-use.** A fresh one is constructed every
   time `_populate_from_catalog` is called. Any cache that lives on
   a `PagedFetcher` instance is empty at the start of each
   denormalize call. **In-memory state on a `PagedFetcher` cannot
   carry information across denormalize calls.**

   The fetcher does carry one piece of state **within** its
   lifetime: the `_counts` dict memoises per-table row counts so
   the cardinality heuristic (`fetch_by_rids_or_predicate`) can
   route without repeated `COUNT(*)` round-trips. The memo is
   populated lazily by the first call that needs the count for a
   given table, and it does *not* invalidate if the catalog mutates
   between two `fetch_*` calls inside the same denormalize. That
   window is rare-but-not-impossible against a write-active catalog,
   and acceptable because the heuristic only chooses between two
   correct strategies — a stale count produces suboptimal HTTP,
   never wrong rows. (SC-04 in the 2026-05-26 audit named the
   prior "memoised row counts only" phrasing as misleading; this
   section is the clarification.)

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
3. **Count.** Memoise per-table row counts within one fetcher
   lifetime so the cardinality heuristic (`fetch_by_rids_or_predicate`)
   doesn't re-issue `COUNT(*)` for repeated visits to the same table.

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

### Missing-RID behavior — engine-enforced

A row whose dict lacks a `"RID"` key (or has `"RID": None`) is a
**programming error** and must surface as an exception. The
implementation does NOT include an explicit `if "RID" not in row:
raise ValueError(...)` guard — instead it relies on the engine's
`NOT NULL` constraint on the `RID` primary key, which fires as
`sqlalchemy.exc.IntegrityError`. This contract is therefore
implemented by the *dialect*, not by the function. It works today
on every supported engine (SQLite, per `local_db/README.md`); a
future engine with different RID-column constraints (e.g.
server-generated RIDs) would silently break the contract. The
dialect coupling is acknowledged here as a known seam (SC-05 in
the 2026-05-26 audit) — if and when a non-SQLite engine is added,
this function gains an explicit pre-insert guard.

### Foreign-key enforcement is OFF during catalog fetch

`_populate_from_catalog` wraps the entire row-fetch loop in
`_foreign_keys_off` (`PRAGMA foreign_keys = OFF` on every connection
checkout). Reason: the join-path walk legitimately inserts a
referencing row before its referent — `Dataset_Image` rows arrive
before `Image` rows because we read `Dataset_Image.Image` to find
out which `Image` RIDs to fetch. Real integrity comes from the
source ERMrest catalog the rows arrived from; the local engine is a
**transport mirror**, not the authoritative store. FK enforcement
is restored on exit from the `_foreign_keys_off` context.

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
  1. Resolves source mode from the dataset type (see §6.1).
  2. Plans the join via `_prepare_wide_table` (planner is pure;
     no I/O — see §2). The FK-graph walker is
     `deriva.bag.path_walker.SchemaPathWalker`, shared with
     deriva-py's `CatalogBagBuilder`; DerivaML-specific rules (the
     default `Dataset_Dataset` / `Execution` skip set and the
     nested-dataset loopback guard) plug in via the walker's
     `edge_filter` hook so the generic walker stays domain-free.
  3. If `source="catalog"`: populates the local SQLite via
     `_populate_from_catalog`. This step issues catalog fetches and
     must satisfy the **row-completeness invariant** (§6.3): when
     the SQL JOIN runs in Step 4, the local cache must contain the
     union of rows every path's `(table, rid_column, rid_set)`
     tuple would fetch. A naïve dedup keyed on table name only
     violates the invariant when two element paths converge on the
     same table from different angles (see §7 row F5).
  4. Emits SQL against the local SQLite, runs it, materialises rows.
  5. Returns a `DenormalizeResult` (rows + column metadata).
- **Output row count.** Determined by the eight Rules in the
  semantics spec. The pipeline is *correct* iff that row count
  matches the server's reality, modulo the freshness caveat below.

### 6.1 Source-mode selection (`Denormalizer.__init__`)

`Denormalizer.__init__(dataset, *, version=None)` derives every
dependency from the `dataset` argument:

- **Live `Dataset`** (has `_ml_instance` pointing at a `DerivaML`):
  `model`, `catalog`, `engine`, and `orm_resolver` come from the
  ML instance's `workspace.local_schema`. `source` defaults to
  `"catalog"` because a `PagedClient` (`ErmrestPagedClient`)
  can be built against the catalog.
- **`DatasetBag` / canned test fixture** (has `model`, `engine`,
  `_orm_resolver` as direct attributes): `source` defaults to
  `"local"`. The engine is assumed pre-populated.

`version` (optional `DatasetVersion | str | None`): when given AND
the dataset is a live `Dataset`, the constructor resolves the
version to a catalog snapshot via the dataset's
`_version_snapshot_catalog` resolver, then builds the
`ErmrestPagedClient` against that snapshot. Member enumeration
(`_anchors_as_dict`) also threads `version` through to
`list_dataset_members` so the anchor set is read from the same
snapshot the SQL join will use. For `DatasetBag` and fixtures the
kwarg is silently ignored — the bag is already pinned to whatever
version it was built from.

**Failure-handling caveats** (silent fallbacks worth knowing):

- If `ErmrestPagedClient` construction raises (offline tests, mock
  catalog, auth not yet established), `__init__` falls back to
  `source="local"` and `paged_client=None` silently. A user
  re-running denormalize on a `Dataset` whose ML instance was
  constructed before auth completed will get whatever rows are in
  the engine — possibly zero — without a warning. (RB-05 in the
  2026-05-26 audit named this silent-zero hazard.) This is a known
  robustness gap; the spec acknowledges it but does not require a
  fix in this revision.
- If snapshot resolution fails (bad version string, history lookup
  error), the constructor **re-raises** — version-pinned
  construction is explicit enough that a silent fallback would be
  worse than the failure.

### 6.2 Anchor classification, nested datasets, and the SQL filter

The Denormalizer's `_run` method partitions the anchor set
(`_anchors_as_dict`, derived from
`dataset.list_dataset_members(recurse=True, version=...)`) into
three buckets via `_classify_anchors`:

- **Scoping** — anchors whose RIDs filter the row_per side. Cases
  1, 2, 4 in semantics spec §3.7 (anchor == row_per, in
  include_tables and reaches row_per, not in include_tables but
  reaches row_per).
- **Orphan** — case 3: anchor is in include_tables but has no FK
  path to row_per. Emits LEFT-JOIN-shaped rows via
  `_emit_orphan_rows`.
- **Ignored** — case 5 (anchor connected to subgraph but doesn't
  reach row_per; silent drop) and case 6 with the
  `ignore_unrelated_anchors` flag set (no FK path at all; would
  otherwise raise `DerivaMLDenormalizeUnrelatedAnchor`).

Empty anchor sets are skipped entirely (e.g. `{"File": []}`
returned by `list_dataset_members` for an association table whose
row count is zero never triggers Rule 8).

After the main SQL join runs, `_run` performs a **per-RID orphan
scan** (Step 4a) to catch upstream scoping anchors whose specific
RIDs didn't appear in the main result — a Subject whose Image set
is empty, for example. Those RIDs are added to the orphan set and
emitted via `_emit_orphan_rows`, producing LEFT-JOIN-shaped output
rows with the row_per-side columns set to `None`.

`_run` also pulls **descendant dataset RIDs** via
`dataset.list_dataset_children(recurse=True)` when the dataset
supports it. These RIDs are passed to `_denormalize_impl` as
`dataset_children_rids` and end up in the SQL `WHERE Dataset.RID IN
(dataset_rid, ...children)` clause. Without them, members of nested
datasets (whose `Dataset_X.Dataset` points at a descendant rather
than the root) never pass the filter and the result comes back
empty. If `list_dataset_children` is not implemented or raises, the
helper falls back to root-only scoping silently — fixture-shaped
datasets often don't implement it. (RB-06 in the 2026-05-26 audit
flags this silent-fallback against transient network errors as a
known robustness gap.)

### 6.3 The row-completeness invariant (§6 step 3 in detail)

The fetch step must leave the local cache in a state where the SQL
JOIN in Step 4 returns every row a *fully-fetched* execution
against the server would return. Concretely:

> For every distinct `(table, rid_column, rid_set)` tuple that the
> planner emits across the join paths, the local cache must
> contain the rows the server would return for that tuple.

This is the **row-completeness invariant**. It implies — but is
strictly stronger than — "every table appears in the cache." Two
paths can target the same table with different `rid_column`s or
different RID sets; the cache must hold the **union** of all rows
those parametrizations would fetch, not just one path's worth.
"Must always re-issue the fetch for every `(table, rid_column,
rid)` tuple in the plan" (the language in earlier revisions of
this doc) is the consequence, not the contract — the contract is
the row-completeness of the cache; how the implementation
achieves it is an implementation detail subject to optimisation.

The current implementation enforces the invariant by deduplicating
fetches on the full tuple `(table_name, rid_column_on_target,
frozenset(rids_to_fetch))`. Only **true** duplicates (identical
table, identical rid column, identical rid set) are skipped; every
distinct parametrization fires its own catalog fetch. Until
2026-05-26 the dedup key was the table name only, which only
*coincidentally* satisfied the invariant under the planner's
current output. See §7 row F5 for the full incident write-up
(rewritten by the in-flight `fix/sc06-row-completeness-invariant`
branch — see the note below F6).

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

Documented; tracked as a known limitation. The deletion/update
regression test promised in §9 (row C.5x, marked `live, xfail`)
is **planned but not yet written** — searching the test suite as
of 2026-05-26 finds no matching `xfail` (TC-04 in the audit). This
spec keeps the row as a placeholder so the next contributor to the
freshness story knows where to put the regression; the test does
not exist today.

There is also no caller-visible signal that a given result was
affected by a stale-cache window — no last-fetch timestamp, no
warning on `DenormalizeResult`, no `reason` field. A user
re-running denormalize in a long-lived process therefore trusts a
row count they have no way to invalidate. (SC-03 in the 2026-05-26
audit named this gap; surfacing it is a future contract change.)

---

## 7. Fragility map — known bug patterns and how the contract prevents them

| # | Pattern | Cause | Status |
|---|---|---|---|
| **F1** | Re-denormalize crashes with `UNIQUE constraint failed: Dataset.RID` | Plain `INSERT` against engine that has prior rows. | Fixed by the INSERT-OR-IGNORE contract in §5. Originally surfaced as 2026-05-21 finding 05 in the model-template e2e run. |
| **F2** | Re-denormalize silently drops rows | `_get_seen` (v1.37.2) hydrated a dedup map from the engine keyed by the caller's `rid_column`. For FK columns with N rows per value, this collapsed the fetch to one row per FK. | Fixed by removing the engine-hydrated seen-set entirely. The fetcher does not dedup based on engine state; conflict handling belongs at INSERT. Originally surfaced as 2026-05-21 finding A01. |
| F3 | Stale local data when server mutates between calls (deletes, updates) | Cache is write-through, never invalidated. | Documented as a known limitation. See §6 freshness caveat. The promised `xfail` regression test (row C.5x in §9) is planned but unwritten as of 2026-05-26 (TC-04 in the audit). |
| F4 | `_collect_fk_values` walks "values currently present in the engine" to decide what to fetch from the server. If a parent table's membership was updated server-side after the engine cached it, downstream fetches use the stale parent set. | Same root cause as F3. | Same status — known limitation, tracked. |
| **F5** | Path-walk order silently determines which rows get loaded when two element tables share intermediate tables | The contract (§6 step 3) requires the local cache to contain the union of rows every path's `(table, rid_column, rids)` tuple would fetch — the *row-completeness invariant*. Until 2026-05-26 `_populate_from_catalog_inner` keyed its `processed` set on table name only, which only *coincidentally* satisfied the invariant under the current planner's output. A future planner change (new element type, FK refactor, split datasets) could silently break the invariant without any code in `_populate_from_catalog_inner` changing. | **Fixed** (2026-05-26): the dedup key is now `(table_name, rid_column, frozenset(rids))`, which implements the invariant directly — each distinct parametrization fires its own fetch; only true duplicates are deduped. Regression test pinned at §8 row D.3 / C.8. Originally surfaced as 2026-05-26 audit finding SC-06 / RB-02. |
| **F6** | `describe()` / `preflight_count` reports `estimated_row_count.total = 0` while the actual fetch returns rows | The estimator counted anchors whose table literally equals `row_per`. When `row_per` is downstream of the anchor table (the common feature-table case), no anchor matches and the sum is silently 0. Mathematically the cardinality is N rows per anchor for an unknown-from-anchor-data N. | Fixed by honest "unknown" semantics: when anchors are downstream of `row_per`, `in_scope_row_per_rows` and `total` return `None` and a `reason` field tells the caller why. The case-1 path (anchor == row_per) still returns an exact integer. Originally surfaced as 2026-05-21 finding A02 (Analyst arc). |

> **Note on F5:** the row above is the wording as it appears on
> `main`. A concurrent fix-pass (`fix/sc06-row-completeness-invariant`)
> rewrites the row to distinguish the row-completeness invariant
> (§6.3) from the coincidence that today's planner output happens
> to satisfy it, and adds the regression test rows referenced in
> the row-completeness invariant section. This revision intentionally
> leaves F5 alone to avoid a merge conflict with that PR; once
> SC-06 lands, the F5 wording will reflect the explicit invariant
> in §6.3.

---

## 8. The public Denormalizer surface

`Denormalizer` (in `src/deriva_ml/local_db/denormalizer.py`) is the
contracted public class. Every method below has a stable input/output
shape; the docstrings on the implementation carry the parameter-level
detail. This section pins the **behavioral contracts** — what is
guaranteed, what can fail, and how the methods relate to each other.

### 8.1 Constructors

#### `Denormalizer(dataset, *, version=None)`

Construct from a `DatasetLike`. Source mode and dependencies are
derived from the dataset's shape (§6.1). `version` snapshots the
catalog for live `Dataset` inputs and is ignored for `DatasetBag`.

#### `Denormalizer.from_rids(anchors, *, ml=None, ...)`

Construct from an explicit RID anchor set without a `Dataset`
context. Anchors may be bare RIDs (table looked up via catalog) or
`(table_name, RID)` tuples. Mixed forms supported. Bare-RID lookup
prefers `ml.resolve_rids` (batched, O(tables) round-trips) over
`catalog.resolve_rid` (per-RID, O(N) round-trips).

**Known limitation — placeholder `dataset_rid`.** The current
`_denormalize_impl` scopes its SQL via `Dataset.RID IN (dataset_rid,
...)`. `from_rids` therefore needs a real dataset RID. When the
caller omits `dataset_rid`, the constructor falls back to using the
first anchor's RID as a pseudo-scope, which **silently returns zero
rows against a production catalog** because no real dataset has
that RID. The fallback exists for fixture-shaped flows where the
anchors are themselves dataset roots, and is harmless there.

This is a discoverability hazard — the user sees an empty DataFrame
with no breadcrumb (SC-02 in the 2026-05-26 audit). The spec
records the limitation here; a future contract change should either
reject `dataset_rid=None` with a `ValueError` when the source is a
live catalog, or surface a `reason` field on the resulting empty
DataFrame in the same shape `describe.estimated_row_count` uses for
A02. Either is acceptable; both close the silent-zero failure
mode. The choice belongs to the change that implements it.

`from_rids` always sets `source="local"` and `paged_client=None`.
Callers who want catalog-side fetching must pre-populate the engine
or pass an explicit `Denormalizer(dataset)` constructed from a real
`Dataset`. (This is a scope-shrink relative to the original
2026-04-17 semantics spec, which described `from_rids` as supporting
both source modes; the implementation took the simpler path.)

**Raises:** `ValueError` for missing `model`/`ml`, unresolvable
bare RIDs, missing catalog for lookup, or malformed `(table, RID)`
tuples (arity ≠ 2).

### 8.2 Materialization methods

#### `as_dataframe(include_tables, *, row_per=None, via=None, ignore_unrelated_anchors=False) -> pandas.DataFrame`

Run the full 4-phase pipeline (planner decisions → anchor
classification → main SQL join → orphan-row combine) and return a
DataFrame. One row per `row_per` instance in scope, plus any orphan
rows whose `row_per`-side columns are `NaN`.

**Raises:**
`DerivaMLDenormalizeMultiLeaf`, `DerivaMLDenormalizeNoSink`,
`DerivaMLDenormalizeDownstreamLeaf`, `DerivaMLDenormalizeAmbiguousPath`,
`DerivaMLDenormalizeUnrelatedAnchor` — the semantic-rule
exceptions. Also `DerivaMLTableNotFound` and
`DerivaMLDenormalizeError` from feature-name resolution (§8.4).
See the semantics spec for the per-rule details.

#### `as_dict(include_tables, *, row_per=None, via=None, ignore_unrelated_anchors=False) -> Generator[dict, None, None]`

Same planner, same rules, same exceptions as `as_dataframe`. Yields
one `dict[str, Any]` per row keyed by `Table.column` /
`schema.Table.column` labels.

**Materialization, not streaming — known doc-rot.** Despite the
"memory-efficient for large results" wording in the current
docstring, the implementation builds the full row list in
`_denormalize_impl` before any row is yielded (rows are eagerly
materialised by `session.execute(final_query)` into a Python list).
`as_dict` then yields from that list via
`DenormalizeResult.iter_rows`. There is no streaming path today.
The "yields one at a time" interface is preserved (the caller's
loop iterates rows one by one) but peak memory equals the full
result set.

This is a contract / docstring divergence (SC-07 in the
2026-05-26 audit). The resolution will land in a separate change:

- **Option A** — implement true streaming. Replace
  `_denormalize_impl`'s materialise step with a generator that
  yields inside the `session.execute` loop. `DenormalizeResult`
  becomes iteration-aware (`row_count` becomes lazy or unknown).
- **Option B** — correct the docstring. Drop the
  "memory-efficient" framing and describe `as_dict` as a row-by-row
  iterator over a materialised result.

This spec does not pick between them; either is a valid resolution
and the choice belongs to the change that implements it. Until
then, treat `as_dict` as "iteration interface, materialised
internals."

**Raises:** same as `as_dataframe`. All planner validation runs
before any row is yielded (the pipeline materialises up front), so
exceptions surface on the first `next()`.

### 8.3 Inspection methods

#### `columns(include_tables, *, row_per=None, via=None) -> list[tuple[str, str]]`

Preview the `(column_name, column_type)` pairs the matching
`as_dataframe` call would produce. Planner-only — no data fetch,
no catalog query, no anchor classification. Useful as a cheap
validator of `include_tables` before committing to a full run.

**Raises:** same planner-rule exceptions as `as_dataframe` (Rules
2/5/6) plus the resolver exceptions from §8.4. Rule 7 and Rule 8
do NOT fire here — anchor classification runs only when rows are
materialised.

#### `describe(include_tables, *, row_per=None, via=None) -> dict[str, Any]`

**The dry-run inspection method.** Returns a 12-key dict describing
what a corresponding `as_dataframe` call would do, **without
raising**. Every failure mode (planner-rule violation, catalog
access error, network timeout, ambiguous resolution) is **swallowed**
and represented in the returned dict as `None` / `[]` / `{}` in the
affected positions. Ambiguities are reported in the `ambiguities`
list so the caller can inspect before committing to a real call.

##### 8.3.1 The dry-run invariant

`describe()` **never raises**. This is a contract, not an
implementation detail — callers may safely wrap a `describe()`
call in code that does not handle exceptions, knowing that any
failure inside the planner / catalog / schema stack collapses to
a well-formed dict with sensible defaults.

The invariant is enforced by wrapping every internal call (planner
hooks, anchor enumeration, ambiguity finder, column-spec builder,
classifier) in a broad `try/except`. The cost of this invariant is
**diagnostic loss** — a user who sees `row_per=None` and
`columns=[]` knows the plan is empty but not *why*. Today there
is no `warnings` / `reason` field on `describe()`'s output to
carry the swallowed-exception message; surfacing it is a future
contract change (R-2 in the 2026-05-26 audit, classified
`unimplemented`). SC-01 and RB-01 name the silent-failure shape;
TC-09 names the test gap.

##### 8.3.2 The 12-key return shape

The returned dict has these 12 keys. Keys are present in **every**
call (never omitted), even if their value is the empty default.

| Key | Type | Meaning when populated | Meaning when empty |
|---|---|---|---|
| `row_per` | `str \| None` | Resolved leaf table name. | `None` — planner couldn't resolve (multi-leaf, no sink, bad explicit value). |
| `row_per_source` | `str` | `"explicit"` if the caller passed `row_per`, else `"auto-inferred"`. | Always present. |
| `row_per_candidates` | `list[str]` | Sink tables from Rule 2 sink-finding (what auto-inference considered). | `[]` — sink-finding raised. |
| `columns` | `list[tuple[str, str]]` | `(name, type)` pairs `as_dataframe` would produce. | `[]` — planner raised before columns could be computed. |
| `include_tables` | `list[str]` | Echo of the caller's input (post feature-name resolution; see §8.4). On resolver failure, falls back to the original input. | Always populated. |
| `via` | `list[str]` | Echo of the caller's input. | `[]` if not supplied. |
| `join_path` | `list[str]` | Ordered table names on the join chain (excludes `Dataset` root). | `[]` — no element tables resolved. |
| `transparent_intermediates` | `list[str]` | Subset of `join_path` not named in `include_tables` (joined through but not projected). | `[]` if no transparent tables. |
| `ambiguities` | `list[dict]` | Per-Rule-6 entries: `{type, from, to, paths, suggestions}`. | `[]` if plan is unambiguous OR if ambiguity detection raised. |
| `estimated_row_count` | `dict` | `{in_scope_row_per_rows, orphan_rows, total[, reason]}`. See §8.3.3. | All three counts `None` if classification raised. |
| `anchors` | `dict` | `{total, by_type}` — RID counts grouped by table. | `{total: 0, by_type: {}}` if anchor enumeration raised. |
| `source` | `str` | `"catalog"` for live Datasets, `"local"` for bags/fixtures, `"slice"` for attached slices. | Always populated. |

##### 8.3.3 `estimated_row_count` semantics (the A02 fix)

Three cases drive the row-count estimate:

1. **Anchor table == row_per** → exact: 1 row per anchor.
   `in_scope_row_per_rows = anchor_count`,
   `total = in_scope_row_per_rows + orphan_rows`. No `reason` key.
2. **Anchor table reaches row_per via FK chain (downstream or
   upstream) but is NOT row_per itself** → unknown without a
   catalog query. `in_scope_row_per_rows = None`,
   `total = None`, and a `reason` string names the downstream
   anchor tables so the caller knows why. `orphan_rows` is still
   exact (orphans contribute exactly one row regardless of
   row_per cardinality).
3. **Anchor table has no FK path to row_per** → orphan (Rule 7
   case 3). Contributes to `orphan_rows` (exact), not to
   `in_scope_row_per_rows`.

Mixed case-1 and case-2 anchors collapse to the case-2 behavior:
`None` with a `reason`. This is the honest answer — we can't sum
a known integer with an unknown integer.

The pre-A02 implementation only honored case 1 and silently
returned 0 for case 2, which produced false "0 rows" estimates
for every feature-table denormalize (the common case). See §7
row F6 for the incident.

##### 8.3.4 The describe-vs-run agreement contract

`describe(include_tables=X)` and `as_dataframe(include_tables=X)`
must agree on whether `X` is a valid input. If `describe` accepts
a name and returns a plan, the matching `as_dataframe` call must
either succeed with a result of the shape `describe` predicted or
raise on a semantic-rule violation `describe` reported in
`ambiguities` — never raise on an input-validation failure
`describe` silently accepted.

The validation is shared via the `_resolve_table_names` helper
(introduced by PR #228, the analyst/01 fix; described in §8.4).
Both paths now recognize **feature names** (e.g.
`"Image_Classification"`) as shorthand for the underlying
feature-association table (e.g.
`"Execution_Image_Image_Classification"`).

The current test suite pins this contract at one place — the
`{name for name, _ in plan["columns"]} == set(df.columns)` check
in `TestFeatureNameResolution::test_describe_and_run_agree`.
**Every other key in the 12-key envelope has the same
analyst/01-shaped asymmetry risk and no parity test today**
(TC-02 in the audit). Future work: a parity test per key.

#### `list_paths(tables=None) -> dict[str, Any]`

Describe the FK graph reachable from the anchor set. Model-only
analysis: no catalog query, no data fetch. Useful for schema
exploration — answers "what tables could I reasonably include in
`include_tables` given my anchor set?"

Returns a 6-key dict:

- `member_types`: dataset element types (same as `anchor_types`
  for `from_rids`-constructed Denormalizers).
- `anchor_types`: sorted list of distinct anchor table names.
- `reachable_tables`: `{anchor_table: [reachable downstream
  tables, sorted]}`.
- `association_tables`: pure M:N association tables in the schema.
- `feature_tables`: feature tables discovered via
  `DerivaModel.find_features`. Empty if the model doesn't expose
  `find_features` or has no features.
- `schema_paths`: `{(source, target): [{path, direct}]}` — FK
  paths between reachable pairs, with a `direct` flag for
  single-hop paths.

Failures inside the model walk (e.g. catalog unreachable when
inspecting `model.schemas`) collapse to empty defaults in the
affected keys — same dry-run posture as `describe`.

### 8.4 Feature-name resolution (`_resolve_table_names`)

`describe`, `as_dataframe`, `as_dict`, and `columns` all share the
same `include_tables` / `via` / `row_per` validation via the
private `_resolve_table_names` helper (introduced by PR #228 —
this section assumes PR #228 has landed; if it hasn't, the
contract is aspirational for the affected methods). The helper
translates **feature names** (e.g. `"Image_Classification"`) into
the underlying **feature-association table name** (e.g.
`"Execution_Image_Image_Classification"`) so callers can pass
either form symmetrically with the rest of the DerivaML API
(`find_features`, `feature_values`, `lookup_feature`).

Resolution algorithm for each input name:

1. If `model.name_to_table(t)` succeeds → keep `t` as-is.
2. Otherwise consult `model.find_features()`. If exactly one
   feature has `feature_name == t` → substitute with
   `feature.feature_table.name`. If multiple matches across
   different target tables → raise `DerivaMLDenormalizeError`
   (ambiguous — caller must name the feature-association table
   directly).
3. If neither path works → raise `DerivaMLTableNotFound`. The
   error message includes the list of known feature names so the
   user can tell whether they typo'd a feature or pointed at the
   wrong table.

On the **dry-run path** (`describe`), the entire resolver call is
wrapped in `try/except Exception` to preserve the dry-run
invariant (§8.3.1). On resolver failure, the original input is
passed through and the downstream planner calls produce their own
empty fields. This preserves the contract — `describe` never
raises — at the cost of diagnostic loss (the caller doesn't know
whether `row_per=None` came from a resolver failure or a
planner-rule violation). SC-01 in the 2026-05-26 audit names this
gap; the fix is the future `warnings` field on describe's output.

On the **run path** (`as_dataframe` / `as_dict` / `columns`), the
resolver raises immediately on ambiguous or unknown names, so the
user gets a clear error before any planner work fires.

The contract: **if `describe(X)` accepts X, the matching run path
accepts X too.** (§8.3.4.)

---

## 9. Test matrix

The matrix below pins the contracts in §4–§8 against concrete
pytest cases. Each row is one test. Cases against a live catalog
are marked `live`; cases against an in-memory engine +
`FakePagedClient` are marked `unit`.

### Layer A — `_insert_rows` (the heart of the new contract)

| # | Scenario | Kind | Assert |
|---|---|---|---|
| A.1 | Insert N rows into empty target | unit | engine has N rows; returned count == N |
| A.2 | Insert overlapping rows twice into one target | unit | engine has N rows; 2nd call returns 0 |
| A.3 | Insert rows where some RIDs already exist | unit | engine = pre ∪ new; returned count == \|new only\| |
| A.4 | Insert rows missing RID — pin behavior | unit | raise `IntegrityError` (engine NOT NULL constraint; see §5 missing-RID note) |
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
| C.5x | Mutation → re-denormalize. **Deletion or update** server-side between calls. | live, `xfail` — **planned, not yet written (TC-04)** | freshness limitation — see §6 |
| C.6 | `split_dataset` then live denormalize of the parent | live | parent's feature rows visible |
| C.7 | Live denormalize a Split parent containing children | live | members from children appear |
| **C.8** | **Two element paths converge on the same target table with disjoint rid sets (F5 / SC-06)** | unit | **every Image RID reachable via either path is fetched; local cache equals the union (covered by `tests/local_db/test_denormalize_impl.py::TestRowCompletenessInvariant`)** |

### Layer D — Cross-channel parity

| # | Scenario | Kind | Assert |
|---|---|---|---|
| D.1 | Same dataset, same include_tables/row_per, fetched via DatasetBag (source="local") AND Dataset (source="catalog") | live | identical row count and column set |
| D.2 | Same as D.1 but with multi-feature-per-anchor data (A01 shape) | live | identical row count |
| **D.3** | **Same as D.1 but with two element paths converging on one table (F5 / SC-06 shape)** | live | **identical row count and column set; both sources see every reachable row** |

### Layer E — `describe()` / preflight estimated_row_count

| # | Scenario | Kind | Assert |
|---|---|---|---|
| E.1 | Anchor table == `row_per` (case 1) | live | exact integer estimate == anchor count |
| E.2 | **Anchor table is downstream of `row_per` (case 2 — feature-table common case)** | live | **`in_scope_row_per_rows` and `total` are `None`; a `reason` field is present (A02 regression)** |
| E.3 | Mixed scoping anchors (some at `row_per`, some downstream) | live | honest `None` with reason (the case-2 contribution can't be added to the case-1 count) |
| E.4 | All anchors orphan (no FK path) | live | `orphan_rows` is exact, `in_scope_row_per_rows == 0`, `total == orphan_rows` |
| E.5 | `describe(<feature_name>)` and `as_dataframe(<feature_name>)` agree on `columns` (analyst/01 regression) | live | `{name for name, _ in plan["columns"]} == set(df.columns)` |
| E.6 | `describe(<unknown_name>)` never raises (dry-run invariant) | unit | returns a 12-key dict with empty defaults; no exception |

### Layer F — `from_rids` constructor

| # | Scenario | Kind | Assert |
|---|---|---|---|
| F.1 | `from_rids` with `(table, RID)` tuples and explicit `dataset_rid` | unit | correct anchor partitioning; row count matches |
| F.2 | `from_rids` with bare RIDs (catalog lookup) | unit | resolves to (table, RID) via `ml.resolve_rids` or `catalog.resolve_rid` |
| F.3 | `from_rids` rejects malformed inputs (no model, no catalog for bare RIDs, bad tuple arity) | unit | `ValueError` with clear message |
| F.4 | `from_rids` without `dataset_rid` against a real-shaped catalog (TC-05) | live, **planned** | placeholder behavior — see §8.1 known limitation; should either reject or surface a `reason`. No test pins this today. |

### Coverage status as of 2026-05-26

- **Layer A**: covered by `tests/local_db/test_paged_fetcher.py`
  (A.1–A.7 in `TestFetchByRids`; A.8 is the new defensive case).
- **Layer B**: B.1, B.2, B.7 covered. B.3, B.4, B.5, B.6, B.8 are
  fix-pass additions for A01.
- **Layer C**: C.1, C.2, C.6 covered by `tests/dataset/test_split.py`
  and `test_denormalize.py`. C.3, C.4, C.5, C.7 are fix-pass
  additions. **C.5x is planned but not yet written** (TC-04).
- **Layer D**: D.1 is partially covered by
  `test_catalog_and_bag_denormalize_consistency` (single-table
  trivial case only); **D.2 is not yet written** (TC-01, TC-10).
  A row exercising the two-element-paths-converge-on-one-table
  scenario is planned by the in-flight
  `fix/sc06-row-completeness-invariant` branch.
- **Layer E**: E.1–E.4 covered by the A02 fix-pass tests in
  `test_denormalizer.py::TestDescribe`. E.5 covered by
  `TestFeatureNameResolution::test_describe_and_run_agree` (one
  key, one input; TC-02 names the gap for the other 11 keys).
  E.6 covered by `test_describe_never_raises_on_*`.
- **Layer F**: F.1–F.3 covered by `TestFromRids` in
  `test_denormalizer.py`. **F.4 is not yet written** (TC-05).

The fix-pass for A01 closed every case in Layers A and B, plus
C.4 and C.5 (positive insertion case). Several rows above are
explicitly marked "planned, not yet written" — those are the
gaps the 2026-05-26 audit named (TC-01, TC-02, TC-04, TC-05,
TC-10). This spec keeps them in the matrix as placeholders so a
future contributor knows where the regression test belongs.

---

## 10. Process commitments

Future engineering on this subsystem should respect:

1. **Reference this doc**, not the implementation plans. Plans are
   point-in-time. This doc is current. If a piece of behavior is
   not in this doc, it is not part of the contract.
2. **Update this doc** when changing any of: the `_insert_rows`
   semantics, the `PagedFetcher` lifetime/contract, the source-mode
   selection in `Denormalizer.__init__`, the freshness model in §6,
   the `describe()` envelope shape in §8.3, or the
   `_resolve_table_names` algorithm in §8.4.
3. **Add a test row** to §9 when fixing any denormalize bug. The
   minimum-failing repro becomes the regression test; the contract
   it exposed gets restated in §4–§8. If the test is not yet
   written, add the row anyway and mark it "planned, not yet
   written" so the gap is visible.
4. **The Rules (1–8) live in the semantics spec.** This doc does
   not duplicate them. If you change a Rule's semantics, update
   `docs/superpowers/specs/2026-04-17-denormalization-semantics-design.md`
   and link the change from here.
5. **Plans go to `docs/superpowers/plans/archive/` when superseded.**
   Don't delete them — they're historical context — but don't
   leave them in `plans/` where future readers will mistake them
   for current design.
6. **Silent zero is a bug class, not an acceptable failure mode.**
   The audit's cross-cutting observation §5.1 names a half-dozen
   sites where a code path returns an empty result with no
   diagnostic — `from_rids`'s placeholder `dataset_rid`,
   `describe`'s broad-except resolver, `__init__`'s fallback to
   `source="local"` on `ErmrestPagedClient` construction failure,
   `_run`'s broad-except around `list_dataset_children`. New code
   in this subsystem must either raise on its silent-zero paths or
   surface a `reason` / `warnings` field. The honest-`None`-plus-
   `reason` pattern `describe.estimated_row_count` already uses
   (§8.3.3) is the canonical template.
