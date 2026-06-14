# Fast `estimate_bag_size` / `bag_info` — redundant `/schema` introspection — Design

**Date:** 2026-06-13
**Status:** Approved in brainstorming; spec for implementation.
**Subproject:** `deriva-ml`

## 1. Problem statement

`estimate_bag_size` (and `bag_info`, which calls it) takes **~8.5 minutes**
on the `www.eye-ai.org` `2-277G` dataset (84 nested descendants, 80 tables,
360 k rows, 18 GB). That is unusable as a "how big before I download?"
preview.

Profiling (live, 2026-06-13) shows the cost is **not** the size queries —
it is **redundant catalog-schema introspection**:

- The walk issues **2,375 serial HTTP GETs**; **391 s of 412 s (95 %) is
  `_ssl._SSLSocket.read`** — pure network wait, ~0 CPU.
- **849 of those GETs are `/schema`** (full catalog-model introspection),
  re-fetched over and over.
- The actual estimate queries (per-table counts + asset `Length`) are
  **~2 %** of the traffic — a few dozen GETs at the bottom of the profile.

### Root cause (deriva-ml, not deriva-py)

`Dataset.list_dataset_members()` and `Dataset.list_dataset_children()` each
call:

```python
version_snapshot_catalog = self._version_snapshot_catalog(version)
pb = version_snapshot_catalog.pathBuilder()
```

`_version_snapshot_catalog(version)` (`dataset.py:2844`) calls
`self._ml_instance.catalog_snapshot(...)` which **builds a fresh,
uncached snapshot-catalog object on every call**. `pathBuilder()` on that
fresh object has an empty cache, so it **re-introspects `/schema`** each
time. (When `version` is `None` it returns the shared `_ml_instance`,
whose pathbuilder *is* cached — so only the versioned path, which the
estimate uses, pays this.)

This fires **once per descendant dataset** during the nesting walk, and the
nesting walk runs **~2.5×** per estimate:

1. `aggregate_queries` → `anchors_for(ds)` → `_iter_descendant_rids` (walk 1:
   `list_dataset_children` per node).
2. `aggregate_queries` → `build_policy(ds)` → `_exclude_empty_associations`
   → `_iter_descendant_rids` **again** (walk 2) **plus** a
   `lookup_dataset(rid)` + `list_dataset_members()` for **every** RID in the
   tree (walk 3, the member scan).

Measured attribution (2-277G):

| Phase | wall | GETs | `/schema` |
|---|---|---|---|
| `anchors_for` alone | 35 s | 676 | 338 |
| `aggregate_queries` total | 415 s | 2 375 | 849 |

So ~98 % of the time is redundant `/schema` fetches + repeated tree walks;
the genuine estimate work is the remaining ~2 %.

**Note (revises an earlier call):** during investigation we tentatively
concluded the fix belonged upstream in deriva-py's `CatalogBagBuilder`.
The profile disproves that — `CatalogBagBuilder._get_model()` and
`getPathBuilder()` are correctly cached per instance, and the FK walk
(`SchemaPathWalker.walk_bfs`) is pure in-memory. The redundant `/schema`
fetches originate in **deriva-ml**'s snapshot-catalog re-creation and
double tree-walk. The fix is deriva-ml-side.

## 2. Goals

- `estimate_bag_size(2-277G)` and `bag_info(2-277G)` drop from ~8.5 min to
  **seconds** (target: < 30 s; the irreducible work is ~the bottom-of-profile
  estimate queries plus one `/schema` and one tree walk). **Requires both**
  Lever A (snapshot construction, §3.1) **and** Lever C (live-instance
  pathBuilder caching, §3.1b) — A1/A2 alone leaves an O(N) live-instance
  `/schema` refetch (discovered during implementation; servers don't honor
  304).
- **No change to what the estimate computes** — exact row counts and exact
  asset bytes, identical numbers, byte-for-byte. (User decision: "keep exact,
  just make it fast.")
- No public-API change to `estimate_bag_size` / `bag_info` signatures.

## 3. The two redundancies, and the fix for each

### 3.1 Redundant `/schema` from snapshot-catalog construction (the 98 %)

The deeper finding: `_version_snapshot_catalog(version)` calls
`self._ml_instance.catalog_snapshot(...)`, and `catalog_snapshot()`
(`core/base.py:766`) **constructs a brand-new `DerivaML(...)` instance**.
`DerivaML.__init__` → `_init_online` unconditionally calls
`self.catalog.getCatalogSchema()` (`base.py:465`) — a full `/schema` fetch —
*even though `_init_online` already writes that schema to a disk
`SchemaCache` it never reads back* (`base.py:435`). So every snapshot
construction re-fetches the whole catalog model from the server.

**The schema does not need to be fetched at all.** A snapshot pins the
*data* view at a snaptime; the **model structure is identical to the live
catalog** the `_ml_instance` already holds in memory (`self.model`, parsed
once at construction via `DerivaModel.from_cached`). So the snapshot
instance can be seeded from the live instance's already-parsed schema and
skip the network entirely.

**Scope of the redundancy (full audit, 2026-06-13).** A repo-wide audit of
every `/schema`-triggering call site (`getCatalogSchema`, `getCatalogModel`/
`Model.fromcatalog`, `getPathBuilder`/`pathBuilder`, and every new
catalog-object construction) found that **`catalog_snapshot()` is the *only*
source of redundant schema fetching** in deriva-ml. Every other site is
either:

- **HARMLESS** — repeated `getCatalogSchema()` / `pathBuilder()` /
  `getCatalogModel()` on the *same* long-lived `_ml_instance.catalog`
  (deriva-py caches the parsed `/schema` per catalog object, so the 77
  `pathBuilder()` call sites share **one** fetch); or
- **NECESSARY** — a genuine first-ever fetch on a freshly-created catalog
  (initial `DerivaML` construction; `clone_via_bag` / `localize` connecting
  to *different* source/dest catalogs), or a **deliberate** refresh that
  purges the cache on purpose (`refresh_schema`, `pin_schema`, `diff_schema`
  each call `purge_cache_by_prefix("/schema")` before re-fetching — that is
  their job).

So fixing `catalog_snapshot()` cures the snapshot-construction class. The one
offender is amplified two ways: **(i)** the nesting walk calls it once per
descendant dataset (85× for 2-277G), and **(ii)** even within a *single*
operation, **8 distinct `Dataset` call sites** (`list_dataset_members`,
`list_dataset_children`, denormalization, etc.) each build their own snapshot
catalog and none reuse it.

### 3.1a CORRECTION (2026-06-13, found during implementation): the audit's "HARMLESS" claim is false on non-304 servers

The audit's "HARMLESS" bucket above assumed deriva-py caches the parsed
`/schema` per catalog object, so repeated `pathBuilder()` /
`getCatalogSchema()` on the same live `_ml_instance.catalog` cost **one**
fetch. **That assumption is wrong for the servers deriva-ml actually talks
to.** deriva-py's cache is *conditional-revalidation*-based:
`getCatalogSchema()` calls `self.get('/schema')` every time and only returns
the memoized parsed dict when the binding reports a **304 Not Modified**
(`cached[0] is r` — same `Response` object). `getPathBuilder()` likewise
re-derives from `getCatalogSchema()` and only hits its wrapper cache when the
returned schema dict is the *same object* — which again only happens on a
304.

**Measured 2026-06-13:** both the localhost Deriva Docker stack **and
production `www.eye-ai.org`** return **HTTP 200 (not 304)** on a conditional
`/schema` GET (`If-None-Match` is sent but ignored). Therefore deriva-py's
schema/pathbuilder cache **never engages** on these servers — every
`pathBuilder()` call re-fetches and re-parses the full `/schema`.

Consequence: there is a **second, independent O(N) source** the original
audit missed. During an estimate, `bag_builder._iter_descendant_rids` calls
`self._ml_instance.lookup_dataset(rid)` for **every** descendant, and
`lookup_dataset` → `pathBuilder()` → `getCatalogSchema()` → a `/schema` GET.
On the demo catalog this measured **~12 `/schema` GETs per descendant**
(76 for 6 descendants; 16 for a leaf). A1/A2 alone (snapshot construction)
does **not** fix this — it is live-instance refetching, not snapshot
construction.

### 3.1b Lever C — cache the live instance's pathBuilder on the DerivaML instance

deriva-ml already holds the authoritative parsed schema (`self._schema_json`,
`self.model`) for the instance's lifetime; the per-call `/schema` refetch is
pure waste regardless of server 304 support. Fix: **`DerivaML.pathBuilder()`
caches the deriva-py wrapper on the DerivaML instance** and returns the cached
wrapper on subsequent calls, instead of calling `self.catalog.getPathBuilder()`
(which re-fetches `/schema`) every time.

- **Invalidation contract:** deriva-ml does **not** auto-refresh the model
  after schema-mutating writes — callers explicitly call `refresh_model()` /
  `refresh_schema()` to pick up changes (the model is a deliberate snapshot
  until refreshed). So the instance's schema is stable between explicit
  refreshes, and caching the pathBuilder is safe **provided the cache is
  cleared in `refresh_model()` and `refresh_schema()`** (and on
  `pin_schema`/`unpin_schema`, which also re-fetch). Add that invalidation.
- **Scope:** this is the `_ml_instance.pathBuilder()` used pervasively
  (77 call sites); caching it fixes the live-instance refetch for *every*
  caller, not just the estimate — a broad correctness/perf win. The risk is
  staleness after an un-refreshed mutation, which the invalidation hooks
  prevent and which matches the existing `refresh_model` contract.
- **Snapshot instances** also benefit: a snapshot `DerivaML`'s pathBuilder is
  likewise cached on its instance (snapshots are immutable, so it never needs
  invalidation).

With Lever C, `_iter_descendant_rids`'s N `lookup_dataset` calls share **one**
pathBuilder on the live instance — collapsing the O(N) live-instance
`/schema` refetch to O(1).

**Lever A (revised — eliminate, don't memoize) — reuse the live model when
building a snapshot catalog.** `catalog_snapshot()` threads the live
instance's parsed `schema_json` into the new instance's construction so
`_init_online` builds its `DerivaModel` via the existing
`DerivaModel.from_cached(schema_json, catalog=<snapshot catalog>, ...)` path
**without calling `getCatalogSchema()`**. Net `/schema` fetches for the whole
estimate: **0** (the live instance's one construction-time fetch already
happened; snapshots reuse it).

Mechanics (cleanest seam, to be finalized in the plan):

- Add an internal opt-in to the online init path that accepts a
  pre-parsed `schema_json` (or a `DerivaModel`) and, when present, skips
  `getCatalogSchema()` and constructs the model from the supplied dict via
  `from_cached`. The snaptime still flows into the
  `server.connect_ermrest(catalog_id@snaptime)` catalog object, so **data**
  reads are correctly snapshot-pinned — only the redundant **schema** fetch
  is removed.
- `catalog_snapshot()` passes `self.model`'s source `schema_json` (the same
  dict `__init__` already wrote to `SchemaCache`) through that opt-in.
- This is strictly better than memoizing the snapshot object: it removes
  the fetch at the source, and it composes with the memoization below.

**Lever A2 — memoize the snapshot instance per snapshot id.** Even with the
fetch eliminated, each of the 8 `Dataset` call sites still *constructs* a new
snapshot `DerivaML` and rebuilds the model object from the dict. Memoize
`_version_snapshot_catalog` (or `catalog_snapshot`) on the `_ml_instance`,
keyed by the resolved snapshot id (`catalog_id@snaptime`), so all call sites
within a process share one snapshot instance. Cache key is the resolved id —
not the raw `version` arg — so `None`, `"4.11.0"`, and an equivalent parsed
`DatasetVersion` collapse to one entry. Snapshots are immutable, so entries
never go stale; the dict is bounded by the number of distinct snapshots a
session touches (normally 1). A1 (no fetch) + A2 (no rebuild) together make
the second-and-later snapshot accesses effectively free.

**Correctness guard (load-bearing):** reusing the live schema for a snapshot
is valid **only because the snapshot's schema equals live's**. This is true
for the snapshots deriva-ml pins (recent dataset-version snaptimes on a
catalog whose schema hasn't been migrated between the snaptime and now). The
plan must (a) state this assumption explicitly, and (b) add a test that the
reused-schema snapshot model is structurally identical to a freshly-fetched
snapshot model on the demo catalog. If a catalog could have a *schema*
migration between the snaptime and live, the live schema would be wrong for
the snapshot — but deriva-ml's snapshot use is for data-version pinning, not
schema-archaeology, so this is acceptable and must be documented as a
precondition.

### 3.2 Redundant tree walks (the 2.5×)

**Lever B — walk the descendant tree once per estimate.**
`_iter_descendant_rids(dataset)` is invoked by both `anchors_for` and
`_exclude_empty_associations`, and `_exclude_empty_associations` *also*
re-fetches members per RID. With Lever A in place each of those calls is far
cheaper (no `/schema` refetch), but the **redundant tree traversal** remains
(N `list_dataset_children` round-trips, done ~twice).

Memoize the descendant-RID list for the duration of one
`aggregate_queries` call so the tree is walked once. Options considered:

- **B1 (recommended):** have `aggregate_queries` compute the descendant set
  **once** and pass it to both `anchors_for` and `build_policy`
  (thread it through, or stash it on the builder for the call). Explicit, no
  hidden state.
- **B2:** memoize `_iter_descendant_rids` on the `DatasetBagBuilder` keyed by
  `dataset.dataset_rid`. Simpler call sites, but adds instance state that
  must not outlive a logical operation.

Lever B is **secondary** — after Lever A removes the `/schema` cost, the
remaining tree-walk is ~85 cheap `Dataset_Dataset`/`Dataset` GETs done twice
(~170, sub-second-each but serial). We will implement A first, **re-measure**,
and only add B if the post-A time still exceeds the < 30 s target. (YAGNI:
don't build B speculatively if A alone hits the goal.)

### 3.3 Out of scope (explicitly not changing)

- The RID-union exactness of counts (kept — user decision).
- Swapping `attributes(RID)` → `aggregates(Cnt(RID))` (the count queries are
  ~2 % of cost; not worth the dedup-semantics risk now).
- Any async/concurrency change to the estimate query layer (already bounded
  by ADR-0008 / #289; not the bottleneck).
- deriva-py `CatalogBagBuilder` (correctly cached; not the cause).

## 4. Deliverables

- `src/deriva_ml/core/base.py`:
  - an internal opt-in on the online init path to skip `getCatalogSchema()`
    and build the model from a supplied parsed schema via
    `DerivaModel.from_cached` (Lever A1); **[DONE]**
  - `catalog_snapshot()` threads the live instance's parsed `schema_json`
    through that opt-in so snapshot construction performs **0** `/schema`
    fetches (A1), and memoizes the constructed snapshot instance by resolved
    snapshot id so the 8 `Dataset` call sites share one instance (A2).
    **[DONE]**
- `src/deriva_ml/core/mixins/path_builder.py` (Lever C — the live-instance
  refetch fix, §3.1b):
  - `DerivaML.pathBuilder()` caches the deriva-py wrapper on the DerivaML
    instance (e.g. `self._path_builder` set on first call) and returns it on
    subsequent calls, instead of calling `self.catalog.getPathBuilder()` (a
    `/schema` refetch) every time.
- `src/deriva_ml/core/base.py` + `src/deriva_ml/model/catalog.py` (Lever C
  invalidation):
  - `refresh_model()` / `refresh_schema()` / `pin_schema()` /
    `unpin_schema()` clear the cached `pathBuilder` so a deliberate schema
    refresh is observed by the next `pathBuilder()` call. (Match the existing
    refresh contract — these are the only paths that legitimately change the
    instance's schema.)
- (Conditional, only if post-A+C re-measure misses target)
  `src/deriva_ml/dataset/bag_builder.py`:
  - single-tree-walk wiring for `aggregate_queries` (Lever B1).
- Tests (below).
- Patch/minor version bump after merge.

## 5. Testing / verification

### 5.1 Unit — `catalog_snapshot()` performs no `/schema` fetch

A unit test that spies on `getCatalogSchema()` (or the underlying `/schema`
GET) and asserts that `catalog_snapshot(version_snapshot)` constructs the
snapshot instance **without** calling it — the snapshot model is built from
the live instance's already-parsed schema via `from_cached`. Drive with a
stub catalog whose `getCatalogSchema` increments a counter; assert the
counter does not increment across a `catalog_snapshot()` call.

### 5.1b Equivalence — reused-schema model == freshly-fetched model

On the demo catalog, build a snapshot instance two ways: (a) via the new
schema-reusing `catalog_snapshot()`, (b) by forcing a real
`getCatalogSchema()` fetch for the same snaptime. Assert the two
`DerivaModel`s are structurally identical (same schemas, tables, columns,
FKs). This pins the load-bearing assumption from §3.1.

### 5.1c Unit — live `pathBuilder()` is cached + invalidated on refresh (Lever C)

Spy on `ErmrestCatalog.getPathBuilder` (or `/schema` GET). Assert:
- repeated `ml.pathBuilder()` calls on a live instance trigger
  `getPathBuilder()` / a `/schema` GET **once**, not per call (the wrapper is
  cached on the DerivaML instance);
- after `ml.refresh_model()` (or `refresh_schema()`), the **next**
  `pathBuilder()` call rebuilds (cache cleared) — so a deliberate schema
  refresh is still observed. This pins the invalidation contract so the cache
  can't mask a real schema change.

### 5.2 Integration — GET-count regression guard (live catalog)

Against the demo catalog (the same `catalog_manager.ensure_datasets` fixture):
build a small **nested** dataset, then run `estimate_bag_size` while counting
catalog GETs (monkeypatch `DerivaBinding.get` to increment a counter, as the
investigation harness did). Assert the **`/schema` GET count does not scale
with the number of descendants** — it should be a small fixed constant
independent of nesting depth (post-fix the snapshot path adds **0** schema
fetches; the only `/schema` GETs are the live instance's own
construction-time fetch plus any deliberate refresh). Concretely: assert the
`/schema` count for an N-descendant dataset equals the count for a
single-node dataset (± a small fixed allowance), **not** O(N). This pins the
regression: if a future change reintroduces per-descendant snapshot-catalog
fetching, the count grows with N and the test fails.

- The demo catalog must exhibit nesting for this to be meaningful; the
  `catalog_with_datasets` fixture already builds nested splits. Assert the
  descendant count is ≥ 2 first (skip with a clear message if not), so the
  guard is exercised.

### 5.3 Correctness — numbers unchanged

Assert `estimate_bag_size` on the demo nested dataset returns the **same**
`total_rows` / `total_asset_bytes` / per-table counts before and after the
change (capture once on `main`, compare). Since the change is pure caching,
the estimate dict must be identical.

### 5.4 Live smoke (manual, recorded in PR)

Re-run `estimate_bag_size(2-277G)` against www.eye-ai.org and record the new
wall-clock + `/schema` GET count in the PR description (target < 30 s, was
~510 s). Not a CI test (needs the production catalog); the demo-catalog GET
guard is the CI pin.

## 6. Risks

- **Schema drift between snaptime and live (the load-bearing assumption):**
  reusing the live schema for a snapshot is correct only when the snapshot's
  schema equals live's. deriva-ml pins snapshots for *data*-version
  reproducibility on catalogs whose schema is not migrated mid-session, so
  this holds. The §5.1b equivalence test pins it; the precondition is
  documented on the new init opt-in. A future caller that needs a snapshot
  from *before* a schema migration must not use the schema-reuse path — call
  this out in the docstring.
- **`version=None` branch unaffected:** `_version_snapshot_catalog(None)`
  already returns the shared `_ml_instance` (no new catalog, no fetch). The
  change only touches the versioned branch. Confirm the `None` branch is
  untouched.
- **Write paths:** the schema-reuse opt-in is for read-oriented snapshot
  construction. Snapshots are read-only by nature; verify no caller writes
  through a snapshot instance built this way.
- **Offline mode:** `_init_offline` already builds from cache and never
  fetches `/schema`; the new opt-in must not change that path.
