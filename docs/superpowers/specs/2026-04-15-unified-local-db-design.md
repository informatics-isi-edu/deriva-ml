# Unified Local SQLite Layer for deriva-ml

**Status:** Draft, pending review.
**Date:** 2026-04-15.
**Scope:** Design spec. Implementation plan follows separately.

## 1. Goals and scope

### 1.1 Goals

Build a unified client-side SQLite layer for deriva-ml that subsumes today's
per-bag `DatabaseModel`, the ad-hoc `WorkingDataCache`, and JSON-based asset
state. It becomes the substrate for:

1. A general client-side caching mechanism based on SQLite that complements
   the existing use of DBCatalog in deriva-ml and builds on that foundation.
2. Robust denormalization, via paged server requests assembled into local SQL
   joins rather than single complex server-side joins.
3. A unified denormalization implementation that serves both live-catalog and
   bag-driven workflows.
4. A foundation for future evolution where arbitrary catalog slices (not only
   BDBags) are stored into a local database, with an eventual upload-back
   path.
5. Replacement of today's filesystem/JSON caches (notably `AssetManifest`)
   with database-backed storage, located in the deriva-ml working directory so
   it can be shared across catalog connections and script invocations.
6. Promotion of client-side cached tabular-read capabilities (currently
   maintained separately in deriva-mcp) into the deriva-ml public API, so any
   consumer — MCP, notebooks, scripts — shares the same semantics.

### 1.2 Non-goals

- Offline editing or conflict resolution across disconnected sessions. We
  target the "work on a local DB while connected" case only.
- Replacing BDBag as a portable interchange format. Bags remain the way to
  hand data to another party; the local DB is the working copy.
- A general-purpose query engine over the catalog. We are not reimplementing
  ERMrest client-side.
- Migrating cheap filesystem markers (`validated_check.txt`, `fetch.txt`,
  etc.) into the database.
- Preserving existing cached bag directories across the upgrade. New layout
  applies to new caches; old caches are abandoned and users can re-download.
  The one narrow exception is in-progress `AssetManifest` state (§4.1).

### 1.3 Migration posture

Additive, not big-bang. The new modules live alongside existing code. Today's
`DatabaseModel` becomes the slice-DB implementation with minimal change.
`WorkingDataCache` is absorbed into the new workspace. `AssetManifest` is
rewritten to persist into the working DB. The two denormalization
implementations are deleted and replaced with one.

## 2. Architecture and layout

### 2.1 On-disk layout

```
{working_dir}/
  catalogs/
    {host}__{cat}/
      working.db              # per-catalog working DB (WAL mode)
      slices/
        {slice_id}/
          slice.db            # immutable slice DB (WAL mode, opened read-only at query time)
          assets/             # materialized asset files (same role as today's bag data/assets/)
          manifest.json       # slice provenance descriptor (root_rid, spec, snapshot_ts, content hash)
```

`slice_id` is a content hash of `(root_rid, reachability_spec, snapshot_ts)`.
For downloaded bags, the existing bag hash is used as `slice_id` so bag
identity is preserved.

Today's `{working_dir}/databases/{bag_hash}/` and
`{working_dir}/working-data/catalog.db` move to this layout. No automatic
migration of existing bag directories is performed. `WorkingDataCache`'s
contents are a cache — users may delete or let the new code start fresh.

### 2.2 Two-tier database model

**Slice databases.** Immutable, native-shape, complete-by-construction. One
per materialized slice. Identity is `(root_rid, spec, snapshot_ts)`. A bag is
a special case of a slice. Slice DBs are opened read-only at query time.

**Working database.** One per `(hostname, catalog_id)`. Mutable. Holds:

- partial table caches populated by the paged fetcher
- cached tabular results (denormalization outputs, `preview_table`
  results, feature-value fetches, etc.)
- asset-metadata cache (rows fetched from asset tables)
- execution state (`AssetManifest` tables)

The working DB can have multiple logical areas distinguished by SQL schema
qualifier within the single file (e.g. `cached_results.*`,
`execution_state.*`, plus the native-shape tables).

### 2.3 Module map

**New modules:**

- `deriva_ml.local_db.schema` — unified SQLAlchemy schema builder. Consumes
  either a bag `schema.json` or a live ERMrest `Model` and produces the same
  metadata shape on both sides, so cross-DB joins are type-safe. Supersedes
  `model/schema_builder.py` (which handles only bags today).

- `deriva_ml.local_db.paged_fetcher` — the fetch primitive. Owns keyset
  paging, RID-set batching with byte-length guard and POST fallback, bounded
  concurrency, and per-operation dedup.

- `deriva_ml.local_db.workspace` — runtime orchestrator. Holds the working DB
  handle, a registry of known slices, and the `attach_slice(...)` mechanism.
  Exposes the cached-tabular-read API.

- `deriva_ml.local_db.denormalize` — single denormalization implementation.

**Refactors:**

- `model/database.py::DatabaseModel` — becomes the slice-DB class. Loses its
  role as the denormalization engine (that moves to the new module). Gains a
  read-only open mode.
- `core/working_data.py::WorkingDataCache` — internal implementation detail
  of `Workspace`; public API replaced by the cached-tabular-read surface.
- `dataset/dataset.py::_denormalize_datapath` and
  `dataset/dataset_bag.py::_denormalize` — deleted. `denormalize_*` methods
  on `Dataset` and `DatasetBag` become thin delegates to the unified module.
- `asset/manifest.py::AssetManifest` — rewritten to persist into working-DB
  tables (same public API; storage swap only).

### 2.4 Component relationships

```
               ┌─────────────────────────┐
               │     Workspace           │  (per host+catalog)
               │  - working.db handle    │
               │  - slice registry       │
               │  - attach_slice()       │
               └──────────┬──────────────┘
                          │ uses
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
  ┌──────────────┐  ┌──────────┐    ┌──────────────┐
  │ PagedFetcher │  │ Denorm-  │    │ AssetManifest│
  │              │  │ alizer   │    │ (DB-backed)  │
  └──────┬───────┘  └────┬─────┘    └──────┬───────┘
         │                │                 │
         │ writes rows    │ SQL joins       │ reads/writes
         ▼                ▼                 ▼
  ┌────────────────────────────────────────────────┐
  │         SQLite connection                       │
  │   main  = working.db                           │
  │   slice = (optional) slice.db  [ATTACH]        │
  └────────────────────────────────────────────────┘
```

`PagedFetcher`, the denormalizer, and `AssetManifest` all hold a `Workspace`
and never directly know whether they're in a live-catalog scenario (main
schema only), a bag scenario (slice attached), or a hybrid case.

### 2.5 Concurrency and locking

SQLite WAL mode is required. Multiple Python processes may share a working DB
(e.g., concurrent scripts in the same working directory). WAL gives readers
non-blocking semantics against writers. Default `PRAGMA synchronous=NORMAL`.
`PagedFetcher` commits every N pages (configurable, default 10) to bound the
worst-case writer-block window.

### 2.6 Schema versioning

The unified schema is versioned via a `schema_meta` table with an integer
version. A small migration runner at workspace-open time upgrades older
working DBs. Slice DBs are pinned to the version of their producer and are
read-only at query time; schema-version mismatches are handled by the
tolerant column resolution described in §5.2 (R2).

### 2.7 Compatibility with existing bags

No migration of bag directories. Existing bag caches are effectively
abandoned — the first access to a dataset after upgrade may re-download.
BDBag materialization is idempotent and resumable via `fetch.txt`, so the
cost of re-downloading is walking the fetch manifest and verifying what's on
disk. Release notes will point users at a one-line shell command
(`mv`/symlink) if they want to preserve a large cache by hand.

### 2.8 Multi-schema handling

The unified schema builder treats SQLAlchemy schema-qualified names exactly
as today (`deriva-ml.Dataset`, `isa.Image`, etc.). Within a single DB,
schema-qualified names coexist naturally. Across attached DBs, SQLite's
database namespace (`main`, `slice`) is orthogonal to the `schema.table`
naming inside each file; the denormalizer emits fully qualified references
(e.g. `slice."isa.Image"`) so callers never hand-write these. Working DB and
slice DB are always built from the same ERMrest `Model` at build time,
guaranteeing alignment. Multi-schema collisions between a slice and the
current catalog (same schema name, different column shape) are out of scope
for this phase and will be revisited when upload-back (goal #4) is designed.

## 3. Paged fetching and unified denormalization

### 3.1 `PagedFetcher` API

Three public methods. The class is constructed from an ERMrest catalog
handle and a `Workspace`.

```python
PagedFetcher.fetch_predicate(
    table,                  # ERMrest table ref
    predicate=None,         # optional ERMrest filter expression
    target_table,           # SQLAlchemy Table in working/slice DB
    sort=("RID",),          # keyset sort — must end in a unique column
    page_size=1000,
    on_page=None,           # optional callback(rows); default inserts into target_table
) -> int
```

Streams everything matching `predicate` into local SQLite via keyset paging
(`@sort(...)@after(...)?limit=N`). No full-result buffering on server or
client.

```python
PagedFetcher.fetch_by_rids(
    table,
    rids: Iterable[str],
    target_table,
    rid_column="RID",       # or any indexed column (e.g. FK column)
    batch_size=500,
    max_url_bytes=6144,
    concurrency=4,
) -> int
```

Fetches rows whose `rid_column` value is in the given set. Default batch size
500, with a byte-length guard that reduces the batch when the serialized URL
would exceed `max_url_bytes`. If even the reduced batch overflows, the request
switches to POST-body filtering. Bounded concurrency across batches.
Per-operation dedup skips RIDs already fetched.

```python
PagedFetcher.fetched_rids(table) -> set[str]
```

Returns RIDs already materialized in the workspace for the given table.
Callers use this to compute deltas before requesting more.

### 3.2 Cardinality heuristic

When a RID set approaches the table size, RID-set batching is the wrong
primitive. The fetcher applies this rule:

- If `|rid_set| / table_row_count > threshold` (default 0.5), fall back to
  `fetch_predicate` (paged keyset scan) and filter locally.
- Otherwise use `fetch_by_rids`.

Row counts come from `/aggregate/T/cnt(*)`, memoized per table per operation.

### 3.3 URL length handling

Default `max_url_bytes=6144` leaves headroom under typical 8KB proxy limits.
Handling escalates in order:

1. Shrink batch size to fit GET.
2. If even the minimum practical batch overflows, switch the request to POST
   with the filter expression in the body (supported by ERMrest and
   deriva-py).
3. If POST also fails at runtime, shrink batch and retry with logging; as a
   last resort, fall back to `fetch_predicate` with a local filter.

Users never see this; the fetcher picks the transport.

### 3.4 Unified denormalization algorithm

Lives in `deriva_ml.local_db.denormalize`. Takes a `Workspace`, an optional
slice to attach, a list of `include_tables`, and an optional root (e.g. a
dataset RID). Produces rows as an iterator, a pandas DataFrame, or a
SQLAlchemy `Select`.

Steps:

1. **Resolve the required table set.** BFS from the primary table through
   `include_tables` using the ERMrest model's FK graph. Detect ambiguous
   paths up front (same check both today's implementations perform, lifted
   into the unified path). Model-only step, no I/O.
2. **Determine required rows.** For each table in the chain, compute the
   required RID set by walking from root forward — root table uses the
   dataset/slice filter predicate; child tables derive RID sets from
   previously fetched rows' FKs or from association tables.
3. **Fetch missing rows only.** For each table, subtract
   `workspace.fetched_rids(table)` from the required set and fetch the
   remainder. If a slice is attached and already contains the rows, skip the
   fetch entirely.
4. **Build the SQL `Select`.** One SQLAlchemy `Select` joining across the
   chain. Columns prefixed by table name (existing `denormalize_column_name`
   behavior). JOIN conditions come from the FK graph from step 1. If a slice
   is attached, the `Select` references the slice version of a table
   (`slice.{table}`) when that table's rows came entirely from the slice,
   the working-DB version (`main.{table}`) when they came entirely from live
   fetches, or a `UNION ALL` subquery over both when both contributed. The
   "who contributed" decision is per-table, made from the step-3 bookkeeping
   (which RIDs came from which source); callers never choose.
5. **Execute and yield.** Same execution path regardless of row source.

### 3.5 What this unifies

- Today's `Dataset._denormalize_datapath` and `DatasetBag._denormalize`
  disappear. Both callers reduce to `workspace.denormalize(include_tables,
  root=...)`. The bag case just attaches the slice first; the catalog case
  fetches on demand.
- Ambiguous-path detection is one implementation.
- Column naming, skip-column handling, multi-schema handling are one
  implementation.
- The hybrid case ("bag plus a fresh feature the catalog has that the bag
  doesn't") becomes free.

### 3.6 Caching denormalization results

Denormalization output can optionally be materialized into `working.db` under
`cached_results.*`. The cache key is
`hash(root, include_tables, slice_id or snapshot, schema_version)`. Managed
by the same cached-tabular-read API described in §4.2, so denormalization
caching and general cached reads share one implementation.

## 4. Asset manifest migration and cached tabular-read API

### 4.1 `AssetManifest` on SQLite

Two tables under an `execution_state` schema in the working DB:

```sql
execution_state.assets (
    execution_rid    TEXT NOT NULL,
    key              TEXT NOT NULL,   -- manifest key (relative path)
    asset_table      TEXT NOT NULL,
    schema           TEXT NOT NULL,
    asset_types      JSON,            -- small list, stored as JSON text
    metadata         JSON,            -- arbitrary column values
    description      TEXT,
    status           TEXT NOT NULL,   -- pending | uploaded | failed
    rid              TEXT,
    uploaded_at      TEXT,
    error            TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    PRIMARY KEY (execution_rid, key)
)

execution_state.features (
    execution_rid    TEXT NOT NULL,
    feature_name     TEXT NOT NULL,
    target_table     TEXT NOT NULL,
    schema           TEXT NOT NULL,
    values_path      TEXT NOT NULL,
    asset_columns    JSON,
    status           TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    PRIMARY KEY (execution_rid, feature_name)
)
```

Indexes `(execution_rid, status)` on both.

**Crash safety.** WAL + `synchronous=NORMAL`, with each `add_asset`,
`mark_uploaded`, and `mark_failed` committed in its own transaction. WAL
recovery rolls forward committed transactions; the uncommitted-write loss
window is equivalent to today's "write-but-not-fsync" edge case.

**JSON import.** On workspace open, scan the working dir for pre-existing
`*.manifest.json` files. For each: parse, upsert into the tables, move the
file to `{path}.migrated.json` (do not delete; users may want it for
forensics). Idempotent via the primary keys.

**JSON export.** `AssetManifest.to_json()` dumps the old file format on
demand, for debugging and postmortems.

### 4.2 Cached tabular-read API

Public methods on `Workspace`, reachable from the `DerivaML` instance.

```python
workspace.cached_table_read(
    table: str | Table,
    predicate=None,                   # ERMrest or SQLAlchemy-style filter
    columns=None,                     # subset; None = all non-system columns
    refresh: bool = False,            # force re-fetch even if cached
    ttl: timedelta | None = None,     # optional freshness bound
) -> CachedResult
```

Returns a `CachedResult` — opaque handle wrapping a cache key and the
underlying SQLite table.

```python
result.to_dataframe()
result.iter_rows()
result.query(sort=..., filter=..., limit=..., offset=...) -> CachedResult
result.row_count
result.cache_key
result.fetched_at
result.invalidate()
```

`query(...)` runs SQL against the cached table without re-fetching. Sort,
filter, and pagination are applied locally. Returns a new `CachedResult` so
chained calls work. This is the operation MCP's `query_cached_result`
delegates to.

**Storage.** One table per cached result under `cached_results.*`, plus a
registry:

```sql
cached_results._registry (
    cache_key       TEXT PRIMARY KEY,
    source          TEXT NOT NULL,   -- "table", "denormalize", "feature_values", ...
    params          JSON NOT NULL,
    row_count       INTEGER,
    fetched_at      TEXT NOT NULL,
    schema_version  INTEGER NOT NULL
)
```

`cache_key = hash(source, params, catalog_snapshot_ts_if_pinned)`.

**Top-level workspace methods:**

```python
workspace.list_cached_results() -> list[CachedResultMeta]
workspace.get_cached_result(cache_key) -> CachedResult
workspace.invalidate_cache(cache_key=None, source=None)
```

**Extensibility.** Denormalization results and feature-value fetches
register with the same machinery by picking a `source` tag and a stable
params hash. One cache, many producers.

### 4.3 MCP migration (Phase 3)

Direct mapping:

- `preview_table` → `workspace.cached_table_read(table, ...).query(limit=...,
  offset=...)`
- `list_cached_results` → `workspace.list_cached_results()`
- `query_cached_result` → `workspace.get_cached_result(cache_key).query(...)`
- `invalidate_cache` → `workspace.invalidate_cache(...)`

MCP keeps its tool surface; the storage and logic live in deriva-ml. Cached
results are then visible to scripts and notebooks connected to the same
working directory and catalog, not siloed in the MCP process.

## 5. Testing, risks, and rollout

### 5.1 Testing

**Unit (no catalog):**

- `PagedFetcher` against a fake ERMrest client: byte-length guard splits
  correctly; POST fallback triggers at the right threshold; dedup across
  calls; cardinality heuristic switches at the expected point.
- `local_db.schema` against both a bag `schema.json` fixture and a canned
  ERMrest `Model` dump; assert identical SQLAlchemy metadata.
- `AssetManifest` DB-backed implementation runs the existing API-level test
  suite unchanged. Migration test seeds JSON manifests, opens workspace,
  asserts tables populated and `.migrated.json` sidecar present.

**Integration (live catalog):**

- One denormalization test matrix parameterized by source (`"bag"`,
  `"catalog"`, `"hybrid"`). Same assertions for all three. Verifies
  unification directly.
- Cached tabular-read lifecycle: fetch → re-query with sort/filter →
  invalidate → re-fetch.
- Multi-schema test: workspace+slice attach produces correct joined results
  when both sides have multi-schema tables.

**Regression safety net:** existing `_denormalize_datapath` and
`_denormalize` tests continue to run during Phase 1. Deleted only when the
unified path passes at parity in Phase 2.

### 5.2 Risks

**R1: SQLite concurrent-write contention under parallel scripts.**
WAL mitigates most of it; a long transaction (e.g. a 100k-row paged fetch
committed as one unit) would block other writers. *Handling:* `PagedFetcher`
commits every N pages (default 10). Worst-case writer-block window is
O(seconds).

**R2: Schema drift between working DB and attached slice.**
Slice built from an older snapshot; working DB reflects evolved catalog
schema. *Handling:* the denormalizer selects explicit column lists from the
FK graph (not `SELECT *`); missing columns read as NULL; extra columns on
one side are ignored. Documented and tested.

**R3: URL length despite POST fallback.**
POST-body filters have their own limits and are not universally supported on
every ERMrest endpoint. *Handling:* fetcher tests include an oversized-batch
case against the live catalog exercising the POST path end-to-end. Runtime
fallback shrinks the batch and retries with logging; last resort falls back
to `fetch_predicate` with a local filter.

**R4: Working DB grows over time.**
Accumulated cached results, fetched rows, old execution manifests.
*Handling:* `workspace.vacuum(older_than=...)` drops cached results and
fetched-row tables past a configurable age (default 30 days). Execution
state is preserved until the execution is final. Users can nuke the DB
entirely; it's a cache.

**R5: Two catalogs open simultaneously.**
Each `DerivaML` instance keyed by `(host, cat)` gets its own working DB. No
global state. Test covers opening two, using both, closing both.

**R6: Small-dataset regression.**
For tiny datasets, today's single-query path is faster than paged fetches.
*Handling:* skip paging when estimated row count is below a threshold
(default 500) — issue one bounded request and done.

### 5.3 Phased rollout

**Phase 1 — Foundation.** No user-visible API changes.
- `local_db.schema`
- `local_db.workspace` (per-catalog DB, slice attach/detach,
  `WorkingDataCache` absorbed)
- `PagedFetcher`
- `AssetManifest` DB-backed + JSON import
- Existing denormalization code unchanged; new code sits beside it.
- Users see no behavioral change except path move of the working cache.

**Phase 2 — Unified denormalization.**
- `local_db.denormalize` using Phase 1 primitives.
- `Dataset.denormalize_*` and `DatasetBag.denormalize_*` become thin
  delegates.
- Old `_denormalize_datapath` and `_denormalize` deleted.
- Cached tabular-read API exposed publicly.
- Parameterized denormalization tests (bag/catalog/hybrid) go green.

**Phase 3 — MCP migration (in deriva-mcp repo).**
- MCP cache tools delegate to workspace APIs.
- MCP-local cache DB deprecated with a migration note in the deriva-mcp
  release.

**Out of scope for this spec:** goal #4 upload-back. Foundation is in place
after Phase 1 (workspace + slice + attach). The actual diff-and-apply logic
is its own design problem (conflict resolution, FK ordering, partial-failure
handling) and will get its own spec once this work ships.

## 6. Open questions

None blocking. Items to revisit in follow-on specs:

- Upload-back from local DB (goal #4, deferred).
- Schema-name collisions between slice and live catalog during upload-back
  (§2.8).
- Whether `validated_check.txt` and `fetch.txt` should eventually move into
  the DB for a fully in-database bag representation (deferred per §1.2).
