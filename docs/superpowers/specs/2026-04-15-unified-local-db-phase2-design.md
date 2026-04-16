# Unified Local SQLite Layer — Phase 2: Denormalization & Cached Reads

**Status:** Draft, pending review.
**Date:** 2026-04-15.
**Depends on:** Phase 1 spec (`2026-04-15-unified-local-db-design.md`) and its
implementation (branch `claude/compassionate-visvesvaraya`).
**Scope:** Design spec. Implementation plan follows separately.

## 1. Goals and scope

### 1.1 Goals

1. **Unified denormalization engine.** One code path produces denormalized
   wide tables regardless of whether data comes from a live catalog, a
   downloaded bag, or a hybrid of both.
2. **Robust catalog denormalization.** Rows are fetched via paged requests
   into local SQLite, then joined locally. No single server-side query that
   can time out.
3. **Cached tabular-read API.** Cached results (denormalized views, table
   previews, feature-value fetches) are queryable with sort/filter/pagination
   without re-fetching. This API is the contract MCP will delegate to in
   Phase 3.
4. **Clean up Phase 1 compatibility shims.** Delete `_LegacyWorkingDataView`,
   `WorkingDataCache`, and the `ml.working_data` property. Replace with a
   direct `ml.workspace` property and the cached-read API.

### 1.2 Non-goals

- MCP migration (Phase 3, separate repo/spec).
- Upload-back from local DB (future spec).
- New slice identity/provenance system.
- Backward compatibility for `ml.working_data` callers (breaking change;
  documented in release notes).

## 2. Working DB layout change

### 2.1 Multi-schema via ATTACH

The working DB changes from a single file (`working.db`) to a directory with
per-schema files, matching `SchemaBuilder`'s multi-schema ATTACH pattern:

```
{working_dir}/
  catalogs/
    {host}__{cat}/
      working/                  # was: working.db (Phase 1)
        main.db                 # ManifestStore, cached_results registry, schema_meta
        isa.db                  # domain schema tables (fetched rows)
        deriva-ml.db            # ML schema tables (fetched rows)
      slices/
        {slice_id}/
          main.db + per-schema files   # unchanged from Phase 1
```

On every connection, the engine (opened on `main.db`) ATTACHes each
per-schema `.db` file. Cross-schema joins work via ORM relationships set up
by `SchemaBuilder`, identical to today's bag path.

`ManifestStore` tables, `schema_meta`, and the `cached_results_registry`
live in `main.db`. Per-schema tables (populated by `PagedFetcher`) live in
their respective ATTACH'd files.

### 2.2 Schema lifecycle

**When built:** Lazily, on first denormalization or paged-fetch request. Not
at workspace creation (which doesn't require a catalog connection).

**How built:** `LocalSchema.build(model=catalog_model, schemas=[ml_schema,
*domain_schemas], database_path=working_dir)` creates the per-schema files,
ATTACH handler, ORM classes, and cross-schema relationships. Cached on the
`Workspace` object for its lifetime.

**Staleness:** Tied to the catalog model at build time. A
`workspace.rebuild_schema()` method disposes and recreates the
`LocalSchema`. Alternatively, delete the working directory (it's a cache).

### 2.3 Slice attachment with multi-schema

When a slice is attached via `workspace.attach_slice(slice_id)`, each of the
slice's per-schema files is ATTACH'd under a prefixed alias:

```
main.db                              # working DB infrastructure
isa.db           AS 'isa'            # working DB domain tables
deriva-ml.db     AS 'deriva-ml'      # working DB ML tables
slice/isa.db     AS 'slice_isa'      # slice domain tables
slice/ml.db      AS 'slice_deriva-ml' # slice ML tables
```

The denormalizer resolves table references to the correct ATTACH alias based
on whether rows for that table came from the working DB or the slice.

All ATTACH'd databases are visible in one SQLite connection, so JOINs across
working DB schemas, slice schemas, and `main.db` all work.

### 2.4 Migration from Phase 1

Phase 1's `working.db` single file is a cache. No migration code — first
access post-upgrade creates the new directory layout. Phase 1 also stored
`ManifestStore` tables in the old `working.db`; those are re-created fresh
(manifest data for in-progress executions is imported from JSON sidecars by
the existing `import_legacy_manifests` path if any exist).

## 3. Unified denormalization engine

### 3.1 Module

`deriva_ml.local_db.denormalize`

### 3.2 Entry point

```python
def denormalize(
    workspace: Workspace,
    model: DerivaModel,
    dataset_rid: str,
    include_tables: list[str],
    version: DatasetVersion | str | None = None,
    slice_id: str | None = None,
    source: str = "catalog",   # "catalog" | "slice" | "hybrid"
) -> DenormalizeResult
```

`DenormalizeResult` wraps the rows (as iterator or DataFrame) plus metadata
(columns, cache key, row count).

### 3.3 Algorithm

**Step 1 — Plan the join.** Call `model._prepare_wide_table(dataset,
dataset_rid, include_tables)` to get `(element_tables, column_specs,
multi_schema)`. This returns join paths, join conditions (including LEFT JOIN
for nullable FKs and composite FKs), and the output column list. No I/O —
pure model analysis. This is the same call both `Dataset._denormalize_datapath`
and `DatasetBag._denormalize` make today.

**Step 2 — Ensure the working DB has a schema.** If
`workspace.local_schema` is None, build it from the live `Model` via
`LocalSchema.build(model, schemas, database_path=working_dir)`. Creates
per-schema `.db` files and ORM. Cached for workspace lifetime.

**Step 3 — Populate rows.** Depends on `source`:

- **`"catalog"`** (live fetch): For each table in the join plan, use
  `PagedFetcher` to populate the working DB. Follow the chain: fetch
  association table rows filtered by `Dataset=dataset_rid`, use resulting
  RID sets to fetch the next table, etc. `fetched_rids` dedup ensures no
  re-fetching of rows already in the working DB. When a `version` is
  specified, the `PagedFetcher` operates against the snapshot catalog
  (via `ErmrestSnapshot`) so fetched rows reflect the pinned version,
  not the current catalog state.

- **`"slice"`** (bag): Attach the slice via
  `workspace.attach_slice(slice_id)`. Rows are already in the slice's
  per-schema files. No fetching needed.

- **`"hybrid"`**: Attach the slice AND fetch any tables that are in the
  join plan but missing from the slice (e.g. a feature table added after the
  bag was exported). Check which tables exist in the slice, fetch only the
  missing ones into the working DB.

**Step 4 — Build the SQL SELECT.** Using the join plan from step 1,
construct a SQLAlchemy `select(*labeled_columns).select_from(...).join(...)`
chain. Table references are resolved to the correct ATTACH alias:

- `"catalog"` mode: all tables reference working DB schemas
  (`isa.Image`, `deriva-ml.Dataset_Image`).
- `"slice"` mode: all tables reference slice schemas
  (`slice_isa.Image`, etc.).
- `"hybrid"` mode: per-table decision based on which source has the rows.

The ORM classes from `_prepare_wide_table()` reference the working DB's
`LocalSchema`. For slice tables, the same column structure is used but with
the ATTACH'd alias. This works because `SchemaBuilder` produces identical
column shapes from the same `Model`.

**Step 5 — Execute and return.** Run the SELECT against the workspace
engine (which has everything ATTACH'd). Yield rows as dicts or materialize
as DataFrame.

### 3.4 What `Dataset.denormalize_as_dataframe` becomes

```python
def denormalize_as_dataframe(self, include_tables, version=None, **kwargs):
    result = denormalize(
        workspace=self._ml.workspace,
        model=self._ml.model,
        dataset_rid=self.rid,
        include_tables=include_tables,
        version=version,
        source="catalog",
    )
    return result.to_dataframe()
```

### 3.5 What `DatasetBag.denormalize_as_dataframe` becomes

```python
def denormalize_as_dataframe(self, include_tables, version=None, **kwargs):
    result = denormalize(
        workspace=self._workspace,
        model=self.model,
        dataset_rid=self.dataset_rid,
        include_tables=include_tables,
        slice_id=self._slice_id,
        source="slice",
    )
    return result.to_dataframe()
```

A `DatasetBag` creates a lightweight `Workspace` pointing at its bag
directory on open. The bag's existing per-schema `.db` files become the
slice. No data movement.

### 3.6 What this deletes

- `Dataset._denormalize_datapath` (~190 lines)
- `DatasetBag._denormalize` (~100 lines)
- Both callers reduce to three-line delegates.
- Ambiguous-path detection, column naming, multi-schema handling — one
  implementation, not two.

## 4. Cached tabular-read API

### 4.1 Module

`deriva_ml.local_db.result_cache`

### 4.2 `CachedResult` handle

```python
class CachedResult:
    cache_key: str
    source: str            # "catalog", "bag", "denormalize", "feature_values"
    row_count: int
    columns: list[str]
    fetched_at: datetime

    def to_dataframe(self) -> pd.DataFrame
    def iter_rows(self) -> Generator[dict, None, None]
    def query(
        self,
        sort_by: str | None = None,
        sort_desc: bool = False,
        filter_col: str | None = None,
        filter_val: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> CachedResult
    def invalidate(self) -> None
```

`query()` runs SQL against the cached table without re-fetching. Returns a
new `CachedResult` so chained calls work.

### 4.3 Workspace methods

```python
workspace.cached_table_read(
    table: str,
    predicate: str | None = None,
    columns: list[str] | None = None,
    source: str = "catalog",
    refresh: bool = False,
    ttl: timedelta | None = None,
) -> CachedResult

workspace.cache_denormalized(
    model: DerivaModel,
    dataset_rid: str,
    include_tables: list[str],
    version: str | None = None,
    source: str = "catalog",
    slice_id: str | None = None,
    refresh: bool = False,
) -> CachedResult

workspace.list_cached_results() -> list[CachedResultMeta]
workspace.get_cached_result(cache_key: str) -> CachedResult
workspace.invalidate_cache(
    cache_key: str | None = None,
    source: str | None = None,
) -> int
```

### 4.4 Storage in `main.db`

```sql
cached_results_registry (
    cache_key       TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    tool_name       TEXT NOT NULL,
    params_json     TEXT NOT NULL,
    columns_json    TEXT NOT NULL,
    row_count       INTEGER NOT NULL,
    created_at      REAL NOT NULL,
    ttl_seconds     INTEGER
)
```

Each cached result is a separate table in `main.db` named by its cache key
(e.g. `rc_a1b2c3d4e5f67890`). Registry tracks what each table represents.

### 4.5 Cache key generation

`sha256(f"{tool_name}:{json.dumps(sorted_params)}")[:16]` prefixed with
`rc_`. Deterministic, collision-resistant, safe as a SQLite table name.

### 4.6 TTL and invalidation

- Bag-sourced results: `ttl=None` (permanent — immutable snapshot).
- Catalog-sourced results: configurable TTL (default: None, manual
  invalidation).
- `invalidate_cache(source="catalog")` drops all catalog-sourced caches.
- `invalidate_cache(cache_key=key)` drops one.
- Expired entries are lazily cleaned on `list_cached_results()`.

### 4.7 MCP contract

Phase 3 (separate spec) will make MCP's tools delegate to these workspace
methods:

- `preview_table` → `workspace.cached_table_read(...).query(limit=...)`
- `list_cached_results` → `workspace.list_cached_results()`
- `query_cached_result` → `workspace.get_cached_result(key).query(...)`
- `invalidate_cache` → `workspace.invalidate_cache(...)`

The API shapes above are designed to match this contract.

## 5. Deletions, migration, testing, and rollout

### 5.1 Code deleted

| File | Removal |
|------|---------|
| `dataset/dataset.py` | `_denormalize_datapath` (~190 lines) |
| `dataset/dataset_bag.py` | `_denormalize` (~100 lines) |
| `core/working_data.py` | Entire module |
| `core/base.py` | `working_data` property, `_LegacyWorkingDataView` refs |
| `local_db/workspace.py` | `_LegacyWorkingDataView`, `legacy_working_data_view()`, `_RESERVED_TABLES` |
| `tests/test_working_data.py` | Entire file |

### 5.2 API replacements

| Old API | New API |
|---------|---------|
| `ml.working_data` | `ml.workspace` (returns `Workspace`) |
| `ml.working_data.cache_table(name, df)` | `workspace.cached_table_read(table)` |
| `ml.working_data.read_table(name)` | `workspace.get_cached_result(key).to_dataframe()` |
| `ml.working_data.query(sql)` | `workspace.get_cached_result(key).query(...)` |
| `ml.working_data.clear()` | `workspace.invalidate_cache()` |
| `ml.cache_table(table_name)` | `workspace.cached_table_read(table_name)` |
| `dataset.cache_denormalized(tables)` | `workspace.cache_denormalized(...)` |
| `dataset.denormalize_as_dataframe(tables)` | Same name, delegates to unified denormalizer |
| `bag.denormalize_as_dataframe(tables)` | Same name, delegates to unified denormalizer |

### 5.3 Migration

**Working DB layout:** Phase 1's `working.db` → Phase 2's `working/`
directory. No migration code — it's a cache. First access creates the new
layout.

**Public API:** `ml.working_data` disappears (breaking change). Release
notes document the migration path. `denormalize_as_dataframe`,
`denormalize_as_dict`, `denormalize_columns` keep their signatures on
`Dataset` and `DatasetBag`.

### 5.4 Testing

**New unit tests (no catalog):**

- `tests/local_db/test_denormalize.py` — unified denormalizer against
  canned bag fixtures. Parameterized by source: `"slice"` (rows in attached
  slice DB) and `"catalog"` (rows in working DB via fake `PagedFetcher`).
  Covers: multi-hop FK chains, LEFT JOIN for nullable FKs, composite FKs,
  multi-schema (isa + deriva-ml), ambiguous-path detection, empty dataset.

- `tests/local_db/test_result_cache.py` — `CachedResult` lifecycle:
  store → query with sort/filter/pagination → invalidate → re-store. TTL
  expiry. `list_cached_results`. Cache key determinism.

**Parameterized regression tests:**

- `tests/dataset/test_denormalize.py` — existing 1487-line test file
  rewritten to use the unified path. Parameterized with
  `source=("catalog", "bag")` where applicable. Same assertions, different
  backend.

**Integration tests (live catalog):**

- `tests/local_db/test_denormalize_live.py` — full pipeline via
  `PagedFetcher` → local SQLite → SQL join → DataFrame. Requires
  `DERIVA_HOST`.

**Deleted:** `tests/test_working_data.py`.

### 5.5 Implementation order

1. Working DB schema lifecycle (convert `working.db` → `working/` directory,
   lazy `LocalSchema` on workspace).
2. Unified denormalizer (`local_db/denormalize.py`).
3. Cached tabular-read API (`local_db/result_cache.py` + workspace methods).
4. Rewire `Dataset.denormalize_*` and `DatasetBag.denormalize_*` as
   delegates.
5. Add `ml.workspace` property, rewire `cache_table` /
   `cache_feature_values`.
6. Delete old code (`_LegacyWorkingDataView`, `WorkingDataCache`,
   `_denormalize_datapath`, `_denormalize`).
7. Test migration: update `test_denormalize.py` to parameterized form, add
   new unit tests.

## 6. Open questions

None blocking. Items for future specs:

- Upload-back from local DB (deferred).
- Cross-workspace slice attachment (attaching a slice from catalog A while
  connected to catalog B) — deferred, requires schema-collision handling.
- Phase 3 MCP migration details (separate spec in `deriva-mcp` repo).
