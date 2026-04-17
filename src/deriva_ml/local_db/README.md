# `deriva_ml.local_db`

Unified client-side SQLite layer for deriva-ml. Phase 1 delivered the
foundation (paths, helpers, workspace, paged fetcher, manifest store).
Phase 2 added the unified denormalization engine, the cached
tabular-read API, and deleted the legacy working-data cache.

## Modules

- `paths.py` — pure path helpers for the working-directory layout.
- `sqlite_helpers.py` — WAL engine factory, ATTACH/DETACH, schema-version
  runner, busy-timeout pragma.
- `schema.py` — `LocalSchema` adapter over `SchemaBuilder`. Owns the
  SQLAlchemy ORM for all catalog tables in a workspace or slice.
- `workspace.py` — `Workspace`: per-catalog directory-based working DB
  (`main.db` + per-schema `.db` files ATTACH'd into one connection), slice
  attach/detach, manifest store, result-cache methods, legacy-manifest
  import.
- `paged_fetcher.py` — transport-agnostic `PagedFetcher` primitive (keyset
  pagination, RID-set batching, URL-length guard, cardinality heuristic).
- `paged_fetcher_ermrest.py` — `ErmrestPagedClient` adapter to deriva-py.
- `manifest_store.py` — SQLite persistence for `AssetManifest`
  (asset + feature entries with per-mutation commits for crash safety).
- `result_cache.py` — `ResultCache`, `CachedResult`, `CachedResultMeta`,
  `QueryResult` for cached tabular reads with TTL, re-query, and
  sort/filter/pagination.
- `denormalize.py` — unified `denormalize()` function. Replaces both the
  old `Dataset._denormalize_datapath` (catalog-side) and
  `DatasetBag._denormalize` (bag-side) with a single implementation.
  Uses `_prepare_wide_table` for join planning and runs SQLAlchemy JOINs
  against local SQLite. Supports three sources: `"catalog"` (fetches rows
  via `PagedFetcher`), `"slice"` (rows from an attached slice), `"local"`
  (caller has already populated rows; used by tests).

## Layout on disk

Phase 2 uses a directory-based working DB so that multi-schema ATTACH works
identically to downloaded bags:

```
{working_dir}/
  catalogs/
    {host}__{cat}/
      working/                  # was a single working.db in Phase 1
        main.db                 # ManifestStore, cached_results_registry, schema_meta
        isa.db                  # domain-schema tables (ATTACH'd as 'isa')
        deriva-ml.db            # ML-schema tables (ATTACH'd as 'deriva-ml')
      slices/
        {slice_id}/
          main.db               # slice-specific metadata (ATTACH'd as 'slice_main')
          isa.db                # ATTACH'd as 'slice_isa'
          deriva-ml.db          # ATTACH'd as 'slice_deriva-ml'
          assets/               # materialized asset files (optional)
```

All schema files are opened on the same SQLAlchemy engine via ATTACH. The
ORM relationships built by `SchemaBuilder` traverse cross-schema FKs so
JOINs span all attached files.

## How the pieces fit together

```
DerivaML.workspace  ─>  Workspace
                          │
                          ├─ local_schema ─> LocalSchema (engine + ORM)
                          ├─ manifest_store() ─> ManifestStore
                          ├─ result_cache ─> ResultCache
                          ├─ attach_slice() ─> multi-schema ATTACH
                          ├─ cached_table_read() / cache_denormalized()
                          └─ list_cached_results() / invalidate_cache()

Dataset.denormalize_as_dataframe  ─>  denormalize(source="catalog",
                                                  paged_client=ErmrestPagedClient)
DatasetBag.denormalize_as_dataframe ─> denormalize(source="slice")
```

## Source modes for `denormalize()`

- **`"local"`** (default) — caller has already populated the engine's tables.
  Used by unit tests with pre-populated fixtures.
- **`"catalog"`** — fetches rows from a live ERMrest catalog via a
  `PagedClient` (typically `ErmrestPagedClient`). Walks the join plan,
  issues RID-batched fetches per table, and commits rows into the working
  DB before running the SQL join. This is the production path for
  `Dataset.denormalize_as_dataframe`.
- **`"slice"`** — rows are already visible via an attached slice database.
  Used by `DatasetBag.denormalize_as_dataframe` where the bag's per-schema
  files are the slice.

## Further reading

- Phase 1 design spec: `docs/superpowers/specs/2026-04-15-unified-local-db-design.md`
- Phase 2 design spec: `docs/superpowers/specs/2026-04-15-unified-local-db-phase2-design.md`
- Phase 1 plan: `docs/superpowers/plans/2026-04-15-unified-local-db-phase1.md`
- Phase 2 plan: `docs/superpowers/plans/2026-04-16-unified-local-db-phase2.md`
