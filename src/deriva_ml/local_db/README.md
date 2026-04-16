# `deriva_ml.local_db`

Unified client-side SQLite layer for deriva-ml. Phase 1 delivers the
foundation; Phase 2 will migrate denormalization onto it.

## Modules

- `paths.py` — pure path helpers for the working-directory layout.
- `sqlite_helpers.py` — WAL engine factory, ATTACH/DETACH, schema-version
  runner.
- `schema.py` — `LocalSchema` adapter over `SchemaBuilder`.
- `workspace.py` — `Workspace`: per-catalog `working.db` handle, slice
  attach, legacy working-data view, manifest store, legacy-manifest import.
- `paged_fetcher.py` — transport-agnostic `PagedFetcher` primitive.
- `paged_fetcher_ermrest.py` — `ErmrestPagedClient` adapter to deriva-py.
- `manifest_store.py` — SQLite persistence for `AssetManifest`.

## Layout on disk

```
{working_dir}/
  catalogs/
    {host}__{cat}/
      working.db
      slices/
        {slice_id}/
          slice.db
          assets/
```

## Further reading

- Design spec: `docs/superpowers/specs/2026-04-15-unified-local-db-design.md`
- Phase 1 plan: `docs/superpowers/plans/2026-04-15-unified-local-db-phase1.md`
