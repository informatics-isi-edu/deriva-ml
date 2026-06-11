# Cache & Storage Introspection API — Design

**Date:** 2026-06-11
**Status:** Approved in brainstorming; spec for implementation.
**Subproject:** `deriva-ml`

## 1. Problem statement

DerivaML keeps three kinds of local storage, and the existing
storage-management API can only *fully* answer questions about one of
them:

| Species | On-disk location | Inspect | Delete |
|---|---|---|---|
| Cached dataset **bags** | `{cache_dir}/bags/` + `{cache_dir}/index.sqlite` (`deriva.bag.cache_index.BagCacheIndex`) | ✗ none (only per-RID `bag_info`, which requires already knowing the dataset) | ✗ only all-or-nothing `clear_cache()` |
| Cached **assets** | `{cache_dir}/assets/{rid}_{md5}/` (written by `Execution.download_asset(use_cache=True)`) | ✗ none | ✗ only via `clear_cache()` |
| **Execution working dirs** | `{working_dir}/…/{execution_rid}/` | ✓ `list_execution_dirs()` | ✓ `clean_execution_dirs()` |

There is no way to ask "what bags are currently in my cache?" or
"which assets have I cached?" without dropping to the deriva-py
`BagCacheIndex` or shell `ls`. Per-item deletion does not exist at
all.

**Latent coherence bug found during design:** `clear_cache()`
(`src/deriva_ml/core/base.py:1410`) walks `cache_dir` with blind
`rmtree`/`unlink`. With `older_than_days` set, it can remove bag
directories while the SQLite index still lists them (or remove
`index.sqlite` while bags remain). The index outlives its referents —
exactly the hazard `BagCacheIndex.purge()`'s docstring warns against.
The status detection in `bag_info` then reports stale answers.

## 2. Goals

1. **Inspect** every storage species directly from the `DerivaML`
   API: list cached bags (with dataset RID, version, status, size),
   list cached assets, and a storage summary that breaks out all
   three species.
2. **Delete** individual cached items: one dataset's bags (optionally
   one version), one cached asset.
3. **Fix `clear_cache()`** to delete *through the index* so the index
   and the disk can never disagree.
4. Typed, self-documenting return values (Pydantic records, per the
   repo's class-idiom rule).

## 3. Non-goals

- MCP-layer exposure (`deriva-ml-mcp` is a separate repo; follow-up).
- Server-side / Hatrac storage management.
- Automatic cache-eviction policy (LRU, quotas). The primitives this
  spec adds are what such a policy would later be built on.
- Retyping the six existing storage commands' dict returns
  (back-compat; they keep their shapes).

## 4. Design

### 4.1 New module: `src/deriva_ml/core/storage.py`

Owns the records and the logic. `DerivaML` gains thin delegating
methods so the public surface stays one flat, discoverable family
alongside `clear_cache` / `get_storage_summary` / etc. The module is
independently unit-testable with nothing but a `tmp_path` (it builds
on `BagCacheIndex`, which is plain SQLite — no catalog needed).

### 4.2 Records (Pydantic `BaseModel`)

```python
class CachedBag(BaseModel):
    dataset_rid: str            # from bag_anchor_rids (table='Dataset')
    version: str | None         # from anchor_summary['version'] if present
    checksum: str               # bag identity in the index
    status: CacheStatus         # reuses dataset.bag_cache.CacheStatus
    built_at: datetime
    size_bytes: int | None      # index's size_bytes (may be None for old rows)
    path: Path                  # BagCacheIndex.bag_dir_for(checksum)

class CachedAsset(BaseModel):
    rid: str                    # parsed from "{rid}_{md5}" dir name
    md5: str
    file_count: int
    size_bytes: int
    modified: datetime          # dir mtime
    path: Path
```

(No `ExecutionDirInfo` record: `list_execution_dirs()` keeps its
existing tested dict shape — YAGNI; revisit only if a caller needs
the typed form.)

Records live in `core/storage.py` and are re-exported from
`deriva_ml` top level.

### 4.3 Inspection API (new `DerivaML` methods)

- `list_cached_bags() -> list[CachedBag]`
  Joins `BagCacheIndex.list_bags()` with the `bag_anchor_rids`
  reverse index (one bag may anchor several RIDs → one `CachedBag`
  per (bag, Dataset-anchor) pair) and `BagCache`'s status detection.
  Most-recently-built first. Empty list when no index exists.

- `list_cached_assets() -> list[CachedAsset]`
  Walks `{cache_dir}/assets/`, parsing the `{rid}_{md5}` naming
  convention. Entries that don't match the convention are skipped
  with a debug log (not an error — the directory is ours but be
  tolerant). Empty list when the directory doesn't exist.

- `get_storage_summary()` — **extended additively**; existing keys
  (`working_dir`, `cache_dir`, `cache_size_mb`, `cache_file_count`,
  `execution_dir_count`, `execution_size_mb`, `total_size_mb`)
  unchanged. New keys: `bag_count`, `bag_size_mb`, `asset_count`,
  `asset_size_mb`.

### 4.4 Deletion API (new `DerivaML` methods)

- `delete_cached_bag(dataset_rid: str, version: str | None = None) -> dict[str, int]`
  Resolves RID → checksums via
  `BagCacheIndex.find_bags_for_rid(table="Dataset", rid=...)`;
  filters to `version` when given (via each bag's
  `anchor_summary['version']`); `purge()`s each match (index row +
  on-disk dir together — no orphan window). Returns
  `{"bags_removed": n, "bytes_freed": n}`. RID not in cache → zeros,
  not an error (idempotent delete).

- `delete_cached_asset(rid: str, md5: str | None = None) -> dict[str, int]`
  Removes `assets/{rid}_{md5}` (or all `assets/{rid}_*` when `md5`
  is None). Returns `{"assets_removed": n, "bytes_freed": n}`.
  Missing → zeros.

Both take RIDs because that is how users identify things everywhere
else in deriva-ml; checksums stay an internal detail surfaced only in
the `CachedBag` record.

### 4.5 `clear_cache()` coherence fix

Rewritten internally (signature and dict return unchanged):

1. Bags: iterate `BagCacheIndex.list_bags()`; for each bag whose
   `built_at` is older than the cutoff (or always, when
   `older_than_days is None`), call `purge(checksum)` — index row
   and directory leave together.
2. Assets: age-filter `assets/*` entries by mtime, `rmtree` matches.
3. Stray top-level entries (anything that is not `bags/`,
   `assets/`, or `index.sqlite`) keep the old mtime-based behavior.
4. `index.sqlite` itself is never deleted while any bag rows remain.

Stats keys unchanged (`files_removed`, `dirs_removed`,
`bytes_freed`, `errors`); bag purges count as `dirs_removed`.

### 4.6 Error handling

- All listing functions return empty lists on missing
  directories/index — inspection never raises for "nothing there".
- Deletion functions are idempotent (missing target → zero counts).
- OS-level failures (permission, I/O) are caught per-entry, logged
  via the module logger, and counted in `errors` where the return
  shape has one — matching the existing `clear_cache` convention.

## 5. Testing

Unit tests (no catalog), `tests/core/test_storage_management.py`
pattern — a fake harness object carrying `cache_dir`/`working_dir`
and the real functions bound to it:

- `list_cached_bags`: empty cache → `[]`; build a real
  `BagCacheIndex` in `tmp_path`, `record()` two synthetic bags (one
  with two Dataset anchors), assert RID/version/status/order.
- `list_cached_assets`: empty → `[]`; create `{rid}_{md5}` dirs with
  files, assert parse + sizes; non-conforming dir skipped.
- `delete_cached_bag`: by RID (all versions), by RID+version, RID
  not cached → zeros; **assert index row AND directory are both gone**.
- `delete_cached_asset`: with/without md5; missing → zeros.
- `clear_cache` coherence: record bags, age-filter, assert the index
  never references a removed directory (the regression test for the
  bug in §1).
- `get_storage_summary`: new keys present and consistent with the
  listings; old keys unchanged.

Integration (live catalog, `tests/execution/test_storage.py`):
`ml.cache_dataset(...)` then `list_cached_bags()` shows it with the
right RID/status; `delete_cached_bag` removes it and `bag_info`
reports `not_cached`.

Doctest examples on every public method (`# doctest: +SKIP` for the
catalog-dependent ones, per repo convention).

## 6. Documentation

- Docstrings (Google style, runnable examples) on all new methods.
- `docs/user-guide/` storage/caching page updated with a "What's in
  my cache?" section showing the three listings and the deletion
  calls.
