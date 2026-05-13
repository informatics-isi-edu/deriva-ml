# Bag-client cutover (Steps 11–12 of the Phase 2 audit)

Date: 2026-05-13
Status: Design — pending implementation

## Context

Phase 2 of the deriva-ml audit
(`deriva-ml-audit-2026-05-phase2-dataset.md` §5.1 + §5.2) flagged two
deeper structural changes deferred from the Phase 2 cleanup and polish
PRs (#109, #110):

- **Step 11** — Delete `Dataset._create_dataset_bag_client` (231 LoC of
  hand-rolled BDBag construction that duplicates
  `deriva.bag.catalog_builder.CatalogBagBuilder.build()`).
- **Step 12** — Cut the dataset-download flow over to the
  content-addressed cache layout (`{cache_root}/bags/{checksum}/`)
  managed by `deriva.bag.cache_index.BagCacheIndex`, and drop the
  legacy `{cache_root}/{rid}_{checksum}_{snapshot}/` glob-fallback in
  `BagCache.cache_status` and `Dataset._get_dataset_minid`.

Together they are the final pieces of ADR-0006's bag-oriented data
movement: the deriva-ml package stops carrying its own bag-construction
and cache-layout knowledge, and instead drives the deriva-py
`CatalogBagBuilder` + `BagCacheIndex` surface end-to-end.

The two are coupled — Step 11 changes how bags get written, and Step 12
changes how the cache layer looks them up — so they should ship in the
same PR.

## Today's state

### Tier 1–3 flow in `Dataset._get_dataset_minid`

```
_get_dataset_minid(version, create, use_minid, …)
├── compute spec via DatasetBagBuilder.generate_dataset_download_spec
├── spec_hash = _hash_spec(spec)
├── cache_dir_name = f"{rid}_{spec_hash[:16]}_{snapshot}"          ← LEGACY KEY
├── Tier 1: filesystem stat on cache_dir / Dataset_{rid}           ← LEGACY LAYOUT
├── Tier 2 (use_minid=True): MINID lookup + S3 download
└── Tier 3 (use_minid=False): _create_dataset_minid(spec=...)
        └── _create_dataset_bag_client(version, spec)              ← 231 LoC TO DELETE
              ├── connect snapshot ErmrestCatalog (custom timeout via session_config)
              ├── make_bag(idempotent=True) on output_dir
              ├── walk query_processors:
              │     - env → POST /entity ... → envars dict
              │     - json → catalog.get(query_path).json() → file
              │     - csv → catalog.get_as_file(... paged=...)
              │     - fetch → catalog.get(query_path).json() →
              │               remote-file-manifest.json
              ├── prune empty directories
              ├── if failed_queries: raise DerivaMLException
              ├── make_bag(update=True, remote_file_manifest=...)
              ├── archive_bag(... "zip")
              └── return file:// URI
```

The legacy cache layout (`{rid}_{checksum}_{snapshot}`) was written
**by `_create_dataset_bag_client` itself** (via `bdb.make_bag` against
the bag directory). `BagCache.cache_status` reads back from the same
layout via `_cache_dir.glob(f"{dataset_rid}_*")`.

### Where `BagCacheIndex` already plays

PR #109's Phase-2 changes introduced `BagCache` as a wrapper around
`deriva.bag.cache_index.BagCacheIndex`. `BagCache.cache_status`
**already** consults the index first, and only falls back to the
legacy glob when the index is empty. So Step 12 is really about (a)
making the *download* path write into the index layout, and (b)
deleting the glob fallback once nothing produces legacy directories
anymore.

### Where the spec hash lives

`_get_dataset_minid` hashes the dict returned by
`DatasetBagBuilder.generate_dataset_download_spec(self)`. The hash
captures the FK traversal plan; `Dataset_Version.Minid_Spec_Hash`
stores the same value on the catalog side. The cutover **must
preserve the spec-hash invariant**: two callers asking for the same
version of the same dataset against the same schema must still
compute the same `spec_hash`, or:

- Tier 1 stops finding cached bags (false cache miss).
- Tier 2 starts re-minting MINIDs on every download (false spec drift).

Concretely, the inputs to `_hash_spec` are the entire spec dict —
`bag.bag_name`, `bag.bag_algorithms`, the env preamble's
`{output_path, query_keys, query_path}` triples, every
`csv`/`json`/`fetch` processor, and the catalog's `host` + `catalog_id`
keys. **`CatalogBagBuilder.get_export_spec()` is the body of that
dict.** `DatasetBagBuilder.generate_dataset_download_spec` overlays the
env preamble and the `Dataset_{RID}` bag-name on top. Neither changes
in this cutover — the spec is unchanged because the underlying
builder is unchanged.

## Design

### Step 11 — bag-client cutover

#### What changes

1. Add a new method `DatasetBagBuilder.build_bag(dataset, output_dir, *, timeout=None) -> Path`.

   - Constructs a `CatalogBagBuilder` with the dataset's anchors and
     policy (same as the existing `_catalog_bag_builder(dataset)`),
     scoped to the snapshot catalog the `DatasetBagBuilder` was
     initialized with.
   - Names the bag directory `Dataset_{rid}` inside `output_dir` (so
     `CatalogBagBuilder` picks that name from `output_dir.name`).
   - Calls `.build()` and returns the resulting `Path`.
   - Applies the optional `(connect, read)` timeout via deriva-py's
     `_session_config` private attribute on the catalog, scoped to
     the call only (`try/finally` restores the prior config).

2. In `Dataset._create_dataset_minid(use_minid=False, spec=..., …)`:

   - Replace the call to `self._create_dataset_bag_client(version, spec, timeout=timeout)`
     with: build a `DatasetBagBuilder` on the snapshot catalog, call
     `build_bag(self, output_dir=tmp_dir, timeout=timeout)`, archive
     the returned bag directory to `.zip` via `bdb.archive_bag`, and
     return `Path(archive).as_uri()`.
   - The pre-computed `spec` parameter becomes vestigial for the
     non-MINID path — `CatalogBagBuilder` recomputes its own spec
     from the same anchors/policy. We keep the parameter on the
     signature because `_get_dataset_minid` passes it through for the
     MINID path (which still drives `DerivaExport`).
   - Net deletion: `_create_dataset_bag_client` (lines 3008–3239).

3. The `fetch.txt` byte-format equivalence question (raised in the
   precursor notes) resolves as follows: the old hand-rolled path
   writes `remote-file-manifest.json` → `bdb.make_bag(remote_file_manifest=…)`
   → bdbag-generated `fetch.txt`. The new path writes
   `fetch.txt` directly via `GenericDownloader`'s `fetch` processor
   handling. Both end states are bdbag-conformant fetch.txt files;
   the per-line column order is identical (URL, length, filename,
   optional md5). Drift would surface as bag-validation failures in
   the existing `test_dataset_caching.py::test_materialized_to_incomplete`
   sequence, so it's covered by CI.

4. The `failed_queries` retry/error message in the old client
   (lines 3193–3206) was bespoke to the deep-FK-join timeout cases on
   `localhost`. `CatalogBagBuilder` propagates the underlying
   `DerivaDownloadError`. We map that to `DerivaMLException` with
   the same actionable advice in `Dataset._create_dataset_minid`'s
   try/except wrapper — preserving the existing error contract.

#### What stays

- The MINID server-side export path (Tier 2). It still drives
  `DerivaExport(...)` with the same pre-computed spec.
- `Dataset_Version.Minid_Spec_Hash` semantics and the spec-hash
  comparison in `_get_dataset_minid`.
- The two-stage `_get_dataset_minid → _create_dataset_minid` split.
  The cutover is internal to `_create_dataset_minid`'s `else` arm.

### Step 12 — cache-layout cutover

#### What changes

1. **Bag-writing path: write into the index layout.** After
   `archive_bag(...)`, `Dataset._create_dataset_minid(use_minid=False)`
   records the bag in the index:

   ```python
   from deriva.bag.cache_index import BagCacheIndex
   index = BagCacheIndex(self._ml_instance.cache_dir)
   index.record(
       checksum=cache_suffix,   # spec_hash[:16] + "_" + snapshot
       profile_id=BAG_PROFILE_ID,
       anchors=[("Dataset", self.dataset_rid)],
       anchor_summary={"version": str(version), "snapshot": snapshot},
       size_bytes=archive_path.stat().st_size,
   )
   ```

   The index entry's `checksum` reuses the existing
   `{spec_hash[:16]}_{snapshot}` string so:
   - Tier-1 lookup keeps working with the same key.
   - The MINID path's `version_record.spec_hash == spec_hash`
     comparison is unaffected (it compares raw spec hashes, not
     index checksums).

2. **Bag-reading path: Tier 1 consults the index, not the glob.**
   Replace the `cached_dir = self._ml_instance.cache_dir / cache_dir_name`
   block in `_get_dataset_minid` (lines 3348–3373) with:

   ```python
   from deriva.bag.cache_index import BagCacheIndex
   index = BagCacheIndex(self._ml_instance.cache_dir)
   for checksum in index.find_bags_for_rid(table="Dataset", rid=self.dataset_rid):
       if checksum == cache_suffix:                           # exact spec+snapshot match
           bag_dir = index.bag_dir_for(checksum) / f"Dataset_{self.dataset_rid}"
           if bag_dir.exists():
               return DatasetMinid(...)
   ```

   The "spec_hash AND snapshot must both match for a cache hit"
   semantic moves into the checksum string itself — we only return a
   hit when `checksum == cache_suffix`, which by construction equals
   the current request's `spec_hash[:16]_snapshot`. Bags built at
   different snapshots have different checksums; bags built against
   different schemas have different checksums; no stale-bag-after-schema-change
   regressions.

3. **Drop the legacy glob fallback.** Once nothing writes the legacy
   `{rid}_{checksum}_{snapshot}` layout (Step 11 + Step 12 together),
   the glob-based `legacy_dirs` branch in
   `BagCache.cache_status`/`_determine_legacy_status` becomes dead.
   Delete the branch and the helper (and the
   `validated_check.txt`-marker logic, which only the legacy layout
   ever wrote).

#### What stays

- `BagCacheIndex` itself — already in deriva-py.
- `BagCache.cache_status`'s public API and return-dict shape. The
  internal fallback goes away but the user-visible contract is
  unchanged.
- `versions_cached` in the return dict — still populated, just from
  index entries only.

### Invariants the cutover must preserve

| # | Invariant | How preserved |
|---|---|---|
| 1 | `spec_hash` for identical request → identical output | Spec is generated by the same `DatasetBagBuilder.generate_dataset_download_spec`. |
| 2 | Tier-1 hit on identical re-request | Cache checksum = `spec_hash[:16]_snapshot`, recorded on every write, queried on every read. |
| 3 | Schema-drift detection (new column/table → no stale cache hit) | `spec_hash` changes when the bag-walk changes; the new checksum doesn't match the old one, so Tier 1 falls through to Tier 3 and rebuilds. |
| 4 | Cross-process index access (concurrent downloads) | `BagCacheIndex` uses `create_wal_engine` (WAL mode SQLite) — already cross-process safe in deriva-py. |
| 5 | Bag-validation failures still surface | `bdb.validate_bag_structure` is called from `BagCache._is_fully_materialized` either way. |
| 6 | `fetch.txt` byte format unchanged | Both old and new paths use `bdbag` to emit `fetch.txt`; only the intermediate manifest format differs. |
| 7 | Error message on FK-join timeout (`failed_queries` → actionable hint) | Mapped into `_create_dataset_minid`'s `DerivaDownloadError` → `DerivaMLException` wrapper. |
| 8 | `file://` URI shape returned to callers | Unchanged — we still `archive_bag` to `.zip` and call `Path(archive_path).as_uri()`. |

### Migration story for users with existing caches

`BagCache.cache_status` keeps the glob-fallback **only until the next
patch release after this PR lands**. During the transition (one
release window), users with legacy directories still get cache hits
through the BagCache surface — but the *download* path doesn't write
legacy directories anymore. On the second release, the glob fallback
is removed.

Alternative considered: write to both layouts during transition.
Rejected — doubles disk usage per bag for no real win.

A separate `deriva-ml migrate-cache` command would walk legacy
directories and `BagCacheIndex.record(...)` them into the new layout.
Not in scope for this PR — handled in a follow-up if user demand
emerges.

## Tests

| Coverage | Test |
|---|---|
| Tier-1 hit with new layout | `test_dataset_caching.py::test_fresh_download_creates_cache` |
| Tier-1 hit cross-process via index | `test_dataset_caching.py::test_redownload_uses_cache` |
| Cache miss → fresh rebuild | `test_dataset_caching.py::test_different_versions_cached_separately` |
| Bag validation surfaces partial materialization | `test_dataset_caching.py::test_materialized_to_incomplete` |
| `fetch.txt` round-trips through bdbag | `test_download.py` (existing — should pass unchanged) |
| FK-join timeout error message | New test in `test_download.py` covering the wrapper |
| Non-default `cache_dir` honored | `test_dataset_caching.py::test_non_default_cache_directory` |

A new test case `test_legacy_layout_no_longer_written` asserts that a
fresh download does NOT create a `{rid}_{checksum}_{snapshot}/`
directory at the cache root, only `bags/{checksum}/...`. This guards
the Step-12 transition during the one-release deprecation window.

## Scope and risk

- **LoC change**: −231 (delete `_create_dataset_bag_client`),
  +60 (`build_bag` + Tier-1 rewrite), −80 (legacy-glob fallback
  removal). Net ≈ **−250 LoC**.
- **Risk: medium**. Touches the load-bearing download path. Mitigated
  by the eight-invariant table and a comprehensive run of
  `tests/dataset/` against a live catalog before merge.
- **Coordination**: no upstream changes needed — `CatalogBagBuilder`
  and `BagCacheIndex` are both already in deriva-py. Verified against
  the current `deriva.bag` submodule on the `deriva-ml` branch.

## Rollout

1. PR #111: implements Steps 11 + 12 as described here. Drops
   `_create_dataset_bag_client`, rewires Tier 1, writes to the
   index layout, keeps the glob fallback for one release.
2. Patch release (`v1.36.0`): users get the new layout. Caches built
   pre-release still load via the fallback.
3. PR #112: removes the glob fallback. Released as `v1.37.0` minimum
   one release after PR #111.
