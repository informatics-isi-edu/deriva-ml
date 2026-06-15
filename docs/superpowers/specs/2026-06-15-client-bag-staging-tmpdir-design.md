# Client-side bag staging: move off `working_dir`, wrap in a TemporaryDirectory

**Date:** 2026-06-15
**Status:** Design — approved approach (Option B, structural)
**Issue origin:** Observed in an eye-ai run — the client-side bag build stages a
multi-GB zip under `~/.deriva-ml/.../client_export/` (the `working_dir`),
spilling onto a disk the user did not point `cache_dir` at, and the staging dir
is only cleaned up on the success path.

## Problem

`create_dataset_minid`'s client arm (`use_minid=False`) builds the bag zip into:

```
{working_dir}/client_export/{spec_hash[:8]}/Dataset_{rid}/Dataset_{rid}.zip
```

([`bag_download.py` `create_dataset_minid`](../../../src/deriva_ml/dataset/bag_download.py)).
It returns a `file://` URI pointing at that zip. A *separate* top-level call,
`download_dataset_minid`, later opens that URI, extracts the bag into the
checksum-keyed cache under `cache_dir`, and **then** removes the `client_export`
dir by string-matching `"client_export"` in the archive's parent path.

Two concrete defects plus one structural smell:

1. **Wrong directory.** The staging zip is download-staging on the way to the
   *cache*. It belongs under `cache_dir` (where the user controls disk), not
   `working_dir`. Setting `cache_dir=/data` does not redirect it.
2. **Cleanup only on the happy path.** The `rmtree` is a bare statement near the
   end of `download_dataset_minid`, reached only if extract + validate succeed.
   A crash/exception between build and cleanup orphans a multi-GB zip in
   `working_dir`. No `try/finally` guarantees removal.
3. **Fragile cross-function coupling.** The producer (`create_dataset_minid`)
   and the cleanup (`download_dataset_minid`) are different functions coupled by
   a magic substring (`"client_export"`). The zip must outlive the producer
   *only because* extraction happens in the other function — which is exactly
   why the existing `with TemporaryDirectory()` in `create_dataset_minid` could
   not be used to wrap it.

## Why a TemporaryDirectory is the clean fix

`create_dataset_minid` is *already* wrapped in `with TemporaryDirectory() as
tmp_dir:` — the MINID arm uses it for its spec file and export output. The
client arm deliberately broke out of it (documented in a code comment) because
the returned `file://` zip must survive the `with` block for the *other*
function to extract it.

Remove that reason and the TemporaryDirectory becomes usable: if the client arm
**extracts into the cache itself**, inside one `with
TemporaryDirectory(dir=cache_dir)`, then:

- the OS/context-manager guarantees cleanup on **every** exit path (success,
  exception, kill);
- nothing lands in `working_dir`;
- the `file://` Tier-3 branch in `download_dataset_minid` and its
  string-matched cleanup both disappear;
- staging sits on the same filesystem as the final cache, so the
  staging→cache move is a cheap same-device `rename`, not a cross-device copy.

## Reachability finding that simplifies the refactor

Auditing every `location`/`bag_url` producer in current code:

- **S3/MINID arm** sets `minid_page_url` and `return`s it directly from
  `create_dataset_minid` — it never flows into `download_dataset_minid` as an
  archive to extract (the S3 download path in `download_dataset_minid` fetches
  from the S3 URL stored on the version record, a different route).
- **Tier-1 hit** sets `location` to a `file://` URI of an *already-extracted*
  cache directory — `download_dataset_minid` returns it immediately at the top
  (`bag_dir.exists()` guard) without re-extracting.
- **Client arm** sets `location` to a `file://` URI of the `client_export`
  zip — the only `file://` *archive* that `download_dataset_minid` actually
  extracts.
- The **legacy `else` branch** in `download_dataset_minid`
  (`DerivaExport.retrieve_file` for a non-`file://`, non-S3 URL) has **no live
  producer** in current code. It is dead for the client/MINID flows.

So the only consumer of the client arm's `file://`-zip-extraction is the
Tier-3 path in `get_dataset_minid` (via `download_dataset_minid`). Collapsing
extraction into the client arm touches exactly that one path.

## Design (Option B)

### New shape of the client arm

`create_dataset_minid`'s `use_minid=False` branch becomes self-contained:

```text
with TemporaryDirectory(dir=cache_dir) as staging:
    zip_path = builder.build_bag(dataset, output_dir=staging, timeout=timeout)
    bag_dir  = _extract_and_cache(dataset, zip_path, cache_suffix)   # extract → validate → atomic move into cache → index.record
return bag_dir            # the FINAL cache path, not a file:// URI
```

- The build + extract + cache-population all happen inside the one
  `TemporaryDirectory(dir=cache_dir)`. On exit the staging zip and any unpacked
  intermediate are gone, guaranteed.
- The function returns the **final cache directory path**
  (`{cache_root}/bags/{cache_suffix}/Dataset_{rid}`), not a `file://` URI.

### Extraction helper

The extract→validate→atomic-move→index.record logic currently living in
`download_dataset_minid` (the staging-dir dance + `index.record`) is factored
into a shared free function, e.g. `_extract_archive_to_cache(dataset, archive_path,
checksum) -> Path`. Both the S3 arm (in `download_dataset_minid`) and the new
client arm call it. This keeps the atomic-cache-population semantics identical
and in one place.

### `download_dataset_minid` after the change

- **Tier-1 already-extracted** branch: unchanged (returns early).
- **S3 (`use_minid=True`)** branch: unchanged behavior, but its inline
  extract/validate/move/record body now calls `_extract_archive_to_cache`.
- **`file://` archive branch**: removed — the client arm no longer returns a
  `file://` zip to extract.
- **Legacy `else` (`DerivaExport.retrieve_file`)**: removed (no live producer).
  Per the workspace "no backwards-compat shims / delete unused" rule.
- The `client_export` string-matched cleanup block: removed.

### `get_dataset_minid` Tier-3 wiring

Tier 3 currently calls `create_dataset_minid` → gets a `file://` `bag_url` →
wraps it in a `DatasetMinid(location=bag_url)` → returns it → caller
(`Dataset.download_dataset_bag`) calls `download_dataset_minid` to extract.

After the change, the client arm already extracted into the cache. Tier 3 wraps
the returned **cache path** as the `DatasetMinid.location` (a `file://` URI of
the already-extracted dir, exactly like Tier-1 produces), so the downstream
`download_dataset_minid(minid)` call hits its top-of-function `bag_dir.exists()`
guard and returns immediately. **No downstream signature changes** —
`Dataset.download_dataset_bag` keeps calling `download_dataset_minid`; it just
becomes a cache-hit no-op for the freshly-built client bag.

This preserves the public flow (`get_dataset_minid` → `materialize_dataset_bag`
/ `download_dataset_minid`) while moving the extraction earlier.

## What does NOT change

- The cache layout: `{cache_root}/bags/{checksum}/Dataset_{rid}/`.
- The checksum / `cache_suffix` (`{spec_hash[:16]}_{snapshot}`) — same key, so
  existing cached bags still hit Tier-1.
- `BagCacheIndex.record(...)` call — same anchors/summary.
- `materialize_dataset_bag` — still materializes the cache bag in place.
- The MINID/S3 arm's externally-visible behavior.
- Public method signatures (`download_dataset_bag`, `get_dataset_minid`,
  `materialize_dataset_bag`).

## Testing

Existing tests to keep green (mock-level, no catalog):
`tests/dataset/test_get_dataset_minid_helpers.py` (Tier-2/Tier-3 dispatch),
`tests/dataset/test_bag_materialize.py`, `tests/dataset/test_multi_anchor_bag_cache.py`.

New coverage:

1. **Staging lives under `cache_dir`, never `working_dir`** — patch
   `build_bag` to assert its `output_dir` is under `cache_dir` and not under
   `working_dir`; assert no `client_export` dir is created under `working_dir`.
2. **Cleanup on success** — after a client-arm build, the
   `TemporaryDirectory(dir=cache_dir)` staging is gone; only the final cache
   dir remains.
3. **Cleanup on failure** — make extract/validate raise after a successful
   `build_bag`; assert the staging dir is removed anyway (the
   TemporaryDirectory guarantee) and nothing is orphaned under `cache_dir` or
   `working_dir`.
4. **Return contract** — client arm returns the final cache path; the
   `DatasetMinid` built from it resolves to an existing extracted bag so the
   downstream `download_dataset_minid` is a no-op cache hit.
5. **Live round-trip** (catalog) — `cache(..., use_minid=False)` on a demo
   dataset still produces a fully-formed, loadable bag in `cache_dir`; assert
   `working_dir` gains no `client_export`.

## Risks

- **Merging build + extract responsibilities.** Mitigated by factoring the
  extraction into a shared `_extract_archive_to_cache` so the atomic-move +
  index-record semantics are identical to today and exercised by both arms.
- **Dead-branch removal.** The legacy `DerivaExport.retrieve_file` branch is
  removed; confirmed no live producer. If a hidden caller exists, it would
  surface as a test failure in the dispatch tests — acceptable, and correct per
  the delete-unused rule.
