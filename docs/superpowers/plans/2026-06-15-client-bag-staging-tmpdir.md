# Client-side bag staging → cache_dir + TemporaryDirectory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the client-side dataset-bag build stage into `cache_dir` inside a `TemporaryDirectory`, extract into the cache itself, and return the final cache path — eliminating the `working_dir/client_export` spill, the success-only cleanup, and the fragile cross-function `file://` hand-off.

**Architecture:** Factor the extract→validate→atomic-move→`index.record` logic out of `download_dataset_minid` into a shared `_extract_archive_to_cache` helper. Rewrite `create_dataset_minid`'s `use_minid=False` arm to build into `TemporaryDirectory(dir=cache_dir)` and call that helper itself, returning the final cache path. Wrap the result as a `DatasetMinid` in `get_dataset_minid` Tier 3 so the downstream `download_dataset_minid` call becomes a cache-hit no-op. Delete the now-dead `file://`-archive and legacy `DerivaExport.retrieve_file` branches.

**Tech Stack:** Python 3.12, `uv`, pytest, `tempfile.TemporaryDirectory`, deriva-py `bdb`/`BagCacheIndex`.

**Spec:** `docs/superpowers/specs/2026-06-15-client-bag-staging-tmpdir-design.md`

---

### Task 1: Extract `_extract_archive_to_cache` helper (pure refactor, no behavior change)

Factor the extract→validate→atomic-move→`index.record` body out of `download_dataset_minid` into a free function. `download_dataset_minid`'s S3 arm calls it. No behavior change yet — this is a safe, separately-verifiable refactor.

**Files:**
- Modify: `src/deriva_ml/dataset/bag_download.py`
- Test: `tests/dataset/test_extract_archive_to_cache.py` (new)

- [ ] **Step 1: Write the failing test**

```python
"""Unit tests for _extract_archive_to_cache (client-bag staging refactor).

The helper owns the atomic cache-population dance: extract a bag archive
into a staging dir, validate, atomically rename into the checksum-keyed
cache location, and record in the BagCacheIndex. Both the S3 arm and the
client arm of the download path call it.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from deriva_ml.dataset.bag_download import _extract_archive_to_cache


def test_extract_archive_to_cache_atomic_move_and_record(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Fake index: bag_dir_for returns a checksum-keyed dir under cache_dir.
    bag_root = cache_dir / "bags" / "deadbeef_2-SNAP"
    index = MagicMock()
    index.bag_dir_for.return_value = bag_root

    # bdb.extract_bag creates the staging dir contents; we simulate it by
    # making the directory it is handed and returning that path.
    def fake_extract(archive, dest):
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "bag-info.txt").write_text("ok")
        return dest

    with (
        patch("deriva_ml.dataset.bag_download.bdb.extract_bag", side_effect=fake_extract),
        patch("deriva_ml.dataset.bag_download.bdb.validate_bag_structure"),
    ):
        result = _extract_archive_to_cache(
            index=index,
            archive_path="/tmp/whatever.zip",
            checksum="deadbeef_2-SNAP",
            dataset_rid="2-ABCD",
            dataset_version="1.0.0",
        )

    # Returns the bag dir inside the cache root.
    assert result == bag_root / "Dataset_2-ABCD"
    # The checksum-keyed cache dir now exists (atomic move happened).
    assert bag_root.exists()
    # Index recorded the bag under the Dataset anchor + version summary.
    index.record.assert_called_once()
    _, kwargs = index.record.call_args
    assert kwargs["checksum"] == "deadbeef_2-SNAP"
    assert kwargs["anchors"] == [("Dataset", "2-ABCD")]
    assert kwargs["anchor_summary"] == {"version": "1.0.0"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_extract_archive_to_cache.py -q`
Expected: FAIL with `ImportError: cannot import name '_extract_archive_to_cache'`.

- [ ] **Step 3: Write the helper and route the S3 arm through it**

Add the free function (near `download_dataset_minid`). Move the staging→validate→rename→record body verbatim into it. Note the helper takes the `BagCacheIndex` and **does not** dispose it (the caller owns the index lifecycle, matching the existing `try/finally index.dispose()` placement).

```python
def _extract_archive_to_cache(
    index: "BagCacheIndex",
    archive_path: str,
    checksum: str,
    dataset_rid: str,
    dataset_version: str,
) -> Path:
    """Extract a bag archive into the checksum-keyed cache, atomically.

    Extracts ``archive_path`` into a staging directory, validates the BDBag
    structure, atomically renames the staging dir to its final cache location
    (``{cache_root}/bags/{checksum}/Dataset_{rid}``), and records the bag in
    ``index`` so the next Tier-1 lookup finds it. The atomic rename prevents
    partial/corrupt cache entries if the process crashes mid-extract.

    The caller owns ``index``'s lifecycle (this function does not dispose it).

    Args:
        index: Open BagCacheIndex rooted at the cache dir.
        archive_path: Path to the bag zip archive to extract.
        checksum: Deterministic cache key (``{spec_hash[:16]}_{snapshot}``)
            or SHA-256 of the archive.
        dataset_rid: Dataset RID, for the cache dir name and index anchor.
        dataset_version: Dataset version string, for the index anchor summary.

    Returns:
        Path to the extracted bag directory inside the cache.

    Example:
        >>> # Illustrative — requires a live index + archive.
        >>> path = _extract_archive_to_cache(idx, "/tmp/b.zip", "ab_S", "2-X", "1.0.0")  # doctest: +SKIP
    """
    bag_root = index.bag_dir_for(checksum)
    bag_dir = bag_root / f"Dataset_{dataset_rid}"

    staging_dir = bag_root.parent / f"{bag_root.name}_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    try:
        extracted_bag_path = bdb.extract_bag(archive_path, staging_dir.as_posix())
        bdb.validate_bag_structure(extracted_bag_path)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    bag_root.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.rename(bag_root)

    index.record(
        checksum=checksum,
        anchors=[("Dataset", dataset_rid)],
        anchor_summary={"version": str(dataset_version)},
    )
    return bag_dir
```

Then in `download_dataset_minid`, replace the inline staging/validate/rename/record block (current lines ~215–243) with a call to the helper. The S3 (`use_minid=True`) and the SHA-256-fallback (`not use_minid and not minid.checksum`) paths both still compute `archive_path` and `minid.checksum` as today; only the extraction body is replaced:

```python
        # (inside download_dataset_minid, after archive_path + minid.checksum resolved)
        bag_dir = _extract_archive_to_cache(
            index=index,
            archive_path=archive_path,
            checksum=minid.checksum,
            dataset_rid=minid.dataset_rid,
            dataset_version=str(minid.dataset_version),
        )
    finally:
        index.dispose()
    return bag_dir
```

Keep the `with TemporaryDirectory() as tmp_dir:` wrapping the archive download for the S3/legacy arms. (The `client_export` cleanup block and the `file://`/legacy branches are removed in Task 3, not here — Task 1 stays behavior-preserving for the S3 arm.)

- [ ] **Step 4: Run the test + existing download tests**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_extract_archive_to_cache.py tests/dataset/test_get_dataset_minid_helpers.py tests/dataset/test_bag_materialize.py -q`
Expected: PASS (new helper test green; existing dispatch/materialize tests unaffected).

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/dataset/bag_download.py tests/dataset/test_extract_archive_to_cache.py
git commit -m "refactor(bag): extract _extract_archive_to_cache helper (no behavior change)"
```

---

### Task 2: Client arm builds into TemporaryDirectory(dir=cache_dir) and extracts itself

Rewrite `create_dataset_minid`'s `use_minid=False` branch so it builds the zip under `TemporaryDirectory(dir=cache_dir)`, extracts into the cache via the Task-1 helper, and returns the **final cache path** instead of a `file://` zip URI.

**Files:**
- Modify: `src/deriva_ml/dataset/bag_download.py` (the `else` arm of `create_dataset_minid`, ~lines 344–390)
- Test: `tests/dataset/test_client_bag_staging.py` (new)

- [ ] **Step 1: Write the failing tests (directory + cleanup-on-failure)**

```python
"""Client-arm bag staging lives under cache_dir, not working_dir, and is
always cleaned up (TemporaryDirectory guarantee), even on extract failure.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deriva_ml.dataset.bag_download import create_dataset_minid


def _fake_dataset(tmp_path):
    ds = MagicMock()
    ds.dataset_rid = "2-ABCD"
    ds._ml_instance.cache_dir = tmp_path / "cache"
    ds._ml_instance.working_dir = tmp_path / "work"
    ds._ml_instance.cache_dir.mkdir()
    ds._ml_instance.working_dir.mkdir()
    ds._ml_instance.s3_bucket = None
    return ds


def test_client_arm_stages_under_cache_dir_not_working_dir(tmp_path):
    ds = _fake_dataset(tmp_path)
    captured = {}

    def fake_build_bag(self, dataset, output_dir, timeout=None):
        captured["output_dir"] = Path(output_dir)
        zip_path = Path(output_dir) / "Dataset_2-ABCD.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path.write_text("zip")
        return zip_path

    with (
        patch("deriva_ml.dataset.bag_download.DatasetBagBuilder.build_bag", new=fake_build_bag),
        patch("deriva_ml.dataset.bag_download._extract_archive_to_cache",
              return_value=ds._ml_instance.cache_dir / "bags" / "x" / "Dataset_2-ABCD"),
        patch("deriva_ml.dataset.bag_download.BagCacheIndex"),
    ):
        result = create_dataset_minid(
            ds, "1.0.0", use_minid=False, spec={"k": "v"}, spec_hash="deadbeefdeadbeef",
        )

    # Staging output_dir is under cache_dir, never working_dir.
    assert ds._ml_instance.cache_dir in captured["output_dir"].parents
    assert ds._ml_instance.working_dir not in captured["output_dir"].parents
    # No client_export dir was created under working_dir.
    assert not (ds._ml_instance.working_dir / "client_export").exists()
    # Returns the final cache path (not a file:// URI string).
    assert isinstance(result, Path)
    assert "bags" in result.parts


def test_client_arm_cleans_staging_on_extract_failure(tmp_path):
    ds = _fake_dataset(tmp_path)
    leaked = []

    def fake_build_bag(self, dataset, output_dir, timeout=None):
        leaked.append(Path(output_dir))
        zip_path = Path(output_dir) / "Dataset_2-ABCD.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path.write_text("zip")
        return zip_path

    with (
        patch("deriva_ml.dataset.bag_download.DatasetBagBuilder.build_bag", new=fake_build_bag),
        patch("deriva_ml.dataset.bag_download._extract_archive_to_cache",
              side_effect=RuntimeError("extract boom")),
        patch("deriva_ml.dataset.bag_download.BagCacheIndex"),
    ):
        with pytest.raises(RuntimeError, match="extract boom"):
            create_dataset_minid(
                ds, "1.0.0", use_minid=False, spec={"k": "v"}, spec_hash="deadbeefdeadbeef",
            )

    # The TemporaryDirectory(dir=cache_dir) staging is gone despite the failure.
    for staging in leaked:
        assert not staging.exists()
    # Nothing orphaned under working_dir.
    assert not (ds._ml_instance.working_dir / "client_export").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_client_bag_staging.py -q`
Expected: FAIL — current code returns a `file://` string and stages under `working_dir/client_export`.

- [ ] **Step 3: Rewrite the client arm**

Replace the `else:` arm of `create_dataset_minid` (the `working_dir/client_export` build + `return zip_path.as_uri()`) with a self-contained build+extract under `TemporaryDirectory(dir=cache_dir)`. The function still returns `str` on the MINID arm (landing-page URL); on the client arm it now returns a `Path` (the cache dir). Update the return annotation to `str | Path` and the docstring.

```python
        else:
            # Client-side bag construction: build into a TemporaryDirectory
            # rooted at cache_dir, then extract straight into the cache. The
            # staging zip never touches working_dir and is removed on every
            # exit path (success or exception) by the context manager — no
            # cross-function file:// hand-off, no manual cleanup.
            from deriva.bag.cache_index import BagCacheIndex

            version_snapshot_catalog = dataset._version_snapshot_catalog(version)
            builder = DatasetBagBuilder(
                ml_instance=version_snapshot_catalog,
                s3_bucket=dataset._ml_instance.s3_bucket,
                use_minid=False,
                exclude_tables=exclude_tables,
            )
            cache_suffix = f"{spec_hash[:16]}_{snapshot}"  # see note below
            cache_dir = Path(dataset._ml_instance.cache_dir)
            with TemporaryDirectory(dir=cache_dir) as staging:
                try:
                    zip_path = builder.build_bag(dataset, output_dir=Path(staging), timeout=timeout)
                except (
                    DerivaDownloadError,
                    DerivaDownloadConfigurationError,
                    DerivaDownloadAuthenticationError,
                    DerivaDownloadAuthorizationError,
                    DerivaDownloadTimeoutError,
                ) as e:
                    raise DerivaMLException(
                        f"Dataset bag export failed: {format_exception(e)}. "
                        "This typically happens when deep multi-table joins "
                        "exceed server query time limits. To fix this, add the "
                        "desired records as direct dataset members using "
                        "add_dataset_members() with the relevant table's RIDs."
                    )
                index = BagCacheIndex(cache_dir)
                try:
                    bag_dir = _extract_archive_to_cache(
                        index=index,
                        archive_path=str(zip_path),
                        checksum=cache_suffix,
                        dataset_rid=dataset.dataset_rid,
                        dataset_version=str(version),
                    )
                finally:
                    index.dispose()
            return bag_dir
```

**Note on `snapshot`/`cache_suffix`:** `create_dataset_minid` currently does not
have `snapshot` in scope on the client arm. Thread it in: the caller
`get_dataset_minid` already computes `cache_suffix = f"{spec_hash[:16]}_{snapshot}"`
(line ~711). Add a `cache_suffix: str | None = None` parameter to
`create_dataset_minid` and pass `cache_suffix` from the Tier-3 call site (Task 3,
Step 3). When `cache_suffix` is provided, use it directly instead of
recomputing; this keeps the cache key identical to what Tier-1 looks up. If
`cache_suffix is None` (defensive), fall back to deriving it from
`spec_hash` + the version's snapshot via `_resolve_version_record(dataset, version).snapshot`.

- [ ] **Step 4: Run the new tests**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_client_bag_staging.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/dataset/bag_download.py tests/dataset/test_client_bag_staging.py
git commit -m "feat(bag): client arm stages under cache_dir in a TemporaryDirectory, extracts in place"
```

---

### Task 3: Wire Tier 3, delete dead branches, prune cleanup

Make `get_dataset_minid` Tier 3 pass `cache_suffix` and wrap the returned cache path as a `DatasetMinid` whose `location` is the extracted dir (so downstream `download_dataset_minid` is a cache-hit no-op). Delete the now-dead `file://`-archive branch, the legacy `DerivaExport.retrieve_file` `else`, and the `client_export` string-matched cleanup in `download_dataset_minid`.

**Files:**
- Modify: `src/deriva_ml/dataset/bag_download.py` (`get_dataset_minid` Tier-3 call site ~line 610; `download_dataset_minid` branches ~lines 186–250)
- Test: `tests/dataset/test_get_dataset_minid_helpers.py` (extend Tier-3 assertions)

- [ ] **Step 1: Write/extend the failing test**

```python
def test_tier3_returns_minid_pointing_at_extracted_cache_dir(tmp_path):
    """Tier 3 wraps the client arm's returned cache PATH as a DatasetMinid
    whose location is the already-extracted bag dir, so the downstream
    download_dataset_minid call is a no-op cache hit (no file:// zip)."""
    from unittest.mock import MagicMock, patch
    from deriva_ml.dataset.bag_download import _tier3_client_path

    ds = MagicMock()
    ds.dataset_rid = "2-ABCD"
    cache_bag = tmp_path / "cache" / "bags" / "deadbeefdeadbeef_2-SNAP" / "Dataset_2-ABCD"
    cache_bag.mkdir(parents=True)

    with patch(
        "deriva_ml.dataset.bag_download.create_dataset_minid",
        return_value=cache_bag,  # now a Path, not a file:// str
    ) as mk:
        minid = _tier3_client_path(
            ds, version="1.0.0", create=True, minid_url=None,
            snapshot="2-SNAP", cache_suffix="deadbeefdeadbeef_2-SNAP",
            spec={"k": "v"}, spec_hash="deadbeefdeadbeef", exclude_tables=None, timeout=None,
        )

    # create_dataset_minid was given the cache_suffix so its key matches Tier 1.
    assert mk.call_args.kwargs["cache_suffix"] == "deadbeefdeadbeef_2-SNAP"
    # The minid location is a file:// URI of the EXTRACTED dir's parent (cache root),
    # matching the Tier-1 shape so download_dataset_minid hits bag_dir.exists().
    assert minid.bag_url.startswith("file://")
    assert "bags" in minid.bag_url
```

(Adjust `_tier3_client_path`'s real signature to taste — the assertion that matters is: `cache_suffix` is forwarded, and `location` points at the extracted cache dir, not a zip.)

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_get_dataset_minid_helpers.py -q`
Expected: FAIL — Tier 3 currently forwards no `cache_suffix` and wraps a `file://` zip URI.

- [ ] **Step 3: Wire Tier 3 + forward cache_suffix**

In the Tier-3 client path (`_tier3_client_path` / the `create_dataset_minid` call at ~610), pass `cache_suffix=cache_suffix`, take the returned `Path`, and build the `DatasetMinid` with `location=bag_dir.parent.as_uri()` (the cache root for the checksum — same shape Tier-1 produces at line ~497, so `download_dataset_minid`'s top-of-function `bag_dir.exists()` guard returns it immediately).

```python
    bag_dir = create_dataset_minid(
        dataset, version, use_minid=False, exclude_tables=exclude_tables,
        spec=spec, spec_hash=spec_hash, cache_suffix=cache_suffix, timeout=timeout,
    )
    return DatasetMinid(
        dataset_version=version,
        RID=_build_version_rid(dataset.dataset_rid, snapshot),
        location=bag_dir.parent.as_uri(),
        checksums=[{"function": "sha256", "value": cache_suffix}],
    )
```

- [ ] **Step 4: Delete dead branches in `download_dataset_minid`**

Remove:
- the `elif minid.bag_url.startswith("file://"):` archive branch (no producer now — the client arm extracts itself and the only `file://` minids are already-extracted Tier-1/Tier-3 dirs that hit the `bag_dir.exists()` guard at the top),
- the legacy `else: exporter = DerivaExport(...).retrieve_file(...)` branch (no live producer),
- the `client_export` string-matched cleanup block (lines ~245–250).

Keep: the Tier-1 `bag_dir.exists()` early return, the S3 (`use_minid=True`) download arm, and the SHA-256-fallback. Remove the now-unused `DerivaExport`/`urlparse` imports if nothing else uses them (grep first).

- [ ] **Step 5: Run full download/minid suite**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_get_dataset_minid_helpers.py tests/dataset/test_extract_archive_to_cache.py tests/dataset/test_client_bag_staging.py tests/dataset/test_bag_materialize.py tests/dataset/test_multi_anchor_bag_cache.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/dataset/bag_download.py tests/dataset/test_get_dataset_minid_helpers.py
git commit -m "refactor(bag): Tier-3 returns extracted cache path; drop dead file:// + legacy branches"
```

---

### Task 4: Live round-trip + working_dir-clean assertion

Confirm the end-to-end client download still produces a loadable bag in `cache_dir` and leaves `working_dir` free of `client_export`. Needs `DERIVA_HOST=localhost`.

**Files:**
- Test: `tests/dataset/test_client_bag_staging.py` (add a live test, marked to need a catalog)

- [ ] **Step 1: Write the live test**

```python
def test_live_client_download_lands_in_cache_not_working_dir(catalog_with_datasets):
    """End-to-end: cache(use_minid=False) builds a loadable bag under cache_dir
    and never creates client_export under working_dir."""
    ml, _ = catalog_with_datasets
    datasets = list(ml.find_datasets())
    if not datasets:
        pytest.skip("Need a dataset.")
    ds = ml.lookup_dataset(datasets[0].dataset_rid)
    versions = list(ds.dataset_history())
    version = str(versions[-1].dataset_version)

    info = ds.cache(version=version, materialize=False)
    assert info["cache_status"] in ("cached_holey", "cached_materialized")
    assert info["cache_path"] is not None
    assert Path(info["cache_path"]).exists()
    # cache_path is under cache_dir, and working_dir has no client_export.
    assert Path(ml.cache_dir) in Path(info["cache_path"]).parents
    assert not (Path(ml.working_dir) / "client_export").exists()
```

- [ ] **Step 2: Run the live test**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_client_bag_staging.py -q`
Expected: PASS (the unit tests + the live round-trip).

- [ ] **Step 3: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add tests/dataset/test_client_bag_staging.py
git commit -m "test(bag): live client download lands in cache_dir, working_dir stays clean"
```

---

### Task 5: cache_tui, lint, full sweep, PR

Update the cache TUI's `working_dir` subdir list (it lists `client_export` — now obsolete under working_dir), lint/format, run the broader dataset suite, open the PR.

**Files:**
- Modify: `src/deriva_ml/cache_tui.py` (drop `client_export` from the working_dir subdir lists at ~276/288, since it no longer lands there)
- Modify: `docs/superpowers/specs/2026-06-15-client-bag-staging-tmpdir-design.md` (status → Implemented) if desired

- [ ] **Step 1: Update cache_tui**

Remove `"client_export"` from the two `working_dir` subdir lists in `cache_tui.py` (it no longer appears under `working_dir`). Grep to confirm no other `working_dir`-rooted `client_export` reference remains:

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && grep -rn "client_export" src/`
Expected: zero hits in `src/` after this (the staging is anonymous TemporaryDirectory names now).

- [ ] **Step 2: Lint + format**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check --fix src/deriva_ml/dataset/bag_download.py src/deriva_ml/cache_tui.py tests/dataset/test_extract_archive_to_cache.py tests/dataset/test_client_bag_staging.py && uv run ruff format src/deriva_ml/dataset/bag_download.py src/deriva_ml/cache_tui.py tests/dataset/`
Expected: clean.

- [ ] **Step 3: Broader dataset suite**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_download.py tests/dataset/test_format_b_bag.py tests/dataset/test_get_dataset_minid_helpers.py tests/dataset/test_extract_archive_to_cache.py tests/dataset/test_client_bag_staging.py tests/dataset/test_bag_materialize.py tests/dataset/test_multi_anchor_bag_cache.py -q`
Expected: all PASS.

- [ ] **Step 4: Branch, commit, PR**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add -A
git commit -m "chore(bag): drop obsolete client_export from cache_tui working_dir list"
git push -u origin feat/client-bag-staging-cache-dir
gh pr create --title "fix(bag): client-side bag staging moves off working_dir into a cache_dir TemporaryDirectory" --body "<summary + spec link + before/after + the two defects fixed>"
```

---

## Self-Review notes

- **Spec coverage:** Task 1 = shared helper; Task 2 = TemporaryDirectory(dir=cache_dir) + self-extract + return cache path; Task 3 = Tier-3 wiring + dead-branch deletion; Task 4 = live working_dir-clean; Task 5 = cache_tui + lint + PR. All spec sections covered.
- **Type consistency:** `create_dataset_minid` return becomes `str | Path` (MINID arm → `str` landing URL; client arm → `Path` cache dir). `_extract_archive_to_cache` returns `Path`. Tier-3 builds `DatasetMinid(location=bag_dir.parent.as_uri())`.
- **Cache-key invariant:** `cache_suffix` is threaded from `get_dataset_minid` into `create_dataset_minid` so the key the client arm writes equals the key Tier-1 reads — existing cached bags keep hitting.
- **Deletion safety:** the `file://`-archive and legacy `DerivaExport` branches are removed only after confirming (in the audit) they have no live producer; if a hidden caller exists it surfaces as a Tier-3 dispatch test failure.
