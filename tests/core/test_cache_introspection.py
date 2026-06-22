"""Unit tests for cache & storage introspection (spec 2026-06-11).

No live catalog: bags are synthesized with ``BagCacheIndex.record()``
plus ``bdbag_api.make_bag`` so status detection sees structurally
valid bags. Assets are synthesized as ``{rid}_{md5}`` directories.

DerivaML methods are exercised unbound against a lightweight harness
(same pattern as tests/core/test_storage_management.py).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_valid_bag(bag_dir: Path, files: dict[str, str] | None = None) -> None:
    """Create a structurally valid BDBag at ``bag_dir``.

    ``BagCache._is_fully_materialized`` runs
    ``bdbag_api.validate_bag_structure``; a hand-rolled directory
    fails it, so build a real bag in place.
    """
    from bdbag import bdbag_api as bdb

    bag_dir.mkdir(parents=True, exist_ok=True)
    for name, content in (files or {"data.csv": "a,b\n1,2\n"}).items():
        (bag_dir / name).write_text(content)
    bdb.make_bag(bag_dir.as_posix())


def _record_bag(
    cache_dir: Path,
    *,
    checksum: str,
    dataset_rid: str,
    version: str = "1.0.0",
    built_at: datetime | None = None,
    on_disk: bool = True,
    holey: bool = False,
) -> Path:
    """Record a synthetic bag in the index (and optionally on disk).

    Mirrors the production record() call in
    ``dataset/bag_download.py`` — anchors ``[("Dataset", rid)]``,
    ``anchor_summary={"version": ...}`` — but deliberately omits
    ``size_bytes`` (which production now persists), leaving the index
    column NULL. That exercises the lazy back-fill path in
    ``BagCache.list_bags`` for pre-existing caches.

    Returns the bag directory path (``bags/{checksum}/Dataset_{rid}``).
    """
    from deriva.bag.cache_index import BagCacheIndex

    index = BagCacheIndex(cache_dir)
    try:
        index.record(
            checksum=checksum,
            anchors=[("Dataset", dataset_rid)],
            anchor_summary={"version": version},
            built_at=built_at,
        )
        bag_dir = index.bag_dir_for(checksum) / f"Dataset_{dataset_rid}"
    finally:
        index.dispose()
    if on_disk:
        _make_valid_bag(bag_dir)
        if holey:
            # A fetch.txt entry referencing a file that was never
            # fetched marks the bag holey.
            (bag_dir / "fetch.txt").write_text("https://example.org/x\t10\tdata/missing.bin\n")
    return bag_dir


def _make_cached_asset(cache_dir: Path, rid: str, md5: str, n_files: int = 1) -> Path:
    """Create a synthetic cached asset dir ``assets/{rid}_{md5}``."""
    asset_dir = cache_dir / "assets" / f"{rid}_{md5}"
    asset_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        (asset_dir / f"file{i}.bin").write_bytes(b"x" * 100)
    return asset_dir


class _StorageHarness:
    """Stand-in for ``self`` in DerivaML storage methods.

    Carries the attributes the storage surface reads:
    ``working_dir``, ``cache_dir``, ``_logger``.
    """

    def __init__(self, working_dir: Path, cache_dir: Path):
        self.working_dir = working_dir
        self.cache_dir = cache_dir
        self._logger = logging.getLogger("test")


@pytest.fixture
def harness(tmp_path: Path) -> _StorageHarness:
    working_dir = tmp_path / "wd"
    cache_dir = tmp_path / "cache"
    working_dir.mkdir()
    cache_dir.mkdir()
    return _StorageHarness(working_dir, cache_dir)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class TestRecords:
    def test_cached_bag_fields_and_dump(self, tmp_path: Path):
        from deriva_ml.core.storage import CachedBag
        from deriva_ml.dataset.bag_cache import CacheStatus

        bag = CachedBag(
            dataset_rid="XYZ1",
            version="1.0.0",
            checksum="abc123",
            status=CacheStatus.cached_materialized,
            built_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
            size_bytes=1024,
            path=tmp_path / "bags" / "abc123" / "Dataset_XYZ1",
        )
        dumped = bag.model_dump()
        assert dumped["dataset_rid"] == "XYZ1"
        assert dumped["status"] == "cached_materialized"

    def test_cached_asset_fields_and_dump(self, tmp_path: Path):
        from deriva_ml.core.storage import CachedAsset

        asset = CachedAsset(
            rid="A1",
            md5="d41d8cd98f00b204e9800998ecf8427e",
            file_count=2,
            size_bytes=200,
            modified=datetime(2026, 6, 1, tzinfo=timezone.utc),
            path=tmp_path / "assets" / "A1_d41d8cd98f00b204e9800998ecf8427e",
        )
        assert asset.model_dump()["rid"] == "A1"


# ---------------------------------------------------------------------------
# BagCache.list_bags
# ---------------------------------------------------------------------------


class TestListBags:
    def test_empty_cache_returns_empty_list(self, tmp_path: Path):
        from deriva_ml.dataset.bag_cache import BagCache

        with BagCache(tmp_path / "cache") as cache:
            assert cache.list_bags() == []

    def test_lists_recorded_bags_with_rid_version_status(self, tmp_path: Path):
        from deriva_ml.dataset.bag_cache import BagCache, CacheStatus

        cache_dir = tmp_path / "cache"
        _record_bag(cache_dir, checksum="aaa111", dataset_rid="RID-A", version="1.0.0")
        _record_bag(cache_dir, checksum="bbb222", dataset_rid="RID-B", version="2.0.0", holey=True)

        with BagCache(cache_dir) as cache:
            bags = cache.list_bags()

        assert {b.dataset_rid for b in bags} == {"RID-A", "RID-B"}
        by_rid = {b.dataset_rid: b for b in bags}
        assert by_rid["RID-A"].version == "1.0.0"
        assert by_rid["RID-A"].status == CacheStatus.cached_materialized
        assert by_rid["RID-B"].status == CacheStatus.cached_holey
        assert by_rid["RID-A"].size_bytes > 0
        assert by_rid["RID-A"].path.exists()

    def test_index_row_with_missing_dir_reports_not_cached(self, tmp_path: Path):
        from deriva_ml.dataset.bag_cache import BagCache, CacheStatus

        cache_dir = tmp_path / "cache"
        _record_bag(cache_dir, checksum="ddd444", dataset_rid="RID-G", on_disk=False)

        with BagCache(cache_dir) as cache:
            bags = cache.list_bags()

        assert len(bags) == 1
        assert bags[0].status == CacheStatus.not_cached

    def test_multi_anchor_bag_yields_one_entry_per_dataset(self, tmp_path: Path):
        from deriva.bag.cache_index import BagCacheIndex

        from deriva_ml.dataset.bag_cache import BagCache, CacheStatus

        cache_dir = tmp_path / "cache"
        _record_bag(cache_dir, checksum="ccc333", dataset_rid="RID-X")
        # Second dataset anchors the same content-addressed bag.
        index = BagCacheIndex(cache_dir)
        try:
            index.record(checksum="ccc333", anchors=[("Dataset", "RID-Y")])
        finally:
            index.dispose()

        with BagCache(cache_dir) as cache:
            bags = cache.list_bags()

        assert len(bags) == 2
        assert {b.dataset_rid for b in bags} == {"RID-X", "RID-Y"}
        assert {b.checksum for b in bags} == {"ccc333"}
        by_rid = {b.dataset_rid: b for b in bags}
        # RID-Y anchors the same checksum but has no Dataset_RID-Y dir
        # on disk — status detection must report it not_cached.
        assert by_rid["RID-Y"].status == CacheStatus.not_cached
        assert by_rid["RID-X"].status == CacheStatus.cached_materialized


# ---------------------------------------------------------------------------
# BagCache.purge_dataset
# ---------------------------------------------------------------------------


class TestPurgeDataset:
    def test_purge_all_versions(self, tmp_path: Path):
        from deriva_ml.dataset.bag_cache import BagCache, CacheStatus

        cache_dir = tmp_path / "cache"
        d1 = _record_bag(cache_dir, checksum="e1", dataset_rid="RID-P", version="1.0.0")
        d2 = _record_bag(cache_dir, checksum="e2", dataset_rid="RID-P", version="2.0.0")
        _record_bag(cache_dir, checksum="e3", dataset_rid="RID-Q", version="1.0.0")

        with BagCache(cache_dir) as cache:
            stats = cache.purge_dataset("RID-P")
            remaining = cache.list_bags()
            status = cache.cache_status("RID-P")

        assert stats["bags_removed"] == 2
        assert stats["bytes_freed"] > 0
        assert not d1.exists() and not d2.exists()
        assert {b.dataset_rid for b in remaining} == {"RID-Q"}
        assert status["status"] == CacheStatus.not_cached.value

    def test_purge_single_version(self, tmp_path: Path):
        from deriva_ml.dataset.bag_cache import BagCache

        cache_dir = tmp_path / "cache"
        _record_bag(cache_dir, checksum="f1", dataset_rid="RID-V", version="1.0.0")
        kept = _record_bag(cache_dir, checksum="f2", dataset_rid="RID-V", version="2.0.0")

        with BagCache(cache_dir) as cache:
            stats = cache.purge_dataset("RID-V", version="1.0.0")
            remaining = cache.list_bags()

        assert stats["bags_removed"] == 1
        assert kept.exists()
        assert [b.version for b in remaining] == ["2.0.0"]
        assert not (cache_dir / "bags" / "f1").exists()

    def test_purge_unknown_rid_is_idempotent_zero(self, tmp_path: Path):
        from deriva_ml.dataset.bag_cache import BagCache

        with BagCache(tmp_path / "cache") as cache:
            assert cache.purge_dataset("NOPE") == {"bags_removed": 0, "bytes_freed": 0}

    def test_purge_continues_past_failing_entry(self, tmp_path: Path, monkeypatch):
        """An OSError purging one bag must not abort the loop or lose stats."""
        from deriva.bag.cache_index import BagCacheIndex  # noqa: E402

        from deriva_ml.dataset.bag_cache import BagCache

        cache_dir = tmp_path / "cache"
        _record_bag(cache_dir, checksum="x1", dataset_rid="RID-ERR", version="1.0.0")
        _record_bag(cache_dir, checksum="x2", dataset_rid="RID-ERR", version="2.0.0")

        original_purge = BagCacheIndex.purge

        def flaky_purge(self, checksum):
            if checksum == "x1":
                raise OSError("simulated rmtree failure")
            return original_purge(self, checksum)

        monkeypatch.setattr(BagCacheIndex, "purge", flaky_purge)

        with BagCache(cache_dir) as cache:
            stats = cache.purge_dataset("RID-ERR")

        assert stats["bags_removed"] == 1  # x2 succeeded despite x1 failing


# ---------------------------------------------------------------------------
# Asset cache: list_cached_assets / delete_cached_asset
# ---------------------------------------------------------------------------

MD5_A = "d41d8cd98f00b204e9800998ecf8427e"
MD5_B = "9e107d9d372bb6826bd81d3542a419d6"


class TestCachedAssets:
    def test_empty_or_missing_assets_dir(self, tmp_path: Path):
        from deriva_ml.core.storage import list_cached_assets

        assert list_cached_assets(tmp_path / "cache") == []

    def test_lists_assets_with_parsed_rid_md5(self, tmp_path: Path):
        from deriva_ml.core.storage import list_cached_assets

        cache_dir = tmp_path / "cache"
        _make_cached_asset(cache_dir, "RID-1", MD5_A, n_files=2)
        _make_cached_asset(cache_dir, "RID-2", MD5_B)

        assets = list_cached_assets(cache_dir)
        assert {(a.rid, a.md5) for a in assets} == {("RID-1", MD5_A), ("RID-2", MD5_B)}
        by_rid = {a.rid: a for a in assets}
        assert by_rid["RID-1"].file_count == 2
        assert by_rid["RID-1"].size_bytes == 200

    def test_nonconforming_entry_skipped(self, tmp_path: Path):
        from deriva_ml.core.storage import list_cached_assets

        cache_dir = tmp_path / "cache"
        _make_cached_asset(cache_dir, "RID-1", MD5_A)
        (cache_dir / "assets" / "not-an-asset").mkdir()
        (cache_dir / "assets" / "stray.txt").write_text("x")

        assets = list_cached_assets(cache_dir)
        assert [a.rid for a in assets] == ["RID-1"]

    def test_delete_specific_md5(self, tmp_path: Path):
        from deriva_ml.core.storage import delete_cached_asset

        cache_dir = tmp_path / "cache"
        gone = _make_cached_asset(cache_dir, "RID-1", MD5_A)
        kept = _make_cached_asset(cache_dir, "RID-1", MD5_B)

        stats = delete_cached_asset(cache_dir, "RID-1", md5=MD5_A)
        assert stats["assets_removed"] == 1
        assert stats["bytes_freed"] == 100
        assert not gone.exists() and kept.exists()

    def test_delete_all_for_rid(self, tmp_path: Path):
        from deriva_ml.core.storage import delete_cached_asset

        cache_dir = tmp_path / "cache"
        _make_cached_asset(cache_dir, "RID-1", MD5_A)
        _make_cached_asset(cache_dir, "RID-1", MD5_B)
        other = _make_cached_asset(cache_dir, "RID-2", MD5_A)

        stats = delete_cached_asset(cache_dir, "RID-1")
        assert stats["assets_removed"] == 2
        assert other.exists()

    def test_delete_missing_is_idempotent_zero(self, tmp_path: Path):
        from deriva_ml.core.storage import delete_cached_asset

        assert delete_cached_asset(tmp_path / "cache", "NOPE") == {
            "assets_removed": 0,
            "bytes_freed": 0,
        }

    def test_delete_does_not_overmatch_prefix_sharing_rids(self, tmp_path: Path):
        from deriva_ml.core.storage import delete_cached_asset

        cache_dir = tmp_path / "cache"
        target = _make_cached_asset(cache_dir, "RID-1", MD5_A)
        # Shares the "RID-1" prefix but is a different rid — glob
        # "RID-1_*" must not delete it.
        bystander = _make_cached_asset(cache_dir, "RID-1_EXTRA", MD5_B)

        stats = delete_cached_asset(cache_dir, "RID-1")
        assert stats["assets_removed"] == 1
        assert not target.exists()
        assert bystander.exists()


# ---------------------------------------------------------------------------
# clear_cache (index-coherent engine)
# ---------------------------------------------------------------------------


class TestClearCacheCoherent:
    def test_clears_bags_assets_and_strays(self, tmp_path: Path):
        from deriva_ml.core.storage import clear_cache
        from deriva_ml.dataset.bag_cache import BagCache

        cache_dir = tmp_path / "cache"
        _record_bag(cache_dir, checksum="g1", dataset_rid="RID-C")
        _make_cached_asset(cache_dir, "RID-1", MD5_A)
        (cache_dir / "stray.txt").write_text("x")

        stats = clear_cache(cache_dir)

        assert stats["errors"] == 0
        assert stats["dirs_removed"] >= 2  # the bag dir + the asset dir
        assert stats["bytes_freed"] > 0
        with BagCache(cache_dir) as cache:
            assert cache.list_bags() == []  # index agrees: nothing cached
        assert list((cache_dir / "assets").iterdir()) == []
        assert not (cache_dir / "stray.txt").exists()

    def test_age_filter_uses_index_built_at_for_bags(self, tmp_path: Path):
        from deriva_ml.core.storage import clear_cache
        from deriva_ml.dataset.bag_cache import BagCache

        cache_dir = tmp_path / "cache"
        old = datetime.now(timezone.utc) - timedelta(days=40)
        _record_bag(cache_dir, checksum="h1", dataset_rid="RID-OLD", built_at=old)
        fresh_dir = _record_bag(cache_dir, checksum="h2", dataset_rid="RID-NEW")

        stats = clear_cache(cache_dir, older_than_days=30)

        assert stats["dirs_removed"] == 1
        with BagCache(cache_dir) as cache:
            remaining = cache.list_bags()
        assert [b.dataset_rid for b in remaining] == ["RID-NEW"]
        assert fresh_dir.exists()

    def test_index_never_references_removed_dirs(self, tmp_path: Path):
        """Regression: the pre-rewrite clear_cache could delete bag
        dirs while the index still listed them (spec section 1)."""
        from deriva.bag.cache_index import BagCacheIndex

        from deriva_ml.core.storage import clear_cache

        cache_dir = tmp_path / "cache"
        old = datetime.now(timezone.utc) - timedelta(days=40)
        _record_bag(cache_dir, checksum="i1", dataset_rid="RID-Z", built_at=old)

        clear_cache(cache_dir, older_than_days=30)

        index = BagCacheIndex(cache_dir)
        try:
            for row in index.list_bags():
                assert index.bag_dir_for(row["checksum"]).exists(), f"index references removed bag {row['checksum']}"
        finally:
            index.dispose()

    def test_missing_cache_dir_returns_zeros(self, tmp_path: Path):
        from deriva_ml.core.storage import clear_cache

        stats = clear_cache(tmp_path / "nope")
        assert stats == {"files_removed": 0, "dirs_removed": 0, "bytes_freed": 0, "errors": 0}

    def test_index_only_row_purged_without_inflating_dirs_removed(self, tmp_path: Path):
        from deriva.bag.cache_index import BagCacheIndex

        from deriva_ml.core.storage import clear_cache

        cache_dir = tmp_path / "cache"
        _record_bag(cache_dir, checksum="j9", dataset_rid="RID-GONE", on_disk=False)

        stats = clear_cache(cache_dir)

        assert stats["dirs_removed"] == 0
        assert stats["errors"] == 0
        index = BagCacheIndex(cache_dir)
        try:
            assert index.list_bags() == []  # stale row repaired
        finally:
            index.dispose()

    def test_survives_corrupt_index(self, tmp_path: Path):
        """A corrupt index.sqlite must not abort passes 2-3 or raise."""
        from deriva_ml.core.storage import clear_cache

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "index.sqlite").write_text("not a sqlite database")
        _make_cached_asset(cache_dir, "RID-CC", MD5_A)
        (cache_dir / "stray.txt").write_text("x")

        stats = clear_cache(cache_dir)

        assert stats["errors"] >= 1  # the unusable index
        assert not (cache_dir / "stray.txt").exists()
        assert list((cache_dir / "assets").iterdir()) == []


# ---------------------------------------------------------------------------
# DerivaML surface (unbound methods against the harness)
# ---------------------------------------------------------------------------


class TestDerivaMLSurface:
    def test_list_cached_bags_delegates(self, harness):
        from deriva_ml.core.base import DerivaML

        _record_bag(harness.cache_dir, checksum="j1", dataset_rid="RID-S")
        bags = DerivaML.list_cached_bags(harness)
        assert [b.dataset_rid for b in bags] == ["RID-S"]

    def test_delete_cached_bag_delegates(self, harness):
        from deriva_ml.core.base import DerivaML

        _record_bag(harness.cache_dir, checksum="k1", dataset_rid="RID-T")
        stats = DerivaML.delete_cached_bag(harness, "RID-T")
        assert stats["bags_removed"] == 1
        assert DerivaML.list_cached_bags(harness) == []

    def test_list_and_delete_cached_assets_delegate(self, harness):
        from deriva_ml.core.base import DerivaML

        _make_cached_asset(harness.cache_dir, "RID-U", MD5_A)
        assert [a.rid for a in DerivaML.list_cached_assets(harness)] == ["RID-U"]
        stats = DerivaML.delete_cached_asset(harness, "RID-U")
        assert stats["assets_removed"] == 1

    def test_clear_cache_is_index_coherent_via_derivaml(self, harness):
        from deriva.bag.cache_index import BagCacheIndex

        from deriva_ml.core.base import DerivaML

        _record_bag(harness.cache_dir, checksum="l1", dataset_rid="RID-W")
        DerivaML.clear_cache(harness)
        index = BagCacheIndex(harness.cache_dir)
        try:
            assert index.list_bags() == []
        finally:
            index.dispose()

    def test_storage_summary_has_species_breakdown(self, harness):
        from deriva_ml.core.base import DerivaML

        # get_storage_summary calls these as self.<method>(); bind the
        # real implementations to the harness (existing pattern in
        # tests/core/test_storage_management.py).
        harness.get_cache_size = lambda: DerivaML.get_cache_size(harness)
        harness.list_execution_dirs = lambda: DerivaML.list_execution_dirs(harness)
        harness.list_cached_bags = lambda: DerivaML.list_cached_bags(harness)
        harness.list_cached_assets = lambda: DerivaML.list_cached_assets(harness)

        _record_bag(harness.cache_dir, checksum="m1", dataset_rid="RID-SUM")
        _make_cached_asset(harness.cache_dir, "RID-AS", MD5_A)

        summary = DerivaML.get_storage_summary(harness)

        # Existing keys unchanged
        for key in (
            "working_dir",
            "cache_dir",
            "cache_size_mb",
            "cache_file_count",
            "execution_dir_count",
            "execution_size_mb",
            "total_size_mb",
        ):
            assert key in summary
        # New per-species keys
        assert summary["bag_count"] == 1
        assert summary["asset_count"] == 1
        assert summary["bag_size_mb"] > 0
        assert summary["asset_size_mb"] > 0


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_records_importable_from_top_level(self):
        from deriva_ml import CachedAsset, CachedBag  # noqa: F401


# ---------------------------------------------------------------------------
# PR #286 review follow-ups: orphan-dir sweep, summary denominators,
# listing order
# ---------------------------------------------------------------------------


class TestClearCacheOrphanSweep:
    def test_orphan_bag_dir_removed(self, tmp_path: Path):
        """A bags/ directory with no index row (e.g. left by the old
        blind-rmtree clear_cache) must be collected, not kept forever."""
        from deriva.bag.cache_index import BagCacheIndex

        from deriva_ml.core.storage import clear_cache

        cache_dir = tmp_path / "cache"
        indexed = _record_bag(cache_dir, checksum="ok1", dataset_rid="RID-OK")
        orphan = cache_dir / "bags" / "orphan-cafe" / "Dataset_RID-GONE"
        orphan.mkdir(parents=True)
        (orphan / "stale.csv").write_text("x" * 64)

        # A generous age filter keeps the fresh orphan (mtime-based,
        # same semantics the dedicated age test pins):
        clear_cache(cache_dir, older_than_days=3650)
        assert (cache_dir / "bags" / "orphan-cafe").exists()

        # With no age filter everything goes, orphan included:
        stats = clear_cache(cache_dir)
        assert not (cache_dir / "bags" / "orphan-cafe").exists()
        assert not indexed.exists()
        assert stats["errors"] == 0
        index = BagCacheIndex(cache_dir)
        try:
            assert index.list_bags() == []
        finally:
            index.dispose()

    def test_orphan_sweep_respects_age_filter(self, tmp_path: Path):
        import os
        import time as _time

        from deriva_ml.core.storage import clear_cache

        cache_dir = tmp_path / "cache"
        old_orphan = cache_dir / "bags" / "old-orphan" / "data"
        old_orphan.mkdir(parents=True)
        (old_orphan / "f.bin").write_bytes(b"x" * 32)
        fresh_orphan = cache_dir / "bags" / "fresh-orphan"
        fresh_orphan.mkdir(parents=True)
        (fresh_orphan / "f.bin").write_bytes(b"y" * 32)
        old_ts = _time.time() - 40 * 86400
        os.utime(cache_dir / "bags" / "old-orphan", (old_ts, old_ts))

        stats = clear_cache(cache_dir, older_than_days=30)

        assert not (cache_dir / "bags" / "old-orphan").exists()
        assert fresh_orphan.exists()
        assert stats["dirs_removed"] == 1


class TestSummaryDenominators:
    def test_multi_anchor_bag_sized_once_at_max(self, harness):
        """A bag with two dataset anchors: counted per anchor in
        bag_count, but sized once — and at the REAL size, not the
        last anchor's (whose Dataset_{rid} dir may not exist)."""
        from deriva.bag.cache_index import BagCacheIndex

        from deriva_ml.core.base import DerivaML

        harness.get_cache_size = lambda: DerivaML.get_cache_size(harness)
        harness.list_execution_dirs = lambda: DerivaML.list_execution_dirs(harness)
        harness.list_cached_bags = lambda: DerivaML.list_cached_bags(harness)
        harness.list_cached_assets = lambda: DerivaML.list_cached_assets(harness)

        _record_bag(harness.cache_dir, checksum="mm1", dataset_rid="RID-A")
        index = BagCacheIndex(harness.cache_dir)
        try:
            index.record(checksum="mm1", anchors=[("Dataset", "RID-B")])  # no dir on disk
        finally:
            index.dispose()

        summary = DerivaML.get_storage_summary(harness)
        assert summary["bag_count"] == 2  # two dataset-anchored entries
        assert summary["bag_size_mb"] > 0  # sized from the anchor that exists


class TestListBagsOrdering:
    def test_most_recently_built_first(self, tmp_path: Path):
        from deriva_ml.dataset.bag_cache import BagCache

        cache_dir = tmp_path / "cache"
        older = datetime.now(timezone.utc) - timedelta(days=2)
        _record_bag(cache_dir, checksum="ord1", dataset_rid="RID-OLDER", built_at=older)
        _record_bag(cache_dir, checksum="ord2", dataset_rid="RID-NEWER")

        with BagCache(cache_dir) as cache:
            bags = cache.list_bags()

        assert [b.dataset_rid for b in bags] == ["RID-NEWER", "RID-OLDER"]


class TestBagSizePersistence:
    """``size_bytes`` is persisted to the index, not re-walked every call.

    Regression: ``list_bags`` read ``row.get('size_bytes') or _dir_size(...)``
    but the column was never written, so every call re-walked the bag — making
    ``get_storage_summary`` / storage inspection O(cache size) on each run.
    Now download persists the size, and a NULL column (pre-existing cache) is
    back-filled on the first listing.
    """

    def _index_size(self, cache_dir: Path, checksum: str) -> int | None:
        from deriva.bag.cache_index import BagCacheIndex

        index = BagCacheIndex(cache_dir)
        try:
            return (index.get(checksum) or {}).get("size_bytes")
        finally:
            index.dispose()

    def test_list_bags_backfills_null_size_to_index(self, tmp_path: Path):
        from deriva_ml.dataset.bag_cache import BagCache

        cache_dir = tmp_path / "cache"
        # _record_bag leaves size_bytes NULL (mirrors a pre-existing cache).
        _record_bag(cache_dir, checksum="sz1", dataset_rid="RID-SZ")
        assert self._index_size(cache_dir, "sz1") is None

        with BagCache(cache_dir) as cache:
            bags = cache.list_bags()
        assert len(bags) == 1
        assert bags[0].size_bytes is not None
        assert bags[0].size_bytes > 0

        # The walked size is now persisted on the index row.
        persisted = self._index_size(cache_dir, "sz1")
        assert persisted == bags[0].size_bytes

    def test_backfill_does_not_re_walk_on_second_call(self, monkeypatch, tmp_path: Path):
        from deriva_ml.core import storage
        from deriva_ml.dataset.bag_cache import BagCache

        cache_dir = tmp_path / "cache"
        _record_bag(cache_dir, checksum="sz2", dataset_rid="RID-SZ2")

        calls = {"n": 0}
        real_dir_size = storage._dir_size

        def counting_dir_size(path):
            calls["n"] += 1
            return real_dir_size(path)

        monkeypatch.setattr(storage, "_dir_size", counting_dir_size)

        # First listing walks once (NULL column) and back-fills.
        with BagCache(cache_dir) as cache:
            cache.list_bags()
        first = calls["n"]
        assert first >= 1

        # Second listing reads size_bytes from the index — no further walk.
        with BagCache(cache_dir) as cache:
            cache.list_bags()
        assert calls["n"] == first, "size_bytes should be read from the index, not re-walked"

    def test_backfill_preserves_anchors_and_summary(self, tmp_path: Path):
        """Re-recording to persist size must not drop anchors or version."""
        from deriva.bag.cache_index import BagCacheIndex

        from deriva_ml.dataset.bag_cache import BagCache

        cache_dir = tmp_path / "cache"
        _record_bag(cache_dir, checksum="sz3", dataset_rid="RID-A3", version="2.1.0")
        # Second dataset anchors the same content-addressed bag.
        index = BagCacheIndex(cache_dir)
        try:
            index.record(checksum="sz3", anchors=[("Dataset", "RID-B3")])
        finally:
            index.dispose()

        with BagCache(cache_dir) as cache:
            cache.list_bags()  # triggers back-fill

        # Both anchors survive the back-fill upsert.
        index = BagCacheIndex(cache_dir)
        try:
            assert set(index.find_bags_for_rid(table="Dataset", rid="RID-A3")) == {"sz3"}
            assert set(index.find_bags_for_rid(table="Dataset", rid="RID-B3")) == {"sz3"}
        finally:
            index.dispose()


class TestStorageSummaryNoFullWalk:
    """``get_storage_summary`` derives the cache total from the index +
    asset listing + stray entries, never re-walking the bags/ subtree.
    """

    def test_summary_does_not_call_get_cache_size(self, harness, monkeypatch):
        from deriva_ml.core.base import DerivaML

        harness.list_execution_dirs = lambda: DerivaML.list_execution_dirs(harness)
        harness.list_cached_bags = lambda: DerivaML.list_cached_bags(harness)
        harness.list_cached_assets = lambda: DerivaML.list_cached_assets(harness)

        # If the summary still routed through the full-tree walker, this would
        # fire. It must not — bag size comes from the index.
        def boom():
            raise AssertionError("get_storage_summary must not call get_cache_size")

        harness.get_cache_size = boom

        _record_bag(harness.cache_dir, checksum="nw1", dataset_rid="RID-NW")
        _make_cached_asset(harness.cache_dir, "RID-NWA", MD5_A)

        summary = DerivaML.get_storage_summary(harness)
        assert summary["bag_count"] == 1
        assert summary["bag_size_mb"] > 0
        assert summary["asset_count"] == 1
        # cache total includes bags + assets.
        assert summary["cache_size_mb"] >= summary["bag_size_mb"] + summary["asset_size_mb"]
