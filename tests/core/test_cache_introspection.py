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
    ``anchor_summary={"version": ...}``.

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
            (bag_dir / "fetch.txt").write_text(
                "https://example.org/x\t10\tdata/missing.bin\n"
            )
    return bag_dir


def _make_cached_asset(
    cache_dir: Path, rid: str, md5: str, n_files: int = 1
) -> Path:
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
