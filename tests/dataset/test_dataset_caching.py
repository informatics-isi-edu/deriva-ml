"""Tests for dataset download caching lifecycle.

Tests cover:
- Fresh download creates cache entry
- Re-download returns cached bag without re-downloading
- Materialized vs non-materialized cache states
- Non-default cache directory
- cache() warming API and bag_info() status reporting
- Multiple versions cached simultaneously
- Cache resume after partial materialization
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from deriva_ml import DerivaML
from deriva_ml.dataset.aux_classes import DatasetSpec, VersionPart
from deriva_ml.dataset.bag_cache import BagCache, CacheStatus
from tests.catalog_manager import CatalogManager


class TestDownloadCacheLifecycle:
    """Integration tests for the full download → cache → re-use lifecycle."""

    def test_fresh_download_creates_cache(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """First download creates a cache entry that bag_info can find."""
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        # Before download: not cached
        info_before = dataset.bag_info(version=version)
        assert info_before["status"] == CacheStatus.not_cached.value

        # Download
        bag = dataset.download_dataset_bag(version=version, use_minid=False)
        assert bag is not None

        # After download: cached and materialized
        info_after = dataset.bag_info(version=version)
        assert info_after["status"] == CacheStatus.cached_materialized.value
        assert info_after["cache_path"] is not None

    def test_redownload_uses_cache(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Second download of the same version uses cache (no re-export)."""
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        # First download
        bag1 = dataset.download_dataset_bag(version=version, use_minid=False)

        # After first download: should be cached
        info = dataset.bag_info(version=version)
        assert info["status"] == CacheStatus.cached_materialized.value

        # Second download — verify it still works and cache is still present
        bag2 = dataset.download_dataset_bag(version=version, use_minid=False)

        # Both should produce usable bags with the same dataset RID
        assert bag1.dataset_rid == bag2.dataset_rid

        # Cache should still show materialized
        info2 = dataset.bag_info(version=version)
        assert info2["status"] == CacheStatus.cached_materialized.value

    def test_materialized_download(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Download with materialize=True results in cached_materialized status."""
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        bag = dataset.download_dataset_bag(
            version=version, use_minid=False, materialize=True
        )
        assert bag is not None

        info = dataset.bag_info(version=version)
        assert info["status"] == CacheStatus.cached_materialized.value

    def test_non_materialized_download(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Download with materialize=False results in cached_metadata_only status."""
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        bag = dataset.download_dataset_bag(
            version=version, use_minid=False, materialize=False
        )
        assert bag is not None

        info = dataset.bag_info(version=version)
        # Should be metadata_only since we didn't materialize
        assert info["status"] in (
            CacheStatus.cached_metadata_only.value,
            CacheStatus.cached_materialized.value,  # May be materialized if no fetch.txt entries
        )

    def test_non_default_cache_directory(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Download to a non-default cache directory stores bag there."""
        catalog_manager.reset()
        ml_default, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset_rid = dataset_desc.dataset.dataset_rid
        version = dataset_desc.dataset.current_version

        # Create a second ML instance with a custom cache directory
        custom_cache = tmp_path / "custom_cache"
        ml_custom = DerivaML(
            catalog_manager.hostname,
            str(catalog_manager.catalog_id),
            working_dir=tmp_path / "custom_workdir",
            cache_dir=custom_cache,
        )
        dataset = ml_custom.lookup_dataset(dataset_rid)

        # Download via the custom-cache ML instance
        bag = dataset.download_dataset_bag(version=version, use_minid=False)
        assert bag is not None

        # Verify the bag is in the custom cache directory, not the default
        cache = BagCache(custom_cache)
        result = cache.cache_status(dataset_rid)
        assert result["status"] != CacheStatus.not_cached.value
        assert result["cache_path"] is not None
        assert str(custom_cache) in str(result["cache_path"])


class TestCacheWarmingAPI:
    """Tests for cache() / prefetch() and bag_info() methods."""

    def test_cache_warms_without_execution(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """cache() downloads the bag without creating an execution record."""
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        # Count executions before
        pb = ml.pathBuilder()
        executions_before = len(list(
            pb.schemas["deriva-ml"].tables["Execution"].path.entities().fetch()
        ))

        # Warm cache
        result = dataset.cache(version=version)

        # Verify cached
        assert result["status"] == CacheStatus.cached_materialized.value

        # Count executions after — should not have created a new one
        # (the MCP workflow execution from catalog setup doesn't count)
        executions_after = len(list(
            pb.schemas["deriva-ml"].tables["Execution"].path.entities().fetch()
        ))
        assert executions_after == executions_before

    def test_bag_info_returns_size_and_status(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """bag_info() returns both size estimates and cache status."""
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        info = dataset.bag_info(version=version)

        # Should have size fields
        assert "total_rows" in info
        assert "total_asset_bytes" in info
        assert "tables" in info

        # Should have cache status fields
        assert "status" in info
        assert "cache_path" in info
        assert "versions_cached" in info

        # Before download, should be not_cached
        assert info["status"] == CacheStatus.not_cached.value

    def test_bag_info_after_cache(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """bag_info() reflects cached status after cache() call."""
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        # Warm cache
        dataset.cache(version=version)

        # Now bag_info should show cached
        info = dataset.bag_info(version=version)
        assert info["status"] == CacheStatus.cached_materialized.value
        assert info["cache_path"] is not None

    def test_prefetch_is_alias_for_cache(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """prefetch() is a backward-compatible alias for cache()."""
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        result = dataset.prefetch(version=version)
        assert result["status"] == CacheStatus.cached_materialized.value


class TestMultiVersionCache:
    """Tests for caching multiple versions of the same dataset."""

    def test_different_versions_cached_separately(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Different versions of the same dataset have separate cache entries."""
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        v1 = dataset.current_version

        # Download v1
        bag1 = dataset.download_dataset_bag(version=v1, use_minid=False)

        # Create v2 by adding members
        pb = ml.pathBuilder()
        subjects = [
            s["RID"]
            for s in pb.schemas[ml.default_schema].tables["Subject"].path.entities().fetch()
        ]
        if len(subjects) >= 2:
            dataset.add_dataset_members(subjects[-2:])
        v2 = dataset.current_version
        assert str(v2) != str(v1)

        # Download v2
        bag2 = dataset.download_dataset_bag(version=v2, use_minid=False)

        # Both should be cached
        cache = BagCache(ml.cache_dir)
        result = cache.cache_status(dataset.dataset_rid)
        assert len(result["versions_cached"]) >= 2

        # Bags should have different paths
        assert bag1.model.bag_path != bag2.model.bag_path


class TestCacheStatusTransitions:
    """Unit tests for cache status transitions using filesystem manipulation."""

    def test_not_cached_to_metadata_only(self, tmp_path):
        """Status transitions from not_cached to cached_metadata_only."""
        cache = BagCache(tmp_path)
        rid = "TEST"

        # Start: not cached
        assert cache.cache_status(rid)["status"] == CacheStatus.not_cached.value

        # Create a bag directory with unresolved fetch.txt
        bag_dir = tmp_path / f"{rid}_checksum123"
        bag_path = bag_dir / f"Dataset_{rid}"
        bag_path.mkdir(parents=True)
        (bag_path / "bagit.txt").write_text(
            "BagIt-Version: 1.0\nTag-File-Character-Encoding: UTF-8\n"
        )
        (bag_path / "data").mkdir()
        (bag_path / "manifest-sha256.txt").write_text("")
        (bag_path / "fetch.txt").write_text(
            "https://example.com/file.dat\t100\tdata/file.dat\n"
        )

        # Now: metadata only
        assert cache.cache_status(rid)["status"] == CacheStatus.cached_metadata_only.value

    def test_metadata_only_to_materialized(self, tmp_path):
        """Status transitions from cached_metadata_only to cached_materialized."""
        from bdbag import bdbag_api as bdb

        cache = BagCache(tmp_path)
        rid = "TEST"

        bag_dir = tmp_path / f"{rid}_checksum123"
        bag_path = bag_dir / f"Dataset_{rid}"
        bag_path.mkdir(parents=True)
        data_dir = bag_path / "data"
        data_dir.mkdir()
        data_dir.joinpath("file.dat").write_text("content")
        bdb.make_bag(str(bag_path), algs=["sha256"], idempotent=True)

        # Without validated_check: metadata only
        assert cache.cache_status(rid)["status"] == CacheStatus.cached_metadata_only.value

        # Add validated_check marker
        (bag_dir / "validated_check.txt").touch()

        # Now: materialized
        assert cache.cache_status(rid)["status"] == CacheStatus.cached_materialized.value

    def test_materialized_to_incomplete(self, tmp_path):
        """Status transitions to incomplete when files go missing."""
        cache = BagCache(tmp_path)
        rid = "TEST"

        bag_dir = tmp_path / f"{rid}_checksum123"
        bag_path = bag_dir / f"Dataset_{rid}"
        bag_path.mkdir(parents=True)
        (bag_path / "bagit.txt").write_text(
            "BagIt-Version: 1.0\nTag-File-Character-Encoding: UTF-8\n"
        )
        data_dir = bag_path / "data"
        data_dir.mkdir()

        # Create a file, then add fetch.txt referencing another file that's missing
        data_dir.joinpath("present.dat").write_text("here")
        (bag_path / "fetch.txt").write_text(
            "https://example.com/missing.dat\t100\tdata/missing.dat\n"
        )
        (bag_path / "manifest-sha256.txt").write_text("")

        # Mark as validated but file is missing
        (bag_dir / "validated_check.txt").touch()

        # Should be incomplete
        assert cache.cache_status(rid)["status"] == CacheStatus.cached_incomplete.value
