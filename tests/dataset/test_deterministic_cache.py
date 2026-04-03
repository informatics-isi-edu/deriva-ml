"""Tests for deterministic cache key (spec_hash + snapshot).

The deterministic cache avoids re-downloading bags when the spec and snapshot
haven't changed. The cache key is {rid}_{spec_hash[:16]}_{snapshot}, computed
from the download spec JSON and the catalog snapshot ID.

Tests verify:
- First download creates deterministic cache entry
- Second download hits cache without re-generating bag
- Cache directory uses spec_hash+snapshot naming
- Schema changes invalidate cache (new spec_hash)
- New dataset version invalidates cache (new snapshot)
- bag_info reflects deterministic cache status
- Coexistence with old SHA-256 cache entries
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from deriva_ml import DerivaML
from deriva_ml.dataset.aux_classes import DatasetVersion, VersionPart
from deriva_ml.dataset.bag_cache import BagCache, CacheStatus
from tests.catalog_manager import CatalogManager


class TestDeterministicCacheKey:
    """Tests for the spec_hash+snapshot deterministic cache key."""

    def test_cache_dir_uses_deterministic_name(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Downloaded bag directory name includes spec_hash prefix and snapshot."""
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        bag = dataset.download_dataset_bag(version=version, use_minid=False)

        # The cache directory should contain the dataset RID
        cache_dirs = list(ml.cache_dir.glob(f"{dataset.dataset_rid}_*"))
        assert len(cache_dirs) >= 1, "Should have at least one cache entry"

        # The newest cache dir should have a deterministic name pattern
        # Format: {rid}_{spec_hash[:16]}_{snapshot}
        newest = max(cache_dirs, key=lambda p: p.stat().st_mtime)
        parts = newest.name.split("_", 1)
        assert parts[0] == dataset.dataset_rid
        # The suffix should contain both spec_hash prefix and snapshot
        suffix = parts[1]
        assert len(suffix) > 16, (
            f"Cache dir suffix '{suffix}' should contain spec_hash + snapshot"
        )

    def test_second_download_skips_bag_generation(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Second download with same version returns immediately from cache."""
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        # First download — generates bag
        bag1 = dataset.download_dataset_bag(version=version, use_minid=False)
        assert bag1 is not None

        # Record cache state
        cache_dirs_after_first = list(ml.cache_dir.glob(f"{dataset.dataset_rid}_*"))

        # Second download — should hit deterministic cache
        bag2 = dataset.download_dataset_bag(version=version, use_minid=False)
        assert bag2 is not None

        # No new cache directories should have been created
        cache_dirs_after_second = list(ml.cache_dir.glob(f"{dataset.dataset_rid}_*"))
        assert len(cache_dirs_after_second) == len(cache_dirs_after_first), (
            "Second download should not create new cache directories"
        )

    def test_same_data_same_cache_key(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Same dataset+version always produces the same cache key."""
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        bag1 = dataset.download_dataset_bag(version=version, use_minid=False)
        bag2 = dataset.download_dataset_bag(version=version, use_minid=False)

        # Both bags should resolve to the same dataset_rid
        assert bag1.dataset_rid == bag2.dataset_rid

        # Cache should show materialized (single entry, not duplicate)
        info = dataset.bag_info(version=version)
        assert info["status"] == CacheStatus.cached_materialized.value

    def test_new_version_creates_new_cache_entry(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """New dataset version creates a different cache entry (new snapshot)."""
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        v1 = dataset.current_version

        # Download v1
        bag1 = dataset.download_dataset_bag(version=v1, use_minid=False)

        # Create v2 by modifying data
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

        # Should have at least 2 cache entries now
        cache_dirs = list(ml.cache_dir.glob(f"{dataset.dataset_rid}_*"))
        assert len(cache_dirs) >= 2, (
            f"Expected >=2 cache entries for different versions, got {len(cache_dirs)}"
        )


class TestDeterministicCacheBagInfo:
    """Tests for bag_info interaction with deterministic cache."""

    def test_bag_info_before_download(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """bag_info shows not_cached before any download."""
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        info = dataset.bag_info(version=version)
        assert info["status"] == CacheStatus.not_cached.value

    def test_bag_info_after_deterministic_cache_hit(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """bag_info shows cached_materialized after deterministic cache is populated."""
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        # Populate cache
        dataset.download_dataset_bag(version=version, use_minid=False)

        # Check bag_info
        info = dataset.bag_info(version=version)
        assert info["status"] == CacheStatus.cached_materialized.value
        assert info["cache_path"] is not None

    def test_cache_warming_uses_deterministic_key(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """cache() warming creates deterministic cache entry."""
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        result = dataset.cache(version=version)
        assert result["status"] == CacheStatus.cached_materialized.value

        # Subsequent download should use the cached version
        bag = dataset.download_dataset_bag(version=version, use_minid=False)
        assert bag is not None


class TestDeterministicCacheInvalidation:
    """Tests for cache invalidation when spec or snapshot changes."""

    def test_delete_cache_forces_redownload(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """Deleting cache directory forces full re-download."""
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        # Download and verify cached
        dataset.download_dataset_bag(version=version, use_minid=False)
        info = dataset.bag_info(version=version)
        assert info["status"] == CacheStatus.cached_materialized.value

        # Delete all cache entries for this dataset
        for d in ml.cache_dir.glob(f"{dataset.dataset_rid}_*"):
            shutil.rmtree(d)

        # Verify cache is gone
        info = dataset.bag_info(version=version)
        assert info["status"] == CacheStatus.not_cached.value

        # Re-download succeeds
        bag = dataset.download_dataset_bag(version=version, use_minid=False)
        assert bag is not None
        info = dataset.bag_info(version=version)
        assert info["status"] == CacheStatus.cached_materialized.value


class TestDeterministicCacheUnit:
    """Unit tests for deterministic cache key computation (no catalog needed)."""

    def test_spec_hash_is_deterministic(self):
        """Same spec JSON produces same hash."""
        spec = {"tables": ["Image", "Subject"], "version": "1.0.0"}
        h1 = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        h2 = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        assert h1 == h2

    def test_different_spec_different_hash(self):
        """Different specs produce different hashes."""
        spec1 = {"tables": ["Image", "Subject"]}
        spec2 = {"tables": ["Image", "Subject", "Observation"]}
        h1 = hashlib.sha256(json.dumps(spec1, sort_keys=True).encode()).hexdigest()
        h2 = hashlib.sha256(json.dumps(spec2, sort_keys=True).encode()).hexdigest()
        assert h1 != h2

    def test_cache_key_format(self):
        """Deterministic cache key has expected format."""
        rid = "4SP"
        spec_hash = hashlib.sha256(b"test spec").hexdigest()
        snapshot = "34T-GGHM-6JBE"
        key = f"{rid}_{spec_hash[:16]}_{snapshot}"

        assert key.startswith("4SP_")
        assert snapshot in key
        # Should have three parts: rid, spec_hash prefix, snapshot
        parts = key.split("_", 1)
        assert parts[0] == rid
        assert len(parts[1]) > 16  # spec_hash[:16] + _ + snapshot

    def test_cache_key_changes_with_snapshot(self):
        """Different snapshots produce different cache keys."""
        rid = "4SP"
        spec_hash = hashlib.sha256(b"same spec").hexdigest()[:16]
        key1 = f"{rid}_{spec_hash}_snap1"
        key2 = f"{rid}_{spec_hash}_snap2"
        assert key1 != key2

    def test_cache_key_changes_with_spec(self):
        """Different spec hashes produce different cache keys."""
        rid = "4SP"
        snapshot = "snap1"
        h1 = hashlib.sha256(b"spec 1").hexdigest()[:16]
        h2 = hashlib.sha256(b"spec 2").hexdigest()[:16]
        key1 = f"{rid}_{h1}_{snapshot}"
        key2 = f"{rid}_{h2}_{snapshot}"
        assert key1 != key2


class TestStaleCacheInvalidation:
    """Regression tests for stale cache entries.

    When the schema changes (e.g., a new table is added to the FK traversal),
    the spec_hash changes even though the snapshot stays the same. The cache
    must NOT return bags created under the old spec_hash.

    This was a real bug: the Tier 1 glob matched on snapshot only
    ({rid}_*_{snapshot}), ignoring the spec_hash prefix. A bag missing
    newly added tables would be returned as a cache hit because its
    directory name ended with the matching snapshot suffix.
    """

    def test_stale_cache_not_returned_after_schema_change(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """A cached bag with old spec_hash is NOT returned when spec changes.

        Simulates the scenario:
        1. Download bag → creates cache entry with spec_hash_A
        2. Schema changes (new table added) → spec_hash_B
        3. Second download must NOT hit the cache from step 1
        """
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        # Step 1: Download to populate cache
        bag1 = dataset.download_dataset_bag(version=version, use_minid=False)

        # Verify cache entry exists
        cache_dirs = list(ml.cache_dir.glob(f"{dataset.dataset_rid}_*"))
        assert len(cache_dirs) >= 1
        original_cache_dir = cache_dirs[-1]  # Most recent
        original_name = original_cache_dir.name

        # Step 2: Create a fake stale cache entry with same snapshot but
        # different spec_hash (simulating what an old download would have
        # produced before a schema change)
        parts = original_name.split("_", 1)  # rid, rest
        rid = parts[0]
        # Extract the snapshot from the cache dir name
        # Format: {rid}_{spec_hash[:16]}_{snapshot}
        rest = parts[1]
        # spec_hash is 16 hex chars, then underscore, then snapshot
        old_spec_hash = rest[:16]
        snapshot_part = rest[17:]  # skip spec_hash + underscore

        # Create stale entry: different spec_hash, same snapshot
        stale_spec_hash = "0000000000000000"
        assert stale_spec_hash != old_spec_hash  # Must differ
        stale_dir = ml.cache_dir / f"{rid}_{stale_spec_hash}_{snapshot_part}"
        stale_bag = stale_dir / f"Dataset_{rid}"
        stale_bag.mkdir(parents=True, exist_ok=True)
        # Put a marker file so we can detect if this stale entry was returned
        (stale_bag / "STALE_MARKER").touch()

        # Step 3: Download again — must hit the CORRECT cache, not the stale one
        bag2 = dataset.download_dataset_bag(version=version, use_minid=False)

        # Verify the returned bag does NOT contain the stale marker
        assert not (bag2._catalog._database_model.bag_path / "STALE_MARKER").exists(), (
            "Stale cache entry was returned! The cache lookup matched on "
            "snapshot alone, ignoring the spec_hash difference."
        )

        # Clean up stale entry
        shutil.rmtree(stale_dir, ignore_errors=True)

    def test_snapshot_only_dir_not_matched(
        self, catalog_manager: CatalogManager, tmp_path: Path
    ):
        """A cache entry matching only the snapshot suffix is NOT returned.

        Ensures the fix works: the old glob pattern {rid}_*_{snapshot} would
        match, but the new exact lookup {rid}_{spec_hash[:16]}_{snapshot} does not.
        """
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        # Create a decoy cache entry with matching snapshot but wrong spec_hash
        history = dataset.dataset_history()
        version_record = next(
            v for v in history if v.dataset_version == str(version)
        )
        snapshot = version_record.snapshot
        decoy_dir = ml.cache_dir / f"{dataset.dataset_rid}_deadbeefdeadbeef_{snapshot}"
        decoy_bag = decoy_dir / f"Dataset_{dataset.dataset_rid}"
        decoy_bag.mkdir(parents=True, exist_ok=True)
        (decoy_bag / "DECOY").touch()

        # Download — should NOT return the decoy
        bag = dataset.download_dataset_bag(version=version, use_minid=False)
        assert not (bag._catalog._database_model.bag_path / "DECOY").exists(), (
            "Decoy cache entry with wrong spec_hash was returned!"
        )

        # Clean up
        shutil.rmtree(decoy_dir, ignore_errors=True)
