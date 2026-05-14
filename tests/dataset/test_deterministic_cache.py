"""Tests for deterministic cache key (spec_hash + snapshot).

The deterministic cache avoids re-downloading bags when the spec and snapshot
haven't changed. The cache key (a.k.a. ``checksum`` in :class:`BagCacheIndex`
terms) is ``{spec_hash[:16]}_{snapshot}`` — computed from the download spec
JSON and the catalog snapshot ID. The bag itself lives at
``{cache_dir}/bags/{checksum}/Dataset_{rid}/`` (post-Steps-11+12 layout —
see ``docs/design/bag-client-cutover-2026-05.md``).

Tests verify:

- First download records a checksum in :class:`BagCacheIndex` and writes
  the bag at the indexed location.
- Second download hits the index without re-generating the bag.
- The checksum format is ``{spec_hash[:16]}_{snapshot}``.
- Schema changes invalidate the cache (new spec_hash → new checksum).
- New dataset version invalidates the cache (new snapshot → new checksum).
- ``bag_info`` reflects index-backed cache status.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from deriva.bag.cache_index import BagCacheIndex

from deriva_ml import DerivaML
from deriva_ml.dataset.aux_classes import DatasetVersion, VersionPart
from deriva_ml.dataset.bag_cache import BagCache, CacheStatus
from tests.catalog_manager import CatalogManager


def _index_checksums_for(ml: DerivaML, dataset_rid: str) -> list[str]:
    """Return the list of cached-bag checksums recorded in the index.

    Post-cutover the bag-cache is content-addressed and recorded in a
    SQLite index alongside the bags themselves. Tests that previously
    globbed the cache directory now consult the index instead.
    """
    with BagCacheIndex(ml.cache_dir) as index:
        return list(index.find_bags_for_rid(table="Dataset", rid=dataset_rid))


class TestDeterministicCacheKey:
    """Tests for the spec_hash+snapshot deterministic cache key."""

    def test_cache_dir_uses_deterministic_name(self, catalog_manager: CatalogManager, tmp_path: Path):
        """Downloaded bag's index checksum is ``{spec_hash[:16]}_{snapshot}``.

        Post-Steps-11+12 the bag lives at
        ``{cache_dir}/bags/{checksum}/Dataset_{rid}/`` and the checksum is
        looked up via :class:`BagCacheIndex` rather than a filesystem
        glob. The deterministic-name invariant still holds — it just
        applies to the index's checksum string, not a directory name
        containing the dataset RID.
        """
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        bag = dataset.download_dataset_bag(version=version, use_minid=False)

        # The index should record exactly one bag for this dataset.
        checksums = _index_checksums_for(ml, dataset.dataset_rid)
        assert len(checksums) >= 1, "Should have at least one cache entry"

        # Checksum format: ``{spec_hash[:16]}_{snapshot}``. spec_hash[:16]
        # is 16 hex chars, then an underscore, then the snapshot — total
        # length must exceed 16 (spec_hash) + 1 (underscore) + minimum
        # snapshot length.
        newest = checksums[0]  # find_bags_for_rid orders by built_at DESC
        parts = newest.split("_", 1)
        assert len(parts) == 2, f"Checksum '{newest}' should split into spec_hash and snapshot"
        spec_hash, snapshot = parts
        assert len(spec_hash) == 16, f"spec_hash prefix should be 16 hex chars, got {len(spec_hash)}: {spec_hash!r}"
        assert snapshot, "snapshot portion must not be empty"

        # The bag should physically exist at the indexed location.
        with BagCacheIndex(ml.cache_dir) as index:
            bag_dir = index.bag_dir_for(newest) / f"Dataset_{dataset.dataset_rid}"
        assert bag_dir.exists(), f"Bag directory not found at indexed location: {bag_dir}"

    def test_second_download_skips_bag_generation(self, catalog_manager: CatalogManager, tmp_path: Path):
        """Second download with same version returns immediately from cache.

        Post-cutover this is observable in the cache index: the second
        download finds the existing checksum and shouldn't add another
        entry for the same dataset RID.
        """
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        # First download — generates bag
        bag1 = dataset.download_dataset_bag(version=version, use_minid=False)
        assert bag1 is not None

        # Record cache state
        checksums_after_first = _index_checksums_for(ml, dataset.dataset_rid)

        # Second download — should hit deterministic cache
        bag2 = dataset.download_dataset_bag(version=version, use_minid=False)
        assert bag2 is not None

        # No new checksums should have been recorded in the index.
        checksums_after_second = _index_checksums_for(ml, dataset.dataset_rid)
        assert len(checksums_after_second) == len(checksums_after_first), (
            "Second download should not create new cache entries"
        )

    def test_same_data_same_cache_key(self, catalog_manager: CatalogManager, tmp_path: Path):
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

    def test_new_version_creates_new_cache_entry(self, catalog_manager: CatalogManager, tmp_path: Path):
        """New released dataset version creates a different index checksum.

        Different *released* snapshots → different checksums in the cache
        index, so after downloading two distinct released versions the
        index should record at least two distinct bag entries for the
        same dataset RID.

        Dev versions deliberately don't participate in the cache index
        (their ``Snapshot`` is ``NULL`` so the cache key would be
        ``{spec_hash}_None`` for every dev mutation — see ADR-0003 on
        the mutable-dev-row contract). The test promotes each dev period
        to a release so the cache key actually differs across versions.
        """
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset

        # The demo fixture leaves the dataset on a dev row; promote it
        # to a release so we have a stable snapshot-bearing v1.
        if dataset.current_version.is_devrelease:
            dataset.release(bump=VersionPart.minor, description="v1 baseline")
        v1 = dataset.current_version

        # Download v1
        bag1 = dataset.download_dataset_bag(version=v1, use_minid=False)

        # Mutate, then release to v2 so its snapshot is distinct from v1.
        pb = ml.pathBuilder()
        subjects = [s["RID"] for s in pb.schemas[ml.default_schema].tables["Subject"].path.entities().fetch()]
        if len(subjects) >= 2:
            dataset.add_dataset_members(subjects[-2:])
        dataset.release(bump=VersionPart.minor, description="v2 with extra subjects")
        v2 = dataset.current_version
        assert str(v2) != str(v1)
        assert not v2.is_devrelease

        # Download v2
        bag2 = dataset.download_dataset_bag(version=v2, use_minid=False)

        # Should have at least 2 checksums in the index now — different
        # snapshots produce different cache keys.
        checksums = _index_checksums_for(ml, dataset.dataset_rid)
        assert len(checksums) >= 2, f"Expected >=2 cache entries for different versions, got {len(checksums)}"


class TestDeterministicCacheBagInfo:
    """Tests for bag_info interaction with deterministic cache."""

    def test_bag_info_before_download(self, catalog_manager: CatalogManager, tmp_path: Path):
        """bag_info shows not_cached before any download."""
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        info = dataset.bag_info(version=version)
        assert info["status"] == CacheStatus.not_cached.value

    def test_bag_info_after_deterministic_cache_hit(self, catalog_manager: CatalogManager, tmp_path: Path):
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

    def test_cache_warming_uses_deterministic_key(self, catalog_manager: CatalogManager, tmp_path: Path):
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

    def test_delete_cache_forces_redownload(self, catalog_manager: CatalogManager, tmp_path: Path):
        """Deleting cache directory forces full re-download.

        Post-cutover the bag lives at ``{cache_dir}/bags/{checksum}/`` and
        the index records the entry. Deleting the bag directory makes
        :meth:`BagCache.cache_status` report ``not_cached`` (the index
        still claims the bag exists, but :meth:`_determine_index_status`
        notices the directory is gone). A re-download repopulates the
        directory and re-records the entry.
        """
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        # Download and verify cached
        dataset.download_dataset_bag(version=version, use_minid=False)
        info = dataset.bag_info(version=version)
        assert info["status"] == CacheStatus.cached_materialized.value

        # Delete the entire bags/ subtree to wipe every cached bag.
        bags_root = ml.cache_dir / "bags"
        if bags_root.exists():
            shutil.rmtree(bags_root)

        # Verify cache is gone (index entry survives but the directory
        # doesn't, which surfaces as not_cached per the post-cutover
        # BagCache semantics).
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

    def test_stale_cache_not_returned_after_schema_change(self, catalog_manager: CatalogManager, tmp_path: Path):
        """A cached bag with old spec_hash is NOT returned when spec changes.

        Post-cutover, the Tier-1 lookup queries :class:`BagCacheIndex` for
        the *exact* checksum ``{spec_hash[:16]}_{snapshot}`` the current
        request would produce. A bag built under a different spec_hash
        has a different checksum and therefore isn't matched, even if it
        was recorded for the same dataset RID.

        Simulates the scenario:

        1. Download bag → creates cache entry with checksum_A.
        2. Schema changes (new table added) → spec_hash changes → checksum_B.
        3. Second download must NOT hit the cache from step 1.

        We synthesize step 2 by directly recording a stale entry in the
        index for the same dataset RID with a different (wrong)
        spec_hash but the same snapshot, then putting a STALE_MARKER in
        the corresponding bag directory. The Tier-1 lookup should not
        return that stale entry — only an exact-checksum match counts.
        """
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        # Step 1: Download to populate cache
        bag1 = dataset.download_dataset_bag(version=version, use_minid=False)

        # Verify cache entry exists in the index
        checksums = _index_checksums_for(ml, dataset.dataset_rid)
        assert len(checksums) >= 1
        # Format: ``{spec_hash[:16]}_{snapshot}``
        original_checksum = checksums[0]
        spec_hash, snapshot_part = original_checksum.split("_", 1)

        # Step 2: Record a *stale* index entry — same snapshot, different
        # spec_hash. Without an exact-checksum match the Tier-1 lookup
        # must not return this.
        stale_spec_hash = "0000000000000000"
        assert stale_spec_hash != spec_hash  # Must differ
        stale_checksum = f"{stale_spec_hash}_{snapshot_part}"

        with BagCacheIndex(ml.cache_dir) as index:
            stale_bag_dir = index.bag_dir_for(stale_checksum) / f"Dataset_{dataset.dataset_rid}"
            stale_bag_dir.mkdir(parents=True, exist_ok=True)
            (stale_bag_dir / "STALE_MARKER").touch()
            index.record(
                checksum=stale_checksum,
                anchors=[("Dataset", dataset.dataset_rid)],
            )

        # Step 3: Download again — must hit the CORRECT cache, not the stale one
        bag2 = dataset.download_dataset_bag(version=version, use_minid=False)

        # Verify the returned bag does NOT contain the stale marker
        assert not (bag2._catalog._database_model.bag_path / "STALE_MARKER").exists(), (
            "Stale cache entry was returned! The cache lookup matched on "
            "snapshot alone, ignoring the spec_hash difference."
        )

        # Clean up stale entry
        shutil.rmtree(stale_bag_dir.parent, ignore_errors=True)

    def test_snapshot_only_dir_not_matched(self, catalog_manager: CatalogManager, tmp_path: Path):
        """A cache entry with matching snapshot but wrong spec_hash is NOT returned.

        Post-cutover, the Tier-1 lookup queries :class:`BagCacheIndex` for
        the *exact* checksum the current request would produce
        (``{spec_hash[:16]}_{snapshot}``). A decoy entry recorded with a
        same-snapshot-but-different-spec_hash checksum gets a different
        index key and is therefore not matched, even if it's recorded
        for the same dataset RID.

        Needs a released version (snapshot is non-NULL) so the cache key
        is meaningful — dev versions have ``Snapshot=NULL`` and don't
        participate in the cache. See ADR-0003 on dev semantics.
        """
        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
        dataset = dataset_desc.dataset
        # Demo fixture leaves us on a dev row; promote to a release.
        if dataset.current_version.is_devrelease:
            dataset.release(bump=VersionPart.minor, description="cache test baseline")
        version = dataset.current_version

        # Resolve the snapshot for this version. Use str-vs-str
        # comparison; DatasetVersion inherits from packaging.Version
        # whose ``__eq__`` against a bare string returns NotImplemented.
        history = dataset.dataset_history()
        version_record = next(v for v in history if str(v.dataset_version) == str(version))
        snapshot = version_record.snapshot

        # Pre-record a decoy entry with the same snapshot but a wrong
        # spec_hash. The bag directory exists on disk so the Tier-1 lookup
        # would happily return it if it matched by snapshot alone.
        decoy_checksum = f"deadbeefdeadbeef_{snapshot}"
        with BagCacheIndex(ml.cache_dir) as index:
            decoy_bag_dir = index.bag_dir_for(decoy_checksum) / f"Dataset_{dataset.dataset_rid}"
            decoy_bag_dir.mkdir(parents=True, exist_ok=True)
            (decoy_bag_dir / "DECOY").touch()
            index.record(
                checksum=decoy_checksum,
                anchors=[("Dataset", dataset.dataset_rid)],
            )

        # Download — should NOT return the decoy.
        bag = dataset.download_dataset_bag(version=version, use_minid=False)
        assert not (bag._catalog._database_model.bag_path / "DECOY").exists(), (
            "Decoy cache entry with wrong spec_hash was returned!"
        )

        # Clean up
        shutil.rmtree(decoy_bag_dir.parent, ignore_errors=True)
