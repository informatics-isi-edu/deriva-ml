"""Tests for BagCache — cache status detection for downloaded dataset bags."""

import hashlib
import pytest
from pathlib import Path
from deriva_ml.dataset.bag_cache import (
    BagCache,
    CacheStatus,
)


def _make_valid_bag(bag_path: Path, files: dict[str, str] | None = None) -> None:
    """Create a minimal valid BDBag at the given path.

    Args:
        bag_path: Directory to create the bag in.
        files: Optional dict of {relative_path: content} for data files.
    """
    from bdbag import bdbag_api as bdb

    bag_path.mkdir(parents=True, exist_ok=True)
    data_dir = bag_path / "data"
    data_dir.mkdir(exist_ok=True)

    # Write data files
    if files:
        for rel_path, content in files.items():
            file_path = data_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)

    # Use bdbag to create a proper bag from the directory
    bdb.make_bag(str(bag_path), algs=["sha256"], idempotent=True)


class TestCacheStatus:
    """Test BagCache.cache_status with various cache states."""

    def test_not_cached(self, tmp_path):
        """Returns not_cached when no bag directory exists."""
        cache = BagCache(tmp_path)
        result = cache.cache_status("XXXX")
        assert result["status"] == CacheStatus.not_cached.value
        assert result["cache_path"] is None
        assert result["versions_cached"] == []

    def test_cached_metadata_only(self, tmp_path):
        """Returns cached_metadata_only when bag exists but no validated_check."""
        # Create a bag directory with fetch.txt containing unresolved entries
        bag_dir = tmp_path / "XXXX_abc123"
        bag_path = bag_dir / "Dataset_XXXX"
        bag_path.mkdir(parents=True)

        # Create a minimal bag structure
        (bag_path / "bagit.txt").write_text("BagIt-Version: 1.0\nTag-File-Character-Encoding: UTF-8\n")
        data_dir = bag_path / "data"
        data_dir.mkdir()

        # Create fetch.txt with an unresolved entry
        (bag_path / "fetch.txt").write_text("https://example.com/file.dat\t100\tdata/file.dat\n")

        # Create manifest (required for bag validation)
        (bag_path / "manifest-sha256.txt").write_text("")

        cache = BagCache(tmp_path)
        result = cache.cache_status("XXXX")
        assert result["status"] == CacheStatus.cached_metadata_only.value
        assert result["cache_path"] is not None
        assert "XXXX_abc123" in result["versions_cached"][0]

    def test_cached_materialized(self, tmp_path):
        """Returns cached_materialized when bag is fully downloaded."""
        bag_dir = tmp_path / "XXXX_abc123"
        bag_path = bag_dir / "Dataset_XXXX"
        _make_valid_bag(bag_path, files={"file.dat": "test data"})

        # Mark as validated
        (bag_dir / "validated_check.txt").touch()

        cache = BagCache(tmp_path)
        result = cache.cache_status("XXXX")
        assert result["status"] == CacheStatus.cached_materialized.value
        assert result["cache_path"] is not None

    def test_cached_incomplete(self, tmp_path):
        """Returns cached_incomplete when validated_check exists but files are missing."""
        bag_dir = tmp_path / "XXXX_abc123"
        bag_path = bag_dir / "Dataset_XXXX"
        bag_path.mkdir(parents=True)

        # Create bag structure
        (bag_path / "bagit.txt").write_text("BagIt-Version: 1.0\nTag-File-Character-Encoding: UTF-8\n")
        data_dir = bag_path / "data"
        data_dir.mkdir()
        (bag_path / "manifest-sha256.txt").write_text("")

        # fetch.txt references a file that doesn't exist
        (bag_path / "fetch.txt").write_text("https://example.com/missing.dat\t100\tdata/missing.dat\n")

        # Has validated_check but file is missing
        (bag_dir / "validated_check.txt").touch()

        cache = BagCache(tmp_path)
        result = cache.cache_status("XXXX")
        assert result["status"] == CacheStatus.cached_incomplete.value

    def test_multiple_cached_versions(self, tmp_path):
        """Returns info about all cached versions."""
        # Create two bag directories for the same dataset
        for checksum in ["abc123", "def456"]:
            bag_dir = tmp_path / f"XXXX_{checksum}"
            bag_path = bag_dir / "Dataset_XXXX"
            bag_path.mkdir(parents=True)
            (bag_path / "bagit.txt").write_text("BagIt-Version: 1.0\nTag-File-Character-Encoding: UTF-8\n")
            (bag_path / "data").mkdir()
            (bag_path / "manifest-sha256.txt").write_text("")

        cache = BagCache(tmp_path)
        result = cache.cache_status("XXXX")
        assert len(result["versions_cached"]) == 2


class TestIsFullyMaterialized:
    """Test the _is_fully_materialized static method."""

    def test_no_fetch_txt(self, tmp_path):
        """Bag without fetch.txt is fully materialized."""
        bag_path = tmp_path / "test_bag"
        _make_valid_bag(bag_path, files={"sample.txt": "hello"})

        assert BagCache._is_fully_materialized(bag_path) is True

    def test_bag_with_all_local_data(self, tmp_path):
        """Bag with all data files present locally is fully materialized."""
        bag_path = tmp_path / "test_bag"
        _make_valid_bag(bag_path, files={"file.dat": "content", "other.dat": "more data"})

        # No fetch.txt needed — all files are local
        assert BagCache._is_fully_materialized(bag_path) is True

    def test_missing_files(self, tmp_path):
        """Bag with fetch.txt where files are missing is NOT fully materialized."""
        bag_path = tmp_path / "test_bag"
        _make_valid_bag(bag_path)

        # Add a fetch.txt entry for a file that does NOT exist
        (bag_path / "fetch.txt").write_text("https://example.com/missing.dat\t100\tdata/missing.dat\n")

        assert BagCache._is_fully_materialized(bag_path) is False


class TestCacheStatusEnum:
    """The CacheStatus enum carries the StrEnum/back-compat invariants."""

    def test_holey_and_incomplete_alias_to_same_member(self):
        """``cached_incomplete`` resolves to the same member as ``cached_holey``.

        This is the back-compat path: pre-migration code used
        ``cached_incomplete``; the renamed term is ``cached_holey``
        (per CONTEXT.md vocabulary). StrEnum aliasing lets both
        names resolve to one member with the new wire value.
        """
        assert CacheStatus.cached_incomplete is CacheStatus.cached_holey
        # Value is the new term — clients writing the wire value get
        # the new name on disk.
        assert CacheStatus.cached_holey.value == "cached_holey"
        assert CacheStatus.cached_incomplete.value == "cached_holey"

    def test_str_enum_string_comparison(self):
        """StrEnum members compare equal to their string values."""
        # Callers like Dataset.bag_info compare against strings;
        # the StrEnum migration preserves that semantics.
        assert CacheStatus.cached_materialized == "cached_materialized"
        assert CacheStatus.not_cached == "not_cached"


class TestBagCacheIndexIntegration:
    """BagCache's cache_status reflects rows written through BagCacheIndex."""

    def test_recording_via_index_appears_in_cache_status(self, tmp_path):
        """A bag recorded in the index shows up via cache_status."""
        from deriva.bag.cache_index import BagCacheIndex

        # Set up the new-style on-disk layout: bags/{checksum}/bag
        cache = BagCache(tmp_path)
        try:
            checksum = "abc123"
            # Construct an index against the same cache_dir; both
            # BagCache and this BagCacheIndex point at the same
            # SQLite file by convention.
            with BagCacheIndex(tmp_path) as index:
                bag_dir = index.bag_dir_for(checksum) / "bag"
                _make_valid_bag(bag_dir)
                index.record(
                    checksum=checksum,
                    anchors=[("Dataset", "DS1")],
                )
            info = cache.cache_status("DS1")
            assert info["status"] in (
                CacheStatus.cached_materialized.value,
                CacheStatus.cached_holey.value,
            )
            assert info["cache_path"] is not None
            assert checksum in info["versions_cached"]
        finally:
            cache.dispose()


