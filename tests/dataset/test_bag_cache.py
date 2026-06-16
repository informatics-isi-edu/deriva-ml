"""Tests for BagCache — cache status detection for downloaded dataset bags."""

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


def _record_bag(cache_dir: Path, dataset_rid: str, checksum: str) -> Path:
    """Register a bag in the cache index and return its bag directory path.

    Post-cutover layout (docs/design/bag-client-cutover-2026-05.md):

    ``{cache_dir}/bags/{checksum}/Dataset_{rid}/``

    The caller is responsible for creating bag contents (bagit.txt, data/,
    manifest-*, fetch.txt) inside the returned directory.
    """
    from deriva.bag.cache_index import BagCacheIndex

    with BagCacheIndex(cache_dir) as index:
        bag_dir = index.bag_dir_for(checksum) / f"Dataset_{dataset_rid}"
        bag_dir.mkdir(parents=True, exist_ok=True)
        index.record(
            checksum=checksum,
            anchors=[("Dataset", dataset_rid)],
        )
    return bag_dir


class TestCacheStatus:
    """Test BagCache.cache_status with various cache states (post-cutover layout)."""

    def test_not_cached(self, tmp_path):
        """Returns not_cached when no bag is recorded in the index."""
        cache = BagCache(tmp_path)
        result = cache.cache_status("XXXX")
        assert result["status"] == CacheStatus.not_cached.value
        assert result["cache_path"] is None
        assert result["versions_cached"] == []

    def test_cached_materialized(self, tmp_path):
        """Index entry + on-disk bag with no missing files → cached_materialized."""
        bag_dir = _record_bag(tmp_path, dataset_rid="XXXX", checksum="abc123")
        _make_valid_bag(bag_dir, files={"file.dat": "test data"})

        cache = BagCache(tmp_path)
        result = cache.cache_status("XXXX")
        assert result["status"] == CacheStatus.cached_materialized.value
        assert result["cache_path"] is not None
        assert "abc123" in result["versions_cached"]

    def test_cached_holey(self, tmp_path):
        """Index entry + on-disk bag with unresolved fetch.txt entries → cached_holey."""
        bag_dir = _record_bag(tmp_path, dataset_rid="XXXX", checksum="abc123")

        # Build a minimal bag whose fetch.txt references a missing file.
        bag_dir.mkdir(parents=True, exist_ok=True)
        (bag_dir / "bagit.txt").write_text("BagIt-Version: 1.0\nTag-File-Character-Encoding: UTF-8\n")
        data_dir = bag_dir / "data"
        data_dir.mkdir(exist_ok=True)
        (bag_dir / "manifest-sha256.txt").write_text("")
        (bag_dir / "fetch.txt").write_text("https://example.com/missing.dat\t100\tdata/missing.dat\n")

        cache = BagCache(tmp_path)
        result = cache.cache_status("XXXX")
        assert result["status"] == CacheStatus.cached_holey.value

    def test_index_entry_without_bag_directory(self, tmp_path):
        """Index claims cached but bag directory missing → not_cached.

        Surfaces the case where someone deleted ``bags/{checksum}/`` by hand
        but the index entry remained. The status check reads the disk, not
        just the index.
        """
        # Record without creating the bag directory.
        from deriva.bag.cache_index import BagCacheIndex

        with BagCacheIndex(tmp_path) as index:
            index.record(checksum="abc123", anchors=[("Dataset", "XXXX")])

        cache = BagCache(tmp_path)
        result = cache.cache_status("XXXX")
        assert result["status"] == CacheStatus.not_cached.value

    def test_multiple_cached_versions(self, tmp_path):
        """Multiple cache entries for the same dataset surface in versions_cached."""
        for checksum in ("abc123", "def456"):
            bag_dir = _record_bag(tmp_path, dataset_rid="XXXX", checksum=checksum)
            _make_valid_bag(bag_dir)

        cache = BagCache(tmp_path)
        result = cache.cache_status("XXXX")
        assert set(result["versions_cached"]) == {"abc123", "def456"}


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

        Back-compat: callers writing ``cached_incomplete`` get the renamed
        ``cached_holey`` member. StrEnum aliasing keeps both names usable.
        """
        assert CacheStatus.cached_incomplete is CacheStatus.cached_holey
        assert CacheStatus.cached_holey.value == "cached_holey"
        assert CacheStatus.cached_incomplete.value == "cached_holey"

    def test_str_enum_string_comparison(self):
        """StrEnum members compare equal to their string values."""
        assert CacheStatus.cached_materialized == "cached_materialized"
        assert CacheStatus.not_cached == "not_cached"


class TestBagCacheIndexIntegration:
    """BagCache's cache_status reflects rows written through BagCacheIndex."""

    def test_recording_via_index_appears_in_cache_status(self, tmp_path):
        """A bag recorded in the index shows up via cache_status."""
        # Set up the post-cutover on-disk layout:
        # ``cache_dir/bags/{checksum}/Dataset_{rid}/``
        cache = BagCache(tmp_path)
        try:
            checksum = "abc123"
            bag_dir = _record_bag(tmp_path, dataset_rid="DS1", checksum=checksum)
            _make_valid_bag(bag_dir)

            info = cache.cache_status("DS1")
            assert info["status"] in (
                CacheStatus.cached_materialized.value,
                CacheStatus.cached_holey.value,
            )
            assert info["cache_path"] is not None
            assert checksum in info["versions_cached"]
        finally:
            cache.dispose()


class TestListBagsResilience:
    """list_bags must survive an unreadable bag dir rather than crashing the
    whole listing (regression: _dir_size caught only FileNotFoundError, so a
    permission-denied subdir inside a cached bag raised straight through)."""

    def test_list_bags_survives_permission_denied_bag(self, tmp_path, monkeypatch):
        # Two recorded bags with no size_bytes in the index, so list_bags falls
        # back to _dir_size(bag_dir) for each. Give each a real data file.
        bag_a = _record_bag(tmp_path, dataset_rid="AAAA", checksum="aaa111")
        bag_b = _record_bag(tmp_path, dataset_rid="BBBB", checksum="bbb222")
        (bag_a / "data.txt").write_text("ok")
        (bag_b / "data.txt").write_text("secret")

        # Make the size lookup for the SECOND bag's data file raise
        # PermissionError (a real unreadable file inside a cached bag). Pre-fix
        # this propagated straight through list_bags and crashed the listing.
        real_stat = Path.stat

        def fake_stat(self, *a, **k):
            if str(bag_b) in str(self) and self.name == "data.txt":
                raise PermissionError(13, "Permission denied", str(self))
            return real_stat(self, *a, **k)

        monkeypatch.setattr(Path, "stat", fake_stat)

        cache = BagCache(tmp_path)
        try:
            # Must NOT raise — returns both bags (the unreadable one included).
            bags = cache.list_bags()
            rids = {b.dataset_rid for b in bags}
            assert "AAAA" in rids
            assert "BBBB" in rids  # the unreadable bag is still listed
        finally:
            cache.dispose()
