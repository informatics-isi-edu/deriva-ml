"""Round-trip tests for multi-anchor bag-cache scenarios (§3.E from
``docs/design/deriva-ml-audit-2026-05-phase2-dataset.md``).

Background
----------

Every cached bag is keyed by its BDBag checksum. The reverse index
``bag_anchor_rids`` (in deriva-py's
:class:`deriva.bag.cache_index.BagCacheIndex`) answers "which bags
reference this RID?". The audit asked us to test the following user
story explicitly:

    User runs ``download_dataset_bag(rid=A)`` then
    ``download_dataset_bag(rid=B)`` on **overlapping content** (i.e.,
    the export produces the same checksum). The audit's hypothesis:
    "they will be storing one bag, not two".

History
-------

When this test file was first added (PR #143), the user story was
**half-broken**: storage was correctly shared (one ``bags`` row), but
``BagCacheIndex.record()`` *replaced* the anchor list on every call,
so the first anchor RID was silently erased. The desired behaviour
was pinned with ``xfail(strict=True)`` while the fix was scoped
(see issue #142).

The fix landed upstream in deriva-py PR #254 (commit ``cc5b141``):
``record()`` now **accumulates** anchors via ``INSERT OR IGNORE`` on
the existing ``PRIMARY KEY (checksum, "table", rid)`` constraint.
This file's tests now pin the **fixed** behaviour positively: both
RIDs resolve after the A→B round trip, the bag row stays unique,
and the deriva-ml ``BagCache`` surface honours both anchors.

The tests run without ``DERIVA_HOST`` (pure-local
``BagCacheIndex`` / ``BagCache`` exercise).
"""

from __future__ import annotations

from pathlib import Path

from bdbag import bdbag_api as bdb
from deriva.bag.cache_index import BagCacheIndex

from deriva_ml.dataset.bag_cache import BagCache, CacheStatus


def _make_valid_bag(bag_path: Path) -> None:
    """Create a minimal valid BDBag at ``bag_path``.

    The bag has an empty ``data/`` directory and a real
    ``manifest-sha256.txt`` — enough for
    :meth:`BagCache._is_fully_materialized` to validate it.
    """
    bag_path.mkdir(parents=True, exist_ok=True)
    (bag_path / "data").mkdir(exist_ok=True)
    bdb.make_bag(str(bag_path), algs=["sha256"], idempotent=True)


def _register(cache_dir: Path, checksum: str, rid: str) -> Path:
    """Simulate ``download_dataset_bag(rid=rid)`` writing into the cache.

    The on-disk layout matches what
    :func:`deriva_ml.dataset.bag_download.download_dataset_minid`
    produces today: ``{cache}/bags/{checksum}/Dataset_{rid}/``.

    Returns the bag directory so callers can populate it with bytes.
    """
    with BagCacheIndex(cache_dir) as index:
        bag_dir = index.bag_dir_for(checksum) / f"Dataset_{rid}"
        bag_dir.mkdir(parents=True, exist_ok=True)
        index.record(checksum=checksum, anchors=[("Dataset", rid)])
    return bag_dir


# ---------------------------------------------------------------------------
# Reverse-index behaviour
# ---------------------------------------------------------------------------


class TestMultiAnchorAccumulation:
    """``BagCacheIndex.record`` accumulates anchors across re-records.

    Pins the post-#254 semantic at the deriva-py boundary: a single
    content-addressed bag may legitimately be anchored from multiple
    RIDs, and each ``record()`` call adds to the reverse index
    without dropping prior entries.
    """

    def test_second_record_with_same_checksum_keeps_prior_anchor(self, tmp_path):
        """``record(X, [A])`` then ``record(X, [B])`` → both A and B resolve."""
        with BagCacheIndex(tmp_path) as index:
            index.record(checksum="deadbeef", anchors=[("Dataset", "A")])
            assert index.find_bags_for_rid(table="Dataset", rid="A") == ["deadbeef"]

            index.record(checksum="deadbeef", anchors=[("Dataset", "B")])
            # Both anchors resolve; A was not erased.
            assert index.find_bags_for_rid(table="Dataset", rid="A") == ["deadbeef"]
            assert index.find_bags_for_rid(table="Dataset", rid="B") == ["deadbeef"]

            # And there is exactly one bag row — the storage is shared.
            assert len(index.list_bags()) == 1

    def test_repeated_anchor_is_deduped_silently(self, tmp_path):
        """Re-inserting an existing ``(checksum, table, rid)`` row is a no-op.

        Exercises the ``INSERT OR IGNORE`` guard. Without it, a re-record
        with an overlapping anchor would raise on the
        ``PRIMARY KEY (checksum, "table", rid)`` constraint.
        """
        with BagCacheIndex(tmp_path) as index:
            index.record(checksum="deadbeef", anchors=[("Dataset", "A")])
            index.record(checksum="deadbeef", anchors=[("Dataset", "A"), ("Dataset", "B")])
            assert index.find_bags_for_rid(table="Dataset", rid="A") == ["deadbeef"]
            assert index.find_bags_for_rid(table="Dataset", rid="B") == ["deadbeef"]

    def test_one_bag_row_per_checksum_across_many_downloads(self, tmp_path):
        """Three downloads of the same content → one bag row, three anchors."""
        checksum = "deadbeef"
        for rid in ("A", "B", "C"):
            _register(tmp_path, checksum=checksum, rid=rid)

        with BagCacheIndex(tmp_path) as index:
            rows = index.list_bags()
            assert len(rows) == 1
            assert rows[0]["checksum"] == checksum
            # All three anchors resolve.
            for rid in ("A", "B", "C"):
                assert index.find_bags_for_rid(table="Dataset", rid=rid) == [checksum]


# ---------------------------------------------------------------------------
# deriva-ml-facing BagCache surface
# ---------------------------------------------------------------------------


class TestMultiAnchorRoundTripViaBagCache:
    """The audit's literal user story, exercised through :class:`BagCache`.

    ``download(rid=A)`` then ``download(rid=B)`` on overlapping content
    leaves both RIDs cached. Storage is shared; both
    :meth:`BagCache.cache_status` calls report ``cached_materialized``.
    """

    def test_both_rids_resolve_after_round_trip(self, tmp_path):
        """End-to-end: A→B round trip → both cache_status calls succeed."""
        checksum = "deadbeef"
        bag_dir_a = _register(tmp_path, checksum=checksum, rid="A")
        _make_valid_bag(bag_dir_a)
        bag_dir_b = _register(tmp_path, checksum=checksum, rid="B")
        _make_valid_bag(bag_dir_b)

        cache = BagCache(tmp_path)
        try:
            assert cache.cache_status("A")["status"] == CacheStatus.cached_materialized.value
            assert cache.cache_status("B")["status"] == CacheStatus.cached_materialized.value
        finally:
            cache.dispose()
