"""Integration tests for multi-anchor bag-cache scenarios (§3.E from
``docs/design/deriva-ml-audit-2026-05-phase2-dataset.md``).

What this file covers
---------------------

deriva-ml's :class:`~deriva_ml.dataset.bag_cache.BagCache` is a thin
wrapper over deriva-py's :class:`~deriva.bag.cache_index.BagCacheIndex`.
The wrapper has its own responsibilities — bag materialization,
``cache_status()`` semantics, FK cascade through the deriva-ml layer —
that are independent of the index itself. These tests pin the
**consumer-side** round-trip: a user who runs
``download_dataset_bag(rid=A)`` then ``download_dataset_bag(rid=B)``
on overlapping content (same checksum) gets ``CacheStatus.CACHED``
for both anchors, sees one ``bags`` row, and both anchors round-trip
through the deriva-ml surface.

Where the contract itself is pinned
-----------------------------------

The underlying contract — that ``BagCacheIndex.record()`` *accumulates*
anchors across calls rather than replacing them — is **not** pinned
here. It lives upstream in
``deriva-py/tests/deriva/bag/test_cache_index.py``:

* ``test_cache_index_record_accumulates_anchors``
* ``test_cache_index_record_dedupes_repeated_anchor``

That's the right home: the contract belongs to the layer that makes
the promise. If a deriva-py refactor breaks ``record()``, upstream CI
catches it before a release ships. These deriva-ml tests are
consumer-side coverage of how ``BagCache`` *uses* the contract, not
the contract pin itself.

History
-------

PR #143 originally introduced this file with ``xfail(strict=True)``
markers because the bug was reachable only at the deriva-ml layer and
upstream coverage did not yet exist. Deriva-py PR #254 (commit
``cc5b141``) shipped the fix together with upstream unit tests; this
file dropped the xfail markers in deriva-ml PR #146. With upstream
now carrying the contract pin, this file's framing is "integration",
not "regression for #142".

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
