"""Round-trip tests for multi-anchor bag-cache scenarios (§3.E from
``docs/design/deriva-ml-audit-2026-05-phase2-dataset.md``).

Background
----------

After the deriva.bag write-side cutover, every cached bag is keyed by
its BDBag checksum. The reverse index ``bag_anchor_rids`` answers
"which bags reference this RID?". The audit asked us to test the
following user story explicitly:

    User runs ``download_dataset_bag(rid=A)`` then
    ``download_dataset_bag(rid=B)`` on **overlapping content** (i.e.,
    the export produces the same checksum). The audit's hypothesis:
    "they will be storing one bag, not two".

Findings from this test file
----------------------------

The hypothesis is *half-true* against the current
:func:`deriva.bag.cache_index.BagCacheIndex.record` implementation
(deriva-py): there is exactly **one row** in the ``bags`` table — but
``record()`` *replaces* the anchor list on every call (its docstring
is explicit: "the most recent producer's claim wins"), so the first
anchor RID is silently erased.

The practical consequence in :class:`deriva_ml.dataset.bag_cache.BagCache`:

- After ``download(rid=A)`` then ``download(rid=B)``:
  - ``cache_status("A")`` returns ``not_cached``.
  - ``cache_status("B")`` returns ``cached_materialized`` (or
    ``cached_holey`` depending on fetch.txt).
- A second invocation of ``download(rid=A)`` cannot reuse the cached
  bytes via the index lookup; it has to re-extract.

The test class :class:`TestMultiAnchorRoundTripPinsCurrentBehaviour`
pins this *current* behaviour so any future change to
``BagCacheIndex.record`` semantics (e.g. switching to anchor
accumulation) is caught explicitly. The test class
:class:`TestMultiAnchorRoundTripDesiredBehaviour` is marked
``xfail(strict=True)`` and documents the **desired** behaviour the
audit's user story implies — both RID lookups should succeed after
the round-trip. When the upstream semantics change, that test will
flip to passing and CI will demand we flip the xfail off.

This file deliberately tests :class:`BagCache` and
:class:`BagCacheIndex` directly (no live catalog) so the round-trip
story is testable in unit-test time and on CI without ``DERIVA_HOST``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
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
# Current behaviour: anchor replacement on re-record
# ---------------------------------------------------------------------------


class TestMultiAnchorRoundTripPinsCurrentBehaviour:
    """Pin the current ``BagCacheIndex.record`` semantics.

    These tests describe what the cache layer **does today** for the
    A→B round-trip the audit asked about. They are *not* the desired
    behaviour — they exist so any upstream change that fixes the
    anchor-erasure problem trips an obvious test failure and forces
    a deliberate sweep of dependent code in deriva-ml.
    """

    def test_second_record_with_same_checksum_replaces_anchors(self, tmp_path):
        """Re-recording the same checksum with a new anchor erases the old anchor.

        Surfaces the upstream contract documented in
        :meth:`BagCacheIndex.record`: "if the anchor RIDs differ for
        the same checksum, the most recent producer's claim wins".
        """
        with BagCacheIndex(tmp_path) as index:
            index.record(checksum="deadbeef", anchors=[("Dataset", "A")])
            assert index.find_bags_for_rid(table="Dataset", rid="A") == ["deadbeef"]

            index.record(checksum="deadbeef", anchors=[("Dataset", "B")])
            # B wins; A is gone.
            assert index.find_bags_for_rid(table="Dataset", rid="B") == ["deadbeef"]
            assert index.find_bags_for_rid(table="Dataset", rid="A") == []

            # And there is exactly one bag row — the storage is shared.
            assert len(index.list_bags()) == 1

    def test_cache_status_after_round_trip_loses_first_anchor(self, tmp_path):
        """``cache_status("A")`` is ``not_cached`` after ``download(B)`` lands.

        End-to-end view of the same issue through the deriva-ml-facing
        :class:`BagCache` surface. After a second download with the
        same checksum but a different anchor RID, the first dataset's
        ``cache_status`` cannot find the bag.
        """
        checksum = "deadbeef"
        # First download: rid=A.
        bag_dir_a = _register(tmp_path, checksum=checksum, rid="A")
        _make_valid_bag(bag_dir_a)

        # Sanity: A is cached at this point.
        cache = BagCache(tmp_path)
        try:
            assert cache.cache_status("A")["status"] == CacheStatus.cached_materialized.value
        finally:
            cache.dispose()

        # Second download: rid=B, same checksum (overlapping content).
        bag_dir_b = _register(tmp_path, checksum=checksum, rid="B")
        _make_valid_bag(bag_dir_b)

        cache = BagCache(tmp_path)
        try:
            # B is now cached…
            assert cache.cache_status("B")["status"] == CacheStatus.cached_materialized.value
            # …but A's reverse-index entry was clobbered.
            assert cache.cache_status("A")["status"] == CacheStatus.not_cached.value
        finally:
            cache.dispose()

    def test_one_bag_row_per_checksum_even_after_multiple_downloads(self, tmp_path):
        """Storage is content-addressed: same checksum → one ``bags`` row.

        This part of the audit's hypothesis ("storing one bag, not
        two") *is* correct — only the reverse-index claim about both
        RIDs being findable is broken.
        """
        checksum = "deadbeef"
        _register(tmp_path, checksum=checksum, rid="A")
        _register(tmp_path, checksum=checksum, rid="B")
        _register(tmp_path, checksum=checksum, rid="C")

        with BagCacheIndex(tmp_path) as index:
            rows = index.list_bags()
        assert len(rows) == 1
        assert rows[0]["checksum"] == checksum


# ---------------------------------------------------------------------------
# Desired behaviour: anchor accumulation (audit's user story)
# ---------------------------------------------------------------------------


class TestMultiAnchorRoundTripDesiredBehaviour:
    """The behaviour the audit's user story implies.

    Marked ``xfail(strict=True)`` because today's
    :meth:`BagCacheIndex.record` replaces anchors. When the upstream
    semantics change (or when deriva-ml grows a merge wrapper that
    accumulates anchors before delegating to ``record``), these tests
    will start passing and ``strict=True`` will demand we drop the
    xfail.

    Tracked in https://github.com/informatics-isi-edu/deriva-ml/issues/142
    (decision between upstream deriva-py fix and a deriva-ml merge wrapper
    lives there).
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BagCacheIndex.record currently replaces anchors on re-record; "
            "see TestMultiAnchorRoundTripPinsCurrentBehaviour and "
            "https://github.com/informatics-isi-edu/deriva-ml/issues/142. "
            "Flip this to passing once anchor accumulation lands."
        ),
    )
    def test_both_rids_resolve_after_round_trip(self, tmp_path):
        """``download(A); download(B)`` → both ``cache_status`` calls succeed.

        The audit's literal user story: "they will be storing one
        bag, not two". The current implementation stores one bag but
        only the most-recently-downloaded RID resolves.
        """
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
