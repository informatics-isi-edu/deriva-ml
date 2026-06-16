"""Unit tests for _validate_pending_asset_leases."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _fake_catalog(found_rids: set[str]):
    """Build a fake ErmrestCatalog whose datapath fetch returns rows
    with RIDs from ``found_rids`` that match ``RID.in_(<chunk>)``.

    Mirrors the implementation's call shape:

        pb = catalog.getPathBuilder()
        t = pb.schemas["public"].tables["ERMrest_RID_Lease"]
        rows = t.filter(t.RID.in_(chunk)).attributes(t.RID).fetch()

    The mock records every chunk passed to ``RID.in_(...)`` so call-count
    assertions still work; access via ``catalog._lease_fetch_calls``.
    """
    catalog = MagicMock()
    catalog._lease_fetch_calls = []  # list of chunks seen

    # The end of the datapath chain — .fetch() — needs to return
    # only the rows whose RID is in the chunk that was filtered.
    # We build a chain mock per fetch() so each call sees the right
    # chunk; using side_effect on .fetch makes that lookup work.

    def fetch_side_effect():
        # ``_current_chunk`` is set by filter() below before the chain
        # reaches .attributes().fetch().
        chunk = catalog._current_chunk
        return [{"RID": rid} for rid in chunk if rid in found_rids]

    def filter_side_effect(predicate):
        # predicate is t.RID.in_(<chunk>); MagicMock has recorded the
        # in_ call's args on the predicate object's parent. Extract
        # the chunk by re-running the path through call_args.
        # Simpler: have the test seed the chunk via the predicate's
        # MagicMock _mock_self closure — but that's brittle. Instead,
        # bind the chunk through a side_effect on RID.in_ itself.
        return catalog._chain

    def in_side_effect(chunk):
        catalog._current_chunk = list(chunk)
        catalog._lease_fetch_calls.append(list(chunk))
        return MagicMock()  # the in_(...) predicate object itself

    pb = MagicMock()
    catalog.getPathBuilder.return_value = pb

    lease_table = MagicMock()
    pb.schemas = {"public": MagicMock()}
    pb.schemas["public"].tables = {"ERMrest_RID_Lease": lease_table}

    # Chain: filter(pred).attributes(col).fetch() → rows
    chain = MagicMock()
    catalog._chain = chain
    lease_table.filter.side_effect = filter_side_effect
    chain.attributes.return_value = chain  # attributes() returns self
    chain.fetch.side_effect = fetch_side_effect

    # RID.in_(chunk) records the chunk and returns a predicate sentinel
    lease_table.RID.in_.side_effect = in_side_effect

    return catalog


def test_empty_entries_returns_none():
    from deriva_ml.execution.rid_lease import _validate_pending_asset_leases

    catalog = MagicMock()
    assert _validate_pending_asset_leases(catalog, []) is None
    # Empty entries should short-circuit before touching the datapath API.
    catalog.getPathBuilder.assert_not_called()


def test_all_leases_valid_passes():
    from deriva_ml.execution.rid_lease import _validate_pending_asset_leases

    catalog = _fake_catalog({"1-ABC", "1-DEF"})
    entries = [("Image/a.png", "1-ABC"), ("Image/b.png", "1-DEF")]
    assert _validate_pending_asset_leases(catalog, entries) is None


def test_single_missing_lease_raises():
    from deriva_ml.execution.rid_lease import _validate_pending_asset_leases
    from deriva_ml.core.exceptions import DerivaMLValidationError

    catalog = _fake_catalog({"1-ABC"})  # 1-DEF NOT there
    entries = [("Image/a.png", "1-ABC"), ("Image/b.png", "1-DEF")]
    with pytest.raises(DerivaMLValidationError) as ei:
        _validate_pending_asset_leases(catalog, entries)
    msg = str(ei.value)
    assert "Image/b.png" in msg
    assert "1-DEF" in msg
    assert "Image/a.png" not in msg  # the valid one isn't listed


def test_multiple_missing_leases_aggregated():
    from deriva_ml.execution.rid_lease import _validate_pending_asset_leases
    from deriva_ml.core.exceptions import DerivaMLValidationError

    catalog = _fake_catalog(set())  # nothing found
    entries = [
        ("Image/z.png", "1-ZZZ"),
        ("Image/a.png", "1-AAA"),
    ]
    with pytest.raises(DerivaMLValidationError) as ei:
        _validate_pending_asset_leases(catalog, entries)
    msg = str(ei.value)
    assert "1-ZZZ" in msg
    assert "1-AAA" in msg
    # Sorted by (key, rid) — 'a.png' before 'z.png'.
    assert msg.index("Image/a.png") < msg.index("Image/z.png")


def test_batched_queries_use_chunk_size(monkeypatch):
    """More entries than chunk size → multiple catalog.get calls."""
    from deriva_ml.execution import rid_lease
    from deriva_ml.execution.rid_lease import _validate_pending_asset_leases

    # Set chunk size to 2 so 3 entries require 2 batched calls.
    monkeypatch.setattr(rid_lease, "PENDING_ROWS_LEASE_CHUNK", 2)
    catalog = _fake_catalog({"1-A", "1-B", "1-C"})
    entries = [("k1", "1-A"), ("k2", "1-B"), ("k3", "1-C")]
    _validate_pending_asset_leases(catalog, entries)
    # Expect 2 chunks: first batch of 2, second batch of 1.
    assert len(catalog._lease_fetch_calls) == 2
    assert len(catalog._lease_fetch_calls[0]) == 2
    assert len(catalog._lease_fetch_calls[1]) == 1
