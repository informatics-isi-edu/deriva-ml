"""Unit tests for _validate_pending_asset_leases."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _fake_catalog(found_rids: set[str]):
    """Build a fake ErmrestCatalog whose .get() returns rows with
    RIDs from ``found_rids`` that match the query's filter."""
    catalog = MagicMock()

    def fake_get(path: str):
        # Parse "/entity/public:ERMrest_RID_Lease/RID=r1;RID=r2;..."
        response = MagicMock()
        assert path.startswith("/entity/public:ERMrest_RID_Lease/")
        filter_part = path.split("/entity/public:ERMrest_RID_Lease/", 1)[1]
        queried = []
        for clause in filter_part.split(";"):
            key, _, value = clause.partition("=")
            if key == "RID":
                queried.append(value)
        response.json.return_value = [
            {"RID": rid, "ID": f"token-{rid}"}
            for rid in queried
            if rid in found_rids
        ]
        return response

    catalog.get.side_effect = fake_get
    return catalog


def test_empty_entries_returns_none():
    from deriva_ml.execution.rid_lease import _validate_pending_asset_leases
    catalog = MagicMock()
    assert _validate_pending_asset_leases(catalog, []) is None
    catalog.get.assert_not_called()


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
    # Expect 2 calls: first batch of 2, second batch of 1.
    assert catalog.get.call_count == 2
