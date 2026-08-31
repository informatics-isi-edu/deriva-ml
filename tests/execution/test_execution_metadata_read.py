"""Unit tests for the Execution_Metadata read API (issue #371).

Offline: dict-backed pathBuilder mocks. The API's contract is the
server-side join — ``Execution_Metadata_Execution(Execution=rid)`` linked
to ``Execution_Metadata`` — never a stream of the execution's output
assets, and absence (zero rows) is a normal empty result while an outer
query failure propagates.

RIDs are programmatically generated; assertions compare only values that
flowed through the code under test.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deriva_ml.execution._helpers import list_execution_metadata


def _rid(n: int) -> str:
    return f"1-{n:04X}"


def _ml_with_metadata_rows(rows: list[dict], lookup_failures: set[str] = frozenset()):
    """DerivaML stand-in whose metadata join returns ``rows``."""
    ml = MagicMock()
    ml.ml_schema = "deriva-ml"

    assoc = MagicMock()
    linked = MagicMock()
    linked.entities.return_value.fetch.return_value = list(rows)
    assoc.filter.return_value.link.return_value = linked

    metadata_table = MagicMock()
    schema = MagicMock()
    schema.tables = {
        "Execution_Metadata_Execution": assoc,
        "Execution_Metadata": metadata_table,
    }
    pb = MagicMock()
    pb.schemas = {"deriva-ml": schema}
    ml.pathBuilder.return_value = pb

    def fake_lookup_asset(rid):
        if rid in lookup_failures:
            from deriva_ml.core.exceptions import DerivaMLException

            raise DerivaMLException(f"asset {rid} unreadable")
        asset = MagicMock()
        asset.asset_rid = rid
        return asset

    ml.lookup_asset.side_effect = fake_lookup_asset
    return ml, assoc


def test_returns_asset_per_metadata_row():
    rows = [{"RID": _rid(1), "Filename": "uv.lock"}, {"RID": _rid(2), "Filename": "snap.txt"}]
    ml, assoc = _ml_with_metadata_rows(rows)
    out = list_execution_metadata(ml_instance=ml, execution_rid=_rid(500))
    assert [a.asset_rid for a in out] == [_rid(1), _rid(2)]
    assert assoc.filter.call_count == 1  # one server-side join, no asset streaming


def test_absence_is_empty_list_not_error():
    ml, _ = _ml_with_metadata_rows([])
    assert list_execution_metadata(ml_instance=ml, execution_rid=_rid(500)) == []


def test_outer_query_failure_propagates():
    """A catalog problem must not read as 'no metadata recorded'."""
    ml, assoc = _ml_with_metadata_rows([])
    assoc.filter.side_effect = ConnectionError("catalog unreachable")
    with pytest.raises(ConnectionError):
        list_execution_metadata(ml_instance=ml, execution_rid=_rid(500))


def test_per_row_lookup_failure_degrades_keeping_others():
    rows = [{"RID": _rid(1)}, {"RID": _rid(2)}, {"RID": _rid(3)}]
    ml, _ = _ml_with_metadata_rows(rows, lookup_failures={_rid(2)})
    out = list_execution_metadata(ml_instance=ml, execution_rid=_rid(500))
    assert [a.asset_rid for a in out] == [_rid(1), _rid(3)]


def test_execution_record_surface():
    """ExecutionRecord.list_metadata() delegates to the helper."""
    from deriva_ml.execution.execution_record import ExecutionRecord

    rows = [{"RID": _rid(1), "Filename": "uv.lock"}]
    ml, _ = _ml_with_metadata_rows(rows)
    record = ExecutionRecord.model_construct(execution_rid=_rid(500))
    record._ml_instance = ml
    out = record.list_metadata()
    assert [a.asset_rid for a in out] == [_rid(1)]


def test_per_row_transport_failure_propagates():
    """Only DerivaMLException degrades per-row; a transport error on
    lookup_asset propagates (codex P2: an outage hitting every lookup must
    not return [] and misreport metadata as 'not recorded')."""
    rows = [{"RID": _rid(1)}]
    ml, _ = _ml_with_metadata_rows(rows)
    ml.lookup_asset.side_effect = ConnectionError("catalog unreachable")
    with pytest.raises(ConnectionError):
        list_execution_metadata(ml_instance=ml, execution_rid=_rid(500))
