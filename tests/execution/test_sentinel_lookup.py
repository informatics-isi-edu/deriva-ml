"""Pins for the non-raising sentinel lookup + positive-only cache (#367)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.core.mixins.execution import ExecutionMixin


def _rid(n: int) -> str:
    return f"1-{n:04X}"


def _mixin_with_sentinel_rows(rows: list[dict]):
    ml = ExecutionMixin.__new__(ExecutionMixin)
    ml.ml_schema = "deriva-ml"
    exe = MagicMock()
    exe.filter.return_value.entities.return_value = iter(rows)
    schema = MagicMock()
    schema.Execution = exe
    pb = MagicMock()
    pb.schemas = {"deriva-ml": schema}
    ml.pathBuilder = lambda: pb
    return ml, exe


def test_absent_returns_none_without_raising():
    ml, _ = _mixin_with_sentinel_rows([])
    assert ml._sentinel_execution_rid_or_none() is None


def test_absence_is_not_cached():
    ml, exe = _mixin_with_sentinel_rows([])
    ml._sentinel_execution_rid_or_none()
    exe.filter.return_value.entities.return_value = iter([{"RID": _rid(7)}])
    assert ml._sentinel_execution_rid_or_none() == _rid(7)  # re-queried


def test_positive_result_cached_no_second_query():
    ml, exe = _mixin_with_sentinel_rows([{"RID": _rid(7)}])
    first = ml._sentinel_execution_rid_or_none()
    second = ml._sentinel_execution_rid_or_none()
    assert first == second == _rid(7)
    assert exe.filter.call_count == 1


def test_transport_error_propagates():
    ml, exe = _mixin_with_sentinel_rows([])
    exe.filter.side_effect = ConnectionError("catalog unreachable")
    with pytest.raises(ConnectionError):
        ml._sentinel_execution_rid_or_none()


def test_public_raising_wrapper_unchanged():
    ml, _ = _mixin_with_sentinel_rows([])
    with pytest.raises(DerivaMLException, match="sentinel"):
        ml.unknown_provenance_execution_rid()


def test_public_non_raising_accessor():
    """unknown_provenance_execution_rid_or_none: public, absence -> None (issue #372)."""
    ml, _ = _mixin_with_sentinel_rows([])
    assert ml.unknown_provenance_execution_rid_or_none() is None

    ml2, _ = _mixin_with_sentinel_rows([{"RID": _rid(7)}])
    assert ml2.unknown_provenance_execution_rid_or_none() == _rid(7)
