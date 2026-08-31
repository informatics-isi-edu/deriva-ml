"""Query-count + shape pins for ExecutionMixin._execution_summaries (#367).

Offline. RIDs are programmatically generated; the mock pathBuilder records
filter() calls so the tests assert request counts, not URL syntax.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from deriva_ml.core.mixins.execution import _SUMMARY_CHUNK_SIZE, ExecutionMixin


def _rid(n: int) -> str:
    return f"1-{n:04X}"


class _Recorder:
    """Table mock: records filter() calls, returns scripted rows."""

    def __init__(self, rows_by_rid: dict):
        self.rows_by_rid = rows_by_rid
        self.filter_calls = 0
        # Predicate operands: table.RID == rid must yield an object carrying
        # the rid so fetch can resolve which rows were requested. A MagicMock
        # __eq__ returning a tagged object does this.
        self.RID = MagicMock()
        self._requested: list[list[str]] = []
        self.RID.__eq__ = lambda _self, other: _Pred([other])

    def filter(self, pred):
        self.filter_calls += 1
        self._requested.append(pred.rids)
        result = MagicMock()
        rows = [self.rows_by_rid[r] for r in pred.rids if r in self.rows_by_rid]
        result.entities.return_value.fetch.return_value = rows
        return result


class _Pred:
    def __init__(self, rids):
        self.rids = list(rids)

    def __or__(self, other):
        return _Pred(self.rids + other.rids)


def _mixin(exec_rows: dict, wf_rows: dict):
    ml = ExecutionMixin.__new__(ExecutionMixin)
    ml.ml_schema = "deriva-ml"
    exec_table = _Recorder(exec_rows)
    wf_table = _Recorder(wf_rows)
    schema = MagicMock()
    schema.tables = {"Execution": exec_table, "Workflow": wf_table}
    pb = MagicMock()
    pb.schemas = {"deriva-ml": schema}
    ml.pathBuilder = lambda: pb
    return ml, exec_table, wf_table


def _exec_row(n: int, wf_n: int | None = None) -> dict:
    return {
        "RID": _rid(n),
        "Description": f"exec {n}",
        "Status": "Completed",
        "Workflow": _rid(wf_n) if wf_n is not None else None,
    }


def test_empty_and_all_none_issue_zero_queries():
    ml, exec_table, wf_table = _mixin({}, {})
    assert ml._execution_summaries([]) == {}
    assert ml._execution_summaries([None, None]) == {}
    assert exec_table.filter_calls == 0
    assert wf_table.filter_calls == 0


def test_duplicates_dedupe_to_one_lookup_entry():
    row = _exec_row(1)
    ml, exec_table, _ = _mixin({row["RID"]: row}, {})
    out = ml._execution_summaries([row["RID"], row["RID"], row["RID"]])
    assert exec_table.filter_calls == 1
    assert list(out) == [row["RID"]]


def test_chunk_boundary_two_requests():
    rows = {(_r := _exec_row(i))["RID"]: _r for i in range(_SUMMARY_CHUNK_SIZE + 1)}
    ml, exec_table, _ = _mixin(rows, {})
    out = ml._execution_summaries(list(rows))
    assert exec_table.filter_calls == 2
    assert len(out) == _SUMMARY_CHUNK_SIZE + 1


def test_workflow_names_batched_not_per_execution():
    wf = {"RID": _rid(900), "Name": "training workflow"}
    rows = {(_r := _exec_row(i, wf_n=900))["RID"]: _r for i in range(1, 4)}
    ml, exec_table, wf_table = _mixin(rows, {wf["RID"]: wf})
    out = ml._execution_summaries(list(rows))
    assert wf_table.filter_calls == 1  # 3 executions, 1 distinct workflow, 1 query
    assert all(s.workflow is not None and s.workflow.name == "training workflow" for s in out.values())


def test_unresolvable_rid_absent_from_result():
    row = _exec_row(1)
    ml, _, _ = _mixin({row["RID"]: row}, {})
    out = ml._execution_summaries([row["RID"], _rid(999)])
    assert _rid(999) not in out
    assert row["RID"] in out
