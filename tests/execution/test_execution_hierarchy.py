"""Tests for ExecutionRecord.list_execution_parents /
list_execution_children (renamed from list_parent_executions /
list_nested_executions)."""

from __future__ import annotations

from unittest.mock import MagicMock

from deriva_ml.execution.execution_record import ExecutionRecord
from deriva_ml.execution.state_store import ExecutionStatus


def test_list_execution_parents_symbol():
    """New name exists; old name is gone (R5.1 hard cutover)."""
    from deriva_ml.execution.execution_record import ExecutionRecord as LiveER

    assert hasattr(LiveER, "list_execution_parents"), "list_execution_parents should exist"
    assert hasattr(LiveER, "list_execution_children"), "list_execution_children should exist"
    assert not hasattr(LiveER, "list_parent_executions"), "list_parent_executions should be removed (R5.1 hard cutover)"
    assert not hasattr(LiveER, "list_nested_executions"), "list_nested_executions should be removed (R5.1 hard cutover)"


def _build_ml_instance_mock(rows: list[dict]) -> MagicMock:
    """Build a mock DerivaML whose pathBuilder() returns the supplied rows.

    Mocks just enough of the path-builder chain
    ``pb.schemas[ml_schema].Execution_Execution.filter(...).link(...).entities().fetch()``
    to return ``rows`` exactly. The mock also stubs out
    ``ml_schema`` (used as a dict key by the impl) and
    ``lookup_workflow``.
    """
    ml = MagicMock()
    ml.ml_schema = "deriva-ml"

    schema = MagicMock()
    exec_exec_path = MagicMock()
    execution_path = MagicMock()

    # The impl writes:
    #   exec_exec_path.filter(...).link(execution_path, on=...).entities().fetch()
    chain = MagicMock()
    chain.fetch.return_value = rows
    exec_exec_path.filter.return_value.link.return_value.entities.return_value = chain

    schema.Execution_Execution = exec_exec_path
    schema.Execution = execution_path

    pb = MagicMock()
    pb.schemas.__getitem__.return_value = schema
    ml.pathBuilder.return_value = pb

    # Workflow lookup returns a benign workflow object for any RID.
    ml.lookup_workflow.return_value = MagicMock(name="workflow")

    return ml


def test_list_execution_children_propagates_duration_fields():
    """Regression for B13: children yield durations populated from the row.

    Surfaced by e2e Phase 4 (catalog 46, multirun execution CTA →
    CVP/D14). Before fix, only execution_rid, workflow, status,
    description were passed to the child ExecutionRecord
    constructor — all three duration fields fell back to None
    even though the fetched row carried them. MCP
    `deriva_ml_list_execution_children` then reported null
    durations for every child. The singular
    `deriva_ml_get_execution` (which uses ``lookup_execution``)
    returned them correctly, confirming the bug is on the list
    path, not the catalog.
    """
    rows = [
        {
            "RID": "CHLA-AAAA",
            "Workflow": "WF-1",
            "Status": "Uploaded",
            "Description": "Job 0",
            "Execution_Duration": "0.0H 0.0min 0.6sec",
            "Download_Duration": "0.0H 0.0min 1.5sec",
            "Upload_Duration": "0.0H 0.0min 0.5sec",
        },
        {
            "RID": "CHLB-AAAA",
            "Workflow": "WF-1",
            "Status": "Uploaded",
            "Description": "Job 1",
            "Execution_Duration": "0.0H 0.0min 11.0sec",
            "Download_Duration": "0.0H 0.0min 1.8sec",
            "Upload_Duration": "0.0H 0.0min 0.8sec",
        },
    ]
    ml = _build_ml_instance_mock(rows)
    parent = ExecutionRecord(
        execution_rid="PRNT-AAAA",
        status=ExecutionStatus.Uploaded,
        _ml_instance=ml,
    )

    children = list(parent.list_execution_children())

    assert len(children) == 2
    assert children[0].execution_rid == "CHLA-AAAA"
    assert children[0].duration == "0.0H 0.0min 0.6sec"
    assert children[0].download_duration == "0.0H 0.0min 1.5sec"
    assert children[0].upload_duration == "0.0H 0.0min 0.5sec"
    assert children[1].execution_rid == "CHLB-AAAA"
    assert children[1].duration == "0.0H 0.0min 11.0sec"
    assert children[1].download_duration == "0.0H 0.0min 1.8sec"
    assert children[1].upload_duration == "0.0H 0.0min 0.8sec"


def test_list_execution_parents_propagates_duration_fields():
    """Regression for B13 (parents side).

    Symmetric to ``list_execution_children`` — both methods build
    ExecutionRecord instances from the same fetched-row shape and
    suffered the same field-drop bug.
    """
    rows = [
        {
            "RID": "PRNT-AAAA",
            "Workflow": "WF-1",
            "Status": "Uploaded",
            "Description": "Multirun parent",
            "Execution_Duration": "0.0H 0.0min 17.1sec",
            "Download_Duration": "0.0H 0.0min 1.1sec",
            "Upload_Duration": "0.0H 0.0min 0.13sec",
        },
    ]
    ml = _build_ml_instance_mock(rows)
    child = ExecutionRecord(
        execution_rid="CHLA-AAAA",
        status=ExecutionStatus.Uploaded,
        _ml_instance=ml,
    )

    parents = list(child.list_execution_parents())

    assert len(parents) == 1
    assert parents[0].execution_rid == "PRNT-AAAA"
    assert parents[0].duration == "0.0H 0.0min 17.1sec"
    assert parents[0].download_duration == "0.0H 0.0min 1.1sec"
    assert parents[0].upload_duration == "0.0H 0.0min 0.13sec"


def test_list_execution_children_handles_missing_duration_columns():
    """Pre-PR-2 catalog rows lack the three duration columns.

    The forward-only migration leaves old rows with the columns
    present but null (or, on a hypothetically-older catalog,
    missing entirely). Either way, the yielded ExecutionRecord
    should construct cleanly with None defaults — no KeyError
    when the column is absent, no validation error when the
    value is None.
    """
    rows = [
        {
            "RID": "OLDA-AAAA",
            "Workflow": "WF-1",
            "Status": "Uploaded",
            "Description": "Pre-PR-2 row",
            # No duration columns at all.
        },
    ]
    ml = _build_ml_instance_mock(rows)
    parent = ExecutionRecord(
        execution_rid="PRNT-AAAA",
        status=ExecutionStatus.Uploaded,
        _ml_instance=ml,
    )

    children = list(parent.list_execution_children())

    assert len(children) == 1
    assert children[0].duration is None
    assert children[0].download_duration is None
    assert children[0].upload_duration is None
