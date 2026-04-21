"""Tests for the new Execution.update_status / ExecutionRecord.update_status API."""

from __future__ import annotations

import pytest


def _make_workflow(test_ml, name: str):
    from deriva_ml import MLVocab as vc

    test_ml.add_term(
        vc.workflow_type,
        "Test Workflow",
        description="for update_status tests",
    )
    return test_ml.create_workflow(
        name=name,
        workflow_type="Test Workflow",
        description="for update_status tests",
    )


def test_execution_update_status_valid_transition(test_ml):
    """Exe.update_status transitions via the state machine and persists to SQLite."""
    from deriva_ml.execution.state_store import ExecutionStatus

    wf = _make_workflow(test_ml, "B1 valid transition")
    exe = test_ml.create_execution(description="valid", workflow=wf)
    store = test_ml.workspace.execution_state_store()

    # Explicit transition Created → Running
    exe.update_status(ExecutionStatus.Running)

    row = store.get_execution(exe.execution_rid)
    assert row["status"] == "Running"


def test_execution_update_status_invalid_transition_raises(test_ml):
    """Violating ALLOWED_TRANSITIONS raises InvalidTransitionError."""
    from deriva_ml.execution.state_machine import InvalidTransitionError
    from deriva_ml.execution.state_store import ExecutionStatus

    wf = _make_workflow(test_ml, "B1 invalid")
    exe = test_ml.create_execution(description="invalid", workflow=wf)
    # Created → Uploaded is not an allowed direct transition.
    with pytest.raises(InvalidTransitionError):
        exe.update_status(ExecutionStatus.Uploaded)


def test_execution_update_status_error_kwarg_on_failed(test_ml):
    """error='...' is written to the error column on Failed."""
    from deriva_ml.execution.state_store import ExecutionStatus

    wf = _make_workflow(test_ml, "B1 error kwarg")
    exe = test_ml.create_execution(description="err", workflow=wf)
    exe.update_status(ExecutionStatus.Running)
    exe.update_status(ExecutionStatus.Failed, error="Network timeout")

    store = test_ml.workspace.execution_state_store()
    row = store.get_execution(exe.execution_rid)
    assert row["status"] == "Failed"
    assert row["error"] == "Network timeout"


def test_execution_update_status_error_kwarg_on_nonterminal_warns(test_ml, caplog):
    """error='...' on a non-terminal transition logs a warning; proceeds normally."""
    import logging
    from deriva_ml.execution.state_store import ExecutionStatus

    wf = _make_workflow(test_ml, "B1 warn")
    exe = test_ml.create_execution(description="warn", workflow=wf)
    with caplog.at_level(logging.WARNING):
        exe.update_status(ExecutionStatus.Running, error="this should warn")
    assert any(
        "error= ignored on non-terminal" in rec.message
        or "non-terminal transition" in rec.message
        for rec in caplog.records
    )
    store = test_ml.workspace.execution_state_store()
    row = store.get_execution(exe.execution_rid)
    assert row["status"] == "Running"
    # The error column is NOT populated for non-terminal transitions.
    assert row["error"] is None


def test_record_update_status_transitions(test_ml):
    """ExecutionRecord.update_status(target, *, ml, error=None) parallel."""
    from deriva_ml.execution.state_store import ExecutionStatus

    wf = _make_workflow(test_ml, "B2 record")
    exe = test_ml.create_execution(description="rec", workflow=wf)
    rec = next(r for r in test_ml.list_executions() if r.rid == exe.execution_rid)

    rec.update_status(ExecutionStatus.Running, ml=test_ml)
    store = test_ml.workspace.execution_state_store()
    row = store.get_execution(exe.execution_rid)
    assert row["status"] == "Running"

    rec.update_status(ExecutionStatus.Failed, ml=test_ml, error="boom")
    row = store.get_execution(exe.execution_rid)
    assert row["status"] == "Failed"
    assert row["error"] == "boom"
