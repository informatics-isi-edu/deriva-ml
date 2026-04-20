"""Tests for DerivaML execution registry API (list, find_incomplete,
resume, gc, create with kwargs)."""

from __future__ import annotations

from datetime import datetime, timezone


def _insert_test_execution(ws, rid, status, mode="online", workflow_rid=None):
    """Helper to insert an executions row without going through
    catalog.  Used only in unit tests."""
    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_store import ExecutionStatus

    store = ws.execution_state_store()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid=rid, workflow_rid=workflow_rid, description=f"test {rid}",
        config_json="{}",
        status=ExecutionStatus(status) if isinstance(status, str) else status,
        mode=ConnectionMode(mode) if isinstance(mode, str) else mode,
        working_dir_rel=f"execution/{rid}",
        created_at=now, last_activity=now,
    )


def test_list_executions_empty(test_ml):
    assert test_ml.list_executions() == []


def test_list_executions_returns_dataclass(test_ml):
    from deriva_ml.execution.execution_record_v2 import ExecutionRecord
    from deriva_ml.execution.state_store import ExecutionStatus

    _insert_test_execution(test_ml.workspace, "EXE-A", ExecutionStatus.stopped)

    rows = test_ml.list_executions()
    assert len(rows) == 1
    assert isinstance(rows[0], ExecutionRecord)
    assert rows[0].rid == "EXE-A"
    assert rows[0].status is ExecutionStatus.stopped


def test_list_executions_status_filter(test_ml):
    from deriva_ml.execution.state_store import ExecutionStatus

    _insert_test_execution(test_ml.workspace, "A", ExecutionStatus.running)
    _insert_test_execution(test_ml.workspace, "B", ExecutionStatus.uploaded)
    _insert_test_execution(test_ml.workspace, "C", ExecutionStatus.failed)

    incomplete = test_ml.list_executions(
        status=[ExecutionStatus.running, ExecutionStatus.failed],
    )
    rids = {r.rid for r in incomplete}
    assert rids == {"A", "C"}


def test_find_incomplete_executions(test_ml):
    from deriva_ml.execution.state_store import ExecutionStatus

    _insert_test_execution(test_ml.workspace, "A", ExecutionStatus.running)
    _insert_test_execution(test_ml.workspace, "B", ExecutionStatus.uploaded)
    _insert_test_execution(test_ml.workspace, "C", ExecutionStatus.stopped)
    _insert_test_execution(test_ml.workspace, "D", ExecutionStatus.aborted)
    _insert_test_execution(test_ml.workspace, "E", ExecutionStatus.pending_upload)

    rows = test_ml.find_incomplete_executions()
    rids = {r.rid for r in rows}
    # Incomplete = anything not terminally uploaded/aborted.
    # That's {created, running, stopped, failed, pending_upload}.
    # C is stopped, E is pending_upload, A is running.
    assert rids == {"A", "C", "E"}


def test_list_executions_carries_pending_counts(test_ml):
    from deriva_ml.execution.state_store import ExecutionStatus

    _insert_test_execution(test_ml.workspace, "EXE-A", ExecutionStatus.stopped)

    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)
    store.insert_pending_row(
        execution_rid="EXE-A", key="k1", target_schema="s",
        target_table="Subject", metadata_json="{}", created_at=now,
    )

    rows = test_ml.list_executions()
    assert len(rows) == 1
    assert rows[0].pending_rows == 1
    assert rows[0].pending_files == 0
