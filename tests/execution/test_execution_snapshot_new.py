"""Temporary test for Task 1 of H3. Deleted in Task 3."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import BaseModel


def test_execution_snapshot_exists_and_is_pydantic():
    from deriva_ml.execution.execution_snapshot import ExecutionSnapshot
    assert issubclass(ExecutionSnapshot, BaseModel)


def test_execution_snapshot_is_frozen():
    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.execution_snapshot import ExecutionSnapshot
    from deriva_ml.execution.state_store import ExecutionStatus

    now = datetime.now(timezone.utc)
    snap = ExecutionSnapshot(
        rid="X", workflow_rid=None, description=None,
        status=ExecutionStatus.Created, mode=ConnectionMode.online,
        working_dir_rel="execution/X",
        start_time=None, stop_time=None, last_activity=now,
        error=None, sync_pending=False, created_at=now,
        pending_rows=0, failed_rows=0, pending_files=0, failed_files=0,
    )
    with pytest.raises(Exception):  # Pydantic raises ValidationError
        snap.rid = "Y"


def test_execution_snapshot_from_row():
    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.execution_snapshot import ExecutionSnapshot
    from deriva_ml.execution.state_store import ExecutionStatus

    now = datetime.now(timezone.utc)
    row = {
        "rid": "EXE-A",
        "workflow_rid": "WFL-1",
        "description": "test",
        "status": "Stopped",
        "mode": "online",
        "working_dir_rel": "execution/EXE-A",
        "start_time": now,
        "stop_time": now,
        "last_activity": now,
        "error": None,
        "sync_pending": False,
        "created_at": now,
        "config_json": "{}",
    }
    snap = ExecutionSnapshot.from_row(
        row,
        pending_rows=3, failed_rows=0,
        pending_files=1, failed_files=0,
    )
    assert snap.rid == "EXE-A"
    assert snap.status is ExecutionStatus.Stopped
    assert snap.mode is ConnectionMode.online
    assert snap.pending_rows == 3
    assert snap.pending_files == 1


def test_execution_snapshot_model_dump():
    """Pydantic conversion gives .model_dump()."""
    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.execution_snapshot import ExecutionSnapshot
    from deriva_ml.execution.state_store import ExecutionStatus

    now = datetime.now(timezone.utc)
    snap = ExecutionSnapshot(
        rid="X", workflow_rid=None, description=None,
        status=ExecutionStatus.Created, mode=ConnectionMode.online,
        working_dir_rel="execution/X",
        start_time=None, stop_time=None, last_activity=now,
        error=None, sync_pending=False, created_at=now,
        pending_rows=0, failed_rows=0, pending_files=0, failed_files=0,
    )
    d = snap.model_dump()
    assert d["rid"] == "X"
    assert d["status"] == "Created"
    assert d["mode"] == "online"
