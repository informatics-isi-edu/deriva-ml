"""Tests for the SQLite-backed ExecutionRecord dataclass."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


def test_execution_record_has_registry_fields():
    from deriva_ml.execution.execution_record_v2 import ExecutionRecord
    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_store import ExecutionStatus

    now = datetime.now(timezone.utc)
    rec = ExecutionRecord(
        rid="EXE-A",
        workflow_rid="WFL-1",
        description="test",
        status=ExecutionStatus.Stopped,
        mode=ConnectionMode.online,
        working_dir_rel="execution/EXE-A",
        start_time=now,
        stop_time=now,
        last_activity=now,
        error=None,
        sync_pending=False,
        created_at=now,
        pending_rows=0,
        failed_rows=0,
        pending_files=0,
        failed_files=0,
    )
    assert rec.rid == "EXE-A"
    assert rec.status is ExecutionStatus.Stopped
    assert rec.mode is ConnectionMode.online
    assert rec.pending_rows == 0


def test_execution_record_is_frozen():
    from deriva_ml.execution.execution_record_v2 import ExecutionRecord
    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_store import ExecutionStatus

    now = datetime.now(timezone.utc)
    rec = ExecutionRecord(
        rid="X", workflow_rid=None, description=None,
        status=ExecutionStatus.Created, mode=ConnectionMode.online,
        working_dir_rel="execution/X",
        start_time=None, stop_time=None, last_activity=now,
        error=None, sync_pending=False, created_at=now,
        pending_rows=0, failed_rows=0, pending_files=0, failed_files=0,
    )
    with pytest.raises((AttributeError, TypeError)):
        rec.rid = "Y"


def test_from_row_constructs_from_sqlite_dict():
    from deriva_ml.execution.execution_record_v2 import ExecutionRecord
    from deriva_ml.core.connection_mode import ConnectionMode
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
    rec = ExecutionRecord.from_row(
        row,
        pending_rows=3, failed_rows=0,
        pending_files=1, failed_files=0,
    )
    assert rec.rid == "EXE-A"
    assert rec.status is ExecutionStatus.Stopped
    assert rec.mode is ConnectionMode.online
    assert rec.pending_rows == 3
    assert rec.pending_files == 1
