"""Tests for the upload engine. Builds up across G3–G8."""

from __future__ import annotations


def test_stage_plain_row_appends_pending(tmp_path):
    from datetime import datetime, timezone

    from sqlalchemy import create_engine

    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_store import (
        ExecutionStateStore,
        ExecutionStatus,
        PendingRowStatus,
    )

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A",
        workflow_rid=None,
        description=None,
        config_json="{}",
        status=ExecutionStatus.running,
        mode=ConnectionMode.online,
        working_dir_rel="execution/EXE-A",
        created_at=now,
        last_activity=now,
    )

    # Harness: stage a plain row. In Phase 2 this goes through
    # exe.table("Subject").insert({...}) — for now, direct store call.
    store.insert_pending_row(
        execution_rid="EXE-A",
        key="k1",
        target_schema="deriva-ml",
        target_table="Subject",
        metadata_json='{"Name": "Alice"}',
        created_at=now,
    )
    rows = store.list_pending_rows(
        execution_rid="EXE-A",
        status=PendingRowStatus.staged,
    )
    assert len(rows) == 1
    assert rows[0]["metadata_json"] == '{"Name": "Alice"}'
