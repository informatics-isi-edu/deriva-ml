"""Tests for ExecutionStateStore — SQLite-backed execution registry.

The pending-rows and directory-rules surface (with its enums, table
definitions, and ~12 CRUD methods) was retired in the Phase 3
cleanup per ``docs/design/deriva-ml-audit-2026-05-phase3-execution.md``
§1.5. Three reader methods —
:meth:`ExecutionStateStore.count_pending_rows`,
:meth:`~ExecutionStateStore.count_pending_by_kind`,
:meth:`~ExecutionStateStore.pending_summary_rows` — survive as no-op
stubs so the four production call sites keep compiling. The tests
below pin both the live executions CRUD and the no-op stubs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, inspect

from deriva_ml.core.connection_mode import ConnectionMode
from deriva_ml.execution.state_store import (
    EXECUTIONS_TABLE,
    ExecutionStateStore,
    ExecutionStatus,
)


def _engine(tmp_path: Path):
    db = tmp_path / "test.db"
    return create_engine(f"sqlite:///{db}")


def test_table_name_constant_uses_prefix():
    assert EXECUTIONS_TABLE == "execution_state__executions"


def test_ensure_schema_creates_executions_table(tmp_path):
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    inspector = inspect(eng)
    tables = set(inspector.get_table_names())
    assert EXECUTIONS_TABLE in tables


def test_ensure_schema_is_idempotent(tmp_path):
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    store.ensure_schema()  # second call must not raise


def test_executions_table_columns(tmp_path):
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    inspector = inspect(eng)
    cols = {c["name"] for c in inspector.get_columns(EXECUTIONS_TABLE)}
    expected = {
        "rid",
        "workflow_rid",
        "description",
        "config_json",
        "status",
        "mode",
        "working_dir_rel",
        "start_time",
        "stop_time",
        "last_activity",
        "error",
        "sync_pending",
        "created_at",
    }
    assert expected.issubset(cols), f"missing: {expected - cols}"


def test_executions_indexes_created(tmp_path):
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    inspector = inspect(eng)
    idxs = {idx["name"] for idx in inspector.get_indexes(EXECUTIONS_TABLE)}
    assert any("status" in n for n in idxs)
    assert any("workflow_rid" in n for n in idxs)
    assert any("last_activity" in n for n in idxs)


def test_workspace_exposes_execution_state_store(tmp_path):
    from deriva_ml.local_db.workspace import Workspace

    ws = Workspace(
        working_dir=tmp_path,
        hostname="test.example.org",
        catalog_id="1",
    )
    try:
        store = ws.execution_state_store()
        assert isinstance(store, ExecutionStateStore)
        # Table must exist — ensure_schema ran.
        inspector = inspect(ws.engine)
        assert EXECUTIONS_TABLE in inspector.get_table_names()
        # Second call returns the same instance (cached).
        assert ws.execution_state_store() is store
    finally:
        ws.close()


def test_execution_status_values():
    assert ExecutionStatus.Created.value == "Created"
    assert ExecutionStatus.Running.value == "Running"
    assert ExecutionStatus.Stopped.value == "Stopped"
    assert ExecutionStatus.Failed.value == "Failed"
    assert ExecutionStatus.Pending_Upload.value == "Pending_Upload"
    assert ExecutionStatus.Uploaded.value == "Uploaded"
    assert ExecutionStatus.Aborted.value == "Aborted"


def test_insert_execution_row(tmp_path):
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A",
        workflow_rid="WFL-1",
        description="test",
        config_json='{"foo": "bar"}',
        status=ExecutionStatus.Created,
        mode=ConnectionMode.online,
        working_dir_rel="execution/EXE-A",
        created_at=now,
        last_activity=now,
    )

    row = store.get_execution("EXE-A")
    assert row is not None
    assert row["rid"] == "EXE-A"
    assert row["status"] == "Created"
    assert row["mode"] == "online"


def test_get_execution_missing_returns_none(tmp_path):
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    assert store.get_execution("NOPE") is None


def test_update_execution_status(tmp_path):
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A",
        workflow_rid=None,
        description=None,
        config_json="{}",
        status=ExecutionStatus.Created,
        mode=ConnectionMode.online,
        working_dir_rel="execution/EXE-A",
        created_at=now,
        last_activity=now,
    )

    store.update_execution(
        rid="EXE-A",
        status=ExecutionStatus.Running,
        start_time=now,
        sync_pending=True,
    )

    row = store.get_execution("EXE-A")
    assert row["status"] == "Running"
    assert row["sync_pending"] is True
    assert row["start_time"] is not None


def test_list_executions_filters_by_status(tmp_path):
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    now = datetime.now(timezone.utc)
    for rid, status in [
        ("A", ExecutionStatus.Running),
        ("B", ExecutionStatus.Stopped),
        ("C", ExecutionStatus.Uploaded),
    ]:
        store.insert_execution(
            rid=rid,
            workflow_rid=None,
            description=None,
            config_json="{}",
            status=status,
            mode=ConnectionMode.online,
            working_dir_rel=f"execution/{rid}",
            created_at=now,
            last_activity=now,
        )

    rows = store.list_executions(status=[ExecutionStatus.Running, ExecutionStatus.Stopped])
    rids = {r["rid"] for r in rows}
    assert rids == {"A", "B"}


def test_delete_execution_removes_row(tmp_path):
    """``delete_execution`` removes the executions row; subsequent
    ``get_execution`` returns ``None``."""
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A",
        workflow_rid=None,
        description=None,
        config_json="{}",
        status=ExecutionStatus.Created,
        mode=ConnectionMode.online,
        working_dir_rel="execution/EXE-A",
        created_at=now,
        last_activity=now,
    )
    assert store.get_execution("EXE-A") is not None

    store.delete_execution("EXE-A")
    assert store.get_execution("EXE-A") is None


# ─── pending-rows reader stubs ─────────────────────────────────────


def test_count_pending_rows_always_zero(tmp_path):
    """``count_pending_rows`` is a vestigial stub — always returns 0.

    The pending-rows write surface was retired in Phase 3 cleanup
    (audit §1.5). The function remains so ``core/base.py``'s
    schema-refresh guard keeps compiling.
    """
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    assert store.count_pending_rows() == 0


def test_count_pending_by_kind_always_zeros(tmp_path):
    """``count_pending_by_kind`` is a vestigial stub — always zeros.

    Production callers (``find_executions``, ``Execution.pending_summary``)
    keep working; they truthfully see no pending work.
    """
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    counts = store.count_pending_by_kind(execution_rid="EXE-IRRELEVANT")
    assert counts == {
        "pending_rows": 0,
        "failed_rows": 0,
        "pending_files": 0,
        "failed_files": 0,
    }


def test_pending_summary_rows_always_empty(tmp_path):
    """``pending_summary_rows`` is a vestigial stub — always empty rollup.

    The ``PendingSummary`` rendering path
    (``Execution.pending_summary``, ``DerivaML.pending_summary``)
    keeps working; every execution is reported as having nothing
    pending.
    """
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    assert store.pending_summary_rows(execution_rid="EXE-IRRELEVANT") == {
        "rows": [],
        "assets": [],
        "diagnostics": [],
    }
