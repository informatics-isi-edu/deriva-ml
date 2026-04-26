"""Tests for ExecutionStateStore — SQLite-backed execution state."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect

from deriva_ml.execution.state_store import (
    EXECUTIONS_TABLE,
    PENDING_ROWS_TABLE,
    DIRECTORY_RULES_TABLE,
    ExecutionStateStore,
)


def _engine(tmp_path: Path):
    db = tmp_path / "test.db"
    return create_engine(f"sqlite:///{db}")


def test_table_name_constants_use_prefix():
    assert EXECUTIONS_TABLE == "execution_state__executions"
    assert PENDING_ROWS_TABLE == "execution_state__pending_rows"
    assert DIRECTORY_RULES_TABLE == "execution_state__directory_rules"


def test_ensure_schema_creates_three_tables(tmp_path):
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    inspector = inspect(eng)
    tables = set(inspector.get_table_names())
    assert EXECUTIONS_TABLE in tables
    assert PENDING_ROWS_TABLE in tables
    assert DIRECTORY_RULES_TABLE in tables


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
        "rid", "workflow_rid", "description", "config_json",
        "status", "mode", "working_dir_rel",
        "start_time", "stop_time", "last_activity",
        "error", "sync_pending", "created_at",
    }
    assert expected.issubset(cols), f"missing: {expected - cols}"


def test_pending_rows_table_columns(tmp_path):
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    inspector = inspect(eng)
    cols = {c["name"] for c in inspector.get_columns(PENDING_ROWS_TABLE)}
    expected = {
        "id", "execution_rid", "key",
        "target_schema", "target_table",
        "rid", "lease_token", "metadata_json",
        "asset_file_path", "asset_types_json", "description",
        "status", "error",
        "created_at", "leased_at", "uploaded_at",
        "rule_id",
    }
    assert expected.issubset(cols), f"missing: {expected - cols}"


def test_directory_rules_table_columns(tmp_path):
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    inspector = inspect(eng)
    cols = {c["name"] for c in inspector.get_columns(DIRECTORY_RULES_TABLE)}
    expected = {
        "id", "execution_rid",
        "target_schema", "target_table",
        "source_dir",
        "glob", "recurse", "copy_files",
        "asset_types_json", "status", "created_at",
    }
    assert expected.issubset(cols), f"missing: {expected - cols}"


def test_indexes_created(tmp_path):
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    inspector = inspect(eng)
    pending_indexes = {idx["name"] for idx in inspector.get_indexes(PENDING_ROWS_TABLE)}
    # Expect at least indexes for (execution_rid, status) and (execution_rid, target_table)
    assert any("execution_rid" in n and "status" in n for n in pending_indexes), \
        f"missing (execution_rid, status) index; have: {pending_indexes}"
    assert any("execution_rid" in n and "target_table" in n for n in pending_indexes), \
        f"missing (execution_rid, target_table) index; have: {pending_indexes}"


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
        # Tables must exist — ensure_schema ran.
        from sqlalchemy import inspect
        inspector = inspect(ws.engine)
        assert EXECUTIONS_TABLE in inspector.get_table_names()
        # Second call returns the same instance (cached).
        assert ws.execution_state_store() is store
    finally:
        ws.close()


def test_execution_status_values():
    from deriva_ml.execution.state_store import ExecutionStatus
    assert ExecutionStatus.Created.value == "Created"
    assert ExecutionStatus.Running.value == "Running"
    assert ExecutionStatus.Stopped.value == "Stopped"
    assert ExecutionStatus.Failed.value == "Failed"
    assert ExecutionStatus.Pending_Upload.value == "Pending_Upload"
    assert ExecutionStatus.Uploaded.value == "Uploaded"
    assert ExecutionStatus.Aborted.value == "Aborted"


def test_pending_row_status_values():
    from deriva_ml.execution.state_store import PendingRowStatus
    for name in ["staged", "leasing", "leased",
                 "uploading", "uploaded", "failed"]:
        assert getattr(PendingRowStatus, name).value == name


def test_directory_rule_status_values():
    from deriva_ml.execution.state_store import DirectoryRuleStatus
    assert DirectoryRuleStatus.active.value == "active"
    assert DirectoryRuleStatus.closed.value == "closed"


def test_insert_execution_row(tmp_path):
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

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
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Created,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
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
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

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
            rid=rid, workflow_rid=None, description=None,
            config_json="{}", status=status,
            mode=ConnectionMode.online, working_dir_rel=f"execution/{rid}",
            created_at=now, last_activity=now,
        )

    rows = store.list_executions(status=[ExecutionStatus.Running, ExecutionStatus.Stopped])
    rids = {r["rid"] for r in rows}
    assert rids == {"A", "B"}


def test_insert_pending_row(tmp_path):
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus, PendingRowStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)

    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    pending_id = store.insert_pending_row(
        execution_rid="EXE-A",
        key="k1",
        target_schema="deriva-ml",
        target_table="Subject",
        metadata_json='{"Name": "x"}',
        created_at=now,
    )
    assert isinstance(pending_id, int)
    assert pending_id > 0

    rows = store.list_pending_rows(execution_rid="EXE-A")
    assert len(rows) == 1
    assert rows[0]["status"] == str(PendingRowStatus.staged)
    assert rows[0]["target_table"] == "Subject"


def test_list_pending_rows_filter_by_status(tmp_path):
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus, PendingRowStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    id1 = store.insert_pending_row(
        execution_rid="EXE-A", key="k1",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )
    id2 = store.insert_pending_row(
        execution_rid="EXE-A", key="k2",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )

    store.update_pending_row(id1, status=PendingRowStatus.uploaded)
    staged = store.list_pending_rows(
        execution_rid="EXE-A", status=PendingRowStatus.staged,
    )
    assert {r["id"] for r in staged} == {id2}


def test_update_pending_rows_batch_applies_all_in_one_transaction(tmp_path):
    """Bulk variant flips every (id, fields) pair in one transaction.

    Regression test for the perf fix that replaced one
    ``engine.begin()`` per upload-engine status callback with a single
    batched transaction. On a 10K-asset upload this is the difference
    between thousands of WAL fsyncs and a handful (the callback flushes
    in chunks). The behavior must match the per-row method except for
    speed: arbitrary fields applied; PendingRowStatus enums coerced to
    strings just like ``update_pending_row``.
    """
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus, PendingRowStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )
    ids = []
    for i in range(3):
        ids.append(store.insert_pending_row(
            execution_rid="EXE-A", key=f"k{i}",
            target_schema="s", target_table="t",
            metadata_json="{}", created_at=now,
        ))

    upload_time = datetime.now(timezone.utc)
    store.update_pending_rows_batch([
        (ids[0], {"status": PendingRowStatus.uploaded, "uploaded_at": upload_time}),
        (ids[1], {"status": PendingRowStatus.uploaded, "uploaded_at": upload_time}),
        (ids[2], {"status": PendingRowStatus.failed, "error": "boom"}),
    ])

    rows = {r["id"]: r for r in store.list_pending_rows(execution_rid="EXE-A")}
    assert rows[ids[0]]["status"] == str(PendingRowStatus.uploaded)
    assert rows[ids[0]]["uploaded_at"] is not None
    assert rows[ids[1]]["status"] == str(PendingRowStatus.uploaded)
    assert rows[ids[2]]["status"] == str(PendingRowStatus.failed)
    assert rows[ids[2]]["error"] == "boom"


def test_update_pending_rows_batch_empty_is_noop(tmp_path):
    """Empty list must not raise and must not cycle a transaction."""
    from deriva_ml.execution.state_store import ExecutionStateStore

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    store.update_pending_rows_batch([])  # contract: no-op


def test_update_pending_rows_batch_rejects_unknown_columns(tmp_path):
    """Validation runs before any write so a bad row doesn't half-commit.

    The per-row ``update_pending_row`` raises KeyError for unknown
    columns; the batch variant must do the same — and crucially, must
    raise *before* opening the transaction so a malformed row in a
    long batch doesn't leave half the upload in an inconsistent state.
    """
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus, PendingRowStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )
    pid = store.insert_pending_row(
        execution_rid="EXE-A", key="k0",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )

    with pytest.raises(KeyError):
        store.update_pending_rows_batch([
            (pid, {"status": PendingRowStatus.uploaded}),
            (pid, {"not_a_real_column": "oops"}),
        ])
    # The first entry must NOT have been applied — validation precedes
    # the transaction.
    rows = {r["id"]: r for r in store.list_pending_rows(execution_rid="EXE-A")}
    assert rows[pid]["status"] == str(PendingRowStatus.staged)


def test_insert_directory_rule(tmp_path):
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus, DirectoryRuleStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    rule_id = store.insert_directory_rule(
        execution_rid="EXE-A",
        target_schema="deriva-ml",
        target_table="Mask",
        source_dir="/tmp/masks",
        glob="*.png",
        recurse=False,
        copy_files=False,
        asset_types_json=None,
        created_at=now,
    )
    assert isinstance(rule_id, int)

    rules = store.list_directory_rules(execution_rid="EXE-A")
    assert len(rules) == 1
    assert rules[0]["status"] == str(DirectoryRuleStatus.active)


def test_count_pending_by_kind(tmp_path):
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus, PendingRowStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    # Two plain rows: one staged, one failed.
    id1 = store.insert_pending_row(
        execution_rid="EXE-A", key="k1",
        target_schema="s", target_table="Subject",
        metadata_json="{}", created_at=now,
    )
    id2 = store.insert_pending_row(
        execution_rid="EXE-A", key="k2",
        target_schema="s", target_table="Subject",
        metadata_json="{}", created_at=now,
    )
    store.update_pending_row(id2, status=PendingRowStatus.failed)

    # Two asset rows: one staged, one uploaded.
    id3 = store.insert_pending_row(
        execution_rid="EXE-A", key="f1",
        target_schema="s", target_table="Image",
        metadata_json="{}", created_at=now,
        asset_file_path="/tmp/a.png",
    )
    id4 = store.insert_pending_row(
        execution_rid="EXE-A", key="f2",
        target_schema="s", target_table="Image",
        metadata_json="{}", created_at=now,
        asset_file_path="/tmp/b.png",
    )
    store.update_pending_row(id4, status=PendingRowStatus.uploaded)

    counts = store.count_pending_by_kind(execution_rid="EXE-A")
    # Plain rows: 1 pending (staged/leasing/leased/uploading), 1 failed.
    # Asset rows: 1 pending, 0 failed (the uploaded one doesn't count).
    assert counts == {
        "pending_rows": 1,
        "failed_rows": 1,
        "pending_files": 1,
        "failed_files": 0,
    }


def test_mark_leasing_sets_token_and_status(tmp_path):
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus, PendingRowStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )
    pid = store.insert_pending_row(
        execution_rid="EXE-A", key="k1",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )

    store.mark_pending_leasing(pid, lease_token="TOKEN-1")
    row = store.list_pending_rows(execution_rid="EXE-A")[0]
    assert row["status"] == str(PendingRowStatus.leasing)
    assert row["lease_token"] == "TOKEN-1"


def test_finalize_lease_sets_rid_and_status(tmp_path):
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus, PendingRowStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )
    pid = store.insert_pending_row(
        execution_rid="EXE-A", key="k1",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )
    store.mark_pending_leasing(pid, lease_token="T1")

    store.finalize_pending_lease(lease_token="T1", assigned_rid="1-NEW")
    row = store.list_pending_rows(execution_rid="EXE-A")[0]
    assert row["status"] == str(PendingRowStatus.leased)
    assert row["rid"] == "1-NEW"


def test_revert_leasing_to_staged(tmp_path):
    """Crash recovery: leasing rows with no matching server lease
    revert to staged so the next attempt reissues them."""
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus, PendingRowStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )
    pid = store.insert_pending_row(
        execution_rid="EXE-A", key="k1",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )
    store.mark_pending_leasing(pid, lease_token="T-LOST")

    store.revert_pending_leasing(lease_token="T-LOST")
    row = store.list_pending_rows(execution_rid="EXE-A")[0]
    assert row["status"] == str(PendingRowStatus.staged)
    assert row["lease_token"] is None


def test_mark_pending_leasing_batch(tmp_path):
    """Bulk Phase 1 of leasing: every (pending_id, token) flips together.

    Regression test for the perf fix that replaced a per-row
    update_pending_row loop in lease_orchestrator with a single
    batched transaction. Crash semantics unchanged: the entire
    batch must commit before the catalog POST so a crash leaves
    every row in a recoverable state.
    """
    from datetime import datetime, timezone
    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus, PendingRowStatus,
    )

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )
    pids = []
    for i in range(3):
        pids.append(store.insert_pending_row(
            execution_rid="EXE-A", key=f"k{i}",
            target_schema="s", target_table="t",
            metadata_json="{}", created_at=now,
        ))

    items = [(pids[0], "T-A"), (pids[1], "T-B"), (pids[2], "T-C")]
    store.mark_pending_leasing_batch(items)

    rows = sorted(store.list_pending_rows(execution_rid="EXE-A"), key=lambda r: r["id"])
    for row, (_, expected_token) in zip(rows, items):
        assert row["status"] == str(PendingRowStatus.leasing)
        assert row["lease_token"] == expected_token


def test_mark_pending_leasing_batch_empty(tmp_path):
    """Empty list is a no-op."""
    from deriva_ml.execution.state_store import ExecutionStateStore
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    store.mark_pending_leasing_batch([])  # must not raise


def test_finalize_pending_leases_batch(tmp_path):
    """Bulk Phase 3 of leasing: every (token, rid) finalizes in one transaction.

    All rows share the leased_at timestamp captured at transaction
    entry, mirroring the wire batch they came from.
    """
    from datetime import datetime, timezone
    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus, PendingRowStatus,
    )

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )
    pids = [
        store.insert_pending_row(
            execution_rid="EXE-A", key=f"k{i}",
            target_schema="s", target_table="t",
            metadata_json="{}", created_at=now,
        )
        for i in range(3)
    ]
    # Phase 1: stage tokens
    store.mark_pending_leasing_batch(
        [(pids[0], "T1"), (pids[1], "T2"), (pids[2], "T3")]
    )

    # Phase 3: finalize in one batch
    store.finalize_pending_leases_batch(
        [("T1", "1-AAA"), ("T2", "1-BBB"), ("T3", "1-CCC")]
    )

    rows = sorted(store.list_pending_rows(execution_rid="EXE-A"), key=lambda r: r["id"])
    expected = [("1-AAA",), ("1-BBB",), ("1-CCC",)]
    for row, (rid,) in zip(rows, expected):
        assert row["status"] == str(PendingRowStatus.leased)
        assert row["rid"] == rid
        assert row["leased_at"] is not None


def test_finalize_pending_leases_batch_empty(tmp_path):
    """Empty list is a no-op."""
    from deriva_ml.execution.state_store import ExecutionStateStore
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    store.finalize_pending_leases_batch([])  # must not raise


# ─── Workspace-wide count_pending_rows ────────────────────────────────


def test_count_pending_rows_workspace_wide(tmp_path):
    """count_pending_rows counts non-terminal rows across ALL executions."""
    from datetime import datetime, timezone

    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus, PendingRowStatus,
    )

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    now = datetime.now(timezone.utc)

    # Two executions; insert rows in various states.
    for rid in ("EXE-A", "EXE-B"):
        store.insert_execution(
            rid=rid, workflow_rid=None, description=None,
            config_json="{}", status=ExecutionStatus.Created,
            mode=ConnectionMode.online, working_dir_rel=f"execution/{rid}",
            created_at=now, last_activity=now,
        )

    # EXE-A: 2 non-terminal (staged, leased), 1 uploaded (terminal).
    store.insert_pending_row(
        execution_rid="EXE-A", key="a1",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
        status=PendingRowStatus.staged,
    )
    store.insert_pending_row(
        execution_rid="EXE-A", key="a2",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
        status=PendingRowStatus.leased,
    )
    pid = store.insert_pending_row(
        execution_rid="EXE-A", key="a3",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
        status=PendingRowStatus.staged,
    )
    store.update_pending_row(pid, status=PendingRowStatus.uploaded)

    # EXE-B: 1 failed (non-terminal for upload_pending), 1 uploaded.
    store.insert_pending_row(
        execution_rid="EXE-B", key="b1",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
        status=PendingRowStatus.failed,
    )
    pid2 = store.insert_pending_row(
        execution_rid="EXE-B", key="b2",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
        status=PendingRowStatus.staged,
    )
    store.update_pending_row(pid2, status=PendingRowStatus.uploaded)

    # Total non-terminal: 2 (EXE-A staged+leased) + 1 (EXE-B failed) = 3.
    assert store.count_pending_rows() == 3


def test_count_pending_rows_empty_store(tmp_path):
    from deriva_ml.execution.state_store import ExecutionStateStore

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    assert store.count_pending_rows() == 0
