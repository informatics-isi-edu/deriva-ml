"""Tests for the execution state machine."""

from __future__ import annotations

import pytest

from deriva_ml.execution.state_machine import (
    ALLOWED_TRANSITIONS,
    InvalidTransitionError,
    validate_transition,
)
from deriva_ml.execution.state_store import ExecutionStatus


def test_allowed_transitions_cover_all_happy_paths():
    # created → running → stopped → pending_upload → uploaded
    assert (ExecutionStatus.created, ExecutionStatus.running) in ALLOWED_TRANSITIONS
    assert (ExecutionStatus.running, ExecutionStatus.stopped) in ALLOWED_TRANSITIONS
    assert (ExecutionStatus.stopped, ExecutionStatus.pending_upload) in ALLOWED_TRANSITIONS
    assert (ExecutionStatus.pending_upload, ExecutionStatus.uploaded) in ALLOWED_TRANSITIONS


def test_allowed_transitions_cover_failure_paths():
    assert (ExecutionStatus.running, ExecutionStatus.failed) in ALLOWED_TRANSITIONS
    assert (ExecutionStatus.pending_upload, ExecutionStatus.failed) in ALLOWED_TRANSITIONS


def test_allowed_transitions_cover_abort():
    # Abort legal from created, running, stopped, failed
    for start in [ExecutionStatus.created, ExecutionStatus.running,
                  ExecutionStatus.stopped, ExecutionStatus.failed]:
        assert (start, ExecutionStatus.aborted) in ALLOWED_TRANSITIONS


def test_retry_from_failed_back_to_pending_upload():
    # retry_failed → pending_upload is legal (upload retry path)
    assert (ExecutionStatus.failed, ExecutionStatus.pending_upload) in ALLOWED_TRANSITIONS


def test_validate_transition_accepts_allowed():
    validate_transition(
        current=ExecutionStatus.running,
        target=ExecutionStatus.stopped,
    )  # must not raise


def test_validate_transition_rejects_disallowed():
    with pytest.raises(InvalidTransitionError) as exc:
        validate_transition(
            current=ExecutionStatus.uploaded,
            target=ExecutionStatus.running,  # can't go back to running
        )
    msg = str(exc.value)
    assert "uploaded" in msg
    assert "running" in msg


def test_invalid_transition_error_is_deriva_ml_exception():
    from deriva_ml.core.exceptions import DerivaMLException
    assert issubclass(InvalidTransitionError, DerivaMLException)


def test_transition_writes_sqlite(tmp_path):
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_machine import transition
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.created,
        mode=ConnectionMode.offline, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    # Offline mode: no catalog argument, no sync attempt.
    transition(
        store=store,
        catalog=None,                # offline → skip catalog sync
        execution_rid="EXE-A",
        current=ExecutionStatus.created,
        target=ExecutionStatus.running,
        mode=ConnectionMode.offline,
        extra_fields={"start_time": now},
    )

    row = store.get_execution("EXE-A")
    assert row["status"] == "running"
    assert row["start_time"] is not None
    # Offline transitions always set sync_pending=True.
    assert row["sync_pending"] is True


def test_transition_rejects_invalid(tmp_path):
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_machine import (
        InvalidTransitionError, transition,
    )
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.uploaded,
        mode=ConnectionMode.offline, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    with pytest.raises(InvalidTransitionError):
        transition(
            store=store, catalog=None, execution_rid="EXE-A",
            current=ExecutionStatus.uploaded,
            target=ExecutionStatus.running,
            mode=ConnectionMode.offline,
        )


class _MockCatalog:
    """Minimal mock for ErmrestCatalog exposing just what transition() uses."""
    def __init__(self, *, put_should_fail: bool = False):
        self.put_should_fail = put_should_fail
        self.put_calls: list[dict] = []

    def put(self, path: str, json: object = None, **_kw):
        if self.put_should_fail:
            raise RuntimeError("simulated network failure")
        self.put_calls.append({"path": path, "json": json})
        return None


def test_online_transition_syncs_catalog(tmp_path):
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_machine import transition
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.created,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    cat = _MockCatalog()
    transition(
        store=store, catalog=cat, execution_rid="EXE-A",
        current=ExecutionStatus.created, target=ExecutionStatus.running,
        mode=ConnectionMode.online, extra_fields={"start_time": now},
    )

    row = store.get_execution("EXE-A")
    assert row["status"] == "running"
    # Online: sync succeeded, no pending flag.
    assert row["sync_pending"] is False
    # Catalog was put-updated.
    assert len(cat.put_calls) == 1
    body = cat.put_calls[0]["json"]
    assert isinstance(body, list) and len(body) == 1
    assert body[0]["RID"] == "EXE-A"
    assert body[0]["Status"] == "running"


def test_online_transition_soft_fails_on_catalog_error(tmp_path):
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_machine import transition
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.created,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    cat = _MockCatalog(put_should_fail=True)
    # SQLite transition must still succeed; the catalog failure is
    # soft — user gets sync_pending=True for the next pass to flush.
    transition(
        store=store, catalog=cat, execution_rid="EXE-A",
        current=ExecutionStatus.created, target=ExecutionStatus.running,
        mode=ConnectionMode.online,
    )

    row = store.get_execution("EXE-A")
    assert row["status"] == "running"
    assert row["sync_pending"] is True
