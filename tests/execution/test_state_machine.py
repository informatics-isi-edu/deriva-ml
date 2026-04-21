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
    assert (ExecutionStatus.Created, ExecutionStatus.Running) in ALLOWED_TRANSITIONS
    assert (ExecutionStatus.Running, ExecutionStatus.Stopped) in ALLOWED_TRANSITIONS
    assert (ExecutionStatus.Stopped, ExecutionStatus.Pending_Upload) in ALLOWED_TRANSITIONS
    assert (ExecutionStatus.Pending_Upload, ExecutionStatus.Uploaded) in ALLOWED_TRANSITIONS


def test_allowed_transitions_cover_failure_paths():
    assert (ExecutionStatus.Running, ExecutionStatus.Failed) in ALLOWED_TRANSITIONS
    assert (ExecutionStatus.Pending_Upload, ExecutionStatus.Failed) in ALLOWED_TRANSITIONS


def test_allowed_transitions_cover_abort():
    # Abort legal from created, running, stopped, failed
    for start in [ExecutionStatus.Created, ExecutionStatus.Running,
                  ExecutionStatus.Stopped, ExecutionStatus.Failed]:
        assert (start, ExecutionStatus.Aborted) in ALLOWED_TRANSITIONS


def test_retry_from_failed_back_to_pending_upload():
    # retry_failed → pending_upload is legal (upload retry path)
    assert (ExecutionStatus.Failed, ExecutionStatus.Pending_Upload) in ALLOWED_TRANSITIONS


def test_validate_transition_accepts_allowed():
    validate_transition(
        current=ExecutionStatus.Running,
        target=ExecutionStatus.Stopped,
    )  # must not raise


def test_validate_transition_rejects_disallowed():
    with pytest.raises(InvalidTransitionError) as exc:
        validate_transition(
            current=ExecutionStatus.Uploaded,
            target=ExecutionStatus.Running,  # can't go back to running
        )
    msg = str(exc.value)
    assert "Uploaded" in msg
    assert "Running" in msg


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
        config_json="{}", status=ExecutionStatus.Created,
        mode=ConnectionMode.offline, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    # Offline mode: no catalog argument, no sync attempt.
    transition(
        store=store,
        catalog=None,                # offline → skip catalog sync
        execution_rid="EXE-A",
        current=ExecutionStatus.Created,
        target=ExecutionStatus.Running,
        mode=ConnectionMode.offline,
        extra_fields={"start_time": now},
    )

    row = store.get_execution("EXE-A")
    assert row["status"] == "Running"
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
        config_json="{}", status=ExecutionStatus.Uploaded,
        mode=ConnectionMode.offline, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    with pytest.raises(InvalidTransitionError):
        transition(
            store=store, catalog=None, execution_rid="EXE-A",
            current=ExecutionStatus.Uploaded,
            target=ExecutionStatus.Running,
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
        config_json="{}", status=ExecutionStatus.Created,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    cat = _MockCatalog()
    transition(
        store=store, catalog=cat, execution_rid="EXE-A",
        current=ExecutionStatus.Created, target=ExecutionStatus.Running,
        mode=ConnectionMode.online, extra_fields={"start_time": now},
    )

    row = store.get_execution("EXE-A")
    assert row["status"] == "Running"
    # Online: sync succeeded, no pending flag.
    assert row["sync_pending"] is False
    # Catalog was put-updated.
    assert len(cat.put_calls) == 1
    body = cat.put_calls[0]["json"]
    assert isinstance(body, list) and len(body) == 1
    assert body[0]["RID"] == "EXE-A"
    assert body[0]["Status"] == "Running"


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
        config_json="{}", status=ExecutionStatus.Created,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    cat = _MockCatalog(put_should_fail=True)
    # SQLite transition must still succeed; the catalog failure is
    # soft — user gets sync_pending=True for the next pass to flush.
    transition(
        store=store, catalog=cat, execution_rid="EXE-A",
        current=ExecutionStatus.Created, target=ExecutionStatus.Running,
        mode=ConnectionMode.online,
    )

    row = store.get_execution("EXE-A")
    assert row["status"] == "Running"
    assert row["sync_pending"] is True


def test_flush_pending_sync_pushes_catalog(tmp_path):
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_machine import flush_pending_sync, transition
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Created,
        mode=ConnectionMode.offline, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )
    # Do an offline transition: SQLite has sync_pending=True.
    transition(
        store=store, catalog=None, execution_rid="EXE-A",
        current=ExecutionStatus.Created, target=ExecutionStatus.Running,
        mode=ConnectionMode.offline,
    )
    assert store.get_execution("EXE-A")["sync_pending"] is True

    # Now flush it against a live (mock) catalog.
    cat = _MockCatalog()
    flush_pending_sync(store=store, catalog=cat, execution_rid="EXE-A")

    assert store.get_execution("EXE-A")["sync_pending"] is False
    assert len(cat.put_calls) == 1


def test_flush_pending_sync_noop_when_not_pending(tmp_path):
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_machine import flush_pending_sync
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Stopped,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
        sync_pending=False,
    )

    cat = _MockCatalog()
    flush_pending_sync(store=store, catalog=cat, execution_rid="EXE-A")
    assert len(cat.put_calls) == 0


class _MockCatalogWithGet(_MockCatalog):
    """Extends the mock with a configurable GET response."""
    def __init__(self, *, get_row: dict | None | str = None, **kw):
        super().__init__(**kw)
        # get_row: dict = returned row, None = 404/no row, "raise" = raise
        self._get_row = get_row

    def get(self, path: str, **_kw):
        if self._get_row == "raise":
            raise RuntimeError("simulated 500")
        class _R:
            def __init__(self, row):
                self._row = row
            def json(self):
                return [self._row] if self._row is not None else []
            status_code = 200
        return _R(self._get_row)


def test_reconcile_no_disagreement(tmp_path):
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.execution.state_machine import reconcile_with_catalog
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Stopped,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now, sync_pending=False,
    )

    cat = _MockCatalogWithGet(get_row={"RID": "EXE-A", "Status": "Stopped"})
    reconcile_with_catalog(store=store, catalog=cat, execution_rid="EXE-A")
    # Unchanged.
    assert store.get_execution("EXE-A")["status"] == "Stopped"


def test_reconcile_catalog_says_aborted(tmp_path):
    """SQLite=running, catalog=aborted → SQLite flips to aborted."""
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.execution.state_machine import reconcile_with_catalog
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now, sync_pending=False,
    )

    cat = _MockCatalogWithGet(get_row={"RID": "EXE-A", "Status": "Aborted"})
    reconcile_with_catalog(store=store, catalog=cat, execution_rid="EXE-A")
    assert store.get_execution("EXE-A")["status"] == "Aborted"


def test_reconcile_catalog_says_uploaded(tmp_path):
    """SQLite=pending_upload, catalog=uploaded → SQLite flips to uploaded."""
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.execution.state_machine import reconcile_with_catalog
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Pending_Upload,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now, sync_pending=False,
    )

    cat = _MockCatalogWithGet(get_row={"RID": "EXE-A", "Status": "Uploaded"})
    reconcile_with_catalog(store=store, catalog=cat, execution_rid="EXE-A")
    assert store.get_execution("EXE-A")["status"] == "Uploaded"


def test_reconcile_sqlite_stopped_catalog_running(tmp_path):
    """SQLite=stopped, catalog=running (stale) → push SQLite to catalog."""
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.execution.state_machine import reconcile_with_catalog
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Stopped,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now, sync_pending=False,
    )

    cat = _MockCatalogWithGet(get_row={"RID": "EXE-A", "Status": "Running"})
    reconcile_with_catalog(store=store, catalog=cat, execution_rid="EXE-A")
    # SQLite unchanged; sync_pending set so next flush pushes.
    assert store.get_execution("EXE-A")["status"] == "Stopped"
    assert store.get_execution("EXE-A")["sync_pending"] is True


def test_reconcile_catalog_missing_raises(tmp_path):
    """SQLite has the row; catalog doesn't → orphan, raise."""
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.core.exceptions import DerivaMLStateInconsistency
    from deriva_ml.execution.state_machine import reconcile_with_catalog
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Stopped,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    cat = _MockCatalogWithGet(get_row=None)
    with pytest.raises(DerivaMLStateInconsistency):
        reconcile_with_catalog(store=store, catalog=cat, execution_rid="EXE-A")


def test_reconcile_catalog_error_logs_and_returns(tmp_path, caplog):
    """Transient catalog error → reconcile skips cleanly (caller decides)."""
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.execution.state_machine import reconcile_with_catalog
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Stopped,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now, sync_pending=False,
    )

    cat = _MockCatalogWithGet(get_row="raise")
    import logging
    with caplog.at_level(logging.WARNING):
        reconcile_with_catalog(store=store, catalog=cat, execution_rid="EXE-A")
    # SQLite unchanged.
    assert store.get_execution("EXE-A")["status"] == "Stopped"
    assert any("reconcile" in r.message.lower() for r in caplog.records)


class _MockCatalogWithInsert(_MockCatalog):
    """Mock that records POSTs to Execution and returns a fake RID."""
    def __init__(self, *, assigned_rid: str = "EXE-NEW", **kw):
        super().__init__(**kw)
        self.assigned_rid = assigned_rid
        self.post_calls: list[dict] = []

    def post(self, path: str, json=None, **_kw):
        self.post_calls.append({"path": path, "json": json})
        class _R:
            def __init__(self, rid): self._rid = rid
            def json(self): return [{"RID": self._rid, **(json[0] if json else {})}]
            status_code = 201
        return _R(self.assigned_rid)


def test_create_catalog_execution_posts_and_returns_rid():
    from deriva_ml.execution.state_machine import create_catalog_execution

    cat = _MockCatalogWithInsert(assigned_rid="EXE-NEW")
    rid = create_catalog_execution(
        catalog=cat,
        workflow_rid="WFL-1",
        description="a test run",
    )
    assert rid == "EXE-NEW"
    assert len(cat.post_calls) == 1
    body = cat.post_calls[0]["json"]
    assert body[0]["Workflow"] == "WFL-1"
    assert body[0]["Description"] == "a test run"


def test_public_api_exported():
    import deriva_ml.execution.state_machine as sm
    expected = {
        "ALLOWED_TRANSITIONS",
        "InvalidTransitionError",
        "validate_transition",
        "transition",
        "flush_pending_sync",
        "reconcile_with_catalog",
        "create_catalog_execution",
    }
    assert expected.issubset(set(sm.__all__))
