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


def test_resume_execution_reads_from_sqlite(test_ml, monkeypatch):
    from deriva_ml.execution.state_store import ExecutionStatus

    _insert_test_execution(test_ml.workspace, "EXE-A", ExecutionStatus.stopped)

    # Isolate SQLite-read behavior: test_ml is online, but the
    # catalog does not have a matching Execution row. Stub reconcile
    # to keep this test focused on the registry read.
    monkeypatch.setattr(
        "deriva_ml.core.mixins.execution.reconcile_with_catalog",
        lambda *, store, catalog, execution_rid: None,
    )

    exe = test_ml.resume_execution("EXE-A")
    assert exe.execution_rid == "EXE-A"


def test_resume_execution_missing_raises(test_ml):
    from deriva_ml.core.exceptions import DerivaMLException

    import pytest
    with pytest.raises(DerivaMLException) as exc:
        test_ml.resume_execution("EXE-NOPE")
    assert "EXE-NOPE" in str(exc.value)


def test_resume_execution_offline_skips_reconcile(catalog_manager, tmp_path):
    """Offline resume must not contact the server."""
    from deriva_ml import ConnectionMode, DerivaML
    from deriva_ml.execution.state_store import ExecutionStatus

    catalog_manager.reset()
    ml = DerivaML(
        catalog_manager.hostname,
        catalog_manager.catalog_id,
        default_schema=catalog_manager.domain_schema,
        working_dir=tmp_path,
        use_minid=False,
        mode=ConnectionMode.offline,
    )
    _insert_test_execution(
        ml.workspace, "EXE-A", ExecutionStatus.stopped, mode="offline",
    )

    # Just must not raise — offline reconcile is a no-op.
    exe = ml.resume_execution("EXE-A")
    assert exe.execution_rid == "EXE-A"


def test_resume_execution_online_flushes_sync_pending(test_ml, monkeypatch):
    """If sync_pending=True on resume (online), flush to catalog."""
    from deriva_ml.execution.state_store import ExecutionStatus

    _insert_test_execution(test_ml.workspace, "EXE-A", ExecutionStatus.stopped)
    # Simulate prior offline transition.
    test_ml.workspace.execution_state_store().update_execution(
        "EXE-A", sync_pending=True,
    )

    flushed_calls = []
    reconcile_calls = []

    def _fake_flush(*, store, catalog, execution_rid):
        flushed_calls.append(execution_rid)
        store.update_execution(execution_rid, sync_pending=False)

    def _fake_reconcile(*, store, catalog, execution_rid):
        reconcile_calls.append(execution_rid)

    monkeypatch.setattr(
        "deriva_ml.core.mixins.execution.flush_pending_sync", _fake_flush,
    )
    monkeypatch.setattr(
        "deriva_ml.core.mixins.execution.reconcile_with_catalog", _fake_reconcile,
    )

    exe = test_ml.resume_execution("EXE-A")
    assert flushed_calls == ["EXE-A"]
    # Reconcile runs AFTER flush (so we compare catalog vs synced state).
    assert reconcile_calls == ["EXE-A"]


def test_gc_executions_deletes_matching(test_ml):
    from datetime import timedelta
    from deriva_ml.execution.state_store import ExecutionStatus

    # Three uploaded executions of different ages.
    _insert_test_execution(test_ml.workspace, "OLD", ExecutionStatus.uploaded)
    _insert_test_execution(test_ml.workspace, "NEW", ExecutionStatus.uploaded)
    _insert_test_execution(test_ml.workspace, "RUN", ExecutionStatus.running)

    # Backdate OLD so it matches older_than.
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)
    store.update_execution(
        "OLD",
        last_activity=now - timedelta(days=30),
        created_at=now - timedelta(days=30),
    )

    n = test_ml.gc_executions(
        status=ExecutionStatus.uploaded,
        older_than=timedelta(days=7),
    )
    assert n == 1
    rids = {r.rid for r in test_ml.list_executions()}
    assert rids == {"NEW", "RUN"}


def test_gc_executions_status_only(test_ml):
    from deriva_ml.execution.state_store import ExecutionStatus

    _insert_test_execution(test_ml.workspace, "A", ExecutionStatus.aborted)
    _insert_test_execution(test_ml.workspace, "B", ExecutionStatus.running)

    n = test_ml.gc_executions(status=ExecutionStatus.aborted)
    assert n == 1
    assert {r.rid for r in test_ml.list_executions()} == {"B"}


def test_gc_executions_delete_working_dir(test_ml):
    from deriva_ml.execution.state_store import ExecutionStatus
    from pathlib import Path

    _insert_test_execution(test_ml.workspace, "EXE-A", ExecutionStatus.uploaded)

    # Create the working directory files.
    work = Path(test_ml.working_dir) / "execution/EXE-A"
    work.mkdir(parents=True, exist_ok=True)
    (work / "scratch.txt").write_text("hi")
    assert work.exists()

    n = test_ml.gc_executions(
        status=ExecutionStatus.uploaded,
        delete_working_dir=True,
    )
    assert n == 1
    assert not work.exists()


# =============================================================================
# D6: create_execution kwargs form + offline guard
# =============================================================================


def test_dataset_spec_from_shorthand():
    """Unit test the string-coercion path without constructing an Execution."""
    import pytest

    from deriva_ml.dataset import DatasetSpec

    # Use a valid RID format (matches the ERMrest RID regex).
    a = DatasetSpec.from_shorthand("1-ABCD@1.0.0")
    assert a.rid == "1-ABCD"
    assert str(a.version) == "1.0.0"

    b = DatasetSpec.from_shorthand("1-ABCD")
    assert b.rid == "1-ABCD"
    # Bare RID means no explicit version; default is 0.0.0.
    assert str(b.version) == "0.0.0"

    with pytest.raises(ValueError):
        DatasetSpec.from_shorthand("")
    with pytest.raises(ValueError):
        DatasetSpec.from_shorthand("1-ABCD@1.0.0@extra")


def test_create_execution_kwargs_form_builds_config(test_ml):
    """Kwargs form with a Workflow object builds an ExecutionConfiguration.

    Doesn't test string shorthand (covered by the unit test above) —
    just confirms kwargs form successfully assembles into a usable
    ExecutionConfiguration for a no-datasets run.
    """
    from deriva_ml import MLVocab as vc

    test_ml.add_term(
        vc.workflow_type,
        "Test Workflow",
        description="for D6 kwargs-form test",
    )
    wf = test_ml.create_workflow(
        name="D6 Test Workflow",
        workflow_type="Test Workflow",
        description="for D6 kwargs-form test",
    )

    exe = test_ml.create_execution(
        workflow=wf,  # Workflow object — skips lookup_workflow_by_url path
        description="kwargs form",
    )
    assert exe.configuration.description == "kwargs form"
    assert exe.configuration.workflow.url == wf.url


def test_create_execution_kwargs_form_workflow_string_lookup(test_ml):
    """workflow=string should route through lookup_workflow_by_url."""
    from deriva_ml import MLVocab as vc

    test_ml.add_term(
        vc.workflow_type,
        "Test Workflow",
        description="for D6 string lookup test",
    )
    wf = test_ml.create_workflow(
        name="D6 String Workflow",
        workflow_type="Test Workflow",
        description="for D6 string lookup test",
    )
    # Register the workflow in the catalog so lookup_workflow_by_url
    # can find it. (add_workflow is idempotent on checksum.)
    test_ml.add_workflow(wf)

    exe = test_ml.create_execution(
        workflow=wf.url,
        description="string workflow",
    )
    assert exe.configuration.workflow.url == wf.url


def test_create_execution_rejects_mixed_forms(test_ml):
    import pytest

    from deriva_ml import MLVocab as vc
    from deriva_ml.execution import ExecutionConfiguration

    test_ml.add_term(
        vc.workflow_type,
        "Test Workflow",
        description="for mixed-form rejection",
    )
    wf = test_ml.create_workflow(
        name="D6 Mixed Test",
        workflow_type="Test Workflow",
        description="for mixed-form rejection",
    )

    cfg = ExecutionConfiguration(
        workflow=wf,
        description="cfg",
    )

    with pytest.raises(TypeError) as exc:
        test_ml.create_execution(cfg, datasets=["1-ABCD@1.0.0"])
    assert (
        "cannot mix" in str(exc.value).lower()
        or "exactly one" in str(exc.value).lower()
    )


def test_create_execution_offline_raises(catalog_manager, tmp_path):
    import pytest

    from deriva_ml import ConnectionMode, DerivaML
    from deriva_ml.core.exceptions import DerivaMLOfflineError

    catalog_manager.reset()
    ml_offline = DerivaML(
        catalog_manager.hostname,
        catalog_manager.catalog_id,
        default_schema=catalog_manager.domain_schema,
        working_dir=tmp_path,
        use_minid=False,
        mode=ConnectionMode.offline,
    )

    with pytest.raises(DerivaMLOfflineError):
        ml_offline.create_execution(
            description="can't",
            workflow="any-url",  # doesn't need to exist — offline check fires first
        )


def test_create_execution_writes_registry_row(test_ml):
    """After create_execution, the workspace SQLite must have the row."""
    from deriva_ml import MLVocab as vc
    from deriva_ml.execution.state_store import ExecutionStatus

    test_ml.add_term(
        vc.workflow_type,
        "Test Workflow",
        description="for D7 registry-write test",
    )
    wf = test_ml.create_workflow(
        name="D7 Registry Test Workflow",
        workflow_type="Test Workflow",
        description="for D7 registry-write test",
    )

    exe = test_ml.create_execution(
        workflow=wf,
        description="registry test",
    )
    store = test_ml.workspace.execution_state_store()
    row = store.get_execution(exe.execution_rid)
    assert row is not None
    assert row["rid"] == exe.execution_rid
    # Initial status is 'created'.
    assert row["status"] == ExecutionStatus.created
    assert row["mode"] == "online"
    assert row["config_json"]  # non-empty
    assert row["working_dir_rel"].startswith("execution/")


def test_restore_execution_symbol_removed(test_ml):
    """Per R5.1 aggressive deprecation, restore_execution is removed."""
    assert not hasattr(test_ml, "restore_execution"), (
        "restore_execution should have been removed in D8; "
        "use resume_execution (see CHANGELOG breaking changes)."
    )


def test_resume_execution_per_rid_lease_reconcile(test_ml, monkeypatch):
    from deriva_ml.execution.state_store import ExecutionStatus

    _insert_test_execution(test_ml.workspace, "EXE-A", ExecutionStatus.stopped)

    # Stub the preceding reconcile step — our synthetic EXE-A is not
    # in the catalog, so the real reconcile_with_catalog would blow up
    # and the F6 hook would never run.
    monkeypatch.setattr(
        "deriva_ml.core.mixins.execution.reconcile_with_catalog",
        lambda *, store, catalog, execution_rid: None,
    )

    calls: list[str | None] = []

    def _spy(*, store, catalog, execution_rid):
        calls.append(execution_rid)

    from deriva_ml.execution import lease_orchestrator
    monkeypatch.setattr(lease_orchestrator, "reconcile_pending_leases", _spy)

    test_ml.resume_execution("EXE-A")
    assert "EXE-A" in calls  # scoped reconciliation for this execution
