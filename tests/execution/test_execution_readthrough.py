"""Tests for SQLite-backed read-through properties on Execution."""

from __future__ import annotations

from datetime import datetime, timezone


def _make_exe(test_ml, description: str):
    """Helper: build a test workflow + execution for readthrough tests."""
    from deriva_ml import MLVocab as vc

    # The workflow_type vocab term may or may not already exist in the
    # freshly-populated test catalog; add_term is idempotent on name
    # collisions per the vocabulary contract (it returns the existing
    # term when one already exists under the same name).
    try:
        test_ml.add_term(
            vc.workflow_type,
            "Test Workflow",
            description="for E1 readthrough tests",
        )
    except Exception:
        # Term likely already exists from a prior call.
        pass

    wf = test_ml.create_workflow(
        name=f"E1 Readthrough: {description}",
        workflow_type="Test Workflow",
        description=f"for {description}",
    )
    return test_ml.create_execution(workflow=wf, description=description)


def test_status_reads_from_sqlite(test_ml):
    """Mutating SQLite via state-machine transition must be visible
    via exe.status without any cache invalidation."""
    from deriva_ml.execution.state_store import ExecutionStatus

    exe = _make_exe(test_ml, "readthrough test")

    # Direct SQLite mutation (simulating state_machine.transition):
    store = test_ml.workspace.execution_state_store()
    store.update_execution(exe.execution_rid, status=ExecutionStatus.running)

    # exe.status must reflect the change on the very next read.
    assert exe.status is ExecutionStatus.running

    store.update_execution(exe.execution_rid, status=ExecutionStatus.stopped)
    assert exe.status is ExecutionStatus.stopped


def test_status_reflects_second_process(test_ml, monkeypatch):
    """Two Execution instances bound to the same rid must see the
    same SQLite row — no per-instance caching."""
    from deriva_ml.execution.state_store import ExecutionStatus

    exe_a = _make_exe(test_ml, "two-views")

    # Bypass reconcile during resume: the legacy update_status path
    # writes catalog Status='Pending' (title-case) which doesn't map
    # cleanly to ExecutionStatus yet (E2 will unify). We're testing
    # read-through here, not reconcile.
    monkeypatch.setattr(
        "deriva_ml.core.mixins.execution.reconcile_with_catalog",
        lambda *, store, catalog, execution_rid: None,
    )

    exe_b = test_ml.resume_execution(exe_a.execution_rid)

    store = test_ml.workspace.execution_state_store()
    store.update_execution(exe_a.execution_rid, status=ExecutionStatus.running)

    assert exe_a.status is ExecutionStatus.running
    assert exe_b.status is ExecutionStatus.running


def test_status_raises_if_registry_gone(test_ml):
    """If the SQLite registry row is deleted (gc), the read-through
    property surfaces clearly rather than returning stale cache."""
    import pytest
    from deriva_ml.core.exceptions import DerivaMLStateInconsistency

    exe = _make_exe(test_ml, "gone")
    store = test_ml.workspace.execution_state_store()
    store.delete_execution(exe.execution_rid)

    with pytest.raises(DerivaMLStateInconsistency):
        _ = exe.status


def test_execute_enter_transitions_to_running(test_ml):
    """Entering `with exe.execute():` transitions status created → running,
    exiting cleanly transitions running → stopped.
    """
    from deriva_ml.execution.state_store import ExecutionStatus

    exe = _make_exe(test_ml, "E2 enter test")
    assert exe.status is ExecutionStatus.created

    with exe.execute() as _e:
        assert exe.status is ExecutionStatus.running

    assert exe.status is ExecutionStatus.stopped


def test_execute_exit_with_exception_transitions_to_failed(test_ml):
    """Exiting the execute() context with an exception transitions to
    `failed` and captures the error message. The exception propagates
    (new __exit__ returns False, unlike the legacy True-suppression)."""
    import pytest
    from deriva_ml.execution.state_store import ExecutionStatus

    exe = _make_exe(test_ml, "E2 failure test")

    with pytest.raises(RuntimeError):
        with exe.execute():
            raise RuntimeError("boom")

    assert exe.status is ExecutionStatus.failed
    # error message captured:
    assert "boom" in (exe.error or "")


def test_abort_transitions_to_aborted(test_ml):
    """abort() is a new public method; legal from any non-terminal state."""
    from deriva_ml.execution.state_store import ExecutionStatus

    exe = _make_exe(test_ml, "E2 abort test")
    exe.abort()
    assert exe.status is ExecutionStatus.aborted


def test_start_stop_time_readthrough(test_ml):
    """E3: start_time / stop_time are SQLite read-through properties.

    Parallels the `status` read-through semantics. Before __enter__,
    both are None; after __enter__, start_time is set; after __exit__
    (clean), stop_time is set and stop_time >= start_time.
    """
    exe = _make_exe(test_ml, "E3 times test")
    assert exe.start_time is None
    assert exe.stop_time is None

    with exe.execute():
        assert exe.start_time is not None
        assert exe.stop_time is None

    assert exe.stop_time is not None
    assert exe.stop_time >= exe.start_time


def test_repr_includes_status_and_pending(test_ml):
    """__repr__ shows the rid + status + pending counts."""
    from datetime import datetime, timezone

    exe = _make_exe(test_ml, "repr")
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)
    store.insert_pending_row(
        execution_rid=exe.execution_rid, key="k",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )

    r = repr(exe)
    assert exe.execution_rid in r
    assert "created" in r   # status (SQLite is lowercase ExecutionStatus)
    assert "1" in r         # pending count
