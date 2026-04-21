"""Tests for the upload engine. Builds up across G3–G8."""

from __future__ import annotations


def _make_workflow(test_ml, name: str):
    """Shared helper: ensure Test Workflow term + create Workflow object."""
    from deriva_ml import MLVocab as vc

    test_ml.add_term(
        vc.workflow_type,
        "Test Workflow",
        description="for upload_engine tests",
    )
    return test_ml.create_workflow(
        name=name,
        workflow_type="Test Workflow",
        description="for upload_engine tests",
    )


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


# ─── G4: _enumerate_work ──────────────────────────────────────────


def test_engine_enumerates_pending_for_executions(test_ml):
    """_enumerate returns one entry per (execution, table) with
    pending or failed-with-retry items."""
    from datetime import datetime, timezone

    from deriva_ml.execution.upload_engine import _enumerate_work

    wf = _make_workflow(test_ml, "G4 enumerate workflow")
    exe_a = test_ml.create_execution(description="a", workflow=wf)
    exe_b = test_ml.create_execution(description="b", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)
    store.insert_pending_row(
        execution_rid=exe_a.execution_rid, key="ka",
        target_schema="s", target_table="Subject",
        metadata_json="{}", created_at=now,
    )
    store.insert_pending_row(
        execution_rid=exe_b.execution_rid, key="kb",
        target_schema="s", target_table="Prediction",
        metadata_json="{}", created_at=now,
    )

    work = _enumerate_work(
        ml=test_ml,
        execution_rids=[exe_a.execution_rid, exe_b.execution_rid],
        retry_failed=False,
    )
    rids = {(w.execution_rid, w.target_table) for w in work}
    assert (exe_a.execution_rid, "Subject") in rids
    assert (exe_b.execution_rid, "Prediction") in rids


def test_engine_retry_failed_includes_failed(test_ml):
    from datetime import datetime, timezone

    from deriva_ml.execution.state_store import PendingRowStatus
    from deriva_ml.execution.upload_engine import _enumerate_work

    wf = _make_workflow(test_ml, "G4 retry workflow")
    exe = test_ml.create_execution(description="retry", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)
    pid = store.insert_pending_row(
        execution_rid=exe.execution_rid, key="k",
        target_schema="s", target_table="T",
        metadata_json="{}", created_at=now,
    )
    store.update_pending_row(pid, status=PendingRowStatus.failed)

    no_retry = _enumerate_work(
        ml=test_ml, execution_rids=[exe.execution_rid], retry_failed=False,
    )
    assert len(no_retry) == 0

    with_retry = _enumerate_work(
        ml=test_ml, execution_rids=[exe.execution_rid], retry_failed=True,
    )
    assert len(with_retry) == 1


def test_engine_none_execution_rids_means_all_pending(test_ml):
    """Pass execution_rids=None → include every execution that has
    pending work."""
    from datetime import datetime, timezone

    from deriva_ml.execution.upload_engine import _enumerate_work

    wf = _make_workflow(test_ml, "G4 all-pending workflow")
    exe_a = test_ml.create_execution(description="a", workflow=wf)
    exe_b = test_ml.create_execution(description="b", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)
    store.insert_pending_row(
        execution_rid=exe_a.execution_rid, key="ka",
        target_schema="s", target_table="T",
        metadata_json="{}", created_at=now,
    )
    # exe_b has no pending rows.

    work = _enumerate_work(ml=test_ml, execution_rids=None, retry_failed=False)
    rids = {w.execution_rid for w in work}
    assert exe_a.execution_rid in rids
    assert exe_b.execution_rid not in rids


# ─── G5: topological sort ────────────────────────────────────────────


def test_topo_sort_puts_parents_before_children():
    """If table A has an FK to table B, B must drain before A."""
    from deriva_ml.execution.upload_engine import (
        _WorkItem,
        topo_sort_work_items,
    )

    items = [
        _WorkItem(execution_rid="E", target_schema="s", target_table="Prediction",
                  pending_ids=[1, 2], is_asset=False),
        _WorkItem(execution_rid="E", target_schema="s", target_table="Subject",
                  pending_ids=[3], is_asset=False),
    ]
    # FK: Prediction.subject → Subject.RID. Subject must come first.
    fk_edges = {
        ("s", "Prediction"): [("s", "Subject")],
    }
    sorted_items = topo_sort_work_items(items, fk_edges=fk_edges)
    ordered_tables = [i.target_table for i in sorted_items]
    assert ordered_tables.index("Subject") < ordered_tables.index("Prediction")


def test_topo_sort_independent_tables_preserved():
    """Tables with no FK relationship come out in a stable order."""
    from deriva_ml.execution.upload_engine import (
        _WorkItem,
        topo_sort_work_items,
    )

    items = [
        _WorkItem(execution_rid="E", target_schema="s", target_table="A",
                  pending_ids=[1], is_asset=False),
        _WorkItem(execution_rid="E", target_schema="s", target_table="B",
                  pending_ids=[2], is_asset=False),
    ]
    sorted_items = topo_sort_work_items(items, fk_edges={})
    assert [i.target_table for i in sorted_items] == ["A", "B"]


def test_topo_sort_detects_cycle():
    """A FK cycle raises DerivaMLCycleError."""
    import pytest

    from deriva_ml.core.exceptions import DerivaMLCycleError
    from deriva_ml.execution.upload_engine import (
        _WorkItem,
        topo_sort_work_items,
    )

    items = [
        _WorkItem(execution_rid="E", target_schema="s", target_table="A",
                  pending_ids=[1], is_asset=False),
        _WorkItem(execution_rid="E", target_schema="s", target_table="B",
                  pending_ids=[2], is_asset=False),
    ]
    fk_edges = {
        ("s", "A"): [("s", "B")],
        ("s", "B"): [("s", "A")],
    }
    with pytest.raises(DerivaMLCycleError):
        topo_sort_work_items(items, fk_edges=fk_edges)
