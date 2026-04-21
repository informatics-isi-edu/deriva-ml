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


# ─── G6: run_upload_engine outer loop ─────────────────────────────────


def test_run_upload_engine_leases_and_marks_uploaded(test_ml, monkeypatch):
    """Engine path: enumerate → lease → deriva-py uploader → status=uploaded.

    We mock deriva-py's uploader to a no-op (returns success).
    """
    from datetime import datetime, timezone

    from deriva_ml.execution.state_store import PendingRowStatus
    from deriva_ml.execution.upload_engine import run_upload_engine

    wf = _make_workflow(test_ml, "G6 engine workflow")
    exe = test_ml.create_execution(description="engine", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)
    store.insert_pending_row(
        execution_rid=exe.execution_rid, key="k1",
        target_schema="deriva-ml", target_table="Subject",
        metadata_json='{"Name": "A"}', created_at=now,
    )

    lease_calls: list[list[int]] = []

    def _fake_acquire(*, store, catalog, execution_rid, pending_ids):
        lease_calls.append(pending_ids)
        for i, pid in enumerate(pending_ids):
            store.update_pending_row(
                pid, rid=f"R-{i}",
                status=PendingRowStatus.leased,
                lease_token=f"T-{i}",
            )

    def _fake_upload(*, store, catalog, work_item, ml=None):
        for pid in work_item.pending_ids:
            store.update_pending_row(
                pid, status=PendingRowStatus.uploaded,
                uploaded_at=datetime.now(timezone.utc),
            )
        return len(work_item.pending_ids)

    # Stub state-machine transitions so test doesn't require traversing
    # a multi-step status path (running → stopped → pending_upload → ...).
    def _fake_transition(*, store, catalog, execution_rid, current, target, mode, extra_fields=None):
        store.update_execution(execution_rid, status=str(target))

    monkeypatch.setattr(
        "deriva_ml.execution.upload_engine.acquire_leases_for_execution",
        _fake_acquire,
    )
    monkeypatch.setattr(
        "deriva_ml.execution.upload_engine._drain_work_item",
        _fake_upload,
    )
    monkeypatch.setattr(
        "deriva_ml.execution.upload_engine.transition",
        _fake_transition,
    )

    report = run_upload_engine(
        ml=test_ml,
        execution_rids=[exe.execution_rid],
        retry_failed=False,
    )
    assert report.total_uploaded == 1
    assert report.total_failed == 0
    rows = store.list_pending_rows(execution_rid=exe.execution_rid)
    assert all(r["status"] == "uploaded" for r in rows)


def test_run_upload_engine_level_fail_fast_finishes_level(test_ml, monkeypatch):
    """At a given topological level, one item's failure must not
    short-circuit its siblings — both independent items get attempted.
    But subsequent levels (not present here) would be aborted."""
    from datetime import datetime, timezone

    from deriva_ml.execution.state_store import PendingRowStatus
    from deriva_ml.execution.upload_engine import run_upload_engine

    wf = _make_workflow(test_ml, "G6 level fail-fast")
    exe = test_ml.create_execution(description="level", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)

    # Two tables at the same level (independence enforced by the
    # _fk_edges_for_work monkeypatch below).
    store.insert_pending_row(
        execution_rid=exe.execution_rid, key="kA",
        target_schema="deriva-ml", target_table="TableA",
        metadata_json="{}", created_at=now,
    )
    store.insert_pending_row(
        execution_rid=exe.execution_rid, key="kB",
        target_schema="deriva-ml", target_table="TableB",
        metadata_json="{}", created_at=now,
    )

    drained: list[str] = []

    def _fake_acquire(*, store, catalog, execution_rid, pending_ids):
        for i, pid in enumerate(pending_ids):
            store.update_pending_row(
                pid, rid=f"R-{i}",
                status=PendingRowStatus.leased,
                lease_token=f"T-{i}",
            )

    def _fake_upload(*, store, catalog, work_item, ml=None):
        drained.append(work_item.target_table)
        if work_item.target_table == "TableA":
            # Simulate failure.
            for pid in work_item.pending_ids:
                store.update_pending_row(
                    pid, status=PendingRowStatus.failed, error="synthetic",
                )
            raise RuntimeError("synthetic drain failure")
        # Success path.
        for pid in work_item.pending_ids:
            store.update_pending_row(
                pid, status=PendingRowStatus.uploaded,
                uploaded_at=datetime.now(timezone.utc),
            )
        return len(work_item.pending_ids)

    def _fake_transition(*, store, catalog, execution_rid, current, target, mode, extra_fields=None):
        store.update_execution(execution_rid, status=str(target))

    monkeypatch.setattr(
        "deriva_ml.execution.upload_engine.acquire_leases_for_execution",
        _fake_acquire,
    )
    monkeypatch.setattr(
        "deriva_ml.execution.upload_engine._drain_work_item",
        _fake_upload,
    )
    monkeypatch.setattr(
        "deriva_ml.execution.upload_engine.transition",
        _fake_transition,
    )
    # Force both tables into the same level (no FK between them).
    monkeypatch.setattr(
        "deriva_ml.execution.upload_engine._fk_edges_for_work",
        lambda *, ml, items: {
            (i.target_schema, i.target_table): [] for i in items
        },
    )

    report = run_upload_engine(
        ml=test_ml,
        execution_rids=[exe.execution_rid],
        retry_failed=False,
    )
    # Both items got drained despite one failing — that's the fix
    # for Issue 1 (per-level fail-fast, not per-item fail-fast).
    assert set(drained) == {"TableA", "TableB"}
    # One uploaded, one failed.
    assert report.total_uploaded == 1
    assert report.total_failed == 1


def test_run_upload_engine_partial_drain_failed_rows_mark_execution_failed(
    test_ml, monkeypatch,
):
    """When level 0 fails and level 1 never runs, the execution ends
    up in 'failed' status because level 0 has failed rows. Level 1
    rows stay in their prior non-terminal status — they are not
    promoted to 'failed' (that's the point of Issue 2's fix). A later
    upload_pending retry without retry_failed would pick them up."""
    from datetime import datetime, timezone

    from deriva_ml.execution.state_store import ExecutionStatus, PendingRowStatus
    from deriva_ml.execution.upload_engine import run_upload_engine

    wf = _make_workflow(test_ml, "G6 partial drain")
    exe = test_ml.create_execution(description="partial", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)

    # Two levels: A at level 0 (fails), B at level 1 (never drained).
    pid_a = store.insert_pending_row(
        execution_rid=exe.execution_rid, key="kA",
        target_schema="deriva-ml", target_table="TableA",
        metadata_json="{}", created_at=now,
    )
    pid_b = store.insert_pending_row(
        execution_rid=exe.execution_rid, key="kB",
        target_schema="deriva-ml", target_table="TableB",
        metadata_json="{}", created_at=now,
    )

    def _fake_acquire(*, store, catalog, execution_rid, pending_ids):
        for i, pid in enumerate(pending_ids):
            store.update_pending_row(
                pid, rid=f"R-{i}",
                status=PendingRowStatus.leased,
                lease_token=f"T-{i}",
            )

    def _fake_upload(*, store, catalog, work_item, ml=None):
        if work_item.target_table == "TableA":
            for pid in work_item.pending_ids:
                store.update_pending_row(
                    pid, status=PendingRowStatus.failed, error="x",
                )
            raise RuntimeError("TableA failed")
        # TableB should never get here because the drain aborts at
        # level boundary. Fail loud if it does.
        raise AssertionError("TableB should not have been drained")

    def _fake_transition(*, store, catalog, execution_rid, current, target, mode, extra_fields=None):
        store.update_execution(execution_rid, status=str(target))

    monkeypatch.setattr(
        "deriva_ml.execution.upload_engine.acquire_leases_for_execution",
        _fake_acquire,
    )
    monkeypatch.setattr(
        "deriva_ml.execution.upload_engine._drain_work_item",
        _fake_upload,
    )
    monkeypatch.setattr(
        "deriva_ml.execution.upload_engine.transition",
        _fake_transition,
    )
    # B depends on A — A must drain first.
    monkeypatch.setattr(
        "deriva_ml.execution.upload_engine._fk_edges_for_work",
        lambda *, ml, items: {
            ("deriva-ml", "TableA"): [],
            ("deriva-ml", "TableB"): [("deriva-ml", "TableA")],
        },
    )

    report = run_upload_engine(
        ml=test_ml,
        execution_rids=[exe.execution_rid],
        retry_failed=False,
    )
    # A failed; B never got drained.
    assert report.total_failed == 1
    assert report.total_uploaded == 0
    # Execution is in 'failed' status because A actually failed.
    row = store.get_execution(exe.execution_rid)
    assert row["status"] == str(ExecutionStatus.failed)
    # Crucially: B's pending row was NOT marked failed. It's still
    # in its prior non-terminal status (leased after _fake_acquire),
    # so a future upload_pending run without retry_failed will pick
    # it up.
    b_row = next(
        r for r in store.list_pending_rows(execution_rid=exe.execution_rid)
        if r["id"] == pid_b
    )
    assert b_row["status"] != str(PendingRowStatus.failed)
    # And A's pending row is failed.
    a_row = next(
        r for r in store.list_pending_rows(execution_rid=exe.execution_rid)
        if r["id"] == pid_a
    )
    assert a_row["status"] == str(PendingRowStatus.failed)


# ─── G7: _drain_work_item real implementation ────────────────────────


def test_drain_work_item_plain_row(test_ml, monkeypatch):
    """Plain-row drain: build catalog-insert body via pathBuilder, mark
    rows uploaded on success."""
    import json
    from datetime import datetime, timezone

    from deriva_ml.execution.state_store import PendingRowStatus
    from deriva_ml.execution.upload_engine import _drain_work_item, _WorkItem

    wf = _make_workflow(test_ml, "G7 drain plain")
    exe = test_ml.create_execution(description="drain-plain", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)
    pid = store.insert_pending_row(
        execution_rid=exe.execution_rid, key="k",
        target_schema="deriva-ml", target_table="Subject",
        metadata_json=json.dumps({"Name": "Alice"}),
        created_at=now, rid="R-1",
        status=PendingRowStatus.leased,
    )

    inserted_bodies: list = []

    class _MockInsertCallable:
        def __call__(self, body, **_kw):
            inserted_bodies.append(body)
            return [{"RID": "R-1", **body[0]}]

    class _FakeTablePath:
        insert = _MockInsertCallable()

    class _FakeTablesContainer:
        def __getitem__(self, name):
            return _FakeTablePath()

    class _FakeSchema:
        tables = _FakeTablesContainer()

    class _FakeSchemas:
        def __getitem__(self, name):
            return _FakeSchema()

    class _FakePathBuilder:
        schemas = _FakeSchemas()

    monkeypatch.setattr(test_ml, "pathBuilder", lambda: _FakePathBuilder())

    item = _WorkItem(
        execution_rid=exe.execution_rid,
        target_schema="deriva-ml", target_table="Subject",
        pending_ids=[pid], is_asset=False,
    )
    n = _drain_work_item(store=store, catalog=test_ml.catalog, work_item=item, ml=test_ml)

    assert n == 1
    row = store.list_pending_rows(execution_rid=exe.execution_rid)[0]
    assert row["status"] == "uploaded"
    assert row["uploaded_at"] is not None
    assert inserted_bodies[0][0]["RID"] == "R-1"
    assert inserted_bodies[0][0]["Name"] == "Alice"


def test_drain_work_item_asset_row_calls_deriva_py_uploader(test_ml, monkeypatch, tmp_path):
    """Asset-row drain: invoke deriva-py uploader for the file."""
    import json
    from datetime import datetime, timezone

    from deriva_ml.execution.state_store import PendingRowStatus
    from deriva_ml.execution.upload_engine import _drain_work_item, _WorkItem

    f = tmp_path / "img.png"
    f.write_bytes(b"fake-png")

    wf = _make_workflow(test_ml, "G7 drain asset")
    exe = test_ml.create_execution(description="drain-asset", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)
    pid = store.insert_pending_row(
        execution_rid=exe.execution_rid, key="k",
        target_schema="deriva-ml", target_table="Image",
        metadata_json=json.dumps({"Description": "x"}),
        created_at=now,
        rid="IMG-1",
        status=PendingRowStatus.leased,
        asset_file_path=str(f),
    )

    uploader_calls = []

    def _fake_uploader(*, ml, files, target_table, execution_rid):
        uploader_calls.append({
            "files": list(files), "target_table": target_table,
            "execution_rid": execution_rid,
        })
        return {"uploaded": [str(f)], "failed": []}

    monkeypatch.setattr(
        "deriva_ml.execution.upload_engine._invoke_deriva_py_uploader",
        _fake_uploader,
    )

    item = _WorkItem(
        execution_rid=exe.execution_rid,
        target_schema="deriva-ml", target_table="Image",
        pending_ids=[pid], is_asset=True,
    )
    n = _drain_work_item(store=store, catalog=test_ml.catalog, work_item=item, ml=test_ml)

    assert n == 1
    assert len(uploader_calls) == 1
    row = store.list_pending_rows(execution_rid=exe.execution_rid)[0]
    assert row["status"] == "uploaded"
