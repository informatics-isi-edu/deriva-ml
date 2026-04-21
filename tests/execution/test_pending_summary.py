"""Tests for PendingSummary dataclasses + render output."""

from __future__ import annotations


def test_pending_row_count_fields():
    from deriva_ml.execution.pending_summary import PendingRowCount

    c = PendingRowCount(
        table="deriva-ml:Subject",
        pending=5, failed=1, uploaded=10,
    )
    assert c.table == "deriva-ml:Subject"
    assert c.pending == 5


def test_pending_asset_count_fields():
    from deriva_ml.execution.pending_summary import PendingAssetCount

    c = PendingAssetCount(
        table="deriva-ml:Image",
        pending_files=3, failed_files=0, uploaded_files=7,
        total_bytes_pending=1024 * 1024 * 50,
    )
    assert c.total_bytes_pending == 52428800


def test_pending_summary_has_pending_true_when_any():
    from deriva_ml.execution.pending_summary import (
        PendingRowCount,
        PendingSummary,
    )

    empty = PendingSummary(
        execution_rid="EXE-A", rows=[], assets=[], diagnostics=[],
    )
    assert empty.has_pending is False

    with_rows = PendingSummary(
        execution_rid="EXE-A",
        rows=[PendingRowCount(table="t", pending=1, failed=0, uploaded=0)],
        assets=[], diagnostics=[],
    )
    assert with_rows.has_pending is True


def test_pending_summary_total_counts():
    from deriva_ml.execution.pending_summary import (
        PendingAssetCount,
        PendingRowCount,
        PendingSummary,
    )

    s = PendingSummary(
        execution_rid="EXE-A",
        rows=[
            PendingRowCount(table="Subject", pending=3, failed=1, uploaded=0),
            PendingRowCount(table="Prediction", pending=5, failed=0, uploaded=10),
        ],
        assets=[
            PendingAssetCount(
                table="Image", pending_files=2, failed_files=0,
                uploaded_files=0, total_bytes_pending=10_000,
            ),
        ],
        diagnostics=[],
    )
    assert s.total_pending_rows == 8
    assert s.total_pending_files == 2


def test_pending_summary_render_has_key_parts():
    from deriva_ml.execution.pending_summary import (
        PendingAssetCount,
        PendingRowCount,
        PendingSummary,
    )

    s = PendingSummary(
        execution_rid="EXE-ABC",
        rows=[PendingRowCount(table="Subject", pending=2, failed=0, uploaded=0)],
        assets=[PendingAssetCount(
            table="Image", pending_files=3, failed_files=1,
            uploaded_files=0, total_bytes_pending=4_200_000,
        )],
        diagnostics=["Image row IMG-42 failed: FK violation"],
    )
    output = s.render()
    assert "EXE-ABC" in output
    assert "Subject" in output
    assert "2 pending" in output
    assert "Image" in output
    assert "3 pending" in output
    assert "FK violation" in output


def test_workspace_pending_summary():
    from deriva_ml.execution.pending_summary import (
        PendingSummary,
        WorkspacePendingSummary,
    )

    ws = WorkspacePendingSummary(per_execution=[
        PendingSummary(
            execution_rid="A", rows=[], assets=[], diagnostics=[],
        ),
        PendingSummary(
            execution_rid="B", rows=[], assets=[], diagnostics=[],
        ),
    ])
    assert ws.total_executions_with_pending == 0  # neither has pending
    rendered = ws.render()
    assert "A" in rendered
    assert "B" in rendered


# ─── G2: integration tests via real ExecutionStateStore + DerivaML ────


def _make_workflow(test_ml, name: str):
    """Shared helper: ensure Test Workflow term + create Workflow object."""
    from deriva_ml import MLVocab as vc

    test_ml.add_term(
        vc.workflow_type,
        "Test Workflow",
        description="for G2 pending_summary tests",
    )
    return test_ml.create_workflow(
        name=name,
        workflow_type="Test Workflow",
        description="for G2 pending_summary tests",
    )


def test_exe_pending_summary_no_pending(test_ml):
    wf = _make_workflow(test_ml, "G2 empty workflow")
    exe = test_ml.create_execution(description="empty", workflow=wf)
    s = exe.pending_summary()
    assert s.execution_rid == exe.execution_rid
    assert s.has_pending is False


def test_exe_pending_summary_aggregates_rows(test_ml):
    from datetime import datetime, timezone

    wf = _make_workflow(test_ml, "G2 aggregates workflow")
    exe = test_ml.create_execution(description="sum", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)
    store.insert_pending_row(
        execution_rid=exe.execution_rid, key="a",
        target_schema="deriva-ml", target_table="Subject",
        metadata_json="{}", created_at=now,
    )
    store.insert_pending_row(
        execution_rid=exe.execution_rid, key="b",
        target_schema="deriva-ml", target_table="Subject",
        metadata_json="{}", created_at=now,
    )
    store.insert_pending_row(
        execution_rid=exe.execution_rid, key="c",
        target_schema="deriva-ml", target_table="Prediction",
        metadata_json="{}", created_at=now,
    )

    s = exe.pending_summary()
    assert s.has_pending is True
    by_table = {r.table: r for r in s.rows}
    assert by_table["deriva-ml:Subject"].pending == 2
    assert by_table["deriva-ml:Prediction"].pending == 1


def test_exe_pending_summary_asset_bytes(test_ml, tmp_path):
    """total_bytes_pending sums actual file sizes."""
    from datetime import datetime, timezone

    wf = _make_workflow(test_ml, "G2 bytes workflow")
    exe = test_ml.create_execution(description="bytes", workflow=wf)
    # Create a file and register a pending asset row.
    f = tmp_path / "img.png"
    f.write_bytes(b"x" * 2048)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)
    store.insert_pending_row(
        execution_rid=exe.execution_rid, key="img",
        target_schema="deriva-ml", target_table="Image",
        metadata_json="{}", created_at=now,
        asset_file_path=str(f),
    )

    s = exe.pending_summary()
    [a] = s.assets
    assert a.pending_files == 1
    assert a.total_bytes_pending == 2048


def test_execution_record_pending_summary(test_ml):
    wf = _make_workflow(test_ml, "G2 record workflow")
    exe = test_ml.create_execution(description="rec", workflow=wf)
    records = test_ml.list_executions()
    assert len(records) >= 1
    # Grab the new record matching our rid.
    rec = next(r for r in records if r.rid == exe.execution_rid)
    s = rec.pending_summary(ml=test_ml)
    assert s.execution_rid == exe.execution_rid


def test_ml_pending_summary_workspace_wide(test_ml):
    wf = _make_workflow(test_ml, "G2 workspace workflow")
    exe_a = test_ml.create_execution(description="a", workflow=wf)
    exe_b = test_ml.create_execution(description="b", workflow=wf)

    ws = test_ml.pending_summary()
    rids = {s.execution_rid for s in ws.per_execution}
    assert {exe_a.execution_rid, exe_b.execution_rid}.issubset(rids)


def test_exe_pending_summary_asset_missing_file_diagnostic(test_ml, tmp_path):
    """Missing asset file surfaces as diagnostic, not silent."""
    from datetime import datetime, timezone

    wf = _make_workflow(test_ml, "G2 missing file")
    exe = test_ml.create_execution(description="miss", workflow=wf)

    # Register a pending asset row whose file doesn't exist.
    missing = tmp_path / "gone.png"
    # deliberately don't write it
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)
    store.insert_pending_row(
        execution_rid=exe.execution_rid, key="gone",
        target_schema="deriva-ml", target_table="Image",
        metadata_json="{}", created_at=now,
        asset_file_path=str(missing),
    )

    s = exe.pending_summary()
    assert any("gone.png" in d or "missing on disk" in d for d in s.diagnostics)
    # pending_files still counts — we still know there's a pending row
    [a] = s.assets
    assert a.pending_files == 1
    # bytes total is 0 because the file doesn't exist
    assert a.total_bytes_pending == 0
