"""Tests for ``PendingSummary`` Pydantic models + the surviving
production-call-site shape.

The pending-rows write surface was retired in Phase 3 cleanup per
``docs/design/deriva-ml-audit-2026-05-phase3-execution.md`` §1.5;
``ExecutionStateStore.pending_summary_rows`` now always returns
``[]``. The ``PendingSummary`` / ``WorkspacePendingSummary``
Pydantic models survive (they read whatever the store returns), so
the public ``Execution.pending_summary()`` /
``DerivaML.pending_summary()`` API continues to work — it just
reports every execution as having nothing pending.

Tests pinning the model invariants stay; tests that staged
pending_rows via the deleted CRUD are gone.
"""

from __future__ import annotations


def test_pending_row_count_fields():
    from deriva_ml.execution.pending_summary import PendingRowCount

    c = PendingRowCount(
        table="deriva-ml:Subject",
        pending=5,
        failed=1,
        uploaded=10,
    )
    assert c.table == "deriva-ml:Subject"
    assert c.pending == 5


def test_pending_asset_count_fields():
    from deriva_ml.execution.pending_summary import PendingAssetCount

    c = PendingAssetCount(
        table="deriva-ml:Image",
        pending_files=3,
        failed_files=0,
        uploaded_files=7,
        total_bytes_pending=1024 * 1024 * 50,
    )
    assert c.total_bytes_pending == 52428800


def test_pending_summary_has_pending_true_when_any():
    from deriva_ml.execution.pending_summary import (
        PendingRowCount,
        PendingSummary,
    )

    empty = PendingSummary(
        execution_rid="EXE-A",
        rows=[],
        assets=[],
        diagnostics=[],
    )
    assert empty.has_pending is False

    with_rows = PendingSummary(
        execution_rid="EXE-A",
        rows=[PendingRowCount(table="t", pending=1, failed=0, uploaded=0)],
        assets=[],
        diagnostics=[],
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
                table="Image",
                pending_files=2,
                failed_files=0,
                uploaded_files=0,
                total_bytes_pending=10_000,
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
        assets=[
            PendingAssetCount(
                table="Image",
                pending_files=3,
                failed_files=1,
                uploaded_files=0,
                total_bytes_pending=4_200_000,
            )
        ],
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

    ws = WorkspacePendingSummary(
        per_execution=[
            PendingSummary(execution_rid="A", rows=[], assets=[], diagnostics=[]),
            PendingSummary(execution_rid="B", rows=[], assets=[], diagnostics=[]),
        ]
    )
    assert ws.total_executions_with_pending == 0  # neither has pending
    rendered = ws.render()
    assert "A" in rendered
    assert "B" in rendered


# ─── Production call sites — empty-state contract ─────────────────


def _make_workflow(test_ml, name: str):
    """Ensure Test Workflow term + create Workflow object."""
    from deriva_ml import MLVocab as vc

    test_ml.add_term(
        vc.workflow_type,
        "Test Workflow",
        description="for pending_summary tests",
    )
    return test_ml.create_workflow(
        name=name,
        workflow_type="Test Workflow",
        description="for pending_summary tests",
    )


def test_exe_pending_summary_no_pending(test_ml):
    """A fresh execution has nothing pending — the no-op stub
    returns empty data and ``PendingSummary.has_pending`` is False.

    This is now the **only** observable state ``pending_summary()``
    can report, since the pending-rows write surface was retired
    (audit §1.5).
    """
    wf = _make_workflow(test_ml, "empty workflow")
    exe = test_ml.create_execution(description="empty", workflow=wf)
    s = exe.pending_summary()
    assert s.execution_rid == exe.execution_rid
    assert s.has_pending is False


def test_ml_pending_summary_workspace_wide(test_ml):
    """``DerivaML.pending_summary`` aggregates per-execution
    snapshots. With pending_rows retired, every snapshot is empty,
    but the per-execution coverage is still real."""
    wf = _make_workflow(test_ml, "workspace workflow")
    exe_a = test_ml.create_execution(description="a", workflow=wf)
    exe_b = test_ml.create_execution(description="b", workflow=wf)

    ws = test_ml.pending_summary()
    rids = {s.execution_rid for s in ws.per_execution}
    assert {exe_a.execution_rid, exe_b.execution_rid}.issubset(rids)
    # None can have pending since the writer surface is retired.
    assert ws.total_executions_with_pending == 0


def test_execution_record_pending_summary(test_ml):
    """``ExecutionRecord.pending_summary`` exposes the same shape."""
    wf = _make_workflow(test_ml, "record workflow")
    exe = test_ml.create_execution(description="rec", workflow=wf)
    records = test_ml.list_executions()
    rec = next(r for r in records if r.rid == exe.execution_rid)
    s = rec.pending_summary(ml=test_ml)
    assert s.execution_rid == exe.execution_rid
    assert s.has_pending is False
