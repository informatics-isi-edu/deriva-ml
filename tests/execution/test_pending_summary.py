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
        PendingAssetCount, PendingRowCount, PendingSummary,
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
        PendingAssetCount, PendingRowCount, PendingSummary,
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
        PendingAssetCount, PendingRowCount, PendingSummary,
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
        PendingSummary, WorkspacePendingSummary,
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
