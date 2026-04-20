"""Per-execution and workspace-wide pending-upload summaries.

Per spec §2.11.1. Returned by exe.pending_summary(),
record.pending_summary(), and ml.pending_summary(). Consumed by the
context-manager exit INFO log, Execution.__repr__, and the upload
engine for pre-flight reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING


@dataclass(frozen=True)
class PendingRowCount:
    """Per-table counts for non-asset pending rows.

    Attributes:
        table: Fully qualified table name "schema:Table".
        pending: Rows in staged/leasing/leased/uploading — not yet
            terminally done.
        failed: Rows in status='failed'.
        uploaded: Rows in status='uploaded'. Included for reporting
            completeness; excluded from has_pending logic.

    Example:
        >>> c = PendingRowCount(table="deriva-ml:Subject",
        ...                     pending=5, failed=0, uploaded=12)
    """
    table: str
    pending: int
    failed: int
    uploaded: int


@dataclass(frozen=True)
class PendingAssetCount:
    """Per-table counts for asset-file rows.

    Attributes:
        table: Fully qualified asset table name.
        pending_files: File rows not yet terminally uploaded.
        failed_files: File rows in status='failed'.
        uploaded_files: File rows in status='uploaded'.
        total_bytes_pending: Sum of bytes pending upload for this
            table, derived from on-disk file sizes. Zero if the
            caller didn't compute it (cheap summaries may skip).

    Example:
        >>> c = PendingAssetCount(table="deriva-ml:Image",
        ...                       pending_files=3, failed_files=0,
        ...                       uploaded_files=0,
        ...                       total_bytes_pending=4_200_000)
    """
    table: str
    pending_files: int
    failed_files: int
    uploaded_files: int
    total_bytes_pending: int


@dataclass(frozen=True)
class PendingSummary:
    """A full snapshot of pending upload state for one execution.

    Attributes:
        execution_rid: Which execution this summary describes.
        rows: Per-table row counts (plain rows, not asset files).
        assets: Per-table asset-file counts.
        diagnostics: Human-readable messages — e.g., "row IMG-42
            failed: FK violation on Subject". Sourced from the
            `error` column of failed pending_rows.

    Example:
        >>> summary = exe.pending_summary()
        >>> if summary.has_pending:
        ...     print(summary.render())
    """
    execution_rid: str
    rows: list[PendingRowCount]
    assets: list[PendingAssetCount]
    diagnostics: list[str]

    @property
    def has_pending(self) -> bool:
        """True if any non-terminal pending items exist.

        Excludes failed and uploaded — "pending" specifically means
        "not yet attempted or not yet completed."
        """
        return (
            any(r.pending > 0 for r in self.rows)
            or any(a.pending_files > 0 for a in self.assets)
        )

    @property
    def total_pending_rows(self) -> int:
        """Sum of pending counts across all row tables."""
        return sum(r.pending for r in self.rows)

    @property
    def total_pending_files(self) -> int:
        """Sum of pending counts across all asset tables."""
        return sum(a.pending_files for a in self.assets)

    def render(self) -> str:
        """Multi-line human-readable rendering for logs / CLI.

        Format matches the exit-log template in spec §2.12:

            Execution EXE-ABC pending state:
              rows:    Subject (2 pending, 0 failed)
                       Prediction (5 pending, 0 failed)
              assets:  Image (3 pending, 1 failed, 4.2MB)
              diagnostics:
                - Image row IMG-42 failed: FK violation

        Empty sections are omitted. Byte sizes use human-readable
        suffixes (KB/MB/GB).

        Returns:
            Multi-line string; caller prints or logs.

        Example:
            >>> print(summary.render())
            Execution EXE-A pending state:
              rows:    Subject (2 pending, 0 failed)
        """
        out = [f"Execution {self.execution_rid} pending state:"]
        if self.rows:
            row_lines = [
                f"    {r.table} ({r.pending} pending, {r.failed} failed)"
                for r in self.rows
                if r.pending or r.failed
            ]
            if row_lines:
                out.append("  rows:")
                out.extend(row_lines)
        if self.assets:
            asset_lines = []
            for a in self.assets:
                if not (a.pending_files or a.failed_files):
                    continue
                size = _humanize_bytes(a.total_bytes_pending)
                asset_lines.append(
                    f"    {a.table} ({a.pending_files} pending, "
                    f"{a.failed_files} failed, {size})"
                )
            if asset_lines:
                out.append("  assets:")
                out.extend(asset_lines)
        if self.diagnostics:
            out.append("  diagnostics:")
            out.extend(f"    - {d}" for d in self.diagnostics)
        return "\n".join(out)


@dataclass(frozen=True)
class WorkspacePendingSummary:
    """A summary spanning multiple executions (workspace-wide view).

    Returned by ml.pending_summary(). Useful for standalone uploader
    processes that want to know everything pending across runs.

    Attributes:
        per_execution: PendingSummary per execution that has at least
            one known-local row (pending or terminal). Empty
            executions are excluded.
    """
    per_execution: list[PendingSummary]

    @property
    def total_executions_with_pending(self) -> int:
        """Count of executions that currently have non-terminal items."""
        return sum(1 for s in self.per_execution if s.has_pending)

    def render(self) -> str:
        """Multi-section rendering; calls PendingSummary.render per
        execution, with a header row.
        """
        if not self.per_execution:
            return "No known-local executions in this workspace."
        out = [f"Workspace pending: {len(self.per_execution)} executions"]
        for s in self.per_execution:
            out.append("")
            out.append(s.render())
        return "\n".join(out)


def _humanize_bytes(n: int) -> str:
    """Format a byte count with KB/MB/GB suffix.

    Keeps one decimal place for MB/GB to match typical CLI conventions.
    """
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f}KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f}MB"
    return f"{n / (1024 * 1024 * 1024):.2f}GB"
