"""SQLite-backed ExecutionRecord — a registry row with derived counts.

Per spec §2.9. A frozen dataclass projection of one execution_state__
row plus convenience counts from pending_rows. Returned by
DerivaML.list_executions, ml.find_incomplete_executions, and as the
handle for resume_execution's just-in-time reconciliation input.

This class will eventually replace the catalog-backed ExecutionRecord
in execution_record.py (Task D8 merges). Built alongside to keep this
refactor reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from deriva_ml.core.connection_mode import ConnectionMode
from deriva_ml.execution.state_store import ExecutionStatus

if TYPE_CHECKING:
    from deriva_ml.core.base import DerivaML  # noqa: F401 (forward-looking)


@dataclass(frozen=True)
class ExecutionRecord:
    """Frozen snapshot of an execution's registry row plus pending counts.

    A value object — no mutation, no server reads on property access.
    If you need lifecycle fields that change over time (live status,
    etc.), use the Execution object returned by resume_execution.

    Attributes:
        rid: Server-assigned Execution RID.
        workflow_rid: Workflow FK; None if not set.
        description: Free-form description from the configuration.
        status: Current lifecycle status as of this snapshot.
        mode: ConnectionMode the execution was last active under.
        working_dir_rel: Relative path to the execution root.
        start_time: Lifecycle start timestamp; None if not yet started.
        stop_time: Lifecycle stop timestamp; None if still running.
        last_activity: Last pending-row mutation time.
        error: Last error message if status in (failed,).
        sync_pending: True if SQLite is ahead of the catalog.
        created_at: When the local registry first knew about this row.
        pending_rows: Count of non-asset pending rows not yet uploaded.
        failed_rows: Count of non-asset rows in status='failed'.
        pending_files: Count of asset-file rows not yet uploaded.
        failed_files: Count of asset-file rows in status='failed'.

    Example:
        >>> records = ml.find_incomplete_executions()
        >>> for r in records:
        ...     print(r.rid, r.status, r.pending_rows)
    """

    rid: str
    workflow_rid: str | None
    description: str | None
    status: ExecutionStatus
    mode: ConnectionMode
    working_dir_rel: str
    start_time: datetime | None
    stop_time: datetime | None
    last_activity: datetime
    error: str | None
    sync_pending: bool
    created_at: datetime
    pending_rows: int
    failed_rows: int
    pending_files: int
    failed_files: int

    @classmethod
    def from_row(
        cls,
        row: dict,
        *,
        pending_rows: int = 0,
        failed_rows: int = 0,
        pending_files: int = 0,
        failed_files: int = 0,
    ) -> "ExecutionRecord":
        """Construct from a SQLite executions row + pending counts.

        Args:
            row: Dict returned by ExecutionStateStore.get_execution or
                list_executions. Must contain all the executions
                columns.
            pending_rows: Count of non-asset pending rows, defaults to 0
                if the caller hasn't queried pending_rows.
            failed_rows: Count of non-asset rows in status='failed'.
            pending_files: Count of asset-file rows not yet uploaded.
            failed_files: Count of asset-file rows in status='failed'.

        Returns:
            A frozen ExecutionRecord instance.

        Example:
            >>> row = store.get_execution("EXE-A")
            >>> counts = store.count_pending_by_kind(execution_rid="EXE-A")
            >>> rec = ExecutionRecord.from_row(row, **counts)
        """
        return cls(
            rid=row["rid"],
            workflow_rid=row["workflow_rid"],
            description=row["description"],
            status=ExecutionStatus(row["status"]),
            mode=ConnectionMode(row["mode"]),
            working_dir_rel=row["working_dir_rel"],
            start_time=row["start_time"],
            stop_time=row["stop_time"],
            last_activity=row["last_activity"],
            error=row["error"],
            sync_pending=bool(row["sync_pending"]),
            created_at=row["created_at"],
            pending_rows=pending_rows,
            failed_rows=failed_rows,
            pending_files=pending_files,
            failed_files=failed_files,
        )
