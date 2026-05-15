"""SQLite-backed store for execution state.

Defines the executions table in the workspace main.db:

- ``execution_state__executions``: per-execution registry row.

The earlier design (per ``2026-04-18-sqlite-execution-state-design.md``)
also defined ``execution_state__pending_rows`` and
``execution_state__directory_rules`` tables for an upload-engine
that was superseded by the bag-commit path before its writer
shipped. That surface was retired in the Phase 3 cleanup
(audit ``docs/design/deriva-ml-audit-2026-05-phase3-execution.md``
§1.5). Three reader methods —
:meth:`ExecutionStateStore.count_pending_rows`,
:meth:`~ExecutionStateStore.count_pending_by_kind`,
:meth:`~ExecutionStateStore.pending_summary_rows` — survive as
no-op stubs so the production call sites that consume their output
(``count`` for the schema-refresh guard, ``by_kind`` for
``find_executions`` / ``Execution.pending_summary``,
``summary_rows`` for the ``PendingSummary`` rendering) continue to
compile and report "nothing pending".

Uses SQLAlchemy Core (no ORM) matching the codebase pattern for
library-bookkeeping tables (ManifestStore, ResultCache). See spec
§2.14 for rationale. Schemas are stable; runtime discovery is not
needed here.

All writes use ``with engine.begin() as conn:`` for atomic
per-transaction commit. WAL mode + per-mutation fsync is inherited
from the Workspace engine configuration.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Index,
    MetaData,
    String,
    Table,
    Text,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine

from deriva_ml.core.logging_config import get_logger

if TYPE_CHECKING:
    from deriva_ml.core.connection_mode import ConnectionMode

logger = get_logger(__name__)


class ExecutionStatus(StrEnum):
    """Lifecycle status for an Execution (see Phase 1 spec §2.2).

    Transitions are:
        Created → Running → {Stopped, Failed} →
            {Pending_Upload → {Uploaded, Failed}}
        Created → Aborted
        Running → Aborted

    Values are title-case to match the catalog Execution.Status field
    directly — ExecutionStatus(row["Status"]) works without translation.
    Python identifiers are title-case to match the values (precedent:
    stdlib http.HTTPStatus uses uppercase identifiers).
    """

    Created = "Created"
    Running = "Running"
    Stopped = "Stopped"
    Failed = "Failed"
    Pending_Upload = "Pending_Upload"
    Uploaded = "Uploaded"
    Aborted = "Aborted"


EXECUTIONS_TABLE = "execution_state__executions"


class ExecutionStateStore:
    """SQLAlchemy Core wrapper for the executions registry table.

    Example:
        >>> store = ExecutionStateStore(engine=workspace.engine)  # doctest: +SKIP
        >>> store.ensure_schema()
        >>> # then use store.executions for queries.

    Attributes:
        engine: The SQLAlchemy Engine (owned by the caller).
        metadata: SQLAlchemy MetaData for the executions table.
        executions: The sqlalchemy.Table for the executions registry.
    """

    def __init__(self, engine: Engine) -> None:
        """Bind the store to an existing Engine.

        Args:
            engine: A SQLAlchemy Engine — typically obtained from
                ``Workspace.engine``. The store does not manage the
                engine's lifecycle; the caller disposes it.
        """
        self.engine = engine
        self.metadata = MetaData()

        # executions — see spec §2.5.1 for column purposes.
        # status values: created|running|stopped|failed|pending_upload|uploaded|aborted
        # mode values: online|offline
        self.executions = Table(
            EXECUTIONS_TABLE,
            self.metadata,
            Column("rid", String, primary_key=True),
            Column("workflow_rid", String, nullable=True),
            Column("description", Text, nullable=True),
            Column("config_json", Text, nullable=False),
            Column("status", String, nullable=False),
            Column("mode", String, nullable=False),
            Column("working_dir_rel", String, nullable=False),
            Column("start_time", DateTime(timezone=True), nullable=True),
            Column("stop_time", DateTime(timezone=True), nullable=True),
            Column("last_activity", DateTime(timezone=True), nullable=False),
            Column("error", Text, nullable=True),
            Column("sync_pending", Boolean, nullable=False, default=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Index("ix_executions_status", "status"),
            Index("ix_executions_workflow_rid", "workflow_rid"),
            Index("ix_executions_last_activity", "last_activity"),
            # Partial index: most rows have sync_pending=False, so a
            # filtered index keeps lookups of pending-sync rows fast
            # without bloating storage.
            Index(
                "ix_executions_sync_pending",
                "sync_pending",
                sqlite_where=Column("sync_pending"),
            ),
        )

    def ensure_schema(self) -> None:
        """Create the executions table if it doesn't already exist.

        Idempotent — safe to call on every DerivaML construction. Uses
        SQLAlchemy's ``create_all`` which issues ``CREATE TABLE IF
        NOT EXISTS`` via dialect-specific SQL, matching the existing
        Workspace pattern (see ManifestStore.ensure_schema).

        Example:
            >>> store = ExecutionStateStore(engine=workspace.engine)  # doctest: +SKIP
            >>> store.ensure_schema()
            >>> # Table now exists; safe to insert/select.
        """
        self.metadata.create_all(self.engine)
        logger.debug("execution_state schema ensured on %s", self.engine.url)

    # ─── executions CRUD ────────────────────────────────────────────

    def insert_execution(
        self,
        *,
        rid: str,
        workflow_rid: str | None,
        description: str | None,
        config_json: str,
        status: ExecutionStatus,
        mode: "ConnectionMode",
        working_dir_rel: str,
        created_at: datetime,
        last_activity: datetime,
        sync_pending: bool = False,
        start_time: datetime | None = None,
        stop_time: datetime | None = None,
        error: str | None = None,
    ) -> None:
        """Insert a new row in the executions table.

        Idempotency is the caller's concern — this method fails if the
        rid already exists (PK constraint).

        Args:
            rid: Server-assigned Execution RID.
            workflow_rid: Workflow FK, or None if not yet attached.
            description: Human-readable description from config.
            config_json: Serialized ExecutionConfiguration.
            status: Initial status, typically ExecutionStatus.Created.
            mode: ConnectionMode the execution is active under.
            working_dir_rel: Path relative to the workspace root.
            created_at: UTC timestamp when the row is written.
            last_activity: Starts equal to created_at; updated on every
                pending-row mutation.
            sync_pending: True if this row is ahead of the catalog.
            start_time / stop_time / error: Populated later by state
                transitions; None at insert time.

        Raises:
            sqlalchemy.exc.IntegrityError: If rid already exists.
        """
        with self.engine.begin() as conn:
            conn.execute(
                insert(self.executions).values(
                    rid=rid,
                    workflow_rid=workflow_rid,
                    description=description,
                    config_json=config_json,
                    status=str(status),
                    mode=str(mode),
                    working_dir_rel=working_dir_rel,
                    start_time=start_time,
                    stop_time=stop_time,
                    last_activity=last_activity,
                    error=error,
                    sync_pending=sync_pending,
                    created_at=created_at,
                )
            )

    def get_execution(self, rid: str) -> dict | None:
        """Return the executions row as a dict, or None if absent.

        Args:
            rid: The execution RID to look up.

        Returns:
            A dict mapping column names to values, or None if no row
            matches. Datetime columns are returned as Python datetime
            objects (timezone-aware).

        Example:
            >>> row = store.get_execution("EXE-A")  # doctest: +SKIP
            >>> row["status"] if row else None
            'running'
        """
        with self.engine.connect() as conn:
            result = conn.execute(select(self.executions).where(self.executions.c.rid == rid)).mappings().first()
        return dict(result) if result is not None else None

    def update_execution(
        self,
        rid: str,
        **fields: object,
    ) -> None:
        """Partial update of an executions row.

        Any column name from the executions table may be passed as
        a kwarg. Status values are coerced to strings automatically.

        Args:
            rid: The execution to update.
            **fields: Columns to set. Missing columns are left alone.

        Raises:
            KeyError: If a kwarg doesn't match a column in the
                executions table.
        """
        valid_cols = {c.name for c in self.executions.columns}
        unknown = set(fields) - valid_cols
        if unknown:
            raise KeyError(f"unknown columns on executions: {unknown}")

        # Coerce enum values to strings — the table columns are plain
        # String, not Enum, so SQLAlchemy won't auto-coerce.
        coerced = {k: str(v) if isinstance(v, ExecutionStatus) else v for k, v in fields.items()}

        with self.engine.begin() as conn:
            conn.execute(update(self.executions).where(self.executions.c.rid == rid).values(**coerced))

    def list_executions(
        self,
        *,
        status: "ExecutionStatus | list[ExecutionStatus] | None" = None,
        workflow_rid: str | None = None,
        mode: "ConnectionMode | None" = None,
        since: datetime | None = None,
    ) -> list[dict]:
        """Filter the executions table and return rows as dicts.

        Args:
            status: Single status or list of statuses to match, or
                None for all.
            workflow_rid: Match only executions attached to this
                workflow, or None for all.
            mode: Match only executions active under this mode.
            since: Return rows where last_activity >= this timestamp.

        Returns:
            List of dicts — one per matching execution row. Empty list
            if nothing matches.

        Example:
            >>> # All incomplete executions:
            >>> incomplete = [ExecutionStatus.Created, ExecutionStatus.Running,  # doctest: +SKIP
            ...               ExecutionStatus.Stopped, ExecutionStatus.Failed,
            ...               ExecutionStatus.Pending_Upload]
            >>> rows = store.list_executions(status=incomplete)
        """
        stmt = select(self.executions)

        if status is not None:
            if isinstance(status, ExecutionStatus):
                statuses = [str(status)]
            else:
                statuses = [str(s) for s in status]
            stmt = stmt.where(self.executions.c.status.in_(statuses))
        if workflow_rid is not None:
            stmt = stmt.where(self.executions.c.workflow_rid == workflow_rid)
        if mode is not None:
            stmt = stmt.where(self.executions.c.mode == str(mode))
        if since is not None:
            stmt = stmt.where(self.executions.c.last_activity >= since)

        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]

    # ─── pending-rows readers — vestigial stubs ────────────────────
    #
    # The pending-rows write surface was retired in Phase 3 cleanup
    # (audit §1.5). Three reader methods survive as no-op stubs so
    # the four production call sites continue to compile and
    # truthfully report "nothing pending". Replace these with real
    # implementations only if a future writer ships.

    def count_pending_rows(self) -> int:
        """Count of non-terminal pending rows across all executions.

        Always returns ``0`` — the pending-rows write surface was
        retired in Phase 3 cleanup (audit §1.5). Kept for the
        schema-refresh guard in ``core/base.py``.
        """
        return 0

    def count_pending_by_kind(
        self,
        *,
        execution_rid: str,
    ) -> dict[str, int]:
        """Per-kind counts of non-terminal / failed pending rows.

        Always returns all-zero counts — the pending-rows write
        surface was retired in Phase 3 cleanup (audit §1.5). Kept
        for ``find_executions`` and ``Execution.pending_summary``.

        Args:
            execution_rid: Ignored; kept for signature stability.

        Returns:
            ``{"pending_rows": 0, "failed_rows": 0,
              "pending_files": 0, "failed_files": 0}``.
        """
        return {
            "pending_rows": 0,
            "failed_rows": 0,
            "pending_files": 0,
            "failed_files": 0,
        }

    def pending_summary_rows(
        self,
        *,
        execution_rid: str,
    ) -> dict:
        """Per-(target_table, status) rollup data for ``PendingSummary``.

        Always returns the empty rollup
        (``{"rows": [], "assets": [], "diagnostics": []}``) — the
        pending-rows write surface was retired in Phase 3 cleanup
        (audit §1.5). The ``PendingSummary`` rendering path
        (``Execution.pending_summary``, ``DerivaML.pending_summary``)
        continues to work; it just sees every execution as having
        nothing pending.

        Args:
            execution_rid: Ignored; kept for signature stability.

        Returns:
            ``{"rows": [], "assets": [], "diagnostics": []}``.
        """
        return {"rows": [], "assets": [], "diagnostics": []}

    # ─── lifecycle ─────────────────────────────────────────────────

    def delete_execution(self, execution_rid: str) -> None:
        """Delete an execution row.

        The earlier design also cascade-deleted the pending_rows and
        directory_rules children, but those tables were retired in
        Phase 3 (audit §1.5). The executions row is now the only
        per-execution state SQLite carries.

        Args:
            execution_rid: Which execution to remove.

        Example:
            >>> store.delete_execution("EXE-A")
            >>> store.get_execution("EXE-A") is None
            True
        """
        with self.engine.begin() as conn:
            conn.execute(delete(self.executions).where(self.executions.c.rid == execution_rid))
