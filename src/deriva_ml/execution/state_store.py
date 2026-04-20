"""SQLite-backed store for execution state.

Defines three tables in the workspace main.db:
- execution_state__executions: per-execution registry row
- execution_state__pending_rows: rows staged for catalog insert
- execution_state__directory_rules: registered asset directories

Uses SQLAlchemy Core (no ORM) matching the codebase pattern for
library-bookkeeping tables (ManifestStore, ResultCache). See spec
§2.14 for rationale. Schemas are stable; runtime discovery is not
needed here.

All writes use ``with engine.begin() as conn:`` for atomic
per-transaction commit. WAL mode + per-mutation fsync is inherited
from the Workspace engine configuration.
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine

if TYPE_CHECKING:
    from deriva_ml.core.connection_mode import ConnectionMode

logger = logging.getLogger(__name__)


class ExecutionStatus(StrEnum):
    """Lifecycle status for an Execution (see spec §2.2).

    Transitions are:
        created → running → {stopped, failed} →
            {pending_upload → {uploaded, failed}}
        created → aborted
        running → aborted

    Values are lowercase strings for direct storage in SQLite and for
    clean comparison against ERMrest's Status vocabulary terms.
    """
    created = "created"
    running = "running"
    stopped = "stopped"
    failed = "failed"
    pending_upload = "pending_upload"
    uploaded = "uploaded"
    aborted = "aborted"


class PendingRowStatus(StrEnum):
    """Per-pending-row status (see spec §2.5.2).

    Transitions are:
        staged → leasing → leased → uploading → {uploaded, failed}
    """
    staged = "staged"
    leasing = "leasing"
    leased = "leased"
    uploading = "uploading"
    uploaded = "uploaded"
    failed = "failed"


class DirectoryRuleStatus(StrEnum):
    """Per-directory-rule status (see spec §2.5.3).

    A rule is `active` until `close()` is called; closed rules reject
    further register/scan calls but their existing pending_rows can
    still drain.
    """
    active = "active"
    closed = "closed"


EXECUTIONS_TABLE = "execution_state__executions"
PENDING_ROWS_TABLE = "execution_state__pending_rows"
DIRECTORY_RULES_TABLE = "execution_state__directory_rules"


class ExecutionStateStore:
    """SQLAlchemy Core wrapper for the three execution-state tables.

    Owns the MetaData and Table definitions but not the engine — the
    engine is provided by the caller (typically ``Workspace.engine``)
    so all library-bookkeeping tables live in a single main.db.

    Usage:
        >>> store = ExecutionStateStore(engine=workspace.engine)
        >>> store.ensure_schema()
        >>> # then use store.executions, store.pending_rows,
        >>> # store.directory_rules for queries.

    Attributes:
        engine: The shared SQLAlchemy Engine.
        metadata: MetaData object holding the three table definitions.
        executions: The sqlalchemy.Table for executions.
        pending_rows: The sqlalchemy.Table for pending_rows.
        directory_rules: The sqlalchemy.Table for directory_rules.
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
            EXECUTIONS_TABLE, self.metadata,
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

        # pending_rows — see spec §2.5.2. status values:
        # staged|leasing|leased|uploading|uploaded|failed
        self.pending_rows = Table(
            PENDING_ROWS_TABLE, self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column(
                "execution_rid", String,
                ForeignKey(f"{EXECUTIONS_TABLE}.rid"),
                nullable=False,
            ),
            Column("key", String, nullable=False),
            Column("target_schema", String, nullable=False),
            Column("target_table", String, nullable=False),
            Column("rid", String, nullable=True),
            Column("lease_token", String, nullable=True),
            Column("metadata_json", Text, nullable=False),
            Column("asset_file_path", String, nullable=True),
            Column("asset_types_json", Text, nullable=True),
            Column("description", Text, nullable=True),
            Column("status", String, nullable=False),
            Column("error", Text, nullable=True),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("leased_at", DateTime(timezone=True), nullable=True),
            Column("uploaded_at", DateTime(timezone=True), nullable=True),
            Column(
                "rule_id", Integer,
                ForeignKey(f"{DIRECTORY_RULES_TABLE}.id"),
                nullable=True,
            ),
            Index("ix_pending_execution_rid_status", "execution_rid", "status"),
            Index("ix_pending_execution_rid_target_table", "execution_rid", "target_table"),
        )

        # directory_rules — see spec §2.5.3.
        # status values: active|closed
        self.directory_rules = Table(
            DIRECTORY_RULES_TABLE, self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column(
                "execution_rid", String,
                ForeignKey(f"{EXECUTIONS_TABLE}.rid"),
                nullable=False,
            ),
            Column("target_schema", String, nullable=False),
            Column("target_table", String, nullable=False),
            Column("source_dir", String, nullable=False),
            Column("glob", String, nullable=False),
            Column("recurse", Boolean, nullable=False, default=False),
            Column("copy_files", Boolean, nullable=False, default=False),
            Column("asset_types_json", Text, nullable=True),
            Column("status", String, nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Index("ix_directory_rules_execution", "execution_rid"),
        )

    def ensure_schema(self) -> None:
        """Create the three tables if they don't already exist.

        Idempotent — safe to call on every DerivaML construction. Uses
        SQLAlchemy's ``create_all`` which issues ``CREATE TABLE IF
        NOT EXISTS`` via dialect-specific SQL, matching the existing
        Workspace pattern (see ManifestStore.ensure_schema).

        Example:
            >>> store = ExecutionStateStore(engine=workspace.engine)
            >>> store.ensure_schema()
            >>> # Tables now exist; safe to insert/select.
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
            status: Initial status, typically ExecutionStatus.created.
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
                    rid=rid, workflow_rid=workflow_rid,
                    description=description, config_json=config_json,
                    status=str(status), mode=str(mode),
                    working_dir_rel=working_dir_rel,
                    start_time=start_time, stop_time=stop_time,
                    last_activity=last_activity, error=error,
                    sync_pending=sync_pending, created_at=created_at,
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
            >>> row = store.get_execution("EXE-A")
            >>> row["status"] if row else None
            'running'
        """
        with self.engine.connect() as conn:
            result = conn.execute(
                select(self.executions).where(self.executions.c.rid == rid)
            ).mappings().first()
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
        coerced = {
            k: str(v) if isinstance(v, ExecutionStatus) else v
            for k, v in fields.items()
        }

        with self.engine.begin() as conn:
            conn.execute(
                update(self.executions)
                .where(self.executions.c.rid == rid)
                .values(**coerced)
            )

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
            >>> incomplete = [ExecutionStatus.created, ExecutionStatus.running,
            ...               ExecutionStatus.stopped, ExecutionStatus.failed,
            ...               ExecutionStatus.pending_upload]
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
