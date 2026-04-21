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
    case,
    delete,
    func,
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
                ForeignKey(
                    f"{EXECUTIONS_TABLE}.rid",
                    name="fk_pending_rows_execution_rid_fkey",
                ),
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
                ForeignKey(
                    f"{DIRECTORY_RULES_TABLE}.id",
                    name="fk_pending_rows_rule_id_fkey",
                ),
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
                ForeignKey(
                    f"{EXECUTIONS_TABLE}.rid",
                    name="fk_directory_rules_execution_rid_fkey",
                ),
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

    # ─── pending_rows CRUD ──────────────────────────────────────────

    def insert_pending_row(
        self,
        *,
        execution_rid: str,
        key: str,
        target_schema: str,
        target_table: str,
        metadata_json: str,
        created_at: datetime,
        rid: str | None = None,
        lease_token: str | None = None,
        asset_file_path: str | None = None,
        asset_types_json: str | None = None,
        description: str | None = None,
        status: PendingRowStatus = PendingRowStatus.staged,
        rule_id: int | None = None,
    ) -> int:
        """Insert one pending_rows entry.

        Args:
            execution_rid: FK to executions.rid.
            key: Stable identifier for dedup (auto-hash for ad-hoc
                rows; rule_id+filename for directory-sourced rows).
            target_schema / target_table: Catalog target.
            metadata_json: Serialized column values.
            created_at: UTC timestamp.
            rid: Leased RID, None until leased.
            lease_token: Token for two-phase lease reconciliation.
            asset_file_path: Local file path, None for plain rows.
            asset_types_json: Serialized asset-type terms.
            description: Optional human-readable description.
            status: Initial status, defaults to 'staged'.
            rule_id: FK to directory_rules.id, None if not from a rule.

        Returns:
            The auto-assigned integer id of the new pending_rows row.
        """
        with self.engine.begin() as conn:
            result = conn.execute(
                insert(self.pending_rows).values(
                    execution_rid=execution_rid, key=key,
                    target_schema=target_schema, target_table=target_table,
                    rid=rid, lease_token=lease_token,
                    metadata_json=metadata_json,
                    asset_file_path=asset_file_path,
                    asset_types_json=asset_types_json,
                    description=description,
                    status=str(status),
                    created_at=created_at,
                    rule_id=rule_id,
                )
            )
            # SQLite returns the auto-increment id via lastrowid.
            return int(result.inserted_primary_key[0])

    def update_pending_row(self, pending_id: int, **fields: object) -> None:
        """Partial update of a pending_rows entry.

        Status / token / rid / timestamps are the common callers. Enum
        values are coerced to strings.

        Args:
            pending_id: The integer id of the row to update.
            **fields: Columns to set.

        Raises:
            KeyError: If a kwarg doesn't match a pending_rows column.
        """
        valid_cols = {c.name for c in self.pending_rows.columns}
        unknown = set(fields) - valid_cols
        if unknown:
            raise KeyError(f"unknown columns on pending_rows: {unknown}")

        coerced = {
            k: str(v) if isinstance(v, PendingRowStatus) else v
            for k, v in fields.items()
        }

        with self.engine.begin() as conn:
            conn.execute(
                update(self.pending_rows)
                .where(self.pending_rows.c.id == pending_id)
                .values(**coerced)
            )

    def list_pending_rows(
        self,
        *,
        execution_rid: str,
        status: "PendingRowStatus | list[PendingRowStatus] | None" = None,
        target_table: str | None = None,
    ) -> list[dict]:
        """Return pending_rows entries scoped to one execution.

        Args:
            execution_rid: Required — pending rows are always scoped
                to a specific execution.
            status: Filter to a status or list of statuses.
            target_table: Filter to a single target table.

        Returns:
            List of dicts — empty if nothing matches.
        """
        stmt = select(self.pending_rows).where(
            self.pending_rows.c.execution_rid == execution_rid
        )
        if status is not None:
            if isinstance(status, PendingRowStatus):
                statuses = [str(status)]
            else:
                statuses = [str(s) for s in status]
            stmt = stmt.where(self.pending_rows.c.status.in_(statuses))
        if target_table is not None:
            stmt = stmt.where(self.pending_rows.c.target_table == target_table)

        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]

    def count_pending_by_kind(
        self,
        *,
        execution_rid: str,
    ) -> dict[str, int]:
        """Return per-kind counts of non-terminal pending rows.

        A "pending" row is in one of staged/leasing/leased/uploading
        (not yet terminally uploaded or failed). A "failed" row is
        specifically in status='failed'. Rows in status='uploaded'
        are excluded from both counts.

        "plain" vs "asset" is determined by asset_file_path — non-null
        means it's an asset row.

        Args:
            execution_rid: Scoping. Required — pending rows are
                execution-scoped.

        Returns:
            A dict with keys pending_rows, failed_rows, pending_files,
            failed_files. All four keys are always present; empty
            tables yield zero for each (via COALESCE).

        Example:
            >>> store.count_pending_by_kind(execution_rid="EXE-A")
            {'pending_rows': 5, 'failed_rows': 0,
             'pending_files': 12, 'failed_files': 1}
        """
        # Single aggregate query, branched by asset_file_path IS NULL.
        # case() produces 1 or 0 per row matching the branch; sum
        # gives the count. This is ~4x faster than 4 separate
        # queries for large pending_rows tables.
        pending_statuses = [
            str(PendingRowStatus.staged),
            str(PendingRowStatus.leasing),
            str(PendingRowStatus.leased),
            str(PendingRowStatus.uploading),
        ]
        failed_status = str(PendingRowStatus.failed)

        is_plain = self.pending_rows.c.asset_file_path.is_(None)
        is_asset = self.pending_rows.c.asset_file_path.isnot(None)
        status_col = self.pending_rows.c.status

        stmt = select(
            func.coalesce(
                func.sum(
                    case((is_plain & status_col.in_(pending_statuses), 1), else_=0)
                ),
                0,
            ).label("pending_rows"),
            func.coalesce(
                func.sum(
                    case((is_plain & (status_col == failed_status), 1), else_=0)
                ),
                0,
            ).label("failed_rows"),
            func.coalesce(
                func.sum(
                    case((is_asset & status_col.in_(pending_statuses), 1), else_=0)
                ),
                0,
            ).label("pending_files"),
            func.coalesce(
                func.sum(
                    case((is_asset & (status_col == failed_status), 1), else_=0)
                ),
                0,
            ).label("failed_files"),
        ).where(self.pending_rows.c.execution_rid == execution_rid)

        with self.engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        return {
            "pending_rows": int(row["pending_rows"]),
            "failed_rows": int(row["failed_rows"]),
            "pending_files": int(row["pending_files"]),
            "failed_files": int(row["failed_files"]),
        }

    def pending_summary_rows(
        self,
        *,
        execution_rid: str,
    ) -> dict:
        """Return the data needed to build a PendingSummary.

        Returns:
            Dict with keys 'rows' (list of per-row-table counts),
            'assets' (list of per-asset-table counts with bytes), and
            'diagnostics' (list of error messages from failed rows).

            rows entries: {table, pending, failed, uploaded}
            assets entries: {table, pending_files, failed_files,
                             uploaded_files, total_bytes_pending}

        Example:
            >>> data = store.pending_summary_rows(execution_rid="EXE-A")
            >>> # Caller builds PendingSummary from this.
        """
        import os

        from sqlalchemy import and_

        pending_statuses = [
            str(PendingRowStatus.staged),
            str(PendingRowStatus.leasing),
            str(PendingRowStatus.leased),
            str(PendingRowStatus.uploading),
        ]
        failed_status = str(PendingRowStatus.failed)
        uploaded_status = str(PendingRowStatus.uploaded)

        is_asset = self.pending_rows.c.asset_file_path.isnot(None)

        stmt = select(
            self.pending_rows.c.target_schema,
            self.pending_rows.c.target_table,
            is_asset.label("is_asset"),
            func.sum(
                case((self.pending_rows.c.status.in_(pending_statuses), 1),
                     else_=0)
            ).label("pending"),
            func.sum(
                case((self.pending_rows.c.status == failed_status, 1),
                     else_=0)
            ).label("failed"),
            func.sum(
                case((self.pending_rows.c.status == uploaded_status, 1),
                     else_=0)
            ).label("uploaded"),
        ).where(
            self.pending_rows.c.execution_rid == execution_rid
        ).group_by(
            self.pending_rows.c.target_schema,
            self.pending_rows.c.target_table,
            is_asset,
        )

        rows_out: list[dict] = []
        assets_out: list[dict] = []
        with self.engine.connect() as conn:
            for r in conn.execute(stmt).mappings().all():
                table_fqn = f"{r['target_schema']}:{r['target_table']}"
                if r["is_asset"]:
                    bytes_stmt = select(
                        self.pending_rows.c.asset_file_path
                    ).where(
                        and_(
                            self.pending_rows.c.execution_rid == execution_rid,
                            self.pending_rows.c.target_schema == r["target_schema"],
                            self.pending_rows.c.target_table == r["target_table"],
                            self.pending_rows.c.status.in_(pending_statuses),
                        )
                    )
                    total_bytes = 0
                    for (p,) in conn.execute(bytes_stmt).all():
                        try:
                            total_bytes += os.path.getsize(p)
                        except OSError:
                            pass
                    assets_out.append({
                        "table": table_fqn,
                        "pending_files": int(r["pending"]),
                        "failed_files": int(r["failed"]),
                        "uploaded_files": int(r["uploaded"]),
                        "total_bytes_pending": int(total_bytes),
                    })
                else:
                    rows_out.append({
                        "table": table_fqn,
                        "pending": int(r["pending"]),
                        "failed": int(r["failed"]),
                        "uploaded": int(r["uploaded"]),
                    })

            diag_stmt = select(
                self.pending_rows.c.target_table,
                self.pending_rows.c.rid,
                self.pending_rows.c.error,
            ).where(
                and_(
                    self.pending_rows.c.execution_rid == execution_rid,
                    self.pending_rows.c.status == failed_status,
                )
            ).limit(20)
            diagnostics: list[str] = []
            for dr in conn.execute(diag_stmt).mappings().all():
                ident = dr["rid"] or "(unleased)"
                diagnostics.append(
                    f"{dr['target_table']} row {ident} failed: {dr['error']}"
                )

        return {
            "rows": rows_out,
            "assets": assets_out,
            "diagnostics": diagnostics,
        }

    # ─── lease two-phase protocol ───────────────────────────────────

    def mark_pending_leasing(self, pending_id: int, *, lease_token: str) -> None:
        """Phase 1 of RID leasing: write lease_token + status='leasing'.

        Must be committed BEFORE the POST to ERMrest_RID_Lease so that
        a crash between this write and the POST is recoverable via
        revert_pending_leasing (token wasn't yet sent, so no server
        state to reconcile).

        Args:
            pending_id: pending_rows.id to transition.
            lease_token: UUID string. Same token goes in the POST body.

        Example:
            >>> token = generate_lease_token()
            >>> store.mark_pending_leasing(pid, lease_token=token)
            >>> # Now POST the token to ERMrest_RID_Lease.
        """
        self.update_pending_row(
            pending_id,
            status=PendingRowStatus.leasing,
            lease_token=lease_token,
        )

    def finalize_pending_lease(
        self,
        *,
        lease_token: str,
        assigned_rid: str,
    ) -> None:
        """Phase 2: POST succeeded, assign the server RID and flip to
        'leased'.

        Identified by lease_token (not pending_id) so this works for
        batched lease responses without requiring the caller to hold
        pending_id↔token mappings.

        Args:
            lease_token: The token we POSTed and got a RID back for.
            assigned_rid: The server-assigned RID from the response.

        Example:
            >>> store.finalize_pending_lease(
            ...     lease_token="uuid...", assigned_rid="1-ABCD"
            ... )
        """
        from datetime import datetime, timezone

        with self.engine.begin() as conn:
            conn.execute(
                update(self.pending_rows)
                .where(self.pending_rows.c.lease_token == lease_token)
                .values(
                    rid=assigned_rid,
                    status=str(PendingRowStatus.leased),
                    leased_at=datetime.now(timezone.utc),
                )
            )

    def revert_pending_leasing(self, *, lease_token: str) -> None:
        """Rollback: clear lease_token and flip back to 'staged'.

        Called either:
          (a) right after a failed POST (token never landed on server), or
          (b) during startup reconciliation when the token query to
              ERMrest_RID_Lease returns nothing (POST failed silently
              or was dropped before persisting).

        Args:
            lease_token: The token to clear.

        Example:
            >>> store.revert_pending_leasing(lease_token="uuid...")
            >>> # Row is now back to status='staged', ready to re-issue.
        """
        with self.engine.begin() as conn:
            conn.execute(
                update(self.pending_rows)
                .where(self.pending_rows.c.lease_token == lease_token)
                .values(
                    lease_token=None,
                    status=str(PendingRowStatus.staged),
                )
            )

    def list_leasing_rows(
        self,
        *,
        execution_rid: str | None = None,
    ) -> list[dict]:
        """Return rows currently in status='leasing' — candidates for
        startup reconciliation.

        Args:
            execution_rid: If set, scope to one execution; if None,
                return all leasing rows across all executions
                (workspace-wide reconciliation).

        Returns:
            List of pending_rows dicts.

        Example:
            >>> # Workspace-wide sweep at startup:
            >>> for r in store.list_leasing_rows():
            ...     print(r["lease_token"], r["execution_rid"])
        """
        stmt = select(self.pending_rows).where(
            self.pending_rows.c.status == str(PendingRowStatus.leasing)
        )
        if execution_rid is not None:
            stmt = stmt.where(self.pending_rows.c.execution_rid == execution_rid)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]

    # ─── directory_rules CRUD ───────────────────────────────────────

    def insert_directory_rule(
        self,
        *,
        execution_rid: str,
        target_schema: str,
        target_table: str,
        source_dir: str,
        glob: str,
        recurse: bool,
        copy_files: bool,
        asset_types_json: str | None,
        created_at: datetime,
        status: DirectoryRuleStatus = DirectoryRuleStatus.active,
    ) -> int:
        """Insert one directory_rules entry; return its auto id.

        Args:
            execution_rid: FK to executions.rid.
            target_schema / target_table: Catalog target for rows
                produced by this rule.
            source_dir: Local directory to scan.
            glob: Pattern for files under source_dir.
            recurse: Whether to scan recursively.
            copy_files: Whether to copy files into workspace staging
                or reference in place.
            asset_types_json: Serialized asset-type terms applied
                to every file registered under this rule.
            created_at: UTC timestamp.
            status: Initial status, defaults to 'active'.

        Returns:
            The auto-assigned integer id.
        """
        with self.engine.begin() as conn:
            result = conn.execute(
                insert(self.directory_rules).values(
                    execution_rid=execution_rid,
                    target_schema=target_schema, target_table=target_table,
                    source_dir=source_dir,
                    glob=glob, recurse=recurse, copy_files=copy_files,
                    asset_types_json=asset_types_json,
                    status=str(status),
                    created_at=created_at,
                )
            )
            return int(result.inserted_primary_key[0])

    def update_directory_rule(self, rule_id: int, **fields: object) -> None:
        """Partial update of a directory_rules entry.

        Args:
            rule_id: The integer id of the rule to update.
            **fields: Columns to set.

        Raises:
            KeyError: If a kwarg doesn't match a directory_rules column.
        """
        valid_cols = {c.name for c in self.directory_rules.columns}
        unknown = set(fields) - valid_cols
        if unknown:
            raise KeyError(f"unknown columns on directory_rules: {unknown}")
        coerced = {
            k: str(v) if isinstance(v, DirectoryRuleStatus) else v
            for k, v in fields.items()
        }

        with self.engine.begin() as conn:
            conn.execute(
                update(self.directory_rules)
                .where(self.directory_rules.c.id == rule_id)
                .values(**coerced)
            )

    def list_directory_rules(
        self,
        *,
        execution_rid: str,
        status: "DirectoryRuleStatus | None" = None,
    ) -> list[dict]:
        """List directory_rules for one execution, optionally filtered.

        Args:
            execution_rid: Required scoping.
            status: Filter to this status, or None for all.

        Returns:
            List of dicts — empty if nothing matches.
        """
        stmt = select(self.directory_rules).where(
            self.directory_rules.c.execution_rid == execution_rid
        )
        if status is not None:
            stmt = stmt.where(self.directory_rules.c.status == str(status))

        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]

    def delete_execution(self, execution_rid: str) -> None:
        """Delete an execution row and all its pending_rows /
        directory_rules.

        Foreign keys cascade via ON DELETE, but SQLite only honors
        that with PRAGMA foreign_keys=ON (which the workspace sets).
        Belt-and-suspenders: we explicitly delete children first, so
        the ORDER of deletions is predictable and so callers running
        without FK-on get sensible behavior.

        Args:
            execution_rid: Which execution to remove.

        Example:
            >>> store.delete_execution("EXE-A")
            >>> store.get_execution("EXE-A") is None
            True
        """
        with self.engine.begin() as conn:
            conn.execute(
                delete(self.pending_rows).where(
                    self.pending_rows.c.execution_rid == execution_rid
                )
            )
            conn.execute(
                delete(self.directory_rules).where(
                    self.directory_rules.c.execution_rid == execution_rid
                )
            )
            conn.execute(
                delete(self.executions).where(
                    self.executions.c.rid == execution_rid
                )
            )
