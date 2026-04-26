"""SQLite-backed persistence for AssetManifest.

Stores per-execution asset entries in table ``execution_state__assets`` and
row-per-record staged feature entries in
``execution_state__feature_records`` in the workspace DB.  WAL +
per-mutation commit gives crash safety equivalent to the old JSON
fsync-on-write.

SQLite has no true schema namespacing; we use the ``execution_state__``
prefix as a logical namespace on table names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    Index,
    Integer,
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

from deriva_ml.asset.manifest import AssetEntry

logger = logging.getLogger(__name__)

ASSETS_TABLE = "execution_state__assets"

# Row-per-record staging for feature writes. Replaces the earlier file-based
# FeatureEntry/FEATURES_TABLE mechanism (which wrote values to per-feature .jsonl
# files and tracked them by values_path). One mechanism, one storage layer.
# See docs/superpowers/specs/2026-04-22-feature-api-consistency-design.md §Architecture.
FEATURE_RECORDS_TABLE = "execution_state__feature_records"


@dataclass(frozen=True)
class StagedFeatureRow:
    """One staged feature-record row awaiting flush to the catalog.

    A StagedFeatureRow is one FeatureRecord serialized as JSON, with
    lifecycle status tracked per row. Rows are created by
    ``Execution.add_features()`` and consumed by
    ``Execution._flush_staged_features()``.

    Attributes:
        stage_id: Autoincrement primary key.
        execution_rid: RID of the owning execution.
        feature_table: Qualified feature table name ("schema.Table").
        feature_name: Feature name (redundant with feature_table but kept
            for ergonomic filtering).
        target_table: Target table name (the table the feature is *on*).
            Stored so flush doesn't have to re-resolve it from the model.
        record_json: JSON encoding of ``FeatureRecord.model_dump_json()``.
        created_at: ISO timestamp of staging.
        status: "pending" | "uploaded" | "failed".
        uploaded_at: ISO timestamp of successful flush, or None.
        error: Error message on Failed status, or None.
    """

    stage_id: int
    execution_rid: str
    feature_table: str
    feature_name: str
    target_table: str
    record_json: str
    created_at: str
    status: str
    uploaded_at: str | None
    error: str | None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ManifestStore:
    """SQLite persistence for asset manifest entries and staged feature records.

    Replaces the old ``asset-manifest.json`` write-through file with a
    crash-safe SQLite-backed store.  Every mutation is committed immediately
    (``engine.begin()`` auto-commit) so a process crash cannot corrupt the
    manifest.

    Two logical tables are managed:

    - ``execution_state__assets``: one row per ``(execution_rid, key)`` pair,
      tracking upload status, metadata, and the remote RID once uploaded.
    - ``execution_state__feature_records``: one row per staged FeatureRecord,
      with lifecycle status (Pending → Uploaded | Failed).

    Usage::

        store = ManifestStore(engine)
        store.ensure_schema()          # idempotent: creates tables if absent
        store.add_asset("EX1", "Image/a.jpg", AssetEntry(...))
        store.mark_asset_uploaded("EX1", "Image/a.jpg", rid="2A3B4C")
        pending = store.pending_assets("EX1")
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._metadata = MetaData()
        self._assets_t = Table(
            ASSETS_TABLE,
            self._metadata,
            Column("execution_rid", String, primary_key=True),
            Column("key", String, primary_key=True),
            Column("asset_table", String, nullable=False),
            Column("schema", String, nullable=False),
            Column("asset_types", JSON),
            Column("metadata", JSON),
            Column("description", Text),
            Column("status", String, nullable=False),
            Column("rid", String),
            Column("uploaded_at", String),
            Column("error", Text),
            Column("created_at", String, nullable=False),
            Column("updated_at", String, nullable=False),
        )
        self._feature_records_t = Table(
            FEATURE_RECORDS_TABLE,
            self._metadata,
            Column("stage_id", Integer, primary_key=True, autoincrement=True),
            Column("execution_rid", String, nullable=False),
            Column("feature_table", String, nullable=False),   # "schema.Table"
            Column("feature_name", String, nullable=False),
            Column("target_table", String, nullable=False),    # target table name — avoids re-lookup at flush
            Column("record_json", Text, nullable=False),
            Column("created_at", String, nullable=False),
            Column("status", String, nullable=False),          # pending | uploaded | failed
            Column("uploaded_at", String),
            Column("error", Text),
        )
        Index("ix_assets_exec_status", self._assets_t.c.execution_rid, self._assets_t.c.status)
        Index(
            "ix_execution_state__feature_records_exec_status",
            self._feature_records_t.c.execution_rid,
            self._feature_records_t.c.status,
        )

    def ensure_schema(self) -> None:
        """Create the tables if they don't exist."""
        self._metadata.create_all(self._engine)

    # ---------------- assets ---------------- #

    def add_asset(self, execution_rid: str, key: str, entry: AssetEntry) -> None:
        """Upsert an asset entry into the manifest.

        If an entry for ``(execution_rid, key)`` already exists it is replaced.

        Args:
            execution_rid: RID of the owning execution.
            key: Asset key (typically ``"{AssetTable}/{filename}"``).
            entry: :class:`~deriva_ml.asset.manifest.AssetEntry` with metadata.
        """
        now = _now_iso()
        row = {
            "execution_rid": execution_rid,
            "key": key,
            "asset_table": entry.asset_table,
            "schema": entry.schema,
            "asset_types": entry.asset_types,
            "metadata": entry.metadata,
            "description": entry.description,
            "status": entry.status,
            "rid": entry.rid,
            "uploaded_at": entry.uploaded_at,
            "error": entry.error,
            "created_at": now,
            "updated_at": now,
        }
        with self._engine.begin() as conn:
            conn.execute(
                delete(self._assets_t).where(
                    (self._assets_t.c.execution_rid == execution_rid) & (self._assets_t.c.key == key)
                )
            )
            conn.execute(insert(self._assets_t), row)

    def get_asset(self, execution_rid: str, key: str) -> AssetEntry:
        """Return the :class:`AssetEntry` for ``(execution_rid, key)``.

        Args:
            execution_rid: Owning execution RID.
            key: Asset key.

        Returns:
            The matching :class:`~deriva_ml.asset.manifest.AssetEntry`.

        Raises:
            KeyError: If no entry exists for the given ``(execution_rid, key)`` pair.
        """
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    select(self._assets_t).where(
                        (self._assets_t.c.execution_rid == execution_rid) & (self._assets_t.c.key == key)
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            raise KeyError(f"Asset '{key}' for execution '{execution_rid}' not found")
        return AssetEntry(
            asset_table=row["asset_table"],
            schema=row["schema"],
            asset_types=row["asset_types"] or [],
            metadata=row["metadata"] or {},
            description=row["description"],
            status=row["status"],
            rid=row["rid"],
            uploaded_at=row["uploaded_at"],
            error=row["error"],
        )

    def list_assets(self, execution_rid: str) -> dict[str, AssetEntry]:
        """Return all asset entries for an execution as ``{key: AssetEntry}``.

        Args:
            execution_rid: Owning execution RID.

        Returns:
            Dict mapping asset key to :class:`~deriva_ml.asset.manifest.AssetEntry`.
        """
        with self._engine.connect() as conn:
            rows = (
                conn.execute(select(self._assets_t).where(self._assets_t.c.execution_rid == execution_rid))
                .mappings()
                .all()
            )
        return {
            r["key"]: AssetEntry(
                asset_table=r["asset_table"],
                schema=r["schema"],
                asset_types=r["asset_types"] or [],
                metadata=r["metadata"] or {},
                description=r["description"],
                status=r["status"],
                rid=r["rid"],
                uploaded_at=r["uploaded_at"],
                error=r["error"],
            )
            for r in rows
        }

    def pending_assets(self, execution_rid: str) -> dict[str, AssetEntry]:
        """Return only assets with ``status == "pending"`` for *execution_rid*.

        Args:
            execution_rid: Owning execution RID.

        Returns:
            Filtered subset of :meth:`list_assets`.
        """
        return {k: v for k, v in self.list_assets(execution_rid).items() if v.status == "pending"}

    def uploaded_assets(self, execution_rid: str) -> dict[str, AssetEntry]:
        """Return only assets with ``status == "uploaded"`` for *execution_rid*.

        Args:
            execution_rid: Owning execution RID.

        Returns:
            Filtered subset of :meth:`list_assets`.
        """
        return {k: v for k, v in self.list_assets(execution_rid).items() if v.status == "uploaded"}

    def update_asset_metadata(self, execution_rid: str, key: str, metadata: dict[str, Any]) -> None:
        """Update the metadata JSON blob for an existing asset entry.

        Args:
            execution_rid: Owning execution RID.
            key: Asset key.
            metadata: New metadata dict to store.

        Raises:
            KeyError: If the entry does not exist.
        """
        self._require_asset(execution_rid, key)
        with self._engine.begin() as conn:
            conn.execute(
                update(self._assets_t)
                .where((self._assets_t.c.execution_rid == execution_rid) & (self._assets_t.c.key == key))
                .values(metadata=metadata, updated_at=_now_iso())
            )

    def update_asset_types(self, execution_rid: str, key: str, asset_types: list[str]) -> None:
        """Replace the ``asset_types`` list for an existing asset entry.

        Args:
            execution_rid: Owning execution RID.
            key: Asset key.
            asset_types: New list of asset type strings.

        Raises:
            KeyError: If the entry does not exist.
        """
        self._require_asset(execution_rid, key)
        with self._engine.begin() as conn:
            conn.execute(
                update(self._assets_t)
                .where((self._assets_t.c.execution_rid == execution_rid) & (self._assets_t.c.key == key))
                .values(asset_types=asset_types, updated_at=_now_iso())
            )

    def mark_asset_uploaded(self, execution_rid: str, key: str, rid: str) -> None:
        """Transition an asset entry to ``status="uploaded"`` and record its catalog RID.

        Args:
            execution_rid: Owning execution RID.
            key: Asset key.
            rid: The catalog RID assigned to the uploaded asset.

        Raises:
            KeyError: If the entry does not exist.
        """
        self._require_asset(execution_rid, key)
        now = _now_iso()
        with self._engine.begin() as conn:
            conn.execute(
                update(self._assets_t)
                .where((self._assets_t.c.execution_rid == execution_rid) & (self._assets_t.c.key == key))
                .values(
                    status="uploaded",
                    rid=rid,
                    uploaded_at=now,
                    error=None,
                    updated_at=now,
                )
            )

    def mark_assets_uploaded(
        self, execution_rid: str, items: "list[tuple[str, str]]"
    ) -> None:
        """Bulk transition multiple asset entries to ``status="uploaded"``.

        Equivalent to calling :meth:`mark_asset_uploaded` for each
        ``(key, rid)`` pair, but executes the UPDATE statements inside a
        **single** SQLite transaction. For an upload of N files, this
        replaces N independent ``engine.begin()`` blocks (each with its
        own commit + WAL fsync) with one — turning an O(N × fsync_cost)
        loop into one batched fsync. On a 10K-file load, the difference
        is roughly 18 minutes (per-row fsync) → a few seconds (single
        fsync), measured against an SSD on macOS.

        Skips the per-row existence check that
        :meth:`mark_asset_uploaded` performs via ``_require_asset`` —
        the caller is expected to be in the post-upload-success path
        where every key in ``items`` was just successfully uploaded
        and is known to be present.

        Args:
            execution_rid: Owning execution RID.
            items: List of ``(key, rid)`` tuples. Empty list is a no-op.

        Note:
            This intentionally does not raise on missing keys (the
            single-row method does). Wrapping the bulk path in
            existence checks would re-introduce the per-row SELECTs
            this method is meant to eliminate. The contract is
            "caller has the keys."
        """
        if not items:
            return
        now = _now_iso()
        with self._engine.begin() as conn:
            for key, rid in items:
                conn.execute(
                    update(self._assets_t)
                    .where(
                        (self._assets_t.c.execution_rid == execution_rid)
                        & (self._assets_t.c.key == key)
                    )
                    .values(
                        status="uploaded",
                        rid=rid,
                        uploaded_at=now,
                        error=None,
                        updated_at=now,
                    )
                )

    def set_asset_rid(self, execution_rid: str, key: str, rid: str) -> None:
        """Assign a pre-leased RID to an asset entry without changing status.

        Used when the RID is known in advance (from ``ERMrest_RID_Lease``)
        so the catalog insert at upload time can use the caller-supplied
        RID. Unlike :meth:`mark_asset_uploaded`, this leaves ``status``
        and ``uploaded_at`` unchanged.

        Args:
            execution_rid: Owning execution RID.
            key: Asset key.
            rid: Pre-allocated RID to assign to the entry.

        Raises:
            KeyError: If the entry does not exist.
        """
        self._require_asset(execution_rid, key)
        with self._engine.begin() as conn:
            conn.execute(
                update(self._assets_t)
                .where((self._assets_t.c.execution_rid == execution_rid) & (self._assets_t.c.key == key))
                .values(rid=rid, updated_at=_now_iso())
            )

    def mark_asset_failed(self, execution_rid: str, key: str, error: str) -> None:
        """Transition an asset entry to ``status="failed"`` and record the error message.

        Args:
            execution_rid: Owning execution RID.
            key: Asset key.
            error: Human-readable error description.

        Raises:
            KeyError: If the entry does not exist.
        """
        self._require_asset(execution_rid, key)
        with self._engine.begin() as conn:
            conn.execute(
                update(self._assets_t)
                .where((self._assets_t.c.execution_rid == execution_rid) & (self._assets_t.c.key == key))
                .values(
                    status="failed",
                    error=error,
                    updated_at=_now_iso(),
                )
            )

    def _require_asset(self, execution_rid: str, key: str) -> None:
        with self._engine.connect() as conn:
            exists = conn.execute(
                select(self._assets_t.c.key).where(
                    (self._assets_t.c.execution_rid == execution_rid) & (self._assets_t.c.key == key)
                )
            ).first()
        if exists is None:
            raise KeyError(f"Asset '{key}' for execution '{execution_rid}' not found")

    # ---------------- staged feature records ---------------- #

    def stage_feature_record(
        self,
        execution_rid: str,
        feature_table: str,
        feature_name: str,
        target_table: str,
        record_json: str,
    ) -> int:
        """Insert one staged feature record row with status ``"pending"``.

        Args:
            execution_rid: RID of the owning execution.
            feature_table: Qualified feature table name (``"schema.Table"``).
            feature_name: Feature name (e.g. ``"Glaucoma"``).
            target_table: Target table name (the table the feature is *on*).
            record_json: JSON-serialized FeatureRecord dict.

        Returns:
            The autoincrement ``stage_id`` of the inserted row.
        """
        now = _now_iso()
        row = {
            "execution_rid": execution_rid,
            "feature_table": feature_table,
            "feature_name": feature_name,
            "target_table": target_table,
            "record_json": record_json,
            "created_at": now,
            "status": "pending",
            "uploaded_at": None,
            "error": None,
        }
        with self._engine.begin() as conn:
            result = conn.execute(insert(self._feature_records_t), row)
            return result.inserted_primary_key[0]

    def _row_to_staged_feature_record(self, r: Any) -> StagedFeatureRow:
        """Convert a DB row mapping to a :class:`StagedFeatureRow`."""
        return StagedFeatureRow(
            stage_id=r["stage_id"],
            execution_rid=r["execution_rid"],
            feature_table=r["feature_table"],
            feature_name=r["feature_name"],
            target_table=r["target_table"],
            record_json=r["record_json"],
            created_at=r["created_at"],
            status=r["status"],
            uploaded_at=r["uploaded_at"],
            error=r["error"],
        )

    def list_feature_records(self, execution_rid: str) -> list[StagedFeatureRow]:
        """Return all staged feature rows for *execution_rid*, ordered by stage_id.

        Args:
            execution_rid: Owning execution RID.

        Returns:
            List of :class:`StagedFeatureRow` in insertion order.
        """
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    select(self._feature_records_t)
                    .where(self._feature_records_t.c.execution_rid == execution_rid)
                    .order_by(self._feature_records_t.c.stage_id)
                )
                .mappings()
                .all()
            )
        return [self._row_to_staged_feature_record(r) for r in rows]

    def list_pending_feature_records(self, execution_rid: str) -> list[StagedFeatureRow]:
        """Return only rows with ``status == "pending"`` for *execution_rid*.

        Args:
            execution_rid: Owning execution RID.

        Returns:
            Subset of :meth:`list_feature_records` filtered to pending rows.
        """
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    select(self._feature_records_t)
                    .where(
                        (self._feature_records_t.c.execution_rid == execution_rid)
                        & (self._feature_records_t.c.status == "pending")
                    )
                    .order_by(self._feature_records_t.c.stage_id)
                )
                .mappings()
                .all()
            )
        return [self._row_to_staged_feature_record(r) for r in rows]

    def _require_feature_record(self, stage_id: int) -> None:
        with self._engine.connect() as conn:
            exists = conn.execute(
                select(self._feature_records_t.c.stage_id).where(
                    self._feature_records_t.c.stage_id == stage_id
                )
            ).first()
        if exists is None:
            raise KeyError(f"Staged feature record with stage_id={stage_id!r} not found")

    def mark_feature_record_uploaded(self, stage_id: int) -> None:
        """Transition a staged feature record to ``status="uploaded"``.

        Args:
            stage_id: Primary key of the row to update.

        Raises:
            KeyError: If no row with *stage_id* exists.
        """
        self._require_feature_record(stage_id)
        now = _now_iso()
        with self._engine.begin() as conn:
            conn.execute(
                update(self._feature_records_t)
                .where(self._feature_records_t.c.stage_id == stage_id)
                .values(status="uploaded", uploaded_at=now, error=None)
            )

    def mark_feature_record_failed(self, stage_id: int, error: str) -> None:
        """Transition a staged feature record to ``status="failed"`` and record the error.

        Args:
            stage_id: Primary key of the row to update.
            error: Human-readable error description.

        Raises:
            KeyError: If no row with *stage_id* exists.
        """
        self._require_feature_record(stage_id)
        with self._engine.begin() as conn:
            conn.execute(
                update(self._feature_records_t)
                .where(self._feature_records_t.c.stage_id == stage_id)
                .values(status="failed", error=error)
            )
