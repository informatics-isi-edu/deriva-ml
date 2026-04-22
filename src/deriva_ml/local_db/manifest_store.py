"""SQLite-backed persistence for AssetManifest.

Stores per-execution asset and feature entries in tables
``execution_state__assets`` and ``execution_state__features`` in the
workspace DB. WAL + per-mutation commit gives crash safety equivalent to the
old JSON fsync-on-write.

SQLite has no true schema namespacing; we use the ``execution_state__``
prefix as a logical namespace on table names.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
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

from deriva_ml.asset.manifest import AssetEntry, FeatureEntry

logger = logging.getLogger(__name__)

ASSETS_TABLE = "execution_state__assets"
FEATURES_TABLE = "execution_state__features"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ManifestStore:
    """SQLite persistence for asset + feature manifest entries.

    Replaces the old ``asset-manifest.json`` write-through file with a
    crash-safe SQLite-backed store.  Every mutation is committed immediately
    (``engine.begin()`` auto-commit) so a process crash cannot corrupt the
    manifest.

    Two logical tables are managed:

    - ``execution_state__assets``: one row per ``(execution_rid, key)`` pair,
      tracking upload status, metadata, and the remote RID once uploaded.
    - ``execution_state__features``: one row per ``(execution_rid, feature_name)``
      pair, tracking the feature table, schema, values path, and upload status.

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
        self._features_t = Table(
            FEATURES_TABLE,
            self._metadata,
            Column("execution_rid", String, primary_key=True),
            Column("feature_name", String, primary_key=True),
            Column("target_table", String, nullable=False),
            Column("schema", String, nullable=False),
            Column("values_path", String, nullable=False),
            Column("asset_columns", JSON),
            Column("status", String, nullable=False),
            Column("created_at", String, nullable=False),
            Column("updated_at", String, nullable=False),
        )
        Index("ix_assets_exec_status", self._assets_t.c.execution_rid, self._assets_t.c.status)
        Index("ix_features_exec_status", self._features_t.c.execution_rid, self._features_t.c.status)

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

    # ---------------- features ---------------- #

    def add_feature(self, execution_rid: str, feature_name: str, entry: FeatureEntry) -> None:
        """Upsert a feature entry into the manifest.

        If an entry for ``(execution_rid, feature_name)`` already exists it is replaced.

        Args:
            execution_rid: RID of the owning execution.
            feature_name: Feature name (e.g., ``"Classification"``).
            entry: :class:`~deriva_ml.asset.manifest.FeatureEntry` with metadata.
        """
        now = _now_iso()
        row = {
            "execution_rid": execution_rid,
            "feature_name": feature_name,
            "target_table": entry.target_table,
            "schema": entry.schema,
            "values_path": entry.values_path,
            "asset_columns": entry.asset_columns,
            "status": entry.status,
            "created_at": now,
            "updated_at": now,
        }
        with self._engine.begin() as conn:
            conn.execute(
                delete(self._features_t).where(
                    (self._features_t.c.execution_rid == execution_rid)
                    & (self._features_t.c.feature_name == feature_name)
                )
            )
            conn.execute(insert(self._features_t), row)

    def list_features(self, execution_rid: str) -> dict[str, FeatureEntry]:
        """Return all feature entries for an execution as ``{feature_name: FeatureEntry}``.

        Args:
            execution_rid: Owning execution RID.

        Returns:
            Dict mapping feature name to :class:`~deriva_ml.asset.manifest.FeatureEntry`.
        """
        with self._engine.connect() as conn:
            rows = (
                conn.execute(select(self._features_t).where(self._features_t.c.execution_rid == execution_rid))
                .mappings()
                .all()
            )
        return {
            r["feature_name"]: FeatureEntry(
                feature_name=r["feature_name"],
                target_table=r["target_table"],
                schema=r["schema"],
                values_path=r["values_path"],
                asset_columns=r["asset_columns"] or {},
                status=r["status"],
            )
            for r in rows
        }
