"""Persistent manifest for tracking asset state during execution.

The manifest is the single source of truth for all asset metadata during
an execution. It supports:
- Per-asset status tracking (pending → uploaded with RID)
- Resume after crash: upload skips entries already marked uploaded
- SQLite (WAL mode) for crash safety and cross-process visibility
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deriva_ml.local_db.manifest_store import ManifestStore


def _json_default(obj: Any) -> Any:
    """JSON serializer for objects not natively handled by json module.

    Handles datetime.datetime, datetime.date, and Path objects that may
    appear in asset metadata from catalog column values.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


logger = logging.getLogger(__name__)


@dataclass
class AssetEntry:
    """A single asset entry in the manifest."""

    asset_table: str
    schema: str
    asset_types: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    description: str | None = None
    status: str = "pending"  # pending | uploaded | failed
    rid: str | None = None
    uploaded_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssetEntry:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class FeatureEntry:
    """A feature entry in the manifest."""

    feature_name: str
    target_table: str
    schema: str
    values_path: str
    asset_columns: dict[str, str] = field(default_factory=dict)
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureEntry:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class AssetManifest:
    """Per-execution asset+feature manifest, backed by :class:`ManifestStore`.

    Public API is unchanged from the JSON-backed implementation. Storage
    swapped to SQLite (WAL mode) for crash safety and cross-process visibility.

    Args:
        store: A ``ManifestStore`` bound to a workspace engine.
        execution_rid: RID of the execution this manifest covers.
    """

    MANIFEST_VERSION = 2  # bumped: storage layer changed

    def __init__(self, store: "ManifestStore", execution_rid: str) -> None:
        self._store = store
        self._execution_rid = execution_rid

    @property
    def execution_rid(self) -> str:
        return self._execution_rid

    @property
    def assets(self) -> dict[str, AssetEntry]:
        return self._store.list_assets(self._execution_rid)

    @property
    def features(self) -> dict[str, FeatureEntry]:
        return self._store.list_features(self._execution_rid)

    def pending_assets(self) -> dict[str, AssetEntry]:
        return self._store.pending_assets(self._execution_rid)

    def uploaded_assets(self) -> dict[str, AssetEntry]:
        return self._store.uploaded_assets(self._execution_rid)

    def add_asset(self, key: str, entry: AssetEntry) -> None:
        self._store.add_asset(self._execution_rid, key, entry)

    def update_asset_metadata(self, key: str, metadata: dict[str, Any]) -> None:
        self._store.update_asset_metadata(self._execution_rid, key, metadata)

    def update_asset_types(self, key: str, asset_types: list[str]) -> None:
        self._store.update_asset_types(self._execution_rid, key, asset_types)

    def mark_uploaded(self, key: str, rid: str) -> None:
        self._store.mark_asset_uploaded(self._execution_rid, key, rid)

    def mark_failed(self, key: str, error: str) -> None:
        self._store.mark_asset_failed(self._execution_rid, key, error)

    def add_feature(self, name: str, entry: FeatureEntry) -> None:
        self._store.add_feature(self._execution_rid, name, entry)

    def to_json(self) -> dict[str, Any]:
        """Return a dict mirroring the legacy JSON file format.

        For debugging/postmortems. Serialize with
        ``json.dumps(manifest.to_json(), default=_json_default)`` to handle
        datetimes and Path values.
        """
        return {
            "version": self.MANIFEST_VERSION,
            "execution_rid": self._execution_rid,
            "assets": {k: v.to_dict() for k, v in self.assets.items()},
            "features": {k: v.to_dict() for k, v in self.features.items()},
        }
