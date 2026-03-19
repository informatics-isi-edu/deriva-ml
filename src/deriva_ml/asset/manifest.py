"""Persistent JSON manifest for tracking asset state during execution.

The manifest is the single source of truth for all asset metadata during
an execution. It supports:
- Write-through + fsync on every mutation for crash safety
- Per-asset status tracking (pending → uploaded with RID)
- Resume after crash: upload skips entries already marked uploaded
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


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
    """Persistent JSON manifest for execution assets.

    Provides write-through + fsync persistence for crash safety. Each mutation
    (add asset, update metadata, mark uploaded) immediately writes the full
    manifest to disk.

    Args:
        path: Filesystem path for the manifest JSON file.
        execution_rid: RID of the execution this manifest belongs to.

    Example:
        >>> manifest = AssetManifest(Path("/tmp/manifest.json"), "4SP")
        >>> manifest.add_asset("Image/scan.jpg", AssetEntry(
        ...     asset_table="Image", schema="test-schema",
        ...     asset_types=["Training_Data"],
        ...     metadata={"Subject": "2-DEF"}
        ... ))
        >>> manifest.mark_uploaded("Image/scan.jpg", "1-ABC")
    """

    MANIFEST_VERSION = 1

    def __init__(self, path: Path, execution_rid: str) -> None:
        self._path = path
        self._execution_rid = execution_rid
        self._assets: dict[str, AssetEntry] = {}
        self._features: dict[str, FeatureEntry] = {}
        self._created_at: str = datetime.now(timezone.utc).isoformat()

        if path.exists():
            self._load()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._save()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def execution_rid(self) -> str:
        return self._execution_rid

    @property
    def assets(self) -> dict[str, AssetEntry]:
        """All asset entries, keyed by '{AssetTable}/{filename}'."""
        return dict(self._assets)

    @property
    def features(self) -> dict[str, FeatureEntry]:
        """All feature entries, keyed by feature name."""
        return dict(self._features)

    def pending_assets(self) -> dict[str, AssetEntry]:
        """Return only assets with status 'pending'."""
        return {k: v for k, v in self._assets.items() if v.status == "pending"}

    def uploaded_assets(self) -> dict[str, AssetEntry]:
        """Return only assets with status 'uploaded'."""
        return {k: v for k, v in self._assets.items() if v.status == "uploaded"}

    def add_asset(self, key: str, entry: AssetEntry) -> None:
        """Add or replace an asset entry. Writes to disk immediately."""
        self._assets[key] = entry
        self._save()

    def update_asset_metadata(self, key: str, metadata: dict[str, Any]) -> None:
        """Update metadata for an existing asset. Writes to disk immediately."""
        if key not in self._assets:
            raise KeyError(f"Asset '{key}' not in manifest")
        self._assets[key].metadata = metadata
        self._save()

    def update_asset_types(self, key: str, asset_types: list[str]) -> None:
        """Update asset types for an existing asset. Writes to disk immediately."""
        if key not in self._assets:
            raise KeyError(f"Asset '{key}' not in manifest")
        self._assets[key].asset_types = asset_types
        self._save()

    def mark_uploaded(self, key: str, rid: str) -> None:
        """Mark an asset as successfully uploaded with its catalog RID."""
        if key not in self._assets:
            raise KeyError(f"Asset '{key}' not in manifest")
        entry = self._assets[key]
        entry.status = "uploaded"
        entry.rid = rid
        entry.uploaded_at = datetime.now(timezone.utc).isoformat()
        entry.error = None
        self._save()

    def mark_failed(self, key: str, error: str) -> None:
        """Mark an asset as failed with an error message."""
        if key not in self._assets:
            raise KeyError(f"Asset '{key}' not in manifest")
        entry = self._assets[key]
        entry.status = "failed"
        entry.error = error
        self._save()

    def add_feature(self, name: str, entry: FeatureEntry) -> None:
        """Add or replace a feature entry. Writes to disk immediately."""
        self._features[name] = entry
        self._save()

    def _save(self) -> None:
        """Write manifest to disk with fsync for crash safety."""
        data = {
            "version": self.MANIFEST_VERSION,
            "execution_rid": self._execution_rid,
            "created_at": self._created_at,
            "assets": {k: v.to_dict() for k, v in self._assets.items()},
            "features": {k: v.to_dict() for k, v in self._features.items()},
        }
        # Write to temp file then rename for atomicity
        tmp_path = self._path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_json_default)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.rename(self._path)

    def _load(self) -> None:
        """Load manifest from disk."""
        with open(self._path, encoding="utf-8") as f:
            data = json.load(f)

        version = data.get("version", 1)
        if version != self.MANIFEST_VERSION:
            logger.warning(
                f"Manifest version {version} != expected {self.MANIFEST_VERSION}"
            )

        self._execution_rid = data.get("execution_rid", self._execution_rid)
        self._created_at = data.get("created_at", self._created_at)

        self._assets = {
            k: AssetEntry.from_dict(v) for k, v in data.get("assets", {}).items()
        }
        self._features = {
            k: FeatureEntry.from_dict(v) for k, v in data.get("features", {}).items()
        }
