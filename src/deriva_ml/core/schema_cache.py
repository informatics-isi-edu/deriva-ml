"""Workspace-backed cache of the catalog schema.

Offline mode reads from this cache; online mode detects drift by
comparing the live catalog's snapshot id to the cached one and
warns the user without auto-refreshing.

File layout on disk at ``<workspace>/schema-cache.json``::

    {
        "snapshot_id": "<ERMrest snapshot id (snaptime)>",
        "hostname": "example.org",
        "catalog_id": "42",
        "ml_schema": "deriva-ml",
        "schema": { ... full ermrest /schema payload ... },
        "pin": {                              # optional; presence = pinned
            "at": "2026-04-22T20:30:00Z",
            "reason": "reproducing 2025 paper analysis"
        }
    }

Writes are atomic: the new contents go to ``schema-cache.json.tmp``,
get ``fsync``'d, then ``os.replace`` moves the tmp over the
original. If anything crashes mid-write, the old file remains
intact.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from deriva_ml.core.exceptions import DerivaMLConfigurationError


class PinStatus(BaseModel):
    """Current pin state of a :class:`SchemaCache`. Frozen Pydantic snapshot.

    Attributes:
        pinned: True iff the cache's JSON payload has a ``"pin"`` key.
        pinned_at: UTC timestamp of the most recent ``pin()`` call,
            or ``None`` when unpinned.
        pin_reason: Caller-supplied reason, or ``None`` when unpinned
            or when ``pin()`` was called without a reason.
        pinned_snapshot_id: The cache's current ``snapshot_id``
            (always present, whether pinned or not). A pinned cache
            is guaranteed to stay at this snapshot until ``unpin()``.
    """

    model_config = ConfigDict(frozen=True)

    pinned: bool
    pinned_at: datetime | None
    pin_reason: str | None
    pinned_snapshot_id: str


class SchemaCache:
    """Single-file schema cache at ``<workspace>/schema-cache.json``."""

    def __init__(self, workspace_root: Path):
        self._path = Path(workspace_root) / "schema-cache.json"

    def exists(self) -> bool:
        """True iff the cache file exists on disk."""
        return self._path.is_file()

    def snapshot_id(self) -> str | None:
        """Snapshot id stored in the cache, or None if no cache exists."""
        if not self.exists():
            return None
        try:
            return self.load()["snapshot_id"]
        except (KeyError, DerivaMLConfigurationError):
            return None

    def load(self) -> dict:
        """Read and parse the cache.

        Raises:
            FileNotFoundError: If the cache file doesn't exist.
            DerivaMLConfigurationError: If the file is unparseable
                JSON (e.g., partial write from before atomic writes
                were added, or manual corruption).
        """
        if not self.exists():
            raise FileNotFoundError(self._path)
        try:
            return json.loads(self._path.read_text())
        except json.JSONDecodeError as exc:
            raise DerivaMLConfigurationError(
                f"schema cache at {self._path} is corrupt "
                f"({exc.__class__.__name__}: {exc}); delete the file "
                f"and re-run online to regenerate."
            ) from exc

    def write(
        self,
        *,
        snapshot_id: str,
        hostname: str,
        catalog_id: str,
        ml_schema: str,
        schema: dict,
    ) -> None:
        """Atomically overwrite the cache.

        The new contents are written to a sibling ``.tmp`` file,
        ``fsync``'d, then moved over the original via ``os.replace``.
        If any step fails, the original file is unchanged.
        """
        payload = {
            "snapshot_id": snapshot_id,
            "hostname": hostname,
            "catalog_id": catalog_id,
            "ml_schema": ml_schema,
            "schema": schema,
        }
        self._write_atomic(payload)

    def _write_atomic(self, payload: dict) -> None:
        """Atomically write ``payload`` as JSON to the cache file.

        Writes to a sibling ``.tmp``, ``fsync``'s, then ``os.replace``'s
        over the target. On failure the original file is unchanged.
        Used by both ``write()`` (full cache refresh) and ``pin()``/
        ``unpin()`` (which rewrite an existing cache's payload with
        a tweaked ``"pin"`` key).
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w") as fp:
            json.dump(payload, fp, indent=2)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp, self._path)

    def pin(self, reason: str | None = None) -> None:
        """Mark the cache pinned at its current snapshot.

        Idempotent: pinning an already-pinned cache updates ``pinned_at``
        and ``reason`` to reflect the most recent call. The on-disk
        write goes through :meth:`_write_atomic` so a crash mid-pin
        leaves the prior cache state intact.

        Args:
            reason: Free-text explanation stored alongside the pin.
                Optional; defaults to ``None`` (stored as JSON null).

        Raises:
            FileNotFoundError: If the cache file doesn't exist. Call
                an online ``DerivaML.__init__`` or ``refresh_schema()``
                first to populate the cache.
        """
        payload = self.load()
        payload["pin"] = {
            "at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "reason": reason,
        }
        self._write_atomic(payload)

    def unpin(self) -> None:
        """Clear pin state. No-op if already unpinned.

        Atomic write when a pin existed; no I/O otherwise. Does not
        alter the cache's schema payload or snapshot_id.

        Raises:
            FileNotFoundError: If the cache file doesn't exist.
        """
        payload = self.load()
        if "pin" not in payload:
            return
        del payload["pin"]
        self._write_atomic(payload)

    def pin_status(self) -> PinStatus:
        """Return current pin state as a frozen :class:`PinStatus`.

        Raises:
            FileNotFoundError: If the cache file doesn't exist.
        """
        payload = self.load()
        pin = payload.get("pin")
        if pin is None:
            return PinStatus(
                pinned=False,
                pinned_at=None,
                pin_reason=None,
                pinned_snapshot_id=payload["snapshot_id"],
            )
        # Parse the ISO string Pydantic-style; handle the trailing "Z".
        raw_at = pin["at"]
        if raw_at.endswith("Z"):
            raw_at = raw_at[:-1] + "+00:00"
        return PinStatus(
            pinned=True,
            pinned_at=datetime.fromisoformat(raw_at),
            pin_reason=pin.get("reason"),
            pinned_snapshot_id=payload["snapshot_id"],
        )
