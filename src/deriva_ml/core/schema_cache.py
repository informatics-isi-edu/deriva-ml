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
        "schema": { ... full ermrest /schema payload ... }
    }

Writes are atomic: the new contents go to ``schema-cache.json.tmp``,
get ``fsync``'d, then ``os.replace`` moves the tmp over the
original. If anything crashes mid-write, the old file remains
intact.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from deriva_ml.core.exceptions import DerivaMLConfigurationError


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
