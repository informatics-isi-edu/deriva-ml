"""Per-catalog working database and slice registry.

A ``Workspace`` is bound to ``(working_dir, hostname, catalog_id)``. It owns:
- A single SQLAlchemy engine over ``working/main.db`` in WAL mode.
- An ``attach_slice`` context manager that ATTACHes a slice DB under alias
  ``"slice"`` for the duration of a single connection.
- ``build_local_schema()`` / ``rebuild_schema()`` for building an ORM over the
  working DB with cross-schema relationships.

The workspace engine is unified with the LocalSchema engine once
``build_local_schema()`` is called, so ManifestStore, result cache, and ORM
tables all share one connection pool.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from sqlalchemy.engine import Connection, Engine

from deriva_ml.local_db import paths as p
from deriva_ml.local_db import sqlite_helpers as sh

if TYPE_CHECKING:
    from deriva_ml.local_db.manifest_store import ManifestStore
    from deriva_ml.local_db.schema import LocalSchema

logger = logging.getLogger(__name__)

WORKING_DB_SCHEMA_VERSION = 1


class Workspace:
    """Per-catalog working database, shared across script invocations."""

    def __init__(
        self,
        *,
        working_dir: Path,
        hostname: str,
        catalog_id: str | int,
    ) -> None:
        self._working_dir = Path(working_dir)
        self._hostname = hostname
        self._catalog_id = str(catalog_id)
        self._engine: Engine | None = None
        self._manifest_store: "ManifestStore | None" = None
        self._local_schema: "LocalSchema | None" = None
        self._closed = False

    # ---- paths ----

    @property
    def root(self) -> Path:
        return p.workspace_root(self._working_dir, self._hostname, self._catalog_id)

    @property
    def working_db_path(self) -> Path:
        """Path to the working DB directory (contains main.db + per-schema files)."""
        return p.working_db_path(self._working_dir, self._hostname, self._catalog_id)

    def slice_db_path(self, slice_id: str) -> Path:
        return p.slice_db_path(self._working_dir, self._hostname, self._catalog_id, slice_id)

    # ---- engine ----

    @property
    def engine(self) -> Engine:
        if self._closed:
            raise RuntimeError("Workspace is closed")
        if self._engine is None:
            main_db = p.working_main_db_path(self._working_dir, self._hostname, self._catalog_id)
            self._engine = sh.create_wal_engine(main_db)
            sh.ensure_schema_meta(self._engine, expected_version=WORKING_DB_SCHEMA_VERSION)
        return self._engine

    # ---- local schema ----

    @property
    def local_schema(self) -> "LocalSchema | None":
        """The LocalSchema for this workspace, or None if not yet built.

        The schema is built lazily via ``build_local_schema(model, schemas)``.
        Once built, it provides ORM classes with cross-schema relationships
        for all tables in the working DB.
        """
        return self._local_schema

    def build_local_schema(
        self,
        *,
        model: Any,
        schemas: list[str],
    ) -> "LocalSchema":
        """Build the ORM schema for this workspace's working DB.

        Creates per-schema .db files in the working directory and sets up
        SQLAlchemy ORM with cross-schema relationships via ATTACH. After
        this call, ``local_schema`` is non-None and ``orm_class()`` works.

        The workspace engine is unified with the LocalSchema engine so that
        ManifestStore, result cache, and ORM tables all share one connection
        pool.

        Args:
            model: ERMrest Model (from live catalog or schema.json).
            schemas: Schema names to include (e.g. ["isa", "deriva-ml"]).
        """
        from deriva_ml.local_db.schema import LocalSchema

        if self._local_schema is not None:
            self._local_schema.dispose()

        self._local_schema = LocalSchema.build(
            model=model,
            schemas=schemas,
            database_path=self.working_db_path,  # directory path
        )

        # Unify engines: workspace engine = LocalSchema engine.
        # ManifestStore tables live in main.db which is the same file
        # LocalSchema's engine opens.
        if self._engine is not None and self._engine is not self._local_schema.engine:
            self._engine.dispose()
        self._engine = self._local_schema.engine

        # Re-initialize manifest store if it was previously created,
        # since it holds a reference to the old engine.
        if self._manifest_store is not None:
            from deriva_ml.local_db.manifest_store import ManifestStore

            self._manifest_store = ManifestStore(self._engine)
            self._manifest_store.ensure_schema()

        return self._local_schema

    def rebuild_schema(self, *, model: Any, schemas: list[str]) -> "LocalSchema":
        """Dispose the current LocalSchema and build a fresh one."""
        return self.build_local_schema(model=model, schemas=schemas)

    def orm_class(self, table_name: str) -> Any | None:
        """Get the ORM class for a table by name.

        Returns None if local_schema hasn't been built or table not found.
        """
        if self._local_schema is None:
            return None
        try:
            return self._local_schema.get_orm_class(table_name)
        except (KeyError, Exception):
            return None

    # ---- slice attach/detach ----

    @contextlib.contextmanager
    def attach_slice(self, slice_id: str, alias: str = "slice") -> Iterator[Connection]:
        """ATTACH a slice DB under ``alias`` for the duration of the block.

        Yields an open SQLAlchemy :class:`Connection` with the slice visible
        as ``{alias}.{table}``.
        """
        slice_path = self.slice_db_path(slice_id)
        if not slice_path.is_file():
            raise FileNotFoundError(f"Slice database not found: {slice_path}")

        conn = self.engine.connect()
        try:
            sh.attach_database(conn, slice_path, alias)
            yield conn
            try:
                sh.detach_database(conn, alias)
            except Exception:  # pragma: no cover — best-effort detach
                logger.debug("detach_database failed; closing connection")
        finally:
            conn.close()

    # ---- manifest store ----

    def manifest_store(self) -> "ManifestStore":
        """Return a ManifestStore backed by this workspace's engine (cached)."""
        if self._manifest_store is None:
            from deriva_ml.local_db.manifest_store import ManifestStore

            s = ManifestStore(self.engine)
            s.ensure_schema()
            self._manifest_store = s
        return self._manifest_store

    def import_legacy_manifests(self) -> int:
        """Import any pre-existing ``asset-manifest.json`` files into the DB.

        Scans ``{working_dir}`` recursively for files named exactly
        ``asset-manifest.json``. For each, parses the JSON, upserts rows into
        the manifest store, and renames the file to
        ``{path}.migrated.json``. Idempotent: already-migrated files are
        skipped.

        Returns:
            The number of manifests newly migrated.
        """
        import json as _json

        from deriva_ml.asset.manifest import AssetEntry, FeatureEntry

        store = self.manifest_store()
        migrated = 0
        for manifest_path in self._working_dir.rglob("asset-manifest.json"):
            try:
                data = _json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to read %s: %s", manifest_path, exc)
                continue

            execution_rid = data.get("execution_rid") or manifest_path.parent.name
            for key, entry_dict in data.get("assets", {}).items():
                entry = AssetEntry.from_dict(entry_dict)
                store.add_asset(execution_rid, key, entry)
            for name, entry_dict in data.get("features", {}).items():
                entry = FeatureEntry.from_dict(entry_dict)
                store.add_feature(execution_rid, name, entry)

            sidecar = manifest_path.with_suffix(manifest_path.suffix + ".migrated.json")
            manifest_path.rename(sidecar)
            migrated += 1
        return migrated

    # ---- lifecycle ----

    def close(self) -> None:
        if self._closed:
            return
        if self._local_schema is not None:
            self._local_schema.dispose()
            self._local_schema = None
            # Engine was unified with LocalSchema; already disposed above.
            self._engine = None
        elif self._engine is not None:
            self._engine.dispose()
            self._engine = None
        self._closed = True

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self.close()
        return False
