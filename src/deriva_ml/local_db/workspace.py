"""Per-catalog working database and slice registry.

A ``Workspace`` is bound to ``(working_dir, hostname, catalog_id)``. It owns:
- A single SQLAlchemy engine over ``working.db`` in WAL mode.
- An ``attach_slice`` context manager that ATTACHes a slice DB under alias
  ``"slice"`` for the duration of a single connection.
- A ``legacy_working_data_view`` compatibility shim exposing the old
  ``WorkingDataCache`` API (pandas table roundtrips) on top of the new file.

Phase 1 intentionally does not build a ``LocalSchema`` over the working DB
eagerly; the paged fetcher and denormalizer drive schema creation lazily.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from deriva_ml.local_db import paths as p
from deriva_ml.local_db import sqlite_helpers as sh

if TYPE_CHECKING:
    import pandas as pd

    from deriva_ml.local_db.manifest_store import ManifestStore

logger = logging.getLogger(__name__)

WORKING_DB_SCHEMA_VERSION = 1

_RESERVED_TABLES = frozenset(
    {
        "schema_meta",
        "execution_state__assets",
        "execution_state__features",
    }
)


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

    # ---- legacy compat ----

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

    def legacy_working_data_view(self) -> "_LegacyWorkingDataView":
        """Return an adapter exposing the old ``WorkingDataCache`` API.

        Lets existing callers (``Dataset.cache_denormalized``, tests) continue
        to use the string-keyed DataFrame table semantics they rely on, backed
        by the new per-catalog working DB.
        """
        return _LegacyWorkingDataView(self)

    # ---- lifecycle ----

    def close(self) -> None:
        if self._closed:
            return
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
        self._closed = True

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self.close()
        return False


class _LegacyWorkingDataView:
    """Adapter providing the old ``WorkingDataCache`` surface on a Workspace.

    Keeps the working DataFrames in the same SQLite file as everything else
    (under their bare user-chosen table names) so existing callers keep
    working without code changes.
    """

    def __init__(self, ws: Workspace) -> None:
        self._ws = ws

    @property
    def workspace(self) -> Workspace:
        """The underlying Workspace."""
        return self._ws

    def cache_table(self, table_name: str, df: "pd.DataFrame") -> Path:
        import pandas as pd  # lazy import

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
        df.to_sql(table_name, self._ws.engine, if_exists="replace", index=False)
        return self._ws.working_db_path

    def read_table(self, table_name: str) -> "pd.DataFrame":
        import pandas as pd

        if not self.has_table(table_name):
            raise ValueError(f"Table '{table_name}' not found in workspace working DB; available: {self.list_tables()}")
        return pd.read_sql_table(table_name, self._ws.engine)

    def query(self, sql: str) -> "pd.DataFrame":
        import pandas as pd

        return pd.read_sql_query(text(sql), self._ws.engine)

    def has_table(self, table_name: str) -> bool:
        from sqlalchemy import inspect

        if not self._ws.working_db_path.exists():
            return False
        return table_name in inspect(self._ws.engine).get_table_names()

    def list_tables(self) -> list[str]:
        from sqlalchemy import inspect

        if not self._ws.working_db_path.exists():
            return []
        return inspect(self._ws.engine).get_table_names()

    def drop_table(self, table_name: str) -> None:
        if table_name in _RESERVED_TABLES:
            return  # Silently refuse to drop reserved infrastructure tables
        if self.has_table(table_name):
            with self._ws.engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS [{table_name}]"))
                conn.commit()

    def clear(self) -> None:
        for t in self.list_tables():
            if t not in _RESERVED_TABLES:
                self.drop_table(t)
