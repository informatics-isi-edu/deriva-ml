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
    from datetime import timedelta

    from deriva_ml.local_db.manifest_store import ManifestStore
    from deriva_ml.local_db.result_cache import CachedResult, CachedResultMeta, ResultCache
    from deriva_ml.local_db.schema import LocalSchema

logger = logging.getLogger(__name__)

WORKING_DB_SCHEMA_VERSION = 1


class Workspace:
    """Per-catalog working database, shared across script invocations.

    A ``Workspace`` is bound to a ``(working_dir, hostname, catalog_id)`` triple
    and provides:

    - A WAL-mode SQLAlchemy engine over ``{working_dir}/catalogs/{host}__{id}/working/main.db``.
    - ``build_local_schema()``: builds SQLAlchemy ORM classes for all catalog tables
      (via :class:`~deriva_ml.local_db.schema.LocalSchema`).
    - ``attach_slice()``: ATTACH a per-dataset snapshot (slice) database for
      query-time join-back.
    - ``manifest_store()``: SQLite-backed asset/feature manifest (replaces the old
      JSON ``asset-manifest.json`` file).
    - ``cached_table_read()`` / ``cache_denormalized()``: result-cache layer so
      expensive reads are stored in SQLite and returned instantly on the second call.

    The workspace engine is *unified* with the ``LocalSchema`` engine once
    ``build_local_schema()`` is called: manifest tables, cache tables, and ORM tables
    all share one SQLite file and connection pool.

    Typical usage::

        ws = Workspace(working_dir=Path("."), hostname="myhost.example.org", catalog_id="42")
        ws.build_local_schema(model=ermrest_model, schemas=["isa", "deriva-ml"])
        df = ws.cached_table_read("Subject").to_dataframe()
        ws.close()

    It also implements the context-manager protocol::

        with Workspace(...) as ws:
            ws.build_local_schema(...)
            ...
        # engine disposed automatically

    Args:
        working_dir: Root directory under which all catalog workspaces live.
        hostname: Deriva server hostname (used to partition workspaces by catalog).
        catalog_id: Catalog identifier (numeric or string).
    """

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
        self._result_cache: "ResultCache | None" = None
        self._closed = False

    # ---- paths ----

    @property
    def root(self) -> Path:
        """Root directory for this catalog's workspace (``{working_dir}/catalogs/{host}__{id}/``)."""
        return p.workspace_root(self._working_dir, self._hostname, self._catalog_id)

    @property
    def working_db_path(self) -> Path:
        """Path to the working DB directory (contains ``main.db`` + per-schema ``.db`` files)."""
        return p.working_db_path(self._working_dir, self._hostname, self._catalog_id)

    def slice_db_path(self, slice_id: str) -> Path:
        """Return the directory for the named slice.

        Args:
            slice_id: Opaque identifier for the slice (typically a dataset RID or version tag).

        Returns:
            Path to the slice directory (``{root}/slices/{slice_id}/``).
        """
        return p.slice_db_path(self._working_dir, self._hostname, self._catalog_id, slice_id)

    # ---- engine ----

    @property
    def engine(self) -> Engine:
        """Lazy-initialised WAL-mode SQLAlchemy engine over ``main.db``.

        The engine is created on first access. After ``build_local_schema()`` is
        called, this property returns the same engine as ``local_schema.engine``
        (they are unified to share one connection pool).

        Raises:
            RuntimeError: If the workspace has been closed via :meth:`close`.
        """
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
        """Dispose the current LocalSchema and build a fresh one.

        Equivalent to calling :meth:`build_local_schema` again.  Useful when
        the catalog model changes and the ORM needs to reflect the new schema.

        Args:
            model: Updated ERMrest Model.
            schemas: Schema names to materialize.

        Returns:
            The newly built :class:`~deriva_ml.local_db.schema.LocalSchema`.
        """
        return self.build_local_schema(model=model, schemas=schemas)

    def orm_class(self, table_name: str) -> Any | None:
        """Get the ORM class for a table by name.

        Convenience wrapper around ``local_schema.get_orm_class()``.

        Args:
            table_name: Bare table name (e.g., ``"Image"``).

        Returns:
            The mapped ORM class, or ``None`` if the schema hasn't been built
            or the table is not found.
        """
        if self._local_schema is None:
            return None
        try:
            return self._local_schema.get_orm_class(table_name)
        except (KeyError, Exception):
            return None

    # ---- slice attach/detach ----

    @contextlib.contextmanager
    def attach_slice(self, slice_id: str) -> Iterator[Connection]:
        """ATTACH a slice's per-schema DB files for the duration of the block.

        Multi-schema slices (directory with ``main.db`` + per-schema files):
            Each ``.db`` file is ATTACH'd under alias ``slice_{stem}`` (e.g.
            ``slice_isa``, ``slice_deriva-ml``).

        Legacy single-file slices (``slice.db``):
            ATTACH'd under alias ``"slice"`` for backward compatibility.

        Yields an open SQLAlchemy :class:`Connection` with the slice schemas
        visible as ``{alias}.{table}``.
        """
        s_dir = p.slice_dir(self._working_dir, self._hostname, self._catalog_id, slice_id)
        if not s_dir.is_dir():
            raise FileNotFoundError(f"Slice directory not found: {s_dir}")

        main_db = s_dir / "main.db"
        legacy_db = s_dir / "slice.db"

        if not main_db.is_file() and not legacy_db.is_file():
            raise FileNotFoundError(f"No database found in slice directory {s_dir} (expected main.db or slice.db)")

        conn = self.engine.connect()
        attached_aliases: list[str] = []
        try:
            if main_db.is_file():
                # Multi-schema layout: attach each .db file under slice_{stem}
                for db_file in sorted(s_dir.glob("*.db")):
                    alias = f"slice_{db_file.stem}"
                    sh.attach_database(conn, db_file, alias)
                    attached_aliases.append(alias)
            else:
                # Legacy single-file layout
                sh.attach_database(conn, legacy_db, "slice")
                attached_aliases.append("slice")

            yield conn

            for alias in attached_aliases:
                try:
                    sh.detach_database(conn, alias)
                except Exception:  # pragma: no cover — best-effort detach
                    logger.debug("detach %s failed; closing connection", alias)
        finally:
            conn.close()

    # ---- manifest store ----

    def manifest_store(self) -> "ManifestStore":
        """Return the :class:`~deriva_ml.local_db.manifest_store.ManifestStore` for this workspace.

        The store is created lazily and cached. It backs the asset/feature
        manifest that was previously stored as ``asset-manifest.json``.  All
        mutations are committed immediately so the store is crash-safe.

        Returns:
            The shared :class:`ManifestStore` instance.
        """
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

    # ---- result cache ----

    def _get_result_cache(self) -> "ResultCache":
        """Lazy-init the result cache backed by this workspace's engine."""
        if self._result_cache is None:
            from deriva_ml.local_db.result_cache import ResultCache

            self._result_cache = ResultCache(self.engine)
            self._result_cache.ensure_schema()
        return self._result_cache

    def cached_table_read(
        self,
        table: str,
        predicate: str | None = None,
        columns: list[str] | None = None,
        source: str = "catalog",
        refresh: bool = False,
        ttl: "timedelta | None" = None,
    ) -> "CachedResult":
        """Read a table from local schema and cache the result.

        On first call, reads all rows from the table in the local schema
        and stores them in the result cache. Subsequent calls return the
        cached data without re-reading. Use refresh=True to force re-read.

        Args:
            table: Table name to read.
            predicate: Unused for now (reserved for future filter pushdown).
            columns: Unused for now (reads all columns).
            source: Source tag for the cache entry.
            refresh: Force re-read even if cached.
            ttl: Optional time-to-live for the cache entry.
        """
        import time

        from sqlalchemy import select

        from deriva_ml.local_db.result_cache import CachedResultMeta, ResultCache

        if self._local_schema is None:
            raise RuntimeError("local_schema not built; call build_local_schema() first")

        rc = self._get_result_cache()
        key = ResultCache.cache_key("table_read", table=table)

        if not refresh and rc.has(key):
            result = rc.get(key)
            if result is not None:
                return result

        sql_table = self._local_schema.find_table(table)
        with self.engine.connect() as conn:
            result_rows = conn.execute(select(sql_table)).mappings().all()

        rows = [dict(r) for r in result_rows]
        col_names = [c.name for c in sql_table.columns]

        meta = CachedResultMeta(
            cache_key=key,
            source=source,
            tool_name="table_read",
            params={"table": table},
            columns=col_names,
            row_count=len(rows),
            created_at=time.time(),
            ttl_seconds=int(ttl.total_seconds()) if ttl else None,
        )
        rc.store(key, col_names, rows, meta)
        return rc.get(key)

    def cache_denormalized(
        self,
        model: Any,
        dataset_rid: str,
        include_tables: list[str],
        version: str | None = None,
        source: str = "local",
        slice_id: str | None = None,
        refresh: bool = False,
        dataset: Any = None,
        dataset_children_rids: list[str] | None = None,
        paged_client: Any = None,
        row_per: str | None = None,
        via: list[str] | None = None,
        ignore_unrelated_anchors: bool = False,
    ) -> "CachedResult":
        """Run denormalization and cache the result.

        On first call, runs the unified denormalizer and stores the result.
        Subsequent calls return cached data. Use refresh=True to re-compute.

        Args:
            model: ``DerivaModel`` used by the denormalizer for join planning.
            dataset_rid: RID of the dataset to denormalize.
            include_tables: Tables to include in the wide table.
            version: Optional dataset version (currently ignored; accepted for
                protocol compatibility with :meth:`Dataset.cache_denormalized`).
            source: Fetch mode forwarded to :func:`_denormalize_impl`.
                ``"local"`` (default) assumes rows are already present.
                ``"catalog"`` requires *paged_client* and fetches rows from
                the live catalog. ``"slice"`` assumes rows are visible via
                an attached slice database.
            slice_id: Reserved for future use with source='slice'.
            refresh: If True, ignore any existing cached entry and re-run.
            dataset: ``DatasetLike`` object for join-plan member enumeration.
            dataset_children_rids: Extra dataset RIDs for the WHERE filter.
            paged_client: Required when ``source='catalog'``. See
                :func:`_denormalize_impl`.
            row_per: Optional explicit leaf table (Rule 2). See
                :class:`~deriva_ml.local_db.denormalizer.Denormalizer`.
            via: Optional path-only intermediates to disambiguate FK paths
                (Rule 6) without adding their columns to the output.
            ignore_unrelated_anchors: Reserved — accepted for protocol
                compatibility but not yet propagated to the low-level
                ``_denormalize_impl`` primitive (which does not yet
                implement anchor classification).

        Returns:
            :class:`CachedResult` handle over the cached result table.
        """
        import time

        from deriva_ml.local_db.denormalize import _denormalize_impl as _denormalize
        from deriva_ml.local_db.result_cache import CachedResultMeta, ResultCache

        if self._local_schema is None:
            raise RuntimeError("local_schema not built; call build_local_schema() first")

        rc = self._get_result_cache()
        key = ResultCache.cache_key(
            "denormalize",
            dataset_rid=dataset_rid,
            tables=sorted(include_tables),
            version=version or "",
            source=source,
            # Include the new planner knobs so each (row_per, via,
            # ignore_unrelated_anchors) combination caches independently.
            row_per=row_per or "",
            via=sorted(via) if via else [],
            ignore_unrelated_anchors=ignore_unrelated_anchors,
        )

        if not refresh and rc.has(key):
            result = rc.get(key)
            if result is not None:
                return result

        orm_resolver = self._local_schema.get_orm_class
        denorm_result = _denormalize(
            model=model,
            engine=self.engine,
            orm_resolver=orm_resolver,
            dataset_rid=dataset_rid,
            include_tables=include_tables,
            dataset=dataset,
            dataset_children_rids=dataset_children_rids,
            source=source,
            paged_client=paged_client,
            row_per=row_per,
            via=via,
        )

        rows = list(denorm_result.iter_rows())
        col_names = [name for name, _ in denorm_result.columns]
        meta = CachedResultMeta(
            cache_key=key,
            source=source,
            tool_name="denormalize",
            params={
                "dataset_rid": dataset_rid,
                "tables": sorted(include_tables),
                "version": version,
                "row_per": row_per,
                "via": sorted(via) if via else None,
                "ignore_unrelated_anchors": ignore_unrelated_anchors,
            },
            columns=col_names,
            row_count=len(rows),
            created_at=time.time(),
        )
        rc.store(key, col_names, rows, meta)
        return rc.get(key)

    def list_cached_results(self) -> "list[CachedResultMeta]":
        """List all non-expired cached result entries.

        Lazily removes expired entries from the registry as a side effect.

        Returns:
            List of :class:`~deriva_ml.local_db.result_cache.CachedResultMeta` for
            every live (non-expired) cache entry.
        """
        return self._get_result_cache().list_cached()

    def get_cached_result(self, cache_key: str) -> "CachedResult | None":
        """Get a cached result handle by key, or None if missing/expired.

        Args:
            cache_key: The ``rc_`` prefixed key returned by :meth:`cached_table_read`
                or :meth:`cache_denormalized`.

        Returns:
            A :class:`~deriva_ml.local_db.result_cache.CachedResult` handle, or
            ``None`` if the entry doesn't exist or has expired.
        """
        return self._get_result_cache().get(cache_key)

    def invalidate_cache(self, cache_key: str | None = None, source: str | None = None) -> int:
        """Invalidate cache entries by key, source, or all.

        Args:
            cache_key: Remove only this specific entry (by exact key).
            source: Remove all entries with this source tag (e.g., ``"catalog"``).
            (Both None): Remove all entries.

        Returns:
            Count of entries removed.
        """
        return self._get_result_cache().invalidate(cache_key=cache_key, source=source)

    # ---- lifecycle ----

    def close(self) -> None:
        """Dispose the engine and release all resources.

        Safe to call multiple times (idempotent). After this call, accessing
        :attr:`engine` will raise :class:`RuntimeError`.
        """
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
