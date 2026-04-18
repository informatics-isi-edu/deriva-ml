"""Unified schema adapter for working and slice databases.

Wraps :class:`deriva_ml.model.schema_builder.SchemaBuilder` in a smaller
interface usable from either a live ERMrest ``Model`` or a bag's
``schema.json`` file. Construction is a two-step process:

    ls = LocalSchema.build(model=m, schemas=[...], database_path=p)

which creates (or opens) the SQLite file, builds tables, and returns a handle
with a WAL-mode SQLAlchemy engine.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from deriva.core.ermrest_model import Model
from sqlalchemy import Table as SQLTable
from sqlalchemy import event
from sqlalchemy.engine import Engine

from deriva_ml.local_db import sqlite_helpers as sh
from deriva_ml.model.schema_builder import SchemaBuilder, SchemaORM

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class LocalSchema:
    """Thin adapter over ``SchemaBuilder`` / ``SchemaORM`` for the local_db layer.

    Normalizes construction and pragma setup so callers don't have to know
    whether the underlying DB is a working DB or a slice DB. All real work
    is delegated to the existing ``SchemaBuilder`` machinery.

    Typical usage::

        ls = LocalSchema.build(
            model=ermrest_model,
            schemas=["isa", "deriva-ml"],
            database_path=Path("/tmp/working"),
        )
        image_cls = ls.get_orm_class("Image")
        with ls.engine.begin() as conn:
            conn.execute(insert(image_cls.__table__).values(RID="IMG-1", ...))
        ls.dispose()

    Attributes:
        engine: The SQLAlchemy :class:`Engine` over the SQLite file(s).
        metadata: The SQLAlchemy :class:`MetaData` reflecting all schema tables.
        schemas: List of schema names materialized in this database.
        database_path: Directory or ``.db`` file path where data is stored.
    """

    def __init__(self, orm: SchemaORM, database_path: Path) -> None:
        self._orm = orm
        self._database_path = database_path

    @staticmethod
    def _main_db_path(database_path: Path) -> Path:
        """Compute the main SQLite file path SchemaBuilder will use.

        Matches the logic in SchemaBuilder.build(): if the path ends in ``.db``
        it's treated as the main file directly; otherwise a ``main.db`` is
        placed inside the directory.
        """
        if database_path.suffix == ".db":
            return database_path
        return database_path / "main.db"

    @classmethod
    def build(
        cls,
        *,
        model: Model,
        schemas: list[str],
        database_path: Path,
        read_only: bool = False,
    ) -> "LocalSchema":
        """Build (or open) the schema at ``database_path``.

        Args:
            model: ERMrest Model (from live catalog or schema.json).
            schemas: Schema names to materialize.
            database_path: Directory or `.db` file path for storage.
            read_only: If True, open the underlying engine read-only. This is
                used for slice DBs at query time. Multi-schema read-only is
                not yet supported and will raise NotImplementedError — Phase 2
                will revisit slice DBs with proper multi-schema support.
        """
        database_path = Path(database_path)

        if read_only and len(schemas) > 1:
            raise NotImplementedError(
                "Multi-schema read-only LocalSchema is not yet supported; "
                "Phase 2 will address slice DBs with proper multi-schema read-only."
            )

        # Pre-configure WAL mode on the database files using raw sqlite3 before
        # SchemaBuilder creates the engine. This ensures WAL is set BEFORE any
        # SQLAlchemy connection is made. For multi-schema file-based DBs, we
        # need to set WAL on both the main database and each schema file.
        # Skip when read_only=True — files already exist and we don't want to
        # write to them.
        if not read_only and database_path != ":memory:":
            main_db = cls._main_db_path(database_path)

            # Set WAL on the main file
            try:
                main_db.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(main_db))
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.close()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to pre-configure WAL on {main_db}: {e}")

            # Set WAL on each schema file
            for schema in schemas:
                try:
                    schema_db = (database_path if database_path.is_dir() else database_path.parent) / f"{schema}.db"
                    conn = sqlite3.connect(str(schema_db))
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.close()
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to pre-configure WAL on {schema_db}: {e}")

        # SchemaBuilder only builds writable engines — we always start writable,
        # then swap in a read-only engine below if requested.
        builder = SchemaBuilder(model=model, schemas=schemas, database_path=database_path)
        orm = builder.build()

        # Also register a listener to ensure foreign keys are always enabled on new connections
        @event.listens_for(orm.engine.pool, "connect")
        def _enforce_fk(dbapi_conn: Any, _r: Any) -> None:
            cur = dbapi_conn.cursor()
            try:
                cur.execute("PRAGMA foreign_keys=ON")
                for schema in schemas:
                    try:
                        cur.execute(f'PRAGMA "{schema}".foreign_keys=ON')
                    except Exception:  # noqa: BLE001
                        pass
            finally:
                cur.close()

        # Track schema version in the DB (while still writable).
        sh.ensure_schema_meta(orm.engine, expected_version=SCHEMA_VERSION)

        # If read-only was requested, swap the writable engine for a read-only
        # one over the same file. The metadata/Base on the ORM remain valid —
        # only the engine changes. For single-schema, also ATTACH the schema
        # file in read-only mode so the ORM tables (stored under the schema
        # namespace) remain visible.
        if read_only:
            main_db = cls._main_db_path(database_path)
            schema_db_dir = database_path if database_path.is_dir() else database_path.parent
            orm.engine.dispose()
            ro_engine = sh.create_wal_engine(main_db, read_only=True)

            # Attach per-schema DB files in read-only mode so table names like
            # ``"<schema>"."<table>"`` resolve. Uses SQLite URI syntax with
            # mode=ro (connection was opened with uri=True by create_wal_engine).
            @event.listens_for(ro_engine, "connect")
            def _attach_ro_schemas(dbapi_conn: Any, _r: Any) -> None:
                cur = dbapi_conn.cursor()
                try:
                    for schema in schemas:
                        schema_file = (schema_db_dir / f"{schema}.db").resolve()
                        uri = f"file:{schema_file}?mode=ro"
                        # ATTACH with URI form; SQLite accepts this when the
                        # connection was opened with uri=True.
                        cur.execute(f"ATTACH DATABASE '{uri}' AS \"{schema}\"")
                finally:
                    cur.close()

            orm.engine = ro_engine

        return cls(orm=orm, database_path=database_path)

    # ---- delegation ----

    @property
    def engine(self) -> Engine:
        """The SQLAlchemy engine for this schema's database file(s)."""
        return self._orm.engine

    @property
    def metadata(self):  # noqa: ANN201 — SQLAlchemy MetaData
        """The SQLAlchemy MetaData reflecting all tables in this schema."""
        return self._orm.metadata

    @property
    def schemas(self) -> list[str]:
        """List of ERMrest schema names materialized in this database."""
        return list(self._orm.schemas)

    @property
    def database_path(self) -> Path:
        """Directory or ``.db`` file path for storage."""
        return self._database_path

    def find_table(self, table_name: str) -> SQLTable:
        """Return the SQLAlchemy :class:`Table` object for a given table name.

        Args:
            table_name: Bare table name or ``schema.TableName`` qualified form.

        Returns:
            The SQLAlchemy Table reflecting the given table.

        Raises:
            KeyError: If the table is not found in the schema.
        """
        return self._orm.find_table(table_name)

    def list_tables(self) -> list[str]:
        """Return a list of all table names known to this schema."""
        return self._orm.list_tables()

    def get_orm_class(self, table_name: str) -> Any | None:
        """Return the SQLAlchemy ORM mapped class for *table_name*, or None.

        The ORM class has attributes for each column (e.g., ``Image.RID``,
        ``Image.Filename``) and a ``__table__`` attribute pointing to the
        underlying :class:`Table`.

        Args:
            table_name: Bare table name (e.g., ``"Image"``).

        Returns:
            The mapped ORM class, or ``None`` if the table is not found.
        """
        return self._orm.get_orm_class(table_name)

    def dispose(self) -> None:
        """Dispose the underlying engine, releasing all connections and file handles."""
        self._orm.dispose()

    def __enter__(self) -> "LocalSchema":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self.dispose()
        return False
