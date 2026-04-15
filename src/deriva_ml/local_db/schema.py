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
    """

    def __init__(self, orm: SchemaORM, database_path: Path) -> None:
        self._orm = orm
        self._database_path = database_path

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
                used for slice DBs at query time.
        """
        database_path = Path(database_path)

        # Pre-configure WAL mode on the database files using raw sqlite3 before
        # SchemaBuilder creates the engine. This ensures WAL is set BEFORE any
        # SQLAlchemy connection is made. For multi-schema file-based DBs, we
        # need to set WAL on both the main database and each schema file.
        if not read_only and database_path != ":memory:":
            # Determine which files will be created by SchemaBuilder
            if database_path.suffix == ".db":
                main_db = database_path
            else:
                main_db = database_path / "main.db"

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

        # Track schema version in the DB.
        sh.ensure_schema_meta(orm.engine, expected_version=SCHEMA_VERSION)

        return cls(orm=orm, database_path=database_path)

    # ---- delegation ----

    @property
    def engine(self) -> Engine:
        return self._orm.engine

    @property
    def metadata(self):  # noqa: ANN201 — SQLAlchemy MetaData
        return self._orm.metadata

    @property
    def schemas(self) -> list[str]:
        return list(self._orm.schemas)

    @property
    def database_path(self) -> Path:
        return self._database_path

    def find_table(self, table_name: str) -> SQLTable:
        return self._orm.find_table(table_name)

    def list_tables(self) -> list[str]:
        return self._orm.list_tables()

    def get_orm_class(self, table_name: str) -> Any | None:
        return self._orm.get_orm_class(table_name)

    def dispose(self) -> None:
        self._orm.dispose()

    def __enter__(self) -> "LocalSchema":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self.dispose()
        return False
