"""SQLite engine and connection helpers for the local_db layer.

Provides:
- ``create_wal_engine``: SQLAlchemy engine factory enforcing WAL + synchronous=NORMAL.
- ``attach_database`` / ``detach_database``: ATTACH / DETACH helpers.
- ``ensure_schema_meta``: idempotent schema-version tracking in a ``schema_meta``
  table. Raises :class:`SchemaVersionError` when the on-disk schema is newer than
  the running code expects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Connection, Engine

logger = logging.getLogger(__name__)

SCHEMA_META_TABLE = "schema_meta"


class SchemaVersionError(RuntimeError):
    """The on-disk schema version is newer than this code supports."""


def create_wal_engine(db_path: Path, *, read_only: bool = False) -> Engine:
    """Create a SQLAlchemy engine for a SQLite file with WAL mode.

    - Ensures the parent directory exists.
    - Sets ``journal_mode=WAL`` and ``synchronous=NORMAL`` on every connection.
    - When ``read_only=True``, opens the file via SQLite's ``mode=ro`` URI.

    Args:
        db_path: Path to the SQLite file.
        read_only: If True, open in read-only mode.

    Returns:
        A SQLAlchemy :class:`Engine`.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if read_only:
        url = f"sqlite:///file:{db_path.resolve()}?mode=ro&uri=true"
        # sqlite3 driver needs uri=True on connect kwargs
        engine = create_engine(
            url,
            future=True,
            connect_args={"uri": True},
        )
    else:
        engine = create_engine(f"sqlite:///{db_path.resolve()}", future=True)

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn: Any, _record: Any) -> None:
        cur = dbapi_conn.cursor()
        try:
            if not read_only:
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
        finally:
            cur.close()

    return engine


def attach_database(conn: Connection, db_path: Path, alias: str) -> None:
    """ATTACH a SQLite file under ``alias`` in the given connection.

    Args:
        conn: An open SQLAlchemy connection.
        db_path: Path to the SQLite file to attach.
        alias: Name to attach it under (used as a schema qualifier).
    """
    path_str = str(Path(db_path).resolve()).replace("'", "''")
    alias_safe = alias.replace('"', '""')
    conn.execute(text(f"ATTACH DATABASE '{path_str}' AS \"{alias_safe}\""))


def detach_database(conn: Connection, alias: str) -> None:
    """DETACH a previously attached database by alias."""
    alias_safe = alias.replace('"', '""')
    conn.execute(text(f'DETACH DATABASE "{alias_safe}"'))


def ensure_schema_meta(engine: Engine, expected_version: int) -> int:
    """Ensure the ``schema_meta`` table exists and records ``expected_version``.

    If the table is empty, inserts ``expected_version`` as the initial version.
    If the table already has a version ≤ ``expected_version``, returns it.
    If the on-disk version is higher, raises :class:`SchemaVersionError`.

    Returns:
        The current schema version (after any initialization).
    """
    with engine.connect() as conn:
        conn.execute(
            text(
                f"CREATE TABLE IF NOT EXISTS {SCHEMA_META_TABLE} ("
                "  version INTEGER PRIMARY KEY,"
                "  recorded_at TEXT NOT NULL DEFAULT (datetime('now'))"
                ")"
            )
        )
        existing = conn.execute(text(f"SELECT MAX(version) FROM {SCHEMA_META_TABLE}")).scalar()
        if existing is None:
            conn.execute(
                text(f"INSERT INTO {SCHEMA_META_TABLE}(version) VALUES (:v)"),
                {"v": expected_version},
            )
            conn.commit()
            return expected_version
        if existing > expected_version:
            raise SchemaVersionError(
                f"Database schema version {existing} is newer than expected {expected_version}; upgrade deriva-ml."
            )
        return int(existing)
