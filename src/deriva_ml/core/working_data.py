"""Working data cache for catalog query results.

Provides a SQLite-backed cache in the working directory for storing
catalog query results that persist across script invocations. This is
separate from the dataset bag cache (which stores version-pinned snapshots).

Use cases:
- Caching full table contents fetched from the live catalog
- Caching denormalized (wide table) views computed from datasets
- Caching feature values fetched from the catalog
- Persisting intermediate results across Claude-generated scripts

Location: {working_dir}/working-data/catalog.db
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, inspect, text

logger = logging.getLogger(__name__)


class WorkingDataCache:
    """SQLite cache for catalog query results in the working directory.

    Stores DataFrames as named tables in a single SQLite database.
    Tables can be added incrementally and queried without loading
    everything into memory.

    Example::

        cache = WorkingDataCache(working_dir)
        cache.cache_table("Subject", df)
        df = cache.read_table("Subject")
        cache.has_table("Subject")  # True
        cache.list_tables()  # ["Subject"]
        cache.clear()
    """

    def __init__(self, working_dir: Path):
        self._db_dir = working_dir / "working-data"
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._db_dir / "catalog.db"
        self._engine = None

    @property
    def db_path(self) -> Path:
        """Path to the SQLite database file."""
        return self._db_path

    @property
    def engine(self):
        """Lazy SQLAlchemy engine for the cache database."""
        if self._engine is None:
            self._engine = create_engine(f"sqlite:///{self._db_path}")
        return self._engine

    def cache_table(self, table_name: str, df: pd.DataFrame) -> Path:
        """Write a DataFrame to the cache as a named table.

        Replaces the table if it already exists.

        Args:
            table_name: Name for the table in the cache database.
            df: DataFrame to store.

        Returns:
            Path to the database file.
        """
        df.to_sql(table_name, self.engine, if_exists="replace", index=False)
        logger.info(f"Cached {len(df)} rows as '{table_name}' in {self._db_path}")
        return self._db_path

    def read_table(self, table_name: str) -> pd.DataFrame:
        """Read a cached table back as a DataFrame.

        Args:
            table_name: Name of the table to read.

        Returns:
            DataFrame with the cached data.

        Raises:
            ValueError: If the table does not exist in the cache.
        """
        if not self.has_table(table_name):
            raise ValueError(
                f"Table '{table_name}' not found in cache. "
                f"Available: {self.list_tables()}"
            )
        return pd.read_sql_table(table_name, self.engine)

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query against the cache database.

        Args:
            sql: SQL query string (e.g., "SELECT * FROM Subject WHERE Age > 30").

        Returns:
            DataFrame with query results.
        """
        return pd.read_sql_query(text(sql), self.engine)

    def has_table(self, table_name: str) -> bool:
        """Check if a table exists in the cache.

        Args:
            table_name: Name of the table to check.

        Returns:
            True if the table exists.
        """
        if not self._db_path.exists():
            return False
        return table_name in inspect(self.engine).get_table_names()

    def list_tables(self) -> list[str]:
        """List all tables in the cache.

        Returns:
            List of table names.
        """
        if not self._db_path.exists():
            return []
        return inspect(self.engine).get_table_names()

    def table_info(self, table_name: str) -> dict[str, Any]:
        """Get metadata about a cached table.

        Args:
            table_name: Name of the table.

        Returns:
            Dict with column names, row count, and column types.
        """
        if not self.has_table(table_name):
            raise ValueError(f"Table '{table_name}' not found in cache.")

        columns = inspect(self.engine).get_columns(table_name)
        with self.engine.connect() as conn:
            row_count = conn.execute(
                text(f"SELECT COUNT(*) FROM [{table_name}]")
            ).scalar()

        return {
            "table_name": table_name,
            "row_count": row_count,
            "columns": [
                {"name": c["name"], "type": str(c["type"])}
                for c in columns
            ],
        }

    def status(self) -> dict[str, Any]:
        """Get overall cache status.

        Returns:
            Dict with database path, file size, table list, and per-table info.
        """
        tables = self.list_tables()
        db_size = self._db_path.stat().st_size if self._db_path.exists() else 0

        return {
            "db_path": str(self._db_path),
            "db_size_bytes": db_size,
            "db_size_human": _human_size(db_size),
            "table_count": len(tables),
            "tables": {t: self.table_info(t) for t in tables},
        }

    def drop_table(self, table_name: str) -> None:
        """Remove a single table from the cache.

        Args:
            table_name: Name of the table to remove.
        """
        if self.has_table(table_name):
            with self.engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS [{table_name}]"))
                conn.commit()
            logger.info(f"Dropped '{table_name}' from cache")

    def clear(self) -> None:
        """Delete the entire cache database."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
        if self._db_path.exists():
            self._db_path.unlink()
            logger.info(f"Cleared working data cache: {self._db_path}")


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"
