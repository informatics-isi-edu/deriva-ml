"""Load data into SQLite database with FK ordering.

This module provides the DataLoader class which loads data from a
DataSource into a SchemaORM database. It handles:

- Automatic FK dependency ordering
- Batch inserts with conflict handling
- Progress tracking

This is Phase 2 of the two-phase pattern:
1. Phase 1 (SchemaBuilder): Create ORM structure without data
2. Phase 2 (DataLoader): Fill database from a data source

Example:
    # Phase 1: Create ORM
    orm = SchemaBuilder(model, schemas).build()

    # Phase 2: Fill with data
    source = BagDataSource(bag_path)
    loader = DataLoader(orm, source)
    counts = loader.load_tables(['Subject', 'Image', 'Diagnosis'])
    print(f"Loaded {sum(counts.values())} rows")
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from deriva.core.ermrest_model import Table as DerivaTable

from .schema_builder import SchemaORM
from .data_sources import DataSource
from .fk_orderer import ForeignKeyOrderer


logger = logging.getLogger(__name__)


class DataLoader:
    """Loads data into a database with FK ordering.

    Phase 2 of the two-phase database creation pattern. Takes a
    SchemaORM (from Phase 1) and populates it from a DataSource.

    Automatically orders tables by FK dependencies to ensure
    referential integrity during loading.

    Example:
        # Phase 1: Create ORM
        orm = SchemaBuilder(model, schemas).build()

        # Phase 2: Fill with data from bag
        source = BagDataSource(bag_path)
        loader = DataLoader(orm, source)
        counts = loader.load_tables()  # All tables
        print(f"Loaded {sum(counts.values())} total rows")

        # Or load specific tables
        counts = loader.load_tables(['Subject', 'Image'])

        # With progress callback
        def on_progress(table, count, total):
            print(f"Loaded {table}: {count} rows")
        loader.load_tables(progress_callback=on_progress)
    """

    def __init__(
        self,
        schema_orm: SchemaORM,
        data_source: DataSource,
    ):
        """Initialize the loader.

        Args:
            schema_orm: ORM structure from SchemaBuilder.
            data_source: Source of data to load (BagDataSource, CatalogDataSource, etc.).
        """
        self.orm = schema_orm
        self.source = data_source
        self.orderer = ForeignKeyOrderer(
            schema_orm.model,
            schema_orm.schemas,
        )

    def load_tables(
        self,
        tables: list[str | DerivaTable] | None = None,
        on_conflict: str = "ignore",
        batch_size: int = 1000,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, int]:
        """Load data into specified tables with FK ordering.

        Tables are automatically ordered by FK dependencies to ensure
        referenced tables are populated first.

        Args:
            tables: Tables to load. If None, loads all tables that have
                data in the source.
            on_conflict: How to handle duplicate keys:
                - "ignore": Skip rows with duplicate keys (default)
                - "replace": Replace existing rows
                - "error": Raise error on duplicates
            batch_size: Number of rows per insert batch.
            progress_callback: Optional callback(table_name, rows_loaded, total_tables)
                called after each table is loaded.

        Returns:
            Dict mapping table names to row counts loaded.
        """
        # Determine tables to load
        if tables is None:
            # Get all tables that have data in source
            available = set(self.source.list_available_tables())
            # Filter to tables that exist in ORM
            orm_tables = set(self.orm.list_tables())

            # Match available tables to ORM tables
            tables_to_load = []
            for orm_table in orm_tables:
                # Check both qualified and unqualified names
                table_name = orm_table.split(".")[-1]
                if orm_table in available or table_name in available:
                    tables_to_load.append(orm_table)
        else:
            tables_to_load = [
                t if isinstance(t, str) else f"{t.schema.name}.{t.name}"
                for t in tables
            ]

        # Compute insertion order
        try:
            ordered_tables = self.orderer.get_insertion_order(tables_to_load)
        except ValueError as e:
            # Some tables might not be in the model, just use original order
            logger.warning(f"Could not compute FK ordering: {e}")
            ordered_tables = [
                self.orderer._to_table(t) if isinstance(t, str) else t
                for t in tables_to_load
                if self._table_exists(t)
            ]

        # Load in order
        counts = {}
        total_tables = len(ordered_tables)

        for i, table in enumerate(ordered_tables):
            table_key = f"{table.schema.name}.{table.name}"

            count = self._load_table(table, on_conflict, batch_size)
            counts[table_key] = count

            if progress_callback:
                progress_callback(table_key, count, total_tables)

            if count > 0:
                logger.info(f"Loaded {count} rows into {table_key}")

        return counts

    def _table_exists(self, table: str | DerivaTable) -> bool:
        """Check if table exists in ORM."""
        try:
            if isinstance(table, str):
                self.orm.find_table(table)
            else:
                self.orm.find_table(f"{table.schema.name}.{table.name}")
            return True
        except KeyError:
            return False

    def _load_table(
        self,
        table: DerivaTable,
        on_conflict: str,
        batch_size: int,
    ) -> int:
        """Load a single table.

        Args:
            table: Table to load.
            on_conflict: Conflict handling strategy.
            batch_size: Rows per batch.

        Returns:
            Number of rows loaded.
        """
        table_key = f"{table.schema.name}.{table.name}"

        # Find SQL table
        try:
            sql_table = self.orm.find_table(table_key)
        except KeyError:
            logger.warning(f"Table {table_key} not found in ORM")
            return 0

        # Check if source has data
        if not self.source.has_table(table):
            logger.debug(f"No data for {table_key} in source")
            return 0

        # Get data from source
        rows_loaded = 0
        batch = []

        with self.orm.engine.begin() as conn:
            for row in self.source.get_table_data(table):
                batch.append(row)

                if len(batch) >= batch_size:
                    rows_loaded += self._insert_batch(
                        conn, sql_table, batch, on_conflict
                    )
                    batch = []

            # Insert remaining rows
            if batch:
                rows_loaded += self._insert_batch(
                    conn, sql_table, batch, on_conflict
                )

        return rows_loaded

    def _insert_batch(
        self,
        conn: Any,
        sql_table: Any,
        rows: list[dict[str, Any]],
        on_conflict: str,
    ) -> int:
        """Insert a batch of rows.

        Args:
            conn: Database connection.
            sql_table: SQLAlchemy table.
            rows: List of row dictionaries.
            on_conflict: Conflict handling strategy.

        Returns:
            Number of rows inserted.
        """
        if not rows:
            return 0

        try:
            if on_conflict == "ignore":
                stmt = sqlite_insert(sql_table).on_conflict_do_nothing()
            elif on_conflict == "replace":
                # For SQLite, we need to specify all columns for upsert
                stmt = sqlite_insert(sql_table)
                update_cols = {
                    c.name: c for c in stmt.excluded
                    if c.name not in ("RID",)  # Don't update primary key
                }
                stmt = stmt.on_conflict_do_update(
                    index_elements=["RID"],
                    set_=update_cols,
                )
            else:
                stmt = sql_table.insert()

            conn.execute(stmt, rows)
            return len(rows)

        except Exception as e:
            logger.error(f"Error inserting into {sql_table.name}: {e}")
            if on_conflict == "error":
                raise
            return 0

    def load_table(
        self,
        table: str | DerivaTable,
        on_conflict: str = "ignore",
        batch_size: int = 1000,
    ) -> int:
        """Load a single table (without FK ordering).

        Use this when you know the dependencies are already satisfied
        or for loading a single table.

        Args:
            table: Table to load.
            on_conflict: Conflict handling strategy.
            batch_size: Rows per batch.

        Returns:
            Number of rows loaded.
        """
        if isinstance(table, str):
            table = self.orderer._to_table(table)

        return self._load_table(table, on_conflict, batch_size)

    def get_load_order(
        self,
        tables: list[str | DerivaTable] | None = None,
    ) -> list[str]:
        """Get the FK-safe load order for tables without loading.

        Useful for previewing or manually controlling load order.

        Args:
            tables: Tables to order. If None, orders all available.

        Returns:
            List of table names in safe insertion order.
        """
        if tables is None:
            available = self.source.list_available_tables()
            tables = [t for t in available if self._table_exists(t)]

        ordered = self.orderer.get_insertion_order(tables)
        return [f"{t.schema.name}.{t.name}" for t in ordered]

    def validate_load_order(
        self,
        tables: list[str | DerivaTable],
    ) -> list[tuple[str, str, str]]:
        """Validate that tables can be loaded in the given order.

        Args:
            tables: Ordered list of tables.

        Returns:
            List of FK violations as (table, missing_dep, fk_name) tuples.
            Empty if order is valid.
        """
        return self.orderer.validate_insertion_order(tables)
