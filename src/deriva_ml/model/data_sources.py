"""Data sources for populating SQLite databases.

This module provides the DataSource protocol and implementations for
reading data from various sources:

- BagDataSource: Reads from BDBag CSV files
- CatalogDataSource: Fetches from remote Deriva catalog via ERMrest API

These are used with DataLoader in Phase 2 of the two-phase pattern.

Example:
    # From bag
    source = BagDataSource(bag_path)
    for row in source.get_table_data(table):
        print(row)

    # From catalog
    source = CatalogDataSource(catalog, schemas=['domain', 'deriva-ml'])
    for row in source.get_table_data(table):
        print(row)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable
from urllib.parse import urlparse

from deriva.core import ErmrestCatalog
from deriva.core.ermrest_model import Model
from deriva.core.ermrest_model import Table as DerivaTable

logger = logging.getLogger(__name__)


# Standard asset table columns
ASSET_COLUMNS = {"Filename", "URL", "Length", "MD5", "Description"}


@runtime_checkable
class DataSource(Protocol):
    """Protocol for data sources that can fill a database.

    Implementations provide data for populating SQLite tables from
    different sources (bags, remote catalogs, etc.).

    This is used with DataLoader in Phase 2 of the two-phase pattern.
    """

    def get_table_data(
        self,
        table: DerivaTable | str,
    ) -> Iterator[dict[str, Any]]:
        """Yield rows for a table as dictionaries.

        Args:
            table: Table object or name to get data for.

        Yields:
            Dictionary per row with column names as keys.
        """
        ...

    def has_table(self, table: DerivaTable | str) -> bool:
        """Check if this source has data for the table.

        Args:
            table: Table object or name to check.

        Returns:
            True if data is available for this table.
        """
        ...

    def list_available_tables(self) -> list[str]:
        """List tables with available data.

        Returns:
            List of table names (may include schema prefix).
        """
        ...


class BagDataSource:
    """DataSource implementation for BDBag directories.

    Reads data from CSV files in a bag's data/ directory.
    Handles asset URL localization via fetch.txt.

    Example:
        source = BagDataSource(Path("/path/to/bag"))

        # List available tables
        print(source.list_available_tables())

        # Get data for a table
        for row in source.get_table_data("Image"):
            print(row["Filename"])
    """

    def __init__(
        self,
        bag_path: Path,
        model: Model | None = None,
        asset_localization: bool = True,
    ):
        """Initialize from a bag path.

        Args:
            bag_path: Path to BDBag directory.
            model: Optional ERMrest Model for schema info. If not provided,
                will try to load from bag's schema.json.
            asset_localization: Whether to localize asset URLs to local paths
                using fetch.txt mapping.
        """
        self.bag_path = Path(bag_path)
        self.data_path = self.bag_path / "data"

        # Load model if not provided
        if model is None:
            schema_file = self.data_path / "schema.json"
            if schema_file.exists():
                self.model = Model.fromfile("file-system", schema_file)
            else:
                self.model = None
                logger.warning(f"No schema.json found in {self.bag_path}")
        else:
            self.model = model

        # Build asset map for URL localization
        self._asset_map = self._build_asset_map() if asset_localization else {}

        # Cache of table name -> csv file path
        self._csv_cache: dict[str, Path] = {}
        self._build_csv_cache()

    def _build_csv_cache(self) -> None:
        """Build cache mapping table names to CSV file paths."""
        for csv_file in self.data_path.rglob("*.csv"):
            table_name = csv_file.stem
            self._csv_cache[table_name] = csv_file

    def _build_asset_map(self) -> dict[str, str]:
        """Build a map from remote URLs to local file paths using fetch.txt.

        Returns:
            Dictionary mapping URL paths to local file paths.
        """
        fetch_map = {}
        fetch_file = self.bag_path / "fetch.txt"

        if not fetch_file.exists():
            logger.debug(f"No fetch.txt in bag {self.bag_path.name}")
            return fetch_map

        try:
            with fetch_file.open(newline="\n") as f:
                for row in f:
                    # Rows in fetch.txt are tab-separated: URL, size, local_path
                    fields = row.split("\t")
                    if len(fields) >= 3:
                        local_file = fields[2].replace("\n", "")
                        local_path = f"{self.bag_path}/{local_file}"
                        fetch_map[urlparse(fields[0]).path] = local_path
        except Exception as e:
            logger.warning(f"Error reading fetch.txt: {e}")

        return fetch_map

    def _get_table_name(self, table: DerivaTable | str) -> str:
        """Extract table name from table object or string."""
        if isinstance(table, DerivaTable):
            return table.name
        # Handle schema.table format
        if "." in table:
            return table.split(".")[-1]
        return table

    def _is_asset_table(self, table_name: str) -> bool:
        """Check if a table is an asset table (has Filename, URL, etc. columns)."""
        if self.model is None:
            return False

        for schema in self.model.schemas.values():
            if table_name in schema.tables:
                table = schema.tables[table_name]
                return ASSET_COLUMNS.issubset({c.name for c in table.columns})
        return False

    def _localize_asset_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Replace URL with local path in asset table row.

        Args:
            row: Dictionary of column values.

        Returns:
            Updated dictionary with localized file path.
        """
        if "URL" in row and "Filename" in row:
            url = row.get("URL")
            if url and url in self._asset_map:
                row = dict(row)  # Copy to avoid mutating original
                row["Filename"] = self._asset_map[url]
        return row

    def get_table_data(
        self,
        table: DerivaTable | str,
    ) -> Iterator[dict[str, Any]]:
        """Read table data from CSV file.

        Args:
            table: Table object or name.

        Yields:
            Dictionary per row with column names as keys.
        """
        table_name = self._get_table_name(table)
        csv_file = self._csv_cache.get(table_name)

        if csv_file is None or not csv_file.exists():
            logger.debug(f"No CSV file found for table {table_name}")
            return

        is_asset = self._is_asset_table(table_name)

        with csv_file.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if is_asset and self._asset_map:
                    row = self._localize_asset_row(row)
                yield row

    def has_table(self, table: DerivaTable | str) -> bool:
        """Check if CSV exists for table.

        Args:
            table: Table object or name.

        Returns:
            True if CSV file exists for this table.
        """
        table_name = self._get_table_name(table)
        return table_name in self._csv_cache

    def list_available_tables(self) -> list[str]:
        """List all CSV files in data directory.

        Returns:
            List of table names (without .csv extension).
        """
        return sorted(self._csv_cache.keys())

    def get_row_count(self, table: DerivaTable | str) -> int:
        """Get the number of rows in a table's CSV file.

        Args:
            table: Table object or name.

        Returns:
            Number of data rows (excluding header).
        """
        table_name = self._get_table_name(table)
        csv_file = self._csv_cache.get(table_name)

        if csv_file is None or not csv_file.exists():
            return 0

        with csv_file.open(newline="") as f:
            # Count lines minus header
            return sum(1 for _ in f) - 1


class CatalogDataSource:
    """DataSource implementation for remote Deriva catalog.

    Fetches data via ERMrest API / datapath with pagination support.

    Example:
        catalog = server.connect_ermrest(catalog_id)
        source = CatalogDataSource(catalog, schemas=['domain', 'deriva-ml'])

        # List available tables
        print(source.list_available_tables())

        # Get data for a table
        for row in source.get_table_data("Image"):
            print(row["Filename"])
    """

    def __init__(
        self,
        catalog: ErmrestCatalog,
        schemas: list[str],
        batch_size: int = 1000,
    ):
        """Initialize from catalog connection.

        Args:
            catalog: ERMrest catalog connection.
            schemas: Schemas to fetch data from.
            batch_size: Number of rows per API request.
        """
        self.catalog = catalog
        self.schemas = schemas
        self.batch_size = batch_size
        self._pb = catalog.getPathBuilder()
        self._model = catalog.getCatalogModel()

    def _get_table_info(self, table: DerivaTable | str) -> tuple[str, str] | None:
        """Get schema and table name for a table.

        Args:
            table: Table object or name.

        Returns:
            Tuple of (schema_name, table_name) or None if not found.
        """
        if isinstance(table, DerivaTable):
            return table.schema.name, table.name

        # Handle schema.table format
        if "." in table:
            parts = table.split(".")
            schema_name, table_name = parts[0], parts[1]
            if schema_name in self.schemas:
                return schema_name, table_name
            return None

        # Search schemas for table
        for schema_name in self.schemas:
            if schema_name in self._model.schemas:
                schema = self._model.schemas[schema_name]
                if table in schema.tables:
                    return schema_name, table

        return None

    def get_table_data(
        self,
        table: DerivaTable | str,
    ) -> Iterator[dict[str, Any]]:
        """Fetch table data via ERMrest API.

        Uses pagination to handle large tables efficiently.

        Args:
            table: Table object or name.

        Yields:
            Dictionary per row with column names as keys.
        """
        table_info = self._get_table_info(table)
        if table_info is None:
            logger.warning(f"Table {table} not found in schemas {self.schemas}")
            return

        schema_name, table_name = table_info

        # Build path
        path = self._pb.schemas[schema_name].tables[table_name]

        # Paginated fetch using RID ordering
        last_rid = None
        while True:
            # Build query with optional RID filter
            query = path.entities()
            if last_rid is not None:
                query = query.filter(path.RID > last_rid)

            # Fetch batch ordered by RID
            try:
                entities = list(query.sort(path.RID).fetch(limit=self.batch_size))
            except Exception as e:
                logger.error(f"Error fetching from {schema_name}.{table_name}: {e}")
                break

            if not entities:
                break

            for entity in entities:
                yield dict(entity)

            # Track last RID for pagination
            last_rid = entities[-1]["RID"]

            if len(entities) < self.batch_size:
                break

    def has_table(self, table: DerivaTable | str) -> bool:
        """Check if table exists in catalog.

        Args:
            table: Table object or name.

        Returns:
            True if table exists in configured schemas.
        """
        return self._get_table_info(table) is not None

    def list_available_tables(self) -> list[str]:
        """List all tables in configured schemas.

        Returns:
            List of fully-qualified table names (schema.table).
        """
        tables = []
        for schema_name in self.schemas:
            if schema_name in self._model.schemas:
                schema = self._model.schemas[schema_name]
                for table_name in schema.tables.keys():
                    tables.append(f"{schema_name}.{table_name}")
        return sorted(tables)

    def get_row_count(self, table: DerivaTable | str) -> int:
        """Get the number of rows in a table.

        Args:
            table: Table object or name.

        Returns:
            Number of rows in the table.
        """
        table_info = self._get_table_info(table)
        if table_info is None:
            return 0

        schema_name, table_name = table_info
        path = self._pb.schemas[schema_name].tables[table_name]

        try:
            # Use count aggregate
            result = path.aggregates(path.RID.cnt.alias("count")).fetch()
            return result[0]["count"] if result else 0
        except Exception as e:
            logger.error(f"Error counting {schema_name}.{table_name}: {e}")
            return 0
