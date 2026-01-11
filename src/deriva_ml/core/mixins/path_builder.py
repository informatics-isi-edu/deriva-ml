"""Path builder mixin for DerivaML.

This module provides the PathBuilderMixin class which handles
catalog path building and table access utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import pandas as pd
import deriva.core.datapath as datapath
from deriva.core.datapath import _SchemaWrapper as SchemaWrapper
from deriva.core.ermrest_catalog import ErmrestCatalog, ErmrestSnapshot
from deriva.core.ermrest_model import Table

from deriva_ml.dataset.upload import table_path as _table_path

if TYPE_CHECKING:
    from deriva_ml.model.catalog import DerivaModel


class PathBuilderMixin:
    """Mixin providing path building and table access utilities.

    This mixin requires the host class to have:
        - catalog: ErmrestCatalog or ErmrestSnapshot instance
        - domain_schema: str - name of the domain schema
        - model: DerivaModel instance
        - working_dir: Path - working directory path

    Methods:
        pathBuilder: Get catalog path builder for queries
        domain_path: Property returning path builder for domain schema
        table_path: Get local filesystem path for table CSV files
        get_table_as_dataframe: Get table contents as pandas DataFrame
        get_table_as_dict: Get table contents as dictionaries
    """

    # Type hints for IDE support - actual attributes from host class
    catalog: ErmrestCatalog | ErmrestSnapshot
    domain_schema: str
    model: "DerivaModel"
    working_dir: Path

    def pathBuilder(self) -> SchemaWrapper:
        """Returns catalog path builder for queries.

        The path builder provides a fluent interface for constructing complex queries against the catalog.
        This is a core component used by many other methods to interact with the catalog.

        Returns:
            datapath._CatalogWrapper: A new instance of the catalog path builder.

        Example:
            >>> path = ml.pathBuilder.schemas['my_schema'].tables['my_table']
            >>> results = path.entities().fetch()
        """
        return self.catalog.getPathBuilder()

    @property
    def domain_path(self) -> datapath.DataPath:
        """Returns path builder for domain schema.

        Provides a convenient way to access tables and construct queries within the domain-specific schema.

        Returns:
            datapath._CatalogWrapper: Path builder object scoped to the domain schema.

        Example:
            >>> domain = ml.domain_path
            >>> results = domain.my_table.entities().fetch()
        """
        return self.pathBuilder().schemas[self.domain_schema]

    def table_path(self, table: str | Table) -> Path:
        """Returns a local filesystem path for table CSV files.

        Generates a standardized path where CSV files should be placed when preparing to upload data to a table.
        The path follows the project's directory structure conventions.

        Args:
            table: Name of the table or Table object to get the path for.

        Returns:
            Path: Filesystem path where the CSV file should be placed.

        Example:
            >>> path = ml.table_path("experiment_results")
            >>> df.to_csv(path) # Save data for upload
        """
        return _table_path(
            self.working_dir,
            schema=self.domain_schema,
            table=self.model.name_to_table(table).name,
        )

    def get_table_as_dataframe(self, table: str) -> pd.DataFrame:
        """Get table contents as a pandas DataFrame.

        Retrieves all contents of a table from the catalog.

        Args:
            table: Name of the table to retrieve.

        Returns:
            DataFrame containing all table contents.
        """
        return pd.DataFrame(list(self.get_table_as_dict(table)))

    def get_table_as_dict(self, table: str) -> Iterable[dict[str, Any]]:
        """Get table contents as dictionaries.

        Retrieves all contents of a table from the catalog.

        Args:
            table: Name of the table to retrieve.

        Returns:
            Iterable yielding dictionaries for each row.
        """
        table_obj = self.model.name_to_table(table)
        pb = self.pathBuilder()
        yield from pb.schemas[table_obj.schema.name].tables[table_obj.name].entities().fetch()
