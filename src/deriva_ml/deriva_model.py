"""
`deriva_ml_base.py` is the core module for the Deriva ML project.  This module implements the DerivaML class, which is
the primary interface to the Deriva based catalogs.  The module also implements the Feature and Vocabulary functions
in the DerivaML.

DerivaML and its associated classes all depend on a catalog that implements a `deriva-ml` schema with tables and
relationships that follow a specific data model.

"""

from deriva.core.ermrest_model import Table, Model
from deriva.core.ermrest_catalog import ErmrestCatalog

from .deriva_definitions import (
    DerivaMLException,
    ML_SCHEMA,
    DerivaSystemColumns,
    TableDefinition,
)

from pydantic import validate_call, ConfigDict


class DerivaModel:
    """Augmented interface to deriva model class.

    Attributes:
        domain_schema: Schema name for domain specific tables and relationships.
        model: ERMRest model for the catalog.
        schemas: ERMRest model for the catalog.
        catalog: ERMRest catalog for the model
        hostname: ERMRest catalog for the model

    """

    def __init__(
        self, model: Model, ml_schema: str = ML_SCHEMA, domain_schema: str = ""
    ):
        """Create and initialize a DerivaML instance.

        This method will connect to a catalog, and initialize local configuration for the ML execution.
        This class is intended to be used as a base class on which domain-specific interfaces are built.

        Args:
        """
        self.model = model
        self.configuration = None
        self.catalog: ErmrestCatalog = self.model.catalog
        self.hostname = (
            self.catalog.deriva_server.server
            if isinstance(self.catalog, ErmrestCatalog)
            else "localhost"
        )
        self.schemas = self.model.schemas

        self.ml_schema = ml_schema
        builtin_schemas = ["public", self.ml_schema, "www"]
        try:
            self.domain_schema = (
                domain_schema
                or [
                    s for s in self.model.schemas.keys() if s not in builtin_schemas
                ].pop()
            )
        except IndexError:
            # No domain schema defined.
            self.domain_schema = domain_schema

    def get_table(self, table: str | Table) -> Table:
        """Return the table object corresponding to the given table name.

        If the table name appears in more than one schema, return the first one you find.

        Args:
          table: A ERMRest table object or a string that is the name of the table.
          table: str | Table:

        Returns:
          Table object.
        """
        if isinstance(table, Table):
            return table
        for s in self.model.schemas.values():
            if table in s.tables.keys():
                return s.tables[table]
        raise DerivaMLException(f"The table {table} doesn't exist.")

    def is_vocabulary(self, table_name: str | Table) -> bool:
        """Check if a given table is a controlled vocabulary table.

        Args:
          table_name: A ERMRest table object or the name of the table.
          table_name: str | Table:

        Returns:
          Table object if the table is a controlled vocabulary, False otherwise.

        Raises:
          DerivaMLException: if the table doesn't exist.

        """
        vocab_columns = {"NAME", "URI", "SYNONYMS", "DESCRIPTION", "ID"}
        table = self.get_table(table_name)
        return vocab_columns.issubset({c.name.upper() for c in table.columns})

    def is_association(
        self, table_name: str | Table, unqualified: bool = True, pure: bool = True
    ) -> bool | set | int:
        """Check the specified table to see if it is an association table.

        Args:
            table_name: param unqualified:
            pure: return: (Default value = True)
            table_name: str | Table:
            unqualified:  (Default value = True)

        Returns:


        """
        table = self.get_table(table_name)
        return table.is_association(unqualified=unqualified, pure=pure)

    def is_asset(self, table_name: str | Table) -> bool:
        """True if the specified table is an asset table.

        Args:
            table_name: str | Table:

        Returns:
            True if the specified table is an asset table, False otherwise.

        """
        asset_columns = {"Filename", "URL", "Length", "MD5", "Description"}
        table = self.get_table(table_name)
        return asset_columns.issubset({c.name for c in table.columns})

    def find_assets(self, with_metadata: bool = False) -> list[Table]:
        """Return the list of asset tables in the current model"""
        return [
            t
            for s in self.model.schemas.values()
            for t in s.tables.values()
            if self.is_asset(t)
        ]

    def find_vocabularies(self) -> list[Table]:
        """Return a list of all the controlled vocabulary tables in the domain schema."""
        return [
            t
            for s in self.model.schemas.values()
            for t in s.tables.values()
            if self.is_vocabulary(t)
        ]

    def asset_metadata(self, table: str | Table) -> set[str]:
        """Return the metadata columns for an asset table."""

        table = self.get_table(table)
        asset_columns = {
            "Filename",
            "URL",
            "Length",
            "MD5",
            "Description",
        }.union(set(DerivaSystemColumns))

        if not self.is_asset(table):
            raise DerivaMLException(f"{table.name} is not an asset table.")
        return {c.name for c in table.columns} - asset_columns

    def apply(self):
        """Call Ermrestmodel.apply"""
        if self.catalog == "file-system":
            raise DerivaMLException("Cannot apply() to non-catalog model.")
        else:
            self.model.apply()

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def create_table(self, table_def: TableDefinition) -> Table:
        """Create a new table from TableDefinition."""
        return self.model.schemas[self.domain_schema].create_table(
            table_def.model_dump()
        )
