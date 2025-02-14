"""
`deriva_ml_base.py` is the core module for the Deriva ML project.  This module implements the DerivaML class, which is
the primary interface to the Deriva based catalogs.  The module also implements the Feature and Vocabulary functions
in the DerivaML.

DerivaML and its associated classes all depend on a catalog that implements a `deriva-ml` schema with tables and
relationships that follow a specific data model.

"""

from typing import Iterable, Optional
from deriva.core.ermrest_model import FindAssociationResult, Table, Model, Schema
from deriva.core.ermrest_catalog import ErmrestCatalog
from pydantic import validate_call, ConfigDict
from .feature import Feature, FeatureRecord


from .deriva_definitions import (
    DerivaMLException,
    ML_SCHEMA,
)


class DerivaModel:
    """Augmented interface to deriva model class.

    Attributes:
        domain_schema: Schema name for domain specific tables and relationships.
        model: ERMRest model for the catalog
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

        self.ml_schema = ml_schema
        self.configuration = None

        builtin_schemas = ["public", self.ml_schema, "www"]
        self.domain_schema = (
            domain_schema
            or [s for s in self.model.schemas.keys() if s not in builtin_schemas].pop()
        )

    @property
    def schemas(self) -> dict[str, Schema]:
        return self.model.schemas

    @property
    def catalog(self) -> Optional[ErmrestCatalog]:
        return self.model.catalog

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

    def find_assets(self) -> list[Table]:
        """ """
        return [
            t
            for s in self.model.schemas.values()
            for t in s.tables.values()
            if self.is_asset(t)
        ]

    def feature_record_class(
        self, table: str | Table, feature_name: str
    ) -> type[FeatureRecord]:
        """Create a pydantic model for entries into the specified feature table.

        For information on how to
        See the pydantic documentation for more details about the pydantic model.

        Args:
            table: table name or object on which the feature is to be associated
            feature_name: name of the feature to be created
            table: str | Table:
            feature_name: str:

        Returns:
            A Feature class that can be used to create instances of the feature.
        """
        return self.lookup_feature(table, feature_name).feature_record_class()

    def lookup_feature(self, table: str | Table, feature_name: str) -> Feature:
        """Lookup the named feature associated with the provided table.

        Args:
            table: param feature_name:
            table: str | Table:
            feature_name: str:

        Returns:
            A Feature class that represents the requested feature.

        Raises:
          DerivaMLException: If the feature cannot be found.
        """
        table = self.get_table(table)
        try:
            return [
                f for f in self.find_features(table) if f.feature_name == feature_name
            ][0]
        except IndexError:
            raise DerivaMLException(
                f"Feature {table.name}:{feature_name} doesn't exist."
            )

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def find_features(self, table: Table | str) -> Iterable[Feature]:
        """List the names of the features in the specified table.

        Args:
            table: The table to find features for.
            table: Table | str:

        Returns:
            An iterable of FeatureResult instances that describe the current features in the table.
        """
        table = self.get_table(table)

        def is_feature(a: FindAssociationResult) -> bool:
            """

            Args:
              a: FindAssociationResult:

            Returns:

            """
            # return {'Feature_Name', 'Execution'}.issubset({c.name for c in a.table.columns})
            return {
                "Feature_Name",
                "Execution",
                a.self_fkey.foreign_key_columns[0].name,
            }.issubset({c.name for c in a.table.columns})

        return [
            Feature(a)
            for a in table.find_associations(min_arity=3, max_arity=3, pure=False)
            if is_feature(a)
        ]

    def find_vocabularies(self) -> Iterable[Table]:
        """Return a list of all the controlled vocabulary tables in the domain schema."""
        return [
            t
            for s in self.model.schemas.values()
            for t in s.tables.values()
            if self.is_vocabulary(t)
        ]

    def apply(self):
        if self.catalog == "file-system":
            raise DerivaMLException("Cannot apply() to non-catalog model.")
