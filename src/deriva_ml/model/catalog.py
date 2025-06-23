"""
Model management for Deriva ML catalogs.

This module provides the DerivaModel class which augments the standard Deriva model class with
ML-specific functionality. It handles schema management, feature definitions, and asset tracking.
"""

from __future__ import annotations

# Standard library imports
from collections import Counter
from typing import Any, Callable, Final, Iterable, NewType, TypeAlias

from deriva.core.ermrest_catalog import ErmrestCatalog

# Deriva imports
from deriva.core.ermrest_model import Column, FindAssociationResult, Model, Schema, Table

# Third-party imports
from pydantic import ConfigDict, validate_call

from deriva_ml.core.definitions import (
    ML_SCHEMA,
    DerivaAssetColumns,
    TableDefinition,
)
from deriva_ml.core.exceptions import DerivaMLException, DerivaMLTableTypeError

# Local imports
from deriva_ml.feature import Feature

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


# Define common types:
TableInput: TypeAlias = str | Table
SchemaDict: TypeAlias = dict[str, Schema]
FeatureList: TypeAlias = Iterable[Feature]
SchemaName = NewType("SchemaName", str)
ColumnSet: TypeAlias = set[Column]
AssociationResult: TypeAlias = FindAssociationResult
TableSet: TypeAlias = set[Table]
PathList: TypeAlias = list[list[Table]]

# Define constants:
VOCAB_COLUMNS: Final[set[str]] = {"NAME", "URI", "SYNONYMS", "DESCRIPTION", "ID"}
ASSET_COLUMNS: Final[set[str]] = {"Filename", "URL", "Length", "MD5", "Description"}

FilterPredicate = Callable[[Table], bool]


class DerivaModel:
    """Augmented interface to deriva model class.

    This class provides a number of DerivaML specific methods that augment the interface in the deriva model class.

    Attributes:
        domain_schema: Schema name for domain-specific tables and relationships.
        model: ERMRest model for the catalog.
        catalog: ERMRest catalog for the model
        hostname: ERMRest catalog for the model
        ml_schema: The ML schema for the catalog.
        domain_schema: The domain schema for the catalog.

    """

    def __init__(
        self,
        model: Model,
        ml_schema: str = ML_SCHEMA,
        domain_schema: str | None = None,
    ):
        """Create and initialize a DerivaML instance.

        This method will connect to a catalog, and initialize local configuration for the ML execution.
        This class is intended to be used as a base class on which domain-specific interfaces are built.

        Args:
            model: The ERMRest model for the catalog.
            ml_schema: The ML schema name.
            domain_schema: The domain schema name.
        """
        self.model = model
        self.configuration = None
        self.catalog: ErmrestCatalog = self.model.catalog
        self.hostname = self.catalog.deriva_server.server if isinstance(self.catalog, ErmrestCatalog) else "localhost"

        self.ml_schema = ml_schema
        builtin_schemas = ("public", self.ml_schema, "www", "WWW")
        if domain_schema:
            self.domain_schema = domain_schema
        else:
            if len(user_schemas := {k for k in self.model.schemas.keys()} - set(builtin_schemas)) == 1:
                self.domain_schema = user_schemas.pop()
            else:
                raise DerivaMLException(f"Ambiguous domain schema: {user_schemas}")

    def refresh_model(self) -> None:
        self.model = self.catalog.getCatalogModel()

    @property
    def schemas(self) -> dict[str, Schema]:
        return self.model.schemas

    @property
    def chaise_config(self) -> dict[str, Any]:
        """Return the chaise configuration."""
        return self.model.chaise_config

    def __getattr__(self, name: str) -> Any:
        # Called only if `name` is not found in Manager.  Delegate attributes to model class.
        return getattr(self.model, name)

    def name_to_table(self, table: TableInput) -> Table:
        """Return the table object corresponding to the given table name.

        If the table name appears in more than one schema, return the first one you find.

        Args:
          table: A ERMRest table object or a string that is the name of the table.

        Returns:
          Table object.
        """
        if isinstance(table, Table):
            return table
        if table in (s := self.model.schemas[self.domain_schema].tables):
            return s[table]
        for s in [self.model.schemas[sname] for sname in [self.domain_schema, self.ml_schema, "WWW"]]:
            if table in s.tables.keys():
                return s.tables[table]
        raise DerivaMLException(f"The table {table} doesn't exist.")

    def is_vocabulary(self, table_name: TableInput) -> bool:
        """Check if a given table is a controlled vocabulary table.

        Args:
          table_name: A ERMRest table object or the name of the table.

        Returns:
          Table object if the table is a controlled vocabulary, False otherwise.

        Raises:
          DerivaMLException: if the table doesn't exist.

        """
        vocab_columns = {"NAME", "URI", "SYNONYMS", "DESCRIPTION", "ID"}
        table = self.name_to_table(table_name)
        return vocab_columns.issubset({c.name.upper() for c in table.columns})

    def is_association(
        self,
        table_name: str | Table,
        unqualified: bool = True,
        pure: bool = True,
        min_arity: int = 2,
        max_arity: int = 2,
    ) -> bool | set[str] | int:
        """Check the specified table to see if it is an association table.

        Args:
            table_name: param unqualified:
            pure: return: (Default value = True)
            table_name: str | Table:
            unqualified:  (Default value = True)

        Returns:


        """
        table = self.name_to_table(table_name)
        return table.is_association(unqualified=unqualified, pure=pure, min_arity=min_arity, max_arity=max_arity)

    def find_association(self, table1: Table | str, table2: Table | str) -> tuple[Table, Column, Column]:
        """Given two tables, return an association table that connects the two and the two columns used to link them..

        Raises:
            DerivaML exception if there is either not an association table or more than one association table.
        """
        table1 = self.name_to_table(table1)
        table2 = self.name_to_table(table2)

        tables = [
            (a.table, a.self_fkey.columns[0].name, other_key.columns[0].name)
            for a in table1.find_associations(pure=False)
            if len(a.other_fkeys) == 1 and (other_key := a.other_fkeys.pop()).pk_table == table2
        ]

        if len(tables) == 1:
            return tables[0]
        elif len(tables) == 0:
            raise DerivaMLException(f"No association tables found between {table1.name} and {table2.name}.")
        else:
            raise DerivaMLException(
                f"There are {len(tables)} association tables between {table1.name} and {table2.name}."
            )

    def is_asset(self, table_name: TableInput) -> bool:
        """True if the specified table is an asset table.

        Args:
            table_name: str | Table:

        Returns:
            True if the specified table is an asset table, False otherwise.

        """
        asset_columns = {"Filename", "URL", "Length", "MD5", "Description"}
        table = self.name_to_table(table_name)
        return asset_columns.issubset({c.name for c in table.columns})

    def find_assets(self, with_metadata: bool = False) -> list[Table]:
        """Return the list of asset tables in the current model"""
        return [t for s in self.model.schemas.values() for t in s.tables.values() if self.is_asset(t)]

    def find_vocabularies(self) -> list[Table]:
        """Return a list of all the controlled vocabulary tables in the domain schema."""
        return [t for s in self.model.schemas.values() for t in s.tables.values() if self.is_vocabulary(t)]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def find_features(self, table: TableInput) -> Iterable[Feature]:
        """List the names of the features in the specified table.

        Args:
            table: The table to find features for.
            table: Table | str:

        Returns:
            An iterable of FeatureResult instances that describe the current features in the table.
        """
        table = self.name_to_table(table)

        def is_feature(a: FindAssociationResult) -> bool:
            """Check if association represents a feature.

            Args:
                a: Association result to check
            Returns:
                bool: True if association represents a feature
            """
            return {
                "Feature_Name",
                "Execution",
                a.self_fkey.foreign_key_columns[0].name,
            }.issubset({c.name for c in a.table.columns})

        return [
            Feature(a, self) for a in table.find_associations(min_arity=3, max_arity=3, pure=False) if is_feature(a)
        ]

    def lookup_feature(self, table: TableInput, feature_name: str) -> Feature:
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
        table = self.name_to_table(table)
        try:
            return [f for f in self.find_features(table) if f.feature_name == feature_name][0]
        except IndexError:
            raise DerivaMLException(f"Feature {table.name}:{feature_name} doesn't exist.")

    def asset_metadata(self, table: str | Table) -> set[str]:
        """Return the metadata columns for an asset table."""

        table = self.name_to_table(table)

        if not self.is_asset(table):
            raise DerivaMLTableTypeError("asset table", table.name)
        return {c.name for c in table.columns} - DerivaAssetColumns

    def apply(self) -> None:
        """Call ERMRestModel.apply"""
        if self.catalog == "file-system":
            raise DerivaMLException("Cannot apply() to non-catalog model.")
        else:
            self.model.apply()

    def _table_relationship(
        self,
        table1: TableInput,
        table2: TableInput,
    ) -> tuple[Column, Column]:
        """Return columns used to relate two tables."""
        table1 = self.name_to_table(table1)
        table2 = self.name_to_table(table2)
        relationships = [
            (fk.foreign_key_columns[0], fk.referenced_columns[0]) for fk in table1.foreign_keys if fk.pk_table == table2
        ]
        relationships.extend(
            [(fk.referenced_columns[0], fk.foreign_key_columns[0]) for fk in table1.referenced_by if fk.table == table2]
        )
        if len(relationships) != 1:
            raise DerivaMLException(f"Ambiguous linkage between {table1.name} and {table2.name}")
        return relationships[0]

    def _schema_to_paths(
        self,
        root: Table | None = None,
        path: list[Table] | None = None,
    ) -> list[list[Table]]:
        """Return a list of paths through the schema graph.

        Args:
            root: The root table to start from.
            path: The current path being built.

        Returns:
            A list of paths through the schema graph.
        """
        path = path or []

        root = root or self.model.schemas[self.ml_schema].tables["Dataset"]
        path = path.copy() if path else []
        parent = path[-1] if path else None  # Table that we are coming from.
        path.append(root)
        paths = [path]

        def find_arcs(table: Table) -> set[Table]:
            """Given a path through the model, return the FKs that link the tables"""
            arc_list = [fk.pk_table for fk in table.foreign_keys] + [fk.table for fk in table.referenced_by]
            arc_list = [t for t in arc_list if t.schema.name in {self.domain_schema, self.ml_schema}]
            domain_tables = [t for t in arc_list if t.schema.name == self.domain_schema]
            if multiple_columns := [c for c, cnt in Counter(domain_tables).items() if cnt > 1]:
                raise DerivaMLException(f"Ambiguous relationship in {table.name} {multiple_columns}")
            return set(arc_list)

        def is_nested_dataset_loopback(n1: Table, n2: Table) -> bool:
            """Test to see if node is an association table used to link elements to datasets."""
            # If we have node_name <- node_name_dataset-> Dataset then we are looping
            # back around to a new dataset element
            dataset_table = self.model.schemas[self.ml_schema].tables["Dataset"]
            assoc_table = [a for a in dataset_table.find_associations() if a.table == n2]
            return len(assoc_table) == 1 and n1 != dataset_table

        # Don't follow vocabulary terms back to their use.
        if self.is_vocabulary(root):
            return paths

        for child in find_arcs(root):
            if child.name in {"Dataset_Execution", "Dataset_Dataset", "Execution"}:
                continue
            if child == parent:
                # Don't loop back via referred_by
                continue
            if is_nested_dataset_loopback(root, child):
                continue
            if child in path:
                raise DerivaMLException(f"Cycle in schema path: {child.name} path:{[p.name for p in path]}")

            paths.extend(self._schema_to_paths(child, path))
        return paths

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def create_table(self, table_def: TableDefinition) -> Table:
        """Create a new table from TableDefinition."""
        return self.model.schemas[self.domain_schema].create_table(table_def.model_dump())
