"""
`deriva_ml_base.py` is the core module for the Deriva ML project.  This module implements the DerivaML class, which is
the primary interface to the Deriva based catalogs.  The module also implements the Feature and Vocabulary functions
in the DerivaML.

DerivaML and its associated classes all depend on a catalog that implements a `deriva-ml` schema with tables and
relationships that follow a specific data model.

"""

from deriva.core.ermrest_model import Table, Column, Model, FindAssociationResult
from deriva.core.ermrest_catalog import ErmrestCatalog
from .feature import Feature

from .deriva_definitions import (
    DerivaMLException,
    ML_SCHEMA,
    DerivaSystemColumns,
    TableDefinition,
)

from collections import Counter
from pydantic import validate_call, ConfigDict
from typing import Iterable


class DerivaModel:
    """Augmented interface to deriva model class.

    Attributes:
        domain_schema: Schema name for domain specific tables and relationships.
        model: ERMRest model for the catalog.
        schemas: ERMRest model for the catalog.
        catalog: ERMRest catalog for the model
        hostname: ERMRest catalog for the model
        ml_schema: The ML schema for the catalog.
        domain_schema: The domain schema for the catalog.

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

    def name_to_table(self, table: str | Table) -> Table:
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
        table = self.name_to_table(table_name)
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
        table = self.name_to_table(table_name)
        return table.is_association(unqualified=unqualified, pure=pure)

    def is_asset(self, table_name: str | Table) -> bool:
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

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def find_features(self, table: Table | str) -> Iterable[Feature]:
        """List the names of the features in the specified table.

        Args:
            table: The table to find features for.
            table: Table | str:

        Returns:
            An iterable of FeatureResult instances that describe the current features in the table.
        """
        table = self.name_to_table(table)

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
            Feature(a, self)
            for a in table.find_associations(min_arity=3, max_arity=3, pure=False)
            if is_feature(a)
        ]

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
        table = self.name_to_table(table)
        try:
            return [
                f for f in self.find_features(table) if f.feature_name == feature_name
            ][0]
        except IndexError:
            raise DerivaMLException(
                f"Feature {table.name}:{feature_name} doesn't exist."
            )

    def asset_metadata(self, table: str | Table) -> set[str]:
        """Return the metadata columns for an asset table."""

        table = self.name_to_table(table)
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
        """Call ERMRestModel.apply"""
        if self.catalog == "file-system":
            raise DerivaMLException("Cannot apply() to non-catalog model.")
        else:
            self.model.apply()

    def _table_relationship(
        self, table1: Table | str, table2: Table | str
    ) -> tuple[Column, Column]:
        """Return columns used to relate two tables."""
        table1 = self.name_to_table(table1)
        table2 = self.name_to_table(table2)
        relationships = [
            (fk.foreign_key_columns[0], fk.referenced_columns[0])
            for fk in table1.foreign_keys
            if fk.pk_table == table2
        ]
        relationships.extend(
            [
                (fk.referenced_columns[0], fk.foreign_key_columns[0])
                for fk in table1.referenced_by
                if fk.table == table2
            ]
        )
        if len(relationships) != 1:
            raise DerivaMLException(
                f"Ambiguous linkage between {table1.name} and {table2.name}"
            )
        return relationships[0]

    def _schema_to_paths(
        self, root: Table = None, path: list[Table] = None
    ) -> list[list[Table]]:
        """Recursively walk over the domain schema graph and extend the current path.

        Walk a schema graph and return a list all  the paths through the graph.

        Args:
            path: Source path so far

        Returns:
          A list of all the paths through the graph.  Each path is a list of tables.

        """
        root = root or self.model.schemas[self.ml_schema].tables["Dataset"]
        path = path.copy() if path else []
        parent = path[-1] if path else None  # Table that we are coming from.
        path.append(root)
        paths = [path]

        def find_arcs(table: Table) -> set[Table]:
            """Given a path through the model, return the FKs that link the tables"""
            arc_list = [fk.pk_table for fk in table.foreign_keys] + [
                fk.table for fk in table.referenced_by
            ]
            arc_list = [
                t
                for t in arc_list
                if t.schema.name in {self.domain_schema, self.ml_schema}
            ]
            domain_tables = [t for t in arc_list if t.schema.name == self.domain_schema]
            if multiple_columns := [
                c for c, cnt in Counter(domain_tables).items() if cnt > 1
            ]:
                raise DerivaMLException(
                    f"Ambiguous relationship in {table.name} {multiple_columns}"
                )
            return set(arc_list)

        def is_nested_dataset_loopback(n1: Table, n2: Table) -> bool:
            """Test to see if node is an association table used to link elements to datasets."""
            # If we have node_name <- node_name_dataset-> Dataset then we are looping
            # back around to a new dataset element
            dataset_table = self.model.schemas[self.ml_schema].tables["Dataset"]
            assoc_table = [
                a for a in dataset_table.find_associations() if a.table == n2
            ]
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
                raise DerivaMLException(
                    f"Cycle in schema path: {child.name} path:{[p.name for p in path]}"
                )

            paths.extend(self._schema_to_paths(child, path))
        return paths

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def create_table(self, table_def: TableDefinition) -> Table:
        """Create a new table from TableDefinition."""
        return self.model.schemas[self.domain_schema].create_table(
            table_def.model_dump()
        )
