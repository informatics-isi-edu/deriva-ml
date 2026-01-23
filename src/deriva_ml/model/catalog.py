"""
Model management for Deriva ML catalogs.

This module provides the DerivaModel class which augments the standard Deriva model class with
ML-specific functionality. It handles schema management, feature definitions, and asset tracking.
"""

from __future__ import annotations

# Standard library imports
from collections import Counter, defaultdict
from graphlib import CycleError, TopologicalSorter
from typing import Any, Callable, Final, Iterable, NewType, TypeAlias

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib
_ermrest_catalog = importlib.import_module("deriva.core.ermrest_catalog")
_ermrest_model = importlib.import_module("deriva.core.ermrest_model")

ErmrestCatalog = _ermrest_catalog.ErmrestCatalog
Column = _ermrest_model.Column
FindAssociationResult = _ermrest_model.FindAssociationResult
Model = _ermrest_model.Model
Schema = _ermrest_model.Schema
Table = _ermrest_model.Table

# Third-party imports
from pydantic import ConfigDict, validate_call

from deriva_ml.core.definitions import (
    ML_SCHEMA,
    RID,
    SYSTEM_SCHEMAS,
    DerivaAssetColumns,
    TableDefinition,
    get_domain_schemas,
    is_system_schema,
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
        model: ERMRest model for the catalog.
        catalog: ERMRest catalog for the model.
        hostname: Hostname of the ERMRest server.
        ml_schema: The ML schema name for the catalog.
        domain_schemas: Frozenset of all domain schema names in the catalog.
        default_schema: The default schema for table creation operations.

    """

    def __init__(
        self,
        model: Model,
        ml_schema: str = ML_SCHEMA,
        domain_schemas: set[str] | None = None,
        default_schema: str | None = None,
    ):
        """Create and initialize a DerivaModel instance.

        This method will connect to a catalog and initialize schema configuration.
        This class is intended to be used as a base class on which domain-specific interfaces are built.

        Args:
            model: The ERMRest model for the catalog.
            ml_schema: The ML schema name.
            domain_schemas: Optional explicit set of domain schema names. If None,
                auto-detects all non-system schemas.
            default_schema: The default schema for table creation operations. If None
                and there is exactly one domain schema, that schema is used as default.
                If there are multiple domain schemas, default_schema must be specified.
        """
        self.model = model
        self.configuration = None
        self.catalog: ErmrestCatalog = self.model.catalog
        self.hostname = self.catalog.deriva_server.server if isinstance(self.catalog, ErmrestCatalog) else "localhost"

        self.ml_schema = ml_schema
        self._system_schemas = frozenset(SYSTEM_SCHEMAS | {ml_schema})

        # Determine domain schemas
        if domain_schemas is not None:
            self.domain_schemas = frozenset(domain_schemas)
        else:
            # Auto-detect all domain schemas
            self.domain_schemas = get_domain_schemas(self.model.schemas.keys(), ml_schema)

        # Determine default schema for table creation
        if default_schema is not None:
            if default_schema not in self.domain_schemas:
                raise DerivaMLException(
                    f"default_schema '{default_schema}' is not in domain_schemas: {self.domain_schemas}"
                )
            self.default_schema = default_schema
        elif len(self.domain_schemas) == 1:
            # Single domain schema - use it as default
            self.default_schema = next(iter(self.domain_schemas))
        elif len(self.domain_schemas) == 0:
            # No domain schemas - default_schema will be None
            self.default_schema = None
        else:
            # Multiple domain schemas, no explicit default
            self.default_schema = None

    def is_system_schema(self, schema_name: str) -> bool:
        """Check if a schema is a system or ML schema.

        Args:
            schema_name: Name of the schema to check.

        Returns:
            True if the schema is a system or ML schema.
        """
        return is_system_schema(schema_name, self.ml_schema)

    def is_domain_schema(self, schema_name: str) -> bool:
        """Check if a schema is a domain schema.

        Args:
            schema_name: Name of the schema to check.

        Returns:
            True if the schema is a domain schema.
        """
        return schema_name in self.domain_schemas

    def _require_default_schema(self) -> str:
        """Get default schema, raising an error if not set.

        Returns:
            The default schema name.

        Raises:
            DerivaMLException: If default_schema is not set.
        """
        if self.default_schema is None:
            raise DerivaMLException(
                f"No default_schema set. With multiple domain schemas {self.domain_schemas}, "
                "you must either specify a default_schema when creating DerivaML or "
                "pass an explicit schema parameter to this method."
            )
        return self.default_schema

    def refresh_model(self) -> None:
        self.model = self.catalog.getCatalogModel()

    @property
    def schemas(self) -> dict[str, Schema]:
        return self.model.schemas

    @property
    def chaise_config(self) -> dict[str, Any]:
        """Return the chaise configuration."""
        return self.model.chaise_config

    def get_schema_description(self, include_system_columns: bool = False) -> dict[str, Any]:
        """Return a JSON description of the catalog schema structure.

        Provides a structured representation of the domain and ML schemas including
        tables, columns, foreign keys, and relationships. Useful for understanding
        the data model structure programmatically.

        Args:
            include_system_columns: If True, include RID, RCT, RMT, RCB, RMB columns.
                Default False to reduce output size.

        Returns:
            Dictionary with schema structure:
            {
                "domain_schemas": ["schema_name1", "schema_name2"],
                "default_schema": "schema_name1",
                "ml_schema": "deriva-ml",
                "schemas": {
                    "schema_name": {
                        "tables": {
                            "TableName": {
                                "comment": "description",
                                "is_vocabulary": bool,
                                "is_asset": bool,
                                "is_association": bool,
                                "columns": [...],
                                "foreign_keys": [...],
                                "features": [...]
                            }
                        }
                    }
                }
            }
        """
        system_columns = {"RID", "RCT", "RMT", "RCB", "RMB"}
        result = {
            "domain_schemas": sorted(self.domain_schemas),
            "default_schema": self.default_schema,
            "ml_schema": self.ml_schema,
            "schemas": {},
        }

        # Include all domain schemas and the ML schema
        for schema_name in [*self.domain_schemas, self.ml_schema]:
            schema = self.model.schemas.get(schema_name)
            if not schema:
                continue

            schema_info = {"tables": {}}

            for table_name, table in schema.tables.items():
                # Get columns
                columns = []
                for col in table.columns:
                    if not include_system_columns and col.name in system_columns:
                        continue
                    columns.append({
                        "name": col.name,
                        "type": str(col.type.typename),
                        "nullok": col.nullok,
                        "comment": col.comment or "",
                    })

                # Get foreign keys
                foreign_keys = []
                for fk in table.foreign_keys:
                    fk_cols = [c.name for c in fk.foreign_key_columns]
                    ref_cols = [c.name for c in fk.referenced_columns]
                    foreign_keys.append({
                        "columns": fk_cols,
                        "referenced_table": f"{fk.pk_table.schema.name}.{fk.pk_table.name}",
                        "referenced_columns": ref_cols,
                    })

                # Get features if this is a domain table
                features = []
                if self.is_domain_schema(schema_name):
                    try:
                        for f in self.find_features(table):
                            features.append({
                                "name": f.feature_name,
                                "feature_table": f.feature_table.name,
                            })
                    except Exception:
                        pass  # Table may not support features

                table_info = {
                    "comment": table.comment or "",
                    "is_vocabulary": self.is_vocabulary(table),
                    "is_asset": self.is_asset(table),
                    "is_association": bool(self.is_association(table)),
                    "columns": columns,
                    "foreign_keys": foreign_keys,
                }
                if features:
                    table_info["features"] = features

                schema_info["tables"][table_name] = table_info

            result["schemas"][schema_name] = schema_info

        return result

    def __getattr__(self, name: str) -> Any:
        # Called only if `name` is not found in Manager.  Delegate attributes to model class.
        return getattr(self.model, name)

    def name_to_table(self, table: TableInput) -> Table:
        """Return the table object corresponding to the given table name.

        Searches domain schemas first (in sorted order), then ML schema, then WWW.
        If the table name appears in more than one schema, returns the first match.

        Args:
          table: A ERMRest table object or a string that is the name of the table.

        Returns:
          Table object.

        Raises:
          DerivaMLException: If the table doesn't exist in any searchable schema.
        """
        if isinstance(table, Table):
            return table

        # Search domain schemas (sorted for deterministic order), then ML schema, then WWW
        search_order = [*sorted(self.domain_schemas), self.ml_schema, "WWW"]
        for sname in search_order:
            if sname not in self.model.schemas:
                continue
            s = self.model.schemas[sname]
            if table in s.tables:
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
        """Return a list of all controlled vocabulary tables in domain and ML schemas."""
        tables = []
        for schema_name in [*self.domain_schemas, self.ml_schema]:
            schema = self.model.schemas.get(schema_name)
            if schema:
                tables.extend(t for t in schema.tables.values() if self.is_vocabulary(t))
        return tables

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def find_features(self, table: TableInput | None = None) -> Iterable[Feature]:
        """List features in the catalog.

        If a table is specified, returns only features for that table.
        If no table is specified, returns all features across all tables in the catalog.

        Args:
            table: Optional table to find features for. If None, returns all features
                in the catalog.

        Returns:
            An iterable of Feature instances describing the features.
        """

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

        def find_table_features(t: Table) -> list[Feature]:
            """Find all features for a single table."""
            return [
                Feature(a, self) for a in t.find_associations(min_arity=3, max_arity=3, pure=False) if is_feature(a)
            ]

        if table is not None:
            # Find features for a specific table
            return find_table_features(self.name_to_table(table))
        else:
            # Find all features across all domain and ML schema tables
            features: list[Feature] = []
            for schema_name in [*self.domain_schemas, self.ml_schema]:
                schema = self.model.schemas.get(schema_name)
                if schema:
                    for t in schema.tables.values():
                        features.extend(find_table_features(t))
            return features

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

    def is_dataset_rid(self, rid: RID, deleted: bool = False) -> bool:
        """Check if a given RID is a dataset RID."""
        try:
            rid_info = self.model.catalog.resolve_rid(rid, self.model)
        except KeyError as _e:
            raise DerivaMLException(f"Invalid RID {rid}")
        if rid_info.table.name != "Dataset":
            return False
        elif deleted:
            # Got a dataset rid. Now check to see if its deleted or not.
            return True
        else:
            return not list(rid_info.datapath.entities().fetch())[0]["Deleted"]

    def list_dataset_element_types(self) -> list[Table]:
        """
        Lists the data types of elements contained within a dataset.

        This method analyzes the dataset and identifies the data types for all
        elements within it. It is useful for understanding the structure and
        content of the dataset and allows for better manipulation and usage of its
        data.

        Returns:
            list[str]: A list of strings where each string represents a data type
            of an element found in the dataset.

        """

        dataset_table = self.name_to_table("Dataset")

        def is_domain_or_dataset_table(table: Table) -> bool:
            return self.is_domain_schema(table.schema.name) or table.name == dataset_table.name

        return [t for a in dataset_table.find_associations() if is_domain_or_dataset_table(t := a.other_fkeys.pop().pk_table)]

    def _prepare_wide_table(
        self, dataset, dataset_rid: RID, include_tables: list[str]
    ) -> tuple[dict[str, Any], list[tuple]]:
        """
        Generates details of a wide table from the model

        Args:
            include_tables (list[str] | None): List of table names to include in the denormalized dataset. If None,
                all tables from the dataset will be included.

        Returns:
            str: SQL query string that represents the process of denormalization.
        """

        # Skip over tables that we don't want to include in the denormalized dataset.
        # Also, strip off the Dataset/Dataset_X part of the path so we don't include dataset columns in the denormalized
        # table.
        include_tables = set(include_tables)
        for t in include_tables:
            # Check to make sure the table is in the catalog.
            _ = self.name_to_table(t)

        table_paths = [
            path
            for path in self._schema_to_paths()
            if path[-1].name in include_tables and include_tables.intersection({p.name for p in path})
        ]
        paths_by_element = defaultdict(list)
        for p in table_paths:
            paths_by_element[p[2].name].append(p)

        skip_columns = {"RCT", "RMT", "RCB", "RMB"}
        element_tables = {}
        for element_table, paths in paths_by_element.items():
            graph = {}
            for path in paths:
                for left, right in zip(path[0:], path[1:]):
                    graph.setdefault(left.name, set()).add(right.name)

            # New lets remove any cycles that we may have in the graph.
            # We will use a topological sort to find the order in which we need to join the tables.
            # If we find a cycle, we will remove the table from the graph and splice in an additional ON clause.
            # We will then repeat the process until there are no cycles.
            graph_has_cycles = True
            element_join_tables = []
            element_join_conditions = {}
            while graph_has_cycles:
                try:
                    ts = TopologicalSorter(graph)
                    element_join_tables = list(reversed(list(ts.static_order())))
                    graph_has_cycles = False
                except CycleError as e:
                    cycle_nodes = e.args[1]
                    if len(cycle_nodes) > 3:
                        raise DerivaMLException(f"Unexpected cycle found when normalizing dataset {cycle_nodes}")
                    # Remove cycle from graph and splice in additional ON constraint.
                    graph[cycle_nodes[1]].remove(cycle_nodes[0])

            # The Dataset_Version table is a special case as it points to dataset and dataset to version.
            if "Dataset_Version" in element_join_tables:
                element_join_tables.remove("Dataset_Version")

            for path in paths:
                for left, right in zip(path[0:], path[1:]):
                    if right.name == "Dataset_Version":
                        # The Dataset_Version table is a special case as it points to dataset and dataset to version.
                        continue
                    if element_join_tables.index(right.name) < element_join_tables.index(left.name):
                        continue
                    table_relationship = self._table_relationship(left, right)
                    element_join_conditions.setdefault(right.name, set()).add(
                        (table_relationship[0], table_relationship[1])
                    )
            element_tables[element_table] = (element_join_tables, element_join_conditions)
        # Get the list of columns that will appear in the final denormalized dataset.
        denormalized_columns = [
            (table_name, c.name)
            for table_name in include_tables
            if not self.is_association(table_name)  # Don't include association columns in the denormalized view.'
            for c in self.name_to_table(table_name).columns
            if (not include_tables or table_name in include_tables) and (c.name not in skip_columns)
        ]
        return element_tables, denormalized_columns

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
            raise DerivaMLException(
                f"Ambiguous linkage between {table1.name} and {table2.name}: {[(r[0].name, r[1].name) for r in relationships]}"
            )
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
            # Valid schemas for traversal: all domain schemas + ML schema
            valid_schemas = self.domain_schemas | {self.ml_schema}
            arc_list = [fk.pk_table for fk in table.foreign_keys] + [fk.table for fk in table.referenced_by]
            arc_list = [t for t in arc_list if t.schema.name in valid_schemas]
            domain_tables = [t for t in arc_list if self.is_domain_schema(t.schema.name)]
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
            #        if child.name in {"Dataset_Execution", "Dataset_Dataset", "Execution"}:
            if child.name in {"Dataset_Dataset", "Execution"}:
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

    def create_table(self, table_def: TableDefinition, schema: str | None = None) -> Table:
        """Create a new table from TableDefinition.

        Args:
            table_def: Table definition (dataclass or dict).
            schema: Schema to create the table in. If None, uses default_schema.

        Returns:
            The newly created Table.

        Raises:
            DerivaMLException: If no schema specified and default_schema is not set.

        Note: @validate_call removed because TableDefinition is now a dataclass from
        deriva.core.typed and Pydantic validation doesn't work well with dataclass fields.
        """
        schema = schema or self._require_default_schema()
        # Handle both TableDefinition (dataclass with to_dict) and plain dicts
        table_dict = table_def.to_dict() if hasattr(table_def, 'to_dict') else table_def
        return self.model.schemas[schema].create_table(table_dict)
