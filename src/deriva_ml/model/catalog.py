"""
Model management for Deriva ML catalogs.

This module provides the DerivaModel class which augments the standard Deriva model class with
ML-specific functionality. It handles schema management, feature definitions, and asset tracking.
"""

from __future__ import annotations

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib
import logging

# Standard library imports
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from typing import Any, Callable, Final, Iterable, NewType, TypeAlias

_ermrest_catalog = importlib.import_module("deriva.core.ermrest_catalog")
_ermrest_model = importlib.import_module("deriva.core.ermrest_model")

ErmrestCatalog = _ermrest_catalog.ErmrestCatalog
Column = _ermrest_model.Column
FindAssociationResult = _ermrest_model.FindAssociationResult
Key = _ermrest_model.Key
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


@dataclass
class JoinNode:
    """A node in the join tree used by ``_prepare_wide_table``.

    The join tree is a rooted tree where each node represents a table that
    participates in the denormalized query.  The root is the element table
    (e.g., Image), and children are tables that should be JOINed to it by
    following FK relationships.

    Attributes:
        table: The ermrest ``Table`` object for this node.
        table_name: Human-readable name (``table.name``).
        join_type: ``"inner"`` or ``"left"`` -- LEFT JOIN is used when the
            FK column is nullable so that rows with NULL FK values are
            preserved.
        fk_columns: ``(fk_col, pk_col)`` pairs describing how this node
            joins to its parent.  ``None`` for the root node.
        is_association: If True, this table is needed for the JOIN chain
            but its columns are excluded from the output (e.g., M:N
            linking tables like ``ClinicalRecord_Observation``).
        children: Child nodes to join after this one.
    """

    table: Any  # ermrest Table
    table_name: str
    join_type: str = "inner"  # "inner" or "left"
    fk_columns: list[tuple] | None = None  # list[(fk_col, pk_col)]
    is_association: bool = False
    children: list["JoinNode"] = field(default_factory=list)

    def walk(self) -> list["JoinNode"]:
        """Return a pre-order traversal of the tree (self first, then children)."""
        result = [self]
        for child in self.children:
            result.extend(child.walk())
        return result

    def walk_edges(self) -> list[tuple["JoinNode", "JoinNode"]]:
        """Return (parent, child) pairs in pre-order traversal."""
        edges = []
        for child in self.children:
            edges.append((self, child))
            edges.extend(child.walk_edges())
        return edges


def denormalize_column_name(
    schema_name: str, table_name: str, column_name: str, multi_schema: bool
) -> str:
    """Build a prefixed column name for denormalized output.

    Uses dot notation to avoid ambiguity with column names that contain
    underscores (e.g., ``Acquisition_Date``).

    Args:
        schema_name: Schema the table belongs to.
        table_name: Table the column belongs to.
        column_name: Raw column name.
        multi_schema: If True, include schema prefix for disambiguation.

    Returns:
        Prefixed column name, e.g. ``Image.Filename`` or ``test-schema.Image.Filename``.
    """
    if multi_schema:
        return f"{schema_name}.{table_name}.{column_name}"
    return f"{table_name}.{column_name}"

logger = logging.getLogger(__name__)

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
        domain_schemas: str | set[str] | None = None,
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
            if isinstance(domain_schemas, str):
                domain_schemas = {domain_schemas}
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
                    except Exception as e:
                        logger.debug(f"Could not enumerate features for table {table.name}: {e}")

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

    def vocab_columns(self, table_name: TableInput) -> dict[str, str]:
        """Return mapping from canonical vocab column name to actual column name.

        Canonical names are TitleCase (Name, ID, URI, Description, Synonyms).
        Actual names reflect the table's schema — could be lowercase for
        FaceBase-style catalogs or TitleCase for DerivaML-native tables.

        Args:
            table_name: A table object or the name of the table.

        Returns:
            Dict mapping canonical name to actual column name in the table.
            E.g. ``{"Name": "name", "ID": "id", ...}`` for FaceBase tables
            or ``{"Name": "Name", "ID": "ID", ...}`` for DerivaML tables.
        """
        table = self.name_to_table(table_name)
        col_map = {c.name.upper(): c.name for c in table.columns}
        return {canon: col_map[canon.upper()] for canon in ("Name", "ID", "URI", "Description", "Synonyms")}

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
        """True if the specified table is a proper asset table.

        Delegates to Table.is_asset() from deriva-py which checks:
        - Required columns exist (URL, Filename, Length, MD5)
        - URL, Length, MD5 are NOT NULL
        - URL has the asset annotation

        Args:
            table_name: str | Table

        Returns:
            True if the specified table is a proper asset table.
        """
        table = self.name_to_table(table_name)
        return table.is_asset()

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

    def _build_join_tree(
        self,
        element_name: str,
        include_tables: set[str],
        all_paths: list[list[Table]],
    ) -> JoinNode:
        """Build a JoinTree rooted at *element_name* that reaches all *include_tables*.

        The algorithm:

        1. Collect all FK paths from `_schema_to_paths()` that start at the element
           table and end at a table in *include_tables*.
        2. For each target table, pick the SHORTEST sub-path from the element.
           If a longer path exists but ALL its intermediates are in *include_tables*,
           prefer it (user disambiguated).  If multiple equally-short paths exist
           and cannot be disambiguated, raise an ambiguity error.
        3. Merge the selected paths into a tree rooted at the element.
        4. Mark association tables (``is_association=True``) so their columns are
           excluded from output but they are still JOINed through.
        5. Set ``join_type="left"`` when the FK column is nullable.

        Args:
            element_name: The dataset element table (tree root), e.g. ``"Image"``.
            include_tables: Set of table names the user wants in the output.
            all_paths: All FK paths from ``_schema_to_paths()``.

        Returns:
            A ``JoinNode`` tree rooted at the element table.

        Raises:
            DerivaMLException: If ambiguous paths cannot be resolved.
        """
        element_table = self.name_to_table(element_name)

        # ── Step 1: collect sub-paths from element to each include_table ─────
        # Each "all_path" has the structure [Dataset, assoc, element, ..., endpoint].
        # We extract the sub-path starting from the element: [element, ..., endpoint].
        subpaths_by_target: dict[str, list[list[Table]]] = defaultdict(list)

        for path in all_paths:
            if len(path) < 3:
                continue
            if path[2].name != element_name:
                continue
            endpoint = path[-1].name
            if endpoint not in include_tables:
                continue
            # Sub-path from element onward
            sub = path[2:]  # [element, ..., endpoint]
            subpaths_by_target[endpoint].append(sub)

        # The element itself (self-path of length 1)
        if element_name in include_tables:
            subpaths_by_target.setdefault(element_name, []).append([element_table])

        # ── Step 2: for each target, pick the best path ──────────────────────
        selected_subpaths: dict[str, list[Table]] = {}

        for target, subpaths in subpaths_by_target.items():
            if target == element_name:
                # Self-path: no join needed
                selected_subpaths[target] = [element_table]
                continue

            # Deduplicate by table-name signature
            seen_sigs: set[tuple[str, ...]] = set()
            unique: list[list[Table]] = []
            for sp in subpaths:
                sig = tuple(t.name for t in sp)
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    unique.append(sp)

            if len(unique) == 1:
                selected_subpaths[target] = unique[0]
                continue

            # Multiple paths — disambiguate.
            # Intermediates are tables between element (sp[0]) and endpoint (sp[-1]).
            path_intermediates = [tuple(t.name for t in sp[1:-1]) for sp in unique]

            # If all have identical intermediates, no ambiguity
            if len(set(path_intermediates)) <= 1:
                selected_subpaths[target] = unique[0]
                continue

            # A path is "selected" if all its non-association intermediates are
            # in include_tables.  Association tables (M:N link tables) are
            # infrastructure that the user shouldn't need to name explicitly —
            # they are transparently included in the join chain.
            #
            # We detect association tables by checking if the Table object has
            # exactly 2 FKs (the definition of a pure association table).
            # This works regardless of model context (bag or catalog).
            def _is_likely_association(tbl: Table) -> bool:
                """Check if table is an association table (M:N link table).

                An association table has only system columns (RID, RCT, RMT,
                RCB, RMB) plus FK columns to the tables it connects.  ERMrest's
                built-in is_association() counts system FKs (RCB/RMB → ERMrest_Client),
                so we use our own check that ignores them.
                """
                system_cols = {'RID', 'RCT', 'RMT', 'RCB', 'RMB'}
                try:
                    cols = {c.name for c in tbl.columns}
                    fks = list(tbl.foreign_keys)
                    # Domain FKs: those NOT to system tables like ERMrest_Client
                    domain_fks = [
                        fk for fk in fks
                        if fk.pk_table.name not in ('ERMrest_Client', 'ERMrest_Group')
                    ]
                    # FK column names
                    fk_col_names = set()
                    for fk in domain_fks:
                        for col in fk.columns:
                            fk_col_names.add(col.name if hasattr(col, 'name') else str(col))
                    # Non-system, non-FK columns
                    user_cols = cols - system_cols - fk_col_names
                    # Association = exactly 2 domain FKs and no other user columns
                    return len(domain_fks) == 2 and len(user_cols) == 0
                except Exception:
                    return False

            def _intermediates_covered(sp: list[Table], ints: tuple[str, ...]) -> bool:
                sp_tables = {t.name: t for t in sp}
                for t in ints:
                    if t in include_tables:
                        continue
                    tbl = sp_tables.get(t)
                    if tbl is not None and _is_likely_association(tbl):
                        continue  # transparent — doesn't need to be in include_tables
                    return False
                return True

            fully_covered = [
                (sp, ints)
                for sp, ints in zip(unique, path_intermediates)
                if _intermediates_covered(sp, ints)
            ]

            if len(fully_covered) == 1:
                sp, ints = fully_covered[0]
                if len(ints) > 0:
                    # User explicitly included intermediates
                    selected_subpaths[target] = sp
                    continue
                # Direct path (no intermediates) — check if there are indirect paths
                has_indirect = any(len(i) > 0 for i in path_intermediates)
                if not has_indirect:
                    selected_subpaths[target] = sp
                    continue
                # Direct FK alongside indirect — prefer direct (shortest)
                selected_subpaths[target] = sp
                continue

            if len(fully_covered) > 1:
                # Multiple fully-covered paths
                has_explicit = [(sp, ints) for sp, ints in fully_covered if len(ints) > 0]
                if len(has_explicit) == 1:
                    selected_subpaths[target] = has_explicit[0][0]
                    continue
                elif len(has_explicit) == 0:
                    # All direct paths — pick shortest
                    shortest = min(fully_covered, key=lambda x: len(x[0]))
                    selected_subpaths[target] = shortest[0]
                    continue
                else:
                    # Multiple explicit — prefer longest (most specific)
                    max_ints = max(len(ints) for _, ints in has_explicit)
                    longest = [sp for sp, ints in has_explicit if len(ints) == max_ints]
                    if len(longest) == 1:
                        selected_subpaths[target] = longest[0]
                        continue

            if len(fully_covered) == 0:
                # No path is fully covered.  Check if direct path exists.
                direct = [sp for sp, ints in zip(unique, path_intermediates) if len(ints) == 0]
                if len(direct) == 1:
                    selected_subpaths[target] = direct[0]
                    continue

            # Ambiguity error
            path_descriptions = []
            all_ints: set[str] = set()
            for sp, ints in zip(unique, path_intermediates):
                names = [t.name for t in sp]
                path_descriptions.append(" → ".join(names))
                all_ints.update(ints)

            suggestion_tables = all_ints - include_tables
            suggestion = ""
            if suggestion_tables:
                suggestion = (
                    f"\nInclude an intermediate table to disambiguate "
                    f"(e.g., add {', '.join(sorted(suggestion_tables))} to include_tables)."
                )

            raise DerivaMLException(
                f"Ambiguous path between {element_name} and {target}: "
                f"found {len(unique)} FK paths:\n"
                + "\n".join(f"  {d}" for d in path_descriptions)
                + suggestion
            )

        # ── Step 3: merge selected paths into a tree ─────────────────────────
        # Build the tree by inserting each selected sub-path into the tree.
        root = JoinNode(
            table=element_table,
            table_name=element_name,
            join_type="inner",
            fk_columns=None,
            is_association=bool(self.is_association(element_name)),
            children=[],
        )

        # Map table_name -> JoinNode for quick lookup during tree building
        node_map: dict[str, JoinNode] = {element_name: root}

        for target, subpath in selected_subpaths.items():
            if target == element_name:
                continue
            # subpath = [element, ..intermediate.., target]
            # Walk the subpath, creating nodes as needed
            for i in range(1, len(subpath)):
                child_table = subpath[i]
                child_name = child_table.name
                parent_table = subpath[i - 1]
                parent_name = parent_table.name

                if child_name in node_map:
                    continue  # Already in tree

                # Get FK column pairs
                col_pairs = self._table_relationship(parent_table, child_table)

                # Determine join type: LEFT for nullable FK columns
                join_type = "inner"
                for fk_col, pk_col in col_pairs:
                    if fk_col.nullok:
                        join_type = "left"
                        break

                node = JoinNode(
                    table=child_table,
                    table_name=child_name,
                    join_type=join_type,
                    fk_columns=col_pairs,
                    is_association=bool(self.is_association(child_name)),
                    children=[],
                )
                node_map[child_name] = node
                # Attach to parent
                if parent_name in node_map:
                    node_map[parent_name].children.append(node)
                else:
                    # Parent not yet in tree — this shouldn't happen since we
                    # process paths from element outward, but handle gracefully
                    logger.warning(
                        f"Parent {parent_name} not in tree when adding {child_name}"
                    )

        return root

    def _prepare_wide_table(
        self, dataset, dataset_rid: RID, include_tables: list[str]
    ) -> tuple[dict[str, Any], list[tuple], bool]:
        """Generate a join plan for denormalizing a dataset into a wide table.

        Uses a **JoinTree** approach that preserves path-specific structure:

        1. **Path discovery** -- ``_schema_to_paths()`` discovers all FK paths
           from Dataset through the schema.
        2. **Path filtering & deduplication** -- keep only paths relevant to
           *include_tables*, dedup duplicate association table routes.
        3. **JoinTree construction** -- for each element type, build a tree
           rooted at the element.  Each node is a table to JOIN; association
           tables are in the tree (for JOIN) but excluded from output columns.
           Nullable FK columns produce LEFT JOINs.
        4. **Flatten to legacy format** -- convert the tree to the
           ``(path, join_conditions, join_types)`` tuple expected by
           ``_denormalize()`` and ``_denormalize_datapath()``.

        Args:
            dataset: A DatasetLike object (DatasetBag or Dataset).
            dataset_rid: RID of the dataset.
            include_tables: List of table names to include in the output.

        Returns:
            ``(element_tables, denormalized_columns, multi_schema)`` where:

            - **element_tables** -- ``dict[str, (path, join_conditions, join_types)]``
              keyed by element table name.
              *path* is a list of table name strings in JOIN order (pre-order walk
              of the JoinTree, starting with "Dataset").
              *join_conditions* maps ``table_name -> set[(fk_col, pk_col)]``.
              *join_types* maps ``table_name -> "inner" | "left"``.
            - **denormalized_columns** -- list of
              ``(schema_name, table_name, column_name, type_name)`` for the output.
            - **multi_schema** -- True if output spans multiple domain schemas.
        """
        include_tables_set = set(include_tables)
        for t in include_tables_set:
            _ = self.name_to_table(t)  # validate existence

        # ── Phase 1: path discovery ──────────────────────────────────────────
        all_paths = self._schema_to_paths()

        # Filter paths: must end at a table in include_tables AND
        # have at least one table in include_tables along the path.
        table_paths = [
            path
            for path in all_paths
            if path[-1].name in include_tables_set
            and include_tables_set.intersection({p.name for p in path})
        ]

        # ── Phase 1b: deduplicate association table routes ───────────────────
        # In some catalogs (e.g., eye-ai), both Image_Dataset and Dataset_Image
        # exist.  Keep only one route per (element, endpoint) via different
        # association tables (path[1]).
        deduplicated_paths: list[list[Table]] = []
        seen_element_endpoint: dict[tuple[str, str], tuple[list[Table], Table]] = {}

        def _is_standard_assoc(assoc_name: str, element_name: str) -> bool:
            """Check if assoc table matches the Dataset_{Element} naming pattern."""
            return assoc_name == f"Dataset_{element_name}"

        for path in table_paths:
            if len(path) < 3:
                deduplicated_paths.append(path)
                continue
            assoc_table = path[1]
            element = path[2]
            endpoint = path[-1]
            key = (element.name, endpoint.name)

            if key not in seen_element_endpoint:
                seen_element_endpoint[key] = (path, assoc_table)
                deduplicated_paths.append(path)
            else:
                existing_path, existing_assoc = seen_element_endpoint[key]
                if existing_assoc.name != assoc_table.name:
                    # Duplicate route via different association table.
                    # Prefer the standard Dataset_{Element} pattern over legacy.
                    if _is_standard_assoc(assoc_table.name, element.name) and not _is_standard_assoc(existing_assoc.name, element.name):
                        # Replace existing with standard pattern
                        deduplicated_paths = [p for p in deduplicated_paths if not (len(p) >= 3 and (p[2].name, p[-1].name) == key)]
                        seen_element_endpoint[key] = (path, assoc_table)
                        deduplicated_paths.append(path)
                    # else: keep existing (either it's standard or both are non-standard)
                else:
                    deduplicated_paths.append(path)

        table_paths = deduplicated_paths

        # ── Phase 1c: group by element, filter to elements in include_tables ─
        paths_by_element: dict[str, list[list[Table]]] = defaultdict(list)
        for p in table_paths:
            if len(p) >= 3:
                paths_by_element[p[2].name].append(p)

        paths_by_element = {
            elem: paths
            for elem, paths in paths_by_element.items()
            if elem in include_tables_set
        }

        # ── Phase 2: build JoinTree per element ──────────────────────────────
        skip_columns = {"RCT", "RMT", "RCB", "RMB"}
        element_tables: dict[str, tuple[list[str], dict[str, set], dict[str, str]]] = {}

        for element_name, paths in paths_by_element.items():
            tree = self._build_join_tree(element_name, include_tables_set, table_paths)

            # ── Phase 3: flatten JoinTree to legacy format ───────────────────
            # Pre-order walk gives us the correct JOIN order.
            # We prepend "Dataset" and the association table that connects
            # Dataset to the element (taken from paths[0][0:3]).

            # Find the Dataset -> assoc -> element prefix from the first path
            if paths and len(paths[0]) >= 3:
                dataset_name = paths[0][0].name  # "Dataset"
                assoc_name = paths[0][1].name    # e.g. "Dataset_Image"
            else:
                dataset_name = "Dataset"
                assoc_name = None

            # Walk the tree to get the join order (element -> children)
            tree_nodes = tree.walk()

            # Build the legacy path: [Dataset, assoc, element, ...tree children...]
            path_names: list[str] = [dataset_name]
            if assoc_name:
                path_names.append(assoc_name)

            # Add tree nodes (element first, then its subtree in pre-order)
            for node in tree_nodes:
                if node.table_name not in path_names:
                    path_names.append(node.table_name)

            # Build join conditions and join types from the tree edges
            join_conditions: dict[str, set[tuple]] = {}
            join_types: dict[str, str] = {}

            # First, add the Dataset -> assoc and assoc -> element conditions
            if assoc_name:
                dataset_table = self.name_to_table(dataset_name)
                assoc_table_obj = self.name_to_table(assoc_name)
                try:
                    col_pairs = self._table_relationship(dataset_table, assoc_table_obj)
                    join_conditions[assoc_name] = set(col_pairs)
                    join_types[assoc_name] = "inner"
                except DerivaMLException:
                    pass

                try:
                    col_pairs = self._table_relationship(assoc_table_obj, tree.table)
                    join_conditions[tree.table_name] = set(col_pairs)
                    join_types[tree.table_name] = "inner"
                except DerivaMLException:
                    pass

            # Add conditions from the JoinTree edges
            for parent_node, child_node in tree.walk_edges():
                if child_node.fk_columns:
                    join_conditions[child_node.table_name] = set(child_node.fk_columns)
                    join_types[child_node.table_name] = child_node.join_type

            element_tables[element_name] = (path_names, join_conditions, join_types)

        # ── Phase 4: build denormalized column list ──────────────────────────
        denormalized_columns = []
        for table_name in include_tables_set:
            if self.is_association(table_name):
                continue
            table = self.name_to_table(table_name)
            for c in table.columns:
                if c.name not in skip_columns:
                    denormalized_columns.append(
                        (table.schema.name, table_name, c.name, c.type.typename)
                    )

        output_schemas = {s for s, _, _, _ in denormalized_columns if self.is_domain_schema(s)}
        multi_schema = len(output_schemas) > 1

        return element_tables, denormalized_columns, multi_schema

    def _table_relationship(
        self,
        table1: TableInput,
        table2: TableInput,
    ) -> list[tuple[Column, Column]]:
        """Return column pairs used to relate two tables.

        For simple FKs, returns a single-element list: [(fk_col, pk_col)].
        For composite FKs, returns multiple pairs: [(fk_col1, pk_col1), (fk_col2, pk_col2)].

        Each FK constraint counts as one relationship (even if composite),
        so ambiguity is detected when multiple separate FK constraints exist
        between the same two tables.
        """
        table1 = self.name_to_table(table1)
        table2 = self.name_to_table(table2)
        # Each FK constraint produces a list of (fk_col, pk_col) pairs
        relationships: list[list[tuple[Column, Column]]] = []
        for fk in table1.foreign_keys:
            if fk.pk_table == table2:
                pairs = list(zip(fk.foreign_key_columns, fk.referenced_columns))
                relationships.append(pairs)
        for fk in table1.referenced_by:
            if fk.table == table2:
                pairs = list(zip(fk.referenced_columns, fk.foreign_key_columns))
                relationships.append(pairs)

        if len(relationships) == 0:
            raise DerivaMLException(
                f"No FK relationship found between {table1.name} and {table2.name}. "
                f"These tables may not be directly connected. Check your include_tables list."
            )
        if len(relationships) > 1:
            path_descriptions = []
            for col_pairs in relationships:
                desc = ", ".join(
                    f"{fk_col.table.name}.{fk_col.name} → {pk_col.table.name}.{pk_col.name}"
                    for fk_col, pk_col in col_pairs
                )
                path_descriptions.append(f"  {desc}")
            raise DerivaMLException(
                f"Ambiguous linkage between {table1.name} and {table2.name}: "
                f"found {len(relationships)} FK relationships:\n"
                + "\n".join(path_descriptions)
            )
        return relationships[0]

    # Default tables to skip during FK path traversal.
    # These are ML schema tables that create unwanted traversal branches:
    # - Dataset_Dataset: nested dataset self-reference (handled separately)
    # - Execution: execution tracking (not useful for data traversal)
    _DEFAULT_SKIP_TABLES = frozenset({"Dataset_Dataset", "Execution"})

    def _schema_to_paths(
        self,
        root: Table | None = None,
        path: list[Table] | None = None,
        exclude_tables: set[str] | None = None,
        skip_tables: frozenset[str] | None = None,
        max_depth: int | None = None,
    ) -> list[list[Table]]:
        """Discover all FK paths through the schema graph via depth-first traversal.

        This is the shared foundation for both bag export (catalog_graph._collect_paths)
        and denormalization (_prepare_wide_table). Changes here affect both systems.

        Traversal rules:
        - Follows both outbound FKs (table.foreign_keys) and inbound FKs (table.referenced_by)
        - Only traverses tables in valid schemas (domain + ML)
        - Terminates at vocabulary tables (paths go INTO vocabs but not OUT)
        - Skips tables in exclude_tables and skip_tables
        - Detects and skips cycles (same table appearing twice in a path)
        - Prevents dataset element loopback (traversing back to Dataset via element associations)
        - When multiple FKs exist between the same two domain tables, deduplicates
          arcs to avoid redundant paths (keeps one arc per target table)

        Args:
            root: Starting table. Defaults to the Dataset table in the ML schema.
            path: Current path being built (used during recursion).
            exclude_tables: Caller-specified table names to skip. These tables and
                all paths through them are pruned from the result.
            skip_tables: Infrastructure table names to skip. Defaults to
                _DEFAULT_SKIP_TABLES (Dataset_Dataset, Execution). Override to
                customize which ML schema tables are excluded from traversal.
            max_depth: Maximum path length (number of tables). None = unlimited.
                Use to protect against pathological schemas with deep chains.

        Returns:
            List of paths, where each path is a list of Table objects starting
            from root. Every prefix of a path is also included (e.g., if
            [Dataset, A, B, C] is a path, then [Dataset], [Dataset, A], and
            [Dataset, A, B] are also in the result).
        """
        exclude_tables = exclude_tables or set()
        skip_tables = skip_tables if skip_tables is not None else self._DEFAULT_SKIP_TABLES

        root = root or self.model.schemas[self.ml_schema].tables["Dataset"]
        path = path.copy() if path else []
        parent = path[-1] if path else None  # Table we are coming from.
        path.append(root)
        paths = [path]

        # Depth limit check
        if max_depth is not None and len(path) >= max_depth:
            return paths

        def find_arcs(table: Table) -> set[Table]:
            """Return reachable tables via FK arcs, deduplicating multi-FK targets."""
            valid_schemas = self.domain_schemas | {self.ml_schema}
            arc_list = (
                [fk.pk_table for fk in table.foreign_keys]
                + [fk.table for fk in table.referenced_by]
            )
            arc_list = [t for t in arc_list if t.schema.name in valid_schemas]
            # Deduplicate: when multiple FKs point to the same target table,
            # keep only one arc. This prevents redundant path branching.
            # Downstream code (_prepare_wide_table, _table_relationship) handles
            # the specific FK selection and ambiguity detection.
            seen = set()
            deduped = []
            for t in arc_list:
                if t not in seen:
                    seen.add(t)
                    deduped.append(t)
            return set(deduped)

        def is_nested_dataset_loopback(n1: Table, n2: Table) -> bool:
            """Check if traversal would loop back to Dataset via an element association.

            Prevents: Subject -> Dataset_Subject -> Dataset (looping back to root).
            Allows: Dataset -> Dataset_Subject -> Subject (the intended direction).
            """
            dataset_table = self.model.schemas[self.ml_schema].tables["Dataset"]
            assoc_table = [a for a in dataset_table.find_associations() if a.table == n2]
            return len(assoc_table) == 1 and n1 != dataset_table

        # Vocabulary tables are terminal — traverse INTO but not OUT.
        if self.is_vocabulary(root):
            return paths

        for child in find_arcs(root):
            if child.name in skip_tables:
                continue
            if child.name in exclude_tables:
                continue
            if child == parent:
                # Don't loop back to immediate parent via referenced_by
                continue
            if is_nested_dataset_loopback(root, child):
                continue
            if child in path:
                # Cycle detected — skip to avoid infinite recursion.
                logger.warning(
                    f"Cycle in schema path: {child.name} "
                    f"path:{[p.name for p in path]}, skipping"
                )
                continue

            paths.extend(
                self._schema_to_paths(child, path, exclude_tables, skip_tables, max_depth)
            )
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

    def _define_association(
        self,
        associates: list,
        metadata: list | None = None,
        table_name: str | None = None,
        comment: str | None = None,
        **kwargs,
    ) -> dict:
        """Build an association table definition with vocab-aware key selection.

        Wraps Table.define_association to ensure non-vocabulary tables use RID
        as their foreign key target. The default key search heuristic in
        define_association prefers Name/ID keys over RID, which is correct for
        vocabulary tables (FK to human-readable Name) but wrong for domain
        tables that happen to have non-nullable Name or ID keys (e.g., tables
        in cloned catalogs like FaceBase).

        Args:
            associates: Reference targets being associated (Table, Key, or tuples).
            metadata: Additional metadata fields and/or reference targets.
            table_name: Name for the association table.
            comment: Comment for the association table.
            **kwargs: Additional arguments passed to Table.define_association.

        Returns:
            Table definition dict suitable for create_table.
        """
        metadata = metadata or []

        def _resolve_key(ref):
            """Convert non-vocabulary Table references to their RID Key."""
            if isinstance(ref, tuple):
                # (name, Table) or (name, nullok, Table) — resolve the Table element
                items = list(ref)
                table_obj = items[-1]
                if isinstance(table_obj, Table) and not table_obj.is_vocabulary():
                    items[-1] = table_obj.key_by_columns(["RID"])
                return tuple(items)
            elif isinstance(ref, Table) and not ref.is_vocabulary():
                return ref.key_by_columns(["RID"])
            return ref  # Key objects or vocabulary Tables pass through

        resolved_associates = [_resolve_key(a) for a in associates]
        resolved_metadata = [_resolve_key(m) for m in metadata]

        return Table.define_association(
            associates=resolved_associates,
            metadata=resolved_metadata,
            table_name=table_name,
            comment=comment,
            **kwargs,
        )
