"""SQLite-backed dataset access for downloaded BDBags.

This module provides the DatasetBag class, which allows querying and navigating
downloaded dataset bags using SQLite. When a dataset is downloaded from a Deriva
catalog, it is stored as a BDBag (Big Data Bag) containing:

- CSV files with table data
- Asset files (images, documents, etc.)
- A schema.json describing the catalog structure
- A fetch.txt manifest of referenced files

The DatasetBag class provides a read-only interface to this data, mirroring
the Dataset class API where possible. This allows code to work uniformly
with both live catalog datasets and downloaded bags.

Key concepts:
- DatasetBag wraps a single dataset within a downloaded bag
- A bag may contain multiple datasets (nested/hierarchical)
- All operations are read-only (bags are immutable snapshots)
- Queries use SQLite via SQLAlchemy ORM
- Table-level access (get_table_as_dict, lookup_term) is on the catalog (DerivaMLDatabase)

Typical usage:
    >>> # Download a dataset from a catalog
    >>> bag = ml.download_dataset_bag(dataset_spec)
    >>> # List dataset members by type
    >>> members = bag.list_dataset_members(recurse=True)
    >>> for image in members.get("Image", []):
    ...     print(image["Filename"])
"""

from __future__ import annotations

# Standard library imports
import logging
import shutil
from collections import defaultdict
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterable, Self, cast

import deriva.core.datapath as datapath

# Third-party imports
import pandas as pd

# Local imports
from deriva.core.ermrest_model import Table

# Deriva imports
from sqlalchemy import CompoundSelect, Engine, RowMapping, Select, and_, inspect, select, union
from sqlalchemy.orm import RelationshipProperty, Session
from sqlalchemy.orm.util import AliasedClass

from deriva_ml.core.definitions import RID
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.aux_classes import DatasetHistory, DatasetVersion
from deriva_ml.feature import Feature

if TYPE_CHECKING:
    from deriva_ml.model.deriva_ml_database import DerivaMLDatabase

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class DatasetBag:
    """Read-only interface to a downloaded dataset bag.

    DatasetBag manages access to a materialized BDBag (Big Data Bag) that contains
    a snapshot of dataset data from a Deriva catalog. It provides methods for:

    - Listing dataset members and their attributes
    - Navigating dataset relationships (parents, children)
    - Accessing feature values
    - Denormalizing data across related tables

    A bag may contain multiple datasets when nested datasets are involved. Each
    DatasetBag instance represents a single dataset within the bag - use
    list_dataset_children() to navigate to nested datasets.

    For catalog-level operations like querying arbitrary tables or looking up
    vocabulary terms, use the DerivaMLDatabase class instead.

    The class implements the DatasetLike protocol, providing the same read interface
    as the Dataset class. This allows code to work with both live catalogs and
    downloaded bags interchangeably.

    Attributes:
        dataset_rid (RID): The unique Resource Identifier for this dataset.
        dataset_types (list[str]): List of vocabulary terms describing the dataset type.
        description (str): Human-readable description of the dataset.
        execution_rid (RID | None): RID of the execution that created this dataset.
        model (DatabaseModel): The DatabaseModel providing SQLite access to bag data.
        engine (Engine): SQLAlchemy engine for database queries.
        metadata (MetaData): SQLAlchemy metadata with table definitions.

    Example:
        >>> # Download a dataset
        >>> bag = dataset.download_dataset_bag(version="1.0.0")
        >>> # List members by type
        >>> members = bag.list_dataset_members()
        >>> for image in members.get("Image", []):
        ...     print(f"File: {image['Filename']}")
        >>> # Navigate to nested datasets
        >>> for child in bag.list_dataset_children():
        ...     print(f"Nested: {child.dataset_rid}")
    """

    def __init__(
        self,
        catalog: "DerivaMLDatabase",
        dataset_rid: RID | None = None,
        dataset_types: str | list[str] | None = None,
        description: str = "",
        execution_rid: RID | None = None,
    ):
        """Initialize a DatasetBag instance for a dataset within a downloaded bag.

        This mirrors the Dataset class initialization pattern, where both classes
        take a catalog-like object as their first argument for consistency.

        Args:
            catalog: The DerivaMLDatabase instance providing access to the bag's data.
                This implements the DerivaMLCatalog protocol.
            dataset_rid: The RID of the dataset to wrap. If None, uses the primary
                dataset RID from the bag.
            dataset_types: One or more dataset type terms. Can be a single string
                or list of strings.
            description: Human-readable description of the dataset.
            execution_rid: RID of the execution that created this dataset. If None,
                will be looked up from the database.

        Raises:
            DerivaMLException: If no dataset_rid is provided and none can be
                determined from the bag, or if the RID doesn't exist in the bag.
        """
        # Store reference to the catalog and extract the underlying model
        self._catalog = catalog
        self.model = catalog.model
        self.engine = cast(Engine, self.model.engine)
        self.metadata = self.model.metadata

        # Use provided RID or fall back to the bag's primary dataset
        self.dataset_rid = dataset_rid or self.model.dataset_rid
        self.description = description
        self.execution_rid = execution_rid or self.model._get_dataset_execution(self.dataset_rid)

        # Normalize dataset_types to always be a list of strings for consistency
        # with the Dataset class interface
        if dataset_types is None:
            self.dataset_types: list[str] = []
        elif isinstance(dataset_types, str):
            self.dataset_types: list[str] = [dataset_types]
        else:
            self.dataset_types: list[str] = list(dataset_types)

        if not self.dataset_rid:
            raise DerivaMLException("No dataset RID provided")

        # Validate that this dataset exists in the bag
        self.model.rid_lookup(self.dataset_rid)

        # Cache the version and dataset table reference
        self._current_version = self.model.dataset_version(self.dataset_rid)
        self._dataset_table = self.model.dataset_table

    def __repr__(self) -> str:
        """Return a string representation of the DatasetBag for debugging."""
        return (f"<deriva_ml.DatasetBag object at {hex(id(self))}: rid='{self.dataset_rid}', "
                f"version='{self.current_version}', types={self.dataset_types}>")

    @property
    def current_version(self) -> DatasetVersion:
        """Get the version of the dataset at the time the bag was downloaded.

        For a DatasetBag, this is the version that was current when the bag was
        created. Unlike the live Dataset class, this value is immutable since
        bags are read-only snapshots.

        Returns:
            DatasetVersion: The semantic version (major.minor.patch) of this dataset.
        """
        return self._current_version

    def list_tables(self) -> list[str]:
        """List all tables available in the bag's SQLite database.

        Returns the fully-qualified names of all tables (e.g., "domain.Image",
        "deriva-ml.Dataset") that were exported in this bag.

        Returns:
            list[str]: Table names in "schema.table" format, sorted alphabetically.
        """
        return self.model.list_tables()

    def get_table_as_dict(self, table: str) -> Generator[dict[str, Any], None, None]:
        """Get table contents as dictionaries.

        Convenience method that delegates to the underlying catalog. This provides
        access to all rows in a table, not just those belonging to this dataset.
        For dataset-filtered results, use list_dataset_members() instead.

        Args:
            table: Name of the table to retrieve (e.g., "Subject", "Image").

        Yields:
            dict: Dictionary for each row in the table.

        Example:
            >>> for subject in bag.get_table_as_dict("Subject"):
            ...     print(subject["Name"])
        """
        return self._catalog.get_table_as_dict(table)

    @staticmethod
    def _find_relationship_attr(source, target):
        """Find the SQLAlchemy relationship attribute connecting two ORM classes.

        Searches for a relationship on `source` that points to `target`, which is
        needed to construct proper JOIN clauses in SQL queries.

        Args:
            source: Source ORM class or AliasedClass.
            target: Target ORM class or AliasedClass.

        Returns:
            InstrumentedAttribute: The relationship attribute on source pointing to target.

        Raises:
            LookupError: If no relationship exists between the two classes.

        Note:
            When multiple relationships exist, prefers MANYTOONE direction as this
            is typically the more natural join direction for denormalization.
        """
        src_mapper = inspect(source).mapper
        tgt_mapper = inspect(target).mapper

        # Collect all relationships on the source mapper that point to target
        candidates: list[RelationshipProperty] = [rel for rel in src_mapper.relationships if rel.mapper is tgt_mapper]

        if not candidates:
            raise LookupError(f"No relationship from {src_mapper.class_.__name__} → {tgt_mapper.class_.__name__}")

        # Prefer MANYTOONE when multiple paths exist (often best for joins)
        candidates.sort(key=lambda r: r.direction.name != "MANYTOONE")
        rel = candidates[0]

        # Return the bound attribute (handles AliasedClass properly)
        return getattr(source, rel.key) if isinstance(source, AliasedClass) else rel.class_attribute

    def _dataset_table_view(self, table: str) -> CompoundSelect[Any]:
        """Build a SQL query for all rows in a table that belong to this dataset.

        Creates a UNION of queries that traverse all possible paths from the
        Dataset table to the target table, filtering by this dataset's RID
        (and any nested dataset RIDs).

        This is necessary because table data may be linked to datasets through
        different relationship paths (e.g., Image might be linked directly to
        Dataset or through an intermediate Subject table).

        Args:
            table: Name of the table to query.

        Returns:
            CompoundSelect: A SQLAlchemy UNION query selecting all matching rows.
        """
        table_class = self.model.get_orm_class_by_name(table)
        dataset_table_class = self.model.get_orm_class_by_name(self._dataset_table.name)

        # Include this dataset and all nested datasets in the query
        dataset_rids = [self.dataset_rid] + [c.dataset_rid for c in self.list_dataset_children(recurse=True)]

        # Find all paths from Dataset to the target table
        paths = [[t.name for t in p] for p in self.model._schema_to_paths() if p[-1].name == table]

        # Build a SELECT query for each path and UNION them together
        sql_cmds = []
        for path in paths:
            path_sql = select(table_class)
            last_class = self.model.get_orm_class_by_name(path[0])
            # Join through each table in the path
            for t in path[1:]:
                t_class = self.model.get_orm_class_by_name(t)
                path_sql = path_sql.join(self._find_relationship_attr(last_class, t_class))
                last_class = t_class
            # Filter to only rows belonging to our dataset(s)
            path_sql = path_sql.where(dataset_table_class.RID.in_(dataset_rids))
            sql_cmds.append(path_sql)
        return union(*sql_cmds)

    def dataset_history(self) -> list[DatasetHistory]:
        """Retrieves the version history of a dataset.

        Returns a chronological list of dataset versions, including their version numbers,
        creation times, and associated metadata.

        Returns:
            list[DatasetHistory]: List of history entries, each containing:
                - dataset_version: Version number (major.minor.patch)
                - minid: Minimal Viable Identifier
                - snapshot: Catalog snapshot time
                - dataset_rid: Dataset Resource Identifier
                - version_rid: Version Resource Identifier
                - description: Version description
                - execution_rid: Associated execution RID

        Raises:
            DerivaMLException: If dataset_rid is not a valid dataset RID.

        Example:
            >>> history = ml.dataset_history("1-abc123")
            >>> for entry in history:
            ...     print(f"Version {entry.dataset_version}: {entry.description}")
        """
        # Query Dataset_Version table directly via the model
        return [
            DatasetHistory(
                dataset_version=DatasetVersion.parse(v["Version"]),
                minid=v["Minid"],
                snapshot=v["Snapshot"],
                dataset_rid=self.dataset_rid,
                version_rid=v["RID"],
                description=v["Description"],
                execution_rid=v["Execution"],
            )
            for v in self.model._get_table_contents("Dataset_Version")
            if v["Dataset"] == self.dataset_rid
        ]

    def list_dataset_members(
        self,
        recurse: bool = False,
        limit: int | None = None,
        _visited: set[RID] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return a list of entities associated with a specific dataset.

        Args:
            recurse: Whether to include members of nested datasets.
            limit: Maximum number of members to return per type. None for no limit.
            _visited: Internal parameter to track visited datasets and prevent infinite recursion.

        Returns:
            Dictionary mapping member types to lists of member records.
        """
        # Initialize visited set for recursion guard
        if _visited is None:
            _visited = set()

        # Prevent infinite recursion by checking if we've already visited this dataset
        if self.dataset_rid in _visited:
            return {}
        _visited.add(self.dataset_rid)

        # Look at each of the element types that might be in the _dataset_table and get the list of rid for them from
        # the appropriate association table.
        members = defaultdict(list)

        dataset_class = self.model.get_orm_class_for_table(self._dataset_table)
        for element_table in self.model.list_dataset_element_types():
            element_class = self.model.get_orm_class_for_table(element_table)

            assoc_class, dataset_rel, element_rel = self.model.get_orm_association_class(dataset_class, element_class)

            element_table = inspect(element_class).mapped_table
            if element_table.schema != self.model.domain_schema and element_table.name not in ["Dataset", "File"]:
                # Look at domain tables and nested datasets.
                continue

            # Get the names of the columns that we are going to need for linking
            with Session(self.engine) as session:
                # For Dataset_Dataset, use Nested_Dataset column to find nested datasets
                # (similar to how the live catalog does it in Dataset.list_dataset_members)
                if element_table.name == "Dataset":
                    sql_cmd = (
                        select(element_class)
                        .join(assoc_class, element_class.RID == assoc_class.__table__.c["Nested_Dataset"])
                        .where(self.dataset_rid == assoc_class.__table__.c["Dataset"])
                    )
                else:
                    # For other tables, use the original join via element_rel
                    sql_cmd = (
                        select(element_class)
                        .join(element_rel)
                        .where(self.dataset_rid == assoc_class.__table__.c["Dataset"])
                    )
                if limit is not None:
                    sql_cmd = sql_cmd.limit(limit)
                # Get back the list of ORM entities and convert them to dictionaries.
                element_entities = session.scalars(sql_cmd).all()
                element_rows = [{c.key: getattr(obj, c.key) for c in obj.__table__.columns} for obj in element_entities]
            members[element_table.name].extend(element_rows)
            if recurse and (element_table.name == self._dataset_table.name):
                # Get the members for all the nested datasets and add to the member list.
                nested_datasets = [d["RID"] for d in element_rows]
                for ds in nested_datasets:
                    nested_dataset = self._catalog.lookup_dataset(ds)
                    for k, v in nested_dataset.list_dataset_members(recurse=recurse, limit=limit, _visited=_visited).items():
                        members[k].extend(v)
        return dict(members)

    def find_features(self, table: str | Table) -> Iterable[Feature]:
        """Find features for a table.

        Args:
            table: The table to find features for.

        Returns:
            An iterable of Feature instances.
        """
        return self.model.find_features(table)

    def list_feature_values(self, table: Table | str, feature_name: str) -> datapath._ResultSet:
        """Return feature values for a table.

        Args:
            table: The table to get feature values for.
            feature_name: Name of the feature.

        Returns:
            Feature values.
        """
        feature = self.model.lookup_feature(table, feature_name)
        feature_class = self.model.get_orm_class_for_table(feature.feature_table)
        with Session(self.engine) as session:
            sql_cmd = select(feature_class)
            return cast(datapath._ResultSet, [row for row in session.execute(sql_cmd).mappings()])

    def list_dataset_element_types(self) -> Iterable[Table]:
        """List the types of elements that can be contained in datasets.

        This method analyzes the dataset and identifies the data types for all
        elements within it. It is useful for understanding the structure and
        content of the dataset and allows for better manipulation and usage of its
        data.

        Returns:
            list[str]: A list of strings where each string represents a data type
            of an element found in the dataset.

        """
        return self.model.list_dataset_element_types()

    def list_dataset_children(
        self, recurse: bool = False, _visited: set[RID] | None = None
    ) -> list[Self]:
        """Get nested datasets.

        Args:
            recurse: Whether to include children of children.
            _visited: Internal parameter to track visited datasets and prevent infinite recursion.

        Returns:
            List of child dataset bags.
        """
        # Initialize visited set for recursion guard
        if _visited is None:
            _visited = set()

        # Prevent infinite recursion by checking if we've already visited this dataset
        if self.dataset_rid in _visited:
            return []
        _visited.add(self.dataset_rid)

        ds_table = self.model.get_orm_class_by_name(f"{self.model.ml_schema}.Dataset")
        nds_table = self.model.get_orm_class_by_name(f"{self.model.ml_schema}.Dataset_Dataset")
        dv_table = self.model.get_orm_class_by_name(f"{self.model.ml_schema}.Dataset_Version")

        with Session(self.engine) as session:
            sql_cmd = (
                select(nds_table.Nested_Dataset, dv_table.Version)
                .join_from(ds_table, nds_table, onclause=ds_table.RID == nds_table.Nested_Dataset)
                .join_from(ds_table, dv_table, onclause=ds_table.Version == dv_table.RID)
                .where(nds_table.Dataset == self.dataset_rid)
            )
            nested = [DatasetBag(self._catalog, r[0]) for r in session.execute(sql_cmd).all()]

        result = copy(nested)
        if recurse:
            for child in nested:
                result.extend(child.list_dataset_children(recurse=recurse, _visited=_visited))
        return result

    def list_dataset_parents(
        self, recurse: bool = False, _visited: set[RID] | None = None
    ) -> list[Self]:
        """Given a dataset_table RID, return a list of RIDs of the parent datasets if this is included in a
        nested dataset.

        Args:
            recurse: If True, recursively return all ancestor datasets.
            _visited: Internal parameter to track visited datasets and prevent infinite recursion.

        Returns:
            List of parent dataset bags.
        """
        # Initialize visited set for recursion guard
        if _visited is None:
            _visited = set()

        # Prevent infinite recursion by checking if we've already visited this dataset
        if self.dataset_rid in _visited:
            return []
        _visited.add(self.dataset_rid)

        nds_table = self.model.get_orm_class_by_name(f"{self.model.ml_schema}.Dataset_Dataset")

        with Session(self.engine) as session:
            sql_cmd = select(nds_table.Dataset).where(nds_table.Nested_Dataset == self.dataset_rid)
            parents = [DatasetBag(self._catalog, r[0]) for r in session.execute(sql_cmd).all()]

        if recurse:
            for parent in parents.copy():
                parents.extend(parent.list_dataset_parents(recurse=True, _visited=_visited))
        return parents

    def _denormalize(self, include_tables: list[str]) -> Select:
        """Build a SQL query that joins multiple tables into a denormalized view.

        This method creates a "wide table" by joining related tables together,
        producing a single query that returns columns from all specified tables.
        This is useful for machine learning pipelines that need flat data.

        The method:
        1. Analyzes the schema to find join paths between tables
        2. Determines the correct join order based on foreign key relationships
        3. Builds SELECT statements with properly aliased columns
        4. Creates a UNION if multiple paths exist to the same tables

        Args:
            include_tables: List of table names to include in the output. Additional
                tables may be included if they're needed to join the requested tables.

        Returns:
            Select: A SQLAlchemy query that produces the denormalized result.

        Note:
            Column names in the result are prefixed with the table name to avoid
            collisions (e.g., "Image.Filename", "Subject.RID").
        """
        # Skip over tables that we don't want to include in the denormalized dataset.
        # Also, strip off the Dataset/Dataset_X part of the path so we don't include dataset columns in the denormalized
        # table.

        def find_relationship(table, join_condition):
            side1 = (join_condition[0].table.name, join_condition[0].name)
            side2 = (join_condition[1].table.name, join_condition[1].name)

            for relationship in inspect(table).relationships:
                local_columns = list(relationship.local_columns)[0].table.name, list(relationship.local_columns)[0].name
                remote_side = list(relationship.remote_side)[0].table.name, list(relationship.remote_side)[0].name
                if local_columns == side1 and remote_side == side2 or local_columns == side2 and remote_side == side1:
                    return relationship
            return None

        join_tables, denormalized_columns = self.model._prepare_wide_table(self, self.dataset_rid, include_tables)

        denormalized_columns = [
            self.model.get_orm_class_by_name(table_name)
            .__table__.columns[column_name]
            .label(f"{table_name}.{column_name}")
            for table_name, column_name in denormalized_columns
        ]
        sql_statements = []
        for key, (path, join_conditions) in join_tables.items():
            sql_statement = select(*denormalized_columns).select_from(
                self.model.get_orm_class_for_table(self._dataset_table)
            )
            for table_name in path[1:]:  # Skip over dataset table
                table_class = self.model.get_orm_class_by_name(table_name)
                on_clause = [
                    getattr(table_class, r.key)
                    for on_condition in join_conditions[table_name]
                    if (r := find_relationship(table_class, on_condition))
                ]
                sql_statement = sql_statement.join(table_class, onclause=and_(*on_clause))
            dataset_rid_list = [self.dataset_rid] + [c.dataset_rid for c in self.list_dataset_children(recurse=True)]
            dataset_class = self.model.get_orm_class_by_name(self._dataset_table.name)
            sql_statement = sql_statement.where(dataset_class.RID.in_(dataset_rid_list))
            sql_statements.append(sql_statement)
        return union(*sql_statements)

    def denormalize_as_dataframe(self, include_tables: list[str]) -> pd.DataFrame:
        """Denormalize the dataset into a single wide DataFrame.

        Joins related tables together to produce a "flat" view of the data,
        with columns from multiple tables combined into a single DataFrame.
        This is particularly useful for machine learning workflows that require
        tabular data with all features in one table.

        Column names are prefixed with the source table name to avoid collisions
        (e.g., "Image.Filename", "Subject.RID").

        Args:
            include_tables: List of table names to include in the output. Tables
                are joined based on their foreign key relationships.

        Returns:
            pd.DataFrame: Wide table with columns from all included tables.

        Example:
            >>> df = bag.denormalize_as_dataframe(["Image", "Diagnosis"])
            >>> # df has columns like "Image.Filename", "Diagnosis.Name", etc.
        """
        return pd.read_sql(self._denormalize(include_tables=include_tables), self.engine)

    def denormalize_as_dict(self, include_tables: list[str]) -> Generator[RowMapping, None, None]:
        """Denormalize the dataset and yield rows as dictionaries.

        Like denormalize_as_dataframe(), but returns a generator of dictionaries
        instead of a DataFrame. Useful for processing large datasets without
        loading everything into memory.

        Column names are prefixed with the source table name to avoid collisions
        (e.g., "Image.Filename", "Subject.RID").

        Args:
            include_tables: List of table names to include in the output.

        Yields:
            RowMapping: Dictionary-like objects with "table.column" keys.

        Example:
            >>> for row in bag.denormalize_as_dict(["Image", "Diagnosis"]):
            ...     print(row["Image.Filename"], row["Diagnosis.Name"])
        """
        with Session(self.engine) as session:
            cursor = session.execute(self._denormalize(include_tables=include_tables))
            yield from cursor.mappings()
            for row in cursor.mappings():
                yield row


    # =========================================================================
    # Asset Restructuring Methods
    # =========================================================================

    def _build_dataset_type_path_map(
        self,
        type_selector: Callable[[list[str]], str] | None = None,
    ) -> dict[RID, list[str]]:
        """Build a mapping from dataset RID to its type path in the hierarchy.

        Recursively traverses nested datasets to create a mapping where each
        dataset RID maps to its hierarchical type path (e.g., ["complete", "training"]).

        Args:
            type_selector: Function to select type when dataset has multiple types.
                Receives list of type names, returns selected type name.
                Defaults to selecting first type or "unknown" if no types.

        Returns:
            Dictionary mapping dataset RID to list of type names from root to leaf.
            e.g., {"4-ABC": ["complete", "training"], "4-DEF": ["complete", "testing"]}
        """
        if type_selector is None:
            type_selector = lambda types: types[0] if types else "unknown"

        type_paths: dict[RID, list[str]] = {}

        def traverse(dataset: DatasetBag, parent_path: list[str], visited: set[RID]) -> None:
            if dataset.dataset_rid in visited:
                return
            visited.add(dataset.dataset_rid)

            current_type = type_selector(dataset.dataset_types)
            current_path = parent_path + [current_type]
            type_paths[dataset.dataset_rid] = current_path

            for child in dataset.list_dataset_children():
                traverse(child, current_path, visited)

        traverse(self, [], set())
        return type_paths

    def _get_asset_dataset_mapping(self, asset_table: str) -> dict[RID, RID]:
        """Map asset RIDs to their containing dataset RID.

        For each asset in the specified table, determines which dataset directly
        contains it (not considering nesting - just direct membership).

        Args:
            asset_table: Name of the asset table (e.g., "Image")

        Returns:
            Dictionary mapping asset RID to the dataset RID that contains it.
        """
        asset_to_dataset: dict[RID, RID] = {}

        def collect_from_dataset(dataset: DatasetBag, visited: set[RID]) -> None:
            if dataset.dataset_rid in visited:
                return
            visited.add(dataset.dataset_rid)

            members = dataset.list_dataset_members(recurse=False)
            for asset in members.get(asset_table, []):
                asset_to_dataset[asset["RID"]] = dataset.dataset_rid

            for child in dataset.list_dataset_children():
                collect_from_dataset(child, visited)

        collect_from_dataset(self, set())
        return asset_to_dataset

    def _load_feature_values_cache(
        self,
        asset_table: str,
        group_keys: list[str],
    ) -> dict[str, dict[RID, Any]]:
        """Load feature values into a cache for efficient lookup.

        Pre-loads feature values for any group_keys that are feature names,
        organizing them by target entity RID for fast lookup.

        Args:
            asset_table: The asset table name to find features for.
            group_keys: List of potential feature names to cache.

        Returns:
            Dictionary mapping feature_name -> {target_rid -> feature_value}
            Only includes entries for keys that are actually features.
        """
        cache: dict[str, dict[RID, Any]] = {}
        logger = logging.getLogger("deriva_ml")

        # Find all features on tables that this asset table references
        asset_table_obj = self.model.name_to_table(asset_table)

        # Check features on the asset table itself
        for feature in self.find_features(asset_table):
            if feature.feature_name in group_keys:
                cache[feature.feature_name] = {}
                try:
                    feature_values = self.list_feature_values(asset_table, feature.feature_name)
                    for fv in feature_values:
                        target_col = asset_table
                        if target_col in fv:
                            # Get the value - prefer term columns, then value columns
                            value = None
                            term_cols = [c.name for c in feature.term_columns]
                            value_cols = [c.name for c in feature.value_columns]
                            if term_cols and term_cols[0] in fv:
                                value = fv[term_cols[0]]
                            elif value_cols and value_cols[0] in fv:
                                value = fv[value_cols[0]]
                            if value is not None:
                                cache[feature.feature_name][fv[target_col]] = value
                except Exception as e:
                    logger.warning(f"Could not load feature {feature.feature_name}: {e}")

        # Also check features on referenced tables (via foreign keys)
        for fk in asset_table_obj.foreign_keys:
            target_table = fk.pk_table
            for feature in self.find_features(target_table):
                if feature.feature_name in group_keys:
                    cache[feature.feature_name] = {}
                    try:
                        feature_values = self.list_feature_values(target_table.name, feature.feature_name)
                        for fv in feature_values:
                            target_col = target_table.name
                            if target_col in fv:
                                value = None
                                term_cols = [c.name for c in feature.term_columns]
                                value_cols = [c.name for c in feature.value_columns]
                                if term_cols and term_cols[0] in fv:
                                    value = fv[term_cols[0]]
                                elif value_cols and value_cols[0] in fv:
                                    value = fv[value_cols[0]]
                                if value is not None:
                                    cache[feature.feature_name][fv[target_col]] = value
                    except Exception as e:
                        logger.warning(f"Could not load feature {feature.feature_name}: {e}")

        return cache

    def _resolve_grouping_value(
        self,
        asset: dict[str, Any],
        group_key: str,
        feature_cache: dict[str, dict[RID, Any]],
    ) -> str:
        """Resolve a grouping value for an asset.

        First checks if group_key is a direct column on the asset record,
        then checks if it's a feature name in the feature cache.

        Args:
            asset: The asset record dictionary.
            group_key: Column name or feature name to group by.
            feature_cache: Pre-loaded feature values keyed by feature name -> target RID -> value.

        Returns:
            The resolved value as a string, or "unknown" if not found or None.
        """
        # First check if it's a direct column on the asset table
        if group_key in asset:
            value = asset[group_key]
            if value is not None:
                return str(value)
            return "unknown"

        # Check if it's a feature name
        if group_key in feature_cache:
            feature_values = feature_cache[group_key]
            # Check each column in the asset that might be a FK to the feature target
            for column_name, column_value in asset.items():
                if column_value and column_value in feature_values:
                    return str(feature_values[column_value])
            # Also check if the asset's own RID is in the feature values
            if asset.get("RID") in feature_values:
                return str(feature_values[asset["RID"]])

        return "unknown"

    def restructure_assets(
        self,
        asset_table: str,
        output_dir: Path | str,
        group_by: list[str] | None = None,
        use_symlinks: bool = True,
        type_selector: Callable[[list[str]], str] | None = None,
    ) -> Path:
        """Restructure downloaded assets into a directory hierarchy.

        Creates a directory structure organizing assets by dataset types and
        grouping values (columns or features). This is useful for ML workflows
        that expect data organized in conventional folder structures.

        Args:
            asset_table: Name of the asset table (e.g., "Image").
            output_dir: Base directory for restructured assets.
            group_by: Column names or feature names to group by. These create
                additional subdirectory levels after the dataset type path.
            use_symlinks: If True (default), create symlinks to original files.
                If False, copy files. Symlinks save disk space but require
                the original bag to remain in place.
            type_selector: Function to select type when dataset has multiple types.
                Receives list of type names, returns selected type name.
                Defaults to selecting first type or "unknown" if no types.

        Returns:
            Path to the output directory.

        Raises:
            DerivaMLException: If asset_table is not found in the dataset.

        Example:
            >>> bag.restructure_assets(
            ...     asset_table="Image",
            ...     output_dir=Path("./ml_data"),
            ...     group_by=["label", "Quality"],
            ...     use_symlinks=True,
            ... )
            # Creates:
            # ./ml_data/complete/training/gotit/Good/image1.jpg
            # ./ml_data/complete/training/gotit/Bad/image2.jpg
            # ./ml_data/complete/testing/dontgotit/Good/image3.jpg
        """
        logger = logging.getLogger("deriva_ml")
        group_by = group_by or []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Build dataset type path map
        type_path_map = self._build_dataset_type_path_map(type_selector)

        # Step 2: Get asset-to-dataset mapping
        asset_dataset_map = self._get_asset_dataset_mapping(asset_table)

        # Step 3: Load feature values cache for relevant features
        feature_cache = self._load_feature_values_cache(asset_table, group_by)

        # Step 4: Get all assets
        members = self.list_dataset_members(recurse=True)
        assets = members.get(asset_table, [])

        if not assets:
            logger.warning(f"No assets found in table '{asset_table}'")
            return output_dir

        # Step 5: Process each asset
        for asset in assets:
            # Get source file path
            filename = asset.get("Filename")
            if not filename:
                logger.warning(f"Asset {asset.get('RID')} has no Filename")
                continue

            source_path = Path(filename)
            if not source_path.exists():
                logger.warning(f"Asset file not found: {source_path}")
                continue

            # Get dataset type path
            dataset_rid = asset_dataset_map.get(asset["RID"])
            type_path = type_path_map.get(dataset_rid, ["unknown"])

            # Resolve grouping values
            group_path = []
            for key in group_by:
                value = self._resolve_grouping_value(asset, key, feature_cache)
                group_path.append(value)

            # Build target directory
            target_dir = output_dir.joinpath(*type_path, *group_path)
            target_dir.mkdir(parents=True, exist_ok=True)

            # Create link or copy
            target_path = target_dir / source_path.name

            # Handle existing files
            if target_path.exists() or target_path.is_symlink():
                target_path.unlink()

            if use_symlinks:
                try:
                    target_path.symlink_to(source_path.resolve())
                except OSError as e:
                    # Fall back to copy on platforms that don't support symlinks
                    logger.warning(f"Symlink failed, falling back to copy: {e}")
                    shutil.copy2(source_path, target_path)
            else:
                shutil.copy2(source_path, target_path)

        return output_dir


# Note: validate_call decorators with Self return types were removed because
# Pydantic doesn't support typing.Self in validate_call contexts.
