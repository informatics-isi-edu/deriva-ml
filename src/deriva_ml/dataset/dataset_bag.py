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
from sqlalchemy import CompoundSelect, Engine, Select, and_, inspect, select, union
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
        execution_rid (RID | None): RID of the execution associated with this dataset version, if any.
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
            execution_rid: RID of the execution associated with this dataset version.
                If None, will be looked up from the Dataset_Version table.

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
        self.execution_rid = execution_rid or (
            self.model._get_dataset_execution(self.dataset_rid) or {}
        ).get("Execution")

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
        version: Any = None,
        **kwargs: Any,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return a list of entities associated with a specific dataset.

        Args:
            recurse: Whether to include members of nested datasets.
            limit: Maximum number of members to return per type. None for no limit.
            _visited: Internal parameter to track visited datasets and prevent infinite recursion.
            version: Ignored (bags are immutable snapshots).
            **kwargs: Additional arguments (ignored, for protocol compatibility).

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
            Feature values as a list of dictionaries with column names as keys.
        """
        feature = self.model.lookup_feature(table, feature_name)
        feature_table = self.model.find_table(feature.feature_table.name)
        with Session(self.engine) as session:
            # Query directly using the SQL table to avoid ORM lazy loading issues
            sql_cmd = select(feature_table)
            result = session.execute(sql_cmd)
            # Convert to list of dicts with column names as keys
            rows = [dict(row._mapping) for row in result]
            return cast(datapath._ResultSet, rows)

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
        self,
        recurse: bool = False,
        _visited: set[RID] | None = None,
        version: Any = None,
        **kwargs: Any,
    ) -> list[Self]:
        """Get nested datasets.

        Args:
            recurse: Whether to include children of children.
            _visited: Internal parameter to track visited datasets and prevent infinite recursion.
            version: Ignored (bags are immutable snapshots).
            **kwargs: Additional arguments (ignored, for protocol compatibility).

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
            nested = [self._catalog.lookup_dataset(r[0]) for r in session.execute(sql_cmd).all()]

        result = copy(nested)
        if recurse:
            for child in nested:
                result.extend(child.list_dataset_children(recurse=recurse, _visited=_visited))
        return result

    def list_dataset_parents(
        self,
        recurse: bool = False,
        _visited: set[RID] | None = None,
        version: Any = None,
        **kwargs: Any,
    ) -> list[Self]:
        """Given a dataset_table RID, return a list of RIDs of the parent datasets if this is included in a
        nested dataset.

        Args:
            recurse: If True, recursively return all ancestor datasets.
            _visited: Internal parameter to track visited datasets and prevent infinite recursion.
            version: Ignored (bags are immutable snapshots).
            **kwargs: Additional arguments (ignored, for protocol compatibility).

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
            parents = [self._catalog.lookup_dataset(r[0]) for r in session.execute(sql_cmd).all()]

        if recurse:
            for parent in parents.copy():
                parents.extend(parent.list_dataset_parents(recurse=True, _visited=_visited))
        return parents

    def list_executions(self) -> list[RID]:
        """List all execution RIDs associated with this dataset.

        Returns all executions that used this dataset as input. This is
        tracked through the Dataset_Execution association table.

        Note:
            Unlike the live Dataset class which returns Execution objects,
            DatasetBag returns a list of execution RIDs since the bag is
            an offline snapshot and cannot look up live execution objects.

        Returns:
            List of execution RIDs associated with this dataset.

        Example:
            >>> bag = ml.download_dataset_bag(dataset_spec)
            >>> execution_rids = bag.list_executions()
            >>> for rid in execution_rids:
            ...     print(f"Associated execution: {rid}")
        """
        de_table = self.model.get_orm_class_by_name(f"{self.model.ml_schema}.Dataset_Execution")

        with Session(self.engine) as session:
            sql_cmd = select(de_table.Execution).where(de_table.Dataset == self.dataset_rid)
            return [r[0] for r in session.execute(sql_cmd).all()]

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

    def _denormalize_from_members(
        self,
        include_tables: list[str],
    ) -> Generator[dict[str, Any], None, None]:
        """Denormalize dataset members by joining related tables.

        This method creates a "wide table" view by joining related tables together,
        using list_dataset_members() as the data source. This ensures consistency
        with the catalog-based denormalize implementation. The result has outer join
        semantics - tables without FK relationships are included with NULL values.

        The method:
        1. Gets the list of dataset members for each included table via list_dataset_members
        2. For each member in the first table, follows foreign key relationships to
           get related records from other tables
        3. Tables without FK connections to the first table are included with NULLs
        4. Includes nested dataset members recursively

        Args:
            include_tables: List of table names to include in the output.

        Yields:
            dict[str, Any]: Rows with column names prefixed by table name (e.g., "Image.Filename").
                Unrelated tables have NULL values for their columns.

        Note:
            Column names in the result are prefixed with the table name to avoid
            collisions (e.g., "Image.Filename", "Subject.RID").
        """
        # Skip system columns in output
        skip_columns = {"RCT", "RMT", "RCB", "RMB"}

        # Get all members for the included tables (recursively includes nested datasets)
        members = self.list_dataset_members(recurse=True)

        # Build a lookup of columns for each table
        table_columns: dict[str, list[str]] = {}
        for table_name in include_tables:
            table = self.model.name_to_table(table_name)
            table_columns[table_name] = [
                c.name for c in table.columns if c.name not in skip_columns
            ]

        # Find the primary table (first non-empty table in include_tables)
        primary_table = None
        for table_name in include_tables:
            if table_name in members and members[table_name]:
                primary_table = table_name
                break

        if primary_table is None:
            # No data at all
            return

        primary_table_obj = self.model.name_to_table(primary_table)

        for member in members[primary_table]:
            # Build the row with all columns from all tables
            row: dict[str, Any] = {}

            # Add primary table columns
            for col_name in table_columns[primary_table]:
                prefixed_name = f"{primary_table}.{col_name}"
                row[prefixed_name] = member.get(col_name)

            # For each other table, try to join or add NULL values
            for other_table_name in include_tables:
                if other_table_name == primary_table:
                    continue

                other_table = self.model.name_to_table(other_table_name)
                other_cols = table_columns[other_table_name]

                # Initialize all columns to None (outer join behavior)
                for col_name in other_cols:
                    prefixed_name = f"{other_table_name}.{col_name}"
                    row[prefixed_name] = None

                # Try to find FK relationship and join
                if other_table_name in members:
                    try:
                        relationship = self.model._table_relationship(
                            primary_table_obj, other_table
                        )
                        fk_col, pk_col = relationship

                        # Look up the related record
                        fk_value = member.get(fk_col.name)
                        if fk_value:
                            for other_member in members.get(other_table_name, []):
                                if other_member.get(pk_col.name) == fk_value:
                                    for col_name in other_cols:
                                        prefixed_name = f"{other_table_name}.{col_name}"
                                        row[prefixed_name] = other_member.get(col_name)
                                    break
                    except DerivaMLException:
                        # No FK relationship - columns remain NULL (outer join)
                        pass

            yield row

    def denormalize_as_dataframe(
        self,
        include_tables: list[str],
        version: Any = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Denormalize the dataset bag into a single wide table (DataFrame).

        Denormalization transforms normalized relational data into a single "wide table"
        (also called a "flat table" or "denormalized table") by joining related tables
        together. This produces a DataFrame where each row contains all related information
        from multiple source tables, with columns from each table combined side-by-side.

        Wide tables are the standard input format for most machine learning frameworks,
        which expect all features for a single observation to be in one row. This method
        bridges the gap between normalized database schemas and ML-ready tabular data.

        **How it works:**

        Tables are joined based on their foreign key relationships stored in the bag's
        schema. For example, if Image has a foreign key to Subject, denormalizing
        ["Subject", "Image"] produces rows where each image appears with its subject's
        metadata.

        **Column naming:**

        Column names are prefixed with the source table name using dots to avoid
        collisions (e.g., "Image.Filename", "Subject.RID"). This differs from the
        live Dataset class which uses underscores.

        Args:
            include_tables: List of table names to include in the output. Tables
                are joined based on their foreign key relationships.
                Order doesn't matter - the join order is determined automatically.
            version: Ignored (bags are immutable snapshots of a specific version).
            **kwargs: Additional arguments (ignored, for protocol compatibility).

        Returns:
            pd.DataFrame: Wide table with columns from all included tables.

        Example:
            Create a training dataset from a downloaded bag::

                >>> # Download and materialize the dataset
                >>> bag = ml.download_dataset_bag(spec, materialize=True)

                >>> # Denormalize into a wide table
                >>> df = bag.denormalize_as_dataframe(["Image", "Diagnosis"])
                >>> print(df.columns.tolist())
                ['Image.RID', 'Image.Filename', 'Image.URL', 'Diagnosis.RID',
                 'Diagnosis.Label', 'Diagnosis.Confidence']

                >>> # Access local file paths for images
                >>> for _, row in df.iterrows():
                ...     local_path = bag.get_asset_path("Image", row["Image.RID"])
                ...     label = row["Diagnosis.Label"]
                ...     # Train on local_path with label

        See Also:
            denormalize_as_dict: Generator version for memory-efficient processing.
        """
        rows = list(self._denormalize_from_members(include_tables=include_tables))
        return pd.DataFrame(rows)

    def denormalize_as_dict(
        self,
        include_tables: list[str],
        version: Any = None,
        **kwargs: Any,
    ) -> Generator[dict[str, Any], None, None]:
        """Denormalize the dataset bag and yield rows as dictionaries.

        This is a memory-efficient alternative to denormalize_as_dataframe() that
        yields one row at a time as a dictionary instead of loading all data into
        a DataFrame. Use this when processing large datasets that may not fit in
        memory, or when you want to process rows incrementally.

        Like denormalize_as_dataframe(), this produces a "wide table" representation
        where each yielded dictionary contains all columns from the joined tables.
        See denormalize_as_dataframe() for detailed explanation of how denormalization
        works.

        **Column naming:**

        Column names are prefixed with the source table name using dots to avoid
        collisions (e.g., "Image.Filename", "Subject.RID"). This differs from the
        live Dataset class which uses underscores.

        Args:
            include_tables: List of table names to include in the output.
                Tables are joined based on their foreign key relationships.
            version: Ignored (bags are immutable snapshots of a specific version).
            **kwargs: Additional arguments (ignored, for protocol compatibility).

        Yields:
            dict[str, Any]: Dictionary representing one row of the wide table.
                Keys are column names in "Table.Column" format.

        Example:
            Stream through a large dataset for training::

                >>> bag = ml.download_dataset_bag(spec, materialize=True)
                >>> for row in bag.denormalize_as_dict(["Image", "Diagnosis"]):
                ...     # Get local file path for this image
                ...     local_path = bag.get_asset_path("Image", row["Image.RID"])
                ...     label = row["Diagnosis.Label"]
                ...     # Process image and label...

            Build a PyTorch dataset efficiently::

                >>> class BagDataset(torch.utils.data.IterableDataset):
                ...     def __init__(self, bag, tables):
                ...         self.bag = bag
                ...         self.tables = tables
                ...     def __iter__(self):
                ...         for row in self.bag.denormalize_as_dict(self.tables):
                ...             img_path = self.bag.get_asset_path("Image", row["Image.RID"])
                ...             yield load_image(img_path), row["Diagnosis.Label"]

        See Also:
            denormalize_as_dataframe: Returns all data as a pandas DataFrame.
        """
        yield from self._denormalize_from_members(include_tables=include_tables)


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
        enforce_vocabulary: bool = True,
    ) -> dict[str, dict[RID, Any]]:
        """Load feature values into a cache for efficient lookup.

        Pre-loads feature values for any group_keys that are feature names,
        organizing them by target entity RID for fast lookup.

        Args:
            asset_table: The asset table name to find features for.
            group_keys: List of potential feature names to cache. Supports two formats:
                - "FeatureName": Uses the first term column (default behavior)
                - "FeatureName.column_name": Uses the specified column from the feature table
            enforce_vocabulary: If True (default), only allow features with
                controlled vocabulary term columns and raise an error if an
                asset has multiple values. If False, allow any feature type
                and use the first value found when multiple exist.

        Returns:
            Dictionary mapping group_key -> {target_rid -> feature_value}
            Only includes entries for keys that are actually features.

        Raises:
            DerivaMLException: If enforce_vocabulary is True and:
                - A feature has no term columns (not vocabulary-based), or
                - An asset has multiple different vocabulary term values for the same feature.
        """
        from deriva_ml.core.exceptions import DerivaMLException

        cache: dict[str, dict[RID, Any]] = {}
        logger = logging.getLogger("deriva_ml")

        # Parse group_keys to extract feature names and optional column specifications
        # Format: "FeatureName" or "FeatureName.column_name"
        feature_column_map: dict[str, str | None] = {}  # group_key -> specific column or None
        feature_names_to_check: set[str] = set()
        for key in group_keys:
            if "." in key:
                parts = key.split(".", 1)
                feature_name = parts[0]
                column_name = parts[1]
                feature_column_map[key] = column_name
                feature_names_to_check.add(feature_name)
            else:
                feature_column_map[key] = None
                feature_names_to_check.add(key)

        def process_feature(feat: Any, table_name: str, group_key: str, specific_column: str | None) -> None:
            """Process a single feature and add its values to the cache."""
            term_cols = [c.name for c in feat.term_columns]
            value_cols = [c.name for c in feat.value_columns]
            all_cols = term_cols + value_cols

            # Determine which column to use for the value
            if specific_column:
                # User specified a specific column
                if specific_column not in all_cols:
                    raise DerivaMLException(
                        f"Column '{specific_column}' not found in feature '{feat.feature_name}'. "
                        f"Available columns: {all_cols}"
                    )
                use_column = specific_column
            elif term_cols:
                # Use first term column (default behavior)
                use_column = term_cols[0]
            elif not enforce_vocabulary and value_cols:
                # Fall back to value columns if allowed
                use_column = value_cols[0]
            else:
                if enforce_vocabulary:
                    raise DerivaMLException(
                        f"Feature '{feat.feature_name}' on table '{table_name}' has no "
                        f"controlled vocabulary term columns. Only vocabulary-based features "
                        f"can be used for grouping when enforce_vocabulary=True. "
                        f"Set enforce_vocabulary=False to allow non-vocabulary features."
                    )
                return

            cache[group_key] = {}
            feature_values = self.list_feature_values(table_name, feat.feature_name)

            for fv in feature_values:
                target_col = table_name
                if target_col not in fv:
                    continue

                target_rid = fv[target_col]

                # Get the value from the specified column
                value = fv.get(use_column) if use_column in fv else None

                if value is None:
                    continue

                # Check for multiple values for the same target
                if target_rid in cache[group_key]:
                    existing_value = cache[group_key][target_rid]
                    if existing_value != value:
                        if enforce_vocabulary:
                            raise DerivaMLException(
                                f"Asset '{target_rid}' has multiple different values for "
                                f"feature '{feat.feature_name}': '{existing_value}' and '{value}'. "
                                f"Each asset must have at most one vocabulary term value per feature "
                                f"when enforce_vocabulary=True. Set enforce_vocabulary=False to use "
                                f"the first value found."
                            )
                        # If not enforcing, keep the first value (already in cache)
                else:
                    cache[group_key][target_rid] = value

        # Find all features on tables that this asset table references
        asset_table_obj = self.model.name_to_table(asset_table)

        # Check features on the asset table itself
        for feature in self.find_features(asset_table):
            if feature.feature_name in feature_names_to_check:
                # Find all group_keys that reference this feature
                for group_key, specific_col in feature_column_map.items():
                    # Check if this group_key references this feature
                    key_feature = group_key.split(".")[0] if "." in group_key else group_key
                    if key_feature == feature.feature_name:
                        try:
                            process_feature(feature, asset_table, group_key, specific_col)
                        except DerivaMLException:
                            raise
                        except Exception as e:
                            logger.warning(f"Could not load feature {feature.feature_name}: {e}")

        # Also check features on referenced tables (via foreign keys)
        for fk in asset_table_obj.foreign_keys:
            target_table = fk.pk_table
            for feature in self.find_features(target_table):
                if feature.feature_name in feature_names_to_check:
                    # Find all group_keys that reference this feature
                    for group_key, specific_col in feature_column_map.items():
                        # Check if this group_key references this feature
                        key_feature = group_key.split(".")[0] if "." in group_key else group_key
                        if key_feature == feature.feature_name:
                            try:
                                process_feature(feature, target_table.name, group_key, specific_col)
                            except DerivaMLException:
                                raise
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
        enforce_vocabulary: bool = True,
    ) -> Path:
        """Restructure downloaded assets into a directory hierarchy.

        Creates a directory structure organizing assets by dataset types and
        grouping values. This is useful for ML workflows that expect data
        organized in conventional folder structures (e.g., PyTorch ImageFolder).

        Args:
            asset_table: Name of the asset table (e.g., "Image").
            output_dir: Base directory for restructured assets.
            group_by: Names to group assets by. Each name creates a subdirectory
                level after the dataset type path. Names can be:

                - **Column names**: Direct columns on the asset table. The column
                  value becomes the subdirectory name.
                - **Feature names**: Features defined on the asset table (or tables
                  it references via foreign keys). The feature's vocabulary term
                  value becomes the subdirectory name.

                Column names are checked first, then feature names. If a value
                is not found, "unknown" is used as the subdirectory name.

            use_symlinks: If True (default), create symlinks to original files.
                If False, copy files. Symlinks save disk space but require
                the original bag to remain in place.
            type_selector: Function to select type when dataset has multiple types.
                Receives list of type names, returns selected type name.
                Defaults to selecting first type or "unknown" if no types.
            enforce_vocabulary: If True (default), only allow features that have
                controlled vocabulary term columns, and raise an error if an asset
                has multiple different values for the same feature. This ensures
                clean, unambiguous directory structures. If False, allow any feature
                type and use the first value found when multiple values exist.

        Returns:
            Path to the output directory.

        Raises:
            DerivaMLException: If asset_table is not found in the dataset, or if
                enforce_vocabulary is True and a feature is not vocabulary-based
                or has multiple values for an asset.

        Examples:
            Group by a column on the asset table::

                # If Image table has a "label" column with values like "cat", "dog"
                bag.restructure_assets(
                    asset_table="Image",
                    output_dir="./ml_data",
                    group_by=["label"],
                )
                # Creates:
                # ./ml_data/training/cat/image1.jpg
                # ./ml_data/training/dog/image2.jpg
                # ./ml_data/testing/cat/image3.jpg

            Group by a feature's vocabulary term value::

                # If Image has a "Diagnosis" feature with vocabulary terms "Normal", "Abnormal"
                bag.restructure_assets(
                    asset_table="Image",
                    output_dir="./ml_data",
                    group_by=["Diagnosis"],
                )
                # Creates:
                # ./ml_data/training/Normal/image1.jpg
                # ./ml_data/training/Abnormal/image2.jpg

            Combine column and feature grouping::

                # Group first by a column, then by a feature
                bag.restructure_assets(
                    asset_table="Image",
                    output_dir="./ml_data",
                    group_by=["modality", "Diagnosis"],
                )
                # Creates:
                # ./ml_data/training/MRI/Normal/image1.jpg
                # ./ml_data/training/CT/Abnormal/image2.jpg

            Allow non-vocabulary features (use with caution)::

                bag.restructure_assets(
                    asset_table="Image",
                    output_dir="./ml_data",
                    group_by=["score_bucket"],
                    enforce_vocabulary=False,  # Allow numeric/text features
                )
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
        feature_cache = self._load_feature_values_cache(asset_table, group_by, enforce_vocabulary)

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
