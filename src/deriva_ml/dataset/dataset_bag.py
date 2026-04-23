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

# Third-party imports
import pandas as pd

# Local imports
from deriva.core.ermrest_model import Table

# Deriva imports
from sqlalchemy import CompoundSelect, Engine, inspect, select, union
from sqlalchemy.orm import RelationshipProperty, Session
from sqlalchemy.orm.util import AliasedClass

from deriva_ml.core.definitions import RID
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.aux_classes import DatasetHistory, DatasetVersion
from deriva_ml.feature import Feature, FeatureRecord

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
        self.execution_rid = execution_rid or (self.model._get_dataset_execution(self.dataset_rid) or {}).get(
            "Execution"
        )

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
        return (
            f"<deriva_ml.DatasetBag object at {hex(id(self))}: rid='{self.dataset_rid}', "
            f"version='{self.current_version}', types={self.dataset_types}>"
        )

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

    def get_table_as_dataframe(self, table: str) -> pd.DataFrame:
        """Get table contents as a pandas DataFrame.

        Convenience method that wraps get_table_as_dict() to return a DataFrame.
        Provides access to all rows in a table, not just those belonging to this
        dataset. For dataset-filtered results, use list_dataset_members() instead.

        Args:
            table: Name of the table to retrieve (e.g., "Subject", "Image").

        Returns:
            DataFrame with one row per record in the table.

        Example:
            >>> df = bag.get_table_as_dataframe("Image")
            >>> print(df.shape)
        """
        return pd.DataFrame(self.get_table_as_dict(table))

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
            if not self.model.is_domain_schema(element_table.schema) and element_table.name not in ["Dataset", "File"]:
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
                    for k, v in nested_dataset.list_dataset_members(
                        recurse=recurse, limit=limit, _visited=_visited
                    ).items():
                        members[k].extend(v)
        return dict(members)

    def find_features(self, table: str | Table) -> Iterable[Feature]:
        """Find all features defined on a table within this dataset bag.

        Features are measurable properties associated with records in a table,
        stored as association tables linking the target table to vocabulary
        terms, assets, or metadata columns. This method discovers all such
        feature definitions for the given table.

        Each returned ``Feature`` object provides:

        - ``feature_name``: The feature's name (e.g., ``"Classification"``)
        - ``target_table``: The table the feature applies to
        - ``feature_table``: The association table storing feature values
        - ``term_columns``, ``asset_columns``, ``value_columns``: Column role sets
        - ``feature_record_class()``: A Pydantic model for reading/writing values

        Args:
            table: The table to find features for (name or Table object).

        Returns:
            An iterable of Feature instances describing each feature
            defined on the table.

        Example:
            >>> for f in bag.find_features("Image"):
            ...     print(f"{f.feature_name}: {len(f.term_columns)} terms, "
            ...           f"{len(f.value_columns)} value columns")
        """
        return self.model.find_features(table)

    def feature_values(
        self,
        table: str | Table,
        feature_name: str,
        selector: Callable[[list[FeatureRecord]], FeatureRecord | None] | None = None,
    ) -> Iterable[FeatureRecord]:
        """Yield offline feature values — same signature as ``DerivaML.feature_values``.

        Reads feature records from the bag's per-feature denormalization cache
        (populated lazily on first access). Because bags are immutable snapshots,
        the cache is stable for the bag's lifetime.

        When *selector* is ``None``, every stored record is yielded in source order.
        When a *selector* is provided, records are grouped by target RID, the
        selector is called once per group (always, even single-element groups),
        and only groups for which the selector returns a non-``None`` value appear
        in the output.

        Args:
            table: Target table name or ``Table`` object (e.g. ``"Image"``).
            feature_name: Name of the feature to read (e.g. ``"Glaucoma"``).
            selector: Optional callable ``(list[FeatureRecord]) -> FeatureRecord | None``
                that resolves multiple values per target to a single record (or
                drops the target when it returns ``None``). Use
                ``FeatureRecord.select_newest`` to pick the most-recently created
                value.

        Yields:
            FeatureRecord instances with typed fields matching the feature
            definition. Selector-filtered records (``None`` return) are omitted.

        Raises:
            DerivaMLException: If *feature_name* does not exist on *table*.
            DerivaMLDataError: If the bag is corrupt (source table missing).

        Example:
            >>> from deriva_ml.feature import FeatureRecord
            >>> for rec in bag.feature_values("Image", "Glaucoma"):
            ...     print(rec.Image, rec.Glaucoma)
            >>> # With selector — one record per image, most recent wins:
            >>> records = list(bag.feature_values(
            ...     "Image", "Glaucoma", selector=FeatureRecord.select_newest,
            ... ))
        """
        from collections import defaultdict

        from deriva_ml.dataset.bag_feature_cache import BagFeatureCache

        if not hasattr(self, "_feature_cache"):
            self._feature_cache = BagFeatureCache(self)

        target_col = table if isinstance(table, str) else table.name
        records = list(self._feature_cache.fetch_feature_records(target_col, feature_name))

        if selector is None:
            yield from records
            return

        # Group by target RID, then apply selector to every group (always call
        # selector — never short-circuit for single-element groups).
        grouped: dict[str, list[FeatureRecord]] = defaultdict(list)
        for rec in records:
            target_rid = getattr(rec, target_col, None)
            if target_rid is not None:
                grouped[target_rid].append(rec)

        for group in grouped.values():
            chosen = selector(group)
            if chosen is not None:
                yield chosen

    def lookup_feature(self, table: str | Table, feature_name: str) -> "Feature":
        """Look up a feature definition from bag metadata — works fully offline.

        Returns a ``Feature`` object with the same shape as the one returned by
        ``DerivaML.lookup_feature``.  The ``feature_record_class()`` method on the
        returned object also works offline, enabling callers to construct
        ``FeatureRecord`` instances from bag data for later staging via
        ``exe.add_features`` when back online.

        Args:
            table: Target table name or ``Table`` object (e.g. ``"Image"``).
            feature_name: Name of the feature (e.g. ``"Glaucoma"``).

        Returns:
            Feature object for *feature_name* on *table*.

        Raises:
            DerivaMLException: If the feature does not exist on *table* in the bag.

        Example:
            >>> feat = bag.lookup_feature("Image", "Glaucoma")
            >>> RecordClass = feat.feature_record_class()
            >>> record = RecordClass(Image="1-ABC", Glaucoma="Normal")
            >>> print(record.Glaucoma)
            Normal
        """
        return self.model.lookup_feature(table, feature_name)

    def list_workflow_executions(self, workflow: str) -> list[str]:
        """Return execution RIDs from the bag that match the given workflow.

        Reads the bag's local SQLite ``Execution`` table.  The *workflow* argument
        is resolved in order:

        1. **Workflow RID** — if a row in the ``Workflow`` table has ``RID ==
           workflow``, return all ``Execution.RID`` values whose ``Workflow``
           column matches.
        2. **Workflow_Type name** — if no RID match, look up workflows via the
           ``Workflow_Workflow_Type`` association table and return executions for
           all matching workflows.

        This mirrors the contract of ``DerivaML.list_workflow_executions`` but
        operates entirely against the bag's offline SQLite data.

        Args:
            workflow: Workflow RID **or** ``Workflow_Type`` name to resolve.

        Returns:
            List of execution RIDs (possibly empty) that are associated with the
            resolved workflow(s).

        Raises:
            DerivaMLException: If *workflow* cannot be resolved as either a
                Workflow RID or a Workflow_Type name in this bag.

        Example:
            >>> rids = bag.list_workflow_executions("Glaucoma_Training_v2")
            >>> print(len(rids))
            3
        """
        from sqlalchemy import select as sa_select

        workflow_table = self.model.find_table("Workflow")
        execution_table = self.model.find_table("Execution")

        with Session(self.engine) as session:
            # Phase 1: try as a Workflow RID.
            wf_rows = session.execute(
                sa_select(workflow_table).where(workflow_table.c.RID == workflow)
            ).mappings().all()

            if wf_rows:
                rows = session.execute(
                    sa_select(execution_table.c.RID).where(
                        execution_table.c.Workflow == workflow
                    )
                ).all()
                return [r[0] for r in rows]

            # Phase 2: fall back to Workflow_Type name via association table.
            wwt = self.model.find_table("Workflow_Workflow_Type")
            wf_of_type = [
                r[0]
                for r in session.execute(
                    sa_select(wwt.c.Workflow).where(wwt.c.Workflow_Type == workflow)
                ).all()
            ]
            if not wf_of_type:
                raise DerivaMLException(
                    f"No workflow resolved for '{workflow}' in bag — tried as "
                    "Workflow RID and Workflow_Type name."
                )
            rows = session.execute(
                sa_select(execution_table.c.RID).where(
                    execution_table.c.Workflow.in_(wf_of_type)
                )
            ).all()
            return [r[0] for r in rows]

    def fetch_table_features(
        self,
        table: Table | str,
        feature_name: str | None = None,
        selector: Callable[[list[FeatureRecord]], FeatureRecord] | None = None,
    ) -> dict[str, list[FeatureRecord]]:
        """Fetch all feature values for a table, grouped by feature name.

        Queries the local SQLite database within this dataset bag and returns
        a dictionary mapping feature names to lists of FeatureRecord instances.
        This is useful for retrieving all annotations on a table in a single
        call — for example, getting all classification labels, quality scores,
        and bounding boxes for a set of images at once.

        **Selector for resolving multiple values:**

        An asset may have multiple values for the same feature — for example,
        labels from different annotators or model runs. When a ``selector`` is
        provided, records are grouped by target RID and the selector is called
        once per group to pick a single value. Groups with only one record
        are passed through unchanged.

        A selector is any callable with signature
        ``(list[FeatureRecord]) -> FeatureRecord``. Built-in selectors:

        - ``FeatureRecord.select_newest`` — picks the record with the most
          recent ``RCT`` (Row Creation Time).

        Custom selector example::

            def select_highest_confidence(records):
                return max(records, key=lambda r: getattr(r, "Confidence", 0))

        Args:
            table: The table to fetch features for (name or Table object).
            feature_name: If provided, only fetch values for this specific
                feature. If ``None``, fetch all features on the table.
            selector: Optional function to select among multiple feature values
                for the same target object. Receives a list of FeatureRecord
                instances (all for the same target RID) and returns the selected
                one.

        Returns:
            dict[str, list[FeatureRecord]]: Keys are feature names, values are
            lists of FeatureRecord instances. When a selector is provided, each
            target object appears at most once per feature.

        Raises:
            DerivaMLException: If a specified ``feature_name`` doesn't exist
                on the table.

        Examples:
            Fetch all features for a table::

                >>> features = bag.fetch_table_features("Image")
                >>> for name, records in features.items():
                ...     print(f"{name}: {len(records)} values")

            Fetch a single feature with newest-value selection::

                >>> features = bag.fetch_table_features(
                ...     "Image",
                ...     feature_name="Classification",
                ...     selector=FeatureRecord.select_newest,
                ... )

            Convert results to a DataFrame::

                >>> features = bag.fetch_table_features("Image", feature_name="Quality")
                >>> import pandas as pd
                >>> df = pd.DataFrame([r.model_dump() for r in features["Quality"]])
        """
        features = list(self.find_features(table))
        if feature_name is not None:
            features = [f for f in features if f.feature_name == feature_name]
            if not features:
                table_name = table if isinstance(table, str) else table.name
                raise DerivaMLException(f"Feature '{feature_name}' not found on table '{table_name}'.")

        result: dict[str, list[FeatureRecord]] = {}

        for feat in features:
            record_class = feat.feature_record_class()
            field_names = set(record_class.model_fields.keys())
            target_col = feat.target_table.name

            # Query raw values from SQLite
            feature_table = self.model.find_table(feat.feature_table.name)
            with Session(self.engine) as session:
                sql_cmd = select(feature_table)
                sql_result = session.execute(sql_cmd)
                rows = [dict(row._mapping) for row in sql_result]

            records: list[FeatureRecord] = []
            for raw_value in rows:
                filtered_data = {k: v for k, v in raw_value.items() if k in field_names}
                records.append(record_class(**filtered_data))

            if selector and records:
                # Group by target RID and apply selector
                grouped: dict[str, list[FeatureRecord]] = defaultdict(list)
                for rec in records:
                    target_rid = getattr(rec, target_col, None)
                    if target_rid is not None:
                        grouped[target_rid].append(rec)
                records = [selector(group) if len(group) > 1 else group[0] for group in grouped.values()]

            result[feat.feature_name] = records

        return result

    def list_feature_values(
        self,
        table: Table | str,
        feature_name: str,
        selector: Callable[[list[FeatureRecord]], FeatureRecord] | None = None,
    ) -> Iterable[FeatureRecord]:
        """Retrieve all values for a single feature as typed FeatureRecord instances.

        Convenience wrapper around ``fetch_table_features()`` for the common
        case of querying a single feature by name. Returns a flat list of
        FeatureRecord objects — one per feature value (or one per target object
        when a ``selector`` is provided).

        Each returned record is a dynamically-generated Pydantic model with
        typed fields matching the feature's definition. For example, an
        ``Image_Classification`` feature might produce records with fields
        ``Image`` (str), ``Image_Class`` (str), ``Execution`` (str),
        ``RCT`` (str), and ``Feature_Name`` (str).

        Args:
            table: The table the feature is defined on (name or Table object).
            feature_name: Name of the feature to retrieve values for.
            selector: Optional function to resolve multiple values per target.
                See ``fetch_table_features`` for details on how selectors work.
                Use ``FeatureRecord.select_newest`` to pick the most recently
                created value.

        Returns:
            Iterable[FeatureRecord]: FeatureRecord instances with:

            - ``Execution``: RID of the execution that created this value
            - ``Feature_Name``: Name of the feature
            - ``RCT``: Row Creation Time (ISO 8601 timestamp)
            - Feature-specific columns as typed attributes (vocabulary terms,
              asset references, or value columns depending on the feature)
            - ``model_dump()``: Convert to a dictionary

        Raises:
            DerivaMLException: If the feature doesn't exist on the table.

        Examples:
            Get typed feature records::

                >>> for record in bag.list_feature_values("Image", "Quality"):
                ...     print(f"Image {record.Image}: {record.ImageQuality}")
                ...     print(f"Created by execution: {record.Execution}")

            Select newest when multiple values exist::

                >>> records = list(bag.list_feature_values(
                ...     "Image", "Quality",
                ...     selector=FeatureRecord.select_newest,
                ... ))

            Convert to a list of dicts::

                >>> dicts = [r.model_dump() for r in
                ...          bag.list_feature_values("Image", "Classification")]
        """
        result = self.fetch_table_features(table, feature_name=feature_name, selector=selector)
        return result.get(feature_name, [])

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

    def get_denormalized_as_dataframe(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
        ignore_unrelated_anchors: bool = False,
    ) -> pd.DataFrame:
        """Return the dataset bag as a denormalized wide table (DataFrame).

        Shortcut for
        :meth:`~deriva_ml.local_db.denormalizer.Denormalizer.as_dataframe`.
        Works against the bag's local SQLite (no catalog needed). See the
        ``Denormalizer`` class docstring for the full semantic rules
        (Rules 1-8).

        Args:
            include_tables: Tables whose columns appear in the output.
            row_per: Optional explicit leaf table (Rule 2).
            via: Optional path-only intermediates (Rule 6).
            ignore_unrelated_anchors: If True, silently drop anchors
                with no FK path (Rule 8).

        Returns:
            A :class:`pandas.DataFrame` with one row per ``row_per``
            instance in the bag. Columns use ``Table.column`` notation.

        Example::

            bag = dataset.download_dataset_bag(version)
            df = bag.get_denormalized_as_dataframe(["Image", "Subject"])
        """
        from deriva_ml.local_db.denormalizer import Denormalizer

        return Denormalizer(self).as_dataframe(
            include_tables,
            row_per=row_per,
            via=via,
            ignore_unrelated_anchors=ignore_unrelated_anchors,
        )

    def get_denormalized_as_dict(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
        ignore_unrelated_anchors: bool = False,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream the denormalized dataset bag rows as dicts.

        Shortcut for
        :meth:`~deriva_ml.local_db.denormalizer.Denormalizer.as_dict`.
        Same rules and exceptions as
        :meth:`get_denormalized_as_dataframe` but yields one dict per
        row. Use this for large bags where a full DataFrame won't fit
        in memory — each row is yielded as soon as it's produced.

        Args:
            include_tables: Tables whose columns appear in the output.
            row_per: Optional explicit leaf table (Rule 2).
            via: Optional path-only intermediates (Rule 6).
            ignore_unrelated_anchors: If True, silently drop anchors
                with no FK path (Rule 8).

        Yields:
            ``dict[str, Any]`` per row — keys are ``Table.column``
            labels, values are raw Python types.

        Example::

            for row in bag.get_denormalized_as_dict(["Image", "Subject"]):
                process(row["Image.RID"], row["Subject.Name"])
        """
        from deriva_ml.local_db.denormalizer import Denormalizer

        yield from Denormalizer(self).as_dict(
            include_tables,
            row_per=row_per,
            via=via,
            ignore_unrelated_anchors=ignore_unrelated_anchors,
        )

    def list_denormalized_columns(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        """List the columns the denormalized table would have.

        Shortcut for
        :meth:`~deriva_ml.local_db.denormalizer.Denormalizer.columns`.
        Model-only — no data fetch. Runs the same Rule 2/5/6 validation
        as :meth:`get_denormalized_as_dataframe` so planner errors
        surface early.

        Args:
            include_tables: Tables whose columns appear in the output.
            row_per: Optional explicit leaf table (Rule 2).
            via: Optional path-only intermediates (Rule 6).

        Returns:
            List of ``(column_name, column_type)`` tuples.

        Example::

            cols = bag.list_denormalized_columns(["Image", "Subject"])
        """
        from deriva_ml.local_db.denormalizer import Denormalizer

        return Denormalizer(self).columns(
            include_tables,
            row_per=row_per,
            via=via,
        )

    def describe_denormalized(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
    ) -> dict[str, Any]:
        """Dry-run the denormalization; return planning metadata.

        Shortcut for
        :meth:`~deriva_ml.local_db.denormalizer.Denormalizer.describe` —
        returns the full plan dict (see that method's docstring for the
        exact 12-key shape). Never raises on ambiguity.
        """
        from deriva_ml.local_db.denormalizer import Denormalizer

        return Denormalizer(self).describe(
            include_tables,
            row_per=row_per,
            via=via,
        )

    def list_schema_paths(
        self,
        tables: list[str] | None = None,
    ) -> dict[str, Any]:
        """List FK paths reachable from this dataset bag's members.

        Shortcut for
        :meth:`~deriva_ml.local_db.denormalizer.Denormalizer.list_paths`.
        Useful for schema exploration — answers "what tables could I
        include in a denormalization of this bag?"

        Args:
            tables: Optional filter — when given, ``schema_paths`` in
                the returned dict includes only entries involving at
                least one of these tables.

        Returns:
            Dict with 6 keys: ``member_types``, ``anchor_types``,
            ``reachable_tables``, ``association_tables``,
            ``feature_tables``, ``schema_paths``. See
            :meth:`Denormalizer.list_paths` for the detailed shape.

        Example::

            info = bag.list_schema_paths()
            print(info["member_types"])
        """
        from deriva_ml.local_db.denormalizer import Denormalizer

        return Denormalizer(self).list_paths(tables=tables)

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
            type_selector = lambda types: types[0] if types else "Testing"

        type_paths: dict[RID, list[str]] = {}

        def traverse(dataset: DatasetBag, parent_path: list[str], visited: set[RID]) -> None:
            if dataset.dataset_rid in visited:
                return
            visited.add(dataset.dataset_rid)

            current_type = type_selector(dataset.dataset_types)
            # None means this dataset's type is structural/container (e.g. "Split")
            # and should not contribute a path component — traverse children
            # with the same parent_path so they get clean paths.
            if current_type is None:
                current_path = parent_path
            else:
                current_path = parent_path + [current_type]
            type_paths[dataset.dataset_rid] = current_path

            for child in dataset.list_dataset_children():
                traverse(child, current_path, visited)

        traverse(self, [], set())
        return type_paths

    def _get_asset_dataset_mapping(self, asset_table: str) -> dict[RID, RID]:
        """Map asset RIDs to their containing dataset RID.

        For each asset in the specified table, determines which dataset it belongs to.
        This uses _dataset_table_view to find assets reachable through any FK path
        from the dataset, not just directly associated assets.

        Assets are mapped to their most specific (leaf) dataset in the hierarchy.
        For example, if a Split dataset contains Training and Testing children,
        and images are members of Training, the images map to Training (not Split).

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

            # Process children FIRST (depth-first) so leaf datasets get priority
            # This ensures assets are mapped to their most specific dataset
            for child in dataset.list_dataset_children():
                collect_from_dataset(child, visited)

            # Then process this dataset's assets
            # Only set if not already mapped (child/leaf dataset wins)
            for asset in dataset._get_reachable_assets(asset_table):
                if asset["RID"] not in asset_to_dataset:
                    asset_to_dataset[asset["RID"]] = dataset.dataset_rid

        collect_from_dataset(self, set())
        return asset_to_dataset

    def _get_reachable_assets(self, asset_table: str) -> list[dict[str, Any]]:
        """Get all assets reachable from this dataset through any FK path.

        Unlike list_dataset_members which only returns directly associated entities,
        this method traverses foreign key relationships to find assets that are
        indirectly connected to the dataset. For example, if a dataset contains
        Subjects, and Subject -> Encounter -> Image, this method will find those
        Images even though they're not directly in the Dataset_Image association table.

        Args:
            asset_table: Name of the asset table (e.g., "Image")

        Returns:
            List of asset records as dictionaries.
        """
        # Use the _dataset_table_view query which traverses all FK paths
        sql_query = self._dataset_table_view(asset_table)

        with Session(self.engine) as session:
            result = session.execute(sql_query)
            # Convert rows to dictionaries
            rows = [dict(row._mapping) for row in result]

        return rows

    def _load_feature_values_cache(
        self,
        asset_table: str,
        group_keys: list[str],
        enforce_vocabulary: bool = True,
        value_selector: Callable | None = None,
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
            value_selector: Optional function to select which feature value to use
                when an asset has multiple values for the same feature. Receives a
                list of FeatureRecord objects and returns the selected one. If not
                provided and multiple values exist, raises DerivaMLException when
                enforce_vocabulary=True or uses the first value when False.

        Returns:
            Dictionary mapping group_key -> {target_rid -> feature_value}
            Only includes entries for keys that are actually features.

        Raises:
            DerivaMLException: If enforce_vocabulary is True and:
                - A feature has no term columns (not vocabulary-based), or
                - An asset has multiple different vocabulary term values for the same feature
                  and no value_selector is provided.
        """
        from deriva_ml.core.exceptions import DerivaMLException
        from deriva_ml.feature import FeatureRecord

        cache: dict[str, dict[RID, Any]] = {}
        # Store FeatureRecord objects directly for later selection
        records_cache: dict[str, dict[RID, list[FeatureRecord]]] = {}
        # Track which column to use for each group_key's value extraction
        column_for_group: dict[str, str] = {}
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

            # Track the column used for this group_key
            column_for_group[group_key] = use_column
            records_cache[group_key] = defaultdict(list)
            feature_values = self.list_feature_values(table_name, feat.feature_name)

            for fv in feature_values:
                target_rid = getattr(fv, table_name, None)
                if target_rid is None:
                    continue

                # Check the value column is populated
                value = getattr(fv, use_column, None)
                if value is None:
                    continue

                records_cache[group_key][target_rid].append(fv)

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

        # Now resolve multiple values using value_selector or error handling
        for group_key, target_records in records_cache.items():
            cache[group_key] = {}
            use_column = column_for_group[group_key]
            for target_rid, records in target_records.items():
                if len(records) == 1:
                    # Single value - straightforward
                    cache[group_key][target_rid] = getattr(records[0], use_column)
                elif len(records) > 1:
                    # Multiple values - need to resolve
                    unique_values = set(getattr(r, use_column) for r in records)
                    if len(unique_values) == 1:
                        # All records have same value, use it
                        cache[group_key][target_rid] = getattr(records[0], use_column)
                    elif value_selector:
                        # Use provided selector function
                        selected = value_selector(records)
                        cache[group_key][target_rid] = getattr(selected, use_column)
                    elif enforce_vocabulary:
                        # Multiple different values without selector - error
                        values_str = ", ".join(f"'{getattr(r, use_column)}' (exec: {r.Execution})" for r in records)
                        raise DerivaMLException(
                            f"Asset '{target_rid}' has multiple different values for "
                            f"feature '{records[0].Feature_Name}': {values_str}. "
                            f"Provide a value_selector function to choose between values, "
                            f"or set enforce_vocabulary=False to use the first value."
                        )
                    else:
                        # Not enforcing - use first value
                        cache[group_key][target_rid] = getattr(records[0], use_column)

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
            The resolved value as a string, or "Unknown" if not found or None.
            Uses "Unknown" (capitalized) to match vocabulary term naming conventions.
        """
        # First check if it's a direct column on the asset table
        if group_key in asset:
            value = asset[group_key]
            if value is not None:
                return str(value)
            return "Unknown"

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

        return "Unknown"

    def _detect_asset_table(self) -> str | None:
        """Auto-detect the asset table from dataset members.

        Searches for asset tables in the dataset members by examining
        the schema. Returns the first asset table found, or None if
        no asset tables are in the dataset.

        Returns:
            Name of the detected asset table, or None if not found.
        """
        members = self.list_dataset_members(recurse=True)
        for table_name in members:
            if table_name == "Dataset":
                continue
            # Check if this table is an asset table
            try:
                table = self.model.name_to_table(table_name)
                if self.model.is_asset(table):
                    return table_name
            except (KeyError, AttributeError):
                continue
        return None

    def _validate_dataset_types(self) -> list[str] | None:
        """Validate that the dataset or its children have Training/Testing types.

        Checks if this dataset is of type Training or Testing, or if it has
        nested children of those types. Returns the valid types found.

        Returns:
            List of Training/Testing type names found, or None if validation fails.
        """
        valid_types = {"Training", "Testing"}
        found_types: set[str] = set()

        def check_dataset(ds: DatasetBag, visited: set[RID]) -> None:
            if ds.dataset_rid in visited:
                return
            visited.add(ds.dataset_rid)

            for dtype in ds.dataset_types:
                if dtype in valid_types:
                    found_types.add(dtype)

            for child in ds.list_dataset_children():
                check_dataset(child, visited)

        check_dataset(self, set())
        return list(found_types) if found_types else None

    def restructure_assets(
        self,
        output_dir: Path | str,
        asset_table: str | None = None,
        group_by: list[str] | None = None,
        use_symlinks: bool = True,
        type_selector: Callable[[list[str]], str] | None = None,
        type_to_dir_map: dict[str, str] | None = None,
        enforce_vocabulary: bool = True,
        value_selector: Callable | None = None,
        file_transformer: Callable[[Path, Path], Path] | None = None,
    ) -> dict[Path, Path]:
        """Restructure downloaded assets into a directory hierarchy.

        Creates a directory structure organizing assets by dataset types and
        grouping values. This is useful for ML workflows that expect data
        organized in conventional folder structures (e.g., PyTorch ImageFolder).

        The dataset should be of type Training or Testing, or have nested
        children of those types. The top-level directory name is determined
        by the dataset type (e.g., "Training" -> "training").

        **Finding assets through foreign key relationships:**

        Assets are found by traversing all foreign key paths from the dataset,
        not just direct associations. For example, if a dataset contains Subjects,
        and the schema has Subject -> Encounter -> Image relationships, this method
        will find all Images reachable through those paths even though they are
        not directly in a Dataset_Image association table.

        **Handling datasets without types (prediction scenarios):**

        If a dataset has no type defined, it is treated as Testing. This is
        common for prediction/inference scenarios where you want to apply a
        trained model to new unlabeled data.

        **Handling missing labels:**

        If an asset doesn't have a value for a group_by key (e.g., no label
        assigned), it is placed in an "Unknown" directory. This allows
        restructure_assets to work with unlabeled data for prediction.

        Args:
            output_dir: Base directory for restructured assets.
            asset_table: Name of the asset table (e.g., "Image"). If None,
                auto-detects from dataset members. Raises DerivaMLException
                if multiple asset tables are found and none is specified.
            group_by: Names to group assets by. Each name creates a subdirectory
                level after the dataset type path. Names can be:

                - **Column names**: Direct columns on the asset table. The column
                  value becomes the subdirectory name.
                - **Feature names**: Features defined on the asset table (or tables
                  it references via foreign keys). The feature's vocabulary term
                  value becomes the subdirectory name.
                - **Feature.column**: Specify a particular column from a multi-term
                  feature (e.g., "Classification.Label" to use the Label column).

                Column names are checked first, then feature names. If a value
                is not found, "unknown" is used as the subdirectory name.

            use_symlinks: If True (default), create symlinks to original files.
                If False, copy files. Symlinks save disk space but require
                the original bag to remain in place. Ignored when
                ``file_transformer`` is provided.
            type_selector: Function to select type when dataset has multiple types.
                Receives list of type names, returns selected type name.
                Defaults to selecting first type or "unknown" if no types.
            type_to_dir_map: Optional mapping from dataset type names to directory
                names. Defaults to {"Training": "training", "Testing": "testing",
                "Unknown": "unknown"}. Use this to customize directory names or
                add new type mappings.
            enforce_vocabulary: If True (default), only allow features that have
                controlled vocabulary term columns, and raise an error if an asset
                has multiple different values for the same feature without a
                value_selector. This ensures clean, unambiguous directory structures.
                If False, allow any feature type and use the first value found
                when multiple values exist.
            value_selector: Optional function to select which feature value to use
                when an asset has multiple values for the same feature. Receives a
                list of FeatureRecord objects (typed Pydantic models with named
                attributes for each feature column) and returns the selected one.
                Use the Execution attribute to distinguish between values from
                different executions. Built-in selectors on FeatureRecord:
                ``select_newest``, ``select_first``, ``select_latest``,
                ``select_majority_vote(column)``.
            file_transformer: Optional callable invoked instead of the default
                symlink/copy step. Receives ``(src_path, dest_path)`` where
                ``dest_path`` is the suggested destination (preserving the original
                filename and extension). The transformer is responsible for writing
                the output file — it may change the extension or format — and must
                return the actual ``Path`` it wrote. When provided, ``use_symlinks``
                is ignored.

                Example — convert DICOM to PNG on placement::

                    def oct_to_png(src: Path, dest: Path) -> Path:
                        img = load_oct_dcm(str(src))
                        out = dest.with_suffix(".png")
                        PILImage.fromarray((img * 255).astype(np.uint8)).save(out)
                        return out

                    bag.restructure_assets(
                        output_dir="./ml_data",
                        group_by=["Diagnosis"],
                        file_transformer=oct_to_png,
                    )

        Returns:
            Manifest dict mapping each source ``Path`` to the actual output
            ``Path`` written. When no ``file_transformer`` is provided, source
            and output paths differ only in directory location. When a
            transformer is provided, the output path may also differ in name
            or extension.

        Raises:
            DerivaMLException: If asset_table cannot be determined (multiple
                asset tables exist without specification), if no valid dataset
                types (Training/Testing) are found, or if enforce_vocabulary
                is True and a feature has multiple values without value_selector.

        Examples:
            Basic restructuring with auto-detected asset table::

                manifest = bag.restructure_assets(
                    output_dir="./ml_data",
                    group_by=["Diagnosis"],
                )
                # Creates:
                # ./ml_data/training/Normal/image1.jpg
                # ./ml_data/testing/Abnormal/image2.jpg

            Custom type-to-directory mapping::

                manifest = bag.restructure_assets(
                    output_dir="./ml_data",
                    group_by=["Diagnosis"],
                    type_to_dir_map={"Training": "train", "Testing": "test"},
                )
                # Creates:
                # ./ml_data/train/Normal/image1.jpg
                # ./ml_data/test/Abnormal/image2.jpg

            Select specific feature column for multi-term features::

                manifest = bag.restructure_assets(
                    output_dir="./ml_data",
                    group_by=["Classification.Label"],  # Use Label column
                )

            Handle multiple feature values with a built-in selector::

                from deriva_ml.feature import FeatureRecord

                manifest = bag.restructure_assets(
                    output_dir="./ml_data",
                    group_by=["Diagnosis"],
                    value_selector=FeatureRecord.select_newest,
                )

            Prediction scenario with unlabeled data::

                # Dataset has no type - treated as Testing
                # Assets have no labels - placed in Unknown directory
                manifest = bag.restructure_assets(
                    output_dir="./prediction_data",
                    group_by=["Diagnosis"],
                )
                # Creates:
                # ./prediction_data/testing/Unknown/image1.jpg
                # ./prediction_data/testing/Unknown/image2.jpg

            Convert DICOM files to PNG during restructuring::

                from PIL import Image as PILImage

                def oct_to_png(src: Path, dest: Path) -> Path:
                    img = load_oct_dcm(str(src))
                    out = dest.with_suffix(".png")
                    PILImage.fromarray((img * 255).astype(np.uint8)).save(out)
                    return out

                manifest = bag.restructure_assets(
                    output_dir="./ml_data",
                    asset_table="OCT_DICOM",
                    group_by=["Image_Diagnosis.Diagnosis_Image"],
                    type_to_dir_map={"Training": "train", "Testing": "test"},
                    file_transformer=oct_to_png,
                )
                # manifest maps each source .dcm Path to its output .png Path:
                # Path(".../bag/OCT/image1.dcm") -> Path("./ml_data/train/Normal/image1.png")
        """
        logger = logging.getLogger("deriva_ml")
        group_by = group_by or []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Default type-to-directory mapping
        if type_to_dir_map is None:
            type_to_dir_map = {"Training": "training", "Testing": "testing", "Unknown": "unknown"}

        # Auto-detect asset table if not provided
        if asset_table is None:
            asset_table = self._detect_asset_table()
            if asset_table is None:
                raise DerivaMLException(
                    "Could not auto-detect asset table. No asset tables found in dataset members. "
                    "Specify the asset_table parameter explicitly."
                )
            logger.info(f"Auto-detected asset table: {asset_table}")

        # Step 1: Build dataset type path map with directory name mapping
        def map_type_to_dir(types: list[str]) -> str | None:
            """Map dataset types to directory name using type_to_dir_map.

            If dataset has no types, treat it as Testing (prediction use case).
            Returns None when the type is not in type_to_dir_map, signalling
            that this dataset is a structural container (e.g. a Split parent)
            and should not contribute a path component. Its children will
            still be traversed and their own types will determine the path.
            """
            if not types:
                # No types defined - treat as Testing for prediction scenarios
                return type_to_dir_map.get("Testing", "testing")
            if type_selector:
                selected_type = type_selector(types)
            else:
                selected_type = types[0]
            if selected_type in type_to_dir_map:
                return type_to_dir_map[selected_type]
            # Type not explicitly mapped — treat as transparent container
            return None

        type_path_map = self._build_dataset_type_path_map(map_type_to_dir)

        # Step 2: Get asset-to-dataset mapping
        asset_dataset_map = self._get_asset_dataset_mapping(asset_table)

        # Step 3: Load feature values cache for relevant features
        feature_cache = self._load_feature_values_cache(asset_table, group_by, enforce_vocabulary, value_selector)

        # Step 4: Get all assets reachable through FK paths
        # This uses _get_reachable_assets which traverses FK relationships,
        # so assets connected via Subject -> Encounter -> Image are found
        # even if the dataset only contains Subjects directly.
        assets = self._get_reachable_assets(asset_table)

        manifest: dict[Path, Path] = {}

        if not assets:
            logger.warning(f"No assets found in table '{asset_table}'")
            return manifest

        # Step 5: Process each asset
        for asset in assets:
            # Get source file path
            filename = asset.get("Filename")
            if not filename:
                logger.warning(f"Asset {asset.get('RID')} has no Filename")
                continue

            source_path = Path(filename)
            if not source_path.exists():
                # Filename may be a bare basename stored in the SQLite cache
                # before image materialization.  Fall back to the canonical
                # BDBag asset layout: data/asset/{RID}/{table}/{filename}.
                try:
                    bag_root = Path(self._catalog._database_model.bag_path)
                    source_path = bag_root / "data" / "asset" / asset.get("RID", "") / asset_table / Path(filename).name
                except AttributeError:
                    pass  # catalog doesn't have _database_model (e.g. in tests)

            if not source_path.exists():
                logger.warning(f"Asset file not found: {filename}")
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

            # Suggested destination preserves the original filename
            target_path = target_dir / source_path.name

            # Handle existing files at the suggested destination
            if target_path.exists() or target_path.is_symlink():
                target_path.unlink()

            if file_transformer is not None:
                # Transformer is responsible for writing the output file.
                # It receives the suggested dest and returns the actual path written,
                # which may differ in name or extension (e.g. DICOM -> PNG).
                actual_path = file_transformer(source_path, target_path)
            elif use_symlinks:
                try:
                    target_path.symlink_to(source_path.resolve())
                except OSError as e:
                    # Fall back to copy on platforms that don't support symlinks
                    logger.warning(f"Symlink failed, falling back to copy: {e}")
                    shutil.copy2(source_path, target_path)
                actual_path = target_path
            else:
                shutil.copy2(source_path, target_path)
                actual_path = target_path

            manifest[source_path] = actual_path

        return manifest


# Note: validate_call decorators with Self return types were removed because
# Pydantic doesn't support typing.Self in validate_call contexts.
