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
    >>> bag = ml.download_dataset_bag(dataset_spec)  # doctest: +SKIP
    >>> # List dataset members by type
    >>> members = bag.list_dataset_members(recurse=True)  # doctest: +SKIP
    >>> for image in members.get("Image", []):  # doctest: +SKIP
    ...     print(image["Filename"])
"""

from __future__ import annotations

# Standard library imports
import logging
import shutil
from collections import defaultdict
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterable, Literal, Self, cast

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
from deriva_ml.core.pd_utils import rows_to_dataframe
from deriva_ml.dataset.aux_classes import DatasetHistory, DatasetVersion
from deriva_ml.feature import Feature, FeatureRecord

if TYPE_CHECKING:
    import tensorflow as tf
    import torch.utils.data

    from deriva_ml.dataset.target_resolution import FeatureSelector
    from deriva_ml.model.deriva_ml_database import DerivaMLDatabase

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


def _default_dir_name_from_target(
    target: "FeatureRecord | dict[str, FeatureRecord] | str | None",
    targets: "list[str] | dict[str, Any] | None",
) -> str:
    """Derive a directory name string from a resolved target without target_transform.

    For single-target with a single-column FeatureRecord: returns the value of the
    first non-FK, non-metadata column (the term column).

    For multi-target or multi-column features: raises DerivaMLException explaining
    that target_transform is required.

    For plain string values (column targets already converted to str): returns as-is.

    Args:
        target: The resolved target from _resolve_targets or column lookup.
        targets: The original targets spec (for error messages).

    Returns:
        Directory name string.

    Raises:
        DerivaMLException: When the target is a dict (multi-target case) and
            no target_transform was provided.
    """
    from deriva_ml.core.exceptions import DerivaMLException

    if target is None:
        return "Unknown"
    if isinstance(target, str):
        return target
    if isinstance(target, dict):
        # Multi-target — can't auto-derive a single string
        raise DerivaMLException(
            f"restructure_assets with multi-target {list(targets)!r} requires "
            f"target_transform to derive a single directory name. "
            f"Provide target_transform=lambda rec: ... that returns a str."
        )
    # Single FeatureRecord — find the first term/value column that has a value
    record_data = target.model_dump()
    # Skip well-known metadata columns to find the label column
    _skip_cols = {"RID", "RCT", "RMT", "RCB", "RMB", "Feature_Name", "Execution"}
    for col, val in record_data.items():
        if col in _skip_cols:
            continue
        if val is not None and not isinstance(val, (dict, list)):
            return str(val)
    return "Unknown"


class DatasetBag:
    """Read-only interface to a downloaded dataset bag.

    DatasetBag manages access to a materialized BDBag (Big Data Bag) that contains
    a snapshot of dataset data from a Deriva catalog. It provides methods for:

    - Listing dataset members and their attributes
    - Navigating dataset relationships (parents, children)
    - Accessing feature values
    - Denormalizing data across related tables
    - Feeding the bag to training frameworks via ``as_torch_dataset`` /
      ``as_tf_dataset`` (framework adapters), or rewriting its layout via
      ``restructure_assets`` for tools that expect a class-folder directory
      tree. All three share the same ``targets`` / ``target_transform`` /
      ``missing`` vocabulary; see the User Guide "How to feed a bag to a
      training framework" section.

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
        >>> bag = dataset.download_dataset_bag(version="1.0.0")  # doctest: +SKIP
        >>> # List members by type
        >>> members = bag.list_dataset_members()  # doctest: +SKIP
        >>> for image in members.get("Image", []):  # doctest: +SKIP
        ...     print(f"File: {image['Filename']}")
        >>> # Navigate to nested datasets
        >>> for child in bag.list_dataset_children():  # doctest: +SKIP
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

    @property
    def path(self) -> Path:
        """Filesystem path to this bag's root directory.

        The bag is a self-contained, immutable snapshot on disk. ``path``
        is the directory containing ``data/``, ``manifest-md5.txt``, and
        the bag's SQLite database. Use it to:

        - Read materialized asset files relative to the bag.
        - Diagnose "which bag is this?" errors in logs.
        - Archive or copy the bag to a new location.

        The directory exists for the lifetime of the bag object. Do not
        mutate anything inside it — bags are immutable by contract.

        Returns:
            Path: Root directory of the materialized bag on disk.

        Example:
            >>> spec = DatasetSpec(rid="1-abc123", version="1.2.0")
            >>> bag = ml.download_dataset_bag(spec)
            >>> print(f"Bag materialized at {bag.path}")
            >>> # Read an asset file relative to the bag root
            >>> manifest = (bag.path / "manifest-md5.txt").read_text()
        """
        return self.model.bag_path

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
            >>> for subject in bag.get_table_as_dict("Subject"):  # doctest: +SKIP
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
            >>> df = bag.get_table_as_dataframe("Image")  # doctest: +SKIP
            >>> print(df.shape)  # doctest: +SKIP
        """
        return rows_to_dataframe(self.get_table_as_dict(table))

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
        # Use UNION (not UNION ALL) to deduplicate rows that are reachable via
        # multiple FK paths. SQLAlchemy's union() is DISTINCT by default, which
        # is exactly what we want here: a dataset member table (e.g., Image) may
        # appear in the bag via two separate FK paths (e.g., directly in the
        # dataset AND via a nested child dataset), and without DISTINCT we would
        # count or return the same row twice. See §6 inline comment gap #4.
        return union(*sql_cmds)

    def dataset_history(self) -> list[DatasetHistory]:
        """Retrieve the version history of this dataset from the bag.

        Returns a list of all recorded versions for this dataset, read from
        the bag's local SQLite ``Dataset_Version`` table.

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
            DerivaMLException: If the bag SQLite database cannot be read.

        Example:
            >>> history = bag.dataset_history()  # doctest: +SKIP
            >>> for entry in history:  # doctest: +SKIP
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
        """Return all members of this dataset bag grouped by table name.

        Queries the local SQLite replica of the downloaded bag. Each key
        in the returned dict is a table name (e.g. ``"Image"``); each value
        is a list of row dicts with the full set of columns for that table.

        Args:
            recurse: If ``True``, recursively include members from nested
                child datasets. Default is ``False``.
            limit: Maximum number of members to return per table. ``None``
                (default) returns all rows.
            _visited: Internal parameter to track visited datasets and prevent
                infinite recursion. Callers should not pass this.
            version: Dataset version string (e.g. ``"1.2.0"``) to query.
                When ``None`` (default), uses the latest materialized version
                in the bag. This parameter exists for API symmetry with the
                live-catalog ``Dataset.list_dataset_members``; bag contents
                are immutable so changing ``version`` only filters which
                version's membership snapshot is read.
            **kwargs: Additional arguments (ignored, for protocol compatibility).

        Returns:
            Dict mapping table name to list of row dicts. Empty dict if
            no members are present.

        Raises:
            DerivaMLException: If the bag SQLite database cannot be read
                or the requested version is not present in the bag.

        Example:
            >>> bag = ml.download_dataset_bag(spec)  # doctest: +SKIP
            >>> members = bag.list_dataset_members(recurse=True)  # doctest: +SKIP
            >>> images = members.get("Image", [])  # doctest: +SKIP
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
            >>> for f in bag.find_features("Image"):  # doctest: +SKIP
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
            >>> from deriva_ml.feature import FeatureRecord  # doctest: +SKIP
            >>> for rec in bag.feature_values("Image", "Glaucoma"):  # doctest: +SKIP
            ...     print(rec.Image, rec.Glaucoma)
            >>> # With selector — one record per image, most recent wins:
            >>> records = list(bag.feature_values(  # doctest: +SKIP
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
            >>> feat = bag.lookup_feature("Image", "Glaucoma")  # doctest: +SKIP
            >>> RecordClass = feat.feature_record_class()  # doctest: +SKIP
            >>> record = RecordClass(Image="1-ABC", Glaucoma="Normal")  # doctest: +SKIP
            >>> print(record.Glaucoma)  # doctest: +SKIP
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
            >>> rids = bag.list_workflow_executions("Glaucoma_Training_v2")  # doctest: +SKIP
            >>> print(len(rids))  # doctest: +SKIP
            3
        """
        from sqlalchemy import select as sa_select

        workflow_table = self.model.find_table("Workflow")
        execution_table = self.model.find_table("Execution")

        with Session(self.engine) as session:
            # Phase 1: try as a Workflow RID.
            wf_rows = (
                session.execute(sa_select(workflow_table).where(workflow_table.c.RID == workflow)).mappings().all()
            )

            if wf_rows:
                rows = session.execute(
                    sa_select(execution_table.c.RID).where(execution_table.c.Workflow == workflow)
                ).all()
                return [r[0] for r in rows]

            # Phase 2: fall back to Workflow_Type name via association table.
            wwt = self.model.find_table("Workflow_Workflow_Type")
            wf_of_type = [
                r[0] for r in session.execute(sa_select(wwt.c.Workflow).where(wwt.c.Workflow_Type == workflow)).all()
            ]
            if not wf_of_type:
                raise DerivaMLException(
                    f"No workflow resolved for '{workflow}' in bag — tried as Workflow RID and Workflow_Type name."
                )
            rows = session.execute(
                sa_select(execution_table.c.RID).where(execution_table.c.Workflow.in_(wf_of_type))
            ).all()
            return [r[0] for r in rows]

    def fetch_table_features(self, *args, **kwargs):
        """Retired — use ``feature_values(table, name)`` or ``Denormalizer``.

        ``DatasetBag.fetch_table_features`` has been removed. Use the new
        ``feature_values`` method to read a single feature::

            for rec in bag.feature_values("Image", "Quality"):
                ...

        For wide-table denormalization across all features use the
        ``Denormalizer`` subsystem.

        Raises:
            DerivaMLException: Always. Points at the replacement API.
        """
        raise DerivaMLException(
            "DatasetBag.fetch_table_features() has been retired. "
            "Use feature_values(table, feature_name) to read a single feature, "
            "or Denormalizer for multi-feature wide tables."
        )

    def list_feature_values(self, *args, **kwargs) -> Iterable[FeatureRecord]:
        """Retired — renamed to ``feature_values``.

        ``DatasetBag.list_feature_values`` has been removed. Use the new
        ``feature_values`` method instead::

            for rec in bag.feature_values("Image", "Quality"):
                ...

        The signature is identical (``table``, ``feature_name``, optional
        ``selector``).

        Raises:
            DerivaMLException: Always. Points at the replacement API.
        """
        raise DerivaMLException(
            "DatasetBag.list_feature_values() has been retired and renamed. "
            "Use feature_values(table, feature_name, selector=...) instead."
        )

    def list_dataset_element_types(self) -> Iterable[Table]:
        """List the ERMrest Table objects that can be members of a dataset.

        Delegates to the underlying ``DerivaMLDatabase`` to return all tables
        that are linked to the Dataset table via association tables in the bag.

        Returns:
            Iterable[Table]: ERMrest ``Table`` objects for each dataset-element
            table (e.g., Image, Subject, nested Dataset).

        Example:
            >>> for table in bag.list_dataset_element_types():  # doctest: +SKIP
            ...     print(table.name)
        """
        return self.model.list_dataset_element_types()

    def list_dataset_children(
        self,
        recurse: bool = False,
        _visited: set[RID] | None = None,
        version: Any = None,
        **kwargs: Any,
    ) -> list[Self]:
        """Return directly nested (child) datasets of this bag.

        Queries the bag's local SQLite ``Dataset_Dataset`` association table
        to find datasets nested inside this one.

        Args:
            recurse: If ``True``, recursively include children of children.
                Default is ``False``.
            _visited: Internal parameter tracking visited RIDs to guard
                against circular references. Callers should not pass this.
            version: Ignored (bags are immutable snapshots; present for
                API symmetry with ``Dataset.list_dataset_children``).
            **kwargs: Additional arguments (ignored, for protocol compatibility).

        Returns:
            List of ``DatasetBag`` instances for each direct child dataset,
            in the order returned by the SQLite query.

        Raises:
            DerivaMLException: If the bag SQLite database cannot be read.

        Example:
            >>> children = bag.list_dataset_children(recurse=True)  # doctest: +SKIP
            >>> for child in children:  # doctest: +SKIP
            ...     print(child.dataset_rid, child.dataset_types)
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
        """Return the parent datasets that contain this dataset as a nested member.

        Queries the bag's local SQLite ``Dataset_Dataset`` association table to
        find datasets in which this dataset appears as a ``Nested_Dataset``.

        Args:
            recurse: If ``True``, recursively return all ancestor datasets up
                to the root. Default is ``False``.
            _visited: Internal parameter tracking visited RIDs to guard against
                circular references. Callers should not pass this.
            version: Ignored (bags are immutable snapshots; present for
                API symmetry with ``Dataset.list_dataset_parents``).
            **kwargs: Additional arguments (ignored, for protocol compatibility).

        Returns:
            List of ``DatasetBag`` instances for each direct parent dataset.
            Empty list if this dataset has no parents in the bag.

        Raises:
            DerivaMLException: If the bag SQLite database cannot be read.

        Example:
            >>> parents = bag.list_dataset_parents()  # doctest: +SKIP
            >>> for p in parents:  # doctest: +SKIP
            ...     print(p.dataset_rid)
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
            >>> bag = ml.download_dataset_bag(dataset_spec)  # doctest: +SKIP
            >>> execution_rids = bag.list_executions()  # doctest: +SKIP
            >>> for rid in execution_rids:  # doctest: +SKIP
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
        """Dry-run the denormalization and return planning metadata.

        Shortcut for
        :meth:`~deriva_ml.local_db.denormalizer.Denormalizer.describe` —
        returns the full plan dict (see that method's docstring for the
        exact 12-key shape). Never raises on ambiguity.

        Args:
            include_tables: Tables whose columns would appear in the output.
            row_per: Optional explicit leaf table (Rule 2).
            via: Optional path-only intermediates (Rule 6).

        Returns:
            dict: Planning metadata with 12 keys including ``anchor``,
            ``row_per``, ``join_path``, ``columns``, ``ambiguities``,
            and related diagnostics. See ``Denormalizer.describe`` for
            the full shape.

        Example::

            plan = bag.describe_denormalized(["Image", "Subject"])
            print(plan["anchor"], plan["row_per"])
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
            feature_values = self.feature_values(table_name, feat.feature_name)

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

    def as_torch_dataset(
        self,
        element_type: str,
        *,
        sample_loader: Callable[[Path | None, dict[str, Any]], Any] | None = None,
        transform: Callable[[Any], Any] | None = None,
        targets: list[str] | dict[str, Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        missing: Literal["error", "skip", "unknown"] = "error",
    ) -> "torch.utils.data.Dataset":
        """Build a ``torch.utils.data.Dataset`` from this bag.

        Creates a lazy PyTorch dataset that reads samples and labels from
        this already-downloaded ``DatasetBag``. The dataset's
        ``__getitem__`` returns individual samples (and optionally labels)
        without materializing the entire dataset into memory at construction
        time. Torch is imported lazily inside the builder so the base
        library stays importable without torch installed.

        This is the recommended path from a ``DatasetBag`` to a
        ``torch.utils.data.DataLoader`` for custom training loops and
        models. For workflows that need the ``ImageFolder``-style class-
        folder directory layout (e.g., ``torchvision.datasets.ImageFolder``
        or third-party fine-tuning scripts), use ``restructure_assets()``
        instead — the two tools are alternatives, not a pipeline.

        Labels come from the bag's feature values via
        ``bag.feature_values(element_type, feature_name, selector=...)``.
        The user's ``target_transform`` maps the typed ``FeatureRecord``
        into whatever numeric shape the loss function expects. The library
        does not hold or auto-fit a class-to-index table (design anchor 2).

        Args:
            element_type: Name of the domain table whose rows become the
                dataset's samples (e.g., ``"Image"``). Must be a table
                present in the bag (same error path as
                ``list_dataset_members``). Whether ``sample_loader`` is
                required depends on whether this is an asset table — see
                the ``sample_loader`` entry below.

            sample_loader: ``Callable[[Path | None, dict], Any]``. Invoked
                once per ``__getitem__`` call. Receives:

                - ``Path | None`` — absolute filesystem path under
                  ``bag.path / "data/assets/<element_type>/<rid>/<filename>"``
                  when ``element_type`` is an asset table; ``None``
                  otherwise.
                - ``dict[str, Any]`` — the raw row dict from the element
                  table (all columns, not just those the framework needs).

                For **asset-table** ``element_type``: no default — raises
                ``DerivaMLException`` at construction if ``sample_loader``
                is ``None``. The error message names common loaders
                (``PIL.Image.open``, ``nibabel.load``, ``h5py.File``) as
                hints.

                For **non-asset-table** ``element_type``: default returns
                ``row_dict`` unchanged. Useful for tabular training where
                the element IS the row data with no file decoding.

            transform: ``Callable[[Any], Any]``. Applied to the sample
                after ``sample_loader`` returns. Standard torchvision-style
                transform pipeline goes here (e.g.,
                ``torchvision.transforms.Compose([...])``). No-op default.

            targets: Source of label data. Three shapes are accepted:

                - ``None`` (default) — unlabeled dataset. ``__getitem__``
                  returns just the sample (not a tuple). Useful for
                  inference loops and self-supervised pretext tasks.
                - ``list[str]`` — feature names read via
                  ``bag.feature_values(element_type, name)`` with no
                  selector. One ``FeatureRecord`` is resolved per element.
                - ``dict[str, FeatureSelector]`` — feature names mapped to
                  selector callables, passed verbatim to
                  ``bag.feature_values(..., selector=...)``. Resolves
                  multi-annotator cases. Built-in selectors:
                  ``FeatureRecord.select_newest``, ``select_first``, etc.

            target_transform: ``Callable`` consuming the raw feature-record
                shape (see target arity below) and returning whatever the
                user's loss function expects (typically an ``int`` or a
                ``torch.Tensor``). No-op default returns the feature
                record(s) as-is.

                Target arity:

                - Single-target (``targets=["A"]``): receives a
                  ``FeatureRecord`` directly.
                - Multi-target (``targets=["A", "B"]``): receives
                  ``dict[str, FeatureRecord]`` keyed by feature name.

            missing: Behavior when a feature value is absent for an element:

                - ``"error"`` (default) — raise ``DerivaMLException`` at
                  adapter construction time, before any ``__getitem__``
                  call. Message includes the list of unlabeled RIDs.
                  Explicit-over-silent: users should know if their dataset
                  is partially labeled.
                - ``"skip"`` — drop unlabeled elements from the dataset
                  entirely. The resulting dataset's ``__len__`` reflects
                  only labeled elements. Index mapping is stable across the
                  dataset's lifetime.
                - ``"unknown"`` — keep all elements; pass ``None`` to the
                  user's ``target_transform`` for unlabeled ones. The
                  ``target_transform`` must handle ``None`` (typically
                  returning a sentinel class index or an ignore-mask value).
                  Useful for semi-supervised and self-training workflows.

        Returns:
            A ``torch.utils.data.Dataset`` whose ``__getitem__`` returns:

            - ``sample`` when ``targets=None`` (unlabeled).
            - ``(sample, target)`` when ``targets`` is set, where
              ``sample = transform(sample_loader(path, row_dict))`` and
              ``target = target_transform(feature_record_shape)``.

            ``__len__`` equals the count of labeled elements when
            ``missing="skip"``, the total element count otherwise.

        Raises:
            ImportError: If PyTorch is not installed. Install with
                ``pip install 'deriva-ml[torch]'`` or
                ``pip install 'torch>=2.0'``.
            DerivaMLException: If ``element_type`` is not in the bag,
                if ``element_type`` is an asset table and ``sample_loader``
                is ``None``, or if ``missing="error"`` and any element
                lacks a feature value (message lists up to 20 unlabeled
                RIDs).
            FileNotFoundError: On ``__getitem__`` if the asset file is
                missing on disk (bag corrupted or removed after
                construction). This is the standard torch convention —
                the library does not retry or fall back.

        Example:
            >>> # Simple image classification with a single feature label:
            >>> import PIL.Image  # doctest: +SKIP
            >>> from torch.utils.data import DataLoader  # doctest: +SKIP
            >>> bag = ml.download_dataset_bag(version="1.0.0")  # doctest: +SKIP
            >>> ds = bag.as_torch_dataset(  # doctest: +SKIP
            ...     element_type="Image",
            ...     sample_loader=lambda p, row: PIL.Image.open(p).convert("RGB"),
            ...     targets=["Glaucoma_Grade"],
            ...     target_transform=lambda rec: CLASS_TO_IDX[rec.Grade],
            ... )
            >>> loader = DataLoader(ds, batch_size=32, shuffle=True)  # doctest: +SKIP

            >>> # Pure-Python assertion — runs for real:
            >>> from deriva_ml.dataset.torch_adapter import build_torch_dataset
            >>> callable(build_torch_dataset)
            True
        """
        from deriva_ml.dataset.torch_adapter import build_torch_dataset

        return build_torch_dataset(
            self,
            element_type,
            sample_loader=sample_loader,
            transform=transform,
            targets=targets,
            target_transform=target_transform,
            missing=missing,
        )

    def as_tf_dataset(
        self,
        element_type: str,
        *,
        sample_loader: Callable[[Path | None, dict[str, Any]], Any] | None = None,
        transform: Callable[[Any], Any] | None = None,
        targets: list[str] | dict[str, Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        missing: Literal["error", "skip", "unknown"] = "error",
        output_signature: "tf.TensorSpec | tuple[tf.TensorSpec, ...] | None" = None,
    ) -> "tf.data.Dataset":
        """Build a ``tf.data.Dataset`` from this bag.

        Creates a ``tf.data.Dataset`` backed by a Python generator that
        reads samples and labels from this already-downloaded
        ``DatasetBag``. TensorFlow is imported lazily inside the builder
        so the base library stays importable without TensorFlow installed.

        Each call to the generator yields one element (sample, or
        ``(sample, target)`` when ``targets`` is supplied). Callers are
        responsible for chaining ``.batch()`` and ``.prefetch()`` to get
        production throughput — the method does not apply batching itself.

        Labels come from the bag's feature values via
        ``bag.feature_values(element_type, feature_name, selector=...)``.
        The user's ``target_transform`` maps the typed ``FeatureRecord``
        into whatever numeric shape the loss function expects. The library
        does not hold or auto-fit a class-to-index table (design anchor 2).

        Args:
            element_type: Name of the domain table whose rows become the
                dataset's samples (e.g., ``"Image"``). Must be a table
                present in the bag (same error path as
                ``list_dataset_members``). Whether ``sample_loader`` is
                required depends on whether this is an asset table — see
                the ``sample_loader`` entry below.

            sample_loader: ``Callable[[Path | None, dict], Any]``. Invoked
                once per generated element. Receives:

                - ``Path | None`` — absolute filesystem path under
                  ``bag.path / "data/assets/<element_type>/<rid>/<filename>"``
                  when ``element_type`` is an asset table; ``None``
                  otherwise.
                - ``dict[str, Any]`` — the raw row dict from the element
                  table (all columns, not just those the framework needs).

                For **asset-table** ``element_type``: no default — raises
                ``DerivaMLException`` at construction if ``sample_loader``
                is ``None``. The error message names common loaders
                (``PIL.Image.open``, ``nibabel.load``, ``h5py.File``) as
                hints.

                For **non-asset-table** ``element_type``: default returns
                ``row_dict`` unchanged. Useful for tabular training where
                the element IS the row data with no file decoding.

            transform: ``Callable[[Any], Any]``. Applied to the sample
                after ``sample_loader`` returns. Use this for any
                preprocessing (e.g., resizing, normalization, conversion
                to ``tf.Tensor``). No-op default.

            targets: Source of label data. Three shapes are accepted:

                - ``None`` (default) — unlabeled dataset. Each element is
                  just the sample (not a tuple). Useful for inference loops
                  and self-supervised pretext tasks.
                - ``list[str]`` — feature names read via
                  ``bag.feature_values(element_type, name)`` with no
                  selector. One ``FeatureRecord`` is resolved per element.
                - ``dict[str, FeatureSelector]`` — feature names mapped to
                  selector callables, passed verbatim to
                  ``bag.feature_values(..., selector=...)``. Resolves
                  multi-annotator cases. Built-in selectors:
                  ``FeatureRecord.select_newest``, ``select_first``, etc.

            target_transform: ``Callable`` consuming the raw feature-record
                shape (see target arity below) and returning whatever the
                user's loss function expects (typically a ``tf.Tensor``).
                No-op default returns the feature record(s) as-is.

                Target arity:

                - Single-target (``targets=["A"]``): receives a
                  ``FeatureRecord`` directly.
                - Multi-target (``targets=["A", "B"]``): receives
                  ``dict[str, FeatureRecord]`` keyed by feature name.

            missing: Behavior when a feature value is absent for an element:

                - ``"error"`` (default) — raise ``DerivaMLException`` at
                  adapter construction time, before any element is generated.
                  Message includes the list of unlabeled RIDs.
                  Explicit-over-silent: users should know if their dataset
                  is partially labeled.
                - ``"skip"`` — drop unlabeled elements from the dataset
                  entirely. Only labeled elements are yielded.
                - ``"unknown"`` — keep all elements; pass ``None`` to the
                  user's ``target_transform`` for unlabeled ones. The
                  ``target_transform`` must handle ``None`` (typically
                  returning a sentinel class index or an ignore-mask value).
                  Useful for semi-supervised and self-training workflows.

            output_signature: ``tf.TypeSpec`` (or nested structure of specs
                such as ``(tf.TensorSpec(...), tf.TensorSpec(...))``)
                describing the shape and dtype of each element produced by
                the generator. When ``None`` (default), the first sample is
                consumed eagerly to infer the signature via
                ``tf.type_spec_from_value``, then the generator is
                re-wrapped so the first sample is not lost. Providing an
                explicit signature avoids the eager-inference overhead and
                is preferred for production use.

        Returns:
            A ``tf.data.Dataset`` whose elements are:

            - ``sample`` when ``targets=None`` (unlabeled).
            - ``(sample, target)`` when ``targets`` is set, where
              ``sample = transform(sample_loader(path, row_dict))`` and
              ``target = target_transform(feature_record_shape)``.

            Callers must chain ``.batch()`` / ``.prefetch()`` themselves —
            the returned dataset is unbatched.

        Raises:
            ImportError: If TensorFlow is not installed. Install with
                ``pip install 'deriva-ml[tf]'`` or
                ``pip install 'tensorflow>=2.15'``.
            DerivaMLException: If ``element_type`` is not in the bag,
                if ``element_type`` is an asset table and ``sample_loader``
                is ``None``, if ``missing="error"`` and any element lacks
                a feature value (message lists up to 20 unlabeled RIDs),
                or if ``output_signature=None`` and the generator produces
                no elements (cannot infer signature).
            FileNotFoundError: During iteration if the asset file is
                missing on disk (bag corrupted or removed after
                construction).

        Example:
            >>> # Simple image classification with a single feature label:
            >>> import PIL.Image  # doctest: +SKIP
            >>> bag = ml.download_dataset_bag(version="1.0.0")  # doctest: +SKIP
            >>> ds = bag.as_tf_dataset(  # doctest: +SKIP
            ...     element_type="Image",
            ...     sample_loader=lambda p, row: tf.image.decode_jpeg(
            ...         tf.io.read_file(str(p))
            ...     ),
            ...     targets=["Glaucoma_Grade"],
            ...     target_transform=lambda rec: CLASS_TO_IDX[rec.Grade],
            ... )
            >>> for batch in ds.batch(32).prefetch(2):  # doctest: +SKIP
            ...     images, labels = batch

            >>> # Pure-Python assertion — runs for real:
            >>> from deriva_ml.dataset.tf_adapter import build_tf_dataset
            >>> callable(build_tf_dataset)
            True
        """
        from deriva_ml.dataset.tf_adapter import build_tf_dataset

        return build_tf_dataset(
            self,
            element_type,
            sample_loader=sample_loader,
            transform=transform,
            targets=targets,
            target_transform=target_transform,
            missing=missing,
            output_signature=output_signature,
        )

    def restructure_assets(
        self,
        output_dir: Path | str,
        *,
        asset_table: str | None = None,
        targets: "list[str] | dict[str, FeatureSelector] | None" = None,
        target_transform: Callable[..., str] | None = None,
        missing: Literal["error", "skip", "unknown"] = "unknown",
        use_symlinks: bool = True,
        type_selector: Callable[[list[str]], str] | None = None,
        type_to_dir_map: dict[str, str] | None = None,
        enforce_vocabulary: bool = True,
        file_transformer: Callable[[Path, Path], Path] | None = None,
    ) -> dict[Path, Path]:
        """Restructure downloaded assets into a directory hierarchy.

        Creates a directory structure organizing assets by dataset types and
        target label values. This is useful for ML workflows that expect data
        organized in conventional folder structures (e.g., PyTorch ImageFolder,
        ``torchvision.datasets.ImageFolder``).

        The dataset should be of type Training or Testing, or have nested
        children of those types. The top-level directory name is determined
        by the dataset type (e.g., ``"Training"`` → ``"training"``).

        **Finding assets through foreign key relationships:**

        Assets are found by traversing all foreign key paths from the dataset,
        not just direct associations. For example, if a dataset contains Subjects,
        and the schema has Subject → Encounter → Image relationships, this method
        will find all Images reachable through those paths even though they are
        not directly in a ``Dataset_Image`` association table.

        **Handling datasets without types (prediction scenarios):**

        If a dataset has no type defined, it is treated as Testing. This is
        common for prediction/inference scenarios where you want to apply a
        trained model to new unlabeled data.

        **Handling missing labels:**

        If an asset doesn't have a value for a requested target, the ``missing``
        parameter controls the behavior: ``"unknown"`` (default) places the asset
        in an ``"Unknown"`` directory; ``"skip"`` omits it from the output tree;
        ``"error"`` raises at construction time listing all unlabeled RIDs.

        Args:
            output_dir: Base directory for restructured assets.
            asset_table: Name of the asset table (e.g., ``"Image"``). If None,
                auto-detects from dataset members. Raises ``DerivaMLException``
                if multiple asset tables are found and none is specified.
            targets: Source of directory-naming label data. Three shapes:

                - ``None`` (default) — no label grouping. Assets are placed
                  directly under the type-derived directory with no further
                  subdirectory levels.
                - ``list[str]`` — feature names (or direct column names) to
                  group by. Each name adds one subdirectory level. For a
                  single feature the resolved ``FeatureRecord`` is passed to
                  ``target_transform`` (if provided). For multiple features
                  a ``dict[str, FeatureRecord]`` is passed.
                - ``dict[str, FeatureSelector]`` — feature names mapped to
                  per-feature selector callables; passed verbatim to
                  ``bag.feature_values(..., selector=...)``. Built-in
                  selectors: ``FeatureRecord.select_newest``,
                  ``select_first``, ``select_majority_vote(column)``.

                Column names (direct columns on the asset table, not features)
                are resolved via column lookup on the asset record. They are
                converted to strings for the directory name; ``target_transform``
                receives the raw column value (as a string).

                Dotted ``"Feature.column"`` syntax from earlier releases is
                **removed** — pass it as a target string with ``target_transform``
                instead (see Migration note below).

            target_transform: ``Callable`` consuming the resolved label shape
                (a ``FeatureRecord`` for single-target, ``dict[str, FeatureRecord]``
                for multi-target, or the raw column value string for column
                targets) and **returning a ``str``** used as the subdirectory
                name. No-op default derives the name from the feature's primary
                value column (single-target) or raises a clear error explaining
                that ``target_transform`` is required for multi-target and
                multi-column feature cases.

                Runtime constraint: the return type is checked at the first
                call; a non-``str`` return raises ``DerivaMLValidationError``
                with a message explaining the requirement.

            missing: Behavior when a target value is absent for an asset:

                - ``"unknown"`` (default) — place the asset in an ``Unknown/``
                  subdirectory. Preserves the pre-alignment behavior.
                - ``"skip"`` — omit the asset from the output tree entirely.
                  New behavior; no directory is created for it.
                - ``"error"`` — raise ``DerivaMLException`` at construction
                  time listing unlabeled RIDs. Useful for ensuring training
                  data is complete before committing to disk.

            use_symlinks: If True (default), create symlinks to original files.
                If False, copy files. Symlinks save disk space but require
                the original bag to remain in place. Ignored when
                ``file_transformer`` is provided.
            type_selector: Function to select type when dataset has multiple types.
                Receives list of type names, returns selected type name.
                Defaults to selecting first type or ``"testing"`` if no types.
            type_to_dir_map: Optional mapping from dataset type names to directory
                names. Defaults to ``{"Training": "training", "Testing": "testing",
                "Unknown": "unknown"}``. Use this to customize directory names or
                add new type mappings.
            enforce_vocabulary: If True (default), only allow features that have
                controlled vocabulary term columns, and raise an error if an asset
                has multiple different values for the same feature. Set to False
                to allow non-vocabulary features and use the first value when
                multiple exist.
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
                        targets=["Diagnosis"],
                        file_transformer=oct_to_png,
                    )

        Returns:
            Manifest dict mapping each source ``Path`` to the actual output
            ``Path`` written. When no ``file_transformer`` is provided, source
            and output paths differ only in directory location. When a
            transformer is provided, the output path may also differ in name
            or extension.

        Raises:
            DerivaMLException: If ``asset_table`` cannot be determined, if
                ``missing="error"`` and any asset lacks a target value, or if
                ``enforce_vocabulary=True`` and a feature has no vocabulary
                term columns.
            DerivaMLValidationError: If ``target_transform`` returns a
                non-``str`` value, or if a dotted ``"Feature.column"`` string
                is passed in ``targets``.

        Examples:
            Basic restructuring with auto-detected asset table::

                manifest = bag.restructure_assets(
                    output_dir="./ml_data",
                    targets=["Diagnosis"],
                )
                # Creates:
                # ./ml_data/training/Normal/image1.jpg
                # ./ml_data/testing/Abnormal/image2.jpg

            Custom type-to-directory mapping::

                manifest = bag.restructure_assets(
                    output_dir="./ml_data",
                    targets=["Diagnosis"],
                    type_to_dir_map={"Training": "train", "Testing": "test"},
                )
                # Creates:
                # ./ml_data/train/Normal/image1.jpg
                # ./ml_data/test/Abnormal/image2.jpg

            Per-feature selector for multi-annotator datasets::

                from deriva_ml.feature import FeatureRecord

                manifest = bag.restructure_assets(
                    output_dir="./ml_data",
                    targets={"Diagnosis": FeatureRecord.select_newest},
                )

            Extract a specific column from a multi-column feature::

                manifest = bag.restructure_assets(
                    output_dir="./ml_data",
                    targets=["Classification"],
                    target_transform=lambda rec: rec.Label,
                )

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
                    targets=["Diagnosis"],
                    type_to_dir_map={"Training": "train", "Testing": "test"},
                    file_transformer=oct_to_png,
                )

        Note:
            Migration note (from pre-D2 signature):

            - ``group_by=["Diagnosis"]`` → ``targets=["Diagnosis"]``
            - ``group_by=["Classification.Label"]`` →
              ``targets=["Classification"], target_transform=lambda rec: rec.Label``
            - ``value_selector=FeatureRecord.select_newest`` →
              ``targets={"Feature": FeatureRecord.select_newest}``

        See Also:
            ``DatasetBag.as_torch_dataset``, ``DatasetBag.as_tf_dataset``:
                Framework adapters. Use these when you want lazy in-place
                iteration and do NOT need a class-folder directory tree.
                They share the same ``targets`` / ``target_transform`` /
                ``missing`` vocabulary as ``restructure_assets``. The two
                paths are alternatives, not a pipeline — pick one per the
                User Guide "How to feed a bag to a training framework".
        """
        from deriva_ml.core.exceptions import DerivaMLValidationError
        from deriva_ml.dataset.target_resolution import _resolve_targets

        logger = logging.getLogger("deriva_ml")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Validate targets: dotted "Feature.column" syntax is removed
        if targets is not None:
            target_names = list(targets) if isinstance(targets, dict) else targets
            for t in target_names:
                if "." in t:
                    raise DerivaMLValidationError(
                        f"Dotted target syntax {t!r} is no longer supported. "
                        f"Replace with: targets=[{t.split('.')[0]!r}], "
                        f"target_transform=lambda rec: rec.{t.split('.')[1]}"
                    )

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

        # Step 3: Separate feature-based targets from column-based targets.
        # _resolve_targets only works with features (via bag.feature_values).
        # Direct column names on the asset table must be handled separately.
        feature_target_map: dict[str, Any] = {}
        column_targets: list[str] = []
        feature_targets_spec: "list[str] | dict[str, Any] | None" = None

        if targets is not None:
            target_names_list = list(targets) if isinstance(targets, dict) else list(targets)

            # Classify each target as feature or column by probing lookup_feature.
            feature_names: list[str] = []
            for t_name in target_names_list:
                try:
                    self.lookup_feature(asset_table, t_name)
                    feature_names.append(t_name)
                except Exception:
                    # Not a recognized feature on asset_table — treat as column
                    column_targets.append(t_name)

            # Build the feature-only targets spec to pass to _resolve_targets
            if feature_names:
                if isinstance(targets, dict):
                    feature_targets_spec = {k: v for k, v in targets.items() if k in feature_names}
                else:
                    feature_targets_spec = feature_names

            # Call _resolve_targets only for feature-based targets
            if feature_targets_spec:
                feature_target_map = _resolve_targets(
                    self,
                    asset_table,
                    targets=feature_targets_spec,
                    missing=missing,
                )

        # Step 4: For column-based targets, load a simple {rid: value_str} map
        # by scanning all asset records once.
        column_value_map: dict[str, dict[str, str]] = {col: {} for col in column_targets}

        # Step 5: Load feature values cache for enforce_vocabulary enforcement.
        # _load_feature_values_cache raises at load time if enforce_vocabulary=True
        # and a feature has no vocabulary term columns. We discard the cache dict
        # itself (naming comes from feature_target_map); this call is for its
        # validation side-effect only.
        group_keys_for_cache = [
            t for t in (list(targets) if isinstance(targets, list) else
                        list(targets.keys()) if isinstance(targets, dict) else [])
            if t not in column_targets
        ]
        if group_keys_for_cache:
            self._load_feature_values_cache(
                asset_table, group_keys_for_cache, enforce_vocabulary, None
            )

        # Step 6: Get all assets reachable through FK paths
        assets = self._get_reachable_assets(asset_table)

        manifest: dict[Path, Path] = {}

        if not assets:
            logger.warning(f"No assets found in table '{asset_table}'")
            return manifest

        # Populate column_value_map from the asset records
        for asset in assets:
            for col in column_targets:
                val = asset.get(col)
                if val is not None:
                    column_value_map[col][asset["RID"]] = str(val)

        # Step 7: Process each asset
        for asset in assets:
            asset_rid = asset.get("RID")

            # Get source file path
            filename = asset.get("Filename")
            if not filename:
                logger.warning(f"Asset {asset_rid} has no Filename")
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
            dataset_rid = asset_dataset_map.get(asset_rid)
            type_path = type_path_map.get(dataset_rid, ["unknown"])

            # Resolve grouping path components from targets.
            # Each target name contributes one directory level. Target names are
            # processed in order (column targets get column lookup; feature targets
            # get resolution from feature_target_map built by _resolve_targets above).
            group_path: list[str] = []
            skip_asset = False

            if targets is not None:
                target_names_list = list(targets) if isinstance(targets, dict) else list(targets)
                # For single-feature target, feature_target_map[rid] is FeatureRecord|None.
                # For multi-feature target, feature_target_map[rid] is dict[str, FeatureRecord].
                # We unpack accordingly when passing to target_transform or _default_dir_name_from_target.
                feature_raw = feature_target_map.get(asset_rid)

                # If the RID is absent from the feature_target_map entirely AND there are
                # feature targets, apply the missing policy at the group level.
                if feature_targets_spec and asset_rid not in feature_target_map:
                    if missing == "skip":
                        skip_asset = True
                    elif missing == "error":
                        raise DerivaMLException(
                            f"Asset {asset_rid!r} has no value for target feature(s). "
                            f"Pass missing='skip' to drop unlabeled assets, or "
                            f"missing='unknown' to place them in Unknown/."
                        )
                    else:
                        # missing="unknown": place in Unknown dir
                        # Build partial path from column targets, then add Unknown for features
                        for t_name in target_names_list:
                            if t_name in column_targets:
                                col_val = column_value_map.get(t_name, {}).get(asset_rid)
                                group_path.append(str(col_val) if col_val is not None else "Unknown")
                            else:
                                group_path.append("Unknown")
                elif not skip_asset:
                    # Normal path: resolve each target in order
                    for t_name in target_names_list:
                        if t_name in column_targets:
                            # Column-based target
                            col_val = column_value_map.get(t_name, {}).get(asset_rid)
                            if col_val is not None:
                                if target_transform is not None:
                                    dir_name = target_transform(col_val)
                                    if not isinstance(dir_name, str):
                                        raise DerivaMLValidationError(
                                            f"restructure_assets target_transform must return str "
                                            f"(for directory naming); got {type(dir_name).__name__}"
                                        )
                                else:
                                    dir_name = col_val
                            else:
                                # Missing column value
                                if missing == "error":
                                    raise DerivaMLException(
                                        f"Asset {asset_rid!r} has no value for column target {t_name!r}. "
                                        f"Pass missing='skip' to drop unlabeled assets, or "
                                        f"missing='unknown' to place them in Unknown/."
                                    )
                                elif missing == "skip":
                                    skip_asset = True
                                    break
                                else:
                                    dir_name = "Unknown"
                            group_path.append(dir_name)
                        else:
                            # Feature-based target: look up from feature_target_map
                            # feature_raw is either FeatureRecord (single) or
                            # dict[str, FeatureRecord] (multi-feature). Extract for this feature.
                            if feature_raw is None:
                                per_feature_val = None
                            elif isinstance(feature_raw, dict):
                                per_feature_val = feature_raw.get(t_name)
                            else:
                                # Single feature target — feature_raw IS the FeatureRecord
                                per_feature_val = feature_raw

                            if per_feature_val is None:
                                dir_name = "Unknown"
                            elif target_transform is not None:
                                dir_name = target_transform(per_feature_val)
                                if not isinstance(dir_name, str):
                                    raise DerivaMLValidationError(
                                        f"restructure_assets target_transform must return str "
                                        f"(for directory naming); got {type(dir_name).__name__}"
                                    )
                            else:
                                dir_name = _default_dir_name_from_target(per_feature_val, [t_name])
                            group_path.append(dir_name)

            if skip_asset:
                continue

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
