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
- Table-level access (get_table_as_dict, lookup_term) is on the catalog (DerivaMLBagView)

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
    from deriva_ml.model.deriva_ml_bag_view import DerivaMLBagView


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
    vocabulary terms, use the DerivaMLBagView class instead.

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
        catalog: "DerivaMLBagView",
        dataset_rid: RID | None = None,
        dataset_types: str | list[str] | None = None,
        description: str = "",
        execution_rid: RID | None = None,
    ):
        """Initialize a DatasetBag instance for a dataset within a downloaded bag.

        This mirrors the Dataset class initialization pattern, where both classes
        take a catalog-like object as their first argument for consistency.

        Args:
            catalog: The DerivaMLBagView instance providing access to the bag's data.
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
        self.execution_rid = execution_rid or (self._catalog._get_dataset_execution(self.dataset_rid) or {}).get(
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
        is the BDBag directory containing ``data/`` (the CSV tables and
        materialized asset files) and ``manifest-md5.txt``. The SQLite
        database that backs queries is **not** inside this directory — it
        lives in a separate ``databases/`` subtree of the cache root
        (the cache is content-addressed by BDBag checksum per ADR-0006).
        Use ``path`` to:

        - Read materialized asset files relative to the bag.
        - Diagnose "which bag is this?" errors in logs.
        - Archive or copy the bag to a new location.

        The directory exists for the lifetime of the bag object. Do not
        mutate anything inside it — bags are immutable by contract.

        Returns:
            Path: Root directory of the materialized BDBag on disk
            (parent of ``data/``). The SQLite mirror is stored
            elsewhere, under the cache's ``databases/`` subtree.

        Example:
            >>> spec = DatasetSpec(rid="1-abc123", version="1.2.0")  # doctest: +SKIP
            >>> bag = ml.download_dataset_bag(spec)  # doctest: +SKIP
            >>> print(f"Bag materialized at {bag.path}")  # doctest: +SKIP
            >>> # Read an asset file relative to the bag root
            >>> manifest = (bag.path / "manifest-md5.txt").read_text()  # doctest: +SKIP
        """
        return self.model.bag_path

    @property
    def source_directory(self) -> str | None:
        """Source folder this directory dataset represents, relative to the
        ingest root.

        Returns the path stored in ``Directory_Dataset`` for this dataset (the
        ingest root stores ``"."``), or ``None`` if the dataset has no
        ``Directory_Dataset`` row — i.e. it was not created from a directory
        tree by :meth:`~deriva_ml.execution.execution.Execution.add_files`.

        The bag is offline — this reads the bag's local SQLite mirror of the
        ``Directory_Dataset`` table; no catalog connection is used.

        Returns:
            str | None: The relative source folder this directory dataset
            represents, relative to the ingest root, or None.

        Example:
            >>> root_bag = ml.download_dataset_bag(spec)  # doctest: +SKIP
            >>> root_bag.source_directory  # doctest: +SKIP
            '.'
            >>> [c.source_directory for c in root_bag.list_dataset_children()]  # doctest: +SKIP
            ['d1', 'd2']
        """
        try:
            rows = [
                r for r in self.model.get_table_contents("Directory_Dataset")
                if r["Dataset"] == self.dataset_rid
            ]
        except KeyError:
            # Directory_Dataset table absent from this bag (e.g. a pre-feature bag)
            return None
        return rows[0]["Path"] if rows else None

    @property
    def is_directory(self) -> bool:
        """Whether this dataset represents a source directory.

        ``True`` iff the dataset has a ``Directory_Dataset`` row (equivalently,
        :attr:`source_directory` is not ``None``) — i.e. it was created by
        :meth:`~deriva_ml.execution.execution.Execution.add_files` to mirror a
        folder. This is the authoritative predicate; it deliberately does NOT
        consult the ``Directory`` ``Dataset_Type`` tag, which can diverge from
        the path row for pre-feature or hand-tagged datasets.

        The bag is offline — this reads the bag's local SQLite mirror; no
        catalog connection is used.

        Returns:
            bool: True if this is a directory dataset.

        Example:
            >>> root_bag = ml.download_dataset_bag(spec)  # doctest: +SKIP
            >>> root_bag.is_directory  # doctest: +SKIP
            True
        """
        return self.source_directory is not None

    def materialize(self, *, fetch_concurrency: int = 8) -> Self:
        """Fetch any not-yet-downloaded files for this bag, in place.

        A :class:`DatasetBag` may be downloaded metadata-only (via
        ``download_dataset_bag(..., materialize=False)``) or be left
        partially materialized (``cached_holey``) after an interrupted
        fetch. This method completes it: it reads the bag's ``fetch.txt``
        (which carries absolute Hatrac/S3 URLs) and downloads every
        referenced file into the bag directory. No catalog connection is
        used — materialization is a pure local operation over the bag
        already on disk.

        The bag's :attr:`path` is unchanged; only the directory contents
        grow. The SQLite mirror is unaffected (it is built from the CSV
        tables already present in a metadata-only bag).

        The call is idempotent: a fully-materialized bag returns
        immediately without fetching.

        Args:
            fetch_concurrency: Maximum number of concurrent file
                downloads. Defaults to 8, matching ``download_dataset_bag`` /
                ``cache``; pass 1 for sequential downloads.

        Returns:
            Self: this same bag (its assets are now present on disk),
            so the call can be chained, e.g.
            ``bag = ml.download_dataset_bag(spec).materialize()``.

        Raises:
            Exception: Propagates any error raised by the underlying
                ``bdbag`` fetch — e.g. a ``fetch.txt`` URL that is
                unreachable (source store down, or asset bytes never
                uploaded to a reachable store). The bag is left
                partially materialized in that case.

        Example:
            >>> spec = DatasetSpec(rid="1-abc123", materialize=False)  # doctest: +SKIP
            >>> bag = ml.download_dataset_bag(spec)  # metadata only      # doctest: +SKIP
            >>> bag.materialize()  # fetch the asset bytes in place       # doctest: +SKIP
            <deriva_ml.DatasetBag object ...>
        """
        from deriva_ml.core.logging_config import get_logger
        from deriva_ml.dataset.bag_download import materialize_bag_dir

        materialize_bag_dir(
            self.path,
            fetch_concurrency=fetch_concurrency,
            logger=get_logger(__name__),
        )
        return self

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
        paths = [[t.name for t in p] for p in self.model._planner._schema_to_paths() if p[-1].name == table]

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
            for v in self.model.get_table_contents("Dataset_Version")
            if v["Dataset"] == self.dataset_rid
        ]

    def list_dataset_members(
        self,
        recurse: bool = False,
        limit: int | None = None,
        _visited: set[RID] | None = None,
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
            **kwargs: Accepted for ``DatasetLike`` protocol compatibility.
                ``Dataset`` honours a ``version=`` keyword to pin the
                catalog snapshot; bags are immutable, so any extras
                (including ``version=``) are silently ignored.

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

            assoc_class, dataset_rel, element_rel = self.model.get_association_class(dataset_class, element_class)

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

    def find_features(self, table: str | Table | None = None) -> Iterable[Feature]:
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
            table: The table to find features for (name or Table object). When
                ``None``, returns every feature in the bag's slice.

        Returns:
            An iterable of Feature instances describing each feature
            defined on the table (or every feature when ``table`` is ``None``).

        Example:
            >>> for f in bag.find_features("Image"):  # doctest: +SKIP
            ...     print(f"{f.feature_name}: {len(f.term_columns)} terms, "
            ...           f"{len(f.value_columns)} value columns")

            >>> # Every feature in the bag:
            >>> for f in bag.find_features():  # doctest: +SKIP
            ...     print(f"{f.target_table.name}: {f.feature_name}")
        """
        if table is None:
            # Delegate to the model's no-arg ``find_features``, which
            # walks the catalog once with the right deduplication.
            # Doing our own per-table loop here re-introduces the
            # multi-FK-target duplicate bug fixed in ``catalog.py``:
            # every feature whose association table has FKs to
            # multiple tables (the common case for asset features
            # with FKs to Execution + the target table + a vocab)
            # would be yielded once per FK target instead of once.
            # See ``DerivaModel.find_features`` (no-arg branch) for
            # the canonical dedup.
            yield from self.model.find_features()
            return
        yield from self.model.find_features(table)

    def feature_values(
        self,
        table: str | Table,
        feature_name: str,
        selector: Callable[[list[FeatureRecord]], FeatureRecord | None] | None = None,
        materialize_limit: int | None = None,
        execution_rids: list[str] | None = None,
    ) -> Iterable[FeatureRecord]:
        """Yield offline feature values — same signature as ``DerivaML.feature_values``.

        Bag-scoped feature read. Because bags are immutable snapshots, the
        result is stable for the bag's lifetime.

        When *selector* is ``None``, every feature record in the bag is yielded.
        When a *selector* is provided, records are grouped by target RID, the
        selector is called once per group (always, even single-element groups),
        and only groups for which the selector returns a non-``None`` value appear
        in the output.

        **Implementation (Stage 3b of the ``feature_values`` /
        ``Denormalizer`` consolidation):** this method delegates to
        :meth:`~deriva_ml.local_db.denormalizer.Denormalizer.feature_records`,
        the same delegation target ``Dataset.feature_values`` uses (Stage 3a).
        The ``Denormalizer`` runs in bag mode (``source="local"``) and reads
        the bag's **typed ORM tables** — ``ArrayAsJson`` / ``StringToDate``
        TypeDecorators decode array and date columns, and the ``RCT`` system
        column is recovered as a UTC-aware ISO-8601 string. This replaces the
        legacy ``BagFeatureCache`` path, which stored every column as SQLite
        ``TEXT`` and let pydantic coerce — that path returned ``RCT`` as a
        tz-naive string, diverging from the live catalog's UTC-aware shape.
        Dataset scoping is enforced by the denormalize SQL join (feature rows
        are reached *through* the bag's membership tables), which subsumes the
        legacy explicit ``target_rids`` dangling-FK filter (#126): a feature
        row can only appear if its target is reachable from the dataset.

        Args:
            table: Target table name or ``Table`` object (e.g. ``"Image"``).
            feature_name: Name of the feature to read (e.g. ``"Glaucoma"``).
            selector: Optional callable ``(list[FeatureRecord]) -> FeatureRecord | None``
                that resolves multiple values per target to a single record (or
                drops the target when it returns ``None``). Use
                ``FeatureRecord.select_newest`` to pick the most-recently created
                value.
            materialize_limit: Optional cap on the number of feature rows
                materialized. Raises ``DerivaMLMaterializeLimitExceeded``
                when exceeded. Default ``None`` preserves unbounded behavior.

                **Post-count guard (Stage 3b behavior note).** The denormalize
                join materializes the bag-scoped feature rows into memory, then
                this wrapper counts them and raises if the count exceeds the
                cap. Mirrors the ``Dataset.feature_values`` guard.
            execution_rids: Optional filter on the ``Execution`` column.
                When set, only feature rows whose ``Execution`` value is in
                this list survive. Applied Python-side after materialization
                (the bag has no server-side query layer). Empty list
                short-circuits to an empty result. The filter runs *before*
                selector reduction, matching the legacy ordering.

        Yields:
            FeatureRecord instances with typed fields matching the feature
            definition. ``RCT`` is populated (UTC-aware ISO-8601 string).
            Selector-filtered records (``None`` return) are omitted.

        Raises:
            DerivaMLException: If *feature_name* does not exist on *table*.
            DerivaMLDataError: If the bag is corrupt (source table missing).
            DerivaMLMaterializeLimitExceeded: If the materialized row count
                exceeds ``materialize_limit``.

        Example:
            >>> from deriva_ml.feature import FeatureRecord  # doctest: +SKIP
            >>> for rec in bag.feature_values("Image", "Glaucoma"):  # doctest: +SKIP
            ...     print(rec.Image, rec.Glaucoma)
            >>> # With selector — one record per image, most recent wins:
            >>> records = list(bag.feature_values(  # doctest: +SKIP
            ...     "Image", "Glaucoma", selector=FeatureRecord.select_newest,
            ... ))
        """
        from deriva_ml.core.exceptions import DerivaMLMaterializeLimitExceeded
        from deriva_ml.feature import reduce_with_selector
        from deriva_ml.local_db.denormalizer import Denormalizer

        # execution_rids=[] short-circuits to empty — preserve the legacy
        # behavior explicitly (matches Dataset.feature_values, audit §10).
        if execution_rids is not None and not execution_rids:
            return

        feat = self.lookup_feature(table, feature_name)
        target_col = feat.target_table.name

        # Delegate the bag-scoped SQL join + RCT recovery + FeatureRecord
        # materialization to the Denormalizer (bag mode, source="local").
        # Selector reduction is applied in this wrapper (not pushed into
        # feature_records) so the execution_rids filter runs first —
        # matching the legacy ordering "filter by execution, then reduce".
        records = Denormalizer(self).feature_records(feat, selector=None)

        # materialize_limit: post-count guard (mirrors Dataset.feature_values).
        if materialize_limit is not None and len(records) > materialize_limit:
            raise DerivaMLMaterializeLimitExceeded(
                actual_count=len(records),
                limit=materialize_limit,
            )

        # execution_rids: post-filter in the wrapper. The bag has no
        # server-side Execution predicate.
        if execution_rids is not None:
            exec_set = set(execution_rids)
            records = [rec for rec in records if rec.Execution in exec_set]

        if selector is None:
            yield from records
            return

        # Group by feature identity, then apply selector to every group
        # — always call selector, never short-circuit for
        # single-element groups. Shared helper so the three
        # feature_values surfaces stay in lockstep. ``qualifier_columns``
        # is empty for ordinary features (group-by-target, unchanged) and
        # carries the identity FKs for key-qualified features.
        yield from reduce_with_selector(records, target_col, selector, feat.qualifier_columns)

    def lookup_feature(self, table: str | Table, feature_name: str) -> Feature:
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

    def list_dataset_element_types(self) -> Iterable[Table]:
        """List the ERMrest Table objects that can be members of a dataset.

        Delegates to the underlying ``DerivaMLBagView`` to return all tables
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
            **kwargs: Accepted for ``DatasetLike`` protocol compatibility.
                ``Dataset`` honours a ``version=`` keyword to pin the
                catalog snapshot; bags are immutable, so any extras
                (including ``version=``) are silently ignored.

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
            **kwargs: Accepted for ``DatasetLike`` protocol compatibility.
                ``Dataset`` honours a ``version=`` keyword to pin the
                catalog snapshot; bags are immutable, so any extras
                (including ``version=``) are silently ignored.

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
        selector: Callable[[list[FeatureRecord]], FeatureRecord | None] | None = None,
        system_columns: list[str] | None = None,
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
            selector: Optional callable
                ``(list[FeatureRecord]) -> FeatureRecord | None`` used to
                reduce multi-row feature groups. See ``FeatureRecord`` for
                built-ins (``select_newest``, ``select_first``, etc.).
                Requires ``include_tables`` to contain exactly one
                feature-association table; raises ``ValueError``
                otherwise. Identical contract to
                :meth:`feature_values`'s ``selector`` argument.
            system_columns: Optional list of per-table system columns to
                retain in the output (any of ``"RCT"``, ``"RMT"``,
                ``"RCB"``, ``"RMB"``). These are dropped by default. Use
                this when you need provenance — e.g. ``["RCB"]`` keeps each
                row's creating-user id so it can be joined against the
                catalog user list. Retained columns are labeled
                ``Table.RCB`` like any other column.

        Returns:
            A :class:`pandas.DataFrame` with one row per ``row_per``
            instance in the bag. Columns use ``Table.column`` notation.

        Example::

            bag = dataset.download_dataset_bag(version)
            df = bag.get_denormalized_as_dataframe(["Image", "Subject"])

            # Reduce multi-annotator feature rows to one row per Image:
            from deriva_ml.feature import FeatureRecord
            df = bag.get_denormalized_as_dataframe(
                ["Image", "Execution_Image_Image_Classification"],
                selector=FeatureRecord.select_newest,
            )

            # Keep the diagnosis row's creating-user id for a grader join:
            df = bag.get_denormalized_as_dataframe(
                ["Image", "Image_Diagnosis"], system_columns=["RCB"]
            )
        """
        from deriva_ml.local_db.denormalizer import Denormalizer

        return Denormalizer(self).as_dataframe(
            include_tables,
            row_per=row_per,
            via=via,
            ignore_unrelated_anchors=ignore_unrelated_anchors,
            selector=selector,
            system_columns=system_columns,
        )

    def get_denormalized_as_dict(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
        ignore_unrelated_anchors: bool = False,
        selector: Callable[[list[FeatureRecord]], FeatureRecord | None] | None = None,
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
            selector: Optional callable
                ``(list[FeatureRecord]) -> FeatureRecord | None`` used to
                reduce multi-row feature groups. Same contract as
                :meth:`get_denormalized_as_dataframe`.

        Yields:
            ``dict[str, Any]`` per row — keys are ``Table.column``
            labels, values are raw Python types.

        Example::

            for row in bag.get_denormalized_as_dict(["Image", "Subject"]):
                process(row["Image.RID"], row["Subject.Name"])

            # With selector reduction:
            from deriva_ml.feature import FeatureRecord
            rows = bag.get_denormalized_as_dict(
                ["Image", "Execution_Image_Image_Classification"],
                selector=FeatureRecord.select_newest,
            )
        """
        from deriva_ml.local_db.denormalizer import Denormalizer

        yield from Denormalizer(self).as_dict(
            include_tables,
            row_per=row_per,
            via=via,
            ignore_unrelated_anchors=ignore_unrelated_anchors,
            selector=selector,
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
        exact 13-key shape). Never raises on ambiguity.

        Args:
            include_tables: Tables whose columns would appear in the output.
            row_per: Optional explicit leaf table (Rule 2).
            via: Optional path-only intermediates (Rule 6).

        Returns:
            dict: Planning metadata with 13 keys including ``anchors``,
            ``row_per``, ``join_path``, ``columns``, ``ambiguities``,
            and related diagnostics. See ``Denormalizer.describe`` for
            the full shape.

        Example::

            plan = bag.describe_denormalized(["Image", "Subject"])
            print(plan["anchors"], plan["row_per"])
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

    def as_torch_dataset(
        self,
        element_type: str,
        *,
        sample_loader: Callable[[Path | None, dict[str, Any]], Any] | None = None,
        transform: Callable[[Any], Any] | None = None,
        targets: list[str] | dict[str, Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        missing: Literal["error", "skip", "unknown"] = "error",
        reachable: bool = True,
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
                  ``bag.path / "data/asset/<rid>/<element_type>/<filename>"``
                  (the canonical BDBag asset layout) when
                  ``element_type`` is an asset table; ``None`` otherwise.
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
                  returns ``(sample, rid)``. Useful for inference loops
                  and self-supervised pretext tasks (the RID is still
                  surfaced so the inference loop can record predictions
                  back to the catalog).
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

            reachable: When ``True`` (default), the dataset's elements are
                every ``element_type`` row reachable from this dataset by any
                FK path — the same traversal ``restructure_assets`` uses. This
                matters for subject-partitioned datasets: if the members are
                ``Subject`` rows and ``Image`` is reachable via
                ``Subject -> Observation -> Image``, those Images are found
                even though they are not direct dataset members. When
                ``False``, only direct members (``list_dataset_members(
                recurse=True)``) are used — the opt-out for callers who want
                strictly the rows enumerated in the dataset's membership.

        Returns:
            A ``torch.utils.data.Dataset`` whose ``__getitem__`` returns:

            - ``(sample, rid)`` when ``targets=None`` (unlabeled).
            - ``(sample, target, rid)`` when ``targets`` is set, where
              ``sample = transform(sample_loader(path, row_dict))`` and
              ``target = target_transform(feature_record_shape)``.

            The element's catalog RID is always the last positional
            value. It's passed through raw — never touched by
            ``transform`` or ``target_transform`` — because the RID
            is the row's catalog identity, not a feature value. The
            motivating use case is downstream code that records
            per-element predictions back to the catalog (the RID is
            the FK target for the feature row).

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
            >>> from deriva_ml.dataset.torch_adapter import build_torch_dataset  # doctest: +SKIP
            >>> callable(build_torch_dataset)  # doctest: +SKIP
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
            reachable=reachable,
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
        reachable: bool = True,
    ) -> "tf.data.Dataset":
        """Build a ``tf.data.Dataset`` from this bag.

        Creates a ``tf.data.Dataset`` backed by a Python generator that
        reads samples and labels from this already-downloaded
        ``DatasetBag``. TensorFlow is imported lazily inside the builder
        so the base library stays importable without TensorFlow installed.

        Each call to the generator yields one element — either
        ``(sample, rid)`` (unlabeled) or ``(sample, target, rid)``
        (with ``targets`` set). The RID is always the last value;
        see the Returns section for details. Callers are
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
                  ``bag.path / "data/asset/<rid>/<element_type>/<filename>"``
                  (the canonical BDBag asset layout) when
                  ``element_type`` is an asset table; ``None`` otherwise.
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
                  the 2-tuple ``(sample, rid)`` (the trailing RID lets you
                  trace a sample back to its catalog row). Useful for
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

            reachable: When ``True`` (default), the dataset's elements are
                every ``element_type`` row reachable from this dataset by any
                FK path — the same traversal ``restructure_assets`` uses. This
                matters for subject-partitioned datasets: if the members are
                ``Subject`` rows and ``Image`` is reachable via
                ``Subject -> Observation -> Image``, those Images are found
                even though they are not direct dataset members. When
                ``False``, only direct members (``list_dataset_members(
                recurse=True)``) are used — the opt-out for callers who want
                strictly the rows enumerated in the dataset's membership.

        Returns:
            A ``tf.data.Dataset`` whose elements are:

            - ``(sample, rid)`` when ``targets=None`` (unlabeled).
            - ``(sample, target, rid)`` when ``targets`` is set, where
              ``sample = transform(sample_loader(path, row_dict))`` and
              ``target = target_transform(feature_record_shape)``.

            The element's catalog RID is always the last positional
            value. It's passed through raw — never touched by
            ``transform`` or ``target_transform`` — because the RID
            is the row's catalog identity, not a feature value. The
            motivating use case is downstream code that records
            per-element predictions back to the catalog (the RID is
            the FK target for the feature row).

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
            >>> from deriva_ml.dataset.tf_adapter import build_tf_dataset  # doctest: +SKIP
            >>> callable(build_tf_dataset)  # doctest: +SKIP
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
            reachable=reachable,
        )

    def restructure_assets(
        self,
        output_dir: "Path | str",
        *,
        asset_table: "str | None" = None,
        targets: "list[str] | dict[str, FeatureSelector] | None" = None,
        target_transform: "Callable[..., str] | None" = None,
        missing: "Literal['error', 'skip', 'unknown']" = "unknown",
        use_symlinks: bool = True,
        type_selector: "Callable[[list[str]], str] | None" = None,
        type_to_dir_map: "dict[str, str] | None" = None,
        enforce_vocabulary: bool = True,
        file_transformer: "Callable[[Path, Path], Path] | None" = None,
    ) -> "dict[Path, Path]":
        """Restructure downloaded assets into a directory hierarchy.

        Thin delegate to :func:`deriva_ml.dataset.restructure.restructure_assets`.
        The implementation lives in its own module per the Phase 3 §3.B
        split (audit-flagged); this method preserves the user-facing
        ``bag.restructure_assets(...)`` call shape.

        See :func:`~deriva_ml.dataset.restructure.restructure_assets` for
        the full parameter documentation.
        """
        from deriva_ml.dataset.restructure import restructure_assets

        return restructure_assets(
            self,
            output_dir,
            asset_table=asset_table,
            targets=targets,
            target_transform=target_transform,
            missing=missing,
            use_symlinks=use_symlinks,
            type_selector=type_selector,
            type_to_dir_map=type_to_dir_map,
            enforce_vocabulary=enforce_vocabulary,
            file_transformer=file_transformer,
        )


# Note: validate_call decorators with Self return types were removed because
# Pydantic doesn't support typing.Self in validate_call contexts.
