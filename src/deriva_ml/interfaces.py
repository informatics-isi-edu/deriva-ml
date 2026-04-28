"""Protocol definitions for DerivaML dataset, asset, and catalog operations.

This module defines the protocols (interfaces) used throughout DerivaML for
type checking and polymorphic access to datasets, assets, and catalogs.

Protocol Hierarchies
--------------------

**Dataset Protocols:**
    DatasetLike: Read-only operations for both live datasets and downloaded bags.
    WritableDataset: Write operations only available on live catalog datasets.

**Asset Protocols:**
    AssetLike: Read-only operations for asset access.
    WritableAsset: Write operations for asset modification.

**Catalog Protocols:**
    DerivaMLCatalogReader: Read-only catalog operations (lookups, queries).
    DerivaMLCatalog: Full catalog operations including write operations.

The separation allows code to express its requirements precisely:
- Code that only reads data can accept DatasetLike, AssetLike, or DerivaMLCatalogReader
- Code that modifies data requires WritableDataset, WritableAsset, or DerivaMLCatalog

API Naming Conventions
----------------------

The DerivaML API follows consistent naming conventions:

- ``lookup_*``: Single item retrieval by identifier. Returns one item or raises exception.
    Examples: lookup_dataset(), lookup_asset(), lookup_term()

- ``find_*``: Search/discovery operations. Returns Iterable of matching items.
    Examples: find_datasets(), find_assets(), find_features()

- ``list_*``: List all items of a type, often with context (e.g., members of a dataset).
    Examples: list_assets(), list_vocabulary_terms(), list_dataset_members()

- ``get_*``: Data retrieval with transformation (e.g., to DataFrame).
    Examples: get_table_as_dataframe(), get_metadata()

- ``create_*``: Create new entities in the catalog.
    Examples: create_dataset(), create_execution(), create_feature()

- ``add_*``: Add items to existing entities or create vocabulary terms.
    Examples: add_term(), add_dataset_members(), add_asset_type()

- ``delete_*`` / ``remove_*``: Remove items from entities.
    Examples: delete_dataset_members(), remove_asset_type()

Implementation Notes
--------------------
- Dataset: Live catalog access via deriva-py/datapath (implements both protocols)
- DatasetBag: Downloaded bag access via SQLAlchemy/SQLite (read-only only)
- Asset: Live catalog access for file-based records (implements WritableAsset)
- DerivaML: Full catalog operations (implements DerivaMLCatalog)
- DerivaMLDatabase: Bag-backed catalog (implements DerivaMLCatalogReader only)
"""

from __future__ import annotations

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Iterable, Protocol, Self, runtime_checkable

import pandas as pd

_deriva_core = importlib.import_module("deriva.core")
_datapath = importlib.import_module("deriva.core.datapath")
_ermrest_catalog = importlib.import_module("deriva.core.ermrest_catalog")
_ermrest_model = importlib.import_module("deriva.core.ermrest_model")

ErmrestSnapshot = _deriva_core.ErmrestSnapshot
SchemaWrapper = _datapath._SchemaWrapper
ErmrestCatalog = _ermrest_catalog.ErmrestCatalog
ResolveRidResult = _ermrest_catalog.ResolveRidResult
Table = _ermrest_model.Table

from deriva_ml.core.definitions import RID, VocabularyTerm
from deriva_ml.core.mixins.rid_resolution import BatchRidResult
from deriva_ml.core.sort import SortSpec
from deriva_ml.feature import Feature, FeatureRecord
from deriva_ml.model.catalog import DerivaModel

if TYPE_CHECKING:
    from deriva_ml.dataset.aux_classes import DatasetHistory, DatasetSpec, DatasetVersion, VersionPart
    from deriva_ml.dataset.dataset import Dataset
    from deriva_ml.execution.execution_record import ExecutionRecord
    from deriva_ml.execution.state_store import ExecutionStatus
    from deriva_ml.execution.workflow import Workflow


@runtime_checkable
class DatasetLike(Protocol):
    """Protocol defining read-only interface for dataset access.

    This protocol is implemented by both Dataset (live catalog) and DatasetBag
    (downloaded bag). It defines the common read interface for accessing dataset
    metadata, members, and relationships.

    The protocol defines the minimal interface that both implementations support.
    Dataset extends this with optional `version` parameters on some methods to
    support querying historical versions. DatasetBag doesn't need version parameters
    since bags are immutable snapshots of a specific version.

    Note on `_visited` parameters: Both implementations use `_visited` internally
    for recursion guards, but this is not part of the protocol as it's an
    implementation detail.

    Attributes:
        dataset_rid: Resource Identifier for the dataset.
        execution_rid: Optional execution RID associated with the dataset.
        description: Description of the dataset.
        dataset_types: Type(s) of the dataset from Dataset_Type vocabulary.
        current_version: Current semantic version of the dataset.
    """

    dataset_rid: RID
    execution_rid: RID | None
    description: str
    dataset_types: list[str]

    @property
    def current_version(self) -> DatasetVersion:
        """Get the current version of the dataset."""
        ...

    def dataset_history(self) -> list[DatasetHistory]:
        """Get the version history of the dataset."""
        ...

    def list_dataset_children(
        self,
        recurse: bool = False,
        _visited: set[RID] | None = None,
        version: Any = None,
        sort: SortSpec = None,
        **kwargs: Any,
    ) -> list[Self]:
        """Get nested child datasets.

        Args:
            recurse: Whether to recursively include children of children.
            _visited: Internal parameter to track visited datasets and prevent infinite recursion.
            version: Dataset version to list children from (Dataset only, ignored by DatasetBag).
            sort: Optional sort spec — see :class:`deriva_ml.core.sort.SortSpec`.
                ``None`` (default) preserves backend order. ``True`` applies
                the method's default. A callable receives the path-builder
                context and returns sort keys. Currently reserved for
                forward-compat — concrete ``Dataset`` and ``DatasetBag``
                implementations may ignore this parameter (it is accepted
                via ``**kwargs``).
            **kwargs: Additional implementation-specific arguments.

        Returns:
            List of child datasets (Dataset or DatasetBag depending on implementation).

        Note:
            Both Dataset and DatasetBag have `recurse` as the first parameter.
            Dataset uses the `version` parameter to query historical versions.
        """
        ...

    def list_dataset_parents(
        self,
        recurse: bool = False,
        _visited: set[RID] | None = None,
        version: Any = None,
        sort: SortSpec = None,
        **kwargs: Any,
    ) -> list[Self]:
        """Get parent datasets that contain this dataset.

        Args:
            recurse: Whether to recursively include parents of parents.
            _visited: Internal parameter to track visited datasets and prevent infinite recursion.
            version: Dataset version to list parents from (Dataset only, ignored by DatasetBag).
            sort: Optional sort spec — see :class:`deriva_ml.core.sort.SortSpec`.
                ``None`` (default) preserves backend order. ``True`` applies
                the method's default. A callable receives the path-builder
                context and returns sort keys. Currently reserved for
                forward-compat — concrete ``Dataset`` and ``DatasetBag``
                implementations may ignore this parameter (it is accepted
                via ``**kwargs``).
            **kwargs: Additional implementation-specific arguments.

        Returns:
            List of parent datasets (Dataset or DatasetBag depending on implementation).

        Note:
            Both Dataset and DatasetBag have `recurse` as the first parameter.
            Dataset uses the `version` parameter to query historical versions.
        """
        ...

    def list_dataset_members(
        self,
        recurse: bool = False,
        limit: int | None = None,
        _visited: set[RID] | None = None,
        version: Any = None,
        sort: SortSpec = None,
        **kwargs: Any,
    ) -> dict[str, list[dict[str, Any]]]:
        """List members of the dataset.

        Args:
            recurse: Whether to include members of nested datasets.
            limit: Maximum number of members per type. None for no limit.
            _visited: Internal parameter to track visited datasets and prevent infinite recursion.
            version: Dataset version to list members from (Dataset only, ignored by DatasetBag).
            sort: Optional sort spec — see :class:`deriva_ml.core.sort.SortSpec`.
                ``None`` (default) preserves backend order. ``True`` applies
                the method's default. A callable receives the path-builder
                context and returns sort keys. Currently reserved for
                forward-compat — concrete ``Dataset`` and ``DatasetBag``
                implementations may ignore this parameter (it is accepted
                via ``**kwargs``).
            **kwargs: Additional implementation-specific arguments.

        Returns:
            Dictionary mapping member types to lists of member records.

        Note:
            Both Dataset and DatasetBag have `recurse` as the first parameter.
            Dataset uses the `version` parameter to query historical versions.
        """
        ...

    def list_dataset_element_types(self) -> Iterable[Table]:
        """List the types of elements that can be contained in this dataset.

        Returns:
            Iterable of Table objects representing element types.
        """
        ...

    def find_features(self, table: str | Table) -> Iterable[Feature]:
        """Find features associated with a table.

        Args:
            table: Table to find features for.

        Returns:
            Iterable of Feature objects.
        """
        ...

    def lookup_feature(self, table: str | Table, feature_name: str) -> Feature:
        """Look up a feature definition by table and name.

        Args:
            table: The table the feature is defined on (name or Table object).
            feature_name: Name of the feature to look up.

        Returns:
            A Feature schema descriptor.

        Raises:
            DerivaMLException: If the feature doesn't exist on the specified table.
        """
        ...

    def feature_values(
        self,
        table: Table | str,
        feature_name: str,
        selector: Any = None,
        materialize_limit: int | None = None,
        execution_rids: list[str] | None = None,
    ) -> Iterable[FeatureRecord]:
        """Yield feature values for a single feature, one record per target RID.

        Args:
            table: Target table the feature is defined on (name or Table).
            feature_name: Name of the feature to read.
            selector: Optional callable ``(list[FeatureRecord]) -> FeatureRecord | None``
                used to reduce multi-value groups.
            materialize_limit: Optional cap on the number of rows that
                may be materialized into memory. When the catalog query
                returns more than this many rows, raises
                ``DerivaMLMaterializeLimitExceeded``. Default ``None``
                preserves the existing unbounded behavior; callers
                driving Python directly opt into responsibility for
                memory management. The ``deriva-ml-mcp`` plugin sets a
                default to keep MCP responses bounded.
            execution_rids: Optional filter — when set, only feature
                rows whose ``Execution`` value is in this list are
                materialized. Lets callers compare metric values
                across a known set of executions in a single
                catalog round-trip rather than N sequential queries.

        Returns:
            Iterator of ``FeatureRecord`` instances.

        Raises:
            DerivaMLException: If ``feature_name`` is not a feature on ``table``.
            DerivaMLMaterializeLimitExceeded: If the result set exceeds
                ``materialize_limit``.
        """
        ...

    def list_workflow_executions(self, workflow: str) -> list[str]:
        """Return execution RIDs that ran the given workflow.

        Args:
            workflow: Workflow RID or Workflow_Type name.

        Returns:
            List of execution RIDs. May be empty.

        Raises:
            DerivaMLException: If ``workflow`` does not resolve.
        """
        ...

    def get_denormalized_as_dataframe(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
        ignore_unrelated_anchors: bool = False,
    ) -> pd.DataFrame:
        """Return the dataset as a denormalized wide table (DataFrame).

        Denormalization transforms normalized relational data into a single
        "wide table" by joining related tables on their foreign key
        relationships. Wide tables are the standard input format for most
        machine learning frameworks.

        Column names are prefixed with the source table using dot notation
        (e.g., ``Image.Filename``, ``Subject.RID``). When the catalog has
        multiple domain schemas, the schema name is included too
        (e.g., ``test-schema.Image.Filename``).

        This is a sugar method that delegates to
        :class:`~deriva_ml.local_db.denormalizer.Denormalizer`; see that
        class for the full API and semantic rules.

        Args:
            include_tables: Tables whose columns appear in the output.
            row_per: Explicit leaf table. Must be in ``include_tables``.
                If None, auto-inferred from the FK graph.
            via: Tables forced into the join chain without contributing
                columns (useful to disambiguate path ambiguity).
            ignore_unrelated_anchors: If True, silently drop anchors whose
                table has no FK path to any requested table.

        Returns:
            pd.DataFrame: Wide table with columns from all included tables.

        See Also:
            list_denormalized_columns: Preview columns without fetching data.
            get_denormalized_as_dict: Generator for memory-efficient streaming.
            describe_denormalized: Dry-run returning planning metadata.
        """
        ...

    def get_denormalized_as_dict(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
        ignore_unrelated_anchors: bool = False,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream the denormalized dataset rows as dicts.

        Memory-efficient alternative to
        :meth:`get_denormalized_as_dataframe` that yields one row at a time
        as a dict. Use for large datasets that may not fit in memory.

        Args:
            include_tables: Tables whose columns appear in the output.
            row_per: Explicit leaf table. Must be in ``include_tables``.
            via: Tables forced into the join chain without contributing columns.
            ignore_unrelated_anchors: See :meth:`get_denormalized_as_dataframe`.

        Yields:
            dict[str, Any]: Dictionary per row; keys are ``Table.Column``.

        See Also:
            list_denormalized_columns: Preview columns without fetching data.
            get_denormalized_as_dataframe: Returns a pandas DataFrame.
        """
        ...

    def list_denormalized_columns(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        """List the columns the denormalized table would have.

        Performs the same path validation as
        :meth:`get_denormalized_as_dataframe` (table existence, FK path
        resolution, ambiguity detection) but stops before executing any
        data queries. Model-only — no data fetch.

        Args:
            include_tables: Tables whose columns would appear in the output.
            row_per: Explicit leaf table. Must be in ``include_tables``.
            via: Tables forced into the join chain without contributing columns.

        Returns:
            List of ``(column_name, column_type)`` tuples. Column names use
            dot notation. Type strings are ermrest type names
            (``text``, ``int4``, ``float8``, etc.).

        Raises:
            DerivaMLException: If tables don't exist or FK paths are ambiguous.

        See Also:
            get_denormalized_as_dataframe: Fetch data as a pandas DataFrame.
            get_denormalized_as_dict: Fetch data as a generator of dicts.
        """
        ...

    def describe_denormalized(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
    ) -> dict[str, Any]:
        """Dry-run a denormalization call; return planning metadata.

        Resolves the FK path, computes the column shape, and estimates
        row counts without materializing any data. Unlike
        :meth:`get_denormalized_as_dataframe`, this method does NOT
        raise on Rule 2/5/6 errors — every failure mode is represented
        in the returned dict with ``None`` / ``[]`` / ``{}`` in the
        affected positions so callers can inspect and react.

        Args:
            include_tables: Tables whose columns would appear in the output.
            row_per: Optional explicit leaf table.
            via: Optional path-only intermediates.

        Returns:
            12-key planning dict (see
            :meth:`~deriva_ml.local_db.denormalizer.Denormalizer.describe`
            for the detailed shape): ``row_per``, ``row_per_source``,
            ``row_per_candidates``, ``columns``, ``include_tables``,
            ``via``, ``join_path``, ``transparent_intermediates``,
            ``ambiguities``, ``estimated_row_count``, ``anchors``,
            ``source``.

        See Also:
            get_denormalized_as_dataframe: Execute the plan and return a
                DataFrame (raises on rule violations).
            list_denormalized_columns: Column shape only (no anchors,
                no row-count estimation).
        """
        ...

    def list_schema_paths(
        self,
        tables: list[str] | None = None,
    ) -> dict[str, Any]:
        """List FK paths reachable from this dataset's members.

        Model-only analysis — no data fetch. Useful for discovering
        what tables are available to include in denormalization when
        the user doesn't know the schema.

        Args:
            tables: Optional filter — when given, ``schema_paths`` in
                the returned dict includes only entries involving at
                least one of these tables.

        Returns:
            6-key dict (see
            :meth:`~deriva_ml.local_db.denormalizer.Denormalizer.list_paths`
            for the detailed shape): ``member_types``, ``anchor_types``,
            ``reachable_tables``, ``association_tables``,
            ``feature_tables``, ``schema_paths``.
        """
        ...


@runtime_checkable
class WritableDataset(DatasetLike, Protocol):
    """Protocol defining write operations for datasets.

    This protocol extends DatasetLike with write operations that are only
    available on live catalog datasets. Downloaded bags (DatasetBag) are
    immutable snapshots and do not implement these methods.

    Use this protocol when you need to express that code requires the ability
    to modify a dataset, not just read from it.

    Example:
        >>> def add_samples(dataset: WritableDataset, sample_rids: list[str]):
        ...     dataset.add_dataset_members(sample_rids)
        ...     dataset.increment_dataset_version(VersionPart.minor)
    """

    def add_dataset_members(
        self,
        members: list[RID] | dict[str, list[RID]],
        validate: bool = True,
        description: str | None = "",
        execution_rid: RID | None = None,
    ) -> None:
        """Add members to the dataset.

        Members can be provided as a list of RIDs (auto-resolved to their tables)
        or as a dict mapping table names to RID lists (faster, skips resolution).

        Adding members automatically increments the dataset's minor version.

        Args:
            members: Either a list of RIDs or a dict mapping table names to RID lists.
            validate: If True, validate that RIDs exist and tables are element types.
            description: Optional description for the version increment.
            execution_rid: Optional execution RID for provenance tracking.
        """
        ...

    def delete_dataset_members(
        self,
        members: list[RID],
        description: str = "",
        execution_rid: RID | None = None,
    ) -> None:
        """Remove members from the dataset.

        Args:
            members: List of RIDs to remove from the dataset.
            description: Optional description of the removal operation.
            execution_rid: Optional RID of execution associated with this operation.
        """
        ...

    def increment_dataset_version(
        self,
        component: "VersionPart",
        description: str | None = "",
        execution_rid: RID | None = None,
    ) -> DatasetVersion:
        """Increment the dataset version.

        Args:
            component: Which version component to increment (VersionPart.major, .minor, or .patch).
            description: Optional description of the changes in this version.
            execution_rid: Optional execution RID to associate with this version.

        Returns:
            The new version after incrementing.
        """
        ...

    def download_dataset_bag(
        self,
        version: DatasetVersion | str,
        materialize: bool = True,
        use_minid: bool = False,
        exclude_tables: set[str] | None = None,
        timeout: tuple[int, int] | None = None,
    ) -> Any:
        """Download the dataset as a BDBag.

        Args:
            version: Version to download (e.g., "1.0.0"). Required.
            materialize: If True (default), download all referenced asset files.
                If False, bag contains only metadata and remote file references.
            use_minid: If True, upload the bag to S3 and create a MINID.
                Requires s3_bucket to be configured on the catalog. Defaults to False.
            exclude_tables: Optional set of table names to exclude from FK path traversal
                during bag export. Useful for avoiding query timeouts on large tables.
            timeout: Optional (connect_timeout, read_timeout) in seconds. Defaults to (10, 610).

        Returns:
            DatasetBag containing the downloaded data.

        Raises:
            DerivaMLException: If use_minid=True but s3_bucket is not configured.
        """
        ...


@runtime_checkable
class AssetLike(Protocol):
    """Protocol defining read-only interface for asset access.

    This protocol defines the common read interface for accessing asset
    metadata, types, and provenance. It parallels DatasetLike but for
    individual file-based records rather than data collections.

    Attributes:
        asset_rid: Resource Identifier for the asset.
        asset_table: Name of the asset table containing this asset.
        filename: Original filename of the asset.
        url: URL to access the asset file.
        length: Size of the asset file in bytes.
        md5: MD5 checksum of the asset file.
        asset_types: Type(s) of the asset from Asset_Type vocabulary.
        description: Description of the asset.
        execution_rid: Optional execution RID that created the asset.
    """

    asset_rid: RID
    asset_table: str
    filename: str
    url: str
    length: int
    md5: str
    asset_types: list[str]
    description: str
    execution_rid: RID | None

    def list_executions(self, asset_role: str | None = None) -> list[dict[str, Any]]:
        """List all executions associated with this asset.

        Args:
            asset_role: Optional filter for asset role ('Input' or 'Output').

        Returns:
            List of records with Execution RID and Asset_Role.
        """
        ...

    def find_features(self) -> Iterable[Feature]:
        """Find features defined on this asset's table.

        Returns:
            Iterable of Feature objects.
        """
        ...

    def get_metadata(self) -> dict[str, Any]:
        """Get all metadata for this asset from the catalog.

        Returns:
            Dictionary of all columns/values for this asset record.
        """
        ...

    def get_chaise_url(self) -> str:
        """Get the Chaise URL for viewing this asset in the web interface.

        Returns:
            URL to view this asset in Chaise.
        """
        ...


@runtime_checkable
class WritableAsset(AssetLike, Protocol):
    """Protocol defining write operations for assets.

    This protocol extends AssetLike with write operations that are only
    available on live catalog assets. Downloaded assets are immutable
    and do not implement these methods.
    """

    def add_asset_type(self, type_name: str) -> None:
        """Add an asset type to this asset.

        Args:
            type_name: Name of the asset type vocabulary term.
        """
        ...

    def remove_asset_type(self, type_name: str) -> None:
        """Remove an asset type from this asset.

        Args:
            type_name: Name of the asset type vocabulary term.
        """
        ...


@runtime_checkable
class DerivaMLCatalogReader(Protocol):
    """Protocol for read-only catalog operations.

    This protocol defines the minimal interface for reading from a catalog,
    implemented by both DerivaML (live catalog) and DerivaMLDatabase (downloaded bags).

    Use this protocol when code only needs to read data and should work with
    both live catalogs and downloaded bags.

    Attributes:
        ml_schema: Name of the ML schema (typically 'deriva-ml').
        domain_schema: Name of the domain-specific schema.
        model: The catalog model containing schema information.
        cache_dir: Directory for caching downloaded data.
        working_dir: Directory for working files.

    Example:
        >>> def analyze_dataset(catalog: DerivaMLCatalogReader, dataset_rid: str):
        ...     dataset = catalog.lookup_dataset(dataset_rid)
        ...     members = dataset.list_dataset_members()
        ...     return len(members)
    """

    ml_schema: str
    domain_schemas: frozenset[str]
    default_schema: str | None
    model: DerivaModel
    cache_dir: Path
    working_dir: Path

    def lookup_dataset(self, dataset: RID | DatasetSpec, deleted: bool = False) -> DatasetLike:
        """Look up a dataset by RID or specification.

        Args:
            dataset: RID or DatasetSpec identifying the dataset.
            deleted: Whether to include deleted datasets.

        Returns:
            The dataset (Dataset for live catalogs, DatasetBag for bags).
        """
        ...

    def find_datasets(self, deleted: bool = False, sort: SortSpec = None) -> Iterable[DatasetLike]:
        """Find all datasets in the catalog.

        Args:
            deleted: Whether to include deleted datasets.
            sort: Optional sort spec — see :class:`deriva_ml.core.sort.SortSpec`.
                ``None`` (default) preserves backend order. ``True`` applies
                newest-first by record creation time (``RCT desc``). A callable
                receives the path-builder context and returns one or more sort keys.

        Returns:
            Iterable of all datasets.
        """
        ...

    def lookup_term(self, table: str | Table, term_name: str) -> VocabularyTerm:
        """Look up a vocabulary term.

        Args:
            table: Vocabulary table name or Table object.
            term_name: Name of the term to look up.

        Returns:
            The vocabulary term.
        """
        ...

    def get_table_as_dataframe(self, table: str) -> pd.DataFrame:
        """Get table contents as a pandas DataFrame.

        Args:
            table: Name of the table to retrieve.

        Returns:
            DataFrame containing table contents.
        """
        ...

    def get_table_as_dict(self, table: str) -> Iterable[dict[str, Any]]:
        """Get table contents as dictionaries.

        Args:
            table: Name of the table to retrieve.

        Returns:
            Iterable of dictionaries for each row.
        """
        ...

    def list_dataset_element_types(self) -> Iterable[Table]:
        """List the types of elements that can be contained in datasets.

        Returns:
            Iterable of Table objects representing element types.
        """
        ...

    def find_features(self, table: str | Table) -> Iterable[Feature]:
        """Find features associated with a table.

        Args:
            table: Table to find features for.

        Returns:
            Iterable of Feature objects.
        """
        ...

    def lookup_workflow(self, rid: RID) -> "Workflow":
        """Look up a workflow by its Resource Identifier (RID).

        Retrieves a workflow from the catalog by its RID. The returned Workflow
        is bound to the catalog, allowing its description to be updated (on
        writable catalogs).

        Args:
            rid: Resource Identifier of the workflow to look up.

        Returns:
            Workflow: The workflow object bound to this catalog.

        Raises:
            DerivaMLException: If the RID does not correspond to a workflow.

        Example:
            >>> workflow = catalog.lookup_workflow("2-ABC1")  # doctest: +SKIP
            >>> print(f"{workflow.name}: {workflow.description}")
        """
        ...

    def find_workflows(self, sort: SortSpec = None) -> Iterable["Workflow"]:
        """Find all workflows in the catalog.

        Returns all workflow definitions, each bound to the catalog for
        potential modification.

        Args:
            sort: Optional sort spec — see :class:`deriva_ml.core.sort.SortSpec`.
                ``None`` (default) preserves backend order. ``True`` applies
                newest-first by record creation time (``RCT desc``). A callable
                receives the path-builder context and returns one or more sort keys.

        Returns:
            Iterable of Workflow objects.

        Example:
            >>> for workflow in catalog.find_workflows():  # doctest: +SKIP
            ...     print(f"{workflow.name}: {workflow.description}")
        """
        ...

    def lookup_workflow_by_url(self, url_or_checksum: str) -> "Workflow":
        """Look up a workflow by URL or checksum.

        Searches for a workflow matching the given GitHub URL or Git object
        hash (checksum) and returns a bound Workflow object.

        Args:
            url_or_checksum: GitHub URL with commit hash, or Git object hash.

        Returns:
            Workflow: The workflow object bound to this catalog.

        Raises:
            DerivaMLException: If no matching workflow is found.

        Example:
            >>> url = "https://github.com/org/repo/blob/abc123/workflow.py"  # doctest: +SKIP
            >>> workflow = catalog.lookup_workflow_by_url(url)
            >>> print(f"{workflow.name}: {workflow.description}")
        """
        ...

    def lookup_execution(self, execution_rid: RID) -> "ExecutionRecord":
        """Look up an execution by RID.

        Returns an ExecutionRecord for querying and modifying execution metadata.

        Args:
            execution_rid: Resource Identifier of the execution.

        Returns:
            ExecutionRecord: The execution record bound to this catalog.

        Raises:
            DerivaMLException: If the RID doesn't refer to an Execution.

        Example:
            >>> record = catalog.lookup_execution("2-ABC1")  # doctest: +SKIP
            >>> print(f"{record.status}: {record.description}")
        """
        ...

    def find_executions(
        self,
        workflow: "Workflow | RID | None" = None,
        workflow_type: str | None = None,
        status: "ExecutionStatus | None" = None,
        sort: SortSpec = None,
    ) -> Iterable["ExecutionRecord"]:
        """List all executions in the catalog.

        Args:
            workflow: Optional Workflow object or RID to filter by.
            workflow_type: Optional workflow type name to filter by.
            status: Optional status to filter by.
            sort: Optional sort spec — see :class:`deriva_ml.core.sort.SortSpec`.
                ``None`` (default) preserves backend order. ``True`` applies
                newest-first by record creation time (``RCT desc``). A callable
                receives the path-builder context and returns one or more sort keys.

        Returns:
            Iterable of ExecutionRecord objects.

        Example:
            >>> for record in catalog.find_executions():  # doctest: +SKIP
            ...     print(f"{record.execution_rid}: {record.status}")
            >>> # Filter by workflow type
            >>> for record in catalog.find_executions(workflow_type="python_script"):  # doctest: +SKIP
            ...     print(f"{record.execution_rid}")
        """
        ...


@runtime_checkable
class DerivaMLCatalog(DerivaMLCatalogReader, Protocol):
    """Protocol for full catalog operations including writes.

    This protocol extends DerivaMLCatalogReader with write operations and
    is implemented by DerivaML (for live catalogs). It provides methods for:
    - Schema and table access
    - Dataset creation and modification
    - Vocabulary term management
    - Catalog snapshots and path building

    Use this protocol when code needs to modify the catalog. For read-only
    operations, prefer DerivaMLCatalogReader.

    Attributes:
        catalog: The underlying ERMrest catalog connection.
        catalog_id: Catalog identifier or name.
        s3_bucket: S3 bucket URL for dataset storage, or None if not configured.
        use_minid: Whether MINID service is enabled for this catalog.

    Example:
        >>> def process_data(catalog: DerivaMLCatalog):
        ...     datasets = list(catalog.find_datasets())
        ...     for ds in datasets:
        ...         print(ds.description)
        ...     return datasets
    """

    catalog: ErmrestCatalog | ErmrestSnapshot
    catalog_id: str | int
    s3_bucket: str | None
    use_minid: bool

    def pathBuilder(self) -> SchemaWrapper:
        """Get a path builder for constructing catalog queries.

        Returns:
            SchemaWrapper for building datapath queries.
        """
        ...

    def catalog_snapshot(self, version_snapshot: str) -> Self:
        """Create a view of the catalog at a specific snapshot time.

        Args:
            version_snapshot: Snapshot timestamp string.

        Returns:
            A new catalog instance bound to the snapshot.
        """
        ...

    def resolve_rid(self, rid: RID) -> ResolveRidResult:
        """Resolve a RID to its catalog location.

        Args:
            rid: Resource Identifier to resolve.

        Returns:
            Information about the RID's location in the catalog.
        """
        ...

    def resolve_rids(
        self,
        rids: set[RID] | list[RID],
        candidate_tables: list[Table] | None = None,
    ) -> dict[RID, BatchRidResult]:
        """Batch resolve multiple RIDs efficiently.

        Resolves multiple RIDs in batched queries, significantly faster than
        calling resolve_rid() for each RID individually.

        Args:
            rids: Set or list of RIDs to resolve.
            candidate_tables: Optional list of Table objects to search in.
                If not provided, searches all tables in domain and ML schemas.

        Returns:
            Mapping from each resolved RID to its BatchRidResult.
        """
        ...

    def add_features(self, features: list[FeatureRecord]) -> int:
        """Add feature values to the catalog in batch.

        Inserts a list of FeatureRecord instances into the appropriate feature table.
        All records must be from the same feature.

        Args:
            features: List of FeatureRecord instances to insert.

        Returns:
            Number of feature records inserted.
        """
        ...

    def lookup_dataset(self, dataset: RID | DatasetSpec, deleted: bool = False) -> "Dataset":
        """Look up a dataset by RID or specification.

        Args:
            dataset: RID or DatasetSpec identifying the dataset.
            deleted: Whether to include deleted datasets.

        Returns:
            The dataset.
        """
        ...

    def find_datasets(self, deleted: bool = False, sort: SortSpec = None) -> Iterable["Dataset"]:
        """Find all datasets in the catalog.

        Args:
            deleted: Whether to include deleted datasets.
            sort: Optional sort spec — see :class:`deriva_ml.core.sort.SortSpec`.
                ``None`` (default) preserves backend order. ``True`` applies
                newest-first by record creation time (``RCT desc``). A callable
                receives the path-builder context and returns one or more sort keys.

        Returns:
            Iterable of all datasets.
        """
        ...

    @property
    def _dataset_table(self) -> Table:
        """Get the Dataset table from the model.

        Returns:
            The Dataset table object.
        """
        ...
