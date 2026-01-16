"""Protocol definitions for DerivaML dataset and catalog operations.

This module defines the protocols (interfaces) used throughout DerivaML for
type checking and polymorphic access to datasets and catalogs. The protocols
are organized into two hierarchies:

Dataset Protocols:
    DatasetLike: Read-only operations for both live datasets and downloaded bags.
    WritableDataset: Write operations only available on live catalog datasets.

Catalog Protocols:
    DerivaMLCatalogReader: Read-only catalog operations (lookups, queries).
    DerivaMLCatalog: Full catalog operations including write operations.

The separation allows code to express its requirements precisely:
- Code that only reads data can accept DatasetLike or DerivaMLCatalogReader
- Code that modifies data requires Dataset or DerivaMLCatalog

Implementation Notes:
    - Dataset: Live catalog access via deriva-py/datapath (implements both protocols)
    - DatasetBag: Downloaded bag access via SQLAlchemy/SQLite (read-only only)
    - DerivaML: Full catalog operations (implements DerivaMLCatalog)
    - DerivaMLDatabase: Bag-backed catalog (implements DerivaMLCatalogReader only)

Classes:
    DatasetLike: Read-only interface for dataset access.
    WritableDataset: Write interface for dataset modification.
    DerivaMLCatalogReader: Read-only interface for catalog access.
    DerivaMLCatalog: Full interface for catalog operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Iterable, Protocol, Self, runtime_checkable

import pandas as pd
from deriva.core import ErmrestSnapshot
from deriva.core.datapath import _SchemaWrapper as SchemaWrapper
from deriva.core.ermrest_catalog import ErmrestCatalog, ResolveRidResult
from deriva.core.ermrest_model import Table

from deriva_ml.core.definitions import RID, VocabularyTerm
from deriva_ml.core.mixins.rid_resolution import BatchRidResult
from deriva_ml.feature import Feature
from deriva_ml.model.catalog import DerivaModel

if TYPE_CHECKING:
    from deriva_ml.dataset.aux_classes import DatasetHistory, DatasetSpec, DatasetVersion
    from deriva_ml.dataset.dataset import Dataset


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
        **kwargs: Any,
    ) -> list[Self]:
        """Get nested child datasets.

        Args:
            recurse: Whether to recursively include children of children.
            _visited: Internal parameter to track visited datasets and prevent infinite recursion.
            version: Dataset version to list children from (Dataset only, ignored by DatasetBag).
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
        **kwargs: Any,
    ) -> list[Self]:
        """Get parent datasets that contain this dataset.

        Args:
            recurse: Whether to recursively include parents of parents.
            _visited: Internal parameter to track visited datasets and prevent infinite recursion.
            version: Dataset version to list parents from (Dataset only, ignored by DatasetBag).
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
        **kwargs: Any,
    ) -> dict[str, list[dict[str, Any]]]:
        """List members of the dataset.

        Args:
            recurse: Whether to include members of nested datasets.
            limit: Maximum number of members per type. None for no limit.
            _visited: Internal parameter to track visited datasets and prevent infinite recursion.
            version: Dataset version to list members from (Dataset only, ignored by DatasetBag).
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

    def denormalize_as_dataframe(
        self,
        include_tables: list[str],
        version: Any = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Denormalize the dataset into a single wide table (DataFrame).

        Denormalization transforms normalized relational data into a single "wide table"
        (also called a "flat table" or "denormalized table") by joining related tables
        together. This produces a DataFrame where each row contains all related information
        from multiple source tables, with columns from each table combined side-by-side.

        Wide tables are the standard input format for most machine learning frameworks,
        which expect all features for a single observation to be in one row. This method
        bridges the gap between normalized database schemas and ML-ready tabular data.

        **How it works:**

        Tables are joined based on their foreign key relationships. For example, if
        Image has a foreign key to Subject, and Diagnosis has a foreign key to Image,
        then denormalizing ["Subject", "Image", "Diagnosis"] produces rows where each
        image appears with its subject's metadata and any associated diagnoses.

        The result uses outer join semantics - if a table has no FK relationship to
        others, its rows are included with NULL values for unrelated columns.

        **Column naming:**

        To avoid column name collisions (e.g., multiple tables having "RID" or "Name"),
        all column names are prefixed with their source table name.

        Args:
            include_tables: List of table names to include in the output.
                Tables are joined based on their foreign key relationships.
                Order doesn't matter - the join order is determined automatically.
            version: Dataset version to query (Dataset only, ignored by DatasetBag).
            **kwargs: Additional implementation-specific arguments.

        Returns:
            pd.DataFrame: Wide table with columns from all included tables.
                Column names are prefixed with the source table name.

        Note:
            Column naming conventions differ between implementations:

            - Dataset (catalog): Uses underscore separator (e.g., "Image_Filename")
            - DatasetBag (bag): Uses dot separator (e.g., "Image.Filename")

        Example:
            Suppose you have a dataset with Images linked to Subjects, and each
            Image has a Diagnosis label::

                # Normalized schema:
                # Subject: RID, Name, Age
                # Image: RID, Filename, Subject (FK)
                # Diagnosis: RID, Image (FK), Label

                # Denormalize into a wide table for ML
                df = dataset.denormalize_as_dataframe(["Subject", "Image", "Diagnosis"])

                # Result has columns like:
                # Subject_RID, Subject_Name, Subject_Age,
                # Image_RID, Image_Filename, Image_Subject,
                # Diagnosis_RID, Diagnosis_Image, Diagnosis_Label

                # Each row represents one Image with its Subject info and Diagnosis
                # Ready for use with sklearn, pandas, or other ML tools

        See Also:
            denormalize_as_dict: Generator version for memory-efficient processing.
        """
        ...

    def denormalize_as_dict(
        self,
        include_tables: list[str],
        version: Any = None,
        **kwargs: Any,
    ) -> Generator[dict[str, Any], None, None]:
        """Denormalize the dataset and yield rows as dictionaries.

        This is a memory-efficient alternative to denormalize_as_dataframe() that
        yields one row at a time as a dictionary instead of loading all data into
        a DataFrame. Use this when processing large datasets that may not fit in
        memory, or when you want to process rows incrementally.

        Like denormalize_as_dataframe(), this produces a "wide table" representation
        where each yielded dictionary contains all columns from the joined tables.
        See denormalize_as_dataframe() for detailed explanation of how denormalization
        works.

        Args:
            include_tables: List of table names to include in the output.
                Tables are joined based on their foreign key relationships.
            version: Dataset version to query (Dataset only, ignored by DatasetBag).
            **kwargs: Additional implementation-specific arguments.

        Yields:
            dict[str, Any]: Dictionary representing one row of the wide table.
                Keys are column names prefixed by table name.

        Note:
            Column naming conventions differ between implementations:

            - Dataset (catalog): Uses underscore separator (e.g., "Image_Filename")
            - DatasetBag (bag): Uses dot separator (e.g., "Image.Filename")

        Example:
            Process a large dataset without loading everything into memory::

                # Stream through rows one at a time
                for row in dataset.denormalize_as_dict(["Image", "Diagnosis"]):
                    image_path = row["Image_Filename"]
                    label = row["Diagnosis_Label"]
                    # Process each image-label pair...

                # Or convert to list if you need random access
                rows = list(dataset.denormalize_as_dict(["Image", "Diagnosis"]))

        See Also:
            denormalize_as_dataframe: Returns all data as a pandas DataFrame.
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

    def add_dataset_members(self, members: list[RID]) -> None:
        """Add members to the dataset.

        Args:
            members: List of RIDs to add to the dataset.
        """
        ...

    def remove_dataset_members(self, members: list[RID]) -> None:
        """Remove members from the dataset.

        Args:
            members: List of RIDs to remove from the dataset.
        """
        ...

    def increment_dataset_version(
        self,
        component: Any,
        description: str | None = "",
        execution_rid: RID | None = None,
    ) -> DatasetVersion:
        """Increment the dataset version.

        Args:
            component: Which version component to increment (major, minor, patch).
            description: Optional description of the changes in this version.
            execution_rid: Optional execution RID to associate with this version.

        Returns:
            The new version after incrementing.
        """
        ...

    def download_dataset_bag(
        self,
        version: DatasetVersion | str | None = None,
        use_minid: bool = False,
    ) -> Any:
        """Download the dataset as a BDBag.

        Args:
            version: Optional version to download. Defaults to current version.
            use_minid: If True, upload the bag to S3 and create a MINID.
                Requires s3_bucket to be configured on the catalog. Defaults to False.

        Returns:
            DatasetBag containing the downloaded data.

        Raises:
            DerivaMLException: If use_minid=True but s3_bucket is not configured.
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
    domain_schema: str
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

    def find_datasets(self, deleted: bool = False) -> Iterable[DatasetLike]:
        """Find all datasets in the catalog.

        Args:
            deleted: Whether to include deleted datasets.

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

    def lookup_dataset(self, dataset: RID | DatasetSpec, deleted: bool = False) -> "Dataset":
        """Look up a dataset by RID or specification.

        Args:
            dataset: RID or DatasetSpec identifying the dataset.
            deleted: Whether to include deleted datasets.

        Returns:
            The dataset.
        """
        ...

    def find_datasets(self, deleted: bool = False) -> Iterable["Dataset"]:
        """Find all datasets in the catalog.

        Args:
            deleted: Whether to include deleted datasets.

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
