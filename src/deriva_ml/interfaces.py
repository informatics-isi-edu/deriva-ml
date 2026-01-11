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
from typing import Any, Generator, Iterable, Protocol, Self, runtime_checkable

import pandas as pd
from deriva.core import ErmrestSnapshot
from deriva.core.datapath import _SchemaWrapper as SchemaWrapper
from deriva.core.ermrest_catalog import ErmrestCatalog, ResolveRidResult
from deriva.core.ermrest_model import Table

from deriva_ml.core.definitions import RID, VocabularyTerm
from deriva_ml.dataset.aux_classes import DatasetHistory, DatasetSpec, DatasetVersion
from deriva_ml.feature import Feature
from deriva_ml.model.catalog import DerivaModel


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

    def list_dataset_children(self, recurse: bool = False) -> list[Self]:
        """Get nested child datasets.

        Args:
            recurse: Whether to recursively include children of children.

        Returns:
            List of child datasets (Dataset or DatasetBag depending on implementation).

        Note:
            Dataset also accepts a `version` parameter to query historical versions.
        """
        ...

    def list_dataset_parents(self, recurse: bool = False) -> list[Self]:
        """Get parent datasets that contain this dataset.

        Args:
            recurse: Whether to recursively include parents of parents.

        Returns:
            List of parent datasets (Dataset or DatasetBag depending on implementation).

        Note:
            Dataset also accepts a `version` parameter to query historical versions.
        """
        ...

    def list_dataset_members(self, recurse: bool = False, limit: int | None = None) -> dict[str, list[dict[str, Any]]]:
        """List members of the dataset.

        Args:
            recurse: Whether to include members of nested datasets.
            limit: Maximum number of members per type. None for no limit.

        Returns:
            Dictionary mapping member types to lists of member records.

        Note:
            Dataset also accepts a `version` parameter to query historical versions.
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

    def increment_dataset_version(self, component: Any) -> DatasetVersion:
        """Increment the dataset version.

        Args:
            component: Which version component to increment (major, minor, patch).

        Returns:
            The new version after incrementing.
        """
        ...

    def download_dataset_bag(
        self,
        version: DatasetVersion | str | None = None,
        use_minid: bool = True,
    ) -> Any:
        """Download the dataset as a BDBag.

        Args:
            version: Optional version to download. Defaults to current version.
            use_minid: Whether to use MINID for dataset identification.

        Returns:
            DatasetBag containing the downloaded data.
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

    Example:
        >>> def create_analysis_dataset(catalog: DerivaMLCatalog):
        ...     dataset = catalog.create_dataset(
        ...         execution_rid=None,
        ...         version="1.0.0",
        ...         description="Analysis results",
        ...         dataset_types=["Analysis"],
        ...     )
        ...     return dataset
    """

    catalog: ErmrestCatalog | ErmrestSnapshot
    catalog_id: str | int

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

    def create_dataset(
        self,
        version: DatasetVersion | str | None = None,
        execution_rid: RID | None = None,
        description: str = "",
        dataset_types: list[str] | None = None,
    ) -> DatasetLike:
        """Create a new dataset in the catalog.

        Args:
            version: Initial version string or DatasetVersion. Defaults to None.
            execution_rid: Optional execution to associate with the dataset.
            description: Description of the dataset. Defaults to empty string.
            dataset_types: List of dataset type terms from vocabulary.

        Returns:
            The newly created dataset.
        """
        ...

    @property
    def _dataset_table(self) -> Table:
        """Get the Dataset table from the model.

        Returns:
            The Dataset table object.
        """
        ...
