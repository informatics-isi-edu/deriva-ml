"""A module defining the DatasetLike protocol for dataset operations.

This module contains the definition of the DatasetLike protocol, which
provides an interface for datasets to implement specific functionality related
to listing dataset children. It is particularly useful for ensuring type
compatibility for objects that mimic datasets in their behavior.

The DatasetLike protocol defines read-only operations that work on both:
- Dataset: Live catalog access via deriva-py/datapath
- DatasetBag: Downloaded bag access via SQLAlchemy/SQLite

Write operations (add_members, increment_version, etc.) are only available
on the Dataset class since bags are immutable snapshots.

Classes:
    DatasetLike: A protocol that specifies methods required for dataset-like
    objects.
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
    dataset_types: list[str] | None

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
            List of child datasets.
        """
        ...

    def list_dataset_parents(self, recurse: bool = False) -> list[Self]:
        """Get parent datasets that contain this dataset.

        Args:
            recurse: Whether to recursively include parents of parents.

        Returns:
            List of parent datasets.
        """
        ...

    def list_dataset_members(
        self, recurse: bool = False, limit: int | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """List members of the dataset.

        Args:
            recurse: Whether to include members of nested datasets.
            limit: Maximum number of members per type. None for no limit.

        Returns:
            Dictionary mapping member types to lists of member records.
        """
        ...

    def list_dataset_element_types(self) -> list[Table]:
        """List the types of elements that can be contained in this dataset.

        Returns:
            List of Table objects representing element types.
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
class DerivaMLCatalog(Protocol):
    """Protocol for catalog-level operations.

    This protocol is implemented by DerivaML (for live catalogs) and can be
    used for type checking catalog operations. It provides methods for:
    - Schema and table access
    - Dataset lookup and creation
    - Vocabulary term lookup
    - Table content retrieval
    """
    ml_schema: str
    domain_schema: str
    model: DerivaModel
    catalog: ErmrestCatalog | ErmrestSnapshot
    cache_dir: Path
    working_dir: Path
    catalog_id: str | int

    def pathBuilder(self) -> SchemaWrapper: ...

    def catalog_snapshot(self, version_snapshot: str) -> Self: ...

    def resolve_rid(self, rid: RID) -> ResolveRidResult: ...

    def lookup_dataset(
        self, dataset: RID | DatasetSpec, deleted: bool = False) -> DatasetLike: ...

    def lookup_term(self, table: str | Table, term_name: str) -> VocabularyTerm: ...

    def create_dataset(
        self,
        execution_rid: RID | None,
        version: DatasetVersion | str | None,
        description: str,
        dataset_types: list[str] | None,
    ) -> DatasetLike: ...

    @property
    def _dataset_table(self) -> Table: ...

    def get_table_as_dataframe(self, table: str) -> pd.DataFrame:
        """Get table contents as a pandas DataFrame.

        Args:
            table: Name of the table to retrieve.

        Returns:
            DataFrame containing table contents.
        """
        ...

    def get_table_as_dict(self, table: str) -> Generator[dict[str, Any], None, None]:
        """Get table contents as dictionaries.

        Args:
            table: Name of the table to retrieve.

        Returns:
            Generator yielding dictionaries for each row.
        """
        ...
