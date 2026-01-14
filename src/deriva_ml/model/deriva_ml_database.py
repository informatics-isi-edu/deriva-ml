"""Database-backed implementation of DerivaMLCatalog protocol.

This module provides a DerivaMLDatabase class that implements the DerivaMLCatalog
protocol using a DatabaseModel (SQLite) instead of a live catalog connection.
This allows code written against the DerivaMLCatalog protocol to work identically
with both live catalogs and downloaded bags.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Iterable

import pandas as pd
from deriva.core.ermrest_model import Table

from deriva_ml.core.definitions import RID, MLVocab, VocabularyTerm
from deriva_ml.core.exceptions import DerivaMLException, DerivaMLInvalidTerm
from deriva_ml.dataset.aux_classes import DatasetSpec, DatasetVersion
from deriva_ml.dataset.dataset_bag import DatasetBag
from deriva_ml.feature import Feature

if TYPE_CHECKING:
    from deriva_ml.model.database import DatabaseModel


class DerivaMLDatabase:
    """Database-backed implementation of DerivaMLCatalog protocol.

    Provides the same interface as DerivaML but operates on downloaded
    bags via SQLite. Read-only operations work; write operations raise
    DerivaMLException since bags are immutable snapshots.

    This class allows code written against the DerivaMLCatalog protocol
    to work identically with both live catalogs (DerivaML) and downloaded
    bags (DerivaMLDatabase).

    Attributes:
        ml_schema: Name of the ML schema.
        domain_schema: Name of the domain schema.
        model: The underlying DatabaseModel.
        working_dir: Working directory path.
        cache_dir: Cache directory path.
    """

    def __init__(self, database_model: "DatabaseModel"):
        """Create a new DerivaMLDatabase.

        Args:
            database_model: The DatabaseModel containing the SQLite database.
        """
        self._database_model = database_model

    # ==================== Protocol Properties ====================

    @property
    def ml_schema(self) -> str:
        """Get the ML schema name."""
        return self._database_model.ml_schema

    @property
    def domain_schema(self) -> str:
        """Get the domain schema name."""
        return self._database_model.domain_schema

    @property
    def model(self) -> "DatabaseModel":
        """Get the underlying database model."""
        return self._database_model

    @property
    def working_dir(self) -> Path:
        """Get the working directory path."""
        return self._database_model.dbase_path

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory path (same as working_dir for bags)."""
        return self._database_model.dbase_path

    @property
    def catalog_id(self) -> str:
        """Get the catalog ID (derived from bag path)."""
        return str(self._database_model.bag_path)

    @property
    def _dataset_table(self) -> Table:
        """Get the Dataset table from the model."""
        return self._database_model.dataset_table

    # ==================== Read Operations (Supported) ====================

    def lookup_dataset(
        self, dataset: RID | DatasetSpec, deleted: bool = False
    ) -> DatasetBag:
        """Look up a dataset by RID or spec.

        Args:
            dataset: Dataset RID or DatasetSpec to look up.
            deleted: Whether to include deleted datasets (ignored for bags).

        Returns:
            DatasetBag for the specified dataset.

        Raises:
            DerivaMLException: If dataset not found in bag.
        """
        if isinstance(dataset, DatasetSpec):
            rid = dataset.rid
        else:
            rid = dataset

        # Validate the dataset exists
        self._database_model.rid_lookup(rid)

        # Get dataset metadata
        dataset_record = next(
            (d for d in self._database_model._get_table_contents("Dataset") if d["RID"] == rid),
            None
        )
        if not dataset_record:
            raise DerivaMLException(f"Dataset {rid} not found in bag")

        # Get dataset types from association table
        atable = f"Dataset_{MLVocab.dataset_type.value}"
        ds_types = [
            t[MLVocab.dataset_type.value]
            for t in self._database_model._get_table_contents(atable)
            if t["Dataset"] == rid
        ]

        return DatasetBag(
            catalog=self,
            dataset_rid=rid,
            description=dataset_record.get("Description", ""),
            execution_rid=self._database_model._get_dataset_execution(rid).get("Execution"),
            dataset_types=ds_types,
        )

    def find_datasets(self, deleted: bool = False) -> Iterable[DatasetBag]:
        """List all datasets in the bag.

        Args:
            deleted: Whether to include deleted datasets (ignored for bags).

        Returns:
            Iterable of DatasetBag objects.
        """
        # Get dataset types for all datasets from association table
        atable = f"Dataset_{MLVocab.dataset_type.value}"
        ds_types = list(self._database_model._get_table_contents(atable))

        datasets = []
        for dataset in self._database_model._get_table_contents("Dataset"):
            my_types = [t[MLVocab.dataset_type.value] for t in ds_types if t["Dataset"] == dataset["RID"]]
            datasets.append(
                DatasetBag(
                    catalog=self,
                    dataset_rid=dataset["RID"],
                    description=dataset.get("Description", ""),
                    execution_rid=self._database_model._get_dataset_execution(dataset["RID"]).get("Execution"),
                    dataset_types=my_types,
                )
            )
        return datasets

    def lookup_term(self, table: str | Table, term_name: str) -> VocabularyTerm:
        """Look up a vocabulary term by name.

        Args:
            table: Vocabulary table to search.
            term_name: Name or synonym of the term.

        Returns:
            The matching VocabularyTerm.

        Raises:
            DerivaMLException: If table is not a vocabulary or term not found.
        """
        # Get table object if string provided
        if isinstance(table, str):
            table_obj = self._database_model.name_to_table(table)
        else:
            table_obj = table

        # Validate it's a vocabulary table
        if not self._database_model.is_vocabulary(table_obj):
            raise DerivaMLException(f"The table {table} is not a controlled vocabulary")

        # Search for term in SQLite
        for term in self.get_table_as_dict(table_obj.name):
            if term_name == term.get("Name") or (
                term.get("Synonyms") and term_name in term.get("Synonyms", [])
            ):
                # Convert synonyms to list if needed
                synonyms = term.get("Synonyms")
                if synonyms and not isinstance(synonyms, list):
                    synonyms = list(synonyms)
                term["Synonyms"] = synonyms or []
                return VocabularyTerm.model_validate(term)

        raise DerivaMLInvalidTerm(table, term_name)

    def get_table_as_dataframe(self, table: str) -> pd.DataFrame:
        """Get table contents as a pandas DataFrame.

        Args:
            table: Name of the table to retrieve.

        Returns:
            DataFrame containing all table contents.
        """
        return pd.DataFrame(list(self.get_table_as_dict(table)))

    def get_table_as_dict(self, table: str) -> Generator[dict[str, Any], None, None]:
        """Get table contents as dictionaries.

        Args:
            table: Name of the table to retrieve.

        Returns:
            Generator yielding dictionaries for each row.
        """
        yield from self._database_model._get_table_contents(table)

    def list_dataset_element_types(self) -> list[Table]:
        """List the types of elements that can be in datasets.

        Returns:
            List of Table objects representing element types.
        """
        return self._database_model.list_dataset_element_types()

    def find_features(self, table: str | Table) -> Iterable[Feature]:
        """Find features associated with a table.

        Args:
            table: Table to find features for.

        Returns:
            Iterable of Feature objects.
        """
        return self._database_model.find_features(table)

    # ==================== Write Operations (Not Supported) ====================

    def create_dataset(
        self,
        execution_rid: RID | None = None,
        version: DatasetVersion | str | None = None,
        description: str = "",
        dataset_types: list[str] | None = None,
    ) -> DatasetBag:
        """Create a new dataset.

        Raises:
            DerivaMLException: Always, since bags are read-only.
        """
        raise DerivaMLException(
            "Cannot create datasets in a downloaded bag. "
            "Bags are immutable snapshots of catalog data."
        )

    def pathBuilder(self):
        """Get the catalog path builder.

        Raises:
            DerivaMLException: Always, since SQLite doesn't use pathBuilder.
        """
        raise DerivaMLException(
            "pathBuilder is not available for database-backed catalogs. "
            "Use get_table_as_dict() or get_table_as_dataframe() instead."
        )

    def catalog_snapshot(self, version_snapshot: str):
        """Create a catalog snapshot.

        Raises:
            DerivaMLException: Always, since bags are already snapshots.
        """
        raise DerivaMLException(
            "catalog_snapshot is not available for database-backed catalogs. "
            "Bags are already immutable snapshots."
        )

    def resolve_rid(self, rid: RID) -> dict[str, Any]:
        """Resolve a RID to its location.

        For database-backed catalogs, this validates that the RID exists
        in the bag and returns basic information about it.

        Args:
            rid: RID to resolve.

        Returns:
            Dictionary with RID and version information.

        Raises:
            DerivaMLException: If RID not found in bag.
        """
        version = self._database_model.rid_lookup(rid)
        return {"RID": rid, "version": version}
