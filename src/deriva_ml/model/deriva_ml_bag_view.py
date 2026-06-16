"""Deriva-ml-domain view over a bag.

This module provides :class:`DerivaMLBagView`, the deriva-ml-domain
layer over a :class:`~deriva_ml.model.database.DatabaseModel` (which
in turn extends :class:`deriva.bag.database.BagDatabase`). It
implements the read-only side of the
:class:`~deriva_ml.interfaces.DerivaMLCatalog` protocol so callers
written against that protocol work identically with a live catalog
(:class:`~deriva_ml.DerivaML`) and a downloaded bag
(:class:`DerivaMLBagView`).

Historical: this class was previously named ``DerivaMLDatabase``. The
``DerivaMLDatabase`` name suggested an alternative implementation
backed by *a database*, but the class is more precisely a *view*
over a bag's contents. The new name (``DerivaMLBagView``) reflects
ADR-0006's three-class consumer layer: :class:`BagDatabase` (generic,
upstream) → :class:`DatabaseModel` (deriva-ml-extended) →
:class:`DerivaMLBagView` (domain-protocol-shaped, with vocab/feature/
element-type lookup over the bag).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Iterable

import pandas as pd
from deriva.core.ermrest_model import Table
from sqlalchemy import select
from sqlalchemy.orm import Session

from deriva_ml.core.definitions import RID, MLVocab, VocabularyTerm
from deriva_ml.core.exceptions import DerivaMLException, DerivaMLInvalidTerm
from deriva_ml.core.pd_utils import rows_to_dataframe
from deriva_ml.dataset.aux_classes import DatasetSpec, DatasetVersion
from deriva_ml.dataset.dataset_bag import DatasetBag
from deriva_ml.feature import Feature

if TYPE_CHECKING:
    from deriva_ml.model.database import DatabaseModel


class DerivaMLBagView:
    """Deriva-ml-domain view over a bag.

    Provides the same read-side interface as :class:`DerivaML` but
    operates on downloaded bags via the inherited
    :class:`BagDatabase` SQLAlchemy layer. Read-only operations work;
    write operations raise :class:`DerivaMLException` since bags are
    immutable snapshots.

    This class lets code written against the
    :class:`~deriva_ml.interfaces.DerivaMLCatalog` protocol work
    identically with both live catalogs (:class:`DerivaML`) and
    downloaded bags (:class:`DerivaMLBagView`).

    Attributes:
        ml_schema: Name of the ML schema.
        domain_schemas: Frozenset of domain schema names.
        default_schema: Default schema for table creation.
        model: The underlying DatabaseModel.
        working_dir: Working directory path.
        cache_dir: Cache directory path.
    """

    def __init__(self, database_model: "DatabaseModel"):
        """Create a new DerivaMLBagView.

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
    def domain_schemas(self) -> frozenset[str]:
        """Get the domain schema names."""
        return self._database_model.domain_schemas

    @property
    def default_schema(self) -> str | None:
        """Get the default schema name."""
        return self._database_model.default_schema

    @property
    def model(self) -> "DatabaseModel":
        """Get the underlying database model."""
        return self._database_model

    @property
    def working_dir(self) -> Path:
        """Get the working directory path."""
        return self._database_model.database_dir

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory path (same as working_dir for bags)."""
        return self._database_model.database_dir

    @property
    def catalog_id(self) -> str:
        """Get the catalog ID (derived from bag path)."""
        return str(self._database_model.bag_path)

    @property
    def _dataset_table(self) -> Table:
        """Get the Dataset table from the model."""
        return self._database_model.dataset_table

    # ==================== Read Operations (Supported) ====================

    def lookup_dataset(self, dataset: RID | DatasetSpec, deleted: bool = False) -> DatasetBag:
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
        dataset_record = next((d for d in self._database_model.get_table_contents("Dataset") if d["RID"] == rid), None)
        if not dataset_record:
            raise DerivaMLException(f"Dataset {rid} not found in bag")

        # Get dataset types from association table
        atable = f"Dataset_{MLVocab.dataset_type.value}"
        ds_types = [
            t[MLVocab.dataset_type.value]
            for t in self._database_model.get_table_contents(atable)
            if t["Dataset"] == rid
        ]

        return DatasetBag(
            catalog=self,
            dataset_rid=rid,
            description=dataset_record.get("Description", ""),
            execution_rid=(self._get_dataset_execution(rid) or {}).get("Execution"),
            dataset_types=ds_types,
        )

    def find_datasets(self, deleted: bool = False) -> Iterable[DatasetBag]:
        """List all datasets in the bag.

        Args:
            deleted: Whether to include deleted datasets (ignored for bags).

        Returns:
            Iterable of DatasetBag objects.
        """
        # Pre-group the dataset-type association rows by Dataset RID
        # so the per-dataset lookup below is O(1) instead of O(N).
        atable = f"Dataset_{MLVocab.dataset_type.value}"
        types_by_rid: dict[str, list[str]] = {}
        for row in self._database_model.get_table_contents(atable):
            types_by_rid.setdefault(row["Dataset"], []).append(row[MLVocab.dataset_type.value])

        datasets = []
        for dataset in self._database_model.get_table_contents("Dataset"):
            rid = dataset["RID"]
            datasets.append(
                DatasetBag(
                    catalog=self,
                    dataset_rid=rid,
                    description=dataset.get("Description", ""),
                    execution_rid=(self._get_dataset_execution(rid) or {}).get("Execution"),
                    dataset_types=types_by_rid.get(rid, []),
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

        # Fast path: indexed SELECT WHERE Name = term_name. SQLAlchemy
        # routes through SQLite's primary-/secondary-index machinery
        # instead of streaming every row through Python.
        orm_class = self._database_model.get_orm_class_by_name(table_obj.name)
        if orm_class is not None:
            with Session(self._database_model.engine) as session:
                stmt = select(orm_class).where(orm_class.Name == term_name)
                row = session.execute(stmt).scalars().first()
                if row is not None:
                    term = {col.name: getattr(row, col.name) for col in row.__table__.columns}
                    synonyms = term.get("Synonyms")
                    if synonyms and not isinstance(synonyms, list):
                        synonyms = list(synonyms)
                    term["Synonyms"] = synonyms or []
                    return VocabularyTerm.model_validate(term)

        # Slow path: scan for synonyms (no portable indexed query for
        # "term_name in JSON-array Synonyms" across all sqlite versions).
        for term in self.get_table_as_dict(table_obj.name):
            if term.get("Synonyms") and term_name in term.get("Synonyms", []):
                synonyms = term.get("Synonyms")
                if synonyms and not isinstance(synonyms, list):
                    synonyms = list(synonyms)
                term["Synonyms"] = synonyms or []
                return VocabularyTerm.model_validate(term)

        raise DerivaMLInvalidTerm(table, term_name)

    def _get_dataset_execution(self, dataset_rid: str) -> dict[str, Any] | None:
        """Return the ``Dataset_Version`` row for ``(rid, current_version)``.

        Lives on the view (not on :class:`DatabaseModel`) because the
        join through ``self._database_model.bag_rids`` plus the
        ``Dataset_Version`` schema-table lookup are both deriva-ml
        domain knowledge — the generic :class:`BagDatabase` layer
        doesn't know that "dataset" means a versioned thing.

        Args:
            dataset_rid: Dataset RID to look up.

        Returns:
            The matching ``Dataset_Version`` row as a dict, or
            ``None`` if either the RID isn't in this bag or no row
            matches the version we hold for it.
        """
        version = self._database_model.bag_rids.get(dataset_rid)
        if not version:
            return None

        dataset_version_table = self._database_model.find_table("Dataset_Version")
        stmt = select(dataset_version_table).where(
            dataset_version_table.columns.Dataset == dataset_rid,
            dataset_version_table.columns.Version == str(version),
        )
        with Session(self._database_model.engine) as session:
            result = session.execute(stmt).mappings().first()
            return dict(result) if result else None

    def get_table_as_dataframe(self, table: str) -> pd.DataFrame:
        """Get table contents as a pandas DataFrame.

        Args:
            table: Name of the table to retrieve.

        Returns:
            DataFrame containing all table contents.
        """
        return rows_to_dataframe(self.get_table_as_dict(table))

    def get_table_as_dict(self, table: str) -> Generator[dict[str, Any], None, None]:
        """Get table contents as dictionaries.

        Args:
            table: Name of the table to retrieve.

        Returns:
            Generator yielding dictionaries for each row.
        """
        yield from self._database_model.get_table_contents(table)

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
            "Cannot create datasets in a downloaded bag. Bags are immutable snapshots of catalog data."
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
            "catalog_snapshot is not available for database-backed catalogs. Bags are already immutable snapshots."
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
