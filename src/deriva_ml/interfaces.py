"""A module defining the DatasetLike protocol for dataset operations.

This module contains the definition of the DatasetLike protocol, which
provides an interface for datasets to implement specific functionality related
to listing dataset children. It is particularly useful for ensuring type
compatibility for objects that mimic datasets in their behavior.

Classes:
    DatasetLike: A protocol that specifies methods required for dataset-like
    objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Self, runtime_checkable

from deriva.core import ErmrestSnapshot
from deriva.core.datapath import _SchemaWrapper as SchemaWrapper
from deriva.core.ermrest_catalog import ErmrestCatalog, ResolveRidResult
from deriva.core.ermrest_model import Table

from deriva_ml.core.definitions import RID
from deriva_ml.dataset.aux_classes import DatasetSpec, DatasetVersion
from deriva_ml.model.catalog import DerivaModel


@runtime_checkable
class DatasetLike(Protocol):
    dataset_rid: RID
    execution_rid: RID | None
    _ml_instance: DerivaMLCatalog
    description: str
    _version: DatasetVersion | None
    _version_snapshot: DerivaMLCatalog | None
    dataset_types: str | list[str] | None

    @property
    def version(self) -> DatasetVersion: ...

    def list_dataset_children(self, recurse: bool = False) -> list[Self]: ...

    def list_dataset_members(
        self, recurse: bool = False, limit: int | None = None
    ) -> dict[str, list[dict[str, Any]]]: ...


@runtime_checkable
class DerivaMLCatalog(Protocol):
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
        self, dataset: RID | DatasetSpec, version: DatasetVersion | None = None, deleted: bool = False
    ) -> DatasetLike: ...

    @property
    def _dataset_table(self) -> Table: ...
