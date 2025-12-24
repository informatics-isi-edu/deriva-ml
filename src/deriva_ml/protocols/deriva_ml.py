from __future__ import annotations

from pathlib import Path
from typing import Protocol, Self, runtime_checkable

from deriva.core.datapath import _SchemaWrapper as SchemaWrapper
from deriva.core.deriva_server import DerivaServer
from deriva.core.ermrest_catalog import ResolveRidResult
from deriva.core.ermrest_model import Table

from ..core.definitions import RID
from ..model.catalog import DerivaModel


@runtime_checkable
class DerivaMLCatalog(Protocol):
    ml_schema: str
    domain_schema: str
    model: DerivaModel
    catalog: DerivaServer
    cache_dir: Path
    working_dir: Path
    _dataset_table: Table

    def pathBuilder(self) -> SchemaWrapper: ...
    def catalog_snapshot(self, version_snapshot: str) -> Self: ...
    def resolve_rid(self, rid: RID) -> ResolveRidResult: ...
