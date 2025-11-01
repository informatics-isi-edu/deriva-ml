from typing import Protocol, runtime_checkable

from deriva_ml.core.definitions import RID

from .aux_classes import DatasetConfig, DatasetConfigList, DatasetSpec, DatasetVersion, VersionPart
from .dataset import Dataset
from .dataset_bag import DatasetBag

__all__ = [
    "Dataset",
    "DatasetSpec",
    "DatasetConfig",
    "DatasetConfigList",
    "DatasetBag",
    "DatasetVersion",
    "VersionPart",
]


@runtime_checkable
class DatasetLike(Protocol):
    def list_dataset_children(self, dataset_rid: RID, recurse: bool = False) -> list[RID]: ...
