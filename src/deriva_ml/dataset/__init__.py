"""Dataset management for DerivaML.

Provides classes for creating, versioning, splitting, and downloading
datasets, as well as hydra-zen configuration helpers for specifying
dataset inputs in ML workflows.
"""

from .aux_classes import DatasetSpec, DatasetSpecConfig, DatasetVersion, VersionPart
from .dataset import Dataset
from .dataset_bag import (
    DatasetBag,
)
from .split import (
    PartitionInfo,
    SelectionFunction,
    SplitResult,
    random_split,
    split_dataset,
    stratified_split,
)

__all__ = [
    "Dataset",
    "DatasetSpec",
    "DatasetSpecConfig",
    "DatasetBag",
    "DatasetVersion",
    "PartitionInfo",
    "SelectionFunction",
    "SplitResult",
    "VersionPart",
    "random_split",
    "split_dataset",
    "stratified_split",
]
