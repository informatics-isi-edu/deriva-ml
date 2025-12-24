"""A module defining the DatasetLike protocol for dataset operations.

This module contains the definition of the DatasetLike protocol, which
provides an interface for datasets to implement specific functionality related
to listing dataset children. It is particularly useful for ensuring type
compatibility for objects that mimic datasets in their behavior.

Classes:
    DatasetLike: A protocol that specifies methods required for dataset-like
    objects.
"""

from typing import Protocol, runtime_checkable

from deriva_ml.core.definitions import RID


@runtime_checkable
class DatasetLike(Protocol):
    def list_dataset_children(self, dataset_rid: RID, recurse: bool = False) -> list[RID]: ...
