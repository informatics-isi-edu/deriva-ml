"""RID-keyed mapping + iterable over DatasetBag instances.

Per spec §2.8. Returned by Execution.datasets so users can write
`bag = exe.datasets["1-XYZ"]` (primary access pattern) or iterate
with `for bag in exe.datasets:`. Replaces the previous
list[DatasetBag] exposure (hard cutover per R5.1).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deriva_ml.dataset.dataset_bag import DatasetBag


class DatasetCollection(Mapping):
    """Immutable RID-keyed mapping plus iterable of DatasetBags.

    Backed by a list — the bags are already materialized when the
    collection is constructed. No lazy loading; no mutation after
    construction.

    Iteration yields bags (not keys, which is the Mapping default).
    This matches the intuition "iterate the datasets I materialized",
    which is overwhelmingly what callers want. Use ``.keys()`` for
    RIDs and ``.items()`` for (rid, bag) pairs.

    Example:
        >>> for bag in exe.datasets:
        ...     print(bag.dataset_rid, len(bag.list_dataset_members()))
        >>> specific = exe.datasets["1-XYZ"]
        >>> "1-XYZ" in exe.datasets
        True
    """

    def __init__(self, bags: "list[DatasetBag]") -> None:
        """Build from a list of DatasetBag instances.

        Args:
            bags: Already-materialized bags, in the order the user
                declared them in ExecutionConfiguration.datasets.
                Multiple bags with the same dataset_rid are not
                supported; the last wins on __getitem__.
        """
        # Preserve order for iteration.
        self._bags = list(bags)
        # Dict for O(1) RID lookup.
        self._by_rid = {b.dataset_rid: b for b in self._bags}

    def __getitem__(self, rid: str) -> "DatasetBag":
        """Retrieve a bag by its dataset RID.

        Args:
            rid: Dataset RID.

        Returns:
            The DatasetBag instance for that RID.

        Raises:
            KeyError: If the RID is not in this collection. Error
                message includes the available RIDs for debugging.
        """
        try:
            return self._by_rid[rid]
        except KeyError:
            # More helpful than KeyError('1-XYZ') alone.
            available = ", ".join(self._by_rid) or "(none)"
            raise KeyError(
                f"dataset {rid!r} not in this execution's inputs. "
                f"Available: {available}"
            ) from None

    def __iter__(self) -> "Iterator[DatasetBag]":
        # Mapping's default would iterate keys; we override to iterate
        # values (bags) because that's the overwhelmingly common use.
        return iter(self._bags)

    def __len__(self) -> int:
        return len(self._bags)

    def __contains__(self, rid: object) -> bool:
        return rid in self._by_rid

    def keys(self):
        """Dataset RIDs in declaration order."""
        return list(self._by_rid.keys())

    def values(self):
        """DatasetBag instances in declaration order."""
        return list(self._bags)

    def items(self):
        """(rid, bag) pairs in declaration order."""
        return [(b.dataset_rid, b) for b in self._bags]
