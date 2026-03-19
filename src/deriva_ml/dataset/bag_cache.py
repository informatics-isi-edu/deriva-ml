"""Cache management for downloaded dataset bags.

Provides a clean interface for checking cache status, prefetching bags,
and getting bag information without requiring an execution context.

The cache is organized as:
    ``~/.deriva-ml/{hostname}/{catalog_id}/datasets/{dataset_rid}_{checksum}/``

Each bag directory contains the extracted BDBag with a ``validated_check.txt``
marker file indicating full materialization.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deriva_ml.dataset.dataset import Dataset

logger = logging.getLogger("deriva_ml")


class CacheStatus(str, Enum):
    """Status of a dataset bag in the local cache."""

    not_cached = "not_cached"
    """No local copy exists."""

    cached_metadata_only = "cached_metadata_only"
    """Table data downloaded, but asset files have not been fetched."""

    cached_materialized = "cached_materialized"
    """Fully downloaded and validated — all assets present."""

    cached_incomplete = "cached_incomplete"
    """Was cached but some assets are missing (needs re-materialization)."""


class BagCache:
    """Manages the local cache of downloaded dataset bags.

    This class provides methods to inspect cache status, prefetch bags,
    and get comprehensive bag information. It operates on the filesystem
    cache without requiring a live catalog connection for status checks.

    Args:
        cache_dir: Root cache directory (e.g., ``~/.deriva-ml/host/catalog/``).
        dataset: The Dataset instance for catalog operations (size estimation,
            downloading). May be None for pure cache-status checks.
    """

    def __init__(self, cache_dir: Path, dataset: Dataset | None = None):
        self._cache_dir = cache_dir
        self._dataset = dataset

    def cache_status(self, dataset_rid: str) -> dict[str, Any]:
        """Check the cache status of a dataset bag.

        Scans the local cache directory for any bags matching this dataset RID.
        Does NOT contact the catalog — this is a purely local operation.

        Args:
            dataset_rid: RID of the dataset to check.

        Returns:
            dict with:
                - status: CacheStatus value
                - cache_path: Path to the bag directory (if cached), else None
                - versions_cached: list of cached version checksums found
        """
        matching_dirs = list(self._cache_dir.glob(f"{dataset_rid}_*"))
        if not matching_dirs:
            return {
                "status": CacheStatus.not_cached.value,
                "cache_path": None,
                "versions_cached": [],
            }

        # Check the most recent (by modification time)
        bag_dir = max(matching_dirs, key=lambda p: p.stat().st_mtime)
        bag_path = bag_dir / f"Dataset_{dataset_rid}"
        validated_check = bag_dir / "validated_check.txt"

        if not bag_path.exists():
            return {
                "status": CacheStatus.not_cached.value,
                "cache_path": None,
                "versions_cached": [d.name for d in matching_dirs],
            }

        status = self._determine_status(bag_path, validated_check)
        return {
            "status": status.value,
            "cache_path": str(bag_path),
            "versions_cached": [d.name for d in matching_dirs],
        }

    def _determine_status(self, bag_path: Path, validated_check: Path) -> CacheStatus:
        """Determine the cache status of a specific bag directory."""
        if not bag_path.exists():
            return CacheStatus.not_cached

        if validated_check.exists():
            # Check if fully materialized
            if self._is_fully_materialized(bag_path):
                return CacheStatus.cached_materialized
            else:
                return CacheStatus.cached_incomplete

        # Bag exists but no validated_check — metadata only
        return CacheStatus.cached_metadata_only

    @staticmethod
    def _is_fully_materialized(bag_path: Path) -> bool:
        """Check whether all fetch.txt entries have been downloaded locally.

        Args:
            bag_path: Path to the BDBag directory.

        Returns:
            True if the bag has no fetch.txt or all entries exist locally.
        """
        from bdbag import bdbag_api as bdb

        try:
            bdb.validate_bag_structure(bag_path.as_posix())
        except Exception as e:
            logger.debug(f"Bag validation check failed for {bag_path}: {e}")
            return False

        fetch_file = bag_path / "fetch.txt"
        if not fetch_file.exists():
            return True

        with fetch_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    rel_path = parts[2]
                    if not (bag_path / rel_path).exists():
                        return False
        return True
