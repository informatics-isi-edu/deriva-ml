"""Cache status detection for downloaded dataset bags.

:class:`BagCache` is the deriva-ml-facing view over the
content-addressed :class:`deriva.bag.cache_index.BagCacheIndex`.
Before the deriva.bag migration this module owned both the on-disk
layout (``{cache_dir}/{dataset_rid}_{checksum}/...``) *and* the
status detection. The on-disk layout had a hard limit: it assumed
exactly one anchoring RID per bag, which doesn't generalize to
the multi-RID / table-anchor / catalog-clone cases that
:class:`~deriva.bag.catalog_builder.CatalogBagBuilder` produces.

After the migration:

- Bag identity lives in the **`BagCacheIndex`** (SQLite at
  ``{cache_dir}/index.sqlite``) — every bag is keyed by its BDBag
  checksum, and the reverse-index ``bag_anchor_rids`` table answers
  "which bags reference this RID?".
- :class:`BagCache` is a thin wrapper over the index for the
  deriva-ml-specific question "what's the cache status for this
  dataset?". The public ``cache_status(dataset_rid)`` API is
  preserved so the one in-tree caller (Dataset.bag_info) doesn't
  change.

The bag-write side (``Dataset.download_dataset_bag``) is rewired
to the new layout in migration step 9 (the ``DatasetBagBuilder``
commit). Until then this class supports *reading* both layouts so
caches written by older code keep working.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from deriva.bag.cache_index import BagCacheIndex
from deriva_ml.core.logging_config import get_logger

logger = get_logger(__name__)


class CacheStatus(StrEnum):
    """Status of a dataset bag in the local cache.

    The value strings preserve the historical names used by the
    pre-migration ``CacheStatus(str, Enum)`` so callers comparing
    against strings keep working. ``cached_incomplete`` is aliased
    to ``cached_holey`` for back-compat (the new term agreed in
    CONTEXT.md).
    """

    not_cached = "not_cached"
    """No local copy exists."""

    cached_metadata_only = "cached_metadata_only"
    """Table data downloaded; asset files not fetched."""

    cached_materialized = "cached_materialized"
    """Fully downloaded — every asset present locally."""

    cached_holey = "cached_holey"
    """Was cached but some asset bytes are missing (per
    ``fetch.txt``). Same state as the historical
    ``cached_incomplete``; the new name matches the
    ``holey bag`` vocabulary in CONTEXT.md."""

    # Back-compat: the historical value name. ``StrEnum`` allows
    # multiple members to share the same name only through aliasing.
    # We surface ``cached_incomplete`` as an attribute that returns
    # the same enum member as ``cached_holey``.
    cached_incomplete = "cached_holey"


class BagCache:
    """Inspect the local cache of downloaded dataset bags.

    Wraps :class:`~deriva.bag.cache_index.BagCacheIndex` to answer
    the deriva-ml-specific question "is this dataset cached, and
    in what state?".

    Args:
        cache_dir: Root cache directory
            (e.g. ``~/.deriva-ml/{hostname}/{catalog_id}/``).

    Example:
        Check whether a dataset bag is cached locally::

            >>> from deriva_ml.dataset.bag_cache import BagCache
            >>> from pathlib import Path
            >>> cache = BagCache(Path.home() / ".deriva-ml" / "host" / "1")  # doctest: +SKIP
            >>> info = cache.cache_status("ABC123")  # doctest: +SKIP
            >>> info["status"]  # doctest: +SKIP
            'cached_materialized'
    """

    def __init__(self, cache_dir: Path):
        self._cache_dir = Path(cache_dir)
        # The index owns the SQLite under ``cache_dir/index.sqlite``
        # and creates the ``bags/`` subtree on demand. Construction
        # is cheap (no eager DB work), so we keep the index for the
        # lifetime of this BagCache.
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index = BagCacheIndex(self._cache_dir)

    @property
    def cache_dir(self) -> Path:
        """Root cache directory."""
        return self._cache_dir

    def cache_status(self, dataset_rid: str) -> dict[str, Any]:
        """Report cache status for a dataset RID.

        Does **not** contact the catalog — this is a purely-local
        filesystem + index operation.

        Args:
            dataset_rid: Dataset RID to look up.

        Returns:
            ``dict`` with:
                - ``status``: :class:`CacheStatus` value (as string).
                - ``cache_path``: Path to the cached bag directory
                  (if cached), else ``None``.
                - ``versions_cached``: List of cache-key checksum
                  strings found for this RID.
        """
        checksums = self._index.find_bags_for_rid(table="Dataset", rid=dataset_rid)

        if not checksums:
            return {
                "status": CacheStatus.not_cached.value,
                "cache_path": None,
                "versions_cached": [],
            }

        # Prefer the most-recently-recorded index entry. ``find_bags_for_rid``
        # already orders by built_at DESC, so the first checksum is the
        # newest. Bags written by ``Dataset._download_dataset_minid`` land at
        # ``bag_dir_for(checksum) / "Dataset_{rid}"`` (the canonical
        # deriva-ml bag name) so we look for the dataset bag at that path.
        checksum = checksums[0]
        bag_dir = self._index.bag_dir_for(checksum) / f"Dataset_{dataset_rid}"
        status = self._determine_index_status(checksum, bag_dir)
        cache_path = str(bag_dir) if bag_dir.exists() else None
        return {
            "status": status.value,
            "cache_path": cache_path,
            "versions_cached": list(checksums),
        }

    # ------------------------------------------------------------------
    # Status detection helpers
    # ------------------------------------------------------------------

    def _determine_index_status(self, checksum: str, bag_path: Path) -> CacheStatus:
        """Compute the status of a bag known to the index.

        The index records that a bag exists; the *on-disk* state
        decides between metadata-only / materialized / holey. We
        check the bag directory's presence and walk
        ``fetch.txt`` (if any) for missing files.
        """
        if not bag_path.exists():
            # Index claims the bag is cached but the directory is
            # gone (manual cleanup, disk full at extract-time).
            # Report not_cached so callers know to re-download.
            return CacheStatus.not_cached
        if self._is_fully_materialized(bag_path):
            return CacheStatus.cached_materialized
        # Directory exists but at least one fetch.txt entry is
        # missing.
        return CacheStatus.cached_holey

    @staticmethod
    def _is_fully_materialized(bag_path: Path) -> bool:
        """Check whether every ``fetch.txt`` entry has been downloaded.

        Args:
            bag_path: Path to the BDBag directory.

        Returns:
            ``True`` when ``fetch.txt`` is absent or every entry's
            referenced file exists locally; ``False`` when at least
            one referenced file is missing or the bag structure
            itself fails validation.
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def dispose(self) -> None:
        """Release the underlying :class:`BagCacheIndex` engine."""
        self._index.dispose()

    def __enter__(self) -> "BagCache":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.dispose()
        return False


__all__ = ["BagCache", "CacheStatus"]
