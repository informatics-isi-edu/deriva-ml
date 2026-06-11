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

import sqlite3
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from deriva.bag.cache_index import BagCacheIndex
from deriva_ml.core.logging_config import get_logger

logger = get_logger(__name__)


def _utc(ts: datetime) -> datetime:
    """Attach UTC to a naive datetime; pass tz-aware through unchanged.

    ``BagCacheIndex.record()`` defaults to tz-aware UTC timestamps,
    but accepts naive ones — normalize so ``CachedBag.built_at`` is
    always comparable.
    """
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


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

            >>> from deriva_ml.dataset.bag_cache import BagCache  # doctest: +SKIP
            >>> from pathlib import Path  # doctest: +SKIP
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

    def list_bags(self) -> "list[CachedBag]":
        """List every dataset-anchored bag in the local cache.

        Joins the index's bag rows with the ``bag_anchor_rids``
        reverse index: one :class:`~deriva_ml.core.storage.CachedBag`
        per (bag, Dataset-anchor) pair, most-recently-built first.
        Purely local — no catalog access.

        Returns:
            List of ``CachedBag`` records (empty when nothing cached).

        Example:
            >>> from deriva_ml.dataset.bag_cache import BagCache  # doctest: +SKIP
            >>> with BagCache(cache_dir) as cache:  # doctest: +SKIP
            ...     for bag in cache.list_bags():  # doctest: +SKIP
            ...         print(bag.dataset_rid, bag.version, bag.status.value)  # doctest: +SKIP
        """
        # Lazy import: core.storage imports CacheStatus from this
        # module at module level; importing the record lazily here
        # keeps the import graph acyclic.
        from deriva_ml.core.storage import CachedBag, _dir_size

        anchors = self._dataset_anchors()
        bags: list[CachedBag] = []
        for row in self._index.list_bags():
            checksum = row["checksum"]
            version = (row.get("anchor_summary") or {}).get("version")
            for rid in anchors.get(checksum, []):
                bag_dir = self._index.bag_dir_for(checksum) / f"Dataset_{rid}"
                status = self._determine_index_status(checksum, bag_dir)
                bags.append(
                    CachedBag(
                        dataset_rid=rid,
                        version=version,
                        checksum=checksum,
                        status=status,
                        built_at=_utc(datetime.fromisoformat(row["built_at"])),
                        size_bytes=row.get("size_bytes") or _dir_size(bag_dir) or None,
                        path=bag_dir,
                    )
                )
        return bags

    def purge_dataset(self, dataset_rid: str, version: str | None = None) -> dict[str, int]:
        """Delete cached bags for a dataset (all versions, or one).

        Each matching bag is removed via
        :meth:`~deriva.bag.cache_index.BagCacheIndex.purge`, which
        drops the index row and the on-disk directory together — the
        index never outlives its referent.

        Caution: a content-addressed bag anchored by *several*
        datasets is removed for all of them (the cache is always
        re-downloadable, so this is at worst a re-fetch).

        Args:
            dataset_rid: Dataset RID whose cached bags to remove.
            version: When given, only the bag(s) whose recorded
                ``anchor_summary['version']`` matches. ``None``
                removes every cached version.

        Returns:
            ``{"bags_removed": n, "bytes_freed": n}``. Unknown RID
            (or version) yields zeros — deletion is idempotent.

        Example:
            >>> with BagCache(cache_dir) as cache:  # doctest: +SKIP
            ...     cache.purge_dataset("1ABC", version="1.2.0")  # doctest: +SKIP
            {'bags_removed': 1, 'bytes_freed': 52431}
        """
        from deriva_ml.core.storage import _dir_size

        stats = {"bags_removed": 0, "bytes_freed": 0}
        for checksum in self._index.find_bags_for_rid(table="Dataset", rid=dataset_rid):
            if version is not None:
                row = self._index.get(checksum) or {}
                recorded = (row.get("anchor_summary") or {}).get("version")
                if recorded != str(version):
                    continue
            bag_root = self._index.bag_dir_for(checksum)
            freed = _dir_size(bag_root)
            if self._index.purge(checksum):
                stats["bags_removed"] += 1
                stats["bytes_freed"] += freed
        return stats

    def _dataset_anchors(self) -> dict[str, list[str]]:
        """Map checksum -> Dataset RIDs from the reverse index.

        ``BagCacheIndex`` exposes rid->checksums
        (:meth:`~deriva.bag.cache_index.BagCacheIndex.find_bags_for_rid`)
        but not the reverse, so read ``bag_anchor_rids`` directly with
        a read-only sqlite3 connection. Safe against schema drift:
        ``self._index`` was constructed first, and its schema-version
        guard raises before this query can see an unexpected layout.
        (Follow-up: a public ``anchors_for()`` upstream would replace
        this.)
        """
        db = self._cache_dir / "index.sqlite"
        if not db.exists():
            return {}
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                'SELECT checksum, rid FROM bag_anchor_rids WHERE "table" = ?',
                ("Dataset",),
            ).fetchall()
        finally:
            conn.close()
        anchors: dict[str, list[str]] = {}
        for checksum, rid in rows:
            anchors.setdefault(checksum, []).append(rid)
        return anchors

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
