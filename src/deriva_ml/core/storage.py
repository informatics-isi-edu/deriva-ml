"""Local cache & storage introspection records and helpers.

Owns the typed records returned by the DerivaML storage-introspection
surface (``list_cached_bags`` / ``list_cached_assets``), the asset-
cache list/delete functions, and the index-coherent ``clear_cache``
engine. Bag listing/purging logic lives on
:class:`deriva_ml.dataset.bag_cache.BagCache`, which owns the
underlying :class:`~deriva.bag.cache_index.BagCacheIndex`.

Import-cycle rule: this module may import from
``deriva_ml.dataset.bag_cache`` at module level (it has no deriva-ml
dependencies beyond logging); ``bag_cache`` imports the records
defined here lazily inside method bodies only.

Example:
    >>> from deriva_ml.core.storage import CachedAsset
    >>> CachedAsset.model_fields["md5"].annotation
    <class 'str'>
"""

from __future__ import annotations

import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from deriva_ml.core.logging_config import get_logger
from deriva_ml.dataset.bag_cache import CacheStatus

logger = get_logger(__name__)

# MD5 hex digest length — used to validate "{rid}_{md5}" asset-cache
# directory names.
_MD5_HEX_LEN = 32


class CachedBag(BaseModel):
    """One dataset-anchored bag in the local cache.

    A single content-addressed bag may be anchored by more than one
    dataset RID (e.g. shared content via clone-via-bag); each
    (bag, Dataset-anchor) pair produces one ``CachedBag``, so two
    entries may share a ``checksum``.

    Example:
        >>> from datetime import datetime, timezone
        >>> from pathlib import Path
        >>> from deriva_ml.dataset.bag_cache import CacheStatus
        >>> bag = CachedBag(
        ...     dataset_rid="1ABC", version="1.0.0", checksum="deadbeef",
        ...     status=CacheStatus.cached_materialized,
        ...     built_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ...     size_bytes=2048, path=Path("/tmp/bags/deadbeef/Dataset_1ABC"),
        ... )
        >>> bag.status.value
        'cached_materialized'
    """

    model_config = ConfigDict(frozen=True)

    dataset_rid: str
    version: str | None
    checksum: str
    status: CacheStatus
    built_at: datetime
    size_bytes: int | None
    path: Path


class CachedAsset(BaseModel):
    """One cached input-asset directory (``assets/{rid}_{md5}/``).

    Written by ``Execution.download_asset(use_cache=True)``; the
    directory name encodes the asset RID and the file's MD5.

    Example:
        >>> from datetime import datetime, timezone
        >>> from pathlib import Path
        >>> a = CachedAsset(
        ...     rid="2XYZ", md5="d41d8cd98f00b204e9800998ecf8427e",
        ...     file_count=1, size_bytes=100,
        ...     modified=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ...     path=Path("/tmp/assets/2XYZ_d41d8cd98f00b204e9800998ecf8427e"),
        ... )
        >>> a.file_count
        1
    """

    model_config = ConfigDict(frozen=True)

    rid: str
    md5: str
    file_count: int
    size_bytes: int
    modified: datetime
    path: Path


def _dir_size(path: Path) -> int:
    """Total bytes of all files under ``path`` (0 if missing).

    Example:
        >>> from pathlib import Path
        >>> _dir_size(Path("/nonexistent/anywhere"))
        0
    """
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except FileNotFoundError:
            # Entry vanished mid-walk (concurrent cleanup) — skip it.
            continue
    return total


def _parse_asset_dir_name(name: str) -> tuple[str, str] | None:
    """Split ``{rid}_{md5}`` (last underscore, md5 = 32 hex chars)."""
    rid, sep, md5 = name.rpartition("_")
    if not sep or not rid or len(md5) != _MD5_HEX_LEN:
        return None
    if any(c not in "0123456789abcdef" for c in md5.lower()):
        return None
    return rid, md5


def list_cached_assets(cache_dir: Path) -> list[CachedAsset]:
    """List cached input assets under ``{cache_dir}/assets/``.

    Each conforming ``{rid}_{md5}`` directory yields one
    :class:`CachedAsset`. Non-conforming entries are skipped with a
    debug log — the directory is deriva-ml's, but listing tolerates
    foreign droppings rather than erroring.

    Args:
        cache_dir: The DerivaML cache directory.

    Returns:
        ``CachedAsset`` records sorted by directory name; empty list
        when the assets directory does not exist.

    Example:
        >>> from pathlib import Path
        >>> list_cached_assets(Path("/nonexistent"))
        []
    """
    assets_dir = Path(cache_dir) / "assets"
    if not assets_dir.exists():
        return []
    assets: list[CachedAsset] = []
    for entry in sorted(assets_dir.iterdir()):
        if not entry.is_dir():
            logger.debug("Skipping non-directory in asset cache: %s", entry)
            continue
        parsed = _parse_asset_dir_name(entry.name)
        if parsed is None:
            logger.debug("Skipping non-conforming asset-cache entry: %s", entry)
            continue
        rid, md5 = parsed
        try:
            files = [f for f in entry.rglob("*") if f.is_file()]
            assets.append(
                CachedAsset(
                    rid=rid,
                    md5=md5,
                    file_count=len(files),
                    size_bytes=sum(f.stat().st_size for f in files),
                    modified=datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc),
                    path=entry,
                )
            )
        except (FileNotFoundError, OSError):
            logger.debug("Asset-cache entry vanished mid-listing: %s", entry)
            continue
    return assets


def delete_cached_asset(cache_dir: Path, rid: str, md5: str | None = None) -> dict[str, int]:
    """Delete cached asset directories for a RID.

    Args:
        cache_dir: The DerivaML cache directory.
        rid: Asset RID whose cache entries to remove.
        md5: When given, only the ``{rid}_{md5}`` entry; ``None``
            removes every cached copy of the asset.

    Returns:
        ``{"assets_removed": n, "bytes_freed": n}``; zeros when
        nothing matched (idempotent).

    Example:
        >>> from pathlib import Path
        >>> delete_cached_asset(Path("/nonexistent"), "1ABC")
        {'assets_removed': 0, 'bytes_freed': 0}
    """
    assets_dir = Path(cache_dir) / "assets"
    stats = {"assets_removed": 0, "bytes_freed": 0}
    if not assets_dir.exists():
        return stats
    pattern = f"{rid}_{md5}" if md5 else f"{rid}_*"
    for entry in assets_dir.glob(pattern):
        parsed = _parse_asset_dir_name(entry.name)
        if not entry.is_dir() or parsed is None or parsed[0] != rid:
            continue
        freed = _dir_size(entry)
        try:
            shutil.rmtree(entry)
        except (OSError, PermissionError) as e:
            logger.warning("Failed to remove cached asset %s: %s", entry, e)
            continue
        stats["assets_removed"] += 1
        stats["bytes_freed"] += freed
    return stats


# Cache-dir entries owned by the bag index / asset cache machinery.
# clear_cache handles these through their own coherent paths; only
# entries outside this set get the legacy mtime-walk treatment.
_PROTECTED_CACHE_ENTRIES = frozenset({"bags", "assets", "index.sqlite", "index.sqlite-wal", "index.sqlite-shm"})


def clear_cache(
    cache_dir: Path,
    older_than_days: int | None = None,
    log: logging.Logger | None = None,
) -> dict[str, int]:
    """Clear the dataset cache directory, index-coherently.

    Three passes:

    1. **Bags through the index** — every bag whose ``built_at`` is
       older than the cutoff is removed with
       :meth:`~deriva.bag.cache_index.BagCacheIndex.purge`, dropping
       the index row and the on-disk directory together. The index
       can never claim a bag whose directory this function removed.
    2. **Assets by mtime** — ``assets/*`` entries older than the
       cutoff are removed.
    3. **Stray top-level entries** (anything not in
       ``_PROTECTED_CACHE_ENTRIES``) keep the legacy mtime behavior.

    Args:
        cache_dir: The DerivaML cache directory.
        older_than_days: Only remove entries older than this many
            days; ``None`` removes everything.
        log: Logger for per-entry failures (defaults to the module
            logger).

    Returns:
        ``{"files_removed", "dirs_removed", "bytes_freed", "errors"}``
        — same shape as the historical ``DerivaML.clear_cache``.

    Example:
        >>> from pathlib import Path
        >>> clear_cache(Path("/nonexistent"))
        {'files_removed': 0, 'dirs_removed': 0, 'bytes_freed': 0, 'errors': 0}
    """
    log = log or logger
    stats = {"files_removed": 0, "dirs_removed": 0, "bytes_freed": 0, "errors": 0}
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return stats

    cutoff = time.time() - older_than_days * 86400 if older_than_days is not None else None

    # Pass 1: bags, through the index (never orphan the index).
    if (cache_dir / "index.sqlite").exists():
        from deriva.bag.cache_index import BagCacheIndex

        try:
            index = BagCacheIndex(cache_dir)
        except Exception as e:  # corrupt index / schema-version mismatch
            log.warning("Bag index unusable, skipping bag pass: %s", e)
            stats["errors"] += 1
            index = None
        if index is not None:
            try:
                for row in index.list_bags():
                    if cutoff is not None:
                        built = datetime.fromisoformat(row["built_at"]).timestamp()
                        if built > cutoff:
                            continue
                    bag_root = index.bag_dir_for(row["checksum"])
                    dir_existed = bag_root.exists()
                    freed = _dir_size(bag_root)
                    try:
                        index.purge(row["checksum"])
                    except (OSError, PermissionError) as e:
                        log.warning("Failed to purge cached bag %s: %s", row["checksum"], e)
                        stats["errors"] += 1
                        continue
                    if dir_existed:
                        stats["dirs_removed"] += 1
                        stats["bytes_freed"] += freed
            finally:
                index.dispose()

    # Pass 2: assets, by mtime.
    assets_dir = cache_dir / "assets"
    if assets_dir.exists():
        for entry in assets_dir.iterdir():
            try:
                if cutoff is not None and entry.stat().st_mtime > cutoff:
                    continue
                freed = _dir_size(entry) if entry.is_dir() else entry.stat().st_size
                if entry.is_dir():
                    shutil.rmtree(entry)
                    stats["dirs_removed"] += 1
                else:
                    entry.unlink()
                    stats["files_removed"] += 1
                stats["bytes_freed"] += freed
            except (OSError, PermissionError) as e:
                log.warning("Failed to remove cached asset %s: %s", entry, e)
                stats["errors"] += 1

    # Pass 3: stray top-level entries (legacy behavior).
    try:
        for entry in cache_dir.iterdir():
            if entry.name in _PROTECTED_CACHE_ENTRIES:
                continue
            try:
                if cutoff is not None and entry.stat().st_mtime > cutoff:
                    continue
                freed = _dir_size(entry) if entry.is_dir() else entry.stat().st_size
                if entry.is_dir():
                    shutil.rmtree(entry)
                    stats["dirs_removed"] += 1
                else:
                    entry.unlink()
                    stats["files_removed"] += 1
                stats["bytes_freed"] += freed
            except (OSError, PermissionError) as e:
                log.warning("Failed to remove cache entry %s: %s", entry, e)
                stats["errors"] += 1
    except OSError as e:
        log.error("Failed to iterate cache directory: %s", e)
        stats["errors"] += 1

    return stats
