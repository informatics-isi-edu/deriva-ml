# Cache & Storage Introspection API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `DerivaML` direct inspect/delete APIs for all three local-storage species — cached bags, cached assets, execution working dirs — and fix `clear_cache()` so the bag index can never disagree with the disk.

**Architecture:** `BagCache` (which owns the SQLite `BagCacheIndex` and the status-detection logic) grows `list_bags()` / `purge_dataset()`. A new `core/storage.py` holds the Pydantic records (`CachedBag`, `CachedAsset`), the asset-cache functions, and the index-coherent `clear_cache` engine. `DerivaML` gains thin delegating methods so the public surface stays one flat family next to the existing six storage commands.

**Tech Stack:** Python 3.12+, Pydantic v2, `deriva.bag.cache_index.BagCacheIndex` (deriva-py), `bdbag` (test fixtures), pytest.

**Spec:** `docs/superpowers/specs/2026-06-11-cache-introspection-design.md`
**Branch:** `feat/cache-introspection` (already created off `origin/main`; the spec commit is on it).

---

## Context for the implementer (read first)

- **CWD discipline:** every command below assumes
  `cd /Users/carl/GitHub/DerivaML/deriva-ml && ...` chained in ONE Bash
  call. Tests run with `DERIVA_ML_ALLOW_DIRTY=true uv run pytest`.
- **On-disk layout** (per `(host, catalog)` cache dir):
  - `{cache_dir}/index.sqlite` — `BagCacheIndex` SQLite (WAL). Tables:
    `bags(checksum, profile_id, built_at, anchor_summary_json,
    size_bytes)` and `bag_anchor_rids(checksum, "table", rid)`.
  - `{cache_dir}/bags/{checksum}/Dataset_{rid}/` — extracted bag dirs.
  - `{cache_dir}/assets/{rid}_{md5}/` — cached input assets.
- **Key upstream APIs** (`deriva.bag.cache_index.BagCacheIndex`):
  `list_bags()` (all bags, built_at DESC, `anchor_summary` inflated),
  `find_bags_for_rid(table=, rid=)` (rid → checksums),
  `get(checksum)`, `purge(checksum)` (index row + on-disk dir
  together), `bag_dir_for(checksum)`, `record(...)` (test fixture
  hook). Constructing `BagCacheIndex` runs a schema-version guard
  (`SchemaVersionError` on mismatch).
- **Upstream API gap:** there is no public checksum→anchors query
  (only rid→checksums). We read `bag_anchor_rids` with a read-only
  stdlib `sqlite3` connection — always AFTER constructing the index in
  the same code path, so the schema-version guard fires before our raw
  query can see an unexpected schema. A public `anchors_for()` in
  deriva-py is noted as a follow-up, not blocking.
- **Status detection** (`BagCache._determine_index_status(checksum,
  bag_path)`): dir missing → `not_cached`; valid bag with no missing
  `fetch.txt` entries → `cached_materialized`; else `cached_holey`.
  Real bags pass `bdbag.bdbag_api.validate_bag_structure`, so test
  fixtures must create *valid* bags via `bdbag_api.make_bag` (a
  hand-rolled directory fails validation and skews status).
- **Production bags are recorded** with
  `anchors=[("Dataset", rid)]` and
  `anchor_summary={"version": str(version)}`
  (see `src/deriva_ml/dataset/bag_download.py:233`).
- **RIDs are opaque** — test RIDs come from fixture-recorded synthetic
  index rows (unit) or live catalog lookups (integration); never
  literals asserted against catalog data. Synthetic unit-test RIDs are
  arbitrary strings fed in AND read back through the same fixture
  call, which is allowed (shape/equality against values the test
  itself routed through the index).

### File map

| File | Action | Responsibility |
|---|---|---|
| `src/deriva_ml/core/storage.py` | create | `CachedBag`/`CachedAsset` records, `_dir_size`, `list_cached_assets`, `delete_cached_asset`, `clear_cache` engine |
| `src/deriva_ml/dataset/bag_cache.py` | modify | `BagCache.list_bags()`, `BagCache.purge_dataset()`, `_dataset_anchors()` helper |
| `src/deriva_ml/core/base.py` | modify | 4 new thin methods; `clear_cache` body → delegation; `get_storage_summary` extension |
| `src/deriva_ml/__init__.py` | modify | lazy exports `CachedBag`, `CachedAsset` |
| `tests/core/test_cache_introspection.py` | create | unit tests (no catalog) |
| `tests/core/test_storage_management.py` | verify-only | existing `clear_cache` tests must still pass (stray-entry back-compat) |
| `tests/execution/test_storage.py` | modify | integration tests (live catalog) |
| `docs/user-guide/offline.md` | modify | "What's in my cache?" section |

Import-cycle rule: `core/storage.py` imports `CacheStatus` from
`dataset/bag_cache.py` at module level (safe — `bag_cache` only
imports deriva-py + logging). `bag_cache.py` imports `CachedBag` from
`core/storage.py` **lazily inside methods** (never at module level).

---

## Task 1: Records module + test scaffolding

**Files:**
- Create: `src/deriva_ml/core/storage.py`
- Create: `tests/core/test_cache_introspection.py`

- [ ] **Step 1: Write the failing test (records construct + serialize)**

Create `tests/core/test_cache_introspection.py`:

```python
"""Unit tests for cache & storage introspection (spec 2026-06-11).

No live catalog: bags are synthesized with ``BagCacheIndex.record()``
plus ``bdbag_api.make_bag`` so status detection sees structurally
valid bags. Assets are synthesized as ``{rid}_{md5}`` directories.

DerivaML methods are exercised unbound against a lightweight harness
(same pattern as tests/core/test_storage_management.py).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_valid_bag(bag_dir: Path, files: dict[str, str] | None = None) -> None:
    """Create a structurally valid BDBag at ``bag_dir``.

    ``BagCache._is_fully_materialized`` runs
    ``bdbag_api.validate_bag_structure``; a hand-rolled directory
    fails it, so build a real bag in place.
    """
    from bdbag import bdbag_api as bdb

    bag_dir.mkdir(parents=True, exist_ok=True)
    for name, content in (files or {"data.csv": "a,b\n1,2\n"}).items():
        (bag_dir / name).write_text(content)
    bdb.make_bag(bag_dir.as_posix())


def _record_bag(
    cache_dir: Path,
    *,
    checksum: str,
    dataset_rid: str,
    version: str = "1.0.0",
    built_at: datetime | None = None,
    on_disk: bool = True,
    holey: bool = False,
) -> Path:
    """Record a synthetic bag in the index (and optionally on disk).

    Mirrors the production record() call in
    ``dataset/bag_download.py`` — anchors ``[("Dataset", rid)]``,
    ``anchor_summary={"version": ...}``.

    Returns the bag directory path (``bags/{checksum}/Dataset_{rid}``).
    """
    from deriva.bag.cache_index import BagCacheIndex

    index = BagCacheIndex(cache_dir)
    try:
        index.record(
            checksum=checksum,
            anchors=[("Dataset", dataset_rid)],
            anchor_summary={"version": version},
            built_at=built_at,
        )
        bag_dir = index.bag_dir_for(checksum) / f"Dataset_{dataset_rid}"
    finally:
        index.dispose()
    if on_disk:
        _make_valid_bag(bag_dir)
        if holey:
            # A fetch.txt entry referencing a file that was never
            # fetched marks the bag holey.
            (bag_dir / "fetch.txt").write_text(
                "https://example.org/x\t10\tdata/missing.bin\n"
            )
    return bag_dir


def _make_cached_asset(
    cache_dir: Path, rid: str, md5: str, n_files: int = 1
) -> Path:
    """Create a synthetic cached asset dir ``assets/{rid}_{md5}``."""
    asset_dir = cache_dir / "assets" / f"{rid}_{md5}"
    asset_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        (asset_dir / f"file{i}.bin").write_bytes(b"x" * 100)
    return asset_dir


class _StorageHarness:
    """Stand-in for ``self`` in DerivaML storage methods.

    Carries the attributes the storage surface reads:
    ``working_dir``, ``cache_dir``, ``_logger``.
    """

    def __init__(self, working_dir: Path, cache_dir: Path):
        self.working_dir = working_dir
        self.cache_dir = cache_dir
        self._logger = logging.getLogger("test")


@pytest.fixture
def harness(tmp_path: Path) -> _StorageHarness:
    working_dir = tmp_path / "wd"
    cache_dir = tmp_path / "cache"
    working_dir.mkdir()
    cache_dir.mkdir()
    return _StorageHarness(working_dir, cache_dir)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class TestRecords:
    def test_cached_bag_fields_and_dump(self, tmp_path: Path):
        from deriva_ml.core.storage import CachedBag
        from deriva_ml.dataset.bag_cache import CacheStatus

        bag = CachedBag(
            dataset_rid="XYZ1",
            version="1.0.0",
            checksum="abc123",
            status=CacheStatus.cached_materialized,
            built_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
            size_bytes=1024,
            path=tmp_path / "bags" / "abc123" / "Dataset_XYZ1",
        )
        dumped = bag.model_dump()
        assert dumped["dataset_rid"] == "XYZ1"
        assert dumped["status"] == "cached_materialized"

    def test_cached_asset_fields_and_dump(self, tmp_path: Path):
        from deriva_ml.core.storage import CachedAsset

        asset = CachedAsset(
            rid="A1",
            md5="d41d8cd98f00b204e9800998ecf8427e",
            file_count=2,
            size_bytes=200,
            modified=datetime(2026, 6, 1, tzinfo=timezone.utc),
            path=tmp_path / "assets" / "A1_d41d8cd98f00b204e9800998ecf8427e",
        )
        assert asset.model_dump()["rid"] == "A1"
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_cache_introspection.py -v 2>&1 | tail -5
```
Expected: FAIL / collection error — `No module named 'deriva_ml.core.storage'`.

- [ ] **Step 3: Create `src/deriva_ml/core/storage.py` with the records**

```python
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

import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
```

- [ ] **Step 4: Run to verify the records tests pass**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_cache_introspection.py -v 2>&1 | tail -8
```
Expected: `TestRecords` 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add src/deriva_ml/core/storage.py tests/core/test_cache_introspection.py && \
git commit -m "feat(storage): CachedBag/CachedAsset records + test scaffolding"
```

---

## Task 2: `BagCache.list_bags()`

**Files:**
- Modify: `src/deriva_ml/dataset/bag_cache.py` (after `cache_status`, before the status helpers)
- Test: `tests/core/test_cache_introspection.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_cache_introspection.py`:

```python
# ---------------------------------------------------------------------------
# BagCache.list_bags
# ---------------------------------------------------------------------------


class TestListBags:
    def test_empty_cache_returns_empty_list(self, tmp_path: Path):
        from deriva_ml.dataset.bag_cache import BagCache

        with BagCache(tmp_path / "cache") as cache:
            assert cache.list_bags() == []

    def test_lists_recorded_bags_with_rid_version_status(self, tmp_path: Path):
        from deriva_ml.dataset.bag_cache import BagCache, CacheStatus

        cache_dir = tmp_path / "cache"
        _record_bag(cache_dir, checksum="aaa111", dataset_rid="RID-A", version="1.0.0")
        _record_bag(cache_dir, checksum="bbb222", dataset_rid="RID-B", version="2.0.0", holey=True)

        with BagCache(cache_dir) as cache:
            bags = cache.list_bags()

        assert {b.dataset_rid for b in bags} == {"RID-A", "RID-B"}
        by_rid = {b.dataset_rid: b for b in bags}
        assert by_rid["RID-A"].version == "1.0.0"
        assert by_rid["RID-A"].status == CacheStatus.cached_materialized
        assert by_rid["RID-B"].status == CacheStatus.cached_holey
        assert by_rid["RID-A"].size_bytes > 0
        assert by_rid["RID-A"].path.exists()

    def test_multi_anchor_bag_yields_one_entry_per_dataset(self, tmp_path: Path):
        from deriva.bag.cache_index import BagCacheIndex
        from deriva_ml.dataset.bag_cache import BagCache

        cache_dir = tmp_path / "cache"
        bag_dir = _record_bag(cache_dir, checksum="ccc333", dataset_rid="RID-X")
        # Second dataset anchors the same content-addressed bag.
        index = BagCacheIndex(cache_dir)
        try:
            index.record(checksum="ccc333", anchors=[("Dataset", "RID-Y")])
        finally:
            index.dispose()

        with BagCache(cache_dir) as cache:
            bags = cache.list_bags()

        assert len(bags) == 2
        assert {b.dataset_rid for b in bags} == {"RID-X", "RID-Y"}
        assert {b.checksum for b in bags} == {"ccc333"}

    def test_index_row_with_missing_dir_reports_not_cached(self, tmp_path: Path):
        from deriva_ml.dataset.bag_cache import BagCache, CacheStatus

        cache_dir = tmp_path / "cache"
        _record_bag(cache_dir, checksum="ddd444", dataset_rid="RID-G", on_disk=False)

        with BagCache(cache_dir) as cache:
            bags = cache.list_bags()

        assert len(bags) == 1
        assert bags[0].status == CacheStatus.not_cached
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_cache_introspection.py::TestListBags -v 2>&1 | tail -6
```
Expected: FAIL — `AttributeError: 'BagCache' object has no attribute 'list_bags'`.

- [ ] **Step 3: Implement on `BagCache`**

In `src/deriva_ml/dataset/bag_cache.py`, add after `cache_status()`
(keep the existing imports; add `import sqlite3` and
`from datetime import datetime` to the module imports):

```python
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
                        built_at=datetime.fromisoformat(row["built_at"]),
                        size_bytes=row.get("size_bytes") or _dir_size(bag_dir) or None,
                        path=bag_dir,
                    )
                )
        return bags

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
```

- [ ] **Step 4: Run to verify pass**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_cache_introspection.py -v 2>&1 | tail -8
```
Expected: all tests pass (TestRecords + TestListBags).

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add src/deriva_ml/dataset/bag_cache.py tests/core/test_cache_introspection.py && \
git commit -m "feat(storage): BagCache.list_bags() — enumerate cached bags with RID/version/status"
```

---

## Task 3: `BagCache.purge_dataset()`

**Files:**
- Modify: `src/deriva_ml/dataset/bag_cache.py` (after `list_bags`)
- Test: `tests/core/test_cache_introspection.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
# ---------------------------------------------------------------------------
# BagCache.purge_dataset
# ---------------------------------------------------------------------------


class TestPurgeDataset:
    def test_purge_all_versions(self, tmp_path: Path):
        from deriva_ml.dataset.bag_cache import BagCache, CacheStatus

        cache_dir = tmp_path / "cache"
        d1 = _record_bag(cache_dir, checksum="e1", dataset_rid="RID-P", version="1.0.0")
        d2 = _record_bag(cache_dir, checksum="e2", dataset_rid="RID-P", version="2.0.0")
        _record_bag(cache_dir, checksum="e3", dataset_rid="RID-Q", version="1.0.0")

        with BagCache(cache_dir) as cache:
            stats = cache.purge_dataset("RID-P")
            remaining = cache.list_bags()
            status = cache.cache_status("RID-P")

        assert stats["bags_removed"] == 2
        assert stats["bytes_freed"] > 0
        assert not d1.exists() and not d2.exists()
        assert {b.dataset_rid for b in remaining} == {"RID-Q"}
        assert status["status"] == CacheStatus.not_cached.value

    def test_purge_single_version(self, tmp_path: Path):
        from deriva_ml.dataset.bag_cache import BagCache

        cache_dir = tmp_path / "cache"
        _record_bag(cache_dir, checksum="f1", dataset_rid="RID-V", version="1.0.0")
        kept = _record_bag(cache_dir, checksum="f2", dataset_rid="RID-V", version="2.0.0")

        with BagCache(cache_dir) as cache:
            stats = cache.purge_dataset("RID-V", version="1.0.0")
            remaining = cache.list_bags()

        assert stats["bags_removed"] == 1
        assert kept.exists()
        assert [b.version for b in remaining] == ["2.0.0"]

    def test_purge_unknown_rid_is_idempotent_zero(self, tmp_path: Path):
        from deriva_ml.dataset.bag_cache import BagCache

        with BagCache(tmp_path / "cache") as cache:
            assert cache.purge_dataset("NOPE") == {"bags_removed": 0, "bytes_freed": 0}
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_cache_introspection.py::TestPurgeDataset -v 2>&1 | tail -6
```
Expected: FAIL — no attribute `purge_dataset`.

- [ ] **Step 3: Implement**

Add to `BagCache` after `list_bags`:

```python
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
```

- [ ] **Step 4: Run to verify pass**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_cache_introspection.py -v 2>&1 | tail -8
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add src/deriva_ml/dataset/bag_cache.py tests/core/test_cache_introspection.py && \
git commit -m "feat(storage): BagCache.purge_dataset() — per-dataset/per-version bag deletion"
```

---

## Task 4: Asset-cache list + delete (`core/storage.py`)

**Files:**
- Modify: `src/deriva_ml/core/storage.py`
- Test: `tests/core/test_cache_introspection.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
# ---------------------------------------------------------------------------
# Asset cache: list_cached_assets / delete_cached_asset
# ---------------------------------------------------------------------------

MD5_A = "d41d8cd98f00b204e9800998ecf8427e"
MD5_B = "9e107d9d372bb6826bd81d3542a419d6"


class TestCachedAssets:
    def test_empty_or_missing_assets_dir(self, tmp_path: Path):
        from deriva_ml.core.storage import list_cached_assets

        assert list_cached_assets(tmp_path / "cache") == []

    def test_lists_assets_with_parsed_rid_md5(self, tmp_path: Path):
        from deriva_ml.core.storage import list_cached_assets

        cache_dir = tmp_path / "cache"
        _make_cached_asset(cache_dir, "RID-1", MD5_A, n_files=2)
        _make_cached_asset(cache_dir, "RID-2", MD5_B)

        assets = list_cached_assets(cache_dir)
        assert {(a.rid, a.md5) for a in assets} == {("RID-1", MD5_A), ("RID-2", MD5_B)}
        by_rid = {a.rid: a for a in assets}
        assert by_rid["RID-1"].file_count == 2
        assert by_rid["RID-1"].size_bytes == 200

    def test_nonconforming_entry_skipped(self, tmp_path: Path):
        from deriva_ml.core.storage import list_cached_assets

        cache_dir = tmp_path / "cache"
        _make_cached_asset(cache_dir, "RID-1", MD5_A)
        (cache_dir / "assets" / "not-an-asset").mkdir()
        (cache_dir / "assets" / "stray.txt").write_text("x")

        assets = list_cached_assets(cache_dir)
        assert [a.rid for a in assets] == ["RID-1"]

    def test_delete_specific_md5(self, tmp_path: Path):
        from deriva_ml.core.storage import delete_cached_asset

        cache_dir = tmp_path / "cache"
        gone = _make_cached_asset(cache_dir, "RID-1", MD5_A)
        kept = _make_cached_asset(cache_dir, "RID-1", MD5_B)

        stats = delete_cached_asset(cache_dir, "RID-1", md5=MD5_A)
        assert stats["assets_removed"] == 1
        assert stats["bytes_freed"] == 100
        assert not gone.exists() and kept.exists()

    def test_delete_all_for_rid(self, tmp_path: Path):
        from deriva_ml.core.storage import delete_cached_asset

        cache_dir = tmp_path / "cache"
        _make_cached_asset(cache_dir, "RID-1", MD5_A)
        _make_cached_asset(cache_dir, "RID-1", MD5_B)
        other = _make_cached_asset(cache_dir, "RID-2", MD5_A)

        stats = delete_cached_asset(cache_dir, "RID-1")
        assert stats["assets_removed"] == 2
        assert other.exists()

    def test_delete_missing_is_idempotent_zero(self, tmp_path: Path):
        from deriva_ml.core.storage import delete_cached_asset

        assert delete_cached_asset(tmp_path / "cache", "NOPE") == {
            "assets_removed": 0,
            "bytes_freed": 0,
        }
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_cache_introspection.py::TestCachedAssets -v 2>&1 | tail -6
```
Expected: FAIL — `cannot import name 'list_cached_assets'`.

- [ ] **Step 3: Implement in `core/storage.py`**

Append:

```python
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
        if not entry.is_dir() or _parse_asset_dir_name(entry.name) is None:
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
```

- [ ] **Step 4: Run to verify pass**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_cache_introspection.py -v 2>&1 | tail -8
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add src/deriva_ml/core/storage.py tests/core/test_cache_introspection.py && \
git commit -m "feat(storage): list/delete cached assets ({rid}_{md5} cache entries)"
```

---

## Task 5: Index-coherent `clear_cache` engine

**Files:**
- Modify: `src/deriva_ml/core/storage.py`
- Test: `tests/core/test_cache_introspection.py`

- [ ] **Step 1: Write the failing tests (incl. the coherence regression test)**

Append:

```python
# ---------------------------------------------------------------------------
# clear_cache (index-coherent engine)
# ---------------------------------------------------------------------------


class TestClearCacheCoherent:
    def test_clears_bags_assets_and_strays(self, tmp_path: Path):
        from deriva_ml.core.storage import clear_cache
        from deriva_ml.dataset.bag_cache import BagCache

        cache_dir = tmp_path / "cache"
        _record_bag(cache_dir, checksum="g1", dataset_rid="RID-C")
        _make_cached_asset(cache_dir, "RID-1", MD5_A)
        (cache_dir / "stray.txt").write_text("x")

        stats = clear_cache(cache_dir)

        assert stats["errors"] == 0
        assert stats["bytes_freed"] > 0
        with BagCache(cache_dir) as cache:
            assert cache.list_bags() == []           # index agrees: nothing cached
        assert list((cache_dir / "assets").iterdir()) == []
        assert not (cache_dir / "stray.txt").exists()

    def test_age_filter_uses_index_built_at_for_bags(self, tmp_path: Path):
        from deriva_ml.core.storage import clear_cache
        from deriva_ml.dataset.bag_cache import BagCache

        cache_dir = tmp_path / "cache"
        old = datetime.now(timezone.utc) - timedelta(days=40)
        _record_bag(cache_dir, checksum="h1", dataset_rid="RID-OLD", built_at=old)
        fresh_dir = _record_bag(cache_dir, checksum="h2", dataset_rid="RID-NEW")

        stats = clear_cache(cache_dir, older_than_days=30)

        assert stats["dirs_removed"] == 1
        with BagCache(cache_dir) as cache:
            remaining = cache.list_bags()
        assert [b.dataset_rid for b in remaining] == ["RID-NEW"]
        assert fresh_dir.exists()

    def test_index_never_references_removed_dirs(self, tmp_path: Path):
        """Regression: the pre-rewrite clear_cache could delete bag
        dirs while the index still listed them (spec §1)."""
        from deriva.bag.cache_index import BagCacheIndex
        from deriva_ml.core.storage import clear_cache

        cache_dir = tmp_path / "cache"
        old = datetime.now(timezone.utc) - timedelta(days=40)
        _record_bag(cache_dir, checksum="i1", dataset_rid="RID-Z", built_at=old)

        clear_cache(cache_dir, older_than_days=30)

        index = BagCacheIndex(cache_dir)
        try:
            for row in index.list_bags():
                assert index.bag_dir_for(row["checksum"]).exists(), (
                    f"index references removed bag {row['checksum']}"
                )
        finally:
            index.dispose()

    def test_missing_cache_dir_returns_zeros(self, tmp_path: Path):
        from deriva_ml.core.storage import clear_cache

        stats = clear_cache(tmp_path / "nope")
        assert stats == {"files_removed": 0, "dirs_removed": 0, "bytes_freed": 0, "errors": 0}
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_cache_introspection.py::TestClearCacheCoherent -v 2>&1 | tail -6
```
Expected: FAIL — `cannot import name 'clear_cache'`.

- [ ] **Step 3: Implement in `core/storage.py`**

Append:

```python
# Cache-dir entries owned by the bag index / asset cache machinery.
# clear_cache handles these through their own coherent paths; only
# entries outside this set get the legacy mtime-walk treatment.
_PROTECTED_CACHE_ENTRIES = frozenset(
    {"bags", "assets", "index.sqlite", "index.sqlite-wal", "index.sqlite-shm"}
)


def clear_cache(
    cache_dir: Path,
    older_than_days: int | None = None,
    log: Any | None = None,
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

        index = BagCacheIndex(cache_dir)
        try:
            for row in index.list_bags():
                if cutoff is not None:
                    built = datetime.fromisoformat(row["built_at"]).timestamp()
                    if built > cutoff:
                        continue
                freed = _dir_size(index.bag_dir_for(row["checksum"]))
                try:
                    index.purge(row["checksum"])
                except (OSError, PermissionError) as e:
                    log.warning("Failed to purge cached bag %s: %s", row["checksum"], e)
                    stats["errors"] += 1
                    continue
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
```

- [ ] **Step 4: Run to verify pass**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_cache_introspection.py -v 2>&1 | tail -8
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add src/deriva_ml/core/storage.py tests/core/test_cache_introspection.py && \
git commit -m "feat(storage): index-coherent clear_cache engine (fixes index/disk divergence)"
```

---

## Task 6: Wire the `DerivaML` surface

**Files:**
- Modify: `src/deriva_ml/core/base.py` — `clear_cache` body (lines ~1410–1476), new methods after `clean_execution_dirs`, `get_storage_summary` (~1631)
- Test: `tests/core/test_cache_introspection.py` + existing `tests/core/test_storage_management.py`

- [ ] **Step 1: Write the failing tests (harness-based, unbound methods)**

Append:

```python
# ---------------------------------------------------------------------------
# DerivaML surface (unbound methods against the harness)
# ---------------------------------------------------------------------------


class TestDerivaMLSurface:
    def test_list_cached_bags_delegates(self, harness):
        from deriva_ml.core.base import DerivaML

        _record_bag(harness.cache_dir, checksum="j1", dataset_rid="RID-S")
        bags = DerivaML.list_cached_bags(harness)
        assert [b.dataset_rid for b in bags] == ["RID-S"]

    def test_delete_cached_bag_delegates(self, harness):
        from deriva_ml.core.base import DerivaML

        _record_bag(harness.cache_dir, checksum="k1", dataset_rid="RID-T")
        stats = DerivaML.delete_cached_bag(harness, "RID-T")
        assert stats["bags_removed"] == 1
        assert DerivaML.list_cached_bags(harness) == []

    def test_list_and_delete_cached_assets_delegate(self, harness):
        from deriva_ml.core.base import DerivaML

        _make_cached_asset(harness.cache_dir, "RID-U", MD5_A)
        assert [a.rid for a in DerivaML.list_cached_assets(harness)] == ["RID-U"]
        stats = DerivaML.delete_cached_asset(harness, "RID-U")
        assert stats["assets_removed"] == 1

    def test_clear_cache_is_index_coherent_via_derivaml(self, harness):
        from deriva.bag.cache_index import BagCacheIndex
        from deriva_ml.core.base import DerivaML

        _record_bag(harness.cache_dir, checksum="l1", dataset_rid="RID-W")
        DerivaML.clear_cache(harness)
        index = BagCacheIndex(harness.cache_dir)
        try:
            assert index.list_bags() == []
        finally:
            index.dispose()

    def test_storage_summary_has_species_breakdown(self, harness):
        from deriva_ml.core.base import DerivaML

        # get_storage_summary calls these as self.<method>(); bind the
        # real implementations to the harness (existing pattern in
        # tests/core/test_storage_management.py).
        harness.get_cache_size = lambda: DerivaML.get_cache_size(harness)
        harness.list_execution_dirs = lambda: DerivaML.list_execution_dirs(harness)
        harness.list_cached_bags = lambda: DerivaML.list_cached_bags(harness)
        harness.list_cached_assets = lambda: DerivaML.list_cached_assets(harness)

        _record_bag(harness.cache_dir, checksum="m1", dataset_rid="RID-SUM")
        _make_cached_asset(harness.cache_dir, "RID-AS", MD5_A)

        summary = DerivaML.get_storage_summary(harness)

        # Existing keys unchanged
        for key in (
            "working_dir", "cache_dir", "cache_size_mb", "cache_file_count",
            "execution_dir_count", "execution_size_mb", "total_size_mb",
        ):
            assert key in summary
        # New per-species keys
        assert summary["bag_count"] == 1
        assert summary["asset_count"] == 1
        assert summary["bag_size_mb"] > 0
        assert summary["asset_size_mb"] > 0
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_cache_introspection.py::TestDerivaMLSurface -v 2>&1 | tail -6
```
Expected: FAIL — `DerivaML` has no attribute `list_cached_bags`.

- [ ] **Step 3: Implement in `base.py`**

**(a)** Replace the entire body of `clear_cache` (keep signature and
docstring's Args/Returns shape; lines ~1410–1476) with a delegation —
the method docstring gains a note about index coherence:

```python
    def clear_cache(self, older_than_days: int | None = None) -> dict[str, int]:
        """Clear the dataset cache directory.

        Removes cached dataset bags and assets. Bags are removed
        *through the bag-cache index* (index row and on-disk directory
        together), so the index never references a removed bag.

        Args:
            older_than_days: If provided, only remove cache entries older than this
                many days (bags age by their recorded ``built_at``; assets and
                stray entries by mtime). If None, removes all cache entries.

        Returns:
            dict: Statistics about the cleanup:
                - 'files_removed': Number of files removed
                - 'dirs_removed': Number of directories removed
                - 'bytes_freed': Total bytes freed
                - 'errors': Number of errors encountered

        Example:
            >>> ml = DerivaML('deriva.example.org', 'my_catalog')  # doctest: +SKIP
            >>> stats = ml.clear_cache(older_than_days=30)  # doctest: +SKIP
            >>> print(f"Freed {stats['bytes_freed'] / 1e6:.1f} MB")  # doctest: +SKIP
        """
        from deriva_ml.core.storage import clear_cache as _clear_cache

        return _clear_cache(self.cache_dir, older_than_days, self._logger)
```

(The old body's `import shutil` / `import time` lines go away with it.)

**(b)** Add the four new methods immediately after
`clean_execution_dirs` (before `get_storage_summary`):

```python
    def list_cached_bags(self) -> "list[CachedBag]":
        """List every dataset bag in the local cache.

        Answers "what bags are currently cached?" without needing to
        know any dataset RID up front. One record per
        (bag, dataset-anchor) pair, most-recently-built first.

        Returns:
            List of :class:`~deriva_ml.core.storage.CachedBag` records
            (empty when nothing is cached).

        Example:
            >>> ml = DerivaML('deriva.example.org', 'my_catalog')  # doctest: +SKIP
            >>> for bag in ml.list_cached_bags():  # doctest: +SKIP
            ...     print(bag.dataset_rid, bag.version, bag.status.value)  # doctest: +SKIP
        """
        from deriva_ml.dataset.bag_cache import BagCache

        with BagCache(self.cache_dir) as cache:
            return cache.list_bags()

    def delete_cached_bag(self, dataset_rid: str, version: str | None = None) -> dict[str, int]:
        """Delete a dataset's cached bag(s).

        Args:
            dataset_rid: Dataset RID whose cached bags to remove.
            version: When given, only the bag for that version;
                ``None`` removes every cached version.

        Returns:
            ``{"bags_removed": n, "bytes_freed": n}``; zeros when the
            dataset isn't cached (idempotent).

        Example:
            >>> ml = DerivaML('deriva.example.org', 'my_catalog')  # doctest: +SKIP
            >>> ml.delete_cached_bag('1-ABC', version='1.2.0')  # doctest: +SKIP
        """
        from deriva_ml.dataset.bag_cache import BagCache

        with BagCache(self.cache_dir) as cache:
            return cache.purge_dataset(dataset_rid, version=version)

    def list_cached_assets(self) -> "list[CachedAsset]":
        """List cached input assets (``assets/{rid}_{md5}`` entries).

        These are written by ``Execution.download_asset(use_cache=True)``
        / ``AssetSpec(cache=True)``.

        Returns:
            List of :class:`~deriva_ml.core.storage.CachedAsset`
            records (empty when no assets are cached).

        Example:
            >>> ml = DerivaML('deriva.example.org', 'my_catalog')  # doctest: +SKIP
            >>> for a in ml.list_cached_assets():  # doctest: +SKIP
            ...     print(a.rid, a.size_bytes)  # doctest: +SKIP
        """
        from deriva_ml.core.storage import list_cached_assets as _list

        return _list(self.cache_dir)

    def delete_cached_asset(self, rid: str, md5: str | None = None) -> dict[str, int]:
        """Delete cached copies of an input asset.

        Args:
            rid: Asset RID whose cache entries to remove.
            md5: When given, only the copy with that checksum;
                ``None`` removes every cached copy.

        Returns:
            ``{"assets_removed": n, "bytes_freed": n}``; zeros when
            nothing matched (idempotent).

        Example:
            >>> ml = DerivaML('deriva.example.org', 'my_catalog')  # doctest: +SKIP
            >>> ml.delete_cached_asset('2-XYZ')  # doctest: +SKIP
        """
        from deriva_ml.core.storage import delete_cached_asset as _delete

        return _delete(self.cache_dir, rid, md5=md5)
```

**(c)** Extend `get_storage_summary` — replace its `return` block (keep
everything above it) with:

```python
        bags = self.list_cached_bags()
        assets = self.list_cached_assets()
        # A multi-anchor bag appears once per dataset RID in the
        # listing; size it once per checksum for the summary.
        bag_bytes = sum(
            {b.checksum: (b.size_bytes or 0) for b in bags}.values()
        )
        asset_bytes = sum(a.size_bytes for a in assets)

        return {
            "working_dir": str(self.working_dir),
            "cache_dir": str(self.cache_dir),
            "cache_size_mb": cache_stats["total_mb"],
            "cache_file_count": cache_stats["file_count"],
            "execution_dir_count": len(exec_dirs),
            "execution_size_mb": exec_size_mb,
            "total_size_mb": cache_stats["total_mb"] + exec_size_mb,
            # Per-species breakdown (spec 2026-06-11)
            "bag_count": len(bags),
            "bag_size_mb": bag_bytes / (1024 * 1024),
            "asset_count": len(assets),
            "asset_size_mb": asset_bytes / (1024 * 1024),
        }
```

and update its docstring's Returns list with the four new keys
(`bag_count`, `bag_size_mb`, `asset_count`, `asset_size_mb`).

**(d)** Add the type-only import at the top of `base.py` inside the
existing `if TYPE_CHECKING:` block:

```python
    from deriva_ml.core.storage import CachedAsset, CachedBag
```

- [ ] **Step 4: Run new tests AND the existing storage suite (back-compat gate)**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_cache_introspection.py tests/core/test_storage_management.py -v 2>&1 | tail -12
```
Expected: ALL pass — including the four pre-existing `clear_cache`
tests in `test_storage_management.py` (they exercise stray-entry
behavior, which Pass 3 preserves). If any existing test fails, the
rewrite broke back-compat: STOP and fix before committing.

Note: `test_storage_management.py`'s `get_storage_summary` tests bind
only `get_cache_size`/`list_execution_dirs` onto its harness; the
extended summary now also calls `self.list_cached_bags()` /
`self.list_cached_assets()`. Update `_make_harness` in
`tests/core/test_storage_management.py` to also bind:

```python
    h.list_cached_bags = lambda: DerivaML.list_cached_bags(h)  # type: ignore[arg-type,method-assign]
    h.list_cached_assets = lambda: DerivaML.list_cached_assets(h)  # type: ignore[arg-type,method-assign]
```

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add src/deriva_ml/core/base.py tests/core/test_cache_introspection.py tests/core/test_storage_management.py && \
git commit -m "feat(storage): DerivaML cache-introspection surface + per-species storage summary"
```

---

## Task 7: Exports, lint, full unit pass

**Files:**
- Modify: `src/deriva_ml/__init__.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_cache_introspection.py`:

```python
class TestExports:
    def test_records_importable_from_top_level(self):
        from deriva_ml import CachedAsset, CachedBag  # noqa: F401
```

Run; expected FAIL (`ImportError`).

- [ ] **Step 2: Add lazy exports**

In `src/deriva_ml/__init__.py`, add to the `__getattr__` chain
(alongside the existing elif branches):

```python
    elif name == "CachedBag":
        from deriva_ml.core.storage import CachedBag

        return CachedBag
    elif name == "CachedAsset":
        from deriva_ml.core.storage import CachedAsset

        return CachedAsset
```

and to `__all__` (in the Definitions section, alphabetical):

```python
    "CachedAsset",
    "CachedBag",
```

- [ ] **Step 3: Lint + format + full unit-test pass (no catalog)**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
uv run ruff check src/deriva_ml/core/storage.py src/deriva_ml/dataset/bag_cache.py src/deriva_ml/core/base.py src/deriva_ml/__init__.py tests/core/test_cache_introspection.py && \
uv run ruff format src/deriva_ml/core/storage.py src/deriva_ml/dataset/bag_cache.py tests/core/test_cache_introspection.py && \
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ tests/core/ -q 2>&1 | tail -4
```
Expected: ruff clean; all unit tests pass (doctests in
`core/storage.py` run via `--doctest-modules` — the pure-Python
examples execute for real).

- [ ] **Step 4: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add -A src tests && \
git commit -m "feat(storage): export CachedBag/CachedAsset from package root"
```

---

## Task 8: Integration tests (live catalog)

**Files:**
- Modify: `tests/execution/test_storage.py` (append a class)

Requires `DERIVA_HOST` (defaults to localhost) with the Docker stack
up. Skip this task if no catalog is reachable — note it in the PR.

- [ ] **Step 1: Append the integration class**

```python
class TestCacheIntrospectionIntegration:
    """Live-catalog round-trip: download → list → delete → verify.

    Uses the session CatalogManager (same pattern as
    tests/dataset/test_dataset_caching.py).
    """

    def test_download_list_delete_roundtrip(self, catalog_manager, tmp_path):
        from deriva_ml.dataset.bag_cache import CacheStatus

        catalog_manager.reset()
        ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "src")
        dataset = dataset_desc.dataset
        version = dataset.current_version

        # Not cached yet → listing has no entry for this RID
        before = [b for b in ml.list_cached_bags() if b.dataset_rid == dataset.dataset_rid]
        assert before == []

        dataset.download_dataset_bag(version=version, use_minid=False)

        listed = [b for b in ml.list_cached_bags() if b.dataset_rid == dataset.dataset_rid]
        assert len(listed) == 1
        assert listed[0].version == str(version)
        assert listed[0].status in (CacheStatus.cached_materialized, CacheStatus.cached_holey)
        assert listed[0].path.exists()

        # Summary sees it
        summary = ml.get_storage_summary()
        assert summary["bag_count"] >= 1

        # Delete and verify both the listing and bag_info agree
        stats = ml.delete_cached_bag(dataset.dataset_rid)
        assert stats["bags_removed"] >= 1
        assert [b for b in ml.list_cached_bags() if b.dataset_rid == dataset.dataset_rid] == []
        info = dataset.bag_info(version=version)
        assert info["status"] == CacheStatus.not_cached.value
```

- [ ] **Step 2: Run it**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest \
  tests/execution/test_storage.py::TestCacheIntrospectionIntegration -v --timeout=600 2>&1 | tail -6
```
Expected: PASS (takes a few minutes — catalog populate + download).

- [ ] **Step 3: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add tests/execution/test_storage.py && \
git commit -m "test(storage): live-catalog round-trip for cache introspection"
```

---

## Task 9: User-guide docs + PR

**Files:**
- Modify: `docs/user-guide/offline.md`

- [ ] **Step 1: Add the docs section**

Read `docs/user-guide/offline.md` and append (or place near its
existing cache discussion) the section:

```markdown
## What's in my cache?

DerivaML keeps three kinds of local storage. Each has a direct
inspect call and a matching delete:

| What | Inspect | Delete |
|---|---|---|
| Cached dataset bags | `ml.list_cached_bags()` | `ml.delete_cached_bag(rid, version=None)` |
| Cached input assets | `ml.list_cached_assets()` | `ml.delete_cached_asset(rid, md5=None)` |
| Execution working dirs | `ml.list_execution_dirs()` | `ml.clean_execution_dirs(...)` |

```python
# Every cached bag, newest first
for bag in ml.list_cached_bags():
    print(bag.dataset_rid, bag.version, bag.status.value, bag.size_bytes)

# Cached input assets (written by AssetSpec(cache=True))
for asset in ml.list_cached_assets():
    print(asset.rid, asset.md5, asset.size_bytes)

# One summary across all three species
summary = ml.get_storage_summary()
print(summary["bag_count"], summary["asset_count"], summary["total_size_mb"])

# Targeted cleanup
ml.delete_cached_bag("1-ABC", version="1.2.0")  # one version
ml.delete_cached_bag("1-ABC")                   # all versions
ml.delete_cached_asset("2-XYZ")                 # all cached copies

# Whole-cache cleanup (bags removed through the index, so the
# cache index always agrees with the disk)
ml.clear_cache(older_than_days=30)
```

Deletion is idempotent — removing something that isn't cached
returns zero counts rather than raising. Everything here is local
and re-downloadable; deleting a cached bag never touches the
catalog.
```

(Adjust the RID strings to keep the doc's illustrative-RID style
consistent with the surrounding page; they are illustrative, not
asserted.)

- [ ] **Step 2: Commit and open the PR**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && \
git add docs/user-guide/offline.md && \
git commit -m "docs(user-guide): what's-in-my-cache section (introspection + cleanup)" && \
git push -u origin feat/cache-introspection && \
gh pr create --title "feat(storage): cache & storage introspection API" \
  --body "Adds direct inspect/delete APIs for all three local-storage species per docs/superpowers/specs/2026-06-11-cache-introspection-design.md:

- ml.list_cached_bags() / ml.delete_cached_bag(rid, version=None) — backed by new BagCache.list_bags()/purge_dataset() over the deriva-py BagCacheIndex
- ml.list_cached_assets() / ml.delete_cached_asset(rid, md5=None) — the assets/{rid}_{md5} cache
- get_storage_summary() extended with per-species keys (bag_count/bag_size_mb/asset_count/asset_size_mb), existing keys unchanged
- clear_cache() rewritten to delete bags *through the index* (purge), fixing the latent bug where the SQLite index could reference bag dirs the old blind rmtree walk had removed
- New Pydantic records CachedBag/CachedAsset exported from package root
- Unit tests (no catalog) + live-catalog round-trip integration test + user-guide section

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

- [ ] **Step 3: Post-merge note**

After the PR merges: `uv run bump-version patch` on a clean, synced
`main` (this is a feature, so `minor` may be more appropriate —
decide at merge time). Never bump from the feature branch.

---

## Self-review

- **Spec coverage:** §4.2 records → Task 1; §4.3 `list_cached_bags` →
  Tasks 2+6, `list_cached_assets` → Tasks 4+6, summary extension →
  Task 6c; §4.4 `delete_cached_bag` → Tasks 3+6, `delete_cached_asset`
  → Tasks 4+6; §4.5 clear_cache coherence → Task 5 + 6a (regression
  test in Task 5 Step 1); §4.6 error handling → idempotent-zero tests
  (Tasks 3–5) and per-entry try/except in implementations; §5 unit →
  Tasks 1–7, integration → Task 8, doctests → Example blocks
  throughout (pure-Python ones runnable); §6 docs → Task 9.
- **Type consistency:** `CachedBag`/`CachedAsset` field names used in
  Tasks 1–6 and the docs match; deletion return shapes
  (`bags_removed`/`assets_removed` + `bytes_freed`) consistent across
  Tasks 3, 4, 6 and the docs; `clear_cache` stats keys match the
  legacy shape everywhere.
- **Known judgment points for the implementer:**
  - `bag_cache.py` needs `import sqlite3` and
    `from datetime import datetime` added to module imports (Task 2).
  - If `bdbag.make_bag` output on this platform fails
    `validate_bag_structure` round-trip (it shouldn't), prefer fixing
    the fixture, not loosening the status assertions.
  - Existing `tests/core/test_storage_management.py` `clear_cache`
    tests are the back-compat gate (Task 6 Step 4) — do not edit their
    assertions; only `_make_harness` gains two bound delegates.
