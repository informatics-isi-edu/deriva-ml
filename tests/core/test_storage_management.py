"""Unit tests for `DerivaML` storage-management methods.

Closes audit §C.1 — `clear_cache`, `list_execution_dirs`,
`clean_execution_dirs`, `get_storage_summary` covered ~256 LoC of
destructive filesystem operations with zero direct test coverage.
A bug in `clean_execution_dirs(older_than_days=30)` could silently
delete data; the lack of coverage was a real gap.

All tests use ``tmp_path`` fixtures so no real cache or working
directory is touched. Methods are invoked unbound via
``DerivaML.method(harness, ...)`` against a lightweight harness
object carrying just the attributes each method reads
(``self.working_dir``, ``self.cache_dir``, ``self._logger``).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest


class _StorageHarness:
    """Stand-in for ``self`` in the storage-management methods.

    All four methods touch only ``self.working_dir``,
    ``self.cache_dir``, and ``self._logger``. Construct a minimal
    object exposing those three attributes; methods are called as
    ``DerivaML.method(harness, ...)``.

    The harness also implements ``get_cache_size`` and
    ``list_execution_dirs`` indirectly via the methods under test
    — ``get_storage_summary`` calls both.
    """

    def __init__(self, working_dir: Path, cache_dir: Path, logger=None):
        self.working_dir = working_dir
        self.cache_dir = cache_dir
        # ``self._logger`` is consulted on the error path of
        # ``clear_cache`` / ``clean_execution_dirs``. A no-op
        # logger keeps tests deterministic.
        import logging

        self._logger = logger or logging.getLogger("test")


def _make_harness(tmp_path: Path) -> _StorageHarness:
    """Build a harness with sibling working_dir / cache_dir.

    ``get_storage_summary`` calls ``self.get_cache_size()`` and
    ``self.list_execution_dirs()`` — bind those as unbound-method
    delegates so the harness behaves like a real ``DerivaML`` for
    the storage-management surface.
    """
    from deriva_ml.core.base import DerivaML

    working_dir = tmp_path / "wd"
    cache_dir = tmp_path / "cache"
    working_dir.mkdir()
    cache_dir.mkdir()
    h = _StorageHarness(working_dir=working_dir, cache_dir=cache_dir)
    # Delegate the two methods get_storage_summary calls back through
    # to the real implementations, bound to this harness instance.
    h.get_cache_size = lambda: DerivaML.get_cache_size(h)  # type: ignore[arg-type,method-assign]
    h.list_execution_dirs = lambda: DerivaML.list_execution_dirs(h)  # type: ignore[arg-type,method-assign]
    h.list_cached_bags = lambda: DerivaML.list_cached_bags(h)  # type: ignore[arg-type,method-assign]
    h.list_cached_assets = lambda: DerivaML.list_cached_assets(h)  # type: ignore[arg-type,method-assign]
    return h


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------


def test_clear_cache_empty_dir_returns_zero(tmp_path):
    """No cache contents → zero counts, no errors."""
    from deriva_ml.core.base import DerivaML

    h = _make_harness(tmp_path)
    result = DerivaML.clear_cache(h)  # type: ignore[arg-type]
    assert result == {"files_removed": 0, "dirs_removed": 0, "bytes_freed": 0, "errors": 0}


def test_clear_cache_missing_dir_returns_zero(tmp_path):
    """Cache dir doesn't exist → method short-circuits to zero counts."""
    from deriva_ml.core.base import DerivaML

    h = _StorageHarness(working_dir=tmp_path / "wd", cache_dir=tmp_path / "nope")
    result = DerivaML.clear_cache(h)  # type: ignore[arg-type]
    assert result == {"files_removed": 0, "dirs_removed": 0, "bytes_freed": 0, "errors": 0}


def test_clear_cache_removes_files_and_dirs(tmp_path):
    """Cache with mixed files + dirs → all removed; counts accurate."""
    from deriva_ml.core.base import DerivaML

    h = _make_harness(tmp_path)
    # Two files at top level + one subdirectory with nested files.
    (h.cache_dir / "a.txt").write_text("aaa")
    (h.cache_dir / "b.txt").write_text("bbbb")
    sub = h.cache_dir / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("cc")

    result = DerivaML.clear_cache(h)  # type: ignore[arg-type]
    assert result["files_removed"] == 2
    assert result["dirs_removed"] == 1
    assert result["bytes_freed"] == len("aaa") + len("bbbb") + len("cc")
    assert result["errors"] == 0
    # Cache dir is empty after the call.
    assert list(h.cache_dir.iterdir()) == []


def test_clear_cache_older_than_days_skips_recent(tmp_path):
    """``older_than_days=1`` skips entries modified within the last day."""
    from deriva_ml.core.base import DerivaML

    h = _make_harness(tmp_path)
    recent = h.cache_dir / "recent.txt"
    recent.write_text("new")
    old = h.cache_dir / "old.txt"
    old.write_text("old")
    # Backdate ``old`` to 2 days ago.
    two_days_ago = time.time() - 2 * 24 * 60 * 60
    os.utime(old, (two_days_ago, two_days_ago))

    result = DerivaML.clear_cache(h, older_than_days=1)  # type: ignore[arg-type]
    assert result["files_removed"] == 1
    assert recent.exists()
    assert not old.exists()


# ---------------------------------------------------------------------------
# list_execution_dirs
# ---------------------------------------------------------------------------


def _make_exec_dir(working_dir: Path, exec_rid: str, contents: dict[str, str]):
    """Materialise a deriva-ml exec dir under ``working_dir``.

    Layout: ``working_dir/deriva-ml/execution/{exec_rid}/<files>``.
    """
    base = working_dir / "deriva-ml" / "execution" / exec_rid
    base.mkdir(parents=True)
    for name, content in contents.items():
        (base / name).write_text(content)
    return base


def test_list_execution_dirs_empty(tmp_path):
    """No deriva-ml/execution dir → empty list."""
    from deriva_ml.core.base import DerivaML

    h = _make_harness(tmp_path)
    assert DerivaML.list_execution_dirs(h) == []  # type: ignore[arg-type]


def test_list_execution_dirs_reports_size_and_files(tmp_path):
    """Each exec dir reports its size, file count, and modification time."""
    from datetime import datetime

    from deriva_ml.core.base import DerivaML

    h = _make_harness(tmp_path)
    _make_exec_dir(h.working_dir, "EXE-A", {"a.txt": "hello"})
    _make_exec_dir(h.working_dir, "EXE-B", {"b1.txt": "x", "b2.txt": "yy"})

    rows = DerivaML.list_execution_dirs(h)  # type: ignore[arg-type]
    assert len(rows) == 2
    by_rid = {row["execution_rid"]: row for row in rows}

    assert by_rid["EXE-A"]["file_count"] == 1
    assert by_rid["EXE-A"]["size_bytes"] == len("hello")
    assert isinstance(by_rid["EXE-A"]["modified"], datetime)

    assert by_rid["EXE-B"]["file_count"] == 2
    assert by_rid["EXE-B"]["size_bytes"] == len("x") + len("yy")


def test_list_execution_dirs_sorted_newest_first(tmp_path):
    """Result is sorted by modification time, newest first."""
    from deriva_ml.core.base import DerivaML

    h = _make_harness(tmp_path)
    old = _make_exec_dir(h.working_dir, "OLD", {"x.txt": "a"})
    new = _make_exec_dir(h.working_dir, "NEW", {"x.txt": "b"})
    # Backdate `old` to 1 day ago.
    one_day_ago = time.time() - 24 * 60 * 60
    os.utime(old, (one_day_ago, one_day_ago))
    # Make sure `new` is genuinely "now".
    os.utime(new, (time.time(), time.time()))

    rows = DerivaML.list_execution_dirs(h)  # type: ignore[arg-type]
    assert [r["execution_rid"] for r in rows] == ["NEW", "OLD"]


# ---------------------------------------------------------------------------
# clean_execution_dirs
# ---------------------------------------------------------------------------


def test_clean_execution_dirs_removes_all_by_default(tmp_path):
    """Default behaviour: every exec dir is removed, bytes are tallied."""
    from deriva_ml.core.base import DerivaML

    h = _make_harness(tmp_path)
    _make_exec_dir(h.working_dir, "EXE-A", {"f": "abc"})
    _make_exec_dir(h.working_dir, "EXE-B", {"f": "wxyz"})

    result = DerivaML.clean_execution_dirs(h)  # type: ignore[arg-type]
    assert result["dirs_removed"] == 2
    assert result["bytes_freed"] == len("abc") + len("wxyz")
    assert result["errors"] == 0
    # No exec dirs left under deriva-ml/execution.
    exec_root = h.working_dir / "deriva-ml" / "execution"
    assert list(exec_root.iterdir()) == []


def test_clean_execution_dirs_exclude_rids_preserved(tmp_path):
    """``exclude_rids`` keeps the named directories untouched."""
    from deriva_ml.core.base import DerivaML

    h = _make_harness(tmp_path)
    _make_exec_dir(h.working_dir, "KEEP", {"f": "a"})
    _make_exec_dir(h.working_dir, "DROP", {"f": "b"})

    result = DerivaML.clean_execution_dirs(h, exclude_rids=["KEEP"])  # type: ignore[arg-type]
    assert result["dirs_removed"] == 1
    exec_root = h.working_dir / "deriva-ml" / "execution"
    remaining = {p.name for p in exec_root.iterdir()}
    assert remaining == {"KEEP"}


def test_clean_execution_dirs_older_than_days_skips_recent(tmp_path):
    """``older_than_days=1`` keeps recent dirs, removes old ones."""
    from deriva_ml.core.base import DerivaML

    h = _make_harness(tmp_path)
    old_dir = _make_exec_dir(h.working_dir, "OLD", {"f": "a"})
    new_dir = _make_exec_dir(h.working_dir, "NEW", {"f": "b"})
    two_days_ago = time.time() - 2 * 24 * 60 * 60
    os.utime(old_dir, (two_days_ago, two_days_ago))
    os.utime(new_dir, (time.time(), time.time()))

    result = DerivaML.clean_execution_dirs(h, older_than_days=1)  # type: ignore[arg-type]
    assert result["dirs_removed"] == 1
    exec_root = h.working_dir / "deriva-ml" / "execution"
    remaining = {p.name for p in exec_root.iterdir()}
    assert remaining == {"NEW"}


def test_clean_execution_dirs_missing_root_returns_zero(tmp_path):
    """No deriva-ml/execution dir → zero counts, no errors."""
    from deriva_ml.core.base import DerivaML

    h = _make_harness(tmp_path)
    result = DerivaML.clean_execution_dirs(h)  # type: ignore[arg-type]
    assert result == {"dirs_removed": 0, "bytes_freed": 0, "errors": 0}


# ---------------------------------------------------------------------------
# get_storage_summary
# ---------------------------------------------------------------------------


def test_get_storage_summary_empty(tmp_path):
    """Empty cache + empty exec dirs → all zeros."""
    from deriva_ml.core.base import DerivaML

    h = _make_harness(tmp_path)
    summary = DerivaML.get_storage_summary(h)  # type: ignore[arg-type]
    assert summary["cache_size_mb"] == 0.0
    assert summary["cache_file_count"] == 0
    assert summary["execution_dir_count"] == 0
    assert summary["execution_size_mb"] == 0
    assert summary["total_size_mb"] == 0


def test_get_storage_summary_tallies_cache_and_executions(tmp_path):
    """Mixed cache + exec contents are reflected in the summary."""
    from deriva_ml.core.base import DerivaML

    h = _make_harness(tmp_path)
    # 1 KB of cache contents.
    (h.cache_dir / "blob.dat").write_bytes(b"x" * 1024)
    # Two exec dirs, 2 KB total.
    _make_exec_dir(h.working_dir, "EXE-A", {"a.txt": "x" * 512})
    _make_exec_dir(h.working_dir, "EXE-B", {"b.txt": "x" * 1536})

    summary = DerivaML.get_storage_summary(h)  # type: ignore[arg-type]
    assert summary["cache_file_count"] == 1
    assert summary["cache_size_mb"] == pytest.approx(1024 / (1024 * 1024))
    assert summary["execution_dir_count"] == 2
    # Cache + executions sum to total.
    assert summary["total_size_mb"] == pytest.approx(summary["cache_size_mb"] + summary["execution_size_mb"])
    # working_dir and cache_dir are reported as strings.
    assert summary["working_dir"] == str(h.working_dir)
    assert summary["cache_dir"] == str(h.cache_dir)
