"""Regression tests for issue #178 — upload staging layout.

Per-execution upload staging used to live at
``<working_dir>/commit-bag-{rid}/`` — right at the cache root,
beside ``cache/`` and ``schema-cache.json``. A user grepping for
cache state to free disk space would encounter the directory and
either delete it mid-upload or wonder why it existed (the name
"commit-bag" is opaque to anyone not familiar with the bag-commit
internals).

Issue #178 moved per-execution scratch under a dedicated
``upload/{rid}/`` parent. The cache root now contains only caches.

These tests pin the new path shape so a regression to the old
layout fails loudly. They avoid the need for a live catalog by
mocking the bag-commit collaborators that follow the path-build
step.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from deriva_ml.execution.execution import Execution


def _fresh_rid() -> str:
    """Return a unique synthetic RID string for path-shape tests.

    The workspace policy bans hardcoded RIDs in tests — they make
    bugs cluster on the same string and hide layout regressions
    that only show up for RIDs with particular characters. A UUID
    suffix gives every call a distinct, throwaway RID.

    Returns:
        A string of the form ``"TEST-<hex>"`` suitable as a
        synthetic ``execution_rid``.

    Example:
        >>> rid = _fresh_rid()
        >>> rid.startswith("TEST-")
        True
        >>> len(rid) > len("TEST-")
        True
    """
    return f"TEST-{uuid.uuid4().hex[:12].upper()}"


def _make_mocked_execution(working_dir: Path, execution_rid: str) -> Execution:
    """Build a minimal :class:`Execution` for path-shape assertions.

    Bypasses the full ``__init__`` (which expects a live
    ``DerivaML`` instance, a configured workflow, etc.) and sets
    only the attributes that :meth:`Execution._bag_commit_upload`
    touches before it hands off to the bag-commit collaborators.
    The collaborators themselves are mocked at the call site so
    the test never reaches a catalog.

    Args:
        working_dir: Filesystem location to use as the
            ``self._working_dir`` cache root.
        execution_rid: Synthetic RID for ``self.execution_rid``.

    Returns:
        A partly-initialised :class:`Execution` ready for the
        ``_bag_commit_upload`` path-shape test below.
    """
    exe = object.__new__(Execution)
    exe._working_dir = working_dir
    exe._logger = logging.getLogger(f"test.execution.{execution_rid}")

    # ``_bag_commit_upload`` reads pending features and manifest
    # rows after the path is built. Stub the manifest store on
    # ``_ml_object.workspace`` so the ``_manifest_store`` /
    # ``_get_manifest`` properties resolve without a real
    # workspace; load_execution_bag is mocked separately at the
    # call site to skip the catalog work.
    manifest_stub = MagicMock()
    manifest_stub.pending_assets.return_value = {}
    manifest_stub.mark_uploaded_batch = MagicMock()

    manifest_store_stub = MagicMock()
    manifest_store_stub.list_pending_feature_records.return_value = []

    ml_object_stub = MagicMock()
    ml_object_stub.workspace.manifest_store.return_value = manifest_store_stub

    exe._ml_object = ml_object_stub
    # ``_get_manifest`` lazily builds an AssetManifest with the
    # workspace store. Short-circuit the lazy path so the test
    # never instantiates the real class.
    exe._manifest = manifest_stub
    exe._set_asset_descriptions = MagicMock()

    # ``execution_rid`` is set as an instance attribute during
    # __init__ (see Execution.__init__ — assigned post-insert or
    # to DRY_RUN_RID for dry runs). We bypassed __init__, so set
    # it directly.
    exe.execution_rid = execution_rid
    return exe


def test_bag_commit_upload_uses_upload_subdir(tmp_path: Path) -> None:
    """Bag dir lives at ``working_dir/upload/{rid}/`` — not at the cache root.

    Captures the ``bag_dir`` argument passed to
    :func:`deriva_ml.execution.bag_commit.build_execution_bag` and
    asserts it has the new layout. A regression to
    ``working_dir/commit-bag-{rid}/`` makes this fail.
    """
    rid = _fresh_rid()
    exe = _make_mocked_execution(tmp_path, rid)

    captured: dict[str, Path] = {}

    def _fake_build(_execution, bag_dir: Path, *, progress_callback=None):
        captured["bag_dir"] = Path(bag_dir)
        return bag_dir

    fake_report = SimpleNamespace(total_rows_inserted=0, table_stats={})

    with (
        patch(
            "deriva_ml.execution.bag_commit.build_execution_bag",
            side_effect=_fake_build,
        ),
        patch(
            "deriva_ml.execution.bag_commit.load_execution_bag",
            return_value=fake_report,
        ),
        patch(
            "deriva_ml.execution.bag_commit.report_to_asset_map",
            return_value={},
        ),
    ):
        exe._bag_commit_upload()

    assert "bag_dir" in captured, "build_execution_bag was never called"
    bag_dir = captured["bag_dir"]
    expected = tmp_path / "upload" / rid
    assert bag_dir == expected, (
        f"bag_dir was {bag_dir!r}; expected {expected!r}. "
        "Per issue #178, per-execution staging lives at "
        "working_dir/upload/{rid}/, not at the cache root."
    )


def test_bag_commit_upload_does_not_pollute_cache_root(tmp_path: Path) -> None:
    """The cache root must not gain a ``commit-bag-*`` or ``upload-*`` sibling.

    After ``_bag_commit_upload`` runs, the immediate children of
    ``working_dir`` (the cache root) should NOT include any
    ``commit-bag-`` directories or ``upload-{rid}`` style entries.
    The only new entry should be the ``upload/`` parent.
    """
    rid = _fresh_rid()
    exe = _make_mocked_execution(tmp_path, rid)

    # Touch the bag_dir from inside the fake build to simulate
    # the real BagBuilder, which creates output_dir on entry.
    def _fake_build(_execution, bag_dir: Path, *, progress_callback=None):
        Path(bag_dir).mkdir(parents=True, exist_ok=True)
        return bag_dir

    fake_report = SimpleNamespace(total_rows_inserted=0, table_stats={})

    with (
        patch(
            "deriva_ml.execution.bag_commit.build_execution_bag",
            side_effect=_fake_build,
        ),
        patch(
            "deriva_ml.execution.bag_commit.load_execution_bag",
            return_value=fake_report,
        ),
        patch(
            "deriva_ml.execution.bag_commit.report_to_asset_map",
            return_value={},
        ),
    ):
        exe._bag_commit_upload()

    cache_root_children = {p.name for p in tmp_path.iterdir()}

    bad = [name for name in cache_root_children if name.startswith("commit-bag-") or name.startswith(f"upload-{rid}")]
    assert not bad, (
        f"Cache root gained scratch siblings {bad!r}; "
        "per-execution staging must live under upload/, not at "
        "the cache root."
    )
    assert "upload" in cache_root_children, (
        f"Expected an ``upload`` parent under the cache root after a bag commit; saw {sorted(cache_root_children)!r}."
    )

    # And the RID-specific subdir must be under upload/.
    assert (tmp_path / "upload" / rid).is_dir(), (
        f"Expected {tmp_path / 'upload' / rid} to exist after the fake build_execution_bag created it."
    )


def test_bag_commit_upload_wipes_stale_attempt(tmp_path: Path) -> None:
    """A pre-existing bag dir at the new path is wiped before rebuild.

    The path-build step calls ``shutil.rmtree(bag_dir)`` if a
    prior attempt left a partial bag in place. Verify that
    behaviour still works at the new ``upload/{rid}/`` location.
    """
    rid = _fresh_rid()
    exe = _make_mocked_execution(tmp_path, rid)

    stale = tmp_path / "upload" / rid
    stale.mkdir(parents=True)
    stale_marker = stale / "STALE_MARKER"
    stale_marker.write_text("stale")
    assert stale_marker.exists()

    def _fake_build(_execution, bag_dir: Path, *, progress_callback=None):
        # After the rmtree, the dir should be gone. Re-create empty.
        assert not Path(bag_dir).exists(), "Stale bag dir was not removed before build_execution_bag was called."
        Path(bag_dir).mkdir(parents=True)
        return bag_dir

    fake_report = SimpleNamespace(total_rows_inserted=0, table_stats={})

    with (
        patch(
            "deriva_ml.execution.bag_commit.build_execution_bag",
            side_effect=_fake_build,
        ),
        patch(
            "deriva_ml.execution.bag_commit.load_execution_bag",
            return_value=fake_report,
        ),
        patch(
            "deriva_ml.execution.bag_commit.report_to_asset_map",
            return_value={},
        ),
    ):
        exe._bag_commit_upload()

    assert not stale_marker.exists(), "Stale marker survived; the prior-attempt wipe did not run."


@pytest.mark.parametrize("_run", range(3))
def test_bag_commit_upload_path_is_dynamic(_run: int, tmp_path: Path) -> None:
    """Path shape holds for many distinct RIDs.

    Parametrised to make sure the path-build step is not
    accidentally hardcoded to a single RID format. ``_run`` is
    unused — each invocation just gives ``_fresh_rid`` a chance
    to mint a new RID.
    """
    rid = _fresh_rid()
    exe = _make_mocked_execution(tmp_path, rid)

    captured: dict[str, Path] = {}

    def _fake_build(_execution, bag_dir: Path, *, progress_callback=None):
        captured["bag_dir"] = Path(bag_dir)
        return bag_dir

    fake_report = SimpleNamespace(total_rows_inserted=0, table_stats={})

    with (
        patch(
            "deriva_ml.execution.bag_commit.build_execution_bag",
            side_effect=_fake_build,
        ),
        patch(
            "deriva_ml.execution.bag_commit.load_execution_bag",
            return_value=fake_report,
        ),
        patch(
            "deriva_ml.execution.bag_commit.report_to_asset_map",
            return_value={},
        ),
    ):
        exe._bag_commit_upload()

    assert captured["bag_dir"] == tmp_path / "upload" / rid
