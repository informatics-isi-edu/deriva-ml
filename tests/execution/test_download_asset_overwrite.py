"""Tests for the ``download_asset`` overwrite-warning guard (issue #181).

PR #179 RID-keyed the platform-default per-asset ``dest_dir`` so the
in-platform download path is collision-free by construction. Callers
building their own download flows on top of ``Execution.download_asset``,
however, can still pass the same ``dest_dir`` to two downloads whose
``Filename``s collide and lose data silently.

Issue #181 closes that hole with a *warn-then-overwrite* guard. The
guard is implemented as the module-level helper
:func:`deriva_ml.execution.execution._check_overwrite_safe`, called
once on the regular download branch and once on the cache-hit symlink
branch. This file pins the guard's behaviour at two layers:

1. Direct unit tests against the helper — exhaustive coverage of the
   md5-match / md5-mismatch / md5-unknown / no-existing-file branches
   without touching the catalog. Fast, deterministic, and these are
   what really prove the contract.

2. Integration tests that exercise the guard through
   ``Execution.download_asset`` against the live test catalog. These
   prove the helper is wired in at both write sites (regular download
   at line ~1346 and cache-hit symlink at line ~1338 in the
   post-fix file).

Reference: ``deriva-ml/src/deriva_ml/execution/execution.py``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from deriva_ml import ExecAssetType, MLAsset
from deriva_ml.execution.execution import (
    Execution,
    ExecutionConfiguration,
    _check_overwrite_safe,
)

# =============================================================================
# Helpers
# =============================================================================


def _write(path: Path, data: bytes) -> str:
    """Write ``data`` to ``path`` and return its hex md5.

    Args:
        path: Destination file. Parent directory must already exist.
        data: Raw bytes to write.

    Returns:
        The hex-digest md5 of ``data``.

    Example:
        >>> import tempfile, pathlib
        >>> with tempfile.TemporaryDirectory() as d:
        ...     md5 = _write(pathlib.Path(d) / "f", b"hello")
        ...     md5
        '5d41402abc4b2a76b9719d911017c592'
    """
    path.write_bytes(data)
    return hashlib.md5(data).hexdigest()


def _warning_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Return the WARNING records from ``deriva_ml`` captured in ``caplog``.

    Args:
        caplog: pytest's log-capture fixture.

    Returns:
        The subset of records at WARNING level emitted by the ``deriva_ml``
        logger hierarchy.

    Example:
        >>> # Filter caplog.records for the records this guard cares about
        >>> # without picking up unrelated warnings from other libraries.
        ...
    """
    return [r for r in caplog.records if r.levelno == logging.WARNING and r.name.startswith("deriva_ml")]


# =============================================================================
# Unit tests against _check_overwrite_safe (no live catalog)
# =============================================================================


class TestCheckOverwriteSafeUnit:
    """Direct unit tests against the helper.

    These cover every branch of the helper without exercising any
    catalog code. The integration tests below prove the wiring; these
    prove the contract.
    """

    def test_no_warning_when_destination_missing(self, tmp_path, caplog):
        """No existing file → no warning."""
        target = tmp_path / "missing.bin"
        assert not target.exists()

        with caplog.at_level(logging.WARNING, logger="deriva_ml"):
            _check_overwrite_safe(target, expected_md5="abc123", asset_rid="RID-1")

        assert _warning_records(caplog) == []

    def test_no_warning_on_idempotent_overwrite(self, tmp_path, caplog):
        """Existing bytes' md5 matches expected_md5 → silent overwrite."""
        target = tmp_path / "idempotent.bin"
        md5 = _write(target, b"identical bytes")

        with caplog.at_level(logging.WARNING, logger="deriva_ml"):
            _check_overwrite_safe(target, expected_md5=md5, asset_rid="RID-1")

        assert _warning_records(caplog) == []

    def test_warning_on_different_bytes(self, tmp_path, caplog):
        """Existing bytes' md5 differs from expected_md5 → WARNING."""
        target = tmp_path / "collide.bin"
        existing_md5 = _write(target, b"first asset's bytes")
        expected_md5 = hashlib.md5(b"second asset's bytes").hexdigest()
        assert existing_md5 != expected_md5

        with caplog.at_level(logging.WARNING, logger="deriva_ml"):
            _check_overwrite_safe(target, expected_md5=expected_md5, asset_rid="RID-COLLIDE")

        warnings = _warning_records(caplog)
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "RID-COLLIDE" in msg
        assert str(target) in msg
        assert existing_md5 in msg
        assert expected_md5 in msg

    def test_warning_when_expected_md5_unknown(self, tmp_path, caplog):
        """No catalog MD5 → conservative WARNING (cannot prove idempotent)."""
        target = tmp_path / "unknown_md5.bin"
        _write(target, b"some bytes")

        with caplog.at_level(logging.WARNING, logger="deriva_ml"):
            _check_overwrite_safe(target, expected_md5=None, asset_rid="RID-UNKNOWN")

        warnings = _warning_records(caplog)
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "RID-UNKNOWN" in msg
        assert "<unknown>" in msg

    def test_warning_on_symlink_pointing_to_different_bytes(self, tmp_path, caplog):
        """Cache-hit branch: existing symlink to a non-matching cache entry → WARNING.

        ``download_asset`` symlinks into a per-cache-key directory. When a
        caller re-uses the same ``dest_dir`` for a second asset that
        happens to share the same ``Filename``, the existing symlink
        points at the *first* asset's cached bytes — overwriting it
        silently swaps which asset the path resolves to. The helper
        catches that case by hashing through the symlink (it calls
        ``Path.resolve``).
        """
        cache_first = tmp_path / "cache_first.bin"
        existing_md5 = _write(cache_first, b"asset one bytes")
        link = tmp_path / "link.bin"
        link.symlink_to(cache_first)

        expected_md5 = hashlib.md5(b"asset two bytes").hexdigest()

        with caplog.at_level(logging.WARNING, logger="deriva_ml"):
            _check_overwrite_safe(link, expected_md5=expected_md5, asset_rid="RID-CACHE-COLLIDE")

        warnings = _warning_records(caplog)
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "RID-CACHE-COLLIDE" in msg
        assert existing_md5 in msg

    def test_no_warning_on_symlink_pointing_to_matching_bytes(self, tmp_path, caplog):
        """Cache-hit branch: existing symlink to the SAME cache entry → silent.

        Re-resolving an asset whose cached file is already the
        destination's symlink target is fully idempotent — the next
        ``unlink`` + ``symlink_to`` lands on identical bytes.
        """
        cached = tmp_path / "cached.bin"
        md5 = _write(cached, b"identical cached bytes")
        link = tmp_path / "link.bin"
        link.symlink_to(cached)

        with caplog.at_level(logging.WARNING, logger="deriva_ml"):
            _check_overwrite_safe(link, expected_md5=md5, asset_rid="RID-CACHE-OK")

        assert _warning_records(caplog) == []


# =============================================================================
# Integration tests through Execution.download_asset (live catalog)
# =============================================================================


@pytest.fixture
def test_workflow(workflow_terms):
    """Create a test workflow for the download tests."""
    ml = workflow_terms
    return ml.create_workflow(
        name="Issue 181 Test Workflow",
        workflow_type="Test Workflow",
        description="Workflow for issue #181 overwrite-warning tests",
    )


@pytest.fixture
def basic_execution(workflow_terms, test_workflow):
    """Create a basic execution for download tests."""
    ml = workflow_terms
    config = ExecutionConfiguration(
        description="Issue 181 Test Execution",
        workflow=test_workflow,
    )
    return ml.create_execution(config)


def _create_test_asset(execution: Execution, filename: str, content: str) -> None:
    """Mirror of the suite's ``create_test_asset`` helper, kept local for clarity."""
    asset_path = execution.asset_file_path(
        MLAsset.execution_asset,
        f"TestAsset/{filename}",
        asset_types=ExecAssetType.model_file,
    )
    with asset_path.open("w") as fp:
        fp.write(content)


class TestDownloadAssetOverwriteIntegration:
    """End-to-end tests that prove the guard is wired into ``download_asset``."""

    def test_warns_when_two_assets_collide_in_dest_dir(self, workflow_terms, test_workflow, caplog):
        """Downloading two distinct assets with the same Filename into one dest_dir warns."""
        ml = workflow_terms

        # Upload two assets with the same Filename across two separate
        # executions — distinct RIDs, distinct MD5s, but the catalog's
        # ``Filename`` column resolves to ``collide.txt`` for both.
        # Two executions (rather than one with two files in distinct
        # subdirs) sidesteps any local-folder dedup that happens
        # before upload.
        config_a = ExecutionConfiguration(description="Issue 181 Asset A", workflow=test_workflow)
        exec_a = ml.create_execution(config_a)
        with exec_a.execute() as execution:
            _create_test_asset(execution, "collide.txt", "first asset")
        uploaded_a_report = exec_a.commit_output_assets()
        uploaded_a = exec_a.uploaded_assets
        rid_a = uploaded_a["deriva-ml/Execution_Asset"][0].asset_rid

        config_b = ExecutionConfiguration(description="Issue 181 Asset B", workflow=test_workflow)
        exec_b = ml.create_execution(config_b)
        with exec_b.execute() as execution:
            _create_test_asset(execution, "collide.txt", "second asset bytes are different")
        uploaded_b_report = exec_b.commit_output_assets()
        uploaded_b = exec_b.uploaded_assets
        rid_b = uploaded_b["deriva-ml/Execution_Asset"][0].asset_rid

        assert rid_a != rid_b

        # Now download both into the same dest_dir. The second call must
        # warn before clobbering the first one's bytes.
        download_config = ExecutionConfiguration(
            description="Issue 181 Collide Download",
            workflow=test_workflow,
        )
        download_execution = ml.create_execution(download_config)

        with TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir)
            download_execution.download_asset(rid_a, dest, update_catalog=False)

            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="deriva_ml"):
                download_execution.download_asset(rid_b, dest, update_catalog=False)

            warnings = _warning_records(caplog)
            collide_warnings = [w for w in warnings if "download_asset" in w.getMessage()]
            assert len(collide_warnings) == 1, (
                f"expected exactly one download_asset overwrite WARNING; got {[w.getMessage() for w in warnings]}"
            )
            msg = collide_warnings[0].getMessage()
            assert rid_b in msg
            assert str(dest / "collide.txt") in msg

            # Asset B's bytes should win on disk.
            with (dest / "collide.txt").open() as f:
                assert f.read() == "second asset bytes are different"

    def test_does_not_warn_on_idempotent_redownload(self, basic_execution, caplog):
        """Downloading the same asset twice into the same dest_dir is silent."""
        ml = basic_execution._ml_object

        with basic_execution.execute() as execution:
            _create_test_asset(execution, "idempotent.txt", "stable bytes")

        uploaded_report = basic_execution.commit_output_assets()
        uploaded = basic_execution.uploaded_assets
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        download_config = ExecutionConfiguration(
            description="Issue 181 Idempotent Download",
            workflow=basic_execution.configuration.workflow,
        )
        download_execution = ml.create_execution(download_config)

        with TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir)
            download_execution.download_asset(asset_rid, dest, update_catalog=False)

            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="deriva_ml"):
                download_execution.download_asset(asset_rid, dest, update_catalog=False)

            warnings = _warning_records(caplog)
            collide_warnings = [w for w in warnings if "download_asset" in w.getMessage()]
            assert collide_warnings == [], (
                f"idempotent re-download should not warn; got {[w.getMessage() for w in collide_warnings]}"
            )
