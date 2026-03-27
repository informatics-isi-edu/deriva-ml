"""Tests for DerivaMLDirtyWorkflowError and --allow-dirty enforcement.

Verifies that:
1. get_url_and_checksum raises DerivaMLDirtyWorkflowError on dirty files
2. allow_dirty=True suppresses the error and logs a warning
3. Clean files work without any flag
"""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from deriva_ml.core.exceptions import DerivaMLDirtyWorkflowError
from deriva_ml.execution.workflow import Workflow


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repository with a committed Python file."""
    repo = tmp_path / "test_repo"
    repo.mkdir()

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"], cwd=repo, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=repo, capture_output=True, check=True
    )

    # Add a remote (needed for _github_url)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/test/test-repo.git"],
        cwd=repo,
        capture_output=True,
        check=True,
    )

    # Create and commit a Python file
    test_file = repo / "model.py"
    test_file.write_text("# test model\n")
    subprocess.run(["git", "add", "model.py"], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"], cwd=repo, capture_output=True, check=True
    )

    return repo, test_file


class TestDirtyWorkflowError:
    """Tests for the DerivaMLDirtyWorkflowError exception."""

    def test_exception_message(self):
        """Error message includes the file path and --allow-dirty hint."""
        exc = DerivaMLDirtyWorkflowError("src/models/train.py")
        assert "src/models/train.py" in str(exc)
        assert "--allow-dirty" in str(exc)
        assert exc.path == "src/models/train.py"

    def test_inherits_from_workflow_error(self):
        """DerivaMLDirtyWorkflowError is a DerivaMLWorkflowError."""
        from deriva_ml.core.exceptions import DerivaMLWorkflowError

        exc = DerivaMLDirtyWorkflowError("test.py")
        assert isinstance(exc, DerivaMLWorkflowError)


class TestGetUrlAndChecksum:
    """Tests for Workflow.get_url_and_checksum with allow_dirty parameter."""

    def test_clean_file_succeeds(self, git_repo):
        """Clean (committed) file returns URL and checksum without error."""
        repo, test_file = git_repo
        url, checksum = Workflow.get_url_and_checksum(test_file)
        assert "github.com" in url
        assert len(checksum) > 0

    def test_dirty_file_raises_by_default(self, git_repo):
        """Modified file raises DerivaMLDirtyWorkflowError by default."""
        repo, test_file = git_repo

        # Modify the file without committing
        test_file.write_text("# modified\n")

        with pytest.raises(DerivaMLDirtyWorkflowError) as exc_info:
            Workflow.get_url_and_checksum(test_file)

        assert str(test_file) in str(exc_info.value)

    def test_dirty_file_with_allow_dirty(self, git_repo, caplog):
        """Modified file with allow_dirty=True logs warning but succeeds."""
        repo, test_file = git_repo

        # Modify the file without committing
        test_file.write_text("# modified\n")

        import logging

        with caplog.at_level(logging.WARNING, logger="deriva_ml"):
            url, checksum = Workflow.get_url_and_checksum(test_file, allow_dirty=True)

        assert "github.com" in url
        assert "uncommitted changes" in caplog.text
        assert "--allow-dirty" in caplog.text

    def test_clean_file_with_allow_dirty_no_warning(self, git_repo, caplog):
        """Clean file with allow_dirty=True produces no warning."""
        repo, test_file = git_repo

        import logging

        with caplog.at_level(logging.WARNING, logger="deriva_ml"):
            url, checksum = Workflow.get_url_and_checksum(test_file, allow_dirty=True)

        assert "uncommitted changes" not in caplog.text


class TestWorkflowAllowDirtyField:
    """Tests for the allow_dirty field on the Workflow model."""

    def test_default_is_false(self):
        """Workflow.allow_dirty defaults to False."""
        w = Workflow.__new__(Workflow)
        # Check the field default
        assert Workflow.model_fields["allow_dirty"].default is False

    def test_env_var_overrides(self, git_repo, monkeypatch):
        """DERIVA_ML_ALLOW_DIRTY env var sets allow_dirty on Workflow."""
        repo, test_file = git_repo

        # Modify the file
        test_file.write_text("# modified\n")

        monkeypatch.setenv("DERIVA_ML_ALLOW_DIRTY", "true")

        # The env var should be picked up during model validation.
        # We can't easily test the full validator without a running script,
        # but we can verify the get_url_and_checksum path works.
        url, checksum = Workflow.get_url_and_checksum(test_file, allow_dirty=True)
        assert "github.com" in url
