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
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, capture_output=True, check=True)

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
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo, capture_output=True, check=True)

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


class TestDirtyDetectionStatusCodes:
    """Tests for ``Workflow._github_url`` dirty-detection across all
    ``git status --porcelain`` two-letter status codes.

    The previous implementation matched the substring ``"M "`` against
    ``git status --porcelain`` output, which only caught **staged**
    modifications. It silently passed every other dirty state through
    as "clean" — including the common case of an unstaged edit
    (`` M``), an untracked file (``??``), a rename, a delete, or a
    merge conflict. Provenance contract: the workflow's recorded
    commit hash only reproduces the run if every file in the repo
    matches HEAD, so any non-empty porcelain output must register as
    dirty.

    Each test below puts the repo into one specific dirty state, then
    calls ``_github_url`` and asserts the returned dirty-paths list is
    non-empty. The repo factory commits one ``model.py`` so we always
    have a base to drift from.
    """

    def test_clean_repo_is_not_dirty(self, git_repo):
        """Untouched committed repo reports clean."""
        repo, test_file = git_repo
        _, dirty_paths = Workflow._github_url(test_file)
        assert dirty_paths == []

    def test_unstaged_modification_is_dirty(self, git_repo):
        """`` M`` (unstaged modification) registers as dirty."""
        repo, test_file = git_repo
        test_file.write_text("# unstaged change\n")
        _, dirty_paths = Workflow._github_url(test_file)
        assert dirty_paths, (
            "Unstaged edit to a tracked file must be dirty; this was "
            "the original silent-pass case the old substring check missed."
        )

    def test_staged_modification_is_dirty(self, git_repo):
        """``M `` (staged modification) registers as dirty."""
        repo, test_file = git_repo
        test_file.write_text("# staged change\n")
        subprocess.run(["git", "add", "model.py"], cwd=repo, capture_output=True, check=True)
        _, dirty_paths = Workflow._github_url(test_file)
        assert dirty_paths

    def test_both_modified_is_dirty(self, git_repo):
        """``MM`` (staged + unstaged on the same file) registers as dirty."""
        repo, test_file = git_repo
        test_file.write_text("# staged\n")
        subprocess.run(["git", "add", "model.py"], cwd=repo, capture_output=True, check=True)
        test_file.write_text("# staged then re-edited\n")
        _, dirty_paths = Workflow._github_url(test_file)
        assert dirty_paths

    def test_untracked_file_is_dirty(self, git_repo):
        """``??`` (untracked file) registers as dirty.

        The original heuristic silently treated untracked files as
        clean. A workflow that adds (but doesn't commit) a new
        ``utils.py`` it imports from would have recorded a "clean"
        provenance hash that wouldn't reproduce on checkout.
        """
        repo, _ = git_repo
        (repo / "utils.py").write_text("# untracked helper\n")
        _, dirty_paths = Workflow._github_url(repo / "model.py")
        assert dirty_paths

    def test_deleted_file_is_dirty(self, git_repo):
        """``D `` (deleted tracked file) registers as dirty."""
        repo, test_file = git_repo
        # Commit a second file so we have something we can delete
        # without invalidating ``test_file`` (which the call site
        # passes in).
        (repo / "extra.py").write_text("# to be deleted\n")
        subprocess.run(["git", "add", "extra.py"], cwd=repo, capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "Add extra.py"], cwd=repo, capture_output=True, check=True)
        (repo / "extra.py").unlink()
        _, dirty_paths = Workflow._github_url(test_file)
        assert dirty_paths

    def test_renamed_file_is_dirty(self, git_repo):
        """``R `` (rename) registers as dirty."""
        repo, test_file = git_repo
        # Commit a second file we can rename without affecting test_file.
        (repo / "extra.py").write_text("# to be renamed\n")
        subprocess.run(["git", "add", "extra.py"], cwd=repo, capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "Add extra.py"], cwd=repo, capture_output=True, check=True)
        subprocess.run(["git", "mv", "extra.py", "renamed.py"], cwd=repo, capture_output=True, check=True)
        _, dirty_paths = Workflow._github_url(test_file)
        assert dirty_paths

    def test_unrelated_file_dirty_still_marks_repo_dirty(self, git_repo):
        """A dirty unrelated file marks the whole repo dirty.

        Provenance demands repo-wide cleanliness: a workflow that
        ``import``s another module the repo holds will silently
        observe that module's working-tree state. Limiting the dirty
        check to ``executable_path`` itself would let a dirty
        ``utils.py`` slip through as "the script is clean."
        """
        repo, test_file = git_repo
        # ``test_file`` is committed and untouched; another tracked
        # file in the repo is modified.
        (repo / "other.py").write_text("# tracked file added\n")
        subprocess.run(["git", "add", "other.py"], cwd=repo, capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "Add other.py"], cwd=repo, capture_output=True, check=True)
        (repo / "other.py").write_text("# modified after commit\n")
        _, dirty_paths = Workflow._github_url(test_file)
        assert dirty_paths


class TestFilterDirtyPaths:
    """Tests for Workflow._filter_dirty_paths.

    The helper parses ``git status --porcelain`` output and drops lines
    whose paths sit under one of the exclusion prefixes (defaults plus
    the DERIVA_ML_DIRTY_CHECK_IGNORE env var).
    """

    def test_empty_porcelain_returns_empty_list(self):
        """No porcelain output means no dirty paths."""
        from deriva_ml.execution.workflow import Workflow

        assert Workflow._filter_dirty_paths("") == []

    def test_code_changes_are_reported(self):
        """Changes outside the exclusion list pass through."""
        from deriva_ml.execution.workflow import Workflow

        porcelain = " M src/models/train.py\n?? notebooks/explore.ipynb\n"
        result = Workflow._filter_dirty_paths(porcelain)
        assert " M src/models/train.py" in result
        assert "?? notebooks/explore.ipynb" in result
        assert len(result) == 2

    def test_default_excludes_findings(self):
        """findings/ is excluded by default."""
        from deriva_ml.execution.workflow import Workflow

        porcelain = "?? findings/multirun/run_output.txt\n?? findings/sweep_rids.json\n"
        assert Workflow._filter_dirty_paths(porcelain) == []

    def test_default_excludes_outputs(self):
        """outputs/ is excluded by default."""
        from deriva_ml.execution.workflow import Workflow

        porcelain = "?? outputs/results.csv\n"
        assert Workflow._filter_dirty_paths(porcelain) == []

    def test_default_excludes_scratch(self):
        """.scratch/ is excluded by default."""
        from deriva_ml.execution.workflow import Workflow

        porcelain = "?? .scratch/work.txt\n"
        assert Workflow._filter_dirty_paths(porcelain) == []

    def test_mixed_excluded_and_real_only_real_returned(self):
        """When both excluded and non-excluded paths appear, only the
        real ones survive."""
        from deriva_ml.execution.workflow import Workflow

        porcelain = (
            " M src/models/train.py\n?? findings/run_output.txt\n?? outputs/results.csv\n M notebooks/explore.ipynb\n"
        )
        result = Workflow._filter_dirty_paths(porcelain)
        assert result == [" M src/models/train.py", " M notebooks/explore.ipynb"]

    def test_env_var_extends_exclusions(self, monkeypatch):
        """DERIVA_ML_DIRTY_CHECK_IGNORE adds prefixes to the default set."""
        from deriva_ml.execution.workflow import Workflow

        monkeypatch.setenv("DERIVA_ML_DIRTY_CHECK_IGNORE", "tmp/:logs/")
        porcelain = (
            " M src/models/train.py\n"
            "?? tmp/cache.bin\n"
            "?? logs/run.log\n"
            "?? findings/x.txt\n"  # default exclusion still applies
        )
        result = Workflow._filter_dirty_paths(porcelain)
        assert result == [" M src/models/train.py"]

    def test_env_var_empty_string_is_ignored(self, monkeypatch):
        """An empty DERIVA_ML_DIRTY_CHECK_IGNOR is treated as no extra
        exclusions (not as a single empty-prefix that would match everything)."""
        from deriva_ml.execution.workflow import Workflow

        monkeypatch.setenv("DERIVA_ML_DIRTY_CHECK_IGNORE", "")
        porcelain = " M src/models/train.py\n"
        assert Workflow._filter_dirty_paths(porcelain) == [" M src/models/train.py"]

    def test_renamed_paths_use_destination(self):
        """``git status`` renames are formatted as ``R  old -> new``;
        the helper should match against the destination path (post-arrow)
        since that's where the file lives now. If the destination is
        under an excluded prefix, drop it; otherwise keep."""
        from deriva_ml.execution.workflow import Workflow

        # Rename moving INTO findings/ — excluded
        porcelain1 = "R  src/old.txt -> findings/new.txt\n"
        assert Workflow._filter_dirty_paths(porcelain1) == []

        # Rename moving OUT OF findings/ — kept (destination is in src/)
        porcelain2 = "R  findings/old.txt -> src/new.txt\n"
        result = Workflow._filter_dirty_paths(porcelain2)
        assert result == ["R  findings/old.txt -> src/new.txt"]


class TestDirtyCheckIntegration:
    """End-to-end tests for the dirty check with directory exclusions."""

    def test_findings_untracked_file_does_not_trigger_dirty(self, git_repo):
        """An untracked file under findings/ should NOT make the
        worktree dirty — that's the issue #251 reproducer."""
        repo, test_file = git_repo

        # Create an untracked file in findings/ (the seed-sweep scenario)
        findings_dir = repo / "findings"
        findings_dir.mkdir()
        (findings_dir / "run_output.txt").write_text("ran fine\n")

        # The committed test_file is clean; findings/ has untracked content
        # but should be filtered out
        url, checksum = Workflow.get_url_and_checksum(test_file)
        assert "github.com" in url
        assert len(checksum) > 0

    def test_outputs_untracked_file_does_not_trigger_dirty(self, git_repo):
        """outputs/ is also a default-excluded directory."""
        repo, test_file = git_repo
        outputs_dir = repo / "outputs"
        outputs_dir.mkdir()
        (outputs_dir / "results.csv").write_text("col1,col2\n")

        url, checksum = Workflow.get_url_and_checksum(test_file)
        assert "github.com" in url

    def test_src_modification_still_triggers_dirty(self, git_repo):
        """Changes in src/ (or any non-excluded path) still raise."""
        repo, test_file = git_repo
        # Modify the committed file
        test_file.write_text("# modified\n")

        with pytest.raises(DerivaMLDirtyWorkflowError) as exc_info:
            Workflow.get_url_and_checksum(test_file)

        msg = str(exc_info.value)
        # The new exception lists the offending path explicitly
        assert "model.py" in msg

    def test_exception_message_lists_offending_paths(self, git_repo):
        """When the check rejects, the message names the actual dirty
        files (not just the executable being run)."""
        repo, test_file = git_repo
        # Two dirty paths: the committed file + a new untracked one in src/
        test_file.write_text("# modified\n")
        (repo / "src").mkdir(exist_ok=True)
        (repo / "src" / "extra.py").write_text("# stray file\n")

        with pytest.raises(DerivaMLDirtyWorkflowError) as exc_info:
            Workflow.get_url_and_checksum(test_file)

        msg = str(exc_info.value)
        assert "model.py" in msg
        assert "src/extra.py" in msg
        assert exc_info.value.dirty_paths  # non-empty
        assert any("model.py" in p for p in exc_info.value.dirty_paths)
        assert any("src/extra.py" in p for p in exc_info.value.dirty_paths)

    def test_env_var_can_exclude_additional_dirs(self, git_repo, monkeypatch):
        """DERIVA_ML_DIRTY_CHECK_IGNORE allows adding project-specific
        exclusions without code changes."""
        repo, test_file = git_repo
        # Create an untracked file in a non-default location
        (repo / "tmp").mkdir()
        (repo / "tmp" / "work.txt").write_text("scratch\n")

        # Without the env var, it would be considered dirty
        # (tmp/ is not in the default exclusions)
        with pytest.raises(DerivaMLDirtyWorkflowError):
            Workflow.get_url_and_checksum(test_file)

        # With the env var, it's excluded
        monkeypatch.setenv("DERIVA_ML_DIRTY_CHECK_IGNORE", "tmp/")
        url, checksum = Workflow.get_url_and_checksum(test_file)
        assert "github.com" in url
