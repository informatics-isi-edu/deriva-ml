# Dirty-Tree Check Scoping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In `deriva-ml`, refine the dirty-tree check at `src/deriva_ml/execution/workflow.py:672-680` so that (a) the user's own output-capture files (under `findings/`, `outputs/`, `.scratch/`) and a configurable extra prefix list don't trigger a false positive, and (b) when the check does reject, the error message lists the actual offending paths instead of just naming the executable. Closes issue [#251](https://github.com/informatics-isi-edu/deriva-ml/issues/251).

**Architecture:**
- Replace the single `bool` flow `_github_url() → is_dirty: bool → raise DerivaMLDirtyWorkflowError(path)` with a richer flow `_github_url() → dirty_paths: list[str] → raise DerivaMLDirtyWorkflowError(executable_path, dirty_paths)`. The exception accepts the path list as a new optional argument and formats it into the message.
- Filter `git status --porcelain` output through a default-prefix exclusion (`findings/`, `outputs/`, `.scratch/`) plus an env-var-extensible list (`DERIVA_ML_DIRTY_CHECK_IGNORE`, colon-separated, like `PATH`). Paths matching any prefix get dropped from the dirty list. The default list is chosen because these are the documented "where to put output capture" patterns.
- The default exclusions are *semantic*: they correspond to directories the project conventions treat as scratch / output / findings space — not code the workflow could have read. The override env var is for projects with non-default conventions.

**Tech Stack:** Python 3.x (deriva-ml's existing version), pytest with an existing `git_repo` fixture in `tests/execution/test_dirty_workflow.py`. No new dependencies.

**Issue reference:** [#251](https://github.com/informatics-isi-edu/deriva-ml/issues/251).

---

## File Structure

| File | Purpose | Lifecycle |
|---|---|---|
| `src/deriva_ml/execution/workflow.py` | The dirty-check site at L672-680 (the `_github_url` helper). Returns a list of offending paths instead of a bool. The exclusion-filter logic lives here as a private helper `_filter_dirty_paths`. | Modified |
| `src/deriva_ml/execution/workflow.py` | The callsite at L502-510 (in `get_url_and_checksum`). Threads the new return shape into the exception. | Modified |
| `src/deriva_ml/core/exceptions.py` | `DerivaMLDirtyWorkflowError.__init__`. Accepts an optional `dirty_paths` list and renders it in the message. | Modified |
| `tests/execution/test_dirty_workflow.py` | Existing test file. Adds tests for the new behaviour: excluded directories don't trigger dirty, env var extends the exclusion list, exception message lists paths. | Modified |
| `tests/core/test_exceptions.py` | Existing test file. Adds a test for the new exception signature accepting a path list. | Modified |

No new files. All changes localised to two source files and two test files.

---

## Task 1: Update `DerivaMLDirtyWorkflowError` to accept dirty_paths

**Files:**
- Modify: `src/deriva_ml/core/exceptions.py`
- Test: `tests/core/test_exceptions.py`

The exception's `__init__` currently takes one arg (`path: str`). Extend it to optionally accept a list of offending paths and render them in the message. Keep the one-arg call signature working — existing callers that pass only `path` should continue to function.

- [ ] **Step 1: Read the current exception class**

In `src/deriva_ml/core/exceptions.py`, find `class DerivaMLDirtyWorkflowError`. The current `__init__` is:

```python
def __init__(self, path: str) -> None:
    super().__init__(
        f"File {path} has uncommitted changes. Commit before running, or use --allow-dirty to override."
    )
    self.path = path
```

- [ ] **Step 2: Write a failing test for the new signature**

In `tests/core/test_exceptions.py`, find the existing `TestDirtyWorkflowError`-style tests (in `tests/execution/test_dirty_workflow.py:50-62`; in `test_exceptions.py` there is one block exercising the same exception). Add this test to `tests/core/test_exceptions.py`:

```python
def test_dirty_workflow_error_with_paths():
    """When dirty_paths is provided, the message lists each path on its own line."""
    from deriva_ml.core.exceptions import DerivaMLDirtyWorkflowError

    exc = DerivaMLDirtyWorkflowError(
        "src/models/train.py",
        dirty_paths=["?? findings/run_output.txt", " M src/models/train.py"],
    )
    msg = str(exc)
    assert "src/models/train.py" in msg
    assert "?? findings/run_output.txt" in msg
    assert " M src/models/train.py" in msg
    assert exc.path == "src/models/train.py"
    assert exc.dirty_paths == ["?? findings/run_output.txt", " M src/models/train.py"]


def test_dirty_workflow_error_without_paths_is_backward_compatible():
    """The single-argument form still works (no dirty_paths passed)."""
    from deriva_ml.core.exceptions import DerivaMLDirtyWorkflowError

    exc = DerivaMLDirtyWorkflowError("src/models/train.py")
    msg = str(exc)
    assert "src/models/train.py" in msg
    assert "--allow-dirty" in msg
    assert exc.path == "src/models/train.py"
    assert exc.dirty_paths == []
```

- [ ] **Step 3: Run the tests, verify they fail**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run pytest tests/core/test_exceptions.py::test_dirty_workflow_error_with_paths tests/core/test_exceptions.py::test_dirty_workflow_error_without_paths_is_backward_compatible -v
```

Expected: both FAIL — `test_dirty_workflow_error_with_paths` fails because `__init__` doesn't accept `dirty_paths`; `test_dirty_workflow_error_without_paths_is_backward_compatible` fails because `exc.dirty_paths` doesn't exist.

- [ ] **Step 4: Implement the new signature**

Replace the existing `DerivaMLDirtyWorkflowError.__init__` in `src/deriva_ml/core/exceptions.py` with:

```python
def __init__(
    self,
    path: str,
    dirty_paths: list[str] | None = None,
) -> None:
    """Args:
        path: Path to the executable file whose workflow is being recorded.
        dirty_paths: Optional list of git-status-porcelain lines describing
            the actual offending paths (e.g. ["?? findings/out.txt",
            " M src/models/train.py"]). When provided, the lines are
            included in the exception message so the user can see what
            tripped the check rather than having to run ``git status``
            themselves.
    """
    self.path = path
    self.dirty_paths = list(dirty_paths) if dirty_paths else []
    if self.dirty_paths:
        offending = "\n".join(f"  {line}" for line in self.dirty_paths)
        message = (
            f"Workflow check rejected: {path} is in a worktree with "
            f"uncommitted changes that affect provenance.\n"
            f"Offending paths:\n{offending}\n"
            f"Either commit these files, or use --allow-dirty / "
            f"DERIVA_ML_ALLOW_DIRTY=true to proceed with a non-reproducible "
            f"recording. To exclude directories from this check, set "
            f"DERIVA_ML_DIRTY_CHECK_IGNORE to a colon-separated list of "
            f"prefixes (defaults already cover findings/, outputs/, .scratch/)."
        )
    else:
        # Backward-compatible single-arg form. Kept for any external
        # caller that constructs this exception without the path list.
        message = (
            f"File {path} has uncommitted changes. Commit before running, "
            f"or use --allow-dirty to override."
        )
    super().__init__(message)
```

Also update the class docstring `Args:` section to document `dirty_paths`. Find the existing `Args:` block (just `path: ...`) and replace with:

```python
    Args:
        path: Path to the file with uncommitted changes.
        dirty_paths: Optional list of git-status-porcelain lines describing
            which paths tripped the check. When provided, the message lists
            them; otherwise falls back to a single-path message.
```

- [ ] **Step 5: Run the tests, verify they pass**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run pytest tests/core/test_exceptions.py -v
```

Expected: all tests in `test_exceptions.py` pass (the two new ones plus any pre-existing).

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/core/exceptions.py tests/core/test_exceptions.py
git commit -m "$(cat <<'EOF'
feat(exceptions): DerivaMLDirtyWorkflowError accepts and renders dirty_paths

When the dirty-tree check rejects a workflow, the exception message
should name the actual offending paths instead of just the executable
being run. Adds an optional `dirty_paths` kwarg to the exception's
__init__ — when present, the porcelain lines are rendered into the
message; when absent, the existing single-path message is used (so
existing callers that don't yet supply paths keep working).

Part 1 of issue #251.
EOF
)"
```

---

## Task 2: Add `_filter_dirty_paths` helper in workflow.py

**Files:**
- Modify: `src/deriva_ml/execution/workflow.py`
- Test: `tests/execution/test_dirty_workflow.py`

A private helper that takes raw `git status --porcelain` output and returns the list of offending lines (whichever lines don't match a default + env-var exclusion prefix). Pure function, easy to unit-test.

- [ ] **Step 1: Write a failing test for the helper**

Append the following to `tests/execution/test_dirty_workflow.py`:

```python
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
            " M src/models/train.py\n"
            "?? findings/run_output.txt\n"
            "?? outputs/results.csv\n"
            " M notebooks/explore.ipynb\n"
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
```

- [ ] **Step 2: Run the tests, verify they fail**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run pytest tests/execution/test_dirty_workflow.py::TestFilterDirtyPaths -v
```

Expected: every test in the class FAILS at collection time with `AttributeError: type object 'Workflow' has no attribute '_filter_dirty_paths'` (or similar). Or all collect and fail individually.

- [ ] **Step 3: Implement the helper**

In `src/deriva_ml/execution/workflow.py`, near the top of the `Workflow` class (or as a module-level helper if module-level helpers already exist there), add the following. The natural placement is as a `@staticmethod` on the `Workflow` class alongside `_github_url` and `_get_git_root`. Add at the same indentation level (class-method level).

```python
    # Default paths excluded from the dirty-tree check. These are
    # directories the project conventions treat as scratch / output /
    # findings space — not code the workflow could have read, so changes
    # under them don't affect provenance.
    _DEFAULT_DIRTY_CHECK_EXCLUSIONS: tuple[str, ...] = (
        "findings/",
        "outputs/",
        ".scratch/",
    )

    @staticmethod
    def _filter_dirty_paths(porcelain_output: str) -> list[str]:
        """Parse ``git status --porcelain`` output, drop excluded paths.

        Lines whose target path sits under one of the default exclusion
        prefixes (``findings/``, ``outputs/``, ``.scratch/``) or under
        any prefix from the ``DERIVA_ML_DIRTY_CHECK_IGNORE`` env var
        (colon-separated, ``PATH``-like) are removed. The remaining lines
        — actual code/config changes — are returned in order.

        The helper handles git-status rename lines (``R  old -> new``)
        by checking the destination path; if the destination is under an
        excluded prefix the line is dropped.

        Args:
            porcelain_output: Raw stdout from ``git status --porcelain``.
                May be empty (clean tree).

        Returns:
            A list of porcelain lines (verbatim, including the two-letter
            status code) that survived the exclusion filter. Empty list
            means "tree is clean for provenance purposes."

        Example:
            >>> Workflow._filter_dirty_paths(" M src/models/train.py\\n?? findings/x.txt\\n")
            [' M src/models/train.py']
        """
        if not porcelain_output.strip():
            return []

        # Build the full exclusion list: built-in defaults + env-var extras.
        # Env-var format mirrors PATH (colon-separated). An empty string
        # value means "no extra exclusions" (NOT a single empty-prefix
        # that would match every path).
        extra = os.environ.get("DERIVA_ML_DIRTY_CHECK_IGNORE", "")
        extra_prefixes = tuple(p for p in extra.split(":") if p)
        exclusions = Workflow._DEFAULT_DIRTY_CHECK_EXCLUSIONS + extra_prefixes

        kept: list[str] = []
        for line in porcelain_output.splitlines():
            if not line.strip():
                continue
            # Porcelain format: two status chars, a space, then the path.
            # Renames are written as ``R  old -> new``; we want to match
            # against the destination.
            path_part = line[3:]
            if " -> " in path_part:
                path_part = path_part.split(" -> ", 1)[1]
            if any(path_part.startswith(prefix) for prefix in exclusions):
                continue
            kept.append(line)
        return kept
```

Also confirm `import os` is already present at the top of `workflow.py`. It is — line 13. If not, add it.

- [ ] **Step 4: Run the tests, verify they pass**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run pytest tests/execution/test_dirty_workflow.py::TestFilterDirtyPaths -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/execution/workflow.py tests/execution/test_dirty_workflow.py
git commit -m "$(cat <<'EOF'
feat(workflow): add _filter_dirty_paths helper with default exclusions

A private helper that parses git status --porcelain output and drops
lines whose paths sit under one of:

  - findings/  (project convention for run outputs and evidence)
  - outputs/   (project convention for analysis artifacts)
  - .scratch/  (project convention for temporary work)
  - any prefix from DERIVA_ML_DIRTY_CHECK_IGNORE (PATH-like colon-list)

These directories are scratch/output space, not code the workflow could
have read, so changes under them don't affect provenance and shouldn't
trip the dirty-tree check.

Rename lines (``R  old -> new``) are matched against the destination
path so a rename INTO findings/ is excluded.

Helper is a pure function; the integration into _github_url comes in
the next commit. Part 2 of issue #251.
EOF
)"
```

---

## Task 3: Wire `_filter_dirty_paths` into `_github_url` and `get_url_and_checksum`

**Files:**
- Modify: `src/deriva_ml/execution/workflow.py:672-700` (the `_github_url` dirty-check) and `:502-510` (the callsite that raises).
- Test: `tests/execution/test_dirty_workflow.py`

The current `_github_url` returns `(url, is_dirty: bool)`. We need richer information at the callsite to populate the exception. Two choices: (a) change the return shape to `(url, dirty_paths: list[str])` (breaking change to a private method, but cleaner), or (b) keep the bool and add a separate accessor. Choice: (a) — `_github_url` is a private static method only called from one place.

- [ ] **Step 1: Write a failing integration test**

Append to `tests/execution/test_dirty_workflow.py` (after `TestFilterDirtyPaths`):

```python
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
```

- [ ] **Step 2: Run the tests, verify they fail**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run pytest tests/execution/test_dirty_workflow.py::TestDirtyCheckIntegration -v
```

Expected: at least the `findings_untracked` and `outputs_untracked` tests FAIL (current code treats them as dirty). The `src_modification_still_triggers_dirty` and `exception_message_lists_offending_paths` tests may pass partially (the existing exception has the executable path in it, not the dirty paths) — but the `.dirty_paths` attribute assertions will fail.

Capture the failure pattern in your task report.

- [ ] **Step 3: Replace the dirty-check block in `_github_url`**

In `src/deriva_ml/execution/workflow.py`, locate the block at L654-700 (from the comment `# Check whether the working tree is clean.` through the `return url, is_dirty` statement). Replace ONLY the dirty-check portion (L654-682) with the version that returns the path list. The existing comment block explaining the design is preserved with an updated note.

Replace:

```python
        # Check whether the working tree is clean. Any non-empty
        # ``git status --porcelain`` output means *something* about
        # the repo differs from HEAD -- staged, unstaged, untracked,
        # renamed, deleted, or merge-conflicted. Provenance demands
        # we treat all of these as dirty: a workflow's recorded
        # commit hash only reproduces if every file the workflow
        # could read from the repo matches that commit.
        #
        # The previous ``"M " in result.stdout.strip()`` heuristic
        # only matched the two-letter code for "staged modified",
        # silently missing unstaged edits (`` M``), untracked files
        # (``??``), renames (``R ``), deletes (``D ``), conflicts
        # (``UU``), etc. -- exactly the cases ``DERIVA_ML_ALLOW_DIRTY``
        # is supposed to gate honesty about.
        #
        # ``cwd`` is the worktree's root; ``git status --porcelain``
        # reports the whole worktree regardless of cwd, but anchoring
        # at ``repo_root`` makes the call site explicit.
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            is_dirty = bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            is_dirty = False  # If the Git command fails, assume no changes
```

with:

```python
        # Check whether the working tree is clean. Any non-empty
        # ``git status --porcelain`` output means *something* about
        # the repo differs from HEAD -- staged, unstaged, untracked,
        # renamed, deleted, or merge-conflicted. Provenance demands
        # we treat all of these as dirty: a workflow's recorded
        # commit hash only reproduces if every file the workflow
        # could read from the repo matches that commit.
        #
        # The previous ``"M " in result.stdout.strip()`` heuristic
        # only matched the two-letter code for "staged modified",
        # silently missing unstaged edits (`` M``), untracked files
        # (``??``), renames (``R ``), deletes (``D ``), conflicts
        # (``UU``), etc. -- exactly the cases ``DERIVA_ML_ALLOW_DIRTY``
        # is supposed to gate honesty about.
        #
        # ``_filter_dirty_paths`` then drops lines whose path sits
        # under one of the project-convention scratch/output prefixes
        # (``findings/``, ``outputs/``, ``.scratch/``) or any prefix
        # from ``DERIVA_ML_DIRTY_CHECK_IGNORE``. Those directories
        # aren't on the workflow's read path, so changes under them
        # don't compromise reproducibility.
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            dirty_paths = Workflow._filter_dirty_paths(result.stdout)
        except subprocess.CalledProcessError:
            dirty_paths = []  # If the Git command fails, assume no changes
```

Then update the function's return statement at the end. Find:

```python
        return url, is_dirty
```

…and change it to:

```python
        return url, dirty_paths
```

Also update the function's return-type annotation at line 452. Find:

```python
    def get_url_and_checksum(executable_path: Path, allow_dirty: bool = False) -> tuple[str, str]:
```

That's the public method, not `_github_url`. The private `_github_url` is at a separate signature. Find:

```python
    @staticmethod
    def _github_url(executable_path: Path) -> tuple[str, bool]:
```

Change to:

```python
    @staticmethod
    def _github_url(executable_path: Path) -> tuple[str, list[str]]:
```

Update the docstring's Returns line from `bool to indicate if uncommited changes` to `list of porcelain lines describing uncommitted changes that affect provenance (empty if clean)`.

- [ ] **Step 4: Update the callsite in `get_url_and_checksum`**

In `src/deriva_ml/execution/workflow.py` around L502-510, find:

```python
        github_url, is_dirty = Workflow._github_url(executable_path)

        if is_dirty:
            if allow_dirty:
                logger.warning(
                    f"File {executable_path} has uncommitted changes. Proceeding with --allow-dirty override."
                )
            else:
                raise DerivaMLDirtyWorkflowError(str(executable_path))
```

Replace with:

```python
        github_url, dirty_paths = Workflow._github_url(executable_path)

        if dirty_paths:
            if allow_dirty:
                offending = ", ".join(p.strip() for p in dirty_paths[:3])
                more = f" (and {len(dirty_paths) - 3} more)" if len(dirty_paths) > 3 else ""
                logger.warning(
                    f"Worktree has uncommitted changes affecting provenance: "
                    f"{offending}{more}. Proceeding with --allow-dirty override."
                )
            else:
                raise DerivaMLDirtyWorkflowError(
                    str(executable_path), dirty_paths=dirty_paths
                )
```

- [ ] **Step 5: Run the new tests, verify they pass**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run pytest tests/execution/test_dirty_workflow.py -v
```

Expected: ALL tests in this file pass — both the pre-existing ones (clean / dirty / allow_dirty) and the new `TestFilterDirtyPaths` and `TestDirtyCheckIntegration` classes.

- [ ] **Step 6: Run the full deriva-ml test suite to confirm nothing else broke**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run pytest 2>&1 | tail -15
```

Expected: PASS. If any other test broke, STOP and triage — `_github_url`'s return type change might have affected somewhere else.

- [ ] **Step 7: Lint**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run ruff check src/deriva_ml/execution/workflow.py src/deriva_ml/core/exceptions.py tests/execution/test_dirty_workflow.py tests/core/test_exceptions.py
uv run ruff format --check src/deriva_ml/execution/workflow.py src/deriva_ml/core/exceptions.py tests/execution/test_dirty_workflow.py tests/core/test_exceptions.py
```

If ruff reports issues, run `uv run ruff format` on those files and re-run check. Expected: clean.

- [ ] **Step 8: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/execution/workflow.py tests/execution/test_dirty_workflow.py
git commit -m "$(cat <<'EOF'
feat(workflow): scope the dirty-tree check to exclude scratch directories

The dirty-tree check in _github_url now filters git status --porcelain
output through _filter_dirty_paths, which drops paths under findings/,
outputs/, .scratch/, or any prefix from DERIVA_ML_DIRTY_CHECK_IGNORE.

Concretely: a `tee findings/run_output.txt` command in the user's
pipeline no longer trips the check on its own output file (the
documented bug in issue #251). The check still rejects any change
to code (src/, notebooks/, pyproject.toml, etc.).

The exception now lists the actual offending paths so the user can
see what tripped the check without running `git status` themselves.
The previous error message named the executable being run, which
sent readers hunting in the wrong place.

The --allow-dirty / DERIVA_ML_ALLOW_DIRTY=true override continues to
work and now also names the offending paths in its warning log.

Closes #251.
EOF
)"
```

---

## Task 4: Add a release-note / changelog entry

**Files:**
- Modify: `CHANGELOG.md` (or equivalent — verify path before editing)

- [ ] **Step 1: Verify changelog location**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
ls CHANGELOG* 2>&1
```

If no changelog file exists, skip this task. If one exists (likely `CHANGELOG.md`), proceed to Step 2.

- [ ] **Step 2: Add an entry at the top under the next-version (Unreleased) section**

The exact wording depends on the existing changelog conventions. Pattern to follow:

```markdown
### Added
- Dirty-tree check now excludes `findings/`, `outputs/`, and `.scratch/` by default; additional prefixes can be configured via `DERIVA_ML_DIRTY_CHECK_IGNORE` (colon-separated, `PATH`-like). The previous behaviour rejected output-capture files (e.g. from `2>&1 | tee findings/out.txt`); now those are correctly recognised as scratch space, not code. (#251)

### Changed
- `DerivaMLDirtyWorkflowError` now lists the actual offending paths in its message instead of just naming the executable being run. The previous message sent readers hunting in `.venv/.../workflow.py` when the dirty file was a user-created output capture. (#251)
```

Read the existing `CHANGELOG.md` format and adapt accordingly. If the project uses a different format (Keep-A-Changelog, conventional commits, etc.), match it.

- [ ] **Step 3: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add CHANGELOG.md
git commit -m "docs(changelog): note dirty-check scoping + clearer error (#251)"
```

If there's no CHANGELOG file, omit this task entirely — don't fabricate one.

---

## Task 5: Update the issue and open a PR

**Files:**
- None modified locally.

- [ ] **Step 1: Branch, push, open PR**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git checkout -b fix/251-dirty-check-scoping
# (Reset main to origin/main if you've been committing there. Otherwise, just branch.)
git push -u origin fix/251-dirty-check-scoping
```

If you've been committing on `main`, reset it first:

```bash
# Save the branch's tip first if you switched branches with uncommitted work.
git branch -f main origin/main  # if main is currently ahead of origin
```

- [ ] **Step 2: Create the PR**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
gh pr create --title "fix(workflow): scope dirty-tree check + clearer error message (#251)" --body "$(cat <<'EOF'
## Summary

Two folded changes addressing issue #251:

1. **Default-exclude scratch directories from the dirty check.** Paths under \`findings/\`, \`outputs/\`, and \`.scratch/\` no longer trip the dirty-tree check. The list can be extended via \`DERIVA_ML_DIRTY_CHECK_IGNORE\` (colon-separated, \`PATH\`-like).

2. **\`DerivaMLDirtyWorkflowError\` now lists the offending paths.** Previously the message named the executable being run, which sent readers hunting in \`.venv/.../workflow.py\` when the actual dirty file was a user-created output capture. Now the message reads, e.g.:

   \`\`\`
   Workflow check rejected: src/models/train.py is in a worktree with uncommitted changes that affect provenance.
   Offending paths:
     ?? findings/run_output.txt
   Either commit these files, or use --allow-dirty / DERIVA_ML_ALLOW_DIRTY=true to proceed with a non-reproducible recording. ...
   \`\`\`

## Why

The seed-sweep arc in deriva-ml-model-template hit this concretely: the documented \`2>&1 | tee findings/run_output.txt\` pattern fails on the very first run because the \`tee\` command creates the output file before \`_github_url\`'s dirty check, and the check sees the untracked file as a worktree mutation. See #251 for the full reproducer.

The intent of the check (provenance honesty) is preserved — any change to code under \`src/\`, \`notebooks/\`, \`pyproject.toml\`, etc. still rejects. The change only excludes paths the project conventions treat as scratch / output / findings space, plus user-extensible prefixes.

## Test plan

- [x] \`uv run pytest tests/core/test_exceptions.py -v\` — 2 new tests for the exception's two-arg signature.
- [x] \`uv run pytest tests/execution/test_dirty_workflow.py -v\` — 9 new tests for \`_filter_dirty_paths\` + 5 new integration tests for the end-to-end behaviour.
- [x] \`uv run pytest\` — full suite passes.
- [x] \`uv run ruff check\` and \`ruff format --check\` clean on changed files.

## Closes #251

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Confirm PR opened**

```bash
gh pr view --json url,title,state 2>&1 | tail -10
```

Expected: PR URL printed, state OPEN.

---

## Success criteria

- ✅ `DerivaMLDirtyWorkflowError.__init__` accepts an optional `dirty_paths` list. Backward-compatible with single-arg form.
- ✅ `Workflow._filter_dirty_paths` is a pure static method that drops excluded-prefix lines from porcelain output.
- ✅ Default exclusions: `findings/`, `outputs/`, `.scratch/`. Extra via `DERIVA_ML_DIRTY_CHECK_IGNORE` env var.
- ✅ `_github_url` returns `tuple[str, list[str]]` instead of `tuple[str, bool]`; callsite in `get_url_and_checksum` threads the list into the exception.
- ✅ Tests: 2 new in `test_exceptions.py`, 14 new in `test_dirty_workflow.py`, all pass.
- ✅ Existing tests in both files still pass.
- ✅ Lint clean on all touched files.
- ✅ Changelog entry (if a CHANGELOG.md exists).
- ✅ PR opened referencing #251.

## Out of scope (carried forward)

- **Scoping options 2 and 3 from issue #251** (only-check-code-paths, import-graph reachability). Option 1 (prefix exclusion) plus the error message improvement get the user-experience win immediately and leave the bigger semantic choices for later.
- **Other friction items from the seed-sweep arc** (bag-cache reuse across multirun children, multirun parent description not auto-composed). Tracked separately.
