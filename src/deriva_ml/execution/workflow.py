"""Workflow model and script-URL resolution for DerivaML executions.

Defines the ``Workflow`` Pydantic model, which represents a versioned
computational workflow in a Deriva catalog. Key responsibilities:

- Stores workflow metadata (URL, type, description, checksum, RID).
- Resolves the calling script's source URL automatically (Git remote, Jupyter
  kernel path, or local file path) when ``url`` is not provided explicitly.
- Supports catalog write-back for ``description`` and ``workflow_type`` when
  the workflow is bound to a live catalog instance.
- Deduplicates workflows by checksum on insert (the private
  ``DerivaML._add_workflow()`` dedup path; create workflows via the
  public ``DerivaML.create_workflow()``).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, PrivateAttr, field_validator, model_validator

from deriva_ml.core.definitions import RID, MLVocab, VocabularyTerm
from deriva_ml.core.exceptions import DerivaMLDirtyWorkflowError, DerivaMLException
from deriva_ml.core.logging_config import get_logger
from deriva_ml.core.validation import VALIDATION_CONFIG
from deriva_ml.execution.find_caller import _get_calling_module

logger = get_logger(__name__)

if TYPE_CHECKING:
    from deriva_ml.interfaces import DerivaMLCatalog


__all__ = ["Workflow"]


class Workflow(BaseModel):
    """Represents a computational workflow in DerivaML.

    A workflow defines a computational process or analysis pipeline. Each workflow has
    a unique identifier, source code location, and type. Workflows are typically
    associated with Git repositories for version control.

    When a Workflow is retrieved via ``lookup_workflow(rid)`` or ``lookup_workflow_by_url()``,
    it is bound to a catalog and its ``description`` and ``workflow_type`` properties become
    writable. Setting these properties will update the catalog record. If the catalog is
    read-only (a snapshot), attempting to set them will raise a ``DerivaMLException``.

    Attributes:
        name (str): Human-readable name of the workflow.
        url (str): URI to the workflow source code (typically a GitHub URL).
        workflow_type (str | list[str]): Type(s) of workflow (must be controlled vocabulary terms).
            Accepts a single string or a list of strings. Internally normalized to a list.
            When the workflow is bound to a writable catalog, setting this property
            will update the catalog record. The new values must be valid terms from
            the Workflow_Type vocabulary.
        version (str | None): Version identifier (semantic versioning).
        description (str | None): Description of workflow purpose and behavior.
            When the workflow is bound to a writable catalog, setting this property
            will update the catalog record.
        workflow_rid (RID | None): Resource Identifier if registered in catalog.
        checksum (str | None): Git hash of workflow source code.
        is_notebook (bool): Whether workflow is a Jupyter notebook.
        git_root (Path | None): Filesystem root of the git checkout the source
            was resolved from, when known. Used to scope dynamic-version
            resolution; ``None`` when no git checkout was detected.
        allow_dirty (bool): When True, uncommitted changes in the source's git
            worktree are downgraded from a hard error to a warning (set by the
            ``--allow-dirty`` CLI flag, dry-run mode, or
            ``DERIVA_ML_ALLOW_DIRTY``). Defaults to False, which keeps the
            clean-checkout precondition that makes provenance reproducible.

    Note:
        The recommended way to create a Workflow is via :meth:`DerivaML.create_workflow()
        <deriva_ml.DerivaML.create_workflow>`, which validates the workflow type against
        the catalog vocabulary::

            >>> workflow = ml.create_workflow(  # doctest: +SKIP
            ...     name="RNA Analysis",
            ...     workflow_type="python_notebook",
            ...     description="RNA sequence analysis"
            ... )

    Example:
        Create a workflow directly (without catalog validation)::

            >>> workflow = Workflow(  # doctest: +SKIP
            ...     name="RNA Analysis",
            ...     url="https://github.com/org/repo/analysis.ipynb",
            ...     workflow_type="python_notebook",
            ...     version="1.0.0",
            ...     description="RNA sequence analysis"
            ... )

        Look up an existing workflow by RID and update its properties::

            >>> workflow = ml.lookup_workflow("2-ABC1")  # doctest: +SKIP
            >>> workflow.description = "Updated description for RNA analysis"  # doctest: +SKIP
            >>> workflow.workflow_type = "python_script"  # doctest: +SKIP
            >>> print(workflow.description)  # doctest: +SKIP
            Updated description for RNA analysis

        Look up by URL and update::

            >>> url = "https://github.com/org/repo/blob/abc123/analysis.py"  # doctest: +SKIP
            >>> workflow = ml.lookup_workflow_by_url(url)  # doctest: +SKIP
            >>> workflow.description = "New description"  # doctest: +SKIP

        Attempting to update on a read-only catalog raises an error::

            >>> snapshot_ml = ml.catalog_snapshot("2023-01-15T10:30:00")  # doctest: +SKIP
            >>> workflow = snapshot_ml.lookup_workflow("2-ABC1")  # doctest: +SKIP
            >>> workflow.description = "New description"  # Raises DerivaMLException  # doctest: +SKIP
    """

    # extra="forbid" guards against the silent kwarg-drop that masked the
    # rid -> workflow_rid rename regression (#226): the renamed field was
    # passed under its old name, Pydantic dropped the unknown kwarg without
    # error, and every find_workflows()/lookup_workflow() result silently
    # carried workflow_rid=None. Forbidding extra fields turns that class of
    # mistake into a loud ValidationError at construction time. Scoped to
    # Workflow only (mirrors the existing STRICT_VALIDATION_CONFIG idiom in
    # core/validation.py) rather than widening the shared VALIDATION_CONFIG.
    model_config = {**VALIDATION_CONFIG, "extra": "forbid"}

    name: str
    workflow_type: str | list[str]
    description: str | None = None
    url: str | None = None
    version: str | None = None
    workflow_rid: RID | None = None
    checksum: str | None = None
    is_notebook: bool = False
    git_root: Path | None = None
    allow_dirty: bool = False

    _ml_instance: "DerivaMLCatalog | None" = PrivateAttr(default=None)
    _logger: logging.Logger = PrivateAttr(default_factory=lambda: get_logger(__name__))

    @field_validator("workflow_type", mode="before")
    @classmethod
    def _normalize_workflow_type(cls, v: str | list[str]) -> list[str]:
        """Normalize workflow_type to always be a list of strings."""
        if isinstance(v, str):
            return [v]
        return list(v)

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to intercept description and workflow_type updates.

        When the workflow is bound to a catalog (via lookup_workflow), setting
        the ``description`` or ``workflow_type`` properties will update the catalog
        record. If the catalog is read-only (a snapshot), a DerivaMLException is raised.

        Args:
            name: The attribute name being set.
            value: The value to set.

        Raises:
            DerivaMLException: If attempting to set properties on a read-only
                catalog (snapshot), or if workflow_type is not a valid vocabulary term.

        Examples:
            Update description::

                >>> workflow = ml.lookup_workflow("2-ABC1")  # doctest: +SKIP
                >>> workflow.description = "Updated description"  # doctest: +SKIP

            Update workflow type::

                >>> workflow = ml.lookup_workflow("2-ABC1")  # doctest: +SKIP
                >>> workflow.workflow_type = "python_notebook"  # doctest: +SKIP
        """
        # Only intercept updates after full initialization
        # Use __dict__ check to avoid recursion during Pydantic model construction
        if (
            "__pydantic_private__" in self.__dict__
            and self.__dict__.get("__pydantic_private__", {}).get("_ml_instance") is not None
        ):
            if name == "description":
                self._update_description_in_catalog(value)
            elif name == "workflow_type":
                # Normalize to list
                if isinstance(value, str):
                    value = [value]
                self._update_workflow_types_in_catalog(value)
        super().__setattr__(name, value)

    def _check_writable_catalog(self, operation: str) -> None:
        """Check that the catalog is writable and workflow is registered.

        Delegates to the shared free helper in
        :mod:`deriva_ml.execution._helpers` — same contract used
        by :class:`ExecutionRecord`. Kept as a thin instance
        method so the in-class call sites (``description`` setter,
        ``_update_description_in_catalog``) read naturally.

        Args:
            operation: Description of the operation being attempted.

        Raises:
            DerivaMLException: If the workflow is not registered (no RID),
                or if the catalog is read-only (a snapshot).
        """
        from deriva_ml.execution._helpers import check_writable_catalog

        check_writable_catalog(
            rid=self.workflow_rid,
            ml_instance=self._ml_instance,
            entity_label="Workflow",
            operation=operation,
        )

    def _update_description_in_catalog(self, new_description: str | None) -> None:
        """Update the description field in the catalog.

        This internal method is called when the description property is set
        on a catalog-bound Workflow object.

        Args:
            new_description: The new description value.

        Raises:
            DerivaMLException: If the workflow is not registered (no RID),
                or if the catalog is read-only (a snapshot).
        """
        from deriva_ml.execution._helpers import update_field_in_catalog

        self._check_writable_catalog("update description")
        update_field_in_catalog(
            rid=self.workflow_rid,
            ml_instance=self._ml_instance,
            table_name="Workflow",
            updates={"Description": new_description},
        )

    def _get_workflow_type_association_table(self):
        """Get the association table for workflow types.

        Returns:
            Tuple of (table_name, table_path) for the Workflow-Workflow_Type association table.
        """
        atable_name = "Workflow_Workflow_Type"
        pb = self._ml_instance.pathBuilder()
        atable_path = pb.schemas[self._ml_instance.ml_schema].tables[atable_name]
        return atable_name, atable_path

    @property
    def workflow_types(self) -> list[str]:
        """Get the workflow types from the catalog.

        This property fetches the current workflow types directly from the catalog,
        ensuring consistency when multiple Workflow instances reference the same
        workflow or when types are modified externally.

        When not bound to a catalog, returns the local ``workflow_type`` field.

        Returns:
            List of workflow type term names from the Workflow_Type vocabulary.
        """
        if self._ml_instance is not None:
            _, atable_path = self._get_workflow_type_association_table()
            wt_types = (
                atable_path.filter(atable_path.Workflow == self.workflow_rid).attributes(atable_path.Workflow_Type).fetch()
            )
            return [wt[MLVocab.workflow_type] for wt in wt_types]
        return list(self.workflow_type)

    def add_workflow_type(self, workflow_type: str | VocabularyTerm) -> None:
        """Add a workflow type to this workflow.

        Adds a type term to this workflow if it's not already present. The term must
        exist in the Workflow_Type vocabulary.

        Args:
            workflow_type: Term name (string) or VocabularyTerm object from Workflow_Type vocabulary.

        Raises:
            DerivaMLException: If the workflow is not registered (no RID),
                the catalog is read-only, or the term doesn't exist.
        """
        self._check_writable_catalog("add workflow_type")

        if isinstance(workflow_type, VocabularyTerm):
            vocab_term = workflow_type
        else:
            vocab_term = self._ml_instance.lookup_term(MLVocab.workflow_type, workflow_type)

        if vocab_term.name in self.workflow_types:
            return

        _, atable_path = self._get_workflow_type_association_table()
        atable_path.insert([{MLVocab.workflow_type: vocab_term.name, "Workflow": self.workflow_rid}])

    def remove_workflow_type(self, workflow_type: str | VocabularyTerm) -> None:
        """Remove a workflow type from this workflow.

        Removes a type term from this workflow if it's currently associated.

        Args:
            workflow_type: Term name (string) or VocabularyTerm object from Workflow_Type vocabulary.

        Raises:
            DerivaMLException: If the workflow is not registered (no RID),
                the catalog is read-only, or the term doesn't exist.
        """
        self._check_writable_catalog("remove workflow_type")

        if isinstance(workflow_type, VocabularyTerm):
            vocab_term = workflow_type
        else:
            vocab_term = self._ml_instance.lookup_term(MLVocab.workflow_type, workflow_type)

        if vocab_term.name not in self.workflow_types:
            return

        _, atable_path = self._get_workflow_type_association_table()
        atable_path.filter((atable_path.Workflow == self.workflow_rid) & (atable_path.Workflow_Type == vocab_term.name)).delete()

    def add_workflow_types(self, workflow_types: str | VocabularyTerm | list[str | VocabularyTerm]) -> None:
        """Add one or more workflow types to this workflow.

        Args:
            workflow_types: Single term or list of terms. Can be strings (term names)
                or VocabularyTerm objects.

        Raises:
            DerivaMLException: If any term doesn't exist in the Workflow_Type vocabulary.
        """
        types_to_add = [workflow_types] if not isinstance(workflow_types, list) else workflow_types

        for term in types_to_add:
            self.add_workflow_type(term)

    def _update_workflow_types_in_catalog(self, new_workflow_types: list[str]) -> None:
        """Replace all workflow types in the catalog with the given list.

        This internal method is called when the workflow_type property is set
        on a catalog-bound Workflow object. Each new type must be a valid
        term from the Workflow_Type vocabulary.

        Args:
            new_workflow_types: List of new workflow type names.

        Raises:
            DerivaMLException: If the workflow is not registered (no RID),
                the catalog is read-only (a snapshot), or any workflow_type
                is not a valid vocabulary term.
        """
        self._check_writable_catalog("update workflow_type")

        # Validate all new types exist in vocabulary
        for wt in new_workflow_types:
            self._ml_instance.lookup_term(MLVocab.workflow_type, wt)

        # Delete all existing type associations
        _, atable_path = self._get_workflow_type_association_table()
        atable_path.filter(atable_path.Workflow == self.workflow_rid).delete()

        # Insert new type associations
        if new_workflow_types:
            atable_path.insert([{MLVocab.workflow_type: wt, "Workflow": self.workflow_rid} for wt in new_workflow_types])

    @model_validator(mode="after")
    def setup_url_checksum(self) -> "Workflow":
        """Pydantic post-construction validator that fills in ``url`` and ``checksum``.

        Runs automatically after every ``Workflow(...)`` construction
        (it is a ``@model_validator(mode="after")``). For any field
        the caller did not provide, the validator derives a value from
        the current execution context:

        - ``url`` — set to the resolved source URL of the calling
          script/notebook. Resolution prefers a Docker image
          identifier (when ``DERIVA_MCP_IN_DOCKER=true``), otherwise a
          GitHub ``/blob/<commit>/<path>`` URL from the local git
          checkout. When the script is not in a git repo and
          ``allow_dirty=True``, ``url`` is left empty (``""``) — there
          is no ``file://`` fallback.
        - ``checksum`` — set to the git commit SHA of the calling
          script (or the Docker image digest when running in Docker).
        - ``version`` — in the local-git path (the common case) it is
          set from ``get_dynamic_version()`` (the version derived from
          the local git checkout); in the Docker path it is set from
          ``DERIVA_MCP_VERSION`` instead.

        Caller-supplied values are never overwritten — this is a
        "fill in the blanks" validator, not a re-derivation.

        Environment variable overrides:
            - ``DERIVA_ML_WORKFLOW_URL``: force-set ``url``.
            - ``DERIVA_ML_WORKFLOW_CHECKSUM``: force-set ``checksum``.
            - ``DERIVA_MCP_IN_DOCKER=true``: use Docker image metadata
              instead of git.

        Docker-only environment variables (consulted when
        ``DERIVA_MCP_IN_DOCKER=true``):
            - ``DERIVA_MCP_VERSION``: semantic version of the Docker image.
            - ``DERIVA_MCP_GIT_COMMIT``: git commit hash at image build time.
            - ``DERIVA_MCP_IMAGE_DIGEST``: image digest (unique identifier).
            - ``DERIVA_MCP_IMAGE_NAME``: image name (e.g.
              ``ghcr.io/informatics-isi-edu/deriva-ml-mcp``).

        Returns:
            Workflow: ``self`` (the same instance, mutated in place).
            Pydantic ``mode="after"`` validators must return the
            model.

        Raises:
            DerivaMLException: If the validator cannot determine a
                URL or checksum from any source (e.g. not in a git
                repo, Docker env vars missing, no explicit overrides).
        """
        self._logger = get_logger(__name__)
        # Check if running in Docker container (no git repo available)
        if os.environ.get("DERIVA_MCP_IN_DOCKER", "").lower() == "true":
            # Use Docker image metadata for provenance
            self.version = self.version or os.environ.get("DERIVA_MCP_VERSION", "")

            # Use image digest as checksum (unique identifier for the container)
            # Fall back to git commit if digest not available
            self.checksum = self.checksum or (
                os.environ.get("DERIVA_MCP_IMAGE_DIGEST", "") or os.environ.get("DERIVA_MCP_GIT_COMMIT", "")
            )

            # Build URL pointing to the Docker image or source repo
            if not self.url:
                image_name = os.environ.get(
                    "DERIVA_MCP_IMAGE_NAME",
                    "ghcr.io/informatics-isi-edu/deriva-ml-mcp",
                )
                image_digest = os.environ.get("DERIVA_MCP_IMAGE_DIGEST", "")
                if image_digest:
                    # URL format: image@sha256:digest
                    self.url = f"{image_name}@{image_digest}"
                else:
                    # Fall back to source repo with git commit
                    source_url = "https://github.com/informatics-isi-edu/deriva-ml-mcp"
                    git_commit = os.environ.get("DERIVA_MCP_GIT_COMMIT", "")
                    self.url = f"{source_url}/commit/{git_commit}" if git_commit else source_url

            return self

        # Check to see if execution file info is being passed in by calling program (notebook runner)
        if "DERIVA_ML_WORKFLOW_URL" in os.environ:
            self.url = os.environ["DERIVA_ML_WORKFLOW_URL"]
            self.checksum = os.environ.get("DERIVA_ML_WORKFLOW_CHECKSUM", "")
            notebook_path = os.environ.get("DERIVA_ML_NOTEBOOK_PATH")
            if notebook_path:
                self.git_root = Workflow._get_git_root(Path(notebook_path))
            self.is_notebook = True
            return self

        # Standard git detection for local development
        # Check env var for allow_dirty (set by CLI --allow-dirty flag or dry-run mode)
        if os.environ.get("DERIVA_ML_ALLOW_DIRTY", "").lower() == "true":
            self.allow_dirty = True
        if os.environ.get("DERIVA_ML_DRY_RUN", "").lower() == "true":
            self.allow_dirty = True

        if not self.url:
            path, self.is_notebook = Workflow._get_python_script()
            self.url, self.checksum = Workflow.get_url_and_checksum(path, allow_dirty=self.allow_dirty)
            self.git_root = Workflow._get_git_root(path)

        self.version = self.version or Workflow.get_dynamic_version(root=str(self.git_root or Path.cwd()))
        return self

    @staticmethod
    def get_url_and_checksum(executable_path: Path, allow_dirty: bool = False) -> tuple[str, str]:
        """Determines the Git URL and checksum for a file.

        Computes the Git repository URL and file checksum for the specified path.
        For notebooks, strips cell outputs before computing the checksum.

        Args:
            executable_path: Path to the workflow file.
            allow_dirty: If True, log a warning instead of raising an error
                when the file has uncommitted changes. Defaults to False.

        Returns:
            tuple[str, str]: (GitHub URL, Git object hash)

        Raises:
            DerivaMLException: If not in a Git repository.
            DerivaMLDirtyWorkflowError: If the file has uncommitted changes
                and allow_dirty is False.

        Example:
            >>> url, checksum = Workflow.get_url_and_checksum(Path("analysis.ipynb"))  # doctest: +SKIP
            >>> print(f"URL: {url}")  # doctest: +SKIP
            >>> print(f"Checksum: {checksum}")  # doctest: +SKIP
        """
        # The "must be inside a git checkout" guard is a precondition
        # for honest provenance. But ``allow_dirty=True`` is the
        # documented escape hatch for ad-hoc / dry-run / notebook
        # workflows that don't need a clean checkout — refusing to
        # run those because the cwd isn't inside git defeats the
        # intent of the flag. When ``allow_dirty=True``, downgrade
        # the no-git precondition to a warning and return empty
        # provenance (URL/checksum both empty strings); when False
        # (the default), keep the hard refuse.
        try:
            subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            if allow_dirty:
                logger.warning(
                    "Not executing in a Git repository; allow_dirty=True, "
                    "returning empty URL/checksum. Provenance will not be "
                    "recoverable for this workflow."
                )
                return "", ""
            raise DerivaMLException("Not executing in a Git repository.")

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
                raise DerivaMLDirtyWorkflowError(str(executable_path), dirty_paths=dirty_paths)

        # If you are in a notebook, strip out the outputs before computing the checksum.
        if executable_path != "REPL":
            if "ipynb" == executable_path.suffix:
                strip_proc = subprocess.run(
                    ["nbstripout", "-t", str(executable_path)],
                    capture_output=True,
                )
                hash_proc = subprocess.run(
                    ["git", "hash-object", "--stdin"],
                    input=strip_proc.stdout,
                    capture_output=True,
                    text=True,
                )
                checksum = hash_proc.stdout.strip()
            else:
                checksum = subprocess.run(
                    ["git", "hash-object", str(executable_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip()
        else:
            checksum = "1"
        return github_url, checksum

    # Default paths excluded from the dirty-tree check. These are
    # directories the project conventions treat as scratch / output /
    # findings space — not code the workflow could have read, so changes
    # under them don't affect provenance.
    _DEFAULT_DIRTY_CHECK_EXCLUSIONS: ClassVar[tuple[str, ...]] = (
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

    @staticmethod
    def _get_git_root(executable_path: Path) -> str | None:
        """Gets the root directory of the Git repository.

        Args:
            executable_path: Path to check for Git repository.

        Returns:
            str | None: Absolute path to repository root, or None if not in repository.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=executable_path.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None  # Not in a git repository

    @staticmethod
    def _check_nbstrip_status() -> None:
        """Checks if nbstripout is installed and configured.

        Verifies that the nbstripout tool is available and properly installed in the
        Git repository. Issues warnings if setup is incomplete.
        """
        logger = get_logger(__name__)
        try:
            if subprocess.run(
                ["nbstripout", "--is-installed"],
                check=False,
                capture_output=True,
            ).returncode:
                logger.warning("nbstripout is not installed in repository. Please run nbstripout --install")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("nbstripout is not found. Please install it with: pip install nbstripout")

    @staticmethod
    def _in_repl():
        # Standard Python interactive mode
        if hasattr(sys, "ps1"):
            return True

        # Interactive mode forced by -i
        if sys.flags.interactive:
            return True

        # IPython / Jupyter detection
        try:
            from IPython import get_ipython

            if get_ipython() is not None:
                return True
        except ImportError:
            pass

        return False

    @staticmethod
    def _get_python_script() -> tuple[Path, bool]:
        """Return the path to the currently executing script.

        Returns:
            ``(script_path, is_notebook)`` — ``script_path`` is the
            resolved absolute path; ``is_notebook`` is ``True`` when
            running inside a Jupyter notebook (``.ipynb`` path
            resolves).
        """
        # ``find_caller._get_notebook_path()`` is the single source
        # of truth for Jupyter session lookup (audit §2.7 / §4.6).
        from deriva_ml.execution.find_caller import _get_notebook_path

        is_notebook = _get_notebook_path() is not None
        return Path(_get_calling_module()), is_notebook

    @staticmethod
    def _github_url(executable_path: Path) -> tuple[str, list[str]]:
        """Return a GitHub URL for the latest commit of the script from which this routine is called.

        This routine is used to be called from a script or notebook (e.g., python -m file). It assumes that
        the file is in a GitHub repository and committed.  It returns a URL to the last commited version of this
        file in GitHub.

        Returns: A tuple with the github_url and a list of porcelain lines describing
            uncommitted changes that affect provenance (empty if clean).

        """

        # Get repo URL from local GitHub repo.
        if executable_path == "REPL":
            return "REPL", ["REPL"]
        # ``check=True`` is load-bearing: without it ``subprocess.run``
        # never raises, ``result.stdout`` for a missing ``origin``
        # remote is an empty string, and ``github_url`` becomes
        # ``""`` — silently recorded as provenance pointing at
        # ``/blob/<sha>/<path>`` with no host. The
        # ``CalledProcessError`` catch was unreachable until this
        # flag was added.
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                cwd=executable_path.parent,
                check=True,
            )
            github_url = result.stdout.strip().removesuffix(".git")
        except subprocess.CalledProcessError:
            raise DerivaMLException("No GIT remote found")

        # Find the root directory for the repository
        repo_root = Workflow._get_git_root(executable_path)

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
            # ``--untracked-files=all`` expands directory-level untracked
            # entries (``?? src/``) into per-file entries (``?? src/extra.py``)
            # so the per-prefix filter and the per-path error message both
            # operate on the leaf filename rather than the parent directory.
            result = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            dirty_paths = Workflow._filter_dirty_paths(result.stdout)
        except subprocess.CalledProcessError:
            dirty_paths = []  # If the Git command fails, assume no changes

        # Get SHA-1 hash of latest commit of the file in the
        # repository. ``check=False`` here is deliberate: a file
        # that has never been committed (a new script in a worktree
        # the user is iterating on) legitimately has no log; the
        # empty ``sha`` string yields a URL pointing at ``/blob//``,
        # which the dirty-flow already treats as a non-clean
        # provenance signal upstream.
        result = subprocess.run(
            ["git", "log", "-n", "1", "--pretty=format:%H", executable_path],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        sha = result.stdout.strip()
        url = f"{github_url}/blob/{sha}/{executable_path.relative_to(repo_root)}"
        return url, dirty_paths

    @staticmethod
    def get_dynamic_version(root: str | os.PathLike | None = None) -> str:
        """Return a dynamic version string derived from VCS state.

        Wraps :func:`setuptools_scm.get_version` (the same mechanism used at
        build time). The returned string includes distance-from-tag and an
        optional ``.dirty`` suffix when the working tree has uncommitted
        changes.

        Args:
            root: Repository root to introspect. When ``None``, uses the
                installed deriva-ml package's parent directory (i.e. the
                source checkout that's being developed against).

        Returns:
            A setuptools-scm-style version string (e.g. ``"1.2.3"``,
            ``"1.2.3.post2+g1234abc"``, or ``"1.2.3.post2+g1234abc.dirty"``).

        Raises:
            RuntimeError: If ``setuptools_scm`` is not importable in the
                current environment.
        """
        # Historical note: this routine used to unconditionally set
        # ``os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"`` as a
        # defensive measure against a ``_distutils_hack`` assertion that
        # older setuptools versions hit on some macOS configurations.
        #
        # That mutation is process-wide and leaks into subprocess.Popen
        # inheritance, which broke third-party packages on Python 3.12+
        # that still import ``distutils.version`` (distutils is removed
        # from the stdlib per PEP 632 but setuptools vendors it back in
        # its own package — the env var short-circuits that vendored
        # fallback). Net effect: any test that called a workflow
        # helper would poison the process env, and any later subprocess
        # under Python 3.13+ would crash importing ``distutils``.
        #
        # Python 3.12 is the floor for this project (pyproject.toml:
        # ``requires-python = ">=3.12"``) so the distutils_hack concern
        # is moot — distutils is simply gone, and setuptools's own
        # fallback handles it. We drop the env mutation entirely.
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="_distutils_hack",
        )
        try:
            from setuptools_scm import get_version
        except Exception as e:  # ImportError or anything environment-specific
            raise RuntimeError(f"setuptools_scm is not available: {e}") from e

        if root is None:
            # Adjust this to point at your repo root if needed
            root = Path(__file__).resolve().parents[1]

        return get_version(root=root)
