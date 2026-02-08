from __future__ import annotations

import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator
from requests import RequestException

from deriva_ml.core.definitions import RID
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.execution.find_caller import _get_calling_module

if TYPE_CHECKING:
    from deriva_ml.interfaces import DerivaMLCatalog

try:
    from IPython.core.getipython import get_ipython
except ImportError:  # Graceful fallback if IPython isn't installed.

    def get_ipython() -> None:
        return None


try:
    from jupyter_server.serverapp import list_running_servers

    def get_servers() -> list[Any]:
        return list(list_running_servers())
except ImportError:

    def list_running_servers():
        return []

    def get_servers() -> list[Any]:
        return list_running_servers()


try:
    from ipykernel.connect import get_connection_file

    def get_kernel_connection() -> str:
        return get_connection_file()
except ImportError:

    def get_connection_file():
        return ""

    def get_kernel_connection() -> str:
        return get_connection_file()


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
        workflow_type (str): Type of workflow (must be a controlled vocabulary term).
            When the workflow is bound to a writable catalog, setting this property
            will update the catalog record. The new value must be a valid term from
            the Workflow_Type vocabulary.
        version (str | None): Version identifier (semantic versioning).
        description (str | None): Description of workflow purpose and behavior.
            When the workflow is bound to a writable catalog, setting this property
            will update the catalog record.
        rid (RID | None): Resource Identifier if registered in catalog.
        checksum (str | None): Git hash of workflow source code.
        is_notebook (bool): Whether workflow is a Jupyter notebook.

    Example:
        Create a workflow programmatically::

            >>> workflow = Workflow(
            ...     name="RNA Analysis",
            ...     url="https://github.com/org/repo/analysis.ipynb",
            ...     workflow_type="python_notebook",
            ...     version="1.0.0",
            ...     description="RNA sequence analysis"
            ... )

        Look up an existing workflow by RID and update its properties::

            >>> workflow = ml.lookup_workflow("2-ABC1")
            >>> workflow.description = "Updated description for RNA analysis"
            >>> workflow.workflow_type = "python_script"
            >>> print(workflow.description)
            Updated description for RNA analysis

        Look up by URL and update::

            >>> url = "https://github.com/org/repo/blob/abc123/analysis.py"
            >>> workflow = ml.lookup_workflow_by_url(url)
            >>> workflow.description = "New description"

        Attempting to update on a read-only catalog raises an error::

            >>> snapshot_ml = ml.catalog_snapshot("2023-01-15T10:30:00")
            >>> workflow = snapshot_ml.lookup_workflow("2-ABC1")
            >>> workflow.description = "New description"  # Raises DerivaMLException
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    workflow_type: str
    description: str | None = None
    url: str | None = None
    version: str | None = None
    rid: RID | None = None
    checksum: str | None = None
    is_notebook: bool = False
    git_root: Path | None = None

    _ml_instance: "DerivaMLCatalog | None" = PrivateAttr(default=None)
    _logger: logging.Logger = PrivateAttr(default=10)

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

                >>> workflow = ml.lookup_workflow("2-ABC1")
                >>> workflow.description = "Updated description"

            Update workflow type::

                >>> workflow = ml.lookup_workflow("2-ABC1")
                >>> workflow.workflow_type = "python_notebook"
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
                self._update_workflow_type_in_catalog(value)
        super().__setattr__(name, value)

    def _check_writable_catalog(self, operation: str) -> None:
        """Check that the catalog is writable and workflow is registered.

        Args:
            operation: Description of the operation being attempted.

        Raises:
            DerivaMLException: If the workflow is not registered (no RID),
                or if the catalog is read-only (a snapshot).
        """
        # Import here to avoid circular dependency at module load
        import importlib
        _deriva_core = importlib.import_module("deriva.core")
        ErmrestSnapshot = _deriva_core.ErmrestSnapshot

        if self.rid is None:
            raise DerivaMLException(
                f"Cannot {operation}: Workflow is not registered in the catalog (no RID)"
            )

        if isinstance(self._ml_instance.catalog, ErmrestSnapshot):
            raise DerivaMLException(
                f"Cannot {operation} on a read-only catalog snapshot. "
                "Use a writable catalog connection instead."
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
        self._check_writable_catalog("update description")

        # Update the catalog record
        pb = self._ml_instance.pathBuilder()
        workflow_path = pb.schemas[self._ml_instance.ml_schema].Workflow
        workflow_path.update([{"RID": self.rid, "Description": new_description}])

    def _update_workflow_type_in_catalog(self, new_workflow_type: str) -> None:
        """Update the workflow_type field in the catalog.

        This internal method is called when the workflow_type property is set
        on a catalog-bound Workflow object. The new workflow type must be a valid
        term from the Workflow_Type vocabulary.

        Args:
            new_workflow_type: The new workflow type (must be a valid vocabulary term).

        Raises:
            DerivaMLException: If the workflow is not registered (no RID),
                the catalog is read-only (a snapshot), or the workflow_type
                is not a valid vocabulary term.
        """
        self._check_writable_catalog("update workflow_type")

        # Validate that the new workflow type exists in vocabulary
        from deriva_ml.core.definitions import MLVocab
        self._ml_instance.lookup_term(MLVocab.workflow_type, new_workflow_type)

        # Update the catalog record
        pb = self._ml_instance.pathBuilder()
        workflow_path = pb.schemas[self._ml_instance.ml_schema].Workflow
        workflow_path.update([{"RID": self.rid, "Workflow_Type": new_workflow_type}])

    @model_validator(mode="after")
    def setup_url_checksum(self) -> "Workflow":
        """Creates a workflow from the current execution context.

        Identifies the currently executing program (script or notebook) and creates
        a workflow definition. Automatically determines the Git repository information
        and source code checksum.

        The behavior can be configured using environment variables:
            - DERIVA_ML_WORKFLOW_URL: Override the detected workflow URL
            - DERIVA_ML_WORKFLOW_CHECKSUM: Override the computed checksum
            - DERIVA_MCP_IN_DOCKER: Set to "true" to use Docker metadata instead of git

        Docker environment variables (used when DERIVA_MCP_IN_DOCKER=true):
            - DERIVA_MCP_VERSION: Semantic version of the Docker image
            - DERIVA_MCP_GIT_COMMIT: Git commit hash at build time
            - DERIVA_MCP_IMAGE_DIGEST: Docker image digest (unique identifier)
            - DERIVA_MCP_IMAGE_NAME: Docker image name (e.g., ghcr.io/org/repo)

        Args:

        Returns:
            Workflow: New workflow instance with detected Git information.

        Raises:
            DerivaMLException: If not in a Git repository or detection fails (non-Docker).

        Example:
            >>> workflow = Workflow.create_workflow(
            ...     name="Sample Analysis",
            ...     workflow_type="python_script",
            ...     description="Process sample data"
            ... )
        """
        self._logger = logging.getLogger("deriva_ml")

        # Check if running in Docker container (no git repo available)
        if os.environ.get("DERIVA_MCP_IN_DOCKER", "").lower() == "true":
            # Use Docker image metadata for provenance
            self.version = self.version or os.environ.get("DERIVA_MCP_VERSION", "")

            # Use image digest as checksum (unique identifier for the container)
            # Fall back to git commit if digest not available
            self.checksum = self.checksum or (
                os.environ.get("DERIVA_MCP_IMAGE_DIGEST", "")
                or os.environ.get("DERIVA_MCP_GIT_COMMIT", "")
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
        if not self.url:
            path, self.is_notebook = Workflow._get_python_script()
            self.url, self.checksum = Workflow.get_url_and_checksum(path)
            self.git_root = Workflow._get_git_root(path)

        self.version = self.version or Workflow.get_dynamic_version(root=str(self.git_root or Path.cwd()))
        return self

    @staticmethod
    def get_url_and_checksum(executable_path: Path) -> tuple[str, str]:
        """Determines the Git URL and checksum for a file.

        Computes the Git repository URL and file checksum for the specified path.
        For notebooks, strips cell outputs before computing the checksum.

        Args:
            executable_path: Path to the workflow file.

        Returns:
            tuple[str, str]: (GitHub URL, Git object hash)

        Raises:
            DerivaMLException: If not in a Git repository.

        Example:
            >>> url, checksum = Workflow.get_url_and_checksum(Path("analysis.ipynb"))
            >>> print(f"URL: {url}")
            >>> print(f"Checksum: {checksum}")
        """
        try:
            subprocess.run(
                "git rev-parse --is-inside-work-tree",
                capture_output=True,
                text=True,
                shell=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            raise DerivaMLException("Not executing in a Git repository.")

        github_url, is_dirty = Workflow._github_url(executable_path)

        if is_dirty:
            logging.getLogger("deriva_ml").warning(
                f"File {executable_path} has been modified since last commit. Consider commiting before executing"
            )

        # If you are in a notebook, strip out the outputs before computing the checksum.
        cmd = (
            f"nbstripout -t {executable_path} | git hash-object --stdin"
            if "ipynb" == executable_path.suffix
            else f"git hash-object {executable_path}"
        )
        checksum = (
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                shell=True,
            ).stdout.strip()
            if executable_path != "REPL"
            else "1"
        )
        return github_url, checksum

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
        logger = logging.getLogger("deriva_ml")
        try:
            if subprocess.run(
                ["nbstripout", "--is-installed"],
                check=False,
                capture_output=True,
            ).returncode:
                logger.warning("nbstripout is not installed in repository. Please run nbstripout --install")
        except subprocess.CalledProcessError:
            logger.error("nbstripout is not found.")

    @staticmethod
    def _get_notebook_path() -> Path | None:
        """Gets the path of the currently executing notebook.

        Returns:
            Path | None: Absolute path to current notebook, or None if not in notebook.
        """

        server, session = Workflow._get_notebook_session()

        if server and session:
            relative_path = session["notebook"]["path"]
            # Join the notebook directory with the relative path
            return Path(server["root_dir"]) / relative_path
        else:
            return None

    @staticmethod
    def _get_notebook_session() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Return the absolute path of the current notebook."""
        # Get the kernel's connection file and extract the kernel ID
        try:
            if not (connection_file := Path(get_kernel_connection()).name):
                return None, None
        except RuntimeError:
            return None, None

        # Extract kernel ID from connection filename.
        # Standard Jupyter format: "kernel-<kernel_id>.json"
        # PyCharm/other formats may vary: "<kernel_id>.json" or other patterns
        kernel_id = None
        if connection_file.startswith("kernel-") and "-" in connection_file:
            # Standard format: kernel-<uuid>.json
            parts = connection_file.split("-", 1)
            if len(parts) > 1:
                kernel_id = parts[1].rsplit(".", 1)[0]
        else:
            # Fallback: assume filename (without extension) is the kernel ID
            kernel_id = connection_file.rsplit(".", 1)[0]

        if not kernel_id:
            return None, None

        # Look through the running server sessions to find the matching kernel ID
        for server in get_servers():
            try:
                # If a token is required for authentication, include it in headers
                token = server.get("token", "")
                headers = {}
                if token:
                    headers["Authorization"] = f"token {token}"

                try:
                    sessions_url = server["url"] + "api/sessions"
                    response = requests.get(sessions_url, headers=headers)
                    response.raise_for_status()
                    sessions = response.json()
                except RequestException as e:
                    raise e
                for sess in sessions:
                    if sess["kernel"]["id"] == kernel_id:
                        return server, sess
            except Exception as _e:
                # Ignore servers we can't connect to.
                pass
        return None, None

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
        """Return the path to the currently executing script"""
        is_notebook = Workflow._get_notebook_path() is not None
        return Path(_get_calling_module()), is_notebook

    @staticmethod
    def _github_url(executable_path: Path) -> tuple[str, bool]:
        """Return a GitHub URL for the latest commit of the script from which this routine is called.

        This routine is used to be called from a script or notebook (e.g., python -m file). It assumes that
        the file is in a GitHub repository and committed.  It returns a URL to the last commited version of this
        file in GitHub.

        Returns: A tuple with the gethub_url and a boolean to indicate if uncommited changes
            have been made to the file.

        """

        # Get repo URL from local GitHub repo.
        if executable_path == "REPL":
            return "REPL", True
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                cwd=executable_path.parent,
            )
            github_url = result.stdout.strip().removesuffix(".git")
        except subprocess.CalledProcessError:
            raise DerivaMLException("No GIT remote found")

        # Find the root directory for the repository
        repo_root = Workflow._get_git_root(executable_path)

        # Now check to see if a file has been modified since the last commit.
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=executable_path.parent,
                capture_output=True,
                text=True,
                check=False,
            )
            is_dirty = bool("M " in result.stdout.strip())  # Returns True if the output indicates a modified file
        except subprocess.CalledProcessError:
            is_dirty = False  # If the Git command fails, assume no changes

        """Get SHA-1 hash of latest commit of the file in the repository"""

        result = subprocess.run(
            ["git", "log", "-n", "1", "--pretty=format:%H", executable_path],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        sha = result.stdout.strip()
        url = f"{github_url}/blob/{sha}/{executable_path.relative_to(repo_root)}"
        return url, is_dirty

    @staticmethod
    def get_dynamic_version(root: str | os.PathLike | None = None) -> str:
        """
        Return a dynamic version string based on VCS state (setuptools_scm),
        including dirty/uncommitted changes if configured.

        Works under uv / Python 3.10+ by forcing setuptools to use stdlib distutils.
        """
        # 1) Tell setuptools to use stdlib distutils (or no override) to avoid
        #    the '_distutils_hack' assertion you hit.
        os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")

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
