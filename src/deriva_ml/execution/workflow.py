import inspect
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests
from requests import RequestException

from deriva_ml.core.definitions import RID
from deriva_ml.core.exceptions import DerivaMLException

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


class Workflow:
    """Represents a computational workflow in DerivaML.

    A workflow defines a computational process or analysis pipeline. Each workflow has
    a unique identifier, source code location, and type. Workflows are typically
    associated with Git repositories for version control.

    Attributes:
        name (str): Human-readable name of the workflow.
        url (str): URI to the workflow source code (typically a GitHub URL).
        workflow_type (str): Type of workflow (must be a controlled vocabulary term).
        version (str | None): Version identifier (semantic versioning).
        description (str | None): Description of workflow purpose and behavior.
        rid (RID | None): Resource Identifier if registered in catalog.
        checksum (str | None): Git hash of workflow source code.
        is_notebook (bool): Whether workflow is a Jupyter notebook.

    Example:
        >>> workflow = Workflow(
        ...     name="RNA Analysis",
        ...     url="https://github.com/org/repo/analysis.ipynb",
        ...     workflow_type="python_notebook",
        ...     version="1.0.0",
        ...     description="RNA sequence analysis"
        ... )
    """

    def __init__(
        self,
        name: str,
        workflow_type: str,
        description: str | None = None,
        url: str | None = None,
        version: str | None = None,
        rid: RID | None = None,
        checksum: str | None = None,
        is_notebook: bool = False,
    ):
        """Creates a workflow from the current execution context.

        Identifies the currently executing program (script or notebook) and creates
        a workflow definition. Automatically determines the Git repository information
        and source code checksum.

        The behavior can be configured using environment variables:
            - DERIVA_ML_WORKFLOW_URL: Override the detected workflow URL
            - DERIVA_ML_WORKFLOW_CHECKSUM: Override the computed checksum

        Args:
            name: Human-readable name for the workflow.
            workflow_type: Type of workflow (must be a vocabulary term).
            description: Optional description of workflow purpose.

        Returns:
            Workflow: New workflow instance with detected Git information.

        Raises:
            DerivaMLException: If not in a Git repository or detection fails.

        Example:
            >>> workflow = Workflow.create_workflow(
            ...     name="Sample Analysis",
            ...     workflow_type="python_script",
            ...     description="Process sample data"
            ... )
        """
        self.name = name
        self.url = url
        self.workflow_type = workflow_type
        self.version = version
        self.description = description
        self.rid = rid
        self.checksum = checksum
        self.is_notebook = is_notebook
        """Initializes logging for the workflow."""

        # Check to see if execution file info is being passed in by calling program.
        if "DERIVA_ML_WORKFLOW_URL" in os.environ:
            self.url = os.environ["DERIVA_ML_WORKFLOW_URL"]
            self.checksum = os.environ["DERIVA_ML_WORKFLOW_CHECKSUM"]
            self.is_notebook = True

        if not self.url:
            path, self.is_notebook = Workflow._get_python_script()
            self.url, self.checksum = Workflow.get_url_and_checksum(path)

        self._logger = logging.getLogger("deriva_ml")

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

        kernel_id = connection_file.split("-", 1)[1].split(".")[0]

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
        is_notebook = True
        if not (filename := Workflow._get_notebook_path()):
            is_notebook = False
            stack = [
                s.filename
                for s in inspect.stack()
                if ("pycharm" not in s.filename) and ("site-packages" not in s.filename)
            ]
            # Get the caller's filename, which is two up the stack from here.
            filename = Path(stack[-1])
            if not filename.exists() or Workflow._in_repl():
                # Being called from the command line interpreter.
                filename = Path.cwd() / Path("REPL")
            # Get the caller's filename, which is two up the stack from here.
        else:
            raise DerivaMLException("Looking for caller failed")  # Stack is too shallow
        return filename, is_notebook

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
