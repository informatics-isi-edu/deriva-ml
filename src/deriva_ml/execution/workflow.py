import inspect
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import requests
from pydantic import (
    BaseModel,
    PrivateAttr,
)
from requests import RequestException

from deriva_ml.core.definitions import RID, DerivaMLException

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
    """A specification of a workflow.  Must have a name, URI to the workflow instance, and a type.  The workflow type
    needs to be an existing-controlled vocabulary term.

    Attributes:
        name: The name of the workflow
        url: The URI to the workflow instance.  In most cases there should be a GitHub URI to the code being executed.
        workflow_type: The type of the workflow.  Must be an existing controlled vocabulary term.
        version: The version of the workflow instance.  Should follow semantic versioning.
        description: A description of the workflow instance.  Can be in Markdown format.
        is_notebook: A boolean indicating whether this workflow instance is a notebook or not.
    """

    name: str
    url: str
    workflow_type: str
    version: str | None = None
    description: str | None = None
    rid: RID | None = None
    checksum: str | None = None
    is_notebook: bool = False

    _logger: Any = PrivateAttr()

    def __post_init__(self):
        self._logger = logging.getLogger("deriva_ml")

    @staticmethod
    def create_workflow(
        name: str,
        workflow_type: str,
        description: str = "",
    ) -> "Workflow":
        """Identify the current executing program and return a workflow RID for it

        Determine the notebook or script that is currently being executed. Assume that this is
        being executed from a cloned GitHub repository.  Determine the remote repository name for
        this object.  Then either retrieve an existing workflow for this executable or create
        a new one.

        Environment variables can be used to configure the behavior of this routine.
            DERIVA_ML_WORKFLOW_URL: The URL of the workflow instance.
            DERIVA_ML_WORKFLOW_CHECKSUM: The expected checksum of the workflow instance.


        Args:
            name: The name of the workflow.
            workflow_type: The type of the workflow.
            description: The description of the workflow.
        """

        # Check to see if execution file info is being passed in by calling program.
        if "DERIVA_ML_WORKFLOW_URL" in os.environ:
            github_url = os.environ["DERIVA_ML_WORKFLOW_URL"]
            checksum = os.environ["DERIVA_ML_WORKFLOW_CHECKSUM"]
            is_notebook = True
        else:
            path, is_notebook = Workflow._get_python_script()
            github_url, checksum = Workflow.get_url_and_checksum(path)

        return Workflow(
            name=name,
            url=github_url,
            checksum=checksum,
            description=description,
            workflow_type=workflow_type,
            is_notebook=is_notebook,
        )

    @staticmethod
    def get_url_and_checksum(executable_path: Path) -> tuple[str, str]:
        """Determine the checksum for a specified executable"""
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
    def _get_git_root(executable_path: Path):
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
        """Check to see if nbstrip is installed"""
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
        """Return the absolute path of the current notebook."""

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
    def _get_python_script() -> tuple[Path, bool]:
        """Return the path to the currently executing script"""
        is_notebook = True
        if not (filename := Workflow._get_notebook_path()):
            is_notebook = False
            stack = inspect.stack()
            # Get the caller's filename, which is two up the stack from here.
            if len(stack) > 1:
                filename = Path(stack[2].filename)
                if not filename.exists():
                    # Being called from the command line interpreter.
                    filename = Path("REPL")
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
                check=True,
            )
            is_dirty = bool("M " in result.stdout.strip())  # Returns True if the output indicates a modified file
        except subprocess.CalledProcessError:
            is_dirty = False  # If the Git command fails, assume no changes

        """Get SHA-1 hash of latest commit of the file in the repository"""
        result = subprocess.run(
            ["git", "log", "-n", "1", "--pretty=format:%H--", executable_path],
            cwd=executable_path.parent,
            capture_output=True,
            text=True,
            check=True,
        )
        sha = result.stdout.strip()
        url = f"{github_url}/blob/{sha}/{executable_path.relative_to(repo_root)}"
        return url, is_dirty
