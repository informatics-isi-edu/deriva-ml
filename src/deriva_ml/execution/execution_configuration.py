"""
Classes that are used to define an execution configuration.
"""

from __future__ import annotations

import inspect
import json
import logging
import os

from requests import RequestException
import requests
import subprocess
from typing import Optional, Any

from pydantic import (
    BaseModel,
    conlist,
    ConfigDict,
    field_validator,
    Field,
    PrivateAttr,
)
from pathlib import Path
import sys


from dataset.dataset_aux_classes import DatasetSpec
from core.deriva_definitions import RID, DerivaMLException

try:
    from IPython import get_ipython
except ImportError:  # Graceful fallback if IPython isn't installed.

    def get_ipython():
        """Dummy routine in case you are not running in IPython."""
        return None


try:
    from jupyter_server.serverapp import list_running_servers
except ImportError:

    def list_running_servers():
        """Dummy routine in case you are not running in Jupyter."""
        return []


try:
    from ipykernel import get_connection_file
except ImportError:

    def get_connection_file():
        """Dummy routine in case you are not running in Jupyter."""
        return ""


class Workflow(BaseModel):
    """A specification of a workflow.  Must have a name, URI to the workflow instance, and a type.  The workflow type
    needs to be an existing-controlled vocabulary term.

    Attributes:
        name: The name of the workflow
        url: The URI to the workflow instance.  In most cases should be a GitHub URI to the code being executed.
        workflow_type: The type of the workflow.  Must be an existing controlled vocabulary term.
        version: The version of the workflow instance.  Should follow semantic versioning.
        description: A description of the workflow instance.  Can be in Markdown format.
        is_notebook: A boolean indicating whether this workflow instance is a notebook or not.
    """

    name: str
    url: str
    workflow_type: str
    version: Optional[str] = None
    description: str = None
    rid: Optional[RID] = None
    checksum: Optional[str] = None
    is_notebook: bool = False

    _logger: Any = PrivateAttr()

    def __post_init__(self):
        self._logger = logging.getLogger("deriva_ml")

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
                logger.warning(
                    "nbstripout is not installed in repository. Please run nbstripout --install"
                )
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
            if not (connection_file := Path(get_connection_file()).name):
                return None, None
        except RuntimeError:
            return None, None

        kernel_id = connection_file.split("-", 1)[1].split(".")[0]

        # Look through the running server sessions to find the matching kernel ID
        for server in list_running_servers():
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
                    # Begin called from command line interpreter.
                    filename = Path("REPL")
                # Get the caller's filename, which is two up the stack from here.
            else:
                raise DerivaMLException(
                    "Looking for caller failed"
                )  # Stack is too shallow
        return filename, is_notebook

    @staticmethod
    def _github_url(executable_path: Path) -> tuple[str, bool]:
        """Return a GitHUB URL for the latest commit of the script from which this routine is called.

        This routine is used to be called from a script or notebook (e.g. python -m file). It assumes that
        the file is in a gitHUB repository and commited.  It returns a URL to the last commited version of this
        file in GitHUB.

        Returns: A tuple with the gethub_url and a boolean to indicated if uncommited changes
            have been made to the file.

        """

        # Get repo URL from local gitHub repo.
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

        # Now check to see if file has been modified since the last commit.
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=executable_path.parent,
                capture_output=True,
                text=True,
                check=True,
            )
            is_dirty = bool(
                "M " in result.stdout.strip()
            )  # Returns True if output indicates a modified file
        except subprocess.CalledProcessError:
            is_dirty = False  # If Git command fails, assume no changes

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
    def create_workflow(
        name: str,
        workflow_type: str,
        description: str = "",
    ) -> Workflow:
        """Identify current executing program and return a workflow RID for it

        Determine the notebook or script that is currently being executed. Assume that  this is
        being executed from a cloned GitHub repository.  Determine the remote repository name for
        this object.  Then either retrieve an existing workflow for this executable or create
        a new one.

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


class ExecutionConfiguration(BaseModel):
    """Define the parameters that are used to configure a specific execution.

    Attributes:
        datasets: List of dataset specifications which specify the dataset RID, version and if the dataset
            should be materialized.
        assets: List of assets to be downloaded prior to execution.  The values must be RIDs in an asset table
        parameters: Either a dictionary or a path to a JSON file that contains configuration parameters for the execution.
        workflow: Either a Workflow object, or a RID for a workflow instance.
        parameters: Either a dictionary or a path to a JSON file that contains configuration parameters for the execution.
        description: A description of the execution.  Can use Markdown format.
    """

    datasets: conlist(DatasetSpec) = []
    assets: list[RID | str] = []  # List of RIDs to model files.
    workflow: RID | Workflow
    parameters: dict[str, Any] | Path = {}
    description: str = ""
    argv: conlist(str) = Field(default_factory=lambda: sys.argv)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, value: Any) -> Any:
        """If a parameter is a file, assume that it has JSON contents for configuration parameters"""
        if isinstance(value, str) or isinstance(value, Path):
            with open(value, "r") as f:
                return json.load(f)
        else:
            return value

    @staticmethod
    def load_configuration(path: Path) -> ExecutionConfiguration:
        """Create a ExecutionConfiguration from a JSON configuration file.

        Args:
          path: File containing JSON version of execution configuration.

        Returns:
          An execution configuration whose values are loaded from the given file.
        """
        with open(path) as fd:
            config = json.load(fd)
        return ExecutionConfiguration.model_validate(config)

    # def download_execution_configuration(
    #     self, configuration_rid: RID
    # ) -> ExecutionConfiguration:
    #     """Create an ExecutionConfiguration object from a catalog RID that points to a JSON representation of that
    #     configuration in hatrac
    #
    #     Args:
    #         configuration_rid: RID that should be to an asset table that refers to an execution configuration
    #
    #     Returns:
    #         A ExecutionConfiguration object for configured by the parameters in the configuration file.
    #     """
    #     AssertionError("Not Implemented")
    #     configuration = self.retrieve_rid(configuration_rid)
    #     with NamedTemporaryFile("w+", delete=False, suffix=".json") as dest_file:
    #         hs = HatracStore("https", self.host_name, self.credential)
    #         hs.get_obj(path=configuration["URL"], destfilename=dest_file.name)
    #         return ExecutionConfiguration.load_configuration(Path(dest_file.name))
