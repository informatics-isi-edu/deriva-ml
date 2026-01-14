"""Command-line interface for executing Jupyter notebooks with DerivaML tracking.

This module provides a CLI tool for running Jupyter notebooks using papermill while
automatically tracking the execution in a Deriva catalog. It handles:

- Parameter injection into notebooks from command-line arguments or config files
- Automatic kernel detection for the current virtual environment
- Execution tracking with workflow provenance
- Conversion of executed notebooks to Markdown format
- Upload of notebook outputs as execution assets

The notebook being executed should use DerivaML's execution context to record
its workflow. When run through this CLI, environment variables are set to
communicate workflow metadata (URL, checksum, notebook path) to the notebook.

Environment Variables Set:
    DERIVA_ML_WORKFLOW_URL: URL to the notebook source (e.g., GitHub URL)
    DERIVA_ML_WORKFLOW_CHECKSUM: MD5 checksum of the notebook file
    DERIVA_ML_NOTEBOOK_PATH: Local filesystem path to the notebook
    DERIVA_ML_SAVE_EXECUTION_RID: Path where notebook should save execution info

Usage:
    deriva-ml-run-notebook notebook.ipynb --host example.org --catalog 1
    deriva-ml-run-notebook notebook.ipynb -p param1 value1 -p param2 value2
    deriva-ml-run-notebook notebook.ipynb --file parameters.yaml
    deriva-ml-run-notebook notebook.ipynb --inspect  # Show available parameters

Example:
    # Run a training notebook with parameters
    deriva-ml-run-notebook train_model.ipynb \\
        --host deriva.example.org \\
        --catalog 42 \\
        -p learning_rate 0.001 \\
        -p epochs 100 \\
        --kernel my_ml_env

See Also:
    - install_kernel: Module for installing Jupyter kernels for virtual environments
    - Workflow: Class that handles workflow registration and Git integration
"""

import json
import os
import tempfile
from pathlib import Path

import nbformat
import papermill as pm
import yaml
from deriva.core import BaseCLI
from jupyter_client.kernelspec import KernelSpecManager
from nbconvert import MarkdownExporter

from deriva_ml import DerivaML, ExecAssetType, MLAsset
from deriva_ml.execution import Execution, ExecutionConfiguration, Workflow


class DerivaMLRunNotebookCLI(BaseCLI):
    """Command-line interface for running Jupyter notebooks with DerivaML execution tracking.

    This CLI extends Deriva's BaseCLI to provide notebook execution capabilities using
    papermill. It automatically detects the appropriate Jupyter kernel for the current
    virtual environment and handles parameter injection from multiple sources.

    The CLI supports:
        - Positional notebook file argument
        - Parameter injection via -p/--parameter flags (multiple allowed)
        - Parameter injection via JSON or YAML configuration files
        - Automatic kernel detection for the active virtual environment
        - Inspection mode to display available notebook parameters
        - Logging output from notebook execution

    Attributes:
        parser: ArgumentParser instance with configured arguments.

    Example:
        >>> cli = DerivaMLRunNotebookCLI(
        ...     description="Run ML notebook",
        ...     epilog="See documentation for more details"
        ... )
        >>> cli.main()  # Parses args and runs notebook
    """

    def __init__(self, description: str, epilog: str, **kwargs) -> None:
        """Initialize the notebook runner CLI with command-line arguments.

        Sets up argument parsing for notebook execution, including the notebook file
        path, parameter injection options, kernel selection, and inspection mode.

        Args:
            description: Description text shown in --help output.
            epilog: Additional text shown after argument help.
            **kwargs: Additional keyword arguments passed to BaseCLI.

        Note:
            Calls Workflow._check_nbstrip_status() to verify nbstripout is configured,
            which helps ensure notebooks are properly cleaned before Git commits.
        """
        BaseCLI.__init__(self, description, epilog, **kwargs)
        # Verify nbstripout is configured for clean notebook version control
        Workflow._check_nbstrip_status()
        self.parser.add_argument("notebook_file", type=Path, help="Path to the notebook file")

        self.parser.add_argument(
            "--file",
            "-f",
            type=Path,
            default=None,
            help="JSON or YAML file with parameter values to inject into the notebook.",
        )

        self.parser.add_argument(
            "--catalog",
            type=str,
            default="1",
            help="Catalog number or identifier"
        )

        self.parser.add_argument(
            "--inspect",
            action="store_true",
            help="Display parameters information for the given notebook path.",
        )

        self.parser.add_argument(
            "--log-output",
            action="store_true",
            help="Display logging output from notebook.",
        )

        self.parser.add_argument(
            "--parameter",
            "-p",
            nargs=2,
            action="append",
            metavar=("KEY", "VALUE"),
            default=[],
            help="Provide a parameter name and value to inject into the notebook.",
        )

        self.parser.add_argument(
            "--kernel",
            "-k",
            type=str,
            help="Name of kernel to run..",
            default=self._find_kernel_for_venv(),
        )

    @staticmethod
    def _coerce_number(val: str) -> int | float | str:
        """Convert a string value to the most appropriate numeric type.

        Attempts to parse the string as an integer first, then as a float.
        If neither succeeds, returns the original string unchanged.

        This is used to convert command-line parameter values (which are always
        strings) to appropriate Python types for notebook parameter injection.

        Args:
            val: String value to convert.

        Returns:
            The value as int if it's a valid integer string,
            as float if it's a valid float string,
            or the original string if neither conversion succeeds.

        Examples:
            >>> DerivaMLRunNotebookCLI._coerce_number("42")
            42
            >>> DerivaMLRunNotebookCLI._coerce_number("3.14")
            3.14
            >>> DerivaMLRunNotebookCLI._coerce_number("hello")
            'hello'
        """
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val

    def main(self) -> None:
        """Parse command-line arguments and execute the notebook.

        This is the main entry point that orchestrates:
        1. Parsing command-line arguments
        2. Loading parameters from file if specified
        3. Validating the notebook file
        4. Either inspecting notebook parameters or executing the notebook

        The method merges parameters from multiple sources with the following
        precedence (later sources override earlier):
        1. Notebook default values
        2. Parameters from --file (JSON/YAML)
        3. Parameters from -p/--parameter flags
        4. Host and catalog from CLI arguments

        Raises:
            SystemExit: If parameter file has invalid extension or notebook file
                is invalid.
        """
        args = self.parse_cli()
        notebook_file: Path = args.notebook_file
        parameter_file = args.file

        # Build parameters dict from command-line -p/--parameter flags
        # args.parameter is a list of [KEY, VALUE] lists, e.g. [['timeout', '30'], ...]
        parameters = {key: self._coerce_number(val) for key, val in args.parameter}
        # Always inject host and catalog for DerivaML connection in the notebook
        parameters['host'] = args.host
        parameters['catalog'] = args.catalog

        # Merge parameters from configuration file if provided
        if parameter_file:
            with parameter_file.open("r") as f:
                if parameter_file.suffix == ".json":
                    parameters |= json.load(f)
                elif parameter_file.suffix == ".yaml":
                    parameters |= yaml.safe_load(f)
                else:
                    print("Parameter file must be an json or YAML file.")
                    exit(1)

        # Validate notebook file exists and has correct extension
        if not (notebook_file.is_file() and notebook_file.suffix == ".ipynb"):
            print(f"Notebook file must be an ipynb file: {notebook_file.name}.")
            exit(1)

        # Use papermill to inspect notebook for parameter cell metadata
        notebook_parameters = pm.inspect_notebook(notebook_file)

        if args.inspect:
            # Display parameter info and exit without executing
            for param, value in notebook_parameters.items():
                print(f"{param}:{value['inferred_type_name']}  (default {value['default']})")
            return
        else:
            # Merge notebook defaults with provided parameters and execute
            notebook_parameters = {k: v["default"] for k, v in notebook_parameters.items()} | parameters
            self.run_notebook(notebook_file.resolve(), parameters, kernel=args.kernel, log=args.log_output)

    @staticmethod
    def _find_kernel_for_venv() -> str | None:
        """Find a Jupyter kernel that matches the current virtual environment.

        Searches through all installed Jupyter kernels to find one whose Python
        executable path matches the VIRTUAL_ENV environment variable. This allows
        automatic kernel selection when running notebooks from within an activated
        virtual environment.

        The method examines each kernel's argv configuration to find the Python
        executable path and compares it to the expected location within the
        virtual environment (venv_path/bin/python).

        Returns:
            The kernel name (str) if a matching kernel is found, or None if
            no virtual environment is active or no matching kernel exists.

        Note:
            This method only works on Unix-like systems where Python executables
            are located at bin/python within the virtual environment. For Windows,
            the path would be Scripts/python.exe.

        Example:
            >>> # With VIRTUAL_ENV=/path/to/myenv and kernel 'myenv' installed
            >>> DerivaMLRunNotebookCLI._find_kernel_for_venv()
            'myenv'
        """
        venv = os.environ.get("VIRTUAL_ENV")
        if not venv:
            return None
        venv_path = Path(venv).resolve()
        ksm = KernelSpecManager()
        for name, spec in ksm.get_all_specs().items():
            kernel_json = spec.get("spec", {})
            argv = kernel_json.get("argv", [])
            # Check each argument for the Python executable path
            for arg in argv:
                try:
                    if Path(arg).resolve() == venv_path.joinpath("bin", "python").resolve():
                        return name
                except Exception:
                    continue
        return None

    def run_notebook(
        self,
        notebook_file: Path,
        parameters: dict,
        kernel: str | None = None,
        log: bool = False,
    ) -> None:
        """Execute a notebook with papermill and upload results to the catalog.

        This method handles the complete notebook execution lifecycle:
        1. Sets environment variables for workflow provenance (URL, checksum, path)
        2. Executes the notebook using papermill with injected parameters
        3. Reads execution metadata saved by the notebook
        4. Converts executed notebook to Markdown format
        5. Uploads both notebook outputs as execution assets
        6. Prints a citation for the execution record

        The notebook is expected to create an execution record during its run
        and save the execution metadata to the path specified in the
        DERIVA_ML_SAVE_EXECUTION_RID environment variable.

        Args:
            notebook_file: Absolute path to the notebook file to execute.
            parameters: Dictionary of parameters to inject into the notebook's
                parameter cell.
            kernel: Name of the Jupyter kernel to use. If None, papermill will
                use the notebook's default kernel.
            log: If True, stream notebook cell outputs to stdout during execution.

        Raises:
            SystemExit: If the notebook doesn't save execution metadata.

        Note:
            The executed notebook and its Markdown conversion are uploaded to
            the catalog as Execution_Asset records with type 'notebook_output'.
        """
        # Get workflow provenance info (URL for Git-tracked files, checksum for integrity)
        url, checksum = Workflow.get_url_and_checksum(Path(notebook_file))
        os.environ["DERIVA_ML_WORKFLOW_URL"] = url
        os.environ["DERIVA_ML_WORKFLOW_CHECKSUM"] = checksum
        os.environ["DERIVA_ML_NOTEBOOK_PATH"] = notebook_file.as_posix()

        with tempfile.TemporaryDirectory() as tmpdirname:
            notebook_output = Path(tmpdirname) / Path(notebook_file).name
            execution_rid_path = Path(tmpdirname) / "execution_rid.json"
            # Tell the notebook where to save its execution metadata
            os.environ["DERIVA_ML_SAVE_EXECUTION_RID"] = execution_rid_path.as_posix()

            # Execute the notebook with papermill, injecting parameters
            pm.execute_notebook(
                input_path=notebook_file,
                output_path=notebook_output,
                parameters=parameters,
                kernel_name=kernel,
                log_output=log,
            )
            print(f"Notebook output saved to {notebook_output}")

            # Read execution metadata that the notebook should have saved
            with execution_rid_path.open("r") as f:
                execution_config = json.load(f)

            if not execution_config:
                print("Execution RID not found.")
                exit(1)

            # Extract execution info to reconnect to the catalog
            execution_rid = execution_config["execution_rid"]
            hostname = execution_config["hostname"]
            catalog_id = execution_config["catalog_id"]
            workflow_rid = execution_config["workflow_rid"]

            # Create DerivaML instance to upload results
            ml_instance = DerivaML(hostname=hostname, catalog_id=catalog_id, working_dir=tmpdirname)
            workflow_rid = ml_instance.retrieve_rid(execution_config["execution_rid"])["Workflow"]

            # Restore the execution context to upload outputs
            execution = Execution(
                configuration=ExecutionConfiguration(workflow=workflow_rid),
                ml_object=ml_instance,
                reload=execution_rid,
            )

            # Convert executed notebook to Markdown for easier viewing
            notebook_output_md = notebook_output.with_suffix(".md")
            with notebook_output.open() as f:
                nb = nbformat.read(f, as_version=4)
            exporter = MarkdownExporter()
            (body, resources) = exporter.from_notebook_node(nb)

            with notebook_output_md.open("w") as f:
                f.write(body)
            nb = nbformat.read(notebook_output, as_version=4)

            # Register both notebook outputs as execution assets
            execution.asset_file_path(
                asset_name=MLAsset.execution_asset,
                file_name=notebook_output,
                asset_types=ExecAssetType.notebook_output,
            )

            execution.asset_file_path(
                asset_name=MLAsset.execution_asset,
                file_name=notebook_output_md,
                asset_types=ExecAssetType.notebook_output,
            )

            # Upload all registered assets to the catalog
            execution.upload_execution_outputs()

            # Print citation info for referencing this execution
            print(ml_instance.cite(execution_rid))


def main():
    """Main entry point for the notebook runner CLI.

    Creates and runs the DerivaMLRunNotebookCLI instance.

    Returns:
        None. Executes the CLI.
    """
    cli = DerivaMLRunNotebookCLI(description="Deriva ML Execution Script Demo", epilog="")
    cli.main()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        exit(1)
