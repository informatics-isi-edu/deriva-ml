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
    deriva-ml-run-notebook notebook.ipynb --inspect       # papermill: list notebook params
    deriva-ml-run-notebook notebook.ipynb assets=my_assets  # Hydra overrides only

CLI surface — four distinct inspection operations:
    - ``--list-configs`` (deriva-ml): the menu of selectable ``group=value``
      options registered in the hydra-zen store. deriva-ml-specific; Hydra has
      no equivalent.
    - ``--cfg [job|hydra|all]`` (Hydra vocabulary): render the fully *resolved*
      config the notebook would run with, without executing it. ``--cfg job``
      is "see what my overrides resolve to" — cheaper than ``dry_run=true``
      (no dataset bag download).
    - ``--info [config|defaults|defaults-tree|plugins|searchpath|all]`` (Hydra
      vocabulary): Hydra's own info modes (composed config, defaults tree,
      search path, plugins).
    - ``--inspect`` (papermill): list the notebook's parameter-cell parameters.

Unlike ``deriva-ml-run``, this runner never hands argv to Hydra — it drives a
papermill kernel that calls ``hydra_zen.launch(...)``. So Hydra's ``--cfg`` /
``--info`` cannot flow through as argv; instead they are served by resolving the
notebook's config IN THE RUNNER PROCESS and rendering it with Hydra's own
machinery, without executing the notebook (see
``execution.base_config.render_notebook_config``). The Hydra-flag vocabulary is
documented at
https://hydra.cc/docs/advanced/hydra-command-line-flags/ and the override
grammar at https://hydra.cc/docs/advanced/override_grammar/basic/.

Note on ``-p``: on this runner ``-p`` is papermill's ``--parameter`` (notebook
parameter injection). It is NOT Hydra's ``--package`` (which is reachable on the
model runner ``deriva-ml-run``). The asymmetry is inherent: papermill owns ``-p``
here and there is no argv to forward to Hydra anyway.

Example:
    # Run a training notebook with explicit host/catalog
    deriva-ml-run-notebook train_model.ipynb \\
        --host deriva.example.org \\
        --catalog 42 \\
        -p learning_rate 0.001 \\
        --kernel my_ml_env

    # Run using Hydra config defaults (no --host/--catalog needed)
    deriva-ml-run-notebook analysis.ipynb assets=roc_comparison_probabilities

    # Inspect without executing
    deriva-ml-run-notebook analysis.ipynb --list-configs          # menu of group=value
    deriva-ml-run-notebook analysis.ipynb assets=roc_all --cfg job  # resolved config
    deriva-ml-run-notebook analysis.ipynb --info config           # Hydra composed config

See Also:
    - install_kernel: Module for installing Jupyter kernels for virtual environments
    - Workflow: Class that handles workflow registration and Git integration
    - run_model: CLI for running Python model functions (forwards every Hydra flag)
"""

import base64
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import nbformat
import papermill as pm
import yaml
from deriva.core import BaseCLI
from jupyter_client.kernelspec import KernelSpecManager
from nbconvert import MarkdownExporter

from deriva_ml import DerivaML, ExecAssetType, MLAsset
from deriva_ml.cli.hydra_overrides import validate_hydra_overrides
from deriva_ml.cli.show_info import render_config_groups
from deriva_ml.core.constants import DRY_RUN_RID
from deriva_ml.core.enums import ExecMetadataType
from deriva_ml.core.exceptions import DerivaMLDirtyWorkflowError
from deriva_ml.execution import Execution, ExecutionConfiguration, Workflow
from deriva_ml.execution.base_config import (
    _derive_config_name_from_notebook,
    load_configs,
    render_notebook_config,
)


def _html_table_to_markdown(html: str) -> str | None:
    """Convert an HTML DataFrame table to markdown format.

    Parses HTML table elements and converts them to a properly formatted
    markdown table with headers and alignment.

    Args:
        html: HTML string potentially containing a DataFrame table.

    Returns:
        Markdown table string if an HTML table was found, None otherwise.
    """
    # Check if this looks like a pandas DataFrame HTML output
    if "<table" not in html or "dataframe" not in html:
        return None

    try:
        # Extract table content using regex (avoid heavy dependency on BeautifulSoup)
        thead_match = re.search(r"<thead>(.*?)</thead>", html, re.DOTALL)
        tbody_match = re.search(r"<tbody>(.*?)</tbody>", html, re.DOTALL)

        if not thead_match or not tbody_match:
            return None

        thead = thead_match.group(1)
        tbody = tbody_match.group(1)

        # Extract header row(s)
        header_rows = re.findall(r"<tr[^>]*>(.*?)</tr>", thead, re.DOTALL)
        if not header_rows:
            return None

        # For pandas DataFrames with named index:
        # - First row contains: empty <th> + column names
        # - Second row (if exists) contains: index name + empty <th>s
        # We need to use the first row for column names and second row for index name

        first_row = header_rows[0]
        first_headers = re.findall(r"<th[^>]*>(.*?)</th>", first_row, re.DOTALL)
        first_headers = [re.sub(r"<[^>]+>", "", h).strip() for h in first_headers]

        # Check if there's a second header row with an index name
        index_name = ""
        if len(header_rows) > 1:
            second_row = header_rows[1]
            second_headers = re.findall(r"<th[^>]*>(.*?)</th>", second_row, re.DOTALL)
            second_headers = [re.sub(r"<[^>]+>", "", h).strip() for h in second_headers]
            # The index name is typically in the first cell of the second row
            if second_headers and second_headers[0]:
                index_name = second_headers[0]

        # Build final headers: use index name for first column if available
        headers = first_headers.copy()
        if headers and not headers[0] and index_name:
            headers[0] = index_name

        # Extract body rows
        body_rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tbody, re.DOTALL)

        rows = []
        for row_html in body_rows:
            # Get both th (index) and td (data) cells
            cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, re.DOTALL)
            cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
            rows.append(cells)

        if not headers or not rows:
            return None

        # Build markdown table
        # Determine column widths for alignment
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(cell))

        # Format header
        header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
        separator = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"

        # Format rows
        formatted_rows = []
        for row in rows:
            # Pad row if needed
            padded = row + [""] * (len(headers) - len(row))
            formatted = (
                "| "
                + " | ".join(
                    padded[i].ljust(col_widths[i]) if i < len(col_widths) else padded[i] for i in range(len(headers))
                )
                + " |"
            )
            formatted_rows.append(formatted)

        return "\n".join([header_line, separator] + formatted_rows)

    except Exception:
        # If parsing fails, return None to use default behavior
        return None


def _convert_dataframe_outputs(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Convert DataFrame HTML outputs in notebook cells to markdown tables.

    Iterates through all code cells and converts any display_data outputs
    containing DataFrame HTML tables to markdown format for better rendering.

    Args:
        nb: The notebook node to process.

    Returns:
        The modified notebook node with converted outputs.
    """
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue

        new_outputs = []
        for output in cell.get("outputs", []):
            if output.get("output_type") in ("display_data", "execute_result"):
                data = output.get("data", {})
                html = data.get("text/html", "")

                if html and "<table" in html and "dataframe" in html:
                    md_table = _html_table_to_markdown(html)
                    if md_table:
                        # Replace the output with markdown text
                        # Keep the original output type but change the data
                        new_output = output.copy()
                        new_output["data"] = {"text/plain": md_table}
                        new_outputs.append(new_output)
                        continue

            new_outputs.append(output)

        cell["outputs"] = new_outputs

    return nb


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
        >>> cli = DerivaMLRunNotebookCLI(  # doctest: +SKIP
        ...     description="Run ML notebook",
        ...     epilog="See documentation for more details"
        ... )
        >>> cli.main()  # doctest: +SKIP
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
            default=None,
            help="Catalog number or identifier (optional if defined in Hydra config)",
        )

        self.parser.add_argument(
            "--inspect",
            action="store_true",
            help="Display parameters information for the given notebook path.",
        )

        self.parser.add_argument(
            "--list-configs",
            action="store_true",
            help=(
                "List the deriva-ml/hydra-zen config groups and the options "
                "registered in each (the menu of group=value choices). This is "
                "deriva-ml-specific; Hydra has no equivalent. To inspect the "
                "fully resolved config the notebook would use, pass --cfg job; "
                "to see Hydra internals, pass --info config (etc.)."
            ),
        )

        # Hydra's own --info vocabulary. The notebook runner never hands argv to
        # Hydra (it drives a papermill kernel via hydra_zen.launch), so these
        # modes are served by resolving the config IN THIS PROCESS and rendering
        # it with Hydra's own machinery, without executing the notebook.
        self.parser.add_argument(
            "--info",
            nargs="?",
            const="all",
            choices=["all", "config", "defaults", "defaults-tree", "plugins", "searchpath"],
            default=None,
            help=(
                "Render Hydra's --info for the resolved notebook config without "
                "executing the notebook. Bare --info defaults to 'all'. "
                "See https://hydra.cc/docs/advanced/hydra-command-line-flags/."
            ),
        )

        self.parser.add_argument(
            "--cfg",
            nargs="?",
            const="job",
            choices=["job", "hydra", "all"],
            default=None,
            help=(
                "Render the resolved notebook config (Hydra's --cfg) without "
                "executing the notebook. Bare --cfg defaults to 'job' (the "
                "composed job config the notebook would run with)."
            ),
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

        self.parser.add_argument(
            "--allow-dirty",
            action="store_true",
            help="Allow execution with uncommitted changes (skips git clean check).",
        )

        self.parser.add_argument(
            "hydra_overrides",
            nargs="*",
            help="Hydra-zen configuration overrides (e.g., assets=roc_quick_probabilities)",
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
        2. Validating Hydra overrides and short-circuiting on inspection flags
           (``--list-configs`` menu, ``--cfg``/``--info`` resolved-config render)
        3. Loading parameters from file if specified
        4. Validating the notebook file
        5. Either inspecting notebook parameters (``--inspect``) or executing the
           notebook

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

        # Pre-validate Hydra overrides before any work happens, so bare
        # positional args (e.g. 'roc_analysis' instead of
        # 'assets=roc_analysis') produce a diagnostic error rather than the
        # cryptic ANTLR "missing EQUAL at '<EOF>'" from Hydra's parser.
        try:
            validate_hydra_overrides(args.hydra_overrides, cli_name="deriva-ml-run-notebook")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            exit(1)

        # Inspection flags short-circuit BEFORE any notebook validation or
        # parameter parsing — they neither read the notebook's parameter cell
        # nor execute it. --list-configs is the deriva-ml group menu; --info /
        # --cfg resolve the notebook's config in this process and render it the
        # way Hydra would, without running the notebook (see _show_resolved_config).
        if args.list_configs:
            self._show_hydra_info(notebook_file)
            return

        if args.info is not None or args.cfg is not None:
            self._show_resolved_config(
                notebook_file,
                overrides=args.hydra_overrides,
                cfg_mode=args.cfg,
                info_mode=args.info,
            )
            return

        # Build parameters dict from command-line -p/--parameter flags
        # args.parameter is a list of [KEY, VALUE] lists, e.g. [['timeout', '30'], ...]
        parameters = {key: self._coerce_number(val) for key, val in args.parameter}
        # Inject host and catalog if provided on command line
        # If not provided, the notebook will use values from Hydra config
        if args.host:
            parameters["host"] = args.host
        if args.catalog:
            parameters["catalog"] = args.catalog

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

        # Determine allow-dirty from CLI flag or environment variable
        allow_dirty = args.allow_dirty or os.environ.get("DERIVA_ML_ALLOW_DIRTY", "").lower() == "true"
        if allow_dirty:
            os.environ["DERIVA_ML_ALLOW_DIRTY"] = "true"

        try:
            # Merge notebook defaults with provided parameters and execute
            notebook_parameters = {k: v["default"] for k, v in notebook_parameters.items()} | parameters
            self.run_notebook(
                notebook_file.resolve(),
                parameters,
                kernel=args.kernel,
                log=args.log_output,
                hydra_overrides=args.hydra_overrides,
                allow_dirty=allow_dirty,
            )
        except DerivaMLDirtyWorkflowError as e:
            print(f"Error: {e}", file=sys.stderr)
            exit(1)
        finally:
            os.environ.pop("DERIVA_ML_ALLOW_DIRTY", None)

    @staticmethod
    def _load_project_configs(notebook_file: Path) -> bool:
        """Import the project's config package so the hydra-zen store is populated.

        Both the ``--list-configs`` menu and the ``--info`` / ``--cfg`` config
        resolution need the project's configs registered in the hydra-zen store.
        The notebook lives in ``<project>/notebooks/``; its ``src/`` sibling holds
        the ``configs`` package. This adds ``src/`` to ``sys.path`` and imports
        every config module (which registers configs as an import side effect).

        Args:
            notebook_file: Path to the notebook file, used to locate the project
                root (assumed to be the notebook's grandparent directory).

        Returns:
            True if the ``configs`` package was found and loaded, False if it
            could not be imported (the caller prints a diagnostic and returns).

        Example:
            >>> from pathlib import Path  # doctest: +SKIP
            >>> DerivaMLRunNotebookCLI._load_project_configs(  # doctest: +SKIP
            ...     Path("notebooks/roc_analysis.ipynb")
            ... )
            True
        """
        # Add src directory to path so we can import configs.
        notebook_dir = notebook_file.parent.resolve()
        project_root = notebook_dir.parent  # Assume notebooks/ is one level down
        src_dir = project_root / "src"
        if src_dir.exists() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        # Try the new API, then the legacy load_all_configs() entry point.
        try:
            loaded = load_configs("configs")
            if not loaded:
                from configs import load_all_configs

                load_all_configs()
        except ImportError:
            return False
        return True

    @staticmethod
    def _show_hydra_info(notebook_file: Path) -> None:
        """Print the deriva-ml/hydra-zen config-group menu (``--list-configs``).

        Loads the project's configs (so the store is populated) and delegates to
        the shared :func:`deriva_ml.cli.show_info.render_config_groups`, so this
        runner and ``deriva-ml-run`` render the same listing. The notebook runner
        has no multirun surface, so the multirun section is omitted.

        Args:
            notebook_file: Path to the notebook file (used to find the project
                root and its ``src/configs`` package).
        """
        if not DerivaMLRunNotebookCLI._load_project_configs(notebook_file):
            print("Could not import configs module. Make sure src/configs/__init__.py exists.")
            print("Available Hydra groups cannot be determined without loading the config module.")
            return
        print(render_config_groups(include_multirun=False))

    @staticmethod
    def _show_resolved_config(
        notebook_file: Path,
        *,
        overrides: list[str],
        cfg_mode: str | None,
        info_mode: str | None,
    ) -> None:
        """Render the notebook's resolved config (``--cfg`` / ``--info``).

        The notebook runner never hands argv to Hydra; it drives a papermill
        kernel that calls :func:`hydra_zen.launch`. So Hydra's ``--cfg`` /
        ``--info`` cannot flow through as argv. Instead this resolves the
        notebook's config **in the runner process** and renders it with Hydra's
        own ``show_cfg`` / ``show_info`` machinery — byte-for-byte what Hydra's
        CLI would print — **without executing the notebook** and **without
        touching a live catalog** (pure config composition; see
        :func:`deriva_ml.execution.base_config.render_notebook_config`).

        The config name follows the DerivaML convention ``X.ipynb`` ↔ config
        ``X``: it is derived from the notebook filename via
        ``DERIVA_ML_NOTEBOOK_PATH`` + ``_derive_config_name_from_notebook()``.

        Args:
            notebook_file: Path to the notebook whose config to resolve.
            overrides: Hydra override strings to apply (the positional
                ``hydra_overrides``).
            cfg_mode: ``--cfg`` mode (``job`` / ``hydra`` / ``all``) or None.
            info_mode: ``--info`` mode (``all`` / ``config`` / ``defaults`` /
                ``defaults-tree`` / ``plugins`` / ``searchpath``) or None.
        """
        if not DerivaMLRunNotebookCLI._load_project_configs(notebook_file):
            print("Could not import configs module. Make sure src/configs/__init__.py exists.")
            print("The resolved config cannot be rendered without loading the config module.")
            return

        # Set DERIVA_ML_NOTEBOOK_PATH so the config name is derived from the
        # notebook filename via the same convention run_notebook() uses.
        os.environ["DERIVA_ML_NOTEBOOK_PATH"] = notebook_file.resolve().as_posix()
        config_name = _derive_config_name_from_notebook()
        print(
            render_notebook_config(
                config_name,
                overrides=overrides,
                cfg_mode=cfg_mode,
                info_mode=info_mode,
            )
        )

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
            >>> DerivaMLRunNotebookCLI._find_kernel_for_venv()  # doctest: +SKIP
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
            # Check if the kernel's Python executable lives inside our venv.
            # We compare the unresolved path prefix rather than resolving symlinks,
            # because uv-managed venvs all symlink to the same Python binary
            # and would otherwise all match each other.
            for arg in argv:
                try:
                    arg_path = Path(arg)
                    if arg_path.is_relative_to(venv_path):
                        return name
                except (TypeError, ValueError):
                    continue
        return None

    def run_notebook(
        self,
        notebook_file: Path,
        parameters: dict,
        kernel: str | None = None,
        log: bool = False,
        hydra_overrides: list[str] | None = None,
        allow_dirty: bool = False,
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
            hydra_overrides: Optional list of Hydra-zen configuration overrides
                (e.g., ["assets=roc_quick_probabilities", "deriva_ml=eye_ai"]).
                These are passed to the notebook via DERIVA_ML_HYDRA_OVERRIDES
                environment variable as a JSON-encoded list.

        Raises:
            SystemExit: If the notebook doesn't save execution metadata.

        Note:
            The executed notebook and its Markdown conversion are uploaded to
            the catalog as Execution_Asset records with type 'notebook_output'.
        """
        # Get workflow provenance info (URL for Git-tracked files, checksum for integrity)
        url, checksum = Workflow.get_url_and_checksum(Path(notebook_file), allow_dirty=allow_dirty)
        os.environ["DERIVA_ML_WORKFLOW_URL"] = url
        os.environ["DERIVA_ML_WORKFLOW_CHECKSUM"] = checksum
        os.environ["DERIVA_ML_NOTEBOOK_PATH"] = notebook_file.as_posix()

        # Pass Hydra overrides to notebook via environment variable
        if hydra_overrides:
            os.environ["DERIVA_ML_HYDRA_OVERRIDES"] = json.dumps(hydra_overrides)
        elif "DERIVA_ML_HYDRA_OVERRIDES" in os.environ:
            del os.environ["DERIVA_ML_HYDRA_OVERRIDES"]

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

            # In dry run mode, skip restoring execution and uploading outputs
            if execution_rid == DRY_RUN_RID:
                print(f"Dry run mode: notebook executed successfully. Output saved to {notebook_output}")
                return

            # Create DerivaML instance to upload results.
            # Use the default working directory (not tmpdirname) so that we share
            # the same asset staging area as the notebook subprocess. The notebook
            # registers output files via asset_file_path() into the default working
            # dir; if we used a different working_dir here, commit_output_assets()
            # would only see the .ipynb/.md files we register below, missing any
            # assets the notebook itself created (e.g., training_curves.png).
            ml_instance = DerivaML(hostname=hostname, catalog_id=catalog_id)
            workflow_rid = ml_instance._retrieve_rid(execution_config["execution_rid"])["Workflow"]

            # Look up the workflow object from the RID
            workflow = ml_instance.lookup_workflow(workflow_rid)

            # Restore the execution context to upload outputs
            execution = Execution(
                configuration=ExecutionConfiguration(workflow=workflow),
                ml_object=ml_instance,
                reload=execution_rid,
            )

            # Convert executed notebook to Markdown for easier viewing
            # We embed images as base64 data URIs so the markdown is self-contained
            notebook_output_md = notebook_output.with_suffix(".md")
            with notebook_output.open() as f:
                nb = nbformat.read(f, as_version=4)

            # Convert DataFrame HTML outputs to markdown tables for better rendering
            nb = _convert_dataframe_outputs(nb)

            exporter = MarkdownExporter()
            (body, resources) = exporter.from_notebook_node(nb)

            # Replace file references with inline base64 data URIs
            if resources.get("outputs"):
                for filename, data in resources["outputs"].items():
                    # Determine mime type from extension
                    if filename.endswith(".png"):
                        mime_type = "image/png"
                    elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
                        mime_type = "image/jpeg"
                    elif filename.endswith(".svg"):
                        mime_type = "image/svg+xml"
                    else:
                        mime_type = "application/octet-stream"

                    # Create data URI and replace in markdown
                    b64_data = base64.b64encode(data).decode("utf-8")
                    data_uri = f"data:{mime_type};base64,{b64_data}"
                    body = body.replace(filename, data_uri)

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

            # Register Hydra's job log as a runner-owned Execution_Metadata
            # asset. The kernel cannot register this safely — Hydra's
            # FileHandler keeps appending to the file during the kernel's
            # own upload pass, and the asset model assumes immutable bytes.
            # By the time we reach this point, papermill has returned and
            # the kernel subprocess has exited, so the FileHandler is
            # guaranteed gone and the file is final. See state-machine
            # docs for the additive-upload semantics this depends on
            # (Uploaded → Pending_Upload).
            hydra_dir_str = execution_config.get("hydra_runtime_output_dir")
            if hydra_dir_str:
                hydra_dir = Path(hydra_dir_str)
                notebook_log = hydra_dir / "notebook.log"
                if notebook_log.exists():
                    timestamp = hydra_dir.parts[-1]
                    execution.asset_file_path(
                        asset_name=MLAsset.execution_metadata,
                        file_name=notebook_log,
                        rename_file=f"hydra-{timestamp}-notebook.log",
                        asset_types=ExecMetadataType.hydra_config.value,
                        description=(
                            "Hydra job log capturing the papermill kernel run "
                            "(stdout/stderr from every logger active during the cells)."
                        ),
                    )

            # Commit all registered assets to the catalog. If the kernel
            # already called commit_output_assets() from its last
            # cell, status is Uploaded; this call cycles through
            # Pending_Upload to ship the runner-owned assets only.
            execution.commit_output_assets()

            # Print execution URL (without snapshot ID for readability)
            print(f"https://{hostname}/id/{catalog_id}/{execution_rid}")


def main():
    """Main entry point for the notebook runner CLI.

    Creates and runs the DerivaMLRunNotebookCLI instance.

    Returns:
        None. Executes the CLI.
    """
    cli = DerivaMLRunNotebookCLI(
        description="Run Jupyter notebooks with DerivaML execution tracking",
        epilog=(
            "Examples:\n"
            "  deriva-ml-run-notebook analysis.ipynb --host localhost --catalog 45\n"
            "  deriva-ml-run-notebook analysis.ipynb assets=roc_comparison_probabilities\n"
            "  deriva-ml-run-notebook analysis.ipynb -p learning_rate 0.001\n"
            "  deriva-ml-run-notebook analysis.ipynb --inspect          # papermill: list notebook params\n"
            "  deriva-ml-run-notebook analysis.ipynb --list-configs     # menu of group=value options\n"
            "  deriva-ml-run-notebook analysis.ipynb assets=roc_all --cfg job  # show the resolved config\n"
            "  deriva-ml-run-notebook analysis.ipynb --info config      # Hydra's composed config\n"
            "On this runner -p is papermill's --parameter, NOT Hydra's --package.\n"
            "Hydra's --cfg/--info vocabulary (rendered without running the notebook):\n"
            "  https://hydra.cc/docs/advanced/hydra-command-line-flags/\n"
        ),
    )
    cli.main()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        exit(1)
