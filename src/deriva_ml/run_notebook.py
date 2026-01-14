"""Module to run a notebook using papermill"""

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
    """Main class to part command line arguments and call model"""

    def __init__(self, description, epilog, **kwargs):
        BaseCLI.__init__(self, description, epilog, **kwargs)
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
    def _coerce_number(val: str):
        """
        Try to convert a string to int, then float; otherwise return str.
        """
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val

    def main(self):
        """Parse arguments and set up execution environment."""
        args = self.parse_cli()
        notebook_file: Path = args.notebook_file
        parameter_file = args.file

        # args.parameter is now a list of [KEY, VALUE] lists
        # e.g. [['timeout', '30'], ['name', 'Alice'], ...]
        parameters = {key: self._coerce_number(val) for key, val in args.parameter}
        parameters['host'] = args.host
        parameters['catalog'] = args.catalog

        if parameter_file:
            with parameter_file.open("r") as f:
                if parameter_file.suffix == ".json":
                    parameters |= json.load(f)
                elif parameter_file.suffix == ".yaml":
                    parameters |= yaml.safe_load(f)
                else:
                    print("Parameter file must be an json or YAML file.")
                    exit(1)

        if not (notebook_file.is_file() and notebook_file.suffix == ".ipynb"):
            print(f"Notebook file must be an ipynb file: {notebook_file.name}.")
            exit(1)

        # Create a workflow instance for this specific version of the script.
        # Return an existing workflow if one is found.
        notebook_parameters = pm.inspect_notebook(notebook_file)

        if args.inspect:
            for param, value in notebook_parameters.items():
                print(f"{param}:{value['inferred_type_name']}  (default {value['default']})")
            return
        else:
            notebook_parameters = {k: v["default"] for k, v in notebook_parameters.items()} | parameters
            self.run_notebook(notebook_file.resolve(), parameters, kernel=args.kernel, log=args.log_output)

    @staticmethod
    def _find_kernel_for_venv() -> str | None:
        """
        Return the name and spec of an existing Jupyter kernel corresponding
        to a given Python virtual environment path.

        Parameters
        ----------
        venv_path : str
            Absolute or relative path to the virtual environment.

        Returns
        -------
        dict | None
            The kernel spec (as a dict) if found, or None if not found.
        """
        venv = os.environ.get("VIRTUAL_ENV")
        if not venv:
            return None
        venv_path = Path(venv).resolve()
        ksm = KernelSpecManager()
        for name, spec in ksm.get_all_specs().items():
            kernel_json = spec.get("spec", {})
            argv = kernel_json.get("argv", [])
            # check for python executable path inside argv
            for arg in argv:
                try:
                    if Path(arg).resolve() == venv_path.joinpath("bin", "python").resolve():
                        return name
                except Exception:
                    continue
        return None

    def run_notebook(self, notebook_file: Path, parameters, kernel=None, log=False):
        url, checksum = Workflow.get_url_and_checksum(Path(notebook_file))
        os.environ["DERIVA_ML_WORKFLOW_URL"] = url
        os.environ["DERIVA_ML_WORKFLOW_CHECKSUM"] = checksum
        os.environ["DERIVA_ML_NOTEBOOK_PATH"] = notebook_file.as_posix()
        with tempfile.TemporaryDirectory() as tmpdirname:
            notebook_output = Path(tmpdirname) / Path(notebook_file).name
            execution_rid_path = Path(tmpdirname) / "execution_rid.json"
            os.environ["DERIVA_ML_SAVE_EXECUTION_RID"] = execution_rid_path.as_posix()

            pm.execute_notebook(
                input_path=notebook_file,
                output_path=notebook_output,
                parameters=parameters,
                kernel_name=kernel,
                log_output=log,
            )
            print(f"Notebook output saved to {notebook_output}")
            with execution_rid_path.open("r") as f:
                execution_config = json.load(f)

            if not execution_config:
                print("Execution RID not found.")
                exit(1)

            execution_rid = execution_config["execution_rid"]
            hostname = execution_config["hostname"]
            catalog_id = execution_config["catalog_id"]
            workflow_rid = execution_config["workflow_rid"]
            ml_instance = DerivaML(hostname=hostname, catalog_id=catalog_id, working_dir=tmpdirname)
            workflow_rid = ml_instance.retrieve_rid(execution_config["execution_rid"])["Workflow"]

            execution = Execution(
                configuration=ExecutionConfiguration(workflow=workflow_rid),
                ml_object=ml_instance,
                reload=execution_rid,
            )

            # Generate an HTML version of the output notebook.
            notebook_output_md = notebook_output.with_suffix(".md")
            with notebook_output.open() as f:
                nb = nbformat.read(f, as_version=4)
            # Convert to Markdown
            exporter = MarkdownExporter()
            (body, resources) = exporter.from_notebook_node(nb)

            with notebook_output_md.open("w") as f:
                f.write(body)
            nb = nbformat.read(notebook_output, as_version=4)

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
            execution.upload_execution_outputs()

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
