"""Module to run a notebook using papermill"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import papermill as pm
import regex as re
from deriva.core import BaseCLI

from deriva_ml import DerivaML, ExecAssetType, MLAsset, Workflow


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
            help="JSON file with parameter values to inject into the notebook.",
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
            "--catalog",
            metavar="<1>",
            default=1,
            help="Catalog number. Default 1",
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

        self.parser.add_argument("--kernel", "-k", nargs=1, help="Name of kernel to run..", default=None)

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
        notebook_file = args.notebook_file
        parameter_file = args.file

        # args.parameter is now a list of [KEY, VALUE] lists
        # e.g. [['timeout', '30'], ['name', 'Alice'], ...]
        parameters = {key: self._coerce_number(val) for key, val in args.parameter}

        if parameter_file:
            if not (parameter_file.is_file() and parameter_file.suffix == ".json"):
                print("Parameter file must be an json file.")
                exit(1)
            with Path(parameter_file).open("r") as f:
                parameters |= json.load(f)

        if not (notebook_file.is_file() and notebook_file.suffix == ".ipynb"):
            print(f"Notebook file must be an ipynb file: {notebook_file.name}.")
            exit(1)

        os.environ["DERIVA_HOST"] = args.host
        os.environ["DERIVA_CATALOG_ID"] = args.catalog

        # Create a workflow instance for this specific version of the script.
        # Return an existing workflow if one is found.
        notebook_parameters = pm.inspect_notebook(notebook_file)
        if args.inspect:
            for param, value in notebook_parameters.items():
                print(f"{param}:{value['inferred_type_name']}  (default {value['default']})")
            return
        else:
            notebook_parameters = (
                {"host": args.host, "catalog_id": args.catalog, "catalog": args.catalog}
                | {k: v["default"] for k, v in notebook_parameters.items()}
                | parameters
            )
            print(f"Running notebook {notebook_file.name} with parameters:")
            for param, value in notebook_parameters.items():
                print(f"  {param}:{value}")
            self.run_notebook(notebook_file.resolve(), parameters, kernel=args.kernel, log=args.log_output)

    def run_notebook(self, notebook_file, parameters, kernel=None, log=False):
        url, checksum = Workflow.get_url_and_checksum(Path(notebook_file))
        os.environ["DERIVA_ML_WORKFLOW_URL"] = url
        os.environ["DERIVA_ML_WORKFLOW_CHECKSUM"] = checksum

        with tempfile.TemporaryDirectory() as tmpdirname:
            notebook_output = Path(tmpdirname) / Path(notebook_file).name
            pm.execute_notebook(
                input_path=notebook_file,
                output_path=notebook_output,
                parameters=parameters,
                kernel_name=kernel,
                log_output=log,
            )
            catalog_id = execution_rid = None
            with Path(notebook_output).open("r") as f:
                for line in f:
                    if m := re.search(
                        r"Execution RID: https://(?P<host>.*)/id/(?P<catalog_id>.*)/(?P<execution_rid>[\w-]+)",
                        line,
                    ):
                        hostname = m["host"]
                        catalog_id = m["catalog_id"]
                        execution_rid = m["execution_rid"]
            if not execution_rid:
                print("Execution RID not found.")
                exit(1)
            print("Uploaded notebook output for Execution RID:", execution_rid)

            ml_instance = DerivaML(hostname=hostname, catalog_id=catalog_id)

            execution = ml_instance.restore_execution(execution_rid)
            execution.asset_file_path(
                asset_name=MLAsset.execution_asset,
                file_name=notebook_output,
                asset_types=ExecAssetType.notebook_output,
            )
            parameter_file = execution.asset_file_path(
                asset_name=MLAsset.execution_asset,
                file_name=f"notebook-parameters-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json",
                asset_types=ExecAssetType.input_file.value,
            )
            with Path(parameter_file).open("w") as f:
                json.dump(parameters, f)

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
    main()
