"""Module to run a notebook using papermill"""

import json
import os
import papermill as pm
from pathlib import Path
import regex as re
import tempfile

from deriva_ml import Workflow, DerivaML, MLVocab
from deriva.core import BaseCLI
from deriva_ml import MLAsset


class DerivaMLRunNotebookCLI(BaseCLI):
    """Main class to part command line arguments and call model"""

    def __init__(self, description, epilog, **kwargs):
        BaseCLI.__init__(self, description, epilog, **kwargs)
        Workflow._check_nbstrip_status()
        self.parser.add_argument(
            "notebook_file", type=Path, help="Path to the notebook file"
        )

        self.parser.add_argument(
            "--file",
            type=Path,
            default=None,
            help="JSON file with parameter values to inject into the notebook.",
        )

        self.parser.add_argument(
            "--inspect",
            action="store_true",
            help="Enable inspect mode (sets inspect=True when present).",
        )

        self.parser.add_argument(
            "--parameter",
            "-p",
            nargs=2,
            action="append",
            metavar=("KEY", "VALUE"),
            default=[],
            help="Provide a parameter name band value to inject into the notebook.",
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
        notebook_file = args.notebook_file
        parameter_file = args.file

        # args.parameter is now a list of [KEY, VALUE] lists
        # e.g. [['timeout', '30'], ['name', 'Alice'], ...]
        parameters = {key: self._coerce_number(val) for key, val in args.parameter}

        if parameter_file:
            if not (parameter_file.is_file() and parameter_file.suffix == ".json"):
                print("Parameter file must be an json file.")
                exit(1)
            with open(parameter_file, "r") as f:
                parameters |= json.load(f)

        if not (notebook_file.is_file() and notebook_file.suffix == ".ipynb"):
            print("Notebook file must be an ipynb file.")
            exit(1)

        # Create a workflow instance for this specific version of the script.  Return an existing workflow if one is found.
        if args.inspect:
            for param, value in pm.inspect_notebook(notebook_file).items():
                print(
                    f"{param}:{value['inferred_type_name']}  (default {value['default']})"
                )
            return
        else:
            self.run_notebook(notebook_file, parameters)

    def run_notebook(self, notebook_file, parameters):
        url, checksum = Workflow.get_url_and_checksum(Path(notebook_file))
        os.environ["DERIVA_ML_WORKFLOW_URL"] = url
        os.environ["DERIVA_ML_WORKFLOW_CHECKSUM"] = checksum

        with tempfile.TemporaryDirectory() as tmpdirname:
            notebook_output = Path(tmpdirname) / Path(notebook_file).name
            pm.execute_notebook(
                input_path=notebook_file,
                output_path=notebook_output,
                parameters=parameters,
            )
            host = catalog_id = execution_rid = None
            with open(notebook_output, "r") as f:
                for line in f:
                    if m := re.search(
                        r"Execution RID: https://(?P<host>.*)/id/(?P<catalog_id>.*)/(?P<execution_rid>[\w-]+)",
                        line,
                    ):
                        host = m["host"]
                        catalog_id = m["catalog_id"]
                        execution_rid = m["execution_rid"]
            if not execution_rid:
                print("Execution RID not found.")
                exit(1)
            print("Execution RID:", execution_rid)
            print("Host:", host)
            print("Catalog ID:", catalog_id)

            ml_instance = DerivaML(hostname=host, catalog_id=catalog_id)
            ml_instance.add_term(
                MLVocab.asset_type,
                "Notebook_Output",
                description="Jupyter Notebook Output",
            )
            execution = ml_instance.restore_execution(execution_rid)
            execution.asset_file_path(
                asset_name=MLAsset.execution_asset,
                file_name=notebook_output,
                asset_types=["Notebook_Output"],
            )
            execution.upload_execution_outputs()


if __name__ == "__main__":
    cli = DerivaMLRunNotebookCLI(
        description="Deriva ML Execution Script Demo", epilog=""
    )
    cli.main()
