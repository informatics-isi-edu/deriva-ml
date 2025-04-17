"""Module to run a notebook using papermill"""

import json
import os
import papermill as pm
from pathlib import Path
import regex as re

from deriva_ml import (
    MLVocab,
    Workflow,
    DerivaML,
)
from deriva.core import BaseCLI

datasets = []
models = []
TEST_SERVER = "dev.eye-ai.org"


class DerivaMLRunNotebookCLI(BaseCLI):
    """Main class to part command line arguments and call model"""

    def __init__(self, description, epilog, **kwargs):
        BaseCLI.__init__(self, description, epilog, **kwargs)

        self.parser.add_argument(
            "notebook_file", type=Path, help="Path to the notebook file"
        )

        self.parser.add_argument(
            "--parameters",
            type=Path, default=None,
            help="JSON file with parameter values to inject into the notebook.",
        )

    def main(self):
        """Parse arguments and set up execution environment."""
        args = self.parse_cli()
        notebook_file = args.notebook_file
        parameter_file = args.parameters

        if not (notebook_file.is_file() and notebook_file.suffix == ".ipynb"):
            print("Notebook file must be an ipynb file.")
            exit(1)

        parameters = {}
        if parameters:
            if not (parameter_file.is_file() and parameter_file.suffix == ".json"):
                print("Parameter file must be an json file.")
                exit(1)
            with open(parameter_file, "r") as f:
                parameters = json.load(f)

        # Create a workflow instance for this specific version of the script.  Return an existing workflow if one is found.
        self.run_notebook(notebook_file, parameters)

    def run_notebook(self, notebook_file, parameters):
        url, checksum = Workflow.get_url_and_checksum(notebook_file)
        os.environ["PAPERMILL_WORKFLOW_URL"] = url
        os.environ["PAPERMILL_WORKFLOW_CHECKSUM"] = checksum

        notebook_output = "foo.ipynb"
        pm.execute_notebook(input_path=notebook_file,
                                output_path=notebook_output,
                                parameters=parameters)
        # look for execution rid in output.
        xecution_rid = re.search("Execution RID: ", notebook_output)
            'asset/Execution_Assets/filename'
            asset_type = ['Notebook_asset']
            upload.asset_file_path()
            upload_path.write_text(notebook_output.name)
            uplost_directory. upload_assets()
)


if __name__ == "__main__":
    cli = DerivaMLRunNotebookCLI(description="Deriva ML Execution Script Demo", epilog="")
    cli.main()
