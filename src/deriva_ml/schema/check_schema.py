import json
import logging
from importlib.resources import files
from pathlib import Path
from pprint import pprint
from typing import Optional

from deepdiff import DeepDiff
from deriva.core import BaseCLI, get_credential
from deriva.core.ermrest_catalog import ErmrestCatalog

from deriva_ml import DerivaML
from deriva_ml.core.definitions import ML_SCHEMA, MLVocab
from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.execution.execution import Execution
from deriva_ml.execution.execution_configuration import ExecutionConfiguration


def check_ml_schema(hostname, catalog_id, schema_file: Path | None = None):
    """Check the ML schema against a reference schema file.
    
    Args:
        hostname: The hostname of the Deriva catalog.
        catalog_id: The catalog ID to check.
        schema_file: Optional path to reference schema file. If None, uses default reference.
        
    Returns:
        None. Prints the diff between target and reference schemas.
    """
    schema_file = schema_file or files("deriva-ml.data").joinpath("deriva-ml-reference.json")

    catalog = ErmrestCatalog("https", hostname, catalog_id, credentials=get_credential(hostname))

    # Need to get rid of attribute dicts....
    target_schema = json.loads(json.dumps(catalog.getCatalogModel().schemas[ML_SCHEMA].prejson()))

    with Path(schema_file).open("r") as f:
        reference_schema = json.load(f)["schemas"][ML_SCHEMA]

    # Compute the diff
    diff = DeepDiff(reference_schema, target_schema, ignore_order=True, view="tree")
    print(f"Diff between {schema_file} and {ML_SCHEMA} schema:")
    # Pretty‐print as JSON
    pprint(diff, indent=2)


def dump_ml_schema(hostname: str, catalog_id: str | int, filename: str) -> None:
    """Dump the schema of the ML catalog to stdout."""
    catalog = ErmrestCatalog("https", hostname, catalog_id=catalog_id, credentials=get_credential(hostname))
    model = catalog.getCatalogModel()
    print(f"Dumping ML schema to {Path(filename).resolve()}...")
    with Path(filename).open("w") as f:
        json.dump(model.prejson(), f, indent=2)


class CheckMLSchemaCLI(BaseCLI):
    """Main class to part command line arguments and call model"""

    def __init__(self, description, epilog, **kwargs):
        BaseCLI.__init__(self, description, epilog, **kwargs)

        self.parser.add_argument("--catalog", default=1, metavar="<1>", help="Catalog number. Default: 1")
        self.parser.add_argument("--test", action="store_true", help="Use demo catalog.")
        self.parser.add_argument("--dry-run", action="store_true", help="Perform execution in dry-run mode.")

        self.execution: Optional[Execution] = None
        self.deriva_ml: Optional[DerivaML] = None
        self.logger = logging.getLogger(__name__)

    def main(self):
        """Parse arguments and set up execution environment."""
        args = self.parse_cli()
        hostname = args.host
        catalog_id = args.catalog

        self.deriva_ml = DerivaML(hostname, catalog_id)  # This should be changed to the domain specific class.
        print(f"Executing script {self.deriva_ml.executable_path} version: {self.deriva_ml.get_version()}")

        # Create a workflow instance for this specific version of the script.
        # Return an existing workflow if one is found.
        self.deriva_ml.add_term(MLVocab.workflow_type, "Demo Notebook", description="Initial setup of Model Notebook")
        workflow = self.deriva_ml.create_workflow("demo-workflow", "Demo Notebook")

        # Create an execution instance that will work with the latest version of the input datasets.
        config = ExecutionConfiguration(
            datasets=[
                DatasetSpec(rid=dataset, version=self.deriva_ml.dataset_version(dataset)) for dataset in datasets
            ],
            assets=models,
            workflow=workflow,
            parameters="parameters..json",
        )

        self.execution = self.deriva_ml.create_execution(config, dry_run=args.dry_run)
        with self.execution as e:
            self.do_stuff(e)
        self.execution.upload_execution_outputs()


if __name__ == "__main__":
    cli = CheckMLSchemaCLI(description="Deriva ML Execution Script Demo", epilog="")
    cli.main()
