import json

from deepdiff import DeepDiff
from deriva.core import BaseCLI, DerivaServer, get_credential
from deriva.core.ermrest_catalog import ErmrestCatalog

from deriva_ml.core.definitions import ML_SCHEMA


def check_ml_schema(hostname, catalog_id, schema_file):
    catalog = ErmrestCatalog("https", hostname, catalog_id, credentials=get_credential(hostname))
    target_schema = catalog.getCatalogModel().schemas[ML_SCHEMA].prejson()["schemas"][ML_SCHEMA]

    with open(schema_file, "r") as f:
        reference_schema = json.load(f)["schemas"][ML_SCHEMA]

    # Compute the diff
    diff = DeepDiff(reference_schema, target_schema, ignore_order=True)

    # Pretty‐print as JSON
    print(json.dumps(diff, indent=2))


def dump_ml_schema(hostname: str, catalog_id: str | int, filename: str) -> None:
    """Dump the schema of the ML catalog to stdout."""
    catalog = ErmrestCatalog("https", hostname, catalog_id=catalog_id, credentials=get_credential(hostname))
    model = catalog.getCatalogModel()
    with open(filename, "w") as f:
        json.dump(model.prejson(), f, indent=2)


def main():
    scheme = "https"
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", type=str, required=True)
    parser.add_argument("--schema_name", type=str, required=True)
    parser.add_argument("--catalog_id", type=str, required=True)
    parser.add_argument("--curie_prefix", type=str, required=True)
    args = parser.parse_args()
    credentials = get_credential(args.hostname)
    server = DerivaServer(scheme, args.hostname, credentials)
    model = server.connect_ermrest(args.catalog_id).getCatalogModel()
    create_ml_schema(model, args.schema_name)


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

        # Create a workflow instance for this specific version of the script.  Return an existing workflow if one is found.
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

    def do_stuff(self, execution: Execution):
        print(f" Execution with parameters: {execution.parameters}")
        print(f" Execution with input assets: {[a.as_posix() for a in execution.asset_paths]}")
        print(f"Execution datasets: {execution.datasets}")


if __name__ == "__main__":
    cli = CheckMLSchemaCLI(description="Deriva ML Execution Script Demo", epilog="")
    cli.main()
