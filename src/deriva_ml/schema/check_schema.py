import json
import re
from importlib.resources import files
from pathlib import Path
from pprint import pprint

from deepdiff import DeepDiff
from deriva.core import AttrDict, BaseCLI, get_credential
from deriva.core.ermrest_catalog import ErmrestCatalog

from deriva_ml.core.definitions import ML_SCHEMA
from deriva_ml.schema.create_schema import create_ml_catalog


def normalize_schema(d):
    if isinstance(d, dict) or isinstance(d, AttrDict):
        m = {}
        for k, v in d.items():
            if k == "acl_bindings" or k == "annotations" or k == "comment":
                continue
            m[k] = normalize_schema(v)
        return m
    elif isinstance(d, list):
        return [normalize_schema(i) for i in d]
    elif isinstance(d, str):
        # ID templates for controlled vocabulary
        if m := re.match("(?P<s>.*):{RID}", d):
            d = d if m["s"] == "deriva-ml" else "reference-catalog:{RID}" if re.match(".*:{RID}", d) else d
        return d
    else:
        return d


def check_ml_schema(hostname, catalog_id, schema_file: Path | None = None):
    """Check the ML schema against a reference schema file.

    Args:
        hostname: The hostname of the Deriva catalog.
        catalog_id: The catalog ID to check.
        schema_file: Optional path to reference schema file. If None, uses default reference.

    Returns:
        None. Prints the diff between target and reference schemas.
    """
    # schema_file = schema_file or files("deriva-ml.data").joinpath("deriva-ml-reference.json")
    schema_file = schema_file or files("deriva_ml.schema").joinpath("deriva-ml-reference.json")

    # Now map

    with Path(schema_file).open("r") as f:
        reference_schema = normalize_schema(json.load(f)["schemas"][ML_SCHEMA])

    catalog = ErmrestCatalog("https", hostname, catalog_id, credentials=get_credential(hostname))
    target_schema = normalize_schema(catalog.getCatalogModel().schemas[ML_SCHEMA].prejson())

    # Compute the diff
    diff = DeepDiff(reference_schema, target_schema, ignore_order=True, view="tree")
    print(f"Diff between {schema_file} and {ML_SCHEMA} schema:")
    # Pretty‐print as JSON
    pprint(diff, indent=2)
    return diff


def dump_ml_schema(hostname: str, filename: str = "deriva-ml-reference.json") -> None:
    """Dump the schema of the ML catalog to stdout."""
    catalog = create_ml_catalog(hostname, "reference-catalog")
    try:
        model = catalog.getCatalogModel()
        print(f"Dumping ML schema to {Path(filename).resolve()}...")
        with Path(filename).open("w") as f:
            json.dump(model.prejson(), f, indent=2)
    finally:
        catalog.delete_ermrest_catalog(really=True)


class CheckMLSchemaCLI(BaseCLI):
    """Main class to part command line arguments and call model"""

    def __init__(self, description, epilog, **kwargs):
        BaseCLI.__init__(self, description, epilog, **kwargs)

        self.parser.add_argument("--catalog", default=1, metavar="<1>", help="Catalog number. Default: 1")
        self.parser.add_argument("--dump", action="store_true", help="Perform execution in dry-run mode.")

    def main(self):
        """Parse arguments and set up execution environment."""
        args = self.parse_cli()
        hostname = args.host
        catalog_id = args.catalog

        if args.dump:
            dump_ml_schema(hostname, catalog_id)
            return

        check_ml_schema(hostname, catalog_id)


def main():
    cli = CheckMLSchemaCLI(description="Check DerivaML Catalog for Compliance", epilog="")
    cli.main()


if __name__ == "__main__":
    main()
