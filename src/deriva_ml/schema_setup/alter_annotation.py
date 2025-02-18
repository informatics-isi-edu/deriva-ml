import sys
import argparse
from deriva.core import get_credential, ErmrestCatalog
from deriva.core.ermrest_model import Model

from deriva_ml.schema_setup.annotations import generate_annotation


def alter_table_annotation(
    model_root: Model, schema_name: str, table_name: str, annotation: dict
):
    table = model_root.schemas[schema_name].tables[table_name]
    table.alter(annotations=annotation)
    model_root.apply()


def alter_schema_annotation(model_root: Model, schema_name: str, annotation: dict):
    schema = model_root.schemas[schema_name]
    schema.alter(annotations=annotation)
    model_root.apply()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", type=str, required=True)
    parser.add_argument("--schema_name", type=str, required=True)
    parser.add_argument("--catalog_id", type=str, required=True)
    args = parser.parse_args()
    credentials = get_credential(args.hostname)
    catalog = ErmrestCatalog("https", args.hostname, args.catalog_id, credentials)
    model = catalog.getCatalogModel()
    annotations = generate_annotation(model)
    alter_table_annotation(
        model, args.schema_name, "Workflow", annotations["workflow_annotation"]
    )
    alter_table_annotation(
        model, args.schema_name, "Execution", annotations["execution_annotation"]
    )
    alter_table_annotation(
        model,
        args.schema_name,
        "Execution_Metadata",
        annotations["execution_metadata_annotation"],
    )
    alter_table_annotation(
        model,
        args.schema_name,
        "Execution_Asset",
        annotations["execution_asset_annotation"],
    )
    alter_schema_annotation(model, args.schema_name, annotations["schema_annotation"])


if __name__ == "__main__":
    sys.exit(main())
