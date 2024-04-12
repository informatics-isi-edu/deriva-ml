import sys
import argparse
from deriva.core import DerivaServer, get_credential, ErmrestCatalog
from deriva.core.ermrest_model import builtin_types, Schema, Table, Column, ForeignKey
from annotation_temp import generate_annotation


def alter_table_annotation(catalog, schema_name: str, table_name: str, annotation: dict):
    model_root = catalog.getCatalogModel()
    table = model_root.table(schema_name, table_name)
    table.alter(annotations=annotation)


def alter_schema_annotation(catalog, schema_name: str, annotation: dict):
    model_root = catalog.getCatalogModel()
    schema = model_root.schemas[schema_name]
    schema.alter(annotations=annotation)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hostname', type=str, required=True)
    parser.add_argument('--schema_name', type=str, required=True)
    parser.add_argument('--catalog_id', type=str, required=True)
    args = parser.parse_args()
    credentials = get_credential(args.hostname)
    catalog = ErmrestCatalog('https', args.hostname, args.catalog_id, credentials)

    annotations = generate_annotation(args.schema_name, args.catalog_id)
    alter_table_annotation(catalog, args.schema_name, 'Workflow', annotations["workflow_annotation"])
    alter_table_annotation(catalog, args.schema_name, 'Execution', annotations["execution_annotation"])
    alter_table_annotation(catalog, args.schema_name, 'Execution_Metadata', annotations["execution_metadata_annotation"])
    alter_table_annotation(catalog, args.schema_name, 'Execution_Assets', annotations["execution_assets_annotation"])
    alter_schema_annotation(catalog, args.schema_name, annotations["schema_annotation"])

if __name__ == "__main__":
    sys.exit(main())
