import argparse
import sys

from deriva.core.ermrest_model import Model,  Key
from deriva.core import DerivaServer, get_credential
from deriva.core.ermrest_model import builtin_types, Schema, Table, Column, ForeignKey

from deriva_ml.schema_setup.annotation_temp import generate_annotation


def define_table_workflow(schema_name: str, workflow_annotation: dict):
    table_def = Table.define(
        "Workflow",
        column_defs=[
            Column.define("Name", builtin_types.text),
            Column.define("Description", builtin_types.markdown),
            Column.define("URL", builtin_types.ermrest_uri),
            Column.define("Checksum", builtin_types.text),
            Column.define("Version", builtin_types.text),
            Column.define("Workflow_Type", builtin_types.text)
        ],
        fkey_defs=[
            ForeignKey.define(["RCB"], "public", "ERMrest_Client", ["ID"]),
            ForeignKey.define(["RMB"], "public", "ERMrest_Client", ["ID"]),
            ForeignKey.define(["Workflow_Type"],
                              schema_name, "Workflow_Type", ["Name"], on_update='CASCADE')
        ],
        annotations=workflow_annotation,
    )
    return table_def


def define_table_dataset(schema_name, dataset_annotation: dict = None):
    table_def = Table.define(
        tname="Dataset",
        column_defs=[
            Column.define("Description", builtin_types.text),
            Column.define("Dataset_Type", builtin_types.text)],
        fkey_defs=[
            ForeignKey.define(["RCB"], "public", "ERMrest_Client", ["ID"]),
            ForeignKey.define(["RMB"], "public", "ERMrest_Client", ["ID"]),
            ForeignKey.define(["Dataset_Type"], schema_name, "Dataset_Type", ["Name"])
        ],
        annotations=dataset_annotation if dataset_annotation is not None else {},
    )
    return table_def


def define_table_execution(sname: str, execution_annotation: dict):
    table_def = Table.define(
        "Execution",
        column_defs=[
            Column.define("Workflow", builtin_types.text),
            Column.define("Description", builtin_types.markdown),
            Column.define("Duration", builtin_types.text),
            Column.define("Status", builtin_types.text),
            Column.define("Status_Detail", builtin_types.text),
        ],
        fkey_defs=[
            ForeignKey.define(["RCB"], "public", "ERMrest_Client", ["ID"]),
            ForeignKey.define(["RMB"], "public", "ERMrest_Client", ["ID"]),
            ForeignKey.define(["Workflow"], sname, "Workflow", ["RID"]),
        ],
        annotations=execution_annotation,
    )
    return table_def


def define_asset_execution_metadata(sname: str, execution_metadata_annotation: dict):
    table_def = Table.define_asset(
        sname=sname,
        tname="Execution_Metadata",
        column_defs=[Column.define("Execution", builtin_types.markdown),
                     Column.define("Execution_Metadata_Type", builtin_types.markdown)],
        hatrac_template="/hatrac/metadata/{{MD5}}.{{Filename}}",
        fkey_defs=[
            ForeignKey.define(["RCB"], "public", "ERMrest_Client", ["ID"]),
            ForeignKey.define(["RMB"], "public", "ERMrest_Client", ["ID"]),
            ForeignKey.define(["Execution"], sname, "Execution", ["RID"]),
            ForeignKey.define(["Execution_Metadata_Type"], sname, "Execution_Metadata_Type", ["Name"])
        ],
        annotations=execution_metadata_annotation,
    )
    return table_def


def define_asset_execution_assets(sname: str, execution_assets_annotation: dict):
    table_def = Table.define_asset(
        sname=sname,
        tname="Execution_Assets",
        hatrac_template="/hatrac/execution_assets/{{MD5}}.{{Filename}}",
        column_defs=[Column.define("Execution_Asset_Type", builtin_types.text)],
        fkey_defs=[
            ForeignKey.define(["RCB"], "public", "ERMrest_Client", ["ID"]),
            ForeignKey.define(["RMB"], "public", "ERMrest_Client", ["ID"]),
            ForeignKey.define(["Execution_Asset_Type"], sname, "Execution_Asset_Type", ["Name"])

        ],
        annotations=execution_assets_annotation,
    )
    return table_def


def setup_ml_workflow(model: Model, schema_name: str, catalog_id, curie_prefix):

    if model.schemas.get(schema_name):
        model.schemas[schema_name].drop(cascade=True)
    schema = model.create_schema(Schema.define(schema_name))

    def create_vocabulary(name):
        curie_template = curie_prefix + ":{RID}"
        schema.create_table(Table.define_vocabulary(
            tname=name, curie_template=curie_template, key_defs=[Key.define(["Name"])]))

    # get annotations
    annotations = generate_annotation(catalog_id, schema_name)
    # Workflow
    for t in ["Dataset_Type", "Workflow_Type", "Execution_Metadata_Type", "Feature_Name", "Execution_Asset_Type"]:
        create_vocabulary(t)

    schema.create_table(define_table_workflow(schema_name, annotations["workflow_annotation"]))
    execution_table = schema.create_table(define_table_execution(schema_name, annotations["execution_annotation"]))
    dataset_table = schema.create_table(define_table_dataset(schema_name, annotations.get("dataset_annotation")))
    schema.create_table(Table.define_association(associates=[dataset_table, execution_table]))

    # Execution Metadata
    schema.create_table(define_asset_execution_metadata(schema.name, annotations["execution_metadata_annotation"]))

    # Execution Asset
    execution_assets_table = schema.create_table(
        define_asset_execution_assets(schema.name, annotations["execution_assets_annotation"])
    )
    schema.create_table(Table.define_association([execution_assets_table, execution_table]))


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
    setup_ml_workflow(model, args.schema_name, args.catalog_id, args.curie_prefix)


if __name__ == "__main__":
    sys.exit(main())
