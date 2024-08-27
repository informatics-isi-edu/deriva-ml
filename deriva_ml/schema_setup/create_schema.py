import argparse
import sys

from deriva.chisel import Model, Schema, Table, Column, ForeignKey, Key
from deriva.core import DerivaServer, get_credential
from deriva.core.ermrest_model import builtin_types, Schema, Table, Column, ForeignKey

from deriva_ml.schema_setup.annotation_temp import generate_annotation


def define_table_workflow(workflow_annotation: dict):
    table_def = Table.define(
        "Workflow",
        column_defs=[
            Column.define("Name", builtin_types.text),
            Column.define("Description", builtin_types.markdown),
            Column.define("URL", builtin_types.ermrest_uri),
            Column.define("Checksum", builtin_types.text),
            Column.define("Version", builtin_types.text),
        ],
        fkey_defs=[
            ForeignKey.define(["RCB"], "public", "ERMrest_Client", ["ID"]),
            ForeignKey.define(["RMB"], "public", "ERMrest_Client", ["ID"]),
        ],
        annotations=workflow_annotation,
    )
    return table_def


def define_table_dataset(dataset_annotation: dict = None):
    table_def = Table.define(
        tname="Dataset",
        column_defs=[Column.define("Description", builtin_types.text)],
        fkey_defs=[
            ForeignKey.define(["RCB"], "public", "ERMrest_Client", ["ID"]),
            ForeignKey.define(["RMB"], "public", "ERMrest_Client", ["ID"]),
        ],
        annotations=dataset_annotation if dataset_annotation is not None else {},
    )
    return table_def


def define_table_execution(execution_annotation: dict):
    table_def = Table.define(
        "Execution",
        column_defs=[
            Column.define("Description", builtin_types.markdown),
            Column.define("Duration", builtin_types.text),
            Column.define("Status", builtin_types.text),
            Column.define("Status_Detail", builtin_types.text),
        ],
        fkey_defs=[
            ForeignKey.define(["RCB"], "public", "ERMrest_Client", ["ID"]),
            ForeignKey.define(["RMB"], "public", "ERMrest_Client", ["ID"]),
        ],
        annotations=execution_annotation,
    )
    return table_def


def define_asset_execution_metadata(schema: str, execution_metadata_annotation: dict):
    table_def = Table.define_asset(
        sname=schema,
        tname="Execution_Metadata",
        hatrac_template="/hatrac/metadata/{{MD5}}.{{Filename}}",
        fkey_defs=[
            ForeignKey.define(["RCB"], "public", "ERMrest_Client", ["ID"]),
            ForeignKey.define(["RMB"], "public", "ERMrest_Client", ["ID"]),
        ],
        annotations=execution_metadata_annotation,
    )
    return table_def


def define_asset_execution_assets(schema: str, execution_assets_annotation: dict):
    table_def = Table.define_asset(
        sname=schema,
        tname="Execution_Assets",
        hatrac_template="/hatrac/execution_assets/{{MD5}}.{{Filename}}",
        fkey_defs=[
            ForeignKey.define(["RCB"], "public", "ERMrest_Client", ["ID"]),
            ForeignKey.define(["RMB"], "public", "ERMrest_Client", ["ID"]),
        ],
        annotations=execution_assets_annotation,
    )
    return table_def


def setup_ml_workflow(model: Model, schema_name: str, catalog_id, curie_prefix):

    if model.schemas.get(schema_name):
        model.schemas[schema_name].drop(cascade=True)
    schema = model.create_schema(Schema.define(schema_name))

    curie_template = curie_prefix + ":{RID}"

    # get annotations
    annotations = generate_annotation(catalog_id, schema_name)
    # Workflow
    table_def_workflow_type_vocab = Table.define_vocabulary(
        tname="Workflow_Type",
        curie_template=curie_template,
        key_defs=[Key.define(["Name"])],
    )

    workflow_table = schema.create_table(
        define_table_workflow(annotations["workflow_annotation"])
    )

    workflow_type_table = schema.tables.get("Workflow_Type") or schema.create_table(
        table_def_workflow_type_vocab
    )
    workflow_table.add_reference(workflow_type_table)

    # Execution
    execution_table = schema.create_table(
        define_table_execution(annotations["execution_annotation"])
    )
    execution_table.add_reference(workflow_table)

    # Dataset
    dataset_table = schema.create_table(
        define_table_dataset(annotations.get("dataset_annotation"))
    )
    association_dataset_execution = schema.create_association(
        dataset_table, execution_table
    )
    table_def_dataset_type_vocab = Table.define_vocabulary(
        tname="Dataset_Type",
        curie_template=curie_template,
        key_defs=[Key.define(["Name"])],
    )
    dataset_type_table = schema.create_table(table_def_dataset_type_vocab)
    dataset_table.add_reference(dataset_type_table)

    # Execution Metadata
    execution_metadata_table = schema.create_table(
        define_asset_execution_metadata(
            schema.name, annotations["execution_metadata_annotation"]
        )
    )
    execution_metadata_table.add_reference(execution_table)
    table_def_metadata_type_vocab = Table.define_vocabulary(
        tname="Execution_Metadata_Type",
        curie_template=curie_template,
        key_defs=[Key.define(["Name"])],
    )
    metadata_type_table = schema.create_table(table_def_metadata_type_vocab)
    execution_metadata_table.add_reference(metadata_type_table)

    # Execution Asset
    execution_assets_table = schema.create_table(
        define_asset_execution_assets(
            schema.name, annotations["execution_assets_annotation"]
        )
    )
    association_execution_execution_asset = schema.create_association(
        execution_assets_table, execution_table
    )

    table_def_execution_product_type_vocab = Table.define_vocabulary(
        tname="Execution_Asset_Type",
        curie_template=curie_template,
        key_defs=[Key.define(["Name"])],
    )
    execution_asset_type_table = schema.create_table(
        table_def_execution_product_type_vocab
    )
    execution_assets_table.add_reference(execution_asset_type_table)

    # Feature Name
    table_def_feature_name_vocab = Table.define_vocabulary(
        tname="Feature_Name",
        curie_template=curie_template,
        key_defs=[Key.define(["Name"])],
    )
    feature_name_vocab_table = schema.create_table(table_def_feature_name_vocab)


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
    model = Model.from_catalog(server.connect_ermrest(args.catalog_id))
    setup_ml_workflow(model, args.schema_name, args.catalog_id, args.curie_prefix)


if __name__ == "__main__":
    sys.exit(main())
