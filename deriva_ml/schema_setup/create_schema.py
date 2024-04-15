import sys

from deriva.core import DerivaServer, get_credential
from deriva.core.ermrest_model import builtin_types, Schema, Table, Column, ForeignKey
from deriva.chisel import Model, Schema, Table, Column, ForeignKey
from deriva_ml.schema_setup.annotation_temp import generate_annotation
import argparse


def create_schema_if_not_exist(model, schema_name, schema_comment=None):
    if schema_name not in model.schemas:
        schema = model.create_schema(Schema.define(schema_name, schema_comment))
        return schema
    else:
        schema = model.schemas[schema_name]
        return schema


def create_table_if_not_exist(schema, table_name, create_spec):
    if table_name not in schema.tables:
        table = schema.create_table(create_spec)
        return table
    else:
        table = schema.tables[table_name]
        return table


def define_table_workflow(workflow_annotation: dict):
    table_def = Table.define(
        'Workflow',
        column_defs=[
            Column.define('Name', builtin_types.text),
            Column.define('Description', builtin_types.markdown),
            Column.define('URL', builtin_types.ermrest_uri),
            Column.define('Checksum', builtin_types.text),
            Column.define('Version', builtin_types.text)
        ],
        fkey_defs=[
            ForeignKey.define(
                ['RCB'],
                'public', 'ERMrest_Client',
                ['ID']
            )
        ],
        annotations=workflow_annotation
    )
    return table_def


def define_table_execution(execution_annotation: dict):
    table_def = Table.define(
        'Execution',
        column_defs=[
            Column.define('Description', builtin_types.markdown),
            Column.define('Duration', builtin_types.text),
            Column.define('Status', builtin_types.text),
            Column.define('Status_Detail', builtin_types.text)
        ],
        fkey_defs=[
            ForeignKey.define(
                ['RCB'],
                'public', 'ERMrest_Client',
                ['ID']
            )
        ],
        annotations=execution_annotation
    )
    return table_def


def define_asset_execution_metadata(schema: str, execution_metadata_annotation: dict):
    table_def = Table.define_asset(
        sname=schema,
        tname='Execution_Metadata',
        hatrac_template='/hatrac/metadata/{{MD5}}.{{Filename}}',
        fkey_defs=[
            ForeignKey.define(
                ['RCB'],
                'public', 'ERMrest_Client',
                ['ID']
            )
        ],
        annotations=execution_metadata_annotation
    )
    return table_def


def define_asset_execution_assets(schema: str, execution_assets_annotation: dict):
    table_def = Table.define_asset(
        sname=schema,
        tname='Execution_Assets',
        hatrac_template='/hatrac/execution_assets/{{MD5}}.{{Filename}}',
        fkey_defs=[
            ForeignKey.define(
                ['RCB'],
                'public', 'ERMrest_Client',
                ['ID']
            )
        ],
        annotations=execution_assets_annotation
    )
    return table_def


def setup_ml_workflow(model, schema_name, catalog_id):
    curie_template = catalog_id+':{RID}'
    schema = create_schema_if_not_exist(model, schema_name)
    # get annotations
    annotations = generate_annotation(catalog_id, schema_name)
    # Workflow
    workflow_table = create_table_if_not_exist(schema, 'Workflow',
                                               define_table_workflow(annotations["workflow_annotation"]))
    table_def_workflow_type_vocab = Table.define_vocabulary(
        tname='Workflow_Type', curie_template=curie_template
    )
    workflow_type_table = schema.create_table(table_def_workflow_type_vocab)
    workflow_table.add_reference(workflow_type_table)

    # Execution
    execution_table = create_table_if_not_exist(schema, 'Execution',
                                                define_table_execution(annotations["execution_annotation"]))
    execution_table.add_reference(workflow_table)
    # dataset_table = create_table_if_not_exist(schema, 'Dataset', define_table_dataset(schema))
    # association_dataset_execution = schema.create_association(dataset_table, execution_table)

    # Execution Metadata
    execution_metadata_table = create_table_if_not_exist(schema, 'Execution_Metadata',
                                                         define_asset_execution_metadata(schema,
                                                                                         annotations["execution_metadata_annotation"]))
    execution_metadata_table.add_reference(execution_table)
    table_def_metadata_type_vocab = Table.define_vocabulary(tname='Execution_Metadata_Type',
                                                            curie_template=curie_template)
    metadata_type_table = schema.create_table(table_def_metadata_type_vocab)
    execution_metadata_table.add_reference(metadata_type_table)

    # Execution Asset
    execution_assets_table = create_table_if_not_exist(schema, 'Execution_Assets',
                                                       define_asset_execution_assets(schema,
                                                                                     annotations["execution_assets_annotation"]))
    association_execution_execution_asset = schema.create_association(execution_assets_table, execution_table)

    table_def_execution_product_type_vocab = Table.define_vocabulary(
        tname='Execution_Asset_Type', curie_template=curie_template
    )
    execution_asset_type_table = schema.create_table(table_def_execution_product_type_vocab)
    execution_assets_table.add_reference(execution_asset_type_table)
    # image_table = create_table_if_not_exist(schema, 'Image', define_asset_image(schema))
    # association_image_execution_asset = schema.create_association(execution_assets_table, image_table)


def main():
    scheme = 'https'
    parser = argparse.ArgumentParser()
    parser.add_argument('--hostname', type=str, required=True)
    parser.add_argument('--schema_name', type=str, required=True)
    parser.add_argument('--catalog_id', type=str, required=True)
    args = parser.parse_args()
    credentials = get_credential(args.hostname)
    server = DerivaServer(scheme, args.hostname, credentials)
    model = Model.from_catalog(server.connect_ermrest(args.catalog_id))
    setup_ml_workflow(model, args.schema_name, args.catalog_id)


if __name__ == "__main__":
    sys.exit(main())
