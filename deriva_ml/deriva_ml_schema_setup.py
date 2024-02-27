from deriva.core import DerivaServer, get_credential
from deriva.core.ermrest_model import builtin_types, Schema, Table, Column, ForeignKey
from deriva.chisel import Model, Schema, Table, Column, ForeignKey
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


def define_table_workflow():
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
        ]
    )
    return table_def


def define_table_execution():
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
        ]
    )
    return table_def


def define_asset_execution_metadata(schema):
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
        ]
    )
    return table_def


def define_asset_execution_assets(schema):
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
        ]
    )
    return table_def


def setup_ml_workflow(model, schema_name):
    schema = create_schema_if_not_exist(model, schema_name)
    # Workflow
    workflow_table = create_table_if_not_exist(schema, 'Workflow', define_table_workflow())
    table_def_workflow_type_vocab = Table.define_vocabulary(
        tname='Workflow_Type', curie_template='eye-ai:{RID}'
    )
    workflow_type_table = schema.create_table(table_def_workflow_type_vocab)
    workflow_table.add_reference(workflow_type_table)

    # Execution
    execution_table = create_table_if_not_exist(schema, 'Execution', define_table_execution())
    execution_table.add_reference(workflow_table)
    # dataset_table = create_table_if_not_exist(schema, 'Dataset', define_table_dataset(schema))
    # association_dataset_execution = schema.create_association(dataset_table, execution_table)

    # Execution Metadata
    execution_metadata_table = create_table_if_not_exist(schema, 'Execution_Metadata',
                                                         define_asset_execution_metadata(schema))
    execution_metadata_table.add_reference(execution_table)
    table_def_metadata_type_vocab = Table.define_vocabulary(tname='Execution_Metadata_Type',
                                                            curie_template='eye-ai:{RID}')
    metadata_type_table = schema.create_table(table_def_metadata_type_vocab)
    execution_metadata_table.add_reference(metadata_type_table)

    # Execution Asset
    execution_assets_table = create_table_if_not_exist(schema, 'Execution_Assets',
                                                       define_asset_execution_assets(schema))
    association_execution_execution_asset = schema.create_association(execution_assets_table, execution_table)

    table_def_execution_product_type_vocab = Table.define_vocabulary(
        tname='Execution_Product_Type', curie_template='eye-ai:{RID}'
    )
    execution_asset_type_table = schema.create_table(table_def_execution_product_type_vocab)
    execution_assets_table.add_reference(execution_asset_type_table)
    # image_table = create_table_if_not_exist(schema, 'Image', define_asset_image(schema))
    # association_image_execution_asset = schema.create_association(execution_assets_table, image_table)


def main(hostname, catalog_id, credentials, schema_name):
    model = Model.from_catalog(DerivaServer('https', hostname, credentials.connect_ermrest(catalog_id)))
    setup_ml_workflow(model, schema_name)


if __name__ == "__main__":
    scheme = 'https'
    parser = argparse.ArgumentParser()
    parser.add_argument('--hostname', type=str, required=True)
    parser.add_argument('--schema_name', type=str, required=True)
    parser.add_argument('--catalog_id', type=str, required=True)
    args = parser.parse_args()
    credentials = get_credential(args.hostname)
    print(credentials)
    main(args.hostname, args.catalog_id, credentials, args.schema_name)
