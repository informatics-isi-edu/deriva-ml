import argparse
import sys

from deriva.core import DerivaServer, ErmrestCatalog, get_credential
from deriva.core.ermrest_model import Model
from deriva.core.ermrest_model import builtin_types, Schema, Table, Column, ForeignKey, Key
from deriva.core.utils.core_utils import tag as chaise_tags

from deriva_ml import MLVocab
from deriva_ml.schema_setup.annotations import generate_annotation


def define_table_workflow(workflow_annotation: dict):
    return Table.define(
        "Workflow",
        column_defs=[
            Column.define("Name", builtin_types.text),
            Column.define("Description", builtin_types.markdown),
            Column.define("URL", builtin_types.ermrest_uri),
            Column.define("Checksum", builtin_types.text),
            Column.define("Version", builtin_types.text),
        ],
        annotations=workflow_annotation,
    )

def define_table_dataset(dataset_annotation: dict = None):
    return Table.define(
        tname="Dataset",
        column_defs=[
            Column.define("Version", builtin_types.text, default="1.0.0", nullok=True),
            Column.define("Description", builtin_types.markdown)],
        annotations=dataset_annotation if dataset_annotation is not None else {},
    )


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
        fkey_defs=[ForeignKey.define(["Workflow"], sname, "Workflow", ["RID"])],
        annotations=execution_annotation,
    )
    return table_def


def define_asset_execution_metadata(sname: str, execution_metadata_annotation: dict):
    return Table.define_asset(
        sname=sname,
        tname="Execution_Metadata",
        column_defs=[Column.define("Execution", builtin_types.markdown)],
        hatrac_template="/hatrac/metadata/{{MD5}}.{{Filename}}",
        fkey_defs=[ForeignKey.define(["Execution"], sname, "Execution", ["RID"])],
        annotations=execution_metadata_annotation,
    )


def define_asset_execution_asset(sname: str, execution_asset_annotation: dict):
    table_def = Table.define_asset(
        sname=sname,
        tname="Execution_Asset",
        hatrac_template="/hatrac/execution_asset/{{MD5}}.{{Filename}}",
        annotations=execution_asset_annotation,
    )
    return table_def

def create_www_schema(model: Model):
    """
    Set up a new schema and tables to hold web-page like content.  The tables include a page table, and a asset
    table that can have images that are referred to by the web page.  Pages are written using markdown.
    :return:
    """
    if model.schemas.get('www'):
        model.schemas['www'].drop(cascade=True)
    www_schema = model.create_schema(
        Schema.define(
        'www', comment='Schema for tables that will be displayed as web content')
    )
    www_schema.create_table(
        Table.define(
            'Page',
            column_defs=[
                Column.define('Title', builtin_types.text, nullok=False, comment='Unique title for the page'),
                Column.define('Content', builtin_types.markdown, comment='Content of the page in markdown')
            ],
            key_defs=[Key.define(['Title'])],
            annotations={
                chaise_tags.table_display: {'detailed': {'hide_column_headers': True, 'collapse_toc_panel': True}
                                            },
                chaise_tags.visible_foreign_keys: {'detailed': {}},
                chaise_tags.visible_columns: {'detailed': ['Content']}}

        )
    )

    return www_schema

def create_ml_schema(model: Model, schema_name: str = 'deriva-ml', project_name: str = None):
    ml_catalog: ErmrestCatalog = model.catalog

    if model.schemas.get(schema_name):
        model.schemas[schema_name].drop(cascade=True)
    # get annotations
    annotations = generate_annotation(ml_catalog.catalog_id, schema_name)
    model.annotations.update(annotations['catalog_annotation'])
    client_annotation = {
        "tag:misd.isi.edu,2015:display": {"name": "Users"},
        "tag:isrd.isi.edu,2016:table-display": {"row_name": {"row_markdown_pattern": "{{{Full_Name}}}"}},
        "tag:isrd.isi.edu,2016:visible-columns": {"compact": ["Full_Name", "Display_Name", "Email", "ID"]}
    }
    model.schemas['public'].tables['ERMrest_Client'].annotations.update(client_annotation)
    model.apply()

    schema = model.create_schema(Schema.define(schema_name, annotations=annotations['schema_annotation']))
    project_name = project_name or schema_name
    # Workflow
    schema.create_table(Table.define_vocabulary("Feature_Name", f'{project_name}:{{RID}}'))

    workflow_table = schema.create_table(define_table_workflow(annotations["workflow_annotation"]))
    workflow_table.create_reference(schema.create_table(
        Table.define_vocabulary(MLVocab.workflow_type, f'{schema_name}:{{RID}}')))

    execution_table = schema.create_table(define_table_execution(schema_name, annotations["execution_annotation"]))

    dataset_table = schema.create_table(define_table_dataset(annotations["dataset_annotation"]))
    dataset_type = schema.create_table(
        Table.define_vocabulary(MLVocab.dataset_type, f'{project_name}:{{RID}}'))
    schema.create_table(
        Table.define_association(associates=[("Dataset", dataset_table), (MLVocab.dataset_type, dataset_type)])
    )
    schema.create_table(
        Table.define_association(associates=[("Dataset", dataset_table), ("Execution", execution_table)])
    )

    # Nested datasets.
    schema.create_table(
        Table.define_association(associates=[("Dataset", dataset_table), ("Nested_Dataset", dataset_table)])
    )

    # Execution Metadata
    execution_metadata_table = schema.create_table(
        define_asset_execution_metadata(schema.name, annotations["execution_metadata_annotation"]))
    execution_metadata_table.create_reference(
        schema.create_table(
            Table.define_vocabulary("Execution_Metadata_Type", f'{project_name}:{{RID}}')))
    schema.create_table(
        Table.define_association([("Execution_Metadata", execution_metadata_table), ("Execution", execution_table)]))

    # Execution Asset
    execution_asset_table = schema.create_table(
        define_asset_execution_asset(schema.name, annotations["execution_asset_annotation"])
    )
    execution_asset_table.create_reference(
        schema.create_table(
            Table.define_vocabulary("Execution_Asset_Type", f'{project_name}:{{RID}}')))
    schema.create_table(
        Table.define_association([("Execution_Asset", execution_asset_table), ("Execution", execution_table)]))

    create_www_schema(model)
    initialize_ml_schema(model, schema_name)

def initialize_ml_schema(model: Model, schema_name: str = 'deriva-ml'):
    catalog = model.catalog
    execution_metadata_type = catalog.getPathBuilder().schemas[schema_name].tables['Execution_Metadata_Type']
    execution_metadata_type.insert([{'Name': 'Execution_Config',
                                     'Description': "Configuration File for execution metadata"},
                                    {'Name': "Runtime_Env",
                                     'Description': "Information about the execution environment"}],
                                   defaults={'ID', 'URI'})


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


if __name__ == "__main__":
    sys.exit(main())
