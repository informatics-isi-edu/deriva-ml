import argparse
import subprocess
import sys
from importlib.resources import files
from typing import Any, Optional

from deriva.core import DerivaServer, ErmrestCatalog, get_credential
from deriva.core.datapath import DataPathException
from deriva.core.ermrest_model import (
    Column,
    ForeignKey,
    Key,
    Model,
    Schema,
    Table,
    builtin_types,
)
from requests.exceptions import HTTPError

from deriva_ml.core.definitions import ML_SCHEMA, MLVocab
from deriva_ml.schema.annotations import asset_annotation, generate_annotation


def create_dataset_table(
    schema: Schema,
    execution_table: Table,
    project_name: str,
    dataset_annotation: Optional[dict] = None,
    version_annotation: Optional[dict] = None,
):
    dataset_table = schema.create_table(
        Table.define(
            tname="Dataset",
            column_defs=[
                Column.define("Description", builtin_types.markdown),
                Column.define("Deleted", builtin_types.boolean),
            ],
            annotations=dataset_annotation if dataset_annotation is not None else {},
        )
    )

    dataset_type = schema.create_table(Table.define_vocabulary(MLVocab.dataset_type, f"{project_name}:{{RID}}"))
    schema.create_table(
        Table.define_association(
            associates=[
                ("Dataset", dataset_table),
                (MLVocab.dataset_type, dataset_type),
            ]
        )
    )

    dataset_version = schema.create_table(define_table_dataset_version(schema.name, version_annotation))
    dataset_table.create_reference(("Version", True, dataset_version))

    # Nested datasets.
    schema.create_table(
        Table.define_association(associates=[("Dataset", dataset_table), ("Nested_Dataset", dataset_table)])
    )
    schema.create_table(
        Table.define_association(associates=[("Dataset", dataset_table), ("Execution", execution_table)])
    )


def define_table_dataset_version(sname: str, annotation: Optional[dict] = None):
    return Table.define(
        tname="Dataset_Version",
        column_defs=[
            Column.define(
                "Version",
                builtin_types.text,
                default="0.1.0",
                comment="Semantic version of dataset",
            ),
            Column.define("Description", builtin_types.markdown),
            Column.define("Dataset", builtin_types.text, comment="RID of dataset"),
            Column.define("Execution", builtin_types.text, comment="RID of execution"),
            Column.define("Minid", builtin_types.text, comment="URL to MINID for dataset"),
            Column.define(
                "Snapshot",
                builtin_types.text,
                comment="Catalog Snapshot ID for dataset",
            ),
        ],
        annotations=annotation,
        key_defs=[Key.define(["Dataset", "Version"])],
        fkey_defs=[
            ForeignKey.define(["Dataset"], sname, "Dataset", ["RID"]),
            ForeignKey.define(["Execution"], sname, "Execution", ["RID"]),
        ],
    )


def create_execution_table(schema, annotation: Optional[dict] = None):
    annotation = annotation if annotation is not None else {}
    execution = schema.create_table(
        Table.define(
            "Execution",
            column_defs=[
                Column.define("Workflow", builtin_types.text),
                Column.define("Description", builtin_types.markdown),
                Column.define("Duration", builtin_types.text),
                Column.define("Status", builtin_types.text),
                Column.define("Status_Detail", builtin_types.text),
            ],
            fkey_defs=[ForeignKey.define(["Workflow"], schema.name, "Workflow", ["RID"])],
            annotations=annotation,
        )
    )
    return execution


def create_asset_table(
    schema,
    asset_name: str,
    execution_table,
    asset_type_table,
    asset_role_table,
):
    asset_table = schema.create_table(
        Table.define_asset(
            sname=schema.name,
            tname=asset_name,
            hatrac_template="/hatrac/metadata/{{MD5}}.{{Filename}}",
        )
    )
    schema.create_table(
        Table.define_association(
            [
                (asset_name, asset_table),
                ("Asset_Type", asset_type_table),
            ],
        )
    )

    atable = schema.create_table(
        Table.define_association(
            [
                (asset_name, asset_table),
                ("Execution", execution_table),
            ],
        )
    )
    atable.create_reference(asset_role_table)
    asset_annotation(asset_table)
    return asset_table


def create_file_table(
    schema: Schema,
    execution_table: Table,
    project_name: str,
    annotation: Optional[dict] = None,
):
    """Define files table structure"""
    annotation = annotation or {}
    file_table = schema.create_table(Table.define_asset(sname=schema.name, tname="File"))

    file_type = schema.create_table(Table.define_vocabulary(MLVocab.file_type, f"{project_name}:{{RID}}"))

    schema.create_table(
        Table.define_association(
            associates=[
                ("File", file_table),
                (MLVocab.file_type, file_type),
            ]
        )
    )
    schema.create_table(
        Table.define_association(
            [
                ("File", file_table),
                ("Execution", execution_table),
            ]
        )
    )


def create_workflow_table(schema: Schema, annotations: Optional[dict[str, Any]] = None):
    annotations = annotations or {}
    workflow_table = schema.create_table(
        Table.define(
            "Workflow",
            column_defs=[
                Column.define("Name", builtin_types.text),
                Column.define("Description", builtin_types.markdown),
                Column.define("URL", builtin_types.ermrest_uri),
                Column.define("Checksum", builtin_types.text),
                Column.define("Version", builtin_types.text),
            ],
            annotations=annotations,
        )
    )
    workflow_table.create_reference(
        schema.create_table(Table.define_vocabulary(MLVocab.workflow_type, f"{schema.name}:{{RID}}"))
    )
    return workflow_table


def create_ml_schema(
    catalog: ErmrestCatalog,
    schema_name: str = "deriva-ml",
    project_name: Optional[str] = None,
):
    project_name = project_name or schema_name

    model = catalog.getCatalogModel()
    if model.schemas.get(schema_name):
        model.schemas[schema_name].drop(cascade=True)

    # get annotations
    annotations = generate_annotation(model, schema_name)

    client_annotation = {
        "tag:misd.isi.edu,2015:display": {"name": "Users"},
        "tag:isrd.isi.edu,2016:table-display": {"row_name": {"row_markdown_pattern": "{{{Full_Name}}}"}},
        "tag:isrd.isi.edu,2016:visible-columns": {"compact": ["Full_Name", "Display_Name", "Email", "ID"]},
    }
    model.schemas["public"].tables["ERMrest_Client"].annotations.update(client_annotation)
    model.apply()

    schema = model.create_schema(Schema.define(schema_name, annotations=annotations["schema_annotation"]))

    # Create workflow and execution table.

    schema.create_table(Table.define_vocabulary("Feature_Name", f"{project_name}:{{RID}}"))
    asset_type_table = schema.create_table(Table.define_vocabulary("Asset_Type", f"{project_name}:{{RID}}"))
    asset_role_table = schema.create_table(Table.define_vocabulary("Asset_Role", f"{project_name}:{{RID}}"))

    create_workflow_table(schema, annotations["workflow_annotation"])
    execution_table = create_execution_table(schema, annotations["execution_annotation"])
    create_dataset_table(
        schema,
        execution_table,
        project_name,
        annotations["dataset_annotation"],
        annotations["dataset_version_annotation"],
    )

    create_asset_table(
        schema,
        "Execution_Metadata",
        execution_table,
        asset_type_table,
        asset_role_table,
    )

    create_asset_table(
        schema,
        "Execution_Asset",
        execution_table,
        asset_type_table,
        asset_role_table,
    )

    # File table
    create_file_table(schema, execution_table, project_name)

    initialize_ml_schema(model, schema_name)


def initialize_ml_schema(model: Model, schema_name: str = "deriva-ml"):
    catalog = model.catalog
    asset_type = catalog.getPathBuilder().schemas[schema_name].tables["Asset_Type"]
    asset_type.insert(
        [
            {
                "Name": "Execution_Config",
                "Description": "Configuration File for execution metadata",
            },
            {
                "Name": "Runtime_Env",
                "Description": "Information about the execution environment",
            },
            {
                "Name": "Execution_Metadata",
                "Description": "Information about the execution environment",
            },
            {
                "Name": "Execution_Asset",
                "Description": "A file generated by an execution",
            },
        ],
        defaults={"ID", "URI"},
    )
    asset_role = catalog.getPathBuilder().schemas[schema_name].tables["Asset_Role"]
    asset_role.insert(
        [
            {"Name": "Input", "Description": "Asset used for input of an execution."},
            {"Name": "Output", "Description": "Asset used for output of an execution."},
        ],
        defaults={"ID", "URI"},
    )


def create_ml_catalog(hostname: str, project_name: str) -> ErmrestCatalog:
    server = DerivaServer("https", hostname, credentials=get_credential(hostname))
    catalog = server.create_ermrest_catalog()
    model = catalog.getCatalogModel()
    model.configure_baseline_catalog()
    policy_file = files("deriva_ml.schema").joinpath("policy.json")
    subprocess.run(
        [
            "deriva-acl-config",
            "--host",
            catalog.deriva_server.server,
            "--config-file",
            policy_file,
            catalog.catalog_id,
        ]
    )
    create_ml_schema(catalog, project_name=project_name)
    return catalog


def reset_ml_schema(catalog: ErmrestCatalog, ml_schema=ML_SCHEMA) -> None:
    builtin_schemas = ("public", ml_schema, "www", "WWW")
    model = catalog.getCatalogModel()

    if sname_list := [s for s in model.schemas if s not in builtin_schemas]:
        sname = sname_list[0]
        for trial in range(3):
            for t in [v for v in model.schemas[sname].tables.values()]:
                try:
                    t.drop()
                except HTTPError:
                    pass
        model.schemas[sname].drop()
    # Empty out remaining tables.
    pb = catalog.getPathBuilder()
    retry = True
    while retry:
        for t in pb.schemas[ML_SCHEMA].tables.values():
            try:
                t.delete()
            except DataPathException as e:  # FK constraint.
                if "Resource not found" in e.message:
                    retry = False
                elif "still referenced from table" in e.message:
                    pass   # Foreign key constraint.
                else:
                    raise e

    initialize_ml_schema(model, ml_schema)


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
