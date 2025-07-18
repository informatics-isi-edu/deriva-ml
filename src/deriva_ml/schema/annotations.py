import argparse

from deriva.core.ermrest_model import Model, Table
from deriva.core.utils.core_utils import tag as deriva_tags

from deriva_ml.core.constants import DerivaAssetColumns
from deriva_ml.dataset.upload import bulk_upload_configuration
from deriva_ml.model.catalog import DerivaModel


def catalog_annotation(model: DerivaModel) -> None:
    """Set the annotations for a catalog.

    This routine will dynamically walk the domain schema and create menu bar for the catalog based on the current
    configuration.  A side effect is that the annotation attribute of the catalog will be updated and the result
    pushed to the catalog.


    Args:
        model: A deriva model to the current catalog.

    """
    catalog_id = model.catalog.catalog_id
    ml_schema = model.ml_schema

    catalog_annotation = {
        deriva_tags.display: {"name_style": {"underline_space": True}},
        deriva_tags.chaise_config: {
            "headTitle": "Catalog ML",
            "navbarBrandText": "ML Data Browser",
            "systemColumnsDisplayEntry": ["RID"],
            "systemColumnsDisplayCompact": ["RID"],
            "defaultTable": {"table": "Dataset", "schema": "deriva-ml"},
            "deleteRecord": True,
            "showFaceting": True,
            "shareCiteAcls": True,
            "exportConfigsSubmenu": {"acls": {"show": ["*"], "enable": ["*"]}},
            "resolverImplicitCatalog": False,
            "navbarMenu": {
                "newTab": False,
                "children": [
                    {
                        "name": "User Info",
                        "children": [
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/public:ERMrest_Client",
                                "name": "Users",
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/public:ERMrest_Group",
                                "name": "Groups",
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/public:ERMrest_RID_Lease",
                                "name": "ERMrest RID Lease",
                            },
                        ],
                    },
                    {  # All the primary tables in deriva-ml schema.
                        "name": "Deriva-ML",
                        "children": [
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{ml_schema}:Workflow",
                                "name": "Workflow",
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{ml_schema}:Execution",
                                "name": "Execution",
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{ml_schema}:Execution_Metadata",
                                "name": "Execution Metadata",
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{ml_schema}:Execution_Asset",
                                "name": "Execution Asset",
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{ml_schema}:Dataset",
                                "name": "Dataset",
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{ml_schema}:Dataset_Version",
                                "name": "Dataset Version",
                            },
                        ],
                    },
                    {  # All the primary tables in deriva-ml schema.
                        "name": "WWW",
                        "children": [
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/WWW:Page",
                                "name": "Page",
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/WWW:File",
                                "name": "File",
                            },
                        ],
                    },
                    {
                        "name": model.domain_schema,
                        "children": [
                            {
                                "name": tname,
                                "url": f"/chaise/recordset/#{catalog_id}/{model.domain_schema}:{tname}",
                            }
                            for tname in model.schemas[model.domain_schema].tables
                            # Don't include controlled vocabularies, association tables, or feature tables.
                            if not (model.is_vocabulary(tname) or model.is_association(tname, pure=False, max_arity=3))
                        ],
                    },
                    {  # Vocabulary menu which will list all the controlled vocabularies in deriva-ml and domain.
                        "name": "Vocabulary",
                        "children": [{"name": f"{ml_schema} Vocabularies", "header": True}]
                        + [
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{ml_schema}:{tname}",
                                "name": tname,
                            }
                            for tname in model.schemas[model.ml_schema].tables
                            if model.is_vocabulary(tname)
                        ]
                        + [
                            {
                                "name": f"{model.domain_schema} Vocabularies",
                                "header": True,
                            }
                        ]
                        + [
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{model.domain_schema}:{tname}",
                                "name": tname,
                            }
                            for tname in model.schemas[model.domain_schema].tables
                            if model.is_vocabulary(tname)
                        ],
                    },
                    {  # List of all of the asset tables in deriva-ml and domain schemas.
                        "name": "Assets",
                        "children": [
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{ml_schema}:{tname}",
                                "name": tname,
                            }
                            for tname in model.schemas[model.ml_schema].tables
                            if model.is_asset(tname)
                        ]
                        + [
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{model.domain_schema}:{tname}",
                                "name": tname,
                            }
                            for tname in model.schemas[model.domain_schema].tables
                            if model.is_asset(tname)
                        ],
                    },
                    {
                        "url": "/chaise/recordset/#0/ermrest:registry@sort(RID)",
                        "name": "Catalog Registry",
                    },
                    {
                        "name": "Documentation",
                        "children": [
                            {
                                "url": "https://github.com/informatics-isi-edu/deriva-ml/blob/main/docs/ml_workflow_instruction.md",
                                "name": "ML Notebook Instruction",
                            },
                            {
                                "url": "https://informatics-isi-edu.github.io/deriva-ml/",
                                "name": "Deriva-ML Documentation",
                            },
                        ],
                    },
                ],
            },
        },
        deriva_tags.bulk_upload: bulk_upload_configuration(model=model),
    }
    model.annotations.update(catalog_annotation)
    model.apply()


def asset_annotation(asset_table: Table):
    """Generate annotations for an asset table.
    
    Args:
        asset_table: The Table object representing the asset table.
        
    Returns:
        A dictionary containing the annotations for the asset table.
    """

    schema = asset_table.schema.name
    asset_name = asset_table.name
    asset_metadata = {c.name for c in asset_table.columns} - DerivaAssetColumns

    def fkey_column(column):
        """Map the column name to a FK if a constraint exists on the column"""
        return next(
            (
                (fk.name[0].name, fk.name[1])
                for fk in asset_table.foreign_keys
                if asset_table.columns[column] in fk.column_map
            ),
            column,
        )

    annotations = {
        deriva_tags.table_display: {"row_name": {"row_markdown_pattern": "{{{Filename}}}"}},
        deriva_tags.visible_columns: {
            "*": [
                "RID",
                "RCT",
                "RMT",
                [schema, f"{asset_name}_RCB_fkey"],
                [schema, f"{asset_name}_RMB_fkey"],
                "URL",
                "Filename",
                "Description",
                "Length",
                "MD5",
                {
                    "source": [
                        {
                            "inbound": [
                                schema,
                                f"{asset_name}_Asset_Type_{asset_name}_fkey",
                            ]
                        },
                        {
                            "outbound": [
                                schema,
                                f"{asset_name}_Asset_Type_Asset_Type_fkey",
                            ]
                        },
                        "RID",
                    ],
                    "markdown_name": "Asset Types",
                },
            ]
            + [fkey_column(c) for c in asset_metadata],
        },
    }
    asset_table.annotations.update(annotations)
    asset_table.schema.model.apply()


def generate_annotation(model: Model, schema: str) -> dict:
    catalog_id = model.catalog.catalog_id
    workflow_annotation = {
        deriva_tags.visible_columns: {
            "*": [
                "RID",
                [schema, "Workflow_RCB_fkey"],
                [schema, "Workflow_RMB_fkey"],
                "Name",
                "Description",
                {
                    "display": {"markdown_pattern": "[{{{URL}}}]({{{URL}}})"},
                    "markdown_name": "URL",
                },
                "Checksum",
                "Version",
                {
                    "source": [
                        {"outbound": [schema, "Workflow_Workflow_Type_fkey"]},
                        "RID",
                    ]
                },
            ]
        }
    }

    execution_annotation = {
        deriva_tags.visible_columns: {
            "*": [
                "RID",
                [schema, "Execution_RCB_fkey"],
                [schema, "Execution_RMB_fkey"],
                "RCT",
                "Description",
                {"source": [{"outbound": [schema, "Execution_Workflow_fkey"]}, "RID"]},
                "Duration",
                "Status",
                "Status_Detail",
            ]
        },
        "tag:isrd.isi.edu,2016:visible-foreign-keys": {
            "detailed": [
                {
                    "source": [
                        {"inbound": [schema, "Dataset_Execution_Execution_fkey"]},
                        {"outbound": [schema, "Dataset_Execution_Dataset_fkey"]},
                        "RID",
                    ],
                    "markdown_name": "Dataset",
                },
                {
                    "source": [
                        {
                            "inbound": [
                                schema,
                                "Execution_Asset_Execution_Execution_fkey",
                            ]
                        },
                        {
                            "outbound": [
                                schema,
                                "Execution_Asset_Execution_Execution_Asset_fkey",
                            ]
                        },
                        "RID",
                    ],
                    "markdown_name": "Execution Asset",
                },
                {
                    "source": [
                        {"inbound": [schema, "Execution_Metadata_Execution_fkey"]},
                        "RID",
                    ],
                    "markdown_name": "Execution Metadata",
                },
            ]
        },
    }

    dataset_annotation = {
        deriva_tags.visible_columns: {
            "*": [
                "RID",
                "Description",
                [schema, "Dataset_RCB_fkey"],
                [schema, "Dataset_RMB_fkey"],
                {
                    "source": [
                        {"outbound": ["deriva-ml", "Dataset_Version_fkey"]},
                        "Version",
                    ],
                    "markdown_name": "Dataset Version",
                },
            ],
            "detailed": [
                "RID",
                "Description",
                {
                    "source": [
                        {"inbound": ["deriva-ml", "Dataset_Dataset_Type_Dataset_fkey"]},
                        {
                            "outbound": [
                                "deriva-ml",
                                "Dataset_Dataset_Type_Dataset_Type_fkey",
                            ]
                        },
                        "RID",
                    ],
                    "markdown_name": "Dataset Types",
                },
                {
                    "source": [
                        {"outbound": ["deriva-ml", "Dataset_Version_fkey"]},
                        "Version",
                    ],
                    "markdown_name": "Dataset Version",
                },
                [schema, "Dataset_RCB_fkey"],
                [schema, "Dataset_RMB_fkey"],
            ],
            "filter": {
                "and": [
                    {"source": "RID"},
                    {"source": "Description"},
                    {
                        "source": [
                            {
                                "inbound": [
                                    "deriva-ml",
                                    "Dataset_Dataset_Type_Dataset_fkey",
                                ]
                            },
                            {
                                "outbound": [
                                    "deriva-ml",
                                    "Dataset_Dataset_Type_Dataset_Type_fkey",
                                ]
                            },
                            "RID",
                        ],
                        "markdown_name": "Dataset Types",
                    },
                    {
                        "source": [{"outbound": [schema, "Dataset_RCB_fkey"]}, "RID"],
                        "markdown_name": "Created By",
                    },
                    {
                        "source": [{"outbound": [schema, "Dataset_RMB_fkey"]}, "RID"],
                        "markdown_name": "Modified By",
                    },
                ]
            },
        }
    }

    schema_annotation = {
        "name_style": {"underline_space": True},
    }

    dataset_version_annotation = {
        deriva_tags.visible_columns: {
            "*": [
                "RID",
                "RCT",
                "RMT",
                [schema, "Dataset_Version_RCB_fkey"],
                [schema, "Dataset_Version_RMB_fkey"],
                {
                    "source": [
                        {"outbound": [schema, "Dataset_Version_Dataset_fkey"]},
                        "RID",
                    ]
                },
                "Description",
                {
                    "display": {
                        "template_engine": "handlebars",
                        "markdown_pattern": "[{{{Version}}}](https://{{{$location.host}}}/id/{{{$catalog.id}}}/{{{Dataset}}}@{{{Snapshot}}})",
                    },
                    "markdown_name": "Version",
                },
                "Minid",
                {
                    "source": [
                        {"outbound": [schema, "Dataset_Version_Execution_fkey"]},
                        "RID",
                    ]
                },
            ]
        },
        deriva_tags.visible_foreign_keys: {"*": []},
        deriva_tags.table_display: {
            "row_name": {"row_markdown_pattern": "{{{$fkey_deriva-ml_Dataset_Version_Dataset_fkey.RID}}}:{{{Version}}}"}
        },
    }

    return {
        "workflow_annotation": workflow_annotation,
        "dataset_annotation": dataset_annotation,
        "execution_annotation": execution_annotation,
        "schema_annotation": schema_annotation,
        "dataset_version_annotation": dataset_version_annotation,
    }


def main():
    """Main entry point for the annotations CLI.
    
    Applies annotations to the ML schema based on command line arguments.
    
    Returns:
        None. Executes the CLI.
    """
    parser = argparse.ArgumentParser(description="Apply annotations to ML schema")
    parser.add_argument("hostname", help="Hostname for the catalog")
    parser.add_argument("catalog_id", help="Catalog ID")
    parser.add_argument("schema-name", default="deriva-ml", help="Schema name (default: deriva-ml)")
    args = parser.parse_args()
    generate_annotation(args.catalog_id, args.schema_name)


if __name__ == "__main__":
    sys.exit(main())
