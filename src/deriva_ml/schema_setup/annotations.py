import argparse
import sys

from deriva.core.utils.core_utils import tag as deriva_tags
from ..deriva_model import DerivaModel
from ..upload import bulk_upload_configuration


def generate_annotation(model: DerivaModel) -> dict:
    catalog_id = model.catalog.catalog_id
    schema = model.ml_schema
    workflow_annotation = {
        deriva_tags.visible_columns: {
            "*": [
                "RID",
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

    execution_asset_annotation = {
        deriva_tags.table_display: {
            "row_name": {"row_markdown_pattern": "{{{Filename}}}"}
        },
        deriva_tags.visible_columns: {
            "compact": [
                "RID",
                "URL",
                "Description",
                "Length",
                [schema, "Execution_Asset_Execution_Asset_Type_fkey"],
                # {
                #     "display": {
                #         "template_engine": "handlebars",
                #         "markdown_pattern": "{{#if (eq  _Execution_Asset_Type \"2-5QME\")}}\n ::: iframe []("
                #                             "https://dev.eye-ai.org/~vivi/deriva-webapps/plot/?config=test-line"
                #                             "-plot&Execution_Asset_RID={{{RID}}}){class=chaise-autofill "
                #                             "style=\"min-width: 500px; min-height: 300px;\"} \\n:::\n {{/if}}"
                #     },
                #     "markdown_name": "ROC Plot"
                # }
            ],
            "detailed": [
                "RID",
                "RCT",
                "RMT",
                "RCB",
                "RMB",
                # {
                #     "display": {
                #         "template_engine": "handlebars",
                #         "markdown_pattern": "{{#if (eq _Execution_Asset_Type \"2-5QME\")}} ::: iframe []("
                #                             "https://dev.eye-ai.org/~vivi/deriva-webapps/plot/?config=test-line"
                #                             "-plot&Execution_Asset_RID={{{RID}}}){style=\"min-width:1000px; "
                #                             "min-height:700px; height:70vh;\" class=\"chaise-autofill\"} \\n::: {"
                #                             "{/if}}"
                #     },
                #     "markdown_name": "ROC Plot"
                # },
                "URL",
                "Filename",
                "Description",
                "Length",
                "MD5",
                [schema, "Execution_Asset_Execution_Asset_Type_fkey"],
            ],
        },
    }

    execution_metadata_annotation = {
        deriva_tags.table_display: {
            "row_name": {"row_markdown_pattern": "{{{Filename}}}"}
        }
    }

    dataset_annotation = {
        # Setup Facet on types
        # Make types in visible columns
        # Have all connected values be visible FK.
    }

    schema_annotation = {
        "name_style": {"underline_space": True},
    }

    catalog_annotation = {
        deriva_tags.display: {"name_style": {"underline_space": True}},
        deriva_tags.chaise_config: {
            "headTitle": "Catalog ML",
            "navbarBrandText": "ML Data Browser",
            "systemColumnsDisplayEntry": ["RID"],
            "systemColumnsDisplayCompact": ["RID"],
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
                    {
                        "name": "Deriva-ML",
                        "children": [
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{schema}:Workflow",
                                "name": "Workflow",
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{schema}:Workflow_Type",
                                "name": "Workflow Type",
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{schema}:Execution",
                                "name": "Execution",
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{schema}:Execution_Metadata",
                                "name": "Execution Metadata",
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{schema}:Execution_Metadata_Type",
                                "name": "Execution Metadata Type",
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{schema}:Execution_Asset",
                                "name": "Execution Asset",
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{schema}:Execution_Asset_Type",
                                "name": "Execution Asset Type",
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{schema}:Dataset",
                                "name": "Dataset",
                            },
                        ],
                    },
                ],
            },
            "defaultTable": {"table": "Dataset", "schema": "deriva-ml"},
            "deleteRecord": True,
            "showFaceting": True,
            "shareCiteAcls": True,
            "exportConfigsSubmenu": {"acls": {"show": ["*"], "enable": ["*"]}},
            "resolverImplicitCatalog": catalog_id,
        },
        deriva_tags.bulk_upload: bulk_upload_configuration(model=DerivaModel(model)),
    }

    return {
        "workflow_annotation": workflow_annotation,
        "dataset_annotation": dataset_annotation,
        "execution_annotation": execution_annotation,
        "execution_asset_annotation": execution_asset_annotation,
        "execution_metadata_annotation": execution_metadata_annotation,
        "schema_annotation": schema_annotation,
        "catalog_annotation": catalog_annotation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog_id", type=str, required=True)
    parser.add_argument("--schema_name", type=str, required=True)
    args = parser.parse_args()
    generate_annotation(args.catalog_id, args.schema_name)


if __name__ == "__main__":
    sys.exit(main())
