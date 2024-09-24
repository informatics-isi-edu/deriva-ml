import argparse
import sys

from deriva.core.ermrest_catalog import ErmrestCatalog
from deriva.core.utils.core_utils import tag as deriva_tags


def generate_annotation(catalog_id: str, schema: str) -> dict:
    workflow_annotation = {
        deriva_tags.visible_columns: {
            "*": [
                "RID",
                "Name",
                "Description",
                {"display": {"markdown_pattern": "[{{{URL}}}]({{{URL}}})"}, "markdown_name": "URL"},
                "Checksum",
                "Version",
                {"source": [{"outbound": [schema, "Workflow_Workflow_Type_fkey"]}, "RID"]}
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
                "Status_Detail"
            ]
        },
        "tag:isrd.isi.edu,2016:visible-foreign-keys": {
            "detailed": [
                {
                    "source": [{"inbound": [schema, "Dataset_Execution_Execution_fkey"]},
                               {"outbound": [schema, "Dataset_Execution_Dataset_fkey"]}, "RID"],
                    "markdown_name": "Dataset"
                },
                {
                    "source": [
                        {"inbound": [schema, "Execution_Assets_Execution_Execution_fkey"]},
                        {"outbound": [schema, "Execution_Assets_Execution_Execution_Assets_fkey"]}, "RID"],
                    "markdown_name": "Execution Assets"
                },
                {
                    "source": [{"inbound": [schema, "Execution_Metadata_Execution_fkey"]}, "RID"],
                    "markdown_name": "Execution Metadata"
                }
            ]
        }
    }

    execution_assets_annotation = {
        deriva_tags.table_display: {
            "row_name": {
                "row_markdown_pattern": "{{{Filename}}}"
            }
        },
        deriva_tags.visible_columns: {
            "compact": [
                "RID",
                "URL",
                "Description",
                "Length", [schema, "Execution_Assets_Execution_Asset_Type_fkey"],
                # {
                #     "display": {
                #         "template_engine": "handlebars",
                #         "markdown_pattern": "{{#if (eq  _Execution_Asset_Type \"2-5QME\")}}\n ::: iframe []("
                #                             "https://dev.eye-ai.org/~vivi/deriva-webapps/plot/?config=test-line"
                #                             "-plot&Execution_Assets_RID={{{RID}}}){class=chaise-autofill "
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
                #                             "-plot&Execution_Assets_RID={{{RID}}}){style=\"min-width:1000px; "
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
                [schema, "Execution_Assets_Execution_Asset_Type_fkey"]
            ]
        }
    }

    execution_metadata_annotation = {
        deriva_tags.table_display: {
            "row_name": {
                "row_markdown_pattern": "{{{Filename}}}"
            }
        }
    }

    schema_annotation = {
        'name_style': {'underline_space': True},
    }

    catalog_annotation = {
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
                                "name": "Users"
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/public:ERMrest_Group",
                                "name": "Groups"
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/public:ERMrest_RID_Lease",
                                "name": "ERMrest RID Lease"
                            }
                        ]
                    },
                    {
                        "name": "Deriva-ML",
                        "children": [
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{schema}:Workflow",
                                "name": "Workflow"
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{schema}:Workflow_Type",
                                "name": "Workflow Type"
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{schema}:Execution",
                                "name": "Execution"
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{schema}:Execution_Metadata",
                                "name": "Execution Metadata"
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{schema}:Execution_Metadata_Type",
                                "name": "Execution Metadata Type"
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{schema}:Execution_Assets",
                                "name": "Execution Assets"
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{schema}:Execution_Asset_Type",
                                "name": "Execution Asset Type"
                            },
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{schema}:Dataset",
                                "name": "Dataset"
                            }
                        ]
                    }
                ]
            }
        },
        deriva_tags.bulk_upload:
            {"asset_mappings": [
                {"column_map": {
                    "MD5": "{md5}",
                    "URL": "{URI}",
                    "Length": "{file_size}",
                    "Filename": "{file_name}",
                    "Execution_Asset_Type": "{execution_asset_type_name}"
                },
                    "file_pattern": "(?i)^.*/Execution_Assets/(?P<execution_asset_type>[A-Za-z0-9_]*)/(?P<file_name>[A-Za-z0-9_-]*)[.](?P<file_ext>[a-z0-9]*)$",
                    "target_table": ["deriva-ml", "Execution_Assets"],
                    "checksum_types": ["sha256", "md5"],
                    "hatrac_options": {"versioned_urls": True},
                    "hatrac_templates": {
                        "hatrac_uri": "/hatrac/execution_assets/{md5}.{file_name}",
                        "content-disposition": "filename*=UTF-8''{file_name}"
                    },
                    "record_query_template": "/entity/{target_table}/MD5={md5}&Filename={file_name}",
                    "metadata_query_templates": [
                        "/attribute/deriva-ml:Execution_Asset_Type/Name={execution_asset_type}/execution_asset_type_name:=Name"
                    ],
                    "create_record_before_upload": False
                },
                {
                    "column_map": {
                        "MD5": "{md5}",
                        "URL": "{URI}",
                        "Length": "{file_size}",
                        "Filename": "{file_name}",
                        "Execution_Metadata_Type": "{execution_metadata_type_name}"
                    },
                    "file_pattern": "(?i)^.*/Execution_Metadata/(?P<execution_metadata_type>[A-Za-z0-9_]*)-(?P<filename>[A-Za-z0-9_]*)[.](?P<file_ext>[a-z0-9]*)$",
                    "target_table": ["deriva-ml", "Execution_Metadata"],
                    "checksum_types": ["sha256", "md5"],
                    "hatrac_options": {"versioned_urls": True},
                    "hatrac_templates": {
                        "hatrac_uri": "/hatrac/execution_metadata/{md5}.{file_name}",
                        "content-disposition": "filename*=UTF-8''{file_name}"
                    },
                    "record_query_template": "/entity/{target_table}/MD5={md5}&Filename={file_name}",
                    "metadata_query_templates": [
                        "/attribute/deriva-ml:Execution_Metadata_Type/Name={execution_metadata_type}/execution_metadata_type_name:=Name"
                    ],
                    "create_record_before_upload": False
                }],
                "version_update_url": "https://github.com/informatics-isi-edu/deriva-client",
                "version_compatibility": [[">=1.4.0", "<2.0.0"]]
            }
    }

    return {"workflow_annotation": workflow_annotation,
            "execution_annotation": execution_annotation,
            "execution_assets_annotation": execution_assets_annotation,
            "execution_metadata_annotation": execution_metadata_annotation,
            "schema_annotation": schema_annotation,
            "catalog_annotation": catalog_annotation,
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog_id', type=str, required=True)
    parser.add_argument('--schema_name', type=str, required=True)
    args = parser.parse_args()
    generate_annotation(args.catalog_id, args.schema_name)


if __name__ == "__main__":
    sys.exit(main())
