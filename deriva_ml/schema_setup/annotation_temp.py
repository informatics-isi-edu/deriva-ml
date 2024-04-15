import argparse
import sys


def generate_annotation(catalog_id: str, schema: str) -> dict:
    workflow_annotation = {
        "tag:isrd.isi.edu,2016:visible-columns": {
            "*": [
                "RID",
                "Name",
                "Description",
                {
                    "display": {"markdown_pattern": "[{{{URL}}}]({{{URL}}})"},
                    "markdown_name": "URL"
                },
                "Checksum",
                "Version",
                {
                    "source": [
                        {
                            "outbound": [
                                schema,
                                "Workflow_Workflow_Type_fkey"
                            ]
                        },
                        "RID"
                    ]
                }
            ]
        }
    }

    execution_annotation = {
        "tag:isrd.isi.edu,2016:visible-columns": {
            "*": [
                "RID",
                [
                    schema,
                    "Execution_RCB_fkey"
                ],
                "RCT",
                "Description",
                {"source": [
                    {"outbound": [
                        "eye-ai",
                        "Execution_Workflow_fkey"
                    ]
                    },
                    "RID"
                ]
                },
                "Duration",
                "Status",
                "Status_Detail"
            ]
        },
        "tag:isrd.isi.edu,2016:visible-foreign-keys": {
            "detailed": [
                {
                    "source": [
                        {
                            "inbound": [
                                schema,
                                "Dataset_Execution_Execution_fkey"
                            ]
                        },
                        {
                            "outbound": [
                                schema,
                                "Dataset_Execution_Dataset_fkey"
                            ]
                        },
                        "RID"
                    ],
                    "markdown_name": "Dataset"
                },
                {
                    "source": [
                        {
                            "inbound": [
                                schema,
                                "Execution_Assets_Execution_Execution_fkey"
                            ]
                        },
                        {
                            "outbound": [
                                schema,
                                "Execution_Assets_Execution_Execution_Assets_fkey"
                            ]
                        },
                        "RID"
                    ],
                    "markdown_name": "Execution Assets"
                },
                {
                    "source": [
                        {
                            "inbound": [
                                schema,
                                "Execution_Metadata_Execution_fkey"
                            ]
                        },
                        "RID"
                    ],
                    "markdown_name": "Execution Metadata"
                }
            ]
        }
    }

    execution_assets_annotation = {
        "tag:isrd.isi.edu,2016:table-display": {
            "row_name": {
                "row_markdown_pattern": "{{{Filename}}}"
            }
        },
        "tag:isrd.isi.edu,2016:visible-columns": {
            "compact": [
                "RID",
                "URL",
                "Description",
                "Length",
                [
                    schema,
                    "Execution_Assets_Execution_Asset_Type_fkey"
                ],
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
                [
                    schema,
                    "Execution_Assets_Execution_Asset_Type_fkey"
                ]
            ]
        }
    }

    execution_metadata_annotation = {
        "tag:isrd.isi.edu,2016:table-display": {
            "row_name": {
                "row_markdown_pattern": "{{{Filename}}}"
            }
        }
    }

    schema_annotation = {
        "headTitle": "Catalog ML",
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
                    "name": "FaceBase-ML",
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
                        }
                    ]
                }
            ]
        },
        "navbarBrandText": "ML Data Browser",
        "systemColumnsDisplayEntry": ["RID"],
        "systemColumnsDisplayCompact": ["RID"]
    }

    return {"workflow_annotation": workflow_annotation,
            "execution_annotation": execution_annotation,
            "execution_assets_annotation": execution_assets_annotation,
            "execution_metadata_annotation": execution_metadata_annotation,
            "schema_annotation": schema_annotation
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog_id', type=str, required=True)
    parser.add_argument('--schema_name', type=str, required=True)
    args = parser.parse_args()
    return generate_annotation(args.catalog_id, args.schema_name)


if __name__ == "__main__":
    sys.exit(main())
