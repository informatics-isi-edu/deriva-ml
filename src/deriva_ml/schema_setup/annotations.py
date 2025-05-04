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
            ],
            "detailed": [
                "RID",
                "RCT",
                "RMT",
                "RCB",
                "RMB",
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
    rcb_name = [schema, "Dataset_RCB_fkey"]
    rmb_name = [schema, "Dataset_RMB_fkey"]
    dataset_annotation = {
        deriva_tags.visible_columns: {
            "*": [
                "RID",
                "Description",
                rcb_name,
                rmb_name,
                {
                    "source": [{"outbound": ["deriva-ml", "Dataset_Version_fkey"]}, "Version"],
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
                    "source": [{"outbound": ["deriva-ml", "Dataset_Version_fkey"]}, "Version"],
                    "markdown_name": "Dataset Version",
                },
                rcb_name,
                rmb_name,
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
                        "source": [{"outbound": rcb_name}, "RID"],
                        "markdown_name": "Created By",
                    },
                    {
                        "source": [{"outbound": rmb_name}, "RID"],
                        "markdown_name": "Modified By",
                    }
                ]
            }
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
                "Description",
               "Version",
                {
                    "source": [
                        {"outbound": [schema, "Dataset_Version_Dataset_fkey"]},
                        "RID",
                    ]
                },
            ]
        },
        deriva_tags.visible_foreign_keys: {"*": []},
        deriva_tags.table_display:
            {
                "row_name": {
                    "row_markdown_pattern": "{{{$fkey_deriva-ml_Dataset_Version_Dataset_fkey.RID}}}:{{{Version}}}"
                }
            }
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
                            {
                                "url": f"/chaise/recordset/#{catalog_id}/{schema}:Dataset_Version",
                                "name": "Dataset Versions",
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
        "dataset_version_annotation": dataset_version_annotation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog_id", type=str, required=True)
    parser.add_argument("--schema_name", type=str, required=True)
    args = parser.parse_args()
    generate_annotation(args.catalog_id)


if __name__ == "__main__":
    sys.exit(main())
