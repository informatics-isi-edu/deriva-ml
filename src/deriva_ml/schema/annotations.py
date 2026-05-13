"""Catalog and table annotation generators for DerivaML schemas.

Applies Chaise display annotations (visible-columns, row-name patterns,
bulk-upload specs, and export configurations) to a catalog model. These
annotations control how the Chaise web interface presents DerivaML tables.

Public entry points:

- ``catalog_annotation``: Apply all standard annotations to a full catalog model.
- ``asset_annotation``: Apply upload and display annotations to a single asset table.
- ``generate_annotation``: Return the full annotation dict for a catalog model
  (used by ``create_ml_schema``).
- ``main``: CLI wrapper — apply annotations to a live catalog.
"""

import argparse

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib
import sys

_ermrest_model = importlib.import_module("deriva.core.ermrest_model")
_core_utils = importlib.import_module("deriva.core.utils.core_utils")

Model = _ermrest_model.Model
Table = _ermrest_model.Table
deriva_tags = _core_utils.tag

from deriva_ml.core.constants import DerivaAssetColumns
from deriva_ml.core.upload_layout import bulk_upload_configuration
from deriva_ml.model.catalog import DerivaModel


def build_navbar_menu(model: DerivaModel) -> dict:
    """Construct the Chaise navbar menu tree for a deriva-ml catalog.

    Walks the live model to enumerate domain-schema tables, vocabularies,
    asset tables, and features, and assembles a navbar menu with the
    DerivaML layout:

        User Info / Deriva-ML / WWW / {domain schemas} /
        Vocabulary / Assets / Features / Catalog Registry / Documentation

    Pure function — no side effects on the model. Used by both
    :func:`catalog_annotation` and :meth:`DerivaML.apply_catalog_annotations`
    so the navbar layout has one source of truth.

    Args:
        model: The :class:`DerivaModel` of the live catalog.

    Returns:
        A ``{"newTab": False, "children": [...]}`` dict suitable for
        the ``navbarMenu`` key inside Chaise's ``chaise_config``
        annotation.
    """
    catalog_id = model.catalog.catalog_id
    ml_schema = model.ml_schema

    # One menu per domain schema (skip vocabs, associations, features).
    domain_schema_menus = []
    for domain_schema in sorted(model.domain_schemas):
        if domain_schema not in model.schemas:
            continue
        domain_schema_menus.append(
            {
                "name": domain_schema,
                "children": [
                    {
                        "name": tname,
                        "url": f"/chaise/recordset/#{catalog_id}/{domain_schema}:{tname}",
                    }
                    for tname in model.schemas[domain_schema].tables
                    if not (model.is_vocabulary(tname) or model.is_association(tname, pure=False, max_arity=3))
                ],
            }
        )

    # Vocabulary menu (ML schema vocabs + per-domain vocabs).
    vocab_children: list[dict] = [{"name": f"{ml_schema} Vocabularies", "header": True}]
    vocab_children.extend(
        {
            "url": f"/chaise/recordset/#{catalog_id}/{ml_schema}:{tname}",
            "name": tname,
        }
        for tname in model.schemas[ml_schema].tables
        if model.is_vocabulary(tname)
    )
    for domain_schema in sorted(model.domain_schemas):
        if domain_schema not in model.schemas:
            continue
        vocab_children.append({"name": f"{domain_schema} Vocabularies", "header": True})
        vocab_children.extend(
            {
                "url": f"/chaise/recordset/#{catalog_id}/{domain_schema}:{tname}",
                "name": tname,
            }
            for tname in model.schemas[domain_schema].tables
            if model.is_vocabulary(tname)
        )

    # Asset menu (ML schema + all domain schemas).
    asset_children: list[dict] = [
        {
            "url": f"/chaise/recordset/#{catalog_id}/{ml_schema}:{tname}",
            "name": tname,
        }
        for tname in model.schemas[ml_schema].tables
        if model.is_asset(tname)
    ]
    for domain_schema in sorted(model.domain_schemas):
        if domain_schema not in model.schemas:
            continue
        asset_children.extend(
            {
                "url": f"/chaise/recordset/#{catalog_id}/{domain_schema}:{tname}",
                "name": tname,
            }
            for tname in model.schemas[domain_schema].tables
            if model.is_asset(tname)
        )

    # Features menu — one entry per (target table, feature) pair.
    feature_children = [
        {
            "url": (f"/chaise/recordset/#{catalog_id}/{f.feature_table.schema.name}:{f.feature_table.name}"),
            "name": f"{f.target_table.name}:{f.feature_name}",
        }
        for f in model.find_features()
    ]

    return {
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
            {
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
            *domain_schema_menus,
            {"name": "Vocabulary", "children": vocab_children},
            {"name": "Assets", "children": asset_children},
            {"name": "Features", "children": feature_children},
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
    }


def catalog_annotation(
    model: DerivaModel,
    *,
    navbar_brand_text: str = "ML Data Browser",
    head_title: str = "Catalog ML",
) -> None:
    """Set the catalog-level annotations (display, chaise_config, bulk_upload).

    Walks the live model to construct the navbar via
    :func:`build_navbar_menu`, attaches it to a :data:`chaise_config`
    annotation, and applies the result. The catalog's
    ``annotations`` attribute is updated and the change pushed.

    Args:
        model: A deriva model to the current catalog.
        navbar_brand_text: Text displayed in Chaise's brand area.
        head_title: Title displayed in the browser tab.
    """
    annotation = {
        deriva_tags.display: {"name_style": {"underline_space": True}},
        deriva_tags.chaise_config: {
            "headTitle": head_title,
            "navbarBrandText": navbar_brand_text,
            "systemColumnsDisplayEntry": ["RID"],
            "systemColumnsDisplayCompact": ["RID"],
            "defaultTable": {"table": "Dataset", "schema": "deriva-ml"},
            "deleteRecord": True,
            "showFaceting": True,
            "shareCiteAcls": True,
            "exportConfigsSubmenu": {"acls": {"show": ["*"], "enable": ["*"]}},
            "resolverImplicitCatalog": False,
            "showWriterEmptyRelatedOnLoad": False,
            "navbarMenu": build_navbar_menu(model),
        },
        deriva_tags.bulk_upload: bulk_upload_configuration(model=model),
    }
    model.annotations.update(annotation)
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

    asset_type_source = {
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
    }

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
                asset_type_source,
            ]
            + [fkey_column(c) for c in asset_metadata],
            "detailed": [
                "RID",
                "Filename",
                "Description",
                asset_type_source,
                "URL",
                "Length",
                "MD5",
                "RCT",
                "RMT",
                [schema, f"{asset_name}_RCB_fkey"],
                [schema, f"{asset_name}_RMB_fkey"],
            ]
            + [fkey_column(c) for c in asset_metadata],
            "filter": {
                "and": [
                    {"source": "RID"},
                    {"source": "Filename"},
                    {"source": "Description"},
                    asset_type_source,
                    {
                        "source": [{"outbound": [schema, f"{asset_name}_RCB_fkey"]}, "RID"],
                        "markdown_name": "Created By",
                    },
                    {
                        "source": [{"outbound": [schema, f"{asset_name}_RMB_fkey"]}, "RID"],
                        "markdown_name": "Modified By",
                    },
                ]
            },
        },
    }
    asset_table.annotations.update(annotations)

    # Enable file preview for text-based files uploaded as application/octet-stream
    # (e.g., uv.lock, .toml). Merges into the existing asset annotation on the URL
    # column that was set by AssetTableDef.
    url_col = asset_table.columns["URL"]
    url_annotation = url_col.annotations.get(deriva_tags.asset, {})
    url_annotation["display"] = {
        "*": {
            "file_preview": {
                "content_type_mapping": {
                    "application/octet-stream": "text",
                }
            }
        }
    }
    url_col.annotations[deriva_tags.asset] = url_annotation

    asset_table.schema.model.apply()


def generate_annotation(model: Model, schema: str) -> dict:
    workflow_annotation = {
        deriva_tags.table_display: {
            "*": {
                "row_order": [{"column": "RCT", "descending": True}],
            },
        },
        deriva_tags.visible_columns: {
            "*": [
                "RID",
                "Name",
                [schema, "Workflow_RCB_fkey"],
                "RCT",
                "Description",
                "Version",
            ],
            "detailed": [
                "RID",
                "Name",
                "Description",
                {
                    "source": [
                        {"inbound": [schema, "Workflow_Workflow_Type_Workflow_fkey"]},
                        {"outbound": [schema, "Workflow_Workflow_Type_Workflow_Type_fkey"]},
                        "RID",
                    ],
                    "markdown_name": "Workflow Types",
                },
                {
                    "display": {"markdown_pattern": "[{{{URL}}}]({{{URL}}})"},
                    "markdown_name": "URL",
                },
                "Checksum",
                "Version",
                [schema, "Workflow_RCB_fkey"],
                [schema, "Workflow_RMB_fkey"],
            ],
            "filter": {
                "and": [
                    {"source": "RID"},
                    {"source": "Name"},
                    {"source": "Description"},
                    {
                        "source": [
                            {"inbound": [schema, "Workflow_Workflow_Type_Workflow_fkey"]},
                            {"outbound": [schema, "Workflow_Workflow_Type_Workflow_Type_fkey"]},
                            "RID",
                        ],
                        "markdown_name": "Workflow Types",
                    },
                    {
                        "source": [{"outbound": [schema, "Workflow_RCB_fkey"]}, "RID"],
                        "markdown_name": "Created By",
                    },
                ],
            },
        },
    }

    execution_annotation = {
        deriva_tags.table_display: {
            "*": {
                "row_order": [{"column": "RCT", "descending": True}],
            },
        },
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
        deriva_tags.visible_foreign_keys: {
            "detailed": [
                {
                    "source": [
                        {"inbound": [schema, "Execution_Execution_Nested_Execution_fkey"]},
                        {"outbound": [schema, "Execution_Execution_Execution_fkey"]},
                        "RID",
                    ],
                    "markdown_name": "Parent Executions",
                },
                {
                    "source": [
                        {"inbound": [schema, "Execution_Execution_Execution_fkey"]},
                        {"outbound": [schema, "Execution_Execution_Nested_Execution_fkey"]},
                        "RID",
                    ],
                    "markdown_name": "Child Executions",
                },
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
                        {"inbound": [schema, "Execution_Metadata_Execution_Execution_fkey"]},
                        {"outbound": [schema, "Execution_Metadata_Execution_Execution_Metadata_fkey"]},
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
