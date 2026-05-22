"""Catalog and table annotation generators for DerivaML schemas.

Applies Chaise display annotations (visible-columns, row-name patterns,
bulk-upload specs, and export configurations) to a catalog model. These
annotations control how the Chaise web interface presents DerivaML tables.

Public entry points:

- ``catalog_annotation``: Apply all standard annotations to a full catalog model.
- ``asset_annotation``: Apply upload and display annotations to a single asset table.
- ``generate_annotation``: Return the full annotation dict for a catalog model
  (used by ``create_ml_schema``).
"""

from deriva.core.ermrest_model import Model, Table
from deriva.core.utils.core_utils import tag as deriva_tags

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


def asset_annotation(asset_table: Table) -> None:
    """Attach Chaise display annotations to an asset table in-place.

    Mutates ``asset_table.annotations`` and ``asset_table.columns['URL'].annotations``
    with the Chaise ``table-display``, ``visible-columns``, and
    ``asset`` annotation tags, then calls ``asset_table.schema.model.apply()``
    to push the changes to the catalog. Does not return anything.

    Args:
        asset_table: The Table object representing the asset table.
            Must have a ``URL`` column (the file-preview annotation
            is attached there).

    Returns:
        None. The function operates by side effect — annotations
        are written to ``asset_table.annotations`` and the catalog
        model is applied before return.

    Side effects:
        * Mutates ``asset_table.annotations`` to add ``table_display``
          and ``visible_columns`` keys.
        * Mutates ``asset_table.columns['URL'].annotations`` to add
          the file-preview ``asset`` annotation.
        * Calls ``asset_table.schema.model.apply()``, which issues an
          HTTP PUT to the catalog.
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
            # ``asset_metadata`` is a set; sort by column name for
            # deterministic output. Without the sort, the visible-
            # columns order jitters run-to-run on rebuilds and
            # breaks downstream diffs of catalog annotations.
            # Same family of invariant as the asset-manifest
            # alphabetic-order rule documented in CLAUDE.md.
            + [fkey_column(c) for c in sorted(asset_metadata)],
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
            + [fkey_column(c) for c in sorted(asset_metadata)],
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
    """Build the Chaise annotation bundles for the canonical ML tables.

    Returns a dict whose values are annotation bundles ready to pass
    as the ``annotations=`` argument to the table-creation helpers
    in ``create_schema.py``. The bundles cover ``visible-columns``,
    ``table-display``, ``visible-foreign-keys``, and related Chaise
    presentation tags for the four canonical ML tables (Workflow,
    Dataset, Execution, Dataset_Version) plus the schema-level
    Chaise config.

    Args:
        model: The catalog's :class:`~deriva.core.ermrest_model.Model`
            instance. Reserved for future model-introspection use
            (e.g., discovering feature association tables to
            customize Dataset's visible-columns). Currently not read.
        schema: Name of the ML schema. Every FK reference in the
            returned bundles is qualified with this name, so a
            non-default ``schema_name`` (e.g., ``"my_ml"``) produces
            annotations that point at the right schema's FKs rather
            than the literal ``"deriva-ml"``.

    Returns:
        A dict with five keys:

        * ``workflow_annotation``
        * ``dataset_annotation``
        * ``execution_annotation``
        * ``schema_annotation``
        * ``dataset_version_annotation``

        Each value is a Chaise-shaped annotation dict (keys are
        ``isrd-tags:...`` URIs). The caller is responsible for
        passing the bundles to the appropriate table-creation
        helpers — this function does not mutate any catalog state.

    Example:
        >>> annotations = generate_annotation(model, "deriva-ml")  # doctest: +SKIP
        >>> dataset_table = schema.create_table(  # doctest: +SKIP
        ...     TableDef(name="Dataset", annotations=annotations["dataset_annotation"]),
        ... )
    """
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
                "Execution_Duration",
                "Download_Duration",
                "Upload_Duration",
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
                        {"outbound": [schema, "Dataset_Version_fkey"]},
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
                        {"inbound": [schema, "Dataset_Dataset_Type_Dataset_fkey"]},
                        {
                            "outbound": [
                                schema,
                                "Dataset_Dataset_Type_Dataset_Type_fkey",
                            ]
                        },
                        "RID",
                    ],
                    "markdown_name": "Dataset Types",
                },
                {
                    "source": [
                        {"outbound": [schema, "Dataset_Version_fkey"]},
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
                                    schema,
                                    "Dataset_Dataset_Type_Dataset_fkey",
                                ]
                            },
                            {
                                "outbound": [
                                    schema,
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


__all__ = [
    "asset_annotation",
    "build_navbar_menu",
    "catalog_annotation",
    "generate_annotation",
]
