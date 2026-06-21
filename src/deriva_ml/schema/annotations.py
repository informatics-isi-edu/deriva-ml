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
                    {
                        # File: reference-only assets (external inputs declared
                        # via LocalFile land here). A first-class provenance
                        # table, so surface it at the top level — not only under
                        # the generic Assets submenu.
                        "url": f"/chaise/recordset/#{catalog_id}/{ml_schema}:File",
                        "name": "File",
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

    # "Produced By": the run that created this asset (the Output-role link in the
    # {Asset}_Execution association). Functionally single-valued — an asset has
    # exactly one producer — so it is shown INLINE as a visible column. An
    # inbound-through-association path is an entity set, which a non-`detailed`
    # context requires to be aggregated; `array_d` over the single Output link
    # renders as one execution. (Consumed-By, which is genuinely many-valued,
    # stays a related-entity in visible-foreign-keys below.)
    produced_by_source = {
        "source": [
            {"inbound": [schema, f"{asset_name}_Execution_{asset_name}_fkey"]},
            {"and": [{"filter": "Asset_Role", "operand_pattern": "Output", "operator": "="}]},
            {"outbound": [schema, f"{asset_name}_Execution_Execution_fkey"]},
            "RID",
        ],
        "aggregate": "array_d",
        "markdown_name": "Produced By",
    }

    annotations = {
        deriva_tags.table_display: {"row_name": {"row_markdown_pattern": "{{{Filename}}}"}},
        deriva_tags.visible_columns: {
            # Assets are write-once (uploaded, not edited), so RMT always equals
            # RCT — showing both in the compact view is redundant. RMT is kept in
            # `detailed` for a complete audit record.
            "*": [
                "RID",
                "Filename",
                "Description",
                asset_type_source,
                produced_by_source,
                "URL",
                "Length",
                "MD5",
                "RCT",
                [schema, f"{asset_name}_RCB_fkey"],
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
                produced_by_source,
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
        # Consumed By: the run(s) that took this asset as an INPUT — genuinely
        # many-valued, so it stays a related-entity (visible FK). Filtered to the
        # Input role in the {Asset}_Execution association. The producer side
        # ("Produced By") is single-valued and shown inline as a visible column
        # above, not here.
        deriva_tags.visible_foreign_keys: {
            "detailed": [
                {
                    "source": [
                        {"inbound": [schema, f"{asset_name}_Execution_{asset_name}_fkey"]},
                        {"and": [{"filter": "Asset_Role", "operand_pattern": "Input", "operator": "="}]},
                        {"outbound": [schema, f"{asset_name}_Execution_Execution_fkey"]},
                        "RID",
                    ],
                    "markdown_name": "Consumed By",
                },
            ]
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


# Columns every feature association table carries by construction (target FK is
# added separately since its name is the target-table name).
_FEATURE_STRUCTURAL_COLUMNS = {"Execution", "Feature_Name"}


def feature_annotation(feature_table: Table, target_table_name: str) -> None:
    """Attach Chaise display annotations to a feature value table in-place.

    Feature tables (``Execution_<Target>_<Feature>``) are created dynamically per
    feature, so they fall outside :func:`generate_annotation`. Without this they
    render with raw Chaise defaults — no sensible row name, no facets, and the
    producing Execution buried among system columns. Feature values *are* the ML
    data an end user explores, so they deserve a curated view:

    * **row name** = the target RID + the feature name (e.g. ``2-ABCD: Chart_Label``);
    * **visible-columns** leads with the target, the feature's own value/term/asset
      columns, then the producing Execution (provenance) and audit columns;
    * **facets** on the target, the feature's term columns (the natural
      "show me all rows labelled X" axis), and the producing Execution.

    Mutates ``feature_table.annotations`` and applies the model.

    Args:
        feature_table: The ``Execution_<Target>_<Feature>`` association table.
        target_table_name: Name of the feature's target table (its FK column on
            the feature table — e.g. ``"Subject"`` or ``"Image"``).
    """
    schema = feature_table.schema.name
    fname = feature_table.name

    # The feature's own payload columns: everything that isn't a system column,
    # the structural Execution/Feature_Name FKs, or the target FK. Sorted for
    # deterministic annotation output (same invariant as asset_annotation).
    skip = set(DerivaAssetColumns) | _FEATURE_STRUCTURAL_COLUMNS | {target_table_name}
    payload_columns = sorted(c.name for c in feature_table.columns if c.name not in skip)

    # A column is term/asset-shaped if it carries an outbound FK (to a vocab or
    # asset table); those make good facets. Map column name -> its FK constraint.
    fk_by_column = {}
    for fk in feature_table.foreign_keys:
        for col in fk.column_map:
            fk_by_column[col.name] = fk.name  # (schema_obj, constraint_name)

    def col_entry(col_name: str):
        """Render a payload column as a FK pseudo-column when it has one."""
        fk = fk_by_column.get(col_name)
        return [fk[0].name, fk[1]] if fk else col_name

    target_source = {
        "source": [{"outbound": [schema, f"{fname}_{target_table_name}_fkey"]}, "RID"],
        "markdown_name": target_table_name,
    }
    execution_source = {
        "source": [{"outbound": [schema, f"{fname}_Execution_fkey"]}, "RID"],
        "markdown_name": "Produced By",
    }

    # Facets: the target, the producing execution, and each term/asset (FK-backed)
    # payload column — the natural "filter rows by this label/value" axes.
    facet_sources = [target_source, execution_source]
    for c in payload_columns:
        if c in fk_by_column:
            facet_sources.append({"source": [{"outbound": [schema, fk_by_column[c][1]]}, "RID"], "markdown_name": c})

    # Row name: "<target RID>: <feature name>" via the target FK pseudo-column.
    target_fkey_ref = f"$fkey_{schema}_{fname}_{target_table_name}_fkey"
    row_name_pattern = "{{{" + target_fkey_ref + ".RID}}}: {{{Feature_Name}}}"

    feature_table.annotations.update(
        {
            deriva_tags.table_display: {
                "row_name": {"row_markdown_pattern": row_name_pattern},
            },
            deriva_tags.visible_columns: {
                "*": [target_source, *[col_entry(c) for c in payload_columns], execution_source, "RCT"],
                "detailed": [
                    "RID",
                    target_source,
                    "Feature_Name",
                    *[col_entry(c) for c in payload_columns],
                    execution_source,
                    "RCT",
                    "RMT",
                ],
                "filter": {"and": [{"source": "RID"}, *facet_sources]},
            },
        }
    )
    feature_table.schema.model.apply()


def vocabulary_annotation(vocab_table: Table) -> None:
    """Attach a light Chaise display annotation to a controlled-vocabulary table.

    Vocabularies (``Name``/``Description``/``Synonyms``/``ID``/``URI`` + audit)
    render acceptably on Chaise defaults, but a small curated annotation keeps
    them consistent with the other dynamically-created tables: the term ``Name``
    is the row identity, the compact view leads with Name + Description, and the
    curie/ID columns move to the detail view. A ``Name`` facet is added for
    "jump to a term" navigation.

    Mutates ``vocab_table.annotations`` and applies the model. Wired into
    :meth:`DerivaML.create_vocabulary` so runtime-created vocabularies get it
    automatically.

    Args:
        vocab_table: The controlled-vocabulary table to annotate.
    """
    schema = vocab_table.schema.name
    vname = vocab_table.name

    vocab_table.annotations.update(
        {
            deriva_tags.table_display: {"row_name": {"row_markdown_pattern": "{{{Name}}}"}},
            deriva_tags.visible_columns: {
                "*": [
                    "Name",
                    "Description",
                    "Synonyms",
                    "RID",
                ],
                "detailed": [
                    "RID",
                    "Name",
                    "Description",
                    "Synonyms",
                    "ID",
                    "URI",
                    "RCT",
                    [schema, f"{vname}_RCB_fkey"],
                ],
                "filter": {
                    "and": [
                        {"source": "Name"},
                        {"source": "Description"},
                        {"source": "RID"},
                    ]
                },
            },
        }
    )
    vocab_table.schema.model.apply()


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
                {
                    # Workflow_Type is multi-valued (an inbound entity set through
                    # the association table). In the compact context that must be
                    # aggregated (array_d → distinct row-names); the full entity
                    # set is only valid in `detailed` (annotation spec).
                    "source": [
                        {"inbound": [schema, "Workflow_Workflow_Type_Workflow_fkey"]},
                        {"outbound": [schema, "Workflow_Workflow_Type_Workflow_Type_fkey"]},
                        "RID",
                    ],
                    "aggregate": "array_d",
                    "markdown_name": "Workflow Types",
                },
                "Version",
                "Description",
                [schema, "Workflow_RCB_fkey"],
                "RCT",
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
        # Runs of this workflow — the inbound Execution_Workflow_fkey. Curated
        # explicitly (rather than left to Chaise defaults) so it carries a clear
        # label: "what executions used this workflow?" is the natural provenance
        # question from a Workflow row.
        deriva_tags.visible_foreign_keys: {
            "detailed": [
                {
                    "source": [
                        {"inbound": [schema, "Execution_Workflow_fkey"]},
                        "RID",
                    ],
                    "markdown_name": "Executions",
                },
            ]
        },
    }

    # Status is a FK to the Execution_Status vocabulary. Surfaced as the resolved
    # term (not the raw FK value) and given a dedicated facet — "show me the
    # Failed / Aborted / stranded runs" is the core run-triage question.
    execution_status_source = {
        "source": [{"outbound": [schema, "Execution_Status_fkey"]}, "Name"],
        "markdown_name": "Status",
    }
    execution_workflow_source = {
        "source": [{"outbound": [schema, "Execution_Workflow_fkey"]}, "RID"],
        "markdown_name": "Workflow",
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
                execution_status_source,
                execution_workflow_source,
                "Description",
                [schema, "Execution_RCB_fkey"],
                "RCT",
                "Execution_Duration",
                "Status_Detail",
            ],
            "detailed": [
                "RID",
                execution_status_source,
                "Status_Detail",
                execution_workflow_source,
                "Description",
                "Execution_Duration",
                "Download_Duration",
                "Upload_Duration",
                "RCT",
                "RMT",
                [schema, "Execution_RCB_fkey"],
                [schema, "Execution_RMB_fkey"],
            ],
            # Facets for run triage / provenance navigation: by Status, by
            # Workflow, by who ran it, and by when.
            "filter": {
                "and": [
                    {"source": "RID"},
                    {"source": "Description"},
                    execution_status_source,
                    {
                        "source": [{"outbound": [schema, "Execution_Workflow_fkey"]}, "RID"],
                        "markdown_name": "Workflow",
                    },
                    {"source": "RCT", "markdown_name": "Created"},
                    {
                        "source": [{"outbound": [schema, "Execution_RCB_fkey"]}, "RID"],
                        "markdown_name": "Created By",
                    },
                ]
            },
        },
        deriva_tags.visible_foreign_keys: {
            "detailed": [
                # ---- Inputs (what this run consumed) ----------------------- #
                {
                    "source": [
                        {"inbound": [schema, "Dataset_Execution_Execution_fkey"]},
                        {"outbound": [schema, "Dataset_Execution_Dataset_fkey"]},
                        "RID",
                    ],
                    "markdown_name": "Input Datasets",
                },
                {
                    "source": [
                        {"inbound": [schema, "Dataset_Execution_Execution_fkey"]},
                        {"outbound": [schema, "Dataset_Execution_Dataset_Version_fkey"]},
                        "RID",
                    ],
                    "markdown_name": "Input Dataset Versions",
                },
                {
                    # External / local files declared as inputs (the LocalFile /
                    # File-table mechanism). Filtered to the Input role so the
                    # same File table isn't double-listed as an output below.
                    "source": [
                        {"inbound": [schema, "File_Execution_Execution_fkey"]},
                        {"outbound": [schema, "File_Execution_File_fkey"]},
                        "RID",
                    ],
                    "markdown_name": "Input Files",
                },
                # ---- Outputs (what this run produced) ---------------------- #
                {
                    # Datasets this execution AUTHORED — the producer edge lives
                    # on Dataset_Version.Execution (authorship-canonical model),
                    # not on Dataset_Execution (which is input-only).
                    "source": [
                        {"inbound": [schema, "Dataset_Version_Execution_fkey"]},
                        {"outbound": [schema, "Dataset_Version_Dataset_fkey"]},
                        "RID",
                    ],
                    "markdown_name": "Output Datasets",
                },
                {
                    "source": [
                        {"inbound": [schema, "Execution_Asset_Execution_Execution_fkey"]},
                        {"outbound": [schema, "Execution_Asset_Execution_Execution_Asset_fkey"]},
                        "RID",
                    ],
                    "markdown_name": "Execution Assets",
                },
                {
                    "source": [
                        {"inbound": [schema, "Execution_Metadata_Execution_Execution_fkey"]},
                        {"outbound": [schema, "Execution_Metadata_Execution_Execution_Metadata_fkey"]},
                        "RID",
                    ],
                    "markdown_name": "Execution Metadata",
                },
                # ---- Orchestration (run hierarchy) ------------------------- #
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
            ]
        },
    }

    # Reusable sources for the Dataset table. Dataset_Type is multi-valued (an
    # inbound entity set). The full entity set is valid in `detailed`/`filter`;
    # the compact context needs the aggregated form (array_d → distinct
    # row-names) per the annotation spec.
    _dataset_types_path = [
        {"inbound": [schema, "Dataset_Dataset_Type_Dataset_fkey"]},
        {"outbound": [schema, "Dataset_Dataset_Type_Dataset_Type_fkey"]},
        "RID",
    ]
    dataset_types_source = {
        "source": _dataset_types_path,
        "markdown_name": "Dataset Types",
    }
    dataset_types_compact = {
        "source": _dataset_types_path,
        "aggregate": "array_d",
        "markdown_name": "Dataset Types",
    }
    dataset_version_source = {
        "source": [{"outbound": [schema, "Dataset_Version_fkey"]}, "Version"],
        "markdown_name": "Current Version",
    }
    # The execution that authored the dataset's CURRENT version, via
    # Dataset → current Dataset_Version → its Execution. NOTE the label is
    # deliberately "Current Version Produced By", not a bare "Produced By":
    # Dataset.Version points at whatever version is current, which is often a
    # *dev* row (post-release drift, ADR-0003) carrying no producing Execution —
    # so this is empty on a drifted dataset even though earlier released versions
    # do have producers. Per-version provenance lives on each Dataset_Version
    # record (see the "Versions" related-entity section); this column reflects
    # only the current version, and its label says so.
    dataset_producer_source = {
        "source": [
            {"outbound": [schema, "Dataset_Version_fkey"]},
            {"outbound": [schema, "Dataset_Version_Execution_fkey"]},
            "RID",
        ],
        "markdown_name": "Current Version Produced By",
    }
    dataset_annotation = {
        deriva_tags.table_display: {
            "*": {"row_order": [{"column": "RCT", "descending": True}]},
        },
        deriva_tags.visible_columns: {
            "*": [
                "RID",
                "Description",
                dataset_types_compact,
                dataset_version_source,
                [schema, "Dataset_RCB_fkey"],
                "RCT",
            ],
            "detailed": [
                "RID",
                "Description",
                dataset_types_source,
                dataset_version_source,
                dataset_producer_source,
                "Deleted",
                [schema, "Dataset_RCB_fkey"],
                [schema, "Dataset_RMB_fkey"],
                "RCT",
                "RMT",
            ],
            "filter": {
                "and": [
                    {"source": "RID"},
                    {"source": "Description"},
                    dataset_types_source,
                    {"source": "Deleted"},
                    {"source": "RCT", "markdown_name": "Created"},
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
        },
        deriva_tags.visible_foreign_keys: {
            "detailed": [
                {
                    # Member datasets (this dataset is the parent collection).
                    "source": [
                        {"inbound": [schema, "Dataset_Dataset_Dataset_fkey"]},
                        {"outbound": [schema, "Dataset_Dataset_Nested_Dataset_fkey"]},
                        "RID",
                    ],
                    "markdown_name": "Member Datasets",
                },
                {
                    # Collections this dataset belongs to (it is a member).
                    "source": [
                        {"inbound": [schema, "Dataset_Dataset_Nested_Dataset_fkey"]},
                        {"outbound": [schema, "Dataset_Dataset_Dataset_fkey"]},
                        "RID",
                    ],
                    "markdown_name": "Parent Collections",
                },
                {
                    # Version history of this dataset.
                    "source": [
                        {"inbound": [schema, "Dataset_Version_Dataset_fkey"]},
                        "RID",
                    ],
                    "markdown_name": "Versions",
                },
                {
                    # Executions that CONSUMED this dataset (input edge). The
                    # "produced by" direction is on the Version → Execution hop
                    # surfaced as a column above.
                    "source": [
                        {"inbound": [schema, "Dataset_Execution_Dataset_fkey"]},
                        {"outbound": [schema, "Dataset_Execution_Execution_fkey"]},
                        "RID",
                    ],
                    "markdown_name": "Consumed By Executions",
                },
            ]
        },
    }

    schema_annotation = {
        "name_style": {"underline_space": True},
    }

    # The producing execution (provenance) for a version. A value pointing at
    # the unknown-provenance sentinel means "origin unknown" (backfilled); NULL
    # on a dev row is expected (dev versions have no producer); NULL on a
    # released row is a contract gap.
    dataset_version_producer_source = {
        "source": [
            {"outbound": [schema, "Dataset_Version_Execution_fkey"]},
            "RID",
        ],
        "markdown_name": "Produced By",
    }
    dataset_version_dataset_source = {
        "source": [
            {"outbound": [schema, "Dataset_Version_Dataset_fkey"]},
            "RID",
        ],
        "markdown_name": "Dataset",
    }
    dataset_version_label_source = {
        "display": {
            "template_engine": "handlebars",
            "markdown_pattern": "[{{{Version}}}](https://{{{$location.host}}}/id/{{{$catalog.id}}}/{{{Dataset}}}@{{{Snapshot}}})",
        },
        "markdown_name": "Version",
    }
    dataset_version_annotation = {
        # Lead with the meaningful columns (Dataset, Version, provenance), then
        # the audit trail — matching the Dataset / asset column ordering.
        deriva_tags.visible_columns: {
            "*": [
                "RID",
                dataset_version_dataset_source,
                dataset_version_label_source,
                "Description",
                "Minid",
                dataset_version_producer_source,
                "RCT",
                "RMT",
                [schema, "Dataset_Version_RCB_fkey"],
            ],
            "detailed": [
                "RID",
                dataset_version_dataset_source,
                dataset_version_label_source,
                "Description",
                "Minid",
                "Snapshot",
                dataset_version_producer_source,
                "RCT",
                "RMT",
                [schema, "Dataset_Version_RCB_fkey"],
                [schema, "Dataset_Version_RMB_fkey"],
            ],
        },
        # Surface which executions CONSUMED this exact version (the input edge).
        # Previously hidden entirely (`{"*": []}`), which dead-ended the
        # "who used this version?" provenance question.
        deriva_tags.visible_foreign_keys: {
            "detailed": [
                {
                    "source": [
                        {"inbound": [schema, "Dataset_Execution_Dataset_Version_fkey"]},
                        {"outbound": [schema, "Dataset_Execution_Execution_fkey"]},
                        "RID",
                    ],
                    "markdown_name": "Consumed By Executions",
                },
            ]
        },
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
    "feature_annotation",
    "generate_annotation",
    "vocabulary_annotation",
]
