"""DerivaML schema creation and catalog initialization.

Provides functions for creating and resetting the ``deriva-ml`` schema in an
ERMrest catalog. The main entry points are:

- ``create_ml_schema``: Create (or DROP+recreate) the core ``deriva-ml`` schema
  with all required tables, vocabularies, and FK relationships.
  **WARNING**: Drops the existing schema with CASCADE if it already exists.
- ``initialize_ml_schema``: Populate vocabulary tables with standard terms
  after schema creation.
- ``create_ml_catalog``: Create a brand-new catalog and install the schema.
"""

import subprocess
import sys
from collections.abc import Sequence
from importlib.resources import files
from typing import Any, Optional

from deriva.core import DerivaServer, ErmrestCatalog, get_credential
from deriva.core.ermrest_model import Model, Schema, Table
from deriva.core.typed import (
    AssetTableDef,
    BuiltinType,
    ColumnDef,
    ForeignKeyDef,
    KeyDef,
    SchemaDef,
    TableDef,
    VocabularyTableDef,
)

from deriva_ml.core.definitions import MLTable, MLVocab
from deriva_ml.core.exceptions import DerivaMLConfigurationError
from deriva_ml.core.logging_config import get_logger
from deriva_ml.schema.annotations import asset_annotation, generate_annotation

logger = get_logger(__name__)


def directory_dataset_table_def(schema_name: str) -> TableDef:
    """TableDef for the Directory_Dataset satellite table.

    Shared by ``create_dataset_table`` (fresh catalogs) and the
    ``add_directory_dataset_table`` migration (existing catalogs) so both
    produce an identical table.

    Args:
        schema_name: The deriva-ml schema name (for the FK's referenced_schema).

    Returns:
        TableDef: definition of the Directory_Dataset table.
    """
    return TableDef(
        name="Directory_Dataset",
        comment=(
            "Source folder a directory dataset (auto-created by add_files) "
            "represents, as a path relative to the ingest root. One row per "
            "directory dataset; absent for datasets not built from a "
            "directory tree."
        ),
        columns=[
            ColumnDef("Dataset", BuiltinType.text, comment="RID of the directory dataset."),
            ColumnDef(
                "Path",
                BuiltinType.text,
                comment=(
                    "Source directory this dataset represents, relative to "
                    "the ingest root. The ingest root stores '.'."
                ),
            ),
        ],
        keys=[KeyDef(columns=["Dataset"])],
        foreign_keys=[
            ForeignKeyDef(
                columns=["Dataset"],
                referenced_schema=schema_name,
                referenced_table="Dataset",
                referenced_columns=["RID"],
            ),
        ],
    )


def create_dataset_table(
    schema: Schema,
    execution_table: Table,
    project_name: str,
    dataset_annotation: Optional[dict] = None,
    version_annotation: Optional[dict] = None,
) -> Table:
    """Create the Dataset table and its supporting vocabulary + association tables.

    Side-effect graph (six tables created on ``schema`` in order):

    1. ``Dataset`` — the main dataset table; columns
       ``Description`` (markdown) and ``Deleted`` (boolean).
    2. ``Dataset_Type`` — controlled-vocabulary table for dataset
       types. ``curie_template`` is ``{project_name}:{{RID}}``.
    3. ``Dataset_Dataset_Type`` — association table linking
       ``Dataset`` ↔ ``Dataset_Type`` (built by
       :meth:`Table.define_association`).
    4. ``Dataset_Version`` — produced by
       :func:`define_table_dataset_version` and referenced by
       ``Dataset.Version`` (an outbound FK; ``True`` selects the
       active version per row).
    5. ``Dataset_Dataset`` — self-association for nested datasets
       (``Nested_Dataset`` ↔ parent ``Dataset``).
    6. ``Dataset_Execution`` — association table linking
       ``Dataset`` ↔ ``Execution`` for execution-output datasets.

    FK edges added (in addition to the association ones above):
    ``Dataset.Version`` → ``Dataset_Version.RID``.

    Args:
        schema: The schema (typically the deriva-ml schema) where
            the six tables are created.
        execution_table: The pre-existing ``Execution`` table; used
            only as the right-hand side of the ``Dataset_Execution``
            association.
        project_name: Project name used as the CURIE namespace for
            the ``Dataset_Type`` vocabulary (``{project_name}:{RID}``).
        dataset_annotation: Optional Chaise annotation bundle for
            the ``Dataset`` table. Pass
            ``generate_annotation(model, schema)["dataset_annotation"]``
            for the canonical shape.
        version_annotation: Optional Chaise annotation bundle for
            the ``Dataset_Version`` table.

    Returns:
        The created ``Dataset`` table. The five sibling tables are
        also created on ``schema`` but not returned — the caller
        can reach them via ``schema.tables[...]``.
    """
    dataset_table = schema.create_table(
        TableDef(
            name=MLTable.dataset,
            comment=(
                "A versioned collection of catalog rows that an execution "
                "consumed or produced. Datasets are typed (see "
                "`Dataset_Type`), carry a current version (`Dataset_Version`), "
                "may contain other datasets as members (`Dataset_Dataset`), "
                "and can be downloaded as a self-describing BDBag. "
                "Soft-deleted rows are marked via the `Deleted` flag rather "
                "than physically removed, so historical references in "
                "executions and citations remain valid."
            ),
            columns=[
                ColumnDef(
                    "Description",
                    BuiltinType.markdown,
                    comment=(
                        "Human-readable description of what this dataset represents. Rendered as Markdown in Chaise."
                    ),
                ),
                ColumnDef(
                    "Deleted",
                    BuiltinType.boolean,
                    comment=(
                        "Soft-delete flag. When true, the dataset is hidden "
                        "from default listings but retained so existing "
                        "execution provenance and citations remain resolvable."
                    ),
                ),
            ],
            annotations=dataset_annotation if dataset_annotation is not None else {},
        )
    )

    dataset_type = schema.create_table(
        VocabularyTableDef(
            name=MLVocab.dataset_type,
            curie_template=f"{project_name}:{{RID}}",
            comment=(
                "Controlled vocabulary classifying datasets by role "
                "(`Training`, `Testing`, `Validation`, `Complete`, `Split`, "
                "`Labeled`, `Unlabeled`, `File`). Domain-specific dataset "
                "categories belong in user vocabularies, not here."
            ),
        )
    )

    # Association table for Dataset <-> Dataset_Type
    schema.create_table(
        Table.define_association(
            associates=[
                ("Dataset", dataset_table),
                (MLVocab.dataset_type, dataset_type),
            ],
            comment=(
                "Many-to-many tag assignments between datasets and dataset "
                "types. A dataset can carry multiple types (for example "
                "`Training` + `Labeled`)."
            ),
        )
    )

    dataset_version = schema.create_table(define_table_dataset_version(schema.name, version_annotation))
    dataset_table.create_reference(("Version", True, dataset_version))

    # Nested datasets.
    schema.create_table(
        Table.define_association(
            associates=[("Dataset", dataset_table), ("Nested_Dataset", dataset_table)],
            comment=(
                "Self-association expressing dataset nesting: the `Dataset` "
                "column is the parent, `Nested_Dataset` is a member. Used to "
                "model train/test/validation splits as nested children of a "
                "parent collection."
            ),
        )
    )
    dataset_execution = schema.create_table(
        Table.define_association(
            associates=[("Dataset", dataset_table), ("Execution", execution_table)],
            comment=(
                "Input-only association between datasets and executions: each "
                "row records that an execution *consumed* a dataset. Output "
                "edges (which execution *produced* a dataset version) live in "
                "`Dataset_Version.Execution`, not here. The optional "
                "`Dataset_Version` FK pins the exact version that was consumed."
            ),
        )
    )
    # Nullable FK recording which Dataset_Version this input edge consumed.
    # The (base_name, nullok, target) tuple form creates both the
    # `Dataset_Version` column and its FK to the Dataset_Version table in one
    # step — mirrors `dataset_table.create_reference(("Version", True, ...))`
    # above. NULL means the consumed version is unknown (e.g. legacy rows).
    dv_cols, _ = dataset_execution.create_reference(("Dataset_Version", True, dataset_version))
    dv_cols[0].alter(comment="RID of the Dataset_Version consumed by this input edge (NULL if unknown).")

    # Directory_Dataset: satellite recording the source folder a directory
    # dataset (created by add_files) represents (see directory_dataset_table_def).
    schema.create_table(directory_dataset_table_def(schema.name))

    return dataset_table


def define_table_dataset_version(sname: str, annotation: Optional[dict] = None) -> TableDef:
    """Define the dataset version table in the specified schema.

    Args:
        sname: The schema name where the table should be created.
        annotation: Optional annotation dictionary for the table.

    Returns:
        A TableDef for the dataset version table.
    """
    return TableDef(
        name=MLTable.dataset_version,
        comment=(
            "Version history for a dataset. Each row pins a (Dataset, "
            "Version) pair, optionally with a catalog Snapshot for released "
            "versions and a MINID URL for the materialized bag. The current "
            "version of a dataset is selected by the outbound FK "
            "`Dataset.Version`."
        ),
        columns=[
            ColumnDef(
                name="Version",
                type=BuiltinType.text,
                default="0.1.0",
                comment=(
                    "PEP 440 version label for this dataset version. "
                    "Released rows carry `MAJOR.MINOR.PATCH` (e.g. `0.4.0`). "
                    "Dev rows carry `<last_release>.post1.devN` "
                    "(e.g. `0.4.0.post1.dev3`) to denote drift since the "
                    "last release. The default `0.1.0` applies to the "
                    "initial release row created at dataset creation time."
                ),
            ),
            ColumnDef(
                "Description",
                BuiltinType.markdown,
                comment=(
                    "Release notes for this version: what changed since the "
                    "previous version, why the bump was made, anything a "
                    "consumer downloading this version should know."
                ),
            ),
            ColumnDef("Dataset", BuiltinType.text, comment="RID of dataset"),
            ColumnDef(
                "Execution",
                BuiltinType.text,
                comment=(
                    "RID of the execution that produced this version (NULL "
                    "for the initial release row, which has no producing "
                    "execution)."
                ),
            ),
            ColumnDef("Minid", BuiltinType.text, comment="URL to MINID for dataset"),
            ColumnDef(
                name="Minid_Spec_Hash",
                type=BuiltinType.text,
                comment="SHA-256 hash of the download spec used to generate the MINID bag. "
                "Used to detect stale MINIDs when the schema or traversal paths change.",
            ),
            ColumnDef(
                name="Snapshot",
                type=BuiltinType.text,
                comment=(
                    "Catalog snapshot ID this version row pins. Populated for "
                    "released rows (the snapshot stamped at release time). "
                    "`NULL` on dev rows, denoting that the row tracks live "
                    "catalog state with no pinned snapshot — the bag this "
                    "dataset would download right now is whatever the "
                    "catalog has at request time."
                ),
            ),
        ],
        annotations=annotation if annotation else {},
        keys=[KeyDef(columns=["Dataset", "Version"])],
        foreign_keys=[
            ForeignKeyDef(
                columns=["Dataset"],
                referenced_schema=sname,
                referenced_table="Dataset",
                referenced_columns=["RID"],
            ),
            ForeignKeyDef(
                columns=["Execution"],
                referenced_schema=sname,
                referenced_table="Execution",
                referenced_columns=["RID"],
            ),
        ],
    )


def create_execution_table(schema: Schema, annotation: Optional[dict] = None) -> Table:
    """Create the execution table in the specified schema.

    Args:
        schema: The schema where the table should be created.
        annotation: Optional annotation dictionary for the table.

    Returns:
        The created Table object.
    """
    annotation = annotation if annotation is not None else {}
    execution = schema.create_table(
        TableDef(
            name=MLTable.execution,
            comment=(
                "One run of a Workflow against specific input Datasets, "
                "producing output Datasets, Features, and Assets. Carries "
                "the execution state machine (`Status`), provenance edges "
                "to its inputs and outputs (via `Dataset_Execution`, "
                "`Execution_Asset_Execution`, `Execution_Metadata_Execution`), "
                "and timing breakdown across the three lifecycle phases "
                "(`Download_Duration`, `Execution_Duration`, `Upload_Duration`)."
            ),
            columns=[
                ColumnDef(
                    "Workflow",
                    BuiltinType.text,
                    comment=(
                        "FK to the Workflow row whose URL + git commit hash "
                        "uniquely identifies the code that ran this execution."
                    ),
                ),
                ColumnDef(
                    "Description",
                    BuiltinType.markdown,
                    comment=(
                        "Human-readable description of what this execution "
                        "is doing — purpose, hyperparameters worth calling "
                        "out, anything a reader scanning a list of "
                        "executions should know. Rendered as Markdown in "
                        "Chaise."
                    ),
                ),
                # Three duration columns, each measuring a distinct phase:
                #   Execution_Duration — algorithm time inside the `with`
                #     block (start of __enter__ → end of __exit__).
                #     Renamed from "Duration" 2026-05-19 so all three
                #     columns share the <Phase>_Duration pattern.
                #   Download_Duration — init / dataset+asset download
                #     time (_initialize_execution).
                #   Upload_Duration   — commit_output_assets time
                #     (bag commit + Hatrac PUTs).
                ColumnDef(
                    "Execution_Duration",
                    BuiltinType.text,
                    comment=(
                        "Wall-clock time spent in the algorithm phase — "
                        "from the start of the `with ml.create_execution()` "
                        "block to its exit, excluding download and upload. "
                        "Stored as an ISO 8601 duration string."
                    ),
                ),
                ColumnDef(
                    "Download_Duration",
                    BuiltinType.text,
                    comment=(
                        "Wall-clock time spent in the initialization phase, "
                        "which downloads input datasets and assets from the "
                        "catalog into the local working directory. Stored "
                        "as an ISO 8601 duration string."
                    ),
                ),
                ColumnDef(
                    "Upload_Duration",
                    BuiltinType.text,
                    comment=(
                        "Wall-clock time spent in the commit phase — "
                        "writing output bags, uploading output assets to "
                        "Hatrac, and finalizing catalog rows. Stored as an "
                        "ISO 8601 duration string."
                    ),
                ),
                ColumnDef(
                    "Status",
                    BuiltinType.text,
                    comment=(
                        "Current state of the execution state machine. FK "
                        "to `Execution_Status` (`Created`, `Running`, "
                        "`Stopped`, `Failed`, `Pending_Upload`, `Uploaded`, "
                        "`Aborted`)."
                    ),
                ),
                ColumnDef(
                    "Status_Detail",
                    BuiltinType.text,
                    comment=(
                        "Free-form context for the current Status — usually "
                        "the most recent stage message, error text on "
                        "`Failed`, or progress indicator on long-running "
                        "phases."
                    ),
                ),
            ],
            foreign_keys=[
                ForeignKeyDef(
                    columns=["Workflow"],
                    referenced_schema=schema.name,
                    referenced_table="Workflow",
                    referenced_columns=["RID"],
                ),
                ForeignKeyDef(
                    columns=["Status"],
                    referenced_schema=schema.name,
                    referenced_table=MLVocab.execution_status,
                    referenced_columns=["Name"],
                ),
            ],
            annotations=annotation,
        )
    )

    # Nested executions - allows grouping executions hierarchically
    # (e.g., a sweep/multirun as parent with individual runs as children)
    schema.create_table(
        Table.define_association(
            associates=[("Execution", execution), ("Nested_Execution", execution)],
            comment=(
                "Self-association expressing execution nesting: the "
                "`Execution` column is the parent (typically a sweep or "
                "multirun controller), `Nested_Execution` is one of its "
                "children. `Sequence` orders children within a parent for "
                "sequential runs; NULL means parallel siblings."
            ),
            metadata=[
                ColumnDef(
                    name="Sequence",
                    type=BuiltinType.int4,
                    nullok=True,
                    comment="Order of nested execution (null if parallel)",
                ).to_dict()  # Convert to dict for Table.define_association()
            ],
        )
    )
    return execution


def create_asset_table(
    schema: Schema,
    asset_name: str,
    execution_table: Table,
    asset_type_table: Table,
    asset_role_table: Table,
    use_hatrac: bool = True,
    comment: Optional[str] = None,
    additional_columns: Sequence[ColumnDef] = (),
) -> Table:
    """Create an asset table with associated type and execution associations.

    Args:
        schema: The schema where the table should be created.
        asset_name: Name for the asset table.
        execution_table: The execution table for association.
        asset_type_table: The asset type vocabulary table.
        asset_role_table: The asset role vocabulary table.
        use_hatrac: When ``True`` (default) the asset table's URL
            column is wired with the Hatrac upload template
            ``/hatrac/metadata/{{MD5}}.{{Filename}}`` — Chaise's
            file-upload UI will deposit bytes into Hatrac at that
            path. When ``False`` (used by the ``File`` table) the
            template is omitted; the URL column carries a plain
            string and no Hatrac upload UI is wired up. Previously
            this parameter was accepted but silently ignored, so
            the ``File`` table got the same Hatrac template as
            ``Execution_Asset``.
        comment: Optional table-level comment describing the
            purpose of this specific asset table. Used to
            distinguish the three asset tables (Execution_Asset,
            Execution_Metadata, File) which otherwise share the
            same generic shape.
        additional_columns: Optional domain-specific columns appended
            to the standard hatrac shape (``URL`` / ``Filename`` /
            ``Length`` / ``MD5`` / ``Description``). Used by the
            public ``DerivaML.create_asset_table`` wrapper; the
            bootstrap call sites leave it empty.

    Returns:
        The created asset Table object.
    """
    # ``hatrac_template=None`` (the AssetTableDef default) means
    # "no Hatrac upload template" — the URL column is a plain
    # string. The Hatrac template is wired up only when the
    # caller asks for it via ``use_hatrac=True``.
    hatrac_template = "/hatrac/metadata/{{MD5}}.{{Filename}}" if use_hatrac else None
    asset_table = schema.create_table(
        AssetTableDef(
            schema_name=schema.name,
            name=asset_name,
            columns=list(additional_columns),
            hatrac_template=hatrac_template,
            comment=comment,
        )
    )
    schema.create_table(
        Table.define_association(
            [
                (asset_name, asset_table),
                ("Asset_Type", asset_type_table),
            ],
            comment=(
                f"Many-to-many tag assignments between {asset_name} rows "
                f"and Asset_Type vocabulary terms. An asset can carry "
                f"multiple types (for example `Model_File` + `Output_File`)."
            ),
        )
    )

    atable = schema.create_table(
        Table.define_association(
            [
                (asset_name, asset_table),
                ("Execution", execution_table),
            ],
            comment=(
                f"Many-to-many association between {asset_name} rows and "
                f"executions. Carries an Asset_Role FK indicating whether "
                f"the asset was an Input to or an Output of the execution."
            ),
        )
    )
    atable.create_reference(asset_role_table)
    asset_annotation(asset_table)
    return asset_table


def create_workflow_table(schema: Schema, annotations: Optional[dict[str, Any]] = None) -> Table:
    """Create the workflow table in the specified schema.

    Args:
        schema: The schema where the table should be created.
        annotations: Optional annotation dictionary for the table.

    Returns:
        The created Table object.
    """
    workflow_table = schema.create_table(
        TableDef(
            name=MLTable.workflow,
            comment=(
                "A versioned reference to the code that knows how to run "
                "an ML step. Workflows are content-addressed by `(URL, "
                "Checksum)` — the same script at the same git commit "
                "always resolves to the same Workflow row. Workflows are "
                "typed via the `Workflow_Workflow_Type` association "
                "(`Training`, `Prediction`, `Analysis`, ...). Each "
                "execution links back to exactly one Workflow row, making "
                "the producing code recoverable for any result in the "
                "catalog."
            ),
            columns=[
                ColumnDef(
                    "Name",
                    BuiltinType.text,
                    comment=(
                        "Short human-readable name for the workflow. Used "
                        "in execution listings and citations; not required "
                        "to be unique."
                    ),
                ),
                ColumnDef(
                    "Description",
                    BuiltinType.markdown,
                    comment=(
                        "Longer description of what the workflow does, "
                        "what inputs it expects, and what outputs it "
                        "produces. Rendered as Markdown in Chaise."
                    ),
                ),
                ColumnDef(
                    "URL",
                    BuiltinType.ermrest_uri,
                    comment=(
                        "Location of the workflow code — typically a "
                        "GitHub URL pinned to a specific commit hash, or "
                        "a notebook URL. Combined with `Checksum` to "
                        "content-address the workflow."
                    ),
                ),
                ColumnDef(
                    "Checksum",
                    BuiltinType.text,
                    comment=(
                        "Git commit hash (or other content hash) of the "
                        "code at `URL`. Together with `URL` this uniquely "
                        "identifies the executable code; two executions "
                        "of the same Workflow row are guaranteed to have "
                        "run identical source."
                    ),
                ),
                ColumnDef(
                    "Version",
                    BuiltinType.text,
                    comment=(
                        "Semantic version string of the workflow (e.g. "
                        "`1.2.0`). Independent from `Checksum`; used for "
                        "human-readable release tracking when the "
                        "workflow code is itself published as a "
                        "versioned package."
                    ),
                ),
            ],
            annotations=annotations if annotations else {},
        )
    )

    workflow_type_table = schema.create_table(
        VocabularyTableDef(
            name=MLVocab.workflow_type,
            curie_template=f"{schema.name}:{{RID}}",
            comment=(
                "Controlled vocabulary classifying workflows by their "
                "role in an ML pipeline (`Training`, `Prediction`, "
                "`Analysis`, `Feature_Creation`, `Data_Cleaning`, ...). "
                "A workflow can carry multiple types via the "
                "`Workflow_Workflow_Type` association."
            ),
        )
    )

    # Association table for Workflow <-> Workflow_Type
    schema.create_table(
        Table.define_association(
            associates=[
                ("Workflow", workflow_table),
                (MLVocab.workflow_type, workflow_type_table),
            ],
            comment=(
                "Many-to-many tag assignments between workflows and "
                "workflow types. A workflow can carry multiple types "
                "(for example `Training` + `Feature_Creation`)."
            ),
        )
    )

    return workflow_table


def create_ml_schema(
    catalog: ErmrestCatalog,
    schema_name: str = "deriva-ml",
    project_name: Optional[str] = None,
):
    """Create or recreate the DerivaML schema in the given catalog.

    WARNING: If the schema already exists, it will be DROPPED with CASCADE,
    destroying all data in the schema. The bag-pipeline clone path
    (:func:`~deriva_ml.catalog.clone_via_bag.clone_via_bag`) **does not**
    call this function — it loads catalog content via
    :class:`BagCatalogLoader` instead. Direct callers (e.g., fresh-catalog
    bootstrap in the model template) own the destructive-call contract.

    Args:
        catalog: An ErmrestCatalog connection to the target catalog.
        schema_name: Name of the schema to create (default: "deriva-ml").
        project_name: Display name for the project. Defaults to schema_name.
    """
    project_name = project_name or schema_name

    model = catalog.getCatalogModel()
    if model.schemas.get(schema_name):
        logger.warning(f"Dropping existing schema '{schema_name}' with CASCADE")
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

    schema = model.create_schema(SchemaDef(name=schema_name, annotations=annotations["schema_annotation"]))

    # Create workflow and execution table.

    schema.create_table(
        VocabularyTableDef(
            name=MLVocab.feature_name,
            curie_template=f"{project_name}:{{RID}}",
            comment=(
                "Controlled vocabulary of feature names. Every Feature "
                "defined on a target table draws its name from a term in "
                "this vocabulary, so the name space of features is "
                "audited and searchable rather than free-form."
            ),
        )
    )
    asset_type_table = schema.create_table(
        VocabularyTableDef(
            name=MLVocab.asset_type,
            curie_template=f"{project_name}:{{RID}}",
            comment=(
                "Controlled vocabulary classifying assets by purpose "
                "(`Model_File`, `Hydra_Config`, `Metrics_File`, "
                "`Notebook_Output`, ...). Used as the FK target for "
                "Asset_Type tagging on every asset table in the catalog "
                "(`Execution_Asset_Asset_Type`, "
                "`Execution_Metadata_Asset_Type`, `File_Asset_Type`, plus "
                "any domain-specific asset tables)."
            ),
        )
    )
    asset_role_table = schema.create_table(
        VocabularyTableDef(
            name=MLVocab.asset_role,
            curie_template=f"{project_name}:{{RID}}",
            comment=(
                "Controlled vocabulary distinguishing inputs from outputs "
                "(`Input`, `Output`). Carried on every "
                "<asset>_Execution association row to record which "
                "direction the asset moves relative to the execution."
            ),
        )
    )
    schema.create_table(
        VocabularyTableDef(
            name=MLVocab.execution_status,
            curie_template=f"{project_name}:{{RID}}",
            comment=(
                "Controlled vocabulary describing the lifecycle state of "
                "an Execution: `Created` → `Running` → `Stopped` → "
                "`Pending_Upload` → `Uploaded`, with `Failed` and "
                "`Aborted` as terminal error states. Managed by the "
                "execution state machine; do not extend without coordinating "
                "with the deriva-ml execution lifecycle."
            ),
        )
    )

    create_workflow_table(schema, annotations["workflow_annotation"])
    execution_table = create_execution_table(schema, annotations["execution_annotation"])
    dataset_table = create_dataset_table(
        schema,
        execution_table,
        project_name,
        annotations["dataset_annotation"],
        annotations["dataset_version_annotation"],
    )

    create_asset_table(
        schema,
        MLTable.execution_metadata,
        execution_table,
        asset_type_table,
        asset_role_table,
        comment=(
            "Asset table for files describing an execution's environment "
            "and configuration — Hydra configs, runtime info, DerivaML "
            "execution configuration JSON. Stored in Hatrac. Distinguished "
            "from `Execution_Asset` (which holds the execution's data "
            "outputs) by purpose, not by file shape."
        ),
    )

    create_asset_table(
        schema,
        MLTable.execution_asset,
        execution_table,
        asset_type_table,
        asset_role_table,
        comment=(
            "Asset table for files produced or consumed by an execution as "
            "data — trained model weights, prediction CSVs, plots, "
            "notebook outputs. Stored in Hatrac. Distinguished from "
            "`Execution_Metadata` (which holds environment/configuration) "
            "by purpose, not by file shape."
        ),
    )

    # File table
    file_table = create_asset_table(
        schema,
        MLTable.file,
        execution_table,
        asset_type_table,
        asset_role_table,
        use_hatrac=False,
        comment=(
            "Asset table for files that live outside Hatrac — typically "
            "external URLs pointing at data the catalog references but "
            "does not host. Same row shape as the Hatrac-backed asset "
            "tables; the difference is that `URL` is a plain string with "
            "no upload template wired up."
        ),
    )
    # And make Files be part of a dataset.
    schema.create_table(
        Table.define_association(
            associates=[
                ("Dataset", dataset_table),
                (MLTable.file, file_table),
            ],
            comment=(
                "Many-to-many membership of File rows in Datasets. A file "
                "can belong to multiple datasets; a dataset can contain "
                "multiple files."
            ),
        )
    )

    initialize_ml_schema(model, schema_name)


def initialize_ml_schema(model: Model, schema_name: str = "deriva-ml"):
    """Initialize the ML schema vocabulary tables with required terms.

    Populates Asset_Type, Asset_Role, Dataset_Type, Workflow_Type, and
    Execution_Status vocabulary tables with their standard terms. Safe
    to call on catalogs that already have some or all terms — existing
    terms are skipped (the helper inserts only names not already
    present).

    Term-selection principle:
        Every term seeded here describes a **platform-level** concept
        — a workflow shape (``Training``, ``Prediction``), an
        execution state (``Running``, ``Uploaded``), an asset role
        (``Input``, ``Output``), an asset purpose (``Model_File``,
        ``Hydra_Config``), or a dataset role (``Training``,
        ``Validation``, ``Complete``). **Domain-specific terms must
        not appear here** — specific model architectures
        (``VGG19``, ``RETFound``), research-area categories
        (``Multimodal``), or dataset/asset names tied to a single
        project all belong in user vocabularies added at the
        catalog level after schema creation.

        A platform that ships opinionated domain defaults pushes
        every downstream user toward those defaults whether or not
        they fit. The defaults seeded here should be the smallest
        set that lets a fresh DerivaML catalog be usable for any
        ML workflow.

    Args:
        model: The ERMrest model to add terms to.
        schema_name: The name of the ML schema. Defaults to "deriva-ml".
    """

    catalog = model.catalog
    pb = catalog.getPathBuilder().schemas[schema_name]

    def _ensure_terms(table_name: str, terms: list[dict]) -> None:
        """Insert terms that don't already exist in a vocabulary table."""
        table = pb.tables[table_name]
        existing = {row["Name"] for row in table.entities()}
        missing = [t for t in terms if t["Name"] not in existing]
        if missing:
            table.insert(missing, defaults={"ID", "URI"})

    _ensure_terms(
        MLVocab.asset_type,
        [
            {"Name": "Execution_Config", "Description": "Configuration File for execution metadata"},
            {"Name": "Runtime_Env", "Description": "Information about the runtime environment"},
            {
                "Name": "Hydra_Config",
                "Description": "Hydra YAML configuration file (config.yaml, overrides.yaml, hydra.yaml)",
            },
            {
                "Name": "Deriva_Config",
                "Description": "DerivaML execution configuration (configuration.json with datasets, assets, workflow)",
            },
            {
                "Name": "Metrics_File",
                "Description": "Training-metric log file (typically JSONL, one record per evaluation point — epoch, step, or eval cycle).",
            },
            {"Name": "Execution_Metadata", "Description": "Information about the execution environment"},
            {"Name": "Execution_Asset", "Description": "A file generated by an execution"},
            {"Name": "File", "Description": "A file that is not managed by Hatrac"},
            {"Name": "Input_File", "Description": "A file input to an execution."},
            {"Name": "Output_File", "Description": "A file output from an execution."},
            {"Name": "Model_File", "Description": "The ML model."},
            {
                "Name": "Notebook_Output",
                "Description": "A Jupyter notebook with output cells filled from an execution.",
            },
        ],
    )

    _ensure_terms(
        MLVocab.asset_role,
        [
            {"Name": "Input", "Description": "Asset used for input of an execution."},
            {"Name": "Output", "Description": "Asset used for output of an execution."},
        ],
    )

    _ensure_terms(
        MLVocab.dataset_type,
        [
            {"Name": "Complete", "Description": "A dataset containing all available records of a given type."},
            {"Name": "File", "Description": "A dataset that contains file assets."},
            {
                "Name": "Directory",
                "Description": (
                    "A dataset auto-created by add_files to mirror an ingested source directory "
                    "structure; nested Directory datasets reflect the source folder hierarchy."
                ),
            },
            {"Name": "Training", "Description": "A dataset subset used for model training."},
            {"Name": "Testing", "Description": "A dataset subset used for model testing/evaluation."},
            {"Name": "Validation", "Description": "A dataset subset used for model validation during training."},
            {"Name": "Split", "Description": "A dataset that contains nested dataset splits."},
            {"Name": "Labeled", "Description": "A dataset containing records with ground truth labels."},
            {"Name": "Unlabeled", "Description": "A dataset containing records without ground truth labels."},
        ],
    )

    _ensure_terms(
        MLVocab.workflow_type,
        [
            {"Name": "Training", "Description": "Model training and fine-tuning workflows."},
            {
                "Name": "Testing",
                "Description": "Workflows that evaluate model performance on held-out data, computing metrics such as accuracy, AUC, confusion matrices, and per-class statistics.",
            },
            {
                "Name": "Prediction",
                "Description": "Workflows that apply a trained model to new data to generate predictions, probability scores, or classification labels.",
            },
            {
                "Name": "Feature_Creation",
                "Description": "Workflows that extract or engineer features from raw data, producing structured feature values linked to source records.",
            },
            {
                "Name": "Visualization",
                "Description": "Workflows that produce visual analyses of data or model results, including plots, charts, and summary dashboards.",
            },
            {
                "Name": "Analysis",
                "Description": "Computational analysis workflows that combine and analyze data from multiple sources without training a model.",
            },
            {
                "Name": "Ingest",
                "Description": "Workflows that load external data into the catalog, including file upload, record creation, and initial metadata population.",
            },
            {
                "Name": "Data_Cleaning",
                "Description": "Workflows that clean and preprocess raw data, including standardizing formats, handling missing values, and filtering invalid records.",
            },
            {"Name": "Dataset_Management", "Description": "Workflows that create, split, version, or manage datasets."},
        ],
    )

    _ensure_terms(
        MLVocab.execution_status,
        [
            {"Name": "Created", "Description": "Execution row has been created; work has not started."},
            {"Name": "Running", "Description": "Execution algorithm is actively running."},
            {"Name": "Stopped", "Description": "Algorithm finished successfully; output assets not yet uploaded."},
            {"Name": "Failed", "Description": "Execution encountered an unrecoverable error."},
            {
                "Name": "Pending_Upload",
                "Description": "Algorithm succeeded; asset upload to the catalog is in progress.",
            },
            {
                "Name": "Uploaded",
                "Description": "Execution ran to success and all outputs are persisted to the catalog.",
            },
            {"Name": "Aborted", "Description": "Execution was canceled by the user before reaching a terminal state."},
        ],
    )

    _ensure_sentinels(pb)


def _ensure_sentinels(pb) -> None:
    """Idempotently seed the three unknown-provenance sentinels.

    Seeds a Workflow, a File, and an Execution row that represent "provenance
    is explicitly unknown" (see ``deriva_ml.core.constants``). They are
    bootstrap substrate — exempt from the provenance contract's producer /
    completeness obligations. Safe to re-run: each is located by its reserved
    identifier and inserted only when absent.

    Order matters: the Workflow must exist before the Execution (which has a
    Workflow FK).

    Args:
        pb: The deriva-ml schema path-builder (``catalog.getPathBuilder().schemas[schema_name]``).
    """
    from deriva_ml.core.constants import (
        SENTINEL_EXECUTION_DESCRIPTION,
        SENTINEL_FILE_MD5,
        SENTINEL_FILE_URL,
        SENTINEL_WORKFLOW_CHECKSUM,
        SENTINEL_WORKFLOW_URL,
    )

    # --- Workflow sentinel (located by reserved URL) ---
    wf_table = pb.tables["Workflow"]
    wf_existing = [r for r in wf_table.filter(wf_table.URL == SENTINEL_WORKFLOW_URL).entities()]
    if wf_existing:
        sentinel_workflow_rid = wf_existing[0]["RID"]
    else:
        sentinel_workflow_rid = wf_table.insert(
            [
                {
                    "Name": "Unknown Provenance",
                    "Description": "Sentinel workflow for artifacts/executions of unknown provenance.",
                    "URL": SENTINEL_WORKFLOW_URL,
                    "Checksum": SENTINEL_WORKFLOW_CHECKSUM,
                }
            ]
        )[0]["RID"]

    # --- File sentinel (located by reserved URL) ---
    file_table = pb.tables["File"]
    file_existing = [r for r in file_table.filter(file_table.URL == SENTINEL_FILE_URL).entities()]
    if not file_existing:
        file_table.insert(
            [
                {
                    "URL": SENTINEL_FILE_URL,
                    "MD5": SENTINEL_FILE_MD5,
                    "Description": "Sentinel file representing an unknown / unrecorded input.",
                    "Length": 0,
                }
            ]
        )

    # --- Execution sentinel (located by reserved Description; needs the
    #     Workflow FK above). Status=Aborted — it is not a real run. ---
    exe_table = pb.tables["Execution"]
    exe_existing = [
        r for r in exe_table.filter(exe_table.Description == SENTINEL_EXECUTION_DESCRIPTION).entities()
    ]
    if not exe_existing:
        exe_table.insert(
            [
                {
                    "Workflow": sentinel_workflow_rid,
                    "Description": SENTINEL_EXECUTION_DESCRIPTION,
                    "Status": "Aborted",
                    "Status_Detail": "Bootstrap sentinel: artifacts attributed here have no real producing execution.",
                }
            ]
        )


def create_ml_catalog(
    hostname: str,
    project_name: str,
    catalog_alias: str | None = None,
) -> ErmrestCatalog:
    """Create a new DerivaML catalog with all ML schema tables.

    Args:
        hostname: Server hostname (e.g., "localhost", "www.eye-ai.org").
        project_name: Name for the project, becomes the domain schema name.
        catalog_alias: Optional alias name for the catalog. If provided, creates
            an alias that points to the new catalog, allowing access via the
            alias name instead of the numeric catalog ID.

    Returns:
        The created ErmrestCatalog instance.

    Example:
        >>> # Create catalog with alias
        >>> catalog = create_ml_catalog(  # doctest: +SKIP
        ...     "localhost", "my_project", catalog_alias="my-project"
        ... )
        >>> # Now accessible as both /ermrest/catalog/<id> and /ermrest/catalog/my-project
    """
    server = DerivaServer("https", hostname, credentials=get_credential(hostname))
    catalog = server.create_ermrest_catalog()
    model = catalog.getCatalogModel()
    model.configure_baseline_catalog()
    policy_file = files("deriva_ml.schema").joinpath("policy.json")

    # The deriva-ml schema must exist *before* acl_config runs, otherwise
    # policy.json's per-table binding rule (which matches all non-``public``
    # schemas) has nothing to bind to, and ``row_owner_guard`` is silently
    # applied to zero tables. The failure mode is invisible until a
    # non-curator user PATCHes ``Execution_Metadata`` and gets HTTP 403.
    # See ``tests/schema/test_acl_application.py``.
    create_ml_schema(catalog, project_name=project_name)

    # Apply the catalog ACL policy via deriva.config.acl_config. We invoke
    # it as a module via the *current* interpreter (sys.executable -m ...)
    # rather than through PATH lookup of the deriva-acl-config console
    # script, because:
    #   - PATH may pick up a system-Python install with a stale shebang or
    #     incompatible Python version (deriva-py requires 3.11+ for
    #     `typing.Self`); a 3.10 system install crashes on import.
    #   - Stale venv shebangs (after a venv move/rename) break the
    #     console-script entry point even when the package is installed.
    # check=True + capture_output=True turns silent failures into a loud
    # DerivaMLConfigurationError that surfaces in the catalog-creation
    # stack trace, instead of leaving a half-configured catalog with no
    # ACLs (the row-level update bindings never get applied, every later
    # asset upload that updates Execution_Metadata then fails with HTTP
    # 403, and the user has no idea why).
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "deriva.config.acl_config",
                "--host",
                catalog.deriva_server.server,
                "--config-file",
                str(policy_file),
                catalog.catalog_id,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise DerivaMLConfigurationError(
            f"Failed to apply ACL policy to catalog {catalog.catalog_id} "
            f"(deriva.config.acl_config exited {e.returncode}). "
            f"The catalog was created but is missing its security policy — "
            f"row-level update/delete bindings will not work. "
            f"stderr:\n{e.stderr}"
        ) from e

    # Create alias if requested
    if catalog_alias:
        server.create_ermrest_alias(
            id=catalog_alias,
            alias_target=catalog.catalog_id,
            name=project_name,
            description=f"Alias for {project_name} catalog (ID: {catalog.catalog_id})",
        )

    return catalog


def create_or_retarget_ml_catalog(
    hostname: str,
    project_name: str,
    *,
    alias: str | None = None,
) -> ErmrestCatalog:
    """Create a new ML catalog and point ``alias`` at it.

    Idempotent on ``alias``: re-running with the same alias always yields a
    *fresh* catalog reachable via that alias. The previously-aliased catalog
    (if any) is left in place — its numeric id is preserved, but the alias
    no longer resolves to it. Callers who want the orphan deleted must do so
    explicitly (e.g. via ``server.delete_ermrest_catalog(old_id)``).

    Compare with :func:`create_ml_catalog`, which always creates a new alias
    and refuses if one with the same name exists. ``create_or_retarget_ml_catalog``
    is the right call when the alias is a *stable handle* for "the current
    catalog of project X" rather than a one-shot decoration.

    Args:
        hostname: Server hostname (e.g., "localhost", "www.eye-ai.org").
        project_name: Name for the project, becomes the domain schema name.
        alias: Optional alias name. When given, retargets the alias if it
            already exists, otherwise creates a new one. When ``None``,
            behaves identically to :func:`create_ml_catalog` with no alias
            (the caller must track the numeric id themselves).

    Returns:
        The created ``ErmrestCatalog`` instance. Callers can read
        ``catalog.catalog_id`` for the numeric id. The alias (if any) points
        to this catalog after the call returns.

    Example:
        >>> # Re-run after a successful run that lost track of the id:
        >>> catalog = create_or_retarget_ml_catalog(  # doctest: +SKIP
        ...     "localhost", "my_project", alias="my-project-dev"
        ... )
        >>> # The old catalog at the old id still exists; the alias now
        >>> # resolves to the new catalog. Delete the orphan separately
        >>> # if desired.
    """
    # Always create a new catalog first.
    new_catalog = create_ml_catalog(hostname, project_name, catalog_alias=None)

    if alias is None:
        return new_catalog

    # Look up the alias. If it exists, retarget. If not, create.
    server = new_catalog.deriva_server
    try:
        existing_alias = server.connect_ermrest_alias(alias)
        # Retarget to the new catalog.
        existing_alias.update(alias_target=str(new_catalog.catalog_id))
    except Exception:
        # Alias doesn't exist; create it.
        server.create_ermrest_alias(
            id=alias,
            alias_target=new_catalog.catalog_id,
            name=project_name,
            description=f"Alias for {project_name} catalog (ID: {new_catalog.catalog_id})",
        )

    return new_catalog


__all__ = [
    "create_ml_catalog",
    "create_or_retarget_ml_catalog",
    "create_ml_schema",
    "initialize_ml_schema",
    "directory_dataset_table_def",
]
