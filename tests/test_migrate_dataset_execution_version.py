"""Integration tests for ``scripts/migrate_dataset_execution_version.py``.

These tests exercise the live-catalog migration that makes
``Dataset_Execution`` the input-only provenance edge:

1. add a nullable ``Dataset_Version`` column + FK to ``Dataset_Execution``
   (idempotent — skipped if already present),
2. delete the redundant *output* rows (a ``(Dataset, Execution)`` pair is an
   output edge iff some ``Dataset_Version`` row for that dataset was authored
   by that execution), and
3. best-effort backfill the consumed version on the surviving input rows.

The tests run against a live catalog (``DERIVA_HOST``, default ``localhost``),
seeding the exact pre-migration shape with the public DerivaML write path
(``create_dataset`` for output edges, ``add_input_dataset`` for input edges)
plus a couple of direct ``pathBuilder`` inserts where that is the cleanest way
to build a legacy shape. They are gated ``@pytest.mark.integration`` and rely
on the suite-wide ``DERIVA_ML_ALLOW_DIRTY=true`` for workflow creation in a
dirty repo.

Run with::

    DERIVA_ML_ALLOW_DIRTY=true uv run pytest \
        tests/test_migrate_dataset_execution_version.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from deriva_ml import ExecutionConfiguration, MLVocab

# ---------------------------------------------------------------------------
# Load the migration module by path (scripts/ is not an importable package).
# ---------------------------------------------------------------------------
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "migrate_dataset_execution_version.py"
_spec = importlib.util.spec_from_file_location("migrate_dataset_execution_version", _SCRIPT_PATH)
migrate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migrate)

ML_SCHEMA = "deriva-ml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _de_table(model):
    return model.schemas[ML_SCHEMA].tables["Dataset_Execution"]


def _has_version_column(catalog) -> bool:
    de = _de_table(catalog.getCatalogModel())
    return "Dataset_Version" in {c.name for c in de.columns}


def _drop_version_column_if_present(catalog) -> None:
    """Simulate a pre-Task-1 (legacy) catalog by removing the FK + column.

    Fresh catalogs created by ``create_ml_catalog`` now carry the
    ``Dataset_Version`` column (Task 1). To test that ``step1_add_column``
    actually adds it, we first strip it back to the legacy shape.
    """
    model = catalog.getCatalogModel()
    de = _de_table(model)
    if "Dataset_Version" not in {c.name for c in de.columns}:
        return
    # Drop any FK that references the Dataset_Version column first.
    for fk in list(de.foreign_keys):
        if any(c.name == "Dataset_Version" for c in fk.foreign_key_columns):
            fk.drop()
    de.columns["Dataset_Version"].drop()


def _dataset_execution_rows(catalog) -> list[dict]:
    pb = catalog.getPathBuilder()
    return list(pb.schemas[ML_SCHEMA].Dataset_Execution.entities().fetch())


def _seed_output_edge(ml) -> tuple[str, str]:
    """Create an OUTPUT edge: an execution that PRODUCED a dataset version.

    ``create_dataset`` writes BOTH a ``Dataset_Execution`` row (the legacy,
    redundant output row this migration deletes) AND a ``Dataset_Version`` row
    authored by the same execution (``Dataset_Version.Execution`` points back).
    That is exactly the pre-migration output shape.

    Returns:
        ``(dataset_rid, execution_rid)`` of the produced dataset.
    """
    workflow = ml.create_workflow(name="Producer Workflow", workflow_type="Test Workflow")
    execution = ml.create_execution(ExecutionConfiguration(description="Produce", workflow=workflow))
    dataset = execution.create_dataset(dataset_types=["TestDS"], description="Produced dataset")
    return dataset.dataset_rid, execution.execution_rid


def _seed_input_edge(ml, dataset_rid: str) -> str:
    """Create an INPUT edge: an execution that CONSUMED an existing dataset.

    A *second* execution (which did not author ``dataset_rid``) records it via
    ``add_input_dataset`` — a single ``Dataset_Execution`` row with no authoring
    ``Dataset_Version``. This is the row the migration must PRESERVE.

    Returns:
        The consuming ``execution_rid``.
    """
    workflow = ml.create_workflow(name="Consumer Workflow", workflow_type="Test Workflow")
    with ml.create_execution(ExecutionConfiguration(description="Consume", workflow=workflow)) as exe:
        exe.add_input_dataset(dataset_rid)
    return exe.execution_rid


def _seed_input_edge_with_config(ml, dataset_rid: str, version: str) -> str:
    """Create an INPUT edge whose execution recorded the consumed version.

    A consuming execution records ``dataset_rid`` as an input AND uploads a
    ``configuration.json`` declaring ``DatasetSpec(rid=dataset_rid, version)``.
    This is the shape ``step3_backfill_input_versions`` can resolve.

    Returns:
        The consuming ``execution_rid``.
    """
    from deriva_ml.dataset.aux_classes import DatasetSpec

    workflow = ml.create_workflow(name="Config Consumer", workflow_type="Test Workflow")
    cfg = ExecutionConfiguration(
        description="Consume with config",
        workflow=workflow,
        datasets=[DatasetSpec(rid=dataset_rid, version=version, materialize=False)],
    )
    with ml.create_execution(cfg) as exe:
        exe.add_input_dataset(dataset_rid)
    exe.commit_output_assets(clean_folder=True)
    return exe.execution_rid


@pytest.fixture()
def seeded_ml(test_ml):
    """A clean ML instance with the vocab terms the seed helpers need."""
    test_ml.add_term(MLVocab.workflow_type, "Test Workflow", description="Test workflow")
    test_ml.add_term(MLVocab.dataset_type, "TestDS", description="Test dataset type")
    return test_ml


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_check_preconditions_reports_column_and_count(seeded_ml):
    """check_preconditions reports column presence and the row count."""
    ml = seeded_ml
    catalog = ml.catalog
    ds_rid, _ = _seed_output_edge(ml)
    _seed_input_edge(ml, ds_rid)

    info = migrate.check_preconditions(catalog, ML_SCHEMA)
    assert info["has_column"] is True  # fresh (post-Task-1) catalog
    assert info["row_count"] == len(_dataset_execution_rows(catalog))
    assert info["row_count"] >= 2  # one output edge + one input edge


@pytest.mark.integration
def test_step1_adds_column_when_missing(seeded_ml):
    """On a legacy catalog (no column), step1 adds the nullable FK column."""
    ml = seeded_ml
    catalog = ml.catalog
    _drop_version_column_if_present(catalog)
    assert _has_version_column(catalog) is False

    added = migrate.step1_add_column(catalog, ML_SCHEMA, dry_run=False)
    assert added is True
    assert _has_version_column(catalog) is True

    # Column is nullable and FKs to the Dataset_Version table.
    de = _de_table(catalog.getCatalogModel())
    dv_col = next(c for c in de.columns if c.name == "Dataset_Version")
    assert dv_col.nullok is True
    fk_targets = {
        fk.referenced_columns[0].table.name
        for fk in de.foreign_keys
        if any(c.name == "Dataset_Version" for c in fk.foreign_key_columns)
    }
    assert "Dataset_Version" in fk_targets


@pytest.mark.integration
def test_step1_skips_when_present(seeded_ml):
    """step1 is a no-op when the column already exists (idempotent)."""
    ml = seeded_ml
    catalog = ml.catalog
    assert _has_version_column(catalog) is True  # fresh catalog
    added = migrate.step1_add_column(catalog, ML_SCHEMA, dry_run=False)
    assert added is False
    assert _has_version_column(catalog) is True


@pytest.mark.integration
def test_output_rows_deleted_input_rows_survive(seeded_ml):
    """Output edges are deleted from Dataset_Execution; input edges survive."""
    ml = seeded_ml
    catalog = ml.catalog
    out_ds, out_exec = _seed_output_edge(ml)
    in_exec = _seed_input_edge(ml, out_ds)

    before = _dataset_execution_rows(catalog)
    before_pairs = {(r["Dataset"], r["Execution"]) for r in before}
    assert (out_ds, out_exec) in before_pairs, "legacy output row should be present pre-migration"
    assert (out_ds, in_exec) in before_pairs, "input row should be present pre-migration"

    result = migrate.run_migration(catalog, ML_SCHEMA, dry_run=False)

    after = _dataset_execution_rows(catalog)
    after_pairs = {(r["Dataset"], r["Execution"]) for r in after}
    # The output edge (dataset authored by that execution) is gone.
    assert (out_ds, out_exec) not in after_pairs
    # The input edge (consumed, not authored) survives.
    assert (out_ds, in_exec) in after_pairs
    assert result["output_rows_deleted"] >= 1

    # Output provenance is preserved in Dataset_Version.Execution.
    pb = catalog.getPathBuilder()
    dv = pb.schemas[ML_SCHEMA].Dataset_Version
    authored = {
        (r["Dataset"], r["Execution"])
        for r in dv.entities().fetch()
        if r.get("Execution")
    }
    assert (out_ds, out_exec) in authored


@pytest.mark.integration
def test_idempotent(seeded_ml):
    """A second run reports nothing added and nothing deleted."""
    ml = seeded_ml
    catalog = ml.catalog
    out_ds, _ = _seed_output_edge(ml)
    _seed_input_edge(ml, out_ds)

    first = migrate.run_migration(catalog, ML_SCHEMA, dry_run=False)
    assert first["output_rows_deleted"] >= 1

    second = migrate.run_migration(catalog, ML_SCHEMA, dry_run=False)
    assert second["columns_added"] == 0
    assert second["output_rows_deleted"] == 0


@pytest.mark.integration
def test_dry_run_makes_no_changes(seeded_ml):
    """--dry-run leaves the catalog identical (preconditions unchanged)."""
    ml = seeded_ml
    catalog = ml.catalog
    # Legacy shape so a real run *would* both add the column and delete rows.
    _drop_version_column_if_present(catalog)
    out_ds, out_exec = _seed_output_edge(ml)
    _seed_input_edge(ml, out_ds)

    before = migrate.check_preconditions(catalog, ML_SCHEMA)
    result = migrate.run_migration(catalog, ML_SCHEMA, dry_run=True)
    after = migrate.check_preconditions(catalog, ML_SCHEMA)

    assert before == after, "dry-run must not change catalog state"
    assert before["has_column"] is False  # still legacy shape
    # Dry-run still reports what it *would* do.
    assert result["columns_added"] == 1
    assert result["output_rows_deleted"] >= 1


@pytest.mark.integration
def test_step3_backfills_input_version_from_config(seeded_ml, tmp_path):
    """Best-effort backfill fills Dataset_Version for an input edge whose
    execution recorded the consumed version in its configuration.json.

    Seeds the legacy shape directly: a released Dataset_Version row plus an
    input Dataset_Execution row, then writes a configuration.json
    Execution_Metadata asset declaring that version as a DatasetSpec.
    """
    ml = seeded_ml
    catalog = ml.catalog

    out_ds, _ = _seed_output_edge(ml)
    # The output edge's version row is the one we expect the input edge to
    # resolve to (DatasetSpec.version == that version).
    pb = catalog.getPathBuilder()
    dv_rows = [r for r in pb.schemas[ML_SCHEMA].Dataset_Version.entities().fetch() if r["Dataset"] == out_ds]
    assert dv_rows, "produced dataset should have a Dataset_Version row"
    target_version = dv_rows[0]["Version"]
    target_dv_rid = dv_rows[0]["RID"]

    # A consuming execution records the dataset as an input AND uploads a
    # configuration.json declaring DatasetSpec(rid=out_ds, version=target).
    in_exec = _seed_input_edge_with_config(ml, out_ds, target_version)

    result = migrate.run_migration(catalog, ML_SCHEMA, dry_run=False)
    backfill = result["backfill"]
    assert backfill["filled"] >= 1, f"expected to backfill the configured input edge: {backfill}"

    # The input edge row now points at the matching Dataset_Version RID.
    de_rows = _dataset_execution_rows(catalog)
    in_rows = [r for r in de_rows if r["Execution"] == in_exec and r["Dataset"] == out_ds]
    assert in_rows, "input edge must survive migration"
    assert in_rows[0].get("Dataset_Version") == target_dv_rid


@pytest.mark.integration
def test_dry_run_does_not_backfill_when_column_present(seeded_ml, tmp_path):
    """With the column already present and a resolvable config, --dry-run
    reports the would-fill count but performs no write to Dataset_Version."""
    ml = seeded_ml
    catalog = ml.catalog

    out_ds, _ = _seed_output_edge(ml)
    pb = catalog.getPathBuilder()
    dv_rows = [r for r in pb.schemas[ML_SCHEMA].Dataset_Version.entities().fetch() if r["Dataset"] == out_ds]
    assert dv_rows, "produced dataset should have a Dataset_Version row"
    target_version = dv_rows[0]["Version"]

    in_exec = _seed_input_edge_with_config(ml, out_ds, target_version)
    assert migrate.check_preconditions(catalog, ML_SCHEMA)["has_column"] is True

    result = migrate.run_migration(catalog, ML_SCHEMA, dry_run=True)
    # Dry-run still reports what it *would* backfill...
    assert result["backfill"]["filled"] >= 1
    # ...but the input row's Dataset_Version is still NULL (no write happened).
    in_rows = [r for r in _dataset_execution_rows(catalog) if r["Execution"] == in_exec and r["Dataset"] == out_ds]
    assert in_rows, "input edge must still be present"
    assert in_rows[0].get("Dataset_Version") is None
