"""Unit tests for the row-per-record execution_state__feature_records table.

Stages individual FeatureRecord instances as JSON rows for later batch flush
to ermrest. Replaces the older file-based FEATURES_TABLE / .jsonl path.

This file contains two test classes:
- SQLite-unit tests (no catalog needed) — original Task 1 tests
- Live-catalog integration tests — added in Task 7
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select

from deriva_ml import BuiltinTypes, ColumnDefinition
from deriva_ml.execution import ExecutionConfiguration
from deriva_ml.local_db.manifest_store import ManifestStore


def test_ensure_schema_is_idempotent(tmp_path: Path) -> None:
    """ManifestStore.ensure_schema() can be called twice without raising."""
    engine = create_engine(f"sqlite:///{tmp_path / 'manifest.sqlite'}")
    store = ManifestStore(engine)
    store.ensure_schema()
    store.ensure_schema()  # second call must not raise


def test_stage_feature_record_inserts_row(tmp_path: Path) -> None:
    """stage_feature_record writes a pending row readable by list_feature_records."""
    engine = create_engine(f"sqlite:///{tmp_path / 'manifest.sqlite'}")
    store = ManifestStore(engine)
    store.ensure_schema()
    store.stage_feature_record(
        execution_rid="EXE-1",
        feature_table="domain.Image_Glaucoma",
        feature_name="Glaucoma",
        target_table="Image",
        record_json=json.dumps({"Image": "IMG-1", "Glaucoma": "Normal"}),
    )
    rows = store.list_feature_records("EXE-1")
    assert len(rows) == 1
    row = rows[0]
    assert row.feature_table == "domain.Image_Glaucoma"
    assert row.feature_name == "Glaucoma"
    assert row.status == "pending"
    assert row.error is None
    assert row.uploaded_at is None
    assert json.loads(row.record_json) == {"Image": "IMG-1", "Glaucoma": "Normal"}


def test_mark_feature_record_uploaded(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'manifest.sqlite'}")
    store = ManifestStore(engine)
    store.ensure_schema()
    store.stage_feature_record(
        execution_rid="EXE-1",
        feature_table="domain.Image_Glaucoma",
        feature_name="Glaucoma",
        target_table="Image",
        record_json=json.dumps({"Image": "IMG-1"}),
    )
    stage_id = store.list_feature_records("EXE-1")[0].stage_id
    store.mark_feature_record_uploaded(stage_id)
    row = store.list_feature_records("EXE-1")[0]
    assert row.status == "uploaded"
    assert row.uploaded_at is not None


def test_mark_feature_record_failed_records_error(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'manifest.sqlite'}")
    store = ManifestStore(engine)
    store.ensure_schema()
    store.stage_feature_record(
        execution_rid="EXE-1",
        feature_table="domain.Image_Glaucoma",
        feature_name="Glaucoma",
        target_table="Image",
        record_json=json.dumps({"Image": "IMG-1"}),
    )
    stage_id = store.list_feature_records("EXE-1")[0].stage_id
    store.mark_feature_record_failed(stage_id, error="ermrest rejected")
    row = store.list_feature_records("EXE-1")[0]
    assert row.status == "failed"
    assert row.error == "ermrest rejected"


def test_list_pending_feature_records_filters_by_status(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'manifest.sqlite'}")
    store = ManifestStore(engine)
    store.ensure_schema()
    # Stage three: one pending, one uploaded, one failed
    for i in range(3):
        store.stage_feature_record(
            execution_rid="EXE-1",
            feature_table="domain.Image_Glaucoma",
            feature_name="Glaucoma",
            target_table="Image",
            record_json=json.dumps({"Image": f"IMG-{i}"}),
        )
    ids = [r.stage_id for r in store.list_feature_records("EXE-1")]
    store.mark_feature_record_uploaded(ids[0])
    store.mark_feature_record_failed(ids[1], error="x")
    pending = store.list_pending_feature_records("EXE-1")
    assert len(pending) == 1
    assert pending[0].stage_id == ids[2]


def test_mark_feature_record_uploaded_unknown_stage_id_raises(tmp_path: Path) -> None:
    """mark_feature_record_uploaded raises KeyError for a nonexistent stage_id."""
    engine = create_engine(f"sqlite:///{tmp_path / 'manifest.sqlite'}")
    store = ManifestStore(engine)
    store.ensure_schema()
    with pytest.raises(KeyError):
        store.mark_feature_record_uploaded(99999)


def test_mark_feature_record_failed_unknown_stage_id_raises(tmp_path: Path) -> None:
    """mark_feature_record_failed raises KeyError for a nonexistent stage_id."""
    engine = create_engine(f"sqlite:///{tmp_path / 'manifest.sqlite'}")
    store = ManifestStore(engine)
    store.ensure_schema()
    with pytest.raises(KeyError):
        store.mark_feature_record_failed(99999, error="some error")


# =============================================================================
# Task 7 — live-catalog integration tests
# =============================================================================


@dataclasses.dataclass
class FeatureIntegrationFixture:
    """Container for a seeded feature on the test catalog."""

    workflow: object          # Workflow object for ExecutionConfiguration
    record_class: type        # FeatureRecord subclass
    schema: str               # domain schema name (e.g. "test-schema")
    feature_table_name: str   # bare table name (e.g. "Image_Image_Label")
    image_rids: list[str]     # target Image RIDs available for test records


@pytest.fixture
def image_feature(populated_catalog):
    """A test catalog with an Image/Image_Label feature pre-created.

    Yields a FeatureIntegrationFixture with workflow, record_class, schema,
    feature_table_name, and image_rids.

    Uses the same ``populated_catalog`` DerivaML instance that the test should
    also use for execution — passing it as ``test_ml`` is NOT safe because
    two separate DerivaML instances have independent model caches.  Tests that
    use this fixture must use ``populated_catalog`` directly (or call
    ``test_ml.model.refresh_model()`` before flushing) when they need the
    feature to be visible.
    """
    ml = populated_catalog
    feature_name = "Image_Label"

    RecordClass = ml.create_feature(
        target_table="Image",
        feature_name=feature_name,
        metadata=[ColumnDefinition(name="Image_Label", type=BuiltinTypes.text)],
    )
    image_rids = [r["RID"] for r in ml.domain_path().tables["Image"].entities().fetch()]
    assert len(image_rids) >= 2, "Need at least 2 Image rows"

    # Derive schema/table from the feature definition
    feat = RecordClass.feature
    schema_name = feat.feature_table.schema.name
    table_name = feat.feature_table.name

    workflow = ml.create_workflow(
        name="image-label-seeder-t7",
        workflow_type="Test Workflow",
    )
    yield FeatureIntegrationFixture(
        workflow=workflow,
        record_class=RecordClass,
        schema=schema_name,
        feature_table_name=table_name,
        image_rids=image_rids,
    )


@pytest.fixture
def other_feature(populated_catalog):
    """A second feature (Image/Quality_Score) for the mixed-feature-def test.

    Must be used alongside ``image_feature`` — both share the same catalog
    but reference different feature tables.

    Like ``image_feature``, this fixture uses ``populated_catalog`` to create
    the feature so that the same DerivaML instance owns both the feature
    definition and the execution — no cross-instance model-cache staleness.
    """
    ml = populated_catalog
    feature_name = "Quality_Score"

    RecordClass = ml.create_feature(
        target_table="Image",
        feature_name=feature_name,
        metadata=[ColumnDefinition(name="Quality", type=BuiltinTypes.int4)],
    )
    image_rids = [r["RID"] for r in ml.domain_path().tables["Image"].entities().fetch()]
    assert image_rids, "No Image rows in test catalog"

    feat = RecordClass.feature
    schema_name = feat.feature_table.schema.name
    table_name = feat.feature_table.name

    workflow = ml.create_workflow(
        name="quality-score-seeder-t7",
        workflow_type="Test Workflow",
    )
    yield FeatureIntegrationFixture(
        workflow=workflow,
        record_class=RecordClass,
        schema=schema_name,
        feature_table_name=table_name,
        image_rids=image_rids,
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_exe_add_features_stages_to_sqlite(populated_catalog, image_feature) -> None:
    """exe.add_features writes Pending rows to execution_state__feature_records, nothing to ermrest yet.

    Flush contract (Case B — upload is the caller's explicit responsibility):
    ``Execution.__exit__`` does NOT auto-upload staged features or assets.  It
    only transitions the execution state (running → stopped/failed) and emits a
    warning log if there are pending rows.  The caller is responsible for
    calling ``exe.upload_execution_outputs()`` AFTER the ``with`` block exits.
    This design keeps the context manager lightweight and allows callers to
    decide whether/when to flush (e.g. after additional post-processing).

    Note: this test uses ``populated_catalog`` (the same DerivaML instance that
    the ``image_feature`` fixture used to create the feature) rather than
    ``test_ml`` — using a separate instance would require cross-instance model
    refresh because ``lookup_feature`` checks the in-memory model.
    """
    ml = populated_catalog
    cfg = ExecutionConfiguration(description="stage test", workflow=image_feature.workflow)
    with ml.create_execution(cfg) as exe:
        RecordClass = image_feature.record_class
        exe.add_features([
            RecordClass(Image=image_feature.image_rids[0], Image_Label="A"),
            RecordClass(Image=image_feature.image_rids[1], Image_Label="B"),
        ])
        # Pending in SQLite — rows are staged but not yet sent to ermrest
        store = exe._manifest_store
        pending = store.list_pending_feature_records(exe.execution_rid)
        assert len(pending) == 2
        # Not in ermrest yet (query the feature table directly)
        pb = ml.pathBuilder()
        rows = list(pb.schemas[image_feature.schema].tables[image_feature.feature_table_name].entities().fetch())
        # Should have NO rows from this execution yet
        assert all(r.get("Execution") != exe.execution_rid for r in rows)
    # __exit__ transitions state to stopped but does NOT flush — rows are still pending.
    # Explicit upload call is required to flush staged features to ermrest.
    exe.upload_execution_outputs()
    pb = ml.pathBuilder()
    rows = list(pb.schemas[image_feature.schema].tables[image_feature.feature_table_name].entities().fetch())
    ours = [r for r in rows if r.get("Execution") == exe.execution_rid]
    assert len(ours) == 2


def test_exe_add_features_auto_fills_execution_rid(populated_catalog, image_feature) -> None:
    """Records without Execution set get it auto-filled from the execution context."""
    ml = populated_catalog
    cfg = ExecutionConfiguration(description="auto-fill test", workflow=image_feature.workflow)
    with ml.create_execution(cfg) as exe:
        RecordClass = image_feature.record_class
        # No Execution set — auto-fill should apply
        exe.add_features([RecordClass(Image=image_feature.image_rids[0], Image_Label="A")])
        pending = exe._manifest_store.list_pending_feature_records(exe.execution_rid)
        payload = json.loads(pending[0].record_json)
        assert payload["Execution"] == exe.execution_rid


def test_exe_add_features_mixed_feature_defs_raises(populated_catalog, image_feature, other_feature) -> None:
    """Records from different features raise DerivaMLValidationError before staging."""
    from deriva_ml.core.exceptions import DerivaMLValidationError

    ml = populated_catalog
    cfg = ExecutionConfiguration(description="mixed test", workflow=image_feature.workflow)
    with ml.create_execution(cfg) as exe:
        mixed = [
            image_feature.record_class(Image=image_feature.image_rids[0], Image_Label="A"),
            other_feature.record_class(Image=other_feature.image_rids[0], Quality=5),
        ]
        with pytest.raises(DerivaMLValidationError):
            exe.add_features(mixed)
        # Nothing staged
        assert exe._manifest_store.list_feature_records(exe.execution_rid) == []


def test_exe_add_features_empty_raises(populated_catalog, image_feature) -> None:
    """An empty features list raises ValueError."""
    ml = populated_catalog
    cfg = ExecutionConfiguration(description="empty test", workflow=image_feature.workflow)
    with ml.create_execution(cfg) as exe:
        with pytest.raises(ValueError):
            exe.add_features([])


def test_flush_happens_after_assets(populated_catalog, image_feature) -> None:
    """Feature flush order: assets first, then features.

    A feature whose asset column points at a staged asset must see the asset's
    uploaded RID in ermrest when the feature row is inserted. Verify by
    checking that asset upload completes before any feature INSERT attempts
    occur.
    """
    pytest.skip("Requires instrumenting upload_execution_outputs — see test plan")


def test_flush_failure_marks_group_failed_but_continues(
    populated_catalog, image_feature
) -> None:
    """If one feature group's insert fails, others still flush; DerivaMLUploadError summarizes."""
    pytest.skip("Requires injecting ermrest failure for one group — see test plan")


@pytest.mark.skip(reason="Task 11 — exercises asset-column rewriting during flush")
def test_flush_rewrites_asset_column_filenames_to_rids(test_ml) -> None:
    """Flush should rewrite local asset filenames to uploaded asset RIDs.

    The asset-column rewriting logic migrated from _update_feature_table must be
    exercised by a feature that has asset columns. Verify by staging records with
    filename strings in asset columns, running upload_execution_outputs, and
    asserting the resulting ermrest rows contain RIDs, not filenames.
    """
    pytest.skip("Task 11 — exercises asset-column rewriting during flush")
