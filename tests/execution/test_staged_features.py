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
from unittest import mock

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


@dataclasses.dataclass
class AssetFeatureFixture:
    """Fixture container for a feature that has an asset column."""

    workflow: object
    record_class: type
    schema: str
    feature_table_name: str
    asset_table_name: str      # bare name of the asset table (e.g. "Patch")
    image_rids: list[str]


@pytest.fixture
def asset_feature(populated_catalog):
    """A feature on Image whose single value column is an asset reference (Patch).

    The ``Patch`` asset table is created fresh each test run and is dynamically
    dropped by catalog_manager.reset().  This fixture uses the same DerivaML
    instance as ``populated_catalog`` so there is no cross-instance model-cache
    issue.
    """
    ml = populated_catalog
    asset_table = ml.create_asset("Patch", comment="Test patch asset", update_navbar=False)

    RecordClass = ml.create_feature(
        target_table="Image",
        feature_name="Patch",
        assets=[asset_table],
        update_navbar=False,
    )
    image_rids = [r["RID"] for r in ml.domain_path().tables["Image"].entities().fetch()]
    assert len(image_rids) >= 1, "Need at least 1 Image row"

    feat = RecordClass.feature
    schema_name = feat.feature_table.schema.name
    table_name = feat.feature_table.name

    workflow = ml.create_workflow(
        name="patch-seeder-t11",
        workflow_type="Test Workflow",
    )
    yield AssetFeatureFixture(
        workflow=workflow,
        record_class=RecordClass,
        schema=schema_name,
        feature_table_name=table_name,
        asset_table_name="Patch",
        image_rids=image_rids,
    )


def test_flush_happens_after_assets(populated_catalog, asset_feature) -> None:
    """Feature flush order: assets first, then features (Task 11).

    Stage a feature record whose asset column holds a local file path. Call
    ``upload_execution_outputs()`` and verify that the ermrest feature row
    contains the uploaded asset RID (not the local filename).  Because the
    rewrite only succeeds if the asset has already been uploaded when
    ``_flush_staged_features`` runs, a correct RID in the row proves
    ordering: assets first, features second.
    """
    ml = populated_catalog
    cfg = ExecutionConfiguration(description="asset-ordering test", workflow=asset_feature.workflow)
    with ml.create_execution(cfg) as exe:
        # Register a local file for upload
        asset_path = exe.asset_file_path(asset_feature.asset_table_name, "test_patch.bin")
        with asset_path.open("w") as fp:
            fp.write("patch data")

        RecordClass = asset_feature.record_class
        exe.add_features([
            RecordClass(
                Image=asset_feature.image_rids[0],
                Patch=asset_path,  # local path — should be rewritten to RID on flush
            )
        ])

        # Not yet in ermrest
        pb = ml.pathBuilder()
        rows_before = list(
            pb.schemas[asset_feature.schema].tables[asset_feature.feature_table_name].entities().fetch()
        )
        assert all(r.get("Execution") != exe.execution_rid for r in rows_before)

    # Upload: assets upload first, then features are flushed with RID rewriting.
    exe.upload_execution_outputs()

    pb = ml.pathBuilder()
    rows = list(
        pb.schemas[asset_feature.schema].tables[asset_feature.feature_table_name].entities().fetch()
    )
    ours = [r for r in rows if r.get("Execution") == exe.execution_rid]
    assert len(ours) == 1, f"Expected 1 feature row; got {len(ours)}"

    patch_value = ours[0].get("Patch")
    # Patch column must contain a catalog RID (not a local filename or path).
    # RIDs in Deriva are short alphanumeric strings (e.g. "1-A2BC"), never a
    # filesystem path or filename.
    assert patch_value is not None, "Patch column is None"
    assert "/" not in str(patch_value) and "\\" not in str(patch_value), (
        f"Patch column still holds a file path instead of a RID: {patch_value!r}"
    )
    assert str(patch_value) != "test_patch.bin", (
        f"Patch column still holds the bare filename: {patch_value!r}"
    )


def test_flush_rewrites_asset_column_filenames_to_rids(populated_catalog, asset_feature) -> None:
    """Flush rewrites local asset filenames to uploaded asset RIDs (Task 11).

    More focused than test_flush_happens_after_assets: asserts the specific
    rewrite behaviour when the staged JSON contains a local file path string
    in an asset column.  After upload the column value must be a catalog RID,
    not the original filename.
    """
    ml = populated_catalog
    cfg = ExecutionConfiguration(description="rewrite test", workflow=asset_feature.workflow)
    with ml.create_execution(cfg) as exe:
        asset_path = exe.asset_file_path(asset_feature.asset_table_name, "rewrite_test.bin")
        with asset_path.open("w") as fp:
            fp.write("rewrite data")

        RecordClass = asset_feature.record_class
        exe.add_features([
            RecordClass(
                Image=asset_feature.image_rids[0],
                Patch=asset_path,
            )
        ])
        # Confirm the staged JSON holds the *path*, not a RID
        pending = exe._manifest_store.list_pending_feature_records(exe.execution_rid)
        assert len(pending) == 1
        staged = json.loads(pending[0].record_json)
        assert "rewrite_test.bin" in staged["Patch"], (
            f"Expected filename in staged JSON, got: {staged['Patch']!r}"
        )

    exe.upload_execution_outputs()

    pb = ml.pathBuilder()
    rows = list(
        pb.schemas[asset_feature.schema].tables[asset_feature.feature_table_name].entities().fetch()
    )
    ours = [r for r in rows if r.get("Execution") == exe.execution_rid]
    assert len(ours) == 1
    patch_value = ours[0]["Patch"]
    assert patch_value is not None
    # Must be a RID, not a path
    assert "rewrite_test.bin" not in str(patch_value), (
        f"Filename still present in ermrest row: {patch_value!r}"
    )
    assert "/" not in str(patch_value), (
        f"File path in ermrest row: {patch_value!r}"
    )


def test_flush_failure_marks_group_failed_but_continues(
    populated_catalog, image_feature, other_feature
) -> None:
    """Per-group failure isolation (Task 11).

    If one feature table's batch insert raises, the other groups still flush
    successfully and ``DerivaMLUploadError`` summarises the failures.

    This test injects a failure by monkey-patching ``pathBuilder`` so that
    inserts on ``other_feature``'s table raise, while ``image_feature``'s
    table succeeds normally.  The monkeypatching targets the table-level
    ``insert`` call inside ``_flush_staged_features``.
    """
    from deriva_ml.core.exceptions import DerivaMLUploadError

    ml = populated_catalog
    # Use image_feature's workflow (both features share the same catalog)
    cfg = ExecutionConfiguration(
        description="failure-isolation test",
        workflow=image_feature.workflow,
    )
    with ml.create_execution(cfg) as exe:
        # Stage records for BOTH feature groups
        ImgClass = image_feature.record_class
        OtherClass = other_feature.record_class
        exe.add_features([ImgClass(Image=image_feature.image_rids[0], Image_Label="ok")])
        exe.add_features([OtherClass(Image=other_feature.image_rids[0], Quality=42)])

        exe_rid = exe.execution_rid
        # Sanity: two groups staged
        all_pending = exe._manifest_store.list_pending_feature_records(exe_rid)
        assert len(all_pending) == 2

    # Patch _flush_staged_features to inject a failure for other_feature's
    # table, while letting image_feature's table succeed via the real code.
    # Strategy: capture the real method, then re-run it after patching the
    # pathBuilder's insert for the target table.
    other_schema = other_feature.schema
    other_table = other_feature.feature_table_name
    real_pb = ml.pathBuilder

    def failing_pathBuilder():
        """pathBuilder wrapper that raises on insert for other_feature's table."""
        real = real_pb()

        class _TableProxy:
            def __init__(self, delegate, should_fail: bool):
                self._delegate = delegate
                self._should_fail = should_fail

            def insert(self, *args, **kwargs):
                if self._should_fail:
                    raise RuntimeError("injected failure for other_feature group")
                return self._delegate.insert(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._delegate, name)

        class _SchemaProxy:
            def __init__(self, delegate_schema, target_table: str):
                self._delegate = delegate_schema
                self._target_table = target_table

            @property
            def tables(self):
                return self

            def __getitem__(self, tname):
                tbl = self._delegate.tables[tname]
                return _TableProxy(tbl, should_fail=(tname == self._target_table))

            def __getattr__(self, name):
                return getattr(self._delegate, name)

        class _PBProxy:
            def __init__(self, delegate, fail_schema: str, fail_table: str):
                self._delegate = delegate
                self._fail_schema = fail_schema
                self._fail_table = fail_table

            @property
            def schemas(self):
                return self

            def __getitem__(self, schema_name):
                real_schema = self._delegate.schemas[schema_name]
                if schema_name == self._fail_schema:
                    return _SchemaProxy(real_schema, self._fail_table)
                return real_schema

            def __getattr__(self, name):
                return getattr(self._delegate, name)

        return _PBProxy(real, fail_schema=other_schema, fail_table=other_table)

    with mock.patch.object(ml, "pathBuilder", failing_pathBuilder):
        with pytest.raises(DerivaMLUploadError) as exc_info:
            exe.upload_execution_outputs()

    # image_feature group must have rows in ermrest
    pb = ml.pathBuilder()
    img_rows = list(
        pb.schemas[image_feature.schema].tables[image_feature.feature_table_name].entities().fetch()
    )
    img_ours = [r for r in img_rows if r.get("Execution") == exe_rid]
    assert len(img_ours) == 1, (
        f"image_feature group should have 1 row after flush; got {len(img_ours)}"
    )

    # other_feature group must be marked Failed in SQLite
    all_records = exe._manifest_store.list_feature_records(exe_rid)
    other_records = [r for r in all_records if other_table in r.feature_table]
    assert len(other_records) == 1
    assert other_records[0].status == "failed", (
        f"Expected 'failed' status for other_feature group; got {other_records[0].status!r}"
    )
    assert "injected failure" in (other_records[0].error or ""), (
        f"Expected injected error in record; got {other_records[0].error!r}"
    )

    # DerivaMLUploadError message must name the failing table
    assert other_table in str(exc_info.value), (
        f"Error message should mention the failing table; got: {exc_info.value!s}"
    )


def test_crash_before_flush_resumes_without_duplicates(populated_catalog, image_feature) -> None:
    """Staged rows persist after simulated crash; resume flushes exactly once (Task 11).

    Simulates a crash-before-flush scenario:
    1. Stage feature records inside the context manager.
    2. Exit the context manager WITHOUT calling upload_execution_outputs().
    3. Verify rows are still Pending in SQLite and absent from ermrest.
    4. Call upload_execution_outputs() to resume.
    5. Verify exactly ONE row per record in ermrest (no duplicates).
    6. Call upload_execution_outputs() again — must not duplicate.
    """
    ml = populated_catalog
    cfg = ExecutionConfiguration(description="crash-resume test", workflow=image_feature.workflow)

    with ml.create_execution(cfg) as exe:
        RecordClass = image_feature.record_class
        exe.add_features([
            RecordClass(Image=image_feature.image_rids[0], Image_Label="crash-A"),
        ])

    exe_rid = exe.execution_rid

    # Simulate crash: Pending rows survive in SQLite; ermrest is clean.
    pending = exe._manifest_store.list_pending_feature_records(exe_rid)
    assert len(pending) == 1, f"Expected 1 pending row; got {len(pending)}"

    pb = ml.pathBuilder()
    rows_before = list(
        pb.schemas[image_feature.schema].tables[image_feature.feature_table_name].entities().fetch()
    )
    assert not any(r.get("Execution") == exe_rid for r in rows_before), (
        "Feature row appeared in ermrest before upload — expected clean state"
    )

    # Resume flush: call upload on the same object (holds the SQLite reference)
    exe.upload_execution_outputs()

    pb = ml.pathBuilder()
    rows_after = list(
        pb.schemas[image_feature.schema].tables[image_feature.feature_table_name].entities().fetch()
    )
    ours = [r for r in rows_after if r.get("Execution") == exe_rid]
    assert len(ours) == 1, f"Expected exactly 1 row after resume flush; got {len(ours)}"

    # Second upload must not duplicate — on_conflict_skip prevents duplicates.
    exe.upload_execution_outputs()

    pb = ml.pathBuilder()
    rows_final = list(
        pb.schemas[image_feature.schema].tables[image_feature.feature_table_name].entities().fetch()
    )
    ours_final = [r for r in rows_final if r.get("Execution") == exe_rid]
    assert len(ours_final) == 1, (
        f"Expected exactly 1 row after second upload; got {len(ours_final)} (duplicates!)"
    )
