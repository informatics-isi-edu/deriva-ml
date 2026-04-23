"""Unit tests for the row-per-record execution_state__feature_records table.

Stages individual FeatureRecord instances as JSON rows for later batch flush
to ermrest. Replaces the older file-based FEATURES_TABLE / .jsonl path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select

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
