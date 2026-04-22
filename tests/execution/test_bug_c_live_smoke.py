"""Live-catalog integration tests for Bug C (asset metadata None-stringification).

Gated on DERIVA_HOST. Three tests:

1. End-to-end happy path with Execution_Asset (zero metadata cols).
2. Upload refused when required metadata missing — validator raises.
3. Upload succeeds with SQL NULL when nullable metadata missing.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import pytest


requires_catalog = pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="Bug C live tests require DERIVA_HOST",
)


def _make_workflow(test_ml, name: str):
    from deriva_ml import MLVocab as vc
    test_ml.add_term(
        vc.workflow_type,
        "Test Workflow",
        description="for Bug C live tests",
    )
    return test_ml.create_workflow(
        name=name,
        workflow_type="Test Workflow",
        description="for Bug C live tests",
    )


@requires_catalog
def test_upload_asset_with_full_metadata_end_to_end(test_ml, tmp_path):
    """Zero-metadata table (Execution_Asset) — upload succeeds, validator is no-op."""
    from deriva_ml.execution.state_store import PendingRowStatus

    f = tmp_path / "smoke.bin"
    f.write_bytes(b"bug-c full-metadata smoke" * 32)

    wf = _make_workflow(test_ml, "Bug C happy path")
    exe = test_ml.create_execution(description="bug-c-happy", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)

    with exe.execute():
        pass

    store.insert_pending_row(
        execution_rid=exe.execution_rid,
        key="k1",
        target_schema="deriva-ml",
        target_table="Execution_Asset",
        metadata_json=json.dumps({}),
        created_at=now,
        rid="EA-BUGC-HAPPY-1",
        status=PendingRowStatus.leased,
        lease_token="happy-lease",
        asset_file_path=str(f),
    )

    report = exe.upload_outputs()
    assert report.total_failed == 0, f"failures: {report.errors}"
    assert report.total_uploaded == 1

    # Catalog has the row with real URL + MD5.
    expected_md5 = hashlib.md5(f.read_bytes()).hexdigest()
    pb = test_ml.pathBuilder()
    asset_path = pb.schemas["deriva-ml"].tables["Execution_Asset"]
    rows = list(
        asset_path.filter(asset_path.MD5 == expected_md5)
        .filter(asset_path.Filename == "smoke.bin")
        .entities().fetch()
    )
    assert len(rows) == 1
    assert "/hatrac/" in rows[0]["URL"]


def _find_asset_table_with_required_metadata(test_ml) -> tuple[str, str, list[str]] | None:
    """Scan all asset tables in the live model and return (schema, table, required_col_names)
    for the first table with at least one NOT-NULL metadata column. Returns None if none found."""
    for schema_name, schema in test_ml.model.model.schemas.items():
        for table_name in schema.tables:
            try:
                cols = test_ml.model.asset_metadata_columns(table_name)
            except Exception:
                continue
            required = [c.name for c in cols if not c.nullok]
            if required:
                return (schema_name, table_name, required)
    return None


def _find_asset_table_with_all_nullable_metadata(test_ml) -> tuple[str, str, str] | None:
    """Return (schema, table, first_nullable_col_name) for the first asset table with
    nullable metadata and NO required metadata columns. Returns None if none found."""
    for schema_name, schema in test_ml.model.model.schemas.items():
        for table_name in schema.tables:
            try:
                cols = test_ml.model.asset_metadata_columns(table_name)
            except Exception:
                continue
            required = [c for c in cols if not c.nullok]
            nullable = [c for c in cols if c.nullok]
            if nullable and not required:
                return (schema_name, table_name, nullable[0].name)
    return None


@requires_catalog
def test_upload_with_missing_required_metadata_raises_validation(test_ml, tmp_path):
    """Bug C reproducer — passing now. Staging an asset whose table has
    a NOT-NULL metadata column, with no metadata supplied, must raise
    DerivaMLValidationError and NOT attempt the upload."""
    from deriva_ml.core.exceptions import DerivaMLValidationError
    from deriva_ml.execution.state_store import PendingRowStatus

    found = _find_asset_table_with_required_metadata(test_ml)
    if not found:
        pytest.skip(
            "Test catalog has no asset table with NOT-NULL metadata; "
            "Bug C required-column path cannot be exercised."
        )
    schema_name, table_name, required_cols = found

    f = tmp_path / "bug-c-required.bin"
    f.write_bytes(b"bug-c required-missing" * 32)

    wf = _make_workflow(test_ml, "Bug C required missing")
    exe = test_ml.create_execution(description="bug-c-required", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)

    with exe.execute():
        pass

    # Pending row with EMPTY metadata — missing required columns.
    store.insert_pending_row(
        execution_rid=exe.execution_rid,
        key="k2",
        target_schema=schema_name,
        target_table=table_name,
        metadata_json=json.dumps({}),
        created_at=now,
        rid="IMG-BUGC-REQ-1",
        status=PendingRowStatus.leased,
        lease_token="req-lease",
        asset_file_path=str(f),
    )

    with pytest.raises(DerivaMLValidationError) as ei:
        exe.upload_outputs()
    msg = str(ei.value)
    for c in required_cols:
        assert c in msg, f"expected column {c} in error message"


@requires_catalog
def test_upload_with_missing_nullable_metadata_succeeds_with_null(test_ml, tmp_path):
    """The sentinel path. Staging an asset with a nullable metadata col
    absent must upload successfully and write SQL NULL to the catalog."""
    from deriva_ml.execution.state_store import PendingRowStatus

    found = _find_asset_table_with_all_nullable_metadata(test_ml)
    if not found:
        pytest.skip(
            "No asset table with all-nullable metadata found in test fixture; "
            "Bug C sentinel path cannot be exercised."
        )
    schema_name, table_name, nullable_col_name = found

    f = tmp_path / "bug-c-null.bin"
    f.write_bytes(b"bug-c nullable-missing" * 32)

    wf = _make_workflow(test_ml, "Bug C nullable missing")
    exe = test_ml.create_execution(description="bug-c-nullable", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)

    with exe.execute():
        pass

    # Supply metadata for no columns — all are nullable.
    store.insert_pending_row(
        execution_rid=exe.execution_rid,
        key="k3",
        target_schema=schema_name,
        target_table=table_name,
        metadata_json=json.dumps({}),
        created_at=now,
        rid="NULL-BUGC-1",
        status=PendingRowStatus.leased,
        lease_token="null-lease",
        asset_file_path=str(f),
    )

    report = exe.upload_outputs()
    assert report.total_failed == 0, f"failures: {report.errors}"
    assert report.total_uploaded == 1

    # The catalog row's nullable column must be actual SQL NULL
    # (Python None after fetch), not the string "__NULL__" and not
    # "None".
    expected_md5 = hashlib.md5(f.read_bytes()).hexdigest()
    pb = test_ml.pathBuilder()
    asset_path = pb.schemas[schema_name].tables[table_name]
    rows = list(
        asset_path.filter(asset_path.MD5 == expected_md5)
        .entities().fetch()
    )
    assert len(rows) == 1
    assert rows[0][nullable_col_name] is None, (
        f"expected SQL NULL for {nullable_col_name}, "
        f"got {rows[0][nullable_col_name]!r}"
    )
