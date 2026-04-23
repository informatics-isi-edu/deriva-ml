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

    # Bug E.2: ERMrest validates pre-allocated RIDs against
    # ERMrest_RID_Lease and decodes them as URL-base32; use a real
    # leased RID instead of a hand-crafted fake one.
    from deriva_ml.execution.rid_lease import (
        generate_lease_token, post_lease_batch,
    )
    _token = generate_lease_token()
    _leased = post_lease_batch(catalog=test_ml.catalog, tokens=[_token])[_token]

    store.insert_pending_row(
        execution_rid=exe.execution_rid,
        key="k1",
        target_schema="deriva-ml",
        target_table="Execution_Asset",
        metadata_json=json.dumps({}),
        created_at=now,
        rid=_leased,
        status=PendingRowStatus.leased,
        lease_token=_token,
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


def _find_asset_table_with_nullable_metadata(test_ml) -> tuple[str, str, dict, str] | None:
    """Return (schema, table, required_metadata, first_nullable_col_name) for the first
    asset table with at least one nullable metadata column. Required columns (if any)
    are returned as a dict {col_name: value} the caller must merge into the pending-row
    metadata to satisfy the validator and catalog FK constraints. For FK required cols,
    an existing row from the referenced table is used (or a new one inserted). Returns
    None if no suitable table or the caller can't satisfy FK constraints."""
    from datetime import date, datetime, timezone
    model = test_ml.model.model
    pb = test_ml.pathBuilder()
    for schema_name, schema in model.schemas.items():
        for table_name in schema.tables:
            try:
                cols = test_ml.model.asset_metadata_columns(table_name)
            except Exception:
                continue
            nullable = [c for c in cols if c.nullok]
            if not nullable:
                continue
            # Build a lookup of FK columns for this table so we can resolve
            # FK-typed required cols to a real referenced-table value.
            table = schema.tables[table_name]
            fk_by_col: dict[str, tuple[str, str, str]] = {}
            try:
                for fk in table.foreign_keys:
                    # Single-column FKs only (enough for common asset tables).
                    if len(fk.column_map) == 1:
                        from_col, to_col = next(iter(fk.column_map.items()))
                        ref_table = to_col.table
                        fk_by_col[from_col.name] = (
                            ref_table.schema.name, ref_table.name, to_col.name
                        )
            except Exception:
                pass

            required_md: dict = {}
            ok = True
            for c in cols:
                if c.nullok:
                    continue
                # FK column — look up an existing row or insert one.
                if c.name in fk_by_col:
                    ref_schema, ref_table_name, ref_col = fk_by_col[c.name]
                    try:
                        ref_path = pb.schemas[ref_schema].tables[ref_table_name]
                        existing = list(ref_path.entities().fetch(limit=1))
                        if existing:
                            required_md[c.name] = existing[0][ref_col]
                            continue
                        # Try to insert a minimal parent row (tables typically
                        # have enough defaults — RID, RCT, etc. are system cols).
                        inserted = ref_path.insert([{}])
                        if inserted:
                            required_md[c.name] = inserted[0][ref_col]
                            continue
                    except Exception:
                        pass
                    ok = False
                    break
                # Non-FK required col — build a dummy value by typename.
                tn = c.type.typename if hasattr(c, "type") else "text"
                if tn in ("text", "longtext", "markdown"):
                    required_md[c.name] = "bug-c-test-value"
                elif tn in ("int2", "int4", "int8"):
                    required_md[c.name] = 1
                elif tn in ("float4", "float8"):
                    required_md[c.name] = 1.0
                elif tn in ("timestamp", "timestamptz"):
                    required_md[c.name] = datetime.now(timezone.utc).isoformat()
                elif tn == "date":
                    required_md[c.name] = date.today().isoformat()
                elif tn == "boolean":
                    required_md[c.name] = True
                else:
                    ok = False
                    break
            if not ok:
                continue
            return (schema_name, table_name, required_md, nullable[0].name)
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

    # Bug E.2: use a real leased RID (ERMrest validates it at insert time).
    from deriva_ml.execution.rid_lease import (
        generate_lease_token, post_lease_batch,
    )
    _token = generate_lease_token()
    _leased = post_lease_batch(catalog=test_ml.catalog, tokens=[_token])[_token]

    # Pending row with EMPTY metadata — missing required columns.
    store.insert_pending_row(
        execution_rid=exe.execution_rid,
        key="k2",
        target_schema=schema_name,
        target_table=table_name,
        metadata_json=json.dumps({}),
        created_at=now,
        rid=_leased,
        status=PendingRowStatus.leased,
        lease_token=_token,
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
    absent must upload successfully and write SQL NULL to the catalog.
    Required columns (if any) are supplied so the validator doesn't
    block the upload."""
    from deriva_ml.execution.state_store import PendingRowStatus

    found = _find_asset_table_with_nullable_metadata(test_ml)
    if not found:
        pytest.skip(
            "No asset table with nullable metadata found in test fixture; "
            "Bug C sentinel path cannot be exercised."
        )
    schema_name, table_name, required_md, nullable_col_name = found

    f = tmp_path / "bug-c-null.bin"
    f.write_bytes(b"bug-c nullable-missing" * 32)

    wf = _make_workflow(test_ml, "Bug C nullable missing")
    exe = test_ml.create_execution(description="bug-c-nullable", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)

    with exe.execute():
        pass

    # Bug E.2: use a real leased RID (ERMrest validates it at insert time).
    from deriva_ml.execution.rid_lease import (
        generate_lease_token, post_lease_batch,
    )
    _token = generate_lease_token()
    _leased = post_lease_batch(catalog=test_ml.catalog, tokens=[_token])[_token]

    # Supply ONLY the required metadata — nullable cols remain absent.
    store.insert_pending_row(
        execution_rid=exe.execution_rid,
        key="k3",
        target_schema=schema_name,
        target_table=table_name,
        metadata_json=json.dumps(required_md),
        created_at=now,
        rid=_leased,
        status=PendingRowStatus.leased,
        lease_token=_token,
        asset_file_path=str(f),
    )

    report = exe.upload_outputs()
    assert report.total_failed == 0, f"failures: {report.errors}"
    assert report.total_uploaded == 1

    # The catalog row's nullable column must be actual SQL NULL
    # (Python None after fetch), not the string "__NULL__" and not "None".
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
