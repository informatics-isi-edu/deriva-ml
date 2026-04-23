"""Live-catalog integration tests for Bug E.2 (pre-allocated RID upload).

Two tests, both gated on DERIVA_HOST:

1. Happy path: upload uses the pre-leased RID.
2. Soft mode: identical file uploaded by two executions resolves to the
   same catalog row (legacy MD5+Filename dedup preserved when the
   strict annotation is absent).

The strict-mode integration test — where an annotated table raises on
mismatch — requires manipulating a test asset table's annotation and
forcing a second upload of a same-MD5 file with a different pre-leased
RID. This is exercised by the unit-level test in
``tests/core/test_strict_preallocated_rid.py`` (helper behavior) plus
the deriva-py unit tests (mismatch-with-annotation → raise); an
end-to-end integration check is deferred as future hardening.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import pytest


requires_catalog = pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="Bug E.2 live tests require DERIVA_HOST",
)


def _make_workflow(test_ml, name: str):
    from deriva_ml import MLVocab as vc
    test_ml.add_term(
        vc.workflow_type,
        "Test Workflow",
        description="for Bug E.2 live tests",
    )
    return test_ml.create_workflow(
        name=name,
        workflow_type="Test Workflow",
        description="for Bug E.2 live tests",
    )


def _lease_one_rid(test_ml) -> tuple[str, str]:
    """POST a single lease to ERMrest_RID_Lease; return (token, rid)."""
    from deriva_ml.execution.rid_lease import (
        generate_lease_token, post_lease_batch,
    )
    token = generate_lease_token()
    result = post_lease_batch(catalog=test_ml.catalog, tokens=[token])
    return token, result[token]


@requires_catalog
def test_upload_asset_uses_pre_leased_rid(test_ml, tmp_path):
    """End-to-end: the catalog row's RID matches the caller's pre-leased RID."""
    from deriva_ml.execution.state_store import PendingRowStatus

    f = tmp_path / "bug-e2-happy.bin"
    f.write_bytes(b"bug-e2 happy path " * 32)

    token, leased_rid = _lease_one_rid(test_ml)

    wf = _make_workflow(test_ml, "Bug E2 happy")
    exe = test_ml.create_execution(description="bug-e2-happy", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)

    with exe.execute():
        pass

    store.insert_pending_row(
        execution_rid=exe.execution_rid,
        key="k-happy",
        target_schema="deriva-ml",
        target_table="Execution_Asset",
        metadata_json=json.dumps({}),
        created_at=now,
        rid=leased_rid,
        status=PendingRowStatus.leased,
        lease_token=token,
        asset_file_path=str(f),
    )

    report = exe.upload_outputs()
    assert report.total_failed == 0, f"failures: {report.errors}"
    assert report.total_uploaded == 1

    # Catalog row must have our pre-leased RID.
    pb = test_ml.pathBuilder()
    asset_path = pb.schemas["deriva-ml"].tables["Execution_Asset"]
    rows = list(
        asset_path.filter(asset_path.RID == leased_rid)
        .entities().fetch()
    )
    assert len(rows) == 1, (
        f"Execution_Asset row with RID={leased_rid} not found — "
        f"Bug E.2 regression: server substituted its own RID instead "
        f"of honoring our lease."
    )
    # Sanity: MD5 matches.
    expected_md5 = hashlib.md5(f.read_bytes()).hexdigest()
    assert rows[0]["MD5"] == expected_md5


@requires_catalog
def test_soft_mode_second_upload_adopts_existing_rid(test_ml, tmp_path):
    """Two executions upload the same file; second adopts the first's catalog RID.

    Soft mode (the default, because Execution_Asset has no strict
    annotation) preserves legacy MD5+Filename dedup: the second
    execution's upload detects the existing row and adopts its RID
    instead of raising. The second execution's pre-leased RID is
    silently discarded.
    """
    from deriva_ml.execution.state_store import PendingRowStatus

    f = tmp_path / "bug-e2-shared.bin"
    f.write_bytes(b"bug-e2 shared artifact " * 32)
    expected_md5 = hashlib.md5(f.read_bytes()).hexdigest()

    wf = _make_workflow(test_ml, "Bug E2 soft-1")

    # Execution 1 — first to upload.
    token1, leased_rid_1 = _lease_one_rid(test_ml)
    exe1 = test_ml.create_execution(description="bug-e2-soft-1", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)
    with exe1.execute():
        pass
    store.insert_pending_row(
        execution_rid=exe1.execution_rid,
        key="k-soft-1",
        target_schema="deriva-ml",
        target_table="Execution_Asset",
        metadata_json=json.dumps({}),
        created_at=now,
        rid=leased_rid_1,
        status=PendingRowStatus.leased,
        lease_token=token1,
        asset_file_path=str(f),
    )
    report1 = exe1.upload_outputs()
    assert report1.total_failed == 0, report1.errors

    # Execution 2 — uploads the SAME file with a DIFFERENT pre-leased RID.
    token2, leased_rid_2 = _lease_one_rid(test_ml)
    assert leased_rid_1 != leased_rid_2

    exe2 = test_ml.create_execution(description="bug-e2-soft-2", workflow=wf)
    with exe2.execute():
        pass
    store.insert_pending_row(
        execution_rid=exe2.execution_rid,
        key="k-soft-2",
        target_schema="deriva-ml",
        target_table="Execution_Asset",
        metadata_json=json.dumps({}),
        created_at=now,
        rid=leased_rid_2,
        status=PendingRowStatus.leased,
        lease_token=token2,
        asset_file_path=str(f),
    )
    # Soft mode → upload succeeds by adopting existing row's RID.
    report2 = exe2.upload_outputs()
    assert report2.total_failed == 0, (
        f"Soft-mode upload should succeed but failed: {report2.errors}"
    )

    # Only ONE catalog row with this MD5 (not two).
    pb = test_ml.pathBuilder()
    asset_path = pb.schemas["deriva-ml"].tables["Execution_Asset"]
    rows = list(
        asset_path.filter(asset_path.MD5 == expected_md5)
        .entities().fetch()
    )
    assert len(rows) == 1, (
        f"expected single row for shared artifact, got {len(rows)}"
    )
    # The row's RID equals the FIRST lease (soft mode preserved legacy
    # MD5+Filename dedup).
    assert rows[0]["RID"] == leased_rid_1
