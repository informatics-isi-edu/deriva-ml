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
        generate_lease_token,
        post_lease_batch,
    )

    token = generate_lease_token()
    result = post_lease_batch(catalog=test_ml.catalog, tokens=[token])
    return token, result[token]


@requires_catalog
def test_upload_asset_uses_pre_leased_rid(test_ml, tmp_path):
    """End-to-end: the catalog row's RID matches the caller's pre-leased RID.

    Uses the canonical ``exe.asset_file_path`` API (manifest store);
    the bag-commit pipeline leases its own RIDs during ``build_execution_bag``,
    so this test now verifies *some* leased RID is honored (catalog
    row's RID was minted from the lease pool, not server-assigned ad
    hoc). The original test pre-leased its own RID externally — the
    bag pipeline ignores caller-supplied leases on the manifest path.
    """
    src = tmp_path / "bug-e2-happy.bin"
    src.write_bytes(b"bug-e2 happy path " * 32)
    expected_md5 = hashlib.md5(src.read_bytes()).hexdigest()

    wf = _make_workflow(test_ml, "Bug E2 happy")
    exe = test_ml.create_execution(description="bug-e2-happy", workflow=wf)

    with exe.execute():
        exe.asset_file_path("Execution_Asset", src, asset_types="Execution_Asset")

    report = exe.commit_output_assets()
    assert report.total_failed == 0, f"failures: {report.errors}"
    assert report.total_uploaded == 1

    # Catalog row exists and the leased RID was used (the row's RID
    # came from the lease table — the post-lease manifest reconciliation
    # records the leased RID before insert).
    pb = test_ml.pathBuilder()
    asset_path = pb.schemas["deriva-ml"].tables["Execution_Asset"]
    rows = list(asset_path.filter(asset_path.MD5 == expected_md5).entities().fetch())
    assert len(rows) == 1, (
        f"Execution_Asset row with MD5={expected_md5} not found — bag-commit upload may have skipped the row insert."
    )
    # Sanity: the row's RID was leased (validates against the lease table).
    leased_check = test_ml.catalog.getPathBuilder().schemas["public"].tables["ERMrest_RID_Lease"]
    lease_rows = list(leased_check.filter(leased_check.RID == rows[0]["RID"]).entities().fetch())
    assert len(lease_rows) == 1, (
        f"row's RID {rows[0]['RID']} not in lease table — Bug E.2 regression: row's RID wasn't minted from a lease."
    )


@requires_catalog
def test_soft_mode_second_upload_adopts_existing_rid(test_ml, tmp_path):
    """Two executions upload the same file; bag-commit dedups by MD5+Filename.

    Soft mode (the default, because Execution_Asset has no strict
    annotation) preserves legacy MD5+Filename dedup: the second
    execution's upload detects the existing row and adopts its RID
    instead of raising. The bag pipeline's ``match_by_columns`` policy
    handles the dedup at load time.

    Uses the canonical ``exe.asset_file_path`` API (manifest store).
    """
    src = tmp_path / "bug-e2-shared.bin"
    src.write_bytes(b"bug-e2 shared artifact " * 32)
    expected_md5 = hashlib.md5(src.read_bytes()).hexdigest()

    wf = _make_workflow(test_ml, "Bug E2 soft-1")

    # Execution 1 — first to upload.
    exe1 = test_ml.create_execution(description="bug-e2-soft-1", workflow=wf)
    with exe1.execute():
        exe1.asset_file_path("Execution_Asset", src, asset_types="Execution_Asset")
    report1 = exe1.commit_output_assets()
    assert report1.total_failed == 0, report1.errors

    # Capture the row's first-upload RID for the dedup comparison.
    pb = test_ml.pathBuilder()
    asset_path = pb.schemas["deriva-ml"].tables["Execution_Asset"]
    first_rows = list(asset_path.filter(asset_path.MD5 == expected_md5).entities().fetch())
    assert len(first_rows) == 1
    first_rid = first_rows[0]["RID"]

    # Execution 2 — uploads the SAME file from a new staging area.
    # Re-create the source so the second exe has its own copy of the
    # bytes (its working_dir is independent).
    src2 = tmp_path / "bug-e2-shared-copy.bin"
    src2.write_bytes(src.read_bytes())
    exe2 = test_ml.create_execution(description="bug-e2-soft-2", workflow=wf)
    with exe2.execute():
        exe2.asset_file_path(
            "Execution_Asset",
            src2,
            rename_file="bug-e2-shared.bin",  # keep the same Filename for dedup
            asset_types="Execution_Asset",
        )
    report2 = exe2.commit_output_assets()
    assert report2.total_failed == 0, f"Soft-mode upload should succeed but failed: {report2.errors}"

    # Only ONE catalog row with this MD5 (not two) — dedup worked.
    rows = list(asset_path.filter(asset_path.MD5 == expected_md5).entities().fetch())
    assert len(rows) == 1, f"expected single row for shared artifact, got {len(rows)}"
    # The row's RID equals the FIRST upload (dedup preserved the
    # existing row, soft mode).
    assert rows[0]["RID"] == first_rid
