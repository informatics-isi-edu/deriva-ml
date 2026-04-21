"""Live-catalog smoke test for the S3 upload engine.

Unlike `test_upload_engine_deriva_py.py` (which uses FakeGenericUploader),
this test drives the REAL `GenericUploader` against a real catalog. It
exercises the full path:

    exe.upload_outputs()
      -> ml.upload_pending()
      -> run_upload_engine()
      -> _drain_work_item()
      -> _invoke_deriva_py_uploader()
      -> GenericUploader.scanDirectory + uploadFiles
      -> hatrac + catalog record insert

It is gated on DERIVA_HOST — without a live catalog, it skips.

Purpose: pre-merge smoke test. Not intended as part of the regular
CI suite (would fail in CI that has no catalog), but useful locally
and as a reproducible manual check.
"""
from __future__ import annotations

import os

import pytest


def _make_workflow(test_ml, name: str):
    from deriva_ml import MLVocab as vc

    test_ml.add_term(
        vc.workflow_type,
        "Test Workflow",
        description="for upload_engine live smoke test",
    )
    return test_ml.create_workflow(
        name=name,
        workflow_type="Test Workflow",
        description="for upload_engine live smoke test",
    )


@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="Live-catalog smoke test requires DERIVA_HOST to be set",
)
def test_upload_engine_live_end_to_end(test_ml, tmp_path):
    """End-to-end: stage a real asset file, upload via GenericUploader,
    verify it lands in the catalog's Execution_Asset table.

    Uses Execution_Asset because it has zero user-metadata columns,
    avoiding Bug C (see test_bug_c_none_stringification below) which
    pre-dates S3 but affects any asset table with non-string metadata.
    """
    import hashlib
    import json
    from datetime import datetime, timezone

    from deriva_ml.execution.state_store import PendingRowStatus

    # Small on-disk file — enough to upload, small enough to not slow
    # the test down.
    f = tmp_path / "smoke.bin"
    f.write_bytes(b"S3 smoke payload " * 64)

    wf = _make_workflow(test_ml, "S3 live smoke")
    exe = test_ml.create_execution(description="live-smoke", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)

    # Drive execution through Created → Running → Stopped so the
    # state machine allows Stopped → Pending_Upload.
    with exe.execute():
        pass

    # Stage a pending asset row in SQLite. Status=leased so the engine
    # treats it as "ready to drain" without needing lease acquisition
    # (which would require a live rid_lease call).
    # Execution_Asset requires no metadata columns, so the pending row
    # carries an empty dict.
    store.insert_pending_row(
        execution_rid=exe.execution_rid,
        key="smoke-k",
        target_schema="deriva-ml",
        target_table="Execution_Asset",
        metadata_json=json.dumps({}),
        created_at=now,
        rid="EA-SMOKE-1",
        status=PendingRowStatus.leased,
        lease_token="smoke-lease",
        asset_file_path=str(f),
    )

    # Fire the real uploader.
    report = exe.upload_outputs()

    # Assertions on the UploadReport.
    assert report.total_failed == 0, f"upload had failures: {report.errors}"
    assert report.total_uploaded == 1, (
        f"expected 1 uploaded row, got {report.total_uploaded}; "
        f"per_table={report.per_table}"
    )

    # The SQLite row should be terminal=uploaded.
    rows = store.list_pending_rows(execution_rid=exe.execution_rid)
    assert len(rows) == 1
    assert rows[0]["status"] == str(PendingRowStatus.uploaded), rows[0]
    assert rows[0]["uploaded_at"] is not None

    # The catalog should have an Execution_Asset row for our upload.
    # deriva-py looks up / upserts by MD5 + Filename (see
    # record_query_template in bulk_upload_configuration), so we query
    # by those rather than by the pre-leased RID (see Bug E below).
    expected_md5 = hashlib.md5(f.read_bytes()).hexdigest()
    pb = test_ml.pathBuilder()
    asset_path = pb.schemas["deriva-ml"].tables["Execution_Asset"]
    results = list(
        asset_path.filter(asset_path.MD5 == expected_md5)
        .filter(asset_path.Filename == "smoke.bin")
        .entities().fetch()
    )
    assert len(results) == 1, (
        f"Execution_Asset row with MD5={expected_md5}, Filename=smoke.bin "
        f"not found in catalog; got {len(results)} results"
    )
    row = results[0]
    # URL should be a hatrac URI.
    assert row["URL"] is not None, row
    assert "/hatrac/" in row["URL"], row


@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="Bug-C reproducer requires DERIVA_HOST to be set",
)
@pytest.mark.xfail(
    reason="Bug C: _invoke_deriva_py_uploader substitutes the literal "
    "string 'None' for missing asset metadata values. This string "
    "flows through deriva-py's column_map and is inserted into the "
    "catalog, which rejects it for non-string columns (e.g., "
    "'invalid input syntax for type timestamp: \"None\"'). The same "
    "bug exists in Execution._build_upload_staging (pre-S3). Fix "
    "requires a design decision on how to handle missing metadata: "
    "require callers to supply all values, filter missing columns "
    "out of the scan path, or send NULL rather than the string "
    "'None' to the catalog.",
    strict=False,  # Bug C may fail in multiple ways; don't require exact match
)
def test_bug_c_none_stringification_corrupts_non_string_metadata(test_ml, tmp_path):
    """Reproducer for Bug C — asset metadata that's None-stringified
    corrupts non-string catalog columns.

    Stages an Image asset (which has `Acquisition_Time` timestamp and
    `Acquisition_Date` date metadata columns) without supplying values
    for those columns. The uploader should recognize missing metadata
    and either reject the upload early or send NULL to the catalog —
    but today it sends the literal string 'None', causing a 400 Bad
    Request from the server.

    When this test starts passing (either naturally or with a fix),
    Bug C is resolved; remove the xfail mark.
    """
    import json
    from datetime import datetime, timezone

    from deriva_ml.execution.state_store import PendingRowStatus

    f = tmp_path / "bug-c.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\nfake-png-payload" * 64)

    wf = _make_workflow(test_ml, "S3 bug-C repro")
    exe = test_ml.create_execution(description="bug-c-repro", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)

    with exe.execute():
        pass

    # Stage an Image row with NO metadata. Image has non-string
    # metadata columns (Acquisition_Time timestamp, Acquisition_Date
    # date) that Bug C will mishandle.
    store.insert_pending_row(
        execution_rid=exe.execution_rid,
        key="bug-c-k",
        target_schema="test-schema",
        target_table="Image",
        metadata_json=json.dumps({}),
        created_at=now,
        rid="IMG-BUGC-1",
        status=PendingRowStatus.leased,
        lease_token="bug-c-lease",
        asset_file_path=str(f),
    )

    report = exe.upload_outputs()

    # When Bug C is fixed, this assertion should pass.
    assert report.total_uploaded == 1, (
        f"expected 1 uploaded (Bug C must be fixed for this to pass); "
        f"errors={report.errors}"
    )
