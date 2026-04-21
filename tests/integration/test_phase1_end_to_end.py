"""End-to-end Phase 1 integration tests.

Mirrors the three-script walkthrough in spec §3:
online create → offline stage → online upload.

Also covers crash-resume semantics: if run_upload_engine raises
mid-drain, a second upload_pending call must cleanly drain the rest.

Requires DERIVA_HOST to point at a live catalog.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


def _make_workflow(ml, name: str):
    """Ensure Test Workflow term + create Workflow object."""
    from deriva_ml import MLVocab as vc

    ml.add_term(
        vc.workflow_type,
        "Test Workflow",
        description="for Phase 1 e2e tests",
    )
    return ml.create_workflow(
        name=name,
        workflow_type="Test Workflow",
        description="for Phase 1 e2e tests",
    )


@pytest.mark.integration
def test_online_create_offline_stage_online_upload(catalog_manager, tmp_path):
    """Three-phase workflow producing a durable catalog result.

    Creates an execution online, stages a Subject row via SQLite in an
    offline DerivaML instance sharing the same working_dir, then uploads
    via a fresh online instance. Verifies the catalog row lands and the
    execution reaches ExecutionStatus.Uploaded.
    """
    from deriva_ml import ConnectionMode, DerivaML
    from deriva_ml.execution.state_store import ExecutionStatus

    # Clean slate — a fresh catalog state is enough; Subject already
    # exists in the domain schema after CatalogManager construction
    # (created by create_domain_schema during __post_init__).
    catalog_manager.reset()

    host = catalog_manager.hostname
    cid = catalog_manager.catalog_id
    domain_schema = catalog_manager.domain_schema

    # Step 1: online — create execution.
    ml_online = DerivaML(
        hostname=host,
        catalog_id=cid,
        default_schema=domain_schema,
        working_dir=str(tmp_path),
        use_minid=False,
        mode=ConnectionMode.online,
    )
    wf = _make_workflow(ml_online, "phase1 e2e")
    exe = ml_online.create_execution(description="phase1 e2e", workflow=wf)
    exe_rid = exe.execution_rid

    # Run the execution context (created → running → stopped).
    with exe.execute():
        pass

    # Step 2: offline — stage rows under the same working_dir.
    ml_offline = DerivaML(
        hostname=host,
        catalog_id=cid,
        default_schema=domain_schema,
        working_dir=str(tmp_path),
        use_minid=False,
        mode=ConnectionMode.offline,
    )
    exe_resumed = ml_offline.resume_execution(exe_rid)
    store = ml_offline.workspace.execution_state_store()
    now = datetime.now(timezone.utc)

    # Subject has a single `Name` text column (see demo_catalog.create_domain_schema).
    store.insert_pending_row(
        execution_rid=exe_rid,
        key="k1",
        target_schema=domain_schema,
        target_table="Subject",
        metadata_json='{"Name": "E2E-Alice"}',
        created_at=now,
    )

    summary = exe_resumed.pending_summary()
    assert summary.has_pending
    assert summary.total_pending_rows == 1

    # Step 3: online — upload via a fresh DerivaML instance.
    ml_upload = DerivaML(
        hostname=host,
        catalog_id=cid,
        default_schema=domain_schema,
        working_dir=str(tmp_path),
        use_minid=False,
        mode=ConnectionMode.online,
    )
    report = ml_upload.upload_pending(execution_rids=[exe_rid])
    assert report.total_failed == 0, f"Report: {report}"
    assert report.total_uploaded == 1, f"Report: {report}"

    # Verify the execution reached `Uploaded` in SQLite and pending work
    # drained. Post-S1a the catalog Status vocabulary is unified with
    # ExecutionStatus (title-case), so resume_execution's reconcile path
    # no longer trips on the legacy 'Pending' value — we can read through
    # the resumed Execution object directly.
    upload_store = ml_upload.workspace.execution_state_store()
    registry_row = upload_store.get_execution(exe_rid)
    assert registry_row is not None
    assert registry_row["status"] == str(ExecutionStatus.Uploaded)
    counts = upload_store.count_pending_by_kind(execution_rid=exe_rid)
    assert counts["pending_rows"] == 0
    assert counts["failed_rows"] == 0


@pytest.mark.integration
def test_crash_resume_after_interrupted_upload(
    catalog_manager, tmp_path, monkeypatch
):
    """Simulate process death during upload → re-run drains the rest.

    First upload_pending call is patched to mark two rows uploaded then
    raise. A second call with the real engine must drain the remaining
    three rows cleanly.
    """
    from deriva_ml import ConnectionMode, DerivaML
    from deriva_ml.execution.upload_engine import run_upload_engine

    catalog_manager.reset()

    host = catalog_manager.hostname
    cid = catalog_manager.catalog_id
    domain_schema = catalog_manager.domain_schema

    ml = DerivaML(
        hostname=host,
        catalog_id=cid,
        default_schema=domain_schema,
        working_dir=str(tmp_path),
        use_minid=False,
        mode=ConnectionMode.online,
    )
    wf = _make_workflow(ml, "crash-resume e2e")
    exe = ml.create_execution(description="crash-resume", workflow=wf)
    with exe.execute():
        pass

    store = ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)
    for i in range(5):
        store.insert_pending_row(
            execution_rid=exe.execution_rid,
            key=f"k{i}",
            target_schema=domain_schema,
            target_table="Subject",
            metadata_json=f'{{"Name": "CrashRes{i}"}}',
            created_at=now,
        )

    original = run_upload_engine
    call_count = {"n": 0}

    def _interrupted(**kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Simulate: some rows made it through before the crash.
            from deriva_ml.execution.state_store import PendingRowStatus

            rows = store.list_pending_rows(
                execution_rid=kw["execution_rids"][0],
            )[:2]
            for r in rows:
                store.update_pending_row(
                    r["id"],
                    status=PendingRowStatus.uploaded,
                    uploaded_at=datetime.now(timezone.utc),
                )
            raise RuntimeError("simulated crash mid-upload")
        return original(**kw)

    monkeypatch.setattr(
        "deriva_ml.core.mixins.execution.run_upload_engine", _interrupted,
    )

    with pytest.raises(RuntimeError, match="simulated crash"):
        ml.upload_pending(execution_rids=[exe.execution_rid])

    # Restore the real engine for the second call.
    monkeypatch.setattr(
        "deriva_ml.core.mixins.execution.run_upload_engine", original,
    )
    report = ml.upload_pending(execution_rids=[exe.execution_rid])
    assert report.total_failed == 0, f"Report: {report}"
    assert report.total_uploaded == 3, f"Report: {report}"

    counts = store.count_pending_by_kind(execution_rid=exe.execution_rid)
    assert counts["pending_rows"] == 0
    assert counts["failed_rows"] == 0
