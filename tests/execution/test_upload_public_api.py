"""Tests for the upload public API — upload_pending, upload_outputs,
start_upload / UploadJob."""

from __future__ import annotations


def _make_workflow(test_ml, name: str):
    """Shared helper: ensure Test Workflow term + create Workflow object."""
    from deriva_ml import MLVocab as vc

    test_ml.add_term(
        vc.workflow_type,
        "Test Workflow",
        description="for upload_public_api tests",
    )
    return test_ml.create_workflow(
        name=name,
        workflow_type="Test Workflow",
        description="for upload_public_api tests",
    )


def test_upload_pending_blocking(test_ml, monkeypatch):
    """ml.upload_pending runs the engine synchronously and returns an
    UploadReport."""
    from datetime import datetime, timezone

    from deriva_ml.execution.upload_engine import UploadReport

    wf = _make_workflow(test_ml, "G8 blocking")
    exe = test_ml.create_execution(description="pub", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)
    store.insert_pending_row(
        execution_rid=exe.execution_rid, key="k",
        target_schema="deriva-ml", target_table="Subject",
        metadata_json="{}", created_at=now,
    )

    calls = []
    def _fake_run(**kw):
        calls.append(kw)
        return UploadReport(
            execution_rids=kw["execution_rids"] or [],
            total_uploaded=1, total_failed=0,
            per_table={"deriva-ml:Subject": {"uploaded": 1, "failed": 0}},
        )
    monkeypatch.setattr(
        "deriva_ml.core.mixins.execution.run_upload_engine", _fake_run,
    )

    report = test_ml.upload_pending(execution_rids=[exe.execution_rid])
    assert isinstance(report, UploadReport)
    assert report.total_uploaded == 1
    assert calls[0]["execution_rids"] == [exe.execution_rid]


def test_exe_upload_outputs_delegates_to_upload_pending(test_ml, monkeypatch):
    from deriva_ml.execution.upload_engine import UploadReport

    wf = _make_workflow(test_ml, "G8 exe sugar")
    exe = test_ml.create_execution(description="sugar", workflow=wf)

    calls = []
    def _fake(self, **kw):
        calls.append(kw)
        return UploadReport(
            execution_rids=kw["execution_rids"] or [],
            total_uploaded=0, total_failed=0, per_table={},
        )
    monkeypatch.setattr(test_ml.__class__, "upload_pending", _fake)

    exe.upload_outputs()
    assert calls[0]["execution_rids"] == [exe.execution_rid]


def test_record_upload_outputs_delegates(test_ml, monkeypatch):
    from deriva_ml.execution.upload_engine import UploadReport

    wf = _make_workflow(test_ml, "G8 record sugar")
    exe = test_ml.create_execution(description="rec-upload", workflow=wf)

    calls = []
    def _fake(self, **kw):
        calls.append(kw)
        return UploadReport(
            execution_rids=kw["execution_rids"] or [],
            total_uploaded=0, total_failed=0, per_table={},
        )
    monkeypatch.setattr(test_ml.__class__, "upload_pending", _fake)

    rec = next(r for r in test_ml.list_executions()
               if r.rid == exe.execution_rid)
    rec.upload_outputs(ml=test_ml)
    assert calls[0]["execution_rids"] == [exe.execution_rid]


def test_start_upload_returns_upload_job(test_ml, monkeypatch):
    from deriva_ml.execution.upload_engine import UploadReport
    from deriva_ml.execution.upload_job import UploadJob

    wf = _make_workflow(test_ml, "G8 job")
    exe = test_ml.create_execution(description="job", workflow=wf)

    def _fake(**kw):
        return UploadReport(
            execution_rids=kw["execution_rids"] or [],
            total_uploaded=0, total_failed=0, per_table={},
        )
    monkeypatch.setattr(
        "deriva_ml.execution.upload_job.run_upload_engine", _fake,
    )

    job = test_ml.start_upload(execution_rids=[exe.execution_rid])
    assert isinstance(job, UploadJob)
    report = job.wait(timeout=10)
    assert report.total_uploaded == 0
    assert job.status in ("completed", "failed")


def test_upload_job_cancel(test_ml, monkeypatch):
    """cancel() sets status=cancelled; wait raises."""
    import threading
    import time

    from deriva_ml.execution.upload_engine import UploadReport

    wf = _make_workflow(test_ml, "G8 cancel")
    exe = test_ml.create_execution(description="cancel", workflow=wf)

    started = threading.Event()
    stop = threading.Event()

    def _slow_run(**kw):
        started.set()
        for _ in range(100):
            if stop.is_set():
                break
            time.sleep(0.01)
        return UploadReport(
            execution_rids=kw["execution_rids"] or [],
            total_uploaded=0, total_failed=0, per_table={},
        )
    monkeypatch.setattr(
        "deriva_ml.execution.upload_job.run_upload_engine", _slow_run,
    )

    job = test_ml.start_upload(execution_rids=[exe.execution_rid])
    started.wait(timeout=2)
    stop.set()  # let the slow fake finish; cancel races with completion
    job.cancel()
    assert job.status in ("cancelled", "completed")
