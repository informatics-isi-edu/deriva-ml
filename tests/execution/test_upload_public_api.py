"""Tests for the upload public API — upload_pending, upload_outputs.

Post-WI2 these test surfaces:

* ``DerivaML.upload_pending`` — drives ``Execution._bag_commit_upload``
  per execution and aggregates the outcomes into an
  :class:`UploadReport`.
* ``Execution.upload_outputs`` and ``ExecutionRecord.upload_outputs`` —
  thin convenience wrappers that call ``upload_pending`` with the
  single execution's RID.

The legacy non-blocking ``_start_upload`` / ``UploadJob`` surface was
retired in WI2; for survive-process uploads callers should run
``deriva-ml upload`` as a subprocess instead.
"""

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


def test_upload_pending_drives_bag_commit_per_execution(test_ml, monkeypatch):
    """``upload_pending`` invokes ``_bag_commit_upload`` once per execution.

    Post-WI2 the implementation iterates the caller-supplied
    ``execution_rids`` (or every execution in the workspace registry
    when ``None``), resumes each :class:`Execution`, and calls its
    ``_bag_commit_upload`` method. Each successful call bumps
    ``UploadReport.total_uploaded`` by one; failures are caught and
    surfaced via ``total_failed`` + ``errors``.
    """
    from deriva_ml.execution.upload_report import UploadReport

    wf = _make_workflow(test_ml, "WI2 pending bag-commit")
    exe = test_ml.create_execution(description="pub", workflow=wf)

    calls: list[str] = []

    def _fake_bag_commit(self, progress_callback=None):
        calls.append(self.execution_rid)
        return {}

    # Patch the per-execution bag-commit so the test doesn't require a
    # live destination catalog.
    from deriva_ml.execution.execution import Execution

    monkeypatch.setattr(Execution, "_bag_commit_upload", _fake_bag_commit)

    report = test_ml.upload_pending(execution_rids=[exe.execution_rid])

    assert isinstance(report, UploadReport)
    assert report.execution_rids == [exe.execution_rid]
    assert report.total_uploaded == 1
    assert report.total_failed == 0
    assert report.errors == []
    assert calls == [exe.execution_rid]


def test_upload_pending_aggregates_failures(test_ml, monkeypatch):
    """Per-execution failures appear in ``total_failed`` and ``errors``."""
    from deriva_ml.core.exceptions import DerivaMLException
    from deriva_ml.execution.execution import Execution
    from deriva_ml.execution.upload_report import UploadReport

    wf = _make_workflow(test_ml, "WI2 pending fail")
    exe_ok = test_ml.create_execution(description="ok", workflow=wf)
    exe_fail = test_ml.create_execution(description="fail", workflow=wf)

    def _fake_bag_commit(self, progress_callback=None):
        if self.execution_rid == exe_fail.execution_rid:
            raise DerivaMLException("simulated bag-commit failure")
        return {}

    monkeypatch.setattr(Execution, "_bag_commit_upload", _fake_bag_commit)

    report = test_ml.upload_pending(
        execution_rids=[exe_ok.execution_rid, exe_fail.execution_rid]
    )

    assert isinstance(report, UploadReport)
    assert report.total_uploaded == 1
    assert report.total_failed == 1
    assert len(report.errors) == 1
    assert exe_fail.execution_rid in report.errors[0]
    # Failure on exe_fail did not skip exe_ok — both were attempted.
    assert set(report.execution_rids) == {exe_ok.execution_rid, exe_fail.execution_rid}


def test_exe_upload_outputs_delegates_to_upload_pending(test_ml, monkeypatch):
    """``Execution.upload_outputs`` is sugar for ``upload_pending(execution_rids=[self.rid])``."""
    from deriva_ml.execution.upload_report import UploadReport

    wf = _make_workflow(test_ml, "WI2 exe sugar")
    exe = test_ml.create_execution(description="sugar", workflow=wf)

    calls = []

    def _fake(self, **kw):
        calls.append(kw)
        return UploadReport(
            execution_rids=kw["execution_rids"] or [],
            total_uploaded=0,
            total_failed=0,
            per_table={},
        )

    monkeypatch.setattr(test_ml.__class__, "upload_pending", _fake)

    exe.upload_outputs()
    assert calls[0]["execution_rids"] == [exe.execution_rid]


def test_record_upload_outputs_delegates(test_ml, monkeypatch):
    """``ExecutionRecord.upload_outputs`` is sugar for the same call."""
    from deriva_ml.execution.upload_report import UploadReport

    wf = _make_workflow(test_ml, "WI2 record sugar")
    exe = test_ml.create_execution(description="rec-upload", workflow=wf)

    calls = []

    def _fake(self, **kw):
        calls.append(kw)
        return UploadReport(
            execution_rids=kw["execution_rids"] or [],
            total_uploaded=0,
            total_failed=0,
            per_table={},
        )

    monkeypatch.setattr(test_ml.__class__, "upload_pending", _fake)

    rec = next(r for r in test_ml.list_executions() if r.rid == exe.execution_rid)
    rec.upload_outputs(ml=test_ml)
    assert calls[0]["execution_rids"] == [exe.execution_rid]
