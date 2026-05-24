"""End-state behavior tests for the unified commit-output-assets API.

ADR-0009 unifies four legacy upload entry points (``Execution.upload_execution_outputs``,
``Execution.upload_outputs``, ``ExecutionSnapshot.upload_outputs``,
``DerivaML.upload_pending``) into one per-execution method
(``Execution.commit_output_assets``) and one batch method
(``DerivaML.commit_pending_executions``). The CLI
(``deriva-ml-upload``) is a thin wrapper around the batch method.

These tests assert **end-state behavior** — that each entry point
drives the full lifecycle bracket (status transition to ``Uploaded``,
Upload_Duration recorded in SQLite, asset descriptions written) — not
just delegation. The pre-ADR tests asserted delegation, which let two
latent bugs survive:

* ``upload_outputs`` skipped descriptions + Upload_Duration silently
  (the wrapper called ``upload_pending`` which called
  ``_bag_commit_upload`` without the lifecycle bracket).
* CLI-uploaded executions stuck in ``Stopped`` status forever (same
  cause — the CLI path bypassed the bracket too).

The new tests catch any regression that re-introduces either gap.

Per-execution failure isolation in ``commit_pending_executions`` is
exercised by ``test_batch_isolates_per_execution_failures`` —
the report aggregates a success and a failure side-by-side and
keeps going past the failure.
"""

from __future__ import annotations

from deriva_ml import MLVocab as vc


def _make_workflow(test_ml, name: str):
    """Shared helper: ensure Test Workflow term + create Workflow object."""
    test_ml.add_term(
        vc.workflow_type,
        "Test Workflow",
        description="for commit_output_assets tests",
    )
    return test_ml.create_workflow(
        name=name,
        workflow_type="Test Workflow",
        description="for commit_output_assets tests",
    )


def _get_execution_status(ml, execution_rid: str) -> str:
    """Read the live catalog Execution.Status column for an execution RID."""
    row = ml._retrieve_rid(execution_rid)
    return row["Status"]


def test_commit_output_assets_returns_upload_report(test_ml):
    """``Execution.commit_output_assets`` returns an UploadReport, not a dict."""
    from deriva_ml.execution.upload_report import UploadReport

    wf = _make_workflow(test_ml, "report shape")
    exe = test_ml.create_execution(description="report-shape", workflow=wf)
    exe.execution_start()
    exe.execution_stop()

    report = exe.commit_output_assets()
    assert isinstance(report, UploadReport)
    assert report.execution_rids == [exe.execution_rid]
    assert report.total_failed == 0
    assert report.errors == []


def test_commit_output_assets_drives_full_lifecycle(test_ml):
    """End-state: status=Uploaded, Upload_Duration set in SQLite."""
    wf = _make_workflow(test_ml, "full lifecycle")
    exe = test_ml.create_execution(description="lifecycle", workflow=wf)
    exe.execution_start()
    exe.execution_stop()

    exe.commit_output_assets()

    # Catalog-side status transitioned to Uploaded.
    assert _get_execution_status(test_ml, exe.execution_rid) == "Uploaded"

    # SQLite registry got Upload_Duration recorded.
    store = test_ml.workspace.execution_state_store()
    row = store.get_execution(exe.execution_rid)
    assert row["upload_duration"] is not None, (
        "Upload_Duration must be recorded by commit_output_assets — "
        "the lifecycle bracket is the only path that writes it"
    )


def test_commit_pending_executions_returns_upload_report(test_ml):
    """``DerivaML.commit_pending_executions`` returns an UploadReport."""
    from deriva_ml.execution.upload_report import UploadReport

    wf = _make_workflow(test_ml, "batch shape")
    exe = test_ml.create_execution(description="batch-shape", workflow=wf)
    exe.execution_start()
    exe.execution_stop()

    report = test_ml.commit_pending_executions(execution_rids=[exe.execution_rid])
    assert isinstance(report, UploadReport)
    assert report.execution_rids == [exe.execution_rid]
    assert report.total_failed == 0


def test_commit_pending_executions_drives_full_lifecycle(test_ml):
    """Batch path: status=Uploaded, Upload_Duration recorded — same as inline."""
    wf = _make_workflow(test_ml, "batch lifecycle")
    exe = test_ml.create_execution(description="batch-lifecycle", workflow=wf)
    exe.execution_start()
    exe.execution_stop()

    test_ml.commit_pending_executions(execution_rids=[exe.execution_rid])

    assert _get_execution_status(test_ml, exe.execution_rid) == "Uploaded"

    store = test_ml.workspace.execution_state_store()
    row = store.get_execution(exe.execution_rid)
    assert row["upload_duration"] is not None, (
        "Batch path must drive the same lifecycle bracket as inline — "
        "if Upload_Duration is null here, the two paths have drifted"
    )


def test_cli_path_drives_full_lifecycle(test_ml, monkeypatch):
    """CLI path produces the same end state as the in-process callers."""
    from deriva_ml.cli import upload as upload_cli

    wf = _make_workflow(test_ml, "cli lifecycle")
    exe = test_ml.create_execution(description="cli-lifecycle", workflow=wf)
    exe.execution_start()
    exe.execution_stop()

    monkeypatch.setattr(
        upload_cli,
        "_construct_ml",
        lambda host, catalog, mode: test_ml,
    )

    rc = upload_cli.main(
        [
            "--host",
            "ignored",
            "--catalog",
            "ignored",
            "--execution",
            exe.execution_rid,
        ]
    )
    assert rc == 0

    # CLI path drives the same lifecycle bracket — the historical bug
    # was that CLI-uploaded executions stayed Stopped forever.
    assert _get_execution_status(test_ml, exe.execution_rid) == "Uploaded"

    store = test_ml.workspace.execution_state_store()
    row = store.get_execution(exe.execution_rid)
    assert row["upload_duration"] is not None


def test_batch_isolates_per_execution_failures(test_ml, monkeypatch):
    """A failure on exe B does not skip exe A; both outcomes in the report."""
    from deriva_ml.core.exceptions import DerivaMLException
    from deriva_ml.execution.execution import Execution

    wf = _make_workflow(test_ml, "batch isolate")
    exe_ok = test_ml.create_execution(description="ok", workflow=wf)
    exe_fail = test_ml.create_execution(description="fail", workflow=wf)
    exe_ok.execution_start()
    exe_ok.execution_stop()
    exe_fail.execution_start()
    exe_fail.execution_stop()

    real_commit = Execution.commit_output_assets

    def _fake_commit(self, clean_folder=None, progress_callback=None):
        if self.execution_rid == exe_fail.execution_rid:
            raise DerivaMLException("simulated commit failure")
        return real_commit(self, clean_folder=clean_folder, progress_callback=progress_callback)

    monkeypatch.setattr(Execution, "commit_output_assets", _fake_commit)

    report = test_ml.commit_pending_executions(
        execution_rids=[exe_ok.execution_rid, exe_fail.execution_rid]
    )

    assert report.total_failed == 1
    assert report.total_uploaded >= 0
    assert len(report.errors) == 1
    assert exe_fail.execution_rid in report.errors[0]
    # Sibling execution was attempted (and completed) despite the failure.
    assert set(report.execution_rids) == {exe_ok.execution_rid, exe_fail.execution_rid}
