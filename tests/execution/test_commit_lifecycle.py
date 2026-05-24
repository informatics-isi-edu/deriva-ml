"""Pinning tests for the three commit-output-assets entry points.

Per ADR-0009, three entry points commit an execution's outputs:

1. **Inline** — ``exe.commit_output_assets()`` called directly on
   the live Execution returned by ``ml.create_execution(...)``.
2. **Resumed** — ``ml.resume_execution(rid).commit_output_assets()``
   on a fresh Execution rebuilt from the workspace registry. Models
   the recovery-after-restart flow.
3. **Batch / CLI** — ``ml.commit_pending_executions(execution_rids=[rid])``
   on a list of pending executions. The ``deriva-ml-upload`` CLI is a
   thin wrapper around this method.

This test pins the contract that all three produce **identical end
state**:

* Execution status transitioned to ``Uploaded`` in the live catalog.
* All output assets present in the catalog with descriptions written.
* ``Asset_Role="Output"`` on every ``{Asset}_Execution`` row.
* ``Output_File`` Asset_Type tag on every committed asset (per PR
  #220's directional-tag contract).
* ``Upload_Duration`` recorded in the SQLite registry.

If any of the three paths drifts away from the others in the future,
this test fails — that's the point. It's the load-bearing regression
guard for the unification.
"""

from __future__ import annotations

import pytest

from deriva_ml import ExecAssetType, MLAsset
from deriva_ml import MLVocab as vc  # noqa: N812
from deriva_ml.execution.execution import Execution, ExecutionConfiguration


# ---------------------------------------------------------------------------
# Shared fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def commit_workflow(test_ml):
    """Test workflow term + Workflow object, shared by all three tests."""
    test_ml.add_term(
        vc.workflow_type,
        "Test Workflow",
        description="for commit-lifecycle pinning tests",
    )
    return test_ml.create_workflow(
        name="commit-lifecycle workflow",
        workflow_type="Test Workflow",
        description="commit-lifecycle pinning workflow",
    )


def _make_pending_execution(test_ml, commit_workflow, label: str) -> Execution:
    """Build an execution in status=Stopped with one staged output asset.

    Mirrors the "kernel finished its run, manifest has pending work"
    state that every commit entry point is supposed to drain.
    """
    cfg = ExecutionConfiguration(
        description=f"commit-lifecycle {label}",
        workflow=commit_workflow,
    )
    exe = test_ml.create_execution(cfg)
    with exe.execute() as inner:
        asset_path = inner.asset_file_path(
            MLAsset.execution_asset,
            f"CommitLifecycle/{label}.txt",
            asset_types=ExecAssetType.model_file,
            description=f"commit-lifecycle output for {label}",
        )
        with asset_path.open("w") as fp:
            fp.write(f"commit-lifecycle bytes for {label}\n")
    # ``with`` block exit transitions Running → Stopped.
    return exe


def _assert_full_lifecycle(test_ml, execution_rid: str) -> None:
    """Assertion bundle shared by all three commit-path tests.

    Checks the end state every commit entry point is contracted to
    produce. The three calling paths exist so that any future
    refactor that breaks one (e.g., a new batch shortcut that skips
    descriptions) fails here on each path that drifts.
    """
    # 1. Catalog status = Uploaded.
    catalog_row = test_ml._retrieve_rid(execution_rid)
    assert catalog_row["Status"] == "Uploaded", (
        f"commit path must transition status to Uploaded; saw {catalog_row['Status']!r}"
    )

    # 2. Upload_Duration recorded in SQLite + mirrored to catalog.
    store = test_ml.workspace.execution_state_store()
    sqlite_row = store.get_execution(execution_rid)
    assert sqlite_row["upload_duration"] is not None, (
        "Upload_Duration must be recorded in SQLite by the lifecycle bracket — "
        "absent value means the commit path skipped the bracket (one of the "
        "two latent bugs ADR-0009 fixes)"
    )
    assert catalog_row["Upload_Duration"] is not None, (
        "Upload_Duration must mirror to the catalog Execution row — "
        "absent value means the state-machine PUT did not carry the measurement"
    )

    # 3. Output assets present at the catalog with descriptions.
    pb = test_ml.pathBuilder().schemas[test_ml.ml_schema]
    asset_exec = pb.Execution_Asset_Execution
    asset = pb.Execution_Asset
    assoc_path = asset_exec.path
    assoc_path.filter(asset_exec.Execution == execution_rid)
    assoc_path.link(asset)
    asset_rows = list(assoc_path.entities().fetch())
    assert asset_rows, "commit must surface Execution_Asset rows for this execution"
    for row in asset_rows:
        # Description written by ``_set_asset_descriptions`` — the
        # second latent bug ADR-0009 fixes (upload_outputs silently
        # skipped this).
        assert row.get("Description"), (
            "asset description must be written by the lifecycle bracket — "
            "absent value means the commit path skipped _set_asset_descriptions"
        )

    # 4. Asset_Role="Output" on every {Asset}_Execution row + Output_File tag.
    for row in asset_rows:
        assert row.get("Asset_Role") == "Output", (
            f"every committed asset must carry Asset_Role=Output on its "
            f"{{Asset}}_Execution row; saw {row.get('Asset_Role')!r} for {row['RID']}"
        )

    # 5. Output_File directional tag on every committed asset (PR #220 contract).
    asset_type_assoc = pb.Execution_Asset_Type
    for asset_row in asset_rows:
        type_path = asset_type_assoc.path
        type_path.filter(asset_type_assoc.Execution_Asset == asset_row["Execution_Asset"])
        types = {t["Asset_Type"] for t in type_path.attributes(asset_type_assoc.Asset_Type).fetch()}
        assert "Output_File" in types, (
            f"every committed asset must carry the Output_File directional tag "
            f"(PR #220 contract); asset {asset_row['Execution_Asset']} has tags {types}"
        )


# ---------------------------------------------------------------------------
# The three paths — each must produce the same end state.
# ---------------------------------------------------------------------------


def test_inline_commit_produces_full_lifecycle(test_ml, commit_workflow):
    """Path 1 — ``exe.commit_output_assets()`` called inline."""
    exe = _make_pending_execution(test_ml, commit_workflow, "inline")

    exe.commit_output_assets()

    _assert_full_lifecycle(test_ml, exe.execution_rid)


def test_resumed_commit_produces_full_lifecycle(test_ml, commit_workflow):
    """Path 2 — ``ml.resume_execution(rid).commit_output_assets()``.

    Models the recovery-after-restart flow: a separate process built
    the execution, the running session ended, a fresh session resumes
    the execution and commits its pending outputs.
    """
    exe = _make_pending_execution(test_ml, commit_workflow, "resumed")
    rid = exe.execution_rid

    # Drop the live reference and resume via the registry — the
    # canonical pattern after a process crash or for the CLI use case.
    resumed = test_ml.resume_execution(rid)
    resumed.commit_output_assets()

    _assert_full_lifecycle(test_ml, rid)


def test_batch_commit_produces_full_lifecycle(test_ml, commit_workflow):
    """Path 3 — ``ml.commit_pending_executions(execution_rids=[rid])``.

    The CLI (``deriva-ml-upload``) is a thin wrapper around this
    method, so this test also pins the CLI path's end state via the
    same code path the CLI takes.
    """
    exe = _make_pending_execution(test_ml, commit_workflow, "batch")
    rid = exe.execution_rid

    report = test_ml.commit_pending_executions(execution_rids=[rid])
    assert report.total_failed == 0
    assert rid in report.execution_rids

    _assert_full_lifecycle(test_ml, rid)
