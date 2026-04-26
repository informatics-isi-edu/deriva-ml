"""Tests for the execution module.

This module provides comprehensive tests for DerivaML execution functionality:

Test Classes:
    TestWorkflow: Workflow creation and management
    TestExecutionLifecycle: Context manager, start/stop, status tracking
    TestExecutionAssets: Asset upload/download operations
    TestAssetDescriptions: Asset description and metadata description resolution
    TestExecutionDatasets: Dataset operations within executions
    TestExecutionFeatures: Feature record management
    TestExecutionRestore: Execution restoration from previous runs
    TestExecutionDryRun: Dry run mode (no catalog writes)
    TestExecutionErrors: Error handling and edge cases
"""

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from deriva_ml import DerivaML, ExecAssetType, MLAsset
from deriva_ml import MLVocab as vc  # noqa: N812
from deriva_ml.core.definitions import BuiltinTypes, ColumnDefinition
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.execution.execution import Execution, ExecutionConfiguration

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def test_workflow(workflow_terms):
    """Create a test workflow."""
    ml = workflow_terms
    return ml.create_workflow(
        name="Test Workflow",
        workflow_type="Test Workflow",
        description="A test workflow for execution testing",
    )


@pytest.fixture
def basic_execution(workflow_terms, test_workflow):
    """Create a basic execution without datasets."""
    ml = workflow_terms
    config = ExecutionConfiguration(
        description="Test Execution",
        workflow=test_workflow,
    )
    return ml.create_execution(config)


@pytest.fixture
def feature_setup(workflow_terms):
    """Set up feature infrastructure for testing."""
    ml = workflow_terms
    # Create a simple feature with numeric Score column
    ml.create_feature(
        "Subject",
        "TestScore",
        metadata=[
            ColumnDefinition(name="Score", type=BuiltinTypes.int2, nullok=False),
        ],
    )

    return ml


# =============================================================================
# Helper Functions
# =============================================================================


def create_test_asset(execution: Execution, filename: str = "test_model.txt", content: str = "Test model content"):
    """Helper to create a test asset file within an execution."""
    asset_path = execution.asset_file_path(
        MLAsset.execution_asset,
        f"TestAsset/{filename}",
        asset_types=ExecAssetType.model_file,
    )
    with asset_path.open("w") as fp:
        fp.write(content)
    return asset_path


def get_execution_status(ml: DerivaML, execution_rid: str) -> str:
    """Get the current status of an execution."""
    return ml._retrieve_rid(execution_rid)["Status"]


def get_execution_metadata_files(ml: DerivaML, execution_rid: str) -> list[str]:
    """Get list of metadata filenames for an execution."""
    pb = ml.pathBuilder().schemas[ml.ml_schema]
    execution_metadata_execution = pb.Execution_Metadata_Execution
    execution_metadata = pb.Execution_Metadata
    metadata_path = execution_metadata_execution.path
    metadata_path.filter(execution_metadata_execution.Execution == execution_rid)
    metadata_path.link(execution_metadata)
    return [a["Filename"] for a in metadata_path.entities().fetch()]


# =============================================================================
# TestWorkflow - Workflow Creation Tests
# =============================================================================


class TestWorkflow:
    """Tests for workflow creation and management."""

    def test_workflow_creation_script(self, test_ml):
        """Test creating a workflow from a Python script."""
        ml_instance = test_ml
        ml_instance.add_term(vc.asset_type, "Test Model", description="Model for our Test workflow")
        ml_instance.add_term(vc.workflow_type, "Test Workflow", description="A ML Workflow that uses Deriva ML API")

        workflow_script = Path(__file__).parent / "workflow-test.py"
        workflow_table = ml_instance.pathBuilder().schemas[ml_instance.ml_schema].Workflow

        # Verify no workflows exist initially
        workflows = list(workflow_table.entities().fetch())
        assert len(workflows) == 0

        # Run the workflow script
        env = os.environ.copy()
        env["SETUPTOOLS_USE_DISTUTILS"] = "local"
        result = subprocess.run(
            [
                sys.executable,
                workflow_script.as_posix(),
                ml_instance.catalog.deriva_server.server,
                ml_instance.catalog_id,
            ],
            capture_output=True,
            text=True,
            env=env,
        )

        # Verify workflow was created
        workflows = list(workflow_table.entities().fetch())
        assert len(workflows) == 1
        workflow_rid = workflows[0]["RID"]
        workflow_url = workflows[0]["URL"]
        assert workflow_url.endswith("workflow-test.py")

        # Verify lookup_workflow_by_url returns the workflow
        looked_up_workflow = ml_instance.lookup_workflow_by_url(workflow_url)
        assert looked_up_workflow.rid == workflow_rid
        assert looked_up_workflow.url == workflow_url

        # Verify lookup_workflow works with RID
        looked_up_workflow_by_rid = ml_instance.lookup_workflow(workflow_rid)
        assert looked_up_workflow_by_rid.rid == workflow_rid
        assert looked_up_workflow_by_rid.url == workflow_url

        # Verify running again doesn't duplicate
        result = subprocess.run(
            [
                sys.executable,
                workflow_script.as_posix(),
                ml_instance.catalog.deriva_server.server,
                ml_instance.catalog_id,
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        new_workflow = result.stdout.strip()
        assert new_workflow == workflow_rid

    def test_workflow_creation_notebook(self, notebook_test):
        """Test creating a workflow from a Jupyter notebook."""
        ml_instance = notebook_test

        notebook_path = Path(__file__).parent / "workflow-test.ipynb"
        ml_instance.add_term(vc.asset_type, "Test Model", description="Model for our Test workflow")
        ml_instance.add_term(vc.workflow_type, "Test Workflow", description="A ML Workflow that uses Deriva ML API")

        workflow_table = ml_instance.pathBuilder().schemas[ml_instance.ml_schema].Workflow
        workflows = list(workflow_table.entities().fetch())
        assert len(workflows) == 0

        # Run the notebook
        env = os.environ.copy()
        env["SETUPTOOLS_USE_DISTUTILS"] = "local"
        result = subprocess.run(
            [
                "deriva-ml-run-notebook",
                notebook_path.as_posix(),
                "--host",
                ml_instance.catalog.deriva_server.server,
                "--catalog",
                ml_instance.catalog_id,
                "--kernel",
                "test_kernel",
                "--log-output",
                "--allow-dirty",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, f"Notebook run failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

        workflows = list(workflow_table.entities().fetch())
        assert len(workflows) == 1
        workflow_url = workflows[0]["URL"]
        assert workflow_url.endswith("workflow-test.ipynb")

    def test_workflow_api_creation(self, test_ml):
        """Test creating a workflow via the API."""
        ml = test_ml
        ml.add_term(vc.asset_type, "Test Model", description="Model for test")
        ml.add_term(vc.workflow_type, "Test Workflow", description="Test workflow type")

        # create_workflow returns a Workflow object (not yet in catalog)
        workflow = ml.create_workflow(
            name="API Test Workflow",
            workflow_type="Test Workflow",
            description="Created via API",
        )

        assert workflow is not None
        assert workflow.name == "API Test Workflow"
        assert workflow.workflow_type == ["Test Workflow"]
        assert workflow.description == "Created via API"

        # _add_workflow adds it to the catalog (or returns existing if same URL/checksum)
        workflow_rid = ml._add_workflow(workflow)
        assert workflow_rid is not None

        # Verify it's in the catalog
        workflows = list(ml.find_workflows())
        assert len(workflows) >= 1
        # Verify our workflow RID is in the catalog
        assert workflow_rid in [w.rid for w in workflows]

    def test_workflow_multiple_types(self, test_ml):
        """Test creating a workflow with multiple types."""
        ml = test_ml
        ml.add_term(vc.workflow_type, "Training", description="Training workflow")
        ml.add_term(vc.workflow_type, "Embedding", description="Embedding workflow")

        workflow = ml.create_workflow(
            name="Multi-Type Workflow",
            workflow_type=["Training", "Embedding"],
            description="A workflow with multiple types",
        )

        assert workflow.workflow_type == ["Training", "Embedding"]

        # Add to catalog and verify
        workflow_rid = ml._add_workflow(workflow)
        looked_up = ml.lookup_workflow(workflow_rid)
        assert sorted(looked_up.workflow_type) == ["Embedding", "Training"]

    def test_workflow_types_property(self, test_ml):
        """Test workflow_types property returns live list from catalog."""
        ml = test_ml
        ml.add_term(vc.workflow_type, "Training", description="Training workflow")
        ml.add_term(vc.workflow_type, "Embedding", description="Embedding workflow")

        workflow = ml.create_workflow(
            name="Types Property Test",
            workflow_type=["Training", "Embedding"],
        )
        workflow_rid = ml._add_workflow(workflow)
        looked_up = ml.lookup_workflow(workflow_rid)

        # workflow_types property should query association table
        types = looked_up.workflow_types
        assert sorted(types) == ["Embedding", "Training"]

    def test_workflow_add_remove_type(self, test_ml):
        """Test add_workflow_type and remove_workflow_type on catalog-bound workflow."""
        ml = test_ml
        ml.add_term(vc.workflow_type, "Training", description="Training workflow")
        ml.add_term(vc.workflow_type, "Embedding", description="Embedding workflow")
        ml.add_term(vc.workflow_type, "Inference", description="Inference workflow")

        workflow = ml.create_workflow(
            name="Add Remove Type Test",
            workflow_type="Training",
        )
        workflow_rid = ml._add_workflow(workflow)
        looked_up = ml.lookup_workflow(workflow_rid)

        assert looked_up.workflow_types == ["Training"]

        # Add a type
        looked_up.add_workflow_type("Embedding")
        assert sorted(looked_up.workflow_types) == ["Embedding", "Training"]

        # Add another
        looked_up.add_workflow_type("Inference")
        assert sorted(looked_up.workflow_types) == ["Embedding", "Inference", "Training"]

        # Remove a type
        looked_up.remove_workflow_type("Embedding")
        assert sorted(looked_up.workflow_types) == ["Inference", "Training"]

        # Idempotent add (already present)
        looked_up.add_workflow_type("Training")
        assert sorted(looked_up.workflow_types) == ["Inference", "Training"]

        # Idempotent remove (not present)
        looked_up.remove_workflow_type("Embedding")
        assert sorted(looked_up.workflow_types) == ["Inference", "Training"]

    def test_workflow_single_type_backward_compat(self, test_ml):
        """Test that single string workflow_type still works."""
        ml = test_ml
        ml.add_term(vc.workflow_type, "Training", description="Training workflow")

        workflow = ml.create_workflow(
            name="Single Type Test",
            workflow_type="Training",
        )
        # Pydantic validator normalizes to list
        assert workflow.workflow_type == ["Training"]

        workflow_rid = ml._add_workflow(workflow)
        looked_up = ml.lookup_workflow(workflow_rid)
        assert looked_up.workflow_types == ["Training"]
        assert looked_up.workflow_type == ["Training"]

    def test_find_executions_by_workflow_type(self, test_ml):
        """Test filtering executions by workflow type through association table."""
        ml = test_ml
        ml.add_term(vc.workflow_type, "Training", description="Training workflow")
        ml.add_term(vc.workflow_type, "Embedding", description="Embedding workflow")

        from deriva_ml.execution.execution import ExecutionConfiguration
        from deriva_ml.execution.workflow import Workflow

        # Create workflows with distinct URLs so add_workflow doesn't deduplicate
        wf_training = Workflow(
            name="Training WF",
            workflow_type="Training",
            url="https://example.com/training_wf.py",
            checksum="training_checksum_123",
        )
        wf_embedding = Workflow(
            name="Embedding WF",
            workflow_type="Embedding",
            url="https://example.com/embedding_wf.py",
            checksum="embedding_checksum_456",
        )

        config1 = ExecutionConfiguration(description="Train Run", workflow=wf_training)
        exec1 = ml.create_execution(config1)

        config2 = ExecutionConfiguration(description="Embed Run", workflow=wf_embedding)
        exec2 = ml.create_execution(config2)

        # Filter by workflow type
        training_execs = list(ml.find_executions(workflow_type="Training"))
        assert len(training_execs) >= 1
        assert any(e.execution_rid == exec1.execution_rid for e in training_execs)
        assert not any(e.execution_rid == exec2.execution_rid for e in training_execs)

        embedding_execs = list(ml.find_executions(workflow_type="Embedding"))
        assert len(embedding_execs) >= 1
        assert any(e.execution_rid == exec2.execution_rid for e in embedding_execs)


# =============================================================================
# TestExecutionLifecycle - Lifecycle and Status Tests
# =============================================================================


class TestExecutionLifecycle:
    """Tests for execution lifecycle management."""

    def test_execution_context_manager(self, basic_execution):
        """Test execution as a context manager.

        Post-S1a: `execute()` is a no-op returning self; __enter__ transitions
        the SQLite status Created → Running via state_machine.transition().
        __exit__ transitions Running → Stopped on clean exit. After S1a the
        catalog Status column and ExecutionStatus share a single title-case
        vocabulary (Created, Running, Stopped, Pending_Upload, Uploaded,
        Failed, Aborted) — SQLite remains authoritative; the catalog is a
        synced mirror.
        """
        from deriva_ml.execution.state_store import ExecutionStatus

        execution = basic_execution

        with execution.execute() as exe:
            assert exe.execution_rid is not None
            # New lifecycle: __enter__ transitions SQLite status to Running.
            assert exe.status is ExecutionStatus.Running

        # After context exit, the authoritative SQLite status is Stopped.
        assert execution.status is ExecutionStatus.Stopped

        # Upload finalizes the catalog-visible lifecycle; after upload we
        # expect the SQLite status to have advanced (pending_upload or
        # uploaded depending on how the current upload pipeline is wired
        # relative to this task). We don't assert a specific terminal
        # state here — later tasks (E5/G-series) will refine.
        execution.upload_execution_outputs()

    def test_execution_manual_start_stop(self, basic_execution):
        """``execution_start()`` / ``execution_stop()`` cycle status correctly.

        Imperative counterpart of the context-manager flow used by
        ``with ml.create_execution(...) as exe:``. Required by code paths
        that can't use a ``with`` block (e.g., the multirun parent execution
        managed by an atexit handler in ``runner._complete_parent_execution``).
        Each call must drive the documented state-machine transition:

            execution_start()           → Created  → Running
            execution_stop()            → Running  → Stopped
            upload_execution_outputs()  → Stopped  → Pending_Upload → Uploaded

        Regression guard: prior to the fix in commit
        "fix(execution): execution_start/stop transition status",
        ``execution_start()`` only wrote ``start_time`` and left status at
        ``Created``, so ``execution_stop()``'s subsequent
        ``update_status(Stopped)`` raised ``InvalidTransitionError`` for
        ``Created → Stopped``. The multirun parent path crashed in
        production before this test was strengthened.
        """
        execution = basic_execution
        ml = execution._ml_object

        execution.execution_start()
        assert get_execution_status(ml, execution.execution_rid) == "Running"

        execution.execution_stop()
        assert get_execution_status(ml, execution.execution_rid) == "Stopped"

        execution.upload_execution_outputs()
        assert get_execution_status(ml, execution.execution_rid) == "Uploaded"

    def test_notebook_lifecycle_auto_stops_running(self, workflow_terms, test_workflow):
        """upload_execution_outputs() auto-stops a still-Running execution.

        Regression test for the notebook code path. ``run_notebook()``
        returns ``(ml, execution, config)`` to the user with the execution
        in status ``Running`` (it calls ``execution_start()`` internally).
        The user's notebook cells run, doing work, and the last cell calls
        ``execution.upload_execution_outputs()`` directly — no
        ``execution_stop()`` call between them, since notebooks have no
        natural ``__exit__`` hook for the kernel-managed cell flow.

        Before this fix, ``upload_execution_outputs()`` only handled
        Stopped → Pending_Upload, and a Running-status execution either
        crashed inside the upload (state machine rejected its terminal
        transitions) or the exception handler tried Created/Running →
        Failed which is illegal from Created. Now ``upload_execution_outputs()``
        auto-calls ``execution_stop()`` if the execution is still Running.

        Tested flow (mirrors ``run_notebook`` + last-cell ``upload``):

            create_execution         status=Created
            execution_start()        status=Running
            (user work in cells)
            upload_execution_outputs status=Uploaded   # auto-stops first
        """
        ml = workflow_terms

        notebook_config = ExecutionConfiguration(
            description="Notebook execution (regression test)",
            workflow=test_workflow,
        )
        execution = ml.create_execution(notebook_config)
        assert get_execution_status(ml, execution.execution_rid) == "Created"

        # run_notebook() does this internally before returning to the user.
        execution.execution_start()
        assert get_execution_status(ml, execution.execution_rid) == "Running"

        # Notebook user code runs in cells here. We simulate "user did
        # something but did not call execution_stop()" — the typical
        # notebook pattern.

        # Final cell: upload directly. Must succeed without a separate
        # execution_stop() call because notebooks have no __exit__ hook.
        execution.upload_execution_outputs()
        assert get_execution_status(ml, execution.execution_rid) == "Uploaded"

    def test_multirun_parent_lifecycle(self, workflow_terms, test_workflow):
        """Multirun parent execution flows through the full state-machine cycle.

        Direct regression test for the runner's multirun parent path. The
        runner does not use a ``with`` context manager for the parent — it
        calls ``execution_start()`` after ``create_execution`` and
        ``execution_stop()`` from an ``atexit`` handler. Both must drive the
        same ``Created → Running → Stopped`` transitions the context
        manager does, otherwise the parent gets stuck at ``Created`` and
        ``execution_stop()`` raises ``InvalidTransitionError`` for the
        illegal ``Created → Stopped`` jump (observed in production with
        the lr_sweep multirun on a fresh catalog).

        Equivalent to ``test_execution_manual_start_stop`` but framed
        around the parent-execution use case so a future refactor that
        breaks one path leaves the other passing visibly named.
        """
        ml = workflow_terms

        # Parent executions in the runner have no datasets — they only
        # group children — so we mirror that here.
        parent_config = ExecutionConfiguration(
            description="Multirun parent execution (regression test)",
            workflow=test_workflow,
        )
        parent = ml.create_execution(parent_config)
        assert get_execution_status(ml, parent.execution_rid) == "Created"

        # Runner calls execution_start() imperatively after create_execution.
        parent.execution_start()
        assert get_execution_status(ml, parent.execution_rid) == "Running"

        # ... children would run here in production. Skip for the unit test.

        # Runner's atexit handler calls execution_stop() then
        # upload_execution_outputs(). Both must succeed.
        parent.execution_stop()
        assert get_execution_status(ml, parent.execution_rid) == "Stopped"

        parent.upload_execution_outputs()
        assert get_execution_status(ml, parent.execution_rid) == "Uploaded"

    def test_execution_status_updates(self, basic_execution):
        """Test updating execution status through the Phase 2 lifecycle.

        The legacy Status_Detail free-text column is no longer part of the
        update_status contract. The new update_status(target, *, error=None)
        validates against ALLOWED_TRANSITIONS; tests verifying an error
        string are covered by test_update_status.py (terminal states only).
        """
        from deriva_ml.execution.state_store import ExecutionStatus

        execution = basic_execution
        ml = execution._ml_object

        with execution.execute():
            # Inside execute() the status is Running — validated by the
            # state machine, no way to re-transition to Running here.
            record = ml._retrieve_rid(execution.execution_rid)
            assert record["Status"] == "Running"

        # After __exit__, status is Stopped.
        record = ml._retrieve_rid(execution.execution_rid)
        assert record["Status"] == "Stopped"

    def test_execution_metadata_files_uploaded(self, basic_execution):
        """Test that execution metadata files are uploaded."""
        execution = basic_execution
        ml = execution._ml_object

        with execution.execute():
            pass

        execution.upload_execution_outputs()

        metadata_files = get_execution_metadata_files(ml, execution.execution_rid)
        assert "configuration.json" in metadata_files
        assert "uv.lock" in metadata_files

    def test_execution_working_directory(self, basic_execution):
        """Test working directory property."""
        execution = basic_execution

        with execution.execute():
            working_dir = execution.working_dir
            assert working_dir.exists()
            assert working_dir.is_dir()


# =============================================================================
# TestExecutionAssets - Asset Upload/Download Tests
# =============================================================================


class TestExecutionAssets:
    """Tests for asset upload and download operations."""

    def test_asset_file_path_creation(self, basic_execution):
        """Test creating asset file paths."""
        with basic_execution.execute() as execution:
            asset_path = create_test_asset(execution)

            assert asset_path.exists()
            assert asset_path.asset_name == "Execution_Asset"
            assert "Model_File" in asset_path.asset_types

    def test_asset_upload(self, basic_execution):
        """Test uploading assets."""
        with basic_execution.execute() as execution:
            create_test_asset(execution, "model1.txt", "Model 1 content")

        uploaded = basic_execution.upload_execution_outputs()
        assert "deriva-ml/Execution_Asset" in uploaded
        assert len(uploaded["deriva-ml/Execution_Asset"]) == 1

    def test_asset_upload_multiple(self, basic_execution):
        """Test uploading multiple assets."""
        with basic_execution.execute() as execution:
            create_test_asset(execution, "model1.txt", "Model 1")
            create_test_asset(execution, "model2.txt", "Model 2")
            create_test_asset(execution, "model3.txt", "Model 3")

        uploaded = basic_execution.upload_execution_outputs()
        assert len(uploaded["deriva-ml/Execution_Asset"]) == 3

    def test_asset_download(self, basic_execution):
        """Test downloading assets."""
        ml = basic_execution._ml_object

        # First upload an asset
        with basic_execution.execute() as execution:
            create_test_asset(execution, "downloadable.txt", "Download me")

        uploaded = basic_execution.upload_execution_outputs()
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        # Now download it in a new execution
        config = ExecutionConfiguration(
            description="Download Test",
            workflow=basic_execution.configuration.workflow,
        )
        download_execution = ml.create_execution(config)

        with TemporaryDirectory() as tmpdir:
            downloaded = download_execution.download_asset(asset_rid, Path(tmpdir), update_catalog=False)
            assert downloaded.exists()
            assert downloaded.name == "downloadable.txt"
            with downloaded.open() as f:
                assert f.read() == "Download me"

    def test_asset_types_preserved(self, basic_execution):
        """Test that asset types are preserved through upload/download."""
        ml = basic_execution._ml_object

        with basic_execution.execute() as execution:
            asset_path = execution.asset_file_path(
                MLAsset.execution_asset,
                "TypedAsset/typed.txt",
                asset_types=ExecAssetType.model_file,
            )
            with asset_path.open("w") as f:
                f.write("typed content")

        uploaded = basic_execution.upload_execution_outputs()
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        # Download and check types
        config = ExecutionConfiguration(
            description="Type Check",
            workflow=basic_execution.configuration.workflow,
        )
        check_execution = ml.create_execution(config)

        with TemporaryDirectory() as tmpdir:
            downloaded = check_execution.download_asset(asset_rid, Path(tmpdir), update_catalog=False)
            assert "Model_File" in downloaded.asset_types

    def test_upload_progress_callback(self, basic_execution):
        """Test that upload progress callback is invoked during upload."""
        from deriva_ml import UploadProgress

        progress_updates = []

        def progress_callback(progress: UploadProgress) -> None:
            progress_updates.append(
                {
                    "file_name": progress.file_name,
                    "phase": progress.phase,
                    "message": progress.message,
                    "percent_complete": progress.percent_complete,
                }
            )

        with basic_execution.execute() as execution:
            # Create multiple assets to ensure we get progress updates
            create_test_asset(execution, "file1.txt", "Content 1")
            create_test_asset(execution, "file2.txt", "Content 2")

        # Upload with progress callback
        uploaded = basic_execution.upload_execution_outputs(progress_callback=progress_callback)

        # Verify callback was invoked
        assert len(progress_updates) > 0, "Progress callback should have been invoked"

        # Verify we got updates with expected fields
        for update in progress_updates:
            assert "phase" in update
            assert "message" in update

        # Verify assets were still uploaded correctly
        assert "deriva-ml/Execution_Asset" in uploaded
        assert len(uploaded["deriva-ml/Execution_Asset"]) == 2

    def test_list_asset_executions(self, basic_execution):
        """Test listing executions associated with an asset."""
        ml = basic_execution._ml_object

        # Create and upload an asset
        with basic_execution.execute() as execution:
            create_test_asset(execution, "traced_asset.txt", "Traceable content")

        uploaded = basic_execution.upload_execution_outputs()
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        # Test list_asset_executions - should return ExecutionRecord objects
        executions = ml.list_asset_executions(asset_rid)
        assert len(executions) == 1
        assert executions[0].execution_rid == basic_execution.execution_rid

        # Test with asset_role filter
        output_executions = ml.list_asset_executions(asset_rid, asset_role="Output")
        assert len(output_executions) == 1

        input_executions = ml.list_asset_executions(asset_rid, asset_role="Input")
        assert len(input_executions) == 0

        # Create a new execution that uses the asset as input
        config = ExecutionConfiguration(
            description="Input Test",
            workflow=basic_execution.configuration.workflow,
            assets=[asset_rid],
        )
        input_execution = ml.create_execution(config)

        # Now test again - should have both associations
        all_executions = ml.list_asset_executions(asset_rid)
        assert len(all_executions) == 2

        # Filter by role
        output_only = ml.list_asset_executions(asset_rid, asset_role="Output")
        assert len(output_only) == 1
        assert output_only[0].execution_rid == basic_execution.execution_rid

        input_only = ml.list_asset_executions(asset_rid, asset_role="Input")
        assert len(input_only) == 1
        assert input_only[0].execution_rid == input_execution.execution_rid

    def test_asset_file_path_with_relative_path(self, basic_execution):
        """Test that asset_file_path works correctly with relative paths.

        This verifies that relative paths are resolved to absolute paths,
        which is important when the current working directory changes
        (e.g., in Hydra workflows).
        """
        with TemporaryDirectory() as tmpdir:
            # Create a file with a relative path from tmpdir
            tmpdir = Path(tmpdir)
            subdir = tmpdir / "subdir"
            subdir.mkdir()
            test_file = subdir / "relative_test.txt"
            test_file.write_text("Test content for relative path")

            # Save original working directory
            original_cwd = os.getcwd()

            try:
                # Change to tmpdir so we can use a relative path
                os.chdir(tmpdir)

                # Use a relative path
                relative_path = Path("subdir/relative_test.txt")
                assert relative_path.exists(), "Relative path should exist from tmpdir"

                with basic_execution.execute() as execution:
                    # Register the file using a relative path
                    asset_path = execution.asset_file_path(
                        MLAsset.execution_asset,
                        relative_path,
                        asset_types=ExecAssetType.model_file,
                    )

                    # The asset_path should exist and be a symlink to the original file
                    assert asset_path.exists(), "Asset path should exist"

                    # Verify the content is accessible
                    content = asset_path.read_text()
                    assert content == "Test content for relative path"

                    # Change to a different directory and verify symlink still works
                    os.chdir(original_cwd)

                    # The symlink should still work because it points to an absolute path
                    assert asset_path.exists(), "Asset path should still exist after cwd change"
                    content_after_chdir = asset_path.read_text()
                    assert content_after_chdir == "Test content for relative path"
            finally:
                os.chdir(original_cwd)

    def test_asset_file_path_with_relative_path_copy(self, basic_execution):
        """Test that asset_file_path copy mode works with relative paths."""
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            test_file = tmpdir / "copy_test.txt"
            test_file.write_text("Content to copy")

            original_cwd = os.getcwd()

            try:
                os.chdir(tmpdir)

                relative_path = Path("copy_test.txt")

                with basic_execution.execute() as execution:
                    # Register with copy_file=True
                    asset_path = execution.asset_file_path(
                        MLAsset.execution_asset,
                        relative_path,
                        asset_types=ExecAssetType.model_file,
                        copy_file=True,
                    )

                    assert asset_path.exists()
                    # Should be a regular file, not a symlink
                    assert not asset_path.is_symlink()
                    assert asset_path.read_text() == "Content to copy"
            finally:
                os.chdir(original_cwd)


# =============================================================================
# TestAssetDescriptions - Asset Description Tests
# =============================================================================


class TestAssetDescriptions:
    """Tests for asset description functionality."""

    def test_get_metadata_description_direct(self):
        """Test direct filename matches for metadata descriptions."""
        assert Execution._get_metadata_description("configuration.json") is not None
        assert "DerivaML execution configuration" in Execution._get_metadata_description("configuration.json")

        assert Execution._get_metadata_description("config.yaml") is not None
        assert "Resolved Hydra configuration" in Execution._get_metadata_description("config.yaml")

        assert Execution._get_metadata_description("overrides.yaml") is not None
        assert "Hydra overrides" in Execution._get_metadata_description("overrides.yaml")

        assert Execution._get_metadata_description("hydra.yaml") is not None
        assert "Hydra runtime config" in Execution._get_metadata_description("hydra.yaml")

        assert Execution._get_metadata_description("uv.lock") is not None
        assert "Python dependency lockfile" in Execution._get_metadata_description("uv.lock")

    def test_get_metadata_description_hydra_renamed(self):
        """Test hydra renamed files resolve to the original description."""
        # Hydra renames files with a timestamp prefix: hydra-{timestamp}-{original_name}
        desc = Execution._get_metadata_description("hydra-14-30-00-config.yaml")
        assert desc is not None
        assert "Resolved Hydra configuration" in desc

        desc = Execution._get_metadata_description("hydra-09-15-42-overrides.yaml")
        assert desc is not None
        assert "Hydra overrides" in desc

        desc = Execution._get_metadata_description("hydra-23-59-59-hydra.yaml")
        assert desc is not None
        assert "Hydra runtime config" in desc

    def test_get_metadata_description_environment_snapshot(self):
        """Test environment snapshot files are recognized."""
        desc = Execution._get_metadata_description("environment_snapshot_20260401_143000.txt")
        assert desc is not None
        assert "Runtime environment snapshot" in desc

        # Different timestamp
        desc = Execution._get_metadata_description("environment_snapshot_20250101_000000.txt")
        assert desc is not None
        assert "Runtime environment snapshot" in desc

    def test_get_metadata_description_unknown(self):
        """Test that unknown files return None."""
        assert Execution._get_metadata_description("unknown_file.txt") is None
        assert Execution._get_metadata_description("random.json") is None
        assert Execution._get_metadata_description("model_weights.pt") is None

    def test_asset_file_path_with_description(self, basic_execution):
        """Test that description is stored in manifest when passed to asset_file_path."""
        with basic_execution.execute() as execution:
            path = execution.asset_file_path(
                MLAsset.execution_asset,
                "predictions.csv",
                asset_types=ExecAssetType.model_file,
                description="Model predictions on test set",
            )
            path.write_text("pred1,pred2")

            # Check manifest has the description
            manifest = execution._get_manifest()
            entry = manifest.assets["Execution_Asset/predictions.csv"]
            assert entry.description == "Model predictions on test set"


# =============================================================================
# TestAssetCaching - Asset Cache Tests
# =============================================================================


class TestAssetCaching:
    """Tests for per-asset checksum-based caching."""

    def test_cache_miss_then_hit(self, basic_execution):
        """Test that first download caches the asset and second uses the cache."""
        ml = basic_execution._ml_object

        # Upload an asset
        with basic_execution.execute() as execution:
            create_test_asset(execution, "weights.bin", "large model weights")

        uploaded = basic_execution.upload_execution_outputs()
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        # First download with use_cache=True (cache miss — downloads and caches)
        config1 = ExecutionConfiguration(
            description="Cache Miss Download",
            workflow=basic_execution.configuration.workflow,
        )
        exec1 = ml.create_execution(config1)

        with TemporaryDirectory() as tmpdir1:
            result1 = exec1.download_asset(asset_rid, Path(tmpdir1), update_catalog=False, use_cache=True)
            assert result1.exists()
            assert result1.is_symlink(), "After cache miss, file should be symlinked from cache"
            with result1.open() as f:
                assert f.read() == "large model weights"

        # Verify the cache directory was created
        cache_assets_dir = ml.cache_dir / "assets"
        assert cache_assets_dir.exists()
        cache_entries = list(cache_assets_dir.iterdir())
        assert len(cache_entries) == 1
        cache_entry = cache_entries[0]
        assert cache_entry.name.startswith(asset_rid)
        cached_file = cache_entry / "weights.bin"
        assert cached_file.exists()
        assert not cached_file.is_symlink(), "Cached file itself should not be a symlink"
        with cached_file.open() as f:
            assert f.read() == "large model weights"

        # Second download with use_cache=True (cache hit — symlinks without download)
        config2 = ExecutionConfiguration(
            description="Cache Hit Download",
            workflow=basic_execution.configuration.workflow,
        )
        exec2 = ml.create_execution(config2)

        with TemporaryDirectory() as tmpdir2:
            result2 = exec2.download_asset(asset_rid, Path(tmpdir2), update_catalog=False, use_cache=True)
            assert result2.exists()
            assert result2.is_symlink(), "Cache hit should produce a symlink"
            # Symlink should point to the same cached file
            assert result2.resolve() == cached_file.resolve()
            with result2.open() as f:
                assert f.read() == "large model weights"

    def test_no_cache_by_default(self, basic_execution):
        """Test that assets are not cached when use_cache=False."""
        ml = basic_execution._ml_object

        with basic_execution.execute() as execution:
            create_test_asset(execution, "ephemeral.txt", "not cached")

        uploaded = basic_execution.upload_execution_outputs()
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        config = ExecutionConfiguration(
            description="No Cache Download",
            workflow=basic_execution.configuration.workflow,
        )
        dl_execution = ml.create_execution(config)

        with TemporaryDirectory() as tmpdir:
            result = dl_execution.download_asset(asset_rid, Path(tmpdir), update_catalog=False, use_cache=False)
            assert result.exists()
            assert not result.is_symlink(), "Without cache, file should be a regular file"
            with result.open() as f:
                assert f.read() == "not cached"

        # Verify no cache directory was created for assets
        cache_assets_dir = ml.cache_dir / "assets"
        if cache_assets_dir.exists():
            assert len(list(cache_assets_dir.iterdir())) == 0

    def test_cache_via_execution_configuration(self, basic_execution):
        """Test that AssetSpec(cache=True) in ExecutionConfiguration triggers caching."""
        from deriva_ml.asset.aux_classes import AssetSpec

        ml = basic_execution._ml_object

        # Upload an asset
        with basic_execution.execute() as execution:
            create_test_asset(execution, "config_cached.txt", "cached via config")

        uploaded = basic_execution.upload_execution_outputs()
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        # Create execution with AssetSpec(cache=True) in the config
        config = ExecutionConfiguration(
            description="Config Cache Test",
            workflow=basic_execution.configuration.workflow,
            assets=[AssetSpec(rid=asset_rid, cache=True)],
        )
        cache_execution = ml.create_execution(config)

        # The asset should have been downloaded and cached during initialization
        assert "Execution_Asset" in cache_execution.asset_paths
        asset_path = cache_execution.asset_paths["Execution_Asset"][0]
        assert asset_path.exists()
        assert asset_path.is_symlink(), "Asset from config with cache=True should be symlinked"
        with asset_path.open() as f:
            assert f.read() == "cached via config"

        # Verify cache was populated
        cache_assets_dir = ml.cache_dir / "assets"
        assert cache_assets_dir.exists()
        cache_entries = list(cache_assets_dir.iterdir())
        assert len(cache_entries) == 1

    def test_uncached_asset_in_config(self, basic_execution):
        """Test that plain RID strings in ExecutionConfiguration do not cache."""
        ml = basic_execution._ml_object

        with basic_execution.execute() as execution:
            create_test_asset(execution, "plain_rid.txt", "not cached")

        uploaded = basic_execution.upload_execution_outputs()
        asset_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid

        # Create execution with plain RID string (no caching)
        config = ExecutionConfiguration(
            description="Plain RID Test",
            workflow=basic_execution.configuration.workflow,
            assets=[asset_rid],
        )
        plain_execution = ml.create_execution(config)

        assert "Execution_Asset" in plain_execution.asset_paths
        asset_path = plain_execution.asset_paths["Execution_Asset"][0]
        assert asset_path.exists()
        assert not asset_path.is_symlink(), "Plain RID should not produce a symlink"

        # Verify no asset cache was created
        cache_assets_dir = ml.cache_dir / "assets"
        if cache_assets_dir.exists():
            assert len(list(cache_assets_dir.iterdir())) == 0

    def test_mixed_cached_and_uncached(self, basic_execution):
        """Test mixed AssetSpec(cache=True) and plain RIDs in same execution."""
        from deriva_ml.asset.aux_classes import AssetSpec

        ml = basic_execution._ml_object

        # Upload two assets
        with basic_execution.execute() as execution:
            create_test_asset(execution, "cached.txt", "I am cached")
            create_test_asset(execution, "uncached.txt", "I am not cached")

        uploaded = basic_execution.upload_execution_outputs()
        cached_rid = uploaded["deriva-ml/Execution_Asset"][0].asset_rid
        uncached_rid = uploaded["deriva-ml/Execution_Asset"][1].asset_rid

        # Create execution with mixed assets
        config = ExecutionConfiguration(
            description="Mixed Cache Test",
            workflow=basic_execution.configuration.workflow,
            assets=[
                AssetSpec(rid=cached_rid, cache=True),
                uncached_rid,  # plain string
            ],
        )
        mixed_execution = ml.create_execution(config)

        paths = mixed_execution.asset_paths["Execution_Asset"]
        assert len(paths) == 2

        # Find which path corresponds to which RID
        cached_path = next(p for p in paths if p.asset_rid == cached_rid)
        uncached_path = next(p for p in paths if p.asset_rid == uncached_rid)

        assert cached_path.is_symlink(), "Cached asset should be a symlink"
        assert not uncached_path.is_symlink(), "Uncached asset should be a regular file"

        # Verify only one cache entry was created
        cache_assets_dir = ml.cache_dir / "assets"
        assert cache_assets_dir.exists()
        assert len(list(cache_assets_dir.iterdir())) == 1


# =============================================================================
# TestExecutionDatasets - Dataset Operations Tests
# =============================================================================


class TestExecutionDatasets:
    """Tests for dataset operations within executions."""

    def test_create_dataset_in_execution(self, basic_execution):
        """Test creating a dataset within an execution."""
        with basic_execution.execute() as execution:
            # Use "File" which is a standard dataset type
            dataset = execution.create_dataset(
                dataset_types=["File"],
                description="Dataset created in execution",
            )
            assert dataset is not None
            assert dataset.dataset_rid is not None
            # Note: execution_rid is not returned on the dataset object;
            # the link is stored in Dataset_Execution table
            ml = execution._ml_object
            pb = ml.pathBuilder().schemas[ml.ml_schema]
            ds_exec = pb.Dataset_Execution
            query = ds_exec.filter(ds_exec.Dataset == dataset.dataset_rid)
            links = list(query.entities().fetch())
            assert len(links) == 1
            assert links[0]["Execution"] == execution.execution_rid

    def test_download_dataset_in_execution(self, dataset_test, tmp_path):
        """Test downloading a dataset within an execution."""
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        ml.add_term(vc.asset_type, "Test Model", description="Model for test")
        ml.add_term(vc.workflow_type, "Test Workflow", description="Test workflow")

        dataset_rid = dataset_test.dataset_description.dataset.dataset_rid
        workflow = ml.create_workflow(
            name="Dataset Download Test",
            workflow_type="Test Workflow",
            description="Test dataset download",
        )

        config = ExecutionConfiguration(
            datasets=[
                DatasetSpec(
                    rid=dataset_rid,
                    version=dataset_test.dataset_description.dataset.current_version,
                ),
            ],
            description="Download Test",
            workflow=workflow,
        )

        execution = ml.create_execution(config)
        with execution.execute() as exe:
            assert len(exe.datasets) == 1
            # DatasetCollection supports both RID-keyed lookup and iteration
            assert exe.datasets[dataset_rid].dataset_rid == dataset_rid

    def test_execution_with_multiple_datasets(self, dataset_test, tmp_path):
        """Test execution with multiple dataset specifications."""
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        ml.add_term(vc.asset_type, "Test Model", description="Model")
        ml.add_term(vc.workflow_type, "Test Workflow", description="Workflow")

        # Get all available datasets
        all_datasets = list(ml.find_datasets())

        if len(all_datasets) >= 2:
            workflow = ml.create_workflow(
                name="Multi-Dataset Test",
                workflow_type="Test Workflow",
                description="Test multiple datasets",
            )

            config = ExecutionConfiguration(
                datasets=[DatasetSpec(rid=ds.dataset_rid, version=ds.current_version) for ds in all_datasets[:2]],
                description="Multi-Dataset Execution",
                workflow=workflow,
            )

            execution = ml.create_execution(config)
            with execution.execute() as exe:
                assert len(exe.datasets) == 2


# =============================================================================
# TestExecutionFeatures - Feature Management Tests
# =============================================================================


class TestExecutionFeatures:
    """Tests for feature record management within executions."""

    def test_add_features(self, feature_setup, test_workflow):
        """Test adding feature records to an execution."""
        ml = feature_setup

        # Get feature record class - dynamically generated with Subject and Score fields
        # Feature_Name has a default, so we only need Subject and Score
        SubjectTestScoreFeature = ml.feature_record_class("Subject", "TestScore")

        # Create subjects to attach features to
        pb = ml.pathBuilder()
        subjects = list(pb.schemas[ml.default_schema].Subject.entities().fetch())

        if not subjects:
            pytest.skip("No subjects in catalog for feature testing")

        config = ExecutionConfiguration(
            description="Feature Test",
            workflow=test_workflow,
        )
        execution = ml.create_execution(config)

        with execution.execute() as exe:
            # The feature class expects: Subject (target table RID) and Score (metadata)
            # Feature_Name has a default value set by the feature definition
            # Note: SubjectTestScoreFeature is dynamically created by pydantic
            features = [
                SubjectTestScoreFeature(Subject=s["RID"], Score=(i + 1) * 10)  # type: ignore[call-arg]
                for i, s in enumerate(subjects[:3])
            ]
            exe.add_features(features)

        execution.upload_execution_outputs()

        # Verify features were uploaded. Phase 2 lifecycle:
        # upload_execution_outputs advances Stopped → Pending_Upload → Uploaded.
        status = get_execution_status(ml, execution.execution_rid)
        assert status == "Uploaded"


# =============================================================================
# TestExecutionDryRun - Dry Run Mode Tests
# =============================================================================


class TestExecutionDryRun:
    """Tests for dry run mode (no catalog writes)."""

    def test_dry_run_no_catalog_write(self, workflow_terms, test_workflow):
        """Test that dry run mode doesn't write to catalog."""
        ml = workflow_terms

        config = ExecutionConfiguration(
            description="Dry Run Test",
            workflow=test_workflow,
        )

        # Get execution count before
        pb = ml.pathBuilder().schemas[ml.ml_schema]
        executions_before = len(list(pb.Execution.entities().fetch()))

        # Create dry run execution
        execution = ml.create_execution(config, dry_run=True)

        with execution.execute() as exe:
            create_test_asset(exe, "dry_run.txt", "Should not upload")

        # In dry run, upload should not create catalog entries
        execution.upload_execution_outputs()

        # Execution count should be the same (dry run uses fake RID)
        executions_after = len(list(pb.Execution.entities().fetch()))
        # Verify execution count didn't increase (dry run doesn't create records)
        assert executions_after == executions_before


# =============================================================================
# TestExecutionErrors - Error Handling Tests
# =============================================================================


class TestExecutionErrors:
    """Tests for error handling in executions."""

    def test_invalid_asset_rid_download(self, basic_execution):
        """Test downloading with an invalid asset RID."""
        with basic_execution.execute() as execution:
            with TemporaryDirectory() as tmpdir:
                with pytest.raises(Exception):  # Could be DerivaMLException or HTTP error
                    execution.download_asset("invalid-rid", Path(tmpdir))

    def test_execution_without_workflow(self, test_ml):
        """Test that execution requires a workflow."""
        ml = test_ml
        ml.add_term(vc.workflow_type, "Test Workflow", description="Test")

        config = ExecutionConfiguration(description="No Workflow")

        # Should raise or handle gracefully
        with pytest.raises(Exception):
            ml.create_execution(config)

    def test_table_path_invalid_table(self, basic_execution):
        """Test table_path with invalid table name."""
        with basic_execution.execute() as execution:
            with pytest.raises(DerivaMLException):
                execution.table_path("NonExistentTable")

    def test_upload_assets_invalid_directory(self, basic_execution):
        """Test uploading from invalid directory."""
        with basic_execution.execute() as execution:
            with pytest.raises(DerivaMLException):
                execution.upload_assets("/nonexistent/path")


# =============================================================================
# TestExecutionIntegration - Integration Tests
# =============================================================================


class TestExecutionIntegration:
    """Integration tests combining multiple execution features."""

    def test_full_execution_workflow(self, dataset_test, tmp_path):
        """Test a complete execution workflow with datasets and assets."""
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        # Setup
        ml.add_term(vc.asset_type, "Test Model", description="Model")
        ml.add_term(vc.workflow_type, "Test Workflow", description="Workflow")

        dataset_rid = dataset_test.dataset_description.dataset.dataset_rid

        workflow = ml.create_workflow(
            name="Integration Test Workflow",
            workflow_type="Test Workflow",
            description="Full integration test",
        )

        # Create execution with dataset
        config = ExecutionConfiguration(
            datasets=[
                DatasetSpec(
                    rid=dataset_rid,
                    version=dataset_test.dataset_description.dataset.current_version,
                ),
            ],
            description="Integration Test Execution",
            workflow=workflow,
        )

        execution = ml.create_execution(config)

        with execution.execute() as exe:
            # Verify dataset is available
            assert len(exe.datasets) == 1

            # Create output asset
            create_test_asset(exe, "integration_output.txt", "Integration test output")

            # Create a new dataset
            new_dataset = exe.create_dataset(
                dataset_types=["File"],
                description="Output dataset from integration test",
            )
            assert new_dataset is not None

        # Upload and verify completion
        uploaded = execution.upload_execution_outputs()
        assert "deriva-ml/Execution_Asset" in uploaded

        # Phase 2 lifecycle: upload_execution_outputs → Uploaded (legacy Completed).
        status = get_execution_status(ml, execution.execution_rid)
        assert status == "Uploaded"


# =============================================================================
# TestWorkflowDocker - Docker Environment Tests
# =============================================================================


class TestWorkflowDocker:
    """Tests for workflow creation in Docker environments (no git repo)."""

    def test_workflow_docker_environment(self, monkeypatch):
        """Test workflow creation with Docker environment variables."""
        from deriva_ml.execution.workflow import Workflow

        # Set Docker environment variables
        monkeypatch.setenv("DERIVA_MCP_IN_DOCKER", "true")
        monkeypatch.setenv("DERIVA_MCP_VERSION", "1.2.3")
        monkeypatch.setenv("DERIVA_MCP_GIT_COMMIT", "abc123def")
        monkeypatch.setenv("DERIVA_MCP_IMAGE_NAME", "ghcr.io/test/image")
        # Clear image digest to test git commit fallback
        monkeypatch.delenv("DERIVA_MCP_IMAGE_DIGEST", raising=False)

        # Create workflow - should not fail even without git
        workflow = Workflow(
            name="Docker Test Workflow",
            workflow_type="Test",
            description="Test workflow in Docker",
        )

        assert workflow.version == "1.2.3"
        assert workflow.checksum == "abc123def"
        # Without image digest, URL falls back to source repo with git commit
        assert "abc123def" in workflow.url
        assert "deriva-ml-mcp" in workflow.url

    def test_workflow_docker_with_image_digest(self, monkeypatch):
        """Test workflow with Docker image digest as checksum."""
        from deriva_ml.execution.workflow import Workflow

        # Set Docker environment with image digest
        monkeypatch.setenv("DERIVA_MCP_IN_DOCKER", "true")
        monkeypatch.setenv("DERIVA_MCP_VERSION", "2.0.0")
        monkeypatch.setenv("DERIVA_MCP_GIT_COMMIT", "abc123")
        monkeypatch.setenv("DERIVA_MCP_IMAGE_DIGEST", "sha256:deadbeef123456")
        monkeypatch.setenv("DERIVA_MCP_IMAGE_NAME", "ghcr.io/org/repo")

        workflow = Workflow(
            name="Digest Test",
            workflow_type="Test",
        )

        # Image digest should be used as checksum (takes precedence over git commit)
        assert workflow.checksum == "sha256:deadbeef123456"
        # URL should use digest format
        assert workflow.url == "ghcr.io/org/repo@sha256:deadbeef123456"

    def test_workflow_docker_minimal_env(self, monkeypatch):
        """Test workflow with minimal Docker environment (only IN_DOCKER set)."""
        from deriva_ml.execution.workflow import Workflow

        # Only set the Docker flag, no other env vars
        monkeypatch.setenv("DERIVA_MCP_IN_DOCKER", "true")
        # Clear any existing env vars
        monkeypatch.delenv("DERIVA_MCP_VERSION", raising=False)
        monkeypatch.delenv("DERIVA_MCP_GIT_COMMIT", raising=False)
        monkeypatch.delenv("DERIVA_MCP_IMAGE_DIGEST", raising=False)

        workflow = Workflow(
            name="Minimal Docker Test",
            workflow_type="Test",
        )

        # Should still create workflow without failing
        assert workflow.name == "Minimal Docker Test"
        assert workflow.version == ""
        assert workflow.checksum == ""
        # URL should fall back to default source URL
        assert "deriva-ml-mcp" in workflow.url

    def test_workflow_docker_explicit_values_preserved(self, monkeypatch):
        """Test that explicitly provided values are preserved in Docker mode."""
        from deriva_ml.execution.workflow import Workflow

        monkeypatch.setenv("DERIVA_MCP_IN_DOCKER", "true")
        monkeypatch.setenv("DERIVA_MCP_VERSION", "env-version")
        monkeypatch.setenv("DERIVA_MCP_GIT_COMMIT", "env-commit")

        # Provide explicit values
        workflow = Workflow(
            name="Explicit Values Test",
            workflow_type="Test",
            version="explicit-version",
            url="https://example.com/workflow",
            checksum="explicit-checksum",
        )

        # Explicit values should be preserved, not overwritten by env vars
        assert workflow.version == "explicit-version"
        assert workflow.url == "https://example.com/workflow"
        assert workflow.checksum == "explicit-checksum"


# =============================================================================
# TestExecutionNesting - Execution Nesting Tests
# =============================================================================


class TestExecutionNesting:
    """Tests for execution nesting (parent-child relationships)."""

    def test_add_nested_execution(self, workflow_terms, test_workflow):
        """Test adding a nested execution to a parent."""
        ml = workflow_terms

        # Create parent execution
        parent_config = ExecutionConfiguration(
            description="Parent Execution (Sweep)",
            workflow=test_workflow,
        )
        parent_exec = ml.create_execution(parent_config)

        # Create child execution
        child_config = ExecutionConfiguration(
            description="Child Execution (Run 1)",
            workflow=test_workflow,
        )
        child_exec = ml.create_execution(child_config)

        # Add child to parent
        parent_exec.add_nested_execution(child_exec, sequence=0)

        # Verify the relationship via the association table
        pb = ml.pathBuilder().schemas[ml.ml_schema]
        exec_exec = pb.Execution_Execution
        query = exec_exec.filter(
            (exec_exec.Execution == parent_exec.execution_rid)
            & (exec_exec.Nested_Execution == child_exec.execution_rid)
        )
        records = list(query.entities().fetch())

        assert len(records) == 1
        assert records[0]["Sequence"] == 0

    def test_add_nested_execution_by_rid(self, workflow_terms, test_workflow):
        """Test adding a nested execution using RID instead of object."""
        ml = workflow_terms

        parent_config = ExecutionConfiguration(
            description="Parent",
            workflow=test_workflow,
        )
        parent_exec = ml.create_execution(parent_config)

        child_config = ExecutionConfiguration(
            description="Child",
            workflow=test_workflow,
        )
        child_exec = ml.create_execution(child_config)

        # Add using RID string
        parent_exec.add_nested_execution(child_exec.execution_rid, sequence=1)

        # Verify
        pb = ml.pathBuilder().schemas[ml.ml_schema]
        exec_exec = pb.Execution_Execution
        records = list(exec_exec.filter(exec_exec.Execution == parent_exec.execution_rid).entities().fetch())

        assert len(records) == 1
        assert records[0]["Nested_Execution"] == child_exec.execution_rid

    def test_add_multiple_nested_executions(self, workflow_terms, test_workflow):
        """Test adding multiple nested executions with sequence ordering."""
        ml = workflow_terms

        parent_config = ExecutionConfiguration(
            description="Parent Sweep",
            workflow=test_workflow,
        )
        parent_exec = ml.create_execution(parent_config)

        # Create multiple children
        children = []
        for i in range(3):
            child_config = ExecutionConfiguration(
                description=f"Child Run {i}",
                workflow=test_workflow,
            )
            child_exec = ml.create_execution(child_config)
            children.append(child_exec)
            parent_exec.add_nested_execution(child_exec, sequence=i)

        # Verify all children are added
        pb = ml.pathBuilder().schemas[ml.ml_schema]
        exec_exec = pb.Execution_Execution
        records = list(exec_exec.filter(exec_exec.Execution == parent_exec.execution_rid).entities().fetch())

        assert len(records) == 3
        # Verify sequences
        sequences = sorted([r["Sequence"] for r in records])
        assert sequences == [0, 1, 2]

    def test_list_execution_children(self, workflow_terms, test_workflow):
        """Test listing nested executions."""
        ml = workflow_terms

        parent_config = ExecutionConfiguration(
            description="Parent",
            workflow=test_workflow,
        )
        parent_exec = ml.create_execution(parent_config)

        # Create and add children
        child_rids = []
        for i in range(2):
            child_config = ExecutionConfiguration(
                description=f"Child {i}",
                workflow=test_workflow,
            )
            child_exec = ml.create_execution(child_config)
            child_rids.append(child_exec.execution_rid)
            parent_exec.add_nested_execution(child_exec, sequence=i)

        # List children (hierarchy queries live on ExecutionRecord per R2.1)
        children = list(parent_exec.execution_record.list_execution_children())

        assert len(children) == 2
        # Verify they are returned in sequence order
        assert children[0].execution_rid == child_rids[0]
        assert children[1].execution_rid == child_rids[1]

    def test_list_execution_parents(self, workflow_terms, test_workflow):
        """Test listing parent executions."""
        ml = workflow_terms

        parent_config = ExecutionConfiguration(
            description="Parent",
            workflow=test_workflow,
        )
        parent_exec = ml.create_execution(parent_config)

        child_config = ExecutionConfiguration(
            description="Child",
            workflow=test_workflow,
        )
        child_exec = ml.create_execution(child_config)

        parent_exec.add_nested_execution(child_exec, sequence=0)

        # List parents from child's perspective (hierarchy on ExecutionRecord)
        parents = list(child_exec.execution_record.list_execution_parents())

        assert len(parents) == 1
        assert parents[0].execution_rid == parent_exec.execution_rid

    def test_list_execution_children_recurse(self, workflow_terms, test_workflow):
        """Test recursively listing nested executions."""
        ml = workflow_terms

        # Create a hierarchy: grandparent -> parent -> child
        grandparent_config = ExecutionConfiguration(
            description="Grandparent",
            workflow=test_workflow,
        )
        grandparent_exec = ml.create_execution(grandparent_config)

        parent_config = ExecutionConfiguration(
            description="Parent",
            workflow=test_workflow,
        )
        parent_exec = ml.create_execution(parent_config)

        child_config = ExecutionConfiguration(
            description="Child",
            workflow=test_workflow,
        )
        child_exec = ml.create_execution(child_config)

        # Build hierarchy
        grandparent_exec.add_nested_execution(parent_exec, sequence=0)
        parent_exec.add_nested_execution(child_exec, sequence=0)

        # Non-recursive should only return direct children
        direct_children = list(
            grandparent_exec.execution_record.list_execution_children(recurse=False)
        )
        assert len(direct_children) == 1
        assert direct_children[0].execution_rid == parent_exec.execution_rid

        # Recursive should return all descendants
        all_descendants = list(
            grandparent_exec.execution_record.list_execution_children(recurse=True)
        )
        assert len(all_descendants) == 2
        descendant_rids = [d.execution_rid for d in all_descendants]
        assert parent_exec.execution_rid in descendant_rids
        assert child_exec.execution_rid in descendant_rids

    def test_list_execution_parents_recurse(self, workflow_terms, test_workflow):
        """Test recursively listing parent executions."""
        ml = workflow_terms

        # Create a hierarchy: grandparent -> parent -> child
        grandparent_config = ExecutionConfiguration(
            description="Grandparent",
            workflow=test_workflow,
        )
        grandparent_exec = ml.create_execution(grandparent_config)

        parent_config = ExecutionConfiguration(
            description="Parent",
            workflow=test_workflow,
        )
        parent_exec = ml.create_execution(parent_config)

        child_config = ExecutionConfiguration(
            description="Child",
            workflow=test_workflow,
        )
        child_exec = ml.create_execution(child_config)

        # Build hierarchy
        grandparent_exec.add_nested_execution(parent_exec, sequence=0)
        parent_exec.add_nested_execution(child_exec, sequence=0)

        # Non-recursive should only return direct parent
        direct_parents = list(
            child_exec.execution_record.list_execution_parents(recurse=False)
        )
        assert len(direct_parents) == 1
        assert direct_parents[0].execution_rid == parent_exec.execution_rid

        # Recursive should return all ancestors
        all_ancestors = list(
            child_exec.execution_record.list_execution_parents(recurse=True)
        )
        assert len(all_ancestors) == 2
        ancestor_rids = [a.execution_rid for a in all_ancestors]
        assert parent_exec.execution_rid in ancestor_rids
        assert grandparent_exec.execution_rid in ancestor_rids

    def test_is_nested(self, workflow_terms, test_workflow):
        """Test is_nested() method."""
        ml = workflow_terms

        parent_config = ExecutionConfiguration(
            description="Parent",
            workflow=test_workflow,
        )
        parent_exec = ml.create_execution(parent_config)

        child_config = ExecutionConfiguration(
            description="Child",
            workflow=test_workflow,
        )
        child_exec = ml.create_execution(child_config)

        # Before nesting
        assert parent_exec.is_nested() is False
        assert child_exec.is_nested() is False

        # After nesting
        parent_exec.add_nested_execution(child_exec, sequence=0)
        assert parent_exec.is_nested() is False  # Parent is not nested
        assert child_exec.is_nested() is True  # Child is nested

    def test_is_parent(self, workflow_terms, test_workflow):
        """Test is_parent() method."""
        ml = workflow_terms

        parent_config = ExecutionConfiguration(
            description="Parent",
            workflow=test_workflow,
        )
        parent_exec = ml.create_execution(parent_config)

        child_config = ExecutionConfiguration(
            description="Child",
            workflow=test_workflow,
        )
        child_exec = ml.create_execution(child_config)

        # Before nesting
        assert parent_exec.is_parent() is False
        assert child_exec.is_parent() is False

        # After nesting
        parent_exec.add_nested_execution(child_exec, sequence=0)
        assert parent_exec.is_parent() is True  # Parent has children
        assert child_exec.is_parent() is False  # Child has no children

    def test_lookup_execution(self, workflow_terms, test_workflow):
        """Test lookup_execution method for lightweight retrieval."""
        ml = workflow_terms

        config = ExecutionConfiguration(
            description="Lookup Test",
            workflow=test_workflow,
        )
        original_exec = ml.create_execution(config)
        original_rid = original_exec.execution_rid

        # Lookup the execution - returns ExecutionRecord
        looked_up = ml.lookup_execution(original_rid)

        assert looked_up.execution_rid == original_rid
        assert looked_up.description == "Lookup Test"

    def test_nested_execution_null_sequence(self, workflow_terms, test_workflow):
        """Test adding nested executions without sequence (parallel)."""
        ml = workflow_terms

        parent_config = ExecutionConfiguration(
            description="Parallel Parent",
            workflow=test_workflow,
        )
        parent_exec = ml.create_execution(parent_config)

        # Add children without sequence (parallel execution)
        for i in range(3):
            child_config = ExecutionConfiguration(
                description=f"Parallel Child {i}",
                workflow=test_workflow,
            )
            child_exec = ml.create_execution(child_config)
            parent_exec.add_nested_execution(child_exec, sequence=None)

        # Verify all children are added with null sequence
        pb = ml.pathBuilder().schemas[ml.ml_schema]
        exec_exec = pb.Execution_Execution
        records = list(exec_exec.filter(exec_exec.Execution == parent_exec.execution_rid).entities().fetch())

        assert len(records) == 3
        # All should have null sequence
        for record in records:
            assert record["Sequence"] is None

    def test_dry_run_add_nested_execution(self, workflow_terms, test_workflow):
        """Test that dry run mode doesn't write nesting records."""
        ml = workflow_terms

        parent_config = ExecutionConfiguration(
            description="Dry Run Parent",
            workflow=test_workflow,
        )
        # Create parent in dry run mode
        parent_exec = ml.create_execution(parent_config, dry_run=True)

        child_config = ExecutionConfiguration(
            description="Dry Run Child",
            workflow=test_workflow,
        )
        child_exec = ml.create_execution(child_config, dry_run=True)

        # This should not write to catalog
        parent_exec.add_nested_execution(child_exec, sequence=0)

        # Verify nothing was written (dry run uses fake RIDs)
        pb = ml.pathBuilder().schemas[ml.ml_schema]
        exec_exec = pb.Execution_Execution
        # Both executions use DRY_RUN_RID, so query should return nothing
        records = list(exec_exec.entities().fetch())
        # Filter for our dry run RIDs - they shouldn't exist
        dry_run_records = [r for r in records if r["Execution"] == parent_exec.execution_rid]
        assert len(dry_run_records) == 0
