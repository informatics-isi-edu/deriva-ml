"""Tests for the execution module.

This module provides comprehensive tests for DerivaML execution functionality:

Test Classes:
    TestWorkflow: Workflow creation and management
    TestExecutionLifecycle: Context manager, start/stop, status tracking
    TestExecutionAssets: Asset upload/download operations
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
from deriva_ml.core.definitions import BuiltinTypes, ColumnDefinition, Status
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.execution.execution import Execution, ExecutionConfiguration


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def workflow_terms(test_ml):
    """Add required vocabulary terms for workflow testing."""
    test_ml.add_term(vc.asset_type, "Test Model", description="Model for our Test workflow")
    test_ml.add_term(vc.workflow_type, "Test Workflow", description="A ML Workflow that uses Deriva ML API")
    return test_ml


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
    ml.create_feature("Subject", "TestScore", metadata=[
        ColumnDefinition(name="Score", type=BuiltinTypes.int2, nullok=False),
    ])

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
    return ml.retrieve_rid(execution_rid)["Status"]


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

        # Verify lookup works
        looked_up_rid = ml_instance.lookup_workflow(workflow_url)
        assert looked_up_rid == workflow_rid

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
            ],
            capture_output=True,
            text=True,
            env=env,
        )

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
        assert workflow.workflow_type == "Test Workflow"
        assert workflow.description == "Created via API"

        # add_workflow adds it to the catalog (or returns existing if same URL/checksum)
        workflow_rid = ml.add_workflow(workflow)
        assert workflow_rid is not None

        # Verify it's in the catalog
        workflows = list(ml.list_workflows())
        assert len(workflows) >= 1
        # Verify our workflow RID is in the catalog
        assert workflow_rid in [w.rid for w in workflows]


# =============================================================================
# TestExecutionLifecycle - Lifecycle and Status Tests
# =============================================================================


class TestExecutionLifecycle:
    """Tests for execution lifecycle management."""

    def test_execution_context_manager(self, basic_execution):
        """Test execution as a context manager."""
        execution = basic_execution
        ml = execution._ml_object

        with execution.execute() as exe:
            assert exe.execution_rid is not None
            # Status is Initializing when execute() is called
            status = get_execution_status(ml, exe.execution_rid)
            assert status == "Initializing"

            # Update to Running
            exe.update_status(Status.running, "Running tests")
            status = get_execution_status(ml, exe.execution_rid)
            assert status == "Running"

        # After context exit, upload and check completion
        execution.upload_execution_outputs()
        status = get_execution_status(ml, execution.execution_rid)
        assert status == "Completed"

    def test_execution_manual_start_stop(self, basic_execution):
        """Test manual execution start and stop."""
        execution = basic_execution
        ml = execution._ml_object

        execution.execution_start()
        # execution_start sets status to Initializing
        assert get_execution_status(ml, execution.execution_rid) == "Initializing"

        # Update to Running
        execution.update_status(Status.running, "Running")
        assert get_execution_status(ml, execution.execution_rid) == "Running"

        execution.execution_stop()
        execution.upload_execution_outputs()
        assert get_execution_status(ml, execution.execution_rid) == "Completed"

    def test_execution_status_updates(self, basic_execution):
        """Test updating execution status."""
        execution = basic_execution
        ml = execution._ml_object

        with execution.execute():
            execution.update_status(Status.running, "Processing step 1")
            record = ml.retrieve_rid(execution.execution_rid)
            assert record["Status"] == "Running"
            assert record["Status_Detail"] == "Processing step 1"

            execution.update_status(Status.running, "Processing step 2")
            record = ml.retrieve_rid(execution.execution_rid)
            assert record["Status_Detail"] == "Processing step 2"

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
            downloaded = download_execution.download_asset(
                asset_rid, Path(tmpdir), update_catalog=False
            )
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
            downloaded = check_execution.download_asset(
                asset_rid, Path(tmpdir), update_catalog=False
            )
            assert "Model_File" in downloaded.asset_types

    def test_upload_progress_callback(self, basic_execution):
        """Test that upload progress callback is invoked during upload."""
        from deriva_ml import UploadProgress

        progress_updates = []

        def progress_callback(progress: UploadProgress) -> None:
            progress_updates.append({
                "file_name": progress.file_name,
                "phase": progress.phase,
                "message": progress.message,
                "percent_complete": progress.percent_complete,
            })

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
            assert exe.datasets[0].dataset_rid == dataset_rid

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
                datasets=[
                    DatasetSpec(rid=ds.dataset_rid, version=ds.current_version)
                    for ds in all_datasets[:2]
                ],
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
        subjects = list(pb.schemas[ml.domain_schema].Subject.entities().fetch())

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

        # Verify features were uploaded
        status = get_execution_status(ml, execution.execution_rid)
        assert status == "Completed"


# =============================================================================
# TestExecutionRestore - Execution Restoration Tests
# =============================================================================


class TestExecutionRestore:
    """Tests for restoring previous executions."""

    def test_restore_execution(self, basic_execution, tmp_path):
        """Test restoring a previous execution."""
        ml = basic_execution._ml_object

        # Create and complete an execution
        with basic_execution.execute() as exe:
            original_rid = exe.execution_rid
            create_test_asset(exe, "original.txt", "Original content")

        basic_execution.upload_execution_outputs()

        # Create a new ML instance and restore
        restored = ml.restore_execution(original_rid)
        assert restored.execution_rid == original_rid


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

        status = get_execution_status(ml, execution.execution_rid)
        assert status == "Completed"
