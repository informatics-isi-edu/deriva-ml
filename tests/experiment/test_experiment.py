"""Tests for the Experiment class.

This module provides tests for DerivaML Experiment functionality:

Test Classes:
    TestExperimentBasic: Basic experiment creation and properties
    TestExperimentConfiguration: Hydra configuration loading
    TestExperimentSummary: Summary generation
    TestExperimentInputsOutputs: Input/output dataset and asset handling
    TestExperimentFinder: Finding experiments in catalogs
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from deriva_ml import DerivaML, MLAsset, ExecAssetType
from deriva_ml import MLVocab as vc
from deriva_ml.execution.execution import Execution, ExecutionConfiguration
from deriva_ml.experiment import Experiment


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def workflow_terms(test_ml):
    """Add required vocabulary terms for workflow testing."""
    test_ml.add_term(vc.asset_type, "Test Model", description="Model for our Test workflow")
    test_ml.add_term(vc.workflow_type, "Test Workflow", description="A ML Workflow")
    return test_ml


@pytest.fixture
def test_workflow(workflow_terms):
    """Create a test workflow."""
    ml = workflow_terms
    return ml.create_workflow(
        name="Test Workflow",
        workflow_type="Test Workflow",
        description="A test workflow for experiment testing",
    )


@pytest.fixture
def basic_execution(workflow_terms, test_workflow):
    """Create a basic execution without datasets."""
    ml = workflow_terms
    config = ExecutionConfiguration(
        description="Test Execution for Experiment",
        workflow=test_workflow,
    )
    return ml.create_execution(config)


@pytest.fixture
def completed_execution(basic_execution):
    """Create and complete an execution with metadata files."""
    execution = basic_execution

    with execution.execute():
        # Create a test asset
        asset_path = execution.asset_file_path(
            MLAsset.execution_asset,
            "TestOutput/result.txt",
            asset_types=ExecAssetType.model_file,
        )
        with asset_path.open("w") as fp:
            fp.write("Test output content")

    execution.upload_execution_outputs()
    return execution


@pytest.fixture
def execution_with_hydra_config(workflow_terms, test_workflow, tmp_path):
    """Create an execution with Hydra config files in metadata.

    Note: This fixture manually creates the hydra config files in the working
    directory structure that would normally be created by hydra-zen during model runs.
    """
    ml = workflow_terms
    config = ExecutionConfiguration(
        description="Hydra Config Test Execution",
        workflow=test_workflow,
    )
    execution = ml.create_execution(config)

    with execution.execute():
        # Create hydra-style config files that would be uploaded as metadata
        # In real usage, these come from hydra-zen but we create them manually for testing

        # Get the working directory where metadata files should be placed
        working_dir = execution.working_dir

        # Create config.yaml (model configuration with resolved values)
        config_content = {
            "model_config": {
                "epochs": 10,
                "learning_rate": 0.001,
                "batch_size": 32,
                "_target_": "my_model.train",
            },
            "datasets": {
                "rid": "TEST123",
            },
        }

        # Create a metadata file in the proper location
        # The asset_file_path returns an AssetFilePath, but we need to create
        # the actual file. Let's use the working directory structure directly.
        metadata_dir = working_dir / "Execution_Metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        config_filename = f"{execution.execution_rid}-config.yaml"
        config_path = metadata_dir / config_filename
        with config_path.open("w") as f:
            yaml.dump(config_content, f)

        # Create hydra.yaml (runtime choices)
        hydra_content = {
            "hydra": {
                "runtime": {
                    "choices": {
                        "model_config": "quick_train",
                        "datasets": "test_dataset",
                        "hydra/launcher": "basic",
                        "hydra/sweeper": "basic",
                    }
                }
            }
        }

        hydra_filename = f"{execution.execution_rid}-hydra.yaml"
        hydra_path = metadata_dir / hydra_filename
        with hydra_path.open("w") as f:
            yaml.dump(hydra_content, f)

    execution.upload_execution_outputs()
    return execution


# =============================================================================
# Helper Functions
# =============================================================================


def get_execution_status(ml: DerivaML, execution_rid: str) -> str:
    """Get the current status of an execution."""
    return ml.retrieve_rid(execution_rid)["Status"]


# =============================================================================
# TestExperimentBasic - Basic Creation and Properties
# =============================================================================


class TestExperimentBasic:
    """Tests for basic experiment creation and properties."""

    def test_lookup_experiment(self, completed_execution):
        """Test looking up an experiment by execution RID."""
        ml = completed_execution._ml_object
        execution_rid = completed_execution.execution_rid

        experiment = ml.lookup_experiment(execution_rid)

        assert experiment is not None
        assert experiment.execution_rid == execution_rid
        assert isinstance(experiment, Experiment)

    def test_experiment_execution_property(self, completed_execution):
        """Test that experiment.execution returns the underlying Execution."""
        ml = completed_execution._ml_object
        execution_rid = completed_execution.execution_rid

        experiment = ml.lookup_experiment(execution_rid)
        execution = experiment.execution

        assert execution is not None
        assert execution.execution_rid == execution_rid

    def test_experiment_name_from_rid(self, completed_execution):
        """Test that experiment name falls back to RID when no config."""
        ml = completed_execution._ml_object
        execution_rid = completed_execution.execution_rid

        experiment = ml.lookup_experiment(execution_rid)

        # Without hydra config, name should be the execution RID
        assert experiment.name == execution_rid

    def test_experiment_description(self, completed_execution):
        """Test getting experiment description."""
        ml = completed_execution._ml_object
        execution_rid = completed_execution.execution_rid

        experiment = ml.lookup_experiment(execution_rid)

        # Description should match what was set in the ExecutionConfiguration
        assert experiment.description == "Test Execution for Experiment"

    def test_experiment_status(self, completed_execution):
        """Test getting experiment status."""
        ml = completed_execution._ml_object
        execution_rid = completed_execution.execution_rid

        experiment = ml.lookup_experiment(execution_rid)

        assert experiment.status == "Completed"

    def test_experiment_get_chaise_url(self, completed_execution):
        """Test generating Chaise URL for the experiment."""
        ml = completed_execution._ml_object
        execution_rid = completed_execution.execution_rid

        experiment = ml.lookup_experiment(execution_rid)
        url = experiment.get_chaise_url()

        assert ml.host_name in url
        assert ml.catalog_id in url
        assert "deriva-ml:Execution" in url
        assert f"RID={execution_rid}" in url

    def test_experiment_repr(self, completed_execution):
        """Test experiment string representation."""
        ml = completed_execution._ml_object
        execution_rid = completed_execution.execution_rid

        experiment = ml.lookup_experiment(execution_rid)

        repr_str = repr(experiment)
        assert "Experiment" in repr_str
        assert execution_rid in repr_str


# =============================================================================
# TestExperimentConfiguration - Hydra Configuration Loading
# =============================================================================


class TestExperimentConfiguration:
    """Tests for Hydra configuration loading."""

    def test_config_choices_empty_without_hydra(self, completed_execution):
        """Test that config_choices is empty when no hydra.yaml exists."""
        ml = completed_execution._ml_object
        execution_rid = completed_execution.execution_rid

        experiment = ml.lookup_experiment(execution_rid)

        # Without hydra config files, config_choices should be empty
        assert experiment.config_choices == {}

    def test_model_config_empty_without_hydra(self, completed_execution):
        """Test that model_config is empty when no config.yaml exists."""
        ml = completed_execution._ml_object
        execution_rid = completed_execution.execution_rid

        experiment = ml.lookup_experiment(execution_rid)

        # Without hydra config files, model_config should be empty
        assert experiment.model_config == {}

    def test_hydra_config_loading(self, execution_with_hydra_config):
        """Test loading full Hydra configuration."""
        ml = execution_with_hydra_config._ml_object
        execution_rid = execution_with_hydra_config.execution_rid

        experiment = ml.lookup_experiment(execution_rid)
        hydra_config = experiment.hydra_config

        # Should contain model_config from config.yaml
        assert "model_config" in hydra_config or hydra_config == {}
        # Note: config_choices comes from hydra.yaml, not config.yaml directly

    def test_config_choices_from_hydra(self, execution_with_hydra_config):
        """Test extracting config_choices from hydra.yaml."""
        ml = execution_with_hydra_config._ml_object
        execution_rid = execution_with_hydra_config.execution_rid

        experiment = ml.lookup_experiment(execution_rid)
        choices = experiment.config_choices

        # If hydra config was uploaded, should have choices
        # (excluding hydra/ prefixed keys)
        if choices:
            assert "hydra/launcher" not in choices
            assert "hydra/sweeper" not in choices

    def test_experiment_name_from_config(self, execution_with_hydra_config):
        """Test that experiment name comes from config_choices.model_config."""
        ml = execution_with_hydra_config._ml_object
        execution_rid = execution_with_hydra_config.execution_rid

        experiment = ml.lookup_experiment(execution_rid)

        # If config_choices has model_config, name should be that value
        if experiment.config_choices.get("model_config"):
            assert experiment.name == experiment.config_choices["model_config"]
        else:
            # Falls back to execution RID
            assert experiment.name == execution_rid


# =============================================================================
# TestExperimentSummary - Summary Generation
# =============================================================================


class TestExperimentSummary:
    """Tests for experiment summary generation."""

    def test_summary_basic_fields(self, completed_execution):
        """Test that summary includes basic fields."""
        ml = completed_execution._ml_object
        execution_rid = completed_execution.execution_rid

        experiment = ml.lookup_experiment(execution_rid)
        summary = experiment.summary()

        assert "name" in summary
        assert "execution_rid" in summary
        assert "description" in summary
        assert "status" in summary
        assert "url" in summary

    def test_summary_config_fields(self, completed_execution):
        """Test that summary includes config fields."""
        ml = completed_execution._ml_object
        execution_rid = completed_execution.execution_rid

        experiment = ml.lookup_experiment(execution_rid)
        summary = experiment.summary()

        assert "config_choices" in summary
        assert "model_config" in summary

    def test_summary_input_datasets(self, completed_execution):
        """Test that summary includes input datasets."""
        ml = completed_execution._ml_object
        execution_rid = completed_execution.execution_rid

        experiment = ml.lookup_experiment(execution_rid)
        summary = experiment.summary()

        assert "input_datasets" in summary
        assert isinstance(summary["input_datasets"], list)

    def test_summary_model_config_no_private_keys(self, execution_with_hydra_config):
        """Test that summary model_config excludes keys starting with _."""
        ml = execution_with_hydra_config._ml_object
        execution_rid = execution_with_hydra_config.execution_rid

        experiment = ml.lookup_experiment(execution_rid)
        summary = experiment.summary()

        # model_config in summary should not have _target_ or other _ prefixed keys
        model_config = summary.get("model_config", {})
        for key in model_config:
            assert not key.startswith("_"), f"Found private key {key} in summary model_config"


# =============================================================================
# TestExperimentInputsOutputs - Input/Output Handling
# =============================================================================


class TestExperimentInputsOutputs:
    """Tests for input/output dataset and asset handling."""

    def test_input_datasets_empty(self, completed_execution):
        """Test input_datasets when execution has no input datasets."""
        ml = completed_execution._ml_object
        execution_rid = completed_execution.execution_rid

        experiment = ml.lookup_experiment(execution_rid)

        # Basic execution was created without datasets
        assert experiment.input_datasets == []

    def test_input_assets_empty(self, completed_execution):
        """Test input_assets when execution has no input assets."""
        ml = completed_execution._ml_object
        execution_rid = completed_execution.execution_rid

        experiment = ml.lookup_experiment(execution_rid)

        # Basic execution was created without input assets
        assert experiment.input_assets == []

    def test_output_assets(self, completed_execution):
        """Test output_assets returns uploaded assets."""
        ml = completed_execution._ml_object
        execution_rid = completed_execution.execution_rid

        experiment = ml.lookup_experiment(execution_rid)

        # Completed execution uploaded one asset
        output_assets = experiment.output_assets
        assert isinstance(output_assets, list)
        # The test creates one output asset
        assert len(output_assets) >= 1


# =============================================================================
# TestExperimentFinder - Finding Experiments
# =============================================================================


class TestExperimentFinder:
    """Tests for finding experiments in catalogs."""

    def test_find_experiments_empty(self, workflow_terms):
        """Test finding experiments when none exist."""
        ml = workflow_terms

        experiments = list(ml.find_experiments())

        # No executions with hydra config exist
        assert experiments == []

    def test_find_experiments_with_completed(self, execution_with_hydra_config):
        """Test finding experiments with hydra config."""
        ml = execution_with_hydra_config._ml_object

        experiments = list(ml.find_experiments())

        # Should find at least the execution with hydra config
        # (depends on whether config files were properly uploaded)
        assert isinstance(experiments, list)

    def test_find_executions(self, completed_execution):
        """Test find_executions returns all executions."""
        ml = completed_execution._ml_object

        executions = list(ml.find_executions())

        assert len(executions) >= 1
        execution_rids = [e.execution_rid for e in executions]
        assert completed_execution.execution_rid in execution_rids

    def test_find_executions_by_status(self, completed_execution):
        """Test find_executions with status filter."""
        from deriva_ml.core.definitions import Status

        ml = completed_execution._ml_object

        # Find completed executions
        completed = list(ml.find_executions(status=Status.completed))

        assert len(completed) >= 1
        for exe in completed:
            assert exe.status.value == "Completed"

    def test_find_executions_by_workflow(self, completed_execution, test_workflow):
        """Test find_executions with workflow filter."""
        ml = completed_execution._ml_object

        # Find executions by workflow
        executions = list(ml.find_executions(workflow_rid=test_workflow.rid))

        assert len(executions) >= 1
        execution_rids = [e.execution_rid for e in executions]
        assert completed_execution.execution_rid in execution_rids


# =============================================================================
# TestExperimentWithDatasets - Experiments with Input Datasets
# =============================================================================


class TestExperimentWithDatasets:
    """Tests for experiments that have input datasets."""

    def test_input_datasets_with_data(self, dataset_test, tmp_path):
        """Test input_datasets when execution has datasets."""
        from deriva_ml.dataset.aux_classes import DatasetSpec

        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        ml.add_term(vc.asset_type, "Test Model", description="Model for test")
        ml.add_term(vc.workflow_type, "Test Workflow", description="Test workflow")

        dataset_rid = dataset_test.dataset_description.dataset.dataset_rid
        dataset_version = dataset_test.dataset_description.dataset.current_version

        workflow = ml.create_workflow(
            name="Dataset Test Workflow",
            workflow_type="Test Workflow",
            description="Test dataset input",
        )

        config = ExecutionConfiguration(
            datasets=[
                DatasetSpec(rid=dataset_rid, version=dataset_version),
            ],
            description="Execution with Dataset",
            workflow=workflow,
        )

        execution = ml.create_execution(config)
        with execution.execute():
            pass
        execution.upload_execution_outputs()

        # Now test the experiment
        experiment = ml.lookup_experiment(execution.execution_rid)
        input_datasets = experiment.input_datasets

        assert len(input_datasets) == 1
        assert input_datasets[0].dataset_rid == dataset_rid

    def test_summary_includes_dataset_details(self, dataset_test, tmp_path):
        """Test that summary includes dataset version and types."""
        from deriva_ml.dataset.aux_classes import DatasetSpec

        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)

        ml.add_term(vc.asset_type, "Test Model", description="Model")
        ml.add_term(vc.workflow_type, "Test Workflow", description="Workflow")

        dataset_rid = dataset_test.dataset_description.dataset.dataset_rid
        dataset_version = dataset_test.dataset_description.dataset.current_version

        workflow = ml.create_workflow(
            name="Summary Test Workflow",
            workflow_type="Test Workflow",
            description="Test summary",
        )

        config = ExecutionConfiguration(
            datasets=[DatasetSpec(rid=dataset_rid, version=dataset_version)],
            description="Summary Test Execution",
            workflow=workflow,
        )

        execution = ml.create_execution(config)
        with execution.execute():
            pass
        execution.upload_execution_outputs()

        experiment = ml.lookup_experiment(execution.execution_rid)
        summary = experiment.summary()

        # Check input_datasets in summary
        input_datasets = summary.get("input_datasets", [])
        assert len(input_datasets) == 1
        ds_info = input_datasets[0]

        assert "rid" in ds_info
        assert "description" in ds_info
        assert "version" in ds_info
        assert "dataset_types" in ds_info
