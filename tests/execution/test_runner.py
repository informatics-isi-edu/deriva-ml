"""Tests for the execution runner module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from functools import partial

from deriva_ml.execution.runner import (
    run_model,
    create_model_config,
    reset_multirun_state,
    _is_multirun,
    _get_job_num,
    _multirun_state,
)
from deriva_ml.execution import ExecutionConfiguration, Workflow


class TestMultirunDetection:
    """Tests for multirun mode detection functions."""

    def test_is_multirun_returns_false_when_no_hydra(self):
        """Test that _is_multirun returns False when Hydra is not initialized."""
        reset_multirun_state()
        assert _is_multirun() is False

    def test_get_job_num_returns_zero_when_no_hydra(self):
        """Test that _get_job_num returns 0 when Hydra is not initialized."""
        reset_multirun_state()
        assert _get_job_num() == 0

    @patch("deriva_ml.execution.runner.HydraConfig")
    def test_is_multirun_returns_true_for_multirun_mode(self, mock_hydra_config):
        """Test that _is_multirun returns True when in multirun mode."""
        mock_cfg = MagicMock()
        mock_cfg.mode.value = 2  # MULTIRUN mode
        mock_hydra_config.get.return_value = mock_cfg

        assert _is_multirun() is True

    @patch("deriva_ml.execution.runner.HydraConfig")
    def test_is_multirun_returns_false_for_single_run(self, mock_hydra_config):
        """Test that _is_multirun returns False in single run mode."""
        mock_cfg = MagicMock()
        mock_cfg.mode.value = 1  # RUN mode
        mock_hydra_config.get.return_value = mock_cfg

        assert _is_multirun() is False

    @patch("deriva_ml.execution.runner.HydraConfig")
    def test_get_job_num_returns_correct_value(self, mock_hydra_config):
        """Test that _get_job_num returns the correct job number."""
        mock_cfg = MagicMock()
        mock_cfg.job.num = 5
        mock_hydra_config.get.return_value = mock_cfg

        assert _get_job_num() == 5


class TestResetMultirunState:
    """Tests for multirun state reset function."""

    def test_reset_clears_all_state(self):
        """Test that reset_multirun_state clears all fields."""
        # Set some state
        _multirun_state.parent_execution_rid = "test-rid"
        _multirun_state.parent_execution = Mock()
        _multirun_state.ml_instance = Mock()
        _multirun_state.job_sequence = 5
        _multirun_state.sweep_dir = "/some/path"

        # Reset
        reset_multirun_state()

        # Verify all cleared
        assert _multirun_state.parent_execution_rid is None
        assert _multirun_state.parent_execution is None
        assert _multirun_state.ml_instance is None
        assert _multirun_state.job_sequence == 0
        assert _multirun_state.sweep_dir is None


class TestCreateModelConfig:
    """Tests for the create_model_config helper function."""

    def test_creates_config_with_default_class(self):
        """Test creating config without specifying a class."""
        config = create_model_config()
        assert config is not None

    def test_creates_config_with_custom_description(self):
        """Test creating config with a custom description."""
        config = create_model_config(description="Custom description")
        assert config is not None

    def test_creates_config_with_custom_defaults(self):
        """Test creating config with custom hydra defaults."""
        custom_defaults = [
            "_self_",
            {"deriva_ml": "custom_deriva"},
        ]
        config = create_model_config(hydra_defaults=custom_defaults)
        assert config is not None

    def test_creates_config_with_deriva_ml_class(self):
        """Test creating config with the DerivaML class explicitly."""
        from deriva_ml import DerivaML

        # Use the actual DerivaML class (which is importable)
        config = create_model_config(DerivaML)
        assert config is not None


class TestRunModelIntegration:
    """Integration tests for run_model using the test catalog.

    These tests require a running Deriva instance.
    """

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset multirun state before each test."""
        reset_multirun_state()
        yield
        reset_multirun_state()

    def test_multirun_state_persists_across_calls(self):
        """Test that multirun state persists correctly across calls."""
        # Simulate setting multirun state
        _multirun_state.parent_execution_rid = "parent-rid"
        _multirun_state.job_sequence = 0

        # Verify it persists
        assert _multirun_state.parent_execution_rid == "parent-rid"
        assert _multirun_state.job_sequence == 0

        # Simulate incrementing
        _multirun_state.job_sequence += 1
        assert _multirun_state.job_sequence == 1


class TestRunModelWithMocks:
    """Tests for run_model using mocks to isolate from catalog."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset multirun state before each test."""
        reset_multirun_state()
        yield
        reset_multirun_state()

    @patch("deriva_ml.execution.runner._is_multirun")
    def test_run_model_calls_model_config(self, mock_is_multirun):
        """Test that run_model calls the model_config function."""
        # Setup mocks
        mock_is_multirun.return_value = False

        mock_ml_class = MagicMock()
        mock_ml_instance = MagicMock()
        mock_ml_class.instantiate.return_value = mock_ml_instance

        mock_execution = MagicMock()
        mock_execution.execute.return_value.__enter__ = Mock(return_value=mock_execution)
        mock_execution.execute.return_value.__exit__ = Mock(return_value=False)
        mock_execution.upload_execution_outputs.return_value = {}
        mock_ml_instance.create_execution.return_value = mock_execution

        mock_model_config = Mock()

        # Create minimal config
        mock_config = MagicMock()
        mock_workflow = MagicMock()

        # Run with explicit ml_class to bypass the internal import
        run_model(
            deriva_ml=mock_config,
            datasets=[],
            assets=[],
            description="Test",
            workflow=mock_workflow,
            model_config=mock_model_config,
            dry_run=False,
            ml_class=mock_ml_class,
        )

        # Verify model_config was called
        mock_model_config.assert_called_once()

    @patch("deriva_ml.execution.runner._is_multirun")
    def test_run_model_skips_model_in_dry_run(self, mock_is_multirun):
        """Test that run_model skips model execution in dry run mode."""
        # Setup mocks
        mock_is_multirun.return_value = False

        mock_ml_class = MagicMock()
        mock_ml_instance = MagicMock()
        mock_ml_class.instantiate.return_value = mock_ml_instance

        mock_execution = MagicMock()
        mock_execution.execute.return_value.__enter__ = Mock(return_value=mock_execution)
        mock_execution.execute.return_value.__exit__ = Mock(return_value=False)
        mock_ml_instance.create_execution.return_value = mock_execution

        mock_model_config = Mock()

        # Create minimal config
        mock_config = MagicMock()
        mock_workflow = MagicMock()

        # Run in dry_run mode with explicit ml_class
        run_model(
            deriva_ml=mock_config,
            datasets=[],
            assets=[],
            description="Test",
            workflow=mock_workflow,
            model_config=mock_model_config,
            dry_run=True,
            ml_class=mock_ml_class,
        )

        # Verify model_config was NOT called
        mock_model_config.assert_not_called()

    @patch("deriva_ml.execution.runner._is_multirun")
    @patch("deriva_ml.execution.runner.HydraConfig")
    def test_run_model_creates_parent_in_multirun(self, mock_hydra_config, mock_is_multirun):
        """Test that run_model creates parent execution in multirun mode."""
        # Setup mocks
        mock_is_multirun.return_value = True

        mock_hydra_cfg = MagicMock()
        mock_hydra_cfg.job.num = 0
        mock_hydra_cfg.overrides.task = ["+experiment=test1,test2"]
        mock_hydra_config.get.return_value = mock_hydra_cfg

        mock_ml_class = MagicMock()
        mock_ml_instance = MagicMock()
        mock_ml_class.instantiate.return_value = mock_ml_instance

        mock_parent_execution = MagicMock()
        mock_parent_execution.execution_rid = "parent-rid"
        mock_child_execution = MagicMock()
        mock_child_execution.execution_rid = "child-rid"
        mock_child_execution.execute.return_value.__enter__ = Mock(return_value=mock_child_execution)
        mock_child_execution.execute.return_value.__exit__ = Mock(return_value=False)
        mock_child_execution.upload_execution_outputs.return_value = {}

        # First call creates parent, second call creates child
        mock_ml_instance.create_execution.side_effect = [
            mock_parent_execution,
            mock_child_execution,
        ]

        mock_model_config = Mock()
        mock_config = MagicMock()
        mock_workflow = MagicMock()

        # Run first job with explicit ml_class
        run_model(
            deriva_ml=mock_config,
            datasets=[],
            assets=[],
            description="Test",
            workflow=mock_workflow,
            model_config=mock_model_config,
            dry_run=False,
            ml_class=mock_ml_class,
        )

        # Verify parent execution was created
        assert _multirun_state.parent_execution is not None
        assert _multirun_state.parent_execution_rid == "parent-rid"

        # Verify child was linked to parent
        mock_parent_execution.add_nested_execution.assert_called_once()

    @patch("deriva_ml.execution.runner._is_multirun")
    def test_run_model_uses_custom_ml_class(self, mock_is_multirun):
        """Test that run_model uses a custom ml_class when provided."""
        # Setup mocks
        mock_is_multirun.return_value = False

        mock_custom_class = MagicMock()
        mock_ml_instance = MagicMock()
        mock_custom_class.instantiate.return_value = mock_ml_instance

        mock_execution = MagicMock()
        mock_execution.execute.return_value.__enter__ = Mock(return_value=mock_execution)
        mock_execution.execute.return_value.__exit__ = Mock(return_value=False)
        mock_execution.upload_execution_outputs.return_value = {}
        mock_ml_instance.create_execution.return_value = mock_execution

        mock_model_config = Mock()
        mock_config = MagicMock()
        mock_workflow = MagicMock()

        # Run with custom class
        run_model(
            deriva_ml=mock_config,
            datasets=[],
            assets=[],
            description="Test",
            workflow=mock_workflow,
            model_config=mock_model_config,
            dry_run=False,
            ml_class=mock_custom_class,
        )

        # Verify custom class was used for instantiation
        mock_custom_class.instantiate.assert_called_once_with(mock_config)
