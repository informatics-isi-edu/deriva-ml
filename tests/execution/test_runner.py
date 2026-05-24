"""Tests for the execution runner module."""

import logging
from unittest.mock import MagicMock, Mock, patch

import pytest

from deriva_ml.core.constants import DRY_RUN_RID
from deriva_ml.execution.upload_report import UploadReport
from deriva_ml.core.exceptions import DerivaMLStateInconsistency
from deriva_ml.execution.runner import (
    _complete_parent_execution,
    _get_job_num,
    _is_multirun,
    _multirun_state,
    create_model_config,
    reset_multirun_state,
    run_model,
)


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
        mock_execution.commit_output_assets.return_value = UploadReport(execution_rids=[], total_uploaded=0, total_failed=0, per_table={}, errors=[])
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
        mock_child_execution.commit_output_assets.return_value = UploadReport(execution_rids=[], total_uploaded=0, total_failed=0, per_table={}, errors=[])

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
        mock_execution.commit_output_assets.return_value = UploadReport(execution_rids=[], total_uploaded=0, total_failed=0, per_table={}, errors=[])
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


class TestRunModelUploadCallContract:
    """Regression tests for runner.run_model's call to commit_output_assets.

    Surfaced 2026-05-19: ``runner.run_model`` was calling
    ``execution.commit_output_assets(timeout=..., chunk_size=...)`` but
    the method's signature only accepts ``clean_folder`` and
    ``progress_callback``. The ``@validate_call`` decorator on the method
    made the mismatch fatal at runtime. Every other runner test in this
    file uses bare ``MagicMock()`` for the execution, which accepts any
    kwargs without complaint — so the bug went undetected until the
    e2e platform test exercised the real method.

    These tests use ``MagicMock(spec=Execution)`` so the mock validates
    kwargs against the real method signature.
    """

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset multirun state before each test."""
        reset_multirun_state()
        yield
        reset_multirun_state()

    @patch("deriva_ml.execution.runner._is_multirun")
    def test_run_model_call_matches_commit_output_assets_signature(self, mock_is_multirun):
        """run_model must only pass kwargs that commit_output_assets accepts.

        Before the 2026-05-19 fix, run_model passed timeout= and chunk_size=
        kwargs that the upload method's signature did not accept. The fix
        strips those (deferred until deriva-py adds per-call timeout and
        plumbed-through chunk_size support). This test pins the contract.
        """
        from deriva_ml.execution.execution import Execution

        mock_is_multirun.return_value = False

        mock_ml_class = MagicMock()
        mock_ml_instance = MagicMock()
        mock_ml_class.instantiate.return_value = mock_ml_instance

        # spec=Execution enforces that method calls match the real
        # Execution surface — passing unknown kwargs raises AttributeError
        # / TypeError at the mock layer, which is what we want.
        mock_execution = MagicMock(spec=Execution)
        mock_execution.execute.return_value.__enter__ = Mock(return_value=mock_execution)
        mock_execution.execute.return_value.__exit__ = Mock(return_value=False)
        mock_execution.commit_output_assets.return_value = UploadReport(execution_rids=[], total_uploaded=0, total_failed=0, per_table={}, errors=[])
        mock_ml_instance.create_execution.return_value = mock_execution

        mock_model_config = Mock()
        mock_config = MagicMock()
        mock_workflow = MagicMock()

        # Call should complete without raising. Pre-fix, this raised
        # because the commit_output_assets MagicMock(spec=...) rejected
        # the timeout=/chunk_size= kwargs the runner passed.
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

        # Verify the actual kwargs the runner passed are a subset of the
        # real method's signature.
        import inspect

        real_sig = inspect.signature(Execution.commit_output_assets)
        real_params = set(real_sig.parameters.keys())

        call_args = mock_execution.commit_output_assets.call_args
        passed_kwargs = set(call_args.kwargs.keys()) if call_args else set()

        unsupported = passed_kwargs - real_params
        assert not unsupported, (
            f"run_model passed kwargs not accepted by "
            f"Execution.commit_output_assets: {unsupported}. "
            f"Real method accepts: {real_params - {'self'}}."
        )


class TestCompleteParentExecutionDryRun:
    """Regression tests for the atexit handler under dry-run multirun.

    Issue #177: the parent supervisor for a dry-run multirun uses
    ``DRY_RUN_RID`` ("0000") as its execution_rid placeholder because the
    parent is intentionally never written to the workspace SQLite registry.
    The atexit handler used to unconditionally call ``execution_stop()`` on
    the placeholder, which raised ``DerivaMLStateInconsistency`` from the
    read-through ``start_time``/``status`` properties. The handler caught
    the exception and emitted a WARNING reading
    "Failed to complete parent execution: Execution 0000 no longer in
    workspace registry..." -- alarming text for behavior that is correct
    by design.

    These tests pin the fix: the handler must short-circuit on the
    ``DRY_RUN_RID`` placeholder, log a clear INFO message, and never touch
    ``execution_stop`` / ``commit_output_assets`` on the placeholder.
    """

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset multirun state before and after each test."""
        reset_multirun_state()
        yield
        reset_multirun_state()

    def test_dry_run_placeholder_does_not_warn(self, caplog):
        """No warning text about the placeholder RID escapes to the log.

        The handler used to call ``execution_stop()`` on a parent whose
        registry row never existed, catch ``DerivaMLStateInconsistency``,
        and log a WARNING. After the fix the placeholder is short-circuited
        before any registry-touching method runs.
        """
        # Build a parent execution that mimics a dry-run placeholder:
        # execution_stop / commit_output_assets would (under the bug)
        # raise DerivaMLStateInconsistency because no SQLite row exists.
        parent = MagicMock()
        parent.execution_rid = DRY_RUN_RID
        parent.execution_stop.side_effect = DerivaMLStateInconsistency(
            f"Execution {DRY_RUN_RID} no longer in workspace registry."
        )
        parent.commit_output_assets.side_effect = DerivaMLStateInconsistency(
            f"Execution {DRY_RUN_RID} no longer in workspace registry."
        )

        _multirun_state.parent_execution = parent
        _multirun_state.parent_execution_rid = DRY_RUN_RID
        _multirun_state.job_sequence = 3

        with caplog.at_level(logging.INFO):
            _complete_parent_execution()

        # No warnings (or higher) about the placeholder or registry.
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        for record in warnings:
            assert DRY_RUN_RID not in record.getMessage()
            assert "no longer in workspace registry" not in record.getMessage()
            assert "Failed to complete parent execution" not in record.getMessage()
        assert warnings == [], f"Unexpected warnings: {[r.getMessage() for r in warnings]}"

        # An INFO line explains the skip.
        info_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any("Dry-run" in m and "not recorded" in m for m in info_messages), (
            f"Expected dry-run skip INFO message, got: {info_messages}"
        )

        # The registry-touching methods were not invoked on the placeholder.
        parent.execution_stop.assert_not_called()
        parent.commit_output_assets.assert_not_called()

    def test_real_parent_still_completed_normally(self, caplog):
        """Non-placeholder parents still go through the full stop/upload path.

        Guards against an over-eager short-circuit that would silently skip
        real parent executions.
        """
        parent = MagicMock()
        parent.execution_rid = "1-ABCD"
        parent.commit_output_assets.return_value = UploadReport(execution_rids=[], total_uploaded=0, total_failed=0, per_table={}, errors=[])

        _multirun_state.parent_execution = parent
        _multirun_state.parent_execution_rid = "1-ABCD"
        _multirun_state.job_sequence = 2

        with caplog.at_level(logging.INFO):
            _complete_parent_execution()

        parent.execution_stop.assert_called_once()
        parent.commit_output_assets.assert_called_once()
        info_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any("Completed parent execution: 1-ABCD" in m for m in info_messages)
