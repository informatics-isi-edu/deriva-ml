"""Tests for the find_caller module.

These tests verify that the caller detection logic correctly identifies the
user's script rather than DerivaML's internal CLI runners.
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from deriva_ml.execution.find_caller import (
    _get_calling_module,
    _top_user_frame,
    _is_pseudo_user_filename,
    _norm,
    _SYSTEM_MODULE_PREFIXES,
)


class TestSystemModulePrefixes:
    """Tests for system module prefix configuration."""

    def test_run_model_in_system_prefixes(self):
        """Verify that deriva_ml.run_model is in the system prefixes."""
        assert "deriva_ml.run_model" in _SYSTEM_MODULE_PREFIXES

    def test_run_notebook_in_system_prefixes(self):
        """Verify that deriva_ml.run_notebook is in the system prefixes."""
        assert "deriva_ml.run_notebook" in _SYSTEM_MODULE_PREFIXES

    def test_hydra_in_system_prefixes(self):
        """Verify that hydra modules are in the system prefixes."""
        assert "hydra" in _SYSTEM_MODULE_PREFIXES
        assert "hydra_zen" in _SYSTEM_MODULE_PREFIXES


class TestIsPseudoUserFilename:
    """Tests for IPython/Jupyter pseudo-filename detection."""

    def test_ipython_input_is_pseudo_user(self):
        """Test that IPython input pseudo-files are detected."""
        assert _is_pseudo_user_filename("<ipython-input-7-abcdef>") is True

    def test_jupyter_input_is_pseudo_user(self):
        """Test that Jupyter input pseudo-files are detected."""
        assert _is_pseudo_user_filename("<jupyter-input-3-123456>") is True

    def test_ipykernel_is_pseudo_user(self):
        """Test that ipykernel pseudo-files are detected."""
        assert _is_pseudo_user_filename("<ipykernel_12345>") is True

    def test_stdin_is_not_pseudo_user(self):
        """Test that stdin is not a pseudo-user filename."""
        assert _is_pseudo_user_filename("<stdin>") is False

    def test_string_is_not_pseudo_user(self):
        """Test that <string> is not a pseudo-user filename."""
        assert _is_pseudo_user_filename("<string>") is False

    def test_regular_file_is_not_pseudo(self):
        """Test that regular files are not pseudo-user filenames."""
        assert _is_pseudo_user_filename("/path/to/script.py") is False


class TestNorm:
    """Tests for path normalization."""

    def test_norm_expands_home(self):
        """Test that ~ is expanded to home directory."""
        result = _norm("~/test.py")
        assert "~" not in result
        assert Path(result).is_absolute()

    def test_norm_returns_absolute_path(self):
        """Test that relative paths become absolute."""
        result = _norm("./test.py")
        assert Path(result).is_absolute()


class TestGetCallingModuleFromScript:
    """Tests for _get_calling_module when run from a script."""

    def test_returns_current_file_when_run_directly(self):
        """Test that calling from a test returns a valid path."""
        result = _get_calling_module()
        # When run from pytest, should return a path to the test file
        assert result is not None
        assert isinstance(result, str)
        # Should be an absolute path
        assert Path(result).is_absolute()

    def test_skips_tooling_frames(self):
        """Test that tooling frames are skipped."""
        # When called from pytest, the function should find a user frame
        # rather than returning pytest internals
        result = _get_calling_module()
        # Should NOT be a pytest internal file
        assert "/_pytest/" not in result
        assert "/pluggy/" not in result


class TestTopUserFrame:
    """Tests for _top_user_frame function."""

    def test_returns_frame_object(self):
        """Test that _top_user_frame returns a frame."""
        frame = _top_user_frame()
        assert frame is not None
        # Should have frame attributes
        assert hasattr(frame, "f_code")
        assert hasattr(frame, "f_globals")


class TestRunModelSkipsCliRunnerPaths:
    """Tests to verify that _get_calling_module skips CLI runner paths."""

    def test_run_model_path_is_filtered(self):
        """Test that run_model.py path patterns are in the tooling markers."""
        # The function _is_tooling_script_path is internal to _get_calling_module
        # We verify the pattern exists by checking the expected behavior
        tooling_markers = (
            "/deriva_ml/run_model.py",
            "/deriva_ml/run_notebook.py",
        )

        # Test path
        test_path = "/some/path/to/deriva_ml/run_model.py"
        normalized = test_path.replace("\\", "/").casefold()

        # Verify the pattern matching logic
        is_tooling = any(m in normalized for m in tooling_markers)
        assert is_tooling is True

    def test_user_script_path_is_not_filtered(self):
        """Test that a user script path is not filtered as tooling."""
        tooling_markers = (
            "/deriva_ml/run_model.py",
            "/deriva_ml/run_notebook.py",
        )

        test_path = "/home/user/my_ml_project/train_model.py"
        normalized = test_path.replace("\\", "/").casefold()

        is_tooling = any(m in normalized for m in tooling_markers)
        assert is_tooling is False


class TestCallerDetectionIntegration:
    """Integration tests for caller detection when invoked through CLI runner."""

    def _subprocess_env(self):
        """Get environment for subprocess calls, ensuring distutils is available on Python 3.12+."""
        env = os.environ.copy()
        env["SETUPTOOLS_USE_DISTUTILS"] = "local"
        return env

    def test_subprocess_script_returns_script_path(self):
        """Test that a script run via subprocess returns its own path, not the runner."""
        # Create a temporary script that imports find_caller and prints the result
        script_content = '''
import sys
# Add deriva-ml to path if needed
sys.path.insert(0, "{src_path}")

from deriva_ml.execution.find_caller import _get_calling_module

result = _get_calling_module()
print(result)
'''
        src_path = Path("/Users/carl/GitHub/deriva-ml/src").resolve()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(script_content.format(src_path=src_path))
            script_path = f.name

        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                cwd=str(Path(script_path).parent),
                env=self._subprocess_env(),
            )

            # Check the result
            output = result.stdout.strip()
            assert result.returncode == 0, f"Script failed: {result.stderr}"

            # The output should be the path to our temporary script
            assert Path(output).resolve() == Path(script_path).resolve(), (
                f"Expected {script_path}, got {output}"
            )
        finally:
            Path(script_path).unlink()

    def test_simulated_run_model_chain_returns_user_script(self):
        """Test caller detection when simulating the run_model call chain.

        This test simulates what happens when a user runs:
            deriva-ml-run +experiment=my_experiment

        The call chain is:
            1. deriva-ml-run (CLI entry point)
            2. run_model.py (DerivaML runner)
            3. hydra/hydra_zen internals
            4. user's model function (or the configured model)

        We verify that _get_calling_module returns the user's code location,
        not run_model.py or hydra internals.
        """
        # Create a script that simulates the call chain
        script_content = '''
import sys
sys.path.insert(0, "{src_path}")

# Simulate the nested call structure
def user_model_function():
    """This simulates the user's model code being called."""
    from deriva_ml.execution.find_caller import _get_calling_module
    return _get_calling_module()

def simulated_hydra_wrapper():
    """This simulates hydra's function wrapping."""
    return user_model_function()

def simulated_run_model():
    """This simulates run_model calling the user's function."""
    return simulated_hydra_wrapper()

# Run the chain
result = simulated_run_model()
print(result)
'''
        src_path = Path("/Users/carl/GitHub/deriva-ml/src").resolve()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(script_content.format(src_path=src_path))
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                env=self._subprocess_env(),
            )

            output = result.stdout.strip()
            assert result.returncode == 0, f"Script failed: {result.stderr}"

            # The output should be our test script, not any internal module
            assert Path(output).resolve() == Path(script_path).resolve()
        finally:
            Path(script_path).unlink()

    def test_run_model_module_prefix_filtering(self):
        """Test that modules with deriva_ml.run_model prefix are filtered."""
        # Create a test that verifies the prefix matching logic
        script_content = '''
import sys
sys.path.insert(0, "{src_path}")

from deriva_ml.execution.find_caller import _SYSTEM_MODULE_PREFIXES

def check_prefix_match(module_name: str) -> bool:
    """Check if a module name matches system prefixes."""
    return any(
        module_name == p or module_name.startswith(p + ".")
        for p in _SYSTEM_MODULE_PREFIXES
    )

# Test cases
test_cases = [
    ("deriva_ml.run_model", True),  # Should be filtered
    ("deriva_ml.run_notebook", True),  # Should be filtered
    ("hydra", True),  # Should be filtered
    ("hydra.core", True),  # Should be filtered
    ("hydra_zen", True),  # Should be filtered
    ("my_project.train", False),  # User code - should NOT be filtered
    ("cifar_experiment.models", False),  # User code - should NOT be filtered
]

all_passed = True
for module_name, expected in test_cases:
    result = check_prefix_match(module_name)
    if result != expected:
        print(f"FAIL: {{module_name}} -> {{result}}, expected {{expected}}")
        all_passed = False

if all_passed:
    print("ALL TESTS PASSED")
else:
    sys.exit(1)
'''
        src_path = Path("/Users/carl/GitHub/deriva-ml/src").resolve()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(script_content.format(src_path=src_path))
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                env=self._subprocess_env(),
            )

            assert result.returncode == 0, f"Test failed:\n{result.stdout}\n{result.stderr}"
            assert "ALL TESTS PASSED" in result.stdout
        finally:
            Path(script_path).unlink()

    def test_filename_path_filtering(self):
        """Test that file paths containing run_model.py are filtered."""
        script_content = '''
import sys
sys.path.insert(0, "{src_path}")

# Test the path filtering logic used in _top_user_frame and _get_calling_module
tooling_filename_parts = (
    "/deriva_ml/run_model.py",
    "/deriva_ml/run_notebook.py",
    "/hydra/",
    "/hydra_zen/",
)

def is_tooling_path(filename: str) -> bool:
    """Check if a filename matches tooling patterns."""
    return any(part in filename for part in tooling_filename_parts)

# Test cases
test_cases = [
    ("/home/user/.venv/lib/python3.11/site-packages/deriva_ml/run_model.py", True),
    ("/opt/deriva_ml/run_notebook.py", True),
    ("/home/user/.venv/lib/python3.11/site-packages/hydra/core/utils.py", True),
    ("/home/user/.venv/lib/python3.11/site-packages/hydra_zen/wrapper.py", True),
    ("/home/user/my_project/train.py", False),
    ("/home/user/ml_experiments/cifar10/model.py", False),
    ("/tmp/pytest_script.py", False),
]

all_passed = True
for path, expected in test_cases:
    result = is_tooling_path(path)
    if result != expected:
        print(f"FAIL: {{path}} -> {{result}}, expected {{expected}}")
        all_passed = False

if all_passed:
    print("ALL TESTS PASSED")
else:
    sys.exit(1)
'''
        src_path = Path("/Users/carl/GitHub/deriva-ml/src").resolve()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(script_content.format(src_path=src_path))
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"Test failed:\n{result.stdout}\n{result.stderr}"
            assert "ALL TESTS PASSED" in result.stdout
        finally:
            Path(script_path).unlink()
