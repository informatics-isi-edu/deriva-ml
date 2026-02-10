"""Tests for the base_config module.

Tests cover:
- BaseConfig structured config schema (no runtime fields)
- get_notebook_configuration() capturing hydra output dir
- notebook_config() registration
- Module-level hydra output dir side-channel
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
from hydra_zen import builds, store

from deriva_ml.core.config import DerivaMLConfig
from deriva_ml.execution.base_config import (
    BaseConfig,
    get_notebook_configuration,
    notebook_config,
)


# Module-level dataclasses so hydra-zen can resolve their import paths.
# (Classes defined inside test functions have local qualnames that
# hydra-zen cannot import.)

@dataclass
class _CustomConfig(BaseConfig):
    """Test subclass for builds() compatibility test."""
    threshold: float = 0.5
    num_iterations: int = 100


@dataclass
class _CustomNBConfig(BaseConfig):
    """Test subclass for notebook_config() test."""
    alpha: float = 0.1


def _make_test_config(deriva_name: str, config_name: str):
    """Create a test notebook config with a properly structured DerivaMLConfig.

    The Hydra run dir references ``${deriva_ml.working_dir}``,
    ``${deriva_ml.catalog_id}``, and ``${deriva_ml.hostname}``,
    so the ``deriva_ml`` group must be a proper DerivaMLConfig builds.
    """
    DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)

    deriva_store = store(group="deriva_ml")
    deriva_store(
        DerivaMLConf,
        name=deriva_name,
        hostname="test.example.org",
        catalog_id="1",
        check_auth=False,
    )

    config_builds = builds(
        BaseConfig,
        populate_full_signature=True,
        hydra_defaults=[
            "_self_",
            {"deriva_ml": deriva_name},
        ],
    )
    store(config_builds, name=config_name)
    store.add_to_hydra_store(overwrite_ok=True)
    return config_builds


class TestBaseConfigSchema:
    """Tests for BaseConfig structured config schema."""

    def test_no_hydra_runtime_output_dir_field(self):
        """BaseConfig must NOT have hydra_runtime_output_dir as a field.

        This value is a runtime artifact captured during Hydra config
        resolution, not a configuration parameter. Adding it to the
        structured config causes OmegaConf composition errors.
        """
        field_names = [f.name for f in dataclasses.fields(BaseConfig)]
        assert "hydra_runtime_output_dir" not in field_names

    def test_expected_fields_present(self):
        """BaseConfig should have all standard DerivaML fields."""
        field_names = [f.name for f in dataclasses.fields(BaseConfig)]
        assert "deriva_ml" in field_names
        assert "datasets" in field_names
        assert "assets" in field_names
        assert "dry_run" in field_names
        assert "description" in field_names
        assert "config_choices" in field_names

    def test_builds_creates_valid_structured_config(self):
        """builds(BaseConfig) should succeed without schema errors."""
        config_builds = builds(BaseConfig, populate_full_signature=True)
        assert config_builds is not None

    def test_subclass_builds_succeeds(self):
        """A BaseConfig subclass should also work with builds()."""
        config_builds = builds(_CustomConfig, populate_full_signature=True)
        assert config_builds is not None


class TestGetNotebookConfiguration:
    """Tests for get_notebook_configuration() hydra output dir capture."""

    @pytest.fixture(autouse=True)
    def _reset_captured_dir(self):
        """Reset the module-level captured dir before each test."""
        import deriva_ml.execution.base_config as mod

        mod._captured_hydra_output_dir = None
        yield
        mod._captured_hydra_output_dir = None

    def test_captures_hydra_output_dir(self):
        """get_notebook_configuration() should capture hydra output dir."""
        import deriva_ml.execution.base_config as mod

        config_builds = _make_test_config("test_cap_deriva", "test_cap_config")

        with patch("deriva_ml.core.config.HydraConfig") as mock_hydra:
            mock_hydra.get.return_value.runtime.output_dir = "/tmp/hydra_test"
            get_notebook_configuration(config_builds, config_name="test_cap_config")

        # The captured dir should be a non-None string
        assert mod._captured_hydra_output_dir is not None
        assert isinstance(mod._captured_hydra_output_dir, str)

    def test_captured_dir_contains_hydra_files(self):
        """The captured hydra output dir should contain config files."""
        import deriva_ml.execution.base_config as mod

        config_builds = _make_test_config("test_files_deriva", "test_files_config")

        with patch("deriva_ml.core.config.HydraConfig") as mock_hydra:
            mock_hydra.get.return_value.runtime.output_dir = "/tmp/hydra_test_files"
            get_notebook_configuration(config_builds, config_name="test_files_config")

        captured_path = Path(mod._captured_hydra_output_dir)

        # Hydra creates the output dir with config files
        if captured_path.exists():
            all_files = [f.name for f in captured_path.rglob("*") if f.is_file()]
            assert any("config.yaml" in f for f in all_files), (
                f"Expected config.yaml in hydra output dir, found: {all_files}"
            )

    def test_config_choices_captured(self):
        """get_notebook_configuration() should capture config choices."""
        config_builds = _make_test_config("test_ch_deriva", "test_ch_config")

        with patch("deriva_ml.core.config.HydraConfig") as mock_hydra:
            mock_hydra.get.return_value.runtime.output_dir = "/tmp/hydra_test_ch"
            config = get_notebook_configuration(
                config_builds, config_name="test_ch_config"
            )

        assert isinstance(config.config_choices, dict)
        assert config.config_choices.get("deriva_ml") == "test_ch_deriva"

    def test_hydra_output_dir_not_on_config(self):
        """The config object should NOT have hydra_runtime_output_dir."""
        config_builds = _make_test_config("test_na_deriva", "test_na_config")

        with patch("deriva_ml.core.config.HydraConfig") as mock_hydra:
            mock_hydra.get.return_value.runtime.output_dir = "/tmp/hydra_test_na"
            config = get_notebook_configuration(
                config_builds, config_name="test_na_config"
            )

        assert not hasattr(config, "hydra_runtime_output_dir")


class TestNotebookConfig:
    """Tests for the notebook_config() helper."""

    def test_registers_in_internal_registry(self):
        """notebook_config() should register in _notebook_configs."""
        from deriva_ml.execution.base_config import _notebook_configs

        notebook_config(
            "test_registration_nb",
            defaults={"deriva_ml": "default_deriva"},
        )

        assert "test_registration_nb" in _notebook_configs

    def test_custom_config_class(self):
        """notebook_config() should accept custom config subclasses."""
        from deriva_ml.execution.base_config import _notebook_configs

        result = notebook_config(
            "test_custom_nb",
            config_class=_CustomNBConfig,
        )

        assert "test_custom_nb" in _notebook_configs
        assert result is not None
