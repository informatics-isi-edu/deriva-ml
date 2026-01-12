"""Tests for hydra-zen configuration integration.

These tests verify that hydra-zen can be used to configure DerivaML
and ExecutionConfiguration, including proper working directory setup
for Hydra output files.
"""

import getpass
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from hydra_zen import builds, instantiate, just, make_config
from omegaconf import OmegaConf

from deriva_ml.core.config import DerivaMLConfig
from deriva_ml.execution.execution_configuration import ExecutionConfiguration
from deriva_ml.dataset.aux_classes import DatasetSpec


class TestHydraZenDerivaMLConfig:
    """Test hydra-zen configuration for DerivaML."""

    def test_builds_creates_valid_config(self):
        """Test that builds() creates a valid structured config for DerivaMLConfig."""
        # Create a structured config using hydra-zen
        DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)

        # Instantiate with test values
        conf = DerivaMLConf(
            hostname="test.example.org",
            catalog_id="123",
            domain_schema="test_schema",
        )

        # Verify the config has expected structure
        assert conf.hostname == "test.example.org"
        assert conf.catalog_id == "123"
        assert conf.domain_schema == "test_schema"

    def test_builds_with_defaults(self):
        """Test that builds() preserves default values."""
        DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)

        conf = DerivaMLConf(hostname="test.example.org")

        # Check defaults are preserved
        assert conf.catalog_id == 1
        assert conf.ml_schema == "deriva-ml"
        assert conf.use_minid is True
        assert conf.check_auth is True

    def test_instantiate_creates_pydantic_model(self):
        """Test that instantiate() creates a proper DerivaMLConfig instance."""
        DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)

        conf = DerivaMLConf(
            hostname="test.example.org",
            catalog_id="42",
            check_auth=False,  # Disable auth check for testing
        )

        # Mock HydraConfig since we're not running under Hydra
        with patch("deriva_ml.core.config.HydraConfig") as mock_hydra:
            mock_hydra.get.return_value.runtime.output_dir = "/tmp/hydra_output"

            # Instantiate the config
            config = instantiate(conf)

            assert isinstance(config, DerivaMLConfig)
            assert config.hostname == "test.example.org"
            assert config.catalog_id == "42"

    def test_compute_workdir_with_custom_path(self, tmp_path):
        """Test that compute_workdir correctly handles custom paths."""
        custom_base = tmp_path / "custom_work"
        result = DerivaMLConfig.compute_workdir(custom_base)

        # Should append username and deriva-ml
        expected = custom_base / getpass.getuser() / "deriva-ml"
        assert result == expected.absolute()

    def test_compute_workdir_with_none(self):
        """Test that compute_workdir uses home directory when None."""
        result = DerivaMLConfig.compute_workdir(None)

        expected = Path.home() / "deriva-ml"
        assert result == expected.absolute()

    def test_omegaconf_resolver_registered(self):
        """Test that the compute_workdir resolver is registered with OmegaConf."""
        # The resolver should be registered when the module is imported
        # Test it by creating a config that uses the resolver
        cfg = OmegaConf.create({"path": "${compute_workdir:null}"})
        resolved = OmegaConf.to_container(cfg, resolve=True)

        expected = Path.home() / "deriva-ml"
        assert Path(resolved["path"]) == expected.absolute()

    def test_working_dir_used_in_hydra_config(self, tmp_path):
        """Test that working_dir is properly used for Hydra output configuration."""
        # Create a config with custom working_dir
        DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)

        conf = DerivaMLConf(
            hostname="test.example.org",
            working_dir=str(tmp_path / "work"),
            check_auth=False,
        )

        # The compute_workdir resolver should use this path
        computed = DerivaMLConfig.compute_workdir(tmp_path / "work")
        assert computed == (tmp_path / "work" / getpass.getuser() / "deriva-ml").absolute()

    def test_config_composition_with_store(self):
        """Test that configs can be composed using hydra-zen store."""
        DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)

        # Create different environment configs
        dev_conf = DerivaMLConf(
            hostname="dev.example.org",
            catalog_id="1",
        )

        prod_conf = DerivaMLConf(
            hostname="prod.example.org",
            catalog_id="100",
        )

        # Verify they have different values
        assert dev_conf.hostname != prod_conf.hostname
        assert dev_conf.catalog_id != prod_conf.catalog_id


class TestHydraZenExecutionConfiguration:
    """Test hydra-zen configuration for ExecutionConfiguration."""

    def test_builds_execution_config(self):
        """Test that builds() creates a valid ExecutionConfiguration."""
        ExecConf = builds(ExecutionConfiguration, populate_full_signature=True)

        conf = ExecConf(
            description="Test execution",
        )

        assert conf.description == "Test execution"

    def test_execution_config_with_datasets(self):
        """Test ExecutionConfiguration with dataset specifications."""
        # Build configs for nested structures
        DatasetSpecConf = builds(DatasetSpec, populate_full_signature=True)
        ExecConf = builds(ExecutionConfiguration, populate_full_signature=True)

        # Create dataset specs
        dataset1 = DatasetSpecConf(
            rid="1-ABC",
            version="1.0.0",
            materialize=True,
        )

        dataset2 = DatasetSpecConf(
            rid="2-DEF",
            version="2.0.0",
            materialize=False,
        )

        # Create execution config with datasets
        conf = ExecConf(
            description="Multi-dataset execution",
            datasets=[dataset1, dataset2],
        )

        assert conf.description == "Multi-dataset execution"
        assert len(conf.datasets) == 2

    def test_instantiate_execution_config(self):
        """Test that instantiate() creates a proper ExecutionConfiguration."""
        ExecConf = builds(ExecutionConfiguration, populate_full_signature=True)

        # Use valid RID format: 1-4 uppercase alphanumeric chars,
        # optionally followed by hyphen-separated groups of exactly 4 chars
        conf = ExecConf(
            description="Instantiation test",
            assets=["1ABC", "2DEF"],  # Valid short RIDs
        )

        # Instantiate
        exec_config = instantiate(conf)

        assert isinstance(exec_config, ExecutionConfiguration)
        assert exec_config.description == "Instantiation test"
        assert exec_config.assets == ["1ABC", "2DEF"]

    def test_execution_config_with_workflow_rid(self):
        """Test ExecutionConfiguration with workflow RID."""
        ExecConf = builds(ExecutionConfiguration, populate_full_signature=True)

        # Use valid RID format
        conf = ExecConf(
            workflow="WXYZ",  # Valid short RID
            description="Workflow test",
        )

        exec_config = instantiate(conf)
        assert exec_config.workflow == "WXYZ"

    def test_make_config_for_custom_execution(self):
        """Test using make_config for custom execution parameters."""
        # Create a custom config that includes execution params
        CustomExecConfig = make_config(
            execution=builds(ExecutionConfiguration),
            threshold=0.5,
            max_iterations=100,
            output_format="json",
        )

        conf = CustomExecConfig(
            execution={"description": "Custom params test"},
            threshold=0.75,
        )

        assert conf.threshold == 0.75
        assert conf.max_iterations == 100
        assert conf.output_format == "json"


class TestHydraZenIntegration:
    """Integration tests for hydra-zen with DerivaML."""

    def test_combined_deriva_and_execution_config(self):
        """Test composing DerivaML and ExecutionConfiguration together."""
        DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)
        ExecConf = builds(ExecutionConfiguration, populate_full_signature=True)

        # Create a combined config using structured configs as defaults
        CombinedConfig = make_config(
            deriva_ml=DerivaMLConf(hostname="default.example.org"),
            execution=ExecConf(description="default"),
        )

        # Create instance with overrides
        conf = CombinedConfig(
            deriva_ml=DerivaMLConf(
                hostname="test.example.org",
                catalog_id="42",
            ),
            execution=ExecConf(
                description="Combined test",
            ),
        )

        assert conf.deriva_ml.hostname == "test.example.org"
        assert conf.execution.description == "Combined test"

    def test_config_override_pattern(self):
        """Test the common pattern of base config with overrides."""
        DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)

        # Base config
        base = DerivaMLConf(
            hostname="base.example.org",
            catalog_id="1",
            use_minid=True,
        )

        # Override specific values using OmegaConf
        base_dict = OmegaConf.structured(base)
        overrides = OmegaConf.create({"hostname": "override.example.org"})
        merged = OmegaConf.merge(base_dict, overrides)

        assert merged.hostname == "override.example.org"
        assert merged.catalog_id == "1"  # Preserved from base
        assert merged.use_minid is True  # Preserved from base

    def test_just_for_non_instantiable_values(self):
        """Test using just() for values that shouldn't be instantiated."""
        # just() wraps a value so it's returned as-is during instantiation
        path_value = just(Path("/some/path"))

        # When used in a config, it won't try to instantiate Path
        conf = OmegaConf.create({"path": path_value})
        result = instantiate(conf)

        assert result.path == Path("/some/path")
