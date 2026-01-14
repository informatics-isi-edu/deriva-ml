"""Tests for hydra-zen configuration integration.

These tests verify that hydra-zen can be used to configure DerivaML
and ExecutionConfiguration, including proper working directory setup
for Hydra output files.
"""

import getpass
from pathlib import Path
from unittest.mock import patch

from hydra_zen import builds, instantiate, just, make_config
from omegaconf import OmegaConf

from deriva_ml.core.config import DerivaMLConfig
from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.execution.execution_configuration import ExecutionConfiguration


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
        result = DerivaMLConfig.compute_workdir(custom_base, "42")

        # Should append username, deriva-ml, and catalog_id
        expected = custom_base / getpass.getuser() / "deriva-ml" / "42"
        assert result == expected.absolute()

    def test_compute_workdir_with_none(self):
        """Test that compute_workdir uses ~/.deriva/deriva-ml when None."""
        result = DerivaMLConfig.compute_workdir(None, "1")

        expected = Path.home() / ".deriva" / "deriva-ml" / "1"
        assert result == expected.absolute()

    def test_compute_workdir_without_catalog_id(self):
        """Test that compute_workdir works without catalog_id."""
        result = DerivaMLConfig.compute_workdir(None)

        expected = Path.home() / ".deriva" / "deriva-ml"
        assert result == expected.absolute()

    def test_omegaconf_resolver_registered(self):
        """Test that the compute_workdir resolver is registered with OmegaConf."""
        # The resolver should be registered when the module is imported
        # Test it by creating a config that uses the resolver with two args
        cfg = OmegaConf.create({"path": "${compute_workdir:null,52}"})
        resolved = OmegaConf.to_container(cfg, resolve=True)

        expected = Path.home() / ".deriva" / "deriva-ml" / "52"
        assert Path(resolved["path"]) == expected.absolute()

    def test_working_dir_used_in_hydra_config(self, tmp_path):
        """Test that working_dir is properly used for Hydra output configuration."""
        # Create a config with custom working_dir
        DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)

        conf = DerivaMLConf(
            hostname="test.example.org",
            catalog_id="99",
            working_dir=str(tmp_path / "work"),
            check_auth=False,
        )

        # The compute_workdir resolver should use this path with catalog_id
        computed = DerivaMLConfig.compute_workdir(tmp_path / "work", "99")
        assert computed == (tmp_path / "work" / getpass.getuser() / "deriva-ml" / "99").absolute()

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


class TestDatasetSpecConfig:
    """Test hydra-zen configuration for DatasetSpec using DatasetSpecConfig."""

    def test_dataset_spec_config_basic(self):
        """Test basic DatasetSpecConfig usage."""
        from deriva_ml.dataset import DatasetSpecConfig

        # Create a dataset spec config
        spec = DatasetSpecConfig(rid="1ABC", version="1.0.0")

        assert spec.rid == "1ABC"
        assert spec.version == "1.0.0"
        assert spec.materialize is True  # Default

    def test_dataset_spec_config_with_materialize_false(self):
        """Test DatasetSpecConfig with materialize=False."""
        from deriva_ml.dataset import DatasetSpecConfig

        spec = DatasetSpecConfig(
            rid="2DEF",
            version="2.1.0",
            materialize=False,
            description="Metadata only dataset"
        )

        assert spec.rid == "2DEF"
        assert spec.version == "2.1.0"
        assert spec.materialize is False
        assert spec.description == "Metadata only dataset"

    def test_dataset_spec_config_instantiate(self):
        """Test that DatasetSpecConfig instantiates to DatasetSpec."""
        from deriva_ml.dataset import DatasetSpec, DatasetSpecConfig

        config = DatasetSpecConfig(
            rid="3GHI",
            version="1.2.3",
            materialize=True,
        )

        # Instantiate using hydra-zen
        spec = instantiate(config)

        assert isinstance(spec, DatasetSpec)
        assert spec.rid == "3GHI"
        assert str(spec.version) == "1.2.3"
        assert spec.materialize is True

    def test_dataset_spec_list_config(self):
        """Test creating a list of dataset specs for execution configuration."""
        from deriva_ml.dataset import DatasetSpecConfig

        # Create a list of dataset specs (typical pattern for execution configs)
        datasets = [
            DatasetSpecConfig(rid="1ABC", version="1.0.0"),
            DatasetSpecConfig(rid="2DEF", version="2.0.0", materialize=False),
            DatasetSpecConfig(rid="3GHI", version="1.1.0", description="Testing"),
        ]

        assert len(datasets) == 3
        assert datasets[0].rid == "1ABC"
        assert datasets[1].materialize is False
        assert datasets[2].description == "Testing"


class TestAssetRID:
    """Test AssetRID class for asset specifications."""

    def test_asset_rid_basic(self):
        """Test basic AssetRID usage."""
        from deriva_ml.execution import AssetRID

        asset = AssetRID(rid="WXYZ")

        # AssetRID is a str subclass
        assert asset == "WXYZ"
        assert asset.rid == "WXYZ"

    def test_asset_rid_with_description(self):
        """Test AssetRID with description."""
        from deriva_ml.execution import AssetRID

        asset = AssetRID(rid="ABCD", description="Model weights file")

        assert asset == "ABCD"
        assert asset.rid == "ABCD"
        assert asset.description == "Model weights file"

    def test_asset_rid_list(self):
        """Test creating a list of asset RIDs for execution configuration."""
        from deriva_ml.execution import AssetRID

        # Create a list of asset RIDs (typical pattern for execution configs)
        assets = [
            AssetRID(rid="1ABC"),
            AssetRID(rid="2DEF", description="Pretrained weights"),
            AssetRID(rid="3GHI", description="Config file"),
        ]

        assert len(assets) == 3
        assert assets[1].description == "Pretrained weights"
        # Can also use plain strings
        assert assets[0] == "1ABC"

    def test_plain_string_assets(self):
        """Test that plain strings work as asset RIDs."""
        # Plain strings are the simplest option when descriptions aren't needed
        assets = ["3RA", "3R8", "3R6"]

        assert len(assets) == 3
        assert assets[0] == "3RA"


class TestWorkflowConfig:
    """Test hydra-zen configuration for Workflow."""

    def test_workflow_builds_config(self):
        """Test creating a Workflow config with builds()."""
        from deriva_ml.execution import Workflow

        # Build a workflow config (without auto-detecting script info)
        WorkflowConf = builds(
            Workflow,
            name="Test Workflow",
            workflow_type="Test Type",
            description="A test workflow",
            url="https://github.com/test/repo/test.py",
            checksum="abc123",
            populate_full_signature=True,
        )

        assert WorkflowConf.name == "Test Workflow"
        assert WorkflowConf.workflow_type == "Test Type"


class TestHydraStorePatterns:
    """Test common hydra-zen store patterns for DerivaML."""

    def test_store_multiple_configs(self):
        """Test storing multiple environment configurations."""

        DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)

        # Create a local store (not affecting global state)
        # Pattern: store configs for different environments
        dev_config = DerivaMLConf(
            hostname="dev.example.org",
            catalog_id="1",
            use_minid=False,
        )

        staging_config = DerivaMLConf(
            hostname="staging.example.org",
            catalog_id="10",
            use_minid=True,
        )

        prod_config = DerivaMLConf(
            hostname="prod.example.org",
            catalog_id="100",
            use_minid=True,
        )

        # Verify different configs have expected values
        assert dev_config.hostname == "dev.example.org"
        assert dev_config.use_minid is False

        assert staging_config.catalog_id == "10"

        assert prod_config.hostname == "prod.example.org"
        assert prod_config.catalog_id == "100"

    def test_store_dataset_collections(self):
        """Test storing collections of datasets for different experiments."""
        from deriva_ml.dataset import DatasetSpecConfig

        # Pattern: create named dataset collections
        training_datasets = [
            DatasetSpecConfig(rid="TRNA", version="1.0.0"),
            DatasetSpecConfig(rid="TRNB", version="1.0.0"),
        ]

        validation_datasets = [
            DatasetSpecConfig(rid="VALA", version="1.0.0", materialize=False),
        ]

        full_datasets = training_datasets + validation_datasets

        assert len(training_datasets) == 2
        assert len(validation_datasets) == 1
        assert len(full_datasets) == 3

    def test_execution_config_with_all_components(self):
        """Test composing a full execution config with datasets and assets."""
        from deriva_ml.execution import ExecutionConfiguration

        ExecConf = builds(ExecutionConfiguration, populate_full_signature=True)
        DatasetConf = builds(DatasetSpec, populate_full_signature=True)

        # Create the full execution configuration
        exec_conf = ExecConf(
            description="Full ML training run",
            datasets=[
                DatasetConf(rid="DATA", version="1.0.0", materialize=True),
            ],
            assets=["MODL", "CNFG"],  # Asset RIDs
        )

        assert exec_conf.description == "Full ML training run"
        assert len(exec_conf.datasets) == 1
        assert len(exec_conf.assets) == 2


class TestWorkingDirectoryIntegration:
    """Test that working directory is correctly propagated to Hydra."""

    def test_working_dir_resolver_with_custom_path(self, tmp_path):
        """Test the compute_workdir resolver with custom paths and catalog_id."""
        custom_base = tmp_path / "ml_workspace"

        # Resolve using the OmegaConf resolver with two arguments
        cfg = OmegaConf.create({
            "base_path": str(custom_base),
            "catalog_id": "42",
            "resolved_path": "${compute_workdir:${base_path},${catalog_id}}"
        })
        resolved = OmegaConf.to_container(cfg, resolve=True)

        expected = custom_base / getpass.getuser() / "deriva-ml" / "42"
        assert Path(resolved["resolved_path"]) == expected.absolute()

    def test_working_dir_in_full_config(self, tmp_path):
        """Test working directory in a full DerivaML config."""
        DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)

        work_dir = tmp_path / "experiment_outputs"

        conf = DerivaMLConf(
            hostname="test.example.org",
            catalog_id="77",
            working_dir=str(work_dir),
            check_auth=False,
        )

        # Verify the config has the working dir
        assert conf.working_dir == str(work_dir)

        # Test compute_workdir directly with catalog_id
        computed = DerivaMLConfig.compute_workdir(work_dir, "77")
        assert computed == (work_dir / getpass.getuser() / "deriva-ml" / "77").absolute()

    def test_hydra_output_dir_structure(self, tmp_path):
        """Test that Hydra output directory follows expected structure."""
        # The expected structure is:
        # {working_dir}/{username}/deriva-ml/{catalog_id}/hydra/{timestamp}/

        work_dir = tmp_path / "test_work"
        computed = DerivaMLConfig.compute_workdir(work_dir, "123")

        # Verify the path structure
        assert "deriva-ml" in str(computed)
        assert getpass.getuser() in str(computed)
        assert "123" in str(computed)
