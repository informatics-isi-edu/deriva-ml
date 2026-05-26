"""Tests for the base_config module.

Tests cover:
- BaseConfig structured config schema (no runtime fields)
- get_notebook_configuration() capturing hydra output dir
- notebook_config() registration
- Module-level hydra output dir side-channel
- _format_description_with_overrides() helper for Execution.description
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hydra_zen import builds, store

from deriva_ml.core.config import DerivaMLConfig
from deriva_ml.execution.base_config import (
    BaseConfig,
    _format_description_with_overrides,
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
            config = get_notebook_configuration(config_builds, config_name="test_ch_config")

        assert isinstance(config.config_choices, dict)
        assert config.config_choices.get("deriva_ml") == "test_ch_deriva"

    def test_hydra_output_dir_not_on_config(self):
        """The config object should NOT have hydra_runtime_output_dir."""
        config_builds = _make_test_config("test_na_deriva", "test_na_config")

        with patch("deriva_ml.core.config.HydraConfig") as mock_hydra:
            mock_hydra.get.return_value.runtime.output_dir = "/tmp/hydra_test_na"
            config = get_notebook_configuration(config_builds, config_name="test_na_config")

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


class TestFormatDescriptionWithOverrides:
    """Tests for the _format_description_with_overrides() helper.

    Regression coverage for analyst/02
    (findings/analyst/02-execution-description-stale-on-asset-override.md):
    an Execution row produced by run_notebook() must reflect the resolved
    Hydra overrides, not just the registration-time default description.
    """

    def test_no_overrides_returns_base_unchanged(self):
        """No overrides should produce the base description verbatim -- no clutter."""
        result = _format_description_with_overrides("ROC curve analysis (default: quick vs extended training)", [])
        assert result == "ROC curve analysis (default: quick vs extended training)"

    def test_none_safe(self):
        """Empty list (the no-overrides case) must not add bracket clutter."""
        # The caller normalises None -> [] before reaching the helper, but
        # the empty-list branch is the load-bearing one.
        result = _format_description_with_overrides("Base description", [])
        assert "[overrides:" not in result
        assert result == "Base description"

    def test_single_override_appears_in_description(self):
        """An assets= override should be visible in the formatted description."""
        result = _format_description_with_overrides(
            "ROC curve analysis (default: quick vs extended training)",
            ["assets=roc_all_six"],
        )
        assert result == ("ROC curve analysis (default: quick vs extended training) [overrides: assets=roc_all_six]")

    def test_multiple_overrides_joined(self):
        """Multiple overrides should appear as a comma-separated list."""
        result = _format_description_with_overrides(
            "Quick run",
            ["assets=roc_all_six", "datasets=cifar10_complete", "deriva_ml=eye_ai"],
        )
        assert result == ("Quick run [overrides: assets=roc_all_six, datasets=cifar10_complete, deriva_ml=eye_ai]")

    def test_override_order_preserved(self):
        """The order of the override list is preserved in the description."""
        result = _format_description_with_overrides("Base", ["b=2", "a=1", "c=3"])
        assert result == "Base [overrides: b=2, a=1, c=3]"

    def test_hydra_package_spec_preserved(self):
        """Hydra ``+key=value`` / ``~key`` syntax should pass through verbatim."""
        result = _format_description_with_overrides("Base", ["+extra_flag=true", "~unused_group"])
        assert result == "Base [overrides: +extra_flag=true, ~unused_group]"


class TestRunNotebookDescriptionFromResolvedConfig:
    """Tests that run_notebook() builds Execution.description from resolved Hydra config.

    Regression coverage for analyst/02. The Execution row's description must
    include any Hydra overrides the user passed on the command line (via
    ``DERIVA_ML_HYDRA_OVERRIDES``) or in the ``overrides=`` kwarg, not just
    the registration-time default carried on the config class.
    """

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """Ensure DERIVA_ML_HYDRA_OVERRIDES is not leaked from another test."""
        monkeypatch.delenv("DERIVA_ML_HYDRA_OVERRIDES", raising=False)
        yield

    def _patch_catalog_io(self):
        """Patch the bits of run_notebook() that talk to a real catalog.

        Returns the captured ExecutionConfiguration so tests can assert
        on the description that would have hit the catalog. The patches
        target the package-level re-exports (``deriva_ml.execution.*``)
        because ``run_notebook`` does its imports inside the function
        body via ``from deriva_ml.execution import ...``.
        """
        captured = {}

        def _fake_execution_configuration(*, workflow, datasets, assets, description, **kwargs):
            cfg = MagicMock()
            cfg.workflow = workflow
            cfg.datasets = datasets
            cfg.assets = assets
            cfg.description = description
            captured["exec_config"] = cfg
            return cfg

        validation_result = MagicMock()
        validation_result.is_valid = True
        validation_result.warnings = []

        fake_ml = MagicMock()
        fake_ml.create_workflow.return_value = MagicMock(name="workflow")

        fake_ml_class = MagicMock(return_value=fake_ml)

        fake_execution = MagicMock()

        patches = [
            patch(
                "deriva_ml.execution.ExecutionConfiguration",
                side_effect=_fake_execution_configuration,
            ),
            patch(
                "deriva_ml.core.validation.validate_execution_config",
                return_value=validation_result,
            ),
            patch("deriva_ml.DerivaML", fake_ml_class),
            patch("deriva_ml.execution.Execution", return_value=fake_execution),
        ]
        return captured, patches, fake_ml_class

    def _make_described_config(self, deriva_name: str, config_name: str, description: str):
        """Register a notebook config with a non-empty default description.

        We bypass ``notebook_config()`` so the test doesn't have to register
        the full assets/datasets defaults -- those groups aren't relevant to
        what we're testing (description formation), and registering them
        cleanly across tests is brittle. Instead, we build the structured
        config directly and register it in the internal registry that
        ``run_notebook`` consults.
        """
        from deriva_ml.execution.base_config import _notebook_configs

        DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)

        deriva_store = store(group="deriva_ml")
        deriva_store(
            DerivaMLConf,
            name=deriva_name,
            hostname="test.example.org",
            catalog_id="1",
        )

        config_builds = builds(
            BaseConfig,
            populate_full_signature=True,
            hydra_defaults=["_self_", {"deriva_ml": deriva_name}],
            description=description,
        )
        store(config_builds, name=config_name)
        _notebook_configs[config_name] = (config_builds, config_name)
        store.add_to_hydra_store(overwrite_ok=True)

    def test_no_overrides_writes_clean_description(self, monkeypatch):
        """With no Hydra overrides, the Execution description is the base description.

        Specifically, it must NOT carry a stray ``[overrides: ...]`` clause.
        """
        from deriva_ml.execution.base_config import run_notebook as run_nb

        self._make_described_config("rn_no_ov_deriva", "rn_no_ov_config", "ROC curve analysis (default)")

        captured, patches, _ = self._patch_catalog_io()

        with patch("deriva_ml.core.config.HydraConfig") as mock_hydra:
            mock_hydra.get.return_value.runtime.output_dir = "/tmp/hydra_rn_no_ov"
            for p in patches:
                p.start()
            try:
                run_nb("rn_no_ov_config")
            finally:
                for p in patches:
                    p.stop()

        description = captured["exec_config"].description
        assert description == "ROC curve analysis (default)"
        assert "[overrides:" not in description

    def test_explicit_overrides_appear_in_description(self):
        """``overrides=["assets=..."]`` must surface in Execution.description."""
        from deriva_ml.execution.base_config import run_notebook as run_nb

        self._make_described_config("rn_ex_ov_deriva", "rn_ex_ov_config", "ROC curve analysis")

        captured, patches, _ = self._patch_catalog_io()

        with patch("deriva_ml.core.config.HydraConfig") as mock_hydra:
            mock_hydra.get.return_value.runtime.output_dir = "/tmp/hydra_rn_ex_ov"
            for p in patches:
                p.start()
            try:
                run_nb("rn_ex_ov_config", overrides=["assets=roc_all_six"])
            finally:
                for p in patches:
                    p.stop()

        description = captured["exec_config"].description
        assert "assets=roc_all_six" in description
        assert description.startswith("ROC curve analysis")
        # The override clause must be clearly distinguished, not silently glued on.
        assert "[overrides: assets=roc_all_six]" in description

    def test_env_overrides_appear_in_description(self, monkeypatch):
        """Overrides from ``DERIVA_ML_HYDRA_OVERRIDES`` (the CLI path) must surface.

        This is the exact reproduction of analyst/02: an analyst runs
        ``deriva-ml-run-notebook ... assets=roc_all_six``, which the CLI
        marshals through the env var rather than the ``overrides=`` kwarg.
        """
        import json as _json

        from deriva_ml.execution.base_config import run_notebook as run_nb

        self._make_described_config("rn_env_ov_deriva", "rn_env_ov_config", "ROC curve analysis")

        monkeypatch.setenv("DERIVA_ML_HYDRA_OVERRIDES", _json.dumps(["assets=roc_all_six"]))

        captured, patches, _ = self._patch_catalog_io()

        with patch("deriva_ml.core.config.HydraConfig") as mock_hydra:
            mock_hydra.get.return_value.runtime.output_dir = "/tmp/hydra_rn_env_ov"
            for p in patches:
                p.start()
            try:
                run_nb("rn_env_ov_config")
            finally:
                for p in patches:
                    p.stop()

        description = captured["exec_config"].description
        assert "[overrides: assets=roc_all_six]" in description
