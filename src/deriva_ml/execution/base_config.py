"""Base configuration for DerivaML applications.

This module defines the base configuration and helper functions that simplify
creating hydra-zen configurations for both script execution and notebooks.

Simple Usage (notebooks using only BaseConfig fields):
    # In configs/my_notebook.py
    from deriva_ml.execution import notebook_config

    notebook_config(
        "my_notebook",
        defaults={"assets": "my_assets", "datasets": "my_dataset"},
    )

    # In notebook
    from deriva_ml.execution import run_notebook
    ml, execution, config = run_notebook("my_notebook")

Advanced Usage (notebooks with custom parameters):
    # In configs/my_analysis.py
    from dataclasses import dataclass
    from deriva_ml.execution import BaseConfig, notebook_config

    @dataclass
    class MyAnalysisConfig(BaseConfig):
        threshold: float = 0.5
        num_samples: int = 100

    notebook_config(
        "my_analysis",
        config_class=MyAnalysisConfig,
        defaults={"assets": "analysis_assets"},
    )

    # In notebook
    from deriva_ml.execution import run_notebook
    ml, execution, config = run_notebook("my_analysis")
    print(config.threshold)  # 0.5
"""

import importlib
import json
import os
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from hydra_zen import builds, instantiate, launch, store

if TYPE_CHECKING:
    from deriva_ml import DerivaML
    from deriva_ml.execution import Execution

T = TypeVar("T")

# Module-level storage for hydra runtime output dir captured during
# get_notebook_configuration(). This is a side-channel because the value
# is a runtime artifact, not a configuration parameter, and should NOT
# be part of the Hydra structured config schema.
_captured_hydra_output_dir: str | None = None

# Standard hydra defaults for DerivaML applications.
# Projects can customize these or define their own defaults.
base_defaults = [
    "_self_",
    {"deriva_ml": "default_deriva"},
    {"datasets": "default_dataset"},
    {"assets": "default_asset"},
    {"workflow": "default_workflow"},
    {"model_config": "default_model"},
]


@dataclass
class BaseConfig:
    """Base configuration for DerivaML applications.

    This dataclass defines the common configuration structure shared by
    both script execution and notebook modes. Project-specific configs
    should inherit from this class to get the standard DerivaML fields.

    Note:
        Fields use ``Any`` type annotations because several DerivaML types
        (DerivaMLConfig, DatasetSpec) are Pydantic models which are not
        compatible with OmegaConf structured configs. The actual types at
        runtime are documented below.

    Attributes:
        deriva_ml: DerivaML connection configuration (DerivaMLConfig at runtime).
        datasets: List of dataset specifications (list[DatasetSpec] at runtime).
        assets: List of asset RIDs to load (list[str] at runtime).
        dry_run: If True, skip catalog writes (for testing/debugging).
        description: Human-readable description of this run.
        config_choices: Dictionary mapping config group names to selected config names.
            This is automatically populated by get_notebook_configuration() with the
            Hydra runtime choices (e.g., {"model_config": "cifar10_quick", "assets": "roc_quick"}).
            Useful for tracking which configurations were used in an execution.

    Example:
        >>> from dataclasses import dataclass
        >>> from deriva_ml.execution import BaseConfig
        >>>
        >>> @dataclass
        ... class MyConfig(BaseConfig):
        ...     learning_rate: float = 0.001
        ...     epochs: int = 10
    """
    deriva_ml: Any = None
    datasets: Any = None
    assets: Any = None
    dry_run: bool = False
    description: str = ""
    config_choices: dict[str, str] = field(default_factory=dict)


# Create and register the base config with hydra-zen store.
# This provides a ready-to-use base that experiments can inherit from.
DerivaBaseConfig = builds(
    BaseConfig,
    populate_full_signature=True,
    hydra_defaults=base_defaults,
)

store(DerivaBaseConfig, name="deriva_base")


def get_notebook_configuration(
    config_class: type[T],
    config_name: str,
    overrides: list[str] | None = None,
    job_name: str = "notebook",
    version_base: str = "1.3",
) -> T:
    """Load and return a hydra-zen configuration for use in notebooks.

    This function is the notebook equivalent of `run_model`. While `run_model`
    launches a full execution with model training, `get_notebook_configuration`
    simply resolves the configuration and returns it for interactive use.

    The function handles:
    - Adding configurations to the hydra store
    - Launching hydra-zen to resolve defaults and overrides
    - Returning the instantiated configuration object

    Args:
        config_class: The hydra-zen builds() class for the configuration.
            This should be a class created with `builds(YourConfig, ...)`.
        config_name: Name of the configuration in the hydra store.
            Must match the name used when calling `store(config_class, name=...)`.
        overrides: Optional list of Hydra override strings (e.g., ["param=value"]).
        job_name: Name for the Hydra job (default: "notebook").
        version_base: Hydra version base (default: "1.3").

    Returns:
        The instantiated configuration object with all defaults resolved.

    Example:
        In your notebook's configuration module (e.g., `configs/roc_analysis.py`):

        >>> from dataclasses import dataclass, field
        >>> from hydra_zen import builds, store
        >>> from deriva_ml.execution import BaseConfig
        >>>
        >>> @dataclass
        ... class ROCAnalysisConfig(BaseConfig):
        ...     execution_rids: list[str] = field(default_factory=list)
        >>>
        >>> ROCAnalysisConfigBuilds = builds(
        ...     ROCAnalysisConfig,
        ...     populate_full_signature=True,
        ...     hydra_defaults=["_self_", {"deriva_ml": "default_deriva"}],
        ... )
        >>> store(ROCAnalysisConfigBuilds, name="roc_analysis")

        In your notebook:

        >>> from configs import load_all_configs
        >>> from configs.roc_analysis import ROCAnalysisConfigBuilds
        >>> from deriva_ml.execution import get_notebook_configuration
        >>>
        >>> # Load all project configs into hydra store
        >>> load_all_configs()
        >>>
        >>> # Get resolved configuration
        >>> config = get_notebook_configuration(
        ...     ROCAnalysisConfigBuilds,
        ...     config_name="roc_analysis",
        ...     overrides=["execution_rids=[3JRC,3KT0]"],
        ... )
        >>>
        >>> # Use the configuration
        >>> print(config.execution_rids)  # ['3JRC', '3KT0']
        >>> print(config.deriva_ml.hostname)  # From default_deriva config

    Environment Variables:
        DERIVA_ML_HYDRA_OVERRIDES: JSON-encoded list of override strings.
            When running via `deriva-ml-run-notebook`, this is automatically
            set from command-line arguments. Overrides from this environment
            variable are applied first, then any overrides passed directly
            to this function are applied (taking precedence).
    """
    # Ensure configs are in the hydra store
    store.add_to_hydra_store(overwrite_ok=True)

    # Collect overrides from environment variable (set by run_notebook CLI)
    env_overrides_json = os.environ.get("DERIVA_ML_HYDRA_OVERRIDES")
    env_overrides = json.loads(env_overrides_json) if env_overrides_json else []

    # Merge overrides: env overrides first, then explicit overrides (higher precedence)
    all_overrides = env_overrides + (overrides or [])

    # Variables to capture from within the task function
    captured_choices: dict[str, str] = {}
    captured_output_dir: str | None = None

    # Define a task function that instantiates and returns the config
    # The cfg from launch() is an OmegaConf DictConfig, so we need to
    # use hydra_zen.instantiate() to convert it to actual Python objects
    def return_instantiated_config(cfg: Any) -> T:
        nonlocal captured_choices, captured_output_dir
        # Capture the Hydra runtime choices (which config names were selected)
        # and runtime output directory (for uploading hydra config files)
        # Filter out None values (some Hydra internal groups have None choices)
        try:
            from hydra.core.hydra_config import HydraConfig
            hydra_cfg = HydraConfig.get()
            choices = hydra_cfg.runtime.choices
            captured_choices = {k: v for k, v in choices.items() if v is not None}
            captured_output_dir = hydra_cfg.runtime.output_dir
        except Exception:
            # If HydraConfig is not available, leave choices empty
            pass
        return instantiate(cfg)

    # Launch hydra-zen to resolve the configuration
    result = launch(
        config_class,
        return_instantiated_config,
        version_base=version_base,
        config_name=config_name,
        job_name=job_name,
        overrides=all_overrides,
    )

    # Inject the captured choices into the config object
    config = result.return_value
    if hasattr(config, "config_choices"):
        config.config_choices = captured_choices

    # Store the hydra output dir in module-level variable for run_notebook() to use.
    # This is NOT stored on the config because it's a runtime artifact, not a
    # configuration parameter, and adding it to the structured config causes
    # Hydra OmegaConf composition errors.
    global _captured_hydra_output_dir
    _captured_hydra_output_dir = captured_output_dir

    return config


# ---------------------------------------------------------------------------
# Registry for notebook configurations
# ---------------------------------------------------------------------------
# Maps config_name -> (config_builds_class, config_name)
_notebook_configs: dict[str, tuple[Any, str]] = {}


def notebook_config(
    name: str,
    config_class: type[BaseConfig] | None = None,
    defaults: dict[str, str] | None = None,
    **field_defaults: Any,
) -> Any:
    """Register a notebook configuration with simplified syntax.

    This is the recommended way to create notebook configurations. It handles
    all the hydra-zen boilerplate (builds, store, defaults) automatically.

    For simple notebooks that only use BaseConfig fields (deriva_ml, datasets,
    assets, etc.), just specify which defaults to use. For notebooks with
    custom parameters, provide a config_class that inherits from BaseConfig.

    Args:
        name: Configuration name. Used both as the hydra config name and
            to look up the config in run_notebook().
        config_class: Optional dataclass inheriting from BaseConfig. If None,
            uses BaseConfig directly (suitable for notebooks that only need
            the standard fields).
        defaults: Dict mapping config group names to config names. These
            override the base defaults. Common groups:
            - "deriva_ml": Connection config (e.g., "default_deriva", "eye_ai")
            - "datasets": Dataset config (e.g., "cifar10_training")
            - "assets": Asset config (e.g., "model_weights")
            - "workflow": Workflow config (e.g., "default_workflow")
        **field_defaults: Default values for fields in config_class.

    Returns:
        The hydra-zen builds() class, in case you need to reference it directly.

    Examples:
        Simple notebook using only standard fields:

            # configs/roc_analysis.py
            from deriva_ml.execution import notebook_config

            notebook_config(
                "roc_analysis",
                defaults={"assets": "roc_comparison_probabilities"},
            )

        Notebook with custom parameters:

            # configs/training_analysis.py
            from dataclasses import dataclass
            from deriva_ml.execution import BaseConfig, notebook_config

            @dataclass
            class TrainingAnalysisConfig(BaseConfig):
                learning_rate: float = 0.001
                batch_size: int = 32

            notebook_config(
                "training_analysis",
                config_class=TrainingAnalysisConfig,
                defaults={"datasets": "cifar10_training"},
                learning_rate=0.01,  # Override default
            )
    """
    # Use BaseConfig if no custom class provided
    actual_class = config_class or BaseConfig

    # Build the hydra defaults list
    hydra_defaults = ["_self_"]

    # Start with base defaults, then apply overrides
    default_groups = {
        "deriva_ml": "default_deriva",
        "datasets": "default_dataset",
        "assets": "default_asset",
    }
    if defaults:
        default_groups.update(defaults)

    for group, config_name in default_groups.items():
        hydra_defaults.append({group: config_name})

    # Create the hydra-zen builds() class
    config_builds = builds(
        actual_class,
        populate_full_signature=True,
        hydra_defaults=hydra_defaults,
        **field_defaults,
    )

    # Register with hydra-zen store
    store(config_builds, name=name)

    # Also register in our internal registry for run_notebook()
    _notebook_configs[name] = (config_builds, name)

    return config_builds


def load_configs(package_name: str = "configs") -> list[str]:
    """Dynamically import all configuration modules from a package.

    This function discovers and imports all Python modules in the specified
    package. Each module is expected to register its configurations with
    the hydra-zen store as a side effect of being imported.

    Args:
        package_name: Name of the package containing config modules.
            Default is "configs" which works for the standard project layout.

    Returns:
        List of module names that were successfully loaded.

    Raises:
        ImportError: If a config module fails to import.

    Example:
        # In your main script or notebook
        from deriva_ml.execution import load_configs

        load_configs()  # Loads from "configs" package
        # or
        load_configs("my_project.configs")  # Custom package

    Note:
        The "experiments" module (if present) is loaded last because it
        typically depends on other configs being registered first.
    """
    loaded_modules = []

    try:
        package = importlib.import_module(package_name)
    except ImportError:
        # Package doesn't exist, return empty
        return []

    package_dir = Path(package.__file__).parent

    # Collect module names, recursing into subpackages
    modules_to_load = []
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.ispkg:
            # Recurse into subpackages (e.g., configs/dev/)
            sub_loaded = load_configs(f"{package_name}.{module_info.name}")
            loaded_modules.extend(sub_loaded)
        else:
            modules_to_load.append(module_info.name)

    # Sort modules but ensure 'experiments' is loaded last
    modules_to_load.sort()
    if "experiments" in modules_to_load:
        modules_to_load.remove("experiments")
        modules_to_load.append("experiments")

    for module_name in modules_to_load:
        full_name = f"{package_name}.{module_name}"
        importlib.import_module(full_name)
        loaded_modules.append(full_name)

    return sorted(loaded_modules)


def run_notebook(
    config_name: str,
    overrides: list[str] | None = None,
    workflow_name: str | None = None,
    workflow_type: str = "Analysis Notebook",
    ml_class: type["DerivaML"] | None = None,
    config_package: str = "configs",
) -> tuple["DerivaML", "Execution", BaseConfig]:
    """Initialize a notebook with DerivaML execution context.

    This is the main entry point for notebooks. It handles all the setup:
    1. Loads all config modules from the config package
    2. Resolves the hydra-zen configuration
    3. Creates the DerivaML connection
    4. Creates a workflow and execution context
    5. Downloads any specified datasets and assets

    Args:
        config_name: Name of the notebook configuration (registered via
            notebook_config() or store()).
        overrides: Optional list of Hydra override strings
            (e.g., ["assets=different_assets"]).
        workflow_name: Name for the workflow. Defaults to config_name.
        workflow_type: Type of workflow (default: "Analysis Notebook").
        ml_class: Optional DerivaML subclass to use. If None, uses DerivaML.
        config_package: Package containing config modules (default: "configs").

    Returns:
        Tuple of (ml_instance, execution, config):
        - ml_instance: Connected DerivaML (or subclass) instance
        - execution: Execution context with downloaded inputs
        - config: Resolved configuration object

    Example:
        # Simple usage
        from deriva_ml.execution import run_notebook

        ml, execution, config = run_notebook("roc_analysis")

        # Access config values
        print(config.assets)
        print(config.deriva_ml.hostname)

        # Use ml and execution
        for asset_table, paths in execution.asset_paths.items():
            for path in paths:
                print(f"Downloaded: {path.file_name}")

        # At the end of notebook
        execution.upload_execution_outputs()

    Example with overrides:
        ml, execution, config = run_notebook(
            "roc_analysis",
            overrides=["assets=roc_quick_probabilities"],
        )

    Example with custom ML class:
        from eye_ai import EyeAI

        ml, execution, config = run_notebook(
            "eye_analysis",
            ml_class=EyeAI,
        )
    """
    # Import here to avoid circular imports
    from deriva_ml import DerivaML
    from deriva_ml.execution import Execution, ExecutionConfiguration

    # Load all config modules
    load_configs(config_package)

    # Get the config builds class from our registry or try the store
    if config_name in _notebook_configs:
        config_builds, _ = _notebook_configs[config_name]
    else:
        # Fall back to looking up in hydra store by building a simple config
        # This handles configs registered the old way
        config_builds = DerivaBaseConfig

    # Resolve the configuration
    config = get_notebook_configuration(
        config_builds,
        config_name=config_name,
        overrides=overrides,
    )

    # Create DerivaML instance, passing the hydra output dir captured during
    # config resolution so that hydra YAML configs get uploaded with the execution.
    actual_ml_class = ml_class or DerivaML
    hydra_output_dir = Path(_captured_hydra_output_dir) if _captured_hydra_output_dir else None
    ml = actual_ml_class(
        hostname=config.deriva_ml.hostname,
        catalog_id=config.deriva_ml.catalog_id,
        hydra_runtime_output_dir=hydra_output_dir,
    )

    # Create workflow
    actual_workflow_name = workflow_name or config_name.replace("_", " ").title()
    workflow = ml.create_workflow(
        name=actual_workflow_name,
        workflow_type=workflow_type,
        description=config.description or f"Running {config_name}",
    )

    # Create execution configuration
    exec_config = ExecutionConfiguration(
        workflow=workflow,
        datasets=config.datasets if config.datasets else [],
        assets=config.assets if config.assets else [],
        description=config.description or f"Execution of {config_name}",
    )

    # Create execution context (downloads inputs)
    execution = Execution(configuration=exec_config, ml_object=ml, dry_run=config.dry_run)

    return ml, execution, config


class DescribedList(list):
    """A list with an attached description.

    This class extends list to add a `description` attribute while maintaining
    full list compatibility. This allows configuration values (like asset RIDs
    or dataset specs) to carry documentation without changing how they're used.

    When stored in hydra-zen and resolved via `instantiate()`, the result is a
    DescribedList that behaves like a regular list but has a `description` attribute.

    Attributes:
        description: Human-readable description of this configuration.

    Example:
        >>> from hydra_zen import store
        >>> from deriva_ml.execution import with_description
        >>>
        >>> asset_store = store(group="assets")
        >>> asset_store(
        ...     with_description(
        ...         ["3WMG", "3XPA"],
        ...         "Model weights from quick and extended training",
        ...     ),
        ...     name="comparison_weights",
        ... )
        >>>
        >>> # After instantiation, usage is identical to a regular list:
        >>> # config.assets[0]  # "3WMG"
        >>> # len(config.assets)  # 2
        >>> # for rid in config.assets: ...
        >>> # config.assets.description  # "Model weights from..."
    """

    def __init__(self, items: list | None = None, description: str = ""):
        """Initialize a DescribedList.

        Args:
            items: Initial list items. If None, creates empty list.
            description: Human-readable description of this list.
        """
        super().__init__(items or [])
        self.description = description

    def __repr__(self) -> str:
        """Return string representation including description."""
        if self.description:
            return f"DescribedList({list(self)!r}, description={self.description!r})"
        return f"DescribedList({list(self)!r})"


def _make_described_list(items: list, description: str = "") -> DescribedList:
    """Factory function for creating DescribedList instances.

    This is used internally by `with_description` to create a hydra-zen
    compatible config.
    """
    return DescribedList(items, description)


# Pre-built config for DescribedList
_DescribedListConfig = builds(_make_described_list, populate_full_signature=True)


def with_description(items: list, description: str) -> Any:
    """Create a hydra-zen config for a list with an attached description.

    Use this to add descriptions to configuration values like asset RIDs
    or dataset specifications. The result is a hydra-zen config that, when
    instantiated, produces a DescribedList.

    Args:
        items: List items (e.g., asset RIDs, dataset specs).
        description: Human-readable description of this configuration.

    Returns:
        A hydra-zen config that instantiates to a DescribedList.

    Example:
        >>> from hydra_zen import store
        >>> from deriva_ml.execution import with_description
        >>>
        >>> # Assets with description
        >>> asset_store = store(group="assets")
        >>> asset_store(
        ...     with_description(
        ...         ["3WMG", "3XPA"],
        ...         "Model weights from quick and extended training runs",
        ...     ),
        ...     name="comparison_weights",
        ... )
        >>>
        >>> # Datasets with description
        >>> from deriva_ml.dataset import DatasetSpecConfig
        >>> datasets_store = store(group="datasets")
        >>> datasets_store(
        ...     with_description(
        ...         [DatasetSpecConfig(rid="28CT", version="0.21.0")],
        ...         "Complete CIFAR-10 dataset with 10,000 images",
        ...     ),
        ...     name="cifar10_complete",
        ... )
        >>>
        >>> # After instantiation:
        >>> # config.assets is a DescribedList
        >>> # config.assets[0]  # "3WMG"
        >>> # config.assets.description  # "Model weights from..."

    Note:
        For model configs created with `builds()`, use the `zen_meta` parameter
        instead:

        >>> model_store(
        ...     Cifar10CNNConfig,
        ...     name="cifar10_quick",
        ...     epochs=3,
        ...     zen_meta={"description": "Quick training - 3 epochs"},
        ... )
    """
    return _DescribedListConfig(items=items, description=description)
