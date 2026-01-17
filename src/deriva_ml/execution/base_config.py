"""Base configuration for DerivaML applications.

This module defines the base configuration that both script execution and
notebooks can inherit from. It provides the common hydra defaults structure
without tying to the run_model function.

Usage:
    # In notebooks or project configs
    from deriva_ml.execution import BaseConfig, base_defaults

    @dataclass
    class MyNotebookConfig(BaseConfig):
        my_param: str = "value"

    # Build and register with hydra-zen
    MyConfig = builds(MyNotebookConfig, hydra_defaults=[...])
    store(MyConfig, name="my_config")
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, TypeVar

from hydra_zen import builds, instantiate, launch, store

T = TypeVar("T")


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
    datasets: Any = field(default_factory=list)
    assets: Any = field(default_factory=list)
    dry_run: bool = False
    description: str = ""


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

    # Define a task function that instantiates and returns the config
    # The cfg from launch() is an OmegaConf DictConfig, so we need to
    # use hydra_zen.instantiate() to convert it to actual Python objects
    def return_instantiated_config(cfg: Any) -> T:
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

    return result.return_value
