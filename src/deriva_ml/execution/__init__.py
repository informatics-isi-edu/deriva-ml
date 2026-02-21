from typing import TYPE_CHECKING

# Safe imports - no circular dependencies
from deriva_ml.execution.base_config import (
    BaseConfig,
    DerivaBaseConfig,
    # Config metadata helpers
    DescribedList,
    base_defaults,
    get_notebook_configuration,
    load_configs,
    # New simplified API
    notebook_config,
    run_notebook,
    with_description,
)
from deriva_ml.execution.execution_configuration import AssetRID, ExecutionConfiguration

# Re-export AssetSpec and AssetSpecConfig from their canonical location
from deriva_ml.asset.aux_classes import AssetSpec, AssetSpecConfig
from deriva_ml.execution.model_protocol import DerivaMLModel
from deriva_ml.execution.multirun_config import (
    MultirunSpec,
    get_all_multirun_configs,
    get_multirun_config,
    list_multirun_configs,
    multirun_config,
)
from deriva_ml.execution.runner import create_model_config, reset_multirun_state, run_model
from deriva_ml.execution.workflow import Workflow

if TYPE_CHECKING:
    from deriva_ml.execution.execution import Execution


# Lazy import for runtime
def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "Execution":
        from deriva_ml.execution.execution import Execution

        return Execution
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Execution",  # Lazy-loaded
    "ExecutionConfiguration",
    "Workflow",
    "AssetRID",
    "AssetSpec",
    "AssetSpecConfig",
    "run_model",
    "create_model_config",
    "reset_multirun_state",
    "DerivaMLModel",
    # Base configuration
    "BaseConfig",
    "DerivaBaseConfig",
    "base_defaults",
    "get_notebook_configuration",
    # Simplified API
    "notebook_config",
    "load_configs",
    "run_notebook",
    # Config metadata helpers
    "DescribedList",
    "with_description",
    # Multirun configuration
    "MultirunSpec",
    "multirun_config",
    "get_multirun_config",
    "list_multirun_configs",
    "get_all_multirun_configs",
]
