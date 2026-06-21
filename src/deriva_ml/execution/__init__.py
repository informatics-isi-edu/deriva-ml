"""Execution management for DerivaML.

Provides the Execution lifecycle, workflow tracking, hydra-zen
configuration helpers (BaseConfig, notebook_config, run_notebook),
and multirun support for running reproducible ML experiments with
full provenance tracking.
"""

from typing import TYPE_CHECKING

# Re-export the input-spec types from their canonical location. AssetSpec names
# a catalog-resident asset by RID; LocalFile names a local file by path (the
# framework registers it in the File table on input resolution). Both are used
# in ``ExecutionConfiguration.assets``, so both belong on the execution surface.
from deriva_ml.asset.aux_classes import (
    AssetSpec,
    AssetSpecConfig,
    LocalFile,
    LocalFileConfig,
)

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
from deriva_ml.execution.execution_configuration import ExecutionConfiguration
from deriva_ml.execution.lineage import (
    AssetSummary,
    DatasetSummary,
    ExecutionSummary,
    LineageNode,
    LineageResult,
    RootDescriptor,
    WorkflowSummary,
)
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
    "AssetSpec",
    "AssetSpecConfig",
    "LocalFile",
    "LocalFileConfig",
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
    # Lineage models (lookup_lineage)
    "AssetSummary",
    "DatasetSummary",
    "ExecutionSummary",
    "LineageNode",
    "LineageResult",
    "RootDescriptor",
    "WorkflowSummary",
]
