from typing import TYPE_CHECKING

# Safe imports - no circular dependencies
from deriva_ml.execution.execution_configuration import AssetRIDConfig, ExecutionConfiguration
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
    "AssetRIDConfig",
]
