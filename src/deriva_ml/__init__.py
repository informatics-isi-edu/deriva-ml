from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

# Safe imports - no circular dependencies
from deriva_ml.core.config import DerivaMLConfig
from deriva_ml.core.definitions import (
    RID,
    BuiltinTypes,
    ColumnDefinition,
    DerivaAssetColumns,
    DerivaSystemColumns,
    ExecAssetType,
    ExecMetadataType,
    FileSpec,
    FileUploadState,
    ForeignKeyDefinition,
    KeyDefinition,
    MLAsset,
    MLVocab,
    TableDefinition,
    UploadState,
)
from deriva_ml.core.exceptions import (
    DerivaMLException,
    DerivaMLInvalidTerm,
    DerivaMLTableTypeError,
)
from deriva_ml.dataset.aux_classes import DatasetConfig, DatasetConfigList, DatasetSpec, DatasetVersion

from .execution import Execution, ExecutionConfiguration

# Type-checking only - avoid circular import at runtime
if TYPE_CHECKING:
    from deriva_ml.core.base import DerivaML


# Lazy import function for runtime usage
def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "DerivaML":
        from deriva_ml.core.base import DerivaML

        return DerivaML
    elif name == "Execution":
        from deriva_ml.execution.execution import Execution

        return Execution
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DerivaML",  # Lazy-loaded
    "DerivaMLConfig",
    "DatasetConfig",
    "DatasetConfigList",
    "DatasetSpec",
    "DatasetVersion",
    "Execution",
    "ExecutionConfiguration",
    # Exceptions
    "DerivaMLException",
    "DerivaMLInvalidTerm",
    "DerivaMLTableTypeError",
    # Definitions
    "RID",
    "BuiltinTypes",
    "ColumnDefinition",
    "DerivaSystemColumns",
    "DerivaAssetColumns",
    "ExecAssetType",
    "ExecMetadataType",
    "FileSpec",
    "FileUploadState",
    "ForeignKeyDefinition",
    "KeyDefinition",
    "MLAsset",
    "MLVocab",
    "TableDefinition",
    "UploadState",
]

try:
    __version__ = version("deriva_ml")
except PackageNotFoundError:
    # package is not installed
    pass
