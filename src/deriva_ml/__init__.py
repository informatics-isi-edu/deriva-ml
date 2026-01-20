# We will be loading get_version from setuptools_scm and it will emit a UserWarning about it being deprecated.

# IMPORTANT: Import deriva package first to prevent shadowing by local 'deriva.py' files.
# This ensures 'deriva' is cached in sys.modules before any other imports that might
# add directories containing a 'deriva.py' file to sys.path.
import deriva.core  # noqa: F401

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
    UploadCallback,
    UploadProgress,
    UploadState,
)
from deriva_ml.core.exceptions import (
    DerivaMLException,
    DerivaMLInvalidTerm,
    DerivaMLTableTypeError,
)

# Type-checking only - avoid circular import at runtime
if TYPE_CHECKING:
    from deriva_ml.core.base import DerivaML


# Lazy import function for runtime usage
def __getattr__(name: str) -> type:
    """Lazy import to avoid circular dependencies."""
    if name == "DerivaML":
        from deriva_ml.core.base import DerivaML

        return DerivaML
    elif name == "Execution":
        from deriva_ml.execution.execution import Execution

        return Execution
    elif name == "Asset":
        from deriva_ml.asset.asset import Asset

        return Asset
    elif name == "AssetFilePath":
        from deriva_ml.asset.aux_classes import AssetFilePath

        return AssetFilePath
    elif name == "AssetSpec":
        from deriva_ml.asset.aux_classes import AssetSpec

        return AssetSpec
    elif name == "FeatureValueRecord":
        from deriva_ml.dataset.dataset_bag import FeatureValueRecord

        return FeatureValueRecord
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DerivaML",  # Lazy-loaded
    "DerivaMLConfig",
    # Asset classes (lazy-loaded)
    "Asset",
    "AssetFilePath",
    "AssetSpec",
    # Feature value record for restructure_assets
    "FeatureValueRecord",
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
    "UploadCallback",
    "UploadProgress",
    "UploadState",
]

try:
    __version__ = version("deriva_ml")
except PackageNotFoundError:
    # package is not installed
    pass
