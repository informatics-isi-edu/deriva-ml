# We will be loading get_version from setuptools_scm and it will emit a UserWarning about it being deprecated.

# IMPORTANT: Import deriva package first to prevent shadowing by local 'deriva.py' files.
# This ensures 'deriva' is cached in sys.modules before any other imports that might
# add directories containing a 'deriva.py' file to sys.path.
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

import deriva.core  # noqa: F401

# Safe imports - no circular dependencies
from deriva_ml.core.config import DerivaMLConfig
from deriva_ml.core.connection_mode import ConnectionMode
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
    DerivaMLDirtyWorkflowError,
    DerivaMLException,
    DerivaMLInvalidTerm,
    DerivaMLMaterializeLimitExceeded,
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
    elif name == "ExecutionConfiguration":
        from deriva_ml.execution.execution_configuration import ExecutionConfiguration

        return ExecutionConfiguration
    elif name == "Workflow":
        from deriva_ml.execution.workflow import Workflow

        return Workflow
    elif name == "Dataset":
        from deriva_ml.dataset.dataset import Dataset

        return Dataset
    elif name == "DatasetSpec":
        from deriva_ml.dataset.aux_classes import DatasetSpec

        return DatasetSpec
    elif name == "DatasetSpecConfig":
        from deriva_ml.dataset.aux_classes import DatasetSpecConfig

        return DatasetSpecConfig
    elif name == "Asset":
        from deriva_ml.asset.asset import Asset

        return Asset
    elif name == "AssetFilePath":
        from deriva_ml.asset.aux_classes import AssetFilePath

        return AssetFilePath
    elif name == "AssetSpec":
        from deriva_ml.asset.aux_classes import AssetSpec

        return AssetSpec
    elif name == "AssetSpecConfig":
        from deriva_ml.asset.aux_classes import AssetSpecConfig

        return AssetSpecConfig
    elif name == "FeatureRecord":
        from deriva_ml.feature import FeatureRecord

        return FeatureRecord
    elif name == "CatalogProvenance":
        from deriva_ml.catalog.provenance import CatalogProvenance

        return CatalogProvenance
    elif name == "CatalogCreationMethod":
        from deriva_ml.catalog.provenance import CatalogCreationMethod

        return CatalogCreationMethod
    elif name == "CloneDetails":
        from deriva_ml.catalog.provenance import CloneDetails

        return CloneDetails
    elif name == "get_catalog_provenance":
        from deriva_ml.catalog.provenance import get_catalog_provenance

        return get_catalog_provenance
    elif name == "set_catalog_provenance":
        from deriva_ml.catalog.provenance import set_catalog_provenance

        return set_catalog_provenance
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DerivaML",  # Lazy-loaded
    "DerivaMLConfig",
    "ConnectionMode",
    # Execution-authoring surface (lazy-loaded)
    "Execution",
    "ExecutionConfiguration",
    "Workflow",
    # Dataset classes (lazy-loaded)
    "Dataset",
    "DatasetSpec",
    "DatasetSpecConfig",
    # Asset classes (lazy-loaded)
    "Asset",
    "AssetFilePath",
    "AssetSpec",
    "AssetSpecConfig",
    # FeatureRecord — base class for dynamically-generated feature
    # rows. Used in user code for typing (FeatureRecord subclasses
    # returned by Feature.feature_record_class()) and as the type
    # for selector functions passed to feature_values() and
    # restructure_assets().
    "FeatureRecord",
    # Catalog provenance (lazy-loaded)
    "CatalogProvenance",
    "CatalogCreationMethod",
    "CloneDetails",
    "get_catalog_provenance",
    "set_catalog_provenance",
    # Exceptions
    "DerivaMLException",
    "DerivaMLInvalidTerm",
    "DerivaMLMaterializeLimitExceeded",
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
