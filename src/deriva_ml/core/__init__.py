"""Core module for DerivaML.

This module provides the primary public interface to DerivaML functionality. It exports
the main DerivaML class along with configuration, definitions, and exceptions needed
for interacting with Deriva-based ML catalogs.

Key exports:
    - DerivaML: Main class for catalog operations and ML workflow management.
    - DerivaMLConfig: Configuration class for DerivaML instances.
    - Exceptions: DerivaMLException and specialized exception types.
    - Definitions: Type definitions, enums, and constants used throughout the package.

Example:
    >>> from deriva_ml.core import DerivaML, DerivaMLConfig
    >>> ml = DerivaML('deriva.example.org', 'my_catalog')
    >>> datasets = ml.find_datasets()
"""

from deriva_ml.core.base import DerivaML
from deriva_ml.core.config import DerivaMLConfig
from deriva_ml.core.definitions import (
    RID,
    BuiltinTypes,
    ColumnDefinition,
    DerivaSystemColumns,
    ExecAssetType,
    ExecMetadataType,
    FileSpec,
    FileUploadState,
    MLAsset,
    MLVocab,
    TableDefinition,
    UploadState,
)
from deriva_ml.core.exceptions import DerivaMLException, DerivaMLInvalidTerm, DerivaMLTableTypeError
from deriva_ml.core.logging_config import LoggerMixin, configure_logging, get_logger, is_hydra_initialized
from deriva_ml.core.validation import DERIVA_ML_CONFIG, STRICT_VALIDATION_CONFIG, VALIDATION_CONFIG

__all__ = [
    "DerivaML",
    "DerivaMLConfig",
    # Exceptions
    "DerivaMLException",
    "DerivaMLInvalidTerm",
    "DerivaMLTableTypeError",
    # Definitions
    "RID",
    "BuiltinTypes",
    "ColumnDefinition",
    "DerivaSystemColumns",
    "ExecAssetType",
    "ExecMetadataType",
    "FileSpec",
    "FileUploadState",
    "MLAsset",
    "MLVocab",
    "TableDefinition",
    "UploadState",
    # Validation
    "VALIDATION_CONFIG",
    "DERIVA_ML_CONFIG",
    "STRICT_VALIDATION_CONFIG",
    # Logging
    "get_logger",
    "configure_logging",
    "is_hydra_initialized",
    "LoggerMixin",
]
