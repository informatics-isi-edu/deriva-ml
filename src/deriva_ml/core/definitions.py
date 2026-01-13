"""Shared definitions for DerivaML modules.

This module serves as the central location for type definitions, constants, enums,
and data models used throughout DerivaML. It re-exports symbols from specialized
submodules for convenience and backwards compatibility.

The module consolidates:
    - Constants: Schema names, RID patterns, column definitions
    - Enums: Status codes, upload states, built-in types, vocabulary identifiers
    - Models: Pydantic models for ERMrest structures (tables, columns, keys)
    - Utilities: FileSpec for file metadata handling

This is the recommended import location for most DerivaML type definitions:
    >>> from deriva_ml.core.definitions import RID, MLVocab, TableDefinition

For more specialized imports, you can import directly from submodules:
    >>> from deriva_ml.core.constants import ML_SCHEMA
    >>> from deriva_ml.core.enums import Status
    >>> from deriva_ml.core.ermrest import ColumnDefinition
"""

from __future__ import annotations

# =============================================================================
# Re-exported Constants
# =============================================================================
# From constants.py: Schema names, RID patterns, and column definitions
from deriva_ml.core.constants import (
    DRY_RUN_RID,
    ML_SCHEMA,
    RID,
    DerivaAssetColumns,
    DerivaSystemColumns,
    rid_part,
    rid_regex,
    snapshot_part,
)

# =============================================================================
# Re-exported Enums
# =============================================================================
# From enums.py: Status codes, type identifiers, and vocabulary names
from deriva_ml.core.enums import (
    BaseStrEnum,
    BuiltinTypes,
    ExecAssetType,
    ExecMetadataType,
    MLAsset,
    MLTable,
    MLVocab,
    Status,
    UploadState,
)

# =============================================================================
# Re-exported ERMrest Models
# =============================================================================
# From ermrest.py: Pydantic models for catalog structure definitions
from deriva_ml.core.ermrest import (
    ColumnDefinition,
    FileUploadState,
    ForeignKeyDefinition,
    KeyDefinition,
    TableDefinition,
    UploadCallback,
    UploadProgress,
    VocabularyTerm,
)

# =============================================================================
# Re-exported Exceptions
# =============================================================================
# From exceptions.py: Exception hierarchy for DerivaML errors
from deriva_ml.core.exceptions import (
    DerivaMLAuthenticationError,
    DerivaMLConfigurationError,
    DerivaMLCycleError,
    DerivaMLDataError,
    DerivaMLDatasetNotFound,
    DerivaMLException,
    DerivaMLExecutionError,
    DerivaMLInvalidTerm,
    DerivaMLNotFoundError,
    DerivaMLReadOnlyError,
    DerivaMLSchemaError,
    DerivaMLTableNotFound,
    DerivaMLTableTypeError,
    DerivaMLUploadError,
    DerivaMLValidationError,
    DerivaMLWorkflowError,
)

# =============================================================================
# Re-exported Utilities
# =============================================================================
# From filespec.py: File metadata and specification handling
from deriva_ml.core.filespec import FileSpec

__all__ = [
    # Constants
    "ML_SCHEMA",
    "DRY_RUN_RID",
    "rid_part",
    "snapshot_part",
    "rid_regex",
    "DerivaSystemColumns",
    "DerivaAssetColumns",
    "RID",
    # Enums
    "BaseStrEnum",
    "UploadState",
    "Status",
    "BuiltinTypes",
    "MLVocab",
    "MLTable",
    "MLAsset",
    "ExecMetadataType",
    "ExecAssetType",
    # Models
    "FileUploadState",
    "FileSpec",
    "VocabularyTerm",
    "ColumnDefinition",
    "KeyDefinition",
    "ForeignKeyDefinition",
    "TableDefinition",
    "UploadProgress",
    "UploadCallback",
    # Exceptions
    "DerivaMLException",
    "DerivaMLConfigurationError",
    "DerivaMLSchemaError",
    "DerivaMLAuthenticationError",
    "DerivaMLDataError",
    "DerivaMLNotFoundError",
    "DerivaMLDatasetNotFound",
    "DerivaMLTableNotFound",
    "DerivaMLInvalidTerm",
    "DerivaMLTableTypeError",
    "DerivaMLValidationError",
    "DerivaMLCycleError",
    "DerivaMLExecutionError",
    "DerivaMLWorkflowError",
    "DerivaMLUploadError",
    "DerivaMLReadOnlyError",
]
