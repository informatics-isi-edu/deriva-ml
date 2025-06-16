"""
Shared definitions that are used in different DerivaML modules.
This module re-exports all symbols from the core submodules for backwards compatibility.
"""

from __future__ import annotations

# Re-export constants
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

# Re-export enums
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

# Re-export models
from deriva_ml.core.ermrest import (
    ColumnDefinition,
    FileUploadState,
    ForeignKeyDefinition,
    KeyDefinition,
    TableDefinition,
    VocabularyTerm,
)

# Re-export exceptions
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
]
