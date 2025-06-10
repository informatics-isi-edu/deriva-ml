from deriva_ml.core.base import DerivaML
from deriva_ml.core.definitions import (
    RID,
    BuiltinTypes,
    ColumnDefinition,
    DerivaMLException,
    DerivaSystemColumns,
    ExecAssetType,
    FileSpec,
    FileUploadState,
    MLAsset,
    MLVocab,
    TableDefinition,
    UploadState,
)

__all__ = [
    DerivaML,
    RID,
    BuiltinTypes,
    ColumnDefinition,
    DerivaMLException,
    DerivaSystemColumns,
    ExecAssetType,
    FileSpec,
    FileUploadState,
    MLAsset,
    MLVocab,
    TableDefinition,
    UploadState,
]
# Optional debug imports
try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa
