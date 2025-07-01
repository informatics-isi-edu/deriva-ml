__all__ = [
    "DerivaML",
    "DerivaMLException",
    "DerivaMLInvalidTerm",
    "DerivaMLTableTypeError",
    "Execution",
    "ExecAssetType",
    "ExecMetadataType",
    "Workflow",
    "DatasetBag",
    "DatasetVersion",
    "DatasetSpec",
    "FileSpec",
    "VersionPart",
    "RID",
    "BuiltinTypes",
    "ColumnDefinition",
    "MLVocab",
    "MLAsset",
    "TableDefinition",
    "ExecutionConfiguration",
]

from importlib.metadata import PackageNotFoundError, version

from deriva_ml.core import (
    RID,
    BuiltinTypes,
    ColumnDefinition,
    DerivaML,
    ExecAssetType,
    ExecMetadataType,
    FileSpec,
    MLAsset,
    MLVocab,
    TableDefinition,
)
from deriva_ml.core.exceptions import DerivaMLException, DerivaMLInvalidTerm, DerivaMLTableTypeError
from deriva_ml.dataset.aux_classes import DatasetSpec, DatasetVersion, VersionPart
from deriva_ml.dataset.dataset_bag import DatasetBag
from deriva_ml.execution.execution import Execution, ExecutionConfiguration
from deriva_ml.execution.workflow import Workflow

try:
    __version__ = version("deriva_ml")
except PackageNotFoundError:
    # package is not installed
    pass
