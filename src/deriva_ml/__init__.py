__all__ = [
    "DerivaML",
    "DerivaMLException",
    "FileUploadState",
    "FileSpec",
    "ExecutionConfiguration",
    "Execution",
    "Workflow",
    "DatasetBag",
    "DatasetVersion",
    "DatasetSpec",
    "ColumnDefinition",
    "TableDefinition",
    "BuiltinTypes",
    "UploadState",
    "MLVocab",
    "ExecMetadataVocab",
    "RID",
    "DerivaSystemColumns",
    "VersionPart",
]

from .dataset_aux_classes import VersionPart, DatasetSpec, DatasetVersion
from .dataset_bag import DatasetBag
from .deriva_definitions import (
    ColumnDefinition,
    TableDefinition,
    BuiltinTypes,
    UploadState,
    FileUploadState,
    FileSpec,
    RID,
    DerivaMLException,
    MLVocab,
    ExecMetadataVocab,
    DerivaSystemColumns,
)
from .deriva_ml_base import DerivaML
from .execution_configuration import (
    ExecutionConfiguration,
    Workflow,
)
from .execution import Execution

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("deriva_ml")
except PackageNotFoundError:
    # package is not installed
    pass

