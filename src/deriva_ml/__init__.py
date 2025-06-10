__all__ = [
    "DerivaML",
    "Execution",
    "Workflow",
    "DatasetBag",
    "DatasetVersion",
    "DatasetSpec",
    "VersionPart",
    "RID",
    "BuiltinTypes",
    "ColumnDefinition",
    "MLVocab",
    "TableDefinition",
    "ExecutionConfiguration",
]

from importlib.metadata import PackageNotFoundError, version

from deriva_ml.core import RID, BuiltinTypes, ColumnDefinition, DerivaML, MLVocab, TableDefinition
from deriva_ml.dataset.aux_classes import DatasetSpec, DatasetVersion, VersionPart
from deriva_ml.dataset.dataset_bag import DatasetBag
from deriva_ml.execution.execution import Execution, ExecutionConfiguration
from deriva_ml.execution.workflow import Workflow

try:
    __version__ = version("deriva_ml")
except PackageNotFoundError:
    # package is not installed
    pass
