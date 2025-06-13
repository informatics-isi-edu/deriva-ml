__all__ = [
    "DerivaML",
    "DerivaMLException",
    "DerivaMLInvalidTerm",
    "DerivaMLTableTypeError",
    "Execution",
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
    "TableDefinition",
    "ExecutionConfiguration",
]

from importlib.metadata import PackageNotFoundError, version

from deriva_ml.core import (
    RID,
    BuiltinTypes,
    ColumnDefinition,
    DerivaML,
    FileSpec,
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

# Optional debug imports
try:
    from icecream import ic
    ic.install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa
