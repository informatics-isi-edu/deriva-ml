__all__ = [
    "ExecutionConfiguration",
    "Execution",
    "Workflow",
    "DatasetBag",
    "DatasetVersion",
    "DatasetSpec",
    "VersionPart",
]

from importlib.metadata import PackageNotFoundError, version

from deriva_ml.core.base import *
from deriva_ml.dataset.aux_classes import DatasetSpec, DatasetVersion, VersionPart
from deriva_ml.dataset.dataset_bag import DatasetBag
from deriva_ml.execution.execution import Execution, ExecutionConfiguration
from deriva_ml.execution.workflow import Workflow

try:
    __version__ = version("deriva_ml")
except PackageNotFoundError:
    # package is not installed
    pass
