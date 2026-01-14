"""Mixins for DerivaML catalog operations.

This module provides mixins that can be used to compose catalog-related
functionality. Each mixin provides a specific set of operations that can
be mixed into classes that have access to a catalog.

Mixins:
    VocabularyMixin: Vocabulary term management (add, lookup, list terms)
    RidResolutionMixin: RID resolution and retrieval
    PathBuilderMixin: Path building and table access utilities
    WorkflowMixin: Workflow management (add, lookup, list, create)
    FeatureMixin: Feature management (create, lookup, delete, list values)
    DatasetMixin: Dataset management (find, create, lookup, delete)
    AssetMixin: Asset management (create, list assets)
    ExecutionMixin: Execution management (create, restore, update status)
    FileMixin: File management (add, list files)
    AnnotationMixin: Annotation management (display, visible-columns, etc.)
"""

from deriva_ml.core.mixins.annotation import AnnotationMixin
from deriva_ml.core.mixins.asset import AssetMixin
from deriva_ml.core.mixins.dataset import DatasetMixin
from deriva_ml.core.mixins.execution import ExecutionMixin
from deriva_ml.core.mixins.feature import FeatureMixin
from deriva_ml.core.mixins.file import FileMixin
from deriva_ml.core.mixins.path_builder import PathBuilderMixin
from deriva_ml.core.mixins.rid_resolution import RidResolutionMixin
from deriva_ml.core.mixins.vocabulary import VocabularyMixin
from deriva_ml.core.mixins.workflow import WorkflowMixin

__all__ = [
    "AnnotationMixin",
    "VocabularyMixin",
    "RidResolutionMixin",
    "PathBuilderMixin",
    "WorkflowMixin",
    "FeatureMixin",
    "DatasetMixin",
    "AssetMixin",
    "ExecutionMixin",
    "FileMixin",
]
