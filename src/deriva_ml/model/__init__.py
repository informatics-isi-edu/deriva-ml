"""Model module for DerivaML.

This module provides catalog and database model classes, plus
annotation builders. Schema/data infrastructure that used to live
here (``SchemaBuilder``, ``DataLoader``, ``DataSource``, etc.) now
lives upstream in :mod:`deriva.bag`; import from there directly.

Key components:
- DerivaModel: Schema analysis utilities
- DatabaseModel: SQLite database from BDBag
- DerivaMLBagView: deriva-ml-domain view over a DatabaseModel

Lazy imports are used for DatabaseModel and DerivaMLBagView to
avoid circular imports with the dataset module.
"""

# Annotation builders - import the most common ones for convenience
from deriva.core.model_handles import ColumnHandle, TableHandle

from deriva_ml.model.annotations import (
    CONTEXT_COMPACT,
    # Context constants
    CONTEXT_DEFAULT,
    CONTEXT_DETAILED,
    CONTEXT_ENTRY,
    CONTEXT_FILTER,
    Aggregate,
    ArrayUxMode,
    ColumnDisplay,
    ColumnDisplayOptions,
    # Builders
    Display,
    Facet,
    FacetList,
    FacetRange,
    FacetUxMode,
    # FK helpers
    InboundFK,
    NameStyle,
    OutboundFK,
    PreFormat,
    PseudoColumn,
    PseudoColumnDisplay,
    SortKey,
    TableDisplay,
    TableDisplayOptions,
    # Enums
    TemplateEngine,
    VisibleColumns,
    VisibleForeignKeys,
    fk_constraint,
)
from deriva_ml.model.catalog import DerivaModel

__all__ = [
    # Core classes
    "DerivaModel",
    "DatabaseModel",
    "DerivaMLBagView",
    "TableHandle",
    "ColumnHandle",
    # Annotation builders
    "Display",
    "VisibleColumns",
    "VisibleForeignKeys",
    "TableDisplay",
    "TableDisplayOptions",
    "ColumnDisplay",
    "ColumnDisplayOptions",
    "PreFormat",
    "PseudoColumn",
    "PseudoColumnDisplay",
    "Facet",
    "FacetList",
    "FacetRange",
    "SortKey",
    "NameStyle",
    # FK helpers
    "InboundFK",
    "OutboundFK",
    "fk_constraint",
    # Enums
    "TemplateEngine",
    "Aggregate",
    "ArrayUxMode",
    "FacetUxMode",
    # Context constants
    "CONTEXT_DEFAULT",
    "CONTEXT_COMPACT",
    "CONTEXT_DETAILED",
    "CONTEXT_ENTRY",
    "CONTEXT_FILTER",
]


def __getattr__(name: str):
    """Lazy import for DatabaseModel and DerivaMLBagView."""
    if name == "DatabaseModel":
        from deriva_ml.model.database import DatabaseModel

        return DatabaseModel
    if name == "DerivaMLBagView":
        from deriva_ml.model.deriva_ml_bag_view import DerivaMLBagView

        return DerivaMLBagView
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
