"""Model module for DerivaML.

This module provides catalog and database model classes, as well as
handle wrappers for ERMrest model objects and annotation builders.

Key components:
- DerivaModel: Schema analysis utilities
- DatabaseModel: SQLite database from BDBag
- SchemaBuilder/SchemaORM: Create ORM from Deriva Model (Phase 1)
- DataLoader: Fill database from data source (Phase 2)
- DataSource: Protocol for data sources (BagDataSource, CatalogDataSource)
- ForeignKeyOrderer: Compute FK-safe insertion order

Lazy imports are used for DatabaseModel and DerivaMLDatabase to avoid
circular imports with the dataset module.
"""

from deriva_ml.model.catalog import DerivaModel
from deriva_ml.model.handles import ColumnHandle, TableHandle

# Two-phase ORM creation components
from deriva_ml.model.schema_builder import SchemaBuilder, SchemaORM
from deriva_ml.model.data_sources import DataSource, BagDataSource, CatalogDataSource
from deriva_ml.model.data_loader import DataLoader
from deriva_ml.model.fk_orderer import ForeignKeyOrderer

# Annotation builders - import the most common ones for convenience
from deriva_ml.model.annotations import (
    # Builders
    Display,
    VisibleColumns,
    VisibleForeignKeys,
    TableDisplay,
    TableDisplayOptions,
    ColumnDisplay,
    ColumnDisplayOptions,
    PreFormat,
    PseudoColumn,
    PseudoColumnDisplay,
    Facet,
    FacetList,
    FacetRange,
    SortKey,
    NameStyle,
    # FK helpers
    InboundFK,
    OutboundFK,
    fk_constraint,
    # Enums
    TemplateEngine,
    Aggregate,
    ArrayUxMode,
    FacetUxMode,
    # Context constants
    CONTEXT_DEFAULT,
    CONTEXT_COMPACT,
    CONTEXT_DETAILED,
    CONTEXT_ENTRY,
    CONTEXT_FILTER,
)

__all__ = [
    # Core classes
    "DerivaModel",
    "DatabaseModel",
    "DerivaMLDatabase",
    "TableHandle",
    "ColumnHandle",
    # Two-phase ORM creation
    "SchemaBuilder",
    "SchemaORM",
    "DataSource",
    "BagDataSource",
    "CatalogDataSource",
    "DataLoader",
    "ForeignKeyOrderer",
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
    """Lazy import for DatabaseModel and DerivaMLDatabase."""
    if name == "DatabaseModel":
        from deriva_ml.model.database import DatabaseModel

        return DatabaseModel
    if name == "DerivaMLDatabase":
        from deriva_ml.model.deriva_ml_database import DerivaMLDatabase

        return DerivaMLDatabase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
