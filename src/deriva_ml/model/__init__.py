"""Model module for DerivaML.

This module provides catalog and database model classes.

Lazy imports are used for DatabaseModel and DerivaMLDatabase to avoid
circular imports with the dataset module.
"""

from deriva_ml.model.catalog import DerivaModel

__all__ = ["DerivaModel", "DatabaseModel", "DerivaMLDatabase"]


def __getattr__(name: str):
    """Lazy import for DatabaseModel and DerivaMLDatabase."""
    if name == "DatabaseModel":
        from deriva_ml.model.database import DatabaseModel

        return DatabaseModel
    if name == "DerivaMLDatabase":
        from deriva_ml.model.deriva_ml_database import DerivaMLDatabase

        return DerivaMLDatabase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
