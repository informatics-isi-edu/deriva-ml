"""Idempotent migration: add the Directory_Dataset table to an existing catalog.

``create_schema`` only builds tables at catalog-creation time, so catalogs that
predate the Directory_Dataset feature need this one-time, additive migration. The
table shape MUST match create_schema's exactly (see create_dataset_table) so a
migrated catalog is indistinguishable from a freshly-created one.
"""

from __future__ import annotations

from deriva_ml.core.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["add_directory_dataset_table"]


def add_directory_dataset_table(ml, *, apply: bool = False) -> bool:
    """Add the ``Directory_Dataset`` table to ``ml``'s catalog if absent.

    Args:
        ml: A DerivaML instance bound to the target catalog.
        apply: When False (default), report what WOULD happen without writing
            (dry-run). When True, create the table.

    Returns:
        bool: True if the table was created (or, in dry-run, would be); False if
            it already exists.

    Example:
        >>> from deriva_ml import DerivaML  # doctest: +SKIP
        >>> ml = DerivaML(hostname="dev.eye-ai.org", catalog_id="eye-ai")  # doctest: +SKIP
        >>> add_directory_dataset_table(ml, apply=False)  # dry-run  # doctest: +SKIP
        True
        >>> add_directory_dataset_table(ml, apply=True)   # create it  # doctest: +SKIP
        True
    """
    model = ml.model.model
    schema = model.schemas[ml.ml_schema]
    if "Directory_Dataset" in schema.tables:
        logger.info("Directory_Dataset already present on %s; nothing to do.", ml.ml_schema)
        return False

    if not apply:
        logger.info("[dry-run] would create Directory_Dataset on %s.", ml.ml_schema)
        return True

    # Reuse the EXACT definition create_schema uses for fresh catalogs (factory
    # added in Task 1) so a migrated catalog is identical to a created one.
    from deriva_ml.schema.create_schema import directory_dataset_table_def

    schema.create_table(directory_dataset_table_def(ml.ml_schema))
    return True
