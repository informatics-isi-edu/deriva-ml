"""Centralized validation configuration for DerivaML.

This module provides shared Pydantic configuration, custom validators,
and RID validation utilities used throughout DerivaML.

The module provides:
    - DERIVA_ML_CONFIG: Shared ConfigDict for Pydantic models
    - VALIDATION_CONFIG: Alias for DERIVA_ML_CONFIG (for use with @validate_call)
    - Custom Pydantic types for common patterns (RID validation, etc.)
    - validate_rids(): Validate that RIDs exist in the catalog
    - ValidationResult: Result container for validation operations

Example (Pydantic config):
    >>> from deriva_ml.core.validation import VALIDATION_CONFIG  # doctest: +SKIP
    >>> from pydantic import validate_call  # doctest: +SKIP
    >>>
    >>> @validate_call(config=VALIDATION_CONFIG)  # doctest: +SKIP
    ... def process_table(table: Table) -> None:
    ...     pass

Example (RID validation):
    >>> from deriva_ml.core.validation import validate_rids  # doctest: +SKIP
    >>>
    >>> result = validate_rids(  # doctest: +SKIP
    ...     ml,
    ...     dataset_rids=["1-ABC", "2-DEF"],
    ...     asset_rids=["3-GHI"],
    ... )
    >>> if not result.is_valid:  # doctest: +SKIP
    ...     for error in result.errors:
    ...         print(f"ERROR: {error}")
"""

from __future__ import annotations


from pydantic import BaseModel, ConfigDict, Field

from deriva_ml.core.logging_config import get_logger

logger = get_logger(__name__)
# =============================================================================
# Shared Pydantic Configuration
# =============================================================================

# Standard configuration for DerivaML Pydantic models and validate_call decorators.
# This allows arbitrary types (like deriva Table, Column, etc.) to be used in
# Pydantic validation without explicit type adapters.
VALIDATION_CONFIG = ConfigDict(
    arbitrary_types_allowed=True,
    # Validate default values during model creation
    validate_default=True,
    # Use enum values instead of enum members for serialization
    use_enum_values=True,
)

# Alias for backwards compatibility and clarity in model definitions
DERIVA_ML_CONFIG = VALIDATION_CONFIG

# Configuration for models that should be strict about extra fields
STRICT_VALIDATION_CONFIG = ConfigDict(
    arbitrary_types_allowed=True,
    validate_default=True,
    use_enum_values=True,
    extra="forbid",  # Raise error if extra fields provided
)

__all__ = [
    "VALIDATION_CONFIG",
    "DERIVA_ML_CONFIG",
    "STRICT_VALIDATION_CONFIG",
    "ValidationResult",
    "validate_rids",
    "validate_execution_config",
]


# =============================================================================
# RID Validation
# =============================================================================

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deriva_ml.core.base import DerivaML


class ValidationResult(BaseModel):
    """Result of configuration validation.

    Pydantic model — provides ``.model_dump()`` for JSON output and
    ``.model_dump_json()`` for one-line serialization, aligning with
    every other user-facing return type in deriva-ml (CLAUDE.md
    "Class idiom choice" guidance).

    When printed, displays a formatted summary of validation results
    including any errors and warnings.

    Attributes:
        is_valid: True if all validations passed, False otherwise.
        errors: List of error messages for failed validations.
        warnings: List of warning messages for potential issues.
        validated_rids: Dictionary mapping RID to its resolved table info.

    Example:
        >>> result = validate_rids(ml, dataset_rids=["1-ABC"])  # doctest: +SKIP
        >>> print(result)  # doctest: +SKIP
        OK Validation passed
          Validated 1 RIDs

        >>> result = validate_rids(ml, dataset_rids=["INVALID"])  # doctest: +SKIP
        >>> print(result)  # doctest: +SKIP
        FAIL Validation failed with 1 error(s)

        Errors:
          - Dataset RID 'INVALID' does not exist in catalog
    """

    model_config = VALIDATION_CONFIG

    is_valid: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    validated_rids: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Add an error message and mark result as invalid."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another validation result into this one."""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.validated_rids.update(other.validated_rids)
        return self

    def __repr__(self) -> str:
        """Return a formatted string representation of the validation result."""
        lines = []

        if self.is_valid:
            lines.append("OK Validation passed")
            if self.validated_rids:
                lines.append(f"  Validated {len(self.validated_rids)} RID(s)")
        else:
            lines.append(f"FAIL Validation failed with {len(self.errors)} error(s)")

        if self.errors:
            lines.append("")
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"  - {error}")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  WARN {warning}")

        return "\n".join(lines)

    def __str__(self) -> str:
        """Return a formatted string for print()."""
        return self.__repr__()


def validate_rids(
    ml: "DerivaML",
    dataset_rids: list[str] | None = None,
    asset_rids: list[str] | None = None,
    dataset_versions: dict[str, str] | None = None,
    workflow_rids: list[str] | None = None,
    execution_rids: list[str] | None = None,
    warn_missing_descriptions: bool = True,
) -> ValidationResult:
    """Validate that RIDs exist in the catalog.

    Performs batch validation of RIDs to ensure they exist before running
    experiments. This catches configuration errors early with clear messages.

    Args:
        ml: Connected DerivaML instance.
        dataset_rids: List of dataset RIDs to validate.
        asset_rids: List of asset RIDs to validate.
        dataset_versions: Dictionary mapping dataset RID to required version string.
            If provided, validates that the dataset has the specified version.
        workflow_rids: List of workflow RIDs to validate.
        execution_rids: List of execution RIDs to validate.
        warn_missing_descriptions: If True (default), warn when datasets or other
            entities are missing descriptions.

    Returns:
        ValidationResult with is_valid flag, error/warning messages, and
        resolved RID information.

    Example:
        >>> result = validate_rids(  # doctest: +SKIP
        ...     ml,
        ...     dataset_rids=["1-ABC", "2-DEF"],
        ...     dataset_versions={"1-ABC": "0.4.0"},
        ...     asset_rids=["3-GHI"],
        ... )
        >>> print(result)  # doctest: +SKIP
        OK Validation passed
          Validated 3 RID(s)
    """
    from deriva_ml.core.exceptions import DerivaMLException, DerivaMLRidsNotFound

    result = ValidationResult()

    # Collect all RIDs for batch resolution
    all_rids: set[str] = set()
    rid_categories: dict[str, str] = {}  # Maps RID to category for error messages

    if dataset_rids:
        for rid in dataset_rids:
            all_rids.add(rid)
            rid_categories[rid] = "dataset"

    if asset_rids:
        for rid in asset_rids:
            all_rids.add(rid)
            rid_categories[rid] = "asset"

    if workflow_rids:
        for rid in workflow_rids:
            all_rids.add(rid)
            rid_categories[rid] = "workflow"

    if execution_rids:
        for rid in execution_rids:
            all_rids.add(rid)
            rid_categories[rid] = "execution"

    if not all_rids:
        return result  # Nothing to validate

    # Batch resolve all RIDs
    try:
        resolved = ml.resolve_rids(all_rids)
        for rid, info in resolved.items():
            result.validated_rids[rid] = {
                "rid": rid,
                "table": info.table_name,
                "schema": info.schema_name,
            }
    except DerivaMLRidsNotFound as e:
        # Read the unresolved set directly off the typed exception —
        # no string parsing. Each unresolved RID gets a category-
        # qualified error message so callers see exactly which input
        # failed to resolve.
        for rid in e.missing_rids:
            category = rid_categories.get(rid, "unknown")
            result.add_error(f"{category.title()} RID '{rid}' does not exist in catalog")
    except DerivaMLException as e:
        # Any other DerivaML failure during resolution (catalog
        # error, network issue, etc.) is reported as a generic
        # validation failure rather than per-RID.
        result.add_error(f"RID validation failed: {e}")

    # Validate dataset versions if specified
    if dataset_versions and dataset_rids:
        for rid, required_version in dataset_versions.items():
            if rid not in result.validated_rids:
                continue  # Already reported as missing

            try:
                dataset = ml.lookup_dataset(rid)
                current_version = str(dataset.current_version) if dataset.current_version else None

                if current_version is None:
                    result.add_warning(
                        f"Dataset '{rid}' has no version information. Required version: {required_version}"
                    )
                elif current_version != required_version:
                    # Check if the required version exists in history.
                    try:
                        history = dataset.dataset_history()
                        version_exists = any(str(h.dataset_version) == required_version for h in history)
                        if not version_exists:
                            result.add_error(
                                f"Dataset '{rid}' does not have version '{required_version}'. "
                                f"Current version: {current_version}. "
                                f"Available versions: {[str(h.dataset_version) for h in history]}"
                            )
                        else:
                            # Version exists but is not current - this is OK
                            result.validated_rids[rid]["version"] = required_version
                            result.validated_rids[rid]["current_version"] = current_version
                    except DerivaMLException:
                        # Narrow catch: ``dataset_history()`` documents
                        # ``DerivaMLException`` as its only failure mode (a
                        # catalog read error, or the RID not being a valid
                        # dataset RID). That is a genuine "can't read history"
                        # condition we degrade to a warning rather than crash.
                        # Deliberately NOT ``except Exception`` — programming
                        # errors (AttributeError, TypeError) must propagate
                        # loudly so a future typo can't silently downgrade a
                        # hard version error to a warning, as a misnamed
                        # ``list_versions()`` call once did.
                        result.add_warning(
                            f"Dataset '{rid}' current version ({current_version}) differs from "
                            f"required version ({required_version}). Could not verify version history."
                        )
                else:
                    result.validated_rids[rid]["version"] = required_version
            except Exception as e:
                result.add_error(f"Failed to validate dataset '{rid}' version: {e}")

    # Validate that datasets are actually in Dataset table
    if dataset_rids:
        for rid in dataset_rids:
            if rid in result.validated_rids:
                info = result.validated_rids[rid]
                if info.get("table") != "Dataset":
                    result.add_error(
                        f"RID '{rid}' specified as dataset but found in table "
                        f"'{info.get('schema')}.{info.get('table')}'"
                    )

    # Validate that workflow RIDs are in Workflow table
    if workflow_rids:
        for rid in workflow_rids:
            if rid in result.validated_rids:
                info = result.validated_rids[rid]
                if info.get("table") != "Workflow":
                    result.add_error(
                        f"RID '{rid}' specified as workflow but found in table "
                        f"'{info.get('schema')}.{info.get('table')}'"
                    )

    # Validate that execution RIDs are in Execution table
    if execution_rids:
        for rid in execution_rids:
            if rid in result.validated_rids:
                info = result.validated_rids[rid]
                if info.get("table") != "Execution":
                    result.add_error(
                        f"RID '{rid}' specified as execution but found in table "
                        f"'{info.get('schema')}.{info.get('table')}'"
                    )

    # Check for missing descriptions
    if warn_missing_descriptions and dataset_rids:
        for rid in dataset_rids:
            if rid in result.validated_rids and result.validated_rids[rid].get("table") == "Dataset":
                try:
                    dataset = ml.lookup_dataset(rid)
                    if not dataset.description or dataset.description.strip() == "":
                        result.add_warning(f"Dataset '{rid}' has no description")
                except Exception as e:
                    logger.debug(f"Could not check description for dataset {rid}: {e}")

    return result


def validate_execution_config(
    ml: "DerivaML",
    datasets: list[Any],
    assets: list[Any],
) -> ValidationResult:
    """Validate that all dataset and asset RIDs in a resolved execution config exist in the catalog.

    This is called automatically by run_model() after Hydra config resolution and
    catalog connection, but before creating the execution or downloading any data.
    It catches configuration errors (typos in RIDs, wrong catalog, missing versions)
    early with clear messages.

    Args:
        ml: Connected DerivaML instance.
        datasets: Resolved dataset specifications (``DatasetSpec`` objects or
            dicts with ``rid``/``version``).
        assets: Resolved asset specifications (``AssetSpec`` objects, dicts, or
            bare RID strings).

    Returns:
        ValidationResult: Validation result with errors for missing RIDs or
        versions, and warnings for stale versions.

    Example:
        >>> result = validate_execution_config(ml, datasets, assets)  # doctest: +SKIP
        >>> if not result.is_valid:  # doctest: +SKIP
        ...     raise DerivaMLException(f"Config validation failed:\\n{result}")
    """
    # Extract dataset RIDs and versions
    dataset_rids: list[str] = []
    dataset_versions: dict[str, str] = {}

    for ds in datasets:
        rid = getattr(ds, "rid", None) or (ds.get("rid") if isinstance(ds, dict) else None)
        if rid is None:
            continue
        rid = str(rid)
        dataset_rids.append(rid)

        # Extract version
        version = getattr(ds, "version", None) or (ds.get("version") if isinstance(ds, dict) else None)
        if version is not None:
            dataset_versions[rid] = str(version)

    # Extract asset RIDs
    asset_rids: list[str] = []
    for a in assets:
        if isinstance(a, str):
            asset_rids.append(a)
        else:
            rid = getattr(a, "rid", None) or (a.get("rid") if isinstance(a, dict) else None)
            if rid is not None:
                asset_rids.append(str(rid))

    if not dataset_rids and not asset_rids:
        return ValidationResult()

    logger.info(
        "Validating execution config: %d dataset(s), %d asset(s)",
        len(dataset_rids),
        len(asset_rids),
    )

    return validate_rids(
        ml,
        dataset_rids=dataset_rids or None,
        asset_rids=asset_rids or None,
        dataset_versions=dataset_versions or None,
        warn_missing_descriptions=False,
    )
