"""Centralized validation configuration for DerivaML.

This module provides shared Pydantic configuration and custom validators
used throughout DerivaML. By centralizing these definitions, we ensure
consistent validation behavior and reduce code duplication.

The module provides:
    - DERIVA_ML_CONFIG: Shared ConfigDict for Pydantic models
    - VALIDATION_CONFIG: Alias for DERIVA_ML_CONFIG (for use with @validate_call)
    - Custom Pydantic types for common patterns (RID validation, etc.)

Example:
    >>> from deriva_ml.core.validation import VALIDATION_CONFIG
    >>> from pydantic import validate_call
    >>>
    >>> @validate_call(config=VALIDATION_CONFIG)
    ... def process_table(table: Table) -> None:
    ...     pass
"""

from pydantic import ConfigDict

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
]
