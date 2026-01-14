"""Constants used throughout the DerivaML package.

This module defines fundamental constants, type aliases, and regular expressions
used for validating and working with Deriva catalog structures.

Constants:
    ML_SCHEMA: Default schema name for ML-related tables ('deriva-ml').
    DRY_RUN_RID: Special RID used for dry-run operations without database changes.

Type Aliases:
    RID: Annotated string type for Resource Identifiers with validation.

Regular Expressions:
    rid_part: Pattern for matching the RID portion of an identifier.
    snapshot_part: Pattern for matching optional snapshot timestamps.
    rid_regex: Complete pattern for validating RID strings.

Column Sets:
    DerivaSystemColumns: Standard Deriva system columns present in all tables.
    DerivaAssetColumns: Columns specific to asset tables (files, etc.).

Example:
    >>> from deriva_ml.core.constants import RID, ML_SCHEMA
    >>> def process_entity(rid: RID) -> None:
    ...     # RID is validated by Pydantic
    ...     pass
"""

from __future__ import annotations

from typing import Annotated

from pydantic import StringConstraints

# =============================================================================
# Schema Constants
# =============================================================================

# Default schema name for ML-related tables in the catalog
ML_SCHEMA = "deriva-ml"

# Special RID value used for dry-run operations that don't modify the database
DRY_RUN_RID = "0000"

# =============================================================================
# RID Regular Expression Components
# =============================================================================

# Pattern for the RID portion: 1-4 alphanumeric chars, optionally followed by
# hyphen-separated groups of exactly 4 alphanumeric chars (e.g., "1ABC" or "1ABC-DEF2-3GHI")
rid_part = r"(?P<rid>(?:[A-Z\d]{1,4}|[A-Z\d]{1,4}(?:-[A-Z\d]{4})+))"

# Pattern for optional snapshot timestamp suffix (e.g., "@2024-01-01T12:00:00")
# Uses the same format as RID for the snapshot identifier
snapshot_part = r"(?:@(?P<snapshot>(?:[A-Z\d]{1,4}|[A-Z\d]{1,4}(?:-[A-Z\d]{4})+)))?"

# Complete regex for validating RID strings with optional snapshot
rid_regex = f"^{rid_part}{snapshot_part}$"

# =============================================================================
# Type Aliases
# =============================================================================

# RID type with Pydantic validation - ensures strings match the RID format
# Used throughout the codebase for type hints and runtime validation
RID = Annotated[str, StringConstraints(pattern=rid_regex)]

# =============================================================================
# Column Definitions
# =============================================================================

# Standard Deriva system columns present in every table:
# - RID: Resource Identifier (unique key)
# - RCT: Record Creation Time
# - RMT: Record Modification Time
# - RCB: Record Created By (user ID)
# - RMB: Record Modified By (user ID)
DerivaSystemColumns = ["RID", "RCT", "RMT", "RCB", "RMB"]

# Columns specific to asset tables (files, images, etc.)
# Includes system columns plus asset-specific metadata
DerivaAssetColumns = {
    "Filename",    # Original filename
    "URL",         # Hatrac storage URL
    "Length",      # File size in bytes
    "MD5",         # MD5 checksum for integrity verification
    "Description", # Optional description of the asset
}.union(set(DerivaSystemColumns))