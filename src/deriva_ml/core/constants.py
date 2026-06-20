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

Internal Helpers:
    _is_system_schema: Check whether a schema name is a system/ML schema.
    _get_domain_schemas: Filter a schema list to only user-defined domain schemas.

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
# Unknown-provenance sentinels
# =============================================================================
# Three rows seeded at catalog initialization (``initialize_ml_schema``) that
# represent "provenance is explicitly unknown" rather than null. They are
# *bootstrap substrate* — exempt from the provenance contract's producer /
# completeness obligations and from the audit's violation checks.
#
# - The unknown-provenance Workflow backs the sentinel Execution (Execution
#   requires a Workflow FK).
# - The sentinel Execution is what a producerless artifact attributes to, so
#   lineage from such an artifact terminates at "unknown origin" instead of a
#   null dead-end.
# - The unknown-provenance File is linked as an Input to an artifact-producer
#   that committed with no declared input (the no-input commit check).
#
# Each is identified idempotently by a reserved ``tag:`` URI / marker so that
# re-running ``initialize_ml_schema`` never duplicates them. ``tag:`` URIs are
# non-dereferenceable by design (RFC 4151) — exactly right for a marker that
# names a concept, not a fetchable resource.
SENTINEL_WORKFLOW_URL = "tag:deriva-ml,unknown-provenance:workflow"
SENTINEL_WORKFLOW_CHECKSUM = "0" * 40  # marker; not a real content hash
SENTINEL_FILE_URL = "tag:deriva-ml,unknown-provenance:file"
SENTINEL_FILE_MD5 = "0" * 32  # marker; not computed from bytes (see contract exemption)
# The sentinel Execution carries this exact Description so it can be located
# idempotently (Execution has no natural unique key like URL).
SENTINEL_EXECUTION_DESCRIPTION = "deriva-ml unknown-provenance sentinel execution"

# System schemas that are part of Deriva infrastructure (not user domain schemas)
# These are excluded when auto-detecting domain schemas
SYSTEM_SCHEMAS: frozenset[str] = frozenset({"public", "www", "WWW"})

# FK cycles that the deriva-ml schema designs in deliberately, and that
# deriva-py's bag pipeline already knows how to break correctly.
#
# Each entry is a frozenset of fully-qualified table names
# (``{schema}.{table}``) participating in one cycle. Passed to
# ``deriva.bag.traversal.FKTraversalPolicy(intentional_cycles=...)``
# at every deriva-ml ``FKTraversalPolicy(...)`` construction site
# (clone, bag export, execution-commit upload). The loader still
# breaks these cycles to sort tables — it just logs the break at
# DEBUG instead of WARNING.
#
# Why only these specific cycles are silenced:
#
# * ``Dataset ↔ Dataset_Version`` — a Dataset row carries a
#   ``current_version`` FK to a Dataset_Version row, and each
#   Dataset_Version carries a ``Dataset`` FK back. Both directions
#   are operationally needed (per-dataset versioning + per-version
#   provenance). The cycle is by design and not changing.
#
# Cycles **not** in this set should still surface as WARNING in
# the bag-loader output — those warnings exist to flag accidental
# schema bugs. Adding to this set is a deliberate opt-in.
INTENTIONAL_FK_CYCLES: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({f"{ML_SCHEMA}.Dataset", f"{ML_SCHEMA}.Dataset_Version"}),
    }
)

# Provenance tables the dataset-bag walk ENTERS but does not traverse
# outward. ``Execution`` and ``Workflow`` describe *how* rows came to
# be, not *what they are*; one Execution aggregates state across every
# Subject/Image/Dataset its run touched and has many inbound FKs (every
# ``*_Execution`` association, plus self-loops via ``Execution_Execution``
# and back-edges to feature tables like ``Annotation``). Following those
# inbound FKs makes the walk fan out across the entire catalog provenance
# graph — an 18-minute hang on large nested datasets (eye-ai 2-277G).
# Marking them terminal keeps the provenance *link* in the bag while
# severing the fan-out. Shared by ``DatasetBagBuilder.build_policy`` and
# ``clone_via_bag`` so the two bag-producing paths cannot diverge.
PROVENANCE_TERMINAL_TABLES: frozenset[tuple[str, str]] = frozenset(
    {
        (ML_SCHEMA, "Execution"),
        (ML_SCHEMA, "Workflow"),
    }
)


__all__ = [
    "ML_SCHEMA",
    "DRY_RUN_RID",
    "SYSTEM_SCHEMAS",
    "INTENTIONAL_FK_CYCLES",
    "PROVENANCE_TERMINAL_TABLES",
    "RID",
    "DerivaSystemColumns",
    "DerivaAssetColumns",
    "rid_part",
    "snapshot_part",
    "rid_regex",
]


def _is_system_schema(schema_name: str, ml_schema: str = ML_SCHEMA) -> bool:
    """Check if a schema is a system or ML schema (not a domain schema).

    System schemas are Deriva infrastructure schemas (public, www, WWW) and the
    ML schema (deriva-ml by default). Domain schemas are user-defined schemas
    containing business logic tables.

    Args:
        schema_name: Name of the schema to check.
        ml_schema: Name of the ML schema (default: 'deriva-ml').

    Returns:
        True if the schema is a system or ML schema, False if it's a domain schema.

    Example:
        >>> _is_system_schema("public")
        True
        >>> _is_system_schema("deriva-ml")
        True
        >>> _is_system_schema("my_project")
        False
    """
    return schema_name.lower() in {s.lower() for s in SYSTEM_SCHEMAS} or schema_name == ml_schema


def _get_domain_schemas(all_schemas: set[str] | list[str], ml_schema: str = ML_SCHEMA) -> frozenset[str]:
    """Return all domain schemas from a collection of schema names.

    Filters out system schemas (public, www, WWW) and the ML schema to return
    only user-defined domain schemas.

    Args:
        all_schemas: Collection of schema names to filter.
        ml_schema: Name of the ML schema to exclude (default: 'deriva-ml').

    Returns:
        Frozen set of domain schema names.

    Example:
        >>> _get_domain_schemas(["public", "deriva-ml", "my_project", "www"])
        frozenset({'my_project'})
    """
    return frozenset(s for s in all_schemas if not _is_system_schema(s, ml_schema))


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
    "Filename",  # Original filename
    "URL",  # Hatrac storage URL
    "Length",  # File size in bytes
    "MD5",  # MD5 checksum for integrity verification
    "Description",  # Optional description of the asset
}.union(set(DerivaSystemColumns))
