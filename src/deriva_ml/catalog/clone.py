"""Enhanced catalog cloning with cross-server and selective asset support.

This module provides catalog cloning that handles the common case of incoherent
row-level policies in source catalogs. When source policies hide some domain
table rows but don't hide the referring rows, foreign key violations occur
during cloning.

The solution uses a three-stage approach:
1. Create schema WITHOUT foreign keys
2. Copy all data
3. Apply foreign keys, handling violations by either:
   - Deleting orphan rows (rows with dangling FK references)
   - Nullifying references (setting dangling FK values to NULL)

This approach is more robust than trying to pre-filter data, as it handles
all edge cases including circular dependencies and complex FK relationships.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.parse import quote as urlquote

from deriva.core import DerivaServer, ErmrestCatalog, get_credential
from deriva.core.hatrac_store import HatracStore

logger = logging.getLogger("deriva_ml")


class CloneIssueSeverity(Enum):
    """Severity level of clone issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CloneIssueCategory(Enum):
    """Category of clone issues."""

    ACCESS_DENIED = "access_denied"
    ORPHAN_ROWS = "orphan_rows"
    DATA_INTEGRITY = "data_integrity"
    SCHEMA_ISSUE = "schema_issue"
    RESTORE_FAILURE = "restore_failure"
    FK_VIOLATION = "fk_violation"
    FK_PRUNED = "fk_pruned"  # FK was intentionally not applied
    POLICY_INCOHERENCE = "policy_incoherence"
    INDEX_REBUILT = "index_rebuilt"  # Index was dropped and rebuilt due to size limits


class OrphanStrategy(Enum):
    """Strategy for handling orphan rows (rows with dangling FK references).

    When cloning a catalog with incoherent row-level policies, some rows may
    reference parent rows that are hidden from the cloning user. These orphan
    rows would violate FK constraints.
    """

    FAIL = "fail"  # Fail the clone if FK violations occur
    DELETE = "delete"  # Delete rows with dangling references
    NULLIFY = "nullify"  # Set dangling FK values to NULL (requires nullok)


class AssetCopyMode(Enum):
    """How to handle assets during catalog cloning."""

    NONE = "none"  # Don't copy assets
    REFERENCES = "refs"  # Keep URLs pointing to source
    FULL = "full"  # Download and re-upload assets


@dataclass
class CloneIssue:
    """A single issue encountered during catalog cloning."""

    severity: CloneIssueSeverity
    category: CloneIssueCategory
    message: str
    table: str | None = None
    details: str | None = None
    action: str | None = None
    row_count: int = 0
    skipped_rids: list[str] | None = None  # RIDs of rows that were skipped

    def to_dict(self) -> dict[str, Any]:
        result = {
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "table": self.table,
            "details": self.details,
            "action": self.action,
            "row_count": self.row_count,
        }
        if self.skipped_rids:
            result["skipped_rids"] = self.skipped_rids
        return result

    def __str__(self) -> str:
        parts = [f"[{self.severity.value.upper()}]"]
        if self.table:
            parts.append(f"{self.table}:")
        parts.append(self.message)
        if self.row_count > 0:
            parts.append(f"({self.row_count} rows)")
        result = " ".join(parts)
        if self.skipped_rids:
            # For small numbers, list the RIDs; for large numbers, just show count
            if len(self.skipped_rids) <= 5:
                result += f"\n    Skipped RIDs: {', '.join(self.skipped_rids)}"
            else:
                result += f"\n    Skipped RIDs: {len(self.skipped_rids)} rows (see JSON for full list)"
        return result


@dataclass
class CloneReport:
    """Comprehensive report of catalog clone operation.

    Tracks all issues encountered during cloning, including:
    - Policy incoherence issues (FK violations due to hidden data)
    - Orphan rows that were deleted or nullified
    - FKs that were pruned or failed
    - Tables that were restored, failed, or skipped

    Provides both JSON and text output formats for reporting.
    """

    issues: list[CloneIssue] = field(default_factory=list)
    tables_restored: dict[str, int] = field(default_factory=dict)
    tables_failed: list[str] = field(default_factory=list)
    tables_skipped: list[str] = field(default_factory=list)
    orphan_details: dict[str, dict] = field(default_factory=dict)
    fkeys_applied: int = 0
    fkeys_failed: int = 0
    fkeys_pruned: int = 0

    def add_issue(self, issue: CloneIssue) -> None:
        self.issues.append(issue)

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serializable dictionary."""
        return {
            "summary": {
                "total_issues": len(self.issues),
                "errors": len([i for i in self.issues if i.severity == CloneIssueSeverity.ERROR]),
                "warnings": len([i for i in self.issues if i.severity == CloneIssueSeverity.WARNING]),
                "tables_restored": len(self.tables_restored),
                "tables_failed": len(self.tables_failed),
                "tables_skipped": len(self.tables_skipped),
                "total_rows_restored": sum(self.tables_restored.values()),
                "orphan_rows_removed": sum(
                    d.get("rows_removed", 0) for d in self.orphan_details.values()
                ),
                "orphan_rows_nullified": sum(
                    d.get("rows_nullified", 0) for d in self.orphan_details.values()
                ),
                "fkeys_applied": self.fkeys_applied,
                "fkeys_failed": self.fkeys_failed,
                "fkeys_pruned": self.fkeys_pruned,
            },
            "issues": [i.to_dict() for i in self.issues],
            "tables_restored": self.tables_restored,
            "tables_failed": self.tables_failed,
            "tables_skipped": self.tables_skipped,
            "orphan_details": self.orphan_details,
        }

    def to_json(self, indent: int = 2) -> str:
        """Return the report as a formatted JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        """Return the report as human-readable text."""
        lines = []
        lines.append("=" * 70)
        lines.append("CATALOG CLONE REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Summary
        summary = self.to_dict()["summary"]
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Tables restored:       {summary['tables_restored']}")
        lines.append(f"Tables failed:         {summary['tables_failed']}")
        lines.append(f"Tables skipped:        {summary['tables_skipped']}")
        lines.append(f"Total rows restored:   {summary['total_rows_restored']}")
        lines.append(f"Orphan rows removed:   {summary['orphan_rows_removed']}")
        lines.append(f"Orphan rows nullified: {summary['orphan_rows_nullified']}")
        lines.append(f"FKs applied:           {summary['fkeys_applied']}")
        lines.append(f"FKs failed:            {summary['fkeys_failed']}")
        lines.append(f"FKs pruned:            {summary['fkeys_pruned']}")
        lines.append(f"Errors:                {summary['errors']}")
        lines.append(f"Warnings:              {summary['warnings']}")
        lines.append("")

        # Issues by severity
        if self.issues:
            lines.append("ISSUES")
            lines.append("-" * 40)

            # Group by severity
            for severity in [CloneIssueSeverity.CRITICAL, CloneIssueSeverity.ERROR,
                           CloneIssueSeverity.WARNING, CloneIssueSeverity.INFO]:
                severity_issues = [i for i in self.issues if i.severity == severity]
                if severity_issues:
                    lines.append(f"\n{severity.value.upper()} ({len(severity_issues)}):")
                    for issue in severity_issues:
                        lines.append(f"  - {issue}")
                        if issue.details:
                            # Truncate long details
                            details = issue.details[:100] + "..." if len(issue.details) > 100 else issue.details
                            lines.append(f"    Details: {details}")
                        if issue.action:
                            lines.append(f"    Action: {issue.action}")
            lines.append("")

        # Orphan details
        if self.orphan_details:
            lines.append("ORPHAN ROW DETAILS")
            lines.append("-" * 40)
            for table, details in self.orphan_details.items():
                removed = details.get("rows_removed", 0)
                nullified = details.get("rows_nullified", 0)
                lines.append(f"  {table}:")
                if removed > 0:
                    lines.append(f"    Rows deleted: {removed}")
                if nullified > 0:
                    lines.append(f"    Rows nullified: {nullified}")
                missing = details.get("missing_references", {})
                for ref_table, count in missing.items():
                    lines.append(f"    -> missing references to {ref_table}: {count}")
            lines.append("")

        # Assessment
        lines.append("CLONE ASSESSMENT")
        lines.append("-" * 40)
        if summary['errors'] > 0:
            lines.append("Clone completed with ERRORS. Some FKs could not be applied.")
            lines.append("The catalog schema may be degraded.")
        elif summary['orphan_rows_removed'] > 0 or summary['orphan_rows_nullified'] > 0:
            lines.append("Clone completed with orphan handling.")
            lines.append("Source catalog may have incoherent row-level policies.")
        elif summary['fkeys_pruned'] > 0:
            lines.append("Clone completed with pruned FKs.")
            lines.append("Some FK constraints were skipped due to hidden reference data.")
        else:
            lines.append("Clone completed successfully.")
        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def __str__(self) -> str:
        """Return text representation of the report."""
        return self.to_text()


@dataclass
class AssetFilter:
    """Filter for selecting which assets to copy during cloning."""

    tables: list[str] | None = None
    rids: list[str] | None = None


@dataclass
class TruncatedValue:
    """Record of a value that was truncated during cloning."""

    table: str
    rid: str
    column: str
    original_bytes: int
    truncated_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "rid": self.rid,
            "column": self.column,
            "original_bytes": self.original_bytes,
            "truncated_bytes": self.truncated_bytes,
        }


@dataclass
class CloneCatalogResult:
    """Result of a catalog clone operation."""

    catalog_id: str
    hostname: str
    schema_only: bool
    asset_mode: AssetCopyMode
    source_hostname: str
    source_catalog_id: str
    source_snapshot: str | None = None
    alias: str | None = None
    ml_schema_added: bool = False
    datasets_reinitialized: int = 0
    orphan_rows_removed: int = 0
    orphan_rows_nullified: int = 0
    fkeys_pruned: int = 0
    rows_skipped: int = 0
    truncated_values: list[TruncatedValue] = field(default_factory=list)
    report: CloneReport | None = None


# Clone state annotation URL (same as deriva-py)
_clone_state_url = "tag:isrd.isi.edu,2018:clone-state"

# Catalog provenance annotation URL
_catalog_provenance_url = "tag:deriva-ml.org,2025:catalog-provenance"

# Pattern to detect btree index size errors
_BTREE_INDEX_ERROR_PATTERN = "index row size"
_BTREE_INDEX_NAME_PATTERN = r'for index "([^"]+)"'


class CatalogCreationMethod(Enum):
    """How a catalog was created."""

    CLONE = "clone"  # Cloned from another catalog
    CREATE = "create"  # Created programmatically (e.g., create_catalog)
    SCHEMA = "schema"  # Created from schema definition
    UNKNOWN = "unknown"  # Unknown or pre-existing catalog


@dataclass
class CloneDetails:
    """Details specific to cloned catalogs."""

    source_hostname: str
    source_catalog_id: str
    source_snapshot: str | None = None
    source_schema_url: str | None = None  # Hatrac URL to source schema JSON
    # Clone parameters
    orphan_strategy: str = "fail"
    truncate_oversized: bool = False
    prune_hidden_fkeys: bool = False
    schema_only: bool = False
    asset_mode: str = "refs"
    exclude_schemas: list[str] = field(default_factory=list)
    exclude_objects: list[str] = field(default_factory=list)
    add_ml_schema: bool = False
    copy_annotations: bool = True
    copy_policy: bool = True
    reinitialize_dataset_versions: bool = True
    # Statistics
    rows_copied: int = 0
    rows_skipped: int = 0
    skipped_rids: list[str] = field(default_factory=list)  # RIDs of skipped rows
    truncated_count: int = 0
    orphan_rows_removed: int = 0
    orphan_rows_nullified: int = 0
    fkeys_pruned: int = 0

    def to_dict(self) -> dict[str, Any]:
        result = {
            "source_hostname": self.source_hostname,
            "source_catalog_id": self.source_catalog_id,
            "source_snapshot": self.source_snapshot,
            "source_schema_url": self.source_schema_url,
            "orphan_strategy": self.orphan_strategy,
            "truncate_oversized": self.truncate_oversized,
            "prune_hidden_fkeys": self.prune_hidden_fkeys,
            "schema_only": self.schema_only,
            "asset_mode": self.asset_mode,
            "exclude_schemas": self.exclude_schemas,
            "exclude_objects": self.exclude_objects,
            "add_ml_schema": self.add_ml_schema,
            "copy_annotations": self.copy_annotations,
            "copy_policy": self.copy_policy,
            "reinitialize_dataset_versions": self.reinitialize_dataset_versions,
            "rows_copied": self.rows_copied,
            "rows_skipped": self.rows_skipped,
            "truncated_count": self.truncated_count,
            "orphan_rows_removed": self.orphan_rows_removed,
            "orphan_rows_nullified": self.orphan_rows_nullified,
            "fkeys_pruned": self.fkeys_pruned,
        }
        if self.skipped_rids:
            result["skipped_rids"] = self.skipped_rids
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CloneDetails":
        return cls(
            source_hostname=data.get("source_hostname", ""),
            source_catalog_id=data.get("source_catalog_id", ""),
            source_snapshot=data.get("source_snapshot"),
            source_schema_url=data.get("source_schema_url"),
            orphan_strategy=data.get("orphan_strategy", "fail"),
            truncate_oversized=data.get("truncate_oversized", False),
            prune_hidden_fkeys=data.get("prune_hidden_fkeys", False),
            schema_only=data.get("schema_only", False),
            asset_mode=data.get("asset_mode", "refs"),
            exclude_schemas=data.get("exclude_schemas", []),
            exclude_objects=data.get("exclude_objects", []),
            add_ml_schema=data.get("add_ml_schema", False),
            copy_annotations=data.get("copy_annotations", True),
            copy_policy=data.get("copy_policy", True),
            reinitialize_dataset_versions=data.get("reinitialize_dataset_versions", True),
            rows_copied=data.get("rows_copied", 0),
            rows_skipped=data.get("rows_skipped", 0),
            skipped_rids=data.get("skipped_rids", []),
            truncated_count=data.get("truncated_count", 0),
            orphan_rows_removed=data.get("orphan_rows_removed", 0),
            orphan_rows_nullified=data.get("orphan_rows_nullified", 0),
            fkeys_pruned=data.get("fkeys_pruned", 0),
        )


@dataclass
class CatalogProvenance:
    """Provenance information for a catalog.

    This metadata is stored as a catalog-level annotation and tracks
    how the catalog was created, by whom, and with what parameters.
    Supports both cloned catalogs and catalogs created by other means.

    Attributes:
        creation_method: How the catalog was created (clone, create, schema, unknown).
        created_at: ISO timestamp when the catalog was created.
        created_by: User or system that created the catalog (Globus identity or description).
        hostname: Hostname where the catalog resides.
        catalog_id: Catalog ID.
        name: Human-readable name for the catalog.
        description: Description of the catalog's purpose.
        workflow_url: URL to the workflow/script that created the catalog (e.g., GitHub URL).
        workflow_version: Version of the workflow (e.g., git commit hash, package version).
        clone_details: If cloned, detailed information about the clone operation.
    """

    creation_method: CatalogCreationMethod
    created_at: str
    hostname: str
    catalog_id: str
    created_by: str | None = None
    name: str | None = None
    description: str | None = None
    workflow_url: str | None = None
    workflow_version: str | None = None
    clone_details: CloneDetails | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "creation_method": self.creation_method.value,
            "created_at": self.created_at,
            "hostname": self.hostname,
            "catalog_id": self.catalog_id,
            "created_by": self.created_by,
            "name": self.name,
            "description": self.description,
            "workflow_url": self.workflow_url,
            "workflow_version": self.workflow_version,
        }
        if self.clone_details:
            result["clone_details"] = self.clone_details.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CatalogProvenance":
        clone_details = None
        if data.get("clone_details"):
            clone_details = CloneDetails.from_dict(data["clone_details"])

        # Handle legacy format where creation_method might be missing
        method_str = data.get("creation_method", "unknown")
        try:
            creation_method = CatalogCreationMethod(method_str)
        except ValueError:
            creation_method = CatalogCreationMethod.UNKNOWN

        return cls(
            creation_method=creation_method,
            created_at=data.get("created_at", ""),
            hostname=data.get("hostname", ""),
            catalog_id=data.get("catalog_id", ""),
            created_by=data.get("created_by"),
            name=data.get("name"),
            description=data.get("description"),
            workflow_url=data.get("workflow_url"),
            workflow_version=data.get("workflow_version"),
            clone_details=clone_details,
        )

    @property
    def is_clone(self) -> bool:
        """Return True if this catalog was cloned from another catalog."""
        return self.creation_method == CatalogCreationMethod.CLONE and self.clone_details is not None


def _upload_source_schema(
    hostname: str,
    catalog_id: str,
    schema_json: dict[str, Any],
    credential: dict | None,
) -> str | None:
    """Upload source schema JSON to Hatrac.

    Args:
        hostname: Destination catalog hostname.
        catalog_id: Destination catalog ID.
        schema_json: The source schema as a dictionary.
        credential: Credential for Hatrac access.

    Returns:
        Hatrac URL for the uploaded schema, or None if upload failed.
    """
    try:
        cred = credential or get_credential(hostname)
        hatrac = HatracStore("https", hostname, credentials=cred)

        # Create namespace for catalog provenance metadata if it doesn't exist
        namespace = f"/hatrac/catalog/{catalog_id}/provenance"
        try:
            hatrac.create_namespace(namespace, parents=True)
        except Exception:
            pass  # Namespace may already exist

        # Upload schema JSON
        schema_bytes = json.dumps(schema_json, indent=2).encode("utf-8")
        object_path = f"{namespace}/source-schema.json"

        url = hatrac.put_obj(
            object_path,
            schema_bytes,
            content_type="application/json",
        )

        logger.info(f"Uploaded source schema to {url}")
        return url

    except Exception as e:
        logger.warning(f"Failed to upload source schema to Hatrac: {e}")
        return None


def _set_catalog_provenance(
    dst_catalog: ErmrestCatalog,
    provenance: CatalogProvenance,
) -> None:
    """Set the catalog provenance annotation on a catalog.

    Args:
        dst_catalog: Catalog connection.
        provenance: Catalog provenance information.
    """
    try:
        dst_catalog.put(
            f"/annotation/{urlquote(_catalog_provenance_url)}",
            json=provenance.to_dict(),
        )
        logger.info("Set catalog provenance annotation")
    except Exception as e:
        logger.warning(f"Failed to set catalog provenance annotation: {e}")


def set_catalog_provenance(
    catalog: ErmrestCatalog,
    name: str | None = None,
    description: str | None = None,
    workflow_url: str | None = None,
    workflow_version: str | None = None,
    creation_method: CatalogCreationMethod = CatalogCreationMethod.CREATE,
) -> CatalogProvenance:
    """Set catalog provenance information for a newly created catalog.

    Use this function when creating a catalog programmatically to record
    how and why it was created. This is similar to workflow metadata but
    at the catalog level.

    Args:
        catalog: The catalog to annotate.
        name: Human-readable name for the catalog.
        description: Description of the catalog's purpose.
        workflow_url: URL to the workflow/script that created the catalog
            (e.g., GitHub URL, notebook URL).
        workflow_version: Version of the workflow (e.g., git commit hash,
            package version, or semantic version).
        creation_method: How the catalog was created. Defaults to CREATE.

    Returns:
        The CatalogProvenance object that was set.

    Example:
        >>> from deriva_ml.catalog import set_catalog_provenance, CatalogCreationMethod
        >>> provenance = set_catalog_provenance(
        ...     catalog,
        ...     name="CIFAR-10 Training Catalog",
        ...     description="Catalog for CIFAR-10 image classification experiments",
        ...     workflow_url="https://github.com/org/repo/blob/main/setup_catalog.py",
        ...     workflow_version="v1.2.0",
        ... )
    """
    # Try to get current user identity
    created_by = None
    try:
        # Get user info from catalog session
        session_info = catalog.get("/authn/session").json()
        if session_info and "client" in session_info:
            client = session_info["client"]
            created_by = client.get("display_name") or client.get("id")
    except Exception:
        pass

    # Get catalog info
    try:
        catalog_info = catalog.get("/").json()
        hostname = catalog_info.get("meta", {}).get("host", "")
        catalog_id = str(catalog.catalog_id)
    except Exception:
        hostname = ""
        catalog_id = str(catalog.catalog_id)

    provenance = CatalogProvenance(
        creation_method=creation_method,
        created_at=datetime.now(timezone.utc).isoformat(),
        hostname=hostname,
        catalog_id=catalog_id,
        created_by=created_by,
        name=name,
        description=description,
        workflow_url=workflow_url,
        workflow_version=workflow_version,
    )

    _set_catalog_provenance(catalog, provenance)
    return provenance


def get_catalog_provenance(catalog: ErmrestCatalog) -> CatalogProvenance | None:
    """Get the catalog provenance information.

    Returns provenance information if the catalog has it set. This includes
    information about how the catalog was created (clone, create, schema),
    who created it, and any workflow information.

    Args:
        catalog: The catalog to check.

    Returns:
        CatalogProvenance if available, None otherwise.
    """
    try:
        model = catalog.getCatalogModel()
        provenance_data = model.annotations.get(_catalog_provenance_url)
        if provenance_data:
            return CatalogProvenance.from_dict(provenance_data)
    except Exception as e:
        logger.debug(f"Could not get catalog provenance: {e}")

    return None


def _parse_index_error(error_msg: str) -> tuple[str | None, str | None]:
    """Parse a btree index size error to extract index name and column.

    Args:
        error_msg: The error message from ERMrest/PostgreSQL.

    Returns:
        Tuple of (index_name, column_name) if this is an index size error,
        (None, None) otherwise.
    """
    import re

    if _BTREE_INDEX_ERROR_PATTERN not in error_msg:
        return None, None

    # Extract index name from error message
    match = re.search(_BTREE_INDEX_NAME_PATTERN, error_msg)
    if not match:
        return None, None

    index_name = match.group(1)

    # Try to extract column name from index name (common pattern: table__column_idx)
    # e.g., "dataset__keywords_idx" -> "keywords"
    if "__" in index_name and index_name.endswith("_idx"):
        parts = index_name.rsplit("__", 1)
        if len(parts) == 2:
            column_name = parts[1].replace("_idx", "")
            return index_name, column_name

    return index_name, None




def _copy_table_data_with_retry(
    src_catalog: ErmrestCatalog,
    dst_catalog: ErmrestCatalog,
    sname: str,
    tname: str,
    page_size: int,
    report: "CloneReport",
    deferred_indexes: dict[str, list[dict]],
    truncate_oversized: bool = False,
) -> tuple[int, int, list[str], list[TruncatedValue]]:
    """Copy data for a single table with retry logic for index errors.

    If a btree index size error occurs, this function will:
    1. Detect the problematic index and column
    2. Switch to row-by-row insertion mode
    3. Either truncate oversized values (if truncate_oversized=True) or skip rows
    4. Record skipped/truncated rows in the report

    Args:
        src_catalog: Source catalog connection.
        dst_catalog: Destination catalog connection.
        sname: Schema name.
        tname: Table name.
        page_size: Number of rows per page.
        report: Clone report for recording issues.
        deferred_indexes: Dict to collect indexes that need rebuilding.
            Key is "schema:table", value is list of index definitions.
        truncate_oversized: If True, truncate oversized values instead of skipping rows.

    Returns:
        Tuple of (rows_copied, rows_skipped, skipped_rids, truncated_values).
        rows_copied is -1 if the copy failed entirely.
    """
    tname_uri = f"{urlquote(sname)}:{urlquote(tname)}"
    table_key = f"{sname}:{tname}"

    # Maximum safe size for btree index values (with margin below 2704 limit)
    MAX_INDEX_VALUE_BYTES = 2600
    TRUNCATE_SUFFIX = "...[TRUNCATED]"

    last = None
    table_rows = 0
    rows_skipped = 0
    skipped_rids: list[str] = []  # Track RIDs of skipped rows
    truncated_values: list[TruncatedValue] = []
    row_by_row_mode = False
    problematic_index = None
    problematic_column = None

    def truncate_row_values(row: dict, column: str | None) -> tuple[dict, list[TruncatedValue]]:
        """Truncate oversized text values in a row.

        Returns the modified row and list of truncation records.
        """
        truncations = []
        modified_row = row.copy()
        rid = row.get('RID', 'unknown')

        # If we know the problematic column, only check that one
        columns_to_check = [column] if column else list(row.keys())

        for col in columns_to_check:
            if col not in modified_row:
                continue
            value = modified_row[col]
            if isinstance(value, str):
                value_bytes = len(value.encode('utf-8'))
                if value_bytes > MAX_INDEX_VALUE_BYTES:
                    # Truncate to safe size, accounting for suffix
                    max_chars = MAX_INDEX_VALUE_BYTES - len(TRUNCATE_SUFFIX.encode('utf-8'))
                    # Be conservative - truncate by character count as approximation
                    # since UTF-8 chars can be multi-byte
                    truncated = value[:max_chars] + TRUNCATE_SUFFIX
                    # Verify the result fits
                    while len(truncated.encode('utf-8')) > MAX_INDEX_VALUE_BYTES:
                        max_chars -= 100
                        truncated = value[:max_chars] + TRUNCATE_SUFFIX

                    modified_row[col] = truncated
                    truncations.append(TruncatedValue(
                        table=table_key,
                        rid=str(rid),
                        column=col,
                        original_bytes=value_bytes,
                        truncated_bytes=len(truncated.encode('utf-8')),
                    ))
                    logger.debug(
                        f"Truncated {table_key}.{col} for RID {rid}: "
                        f"{value_bytes} -> {len(truncated.encode('utf-8'))} bytes"
                    )

        return modified_row, truncations

    while True:
        after_clause = f"@after({urlquote(last)})" if last else ""
        try:
            page = src_catalog.get(
                f"/entity/{tname_uri}@sort(RID){after_clause}?limit={page_size}"
            ).json()
        except Exception as e:
            logger.warning(f"Failed to read from {sname}:{tname}: {e}")
            return -1, rows_skipped, skipped_rids, truncated_values

        if not page:
            break

        if row_by_row_mode:
            # Insert rows one at a time, handling oversized values
            for row in page:
                row_to_insert = row

                # If truncation is enabled, try to truncate first
                if truncate_oversized and problematic_column:
                    row_to_insert, truncations = truncate_row_values(row, problematic_column)
                    truncated_values.extend(truncations)

                try:
                    dst_catalog.post(
                        f"/entity/{tname_uri}?nondefaults=RID,RCT,RCB",
                        json=[row_to_insert]
                    )
                    table_rows += 1
                except Exception as row_error:
                    error_msg = str(row_error)
                    if _BTREE_INDEX_ERROR_PATTERN in error_msg:
                        # This row has a value too large for the index
                        if truncate_oversized:
                            # Try truncating all text columns
                            row_to_insert, truncations = truncate_row_values(row, None)
                            truncated_values.extend(truncations)
                            try:
                                dst_catalog.post(
                                    f"/entity/{tname_uri}?nondefaults=RID,RCT,RCB",
                                    json=[row_to_insert]
                                )
                                table_rows += 1
                                continue
                            except Exception:
                                pass  # Fall through to skip

                        rows_skipped += 1
                        rid = row.get('RID', 'unknown')
                        skipped_rids.append(rid)
                        logger.debug(f"Skipping row {rid} in {table_key} due to index size limit")
                    else:
                        # Different error - log and skip
                        rows_skipped += 1
                        rid = row.get('RID', 'unknown')
                        skipped_rids.append(rid)
                        logger.debug(f"Skipping row {rid} in {table_key}: {row_error}")
            last = page[-1]['RID']
        else:
            # Normal batch mode
            try:
                dst_catalog.post(
                    f"/entity/{tname_uri}?nondefaults=RID,RCT,RCB",
                    json=page
                )
                last = page[-1]['RID']
                table_rows += len(page)
            except Exception as e:
                error_msg = str(e)

                # Check if this is a btree index size error
                index_name, column_name = _parse_index_error(error_msg)

                if index_name:
                    action_desc = "Values will be truncated" if truncate_oversized else "Rows with oversized values will be skipped"
                    logger.info(
                        f"Detected btree index size error for '{index_name}' on {table_key}. "
                        f"Switching to row-by-row mode. {action_desc}."
                    )
                    problematic_index = index_name
                    problematic_column = column_name
                    row_by_row_mode = True

                    # Record the issue
                    report.add_issue(CloneIssue(
                        severity=CloneIssueSeverity.WARNING,
                        category=CloneIssueCategory.INDEX_REBUILT,
                        message=f"Index '{index_name}' has oversized values, using row-by-row mode",
                        table=table_key,
                        details=f"Column '{column_name}' has values exceeding btree 2704 byte limit",
                        action=action_desc,
                    ))

                    # Retry this page in row-by-row mode
                    for row in page:
                        row_to_insert = row

                        # If truncation is enabled, try to truncate first
                        if truncate_oversized and problematic_column:
                            row_to_insert, truncations = truncate_row_values(row, problematic_column)
                            truncated_values.extend(truncations)

                        try:
                            dst_catalog.post(
                                f"/entity/{tname_uri}?nondefaults=RID,RCT,RCB",
                                json=[row_to_insert]
                            )
                            table_rows += 1
                        except Exception as row_error:
                            error_msg_row = str(row_error)
                            if _BTREE_INDEX_ERROR_PATTERN in error_msg_row:
                                # Try truncating all columns if not already done
                                if truncate_oversized:
                                    row_to_insert, truncations = truncate_row_values(row, None)
                                    truncated_values.extend(truncations)
                                    try:
                                        dst_catalog.post(
                                            f"/entity/{tname_uri}?nondefaults=RID,RCT,RCB",
                                            json=[row_to_insert]
                                        )
                                        table_rows += 1
                                        continue
                                    except Exception:
                                        pass  # Fall through to skip

                                rows_skipped += 1
                                rid = row.get('RID', 'unknown')
                                skipped_rids.append(rid)
                                logger.debug(f"Skipping row {rid} due to index size limit")
                            else:
                                rows_skipped += 1
                                rid = row.get('RID', 'unknown')
                                skipped_rids.append(rid)
                                logger.debug(f"Skipping row {rid}: {row_error}")
                    last = page[-1]['RID']
                else:
                    logger.warning(f"Failed to write to {sname}:{tname}: {e}")
                    return -1, rows_skipped, skipped_rids, truncated_values

    # Report skipped rows
    if rows_skipped > 0:
        report.add_issue(CloneIssue(
            severity=CloneIssueSeverity.WARNING,
            category=CloneIssueCategory.DATA_INTEGRITY,
            message=f"Skipped {rows_skipped} rows due to index size limits",
            table=table_key,
            details=f"Index '{problematic_index}' on column '{problematic_column}'",
            action="These rows have values too large for btree index (>2704 bytes)",
            row_count=rows_skipped,
            skipped_rids=skipped_rids if skipped_rids else None,
        ))
        logger.warning(f"Skipped {rows_skipped} rows in {table_key} due to index size limits: RIDs={skipped_rids}")

    # Report truncated values
    if truncated_values:
        report.add_issue(CloneIssue(
            severity=CloneIssueSeverity.INFO,
            category=CloneIssueCategory.DATA_INTEGRITY,
            message=f"Truncated {len(truncated_values)} values to fit index size limits",
            table=table_key,
            details=f"Values in column '{problematic_column}' were truncated to <{MAX_INDEX_VALUE_BYTES} bytes",
            action="Original data was preserved with '[TRUNCATED]' suffix",
            row_count=len(truncated_values),
        ))
        logger.info(f"Truncated {len(truncated_values)} values in {table_key}")

    return table_rows, rows_skipped, skipped_rids, truncated_values




def _rebuild_deferred_indexes(
    dst_catalog: ErmrestCatalog,
    deferred_indexes: dict[str, list[dict]],
    report: "CloneReport",
) -> None:
    """Note any indexes that had issues during data copy.

    This function is called after data copy to report on any index-related
    issues that were encountered. Since ERMrest doesn't provide direct index
    management, we can only report these issues for manual follow-up.

    Args:
        dst_catalog: Destination catalog.
        deferred_indexes: Dict of table -> list of index definitions with issues.
        report: Clone report.
    """
    if not deferred_indexes:
        return

    logger.info(f"Reporting {sum(len(v) for v in deferred_indexes.values())} index issues...")


def clone_catalog(
    source_hostname: str,
    source_catalog_id: str,
    dest_hostname: str | None = None,
    alias: str | None = None,
    add_ml_schema: bool = False,
    schema_only: bool = False,
    asset_mode: AssetCopyMode = AssetCopyMode.REFERENCES,
    asset_filter: AssetFilter | None = None,
    copy_annotations: bool = True,
    copy_policy: bool = True,
    exclude_schemas: list[str] | None = None,
    exclude_objects: list[str] | None = None,
    source_credential: dict | None = None,
    dest_credential: dict | None = None,
    reinitialize_dataset_versions: bool = True,
    orphan_strategy: OrphanStrategy = OrphanStrategy.FAIL,
    prune_hidden_fkeys: bool = False,
    truncate_oversized: bool = False,
) -> CloneCatalogResult:
    """Clone a catalog with robust handling of policy-induced FK violations.

    This function handles the common case where source catalog policies are
    incoherent - some domain tables have row-level policies hiding data, but
    referring tables don't have matching policies, leading to visible references
    to invisible rows.

    Uses a three-stage approach:
    1. Create schema WITHOUT foreign keys
    2. Copy all accessible data
    3. Apply foreign keys, handling violations based on orphan_strategy

    Args:
        source_hostname: Hostname of the source catalog server.
        source_catalog_id: ID of the catalog to clone.
        dest_hostname: Destination hostname. If None, clones to same server.
        alias: Optional alias name for the new catalog.
        add_ml_schema: If True, add the DerivaML schema to the clone.
        schema_only: If True, copy only schema structure without data.
        asset_mode: How to handle assets during cloning.
        asset_filter: Optional filter to selectively copy assets.
        copy_annotations: If True (default), copy all catalog annotations.
        copy_policy: If True (default), copy ACL policies (requires ownership).
        exclude_schemas: List of schema names to exclude from cloning.
        exclude_objects: List of specific tables to exclude ("schema:table").
        source_credential: Optional credential dict for source server.
        dest_credential: Optional credential dict for destination server.
        reinitialize_dataset_versions: If True, reset dataset versions for clone.
        orphan_strategy: How to handle rows with dangling FK references:
            - FAIL: Abort if FK violations occur (default)
            - DELETE: Delete orphan rows
            - NULLIFY: Set dangling FK values to NULL
        prune_hidden_fkeys: If True, skip FKs where referenced columns have
            "select": null rights (indicating potentially hidden data). This
            prevents FK violations but degrades schema structure.
        truncate_oversized: If True, automatically truncate text values that
            exceed PostgreSQL's btree index size limit (2704 bytes). Truncated
            values will have "...[TRUNCATED]" appended. If False (default),
            rows with oversized values are skipped. All truncations are recorded
            in the result's truncated_values list.

    Returns:
        CloneCatalogResult with details of the cloned catalog, including:
        - truncated_values: List of TruncatedValue records for any values
          that were truncated due to index size limits.
        - rows_skipped: Count of rows skipped due to index size limits
          (when truncate_oversized=False).

    Raises:
        ValueError: If invalid parameters or FK violations with FAIL strategy.

    Example:
        >>> # Clone with orphan deletion
        >>> result = clone_catalog(
        ...     "source.org", "21",
        ...     dest_hostname="localhost",
        ...     orphan_strategy=OrphanStrategy.DELETE,
        ... )

        >>> # Conservative clone that prunes problematic FKs
        >>> result = clone_catalog(
        ...     "source.org", "21",
        ...     dest_hostname="localhost",
        ...     prune_hidden_fkeys=True,
        ... )
    """
    # Determine destination
    is_same_server = dest_hostname is None or dest_hostname == source_hostname
    effective_dest_hostname = source_hostname if dest_hostname is None else dest_hostname

    # Get source snapshot for provenance
    source_snapshot = _get_catalog_snapshot(
        source_hostname, source_catalog_id, source_credential
    )

    # Connect to source
    src_cred = source_credential or get_credential(source_hostname)
    src_server = DerivaServer("https", source_hostname, credentials=src_cred)
    src_catalog = src_server.connect_ermrest(source_catalog_id)

    # Capture source schema for provenance before any modifications
    source_schema_json = src_catalog.get("/schema").json()

    # Connect to destination and create new catalog
    if is_same_server:
        dst_cred = src_cred
        dst_server = src_server
    else:
        dst_cred = dest_credential or get_credential(effective_dest_hostname)
        dst_server = DerivaServer("https", effective_dest_hostname, credentials=dst_cred)

    dst_catalog = dst_server.create_ermrest_catalog(
        name=f"Clone of {source_catalog_id}",
        description=f"Cloned from {source_hostname}:{source_catalog_id}",
    )

    report = CloneReport()

    # Track truncated values
    truncated_values: list[TruncatedValue] = []
    rows_skipped = 0

    # Record clone timestamp
    clone_timestamp = datetime.now(timezone.utc).isoformat()

    # Perform the three-stage clone
    orphan_rows_removed, orphan_rows_nullified, fkeys_pruned, rows_skipped, skipped_rids, truncated_values = _clone_three_stage(
        src_catalog=src_catalog,
        dst_catalog=dst_catalog,
        copy_data=not schema_only,
        copy_annotations=copy_annotations,
        copy_policy=copy_policy,
        exclude_schemas=exclude_schemas or [],
        exclude_objects=exclude_objects or [],
        orphan_strategy=orphan_strategy,
        prune_hidden_fkeys=prune_hidden_fkeys,
        truncate_oversized=truncate_oversized,
        report=report,
    )

    result = CloneCatalogResult(
        catalog_id=str(dst_catalog.catalog_id),
        hostname=effective_dest_hostname,
        schema_only=schema_only,
        asset_mode=asset_mode,
        source_hostname=source_hostname,
        source_catalog_id=source_catalog_id,
        source_snapshot=source_snapshot,
        orphan_rows_removed=orphan_rows_removed,
        orphan_rows_nullified=orphan_rows_nullified,
        fkeys_pruned=fkeys_pruned,
        rows_skipped=rows_skipped,
        truncated_values=truncated_values,
        report=report,
    )

    # Upload source schema to Hatrac and set catalog provenance
    source_schema_url = _upload_source_schema(
        hostname=effective_dest_hostname,
        catalog_id=result.catalog_id,
        schema_json=source_schema_json,
        credential=dst_cred,
    )

    # Calculate total rows copied from report
    total_rows_copied = sum(report.tables_restored.values())

    # Try to get current user identity
    created_by = None
    try:
        session_info = dst_catalog.get("/authn/session").json()
        if session_info and "client" in session_info:
            client = session_info["client"]
            created_by = client.get("display_name") or client.get("id")
    except Exception:
        pass

    # Create clone details
    clone_details = CloneDetails(
        source_hostname=source_hostname,
        source_catalog_id=source_catalog_id,
        source_snapshot=source_snapshot,
        source_schema_url=source_schema_url,
        orphan_strategy=orphan_strategy.value,
        truncate_oversized=truncate_oversized,
        prune_hidden_fkeys=prune_hidden_fkeys,
        schema_only=schema_only,
        asset_mode=asset_mode.value,
        exclude_schemas=exclude_schemas or [],
        exclude_objects=exclude_objects or [],
        add_ml_schema=add_ml_schema,
        copy_annotations=copy_annotations,
        copy_policy=copy_policy,
        reinitialize_dataset_versions=reinitialize_dataset_versions,
        rows_copied=total_rows_copied,
        rows_skipped=rows_skipped,
        skipped_rids=skipped_rids,
        truncated_count=len(truncated_values),
        orphan_rows_removed=orphan_rows_removed,
        orphan_rows_nullified=orphan_rows_nullified,
        fkeys_pruned=fkeys_pruned,
    )

    # Create and set catalog provenance annotation
    provenance = CatalogProvenance(
        creation_method=CatalogCreationMethod.CLONE,
        created_at=clone_timestamp,
        hostname=effective_dest_hostname,
        catalog_id=result.catalog_id,
        created_by=created_by,
        name=alias or f"Clone of {source_catalog_id}",
        description=f"Cloned from {source_hostname}:{source_catalog_id}",
        clone_details=clone_details,
    )
    _set_catalog_provenance(dst_catalog, provenance)

    # Post-clone operations
    result = _post_clone_operations(
        result=result,
        alias=alias,
        add_ml_schema=add_ml_schema,
        credential=dst_cred,
    )

    if reinitialize_dataset_versions and not schema_only:
        result = _reinitialize_dataset_versions(
            result=result,
            credential=dst_cred,
        )

    return result


def _clone_three_stage(
    src_catalog: ErmrestCatalog,
    dst_catalog: ErmrestCatalog,
    copy_data: bool,
    copy_annotations: bool,
    copy_policy: bool,
    exclude_schemas: list[str],
    exclude_objects: list[str],
    orphan_strategy: OrphanStrategy,
    prune_hidden_fkeys: bool,
    truncate_oversized: bool,
    report: CloneReport,
) -> tuple[int, int, int, int, list[str], list[TruncatedValue]]:
    """Perform three-stage catalog cloning.

    Returns: (orphan_rows_removed, orphan_rows_nullified, fkeys_pruned, rows_skipped, skipped_rids, truncated_values)
    """
    src_model = src_catalog.getCatalogModel()

    # Parse exclude_objects
    excluded_tables: set[tuple[str, str]] = set()
    for obj in exclude_objects:
        if ":" in obj:
            schema, table = obj.split(":", 1)
            excluded_tables.add((schema, table))

    # Set top-level config
    if copy_policy and src_model.acls:
        try:
            dst_catalog.put('/acl', json=src_model.acls)
        except Exception as e:
            logger.warning(f"Could not copy ACLs (may not be owner): {e}")

    if copy_annotations:
        dst_catalog.put('/annotation', json=src_model.annotations)

    # Build model content
    new_model = []
    clone_states = {}
    fkeys_deferred = []
    fkeys_pruned = 0

    def prune_parts(d, *extra_victims):
        victims = set(extra_victims)
        if not copy_annotations:
            victims |= {'annotations'}
        if not copy_policy:
            victims |= {'acls', 'acl_bindings'}
        for k in victims:
            d.pop(k, None)
        return d

    def copy_sdef(s):
        d = prune_parts(s.prejson(), 'tables')
        return d

    def copy_tdef_core(t):
        d = prune_parts(t.prejson(), 'foreign_keys')
        d['column_definitions'] = [prune_parts(c) for c in d['column_definitions']]
        d['keys'] = [prune_parts(k) for k in d.get('keys', [])]
        d.setdefault('annotations', {})[_clone_state_url] = 1 if copy_data else None
        return d

    def should_prune_fkey(fkdef, src_table):
        """Check if FK should be pruned due to hidden data."""
        if not prune_hidden_fkeys:
            return False

        # Check if referenced columns have "select": null
        for ref_col in fkdef.get('referenced_columns', []):
            ref_schema = ref_col.get('schema_name')
            ref_table = ref_col.get('table_name')
            ref_col_name = ref_col.get('column_name')

            if ref_schema and ref_table and ref_col_name:
                try:
                    ref_table_obj = src_model.schemas[ref_schema].tables[ref_table]
                    col_obj = ref_table_obj.column_definitions[ref_col_name]
                    # Check column rights
                    rights = getattr(col_obj, 'rights', None)
                    if rights and rights.get('select') is None:
                        return True
                except (KeyError, AttributeError):
                    pass
        return False

    def copy_tdef_fkeys(t, sname, tname):
        """Extract FKs, optionally pruning those with hidden references."""
        nonlocal fkeys_pruned
        fkeys = []
        for fkdef in t.prejson().get('foreign_keys', []):
            # Skip FKs to system tables
            skip = False
            for ref_col in fkdef.get('referenced_columns', []):
                if ref_col.get('schema_name') == 'public' \
                   and ref_col.get('table_name') in {'ERMrest_Client', 'ERMrest_Group', 'ERMrest_RID_Lease'}:
                    skip = True
                    break

            if skip:
                continue

            if should_prune_fkey(fkdef, t):
                fkeys_pruned += 1
                fk_name = fkdef.get('names', [[sname, 'unknown']])[0]
                report.add_issue(CloneIssue(
                    severity=CloneIssueSeverity.WARNING,
                    category=CloneIssueCategory.FK_PRUNED,
                    message=f"FK pruned due to hidden reference data",
                    table=f"{sname}:{tname}",
                    details=f"FK {fk_name} references columns with 'select': null",
                    action="Source catalog may have incoherent policies",
                ))
                continue

            fkeys.append(prune_parts(fkdef.copy()))
        return fkeys

    # Collect schemas and tables
    for sname, schema in src_model.schemas.items():
        if sname in exclude_schemas:
            continue

        new_model.append(copy_sdef(schema))

        for tname, table in schema.tables.items():
            if (sname, tname) in excluded_tables:
                report.tables_skipped.append(f"{sname}:{tname}")
                continue

            if table.kind != 'table':
                continue

            if 'RID' not in table.column_definitions.elements:
                logger.warning(f"Table {sname}.{tname} lacks system columns, skipping")
                report.tables_skipped.append(f"{sname}:{tname}")
                continue

            new_model.append(copy_tdef_core(table))
            clone_states[(sname, tname)] = 1 if copy_data else None

            # Collect FKs for deferred application
            table_fkeys = copy_tdef_fkeys(table, sname, tname)
            for fk in table_fkeys:
                fkeys_deferred.append((sname, tname, fk))

    # Stage 1: Apply schema without FKs
    logger.info("Stage 1: Creating schema without foreign keys...")
    if new_model:
        dst_catalog.post("/schema", json=new_model)

    # Stage 2: Copy data
    total_rows = 0
    total_rows_skipped = 0
    all_skipped_rids: list[str] = []
    all_truncated_values: list[TruncatedValue] = []
    deferred_indexes: dict[str, list[dict]] = {}  # Track indexes dropped for later rebuild

    if copy_data:
        logger.info("Stage 2: Copying data...")
        page_size = 10000

        for (sname, tname), state in clone_states.items():
            if state != 1:
                continue

            table_key = f"{sname}:{tname}"
            logger.debug(f"Copying data for {table_key}")

            # Use the new copy function with index error handling
            table_rows, rows_skipped, skipped_rids, truncated = _copy_table_data_with_retry(
                src_catalog=src_catalog,
                dst_catalog=dst_catalog,
                sname=sname,
                tname=tname,
                page_size=page_size,
                report=report,
                deferred_indexes=deferred_indexes,
                truncate_oversized=truncate_oversized,
            )

            total_rows_skipped += rows_skipped
            all_skipped_rids.extend(skipped_rids)
            all_truncated_values.extend(truncated)

            if table_rows < 0:
                # Copy failed
                report.tables_failed.append(table_key)
            else:
                report.tables_restored[table_key] = table_rows
                total_rows += table_rows

                # Mark complete
                try:
                    dst_catalog.put(
                        f"/schema/{urlquote(sname)}/table/{urlquote(tname)}/annotation/{urlquote(_clone_state_url)}",
                        json=2
                    )
                except Exception:
                    pass

    logger.info(f"Stage 2 complete: {total_rows} rows copied")

    # Rebuild any indexes that were dropped during data copy
    if deferred_indexes:
        _rebuild_deferred_indexes(dst_catalog, deferred_indexes, report)

    # Stage 3: Apply foreign keys
    logger.info("Stage 3: Applying foreign keys...")
    orphan_rows_removed = 0
    orphan_rows_nullified = 0

    if orphan_strategy == OrphanStrategy.DELETE:
        # For DELETE strategy, we use a three-phase approach:
        # Phase 1: Identify all FK violations without applying FKs yet
        # Phase 2: Delete orphan rows in dependency order (leaf tables first)
        # Phase 3: Apply all FKs
        # This ensures deletions aren't blocked by already-applied FKs.

        # Phase 1: Identify orphan values for each FK
        logger.info("Phase 1: Identifying orphan values...")
        fk_orphans: list[tuple[str, str, dict, set]] = []  # (sname, tname, fk, orphan_values)

        for sname, tname, fk in fkeys_deferred:
            orphan_values = _identify_orphan_values(dst_catalog, sname, tname, fk)
            if orphan_values:
                fk_orphans.append((sname, tname, fk, orphan_values))
                logger.info(f"Found {len(orphan_values)} orphan values in {sname}:{tname}")

        # Phase 2: Delete orphan rows in dependency order
        # We need to delete from "leaf" tables first (tables that reference others
        # but are not referenced themselves), then work our way up
        if fk_orphans:
            logger.info("Phase 2: Deleting orphan rows...")

            # Build a map of which tables have orphans and which tables they reference
            tables_with_orphans: set[tuple[str, str]] = set()
            table_references: dict[tuple[str, str], set[tuple[str, str]]] = {}

            for sname, tname, fk, orphan_values in fk_orphans:
                table_key = (sname, tname)
                tables_with_orphans.add(table_key)
                if table_key not in table_references:
                    table_references[table_key] = set()
                for ref_col in fk.get('referenced_columns', []):
                    ref_key = (ref_col.get('schema_name'), ref_col.get('table_name'))
                    if ref_key[0] and ref_key[1]:
                        table_references[table_key].add(ref_key)

            # Also track which tables have FKs pointing TO them
            referenced_by: dict[tuple[str, str], set[tuple[str, str]]] = {}
            for sname, tname, fk in fkeys_deferred:
                for ref_col in fk.get('referenced_columns', []):
                    ref_key = (ref_col.get('schema_name'), ref_col.get('table_name'))
                    if ref_key[0] and ref_key[1]:
                        if ref_key not in referenced_by:
                            referenced_by[ref_key] = set()
                        referenced_by[ref_key].add((sname, tname))

            # Process deletions in waves with cascading orphan detection
            # After each wave of deletions, we may have created new orphans in
            # tables that referenced the deleted rows
            max_waves = 20
            all_processed_fks: set[tuple[str, str, str]] = set()  # (schema, table, fk_name)

            for wave in range(max_waves):
                # Re-identify orphans for all FKs not yet fully processed
                current_orphans: list[tuple[str, str, dict, set]] = []

                for sname, tname, fk in fkeys_deferred:
                    fk_names = fk.get('names', [])
                    fk_id = (sname, tname, str(fk_names))
                    if fk_id in all_processed_fks:
                        continue

                    orphan_values = _identify_orphan_values(dst_catalog, sname, tname, fk)
                    if orphan_values:
                        current_orphans.append((sname, tname, fk, orphan_values))

                if not current_orphans:
                    logger.info(f"Deletion wave {wave + 1}: no more orphans found")
                    break

                logger.info(f"Deletion wave {wave + 1}: processing {len(current_orphans)} FKs with orphans")

                # Delete orphans for each FK
                wave_deleted = 0
                for sname, tname, fk, orphan_values in current_orphans:
                    removed, nullified = _delete_orphan_rows(
                        dst_catalog, sname, tname, fk, orphan_values, report
                    )
                    orphan_rows_removed += removed
                    orphan_rows_nullified += nullified
                    wave_deleted += removed

                    # Mark this FK as processed if we deleted all orphans
                    if removed == len(orphan_values):
                        fk_names = fk.get('names', [])
                        fk_id = (sname, tname, str(fk_names))
                        all_processed_fks.add(fk_id)

                if wave_deleted == 0:
                    # No deletions in this wave - might be stuck
                    logger.warning(f"Deletion wave {wave + 1}: no rows deleted, may have circular dependencies")
                    break

        # Phase 3: Apply all FKs
        logger.info("Phase 3: Applying foreign keys...")
        failed_fks = []

        for sname, tname, fk in fkeys_deferred:
            try:
                dst_catalog.post("/schema", json=[fk])
                report.fkeys_applied += 1
            except Exception as e:
                failed_fks.append((sname, tname, fk, str(e)))

        # Retry failed FKs with additional orphan cleanup
        for retry_round in range(10):
            if not failed_fks:
                break

            logger.info(f"FK retry round {retry_round + 1}: {len(failed_fks)} FKs still failing")

            # Try to clean up any remaining orphans for failed FKs
            for sname, tname, fk, last_error in failed_fks:
                orphan_values = _identify_orphan_values(dst_catalog, sname, tname, fk)
                if orphan_values:
                    removed, nullified = _delete_orphan_rows(
                        dst_catalog, sname, tname, fk, orphan_values, report
                    )
                    orphan_rows_removed += removed
                    orphan_rows_nullified += nullified

            # Try to apply the failed FKs
            still_failed = []
            for sname, tname, fk, last_error in failed_fks:
                try:
                    dst_catalog.post("/schema", json=[fk])
                    report.fkeys_applied += 1
                except Exception as e:
                    still_failed.append((sname, tname, fk, str(e)))

            if len(still_failed) == len(failed_fks):
                # No progress - stop retrying
                logger.warning(f"FK retry round {retry_round + 1}: no progress, stopping retries")
                break

            failed_fks = still_failed

        # Record final failures
        for sname, tname, fk, error_msg in failed_fks:
            report.fkeys_failed += 1
            report.add_issue(CloneIssue(
                severity=CloneIssueSeverity.ERROR,
                category=CloneIssueCategory.FK_VIOLATION,
                message="FK still failing after handling orphans",
                table=f"{sname}:{tname}",
                details=error_msg[:500],
            ))

    else:
        # For NULLIFY or FAIL strategies, use the simpler single-pass approach
        for sname, tname, fk in fkeys_deferred:
            try:
                dst_catalog.post("/schema", json=[fk])
                report.fkeys_applied += 1
            except Exception as e:
                error_msg = str(e)

                if orphan_strategy == OrphanStrategy.FAIL:
                    report.fkeys_failed += 1
                    report.add_issue(CloneIssue(
                        severity=CloneIssueSeverity.ERROR,
                        category=CloneIssueCategory.FK_VIOLATION,
                        message="FK constraint failed",
                        table=f"{sname}:{tname}",
                        details=error_msg[:500],
                        action="Use orphan_strategy=DELETE or NULLIFY to handle",
                    ))
                    continue

                # NULLIFY strategy
                removed, nullified = _handle_fk_violation(
                    dst_catalog, sname, tname, fk, orphan_strategy, report
                )
                orphan_rows_removed += removed
                orphan_rows_nullified += nullified

                # Retry FK application
                try:
                    dst_catalog.post("/schema", json=[fk])
                    report.fkeys_applied += 1
                except Exception as retry_error:
                    report.fkeys_failed += 1
                    report.add_issue(CloneIssue(
                        severity=CloneIssueSeverity.ERROR,
                        category=CloneIssueCategory.FK_VIOLATION,
                        message="FK still failing after nullifying orphans",
                        table=f"{sname}:{tname}",
                        details=str(retry_error)[:500],
                    ))

    report.fkeys_pruned = fkeys_pruned

    # Stage 3b: Copy configuration
    if copy_annotations or copy_policy:
        _copy_configuration(src_model, dst_catalog, copy_annotations, copy_policy, exclude_schemas, excluded_tables)

    return orphan_rows_removed, orphan_rows_nullified, fkeys_pruned, total_rows_skipped, all_skipped_rids, all_truncated_values


def _identify_orphan_values(
    dst_catalog: ErmrestCatalog,
    sname: str,
    tname: str,
    fk_def: dict,
) -> set:
    """Identify orphan FK values without deleting them.

    Returns: Set of values that exist in the FK column but not in the referenced table.
    """
    fk_columns = fk_def.get('foreign_key_columns', [])
    ref_columns = fk_def.get('referenced_columns', [])

    if not fk_columns or not ref_columns:
        return set()

    src_col = fk_columns[0].get('column_name')
    ref_schema = ref_columns[0].get('schema_name')
    ref_table = ref_columns[0].get('table_name')
    ref_col = ref_columns[0].get('column_name')

    if not all([src_col, ref_schema, ref_table, ref_col]):
        return set()

    src_uri = f"{urlquote(sname)}:{urlquote(tname)}"
    ref_uri = f"{urlquote(ref_schema)}:{urlquote(ref_table)}"

    try:
        src_values = dst_catalog.get(
            f"/attributegroup/{src_uri}/{urlquote(src_col)}"
        ).json()
        src_value_set = {row[src_col] for row in src_values if row.get(src_col) is not None}
    except Exception as e:
        logger.error(f"Failed to get source values for {sname}:{tname}.{src_col}: {e}")
        return set()

    try:
        ref_values = dst_catalog.get(
            f"/attributegroup/{ref_uri}/{urlquote(ref_col)}"
        ).json()
        ref_value_set = {row[ref_col] for row in ref_values if row.get(ref_col) is not None}
    except Exception as e:
        logger.error(f"Failed to get reference values for {ref_schema}:{ref_table}.{ref_col}: {e}")
        return set()

    return src_value_set - ref_value_set


def _delete_orphan_rows(
    dst_catalog: ErmrestCatalog,
    sname: str,
    tname: str,
    fk_def: dict,
    orphan_values: set,
    report: CloneReport,
) -> tuple[int, int]:
    """Delete rows with orphan FK values.

    Returns: (rows_removed, rows_nullified)
    """
    fk_columns = fk_def.get('foreign_key_columns', [])
    ref_columns = fk_def.get('referenced_columns', [])

    if not fk_columns or not ref_columns:
        return 0, 0

    src_col = fk_columns[0].get('column_name')
    ref_schema = ref_columns[0].get('schema_name')
    ref_table = ref_columns[0].get('table_name')

    src_uri = f"{urlquote(sname)}:{urlquote(tname)}"

    rows_removed = 0
    for value in orphan_values:
        encoded_value = urlquote(str(value), safe='') if isinstance(value, str) else str(value)
        try:
            dst_catalog.delete(f"/entity/{src_uri}/{urlquote(src_col)}={encoded_value}")
            rows_removed += 1
        except Exception as e:
            # Log but don't fail - the row might have been deleted by a previous operation
            # or might be blocked by another FK that will be handled later
            logger.debug(f"Could not delete {sname}:{tname} where {src_col}={value}: {e}")

    # Record in report
    if rows_removed > 0:
        table_key = f"{sname}:{tname}"
        if table_key not in report.orphan_details:
            report.orphan_details[table_key] = {
                "rows_removed": 0,
                "rows_nullified": 0,
                "missing_references": {},
            }

        report.orphan_details[table_key]["rows_removed"] += rows_removed
        ref_key = f"{ref_schema}:{ref_table}"
        report.orphan_details[table_key]["missing_references"][ref_key] = len(orphan_values)

        report.add_issue(CloneIssue(
            severity=CloneIssueSeverity.WARNING,
            category=CloneIssueCategory.ORPHAN_ROWS,
            message=f"Orphan rows deleted",
            table=table_key,
            details=f"Missing references to: {ref_key} ({len(orphan_values)})",
            action="Source catalog may have incoherent row-level policies",
            row_count=rows_removed,
        ))

    return rows_removed, 0


def _handle_fk_violation(
    dst_catalog: ErmrestCatalog,
    sname: str,
    tname: str,
    fk_def: dict,
    strategy: OrphanStrategy,
    report: CloneReport,
) -> tuple[int, int]:
    """Handle FK violation by deleting or nullifying orphan rows.

    Returns: (rows_removed, rows_nullified)
    """
    fk_columns = fk_def.get('foreign_key_columns', [])
    ref_columns = fk_def.get('referenced_columns', [])

    if not fk_columns or not ref_columns:
        return 0, 0

    src_col = fk_columns[0].get('column_name')
    ref_schema = ref_columns[0].get('schema_name')
    ref_table = ref_columns[0].get('table_name')
    ref_col = ref_columns[0].get('column_name')

    if not all([src_col, ref_schema, ref_table, ref_col]):
        return 0, 0

    src_uri = f"{urlquote(sname)}:{urlquote(tname)}"
    ref_uri = f"{urlquote(ref_schema)}:{urlquote(ref_table)}"

    # Find orphan values
    try:
        src_values = dst_catalog.get(
            f"/attributegroup/{src_uri}/{urlquote(src_col)}"
        ).json()
        src_value_set = {row[src_col] for row in src_values if row.get(src_col) is not None}
    except Exception as e:
        logger.error(f"Failed to get source values: {e}")
        return 0, 0

    try:
        ref_values = dst_catalog.get(
            f"/attributegroup/{ref_uri}/{urlquote(ref_col)}"
        ).json()
        ref_value_set = {row[ref_col] for row in ref_values if row.get(ref_col) is not None}
    except Exception as e:
        logger.error(f"Failed to get reference values: {e}")
        return 0, 0

    orphan_values = src_value_set - ref_value_set

    if not orphan_values:
        return 0, 0

    logger.info(f"Found {len(orphan_values)} orphan values in {sname}:{tname}.{src_col}")

    rows_removed = 0
    rows_nullified = 0

    for value in orphan_values:
        encoded_value = urlquote(str(value), safe='') if isinstance(value, str) else str(value)

        if strategy == OrphanStrategy.DELETE:
            try:
                dst_catalog.delete(f"/entity/{src_uri}/{urlquote(src_col)}={encoded_value}")
                rows_removed += 1
            except Exception as e:
                logger.warning(f"Failed to delete orphans for {src_col}={value}: {e}")
        elif strategy == OrphanStrategy.NULLIFY:
            try:
                # Set FK column to NULL for orphan rows
                dst_catalog.put(
                    f"/attributegroup/{src_uri}/{urlquote(src_col)}={encoded_value}/{urlquote(src_col)}",
                    json=None
                )
                rows_nullified += 1
            except Exception as e:
                logger.warning(f"Failed to nullify {src_col}={value}: {e}")

    # Record in report
    table_key = f"{sname}:{tname}"
    if table_key not in report.orphan_details:
        report.orphan_details[table_key] = {
            "rows_removed": 0,
            "rows_nullified": 0,
            "missing_references": {},
        }

    report.orphan_details[table_key]["rows_removed"] += rows_removed
    report.orphan_details[table_key]["rows_nullified"] += rows_nullified
    ref_key = f"{ref_schema}:{ref_table}"
    report.orphan_details[table_key]["missing_references"][ref_key] = len(orphan_values)

    action_taken = "deleted" if strategy == OrphanStrategy.DELETE else "nullified"
    report.add_issue(CloneIssue(
        severity=CloneIssueSeverity.WARNING,
        category=CloneIssueCategory.ORPHAN_ROWS,
        message=f"Orphan rows {action_taken}",
        table=table_key,
        details=f"Missing references to: {ref_key} ({len(orphan_values)})",
        action="Source catalog may have incoherent row-level policies",
        row_count=rows_removed + rows_nullified,
    ))

    return rows_removed, rows_nullified


def _copy_configuration(
    src_model,
    dst_catalog: ErmrestCatalog,
    copy_annotations: bool,
    copy_policy: bool,
    exclude_schemas: list[str],
    excluded_tables: set[tuple[str, str]],
) -> None:
    """Copy annotations and policies after FK application."""
    dst_model = dst_catalog.getCatalogModel()

    for sname, src_schema in src_model.schemas.items():
        if sname in exclude_schemas or sname not in dst_model.schemas:
            continue

        dst_schema = dst_model.schemas[sname]

        if copy_annotations:
            for k, v in src_schema.annotations.items():
                if k != _clone_state_url:
                    dst_schema.annotations[k] = v

        if copy_policy:
            if hasattr(dst_schema, 'acls') and hasattr(src_schema, 'acls'):
                dst_schema.acls.update(src_schema.acls)
            if hasattr(dst_schema, 'acl_bindings') and hasattr(src_schema, 'acl_bindings'):
                dst_schema.acl_bindings.update(src_schema.acl_bindings)

        for tname, src_table in src_schema.tables.items():
            if (sname, tname) in excluded_tables or tname not in dst_schema.tables:
                continue

            dst_table = dst_schema.tables[tname]

            if copy_annotations:
                for k, v in src_table.annotations.items():
                    if k != _clone_state_url:
                        dst_table.annotations[k] = v

            if copy_policy:
                if hasattr(dst_table, 'acls') and hasattr(src_table, 'acls'):
                    dst_table.acls.update(src_table.acls)
                if hasattr(dst_table, 'acl_bindings') and hasattr(src_table, 'acl_bindings'):
                    dst_table.acl_bindings.update(src_table.acl_bindings)

    try:
        dst_model.apply()
    except Exception as e:
        logger.warning(f"Failed to apply some configuration: {e}")


def _get_catalog_snapshot(
    hostname: str,
    catalog_id: str,
    credential: dict | None,
) -> str | None:
    """Get the current snapshot ID for a catalog."""
    try:
        cred = credential or get_credential(hostname)
        server = DerivaServer("https", hostname, credentials=cred)
        catalog = server.connect_ermrest(catalog_id)
        response = catalog.get("/")
        if response.status_code == 200:
            data = response.json()
            return data.get("snaptime")
    except Exception as e:
        logger.warning(f"Could not get catalog snapshot: {e}")
    return None


def _post_clone_operations(
    result: CloneCatalogResult,
    alias: str | None,
    add_ml_schema: bool,
    credential: dict | None,
) -> CloneCatalogResult:
    """Perform post-clone operations."""
    cred = credential or get_credential(result.hostname)
    server = DerivaServer("https", result.hostname, credentials=cred)

    if alias:
        try:
            server.post(
                f"/ermrest/catalog/{result.catalog_id}/alias/{urlquote(alias)}",
                json={}
            )
            result.alias = alias
        except Exception as e:
            logger.warning(f"Failed to create alias '{alias}': {e}")

    if add_ml_schema:
        try:
            from deriva_ml.schema import create_ml_schema
            catalog = server.connect_ermrest(result.catalog_id)
            create_ml_schema(catalog)
            result.ml_schema_added = True

            # Apply catalog annotations (chaise-config, navbar, etc.)
            try:
                from deriva_ml import DerivaML
                ml = DerivaML(result.hostname, result.catalog_id, check_auth=False)
                ml.apply_catalog_annotations()
                logger.info("Applied catalog annotations (chaise-config, navbar)")
            except Exception as e:
                logger.warning(f"Failed to apply catalog annotations: {e}")
                if result.report:
                    result.report.add_issue(CloneIssue(
                        severity=CloneIssueSeverity.WARNING,
                        category=CloneIssueCategory.SCHEMA_ISSUE,
                        message="Failed to apply catalog annotations",
                        details=str(e),
                        action="Manually call apply_catalog_annotations() after clone",
                    ))
        except Exception as e:
            logger.warning(f"Failed to add ML schema: {e}")
            if result.report:
                result.report.add_issue(CloneIssue(
                    severity=CloneIssueSeverity.ERROR,
                    category=CloneIssueCategory.SCHEMA_ISSUE,
                    message="Failed to add DerivaML schema",
                    details=str(e),
                    action="ML schema was not added to the clone",
                ))

    return result


def _reinitialize_dataset_versions(
    result: CloneCatalogResult,
    credential: dict | None,
) -> CloneCatalogResult:
    """Reinitialize dataset versions after cloning."""
    try:
        cred = credential or get_credential(result.hostname)
        server = DerivaServer("https", result.hostname, credentials=cred)
        catalog = server.connect_ermrest(result.catalog_id)

        model = catalog.getCatalogModel()
        if "deriva-ml" not in model.schemas:
            return result

        datasets = catalog.get("/entity/deriva-ml:Dataset").json()

        for dataset in datasets:
            try:
                rid = dataset["RID"]
                catalog.post(
                    "/entity/deriva-ml:Dataset_Version",
                    json=[{
                        "Dataset": rid,
                        "Version": "0.0.1",
                        "Description": f"Cloned from {result.source_hostname}:{result.source_catalog_id}",
                    }]
                )
                result.datasets_reinitialized += 1
            except Exception as e:
                logger.warning(f"Failed to reinitialize version for dataset {rid}: {e}")

    except Exception as e:
        logger.warning(f"Failed to reinitialize dataset versions: {e}")

    return result
