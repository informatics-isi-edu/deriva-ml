"""Enhanced catalog cloning with cross-server and selective asset support."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

from deriva.core import DerivaServer, get_credential

logger = logging.getLogger("deriva_ml")


class CloneIssueSeverity(Enum):
    """Severity level of clone issues."""

    INFO = "info"  # Informational, no action needed
    WARNING = "warning"  # May affect some functionality
    ERROR = "error"  # Data loss or integrity issue
    CRITICAL = "critical"  # Clone may be unusable


class CloneIssueCategory(Enum):
    """Category of clone issues."""

    ACCESS_DENIED = "access_denied"  # Could not access source data
    ORPHAN_ROWS = "orphan_rows"  # FK references to inaccessible rows
    DATA_INTEGRITY = "data_integrity"  # NOT NULL, constraint violations
    SCHEMA_ISSUE = "schema_issue"  # Index size, missing constraints
    RESTORE_FAILURE = "restore_failure"  # Table restore failed
    FK_VIOLATION = "fk_violation"  # Foreign key constraint violation
    CIRCULAR_DEPENDENCY = "circular_dependency"  # Circular FK references
    MISSING_DATA = "missing_data"  # Expected data file not found


@dataclass
class CloneIssue:
    """A single issue encountered during catalog cloning.

    Attributes:
        severity: How serious the issue is.
        category: What type of issue this is.
        table: The affected table (schema:table format), if applicable.
        message: Human-readable description of the issue.
        details: Additional details (e.g., specific RIDs, error messages).
        action: Suggested corrective action, if any.
        row_count: Number of rows affected, if applicable.
    """

    severity: CloneIssueSeverity
    category: CloneIssueCategory
    message: str
    table: str | None = None
    details: str | None = None
    action: str | None = None
    row_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "table": self.table,
            "details": self.details,
            "action": self.action,
            "row_count": self.row_count,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        parts = [f"[{self.severity.value.upper()}]"]
        if self.table:
            parts.append(f"{self.table}:")
        parts.append(self.message)
        if self.row_count > 0:
            parts.append(f"({self.row_count} rows)")
        return " ".join(parts)


@dataclass
class CloneReport:
    """Comprehensive report of catalog clone operation.

    Provides detailed information about what was cloned, what issues were
    encountered, and what actions may be needed.

    Attributes:
        issues: List of all issues encountered during cloning.
        tables_restored: Tables that were successfully restored with row counts.
        tables_failed: Tables that failed to restore.
        tables_skipped: Tables that were skipped (excluded or no data).
        orphan_details: Details about orphan rows by table.
    """

    issues: list[CloneIssue] = field(default_factory=list)
    tables_restored: dict[str, int] = field(default_factory=dict)
    tables_failed: list[str] = field(default_factory=list)
    tables_skipped: list[str] = field(default_factory=list)
    orphan_details: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_issue(
        self,
        severity: CloneIssueSeverity,
        category: CloneIssueCategory,
        message: str,
        table: str | None = None,
        details: str | None = None,
        action: str | None = None,
        row_count: int = 0,
    ) -> None:
        """Add an issue to the report."""
        self.issues.append(
            CloneIssue(
                severity=severity,
                category=category,
                message=message,
                table=table,
                details=details,
                action=action,
                row_count=row_count,
            )
        )

    @property
    def has_errors(self) -> bool:
        """True if any ERROR or CRITICAL issues exist."""
        return any(
            i.severity in (CloneIssueSeverity.ERROR, CloneIssueSeverity.CRITICAL)
            for i in self.issues
        )

    @property
    def has_warnings(self) -> bool:
        """True if any WARNING issues exist."""
        return any(i.severity == CloneIssueSeverity.WARNING for i in self.issues)

    @property
    def error_count(self) -> int:
        """Count of ERROR and CRITICAL issues."""
        return sum(
            1
            for i in self.issues
            if i.severity in (CloneIssueSeverity.ERROR, CloneIssueSeverity.CRITICAL)
        )

    @property
    def warning_count(self) -> int:
        """Count of WARNING issues."""
        return sum(1 for i in self.issues if i.severity == CloneIssueSeverity.WARNING)

    @property
    def total_orphan_rows(self) -> int:
        """Total number of orphan rows filtered."""
        return sum(d.get("rows_removed", 0) for d in self.orphan_details.values())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "total_issues": len(self.issues),
                "errors": self.error_count,
                "warnings": self.warning_count,
                "tables_restored": len(self.tables_restored),
                "tables_failed": len(self.tables_failed),
                "tables_skipped": len(self.tables_skipped),
                "total_rows_restored": sum(self.tables_restored.values()),
                "total_orphan_rows_filtered": self.total_orphan_rows,
            },
            "issues": [i.to_dict() for i in self.issues],
            "tables_restored": self.tables_restored,
            "tables_failed": self.tables_failed,
            "tables_skipped": self.tables_skipped,
            "orphan_details": self.orphan_details,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def __str__(self) -> str:
        """Human-readable summary report."""
        lines = []
        lines.append("=" * 70)
        lines.append("CATALOG CLONE REPORT")
        lines.append("=" * 70)

        # Summary
        lines.append("")
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Tables restored:  {len(self.tables_restored)}")
        lines.append(f"Tables failed:    {len(self.tables_failed)}")
        lines.append(f"Tables skipped:   {len(self.tables_skipped)}")
        lines.append(f"Total rows:       {sum(self.tables_restored.values())}")
        lines.append(f"Orphan rows:      {self.total_orphan_rows}")
        lines.append(f"Errors:           {self.error_count}")
        lines.append(f"Warnings:         {self.warning_count}")

        # Issues by severity
        if self.issues:
            lines.append("")
            lines.append("ISSUES")
            lines.append("-" * 40)

            # Group by severity
            for severity in [
                CloneIssueSeverity.CRITICAL,
                CloneIssueSeverity.ERROR,
                CloneIssueSeverity.WARNING,
                CloneIssueSeverity.INFO,
            ]:
                severity_issues = [i for i in self.issues if i.severity == severity]
                if severity_issues:
                    lines.append(f"\n{severity.value.upper()} ({len(severity_issues)}):")
                    for issue in severity_issues:
                        lines.append(f"  - {issue}")
                        if issue.details:
                            # Truncate long details
                            details = issue.details
                            if len(details) > 100:
                                details = details[:100] + "..."
                            lines.append(f"    Details: {details}")
                        if issue.action:
                            lines.append(f"    Action: {issue.action}")

        # Orphan details
        if self.orphan_details:
            lines.append("")
            lines.append("ORPHAN ROW DETAILS")
            lines.append("-" * 40)
            for table, details in self.orphan_details.items():
                rows = details.get("rows_removed", 0)
                refs = details.get("missing_references", {})
                lines.append(f"  {table}: {rows} rows removed")
                for ref_table, count in refs.items():
                    lines.append(f"    -> missing references to {ref_table}: {count}")

        # Failed tables
        if self.tables_failed:
            lines.append("")
            lines.append("FAILED TABLES")
            lines.append("-" * 40)
            for table in self.tables_failed:
                lines.append(f"  - {table}")

        # Integrity assessment
        lines.append("")
        lines.append("CLONE INTEGRITY ASSESSMENT")
        lines.append("-" * 40)
        if not self.has_errors and not self.has_warnings:
            lines.append("Clone completed successfully with no issues.")
        elif not self.has_errors:
            lines.append("Clone completed with warnings. Review the issues above.")
            lines.append("The clone should be functional but may have minor gaps.")
        else:
            lines.append("Clone completed with ERRORS. Data integrity may be affected.")
            lines.append("Review the errors above and consider corrective actions.")
            if self.tables_failed:
                lines.append(f"\nFailed tables ({len(self.tables_failed)}) will be empty in the clone.")
                lines.append("Tables referencing them may also have missing data.")

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)


class AssetCopyMode(Enum):
    """Asset handling mode for catalog cloning.

    Attributes:
        NONE: Don't copy assets. Asset columns will be empty in the clone.
        REFERENCES: Copy asset URLs only. Assets remain on source server's hatrac.
            Useful for cross-server cloning when assets should stay in place.
        FULL: Download assets from source and upload to destination hatrac.
            Creates a fully independent clone with all files copied.
    """

    NONE = "none"
    REFERENCES = "refs"
    FULL = "full"


@dataclass
class AssetFilter:
    """Filter for selective asset copying.

    Specify which assets to include during cloning. If both tables and rids
    are None, all assets are included (subject to asset_mode).

    Attributes:
        tables: List of asset table names to include. Only assets from these
            tables will be copied. Example: ["Image", "Model_Weights"]
        rids: List of specific asset RIDs to include. Only these exact assets
            will be copied, regardless of table.

    Note:
        If both tables and rids are specified, assets matching EITHER criterion
        are included (logical OR).
    """

    tables: list[str] | None = None
    rids: list[str] | None = None


@dataclass
class CloneCatalogResult:
    """Result of a catalog clone operation.

    Attributes:
        catalog_id: The catalog ID of the new clone.
        hostname: The hostname where the clone was created.
        alias: The alias name if one was created, None otherwise.
        ml_schema_added: True if the DerivaML schema was added to the clone.
        asset_mode: The asset copy mode used.
        schema_only: True if only schema was copied (no data).
        source_hostname: The source catalog's hostname.
        source_catalog_id: The source catalog's ID.
        source_snapshot: The source catalog's snapshot ID at clone time.
        datasets_reinitialized: Number of datasets that had their versions reinitialized.
        orphan_rows_skipped: Number of rows skipped due to missing foreign key references.
        report: Detailed report of the clone operation including all issues.
        errors: Legacy field for backward compatibility (deprecated, use report).
    """

    catalog_id: str
    hostname: str
    alias: str | None = None
    ml_schema_added: bool = False
    asset_mode: AssetCopyMode = AssetCopyMode.REFERENCES
    schema_only: bool = False
    source_hostname: str = ""
    source_catalog_id: str = ""
    source_snapshot: str = ""
    datasets_reinitialized: int = 0
    orphan_rows_skipped: int = 0
    report: CloneReport = field(default_factory=CloneReport)
    errors: list[str] = field(default_factory=list)  # Deprecated, use report

    @property
    def success(self) -> bool:
        """True if clone completed without critical errors."""
        return bool(self.catalog_id) and not self.report.has_errors

    def print_report(self) -> None:
        """Print a human-readable report to stdout."""
        print(self.report)

    def to_json(self, indent: int = 2) -> str:
        """Get full result as JSON string."""
        return json.dumps(
            {
                "catalog_id": self.catalog_id,
                "hostname": self.hostname,
                "alias": self.alias,
                "source_hostname": self.source_hostname,
                "source_catalog_id": self.source_catalog_id,
                "source_snapshot": self.source_snapshot,
                "schema_only": self.schema_only,
                "asset_mode": self.asset_mode.value,
                "ml_schema_added": self.ml_schema_added,
                "datasets_reinitialized": self.datasets_reinitialized,
                "orphan_rows_skipped": self.orphan_rows_skipped,
                "success": self.success,
                "report": self.report.to_dict(),
            },
            indent=indent,
        )


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
    ignore_acl: bool = True,
    skip_orphan_rows: bool = False,
) -> CloneCatalogResult:
    """Clone a catalog with optional cross-server support and selective asset copying.

    This function creates a clone containing only the data you have access to in
    the source catalog. Due to row-level access controls, some records may not be
    visible to you, and therefore won't be included in the clone. The clone
    represents a consistent snapshot of your accessible view of the catalog.

    This function provides flexible catalog cloning with several modes:

    **Same-server clone** (fast):
    When dest_hostname is None or matches source_hostname, uses ERMrest's native
    clone_catalog() for efficient same-server cloning.

    **Cross-server clone** (via backup/restore):
    When dest_hostname differs from source_hostname, uses DerivaBackup/DerivaRestore
    to migrate the catalog to the new server.

    **Schema-only clone**:
    Set schema_only=True to copy only the schema structure without any data.
    Useful for creating empty development/test catalogs.

    **Asset handling**:
    - NONE: Don't copy assets (asset columns will be empty)
    - REFERENCES: Keep URLs pointing to source hatrac (cross-server only)
    - FULL: Download and re-upload all assets (creates independent copy)

    **Dataset version handling**:
    Cloned catalogs do not inherit the source catalog's history. This means that
    dataset versions copied from the source reference snapshot IDs that don't
    exist in the clone, making those versions unusable for bag downloads.
    Previous versions from the source catalog are not accessible in the clone.

    By default (reinitialize_dataset_versions=True), this function automatically
    handles this by:
    1. Pruning old version history (previous versions are not accessible anyway)
    2. Incrementing the patch version for all datasets, creating new version
       records with valid snapshots in the clone's history
    3. Including a URL to the source catalog snapshot in version descriptions
       for provenance tracking (e.g., "Cloned from
       https://source.org/chaise/recordset/#21@2024-01-15T10:30:00")

    This ensures that datasets in the cloned catalog can immediately be
    downloaded as bags without manual intervention. Set
    reinitialize_dataset_versions=False to skip this behavior if you want
    to handle version updates manually.

    Args:
        source_hostname: Hostname of the source catalog server.
        source_catalog_id: ID of the catalog to clone.
        dest_hostname: Destination hostname. If None or same as source, uses
            fast same-server cloning. Otherwise uses backup/restore.
        alias: Optional alias name for the new catalog. Creates a friendly
            name like /ermrest/catalog/my-project instead of /ermrest/catalog/45.
        add_ml_schema: If True, add the DerivaML schema to the clone if not
            already present. Useful for converting plain ERMrest catalogs.
        schema_only: If True, copy only schema structure without data.
        asset_mode: How to handle assets during cloning. See AssetCopyMode.
        asset_filter: Optional filter to selectively copy only certain assets.
        copy_annotations: If True (default), copy all catalog annotations.
        copy_policy: If True (default), copy ACL policies.
        exclude_schemas: List of schema names to exclude from cloning.
        exclude_objects: List of specific tables to exclude from cloning, in
            "schema:table" format (e.g., ["isa:dataset_qc_issue"]). Use this
            to skip problematic tables that may have schema issues like missing
            key constraints. These tables will be excluded from both schema
            and data backup/restore operations.
        source_credential: Optional credential dict for source server.
            If None, uses credential from ~/.deriva/credentials.
        dest_credential: Optional credential dict for destination server.
            If None, uses credential from ~/.deriva/credentials.
        reinitialize_dataset_versions: If True (default), increment dataset
            versions after cloning. This is necessary because cloned catalogs
            don't inherit the source catalog's history, so existing dataset
            versions reference invalid snapshots (previous versions are not
            accessible). When enabled, old version history is pruned and the
            patch version is incremented with a new snapshot in the clone's
            history. The version description includes a URL to the source
            catalog snapshot for provenance.
        ignore_acl: If True (default), allow backup without catalog owner
            permissions. Set to False to require owner permissions.
        skip_orphan_rows: If True, skip rows that reference non-existent parent
            rows via foreign keys. This commonly occurs when cloning catalogs with
            row-level access controls where child records are visible but their
            parent records are access-restricted. When enabled, orphan rows are
            removed from the backup before restore, ensuring referential integrity
            in your accessible subset of the catalog. Default: False.

    Returns:
        CloneCatalogResult with details of the cloned catalog.

    Raises:
        ValueError: If invalid parameters are provided.
        Exception: If cloning fails due to permissions or connectivity.

    Examples:
        Same-server clone:
            >>> result = clone_catalog("localhost", "21")
            >>> print(f"Created clone: {result.catalog_id}")

        Cross-server clone with alias:
            >>> result = clone_catalog(
            ...     "dev.example.org", "21",
            ...     dest_hostname="prod.example.org",
            ...     alias="my-project",
            ... )

        Schema-only clone for development:
            >>> result = clone_catalog(
            ...     "prod.example.org", "21",
            ...     dest_hostname="localhost",
            ...     schema_only=True,
            ... )

        Clone with ML schema addition:
            >>> result = clone_catalog(
            ...     "source.org", "5",
            ...     dest_hostname="dest.org",
            ...     add_ml_schema=True,
            ... )

        Selective asset copying:
            >>> result = clone_catalog(
            ...     "source.org", "21",
            ...     dest_hostname="dest.org",
            ...     asset_mode=AssetCopyMode.FULL,
            ...     asset_filter=AssetFilter(tables=["Image"]),
            ... )
    """
    # Determine if this is same-server or cross-server clone
    is_same_server = dest_hostname is None or dest_hostname == source_hostname
    effective_dest_hostname = source_hostname if dest_hostname is None else dest_hostname

    # Get the source catalog's current snapshot ID for provenance tracking
    source_snapshot = _get_catalog_snapshot(
        source_hostname, source_catalog_id, source_credential
    )

    if is_same_server:
        result = _clone_same_server(
            source_hostname=source_hostname,
            source_catalog_id=source_catalog_id,
            schema_only=schema_only,
            copy_annotations=copy_annotations,
            copy_policy=copy_policy,
            exclude_schemas=exclude_schemas,
            credential=source_credential,
        )
    else:
        result = _clone_via_backup_restore(
            source_hostname=source_hostname,
            source_catalog_id=source_catalog_id,
            dest_hostname=effective_dest_hostname,
            schema_only=schema_only,
            asset_mode=asset_mode,
            asset_filter=asset_filter,
            copy_annotations=copy_annotations,
            copy_policy=copy_policy,
            exclude_schemas=exclude_schemas,
            exclude_objects=exclude_objects,
            source_credential=source_credential,
            dest_credential=dest_credential,
            ignore_acl=ignore_acl,
            skip_orphan_rows=skip_orphan_rows,
        )

    # Store source snapshot in result
    result.source_snapshot = source_snapshot

    # Post-clone operations: alias creation and ML schema addition
    result = _post_clone_operations(
        result=result,
        alias=alias,
        add_ml_schema=add_ml_schema,
        credential=dest_credential,
    )

    # Reinitialize dataset versions if requested and data was copied
    if reinitialize_dataset_versions and not schema_only:
        result = _reinitialize_dataset_versions(
            result=result,
            credential=dest_credential,
        )

    return result


def _get_catalog_snapshot(
    hostname: str,
    catalog_id: str,
    credential: dict | None,
) -> str:
    """Get the current snapshot ID of a catalog.

    Args:
        hostname: The server hostname.
        catalog_id: The catalog ID.
        credential: Optional credential dict.

    Returns:
        The current snapshot ID string, or empty string if unable to retrieve.
    """
    try:
        cred = credential or get_credential(hostname)
        server = DerivaServer("https", hostname, credentials=cred)
        catalog = server.connect_ermrest(catalog_id)
        cat_desc = catalog.get("/").json()
        return cat_desc.get("snaptime", "")
    except Exception as e:
        logger.warning(f"Could not get snapshot ID for catalog {catalog_id}: {e}")
        return ""


def _reinitialize_dataset_versions(
    result: CloneCatalogResult,
    credential: dict | None,
) -> CloneCatalogResult:
    """Reinitialize dataset versions in the cloned catalog.

    Cloned catalogs don't inherit the source catalog's history, so existing
    dataset versions reference invalid snapshot IDs. This function:
    1. Finds all datasets in the cloned catalog
    2. Prunes old version history (keeps only the latest version record per dataset)
    3. Creates new versions with valid snapshots in the clone's history
    4. Includes source catalog and snapshot info in the version description

    Args:
        result: The clone result with catalog information.
        credential: Optional credential dict for the destination server.

    Returns:
        Updated CloneCatalogResult with datasets_reinitialized count.
    """
    from tempfile import TemporaryDirectory

    try:
        from deriva_ml import DerivaML
        from deriva_ml.dataset.aux_classes import VersionPart

        with TemporaryDirectory() as tmpdir:
            # Connect to the cloned catalog
            ml = DerivaML(
                result.hostname,
                result.catalog_id,
                working_dir=tmpdir,
            )

            # Check if this is a DerivaML catalog with datasets
            model = ml.catalog.getCatalogModel()
            if "deriva-ml" not in model.schemas:
                logger.info("No DerivaML schema found, skipping dataset version reinitialization")
                return result

            ml_schema = model.schemas["deriva-ml"]
            if "Dataset" not in ml_schema.tables or "Dataset_Version" not in ml_schema.tables:
                logger.info("No Dataset tables found, skipping dataset version reinitialization")
                return result

            # Get all datasets
            pb = ml.pathBuilder()
            ml_path = pb.schemas["deriva-ml"]
            datasets = list(ml_path.tables["Dataset"].path.entities().fetch())

            if not datasets:
                logger.info("No datasets found in cloned catalog")
                return result

            # Build provenance description with URL to source catalog snapshot
            if result.source_snapshot:
                # Include full URL to the source catalog at the specific snapshot
                source_url = (
                    f"https://{result.source_hostname}/chaise/recordset/"
                    f"#{result.source_catalog_id}@{result.source_snapshot}"
                )
                provenance_desc = (
                    f"Cloned from {source_url}"
                )
            else:
                # Fallback without snapshot
                provenance_desc = (
                    f"Cloned from catalog {result.source_catalog_id} "
                    f"on {result.source_hostname}"
                )

            # Process each dataset
            datasets_processed = 0
            for dataset_record in datasets:
                try:
                    dataset_rid = dataset_record["RID"]
                    dataset = ml.lookup_dataset(dataset_rid)

                    # Prune old version history for this dataset
                    _prune_dataset_version_history(ml, dataset_rid)

                    # Create a new version with valid snapshot in clone's history
                    dataset.increment_dataset_version(
                        component=VersionPart.patch,
                        description=provenance_desc,
                    )
                    datasets_processed += 1
                    logger.debug(f"Reinitialized version for dataset {dataset_rid}")

                except Exception as e:
                    error_msg = f"Failed to reinitialize dataset {dataset_record.get('RID', 'unknown')}: {e}"
                    logger.warning(error_msg)
                    result.errors.append(error_msg)

            result.datasets_reinitialized = datasets_processed
            logger.info(f"Reinitialized versions for {datasets_processed} datasets")

    except ImportError as e:
        error_msg = f"Could not import DerivaML for dataset reinitialization: {e}"
        logger.warning(error_msg)
        result.errors.append(error_msg)
    except Exception as e:
        error_msg = f"Failed to reinitialize dataset versions: {e}"
        logger.error(error_msg)
        result.errors.append(error_msg)

    return result


def _prune_dataset_version_history(ml: Any, dataset_rid: str) -> None:
    """Prune old version history for a dataset, keeping only the latest version.

    This removes version records with invalid snapshot references from the
    source catalog while preserving the latest version number.

    Args:
        ml: DerivaML instance connected to the catalog.
        dataset_rid: RID of the dataset to prune history for.
    """
    try:
        pb = ml.pathBuilder()
        version_table = pb.schemas["deriva-ml"].tables["Dataset_Version"]

        # Get all versions for this dataset, ordered by version number descending
        versions = list(
            version_table.path
            .filter(version_table.Dataset == dataset_rid)
            .entities()
            .fetch()
        )

        if len(versions) <= 1:
            # Nothing to prune
            return

        # Sort by version to find the latest
        # Versions are semantic versions like "1.2.3"
        def parse_version(v: str) -> tuple:
            try:
                parts = v.split(".")
                return tuple(int(p) for p in parts)
            except (ValueError, AttributeError):
                return (0, 0, 0)

        versions.sort(key=lambda v: parse_version(v.get("Version", "0.0.0")), reverse=True)

        # Keep the latest, delete the rest
        versions_to_delete = versions[1:]  # All except the first (latest)

        for old_version in versions_to_delete:
            try:
                version_table.path.filter(
                    version_table.RID == old_version["RID"]
                ).delete()
                logger.debug(f"Deleted old version {old_version.get('Version')} for dataset {dataset_rid}")
            except Exception as e:
                logger.warning(f"Could not delete version {old_version.get('RID')}: {e}")

    except Exception as e:
        logger.warning(f"Could not prune version history for dataset {dataset_rid}: {e}")


def _clone_same_server(
    source_hostname: str,
    source_catalog_id: str,
    schema_only: bool,
    copy_annotations: bool,
    copy_policy: bool,
    exclude_schemas: list[str] | None,
    credential: dict | None,
) -> CloneCatalogResult:
    """Clone a catalog on the same server using native ERMrest clone.

    This is the fast path for same-server cloning.
    """
    cred = credential or get_credential(source_hostname)
    server = DerivaServer("https", source_hostname, credentials=cred)
    source_catalog = server.connect_ermrest(source_catalog_id)

    # Clone the catalog
    dest_catalog = source_catalog.clone_catalog(
        dst_catalog=None,  # Create new catalog
        copy_data=not schema_only,
        copy_annotations=copy_annotations,
        copy_policy=copy_policy,
        truncate_after=True,
        exclude_schemas=exclude_schemas,
    )

    return CloneCatalogResult(
        catalog_id=str(dest_catalog.catalog_id),
        hostname=source_hostname,
        schema_only=schema_only,
        asset_mode=AssetCopyMode.REFERENCES,  # Same-server clone preserves asset refs
        source_hostname=source_hostname,
        source_catalog_id=source_catalog_id,
    )


def _clone_via_backup_restore(
    source_hostname: str,
    source_catalog_id: str,
    dest_hostname: str,
    schema_only: bool,
    asset_mode: AssetCopyMode,
    asset_filter: AssetFilter | None,
    copy_annotations: bool,
    copy_policy: bool,
    exclude_schemas: list[str] | None,
    exclude_objects: list[str] | None,
    source_credential: dict | None,
    dest_credential: dict | None,
    ignore_acl: bool = True,
    skip_orphan_rows: bool = False,
) -> CloneCatalogResult:
    """Clone a catalog across servers using DerivaBackup/DerivaRestore."""
    from deriva.transfer.backup.deriva_backup import DerivaBackup
    from deriva.transfer.restore.deriva_restore import DerivaRestore

    errors: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        backup_dir = Path(tmpdir) / "backup"
        backup_dir.mkdir()

        # Determine asset handling for backup
        if asset_mode == AssetCopyMode.NONE:
            include_assets = None  # Don't include assets
        elif asset_mode == AssetCopyMode.REFERENCES:
            include_assets = "references"  # Store URLs only
        else:  # AssetCopyMode.FULL
            include_assets = True  # Download assets

        # Build exclude_data list - includes both schemas and specific objects
        exclude_data = list(exclude_schemas) if exclude_schemas else []
        if exclude_objects:
            exclude_data.extend(exclude_objects)

        # Step 1: Backup the source catalog
        logger.info(f"Backing up catalog {source_hostname}:{source_catalog_id}")

        backup_args: dict[str, Any] = {
            "host": source_hostname,
            "protocol": "https",
            "catalog_id": source_catalog_id,
        }

        # Add credential if provided
        if source_credential:
            backup_args["credentials"] = source_credential

        backup = DerivaBackup(
            backup_args,
            output_dir=str(backup_dir),
            no_data=schema_only,
            include_assets=include_assets,
            exclude_data=exclude_data,
            ignore_acl=ignore_acl,
        )
        backup.download()

        # Find the backup output
        bag_dirs = list(backup_dir.glob("*"))
        if not bag_dirs:
            raise RuntimeError("Backup failed: no output created")
        bag_path = bag_dirs[0]

        # Filter excluded objects from the backup schema
        if exclude_objects:
            _filter_excluded_objects_from_bag(bag_path, exclude_objects)

        # Apply asset filtering if specified
        if asset_filter and not schema_only:
            _apply_asset_filter(bag_path, asset_filter, asset_mode)

        # Create the report for tracking issues
        report = CloneReport()

        # Filter orphan rows if requested
        orphan_rows_skipped = 0
        if skip_orphan_rows and not schema_only:
            orphan_rows_skipped = _filter_orphan_rows_from_bag(
                bag_path, exclude_schemas, report
            )

        # Step 2: Restore to destination server
        logger.info(f"Restoring to {dest_hostname}")

        restore_args: dict[str, Any] = {
            "host": dest_hostname,
            "protocol": "https",
            "catalog_id": None,  # Create new catalog
        }

        if dest_credential:
            restore_args["credentials"] = dest_credential

        # Determine whether to restore assets
        no_assets = asset_mode == AssetCopyMode.NONE or asset_mode == AssetCopyMode.REFERENCES

        # Build restore exclude lists - separate schemas from specific objects
        restore_exclude_schemas = list(exclude_schemas) if exclude_schemas else []
        restore_exclude_data = list(exclude_objects) if exclude_objects else []

        restore = DerivaRestore(
            restore_args,
            input_path=str(bag_path),
            no_data=schema_only,
            no_annotations=not copy_annotations,
            no_policy=not copy_policy,
            no_assets=no_assets,
            exclude_schemas=restore_exclude_schemas,
            exclude_data=restore_exclude_data,
        )

        # Run restore and capture any errors
        restore_error = None
        try:
            restore.restore()
        except Exception as e:
            restore_error = e
            error_msg = str(e)
            logger.error(f"Restore failed: {error_msg}")

            # Parse the error to provide actionable information
            if "violates foreign key constraint" in error_msg:
                # Extract table and FK info from error message
                report.add_issue(
                    severity=CloneIssueSeverity.CRITICAL,
                    category=CloneIssueCategory.FK_VIOLATION,
                    message="Restore failed due to foreign key constraint violation",
                    details=error_msg[:500] if len(error_msg) > 500 else error_msg,
                    action="The referenced parent table may have failed to restore. "
                    "Check for earlier errors or add problematic tables to exclude_objects.",
                )
            elif "violates not-null constraint" in error_msg:
                report.add_issue(
                    severity=CloneIssueSeverity.ERROR,
                    category=CloneIssueCategory.DATA_INTEGRITY,
                    message="Restore failed due to NOT NULL constraint violation",
                    details=error_msg[:500] if len(error_msg) > 500 else error_msg,
                    action="Source data has NULL values in required columns. "
                    "Add this table to exclude_objects or fix source data.",
                )
            elif "index row size" in error_msg:
                report.add_issue(
                    severity=CloneIssueSeverity.ERROR,
                    category=CloneIssueCategory.SCHEMA_ISSUE,
                    message="Restore failed due to PostgreSQL index size limit",
                    details=error_msg[:500] if len(error_msg) > 500 else error_msg,
                    action="A column value exceeds PostgreSQL's btree index limit. "
                    "Add this table to exclude_objects.",
                )
            else:
                report.add_issue(
                    severity=CloneIssueSeverity.CRITICAL,
                    category=CloneIssueCategory.RESTORE_FAILURE,
                    message="Restore failed with unexpected error",
                    details=error_msg[:500] if len(error_msg) > 500 else error_msg,
                    action="Review the error details and check catalog configuration.",
                )

        dest_catalog_id = str(restore.dst_catalog.catalog_id) if restore.dst_catalog else ""

    result = CloneCatalogResult(
        catalog_id=dest_catalog_id,
        hostname=dest_hostname,
        schema_only=schema_only,
        asset_mode=asset_mode,
        source_hostname=source_hostname,
        source_catalog_id=source_catalog_id,
        orphan_rows_skipped=orphan_rows_skipped,
        report=report,
        errors=errors,
    )

    # Log summary
    if report.has_errors:
        logger.warning(
            f"Clone completed with {report.error_count} errors and "
            f"{report.warning_count} warnings"
        )
    elif report.has_warnings:
        logger.info(
            f"Clone completed with {report.warning_count} warnings"
        )

    return result


def _filter_excluded_objects_from_bag(
    bag_path: Path,
    exclude_objects: list[str],
) -> None:
    """Remove excluded tables from the backup bag's schema.

    Modifies the catalog-schema.json in the bag to remove tables that are
    specified in exclude_objects. This is necessary because DerivaRestore
    will fail on tables with schema issues (e.g., missing key constraints).

    Args:
        bag_path: Path to the extracted backup bag directory (or .tgz file).
        exclude_objects: List of "schema:table" strings to exclude.
    """
    import json
    import tarfile

    if not exclude_objects:
        return

    # Parse exclude_objects into a set of (schema, table) tuples
    tables_to_exclude: set[tuple[str, str]] = set()
    for obj in exclude_objects:
        if ":" in obj:
            schema, table = obj.split(":", 1)
            tables_to_exclude.add((schema, table))
        else:
            logger.warning(f"Invalid exclude_object format '{obj}', expected 'schema:table'")

    if not tables_to_exclude:
        return

    # Handle both extracted directories and .tgz files
    if bag_path.suffix == ".tgz":
        # Need to extract, modify, and repack
        import shutil

        extract_dir = bag_path.parent / f"{bag_path.stem}_extracted"
        with tarfile.open(bag_path, "r:gz") as tar:
            tar.extractall(extract_dir)

        # Find the actual bag directory inside
        inner_dirs = list(extract_dir.iterdir())
        if len(inner_dirs) == 1 and inner_dirs[0].is_dir():
            actual_bag_path = inner_dirs[0]
        else:
            actual_bag_path = extract_dir

        # Modify the schema
        _modify_schema_in_bag(actual_bag_path, tables_to_exclude)

        # Repack the bag
        bag_path.unlink()
        with tarfile.open(bag_path, "w:gz") as tar:
            for item in extract_dir.iterdir():
                tar.add(item, arcname=item.name)

        # Clean up
        shutil.rmtree(extract_dir)
    else:
        # Direct modification of extracted bag
        _modify_schema_in_bag(bag_path, tables_to_exclude)


def _modify_schema_in_bag(
    bag_path: Path,
    tables_to_exclude: set[tuple[str, str]],
) -> None:
    """Modify the catalog-schema.json to remove excluded tables.

    Args:
        bag_path: Path to the extracted bag directory.
        tables_to_exclude: Set of (schema, table) tuples to remove.
    """
    import json

    schema_file = bag_path / "data" / "catalog-schema.json"
    if not schema_file.exists():
        logger.warning(f"Schema file not found at {schema_file}")
        return

    # Load the schema
    with open(schema_file) as f:
        catalog_schema = json.load(f)

    # Remove excluded tables
    schemas = catalog_schema.get("schemas", {})
    tables_removed = []

    for schema_name, table_name in tables_to_exclude:
        if schema_name in schemas:
            schema_def = schemas[schema_name]
            tables = schema_def.get("tables", {})
            if table_name in tables:
                del tables[table_name]
                tables_removed.append(f"{schema_name}:{table_name}")
                logger.info(f"Removed table {schema_name}:{table_name} from backup schema")

    if tables_removed:
        # Write the modified schema back
        with open(schema_file, "w") as f:
            json.dump(catalog_schema, f, indent=2)
        logger.info(f"Removed {len(tables_removed)} tables from backup schema: {tables_removed}")


def _apply_asset_filter(
    bag_path: Path,
    asset_filter: AssetFilter,
    asset_mode: AssetCopyMode,
) -> None:
    """Apply asset filtering to a backup bag.

    Modifies the bag's data files to exclude assets not matching the filter.
    This is a selective operation that removes rows from asset tables or
    removes asset files from the bag's data directory.
    """
    # For now, log a warning if filtering is requested but not yet implemented
    # Full implementation would require parsing the bag's CSV files and
    # filtering based on table names or RIDs
    if asset_filter.tables or asset_filter.rids:
        logger.warning(
            "Asset filtering is specified but detailed filtering is not yet "
            "fully implemented. All assets matching the asset_mode will be included."
        )


def _filter_orphan_rows_from_bag(
    bag_path: Path,
    exclude_schemas: list[str] | None = None,
    report: CloneReport | None = None,
) -> int:
    """Filter out rows with orphan foreign key references from a backup bag.

    This function processes the backup data to remove rows that reference
    non-existent parent rows via foreign keys. This can happen when cloning
    catalogs with row-level access controls where:
    - Child table records are visible
    - Parent table records they reference are access-restricted

    The algorithm:
    1. Load the catalog schema to get foreign key definitions
    2. Topologically sort tables by FK dependencies (parents before children)
    3. Process tables in order, tracking all RIDs that exist
    4. For each child table, filter out rows referencing missing parent RIDs
    5. Write filtered data back to the bag

    Args:
        bag_path: Path to the backup bag (can be .tgz file or extracted directory).
        exclude_schemas: List of schema names to skip during processing.
        report: Optional CloneReport to populate with detailed issue information.

    Returns:
        Total number of orphan rows that were filtered out.
    """
    import shutil
    import tarfile

    if report is None:
        report = CloneReport()

    # Handle both extracted directories and .tgz files
    if bag_path.suffix == ".tgz":
        # Need to extract, modify, and repack
        extract_dir = bag_path.parent / f"{bag_path.stem}_orphan_filter"
        with tarfile.open(bag_path, "r:gz") as tar:
            tar.extractall(extract_dir)

        # Find the actual bag directory inside
        inner_dirs = list(extract_dir.iterdir())
        if len(inner_dirs) == 1 and inner_dirs[0].is_dir():
            actual_bag_path = inner_dirs[0]
        else:
            actual_bag_path = extract_dir

        # Filter orphan rows
        total_orphans_removed = _do_filter_orphan_rows(
            actual_bag_path, exclude_schemas, report
        )

        # Repack the bag
        bag_path.unlink()
        with tarfile.open(bag_path, "w:gz") as tar:
            for item in extract_dir.iterdir():
                tar.add(item, arcname=item.name)

        # Clean up
        shutil.rmtree(extract_dir)

        return total_orphans_removed
    else:
        # Direct modification of extracted bag
        return _do_filter_orphan_rows(bag_path, exclude_schemas, report)


def _do_filter_orphan_rows(
    bag_path: Path,
    exclude_schemas: list[str] | None = None,
    report: CloneReport | None = None,
) -> int:
    """Internal function to filter orphan rows from an extracted bag directory.

    Args:
        bag_path: Path to the extracted backup bag directory.
        exclude_schemas: List of schema names to skip during processing.
        report: Optional CloneReport to populate with detailed issue information.

    Returns:
        Total number of orphan rows that were filtered out.
    """
    import json
    from collections import defaultdict
    from urllib.parse import quote as urlquote

    exclude_schemas = exclude_schemas or []
    if report is None:
        report = CloneReport()
    total_orphans_removed = 0

    # Find the schema file
    schema_file = bag_path / "data" / "catalog-schema.json"
    if not schema_file.exists():
        logger.warning(f"Schema file not found at {schema_file}, skipping orphan filtering")
        return 0

    # Load the catalog schema
    with open(schema_file) as f:
        catalog_schema = json.load(f)

    schemas = catalog_schema.get("schemas", {})

    # Build foreign key dependency graph and collect FK info
    # fk_info: {(schema, table): [(fk_column, ref_schema, ref_table, ref_column), ...]}
    fk_info: dict[tuple[str, str], list[tuple[str, str, str, str]]] = defaultdict(list)
    # dependencies: {(schema, table): set of (ref_schema, ref_table)}
    dependencies: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    all_tables: set[tuple[str, str]] = set()

    for schema_name, schema_def in schemas.items():
        if schema_name in exclude_schemas:
            continue
        if schema_name == "public":
            # Skip system tables
            continue

        tables = schema_def.get("tables", {})
        for table_name, table_def in tables.items():
            table_key = (schema_name, table_name)
            all_tables.add(table_key)

            # Process foreign keys
            for fkey in table_def.get("foreign_keys", []):
                fk_columns = fkey.get("foreign_key_columns", [])
                ref_columns = fkey.get("referenced_columns", [])

                if not fk_columns or not ref_columns:
                    continue

                # Handle single-column FKs (most common case)
                # For multi-column FKs, we'd need more complex handling
                if len(fk_columns) == 1 and len(ref_columns) == 1:
                    fk_col = fk_columns[0].get("column_name")
                    ref_schema = ref_columns[0].get("schema_name")
                    ref_table = ref_columns[0].get("table_name")
                    ref_col = ref_columns[0].get("column_name")

                    # Skip references to system tables
                    if ref_schema == "public" and ref_table in {
                        "ERMrest_Client",
                        "ERMrest_Group",
                        "ERMrest_RID_Lease",
                    }:
                        continue

                    if fk_col and ref_schema and ref_table and ref_col:
                        fk_info[table_key].append((fk_col, ref_schema, ref_table, ref_col))
                        ref_key = (ref_schema, ref_table)
                        if ref_key in all_tables or ref_schema not in exclude_schemas:
                            dependencies[table_key].add(ref_key)

    # Topologically sort tables (parents before children)
    sorted_tables, circular = _topological_sort(all_tables, dependencies)

    # Report circular dependencies if found
    if circular:
        circular_tables = [f"{s}:{t}" for s, t in circular]
        logger.warning(f"Circular dependencies detected for tables: {circular_tables}")
        report.add_issue(
            severity=CloneIssueSeverity.INFO,
            category=CloneIssueCategory.CIRCULAR_DEPENDENCY,
            message=f"Circular FK dependencies detected in {len(circular)} tables",
            details=", ".join(circular_tables[:10]) + ("..." if len(circular_tables) > 10 else ""),
            action="Orphan detection may be incomplete for these tables",
        )

    # Track existing RIDs for each table
    # existing_rids: {(schema, table): set of RID values}
    existing_rids: dict[tuple[str, str], set[str]] = {}

    # Process tables in topological order
    records_dir = bag_path / "data" / "records"

    for schema_name, table_name in sorted_tables:
        table_key = (schema_name, table_name)

        # Find the data file for this table
        table_file = records_dir / urlquote(schema_name) / f"{urlquote(table_name)}.json"

        if not table_file.exists():
            # No data for this table, initialize empty RID set
            existing_rids[table_key] = set()
            continue

        # Read all rows from the table
        rows = []
        with open(table_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON line in {table_file}: {e}")

        # Get FK constraints for this table
        table_fks = fk_info.get(table_key, [])

        if not table_fks:
            # No FKs to check, just collect RIDs
            existing_rids[table_key] = {row.get("RID") for row in rows if row.get("RID")}
            continue

        # Filter rows based on FK constraints
        valid_rows = []
        orphans_removed = 0
        missing_refs_by_table: dict[str, int] = defaultdict(int)  # Track missing refs

        for row in rows:
            is_valid = True
            blocking_ref = None

            for fk_col, ref_schema, ref_table, ref_col in table_fks:
                fk_value = row.get(fk_col)

                # NULL FK values are allowed (if column is nullable)
                if fk_value is None:
                    continue

                # Check if the referenced row exists
                ref_key = (ref_schema, ref_table)
                ref_rids = existing_rids.get(ref_key, set())

                # For RID references (most common), check directly
                # For non-RID references, we'd need to track those columns too
                if ref_col == "RID":
                    if fk_value not in ref_rids:
                        is_valid = False
                        blocking_ref = f"{ref_schema}:{ref_table}"
                        logger.debug(
                            f"Orphan row in {schema_name}:{table_name} "
                            f"RID={row.get('RID')}: FK {fk_col}={fk_value} "
                            f"references missing {ref_schema}:{ref_table}"
                        )
                        break
                else:
                    # For non-RID FK references, we can't easily validate
                    # without tracking all unique key values. Skip validation.
                    pass

            if is_valid:
                valid_rows.append(row)
            else:
                orphans_removed += 1
                if blocking_ref:
                    missing_refs_by_table[blocking_ref] += 1

        # Update RID tracking with valid rows only
        existing_rids[table_key] = {row.get("RID") for row in valid_rows if row.get("RID")}

        # Write filtered data back if any rows were removed
        if orphans_removed > 0:
            total_orphans_removed += orphans_removed
            table_name_full = f"{schema_name}:{table_name}"
            logger.info(
                f"Filtered {orphans_removed} orphan rows from {table_name_full}"
            )

            # Record detailed orphan info in the report
            report.orphan_details[table_name_full] = {
                "rows_removed": orphans_removed,
                "rows_remaining": len(valid_rows),
                "missing_references": dict(missing_refs_by_table),
            }

            # Add issue to report
            ref_summary = ", ".join(
                f"{t} ({c})" for t, c in missing_refs_by_table.items()
            )
            report.add_issue(
                severity=CloneIssueSeverity.WARNING,
                category=CloneIssueCategory.ORPHAN_ROWS,
                message=f"Removed {orphans_removed} rows with missing FK references",
                table=table_name_full,
                details=f"Missing references to: {ref_summary}",
                action="These rows referenced parent records you don't have access to",
                row_count=orphans_removed,
            )

            if valid_rows:
                # Write remaining valid rows
                with open(table_file, "w", encoding="utf-8") as f:
                    for row in valid_rows:
                        f.write(json.dumps(row) + "\n")
            else:
                # All rows were orphans, delete the file to avoid restore errors
                table_file.unlink()
                logger.info(
                    f"Deleted {table_name_full} data file (all rows were orphans)"
                )
                report.add_issue(
                    severity=CloneIssueSeverity.WARNING,
                    category=CloneIssueCategory.ORPHAN_ROWS,
                    message="All rows removed (table will be empty in clone)",
                    table=table_name_full,
                    action="All data in this table referenced inaccessible parent records",
                )

    if total_orphans_removed > 0:
        logger.info(f"Total orphan rows filtered: {total_orphans_removed}")

    return total_orphans_removed


def _topological_sort(
    nodes: set[tuple[str, str]],
    dependencies: dict[tuple[str, str], set[tuple[str, str]]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Topologically sort tables so that parent tables come before children.

    Args:
        nodes: Set of (schema, table) tuples.
        dependencies: Dict mapping each table to its dependencies (tables it references).

    Returns:
        Tuple of (sorted_tables, circular_tables):
        - sorted_tables: List of (schema, table) tuples in topological order.
        - circular_tables: List of tables involved in circular dependencies.
    """
    # Kahn's algorithm
    in_degree: dict[tuple[str, str], int] = {node: 0 for node in nodes}
    dependents: dict[tuple[str, str], list[tuple[str, str]]] = {node: [] for node in nodes}

    for node, deps in dependencies.items():
        if node not in in_degree:
            continue
        for dep in deps:
            if dep in nodes:
                in_degree[node] += 1
                dependents[dep].append(node)

    # Start with nodes that have no dependencies
    queue = [node for node, degree in in_degree.items() if degree == 0]
    result = []

    while queue:
        node = queue.pop(0)
        result.append(node)

        for dependent in dependents.get(node, []):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    # Handle cycles by adding remaining nodes (shouldn't happen in valid schemas)
    circular = [node for node in nodes if node not in result]
    if circular:
        result.extend(circular)

    return result, circular


def _post_clone_operations(
    result: CloneCatalogResult,
    alias: str | None,
    add_ml_schema: bool,
    credential: dict | None,
) -> CloneCatalogResult:
    """Perform post-clone operations: alias creation and ML schema addition."""
    cred = credential or get_credential(result.hostname)
    server = DerivaServer("https", result.hostname, credentials=cred)

    # Create alias if requested
    if alias:
        try:
            server.create_ermrest_alias(
                id=alias,
                alias_target=result.catalog_id,
                name=alias,
                description=f"Alias for catalog {result.catalog_id}",
            )
            result.alias = alias
            logger.info(f"Created alias '{alias}' for catalog {result.catalog_id}")
        except Exception as e:
            error_msg = f"Failed to create alias '{alias}': {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)

    # Add ML schema if requested
    if add_ml_schema:
        try:
            from deriva_ml.schema import create_ml_schema

            dest_catalog = server.connect_ermrest(result.catalog_id)
            model = dest_catalog.getCatalogModel()

            if "deriva-ml" not in model.schemas:
                create_ml_schema(dest_catalog, schema_name="deriva-ml")
                result.ml_schema_added = True
                logger.info(f"Added DerivaML schema to catalog {result.catalog_id}")
            else:
                logger.info(f"Catalog {result.catalog_id} already has DerivaML schema")
        except Exception as e:
            error_msg = f"Failed to add ML schema: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)

    return result
