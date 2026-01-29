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

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import quote as urlquote

from deriva.core import DerivaServer, ErmrestCatalog, get_credential

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

    def to_dict(self) -> dict[str, Any]:
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
    report: CloneReport | None = None


# Clone state annotation URL (same as deriva-py)
_clone_state_url = "tag:isrd.isi.edu,2018:clone-state"


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

    Returns:
        CloneCatalogResult with details of the cloned catalog.

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

    # Perform the three-stage clone
    orphan_rows_removed, orphan_rows_nullified, fkeys_pruned = _clone_three_stage(
        src_catalog=src_catalog,
        dst_catalog=dst_catalog,
        copy_data=not schema_only,
        copy_annotations=copy_annotations,
        copy_policy=copy_policy,
        exclude_schemas=exclude_schemas or [],
        exclude_objects=exclude_objects or [],
        orphan_strategy=orphan_strategy,
        prune_hidden_fkeys=prune_hidden_fkeys,
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
        report=report,
    )

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
    report: CloneReport,
) -> tuple[int, int, int]:
    """Perform three-stage catalog cloning.

    Returns: (orphan_rows_removed, orphan_rows_nullified, fkeys_pruned)
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
    if copy_data:
        logger.info("Stage 2: Copying data...")
        page_size = 10000

        for (sname, tname), state in clone_states.items():
            if state != 1:
                continue

            tname_uri = f"{urlquote(sname)}:{urlquote(tname)}"
            logger.debug(f"Copying data for {sname}:{tname}")

            last = None
            table_rows = 0

            while True:
                after_clause = f"@after({urlquote(last)})" if last else ""
                try:
                    page = src_catalog.get(
                        f"/entity/{tname_uri}@sort(RID){after_clause}?limit={page_size}"
                    ).json()
                except Exception as e:
                    logger.warning(f"Failed to read from {sname}:{tname}: {e}")
                    report.tables_failed.append(f"{sname}:{tname}")
                    break

                if page:
                    try:
                        dst_catalog.post(
                            f"/entity/{tname_uri}?nondefaults=RID,RCT,RCB",
                            json=page
                        )
                        last = page[-1]['RID']
                        table_rows += len(page)
                    except Exception as e:
                        logger.warning(f"Failed to write to {sname}:{tname}: {e}")
                        report.tables_failed.append(f"{sname}:{tname}")
                        break
                else:
                    break

            if f"{sname}:{tname}" not in report.tables_failed:
                report.tables_restored[f"{sname}:{tname}"] = table_rows
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

    return orphan_rows_removed, orphan_rows_nullified, fkeys_pruned


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
            from deriva_ml.schema import add_ml_schema as add_schema
            catalog = server.connect_ermrest(result.catalog_id)
            add_schema(catalog)
            result.ml_schema_added = True
        except Exception as e:
            logger.warning(f"Failed to add ML schema: {e}")

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
