#!/usr/bin/env python3
"""Migrate Workflow_Type from direct FK column to association table.

This script migrates the Workflow → Workflow_Type relationship from a direct
foreign key column on the Workflow table to a many-to-many association table
(Workflow_Workflow_Type), matching the existing Dataset_Dataset_Type pattern.

Migration steps:
1. Create the Workflow_Workflow_Type association table
2. Copy existing Workflow_Type values to association rows
3. Verify all data was migrated correctly
4. Drop the Workflow_Type FK and column from the Workflow table

Usage:
    # Dry run (preview only, no changes)
    python scripts/migrate_workflow_types.py dev.eye-ai.org CATALOG_ID --dry-run

    # Execute migration
    python scripts/migrate_workflow_types.py dev.eye-ai.org CATALOG_ID

    # With custom schema name
    python scripts/migrate_workflow_types.py dev.eye-ai.org CATALOG_ID --schema deriva-ml
"""

from __future__ import annotations

import argparse
import sys

from deriva.core import ErmrestCatalog, get_credential
from deriva.core.ermrest_model import Table


def get_catalog(hostname: str, catalog_id: str) -> ErmrestCatalog:
    """Connect to an ERMrest catalog."""
    credentials = get_credential(hostname)
    return ErmrestCatalog("https", hostname, catalog_id, credentials=credentials)


def check_preconditions(catalog: ErmrestCatalog, ml_schema: str) -> dict:
    """Check the current state of the catalog and return migration info.

    Returns:
        dict with keys:
            - has_column: bool - Workflow table has Workflow_Type column
            - has_assoc: bool - Workflow_Workflow_Type table exists
            - workflow_count: int - total number of workflows
            - workflows_with_type: list - workflows with a Workflow_Type value
    """
    model = catalog.getCatalogModel()
    schema = model.schemas[ml_schema]

    # Check if association table already exists
    has_assoc = "Workflow_Workflow_Type" in schema.tables

    # Check if Workflow table still has Workflow_Type column
    workflow_table = schema.tables["Workflow"]
    has_column = "Workflow_Type" in {c.name for c in workflow_table.columns}

    # Get workflow data
    pb = catalog.getPathBuilder()
    workflow_path = pb.schemas[ml_schema].Workflow
    workflows = list(workflow_path.entities().fetch())

    workflows_with_type = []
    if has_column:
        workflows_with_type = [
            {"RID": w["RID"], "Workflow_Type": w["Workflow_Type"]}
            for w in workflows
            if w.get("Workflow_Type")
        ]

    return {
        "has_column": has_column,
        "has_assoc": has_assoc,
        "workflow_count": len(workflows),
        "workflows_with_type": workflows_with_type,
    }


def step1_create_association_table(catalog: ErmrestCatalog, ml_schema: str, dry_run: bool) -> bool:
    """Create the Workflow_Workflow_Type association table.

    Returns:
        True if table was created (or would be in dry run), False if already exists.
    """
    model = catalog.getCatalogModel()
    schema = model.schemas[ml_schema]

    if "Workflow_Workflow_Type" in schema.tables:
        print("  [SKIP] Workflow_Workflow_Type table already exists")
        return False

    workflow_table = schema.tables["Workflow"]
    workflow_type_table = schema.tables["Workflow_Type"]

    assoc_def = Table.define_association(
        associates=[
            ("Workflow", workflow_table),
            ("Workflow_Type", workflow_type_table),
        ]
    )

    if dry_run:
        print("  [DRY RUN] Would create Workflow_Workflow_Type association table")
        print("    Associates: Workflow <-> Workflow_Type")
        return True

    schema.create_table(assoc_def)
    print("  [OK] Created Workflow_Workflow_Type association table")
    return True


def step2_copy_types_to_association(
    catalog: ErmrestCatalog, ml_schema: str, workflows_with_type: list, dry_run: bool
) -> int:
    """Copy existing Workflow_Type column values to the association table.

    Returns:
        Number of association rows created.
    """
    if not workflows_with_type:
        print("  [SKIP] No workflows have Workflow_Type values to migrate")
        return 0

    pb = catalog.getPathBuilder()
    assoc_path = pb.schemas[ml_schema].Workflow_Workflow_Type

    # Check which associations already exist
    existing_assocs = set()
    try:
        for row in assoc_path.entities().fetch():
            existing_assocs.add((row["Workflow"], row["Workflow_Type"]))
    except Exception:
        pass  # Table might be empty or just created

    rows_to_insert = []
    for w in workflows_with_type:
        key = (w["RID"], w["Workflow_Type"])
        if key not in existing_assocs:
            rows_to_insert.append({
                "Workflow": w["RID"],
                "Workflow_Type": w["Workflow_Type"],
            })

    if not rows_to_insert:
        print(f"  [SKIP] All {len(workflows_with_type)} associations already exist")
        return 0

    if dry_run:
        print(f"  [DRY RUN] Would create {len(rows_to_insert)} association rows:")
        for row in rows_to_insert:
            print(f"    Workflow {row['Workflow']} -> {row['Workflow_Type']}")
        return len(rows_to_insert)

    assoc_path.insert(rows_to_insert)
    print(f"  [OK] Created {len(rows_to_insert)} association rows")
    return len(rows_to_insert)


def step3_verify_migration(
    catalog: ErmrestCatalog, ml_schema: str, workflows_with_type: list, dry_run: bool
) -> bool:
    """Verify that all workflow types were migrated to the association table.

    Returns:
        True if verification passed, False otherwise.
    """
    if dry_run:
        print(f"  [DRY RUN] Would verify {len(workflows_with_type)} workflows")
        return True

    pb = catalog.getPathBuilder()
    assoc_path = pb.schemas[ml_schema].Workflow_Workflow_Type

    # Build lookup of association rows
    assoc_rows = {}
    for row in assoc_path.entities().fetch():
        wf_rid = row["Workflow"]
        if wf_rid not in assoc_rows:
            assoc_rows[wf_rid] = set()
        assoc_rows[wf_rid].add(row["Workflow_Type"])

    errors = []
    for w in workflows_with_type:
        wf_rid = w["RID"]
        expected_type = w["Workflow_Type"]
        actual_types = assoc_rows.get(wf_rid, set())
        if expected_type not in actual_types:
            errors.append(
                f"  Workflow {wf_rid}: expected type '{expected_type}' "
                f"not found in association table (found: {actual_types})"
            )

    if errors:
        print(f"  [FAIL] Verification failed with {len(errors)} errors:")
        for err in errors:
            print(err)
        return False

    print(f"  [OK] All {len(workflows_with_type)} workflow types verified in association table")
    return True


def step4_drop_workflow_type_column(catalog: ErmrestCatalog, ml_schema: str, dry_run: bool) -> bool:
    """Drop the Workflow_Type FK constraint and column from the Workflow table.

    Returns:
        True if column was dropped, False if already gone.
    """
    model = catalog.getCatalogModel()
    schema = model.schemas[ml_schema]
    workflow_table = schema.tables["Workflow"]

    # Check if column exists
    col_names = {c.name for c in workflow_table.columns}
    if "Workflow_Type" not in col_names:
        print("  [SKIP] Workflow_Type column already removed from Workflow table")
        return False

    if dry_run:
        print("  [DRY RUN] Would drop FK constraint and Workflow_Type column from Workflow table")
        return True

    # First drop any FK constraints referencing this column
    fks_to_drop = []
    for fk in workflow_table.foreign_keys:
        if workflow_table.columns["Workflow_Type"] in fk.column_map:
            fks_to_drop.append(fk)

    for fk in fks_to_drop:
        fk_name = fk.name if hasattr(fk, "name") else str(fk)
        print(f"  Dropping FK constraint: {fk_name}")
        fk.drop()

    # Then drop the column
    workflow_table.columns["Workflow_Type"].drop()
    print("  [OK] Dropped Workflow_Type column from Workflow table")
    return True


def migrate(hostname: str, catalog_id: str, ml_schema: str = "deriva-ml", dry_run: bool = False) -> bool:
    """Run the full migration.

    Returns:
        True if migration completed successfully.
    """
    mode = "[DRY RUN] " if dry_run else ""
    print(f"\n{mode}Migrating Workflow_Type to association table")
    print(f"  Catalog: {hostname}/{catalog_id}")
    print(f"  Schema: {ml_schema}")
    print()

    catalog = get_catalog(hostname, catalog_id)

    # Check preconditions
    print("Checking preconditions...")
    info = check_preconditions(catalog, ml_schema)
    print(f"  Total workflows: {info['workflow_count']}")
    print(f"  Workflows with Workflow_Type: {len(info['workflows_with_type'])}")
    print(f"  Association table exists: {info['has_assoc']}")
    print(f"  Workflow_Type column exists: {info['has_column']}")

    if not info["has_column"] and info["has_assoc"]:
        print("\nMigration already complete! Nothing to do.")
        return True

    if not info["has_column"] and not info["has_assoc"]:
        print("\nERROR: Workflow_Type column is missing but association table doesn't exist.")
        print("The catalog is in an unexpected state.")
        return False

    # Step 1: Create association table
    print("\nStep 1: Create association table")
    step1_create_association_table(catalog, ml_schema, dry_run)

    # Step 2: Copy types to association table
    print("\nStep 2: Copy existing types to association table")
    step2_copy_types_to_association(catalog, ml_schema, info["workflows_with_type"], dry_run)

    # Step 3: Verify migration
    print("\nStep 3: Verify migration")
    if not step3_verify_migration(catalog, ml_schema, info["workflows_with_type"], dry_run):
        print("\nMigration verification FAILED. Aborting before dropping column.")
        return False

    # Step 4: Drop old column
    print("\nStep 4: Drop Workflow_Type column from Workflow table")
    step4_drop_workflow_type_column(catalog, ml_schema, dry_run)

    print(f"\n{mode}Migration complete!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Workflow_Type from direct FK column to association table"
    )
    parser.add_argument("hostname", help="Catalog hostname (e.g., dev.eye-ai.org)")
    parser.add_argument("catalog_id", help="Catalog ID")
    parser.add_argument(
        "--schema", default="deriva-ml", help="ML schema name (default: deriva-ml)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without making changes",
    )
    args = parser.parse_args()

    success = migrate(args.hostname, args.catalog_id, args.schema, args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
