#!/usr/bin/env python3
"""Consolidate workflow types into clean categorical types.

Replaces 43+ specific workflow types (e.g., "VGG19 Model Train", "RETFound Model Train")
with clean categorical types along two dimensions:

  - Operation types: Training, Testing, Prediction, Feature_Creation, Embedding,
    Visualization, Analysis, Ingest, Data_Cleaning, Dataset_Management
  - Model types: VGG19, RETFound, Multimodal

Each workflow gets reassigned from its old specific type(s) to the appropriate
categorical type(s). Old types are then deleted.

Usage:
    # Dry run (preview only, no changes)
    python scripts/consolidate_workflow_types.py www.eye-ai.org eye-ai --dry-run

    # Execute consolidation
    python scripts/consolidate_workflow_types.py www.eye-ai.org eye-ai
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from deriva.core import ErmrestCatalog, get_credential


# ── Mapping: old type name → (operation_types, model_types) ──────────────

TYPE_MAPPING: dict[str, tuple[list[str], list[str]]] = {
    # (old_type_name): ([operation_types], [model_types])
    "Test Workflow": (["Testing"], []),
    "VGG19 Model Train": (["Training"], ["VGG19"]),
    "VGG Prediction Workflow": (["Prediction"], ["VGG19"]),
    "Multimodal Workflow": (["Analysis"], ["Multimodal"]),
    "RETFound Model Train": (["Training"], ["RETFound"]),
    "Diagnosis Analysis": (["Analysis"], []),
    "Image Cropping": (["Feature_Creation"], []),
    "Analysis Notebook": (["Analysis"], []),
    "Multimodal Data Analysis": (["Analysis"], ["Multimodal"]),
    "Diagnosis": (["Feature_Creation"], []),
    "VGG19 Model Prediction": (["Prediction"], ["VGG19"]),
    "Deriva MCP": (["Deriva MCP"], []),  # Keep as-is
    "Dataset Creation": (["Dataset_Management"], []),
    "Template": (["Testing"], []),
    "Training Visualization": (["Visualization"], []),
    "VGG19 LAC DHS Quality Training": (["Training"], ["VGG19"]),
    "Multimodal Feature Engineering": (["Feature_Creation"], ["Multimodal"]),
    "RETFound Model Evaluate": (["Testing"], ["RETFound"]),
    "Template Model Script": (["Testing"], []),
    "Asset Upload": (["Ingest"], []),
    "Training": (["Training"], []),
    "Feature Creation": (["Feature_Creation"], []),
    "VGG19 Hyperparameter Tuning": (["Training"], ["VGG19"]),
    "VGG19 LAC DHS Angle Training": (["Training"], ["VGG19"]),
    "VGG19 LAC DHS Laterality Training": (["Training"], ["VGG19"]),
    "VGG19 Van Fine-Tuned Prediction": (["Prediction"], ["VGG19"]),
    "VGG19 Model Architecture Visualization": (["Visualization"], ["VGG19"]),
    "LAC Data Cleaning": (["Data_Cleaning"], []),
    "Image Grading": (["Feature_Creation"], []),
    "VGG19 Model Test": (["Testing"], ["VGG19"]),
    "Label Injection": (["Feature_Creation"], []),
    "Feature Notebook Workflow": (["Feature_Creation"], []),
    "Ingestion Workflow": (["Ingest"], []),
    "Wide Table Workflow": (["Analysis"], ["Multimodal"]),
    "VGG19 Model Script": (["Training"], ["VGG19"]),
    "Feature Extraction": (["Feature_Creation"], []),
    "CGM Analysis Notebook": (["Visualization"], []),
    "Embedding Visualization": (["Visualization"], []),
    "AIREADI Cleaning Workflow": (["Data_Cleaning"], []),
    "Data Manipulation": (["Data_Cleaning"], []),
    "AIREADI Image Processing Workflow": (["Ingest"], []),
    "Data Model Changes": (["Data Model Changes"], []),  # Keep as-is
    "AIREADI Clinic Record Processing Workflow": (["Ingest"], []),
}

# New vocabulary terms to create (if they don't already exist)
NEW_VOCAB_TERMS = {
    # Model types
    "VGG19": "VGG19 convolutional neural network for image classification.",
    "RETFound": "RETFound vision transformer (ViT-Large) foundation model for retinal images.",
    "Multimodal": "Workflows combining multiple data modalities (e.g., imaging + clinical records).",
    # Operation type additions
    "Dataset_Management": "Workflows that create, split, version, or manage datasets.",
    # Keep-as-is types that may not exist yet
    "Deriva MCP": "Operations performed through the DerivaML MCP Server.",
    "Data Model Changes": "Workflows that modify the catalog data model (schema changes).",
}


def get_catalog(hostname: str, catalog_id: str) -> ErmrestCatalog:
    """Connect to an ERMrest catalog."""
    credentials = get_credential(hostname)
    return ErmrestCatalog("https", hostname, catalog_id, credentials=credentials)


def get_existing_types(catalog: ErmrestCatalog, ml_schema: str) -> dict[str, str]:
    """Get all existing Workflow_Type terms. Returns {Name: RID}."""
    pb = catalog.getPathBuilder()
    wt_path = pb.schemas[ml_schema].Workflow_Type
    return {row["Name"]: row["RID"] for row in wt_path.entities().fetch()}


def get_all_associations(catalog: ErmrestCatalog, ml_schema: str) -> list[dict]:
    """Get all Workflow_Workflow_Type association rows."""
    pb = catalog.getPathBuilder()
    assoc_path = pb.schemas[ml_schema].Workflow_Workflow_Type
    return list(assoc_path.entities().fetch())


def step1_create_new_types(
    catalog: ErmrestCatalog, ml_schema: str, existing_types: dict[str, str], dry_run: bool
) -> dict[str, str]:
    """Create any new Workflow_Type vocabulary terms that don't exist yet.

    Returns:
        Updated {Name: RID} dict of all types.
    """
    to_create = []
    for name, desc in NEW_VOCAB_TERMS.items():
        if name not in existing_types:
            to_create.append({"Name": name, "Description": desc})

    if not to_create:
        print("  [SKIP] All new vocabulary terms already exist")
        return existing_types

    if dry_run:
        print(f"  [DRY RUN] Would create {len(to_create)} new Workflow_Type terms:")
        for t in to_create:
            print(f"    {t['Name']}: {t['Description'][:60]}...")
        return existing_types

    pb = catalog.getPathBuilder()
    wt_path = pb.schemas[ml_schema].Workflow_Type
    wt_path.insert(to_create, defaults={"ID", "URI"})
    print(f"  [OK] Created {len(to_create)} new Workflow_Type terms:")
    for t in to_create:
        print(f"    {t['Name']}")

    # Refresh types
    return get_existing_types(catalog, ml_schema)


def step2_reassign_types(
    catalog: ErmrestCatalog,
    ml_schema: str,
    existing_types: dict[str, str],
    associations: list[dict],
    dry_run: bool,
) -> tuple[int, int]:
    """Replace old type associations with new categorical types.

    For each workflow:
    1. Look up its current type(s)
    2. Map each old type to new type(s) via TYPE_MAPPING
    3. Delete old association rows
    4. Insert new association rows

    Returns:
        (rows_deleted, rows_inserted)
    """
    # Build workflow → current types mapping
    workflow_types: dict[str, set[str]] = defaultdict(set)
    for row in associations:
        workflow_types[row["Workflow"]].add(row["Workflow_Type"])

    # Build reverse lookup: type_name → RID
    type_name_by_rid = {rid: name for name, rid in existing_types.items()}

    # Compute new assignments
    rows_to_delete = []  # (Workflow_RID, old_type_name)
    rows_to_insert = []  # {"Workflow": RID, "Workflow_Type": name}
    unmapped_types = set()

    for wf_rid, old_type_names in workflow_types.items():
        new_types_for_workflow = set()

        for old_type_name in old_type_names:
            if old_type_name in TYPE_MAPPING:
                op_types, model_types = TYPE_MAPPING[old_type_name]
                new_types_for_workflow.update(op_types)
                new_types_for_workflow.update(model_types)
                # Only delete if the old type is being replaced (not kept as-is)
                if old_type_name not in new_types_for_workflow:
                    rows_to_delete.append((wf_rid, old_type_name))
            else:
                # Type not in mapping — check if it's already a standard type
                unmapped_types.add(old_type_name)

        # Add new types (skip if workflow already has them)
        for new_type in new_types_for_workflow:
            if new_type not in old_type_names:
                if new_type in existing_types:
                    rows_to_insert.append({"Workflow": wf_rid, "Workflow_Type": new_type})
                else:
                    print(f"  [WARN] Type '{new_type}' not found in vocabulary — skipping")

    if unmapped_types:
        print(f"  [INFO] {len(unmapped_types)} types not in mapping (already standard):")
        for t in sorted(unmapped_types):
            print(f"    {t}")

    print(f"  Associations to delete: {len(rows_to_delete)}")
    print(f"  Associations to insert: {len(rows_to_insert)}")

    if dry_run:
        # Show sample of changes
        print(f"\n  [DRY RUN] Sample deletions (first 10):")
        for wf_rid, old_type in rows_to_delete[:10]:
            print(f"    Workflow {wf_rid}: remove '{old_type}'")
        if len(rows_to_delete) > 10:
            print(f"    ... and {len(rows_to_delete) - 10} more")

        print(f"\n  [DRY RUN] Sample insertions (first 10):")
        for row in rows_to_insert[:10]:
            print(f"    Workflow {row['Workflow']}: add '{row['Workflow_Type']}'")
        if len(rows_to_insert) > 10:
            print(f"    ... and {len(rows_to_insert) - 10} more")

        return len(rows_to_delete), len(rows_to_insert)

    pb = catalog.getPathBuilder()
    assoc_path = pb.schemas[ml_schema].Workflow_Workflow_Type

    # Delete old associations
    deleted = 0
    for wf_rid, old_type in rows_to_delete:
        try:
            assoc_path.filter(
                (assoc_path.Workflow == wf_rid) & (assoc_path.Workflow_Type == old_type)
            ).delete()
            deleted += 1
        except Exception as e:
            print(f"  [WARN] Failed to delete ({wf_rid}, {old_type}): {e}")

    print(f"  [OK] Deleted {deleted} old association rows")

    # Insert new associations (batch insert, skip duplicates)
    if rows_to_insert:
        # Deduplicate
        seen = set()
        unique_rows = []
        for row in rows_to_insert:
            key = (row["Workflow"], row["Workflow_Type"])
            if key not in seen:
                seen.add(key)
                unique_rows.append(row)

        # Check for existing rows to avoid conflicts
        current_assocs = set()
        for row in assoc_path.entities().fetch():
            current_assocs.add((row["Workflow"], row["Workflow_Type"]))

        to_insert = [
            row for row in unique_rows
            if (row["Workflow"], row["Workflow_Type"]) not in current_assocs
        ]

        if to_insert:
            assoc_path.insert(to_insert)
            print(f"  [OK] Inserted {len(to_insert)} new association rows")
        else:
            print(f"  [SKIP] All {len(unique_rows)} new associations already exist")

    return deleted, len(rows_to_insert)


def step3_verify_assignments(
    catalog: ErmrestCatalog,
    ml_schema: str,
    existing_types: dict[str, str],
    dry_run: bool,
) -> bool:
    """Verify that all workflows now have at least one standard type."""
    if dry_run:
        print("  [DRY RUN] Would verify all workflows have standard types")
        return True

    # Standard types = NEW_VOCAB_TERMS keys + existing standard types
    standard_types = set(NEW_VOCAB_TERMS.keys()) | {
        "Training", "Testing", "Prediction", "Feature_Creation", "Embedding",
        "Visualization", "Analysis", "Ingest", "Data_Cleaning",
    }

    associations = get_all_associations(catalog, ml_schema)
    workflow_types: dict[str, set[str]] = defaultdict(set)
    for row in associations:
        workflow_types[row["Workflow"]].add(row["Workflow_Type"])

    errors = []
    for wf_rid, types in workflow_types.items():
        if not types & standard_types:
            errors.append(f"  Workflow {wf_rid} has no standard types: {types}")

    if errors:
        print(f"  [WARN] {len(errors)} workflows have no standard types:")
        for err in errors[:10]:
            print(err)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        return False

    print(f"  [OK] All {len(workflow_types)} workflows have at least one standard type")
    return True


def step4_delete_old_types(
    catalog: ErmrestCatalog,
    ml_schema: str,
    existing_types: dict[str, str],
    dry_run: bool,
) -> int:
    """Delete old workflow type terms that are no longer referenced.

    Only deletes types that:
    1. Are in the TYPE_MAPPING (i.e., were remapped)
    2. Are NOT a target type in any mapping
    3. Have no remaining association rows

    Returns:
        Number of types deleted.
    """
    # Types that are mapping targets (should be kept)
    target_types = set()
    for op_types, model_types in TYPE_MAPPING.values():
        target_types.update(op_types)
        target_types.update(model_types)

    # Types that were mapped away (candidates for deletion)
    mapped_away = set(TYPE_MAPPING.keys()) - target_types

    # Check which ones have no remaining associations
    associations = get_all_associations(catalog, ml_schema)
    types_in_use = {row["Workflow_Type"] for row in associations}

    to_delete = []
    for type_name in sorted(mapped_away):
        if type_name not in types_in_use and type_name in existing_types:
            to_delete.append(type_name)

    if not to_delete:
        print("  [SKIP] No old types to delete (all still in use or already deleted)")
        return 0

    if dry_run:
        print(f"  [DRY RUN] Would delete {len(to_delete)} old Workflow_Type terms:")
        for name in to_delete:
            print(f"    {name}")
        return len(to_delete)

    pb = catalog.getPathBuilder()
    wt_path = pb.schemas[ml_schema].Workflow_Type
    deleted = 0
    for type_name in to_delete:
        try:
            wt_path.filter(wt_path.Name == type_name).delete()
            deleted += 1
        except Exception as e:
            print(f"  [WARN] Failed to delete type '{type_name}': {e}")

    print(f"  [OK] Deleted {deleted} old Workflow_Type terms")
    return deleted


def step5_show_summary(catalog: ErmrestCatalog, ml_schema: str, dry_run: bool) -> None:
    """Show summary of final state."""
    if dry_run:
        print("  [DRY RUN] Skipping summary")
        return

    existing_types = get_existing_types(catalog, ml_schema)
    associations = get_all_associations(catalog, ml_schema)

    # Count usage per type
    type_counts: dict[str, int] = defaultdict(int)
    for row in associations:
        type_counts[row["Workflow_Type"]] += 1

    print(f"\n  Final Workflow_Type terms ({len(existing_types)}):")
    print(f"  {'Type Name':<25} {'Workflows':>10}")
    print(f"  {'-'*25} {'-'*10}")
    for name in sorted(existing_types.keys()):
        count = type_counts.get(name, 0)
        print(f"  {name:<25} {count:>10}")

    total_workflows = len({row["Workflow"] for row in associations})
    print(f"\n  Total workflows with types: {total_workflows}")
    print(f"  Total association rows: {len(associations)}")


def consolidate(
    hostname: str, catalog_id: str, ml_schema: str = "deriva-ml", dry_run: bool = False
) -> bool:
    """Run the full consolidation.

    Returns:
        True if consolidation completed successfully.
    """
    mode = "[DRY RUN] " if dry_run else ""
    print(f"\n{mode}Consolidating Workflow Types")
    print(f"  Catalog: {hostname}/{catalog_id}")
    print(f"  Schema: {ml_schema}")
    print()

    catalog = get_catalog(hostname, catalog_id)

    # Get current state
    print("Current state...")
    existing_types = get_existing_types(catalog, ml_schema)
    associations = get_all_associations(catalog, ml_schema)
    print(f"  Workflow_Type terms: {len(existing_types)}")
    print(f"  Association rows: {len(associations)}")
    print(f"  Unique workflows with types: {len({r['Workflow'] for r in associations})}")

    # Step 1: Create new vocabulary terms
    print("\nStep 1: Create new Workflow_Type vocabulary terms")
    existing_types = step1_create_new_types(catalog, ml_schema, existing_types, dry_run)

    # Step 2: Reassign workflows from old types to new types
    print("\nStep 2: Reassign workflow types")
    step2_reassign_types(catalog, ml_schema, existing_types, associations, dry_run)

    # Step 3: Verify all workflows have standard types
    print("\nStep 3: Verify assignments")
    step3_verify_assignments(catalog, ml_schema, existing_types, dry_run)

    # Step 4: Delete old types
    print("\nStep 4: Delete old Workflow_Type terms")
    step4_delete_old_types(catalog, ml_schema, existing_types, dry_run)

    # Step 5: Summary
    print("\nStep 5: Summary")
    step5_show_summary(catalog, ml_schema, dry_run)

    print(f"\n{mode}Consolidation complete!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate workflow types into clean categorical types"
    )
    parser.add_argument("hostname", help="Catalog hostname (e.g., www.eye-ai.org)")
    parser.add_argument("catalog_id", help="Catalog ID or alias (e.g., eye-ai)")
    parser.add_argument(
        "--schema", default="deriva-ml", help="ML schema name (default: deriva-ml)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying the catalog",
    )
    args = parser.parse_args()

    success = consolidate(args.hostname, args.catalog_id, args.schema, args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
