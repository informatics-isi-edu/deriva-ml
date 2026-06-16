#!/usr/bin/env python3
"""Migrate an existing catalog's ``Dataset_Execution`` to the input-only shape.

The "authorship-canonical" provenance model makes ``Dataset_Execution`` the
*input-only* edge table: each row records that an execution **consumed** a
dataset. Output edges (which execution **produced** a dataset version) live in
``Dataset_Version.Execution``, NOT here.

Historically ``create_dataset`` wrote BOTH a ``Dataset_Version`` row authored by
the producing execution AND a redundant output row into ``Dataset_Execution``.
This script brings a live catalog to the new shape, idempotently:

1. Add a nullable ``Dataset_Version`` column + FK to ``Dataset_Execution``
   (pins the exact version an input edge consumed). Skipped if present.
2. Delete the redundant **output** rows from ``Dataset_Execution``: a
   ``(Dataset, Execution)`` pair is an output edge iff some ``Dataset_Version``
   row for that dataset is authored by that execution (``Dataset_Version.Execution``
   points back). Deleting these loses no information — output provenance stays
   in ``Dataset_Version.Execution``.
3. BEST-EFFORT backfill the consumed version on each surviving **input** row by
   reading the execution's ``configuration.json`` (an ``Execution_Metadata``
   asset) and matching its ``DatasetSpec.version`` to a ``Dataset_Version`` RID.
   Rows whose config is missing/unreadable, or whose ``add_input_dataset`` call
   recorded no version, stay NULL. This is the ONLY place ``configuration.json``
   is read, and ONLY at migration time.

The migration short-circuits when there is nothing to do.

Usage:
    # Dry run (preview only, no changes)
    python scripts/migrate_dataset_execution_version.py dev.eye-ai.org CATALOG_ID --dry-run

    # Execute migration
    python scripts/migrate_dataset_execution_version.py dev.eye-ai.org CATALOG_ID

    # With custom schema name
    python scripts/migrate_dataset_execution_version.py dev.eye-ai.org CATALOG_ID --schema deriva-ml
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from deriva.core import ErmrestCatalog, get_credential

_DV_COLUMN = "Dataset_Version"
_DV_COMMENT = "RID of the Dataset_Version consumed by this input edge (NULL if unknown)."
_CONFIG_FILENAME = "configuration.json"


def get_catalog(hostname: str, catalog_id: str) -> ErmrestCatalog:
    """Connect to an ERMrest catalog."""
    credentials = get_credential(hostname)
    return ErmrestCatalog("https", hostname, catalog_id, credentials=credentials)


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------
def check_preconditions(catalog: ErmrestCatalog, ml_schema: str) -> dict:
    """Report the current state of ``Dataset_Execution``.

    Returns:
        dict with keys:
            - has_column: bool - Dataset_Execution has the Dataset_Version column
            - has_fk: bool - that column has an FK to the Dataset_Version table
            - row_count: int - number of Dataset_Execution rows
    """
    model = catalog.getCatalogModel()
    de = model.schemas[ml_schema].tables["Dataset_Execution"]

    col_names = {c.name for c in de.columns}
    has_column = _DV_COLUMN in col_names

    has_fk = has_column and any(any(c.name == _DV_COLUMN for c in fk.foreign_key_columns) for fk in de.foreign_keys)

    pb = catalog.getPathBuilder()
    de_path = pb.schemas[ml_schema].Dataset_Execution
    row_count = len(list(de_path.entities().fetch()))

    return {"has_column": has_column, "has_fk": has_fk, "row_count": row_count}


# ---------------------------------------------------------------------------
# Step 1: add the nullable Dataset_Version column + FK
# ---------------------------------------------------------------------------
def step1_add_column(catalog: ErmrestCatalog, ml_schema: str, dry_run: bool) -> bool:
    """Add the nullable ``Dataset_Version`` column + FK to ``Dataset_Execution``.

    Returns:
        True if the column was added (or would be, in dry-run); False if it
        already exists.
    """
    model = catalog.getCatalogModel()
    schema = model.schemas[ml_schema]
    de = schema.tables["Dataset_Execution"]

    if _DV_COLUMN in {c.name for c in de.columns}:
        print(f"  [SKIP] {_DV_COLUMN} column already present on Dataset_Execution")
        return False

    if dry_run:
        print(f"  [DRY-RUN] Would add nullable {_DV_COLUMN} column + FK to Dataset_Execution")
        return True

    dataset_version_table = schema.tables["Dataset_Version"]
    # The ``(base_name, nullok, target)`` tuple form of ``create_reference``
    # adds BOTH the nullable ``Dataset_Version`` column and its FK to the
    # Dataset_Version table in one step — the same idiom the fresh-catalog
    # schema uses (create_schema.py). Passing the bare table instead would
    # auto-generate a *second*, NOT-NULL column (``Dataset_Version2``).
    dv_cols, _fk = de.create_reference((_DV_COLUMN, True, dataset_version_table))
    dv_cols[0].alter(comment=_DV_COMMENT)
    print(f"  [OK] Added nullable {_DV_COLUMN} column + FK to Dataset_Execution")
    return True


# ---------------------------------------------------------------------------
# Output-edge identification
# ---------------------------------------------------------------------------
def _authored_pairs(catalog: ErmrestCatalog, ml_schema: str) -> set[tuple[str, str]]:
    """Set of ``(Dataset, Execution)`` pairs authored via ``Dataset_Version``.

    A pair is in this set iff some ``Dataset_Version`` row for that dataset has
    a non-null ``Execution`` equal to that execution. These are the output
    edges.

    DELIBERATELY broader than the runtime rule. The runtime
    ``_producer_of_dataset`` / ``list_input_datasets`` only consider the
    *latest* version's author; this migration considers *every* version's
    author. That is intentional and spec-endorsed: every version author is an
    output producer, so an execution that produced a now-superseded version
    still left a redundant output row in ``Dataset_Execution`` that must be
    de-duplicated here. Do NOT narrow this to match ``_producer_of_dataset`` —
    doing so would leave legacy output rows on superseded versions behind.
    """
    pb = catalog.getPathBuilder()
    dv_path = pb.schemas[ml_schema].Dataset_Version
    pairs: set[tuple[str, str]] = set()
    for row in dv_path.entities().fetch():
        execution = row.get("Execution")
        dataset = row.get("Dataset")
        if execution and dataset:
            pairs.add((dataset, execution))
    return pairs


# ---------------------------------------------------------------------------
# Step 2: delete redundant output rows
# ---------------------------------------------------------------------------
def step2_delete_output_rows(catalog: ErmrestCatalog, ml_schema: str, dry_run: bool) -> int:
    """Delete ``Dataset_Execution`` rows whose pair is an output edge.

    Returns:
        Number of rows deleted (or that would be, in dry-run).
    """
    pb = catalog.getPathBuilder()
    de_path = pb.schemas[ml_schema].Dataset_Execution

    authored = _authored_pairs(catalog, ml_schema)
    rows = list(de_path.entities().fetch())
    output_rows = [r for r in rows if (r.get("Dataset"), r.get("Execution")) in authored]

    if not output_rows:
        print("  [SKIP] No output rows to delete from Dataset_Execution")
        return 0

    if dry_run:
        print(f"  [DRY-RUN] Would delete {len(output_rows)} output row(s) from Dataset_Execution")
        for r in output_rows[:5]:
            print(f"    (Dataset={r.get('Dataset')}, Execution={r.get('Execution')})")
        if len(output_rows) > 5:
            print(f"    ... and {len(output_rows) - 5} more")
        return len(output_rows)

    # Delete by RID. Filtering ``RID == r["RID"]`` per row keeps the delete
    # narrow and avoids any reliance on multi-column delete predicates.
    for r in output_rows:
        de_path.filter(de_path.RID == r["RID"]).delete()
    print(f"  [OK] Deleted {len(output_rows)} output row(s) from Dataset_Execution")
    return len(output_rows)


# ---------------------------------------------------------------------------
# Step 3: best-effort backfill of the consumed version on input rows
# ---------------------------------------------------------------------------
def _config_metadata_url_for_execution(catalog: ErmrestCatalog, ml_schema: str) -> dict[str, str]:
    """Map ``execution_rid -> configuration.json hatrac URL``.

    Joins ``Execution_Metadata`` (the asset rows) to its
    ``Execution_Metadata_Execution`` association table, keeping only rows whose
    ``Filename`` is ``configuration.json``. An execution normally has exactly
    one ``configuration.json`` (written once at init time); on the rare chance
    more than one is associated, any one is picked. RID strings are not strictly
    creation-ordered, so there is no meaningful "latest" to prefer — first match
    wins.
    """
    pb = catalog.getPathBuilder()
    schema = pb.schemas[ml_schema]

    # asset RID -> URL for configuration.json assets
    config_url_by_meta: dict[str, str] = {}
    for meta in schema.Execution_Metadata.entities().fetch():
        if meta.get("Filename") == _CONFIG_FILENAME and meta.get("URL"):
            config_url_by_meta[meta["RID"]] = meta["URL"]

    if not config_url_by_meta:
        return {}

    url_by_exec: dict[str, str] = {}
    for assoc in schema.Execution_Metadata_Execution.entities().fetch():
        meta_rid = assoc.get("Execution_Metadata")
        exec_rid = assoc.get("Execution")
        if exec_rid and meta_rid in config_url_by_meta:
            # First config.json seen for this execution wins; we don't try to
            # rank multiple configs since RIDs aren't strictly creation-ordered.
            url_by_exec.setdefault(exec_rid, config_url_by_meta[meta_rid])
    return url_by_exec


def _version_rid_index(catalog: ErmrestCatalog, ml_schema: str) -> dict[tuple[str, str], str]:
    """Map ``(Dataset, Version-string) -> Dataset_Version RID``."""
    pb = catalog.getPathBuilder()
    index: dict[tuple[str, str], str] = {}
    for row in pb.schemas[ml_schema].Dataset_Version.entities().fetch():
        dataset = row.get("Dataset")
        version = row.get("Version")
        if dataset and version:
            index[(dataset, str(version))] = row["RID"]
    return index


def _read_consumed_versions(hostname: str, url: str) -> dict[str, str]:
    """Download + parse a ``configuration.json`` from hatrac.

    Returns:
        ``{dataset_rid: version_string}`` for every ``DatasetSpec`` whose
        version serializes to a non-empty string. Empty dict on any failure
        (the caller treats this as "config unreadable").
    """
    # Local import: HatracStore carries network deps. Mirrors deriva_ml's own
    # lazy-import idiom in asset_upload.download_asset.
    from deriva.core.hatrac_store import HatracStore

    from deriva_ml.execution.execution_configuration import ExecutionConfiguration

    credential = get_credential(hostname)
    hs = HatracStore("https", hostname, credentials=credential)
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / _CONFIG_FILENAME
        hs.get_obj(path=url, destfilename=dest.as_posix())
        config = ExecutionConfiguration.load_configuration(dest)

    consumed: dict[str, str] = {}
    for spec in config.datasets:
        # DatasetSpec.version is a DatasetVersion; its str() is the same
        # PEP-440 form stored in Dataset_Version.Version (e.g. "0.1.0").
        version_str = str(spec.version)
        if spec.rid and version_str:
            consumed[spec.rid] = version_str
    return consumed


def step3_backfill_input_versions(catalog: ErmrestCatalog, hostname: str, ml_schema: str, dry_run: bool) -> dict:
    """Best-effort backfill of ``Dataset_Version`` on surviving input rows.

    For each remaining ``Dataset_Execution`` row that has no ``Dataset_Version``
    yet, resolve the consumed version from the execution's ``configuration.json``
    (a ``DatasetSpec.version`` whose ``rid`` matches the row's ``Dataset``) and
    set the row's ``Dataset_Version`` to the matching ``Dataset_Version`` RID.

    Each row is wrapped in try/except so one unreadable config never aborts the
    run. Rows that cannot be resolved stay NULL.

    Returns:
        dict with counts:
            - filled: rows set to a Dataset_Version RID
            - null_config: rows whose execution has no readable configuration.json
            - null_no_record: rows whose config has no matching DatasetSpec/version
            - errors: rows skipped due to an unexpected internal failure
              (distinct from "config unreadable" so a summary doesn't mislead)
    """
    pb = catalog.getPathBuilder()
    de_path = pb.schemas[ml_schema].Dataset_Execution

    # Only consider rows that still need a value.
    rows = [r for r in de_path.entities().fetch() if r.get("Dataset") and not r.get(_DV_COLUMN)]

    counts = {"filled": 0, "null_config": 0, "null_no_record": 0, "errors": 0}
    if not rows:
        print("  [SKIP] No input rows need version backfill")
        return counts

    url_by_exec = _config_metadata_url_for_execution(catalog, ml_schema)
    version_index = _version_rid_index(catalog, ml_schema)
    # Cache parsed configs per execution so we download each config once.
    consumed_cache: dict[str, dict[str, str] | None] = {}

    updates: list[dict[str, str]] = []
    for r in rows:
        try:
            exec_rid = r.get("Execution")
            dataset_rid = r.get("Dataset")
            url = url_by_exec.get(exec_rid)
            if not url:
                counts["null_config"] += 1
                continue

            if exec_rid not in consumed_cache:
                try:
                    consumed_cache[exec_rid] = _read_consumed_versions(hostname, url)
                except Exception as exc:  # noqa: BLE001 - one bad config must not abort
                    print(f"    [WARN] Could not read config for execution {exec_rid}: {exc}")
                    consumed_cache[exec_rid] = None

            consumed = consumed_cache[exec_rid]
            if not consumed:
                counts["null_config"] += 1
                continue

            version_str = consumed.get(dataset_rid)
            if not version_str:
                counts["null_no_record"] += 1
                continue

            dv_rid = version_index.get((dataset_rid, version_str))
            if not dv_rid:
                counts["null_no_record"] += 1
                continue

            updates.append({"RID": r["RID"], _DV_COLUMN: dv_rid})
            counts["filled"] += 1
        except Exception as exc:  # noqa: BLE001 - defensive: never abort the run
            print(f"    [WARN] Skipping row {r.get('RID')} during backfill: {exc}")
            counts["errors"] += 1

    detail = (
        f"null_config={counts['null_config']}, null_no_record={counts['null_no_record']}, errors={counts['errors']}"
    )
    if dry_run:
        print(f"  [DRY-RUN] Would backfill {counts['filled']} input row(s) ({detail})")
        return counts

    if updates:
        de_path.update(updates)
        print(f"  [OK] Backfilled {counts['filled']} input row(s) ({detail})")
    else:
        print(f"  [SKIP] No input rows resolvable from config ({detail})")
    return counts


# ---------------------------------------------------------------------------
# Verify (folded into the run summary)
# ---------------------------------------------------------------------------
def _verify(catalog: ErmrestCatalog, ml_schema: str) -> dict:
    """Post-run verification.

    Returns:
        dict with keys:
            - output_rows_remaining: int - output edges still in Dataset_Execution
              (should be 0 after a real run)
            - null_version_count: int - surviving rows with NULL Dataset_Version
    """
    pb = catalog.getPathBuilder()
    de_path = pb.schemas[ml_schema].Dataset_Execution
    rows = list(de_path.entities().fetch())

    authored = _authored_pairs(catalog, ml_schema)
    output_remaining = sum(1 for r in rows if (r.get("Dataset"), r.get("Execution")) in authored)
    null_versions = sum(1 for r in rows if not r.get(_DV_COLUMN))
    return {"output_rows_remaining": output_remaining, "null_version_count": null_versions}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_migration(catalog: ErmrestCatalog, ml_schema: str, dry_run: bool) -> dict:
    """Run the full migration against an already-connected catalog.

    Returns:
        ``{"columns_added": n, "output_rows_deleted": n, "backfill": {...}}``.
    """
    hostname = catalog.deriva_server.server

    print("\nStep 1: Add nullable Dataset_Version column + FK")
    columns_added = 1 if step1_add_column(catalog, ml_schema, dry_run) else 0

    print("\nStep 2: Delete redundant output rows")
    output_rows_deleted = step2_delete_output_rows(catalog, ml_schema, dry_run)

    print("\nStep 3: Backfill consumed version on input rows (best-effort)")
    # The column must exist to backfill. In dry-run on a legacy catalog it does
    # not exist yet, so skip backfill and report zero counts.
    if dry_run and columns_added == 1:
        print("  [DRY-RUN] Skipping backfill preview (column does not exist yet)")
        backfill = {"filled": 0, "null_config": 0, "null_no_record": 0, "errors": 0}
    else:
        backfill = step3_backfill_input_versions(catalog, hostname, ml_schema, dry_run)

    result = {
        "columns_added": columns_added,
        "output_rows_deleted": output_rows_deleted,
        "backfill": backfill,
    }

    print("\nSummary:")
    print(f"  Columns added:       {columns_added}")
    print(f"  Output rows deleted: {output_rows_deleted}")
    print(
        f"  Backfilled versions: {backfill['filled']} "
        f"(null_config={backfill['null_config']}, "
        f"null_no_record={backfill['null_no_record']}, "
        f"errors={backfill['errors']})"
    )

    if not dry_run:
        verify = _verify(catalog, ml_schema)
        print("\nVerification:")
        print(f"  Output rows remaining in Dataset_Execution: {verify['output_rows_remaining']}")
        print(f"  Surviving rows with NULL Dataset_Version:   {verify['null_version_count']}")
        if verify["output_rows_remaining"]:
            print("  [WARN] Output rows still present — re-run or investigate.")
        result["verify"] = verify

    return result


def migrate(
    hostname: str,
    catalog_id: str,
    ml_schema: str = "deriva-ml",
    dry_run: bool = False,
    backfill_only: bool = False,
) -> bool:
    """Connect, report preconditions, short-circuit, and run the migration.

    Args:
        backfill_only: When True, skip the structural steps (add column / delete
            output rows) and the "already complete" short-circuit, and run ONLY
            the best-effort version backfill (Step 3) against existing
            NULL-version input rows. Use this to re-attempt backfill on an
            already-structurally-migrated catalog after a change makes more
            ``configuration.json`` files resolvable (e.g. a config-parser fix).
            Requires the ``Dataset_Version`` column to already exist.

    Returns:
        True on success.
    """
    mode = "[DRY-RUN] " if dry_run else ""
    action = "Backfilling input versions for" if backfill_only else "Migrating"
    print(f"\n{mode}{action} Dataset_Execution")
    print(f"  Catalog: {hostname}/{catalog_id}")
    print(f"  Schema:  {ml_schema}")

    catalog = get_catalog(hostname, catalog_id)

    print("\nChecking preconditions...")
    info = check_preconditions(catalog, ml_schema)
    print(f"  Dataset_Version column present: {info['has_column']}")
    print(f"  Dataset_Version FK present:     {info['has_fk']}")
    print(f"  Dataset_Execution row count:    {info['row_count']}")

    if backfill_only:
        # Explicit operator intent to re-run ONLY the backfill. Bypass the
        # short-circuit and the structural steps. The column must already exist.
        if not info["has_column"]:
            print(
                "\n[ERROR] --backfill-only requires the Dataset_Version column to exist; run the full migration first."
            )
            return False
        print("\nStep 3 (only): Backfill consumed version on input rows (best-effort)")
        backfill = step3_backfill_input_versions(catalog, hostname, ml_schema, dry_run)
        print("\nSummary:")
        print(
            f"  Backfilled versions: {backfill['filled']} "
            f"(null_config={backfill['null_config']}, "
            f"null_no_record={backfill['null_no_record']}, "
            f"errors={backfill['errors']})"
        )
        if not dry_run:
            verify = _verify(catalog, ml_schema)
            print("\nVerification:")
            print(f"  Surviving rows with NULL Dataset_Version:   {verify['null_version_count']}")
        print(f"\n{mode}Backfill complete!")
        return True

    # Short-circuit: nothing to do when the column already exists AND no output
    # rows remain to delete. (Backfill is best-effort and never the sole reason
    # to keep re-running; the NULL rows that remain are expected. Use
    # --backfill-only to deliberately re-attempt backfill after a parser fix.)
    if info["has_column"] and info["has_fk"]:
        authored = _authored_pairs(catalog, ml_schema)
        pb = catalog.getPathBuilder()
        rows = list(pb.schemas[ml_schema].Dataset_Execution.entities().fetch())
        output_remaining = any((r.get("Dataset"), r.get("Execution")) in authored for r in rows)
        if not output_remaining:
            print("\nMigration already complete (column present, no output rows). Nothing to do.")
            print("  (To re-attempt version backfill, re-run with --backfill-only.)")
            return True

    run_migration(catalog, ml_schema, dry_run)
    print(f"\n{mode}Migration complete!")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate Dataset_Execution to the input-only provenance shape")
    parser.add_argument("hostname", help="Catalog hostname (e.g., dev.eye-ai.org)")
    parser.add_argument("catalog_id", help="Catalog ID")
    parser.add_argument("--schema", default="deriva-ml", help="ML schema name (default: deriva-ml)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without making changes",
    )
    parser.add_argument(
        "--backfill-only",
        action="store_true",
        help=(
            "Skip the structural steps and the short-circuit; run only the "
            "best-effort version backfill against existing NULL-version input "
            "rows. Use to re-attempt backfill on an already-migrated catalog "
            "(requires the Dataset_Version column to already exist)."
        ),
    )
    args = parser.parse_args()

    success = migrate(
        args.hostname,
        args.catalog_id,
        args.schema,
        args.dry_run,
        backfill_only=args.backfill_only,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
