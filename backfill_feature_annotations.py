"""Backfill the standard feature-table annotation across an eye-ai catalog.

``feature_annotation()`` only runs at ``create_feature()`` time, so feature
tables that predate it (8 of 10 on eye-ai) carry no visible-columns annotation
at all. This re-applies the current standard annotation — including the new
Workflow / Workflow Name / Workflow Type provenance facets — to every feature
table.

Target resolution: the ``Execution_<Target>_<Feature>`` naming convention picks
the target FK. ``find_features()`` is NOT used for this because it mis-reports
the target on multi-FK tables (it returns ``Image_Side`` for
``Execution_Subject_Chart_Label``, whose real target is ``Subject``).

Tables that don't follow the convention are skipped with a warning rather than
guessed at. On eye-ai the only two non-conforming tables (``Annotation``,
``Image_Diagnosis``) are slated for deletion, so skipping them is correct.

Usage:
    # Show the diff without writing anything (default):
    uv run python backfill_feature_annotations.py --host dev.eye-ai.org --catalog 2073

    # Actually write:
    uv run python backfill_feature_annotations.py --host dev.eye-ai.org --catalog 2073 --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from unittest.mock import patch

from deriva.core import tag as deriva_tags

from deriva_ml import DerivaML
from deriva_ml.schema.annotations import feature_annotation

SCHEMA = "eye-ai"


def resolve_target(table) -> str | None:
    """Return the feature table's target-table FK column name, or None.

    Only the ``Execution_<Target>_<Feature>`` convention is honored. A table
    that doesn't conform returns None and is skipped — guessing the target
    would annotate the row-name and target facet against the wrong column.
    """
    if not table.name.startswith("Execution_"):
        return None
    rest = table.name[len("Execution_") :]
    for fk in table.foreign_keys:
        col = fk.columns[0].name
        if col in ("RCB", "RMB", "Execution", "Feature_Name"):
            continue
        # The target is the FK whose column name prefixes the <Feature> segment.
        if rest.startswith(col + "_"):
            return col
    return None


def is_feature_table(table) -> bool:
    cols = {c.name for c in table.columns}
    return {"Execution", "Feature_Name"} <= cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--apply", action="store_true", help="write to the catalog (default: dry run)")
    args = ap.parse_args()

    ml = DerivaML(args.host, catalog_id=args.catalog, default_schema=SCHEMA)
    model = ml.model

    targets = []
    for table in model.schemas[SCHEMA].tables.values():
        if not is_feature_table(table):
            continue
        target = resolve_target(table)
        if target is None:
            print(f"  SKIP {table.name}: cannot resolve target table", file=sys.stderr)
            continue
        targets.append((table, target))

    print(f"{'=' * 78}\n{args.host}/{args.catalog} — {len(targets)} feature tables\n{'=' * 78}")

    changed = 0
    for table, target in sorted(targets, key=lambda t: t[0].name):
        # Deep-copy the pre-state: feature_annotation mutates table.annotations
        # in place, so a shallow reference would compare equal to itself and
        # report every table as "unchanged".
        before = json.loads(json.dumps(table.annotations.get(deriva_tags.visible_columns, {})))
        had_annotation = bool(before)

        # Build the annotation in memory. model.apply() is blocked unless --apply.
        if args.apply:
            feature_annotation(table, target)
        else:
            with patch.object(type(model), "apply", lambda self, *a, **k: None):
                feature_annotation(table, target)

        after = table.annotations.get(deriva_tags.visible_columns, {})
        facets = [f.get("markdown_name", f.get("source")) for f in after["filter"]["and"]]
        had = "yes" if had_annotation else "NO (unannotated)"
        status = "unchanged" if before == after else "UPDATED"
        if before != after:
            changed += 1
        print(f"\n{table.name}  (target={target})")
        print(f"  had annotation: {had}   -> {status}")
        print(f"  facets: {' | '.join(str(f) for f in facets)}")

    verb = "APPLIED to" if args.apply else "would change"
    print(f"\n{'=' * 78}\n{verb} {changed}/{len(targets)} tables")
    if not args.apply:
        print("Dry run — nothing written. Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
