#!/usr/bin/env python3
"""One-time adoption backfill for the DerivaML provenance contract.

Brings an existing catalog to conformance (the contract is whole-catalog, so
adoption is a migration, not optional cleanup). Two conformance-restoring moves,
neither fabricating provenance:

1. Attribute orphan ``Dataset_Version`` rows (null ``Execution`` producer) to the
   seeded unknown-provenance Execution sentinel — recorded as honest "unknown
   origin", not fabricated provenance.
2. Abort stranded non-terminal executions (``Created`` / ``Running`` /
   ``Pending_Upload``) — their only honest terminal state is ``Aborted``.

The run records its own provenance on the catalog provenance annotation, and a
follow-up audit should report zero null-producer violations.

**DRY-RUN BY DEFAULT.** Without ``--apply`` the script only reports what it would
change. Intended workflow: run against **dev** first (dry-run, then ``--apply``),
verify with the audit it prints, then run against **prod**.

Usage:
    # 1. Dry-run on dev (no changes)
    uv run python scripts/provenance_backfill.py dev.eye-ai.org eye-ai

    # 2. Apply on dev
    uv run python scripts/provenance_backfill.py dev.eye-ai.org eye-ai --apply

    # 3. After verifying dev, apply on prod
    uv run python scripts/provenance_backfill.py www.eye-ai.org eye-ai --apply
"""

from __future__ import annotations

import argparse
import sys

from deriva_ml import DerivaML
from deriva_ml.execution.provenance_backfill import backfill_provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adoption backfill for the DerivaML provenance contract (dry-run by default)."
    )
    parser.add_argument("hostname", help="Catalog hostname (e.g., dev.eye-ai.org, www.eye-ai.org)")
    parser.add_argument("catalog_id", help="Catalog ID or alias (e.g., eye-ai)")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually mutate the catalog. Without this flag the script is a dry-run (reports only).",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Skip the post-run audit summary (audit runs by default to confirm conformance).",
    )
    args = parser.parse_args()

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"\nProvenance backfill — {mode}")
    print(f"  host:    {args.hostname}")
    print(f"  catalog: {args.catalog_id}\n")

    if args.apply:
        # A prod migration is hard to reverse — make the operator confirm.
        confirm = input(f"About to MUTATE {args.hostname}/{args.catalog_id}. Type the hostname to proceed: ")
        if confirm.strip() != args.hostname:
            print("Confirmation did not match hostname. Aborting (no changes made).")
            sys.exit(1)

    ml = DerivaML(hostname=args.hostname, catalog_id=args.catalog_id)

    # ── Pre-run audit (baseline) ──────────────────────────────────────────
    if not args.no_audit:
        before = ml.audit_provenance()
        print(f"BEFORE: {before.summary()}")

    # ── Backfill ──────────────────────────────────────────────────────────
    result = backfill_provenance(ml, apply=args.apply)
    print("\n" + result.summary())
    def _sample(rids: list[str]) -> str:
        head = ", ".join(rids[:10])
        more = "" if len(rids) <= 10 else f" (+{len(rids) - 10} more)"
        return head + more

    if result.orphan_datasets:
        print(f"  orphan dataset sample: {_sample(result.orphan_datasets)}")
    if result.stranded_executions:
        print(f"  stranded execution sample:     {_sample(result.stranded_executions)}")
    for note in result.notes:
        print(f"  NOTE: {note}")

    # ── Post-run audit (only meaningful after --apply) ────────────────────
    if not args.no_audit and args.apply:
        after = ml.audit_provenance()
        print(f"\nAFTER:  {after.summary()}")
        if after.violations:
            print(f"  WARNING: {len(after.violations)} violation(s) remain — review:")
            for v in after.violations[:10]:
                print(f"    {v}")

    if not args.apply:
        print("\n(DRY-RUN — no changes made. Re-run with --apply to mutate.)")


if __name__ == "__main__":
    main()
