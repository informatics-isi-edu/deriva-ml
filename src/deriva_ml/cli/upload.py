"""Command-line interface for deriva-ml upload.

Wraps DerivaML.upload_pending so operator-driven uploads can be
scheduled, backgrounded, or run on a different host from the
compute. Typical invocations:

    deriva-ml-upload --host example.org --catalog 42

    deriva-ml-upload --host example.org --catalog 42 \\
        --execution EXE-A --execution EXE-B

    nohup deriva-ml-upload --host example.org --catalog 42 &

Per spec §2.11.4. Drives the same engine as ml.upload_pending.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deriva_ml import DerivaML

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser. Factored so tests can import."""
    p = argparse.ArgumentParser(
        prog="deriva-ml-upload",
        description=(
            "Upload pending execution outputs to the catalog. Drains "
            "workspace SQLite-staged rows and asset files via deriva-py's "
            "resumable uploader. Safe to re-run after crashes — idempotent."
        ),
    )
    p.add_argument(
        "--host", required=True,
        help="Deriva catalog hostname (e.g., example.org).",
    )
    p.add_argument(
        "--catalog", required=True,
        help="Catalog ID (e.g., 42).",
    )
    p.add_argument(
        "--execution", action="append", dest="execution_rids",
        help=(
            "Execution RID to upload. Can be given multiple times. "
            "If omitted, drains every execution that has pending work."
        ),
    )
    p.add_argument(
        "--retry-failed", action="store_true",
        help="Include rows currently in status='failed'.",
    )
    p.add_argument(
        "--working-dir", default=".",
        help="Workspace root (default: current directory).",
    )
    p.add_argument(
        "--mode", choices=["online", "offline"], default="online",
        help="Connection mode (default online — upload requires online).",
    )
    return p


def _construct_ml(host: str, catalog: str, mode: str) -> "DerivaML":
    """Construct a DerivaML instance from CLI args.

    Factored so tests can monkeypatch a prebuilt test_ml instead of
    connecting to a real catalog.
    """
    from deriva_ml import ConnectionMode, DerivaML
    return DerivaML(hostname=host, catalog_id=catalog, mode=ConnectionMode(mode))


def main(argv: "list[str] | None" = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments (without program name). If None,
            use sys.argv[1:]. Argparse prints help to stderr and
            exits with code 2 on parse error.

    Returns:
        Exit code: 0 success, 1 if any failures reported, 2 on fatal error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ml = _construct_ml(args.host, args.catalog, args.mode)
    try:
        report = ml.upload_pending(
            execution_rids=args.execution_rids,
            retry_failed=args.retry_failed,
        )
    except Exception as exc:
        logger.error("upload_pending failed: %s", exc)
        return 2

    print(
        f"upload complete: {report.total_uploaded} items uploaded, "
        f"{report.total_failed} failed.",
        file=sys.stderr,
    )
    for fqn, counts in report.per_table.items():
        print(f"  {fqn}: +{counts['uploaded']} / -{counts['failed']}",
              file=sys.stderr)
    if report.errors:
        print("failures:", file=sys.stderr)
        for err in report.errors[:10]:
            print(f"  - {err}", file=sys.stderr)
        if len(report.errors) > 10:
            print(f"  ... and {len(report.errors) - 10} more", file=sys.stderr)

    return 0 if report.total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
