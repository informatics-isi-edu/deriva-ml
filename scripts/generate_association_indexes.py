"""Generate SQL for adding composite indexes on association tables.

Reads a deriva catalog via the ermrest HTTP API, finds every pure
binary association table, and writes a single ``.sql`` file with
``CREATE INDEX CONCURRENTLY IF NOT EXISTS`` statements for both
join orderings. Each association gets four indexes:

  * live table — composite (FK1, FK2) and (FK2, FK1)
  * ``_ermrest_history."t<rid>"`` — expression indexes on
    ``rowdata->>'<column_rid>'`` matching the live-side pair

The script does **not** execute any DDL. A DBA reviews the output
and applies it via pgAdmin or ``psql -f`` against the live Postgres
backing the catalog. ``CREATE INDEX CONCURRENTLY`` does not acquire
``ACCESS EXCLUSIVE`` so the apply runs against a live catalog
without blocking; ``IF NOT EXISTS`` makes re-runs safe no-ops.

This is an ad-hoc operator tool — not part of the deriva-ml library,
not registered as a CLI entry-point. It lives in ``scripts/`` next
to ``cutover_smoke_check.py``.

Run:
    uv run python scripts/generate_association_indexes.py \\
        --hostname www.eye-ai.org \\
        --catalog-id eye-ai \\
        --output /tmp/eye-ai-indexes.sql

Without ``--output`` the file lands in cwd with a timestamped name
``association-indexes-<hostname>-<catalog_id>-<UTC-timestamp>.sql``.

Background: ERMrest creates a per-FK index on the live association
table but does not create the composite-pair indexes needed for
fast joins in either direction, and never creates expression
indexes on the JSONB ``rowdata`` of the corresponding history table.
On a real catalog (eye-ai) the composite indexes are noticeably
faster for join queries through pure binary associations.

Origin: design archived at
``docs/superpowers/specs/archive/2026-04-30-association-index-sql-generator-design.md``.
This script is the standalone-operator-tool flavor of that design —
the deriva-ml-library integration proposed there was deferred in
favor of running this by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# System schemas to skip entirely. ``_ermrest`` and ``_ermrest_history``
# are deriva internals; ``pg_catalog`` is Postgres metadata; ``public``
# and ``www`` are conventional public/static buckets that catalogs may
# expose but never contain user association tables.
SYSTEM_SCHEMAS = frozenset({"_ermrest", "_ermrest_history", "pg_catalog", "public", "www"})

# Postgres identifier limit. ``CREATE INDEX`` names exceeding this are
# silently truncated by the server — which can produce collisions when
# two long names share a long common prefix. We truncate explicitly and
# append a hash suffix so distinct intended names always produce
# distinct emitted names.
POSTGRES_IDENT_MAX = 63
_HASH_SUFFIX_LEN = 8


# ---------------------------------------------------------------------------
# SQL string builders — pure functions, no I/O.
# ---------------------------------------------------------------------------


def quote_ident(name: str) -> str:
    """Quote a Postgres identifier (schema, table, column, index name).

    Wraps ``name`` in double quotes and doubles any embedded double
    quotes per SQL identifier-quoting rules.

    Example:
        >>> quote_ident("Subject")
        '"Subject"'
        >>> quote_ident("eye-ai")
        '"eye-ai"'
        >>> quote_ident('weird"name')
        '"weird""name"'
    """
    return '"' + name.replace('"', '""') + '"'


def quote_literal(text: str) -> str:
    """Quote a Postgres string literal.

    Used inside the ``rowdata->>'<value>'`` expressions where
    ``<value>`` is a column RID. RIDs are alphanumeric plus ``-``,
    so the escape is defensive but cheap.

    Example:
        >>> quote_literal("1-ABCD")
        "'1-ABCD'"
        >>> quote_literal("o'reilly")
        "'o''reilly'"
    """
    return "'" + text.replace("'", "''") + "'"


def truncate_index_name(name: str) -> str:
    """Cap an index name at Postgres's 63-byte identifier limit.

    Names that fit are returned unchanged. Longer names are truncated
    and a deterministic 8-character md5 suffix is appended so two
    distinct long names always produce distinct truncated names.

    Example:
        >>> truncate_index_name("short_idx")
        'short_idx'
        >>> len(truncate_index_name("a" * 100))
        63
    """
    if len(name) <= POSTGRES_IDENT_MAX:
        return name
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:_HASH_SUFFIX_LEN]
    head_len = POSTGRES_IDENT_MAX - _HASH_SUFFIX_LEN - 1
    return name[:head_len] + "_" + digest


def live_index_sql(schema: str, table: str, index_name: str, columns: list[str]) -> str:
    """Emit one ``CREATE INDEX CONCURRENTLY IF NOT EXISTS`` for a live table.

    The index covers ``columns`` in the given order; the caller is
    responsible for deciding the order (forward or reverse FK join).

    Example:
        >>> print(live_index_sql("eye-ai", "Subject_Image",
        ...                      "Subject_Image_assoc_fwd_idx",
        ...                      ["Subject_RID", "Image_RID"]))
        CREATE INDEX CONCURRENTLY IF NOT EXISTS "Subject_Image_assoc_fwd_idx"
          ON "eye-ai"."Subject_Image" ("Subject_RID", "Image_RID");
    """
    col_list = ", ".join(quote_ident(c) for c in columns)
    return (
        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {quote_ident(index_name)}\n"
        f"  ON {quote_ident(schema)}.{quote_ident(table)} ({col_list});"
    )


def history_index_sql(table_rid: str, index_name: str, column_rids: list[str]) -> str:
    """Emit one expression index on ``_ermrest_history."t<rid>"`` row-data.

    History rows store column values as JSONB in ``rowdata`` keyed by
    column RID (not name). The expression index uses
    ``(rowdata->>'<col_rid>')`` for each column so joins through history
    can use the same composite-pair lookup the live-side index covers.

    Example:
        >>> print(history_index_sql("1-ABCD",
        ...                         "Subject_Image_hist_assoc_fwd_idx",
        ...                         ["1-EEEE", "1-FFFF"]))
        CREATE INDEX CONCURRENTLY IF NOT EXISTS "Subject_Image_hist_assoc_fwd_idx"
          ON _ermrest_history."t1-ABCD"
             (((rowdata->>'1-EEEE')), ((rowdata->>'1-FFFF')));
    """
    expressions = ", ".join(f"((rowdata->>{quote_literal(rid)}))" for rid in column_rids)
    table_ref = f'_ermrest_history."t{table_rid}"'
    return f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {quote_ident(index_name)}\n  ON {table_ref}\n     ({expressions});"


# ---------------------------------------------------------------------------
# Catalog walk + emit.
# ---------------------------------------------------------------------------


def _build_rid_lookup(catalog: Any) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    """Build ``(schema, table) -> RID`` and ``(table_rid, column) -> RID`` maps.

    ERMrest does not expose ``_ermrest.known_tables`` /
    ``known_columns`` as queryable schemas over the data API, but the
    raw schema document carries the ``"RID"`` for every table and
    column inline. We fetch the doc once and walk it.

    Args:
        catalog: An ``ErmrestCatalog`` (open).

    Returns:
        ``(table_rid_map, column_rid_map)`` where:
        * ``table_rid_map[(schema, table)] = table_rid``
        * ``column_rid_map[(table_rid, column)] = column_rid``
    """
    raw = catalog.get("/schema").json()
    table_rid: dict[tuple[str, str], str] = {}
    column_rid: dict[tuple[str, str], str] = {}
    for schema_name, schema_def in raw.get("schemas", {}).items():
        for table_name, table_def in schema_def.get("tables", {}).items():
            t_rid = table_def.get("RID")
            if not t_rid:
                # Older ermrest servers may not populate this; skip,
                # the caller will warn on the resulting lookup miss.
                continue
            table_rid[(schema_name, table_name)] = t_rid
            for col in table_def.get("column_definitions", []):
                c_name = col.get("name")
                c_rid = col.get("RID")
                if c_name and c_rid:
                    column_rid[(t_rid, c_name)] = c_rid
    return table_rid, column_rid


def _walk_associations(model: Any) -> list[dict[str, Any]]:
    """Yield one record per pure binary association table.

    Uses ``Table.is_association(return_fkeys=True)`` with all defaults
    (pure, unqualified, non-overlapping, binary). Skips system schemas
    entirely.

    Args:
        model: A deriva-py ``Model`` from ``catalog.getCatalogModel()``.

    Returns:
        List of dicts with keys ``schema_name``, ``table_name``,
        ``fk1_columns``, ``fk1_target_schema``, ``fk1_target_table``,
        ``fk2_columns``, ``fk2_target_schema``, ``fk2_target_table``.
    """
    associations: list[dict[str, Any]] = []
    for schema_name, schema in model.schemas.items():
        if schema_name in SYSTEM_SCHEMAS:
            continue
        for table_name, table in schema.tables.items():
            fkeys = table.is_association(return_fkeys=True)
            if not fkeys:
                continue
            # ``return_fkeys=True`` returns a 2-tuple/set of ForeignKey
            # objects. Order matters for deterministic output; sort by
            # the first FK column's name so two runs produce the same
            # fwd/rev assignment regardless of dict iteration order.
            fk1, fk2 = sorted(
                fkeys,
                key=lambda fk: fk.foreign_key_columns[0].name,
            )
            associations.append(
                {
                    "schema_name": schema_name,
                    "table_name": table_name,
                    "fk1_columns": [c.name for c in fk1.foreign_key_columns],
                    "fk1_target_schema": fk1.pk_table.schema.name,
                    "fk1_target_table": fk1.pk_table.name,
                    "fk2_columns": [c.name for c in fk2.foreign_key_columns],
                    "fk2_target_schema": fk2.pk_table.schema.name,
                    "fk2_target_table": fk2.pk_table.name,
                }
            )
    # Sort associations by (schema, table) so output is deterministic
    # across runs.
    associations.sort(key=lambda a: (a["schema_name"], a["table_name"]))
    return associations


def _emit_association_block(
    assoc: dict[str, Any],
    table_rid: dict[tuple[str, str], str],
    column_rid: dict[tuple[str, str], str],
    warnings: list[str],
) -> str:
    """Emit the four CREATE INDEX statements for one association.

    On RID-lookup misses:
      * If the table's RID can't be resolved, only live-side indexes
        are emitted; history-side is skipped and a warning recorded.
      * If a column's RID can't be resolved, the history-side index
        for THAT direction is skipped (live still emitted) and a
        warning recorded.

    Returns:
        The full block as a string (comment header + up to 4 CREATE
        INDEX statements separated by blank lines).
    """
    sname = assoc["schema_name"]
    tname = assoc["table_name"]
    fk1_cols = assoc["fk1_columns"]
    fk2_cols = assoc["fk2_columns"]
    fk1_target = f"{assoc['fk1_target_schema']}.{assoc['fk1_target_table']}"
    fk2_target = f"{assoc['fk2_target_schema']}.{assoc['fk2_target_table']}"

    t_rid = table_rid.get((sname, tname))
    if t_rid is None:
        warnings.append(f"{sname}.{tname}: table RID not found in /schema; history indexes skipped")

    fk1_col_rids: list[str] | None
    fk2_col_rids: list[str] | None
    if t_rid is not None:
        fk1_col_rids = [column_rid.get((t_rid, c)) for c in fk1_cols]
        fk2_col_rids = [column_rid.get((t_rid, c)) for c in fk2_cols]
        if None in fk1_col_rids:
            missing = [c for c, r in zip(fk1_cols, fk1_col_rids, strict=False) if r is None]
            warnings.append(f"{sname}.{tname}: column RID(s) {missing} not found; forward history index skipped")
            fk1_col_rids = None
        if None in fk2_col_rids:
            missing = [c for c, r in zip(fk2_cols, fk2_col_rids, strict=False) if r is None]
            warnings.append(f"{sname}.{tname}: column RID(s) {missing} not found; reverse history index skipped")
            fk2_col_rids = None
    else:
        fk1_col_rids = None
        fk2_col_rids = None

    fk1_str = ", ".join(fk1_cols)
    fk2_str = ", ".join(fk2_cols)
    header = (
        "-- " + "=" * 72 + "\n"
        f"-- {tname}  ({sname}.{tname})\n"
        f"--   Table RID: {t_rid or '<missing>'}\n"
        f"--   FK1: ({fk1_str})  ->  {fk1_target}\n"
        f"--   FK2: ({fk2_str})  ->  {fk2_target}\n"
        "-- " + "=" * 72
    )

    # Index names: <table>_assoc_{fwd,rev}_idx for live, prefixed
    # _hist_ for history. Truncated to 63 bytes with hash suffix.
    fwd_live_name = truncate_index_name(f"{tname}_assoc_fwd_idx")
    rev_live_name = truncate_index_name(f"{tname}_assoc_rev_idx")
    fwd_hist_name = truncate_index_name(f"{tname}_hist_assoc_fwd_idx")
    rev_hist_name = truncate_index_name(f"{tname}_hist_assoc_rev_idx")

    parts = [header, ""]
    # Live forward: (FK1, FK2)
    parts.append(live_index_sql(sname, tname, fwd_live_name, fk1_cols + fk2_cols))
    parts.append("")
    # Live reverse: (FK2, FK1)
    parts.append(live_index_sql(sname, tname, rev_live_name, fk2_cols + fk1_cols))
    parts.append("")
    # History forward (if we have all the RIDs)
    if t_rid is not None and fk1_col_rids is not None and fk2_col_rids is not None:
        parts.append(history_index_sql(t_rid, fwd_hist_name, fk1_col_rids + fk2_col_rids))
        parts.append("")
        parts.append(history_index_sql(t_rid, rev_hist_name, fk2_col_rids + fk1_col_rids))
        parts.append("")
    return "\n".join(parts)


def generate(hostname: str, catalog_id: str, output_path: Path, verbose: bool = False) -> dict:
    """Walk the catalog and write the SQL file.

    Args:
        hostname: Deriva catalog hostname (e.g. ``www.eye-ai.org``).
        catalog_id: Catalog ID.
        output_path: Where to write the .sql output.
        verbose: If True, log per-association progress.

    Returns:
        Summary dict: ``{"associations_found": int, "warnings": list[str],
        "output_path": str}``.
    """
    # Imports inside the function so the module's docstring + helper
    # tests don't require deriva-py at import time (matches the style
    # of scripts/cutover_smoke_check.py).
    from deriva.core import DerivaServer, get_credential

    credential = get_credential(hostname)
    server = DerivaServer("https", hostname, credentials=credential)
    catalog = server.connect_ermrest(catalog_id)

    model = catalog.getCatalogModel()
    table_rid, column_rid = _build_rid_lookup(catalog)

    associations = _walk_associations(model)
    if verbose:
        print(
            f"Found {len(associations)} pure binary association(s) "
            f"across {len(model.schemas) - sum(1 for s in model.schemas if s in SYSTEM_SCHEMAS)} schema(s).",
            file=sys.stderr,
        )

    warnings: list[str] = []
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    header = (
        "-- Generated by scripts/generate_association_indexes.py\n"
        f"-- Catalog:    https://{hostname}/ermrest/catalog/{catalog_id}\n"
        f"-- Generated:  {generated_at}\n"
        f"-- Associations found: {len(associations)}\n"
        "--\n"
        "-- Review before running. Apply via pgAdmin or `psql -f`.\n"
        "-- Each CREATE INDEX uses CONCURRENTLY so it does not block live\n"
        "-- traffic, and IF NOT EXISTS so re-runs are safe no-ops.\n"
        "-- Expression indexes on _ermrest_history.* tables key on column\n"
        "-- RIDs, not column names — see ermrest_schema.sql for details.\n"
        "--\n"
        "-- Recommended apply: psql -f <this-file> -v ON_ERROR_STOP=1\n"
    )

    blocks = []
    for assoc in associations:
        if verbose:
            print(
                f"  {assoc['schema_name']}.{assoc['table_name']}",
                file=sys.stderr,
            )
        blocks.append(_emit_association_block(assoc, table_rid, column_rid, warnings))

    body = "\n".join(blocks) if blocks else "-- No pure binary association tables found.\n"

    output_path.write_text(header + "\n" + body, encoding="utf-8")

    if warnings:
        print("Warnings:", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)

    return {
        "associations_found": len(associations),
        "warnings": warnings,
        "output_path": str(output_path),
    }


def _default_output_path(hostname: str, catalog_id: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path.cwd() / f"association-indexes-{hostname}-{catalog_id}-{ts}.sql"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="generate_association_indexes",
        description=(
            "Generate CREATE INDEX SQL for pure binary association tables "
            "in a deriva catalog. Read-only; output is reviewed and applied "
            "manually by a DBA."
        ),
    )
    parser.add_argument("--hostname", required=True, help="Deriva catalog hostname (e.g. www.eye-ai.org).")
    parser.add_argument("--catalog-id", required=True, help="Catalog ID (e.g. eye-ai or 1).")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path. Defaults to ./association-indexes-<host>-<catalog>-<UTC>.sql.",
    )
    parser.add_argument("--verbose", action="store_true", help="Log per-association progress to stderr.")
    args = parser.parse_args(argv)

    output_path = args.output or _default_output_path(args.hostname, args.catalog_id)

    try:
        summary = generate(args.hostname, args.catalog_id, output_path, verbose=args.verbose)
    except Exception as e:  # noqa: BLE001 — top-level CLI error envelope
        print(f"Error: {e}", file=sys.stderr)
        return 2

    print(
        f"Wrote {summary['associations_found']} association block(s) to {summary['output_path']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
