"""Final-assembly helper for :meth:`Dataset.estimate_bag_size`.

``estimate_bag_size`` runs the client-side FK-reachability engine
(:mod:`deriva_ml.dataset._reachability`) to discover which tables a
dataset reaches and how many rows / asset bytes each contributes, then
hands the engine's result maps to :func:`assemble_estimate` for the final
dict assembly.

This module exposes only :func:`assemble_estimate` — a pure data
transform that turns the reachability engine's ``rids_by_table`` /
``asset_lengths_by_table`` / ``sample_rows_by_table`` maps plus the set of
asset tables into the user-facing estimate dict. Keeping it free of the
:class:`Dataset` import (CSV-byte and human-readable-size formatters are
passed in as callables) makes it unit-testable against synthetic inputs
with no live catalog.
"""

from __future__ import annotations

from typing import Any


def assemble_estimate(
    *,
    asset_tables: set[str],
    rids_by_table: dict[str, set[str]],
    asset_lengths_by_table: dict[str, dict[str, int]],
    sample_rows_by_table: dict[str, list[dict]],
    estimate_csv_bytes: Any,
    human_readable_size: Any,
) -> dict[str, Any]:
    """Assemble the final estimate dict from reachability-engine outputs.

    Pure data transform — same shape as the legacy method's closing block
    but isolated for testability against synthetic inputs.

    Args:
        asset_tables: Set of table names that are asset tables (carry
            downloadable bytes).
        rids_by_table: ``compute_reachability`` row-RID map — table name →
            set of reached RIDs.
        asset_lengths_by_table: ``compute_reachability`` asset-length map —
            table name → ``{RID: Length}`` for reached asset rows.
        sample_rows_by_table: ``sample_rows_from_fetched`` output — table
            name → list of sample rows for CSV-size estimation.
        estimate_csv_bytes: Callable ``(sample_rows, row_count) -> int`` —
            :meth:`Dataset._estimate_csv_bytes`. Passed as a callable so
            this helper stays free of the :class:`Dataset` import. Same
            trick for ``human_readable_size``.
        human_readable_size: Callable ``(bytes) -> str`` —
            :meth:`Dataset._human_readable_size`.

    Returns:
        The full estimate dict — the pre-extraction
        :meth:`Dataset.estimate_bag_size` keys plus, per table, an
        ``incomplete`` bool, and top-level ``incomplete`` /
        ``incomplete_tables`` keys. The client-side reachability
        engine has no per-query failures, so ``incomplete`` is
        always ``False`` and ``incomplete_tables`` always ``[]`` —
        the keys are retained for backward-compatible output shape.

    Example:
        >>> from deriva_ml.dataset._estimate import assemble_estimate
        >>> result = assemble_estimate(
        ...     asset_tables=set(),
        ...     rids_by_table={},
        ...     asset_lengths_by_table={},
        ...     sample_rows_by_table={},
        ...     estimate_csv_bytes=lambda rows, n: 0,
        ...     human_readable_size=lambda n: f"{n}B",
        ... )
        >>> result["total_rows"]
        0
        >>> result["total_estimated_size"]
        '0B'
    """
    # CSV bytes from samples.
    csv_bytes_by_table: dict[str, int] = {}
    for table_name, sample_rows in sample_rows_by_table.items():
        row_count = len(rids_by_table.get(table_name, set()))
        csv_bytes_by_table[table_name] = estimate_csv_bytes(sample_rows, row_count)

    table_estimates: dict[str, dict[str, Any]] = {}
    total_rows = 0
    total_asset_bytes = 0
    total_csv_bytes = 0

    for table_name, rids in rids_by_table.items():
        row_count = len(rids)
        is_asset = table_name in asset_tables
        # ``asset_lengths_by_table`` is typed as a plain dict so
        # the helper accepts both ``dict`` and ``defaultdict``
        # inputs — use ``.get`` to avoid a KeyError when a non-
        # asset table has no entry. Original code used a
        # defaultdict so this branch was implicit; the helper
        # is testable with a plain dict.
        asset_bytes = sum(asset_lengths_by_table.get(table_name, {}).values())
        csv_bytes = csv_bytes_by_table.get(table_name, 0)
        table_estimates[table_name] = {
            "row_count": row_count,
            "is_asset": is_asset,
            "asset_bytes": asset_bytes,
            "csv_bytes": csv_bytes,
            "incomplete": False,
        }
        total_rows += row_count
        total_asset_bytes += asset_bytes
        total_csv_bytes += csv_bytes

    # Tables that only appear in ``fetch`` results (unlikely
    # but defensive — an asset path with no ``csv`` query would
    # land here).
    for table_name, lengths in asset_lengths_by_table.items():
        if table_name not in table_estimates:
            csv_bytes = csv_bytes_by_table.get(table_name, 0)
            table_estimates[table_name] = {
                "row_count": len(lengths),
                "is_asset": True,
                "asset_bytes": sum(lengths.values()),
                "csv_bytes": csv_bytes,
                "incomplete": False,
            }
            total_rows += len(lengths)
            total_asset_bytes += sum(lengths.values())
            total_csv_bytes += csv_bytes

    # The client-side reachability engine has no per-query failures, so no
    # table is ever incomplete. The comprehension still yields [] and the
    # output keys are retained for backward-compatible shape.
    incomplete_tables = sorted(t for t, d in table_estimates.items() if d["incomplete"])

    total_size = total_asset_bytes + total_csv_bytes
    return {
        "tables": table_estimates,
        "total_rows": total_rows,
        "total_asset_bytes": total_asset_bytes,
        "total_asset_size": human_readable_size(total_asset_bytes),
        "total_csv_bytes": total_csv_bytes,
        "total_csv_size": human_readable_size(total_csv_bytes),
        "total_estimated_bytes": total_size,
        "total_estimated_size": human_readable_size(total_size),
        # Failure visibility: when True, counts/sizes are lower
        # bounds — at least one estimate query failed.
        "incomplete": bool(incomplete_tables),
        "incomplete_tables": incomplete_tables,
    }


__all__ = ["assemble_estimate"]
