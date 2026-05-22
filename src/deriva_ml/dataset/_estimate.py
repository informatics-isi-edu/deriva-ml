"""Extracted helpers for :meth:`Dataset.estimate_bag_size` (audit P1 Ds-est).

Pre-extraction, ``estimate_bag_size`` was a 220-line method that
did everything from snapshot resolution through async query
orchestration through final dict assembly. The audit flagged it
as a god-function because the fine-grained steps — URI
extraction, query-item building, result aggregation, CSV
estimation, asset detection, final assembly — could not be
unit-tested without standing up a live catalog.

This module breaks the work into three free functions:

- :func:`build_estimate_queries` — pure: walks the
  ``aggregate_queries`` output and constructs the list of
  ``QueryItem`` records that the async layer will execute. No
  catalog calls; testable with synthetic ``table_queries`` input.
- :func:`run_estimate_queries` — orchestration: fires every
  query against the async catalog and groups results by table +
  query type. Testable with a mock catalog that returns canned
  JSON.
- :func:`assemble_estimate` — pure: takes the orchestrator's
  three result maps and produces the final estimate dict.
  Testable with synthetic dicts.

The ``Dataset.estimate_bag_size`` method is now a thin shim
that composes these three steps. The async-from-sync bridge
still lives in the calling method (``run_async``) because it
imports a deriva-ml internal helper that's awkward to reach
from a leaf module.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from deriva.core import AsyncErmrestCatalog, AsyncErmrestSnapshot


QueryType = Literal["csv", "fetch", "sample"]
"""Three classes of query needed for an estimate.

- ``csv``: fetch the full RID list for a table along an FK path
  (union of paths gives the exact row count).
- ``fetch``: fetch ``(RID, Length)`` pairs for an asset table
  (sum of de-duplicated lengths gives the exact asset bytes).
- ``sample``: fetch up to 100 rows for CSV serialisation size
  estimation (one sample per table; first path wins).
"""


@dataclass(frozen=True)
class QueryItem:
    """One unit of work for the async estimate orchestrator.

    Attributes:
        table_name: Target table the query is about.
        path: Catalog-relative URI, starting at one of
            ``/aggregate/``, ``/entity/``, or ``/attribute/``.
            Already stripped of the
            ``https://host/ermrest/catalog/N`` prefix.
        query_type: Which post-processing branch applies to the
            response — see :data:`QueryType`.
    """

    table_name: str
    path: str
    query_type: QueryType


def _extract_path(uri: str) -> str:
    """Strip the catalog-server prefix from a full datapath URI.

    Datapath builders return absolute URIs like
    ``https://host/ermrest/catalog/N/entity/Schema:Table``;
    the async catalog wrapper wants the path starting from
    the operation marker (``/entity/``, ``/aggregate/``,
    ``/attribute/``).

    Args:
        uri: A full datapath URI.

    Returns:
        Catalog-relative path, e.g. ``/entity/Schema:Table``.

    Raises:
        ValueError: If the URI doesn't contain a recognised
            operation marker.

    Example:
        >>> _extract_path("https://srv/ermrest/catalog/3/entity/X:Y")
        '/entity/X:Y'
        >>> _extract_path("https://srv/ermrest/catalog/3/aggregate/X:Y/cnt(RID)")
        '/aggregate/X:Y/cnt(RID)'
    """
    for marker in ("/aggregate/", "/entity/", "/attribute/"):
        idx = uri.find(marker)
        if idx >= 0:
            return uri[idx:]
    raise ValueError(f"Cannot extract catalog path from URI: {uri}")


def build_estimate_queries(
    table_queries: dict[str, list[tuple[Any, Any, bool]]],
) -> list[QueryItem]:
    """Build the full list of queries for an estimate run.

    For each ``(table_name, [(datapath, target_table, is_asset), ...])``
    entry in ``table_queries``, emits:

    - One ``csv`` query per FK path (RID list for union counting).
    - One ``fetch`` query per FK path **when ``is_asset`` is True**
      (``(RID, Length)`` pairs for asset size summation).
    - One ``sample`` query **per table** (only the first path's
      sample is taken; subsequent paths share the same row shape
      so re-sampling is wasted work).

    Args:
        table_queries: ``DatasetBagBuilder.aggregate_queries(dataset)``
            output — a mapping from table name to a list of
            ``(datapath, target_table, is_asset)`` triples, one
            per FK path that reaches the table.

    Returns:
        Ordered list of :class:`QueryItem` ready to feed to
        :func:`run_estimate_queries`. Order is
        ``(per table, per FK path, csv → fetch → first-time sample)``.

    Example:
        >>> # Pure-Python contract — see test_estimate_helpers.py for
        >>> # a fixture-driven example. Helpers don't fabricate datapaths
        >>> # in a doctest.
        >>> from deriva_ml.dataset._estimate import build_estimate_queries
        >>> build_estimate_queries({})
        []
    """
    items: list[QueryItem] = []
    sampled_tables: set[str] = set()

    for table_name, path_entries in table_queries.items():
        for dp, target_table, is_asset in path_entries:
            # Fetch the RID list for row-count union.
            rid_rs = dp.attributes(target_table.RID)
            items.append(
                QueryItem(
                    table_name=table_name,
                    path=_extract_path(rid_rs.uri),
                    query_type="csv",
                )
            )

            if is_asset:
                entity_path = _extract_path(dp.uri).removeprefix("/entity/")
                items.append(
                    QueryItem(
                        table_name=table_name,
                        path=f"/attribute/{entity_path}/RID,Length",
                        query_type="fetch",
                    )
                )

            # Sample a few rows to estimate CSV serialisation size.
            # Only one sample per table — first path wins. Subsequent
            # paths through the same table share the same row shape,
            # so a second sample would be wasted work.
            if table_name not in sampled_tables:
                sampled_tables.add(table_name)
                items.append(
                    QueryItem(
                        table_name=table_name,
                        path=f"{_extract_path(dp.uri)}?limit=100",
                        query_type="sample",
                    )
                )

    return items


async def run_estimate_queries(
    catalog: "AsyncErmrestCatalog | AsyncErmrestSnapshot",
    items: list[QueryItem],
    *,
    logger: Any = None,
) -> tuple[
    dict[str, set[str]],
    dict[str, dict[str, int]],
    dict[str, list[dict]],
]:
    """Fire every query concurrently; group results by table + type.

    Single failure per query is **swallowed with a debug log**
    rather than aborting the whole estimate — partial estimates
    are still useful and matches the pre-extraction contract.
    Caller-side failures (e.g. credential resolution, snapshot
    resolution) still propagate.

    Args:
        catalog: A snapshot-scoped async catalog connection. The
            function calls ``catalog.get_async`` for each item
            and then closes the connection on its way out.
        items: Output of :func:`build_estimate_queries`.
        logger: Optional logger for per-query failure debug
            lines. Defaults to the dataset module logger.

    Returns:
        Three dicts in a tuple:

        - ``rids_by_table`` — table name → set of RIDs from
          ``csv`` queries (union across FK paths).
        - ``asset_lengths_by_table`` — table name → ``{RID:
          Length}`` from ``fetch`` queries (first occurrence
          wins; the same asset across paths must have the same
          length so collisions are benign).
        - ``sample_rows_by_table`` — table name → list of sample
          rows from the ``sample`` query (only one sample per
          table; subsequent ``sample`` results for the same
          table are ignored).

    Example:
        >>> # See test_estimate_helpers.py for a mock-catalog
        >>> # exercise that doesn't need a live server.
        >>> ...  # doctest: +SKIP
    """
    import asyncio

    if logger is None:
        from deriva_ml.core.logging_config import get_logger

        logger = get_logger(__name__)

    async def _run_one(item: QueryItem) -> tuple[str, QueryType, Any]:
        try:
            response = await catalog.get_async(item.path)
            return item.table_name, item.query_type, response.json()
        except Exception as exc:
            logger.debug(
                "estimate_bag_size query failed for %s (%s): %s",
                item.table_name,
                item.path,
                exc,
            )
            return item.table_name, item.query_type, []

    try:
        results = await asyncio.gather(*[_run_one(it) for it in items])
    finally:
        await catalog.close()

    rids_by_table: dict[str, set[str]] = defaultdict(set)
    asset_lengths_by_table: dict[str, dict[str, int]] = defaultdict(dict)
    sample_rows_by_table: dict[str, list[dict]] = {}

    for table_name, query_type, rows in results:
        if query_type == "csv":
            rids_by_table[table_name].update(r["RID"] for r in rows if "RID" in r)
        elif query_type == "fetch":
            for r in rows:
                rid = r.get("RID")
                if rid and rid not in asset_lengths_by_table[table_name]:
                    asset_lengths_by_table[table_name][rid] = r.get("Length") or 0
        elif query_type == "sample":
            # Only the first sample per table — set during query
            # building. Subsequent ``sample`` responses for the
            # same table are dropped.
            if table_name not in sample_rows_by_table and rows:
                sample_rows_by_table[table_name] = rows

    return rids_by_table, asset_lengths_by_table, sample_rows_by_table


def assemble_estimate(
    *,
    table_queries: dict[str, list[tuple[Any, Any, bool]]],
    rids_by_table: dict[str, set[str]],
    asset_lengths_by_table: dict[str, dict[str, int]],
    sample_rows_by_table: dict[str, list[dict]],
    estimate_csv_bytes: Any,
    human_readable_size: Any,
) -> dict[str, Any]:
    """Assemble the final estimate dict from orchestrator outputs.

    Pure data transform — same shape as the legacy method's
    closing block but isolated for testability against synthetic
    inputs.

    Args:
        table_queries: Same input that
            :func:`build_estimate_queries` received; needed here
            to determine which tables are assets (the
            ``is_asset`` flag on any FK path).
        rids_by_table: First output of
            :func:`run_estimate_queries`.
        asset_lengths_by_table: Second output.
        sample_rows_by_table: Third output.
        estimate_csv_bytes: Callable
            ``(sample_rows, row_count) -> int`` —
            :meth:`Dataset._estimate_csv_bytes`. Passed as a
            callable so this helper stays free of the
            :class:`Dataset` import. Same trick for
            ``human_readable_size``.
        human_readable_size: Callable ``(bytes) -> str`` —
            :meth:`Dataset._human_readable_size`.

    Returns:
        The full estimate dict — same shape and keys as the
        pre-extraction :meth:`Dataset.estimate_bag_size` return
        value.

    Example:
        >>> from deriva_ml.dataset._estimate import assemble_estimate
        >>> result = assemble_estimate(
        ...     table_queries={},
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

    # Which tables are assets (any FK path declared it so).
    asset_tables = {
        table_name
        for table_name, entries in table_queries.items()
        if any(is_asset for _, _, is_asset in entries)
    }

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
            }
            total_rows += len(lengths)
            total_asset_bytes += sum(lengths.values())
            total_csv_bytes += csv_bytes

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
    }


__all__ = [
    "QueryItem",
    "QueryType",
    "assemble_estimate",
    "build_estimate_queries",
    "run_estimate_queries",
]
