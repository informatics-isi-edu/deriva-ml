"""Paged fetching primitive for the local_db layer.

Exposes three public methods:

- ``fetch_predicate``: keyset-paged scan of rows matching an ERMrest
  predicate into a SQLAlchemy ``Table`` in local SQLite.
- ``fetch_by_rids``: RID-set batched fetch with URL byte-length guard, GET
  shrink-then-POST fallback, and per-operation dedup against a
  ``target_table``.
- ``fetch_by_rids_or_predicate``: dispatches between the two based on a
  cardinality threshold (``|rid_set| / table_row_count``).

The class is parameterized on a ``client`` object providing four methods:
``count``, ``fetch_page``, ``fetch_rid_batch``, and — optionally — a `POST`
mode flag to ``fetch_rid_batch`` for the oversized-URL fallback. See the
fake client used in tests for the exact interface contract.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Protocol

from sqlalchemy import Table, insert, select
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = 1000
DEFAULT_BATCH_SIZE = 500
DEFAULT_MAX_URL_BYTES = 6144
DEFAULT_CARDINALITY_THRESHOLD = 0.5


class PagedClient(Protocol):
    """Narrow protocol the :class:`PagedFetcher` depends on.

    Any object implementing these three methods can be used as a client.
    The production implementation is :class:`~deriva_ml.local_db.paged_fetcher_ermrest.ErmrestPagedClient`;
    the test suite uses a simple in-memory fake.

    All table names are passed in ERMrest *qualified* form (``"schema:table"``).
    """

    def count(self, table: str) -> int:
        """Return the total row count for *table* (used for predicate vs. RID routing)."""
        ...

    def fetch_page(
        self,
        table: str,
        sort: tuple[str, ...],
        after: tuple | None,
        predicate: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch one page of rows from *table*.

        Args:
            table: Qualified table name (``"schema:table"``).
            sort: Column names to sort by (used for keyset pagination).
            after: Keyset cursor values for ``@after(...)`` — ``None`` for first page.
            predicate: Optional ERMrest filter predicate (e.g., ``"Status=active"``).
            limit: Maximum rows to return.

        Returns:
            List of row dicts.  An empty list signals end-of-data.
        """
        ...

    def fetch_rid_batch(
        self,
        table: str,
        column: str,
        rids: list[str],
        method: str = "GET",
    ) -> list[dict[str, Any]]:
        """Fetch rows whose *column* matches any value in *rids*.

        Args:
            table: Qualified table name.
            column: Column to filter on (typically ``"RID"``).
            rids: List of values to match.
            method: Transport method — ``"GET"`` or ``"POST"``.  Not all clients
                support ``"POST"``; see :class:`ErmrestPagedClient` for details.

        Returns:
            List of matching row dicts.
        """
        ...


class PagedFetcher:
    """Stream rows from an ERMrest-like client into a local SQLAlchemy Table.

    Provides three fetch strategies:

    - :meth:`fetch_predicate`: keyset-paginated full-table (or filtered) scan.
      Efficient for high-cardinality fetches where most rows are wanted.
    - :meth:`fetch_by_rids`: batched RID-set fetch with URL length guard and
      automatic GET→POST fallback.  Efficient for sparse/small RID sets.
    - :meth:`fetch_by_rids_or_predicate`: automatically routes between the two
      strategies based on ``|rids| / table_row_count`` cardinality ratio.

    Deduplication is performed per ``(table, column)`` pair: rows already
    inserted in a previous fetch call are skipped.

    Args:
        client: A :class:`PagedClient` implementation (e.g., :class:`ErmrestPagedClient`).
        engine: SQLAlchemy engine where fetched rows will be inserted.

    Example::

        fetcher = PagedFetcher(client=ErmrestPagedClient(catalog=cat), engine=engine)
        n = fetcher.fetch_by_rids("isa:Image", rids=["IMG-1", "IMG-2"], target_table=image_t)
        print(f"Fetched {n} rows")
    """

    def __init__(self, *, client: PagedClient, engine: Engine) -> None:
        self._client = client
        self._engine = engine
        # Tracks already-inserted (table, column) → set of seen values, preventing
        # double-inserts when the same fetcher is reused across multiple calls.
        self._seen: dict[tuple[str, str], set[str]] = {}
        # Cached row counts per table, populated lazily by fetch_by_rids_or_predicate.
        self._counts: dict[str, int] = {}

    def fetch_predicate(
        self,
        table: str,
        predicate: str | None,
        target_table: Table,
        sort: tuple[str, ...] = ("RID",),
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> int:
        """Fetch all rows matching *predicate* via keyset pagination.

        Iterates ``@after`` pages until the client returns fewer rows than
        *page_size*, which signals end-of-data.  Rows are inserted into
        *target_table* and the fetched RIDs are recorded for deduplication.

        Args:
            table: Qualified table name (``"schema:table"``).
            predicate: ERMrest filter predicate, or ``None`` for all rows.
            target_table: SQLAlchemy :class:`Table` to insert into.
            sort: Column(s) to sort by for keyset pagination.
            page_size: Rows per page (also the stop condition threshold).

        Returns:
            Total number of rows inserted.
        """
        n = 0
        after: tuple | None = None
        while True:
            page = self._client.fetch_page(
                table=table,
                sort=sort,
                after=after,
                predicate=predicate,
                limit=page_size,
            )
            if not page:
                break
            self._insert_rows(target_table, page)
            key = (table, "RID")
            self._seen.setdefault(key, set()).update(str(r["RID"]) for r in page if "RID" in r)
            n += len(page)
            if len(page) < page_size:
                break
            last = page[-1]
            after = tuple(last[c] for c in sort)
        return n

    def fetch_by_rids(
        self,
        table: str,
        rids: Iterable[str],
        target_table: Table,
        rid_column: str = "RID",
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_url_bytes: int = DEFAULT_MAX_URL_BYTES,
    ) -> int:
        """Fetch rows by explicit RID list, batching and deduplicating automatically.

        Already-seen RIDs (from previous calls on this fetcher) are skipped.
        Each batch is fetched via :meth:`_fetch_rid_batch_with_fallback`, which
        tries GET first and falls back to POST if the URL would be too long.

        Args:
            table: Qualified table name (``"schema:table"``).
            rids: Iterable of RID values to fetch.
            target_table: SQLAlchemy :class:`Table` to insert into.
            rid_column: Column name that holds the RID values (default ``"RID"``).
            batch_size: Number of RIDs per HTTP request.
            max_url_bytes: Byte threshold above which GET is split into POST.

        Returns:
            Number of rows actually inserted (may be less than ``len(rids)`` if
            some RIDs were already in the seen-set or don't exist on the server).
        """
        key = (table, rid_column)
        seen = self._seen.setdefault(key, set())
        to_fetch = [r for r in dict.fromkeys(str(x) for x in rids) if r not in seen]
        if not to_fetch:
            return 0

        n = 0
        i = 0
        while i < len(to_fetch):
            batch = to_fetch[i : i + batch_size]
            rows = self._fetch_rid_batch_with_fallback(
                table=table,
                column=rid_column,
                rids=batch,
                max_url_bytes=max_url_bytes,
            )
            self._insert_rows(target_table, rows)
            seen.update(batch)
            n += len(rows)
            i += batch_size
        return n

    def _fetch_rid_batch_with_fallback(
        self,
        *,
        table: str,
        column: str,
        rids: list[str],
        max_url_bytes: int,
    ) -> list[dict[str, Any]]:
        attempt = list(rids)
        while attempt:
            estimated = 128 + 13 * len(attempt)
            if estimated <= max_url_bytes:
                try:
                    return self._client.fetch_rid_batch(table=table, column=column, rids=attempt, method="GET")
                except RuntimeError as exc:
                    if "too long" not in str(exc).lower():
                        raise
            half = len(attempt) // 2
            if half == 0:
                logger.debug(
                    "POST fallback for table=%s column=%s rids=%d",
                    table,
                    column,
                    len(rids),
                )
                return self._client.fetch_rid_batch(table=table, column=column, rids=rids, method="POST")
            attempt = attempt[:half]
        return []

    def fetch_by_rids_or_predicate(
        self,
        table: str,
        rids: list[str],
        target_table: Table,
        rid_column: str = "RID",
        sort: tuple[str, ...] = ("RID",),
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_url_bytes: int = DEFAULT_MAX_URL_BYTES,
        cardinality_threshold: float = DEFAULT_CARDINALITY_THRESHOLD,
    ) -> int:
        """Route between predicate scan and RID-batch fetch based on cardinality.

        If ``|rids| / table_row_count > cardinality_threshold``, it's cheaper to
        fetch the whole table and discard unwanted rows locally.  Otherwise,
        batched RID fetches are used.

        After a predicate scan, rows NOT in *rids* are deleted from
        *target_table* so the table contains only the requested rows.

        Args:
            table: Qualified table name.
            rids: RID values to fetch.
            target_table: SQLAlchemy :class:`Table` to insert into.
            rid_column: RID column name.
            sort: Sort columns for keyset pagination (predicate path only).
            batch_size: Batch size for RID fetches.
            max_url_bytes: URL length guard for GET requests.
            cardinality_threshold: Ratio threshold that triggers a predicate scan.

        Returns:
            Approximate number of rows inserted.
        """
        total = self._counts.get(table)
        if total is None:
            total = self._client.count(table)
            self._counts[table] = total

        if total > 0 and len(rids) / total > cardinality_threshold:
            logger.debug(
                "Using predicate fetch for %s (|rids|=%d, total=%d)",
                table,
                len(rids),
                total,
            )
            n_fetched = self.fetch_predicate(
                table=table,
                predicate=None,
                target_table=target_table,
                sort=sort,
                page_size=max(batch_size, DEFAULT_PAGE_SIZE),
            )
            wanted = set(rids)
            with self._engine.begin() as conn:
                all_rids = [row[0] for row in conn.execute(select(target_table.c[rid_column]))]
                to_delete = [r for r in all_rids if r not in wanted]
                if to_delete:
                    conn.execute(target_table.delete().where(target_table.c[rid_column].in_(to_delete)))
            return min(n_fetched, len(wanted))

        return self.fetch_by_rids(
            table=table,
            rids=rids,
            target_table=target_table,
            rid_column=rid_column,
            batch_size=batch_size,
            max_url_bytes=max_url_bytes,
        )

    def fetched_rids(self, table: str, target_table: Table | None = None) -> set[str]:
        """Return the set of RIDs that have been fetched for *table*.

        If no RIDs have been tracked yet but *target_table* is provided, queries
        the local table directly to populate the seen-set.

        Args:
            table: Qualified table name.
            target_table: Optional SQLAlchemy Table to fall back to for lookup.

        Returns:
            Set of RID strings (may be empty if nothing has been fetched yet).
        """
        for (t, _col), s in self._seen.items():
            if t == table:
                return set(s)
        if target_table is not None:
            with self._engine.connect() as conn:
                rids = {str(row[0]) for row in conn.execute(select(target_table.c.RID))}
            self._seen[(table, "RID")] = set(rids)
            return set(rids)
        return set()

    def _insert_rows(self, target_table: Table, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        cols = {c.name for c in target_table.columns}
        projected = [{k: v for k, v in r.items() if k in cols} for r in rows]
        with self._engine.begin() as conn:
            conn.execute(insert(target_table), projected)
