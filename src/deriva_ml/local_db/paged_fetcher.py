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
    """Narrow surface the PagedFetcher depends on."""

    def count(self, table: str) -> int: ...

    def fetch_page(
        self,
        table: str,
        sort: tuple[str, ...],
        after: tuple | None,
        predicate: str | None,
        limit: int,
    ) -> list[dict[str, Any]]: ...

    def fetch_rid_batch(
        self,
        table: str,
        column: str,
        rids: list[str],
        method: str = "GET",
    ) -> list[dict[str, Any]]: ...


class PagedFetcher:
    """Stream rows from an ERMrest-like client into a local SQLAlchemy Table."""

    def __init__(self, *, client: PagedClient, engine: Engine) -> None:
        self._client = client
        self._engine = engine
        self._seen: dict[tuple[str, str], set[str]] = {}
        self._counts: dict[str, int] = {}

    def fetch_predicate(
        self,
        table: str,
        predicate: str | None,
        target_table: Table,
        sort: tuple[str, ...] = ("RID",),
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> int:
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
