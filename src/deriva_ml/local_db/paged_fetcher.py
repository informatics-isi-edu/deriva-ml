"""Paged fetching primitive for the local_db layer.

Stateless transport adapter between an ERMrest-like ``PagedClient`` and a
SQLAlchemy engine's local SQLite. Three public methods:

- ``fetch_predicate``: keyset-paged scan of rows matching an ERMrest
  predicate into a SQLAlchemy ``Table`` in local SQLite.
- ``fetch_by_rids``: RID-set batched fetch with URL byte-length guard and
  GET-shrink-then-POST fallback.
- ``fetch_by_rids_or_predicate``: dispatches between the two based on a
  cardinality threshold (``|rid_set| / table_row_count``).

Contract: the only mutation point against the engine is ``_insert_rows``,
which uses ``INSERT OR IGNORE`` semantics (rows whose ``RID`` already
exists in the target table are skipped, not overwritten, not raised on).
``fetch_*`` methods do NOT consult engine state to decide what to
request — the server is the authority for what rows match a query, and
the database's UNIQUE constraint is the authority for insert collisions.

See ``docs/user-guide/denormalization.md`` for the full pipeline architecture,
state model, and contract. The 2026-05-21 model-template e2e run
documented two failure modes this design closes (finding 05: re-INSERT
crash; finding A01: silent row-drop on FK-column fetch).

The class is parameterized on a ``client`` object providing three methods:
``count``, ``fetch_page``, ``fetch_rid_batch``. See the fake client used
in tests for the exact interface contract.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Protocol

from sqlalchemy import Table, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from deriva_ml.core.logging_config import get_logger

logger = get_logger(__name__)
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

    Statelessness contract (``docs/user-guide/denormalization.md`` §4):

    - **No engine-derived dedup.** ``fetch_by_rids``/``fetch_predicate`` do
      not consult the engine to decide what to request. The server is the
      authority for "what rows match this query right now."
    - **INSERT-OR-IGNORE at the only mutation point.** Insert collisions
      are resolved by the database's UNIQUE constraint via
      ``INSERT OR IGNORE``, so re-entering the populate path against an
      engine that already holds some rows (within a session OR across
      sessions, on the same on-disk SQLite workspace) is safe and idempotent.
    - **No in-memory state survives across calls.** The fetcher carries
      a per-table row-count memo (used by the cardinality heuristic) and
      nothing else.

    This design closes two failure modes from the 2026-05-21
    model-template e2e run:

    - **Finding 05** (re-INSERT crashes with ``UNIQUE constraint failed``):
      ``_insert_rows`` no longer raises on existing RIDs.
    - **Finding A01** (silent row-drop on FK-column fetches): the fetcher
      doesn't try to dedup at the fetch boundary at all, so it can't
      conflate "we've seen this PK" with "we've seen this FK value".

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
        # Cached row counts per table, populated lazily by
        # fetch_by_rids_or_predicate. Per-fetcher; not engine-derived.
        # This is the only state the fetcher carries across calls within
        # its own lifetime, and it survives only the one denormalize call
        # that built the fetcher.
        self._counts: dict[str, int] = {}
        # Freshness ledger: ``time.monotonic()`` of the first fetch attempt
        # for each ``(table, rid_column, frozenset(rids))`` tuple. Used by
        # the denormalize layer to compute
        # :attr:`DenormalizeResult.cache_age_seconds` — the wall-clock age of
        # the *oldest* fetch that participated in this fetcher's lifetime
        # (SC-03, spec §6 "Freshness caveat" caller-visible freshness signal).
        # Recorded once per distinct key: a cache-hit re-visit by the
        # dedup-processed set in ``_populate_from_catalog_inner`` does NOT
        # overwrite an existing timestamp, so the ledger reports "when did
        # we first fetch this key", not "when was it most recently touched."
        self._fetch_ledger: dict[tuple[str, str, frozenset[str]], float] = {}

    def record_fetch_start(self, table: str, rid_column: str, rids: Iterable[str]) -> None:
        """Record the start time of a fetch attempt in the freshness ledger.

        Idempotent on key — if the same ``(table, rid_column, frozenset(rids))``
        tuple was already recorded, the existing timestamp is preserved. This
        is the contract that makes :attr:`DenormalizeResult.cache_age_seconds`
        report "wall-clock age of the *first* fetch that participated", not
        "the most recent re-touch."

        Callers in :func:`_populate_from_catalog_inner` invoke this
        immediately before :meth:`fetch_by_rids` for each distinct fetch
        key, so the ledger reflects fetch *attempts*, not cache hits. A
        cache-hit re-visit (where the key is already in the dedup
        ``processed`` set) does not call this and does not pollute the ledger.

        Args:
            table: Qualified table name (``"schema:table"``) — the same form
                passed to :meth:`fetch_by_rids`.
            rid_column: Column on *table* to filter on.
            rids: RID values being fetched. Stringified and frozen for use
                as the ledger key.

        Example::

            fetcher.record_fetch_start("isa:Image", "RID", ["IMG-1", "IMG-2"])
            fetcher.fetch_by_rids(
                table="isa:Image",
                rids=["IMG-1", "IMG-2"],
                target_table=image_t,
                rid_column="RID",
            )
        """
        key = (table, rid_column, frozenset(str(r) for r in rids))
        if key not in self._fetch_ledger:
            self._fetch_ledger[key] = time.monotonic()

    @property
    def fetch_ledger(self) -> dict[tuple[str, str, frozenset[str]], float]:
        """Read-only view of the freshness ledger.

        Maps ``(table, rid_column, frozenset(rids))`` to the
        ``time.monotonic()`` of the first fetch attempt for that key.
        Used by :func:`_denormalize_impl` to compute
        :attr:`DenormalizeResult.cache_age_seconds`.

        Returns:
            The ledger dict. Mutating the returned dict will mutate the
            fetcher's internal state; treat as read-only.
        """
        return self._fetch_ledger

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
        *page_size*, which signals end-of-data. Rows are inserted into
        *target_table* via :meth:`_insert_rows`, which uses INSERT-OR-IGNORE
        semantics — so a page row whose RID is already in the engine is
        skipped without error.

        Args:
            table: Qualified table name (``"schema:table"``).
            predicate: ERMrest filter predicate, or ``None`` for all rows.
            target_table: SQLAlchemy :class:`Table` to insert into.
            sort: Column(s) to sort by for keyset pagination.
            page_size: Rows per page (also the stop condition threshold).

        Returns:
            Number of rows actually written (skipped duplicates do not count).
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
            n += self._insert_rows(target_table, page)
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
        """Fetch rows whose *rid_column* matches any value in *rids*.

        Issues one or more HTTP requests via :meth:`_fetch_rid_batch_with_fallback`
        (GET → POST shrink-fallback for oversized URLs), then inserts the
        returned rows into *target_table* via :meth:`_insert_rows`.

        ``rid_column`` may be the table's primary key (``"RID"`` — the
        default and most common case) or a foreign-key column. When it
        is an FK, the server may return many rows per filter value;
        the fetcher does not assume one-to-one. This is the A01 case
        that previously caused silent row-drops.

        Args:
            table: Qualified table name (``"schema:table"``).
            rids: Iterable of values to filter by on *rid_column*.
            target_table: SQLAlchemy :class:`Table` to insert into.
            rid_column: Column name on *table* to filter on (default ``"RID"``).
            batch_size: Number of values per HTTP request.
            max_url_bytes: Byte threshold above which GET shrinks to POST.

        Returns:
            Number of rows actually written (rows already present by RID
            are skipped via INSERT-OR-IGNORE and do not count).
        """
        deduped = list(dict.fromkeys(str(x) for x in rids))
        if not deduped:
            return 0

        n = 0
        i = 0
        while i < len(deduped):
            batch = deduped[i : i + batch_size]
            rows = self._fetch_rid_batch_with_fallback(
                table=table,
                column=rid_column,
                rids=batch,
                max_url_bytes=max_url_bytes,
            )
            n += self._insert_rows(target_table, rows)
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
        """Fetch rows for *rids* via GET, shrinking + chunking if URL too long.

        Strategy, designed to preserve the row-completeness invariant
        (every RID in ``rids`` is requested exactly once across one or
        more HTTP calls, and every response row is returned):

        1. Try GET with the full list. If the estimated URL fits under
           ``max_url_bytes`` and the server accepts it, return the rows.
        2. If the URL is too long (either by local estimate or by a
           runtime "URL too long" error), shrink to the largest prefix
           that does fit and request *that* prefix via GET — then
           recurse on the remaining suffix, accumulating rows.
        3. If even a single-RID URL exceeds the limit (impossible in
           practice, but defensive), fall back to a POST request with
           the full original ``rids`` list.

        The earlier implementation shrunk to the first prefix that fit
        and returned the result, silently dropping the suffix — a SC-06
        row-completeness violation at the fetcher layer. The chunk-loop
        below preserves the "always POST as a last resort" fallback for
        clients without POST while guaranteeing every RID is requested.

        Args:
            table: Qualified table name (``"schema:table"``).
            column: Column to filter on.
            rids: RIDs to request.
            max_url_bytes: URL-length guard threshold.

        Returns:
            All rows from the server for *rids*, concatenated.
        """
        remaining = list(rids)
        out: list[dict[str, Any]] = []
        while remaining:
            estimated = 128 + 13 * len(remaining)
            if estimated <= max_url_bytes:
                # URL should fit — try GET on the entire remaining set.
                try:
                    out.extend(
                        self._client.fetch_rid_batch(
                            table=table, column=column, rids=remaining, method="GET"
                        )
                    )
                    return out
                except RuntimeError as exc:
                    if "too long" not in str(exc).lower():
                        raise
                    # Local estimate was optimistic — fall through to
                    # shrink-and-chunk.

            # Find the largest prefix that fits under the URL guard.
            # Solve 128 + 13 * k <= max_url_bytes for k.
            fits = (max_url_bytes - 128) // 13
            if fits <= 0:
                # max_url_bytes is so small that even one RID doesn't
                # fit — last-resort POST with the original rids list.
                logger.debug(
                    "POST fallback for table=%s column=%s rids=%d",
                    table,
                    column,
                    len(rids),
                )
                return self._client.fetch_rid_batch(
                    table=table, column=column, rids=rids, method="POST"
                )
            chunk = remaining[:fits]
            try:
                out.extend(
                    self._client.fetch_rid_batch(
                        table=table, column=column, rids=chunk, method="GET"
                    )
                )
            except RuntimeError as exc:
                if "too long" not in str(exc).lower():
                    raise
                # Local estimate was wrong even for this chunk size.
                # Halve and retry once; if that also rejects, POST is
                # the only remaining option.
                half = max(1, len(chunk) // 2)
                chunk = chunk[:half]
                try:
                    out.extend(
                        self._client.fetch_rid_batch(
                            table=table, column=column, rids=chunk, method="GET"
                        )
                    )
                except RuntimeError as exc2:
                    if "too long" not in str(exc2).lower():
                        raise
                    logger.debug(
                        "POST fallback after GET-shrink rejected: table=%s column=%s rids=%d",
                        table,
                        column,
                        len(rids),
                    )
                    return self._client.fetch_rid_batch(
                        table=table, column=column, rids=rids, method="POST"
                    )
            remaining = remaining[len(chunk) :]
        return out

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
        """Return the set of RIDs currently in *target_table*.

        Read directly from the engine — the engine is the only authoritative
        source for "what's in this workspace right now." There is no
        in-memory cache to consult.

        Args:
            table: Qualified table name (informational; used in the API for
                consistency with the rest of the class).
            target_table: SQLAlchemy ``Table`` to read RIDs from. Required;
                if ``None``, returns the empty set (we have no other way to
                know which engine table corresponds to *table*).

        Returns:
            Set of RID strings currently present in *target_table*.
        """
        if target_table is None:
            return set()
        with self._engine.connect() as conn:
            return {str(row[0]) for row in conn.execute(select(target_table.c.RID)) if row[0] is not None}

    def _insert_rows(self, target_table: Table, rows: list[dict[str, Any]]) -> int:
        """Insert *rows* into *target_table* using INSERT-OR-IGNORE.

        Contract (``docs/user-guide/denormalization.md`` §5):

        - For each row: if a row with the same RID already exists in
          ``target_table``, skip (do not update, do not crash).
          Otherwise insert.
        - Extra columns on incoming rows that are not declared on
          ``target_table`` are silently dropped (preserves the prior
          contract; useful because ERMrest queries return system
          columns like ``RCB`` that the local schema may not mirror).
        - Returns the count of rows actually written (not skipped).

        Uses SQLite's ``INSERT OR IGNORE`` via SQLAlchemy's
        ``sqlite_insert(...).on_conflict_do_nothing()``. The engine is
        always SQLite in this layer (see ``local_db/README.md``); if
        that ever changes, this method becomes the dialect-aware seam.
        """
        if not rows:
            return 0
        # SC-05: explicit guard — rows arriving without a ``RID`` key are
        # a programming error. The legacy contract relied on SQLite's
        # NOT NULL constraint to surface this (every supported engine
        # declares ``RID`` as the PK, and PKs are implicitly NOT NULL),
        # which produced an opaque ``IntegrityError`` from deep in the
        # dialect layer. Raise a clear ``ValueError`` instead so the
        # caller sees the bad row, and the contract is no longer
        # dialect-coupled.
        for r in rows:
            if "RID" not in r:
                raise ValueError(f"_insert_rows: row missing RID — programming error: {r!r}")
        cols = {c.name for c in target_table.columns}
        projected = [{k: v for k, v in r.items() if k in cols} for r in rows]
        if not projected:
            return 0
        stmt = sqlite_insert(target_table).on_conflict_do_nothing(index_elements=["RID"])
        with self._engine.begin() as conn:
            result = conn.execute(stmt, projected)
        # SQLAlchemy reports rowcount as the number of rows the database
        # actually inserted; for INSERT OR IGNORE, skipped rows do not count.
        # Fall back to len(projected) on dialects that don't report rowcount
        # reliably (none we use today; this is a defensive default).
        written = result.rowcount if result.rowcount is not None and result.rowcount >= 0 else len(projected)
        return written
