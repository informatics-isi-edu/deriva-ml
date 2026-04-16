"""Unit tests for local_db.paged_fetcher (against a fake ERMrest client)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from sqlalchemy import Column, MetaData, String, Table, create_engine, select

from deriva_ml.local_db.paged_fetcher import PagedFetcher


@dataclass
class FakePagedClient:
    """Deterministic stand-in for PagedFetcher's ERMrest dependency."""

    rows_by_table: dict[str, list[dict[str, Any]]]
    requests: list[tuple[str, dict]] = field(default_factory=list)
    max_get_bytes: int = 6144

    def count(self, table: str) -> int:
        self.requests.append(("count", {"table": table}))
        return len(self.rows_by_table.get(table, []))

    def fetch_page(
        self,
        table: str,
        sort: tuple[str, ...],
        after: tuple | None,
        predicate: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        self.requests.append(
            (
                "fetch_page",
                {
                    "table": table,
                    "sort": sort,
                    "after": after,
                    "predicate": predicate,
                    "limit": limit,
                },
            )
        )
        rows = list(self.rows_by_table.get(table, []))
        rows.sort(key=lambda r: tuple(r[c] for c in sort))
        if after is not None:
            rows = [r for r in rows if tuple(r[c] for c in sort) > tuple(after)]
        return rows[:limit]

    def fetch_rid_batch(
        self,
        table: str,
        column: str,
        rids: list[str],
        method: str = "GET",
    ) -> list[dict[str, Any]]:
        self.requests.append(
            (
                "fetch_rid_batch",
                {
                    "table": table,
                    "column": column,
                    "rids": list(rids),
                    "method": method,
                },
            )
        )
        if method == "GET":
            approx_url_bytes = 128 + 13 * len(rids)
            if approx_url_bytes > self.max_get_bytes:
                raise RuntimeError(f"GET URL too long ({approx_url_bytes} bytes)")
        rows = self.rows_by_table.get(table, [])
        want = set(rids)
        return [r for r in rows if r.get(column) in want]


def _make_target_table(engine, name: str = "Image") -> Table:
    md = MetaData()
    t = Table(
        name,
        md,
        Column("RID", String, primary_key=True),
        Column("Filename", String),
        Column("Subject", String),
    )
    md.create_all(engine)
    return t


def _rows_count(engine, table: Table) -> int:
    with engine.connect() as conn:
        return len(conn.execute(select(table)).fetchall())


def _make_rows(n: int, prefix: str = "R") -> list[dict[str, Any]]:
    """Generate n rows with RID=prefix+index, Filename, Subject."""
    return [{"RID": f"{prefix}{i:04d}", "Filename": f"file{i}.jpg", "Subject": f"sub{i}"} for i in range(n)]


@pytest.fixture
def engine():
    e = create_engine("sqlite:///:memory:", echo=False)
    return e


# ---------------------------------------------------------------------------
# TestFetchPredicate
# ---------------------------------------------------------------------------


class TestFetchPredicate:
    def test_keyset_paging_fetches_all(self, engine):
        """25 rows, page_size=10 → 3 page requests and all 25 rows in local DB."""
        rows = _make_rows(25)
        client = FakePagedClient(rows_by_table={"Image": rows})
        table = _make_target_table(engine, "Image")
        fetcher = PagedFetcher(client=client, engine=engine)

        result = fetcher.fetch_predicate("Image", None, table, sort=("RID",), page_size=10)

        assert result == 25
        assert _rows_count(engine, table) == 25
        page_requests = [r for r in client.requests if r[0] == "fetch_page"]
        # 3 requests: pages of 10, 10, 5 — the 5-row page terminates the loop
        assert len(page_requests) == 3

    def test_respects_predicate(self, engine):
        """Predicate string is forwarded to every fetch_page call."""
        rows = _make_rows(5)
        client = FakePagedClient(rows_by_table={"Image": rows})
        table = _make_target_table(engine, "Image")
        fetcher = PagedFetcher(client=client, engine=engine)

        fetcher.fetch_predicate("Image", "Subject=sub0", table, sort=("RID",), page_size=100)

        page_reqs = [r for r in client.requests if r[0] == "fetch_page"]
        assert all(r[1]["predicate"] == "Subject=sub0" for r in page_reqs)

    def test_empty_table_returns_zero(self, engine):
        """No rows → returns 0, nothing inserted."""
        client = FakePagedClient(rows_by_table={"Image": []})
        table = _make_target_table(engine, "Image")
        fetcher = PagedFetcher(client=client, engine=engine)

        result = fetcher.fetch_predicate("Image", None, table)

        assert result == 0
        assert _rows_count(engine, table) == 0

    def test_exact_page_boundary(self, engine):
        """20 rows, page_size=10 — should issue 3 requests (10, 10, then empty)."""
        rows = _make_rows(20)
        client = FakePagedClient(rows_by_table={"Image": rows})
        table = _make_target_table(engine, "Image")
        fetcher = PagedFetcher(client=client, engine=engine)

        result = fetcher.fetch_predicate("Image", None, table, sort=("RID",), page_size=10)

        assert result == 20
        assert _rows_count(engine, table) == 20
        page_requests = [r for r in client.requests if r[0] == "fetch_page"]
        # 2 full pages of 10 → must issue a 3rd to confirm exhaustion
        assert len(page_requests) == 3

    def test_single_page(self, engine):
        """Fewer rows than page_size → exactly 1 request."""
        rows = _make_rows(7)
        client = FakePagedClient(rows_by_table={"Image": rows})
        table = _make_target_table(engine, "Image")
        fetcher = PagedFetcher(client=client, engine=engine)

        result = fetcher.fetch_predicate("Image", None, table, sort=("RID",), page_size=100)

        assert result == 7
        page_requests = [r for r in client.requests if r[0] == "fetch_page"]
        assert len(page_requests) == 1


# ---------------------------------------------------------------------------
# TestFetchByRids
# ---------------------------------------------------------------------------


class TestFetchByRids:
    def test_batches_at_default_size(self, engine):
        """1200 rows, batch_size=500 → 3 batches (500, 500, 200).

        Both the fetcher's max_url_bytes and the fake client's max_get_bytes
        are set to 10000 to prevent URL-length shrinkage from interfering.
        """
        rows = _make_rows(1200)
        client = FakePagedClient(rows_by_table={"Image": rows}, max_get_bytes=10000)
        table = _make_target_table(engine, "Image")
        fetcher = PagedFetcher(client=client, engine=engine)
        rids = [r["RID"] for r in rows]

        fetcher.fetch_by_rids("Image", rids, table, batch_size=500, max_url_bytes=10000)

        batch_requests = [r for r in client.requests if r[0] == "fetch_rid_batch"]
        assert len(batch_requests) == 3
        assert len(batch_requests[0][1]["rids"]) == 500
        assert len(batch_requests[1][1]["rids"]) == 500
        assert len(batch_requests[2][1]["rids"]) == 200

    def test_deduplication_across_calls(self, engine):
        """Second call with overlapping RIDs only requests the new ones."""
        rows = _make_rows(5)
        client = FakePagedClient(rows_by_table={"Image": rows})
        table = _make_target_table(engine, "Image")
        fetcher = PagedFetcher(client=client, engine=engine)
        all_rids = [r["RID"] for r in rows]

        # First call: R0000, R0001, R0002
        fetcher.fetch_by_rids("Image", all_rids[:3], table, batch_size=500)
        client.requests.clear()

        # Second call: R0001, R0002 already seen; only R0003, R0004 should be fetched
        fetcher.fetch_by_rids("Image", all_rids[1:], table, batch_size=500)

        batch_requests = [r for r in client.requests if r[0] == "fetch_rid_batch"]
        assert len(batch_requests) == 1
        assert set(batch_requests[0][1]["rids"]) == {all_rids[3], all_rids[4]}

    def test_post_fallback_on_long_url(self, engine):
        """600 rows, max_url_bytes tiny → POST fallback triggered.

        We set max_url_bytes=1 so the estimated URL always exceeds it and
        the shrink loop immediately falls through to POST.
        """
        rows = _make_rows(600)
        # client max_get_bytes also set to 1 so the fake raises for any GET
        client = FakePagedClient(rows_by_table={"Image": rows}, max_get_bytes=1)
        table = _make_target_table(engine, "Image")
        fetcher = PagedFetcher(client=client, engine=engine)
        rids = [r["RID"] for r in rows]

        fetcher.fetch_by_rids("Image", rids, table, batch_size=600, max_url_bytes=1)

        post_requests = [r for r in client.requests if r[0] == "fetch_rid_batch" and r[1]["method"] == "POST"]
        assert len(post_requests) >= 1

    def test_shrinks_batch_before_post(self, engine):
        """200 rows split into two outer batches of 100; each fits GET limit → GET only.

        We set batch_size=100 so the outer loop produces two batches of exactly
        100.  max_url_bytes = 128 + 13*100 = 1428 fits 100 but not 200, and the
        fake client's max_get_bytes matches.  Both batches should succeed via GET
        with no POST fallback.
        """
        rows = _make_rows(200)
        # 128 + 13*100 = 1428 fits 100; 128 + 13*200 = 2728 does not
        threshold = 128 + 13 * 100  # 1428
        client = FakePagedClient(rows_by_table={"Image": rows}, max_get_bytes=threshold)
        table = _make_target_table(engine, "Image")
        fetcher = PagedFetcher(client=client, engine=engine)
        rids = [r["RID"] for r in rows]

        fetcher.fetch_by_rids("Image", rids, table, batch_size=100, max_url_bytes=threshold)

        batch_requests = [r for r in client.requests if r[0] == "fetch_rid_batch"]
        get_requests = [r for r in batch_requests if r[1]["method"] == "GET"]
        post_requests = [r for r in batch_requests if r[1]["method"] == "POST"]
        # Two outer batches of 100, each fits GET — no POST
        assert len(get_requests) == 2
        assert len(post_requests) == 0

    def test_empty_rid_set_returns_zero(self, engine):
        """Empty rids list → 0 returned, no requests issued."""
        client = FakePagedClient(rows_by_table={"Image": _make_rows(10)})
        table = _make_target_table(engine, "Image")
        fetcher = PagedFetcher(client=client, engine=engine)

        result = fetcher.fetch_by_rids("Image", [], table)

        assert result == 0
        assert len(client.requests) == 0

    def test_no_duplicate_inserts(self, engine):
        """Fetch same RIDs twice → DB still has each row only once."""
        rows = _make_rows(5)
        client = FakePagedClient(rows_by_table={"Image": rows})
        table = _make_target_table(engine, "Image")
        fetcher = PagedFetcher(client=client, engine=engine)
        rids = [r["RID"] for r in rows]

        fetcher.fetch_by_rids("Image", rids, table, batch_size=500)
        fetcher.fetch_by_rids("Image", rids, table, batch_size=500)

        assert _rows_count(engine, table) == 5


# ---------------------------------------------------------------------------
# TestFetchedRids
# ---------------------------------------------------------------------------


class TestFetchedRids:
    def test_tracks_rids_from_predicate_fetch(self, engine):
        """After fetch_predicate, fetched_rids returns the correct set."""
        rows = _make_rows(5)
        client = FakePagedClient(rows_by_table={"Image": rows})
        table = _make_target_table(engine, "Image")
        fetcher = PagedFetcher(client=client, engine=engine)

        fetcher.fetch_predicate("Image", None, table)
        result = fetcher.fetched_rids("Image")

        expected = {r["RID"] for r in rows}
        assert result == expected

    def test_tracks_rids_from_rid_fetch(self, engine):
        """After fetch_by_rids, fetched_rids returns the fetched RIDs."""
        rows = _make_rows(5)
        client = FakePagedClient(rows_by_table={"Image": rows})
        table = _make_target_table(engine, "Image")
        fetcher = PagedFetcher(client=client, engine=engine)
        rids = [r["RID"] for r in rows[:3]]

        fetcher.fetch_by_rids("Image", rids, table, batch_size=500)
        result = fetcher.fetched_rids("Image")

        assert result == set(rids)

    def test_empty_when_nothing_fetched(self, engine):
        """Returns empty set before any fetch."""
        client = FakePagedClient(rows_by_table={"Image": _make_rows(5)})
        fetcher = PagedFetcher(client=client, engine=engine)

        result = fetcher.fetched_rids("Image")
        assert result == set()

    def test_reads_from_db_when_no_memory_cache(self, engine):
        """New PagedFetcher with empty cache reads RIDs from the DB via target_table."""
        rows = _make_rows(4)
        client = FakePagedClient(rows_by_table={"Image": rows})
        table = _make_target_table(engine, "Image")

        # Pre-populate DB using a first fetcher
        fetcher1 = PagedFetcher(client=client, engine=engine)
        fetcher1.fetch_predicate("Image", None, table)

        # New fetcher — no in-memory cache
        fetcher2 = PagedFetcher(client=client, engine=engine)
        result = fetcher2.fetched_rids("Image", target_table=table)

        expected = {r["RID"] for r in rows}
        assert result == expected


# ---------------------------------------------------------------------------
# TestCardinalityHeuristic
# ---------------------------------------------------------------------------


class TestCardinalityHeuristic:
    def test_switches_to_predicate_when_set_large(self, engine):
        """10 rows total, requesting 9 (>50%) → fetch_page used, not fetch_rid_batch."""
        rows = _make_rows(10)
        client = FakePagedClient(rows_by_table={"Image": rows})
        table = _make_target_table(engine, "Image")
        fetcher = PagedFetcher(client=client, engine=engine)
        rids = [r["RID"] for r in rows[:9]]

        fetcher.fetch_by_rids_or_predicate("Image", rids, table, cardinality_threshold=0.5)

        page_reqs = [r for r in client.requests if r[0] == "fetch_page"]
        rid_batch_reqs = [r for r in client.requests if r[0] == "fetch_rid_batch"]
        assert len(page_reqs) >= 1
        assert len(rid_batch_reqs) == 0

    def test_uses_rid_batch_when_set_small(self, engine):
        """100 rows total, requesting 5 (5%) → fetch_rid_batch used, not fetch_page."""
        rows = _make_rows(100)
        client = FakePagedClient(rows_by_table={"Image": rows})
        table = _make_target_table(engine, "Image")
        fetcher = PagedFetcher(client=client, engine=engine)
        rids = [r["RID"] for r in rows[:5]]

        fetcher.fetch_by_rids_or_predicate("Image", rids, table, cardinality_threshold=0.5)

        page_reqs = [r for r in client.requests if r[0] == "fetch_page"]
        rid_batch_reqs = [r for r in client.requests if r[0] == "fetch_rid_batch"]
        assert len(rid_batch_reqs) >= 1
        assert len(page_reqs) == 0

    def test_memoizes_count(self, engine):
        """count() called only once even across two calls to fetch_by_rids_or_predicate."""
        rows = _make_rows(100)
        client = FakePagedClient(rows_by_table={"Image": rows})
        table = _make_target_table(engine, "Image")
        fetcher = PagedFetcher(client=client, engine=engine)
        rids = [r["RID"] for r in rows[:5]]

        fetcher.fetch_by_rids_or_predicate("Image", rids, table)
        fetcher.fetch_by_rids_or_predicate("Image", rids, table)

        count_reqs = [r for r in client.requests if r[0] == "count"]
        assert len(count_reqs) == 1

    def test_predicate_path_filters_unwanted_rids(self, engine):
        """When predicate path used, rows not in wanted set are removed from target_table."""
        rows = _make_rows(10)
        client = FakePagedClient(rows_by_table={"Image": rows})
        table = _make_target_table(engine, "Image")
        fetcher = PagedFetcher(client=client, engine=engine)
        # Ask for 9 rows (>50%), so predicate path is taken; only those 9 should remain
        wanted_rids = [r["RID"] for r in rows[:9]]

        fetcher.fetch_by_rids_or_predicate("Image", wanted_rids, table, cardinality_threshold=0.5)

        with engine.connect() as conn:
            stored = {row[0] for row in conn.execute(select(table.c["RID"])).fetchall()}
        assert stored == set(wanted_rids)
