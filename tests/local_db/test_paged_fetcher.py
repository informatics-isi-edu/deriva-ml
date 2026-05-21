"""Unit tests for local_db.paged_fetcher (against a fake ERMrest client)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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

    def test_overlapping_calls_keep_engine_consistent(self, engine):
        """Two ``fetch_by_rids`` calls with overlapping RIDs land the engine
        in the right shape (each row present exactly once), even though
        the stateless contract re-requests overlapping RIDs from the
        server on the second call.

        Pre-2026-05-22 the fetcher carried a per-instance ``_seen`` set
        that suppressed the second-call network request for overlapping
        RIDs. That dedup was removed when finding A01 showed it could
        not be made safe for non-PK ``rid_column`` values (see
        ``docs/design/denormalization.md`` §4). The new contract is:
        the server is the authority for what rows match; the database's
        UNIQUE constraint via INSERT-OR-IGNORE prevents the second
        call's writes from crashing or duplicating. Test reflects that:
        we assert correctness of the engine state, not the network
        request count.
        """
        rows = _make_rows(5)
        client = FakePagedClient(rows_by_table={"Image": rows})
        table = _make_target_table(engine, "Image")
        fetcher = PagedFetcher(client=client, engine=engine)
        all_rids = [r["RID"] for r in rows]

        fetcher.fetch_by_rids("Image", all_rids[:3], table, batch_size=500)
        assert _rows_count(engine, table) == 3

        # Second call overlaps the first by R0001, R0002 and adds R0003, R0004.
        # Stateless contract: the request goes out; INSERT-OR-IGNORE handles
        # duplicates; the engine ends up with all four new RIDs and zero
        # duplicates.
        fetcher.fetch_by_rids("Image", all_rids[1:], table, batch_size=500)
        assert _rows_count(engine, table) == 5
        assert fetcher.fetched_rids("Image", target_table=table) == set(all_rids)

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

    def test_two_fetchers_same_engine_overlapping_rids(self, engine):
        """Two PagedFetcher instances against one engine — second must not
        crash on the rows the first inserted.

        Regression for the finding-05 root cause: ``_populate_from_catalog``
        builds a fresh ``PagedFetcher`` on every call, and the per-fetcher
        ``_seen`` set starts empty. Before this test was added, the second
        fetcher's ``fetch_by_rids`` would try to INSERT rows already in the
        DB and raise ``UNIQUE constraint failed``.
        """
        rows = _make_rows(5)
        client = FakePagedClient(rows_by_table={"Image": rows})
        table = _make_target_table(engine, "Image")
        rids = [r["RID"] for r in rows]

        # First fetcher inserts all five rows.
        fetcher_a = PagedFetcher(client=client, engine=engine)
        fetcher_a.fetch_by_rids("Image", rids, table, batch_size=500)
        assert _rows_count(engine, table) == 5

        # Second fetcher (fresh instance, fresh _seen) requests the same
        # RIDs. It must not crash on the duplicate INSERT.
        fetcher_b = PagedFetcher(client=client, engine=engine)
        fetcher_b.fetch_by_rids("Image", rids, table, batch_size=500)
        assert _rows_count(engine, table) == 5

    def test_fresh_fetcher_against_pre_populated_engine(self, engine):
        """Cross-session simulation — engine has rows from a 'prior' run,
        a fresh fetcher must cope.

        The on-disk SQLite workspace persists across Python processes, so
        a new ``DerivaML`` instance pointed at the same catalog sees rows
        from the previous session. The fetcher must not blindly INSERT.
        """
        rows = _make_rows(5)
        client = FakePagedClient(rows_by_table={"Image": rows})
        table = _make_target_table(engine, "Image")
        rids = [r["RID"] for r in rows]

        # Simulate a prior session having populated the DB.
        with engine.begin() as conn:
            from sqlalchemy import insert as _insert

            conn.execute(_insert(table), rows)
        assert _rows_count(engine, table) == 5

        # New fetcher (mimicking a fresh process / fresh DerivaML) tries
        # to fetch the same RIDs. Must succeed without crash.
        fetcher = PagedFetcher(client=client, engine=engine)
        fetcher.fetch_by_rids("Image", rids, table, batch_size=500)
        assert _rows_count(engine, table) == 5

    def test_two_fetchers_partial_overlap(self, engine):
        """Two fetchers, overlapping but not identical RID sets — new RIDs
        must be inserted, overlapping ones must be no-ops.
        """
        rows = _make_rows(10)
        client = FakePagedClient(rows_by_table={"Image": rows})
        table = _make_target_table(engine, "Image")
        rids = [r["RID"] for r in rows]

        fetcher_a = PagedFetcher(client=client, engine=engine)
        fetcher_a.fetch_by_rids("Image", rids[:6], table, batch_size=500)
        assert _rows_count(engine, table) == 6

        # Second fetcher requests RIDs 4..9 — 4 and 5 already present, 6..9 new.
        fetcher_b = PagedFetcher(client=client, engine=engine)
        fetcher_b.fetch_by_rids("Image", rids[4:], table, batch_size=500)
        assert _rows_count(engine, table) == 10

    def test_two_fetchers_predicate_then_rids_overlap(self, engine):
        """First fetcher populates via the page/predicate path; second
        fetcher's fetch_by_rids on the same RIDs must still cope.

        This exercises the predicate→RID interleaving inside one engine
        but across fetcher instances.
        """
        rows = _make_rows(5)
        client = FakePagedClient(rows_by_table={"Image": rows})
        table = _make_target_table(engine, "Image")
        rids = [r["RID"] for r in rows]

        fetcher_a = PagedFetcher(client=client, engine=engine)
        fetcher_a.fetch_predicate("Image", predicate=None, target_table=table, sort=("RID",))
        assert _rows_count(engine, table) == 5

        fetcher_b = PagedFetcher(client=client, engine=engine)
        fetcher_b.fetch_by_rids("Image", rids, table, batch_size=500)
        assert _rows_count(engine, table) == 5

    def test_fetch_by_fk_column_multiple_rows_per_value(self, engine):
        """B.4 in the test matrix (``docs/design/denormalization.md`` §8):
        ``fetch_by_rids`` filtering on an FK column where multiple rows
        share the same FK value must fetch ALL matching rows, not collapse
        to one per FK.

        This is the unit-level reproduction of finding A01. The pre-fix
        implementation hydrated a "seen" set from the engine's
        ``rid_column`` values, then treated any FK value already in the
        engine as "we've seen this row" — silently dropping every row
        beyond the first per FK. The stateless contract has no such
        cache; the server's response is taken at face value and inserted
        via INSERT-OR-IGNORE (which dedups by RID, not by FK).
        """
        # Build an "Execution_Image_Quality"-shaped fake table: 4 Images,
        # 3 feature rows per Image, each with a unique RID. The FK column
        # is "Image" — non-unique across rows.
        feature_rows = []
        for i in range(4):
            for j in range(3):
                feature_rows.append(
                    {
                        "RID": f"F{i:02d}{j}",
                        "Filename": f"feature_{i}_{j}",
                        "Subject": f"img_{i}",  # the "Image" FK
                    }
                )
        # The FakePagedClient filters by ``column`` arg, so use Subject
        # as the stand-in for the FK column in this fixture.
        client = FakePagedClient(rows_by_table={"FeatureValues": feature_rows})
        feature_table = _make_target_table(engine, "FeatureValues")
        fetcher = PagedFetcher(client=client, engine=engine)

        # Request all rows for the 4 Image FK values. Expected: 12 rows
        # in the engine (4 Images × 3 features each).
        image_fks = [f"img_{i}" for i in range(4)]
        fetcher.fetch_by_rids(
            "FeatureValues",
            image_fks,
            feature_table,
            rid_column="Subject",  # the FK column, not the PK
            batch_size=500,
        )
        assert _rows_count(engine, feature_table) == 12, (
            "Expected 12 rows (4 Images × 3 features each); finding A01 "
            "previously caused this to be 4 (one per Image, the rest "
            "silently dropped)."
        )

        # A second fetcher against the now-populated engine must also
        # produce the same 12 rows when asked for the same FK values —
        # i.e., the operation is idempotent across fetcher instances
        # AND across the cross-session form of finding 05.
        fetcher_b = PagedFetcher(client=client, engine=engine)
        fetcher_b.fetch_by_rids(
            "FeatureValues",
            image_fks,
            feature_table,
            rid_column="Subject",
            batch_size=500,
        )
        assert _rows_count(engine, feature_table) == 12

    def test_fetch_by_fk_column_partial_overlap(self, engine):
        """B.5 in the test matrix: partial-overlap FK fetch.

        Some FK values' rows are already in the engine; the server has
        additional rows for some of the same FK values that the prior
        fetch did not capture (because the prior fetch was for a
        different scope). A second ``fetch_by_rids`` by the FK column
        must pull in the new server rows and integrate them with the
        already-present ones via INSERT-OR-IGNORE.
        """
        # Same shape as B.4 but only half the feature rows are in the
        # engine at the start.
        feature_rows = []
        for i in range(2):
            for j in range(3):
                feature_rows.append(
                    {
                        "RID": f"P{i:02d}{j}",
                        "Filename": f"f_{i}_{j}",
                        "Subject": f"img_{i}",
                    }
                )
        client = FakePagedClient(rows_by_table={"FeatureValues": feature_rows})
        feature_table = _make_target_table(engine, "FeatureValues")

        # Pre-populate engine with the first Image's feature rows.
        from sqlalchemy import insert as _insert

        with engine.begin() as conn:
            conn.execute(_insert(feature_table), feature_rows[:3])
        assert _rows_count(engine, feature_table) == 3

        # Now ask the fetcher for BOTH Images. Engine should end with
        # all 6 rows; the existing 3 are kept, the 3 new ones are added.
        fetcher = PagedFetcher(client=client, engine=engine)
        fetcher.fetch_by_rids(
            "FeatureValues",
            ["img_0", "img_1"],
            feature_table,
            rid_column="Subject",
        )
        assert _rows_count(engine, feature_table) == 6

    def test_insert_with_missing_rid_in_row(self, tmp_path: Path) -> None:
        """A.4 in the test matrix: insert behavior when an incoming row
        has no ``RID`` column.

        Pin the behavior: INSERT-OR-IGNORE relies on RID being the
        conflict-target column. A row without RID violates the table's
        NOT-NULL primary-key constraint and the engine raises. This is
        the right behavior — rows from ermrest always carry RID, so a
        missing RID is a programming error worth surfacing.
        """
        from sqlalchemy.exc import IntegrityError

        engine = create_engine(f"sqlite:///{tmp_path / 'wd.sqlite'}", future=True)
        target = _make_target_table(engine)
        client = FakePagedClient(rows_by_table={"Image": []})
        fetcher = PagedFetcher(client=client, engine=engine)

        # Calling _insert_rows directly with a bad row should raise.
        with pytest.raises(IntegrityError):
            fetcher._insert_rows(target, [{"Filename": "x", "Subject": "y"}])

    def test_predicate_against_pre_populated_engine(self, engine):
        """B.6 in the test matrix: ``fetch_predicate`` against an engine
        that already holds some of the rows. INSERT-OR-IGNORE keeps the
        engine consistent without crashing.

        Pre-A01 the predicate path conditioned page-row insertion on the
        seen-set, which is sufficient when each row's RID is the only
        identity. Post-A01 we removed the seen-set, so the predicate
        path now also relies on INSERT-OR-IGNORE for re-entry safety.
        This test pins that behavior.
        """
        rows = _make_rows(5)
        client = FakePagedClient(rows_by_table={"Image": rows})
        table = _make_target_table(engine, "Image")

        # Pre-populate with the first 2 rows (simulating a prior session).
        from sqlalchemy import insert as _insert

        with engine.begin() as conn:
            conn.execute(_insert(table), rows[:2])
        assert _rows_count(engine, table) == 2

        # fetch_predicate over the same table — should fetch all 5,
        # INSERT-OR-IGNORE the 2 already present, insert the 3 new.
        fetcher = PagedFetcher(client=client, engine=engine)
        fetcher.fetch_predicate("Image", predicate=None, target_table=table)
        assert _rows_count(engine, table) == 5

    def test_extra_columns_in_rows_are_silently_dropped(self, tmp_path: Path) -> None:
        engine = create_engine(f"sqlite:///{tmp_path / 'wd.sqlite'}", future=True)
        target = _make_target_table(engine)  # has RID, Filename, Subject

        # Rows have an extra column not in the target table
        rows = [
            {"RID": "R1", "Filename": "f1", "Subject": "S1", "Extra_Column": "should_be_dropped"},
        ]
        client = FakePagedClient(rows_by_table={"Image": rows})

        f = PagedFetcher(client=client, engine=engine)
        n = f.fetch_by_rids("Image", ["R1"], target, rid_column="RID")
        assert n == 1
        assert _rows_count(engine, target) == 1


# ---------------------------------------------------------------------------
# TestFetchedRids
# ---------------------------------------------------------------------------


class TestFetchedRids:
    def test_tracks_rids_from_predicate_fetch(self, engine):
        """After ``fetch_predicate``, ``fetched_rids`` (passing the target
        table) returns the set actually present in the engine.

        Post-stateless-refactor (``docs/design/denormalization.md`` §4),
        ``fetched_rids`` reads directly from the engine; callers must
        pass ``target_table`` so the fetcher knows which SQLAlchemy
        table to read.
        """
        rows = _make_rows(5)
        client = FakePagedClient(rows_by_table={"Image": rows})
        table = _make_target_table(engine, "Image")
        fetcher = PagedFetcher(client=client, engine=engine)

        fetcher.fetch_predicate("Image", None, table)
        result = fetcher.fetched_rids("Image", target_table=table)

        expected = {r["RID"] for r in rows}
        assert result == expected

    def test_tracks_rids_from_rid_fetch(self, engine):
        """After ``fetch_by_rids``, ``fetched_rids`` (passing the target
        table) returns the set actually present in the engine.
        """
        rows = _make_rows(5)
        client = FakePagedClient(rows_by_table={"Image": rows})
        table = _make_target_table(engine, "Image")
        fetcher = PagedFetcher(client=client, engine=engine)
        rids = [r["RID"] for r in rows[:3]]

        fetcher.fetch_by_rids("Image", rids, table, batch_size=500)
        result = fetcher.fetched_rids("Image", target_table=table)

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

    def test_zero_total_uses_rid_batch(self, tmp_path: Path) -> None:
        """When the table is empty (count=0), RID-batch path is taken (no division by zero)."""
        engine = create_engine(f"sqlite:///{tmp_path / 'wd.sqlite'}", future=True)
        target = _make_target_table(engine)

        client = FakePagedClient(rows_by_table={"Image": []})

        f = PagedFetcher(client=client, engine=engine)
        n = f.fetch_by_rids_or_predicate(
            table="Image",
            rids=["R0", "R1"],
            target_table=target,
            rid_column="RID",
            sort=("RID",),
            cardinality_threshold=0.5,
        )
        assert n == 0
        # Should not have used fetch_page (since total=0, the guard prevents division)
        methods_used = {r[0] for r in client.requests}
        assert "count" in methods_used
        assert "fetch_rid_batch" in methods_used


# ---------- ErmrestPagedClient URL-construction tests ---------- #


class _MockCatalog:
    """Minimal mock of ErmrestCatalog for adapter unit tests."""

    def __init__(self, *, get_responses=None, post_responses=None):
        self.get_calls: list[str] = []
        self.post_calls: list[tuple[str, Any]] = []
        self._get_responses = get_responses or {}
        self._post_responses = post_responses or {}
        self.catalog_id = "1"

    def get(self, url, headers=None):
        self.get_calls.append(url)

        class R:
            def __init__(self, data):
                self._d = data

            def json(self):
                return self._d

            def raise_for_status(self):
                return None

        return R(self._get_responses.get(url, []))

    def post(self, url, json=None, headers=None):
        self.post_calls.append((url, json))

        class R:
            def __init__(self, data):
                self._d = data

            def json(self):
                return self._d

            def raise_for_status(self):
                return None

        return R(self._post_responses.get(url, []))


class TestErmrestPagedClient:
    def test_count_uses_aggregate_endpoint(self) -> None:
        from deriva_ml.local_db.paged_fetcher_ermrest import ErmrestPagedClient

        cat = _MockCatalog(
            get_responses={
                "/aggregate/isa:Image/n:=cnt(*)": [{"n": 42}],
            }
        )
        c = ErmrestPagedClient(catalog=cat, catalog_id="1")
        assert c.count("isa:Image") == 42
        assert len(cat.get_calls) == 1

    def test_fetch_page_constructs_correct_url(self) -> None:
        from deriva_ml.local_db.paged_fetcher_ermrest import ErmrestPagedClient

        cat = _MockCatalog(
            get_responses={
                "/entity/isa:Image@sort(RID)?limit=10": [{"RID": "R1", "Filename": "f1"}],
            }
        )
        c = ErmrestPagedClient(catalog=cat, catalog_id="1")
        rows = c.fetch_page("isa:Image", ("RID",), None, None, 10)
        assert len(rows) == 1
        assert "sort(RID)" in cat.get_calls[0]

    def test_fetch_page_with_after(self) -> None:
        from deriva_ml.local_db.paged_fetcher_ermrest import ErmrestPagedClient

        cat = _MockCatalog()
        c = ErmrestPagedClient(catalog=cat, catalog_id="1")
        c.fetch_page("isa:Image", ("RID",), ("R5",), None, 10)
        assert "@after(R5)" in cat.get_calls[0]

    def test_fetch_page_with_predicate(self) -> None:
        from deriva_ml.local_db.paged_fetcher_ermrest import ErmrestPagedClient

        cat = _MockCatalog()
        c = ErmrestPagedClient(catalog=cat, catalog_id="1")
        c.fetch_page("isa:Image", ("RID",), None, "Subject=S1", 10)
        assert "/Subject=S1@sort" in cat.get_calls[0]

    def test_fetch_rid_batch_get(self) -> None:
        from deriva_ml.local_db.paged_fetcher_ermrest import ErmrestPagedClient

        cat = _MockCatalog(
            get_responses={
                "/entity/isa:Image/RID=any(R1,R2)": [{"RID": "R1"}, {"RID": "R2"}],
            }
        )
        c = ErmrestPagedClient(catalog=cat, catalog_id="1")
        rows = c.fetch_rid_batch("isa:Image", "RID", ["R1", "R2"], method="GET")
        assert len(rows) == 2

    def test_fetch_rid_batch_raises_on_long_url(self) -> None:
        from deriva_ml.local_db.paged_fetcher_ermrest import ErmrestPagedClient

        cat = _MockCatalog()
        c = ErmrestPagedClient(catalog=cat, catalog_id="1")
        long_rids = [f"R{i:05d}" for i in range(1000)]  # ~8000+ chars
        with pytest.raises(RuntimeError, match="too long"):
            c.fetch_rid_batch("isa:Image", "RID", long_rids, method="GET")

    def test_fetch_rid_batch_post_raises(self) -> None:
        from deriva_ml.local_db.paged_fetcher_ermrest import ErmrestPagedClient

        cat = _MockCatalog()
        c = ErmrestPagedClient(catalog=cat, catalog_id="1")
        with pytest.raises(RuntimeError, match="not supported"):
            c.fetch_rid_batch("isa:Image", "RID", ["R1"], method="POST")

    def test_catalog_id_from_attribute(self) -> None:
        from deriva_ml.local_db.paged_fetcher_ermrest import ErmrestPagedClient

        cat = _MockCatalog()
        cat.catalog_id = "99"
        c = ErmrestPagedClient(catalog=cat)  # No explicit catalog_id
        assert c._catalog_id == "99"


class _FailingMockCatalog:
    """Mock catalog whose get() returns an error response."""

    catalog_id = "1"

    def get(self, url, headers=None):
        class R:
            def raise_for_status(self):
                import requests

                raise requests.exceptions.HTTPError("404 Not Found")

            def json(self):
                return []

        return R()

    def post(self, url, json=None, headers=None):
        return self.get(url)


class TestErmrestPagedClientErrors:
    def test_count_propagates_http_error(self) -> None:
        from deriva_ml.local_db.paged_fetcher_ermrest import ErmrestPagedClient

        c = ErmrestPagedClient(catalog=_FailingMockCatalog(), catalog_id="1")
        import requests.exceptions

        with pytest.raises(requests.exceptions.HTTPError):
            c.count("isa:Image")

    def test_fetch_page_propagates_http_error(self) -> None:
        from deriva_ml.local_db.paged_fetcher_ermrest import ErmrestPagedClient

        c = ErmrestPagedClient(catalog=_FailingMockCatalog(), catalog_id="1")
        import requests.exceptions

        with pytest.raises(requests.exceptions.HTTPError):
            c.fetch_page("isa:Image", ("RID",), None, None, 10)
