"""Unit tests for ``deriva_ml.dataset._estimate`` (audit P1 Ds-est).

Pre-extraction, ``Dataset.estimate_bag_size`` was a 220-line
god-function whose fine-grained steps could not be unit-tested
without a live catalog. The audit asked for three helpers that
each have a clear contract; this file pins those contracts with
pure-Python tests.

Coverage layers:

1. ``_extract_path`` — URI string-stripping. Pure.
2. ``build_estimate_queries`` — given a synthetic ``table_queries``
   shape, returns the expected list of ``QueryItem`` records.
   Uses ``MagicMock`` to stand in for datapaths and tables.
3. ``run_estimate_queries`` — async orchestration with a mock
   catalog that returns canned JSON. Verifies grouping by table
   + query type, the per-query failure-swallowing contract, and
   that ``catalog.close`` is called even on partial failure.
4. ``assemble_estimate`` — given the three orchestrator outputs,
   produces the final estimate dict. Pinned via synthetic inputs.

No live catalog required.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from deriva_ml.dataset._estimate import (
    QueryItem,
    _extract_path,
    assemble_estimate,
    build_estimate_queries,
    run_estimate_queries,
)


class TestExtractPath:
    """``_extract_path`` strips the catalog-server prefix."""

    def test_entity_uri(self):
        assert (
            _extract_path("https://srv/ermrest/catalog/3/entity/X:Y")
            == "/entity/X:Y"
        )

    def test_aggregate_uri(self):
        assert (
            _extract_path("https://srv/ermrest/catalog/3/aggregate/X:Y/cnt(RID)")
            == "/aggregate/X:Y/cnt(RID)"
        )

    def test_attribute_uri(self):
        assert (
            _extract_path("https://srv/ermrest/catalog/3/attribute/X:Y/RID,Length")
            == "/attribute/X:Y/RID,Length"
        )

    def test_unrecognised_uri_raises(self):
        with pytest.raises(ValueError, match="Cannot extract"):
            _extract_path("https://srv/ermrest/catalog/3/some-other-shape")


def _fake_datapath(entity_uri: str, attr_uri: str) -> MagicMock:
    """Build a datapath stub whose ``.uri`` returns ``entity_uri`` and whose
    ``.attributes(...)`` returns an object with ``.uri == attr_uri``."""
    dp = MagicMock()
    dp.uri = entity_uri
    dp.attributes.return_value.uri = attr_uri
    return dp


class TestBuildEstimateQueries:
    """``build_estimate_queries`` emits one item per query type."""

    def test_empty_input(self):
        assert build_estimate_queries({}) == []

    def test_non_asset_single_path_emits_csv_and_sample(self):
        """Non-asset table, one path → ``csv`` + ``sample``, no ``fetch``."""
        dp = _fake_datapath(
            entity_uri="https://srv/ermrest/catalog/3/entity/X:Y",
            attr_uri="https://srv/ermrest/catalog/3/attribute/X:Y/RID",
        )
        target = MagicMock()
        target.RID = "RID-col-stub"
        items = build_estimate_queries({"Y": [(dp, target, False)]})
        assert [it.query_type for it in items] == ["csv", "sample"]
        assert items[0].path == "/attribute/X:Y/RID"
        assert items[1].path == "/entity/X:Y?limit=100"

    def test_asset_single_path_emits_csv_fetch_sample(self):
        """Asset table → ``csv`` + ``fetch`` + ``sample``."""
        dp = _fake_datapath(
            entity_uri="https://srv/ermrest/catalog/3/entity/X:Y",
            attr_uri="https://srv/ermrest/catalog/3/attribute/X:Y/RID",
        )
        target = MagicMock()
        target.RID = "RID-col-stub"
        items = build_estimate_queries({"Y": [(dp, target, True)]})
        assert [it.query_type for it in items] == ["csv", "fetch", "sample"]
        assert items[1].path == "/attribute/X:Y/RID,Length"

    def test_two_paths_to_same_table_only_one_sample(self):
        """Multiple FK paths to one table → one sample query (first wins).

        Subsequent paths through the same table share row shape;
        re-sampling would be wasted work.
        """
        dp1 = _fake_datapath(
            entity_uri="https://srv/ermrest/catalog/3/entity/A:Y",
            attr_uri="https://srv/ermrest/catalog/3/attribute/A:Y/RID",
        )
        dp2 = _fake_datapath(
            entity_uri="https://srv/ermrest/catalog/3/entity/B:Y",
            attr_uri="https://srv/ermrest/catalog/3/attribute/B:Y/RID",
        )
        target = MagicMock()
        items = build_estimate_queries({"Y": [(dp1, target, False), (dp2, target, False)]})
        sample_items = [it for it in items if it.query_type == "sample"]
        assert len(sample_items) == 1
        # The first path's URI wins.
        assert sample_items[0].path == "/entity/A:Y?limit=100"
        # Both paths still get their own ``csv`` query.
        csv_items = [it for it in items if it.query_type == "csv"]
        assert len(csv_items) == 2


# ---------------------------------------------------------------------------
# run_estimate_queries — async orchestration with mock catalog
# ---------------------------------------------------------------------------


class _FakeAsyncResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeAsyncCatalog:
    """Mock async catalog that returns canned responses per path.

    Attributes:
        responses: dict mapping path → payload (or callable raising
            for that path).
        get_calls: list of paths the orchestrator requested.
        closed: True after ``close()`` is awaited.
    """

    def __init__(self, responses):
        self.responses = responses
        self.get_calls: list[str] = []
        self.closed = False

    async def get_async(self, path):
        self.get_calls.append(path)
        resp = self.responses.get(path)
        if callable(resp):
            resp()  # may raise
        return _FakeAsyncResponse(resp)

    async def close(self):
        self.closed = True


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if not _have_running_loop() else asyncio.run(coro)


def _have_running_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


class TestRunEstimateQueries:
    """``run_estimate_queries`` orchestrates queries and groups results."""

    def test_csv_results_union_rids(self):
        cat = _FakeAsyncCatalog(
            {
                "/p1": [{"RID": "A"}, {"RID": "B"}],
                "/p2": [{"RID": "B"}, {"RID": "C"}],
            }
        )
        items = [
            QueryItem("T", "/p1", "csv"),
            QueryItem("T", "/p2", "csv"),
        ]
        rids, lengths, samples = asyncio.run(run_estimate_queries(cat, items))
        # Union, not concat.
        assert rids == {"T": {"A", "B", "C"}}
        assert lengths == {}
        assert samples == {}
        assert cat.closed is True

    def test_fetch_results_collect_rid_length(self):
        cat = _FakeAsyncCatalog(
            {
                "/p1": [
                    {"RID": "A", "Length": 100},
                    {"RID": "B", "Length": 200},
                ]
            }
        )
        items = [QueryItem("T", "/p1", "fetch")]
        rids, lengths, samples = asyncio.run(run_estimate_queries(cat, items))
        assert lengths == {"T": {"A": 100, "B": 200}}

    def test_fetch_first_occurrence_wins(self):
        """Same RID appearing twice keeps the first Length value."""
        cat = _FakeAsyncCatalog(
            {
                "/p1": [{"RID": "A", "Length": 100}],
                "/p2": [{"RID": "A", "Length": 999}],
            }
        )
        items = [
            QueryItem("T", "/p1", "fetch"),
            QueryItem("T", "/p2", "fetch"),
        ]
        rids, lengths, samples = asyncio.run(run_estimate_queries(cat, items))
        assert lengths == {"T": {"A": 100}}

    def test_sample_only_keeps_first_per_table(self):
        cat = _FakeAsyncCatalog(
            {
                "/p1": [{"row": 1}],
                "/p2": [{"row": 2}, {"row": 3}],
            }
        )
        items = [
            QueryItem("T", "/p1", "sample"),
            QueryItem("T", "/p2", "sample"),
        ]
        rids, lengths, samples = asyncio.run(run_estimate_queries(cat, items))
        assert samples == {"T": [{"row": 1}]}

    def test_per_query_failure_is_swallowed_with_debug_log(self):
        """A failing query produces an empty-list result; other queries succeed."""

        def _boom():
            raise RuntimeError("query failed")

        cat = _FakeAsyncCatalog({"/good": [{"RID": "A"}], "/bad": _boom})
        items = [
            QueryItem("T", "/good", "csv"),
            QueryItem("T", "/bad", "csv"),
        ]
        rids, lengths, samples = asyncio.run(run_estimate_queries(cat, items))
        # The good query landed; the bad one returned an empty list,
        # contributing zero RIDs to the union.
        assert rids == {"T": {"A"}}

    def test_close_runs_even_on_partial_failure(self):
        """``catalog.close()`` is awaited even when individual queries failed."""

        def _boom():
            raise RuntimeError("boom")

        cat = _FakeAsyncCatalog({"/p1": _boom})
        items = [QueryItem("T", "/p1", "csv")]
        asyncio.run(run_estimate_queries(cat, items))
        assert cat.closed is True


# ---------------------------------------------------------------------------
# assemble_estimate — pure data assembly
# ---------------------------------------------------------------------------


def _stub_estimate_csv_bytes(rows, n):
    # Trivial: 50 bytes per row, scaled by row count.
    return 50 * n


def _stub_human_readable(n):
    return f"{n}B"


class TestAssembleEstimate:
    """``assemble_estimate`` produces the final estimate dict."""

    def test_empty_inputs(self):
        result = assemble_estimate(
            table_queries={},
            rids_by_table={},
            asset_lengths_by_table={},
            sample_rows_by_table={},
            estimate_csv_bytes=_stub_estimate_csv_bytes,
            human_readable_size=_stub_human_readable,
        )
        assert result["total_rows"] == 0
        assert result["total_asset_bytes"] == 0
        assert result["total_csv_bytes"] == 0
        assert result["total_estimated_bytes"] == 0
        assert result["tables"] == {}

    def test_non_asset_table(self):
        table_queries = {"T": [(MagicMock(), MagicMock(), False)]}
        rids = {"T": {"A", "B", "C"}}
        samples = {"T": [{"row": 1}]}
        result = assemble_estimate(
            table_queries=table_queries,
            rids_by_table=rids,
            asset_lengths_by_table={},
            sample_rows_by_table=samples,
            estimate_csv_bytes=_stub_estimate_csv_bytes,
            human_readable_size=_stub_human_readable,
        )
        assert result["tables"]["T"]["row_count"] == 3
        assert result["tables"]["T"]["is_asset"] is False
        assert result["tables"]["T"]["asset_bytes"] == 0
        # 50 bytes/row × 3 rows = 150
        assert result["tables"]["T"]["csv_bytes"] == 150
        assert result["total_rows"] == 3
        assert result["total_csv_bytes"] == 150

    def test_asset_table_sums_lengths(self):
        table_queries = {"A": [(MagicMock(), MagicMock(), True)]}
        rids = {"A": {"R1", "R2"}}
        lengths = {"A": {"R1": 1000, "R2": 2000}}
        result = assemble_estimate(
            table_queries=table_queries,
            rids_by_table=rids,
            asset_lengths_by_table=lengths,
            sample_rows_by_table={},
            estimate_csv_bytes=_stub_estimate_csv_bytes,
            human_readable_size=_stub_human_readable,
        )
        assert result["tables"]["A"]["is_asset"] is True
        assert result["tables"]["A"]["asset_bytes"] == 3000
        assert result["total_asset_bytes"] == 3000

    def test_asset_only_in_fetch_results_still_counted(self):
        """An asset table that has fetch results but no csv RIDs still appears.

        Defensive: the pre-extraction code had a safety branch for
        this, and the helper preserves it.
        """
        table_queries = {"A": [(MagicMock(), MagicMock(), True)]}
        # ``A`` has fetch lengths but the csv branch produced no RIDs.
        result = assemble_estimate(
            table_queries=table_queries,
            rids_by_table={},
            asset_lengths_by_table={"A": {"R1": 500}},
            sample_rows_by_table={},
            estimate_csv_bytes=_stub_estimate_csv_bytes,
            human_readable_size=_stub_human_readable,
        )
        assert "A" in result["tables"]
        assert result["tables"]["A"]["row_count"] == 1
        assert result["tables"]["A"]["asset_bytes"] == 500

    def test_human_readable_size_callable_invoked(self):
        """The ``human_readable_size`` callable formats every total."""
        result = assemble_estimate(
            table_queries={},
            rids_by_table={},
            asset_lengths_by_table={},
            sample_rows_by_table={},
            estimate_csv_bytes=_stub_estimate_csv_bytes,
            human_readable_size=lambda n: f"~{n}!",
        )
        assert result["total_asset_size"] == "~0!"
        assert result["total_csv_size"] == "~0!"
        assert result["total_estimated_size"] == "~0!"
