"""Tests for estimate_bag_size with exact RID-union semantics.

When the same table is reachable via multiple FK paths, estimate_bag_size
fetches RID lists from each path and computes the exact set union to get
the true row count.  For asset tables it fetches (RID, Length) pairs and
deduplicates by RID to get the exact total bytes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch


class _MockColumn:
    """Minimal mock column that satisfies datapath aggregate/attribute functions.

    Provides ``_instancename`` so that ``Cnt(_MockColumn('RID'))`` produces
    the correct ERMrest projection string ``cnt(RID)``.
    """

    def __init__(self, name: str):
        self.name = name

    @property
    def _instancename(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    @property
    def _projection_name(self) -> str:
        return self.name


def _make_mock_target_table() -> MagicMock:
    """Create a mock pathBuilder table with column attributes like ``.RID``.

    This simulates the target table reference returned by
    ``_aggregate_queries`` alongside the linked datapath.  The datapath
    itself does *not* expose column attributes after ``.link()`` calls;
    the separate table reference is needed to access ``.RID`` etc.
    """
    tbl = MagicMock()
    tbl.RID = _MockColumn("RID")
    tbl.Length = _MockColumn("Length")
    tbl.URL = _MockColumn("URL")
    return tbl


def _make_mock_datapath(uri: str) -> tuple[MagicMock, MagicMock]:
    """Create a mock (datapath, target_table) pair.

    The datapath mock does *not* have ``.RID`` — just like real linked
    datapaths.  Column attributes live on the separate target_table mock.
    ``.attributes()`` accepts column objects from the target table.

    Returns:
        ``(dp, target_table)`` tuple matching the shape returned by
        ``CatalogGraph._aggregate_queries``.
    """
    dp = MagicMock(spec=[])  # spec=[] prevents auto-creating attributes
    dp.uri = uri

    target_table = _make_mock_target_table()

    def _mock_attributes(*cols):
        """Build a mock ResultSet with the correct attribute URI."""
        col_names = ",".join(c._projection_name if hasattr(c, '_projection_name') else str(c) for c in cols)
        attr_uri = uri.replace("/entity/", "/attribute/") + "/" + col_names
        rs = MagicMock()
        rs.uri = attr_uri
        return rs

    dp.attributes = _mock_attributes
    return dp, target_table


def _make_rids(*rids: str) -> list[dict]:
    """Helper to build a RID-only response list."""
    return [{"RID": r} for r in rids]


def _make_asset_rows(*entries: tuple[str, int]) -> list[dict]:
    """Helper to build a (RID, Length) response list for asset queries."""
    return [{"RID": rid, "Length": length} for rid, length in entries]


def _make_mock_async_catalog(catalog_responses: dict) -> MagicMock:
    """Create a mock async catalog that returns predefined responses.

    Args:
        catalog_responses: Mapping from query path substring to response JSON.
            If a value is an Exception, that exception is raised.
    """
    mock_catalog = MagicMock()

    async def mock_get_async(path, **kwargs):
        for pattern, result in catalog_responses.items():
            if pattern in path:
                if isinstance(result, Exception):
                    raise result
                response = MagicMock()
                response.json.return_value = result
                return response
        # Default: return empty result
        response = MagicMock()
        response.json.return_value = []
        return response

    mock_catalog.get_async = mock_get_async
    mock_catalog.close = AsyncMock()
    return mock_catalog


def _run_estimate(
    aggregate_queries: dict[str, list[tuple[MagicMock, MagicMock, bool]]],
    catalog_responses: dict,
) -> dict:
    """Run estimate_bag_size with mocked internals.

    Args:
        aggregate_queries: Return value for CatalogGraph._aggregate_queries.
            Maps table name to list of (mock_datapath, target_table, is_asset) tuples.
        catalog_responses: Mapping from query path substring to response JSON.
            For csv queries, responses should be RID arrays: [{"RID": "X"}, ...].
            For fetch queries, responses should be [{RID, Length}, ...].
    """
    from deriva_ml.dataset.dataset import Dataset

    dataset = MagicMock(spec=Dataset)
    dataset.dataset_rid = "TEST-RID"
    dataset._ml_instance = MagicMock()
    dataset._ml_instance.catalog.deriva_server.server = "test.example.org"
    dataset._ml_instance.catalog.deriva_server.scheme = "https"
    dataset._ml_instance.catalog.catalog_id = "99"
    dataset._logger = MagicMock()
    dataset._version_snapshot_catalog.return_value = MagicMock()
    dataset._version_snapshot_catalog_id.return_value = "99@2024-01-01T00:00:00"
    dataset._human_readable_size.side_effect = Dataset._human_readable_size

    mock_catalog = _make_mock_async_catalog(catalog_responses)

    with (
        patch("deriva_ml.dataset.dataset.CatalogGraph") as mock_graph_cls,
        patch("deriva.core.get_credential", return_value={}),
        patch("deriva_ml.dataset.dataset.AsyncErmrestSnapshot", return_value=mock_catalog),
    ):
        mock_graph = MagicMock()
        mock_graph._aggregate_queries.return_value = aggregate_queries
        mock_graph_cls.return_value = mock_graph

        return Dataset.estimate_bag_size(dataset, "1.0.0")


def _run_estimate_counting(
    aggregate_queries: dict[str, list[tuple[MagicMock, MagicMock, bool]]],
) -> int:
    """Run estimate_bag_size and return the total number of get_async() calls."""
    from deriva_ml.dataset.dataset import Dataset

    dataset = MagicMock(spec=Dataset)
    dataset.dataset_rid = "TEST-RID"
    dataset._ml_instance = MagicMock()
    dataset._ml_instance.catalog.deriva_server.server = "test.example.org"
    dataset._ml_instance.catalog.deriva_server.scheme = "https"
    dataset._ml_instance.catalog.catalog_id = "99"
    dataset._logger = MagicMock()
    dataset._version_snapshot_catalog.return_value = MagicMock()
    dataset._version_snapshot_catalog_id.return_value = "99@2024-01-01T00:00:00"
    dataset._human_readable_size.side_effect = Dataset._human_readable_size

    call_count = 0
    mock_catalog = MagicMock()

    async def counting_get_async(path, **kwargs):
        nonlocal call_count
        call_count += 1
        response = MagicMock()
        # Return a small RID list so the code can process it
        response.json.return_value = [{"RID": f"R{call_count}", "Length": 1000}]
        return response

    mock_catalog.get_async = counting_get_async
    mock_catalog.close = AsyncMock()

    with (
        patch("deriva_ml.dataset.dataset.CatalogGraph") as mock_graph_cls,
        patch("deriva.core.get_credential", return_value={}),
        patch("deriva_ml.dataset.dataset.AsyncErmrestSnapshot", return_value=mock_catalog),
    ):
        mock_graph = MagicMock()
        mock_graph._aggregate_queries.return_value = aggregate_queries
        mock_graph_cls.return_value = mock_graph

        Dataset.estimate_bag_size(dataset, "1.0.0")

    return call_count


class TestEstimateBagSizeUnionSemantics:
    """Test that multiple FK paths to the same table use exact RID-union counts."""

    def test_single_path_csv_and_fetch(self):
        """Basic case: one path to an asset table."""
        aggregate_queries = {
            "Image": [
                (*_make_mock_datapath("https://test.example.org/ermrest/catalog/99/entity/S:Dataset/RID=TEST-RID/S:Dataset_Image/S:Image"), True),
            ],
        }
        # 100 RIDs for the csv query, 100 (RID, Length) pairs for the fetch query
        rids = _make_rids(*[f"IMG-{i}" for i in range(100)])
        assets = _make_asset_rows(*[(f"IMG-{i}", 50) for i in range(100)])

        result = _run_estimate(
            aggregate_queries,
            {
                "S:Dataset_Image/S:Image/RID": rids,
                "S:Dataset_Image/S:Image/!(URL::null::)/RID,Length": assets,
            },
        )

        assert result["tables"]["Image"]["row_count"] == 100
        assert result["tables"]["Image"]["asset_bytes"] == 5000  # 100 * 50
        assert result["tables"]["Image"]["is_asset"] is True
        assert result["total_rows"] == 100
        assert result["total_asset_bytes"] == 5000

    def test_duplicate_table_exact_union(self):
        """Same table via two paths with overlapping RIDs — exact union count."""
        aggregate_queries = {
            "OCT_DICOM": [
                (*_make_mock_datapath("https://test.example.org/ermrest/catalog/99/entity/S:Dataset/RID=TEST-RID/S:Dataset_OCT/S:OCT_DICOM"), True),
                (*_make_mock_datapath("https://test.example.org/ermrest/catalog/99/entity/S:Dataset/RID=TEST-RID/S:Dataset_CGM/S:CGM/S:OCT_DICOM"), True),
            ],
        }
        # Path 1: 200 rows (RID-001..RID-200), each 3.44MB
        path1_rids = _make_rids(*[f"RID-{i:03d}" for i in range(1, 201)])
        path1_assets = _make_asset_rows(*[(f"RID-{i:03d}", 3_440_000) for i in range(1, 201)])
        # Path 2: 150 rows, 50 overlap with path 1 (RID-151..RID-200 are shared),
        # 100 new (RID-201..RID-300)
        path2_rids = _make_rids(*[f"RID-{i:03d}" for i in range(151, 301)])
        path2_assets = _make_asset_rows(*[(f"RID-{i:03d}", 3_440_000) for i in range(151, 301)])

        result = _run_estimate(
            aggregate_queries,
            {
                # Path 1 queries
                "S:Dataset_OCT/S:OCT_DICOM/RID": path1_rids,
                "S:Dataset_OCT/S:OCT_DICOM/!(URL::null::)/RID,Length": path1_assets,
                # Path 2 queries
                "S:Dataset_CGM/S:CGM/S:OCT_DICOM/RID": path2_rids,
                "S:Dataset_CGM/S:CGM/S:OCT_DICOM/!(URL::null::)/RID,Length": path2_assets,
            },
        )

        # Exact union: 200 + 150 - 50 overlap = 300 unique RIDs
        assert result["tables"]["OCT_DICOM"]["row_count"] == 300
        # 300 unique assets * 3.44MB each = 1,032,000,000 bytes
        assert result["tables"]["OCT_DICOM"]["asset_bytes"] == 300 * 3_440_000
        assert result["tables"]["OCT_DICOM"]["is_asset"] is True
        assert result["total_asset_bytes"] == 300 * 3_440_000

    def test_first_path_zero_second_path_has_data(self):
        """First path returns 0, second has real data — union picks the real data."""
        aggregate_queries = {
            "Observation": [
                (*_make_mock_datapath("https://test.example.org/ermrest/catalog/99/entity/S:Dataset/RID=TEST-RID/S:Path_A/S:Observation"), False),
                (*_make_mock_datapath("https://test.example.org/ermrest/catalog/99/entity/S:Dataset/RID=TEST-RID/S:Path_B/S:Observation"), False),
            ],
        }
        result = _run_estimate(
            aggregate_queries,
            {
                # Path A returns 0 rows
                "S:Path_A/S:Observation/RID": [],
                # Path B returns 75 rows
                "S:Path_B/S:Observation/RID": _make_rids(*[f"OBS-{i}" for i in range(75)]),
            },
        )

        assert result["tables"]["Observation"]["row_count"] == 75
        assert result["tables"]["Observation"]["is_asset"] is False
        assert result["total_rows"] == 75

    def test_all_paths_queried(self):
        """All paths for a table are queried (no first-wins dedup)."""
        aggregate_queries = {
            "OCT": [
                (*_make_mock_datapath("https://test.example.org/ermrest/catalog/99/entity/S:Dataset/RID=TEST-RID/S:PathA/S:OCT"), True),
                (*_make_mock_datapath("https://test.example.org/ermrest/catalog/99/entity/S:Dataset/RID=TEST-RID/S:PathB/S:OCT"), True),
                (*_make_mock_datapath("https://test.example.org/ermrest/catalog/99/entity/S:Dataset/RID=TEST-RID/S:PathC/S:OCT"), True),
            ],
        }

        # 3 paths x (1 csv + 1 fetch) = 6 queries total
        call_count = _run_estimate_counting(aggregate_queries)
        assert call_count == 6, f"Expected 6 queries (3 paths x 2 query types), got {call_count}"

    def test_csv_only_no_fetch(self):
        """Table with no assets (is_asset=False) only produces csv queries."""
        aggregate_queries = {
            "Subject": [
                (*_make_mock_datapath("https://test.example.org/ermrest/catalog/99/entity/S:Dataset/RID=TEST-RID/S:Dataset_Subject/S:Subject"), False),
            ],
        }
        result = _run_estimate(
            aggregate_queries,
            {
                "S:Dataset_Subject/S:Subject/RID": _make_rids(*[f"SUBJ-{i}" for i in range(42)]),
            },
        )

        assert result["tables"]["Subject"]["row_count"] == 42
        assert result["tables"]["Subject"]["is_asset"] is False
        assert result["tables"]["Subject"]["asset_bytes"] == 0
        assert result["total_rows"] == 42
        assert result["total_asset_bytes"] == 0

    def test_failed_query_returns_zero(self):
        """When a query fails, it should contribute 0 (not break)."""
        aggregate_queries = {
            "BadTable": [
                (*_make_mock_datapath("https://test.example.org/ermrest/catalog/99/entity/S:Dataset/RID=TEST-RID/S:BadTable"), True),
            ],
        }
        result = _run_estimate(
            aggregate_queries,
            {
                "S:BadTable/RID": Exception("timeout"),
                "S:BadTable/!(URL::null::)/RID,Length": Exception("timeout"),
            },
        )

        assert result["tables"]["BadTable"]["row_count"] == 0
        assert result["tables"]["BadTable"]["asset_bytes"] == 0
        assert result["total_asset_bytes"] == 0

    def test_total_matches_per_table_sum(self):
        """Verify total_asset_bytes equals sum of all per-table asset_bytes."""
        aggregate_queries = {
            "Image": [
                (*_make_mock_datapath("https://test.example.org/ermrest/catalog/99/entity/S:Dataset/RID=TEST-RID/S:Dataset_Image/S:Image"), True),
            ],
            "Report": [
                (*_make_mock_datapath("https://test.example.org/ermrest/catalog/99/entity/S:Dataset/RID=TEST-RID/S:Dataset_Report/S:Report"), True),
            ],
        }
        result = _run_estimate(
            aggregate_queries,
            {
                "S:Dataset_Image/S:Image/RID": _make_rids(*[f"IMG-{i}" for i in range(10)]),
                "S:Dataset_Image/S:Image/!(URL::null::)/RID,Length": _make_asset_rows(*[(f"IMG-{i}", 100) for i in range(10)]),
                "S:Dataset_Report/S:Report/RID": _make_rids(*[f"RPT-{i}" for i in range(5)]),
                "S:Dataset_Report/S:Report/!(URL::null::)/RID,Length": _make_asset_rows(*[(f"RPT-{i}", 100) for i in range(5)]),
            },
        )

        per_table_total = sum(t["asset_bytes"] for t in result["tables"].values())
        assert result["total_asset_bytes"] == per_table_total, (
            f"total_asset_bytes ({result['total_asset_bytes']}) != "
            f"sum of per-table asset_bytes ({per_table_total})"
        )

    def test_disjoint_paths_exact_count(self):
        """Two paths with completely disjoint RIDs — count is the full union (not max)."""
        aggregate_queries = {
            "Image": [
                (*_make_mock_datapath("https://test.example.org/ermrest/catalog/99/entity/S:Dataset/RID=TEST-RID/S:PathA/S:Image"), False),
                (*_make_mock_datapath("https://test.example.org/ermrest/catalog/99/entity/S:Dataset/RID=TEST-RID/S:PathB/S:Image"), False),
            ],
        }
        result = _run_estimate(
            aggregate_queries,
            {
                # Path A: 100 rows
                "S:PathA/S:Image/RID": _make_rids(*[f"A-{i}" for i in range(100)]),
                # Path B: 200 completely different rows
                "S:PathB/S:Image/RID": _make_rids(*[f"B-{i}" for i in range(200)]),
            },
        )

        # Old max approach would give 200; exact union gives 300
        assert result["tables"]["Image"]["row_count"] == 300
        assert result["total_rows"] == 300

    def test_datapath_does_not_expose_rid_attribute(self):
        """Regression: linked datapaths must not be used for column access.

        After ``.link()`` calls, a DataPath object does not expose column
        attributes like ``.RID``.  The code must use the separate
        ``target_table`` reference returned by ``_aggregate_queries``
        instead of ``dp.RID``.  This test verifies that the mock datapath
        (which mirrors real linked datapaths) does *not* have ``.RID``,
        and that the target_table *does*.
        """
        dp, target_table = _make_mock_datapath(
            "https://test.example.org/ermrest/catalog/99/entity/S:Dataset/RID=TEST-RID/S:Image"
        )
        # Linked datapaths should NOT have .RID
        assert not hasattr(dp, "RID"), (
            "Mock datapath should not expose .RID — "
            "real linked datapaths don't have column attributes"
        )
        # The target_table reference SHOULD have .RID
        assert hasattr(target_table, "RID"), (
            "target_table must expose .RID for use in .attributes() calls"
        )
