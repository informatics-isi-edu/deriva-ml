"""Tests for estimate_bag_size, focusing on the accumulation logic.

The key bug was that when the same table appears via multiple FK paths in the
download spec (e.g., OCT_DICOM reachable both directly and via CGM_Blood_Glucose),
the second query's results would overwrite the first instead of accumulating.
If the second query failed silently, the per-table entry would show 0 while
total_asset_bytes had already counted the successful first query.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_spec(query_processors: list[dict]) -> dict:
    """Build a minimal download spec with the given query processors."""
    return {
        "catalog": {
            "query_processors": [
                {"processor": "env", "processor_params": {"output_path": "Dataset", "query_path": "/"}},
            ]
            + query_processors,
        }
    }


def _csv_qp(output_path: str, query_path: str) -> dict:
    return {
        "processor": "csv",
        "processor_params": {
            "query_path": query_path,
            "output_path": output_path,
            "paged_query": True,
        },
    }


def _fetch_qp(table_name: str, query_path: str) -> dict:
    return {
        "processor": "fetch",
        "processor_params": {
            "query_path": query_path,
            "output_path": f"asset/{{asset_rid}}/{table_name}",
        },
    }


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
        response.json.return_value = [{"cnt": 0, "total": 0}]
        return response

    mock_catalog.get_async = mock_get_async
    mock_catalog.close = AsyncMock()
    return mock_catalog


def _run_estimate(spec: dict, catalog_responses: dict) -> dict:
    """Run estimate_bag_size with mocked internals.

    Args:
        spec: The download spec to use.
        catalog_responses: Mapping from aggregate query path substring to response JSON.
            If a value is an Exception, that exception is raised.
    """
    from deriva_ml.dataset.dataset import Dataset

    # Mock the Dataset instance — set return values directly on the instance
    # (patch.object on the class does NOT affect MagicMock instances).
    dataset = MagicMock(spec=Dataset)
    dataset.dataset_rid = "TEST-RID"
    dataset._ml_instance = MagicMock()
    dataset._ml_instance.catalog.deriva_server.server = "test.example.org"
    dataset._ml_instance.catalog.deriva_server.scheme = "https"
    dataset._ml_instance.catalog.catalog_id = "99"
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
        mock_graph.generate_dataset_download_spec.return_value = spec
        mock_graph_cls.return_value = mock_graph

        return Dataset.estimate_bag_size(dataset, "1.0.0")


def _run_estimate_counting(spec: dict) -> int:
    """Run estimate_bag_size and return the total number of get_async() calls."""
    from deriva_ml.dataset.dataset import Dataset

    dataset = MagicMock(spec=Dataset)
    dataset.dataset_rid = "TEST-RID"
    dataset._ml_instance = MagicMock()
    dataset._ml_instance.catalog.deriva_server.server = "test.example.org"
    dataset._ml_instance.catalog.deriva_server.scheme = "https"
    dataset._ml_instance.catalog.catalog_id = "99"
    dataset._version_snapshot_catalog.return_value = MagicMock()
    dataset._version_snapshot_catalog_id.return_value = "99@2024-01-01T00:00:00"
    dataset._human_readable_size.side_effect = Dataset._human_readable_size

    call_count = 0

    mock_catalog = MagicMock()

    async def counting_get_async(path, **kwargs):
        nonlocal call_count
        call_count += 1
        response = MagicMock()
        response.json.return_value = [{"cnt": 10, "total": 1000}]
        return response

    mock_catalog.get_async = counting_get_async
    mock_catalog.close = AsyncMock()

    with (
        patch("deriva_ml.dataset.dataset.CatalogGraph") as mock_graph_cls,
        patch("deriva.core.get_credential", return_value={}),
        patch("deriva_ml.dataset.dataset.AsyncErmrestSnapshot", return_value=mock_catalog),
    ):
        mock_graph = MagicMock()
        mock_graph.generate_dataset_download_spec.return_value = spec
        mock_graph_cls.return_value = mock_graph

        Dataset.estimate_bag_size(dataset, "1.0.0")

    return call_count


class TestEstimateBagSizeAccumulation:
    """Test that duplicate table entries from multiple FK paths are handled correctly."""

    def test_single_path_csv_and_fetch(self):
        """Basic case: one csv + one fetch for the same table."""
        spec = _make_spec(
            [
                _csv_qp("Dataset/Dataset_Image/Image", "/entity/Image/RID={RID}"),
                _fetch_qp("Image", "/attribute/Image/RID={RID}/!(URL::null::)/url:=URL,length:=Length"),
            ]
        )
        result = _run_estimate(
            spec,
            {
                "/aggregate/Image/RID=TEST-RID/cnt:=cnt(RID)": [{"cnt": 100}],
                "/aggregate/Image/RID=TEST-RID/!(URL::null::)/total:=sum(Length),cnt:=cnt(RID)": [
                    {"total": 5000, "cnt": 100}
                ],
            },
        )

        assert result["tables"]["Image"]["row_count"] == 100
        assert result["tables"]["Image"]["asset_bytes"] == 5000
        assert result["tables"]["Image"]["is_asset"] is True
        assert result["total_rows"] == 100
        assert result["total_asset_bytes"] == 5000

    def test_duplicate_table_deduplication(self):
        """Same table via two paths — dedup means only first path is queried."""
        spec = _make_spec(
            [
                _csv_qp("Dataset/Dataset_OCT/OCT_DICOM", "/entity/path1/OCT_DICOM/RID={RID}"),
                _fetch_qp("OCT_DICOM", "/attribute/path1/OCT_DICOM/RID={RID}/!(URL::null::)/url:=URL,length:=Length"),
                _csv_qp("Dataset/Dataset_CGM/CGM/Observation/OCT_DICOM", "/entity/path2/OCT_DICOM/RID={RID}"),
                _fetch_qp(
                    "OCT_DICOM", "/attribute/path2/OCT_DICOM/RID={RID}/!(URL::null::)/url:=URL,length:=Length"
                ),
            ]
        )
        result = _run_estimate(
            spec,
            {
                "path1/OCT_DICOM/RID=TEST-RID/cnt:=cnt(RID)": [{"cnt": 200}],
                "path1/OCT_DICOM/RID=TEST-RID/!(URL::null::)/total:=sum(Length)": [{"total": 688_000_000, "cnt": 200}],
            },
        )

        assert result["tables"]["OCT_DICOM"]["row_count"] == 200
        assert result["tables"]["OCT_DICOM"]["asset_bytes"] == 688_000_000
        assert result["tables"]["OCT_DICOM"]["is_asset"] is True
        # total must match per-table (no double counting)
        assert result["total_asset_bytes"] == result["tables"]["OCT_DICOM"]["asset_bytes"]

    def test_total_matches_per_table_sum(self):
        """Verify total_asset_bytes equals sum of all per-table asset_bytes."""
        spec = _make_spec(
            [
                _csv_qp("Dataset/Image", "/entity/Image/RID={RID}"),
                _fetch_qp("Image", "/attribute/Image/RID={RID}/!(URL::null::)/url:=URL,length:=Length"),
                _csv_qp("Dataset/Report", "/entity/Report/RID={RID}"),
                _fetch_qp("Report", "/attribute/Report/RID={RID}/!(URL::null::)/url:=URL,length:=Length"),
            ]
        )
        result = _run_estimate(
            spec,
            {
                "Image/RID=TEST-RID/cnt:=cnt(RID)": [{"cnt": 10}],
                "Image/RID=TEST-RID/!(URL::null::)/total:=sum(Length)": [{"total": 1000, "cnt": 10}],
                "Report/RID=TEST-RID/cnt:=cnt(RID)": [{"cnt": 5}],
                "Report/RID=TEST-RID/!(URL::null::)/total:=sum(Length)": [{"total": 500, "cnt": 5}],
            },
        )

        per_table_total = sum(t["asset_bytes"] for t in result["tables"].values())
        assert result["total_asset_bytes"] == per_table_total, (
            f"total_asset_bytes ({result['total_asset_bytes']}) != "
            f"sum of per-table asset_bytes ({per_table_total})"
        )

    def test_csv_only_no_fetch(self):
        """Table with no assets (csv only, no fetch processor)."""
        spec = _make_spec(
            [
                _csv_qp("Dataset/Subject", "/entity/Subject/RID={RID}"),
            ]
        )
        result = _run_estimate(
            spec,
            {
                "Subject/RID=TEST-RID/cnt:=cnt(RID)": [{"cnt": 42}],
            },
        )

        assert result["tables"]["Subject"]["row_count"] == 42
        assert result["tables"]["Subject"]["is_asset"] is False
        assert result["tables"]["Subject"]["asset_bytes"] == 0
        assert result["total_rows"] == 42
        assert result["total_asset_bytes"] == 0

    def test_failed_query_returns_zero(self):
        """When a query fails, it should contribute 0 (not break)."""
        spec = _make_spec(
            [
                _csv_qp("Dataset/BadTable", "/entity/BadTable/RID={RID}"),
                _fetch_qp("BadTable", "/attribute/BadTable/RID={RID}/!(URL::null::)/url:=URL,length:=Length"),
            ]
        )
        result = _run_estimate(
            spec,
            {
                "BadTable/RID=TEST-RID/cnt:=cnt(RID)": Exception("timeout"),
                "BadTable/RID=TEST-RID/!(URL::null::)/total:=sum(Length)": Exception("timeout"),
            },
        )

        assert result["tables"]["BadTable"]["row_count"] == 0
        assert result["tables"]["BadTable"]["asset_bytes"] == 0
        assert result["total_asset_bytes"] == 0

    def test_deduplication_reduces_queries(self):
        """Verify that duplicate table names only produce one query each."""
        spec = _make_spec(
            [
                _csv_qp("Dataset/A/OCT", "/entity/pathA/OCT/RID={RID}"),
                _fetch_qp("OCT", "/attribute/pathA/OCT/RID={RID}/!(URL::null::)/url:=URL,length:=Length"),
                _csv_qp("Dataset/B/OCT", "/entity/pathB/OCT/RID={RID}"),
                _fetch_qp("OCT", "/attribute/pathB/OCT/RID={RID}/!(URL::null::)/url:=URL,length:=Length"),
                _csv_qp("Dataset/C/OCT", "/entity/pathC/OCT/RID={RID}"),
                _fetch_qp("OCT", "/attribute/pathC/OCT/RID={RID}/!(URL::null::)/url:=URL,length:=Length"),
            ]
        )

        call_count = _run_estimate_counting(spec)

        # With dedup: 1 csv + 1 fetch = 2 queries (not 3+3=6)
        assert call_count == 2, f"Expected 2 queries (deduplicated), got {call_count}"
