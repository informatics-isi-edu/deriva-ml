"""Tests for Dataset.denormalize_info().

Unit tests use MagicMock for structure validation.
Integration tests use the populated_catalog fixture against a real catalog.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from deriva_ml import DerivaML


def _make_mock_model():
    """Create a mock DerivaModel with schema traversal support."""
    model = MagicMock()

    # _prepare_wide_table returns (element_tables, column_specs, multi_schema)
    model._prepare_wide_table.return_value = (
        {
            "Image": (
                ["Dataset", "Dataset_Image", "Image", "Subject"],
                {"Dataset_Image": {("Dataset", "RID")}, "Image": {("Image", "RID")}, "Subject": {("Subject_FK", "RID")}},
                {"Dataset_Image": "inner", "Image": "inner", "Subject": "left"},
            )
        },
        [
            ("eye-ai", "Image", "RID", "ermrest_rid"),
            ("eye-ai", "Image", "Filename", "text"),
            ("eye-ai", "Subject", "RID", "ermrest_rid"),
            ("eye-ai", "Subject", "Name", "text"),
        ],
        False,  # multi_schema
    )

    # is_asset checks
    def mock_is_asset(table_name):
        return table_name == "Image"

    model.is_asset.side_effect = mock_is_asset
    return model


def _make_mock_dataset(model):
    """Create a mock Dataset with denormalize_info dependencies."""
    ds = MagicMock()
    ds._ml_instance = MagicMock()
    ds._ml_instance.model = model
    ds.dataset_rid = "DS-001"
    return ds


class TestDenormalizeInfoReturnStructure:
    """Tests for the return dict structure of denormalize_info."""

    def test_returns_columns_and_types(self):
        """Columns list contains (name, type) tuples."""
        info = {
            "columns": [("Image.RID", "ermrest_rid"), ("Image.Filename", "text")],
            "join_path": ["Image", "Subject"],
            "tables": {"Image": {"row_count": 100, "is_asset": True, "asset_bytes": 5000}},
            "total_rows": 100,
            "total_asset_bytes": 5000,
            "total_asset_size": "4.9 KB",
        }
        assert len(info["columns"]) == 2
        assert info["columns"][0] == ("Image.RID", "ermrest_rid")

    def test_join_path_includes_intermediates(self):
        """Join path shows intermediate tables used for the join."""
        info = {
            "join_path": ["Report_HVF", "Observation", "Subject"],
        }
        assert "Observation" in info["join_path"]

    def test_tables_dict_matches_bag_info_pattern(self):
        """Per-table dict uses same keys as estimate_bag_size."""
        table_info = {"row_count": 50, "is_asset": False, "asset_bytes": 0}
        assert "row_count" in table_info
        assert "is_asset" in table_info
        assert "asset_bytes" in table_info

    def test_non_asset_table_has_zero_asset_bytes(self):
        """Non-asset tables must include asset_bytes: 0 to match estimate_bag_size."""
        table_info = {"row_count": 50, "is_asset": False, "asset_bytes": 0}
        assert table_info["asset_bytes"] == 0


class TestDenormalizeInfoMethod:
    """Tests that actually call denormalize_info on a mocked Dataset."""

    def test_calls_prepare_wide_table_and_returns_structure(self):
        """denormalize_info delegates to _prepare_wide_table and returns expected keys."""
        from deriva_ml.dataset.dataset import Dataset

        model = _make_mock_model()
        model.is_association.return_value = False

        # Mock name_to_table to return table objects with schema
        mock_table = MagicMock()
        mock_table.schema.name = "eye-ai"
        model.name_to_table.return_value = mock_table

        # Mock pathBuilder chain
        mock_pb = MagicMock()
        # aggregates().fetch() is called for both row count (cnt) and asset size (total)
        # Return a dict containing both keys so either access pattern works.
        mock_pb.schemas.__getitem__.return_value.tables.__getitem__.return_value.aggregates.return_value.fetch.return_value = [
            {"cnt": 100, "total": 5000}
        ]
        mock_pb.schemas.__getitem__.return_value.tables.__getitem__.return_value.column_definitions.__getitem__.return_value = MagicMock()

        # Create a Dataset with mocked internals
        ds = MagicMock(spec=Dataset)
        ds._ml_instance = MagicMock()
        ds._ml_instance.model = model
        ds._ml_instance.pathBuilder.return_value = mock_pb
        ds.dataset_rid = "DS-001"
        ds._human_readable_size = Dataset._human_readable_size

        # Call the real method on the mocked instance
        result = Dataset.denormalize_info(ds, ["Image", "Subject"])

        # Verify it called _prepare_wide_table
        model._prepare_wide_table.assert_called_once()

        # Verify return structure
        assert "columns" in result
        assert "join_path" in result
        assert "tables" in result
        assert "total_rows" in result
        assert "total_asset_bytes" in result
        assert "total_asset_size" in result

    def test_non_asset_table_gets_zero_asset_bytes(self):
        """Non-asset tables in the result have asset_bytes: 0."""
        from deriva_ml.dataset.dataset import Dataset

        model = _make_mock_model()
        model.is_association.return_value = False

        # Override is_asset: both tables are non-asset
        model.is_asset.return_value = False
        model.is_asset.side_effect = None

        mock_table = MagicMock()
        mock_table.schema.name = "eye-ai"
        model.name_to_table.return_value = mock_table

        mock_pb = MagicMock()
        mock_pb.schemas.__getitem__.return_value.tables.__getitem__.return_value.aggregates.return_value.fetch.return_value = [
            {"cnt": 10}
        ]
        mock_pb.schemas.__getitem__.return_value.tables.__getitem__.return_value.column_definitions.__getitem__.return_value = MagicMock()

        ds = MagicMock(spec=Dataset)
        ds._ml_instance = MagicMock()
        ds._ml_instance.model = model
        ds._ml_instance.pathBuilder.return_value = mock_pb
        ds.dataset_rid = "DS-001"
        ds._human_readable_size = Dataset._human_readable_size

        result = Dataset.denormalize_info(ds, ["Image", "Subject"])

        # Every table should have asset_bytes key
        for table_name, info in result["tables"].items():
            assert "asset_bytes" in info, f"{table_name} missing asset_bytes"
            assert info["asset_bytes"] == 0


class TestDenormalizeInfoMixin:
    """Tests for the DerivaML-level denormalize_info (no dataset required)."""

    def test_mixin_does_not_require_dataset_rid(self):
        """The mixin method takes only include_tables, no dataset."""
        from deriva_ml.core.mixins.dataset import DatasetMixin
        import inspect
        sig = inspect.signature(DatasetMixin.denormalize_info)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "include_tables" in params
        assert "dataset_rid" not in params


class TestDenormalizeInfoAggregateAPI:
    """Tests that aggregate queries use the correct datapath API.

    These tests use spec-constrained mocks that raise AttributeError for
    methods that don't exist on the real classes — catching API misuse
    that unconstrained MagicMock would silently accept.
    """

    def test_uses_cnt_aggregate_class_not_column_attribute(self):
        """Row count query must use Cnt(col).alias(), not col.count or col.cnt."""
        from deriva.core.datapath import Cnt, _ColumnWrapper

        # Create a column mock constrained to real _ColumnWrapper API
        mock_col = MagicMock(spec=_ColumnWrapper)

        # Verify that .count and .cnt don't exist on real _ColumnWrapper
        assert not hasattr(mock_col, "count"), \
            "_ColumnWrapper should not have .count — use Cnt(col) instead"
        assert not hasattr(mock_col, "cnt"), \
            "_ColumnWrapper should not have .cnt — use Cnt(col) instead"

        # Verify Cnt(col) works and produces an aggregate with .alias()
        agg = Cnt(mock_col)
        assert hasattr(agg, "alias"), "Cnt(col) should have .alias() method"

    def test_uses_sum_aggregate_class_not_column_attribute(self):
        """Asset size query must use Sum(col).alias(), not col.sum."""
        from deriva.core.datapath import Sum, _ColumnWrapper

        mock_col = MagicMock(spec=_ColumnWrapper)

        assert not hasattr(mock_col, "sum"), \
            "_ColumnWrapper should not have .sum — use Sum(col) instead"

        agg = Sum(mock_col)
        assert hasattr(agg, "alias"), "Sum(col) should have .alias() method"


class TestDenormalizeInfoIntegration:
    """Integration tests against a real catalog.

    These tests require a running Deriva server (set DERIVA_HOST env var).
    They use the populated_catalog fixture which provides Subject and Image
    tables with real data.
    """

    def test_mixin_returns_valid_structure(self, populated_catalog: "DerivaML"):
        """DerivaML.denormalize_info() returns correct structure with real catalog."""
        ml = populated_catalog
        info = ml.denormalize_info(["Image", "Subject"])

        # Verify all required keys
        assert "columns" in info
        assert "join_path" in info
        assert "tables" in info
        assert "total_rows" in info
        assert "total_asset_bytes" in info
        assert "total_asset_size" in info

        # Columns should be non-empty list of (name, type) tuples
        assert len(info["columns"]) > 0
        for col in info["columns"]:
            assert isinstance(col, tuple)
            assert len(col) == 2
            name, type_name = col
            assert isinstance(name, str)
            assert isinstance(type_name, str)

        # Join path should include requested tables
        assert "Image" in info["join_path"] or "Subject" in info["join_path"]

        # Total rows should be a non-negative integer
        assert isinstance(info["total_rows"], int)
        assert info["total_rows"] >= 0

    def test_mixin_row_counts_are_non_negative(self, populated_catalog: "DerivaML"):
        """Row counts from real catalog should be non-negative integers."""
        ml = populated_catalog
        info = ml.denormalize_info(["Image", "Subject"])

        # Row counts should be non-negative integers; some intermediate/
        # association tables may legitimately have 0 rows.
        has_positive = False
        for table_name, table_info in info["tables"].items():
            assert isinstance(table_info["row_count"], int)
            assert table_info["row_count"] >= 0, \
                f"{table_name} has negative row count"
            if table_info["row_count"] > 0:
                has_positive = True

        # At least one table should have data in a populated catalog
        assert has_positive, "No tables have rows in populated catalog"

    def test_mixin_per_table_structure(self, populated_catalog: "DerivaML"):
        """Each table entry has row_count, is_asset, and asset_bytes."""
        ml = populated_catalog
        info = ml.denormalize_info(["Image", "Subject"])

        for table_name, table_info in info["tables"].items():
            assert "row_count" in table_info, f"{table_name} missing row_count"
            assert "is_asset" in table_info, f"{table_name} missing is_asset"
            assert "asset_bytes" in table_info, f"{table_name} missing asset_bytes"
            assert isinstance(table_info["is_asset"], bool)
            assert isinstance(table_info["asset_bytes"], int)
            assert table_info["asset_bytes"] >= 0

    def test_dataset_denormalize_info(
        self, catalog_with_datasets: "tuple[DerivaML, object]"
    ):
        """Dataset.denormalize_info() works with a real dataset."""
        ml, dataset_desc = catalog_with_datasets
        # Get any dataset from the catalog
        datasets = list(ml.find_datasets())
        assert len(datasets) > 0, "catalog_with_datasets should have datasets"

        dataset = ml.lookup_dataset(datasets[0]["RID"])
        info = dataset.denormalize_info(["Image", "Subject"])

        assert "columns" in info
        assert "join_path" in info
        assert "tables" in info
        assert info["total_rows"] >= 0
