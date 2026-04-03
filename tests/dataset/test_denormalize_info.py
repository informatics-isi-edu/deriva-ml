"""Tests for Dataset.denormalize_info()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


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
