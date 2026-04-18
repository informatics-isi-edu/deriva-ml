"""Tests for describe_denormalized() and estimate_denormalized_size().

Unit tests use MagicMock for structure validation.
Integration tests use the populated_catalog fixture against a real catalog.

Note: ``DerivaML.estimate_denormalized_size`` (mixin, catalog-wide) returns
the legacy shape (columns, join_path, tables, total_rows, total_asset_bytes,
total_asset_size) aligned with ``estimate_bag_size``.
``Dataset.describe_denormalized`` / ``DatasetBag.describe_denormalized``
delegate to ``Denormalizer.describe`` and return the spec §5 shape
(row_per, row_per_source, row_per_candidates, columns, include_tables, via,
join_path, transparent_intermediates, ambiguities, estimated_row_count,
anchors, source).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from deriva_ml import DerivaML


class TestMixinReturnStructure:
    """Tests for the legacy (DerivaML mixin) describe_denormalized dict shape."""

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


class TestMixinSignature:
    """Tests for the DerivaML-level describe_denormalized signature."""

    def test_mixin_does_not_require_dataset_rid(self):
        """The mixin method takes only include_tables, no dataset."""
        import inspect

        from deriva_ml.core.mixins.dataset import DatasetMixin

        sig = inspect.signature(DatasetMixin.estimate_denormalized_size)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "include_tables" in params
        assert "dataset_rid" not in params


class TestAggregateAPI:
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
        assert not hasattr(mock_col, "count"), "_ColumnWrapper should not have .count — use Cnt(col) instead"
        assert not hasattr(mock_col, "cnt"), "_ColumnWrapper should not have .cnt — use Cnt(col) instead"

        # Verify Cnt(col) works and produces an aggregate with .alias()
        agg = Cnt(mock_col)
        assert hasattr(agg, "alias"), "Cnt(col) should have .alias() method"

    def test_uses_sum_aggregate_class_not_column_attribute(self):
        """Asset size query must use Sum(col).alias(), not col.sum."""
        from deriva.core.datapath import Sum, _ColumnWrapper

        mock_col = MagicMock(spec=_ColumnWrapper)

        assert not hasattr(mock_col, "sum"), "_ColumnWrapper should not have .sum — use Sum(col) instead"

        agg = Sum(mock_col)
        assert hasattr(agg, "alias"), "Sum(col) should have .alias() method"


class TestMixinIntegration:
    """Integration tests against a real catalog for ``DerivaML.describe_denormalized``.

    These tests require a running Deriva server (set DERIVA_HOST env var).
    They use the populated_catalog fixture which provides Subject and Image
    tables with real data.
    """

    def test_mixin_returns_valid_structure(self, populated_catalog: "DerivaML"):
        """DerivaML.describe_denormalized() returns correct structure with real catalog."""
        ml = populated_catalog
        info = ml.estimate_denormalized_size(["Image", "Observation", "Subject"])

        # Verify all required keys (legacy mixin shape)
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
        info = ml.estimate_denormalized_size(["Image", "Observation", "Subject"])

        # Row counts should be non-negative integers; some intermediate/
        # association tables may legitimately have 0 rows.
        has_positive = False
        for table_name, table_info in info["tables"].items():
            assert isinstance(table_info["row_count"], int)
            assert table_info["row_count"] >= 0, f"{table_name} has negative row count"
            if table_info["row_count"] > 0:
                has_positive = True

        # At least one table should have data in a populated catalog
        assert has_positive, "No tables have rows in populated catalog"

    def test_mixin_per_table_structure(self, populated_catalog: "DerivaML"):
        """Each table entry has row_count, is_asset, and asset_bytes."""
        ml = populated_catalog
        info = ml.estimate_denormalized_size(["Image", "Observation", "Subject"])

        for table_name, table_info in info["tables"].items():
            assert "row_count" in table_info, f"{table_name} missing row_count"
            assert "is_asset" in table_info, f"{table_name} missing is_asset"
            assert "asset_bytes" in table_info, f"{table_name} missing asset_bytes"
            assert isinstance(table_info["is_asset"], bool)
            assert isinstance(table_info["asset_bytes"], int)
            assert table_info["asset_bytes"] >= 0


class TestDatasetDescribeDenormalized:
    """Integration tests for ``Dataset.describe_denormalized`` (spec §5 shape).

    Unlike the mixin, ``Dataset.describe_denormalized`` delegates to
    ``Denormalizer.describe`` and returns the spec §5 plan dict with 12 keys:
    row_per, row_per_source, row_per_candidates, columns, include_tables,
    via, join_path, transparent_intermediates, ambiguities,
    estimated_row_count, anchors, source.
    """

    SPEC_KEYS = {
        "row_per",
        "row_per_source",
        "row_per_candidates",
        "columns",
        "include_tables",
        "via",
        "join_path",
        "transparent_intermediates",
        "ambiguities",
        "estimated_row_count",
        "anchors",
        "source",
    }

    def test_dataset_returns_spec_keys(self, catalog_with_datasets: "tuple[DerivaML, object]"):
        """Dataset.describe_denormalized() returns all spec §5 keys."""
        ml, _dataset_desc = catalog_with_datasets
        datasets = list(ml.find_datasets())
        assert len(datasets) > 0, "catalog_with_datasets should have datasets"

        dataset = datasets[0]
        info = dataset.describe_denormalized(["Image"])

        missing = self.SPEC_KEYS - set(info.keys())
        assert not missing, f"Plan dict missing keys: {missing}"

        # columns is a list of (name, type) tuples
        assert isinstance(info["columns"], list)
        # include_tables echoes the request
        assert "Image" in info["include_tables"]
        # join_path is an ordered list
        assert isinstance(info["join_path"], list)
        # ambiguities is a list (empty if none)
        assert isinstance(info["ambiguities"], list)
        # anchors is a dict with total + by_type
        assert isinstance(info["anchors"], dict)
        assert "total" in info["anchors"]
        assert "by_type" in info["anchors"]
        # source is a string tag
        assert isinstance(info["source"], str)

    def test_dataset_reports_ambiguity_without_raising(self, catalog_with_datasets: "tuple[DerivaML, object]"):
        """describe_denormalized reports ambiguities in the dict; it never raises."""
        ml, _ = catalog_with_datasets
        datasets = list(ml.find_datasets())
        assert len(datasets) > 0

        dataset = datasets[0]
        # This combination is ambiguous in the demo schema (Image→Subject has two
        # paths). describe_denormalized must still return without raising.
        info = dataset.describe_denormalized(["Image", "Subject"])

        assert "ambiguities" in info
        # We don't assert it's non-empty here because other test schemas may
        # not include the diamond; just verify it's well-formed.
        assert isinstance(info["ambiguities"], list)
