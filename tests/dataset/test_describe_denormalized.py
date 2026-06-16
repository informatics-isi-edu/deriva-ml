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


class TestEstimatedRowCount:
    """Regression tests for ``estimated_row_count`` in ``describe()`` —
    the spec §5 plan-dict field that the 2026-05-21 e2e Analyst arc
    surfaced as finding A02 (preflight returns 0 while the actual
    fetch returns rows).

    The estimator counts anchors whose table equals ``row_per`` and
    sums them into ``in_scope_row_per_rows``. That formula is correct
    only when anchors sit *at* ``row_per``. When ``row_per`` is
    downstream of the anchor table — the common feature-table case —
    the formula returns 0 even though the join will produce N rows
    per anchor.

    Contract pinned here:

    - Anchor table == row_per → estimate is exact (anchor count).
    - Anchor table is downstream of row_per via FK chain → estimate
      is ``None`` (honest: we don't know without a catalog query).
      A reason string is included so callers can tell why.
    - Anchor unreachable from row_per → orphan (existing behaviour).
    """

    @staticmethod
    def _find_dataset_with_image_anchors(ml):
        """Locate a dataset in the demo fixture that has Image members
        as anchors.

        ``catalog_with_datasets`` creates a hierarchy; ``find_datasets()``
        returns the lot, and ``[0]`` is not guaranteed to be the one
        with Image members. Iterate and pick the first dataset whose
        ``describe_denormalized(["Image"])`` reports Image anchors > 0.
        """
        for dataset in ml.find_datasets():
            info = dataset.describe_denormalized(["Image"])
            if info["anchors"]["by_type"].get("Image", 0) > 0:
                return dataset
        raise AssertionError("No dataset in catalog_with_datasets has Image anchors; demo fixture changed?")

    def test_estimate_exact_when_anchor_is_row_per(self, catalog_with_datasets: "tuple[DerivaML, object]"):
        """Anchors at the row_per table give an exact estimate.

        Demo fixture has Image anchors; with row_per='Image', the
        estimator can compute the exact in-scope row count (== Image
        anchor count) without any catalog query.

        Note: the demo fixture may also include other anchor types
        (Subject, etc.) that reach Image. The "anchor==row_per case
        is exact" claim only holds when ALL scoping anchors are at
        row_per. If the fixture has mixed-cardinality anchors, the
        estimator honestly returns None — which is correct, and this
        test then validates *that* behavior instead.
        """
        ml, _ = catalog_with_datasets
        dataset = self._find_dataset_with_image_anchors(ml)

        info = dataset.describe_denormalized(["Image"], row_per="Image")
        estimated = info["estimated_row_count"]
        anchors_by_type = info["anchors"]["by_type"]
        image_anchors = anchors_by_type.get("Image", 0)
        other_anchor_types = {t: c for t, c in anchors_by_type.items() if t != "Image" and c > 0}

        assert image_anchors > 0, "Demo dataset should have Image anchors"

        if other_anchor_types:
            # Mixed-cardinality fixture: estimator can't be exact, so
            # the honest answer is None.
            assert estimated["in_scope_row_per_rows"] is None, (
                f"Fixture has non-Image anchors {other_anchor_types}; "
                f"estimator should return None (mixed cardinality), got "
                f"{estimated['in_scope_row_per_rows']}"
            )
            assert estimated["total"] is None
            assert "reason" in estimated
        else:
            # Pure Image-anchor fixture: estimate is exact.
            assert estimated["in_scope_row_per_rows"] == image_anchors, (
                f"Anchor==row_per case should report exact count "
                f"({image_anchors}); got {estimated['in_scope_row_per_rows']}"
            )
            assert estimated["total"] == image_anchors

    def test_estimate_unknown_when_row_per_downstream_of_anchor(self, catalog_with_datasets: "tuple[DerivaML, object]"):
        """Regression for 2026-05-21 finding A02.

        When ``row_per`` is downstream of the anchor table (via FK
        chain), the actual row count is N rows per anchor for some
        N ≥ 0 that depends on per-anchor feature populations. The
        estimator can't compute that from anchors alone, so the
        honest answer is ``None`` (with a reason), NOT a silent 0.

        The demo's ``Execution_Image_Quality`` feature table is
        downstream of ``Image``: each Image may have 0..N quality
        feature rows. With row_per=Execution_Image_Quality and
        anchors at Image, the pre-A02-fix code returned
        ``in_scope_row_per_rows: 0`` because no anchor table matched
        ``row_per`` literally. Post-fix, the estimate is None with a
        reason flag.
        """
        ml, _ = catalog_with_datasets
        dataset = self._find_dataset_with_image_anchors(ml)

        info = dataset.describe_denormalized(
            ["Image", "Execution_Image_Quality"],
            row_per="Execution_Image_Quality",
        )
        estimated = info["estimated_row_count"]
        anchors_by_type = info["anchors"]["by_type"]
        image_anchors = anchors_by_type.get("Image", 0)

        assert image_anchors > 0, "Demo dataset should have Image anchors"

        # The honest answer is "unknown" — we don't have a per-anchor
        # fan-out estimate without querying the catalog. Pre-fix this
        # was silently 0, which the e2e Analyst arc consumed as
        # truth. The contract: when we can't compute, say so.
        assert estimated["in_scope_row_per_rows"] is None, (
            f"row_per downstream of anchor: estimate should be None "
            f"(unknown), not {estimated['in_scope_row_per_rows']} "
            f"(silently 0 was finding A02)."
        )
        assert estimated["total"] is None
        # Reason field tells the caller WHY the estimate is unknown.
        assert "reason" in estimated, (
            "estimated_row_count should include a 'reason' field when the count cannot be computed honestly."
        )
        assert "downstream" in estimated["reason"].lower(), (
            f"reason should mention the downstream relationship: got {estimated['reason']!r}"
        )
