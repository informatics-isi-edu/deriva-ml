"""Unit tests for ``deriva_ml.dataset._estimate.assemble_estimate``.

``assemble_estimate`` is the final-assembly helper for
``Dataset.estimate_bag_size``: given the client-side reachability
engine's result maps (``rids_by_table`` / ``asset_lengths_by_table`` /
``sample_rows_by_table``) plus the set of asset tables, it produces the
user-facing estimate dict. This file pins that contract with pure-Python
tests against synthetic inputs — no live catalog required.
"""

from __future__ import annotations

from deriva_ml.dataset._estimate import assemble_estimate


def _stub_estimate_csv_bytes(rows, n):
    # Trivial: 50 bytes per row, scaled by row count.
    return 50 * n


def _stub_human_readable(n):
    return f"{n}B"


class TestAssembleEstimate:
    """``assemble_estimate`` produces the final estimate dict."""

    def test_empty_inputs(self):
        result = assemble_estimate(
            asset_tables=set(),
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
        rids = {"T": {"A", "B", "C"}}
        samples = {"T": [{"row": 1}]}
        result = assemble_estimate(
            asset_tables=set(),
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
        rids = {"A": {"R1", "R2"}}
        lengths = {"A": {"R1": 1000, "R2": 2000}}
        result = assemble_estimate(
            asset_tables={"A"},
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
        # ``A`` has fetch lengths but the csv branch produced no RIDs.
        result = assemble_estimate(
            asset_tables={"A"},
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
            asset_tables=set(),
            rids_by_table={},
            asset_lengths_by_table={},
            sample_rows_by_table={},
            estimate_csv_bytes=_stub_estimate_csv_bytes,
            human_readable_size=lambda n: f"~{n}!",
        )
        assert result["total_asset_size"] == "~0!"
        assert result["total_csv_size"] == "~0!"
        assert result["total_estimated_size"] == "~0!"

    def test_normal_estimate_is_never_incomplete(self):
        """The reachability engine has no per-query failures: a normal
        non-empty estimate always reports ``incomplete`` False and an
        empty ``incomplete_tables`` list."""
        result = assemble_estimate(
            asset_tables=set(),
            rids_by_table={"T": {"A", "B"}},
            asset_lengths_by_table={},
            sample_rows_by_table={},
            estimate_csv_bytes=_stub_estimate_csv_bytes,
            human_readable_size=_stub_human_readable,
        )
        assert result["incomplete"] is False
        assert result["incomplete_tables"] == []
        assert result["tables"]["T"]["incomplete"] is False
