"""Tests for ``Dataset.estimate_bag_size`` exactness and CSV-byte estimation.

The estimate engine reconstructs FK reachability **client-side** (see
``src/deriva_ml/dataset/_reachability.py``) and computes exact per-table RID
unions in memory, replacing the old deep server-side FK-join query path.

This module holds two complementary checks:

- ``test_reachability_matches_server_union`` — a LIVE exactness guard that
  runs the fast client-side estimate against a real catalog and, as an
  independent oracle, recomputes each reached table's row count the slow way
  (fetch RID lists from every FK path, union them server-side-shaped) and
  asserts they agree table by table. This is the regression that makes
  "exact" enforceable end-to-end.
- ``TestEstimateCsvBytesStatic`` — a pure-Python unit test for the
  ``Dataset._estimate_csv_bytes`` static helper (no catalog needed). The
  union/assembly logic itself is covered separately in
  ``tests/dataset/test_estimate_helpers.py``.
"""

from __future__ import annotations

import os

import pytest

from deriva_ml.dataset.bag_builder import DatasetBagBuilder


@pytest.mark.skipif(
    os.environ.get("DERIVA_HOST") in (None, ""),
    reason="needs a live catalog",
)
def test_reachability_matches_server_union(catalog_with_datasets):
    """The client-side reachability engine's per-table counts must equal the
    server-side RID-union for every reached table (exactness guard).

    This is the regression that makes 'exact' enforceable: it runs the fast
    client-side estimate AND, as an independent oracle, recomputes each
    reached table's row count the slow way (fetch RID lists from every FK
    path, union them server-side-shaped) and asserts they agree table by table.
    """
    ml, _dataset_desc = catalog_with_datasets

    # Pick a nested dataset (has multi-path tables, the interesting case).
    datasets = list(ml.find_datasets())
    assert datasets, "fixture produced no datasets"
    nested = None
    for ds_obj in datasets:
        if ds_obj.list_dataset_children():
            nested = ds_obj
            break
    if nested is None:
        # A non-nested dataset wouldn't exercise the multi-path RID union
        # this test exists to guard — a green pass there is false comfort.
        pytest.skip("fixture has no nested dataset to exercise multi-path union")

    version = nested.current_version

    # Client-side engine result.
    est = nested.estimate_bag_size(version)
    client_counts = {t: d["row_count"] for t, d in est["tables"].items()}

    # Server-union reference via aggregate_queries (the oracle): for each
    # reached table, fetch RID lists from every FK path and union them.
    snap = nested._version_snapshot_catalog(version)
    builder = DatasetBagBuilder(ml_instance=snap)
    tq = builder.aggregate_queries(nested)
    assert tq, "aggregate_queries reached no tables"
    for tname, entries in tq.items():
        union: set[str] = set()
        for dp, pb_tbl, _is_asset in entries:
            for row in dp.attributes(pb_tbl.RID).fetch():
                union.add(row["RID"])
        assert client_counts.get(tname, 0) == len(union), (
            f"{tname}: client={client_counts.get(tname)} server={len(union)}"
        )


class TestEstimateCsvBytesStatic:
    """Unit tests for the ``Dataset._estimate_csv_bytes`` static method."""

    def test_estimate_csv_bytes_static(self):
        """``_estimate_csv_bytes`` scales the sample's average row size by the
        total row count and returns 0 for an empty sample."""
        from deriva_ml.dataset.dataset import Dataset

        # No rows -> 0
        assert Dataset._estimate_csv_bytes([], 0) == 0

        # Some rows
        sample = [
            {"RID": "A", "Name": "Alice", "Score": 95},
            {"RID": "B", "Name": "Bob", "Score": 87},
        ]
        result = Dataset._estimate_csv_bytes(sample, 1000)
        assert result > 0
        # Should be roughly: header + 1000 * avg_row_size
        # Each row is ~15-20 bytes in CSV, so total ~15-20KB
        assert 10000 < result < 50000
