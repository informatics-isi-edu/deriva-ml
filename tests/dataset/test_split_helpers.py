"""Unit tests for ``split_dataset`` helpers (audit P1 Ds-split).

Pre-extraction, ``split_dataset`` was a 590-line function with
the dry-run path threading through the same body as the
live-catalog path. The audit asked for two helpers
(``_compute_partitions``, ``_create_split_hierarchy``) plus a
small validation helper, so the dry-run path could short-circuit
cleanly and each step could be unit-tested.

This file pins:

1. ``_validate_split_inputs`` — argument-shape checks (pure).
2. ``_compute_partitions`` — random-split path with a stubbed
   :class:`Dataset`. Tests the auto-detect element-table branch,
   single-table validation, partition splitting, and shuffling
   reproducibility. The stratified and ``selection_fn`` paths
   need pandas-DataFrame machinery already covered by the
   integration suite (``test_split.py``); the helper extraction
   itself is mechanical so the integration coverage is the right
   layer for those.

Pure-Python tests; no live catalog required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deriva_ml.dataset.split import (
    _compute_partitions,
    _validate_split_inputs,
)


class TestValidateSplitInputs:
    """``_validate_split_inputs`` enforces three mutual-exclusion / requires rules."""

    def test_no_extras_is_ok(self):
        _validate_split_inputs(stratify_by_column=None, selection_fn=None, include_tables=None)

    def test_stratify_and_selection_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _validate_split_inputs(
                stratify_by_column="x",
                selection_fn=lambda df, sizes, seed: {},
                include_tables=["T"],
            )

    def test_stratify_requires_include_tables(self):
        with pytest.raises(ValueError, match="include_tables is required"):
            _validate_split_inputs(stratify_by_column="x", selection_fn=None, include_tables=None)

    def test_stratify_with_empty_include_tables_raises(self):
        """Empty list is falsy — must trigger the requires check.

        Pin against a future "is None" drift that would silently
        accept ``include_tables=[]`` and fail downstream with a
        less-helpful error.
        """
        with pytest.raises(ValueError, match="include_tables is required"):
            _validate_split_inputs(stratify_by_column="x", selection_fn=None, include_tables=[])

    def test_selection_fn_requires_include_tables(self):
        with pytest.raises(ValueError, match="include_tables is required"):
            _validate_split_inputs(
                stratify_by_column=None,
                selection_fn=lambda df, sizes, seed: {},
                include_tables=None,
            )

    def test_stratify_alone_with_include_tables_is_ok(self):
        _validate_split_inputs(stratify_by_column="x", selection_fn=None, include_tables=["T"])

    def test_selection_fn_alone_with_include_tables_is_ok(self):
        _validate_split_inputs(
            stratify_by_column=None,
            selection_fn=lambda df, sizes, seed: {},
            include_tables=["T"],
        )


def _fake_dataset(members: dict[str, list[dict]]) -> MagicMock:
    """Build a :class:`Dataset` stand-in returning ``members``."""
    ds = MagicMock()
    ds.list_dataset_members.return_value = members
    return ds


class TestComputePartitionsRandom:
    """``_compute_partitions`` on the random (non-denormalized) path."""

    def test_auto_detects_single_candidate_table(self):
        members = {
            "Dataset": [],
            "Image": [{"RID": f"img-{i}"} for i in range(10)],
        }
        ds = _fake_dataset(members)
        partition_rids, partition_sizes, strategy, element_table = _compute_partitions(
            source_ds=ds,
            source_dataset_rid="DS-1",
            element_table=None,
            test_size=0.2,
            train_size=None,
            val_size=None,
            shuffle=False,
            seed=42,
            stratify_by_column=None,
            stratify_missing="error",
            include_tables=None,
            selection_fn=None,
            row_per=None,
            via=None,
            ignore_unrelated_anchors=False,
        )
        assert element_table == "Image"
        assert strategy == "random"
        assert partition_sizes == {"Training": 8, "Testing": 2}
        # No shuffle → deterministic ordering.
        assert partition_rids["Training"] == [f"img-{i}" for i in range(8)]
        assert partition_rids["Testing"] == ["img-8", "img-9"]

    def test_multiple_candidate_tables_requires_explicit_element_table(self):
        members = {
            "Dataset": [],
            "Image": [{"RID": "i1"}],
            "Subject": [{"RID": "s1"}],
        }
        ds = _fake_dataset(members)
        with pytest.raises(ValueError, match="multiple tables"):
            _compute_partitions(
                source_ds=ds,
                source_dataset_rid="DS-1",
                element_table=None,
                test_size=0.5,
                train_size=None,
                val_size=None,
                shuffle=False,
                seed=42,
                stratify_by_column=None,
                stratify_missing="error",
                include_tables=None,
                selection_fn=None,
                row_per=None,
                via=None,
                ignore_unrelated_anchors=False,
            )

    def test_empty_dataset_raises(self):
        members = {"Dataset": [], "Image": []}
        ds = _fake_dataset(members)
        with pytest.raises(ValueError, match="no members"):
            _compute_partitions(
                source_ds=ds,
                source_dataset_rid="DS-1",
                element_table=None,
                test_size=0.5,
                train_size=None,
                val_size=None,
                shuffle=False,
                seed=42,
                stratify_by_column=None,
                stratify_missing="error",
                include_tables=None,
                selection_fn=None,
                row_per=None,
                via=None,
                ignore_unrelated_anchors=False,
            )

    def test_specified_element_table_with_no_members_raises(self):
        members = {"Image": [{"RID": "i1"}], "Subject": []}
        ds = _fake_dataset(members)
        with pytest.raises(ValueError, match="no members in"):
            _compute_partitions(
                source_ds=ds,
                source_dataset_rid="DS-1",
                element_table="Subject",
                test_size=0.5,
                train_size=None,
                val_size=None,
                shuffle=False,
                seed=42,
                stratify_by_column=None,
                stratify_missing="error",
                include_tables=None,
                selection_fn=None,
                row_per=None,
                via=None,
                ignore_unrelated_anchors=False,
            )

    def test_three_way_split_with_val_size(self):
        members = {"Image": [{"RID": f"img-{i}"} for i in range(10)]}
        ds = _fake_dataset(members)
        partition_rids, partition_sizes, _, _ = _compute_partitions(
            source_ds=ds,
            source_dataset_rid="DS-1",
            element_table="Image",
            test_size=0.2,
            train_size=0.6,
            val_size=0.2,
            shuffle=False,
            seed=42,
            stratify_by_column=None,
            stratify_missing="error",
            include_tables=None,
            selection_fn=None,
            row_per=None,
            via=None,
            ignore_unrelated_anchors=False,
        )
        # Three partitions, sizes sum to 10.
        assert set(partition_sizes.keys()) == {"Training", "Validation", "Testing"}
        assert sum(partition_sizes.values()) == 10
        assert sum(len(rids) for rids in partition_rids.values()) == 10

    def test_shuffle_is_reproducible_with_seed(self):
        members = {"Image": [{"RID": f"img-{i}"} for i in range(20)]}
        ds_a = _fake_dataset(members)
        ds_b = _fake_dataset(members)

        partition_a, _, _, _ = _compute_partitions(
            source_ds=ds_a,
            source_dataset_rid="DS-1",
            element_table="Image",
            test_size=0.5,
            train_size=None,
            val_size=None,
            shuffle=True,
            seed=42,
            stratify_by_column=None,
            stratify_missing="error",
            include_tables=None,
            selection_fn=None,
            row_per=None,
            via=None,
            ignore_unrelated_anchors=False,
        )
        partition_b, _, _, _ = _compute_partitions(
            source_ds=ds_b,
            source_dataset_rid="DS-1",
            element_table="Image",
            test_size=0.5,
            train_size=None,
            val_size=None,
            shuffle=True,
            seed=42,
            stratify_by_column=None,
            stratify_missing="error",
            include_tables=None,
            selection_fn=None,
            row_per=None,
            via=None,
            ignore_unrelated_anchors=False,
        )
        # Same seed → identical splits.
        assert partition_a == partition_b

    def test_shuffle_with_different_seeds_diverges(self):
        members = {"Image": [{"RID": f"img-{i}"} for i in range(20)]}
        partition_a, _, _, _ = _compute_partitions(
            source_ds=_fake_dataset(members),
            source_dataset_rid="DS-1",
            element_table="Image",
            test_size=0.5,
            train_size=None,
            val_size=None,
            shuffle=True,
            seed=1,
            stratify_by_column=None,
            stratify_missing="error",
            include_tables=None,
            selection_fn=None,
            row_per=None,
            via=None,
            ignore_unrelated_anchors=False,
        )
        partition_b, _, _, _ = _compute_partitions(
            source_ds=_fake_dataset(members),
            source_dataset_rid="DS-1",
            element_table="Image",
            test_size=0.5,
            train_size=None,
            val_size=None,
            shuffle=True,
            seed=99999,
            stratify_by_column=None,
            stratify_missing="error",
            include_tables=None,
            selection_fn=None,
            row_per=None,
            via=None,
            ignore_unrelated_anchors=False,
        )
        # Different seeds → different orderings (very high probability for n=20).
        assert partition_a != partition_b

    def test_strategy_description_random(self):
        members = {"Image": [{"RID": "i1"}, {"RID": "i2"}]}
        _, _, strategy, _ = _compute_partitions(
            source_ds=_fake_dataset(members),
            source_dataset_rid="DS-1",
            element_table="Image",
            test_size=0.5,
            train_size=None,
            val_size=None,
            shuffle=False,
            seed=42,
            stratify_by_column=None,
            stratify_missing="error",
            include_tables=None,
            selection_fn=None,
            row_per=None,
            via=None,
            ignore_unrelated_anchors=False,
        )
        assert strategy == "random"
