"""Unit tests for the ``partition_by`` parameter on ``split_dataset``.

These tests pin the design decision that ``split_dataset`` must
make the partition unit (element vs row) an explicit caller
intent whenever the (``row_per``, ``element_table``) pair is
ambiguous. The motivating finding is the e2e-run train/test
leakage where ``row_per=<feature_table>`` silently sent the same
Image RID's annotation rows to opposite partitions (see
``findings/curator/02-train-test-leakage-in-labeled-split-datasets.md``
in the model template's e2e archive).

The unit tests use a stubbed :class:`Dataset` whose
``get_denormalized_as_dataframe`` returns a synthetic DataFrame
modelling the relevant 1:N-feature shape, so they run without a
live catalog.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from deriva_ml.dataset.split import (
    _compute_partitions,
    _dedupe_for_element_partition,
    _validate_split_inputs,
    split_dataset,
)


def _fake_dataset_with_denorm(members: dict[str, list[dict]], denorm_df: pd.DataFrame) -> MagicMock:
    """Build a :class:`Dataset` stand-in for the denormalization path.

    Args:
        members: ``list_dataset_members`` return value.
        denorm_df: DataFrame that ``get_denormalized_as_dataframe``
            returns regardless of arguments.
    """
    ds = MagicMock()
    ds.list_dataset_members.return_value = members
    ds.get_denormalized_as_dataframe.return_value = denorm_df
    return ds


def _multi_row_feature_df(
    *,
    n_images: int,
    rows_per_image: int = 2,
    agreeing_labels: bool = True,
    disagreement_rids: list[str] | None = None,
) -> pd.DataFrame:
    """Build a synthetic denormalized DataFrame modelling a 1:N feature.

    Args:
        n_images: Number of distinct Image RIDs to synthesise.
        rows_per_image: How many annotation rows each image carries.
            Mimics two annotators per image when 2.
        agreeing_labels: When True, every image's annotation rows
            carry the same label (so the within-element check
            passes). When False, ``disagreement_rids`` are flipped.
        disagreement_rids: Image RIDs to give disagreeing labels.
            Only used when ``agreeing_labels`` is False.

    Returns:
        DataFrame with columns ``Image.RID`` (str),
        ``Image_Classification.Image_Class`` (str), and a dummy
        ``Image_Classification.RID`` column.
    """
    rng = np.random.default_rng(123)
    classes = np.array(["cat", "dog", "bird", "fish"])
    image_rids = [f"img-{i:04d}" for i in range(n_images)]
    image_labels = {rid: classes[rng.integers(len(classes))] for rid in image_rids}
    disagree = set(disagreement_rids or [])

    rows = []
    for rid in image_rids:
        for k in range(rows_per_image):
            if not agreeing_labels and rid in disagree and k == 1:
                # Pick a class different from the chosen label.
                other = next(c for c in classes if c != image_labels[rid])
                label = other
            else:
                label = image_labels[rid]
            rows.append(
                {
                    "Image.RID": rid,
                    "Image_Classification.Image_Class": label,
                    "Image_Classification.RID": f"feat-{rid}-{k}",
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# _validate_split_inputs — partition_by rules
# =============================================================================


class TestValidatePartitionBy:
    """Pin the (row_per, element_table, partition_by) validation matrix."""

    def test_partition_by_defaults_to_element_when_row_per_omitted(self):
        result = _validate_split_inputs(
            stratify_by_column=None,
            selection_fn=None,
            include_tables=None,
            row_per=None,
            element_table="Image",
            partition_by=None,
        )
        assert result == "element"

    def test_partition_by_defaults_to_element_when_row_per_equals_element_table(self):
        result = _validate_split_inputs(
            stratify_by_column="Image_Classification.Image_Class",
            selection_fn=None,
            include_tables=["Image", "Image_Classification"],
            row_per="Image",
            element_table="Image",
            partition_by=None,
        )
        assert result == "element"

    def test_partition_by_required_when_row_per_differs_from_element_table(self):
        with pytest.raises(ValueError) as exc:
            _validate_split_inputs(
                stratify_by_column="Image_Classification.Image_Class",
                selection_fn=None,
                include_tables=["Image", "Execution_Image_Image_Classification"],
                row_per="Execution_Image_Image_Classification",
                element_table="Image",
                partition_by=None,
            )
        msg = str(exc.value)
        # Message must name both modes so the caller can fix the call site.
        assert "partition_by" in msg
        assert "'element'" in msg
        assert "'row'" in msg
        assert "row_per" in msg

    def test_explicit_partition_by_element_returns_element(self):
        result = _validate_split_inputs(
            stratify_by_column="Image_Classification.Image_Class",
            selection_fn=None,
            include_tables=["Image", "Execution_Image_Image_Classification"],
            row_per="Execution_Image_Image_Classification",
            element_table="Image",
            partition_by="element",
        )
        assert result == "element"

    def test_explicit_partition_by_row_returns_row(self):
        result = _validate_split_inputs(
            stratify_by_column="Image_Classification.Image_Class",
            selection_fn=None,
            include_tables=["Image", "Execution_Image_Image_Classification"],
            row_per="Execution_Image_Image_Classification",
            element_table="Image",
            partition_by="row",
        )
        assert result == "row"

    def test_invalid_partition_by_raises(self):
        with pytest.raises(ValueError, match="partition_by must be"):
            _validate_split_inputs(
                stratify_by_column=None,
                selection_fn=None,
                include_tables=None,
                row_per=None,
                element_table="Image",
                partition_by="oops",  # type: ignore[arg-type]
            )


# =============================================================================
# _dedupe_for_element_partition — uniformity check
# =============================================================================


class TestDedupeForElementPartition:
    """Within-element uniformity + stable dedupe."""

    def test_agreeing_labels_dedupe_to_one_row_per_rid(self):
        df = _multi_row_feature_df(n_images=10, rows_per_image=2)
        deduped = _dedupe_for_element_partition(
            df,
            rid_column="Image.RID",
            element_table="Image",
            stratify_by_column="Image_Classification.Image_Class",
            stratify_missing="error",
            using_selection_fn=False,
        )
        # 10 distinct Image RIDs in the synthetic source.
        assert len(deduped) == 10
        assert set(deduped["Image.RID"]) == set(df["Image.RID"])

    def test_dedupe_is_seed_deterministic(self):
        df = _multi_row_feature_df(n_images=10, rows_per_image=3)
        a = _dedupe_for_element_partition(
            df,
            rid_column="Image.RID",
            element_table="Image",
            stratify_by_column="Image_Classification.Image_Class",
            stratify_missing="error",
            using_selection_fn=False,
        )
        b = _dedupe_for_element_partition(
            df,
            rid_column="Image.RID",
            element_table="Image",
            stratify_by_column="Image_Classification.Image_Class",
            stratify_missing="error",
            using_selection_fn=False,
        )
        pd.testing.assert_frame_equal(a, b)

    def test_disagreement_raises_with_consensus_pattern_message(self):
        df = _multi_row_feature_df(
            n_images=10,
            rows_per_image=2,
            agreeing_labels=False,
            disagreement_rids=["img-0003", "img-0007"],
        )
        with pytest.raises(ValueError) as exc:
            _dedupe_for_element_partition(
                df,
                rid_column="Image.RID",
                element_table="Image",
                stratify_by_column="Image_Classification.Image_Class",
                stratify_missing="error",
                using_selection_fn=False,
            )
        msg = str(exc.value)
        # Names the consensus-feature pattern explicitly.
        assert "consensus" in msg.lower()
        # Suggests partition_by='row' as the alternative.
        assert "partition_by='row'" in msg
        # Surfaces at least one offending RID.
        assert "img-0003" in msg or "img-0007" in msg

    def test_selection_fn_path_skips_uniformity_check(self):
        df = _multi_row_feature_df(
            n_images=10,
            rows_per_image=2,
            agreeing_labels=False,
            disagreement_rids=["img-0003"],
        )
        # Even with disagreeing labels, using_selection_fn=True
        # bypasses the check because the read set is opaque.
        deduped = _dedupe_for_element_partition(
            df,
            rid_column="Image.RID",
            element_table="Image",
            stratify_by_column=None,
            stratify_missing="error",
            using_selection_fn=True,
        )
        assert len(deduped) == 10


# =============================================================================
# _compute_partitions — end-to-end through the denormalize path
# =============================================================================


class TestComputePartitionsElementMode:
    """``_compute_partitions`` with ``partition_by="element"``."""

    def test_element_mode_dedupes_multi_row_feature(self):
        """Two-row-per-image dataframe partitions disjointly at element level."""
        n_images = 100
        df = _multi_row_feature_df(n_images=n_images, rows_per_image=2)
        members = {"Image": [{"RID": f"img-{i:04d}"} for i in range(n_images)]}
        ds = _fake_dataset_with_denorm(members, df)

        partition_rids, partition_sizes, _, _ = _compute_partitions(
            source_ds=ds,
            source_dataset_rid="DS-1",
            element_table="Image",
            test_size=0.2,
            train_size=None,
            val_size=None,
            shuffle=True,
            seed=42,
            stratify_by_column="Image_Classification.Image_Class",
            stratify_missing="error",
            include_tables=["Image", "Execution_Image_Image_Classification"],
            selection_fn=None,
            row_per="Execution_Image_Image_Classification",
            via=None,
            ignore_unrelated_anchors=False,
            partition_by="element",
        )

        # Element-level disjointness — the property the e2e leak violated.
        train_set = set(partition_rids["Training"])
        test_set = set(partition_rids["Testing"])
        assert train_set.isdisjoint(test_set)
        # Sizes line up with the requested fractions of the deduped count.
        assert partition_sizes == {"Training": 80, "Testing": 20}
        assert len(partition_rids["Training"]) == 80
        assert len(partition_rids["Testing"]) == 20

    def test_element_mode_errors_on_within_element_disagreement(self):
        """Stratifying on a disagreeing feature raises with consensus guidance."""
        n_images = 20
        df = _multi_row_feature_df(
            n_images=n_images,
            rows_per_image=2,
            agreeing_labels=False,
            disagreement_rids=["img-0005", "img-0012"],
        )
        members = {"Image": [{"RID": f"img-{i:04d}"} for i in range(n_images)]}
        ds = _fake_dataset_with_denorm(members, df)

        with pytest.raises(ValueError) as exc:
            _compute_partitions(
                source_ds=ds,
                source_dataset_rid="DS-1",
                element_table="Image",
                test_size=0.2,
                train_size=None,
                val_size=None,
                shuffle=True,
                seed=42,
                stratify_by_column="Image_Classification.Image_Class",
                stratify_missing="error",
                include_tables=["Image", "Execution_Image_Image_Classification"],
                selection_fn=None,
                row_per="Execution_Image_Image_Classification",
                via=None,
                ignore_unrelated_anchors=False,
                partition_by="element",
            )
        msg = str(exc.value)
        assert "consensus" in msg.lower()
        # At least one offending RID surfaces.
        assert "img-0005" in msg or "img-0012" in msg


class TestComputePartitionsRowMode:
    """``_compute_partitions`` with ``partition_by="row"``."""

    def test_row_mode_allows_element_overlap(self):
        """``partition_by='row'`` produces row-level partitions without dedupe.

        The defining property of row-mode is that the same element_table
        RID may appear in multiple partitions — each denormalized row
        is its own observation. With 2 rows per image and a stratified
        50/50 split, we expect overlap with very high probability.
        """
        n_images = 50
        # Force disagreement to confirm uniformity is NOT enforced in
        # row mode. We use a label that already exists in the synthesised
        # source so stratified sampling stays well-conditioned.
        df = _multi_row_feature_df(
            n_images=n_images,
            rows_per_image=2,
            agreeing_labels=False,
            disagreement_rids=[f"img-{i:04d}" for i in range(0, n_images, 3)],
        )
        members = {"Image": [{"RID": f"img-{i:04d}"} for i in range(n_images)]}
        ds = _fake_dataset_with_denorm(members, df)

        partition_rids, _, _, _ = _compute_partitions(
            source_ds=ds,
            source_dataset_rid="DS-1",
            element_table="Image",
            test_size=0.5,
            train_size=None,
            val_size=None,
            shuffle=True,
            seed=42,
            stratify_by_column="Image_Classification.Image_Class",
            stratify_missing="error",
            include_tables=["Image", "Execution_Image_Image_Classification"],
            selection_fn=None,
            row_per="Execution_Image_Image_Classification",
            via=None,
            ignore_unrelated_anchors=False,
            partition_by="row",
        )

        # Row-level partitioning -> the same Image RID may appear in both.
        train_set = set(partition_rids["Training"])
        test_set = set(partition_rids["Testing"])
        overlap = train_set & test_set
        # With 2 rows per image and a 50/50 stratified split, overlap is
        # essentially certain. The test asserts the property the API now
        # exposes: row mode does not enforce element-level disjointness.
        assert len(overlap) > 0, (
            "partition_by='row' must allow element-level overlap "
            "(per-annotation partitioning is the legitimate use case)"
        )


# =============================================================================
# Defensive disjointness assertion
# =============================================================================


class TestDisjointnessInvariant:
    """The final ``assert`` in ``_compute_partitions``'s element path."""

    def test_disjointness_invariant_fires_when_dedupe_regresses(self, monkeypatch: pytest.MonkeyPatch):
        """A broken dedupe (duplicates leak through) must trip the assertion.

        The assertion exists as a defensive invariant — it cannot fire
        on a correct dedupe. To exercise it we monkeypatch the dedupe
        helper to a no-op, simulating a future regression in the
        dedupe logic. The subsequent random selector then scatters the
        duplicate rows across partitions, and the disjointness check
        catches it.
        """
        from deriva_ml.dataset import split as split_module

        def _broken_dedupe(df, **_kwargs):
            # Return the input unchanged — fail to dedupe.
            return df.reset_index(drop=True)

        monkeypatch.setattr(split_module, "_dedupe_for_element_partition", _broken_dedupe)

        n_images = 40
        df = _multi_row_feature_df(n_images=n_images, rows_per_image=2)
        members = {"Image": [{"RID": f"img-{i:04d}"} for i in range(n_images)]}
        ds = _fake_dataset_with_denorm(members, df)

        with pytest.raises(AssertionError, match="disjointness invariant"):
            _compute_partitions(
                source_ds=ds,
                source_dataset_rid="DS-1",
                element_table="Image",
                test_size=0.5,
                train_size=None,
                val_size=None,
                shuffle=True,
                seed=42,
                stratify_by_column="Image_Classification.Image_Class",
                stratify_missing="error",
                include_tables=["Image", "Execution_Image_Image_Classification"],
                selection_fn=None,
                row_per="Execution_Image_Image_Classification",
                via=None,
                ignore_unrelated_anchors=False,
                partition_by="element",
            )


# =============================================================================
# Public surface — split_dataset signature carries partition_by
# =============================================================================


class TestSplitDatasetSurface:
    """Confirm the public function exposes ``partition_by`` correctly."""

    def test_partition_by_in_signature(self):
        import inspect

        sig = inspect.signature(split_dataset)
        assert "partition_by" in sig.parameters
        # Default is None (caller-must-decide-when-ambiguous semantics).
        assert sig.parameters["partition_by"].default is None
