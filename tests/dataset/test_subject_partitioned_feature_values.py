"""Subject-partitioned feature reads: members are Subjects, Image is FK-reachable.
Regression for feature_values returning 0 (denormalize-fk-reachable-paths)."""

import os

import pytest

from deriva_ml.dataset.target_resolution import resolve_element_rids


@pytest.mark.skipif(os.environ.get("DERIVA_HOST") in (None, ""), reason="needs a live catalog")
def test_fixture_is_subject_partitioned(subject_partitioned_dataset):
    ml, ds = subject_partitioned_dataset
    bag = ds.download_dataset_bag(version=ds.current_version)
    assert len(resolve_element_rids(bag, "Image", reachable=False)) == 0  # no direct Image members
    assert len(resolve_element_rids(bag, "Image", reachable=True)) > 0  # but FK-reachable


@pytest.mark.skipif(os.environ.get("DERIVA_HOST") in (None, ""), reason="needs a live catalog")
def test_feature_values_resolves_via_fk_reachable_path(subject_partitioned_dataset):
    """feature_values('Image','Quality') must return rows for FK-reachable Images
    even though Image has 0 direct dataset members. Pre-fix this returned 0."""
    ml, ds = subject_partitioned_dataset
    bag = ds.download_dataset_bag(version=ds.current_version)

    reachable_images = set(resolve_element_rids(bag, "Image", reachable=True))
    assert reachable_images, "fixture must have FK-reachable Images"

    fv = list(bag.feature_values("Image", "Quality"))
    assert fv, "feature_values returned 0 — FK-reachable feature read is broken"
    # Every returned feature row targets a reachable Image (no leakage, no loss).
    fv_targets = {rec.Image for rec in fv}
    assert fv_targets <= reachable_images


@pytest.mark.skipif(os.environ.get("DERIVA_HOST") in (None, ""), reason="needs a live catalog")
def test_feature_count_matches_estimate_oracle(subject_partitioned_dataset):
    """Net-2 exactness oracle: feature_values row count == estimate's reachable
    feature-assoc-table count. Two independent code paths (planner UNION vs
    reachability BFS) must agree — catches a non-empty-but-wrong-count regression."""
    ml, ds = subject_partitioned_dataset
    bag = ds.download_dataset_bag(version=ds.current_version)

    # feature_table.name is the feature-association table backing "Quality"
    # (NOT "Image"): one feature_values row == one feature-assoc row.
    feat_assoc = bag.lookup_feature("Image", "Quality").feature_table.name

    # estimate_bag_size returns {"tables": {table_name: {"row_count", ...}}}
    # keyed by bare table name; row_count is the RID-distinct reachable count.
    est = ds.estimate_bag_size(ds.current_version)
    expected = est["tables"].get(feat_assoc, {}).get("row_count", 0)
    if not expected:
        pytest.skip(f"estimate has no reachable rows for {feat_assoc}")

    got = len(list(bag.feature_values("Image", "Quality")))
    assert got == expected, f"feature_values={got} estimate={expected} for {feat_assoc}"


@pytest.mark.skipif(os.environ.get("DERIVA_HOST") in (None, ""), reason="needs a live catalog")
def test_as_tf_dataset_labels_resolve_subject_partitioned(subject_partitioned_dataset):
    """as_tf_dataset with targets= must yield labeled (sample, target, rid) for
    FK-reachable images on a subject-partitioned dataset (was an empty generator)."""
    tf = pytest.importorskip("tensorflow")
    ml, ds = subject_partitioned_dataset
    bag = ds.download_dataset_bag(version=ds.current_version)
    reachable = set(resolve_element_rids(bag, "Image", reachable=True))

    dataset = bag.as_tf_dataset(
        element_type="Image",
        sample_loader=lambda p, row: tf.constant([0.0]),
        targets=["Quality"],
        target_transform=lambda rec: 0,  # any int; we assert count/shape, not value
        missing="skip",
    )
    rids = {r.decode() if isinstance(r, bytes) else r for *_rest, r in dataset.as_numpy_iterator()}
    assert rids, "empty generator — labels did not resolve (the reported bug)"
    assert rids <= reachable
