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
