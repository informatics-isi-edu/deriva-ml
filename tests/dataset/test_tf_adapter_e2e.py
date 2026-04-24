"""End-to-end TensorFlow adapter test using a real bag fixture.

Gated on tensorflow being importable. Uses catalog_with_datasets fixture
to build a real bag, then iterates under tf.data.Dataset.batch().prefetch().
"""
from __future__ import annotations

import pytest

tf = pytest.importorskip("tensorflow")


def test_bag_to_tf_dataset_yields_batches(catalog_with_datasets, tmp_path):
    ml, dataset_desc = catalog_with_datasets
    bag = dataset_desc.dataset.download_dataset_bag(version="1.0.0")

    ds = bag.as_tf_dataset(
        element_type="Image",
        sample_loader=lambda p, row: tf.constant([1.0, 2.0, 3.0]),
        targets=None,  # unlabeled — just exercise the plumbing
    )
    ds = ds.batch(2).prefetch(2)
    batches = list(ds)
    assert len(batches) > 0
    assert batches[0].shape[0] <= 2
