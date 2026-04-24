"""End-to-end torch adapter test using a real bag fixture.

Gated on torch being importable. Uses catalog_with_datasets fixture
to build a real bag, then iterates under a real DataLoader.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader  # noqa: E402


def test_bag_to_dataloader_yields_tensors(catalog_with_datasets, tmp_path):
    ml, dataset_desc = catalog_with_datasets
    bag = dataset_desc.dataset.download_dataset_bag(version="1.0.0")

    ds = bag.as_torch_dataset(
        element_type="Image",
        sample_loader=lambda p, row: torch.tensor([1.0, 2.0, 3.0]),
        targets=None,  # unlabeled — just exercise the plumbing
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) > 0
    assert batches[0].shape[0] <= 2
