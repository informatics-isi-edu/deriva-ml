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
    ds_obj = dataset_desc.dataset
    # current_version is the snapshot AFTER add_dataset_members; "1.0.0" is the
    # empty pre-membership snapshot and would yield zero Image samples.
    bag = ds_obj.download_dataset_bag(version=ds_obj.current_version)

    ds = bag.as_torch_dataset(
        element_type="Image",
        sample_loader=lambda p, row: torch.tensor([1.0, 2.0, 3.0]),
        targets=None,  # unlabeled — just exercise the plumbing
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) > 0
    # Unlabeled dataset yields (sample, rid); default collate gives
    # [sample_batch, rid_batch]. The sample batch carries the batch dim.
    sample_batch, rid_batch = batches[0]
    assert sample_batch.shape[0] <= 2
    assert len(rid_batch) == sample_batch.shape[0]


def test_as_torch_dataset_enumerates_fk_reachable_images(catalog_with_datasets, tmp_path):
    """The torch adapter enumerates the SAME element set as the shared
    FK-reachable core (resolve_element_rids / resolve_reachable_rows) that
    restructure_assets uses — Images reachable via Subject -> ... -> Image,
    not just direct members.

    Regression for the bug where as_torch_dataset enumerated via
    list_dataset_members (direct + nested only) and missed FK-reachable assets.
    """
    from deriva_ml.dataset.target_resolution import (
        resolve_element_rids,
        resolve_reachable_rows,
    )

    ml, dataset_desc = catalog_with_datasets
    ds_obj = dataset_desc.dataset
    # current_version is the snapshot AFTER add_dataset_members; "1.0.0" is the
    # empty pre-membership snapshot and has no Image rows.
    bag = ds_obj.download_dataset_bag(version=ds_obj.current_version)

    # The shared core (what restructure_assets delegates to) is the oracle.
    expected = set(resolve_element_rids(bag, "Image", reachable=True))
    assert expected, "fixture bag must contain FK-reachable Image rows"

    # The adapter must yield exactly that RID set (RID is the last item).
    ds = bag.as_torch_dataset(
        element_type="Image",
        sample_loader=lambda p, row: torch.tensor([0.0]),
        targets=None,
    )
    yielded = {ds[i][-1] for i in range(len(ds))}
    assert yielded == expected

    # resolve_reachable_rows (un-deduped rows) dedups to the same RID set.
    row_rids = {r["RID"] for r in resolve_reachable_rows(bag, "Image")}
    assert row_rids == expected

    # Opt-out: reachable=False yields the direct-member RID set.
    direct = set(resolve_element_rids(bag, "Image", reachable=False))
    ds_direct = bag.as_torch_dataset(
        element_type="Image",
        sample_loader=lambda p, row: torch.tensor([0.0]),
        targets=None,
        reachable=False,
    )
    yielded_direct = {ds_direct[i][-1] for i in range(len(ds_direct))}
    assert yielded_direct == direct

    # The bug: FK-reachable enumeration surfaces Images (reachable via the
    # nested Subject datasets) that direct-membership recursion misses. On this
    # fixture, reachable finds strictly more than direct.
    assert direct <= expected
    assert len(expected) > len(direct)
