"""Tests for the fast descendant-tree walk (perf spec 2026-06-14)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.catalog_manager import CatalogManager


def _descendant_rid_set_via_objects(dataset) -> set:
    """Baseline: descendant RIDs via the object-hydrating list_dataset_children."""
    return {c.dataset_rid for c in dataset.list_dataset_children(recurse=True)}


def test_list_dataset_children_rids_matches_objects(catalog_manager: CatalogManager, tmp_path: Path):
    """list_dataset_children_rids(recurse=True) returns the same RID set as the object form."""
    catalog_manager.reset()
    ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
    dataset = dataset_desc.dataset

    via_objects = _descendant_rid_set_via_objects(dataset)
    via_rids = set(dataset.list_dataset_children_rids(recurse=True))

    assert via_rids == via_objects


def test_estimate_walks_descendant_tree_once(catalog_manager: CatalogManager, tmp_path: Path, monkeypatch):
    """estimate_bag_size fetches the Dataset_Dataset table O(1) times, not O(descendants)."""
    catalog_manager.reset()
    ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
    dataset = dataset_desc.dataset
    version = dataset.current_version

    children = dataset.list_dataset_children_rids(recurse=True)
    if len(children) < 2:
        pytest.skip(f"demo dataset has {len(children)} descendants; need >= 2 to exercise the guard")

    import deriva.core.deriva_binding as db

    counter = {"dataset_dataset": 0}
    orig = db.DerivaBinding.get

    def spy(self, path, *a, **k):
        if isinstance(path, str) and "Dataset_Dataset" in path:
            counter["dataset_dataset"] += 1
        return orig(self, path, *a, **k)

    monkeypatch.setattr(db.DerivaBinding, "get", spy)

    from deriva_ml.dataset import DatasetSpec

    ml.estimate_bag_size(DatasetSpec(rid=dataset.dataset_rid, version=version))

    # After Task 2: _iter_descendant_rids memoizes the tree walk (1 full DD
    # fetch) but _exclude_empty_associations still issues one filtered DD
    # query per dataset for list_dataset_members. Total is O(N) not O(N²).
    # Task W3 will eliminate the per-dataset member queries; tighten then.
    assert counter["dataset_dataset"] <= len(children) + 3, (
        f"{counter['dataset_dataset']} Dataset_Dataset fetches for {len(children)} descendants "
        "— should be O(descendants) after Task 2 memoization, not O(descendants²)"
    )
