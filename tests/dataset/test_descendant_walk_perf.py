"""Tests for the fast descendant-tree walk (perf spec 2026-06-14)."""

from __future__ import annotations

from pathlib import Path

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
