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

    # After W3 (this commit): the per-descendant list_dataset_members scan is
    # gone. _iter_descendant_rids memoizes the tree walk (1 full DD fetch) and
    # _exclude_empty_associations issues exactly one DD presence query for the
    # Dataset_Dataset association — ~2 total, independent of descendant count
    # (measured 2 for 6 descendants). Bound is a tight constant, not O(N).
    assert counter["dataset_dataset"] <= 5, (
        f"{counter['dataset_dataset']} Dataset_Dataset fetches for {len(children)} descendants "
        "— should be a small constant (~2) after the W3 aggregate-membership rewrite, not O(descendants)"
    )


def test_exclude_empty_associations_unchanged(catalog_manager: CatalogManager, tmp_path: Path):
    """The aggregate-membership rewrite excludes exactly the empty associations.

    Pins BOTH directions: a member-bearing association (Dataset_Subject/Image)
    is kept, and a genuinely-empty one (Dataset_File, no File members in the
    demo tree) is excluded. Also re-derives the OLD per-descendant member-scan
    result in-test and asserts set-equality — the load-bearing equivalence pin.
    """
    catalog_manager.reset()
    ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
    dataset = dataset_desc.dataset

    from deriva_ml.dataset.bag_builder import DatasetBagBuilder

    builder = DatasetBagBuilder(ml)
    excluded = builder._exclude_empty_associations(dataset)

    # --- Baseline: the OLD algorithm (per-descendant member scan) ---
    model = ml.model
    dataset_table = model.schemas[ml.ml_schema].tables["Dataset"]
    rids = [dataset.dataset_rid] + list(builder._iter_descendant_rids(dataset))
    member_element_types = set()
    for rid in rids:
        ds = ml.lookup_dataset(rid)
        for name, members in ds.list_dataset_members().items():
            if members:
                member_element_types.add(model.name_to_table(name))
    vocab_tables = {t for s in model.schemas.values() for t in s.tables.values() if model.is_vocabulary(t)}
    baseline = set()
    for assoc in dataset_table.find_associations():
        at = assoc.table
        links_member = any(fk.pk_table in member_element_types for fk in assoc.other_fkeys)
        links_vocab = any(fk.pk_table in vocab_tables for fk in assoc.other_fkeys)
        if not (links_member or links_vocab):
            baseline.add((at.schema.name, at.name))

    assert excluded == baseline, f"new={sorted(excluded)} != baseline={sorted(baseline)}"
    # Direction checks (belt and suspenders):
    excluded_names = {t for _, t in excluded}
    assert "Dataset_Subject" not in excluded_names
    assert "Dataset_Image" not in excluded_names
    assert ("deriva-ml", "Dataset_File") in excluded
