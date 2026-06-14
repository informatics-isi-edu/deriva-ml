"""Regression guard: estimate_bag_size /schema fetches don't scale with nesting.

Pins the fix from
docs/superpowers/specs/2026-06-13-estimate-bag-size-perf-design.md.
Before the fix, estimate_bag_size on an N-descendant dataset issued
O(N) /schema fetches (a fresh snapshot catalog per descendant). After,
it issues a small fixed number independent of N.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deriva_ml.dataset import DatasetSpec


def _count_schema_gets(monkeypatch) -> dict:
    """Patch DerivaBinding.get to count /schema requests; returns a live counter."""
    import deriva.core.deriva_binding as db

    counter = {"schema": 0, "total": 0}
    orig = db.DerivaBinding.get

    def spy(self, path, *a, **k):
        counter["total"] += 1
        if isinstance(path, str) and path.split("?")[0].endswith("/schema"):
            counter["schema"] += 1
        return orig(self, path, *a, **k)

    monkeypatch.setattr(db.DerivaBinding, "get", spy)
    return counter


def test_schema_fetches_independent_of_nesting(catalog_manager, tmp_path: Path, monkeypatch):
    """estimate_bag_size /schema count is a small constant, not O(descendants)."""
    catalog_manager.reset()
    ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
    dataset = dataset_desc.dataset
    version = dataset.current_version

    # Confirm the demo dataset actually nests; otherwise the guard is vacuous.
    children = dataset.list_dataset_children(recurse=True)
    if len(children) < 2:
        pytest.skip(f"demo dataset has {len(children)} descendants; need >= 2 to exercise the guard")

    counter = _count_schema_gets(monkeypatch)
    ml.estimate_bag_size(DatasetSpec(rid=dataset.dataset_rid, version=version))

    # Post-fix the count is a fixed constant from two sources:
    # 1. DerivaModel.is_dataset_rid() calls self.getPathBuilder() directly
    #    on the raw catalog object (2 calls, fixed).
    # 2. CatalogBagBuilder.iter_table_datapaths() / _validate_anchors()
    #    call self.catalog.getPathBuilder() directly — deriva-py internals
    #    that bypass the DerivaML mixin cache (5 calls, fixed per build).
    # Neither scales with the number of descendants — they are constant
    # overhead per estimate_bag_size call. Ceiling is 10 to absorb any
    # minor deriva-py version variance while still blocking O(N) regression.
    assert counter["schema"] <= 10, (
        f"estimate issued {counter['schema']} /schema fetches for "
        f"{len(children)} descendants — should be a small constant, not O(N)"
    )
