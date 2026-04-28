"""Integration tests for ``find_datasets(sort=...)``.

Catalog-required. Validates the three sort modes (None / True / callable).
``Dataset`` has no public ``rct`` attribute, so the "newest-first" test
asserts on ``dataset_rid`` (descending) as a proxy: RIDs are server-
assigned monotonically on a fresh catalog, so the lexicographic order
correlates with insertion order.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_find_datasets_sort_none_returns_records(catalog_with_datasets):
    """sort=None preserves backend order; just verify it returns rows."""
    ml, _ = catalog_with_datasets
    datasets = list(ml.find_datasets())
    assert len(datasets) >= 2, "fixture should provide at least 2 datasets"


@pytest.mark.integration
def test_find_datasets_sort_true_returns_newest_first(catalog_with_datasets):
    """sort=True yields records ordered by RCT descending.

    Dataset has no public rct attribute, so we use dataset_rid as a
    proxy: server-assigned RIDs are monotonic on a fresh catalog,
    so RID-desc correlates with RCT-desc.
    """
    ml, _ = catalog_with_datasets
    datasets = list(ml.find_datasets(sort=True))
    rids = [d.dataset_rid for d in datasets]
    assert rids == sorted(rids, reverse=True), f"datasets should be newest-first (RID-desc proxy); got rids={rids}"


@pytest.mark.integration
def test_find_datasets_sort_callable_applies_user_keys(catalog_with_datasets):
    """User-supplied sort callable applies."""
    ml, _ = catalog_with_datasets

    def by_rid_asc(path):
        return path.RID

    datasets = list(ml.find_datasets(sort=by_rid_asc))
    rids = [d.dataset_rid for d in datasets]
    assert rids == sorted(rids), f"datasets should be RID-ascending; got {rids}"


@pytest.mark.integration
def test_find_datasets_sort_invalid_type_raises(catalog_with_datasets):
    """Passing a bare string (not None/True/callable) raises TypeError."""
    ml, _ = catalog_with_datasets
    with pytest.raises(TypeError, match="sort must be None, True, or a callable"):
        list(ml.find_datasets(sort="newest"))  # type: ignore[arg-type]
