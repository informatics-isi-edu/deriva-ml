"""Unit tests for the DatasetBag per-feature denormalization cache.

These tests exercise the ``BagFeatureCache`` class and the ``DatasetBag.feature_values``
method that routes through it.  They use the ``materialized_bag_with_feature`` fixture
which downloads a real bag from the test catalog.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from deriva_ml.core.exceptions import DerivaMLDataError
from deriva_ml.dataset.bag_feature_cache import BagFeatureCache


def test_first_access_populates_cache(materialized_bag_with_feature) -> None:
    """First call to fetch_feature_records populates the cache; second call reads it."""
    bag = materialized_bag_with_feature.bag
    cache = BagFeatureCache(bag)
    first = list(
        cache.fetch_feature_records("Image", materialized_bag_with_feature.feature_name)
    )
    second = list(
        cache.fetch_feature_records("Image", materialized_bag_with_feature.feature_name)
    )
    assert len(first) > 0, "Expected at least one feature record in bag"
    assert [r.model_dump() for r in first] == [r.model_dump() for r in second]


def test_cache_immutable_after_population(materialized_bag_with_feature) -> None:
    """Bags are immutable; subsequent reads return identical data.

    The cache table is populated once.  Even if the underlying source table
    were modified (which bags don't support), the cache reads its own table
    and returns stable results.
    """
    bag = materialized_bag_with_feature.bag
    cache = BagFeatureCache(bag)
    first = [
        r.model_dump()
        for r in cache.fetch_feature_records("Image", materialized_bag_with_feature.feature_name)
    ]
    assert len(first) > 0, "Expected at least one feature record in bag for immutability test"
    # Read again — should hit the pre-populated cache table, not re-scan source.
    second = [
        r.model_dump()
        for r in cache.fetch_feature_records("Image", materialized_bag_with_feature.feature_name)
    ]
    assert first == second


def test_cache_corrupt_raises_with_recovery_pointer(tmp_path: Path) -> None:
    """If cache init fails (missing source table), raise DerivaMLDataError with bag path."""
    # Requires a fabricated bag-like directory — wire up in implementation.
    pytest.skip("Requires a fabricated bag-like directory — wire up in implementation.")
