"""Tests for ``DatasetBag.feature_values`` (Stage 3b).

Stage 3b rewired ``DatasetBag.feature_values`` to delegate to
``Denormalizer.feature_records`` (bag mode, ``source="local"``), replacing
the legacy ``BagFeatureCache`` TEXT-storage path. These tests exercise the
new path against a real materialized bag (the ``materialized_bag_with_feature``
fixture downloads one from the test catalog) — no synthetic fixtures, per the
"fixtures lie" lesson from the consolidation work.

The primary equivalence evidence is the live three-way oracle captured in the
PR body (bag-new == live-catalog ground truth, bag-old differed only by
returning tz-naive RCT). These tests pin the bag-side contract so it can't
regress.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from deriva_ml.core.exceptions import DerivaMLMaterializeLimitExceeded
from deriva_ml.feature import FeatureRecord


def test_feature_values_yields_typed_records(materialized_bag_with_feature) -> None:
    """``feature_values`` returns FeatureRecord instances for the bag's feature."""
    bag = materialized_bag_with_feature.bag
    records = list(
        bag.feature_values(
            materialized_bag_with_feature.target_table,
            materialized_bag_with_feature.feature_name,
        )
    )
    assert len(records) > 0, "Expected at least one feature record in bag"
    assert all(isinstance(r, FeatureRecord) for r in records)
    # Target FK is populated on every record (null-target rows are dropped).
    target = materialized_bag_with_feature.target_table
    assert all(getattr(r, target, None) is not None for r in records)


def test_feature_values_rct_is_utc_aware_iso(materialized_bag_with_feature) -> None:
    """RCT comes back as a UTC-aware ISO-8601 string — matching the live catalog.

    Regression guard for the divergence the legacy ``BagFeatureCache`` path
    had: it stored RCT as TEXT and returned it tz-naive (``...232194``),
    whereas the live catalog returns it UTC-aware (``...232194+00:00``). The
    Denormalizer path recovers RCT via ``_recover_system_columns`` with UTC
    re-attached, so the bag and the catalog now agree.
    """
    bag = materialized_bag_with_feature.bag
    records = list(
        bag.feature_values(
            materialized_bag_with_feature.target_table,
            materialized_bag_with_feature.feature_name,
        )
    )
    with_rct = [r for r in records if getattr(r, "RCT", None) is not None]
    assert with_rct, "Expected at least one record carrying an RCT"
    for r in with_rct:
        assert isinstance(r.RCT, str)
        parsed = datetime.fromisoformat(r.RCT)
        assert parsed.tzinfo is not None, f"RCT must be tz-aware, got {r.RCT!r}"
        assert parsed.utcoffset() == timezone.utc.utcoffset(None), (
            f"RCT must be UTC, got offset {parsed.utcoffset()} from {r.RCT!r}"
        )


def test_feature_values_selector_reduces_to_one_per_target(materialized_bag_with_feature) -> None:
    """A selector yields at most one record per target RID."""
    bag = materialized_bag_with_feature.bag
    target = materialized_bag_with_feature.target_table
    reduced = list(
        bag.feature_values(
            target,
            materialized_bag_with_feature.feature_name,
            selector=FeatureRecord.select_newest,
        )
    )
    target_rids = [getattr(r, target) for r in reduced]
    assert len(target_rids) == len(set(target_rids)), "selector must yield one record per target RID"


def test_feature_values_execution_rids_empty_short_circuits(materialized_bag_with_feature) -> None:
    """``execution_rids=[]`` short-circuits to an empty result."""
    bag = materialized_bag_with_feature.bag
    records = list(
        bag.feature_values(
            materialized_bag_with_feature.target_table,
            materialized_bag_with_feature.feature_name,
            execution_rids=[],
        )
    )
    assert records == []


def test_feature_values_materialize_limit_raises(materialized_bag_with_feature) -> None:
    """``materialize_limit`` below the row count raises DerivaMLMaterializeLimitExceeded."""
    bag = materialized_bag_with_feature.bag
    target = materialized_bag_with_feature.target_table
    feature = materialized_bag_with_feature.feature_name
    total = len(list(bag.feature_values(target, feature)))
    assert total > 0
    with pytest.raises(DerivaMLMaterializeLimitExceeded):
        list(bag.feature_values(target, feature, materialize_limit=total - 1))
