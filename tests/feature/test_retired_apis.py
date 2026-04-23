"""Retired-API shims — each raises a DerivaMLException with a replacement pointer.

These tests enforce the S2 migration contract: every retired method must raise
with a message that tells the user exactly where to go for the replacement.
Silent fallthrough or generic errors are not acceptable.
"""
from __future__ import annotations

import dataclasses

import pytest

from deriva_ml.core.exceptions import DerivaMLException


def test_ml_add_features_raises_with_pointer(test_ml) -> None:
    with pytest.raises(DerivaMLException, match=r"exe\.add_features"):
        test_ml.add_features([])


def test_ml_fetch_table_features_raises_with_pointer(test_ml) -> None:
    with pytest.raises(DerivaMLException, match=r"feature_values|Denormalizer"):
        test_ml.fetch_table_features("Image")


def test_ml_list_feature_values_raises_with_pointer(test_ml) -> None:
    with pytest.raises(DerivaMLException, match=r"feature_values"):
        test_ml.list_feature_values("Image", "x")


def test_ml_select_by_workflow_raises_with_pointer(test_ml) -> None:
    with pytest.raises(DerivaMLException, match=r"select_by_workflow.*container"):
        test_ml.select_by_workflow([], "wf")


# Bag-side retirements — use the materialized_bag_with_feature fixture from
# tests/dataset/conftest.py (visible via pytest conftest discovery).


def test_bag_fetch_table_features_raises(materialized_bag_with_feature) -> None:
    bag = materialized_bag_with_feature.bag
    with pytest.raises(DerivaMLException, match=r"feature_values|Denormalizer"):
        bag.fetch_table_features("Image")


def test_bag_list_feature_values_raises(materialized_bag_with_feature) -> None:
    bag = materialized_bag_with_feature.bag
    with pytest.raises(DerivaMLException, match=r"feature_values"):
        bag.list_feature_values("Image", "x")
