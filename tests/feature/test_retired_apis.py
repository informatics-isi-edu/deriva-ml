"""Retired-API shims — each raises a DerivaMLException with a replacement pointer.

These tests enforce the S2 migration contract: every retired method must raise
with a message that tells the user exactly where to go for the replacement.
Silent fallthrough or generic errors are not acceptable.
"""
from __future__ import annotations

import pytest

from deriva_ml.core.exceptions import DerivaMLException


def test_ml_add_features_raises_with_pointer(test_ml) -> None:
    with pytest.raises(DerivaMLException, match=r"exe\.add_features"):
        test_ml.add_features([])


# Three ml-side tombstones (``fetch_table_features``,
# ``list_feature_values``, ``select_by_workflow``) were deleted
# outright in the core/ P1 sweep per the workspace "no
# backwards-compat shims" rule. Callers now see a clean
# ``AttributeError`` from Python; the message lives in the
# changelog rather than in a method body that exists solely to
# raise.


def test_ml_fetch_table_features_no_longer_exists(test_ml) -> None:
    """The retired shim is gone; Python raises AttributeError."""
    with pytest.raises(AttributeError, match=r"fetch_table_features"):
        test_ml.fetch_table_features("Image")


def test_ml_list_feature_values_no_longer_exists(test_ml) -> None:
    """The retired shim is gone; Python raises AttributeError."""
    with pytest.raises(AttributeError, match=r"list_feature_values"):
        test_ml.list_feature_values("Image", "x")


def test_ml_select_by_workflow_no_longer_exists(test_ml) -> None:
    """The retired shim is gone; Python raises AttributeError.

    Replacement is the classmethod factory
    ``FeatureRecord.select_by_workflow(workflow, container=ml)``.
    """
    with pytest.raises(AttributeError, match=r"select_by_workflow"):
        test_ml.select_by_workflow([], "wf")


# Bag-side retirements — use the materialized_bag_with_feature fixture from
# tests/dataset/conftest.py (visible via pytest conftest discovery).
#
# These mirror the ml-side cleanup: the two ``DatasetBag`` tombstones
# (``fetch_table_features``, ``list_feature_values``) were deleted outright
# in the dataset/ P1 sweep. Callers now see a clean ``AttributeError``;
# the replacement guidance lives in the changelog, not in a method body
# that exists solely to raise.


def test_bag_fetch_table_features_no_longer_exists(materialized_bag_with_feature) -> None:
    """The retired shim is gone; Python raises AttributeError."""
    bag = materialized_bag_with_feature.bag
    with pytest.raises(AttributeError, match=r"fetch_table_features"):
        bag.fetch_table_features("Image")


def test_bag_list_feature_values_no_longer_exists(materialized_bag_with_feature) -> None:
    """The retired shim is gone; Python raises AttributeError."""
    bag = materialized_bag_with_feature.bag
    with pytest.raises(AttributeError, match=r"list_feature_values"):
        bag.list_feature_values("Image", "x")
