"""Tests for DatasetCollection — RID-keyed mapping + iterable of
DatasetBags accessible via exe.datasets."""

from __future__ import annotations

import pytest


class _FakeBag:
    """Stand-in for DatasetBag for isolated unit tests."""
    def __init__(self, rid: str):
        self.dataset_rid = rid


def test_collection_rid_lookup():
    from deriva_ml.execution.dataset_collection import DatasetCollection

    bag_a = _FakeBag("1-AAA")
    bag_b = _FakeBag("1-BBB")
    coll = DatasetCollection([bag_a, bag_b])

    assert coll["1-AAA"] is bag_a
    assert coll["1-BBB"] is bag_b


def test_collection_missing_key_raises():
    from deriva_ml.execution.dataset_collection import DatasetCollection

    coll = DatasetCollection([_FakeBag("1-AAA")])
    with pytest.raises(KeyError) as exc:
        _ = coll["NOPE"]
    # Error message lists what IS available.
    assert "1-AAA" in str(exc.value)


def test_collection_iteration_yields_bags():
    from deriva_ml.execution.dataset_collection import DatasetCollection

    bag_a = _FakeBag("1-AAA")
    bag_b = _FakeBag("1-BBB")
    coll = DatasetCollection([bag_a, bag_b])

    assert list(coll) == [bag_a, bag_b]


def test_collection_len():
    from deriva_ml.execution.dataset_collection import DatasetCollection

    assert len(DatasetCollection([])) == 0
    assert len(DatasetCollection([_FakeBag("1")])) == 1
    assert len(DatasetCollection([_FakeBag("1"), _FakeBag("2")])) == 2


def test_collection_contains():
    from deriva_ml.execution.dataset_collection import DatasetCollection

    coll = DatasetCollection([_FakeBag("1-AAA")])
    assert "1-AAA" in coll
    assert "NOPE" not in coll


def test_collection_keys_values():
    from deriva_ml.execution.dataset_collection import DatasetCollection

    bags = [_FakeBag("A"), _FakeBag("B")]
    coll = DatasetCollection(bags)
    assert list(coll.keys()) == ["A", "B"]
    assert list(coll.values()) == bags
