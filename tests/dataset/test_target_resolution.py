"""Unit tests for the shared target-resolution helper.

Framework-agnostic: tests the selector/missing/arity logic in isolation
from torch, tensorflow, and the restructure_assets call-site. If these
tests fail, the adapter and restructure alignment both break in the
same way, which is a feature — one helper, one set of semantics.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.target_resolution import _resolve_targets
from deriva_ml.feature import FeatureRecord


class _FakeRecord(FeatureRecord):
    """Minimal FeatureRecord stand-in for target-resolution tests."""
    Image: str  # target column: asset RID this record describes
    Grade: str  # scalar label
    Feature_Name: str = "Grade"  # default so tests don't repeat it


def _make_bag(feature_returns: dict[str, dict[str, list[_FakeRecord]]]):
    """Build a MagicMock bag whose feature_values returns canned data.

    Args:
        feature_returns: {feature_name: {element_rid: [records...]}}.

    Returns:
        A MagicMock standing in for a DatasetBag.
    """
    bag = MagicMock()

    def fake_feature_values(element_type, feature_name, selector=None):
        per_rid = feature_returns.get(feature_name, {})
        for rid, records in per_rid.items():
            if selector is None:
                yield from records
            else:
                selected = selector(records)
                if selected is not None:
                    yield selected

    bag.feature_values = fake_feature_values
    bag.list_dataset_members = MagicMock(
        return_value={"Image": [{"RID": rid} for rid in feature_returns.get(
            next(iter(feature_returns), "Grade"), {}).keys()]}
    )
    return bag


def test_resolve_targets_none_returns_unlabeled_per_rid():
    """targets=None → empty dict (no label resolution needed)."""
    bag = _make_bag({})
    result = _resolve_targets(bag, "Image", targets=None, missing="error")
    assert result == {}


def test_resolve_targets_single_target_returns_featurerecord_per_rid():
    """Single-target list yields one FeatureRecord per element RID."""
    recs_a = [_FakeRecord(Image="1-IMG1", Grade="Mild")]
    recs_b = [_FakeRecord(Image="1-IMG2", Grade="Severe")]
    bag = _make_bag({"Grade": {"1-IMG1": recs_a, "1-IMG2": recs_b}})
    result = _resolve_targets(bag, "Image", targets=["Grade"], missing="error")
    assert result["1-IMG1"].Grade == "Mild"
    assert result["1-IMG2"].Grade == "Severe"


def test_resolve_targets_missing_error_raises_with_rid_list():
    """missing='error' with sparse labels raises and names unlabeled RIDs."""
    recs = [_FakeRecord(Image="1-IMG1", Grade="Mild")]
    bag = _make_bag({"Grade": {"1-IMG1": recs, "1-IMG2": []}})
    with pytest.raises(DerivaMLException, match=r"1-IMG2"):
        _resolve_targets(bag, "Image", targets=["Grade"], missing="error")


def test_resolve_targets_missing_skip_drops_unlabeled():
    """missing='skip' omits the unlabeled RID from the result entirely."""
    recs = [_FakeRecord(Image="1-IMG1", Grade="Mild")]
    bag = _make_bag({"Grade": {"1-IMG1": recs, "1-IMG2": []}})
    result = _resolve_targets(bag, "Image", targets=["Grade"], missing="skip")
    assert "1-IMG1" in result
    assert "1-IMG2" not in result


def test_resolve_targets_missing_unknown_yields_none():
    """missing='unknown' keeps the RID with None as its target value."""
    recs = [_FakeRecord(Image="1-IMG1", Grade="Mild")]
    bag = _make_bag({"Grade": {"1-IMG1": recs, "1-IMG2": []}})
    result = _resolve_targets(bag, "Image", targets=["Grade"], missing="unknown")
    assert result["1-IMG1"].Grade == "Mild"
    assert result["1-IMG2"] is None


def test_resolve_targets_multi_target_returns_dict_keyed_by_feature():
    """Multi-target yields dict[feature_name, FeatureRecord] per RID."""
    grade_recs = [_FakeRecord(Image="1-IMG1", Grade="Mild")]
    severity_recs = [_FakeRecord(Image="1-IMG1", Grade="Low")]
    bag = _make_bag({
        "Grade": {"1-IMG1": grade_recs},
        "Severity": {"1-IMG1": severity_recs},
    })
    result = _resolve_targets(
        bag, "Image", targets=["Grade", "Severity"], missing="error"
    )
    assert set(result["1-IMG1"].keys()) == {"Grade", "Severity"}
    assert result["1-IMG1"]["Grade"].Grade == "Mild"


def test_resolve_targets_selector_dict_applies_per_feature():
    """dict form passes selectors per-feature to bag.feature_values."""
    selector = FeatureRecord.select_newest
    r1 = _FakeRecord(Image="1-IMG1", Grade="Mild")
    r2 = _FakeRecord(Image="1-IMG1", Grade="Severe")
    bag = _make_bag({"Grade": {"1-IMG1": [r1, r2]}})
    result = _resolve_targets(
        bag, "Image", targets={"Grade": selector}, missing="error"
    )
    # The selector picks one; exact choice depends on select_newest's
    # impl, but the result should be one of the two records.
    assert result["1-IMG1"].Grade in ("Mild", "Severe")


def test_resolve_targets_unknown_feature_raises_at_construction():
    """Passing a feature name that doesn't exist raises the same error
    bag.feature_values would raise."""
    bag = MagicMock()
    bag.feature_values = MagicMock(side_effect=DerivaMLException("No such feature"))
    bag.list_dataset_members = MagicMock(return_value={"Image": []})
    with pytest.raises(DerivaMLException):
        _resolve_targets(bag, "Image", targets=["Bogus"], missing="error")
