"""Tests for FeatureRecord.select_majority_vote selector factory.

Unit tests; no live catalog required.

Closes audit Phase 3 feature/ §3.1.
"""

from __future__ import annotations

import pytest

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.feature import FeatureRecord


def _make_record(
    execution: str, rct: str, label: str, feature_name: str = "Diagnosis"
) -> FeatureRecord:
    """Create a minimal FeatureRecord with a `Diagnosis` term-style column.

    We attach the label as a dynamic attribute since the base
    FeatureRecord doesn't have a Diagnosis field; the selector
    reads via getattr(r, col, None).
    """
    rec = FeatureRecord(Execution=execution, Feature_Name=feature_name, RCT=rct)
    # The selector reads via getattr; setting via Pydantic's
    # __dict__ is the simplest way without a real subclass.
    object.__setattr__(rec, "Diagnosis", label)
    return rec


def test_select_majority_vote_picks_most_frequent_value() -> None:
    """The most-frequent label wins the vote."""
    selector = FeatureRecord.select_majority_vote(column="Diagnosis")
    records = [
        _make_record("e1", "2024-01-01T00:00:00", "benign"),
        _make_record("e2", "2024-01-02T00:00:00", "benign"),
        _make_record("e3", "2024-01-03T00:00:00", "benign"),
        _make_record("e4", "2024-01-04T00:00:00", "malignant"),
        _make_record("e5", "2024-01-05T00:00:00", "malignant"),
    ]
    result = selector(records)
    assert result.Diagnosis == "benign"


def test_select_majority_vote_breaks_ties_by_newest_rct() -> None:
    """When two values tie, pick the candidate record with the newest RCT.

    Three votes each for 'benign' and 'malignant'. The newest
    record overall is malignant — tie-break by RCT puts that
    record in the result.
    """
    selector = FeatureRecord.select_majority_vote(column="Diagnosis")
    records = [
        _make_record("e1", "2024-01-01T00:00:00", "benign"),
        _make_record("e2", "2024-01-02T00:00:00", "benign"),
        _make_record("e3", "2024-01-03T00:00:00", "benign"),
        _make_record("e4", "2024-06-04T00:00:00", "malignant"),
        _make_record("e5", "2024-06-05T00:00:00", "malignant"),
        _make_record("e6", "2024-12-15T00:00:00", "malignant"),  # newest overall
    ]
    result = selector(records)
    # Tie at 3-3; both label groups are majority candidates.
    # Among the 6 candidate records, the newest is e6 (malignant).
    assert result.Diagnosis == "malignant"
    assert result.Execution == "e6"


def test_select_majority_vote_picks_when_single_dominant() -> None:
    """One value with clear plurality wins regardless of RCT order."""
    selector = FeatureRecord.select_majority_vote(column="Diagnosis")
    records = [
        _make_record("e1", "2025-12-31T23:59:59", "benign"),  # newest, but minority
        _make_record("e2", "2024-01-01T00:00:00", "malignant"),
        _make_record("e3", "2024-01-02T00:00:00", "malignant"),
        _make_record("e4", "2024-01-03T00:00:00", "malignant"),
    ]
    result = selector(records)
    assert result.Diagnosis == "malignant"


def test_select_majority_vote_raises_without_column_when_no_metadata() -> None:
    """Calling with column=None and no feature metadata → useful error.

    The base FeatureRecord doesn't have a feature ClassVar
    populated, so the auto-detect branch fails with a clear
    message rather than crashing on attribute access.
    """
    selector = FeatureRecord.select_majority_vote()  # column=None
    records = [_make_record("e1", "2024-01-01T00:00:00", "x")]
    with pytest.raises(DerivaMLException, match="could not auto-detect"):
        selector(records)


def test_select_majority_vote_with_explicit_column_overrides_auto_detect() -> None:
    """Explicit column kwarg is used regardless of metadata state."""
    selector = FeatureRecord.select_majority_vote(column="Diagnosis")
    records = [
        _make_record("e1", "2024-01-01T00:00:00", "benign"),
        _make_record("e2", "2024-01-02T00:00:00", "benign"),
    ]
    result = selector(records)
    assert result.Diagnosis == "benign"
