"""Tests for FeatureRecord.select_by_execution selector factory.

Unit tests; no live catalog required.

Closes audit Phase 3 feature/ §3.2.
"""

from __future__ import annotations

import pytest

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.feature import FeatureRecord


def _make_record(
    execution: str, rct: str, feature_name: str = "Label"
) -> FeatureRecord:
    """Create a minimal FeatureRecord with the given execution and RCT."""
    return FeatureRecord(Execution=execution, Feature_Name=feature_name, RCT=rct)


def test_select_by_execution_picks_matching_record() -> None:
    """The selector returns the record whose Execution matches."""
    selector = FeatureRecord.select_by_execution("exec-target")
    records = [
        _make_record("exec-other", "2024-01-01T00:00:00"),
        _make_record("exec-target", "2024-01-02T00:00:00"),
    ]
    result = selector(records)
    assert result.Execution == "exec-target"


def test_select_by_execution_picks_newest_among_matches() -> None:
    """When multiple records share the execution RID, newest by RCT wins.

    Mirrors the contract documented in the docstring: ties broken
    by select_newest semantics.
    """
    selector = FeatureRecord.select_by_execution("exec-multi")
    records = [
        _make_record("exec-multi", "2024-01-01T00:00:00"),
        _make_record("exec-multi", "2024-06-15T12:00:00"),  # newer
        _make_record("exec-other", "2025-01-01T00:00:00"),  # newer but wrong exec
    ]
    result = selector(records)
    assert result.Execution == "exec-multi"
    assert result.RCT == "2024-06-15T12:00:00"


def test_select_by_execution_raises_when_no_match() -> None:
    """No record with the requested Execution → DerivaMLException."""
    selector = FeatureRecord.select_by_execution("exec-missing")
    records = [
        _make_record("exec-a", "2024-01-01T00:00:00"),
        _make_record("exec-b", "2024-01-02T00:00:00"),
    ]
    with pytest.raises(DerivaMLException, match="No feature records match execution"):
        selector(records)
