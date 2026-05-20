"""Tests for FeatureRecord.select_by_execution selector factory.

Unit tests; no live catalog required.

Closes audit Phase 3 feature/ §3.2.
"""

from __future__ import annotations

from deriva_ml.feature import FeatureRecord


def _make_record(execution: str, rct: str, feature_name: str = "Label") -> FeatureRecord:
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
    assert result is not None
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
    assert result is not None
    assert result.Execution == "exec-multi"
    assert result.RCT == "2024-06-15T12:00:00"


def test_select_by_execution_returns_none_when_no_match() -> None:
    """No record with the requested Execution → returns None.

    Mirrors ``select_by_workflow``'s behavior. ``feature_values``
    calls the selector once per target row; targets whose record
    group lacks a match should be silently omitted from the
    output, not abort the whole query. The earlier contract
    (raise ``DerivaMLException``) was retired in 2026-05 because
    it made the selector unusable in any catalog where a feature
    is sparsely annotated by execution — the common case.
    """
    selector = FeatureRecord.select_by_execution("exec-missing")
    records = [
        _make_record("exec-a", "2024-01-01T00:00:00"),
        _make_record("exec-b", "2024-01-02T00:00:00"),
    ]
    assert selector(records) is None


def test_select_by_execution_returns_none_on_empty_records() -> None:
    """selector([]) returns None — no record group can satisfy an execution filter.

    Parallel to ``select_by_workflow``'s empty-records test.
    """
    selector = FeatureRecord.select_by_execution("exec-anything")
    assert selector([]) is None


def test_select_by_execution_skips_unrelated_targets_in_mixed_groups() -> None:
    """Simulates the feature_values per-target loop with mixed groups.

    feature_values calls the selector once per target. Groups that
    happen to contain a matching record return that record;
    groups that don't return None and are silently skipped by the
    caller. This is the e2e shape that surfaced B12.
    """
    selector = FeatureRecord.select_by_execution("exec-target")
    # Group A: has the target execution + others (yields the target).
    group_a = [
        _make_record("exec-other-1", "2024-01-01T00:00:00"),
        _make_record("exec-target", "2024-02-01T00:00:00"),
        _make_record("exec-other-2", "2024-03-01T00:00:00"),
    ]
    # Group B: only other executions (yields None — skipped).
    group_b = [
        _make_record("exec-other-1", "2024-01-01T00:00:00"),
        _make_record("exec-other-2", "2024-02-01T00:00:00"),
    ]
    a = selector(group_a)
    b = selector(group_b)
    assert a is not None and a.Execution == "exec-target"
    assert b is None
