"""Tests for FeatureRecord.select_by_workflow selector factory.

Unit tests using a stub container (no live catalog required).
"""

from dataclasses import dataclass, field

import pytest

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.feature import FeatureRecord


# ---------------------------------------------------------------------------
# Stub infrastructure
# ---------------------------------------------------------------------------

KNOWN_WORKFLOWS = {
    "wf-alpha": ["exec-1", "exec-2"],
    "wf-beta": ["exec-3"],
}


class _StubContainer:
    """Minimal container stub implementing list_workflow_executions."""

    def __init__(self):
        self.call_count = 0

    def list_workflow_executions(self, workflow: str) -> list[str]:
        self.call_count += 1
        if workflow not in KNOWN_WORKFLOWS:
            raise DerivaMLException(f"Unknown workflow '{workflow}'")
        return list(KNOWN_WORKFLOWS[workflow])


def _make_record(execution: str, rct: str, feature_name: str = "Label") -> FeatureRecord:
    """Create a minimal FeatureRecord with the given execution and RCT."""
    return FeatureRecord(Execution=execution, Feature_Name=feature_name, RCT=rct)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_container_kwarg_required_positional_raises():
    """Passing container positionally must raise TypeError."""
    container = _StubContainer()
    with pytest.raises(TypeError):
        # container is keyword-only; passing it positionally should fail
        FeatureRecord.select_by_workflow("wf-alpha", container)


def test_unknown_workflow_raises_at_construction():
    """Unknown workflow name raises DerivaMLException at factory-call time, not later."""
    container = _StubContainer()
    with pytest.raises(DerivaMLException, match="Unknown workflow"):
        FeatureRecord.select_by_workflow("wf-nonexistent", container=container)


def test_picks_matching_record():
    """Selector returns the record whose Execution is in the workflow's execution set."""
    container = _StubContainer()
    selector = FeatureRecord.select_by_workflow("wf-beta", container=container)

    records = [
        _make_record("exec-3", "2024-01-01T00:00:00"),
        _make_record("exec-99", "2024-01-02T00:00:00"),  # not in wf-beta
    ]
    result = selector(records)
    assert result is not None
    assert result.Execution == "exec-3"


def test_picks_newest_among_matches():
    """When multiple records match the workflow, select_newest picks the latest RCT."""
    container = _StubContainer()
    selector = FeatureRecord.select_by_workflow("wf-alpha", container=container)

    records = [
        _make_record("exec-1", "2024-01-01T00:00:00"),
        _make_record("exec-2", "2024-06-15T12:00:00"),  # newer, also in wf-alpha
        _make_record("exec-99", "2024-12-31T00:00:00"),  # not in wf-alpha
    ]
    result = selector(records)
    assert result is not None
    assert result.Execution == "exec-2"


def test_returns_none_when_no_match():
    """Selector returns None when no record in the group matches the workflow."""
    container = _StubContainer()
    selector = FeatureRecord.select_by_workflow("wf-beta", container=container)

    records = [
        _make_record("exec-1", "2024-01-01T00:00:00"),
        _make_record("exec-2", "2024-01-02T00:00:00"),
    ]
    result = selector(records)
    assert result is None


def test_select_by_workflow_returns_none_on_empty_records() -> None:
    """selector([]) returns None — no record group can satisfy a workflow filter."""
    container = _StubContainer()
    selector = FeatureRecord.select_by_workflow("wf-alpha", container=container)
    assert selector([]) is None


def test_resolves_executions_once():
    """Container.list_workflow_executions is called exactly once at factory construction."""
    container = _StubContainer()
    assert container.call_count == 0

    selector = FeatureRecord.select_by_workflow("wf-alpha", container=container)
    assert container.call_count == 1  # resolved at construction

    # Multiple selector invocations must not trigger additional lookups
    records = [_make_record("exec-1", "2024-01-01T00:00:00")]
    selector(records)
    selector(records)
    assert container.call_count == 1  # still only 1
