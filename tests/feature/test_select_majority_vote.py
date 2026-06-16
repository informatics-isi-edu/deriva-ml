"""Tests for FeatureRecord.select_majority_vote selector factory.

Unit tests; no live catalog required.

Closes audit Phase 3 feature/ §3.1.
"""

from __future__ import annotations

import pytest

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.feature import FeatureRecord


def _make_record(execution: str, rct: str, label: str, feature_name: str = "Diagnosis") -> FeatureRecord:
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


# ---------------------------------------------------------------------------
# Regression: column=None auto-detect on a single-term feature (was broken)
# ---------------------------------------------------------------------------
#
# Earlier versions of ``select_majority_vote`` did
# ``record_cls.feature.term_columns[0].name`` to auto-detect the
# single term column. ``term_columns`` is a ``set[Column]`` (see
# ``Feature.__init__``), and Python sets aren't indexable: the
# call crashed with ``TypeError: 'set' object is not subscriptable``
# the moment a real caller (single-term feature, ``column=None``)
# hit the happy path. The bug went undetected because the
# pre-existing tests above all pass ``column=...`` explicitly or
# hit the no-metadata error branch. The two tests below close that
# gap.


class _FakeColumn:
    """Stand-in for ``deriva.core.ermrest_model.Column`` — only
    ``.name`` is read by the selector under test.
    """

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeFeature:
    """Stand-in for ``Feature`` — only ``.term_columns`` is read."""

    def __init__(self, term_column_names: list[str]) -> None:
        # Matches Feature.__init__: a set, not a list.
        self.term_columns: set[_FakeColumn] = {_FakeColumn(n) for n in term_column_names}


def test_select_majority_vote_auto_detect_single_term_column() -> None:
    """``column=None`` works when the record class's feature has one term column.

    Regression test for the
    ``term_columns[0]``-on-a-set TypeError. Before the fix, this
    test failed with::

        TypeError: 'set' object is not subscriptable

    After the fix, the selector pulls the single column out via
    ``next(iter(term_columns))`` and votes correctly.
    """

    # Bare subclass; we attach ``feature`` as a class attribute
    # AFTER class creation so Pydantic doesn't pull it into the
    # field machinery. The selector under test reads via
    # ``hasattr(record_cls, "feature")`` + attribute access, which
    # picks up class attributes the same as ClassVars.
    class SingleTermRecord(FeatureRecord):
        """Mimics what ``Feature.feature_record_class()`` returns
        for a single-term feature."""

    SingleTermRecord.feature = _FakeFeature(term_column_names=["Diagnosis"])

    def _make(execution: str, rct: str, label: str) -> "SingleTermRecord":
        rec = SingleTermRecord(Execution=execution, Feature_Name="Diagnosis", RCT=rct)
        object.__setattr__(rec, "Diagnosis", label)
        return rec

    selector = FeatureRecord.select_majority_vote()  # column=None
    records = [
        _make("e1", "2024-01-01T00:00:00", "benign"),
        _make("e2", "2024-01-02T00:00:00", "benign"),
        _make("e3", "2024-01-03T00:00:00", "malignant"),
    ]
    result = selector(records)
    assert result.Diagnosis == "benign"


def test_select_majority_vote_auto_detect_rejects_multi_term_feature() -> None:
    """``column=None`` on a multi-term feature raises with a clear message.

    The error message lists the available column names so the
    caller knows what to pass. Sorted for deterministic output
    across set-iteration orders.
    """

    class MultiTermRecord(FeatureRecord):
        pass

    MultiTermRecord.feature = _FakeFeature(term_column_names=["Diagnosis", "Severity"])

    rec = MultiTermRecord(Execution="e1", Feature_Name="Diagnosis", RCT="2024-01-01T00:00:00")
    object.__setattr__(rec, "Diagnosis", "benign")

    selector = FeatureRecord.select_majority_vote()  # column=None
    with pytest.raises(DerivaMLException, match=r"multiple term columns") as exc_info:
        selector([rec])
    # Deterministic available-list in the message (sorted).
    msg = str(exc_info.value)
    assert "['Diagnosis', 'Severity']" in msg, f"Expected sorted column list in error; got: {msg}"


def test_select_majority_vote_returns_none_on_empty_records() -> None:
    """Empty ``records`` → ``None``, matching the selector convention.

    Pre-fix (audit F-2) the inner ``records[0]`` access raised
    ``IndexError`` on empty input; ``select_by_workflow`` and
    ``select_by_execution`` already returned ``None`` in the
    same situation. The fix aligned ``select_majority_vote``
    with the convention; ``feature_values`` (and friends)
    drop ``None`` survivors during group reduction, so a
    feature with no records for an asset is silently skipped
    rather than crashing the whole iteration.

    Audit F-20 #3 — explicit empty-records test.
    """
    selector = FeatureRecord.select_majority_vote()  # column=None
    # Pass an empty list with no record-class context — the
    # auto-detect branch must not run.
    result = selector([])
    assert result is None

    # Same behavior with an explicit column — pinning the
    # short-circuit so a future "split the column-vs-no-column
    # branches" refactor doesn't lose the empty guard.
    selector_with_col = FeatureRecord.select_majority_vote(column="Anything")
    assert selector_with_col([]) is None
