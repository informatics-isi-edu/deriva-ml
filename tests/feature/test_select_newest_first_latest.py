"""Tests for the pure-Python ``select_newest`` / ``select_first`` /
``select_latest`` selectors on FeatureRecord.

Unit tests; no live catalog required.

Closes audit Phase 3 feature/ §3.3.
"""

from __future__ import annotations

from deriva_ml.feature import FeatureRecord


def _make_record(execution: str, rct: str) -> FeatureRecord:
    """Create a minimal FeatureRecord with the given execution and RCT."""
    return FeatureRecord(Execution=execution, Feature_Name="Label", RCT=rct)


# ---------------------------------------------------------------------------
# select_newest
# ---------------------------------------------------------------------------


def test_select_newest_picks_max_rct() -> None:
    """``select_newest`` returns the record with the lexicographically max RCT."""
    records = [
        _make_record("e1", "2024-01-01T00:00:00"),
        _make_record("e2", "2024-06-15T12:00:00"),  # newest
        _make_record("e3", "2024-03-01T00:00:00"),
    ]
    result = FeatureRecord.select_newest(records)
    assert result.Execution == "e2"


def test_select_newest_treats_none_rct_as_oldest() -> None:
    """A record with ``RCT=None`` is treated as older than any timestamped one.

    The selector uses ``r.RCT or ""`` for the comparison key,
    so None coerces to the empty string which sorts before any
    ISO 8601 timestamp.
    """
    records = [
        _make_record("e1", "2024-01-01T00:00:00"),
        FeatureRecord(Execution="e2", Feature_Name="Label", RCT=None),
    ]
    result = FeatureRecord.select_newest(records)
    assert result.Execution == "e1"


def test_select_newest_single_record() -> None:
    """One-record input → that record."""
    records = [_make_record("e1", "2024-01-01T00:00:00")]
    assert FeatureRecord.select_newest(records).Execution == "e1"


# ---------------------------------------------------------------------------
# select_first
# ---------------------------------------------------------------------------


def test_select_first_picks_min_rct() -> None:
    """``select_first`` returns the record with the lexicographically min RCT."""
    records = [
        _make_record("e1", "2024-06-15T12:00:00"),
        _make_record("e2", "2024-01-01T00:00:00"),  # oldest
        _make_record("e3", "2024-03-01T00:00:00"),
    ]
    result = FeatureRecord.select_first(records)
    assert result.Execution == "e2"


def test_select_first_treats_none_rct_as_oldest() -> None:
    """A record with ``RCT=None`` wins ``select_first`` over any timestamp.

    Same coercion as select_newest, just on the other end of the sort.
    """
    records = [
        _make_record("e1", "2024-01-01T00:00:00"),
        FeatureRecord(Execution="e2", Feature_Name="Label", RCT=None),
    ]
    result = FeatureRecord.select_first(records)
    assert result.Execution == "e2"


# ---------------------------------------------------------------------------
# select_latest — alias for select_newest
# ---------------------------------------------------------------------------


def test_select_latest_is_alias_of_select_newest() -> None:
    """``select_latest`` delegates to ``select_newest`` (documented contract)."""
    records = [
        _make_record("e1", "2024-01-01T00:00:00"),
        _make_record("e2", "2024-06-15T12:00:00"),
    ]
    assert FeatureRecord.select_latest(records) is FeatureRecord.select_newest(records)


def test_selectors_are_callable_factories() -> None:
    """All three selectors are usable directly as ``selector=`` arguments.

    Verifies the bare-function (no factory call) shape that
    ``feature_values(selector=FeatureRecord.select_newest)`` relies on.
    """
    assert callable(FeatureRecord.select_newest)
    assert callable(FeatureRecord.select_first)
    assert callable(FeatureRecord.select_latest)
