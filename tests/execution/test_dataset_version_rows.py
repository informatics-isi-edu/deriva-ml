"""Unit tests for ExecutionMixin._dataset_version_rows ordering (issue #367).

Offline: the pathBuilder is a dict-backed mock. RIDs are generated
programmatically by _rid() — never hand-written literals — and assertions
compare only values that flowed through the helper.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from deriva_ml.core.mixins.execution import ExecutionMixin, _version_row_sort_key


def _rid(n: int) -> str:
    """Programmatic RID-shaped string (pattern [A-Z\\d]{1,4} segments)."""
    return f"1-{n:04X}"


def _mixin_with_version_rows(rows: list[dict]):
    """ExecutionMixin whose Dataset_Version fetch returns `rows`."""
    ml = ExecutionMixin.__new__(ExecutionMixin)
    ml.ml_schema = "deriva-ml"
    version_path = MagicMock()
    version_path.filter.return_value.entities.return_value.fetch.return_value = list(rows)
    schema = MagicMock()
    schema.tables = {"Dataset_Version": version_path}
    pb = MagicMock()
    pb.schemas = {"deriva-ml": schema}
    ml.pathBuilder = lambda: pb
    return ml


def _row(rct: str, version: str, exec_n: int | None, ver_rid: int = 0) -> dict:
    return {
        "RID": _rid(1000 + ver_rid),
        "RCT": rct,
        "Version": version,
        "Execution": _rid(exec_n) if exec_n is not None else None,
        "Description": f"notes for {version}",
    }


def test_rct_primary_beats_version_label():
    """A later-created row with a LOWER version label sorts after."""
    early = _row("2025-01-01T00:00:00Z", "1.0.0", 1, ver_rid=1)
    late_low_label = _row("2026-01-01T00:00:00Z", "0.9.0", 2, ver_rid=2)
    ml = _mixin_with_version_rows([late_low_label, early])
    out = ml._dataset_version_rows(_rid(500))
    assert [r["RID"] for r in out] == [early["RID"], late_low_label["RID"]]


def test_rct_tie_falls_back_to_pep440_label():
    t = "2025-01-01T00:00:00Z"
    v2 = _row(t, "0.2.0", 2, ver_rid=2)
    v10 = _row(t, "0.10.0", 3, ver_rid=3)
    ml = _mixin_with_version_rows([v10, v2])
    out = ml._dataset_version_rows(_rid(500))
    assert [r["Version"] for r in out] == ["0.2.0", "0.10.0"]  # numeric, not lexical


def test_unparseable_labels_sort_after_parseable_within_rct_tie():
    t = "2025-01-01T00:00:00Z"
    good = _row(t, "0.1.0", 1, ver_rid=1)
    bad = _row(t, "not-a-version", 2, ver_rid=2)
    ml = _mixin_with_version_rows([bad, good])
    out = ml._dataset_version_rows(_rid(500))
    assert out[0]["Version"] == "0.1.0"


def test_two_unparseable_labels_fall_back_to_raw_string_order():
    t = "2025-01-01T00:00:00Z"
    a = _row(t, "aaa-bad", 1, ver_rid=1)
    b = _row(t, "bbb-bad", 2, ver_rid=2)
    ml = _mixin_with_version_rows([b, a])
    out = ml._dataset_version_rows(_rid(500))
    assert [r["Version"] for r in out] == ["aaa-bad", "bbb-bad"]


def test_normalize_equal_labels_resolve_by_raw_string():
    """'1.0' and '1.0.0' parse equal — raw string breaks the tie, no crash."""
    t = "2025-01-01T00:00:00Z"
    short = _row(t, "1.0", 1, ver_rid=1)
    long = _row(t, "1.0.0", 2, ver_rid=2)
    ml = _mixin_with_version_rows([long, short])
    out = ml._dataset_version_rows(_rid(500))
    assert [r["Version"] for r in out] == ["1.0", "1.0.0"]


def test_missing_rct_sorts_first():
    """A row with no usable RCT is treated as epoch-minimum."""
    no_rct = {"RID": _rid(1001), "RCT": None, "Version": "5.0.0", "Execution": None, "Description": None}
    dated = _row("2025-01-01T00:00:00Z", "0.1.0", 1, ver_rid=2)
    ml = _mixin_with_version_rows([dated, no_rct])
    out = ml._dataset_version_rows(_rid(500))
    assert out[0]["RID"] == no_rct["RID"]


def test_single_fetch_per_call():
    ml = _mixin_with_version_rows([_row("2025-01-01T00:00:00Z", "0.1.0", 1)])
    pb = ml.pathBuilder()
    ml._dataset_version_rows(_rid(500))
    assert pb.schemas["deriva-ml"].tables["Dataset_Version"].filter.call_count == 1


def test_sort_key_is_module_level_and_total():
    """Direct key checks: flag separates parseable from not at equal RCT."""
    t = "2025-01-01T00:00:00Z"
    k_good = _version_row_sort_key({"RCT": t, "Version": "1.0.0"})
    k_bad = _version_row_sort_key({"RCT": t, "Version": "junk"})
    assert k_good < k_bad
