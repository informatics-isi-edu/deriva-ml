"""Grain-inference robustness on real-catalog FK shapes.

Two concerns surfaced by evaluating the planner against the live eye-ai and
facebase catalogs:

1. **CV-name (non-RID) FK targets.** A controlled-vocabulary term's *name*
   (a non-RID unique key) is a legitimate FK target in Deriva. Grain
   inference must treat such an FK as a real downstream edge, exactly like
   an FK that targets ``RID``. (This already works; the test pins it so a
   future RID-only filter can't silently regress it.)

2. **Disconnected-sink diagnostics.** When two requested tables are not on
   one downstream FK chain *within the requested set*, ``_determine_row_per``
   raises ``DerivaMLDenormalizeMultiLeaf``. The error must name the
   intermediate **bridge table(s)** that would connect them (eye-ai's
   ``["Subject", "Clinical_Records"]`` is the motivating real case), so the
   user knows what to add to ``include_tables`` instead of hitting a dead
   end.

These run catalog-free: a synthetic ``Model`` is built from a hand-authored
schema JSON, so no live catalog is needed.
"""

from __future__ import annotations

import json

import pytest
from deriva.core.ermrest_model import Model

from deriva_ml.core.exceptions import DerivaMLDenormalizeMultiLeaf
from deriva_ml.model.catalog import DerivaModel

# ---------------------------------------------------------------------------
# Synthetic-schema helpers (catalog-free)
# ---------------------------------------------------------------------------

_SYS_COLS = ["RID", "RCT", "RMT", "RCB", "RMB"]


def _col(name: str, typename: str = "text", nullok: bool = True) -> dict:
    return {"name": name, "type": {"typename": typename}, "nullok": nullok}


def _sys_cols() -> list[dict]:
    return [
        _col("RID", "ermrest_rid", nullok=False),
        _col("RCT", "ermrest_rct"),
        _col("RMT", "ermrest_rmt"),
        _col("RCB", "ermrest_rcb"),
        _col("RMB", "ermrest_rmb"),
    ]


def _table(name: str, extra_cols: list[dict], fks: list[dict], extra_keys: list[list[str]] | None = None) -> dict:
    keys = [{"names": [["test", f"{name}_RID_key"]], "unique_columns": ["RID"]}]
    for kcols in extra_keys or []:
        keys.append({"names": [["test", f"{name}_{'_'.join(kcols)}_key"]], "unique_columns": kcols})
    return {
        "kind": "table",
        "schema_name": "test",
        "table_name": name,
        "column_definitions": _sys_cols() + extra_cols,
        "keys": keys,
        "foreign_keys": fks,
    }


def _fk(name: str, from_table: str, from_col: str, to_table: str, to_col: str) -> dict:
    return {
        "names": [["test", name]],
        "foreign_key_columns": [{"schema_name": "test", "table_name": from_table, "column_name": from_col}],
        "referenced_columns": [{"schema_name": "test", "table_name": to_table, "column_name": to_col}],
    }


def _model(tables: dict) -> DerivaModel:
    schema = {"schemas": {"test": {"schema_name": "test", "tables": tables}}}
    m = Model.fromfile("file-system", _write_tmp(schema))
    return DerivaModel(model=m, ml_schema="test", domain_schemas={"test"})


def _write_tmp(obj: dict) -> str:
    import tempfile

    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(obj, f)
    f.flush()
    return f.name


# ---------------------------------------------------------------------------
# 1. CV-name / non-RID FK targets infer grain correctly (pin already-correct)
# ---------------------------------------------------------------------------


def test_cv_name_fk_is_a_downstream_edge():
    """An FK targeting a CV's Name (non-RID unique key) must be a real edge.

    ``Sample.Stage → Stage.Name`` (not ``Stage.RID``) is the controlled-
    vocabulary pattern. The planner's downstream-edge detection
    (``referenced_by``-based) must see it, so ``Stage`` is *not* mistaken
    for a co-sink with ``Sample``.
    """
    tables = {
        "Stage": _table("Stage", [_col("Name", nullok=False)], [], extra_keys=[["Name"]]),
        "Sample": _table(
            "Sample",
            [_col("Label"), _col("Stage")],
            [_fk("Sample_Stage_fk", "Sample", "Stage", "Stage", "Name")],  # FK -> Stage.Name, NOT RID
        ),
    }
    pl = _model(tables)._planner

    # Stage is pointed at by Sample.Stage -> downstream_fk_sources(Stage) == {Sample}
    downstream_of_stage = {t.name for t in pl._downstream_fk_sources("Stage")}
    assert downstream_of_stage == {"Sample"}, (
        f"CV-name FK (Sample.Stage -> Stage.Name) must register as a downstream edge; "
        f"got downstream_fk_sources(Stage)={downstream_of_stage}"
    )

    # Grain inference picks Sample (downstream of Stage), NOT a MultiLeaf.
    row_per = pl._determine_row_per(include_tables=["Stage", "Sample"], via=[], row_per=None)
    assert row_per == "Sample", f"expected row_per=Sample for the CV-name-FK chain, got {row_per!r}"


def test_cv_name_fk_chain_three_tables():
    """A mixed chain — CV-name FK then RID FK — still infers the deepest table."""
    tables = {
        "Stage": _table("Stage", [_col("Name", nullok=False)], [], extra_keys=[["Name"]]),
        "Sample": _table(
            "Sample",
            [_col("Label"), _col("Stage")],
            [_fk("Sample_Stage_fk", "Sample", "Stage", "Stage", "Name")],
        ),
        "Image": _table(
            "Image",
            [_col("File"), _col("Sample")],
            [_fk("Image_Sample_fk", "Image", "Sample", "Sample", "RID")],  # normal RID FK
        ),
    }
    pl = _model(tables)._planner
    row_per = pl._determine_row_per(include_tables=["Stage", "Sample", "Image"], via=[], row_per=None)
    assert row_per == "Image", f"expected row_per=Image (deepest), got {row_per!r}"


# ---------------------------------------------------------------------------
# 2. MultiLeaf must suggest the connecting bridge table(s)
# ---------------------------------------------------------------------------


def test_multileaf_suggests_bridge_table():
    """Two requested tables that are genuine co-sinks (neither downstream of
    the other) but connect through an *unrequested* upstream/association
    bridge must name that bridge in the error.

    Models eye-ai's ``["Subject", "Clinical_Records"]`` exactly:

        Subject ◄── Observation ◄── CR_Observation ──► Clinical_Records

    Both ``Subject`` and ``Clinical_Records`` are sinks of the requested set
    (neither has an FK to the other). They connect only through ``Observation``
    (a common upstream of Subject) and ``CR_Observation`` (an M:N association
    pointing at both Observation and Clinical_Records). The user should be
    told to add ``Observation`` / ``CR_Observation``.
    """
    tables = {
        "Subject": _table("Subject", [_col("Name")], []),
        "Observation": _table(
            "Observation",
            [_col("Date"), _col("Subject")],
            [_fk("Obs_Subject_fk", "Observation", "Subject", "Subject", "RID")],  # Observation -> Subject
        ),
        "Clinical_Records": _table("Clinical_Records", [_col("Diagnosis")], []),
        # M:N association linking Observation and Clinical_Records.
        "CR_Observation": _table(
            "CR_Observation",
            [_col("Observation"), _col("Clinical_Records")],
            [
                _fk("CRO_Obs_fk", "CR_Observation", "Observation", "Observation", "RID"),
                _fk("CRO_CR_fk", "CR_Observation", "Clinical_Records", "Clinical_Records", "RID"),
            ],
        ),
    }
    pl = _model(tables)._planner

    with pytest.raises(DerivaMLDenormalizeMultiLeaf) as exc:
        pl._determine_row_per(include_tables=["Subject", "Clinical_Records"], via=[], row_per=None)

    e = exc.value
    assert set(e.candidates) == {"Subject", "Clinical_Records"}
    # The fix: the error carries the bridge suggestion.
    assert hasattr(e, "bridge_suggestions"), "MultiLeaf must expose bridge_suggestions"
    assert e.bridge_suggestions, f"a connecting bridge should be suggested; got {e.bridge_suggestions!r}"
    # The connecting path runs through Observation and/or CR_Observation.
    assert {"Observation", "CR_Observation"} & set(e.bridge_suggestions), (
        f"bridge should include Observation/CR_Observation; got {e.bridge_suggestions!r}"
    )
    # ...and a suggested table should appear in the message.
    assert any(b in str(e) for b in e.bridge_suggestions), f"bridge table should appear in the message: {e}"


def test_multileaf_no_bridge_when_genuinely_disconnected():
    """Two tables with no connecting path at all: still MultiLeaf, but
    bridge_suggestions is empty (don't fabricate a bridge)."""
    tables = {
        "Alpha": _table("Alpha", [_col("A")], []),
        "Beta": _table("Beta", [_col("B")], []),
    }
    pl = _model(tables)._planner
    with pytest.raises(DerivaMLDenormalizeMultiLeaf) as exc:
        pl._determine_row_per(include_tables=["Alpha", "Beta"], via=[], row_per=None)
    assert exc.value.bridge_suggestions == [], (
        f"no path exists, so no bridge should be invented; got {exc.value.bridge_suggestions!r}"
    )
