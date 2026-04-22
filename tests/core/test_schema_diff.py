"""Unit tests for compute_diff + SchemaDiff rendering."""
from __future__ import annotations

import pytest


def _schema(tables: dict | None = None, schema_name: str = "deriva-ml") -> dict:
    """Minimal ERMrest /schema payload for tests."""
    return {
        "schemas": {
            schema_name: {
                "schema_name": schema_name,
                "tables": tables or {},
            }
        }
    }


def _table(columns=None, fkeys=None, name="T"):
    return {
        "schema_name": "deriva-ml",
        "table_name": name,
        "column_definitions": columns or [],
        "foreign_keys": fkeys or [],
    }


def _col(name, typename="text"):
    return {"name": name, "type": {"typename": typename}}


def _fkey(columns, ref_schema, ref_table, ref_columns):
    return {
        "foreign_key_columns": [
            {"schema_name": "deriva-ml", "table_name": "X", "column_name": c}
            for c in columns
        ],
        "referenced_columns": [
            {"schema_name": ref_schema, "table_name": ref_table, "column_name": c}
            for c in ref_columns
        ],
    }


def test_empty_diff_when_schemas_identical():
    from deriva_ml.core.schema_diff import compute_diff
    s = _schema({"T": _table(columns=[_col("a")])})
    diff = compute_diff(s, s)
    assert diff.is_empty()
    assert diff.added_schemas == []
    assert diff.removed_schemas == []


def test_added_schema():
    from deriva_ml.core.schema_diff import compute_diff
    cached = _schema()
    live = {
        "schemas": {
            "deriva-ml": {"schema_name": "deriva-ml", "tables": {}},
            "newsch":    {"schema_name": "newsch",    "tables": {}},
        }
    }
    diff = compute_diff(cached, live)
    assert diff.added_schemas == ["newsch"]
    assert diff.removed_schemas == []
    assert not diff.is_empty()


def test_removed_schema():
    from deriva_ml.core.schema_diff import compute_diff
    cached = {
        "schemas": {
            "deriva-ml": {"schema_name": "deriva-ml", "tables": {}},
            "gone":      {"schema_name": "gone",      "tables": {}},
        }
    }
    live = _schema()
    diff = compute_diff(cached, live)
    assert diff.removed_schemas == ["gone"]


def test_added_table():
    from deriva_ml.core.schema_diff import compute_diff
    cached = _schema({"T1": _table(name="T1")})
    live = _schema({
        "T1": _table(name="T1"),
        "T2": _table(name="T2"),
    })
    diff = compute_diff(cached, live)
    assert [t.table for t in diff.added_tables] == ["T2"]
    assert all(t.schema == "deriva-ml" for t in diff.added_tables)


def test_removed_table():
    from deriva_ml.core.schema_diff import compute_diff
    cached = _schema({
        "T1": _table(name="T1"),
        "T2": _table(name="T2"),
    })
    live = _schema({"T1": _table(name="T1")})
    diff = compute_diff(cached, live)
    assert [t.table for t in diff.removed_tables] == ["T2"]


def test_added_column():
    from deriva_ml.core.schema_diff import compute_diff
    cached = _schema({"T": _table(columns=[_col("a")])})
    live = _schema({"T": _table(columns=[_col("a"), _col("b", "int4")])})
    diff = compute_diff(cached, live)
    assert len(diff.added_columns) == 1
    add = diff.added_columns[0]
    assert add.schema == "deriva-ml"
    assert add.table == "T"
    assert add.column == "b"
    assert add.type == "int4"


def test_removed_column():
    from deriva_ml.core.schema_diff import compute_diff
    cached = _schema({"T": _table(columns=[_col("a"), _col("b")])})
    live = _schema({"T": _table(columns=[_col("a")])})
    diff = compute_diff(cached, live)
    assert [c.column for c in diff.removed_columns] == ["b"]


def test_column_type_change():
    from deriva_ml.core.schema_diff import compute_diff
    cached = _schema({"T": _table(columns=[_col("a", "text")])})
    live = _schema({"T": _table(columns=[_col("a", "int4")])})
    diff = compute_diff(cached, live)
    assert len(diff.column_type_changes) == 1
    chg = diff.column_type_changes[0]
    assert chg.column == "a"
    assert chg.cached_type == "text"
    assert chg.live_type == "int4"


def test_added_fkey():
    from deriva_ml.core.schema_diff import compute_diff
    cached = _schema({"T": _table(columns=[_col("x")])})
    live = _schema({
        "T": _table(
            columns=[_col("x")],
            fkeys=[_fkey(["x"], "deriva-ml", "Other", ["y"])],
        ),
    })
    diff = compute_diff(cached, live)
    assert len(diff.added_fkeys) == 1
    fk = diff.added_fkeys[0]
    assert fk.columns == ["x"]
    assert fk.referenced_table == "Other"
    assert fk.referenced_columns == ["y"]


def test_removed_fkey():
    from deriva_ml.core.schema_diff import compute_diff
    cached = _schema({
        "T": _table(
            columns=[_col("x")],
            fkeys=[_fkey(["x"], "deriva-ml", "Other", ["y"])],
        ),
    })
    live = _schema({"T": _table(columns=[_col("x")])})
    diff = compute_diff(cached, live)
    assert len(diff.removed_fkeys) == 1


def test_diff_render_produces_human_readable():
    from deriva_ml.core.schema_diff import compute_diff
    cached = _schema({"T": _table(columns=[_col("a")])})
    live = _schema({"T": _table(columns=[_col("a"), _col("b", "int4")])})
    diff = compute_diff(cached, live)
    text = diff.render()
    assert "deriva-ml.T.b" in text
    assert "int4" in text
    assert text == str(diff)
    # Empty diff renders empty-ish, no crash
    empty = compute_diff(cached, cached)
    assert empty.render() == "" or "no changes" in empty.render().lower()


def test_diff_determinism():
    """Two runs over the same inputs produce identical diffs (sorted)."""
    from deriva_ml.core.schema_diff import compute_diff
    cached = _schema({"T": _table(columns=[_col("a")])})
    live = _schema({
        "T": _table(columns=[_col("a"), _col("z"), _col("m"), _col("b")]),
    })
    d1 = compute_diff(cached, live)
    d2 = compute_diff(cached, live)
    assert d1 == d2
    assert [c.column for c in d1.added_columns] == ["b", "m", "z"]
