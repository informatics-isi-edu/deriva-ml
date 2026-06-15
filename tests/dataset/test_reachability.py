"""Unit tests for the client-side FK-reachability engine (no live catalog)."""

from types import SimpleNamespace

from deriva_ml.dataset._reachability import _fk_join_columns


def _col(name):
    return SimpleNamespace(name=name)


def _make_model():
    """Minimal model: Child.parent_fk -> Parent.RID (FK lives on Child)."""
    parent = SimpleNamespace(foreign_keys=[], referenced_by=[])
    fk = SimpleNamespace(
        foreign_key_columns=[_col("parent_fk")],
        referenced_columns=[_col("RID")],
    )
    child = SimpleNamespace(foreign_keys=[fk], referenced_by=[])
    fk.pk_table = parent
    parent.referenced_by = [fk]
    schemas = {"S": SimpleNamespace(tables={"Parent": parent, "Child": child})}
    return SimpleNamespace(schemas=schemas)


def test_fk_join_columns_fk_on_child():
    model = _make_model()
    # Following Parent -> Child: the FK lives on Child (cur).
    constraints = _fk_join_columns(("S", "Parent"), ("S", "Child"), model)
    assert constraints == [{"fk_on": "cur", "pairs": [("RID", "parent_fk")]}]


def test_fk_join_columns_fk_on_parent():
    model = _make_model()
    # Following Child -> Parent: the FK lives on Child (prev).
    constraints = _fk_join_columns(("S", "Child"), ("S", "Parent"), model)
    assert constraints == [{"fk_on": "prev", "pairs": [("parent_fk", "RID")]}]


def test_fk_join_columns_no_edge_returns_empty():
    model = _make_model()
    # Parent has no FK to itself.
    assert _fk_join_columns(("S", "Parent"), ("S", "Parent"), model) == []
