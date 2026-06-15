"""Unit tests for the client-side FK-reachability engine (no live catalog)."""

from types import SimpleNamespace

from deriva_ml.dataset._reachability import _fk_join_columns, _needed_columns


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


def _make_vocab_model():
    """Asset_Role(Name) <- Child.role_fk -> Asset_Role.Name (vocab FK to Name)."""
    role = SimpleNamespace(foreign_keys=[], column_definitions=SimpleNamespace(elements={}))
    vocab_fk = SimpleNamespace(
        foreign_key_columns=[_col("role_fk")],
        referenced_columns=[_col("Name")],  # references Name, NOT RID
    )
    child = SimpleNamespace(
        foreign_keys=[vocab_fk],
        referenced_by=[],
        column_definitions=SimpleNamespace(elements={}),
    )
    vocab_fk.pk_table = role
    role.referenced_by = [vocab_fk]
    role.foreign_keys = []
    child.is_asset = lambda: False
    role.is_asset = lambda: False
    schemas = {"S": SimpleNamespace(tables={"Asset_Role": role, "Child": child})}
    return SimpleNamespace(schemas=schemas)


def test_needed_columns_includes_inbound_referenced_name():
    model = _make_vocab_model()
    # Asset_Role is the FK TARGET of an inbound vocab FK on Name. The engine
    # must project Name or the in-memory join silently drops rows.
    cols = _needed_columns(("S", "Asset_Role"), model)
    assert "Name" in cols
    assert "RID" in cols


def test_needed_columns_includes_outbound_fk_and_length():
    """Asset table with an outbound FK and a Length column."""
    length_col = SimpleNamespace(name="Length")
    fk = SimpleNamespace(foreign_key_columns=[_col("parent_fk")], referenced_columns=[_col("RID")])
    asset = SimpleNamespace(
        foreign_keys=[fk],
        referenced_by=[],
        column_definitions=SimpleNamespace(elements={"Length": length_col}),
    )
    asset.is_asset = lambda: True
    model = SimpleNamespace(schemas={"S": SimpleNamespace(tables={"A": asset})})
    cols = _needed_columns(("S", "A"), model)
    assert cols == {"RID", "parent_fk", "Length"}
