"""Unit tests for the client-side FK-reachability engine (no live catalog)."""

from types import SimpleNamespace

from deriva_ml.dataset._reachability import _fk_join_columns, _needed_columns, _reached_rids_for_path


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


def test_reached_rids_single_hop_fk_on_cur():
    """Anchor Parent={p1}; Child rows c1,c2 point to p1, c3 to p2.
    Following Parent -> Child should reach {c1, c2}."""
    model = _make_model()  # Child.parent_fk -> Parent.RID
    fetched = {
        ("S", "Parent"): [{"RID": "p1"}, {"RID": "p2"}],
        ("S", "Child"): [
            {"RID": "c1", "parent_fk": "p1"},
            {"RID": "c2", "parent_fk": "p1"},
            {"RID": "c3", "parent_fk": "p2"},
        ],
    }
    fk_path = (("S", "Parent"), ("S", "Child"))
    result = _reached_rids_for_path(fk_path, anchor_rids={"p1"}, fetched_rows=fetched, model=model)
    assert result == {"c1", "c2"}


def test_reached_rids_anchor_table_only():
    """A length-1 path (the anchor table itself) returns anchor RIDs that exist."""
    model = _make_model()
    fetched = {("S", "Parent"): [{"RID": "p1"}, {"RID": "p2"}, {"RID": "p3"}]}
    result = _reached_rids_for_path(
        (("S", "Parent"),), anchor_rids={"p1", "p2", "p99"}, fetched_rows=fetched, model=model
    )
    assert result == {"p1", "p2"}  # p99 not in the table


def test_reached_rids_no_fk_returns_none():
    """A path with no FK between adjacent segments is unfollowable -> None."""
    model = _make_model()
    fetched = {("S", "Parent"): [{"RID": "p1"}], ("S", "Parent2"): []}
    # Parent -> Parent (no self-FK in the synthetic model)
    result = _reached_rids_for_path(
        (("S", "Parent"), ("S", "Parent")), anchor_rids={"p1"}, fetched_rows=fetched, model=model
    )
    assert result is None


def _make_composite_model(fk_on):
    """Composite 2-col FK between Order and Region.

    fk_on="prev": FK held on Order (the prev/child in an Order->Region hop),
        Order.(o_region, o_code) -> Region.(r_region, r_code).
    fk_on="cur": FK held on Order (the cur/child in a Region->Order hop) — same
        physical FK, walked the other direction.
    Non-RID referenced columns so the GENERAL (not fast) branch is exercised.
    """
    region = SimpleNamespace(foreign_keys=[], referenced_by=[])
    fk = SimpleNamespace(
        foreign_key_columns=[_col("o_region"), _col("o_code")],
        referenced_columns=[_col("r_region"), _col("r_code")],
    )
    order = SimpleNamespace(foreign_keys=[fk], referenced_by=[])
    fk.pk_table = region
    region.referenced_by = [fk]
    schemas = {"S": SimpleNamespace(tables={"Region": region, "Order": order})}
    return SimpleNamespace(schemas=schemas)


def test_reached_rids_composite_fk_on_prev_no_overcount():
    """Composite FK held on prev (Order). Following Order -> Region must match
    the FULL (region, code) tuple, not just the first column — else a Region
    sharing only the first column is wrongly included."""
    model = _make_composite_model(fk_on="prev")
    fetched = {
        ("S", "Order"): [{"RID": "o1", "o_region": "US", "o_code": "A"}],
        ("S", "Region"): [
            {"RID": "r1", "r_region": "US", "r_code": "A"},  # full match
            {"RID": "r2", "r_region": "US", "r_code": "B"},  # shares region only
        ],
    }
    # Order -> Region: FK lives on Order (prev), references Region non-RID cols.
    result = _reached_rids_for_path(
        (("S", "Order"), ("S", "Region")),
        anchor_rids={"o1"},
        fetched_rows=fetched,
        model=model,
    )
    assert result == {"r1"}  # NOT {"r1", "r2"}


def test_reached_rids_composite_fk_on_cur_no_overcount():
    """Composite FK held on cur (Order). Following Region -> Order must match
    the full tuple, not just the first column."""
    model = _make_composite_model(fk_on="cur")
    fetched = {
        ("S", "Region"): [{"RID": "r1", "r_region": "US", "r_code": "A"}],
        ("S", "Order"): [
            {"RID": "o1", "o_region": "US", "o_code": "A"},  # full match
            {"RID": "o2", "o_region": "US", "o_code": "B"},  # shares region only
        ],
    }
    # Region -> Order: FK lives on Order (cur), references Region non-RID cols.
    result = _reached_rids_for_path(
        (("S", "Region"), ("S", "Order")),
        anchor_rids={"r1"},
        fetched_rows=fetched,
        model=model,
    )
    assert result == {"o1"}  # NOT {"o1", "o2"}
