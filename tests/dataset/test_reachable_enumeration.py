"""resolve_reachable_rows / resolve_element_rids — the shared FK-reachable
element enumeration used by restructure_assets AND the tf/torch adapters.

Catalog-free: the bag's _dataset_table_view + SQLAlchemy Session are stubbed so
the traversal contract is exercised without a live catalog.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.target_resolution import (
    resolve_element_rids,
    resolve_reachable_rows,
)


def _bag_yielding(rows, *, table_exists=True):
    """A bag whose _dataset_table_view + engine produce `rows` (list of dicts).

    resolve_reachable_rows runs ``Session(bag.engine).execute(query).mappings()``
    — we stub _dataset_table_view to return a sentinel and patch the Session in
    the test so .mappings().all() yields `rows`.
    """
    bag = MagicMock()
    bag._dataset_table_view.return_value = "SENTINEL_QUERY"
    if table_exists:
        bag.model.name_to_table.return_value = object()
    else:
        bag.model.name_to_table.side_effect = KeyError("no such table")
    return bag


def _patched_session(rows):
    """Context-manager patch making Session(...).execute(q).mappings().all() == rows."""
    session_cm = MagicMock()
    session = session_cm.__enter__.return_value
    session.execute.return_value.mappings.return_value.all.return_value = rows
    return patch("deriva_ml.dataset.target_resolution.Session", return_value=session_cm)


def test_resolve_reachable_rows_returns_full_rows():
    rows = [{"RID": "i1", "Filename": "a.png"}, {"RID": "i2", "Filename": "b.png"}]
    bag = _bag_yielding(rows)
    with _patched_session(rows):
        out = resolve_reachable_rows(bag, "Image")
    assert out == rows  # full rows, NOT deduped here
    bag._dataset_table_view.assert_called_once_with("Image")


def test_resolve_reachable_rows_unknown_table_raises():
    bag = _bag_yielding([], table_exists=False)
    with pytest.raises(DerivaMLException, match="not found|not resolvable|Image"):
        resolve_reachable_rows(bag, "Image")


def test_resolve_element_rids_reachable_dedups_preserving_order():
    # Same RID surfaced twice via two FK paths -> yield once, first-seen order.
    rows = [{"RID": "i1"}, {"RID": "i2"}, {"RID": "i1"}, {"RID": "i3"}]
    bag = _bag_yielding(rows)
    with _patched_session(rows):
        out = resolve_element_rids(bag, "Image", reachable=True)
    assert out == ["i1", "i2", "i3"]


def test_resolve_element_rids_direct_uses_list_members():
    bag = MagicMock()
    bag.list_dataset_members.return_value = {"Image": [{"RID": "d1"}, {"RID": "d2"}]}
    out = resolve_element_rids(bag, "Image", reachable=False)
    assert out == ["d1", "d2"]
    bag.list_dataset_members.assert_called_once_with(recurse=True)


def test_resolve_element_rids_direct_unknown_type_raises():
    bag = MagicMock()
    bag.list_dataset_members.return_value = {"Subject": [{"RID": "s1"}]}
    with pytest.raises(DerivaMLException, match="not found|Image"):
        resolve_element_rids(bag, "Image", reachable=False)
