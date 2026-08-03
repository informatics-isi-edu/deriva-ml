"""Let MagicMock bags satisfy the adapters' ``reachable=True`` default.

Shared by ``test_tf_adapter_logic.py`` and ``test_torch_adapter_logic.py``.
Both build MagicMock ``DatasetBag`` objects, and both adapters default to
``reachable=True``, which routes through
:func:`~deriva_ml.dataset.target_resolution.resolve_reachable_rows`::

    Session(bag.engine).execute(bag._dataset_table_view(table)).mappings().all()

That is real SQL over the bag's SQLite database. A MagicMock bag returns a
MagicMock where SQLAlchemy expects a ``text()`` construct, so every test in
both modules raised ``ArgumentError`` once ``reachable=True`` became the
default — undetected because neither torch nor tensorflow is installed in
the default dev environment, so ``pytest.importorskip`` skipped both files
wholesale.

The alternative fix was passing ``reachable=False`` throughout, but that
opts the logic tests out of the branch real callers actually take. Stubbing
the ``Session`` keeps them on the default path, so a future change to
reachable-vs-direct routing shows up here instead of being silently
bypassed.

Order and duplicates pass through untouched: ``resolve_element_rids`` owns
dedup, and its contract is pinned separately in
``test_reachable_enumeration.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class _TableViewSentinel:
    """What a mocked ``bag._dataset_table_view(table)`` returns.

    Carries the rows the stubbed Session should yield for that table, so
    the stub needs no knowledge of which bag it was called for.
    """

    def __init__(self, rows: list[dict]):
        self.rows = rows


class _RowsResult:
    """Stands in for a SQLAlchemy ``Result`` over a sentinel's rows."""

    def __init__(self, sentinel):
        self._rows = getattr(sentinel, "rows", [])

    def mappings(self):
        return self

    def all(self):
        return self._rows


def patch_session():
    """Patch the SQLAlchemy Session used by ``resolve_reachable_rows``.

    Returns:
        A patch context manager. Use as an autouse fixture:

        >>> @pytest.fixture(autouse=True)  # doctest: +SKIP
        ... def _reachable():
        ...     with patch_session():
        ...         yield
    """

    def fake_session(engine):
        session_cm = MagicMock()
        session = session_cm.__enter__.return_value
        session.execute.side_effect = lambda sentinel: _RowsResult(sentinel)
        return session_cm

    return patch("deriva_ml.dataset.target_resolution.Session", side_effect=fake_session)


def wire_reachable(bag, rows_by_table: dict[str, list[dict]]):
    """Point a mock bag's ``_dataset_table_view`` at the given rows.

    Args:
        bag: The MagicMock DatasetBag to wire.
        rows_by_table: Mapping of table name to the row dicts the reachable
            enumeration should surface for it. Usually the same rows the
            bag's ``list_dataset_members`` returns.

    Returns:
        The same bag, for chaining.
    """
    bag._dataset_table_view = MagicMock(side_effect=lambda table: _TableViewSentinel(rows_by_table.get(table, [])))
    return bag
