"""Unit tests for ``ExecutionMixin._producers_of_asset`` / ``_producer_of_asset``.

Mocks ``self.model.find_association`` and ``self.pathBuilder()`` in the same
style as ``tests/execution/test_producers_of_dataset_members.py``
(``_distinct_member_output_producers`` join-shape tests): an
``ExecutionMixin.__new__`` instance wired with a ``MagicMock`` model and a
chainable ``pathBuilder()`` mock, exercising the real association-fetch
query rather than overriding the method itself.

``_producer_of_asset`` delegates to ``_producers_of_asset`` (first fetched
row, or None) — see ``execution.py`` around line 1641.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from deriva_ml.core.mixins.execution import ExecutionMixin


def _mixin_with_assoc_rows(rows: list[dict]):
    """An ExecutionMixin wired so the Output-association fetch returns ``rows``.

    Mirrors the chainable-path mocking in
    ``test_producers_of_dataset_members.py::test_join_filters_by_dataset_not_member_rid_in``.
    """
    ml = ExecutionMixin.__new__(ExecutionMixin)
    ml.model = MagicMock()

    assoc_table = SimpleNamespace(name="Image_Execution", schema=SimpleNamespace(name="domain"))
    # find_association(asset_table, "Execution") -> (assoc_table, asset_fk, exec_fk)
    ml.model.find_association = lambda asset_table, target: (assoc_table, "Image", "Execution")

    assoc_path = MagicMock()
    assoc_path.filter.return_value = assoc_path
    assoc_path.entities.return_value.fetch.return_value = rows

    schema = MagicMock()
    schema.name = "domain"
    schema.tables = {"Image_Execution": assoc_path}
    pb = MagicMock()
    pb.schemas = {"domain": schema}
    ml.pathBuilder = lambda: pb

    return ml


def test_two_output_rows_returns_both_in_fetched_order():
    ml = _mixin_with_assoc_rows(
        [
            {"Execution": "2-EXAA", "Asset_Role": "Output"},
            {"Execution": "2-EXAB", "Asset_Role": "Output"},
        ]
    )
    asset_table = SimpleNamespace(name="Image", schema=SimpleNamespace(name="domain"))

    assert ml._producers_of_asset("4-ASAA", asset_table) == ["2-EXAA", "2-EXAB"]
    assert ml._producer_of_asset("4-ASAA", asset_table) == "2-EXAA"


def test_zero_rows_returns_empty_list_and_none():
    ml = _mixin_with_assoc_rows([])
    asset_table = SimpleNamespace(name="Image", schema=SimpleNamespace(name="domain"))

    assert ml._producers_of_asset("4-ASAA", asset_table) == []
    assert ml._producer_of_asset("4-ASAA", asset_table) is None


def test_duplicate_rows_deduped_keeping_first():
    ml = _mixin_with_assoc_rows(
        [
            {"Execution": "2-EXAA", "Asset_Role": "Output"},
            {"Execution": "2-EXAB", "Asset_Role": "Output"},
            {"Execution": "2-EXAA", "Asset_Role": "Output"},
        ]
    )
    asset_table = SimpleNamespace(name="Image", schema=SimpleNamespace(name="domain"))

    assert ml._producers_of_asset("4-ASAA", asset_table) == ["2-EXAA", "2-EXAB"]
    assert ml._producer_of_asset("4-ASAA", asset_table) == "2-EXAA"
