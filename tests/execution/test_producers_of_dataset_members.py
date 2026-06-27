"""Unit tests for ``ExecutionMixin._producers_of_dataset_members``.

Mocks ``lookup_dataset``/``list_dataset_members`` and the per-table
association fetch so the helper's enumerate-members → distinct-producers
logic is exercised offline. The catalog-bound query path is covered by
the live test in ``tests/execution/test_lookup_lineage_live.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from deriva_ml.core.mixins.execution import ExecutionMixin


class _FakeMembersML(ExecutionMixin):
    """ExecutionMixin host scripting just what _producers_of_dataset_members needs.

    Scripts:
      * lookup_dataset(rid).list_dataset_members() -> members dict
      * model.is_asset(table) -> whether a member-type table is an asset table
      * the per-(asset_table, member_rids) Output-producer lookup
    """

    def __init__(self) -> None:
        # dataset_rid -> {member_type: [ {"RID": ...}, ... ]}
        self._members: dict[str, dict[str, list[dict[str, Any]]]] = {}
        # member_type name -> bool (is it an asset table?)
        self._asset_types: set[str] = set()
        # (asset_table, member_rid) -> producer execution RID or None
        self._member_producers: dict[tuple[str, str], str | None] = {}
        # asset tables that have NO <Asset>_Execution association at all
        self._no_assoc_tables: set[str] = set()

        self.model = MagicMock()
        self.model.is_asset = lambda table: getattr(table, "name", table) in self._asset_types
        self.model.name_to_table = lambda name: _T(name)

    # scripting helpers -----------------------------------------------------
    def set_members(self, dataset_rid: str, members: dict[str, list[dict[str, Any]]]) -> None:
        self._members[dataset_rid] = members

    def mark_asset_types(self, *names: str) -> None:
        self._asset_types.update(names)

    def set_member_producer(self, asset_table: str, member_rid: str, producer: str | None) -> None:
        self._member_producers[(asset_table, member_rid)] = producer

    def mark_no_association(self, asset_table: str) -> None:
        self._no_assoc_tables.add(asset_table)

    # mocked primitives -----------------------------------------------------
    def lookup_dataset(self, rid: str) -> Any:  # type: ignore[override]
        ds = MagicMock()
        ds.list_dataset_members = lambda version=None: self._members.get(rid, {})
        return ds

    def _distinct_member_output_producers(self, asset_table: str, member_rids: list[str]) -> set[str]:
        """Test seam the real helper calls once per asset table.

        The real implementation issues the chunked association query here;
        the test scripts its result directly.
        """
        if asset_table in self._no_assoc_tables:
            return set()
        out: set[str] = set()
        for rid in member_rids:
            p = self._member_producers.get((asset_table, rid))
            if p:
                out.add(p)
        return out


class _T:
    def __init__(self, name: str) -> None:
        self.name = name


def test_distinct_producers_dedup_across_many_members():
    """2000 members sharing one producer -> a single producer RID."""
    ml = _FakeMembersML()
    ml.mark_asset_types("Image")
    members = {"Image": [{"RID": f"4-IMG{i}"} for i in range(2000)]}
    ml.set_members("1-DSAA", members)
    for i in range(2000):
        ml.set_member_producer("Image", f"4-IMG{i}", "2-EXUP")

    result = ml._producers_of_dataset_members("1-DSAA")

    assert result == {"2-EXUP"}


def test_no_members_returns_empty_set():
    ml = _FakeMembersML()
    ml.set_members("1-DSAA", {})
    assert ml._producers_of_dataset_members("1-DSAA") == set()


def test_members_with_no_output_producer_returns_empty_set():
    ml = _FakeMembersML()
    ml.mark_asset_types("Image")
    ml.set_members("1-DSAA", {"Image": [{"RID": "4-IMG0"}]})
    ml.set_member_producer("Image", "4-IMG0", None)
    assert ml._producers_of_dataset_members("1-DSAA") == set()


def test_union_across_multiple_asset_tables():
    ml = _FakeMembersML()
    ml.mark_asset_types("Image", "File")
    ml.set_members(
        "1-DSAA",
        {"Image": [{"RID": "4-IMG0"}], "File": [{"RID": "4-FIL0"}]},
    )
    ml.set_member_producer("Image", "4-IMG0", "2-EXAA")
    ml.set_member_producer("File", "4-FIL0", "2-EXAB")
    assert ml._producers_of_dataset_members("1-DSAA") == {"2-EXAA", "2-EXAB"}


def test_nested_dataset_and_non_asset_members_skipped():
    ml = _FakeMembersML()
    ml.mark_asset_types("Image")  # "Dataset" is intentionally NOT an asset type
    ml.set_members(
        "1-DSAA",
        {
            "Image": [{"RID": "4-IMG0"}],
            "Dataset": [{"RID": "1-DSCH"}],  # nested dataset member -> skipped
        },
    )
    ml.set_member_producer("Image", "4-IMG0", "2-EXAA")
    # Even if a producer were scriptable for the nested dataset, it must be ignored.
    assert ml._producers_of_dataset_members("1-DSAA") == {"2-EXAA"}


def test_asset_table_without_execution_association_skipped():
    ml = _FakeMembersML()
    ml.mark_asset_types("Image")
    ml.mark_no_association("Image")
    ml.set_members("1-DSAA", {"Image": [{"RID": "4-IMG0"}]})
    ml.set_member_producer("Image", "4-IMG0", "2-EXAA")  # ignored: no assoc table
    assert ml._producers_of_dataset_members("1-DSAA") == set()


def test_distinct_member_output_producers_chunks_not_per_member():
    """The real seam issues O(chunks) association fetches, never O(members)."""
    from unittest.mock import MagicMock

    ml = _FakeMembersML()
    # Real model.find_association + name_to_table: return stub table/assoc.
    assoc_tbl = MagicMock()
    assoc_tbl.schema.name = "deriva-ml"
    assoc_tbl.name = "Image_Execution"
    ml.model.find_association = lambda asset_table, target: (assoc_tbl, "Image", "Execution")
    ml.model.name_to_table = lambda name: _T(name)

    # Mock pathBuilder so .filter(...).filter(...).entities().fetch() returns []
    # and counts fetch() calls.
    fetch_calls = {"n": 0}

    def _fetch():
        fetch_calls["n"] += 1
        return []

    entities = MagicMock()
    entities.fetch = _fetch
    path = MagicMock()
    path.entities.return_value = entities
    path.filter.return_value = path  # chainable .filter(...).filter(...)
    # column access: assoc_path.columns[asset_fk] and assoc_path.Asset_Role
    path.columns = {"Image": MagicMock()}
    path.columns["Image"].in_ = lambda values: MagicMock()
    path.Asset_Role = MagicMock()

    schema = MagicMock()
    schema.tables = {"Image_Execution": path}
    pb = MagicMock()
    pb.schemas = {"deriva-ml": schema}
    ml.pathBuilder = lambda: pb

    member_rids = [f"4-IMG{i}" for i in range(2000)]
    # Call the real ExecutionMixin implementation, bypassing _FakeMembersML's
    # seam override, so the chunked-query behavior is actually exercised.
    ExecutionMixin._distinct_member_output_producers(ml, "Image", member_rids)

    # 2000 members / 500 chunk = 4 fetches. The point: it does NOT scale with
    # the number of members (would be 2000 if implemented per-asset).
    assert fetch_calls["n"] == 4, f"expected 4 chunked fetches, got {fetch_calls['n']}"
