"""Unit tests for ``ExecutionMixin._producers_of_dataset_members``.

Mocks ``lookup_dataset`` and the per-table association fetch so the helper's
discover-associations → distinct-producers logic is exercised offline. The
server-side join query path is covered by the live test in
``tests/execution/test_lookup_lineage_live.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from deriva_ml.core.mixins.execution import ExecutionMixin


class _FakeAssoc:
    """Stand-in for a membership association table from find_associations()."""

    def __init__(self, membership_name, member_table_name, member_col="Image", target_col="RID"):
        self.table = SimpleNamespace(name=membership_name, schema=SimpleNamespace(name="domain"))
        member_tbl = SimpleNamespace(name=member_table_name, schema=SimpleNamespace(name="domain"))
        ofk = SimpleNamespace(
            pk_table=member_tbl,
            foreign_key_columns=[SimpleNamespace(name=member_col)],
            referenced_columns=[SimpleNamespace(name=target_col)],
        )
        # other_fkeys.pop() must return ofk
        self.other_fkeys = [ofk]


class _FakeMembersML(ExecutionMixin):
    """Scripts what the rewritten _producers_of_dataset_members needs:
    a dataset table whose find_associations() yields membership associations,
    is_asset() per member table, and a seam (_distinct_member_output_producers)
    that returns scripted producer sets per member table.
    """

    def __init__(self):
        # member_table_name -> set of producer RIDs (the seam result)
        self._producers_by_table: dict[str, set[str]] = {}
        # which member tables are asset tables
        self._asset_tables: set[str] = set()
        # membership associations the dataset table "has"
        self._assocs: list[_FakeAssoc] = []

        self.model = MagicMock()
        self.model.is_asset = lambda table: getattr(table, "name", table) in self._asset_tables

        # _dataset_table.find_associations() -> self._assocs
        self._dataset_table = SimpleNamespace(find_associations=lambda: list(self._assocs))

    # scripting --------------------------------------------------------------
    def add_member_table(
        self,
        membership_name,
        member_table_name,
        *,
        is_asset=True,
        producers=frozenset(),
        member_col="Image",
        target_col="RID",
    ):
        self._assocs.append(_FakeAssoc(membership_name, member_table_name, member_col, target_col))
        if is_asset:
            self._asset_tables.add(member_table_name)
        self._producers_by_table[member_table_name] = set(producers)

    # mocked primitives ------------------------------------------------------
    def lookup_dataset(self, rid):  # type: ignore[override]
        snap = MagicMock()
        snap.pathBuilder = lambda: MagicMock()
        ds = SimpleNamespace(
            dataset_rid=rid,
            _version_snapshot_catalog=lambda version: snap,
            _dataset_table=self._dataset_table,
        )
        return ds

    def _distinct_member_output_producers(
        self,
        snapshot_pb,
        membership_table,
        member_table,  # type: ignore[override]
        member_link_column,
        target_column,
        dataset_rid,
        model=None,
    ):
        # Seam: return scripted producers for this member table.
        return set(self._producers_by_table.get(getattr(member_table, "name", member_table), set()))


# ---------------------------------------------------------------------------
# Logic tests (via the seam)
# ---------------------------------------------------------------------------


def test_dedup_across_member_tables():
    ml = _FakeMembersML()
    ml.add_member_table("Dataset_Image", "Image", producers={"2-EXUP"})
    assert ml._producers_of_dataset_members("1-DSAA") == {"2-EXUP"}


def test_union_across_multiple_asset_tables():
    ml = _FakeMembersML()
    ml.add_member_table("Dataset_Image", "Image", producers={"2-EXAA"})
    ml.add_member_table("Dataset_File", "File", producers={"2-EXAB"}, member_col="File")
    assert ml._producers_of_dataset_members("1-DSAA") == {"2-EXAA", "2-EXAB"}


def test_non_asset_member_tables_skipped():
    ml = _FakeMembersML()
    ml.add_member_table("Dataset_Image", "Image", producers={"2-EXAA"})
    ml.add_member_table("Dataset_Dataset", "Dataset", is_asset=False, producers={"2-EXNO"})
    assert ml._producers_of_dataset_members("1-DSAA") == {"2-EXAA"}


def test_no_producers_returns_empty():
    ml = _FakeMembersML()
    ml.add_member_table("Dataset_Image", "Image", producers=set())
    assert ml._producers_of_dataset_members("1-DSAA") == set()


def test_no_member_tables_returns_empty():
    ml = _FakeMembersML()
    assert ml._producers_of_dataset_members("1-DSAA") == set()


# ---------------------------------------------------------------------------
# Join-shape test — regression guard for tk-023 URL-length bug class
# ---------------------------------------------------------------------------


def test_join_filters_by_dataset_not_member_rid_in():
    """The real join filters membership by Dataset==rid and never builds an
    .in_() over member RIDs (the tk-023 URL-length bug class)."""
    calls = {"dataset_filter": False, "member_in_called": False, "asset_role_output": False}

    # Column whose .in_() would be the BUG — flag it if ever called.
    member_col = MagicMock()
    member_col.in_ = lambda *a, **k: calls.__setitem__("member_in_called", True) or MagicMock()

    def _make_table(name):
        t = MagicMock()
        t.columns = {"RID": member_col, "Image": MagicMock()}

        # membership_path.Dataset == rid  -> record it
        def _eq(other):
            calls["dataset_filter"] = True
            return MagicMock()

        t.Dataset = MagicMock()
        t.Dataset.__eq__ = lambda self_, other: _eq(other)
        # Asset_Role == "Output"
        t.Asset_Role = MagicMock()

        def _asset_role_eq(self_, other):
            calls["asset_role_output"] = other == "Output"
            return MagicMock()

        t.Asset_Role.__eq__ = _asset_role_eq
        t.Execution = MagicMock()
        # chainable path: filter/link return a path with attributes().fetch()
        path = MagicMock()
        path.filter.return_value = path
        path.link.return_value = path
        path.attributes.return_value.fetch.return_value = [{"Execution": "2-EXUP"}]
        t.filter = path.filter
        return t, path

    membership_t, mpath = _make_table("Dataset_Image")
    member_t, _ = _make_table("Image")
    exec_t, _ = _make_table("Image_Execution")

    schema = MagicMock()
    schema.name = "domain"
    schema.tables = {"Dataset_Image": membership_t, "Image": member_t, "Image_Execution": exec_t}
    pb = MagicMock()
    pb.schemas = {"domain": schema}

    ml = ExecutionMixin.__new__(ExecutionMixin)
    ml.model = MagicMock()
    # find_association(member, "Execution") -> (exec_assoc_table, member_fk, exec_fk)
    exec_assoc = SimpleNamespace(name="Image_Execution", schema=SimpleNamespace(name="domain"))
    ml.model.find_association = lambda mt, target: (exec_assoc, "Image", "Execution")

    membership_table = SimpleNamespace(name="Dataset_Image", schema=SimpleNamespace(name="domain"))
    member_table = SimpleNamespace(name="Image", schema=SimpleNamespace(name="domain"))

    result = ml._distinct_member_output_producers(pb, membership_table, member_table, "Image", "RID", "1-DSAA")

    assert result == {"2-EXUP"}
    assert calls["dataset_filter"] is True, "must filter membership by Dataset==rid"
    assert calls["member_in_called"] is False, "must NOT build an .in_() over member RIDs (tk-023)"


# ---------------------------------------------------------------------------
# Snapshot-schema guards (issue #365)
#
# The snapshot pathBuilder reflects the catalog schema as of the dataset
# version's snaptime. A table created after that snaptime is absent from the
# snapshot schema, so `snapshot_pb.schemas[...].tables[...]` raises KeyError.
#
# Only the membership lookup was guarded; member_table and the
# <member>_Execution association were not. Because the exception propagated
# out through lookup_lineage, an affected execution lost its ENTIRE lineage
# walk — consumed datasets, consumed assets, and parents — not merely this
# producer set. The snapshot holds no rows for a table that did not yet exist,
# so an empty producer set is the correct answer for all three.
# ---------------------------------------------------------------------------


def _mixin_for_snapshot_lookup():
    """An ExecutionMixin wired for _distinct_member_output_producers.

    Returns:
        Tuple of ``(ml, membership_table, member_table)`` ready to call the
        helper. The caller supplies the snapshot pathBuilder, which is what
        each test varies.
    """
    ml = ExecutionMixin.__new__(ExecutionMixin)
    ml.model = MagicMock()
    exec_assoc = SimpleNamespace(name="Image_Execution", schema=SimpleNamespace(name="domain"))
    ml.model.find_association = lambda mt, target: (exec_assoc, "Image", "Execution")

    membership_table = SimpleNamespace(name="Dataset_Image", schema=SimpleNamespace(name="domain"))
    member_table = SimpleNamespace(name="Image", schema=SimpleNamespace(name="domain"))
    return ml, membership_table, member_table


def _snapshot_pb_with(table_names):
    """A snapshot pathBuilder exposing only ``table_names``.

    Anything absent raises KeyError on lookup, exactly as a real snapshot
    predating that table's creation does.
    """
    schema = MagicMock()
    schema.name = "domain"
    schema.tables = {name: MagicMock() for name in table_names}
    pb = MagicMock()
    pb.schemas = {"domain": schema}
    return pb


def test_membership_table_absent_from_snapshot_returns_empty():
    """Membership table postdating the snapshot -> empty set, no raise."""
    ml, membership_table, member_table = _mixin_for_snapshot_lookup()
    pb = _snapshot_pb_with(["Image", "Image_Execution"])  # no Dataset_Image

    result = ml._distinct_member_output_producers(pb, membership_table, member_table, "Image", "RID", "1-DSAA")

    assert result == set()


def test_member_table_absent_from_snapshot_returns_empty():
    """Member table postdating the snapshot -> empty set, no raise.

    Previously unguarded: raised KeyError out through lookup_lineage.
    """
    ml, membership_table, member_table = _mixin_for_snapshot_lookup()
    pb = _snapshot_pb_with(["Dataset_Image", "Image_Execution"])  # no Image

    result = ml._distinct_member_output_producers(pb, membership_table, member_table, "Image", "RID", "1-DSAA")

    assert result == set()


def test_exec_association_absent_from_snapshot_returns_empty():
    """<member>_Execution postdating the snapshot -> empty set, no raise.

    This is the exact failure reported in #365: KeyError 'Image_Execution'
    aborted the whole lineage walk for eye-ai executions 7-QCAA / 5-E9WE.
    """
    ml, membership_table, member_table = _mixin_for_snapshot_lookup()
    pb = _snapshot_pb_with(["Dataset_Image", "Image"])  # no Image_Execution

    result = ml._distinct_member_output_producers(pb, membership_table, member_table, "Image", "RID", "1-DSAA")

    assert result == set()


def test_whole_schema_absent_from_snapshot_returns_empty():
    """A snapshot predating the domain schema itself -> empty set, no raise."""
    ml, membership_table, member_table = _mixin_for_snapshot_lookup()
    pb = MagicMock()
    pb.schemas = {}  # schema lookup itself raises KeyError

    result = ml._distinct_member_output_producers(pb, membership_table, member_table, "Image", "RID", "1-DSAA")

    assert result == set()
