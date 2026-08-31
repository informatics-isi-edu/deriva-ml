"""Unit tests for DerivaML.find_feature_producers (issues #370, #385).

The API answers: which executions wrote feature values onto a dataset's
members — where "members" follows the binding definition (provenance
contract): objects that are members OR are FK-reachable from members
(subject-partitioned datasets), with optional version-snapshot scoping.

Offline: the planner's path enumeration, feature discovery, and the
datapath chain execution are mocked at their seams. RIDs are
programmatically generated; assertions compare only flow-through values.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.core.mixins.execution import ExecutionMixin
from deriva_ml.feature import FeatureProducerRecord


def _rid(n: int) -> str:
    return f"1-{n:04X}"


def _tbl(name: str):
    return SimpleNamespace(name=name, schema=SimpleNamespace(name="domain"))


def _feature(name: str, target, ftable_name: str):
    return SimpleNamespace(
        feature_name=name,
        target_table=target,
        feature_table=_tbl(ftable_name),
    )


class _ChainPath:
    """Datapath stand-in: filter/link chain ending in attributes().fetch()."""

    def __init__(self, harness, tables_visited):
        self.harness = harness
        self.tables = tables_visited

    def link(self, table, on=None):
        return _ChainPath(self.harness, self.tables + [table._name])

    def attributes(self, *cols):
        terminal = self.tables[-1]
        rows = self.harness.rows_by_terminal.get(terminal)
        if isinstance(rows, Exception):
            raise rows
        result = MagicMock()
        result.fetch.return_value = list(rows or [])
        self.harness.executed_paths.append(list(self.tables))
        return result


class _PBTable:
    def __init__(self, harness, name):
        self.harness = harness
        self._name = name
        self.RID = MagicMock()
        self.Execution = MagicMock()

    @property
    def columns(self):
        rec = MagicMock()
        rec.__getitem__ = lambda _s, key: MagicMock()
        return rec

    def filter(self, pred):
        return _ChainPath(self.harness, [self._name])


class _Harness:
    """Wires an ExecutionMixin with mocked planner paths + datapath."""

    def __init__(self, *, paths, features, rows_by_terminal, relationships=None, missing_tables=frozenset()):
        self.rows_by_terminal = rows_by_terminal
        self.executed_paths = []
        self.missing_tables = set(missing_tables)

        ml = ExecutionMixin.__new__(ExecutionMixin)
        ml.ml_schema = "deriva-ml"

        planner = MagicMock()
        planner._schema_to_paths.return_value = [[_tbl(n) for n in p] for p in paths]

        def table_relationship(a, b):
            key = (getattr(a, "name", a), getattr(b, "name", b))
            rel = (relationships or {}).get(key, "ok")
            if rel == "ambiguous":
                raise DerivaMLException(f"Ambiguous linkage between {key[0]} and {key[1]}")
            col = SimpleNamespace(name="RID", table=a)
            col2 = SimpleNamespace(name=key[0], table=b)
            return [(col, col2)]

        planner._table_relationship.side_effect = table_relationship

        model = MagicMock()
        model._planner = planner
        model.find_features.return_value = list(features)
        ml.model = model
        ml.lookup_dataset = lambda rid: SimpleNamespace(
            dataset_rid=rid,
            _version_snapshot_catalog=lambda v: SimpleNamespace(pathBuilder=lambda: self._pb()),
        )
        ml.pathBuilder = lambda: self._pb()
        self.ml = ml

    def _pb(self):
        harness = self

        class _Schema:
            @property
            def tables(self):
                class _T:
                    def __getitem__(_s, name):
                        if name in harness.missing_tables:
                            raise KeyError(name)
                        return _PBTable(harness, name)

                return _T()

        pb = MagicMock()
        pb.schemas = {"domain": _Schema(), "deriva-ml": _Schema()}
        return pb


IMG = "Image"
SUBJ = "Subject"
FT_IMG = "Execution_Image_Annotation"
FT_SUBJ = "Execution_Subject_Grade"


def _direct_paths():
    return [["Dataset", "Dataset_Image", IMG, FT_IMG]]


def _reachable_paths():
    return [["Dataset", "Dataset_Subject", SUBJ, "Observation", IMG, FT_IMG]]


def test_direct_membership_still_works():
    e1 = _rid(10)
    h = _Harness(
        paths=_direct_paths(),
        features=[_feature("Annotation", _tbl(IMG), FT_IMG)],
        rows_by_terminal={FT_IMG: [{"RID": _rid(100), "Execution": e1}, {"RID": _rid(101), "Execution": e1}]},
    )
    out = h.ml.find_feature_producers(_rid(500))
    assert [(r.execution_rid, r.feature_name, r.element_type, r.value_count) for r in out] == [
        (e1, "Annotation", IMG, 2)
    ]
    assert all(isinstance(r, FeatureProducerRecord) for r in out)


def test_fk_reachable_members_are_in_scope():
    """THE #385 blind-spot pin: a subject-partitioned dataset (members =
    Subjects; features bound to FK-reachable Images) must return the
    producers. The shipped direct-membership implementation returns []."""
    e1 = _rid(10)
    h = _Harness(
        paths=_reachable_paths(),
        features=[_feature("Annotation", _tbl(IMG), FT_IMG)],
        rows_by_terminal={FT_IMG: [{"RID": _rid(100), "Execution": e1}]},
    )
    out = h.ml.find_feature_producers(_rid(500))
    assert [(r.execution_rid, r.element_type, r.value_count) for r in out] == [(e1, IMG, 1)]
    # And the executed chain really walked the multi-hop path.
    assert ["Dataset", "Dataset_Subject", SUBJ, "Observation", IMG, FT_IMG] in h.executed_paths


def test_multiple_paths_union_feature_rows_not_double_count():
    """A feature row reachable via two FK routes counts once (#316 lesson)."""
    e1 = _rid(10)
    shared_row = {"RID": _rid(100), "Execution": e1}
    h = _Harness(
        paths=[
            ["Dataset", "Dataset_Image", IMG, FT_IMG],
            ["Dataset", "Dataset_Subject", SUBJ, "Observation", IMG, FT_IMG],
        ],
        features=[_feature("Annotation", _tbl(IMG), FT_IMG)],
        rows_by_terminal={FT_IMG: [shared_row]},
    )
    out = h.ml.find_feature_producers(_rid(500))
    assert [(r.execution_rid, r.value_count) for r in out] == [(e1, 1)]  # not 2


def test_ambiguous_hop_skips_path_keeping_others():
    e1 = _rid(10)
    h = _Harness(
        paths=[
            ["Dataset", "Dataset_Image", IMG, FT_IMG],
            ["Dataset", "Dataset_Subject", SUBJ, "Observation", IMG, FT_IMG],
        ],
        features=[_feature("Annotation", _tbl(IMG), FT_IMG)],
        rows_by_terminal={FT_IMG: [{"RID": _rid(100), "Execution": e1}]},
        relationships={("Observation", IMG): "ambiguous"},
    )
    out = h.ml.find_feature_producers(_rid(500))
    assert [(r.execution_rid, r.value_count) for r in out] == [(e1, 1)]
    assert ["Dataset", "Dataset_Subject", SUBJ, "Observation", IMG, FT_IMG] not in h.executed_paths


def test_null_execution_is_first_class_result():
    h = _Harness(
        paths=_direct_paths(),
        features=[_feature("Annotation", _tbl(IMG), FT_IMG)],
        rows_by_terminal={FT_IMG: [{"RID": _rid(100), "Execution": None}]},
    )
    out = h.ml.find_feature_producers(_rid(500))
    assert len(out) == 1 and out[0].execution_rid is None and out[0].value_count == 1


def test_per_feature_failure_degrades_keeping_others():
    e1 = _rid(10)
    h = _Harness(
        paths=[
            ["Dataset", "Dataset_Image", IMG, FT_IMG],
            ["Dataset", "Dataset_Subject", SUBJ, FT_SUBJ],
        ],
        features=[
            _feature("Annotation", _tbl(IMG), FT_IMG),
            _feature("Grade", _tbl(SUBJ), FT_SUBJ),
        ],
        rows_by_terminal={
            FT_IMG: RuntimeError("feature table unreadable"),
            FT_SUBJ: [{"RID": _rid(100), "Execution": e1}],
        },
    )
    out = h.ml.find_feature_producers(_rid(500))
    assert [(r.feature_name, r.execution_rid) for r in out] == [("Grade", e1)]


def test_deterministic_order():
    e1 = _rid(10)
    h = _Harness(
        paths=[
            ["Dataset", "Dataset_Image", IMG, FT_IMG],
            ["Dataset", "Dataset_Subject", SUBJ, FT_SUBJ],
        ],
        features=[
            _feature("Zeta", _tbl(IMG), FT_IMG),
            _feature("Alpha", _tbl(SUBJ), FT_SUBJ),
        ],
        rows_by_terminal={
            FT_IMG: [{"RID": _rid(100), "Execution": e1}],
            FT_SUBJ: [{"RID": _rid(101), "Execution": e1}],
        },
    )
    out = h.ml.find_feature_producers(_rid(500))
    assert [(r.feature_name, r.element_type) for r in out] == [("Alpha", SUBJ), ("Zeta", IMG)]


def test_version_scoping_uses_snapshot_and_absent_table_is_empty():
    """version= resolves via the version snapshot; a feature table absent
    from that snapshot (predates it) yields no rows for that path — never
    a raise (the #365 discipline)."""
    e1 = _rid(10)
    h = _Harness(
        paths=[
            ["Dataset", "Dataset_Image", IMG, FT_IMG],
            ["Dataset", "Dataset_Subject", SUBJ, FT_SUBJ],
        ],
        features=[
            _feature("Annotation", _tbl(IMG), FT_IMG),
            _feature("Grade", _tbl(SUBJ), FT_SUBJ),
        ],
        rows_by_terminal={
            FT_IMG: [{"RID": _rid(100), "Execution": e1}],
            FT_SUBJ: [{"RID": _rid(101), "Execution": e1}],
        },
        missing_tables={FT_SUBJ},  # absent from the snapshot schema
    )
    out = h.ml.find_feature_producers(_rid(500), version="0.4.0")
    assert [(r.feature_name, r.execution_rid) for r in out] == [("Annotation", e1)]


def test_no_features_empty():
    h = _Harness(paths=_direct_paths(), features=[], rows_by_terminal={})
    assert h.ml.find_feature_producers(_rid(500)) == []
