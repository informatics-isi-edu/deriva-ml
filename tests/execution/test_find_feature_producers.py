"""Unit tests for DerivaML.find_feature_producers (issues #370, #385).

The API follows the binding definition (provenance contract): the
executions bound to a dataset's FK-reachable members are provenance by
construction, version-scoped when requested. Path discovery targets the
feature's TARGET table (never a value/asset FK into the feature table),
feature tables are excluded as reachability intermediates, and in
version mode discovery binds to the snapshot model.

Offline: planner, feature discovery, and datapath execution mocked at
their seams. RIDs are programmatically generated.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.core.mixins.execution import BindingDiagnostic, ExecutionMixin
from deriva_ml.feature import FeatureProducerRecord


def _rid(n: int) -> str:
    return f"1-{n:04X}"


def _tbl(name: str, schema: str = "domain"):
    return SimpleNamespace(name=name, schema=SimpleNamespace(name=schema))


def _feature(name: str, target, ftable):
    return SimpleNamespace(feature_name=name, target_table=target, feature_table=ftable)


class _ChainPath:
    def __init__(self, harness, tables_visited):
        self.harness = harness
        self.tables = tables_visited
        self._grouped = False

    def link(self, table, on=None):
        return _ChainPath(self.harness, self.tables + [(table._schema, table._name)])

    def groupby(self, col):
        self._grouped = True
        return self

    def attributes(self, *cols):
        terminal = self.tables[-1]
        source = self.harness.grouped_rows if self._grouped else self.harness.raw_rows
        rows = source.get(terminal)
        if isinstance(rows, Exception):
            raise rows
        result = MagicMock()
        result.fetch.return_value = list(rows or [])
        self.harness.executed.append((list(self.tables), "grouped" if self._grouped else "raw"))
        return result


class _PBTable:
    def __init__(self, harness, schema, name):
        self.harness = harness
        self._schema = schema
        self._name = name
        self.RID = MagicMock()
        self.Execution = MagicMock()

    @property
    def columns(self):
        rec = MagicMock()
        rec.__getitem__ = lambda _s, key: MagicMock()
        return rec

    def filter(self, pred):
        return _ChainPath(self.harness, [(self._schema, self._name)])


class _Source:
    """A catalog source (live mixin or snapshot) with its own model/planner."""

    def __init__(self, harness, paths, features, relationships=None):
        self.harness = harness
        planner = MagicMock()
        planner._schema_to_paths.return_value = [[_tbl(n, s) for s, n in p] for p in paths]

        def table_relationship(a, b):
            key = (getattr(a, "name", a), getattr(b, "name", b))
            rel = (relationships or {}).get(key, "ok")
            if rel == "ambiguous":
                raise DerivaMLException(f"Ambiguous linkage between {key[0]} and {key[1]}")
            return [(SimpleNamespace(name="RID", table=a), SimpleNamespace(name=key[0], table=b))]

        planner._table_relationship.side_effect = table_relationship
        self.planner = planner
        self.model = MagicMock()
        self.model._planner = planner
        self.model.find_features.return_value = list(features)

    def pathBuilder(self):
        harness = self.harness

        class _Schema:
            def __init__(self, sname):
                self.sname = sname

            @property
            def tables(self):
                sname = self.sname

                class _T:
                    def __getitem__(_s, name):
                        if (sname, name) in harness.missing_tables:
                            raise KeyError(name)
                        return _PBTable(harness, sname, name)

                return _T()

        pb = MagicMock()
        pb.schemas = MagicMock()
        pb.schemas.__getitem__ = lambda _s, sname: _Schema(sname)
        return pb


class _Harness:
    def __init__(
        self,
        *,
        paths,
        features,
        grouped_rows=None,
        raw_rows=None,
        relationships=None,
        missing_tables=frozenset(),
        snapshot=None,
    ):
        self.grouped_rows = grouped_rows or {}
        self.raw_rows = raw_rows or {}
        self.executed = []
        self.missing_tables = set(missing_tables)

        ml = ExecutionMixin.__new__(ExecutionMixin)
        ml.ml_schema = "deriva-ml"
        live = _Source(self, paths, features, relationships)
        ml.model = live.model
        ml.pathBuilder = live.pathBuilder
        self.live = live
        self.snapshot = snapshot
        snap_obj = snapshot if snapshot is not None else live
        ml.lookup_dataset = lambda rid: SimpleNamespace(dataset_rid=rid, _version_snapshot_catalog=lambda v: snap_obj)
        self.ml = ml


# (schema, name) path shorthand
def _p(*names, schema="domain"):
    out = [("deriva-ml", "Dataset")] + [(schema, n) for n in names]
    return out


IMG, SUBJ = "Image", "Subject"
FT_IMG, FT_SUBJ = "Execution_Image_Annotation", "Execution_Subject_Grade"


def test_direct_membership_counts_server_side():
    """Single-route features count via server-side groupby — nothing raw-fetched."""
    e1 = _rid(10)
    h = _Harness(
        paths=[_p("Dataset_Image", IMG)],
        features=[_feature("Annotation", _tbl(IMG), _tbl(FT_IMG))],
        grouped_rows={("domain", FT_IMG): [{"Execution": e1, "value_count": 2}]},
    )
    out = h.ml.find_feature_producers(_rid(500))
    assert [(r.execution_rid, r.feature_name, r.element_type, r.value_count) for r in out] == [
        (e1, "Annotation", IMG, 2)
    ]
    assert all(isinstance(r, FeatureProducerRecord) for r in out)
    assert all(mode == "grouped" for _t, mode in h.executed)  # P2: server-side


def test_fk_reachable_members_are_in_scope():
    """The #385 blind-spot pin: subject-partitioned dataset finds Image producers."""
    e1 = _rid(10)
    h = _Harness(
        paths=[_p("Dataset_Subject", SUBJ, "Observation", IMG)],
        features=[_feature("Annotation", _tbl(IMG), _tbl(FT_IMG))],
        grouped_rows={("domain", FT_IMG): [{"Execution": e1, "value_count": 1}]},
    )
    out = h.ml.find_feature_producers(_rid(500))
    assert [(r.execution_rid, r.element_type, r.value_count) for r in out] == [(e1, IMG, 1)]
    tables, _mode = h.executed[0]
    assert [n for _s, n in tables] == ["Dataset", "Dataset_Subject", SUBJ, "Observation", IMG, FT_IMG]


def test_value_fk_arrival_at_feature_table_never_counts():
    """Codex P1 pin: a path reaching the FEATURE table via a value/asset FK
    (no path to the feature's TARGET) yields nothing — the binding
    attaches to its target object only."""
    h = _Harness(
        # Reachable path ends at an asset table the feature references as a
        # VALUE — not at the feature's target (Image), to which no path exists.
        paths=[_p("Dataset_File", "BBox_File")],
        features=[_feature("Annotation", _tbl(IMG), _tbl(FT_IMG))],
        grouped_rows={("domain", FT_IMG): [{"Execution": _rid(10), "value_count": 5}]},
    )
    assert h.ml.find_feature_producers(_rid(500)) == []


def test_feature_tables_excluded_as_intermediates():
    """Codex P1 pin: path enumeration excludes feature tables so
    reachability never tunnels through bindings."""
    h = _Harness(
        paths=[_p("Dataset_Image", IMG)],
        features=[
            _feature("Annotation", _tbl(IMG), _tbl(FT_IMG)),
            _feature("Grade", _tbl(SUBJ), _tbl(FT_SUBJ)),
        ],
        grouped_rows={("domain", FT_IMG): []},
    )
    h.ml.find_feature_producers(_rid(500))
    (_args, kwargs) = h.live.planner._schema_to_paths.call_args
    assert {FT_IMG, FT_SUBJ}.issubset(kwargs["exclude_tables"])


def test_multiple_paths_union_feature_rows_not_double_count():
    e1 = _rid(10)
    shared = {"RID": _rid(100), "Execution": e1}
    h = _Harness(
        paths=[
            _p("Dataset_Image", IMG),
            _p("Dataset_Subject", SUBJ, "Observation", IMG),
        ],
        features=[_feature("Annotation", _tbl(IMG), _tbl(FT_IMG))],
        raw_rows={("domain", FT_IMG): [shared]},
    )
    out = h.ml.find_feature_producers(_rid(500))
    assert [(r.execution_rid, r.value_count) for r in out] == [(e1, 1)]  # not 2


def test_ambiguous_hop_skips_path_keeping_others():
    e1 = _rid(10)
    h = _Harness(
        paths=[
            _p("Dataset_Image", IMG),
            _p("Dataset_Subject", SUBJ, "Observation", IMG),
        ],
        features=[_feature("Annotation", _tbl(IMG), _tbl(FT_IMG))],
        grouped_rows={("domain", FT_IMG): [{"Execution": e1, "value_count": 1}]},
        relationships={("Observation", IMG): "ambiguous"},
    )
    out = h.ml.find_feature_producers(_rid(500))
    assert [(r.execution_rid, r.value_count) for r in out] == [(e1, 1)]
    # Ambiguous route dropped -> single usable path -> server-side grouped.
    assert all(mode == "grouped" for _t, mode in h.executed)


def test_schema_qualified_target_keys_no_collision():
    """Codex P2 pin: same-named target tables in two schemas do not merge."""
    e1, e2 = _rid(10), _rid(11)
    h = _Harness(
        paths=[
            _p("Dataset_Image", IMG, schema="alpha"),
            _p("Dataset_Image", IMG, schema="beta"),
        ],
        features=[
            _feature("Annotation", _tbl(IMG, "alpha"), _tbl(FT_IMG, "alpha")),
            _feature("Annotation", _tbl(IMG, "beta"), _tbl(FT_IMG, "beta")),
        ],
        grouped_rows={
            ("alpha", FT_IMG): [{"Execution": e1, "value_count": 3}],
            ("beta", FT_IMG): [{"Execution": e2, "value_count": 4}],
        },
    )
    out = h.ml.find_feature_producers(_rid(500))
    assert [(r.execution_rid, r.value_count) for r in out] == [(e1, 3), (e2, 4)]


def test_null_execution_is_first_class_result():
    h = _Harness(
        paths=[_p("Dataset_Image", IMG)],
        features=[_feature("Annotation", _tbl(IMG), _tbl(FT_IMG))],
        grouped_rows={("domain", FT_IMG): [{"Execution": None, "value_count": 7}]},
    )
    out = h.ml.find_feature_producers(_rid(500))
    assert len(out) == 1 and out[0].execution_rid is None and out[0].value_count == 7


def test_per_feature_failure_degrades_keeping_others():
    e1 = _rid(10)
    h = _Harness(
        paths=[_p("Dataset_Image", IMG), _p("Dataset_Subject", SUBJ)],
        features=[
            _feature("Annotation", _tbl(IMG), _tbl(FT_IMG)),
            _feature("Grade", _tbl(SUBJ), _tbl(FT_SUBJ)),
        ],
        grouped_rows={
            ("domain", FT_IMG): RuntimeError("feature table unreadable"),
            ("domain", FT_SUBJ): [{"Execution": e1, "value_count": 1}],
        },
    )
    out = h.ml.find_feature_producers(_rid(500))
    assert [(r.feature_name, r.execution_rid) for r in out] == [("Grade", e1)]


def test_version_uses_snapshot_model_and_absent_table_is_empty():
    """Codex P1 pin: version= binds discovery AND data to the snapshot —
    the snapshot's find_features/planner are used, the live model's are
    not; a table absent from the snapshot contributes nothing (#365)."""
    e1 = _rid(10)
    h = _Harness(paths=[], features=[])  # live source: empty on purpose
    snap = _Source(
        h,
        paths=[_p("Dataset_Image", IMG), _p("Dataset_Subject", SUBJ)],
        features=[
            _feature("Annotation", _tbl(IMG), _tbl(FT_IMG)),
            _feature("Grade", _tbl(SUBJ), _tbl(FT_SUBJ)),
        ],
    )
    h.snapshot = snap
    h.ml.lookup_dataset = lambda rid: SimpleNamespace(dataset_rid=rid, _version_snapshot_catalog=lambda v: snap)
    h.grouped_rows = {("domain", FT_IMG): [{"Execution": e1, "value_count": 2}]}
    h.missing_tables = {("domain", FT_SUBJ)}  # predates the snapshot

    out = h.ml.find_feature_producers(_rid(500), version="0.4.0")
    assert [(r.feature_name, r.execution_rid, r.value_count) for r in out] == [("Annotation", e1, 2)]
    assert snap.model.find_features.called
    assert not h.live.model.find_features.called


def test_deterministic_order():
    e1 = _rid(10)
    h = _Harness(
        paths=[_p("Dataset_Image", IMG), _p("Dataset_Subject", SUBJ)],
        features=[
            _feature("Zeta", _tbl(IMG), _tbl(FT_IMG)),
            _feature("Alpha", _tbl(SUBJ), _tbl(FT_SUBJ)),
        ],
        grouped_rows={
            ("domain", FT_IMG): [{"Execution": e1, "value_count": 1}],
            ("domain", FT_SUBJ): [{"Execution": e1, "value_count": 4}],
        },
    )
    out = h.ml.find_feature_producers(_rid(500))
    assert [(r.feature_name, r.element_type) for r in out] == [("Alpha", SUBJ), ("Zeta", IMG)]


def test_no_features_empty():
    h = _Harness(paths=[_p("Dataset_Image", IMG)], features=[])
    assert h.ml.find_feature_producers(_rid(500)) == []


# --- Task 6: _find_feature_producers_impl diagnostics -----------------


def test_impl_happy_path_returns_empty_diagnostics():
    e1 = _rid(10)
    h = _Harness(
        paths=[_p("Dataset_Image", IMG)],
        features=[_feature("Annotation", _tbl(IMG), _tbl(FT_IMG))],
        grouped_rows={("domain", FT_IMG): [{"Execution": e1, "value_count": 2}]},
    )
    records, diagnostics = h.ml._find_feature_producers_impl(_rid(500))
    assert [(r.execution_rid, r.feature_name, r.element_type, r.value_count) for r in records] == [
        (e1, "Annotation", IMG, 2)
    ]
    assert diagnostics == []


def test_wrapper_output_matches_impl_records():
    e1 = _rid(10)
    h = _Harness(
        paths=[_p("Dataset_Image", IMG)],
        features=[_feature("Annotation", _tbl(IMG), _tbl(FT_IMG))],
        grouped_rows={("domain", FT_IMG): [{"Execution": e1, "value_count": 2}]},
    )
    records, _diagnostics = h.ml._find_feature_producers_impl(_rid(500))
    assert h.ml.find_feature_producers(_rid(500)) == records


def test_impl_ambiguous_hop_yields_diagnostic_and_keeps_records():
    e1 = _rid(10)
    h = _Harness(
        paths=[
            _p("Dataset_Image", IMG),
            _p("Dataset_Subject", SUBJ, "Observation", IMG),
        ],
        features=[_feature("Annotation", _tbl(IMG), _tbl(FT_IMG))],
        grouped_rows={("domain", FT_IMG): [{"Execution": e1, "value_count": 1}]},
        relationships={("Observation", IMG): "ambiguous"},
    )
    records, diagnostics = h.ml._find_feature_producers_impl(_rid(500))
    assert [(r.execution_rid, r.value_count) for r in records] == [(e1, 1)]
    assert any(d.kind == "ambiguous_hop" for d in diagnostics)
    assert all(isinstance(d, BindingDiagnostic) for d in diagnostics)


def test_impl_ambiguous_final_hop_yields_diagnostic():
    """The target->feature-table ambiguous hop (skips the whole feature),
    distinct from an ambiguous intermediate hop within a path."""
    h = _Harness(
        paths=[_p("Dataset_Image", IMG)],
        features=[_feature("Annotation", _tbl(IMG), _tbl(FT_IMG))],
        relationships={(IMG, FT_IMG): "ambiguous"},
    )
    records, diagnostics = h.ml._find_feature_producers_impl(_rid(500))
    assert records == []
    assert any(d.kind == "ambiguous_hop" and d.subject == "Annotation" for d in diagnostics)


def test_impl_snapshot_absent_table_yields_diagnostic():
    e1 = _rid(10)
    h = _Harness(paths=[], features=[])  # live source: empty on purpose
    snap = _Source(
        h,
        paths=[_p("Dataset_Image", IMG), _p("Dataset_Subject", SUBJ)],
        features=[
            _feature("Annotation", _tbl(IMG), _tbl(FT_IMG)),
            _feature("Grade", _tbl(SUBJ), _tbl(FT_SUBJ)),
        ],
    )
    h.snapshot = snap
    h.ml.lookup_dataset = lambda rid: SimpleNamespace(dataset_rid=rid, _version_snapshot_catalog=lambda v: snap)
    h.grouped_rows = {("domain", FT_IMG): [{"Execution": e1, "value_count": 2}]}
    h.missing_tables = {("domain", FT_SUBJ)}  # predates the snapshot

    records, diagnostics = h.ml._find_feature_producers_impl(_rid(500), version="0.4.0")
    assert [(r.feature_name, r.execution_rid, r.value_count) for r in records] == [("Annotation", e1, 2)]
    assert any(d.kind == "snapshot_absent" for d in diagnostics)


def test_impl_per_feature_query_failure_yields_diagnostic():
    e1 = _rid(10)
    h = _Harness(
        paths=[_p("Dataset_Image", IMG), _p("Dataset_Subject", SUBJ)],
        features=[
            _feature("Annotation", _tbl(IMG), _tbl(FT_IMG)),
            _feature("Grade", _tbl(SUBJ), _tbl(FT_SUBJ)),
        ],
        grouped_rows={
            ("domain", FT_IMG): RuntimeError("feature table unreadable"),
            ("domain", FT_SUBJ): [{"Execution": e1, "value_count": 1}],
        },
    )
    records, diagnostics = h.ml._find_feature_producers_impl(_rid(500))
    assert [(r.feature_name, r.execution_rid) for r in records] == [("Grade", e1)]
    assert any(d.kind == "query_failed" for d in diagnostics)


def test_impl_multi_path_query_failure_yields_diagnostic():
    """The multi-route branch's per-path failure also produces a diagnostic."""
    h = _Harness(
        paths=[
            _p("Dataset_Image", IMG),
            _p("Dataset_Subject", SUBJ, "Observation", IMG),
        ],
        features=[_feature("Annotation", _tbl(IMG), _tbl(FT_IMG))],
        raw_rows={("domain", FT_IMG): RuntimeError("boom")},
    )
    records, diagnostics = h.ml._find_feature_producers_impl(_rid(500))
    assert records == []
    assert any(d.kind == "query_failed" for d in diagnostics)


def test_impl_discovery_failed_find_features_raises():
    h = _Harness(paths=[_p("Dataset_Image", IMG)], features=[])
    h.live.model.find_features.side_effect = RuntimeError("catalog unreachable")
    records, diagnostics = h.ml._find_feature_producers_impl(_rid(500))
    assert records == []
    assert any(d.kind == "discovery_failed" for d in diagnostics)


def test_impl_discovery_failed_schema_to_paths_raises():
    h = _Harness(
        paths=[_p("Dataset_Image", IMG)],
        features=[_feature("Annotation", _tbl(IMG), _tbl(FT_IMG))],
    )
    h.live.planner._schema_to_paths.side_effect = RuntimeError("planner exploded")
    records, diagnostics = h.ml._find_feature_producers_impl(_rid(500))
    assert records == []
    assert any(d.kind == "discovery_failed" for d in diagnostics)
