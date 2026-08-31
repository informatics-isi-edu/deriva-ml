"""Unit tests for DerivaML.find_feature_producers (issue #370).

The API answers: which executions wrote feature values onto a dataset's
members? This is the feature arc from issue #367 §6 — the traversal that
recovers contributors (e.g. an annotator or cropper run) that are sibling
consumers rather than data-flow ancestors, so no lineage walk reaches them.

Offline: dataset-table associations, feature discovery, and the
server-side groupby path are all mocked in the style of
test_producers_of_dataset_members. RIDs are programmatically generated.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from deriva_ml.core.mixins.execution import ExecutionMixin
from deriva_ml.feature import FeatureProducerRecord


def _rid(n: int) -> str:
    return f"1-{n:04X}"


class _FakeAssoc:
    """Stand-in for a membership association from find_associations()."""

    def __init__(self, membership_name, member_table, member_col, target_col="RID"):
        self.table = SimpleNamespace(name=membership_name, schema=SimpleNamespace(name="domain"))
        ofk = SimpleNamespace(
            pk_table=member_table,
            foreign_key_columns=[SimpleNamespace(name=member_col)],
            referenced_columns=[SimpleNamespace(name=target_col)],
        )
        self.other_fkeys = [ofk]


def _feature(feature_name, member_table, table_name):
    return SimpleNamespace(
        feature_name=feature_name,
        target_table=member_table,
        feature_table=SimpleNamespace(name=table_name, schema=SimpleNamespace(name="domain")),
    )


def _groupby_table(rows, fail=False):
    """pathBuilder table mock whose grouped fetch yields `rows`."""
    t = MagicMock()
    if fail:
        t.filter.side_effect = RuntimeError("feature table unreadable")
        return t
    path = MagicMock()
    t.filter.return_value = path
    path.link.return_value = path
    path.groupby.return_value.attributes.return_value.fetch.return_value = list(rows)
    return t


def _mixin(associations, features_by_table, grouped_rows_by_feature, fail_features=frozenset()):
    """ExecutionMixin wired for find_feature_producers.

    Args:
        associations: list of (membership_name, member_table_name, member_col).
        features_by_table: member table name -> list of feature names.
        grouped_rows_by_feature: feature table name -> grouped result rows.
        fail_features: feature table names whose query raises.
    """
    ml = ExecutionMixin.__new__(ExecutionMixin)
    ml.ml_schema = "deriva-ml"

    member_tables = {}
    assoc_objs = []
    for membership_name, member_name, member_col in associations:
        member_tables.setdefault(member_name, SimpleNamespace(name=member_name, schema=SimpleNamespace(name="domain")))
        assoc_objs.append(_FakeAssoc(membership_name, member_tables[member_name], member_col))

    dataset = SimpleNamespace(
        dataset_rid=None,
        _dataset_table=SimpleNamespace(find_associations=lambda: list(assoc_objs)),
    )
    ml.lookup_dataset = lambda rid: dataset

    def fake_find_features(table):
        name = getattr(table, "name", table)
        return [_feature(fn, member_tables[name], f"Feature_{name}_{fn}") for fn in features_by_table.get(name, [])]

    ml.find_features = fake_find_features

    tables = {}
    for membership_name, _m, _c in associations:
        tables[membership_name] = _groupby_table([])  # membership path start
    for ftable_name, rows in grouped_rows_by_feature.items():
        tables[ftable_name] = MagicMock()
    schema = MagicMock()

    # The path starts from the MEMBERSHIP table; scripted rows are keyed by
    # the feature table linked in. Wire: membership.filter().link(ftable)...
    def table_getter(name):
        return tables[name]

    membership_paths = {}

    class _MembershipTable:
        def __init__(self, name):
            self.name = name
            self.filter_calls = 0
            self.Dataset = MagicMock()
            self.columns = MagicMock()

        def filter(self, pred):
            self.filter_calls += 1

            class _Path:
                def link(self, ftable, on=None):
                    p = MagicMock()
                    fname = ftable._fname
                    if fname in fail_features:
                        raise RuntimeError("feature table unreadable")
                    p.groupby.return_value.attributes.return_value.fetch.return_value = list(
                        grouped_rows_by_feature.get(fname, [])
                    )
                    return p

            return _Path()

    class _FeatureTable:
        def __init__(self, name):
            self._fname = name
            self.Execution = MagicMock()
            self.RID = MagicMock()
            self.columns = MagicMock()

    all_tables = {}
    for membership_name, _m, _c in associations:
        mt = _MembershipTable(membership_name)
        membership_paths[membership_name] = mt
        all_tables[membership_name] = mt
    for member_name, fns in features_by_table.items():
        for fn in fns:
            fname = f"Feature_{member_name}_{fn}"
            all_tables[fname] = _FeatureTable(fname)

    schema.tables = all_tables
    pb = MagicMock()
    pb.schemas = {"domain": schema, "deriva-ml": schema}
    ml.pathBuilder = lambda: pb
    return ml, membership_paths


def test_aggregates_per_execution_feature_element():
    e1, e2 = _rid(10), _rid(11)
    ml, paths = _mixin(
        associations=[("Dataset_Image", "Image", "Image")],
        features_by_table={"Image": ["Annotation"]},
        grouped_rows_by_feature={
            "Feature_Image_Annotation": [
                {"Execution": e1, "value_count": 591},
                {"Execution": e2, "value_count": 3},
            ]
        },
    )
    out = ml.find_feature_producers(_rid(500))
    assert [(r.execution_rid, r.feature_name, r.element_type, r.value_count) for r in out] == [
        (e1, "Annotation", "Image", 591),
        (e2, "Annotation", "Image", 3),
    ]
    assert all(isinstance(r, FeatureProducerRecord) for r in out)
    assert paths["Dataset_Image"].filter_calls == 1  # one server-side join per feature


def test_null_execution_is_first_class_result():
    ml, _ = _mixin(
        associations=[("Dataset_Image", "Image", "Image")],
        features_by_table={"Image": ["Annotation"]},
        grouped_rows_by_feature={"Feature_Image_Annotation": [{"Execution": None, "value_count": 7}]},
    )
    out = ml.find_feature_producers(_rid(500))
    assert len(out) == 1
    assert out[0].execution_rid is None
    assert out[0].value_count == 7


def test_per_feature_failure_degrades_keeping_others():
    e1 = _rid(10)
    ml, _ = _mixin(
        associations=[("Dataset_Image", "Image", "Image")],
        features_by_table={"Image": ["Broken", "Annotation"]},
        grouped_rows_by_feature={"Feature_Image_Annotation": [{"Execution": e1, "value_count": 2}]},
        fail_features={"Feature_Image_Broken"},
    )
    out = ml.find_feature_producers(_rid(500))
    assert [(r.feature_name, r.execution_rid) for r in out] == [("Annotation", e1)]


def test_multiple_member_tables_and_deterministic_order():
    e1 = _rid(10)
    ml, _ = _mixin(
        associations=[
            ("Dataset_Subject", "Subject", "Subject"),
            ("Dataset_Image", "Image", "Image"),
        ],
        features_by_table={"Image": ["Zeta"], "Subject": ["Alpha"]},
        grouped_rows_by_feature={
            "Feature_Image_Zeta": [{"Execution": e1, "value_count": 1}],
            "Feature_Subject_Alpha": [{"Execution": e1, "value_count": 4}],
        },
    )
    out = ml.find_feature_producers(_rid(500))
    # Sorted by (feature_name, element_type, execution_rid)
    assert [(r.feature_name, r.element_type) for r in out] == [
        ("Alpha", "Subject"),
        ("Zeta", "Image"),
    ]


def test_nested_dataset_members_skipped_and_no_features_empty():
    ml, _ = _mixin(
        associations=[
            ("Dataset_Dataset", "Dataset", "Nested_Dataset"),
            ("Dataset_Image", "Image", "Image"),
        ],
        features_by_table={},  # no features anywhere
        grouped_rows_by_feature={},
    )
    assert ml.find_feature_producers(_rid(500)) == []
