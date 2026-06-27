"""Unit test for list_input_datasets_with_versions.

Mocks the pathBuilder Dataset_Execution fetch and lookup_dataset so the
(Dataset, consumed_version) pairing is exercised offline.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from deriva_ml.core.mixins.execution import ExecutionMixin
from deriva_ml.execution._helpers import list_input_datasets_with_versions


def _make_ml(de_rows, version_rows=None):
    """Build a fake ml_instance whose Dataset_Execution and Dataset_Version fetches return realistic data.

    Args:
        de_rows: Rows to return from Dataset_Execution.filter(...).entities().fetch().
            ``Dataset_Version`` values in each row must be RIDs (e.g. ``"VR1"``),
            not version strings — faithfully modelling what ERMrest returns for an FK
            column.
        version_rows: Rows to return from the bounded Dataset_Version fetch. Each row
            must have ``"RID"`` and ``"Version"`` keys so the helper can build the
            RID → version-string map.  Defaults to an empty list (no version rows).

    Returns:
        A MagicMock ``ml`` object.  After calling
        :func:`list_input_datasets_with_versions`, inspect ``ml._dv_calls``:

        - ``ml._dv_calls["in_rids"]``: list of RID lists passed to ``.in_()``, one
          entry per chunk.
        - ``ml._dv_calls["unfiltered_fetch"]``: ``True`` if the UNFILTERED
          ``dv_path.entities()`` path was hit (the bounded helper must NEVER set this).
    """
    if version_rows is None:
        version_rows = []

    # Dataset_Execution path — filter(...).entities() -> fetch() returns de_rows.
    de_entities = MagicMock()
    de_entities.fetch = lambda: de_rows
    de_path = MagicMock()
    de_path.filter.return_value = MagicMock(entities=lambda: de_entities)

    # Dataset_Version path — the bounded helper calls
    # dv_path.filter(dv_path.RID.in_(chunk)).entities().fetch(). Record the
    # RIDs requested via .in_() and serve only matching version_rows. Also
    # expose a flag for whether the UNFILTERED entities() path was hit.
    version_by_rid = {r["RID"]: r for r in version_rows}
    calls = {"in_rids": [], "unfiltered_fetch": False}

    def _dv_in(rids):
        rids = list(rids)
        calls["in_rids"].append(rids)
        matched = [version_by_rid[r] for r in rids if r in version_by_rid]
        filtered_entities = MagicMock()
        filtered_entities.fetch = lambda: matched
        filtered_path = MagicMock()
        filtered_path.entities = lambda: filtered_entities
        return filtered_path

    dv_rid_col = MagicMock()
    dv_rid_col.in_ = _dv_in

    def _dv_unfiltered_entities():
        calls["unfiltered_fetch"] = True
        e = MagicMock()
        e.fetch = lambda: version_rows
        return e

    dv_path = MagicMock()
    dv_path.RID = dv_rid_col
    dv_path.filter = lambda predicate: predicate  # predicate IS the filtered_path from _dv_in
    dv_path.entities = _dv_unfiltered_entities

    schema = MagicMock()
    schema.Dataset_Execution = de_path
    schema.tables = {"Dataset_Version": dv_path}

    pb = MagicMock()
    pb.schemas = {"deriva-ml": schema}

    ml = MagicMock()
    ml.ml_schema = "deriva-ml"
    ml.pathBuilder.return_value = pb
    # lookup_dataset returns a stand-in carrying its rid so the test can assert.
    ml.lookup_dataset = lambda rid: SimpleNamespace(dataset_rid=rid)
    ml._dv_calls = calls
    return ml


def test_pairs_dataset_with_consumed_version():
    """Consumed-version RIDs are resolved to version strings via the Dataset_Version map."""
    ml = _make_ml(
        de_rows=[
            {"Dataset": "1-DSAA", "Dataset_Version": "VR1", "Execution": "2-EXAA"},
            {"Dataset": "1-DSAB", "Dataset_Version": "VR2", "Execution": "2-EXAA"},
        ],
        version_rows=[
            {"RID": "VR1", "Version": "1.0.0"},
            {"RID": "VR2", "Version": "2.3.0"},
        ],
    )
    result = list_input_datasets_with_versions(ml_instance=ml, execution_rid="2-EXAA")
    pairs = {(ds.dataset_rid, v) for ds, v in result}
    assert pairs == {("1-DSAA", "1.0.0"), ("1-DSAB", "2.3.0")}
    assert ml._dv_calls["unfiltered_fetch"] is False


def test_version_none_when_edge_has_no_pin():
    """An edge with no Dataset_Version key yields consumed_version=None."""
    ml = _make_ml(
        de_rows=[{"Dataset": "1-DSAA", "Execution": "2-EXAA"}],  # no Dataset_Version key
        version_rows=[],
    )
    result = list_input_datasets_with_versions(ml_instance=ml, execution_rid="2-EXAA")
    assert len(result) == 1
    ds, version = result[0]
    assert ds.dataset_rid == "1-DSAA"
    assert version is None
    assert ml._dv_calls["in_rids"] == []
    assert ml._dv_calls["unfiltered_fetch"] is False


def test_skips_rows_without_dataset():
    """Rows where Dataset is None are filtered out."""
    ml = _make_ml(
        de_rows=[{"Dataset": None, "Dataset_Version": "VR1", "Execution": "2-EXAA"}],
        version_rows=[{"RID": "VR1", "Version": "1.0.0"}],
    )
    result = list_input_datasets_with_versions(ml_instance=ml, execution_rid="2-EXAA")
    assert result == []
    assert ml._dv_calls["unfiltered_fetch"] is False


def test_missing_fk_rid_resolves_to_none():
    """A consumed-version RID absent from Dataset_Version yields None, not a crash."""
    ml = _make_ml(
        de_rows=[{"Dataset": "1-DSAA", "Dataset_Version": "VR_MISSING", "Execution": "2-EXAA"}],
        version_rows=[{"RID": "VR1", "Version": "1.0.0"}],  # VR_MISSING not present
    )
    result = list_input_datasets_with_versions(ml_instance=ml, execution_rid="2-EXAA")
    assert result == [] or result  # non-empty; assert the pair below
    ds, version = result[0]
    assert ds.dataset_rid == "1-DSAA"
    assert version is None
    assert ml._dv_calls["unfiltered_fetch"] is False


def test_mixed_pinned_and_unpinned_inputs():
    """One pinned + one unpinned edge in the same execution both survive."""
    ml = _make_ml(
        de_rows=[
            {"Dataset": "1-DSAA", "Dataset_Version": "VR1", "Execution": "2-EXAA"},
            {"Dataset": "1-DSAB", "Execution": "2-EXAA"},  # no Dataset_Version key
        ],
        version_rows=[{"RID": "VR1", "Version": "1.0.0"}],
    )
    result = list_input_datasets_with_versions(ml_instance=ml, execution_rid="2-EXAA")
    pairs = {(ds.dataset_rid, v) for ds, v in result}
    assert pairs == {("1-DSAA", "1.0.0"), ("1-DSAB", None)}
    # Only the pinned RID was requested via .in_().
    assert ml._dv_calls["in_rids"] == [["VR1"]]
    assert ml._dv_calls["unfiltered_fetch"] is False


def test_empty_inputs_never_fetches_versions():
    """No input edges -> [] and the Dataset_Version table is never touched."""
    ml = _make_ml(de_rows=[], version_rows=[{"RID": "VR1", "Version": "1.0.0"}])
    result = list_input_datasets_with_versions(ml_instance=ml, execution_rid="2-EXAA")
    assert result == []
    assert ml._dv_calls["in_rids"] == []
    assert ml._dv_calls["unfiltered_fetch"] is False


def test_bounded_fetch_requests_only_consumed_rids():
    """The version fetch is filtered to exactly the distinct consumed RIDs."""
    ml = _make_ml(
        de_rows=[
            {"Dataset": "1-DSAA", "Dataset_Version": "VR1", "Execution": "2-EXAA"},
            {"Dataset": "1-DSAB", "Dataset_Version": "VR2", "Execution": "2-EXAA"},
        ],
        version_rows=[{"RID": "VR1", "Version": "1.0.0"}, {"RID": "VR2", "Version": "2.0.0"}],
    )
    list_input_datasets_with_versions(ml_instance=ml, execution_rid="2-EXAA")
    # One chunk, containing exactly the two consumed RIDs (order-insensitive).
    assert len(ml._dv_calls["in_rids"]) == 1
    assert set(ml._dv_calls["in_rids"][0]) == {"VR1", "VR2"}
    assert ml._dv_calls["unfiltered_fetch"] is False


def _ml_with_versions(version_rows):
    """Fake ExecutionMixin host whose Dataset_Version fetch returns version_rows."""
    entities = MagicMock()
    entities.fetch = lambda: version_rows
    vp = MagicMock()
    vp.filter.return_value = MagicMock(entities=lambda: entities)
    tables = {"Dataset_Version": vp}
    schema = MagicMock()
    schema.tables = tables
    pb = MagicMock()
    pb.schemas = {"deriva-ml": schema}
    ml = ExecutionMixin.__new__(ExecutionMixin)
    ml.pathBuilder = lambda: pb
    ml.ml_schema = "deriva-ml"
    return ml


def test_producer_of_dataset_latest_when_version_none():
    ml = _ml_with_versions(
        [
            {"Version": "1.0.0", "Execution": "2-EXV1", "Dataset": "1-DSAA"},
            {"Version": "1.2.0", "Execution": "2-EXV2", "Dataset": "1-DSAA"},
        ]
    )
    assert ml._producer_of_dataset("1-DSAA") == "2-EXV2"  # latest


def test_producer_of_dataset_specific_version():
    ml = _ml_with_versions(
        [
            {"Version": "1.0.0", "Execution": "2-EXV1", "Dataset": "1-DSAA"},
            {"Version": "1.2.0", "Execution": "2-EXV2", "Dataset": "1-DSAA"},
        ]
    )
    assert ml._producer_of_dataset("1-DSAA", version="1.0.0") == "2-EXV1"  # consumed


def test_producer_of_dataset_missing_version_returns_none():
    ml = _ml_with_versions(
        [
            {"Version": "1.0.0", "Execution": "2-EXV1", "Dataset": "1-DSAA"},
        ]
    )
    assert ml._producer_of_dataset("1-DSAA", version="9.9.9") is None


def test_input_dataset_pairs_forwards_to_helper(monkeypatch):
    """ExecutionMixin._input_dataset_pairs calls the module helper with
    ml_instance=self and the given execution_rid (guards the real seam wiring
    that otherwise only the live test exercises)."""
    captured = {}

    def _fake_helper(*, ml_instance, execution_rid):
        captured["ml_instance"] = ml_instance
        captured["execution_rid"] = execution_rid
        return [("sentinel-ds", "1.0.0")]

    monkeypatch.setattr("deriva_ml.execution._helpers.list_input_datasets_with_versions", _fake_helper)

    ml = ExecutionMixin.__new__(ExecutionMixin)
    result = ml._input_dataset_pairs("2-EXAA")

    assert captured["ml_instance"] is ml
    assert captured["execution_rid"] == "2-EXAA"
    assert result == [("sentinel-ds", "1.0.0")]
