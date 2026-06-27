"""Unit test for list_input_datasets_with_versions.

Mocks the pathBuilder Dataset_Execution fetch and lookup_dataset so the
(Dataset, consumed_version) pairing is exercised offline.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from deriva_ml.core.mixins.execution import ExecutionMixin
from deriva_ml.execution._helpers import list_input_datasets_with_versions


def _make_ml(rows):
    """Build a fake ml_instance whose Dataset_Execution fetch returns `rows`."""
    entities = MagicMock()
    entities.fetch = lambda: rows
    de_path = MagicMock()
    de_path.filter.return_value = MagicMock(entities=lambda: entities)
    schema = MagicMock()
    schema.Dataset_Execution = de_path
    pb = MagicMock()
    pb.schemas = {"deriva-ml": schema}
    ml = MagicMock()
    ml.ml_schema = "deriva-ml"
    ml.pathBuilder.return_value = pb
    # lookup_dataset returns a stand-in carrying its rid so the test can assert.
    ml.lookup_dataset = lambda rid: SimpleNamespace(dataset_rid=rid)
    return ml


def test_pairs_dataset_with_consumed_version():
    ml = _make_ml(
        [
            {"Dataset": "1-DSAA", "Dataset_Version": "1.0.0", "Execution": "2-EXAA"},
            {"Dataset": "1-DSAB", "Dataset_Version": "2.3.0", "Execution": "2-EXAA"},
        ]
    )
    result = list_input_datasets_with_versions(ml_instance=ml, execution_rid="2-EXAA")
    pairs = {(ds.dataset_rid, v) for ds, v in result}
    assert pairs == {("1-DSAA", "1.0.0"), ("1-DSAB", "2.3.0")}


def test_version_none_when_edge_has_no_pin():
    ml = _make_ml([{"Dataset": "1-DSAA", "Execution": "2-EXAA"}])  # no Dataset_Version key
    result = list_input_datasets_with_versions(ml_instance=ml, execution_rid="2-EXAA")
    assert len(result) == 1
    ds, version = result[0]
    assert ds.dataset_rid == "1-DSAA"
    assert version is None


def test_skips_rows_without_dataset():
    ml = _make_ml([{"Dataset": None, "Dataset_Version": "1.0.0", "Execution": "2-EXAA"}])
    result = list_input_datasets_with_versions(ml_instance=ml, execution_rid="2-EXAA")
    assert result == []


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
