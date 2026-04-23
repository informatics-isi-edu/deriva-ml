"""Tests for the unified feature_values read surface.

This task (Task 4) covers DerivaML online reads. The parametrized symmetry
suite across DerivaML / Dataset / DatasetBag is added in Task 8 once all
three containers have the method.
"""
from __future__ import annotations

import dataclasses
import time

import pytest

from deriva_ml import BuiltinTypes, ColumnDefinition
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.execution import ExecutionConfiguration
from deriva_ml.feature import FeatureRecord


@dataclasses.dataclass
class FeatureFixture:
    """Container for a catalog seeded with a single feature."""

    ml: object
    feature_name: str
    image_rids: list[str]
    target_table: str = "Image"


@pytest.fixture
def test_ml_with_feature(populated_catalog):
    """DerivaML instance with one Image/Label feature, one value per image."""
    ml = populated_catalog
    feature_name = "Label"

    ml.create_feature(
        target_table="Image",
        feature_name=feature_name,
        metadata=[ColumnDefinition(name="Label", type=BuiltinTypes.text)],
    )

    image_rids = [r["RID"] for r in ml.domain_path().tables["Image"].entities().fetch()]
    assert image_rids, "No Image rows in test catalog"

    workflow = ml.create_workflow(
        name="Label seeder",
        workflow_type="Test Workflow",
    )
    execution = ml.create_execution(
        ExecutionConfiguration(
            description="Seed Label feature values",
            workflow=workflow,
        )
    )
    LabelFeature = ml.feature_record_class("Image", feature_name)
    with execution.execute() as exe:
        for rid in image_rids:
            exe.add_features([LabelFeature(Image=rid, Label="positive")])
    execution.upload_execution_outputs()

    yield FeatureFixture(ml=ml, feature_name=feature_name, image_rids=image_rids)


@pytest.fixture
def test_ml_with_feature_multi(populated_catalog):
    """DerivaML instance with Image/Score feature, multiple values per image."""
    ml = populated_catalog
    feature_name = "Score"

    ml.create_feature(
        target_table="Image",
        feature_name=feature_name,
        metadata=[ColumnDefinition(name="Score", type=BuiltinTypes.text)],
    )

    image_rids = [r["RID"] for r in ml.domain_path().tables["Image"].entities().fetch()]
    assert image_rids, "No Image rows in test catalog"

    ScoreFeature = ml.feature_record_class("Image", feature_name)

    for i, label in enumerate(["high", "low"]):
        workflow = ml.create_workflow(
            name=f"Score seeder round {i}",
            workflow_type="Test Workflow",
        )
        execution = ml.create_execution(
            ExecutionConfiguration(
                description=f"Seed Score feature values round {i}",
                workflow=workflow,
            )
        )
        with execution.execute() as exe:
            for rid in image_rids:
                exe.add_features([ScoreFeature(Image=rid, Score=label)])
        execution.upload_execution_outputs()
        if i == 0:
            time.sleep(0.1)

    yield FeatureFixture(ml=ml, feature_name=feature_name, image_rids=image_rids)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_feature_values_yields_feature_records(test_ml_with_feature) -> None:
    """feature_values yields FeatureRecord instances with the expected attrs."""
    records = list(test_ml_with_feature.ml.feature_values(
        "Image", test_ml_with_feature.feature_name,
    ))
    assert len(records) > 0
    rec = records[0]
    assert isinstance(rec, FeatureRecord)
    # Target RID attribute present and non-empty
    assert getattr(rec, "Image", None)
    # Feature_Name populated
    assert rec.Feature_Name == test_ml_with_feature.feature_name


def test_feature_values_unknown_feature_raises(test_ml) -> None:
    with pytest.raises(DerivaMLException):
        list(test_ml.feature_values("Image", "no_such_feature_xyz"))


def test_feature_values_is_iterable_not_list(test_ml_with_feature) -> None:
    """Return type is an iterator (streaming), not a materialized list."""
    import collections.abc
    result = test_ml_with_feature.ml.feature_values(
        "Image", test_ml_with_feature.feature_name,
    )
    assert isinstance(result, collections.abc.Iterator)


def test_feature_values_with_select_newest(test_ml_with_feature_multi) -> None:
    """Selector reduces multi-value groups to one record per target RID."""
    records = list(test_ml_with_feature_multi.ml.feature_values(
        "Image",
        test_ml_with_feature_multi.feature_name,
        selector=FeatureRecord.select_newest,
    ))
    # After selector: one record per target Image
    rids = [r.Image for r in records]
    assert len(rids) == len(set(rids))


def test_feature_values_selector_returning_none_omits_target(test_ml_with_feature) -> None:
    """Selector returning None removes that target from the iterator."""
    def reject_all(records):
        return None
    records = list(test_ml_with_feature.ml.feature_values(
        "Image", test_ml_with_feature.feature_name, selector=reject_all,
    ))
    assert records == []
