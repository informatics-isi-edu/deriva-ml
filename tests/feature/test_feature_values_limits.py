"""Integration tests for the materialize_limit= and execution_rids=
parameters on ``feature_values`` (catalog FeatureMixin).
"""

from __future__ import annotations

import pytest

from deriva_ml import BuiltinTypes, ColumnDefinition
from deriva_ml.core.exceptions import DerivaMLMaterializeLimitExceeded
from deriva_ml.execution import ExecutionConfiguration


@pytest.fixture
def catalog_with_feature_values(populated_catalog):
    """Create a Feature on Image and populate values from 2 executions.

    Returns the DerivaML instance with the feature populated.
    """
    ml = populated_catalog
    feature_name = "Quality"

    ml.create_feature(
        target_table="Image",
        feature_name=feature_name,
        metadata=[ColumnDefinition(name="score", type=BuiltinTypes.float8)],
    )

    # Find a few image RIDs to attach feature values to
    image_rids = [r["RID"] for r in ml._domain_path().tables["Image"].entities().fetch()][:3]
    if not image_rids:
        pytest.skip("populated_catalog produced no Image rows")

    record_class = ml.feature_record_class("Image", feature_name)

    # Run 2 distinct executions sharing one workflow.  Each create_execution
    # call produces a new Execution row (distinct RID) even when the workflow
    # is the same -- that is all we need for the execution_rids filter tests.
    workflow = ml.create_workflow(
        name="fvl_test_workflow",
        workflow_type="Test Workflow",
    )
    for i in range(2):
        execution = ml.create_execution(
            ExecutionConfiguration(
                description=f"fvl-execution-{i}",
                workflow=workflow,
            )
        )
        with execution.execute() as exe:
            for img_rid in image_rids:
                exe.add_features([record_class(Image=img_rid, score=0.5 + i * 0.1)])

    return ml


@pytest.mark.integration
def test_feature_values_materialize_limit_not_exceeded(catalog_with_feature_values):
    """When the row count is below the limit, no exception is raised."""
    ml = catalog_with_feature_values
    records = list(ml.feature_values("Image", "Quality", materialize_limit=10_000))
    assert isinstance(records, list)
    assert len(records) > 0


@pytest.mark.integration
def test_feature_values_materialize_limit_exceeded_raises(catalog_with_feature_values):
    """When the row count exceeds the limit, raises DerivaMLMaterializeLimitExceeded."""
    ml = catalog_with_feature_values
    with pytest.raises(DerivaMLMaterializeLimitExceeded) as exc_info:
        list(ml.feature_values("Image", "Quality", materialize_limit=0))
    assert exc_info.value.limit == 0
    assert exc_info.value.actual_count > 0


@pytest.mark.integration
def test_feature_values_execution_rids_filters_results(catalog_with_feature_values):
    """execution_rids= restricts results to the named executions only."""
    ml = catalog_with_feature_values
    all_records = list(ml.feature_values("Image", "Quality"))
    all_exec_rids = sorted({r.Execution for r in all_records if r.Execution})
    assert len(all_exec_rids) >= 2, "fixture should produce records from at least 2 executions"
    target_rid = all_exec_rids[0]
    filtered = list(ml.feature_values("Image", "Quality", execution_rids=[target_rid]))
    assert len(filtered) > 0
    assert all(r.Execution == target_rid for r in filtered)
    assert len(filtered) < len(all_records)


@pytest.mark.integration
def test_feature_values_execution_rids_empty_list_returns_nothing(catalog_with_feature_values):
    """execution_rids=[] short-circuits to an empty result."""
    ml = catalog_with_feature_values
    records = list(ml.feature_values("Image", "Quality", execution_rids=[]))
    assert records == []


@pytest.mark.integration
def test_feature_values_execution_rids_with_materialize_limit_combine(
    catalog_with_feature_values,
):
    """The two parameters compose -- filter is applied first, limit checked after."""
    ml = catalog_with_feature_values
    all_records = list(ml.feature_values("Image", "Quality"))
    all_exec_rids = sorted({r.Execution for r in all_records if r.Execution})
    target_rid = all_exec_rids[0]
    target_count = sum(1 for r in all_records if r.Execution == target_rid)

    # Limit higher than the filtered count -- should succeed.
    records = list(
        ml.feature_values(
            "Image",
            "Quality",
            execution_rids=[target_rid],
            materialize_limit=target_count + 10,
        )
    )
    assert len(records) == target_count

    # Limit below the filtered count -- should raise.
    with pytest.raises(DerivaMLMaterializeLimitExceeded):
        list(
            ml.feature_values(
                "Image",
                "Quality",
                execution_rids=[target_rid],
                materialize_limit=0,
            )
        )
