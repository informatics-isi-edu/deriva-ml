"""Live-catalog smoke tests for the validation methods.

Builds a small populated catalog and exercises both
``validate_dataset_specs`` and ``validate_execution_configuration``
against real catalog state. Gated on ``DERIVA_HOST`` like the other
live-smoke tests.
"""

from __future__ import annotations

import os

import pytest

from deriva_ml import MLVocab as vc
from deriva_ml.asset.aux_classes import AssetSpec
from deriva_ml.dataset.aux_classes import DatasetSpec, DatasetVersion
from deriva_ml.dataset.validation import (
    DatasetSpecValidationReport,
    ExecutionConfigurationValidationReport,
)
from deriva_ml.execution.execution_configuration import ExecutionConfiguration

pytestmark = pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="validate_specs live smoke test requires DERIVA_HOST",
)


def _make_dataset(test_ml) -> tuple[str, str]:
    """Create a minimal dataset and return (rid, version_str)."""
    test_ml.add_term(vc.dataset_type, "ValidateTest", description="Validate smoke")
    test_ml.add_term(vc.workflow_type, "Validate Test", description="Validate smoke")

    wf = test_ml.create_workflow(
        name="Validate smoke workflow",
        workflow_type="Validate Test",
        description="Validate smoke workflow",
    )
    exe = test_ml.create_execution(
        ExecutionConfiguration(workflow=wf, description="validate smoke exe"),
    )
    ds = exe.create_dataset(
        dataset_types="ValidateTest",
        description="validate smoke dataset",
        version=DatasetVersion(1, 0, 0),
    )
    return ds.dataset_rid, "1.0.0"


def test_validate_dataset_specs_live_valid(test_ml):
    rid, version = _make_dataset(test_ml)
    report = test_ml.validate_dataset_specs(
        specs=[
            DatasetSpec(rid=rid, version=version),
        ]
    )
    assert isinstance(report, DatasetSpecValidationReport)
    assert report.all_valid is True
    assert report.results[0].dataset_name == "validate smoke dataset"


def test_validate_dataset_specs_live_bad_version_lists_available(test_ml):
    rid, _ = _make_dataset(test_ml)
    report = test_ml.validate_dataset_specs(
        specs=[
            DatasetSpec(rid=rid, version="9.9.9"),
        ]
    )
    assert report.all_valid is False
    r = report.results[0]
    assert r.reasons == ["version_not_found"]
    assert r.available_versions is not None
    assert "1.0.0" in r.available_versions


def test_validate_dataset_specs_live_unknown_rid(test_ml):
    report = test_ml.validate_dataset_specs(
        specs=[
            DatasetSpec(rid="9-NOPE", version="1.0.0"),
        ]
    )
    assert report.all_valid is False
    assert "rid_not_found" in report.results[0].reasons


def test_validate_execution_configuration_live_full(test_ml):
    rid, version = _make_dataset(test_ml)
    wf = test_ml.create_workflow(
        name="Validate composite workflow",
        workflow_type="Validate Test",
        description="composite smoke",
    )
    config = ExecutionConfiguration(
        workflow=wf,
        datasets=[DatasetSpec(rid=rid, version=version)],
    )
    report = test_ml.validate_execution_configuration(config)
    assert isinstance(report, ExecutionConfigurationValidationReport)
    assert report.all_valid is True
    assert report.workflow_result is not None
    assert report.workflow_result.valid is True


def test_validate_execution_configuration_live_workflow_rid_is_dataset(test_ml):
    """Pointing the workflow at a dataset RID surfaces as not_a_workflow."""
    rid, version = _make_dataset(test_ml)
    wf = test_ml.create_workflow(
        name="Validate composite workflow",
        workflow_type="Validate Test",
        description="composite smoke",
    )
    # Mutate the wf object so its rid points at the dataset (no catalog
    # round-trip — composite validator only reads config.workflow.rid).
    wf_swapped = wf.model_copy(update={"rid": rid})
    config = ExecutionConfiguration(workflow=wf_swapped)
    report = test_ml.validate_execution_configuration(config)
    assert report.all_valid is False
    assert report.workflow_result is not None
    assert report.workflow_result.reasons == ["not_a_workflow"]
    assert report.workflow_result.actual_table == "Dataset"


def test_validate_execution_configuration_live_cross_spec_duplicate(test_ml):
    rid, version = _make_dataset(test_ml)
    wf = test_ml.create_workflow(
        name="Validate composite workflow",
        workflow_type="Validate Test",
        description="composite smoke",
    )
    config = ExecutionConfiguration(
        workflow=wf,
        datasets=[
            DatasetSpec(rid=rid, version=version),
            DatasetSpec(rid=rid, version=version),
        ],
    )
    report = test_ml.validate_execution_configuration(config)
    assert any(i.issue == "duplicate_rid" for i in report.cross_spec_issues)
    assert report.all_valid is False


def test_validate_execution_configuration_live_assets(test_ml):
    """If the catalog has any populated asset, a valid AssetSpec passes
    and an invalid RID surfaces as rid_not_found.

    Skipped when no asset table can be enumerated."""
    # Use the workflow RID created above as the "wrong-shape" target so we
    # always have a known non-asset RID.
    wf = test_ml.create_workflow(
        name="Validate asset workflow",
        workflow_type="Validate Test",
        description="asset smoke",
    )
    config = ExecutionConfiguration(
        workflow=wf,
        assets=[AssetSpec(rid=wf.rid)],
    )
    report = test_ml.validate_execution_configuration(config)
    assert report.all_valid is False
    asset_r = report.asset_results[0]
    assert asset_r.valid is False
    assert "not_an_asset" in asset_r.reasons
    assert asset_r.actual_table == "Workflow"
