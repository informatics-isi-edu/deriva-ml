"""Live-catalog smoke test for ``DerivaML.lookup_lineage``.

Builds a two-execution data-flow chain against a real catalog and
walks the lineage of the leaf dataset back to the root execution.
Gated on ``DERIVA_HOST`` like the other live-smoke tests in this
directory.
"""

from __future__ import annotations

import os

import pytest

from deriva_ml import MLVocab as vc
from deriva_ml.dataset.aux_classes import DatasetSpec, DatasetVersion
from deriva_ml.execution.execution_configuration import ExecutionConfiguration
from deriva_ml.execution.lineage import LineageResult


@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="lookup_lineage live smoke test requires DERIVA_HOST",
)
def test_lookup_lineage_two_execution_chain(test_ml):
    """exe1 -> ds1 -> exe2 -> ds2; walk from ds2 back to exe1."""
    test_ml.add_term(vc.dataset_type, "LineageTest", description="Lineage smoke")
    test_ml.add_term(vc.workflow_type, "Lineage Test", description="Lineage smoke")

    wf = test_ml.create_workflow(
        name="Lineage smoke workflow",
        workflow_type="Lineage Test",
        description="Lineage smoke test workflow",
    )

    # Execution 1 produces ds1.
    exe1 = test_ml.create_execution(
        ExecutionConfiguration(workflow=wf, description="lineage exe1"),
    )
    ds1 = exe1.create_dataset(
        dataset_types="LineageTest",
        description="lineage smoke ds1",
        version=DatasetVersion(1, 0, 0),
    )

    # Execution 2 consumes ds1 as input and produces ds2.
    exe2 = test_ml.create_execution(
        ExecutionConfiguration(
            workflow=wf,
            description="lineage exe2",
            datasets=[DatasetSpec(rid=ds1.dataset_rid, version=ds1.current_version)],
        ),
    )
    ds2 = exe2.create_dataset(
        dataset_types="LineageTest",
        description="lineage smoke ds2",
        version=DatasetVersion(1, 0, 0),
    )

    # Walk lineage from ds2.
    result = test_ml.lookup_lineage(ds2.dataset_rid)

    assert isinstance(result, LineageResult)
    assert result.walked_complete is True
    assert result.cycle_detected is False
    assert result.root.type == "Dataset"
    assert result.root.rid == ds2.dataset_rid

    # Producing execution of ds2 is exe2.
    assert result.root.producing_execution is not None
    assert result.root.producing_execution.rid == exe2.execution_rid
    assert result.lineage is not None
    assert result.lineage.execution.rid == exe2.execution_rid

    # exe2 consumed ds1.
    consumed_rids = {ds.rid for ds in result.lineage.consumed_datasets}
    assert ds1.dataset_rid in consumed_rids

    # exe1 should appear as a parent of exe2.
    parent_rids = {p.execution.rid for p in result.lineage.parents}
    assert exe1.execution_rid in parent_rids
    assert result.executions_visited >= 2


@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="lookup_lineage live smoke test requires DERIVA_HOST",
)
def test_lookup_lineage_dataset_with_no_producer(test_ml):
    """A Dataset with no Dataset_Version row at all has no producer."""
    test_ml.add_term(vc.dataset_type, "LineageOrphan", description="Orphan smoke")

    # Use the Dataset module directly to skip the version-creation path.
    from deriva_ml.dataset.dataset import Dataset

    rid = (
        Dataset._create_dataset_record(  # type: ignore[attr-defined]
            ml_instance=test_ml,
            dataset_types=["LineageOrphan"],
            description="lineage orphan smoke",
            execution_rid=None,
        )
        if hasattr(Dataset, "_create_dataset_record")
        else None
    )

    if rid is None:
        pytest.skip(
            "Cannot create a dataset without a version row in this build; no public API exposes the producer-less path."
        )

    result = test_ml.lookup_lineage(rid)
    assert result.root.type == "Dataset"
    assert result.root.producing_execution is None
    assert result.lineage is None
    assert result.walked_complete is True
