"""Live-catalog tests for forward lineage + multirun status summary (issue #76).

``find_executions_consuming`` is the forward complement of
``lookup_lineage`` (which walks backward): given a Dataset or asset
RID, which executions CONSUMED it? Needed for "is it safe to delete
this?" questions. ``multirun_status_summary`` aggregates execution
status counts for one workflow in a single query -- the "is the sweep
done?" question.

Builds the same two-execution chain as the lookup_lineage live smoke
test: exe1 produces ds1; exe2 consumes ds1 and produces ds2.
"""

from __future__ import annotations

import os

import pytest

from deriva_ml import DerivaMLException
from deriva_ml import MLVocab as vc
from deriva_ml.dataset.aux_classes import DatasetSpec, DatasetVersion
from deriva_ml.execution.execution_configuration import ExecutionConfiguration

pytestmark = pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="forward-lineage live tests require DERIVA_HOST",
)


@pytest.fixture
def chain(test_ml):
    """exe1 -> ds1 -> exe2 -> ds2 under one workflow."""
    test_ml.add_term(vc.dataset_type, "FwdLineageTest", description="Forward lineage smoke")
    test_ml.add_term(vc.workflow_type, "Fwd Lineage Test", description="Forward lineage smoke")

    wf = test_ml.create_workflow(
        name="Forward lineage smoke workflow",
        workflow_type="Fwd Lineage Test",
        description="Forward lineage smoke test workflow",
    )

    exe1 = test_ml.create_execution(
        ExecutionConfiguration(workflow=wf, description="fwd lineage exe1"),
    )
    ds1 = exe1.create_dataset(
        dataset_types="FwdLineageTest",
        description="fwd lineage smoke ds1",
        version=DatasetVersion(1, 0, 0),
    )

    exe2 = test_ml.create_execution(
        ExecutionConfiguration(
            workflow=wf,
            description="fwd lineage exe2",
            datasets=[DatasetSpec(rid=ds1.dataset_rid, version=ds1.current_version)],
        ),
    )
    ds2 = exe2.create_dataset(
        dataset_types="FwdLineageTest",
        description="fwd lineage smoke ds2",
        version=DatasetVersion(1, 0, 0),
    )
    # create_workflow returns a LOCAL spec object (no RID -- the catalog
    # row is registered lazily when an execution uses it). Read the
    # registered Workflow RID off the execution row.
    wf_rid = test_ml._retrieve_rid(exe1.execution_rid)["Workflow"]
    return test_ml, wf_rid, exe1, ds1, exe2, ds2


class TestFindExecutionsConsuming:
    def test_consumer_found_producer_excluded(self, chain):
        """exe2 consumed ds1; exe1 (the producer) is not a consumer."""
        ml, _wf_rid, exe1, ds1, exe2, _ds2 = chain
        consuming = ml.find_executions_consuming(ds1.dataset_rid)
        rids = {e.execution_rid for e in consuming}
        assert exe2.execution_rid in rids
        assert exe1.execution_rid not in rids

    def test_unconsumed_dataset_is_safe_to_delete(self, chain):
        """ds2 (leaf) has no consumers -- the 'safe to delete?' answer."""
        ml, *_rest, ds2 = chain
        assert ml.find_executions_consuming(ds2.dataset_rid) == []

    def test_asset_rid_dispatches(self, chain):
        """An asset RID routes through the asset Input-edge path."""
        ml = chain[0]
        images = ml.list_assets("Image")
        if not images:
            pytest.skip("demo catalog has no Image assets in this state")
        result = ml.find_executions_consuming(images[0].asset_rid)
        assert isinstance(result, list)  # may be empty; dispatch must not raise

    def test_non_artifact_rid_raises(self, chain):
        """A Workflow RID is not lineage-shaped -> clear DerivaMLException."""
        ml, wf_rid, *_rest = chain
        with pytest.raises(DerivaMLException):
            ml.find_executions_consuming(wf_rid)


class TestMultirunStatusSummary:
    def test_counts_match_executions(self, chain):
        """Summary totals agree with find_executions for the workflow."""
        ml, wf_rid, *_rest = chain
        summary = ml.multirun_status_summary(wf_rid)
        assert summary.workflow_rid == wf_rid
        assert summary.total == sum(summary.counts.values())
        listed = list(ml.find_executions(workflow=wf_rid))
        assert summary.total == len(listed) >= 2

    def test_unknown_workflow_raises(self, chain):
        ml = chain[0]
        with pytest.raises(DerivaMLException):
            ml.multirun_status_summary("9-NOPE")
