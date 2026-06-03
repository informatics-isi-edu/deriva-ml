"""Integration tests for ``find_executions(dataset=..., dataset_role=...)``.

Catalog-required (except the ValueError guard, which raises before any
catalog work). Validates the dataset filter against the
authorship-canonical model:

  - Output edges (PRODUCED) live in ``Dataset_Version.Execution``.
  - Input edges (CONSUMED) live in ``Dataset_Execution`` (with an
    optional ``Dataset_Version`` FK pinning the consumed version).

Also a regression that a dataset a *pure producer* created does NOT
show up in that producer's ``list_input_datasets()``.
"""

from __future__ import annotations

import pytest

from deriva_ml import MLVocab as vc
from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.execution.execution_configuration import ExecutionConfiguration


def _setup_workflow(ml):
    """Register the vocab terms and a workflow used by the tests."""
    ml.add_term(vc.workflow_type, "DS Filter Workflow", description="find_executions dataset filter tests")
    ml.add_term(vc.dataset_type, "TestSet", description="A test dataset type")
    return ml.create_workflow(
        name="ds_filter_test",
        workflow_type="DS Filter Workflow",
        description="Workflow for find_executions dataset filter tests",
    )


@pytest.fixture
def producer_and_consumer(test_ml):
    """Create a producer execution + dataset and a consumer execution.

    Returns ``(ml, producer_rid, consumer_rid, dataset_rid, version)``:

      - ``producer`` created ``dataset`` via ``create_dataset`` (output
        edge in ``Dataset_Version.Execution``).
      - ``consumer`` consumed ``dataset`` via ``add_input_dataset`` at
        ``dataset.current_version`` (input edge in ``Dataset_Execution``
        with the version pinned on ``Dataset_Version``).
    """
    ml = test_ml
    workflow = _setup_workflow(ml)

    producer = ml.create_execution(
        ExecutionConfiguration(description="Producer", workflow=workflow)
    )
    dataset = producer.create_dataset(dataset_types=["TestSet"], description="Produced dataset")
    version = dataset.current_version

    consumer = ml.create_execution(
        ExecutionConfiguration(description="Consumer", workflow=workflow)
    )
    consumer.add_input_dataset(dataset.dataset_rid, version=version)

    return ml, producer.execution_rid, consumer.execution_rid, dataset.dataset_rid, version


@pytest.mark.integration
def test_find_executions_input_returns_consumer(producer_and_consumer):
    """dataset_role='input' returns the execution that CONSUMED the dataset."""
    ml, producer_rid, consumer_rid, dataset_rid, _version = producer_and_consumer
    rids = {r.execution_rid for r in ml.find_executions(dataset=dataset_rid, dataset_role="input")}
    assert consumer_rid in rids
    assert producer_rid not in rids


@pytest.mark.integration
def test_find_executions_output_returns_producer(producer_and_consumer):
    """dataset_role='output' returns the execution that PRODUCED the dataset."""
    ml, producer_rid, consumer_rid, dataset_rid, _version = producer_and_consumer
    rids = {r.execution_rid for r in ml.find_executions(dataset=dataset_rid, dataset_role="output")}
    assert producer_rid in rids
    assert consumer_rid not in rids


@pytest.mark.integration
def test_find_executions_any_returns_union(producer_and_consumer):
    """dataset_role='any' returns both producer and consumer."""
    ml, producer_rid, consumer_rid, dataset_rid, _version = producer_and_consumer
    rids = {r.execution_rid for r in ml.find_executions(dataset=dataset_rid, dataset_role="any")}
    assert producer_rid in rids
    assert consumer_rid in rids


@pytest.mark.integration
def test_find_executions_dataset_spec_version_pin(producer_and_consumer):
    """A DatasetSpec(rid, version) input filter matches only the consumed version."""
    ml, _producer_rid, consumer_rid, dataset_rid, version = producer_and_consumer

    # The consumed version matches -> consumer is returned.
    spec = DatasetSpec(rid=dataset_rid, version=str(version))
    rids = {r.execution_rid for r in ml.find_executions(dataset=spec, dataset_role="input")}
    assert consumer_rid in rids

    # A version the dataset never had -> no input executions match.
    bogus = DatasetSpec(rid=dataset_rid, version="99.0.0")
    rids_bogus = {r.execution_rid for r in ml.find_executions(dataset=bogus, dataset_role="input")}
    assert consumer_rid not in rids_bogus


def test_find_executions_role_without_dataset_raises(test_ml):
    """dataset_role without a dataset argument raises ValueError early."""
    with pytest.raises(ValueError, match="dataset_role requires a dataset"):
        list(test_ml.find_executions(dataset_role="input"))


@pytest.mark.integration
def test_produced_dataset_not_in_inputs(producer_and_consumer):
    """Regression: a dataset a producer created is NOT among its inputs.

    The producer's ``list_input_datasets()`` reads ``Dataset_Execution``
    (input-only). Since the producer recorded no input edge, the dataset
    it produced must not appear.
    """
    ml, producer_rid, _consumer_rid, dataset_rid, _version = producer_and_consumer
    producer = ml.lookup_execution(producer_rid)
    input_rids = {d.dataset_rid for d in producer.list_input_datasets()}
    assert dataset_rid not in input_rids
