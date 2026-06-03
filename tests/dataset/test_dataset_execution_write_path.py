"""Write-path tests for the authorship-canonical provenance model.

Under the authorship-canonical model:

* **Output edges** (an execution *produced* a dataset version) live ONLY in
  ``Dataset_Version.Execution`` (the per-version author FK). They must NOT be
  written to ``Dataset_Execution``.
* **Input edges** (an execution *consumed* a dataset) live in
  ``Dataset_Execution``, whose new nullable ``Dataset_Version`` FK records the
  consumed version.

These integration tests pin the WRITE path (Task 4):

1. ``create_dataset`` under an execution writes NO ``Dataset_Execution`` row for
   that dataset, while ``_producer_of_dataset`` still resolves to the creating
   execution (via the ``Dataset_Version.Execution`` link).
2. ``add_input_dataset(rid, version=v)`` records the consumed version on the
   ``Dataset_Execution`` row's ``Dataset_Version`` FK.
3. ``add_input_dataset(rid)`` (no version) leaves ``Dataset_Version`` NULL.

Requires a live catalog (``DERIVA_HOST``); each test uses the ``test_ml``
fixture, which resets the session catalog to a clean state.
"""

from __future__ import annotations

import pytest

from deriva_ml import MLVocab as vc
from deriva_ml.dataset.aux_classes import DatasetVersion
from deriva_ml.execution.execution_configuration import ExecutionConfiguration


def _dataset_execution_rows(ml, dataset_rid, execution_rid):
    """Return ``Dataset_Execution`` rows for the (dataset, execution) pair."""
    pb = ml.pathBuilder()
    dataset_exec = pb.schemas[ml.ml_schema].Dataset_Execution
    return [
        row
        for row in dataset_exec.filter(dataset_exec.Execution == execution_rid).entities().fetch()
        if row.get("Dataset") == dataset_rid
    ]


def _make_execution(ml, *, datasets=None):
    """Create a workflow + execution; return the execution context."""
    ml.add_term(vc.workflow_type, "WritePath Test", description="write-path smoke")
    wf = ml.create_workflow(
        name="WritePath smoke workflow",
        workflow_type="WritePath Test",
        description="write-path smoke test workflow",
    )
    return ml.create_execution(
        ExecutionConfiguration(
            workflow=wf,
            description="write-path exe",
            datasets=datasets or [],
        ),
    )


@pytest.mark.integration
def test_create_dataset_writes_no_dataset_execution_row(test_ml):
    """create_dataset records the producer via Dataset_Version, not Dataset_Execution."""
    test_ml.add_term(vc.dataset_type, "WritePathTest", description="write-path smoke")
    exe = _make_execution(test_ml)
    ds = exe.create_dataset(
        dataset_types="WritePathTest",
        description="write-path created dataset",
        version=DatasetVersion(1, 0, 0),
    )

    # No Dataset_Execution row for the produced dataset.
    rows = _dataset_execution_rows(test_ml, ds.dataset_rid, exe.execution_rid)
    assert rows == [], (
        "create_dataset must NOT write a Dataset_Execution row for an output dataset; "
        f"found {rows!r}"
    )

    # Producer is recorded via the Dataset_Version.Execution link.
    assert test_ml._producer_of_dataset(ds.dataset_rid) == exe.execution_rid


@pytest.mark.integration
def test_add_input_dataset_with_version_records_dataset_version(test_ml):
    """add_input_dataset(rid, version=v) sets the Dataset_Version FK on the input edge."""
    test_ml.add_term(vc.dataset_type, "WritePathTest", description="write-path smoke")

    # Producer execution creates the source dataset.
    producer = _make_execution(test_ml)
    ds = producer.create_dataset(
        dataset_types="WritePathTest",
        description="source dataset",
        version=DatasetVersion(1, 0, 0),
    )

    # Consumer execution records the dataset as an input at a specific version.
    consumer = _make_execution(test_ml)
    consumer.add_input_dataset(ds.dataset_rid, version=ds.current_version)

    rows = _dataset_execution_rows(test_ml, ds.dataset_rid, consumer.execution_rid)
    assert len(rows) == 1, f"expected exactly one input edge, found {rows!r}"
    assert rows[0].get("Dataset_Version") is not None, (
        "add_input_dataset(version=...) must set the Dataset_Version FK on the input edge"
    )


@pytest.mark.integration
def test_add_input_dataset_without_version_leaves_dataset_version_null(test_ml):
    """add_input_dataset(rid) (no version) leaves Dataset_Version NULL — backward compatible."""
    test_ml.add_term(vc.dataset_type, "WritePathTest", description="write-path smoke")

    producer = _make_execution(test_ml)
    ds = producer.create_dataset(
        dataset_types="WritePathTest",
        description="source dataset",
        version=DatasetVersion(1, 0, 0),
    )

    consumer = _make_execution(test_ml)
    consumer.add_input_dataset(ds.dataset_rid)

    rows = _dataset_execution_rows(test_ml, ds.dataset_rid, consumer.execution_rid)
    assert len(rows) == 1, f"expected exactly one input edge, found {rows!r}"
    assert rows[0].get("Dataset_Version") is None, (
        "add_input_dataset() with no version must leave Dataset_Version NULL"
    )
