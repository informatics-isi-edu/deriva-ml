from pprint import pformat

try:
    from icecream import ic
except ImportError:
    ic = lambda *a, **kw: None

from deriva_ml.dataset.aux_classes import DatasetVersion, VersionPart
from deriva_ml.execution.execution import ExecutionConfiguration


class TestDatasetVersion:
    def test_dataset_version_simple(self, test_ml):
        ml_instance = test_ml
        type_rid = ml_instance.add_term("Dataset_Type", "TestSet", description="A test")
        ml_instance.add_term("Workflow_Type", "Manual Workflow", description="A manual workflow")

        # Create a workflow and execution for dataset creation
        workflow = ml_instance.create_workflow(
            name="Test Workflow",
            workflow_type="Manual Workflow",
            description="Workflow for testing",
        )
        execution = ml_instance.create_execution(
            ExecutionConfiguration(description="Test Execution", workflow=workflow)
        )

        dataset = execution.create_dataset(
            dataset_types=type_rid.name,
            description="A New Dataset",
            version=DatasetVersion(1, 0, 0),
        )
        v0 = dataset.current_version
        assert "1.0.0" == str(v0)
        v1 = dataset.increment_dataset_version(component=VersionPart.minor)
        assert "1.1.0" == str(v1)
        assert "1.1.0" == str(dataset.current_version)

    def test_dataset_version_history(self, test_ml):
        ml_instance = test_ml
        type_rid = ml_instance.add_term("Dataset_Type", "TestSet", description="A test")
        ml_instance.add_term("Workflow_Type", "Manual Workflow", description="A manual workflow")

        # Create a workflow and execution for dataset creation
        workflow = ml_instance.create_workflow(
            name="Test Workflow",
            workflow_type="Manual Workflow",
            description="Workflow for testing",
        )
        execution = ml_instance.create_execution(
            ExecutionConfiguration(description="Test Execution", workflow=workflow)
        )

        dataset = execution.create_dataset(
            dataset_types=type_rid.name,
            description="A New Dataset",
            version=DatasetVersion(1, 0, 0),
        )
        assert 1 == len(dataset.dataset_history())
        v1 = dataset.increment_dataset_version(component=VersionPart.minor)
        assert 2 == len(dataset.dataset_history())

    def test_dataset_version(self, dataset_test, tmp_path):
        dataset_description = dataset_test.dataset_description
        ml_instance = dataset_description.dataset._ml_instance
        nested_datasets = [ml_instance.lookup_dataset(ds) for ds in dataset_description.member_rids.get("Dataset", [])]
        datasets = [
            ml_instance.lookup_dataset(dataset)
            for nested_description in dataset_description.members.get("Dataset", [])
            for dataset in nested_description.member_rids.get("Dataset", [])
        ]
        ic(datasets)
        _versions = {
            "d0": dataset_description.dataset.current_version,
            "d1": [ds.current_version for ds in nested_datasets],
            "d2": [ds.current_version for ds in datasets],
        }
        nested_datasets[0].increment_dataset_version(VersionPart.major)
        new_versions = {
            "d0": dataset_description.dataset.current_version,
            "d1": [ds.current_version for ds in nested_datasets],
            "d2": [ds.current_version for ds in datasets],
        }
        ic(_versions)
        ic(new_versions)
        assert new_versions["d0"].major == 2
        assert new_versions["d2"][0].major == 2


class TestMarkDev:
    """Integration tests for ``Dataset.mark_dev`` and the dev-row lifecycle.

    Covers ADR-0003's lazy-mutable-dev-row contract: first call after a
    release creates the dev row at ``.dev1``; subsequent calls advance
    ``.devN`` in place; ``Description`` is replaced (not appended);
    ``current_version`` and ``dataset_history`` reflect the dev state.
    """

    def _setup_dataset(self, ml_instance):
        """Helper: register vocabulary, create workflow+execution, return a fresh dataset."""
        ml_instance.add_term("Dataset_Type", "DevTest", description="A test")
        ml_instance.add_term("Workflow_Type", "Manual Workflow", description="A manual workflow")
        workflow = ml_instance.create_workflow(
            name="MarkDev Workflow",
            workflow_type="Manual Workflow",
            description="Workflow for mark_dev tests",
        )
        execution = ml_instance.create_execution(
            ExecutionConfiguration(description="MarkDev Execution", workflow=workflow)
        )
        dataset = execution.create_dataset(
            dataset_types="DevTest",
            description="Dataset for mark_dev tests",
            version=DatasetVersion(0, 4, 0),
        )
        return dataset, execution

    def test_first_mark_dev_creates_dev1(self, test_ml):
        dataset, _execution = self._setup_dataset(test_ml)
        assert str(dataset.current_version) == "0.4.0"

        dataset.mark_dev("Picked up classifier output")

        assert str(dataset.current_version) == "0.4.0.post1.dev1"
        assert dataset.current_version.is_devrelease

    def test_subsequent_mark_dev_advances_devN(self, test_ml):
        dataset, _execution = self._setup_dataset(test_ml)
        dataset.mark_dev("First drift")
        dataset.mark_dev("Second drift")
        dataset.mark_dev("Third drift")

        assert str(dataset.current_version) == "0.4.0.post1.dev3"

    def test_dev_row_snapshot_is_null(self, test_ml):
        dataset, _execution = self._setup_dataset(test_ml)
        dataset.mark_dev("Some drift")

        history = dataset.dataset_history()
        dev_entries = [h for h in history if h.dataset_version.is_devrelease]
        assert len(dev_entries) == 1
        assert dev_entries[0].snapshot is None

    def test_dev_row_count_stays_one(self, test_ml):
        """Mutable dev row: advancing .devN updates the same row."""
        dataset, _execution = self._setup_dataset(test_ml)
        dataset.mark_dev("First")
        dataset.mark_dev("Second")
        dataset.mark_dev("Third")

        history = dataset.dataset_history()
        dev_entries = [h for h in history if h.dataset_version.is_devrelease]
        assert len(dev_entries) == 1, "expected exactly one dev row across calls"

    def test_description_is_replaced_not_appended(self, test_ml):
        dataset, _execution = self._setup_dataset(test_ml)
        dataset.mark_dev("First description")
        dataset.mark_dev("Second description")

        history = dataset.dataset_history()
        dev = next(h for h in history if h.dataset_version.is_devrelease)
        assert dev.description == "Second description"
        assert "First description" not in (dev.description or "")

    def test_history_includes_dev_row(self, test_ml):
        dataset, _execution = self._setup_dataset(test_ml)
        history_before = dataset.dataset_history()
        assert len(history_before) == 1

        dataset.mark_dev("Some drift")

        history_after = dataset.dataset_history()
        assert len(history_after) == 2
        assert any(h.dataset_version.is_devrelease for h in history_after)

    def test_history_is_sorted_ascending(self, test_ml):
        dataset, _execution = self._setup_dataset(test_ml)
        # Make a few releases plus a dev row.
        dataset.increment_dataset_version(VersionPart.minor, "first bump")
        dataset.increment_dataset_version(VersionPart.minor, "second bump")
        dataset.mark_dev("then dev")

        history = dataset.dataset_history()
        labels = [str(h.dataset_version) for h in history]
        # Released ones first, dev label last (sorts after its anchor release).
        assert labels == sorted(labels, key=lambda s: DatasetVersion.parse(s))
        # And the last entry is the dev label.
        assert history[-1].dataset_version.is_devrelease

    def test_mark_dev_with_execution_attaches_execution(self, test_ml):
        dataset, execution = self._setup_dataset(test_ml)
        dataset.mark_dev("With execution", execution=execution)

        history = dataset.dataset_history()
        dev = next(h for h in history if h.dataset_version.is_devrelease)
        assert dev.execution_rid == execution.execution_rid

    def test_mark_dev_without_execution_leaves_null(self, test_ml):
        dataset, _execution = self._setup_dataset(test_ml)
        dataset.mark_dev("Without execution")

        history = dataset.dataset_history()
        dev = next(h for h in history if h.dataset_version.is_devrelease)
        # DatasetHistory normalises empty/missing execution_rid to None.
        assert dev.execution_rid is None
