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
        assert "1.1.0" == dataset.current_version

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
