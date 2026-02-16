""" Test catalog level functions that support datasets."""
from pprint import pformat

try:
    from icecream import ic
except ImportError:
    ic = lambda *a, **kw: None

from deriva_ml import (
    DerivaML,
    MLVocab,
)
from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.execution.execution import ExecutionConfiguration

class TestCatalogDatasetFunctions:
    def test_dataset_creation(self, deriva_catalog, tmp_path):
        """Test dataset creation and modification."""

        ml_instance = DerivaML(
            deriva_catalog.hostname, deriva_catalog.catalog_id, working_dir=tmp_path, use_minid=False
        )

        # Find existing datasets for reference
        existing = list(ml_instance.find_datasets())
        initial_count = len(existing)
        ml_instance.add_term(MLVocab.dataset_type, "Testing", description="A test dataset")
        ml_instance.add_term(MLVocab.workflow_type, "Manual Workflow", description="A manual workflow")

        # Create a workflow and execution for dataset creation
        workflow = ml_instance.create_workflow(
            name="Test Workflow",
            workflow_type="Manual Workflow",
            description="Workflow for testing dataset creation",
        )
        execution = ml_instance.create_execution(
            ExecutionConfiguration(description="Test Execution", workflow=workflow)
        )

        # Create a new dataset via execution
        dataset = execution.create_dataset(description="Dataset for testing", dataset_types=["Testing"])
        assert dataset is not None

        # Verify dataset was created
        updated = list(ml_instance.find_datasets())
        assert len(updated) == initial_count + 1

        # Find the new dataset
        new_dataset = next(ds for ds in updated if ds.dataset_rid == dataset.dataset_rid)
        assert new_dataset.description == "Dataset for testing"
        assert new_dataset.dataset_types == ["Testing"]

    def test_dataset_find(self, dataset_test, tmp_path):
        """Test finding datasets."""
        # Find all datasets
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)
        dset_description = dataset_test.dataset_description
        reference_datasets = {ds.dataset.dataset_rid for ds in dataset_test.list_datasets(dset_description)}
        # Check all of the dataset.
        assert reference_datasets == {ds.dataset_rid for ds in ml_instance.find_datasets()}

        for ds in ml_instance.find_datasets():
            dataset_types = ds.dataset_types
            for t in dataset_types:
                assert ml_instance.lookup_term(MLVocab.dataset_type, t) is not None

    def test_dataset_add_delete(self, test_ml):
        ml_instance = test_ml
        type_rid = ml_instance.add_term("Dataset_Type", "TestSet", description="A test")
        ml_instance.add_term(MLVocab.workflow_type, "Manual Workflow", description="A manual workflow")

        # Create a workflow and execution for dataset creation
        workflow = ml_instance.create_workflow(
            name="Test Workflow",
            workflow_type="Manual Workflow",
            description="Workflow for testing",
        )
        execution = ml_instance.create_execution(
            ExecutionConfiguration(description="Test Execution", workflow=workflow)
        )

        dataset = execution.create_dataset(dataset_types=type_rid.name, description="A Dataset")
        datasets = list(ml_instance.find_datasets())
        assert dataset.dataset_rid in [d.dataset_rid for d in datasets]

        ml_instance.delete_dataset(dataset)
        assert len(ml_instance.find_datasets()) == 0
        assert len(ml_instance.find_datasets(deleted=True)) == 1

    def test_dataset_spec(self):
        """Test DatasetSpec creation and validation."""
        # Create with required fields
        spec = DatasetSpec(rid="1234", version="1.0.0")
        assert spec.rid == "1234"
        assert spec.version == "1.0.0"
        assert spec.materialize  # Default value

        # Create with all fields
        spec = DatasetSpec(rid="1234", version="1.0.0", materialize=True)
        assert spec.materialize

    def test_dataset_type_manipulation(self, test_ml):
        """Test adding and removing dataset types."""
        ml_instance = test_ml
        ml_instance.add_term("Dataset_Type", "TypeA", description="Type A")
        ml_instance.add_term("Dataset_Type", "TypeB", description="Type B")
        ml_instance.add_term("Dataset_Type", "TypeC", description="Type C")
        ml_instance.add_term(MLVocab.workflow_type, "Manual Workflow", description="A manual workflow")

        # Create a workflow and execution for dataset creation
        workflow = ml_instance.create_workflow(
            name="Test Workflow",
            workflow_type="Manual Workflow",
            description="Workflow for testing",
        )
        execution = ml_instance.create_execution(
            ExecutionConfiguration(description="Test Execution", workflow=workflow)
        )

        # Create dataset with one type
        dataset = execution.create_dataset(dataset_types=["TypeA"], description="Test dataset")
        assert dataset.dataset_types == ["TypeA"]

        # Add another type
        dataset.add_dataset_type("TypeB")
        assert set(dataset.dataset_types) == {"TypeA", "TypeB"}

        # Adding same type again should be a no-op
        dataset.add_dataset_type("TypeA")
        assert set(dataset.dataset_types) == {"TypeA", "TypeB"}

        # Add multiple types at once (one new, one existing)
        dataset.add_dataset_types(["TypeB", "TypeC"])
        assert set(dataset.dataset_types) == {"TypeA", "TypeB", "TypeC"}

        # Remove a type
        dataset.remove_dataset_type("TypeB")
        assert set(dataset.dataset_types) == {"TypeA", "TypeC"}

        # Remove same type again should be a no-op
        dataset.remove_dataset_type("TypeB")
        assert set(dataset.dataset_types) == {"TypeA", "TypeC"}

        # Verify changes persist when looking up the dataset again
        dataset2 = ml_instance.lookup_dataset(dataset.dataset_rid)
        assert set(dataset2.dataset_types) == {"TypeA", "TypeC"}
