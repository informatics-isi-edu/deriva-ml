"""
Tests for dataset functionality.
"""

from pprint import pformat

from icecream import ic

from deriva_ml import (
    BuiltinTypes,
    ColumnDefinition,
    DerivaML,
    MLVocab,
    TableDefinition,
)
from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.dataset.catalog_graph import CatalogGraph
from deriva_ml.demo_catalog import DatasetDescription
from deriva_ml.execution.execution import ExecutionConfiguration

ic.configureOutput(
    argToStringFunction=lambda x: pformat(x.model_dump() if hasattr(x, "model_dump") else x, width=80, depth=10)
)


class TestDataset:
    def test_dataset_elements(self, deriva_catalog, tmp_path):
        ml_instance = DerivaML(
            deriva_catalog.hostname, deriva_catalog.catalog_id, working_dir=tmp_path, use_minid=False
        )
        _test_table = ml_instance.model.create_table(
            TableDefinition(
                name="TestTable",
                column_defs=[ColumnDefinition(name="Col1", type=BuiltinTypes.text)],
            )
        )
        ml_instance.add_dataset_element_type("TestTable")
        assert "TestTable" in [t.name for t in ml_instance.list_dataset_element_types()]
        # Check for repeat addition.
        ml_instance.add_dataset_element_type("TestTable")

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

    def test_nested_datasets(self, dataset_test, tmp_path):
        """Test finding datasets."""
        # Find all datasets
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)
        dset_description = dataset_test.dataset_description
        reference_datasets = {ds.dataset.dataset_rid for ds in dataset_test.list_datasets(dset_description)}
        # Now check top level nesting
        child_rids = set(ds.dataset_rid for ds in dset_description.dataset.list_dataset_children())
        assert set(dset_description.member_rids["Dataset"]) == child_rids
        # Now look two levels down

        for member_ds in dset_description.members["Dataset"]:
            child_rids = set(ds.dataset_rid for ds in member_ds.dataset.list_dataset_children())
            assert set(member_ds.member_rids["Dataset"]) == child_rids

        # Now check recursion
        nested_datasets = reference_datasets - {dset_description.dataset.dataset_rid}
        assert nested_datasets == set(
            ds.dataset_rid for ds in dset_description.dataset.list_dataset_children(recurse=True)
        )

        def check_relationships(description: DatasetDescription):
            """Check relationships between datasets."""
            dataset_children = description.dataset.list_dataset_children()
            assert set(description.member_rids.get("Dataset", [])) == set(ds.dataset_rid for ds in dataset_children)
            for child in dataset_children:
                assert child.list_dataset_parents()[0].dataset_rid == description.dataset.dataset_rid
            for nested_dataset in description.members.get("Dataset", []):
                check_relationships(nested_dataset)

        check_relationships(dset_description)

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

    def test_dataset_members(self, dataset_test, tmp_path):
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)
        dataset_description = dataset_test.dataset_description
        catalog_datasets = ml_instance.find_datasets()
        reference_datasets = dataset_test.list_datasets(dataset_description)
        assert len(list(catalog_datasets)) == len(reference_datasets)

        assert CatalogGraph(ml_instance=ml_instance, s3_bucket=ml_instance.s3_bucket)._dataset_nesting_depth() == 2

        for dataset in reference_datasets:
            # See if the list of RIDs in the dataset matches up with what is expected.
            for member_type, dataset_members in dataset.dataset.list_dataset_members().items():
                if member_type == "File":
                    continue
                member_rids = {e["RID"] for e in dataset_members}
                assert set(dataset.member_rids.get(member_type, set())) == set(member_rids)

        for dataset in reference_datasets:
            reference_members = dataset_test.collect_rids(dataset)
            member_rids = {dataset.dataset.dataset_rid}
            for member_type, dataset_members in dataset.dataset.list_dataset_members(recurse=True).items():
                if member_type == "File":
                    continue
                member_rids |= {e["RID"] for e in dataset_members}
            assert reference_members == member_rids

    def test_dataset_execution(self, test_ml):
        ml_instance = test_ml
        ml_instance.model.create_table(
            TableDefinition(
                name="TestTableExecution",
                column_defs=[ColumnDefinition(name="Col1", type=BuiltinTypes.text)],
            )
        )
        ml_instance.add_dataset_element_type("TestTableExecution")
        table_path = (
            ml_instance.catalog.getPathBuilder().schemas[ml_instance.domain_schema].tables["TestTableExecution"]
        )
        table_path.insert([{"Col1": f"Thing{t + 1}"} for t in range(4)])
        test_rids = [i["RID"] for i in table_path.entities().fetch()]

        ml_instance.add_term(
            MLVocab.workflow_type,
            "Manual Workflow",
            description="Initial setup of Model File",
        )
        ml_instance.add_term("Dataset_Type", "TestSet", description="A test")

        api_workflow = ml_instance.create_workflow(
            name="Manual Workflow",
            workflow_type="Manual Workflow",
            description="A manual operation",
        )
        manual_execution = ml_instance.create_execution(
            ExecutionConfiguration(description="Sample Execution", workflow=api_workflow)
        )

        dataset = manual_execution.create_dataset(dataset_types=["TestSet"], description="A dataset")
        dataset.add_dataset_members(test_rids)
        history = dataset.dataset_history()
        assert manual_execution.execution_rid == history[0].execution_rid
