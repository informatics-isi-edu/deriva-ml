"""
Tests for dataset functionality.
"""

import pytest

from deriva_ml import (
    BuiltinTypes,
    ColumnDefinition,
    DatasetSpec,
    DatasetVersion,
    DerivaMLException,
    ExecutionConfiguration,
    MLVocab,
    TableDefinition,
    VersionPart,
)


class TestDatasetManagement:
    def test_dataset_find(self, test_ml_catalog_populated):
        """Test finding datasets."""
        # Find all datasets
        ml_instance = test_ml_catalog_populated
        datasets = ml_instance.find_datasets()
        assert len(datasets) > 0

        # Verify dataset types exist
        dataset_types = {ds["Dataset_Type"] for ds in datasets}
        assert "Training" in dataset_types
        assert "Testing" in dataset_types
        assert "Partitioned" in dataset_types

    def test_dataset_version(self, test_ml_catalog):
        """Test dataset versioning."""
        # Get a dataset RID
        datasets = test_ml_catalog.find_datasets()
        dataset_rid = datasets[0]["RID"]

        # Get version
        version = test_ml_catalog.dataset_version(dataset_rid)
        assert version is not None
        assert isinstance(version, str)

    def test_dataset_spec(self, test_ml_catalog):
        """Test DatasetSpec creation and validation."""
        # Create with required fields
        spec = DatasetSpec(rid="1234", version="1.0")
        assert spec.rid == "1234"
        assert spec.version == "1.0"
        assert not spec.materialize  # Default value

        # Create with all fields
        spec = DatasetSpec(rid="1234", version="1.0.0", materialize=True)
        assert spec.materialize

    def test_dataset_creation(self, test_ml_catalog):
        """Test dataset creation and modification."""
        # Find existing datasets for reference
        existing = test_ml_catalog.find_datasets()
        initial_count = len(existing)

        # Create a new dataset
        dataset = test_ml_catalog.create_dataset(
            name="Test Dataset", description="Dataset for testing", dataset_type="Testing"
        )
        assert dataset is not None

        # Verify dataset was created
        updated = test_ml_catalog.find_datasets()
        assert len(updated) == initial_count + 1

        # Find the new dataset
        new_dataset = next(ds for ds in updated if ds["Name"] == "Test Dataset")
        assert new_dataset["Description"] == "Dataset for testing"
        assert new_dataset["Dataset_Type"] == "Testing"


class TestDataset:
    def test_dataset_elements(self, test_ml_catalog):
        ml_instance = test_ml_catalog
        ml_instance.model.create_table(
            TableDefinition(
                name="TestTable",
                column_defs=[ColumnDefinition(name="Col1", type=BuiltinTypes.text)],
            )
        )
        ml_instance.add_dataset_element_type("TestTable")
        assert "TestTable" in [t.name for t in ml_instance.list_dataset_element_types()]

    def test_dataset_add_delete(self, test_ml_catalog):
        ml_instance = test_ml_catalog
        type_rid = ml_instance.add_term("Dataset_Type", "TestSet", description="A test")
        self.dataset_rid = ml_instance.create_dataset(type_rid.name, description="A Dataset")
        datasets = list(ml_instance.find_datasets())
        assert self.dataset_rid in [d["RID"] for d in datasets]
        ds_type = [d["Dataset_Type"] for d in datasets if d["RID"] == self.dataset_rid][0]
        assert "TestSet" in ds_type

        dataset_cnt = len(datasets)
        ml_instance.delete_dataset(datasets[0]["RID"])
        assert dataset_cnt - 1 == len(ml_instance.find_datasets())
        assert dataset_cnt == len(ml_instance.find_datasets(deleted=True))
        with pytest.raises(DerivaMLException):
            ml_instance.list_dataset_members, datasets[0]["RID"]

    def test_dataset_version(self, test_ml_catalog):
        ml_instance = test_ml_catalog
        type_rid = ml_instance.add_term("Dataset_Type", "TestSet", description="A test")
        dataset_rid = ml_instance.create_dataset(
            type_rid.name,
            description="A New Dataset",
            version=DatasetVersion(1, 0, 0),
        )
        v0 = ml_instance.dataset_version(dataset_rid)
        assert "1.0.0" == str(v0)
        v1 = ml_instance.increment_dataset_version(dataset_rid=dataset_rid, component=VersionPart.minor)
        assert "1.1.0" == str(v1)

    def test_dataset_members(self, test_ml_catalog):
        ml_instance = test_ml_catalog
        type_rid = ml_instance.add_term("Dataset_Type", "TestSet", description="A test")
        dataset_rid = ml_instance.create_dataset(
            type_rid.name,
            description="A New Dataset",
            version=DatasetVersion(1, 0, 0),
        )

        ml_instance.model.create_table(
            TableDefinition(
                name="TestTableMembers",
                column_defs=[ColumnDefinition(name="Col1", type=BuiltinTypes.text)],
            )
        )
        ml_instance.add_dataset_element_type("TestTableMembers")
        assert "TestTableMembers" in [t.name for t in ml_instance.list_dataset_element_types()]

        table_path = ml_instance.catalog.getPathBuilder().schemas[self.domain_schema].tables["TestTableMembers"]
        table_path.insert([{"Col1": f"Thing{t + 1}"} for t in range(4)])
        test_rids = [i["RID"] for i in table_path.entities().fetch()]
        member_cnt = len(test_rids)
        ml_instance.add_dataset_members(dataset_rid=dataset_rid, members=test_rids)
        assert len(ml_instance.list_dataset_members(dataset_rid)["TestTableMembers"]) == len(test_rids)

        assert len(ml_instance.dataset_history(dataset_rid)) == 2
        assert str(ml_instance.dataset_version(dataset_rid)) == "1.1.0"

        ml_instance.delete_dataset_members(dataset_rid, test_rids[0:2])
        test_rids = ml_instance.list_dataset_members(dataset_rid)["TestTableMembers"]
        assert member_cnt - 2 == len(test_rids)
        assert len(ml_instance.dataset_history(dataset_rid)) == 3
        assert str(ml_instance.dataset_version(dataset_rid)) == "1.2.0"

    def test_nested_datasets(self, test_ml_catalog):
        ml_instance = test_ml_catalog
        double_nested_dataset, nested_datasets, datasets = self.create_nested_dataset()

        assert 2 == ml_instance._dataset_nesting_depth()
        assert set(nested_datasets) == set(ml_instance.list_dataset_children(double_nested_dataset))
        assert set(nested_datasets + datasets) == set(
            ml_instance.list_dataset_children(double_nested_dataset, recurse=True)
        )

        # Check parents and children.
        assert 2 == len(ml_instance.list_dataset_children(nested_datasets[0]))
        assert double_nested_dataset == ml_instance.list_dataset_parents(nested_datasets[0])[0]

        self.logger.info("Checking versions.")

        versions = {
            "d0": ml_instance.dataset_version(double_nested_dataset),
            "d1": [ml_instance.dataset_version(v) for v in nested_datasets],
            "d2": [ml_instance.dataset_version(v) for v in datasets],
        }
        ml_instance.increment_dataset_version(nested_datasets[0], VersionPart.major)
        new_versions = {
            "d0": ml_instance.dataset_version(double_nested_dataset),
            "d1": [ml_instance.dataset_version(v) for v in nested_datasets],
            "d2": [ml_instance.dataset_version(v) for v in datasets],
        }

        assert new_versions["d0"].major == 2
        assert new_versions["d2"][0].major == 2

    def test_dataset_execution(self, test_ml_catalog):
        ml_instance = test_ml_catalog
        ml_instance.model.create_table(
            TableDefinition(
                name="TestTableExecution",
                column_defs=[ColumnDefinition(name="Col1", type=BuiltinTypes.text)],
            )
        )
        ml_instance.add_dataset_element_type("TestTableExecution")
        table_path = ml_instance.catalog.getPathBuilder().schemas[self.domain_schema].tables["TestTableExecution"]
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

        dataset_rid = manual_execution.create_dataset(dataset_types=["TestSet"], description="A dataset")
        manual_execution.add_dataset_members(dataset_rid, test_rids)
        history = ml_instance.dataset_history(dataset_rid=dataset_rid)
        assert manual_execution.execution_rid == history[0].execution_rid
