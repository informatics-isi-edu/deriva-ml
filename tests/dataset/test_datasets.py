"""
Tests for dataset functionality.
"""

from deriva_ml import DatasetSpec


def test_dataset_find(test_ml_catalog):
    """Test finding datasets."""
    # Find all datasets
    datasets = test_ml_catalog.find_datasets()
    assert len(datasets) > 0

    # Verify dataset types exist
    dataset_types = {ds["Dataset_Type"] for ds in datasets}
    assert "Training" in dataset_types
    assert "Testing" in dataset_types
    assert "Partitioned" in dataset_types


def test_dataset_version(test_ml_catalog):
    """Test dataset versioning."""
    # Get a dataset RID
    datasets = test_ml_catalog.find_datasets()
    dataset_rid = datasets[0]["RID"]

    # Get version
    version = test_ml_catalog.dataset_version(dataset_rid)
    assert version is not None
    assert isinstance(version, str)


def test_dataset_spec():
    """Test DatasetSpec creation and validation."""
    # Create with required fields
    spec = DatasetSpec(rid="1234", version="1.0")
    assert spec.rid == "1234"
    assert spec.version == "1.0"
    assert not spec.materialize  # Default value

    # Create with all fields
    spec = DatasetSpec(rid="1234", version="1.0", materialize=True)
    assert spec.materialize


def test_dataset_creation(test_ml_catalog):
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


def test_dataset_relationships(test_ml_catalog):
    """Test dataset relationship management."""
    # Create two datasets
    parent = test_ml_catalog.create_dataset(
        name="Parent Dataset", description="Parent for testing", dataset_type="Training"
    )
    child = test_ml_catalog.create_dataset(
        name="Child Dataset", description="Child for testing", dataset_type="Testing"
    )

    # Link datasets
    test_ml_catalog.link_datasets(parent_rid=parent["RID"], child_rid=child["RID"], relationship_type="Derived")

    # Verify relationship
    children = test_ml_catalog.get_dataset_children(parent["RID"])
    assert any(c["RID"] == child["RID"] for c in children)

    parents = test_ml_catalog.get_dataset_parents(child["RID"])
    assert any(p["RID"] == parent["RID"] for p in parents)


class TestDataset:
    def test_dataset_elements(self):
        self.ml_instance.model.create_table(
            TableDefinition(
                name="TestTable",
                column_defs=[ColumnDefinition(name="Col1", type=BuiltinTypes.text)],
            )
        )
        self.ml_instance.add_dataset_element_type("TestTable")
        self.assertIn(
            "TestTable",
            [t.name for t in self.ml_instance.list_dataset_element_types()],
        )

    def test_dataset_add_delete(self):
        type_rid = self.ml_instance.add_term("Dataset_Type", "TestSet", description="A test")
        self.dataset_rid = self.ml_instance.create_dataset(type_rid.name, description="A Dataset")
        datasets = list(self.ml_instance.find_datasets())
        self.assertIn(self.dataset_rid, [d["RID"] for d in datasets])
        ds_type = [d["Dataset_Type"] for d in datasets if d["RID"] == self.dataset_rid][0]
        self.assertIn("TestSet", ds_type)

        dataset_cnt = len(datasets)
        self.ml_instance.delete_dataset(datasets[0]["RID"])
        self.assertEqual(dataset_cnt - 1, len(self.ml_instance.find_datasets()))
        self.assertEqual(dataset_cnt, len(self.ml_instance.find_datasets(deleted=True)))
        self.assertRaises(DerivaMLException, self.ml_instance.list_dataset_members, datasets[0]["RID"])

    def test_dataset_version(self):
        type_rid = self.ml_instance.add_term("Dataset_Type", "TestSet", description="A test")
        dataset_rid = self.ml_instance.create_dataset(
            type_rid.name,
            description="A New Dataset",
            version=DatasetVersion(1, 0, 0),
        )
        v0 = self.ml_instance.dataset_version(dataset_rid)
        self.assertEqual("1.0.0", str(v0))
        v1 = self.ml_instance.increment_dataset_version(dataset_rid=dataset_rid, component=VersionPart.minor)
        self.assertEqual("1.1.0", str(v1))

    def test_dataset_members(self):
        type_rid = self.ml_instance.add_term("Dataset_Type", "TestSet", description="A test")
        dataset_rid = self.ml_instance.create_dataset(
            type_rid.name,
            description="A New Dataset",
            version=DatasetVersion(1, 0, 0),
        )

        self.ml_instance.model.create_table(
            TableDefinition(
                name="TestTableMembers",
                column_defs=[ColumnDefinition(name="Col1", type=BuiltinTypes.text)],
            )
        )
        self.ml_instance.add_dataset_element_type("TestTableMembers")
        self.assertIn(
            "TestTableMembers",
            [t.name for t in self.ml_instance.list_dataset_element_types()],
        )
        table_path = self.ml_instance.catalog.getPathBuilder().schemas[self.domain_schema].tables["TestTableMembers"]
        table_path.insert([{"Col1": f"Thing{t + 1}"} for t in range(4)])
        test_rids = [i["RID"] for i in table_path.entities().fetch()]
        member_cnt = len(test_rids)
        self.ml_instance.add_dataset_members(dataset_rid=dataset_rid, members=test_rids)
        self.assertEqual(
            len(self.ml_instance.list_dataset_members(dataset_rid)["TestTableMembers"]),
            len(test_rids),
        )
        self.assertEqual(len(self.ml_instance.dataset_history(dataset_rid)), 2)
        self.assertEqual(str(self.ml_instance.dataset_version(dataset_rid)), "1.1.0")

        self.ml_instance.delete_dataset_members(dataset_rid, test_rids[0:2])
        test_rids = self.ml_instance.list_dataset_members(dataset_rid)["TestTableMembers"]
        self.assertEqual(member_cnt - 2, len(test_rids))
        self.assertEqual(len(self.ml_instance.dataset_history(dataset_rid)), 3)
        self.assertEqual(str(self.ml_instance.dataset_version(dataset_rid)), "1.2.0")

    def test_nested_datasets(self):
        self.reset_catalog()
        double_nested_dataset, nested_datasets, datasets = self.create_nested_dataset()

        self.assertEqual(2, self.ml_instance._dataset_nesting_depth())
        self.assertEqual(
            set(nested_datasets),
            set(self.ml_instance.list_dataset_children(double_nested_dataset)),
        )

        self.assertEqual(
            set(nested_datasets + datasets),
            set(self.ml_instance.list_dataset_children(double_nested_dataset, recurse=True)),
        )

        # Check parents and children.
        self.assertEqual(2, len(self.ml_instance.list_dataset_children(nested_datasets[0])))

        self.assertEqual(
            double_nested_dataset,
            self.ml_instance.list_dataset_parents(nested_datasets[0])[0],
        )

        self.logger.info("Checking versions.")

        versions = {
            "d0": self.ml_instance.dataset_version(double_nested_dataset),
            "d1": [self.ml_instance.dataset_version(v) for v in nested_datasets],
            "d2": [self.ml_instance.dataset_version(v) for v in datasets],
        }
        self.ml_instance.increment_dataset_version(nested_datasets[0], VersionPart.major)
        new_versions = {
            "d0": self.ml_instance.dataset_version(double_nested_dataset),
            "d1": [self.ml_instance.dataset_version(v) for v in nested_datasets],
            "d2": [self.ml_instance.dataset_version(v) for v in datasets],
        }

        self.assertEqual(new_versions["d0"].major, 2)
        self.assertEqual(new_versions["d2"][0].major, 2)

    def test_dataset_execution(self):
        self.ml_instance.model.create_table(
            TableDefinition(
                name="TestTableExecution",
                column_defs=[ColumnDefinition(name="Col1", type=BuiltinTypes.text)],
            )
        )
        self.ml_instance.add_dataset_element_type("TestTableExecution")
        table_path = self.ml_instance.catalog.getPathBuilder().schemas[self.domain_schema].tables["TestTableExecution"]
        table_path.insert([{"Col1": f"Thing{t + 1}"} for t in range(4)])
        test_rids = [i["RID"] for i in table_path.entities().fetch()]

        self.ml_instance.add_term(
            MLVocab.workflow_type,
            "Manual Workflow",
            description="Initial setup of Model File",
        )
        self.ml_instance.add_term("Dataset_Type", "TestSet", description="A test")

        api_workflow = self.ml_instance.create_workflow(
            name="Manual Workflow",
            workflow_type="Manual Workflow",
            description="A manual operation",
        )
        manual_execution = self.ml_instance.create_execution(
            ExecutionConfiguration(description="Sample Execution", workflow=api_workflow)
        )

        dataset_rid = manual_execution.create_dataset(dataset_types=["TestSet"], description="A dataset")
        manual_execution.add_dataset_members(dataset_rid, test_rids)
        history = self.ml_instance.dataset_history(dataset_rid=dataset_rid)
        self.assertEqual(manual_execution.execution_rid, history[0].execution_rid)
