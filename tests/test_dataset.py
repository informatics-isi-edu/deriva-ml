from derivaml_test import TestDerivaML
from deriva_ml import (
    DerivaMLException,
    TableDefinition,
    ColumnDefinition,
    BuiltinTypes,
    VersionPart,
    DatasetVersion,
    ExecutionConfiguration,
    MLVocab,
    Workflow,
)


class TestDataset(TestDerivaML):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        type_rid = self.ml_instance.add_term(
            "Dataset_Type", "TestSet", description="A test"
        )
        self.dataset_rid = self.ml_instance.create_dataset(
            type_rid.name, description="A Dataset"
        )
        datasets = list(self.ml_instance.find_datasets())
        self.assertIn(self.dataset_rid, [d["RID"] for d in datasets])
        ds_type = [d["Dataset_Type"] for d in datasets if d["RID"] == self.dataset_rid][
            0
        ]
        self.assertIn("TestSet", ds_type)

        dataset_cnt = len(datasets)
        self.ml_instance.delete_dataset(datasets[0]["RID"])
        self.assertEqual(dataset_cnt - 1, len(self.ml_instance.find_datasets()))
        self.assertEqual(dataset_cnt, len(self.ml_instance.find_datasets(deleted=True)))
        self.assertRaises(
            DerivaMLException, self.ml_instance.list_dataset_members, datasets[0]["RID"]
        )

    def test_dataset_version(self):
        type_rid = self.ml_instance.add_term(
            "Dataset_Type", "TestSet", description="A test"
        )
        dataset_rid = self.ml_instance.create_dataset(
            type_rid.name,
            description="A New Dataset",
            version=DatasetVersion(1, 0, 0),
        )
        v0 = self.ml_instance.dataset_version(dataset_rid)
        self.assertEqual("1.0.0", str(v0))
        v1 = self.ml_instance.increment_dataset_version(
            dataset_rid=dataset_rid, component=VersionPart.minor
        )
        self.assertEqual("1.1.0", str(v1))

    def test_dataset_members(self):
        type_rid = self.ml_instance.add_term(
            "Dataset_Type", "TestSet", description="A test"
        )
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
        table_path = (
            self.ml_instance.catalog.getPathBuilder()
            .schemas[self.domain_schema]
            .tables["TestTableMembers"]
        )
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
        test_rids = self.ml_instance.list_dataset_members(dataset_rid)[
            "TestTableMembers"
        ]
        self.assertEqual(member_cnt - 2, len(test_rids))
        self.assertEqual(len(self.ml_instance.dataset_history(dataset_rid)), 3)
        self.assertEqual(str(self.ml_instance.dataset_version(dataset_rid)), "1.2.0")

    def test_nested_datasets(self):
        self.reset_catalog()
        double_nested_dataset, nested_datasets, datasets = self.create_nested_dataset()

        print(f"double_dataset {double_nested_dataset}")
        print(f"nested_datasets {nested_datasets}")
        print(f"datasets {datasets}")
        import pprint

        print("double_nested_dataset")
        pprint.pprint(
            self.ml_instance.list_dataset_members(dataset_rid=double_nested_dataset)
        )

        print("nested_dataset")
        pprint.pprint(
            [
                self.ml_instance.list_dataset_members(dataset_rid=ds)
                for ds in nested_datasets
            ]
        )

        print("dataset")
        pprint.pprint(
            [self.ml_instance.list_dataset_members(dataset_rid=ds) for ds in datasets]
        )

        print(
            "double nested children",
            self.ml_instance.list_dataset_children(dataset_rid=double_nested_dataset),
        )
        self.assertEqual(2, self.ml_instance._dataset_nesting_depth())
        self.assertEqual(
            set(nested_datasets),
            set(self.ml_instance.list_dataset_children(double_nested_dataset)),
        )

        self.assertEqual(
            set(nested_datasets + datasets),
            set(
                self.ml_instance.list_dataset_children(
                    double_nested_dataset, recurse=True
                )
            ),
        )

        # Check parents and children.
        self.assertEqual(
            2, len(self.ml_instance.list_dataset_children(nested_datasets[0]))
        )

        self.assertEqual(
            double_nested_dataset,
            self.ml_instance.list_dataset_parents(nested_datasets[0])[0],
        )

        self.logger.info("Checking versions.")
        print("Checking versions.")
        versions = {
            "d0": self.ml_instance.dataset_version(double_nested_dataset),
            "d1": [self.ml_instance.dataset_version(v) for v in nested_datasets],
            "d2": [self.ml_instance.dataset_version(v) for v in datasets],
        }
        self.ml_instance.increment_dataset_version(
            nested_datasets[0], VersionPart.major
        )
        new_versions = {
            "d0": self.ml_instance.dataset_version(double_nested_dataset),
            "d1": [self.ml_instance.dataset_version(v) for v in nested_datasets],
            "d2": [self.ml_instance.dataset_version(v) for v in datasets],
        }
        print(versions)
        print(new_versions)

    def test_dataset_execution(self):
        self.ml_instance.model.create_table(
            TableDefinition(
                name="TestTableExecution",
                column_defs=[ColumnDefinition(name="Col1", type=BuiltinTypes.text)],
            )
        )
        self.ml_instance.add_dataset_element_type("TestTableExecution")
        table_path = (
            self.ml_instance.catalog.getPathBuilder()
            .schemas[self.domain_schema]
            .tables["TestTableExecution"]
        )
        table_path.insert([{"Col1": f"Thing{t + 1}"} for t in range(4)])
        test_rids = [i["RID"] for i in table_path.entities().fetch()]

        self.ml_instance.add_term(
            MLVocab.workflow_type,
            "Manual Workflow",
            description="Initial setup of Model File",
        )
        self.ml_instance.add_term("Dataset_Type", "TestSet", description="A test")

        api_workflow = self.ml_instance.add_workflow(
            Workflow(
                name="Manual Workflow",
                url="https://github.com/informatics-isi-edu/deriva-ml/blob/main/tests/test_upload.py",
                workflow_type="Manual Workflow",
                description="A manual operation",
            )
        )
        manual_execution = self.ml_instance.create_execution(
            ExecutionConfiguration(
                description="Sample Execution", workflow=api_workflow
            )
        )

        dataset_rid = manual_execution.create_dataset(
            dataset_types=["TestSet"], description="A dataset"
        )
        manual_execution.add_dataset_members(dataset_rid, test_rids)
        history = self.ml_instance.dataset_history(dataset_rid=dataset_rid)
        self.assertEqual(manual_execution.execution_rid, history[0].execution_rid)
