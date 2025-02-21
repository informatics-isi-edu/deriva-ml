from derivaml_test import TestDerivaML
from deriva_ml import (
    RID,
    DerivaMLException,
    TableDefinition,
    ColumnDefinition,
    BuiltinTypes,
    VersionPart,
    DatasetVersion,
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

    def test_dataset_creation(self):
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
        table_path = (
            self.ml_instance.catalog.getPathBuilder()
            .schemas[self.domain_schema]
            .tables["TestTable"]
        )
        table_path.insert([{"Col1": f"Thing{t + 1}"} for t in range(4)])
        test_rids = [i["RID"] for i in table_path.entities().fetch()]
        member_cnt = len(test_rids)
        self.ml_instance.add_dataset_members(dataset_rid=dataset_rid, members=test_rids)
        self.assertEqual(
            len(self.ml_instance.list_dataset_members(dataset_rid)["TestTable"]),
            len(test_rids),
        )
        self.assertEqual(len(self.ml_instance.dataset_history(dataset_rid)), 2)
        self.assertEqual(str(self.ml_instance.dataset_version(dataset_rid)), "1.1.0")

        self.ml_instance.delete_dataset_members(dataset_rid, test_rids[0:2])
        test_rids = self.ml_instance.list_dataset_members(dataset_rid)["TestTable"]
        self.assertEqual(member_cnt - 2, len(test_rids))
        self.assertEqual(len(self.ml_instance.dataset_history(dataset_rid)), 3)
        self.assertEqual(str(self.ml_instance.dataset_version(dataset_rid)), "1.2.0")

    def test_dataset_delete(self):
        dataset_rids = [d["RID"] for d in self.ml_instance.find_datasets()]
        dataset_cnt = len(dataset_rids)
        self.ml_instance.delete_dataset(dataset_rids[0])
        self.assertEqual(dataset_cnt - 1, len(self.ml_instance.find_datasets()))
        self.assertEqual(dataset_cnt, len(self.ml_instance.find_datasets(deleted=True)))
        self.assertRaises(
            DerivaMLException, self.ml_instance.list_dataset_members, dataset_rids[0]
        )

    def test_nested_datasets(self):
        double_nested_dataset, nested_datasets, datasets = self.create_nested_dataset()

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

        # check inrementing datasest version
        # check incrmenting nested version with recurse.
