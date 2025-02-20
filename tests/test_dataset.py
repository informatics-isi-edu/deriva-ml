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
from deriva_ml.demo_catalog import (
    reset_demo_catalog,
    create_demo_datasets,
)


class TestDataset(TestDerivaML):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_dataset_1(self):
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

    def test_dataset_2(self):
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

    def test_dataset_3(self):
        type_rid = self.ml_instance.lookup_term("Dataset_Type", "TestSet")
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

    def test_dataset_4(self):
        type_rid = self.ml_instance.lookup_term("Dataset_Type", "TestSet")
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

    def test_dataset_5(self):
        dataset_rids = [d["RID"] for d in self.ml_instance.find_datasets()]
        dataset_cnt = len(dataset_rids)
        self.ml_instance.delete_dataset(dataset_rids[0])
        self.assertEqual(dataset_cnt - 1, len(self.ml_instance.find_datasets()))
        self.assertEqual(dataset_cnt, len(self.ml_instance.find_datasets(deleted=True)))
        self.assertRaises(
            DerivaMLException, self.ml_instance.list_dataset_members, dataset_rids[0]
        )

    def test_nested_datasets(self):
        type_rid = self.ml_instance.lookup_term("Dataset_Type", "TestSet")
        table_path = (
            self.ml_instance.catalog.getPathBuilder()
            .schemas[self.domain_schema]
            .tables["TestTable"]
        )
        test_rids = [i["RID"] for i in table_path.entities().fetch()]
        dataset_rids = []
        for r in test_rids[0:4]:
            d = self.ml_instance.create_dataset(
                type_rid.name,
                description=f"Dataset {r}",
                version=DatasetVersion(1, 0, 0),
            )
            self.ml_instance.add_dataset_members(d, r)
            dataset_rids.append(d)
        nested_datasets = []
        for i in range(0, 4, 2):
            nested_dataset = self.ml_instance.create_dataset(
                type_rid.name,
                description=f"Nested Dataset {i}",
                version=DatasetVersion(1, 0, 0),
            )
            self.ml_instance.add_dataset_members(
                nested_dataset, dataset_rids[i : i + 1]
            )
            nested_datasets.append(nested_dataset)
        double_nested_dataset = self.ml_instance.create_dataset()
        double_nested_dataset.add_dataset_members(nested_datasets)

        self.assertEqual(2, double_nested_dataset._dataset_nesting_depth())
        self.assertEqual(2, len(nested_datasets[0].dataset_children()))
        self.assertEqual(double_nested_dataset, nested_datasets[0].dataset_parents()[0])
        print(double_nested_dataset.dataset_children(recurse=True))

    def test_dataset_version(self):
        # check inrementing datasest version
        # check incrmenting nested version with recurse.
        pass
