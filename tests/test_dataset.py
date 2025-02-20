from derivaml_test import TestDerivaML
from deriva_ml import RID, DerivaMLException
from deriva_ml.demo_catalog import (
    reset_demo_catalog,
    create_demo_datasets,
)


class TestDataset(TestDerivaML):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_add_element_type(self):
        reset_demo_catalog(self.ml_instance, self.domain_schema)
        self.ml_instance.add_dataset_element_type("Subject")
        self.assertEqual(len(list(self.ml_instance.list_dataset_element_types())), 2)

    def test_create_dataset(self) -> RID:
        reset_demo_catalog(self.ml_instance, self.domain_schema)
        self.ml_instance.add_dataset_element_type("Subject")
        type_rid = self.ml_instance.add_term(
            "Dataset_Type", "TestSet", description="A test"
        )
        dataset_rid = self.ml_instance.create_dataset(
            type_rid.name, description="A Dataset"
        )
        self.assertEqual(len(list(self.ml_instance.find_datasets())), 1)
        return dataset_rid

    def test_add_dataset_members(self):
        reset_demo_catalog(self.ml_instance, self.domain_schema)
        subject_path = (
            self.ml_instance.catalog.getPathBuilder()
            .schemas[self.domain_schema]
            .tables["Subject"]
        )
        dataset_rid = self.test_create_dataset()
        subject_path.insert([{"Name": f"Thing{t + 1}"} for t in range(5)])
        subject_rids = [i["RID"] for i in subject_path.entities().fetch()]
        self.ml_instance.add_dataset_members(
            dataset_rid=dataset_rid, members=subject_rids
        )
        self.assertEqual(
            len(self.ml_instance.list_dataset_members(dataset_rid)["Subject"]),
            len(subject_rids),
        )
        self.assertEqual(len(self.ml_instance.dataset_history(dataset_rid)), 2)
        self.assertEqual(str(self.ml_instance.dataset_version(dataset_rid)), "0.2.0")

    def test_delete_dataset_members(self):
        reset_demo_catalog(self.ml_instance, self.domain_schema)
        subject_path = (
            self.ml_instance.catalog.getPathBuilder()
            .schemas[self.domain_schema]
            .tables["Subject"]
        )
        dataset_rid = self.test_create_dataset()
        subject_path.insert([{"Name": f"Thing{t + 1}"} for t in range(5)])
        subject_rids = [i["RID"] for i in subject_path.entities().fetch()]
        self.ml_instance.add_dataset_members(
            dataset_rid=dataset_rid, members=subject_rids
        )

        subject_rids = [
            s["RID"]
            for s in self.ml_instance.list_dataset_members(dataset_rid)["Subject"]
        ]
        self.assertEqual(len(subject_rids), 5)
        self.ml_instance.delete_dataset_members(dataset_rid, subject_rids[0:2])
        subject_rids = self.ml_instance.list_dataset_members(dataset_rid)["Subject"]
        self.assertEqual(len(subject_rids), 3)
        self.assertEqual(len(self.ml_instance.dataset_history(dataset_rid)), 3)
        self.assertEqual(str(self.ml_instance.dataset_version(dataset_rid)), "0.3.0")

    def test_delete_dataset(self):
        subject_path = (
            self.ml_instance.catalog.getPathBuilder()
            .schemas[self.domain_schema]
            .tables["Subject"]
        )
        reset_demo_catalog(self.ml_instance, self.domain_schema)
        reset_demo_catalog(self.ml_instance, self.domain_schema)
        self.ml_instance.add_dataset_element_type("Subject")
        type_rid = self.ml_instance.add_term(
            "Dataset_Type", "TestSet", description="A test"
        )
        dataset_rids = [
            self.ml_instance.create_dataset(type_rid.name, description="A Dataset")
            for i in range(5)
        ]
        subject_path.insert([{"Name": f"Thing{t + 1}"} for t in range(5)])
        subject_rids = [i["RID"] for i in subject_path.entities().fetch()]
        self.ml_instance.add_dataset_members(
            dataset_rid=dataset_rids[0], members=subject_rids
        )
        self.assertEqual(5, len(self.ml_instance.find_datasets()))
        self.ml_instance.delete_dataset(dataset_rids[0])
        self.assertEqual(4, len(self.ml_instance.find_datasets()))
        self.assertEqual(5, len(self.ml_instance.find_datasets(deleted=True)))
        self.ml_instance.delete_dataset(dataset_rids[1])
        self.assertRaises(
            DerivaMLException, self.ml_instance.list_dataset_members, dataset_rids[0]
        )

    def test_nested_datasets(self):
        reset_demo_catalog(self.ml_instance, self.domain_schema)
        create_demo_datasets(self.ml_instance)
        datasets = self.ml_instance.find_datasets()

        [nested_dataset, double_nested_dataset] = [
            d["RID"] for d in datasets if "Partitioned" in d["Dataset_Type"]
        ]

        if self.ml_instance._dataset_depth(nested_dataset) == 2:
            nested_dataset, double_nested_dataset = (
                double_nested_dataset,
                nested_dataset,
            )

        self.assertEqual(2, len(nested_dataset.dataset_children()))
        self.assertEqual(double_nested_dataset, nested_dataset.dataset_parents()[0])
        print(double_nested_dataset.dataset_children(recurse=True))

    def test_dataset_version(self):
        # check inrementing datasest version
        # check incrmenting nested version with recurse.
        pass
