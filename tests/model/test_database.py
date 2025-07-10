from conftest import DatasetDescription, MLDatasetTest
from deriva.core.datapath import Any

from deriva_ml import DatasetSpec


class TestDataBaseModel:
    def make_frozen(self, e) -> frozenset:
        e.pop("RMT")
        e.pop("RCT")
        if "Filename" in e:
            e.pop("Filename")
        if "Description" in e and not e["Description"]:
            e["Description"] = ""
        return frozenset(e.items())

    def list_datasets(self, dataset_description: DatasetDescription) -> set[str]:
        nested_datasets = {
            ds
            for dset_member in dataset_description.members.get("Dataset", [])
            for ds in self.list_datasets(dset_member)
        }
        return {dataset_description.rid} | nested_datasets

    def compare_catalogs(self, dataset: MLDatasetTest, dataset_spec: DatasetSpec):
        ml_instance = dataset.ml_instance
        reference_datasets = self.list_datasets(dataset.dataset_description)

        snapshot_catalog = dataset.snapshot_catalog(dataset_spec)
        bag = ml_instance.download_dataset_bag(dataset_spec)

        pb = snapshot_catalog.pathBuilder
        ds = pb.schemas[snapshot_catalog.ml_schema].tables["Dataset"]
        subject = pb.schemas[snapshot_catalog.domain_schema].tables["Subject"]
        image = pb.schemas[snapshot_catalog.domain_schema].tables["Image"]
        ds_ds = pb.schemas[snapshot_catalog.ml_schema].tables["Dataset_Dataset"]
        ds_subject = pb.schemas[snapshot_catalog.domain_schema].tables["Dataset_Subject"]
        ds_image = pb.schemas[snapshot_catalog.domain_schema].tables["Dataset_Image"]

        # Check to make sure that all of the datasets are present.
        assert {r for r in bag.model.bag_rids.keys()} == {r for r in reference_datasets}
        for dataset_rid in reference_datasets:
            dataset_bag = bag.model.get_dataset(dataset_rid)
            dataset_rids = tuple(ml_instance.list_dataset_children(dataset_rid, recurse=True) + [dataset_rid])
            bag_rids = [b.dataset_rid for b in dataset_bag.list_dataset_children(recurse=True)] + [
                dataset_bag.dataset_rid
            ]
            assert list(dataset_rids).sort() == bag_rids.sort()

            ds_path = ds.path.link(ds_ds).filter(ds_ds.Dataset == Any(*dataset_rids)).link(ds)
            subject_path = ds.path.link(ds_subject).filter(ds_subject.Dataset == Any(*dataset_rids)).link(subject)
            image_path = ds.path.link(ds_image).filter(ds_image.Dataset == Any(*dataset_rids)).link(image)
            subject_path_1 = (
                ds.path.link(ds_image).filter(ds_image.Dataset == Any(*dataset_rids)).link(image).link(subject)
            )
            image_path_1 = (
                ds.path.link(ds_subject).filter(ds_subject.Dataset == Any(*dataset_rids)).link(subject).link(image)
            )
            dataset_path_1 = ds.path.filter(ds.RID == Any(*dataset_rids))
            datasets = list(ds_path.entities().fetch()) + list(dataset_path_1.entities().fetch())
            subjects = list(subject_path.entities().fetch()) + list(subject_path_1.entities().fetch())
            images = list(image_path.entities().fetch()) + list(image_path_1.entities().fetch())

            catalog_dataset = set([self.make_frozen(d) for d in datasets])
            catalog_subject = set([self.make_frozen(s) for s in subjects])
            catalog_image = set([self.make_frozen(i) for i in images])

            dataset_table = set([self.make_frozen(d) for d in dataset_bag.get_table_as_dict("Dataset")])
            subject_table = set([self.make_frozen(s) for s in dataset_bag.get_table_as_dict("Subject")])
            image_table = set([self.make_frozen(i) for i in dataset_bag.get_table_as_dict("Image")])

            assert len(catalog_dataset) == len(dataset_table)
            assert len(catalog_subject) == len(subject_table)
            assert len(catalog_image) == len(image_table)

            assert catalog_dataset == dataset_table
            assert catalog_subject == subject_table
            assert catalog_image == image_table

    def test_database_methods(self, test_ml_catalog_dataset):
        ml_instance = test_ml_catalog_dataset.ml_instance
        dataset_description = test_ml_catalog_dataset.dataset_description
        current_version = ml_instance.dataset_version(dataset_description.rid)
        current_spec = DatasetSpec(rid=dataset_description.rid, version=current_version)
        current_bag = ml_instance.download_dataset_bag(current_spec)
        tables = current_bag.model.list_tables()
        schemas = ml_instance.model.schemas
        catalog_tables = len(schemas[ml_instance.domain_schema].tables) + len(schemas[ml_instance.ml_schema].tables)
        assert catalog_tables == len(tables)

    def test_table_as_dict(self, test_ml_catalog_dataset):
        ml_instance = test_ml_catalog_dataset.ml_instance
        dataset_description = test_ml_catalog_dataset.dataset_description

        current_version = ml_instance.dataset_version(dataset_description.rid)
        dataset_spec = DatasetSpec(rid=dataset_description.rid, version=current_version)
        self.compare_catalogs(test_ml_catalog_dataset, dataset_spec)

    def test_table_versions(self, test_ml_catalog_dataset):
        ml_instance = test_ml_catalog_dataset.ml_instance
        dataset_description = test_ml_catalog_dataset.dataset_description

        current_version = ml_instance.dataset_version(dataset_description.rid)
        current_spec = DatasetSpec(rid=dataset_description.rid, version=current_version)
        self.compare_catalogs(
            test_ml_catalog_dataset, DatasetSpec(rid=dataset_description.rid, version=current_version)
        )

        pb = ml_instance.pathBuilder
        subjects = [s["RID"] for s in pb.schemas[ml_instance.domain_schema].tables["Subject"].path.entities().fetch()]
        ml_instance.add_dataset_members(dataset_description.rid, subjects[-2:])
        new_version = ml_instance.dataset_version(dataset_description.rid)
        new_spec = DatasetSpec(rid=dataset_description.rid, version=new_version)
        current_bag = ml_instance.download_dataset_bag(current_spec)
        new_bag = ml_instance.download_dataset_bag(new_spec)
        subjects_current = list(current_bag.get_table_as_dict("Subject"))
        subjects_new = list(new_bag.get_table_as_dict("Subject"))

        # Make sure that there is a difference between to old and new catalogs.
        print(subjects_current)
        print(subjects_new)
        assert len(subjects_new) == len(subjects_current) + 2
        print("compare")
        self.compare_catalogs(test_ml_catalog_dataset, current_spec)
        self.compare_catalogs(test_ml_catalog_dataset, new_spec)
