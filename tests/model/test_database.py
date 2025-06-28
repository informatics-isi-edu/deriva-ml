from deriva.core.datapath import Any
from icecream import ic

from deriva_ml import DatasetSpec, DerivaML


class TestDataBaseModel:
    def make_frozen(self, e) -> frozenset:
        e.pop("RMT")
        e.pop("RCT")
        if "Filename" in e:
            e.pop("Filename")
        if "Description" in e and not e["Description"]:
            e["Description"] = ""
        return frozenset(e.items())

    def collect_rids(self, dataset, rid_list=None):
        if not rid_list:
            rid_list = dataset.member_rids
        for ds in dataset.members():
            rid_list = {t: rid_list[t] + rids for t, rids in self.collect_rids(ds, rid_list)}
        return rid_list

    def test_table_as_dict(self, test_ml_catalog_dataset):
        ml_instance = test_ml_catalog_dataset.deriva_ml
        dataset_description = test_ml_catalog_dataset.dataset_description
        reference_datasets = test_ml_catalog_dataset.find_datasets()

        dataset = dataset_description
        dataset_rid = dataset.rid
        dataset_rids = tuple(ml_instance.list_dataset_children(dataset_rid, recurse=True) + [dataset_rid])

        current_version = ml_instance.dataset_version(dataset_description.rid)
        dataset_spec = DatasetSpec(rid=dataset_description.rid, version=current_version)

        snapshot_catalog = DerivaML(ml_instance.host_name, ml_instance._version_snapshot(dataset_spec))
        bag = ml_instance.download_dataset_bag(dataset_spec)

        # Check to make sure that all of the datasets are present.
        assert {r for r in bag.model.bag_rids.keys()} == {r for r in reference_datasets}


        for dataset_rid in [dataset.rid]:
            dataset_bag = bag.model.get_dataset(dataset_rid)

            pb = snapshot_catalog.pathBuilder
            ds = pb.schemas[snapshot_catalog.ml_schema].tables["Dataset"]
            subject = pb.schemas[snapshot_catalog.domain_schema].tables["Subject"]
            image = pb.schemas[snapshot_catalog.domain_schema].tables["Image"]
            ds_ds = pb.schemas[snapshot_catalog.ml_schema].tables["Dataset_Dataset"]
            ds_subject = pb.schemas[snapshot_catalog.domain_schema].tables["Dataset_Subject"]
            ds_image = pb.schemas[snapshot_catalog.domain_schema].tables["Dataset_Image"]
            version = pb.schemas[snapshot_catalog.ml_schema].tables["Dataset_Version"]

            ds_versions = version.path.filter(version.Dataset == Any(*dataset_rids))
            ds_path = ds.path.link(ds_ds).filter(ds_ds.Dataset == Any(*dataset_rids)).link(ds)
            subject_path = ds.path.link(ds_subject).filter(ds_subject.Dataset == Any(*dataset_rids)).link(subject)
            image_path = ds.path.link(ds_image).filter(ds_image.Dataset == Any(*dataset_rids)).link(image)
            subject_path_1 = ds.path.link(ds_image).filter(ds_image.Dataset == Any(*dataset_rids)).link(image).link(subject)
            image_path_1 = (
                ds.path.link(ds_subject).filter(ds_subject.Dataset == Any(*dataset_rids)).link(subject).link(image)
            )

            datasets = list(ds_path.entities().fetch())
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
