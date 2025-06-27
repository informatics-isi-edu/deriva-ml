from deriva.core.datapath import Any

from deriva_ml import DatasetSpec


class TestDataBaseModel:
    def test_table_as_dict(self, test_ml_catalog_dataset):
        ml_instance = test_ml_catalog_dataset.deriva_ml
        dataset_description = test_ml_catalog_dataset.dataset_description
        reference_datasets = test_ml_catalog_dataset.find_datasets()

        current_version = ml_instance.dataset_version(dataset_description.rid)
        bag = ml_instance.download_dataset_bag(DatasetSpec(rid=dataset_description.rid, version=current_version))

        # Check to make sure that all of the datasets are present.
        assert {r for r in bag.model.bag_rids.keys()} == {r for r in reference_datasets}

        def collect_rids(dataset, rid_list=None):
            if not rid_list:
                rid_list = dataset.member_rids
            for ds in dataset.members():
                rid_list = {t: rid_list[t] + rids for t, rids in collect_rids(ds, rid_list)}
            return rid_list

        dataset = dataset_description
        dataset_bag = bag.model.get_dataset(dataset.rid)
        dataset_rid = dataset.rid
        dataset_rids = tuple(ml_instance.list_dataset_children(dataset_rid, recurse=True) + [dataset_rid])
        print(dataset_rid)
        print(dataset_rids)
        print(dataset_bag.model.snaptime)

        pb = ml_instance.pathBuilder
        ds = pb.schemas[ml_instance.ml_schema].tables["Dataset"]
        subject = pb.schemas[ml_instance.domain_schema].tables["Subject"]
        image = pb.schemas[ml_instance.domain_schema].tables["Image"]
        ds_ds = pb.schemas[ml_instance.ml_schema].tables["Dataset_Dataset"]
        ds_subject = pb.schemas[ml_instance.domain_schema].tables["Dataset_Subject"]
        ds_image = pb.schemas[ml_instance.domain_schema].tables["Dataset_Image"]

        ds_path = ds.path.link(ds_ds).filter(ds_ds.Dataset == Any(dataset_rid)).link(ds)
        subject_path = ds.path.link(ds_subject).filter(ds_subject.Dataset == Any(*dataset_rids)).link(subject)
        image_path = ds.path.link(ds_image).filter(ds_image.Dataset == Any(dataset_rid)).link(image)
        subject_path_1 = image_path.link(subject)
        image_path_1 = subject_path.link(image)

        datasets = list(ds_path.entities().fetch())
        subjects = list(subject_path.entities().fetch()) + list(subject_path_1.entities().fetch())
        images = list(image_path.entities().fetch()) + list(image_path_1.entities().fetch())

        catalog_dataset = set([frozenset(d.items()) for d in datasets])
        catalog_subject = set([frozenset(s.items()) for s in subjects])
        catalog_image = set([frozenset(i.items()) for i in images])

        dataset_table = set([frozenset(d.items()) for d in bag.get_table_as_dict("Dataset")])
        subject_table = set([frozenset(s.items()) for s in bag.get_table_as_dict("Subject")])
        image_table = set([frozenset(i.items()) for i in bag.get_table_as_dict("Image")])

        assert len(catalog_dataset) == len(dataset_table)
        assert len(catalog_subject) == len(subject_table)
        assert len(catalog_image) == len(image_table)

        assert catalog_dataset == dataset_table
        assert catalog_subject == subject_table
        assert catalog_image == image_table
