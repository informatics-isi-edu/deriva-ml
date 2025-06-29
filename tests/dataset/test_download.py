from pathlib import Path

from conftest import MLDatasetTest

# Local imports
from deriva_ml import DatasetSpec, DerivaML


class TestDatasetDownload:
    def compare_datasets(self, ml_instance: DerivaML, dataset: MLDatasetTest, dataset_spec: DatasetSpec):
        reference_datasets = dataset.find_datasets()

        snapshot_catalog = DerivaML(ml_instance.host_name, ml_instance._version_snapshot(dataset_spec))
        bag = ml_instance.download_dataset_bag(dataset_spec)

        # Check to see if all of the files have been downloaded.
        files = [Path(r["Filename"]) for r in bag.get_table_as_dict("Image")]
        for f in files:
            assert f.exists()

        # Check to make sure that all of the datasets are present.
        assert {r for r in bag.model.bag_rids.keys()} == {r for r in reference_datasets}

        # Now look at each dataset to see if they line up.
        for dataset_rid in reference_datasets:
            print(f"Checking {dataset_rid} {reference_datasets[dataset_rid]} ")
            dataset_bag = bag.model.get_dataset(dataset_rid)
            dataset_rids = tuple(ml_instance.list_dataset_children(dataset_rid, recurse=True) + [dataset_rid])
            bag_rids = [b.dataset_rid for b in dataset_bag.list_dataset_children(recurse=True)] + [
                dataset_bag.dataset_rid
            ]
            assert list(dataset_rids).sort() == bag_rids.sort()

            catalog__elements = snapshot_catalog.list_dataset_members(dataset_rid)
            bag_elements = dataset_bag.list_dataset_members()

            assert len(catalog__elements) == len(bag_elements)
            for t, members in catalog__elements.items():
                assert len(members) == len(bag_elements[t])
            for t, members in catalog__elements.items():
                for m, bm in zip(members, bag_elements[t]):
                    m.pop("RCT", None)
                    m.pop("RMT", None)
                    bm.pop("RCT", None)
                    bm.pop("RMT", None)
                    assert m == bm

    def test_dataset_download(self, test_ml_catalog_dataset):
        ml_instance = test_ml_catalog_dataset.deriva_ml
        dataset_description = test_ml_catalog_dataset.dataset_description

        current_version = ml_instance.dataset_version(dataset_description.rid)
        dataset_spec = DatasetSpec(rid=dataset_description.rid, version=current_version)

        self.compare_datasets(ml_instance, test_ml_catalog_dataset, dataset_spec)

    def test_table_versions(self, test_ml_catalog_dataset):
        ml_instance = test_ml_catalog_dataset.deriva_ml
        dataset_description = test_ml_catalog_dataset.dataset_description

        current_version = ml_instance.dataset_version(dataset_description.rid)
        current_spec = DatasetSpec(rid=dataset_description.rid, version=current_version)
        self.compare_datasets(
            ml_instance, test_ml_catalog_dataset, DatasetSpec(rid=dataset_description.rid, version=current_version)
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
        assert len(subjects_new) == len(subjects_current) + 2

        self.compare_datasets(ml_instance, test_ml_catalog_dataset, current_spec)
        self.compare_datasets(ml_instance, test_ml_catalog_dataset, new_spec)
