from pathlib import Path

# Local imports
from deriva_ml import DatasetSpec, DerivaML, MLVocab
from deriva_ml.demo_catalog import DatasetDescription
from tests.test_utils import MLDatasetCatalog


class TestDatasetDownload:
    def list_datasets(self, dataset_description: DatasetDescription) -> set[str]:
        nested_datasets = {
            ds
            for dset_member in dataset_description.members.get("Dataset", [])
            for ds in self.list_datasets(dset_member)
        }
        return {dataset_description.rid} | nested_datasets

    def compare_datasets(
        self, ml_instance: DerivaML, dataset: MLDatasetCatalog, dataset_spec: DatasetSpec, recurse=False
    ):
        reference_datasets = self.list_datasets(dataset.dataset_description)

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
            dataset_bag = bag.model.get_dataset(dataset_rid)
            catalog_elements = snapshot_catalog.list_dataset_members(dataset_rid, recurse=recurse)
            bag_elements = dataset_bag.list_dataset_members(recurse=recurse)
            assert len(catalog_elements) == len(bag_elements)

            for t, members in catalog_elements.items():
                print("Checking element", t, len(members), len(bag_elements[t]))
                bag_members = bag_elements[t]
                bag_members.sort(key=lambda x: x["RID"])
                members.sort(key=lambda x: x["RID"])
                assert len(members) == len(bag_elements[t])
                for m, bm in zip(members, bag_members):
                    skip_keys = ["Description", "RMT", "RCT", "RCB", "RMB", "Filename"]
                    m = {k: v for k, v in m.items() if k not in skip_keys}
                    bm = {k: v for k, v in bm.items() if k not in skip_keys}
                    assert m == bm

    def test_bag_dataset_find(self, dataset_test, tmp_path):
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)
        dataset_description = dataset_test.dataset_description
        current_version = ml_instance.dataset_version(dataset_description.rid)
        dataset_spec = DatasetSpec(rid=dataset_description.rid, version=current_version)
        bag = ml_instance.download_dataset_bag(dataset_spec)

        reference_datasets = {ds.rid for ds in dataset_test.list_datasets(dataset_description)}
        assert reference_datasets == {ds["RID"] for ds in bag.model.find_datasets()}

        for ds in bag.model.find_datasets():
            dataset_types = ds["Dataset_Type"]
            for t in dataset_types:
                assert bag.lookup_term(MLVocab.dataset_type, t) is not None

        # Now check top level nesting
        assert set(dataset_description.member_rids["Dataset"]) == set(
            ml_instance.list_dataset_children(dataset_description.rid)
        )
        # Now look two levels down
        for ds in dataset_description.members["Dataset"]:
            assert set(ds.member_rids["Dataset"]) == set(ml_instance.list_dataset_children(ds.rid))

        def check_relationships(description: DatasetDescription):
            """Check relationships between datasets."""
            dataset_children = ml_instance.list_dataset_children(description.rid)
            assert set(description.member_rids.get("Dataset", [])) == set(dataset_children)

            for child in dataset_children:
                assert ml_instance.list_dataset_parents(child)[0] == description.rid
            for nested_dataset in description.members.get("Dataset", []):
                check_relationships(nested_dataset)

        check_relationships(dataset_description)

    def test_dataset_download(self, dataset_test, tmp_path):
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)
        dataset_description = dataset_test.dataset_description

        current_version = ml_instance.dataset_version(dataset_description.rid)
        dataset_spec = DatasetSpec(rid=dataset_description.rid, version=current_version)

        self.compare_datasets(ml_instance, dataset_test, dataset_spec)

    def test_dataset_download_recurse(self, dataset_test, tmp_path):
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)
        dataset_description = dataset_test.dataset_description
        reference_datasets = dataset_test.list_datasets(dataset_description)

        current_version = ml_instance.dataset_version(dataset_description.rid)
        dataset_spec = DatasetSpec(rid=dataset_description.rid, version=current_version)
        bag = ml_instance.download_dataset_bag(dataset_spec)

        for dataset in reference_datasets:
            reference_members = dataset_test.collect_rids(dataset)
            member_rids = {dataset.rid}
            dataset_bag = bag.model.get_dataset(dataset.rid)
            for member_type, dataset_members in dataset_bag.list_dataset_members(recurse=True).items():
                if member_type == "File":
                    continue
                member_rids |= {e["RID"] for e in dataset_members}
            assert reference_members == member_rids

    def test_dataset_download_versions(self, dataset_test, tmp_path):
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)
        dataset_description = dataset_test.dataset_description

        current_version = ml_instance.dataset_version(dataset_description.rid)
        current_spec = DatasetSpec(rid=dataset_description.rid, version=current_version)
        self.compare_datasets(ml_instance, dataset_test, current_spec)

        pb = ml_instance.pathBuilder
        subjects = [s["RID"] for s in pb.schemas[ml_instance.domain_schema].tables["Subject"].path.entities().fetch()]

        ml_instance.add_dataset_members(dataset_description.rid, subjects[-2:])
        new_version = ml_instance.dataset_version(dataset_description.rid)
        new_spec = DatasetSpec(rid=dataset_description.rid, version=new_version)
        current_bag = ml_instance.download_dataset_bag(current_spec)
        new_bag = ml_instance.download_dataset_bag(new_spec)
        print([m["RID"] for m in ml_instance.list_dataset_members(dataset_description.rid)["Subject"]])
        subjects_current = list(current_bag.get_table_as_dict("Subject"))
        subjects_new = list(new_bag.get_table_as_dict("Subject"))

        # Make sure that there is a difference between to old and new catalogs.
        #    assert len(subjects_new) == len(subjects_current) + 2

        self.compare_datasets(ml_instance, dataset_test, current_spec)
        self.compare_datasets(ml_instance, dataset_test, new_spec)
