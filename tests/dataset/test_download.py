from pathlib import Path

# Local imports
from deriva_ml import DatasetSpec


class TestDatasetDownload:
    def test_download(self, test_ml_catalog_dataset):
        ml_instance = test_ml_catalog_dataset.deriva_ml
        dataset_description = test_ml_catalog_dataset.dataset_description

        nested_datasets = dataset_description.member_rids.get("Dataset", [])
        datasets = [
            dataset
            for nested_description in dataset_description.members.get("Dataset", [])
            for dataset in nested_description.member_rids.get("Dataset", [])
        ]

        current_version = ml_instance.dataset_version(dataset_description.rid)
        # subject_rid = ml_instance.list_dataset_members(datasets[0])["Subject"][0]["RID"]
        # ml_instance.add_dataset_members(dataset_description.rid, [subject_rid])
        # new_version = ml_instance.dataset_version(double_nested_dataset)
        bag = ml_instance.download_dataset_bag(DatasetSpec(rid=dataset_description.rid, version=current_version))
        assert current_version == bag.version
        # new_bag = ml_instance.download_dataset_bag(DatasetSpec(rid=double_nested_dataset, version=new_version))

        # The datasets in the bag should be all the datasets we started with.
        assert set([dataset_description.rid] + nested_datasets + datasets) == {k for k in bag.model.bag_rids.keys()}
        assert len(bag.list_dataset_children()) == len(ml_instance.list_dataset_children(dataset_description.rid))
        assert len(bag.list_dataset_children(recurse=True)) == len(
            ml_instance.list_dataset_children(dataset_description.rid, recurse=True)
        )

        # Children of top level bag should be in datasets variable
        assert set(nested_datasets) == {ds.dataset_rid for ds in bag.list_dataset_children()}
        assert set(nested_datasets + datasets) == {ds.dataset_rid for ds in bag.list_dataset_children(recurse=True)}

        # Check to see if all of the files have been downloaded.
        files = [Path(r["Filename"]) for r in bag.get_table_as_dict("Image")]
        for f in files:
            assert f.exists()

        def strip_times(members):
            return {
                element: [{k: v for k, v in m.items() if k not in ["RCT", "RMT"]} for m in mlist]
                for element, mlist in members.items()
            }

        assert ml_instance.list_dataset_members(dataset_description.rid) == bag.list_dataset_members()

        # Check all of the datasets to see if they have the same members in the bag and catalog.
        for ds in ml_instance.find_datasets():
            dataset_bag = bag.model.get_dataset(ds["RID"])
            bag_members = dataset_bag.list_dataset_members()
            catalog_members = ml_instance.list_dataset_members(ds["RID"])
            bag_member_rids = {table: {m["RID"] for m in members} for table, members in bag_members.items()}
            catalog_member_rids = {table: {m["RID"] for m in members} for table, members in catalog_members.items()}
            # assert catalog_member_rids == bag_member_rids

            # Now check the actual entries
            bag_members = strip_times(bag_members)
            catalog_members = strip_times(catalog_members)
            assert bag_members == catalog_members
