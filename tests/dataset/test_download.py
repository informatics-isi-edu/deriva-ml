from pathlib import Path

from deriva_ml import DatasetSpec


class TestDownload:
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
        # new_bag = ml_instance.download_dataset_bag(DatasetSpec(rid=double_nested_dataset, version=new_version))

        # The datasets in the bag should be all the datasets we started with.
        assert set([dataset_description.rid] + nested_datasets + datasets) == {k for k in bag.model.bag_rids.keys()}

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

        for ds in ml_instance.find_datasets():
            bag_members = {
                table: {m["RID"] for m in members}
                for table, members in bag.model.list_dataset_members(ds["RID"]).items()
            }
            catalog_members = {
                table: {m["RID"] for m in members}
                for table, members in ml_instance.list_dataset_members(ds["RID"]).items()
            }
            print(bag_members)
            print(catalog_members)
            assert bag_members == catalog_members
