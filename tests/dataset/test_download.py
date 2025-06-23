from pathlib import Path

from deriva_ml import DatasetSpec


class TestDownload:
    def test_download(self, test_ml_catalog_dataset):
        ml_instance = test_ml_catalog_dataset.deriva_ml
        double_nested_dataset = test_ml_catalog_dataset.double_nested_dataset
        nested_datasets = test_ml_catalog_dataset.nested_datasets
        datasets = test_ml_catalog_dataset.datasets

        current_version = ml_instance.dataset_version(double_nested_dataset)
        subject_rid = ml_instance.list_dataset_members(datasets[0])["Subject"][0]["RID"]
        ml_instance.add_dataset_members(double_nested_dataset, [subject_rid])
        # new_version = ml_instance.dataset_version(double_nested_dataset)
        bag = ml_instance.download_dataset_bag(DatasetSpec(rid=double_nested_dataset, version=current_version))
        # new_bag = ml_instance.download_dataset_bag(DatasetSpec(rid=double_nested_dataset, version=new_version))

        # The datasets in the bag should be all the datasets we started with.
        assert set([double_nested_dataset] + nested_datasets + datasets) == {k for k in bag.model.bag_rids.keys()}

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
            print(ds)
            bag_members = bag.model.list_dataset_members(ds["RID"])
            catalog_members = ml_instance.list_dataset_members(ds["RID"])
            assert strip_times(bag_members) == strip_times(catalog_members)
