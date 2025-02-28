from derivaml_test import TestDerivaML
from deriva_ml import DatasetSpec
from pathlib import Path
from pprint import pprint


class TestDownload(TestDerivaML):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_download(self):
        double_nested_dataset, nested_datasets, datasets = self.create_nested_dataset()
        current_version = self.ml_instance.dataset_version(double_nested_dataset)
        subject_rid = self.ml_instance.list_dataset_members(datasets[0])["Subject"][0][
            "RID"
        ]
        self.ml_instance.add_dataset_members(double_nested_dataset, [subject_rid])
        new_version = self.ml_instance.dataset_version(double_nested_dataset)
        bag = self.ml_instance.download_dataset_bag(
            DatasetSpec(rid=double_nested_dataset, version=current_version)
        )
        new_bag = self.ml_instance.download_dataset_bag(
            DatasetSpec(rid=double_nested_dataset, version=new_version)
        )

        # The datasets in the bag should be all the datasets we started with.
        self.assertEqual(
            set([double_nested_dataset] + nested_datasets + datasets),
            {k for k in bag.model.bag_rids.keys()},
        )

        # Children of top level bag should be in datasets variable
        self.assertCountEqual(
            nested_datasets, {ds.dataset_rid for ds in bag.list_dataset_children()}
        )

        self.assertCountEqual(
            nested_datasets + datasets,
            {ds.dataset_rid for ds in bag.list_dataset_children(recurse=True)},
        )

        # Check to see if all of the files have been downloaded.
        files = [Path(r["Filename"]) for r in bag.get_table_as_dict("Image")]
        for f in files:
            self.assertTrue(f.exists())

        self.assertEqual(1, len(new_bag.list_dataset_members()["Subject"]))
        self.assertEqual(0, len(bag.list_dataset_members()["Subject"]))

        children = new_bag.list_dataset_children()
        print(children)
        for c in children:
            print(c)
            pprint(c.list_dataset_members())
            pprint(list(c.get_table_as_dict("Image")))
            c.get_table_as_dict("Subject")
