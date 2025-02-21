from derivaml_test import TestDerivaML
from deriva_ml.demo_catalog import (
    reset_demo_catalog,
    populate_demo_catalog,
    create_demo_datasets,
)
from deriva_ml import DatasetSpec
from pathlib import Path


class TestDownload(TestDerivaML):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_download(self):
        double_nested_dataset, nested_datasets, datasets = self.create_nested_dataset()
        bag = self.ml_instance.download_dataset_bag(
            DatasetSpec(
                rid=double_nested_dataset,
                version=self.ml_instance.dataset_version(double_nested_dataset),
            )
        )

        self.assertEqual(
            set(nested_datasets), {ds.dataset_rid for ds in bag.list_dataset_children()}
        )

        print(bag.list_dataset_children(recurse=True))
        files = [Path(r["Filename"]) for r in bag.get_table_as_dict("Image")]
        for f in files:
            self.assertTrue(f.exists())
