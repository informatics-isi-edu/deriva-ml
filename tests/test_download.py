from derivaml_test import TestDerivaML
from deriva_ml.demo_catalog import (
    reset_demo_catalog,
    populate_demo_catalog,
    create_demo_datasets,
)


class TestDownload(TestDerivaML):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_download(self):
        populate_demo_catalog(self.ml_instance, self.domain_schema)
        create_demo_datasets(self.ml_instance)
        nested_dataset_rid = [
            ds["RID"]
            for ds in self.ml_instance.find_datasets()
            if "Partitioned" in ds["Dataset_Type"]
        ][0]
        bag = self.ml_instance.download_dataset_bag(nested_dataset_rid)
