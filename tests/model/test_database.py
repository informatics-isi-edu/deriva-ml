from deriva_ml import DatasetSpec


class TestDataBase:
    def test_table_as_dict(self, test_ml_catalog_dataset):
        ml_instance, double_nested_dataset, nested_datasets, datasets = test_ml_catalog_dataset
        current_version = ml_instance.dataset_version(double_nested_dataset)
        subject_rid = ml_instance.list_dataset_members(datasets[0])["Subject"][0]["RID"]
        ml_instance.add_dataset_members(double_nested_dataset, [subject_rid])
        # new_version = ml_instance.dataset_version(double_nested_dataset)
        bag = ml_instance.download_dataset_bag(DatasetSpec(rid=double_nested_dataset, version=current_version))

        d = bag.get_table_as_dict("Dataset")
        print(list(d))
