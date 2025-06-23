from deriva_ml import DatasetSpec


class TestDataBase:
    def test_table_as_dict(self, test_ml_catalog_dataset):
        ml_instance = test_ml_catalog_dataset.deriva_ml
        double_nested_dataset = test_ml_catalog_dataset.double_nested_dataset
        nested_datasets = test_ml_catalog_dataset.nested_datasets
        datasets = test_ml_catalog_dataset.datasets

        current_version = ml_instance.dataset_version(double_nested_dataset)
        subject_rid = ml_instance.list_dataset_members(datasets[0])["Subject"][0]["RID"]
        ml_instance.add_dataset_members(double_nested_dataset, [subject_rid])
        # new_version = ml_instance.dataset_version(double_nested_dataset)
        bag = ml_instance.download_dataset_bag(DatasetSpec(rid=double_nested_dataset, version=current_version))

        for table, members in ml_instance.list_dataset_members(double_nested_dataset, recurse=True).items():
            print(f"checking {table}")
            assert len(list(bag.get_table_as_dict(table))) == len(members) + (1 if table == "Dataset" else 0)

        for d in nested_datasets:
            ds_bag = bag.get_dataset_bag(d)
            for table, members in ml_instance.list_dataset_members(d, recurse=True).items():
                print(f"checking {table}")
                assert len(list(ds_bag.get_table_as_dict(table))) == len(members) + (1 if table == "Dataset" else 0)

        for d in datasets:
            ds_bag = bag.get_dataset_bag(d)
            for table, members in ml_instance.list_dataset_members(d, recurse=True).items():
                print(f"checking {table}")
                assert len(list(ds_bag.get_table_as_dict(table))) == len(members) + (1 if table == "Dataset" else 0)

