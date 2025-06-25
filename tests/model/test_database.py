from deriva_ml import DatasetSpec


class TestDataBase:
    def test_table_as_dict(self, test_ml_catalog_dataset):
        ml_instance = test_ml_catalog_dataset.deriva_ml
        dataset_description = test_ml_catalog_dataset.dataset_description
        reference_datasets = test_ml_catalog_dataset.find_datasets()

        current_version = ml_instance.dataset_version(dataset_description.rid)
        bag = ml_instance.download_dataset_bag(DatasetSpec(rid=dataset_description.rid, version=current_version))

        # Check to make sure that all of the datasets are present.
        assert {r for r in bag.model.bag_rids.keys()} == {r for r in reference_datasets}

        def strip_times(members):
            members =  [m | {"RMT": 0, "RCT": 0} for m in members]
            members.sort(key=lambda x: x["RID"])
            return members

        # Now look in the dataset to see if contents match
        for table, members in ml_instance.list_dataset_members(dataset_description.rid, recurse=True).items():
            print(f"checking {table}")
            bag_members = strip_times(bag.get_table_as_dict(table))

            print(bag_members)
            print(strip_times(members))
            assert bag_members == strip_times(members)

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
