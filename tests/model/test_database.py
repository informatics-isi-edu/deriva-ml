from deriva_ml import DatasetSpec
import datetime

class TestDataBaseModel:
    def test_table_as_dict(self, test_ml_catalog_dataset):
        ml_instance = test_ml_catalog_dataset.deriva_ml
        dataset_description = test_ml_catalog_dataset.dataset_description
        reference_datasets = test_ml_catalog_dataset.find_datasets()

        current_version = ml_instance.dataset_version(dataset_description.rid)
        bag = ml_instance.download_dataset_bag(DatasetSpec(rid=dataset_description.rid, version=current_version))

        # Check to make sure that all of the datasets are present.
        assert {r for r in bag.model.bag_rids.keys()} == {r for r in reference_datasets}

        def strip_times(members):
            for m in members:
                rct = members["RCT"]
                rmt = members["RMT"]
                # Fix the timezone to be valid for parsing
                rct_str = rct + (":00" if rmt.endswith("+00") else "")
                rmt_str = rmt + (":00" if rmt.endswith("+00") else "")

                # Parse the datetime
                rct_dt = datetime.fromisoformat(rct_str)
                rmt_dt = datetime.fromisoformat(rmt_str)

                m = m | {"RMT": rmt_dt.isoformat(), "RCT": rct_dt.isoformat()}
                # Convert to ISO 8601 format
            members.sort(key=lambda x: x["RID"])
            return members

        def collect_rids(dataset, rid_list=None):
            if not rid_list:
                rid_list = dataset.member_rids
            for ds in dataset.members():
                rid_list = {t: rid_list[t] + rids for t, rids in collect_rids(ds, rid_list)}
            return rid_list

        dataset = dataset_description
        dataset_bag = bag.model.get_dataset(dataset.rid)
        dataset_rid = dataset.rid
        print(dataset_rid)
        print(dataset_bag.model.snaptime)

        # Now look in the dataset to see if contents match
        for table, members in ml_instance.list_dataset_members(dataset_description.rid, recurse=True).items():
            print(f"checking {table} {members[0]['RCT']} {next(dataset_bag.get_table_as_dict(table))['RCT']}")
            # dataset table includes dataset you are looking for.
            bag_members = strip_times([m for m in dataset_bag.get_table_as_dict(table) if m["RID"] != dataset_rid])
            assert len(members) == len(bag_members)
            assert strip_times(members) == bag_members

        dataset_member_rids = collect_rids(dataset)
        for table in ["Subject", "Image"]:
            bag_rids = dataset_bag.get_table_as_dict(table)
            assert bag_rids == dataset_member_rids
            print(f"checking {table}")
            assert len(list(dataset_bag.get_table_as_dict(table))) == len(members) + (1 if table == "Dataset" else 0)
