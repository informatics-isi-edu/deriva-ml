from deriva_ml import (
    DatasetVersion,
    DerivaML,
    VersionPart,
)


class TestDatasetVersion:
    def test_dataset_version_simple(self, test_ml):
        ml_instance = test_ml
        type_rid = ml_instance.add_term("Dataset_Type", "TestSet", description="A test")
        dataset_rid = ml_instance.create_dataset(
            type_rid.name,
            description="A New Dataset",
            version=DatasetVersion(1, 0, 0),
        )
        v0 = ml_instance.dataset_version(dataset_rid)
        assert "1.0.0" == str(v0)
        v1 = ml_instance.increment_dataset_version(dataset_rid=dataset_rid, component=VersionPart.minor)
        assert "1.1.0" == str(v1)

    def test_dataset_version_history(self, test_ml):
        ml_instance = test_ml
        type_rid = ml_instance.add_term("Dataset_Type", "TestSet", description="A test")
        dataset_rid = ml_instance.create_dataset(
            type_rid.name,
            description="A New Dataset",
            version=DatasetVersion(1, 0, 0),
        )
        assert 1 == len(ml_instance.dataset_history(dataset_rid))
        v1 = ml_instance.increment_dataset_version(dataset_rid=dataset_rid, component=VersionPart.minor)
        assert 2 == len(ml_instance.dataset_history(dataset_rid))

    def test_dataset_version(self, dataset_test, tmp_path):
        hostname = dataset_test.catalog.hostname
        catalog_id = dataset_test.catalog.catalog_id
        ml_instance = DerivaML(hostname, catalog_id, working_dir=tmp_path, use_minid=False)
        dataset_description = dataset_test.dataset_description

        nested_datasets = dataset_description.member_rids.get("Dataset", [])
        datasets = [
            dataset
            for nested_description in dataset_description.members.get("Dataset", [])
            for dataset in nested_description.member_rids.get("Dataset", [])
        ]
        _versions = {
            "d0": ml_instance.dataset_version(dataset_description.rid),
            "d1": [ml_instance.dataset_version(v) for v in nested_datasets],
            "d2": [ml_instance.dataset_version(v) for v in datasets],
        }
        ml_instance.increment_dataset_version(nested_datasets[0], VersionPart.major)
        new_versions = {
            "d0": ml_instance.dataset_version(dataset_description.rid),
            "d1": [ml_instance.dataset_version(v) for v in nested_datasets],
            "d2": [ml_instance.dataset_version(v) for v in datasets],
        }

        assert new_versions["d0"].major == 2
        assert new_versions["d2"][0].major == 2
