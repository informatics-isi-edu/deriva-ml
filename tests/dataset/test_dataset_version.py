from demo_catalog import DatasetDescription
from deriva_ml import (
    BuiltinTypes,
    ColumnDefinition,
    DatasetSpec,
    DatasetVersion,
    ExecutionConfiguration,
    MLVocab,
    TableDefinition,
    VersionPart,
)

class TestDatasetVersion:
    def test_dataset_version_simple(self, test_ml_catalog):
        ml_instance = test_ml_catalog
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