"""
Tests for dataset functionality.
"""

from deriva_ml import DatasetSpec


def test_dataset_find(test_ml_catalog):
    """Test finding datasets."""
    # Find all datasets
    datasets = test_ml_catalog.find_datasets()
    assert len(datasets) > 0

    # Verify dataset types exist
    dataset_types = {ds["Dataset_Type"] for ds in datasets}
    assert "Training" in dataset_types
    assert "Testing" in dataset_types
    assert "Partitioned" in dataset_types


def test_dataset_version(test_ml_catalog):
    """Test dataset versioning."""
    # Get a dataset RID
    datasets = test_ml_catalog.find_datasets()
    dataset_rid = datasets[0]["RID"]

    # Get version
    version = test_ml_catalog.dataset_version(dataset_rid)
    assert version is not None
    assert isinstance(version, str)


def test_dataset_spec():
    """Test DatasetSpec creation and validation."""
    # Create with required fields
    spec = DatasetSpec(rid="1234", version="1.0")
    assert spec.rid == "1234"
    assert spec.version == "1.0"
    assert not spec.materialize  # Default value

    # Create with all fields
    spec = DatasetSpec(rid="1234", version="1.0", materialize=True)
    assert spec.materialize


def test_dataset_creation(test_ml_catalog):
    """Test dataset creation and modification."""
    # Find existing datasets for reference
    existing = test_ml_catalog.find_datasets()
    initial_count = len(existing)

    # Create a new dataset
    dataset = test_ml_catalog.create_dataset(
        name="Test Dataset", description="Dataset for testing", dataset_type="Testing"
    )
    assert dataset is not None

    # Verify dataset was created
    updated = test_ml_catalog.find_datasets()
    assert len(updated) == initial_count + 1

    # Find the new dataset
    new_dataset = next(ds for ds in updated if ds["Name"] == "Test Dataset")
    assert new_dataset["Description"] == "Dataset for testing"
    assert new_dataset["Dataset_Type"] == "Testing"


def test_dataset_relationships(test_ml_catalog):
    """Test dataset relationship management."""
    # Create two datasets
    parent = test_ml_catalog.create_dataset(
        name="Parent Dataset", description="Parent for testing", dataset_type="Training"
    )
    child = test_ml_catalog.create_dataset(
        name="Child Dataset", description="Child for testing", dataset_type="Testing"
    )

    # Link datasets
    test_ml_catalog.link_datasets(parent_rid=parent["RID"], child_rid=child["RID"], relationship_type="Derived")

    # Verify relationship
    children = test_ml_catalog.get_dataset_children(parent["RID"])
    assert any(c["RID"] == child["RID"] for c in children)

    parents = test_ml_catalog.get_dataset_parents(child["RID"])
    assert any(p["RID"] == parent["RID"] for p in parents)
