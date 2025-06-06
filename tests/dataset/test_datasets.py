"""
Tests for dataset functionality.
"""

import pytest

from deriva_ml import DatasetSpec


def test_dataset_find(demo_ml):
    """Test finding datasets."""
    # Find all datasets
    datasets = demo_ml.find_datasets()
    assert len(datasets) > 0
    
    # Verify dataset types exist
    dataset_types = {ds["Dataset_Type"] for ds in datasets}
    assert "Training" in dataset_types
    assert "Testing" in dataset_types
    assert "Partitioned" in dataset_types


def test_dataset_version(demo_ml):
    """Test dataset versioning."""
    # Get a dataset RID
    datasets = demo_ml.find_datasets()
    dataset_rid = datasets[0]["RID"]
    
    # Get version
    version = demo_ml.dataset_version(dataset_rid)
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


def test_dataset_creation(demo_ml):
    """Test dataset creation and modification."""
    # Find existing datasets for reference
    existing = demo_ml.find_datasets()
    initial_count = len(existing)
    
    # Create a new dataset
    dataset = demo_ml.create_dataset(
        name="Test Dataset",
        description="Dataset for testing",
        dataset_type="Testing"
    )
    assert dataset is not None
    
    # Verify dataset was created
    updated = demo_ml.find_datasets()
    assert len(updated) == initial_count + 1
    
    # Find the new dataset
    new_dataset = next(
        ds for ds in updated 
        if ds["Name"] == "Test Dataset"
    )
    assert new_dataset["Description"] == "Dataset for testing"
    assert new_dataset["Dataset_Type"] == "Testing"


def test_dataset_relationships(demo_ml):
    """Test dataset relationship management."""
    # Create two datasets
    parent = demo_ml.create_dataset(
        name="Parent Dataset",
        description="Parent for testing",
        dataset_type="Training"
    )
    child = demo_ml.create_dataset(
        name="Child Dataset",
        description="Child for testing",
        dataset_type="Testing"
    )
    
    # Link datasets
    demo_ml.link_datasets(
        parent_rid=parent["RID"],
        child_rid=child["RID"],
        relationship_type="Derived"
    )
    
    # Verify relationship
    children = demo_ml.get_dataset_children(parent["RID"])
    assert any(c["RID"] == child["RID"] for c in children)
    
    parents = demo_ml.get_dataset_parents(child["RID"])
    assert any(p["RID"] == parent["RID"] for p in parents) 