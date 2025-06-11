"""
Tests for demo catalog functionality.
"""

from deriva_ml.demo_catalog import DemoML, create_demo_catalog


def test_create_demo_catalog(test_host):
    """Test creating a demo catalog."""
    # Create catalog with all features
    catalog = create_demo_catalog(test_host, "test-schema", create_features=True, create_datasets=True)
    assert catalog is not None
    assert catalog.catalog_id is not None

    # Create ML instance
    ml = DemoML(test_host, catalog.catalog_id)

    # Verify features were created
    features = ml.find_features()
    assert len(features) > 0

    # Verify datasets were created
    datasets = ml.find_datasets()
    assert len(datasets) > 0

    # Verify dataset types
    dataset_types = {ds["Dataset_Type"] for ds in datasets}
    assert "Training" in dataset_types
    assert "Testing" in dataset_types
    assert "Partitioned" in dataset_types


def test_demo_catalog_minimal(test_host):
    """Test creating a minimal demo catalog."""
    # Create catalog without features and datasets
    catalog = create_demo_catalog(test_host, "test-schema-minimal", create_features=False, create_datasets=False)
    assert catalog is not None
    assert catalog.catalog_id is not None

    # Create ML instance
    ml = DemoML(test_host, catalog.catalog_id)

    # Verify no features were created
    features = ml.find_features()
    assert len(features) == 0

    # Verify no datasets were created
    datasets = ml.find_datasets()
    assert len(datasets) == 0


def test_demo_ml_instance(test_ml_catalog):
    """Test DemoML instance functionality."""
    # Test vocabulary terms
    terms = test_ml_catalog.get_terms("Dataset_Type")
    assert "Training" in {t["Name"] for t in terms}
    assert "Testing" in {t["Name"] for t in terms}

    # Test feature creation
    feature = test_ml_catalog.create_feature(
        name="Demo Feature", description="Feature for demo testing", feature_type="Numeric", value_type="float"
    )
    assert feature is not None

    # Test dataset creation
    dataset = test_ml_catalog.create_dataset(
        name="Demo Dataset", description="Dataset for demo testing", dataset_type="Training"
    )
    assert dataset is not None

    # Test linking feature to dataset
    test_ml_catalog.add_dataset_features(dataset_rid=dataset["RID"], feature_rids=[feature["RID"]])

    # Verify link
    dataset_features = test_ml_catalog.get_dataset_features(dataset["RID"])
    assert any(f["RID"] == feature["RID"] for f in dataset_features)
