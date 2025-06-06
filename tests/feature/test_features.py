"""
Tests for feature functionality.
"""

import pytest

from deriva_ml import Feature, FeatureSet


def test_feature_creation(demo_ml):
    """Test feature creation and management."""
    # Create a feature
    feature = demo_ml.create_feature(
        name="Test Feature",
        description="Feature for testing",
        feature_type="Numeric",
        value_type="float"
    )
    assert feature is not None
    assert feature["Name"] == "Test Feature"
    assert feature["Description"] == "Feature for testing"
    
    # Find the feature
    features = demo_ml.find_features()
    assert any(f["RID"] == feature["RID"] for f in features)


def test_feature_set_creation(demo_ml):
    """Test feature set creation and management."""
    # Create features
    feature1 = demo_ml.create_feature(
        name="Feature 1",
        description="First test feature",
        feature_type="Numeric",
        value_type="float"
    )
    feature2 = demo_ml.create_feature(
        name="Feature 2",
        description="Second test feature",
        feature_type="Categorical",
        value_type="string"
    )
    
    # Create feature set
    feature_set = demo_ml.create_feature_set(
        name="Test Feature Set",
        description="Feature set for testing",
        features=[feature1["RID"], feature2["RID"]]
    )
    assert feature_set is not None
    
    # Verify feature set contents
    set_features = demo_ml.get_feature_set_features(feature_set["RID"])
    assert len(set_features) == 2
    feature_rids = {f["RID"] for f in set_features}
    assert feature1["RID"] in feature_rids
    assert feature2["RID"] in feature_rids


def test_feature_value_types(demo_ml):
    """Test different feature value types."""
    # Test numeric feature
    numeric = demo_ml.create_feature(
        name="Numeric Feature",
        description="Test numeric feature",
        feature_type="Numeric",
        value_type="float",
        min_value=0.0,
        max_value=1.0
    )
    assert numeric["Value_Type"] == "float"
    assert numeric["Min_Value"] == 0.0
    assert numeric["Max_Value"] == 1.0
    
    # Test categorical feature
    categorical = demo_ml.create_feature(
        name="Categorical Feature",
        description="Test categorical feature",
        feature_type="Categorical",
        value_type="string",
        categories=["A", "B", "C"]
    )
    assert categorical["Value_Type"] == "string"
    assert set(categorical["Categories"]) == {"A", "B", "C"}
    
    # Test boolean feature
    boolean = demo_ml.create_feature(
        name="Boolean Feature",
        description="Test boolean feature",
        feature_type="Boolean",
        value_type="boolean"
    )
    assert boolean["Value_Type"] == "boolean"


def test_feature_relationships(demo_ml):
    """Test feature relationships and dependencies."""
    # Create features
    base = demo_ml.create_feature(
        name="Base Feature",
        description="Base feature for testing",
        feature_type="Numeric",
        value_type="float"
    )
    derived = demo_ml.create_feature(
        name="Derived Feature",
        description="Derived feature for testing",
        feature_type="Numeric",
        value_type="float"
    )
    
    # Link features
    demo_ml.link_features(
        base_rid=base["RID"],
        derived_rid=derived["RID"],
        relationship_type="Derived"
    )
    
    # Verify relationships
    derived_features = demo_ml.get_derived_features(base["RID"])
    assert any(f["RID"] == derived["RID"] for f in derived_features)
    
    base_features = demo_ml.get_base_features(derived["RID"])
    assert any(f["RID"] == base["RID"] for f in base_features)


def test_feature_set_operations(demo_ml):
    """Test feature set operations."""
    # Create features
    features = [
        demo_ml.create_feature(
            name=f"Feature {i}",
            description=f"Test feature {i}",
            feature_type="Numeric",
            value_type="float"
        )
        for i in range(3)
    ]
    
    # Create feature sets
    set1 = demo_ml.create_feature_set(
        name="Set 1",
        description="First feature set",
        features=[features[0]["RID"], features[1]["RID"]]
    )
    set2 = demo_ml.create_feature_set(
        name="Set 2",
        description="Second feature set",
        features=[features[1]["RID"], features[2]["RID"]]
    )
    
    # Test union
    union = demo_ml.feature_set_union(
        name="Union Set",
        description="Union of sets",
        feature_sets=[set1["RID"], set2["RID"]]
    )
    union_features = demo_ml.get_feature_set_features(union["RID"])
    assert len(union_features) == 3
    
    # Test intersection
    intersection = demo_ml.feature_set_intersection(
        name="Intersection Set",
        description="Intersection of sets",
        feature_sets=[set1["RID"], set2["RID"]]
    )
    intersection_features = demo_ml.get_feature_set_features(intersection["RID"])
    assert len(intersection_features) == 1
    assert intersection_features[0]["RID"] == features[1]["RID"] 