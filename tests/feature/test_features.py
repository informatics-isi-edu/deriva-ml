"""
Tests for feature functionality.
"""


def test_feature_creation(test_ml_catalog):
    """Test feature creation and management."""
    # Create a feature
    feature = test_ml_catalog.create_feature(
        name="Test Feature", description="Feature for testing", feature_type="Numeric", value_type="float"
    )
    assert feature is not None
    assert feature["Name"] == "Test Feature"
    assert feature["Description"] == "Feature for testing"

    # Find the feature
    features = test_ml_catalog.find_features()
    assert any(f["RID"] == feature["RID"] for f in features)


def test_feature_set_creation(test_ml_catalog):
    """Test feature set creation and management."""
    # Create features
    feature1 = test_ml_catalog.create_feature(
        name="Feature 1", description="First test feature", feature_type="Numeric", value_type="float"
    )
    feature2 = test_ml_catalog.create_feature(
        name="Feature 2", description="Second test feature", feature_type="Categorical", value_type="string"
    )

    # Create feature set
    feature_set = test_ml_catalog.create_feature_set(
        name="Test Feature Set", description="Feature set for testing", features=[feature1["RID"], feature2["RID"]]
    )
    assert feature_set is not None

    # Verify feature set contents
    set_features = test_ml_catalog.get_feature_set_features(feature_set["RID"])
    assert len(set_features) == 2
    feature_rids = {f["RID"] for f in set_features}
    assert feature1["RID"] in feature_rids
    assert feature2["RID"] in feature_rids


def test_feature_value_types(test_ml_catalog):
    """Test different feature value types."""
    # Test numeric feature
    numeric = test_ml_catalog.create_feature(
        name="Numeric Feature",
        description="Test numeric feature",
        feature_type="Numeric",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
    )
    assert numeric["Value_Type"] == "float"
    assert numeric["Min_Value"] == 0.0
    assert numeric["Max_Value"] == 1.0

    # Test categorical feature
    categorical = test_ml_catalog.create_feature(
        name="Categorical Feature",
        description="Test categorical feature",
        feature_type="Categorical",
        value_type="string",
        categories=["A", "B", "C"],
    )
    assert categorical["Value_Type"] == "string"
    assert set(categorical["Categories"]) == {"A", "B", "C"}

    # Test boolean feature
    boolean = test_ml_catalog.create_feature(
        name="Boolean Feature", description="Test boolean feature", feature_type="Boolean", value_type="boolean"
    )
    assert boolean["Value_Type"] == "boolean"


def test_feature_relationships(test_ml_catalog):
    """Test feature relationships and dependencies."""
    # Create features
    base = test_ml_catalog.create_feature(
        name="Base Feature", description="Base feature for testing", feature_type="Numeric", value_type="float"
    )
    derived = test_ml_catalog.create_feature(
        name="Derived Feature", description="Derived feature for testing", feature_type="Numeric", value_type="float"
    )

    # Link features
    test_ml_catalog.link_features(base_rid=base["RID"], derived_rid=derived["RID"], relationship_type="Derived")

    # Verify relationships
    derived_features = test_ml_catalog.get_derived_features(base["RID"])
    assert any(f["RID"] == derived["RID"] for f in derived_features)

    base_features = test_ml_catalog.get_base_features(derived["RID"])
    assert any(f["RID"] == base["RID"] for f in base_features)


def test_feature_set_operations(test_ml_catalog):
    """Test feature set operations."""
    # Create features
    features = [
        test_ml_catalog.create_feature(
            name=f"Feature {i}", description=f"Test feature {i}", feature_type="Numeric", value_type="float"
        )
        for i in range(3)
    ]

    # Create feature sets
    set1 = test_ml_catalog.create_feature_set(
        name="Set 1", description="First feature set", features=[features[0]["RID"], features[1]["RID"]]
    )
    set2 = test_ml_catalog.create_feature_set(
        name="Set 2", description="Second feature set", features=[features[1]["RID"], features[2]["RID"]]
    )

    # Test union
    union = test_ml_catalog.feature_set_union(
        name="Union Set", description="Union of sets", feature_sets=[set1["RID"], set2["RID"]]
    )
    union_features = test_ml_catalog.get_feature_set_features(union["RID"])
    assert len(union_features) == 3

    # Test intersection
    intersection = test_ml_catalog.feature_set_intersection(
        name="Intersection Set", description="Intersection of sets", feature_sets=[set1["RID"], set2["RID"]]
    )
    intersection_features = test_ml_catalog.get_feature_set_features(intersection["RID"])
    assert len(intersection_features) == 1
    assert intersection_features[0]["RID"] == features[1]["RID"]


from deriva_ml import BuiltinTypes, ColumnDefinition, DatasetSpec


class TestFeatures(TestDerivaML):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_create_feature(self):
        self.populate_catalog()
        self.ml_instance.create_vocabulary("FeatureValue", "A vocab")
        self.ml_instance.add_term("FeatureValue", "V1", description="A Feature Value")

        a = self.ml_instance.create_asset("TestAsset", comment="A asset")

        self.ml_instance.create_feature(
            feature_name="Feature1",
            target_table="Image",
            terms=["FeatureValue"],
            assets=[a],
            metadata=[ColumnDefinition(name="TestCol", type=BuiltinTypes.int2)],
        )
        self.assertIn(
            "Feature1",
            [f.feature_name for f in self.ml_instance.find_features("Image")],
        )
        self.assertIn(
            "Execution_Image_Feature1",
            [f.feature_table.name for f in self.ml_instance.find_features("Image")],
        )

    def test_add_feature(self):
        self.test_create_feature()
        TestFeature = self.ml_instance.feature_record_class("Image", "Feature1")
        # Create the name for this feature and then create the feature.
        # Get some images to attach the feature value to.
        domain_path = self.ml_instance.catalog.getPathBuilder().schemas[self.domain_schema]
        image_rids = [i["RID"] for i in domain_path.tables["Image"].entities().fetch()]
        asset_rid = domain_path.tables["TestAsset"].insert([{"Name": "foo", "URL": "foo/bar", "Length": 2, "MD5": 4}])[
            0
        ]["RID"]
        # Get an execution RID.
        ml_path = self.ml_instance.catalog.getPathBuilder().schemas["deriva-ml"]
        self.ml_instance.add_term("Workflow_Type", "TestWorkflow", description="A workflow")
        workflow_rid = ml_path.tables["Workflow"].insert([{"Name": "Test Workflow", "Workflow_Type": "TestWorkflow"}])[
            0
        ]["RID"]
        execution_rid = ml_path.tables["Execution"].insert(
            [{"Description": "Test execution", "Workflow": workflow_rid}]
        )[0]["RID"]
        # Now create a list of features using the feature creation class returned by create_feature.
        feature_list = [
            TestFeature(
                Image=i,
                Execution=execution_rid,
                FeatureValue="V1",
                TestAsset=asset_rid,
                TestCol=23,
            )
            for i in image_rids
        ]
        self.ml_instance.add_features(feature_list)
        features = self.ml_instance.list_feature_values("Image", "Feature1")
        self.assertEqual(len(features), len(image_rids))

    def test_download_feature(self):
        self.create_features()
        double_nested_dataset, y, z = self.create_nested_dataset()
        bag = self.ml_instance.download_dataset_bag(
            DatasetSpec(
                rid=double_nested_dataset,
                version=self.ml_instance.dataset_version(double_nested_dataset),
            )
        )
        s_features = [f"{f.target_table.name}:{f.feature_name}" for f in self.ml_instance.find_features("Subject")]
        s_features_bag = [f"{f.target_table.name}:{f.feature_name}" for f in bag.find_features("Subject")]
        print(s_features)
        print(s_features_bag)

        for f in self.ml_instance.find_features("Subject"):
            self.assertEqual(
                len(list(self.ml_instance.list_feature_values("Subject", f.feature_name))),
                len(list(bag.list_feature_values("Subject", f.feature_name))),
            )
        for f in self.ml_instance.find_features("Image"):
            self.assertEqual(
                len(list(self.ml_instance.list_feature_values("Image", f.feature_name))),
                len(list(bag.list_feature_values("Image", f.feature_name))),
            )
