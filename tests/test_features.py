@unittest.skipUnless(hostname, "Test host not specified")
class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.ml_instance = DerivaML(
            hostname, test_catalog.catalog_id, SNAME_DOMAIN, "", "", "1"
        )
        self.domain_schema = self.ml_instance.model.schemas[SNAME_DOMAIN]
        self.model = self.ml_instance.model

    def test_create_feature(self):
        populate_test_catalog(self.ml_instance, SNAME_DOMAIN)
        self.ml_instance.create_vocabulary("FeatureValue", "A vocab")
        self.ml_instance.add_term("FeatureValue", "V1", description="A Feature Vale")

        a = self.ml_instance.create_asset("TestAsset", comment="A asset")

        self.ml_instance.create_feature(
            "Feature1",
            "Image",
            terms=["FeatureValue"],
            assets=[a],
            metadata=[ColumnDefinition(name="TestCol", type=BuiltinTypes.int2)],
        )
        self.assertIn(
            "Execution_Image_Feature1",
            [f.name for f in self.ml_instance.find_features("Image")],
        )

    def test_add_feature(self):
        self.test_create_feature()
        TestFeature = self.ml_instance.feature_record_class("Image", "Feature1")
        # Create the name for this feature and then create the feature.
        # Get some images to attach the feature value to.
        domain_path = self.ml_instance.catalog.getPathBuilder().schemas[SNAME_DOMAIN]
        image_rids = [i["RID"] for i in domain_path.tables["Image"].entities().fetch()]
        asset_rid = domain_path.tables["TestAsset"].insert(
            [{"Name": "foo", "URL": "foo/bar", "Length": 2, "MD5": 4}]
        )[0]["RID"]
        # Get an execution RID.
        ml_path = self.ml_instance.catalog.getPathBuilder().schemas["deriva-ml"]
        self.ml_instance.add_term(
            "Workflow_Type", "TestWorkflow", description="A workflow"
        )
        workflow_rid = ml_path.tables["Workflow"].insert(
            [{"Name": "Test Workflow", "Workflow_Type": "TestWorkflow"}]
        )[0]["RID"]
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
