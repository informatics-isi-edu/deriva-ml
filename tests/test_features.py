from derivaml_test import TestDerivaML
from deriva_ml import ColumnDefinition, BuiltinTypes, DatasetSpec


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
        domain_path = self.ml_instance.catalog.getPathBuilder().schemas[
            self.domain_schema
        ]
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

    def test_download_feature(self):
        self.create_features()
        double_nested_dataset, y, z = self.create_nested_dataset()
        bag = self.ml_instance.download_dataset_bag(
            DatasetSpec(
                rid=double_nested_dataset,
                version=self.ml_instance.dataset_version(double_nested_dataset),
            )
        )
        s_features = [
            f"{f.target_table.name}:{f.feature_name}"
            for f in self.ml_instance.find_features("Subject")
        ]
        s_features_bag = [
            f"{f.target_table.name}:{f.feature_name}"
            for f in bag.find_features("Subject")
        ]
        print(s_features)
        print(s_features_bag)

        for f in self.ml_instance.find_features("Subject"):
            self.assertEqual(
                len(
                    list(
                        self.ml_instance.list_feature_values("Subject", f.feature_name)
                    )
                ),
                len(list(bag.list_feature_values("Subject", f.feature_name))),
            )
        for f in self.ml_instance.find_features("Image"):
            self.assertEqual(
                len(
                    list(self.ml_instance.list_feature_values("Image", f.feature_name))
                ),
                len(list(bag.list_feature_values("Image", f.feature_name))),
            )
