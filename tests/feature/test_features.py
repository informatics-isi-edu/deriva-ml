"""
Tests for feature functionality.
"""

import pytest

from deriva_ml import BuiltinTypes, ColumnDefinition, DatasetSpec, DerivaMLException, ExecutionConfiguration
from deriva_ml import MLVocab as vc


class TestFeatures:
    def test_create_feature(self, test_ml_catalog):
        ml_instance = test_ml_catalog
        ml_instance.create_vocabulary("FeatureValue", "A vocab")
        ml_instance.add_term("FeatureValue", "V1", description="A Feature Value")

        a = ml_instance.create_asset("TestAsset", comment="A asset")

        ml_instance.create_feature(
            feature_name="Feature1",
            target_table="Image",
            terms=["FeatureValue"],
            assets=[a],
            metadata=[ColumnDefinition(name="TestCol", type=BuiltinTypes.int2)],
        )
        ml_instance.create_vocabulary("SubjectHealth", "A vocab")
        ml_instance.add_term(
            "SubjectHealth",
            "Sick",
            description="The subject self reports that they are sick",
        )
        ml_instance.add_term(
            "SubjectHealth",
            "Well",
            description="The subject self reports that they feel well",
        )
        ml_instance.create_vocabulary("ImageQuality", "Controlled vocabulary for image quality")
        ml_instance.add_term("ImageQuality", "Good", description="The image is good")
        ml_instance.add_term("ImageQuality", "Bad", description="The image is bad")
        box_asset = ml_instance.create_asset("BoundingBox", comment="A file that contains a cropped version of a image")

        ml_instance.create_feature(
            "Subject",
            "Health",
            terms=["SubjectHealth"],
            metadata=[ColumnDefinition(name="Scale", type=BuiltinTypes.int2, nullok=True)],
            optional=["Scale"],
        )
        assert "Health" in [f.feature_name for f in ml_instance.model.find_features("Subject")]
        ml_instance.create_feature("Image", "BoundingBox", assets=[box_asset])
        ml_instance.create_feature("Image", "Quality", terms=["ImageQuality"])

        assert "BoundingBox" in [f.feature_name for f in ml_instance.model.find_features("Image")]
        assert "Quality" in [f.feature_name for f in ml_instance.model.find_features("Image")]

    def test_lookup_feature(self, test_ml_catalog_populated):
        ml_instance = test_ml_catalog_populated
        assert "Health" == ml_instance.lookup_feature("Subject", "Health").feature_name
        with pytest.raises(DerivaMLException):
            ml_instance.lookup_feature("Foobar", "Health")

        with pytest.raises(DerivaMLException):
            ml_instance.lookup_feature("Subject", "SubjectHealth1")

    def test_add_feature(self, test_ml_catalog_populated):
        ml_instance = test_ml_catalog_populated
        subject_table = ml_instance.pathBuilder.schemas[ml_instance.domain_schema].tables["Subject"]
        subject_rids = [s["RID"] for s in subject_table.path.entities().fetch()]

        ml_instance.add_term(
            vc.workflow_type, "Test Feature Workflow", description="A ML Workflow that uses Deriva ML API"
        )
        api_workflow = ml_instance.create_workflow(
            name="Test Feature Workflow",
            workflow_type="Test Feature Workflow",
            description="A test operation",
        )
        feature_execution = ml_instance.create_execution(
            ExecutionConfiguration(description="Feature Execution", workflow=api_workflow)
        )

        with feature_execution.execute() as exe:
            SubjectHealthFeature = ml_instance.feature_record_class("Subject", "Health")
            exe.add_features([SubjectHealthFeature(Subject=subject_rids[0], Health="Good", Scale=23)])

            features = ml_instance.list_feature_values("Subject", "Health")
            assert len(features) == 1

            ImageBoundingboxFeature = ml_instance.feature_record_class("Image", "BoundingBox")
            ImageQualtiyFeature = ml_instance.feature_record_class("Image", "Quality")

    def test_download_feature(self, test_ml_catalog_populated):
        ml_instance = test_ml_catalog_populated.ml_instance

        # Add some feature values

        # Download a bag with feature values in it

        # Check to see if feature value is in the bag.
        bag = ml_instance.download_dataset_bag(
            DatasetSpec(
                rid=double_nested_dataset,
                version=self.ml_instance.dataset_version(double_nested_dataset),
            )
        )
        s_features = [f"{f.target_table.name}:{f.feature_name}" for f in self.ml_instance.find_features("Subject")]
        s_features_bag = [f"{f.target_table.name}:{f.feature_name}" for f in bag.find_features("Subject")]
        print(s_features)
        print(s_features_bag)

        for f in ml_instance.find_features("Subject"):
            assert len(list(ml_instance.list_feature_values("Subject", f.feature_name))) == len(
                list(bag.list_feature_values("Subject", f.feature_name))
            )
        for f in ml_instance.find_features("Image"):
            assert len(list(ml_instance.list_feature_values("Image", f.feature_name))) == len(
                list(bag.list_feature_values("Image", f.feature_name))
            )

    def test_delete_feature(self, test_ml_catalog):
        pass
