"""
Tests for feature functionality.
"""

import pytest

from deriva_ml import (
    BuiltinTypes,
    ColumnDefinition,
    DatasetSpec,
    DerivaML,
    DerivaMLException,
    ExecutionConfiguration,
)
from deriva_ml import MLVocab as vc


class TestFeatures:
    def test_create_feature(self, test_ml_catalog):
        ml_instance = test_ml_catalog

        self.create_features(ml_instance)
        assert "Health" in [f.feature_name for f in ml_instance.model.find_features("Subject")]
        assert "BoundingBox" in [f.feature_name for f in ml_instance.model.find_features("Image")]
        assert "Quality" in [f.feature_name for f in ml_instance.model.find_features("Image")]

    def test_lookup_feature(self, test_ml_demo_catalog):
        ml_instance = test_ml_demo_catalog

        assert "Health" == ml_instance.lookup_feature("Subject", "Health").feature_name
        with pytest.raises(DerivaMLException):
            ml_instance.lookup_feature("Foobar", "Health")

        with pytest.raises(DerivaMLException):
            ml_instance.lookup_feature("Subject", "SubjectHealth1")

    def test_add_feature(self, test_ml_catalog_populated):
        ml_instance = test_ml_catalog_populated
        subject_table = ml_instance.pathBuilder.schemas[ml_instance.domain_schema].tables["Subject"]
        subject_rids = [s["RID"] for s in subject_table.path.entities().fetch()]
        self.create_features(ml_instance)

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

        feature_execution.upload_execution_outputs()
        features = list(ml_instance.list_feature_values("Subject", "Health"))
        assert len(features) == 1

        ImageBoundingboxFeature = ml_instance.feature_record_class("Image", "BoundingBox")
        ImageQualtiyFeature = ml_instance.feature_record_class("Image", "Quality")

    def test_download_feature(self, test_ml_catalog_dataset):
        ml_instance = test_ml_catalog_dataset.ml_instance
        dataset_rid = test_ml_catalog_dataset.dataset_description.rid

        bag = ml_instance.download_dataset_bag(
            DatasetSpec(rid=dataset_rid, version=ml_instance.dataset_version(dataset_rid))
        )

        # Get the members of the dataset....
        dataset_rids = {
            t: [m["RID"] for m in members]
            for t, members in ml_instance.list_dataset_members(dataset_rid, recurse=True).items()
        }

        bag_rids = {t: [m["RID"] for m in members] for t, members in bag.list_dataset_members(recurse=True).items()}

        print("dataset_rids", dataset_rids["Subject"])
        print("bag_rids", bag_rids)
        # Download a bag with feature values in it

        # Check to see if feature value is in the bag.
        s_features = [f"{f.target_table.name}:{f.feature_name}" for f in ml_instance.model.find_features("Subject")]
        s_features_bag = [f"{f.target_table.name}:{f.feature_name}" for f in bag.find_features("Subject")]
        assert s_features == s_features_bag

        print("subject rids", dataset_rids["Subject"])
        print([m for m in ml_instance.list_feature_values("Subject", "Health")])
        catalog_feature_values = {
            f["RID"]
            for f in ml_instance.list_feature_values("Subject", "Health")
            if f["Subject"] in dataset_rids["Subject"]
        }
        bag_feature_values = {f["RID"] for f in bag.list_feature_values("Subject", "Health")}
        print("bag_features", [f for f in bag.list_feature_values("Subject", "Health")])
        assert catalog_feature_values == bag_feature_values

        for f in ml_instance.model.find_features("Subject"):
            catalog_subject_features = [
                {"Execution": e["Execution"], "Feature_Name": e["Feature_Name"], "Subject": e["Subject"]}
                for e in ml_instance.list_feature_values("Subject", f.feature_name)
                if e["Subject"] in dataset_rids["Subject"]
            ]
            bag_subject_features = [
                {"Execution": e["Execution"], "Feature_Name": e["Feature_Name"], "Subject": e["Subject"]}
                for e in bag.list_feature_values("Subject", f.feature_name)
            ]
            print(len(catalog_subject_features), len(bag_subject_features))
            assert catalog_subject_features == bag_subject_features

        for f in ml_instance.model.find_features("Image"):
            assert len(list(ml_instance.list_feature_values("Image", f.feature_name))) == len(
                list(bag.list_feature_values("Image", f.feature_name))
            )

    def test_delete_feature(self, test_ml_catalog):
        pass

    def create_features(self, ml_instance: DerivaML):
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
        ml_instance.create_feature("Image", "BoundingBox", assets=[box_asset])
        ml_instance.create_feature("Image", "Quality", terms=["ImageQuality"])
