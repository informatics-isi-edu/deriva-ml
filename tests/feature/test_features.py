"""
Tests for feature functionality.
"""

from unittest.mock import Mock

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
from deriva_ml.feature import FeatureRecord


class TestFeatureRecord:
    """Test cases for the FeatureRecord base class."""

    def test_feature_record_creation(self):
        """Test basic FeatureRecord creation."""
        # Create a mock feature
        mock_feature = Mock()
        mock_feature.feature_columns = {Mock(name="value"), Mock(name="confidence")}
        mock_feature.asset_columns = {Mock(name="image_file")}
        mock_feature.term_columns = {Mock(name="category")}
        mock_feature.value_columns = {Mock(name="score")}

        # Create a test class that inherits from FeatureRecord
        class TestFeature(FeatureRecord):
            value: str
            confidence: float
            image_file: str
            category: str
            score: float

        # Set the feature reference
        TestFeature.feature = mock_feature

        # Test creation
        record = TestFeature(
            Feature_Name="test_feature",
            value="high",
            confidence=0.95,
            image_file="path/to/image.jpg",
            category="good",
            score=0.8,
        )

        assert record.Feature_Name == "test_feature"
        assert record.value == "high"
        assert record.confidence == 0.95
        assert record.image_file == "path/to/image.jpg"
        assert record.category == "good"
        assert record.score == 0.8

    def test_feature_record_column_methods(self):
        """Test the column access methods of FeatureRecord."""
        # Create mock columns
        value_col = Mock(name="value")
        confidence_col = Mock(name="confidence")
        asset_col = Mock(name="image_file")
        term_col = Mock(name="category")
        value_only_col = Mock(name="score")

        # Create a mock feature
        mock_feature = Mock()
        mock_feature.feature_columns = {value_col, confidence_col, asset_col, term_col, value_only_col}
        mock_feature.asset_columns = {asset_col}
        mock_feature.term_columns = {term_col}
        mock_feature.value_columns = {value_col, confidence_col, value_only_col}

        # Create a test class
        class TestFeature(FeatureRecord):
            value: str
            confidence: float
            image_file: str
            category: str
            score: float

        TestFeature.feature = mock_feature

        # Test column access methods
        assert TestFeature.feature_columns() == mock_feature.feature_columns
        assert TestFeature.asset_columns() == mock_feature.asset_columns
        assert TestFeature.term_columns() == mock_feature.term_columns
        assert TestFeature.value_columns() == mock_feature.value_columns

    def test_feature_record_with_execution(self):
        """Test FeatureRecord with execution RID."""

        class TestFeature(FeatureRecord):
            value: str

        record = TestFeature(Feature_Name="test_feature", Execution="1-abc123", value="test_value")

        assert record.Feature_Name == "test_feature"
        assert record.Execution == "1-abc123"
        assert record.value == "test_value"


class TestFeatures:
    def test_create_feature(self, dataset_test, tmp_path):
        ml_instance = DerivaML(
            dataset_test.catalog.hostname, dataset_test.catalog.catalog_id, working_dir=tmp_path, use_minid=False
        )
        assert "Health" in [f.feature_name for f in ml_instance.model.find_features("Subject")]
        assert "BoundingBox" in [f.feature_name for f in ml_instance.model.find_features("Image")]
        assert "Quality" in [f.feature_name for f in ml_instance.model.find_features("Image")]

        subject_health_feature = ml_instance.feature_record_class("Subject", "Health")
        assert len(subject_health_feature.asset_columns()) == 0
        assert len(subject_health_feature.term_columns()) == 1
        assert len(subject_health_feature.feature_columns()) == 2
        assert len(subject_health_feature.value_columns()) == 1

        bounding_box_feature = ml_instance.feature_record_class("Image", "BoundingBox")
        assert len(bounding_box_feature.asset_columns()) == 1
        assert len(bounding_box_feature.term_columns()) == 0
        assert len(bounding_box_feature.feature_columns()) == 1
        assert len(bounding_box_feature.value_columns()) == 0

        image_quality_feature = ml_instance.feature_record_class("Image", "Quality")
        assert len(image_quality_feature.asset_columns()) == 0
        assert len(image_quality_feature.term_columns()) == 1
        assert len(image_quality_feature.feature_columns()) == 1

    def test_lookup_feature(self, dataset_test, tmp_path):
        ml_instance = DerivaML(
            dataset_test.catalog.hostname, dataset_test.catalog.catalog_id, working_dir=tmp_path, use_minid=False
        )

        assert "Health" == ml_instance.lookup_feature("Subject", "Health").feature_name
        with pytest.raises(DerivaMLException):
            ml_instance.lookup_feature("Foobar", "Health")

        with pytest.raises(DerivaMLException):
            ml_instance.lookup_feature("Subject", "SubjectHealth1")

    def test_add_feature(self, dataset_test, tmp_path):
        ml_instance = DerivaML(
            dataset_test.catalog.hostname, dataset_test.catalog.catalog_id, working_dir=tmp_path, use_minid=False
        )
        subject_table = ml_instance.pathBuilder.schemas[ml_instance.domain_schema].tables["Subject"]
        subject_rids = [s["RID"] for s in subject_table.path.entities().fetch()]

        assert "Health" in [f.feature_name for f in ml_instance.model.find_features("Subject")]
        assert "BoundingBox" in [f.feature_name for f in ml_instance.model.find_features("Image")]
        assert "Quality" in [f.feature_name for f in ml_instance.model.find_features("Image")]

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

    def test_download_feature(self, dataset_test, tmp_path):
        ml_instance = DerivaML(
            dataset_test.catalog.hostname, dataset_test.catalog.catalog_id, working_dir=tmp_path, use_minid=False
        )
        dataset_rid = dataset_test.dataset_description.rid

        bag = ml_instance.download_dataset_bag(
            DatasetSpec(rid=dataset_rid, version=ml_instance.dataset_version(dataset_rid))
        )

        # Get the lists of all of the rinds in the datasets....datasets....
        subject_rids = {r["RID"] for r in bag.get_table_as_dict("Subject")}
        image_rids = {r["RID"] for r in bag.get_table_as_dict("Image")}

        # Check to see if the bag has the same features defined as the catalog.
        s_features = [f"{f.target_table.name}:{f.feature_name}" for f in ml_instance.model.find_features("Subject")]
        s_features_bag = [f"{f.target_table.name}:{f.feature_name}" for f in bag.find_features("Subject")]
        assert s_features == s_features_bag

        s_features = [f"{f.target_table.name}:{f.feature_name}" for f in ml_instance.model.find_features("Image")]
        s_features_bag = [f"{f.target_table.name}:{f.feature_name}" for f in bag.find_features("Image")]
        assert s_features == s_features_bag

        catalog_feature_values = {
            f["RID"] for f in ml_instance.list_feature_values("Subject", "Health") if f["Subject"] in subject_rids
        }

        bag_feature_values = {f["RID"] for f in bag.list_feature_values("Subject", "Health")}
        assert catalog_feature_values == bag_feature_values

        for t in ["Subject", "Image"]:
            for f in ml_instance.model.find_features(t):
                catalog_features = [
                    {"Execution": e["Execution"], "Feature_Name": e["Feature_Name"], t: e[t]}
                    for e in ml_instance.list_feature_values(t, f.feature_name)
                    if e[t] in (subject_rids | image_rids)
                ]
                catalog_features.sort(key=lambda x: x[t])
                bag_features = [
                    {"Execution": e["Execution"], "Feature_Name": e["Feature_Name"], t: e[t]}
                    for e in bag.list_feature_values(t, f.feature_name)
                ]
                bag_features.sort(key=lambda x: x[t])
                assert catalog_features == bag_features

    def test_delete_feature(self, test_ml):
        pass

    def create_features(self, ml_instance: DerivaML):
        ml_instance.create_vocabulary("SubjectHealth", "A vocab")
        ml_instance.create_vocabulary("SubjectHealth1", "A vocab")
        for t in ["SubjectHeath", "SubjectHealth1"]:
            ml_instance.add_term(
                t,
                "Sick",
                description="The subject self reports that they are sick",
            )
            ml_instance.add_term(
                t,
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
            terms=["SubjectHealth", "SubjectHealth1"],
            metadata=[ColumnDefinition(name="Scale", type=BuiltinTypes.int2, nullok=True)],
            optional=["Scale"],
        )
        ml_instance.create_feature("Image", "BoundingBox", assets=[box_asset])
        ml_instance.create_feature("Image", "Quality", terms=["ImageQuality"])
