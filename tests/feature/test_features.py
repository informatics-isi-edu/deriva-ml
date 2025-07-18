"""
Tests for feature functionality.
"""

from unittest.mock import MagicMock, Mock

import pytest
from deriva.core.ermrest_model import Column, FindAssociationResult, ForeignKey, Table

from deriva_ml import (
    BuiltinTypes,
    ColumnDefinition,
    DatasetSpec,
    DerivaML,
    DerivaMLException,
    ExecutionConfiguration,
)
from deriva_ml import MLVocab as vc
from deriva_ml.feature import Feature, FeatureRecord


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


class TestFeature:
    """Test cases for the Feature class."""

    def test_feature_initialization(self):
        """Test Feature class initialization."""
        # Create mock objects
        mock_table = Mock(spec=Table)
        mock_table.name = "TestFeature"
        mock_table.columns = {
            "Feature_Name": Mock(default="test_feature"),
            "RID": Mock(),
            "RMB": Mock(),
            "RCB": Mock(),
            "RCT": Mock(),
            "RMT": Mock(),
            "Execution": Mock(),
            "value": Mock(),
            "confidence": Mock(),
        }

        mock_target_table = Mock(spec=Table)
        mock_target_table.name = "TestTable"

        mock_self_fkey = Mock(spec=ForeignKey)
        mock_self_fkey.pk_table = mock_target_table

        mock_other_fkey = Mock(spec=ForeignKey)
        mock_other_fkey.pk_table = Mock(spec=Table)

        mock_atable = Mock(spec=FindAssociationResult)
        mock_atable.table = mock_table
        mock_atable.self_fkey = mock_self_fkey
        mock_atable.other_fkeys = {mock_other_fkey}

        mock_model = Mock()
        mock_model.is_asset.return_value = False
        mock_model.is_vocabulary.return_value = False

        # Create feature
        feature = Feature(mock_atable, mock_model)

        assert feature.feature_table == mock_table
        assert feature.target_table == mock_target_table
        assert feature.feature_name == "test_feature"
        assert feature._model == mock_model

    def test_feature_column_categorization(self):
        """Test that columns are properly categorized."""
        # Create mock tables and columns
        mock_table = Mock(spec=Table)
        mock_table.name = "TestFeature"
        mock_table.columns = {
            "Feature_Name": Mock(default="test_feature"),
            "RID": Mock(),
            "RMB": Mock(),
            "RCB": Mock(),
            "RCT": Mock(),
            "RMT": Mock(),
            "Execution": Mock(),
            "TestTable": Mock(),
            "value": Mock(),
            "confidence": Mock(),
            "asset_ref": Mock(),
            "term_ref": Mock(),
        }

        mock_target_table = Mock(spec=Table)
        mock_target_table.name = "TestTable"

        # Create foreign keys
        asset_fk = Mock(spec=ForeignKey)
        asset_fk.foreign_key_columns = [Mock(name="asset_ref")]
        asset_fk.pk_table = Mock(spec=Table)

        term_fk = Mock(spec=ForeignKey)
        term_fk.foreign_key_columns = [Mock(name="term_ref")]
        term_fk.pk_table = Mock(spec=Table)

        mock_self_fkey = Mock(spec=ForeignKey)
        mock_self_fkey.pk_table = mock_target_table

        mock_atable = Mock(spec=FindAssociationResult)
        mock_atable.table = mock_table
        mock_atable.self_fkey = mock_self_fkey
        mock_atable.other_fkeys = set()

        # Mock the model methods
        mock_model = Mock()
        mock_model.is_asset.side_effect = lambda table: table == asset_fk.pk_table
        mock_model.is_vocabulary.side_effect = lambda table: table == term_fk.pk_table

        # Set up foreign keys
        mock_table.foreign_keys = [asset_fk, term_fk]

        # Create feature
        feature = Feature(mock_atable, mock_model)

        # Test column categorization
        assert "value" in {c.name for c in feature.value_columns}
        assert "confidence" in {c.name for c in feature.value_columns}
        assert "asset_ref" in {c.name for c in feature.asset_columns}
        assert "term_ref" in {c.name for c in feature.term_columns}

        # Test that system columns are excluded
        system_columns = {"RID", "RMB", "RCB", "RCT", "RMT", "Feature_Name", "TestTable", "Execution"}
        for col_name in system_columns:
            assert col_name not in {c.name for c in feature.feature_columns}

    def test_feature_record_class_generation(self):
        """Test dynamic feature record class generation."""
        # Create mock table with different column types
        mock_table = Mock(spec=Table)
        mock_table.name = "TestFeature"

        # Create columns with different types
        text_col = Mock(spec=Column)
        text_col.name = "text_field"
        text_col.type.typename = "text"
        text_col.nullok = False
        text_col.default = None

        int_col = Mock(spec=Column)
        int_col.name = "int_field"
        int_col.type.typename = "int4"
        int_col.nullok = True
        int_col.default = None

        float_col = Mock(spec=Column)
        float_col.name = "float_field"
        float_col.type.typename = "float8"
        float_col.nullok = False
        float_col.default = None

        asset_col = Mock(spec=Column)
        asset_col.name = "asset_field"
        asset_col.type.typename = "text"
        asset_col.nullok = False
        asset_col.default = None

        mock_table.columns = {
            "Feature_Name": Mock(default="test_feature"),
            "RID": Mock(),
            "RMB": Mock(),
            "RCB": Mock(),
            "RCT": Mock(),
            "RMT": Mock(),
            "Execution": Mock(),
            "TestTable": Mock(),
            "text_field": text_col,
            "int_field": int_col,
            "float_field": float_col,
            "asset_field": asset_col,
        }

        mock_target_table = Mock(spec=Table)
        mock_target_table.name = "TestTable"

        mock_self_fkey = Mock(spec=ForeignKey)
        mock_self_fkey.pk_table = mock_target_table

        mock_atable = Mock(spec=FindAssociationResult)
        mock_atable.table = mock_table
        mock_atable.self_fkey = mock_self_fkey
        mock_atable.other_fkeys = set()

        # Mock the model
        mock_model = Mock()
        mock_model.is_asset.return_value = True  # asset_field is an asset
        mock_model.is_vocabulary.return_value = False

        # Set up foreign keys for asset column
        asset_fk = Mock(spec=ForeignKey)
        asset_fk.foreign_key_columns = [asset_col]
        asset_fk.pk_table = Mock(spec=Table)
        mock_table.foreign_keys = [asset_fk]

        # Create feature
        feature = Feature(mock_atable, mock_model)

        # Generate feature record class
        FeatureClass = feature.feature_record_class()

        # Test that the class inherits from FeatureRecord
        assert issubclass(FeatureClass, FeatureRecord)

        # Test that the class has the expected attributes
        assert hasattr(FeatureClass, "text_field")
        assert hasattr(FeatureClass, "int_field")
        assert hasattr(FeatureClass, "float_field")
        assert hasattr(FeatureClass, "asset_field")
        assert hasattr(FeatureClass, "TestTable")
        assert hasattr(FeatureClass, "Feature_Name")

        # Test that the feature reference is set
        assert FeatureClass.feature == feature

        # Test creating an instance with dynamic attributes
        # Use **kwargs to avoid type checker issues with dynamic parameters
        instance_data = {
            "Feature_Name": "test_feature",
            "TestTable": "1-abc123",
            "text_field": "test text",
            "int_field": 42,
            "float_field": 3.14,
            "asset_field": "path/to/file.jpg",
        }
        instance = FeatureClass(**instance_data)

        # Use getattr to avoid type checker issues with dynamic attributes
        assert getattr(instance, "Feature_Name") == "test_feature"
        assert getattr(instance, "TestTable") == "1-abc123"
        assert getattr(instance, "text_field") == "test text"
        assert getattr(instance, "int_field") == 42
        assert getattr(instance, "float_field") == 3.14
        assert getattr(instance, "asset_field") == "path/to/file.jpg"

    def test_feature_type_mapping(self):
        """Test the map_type function for different column types."""
        # Create mock columns with different types
        text_col = Mock(spec=Column)
        text_col.name = "text_field"
        text_col.type.typename = "text"

        int_col = Mock(spec=Column)
        int_col.name = "int_field"
        int_col.type.typename = "int4"

        float_col = Mock(spec=Column)
        float_col.name = "float_field"
        float_col.type.typename = "float8"

        unknown_col = Mock(spec=Column)
        unknown_col.name = "unknown_field"
        unknown_col.type.typename = "unknown_type"

        # Create feature with asset columns
        mock_table = Mock(spec=Table)
        mock_table.name = "TestFeature"
        mock_table.columns = {
            "Feature_Name": Mock(default="test_feature"),
            "RID": Mock(),
            "RMB": Mock(),
            "RCB": Mock(),
            "RCT": Mock(),
            "RMT": Mock(),
            "Execution": Mock(),
            "TestTable": Mock(),
            "text_field": text_col,
            "int_field": int_col,
            "float_field": float_col,
            "unknown_field": unknown_col,
        }

        mock_target_table = Mock(spec=Table)
        mock_target_table.name = "TestTable"

        mock_self_fkey = Mock(spec=ForeignKey)
        mock_self_fkey.pk_table = mock_target_table

        mock_atable = Mock(spec=FindAssociationResult)
        mock_atable.table = mock_table
        mock_atable.self_fkey = mock_self_fkey
        mock_atable.other_fkeys = set()

        mock_model = Mock()
        mock_model.is_asset.return_value = False
        mock_model.is_vocabulary.return_value = False

        mock_table.foreign_keys = []

        feature = Feature(mock_atable, mock_model)

        # Test type mapping through feature record class generation
        FeatureClass = feature.feature_record_class()

        # Check that the types are correctly mapped
        # Note: We can't directly test the map_type function as it's internal,
        # but we can test that the generated class has the right types
        assert FeatureClass.__annotations__["text_field"] == str
        assert FeatureClass.__annotations__["int_field"] == int
        assert FeatureClass.__annotations__["float_field"] == float
        assert FeatureClass.__annotations__["unknown_field"] == str

    def test_feature_repr(self):
        """Test the __repr__ method of Feature."""
        mock_table = Mock(spec=Table)
        mock_table.name = "TestFeature"
        # Create proper column objects
        feature_name_col = Mock(spec=Column)
        feature_name_col.name = "Feature_Name"
        feature_name_col.default = "test_feature"

        # Make columns dict-like but iterable over values
        mock_table.columns = MagicMock()
        mock_table.columns.__iter__.return_value = iter([feature_name_col])
        mock_table.columns.values.return_value = [feature_name_col]
        mock_table.columns.__getitem__.side_effect = lambda key: feature_name_col if key == "Feature_Name" else None

        mock_target_table = Mock(spec=Table)
        mock_target_table.name = "TestTable"

        mock_self_fkey = Mock(spec=ForeignKey)
        mock_self_fkey.pk_table = mock_target_table

        mock_atable = Mock(spec=FindAssociationResult)
        mock_atable.table = mock_table
        mock_atable.self_fkey = mock_self_fkey
        mock_atable.other_fkeys = set()

        mock_model = Mock()
        mock_model.is_asset.return_value = False
        mock_model.is_vocabulary.return_value = False

        mock_table.foreign_keys = []

        feature = Feature(mock_atable, mock_model)

        expected_repr = "Feature(target_table=TestTable, feature_name=test_feature, feature_table=TestFeature)"
        assert repr(feature) == expected_repr


class TestFeatures:
    def test_create_feature(self, dataset_test, tmp_path):
        ml_instance = DerivaML(
            dataset_test.catalog.hostname, dataset_test.catalog.catalog_id, working_dir=tmp_path, use_minid=False
        )
        assert "Health" in [f.feature_name for f in ml_instance.model.find_features("Subject")]
        assert "BoundingBox" in [f.feature_name for f in ml_instance.model.find_features("Image")]
        assert "Quality" in [f.feature_name for f in ml_instance.model.find_features("Image")]

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
