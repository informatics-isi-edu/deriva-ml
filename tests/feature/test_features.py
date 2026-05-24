"""
Tests for feature functionality.
"""

import pytest
from pydantic import ValidationError

from deriva_ml import (
    BuiltinTypes,
    ColumnDefinition,
    DerivaML,
    DerivaMLException,
)
from deriva_ml import MLVocab as vc
from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.execution import ExecutionConfiguration
from deriva_ml.feature import FeatureRecord


class TestFeatureRecord:
    """Test cases for the FeatureRecord base class."""

    def test_feature_record_creation(self, mocker):
        """Test basic FeatureRecord creation."""
        # Create a mock feature
        mock_feature = mocker.Mock()
        mock_feature.feature_columns = {mocker.Mock(name="value"), mocker.Mock(name="confidence")}
        mock_feature.asset_columns = {mocker.Mock(name="image_file")}
        mock_feature.term_columns = {mocker.Mock(name="category")}
        mock_feature.value_columns = {mocker.Mock(name="score")}

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

    def test_feature_record_column_methods(self, mocker):
        """Test the column access methods of FeatureRecord."""
        # Create mock columns
        value_col = mocker.Mock(name="value")
        confidence_col = mocker.Mock(name="confidence")
        asset_col = mocker.Mock(name="image_file")
        term_col = mocker.Mock(name="category")
        value_only_col = mocker.Mock(name="score")

        # Create a mock feature
        mock_feature = mocker.Mock()
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

    def test_feature_init_classifies_columns_by_name_not_just_count(self, dataset_test, tmp_path):
        """Inspect ``Feature.__init__``'s FK-classification by column name.

        The existing ``test_create_feature`` only checks bucket
        sizes. Bucket *contents* are uncovered: a regression that
        swapped two columns between buckets (e.g. accidentally
        put the structural ``Execution`` FK in ``value_columns``)
        would pass that test while breaking selectors and asset
        upload.

        This test pins the classification by name:

        1. Each known feature column lands in the named bucket.
        2. The three structural FKs (``Execution``, ``Feature_Name``,
           the target-table FK) are NOT in any of
           ``asset_columns`` / ``term_columns`` / ``value_columns``.
           See ``Feature.__init__``'s ``assoc_fkeys`` subtraction at
           ``feature.py:546-558``.

        Uses the demo catalog's three features (Health on Subject,
        BoundingBox on Image, Quality on Image). Coverage gap from
        audit P1 F-6.
        """
        ml_instance = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )

        # --- Subject.Health: one term column, one value column,
        # zero asset columns. The term column is the controlled-
        # vocab FK ``SubjectHealth``; the value column is the
        # int2 ``Scale`` metadata column (see demo_catalog.py's
        # ``create_feature("Subject", "Health", terms=...,
        # metadata=[ColumnDefinition("Scale", ...)])`` call).
        health = next(
            f for f in ml_instance.model.find_features("Subject")
            if f.feature_name == "Health"
        )
        term_names = {c.name for c in health.term_columns}
        value_names = {c.name for c in health.value_columns}
        asset_names = {c.name for c in health.asset_columns}
        assert term_names == {"SubjectHealth"}, (
            f"Subject.Health.term_columns: expected {{'SubjectHealth'}}, "
            f"got {term_names}."
        )
        assert value_names == {"Scale"}, (
            f"Subject.Health.value_columns: expected {{'Scale'}}, "
            f"got {value_names}."
        )
        assert asset_names == set()

        # --- Image.BoundingBox: one asset column (the box-mask file),
        # zero terms, zero plain values.
        bbox = next(
            f for f in ml_instance.model.find_features("Image")
            if f.feature_name == "BoundingBox"
        )
        asset_names = {c.name for c in bbox.asset_columns}
        assert len(asset_names) == 1, (
            f"Image.BoundingBox.asset_columns: expected exactly one column; "
            f"got {asset_names}."
        )
        assert {c.name for c in bbox.term_columns} == set()
        assert {c.name for c in bbox.value_columns} == set()

        # --- Image.Quality: one term column (the ImageQuality vocab FK),
        # zero asset/value.
        quality = next(
            f for f in ml_instance.model.find_features("Image")
            if f.feature_name == "Quality"
        )
        term_names = {c.name for c in quality.term_columns}
        assert term_names == {"ImageQuality"}, (
            f"Image.Quality.term_columns: expected {{'ImageQuality'}}, "
            f"got {term_names}."
        )
        assert {c.name for c in quality.asset_columns} == set()
        assert {c.name for c in quality.value_columns} == set()

        # --- Structural FKs must not appear in ANY bucket.
        # ``Feature.__init__`` subtracts them via ``assoc_fkeys``;
        # this pin guards against a regression that drops the
        # subtraction.
        STRUCTURAL = {"Execution", "Feature_Name"}
        for feat, target_fk in [(health, "Subject"), (bbox, "Image"), (quality, "Image")]:
            all_buckets = (
                {c.name for c in feat.asset_columns}
                | {c.name for c in feat.term_columns}
                | {c.name for c in feat.value_columns}
            )
            leaked = (STRUCTURAL | {target_fk}) & all_buckets
            assert not leaked, (
                f"{feat.target_table.name}.{feat.feature_name}: structural "
                f"FK columns leaked into a classification bucket: {leaked}. "
                f"Feature.__init__'s ``assoc_fkeys`` subtraction must "
                f"remove every column whose FK constraint is part of the "
                f"association table's primary key."
            )

    def test_create_feature_with_non_asset_table_raises(self, test_ml):
        """create_feature surfaces the bad table name in the error.

        Closes audit Phase 3 feature/ §3.5 — pairs with the §1.4
        error-message fix in fix(feature): correct duplicated/wrong
        validation message in create_feature. A user passing a
        regular domain table as `assets=[...]` should see the
        offending table name in the exception message.
        """
        # Subject is a domain table, not an asset table.
        with pytest.raises(DerivaMLException, match="asset table"):
            test_ml.create_feature(
                target_table="Subject",
                feature_name="BadAssetFeature",
                assets=["Subject"],  # ← not an asset table
                terms=[],
                metadata=[ColumnDefinition(name="value", type=BuiltinTypes.text)],
            )

    def test_create_feature_with_non_vocabulary_table_raises(self, test_ml):
        """create_feature names the right parameter when `terms` is wrong.

        Closes audit Phase 3 feature/ §3.5 — pre-fix, the error
        said "asset table" even though the failure was on
        `terms`. Post-fix, the message correctly identifies the
        vocabulary-table check.
        """
        # Subject is a domain table, not a vocabulary table.
        with pytest.raises(DerivaMLException, match="vocabulary table"):
            test_ml.create_feature(
                target_table="Image",
                feature_name="BadVocabFeature",
                assets=[],
                terms=["Subject"],  # ← not a vocabulary
                metadata=[ColumnDefinition(name="value", type=BuiltinTypes.text)],
            )

    def test_lookup_feature(self, dataset_test, tmp_path):
        ml_instance = DerivaML(
            dataset_test.catalog.hostname, dataset_test.catalog.catalog_id, working_dir=tmp_path, use_minid=False
        )

        assert "Health" == ml_instance.lookup_feature("Subject", "Health").feature_name
        with pytest.raises(DerivaMLException):
            ml_instance.lookup_feature("Foobar", "Health")

        with pytest.raises(DerivaMLException):
            ml_instance.lookup_feature("Subject", "SubjectHealth1")

    def test_find_features(self, dataset_test, tmp_path):
        """Test find_features with and without table argument."""
        ml_instance = DerivaML(
            dataset_test.catalog.hostname, dataset_test.catalog.catalog_id, working_dir=tmp_path, use_minid=False
        )

        # Test finding features for a specific table
        subject_features = ml_instance.find_features("Subject")
        assert "Health" in [f.feature_name for f in subject_features]
        # All returned features should be for Subject table
        assert all(f.target_table.name == "Subject" for f in subject_features)

        image_features = ml_instance.find_features("Image")
        assert "BoundingBox" in [f.feature_name for f in image_features]
        assert "Quality" in [f.feature_name for f in image_features]
        # All returned features should be for Image table
        assert all(f.target_table.name == "Image" for f in image_features)

        # Test finding all features (no table argument)
        all_features = list(ml_instance.find_features())
        all_feature_names = [f.feature_name for f in all_features]
        # Should include features from both tables
        assert "Health" in all_feature_names
        assert "BoundingBox" in all_feature_names
        assert "Quality" in all_feature_names

        # No duplicates: each feature appears exactly once. Without
        # this assertion the no-table-arg branch of find_features
        # rediscovers each association table once per FK target
        # (target table, Execution, Feature_Name vocab, and every
        # term-vocab the feature references), producing N copies of
        # each Feature object. See
        # docs/bugs/2026-05-19-find-features-duplicates.md.
        feature_table_qnames = [f"{f.feature_table.schema.name}.{f.feature_table.name}" for f in all_features]
        assert len(feature_table_qnames) == len(set(feature_table_qnames)), (
            f"find_features() returned duplicates: {feature_table_qnames}"
        )

        # Total equals the sum of per-target-table feature lists.
        # Stricter than the historical `>=` assertion that accommodated
        # the duplicates bug.
        assert len(all_features) == len(list(subject_features)) + len(list(image_features))

        # Every returned Feature has a target_table that is the actual
        # feature target -- not the Execution table, and not a
        # vocabulary table. Validates the dedup tie-breaker.
        for f in all_features:
            assert f.target_table.name != "Execution", f"Feature {f.feature_name} has Execution as target_table"
            assert not ml_instance.model.is_vocabulary(f.target_table), (
                f"Feature {f.feature_name} has vocabulary {f.target_table.name} as target_table"
            )

    def test_feature_record(self, dataset_test, tmp_path):
        ml_instance = DerivaML(
            dataset_test.catalog.hostname, dataset_test.catalog.catalog_id, working_dir=tmp_path, use_minid=False
        )
        SubjectHealthFeature = ml_instance.feature_record_class("Subject", "Health")
        print(SubjectHealthFeature.model_fields.keys())

        print(SubjectHealthFeature.feature_columns())

        with pytest.raises(ValidationError):
            SubjectHealthFeature(Subject="SubjectRID", Health="Good", Scale=23, Foo="Bar")
        print(SubjectHealthFeature.value_columns())
        print(SubjectHealthFeature.term_columns())
        print(SubjectHealthFeature.asset_columns())
        print(SubjectHealthFeature.feature_columns())

    def test_add_feature(self, dataset_test, tmp_path):
        ml_instance = DerivaML(
            dataset_test.catalog.hostname, dataset_test.catalog.catalog_id, working_dir=tmp_path, use_minid=False
        )
        subject_table = ml_instance.pathBuilder().schemas[ml_instance.default_schema].tables["Subject"]
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
            print(SubjectHealthFeature.feature_columns())
            exe.add_features([SubjectHealthFeature(Subject=subject_rids[0], SubjectHealth="Sick", Scale=23)])

        feature_execution.commit_output_assets()
        # Filter to records from THIS execution — the demo catalog populates
        # Health feature values for every Subject during ensure_features, so the
        # unfiltered count will be (# demo subjects) + 1.
        features = [
            f for f in ml_instance.feature_values("Subject", "Health") if f.Execution == feature_execution.execution_rid
        ]
        assert len(features) == 1

        _ImageBoundingboxFeature = ml_instance.feature_record_class("Image", "BoundingBox")
        _ImageQualtiyFeature = ml_instance.feature_record_class("Image", "Quality")

    def test_add_asset_feature(self, dataset_test, tmp_path):
        """Test adding feature values that reference asset files."""
        ml_instance = DerivaML(
            dataset_test.catalog.hostname, dataset_test.catalog.catalog_id, working_dir=tmp_path, use_minid=False
        )

        # Get image RIDs to annotate
        image_table = ml_instance.pathBuilder().schemas[ml_instance.default_schema].tables["Image"]
        image_rids = [img["RID"] for img in image_table.path.entities().fetch()]
        assert len(image_rids) > 0

        # Verify the BoundingBox feature exists and has an asset column
        ImageBoundingBoxFeature = ml_instance.feature_record_class("Image", "BoundingBox")
        assert len(ImageBoundingBoxFeature.asset_columns()) == 1
        assert len(ImageBoundingBoxFeature.term_columns()) == 0

        # Create a workflow and execution for adding asset features
        ml_instance.add_term(
            vc.workflow_type, "Test Asset Feature Workflow", description="Workflow for testing asset features"
        )
        asset_workflow = ml_instance.create_workflow(
            name="Test Asset Feature Workflow",
            workflow_type="Test Asset Feature Workflow",
            description="A test workflow for asset features",
        )
        asset_execution = ml_instance.create_execution(
            ExecutionConfiguration(description="Asset Feature Execution", workflow=asset_workflow)
        )

        with asset_execution.execute() as exe:
            # Create asset files and feature records
            feature_records = []
            for i, image_rid in enumerate(image_rids[:3]):
                # Create a bounding box asset file
                bb_file = exe.asset_file_path("BoundingBox", f"bbox_{i}.txt")
                with bb_file.open("w") as f:
                    f.write(f"bounding box data for image {i}")

                feature_records.append(
                    ImageBoundingBoxFeature(
                        Image=image_rid,
                        BoundingBox=bb_file,
                    )
                )

            exe.add_features(feature_records)

        # Upload outputs (assets + feature values)
        asset_execution.commit_output_assets()

        # Verify the feature values were created — filter to THIS execution,
        # since the demo catalog also populates BoundingBox feature values
        # during ensure_features.
        features = [
            f
            for f in ml_instance.feature_values("Image", "BoundingBox")
            if f.Execution == asset_execution.execution_rid
        ]
        assert len(features) == 3

        # Verify each feature value has a valid asset RID (not a file path)
        for f in features:
            dumped = f.model_dump()
            assert dumped["BoundingBox"] is not None
            # The BoundingBox column should contain an RID, not a file path
            assert "/" not in str(dumped["BoundingBox"]), "BoundingBox should be an asset RID, not a file path"
            assert dumped["Execution"] is not None
            assert dumped["Image"] in image_rids[:3]

    def test_download_feature(self, dataset_test, tmp_path):
        ml_instance = DerivaML(
            dataset_test.catalog.hostname, dataset_test.catalog.catalog_id, working_dir=tmp_path, use_minid=False
        )
        dataset_rid = dataset_test.dataset_description.dataset.dataset_rid

        bag = ml_instance.download_dataset_bag(
            DatasetSpec(rid=dataset_rid, version=dataset_test.dataset_description.dataset.current_version)
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

        # feature_values returns FeatureRecord instances. The feature-row's own
        # RID is not exposed on FeatureRecord (stripped by Feature's skip_columns),
        # so we compare by the identifying tuple (Subject, Execution) instead —
        # which uniquely identifies a feature value per (target, execution) pair.
        catalog_feature_values = {
            (f.Subject, f.Execution)
            for f in ml_instance.feature_values("Subject", "Health")
            if f.Subject in subject_rids
        }

        bag_feature_values = {(f.Subject, f.Execution) for f in bag.feature_values("Subject", "Health")}
        assert catalog_feature_values == bag_feature_values

        for t in ["Subject", "Image"]:
            for f in ml_instance.model.find_features(t):
                catalog_features = [
                    {"Execution": e.model_dump()["Execution"], "Feature_Name": e.Feature_Name, t: e.model_dump()[t]}
                    for e in ml_instance.feature_values(t, f.feature_name)
                    if e.model_dump()[t] in (subject_rids | image_rids)
                ]
                catalog_features.sort(key=lambda x: x[t])
                bag_features = [
                    {"Execution": e.model_dump()["Execution"], "Feature_Name": e.Feature_Name, t: e.model_dump()[t]}
                    for e in bag.feature_values(t, f.feature_name)
                ]
                bag_features.sort(key=lambda x: x[t])
                assert catalog_features == bag_features

    def test_bag_find_features_no_arg_dedups(self, dataset_test, tmp_path):
        """``DatasetBag.find_features()`` (no arg) yields each feature exactly once.

        Regression test for the bag-side counterpart of the
        find_features-duplicates bug already fixed on the live
        catalog. Earlier versions of ``DatasetBag.find_features``
        iterated every schema-then-table and called the model's
        per-table ``find_features(t)`` once per table; each
        association table got revisited once per FK target it
        carried (target table + ``Execution`` + every vocab it
        references), yielding 3+ copies of every Feature object.

        The fix delegates the no-arg case directly to the model's
        no-arg ``find_features`` (which contains the canonical
        dedup walk). This test pins that behaviour by checking
        that:

        1. The bag's no-arg call returns the same set of features
           as the live catalog's no-arg call (modulo iteration
           order).
        2. The bag's result contains no duplicates by
           ``(feature_table.schema.name, feature_table.name)``
           qualified name.
        3. The bag's result equals the union of its per-table
           calls — i.e., the no-arg branch isn't missing features
           that the per-table branch finds.
        """
        ml_instance = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )
        dataset_rid = dataset_test.dataset_description.dataset.dataset_rid
        bag = ml_instance.download_dataset_bag(
            DatasetSpec(
                rid=dataset_rid,
                version=dataset_test.dataset_description.dataset.current_version,
            )
        )

        bag_features = list(bag.find_features())
        catalog_features = list(ml_instance.find_features())

        # (1) Same set as the live catalog.
        bag_keys = {
            (f.feature_table.schema.name, f.feature_table.name) for f in bag_features
        }
        catalog_keys = {
            (f.feature_table.schema.name, f.feature_table.name) for f in catalog_features
        }
        assert bag_keys == catalog_keys, (
            f"bag.find_features() and ml.find_features() should agree on "
            f"the feature set. bag only: {bag_keys - catalog_keys}; "
            f"catalog only: {catalog_keys - bag_keys}."
        )

        # (2) No duplicates -- the regression we're guarding against.
        bag_qnames = [
            f"{f.feature_table.schema.name}.{f.feature_table.name}" for f in bag_features
        ]
        assert len(bag_qnames) == len(set(bag_qnames)), (
            f"bag.find_features() emitted duplicates: {bag_qnames}. "
            f"The bag's no-arg branch must delegate to the model's "
            f"deduped walk, not iterate per-table itself."
        )

        # (3) No-arg result covers the per-table results.
        per_table_keys: set[tuple[str, str]] = set()
        for tname in ("Subject", "Image"):
            for f in bag.find_features(tname):
                per_table_keys.add((f.feature_table.schema.name, f.feature_table.name))
        assert per_table_keys.issubset(bag_keys), (
            f"bag.find_features() missed features that per-table calls "
            f"surfaced: {per_table_keys - bag_keys}."
        )

    def test_delete_feature_success(self, test_ml):
        """delete_feature returns True after a successful drop.

        Closes audit Phase 3 feature/ §1.5 — the placeholder
        `def test_delete_feature: pass` left both real branches
        of delete_feature() unexercised. Verify the success
        path: create a feature, delete it, observe True + the
        feature is gone from find_features.
        """
        self.create_features(test_ml)
        # Sanity: feature exists before delete
        assert "Health" in [f.feature_name for f in test_ml.model.find_features("Subject")]

        result = test_ml.delete_feature("Subject", "Health")
        assert result is True

        # And it's actually gone
        assert "Health" not in [f.feature_name for f in test_ml.model.find_features("Subject")]

    def test_delete_feature_missing_returns_false(self, test_ml):
        """delete_feature returns False when the feature doesn't exist.

        Closes audit Phase 3 feature/ §1.5 — the StopIteration
        branch in delete_feature() (line 276) was previously
        untested.
        """
        # The Subject table exists (test_ml fixture provides it)
        # but a feature named 'NonexistentFeature' has never been
        # created. delete_feature should return False, not raise.
        result = test_ml.delete_feature("Subject", "NonexistentFeature")
        assert result is False

    def create_features(self, ml_instance: DerivaML):
        ml_instance.create_vocabulary("SubjectHealth", "A vocab")
        ml_instance.create_vocabulary("SubjectHealth1", "A vocab")
        for t in ["SubjectHealth", "SubjectHealth1"]:
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


def test_list_workflow_executions_returns_matching_rids(test_ml) -> None:
    """list_workflow_executions returns all execution RIDs that ran a given workflow."""
    from deriva_ml import MLVocab as vc
    from deriva_ml.execution import ExecutionConfiguration
    from deriva_ml.execution.workflow import Workflow

    test_ml.add_term(vc.workflow_type, "Test Workflow", description="Workflow type for testing")
    wf_rid = test_ml._add_workflow(
        Workflow(
            name="S2_test_wf",
            url="https://example.com/s2_test",
            workflow_type="Test Workflow",
            description="S2 list_workflow_executions coverage",
            checksum="a" * 64,
        )
    )
    # Two executions against the same workflow
    wf = test_ml.lookup_workflow(wf_rid)
    cfg = ExecutionConfiguration(description="exec 1", workflow=wf)
    with test_ml.create_execution(cfg) as exe1:
        pass
    with test_ml.create_execution(cfg) as exe2:
        pass
    rids = test_ml.list_workflow_executions(wf_rid)
    assert exe1.execution_rid in rids
    assert exe2.execution_rid in rids
    # Unique entries
    assert len(rids) == len(set(rids))


def test_list_workflow_executions_by_workflow_type_name(test_ml) -> None:
    """list_workflow_executions accepts a Workflow_Type name (not just an RID)."""
    from deriva_ml import MLVocab as vc
    from deriva_ml.execution import ExecutionConfiguration
    from deriva_ml.execution.workflow import Workflow

    # Use a test-unique Workflow_Type name to avoid cross-test pollution from
    # any other test that also creates workflows tagged "Test Workflow".
    wf_type_name = "S2_ListWF_TypeNameTest"
    test_ml.add_term(vc.workflow_type, wf_type_name, description="Workflow type for type-name test")
    wf_rid = test_ml._add_workflow(
        Workflow(
            name="S2_type_wf",
            url="https://example.com/s2_type",
            workflow_type=wf_type_name,
            description="S2 workflow type name coverage",
            checksum="b" * 64,
        )
    )
    wf = test_ml.lookup_workflow(wf_rid)
    cfg = ExecutionConfiguration(description="exec", workflow=wf)
    with test_ml.create_execution(cfg) as exe:
        pass
    rids = test_ml.list_workflow_executions(wf_type_name)
    assert exe.execution_rid in rids
    # Exact set — this test owns all executions of this unique type
    assert set(rids) == {exe.execution_rid}


def test_list_workflow_executions_unknown_raises(test_ml) -> None:
    """Unknown workflow → DerivaMLException."""
    from deriva_ml.core.exceptions import DerivaMLException

    with pytest.raises(DerivaMLException):
        test_ml.list_workflow_executions("nonexistent-workflow-xyz")
