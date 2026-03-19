"""Tests for fetch_table_features, selector support, and select_by_workflow.

Tests the new feature query functionality added to FeatureMixin:
- fetch_table_features(): grouped feature value retrieval
- selector parameter on list_feature_values() and fetch_table_features()
- FeatureRecord.select_newest: RCT-based selection
- FeatureMixin.select_by_workflow(): workflow-aware selection
- RCT field on FeatureRecord base class
"""

import time
from random import choice, randint

import pytest

from deriva_ml import DerivaML, DerivaMLException
from deriva_ml import MLVocab as vc
from deriva_ml.dataset.aux_classes import DatasetSpec, VersionPart
from deriva_ml.execution import ExecutionConfiguration
from deriva_ml.feature import FeatureRecord


def _add_feature_values(ml: DerivaML, rounds: int = 2) -> None:
    """Add feature values to the catalog via multiple executions.

    The dataset_test fixture creates feature definitions but does not upload
    feature values (missing upload_execution_outputs call). This helper adds
    values for Subject/Health and Image/Quality features.

    Uses separate executions per round so that each (Subject, Execution) pair
    is unique (satisfying the table's unique key constraint). Multiple rounds
    give subjects multiple feature values for selector testing.

    Args:
        ml: Connected DerivaML instance.
        rounds: Number of rounds of feature values to add per subject.
    """
    SubjectHealthFeature = ml.feature_record_class("Subject", "Health")
    ImageQualityFeature = ml.feature_record_class("Image", "Quality")

    subject_rids = [s["RID"] for s in ml.domain_path().tables["Subject"].entities().fetch()]
    image_rids = [i["RID"] for i in ml.domain_path().tables["Image"].entities().fetch()]

    for i in range(rounds):
        workflow = ml.create_workflow(
            name=f"Test Feature Population {i}",
            workflow_type="Test Workflow",
        )
        execution = ml.create_execution(
            ExecutionConfiguration(
                description=f"Populate feature values round {i}",
                workflow=workflow,
            )
        )

        with execution.execute() as exe:
            for subject_rid in subject_rids:
                exe.add_features(
                    [
                        SubjectHealthFeature(
                            Subject=subject_rid,
                            SubjectHealth=choice(["Well", "Sick"]),
                            Scale=randint(1, 10),
                        )
                    ]
                )

            # Only add quality values in first round (single value per image is fine)
            if i == 0:
                for image_rid in image_rids:
                    exe.add_features(
                        [
                            ImageQualityFeature(
                                Image=image_rid,
                                ImageQuality=choice(["Good", "Bad"]),
                            )
                        ]
                    )

        execution.upload_execution_outputs()
        # Small delay to ensure distinct RCT timestamps between rounds
        if i < rounds - 1:
            time.sleep(0.1)


class TestFeatureRecordRCT:
    """Test the RCT field on FeatureRecord."""

    def test_rct_field_exists(self):
        """FeatureRecord has an RCT field defaulting to None."""

        class TestFeature(FeatureRecord):
            value: str

        record = TestFeature(Feature_Name="test", value="x")
        assert record.RCT is None

    def test_rct_field_accepts_timestamp(self):
        """RCT can be set to an ISO 8601 timestamp string."""

        class TestFeature(FeatureRecord):
            value: str

        record = TestFeature(
            Feature_Name="test",
            value="x",
            RCT="2024-06-15T10:30:00.000000+00:00",
        )
        assert record.RCT == "2024-06-15T10:30:00.000000+00:00"

    def test_rct_in_model_dump(self):
        """RCT is present in model_dump output."""

        class TestFeature(FeatureRecord):
            value: str

        record = TestFeature(Feature_Name="test", value="x", RCT="2024-01-01T00:00:00+00:00")
        dumped = record.model_dump()
        assert "RCT" in dumped

    def test_rct_can_be_excluded_from_model_dump(self):
        """RCT can be explicitly excluded from model_dump for inserts."""

        class TestFeature(FeatureRecord):
            value: str

        record = TestFeature(Feature_Name="test", value="x", RCT="2024-01-01T00:00:00+00:00")
        dumped = record.model_dump(exclude={"RCT"})
        assert "RCT" not in dumped


class TestSelectNewest:
    """Test FeatureRecord.select_newest static method."""

    def test_select_newest_picks_latest_rct(self):
        """select_newest returns the record with the most recent RCT."""

        class TestFeature(FeatureRecord):
            value: str

        records = [
            TestFeature(Feature_Name="f", value="old", RCT="2024-01-01T00:00:00+00:00"),
            TestFeature(Feature_Name="f", value="newest", RCT="2024-06-15T10:30:00+00:00"),
            TestFeature(Feature_Name="f", value="middle", RCT="2024-03-01T00:00:00+00:00"),
        ]
        selected = FeatureRecord.select_newest(records)
        assert selected.value == "newest"

    def test_select_newest_handles_none_rct(self):
        """Records with None RCT are treated as older than any timestamped record."""

        class TestFeature(FeatureRecord):
            value: str

        records = [
            TestFeature(Feature_Name="f", value="no_rct", RCT=None),
            TestFeature(Feature_Name="f", value="has_rct", RCT="2024-01-01T00:00:00+00:00"),
        ]
        selected = FeatureRecord.select_newest(records)
        assert selected.value == "has_rct"

    def test_select_newest_single_record(self):
        """select_newest works with a single record."""

        class TestFeature(FeatureRecord):
            value: str

        records = [TestFeature(Feature_Name="f", value="only", RCT="2024-01-01T00:00:00+00:00")]
        selected = FeatureRecord.select_newest(records)
        assert selected.value == "only"

    def test_select_newest_all_none_rct(self):
        """select_newest returns some record when all have None RCT."""

        class TestFeature(FeatureRecord):
            value: str

        records = [
            TestFeature(Feature_Name="f", value="a", RCT=None),
            TestFeature(Feature_Name="f", value="b", RCT=None),
        ]
        selected = FeatureRecord.select_newest(records)
        assert selected in records


class TestFetchTableFeatures:
    """Test fetch_table_features on a live catalog."""

    def test_fetch_all_features(self, dataset_test, tmp_path):
        """fetch_table_features returns all features for a table."""
        ml = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )
        _add_feature_values(ml)

        result = ml.fetch_table_features("Image")
        assert isinstance(result, dict)
        assert "BoundingBox" in result
        assert "Quality" in result
        assert len(result["Quality"]) > 0

        for feature_name, records in result.items():
            assert isinstance(records, list)
            for record in records:
                assert isinstance(record, FeatureRecord)
                assert record.Feature_Name == feature_name

    def test_fetch_single_feature(self, dataset_test, tmp_path):
        """fetch_table_features with feature_name filters to one feature."""
        ml = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )
        _add_feature_values(ml)

        result = ml.fetch_table_features("Image", feature_name="Quality")
        assert list(result.keys()) == ["Quality"]
        assert len(result["Quality"]) > 0

    def test_fetch_nonexistent_feature_raises(self, dataset_test, tmp_path):
        """fetch_table_features raises DerivaMLException for unknown feature."""
        ml = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )

        with pytest.raises(DerivaMLException):
            ml.fetch_table_features("Image", feature_name="NonexistentFeature")

    def test_rct_populated_in_records(self, dataset_test, tmp_path):
        """Feature records returned by fetch_table_features have RCT populated."""
        ml = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )
        _add_feature_values(ml)

        result = ml.fetch_table_features("Image", feature_name="Quality")
        assert len(result["Quality"]) > 0
        for record in result["Quality"]:
            assert record.RCT is not None, "RCT should be populated from catalog query"

    def test_fetch_with_selector_newest(self, dataset_test, tmp_path):
        """fetch_table_features with select_newest deduplicates by target RID."""
        ml = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )
        _add_feature_values(ml, rounds=2)

        all_values = ml.fetch_table_features("Subject", feature_name="Health")
        newest_values = ml.fetch_table_features(
            "Subject",
            feature_name="Health",
            selector=FeatureRecord.select_newest,
        )

        newest_records = newest_values["Health"]
        target_rids = [r.model_dump()["Subject"] for r in newest_records]
        assert len(target_rids) == len(set(target_rids)), "selector should deduplicate by target RID"
        # 2 rounds means 2x records; selector should reduce
        assert len(newest_records) < len(all_values["Health"])


class TestListFeatureValuesWithSelector:
    """Test the selector parameter on list_feature_values."""

    def test_list_feature_values_without_selector(self, dataset_test, tmp_path):
        """list_feature_values without selector returns all values."""
        ml = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )
        _add_feature_values(ml)

        values = list(ml.list_feature_values("Subject", "Health"))
        assert len(values) > 0
        for v in values:
            assert isinstance(v, FeatureRecord)

    def test_list_feature_values_with_selector(self, dataset_test, tmp_path):
        """list_feature_values with selector deduplicates results."""
        ml = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )
        _add_feature_values(ml, rounds=2)

        all_values = list(ml.list_feature_values("Subject", "Health"))
        newest_values = list(ml.list_feature_values("Subject", "Health", selector=FeatureRecord.select_newest))

        subject_rids = [v.model_dump()["Subject"] for v in newest_values]
        assert len(subject_rids) == len(set(subject_rids))
        assert len(newest_values) < len(all_values)

    def test_list_feature_values_rct_populated(self, dataset_test, tmp_path):
        """list_feature_values returns records with RCT populated."""
        ml = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )
        _add_feature_values(ml)

        values = list(ml.list_feature_values("Image", "Quality"))
        assert len(values) > 0
        for v in values:
            assert v.RCT is not None


class TestSelectByWorkflow:
    """Test FeatureMixin.select_by_workflow."""

    def _setup_with_workflows(self, ml):
        """Add feature values via two distinct workflow types.

        Returns (workflow_a_rid, workflow_b_rid, subject_rids).
        Workflows need unique URLs to avoid deduplication in add_workflow().
        """
        ml.add_term(vc.workflow_type, "Workflow_A", description="First workflow type")
        ml.add_term(vc.workflow_type, "Workflow_B", description="Second workflow type")

        workflow_a = ml.create_workflow(
            name="Workflow A", workflow_type="Workflow_A", description="First test workflow"
        )
        workflow_a.url = "https://test.example.com/workflow_a"
        workflow_a.checksum = "aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000"
        workflow_b = ml.create_workflow(
            name="Workflow B", workflow_type="Workflow_B", description="Second test workflow"
        )
        workflow_b.url = "https://test.example.com/workflow_b"
        workflow_b.checksum = "bbbb0000bbbb0000bbbb0000bbbb0000bbbb0000"

        subject_table = ml.pathBuilder().schemas[ml.default_schema].tables["Subject"]
        subject_rids = [s["RID"] for s in subject_table.path.entities().fetch()]
        assert len(subject_rids) >= 2
        test_rids = subject_rids[:2]

        SubjectHealthFeature = ml.feature_record_class("Subject", "Health")

        # Add features via workflow A
        exec_a = ml.create_execution(ExecutionConfiguration(description="Workflow A execution", workflow=workflow_a))
        with exec_a.execute() as exe:
            for rid in test_rids:
                exe.add_features([SubjectHealthFeature(Subject=rid, SubjectHealth="Well", Scale=1)])
        exec_a.upload_execution_outputs()

        # Add features via workflow B (sleep to ensure distinct RCT)
        time.sleep(0.1)
        exec_b = ml.create_execution(ExecutionConfiguration(description="Workflow B execution", workflow=workflow_b))
        with exec_b.execute() as exe:
            for rid in test_rids:
                exe.add_features([SubjectHealthFeature(Subject=rid, SubjectHealth="Sick", Scale=5)])
        exec_b.upload_execution_outputs()

        return exec_a.workflow_rid, exec_b.workflow_rid, test_rids

    def test_select_by_workflow_rid(self, dataset_test, tmp_path):
        """select_by_workflow filters by workflow RID."""
        ml = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )

        wf_a_rid, wf_b_rid, subject_rids = self._setup_with_workflows(ml)

        all_values = list(ml.list_feature_values("Subject", "Health"))
        target_values = [v for v in all_values if v.model_dump()["Subject"] == subject_rids[0]]
        assert len(target_values) >= 2

        selected = ml.select_by_workflow(target_values, wf_a_rid)
        assert isinstance(selected, FeatureRecord)

    def test_select_by_workflow_type_name(self, dataset_test, tmp_path):
        """select_by_workflow filters by Workflow_Type name."""
        ml = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )

        wf_a_rid, wf_b_rid, subject_rids = self._setup_with_workflows(ml)

        all_values = list(ml.list_feature_values("Subject", "Health"))
        target_values = [v for v in all_values if v.model_dump()["Subject"] == subject_rids[0]]

        selected = ml.select_by_workflow(target_values, "Workflow_B")
        assert isinstance(selected, FeatureRecord)

    def test_select_by_workflow_no_match_raises(self, dataset_test, tmp_path):
        """select_by_workflow raises DerivaMLException when no records match."""
        ml = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )

        # Need some feature values to test against
        wf_a_rid, _, subject_rids = self._setup_with_workflows(ml)

        ml.add_term(vc.workflow_type, "Unused_Workflow_Type", description="Never used")

        all_values = list(ml.list_feature_values("Subject", "Health"))
        assert len(all_values) > 0

        with pytest.raises(DerivaMLException):
            ml.select_by_workflow(all_values, "Unused_Workflow_Type")

    def test_select_by_invalid_workflow_raises(self, dataset_test, tmp_path):
        """select_by_workflow raises when workflow identifier doesn't exist."""
        ml = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )

        wf_a_rid, _, subject_rids = self._setup_with_workflows(ml)

        all_values = list(ml.list_feature_values("Subject", "Health"))
        assert len(all_values) > 0

        with pytest.raises(DerivaMLException):
            ml.select_by_workflow(all_values, "Completely_Nonexistent_Type")


class TestFetchTableFeaturesBag:
    """Test fetch_table_features on DatasetBag (offline SQLite)."""

    def _get_bag_with_features(self, dataset_test, ml):
        """Add feature values, increment version, then download bag."""
        _add_feature_values(ml, rounds=2)
        dataset_rid = dataset_test.dataset_description.dataset.dataset_rid
        ds = ml.lookup_dataset(dataset_rid)
        ds.increment_dataset_version(VersionPart.minor, description="Added test feature values")
        ds = ml.lookup_dataset(dataset_rid)  # Refresh to get new version
        return ml.download_dataset_bag(DatasetSpec(rid=dataset_rid, version=str(ds.current_version)))

    def test_bag_fetch_table_features(self, dataset_test, tmp_path):
        """DatasetBag.fetch_table_features returns grouped feature values."""
        ml = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )
        bag = self._get_bag_with_features(dataset_test, ml)

        result = bag.fetch_table_features("Image")
        assert isinstance(result, dict)
        for feature_name in result:
            assert isinstance(result[feature_name], list)
            for record in result[feature_name]:
                assert isinstance(record, FeatureRecord)
                assert record.Feature_Name == feature_name

    def test_bag_fetch_with_selector(self, dataset_test, tmp_path):
        """DatasetBag.fetch_table_features supports selector parameter."""
        ml = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )
        bag = self._get_bag_with_features(dataset_test, ml)

        all_result = bag.fetch_table_features("Subject", feature_name="Health")
        newest_result = bag.fetch_table_features(
            "Subject",
            feature_name="Health",
            selector=FeatureRecord.select_newest,
        )

        if "Health" in newest_result and newest_result["Health"]:
            newest_rids = [r.model_dump()["Subject"] for r in newest_result["Health"]]
            assert len(newest_rids) == len(set(newest_rids))
            assert len(newest_result["Health"]) <= len(all_result.get("Health", []))

    def test_bag_list_feature_values_with_selector(self, dataset_test, tmp_path):
        """DatasetBag.list_feature_values supports selector parameter."""
        ml = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )
        bag = self._get_bag_with_features(dataset_test, ml)

        all_values = list(bag.list_feature_values("Subject", "Health"))
        newest_values = list(bag.list_feature_values("Subject", "Health", selector=FeatureRecord.select_newest))

        if newest_values:
            subject_rids = [v.model_dump()["Subject"] for v in newest_values]
            assert len(subject_rids) == len(set(subject_rids))
        assert len(newest_values) <= len(all_values)

    def test_bag_rct_populated(self, dataset_test, tmp_path):
        """Feature records from DatasetBag have RCT populated."""
        ml = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )
        bag = self._get_bag_with_features(dataset_test, ml)

        values = list(bag.list_feature_values("Subject", "Health"))
        assert len(values) > 0
        for v in values:
            assert v.RCT is not None, "RCT should be populated from bag SQLite data"


class TestCustomSelector:
    """Test that custom selectors work with fetch_table_features."""

    def test_custom_selector_function(self, dataset_test, tmp_path):
        """A custom selector function can be passed to fetch_table_features."""
        ml = DerivaML(
            dataset_test.catalog.hostname,
            dataset_test.catalog.catalog_id,
            working_dir=tmp_path,
            use_minid=False,
        )
        _add_feature_values(ml, rounds=2)

        def select_lowest_scale(records):
            return min(records, key=lambda r: getattr(r, "Scale", 0) or 0)

        result = ml.fetch_table_features(
            "Subject",
            feature_name="Health",
            selector=select_lowest_scale,
        )

        if "Health" in result:
            subject_rids = [r.model_dump()["Subject"] for r in result["Health"]]
            assert len(subject_rids) == len(set(subject_rids))


class TestSelectFirst:
    """Test FeatureRecord.select_first — picks earliest RCT."""

    def test_select_first_picks_earliest_rct(self):
        """select_first returns the record with the earliest RCT."""

        class TestFeature(FeatureRecord):
            value: str

        records = [
            TestFeature(Feature_Name="f", value="newest", RCT="2024-06-15T10:30:00+00:00"),
            TestFeature(Feature_Name="f", value="oldest", RCT="2024-01-01T00:00:00+00:00"),
            TestFeature(Feature_Name="f", value="middle", RCT="2024-03-01T00:00:00+00:00"),
        ]
        selected = FeatureRecord.select_first(records)
        assert selected.value == "oldest"

    def test_select_first_handles_none_rct(self):
        """Records with None RCT sort as oldest (empty string)."""

        class TestFeature(FeatureRecord):
            value: str

        records = [
            TestFeature(Feature_Name="f", value="no_rct", RCT=None),
            TestFeature(Feature_Name="f", value="has_rct", RCT="2024-01-01T00:00:00+00:00"),
        ]
        selected = FeatureRecord.select_first(records)
        assert selected.value == "no_rct"

    def test_select_first_single_record(self):
        """select_first works with a single record."""

        class TestFeature(FeatureRecord):
            value: str

        records = [TestFeature(Feature_Name="f", value="only", RCT="2024-01-01T00:00:00+00:00")]
        selected = FeatureRecord.select_first(records)
        assert selected.value == "only"

    def test_select_first_all_none_rct(self):
        """select_first returns some record when all have None RCT."""

        class TestFeature(FeatureRecord):
            value: str

        records = [
            TestFeature(Feature_Name="f", value="a", RCT=None),
            TestFeature(Feature_Name="f", value="b", RCT=None),
        ]
        selected = FeatureRecord.select_first(records)
        assert selected in records


class TestSelectLatest:
    """Test FeatureRecord.select_latest — alias for select_newest."""

    def test_select_latest_is_equivalent_to_newest(self):
        """select_latest returns the same result as select_newest."""

        class TestFeature(FeatureRecord):
            value: str

        records = [
            TestFeature(Feature_Name="f", value="old", RCT="2024-01-01T00:00:00+00:00"),
            TestFeature(Feature_Name="f", value="newest", RCT="2024-06-15T10:30:00+00:00"),
            TestFeature(Feature_Name="f", value="middle", RCT="2024-03-01T00:00:00+00:00"),
        ]
        result_latest = FeatureRecord.select_latest(records)
        result_newest = FeatureRecord.select_newest(records)
        assert result_latest.value == "newest"
        assert result_latest == result_newest


class TestSelectMajorityVote:
    """Test FeatureRecord.select_majority_vote — picks most common value."""

    def test_majority_vote_picks_most_common(self):
        """Majority vote returns a record with the most frequent value."""

        class TestFeature(FeatureRecord):
            Diagnosis_Type: str

        records = [
            TestFeature(Feature_Name="f", Diagnosis_Type="Normal", RCT="2024-01-01T00:00:00+00:00"),
            TestFeature(Feature_Name="f", Diagnosis_Type="Abnormal", RCT="2024-01-02T00:00:00+00:00"),
            TestFeature(Feature_Name="f", Diagnosis_Type="Normal", RCT="2024-01-03T00:00:00+00:00"),
        ]
        selector = FeatureRecord.select_majority_vote("Diagnosis_Type")
        result = selector(records)
        assert result.Diagnosis_Type == "Normal"

    def test_majority_vote_tie_breaks_by_newest(self):
        """Ties are broken by most recent RCT."""

        class TestFeature(FeatureRecord):
            Diagnosis_Type: str

        records = [
            TestFeature(Feature_Name="f", Diagnosis_Type="Normal", RCT="2024-01-01T00:00:00+00:00"),
            TestFeature(Feature_Name="f", Diagnosis_Type="Abnormal", RCT="2024-01-03T00:00:00+00:00"),
        ]
        selector = FeatureRecord.select_majority_vote("Diagnosis_Type")
        result = selector(records)
        # Tie (1 each) — break by newest RCT
        assert result.RCT == "2024-01-03T00:00:00+00:00"

    def test_majority_vote_auto_detect_single_term_column(self):
        """Auto-detect column when feature metadata has one term column."""
        from unittest.mock import MagicMock

        mock_feature = MagicMock()
        mock_col = MagicMock()
        mock_col.name = "Diagnosis_Type"
        mock_feature.term_columns = [mock_col]

        class TestFeature(FeatureRecord):
            Diagnosis_Type: str
            feature = mock_feature

        records = [
            TestFeature(Feature_Name="f", Diagnosis_Type="Normal", RCT="2024-01-01T00:00:00+00:00"),
            TestFeature(Feature_Name="f", Diagnosis_Type="Normal", RCT="2024-01-02T00:00:00+00:00"),
            TestFeature(Feature_Name="f", Diagnosis_Type="Abnormal", RCT="2024-01-03T00:00:00+00:00"),
        ]
        selector = TestFeature.select_majority_vote()
        result = selector(records)
        assert result.Diagnosis_Type == "Normal"

    def test_majority_vote_raises_without_column_or_metadata(self):
        """Raises when column is None and no feature metadata available."""
        from deriva_ml.core.exceptions import DerivaMLException

        class TestFeature(FeatureRecord):
            Diagnosis_Type: str

        records = [
            TestFeature(Feature_Name="f", Diagnosis_Type="Normal", RCT="2024-01-01T00:00:00+00:00"),
        ]
        selector = FeatureRecord.select_majority_vote()
        with pytest.raises(DerivaMLException, match="requires a column name"):
            selector(records)
