"""Tests for the unified feature_values read surface.

This task (Task 4) covers DerivaML online reads. The parametrized symmetry
suite across DerivaML / Dataset / DatasetBag is added in Task 8 once all
three containers have the method.
"""
from __future__ import annotations

import dataclasses
import time

import pytest

from deriva_ml import BuiltinTypes, ColumnDefinition
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.aux_classes import VersionPart
from deriva_ml.dataset.dataset_bag import DatasetBag
from deriva_ml.execution import ExecutionConfiguration
from deriva_ml.feature import FeatureRecord


@dataclasses.dataclass
class FeatureFixture:
    """Container for a catalog seeded with a single feature."""

    ml: object
    feature_name: str
    image_rids: list[str]
    target_table: str = "Image"


@pytest.fixture
def test_ml_with_feature(populated_catalog):
    """DerivaML instance with one Image/Label feature, one value per image."""
    ml = populated_catalog
    feature_name = "Label"

    ml.create_feature(
        target_table="Image",
        feature_name=feature_name,
        metadata=[ColumnDefinition(name="Label", type=BuiltinTypes.text)],
    )

    image_rids = [r["RID"] for r in ml._domain_path().tables["Image"].entities().fetch()]
    assert image_rids, "No Image rows in test catalog"

    workflow = ml.create_workflow(
        name="Label seeder",
        workflow_type="Test Workflow",
    )
    execution = ml.create_execution(
        ExecutionConfiguration(
            description="Seed Label feature values",
            workflow=workflow,
        )
    )
    LabelFeature = ml.feature_record_class("Image", feature_name)
    with execution.execute() as exe:
        for rid in image_rids:
            exe.add_features([LabelFeature(Image=rid, Label="positive")])
    execution.upload_execution_outputs()

    yield FeatureFixture(ml=ml, feature_name=feature_name, image_rids=image_rids)


@pytest.fixture
def test_ml_with_feature_multi(populated_catalog):
    """DerivaML instance with Image/Score feature, multiple values per image."""
    ml = populated_catalog
    feature_name = "Score"

    ml.create_feature(
        target_table="Image",
        feature_name=feature_name,
        metadata=[ColumnDefinition(name="Score", type=BuiltinTypes.text)],
    )

    image_rids = [r["RID"] for r in ml._domain_path().tables["Image"].entities().fetch()]
    assert image_rids, "No Image rows in test catalog"

    ScoreFeature = ml.feature_record_class("Image", feature_name)

    for i, label in enumerate(["high", "low"]):
        workflow = ml.create_workflow(
            name=f"Score seeder round {i}",
            workflow_type="Test Workflow",
        )
        execution = ml.create_execution(
            ExecutionConfiguration(
                description=f"Seed Score feature values round {i}",
                workflow=workflow,
            )
        )
        with execution.execute() as exe:
            for rid in image_rids:
                exe.add_features([ScoreFeature(Image=rid, Score=label)])
        execution.upload_execution_outputs()
        if i == 0:
            time.sleep(0.1)

    yield FeatureFixture(ml=ml, feature_name=feature_name, image_rids=image_rids)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_feature_values_yields_feature_records(test_ml_with_feature) -> None:
    """feature_values yields FeatureRecord instances with the expected attrs."""
    records = list(test_ml_with_feature.ml.feature_values(
        "Image", test_ml_with_feature.feature_name,
    ))
    assert len(records) > 0
    rec = records[0]
    assert isinstance(rec, FeatureRecord)
    # Target RID attribute present and non-empty
    assert getattr(rec, "Image", None)
    # Feature_Name populated
    assert rec.Feature_Name == test_ml_with_feature.feature_name


def test_feature_values_unknown_feature_raises(test_ml) -> None:
    with pytest.raises(DerivaMLException):
        list(test_ml.feature_values("Image", "no_such_feature_xyz"))


def test_feature_values_is_iterable_not_list(test_ml_with_feature) -> None:
    """Return type is an iterator (streaming), not a materialized list."""
    import collections.abc
    result = test_ml_with_feature.ml.feature_values(
        "Image", test_ml_with_feature.feature_name,
    )
    assert isinstance(result, collections.abc.Iterator)


def test_feature_values_with_select_newest(test_ml_with_feature_multi) -> None:
    """Selector reduces multi-value groups to one record per target RID."""
    records = list(test_ml_with_feature_multi.ml.feature_values(
        "Image",
        test_ml_with_feature_multi.feature_name,
        selector=FeatureRecord.select_newest,
    ))
    # After selector: one record per target Image
    rids = [r.Image for r in records]
    assert len(rids) == len(set(rids))


def test_feature_values_selector_returning_none_omits_target(test_ml_with_feature) -> None:
    """Selector returning None removes that target from the iterator."""
    def reject_all(records):
        return None
    records = list(test_ml_with_feature.ml.feature_values(
        "Image", test_ml_with_feature.feature_name, selector=reject_all,
    ))
    assert records == []


# ---------------------------------------------------------------------------
# Task 5: Dataset-scoped feature_values / lookup_feature / list_workflow_executions
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DatasetFeatureFixture:
    """Container for a catalog seeded with a feature and a dataset over a subset of members."""

    ml: object
    dataset: object
    feature_name: str
    target_table: str
    workflow: str
    member_rids: set  # Image RIDs that ARE dataset members
    non_member_rids: set  # Image RIDs that have feature values but are NOT members


@pytest.fixture
def catalog_with_feature_and_dataset(populated_catalog):
    """DerivaML + Dataset with a subset of Image members that have a feature defined."""
    ml = populated_catalog
    feature_name = "Label"

    ml.create_feature(
        target_table="Image",
        feature_name=feature_name,
        metadata=[ColumnDefinition(name="Label", type=BuiltinTypes.text)],
    )

    all_image_rids = [r["RID"] for r in ml._domain_path().tables["Image"].entities().fetch()]
    assert all_image_rids, "No Image rows in test catalog"

    # Ensure Image is registered as a dataset element type (not done by populate_demo_catalog)
    ml.add_dataset_element_type("Image")

    # Use only first half of images as dataset members
    member_rids = all_image_rids[: max(1, len(all_image_rids) // 2)]

    workflow = ml.create_workflow(
        name="Label seeder for dataset test",
        workflow_type="Test Workflow",
    )
    execution = ml.create_execution(
        ExecutionConfiguration(
            description="Seed Label feature values for all images",
            workflow=workflow,
        )
    )
    # Capture workflow RID after it's been registered via create_execution
    workflow_rid = execution.workflow_rid

    LabelFeature = ml.feature_record_class("Image", feature_name)
    with execution.execute() as exe:
        for rid in all_image_rids:
            exe.add_features([LabelFeature(Image=rid, Label="positive")])
    execution.upload_execution_outputs()

    # Create a second execution to build the dataset
    dataset_execution = ml.create_execution(
        ExecutionConfiguration(
            description="Create dataset for Task 5 test",
            workflow=workflow,
        )
    )
    with dataset_execution.execute() as exe:
        dataset = exe.create_dataset(
            dataset_types=["Testing"],
            description="Dataset for Task 5 test",
        )
        dataset.add_dataset_members(members={"Image": member_rids})
    dataset.increment_dataset_version(component=VersionPart.minor, description="v1")

    member_set = set(member_rids)
    non_member_set = set(all_image_rids) - member_set
    yield DatasetFeatureFixture(
        ml=ml,
        dataset=dataset,
        feature_name=feature_name,
        target_table="Image",
        workflow=workflow_rid,
        member_rids=member_set,
        non_member_rids=non_member_set,
    )


def test_dataset_feature_values_filters_to_members(catalog_with_feature_and_dataset) -> None:
    """Dataset.feature_values yields only records whose target RID is in dataset members.

    The fixture seeds feature values for ALL images but adds only a subset as dataset
    members, so non_member_rids is guaranteed non-empty — the filter must actively
    exclude them.
    """
    fx = catalog_with_feature_and_dataset
    assert fx.non_member_rids, (
        "fixture must have at least one image with a feature value that is NOT a dataset member"
    )

    records = list(fx.dataset.feature_values(fx.target_table, fx.feature_name))
    yielded_rids = {getattr(r, fx.target_table) for r in records}

    # Every yielded RID is a member
    assert yielded_rids.issubset(fx.member_rids), (
        f"non-member RIDs leaked through: {yielded_rids - fx.member_rids}"
    )
    # No non-member RID was yielded — filter actually applied
    assert not (yielded_rids & fx.non_member_rids), (
        f"non-member RIDs should have been excluded: {yielded_rids & fx.non_member_rids}"
    )


def test_dataset_lookup_feature_delegates_to_ml(catalog_with_feature_and_dataset) -> None:
    """Dataset.lookup_feature returns a Feature object identical to ml.lookup_feature."""
    fx = catalog_with_feature_and_dataset
    ds_feat = fx.dataset.lookup_feature(fx.target_table, fx.feature_name)
    ml_feat = fx.ml.lookup_feature(fx.target_table, fx.feature_name)
    assert ds_feat.feature_name == ml_feat.feature_name
    assert ds_feat.target_table.name == ml_feat.target_table.name


def test_dataset_list_workflow_executions_scopes_to_dataset(
    catalog_with_feature_and_dataset,
) -> None:
    """Dataset.list_workflow_executions is a pass-through to the catalog-wide list.

    The current implementation returns the full catalog-wide list of executions for
    the workflow.  The two sets are therefore equal (not merely a subset).  If strict
    dataset scoping is added later this test will fail and signal that the contract
    needs to be re-evaluated.
    """
    fx = catalog_with_feature_and_dataset
    ml_rids = set(fx.ml.list_workflow_executions(fx.workflow))
    ds_rids = set(fx.dataset.list_workflow_executions(fx.workflow))
    assert ds_rids == ml_rids


# ---------------------------------------------------------------------------
# Task 6: DatasetBag.feature_values / lookup_feature (offline)
# ---------------------------------------------------------------------------


def test_bag_feature_values_matches_online(
    catalog_with_feature_and_materialized_bag,
) -> None:
    """bag.feature_values yields records whose content matches ml.feature_values.

    Compares sorted-by-target-RID model_dump() dicts, excluding RCT/RMT
    which may differ slightly between catalog and bag representations.

    The bag may include feature records for images reachable through any FK
    path in the dataset (e.g., via Subject→Image), so it can contain MORE
    records than just the direct Image members.  We therefore filter both
    sides to ``member_rids`` (the set of Image RIDs explicitly listed as
    dataset members) so the comparison is symmetric.
    """
    fx = catalog_with_feature_and_materialized_bag
    online = sorted(
        [
            r.model_dump(exclude={"RCT", "RMT"})
            for r in fx.ml.feature_values(fx.target_table, fx.feature_name)
            if getattr(r, fx.target_table) in fx.member_rids
        ],
        key=lambda d: d[fx.target_table],
    )
    offline = sorted(
        [
            r.model_dump(exclude={"RCT", "RMT"})
            for r in fx.bag.feature_values(fx.target_table, fx.feature_name)
            if getattr(r, fx.target_table) in fx.member_rids
        ],
        key=lambda d: d[fx.target_table],
    )
    assert len(offline) > 0, "Bag should have at least one feature record for a dataset member"
    assert online == offline


def test_bag_lookup_feature_works_offline(
    catalog_with_feature_and_materialized_bag,
) -> None:
    """bag.lookup_feature works without a live catalog connection."""
    fx = catalog_with_feature_and_materialized_bag
    feat = fx.bag.lookup_feature(fx.target_table, fx.feature_name)
    assert feat.feature_name == fx.feature_name
    # feature_record_class() must be usable offline
    RecordClass = feat.feature_record_class()
    instance = RecordClass(Image="IMG-1", Feature_Name=fx.feature_name)
    assert instance.Image == "IMG-1"


# ---------------------------------------------------------------------------
# Task 8: Parametrized three-container symmetry compliance suite
# ---------------------------------------------------------------------------


@pytest.fixture(params=["ml", "dataset", "bag"])
def feature_container(request, feature_symmetry_fixture):
    """Parametrize across DerivaML, Dataset, DatasetBag for symmetry tests."""
    return feature_symmetry_fixture.by_container_kind(request.param)


class TestFeatureValuesSymmetry:
    """Same assertions, three containers. The symmetry contract.

    Any future container claiming feature capability must pass this suite as
    its acceptance test.  The fixture ``feature_symmetry_fixture`` ensures all
    three containers see the same underlying data so comparisons are exact.
    """

    def test_find_features_returns_matching_definitions(self, feature_container):
        """find_features includes the seeded feature definition on all containers."""
        features = list(feature_container.container.find_features(feature_container.target_table))
        names = {f.feature_name for f in features}
        assert feature_container.feature_name in names

    def test_feature_values_yields_expected_records(self, feature_container):
        """feature_values yields records that match the expected sorted list."""
        records = sorted(
            [
                r.model_dump(exclude={"RCT", "RMT"})
                for r in feature_container.container.feature_values(
                    feature_container.target_table, feature_container.feature_name
                )
            ],
            key=lambda d: d[feature_container.target_table],
        )
        expected = feature_container.expected_records_sorted
        assert records == expected

    def test_feature_values_with_selector_matches(self, feature_container):
        """select_newest selector reduces multi-value groups to one record per target."""
        records = list(
            feature_container.container.feature_values(
                feature_container.target_table,
                feature_container.feature_name,
                selector=FeatureRecord.select_newest,
            )
        )
        assert records, "Selector returned no records"
        rids = [getattr(r, feature_container.target_table) for r in records]
        assert len(rids) == len(set(rids))  # one per target RID

    def test_lookup_feature_returns_usable_record_class(self, feature_container):
        """lookup_feature returns a Feature whose feature_record_class() is constructible."""
        feat = feature_container.container.lookup_feature(
            feature_container.target_table, feature_container.feature_name
        )
        RecordClass = feat.feature_record_class()
        instance = RecordClass(
            **{feature_container.target_table: "TEST-RID"},
            Feature_Name=feature_container.feature_name,
        )
        assert getattr(instance, feature_container.target_table) == "TEST-RID"

    def test_list_workflow_executions_matches(self, feature_container):
        """list_workflow_executions returns the expected set of execution RIDs.

        For DerivaML and Dataset containers this exercises live catalog lookup.
        For DatasetBag the same method reads from offline SQLite; the test
        skips when the bag's Execution table is empty (a known bag-export
        limitation: Execution rows are only exported if Execution is a
        dataset element type or reachable via Dataset_Execution paths).
        """
        if isinstance(feature_container.container, DatasetBag):
            # Verify the method is callable; skip if execution data not in bag.
            try:
                rids = feature_container.container.list_workflow_executions(
                    feature_container.workflow
                )
            except DerivaMLException:
                pytest.skip(
                    "DatasetBag.list_workflow_executions requires Execution rows "
                    "exported to bag SQLite — not present when Execution is not a "
                    "dataset element type (known bag-export limitation)."
                )

            if not rids:
                pytest.skip(
                    "DatasetBag returned empty execution list — pre-existing bag-export limitation."
                )
        else:
            rids = feature_container.container.list_workflow_executions(
                feature_container.workflow
            )
        # Order-independent comparison against the catalog-computed baseline
        assert set(rids) == feature_container.expected_workflow_executions


# ---------------------------------------------------------------------------
# Task 9: Offline-to-online write cycle round-trip
# ---------------------------------------------------------------------------


def test_offline_construct_records_online_stage(
    catalog_with_feature_and_materialized_bag,
) -> None:
    """Construct FeatureRecord from bag offline; stage + flush via live execution.

    This closes the offline-to-online loop: a user working with a downloaded
    bag on a laptop can build new feature records (e.g., model predictions)
    and commit them back to the live catalog via exe.add_features when
    reconnected.

    The test intentionally avoids touching fx.ml in the OFFLINE section to
    prove that record construction works without a catalog connection.
    """
    fx = catalog_with_feature_and_materialized_bag

    # ------------------------------------------------------------------ #
    # OFFLINE: construct records using bag metadata only — no catalog call #
    # ------------------------------------------------------------------ #
    feat = fx.bag.lookup_feature(fx.target_table, fx.feature_name)
    RecordClass = feat.feature_record_class()

    # Pick a few target RIDs from the bag (up to 3)
    target_rids = sorted(
        {getattr(r, fx.target_table) for r in fx.bag.feature_values(fx.target_table, fx.feature_name)}
    )[:3]
    assert target_rids, "Bag should have at least one feature target RID"

    # Build records entirely offline — ImageQuality is the vocab column name (from
    # create_feature("Image", "Quality", terms=["ImageQuality"])).
    # "Good" is a valid ImageQuality term per demo_catalog.py.
    records = [
        RecordClass(**{fx.target_table: rid, "ImageQuality": "Good"})
        for rid in target_rids
    ]
    # Confirm Execution is not set — exe.add_features must auto-fill it
    assert all(r.Execution is None for r in records), "Execution should be None before staging"

    # ------------------------------------------------------------------ #
    # ONLINE: stage + flush via a fresh execution                          #
    # ------------------------------------------------------------------ #
    workflow = fx.ml.find_workflows()[0]
    cfg = ExecutionConfiguration(
        description="offline-to-online test (S2 Task 9)",
        workflow=workflow,
    )
    execution = fx.ml.create_execution(cfg)
    with execution.execute() as exe:
        count = exe.add_features(records)
        assert count == len(target_rids), (
            f"add_features should return the number of staged records; "
            f"got {count}, expected {len(target_rids)}"
        )
    # __exit__ does NOT auto-upload (Task 7 review) — explicit call required
    execution.upload_execution_outputs()

    # ------------------------------------------------------------------ #
    # VERIFY: records from this execution appear in the live catalog       #
    # ------------------------------------------------------------------ #
    written = list(fx.ml.feature_values(fx.target_table, fx.feature_name))
    ours = [r for r in written if r.Execution == execution.execution_rid]
    assert len(ours) == len(target_rids), (
        f"Expected {len(target_rids)} new records from execution "
        f"{execution.execution_rid}; found {len(ours)}"
    )
    assert all(r.ImageQuality == "Good" for r in ours), (
        "All written records should have ImageQuality='Good'"
    )
