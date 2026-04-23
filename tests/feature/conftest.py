"""Shared fixtures for feature tests.

Provides fixtures that combine an online DerivaML instance with a downloaded
+ materialized bag so that online/offline symmetry can be tested.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from deriva_ml import BuiltinTypes, ColumnDefinition
from deriva_ml.dataset.aux_classes import VersionPart
from deriva_ml.execution import ExecutionConfiguration


# ---------------------------------------------------------------------------
# Bag fixture for retired-API tests
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MaterializedBagFixture:
    """Container for a downloaded + materialized bag with feature data."""

    ml: object
    bag: object
    feature_name: str
    target_table: str = "Image"


@pytest.fixture
def materialized_bag_with_feature(catalog_manager, tmp_path):
    """Download a materialized bag that includes Image/Quality feature values.

    Mirrors tests/dataset/conftest.py::materialized_bag_with_feature so that
    tests/feature/ tests can also use this fixture.

    Yields:
        MaterializedBagFixture with .ml, .bag, .feature_name, .target_table.
    """
    catalog_manager.reset()
    ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path)

    dataset = dataset_desc.dataset
    version = dataset.current_version

    bag = dataset.download_dataset_bag(version=version, use_minid=False)

    yield MaterializedBagFixture(
        ml=ml,
        bag=bag,
        feature_name="Quality",
        target_table="Image",
    )


@dataclasses.dataclass
class BagFeatureSymmetryFixture:
    """Container for both an online ml and a matching offline bag with feature data."""

    ml: object
    bag: object
    feature_name: str
    target_table: str
    member_rids: set


@pytest.fixture
def catalog_with_feature_and_materialized_bag(catalog_manager, tmp_path):
    """Online DerivaML + matching offline bag for feature symmetry tests.

    Seeded with the demo catalog's WITH_DATASETS state (Image/Quality feature).
    Downloads the top-level dataset bag so that both online and offline reads
    should return equivalent feature records.

    Yields:
        BagFeatureSymmetryFixture with .ml, .bag, .feature_name, .target_table,
        .member_rids.
    """
    catalog_manager.reset()
    ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path)

    dataset = dataset_desc.dataset
    version = dataset.current_version
    bag = dataset.download_dataset_bag(version=version, use_minid=False)

    # Collect all Image RIDs that are members of this dataset (including nested).
    members = bag.list_dataset_members(recurse=True)
    member_rids = {r["RID"] for r in members.get("Image", [])}

    yield BagFeatureSymmetryFixture(
        ml=ml,
        bag=bag,
        feature_name="Quality",
        target_table="Image",
        member_rids=member_rids,
    )


# ---------------------------------------------------------------------------
# Task 8: Three-container symmetry fixtures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SymmetryContainer:
    """One container (DerivaML, Dataset, or DatasetBag) with shared test data."""

    container: Any
    target_table: str
    feature_name: str
    workflow: str
    expected_records_sorted: list[dict]
    expected_workflow_executions: set[str]


@dataclasses.dataclass
class FeatureSymmetryFixture:
    """All three containers plus shared expectations for the symmetry suite."""

    ml: Any
    dataset: Any
    bag: Any
    target_table: str
    feature_name: str
    workflow: str
    expected_records_sorted: list[dict]
    expected_workflow_executions: set[str]

    def by_container_kind(self, kind: str) -> SymmetryContainer:
        """Return a SymmetryContainer for the given container kind.

        Args:
            kind: One of ``"ml"``, ``"dataset"``, ``"bag"``.

        Returns:
            SymmetryContainer with the selected container and shared expectations.
        """
        containers = {"ml": self.ml, "dataset": self.dataset, "bag": self.bag}
        if kind not in containers:
            raise ValueError(f"Unknown container kind {kind!r}; expected one of {sorted(containers)}")
        container = containers[kind]
        return SymmetryContainer(
            container=container,
            target_table=self.target_table,
            feature_name=self.feature_name,
            workflow=self.workflow,
            expected_records_sorted=self.expected_records_sorted,
            expected_workflow_executions=self.expected_workflow_executions,
        )


@pytest.fixture(scope="session")
def feature_symmetry_fixture(catalog_manager, tmp_path_factory):
    """Build a catalog + all-image dataset + materialized bag with Image/Quality feature.

    Uses the demo catalog ``WITH_DATASETS`` state which already seeds the
    ``Image/Quality`` feature with values for every Image row.  A new
    "symmetry test" dataset is created that contains ALL Image RIDs so that
    ``ml``, ``dataset``, and ``bag`` all return the same feature records.

    Yields:
        FeatureSymmetryFixture with ``.ml``, ``.dataset``, ``.bag``,
        ``.target_table``, ``.feature_name``, ``.workflow``,
        ``.expected_records_sorted``, ``.expected_workflow_executions``.
    """
    tmp_path = tmp_path_factory.mktemp("symmetry")
    catalog_manager.reset()
    ml, _dataset_desc = catalog_manager.ensure_datasets(tmp_path)

    feature_name = "Quality"
    target_table = "Image"

    # All Image RIDs (feature values exist for all of them from demo catalog seeding)
    all_image_rids = [
        r["RID"] for r in ml.domain_path().tables[target_table].entities().fetch()
    ]
    assert all_image_rids, "Demo catalog must have Image rows"

    # Find a workflow that has executions for the Quality feature.
    # Note: DerivaML deduplicates workflows by script checksum, so all
    # ensure_populated/ensure_features/ensure_datasets calls may share one
    # workflow row. Use the first workflow with executions as the reference.
    all_workflows = ml.find_workflows()
    assert all_workflows, "At least one workflow must exist after ensure_datasets"

    # DerivaML deduplicates workflows by script checksum — all create_workflow calls
    # from catalog_manager (ensure_populated, ensure_features, ensure_datasets) run
    # from the same module and collapse into a SINGLE Workflow row. The surviving name
    # is whichever registered first ("Test Population"), not "Feature Creation".
    # The correct invariant is: exactly one workflow survives after test catalog setup.
    # This assertion encodes that dedup invariant and will fail loudly if it changes.
    assert len(all_workflows) == 1, (
        f"Expected exactly one workflow after test catalog setup (workflow dedup by "
        f"checksum), but found {len(all_workflows)}: {[w.name for w in all_workflows]}"
    )
    feature_workflow_rid = all_workflows[0].rid
    assert feature_workflow_rid, "Workflow must have a RID"

    # Create a new dataset that contains ALL image RIDs so that the dataset-scoped
    # feature_values returns the same records as the catalog-wide query.
    symmetry_workflow = ml.create_workflow(
        name="Symmetry dataset creator",
        workflow_type="Test Workflow",
    )
    symmetry_execution = ml.create_execution(
        ExecutionConfiguration(
            description="Create all-image dataset for symmetry tests",
            workflow=symmetry_workflow,
        )
    )
    with symmetry_execution.execute() as exe:
        symmetry_dataset = exe.create_dataset(
            dataset_types=["Testing"],
            description="All-image dataset for Task 8 symmetry tests",
        )
        symmetry_dataset.add_dataset_members(members={target_table: all_image_rids})

    symmetry_dataset.increment_dataset_version(
        component=VersionPart.minor,
        description="v1 for symmetry test",
    )

    # Download and materialize a bag from the all-image dataset.
    bag = symmetry_dataset.download_dataset_bag(
        version=symmetry_dataset.current_version, use_minid=False
    )

    # Compute expected records and workflow executions AFTER all catalog setup
    # (including the symmetry execution) so the baseline is complete.
    # The symmetry execution uses the same deduplicated workflow as the feature
    # creation execution, so expected_executions reflects the full catalog state.
    all_quality_records = sorted(
        [
            r.model_dump(exclude={"RCT", "RMT"})
            for r in ml.feature_values(target_table, feature_name)
        ],
        key=lambda d: d[target_table],
    )
    assert all_quality_records, "Must have at least one Quality feature record"

    expected_executions = set(ml.list_workflow_executions(feature_workflow_rid))
    assert expected_executions, "Feature Creation workflow must have at least one execution"

    yield FeatureSymmetryFixture(
        ml=ml,
        dataset=symmetry_dataset,
        bag=bag,
        target_table=target_table,
        feature_name=feature_name,
        workflow=feature_workflow_rid,
        expected_records_sorted=all_quality_records,
        expected_workflow_executions=expected_executions,
    )
