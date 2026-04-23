"""Shared fixtures for feature tests.

Provides fixtures that combine an online DerivaML instance with a downloaded
+ materialized bag so that online/offline symmetry can be tested.
"""
from __future__ import annotations

import dataclasses

import pytest

from deriva_ml import BuiltinTypes, ColumnDefinition
from deriva_ml.dataset.aux_classes import VersionPart
from deriva_ml.execution import ExecutionConfiguration


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
