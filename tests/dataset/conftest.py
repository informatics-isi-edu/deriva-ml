"""Shared fixtures for dataset tests.

Provides fixtures that download materialized bags for offline testing,
including bags that contain feature data.
"""
from __future__ import annotations

import dataclasses

import pytest

from deriva_ml import BuiltinTypes, ColumnDefinition
from deriva_ml.dataset.aux_classes import VersionPart
from deriva_ml.execution import ExecutionConfiguration


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

    Uses the CatalogManager's WITH_DATASETS state (which already seeds Image/Quality
    feature values via create_demo_features).  Creates a simple dataset covering
    all Images, increments its version, downloads, and materializes.

    Yields:
        MaterializedBagFixture with .ml, .bag, .feature_name, .target_table.
    """
    catalog_manager.reset()
    ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path)

    # The top-level dataset from the demo catalog already contains Image members
    # and Image/Quality feature values.
    dataset = dataset_desc.dataset
    version = dataset.current_version

    bag = dataset.download_dataset_bag(version=version, use_minid=False)

    yield MaterializedBagFixture(
        ml=ml,
        bag=bag,
        feature_name="Quality",
        target_table="Image",
    )
