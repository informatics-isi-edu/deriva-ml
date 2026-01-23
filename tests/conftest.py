"""Pytest configuration and shared fixtures.

This module provides a hierarchical fixture system optimized for test performance:

Session-Scoped (created once per test session):
    - catalog_host: Test server hostname
    - catalog_manager: CatalogManager instance that owns the test catalog

Function-Scoped (reset per test):
    - deriva_catalog: Legacy fixture for compatibility (uses catalog_manager)
    - test_ml: Clean DerivaML instance with empty catalog
    - populated_catalog: DerivaML instance with subjects/images
    - catalog_with_datasets: Full dataset hierarchy

The key optimization is that catalog creation (~5-10 seconds) happens once per
session, while table-level reset (~0.1-0.5 seconds) happens between tests.

Configuration:
    DERIVA_HOST: Environment variable for test server (default: localhost)
    DERIVA_TEST_SCOPE: Set to "function" to create fresh catalog per test (slower)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from deriva_ml import DerivaML

from .catalog_manager import CatalogManager, CatalogState
from .test_utils import (
    MLCatalog,
    create_jupyter_kernel,
    destroy_jupyter_kernel,
)

if TYPE_CHECKING:
    from pathlib import Path

    from deriva_ml.demo_catalog import DatasetDescription


# =============================================================================
# Session-Scoped Fixtures (Created Once)
# =============================================================================


@pytest.fixture(scope="session")
def catalog_host() -> str:
    """Get the test host from the environment or use default."""
    return os.environ.get("DERIVA_HOST", "localhost")


@pytest.fixture(scope="session")
def catalog_manager(catalog_host: str) -> CatalogManager:
    """Create a session-scoped catalog manager.

    This is the core fixture that owns the test catalog. The catalog is
    created once at session start and destroyed at session end. Individual
    tests reset the catalog data without recreating the catalog itself.

    Yields:
        CatalogManager instance managing the test catalog.
    """
    manager = CatalogManager(catalog_host)
    print(f"\n🚀 Created session catalog {manager.catalog_id}")
    yield manager
    print(f"\n🗑️ Destroying session catalog {manager.catalog_id}")
    manager.destroy()


# =============================================================================
# Function-Scoped Fixtures (Reset Per Test)
# =============================================================================


@pytest.fixture(scope="function")
def deriva_catalog(catalog_manager: CatalogManager) -> MLCatalog:
    """Create a demo ML instance for testing with schema, but no data.

    This fixture provides backward compatibility with the original MLCatalog
    interface while using the session-scoped catalog under the hood.

    Yields:
        MLCatalog wrapper around the session catalog.
    """
    catalog_manager.reset()

    # Create an MLCatalog-compatible wrapper
    class SessionMLCatalog:
        """Wrapper providing MLCatalog interface over session catalog."""

        def __init__(self, manager: CatalogManager):
            self._manager = manager
            self.catalog = manager.catalog
            self.catalog_id = manager.catalog_id
            self.hostname = manager.hostname
            self.default_schema = manager.default_schema

        def cleanup(self) -> None:
            """No-op: session catalog is cleaned up at session end."""
            pass

        def reset_demo_catalog(self) -> None:
            """Reset catalog to clean state."""
            self._manager.reset()

    wrapper = SessionMLCatalog(catalog_manager)
    yield wrapper
    # Mark state as unknown since test may have modified catalog
    catalog_manager.state = CatalogState.POPULATED


@pytest.fixture(scope="function")
def test_ml(catalog_manager: CatalogManager, tmp_path: Path) -> DerivaML:
    """Create a clean DerivaML instance for testing.

    Resets the catalog before the test to ensure isolation.

    Yields:
        DerivaML instance with empty catalog.
    """
    catalog_manager.reset()
    ml = catalog_manager.get_ml_instance(tmp_path)
    yield ml
    # Mark state as unknown since test may have modified catalog
    # This ensures next test's reset() actually runs
    catalog_manager.state = CatalogState.POPULATED


@pytest.fixture(scope="function")
def populated_catalog(catalog_manager: CatalogManager, tmp_path: Path) -> DerivaML:
    """Create a DerivaML instance with populated data (subjects, images).

    Yields:
        DerivaML instance with populated domain schema.
    """
    catalog_manager.reset()
    ml = catalog_manager.ensure_populated(tmp_path)
    yield ml
    # Note: No after-test reset - next test's before-reset handles isolation


@pytest.fixture(scope="function")
def catalog_with_datasets(
    catalog_manager: CatalogManager, tmp_path: Path
) -> tuple[DerivaML, DatasetDescription]:
    """Create a DerivaML instance with full dataset hierarchy.

    This fixture provides both the ML instance and the DatasetDescription
    that describes the created dataset hierarchy.

    Yields:
        Tuple of (DerivaML instance, DatasetDescription).
    """
    catalog_manager.reset()
    ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path)
    yield ml, dataset_desc
    # Note: No after-test reset - next test's before-reset handles isolation


@pytest.fixture(scope="function")
def dataset_test(
    catalog_manager: CatalogManager, tmp_path: Path
) -> SessionMLDatasetCatalog:
    """Create a dataset test fixture with MLDatasetCatalog-compatible interface.

    This provides backward compatibility with tests expecting the MLDatasetCatalog
    interface (catalog.hostname, catalog.catalog_id, list_datasets, etc).

    Yields:
        SessionMLDatasetCatalog with dataset hierarchy.
    """
    catalog_manager.reset()
    ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path)
    wrapper = SessionMLDatasetCatalog(catalog_manager, ml, dataset_desc)
    yield wrapper
    # Note: No after-test reset - next test's before-reset handles isolation


# =============================================================================
# Legacy Compatibility Fixtures
# =============================================================================


@pytest.fixture(scope="function")
def test_ml_demo_catalog(catalog_manager: CatalogManager, tmp_path: Path) -> DerivaML:
    """Legacy fixture: DerivaML with full demo catalog.

    Creates subjects, images, features, and datasets.
    """
    catalog_manager.reset()
    ml, _ = catalog_manager.ensure_datasets(tmp_path)
    return ml


@pytest.fixture(scope="function")
def notebook_test(catalog_manager: CatalogManager, tmp_path: Path) -> DerivaML:
    """Create a DerivaML instance for notebook testing.

    Creates a Jupyter kernel and cleans up after the test.

    Yields:
        DerivaML instance for notebook testing.
    """
    catalog_manager.reset()
    create_jupyter_kernel("test_kernel", tmp_path)
    ml = catalog_manager.get_ml_instance(tmp_path)
    yield ml
    catalog_manager.reset()
    destroy_jupyter_kernel("test_kernel")


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def log_test_boundaries():
    """Print test start/end markers for debugging."""
    print("\n--- Starting test ---")
    yield
    print("\n--- Ending test ---")


# =============================================================================
# Backward Compatibility: MLDatasetCatalog wrapper
# =============================================================================


class SessionMLDatasetCatalog:
    """Wrapper providing MLDatasetCatalog interface over session catalog.

    This class provides backward compatibility with tests that expect
    the MLDatasetCatalog interface (e.g., list_datasets, collect_rids).

    Attributes:
        catalog: MLCatalog-like object with hostname and catalog_id.
        ml_instance: The DerivaML instance.
        dataset_description: The DatasetDescription for the created hierarchy.
    """

    def __init__(
        self,
        manager: CatalogManager,
        ml_instance: DerivaML,
        dataset_description: DatasetDescription,
    ):
        self._manager = manager
        self.ml_instance = ml_instance
        self.dataset_description = dataset_description
        self.hostname = manager.hostname

        # Create a catalog-like object for backward compatibility
        class CatalogWrapper:
            """Wrapper providing catalog.hostname and catalog.catalog_id."""

            def __init__(inner_self):
                inner_self.hostname = manager.hostname
                inner_self.catalog_id = manager.catalog_id

        self.catalog = CatalogWrapper()

    def cleanup(self) -> None:
        """No-op: session catalog cleanup happens at session end."""
        pass

    def list_datasets(self, dataset_description: DatasetDescription) -> list[DatasetDescription]:
        """Return a list of all datasets in the hierarchy."""
        nested_datasets = [
            ds
            for dset_member in dataset_description.members.get("Dataset", [])
            for ds in self.list_datasets(dset_member)
        ]
        return [dataset_description] + nested_datasets

    def collect_rids(self, description: DatasetDescription) -> set[str]:
        """Collect RIDs for a dataset and its nested datasets."""
        rids = {description.dataset.dataset_rid}
        for member_type, member_descriptor in description.members.items():
            rids |= set(description.member_rids.get(member_type, []))
            if member_type == "Dataset":
                for dataset in member_descriptor:
                    rids |= self.collect_rids(dataset)
        return rids

    def reset_catalog(self) -> None:
        """Reset and recreate datasets."""
        self._manager.reset()
        # Note: This changes dataset RIDs, which may break some tests
        _, self.dataset_description = self._manager.ensure_datasets(
            self.ml_instance.working_dir
        )


@pytest.fixture(scope="function")
def ml_dataset_catalog(
    catalog_manager: CatalogManager, tmp_path: Path
) -> SessionMLDatasetCatalog:
    """Create an MLDatasetCatalog-compatible fixture.

    This provides the same interface as MLDatasetCatalog for tests that
    need methods like list_datasets() and collect_rids().

    Yields:
        SessionMLDatasetCatalog with dataset hierarchy.
    """
    catalog_manager.reset()
    ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path)
    wrapper = SessionMLDatasetCatalog(catalog_manager, ml, dataset_desc)
    yield wrapper
    catalog_manager.reset()
