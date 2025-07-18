"""
Pytest configuration and shared fixtures.
"""

import os

import pytest
from test_utils import MLCatalog, MLDatasetCatalog

from deriva_ml import DerivaML
from deriva_ml.demo_catalog import (
    create_demo_datasets,
    create_demo_features,
    populate_demo_catalog,
)


@pytest.fixture(scope="session")
def catalog_host():
    """Get the test host from the environment or use default."""
    return os.environ.get("DERIVA_HOST", "localhost")


@pytest.fixture(scope="session")
def deriva_catalog(catalog_host):
    """Create a demo ML instance for testing with schema, but no data..""" ""
    resource = MLCatalog(catalog_host)
    yield resource
    resource.cleanup()


@pytest.fixture(scope="session")
def populated_catalog(deriva_catalog, tmp_path):
    """Create a demo ML instance for testing with populated domain schema.   Resets after each test."""
    print("Setting up populated catalog for testing... ", end="")

    ml_instance = DerivaML(deriva_catalog.hostname, deriva_catalog.catalog_id, use_minid=False, working_dir=tmp_path)
    populate_demo_catalog(ml_instance)
    return ml_instance


@pytest.fixture(scope="session")
def catalog_with_datasets(deriva_catalog):
    """Create a demo ML instance for testing with populated domain schema.   Resets after each test."""
    print("Setting up catalog with datasets for testing... ", end="")
    return MLDatasetCatalog(deriva_catalog)


@pytest.fixture(scope="function")
def test_ml(deriva_catalog, tmp_path):
    deriva_catalog.reset_demo_catalog()
    """Create a demo ML instance for testing.   Resets after each class.""" ""
    yield DerivaML(deriva_catalog.hostname, deriva_catalog.catalog_id, use_minid=False, working_dir=tmp_path)
    print("Resetting catalog... ", end="")
    deriva_catalog.reset_demo_catalog()


@pytest.fixture(scope="function")
def dataset_test(catalog_with_datasets):
    catalog_with_datasets.reset_catalog()
    return catalog_with_datasets


@pytest.fixture(scope="function")
def test_ml_demo_catalog(ml_catalog, tmp_path):
    # reset_demo_catalog(ml_catalog.catalog)
    ml_instance = DerivaML(ml_catalog.hostname, ml_catalog.catalog_id, use_minid=False, working_dir=tmp_path)
    populate_demo_catalog(ml_instance)
    create_demo_features(ml_instance)
    create_demo_datasets(ml_instance)
    return ml_instance


@pytest.fixture(autouse=True)
def log_start():
    print("\n--- Starting test ---")
    yield
    print("\n--- Ending test ---")
