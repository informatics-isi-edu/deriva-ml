"""
Pytest configuration and shared fixtures.
"""

import os

import pytest
from deriva.core.datapath import DataPathException

from deriva_ml import DerivaML
from deriva_ml.demo_catalog import (
    create_demo_datasets,
    create_demo_features,
    populate_demo_catalog,
)
from deriva_ml.execution import ExecutionConfiguration
from deriva_ml.model.database import DatabaseModel, DatabaseModelMeta

from .test_utils import MLCatalog, MLDatasetCatalog, create_jupyter_kernel, destroy_jupyter_kernel


@pytest.fixture(scope="session")
def catalog_host():
    """Get the test host from the environment or use default."""
    return os.environ.get("DERIVA_HOST", "localhost")


@pytest.fixture(scope="function")
def deriva_catalog(catalog_host):
    """Create a demo ML instance for testing with schema, but no data..""" ""
    resource = MLCatalog(catalog_host)
    yield resource
    resource.cleanup()


@pytest.fixture(scope="function")
def populated_catalog(deriva_catalog, tmp_path):
    """Create a demo ML instance for testing with populated domain schema.   Resets after each test."""
    print("Setting up populated catalog for testing... ", end="")

    ml_instance = DerivaML(deriva_catalog.hostname, deriva_catalog.catalog_id, use_minid=False, working_dir=tmp_path)
    populate_workflow = ml_instance.create_workflow(name="Demo Creation", workflow_type="Demo Catalog Creation")
    execution = ml_instance.create_execution(workflow=populate_workflow, configuration=ExecutionConfiguration())
    with execution.execute() as exe:
        populate_demo_catalog(exe)
    return ml_instance


@pytest.fixture(scope="function")
def catalog_with_datasets(deriva_catalog):
    """Create a demo ML instance for testing with populated domain schema.   Resets after each test."""
    print("Setting up catalog with datasets for testing... ", end="")
    dataset_catalog = MLDatasetCatalog(deriva_catalog)
    yield dataset_catalog
    # Clean up the temporary directory when the fixture is done
    dataset_catalog.cleanup()


@pytest.fixture(scope="function")
def test_ml(deriva_catalog, tmp_path):
    deriva_catalog.reset_demo_catalog()
    pb = deriva_catalog.catalog.getPathBuilder()
    domain_path = pb.schemas[deriva_catalog.domain_schema]
    for t in [
        "TestTableExecution",
    ]:
        try:
            domain_path.tables[t].path.delete()
        except DataPathException:
            pass
        except Exception:
                pass
    """Create a demo ML instance for testing.   Resets after each class.""" ""
    yield DerivaML(deriva_catalog.hostname, deriva_catalog.catalog_id, use_minid=False, working_dir=tmp_path)
    print("Resetting catalog... ", end="")
    deriva_catalog.reset_demo_catalog()


@pytest.fixture(scope="function")
def dataset_test(catalog_with_datasets):
    # catalog_with_datasets already provides a fresh catalog for each test
    # No need to reset - that would create new datasets with different RIDs
    return catalog_with_datasets


@pytest.fixture(scope="function")
def notebook_test(deriva_catalog, tmp_path):
    deriva_catalog.reset_demo_catalog()
    create_jupyter_kernel("test_kernel", tmp_path)
    yield DerivaML(deriva_catalog.hostname, deriva_catalog.catalog_id, use_minid=False, working_dir=tmp_path)
    print("Resetting catalog... ", end="")
    deriva_catalog.reset_demo_catalog()
    destroy_jupyter_kernel("test_kernel")


@pytest.fixture(scope="function")
def test_ml_demo_catalog(ml_catalog, tmp_path):
    # reset_demo_catalog(ml_catalog.catalog)
    ml_instance = DerivaML(ml_catalog.hostname, ml_catalog.catalog_id, use_minid=False, working_dir=tmp_path)
    populate_workflow = ml_instance.create_workflow(name="Demo Creation", workflow_type="Demo Catalog Creation")
    execution = ml_instance.create_execution(workflow=populate_workflow, configuration=ExecutionConfiguration())
    with execution.execute() as exe:
        populate_demo_catalog(exe)
        create_demo_features(exe)
        create_demo_datasets(exe)
    return ml_instance


@pytest.fixture(autouse=True)
def log_start():
    print("\n--- Starting test ---")
    yield
    print("\n--- Ending test ---")

@pytest.fixture(autouse=True)
def clear_database_model_caches():
    """Fixture to ensure clean DatabaseModel state between tests.

    This is critical for preventing state leakage that can cause infinite recursion
    or other nondeterministic behavior when tests share cached DatabaseModel instances.
    """
    # Dispose all existing DatabaseModel instances before test starts
    for model in list(DatabaseModelMeta._paths_loaded.values()):
        try:
            model.dispose()
        except Exception:
            pass  # Ignore errors during cleanup

    # Clear all caches before test
    DatabaseModelMeta._paths_loaded.clear()
    DatabaseModel._rid_map.clear()

    yield

    # Dispose all DatabaseModel instances to clean up SQLAlchemy state after test
    for model in list(DatabaseModelMeta._paths_loaded.values()):
        try:
            model.dispose()
        except Exception:
            pass  # Ignore errors during cleanup

    # Clear caches after test to ensure no state leaks to next test
    DatabaseModelMeta._paths_loaded.clear()
    DatabaseModel._rid_map.clear()
