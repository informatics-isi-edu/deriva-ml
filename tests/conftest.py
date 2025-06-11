"""
Pytest configuration and shared fixtures.
"""

import logging
import os
from pathlib import Path

import pytest

from deriva_ml import DerivaML
from deriva_ml.demo_catalog import create_demo_catalog, reset_demo_catalog


class MLCatalog:
    def __init__(self, hostname, tmpdir):
        test_catalog = create_demo_catalog(hostname, populate=False, on_exit_delete=False)
        self.deriva_ml = DerivaML(
            hostname=hostname,
            catalog_id=test_catalog.catalog_id,
            domain_schema="deriva-test",
            logging_level=logging.WARN,
            working_dir=tmpdir,
            use_minid=False,
        )
        print("🚀 Created demo catalog")

    def cleanup(self):
        self.deriva_ml.catalog.delete_ermrest_catalog(really=True)


@pytest.fixture(scope="session")
def test_host():
    """Get the test host from the environment or use default."""
    return os.environ.get("DERIVA_HOST", "localhost")


@pytest.fixture
def test_catalog_id():
    """Get the test catalog ID from environment or use default."""
    return os.environ.get("DERIVA_CATALOG_ID", "eye-ai")


@pytest.fixture(scope="session")
def shared_tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("deriva_ml")


@pytest.fixture(scope="session")
def ml_catalog(test_host, shared_tmp_path):
    resource = MLCatalog(test_host, shared_tmp_path)
    yield resource
    resource.cleanup()


@pytest.fixture(scope="class")
def test_ml_catalog(ml_catalog):
    """Create a demo ML instance for testing."""
    hostname = ml_catalog.deriva_ml.catalog.deriva_server.server
    yield DerivaML(hostname, ml_catalog.deriva_ml.catalog_id, use_minid=False)
    reset_demo_catalog(ml_catalog.deriva_ml)


@pytest.fixture
def test_files_dir():
    """Get the path to the test files directory."""
    return Path(__file__).parent / "test_files"


@pytest.fixture(autouse=True)
def log_start():
    print("\n--- Starting test ---")
    yield
    print("\n--- Ending test ---")
