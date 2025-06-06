"""
Pytest configuration and shared fixtures.
"""

import os
import pytest
from pathlib import Path

from deriva_ml import DerivaML, DemoML
from deriva_ml.demo_catalog import create_demo_catalog


@pytest.fixture
def test_host():
    """Get the test host from environment or use default."""
    return os.environ.get("DERIVA_HOST", "dev.eye-ai.org")


@pytest.fixture
def test_catalog_id():
    """Get the test catalog ID from environment or use default."""
    return os.environ.get("DERIVA_CATALOG_ID", "eye-ai")


@pytest.fixture
def demo_ml(test_host):
    """Create a demo ML instance for testing."""
    test_catalog = create_demo_catalog(
        test_host,
        "test-schema",
        create_features=True,
        create_datasets=True,
    )
    return DemoML(test_host, test_catalog.catalog_id)


@pytest.fixture
def ml_instance(test_host, test_catalog_id):
    """Create a DerivaML instance for testing."""
    return DerivaML(test_host, test_catalog_id)


@pytest.fixture
def test_files_dir():
    """Get the path to the test files directory."""
    return Path(__file__).parent / "test_files" 