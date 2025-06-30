"""
Pytest configuration and shared fixtures.
"""

import os
from pathlib import Path

import pytest
from demo_catalog import DatasetDescription

from deriva_ml import RID, DerivaML
from deriva_ml.demo_catalog import (
    create_demo_catalog,
    create_demo_datasets,
    populate_demo_catalog,
    reset_demo_catalog,
)


class MLCatalog:
    def __init__(self, hostname):
        self.catalog = create_demo_catalog(
            hostname,
            domain_schema="test-schema",
            project_name="ml-test",
            populate=False,
            create_features=False,
            create_datasets=False,
            on_exit_delete=False,
        )
        self.catalog_id = self.catalog.catalog_id
        self.hostname = hostname
        self.domain_schema = "ml-test"
        print("🚀 Created demo catalog")

    def cleanup(self):
        print("Deleting demo catalog")
        self.catalog.delete_ermrest_catalog(really=True)


class MLDatasetTest:
    def __init__(self, catalog: MLCatalog, tmp_dir: Path):
        reset_demo_catalog(catalog.catalog)
        self.ml_instance = DerivaML(catalog.hostname, catalog.catalog_id, working_dir=tmp_dir, use_minid=False)

        populate_demo_catalog(self.ml_instance)
        self.dataset_description: DatasetDescription = create_demo_datasets(self.ml_instance)
        self.catalog = catalog

    def find_datasets(self) -> dict[RID, list[RID]]:
        """Return a dictionary that whose key is a dataset RID and whose value is a dictionary of member dataset RIDs."""
        return self._find_datasets(self.dataset_description)

    def _find_datasets(self, dataset: DatasetDescription) -> dict[RID, list[RID]]:
        return {dataset.rid: dataset.member_rids} | {
            member_type: member_list
            for dset in dataset.members.get("Dataset", [])
            for member_type, member_list in self._find_datasets(dset).items()
            if member_list != []
        }

    def member_list(self) -> list[tuple[RID, list[RID]]]:
        return self._member_list(self.dataset_description)

    def _member_list(self, dataset: DatasetDescription) -> list[tuple[RID, list[RID]]]:
        return [(dataset.rid, dataset.member_rids)] + [
            self._member_list(d) for t, d in dataset.members.items() if t == "Dataset"
        ]


@pytest.fixture(scope="session")
def test_host():
    """Get the test host from the environment or use default."""
    return os.environ.get("DERIVA_HOST", "localhost")


@pytest.fixture
def test_catalog_id():
    """Get the test catalog ID from environment or use default."""
    return os.environ.get("DERIVA_CATALOG_ID", "eye-ai")


@pytest.fixture(scope="session")
def ml_catalog(test_host):
    """Create a demo ML instance for testing with schema, but no data..""" ""
    resource = MLCatalog(test_host)
    yield resource
    resource.cleanup()


@pytest.fixture(scope="function")
def test_ml_catalog(ml_catalog, tmp_path):
    """Create a demo ML instance for testing.   Resets after each class.""" ""
    reset_demo_catalog(ml_catalog.catalog)
    return DerivaML(ml_catalog.hostname, ml_catalog.catalog_id, use_minid=False, working_dir=tmp_path)


@pytest.fixture(scope="function")
def test_ml_catalog_populated(ml_catalog, tmp_path):
    """Create a demo ML instance for testing with populated domain schema.   Resets after each test."""
    print("Setting up populated catalog for testing... ", end="")
    reset_demo_catalog(ml_catalog.catalog)
    ml_instance = DerivaML(ml_catalog.hostname, ml_catalog.catalog_id, use_minid=False, working_dir=tmp_path)
    populate_demo_catalog(ml_instance)
    return ml_instance


@pytest.fixture(scope="function")
def test_ml_catalog_dataset(ml_catalog, tmp_path):
    """Create a demo ML instance for testing with populated domain schema.   Resets after each test."""
    print("Setting up catalog with datasets for testing... ", end="")
    return MLDatasetTest(ml_catalog, tmp_path)


@pytest.fixture
def test_files_dir():
    """Get the path to the test files directory."""
    return Path(__file__).parent / "test_files"


@pytest.fixture(autouse=True)
def log_start():
    print("\n--- Starting test ---")
    yield
    print("\n--- Ending test ---")
