"""Catalog management for efficient test fixture handling.

This module provides a CatalogManager class that efficiently manages catalog
lifecycle for testing. It supports:

1. Session-scoped catalog creation (expensive, done once)
2. Fast table-level reset between tests (cheap)
3. Optional population states for different test needs
4. Proper cleanup and resource management

The key insight is that catalog creation (~5-10 seconds) is far more expensive
than table-level cleanup (~0.1-0.5 seconds). By reusing catalogs across tests
and only resetting data, we can dramatically reduce test suite runtime.

Example:
    @pytest.fixture(scope="session")
    def catalog_manager(catalog_host):
        manager = CatalogManager(catalog_host)
        yield manager
        manager.destroy()

    @pytest.fixture(scope="function")
    def clean_ml(catalog_manager, tmp_path):
        catalog_manager.reset()
        return catalog_manager.get_ml_instance(tmp_path)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from urllib.parse import quote as urlquote

from deriva.core import ErmrestCatalog
from deriva.core.datapath import DataPathException

from deriva_ml import DerivaML
from deriva_ml.core.definitions import MLVocab
from deriva_ml.demo_catalog import (
    DatasetDescription,
    create_demo_datasets,
    create_demo_features,
    create_domain_schema,
    populate_demo_catalog,
)
from deriva_ml.execution import ExecutionConfiguration
from deriva_ml.schema import create_ml_catalog

if TYPE_CHECKING:
    pass


class CatalogState(Enum):
    """Current state of the catalog data."""

    EMPTY = auto()  # Schema only, no data
    POPULATED = auto()  # Has subjects/images
    WITH_FEATURES = auto()  # Has features defined
    WITH_DATASETS = auto()  # Has dataset hierarchy


@dataclass
class CatalogManager:
    """Manages a test catalog lifecycle efficiently.

    This class provides efficient catalog management by:
    1. Creating the catalog once (expensive)
    2. Resetting data between tests (cheap)
    3. Tracking catalog state to avoid redundant work

    The manager supports different "population levels" so tests can request
    the minimum state they need without over-populating.

    Attributes:
        hostname: The Deriva server hostname.
        catalog: The underlying ErmrestCatalog instance.
        catalog_id: The catalog identifier.
        domain_schema: Name of the domain schema.
        state: Current population state of the catalog.
    """

    hostname: str
    domain_schema: str = "test-schema"
    project_name: str = "ml-test"
    catalog: ErmrestCatalog | None = field(default=None, init=False)
    catalog_id: str | int | None = field(default=None, init=False)
    state: CatalogState = field(default=CatalogState.EMPTY, init=False)
    _dataset_description: DatasetDescription | None = field(default=None, init=False)
    _tmpdir: TemporaryDirectory | None = field(default=None, init=False)
    _logger: logging.Logger = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Create the catalog after initialization."""
        self._logger = logging.getLogger(__name__)
        self._create_catalog()

    def _create_catalog(self) -> None:
        """Create the ML catalog and domain schema."""
        self._logger.info(f"Creating test catalog on {self.hostname}")
        self.catalog = create_ml_catalog(self.hostname, project_name=self.project_name)
        self.catalog_id = self.catalog.catalog_id
        create_domain_schema(self.catalog, self.domain_schema)
        self.state = CatalogState.EMPTY
        self._logger.info(f"Created catalog {self.catalog_id}")

    def destroy(self) -> None:
        """Destroy the catalog and clean up resources."""
        if self._tmpdir:
            self._tmpdir.cleanup()
            self._tmpdir = None

        if self.catalog:
            self._logger.info(f"Deleting catalog {self.catalog_id}")
            self.catalog.delete_ermrest_catalog(really=True)
            self.catalog = None
            self.catalog_id = None
            self.state = CatalogState.EMPTY

    def reset(self, force: bool = False) -> None:
        """Reset catalog to empty state (schema only, no data).

        This is a fast operation that clears data from tables while
        preserving the schema structure. Much faster than destroying
        and recreating the catalog.

        Args:
            force: If True, perform full reset even if state is already EMPTY.
        """
        if self.state == CatalogState.EMPTY and not force:
            self._logger.debug("Catalog already empty, skipping reset")
            return

        self._logger.debug("Resetting catalog to empty state")
        pb = self.catalog.getPathBuilder()
        ml_path = pb.schemas["deriva-ml"]
        domain_path = pb.schemas[self.domain_schema]

        # Clear ML schema tables in dependency order
        ml_tables = [
            "Dataset_Execution",
            "Dataset_Version",
            "Dataset_Dataset",
            "Dataset",
            "Workflow_Execution",
            "Execution",
            "Workflow",
        ]
        for t in ml_tables:
            self._delete_table_data(ml_path, t)

        # Clear domain schema association tables
        domain_assoc_tables = [
            "Dataset_Subject",
            "Dataset_Image",
            "Image_Subject",
        ]
        for t in domain_assoc_tables:
            self._delete_table_data(domain_path, t)

        # Clear feature execution tables
        feature_tables = [
            "Execution_Image_BoundingBox",
            "Execution_Image_Quality",
            "Execution_Subject_Health",
        ]
        for t in feature_tables:
            self._delete_table_data(domain_path, t)

        # Clear data tables (Image before Subject due to FK)
        for t in ["Image", "Subject"]:
            self._delete_table_data(domain_path, t)

        # Clear custom vocabularies (domain schema) - just data
        domain_vocab_tables = [
            "SubjectHealth",
            "ImageQuality",
        ]
        for t in domain_vocab_tables:
            self._delete_table_data(domain_path, t)

        # Note: We do NOT clear ML schema vocabulary tables (Dataset_Type,
        # Workflow_Type, Asset_Type) because they contain system-required terms
        # like "Execution_Config" that are created during schema initialization.
        # Deleting these would break the catalog.

        # Drop dynamically created tables in dependency order:
        # 1. First drop tables that reference other tables (FK children)
        # 2. Then drop the referenced tables (FK parents)

        # Feature execution tables (reference Image, Subject, and assets)
        feature_execution_tables = [
            "Execution_Image_BoundingBox",
            "Execution_Image_Quality",
            "Execution_Subject_Health",
        ]
        for t in feature_execution_tables:
            self._drop_table_if_exists(self.domain_schema, t)

        # Asset association tables (created automatically for assets)
        # These reference the asset tables and must be dropped first
        asset_assoc_tables = [
            "BoundingBox_Asset_Type",
            "BoundingBox_Execution",
        ]
        for t in asset_assoc_tables:
            self._drop_table_if_exists(self.domain_schema, t)

        # Asset tables created by create_asset() (referenced by association tables)
        asset_tables = ["BoundingBox"]
        for t in asset_tables:
            self._drop_table_if_exists(self.domain_schema, t)

        # Custom vocabulary tables created by create_vocabulary()
        for t in domain_vocab_tables:
            self._drop_table_if_exists(self.domain_schema, t)

        # Test-specific tables that may be created by individual tests
        # First drop association tables, then the main tables
        test_assoc_tables = [
            "TestTableExecution_Execution",
            "TestTableExecution_Asset_Type",
            "TestTable_Execution",
            "TestTable_Asset_Type",
            "Dataset_TestTableExecution",
            "Dataset_TestTable",
        ]
        for t in test_assoc_tables:
            self._drop_table_if_exists(self.domain_schema, t)

        test_tables = ["TestTableExecution", "TestTable"]
        for t in test_tables:
            self._drop_table_if_exists(self.domain_schema, t)

        # Clear catalog history snapshots
        self._clear_history()

        self.state = CatalogState.EMPTY
        self._dataset_description = None

    def _delete_table_data(self, schema_path, table_name: str) -> None:
        """Delete all data from a table, ignoring missing tables."""
        try:
            schema_path.tables[table_name].path.delete()
        except (DataPathException, KeyError):
            pass
        except Exception as e:
            # Log but don't fail - table may not exist in all configurations
            self._logger.debug(f"Could not delete from {table_name}: {e}")

    def _drop_table_if_exists(self, schema_name: str, table_name: str) -> None:
        """Drop a table from the schema if it exists."""
        try:
            model = self.catalog.getCatalogModel()
            if schema_name not in model.schemas:
                return
            schema = model.schemas[schema_name]
            if table_name in schema.tables:
                self._logger.info(f"Dropping table {schema_name}.{table_name}")
                schema.tables[table_name].drop()
                self._logger.info(f"Successfully dropped table {schema_name}.{table_name}")
        except KeyError:
            # Table or schema doesn't exist - that's fine
            pass
        except Exception as e:
            self._logger.warning(f"Could not drop table {schema_name}.{table_name}: {e}")

    def _clear_history(self) -> None:
        """Clear catalog history snapshots."""
        try:
            cat_desc = self.catalog.get("/").json()
            latest = cat_desc["snaptime"]
            self.catalog.delete("/history/,%s" % (urlquote(latest),))
        except Exception as e:
            self._logger.debug(f"Could not clear history: {e}")

    def get_ml_instance(self, working_dir: Path | str) -> DerivaML:
        """Get a DerivaML instance for this catalog.

        Args:
            working_dir: Working directory for the ML instance.

        Returns:
            A DerivaML instance connected to this catalog.
        """
        return DerivaML(
            self.hostname,
            self.catalog_id,
            default_schema=self.domain_schema,
            working_dir=working_dir,
            use_minid=False,
        )

    def ensure_populated(self, working_dir: Path | str) -> DerivaML:
        """Ensure catalog has basic data (subjects, images).

        If already populated, returns existing state. Otherwise populates
        and returns the ML instance.

        Args:
            working_dir: Working directory for the ML instance.

        Returns:
            A DerivaML instance with populated data.
        """
        ml = self.get_ml_instance(working_dir)

        if self.state.value >= CatalogState.POPULATED.value:
            return ml

        self._add_workflow_type(ml)
        workflow = ml.create_workflow(name="Test Population", workflow_type="Test Workflow")
        execution = ml.create_execution(workflow=workflow, configuration=ExecutionConfiguration())
        with execution.execute() as exe:
            populate_demo_catalog(exe)

        self.state = CatalogState.POPULATED
        return ml

    def ensure_features(self, working_dir: Path | str) -> DerivaML:
        """Ensure catalog has features defined.

        Args:
            working_dir: Working directory for the ML instance.

        Returns:
            A DerivaML instance with features.
        """
        ml = self.ensure_populated(working_dir)

        if self.state.value >= CatalogState.WITH_FEATURES.value:
            return ml

        workflow = ml.create_workflow(name="Feature Creation", workflow_type="Test Workflow")
        execution = ml.create_execution(workflow=workflow, configuration=ExecutionConfiguration())
        with execution.execute() as exe:
            create_demo_features(exe)

        self.state = CatalogState.WITH_FEATURES
        return ml

    def ensure_datasets(self, working_dir: Path | str) -> tuple[DerivaML, DatasetDescription]:
        """Ensure catalog has dataset hierarchy.

        Args:
            working_dir: Working directory for the ML instance.

        Returns:
            Tuple of (DerivaML instance, DatasetDescription).
        """
        ml = self.ensure_features(working_dir)

        if self.state == CatalogState.WITH_DATASETS and self._dataset_description:
            return ml, self._dataset_description

        workflow = ml.create_workflow(name="Dataset Creation", workflow_type="Test Workflow")
        execution = ml.create_execution(workflow=workflow, configuration=ExecutionConfiguration())
        with execution.execute() as exe:
            self._dataset_description = create_demo_datasets(exe)

        self.state = CatalogState.WITH_DATASETS
        return ml, self._dataset_description

    def _add_workflow_type(self, ml: DerivaML) -> None:
        """Add the test workflow type if not already present."""
        try:
            ml.lookup_term(MLVocab.workflow_type, "Test Workflow")
        except Exception:
            ml.add_term(
                MLVocab.workflow_type,
                "Test Workflow",
                description="Workflow type for testing",
            )

    @property
    def dataset_description(self) -> DatasetDescription | None:
        """Get the current dataset description, if datasets have been created."""
        return self._dataset_description

    @property
    def default_schema(self) -> str:
        """Alias for domain_schema to match DerivaML API."""
        return self.domain_schema


# Fixture helper functions for common patterns


def reset_for_test(manager: CatalogManager) -> None:
    """Reset catalog for a new test.

    Call this at the start of each test that needs a clean slate.
    """
    manager.reset()


def get_clean_ml(manager: CatalogManager, working_dir: Path) -> DerivaML:
    """Get a clean ML instance with empty catalog.

    Args:
        manager: The catalog manager.
        working_dir: Working directory for the instance.

    Returns:
        A DerivaML instance with clean catalog.
    """
    manager.reset()
    return manager.get_ml_instance(working_dir)


def get_populated_ml(manager: CatalogManager, working_dir: Path) -> DerivaML:
    """Get an ML instance with populated data.

    Resets first to ensure clean state, then populates.

    Args:
        manager: The catalog manager.
        working_dir: Working directory for the instance.

    Returns:
        A DerivaML instance with populated data.
    """
    manager.reset()
    return manager.ensure_populated(working_dir)


def get_ml_with_datasets(
    manager: CatalogManager, working_dir: Path
) -> tuple[DerivaML, DatasetDescription]:
    """Get an ML instance with full dataset hierarchy.

    Resets first to ensure clean state, then creates everything.

    Args:
        manager: The catalog manager.
        working_dir: Working directory for the instance.

    Returns:
        Tuple of (DerivaML instance, DatasetDescription).
    """
    manager.reset()
    return manager.ensure_datasets(working_dir)
