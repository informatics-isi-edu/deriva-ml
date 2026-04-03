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
            "Image_Dataset_Legacy",
            "Dataset_Subject",
            "Dataset_Image",
            "Image_Subject",
            "ClinicalRecord_Observation",
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

        # Clear Report-related tables
        for t in ["OCR_Report", "Report_Asset_Type", "Report_Execution", "Report"]:
            self._delete_table_data(domain_path, t)

        # Clear data tables in dependency order (FK children before parents)
        # Note: ClinicalRecord_Observation is already cleared in domain_assoc_tables above
        for t in ["ClinicalRecord", "Image", "Observation", "Subject"]:
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

        # Drop ALL dynamically created tables (anything not in the permanent schema).
        # This avoids a hardcoded list that must be updated every time a test creates
        # new tables. Uses FK-aware ordering to drop children before parents.
        # Permanent tables created by create_domain_schema() and initial
        # asset/element registration. These survive resets; all others are dropped.
        permanent_tables = {
            # Domain tables from create_domain_schema()
            "Subject", "Image", "Observation", "ClinicalRecord",
            "ClinicalRecord_Observation", "Image_Dataset_Legacy",
            "Report", "OCR_Report",
            # Asset metadata tables (created automatically for Image/Report assets)
            "Image_Asset_Type", "Image_Execution",
            "Report_Asset_Type", "Report_Execution",
            # Element type association tables (created by add_dataset_element_type)
            "Dataset_Subject", "Dataset_Image",
        }
        self._drop_dynamic_tables(permanent_tables)

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

    def _drop_dynamic_tables(self, permanent_tables: set[str]) -> None:
        """Drop all domain schema tables that aren't in the permanent set.

        Uses FK-aware ordering: repeatedly attempts to drop tables, deferring
        any that have inbound FK dependencies. Converges because each pass
        drops at least one leaf table.
        """
        try:
            model = self.catalog.getCatalogModel()
            if self.domain_schema not in model.schemas:
                return
            schema = model.schemas[self.domain_schema]
            dynamic_tables = {
                t for t in schema.tables if t not in permanent_tables
            }
            if not dynamic_tables:
                return

            self._logger.info(f"Dropping {len(dynamic_tables)} dynamic tables: {sorted(dynamic_tables)}")

            # Iteratively drop tables, deferring those with FK dependencies.
            # Each pass should drop at least one table (leaf nodes first).
            max_passes = len(dynamic_tables) + 1  # Worst case: one per pass
            for pass_num in range(max_passes):
                if not dynamic_tables:
                    break
                # Refresh model to reflect drops from previous pass
                model = self.catalog.getCatalogModel()
                if self.domain_schema not in model.schemas:
                    break
                schema = model.schemas[self.domain_schema]

                deferred = set()
                dropped_any = False
                for t in sorted(dynamic_tables):
                    if t not in schema.tables:
                        # Already gone (dropped in a previous iteration this pass)
                        continue
                    try:
                        schema.tables[t].drop()
                        self._logger.debug(f"Dropped {t} (pass {pass_num + 1})")
                        dropped_any = True
                    except Exception as e:
                        err_str = str(e).lower()
                        if "depend" in err_str or "foreign key" in err_str or "conflict" in err_str:
                            deferred.add(t)
                            self._logger.debug(f"Deferred {t} (pass {pass_num + 1})")
                        else:
                            self._logger.warning(f"Could not drop {t}: {e}")

                dynamic_tables = deferred
                if not dropped_any and dynamic_tables:
                    # No progress — remaining tables have circular deps or permanent refs
                    self._logger.warning(
                        f"No tables dropped in pass {pass_num + 1}, "
                        f"remaining: {sorted(dynamic_tables)}"
                    )
                    break

            if dynamic_tables:
                self._logger.warning(f"Could not drop tables: {sorted(dynamic_tables)}")
        except Exception as e:
            self._logger.warning(f"Error in _drop_dynamic_tables: {e}")

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
            # Verify data actually exists — state flag can be stale if a
            # previous fixture's teardown set state without repopulating.
            pb = self.catalog.getPathBuilder()
            domain_path = pb.schemas[self.domain_schema]
            try:
                subjects = list(domain_path.tables["Subject"].path.entities().fetch())
                if len(subjects) > 0:
                    return ml
                self._logger.info(
                    "State is POPULATED but Subject table is empty — repopulating"
                )
                self.state = CatalogState.EMPTY
            except Exception:
                self.state = CatalogState.EMPTY

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
