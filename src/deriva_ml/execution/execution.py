"""Execution management for DerivaML.

This module provides functionality for managing and tracking executions in DerivaML. An execution
represents a computational or manual process that operates on datasets and produces outputs.
The module includes:

- Execution class: Core class for managing execution state and context
- Asset management: Track input and output files
- Status tracking: Monitor and update execution progress
- Dataset handling: Download and materialize required datasets
- Provenance tracking: Record relationships between inputs, processes, and outputs

The Execution class serves as the primary interface for managing the lifecycle of a computational
or manual process within DerivaML.

Typical usage example:
    >>> config = ExecutionConfiguration(workflow="analysis_workflow", description="Data analysis")
    >>> with ml.create_execution(config) as execution:
    ...     execution.download_dataset_bag(dataset_spec)
    ...     # Run analysis
    ...     path = execution.asset_file_path("Model", "model.pt")
    ...     # Write model to path...
    ...
    >>> # IMPORTANT: Upload AFTER the context manager exits
    >>> execution.upload_execution_outputs()

The context manager handles start/stop timing automatically. The upload_execution_outputs()
call must happen AFTER exiting the context manager to ensure proper status tracking.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, List

from deriva.core import format_exception

if TYPE_CHECKING:
    from deriva_ml.asset.asset import Asset
from deriva.core.hatrac_store import HatracStore
from pydantic import ConfigDict, validate_call

from deriva_ml.asset.aux_classes import AssetFilePath
from deriva_ml.asset.manifest import AssetEntry, AssetManifest
from deriva_ml.core.base import DerivaML
from deriva_ml.core.definitions import (
    DRY_RUN_RID,
    RID,
    ExecMetadataType,
    FileSpec,
    FileUploadState,
    MLAsset,
    MLVocab,
    Status,
    UploadProgress,
)
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.aux_classes import DatasetSpec, DatasetVersion
from deriva_ml.dataset.dataset import Dataset
from deriva_ml.dataset.dataset_bag import DatasetBag
from deriva_ml.dataset.upload import (
    asset_root,
    asset_type_path,
    execution_root,
    feature_root,
    feature_value_path,
    flat_asset_dir,
    is_feature_dir,
    normalize_asset_dir,
    table_path,
    upload_directory,
)
from deriva_ml.execution.environment import get_execution_environment
from deriva_ml.execution.execution_configuration import ExecutionConfiguration
from deriva_ml.execution.execution_record import ExecutionRecord
from deriva_ml.execution.workflow import Workflow
from deriva_ml.feature import FeatureRecord
from deriva_ml.model.deriva_ml_database import DerivaMLDatabase

# Keep pycharm from complaining about undefined references in docstrings.
execution: Execution
ml: DerivaML
dataset_spec: DatasetSpec

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


try:
    from IPython.display import Markdown, display
except ImportError:

    def display(s):
        print(s)

    def Markdown(s):
        return s


# Descriptions for execution metadata files, keyed by original filename.
_METADATA_DESCRIPTIONS: dict[str, str] = {
    "config.yaml": ("Resolved Hydra configuration: complete merged config values used for this execution"),
    "overrides.yaml": ("Hydra overrides: command-line and sweep parameters that modified the default configuration"),
    "hydra.yaml": ("Hydra runtime config: job name, config group choices, sweep parameters, and runtime metadata"),
    "configuration.json": ("DerivaML execution configuration: datasets, assets, workflow, and config group choices"),
    "uv.lock": ("Python dependency lockfile: pinned versions of all packages in the execution environment"),
}

_ENV_SNAPSHOT_DESCRIPTION = "Runtime environment snapshot: installed packages, OS, platform, and Python configuration"


class Execution:
    """Manages the lifecycle and context of a DerivaML execution.

    An Execution represents a computational or manual process within DerivaML. It provides:
    - Dataset materialization and access
    - Asset management (inputs and outputs)
    - Status tracking and updates
    - Provenance recording
    - Result upload and cataloging

    The class handles downloading required datasets and assets, tracking execution state,
    and managing the upload of results. Every dataset and file generated is associated
    with an execution record for provenance tracking.

    Attributes:
        dataset_rids (list[RID]): RIDs of datasets used in the execution.
        datasets (list[DatasetBag]): Materialized dataset objects.
        configuration (ExecutionConfiguration): Execution settings and parameters.
        workflow_rid (RID): RID of the associated workflow.
        status (Status): Current execution status.
        asset_paths (list[AssetFilePath]): Paths to execution assets.
        start_time (datetime | None): When execution started.
        stop_time (datetime | None): When execution completed.

    Example:
        The context manager handles start/stop timing. Upload must be called AFTER
        the context manager exits::

            >>> config = ExecutionConfiguration(
            ...     workflow="analysis",
            ...     description="Process samples",
            ... )
            >>> with ml.create_execution(config) as execution:
            ...     bag = execution.download_dataset_bag(dataset_spec)
            ...     # Run analysis using bag.path
            ...     output_path = execution.asset_file_path("Model", "model.pt")
            ...     # Write results to output_path
            ...
            >>> # IMPORTANT: Call upload AFTER exiting the context manager
            >>> execution.upload_execution_outputs()
    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        configuration: ExecutionConfiguration,
        ml_object: DerivaML,
        workflow: Workflow | None = None,
        reload: RID | None = None,
        dry_run: bool = False,
    ):
        """Initializes an Execution instance.

        Creates a new execution or reloads an existing one. Initializes the execution
        environment, downloads required datasets, and sets up asset tracking.

        Args:
            configuration: Settings and parameters for the execution.
            ml_object: DerivaML instance managing the execution.
            workflow: Optional Workflow object. If not specified, the workflow is taken from
                the ExecutionConfiguration object. Must be a Workflow object, not a RID.
            reload: Optional RID of existing execution to reload.
            dry_run: If True, don't create catalog records or upload results.

        Raises:
            DerivaMLException: If initialization fails, configuration is invalid,
                or workflow is not a Workflow object.

        Example:
            Create an execution with a workflow::

                >>> workflow = ml.lookup_workflow("2-ABC1")
                >>> config = ExecutionConfiguration(
                ...     workflow=workflow,
                ...     description="Process data"
                ... )
                >>> execution = Execution(config, ml)

            Or pass workflow separately::

                >>> workflow = ml.lookup_workflow_by_url(
                ...     "https://github.com/org/repo/blob/abc123/analysis.py"
                ... )
                >>> config = ExecutionConfiguration(description="Run analysis")
                >>> execution = Execution(config, ml, workflow=workflow)
        """

        self.asset_paths: dict[str, list[AssetFilePath]] = {}
        self.configuration = configuration
        self._ml_object = ml_object
        self._model = ml_object.model
        self._logger = ml_object._logger
        self.start_time = None
        self.stop_time = None
        self._status = Status.created
        self.uploaded_assets: dict[str, list[AssetFilePath]] | None = None
        self.configuration.argv = sys.argv
        self._execution_record: ExecutionRecord | None = None  # Lazily created after RID is assigned

        self.dataset_rids: List[RID] = []
        self.datasets: list[DatasetBag] = []

        self._working_dir = self._ml_object.working_dir
        self._cache_dir = self._ml_object.cache_dir
        if self._working_dir is None:
            raise DerivaMLException(
                "DerivaML working_dir is not set. "
                "Ensure the DerivaML instance was initialized with a valid working_dir."
            )
        self._dry_run = dry_run

        # Make sure we have a valid Workflow object.
        if workflow:
            self.configuration.workflow = workflow

        if self.configuration.workflow is None:
            raise DerivaMLException("Workflow must be specified either in configuration or as a parameter")

        if not isinstance(self.configuration.workflow, Workflow):
            raise DerivaMLException(
                f"Workflow must be a Workflow object, not {type(self.configuration.workflow).__name__}. "
                "Use ml.lookup_workflow(rid) or ml.lookup_workflow_by_url(url) to get a Workflow object."
            )

        # Validate workflow type(s) and register in catalog
        for wt in self.configuration.workflow.workflow_type:
            self._ml_object.lookup_term(MLVocab.workflow_type, wt)
        self.workflow_rid = (
            self._ml_object.add_workflow(self.configuration.workflow) if not self._dry_run else DRY_RUN_RID
        )

        # Validate the datasets and assets to be valid.
        for d in self.configuration.datasets:
            if self._ml_object.resolve_rid(d.rid).table.name != "Dataset":
                raise DerivaMLException("Dataset specified in execution configuration is not a dataset")

        for a in self.configuration.assets:
            if not self._model.is_asset(self._ml_object.resolve_rid(a.rid).table.name):
                raise DerivaMLException("Asset specified in execution configuration is not an asset table")

        schema_path = self._ml_object.pathBuilder().schemas[self._ml_object.ml_schema]
        if reload:
            self.execution_rid = reload
            if self.execution_rid == DRY_RUN_RID:
                self._dry_run = True
        elif self._dry_run:
            self.execution_rid = DRY_RUN_RID
        else:
            self.execution_rid = schema_path.Execution.insert(
                [
                    {
                        "Description": self.configuration.description,
                        "Workflow": self.workflow_rid,
                    }
                ]
            )[0]["RID"]

        if rid_path := os.environ.get("DERIVA_ML_SAVE_EXECUTION_RID", None):
            # Put execution_rid into the provided file path so we can find it later.
            with Path(rid_path).open("w") as f:
                json.dump(
                    {
                        "hostname": self._ml_object.host_name,
                        "catalog_id": self._ml_object.catalog_id,
                        "workflow_rid": self.workflow_rid,
                        "execution_rid": self.execution_rid,
                    },
                    f,
                )

        # Create a directory for execution rid so we can recover the state in case of a crash.
        execution_root(prefix=self._ml_object.working_dir, exec_rid=self.execution_rid)

        # Create the ExecutionRecord to handle catalog state operations
        if not self._dry_run:
            self._execution_record = ExecutionRecord(
                execution_rid=self.execution_rid,
                workflow=self.configuration.workflow,
                status=Status.created,
                description=self.configuration.description,
                _ml_instance=self._ml_object,
                _logger=self._logger,
            )

        self._initialize_execution(reload)

    @classmethod
    def _from_registry(cls, *, ml_object, execution_rid: str) -> "Execution":
        """Bind an Execution to an existing SQLite registry row.

        Distinct from create_execution: does NOT contact the catalog and
        does NOT POST a new row. Called by ml.resume_execution.

        Temporary implementation for Group D — Group E replaces the body
        to wire up read-through lifecycle properties.

        Args:
            ml_object: The DerivaML instance this Execution is bound to.
            execution_rid: The pre-existing Execution RID.

        Returns:
            A minimally-initialized Execution with just enough state for
            execution_rid lookup.
        """
        # Minimal construction: skip the existing __init__'s catalog
        # interactions. Store the rid and ml_object so Group E has a
        # starting point.
        instance = cls.__new__(cls)
        instance._ml_object = ml_object
        instance._model = ml_object.model
        instance._logger = getattr(ml_object, "_logger", None)
        instance.execution_rid = execution_rid
        instance._dry_run = False
        # Fields the existing class expects to exist:
        instance.datasets = []
        instance.dataset_rids = []
        instance.asset_paths = {}
        instance.configuration = None  # Group E loads from config_json
        instance._working_dir = ml_object.working_dir
        instance._cache_dir = ml_object.cache_dir
        instance.start_time = None
        instance.stop_time = None
        instance._status = Status.created
        instance.uploaded_assets = None
        instance._execution_record = None
        instance.workflow_rid = None
        return instance

    def _save_runtime_environment(self):
        runtime_env_path = self.asset_file_path(
            asset_name="Execution_Metadata",
            file_name=f"environment_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            asset_types=ExecMetadataType.runtime_env.value,
            description=_ENV_SNAPSHOT_DESCRIPTION,
        )
        with Path(runtime_env_path).open("w") as fp:
            json.dump(get_execution_environment(), fp)

    def _upload_hydra_config_assets(self):
        """Upload hydra assets to the catalog with Hydra_Config type."""
        hydra_runtime_output_dir = self._ml_object.hydra_runtime_output_dir
        if hydra_runtime_output_dir:
            timestamp = hydra_runtime_output_dir.parts[-1]
            for hydra_asset in hydra_runtime_output_dir.rglob("*"):
                if hydra_asset.is_dir():
                    continue
                # Register file for upload (side effect); result intentionally unused
                # Use Hydra_Config type for Hydra YAML configuration files
                self.asset_file_path(
                    asset_name=MLAsset.execution_metadata,
                    file_name=hydra_runtime_output_dir / hydra_asset,
                    rename_file=f"hydra-{timestamp}-{hydra_asset.name}",
                    asset_types=ExecMetadataType.hydra_config.value,
                    description=self._get_metadata_description(hydra_asset.name),
                )

    @staticmethod
    def _get_metadata_description(file_name: str) -> str | None:
        """Resolve a description for an execution metadata file.

        Handles both direct filenames (configuration.json, uv.lock) and
        hydra-renamed files (hydra-{timestamp}-{original_name}).

        Args:
            file_name: The filename as it appears in the catalog.

        Returns:
            A human-readable description, or None if the file is unrecognized.
        """
        if file_name in _METADATA_DESCRIPTIONS:
            return _METADATA_DESCRIPTIONS[file_name]

        # Hydra renamed files: "hydra-{timestamp}-{original_name}"
        if file_name.startswith("hydra-"):
            for original, desc in _METADATA_DESCRIPTIONS.items():
                if file_name.endswith(f"-{original}"):
                    return desc

        if file_name.startswith("environment_snapshot_"):
            return _ENV_SNAPSHOT_DESCRIPTION

        return None

    def _set_asset_descriptions(self, uploaded_assets: dict[str, list[AssetFilePath]]) -> None:
        """Set Description on asset records after upload.

        Applies descriptions from two sources:
        1. Hardcoded descriptions for known execution metadata files.
        2. User-provided descriptions passed via the ``description`` parameter
           of ``asset_file_path()``.

        Args:
            uploaded_assets: Dict mapping "{schema}/{table}" to uploaded AssetFilePaths.
        """
        manifest = self._get_manifest()
        pb = self._ml_object.pathBuilder()

        # Group updates by schema/table for batch efficiency
        table_updates: dict[str, list[dict[str, str]]] = {}

        for table_key, assets in uploaded_assets.items():
            for asset in assets:
                # Determine description: check manifest first, then fall back to
                # hardcoded metadata descriptions for Execution_Metadata files.
                table_name = table_key.split("/")[1] if "/" in table_key else table_key
                manifest_key = f"{table_name}/{asset.file_name}"
                entry = manifest.assets.get(manifest_key)

                description = None
                if entry and entry.description:
                    description = entry.description
                elif table_name == "Execution_Metadata":
                    description = self._get_metadata_description(asset.file_name)

                if description and asset.asset_rid:
                    table_updates.setdefault(table_key, []).append({"RID": asset.asset_rid, "Description": description})

        for table_key, updates in table_updates.items():
            schema, table_name = table_key.split("/", 1) if "/" in table_key else ("", table_key)
            pb.schemas[schema].tables[table_name].update(updates)

    def _initialize_execution(self, reload: RID | None = None) -> None:
        """Initialize the execution environment.

        Sets up the working directory, downloads required datasets and assets,
        and saves initial configuration metadata.

        Args:
            reload: Optional RID of a previously initialized execution to reload.

        Raises:
            DerivaMLException: If initialization fails.
        """
        # Materialize bdbag
        for dataset in self.configuration.datasets:
            self.update_status(Status.initializing, f"Materialize bag {dataset.rid}... ")
            self.datasets.append(self.download_dataset_bag(dataset))
            self.dataset_rids.append(dataset.rid)

        # Update execution info
        schema_path = self._ml_object.pathBuilder().schemas[self._ml_object.ml_schema]
        if self.dataset_rids and not (reload or self._dry_run):
            schema_path.Dataset_Execution.insert(
                [{"Dataset": d, "Execution": self.execution_rid} for d in self.dataset_rids]
            )

        # Download assets....
        self.update_status(Status.running, "Downloading assets ...")
        self.asset_paths = {}
        for asset_spec in self.configuration.assets:
            asset_rid = asset_spec.rid
            use_cache = asset_spec.cache
            asset_table = self._ml_object.resolve_rid(asset_rid).table.name
            dest_dir = (
                execution_root(self._ml_object.working_dir, self.execution_rid) / "downloaded-assets" / asset_table
            )
            dest_dir.mkdir(parents=True, exist_ok=True)
            self.asset_paths.setdefault(asset_table, []).append(
                self.download_asset(
                    asset_rid=asset_rid,
                    dest_dir=dest_dir,
                    update_catalog=not (reload or self._dry_run),
                    use_cache=use_cache,
                )
            )

        # Save configuration details and upload (skip in dry_run mode)
        if not reload and not self._dry_run:
            # Save DerivaML configuration with Deriva_Config type
            cfile = self.asset_file_path(
                asset_name=MLAsset.execution_metadata,
                file_name="configuration.json",
                asset_types=ExecMetadataType.deriva_config.value,
                description=_METADATA_DESCRIPTIONS["configuration.json"],
            )

            with Path(cfile).open("w", encoding="utf-8") as config_file:
                json.dump(self.configuration.model_dump(mode="json"), config_file)
            # Only try to copy uv.lock if git_root is available (local workflow)
            if self.configuration.workflow.git_root:
                lock_file = Path(self.configuration.workflow.git_root) / "uv.lock"
            else:
                lock_file = None
            if lock_file and lock_file.exists():
                _ = self.asset_file_path(
                    asset_name=MLAsset.execution_metadata,
                    file_name=lock_file,
                    asset_types=ExecMetadataType.execution_config.value,
                    description=_METADATA_DESCRIPTIONS["uv.lock"],
                )

            self._upload_hydra_config_assets()

            # save runtime env
            self._save_runtime_environment()

            # Now upload the files so we have the info in case the execution fails.
            self.uploaded_assets = self._upload_execution_dirs()
            self._set_asset_descriptions(self.uploaded_assets)
        self.start_time = datetime.now()
        self.update_status(Status.pending, "Initialize status finished.")

    @property
    def status(self) -> Status:
        """Get the current execution status.

        Returns:
            Status: The current status (Created, Running, Completed, Failed, etc.).
        """
        if self._execution_record is not None:
            return self._execution_record.status
        return self._status

    @status.setter
    def status(self, value: Status) -> None:
        """Set the execution status.

        Args:
            value: The new status value.
        """
        self._status = value
        if self._execution_record is not None:
            self._execution_record._status = value

    @property
    def execution_record(self) -> ExecutionRecord | None:
        """Get the ExecutionRecord for catalog operations.

        Returns:
            ExecutionRecord if not in dry_run mode, None otherwise.
        """
        return self._execution_record

    @property
    def working_dir(self) -> Path:
        """Return the working directory for the execution."""
        return self._execution_root

    @property
    def _execution_root(self) -> Path:
        """Get the root directory for this execution's files.

        Returns:
            Path to the execution-specific directory.
        """
        return execution_root(self._working_dir, self.execution_rid)

    @property
    def _feature_root(self) -> Path:
        """Get the root directory for feature files.

        Returns:
            Path to the feature directory within the execution.
        """
        return feature_root(self._working_dir, self.execution_rid)

    @property
    def _asset_root(self) -> Path:
        """Get the root directory for asset files.

        Returns:
            Path to the asset directory within the execution.
        """
        return asset_root(self._working_dir, self.execution_rid)

    @property
    def database_catalog(self) -> DerivaMLDatabase | None:
        """Get a catalog-like interface for downloaded datasets.

        Returns a DerivaMLDatabase that implements the DerivaMLCatalog
        protocol, allowing the same code to work with both live catalogs
        and downloaded bags.

        This is useful for writing code that can operate on either a live
        catalog (via DerivaML) or on downloaded bags (via DerivaMLDatabase).

        Returns:
            DerivaMLDatabase wrapping the primary downloaded dataset's model,
            or None if no datasets have been downloaded.

        Example:
            >>> with ml.create_execution(config) as exe:
            ...     if exe.database_catalog:
            ...         db = exe.database_catalog
            ...         # Use same interface as DerivaML
            ...         dataset = db.lookup_dataset("4HM")
            ...         term = db.lookup_term("Diagnosis", "cancer")
            ...     else:
            ...         # No datasets downloaded, use live catalog
            ...         pass
        """
        if not self.datasets:
            return None
        # Use the first dataset's model as the primary
        return DerivaMLDatabase(self.datasets[0].model)

    @property
    def catalog(self) -> "DerivaML":
        """Get the live catalog (DerivaML) instance for this execution.

        This provides access to the live catalog for operations that require
        catalog connectivity, such as looking up datasets or other read operations.

        Returns:
            DerivaML: The live catalog instance.

        Example:
            >>> with ml.create_execution(config) as exe:
            ...     # Use live catalog for lookups
            ...     existing_dataset = exe.catalog.lookup_dataset("1-ABC")
        """
        return self._ml_object

    def add_features(self, features: list[FeatureRecord]) -> int:
        """Add feature values to the catalog in batch.

        Convenience method that delegates to ``catalog.add_features()``.
        Automatically sets the ``Execution`` field on each record to this
        execution's RID if not already set.

        Args:
            features: List of FeatureRecord instances to insert. All must share
                the same feature definition. Create records using the class
                returned by ``Feature.feature_record_class()``.

        Returns:
            Number of feature records inserted.

        Example:
            >>> feature = exe.catalog.lookup_feature("Image", "Classification")
            >>> RecordClass = feature.feature_record_class()
            >>> records = [
            ...     RecordClass(Image="1-ABC", Image_Class="Normal"),
            ...     RecordClass(Image="1-DEF", Image_Class="Abnormal"),
            ... ]
            >>> exe.add_features(records)  # Execution RID set automatically
        """
        for f in features:
            if f.Execution is None:
                f.Execution = self.execution_rid
        return self._ml_object.add_features(features)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def download_dataset_bag(self, dataset: DatasetSpec) -> DatasetBag:
        """Downloads and materializes a dataset for use in the execution.

        Downloads the specified dataset as a BDBag and materializes it in the execution's
        working directory. The dataset version is determined by the DatasetSpec.

        Args:
            dataset: Specification of the dataset to download, including version and
                materialization options.

        Returns:
            DatasetBag: Object containing:
                - path: Local filesystem path to downloaded dataset
                - rid: Dataset's Resource Identifier
                - minid: Dataset's Minimal Viable Identifier

        Raises:
            DerivaMLException: If download or materialization fails.

        Example:
            >>> spec = DatasetSpec(rid="1-abc123", version="1.2.0")
            >>> bag = execution.download_dataset_bag(spec)
            >>> print(f"Downloaded to {bag.path}")
        """
        return self._ml_object.download_dataset_bag(dataset)

    @validate_call
    def update_status(self, status: Status, msg: str) -> None:
        """Updates the execution's status in the catalog.

        Records a new status and associated message in the catalog, allowing remote
        tracking of execution progress.

        Args:
            status: New status value (e.g., running, completed, failed).
            msg: Description of the status change or current state.

        Raises:
            DerivaMLException: If status update fails.

        Example:
            >>> execution.update_status(Status.running, "Processing sample 1 of 10")
        """
        self._status = status
        self._logger.info(msg)

        if self._dry_run:
            return

        # Delegate to ExecutionRecord for catalog updates
        if self._execution_record is not None:
            self._execution_record.update_status(status, msg)
        else:
            # Fallback for cases where ExecutionRecord isn't available
            self._ml_object.pathBuilder().schemas[self._ml_object.ml_schema].Execution.update(
                [
                    {
                        "RID": self.execution_rid,
                        "Status": status.value,
                        "Status_Detail": msg,
                    }
                ]
            )

    def execution_start(self) -> None:
        """Marks the execution as started.

        Records the start time and updates the execution's status to 'running'.
        This should be called before beginning the main execution work.

        Example:
            >>> execution.execution_start()
            >>> try:
            ...     # Run analysis
            ...     execution.execution_stop()
            ... except Exception:
            ...     execution.update_status(Status.failed, "Analysis error")
        """
        self.start_time = datetime.now()
        self.uploaded_assets = None
        self.update_status(Status.initializing, "Start execution  ...")

    def execution_stop(self) -> None:
        """Marks the execution as completed.

        Records the stop time and updates the execution's status to 'completed'.
        This should be called after all execution work is finished.

        Example:
            >>> try:
            ...     # Run analysis
            ...     execution.execution_stop()
            ... except Exception:
            ...     execution.update_status(Status.failed, "Analysis error")
        """
        self.stop_time = datetime.now()
        duration = self.stop_time - self.start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration = f"{round(hours, 0)}H {round(minutes, 0)}min {round(seconds, 4)}sec"

        self.update_status(Status.completed, "Algorithm execution ended.")
        if not self._dry_run:
            self._ml_object.pathBuilder().schemas[self._ml_object.ml_schema].Execution.update(
                [{"RID": self.execution_rid, "Duration": duration}]
            )

    def _build_upload_staging(self) -> Path:
        """Build ephemeral symlink tree from manifest for GenericUploader.

        Reads the manifest and creates symlinks from the flat assets/ directory
        into the regex-expected tree structure under asset/ that the
        GenericUploader needs for pattern matching.

        Returns:
            Path to the asset root (with staged symlinks) for upload.
        """
        manifest = self._get_manifest()
        pending = manifest.pending_assets()

        if not pending:
            # No manifest entries — fall back to old asset/ tree if it exists
            return self._asset_root

        staging_root = asset_root(self._working_dir, self.execution_rid)

        for key, entry in pending.items():
            # key is "{AssetTable}/{filename}"
            parts = key.split("/", 1)
            if len(parts) != 2:
                continue
            asset_table_name, filename = parts

            # Source file in flat storage
            flat_dir = flat_asset_dir(self._working_dir, self.execution_rid, asset_table_name)
            source = flat_dir / filename
            if not source.exists():
                self._logger.warning(f"Asset file not found: {source}")
                continue

            # Build metadata subdirectory path using ALL metadata columns
            # from the asset table (not just those in the manifest entry).
            # This must match the regex group order in asset_table_upload_spec()
            # which uses sorted(model.asset_metadata(asset_table)).
            # Missing metadata values get "None" (matching legacy asset_file_path).
            all_metadata_cols = sorted(self._model.asset_metadata(asset_table_name))
            metadata_parts = (
                [str(entry.metadata.get(k, "None")) for k in all_metadata_cols] if all_metadata_cols else []
            )
            target_dir = staging_root / entry.schema / asset_table_name
            for part in metadata_parts:
                target_dir = target_dir / part
            target_dir.mkdir(parents=True, exist_ok=True)

            target = target_dir / filename
            if not target.exists():
                try:
                    target.symlink_to(source.resolve())
                except (OSError, PermissionError):
                    import shutil

                    shutil.copy2(source, target)

        return staging_root

    def _cleanup_upload_staging(self) -> None:
        """Remove ephemeral symlinks created by _build_upload_staging."""
        staging_root = asset_root(self._working_dir, self.execution_rid)
        if staging_root.exists():
            import shutil

            shutil.rmtree(staging_root, ignore_errors=True)

    def _upload_execution_dirs(
        self,
        progress_callback: Callable[[UploadProgress], None] | None = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        timeout: tuple[int, int] | None = None,
        chunk_size: int | None = None,
    ) -> dict[str, list[AssetFilePath]]:
        """Upload execution assets using manifest-driven staging.

        Builds an ephemeral symlink tree from the manifest, uploads via
        GenericUploader, then updates the manifest with RIDs for uploaded assets.

        Args:
            progress_callback: Optional callback for upload progress updates.
            max_retries: Maximum retry attempts for failed uploads (default: 3).
            retry_delay: Initial delay between retries in seconds (default: 5.0).
            timeout: (connect_timeout, read_timeout) in seconds. Default (600, 600).
            chunk_size: Optional chunk size in bytes for hatrac uploads.

        Returns:
            dict mapping "{schema}/{table}" to list of AssetFilePath with RIDs.

        Raises:
            DerivaMLException: If upload fails.
        """
        # Build staging symlinks from manifest into the regex-expected tree
        upload_root = self._build_upload_staging()

        try:
            self.update_status(Status.running, "Uploading execution files...")
            results = upload_directory(
                self._model,
                upload_root,
                progress_callback=progress_callback,
                max_retries=max_retries,
                retry_delay=retry_delay,
                timeout=timeout,
                chunk_size=chunk_size,
            )
        except (RuntimeError, DerivaMLException) as e:
            error = format_exception(e)
            self.update_status(Status.failed, error)
            raise DerivaMLException(f"Failed to upload execution_assets: {error}")

        # Update manifest with upload results
        manifest = self._get_manifest()

        asset_map = {}
        for path, status in results.items():
            asset_table, file_name = normalize_asset_dir(path)

            # Find and update manifest entry
            table_name = asset_table.split("/")[1] if "/" in asset_table else asset_table
            manifest_key = f"{table_name}/{file_name}"
            rid = status.result["RID"]
            try:
                manifest.mark_uploaded(manifest_key, rid)
            except KeyError:
                pass  # File wasn't in manifest (legacy flow)

            asset_map.setdefault(asset_table, []).append(
                AssetFilePath(
                    asset_path=path,
                    asset_table=asset_table,
                    file_name=file_name,
                    asset_metadata={
                        k: v for k, v in status.result.items() if k in self._model.asset_metadata(table_name)
                    },
                    asset_types=[],
                    asset_rid=rid,
                )
            )
        self._update_asset_execution_table(asset_map)
        self.update_status(Status.running, "Updating features...")

        for p in self._feature_root.glob("**/*.jsonl"):
            m = is_feature_dir(p.parent)
            self._update_feature_table(
                target_table=m["target_table"],
                feature_name=m["feature_name"],
                feature_file=p,
                uploaded_files=asset_map,
            )

        self.update_status(Status.running, "Upload assets complete")
        return asset_map

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def download_asset(
        self, asset_rid: RID, dest_dir: Path, update_catalog: bool = True, use_cache: bool = False
    ) -> AssetFilePath:
        """Download an asset from a URL and place it in a local directory.

        Args:
            asset_rid: RID of the asset.
            dest_dir: Destination directory for the asset.
            update_catalog: Whether to update the catalog execution information after downloading.
            use_cache: If True, check the cache directory for a previously downloaded copy
                with a matching MD5 checksum before downloading. Cached copies are stored
                in ``cache_dir/assets/{rid}_{md5}/`` and symlinked into the destination.

        Returns:
            An AssetFilePath with the path to the downloaded (or cached) asset file.
        """

        asset_table = self._ml_object.resolve_rid(asset_rid).table
        if not self._model.is_asset(asset_table):
            raise DerivaMLException(f"RID {asset_rid}  is not for an asset table.")

        asset_record = self._ml_object.retrieve_rid(asset_rid)
        asset_metadata = {k: v for k, v in asset_record.items() if k in self._model.asset_metadata(asset_table)}
        asset_url = asset_record["URL"]
        asset_filename = dest_dir / asset_record["Filename"]

        # Check cache before downloading
        cache_hit = False
        if use_cache:
            md5 = asset_record.get("MD5")
            if md5:
                asset_cache_dir = self._ml_object.cache_dir / "assets"
                asset_cache_dir.mkdir(parents=True, exist_ok=True)
                cache_key = f"{asset_rid}_{md5}"
                cached_file = asset_cache_dir / cache_key / asset_record["Filename"]
                if cached_file.exists():
                    # Cache hit — symlink from cache to destination
                    self._logger.info(f"Using cached asset {asset_rid} (MD5: {md5})")
                    if asset_filename.exists() or asset_filename.is_symlink():
                        asset_filename.unlink()
                    asset_filename.symlink_to(cached_file)
                    cache_hit = True

        if not cache_hit:
            hs = HatracStore("https", self._ml_object.host_name, self._ml_object.credential)
            hs.get_obj(path=asset_url, destfilename=asset_filename.as_posix())

            # Store in cache for future use
            if use_cache:
                md5 = asset_record.get("MD5")
                if md5:
                    asset_cache_dir = self._ml_object.cache_dir / "assets"
                    asset_cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_key = f"{asset_rid}_{md5}"
                    cache_entry_dir = asset_cache_dir / cache_key
                    cache_entry_dir.mkdir(parents=True, exist_ok=True)
                    cached_file = cache_entry_dir / asset_record["Filename"]
                    # Move file to cache, then symlink back
                    shutil.move(str(asset_filename), str(cached_file))
                    asset_filename.symlink_to(cached_file)
                    self._logger.info(f"Cached asset {asset_rid} (MD5: {md5})")

        asset_type_table, _col_l, _col_r = self._model.find_association(asset_table, MLVocab.asset_type)
        type_path = self._ml_object.pathBuilder().schemas[asset_type_table.schema.name].tables[asset_type_table.name]
        asset_types = [
            asset_type[MLVocab.asset_type.value]
            for asset_type in type_path.filter(type_path.columns[asset_table.name] == asset_rid)
            .attributes(type_path.Asset_Type)
            .fetch()
        ]

        asset_path = AssetFilePath(
            file_name=asset_filename,
            asset_rid=asset_rid,
            asset_path=asset_filename,
            asset_metadata=asset_metadata,
            asset_table=asset_table.name,
            asset_types=asset_types,
        )

        if update_catalog:
            self._update_asset_execution_table(
                {f"{asset_table.schema.name}/{asset_table.name}": [asset_path]},
                asset_role="Input",
            )
        return asset_path

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def upload_assets(
        self,
        assets_dir: str | Path,
    ) -> dict[Any, FileUploadState] | None:
        """Uploads assets from a directory to the catalog.

        Scans the specified directory for assets and uploads them to the catalog,
        recording their metadata and types. Assets are organized by their types
        and associated with the execution.

        Args:
            assets_dir: Directory containing assets to upload.

        Returns:
            dict[Any, FileUploadState] | None: Mapping of assets to their upload states,
                or None if no assets were found.

        Raises:
            DerivaMLException: If upload fails or assets are invalid.

        Example:
            >>> states = execution.upload_assets("output/results")
            >>> for asset, state in states.items():
            ...     print(f"{asset}: {state}")
        """

        def path_to_asset(path: str) -> str:
            """Pull the asset name out of a path to that asset in the filesystem"""
            components = path.split("/")
            return components[components.index("asset") + 2]  # Look for asset in the path to find the name

        if not self._model.is_asset(Path(assets_dir).name):
            raise DerivaMLException("Directory does not have name of an asset table.")
        results = upload_directory(self._model, assets_dir)
        return {path_to_asset(p): r for p, r in results.items()}

    def upload_execution_outputs(
        self,
        clean_folder: bool | None = None,
        progress_callback: Callable[[UploadProgress], None] | None = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        timeout: tuple[int, int] | None = None,
        chunk_size: int | None = None,
    ) -> dict[str, list[AssetFilePath]]:
        """Uploads all outputs from the execution to the catalog.

        Scans the execution's output directories for assets, features, and other results,
        then uploads them to the catalog. Can optionally clean up the output folders
        after successful upload.

        IMPORTANT: This method must be called AFTER exiting the context manager, not inside it.
        The context manager handles execution timing (start/stop), while this method handles
        the separate upload step.

        Args:
            clean_folder: Whether to delete output folders after upload. If None (default),
                uses the DerivaML instance's clean_execution_dir setting. Pass True/False
                to override for this specific execution.
            progress_callback: Optional callback function to receive upload progress updates.
                Called with UploadProgress objects containing file name, bytes uploaded,
                total bytes, percent complete, phase, and status message.
            max_retries: Maximum number of retry attempts for failed uploads (default: 3).
            retry_delay: Initial delay in seconds between retries, doubles with each attempt (default: 5.0).
            timeout: Tuple of (connect_timeout, read_timeout) in seconds. Default is (600, 600).
                Note: urllib3 uses connect_timeout as the socket timeout during request body
                writes, so it must be large enough for a full chunk upload.
            chunk_size: Optional chunk size in bytes for hatrac uploads. If provided,
                large files will be uploaded in chunks of this size.

        Returns:
            dict[str, list[AssetFilePath]]: Mapping of asset types to their file paths.

        Raises:
            DerivaMLException: If upload fails or outputs are invalid.

        Example:
            >>> with ml.create_execution(config) as execution:
            ...     # Do ML work, register output files with asset_file_path()
            ...     path = execution.asset_file_path("Model", "model.pt")
            ...     # Write to path...
            ...
            >>> # Upload AFTER the context manager exits
            >>> def my_callback(progress):
            ...     print(f"Uploading {progress.file_name}: {progress.percent_complete:.1f}%")
            >>> outputs = execution.upload_execution_outputs(progress_callback=my_callback)
            >>>
            >>> # Upload large files with increased timeout (30 min per chunk)
            >>> outputs = execution.upload_execution_outputs(timeout=(6, 1800))
            >>>
            >>> # Override cleanup setting for this execution
            >>> outputs = execution.upload_execution_outputs(clean_folder=False)  # Keep files
        """
        if self._dry_run:
            return {}

        # Use DerivaML instance setting if not explicitly provided
        if clean_folder is None:
            clean_folder = getattr(self._ml_object, "clean_execution_dir", True)

        try:
            self.uploaded_assets = self._upload_execution_dirs(
                progress_callback=progress_callback,
                max_retries=max_retries,
                retry_delay=retry_delay,
                timeout=timeout,
                chunk_size=chunk_size,
            )
            self._set_asset_descriptions(self.uploaded_assets)
            self.update_status(Status.completed, "Successfully end the execution.")
            if clean_folder:
                self._clean_folder_contents(self._execution_root)
            return self.uploaded_assets
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error)
            raise e

    def _clean_folder_contents(self, folder_path: Path, remove_folder: bool = True):
        """Clean up folder contents and optionally the folder itself.

        Removes all files and subdirectories within the specified folder.
        Uses retry logic for Windows compatibility where files may be temporarily locked.

        Args:
            folder_path: Path to the folder to clean.
            remove_folder: If True (default), also remove the folder itself after
                cleaning its contents. If False, only remove contents.
        """
        MAX_RETRIES = 3
        RETRY_DELAY = 1  # seconds

        def remove_with_retry(path: Path, is_dir: bool = False) -> bool:
            for attempt in range(MAX_RETRIES):
                try:
                    if is_dir:
                        shutil.rmtree(path)
                    else:
                        Path(path).unlink()
                    return True
                except (OSError, PermissionError) as e:
                    if attempt == MAX_RETRIES - 1:
                        logging.warning(f"Failed to remove {path}: {e}")
                        return False
                    time.sleep(RETRY_DELAY)
            return False

        try:
            # First remove all contents
            with os.scandir(folder_path) as entries:
                for entry in entries:
                    if entry.is_dir() and not entry.is_symlink():
                        remove_with_retry(Path(entry.path), is_dir=True)
                    else:
                        remove_with_retry(Path(entry.path))

            # Then remove the folder itself if requested
            if remove_folder:
                remove_with_retry(folder_path, is_dir=True)

        except OSError as e:
            logging.warning(f"Failed to clean folder {folder_path}: {e}")

    def _update_feature_table(
        self,
        target_table: str,
        feature_name: str,
        feature_file: str | Path,
        uploaded_files: dict[str, list[AssetFilePath]],
    ) -> None:
        """Update the feature table with values from a JSONL file.

        Reads feature values from a file and inserts them into the catalog,
        replacing file paths with the RIDs of uploaded assets.

        Args:
            target_table: Name of the table the feature is defined on.
            feature_name: Name of the feature to update.
            feature_file: Path to JSONL file containing feature values.
            uploaded_files: Map from asset table names to their uploaded AssetFilePath objects.
        """

        # Get the column names of all the Feature columns that should be the RID of an asset
        asset_columns = [
            c.name for c in self._ml_object.feature_record_class(target_table, feature_name).feature.asset_columns
        ]

        # Get the names of the columns in the feature that are assets.
        asset_columns = [
            c.name for c in self._ml_object.feature_record_class(target_table, feature_name).feature.asset_columns
        ]

        feature_table = self._ml_object.feature_record_class(target_table, feature_name).feature.feature_table.name
        asset_map = {
            (asset_table, asset.file_name): asset.asset_rid
            for asset_table, assets in uploaded_files.items()
            for asset in assets
        }

        # Build a secondary lookup by (table_name, filename) to handle flat
        # asset paths (assets/{table}/file) written by the manifest-first
        # storage layout.  normalize_asset_dir only recognises the staging
        # path format (asset/{schema}/{table}/...), so flat paths return None.
        asset_map_by_table = {
            (asset_table.split("/")[1] if "/" in asset_table else asset_table, asset.file_name): asset.asset_rid
            for asset_table, assets in uploaded_files.items()
            for asset in assets
        }

        def map_path(e):
            """Go through the asset columns and replace the file name with the RID for the uploaded file."""
            for c in asset_columns:
                key = normalize_asset_dir(e[c])
                if key is not None:
                    e[c] = asset_map[key]
                else:
                    # Fall back to flat-path lookup: extract the table name
                    # and filename from the path.  Flat paths look like
                    # .../assets/{table}/{filename}
                    p = Path(e[c])
                    e[c] = asset_map_by_table[(p.parent.name, p.name)]
            return e

        # Load the JSON file that has the set of records that contain the feature values.
        with Path(feature_file).open("r") as feature_values:
            entities = [json.loads(line.strip()) for line in feature_values]
        # Update the asset columns in the feature and add to the catalog.
        self._ml_object.domain_path().tables[feature_table].insert(
            [map_path(e) for e in entities], on_conflict_skip=True
        )

    def _update_asset_execution_table(
        self,
        uploaded_assets: dict[str, list[AssetFilePath]],
        asset_role: str = "Output",
    ) -> None:
        """Add entry to the association table connecting an asset to an execution RID

        Args:
            uploaded_assets: Dictionary whose key is the name of an asset table and whose value is a list of RIDs for
                newly added assets to that table.
             asset_role: A term or list of terms from the Asset_Role vocabulary.
        """
        # Make sure the asset role is in the controlled vocabulary table.
        if self._dry_run:
            # Don't do any updates of we are doing a dry run.
            return
        self._ml_object.lookup_term(MLVocab.asset_role, asset_role)

        pb = self._ml_object.pathBuilder()
        for asset_table, asset_list in uploaded_assets.items():
            asset_table_name = asset_table.split("/")[1]  # Peel off the schema from the asset table
            asset_exe, asset_fk, execution_fk = self._model.find_association(asset_table_name, "Execution")
            asset_exe_path = pb.schemas[asset_exe.schema.name].tables[asset_exe.name]

            asset_exe_path.insert(
                [
                    {
                        asset_fk: asset_path.asset_rid,
                        execution_fk: self.execution_rid,
                        "Asset_Role": asset_role,
                    }
                    for asset_path in asset_list
                ],
                on_conflict_skip=True,
            )

            # Now add in the type names via the asset_asset_type association table.
            # Get the list of types for each file in the asset.
            if asset_role == "Input":
                return
            asset_type_map = {}
            with Path(
                asset_type_path(
                    self._working_dir,
                    self.execution_rid,
                    self._model.name_to_table(asset_table_name),
                )
            ).open("r") as asset_type_file:
                for line in asset_type_file:
                    asset_type_map.update(json.loads(line.strip()))
            for asset_path in asset_list:
                asset_path.asset_types = asset_type_map[asset_path.file_name]

            asset_asset_type, _, _ = self._model.find_association(asset_table_name, "Asset_Type")
            type_path = pb.schemas[asset_asset_type.schema.name].tables[asset_asset_type.name]

            type_path.insert(
                [
                    {asset_table_name: asset.asset_rid, "Asset_Type": t}
                    for asset in asset_list
                    for t in asset_type_map[asset.file_name]
                ],
                on_conflict_skip=True,
            )

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def asset_file_path(
        self,
        asset_name: str,
        file_name: str | Path,
        asset_types: list[str] | str | None = None,
        copy_file=False,
        rename_file: str | None = None,
        metadata=None,
        description: str | None = None,
        **kwargs,
    ) -> AssetFilePath:
        """Register a file for upload and return a path to write to.

        This routine has three modes depending on whether file_name refers to an existing file:
        1. **New file**: file_name doesn't exist — returns a path to write to.
        2. **Symlink**: file_name exists, copy_file=False — symlinks into staging.
        3. **Copy**: file_name exists, copy_file=True — copies into staging.

        Files are stored in a flat per-table directory (``assets/{AssetTable}/``).
        Metadata is tracked in a persistent JSON manifest for crash safety.
        Metadata can be set at registration time via the ``metadata`` parameter
        (an AssetRecord or dict) or incrementally after via the returned
        AssetFilePath's ``metadata`` property.

        Args:
            asset_name: Name of the asset table. Must be a valid asset table.
            file_name: Name of file to be uploaded, or path to an existing file.
            asset_types: Asset type terms from Asset_Type vocabulary. Defaults to asset_name.
            copy_file: Whether to copy the file rather than creating a symbolic link.
            rename_file: If provided, rename the file during staging.
            metadata: An AssetRecord instance or dict of metadata column values.
            description: Optional description for the asset record.
            **kwargs: Additional metadata values (legacy support, merged with metadata).

        Returns:
            AssetFilePath bound to the manifest for write-through metadata updates.

        Raises:
            DerivaMLException: If the asset table doesn't exist.
            DerivaMLValidationError: If asset_types contains invalid terms.
        """
        if not self._model.is_asset(asset_name):
            raise DerivaMLException(f"Table {asset_name} is not an asset")

        asset_table = self._model.name_to_table(asset_name)
        schema = asset_table.schema.name

        # Validate and normalize asset types
        asset_types = asset_types or kwargs.pop("Asset_Type", None) or asset_name
        asset_types = [asset_types] if isinstance(asset_types, str) else asset_types
        for t in asset_types:
            self._ml_object.lookup_term(MLVocab.asset_type, t)

        # Resolve metadata from AssetRecord, dict, or kwargs
        metadata_dict: dict[str, Any] = {}
        if metadata is not None:
            if hasattr(metadata, "model_dump"):
                metadata_dict = {k: v for k, v in metadata.model_dump().items() if v is not None}
            else:
                metadata_dict = dict(metadata)
        # Merge any kwargs that aren't standard parameters
        metadata_dict.update(kwargs)

        # Determine file name and path
        file_name = Path(file_name)
        if file_name.name == "_implementations.log":
            file_name = file_name.with_name("-implementations.log")

        if not file_name.is_absolute():
            file_name = file_name.resolve()

        target_name = Path(rename_file) if file_name.exists() and rename_file else file_name

        # Store file in flat per-table directory
        flat_dir = flat_asset_dir(self._working_dir, self.execution_rid, asset_name)
        flat_path = flat_dir / target_name.name

        if file_name.exists():
            if copy_file:
                flat_path.write_bytes(file_name.read_bytes())
            else:
                try:
                    flat_path.symlink_to(file_name)
                except (OSError, PermissionError):
                    flat_path.write_bytes(file_name.read_bytes())

        # Register in manifest (write-through + fsync)
        manifest = self._get_manifest()
        manifest_key = f"{asset_name}/{target_name.name}"
        manifest.add_asset(
            manifest_key,
            AssetEntry(
                asset_table=asset_name,
                schema=schema,
                asset_types=asset_types,
                metadata=metadata_dict,
                description=description,
            ),
        )

        # Also write legacy asset-type JSONL for backward compatibility with upload
        with Path(asset_type_path(self._working_dir, self.execution_rid, asset_table)).open("a") as f:
            f.write(json.dumps({target_name.name: asset_types}) + "\n")

        result = AssetFilePath(
            asset_path=flat_path,
            asset_table=asset_name,
            file_name=target_name.name,
            asset_metadata=metadata_dict,
            asset_types=asset_types,
        )
        result._bind_manifest(manifest, manifest_key)
        return result

    def _get_manifest(self) -> AssetManifest:
        """Get or create the asset manifest for this execution."""
        if not hasattr(self, "_manifest") or self._manifest is None:
            ws = self._ml_object.workspace
            self._manifest = AssetManifest(ws.manifest_store(), self.execution_rid)
        return self._manifest

    def table_path(self, table: str) -> Path:
        """Return a local file path to a CSV to add values to a table on upload.

        Args:
            table: Name of table to be uploaded.

        Returns:
            Pathlib path to the file in which to place table values.
        """
        # Find which domain schema contains this table
        table_schema = None
        for domain_schema in self._ml_object.domain_schemas:
            if domain_schema in self._model.schemas:
                if table in self._model.schemas[domain_schema].tables:
                    table_schema = domain_schema
                    break

        if table_schema is None:
            raise DerivaMLException("Table '{}' not found in any domain schema".format(table))

        return table_path(self._working_dir, schema=table_schema, table=table)

    def execute(self) -> Execution:
        """Initiate an execution with the provided configuration. Can be used in a context manager."""
        self.execution_start()
        return self

    @validate_call
    def add_features(self, features: Iterable[FeatureRecord]) -> None:
        """Adds feature records to the catalog.

        Associates feature records with this execution and uploads them to the catalog.
        Features represent measurable properties or characteristics of records.

        NOTE: The catalog is not updated until upload_execution_outputs() is called.

        Args:
            features: Feature records to add, each containing a value and metadata.

        Raises:
            DerivaMLException: If feature addition fails or features are invalid.

        Example:
            >>> feature = FeatureRecord(value="high", confidence=0.95)
            >>> execution.add_features([feature])
        """

        # Make sure feature list is homogeneous:
        sorted_features = defaultdict(list)
        for f in features:
            sorted_features[type(f)].append(f)
        for fs in sorted_features.values():
            self._add_features(fs)

    def _add_features(self, features: list[FeatureRecord]) -> None:
        # Update feature records to include current execution_rid
        first_row = features[0]
        feature = first_row.feature
        # Use the schema from the feature table
        feature_schema = feature.feature_table.schema.name
        json_path = feature_value_path(
            self._working_dir,
            schema=feature_schema,
            target_table=feature.target_table.name,
            feature_name=feature.feature_name,
            exec_rid=self.execution_rid,
        )
        with Path(json_path).open("a", encoding="utf-8") as file:
            for feature in features:
                feature.Execution = self.execution_rid
                file.write(json.dumps(feature.model_dump(mode="json", exclude={"RCT"})) + "\n")

    def list_input_datasets(self) -> list[Dataset]:
        """List all datasets that were inputs to this execution.

        Returns:
            List of Dataset objects that were used as inputs.

        Example:
            >>> for ds in execution.list_input_datasets():
            ...     print(f"Input: {ds.dataset_rid} - {ds.description}")
        """
        if self._execution_record is not None:
            return self._execution_record.list_input_datasets()

        # Fallback for dry_run mode
        pb = self._ml_object.pathBuilder()
        dataset_exec = pb.schemas[self._ml_object.ml_schema].Dataset_Execution

        records = list(dataset_exec.filter(dataset_exec.Execution == self.execution_rid).entities().fetch())

        return [self._ml_object.lookup_dataset(r["Dataset"]) for r in records]

    def list_assets(self, asset_role: str | None = None) -> list["Asset"]:
        """List all assets that were inputs or outputs of this execution.

        Args:
            asset_role: Optional filter: "Input" or "Output". If None, returns all.

        Returns:
            List of Asset objects associated with this execution.

        Example:
            >>> inputs = execution.list_assets(asset_role="Input")
            >>> outputs = execution.list_assets(asset_role="Output")
        """
        if self._execution_record is not None:
            return self._execution_record.list_assets(asset_role=asset_role)

        # Fallback for dry_run mode

        pb = self._ml_object.pathBuilder()
        asset_exec = pb.schemas[self._ml_object.ml_schema].Execution_Asset_Execution

        query = asset_exec.filter(asset_exec.Execution == self.execution_rid)
        if asset_role:
            query = query.filter(asset_exec.Asset_Role == asset_role)

        records = list(query.entities().fetch())

        assets = []
        for r in records:
            try:
                asset = self._ml_object.lookup_asset(r["Execution_Asset"])
                assets.append(asset)
            except Exception:
                pass  # Skip assets that can't be looked up
        return assets

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def create_dataset(
        self,
        dataset_types: str | list[str] | None = None,
        version: DatasetVersion | str | None = None,
        description: str = "",
    ) -> Dataset:
        """Create a new dataset with specified types.

        Creates a dataset associated with this execution for provenance tracking.

        Args:
            dataset_types: One or more dataset type terms from Dataset_Type vocabulary.
            description: Markdown description of the dataset being created.
            version: Dataset version. Defaults to 0.1.0.

        Returns:
            The newly created Dataset.
        """
        return Dataset.create_dataset(
            ml_instance=self._ml_object,
            execution_rid=self.execution_rid,
            dataset_types=dataset_types,
            version=version,
            description=description,
        )

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def add_files(
        self,
        files: Iterable[FileSpec],
        dataset_types: str | list[str] | None = None,
        description: str = "",
    ) -> "Dataset":
        """Adds files to the catalog with their metadata.

        Registers files in the catalog along with their metadata (MD5, length, URL) and associates them with
        specified file types.

        Args:
            files: File specifications containing MD5 checksum, length, and URL.
            dataset_types: One or more dataset type terms from File_Type vocabulary.
            description: Description of the files.

        Returns:
            RID: Dataset  that identifies newly added files. Will be nested to mirror original directory structure
            of the files.

        Raises:
            DerivaMLInvalidTerm: If file_types are invalid or execution_rid is not an execution record.

        Examples:
            Add a single file type:
                >>> files = [FileSpec(url="path/to/file.txt", md5="abc123", length=1000)]
                >>> rids = exe.add_files(files, file_types="text")

            Add multiple file types:
                >>> rids = exe.add_files(
                ...     files=[FileSpec(url="image.png", md5="def456", length=2000)],
                ...     file_types=["image", "png"],
                ... )
        """
        return self._ml_object.add_files(
            files=files,
            execution_rid=self.execution_rid,
            dataset_types=dataset_types,
            description=description,
        )

    # =========================================================================
    # Execution Nesting Methods
    # =========================================================================

    def add_nested_execution(
        self,
        nested_execution: "Execution | ExecutionRecord | RID",
        sequence: int | None = None,
    ) -> None:
        """Add a nested (child) execution to this execution.

        Creates a parent-child relationship between this execution and another.
        This is useful for grouping related executions, such as parameter sweeps
        or pipeline stages.

        Args:
            nested_execution: The child execution to add (Execution, ExecutionRecord, or RID).
            sequence: Optional ordering index (0, 1, 2...). Use None for parallel executions.

        Raises:
            DerivaMLException: If the association cannot be created.

        Example:
            >>> parent_exec = ml.create_execution(parent_config)
            >>> child_exec = ml.create_execution(child_config)
            >>> parent_exec.add_nested_execution(child_exec, sequence=0)
        """
        if self._dry_run:
            return

        # Get the RID from the nested execution
        if isinstance(nested_execution, Execution):
            nested_rid = nested_execution.execution_rid
        elif isinstance(nested_execution, ExecutionRecord):
            nested_rid = nested_execution.execution_rid
        else:
            nested_rid = nested_execution

        # Delegate to ExecutionRecord if available
        if self._execution_record is not None:
            self._execution_record.add_nested_execution(nested_rid, sequence=sequence)
        else:
            # Fallback for cases without execution record
            pb = self._ml_object.pathBuilder()
            execution_execution = pb.schemas[self._ml_object.ml_schema].Execution_Execution

            record = {
                "Execution": self.execution_rid,
                "Nested_Execution": nested_rid,
            }
            if sequence is not None:
                record["Sequence"] = sequence

            execution_execution.insert([record])

    def list_nested_executions(
        self,
        recurse: bool = False,
        _visited: set[RID] | None = None,
    ) -> list["ExecutionRecord"]:
        """List all nested (child) executions of this execution.

        Args:
            recurse: If True, recursively return all descendant executions.
            _visited: Internal parameter to track visited executions and prevent infinite recursion.

        Returns:
            List of nested ExecutionRecord objects, ordered by sequence if available.
            To get full Execution objects with lifecycle management, use restore_execution().

        Example:
            >>> children = parent_exec.list_nested_executions()
            >>> all_descendants = parent_exec.list_nested_executions(recurse=True)
        """
        if self._execution_record is not None:
            return list(self._execution_record.list_nested_executions(recurse=recurse, _visited=_visited))

        # Fallback for dry_run mode
        if _visited is None:
            _visited = set()

        if self.execution_rid in _visited:
            return []
        _visited.add(self.execution_rid)

        pb = self._ml_object.pathBuilder()
        execution_execution = pb.schemas[self._ml_object.ml_schema].Execution_Execution

        # Query for nested executions, ordered by sequence
        nested = list(
            execution_execution.filter(execution_execution.Execution == self.execution_rid).entities().fetch()
        )

        # Sort by sequence (None values at the end)
        nested.sort(key=lambda x: (x.get("Sequence") is None, x.get("Sequence")))

        children = []
        for record in nested:
            child = self._ml_object.lookup_execution(record["Nested_Execution"])
            children.append(child)
            if recurse:
                children.extend(child.list_nested_executions(recurse=True, _visited=_visited))

        return children

    def list_parent_executions(
        self,
        recurse: bool = False,
        _visited: set[RID] | None = None,
    ) -> list["ExecutionRecord"]:
        """List all parent executions that contain this execution as a nested child.

        Args:
            recurse: If True, recursively return all ancestor executions.
            _visited: Internal parameter to track visited executions and prevent infinite recursion.

        Returns:
            List of parent ExecutionRecord objects.
            To get full Execution objects with lifecycle management, use restore_execution().

        Example:
            >>> parents = child_exec.list_parent_executions()
            >>> all_ancestors = child_exec.list_parent_executions(recurse=True)
        """
        if self._execution_record is not None:
            return list(self._execution_record.list_parent_executions(recurse=recurse, _visited=_visited))

        # Fallback for dry_run mode
        if _visited is None:
            _visited = set()

        if self.execution_rid in _visited:
            return []
        _visited.add(self.execution_rid)

        pb = self._ml_object.pathBuilder()
        execution_execution = pb.schemas[self._ml_object.ml_schema].Execution_Execution

        parent_records = list(
            execution_execution.filter(execution_execution.Nested_Execution == self.execution_rid).entities().fetch()
        )

        parents = []
        for record in parent_records:
            parent = self._ml_object.lookup_execution(record["Execution"])
            parents.append(parent)
            if recurse:
                parents.extend(parent.list_parent_executions(recurse=True, _visited=_visited))

        return parents

    def is_nested(self) -> bool:
        """Check if this execution is nested within another execution.

        Returns:
            True if this execution has at least one parent execution.
        """
        if self._execution_record is not None:
            return self._execution_record.is_nested()
        return len(self.list_parent_executions()) > 0

    def is_parent(self) -> bool:
        """Check if this execution has nested child executions.

        Returns:
            True if this execution has at least one nested execution.
        """
        if self._execution_record is not None:
            return self._execution_record.is_parent()
        return len(self.list_nested_executions()) > 0

    def __str__(self):
        items = [
            f"caching_dir: {self._cache_dir}",
            f"_working_dir: {self._working_dir}",
            f"execution_rid: {self.execution_rid}",
            f"workflow_rid: {self.workflow_rid}",
            f"asset_paths: {self.asset_paths}",
            f"configuration: {self.configuration}",
        ]
        return "\n".join(items)

    def __enter__(self):
        """
        Method invoked when entering the context.

        Returns:
        - self: The instance itself.

        """
        self.execution_start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> bool:
        """
        Method invoked when exiting the context.

        Args:
           exc_type: Exception type.
           exc_value: Exception value.
           exc_tb: Exception traceback.

        Returns:
           bool: True if execution completed successfully, False otherwise.
        """
        if not exc_type:
            self.update_status(Status.running, "Successfully run Ml.")
            self.execution_stop()
            return True
        else:
            self.update_status(
                Status.failed,
                f"Exception type: {exc_type}, Exception value: {exc_value}",
            )
            logging.error(f"Exception type: {exc_type}, Exception value: {exc_value}, Exception traceback: {exc_tb}")
            return False
