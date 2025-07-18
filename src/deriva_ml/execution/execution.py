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
    ...     execution.upload_execution_outputs()
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List

from deriva.core import format_exception
from deriva.core.hatrac_store import HatracStore
from pydantic import ConfigDict, validate_call

from deriva_ml.core.base import DerivaML
from deriva_ml.core.definitions import (
    DRY_RUN_RID,
    RID,
    ExecAssetType,
    ExecMetadataType,
    FileSpec,
    FileUploadState,
    MLAsset,
    MLVocab,
    Status,
)
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.aux_classes import DatasetSpec, DatasetVersion, VersionPart
from deriva_ml.dataset.dataset_bag import DatasetBag
from deriva_ml.dataset.upload import (
    asset_file_path,
    asset_root,
    asset_type_path,
    execution_root,
    feature_root,
    feature_value_path,
    is_feature_dir,
    normalize_asset_dir,
    table_path,
    upload_directory,
)
from deriva_ml.execution.environment import get_execution_environment
from deriva_ml.execution.execution_configuration import ExecutionConfiguration
from deriva_ml.execution.workflow import Workflow
from deriva_ml.feature import FeatureRecord

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


# Platform-specific base class
if sys.version_info >= (3, 12):

    class AssetFilePath(Path):
        """Extended Path class for managing asset files.

        Represents a file path with additional metadata about its role as an asset in the catalog.
        This class extends the standard Path class to include information about the asset's
        catalog representation and type.

        Attributes:
            asset_name (str): Name of the asset in the catalog (e.g., asset table name).
            file_name (str): Name of the local file containing the asset.
            asset_metadata (dict[str, Any]): Additional columns beyond URL, Length, and checksum.
            asset_types (list[str]): Terms from the Asset_Type controlled vocabulary.
            asset_rid (RID | None): Resource Identifier if uploaded to an asset table.

        Example:
            >>> path = AssetFilePath(
            ...     "/path/to/file.txt",
            ...     asset_name="analysis_output",
            ...     file_name="results.txt",
            ...     asset_metadata={"version": "1.0"},
            ...     asset_types=["text", "results"]
            ... )
        """

        def __init__(
            self,
            asset_path: str | Path,
            asset_name: str,
            file_name: str,
            asset_metadata: dict[str, Any],
            asset_types: list[str] | str,
            asset_rid: RID | None = None,
        ):
            """Initializes an AssetFilePath instance.

            Args:
                asset_path: Local path to the asset file.
                asset_name: Name of the asset in the catalog.
                file_name: Name of the local file.
                asset_metadata: Additional metadata columns.
                asset_types: One or more asset type terms.
                asset_rid: Optional Resource Identifier if already in catalog.
            """
            super().__init__(asset_path)
            self.asset_name = asset_name
            self.file_name = file_name
            self.asset_metadata = asset_metadata
            self.asset_types = asset_types if isinstance(asset_types, list) else [asset_types]
            self.asset_rid = asset_rid
else:

    class AssetFilePath(type(Path())):
        """
        Create a new Path object that has additional information related to the use of this path as an asset.

        Attrubytes:
            asset_path: Local path to the location of the asset.
            asset_name:  The name of the asset in the catalog (e.g., the asset table name).
            file_name:  Name of the local file that contains the contents of the asset.
            asset_metadata: Any additional columns associated with this asset beyond the URL, Length, and checksum.
            asset_types:  A list of terms from the Asset_Type controlled vocabulary.
            asset_rid:  The RID of the asset if it has been uploaded into an asset table
        """

        def __new__(
            cls,
            asset_path: str | Path,
            asset_name: str,
            file_name: str,
            asset_metadata: dict[str, Any],
            asset_types: list[str] | str,
            asset_rid: RID | None = None,
        ):
            # Only pass the path to the base Path class
            obj = super().__new__(cls, asset_path)
            obj.asset_name = asset_name
            obj.file_name = file_name
            obj.asset_metadata = asset_metadata
            obj.asset_types = asset_types if isinstance(asset_types, list) else [asset_types]
            obj.asset_rid = asset_rid
            return obj


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
        parameters (dict): Execution parameters.
        start_time (datetime | None): When execution started.
        stop_time (datetime | None): When execution completed.

    Example:
        >>> config = ExecutionConfiguration(
        ...     workflow="analysis",
        ...     description="Process samples",
        ...     parameters={"threshold": 0.5}
        ... )
        >>> with ml.create_execution(config) as execution:
        ...     execution.download_dataset_bag(dataset_spec)
        ...     # Run analysis
        ...     execution.upload_execution_outputs()
    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        configuration: ExecutionConfiguration,
        ml_object: DerivaML,
        reload: RID | None = None,
        dry_run: bool = False,
    ):
        """Initializes an Execution instance.

        Creates a new execution or reloads an existing one. Initializes the execution
        environment, downloads required datasets, and sets up asset tracking.

        Args:
            configuration: Settings and parameters for the execution.
            ml_object: DerivaML instance managing the execution.
            reload: Optional RID of existing execution to reload.
            dry_run: If True, don't create catalog records or upload results.

        Raises:
            DerivaMLException: If initialization fails or configuration is invalid.
        """
        self.asset_paths: list[AssetFilePath] = []
        self.configuration = configuration
        self._ml_object = ml_object
        self._model = ml_object.model
        self._logger = ml_object._logger
        self.start_time = None
        self.stop_time = None
        self.status = Status.created
        self.uploaded_assets: dict[str, list[AssetFilePath]] | None = None
        self.configuration.argv = sys.argv

        self.dataset_rids: List[RID] = []
        self.datasets: list[DatasetBag] = []
        self.parameters = self.configuration.parameters

        self._working_dir = self._ml_object.working_dir
        self._cache_dir = self._ml_object.cache_dir
        self._dry_run = dry_run

        # Make sure we have a good workflow.
        if isinstance(self.configuration.workflow, Workflow):
            self.workflow_rid = (
                self._ml_object.add_workflow(self.configuration.workflow) if not self._dry_run else DRY_RUN_RID
            )
        else:
            self.workflow_rid = self.configuration.workflow
            if self._ml_object.resolve_rid(configuration.workflow).table.name != "Workflow":
                raise DerivaMLException("Workflow specified in execution configuration is not a Workflow")

        # Validate the datasets and assets to be valid.
        for d in self.configuration.datasets:
            if self._ml_object.resolve_rid(d.rid).table.name != "Dataset":
                raise DerivaMLException("Dataset specified in execution configuration is not a dataset")

        for a in self.configuration.assets:
            if not self._model.is_asset(self._ml_object.resolve_rid(a).table.name):
                raise DerivaMLException("Asset specified in execution configuration is not a asset table")

        schema_path = self._ml_object.pathBuilder.schemas[self._ml_object.ml_schema]
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

        if isinstance(self.configuration.workflow, Workflow) and self.configuration.workflow.is_notebook:
            # Put execution_rid into the cell output so we can find it later.
            display(Markdown(f"Execution RID: {self._ml_object.cite(self.execution_rid)}"))

        # Create a directory for execution rid so we can recover the state in case of a crash.
        execution_root(prefix=self._ml_object.working_dir, exec_rid=self.execution_rid)
        self._initialize_execution(reload)

    def _save_runtime_environment(self):
        runtime_env_path = self.asset_file_path(
            "Execution_Metadata",
            f"environment_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            ExecMetadataType.runtime_env.value,
        )
        with Path(runtime_env_path).open("w") as fp:
            json.dump(get_execution_environment(), fp)

    def _initialize_execution(self, reload: RID | None = None) -> None:
        """Initialize the execution by a configuration in the Execution_Metadata table.
        Set up a working directory and download all the assets and data.

        :raise DerivaMLException: If there is an issue initializing the execution.

        Args:
            reload: RID of previously initialized execution.

        Returns:

        """
        # Materialize bdbag
        for dataset in self.configuration.datasets:
            self.update_status(Status.initializing, f"Materialize bag {dataset.rid}... ")
            self.datasets.append(self.download_dataset_bag(dataset))
            self.dataset_rids.append(dataset.rid)

        # Update execution info
        schema_path = self._ml_object.pathBuilder.schemas[self._ml_object.ml_schema]
        if self.dataset_rids and not (reload or self._dry_run):
            schema_path.Dataset_Execution.insert(
                [{"Dataset": d, "Execution": self.execution_rid} for d in self.dataset_rids]
            )

        # Download assets....
        self.update_status(Status.running, "Downloading assets ...")
        self.asset_paths = {}
        for asset_rid in self.configuration.assets:
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
                )
            )

        # Save configuration details for later upload
        if not reload:
            cfile = self.asset_file_path(
                MLAsset.execution_metadata,
                "configuration.json",
                ExecMetadataType.execution_config.value,
            )
            with Path(cfile).open("w", encoding="utf-8") as config_file:
                json.dump(self.configuration.model_dump(), config_file)

            for parameter_file in self.configuration.parameters:
                self.asset_file_path(
                    MLAsset.execution_asset,
                    parameter_file,
                    ExecAssetType.input_file.value,
                )

            # save runtime env
            self._save_runtime_environment()

        self.start_time = datetime.now()
        self.update_status(Status.pending, "Initialize status finished.")

    @property
    def _execution_root(self) -> Path:
        """

        Args:

        Returns:
          :return:

        """
        return execution_root(self._working_dir, self.execution_rid)

    @property
    def _feature_root(self) -> Path:
        """The root path to all execution-specific files.
        :return:

        Args:

        Returns:

        """
        return feature_root(self._working_dir, self.execution_rid)

    @property
    def _asset_root(self) -> Path:
        """The root path to all execution-specific files.
        :return:

        Args:

        Returns:

        """
        return asset_root(self._working_dir, self.execution_rid)

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
        return self._ml_object.download_dataset_bag(dataset, execution_rid=self.execution_rid)

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
        self.status = status
        self._logger.info(msg)

        if self._dry_run:
            return

        self._ml_object.pathBuilder.schemas[self._ml_object.ml_schema].Execution.update(
            [
                {
                    "RID": self.execution_rid,
                    "Status": self.status.value,
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
            self._ml_object.pathBuilder.schemas[self._ml_object.ml_schema].Execution.update(
                [{"RID": self.execution_rid, "Duration": duration}]
            )

    def _upload_execution_dirs(self) -> dict[str, list[AssetFilePath]]:
        """Upload execution assets at _working_dir/Execution_asset.

        This routine uploads the contents of the
        Execution_Asset directory and then updates the execution_asset table in the ML schema to have references
        to these newly uploaded files.

        Returns:
          dict: Results of the upload operation.

        Raises:
          DerivaMLException: If there is an issue when uploading the assets.
        """

        try:
            self.update_status(Status.running, "Uploading execution files...")
            results = upload_directory(self._model, self._asset_root)
        except RuntimeError as e:
            error = format_exception(e)
            self.update_status(Status.failed, error)
            raise DerivaMLException(f"Fail to upload execution_assets. Error: {error}")

        asset_map = {}
        for path, status in results.items():
            asset_table, file_name = normalize_asset_dir(path)

            asset_map.setdefault(asset_table, []).append(
                AssetFilePath(
                    asset_path=path,
                    asset_name=asset_table,
                    file_name=file_name,
                    asset_metadata={
                        k: v
                        for k, v in status.result.items()
                        if k in self._model.asset_metadata(asset_table.split("/")[1])
                    },
                    asset_types=[],
                    asset_rid=status.result["RID"],
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
    def download_asset(self, asset_rid: RID, dest_dir: Path, update_catalog=True) -> AssetFilePath:
        """Download an asset from a URL and place it in a local directory.

        Args:
            asset_rid: RID of the asset.
            dest_dir: Destination directory for the asset.
            update_catalog: Whether to update the catalog execution information after downloading.

        Returns:
            A tuple with the name of the asset table and a Path object to the downloaded asset.
        """

        asset_table = self._ml_object.resolve_rid(asset_rid).table
        if not self._model.is_asset(asset_table):
            raise DerivaMLException(f"RID {asset_rid}  is not for an asset table.")

        asset_record = self._ml_object.retrieve_rid(asset_rid)
        asset_metadata = {k: v for k, v in asset_record.items() if k in self._model.asset_metadata(asset_table)}
        asset_url = asset_record["URL"]
        asset_filename = dest_dir / asset_record["Filename"]
        hs = HatracStore("https", self._ml_object.host_name, self._ml_object.credential)
        hs.get_obj(path=asset_url, destfilename=asset_filename.as_posix())

        asset_type_table, _col_l, _col_r = self._model.find_association(asset_table, MLVocab.asset_type)
        type_path = self._ml_object.pathBuilder.schemas[asset_type_table.schema.name].tables[asset_type_table.name]
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
            asset_name=asset_table.name,
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

    def upload_execution_outputs(self, clean_folder: bool = True) -> dict[str, list[AssetFilePath]]:
        """Uploads all outputs from the execution to the catalog.

        Scans the execution's output directories for assets, features, and other results,
        then uploads them to the catalog. Can optionally clean up the output folders
        after successful upload.

        Args:
            clean_folder: Whether to delete output folders after upload. Defaults to True.

        Returns:
            dict[str, list[AssetFilePath]]: Mapping of asset types to their file paths.

        Raises:
            DerivaMLException: If upload fails or outputs are invalid.

        Example:
            >>> outputs = execution.upload_execution_outputs()
            >>> for type_name, paths in outputs.items():
            ...     print(f"{type_name}: {len(paths)} files")
        """
        if self._dry_run:
            return {}
        try:
            self.uploaded_assets = self._upload_execution_dirs()
            self.update_status(Status.completed, "Successfully end the execution.")
            if clean_folder:
                self._clean_folder_contents(self._execution_root)
            return self.uploaded_assets
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error)
            raise e

    def _clean_folder_contents(self, folder_path: Path):
        """Clean up folder contents with Windows-compatible error handling.

        Args:
            folder_path: Path to the folder to clean
        """
        import time

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
                        self.update_status(Status.failed, format_exception(e))
                        return False
                    time.sleep(RETRY_DELAY)
            return False

        try:
            with os.scandir(folder_path) as entries:
                for entry in entries:
                    if entry.is_dir() and not entry.is_symlink():
                        remove_with_retry(Path(entry.path), is_dir=True)
                    else:
                        remove_with_retry(Path(entry.path))
        except OSError as e:
            self.update_status(Status.failed, format_exception(e))

    def _update_feature_table(
        self,
        target_table: str,
        feature_name: str,
        feature_file: str | Path,
        uploaded_files: dict[str, list[AssetFilePath]],
    ) -> None:
        """

        Args:
            target_table: str:
            feature_name: str:
            feature_file: str | Path:
            uploaded_files: Dictionary whose key is an asset name, file-name pair, and whose value is a filename,
                RID of that asset.
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

        def map_path(e):
            """Go through the asset columns and replace the file name with the RID for the uploaded file."""
            for c in asset_columns:
                e[c] = asset_map[normalize_asset_dir(e[c])]
            return e

        # Load the JSON file that has the set of records that contain the feature values.
        with Path(feature_file).open("r") as feature_values:
            entities = [json.loads(line.strip()) for line in feature_values]
        # Update the asset columns in the feature and add to the catalog.
        self._ml_object.domain_path.tables[feature_table].insert([map_path(e) for e in entities], on_conflict_skip=True)

    def _update_asset_execution_table(
        self,
        uploaded_assets: dict[str, list[AssetFilePath]],
        asset_role: str = "Output",
    ):
        """Add entry to the association table connecting an asset to an execution RID

        Args:
            uploaded_assets: Dictionary whose key is the name of an asset table and whose value is a list of RIDs for
                newly added assets to that table.
             asset_role: A term or list of terms from the Asset_Role vocabulary.
        """
        # Make sure the asset role is in the controlled vocabulary table.
        self._ml_object.lookup_term(MLVocab.asset_role, asset_role)

        pb = self._ml_object.pathBuilder
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
        **kwargs,
    ) -> AssetFilePath:
        """Return a pathlib Path to the directory in which to place files for the specified execution_asset type.

        Given the name of an asset table, and a file name, register the file for upload and return a path to that
        file in the upload directory.  In addition to the filename, additional asset metadata and file asset types may
        be specified.

        This routine has three modes, depending on if file_name refers to an existing file.  If it doesn't, a path
        to a new file with the specified name is returned.  The caller can then open that file for writing.

        If the provided filename refers to an existing file and the copy_file argument is False (the default), then the
        returned path contains a symbolic link to that file.  If the copy_file argument is True, then the contents of
        file_name are copied into the target directory.

        Args:
            asset_name: Type of asset to be uploaded.  Must be a term in Asset_Type controlled vocabulary.
            file_name: Name of file to be uploaded.
            asset_types: Type of asset to be uploaded.  Defaults to the name of the asset.
            **kwargs: Any additional metadata values that may be part of the asset table.

        Returns:
            Path in which to place asset files.

        Raises:
            DerivaException: If the asset type is not defined.
        """
        if not self._model.is_asset(asset_name):
            DerivaMLException(f"Table {asset_name} is not an asset")

        asset_table = self._model.name_to_table(asset_name)

        asset_types = asset_types or kwargs.get("Asset_Type", None) or asset_name
        asset_types = [asset_types] if isinstance(asset_types, str) else asset_types
        for t in asset_types:
            self._ml_object.lookup_term(MLVocab.asset_type, t)

        file_name = Path(file_name)
        asset_path = asset_file_path(
            self._working_dir,
            self.execution_rid,
            self._model.name_to_table(asset_name),
            file_name.name,
            metadata=kwargs,
        )

        if file_name.exists():
            if copy_file:
                asset_path.write_bytes(file_name.read_bytes())
            else:
                try:
                    asset_path.symlink_to(file_name)
                except (OSError, PermissionError):
                    # Fallback to copy if symlink fails (common on Windows)
                    asset_path.write_bytes(file_name.read_bytes())

        # Persist the asset types into a file
        with Path(
            asset_type_path(self._working_dir, self.execution_rid, asset_table)
        ).open("a") as asset_type_file:
            asset_type_file.write(json.dumps({file_name.name: asset_types}) + "\n")

        return AssetFilePath(
            asset_path=asset_path,
            asset_name=asset_name,
            file_name=file_name.name,
            asset_metadata=kwargs,
            asset_types=asset_types,
        )

    def table_path(self, table: str) -> Path:
        """Return a local file path to a CSV to add values to a table on upload.

        Args:
            table: Name of table to be uploaded.

        Returns:
            Pathlib path to the file in which to place table values.
        """
        if table not in self._model.schemas[self._ml_object.domain_schema].tables:
            raise DerivaMLException("Table '{}' not found in domain schema".format(table))

        return table_path(self._working_dir, schema=self._ml_object.domain_schema, table=table)

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
        json_path = feature_value_path(
            self._working_dir,
            schema=self._ml_object.domain_schema,
            target_table=feature.target_table.name,
            feature_name=feature.feature_name,
            exec_rid=self.execution_rid,
        )
        with Path(json_path).open("a", encoding="utf-8") as file:
            for feature in features:
                feature.Execution = self.execution_rid
                file.write(json.dumps(feature.model_dump(mode="json")) + "\n")

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def create_dataset(
        self,
        dataset_types: str | list[str],
        description: str,
        version: DatasetVersion | None = None,
    ) -> RID:
        """Create a new dataset with specified types.

        Args:
            dataset_types: param description:
            description: Markdown description of the dataset being created.
            version: Version to assign to the dataset.  Defaults to 0.1.0

        Returns:
            RID of the newly created dataset.
        """
        return self._ml_object.create_dataset(dataset_types, description, self.execution_rid, version=version)

    def add_dataset_members(
        self,
        dataset_rid: RID,
        members: list[RID] | dict[str, list[RID]],
        validate: bool = True,
        description: str = "",
    ) -> None:
        """Add additional elements to an existing dataset_table.

        Add new elements to an existing dataset. In addition to adding new members, the minor version number of the
        dataset is incremented and the description, if provide is applied to that new version.

        The RIDs in the list to not have to be all from the same table, but they must be from a table that has
        been configured to be a dataset element type.

        Args:
            dataset_rid: RID of dataset_table to extend or None if a new dataset_table is to be created.
            members: List of RIDs of members to add to the  dataset_table. RID must be to a table type that is a
                dataset element type (see DerivaML.add_dataset_element_type).
            validate: Check rid_list to make sure elements are not already in the dataset_table.
            description: Markdown description of the updated dataset.
        """
        return self._ml_object.add_dataset_members(
            dataset_rid=dataset_rid,
            members=members,
            validate=validate,
            description=description,
            execution_rid=self.execution_rid,
        )

    def increment_dataset_version(
        self, dataset_rid: RID, component: VersionPart, description: str = ""
    ) -> DatasetVersion:
        """Increment the version of the specified dataset_table.

        Args:
          dataset_rid: RID to a dataset_table
          component: Which version of the dataset_table to increment.
          dataset_rid: RID of the dataset whose version is to be incremented.
          component: Major, Minor, or Patch
          description: Description of the version update of the dataset_table.

        Returns:
          new semantic version of the dataset_table as a 3-tuple

        Raises:
          DerivaMLException: if provided RID is not to a dataset_table.
        """
        return self._ml_object.increment_dataset_version(
            dataset_rid=dataset_rid,
            component=component,
            description=description,
            execution_rid=self.execution_rid,
        )

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def add_files(
        self,
        files: Iterable[FileSpec],
        dataset_types: str | list[str] | None = None,
        description: str = "",
    ) -> RID:
        """Adds files to the catalog with their metadata.

        Registers files in the catalog along with their metadata (MD5, length, URL) and associates them with
        specified file types.

        Args:
            files: File specifications containing MD5 checksum, length, and URL.
            dataset_types: One or more dataset type terms from File_Type vocabulary.
            description: Description of the files.

        Returns:
            RID: Dataset RID that identifes newly added files. Will be nested to mirror origioanl directory structure
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
            dataset_types=dataset_types,
            execution_rid=self.execution_rid,
            description=description,
        )

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
