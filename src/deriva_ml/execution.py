"""
This module defined the Execution class which is used to interact with the state of an active execution.
"""

from __future__ import annotations

import csv
from collections import defaultdict
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, Any, Optional

from deriva.core import format_exception
from pydantic import validate_call, ConfigDict
import sys

from .deriva_definitions import ExecMetadataVocab
from .deriva_definitions import RID, Status, FileUploadState, DerivaMLException, MLVocab
from .deriva_ml_base import DerivaML, FeatureRecord
from .dataset_aux_classes import DatasetSpec, DatasetVersion, VersionPart
from .dataset_bag import DatasetBag
from .execution_configuration import ExecutionConfiguration, Workflow
from .execution_environment import get_execution_environment
from .upload import (
    asset_file_path,
    execution_root,
    feature_root,
    asset_root,
    feature_value_path,
    is_feature_dir,
    table_path,
    upload_directory,
    normalize_asset_dir,
    AssetFilePath,
    asset_type_path
)

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


try:
    from jupyter_server.serverapp import list_running_servers
except ImportError:

    def list_running_servers():
        return []


class Execution:
    """The Execution class is used to capture the context of an activity within DerivaML.  While these are primarily
    computational, manual processes can be represented by an execution as well.

    Within DerivaML, Executions are used to provide providence. Every dataset_table and data file that is generated is
    associated with an execution, which records which program and input parameters were used to generate that data.

    Execution objects are created from an ExecutionConfiguration, which provides information about what DerivaML
    datasets will be used, what additional files (assets) are required, what code is being run (Workflow) and an
    optional description of the Execution.  Side effects of creating an execution object are:

    1. An execution record is created in the catalog and the RID of that record  recorded,
    2. Any specified datasets are downloaded and materialized
    3. Any additional required assets are downloaded.

    Once execution is complete, a method can be called to upload any data produced by the execution. In addition, the
    Execution object provides methods for locating where to find downloaded datasets and assets, and also where to
    place any data that may be uploaded.

    Finally, the execution object can update its current state in the DerivaML catalog, allowing users to remotely
    track the progress of their execution.

    Attributes:
        dataset_rids (list[RID]): The RIDs of the datasets to be downloaded and materialized as part of the execution.
        datasets (list[DatasetBag]): List of datasetBag objects that referred the materialized datasets specified in.
            `dataset_rids`.
        configuration (ExecutionConfiguration): The configuration of the execution.
        workflow_rid (RID): The RID of the workflow associated with the execution.
        status (Status): The status of the execution.
    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        configuration: ExecutionConfiguration,
        ml_object: "DerivaML",
        reload: Optional[RID] = None,
        dry_run: bool = False,
    ):
        """

        Args:
            configuration:
            ml_object:
            reload: RID of previously initialized execution object.
        """
        self.asset_paths: list[Path] = []
        self.configuration = configuration
        self._ml_object = ml_object
        self._logger = ml_object._logger
        self.start_time = None
        self.stop_time = None
        self.status = Status.created
        self.uploaded_assets: list[Path] = []
        self.configuration.argv = sys.argv

        self.dataset_rids: list[RID] = []
        self.datasets: list[DatasetBag] = []
        self.parameters = self.configuration.parameters

        self._working_dir = self._ml_object.working_dir
        self._cache_dir = self._ml_object.cache_dir
        self._dry_run = dry_run

        if isinstance(self.configuration.workflow, Workflow):
            self.workflow_rid = (
                self._ml_object.add_workflow(self.configuration.workflow)
                if not self._dry_run
                else "0000"
            )
        else:
            self.workflow_rid = self.configuration.workflow
            if (
                self._ml_object.resolve_rid(configuration.workflow).table.name
                != "Workflow"
            ):
                raise DerivaMLException(
                    "Workflow specified in execution configuration is not a Workflow"
                )

        for d in self.configuration.datasets:
            if self._ml_object.resolve_rid(d.rid).table.name != "Dataset":
                raise DerivaMLException(
                    "Dataset specified in execution configuration is not a dataset"
                )

        for a in self.configuration.assets:
            if not self._ml_object.model.is_asset(
                self._ml_object.resolve_rid(a).table.name
            ):
                raise DerivaMLException(
                    "Asset specified in execution configuration is not a asset table"
                )

        schema_path = self._ml_object.pathBuilder.schemas[self._ml_object.ml_schema]
        if reload:
            self.execution_rid = reload
            if self.execution_rid == "0000":
                self._dry_run = True
        elif self._dry_run:
            self.execution_rid = "0000"
        else:
            self.execution_rid = schema_path.Execution.insert(
                [
                    {
                        "Description": self.configuration.description,
                        "Workflow": self.workflow_rid,
                    }
                ]
            )[0]["RID"]

        # Create a directory for execution rid so we can recover state in case of a crash.
        execution_root(prefix=self._ml_object.working_dir, exec_rid=self.execution_rid)
        self._initialize_execution(reload)

    def _save_runtime_environment(self):
        runtime_env_path = self.asset_file_path(
            asset_name="Execution_Metadata",
            file_name=f"environment_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            asset_types=ExecMetadataVocab.runtime_env.value,
        )
        with open(runtime_env_path, "w") as fp:
            json.dump(get_execution_environment(), fp)

    def _initialize_execution(self, reload: Optional[RID] = None) -> None:
        """Initialize the execution by a configuration  in the Execution_Metadata table.
        Setup working directory and download all the assets and data.

        :raise DerivaMLException: If there is an issue initializing the execution.

        Args:
            reload: RID of previously initialized execution.

        Returns:

        """
        # Materialize bdbag
        for dataset in self.configuration.datasets:
            self.update_status(
                Status.initializing, f"Materialize bag {dataset.rid}... "
            )
            self.datasets.append(self.download_dataset_bag(dataset))
            self.dataset_rids.append(dataset.rid)
        # Update execution info
        schema_path = self._ml_object.pathBuilder.schemas[self._ml_object.ml_schema]
        if self.dataset_rids and not (reload or self._dry_run):
            schema_path.Dataset_Execution.insert(
                [
                    {"Dataset": d, "Execution": self.execution_rid}
                    for d in self.dataset_rids
                ]
            )

        # Download assets....
        self.update_status(Status.running, "Downloading assets ...")
        self.asset_paths = []
        for asset_rid in self.configuration.assets:
            asset_table = self._ml_object.resolve_rid(asset_rid).table.name
            dest_dir = self._ml_object.working_dir / "downloaded-assets" / asset_table
            dest_dir.mkdir(parents=True, exist_ok=True)
            self.asset_paths.append(
                self._ml_object.download_asset(asset_rid=asset_rid, dest_dir=dest_dir)
            )

        if self.asset_paths and not (reload or self._dry_run):
            self._update_asset_execution_table(
                {"Execution_Assets": self.configuration.assets}, asset_role="Input"
            )

        # Save configuration details for later upload
        cfile = self.asset_file_path(
            "Execution_Metadata",
            file_name="configuration.json",
            asset_type=ExecMetadataVocab.execution_config.value,
        )
        with open(cfile, "w", encoding="utf-8") as config_file:
            json.dump(self.configuration.model_dump(), config_file)

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
        """The root path to all execution specific files.
        :return:

        Args:

        Returns:

        """
        return feature_root(self._working_dir, self.execution_rid)

    @property
    def _asset_root(self) -> Path:
        """The root path to all execution specific files.
        :return:

        Args:

        Returns:

        """
        return asset_root(self._working_dir, self.execution_rid)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def download_dataset_bag(self, dataset: DatasetSpec) -> DatasetBag:
        """Given a RID to a dataset_table, or a MINID to an existing bag, download the bag file, extract it and validate
        that all the metadata is correct

        Args:
            dataset: A dataset specification of a dataset_table or a minid to an existing bag.

        Returns:
            the location of the unpacked and validated dataset_table bag and the RID of the bag
        """
        return self._ml_object.download_dataset_bag(
            dataset, execution_rid=self.execution_rid
        )

    @validate_call
    def update_status(self, status: Status, msg: str) -> None:
        """Update the status information in the execution record in the DerivaML catalog.

        Args:
            status: A value from the Status Enum
            msg: Additional information about the status
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
        """Start an execution, uploading status to catalog"""

        self.start_time = datetime.now()
        self.uploaded_assets = None
        self.update_status(Status.initializing, "Start execution  ...")

    def execution_stop(self) -> None:
        """Finish the execution and update the duration and status of execution."""
        self.stop_time = datetime.now()
        duration = self.stop_time - self.start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration = f"{round(hours, 0)}H {round(minutes, 0)}min {round(seconds, 4)}sec"

        self.update_status(Status.completed, "Algorithm execution ended.")
        if not self._dry_run:
            self._ml_object.pathBuilder.schemas[
                self._ml_object.ml_schema
            ].Execution.update([{"RID": self.execution_rid, "Duration": duration}])

    def _upload_execution_dirs(self) -> dict[str, RID]:
        """Upload execution assets at _working_dir/Execution_asset.

        This routine uploads the contents of the
        Execution_Asset directory, and then updates the execution_asset table in the ML schema to have references
        to these newly uploaded files.

        Returns:
          dict: Results of the upload operation.

        Raises:
          DerivaMLException: If there is an issue uploading the assets.
        """

        try:
            self.update_status(Status.running, "Uploading execution files...")
            results = upload_directory(self._ml_object.model, self._asset_root)
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error)
            raise DerivaMLException(f"Fail to upload execution_assets. Error: {error}")

        asset_rids = defaultdict(list)
        asset_rid_map = defaultdict(list)
        for path, status in results.items():
            asset_table, file_name = normalize_asset_dir(path)
            asset_rids[asset_table].append((file_name, status.result["RID"]))
            asset_rid_map[(asset_table, file_name)] = status.result["RID"]

        self._update_asset_execution_table(asset_rids)
        self.update_status(Status.running, "Updating features...")

        for p in self._feature_root.glob("**/*.jsonl"):
            m = is_feature_dir(p.parent)
            self._update_feature_table(
                target_table=m["target_table"],
                feature_name=m["feature_name"],
                feature_file=p,
                uploaded_files=asset_rid_map,
            )

        self.update_status(Status.running, "Upload assets complete")
        return results

    def upload_execution_outputs(
        self, clean_folder: bool = True
    ) -> dict[str, FileUploadState]:
        """Upload all the assets and metadata associated with the current execution.

        This will include any new assets, features, or table values.

        Args:
            clean_folder: bool:  (Default value = True)

        Returns:
            Results of the upload operation. Asset names are all relative to the execution upload directory.
            Uploaded assets with key as assets' suborder name, values as an
            ordered dictionary with RID and metadata in the Execution_Asset table.
        """
        if self._dry_run:
            return {}
        try:
            uploaded_assets = self._upload_execution_dirs()
            self.update_status(Status.completed, "Successfully end the execution.")
            if clean_folder:
                self._clean_folder_contents(self._execution_root)
            return uploaded_assets
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error)
            raise e

    def _clean_folder_contents(self, folder_path: Path):
        """

        Args:
            folder_path: Path:
        """
        try:
            with os.scandir(folder_path) as entries:
                for entry in entries:
                    if entry.is_dir() and not entry.is_symlink():
                        shutil.rmtree(entry.path)
                    else:
                        os.remove(entry.path)
        except OSError as e:
            error = format_exception(e)
            self.update_status(Status.failed, error)

    def _update_feature_table(
        self,
        target_table: str,
        feature_name: str,
        feature_file: str | Path,
        uploaded_files: dict[tuple[str, str], tuple[str, RID]],
    ) -> None:
        """

        Args:
            target_table: str:
            feature_name: str:
            feature_file: str | Path:
            uploaded_files: Dictionary whose key ia an asset name, file-name pair, and whose value is a filenam, RID of that asset.
        """

        # Get the column names of all the Feature columns that should be the RID of an asset
        asset_columns = [
            c.name
            for c in self._ml_object.feature_record_class(
                target_table, feature_name
            ).feature.asset_columns
        ]
        feature_table = self._ml_object.feature_record_class(
            target_table, feature_name
        ).feature.feature_table.name

        def map_path(e):
            """Go through the asset columns and replace the file name with the RID for the uploaded file."""
            for c in asset_columns:
                asset_table, asset_file = normalize_asset_dir(e[c])
                e[c] = uploaded_files[normalize_asset_dir(e[c])]
            return e

        # Load the JSON file that has the set of records that contain the feature values.
        with open(feature_file, "r") as feature_values:
            entities = [json.loads(l) for l in feature_values.readline()]

        # Update the asset columns in the feature and add to the catalog.
        self._ml_object.domain_path.tables[feature_table].insert(
            [map_path(e) for e in entities]
        )

    def _update_asset_execution_table(
        self,
        uploaded_assets: dict[str, list[RID]],
        asset_role: str = "Output",
    ):
        """Add entry to association table connecting an asset to an execution RID

        Args:
            uploaded_assets: Dictionary whose key is the name of an asset table, and whose value is a list of RIDs for
                newly added assets to that table.
             asset_role: A term or list of terms from the Asset_Role vocabulary.
        """

        # Make sure  the asset role is in the controlled vocabulary table.
        self._ml_object.lookup_term(MLVocab.asset_role, asset_role)
        # Now we need to attach the asset types
        asset_type_map = {}
        with open(asset_type_path(prefix, schema, asset_type), "r") as f:
            asset_type_map.update(json.loads(f.readline()))


        pb = self._ml_object.pathBuilder
        for asset_table_name, rid_list in uploaded_assets.items():
            asset_exe = self._ml_object.model.find_association(asset_table_name, "Execution")
            asset_asset_type = self._ml_object.model.find_association(asset_table_name, "Execution")
            asset_exe_path = pb.schemas[asset_exe.schema.name].tables[asset_exe.name]
            asset_exe_path.insert(
                [
                    {
                        asset_table_name: rid,
                        "Execution": self.execution_rid,
                        "Asset_Role": asset_role,
                    }
                    for rid in rid_list
                ]
            )

            asset_type_path = pb.schemas[asset_type.schema.name].tables[asset_type.name]
            asset_type_path.insert([{asset_table_name: rid, 'Asset_Type': t} for t in asset_types[] for rid in rid_list]
                                   )





    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def asset_file_path(
        self,
        asset_name: str,
        file_name: str,
        asset_types: Optional[list[str] | str] = None,
        **kwargs,
    ) -> AssetFilePath:
        """Return a pathlib Path to the directory in which to place files for the specified execution_asset type.

        These files are uploaded as part of the upload_execution method in DerivaML class.

        Args:
            asset_name: Type of asset to be uploaded.  Must be a term in Asset_Type controlled vocabulary.
            asset_types: Type of asset to be uploaded.  Defaults to name of the asset.
            file_name: Name of file to be uploaded.
            **kwargs: Any additional metadata values that may be part of the asset table.

        Returns:
            Path in which to place asset files.

        Raises:
            DerivaException: If the asset type is not defined.
        """
        if not self._ml_object.model.is_asset(asset_name):
            DerivaMLException(f"Table {asset_name} is not an asset")

        asset_types = asset_types or kwargs.get("Asset_Type", None) or asset_name
        asset_type = [asset_types] if isinstance(asset_types, str) else asset_types
        for t in asset_type:
            self._ml_object.lookup_term(MLVocab.asset_type, t)

        with open(asset_type_path(self._working_dir, asset_name), "a") as f:
            csvwriter = csv.writer(f)
            for t in asset_types:
                csvwriter.writerow([asset_name, file_name, t])
            csvwriter.writerow([])
        )
        return AssetFilePath(
            asset_file_path(
                self._working_dir,
                exec_rid=self.execution_rid,
                asset_table=self._ml_object.model.name_to_table(asset_name),
                file_name=file_name,
                metadata=kwargs,
            ),
            asset_types=asset_types,
        )

    def table_path(self, table: str) -> Path:
        """Return a local file path to a CSV to add values to a table on upload.

        Args:
            table: Name of table to be uploaded.

        Returns:
            Pathlib path to the file in which to place table values.
        """
        if (
            table
            not in self._ml_object.model.schemas[self._ml_object.domain_schema].tables
        ):
            raise DerivaMLException(
                "Table '{}' not found in domain schema".format(table)
            )

        return table_path(
            self._working_dir, schema=self._ml_object.domain_schema, table=table
        )

    def execute(self) -> Execution:
        """Initiate an execution with provided configuration. Can be used in a context manager."""
        return self

    @validate_call
    def add_features(self, features: Iterable[FeatureRecord]) -> None:
        """Given a collection of Feature records, write out a CSV file in the appropriate assets directory so that this
        feature gets uploaded when the execution is complete.

        Args:
            features: Iterable of Feature records to write.
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

        with open(json_path, "a", encoding="utf-8") as file:
            for feature in features:
                feature.Execution = self.execution_rid
                file.write(json.dumps(feature.model_dump(mode='json')) + "\n")


    @validate_call
    def create_dataset(self, dataset_types: str | list[str], description: str) -> RID:
        """Create a new dataset with specified types.

        Args:
            dataset_types: param description:
            description: Markdown description of the dataset being created.

        Returns:
            RID of the newly created dataset.
        """
        return self._ml_object.create_dataset(
            dataset_types, description, self.execution_rid
        )

    def add_dataset_members(
        self,
        dataset_rid: RID,
        members: list[RID],
        validate: bool = True,
        description: str = "",
    ) -> None:
        """Add additional elements to an existing dataset_table.

        Add new elements to an existing dataset. In addition to adding new members, the minor version number of the
        dataset is incremented and the description, if provide is applied to that new version.

        Args:
            dataset_rid: RID of dataset_table to extend or None if new dataset_table is to be created.
            members: List of RIDs of members to add to the  dataset_table.
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
          component: Major, Minor or Patch
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
            logging.error(
                f"Exception type: {exc_type}, Exception value: {exc_value}, Exception traceback: {exc_tb}"
            )
            return False
