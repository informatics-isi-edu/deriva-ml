"""
This module defined the Execution class which is used to interact with the state of an active execution.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Any, Optional
from deriva.core import format_exception
from deriva.core.ermrest_model import Table
from pydantic import validate_call, ConfigDict

from .deriva_definitions import MLVocab, ExecMetadataVocab
from .deriva_definitions import (
    RID,
    Status,
    FileUploadState,
    UploadState,
    DerivaMLException,
)
from .deriva_ml_base import DerivaML, FeatureRecord
from .dataset_aux_classes import DatasetSpec, DatasetVersion, VersionPart
from .dataset import Dataset
from .dataset_bag import DatasetBag
from .execution_configuration import ExecutionConfiguration
from .execution_environment import get_execution_environment
from .upload import (
    execution_metadata_dir,
    execution_asset_dir,
    execution_root,
    feature_root,
    feature_asset_dir,
    feature_value_path,
    is_feature_dir,
    is_feature_asset_dir,
    table_path,
    upload_directory,
)

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


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
        self.start_time = None
        self.status = Status.created
        self.uploaded_assets: list[Path] = []

        self.dataset_rids: list[RID] = []
        self.datasets: list[DatasetBag] = []

        self._working_dir = self._ml_object.working_dir
        self._cache_dir = self._ml_object.cache_dir

        self.workflow_rid = self.configuration.workflow

        if self._ml_object.resolve_rid(configuration.workflow).table.name != "Workflow":
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

        runtime_env_path = ExecMetadataVocab.runtime_env.value
        runtime_env_dir = self.execution_metadata_path(runtime_env_path)
        with NamedTemporaryFile(
            "w+",
            dir=runtime_env_dir,
            prefix="environment_snapshot_",
            suffix=".txt",
            delete=False,
        ) as fp:
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
        if self.dataset_rids and not reload:
            schema_path.Dataset_Execution.insert(
                [
                    {"Dataset": d, "Execution": self.execution_rid}
                    for d in self.dataset_rids
                ]
            )

        # Download assets....
        self.update_status(Status.running, "Downloading assets ...")
        self.asset_paths = [
            self._ml_object.download_asset(asset_rid=a, dest_dir=self._asset_dir())
            for a in self.configuration.assets
        ]
        if self.asset_paths and not reload:
            self._update_execution_asset_table(self.configuration.assets)

        # Save configuration details for later upload
        exec_config_path = ExecMetadataVocab.execution_config.value
        cfile = self.execution_metadata_path(exec_config_path) / "configuration.json"
        with open(cfile, "w", encoding="utf-8") as config_file:
            json.dump(self.configuration.model_dump(), config_file)

        # save runtime env
        self._save_runtime_environment()

        self.start_time = datetime.now()
        self.update_status(Status.pending, "Initialize status finished.")

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def download_dataset_bag(self, dataset: DatasetSpec) -> DatasetBag:
        """Given a RID to a dataset_table, or a MINID to an existing bag, download the bag file, extract it and validate
        that all the metadata is correct

        Args:
            dataset: A dataset specification of a dataset_table or a minid to an existing bag.

        Returns:
            the location of the unpacked and validated dataset_table bag and the RID of the bag
        """
        ds = Dataset(self._ml_object.model, cache_dir=self._cache_dir)
        return ds.download_dataset_bag(dataset, execution_rid=self.execution_rid)

    @validate_call
    def update_status(self, status: Status, msg: str) -> None:
        """Update the status information in the execution record in the DerivaML catalog.

        Args:
            status: A value from the Status Enum
            msg: Additional information about the status
        """
        self.status = status
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
        self.update_status(Status.initializing, "Start ML algorithm ...")

    def execution_stop(self) -> None:
        """Finish the execution and update the duration and status of execution."""
        duration = datetime.now() - self.start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration = f"{round(hours, 0)}H {round(minutes, 0)}min {round(seconds, 4)}sec"

        self.update_status(Status.completed, "Algorithm execution ended.")
        self._ml_object.pathBuilder.schemas[self._ml_object.ml_schema].Execution.update(
            [{"RID": self.execution_rid, "Duration": duration}]
        )

    def _upload_execution_dirs(self) -> dict[str, FileUploadState]:
        """Upload execution assets at _working_dir/Execution_asset.

        This routine uploads the contents of the
        Execution_Asset directory, and then updates the execution_asset table in the ML schema to have references
        to these newly uploaded files.

        Returns:
          dict: Results of the upload operation.

        Raises:
          DerivaMLException: If there is an issue uploading the assets.
        """

        def asset_name(p: str) -> str:
            return Path(*Path(p).parts[-2:]).as_posix()

        try:
            self.update_status(Status.running, "Uploading execution files...")
            results = upload_directory(self._ml_object.model, self._execution_root)
            results = {asset_name(k): v for k, v in results.items()}

            execution_assets = [
                r.result["RID"]
                for r in results.values()
                if r.state == UploadState.success and "Execution_Asset_Type" in r.result
            ]
            execution_metadata = [
                r.result["RID"]
                for r in results.values()
                if r.state == UploadState.success
                and "Execution_Metadata_Type" in r.result
            ]
            self._update_execution_asset_table(execution_assets)
            self._update_execution_metadata_table(execution_metadata)

        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error)
            raise DerivaMLException(f"Fail to upload execution_assets. Error: {error}")

        self.update_status(Status.running, "Updating features...")

        feature_assets = defaultdict(dict)

        def traverse_bottom_up(directory: Path):
            """Traverses the directory tree in a bottom-up order.

            Args:
              directory: Path:

            Returns:

            """
            entries = list(directory.iterdir())
            for entry in entries:
                if entry.is_dir():
                    yield from traverse_bottom_up(entry)
            yield directory

        for p in traverse_bottom_up(self._feature_root):
            if m := is_feature_asset_dir(p):
                try:
                    self.update_status(
                        Status.running, f'Uploading feature {m["feature_name"]}...'
                    )
                    feature_assets[m["target_table"], m["feature_name"]] = (
                        self._ml_object.upload_assets(p)
                    )
                    results |= feature_assets[m["target_table"], m["feature_name"]]
                except Exception as e:
                    error = format_exception(e)
                    self.update_status(Status.failed, error)
                    raise DerivaMLException(
                        f"Fail to upload execution metadata. Error: {error}"
                    )
            elif m := is_feature_dir(p):
                files = [f for f in p.iterdir() if f.is_file()]
                if files:
                    self._update_feature_table(
                        target_table=m["target_table"],
                        feature_name=m["feature_name"],
                        feature_file=files[0],
                        uploaded_files=feature_assets[
                            m["target_table"], m["feature_name"]
                        ],
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

    def _asset_dir(self) -> Path:
        """

        Args:

        Returns:
          :return: PathLib path object to model directory.

        """
        path = self._working_dir / self.execution_rid / "asset"
        path.mkdir(parents=True, exist_ok=True)
        return path

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
        uploaded_files: dict[str, FileUploadState],
    ) -> None:
        """

        Args:
            target_table: str:
            feature_name: str:
            feature_file: str | Path:
            uploaded_files: dict[str: FileUploadState]:
        """

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
            """

            Args:
              e:

            Returns:

            """
            # Go through the asset columns and replace the file name with the RID for the uploaded file.
            for c in asset_columns:
                e[c] = asset_map[e[c]]
            return e

        # Create a map between a file name that appeared in the file to the RID of the uploaded file.
        asset_map = {
            file: asset.result["RID"]
            for file, asset in uploaded_files.items()
            if asset.state == UploadState.success and asset.result
        }
        with open(feature_file, "r") as feature_values:
            entities = [map_path(e) for e in csv.DictReader(feature_values)]
        self._ml_object.domain_path.tables[feature_table].insert(entities)

    def _update_execution_metadata_table(self, assets: list[RID]) -> None:
        """Upload execution metadata at _working_dir/Execution_metadata."""
        ml_schema_path = self._ml_object.pathBuilder.schemas[self._ml_object.ml_schema]
        entities = [
            {"Execution_Metadata": metadata_rid, "Execution": self.execution_rid}
            for metadata_rid in assets
        ]
        ml_schema_path.Execution_Metadata_Execution.insert(entities)

    def _update_execution_asset_table(self, assets: list[RID]) -> None:
        """Assets associated with an execution must be linked to an execution entity after they are uploaded into
        the catalog. This routine takes a list of uploaded assets and makes that association.

        Args:
            assets: list of RIDS for execution assets.:
        """
        ml_schema_path = self._ml_object.pathBuilder.schemas[self._ml_object.ml_schema]
        entities = [
            {"Execution_Asset": asset_rid, "Execution": self.execution_rid}
            for asset_rid in assets
        ]
        ml_schema_path.Execution_Asset_Execution.insert(entities)

    @property
    def _execution_metadata_dir(self) -> Path:
        """

        Args:

        Returns:
          to the catalog by the execution_upload method in an execution object.

          :return:

        """
        return execution_metadata_dir(
            self._working_dir, exec_rid=self.execution_rid, metadata_type=""
        )

    def execution_metadata_path(self, metadata_type: str) -> Path:
        """Return a pathlib Path to the directory in which to place files of type metadata_type.

        These files are uploaded to the catalog as part of the execution of the upload_execution method in DerivaML.

        Args:
            metadata_type: Type of metadata to be uploaded.  Must be a term in Metadata_Type controlled vocabulary.

        Returns:
            Path to the directory in which to place files of type metadata_type.
        """
        self._ml_object.lookup_term(
            MLVocab.execution_metadata_type, metadata_type
        )  # Make sure metadata type exists.
        return execution_metadata_dir(
            self._working_dir, exec_rid=self.execution_rid, metadata_type=metadata_type
        )

    @property
    def _execution_asset_dir(self) -> Path:
        """

        Args:

        Returns:
          :return:

        """
        return execution_asset_dir(
            self._working_dir, exec_rid=self.execution_rid, asset_type=""
        )

    def execution_asset_path(self, asset_type: str) -> Path:
        """Return a pathlib Path to the directory in which to place files for the specified execution_asset type.

        These files are uploaded as part of the upload_execution method in DerivaML class.

        Args:
            asset_type: Type of asset to be uploaded.  Must be a term in Asset_Type controlled vocabulary.

        Returns:
            Path in which to place asset files.

        Raises:
            DerivaException: If the asset type is not defined.
        """
        self._ml_object.lookup_term(MLVocab.execution_asset_type, asset_type)

        return execution_asset_dir(
            self._working_dir, exec_rid=self.execution_rid, asset_type=asset_type
        )

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

    def feature_paths(
        self, table: Table | str, feature_name: str
    ) -> tuple[Path, dict[str, Path]]:
        """Return the file path of where to place feature values, and assets for the named feature and table.

        A side effect of calling this routine is that the directories in which to place the feature values and assets
        will be created

        Args:
            table: The table with which the feature is associated.
            feature_name: Name of the feature

        Returns:
            A tuple whose first element is the path for the feature values and whose second element is a dictionary
            of associated asset table names and corresponding paths.
        """
        feature = self._ml_object.lookup_feature(table, feature_name)

        tpath = feature_value_path(
            self._working_dir,
            schema=self._ml_object.domain_schema,
            target_table=feature.target_table.name,
            feature_name=feature_name,
            exec_rid=self.execution_rid,
        )
        asset_paths = {
            asset_table.name: feature_asset_dir(
                self._working_dir,
                exec_rid=self.execution_rid,
                schema=self._ml_object.domain_schema,
                target_table=feature.target_table.name,
                feature_name=feature_name,
                asset_table=asset_table.name,
            )
            for asset_table in feature.asset_columns
        }
        return tpath, asset_paths

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
    def write_feature_file(self, features: Iterable[FeatureRecord]) -> None:
        """Given a collection of Feature records, write out a CSV file in the appropriate assets directory so that this
        feature gets uploaded when the execution is complete.

        Args:
            features: Iterable of Feature records to write.
        """

        feature_iter = iter(features)
        first_row = next(feature_iter)
        feature = first_row.feature
        csv_path, _ = self.feature_paths(
            feature.target_table.name, feature.feature_name
        )

        fieldnames = {"Execution", "Feature_Name", feature.target_table.name}
        fieldnames |= {f.name for f in feature.feature_columns}

        with open(csv_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(first_row.model_dump())
            for feature in feature_iter:
                writer.writerow(feature.model_dump())

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
