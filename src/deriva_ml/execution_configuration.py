import csv
from datetime import datetime
from collections import defaultdict
from deriva.core import format_exception
from deriva.core.ermrest_model import Table
from .deriva_ml_base import MLVocab, DerivaML, DerivaMLException, ExecMetadataVocab, FeatureRecord
from .deriva_definitions import RID, Status
from .upload import is_feature_dir, is_feature_asset_dir
from .upload import execution_metadata_dir, execution_assets_dir, asset_dir, execution_root
from .upload import table_path
from .upload import feature_root, feature_asset_dir, feature_value_path
import json
import logging
import os
from typing import Optional
from tempfile import NamedTemporaryFile
from pathlib import Path
from pydantic import BaseModel, ValidationError

from typing import Iterator, Iterable

class Workflow(BaseModel):
    """
    A specification of a workflow.  Must have a name, URI to the workflow instance, and a type.
    """
    name: str
    url: str
    workflow_type: str
    version: Optional[str] = None
    description: str = None


class Execution(BaseModel):
    description: str


class ExecutionConfiguration(BaseModel):
    datasets: list[str] = []
    assets: list[RID] = []      # List of RIDs to model files.
    workflow: Workflow
    execution: Execution
    description: str = ""

    @staticmethod
    def load_configuration(file: str) -> "ExecutionConfiguration":
        with open(file) as fd:
                return ExecutionConfiguration.model_validate(json.load(fd))

class Execution:
    """
    Data model representing configuration records.

    Attributes:
    - vocabs (dict): Dictionary containing vocabulary terms with key as vocabulary table name,
    and values as a list of dict containing name, rid pairs.
    - execution_rid (str): Execution identifier in catalog.
    - workflow_rid (str): Workflow identifier in catalog.
    - bag_paths (list): List of paths to bag files.
    - asset_paths (list): List of paths to assets.
    - configuration(Path): Path to the configuration file.

    """
    def __init__(self,
        configuration: ExecutionConfiguration,
        ml_object: 'DerivaML'
    ):
        try:
            self.configuration = ExecutionConfiguration.model_validate(configuration)
            logging.info('Configuration validation successful!')
        except ValidationError as e:
            raise DerivaMLException(f'configuration validation failed: {e}')

        self.bag_paths: list[Path] = []
        self.dataset_rids: list[RID] = []
        self.asset_paths: list[Path] = []
        self._ml_object = ml_object
        self.start_time = None
        self.status = Status.pending.value

        self.workflow_rid = self._add_workflow()
        self.execution_rid = self._add_execution(description=self.configuration.execution.description)
        self._initialize_execution()


    def _add_execution(self, workflow_rid: str = '',
                       datasets: list[str] = None,
                       description: str = '') -> RID:
        """
        Add an execution to the Execution table.

        Args:
        - workflow_rid (str): Resource Identifier (RID) of the workflow.
        - datasets (List[str]): List of dataset RIDs.
        - description (str): Description of the execution.

        Returns:
        - str: Resource Identifier (RID) of the added execution.

        """
        datasets = datasets or []
        ml_schema_path = self._ml_object.pathBuilder.schemas[self._ml_object.ml_schema]
        if workflow_rid:
            execution_rid = (
                ml_schema_path.Execution.insert([{'Description': description, 'Workflow': workflow_rid}]))[0]['RID']
        else:
            execution_rid = ml_schema_path.Execution.insert([{'Description': description}])[0]['RID']
        if datasets:
            ml_schema_path.Dataset_Execution.insert([{'Dataset': d, 'Execution': execution_rid} for d in datasets])
        return execution_rid

    def _update_execution(self,
                          datasets: list[str] = None,
                          description: str = ''):
        """
        Update an existing execution to build the linkage between the
        Execution table and the Workflow and Dataset table.

        Args:
        - datasets (List[str]): List of dataset identifiers.
        - description (str): Description of the execution.

        """
        datasets = datasets or []
        schema_path = self._ml_object.pathBuilder.schemas[self._ml_object.ml_schema]
        schema_path.Execution.update([{'RID': self.execution_rid, 'Workflow': self.workflow_rid, 'Description': description}])
        if datasets:
            schema_path.Dataset_Execution.insert([{'Dataset': d, 'Execution': self.execution_rid} for d in datasets])

    def _add_workflow(self, workflow_name: str, url: str, workflow_type: str,
                      version: str = '',
                      description: str = '') -> RID:
        """
        Add a workflow to the Workflow table.

        Args:
        - workflow_name (str): Name of the workflow.
        - url (str): URL of the workflow.
        - workflow_type (str): Type of the workflow.
        - version (str): Version of the workflow.
        - description (str): Description of the workflow.

        Returns:
        - str: Resource Identifier (RID) of the added workflow.

        """
        self.update_status(Status.running, 'Inserting workflow... ')
        workflow = self.configuration.workflow
        # Check to make sure that the workflow is not already in the table. If it's not, add it.
        ml_schema_path = self._ml_object.pathBuilder.schemas[self._ml_object.ml_schema]
        try:
            url_column = ml_schema_path.Workflow.URL
            workflow_record = list(ml_schema_path.Workflow.filter(url_column == url).entities())[0]
            workflow_rid = workflow_record['RID']
        except IndexError:
            # Record doesn't exist already
            workflow_record = {
                'URL': url,
                'Name': workflow.name,
                'Description': workflow.description,
                'Checksum': self._get_checksum(url),
                'Version': version,
                MLVocab.workflow_type: self.lookup_term(MLVocab.workflow_type, workflow_type).name}
            workflow_rid = ml_schema_path.Workflow.insert([workflow_record])[0]['RID']
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error)
            raise DerivaMLException(f'Failed to insert workflow. Error: {error}')
        return workflow_rid

    def _initialize_execution(self) -> None:
        """
        Initialize the execution by a configuration  in the Execution_Metadata table.
        Setup working directory and download all the assets and data.

        :raise DerivaMLException: If there is an issue initializing the execution.
        """


        # Materialize bdbag
        dataset_rids = []
        self.bag_paths = []
        for bag in self.configuration.datasets:
            self.update_status(Status.running, f'Materialize bag {bag}... ')
            bag_path, dataset_rid = self._ml_object.materialize_dataset_bag(bag, self.execution_rid)
            dataset_rids.append(dataset_rid)
            self.bag_paths.append(bag_path)

        # Download model
        self.update_status(Status.running, 'Downloading assets ...')
        asset_path = self.asset_dir(self.execution_rid).as_posix()
        self.asset_paths = [self._ml_instance._download_execution_file(
            file_rid=m,
            execution_rid=self.execution_rid,
            dest_dir=asset_path) for m in self.configuration.assets]


        # Save configuration details for later upload
        exec_config_path = ExecMetadataVocab.execution_config.value
        cfile = configuration_record.execution_metadata_path(exec_config_path) / "configuration.json"
        with open(cfile, 'w', encoding="utf-8") as config_file:
            json.dump(self.configuration.model_dump_json(), config_file)

        # Update execution info
        execution_rid = self._update_execution(self.execution_rid, self.workflow_rid, dataset_rids,
                                               self.configuration.execution.description)
        self.update_status(Status.running, 'Execution created ')

        # save runtime env
        runtime_env_path = ExecMetadataVocab.runtime_env.value
        runtime_env_dir = self.execution_metadata_path(runtime_env_path)
        with NamedTemporaryFile('w+', dir=runtime_env_dir, prefix='environment_snapshot_',
                                suffix='.txt', delete=False) as fp:
            for dist in distributions():
                fp.write(f"{dist.metadata['Name']}=={dist.version}\n")
        self.start_time = datetime.now()
        self.update_status(Status.running, 'Initialize status finished.')

    def update_status(self, status: Status, msg: str):
        self.status = status.value
        self._ml_object.pathBuilder.schemas[self._ml_object.ml_schema].Execution.update(
            [{'RID': self.execution_rid, 'Status': self.status, 'Status_Detail': msg}]
        )

    def execution_end(self, execution_rid: RID) -> None:
        """
        Finish the execution and update the duration and status of execution.

        Args:
        - execution_rid (str): Resource Identifier (RID) of the execution.

        Returns:
        - dict: Uploaded assets with key as assets' suborder name,
        values as an ordered dictionary with RID and metadata in the Execution_Assets table.

        """
        duration = datetime.now() - self.start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration = f'{round(hours, 0)}H {round(minutes, 0)}min {round(seconds, 4)}sec'

        self.update_status(Status.running, 'Algorithm execution ended.')
        self._ml_object.pathBuilder.schemas[self._ml_object.ml_schema].Execution.update(
            [{'RID': execution_rid, 'Duration': duration}])

    def _upload_execution_dirs(self, configuration: Execution) -> dict[str, FileUploadState]:
        """
        Upload execution assets at working_dir/Execution_assets.  This routine uploads the contents of the
        Execution_Assets directory, and then updates the execution_assets table in the ML schema to have references
        to these newly uploaded files.

        Args:
        - execution_rid (str): Resource Identifier (RID) of the execution.

        Returns:
        - dict: Results of the upload operation.

        Raises:
        - DerivaMLException: If there is an issue uploading the assets.

        """
        results = {}
        try:
            self.update_status(Status.running, 'Uploading execution assets...')
            execution_asset_files = self.upload_assets(configuration._execution_assets_dir)
            self._update_execution_asset_table(self.execution_rid, execution_asset_files)
            results |= execution_asset_files
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error, configuration.execution_rid)
            raise DerivaMLException(
                f'Fail to upload execution_assets. Error: {error}')
        try:
            self.update_status(Status.running, 'Uploading execution metadata...', configuration.execution_rid)
            execution_metadata_files = self.upload_assets(configuration._execution_metadata_dir)
            self._update_execution_metadata_table(configuration.execution_rid, execution_metadata_files)
            results |= execution_metadata_files
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error)
            raise DerivaMLException(
                f'Fail to upload execution metadata. Error: {error}')

        self.update_status(Status.running, f'Updating features...')
        feature_assets = defaultdict(dict)

        def traverse_bottom_up(directory: Path):
            """
            Traverses the directory tree in a bottom-up order.
            """
            entries = list(directory.iterdir())
            for entry in entries:
                if entry.is_dir():
                    yield from traverse_bottom_up(entry)
            yield directory

        # for p, _, files in configuration.feature_root.walk(top_down=False):
        for p in traverse_bottom_up(configuration.feature_root):
            if m := is_feature_asset_dir(p):
                try:
                    self.update_status(Status.running, f'Uploading feature {m["feature_name"]}...')
                    feature_assets[m['target_table'], m['feature_name']] = self.upload_assets(p)
                    results |= feature_assets[m['target_table'], m['feature_name']]
                except Exception as e:
                    error = format_exception(e)
                    self.update_status(Status.failed, error)
                    raise DerivaMLException(
                        f'Fail to upload execution metadata. Error: {error}')
            elif m := is_feature_dir(p):
                # self._update_feature_table(target_table=m['target_table'],
                #                            feature_name=m['feature_name'],
                #                            feature_file=p / files[0],
                #                            uploaded_files=feature_assets[m['target_table'], m['feature_name']])
                files = list(p.iterdir())
                if files:
                    self._update_feature_table(target_table=m['target_table'],
                                               feature_name=m['feature_name'],
                                               feature_file=files[0],
                                               uploaded_files=feature_assets[m['target_table'], m['feature_name']])

        self.update_status(Status.running, f'Upload assets complete')
        return results

    def upload_execution(self, configuration: Execution, clean_folder: bool = True) -> None:
        """
        Upload all the assets and metadata associated with the current execution.

        Args:
        - execution_rid (str): Resource Identifier (RID) of the execution.

        Returns:
        - dict: Uploaded assets with key as assets' suborder name,
        values as an ordered dictionary with RID and metadata in the Execution_Assets table.

        """
        try:
            #uploaded_assets =
            self._upload_execution_dirs(configuration)
            self.update_status(Status.completed,
                               'Successfully end the execution.')
            if clean_folder:
                self._clean_folder_contents(configuration.execution_root, self.execution_rid)
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error)
            raise e

    def _upload_execution_asset(self, file, asset_type: str) -> dict[str, Any]:
        return self.upload_asset(file, "Execution_Assets", Execution_Asset_Type=asset_type)
    def _download_execution_file(self, file_rid: RID, execution_rid='', dest_dir: str = '') -> Path:
        """
        Download execution assets.

        Args:
            - table_name (str): Name of the table (Execution_Assets or Execution_Metadata)
            - file_rid (str): Resource Identifier (RID) of the file.
            - execution_rid (str): Resource Identifier (RID) of the current execution.
            - dest_dir (str): Destination directory for the downloaded assets.

        Returns:
        - Path: Path to the downloaded asset.

        Raises:
        - DerivaMLException: If there is an issue downloading the assets.

        """
        table = self.resolve_rid(file_rid).table
        if not self.is_asset(table):
            raise DerivaMLException(f'Table {table} is not an asset table.')

        pb = self.pathBuilder
        ml_schema_path = pb.schemas[self.ml_schema]
        tpath = pb.schemas[table.schema.name].tables[table.name]
        file_metadata = list(tpath.filter(tpath.RID == file_rid).entities())[0]
        file_url = file_metadata['URL']
        file_name = file_metadata['Filename']
        try:
            self.update_status(Status.running, f'Downloading {table.name}...')
            file_path = self.download_asset(file_url, str(dest_dir) + '/' + file_name)
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error, execution_rid)
            raise DerivaMLException(f'Failed to download the file {file_rid}. Error: {error}')

        if execution_rid != '':
            ass_table = table.name + '_Execution'
            ass_table_path = ml_schema_path.tables[ass_table]
            exec_file_exec_entities = ass_table_path.filter(ass_table_path.columns[table.name] == file_rid).entities()
            exec_list = [e['Execution'] for e in exec_file_exec_entities]
            if execution_rid not in exec_list:
                tpath = pb.schemas[self.ml_schema].tables[ass_table]
                tpath.insert([{table.name: file_rid, 'Execution': execution_rid}])
        self.update_status(Status.running, f'Successfully download {table.name}...')
        return Path(file_path)

    def _clean_folder_contents(self, folder_path: Path, execution_rid: RID):
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

    def _update_execution_metadata_table(self, execution_rid, assets: dict[str, FileUploadState]) -> None:
        """
        Upload execution metadata at working_dir/Execution_metadata.

        Args:
        - execution_rid (str): Resource Identifier (RID) of the execution.

        Raises:
        - DerivaMLException: If there is an issue uploading the metadata.

        """
        ml_schema_path = self.pathBuilder.schemas[self.ml_schema]
        a_table = list(self.model.schemas[self.ml_schema].tables['Execution_Metadata'].find_associations())[0].name

        def asset_rid(asset) -> str:
            return asset.state == UploadState.success and asset.result and asset.result['RID']

        entities = [{'Execution_Metadata': rid, 'Execution': execution_rid} for asset in assets.values() if
                    (rid := asset_rid(asset))]
        ml_schema_path.tables[a_table].insert(entities)

    def _update_feature_table(self,
                              target_table: str,
                              feature_name: str,
                              feature_file: str | Path,
                              uploaded_files: dict[str, FileUploadState]) -> None:

        asset_columns = [c.name for c in self.feature_record_class(target_table, feature_name).feature.asset_columns]
        feature_table = self.feature_record_class(target_table, feature_name).feature.feature_table.name

        def map_path(e):
            # Go through the asset columns and replace the file name with the RID for the uploaded file.
            for c in asset_columns:
                e[c] = asset_map[e[c]]
            return e

        # Create a map between a file name that appeared in the file to the RID of the uploaded file.
        asset_map = {file: asset.result['RID']
                     for file, asset in uploaded_files.items() if asset.state == UploadState.success and asset.result}
        with open(feature_file, 'r') as feature_values:
            entities = [map_path(e) for e in csv.DictReader(feature_values)]
        self.domain_path.tables[feature_table].insert(entities)

    def _download_dataset_bag(self, dataset_rid: RID | str) -> tuple[Path, RID]:
        """
        Given a RID to a dataset, or a MINID to an existing bag, download the bag file, extract it and validate
        that all the metadata is correct

        :param dataset_rid: The RID of a dataset or a minid to an existing bag.
        :return: the location of the unpacked and validated dataset bag and the RID of the bag
        """
        if not any([dataset_rid == ds['RID'] for ds in self.find_datasets()]):
            raise DerivaMLException(f'RID {dataset_rid} is not a dataset')

        with TemporaryDirectory() as tmp_dir:
            if dataset_rid.startswith('minid'):
                # If provided a MINID, use the MINID metadata to get the checksum and download the bag.
                r = requests.get(f"https://identifiers.org/{dataset_rid}", headers={'accept': 'application/json'})
                metadata = r.json()['metadata']
                dataset_rid = metadata['Dataset_RID'].split('@')[0]
                checksum_value = ""
                for checksum in r.json().get('checksums', []):
                    if checksum.get('function') == 'sha256':
                        checksum_value = checksum.get('value')
                        break
                archive_path = fetch_single_file(dataset_rid, tmp_dir)
            else:
                # We are given the RID to a dataset, so we are going to have to export as a bag and place into
                # local file system.  The first step is to generate a downloadspec to create the bag, put the sped
                # into a local file and then use the downloader to create and download the desired bdbag.
                spec_file = f'{tmp_dir}/download_spec.json'
                with open(spec_file, 'w', encoding="utf-8") as ds:
                    json.dump(generate_dataset_download_spec(self.model), ds)
                downloader = GenericDownloader(
                    server={"catalog_id": self.catalog_id, "protocol": "https", "host": self.host_name},
                    config_file=spec_file,
                    output_dir=tmp_dir,
                    envars={"Dataset_RID": dataset_rid})
                result = downloader.download()
                archive_path = list(result.values())[0]["local_path"]
                checksum_value = compute_file_hashes(archive_path, hashes=['sha256'])['sha256'][0]

            # Check to see if we have an existing idempotent materialization of the desired bag. If so, then just reuse
            # it.  If not, then we need to extract the contents of the archive into our cache directory.
            bag_dir = self.cache_dir / f'{dataset_rid}_{checksum_value}'
            if bag_dir.exists():
                bag_path = (bag_dir / f'Dataset_{dataset_rid}').as_posix()
            else:
                bag_dir.mkdir(parents=True, exist_ok=True)
                bag_path = bdb.extract_bag(archive_path, bag_dir)
            bdb.validate_bag_structure(bag_path)
            return Path(bag_path), dataset_rid

    def _update_execution_asset_table(self, execution_rid: RID, assets: dict[str, FileUploadState]) -> None:
        """
        Assets associated with an execution must be linked to an execution entity after they are uploaded into
        the catalog. This routine takes a list of uploaded assets and makes that association.
        :param execution_rid:
        :param assets:
        :return:
        """
        ml_schema_path = self._ml_object.pathBuilder.schemas[self._ml_object.ml_schema]
        asset_exec_entities = ml_schema_path.Execution_Assets_Execution.filter(
            ml_schema_path.Execution_Assets_Execution.Execution == execution_rid).entities()
        existing_assets = {e['Execution_Assets'] for e in asset_exec_entities}

        # Now got through the list of recently added assets, and add an entry for this asset if it
        # doesn't already exist.
        def asset_rid(asset) -> str:
            return asset.state == UploadState.success and asset.result and asset.result['RID']

        entities = [{'Execution_Assets': rid, 'Execution': execution_rid}
                    for asset in assets.values() if (rid := asset_rid(asset)) not in existing_assets]
        ml_schema_path.Execution_Assets_Execution.insert(entities)
    @property
    def _execution_metadata_dir(self) -> Path:
        """
        Return a pathlib Path to the execution metadata directory. Files placed in this directory will be uploaded
        to the catalog by the execution_upload method in an execution object.

        :return:
        """
        return execution_metadata_dir(self.working_dir, exec_rid=self.execution_rid, metadata_type='')

    def execution_metadata_path(self, metadata_type: str) -> Path:
        """
        Return a pathlib Path to the directory in which to place files of type metadata_type.  These files
        are uploaded to the catalog as part of the execution of the upload_execution method in DerivaML.

        :param metadata_type:  Type of metadata to be uploaded.  Must be a term in Metadata_Type controlled vocabulary.
        :return: Path to the directory in which to place files of type metadata_type.
        """
        self._ml_object.lookup_term(MLVocab.execution_metadata_type, metadata_type)   # Make sure metadata type exists.
        return execution_metadata_dir(self.working_dir, exec_rid=self.execution_rid, metadata_type=metadata_type)

    @property
    def _execution_assets_dir(self) -> Path:
        """
        Return a pathlib Path to the directory in which to place directories for execution_assets.
        :return:
        """
        return execution_assets_dir(self.working_dir, exec_rid=self.execution_rid, asset_type='')

    def execution_assets_path(self, asset_type: str) -> Path:
        """
        Return a pathlib Path to the directory in which to place files for the specified execution_asset type. These
         files are uploaded as part of the upload_execution method in DerivaML class.

        :param asset_type: Type of asset to be uploaded.  Must be a term in Asset_Type controlled vocabulary.
        :return: Path in which to place asset files.
        """
        return execution_assets_dir(self.working_dir, exec_rid=self.execution_rid, asset_type=asset_type)

    @property
    def execution_root(self) -> Path:
        """
        Return the root path to all execution specific files.
        :return:
        """
        return execution_root(self.working_dir, self.execution_rid)

    @property
    def feature_root(self) -> Path:
        """
        The root path to all execution specific files.
        :return:
        """
        return feature_root(self.working_dir, self.execution_rid)

    def feature_paths(self, table:  Table | str, feature_name: str) -> tuple[Path, dict[str, Path]]:
        """
        Return the file path of where to place feature values, and assets for the named feature and table. A side
        effect of calling this routine is that the directories in which to place th feature values and assets will be
        created
        :param table:
        :param feature_name:
        :return: A tuple whose first element is the path for the feature values and whose second element is a dictionary
        of associated asset table names and corresponding paths.**
        """
        feature = self._ml_object.lookup_feature(table, feature_name)

        tpath = feature_value_path(self.working_dir,
                                        schema=self._ml_object.domain_schema,
                                        target_table=feature.target_table.name,
                                        feature_name=feature_name,
                                        exec_rid=self.execution_rid)
        asset_paths = {
            asset_table.name: feature_asset_dir(self.working_dir,
                                        exec_rid=self.execution_rid,
                                        schema=self._ml_object.domain_schema,
                                        target_table=feature.target_table.name,
                                        feature_name=feature_name,
                                        asset_table=asset_table.name)
                for asset_table in feature.asset_columns
        }
        return tpath, asset_paths

    def table_path(self, table: str) -> Path:
        """
        Return a local file path in which to place a CSV to add values to a table on upload.

        :param table: Name of table to be uploaded.
        :return: Pathlib path to the file in which to place table values.
        """
        return table_path(self.working_dir,
                                 schema=self._ml_object.domain_schema,
                                 table=table)

    def asset_directory(self, table: str) -> Path:
        """
        Return a local file path in which to place a files for an asset table.  This needs to be kept in sync with
        bulk_upload specification.

        :param table: Name of the asset table to be uploaded.
        :return: Pathlib path to the directory in which to place asset files.
        """
        return asset_dir(prefix=self.working_dir,
                                schema=self._ml_object.domain_schema,
                                asset_table=table)

    def write_feature_file(self, features: Iterator[FeatureRecord] | Iterable[FeatureRecord]) -> None:
        """
        Given a collection of Feature records, write out a CSV file in the appropriate assets directory so that this
        feature gets uploaded when the execution is complete.

        :param features: Iterable or Iterator of Feature records to write.
        """
        features = features if isinstance(features, Iterator) else iter(features)
        first_row = next(features)
        feature = first_row.feature
        csv_path, _ = self.feature_paths(feature.target_table.name, feature.feature_name)

        fieldnames = {'Execution', 'Feature_Name', feature.target_table.name}
        fieldnames |= {f.name for f in feature.feature_columns}

        with open(csv_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(first_row.model_dump())
            for feature in features:
                writer.writerow(feature.model_dump())


    def __str__(self):
        items = [
            f"caching_dir: {self.caching_dir}",
            f"working_dir: {self.working_dir}",
            f"execution_rid: {self.execution_rid}",
            f"workflow_rid: {self.workflow_rid}",
            f"bag_paths: {self.bag_paths}",
            f"asset_paths: {self.asset_paths}",
            f"configuration: {self.configuration}",
        ]
        return "\n".join(items)

class DerivaMlExec:
    """
    Context manager for managing DerivaML execution.  Provides status updates.

    Args:
    - catalog_ml: Instance of DerivaML class.
    - execution_rid (str): Execution resource identifier.

    """

    def __init__(self, execution: Execution):
        self.configuration = execution
        self.execution_rid = execution.execution_rid
        self.start_time = datetime.now()
        self.uploaded_assets = None

    def __enter__(self):
        """
        Method invoked when entering the context.

        Returns:
        - self: The instance itself.

        """
        self.configuration.update_status(Status.running,'Start ML algorithm.')
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """
         Method invoked when exiting the context.

         Args:
         - exc_type: Exception type.
         - exc_value: Exception value.
         - exc_tb: Exception traceback.

         Returns:
         - bool: True if execution completed successfully, False otherwise.

         """
        if not exc_type:
            self.configuration.update_status(Status.running,'Successfully run Ml.')
            self.configuration.execution_end(self.execution_rid)
        else:
            self.configuration.update_status(Status.failed,
                                          f'Exception type: {exc_type}, Exception value: {exc_value}')
            logging.error(f'Exception type: {exc_type}, Exception value: {exc_value}, Exception traceback: {exc_tb}')
            return False