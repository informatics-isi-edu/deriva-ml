from deriva.core import ErmrestCatalog, get_credential, format_exception
import deriva.core.ermrest_model as ermrest_model
import deriva.core.datapath as datapath
from deriva.transfer.upload.deriva_upload import GenericUploader
from deriva.core.hatrac_store import HatracStore
from bdbag import bdbag_api as bdb
from deriva_ml.execution_configuration import ExecutionConfiguration
from datetime import datetime
from itertools import islice
import json
import pandas as pd
from pathlib import Path
import requests
from typing import List, Sequence, Optional
from copy import deepcopy
from pydantic import ValidationError
from enum import Enum


class DerivaMLException(Exception):
    def __init__(self, msg=""):
        super().__init__(msg)
        self._msg = msg


class Status(Enum):
    running = "Running"
    pending = "Pending"
    completed = "Completed"
    failed = "Failed"


class DerivaML:
    def __init__(self, hostname: str, catalog_id: str, schema_name):
        self.credential = get_credential(hostname)
        self.catalog = ErmrestCatalog('https', hostname, catalog_id, self.credential)
        self.model = self.catalog.getCatalogModel()
        self.pb = self.catalog.getPathBuilder()
        self.host_name = hostname
        self.schema_name = schema_name
        self.catalog_id = catalog_id
        self.schema = self.pb.schemas[schema_name]
        self.configuration = None

        self.start_time = datetime.now()
        self.status = Status.pending.value
        self.download_path = Path("./download/")
        self.upload_path = Path("./ExecutionAssets/")
        self.download_path.mkdir(parents=True, exist_ok=True)
        self.upload_path.mkdir(parents=True, exist_ok=True)

    def is_vocabulary(self, table_name: str) -> bool:
        """
        Check if a given table is a controlled vocabulary table.

        Args:
        - table_name (str): The name of the table.

        Returns:
        - bool: True if the table is a controlled vocabulary, False otherwise.

        """
        vocab_columns = {'Name', 'URI', 'Synonyms', 'Description', 'ID'}
        try:
            table = self.model.schemas[self.schema_name].tables[table_name]
        except KeyError:
            raise DerivaMLException(f"The vocabulary table {table_name} doesn't exist.")
        return vocab_columns.issubset({c.name for c in table.columns})

    def _vocab_columns(self, table: ermrest_model.Table):
        """
        Return the list of columns in the table that are control vocabulary terms.

        Args:
        - table (ermrest_model.Table): The table.

        Returns:
        - List[str]: List of column names that are control vocabulary terms.
        """
        return [fk.columns[0].name for fk in table.foreign_keys
                if len(fk.columns) == 1 and self.is_vocabulary(fk.pk_table)]

    def add_record(self, table: datapath._TableWrapper,
                   record: dict[str, str],
                   unique_col: str,
                   exist_ok: bool = False) -> str:
        try:
            entities = table.entities()
            name_list = [e['Name'] for e in entities]
            record_rid = entities[name_list.index(record[unique_col])]['RID']
        except ValueError:
            record_rid = table.insert([record])[0]['RID']
        else:
            if not exist_ok:
                raise DerivaMLException(
                    f"{record[unique_col]} existed with RID {entities[name_list.index(record[unique_col])]['RID']}")
        return record_rid

    def add_process(self, process_name: str, process_tag_name: str = "", description: str = "",
                    github_owner: str = "", github_repo: str = "", github_file_path: str = "",
                    exist_ok: bool = False) -> str:
        """
        Add a new process to the catalog.

        Args:
        - process_name (str): Name of the new process.
        - github_url (str, optional): GitHub URL associated with the process.
        - process_tag (str, optional): Tag for the process.
        - description (str, optional): Description of the process.
        - github_checksum (str, optional): Checksum of the GitHub repository.
        - exists_ok (bool, optional):  Optional flag indicating whether to allow creation if the control
          vocabulary name already exists. Defaults to False.

        Returns:
        - str: RID (Record ID) of the newly created process.

        Raises:
        - Exception: If the process already exists and exists_ok is False.
        """
        process_tag_rid = self.lookup_term("Process_Tag", process_tag_name)

        github_metadata = self._github_metadata(github_owner, github_repo, github_file_path)
        process_rid = self.add_record(self.schema.Process,
                                      {'Github_URL': github_metadata["Github_URL"],
                                       'Name': process_name,
                                       'Process_Tag': process_tag_rid,
                                       'Description': description,
                                       'Github_Checksum': github_metadata["Github_Checksum"]},
                                      "Name", exist_ok)
        return process_rid

    def add_workflow(self, workflow_name: str, description: str = "",
                     github_owner: str = "", github_repo: str = "", github_file_path: str = "",
                     process_list: Optional[List] = None,
                     exist_ok: bool = False) -> str:

        process_list = process_list or ""
        github_metadata = self._github_metadata(github_owner, github_repo, github_file_path)
        workflow_rid = self.add_record(self.schema.Workflow,
                                       {'Github_URL': github_metadata["Github_URL"],
                                        'Name': workflow_name,
                                        'Description': description,
                                        'Github_Checksum': github_metadata["Github_Checksum"]},
                                       'Name', exist_ok)
        proc_work_entities = self.schema.Workflow_Process.filter(
            self.schema.Workflow_Process.Workflow == workflow_rid).entities()
        proc_work_list = [e['Process'] for e in proc_work_entities]
        asso_entities = [{"Process": p, "Workflow": workflow_rid} for p in process_list if p not in proc_work_list]
        self._batch_insert(self.schema.Workflow_Process, asso_entities)
        return workflow_rid

    def add_execution(self, execution_name: str, workflow_rid: str, datasets: List[str],
                      description: str = "", exist_ok: bool = False) -> str:
        execution_rid = self.add_record(self.schema.Execution,
                                        {'Name': execution_name,
                                         'Description': description,
                                         'Workflow': workflow_rid},
                                        "Name", exist_ok)
        self._batch_insert(self.schema.Dataset_Execution,
                           [{"Dataset": d, "Execution": execution_rid} for d in datasets])
        return execution_rid

    def add_term(self, table_name: str,
                 name: str,
                 description: str,
                 synonyms: Optional[List[str]] = None,
                 exist_ok: bool = False):
        """
        Creates a new control vocabulary term in the control vocabulary table.

        Args:
        - table_name (str): The name of the control vocabulary table.
        - name (str): The name of the new control vocabulary.
        - description (str): The description of the new control vocabulary.
        - synonyms (List[str]): Optional list of synonyms for the new control vocabulary. Defaults to an empty list.
        - exist_ok (bool): Optional flag indicating whether to allow creation if the control vocabulary name
          already exists. Defaults to False.

        Returns:
        - str: The RID of the newly created control vocabulary.

        Raises:
        - EyeAIException: If the control vocabulary name already exists and exist_ok is False.
        """
        synonyms = synonyms or []

        try:
            if not self.is_vocabulary(table_name):
                raise DerivaMLException(f"The table {table_name} is not a controlled vocabulary")
        except KeyError:
            raise DerivaMLException(f"The schema or vocabulary table {table_name} doesn't exist.")

        try:
            entities = self.schema.tables[table_name].entities()
            name_list = [e['Name'] for e in entities]
            term_rid = entities[name_list.index(name)]['RID']
        except ValueError:
            # Name is not in list of current terms
            term_rid = self.schema.tables[table_name].insert(
                [{'Name': name, 'Description': description, 'Synonyms': synonyms}],
                defaults={'ID', 'URI'})[0]['RID']
        else:
            # Name is list of current terms.
            if not exist_ok:
                raise DerivaMLException(f"{name} existed with RID {entities[name_list.index(name)]['RID']}")
        return term_rid

    def lookup_term(self, table_name: str, term_name: str) -> str:
        """
        Given a term name, return the RID of the associated term (or synonym).

        Args:
        - table_name (str): The name of the controlled vocabulary table.
        - term_name (str): The name of the term to look up.

        Returns:
        - str: The RID of the associated term or synonym.

        Raises:
        - EyeAIException: If the schema or vocabulary table doesn't exist, or if the term is not
          found in the vocabulary.

        """
        try:
            vocab_table = self.is_vocabulary(table_name)
        except KeyError:
            raise DerivaMLException(f"The schema or vocabulary table {table_name} doesn't exist.")

        if not vocab_table:
            raise DerivaMLException(f"The table {table_name} is not a controlled vocabulary")

        for term in self.schema.tables[table_name].entities():
            if term_name == term['Name'] or (term['Synonyms'] and term_name in term['Synonyms']):
                return term['RID']

        raise DerivaMLException(f"Term {term_name} is not in vocabuary {table_name}")

    def list_vocabularies(self):
        """
        Return a list of all of the controlled vocabulary tables in the schema.

        Returns:
         - List[str]: A list of table names representing controlled vocabulary tables in the schema.

        """
        return [t for t in self.schema.tables if self.is_vocabulary(t)]

    def list_vocabulary(self, table_name: str) -> pd.DataFrame:
        """
        Return the dataframe of terms that are in a vocabulary table.

        Args:
        - table_name (str): The name of the controlled vocabulary table.

        Returns:
        - pd.DataFrame: A DataFrame containing the terms in the specified controlled vocabulary table.

        Raises:
        - EyeAIException: If the schema or vocabulary table doesn't exist, or if the table is not
          a controlled vocabulary.
        """
        try:
            vocab_table = self.is_vocabulary(table_name)
        except KeyError:
            raise DerivaMLException(f"The schema or vocabulary table {table_name} doesn't exist.")

        if not vocab_table:
            raise DerivaMLException(f"The table {table_name} is not a controlled vocabulary")

        return pd.DataFrame(self.schema.tables[table_name].entities().fetch())

    def user_list(self) -> pd.DataFrame:
        """
        Return a DataFrame containing user information.
        """
        users = self.pb.schemas['public']
        path = users.ERMrest_Client.path
        return pd.DataFrame(path.entities().fetch())[['ID', 'Full_Name']]

    @staticmethod
    def _batch_insert(table: datapath._TableWrapper, entities: Sequence[dict]):
        it = iter(entities)
        while chunk := list(islice(it, 2000)):
            table.insert(chunk)

    @staticmethod
    def _batch_update(table: datapath._TableWrapper, entities: Sequence[dict],
                      update_cols: List[datapath._ColumnWrapper]):
        it = iter(entities)
        while chunk := list(islice(it, 2000)):
            table.update(chunk, [table.RID], update_cols)

    @staticmethod
    def _github_metadata(owner: str, repo: str, file_path: str) -> dict[str, str]:
        try:
            response = requests.get(f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}")
            response.raise_for_status()
        except Exception:
            raise DerivaMLException(f"Invalid GitHub repo for owner: {owner}, repo: {repo}, file_path: {file_path}")
        else:
            github_metadata = response.json()
            return {"Github_Checksum": github_metadata['sha'], "Github_URL": github_metadata["html_url"]}

    def download_asset(self, asset_url: str, destfilename: str):
        hs = HatracStore("https", self.host_name, self.credential)
        hs.get_obj(path=asset_url, destfilename=destfilename)
        return destfilename

    def upload_assets(self, assets_dir: str):
        uploader = GenericUploader(server={"host": self.host_name, "protocol": "https", "catalog_id": self.catalog_id})
        uploader.getUpdatedConfig()
        uploader.scanDirectory(assets_dir)
        results = deepcopy(uploader.uploadFiles())
        uploader.cleanup()
        return results

    def update_status(self, new_status: Status, status_detail: str, execution_rid: str):
        self.status = new_status.value
        self._batch_update(self.schema.Execution,
                           [{"RID": execution_rid, "Status": self.status, "Status_Detail": status_detail}],
                           [self.schema.Execution.Status, self.schema.Execution.Status_Detail])

    def download_execution_asset(self, asset_rid: str, execution_rid="", dest_dir: str = "") -> str:
        asset_metadata = self.schema.Execution_Asset.filter(self.schema.Execution_Asset.RID == asset_rid).entities()[0]
        asset_url = asset_metadata['URL']
        file_name = asset_metadata['Filename']
        try:
            file_path = self.download_asset(asset_url, str(dest_dir) + '/' + file_name)
            self.update_status(Status.running, "Downloading assets...", execution_rid)
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error, execution_rid)
            raise DerivaMLException(f"Faild to download the asset {asset_rid}. Error: {error}")

        if execution_rid != '':
            asset_exec_entities = self.schema.Execution_Asset_Execution.filter(
                self.schema.Execution_Asset_Execution.Execution_Asset == asset_rid).entities()
            exec_list = [e['Execution'] for e in asset_exec_entities]
            if execution_rid not in exec_list:
                self._batch_insert(self.schema.Execution_Asset_Execution,
                                   [{"Execution_Asset": asset_rid, "Execution": execution_rid}])
        return file_path

    def upload_execution_assets(self, execution_rid: str):
        try:
            results = self.upload_assets(str(self.upload_path))
            self.update_status(Status.running, "Uploading assets...", execution_rid)
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error, execution_rid)
            raise DerivaMLException(
                f"Fail to upload the files in {self.upload_path} to Executoin_Asset table. Error: {error}")
        else:
            asset_exec_entities = self.schema.Execution_Asset_Execution.filter(
                self.schema.Execution_Asset_Execution.Execution == execution_rid).entities()
            assets_list = [e['Execution_Asset'] for e in asset_exec_entities]
            entities = []
            for asset in results.values():
                if asset["State"] == 0 and asset["Result"] is not None:
                    rid = asset["Result"].get("RID")
                    if (rid is not None) and (rid not in assets_list):
                        entities.append({"Execution_Asset": rid, "Execution": execution_rid})
            self._batch_insert(self.schema.Execution_Asset_Execution, entities)
        return results

    def execution_end(self, execution_rid: str):
        self.upload_execution_assets(execution_rid)

        duration = datetime.now() - self.start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration = f'{round(hours, 0)}H {round(minutes, 0)}min {round(seconds, 4)}sec'

        self.update_status(Status.completed, "Execution ended.", execution_rid)
        self._batch_update(self.schema.Execution, [{"RID": execution_rid, "Duration": duration}],
                           [self.schema.Execution.Duration])

    def execution_init(self, configuration_rid: str) -> tuple:
        # Download configuration json file
        configuration_path = self.download_execution_asset(asset_rid=configuration_rid,
                                                           dest_dir=str(self.download_path))
        with open(configuration_path, 'r') as file:
            configuration = json.load(file)
        # check input configuration
        try:
            self.configuration = ExecutionConfiguration.model_validate_json(configuration)
            print("Configuration validation successful!")
        except ValidationError as e:
            raise DerivaMLException(f"configuration validation failed: {e}")
        configuration_records = {}
        # Insert processes
        process = []
        for proc in self.configuration.process:
            proc_rid = self.add_process(proc.name, proc.process_tag_name, proc.description,
                                        proc.owner, proc.repo, proc.file_path, exist_ok=True)
            process.append(proc_rid)
        # Insert or return Workflow
        workflow_rid = self.add_workflow(self.configuration.workflow.name,
                                         self.configuration.workflow.description,
                                         self.configuration.workflow.owner,
                                         self.configuration.workflow.repo,
                                         self.configuration.workflow.file_path,
                                         process, exist_ok=True)
        # Insert or return Execution
        execution_rid = self.add_execution(self.configuration.execution.name, workflow_rid,
                                           self.configuration.dataset_rid, self.configuration.execution.description)
        self.update_status(Status.running, "Inserting configuration... ", execution_rid)
        # build association: execution - configuration asset
        self._batch_insert(self.schema.Execution_Asset_Execution,
                           [{"Execution_Asset": configuration_rid, "Execution": execution_rid}])
        # Insert tags
        annot_tag = configuration.get("annotation_tag")
        if annot_tag is not None:
            annot_tag_rid = self.add_term(table_name='Annotation_Tag', name=annot_tag['name'],
                                          description=annot_tag['description'],
                                          synonyms=annot_tag['synonyms'], exist_ok=True)
            configuration_records['annotation_tag_rid'] = annot_tag_rid
        diag_tag = configuration.get("diagnosis_Tag")
        if diag_tag is not None:
            diag_tag_rid = self.add_term(table_name='Diagnosis_Tag', name=diag_tag['name'],
                                         description=diag_tag['description'],
                                         synonyms=diag_tag['synonyms'], exist_ok=True)
            configuration_records['diagnosis_tag_rid'] = diag_tag_rid
        # Materialize bdbag
        bdb.configure_logging(force=True)
        bag_paths = [bdb.materialize(url) for url in self.configuration.bdbag_url]
        # download model
        model_paths = [self.download_execution_asset(m, execution_rid, dest_dir=str(self.download_path)) for m in
                       self.configuration.models]
        configuration_records.update(
            {"execution": execution_rid, "workflow": workflow_rid, "process": process, "bag_paths": bag_paths,
             "model_paths": model_paths, 'configuration_path': configuration_path})
        self.start_time = datetime.now()
        return configuration_records, DerivaMlExec(self, execution_rid, str(self.upload_path))


class DerivaMlExec:
    def __init__(self, catalog_ml, execution_rid: str, assets_dir: str):
        self.execution_rid = execution_rid
        self.catalog_ml = catalog_ml
        self.assets_dir = assets_dir
        self.catalog_ml.start_time = datetime.now()

    def __enter__(self):
        return self.execution_rid

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(f"Exeption type: {exc_type}, Exeption value: {exc_value}, Exeption traceback: {exc_tb}")
        self.catalog_ml.execution_end(self.execution_rid)
        return True
