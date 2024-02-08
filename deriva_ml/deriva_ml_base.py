from deriva.core import ErmrestCatalog, get_credential, format_exception, urlquote
from deriva.core.utils import hash_utils, mime_utils
import deriva.core.ermrest_model as ermrest_model
import deriva.core.datapath as datapath
from deriva.transfer.upload.deriva_upload import GenericUploader
from deriva.core.hatrac_store import HatracStore
from bdbag import bdbag_api as bdb
from deriva_ml.execution_configuration import ExecutionConfiguration
from datetime import datetime
from itertools import islice
import json
import re
import pandas as pd
from pathlib import Path
import requests
from typing import List, Sequence, Optional
from copy import deepcopy
from pydantic import ValidationError
from enum import Enum
from urllib.parse import urlparse
import hashlib
import pkg_resources


class DerivaMLException(Exception):
    def __init__(self, msg=""):
        super().__init__(msg)
        self._msg = msg


class Status(Enum):
    running = "Running"
    pending = "Pending"
    completed = "Completed"
    failed = "Failed"


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
        self.execution_product_path = Path("./Execution_Products/")
        self.execution_metadata_path = Path("./Execution_Metadata/")
        self.execution_product_path.mkdir(parents=True, exist_ok=True)
        self.execution_metadata_path.mkdir(parents=True, exist_ok=True)

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
            name_list = [e[unique_col] for e in entities]
            record_rid = entities[name_list.index(record[unique_col])]['RID']
        except ValueError:
            record_rid = table.insert([record])[0]['RID']
        else:
            if not exist_ok:
                raise DerivaMLException(
                    f"{record[unique_col]} existed with RID {entities[name_list.index(record[unique_col])]['RID']}")
        return record_rid

    def add_workflow(self, workflow_name: str, url: str, workflow_type : str,
                     version: str = "",
                     description: str = "",
                     exist_ok: bool = False) -> str:

        workflow_type_rid = self.lookup_term("Workflow_Type", workflow_type)
        checksum = self._get_checksum(url)
        workflow_rid = self.add_record(self.schema.Workflow,
                                       {'URL': url,
                                        'Name': workflow_name,
                                        'Description': description,
                                        'Checksum': checksum,
                                        'Version': version,
                                        'Workflow_Type': workflow_type_rid},
                                        'URL', True)
        return workflow_rid

    def add_execution(self, workflow_rid: str, datasets: List[str],
                      description: str = "") -> str:
        execution_rid = self.schema.Execution.insert([{'Description': description,
                                                       'Workflow': workflow_rid}])[0]['RID']
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
    def _get_checksum(url) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()
        except Exception:
            raise DerivaMLException(f"Invalid URL: {url}")
        else:
            sha256_hash = hashlib.sha256()
            sha256_hash.update(response.content)
            checksum = 'SHA-256: ' + sha256_hash.hexdigest()
        return checksum

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

    def download_execution_product(self, product_rid: str, execution_rid="", dest_dir: str = "") -> str:
        asset_metadata = self.schema.Execution_Products.filter(self.schema.Execution_Products.RID ==  product_rid).entities()[0]
        asset_url = asset_metadata['URL']
        file_name = asset_metadata['Filename']
        try:
            file_path = self.download_asset(asset_url, str(dest_dir) + '/' + file_name)
            self.update_status(Status.running, "Downloading assets...", execution_rid)
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error, execution_rid)
            raise DerivaMLException(f"Faild to download the asset {product_rid}. Error: {error}")

        if execution_rid != '':
            exec_prod_exec_entities = self.schema.Execution_Products_Execution.filter(
                self.schema.Execution_Products_Execution.Execution_Products == product_rid).entities()
            exec_list = [e['Execution'] for e in exec_prod_exec_entities]
            if execution_rid not in exec_list:
                self._batch_insert(self.schema.Execution_Products_Execution,
                                   [{"Execution_Products":  product_rid, "Execution": execution_rid}])
        return file_path
    
    def download_execution_metadata(self, metadata_rid: str, execution_rid="", dest_dir: str = "") -> str:
        asset_metadata = self.schema.Execution_Metadata.filter(self.schema.Execution_Metadata.RID ==  metadata_rid).entities()[0]
        asset_url = asset_metadata['URL']
        file_name = asset_metadata['Filename']
        try:
            file_path = self.download_asset(asset_url, str(dest_dir) + '/' + file_name)
            self.update_status(Status.running, "Downloading assets...", execution_rid)
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error, execution_rid)
            raise DerivaMLException(f"Faild to download the asset {metadata_rid}. Error: {error}")

        if execution_rid != '':
            exec_metadata_exec_entities = self.schema.Execution_Metadata.filter(
                self.schema.Execution_Metadata.RID == metadata_rid).entities()
            exec_list = [e['Execution'] for e in exec_metadata_exec_entities]
            if execution_rid not in exec_list:
                self._batch_update(self.Execution_Metadata,
                                   [{"Execution": execution_rid}],
                                   [self.schema.Execution_Metadata.Execution])
        return file_path

    def upload_execution_configuration(self, config_file: str, desc: str):
        file_path = Path(config_file)
        file_name = file_path.name
        file_size = file_path.stat().st_size
        try:
            hs = HatracStore("https", self.host_name, self.credential)
            md5 = hash_utils.compute_file_hashes(config_file, ['md5'])['md5'][1]
            sanitized_filename = urlquote(re.sub("[^a-zA-Z0-9_.-]", "_",  md5 + "." + file_name))
            hatrac_path = f"/hatrac/execution_metadata/{sanitized_filename}"
            hatrac_uri = hs.put_obj(hatrac_path,
                                    config_file,
                                    md5=md5,
                                    content_type=mime_utils.guess_content_type(config_file),
                                    content_disposition="filename*=UTF-8''" + file_name)
        except Exception as e:
            error = format_exception(e)
            raise DerivaMLException(
                f"Failed to upload execution configuration file {config_file} to object store. Error: {error}")
        try:
            execution_metadata_type_rid = self.lookup_term(self.schema.Execution_Metadata_Type, "Execution Config")
            self._batch_insert(self.schema.Execution_Metadata,
                               [{"URL": hatrac_uri,
                                 "Filename": file_name,
                                 "Length": file_size,
                                 "MD5": md5,
                                 "Description": desc,
                                 "Execution_Metadata_Type": execution_metadata_type_rid}])  # TODO: Fix this by doing a RID lookup based on an input parameter and add that parameter to the function arguments
        except Exception as e:
            error = format_exception(e)
            raise DerivaMLException(
                f"Failed to update Execution_Asset table with configuration file metadata. Error: {error}")
    
    def upload_execution_metadata(self, execution_rid: str):
        try:
            results = self.upload_assets(str(self.execution_metadata_path))
            self.update_status(Status.running, "Uploading assets...", execution_rid)
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error, execution_rid)
            raise DerivaMLException(
                f"Fail to upload the files in {self.execution_metadata_path} to Executoin_Metadata table. Error: {error}")
        else:
            entities = []
            for asset in results.values():
                if asset["State"] == 0 and asset["Result"] is not None:
                    rid = asset["Result"].get("RID")
                    if rid is not None:
                        entities.append({"RID": rid, "Execution": execution_rid})
            self._batch_update(self.schema.Execution_Metadata,
                               entities,
                               [self.schema.Execution_Metadata.Execution])

    def upload_execution_products(self, execution_rid: str):
        try:
            results = self.upload_assets(str(self.execution_product_path))
            self.update_status(Status.running, "Uploading assets...", execution_rid)
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error, execution_rid)
            raise DerivaMLException(
                f"Fail to upload the files in {self.execution_product_path} to Executoin_Products table. Error: {error}")
        else:
            asset_exec_entities = self.schema.Execution_Products_Execution.filter(
                self.schema.Execution_Products_Execution.Execution == execution_rid).entities()
            assets_list = [e['Execution_Products'] for e in asset_exec_entities]
            entities = []
            for asset in results.values():
                if asset["State"] == 0 and asset["Result"] is not None:
                    rid = asset["Result"].get("RID")
                    if (rid is not None) and (rid not in assets_list):
                        entities.append({"Execution_Products": rid, "Execution": execution_rid})
            self._batch_insert(self.schema.Execution_Products_Execution, entities)
        return results

    def execution_end(self, execution_rid: str):
        self.upload_execution_products(execution_rid)
        self.upload_execution_metadata(execution_rid)

        duration = datetime.now() - self.start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration = f'{round(hours, 0)}H {round(minutes, 0)}min {round(seconds, 4)}sec'

        self.update_status(Status.completed, "Execution ended.", execution_rid)
        self._batch_update(self.schema.Execution, [{"RID": execution_rid, "Duration": duration}],
                           [self.schema.Execution.Duration])

    def execution_init(self, configuration_rid: str) -> dict:
        # Download configuration json file
        configuration_path = self.download_execution_metadata(metadata_rid=configuration_rid,
                                                              dest_dir=str(self.execution_metadata_path ))
        with open(configuration_path, 'r') as file:
            configuration = json.load(file)
        # check input configuration
        try:
            self.configuration = ExecutionConfiguration.model_validate(configuration)
            print("Configuration validation successful!")
        except ValidationError as e:
            raise DerivaMLException(f"configuration validation failed: {e}")
        configuration_records = {}
        workflow_rid = self.add_workflow(self.configuration.workflow.name,
                                         self.configuration.workflow.url,
                                         self.configuration.workflow.workflow_type,
                                         self.configuration.workflow.version,
                                         self.configuration.workflow.description,
                                         exist_ok=True)
        # Insert or return Execution
        execution_rid = self.add_execution(workflow_rid,
                                           self.configuration.dataset_rid, 
                                           self.configuration.execution.description)
        self.update_status(Status.running, "Inserting configuration... ", execution_rid)
        # Insert tags
        for tag in configuration.get("workflow_tags"):
            annot_tag_rid = self.add_term(table_name=tag["tag"],
                                          name=tag["name"],
                                          description=tag["description"],
                                          exist_ok=True)
            tag_records = configuration_records.get(tag["tag"], [])
            tag_records.append({"name": tag["name"], "RID": annot_tag_rid})
            configuration_records[tag["tag"]] = tag_records
        # Materialize bdbag
        bdb.configure_logging(force=True)
        bag_paths = [bdb.materialize(url) for url in self.configuration.bdbag_url]
        # download model
        model_paths = [self.download_execution_product(m, execution_rid, dest_dir=str(self.execution_product_path)) for m in
                       self.configuration.models]
        configuration_records.update(
            {"execution": execution_rid, "workflow": workflow_rid, "bag_paths": bag_paths,
             "model_paths": model_paths, 'configuration_path': configuration_path})
        # save runtime env
        runtime_env_file = str(self.execution_metadata_path)+'/Runtime_Env-python_environment_snapshot.txt'
        with open(runtime_env_file, 'w') as file:
            for package in pkg_resources.working_set:
                file.write(str(package) + "\n")
        self.start_time = datetime.now()
        return configuration_records

    def start_execution(self, execution_rid: str) -> DerivaMlExec:
        return DerivaMlExec(self, execution_rid, str(self.execution_product_path))

