import getpass
import hashlib
import json
import logging
import os
import re
import shutil
from copy import deepcopy
from datetime import datetime
from enum import Enum
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Sequence, Optional, Any, NewType

import deriva.core.datapath as datapath
import pandas as pd
import pkg_resources
import requests
from bdbag import bdbag_api as bdb
from deriva.core import ErmrestCatalog, get_credential, format_exception, urlquote, DEFAULT_SESSION_CONFIG
from deriva.core.ermrest_catalog import ResolveRidResult
from deriva.core.ermrest_model import Table
from deriva.core.hatrac_store import HatracStore
from deriva.core.utils import hash_utils, mime_utils
from deriva.transfer.upload.deriva_upload import GenericUploader
from pydantic import BaseModel, ValidationError

from deriva_ml.execution_configuration import ExecutionConfiguration

RID = NewType("RID", str)
TableName = tuple[str, str]


class DerivaMLException(Exception):
    """
    Exception class specific to DerivaML module.

    Args:
    - msg (str): Optional message for the exception.

    """

    def __init__(self, msg=""):
        super().__init__(msg)
        self._msg = msg


class Status(Enum):
    """
    Enumeration class defining execution status.

    Attributes:
    - running: Execution is currently running.
    - pending: Execution is pending.
    - completed: Execution has been completed successfully.
    - failed: Execution has failed.

    """
    running = "Running"
    pending = "Pending"
    completed = "Completed"
    failed = "Failed"


class DatasetElement(BaseModel):
    association_table: tuple[str, str]
    dataset_column: str
    element_column: str


class DataSet:
    """
    A DerivaML dataset is a collection of RIDs.  It is represented in the library by a base dataset table and a set of
    binary assocation tables.  This class and associate methods are used to create and modify datasets.
    """

    def __init__(self, ml_base: "DerivaML"):
        self.model = ml_base.model
        self.domain_schema = ml_base.ml_schema_name
        self.dataset_schema = ml_base.ml_schema_name
        self.catalog = self.model.catalog
        self.pb = ml_base.pb
        self.dataset_table = self.model.table(self.dataset_schema, 'Dataset')
        self.element_types: dict[TableName, DatasetElement] = self._collect_element_types()

    def _collect_element_types(self) -> dict[TableName, DatasetElement]:
        # Get a list of all the tables that are connected to the dataset via a binary association table.
        # Go through all of the incoming foreign keys to the dataset.
        # For each incoming foreign key, look at the table on the other side and check to see if it has
        # two foreign keys, which would make it a binary association table.  Figure out which key is which, and
        # record
        element_types = {}
        for fk in self.dataset_table.referenced_by:
            # Is the incoming fkey from an binary association table.
            association_table = fk.table
            if len(fkeys := association_table.foreign_keys) == 2:
                # Figure out which of the two keys is pointing to the dataset, and get the table for the other key.
                dataset_fkey, element_fkey = (0, 1) if fkeys[0].table == self.dataset_table else (1, 0)
                element_table = (fkeys[element_fkey].pk_table.schema.name, fkeys[element_fkey].pk_table.name)
                element_types[element_table] = DatasetElement(
                    association_table=(association_table.schema.name, association_table.name),
                    dataset_column=fkeys[dataset_fkey].columns[0].name,
                    element_column=fkeys[element_fkey].columns[0].name
                )
        return element_types

    def _rid_table(self, rid_list: list[RID]) -> list[ResolveRidResult]:
        """
        Return a named tuple with information about the specified RID.
        :param rid_list:
        :return:
        """
        try:
            return [self.catalog.resolve_rid(rid) for rid in rid_list]
        except KeyError as _e:
            raise DerivaMLException(f'Invalid RID')

    def create_dataset(self, description: str, rid_list: list[RID]) -> RID:
        """
        Create a new dataset from the specified list of RIDs.
        :param description:  Description of the dataset.
        :param rid_list:  List of RIDs to include in the dataset.
        :return:
        """
        # Get the tables associated with every rid.  Check to make sure we are not trying to include an element
        # in the dataset for which there is not association table.
        rid_tables = self._rid_table(rid_list)
        rid_type_names = set((r.table.schema.name, r.table.name) for r in rid_tables)
        element_names = {t for t in self.element_types}
        if not (rid_type_names < element_names):
            raise DerivaMLException(f'RID type cannot be dataset element: {[r for r in rid_type_names]}')
        dataset_elements = {}

        pb = self.catalog.getPathBuilder()
        # Create the entry for the new dataset and get its RID.
        dataset_table_path = pb.schemas[self.dataset_table.schema.name].tables[self.dataset_table.name]
        dataset_rid = dataset_table_path.insert([{'Description': description}])[0]['RID']

        # Now go through every rid to be added to the data set and sort them based on what association table entries
        # need to be made.
        for r in rid_tables:
            table_name = (r.table.schema.name, r.table.name)
            association_table = self.element_types[table_name]
            dataset_elements.setdefault(association_table.association_table, []).append(
                {association_table.dataset_column: dataset_rid,
                 association_table.element_column: r.rid, }
            )

        # Now make the entries into the association tables.
        for assoc_table, entries in dataset_elements.items():
            pb.schemas[assoc_table[0]].tables[assoc_table[1]].insert(entries)
        return dataset_rid

    def dataset_rids(self, dataset_rid: RID) -> list[RID]:
        """
        Return a list of RIDs associated with a specific dataset.
        :param dataset_rid:
        :return:
        """
        rid_list = []
        pb = self.catalog.getPathBuilder()

        dataset_path = pb.schemas[self.dataset_table.schema.name].tables[self.dataset_table.name]
        dataset_exists = list(dataset_path.filter(dataset_path.columns['RID'] == dataset_rid).entities().fetch())

        if len(dataset_exists) != 1:
            raise DerivaMLException(f'Invalid RID: {dataset_rid}')

        # Look at each of the element types that might be in the dataset and get the list of rid for them from
        # the appropriate association table.
        for assoc_table in self.element_types.values():
            schema_path = pb.schemas[assoc_table['association_table'][0]]
            table_path = schema_path.tables[assoc_table['association_table'][1]]
            dataset_column = table_path.columns[assoc_table['dataset_column']]
            element_column = table_path.columns[assoc_table['element_column']]
            assoc_rids = table_path.filter(dataset_column == dataset_rid).attributes(element_column).fetch()
            rid_list.extend([e[assoc_table['element_column']] for e in assoc_rids])
        return rid_list

    def extend_dataset(self, dataset_rid: RID, rid_list: list[RID]):
        # Get
        pass

    def delete_dataset(self, dataset_rid: RID) -> None:
        """
        Delete a dataset from the catalog.
        :param dataset_rid:  RID of the dataset to delete.
        :return:
        """
        # Get association table entries for this dataset
        # Delete association table entries

        for assoc_table in self.element_types.values():
            schema_path = self.pb.schemas[assoc_table.association_table[0]]
            table_path = schema_path.tables[assoc_table.association_table[1]]
            dataset_column = table_path.columns[assoc_table.dataset_column]
            dataset_entries = table_path.filter(dataset_column == dataset_rid)
            if len(dataset_entries.entities().fetch(limit=1)) == 1:
                dataset_entries.delete()

        # Delete dataset.
        dataset_path = self.pb.schemas[self.dataset_table.schema.name].tables[self.dataset_table.name]
        dataset_path.filter(dataset_path.columns['RID'] == dataset_rid).delete()

    def add_dataset_association(self, element: str) -> Table:
        # Add table to map
        element_table = self.model.schemas[self.domain_schema].tables[element]
        assoc_table = self.model.schemas[self.domain_schema].create_association(self.dataset_table, element_table)
        self.element_types = self._collect_element_types()
        return assoc_table


class DerivaMlExec:
    """
    Context manager for managing DerivaML execution.

    Args:
    - catalog_ml: Instance of DerivaML class.
    - execution_rid (str): Execution resource identifier.

    """

    def __init__(self, catalog_ml, execution_rid: str):
        self.execution_rid = execution_rid
        self.catalog_ml = catalog_ml
        self.catalog_ml.start_time = datetime.now()
        self.uploaded_assets = None

    def __enter__(self):
        """
        Method invoked when entering the context.

        Returns:
        - self: The instance itself.

        """
        self.catalog_ml.update_status(Status.running,
                                      "Start ML algorithm.",
                                      self.execution_rid)
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
            self.catalog_ml.update_status(Status.running,
                                          "Successfully run Ml.",
                                          self.execution_rid)
            self.catalog_ml.execution_end(self.execution_rid)
        else:
            self.catalog_ml.update_status(Status.failed,
                                          f"Exception type: {exc_type}, Exception value: {exc_value}",
                                          self.execution_rid)
            logging.error(f"Exception type: {exc_type}, Exception value: {exc_value}, Exception traceback: {exc_tb}")
            return False


class Term(BaseModel):
    """
    Data model representing a controlled vocabulary term.

    Attributes:
    - name (str): The name of the term.
    - rid (str): The ID of the term in catalog.

    """
    name: str
    rid: str


class ConfigurationRecord(BaseModel):
    """
    Data model representing configuration records.

    Attributes:
    - vocabs (dict): Dictionary containing vocabulary terms with key as vocabulary table name,
    and values as a list of dict containing name, rid pairs.
    - execution_rid (str): Execution identifier in catalog.
    - workflow_rid (str): Workflow identifier in catalog.
    - bag_paths (list): List of paths to bag files.
    - assets_paths (list): List of paths to assets.
    - configuration_path (Path): Path to the configuration file.

    """
    caching_dir: Path
    working_dir: Path
    vocabs: dict[str, list[Term]]
    execution_rid: str
    workflow_rid: str
    bag_paths: list[Path]
    assets_paths: list[Path]
    configuration_path: Path

    class Config:
        frozen = True
        protected_namespaces = ()


class DerivaML:
    """
    Class for managing Machine Learning experiments with data and metadata stored in Deriva platform.

    Args:
    - hostname (str): Hostname of the Deriva server.
    - catalog_id (str): Catalog ID.
    - schema_name (str): Schema name.
    - cache_dir (str): Directory path for caching data.
    - working_dir (str): Directory path for storing temporary data.

    """

    def __init__(self, hostname: str, catalog_id: str, ml_schema: str,
                 cache_dir: str, working_dir: str,
                 model_version: str):
        self.credential = get_credential(hostname)
        self.catalog = ErmrestCatalog('https', hostname, catalog_id,
                                      self.credential,
                                      session_config=self._get_session_config())
        self.model = self.catalog.getCatalogModel()
        self.pb = self.catalog.getPathBuilder()
        self.host_name = hostname
        self.ml_schema_name = ml_schema

        self.catalog_id = catalog_id
        self.ml_schema = self.pb.schemas[ml_schema]
        self.configuration = None

        self.start_time = datetime.now()
        self.status = Status.pending.value
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            tdir = TemporaryDirectory(delete=False)
            self.cache_dir = Path(tdir.name)
        default_workdir = self.__class__.__name__ + "_working"
        if working_dir is not None:
            self.working_dir = Path(working_dir).joinpath(getpass.getuser(), default_workdir)
        else:
            self.working_dir = Path(os.path.expanduser('~')).joinpath(default_workdir)
        self.execution_assets_path = self.working_dir / "Execution_Assets/"
        self.execution_metadata_path = self.working_dir / "Execution_Metadata/"
        self.execution_assets_path.mkdir(parents=True, exist_ok=True)
        self.execution_metadata_path.mkdir(parents=True, exist_ok=True)
        self.version = model_version
        self.dataset = DataSet(self)

        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        if "dirty" in self.version:
            logging.info(f"Loading dirty model.  Consider commiting and tagging: {self.version}")

    @staticmethod
    def _get_session_config():
        session_config = DEFAULT_SESSION_CONFIG.copy()
        session_config.update({
            # our PUT/POST to ermrest is idempotent
            "allow_retry_on_all_methods": True,
            # do more retries before aborting
            "retry_read": 8,
            "retry_connect": 5,
            # increase delay factor * 2**(n-1) for Nth retry
            "retry_backoff_factor": 5,
        })
        return session_config

    def find_table_schema(self, table_name: str) -> str:
        schemas = self.pb.schemas.keys()
        schema = None
        for s in schemas:
            if table_name in self.pb.schemas[s].tables:
                schema = s
        if schema is None:
            raise DerivaMLException(f"The table {table_name} doesn't exist.")
        else:
            return schema

    def is_vocabulary(self, table_name: str) -> bool:
        """
        Check if a given table is a controlled vocabulary table.

        Args:
        - table_name (str): The name of the table.

        Returns:
        - bool: True if the table is a controlled vocabulary, False otherwise.

        """
        vocab_columns = {'NAME', 'URI', 'SYNONYMS', 'DESCRIPTION', 'ID'}
        schema = self.find_table_schema(table_name)
        try:
            table = self.model.schemas[schema].tables[table_name]
        except KeyError:
            raise DerivaMLException(f"The vocabulary table {table_name} doesn't exist.")
        return vocab_columns.issubset({c.name.upper() for c in table.columns})

    def _vocab_columns(self, table: Table):
        """
        Return the list of columns in the table that are control vocabulary terms.

        Args:
        - table (ermrest_model.Table): The table.

        Returns:
        - List[str]: List of column names that are control vocabulary terms.
        """
        return [fk.columns[0].name for fk in table.foreign_keys
                if len(fk.columns) == 1 and self.is_vocabulary(fk.pk_table)]

    @staticmethod
    def _add_record(table: datapath._TableWrapper,
                    record: dict[str, str],
                    unique_col: str,
                    exist_ok: bool = False) -> str:
        """
        Add a record to a table.

        Args:
        - table (datapath._TableWrapper): Table wrapper object.
        - record (dict): Record to be added.
        - unique_col (str): Name of the column need to be unique.
        - exist_ok (bool): Flag indicating whether to allow creation if the record already exists.

        Returns:
        - str: Resource Identifier (RID) of the added record.

        """
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

    def add_workflow(self, workflow_name: str, url: str, workflow_type: str,
                     version: str = "",
                     description: str = "") -> str:
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
        workflow_type_rid = self.lookup_term("Workflow_Type", workflow_type)
        checksum = self._get_checksum(url)
        workflow_rid = self._add_record(self.ml_schema.Workflow,
                                        {'URL': url,
                                         'Name': workflow_name,
                                         'Description': description,
                                         'Checksum': checksum,
                                         'Version': version,
                                         'Workflow_Type': workflow_type_rid},
                                        'URL',
                                        True)
        return workflow_rid

    def add_execution(self, workflow_rid: str = "", datasets: List[str] = None,
                      description: str = "") -> str:
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
        if workflow_rid:
            execution_rid = self.ml_schema.Execution.insert([{'Description': description,
                                                              'Workflow': workflow_rid}])[0]['RID']
        else:
            execution_rid = self.ml_schema.Execution.insert([{'Description': description}])[0]['RID']
        if datasets:
            self._batch_insert(self.ml_schema.Dataset_Execution,
                               [{"Dataset": d, "Execution": execution_rid} for d in datasets])
        return execution_rid

    def update_execution(self, execution_rid: str, workflow_rid: str = "", datasets: List[str] = None,
                         description: str = "") -> str:
        """
        Update an existing execution to build the linkage between the
        Execution table and the Workflow and Dataset table.

        Args:
        - execution_rid (str): Resource Identifier (RID) of the execution to update.
        - workflow_rid (str): Resource Identifier (RID) of the workflow.
        - datasets (List[str]): List of dataset identifiers.
        - description (str): Description of the execution.

        Returns:
        - str: Resource Identifier (RID) of the updated execution.

        """

        datasets = datasets or []
        self._batch_update(self.ml_schema.Execution,
                           [{"RID": execution_rid, "Workflow": workflow_rid, "Description": description}],
                           [self.ml_schema.Execution.Workflow, self.ml_schema.Execution.Description])
        if datasets:
            self._batch_insert(self.ml_schema.Dataset_Execution,
                               [{"Dataset": d, "Execution": execution_rid} for d in datasets])
        return execution_rid

    def add_term(self,
                 table_name: str,
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

        schema = self.find_table_schema(table_name)
        try:
            entities = self.pb.schemas[schema].tables[table_name].entities()
            entities_upper = [{key.upper(): value for key, value in e.items()} for e in entities]
            name_list = [e['NAME'] for e in entities_upper]
            term_rid = entities[name_list.index(name)]['RID']
        except ValueError:
            # Name is not in list of current terms
            col_map = {col.upper(): col for col in self.pb.schemas[schema].tables[table_name].columns.keys()}
            term_rid = self.pb.schemas[schema].tables[table_name].insert(
                [{col_map['NAME']: name, col_map['DESCRIPTION']: description, col_map['SYNONYMS']: synonyms}],
                defaults={col_map['ID'], col_map['URI']})[0]['RID']
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

        schema = self.find_table_schema(table_name)
        for term in self.pb.schemas[schema].tables[table_name].entities():
            term_upper = {key.upper(): value for key, value in term.items()}
            if term_name == term_upper['NAME'] or (term_upper['SYNONYMS'] and term_name in term_upper['SYNONYMS']):
                return term['RID']

        raise DerivaMLException(f"Term {term_name} is not in vocabulary {table_name}")

    def list_vocabularies(self):
        """
        Return a list of all the controlled vocabulary tables in the schema.

        Returns:
         - List[str]: A list of table names representing controlled vocabulary tables in the schema.

        """
        return [t for s in self.pb.schemas.keys() for t in self.pb.schemas[s].tables if self.is_vocabulary(t)]

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

        return pd.DataFrame(self.pb.schemas[self.find_table_schema(table_name)].tables[table_name].entities().fetch())

    def resolve_rid(self, rid: str) -> ResolveRidResult:
        """
        Return a named tuple with information about the specified RID.
        :param rid:
        :return:
        """
        try:
            return self.catalog.resolve_rid(rid, self.model, self.pb)
        except KeyError as _e:
            raise DerivaMLException(f'Invalid RID {rid}')

    def retrieve_rid(self, rid: RID) -> dict[str, Any]:
        """
        Return a dictionary that represents the values of the specified RID.
        :param rid:
        :return:
        """
        return self.resolve_rid(rid).datapath.entities().fetch()[0]

    def user_list(self) -> pd.DataFrame:
        """
        Return a DataFrame containing user information of current catalog.

        Returns:
        - pd.DataFrame: DataFrame containing user information.

        """
        users = self.pb.schemas['public']
        path = users.ERMrest_Client.path
        return pd.DataFrame(path.entities().fetch())[['ID', 'Full_Name']]

    @staticmethod
    def _batch_insert(table: datapath._TableWrapper, entities: Sequence[dict]):
        """
        Batch insert entities into a table.

        Args:
        - table (datapath._TableWrapper): Table wrapper object.
        - entities (Sequence[dict]): Sequence of entity dictionaries to insert.

        """
        it = iter(entities)
        while chunk := list(islice(it, 2000)):
            table.insert(chunk)

    @staticmethod
    def _batch_update(table: datapath._TableWrapper, entities: Sequence[dict],
                      update_cols: List[datapath._ColumnWrapper]):
        """
        Batch update entities in a table.

        Args:
        - table (datapath._TableWrapper): Table wrapper object.
        - entities (Sequence[dict]): Sequence of entity dictionaries to update, must include RID.
        - update_cols (List[datapath._ColumnWrapper]): List of columns to update.

        """
        it = iter(entities)
        while chunk := list(islice(it, 2000)):
            table.update(chunk, [table.RID], update_cols)

    @staticmethod
    def _get_checksum(url) -> str:
        """
        Get the checksum of a file from a URL.

        Args:
        - url: URL of the file.

        Returns:
        - str: Checksum of the file.

        Raises:
        - DerivaMLException: If the URL is invalid or the file cannot be accessed.

        """
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

    def materialize_bdbag(self, minid: str, execution_rid: str) -> tuple:
        """
        Materialize a BDBag.

        Args:
        - minid (str): Minimum viable identifier (minid) of the bag.
        - execution_rid (str): Resource Identifier (RID) of the execution.

        Returns:
        - tuple: Tuple containing the path to the bag and the RID of the associated dataset.

        Raises:
        - DerivaMLException: If there is an issue materializing the bag.

        """

        def fetch_progress_callback(current, total):
            self.update_status(Status.running,
                               f"Materializing bag: {current} of {total} file(s) downloaded.", execution_rid)
            return True

        def validation_progress_callback(current, total):
            self.update_status(Status.running,
                               f"Validating bag: {current} of {total} file(s) validated.", execution_rid)
            return True

        # request metadata
        r = requests.get(f"https://identifiers.org/{minid}", headers={'accept': 'application/json'})
        metadata = r.json()['metadata']
        dataset_rid = metadata['Dataset_RID'].split('@')[0]
        checksum_value = ""
        for checksum in r.json().get('checksums', []):
            if checksum.get('function') == 'sha256':
                checksum_value = checksum.get('value')
                break

        bag_dir = self.cache_dir / f"{dataset_rid}_{checksum_value}"
        bag_dir.mkdir(parents=True, exist_ok=True)
        validated_check = bag_dir / "validated_check.txt"
        bags = [str(item) for item in bag_dir.iterdir() if item.is_dir()]
        if not bags:
            bag_path = bdb.materialize(minid,
                                       bag_dir,
                                       fetch_callback=fetch_progress_callback,
                                       validation_callback=validation_progress_callback)
            validated_check.touch()
        else:
            is_bag = [bdb.is_bag(bag) for bag in bags]
            if sum(is_bag) != 1:
                raise DerivaMLException(f'Invalid bag directory: {bag_dir}')
            else:
                bag_path = bags[is_bag.index(True)]
                if not validated_check.exists():
                    bdb.materialize(bag_path,
                                    fetch_callback=fetch_progress_callback,
                                    validation_callback=validation_progress_callback)
                    validated_check.touch()
        return Path(bag_path), dataset_rid

    def download_asset(self, asset_url: str, dest_filename: str) -> str:
        """
        Download an asset from a URL.

        Args:
        - asset_url (str): URL of the asset.
        - dest_filename (str): Destination filename.

        Returns:
        - str: Path to the downloaded asset.

        Raises:
        - DerivaMLException: If there is an issue downloading the asset.

        """
        hs = HatracStore("https", self.host_name, self.credential)
        hs.get_obj(path=asset_url, destfilename=dest_filename)
        return dest_filename

    def upload_assets(self, assets_dir: str):
        """
        Upload assets from a directory.

        Args:
        - assets_dir (str): Directory containing the assets to upload.

        Returns:
        - dict: Results of the upload operation.

        Raises:
        - DerivaMLException: If there is an issue uploading the assets.

           """
        uploader = GenericUploader(server={"host": self.host_name, "protocol": "https", "catalog_id": self.catalog_id})
        uploader.getUpdatedConfig()
        uploader.scanDirectory(assets_dir)
        results = deepcopy(uploader.uploadFiles())
        uploader.cleanup()
        return results

    def update_status(self, new_status: Status, status_detail: str, execution_rid: str):
        """
        Update the status of an execution.

        Args:
        - new_status (Status): New status.
        - status_detail (str): Details of the status.
        - execution_rid (str): Resource Identifier (RID) of the execution.

        """
        self.status = new_status.value
        self._batch_update(self.ml_schema.Execution,
                           [{"RID": execution_rid, "Status": self.status, "Status_Detail": status_detail}],
                           [self.ml_schema.Execution.Status, self.ml_schema.Execution.Status_Detail])

    def download_execution_files(self, table_name: str, file_rid: str, execution_rid="", dest_dir: str = "") -> Path:
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
        table = self.ml_schema.tables[table_name]
        file_metadata = table.filter(table.RID == file_rid).entities()[0]
        file_url = file_metadata['URL']
        file_name = file_metadata['Filename']
        try:
            self.update_status(Status.running, f"Downloading {table_name}...", execution_rid)
            file_path = self.download_asset(file_url, str(dest_dir) + '/' + file_name)
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error, execution_rid)
            raise DerivaMLException(f"Failed to download the file {file_rid}. Error: {error}")

        if execution_rid != '':
            ass_table = self.ml_schema.tables[table_name + '_Execution']
            exec_file_exec_entities = ass_table.filter(ass_table.columns[table_name] == file_rid).entities()
            exec_list = [e['Execution'] for e in exec_file_exec_entities]
            if execution_rid not in exec_list:
                self._batch_insert(ass_table, [{table_name: file_rid, "Execution": execution_rid}])

        self.update_status(Status.running, f"Successfully download {table_name}...", execution_rid)
        return Path(file_path)

    def upload_execution_configuration(self, config_file: str, desc: str):
        """
        Upload execution configuration to Execution_Metadata table with Execution Metadata Type = Execution_Config.

        Args:
        - config_file (str): Path to the configuration file.
        - desc (str): Description of the configuration.

        Raises:
        - DerivaMLException: If there is an issue uploading the configuration.

        """
        file_path = Path(config_file)
        file_name = file_path.name
        file_size = file_path.stat().st_size
        try:
            hs = HatracStore("https", self.host_name, self.credential)
            md5 = hash_utils.compute_file_hashes(config_file, ['md5'])['md5'][1]
            sanitized_filename = urlquote(re.sub("[^a-zA-Z0-9_.-]", "_", md5 + "." + file_name))
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
            execution_metadata_type_rid = self.lookup_term(self.ml_schema.Execution_Metadata_Type, "Execution Config")
            self._batch_insert(self.ml_schema.Execution_Metadata,
                               [{"URL": hatrac_uri,
                                 "Filename": file_name,
                                 "Length": file_size,
                                 "MD5": md5,
                                 "Description": desc,
                                 "Execution_Metadata_Type": execution_metadata_type_rid}])
        except Exception as e:
            error = format_exception(e)
            raise DerivaMLException(
                f"Failed to update Execution_Asset table with configuration file metadata. Error: {error}")

    def upload_execution_metadata(self, execution_rid: str):
        """
        Upload execution metadata at working_dir/Execution_metadata.

        Args:
        - execution_rid (str): Resource Identifier (RID) of the execution.

        Raises:
        - DerivaMLException: If there is an issue uploading the metadata.

        """
        self.update_status(Status.running, "Uploading assets...", execution_rid)
        try:
            results = self.upload_assets(str(self.execution_metadata_path))
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error, execution_rid)
            raise DerivaMLException(
                f"Fail to upload the files in {self.execution_metadata_path}"
                f" to Execution_Metadata table. Error: {error}")

        else:
            meta_exec_entities = self.ml_schema.Execution_Metadata_Execution.filter(
                self.ml_schema.Execution_Metadata_Execution.Execution == execution_rid).entities()
            meta_list = [e['Execution_Metadata'] for e in meta_exec_entities]
            entities = []
            for metadata in results.values():
                if metadata["State"] == 0 and metadata["Result"] is not None:
                    rid = metadata["Result"].get("RID")
                    if (rid is not None) and (rid not in meta_list):
                        entities.append({"Execution_Metadata": rid, "Execution": execution_rid})
        self._batch_insert(self.ml_schema.Execution_Metadata_Execution, entities)

        return results

    def upload_execution_assets(self, execution_rid: str) -> dict:
        """
        Upload execution assets at working_dir/Execution_assets.

        Args:
        - execution_rid (str): Resource Identifier (RID) of the execution.

        Returns:
        - dict: Results of the upload operation.

        Raises:
        - DerivaMLException: If there is an issue uploading the assets.

        """
        results = {}
        for folder_path in self.execution_assets_path.iterdir():
            self.update_status(Status.running, f"Uploading assets {folder_path}...", execution_rid)
            if folder_path.is_dir():
                try:
                    result = self.upload_assets(str(folder_path))
                    self.update_status(Status.running, "Uploading assets...", execution_rid)
                except Exception as e:
                    error = format_exception(e)
                    self.update_status(Status.failed, error, execution_rid)
                    raise DerivaMLException(
                        f"Fail to upload the files in {folder_path} to Execution_Assets table. Error: {error}")
                else:
                    asset_exec_entities = self.ml_schema.Execution_Assets_Execution.filter(
                        self.ml_schema.Execution_Assets_Execution.Execution == execution_rid).entities()
                    assets_list = [e['Execution_Assets'] for e in asset_exec_entities]
                    entities = []
                    for asset in result.values():
                        if asset["State"] == 0 and asset["Result"] is not None:
                            rid = asset["Result"].get("RID")
                            if (rid is not None) and (rid not in assets_list):
                                entities.append({"Execution_Assets": rid, "Execution": execution_rid})
                    self._batch_insert(self.ml_schema.Execution_Assets_Execution, entities)
                    results[str(folder_path)] = result
        return results

    def execution_end(self, execution_rid: str) -> None:
        """
        Finish the execution and update the duration and status of execution.

        Args:
        - execution_rid (str): Resource Identifier (RID) of the execution.

        Returns:
        - dict: Uploaded assets with key as assets' suborder name,
        values as an ordered dictionary with RID and metadata in the Execution_Assets table."

        """
        duration = datetime.now() - self.start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration = f'{round(hours, 0)}H {round(minutes, 0)}min {round(seconds, 4)}sec'

        self.update_status(Status.running, "Algorithm execution ended.", execution_rid)
        self._batch_update(self.ml_schema.Execution, [{"RID": execution_rid, "Duration": duration}],
                           [self.ml_schema.Execution.Duration])

    def execution_init(self, configuration_rid: str) -> ConfigurationRecord:
        """
        Initialize the execution by a configuration file in the Execution_Metadata table.
        Setup working directory and download all the assets and data.

        Args:
        - configuration_rid (str): Resource Identifier (RID) of the configuration.

        Returns:
        - ConfigurationRecord: Configurations' RID including Workflow, Execution, bag_paths(data directory),
        assets_paths(model directory), and vocabs (dict of controlled vocabularies).

        Raises:
        - DerivaMLException: If there is an issue initializing the execution.

        """
        # Download configuration json file
        configuration_path = self.download_execution_files('Execution_Metadata', configuration_rid,
                                                           dest_dir=str(self.execution_metadata_path))
        with open(configuration_path, 'r') as file:
            configuration = json.load(file)
        # Check input configuration
        try:
            self.configuration = ExecutionConfiguration.model_validate(configuration)
            logging.info("Configuration validation successful!")
        except ValidationError as e:
            raise DerivaMLException(f"configuration validation failed: {e}")
        # Insert Execution
        execution_rid = self.add_execution(description=self.configuration.execution.description)
        # Insert terms
        self.update_status(Status.running, "Inserting tags... ", execution_rid)
        vocabs = {}
        for term in configuration.get("workflow_terms"):
            term_rid = self.add_term(table_name=term["term"],
                                     name=term["name"],
                                     description=term["description"],
                                     exist_ok=True)
            term_records = vocabs.get(term["term"], [])
            term_records.append(Term(name=term["name"], rid=term_rid))
            vocabs[term["term"]] = term_records
        # Materialize bdbag
        dataset_rids = []
        bag_paths = []
        for url in self.configuration.bdbag_url:
            self.update_status(Status.running, f"Inserting bag {url}... ", execution_rid)
            bag_path, dataset_rid = self.materialize_bdbag(url, execution_rid)
            dataset_rids.append(dataset_rid)
            bag_paths.append(bag_path)
        # Insert workflow
        self.update_status(Status.running, "Inserting workflow... ", execution_rid)
        try:
            workflow_rid = self.add_workflow(self.configuration.workflow.name,
                                             self.configuration.workflow.url,
                                             self.configuration.workflow.workflow_type,
                                             self.configuration.workflow.version,
                                             self.configuration.workflow.description)
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error, execution_rid)
            raise DerivaMLException(f"Failed to insert workflow. Error: {error}")
        # Update execution info
        execution_rid = self.update_execution(execution_rid, workflow_rid, dataset_rids,
                                              self.configuration.execution.description)
        self.update_status(Status.running, "Execution created ", execution_rid)

        # Download model
        self.update_status(Status.running, "Downloading models ...", execution_rid)
        assets_paths = [self.download_execution_files('Execution_Assets', m, execution_rid,
                                                      dest_dir=str(self.execution_assets_path))
                        for m in self.configuration.models]
        configuration_records = ConfigurationRecord(
            caching_dir=self.cache_dir,
            working_dir=self.working_dir,
            execution_rid=execution_rid,
            workflow_rid=workflow_rid,
            bag_paths=bag_paths,
            vocabs=vocabs,
            assets_paths=assets_paths,
            configuration_path=configuration_path)
        # save runtime env
        runtime_env_file = str(self.execution_metadata_path) + '/Runtime_Env-python_environment_snapshot.txt'
        with open(runtime_env_file, 'w') as file:
            for package in pkg_resources.working_set:
                file.write(str(package) + "\n")
        self.start_time = datetime.now()
        self.update_status(Status.running, "Initialize status finished.", execution_rid)
        return configuration_records

    def execution(self, execution_rid: str) -> DerivaMlExec:
        """
        Start the execution by initializing the context manager DerivaMlExec.

        Args:
        - execution_rid (str): Resource Identifier (RID) of the execution.

        Returns:
        - DerivaMlExec: Execution object.

        """
        return DerivaMlExec(self, execution_rid)

    def _clean_folder_contents(self, folder_path: Path, execution_rid: str):
        try:
            shutil.rmtree(folder_path)
        except OSError as e:
            error = format_exception(e)
            self.update_status(Status.failed, error, execution_rid)

    def execution_upload(self, execution_rid: str, clean_folder: bool = True) -> dict[str, dict]:
        """
        Upload all the assets and metadata associated with the current execution.

        Args:
        - execution_rid (str): Resource Identifier (RID) of the execution.

        Returns:
        - dict: Uploaded assets with key as assets' suborder name,
        values as an ordered dictionary with RID and metadata in the Execution_Assets table."

        """
        try:
            uploaded_assets = self.upload_execution_assets(execution_rid)
            self.upload_execution_metadata(execution_rid)

            self.update_status(Status.completed,
                               "Successfully end the execution.",
                               execution_rid)
            if clean_folder:
                self._clean_folder_contents(self.execution_assets_path, execution_rid)
                self._clean_folder_contents(self.execution_metadata_path, execution_rid)
            return uploaded_assets
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error, execution_rid)
