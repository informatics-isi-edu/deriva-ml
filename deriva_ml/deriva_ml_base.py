import getpass
import hashlib
import json
import logging
import os
import re
import shutil
import warnings
from datetime import datetime
from enum import Enum
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Sequence, Optional, Any, NewType, Iterable

import pandas as pd
import pkg_resources
import requests
from bdbag import bdbag_api as bdb
from deriva.core import ErmrestCatalog, get_credential, format_exception, urlquote, DEFAULT_SESSION_CONFIG
from deriva.core.datapath import DataPathException
from deriva.core.ermrest_catalog import ResolveRidResult
from deriva.core.ermrest_model import Table, Column, ForeignKey
from deriva.core.hatrac_store import HatracStore
from deriva.core.utils import hash_utils, mime_utils
from deriva.transfer.upload.deriva_upload import GenericUploader
from pydantic import BaseModel, ValidationError

from deriva_ml.execution_configuration import ExecutionConfiguration

RID = NewType("RID", str)

# We are going to use schema as a field name and this collides with method in pydantic base class
warnings.filterwarnings("ignore",
                        message='Field name "schema"',
                        category=Warning,
                        module='pydantic')


# For some reason, deriva-py doesn't use the proper enum class!!
class UploadState(str, Enum):
    success = "Success"
    failed = "Failed"
    pending = "Pending"
    running = "Running"
    paused = "Paused"
    aborted = "Aborted"
    cancelled = "Cancelled"
    timeout = "Timeout"


class BuiltinTypes(str, Enum):
    text = 'text',
    int2 = 'int2',
    jsonb = 'jsonb',
    float8 = 'float8',
    timestamp = 'timestamp',
    int8 = 'int8',
    boolean = 'boolean',
    json = 'json',
    float4 = 'float4',
    int4 = 'int4',
    timestamptz = 'timestamptz',
    date = 'date',
    text_array = 'text[]',
    int2_array = 'int2[]',
    jsonb_array = 'jsonb[]',
    float8_array = 'float8[]',
    timestamp_array = 'timestamp[]',
    int8_array = 'int8[]',
    boolean_array = 'boolean[]',
    json_array = 'json[]',
    float4_array = 'float4[]',
    int4_array = 'int4[]',
    timestamptz_array = 'timestamptz[]',
    date_array = 'date[]',
    ermrest_rid = 'ermrest_rid',
    ermrest_rcb = 'ermrest_rcb',
    ermrest_rmb = 'ermrest_rmb',
    ermrest_rct = 'ermrest_rct',
    ermrest_rmt = 'ermrest_rmt',
    markdown = 'markdown',
    longtext = 'longtext',
    ermrest_curie = 'ermrest_curie',
    ermrest_uri = 'ermrest_uri',
    color_rgb_hex = 'color_rgb_hex',
    serial2 = 'serial2',
    serial4 = 'serial4',
    serial8 = 'serial8'


class ColumnDefinitions(BaseModel):
    cname: str
    ctype: BuiltinTypes
    nullable: bool = True
    default: Any = None
    comment: Optional[str] = None
    acls: dict = {}
    acl_bindings: dict = {}
    annotations: dict = {}


class AssociatedTable(BaseModel, frozen=True, arbitrary_types_allowed=True):
    """
    Information from deriva-model for an association table, all in one place.
    This has a place to store additional attributes.
    """
    association_table: Table
    left_table: Table
    right_table: Table
    left_column: Column
    right_column: Column
    attributes: list[str] = []

    @property
    def linked_tables(self) -> set[Table]:
        """
        The linked tables in an association table returned as a set.
        :return: set of tables.
        """
        return {self.left_table, self.right_table}

    def __repr__(self):
        return f"<{self.association_table.name}>"


class FileUploadState(BaseModel):
    state: UploadState
    status: str
    result: Any


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
    Base class for ML operations on a Deriva catalog.  This class is intended to be used as a base class on which
    more domain specific interfaces are built.
    """
    def __init__(self, hostname: str, catalog_id: str, domain_schema: str,
                 cache_dir: str, working_dir: str,
                 model_version: str):
        """

        :param hostname: Hostname of the Deriva server.
        :param catalog_id: Catalog ID.
        :param domain_schema: Schema name for domain specific tables and relationships.
        :param cache_dir: Directory path for caching data.
        :param working_dir: Directory path for storing temporary data.
        :param model_version:
        """
        self.host_name = hostname
        self.catalog_id = catalog_id
        self.domain_schema = domain_schema
        self.version = model_version

        self.credential = get_credential(hostname)
        self.catalog = ErmrestCatalog('https', hostname, catalog_id,
                                      self.credential,
                                      session_config=self._get_session_config())
        self.model = self.catalog.getCatalogModel()
        self.pb = self.catalog.getPathBuilder()
        self.dataset_schema = domain_schema
        self.ml_schema_path = self.pb.schemas[self.domain_schema]
        self.dataset_table = self.model.table(self.dataset_schema, 'Dataset')
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
        """
        Given a table name, return the name of its schema.
        :param table_name:
        :return:
        """
        schemas = self.pb.schemas.keys()
        schema = None
        for s in schemas:
            if table_name in self.pb.schemas[s].tables:
                schema = s
        if schema is None:
            raise DerivaMLException(f"The table {table_name} doesn't exist.")
        else:
            return schema

    def find_association_tables(self,
                                table: Table | str,
                                target_table: Optional[Table | str] = None,
                                follow_naming_convention: bool = True) -> list[AssociatedTable]:
        """
        Find a table linking the two named tables.
        :param table: Table name that is considered to be on the "left" hand side of the association table.
        :param target_table:
        :param follow_naming_convention: If target table is not provided, use the association naming convention of
               t1_t2 to determine the right hand table in the association.  If False, assume that all foreign_key
               tables are potentially associated.
        :return: A list of association tables.
        """

        # Normalize input arguments.
        if isinstance(table, str):
            try:
                table = self.model.schemas[self.domain_schema].tables[table]
            except KeyError:
                raise DerivaMLException(f"The table {table} doesn't exist.")

        if isinstance(target_table, str):
            try:
                target_table = target_table and self.model.schemas[self.domain_schema].tables[target_table]
            except KeyError:
                raise DerivaMLException(f"The table {target_table.table} doesn't exist.")

        linked_tables = []
        for assoc in table.find_associations():
            a_table = assoc.table
            for other_fkey in assoc.other_fkeys:
                left_table, right_table = assoc.self_fkey.pk_table, other_fkey.pk_table
                left_column, right_column = assoc.self_fkey.columns[0], other_fkey.columns[0]
                if target_table and right_table != target_table:
                    continue

                # Check to see if association table follows standard naming convention.
                if follow_naming_convention and not target_table:
                    if f"{right_table.name}_{left_table.name}" == a_table.name:
                        left_table, right_table = right_table, left_table
                        left_column, right_column = right_column, left_column
                    elif f"{left_table.name}_{right_table.name}" != a_table.name:
                        # If we are following the naming convention, only use tables that are called out in the name.
                        continue

                skip_columns = ['RID', 'RCT', 'RMT', 'RCB', 'RMB', left_column.name, right_column.name]
                linked_tables.append(AssociatedTable(
                    association_table=a_table,
                    left_table=left_table,
                    right_table=right_table,
                    left_column=left_column,
                    right_column=right_column,
                    attributes=[c.name for c in a_table.columns if c.name not in skip_columns]
                ))

        if target_table:
            linked_tables = [lt for lt in linked_tables if target_table in lt.linked_tables]
        return linked_tables

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

    def _vocab_columns(self, table: Table) -> list[str]:
        """
        Return the list of columns in the table that are control vocabulary terms.

        Args:
        - table (ermrest_model.Table): The table.

        Returns:
        - List[str]: List of column names that are control vocabulary terms.
        """
        return [fk.columns[0].name for fk in table.foreign_keys
                if len(fk.columns) == 1 and self.is_vocabulary(fk.pk_table)]

    def add_workflow(self, workflow_name: str, url: str, workflow_type: str,
                     version: str = "",
                     description: str = "") -> RID:
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

        # Check to make sure that the workflow is not already in the table. If its not, add it.
        try:
            url_column = self.ml_schema_path.Workflow.URL
            workflow_record = list(self.ml_schema_path.Workflow.filter(url_column == url).entities())[0]
            workflow_rid = workflow_record['RID']
        except IndexError:
            # Record doesn't exist already
            workflow_record = {
                'URL': url,
                'Name': workflow_name,
                'Description': description,
                'Checksum': self._get_checksum(url),
                'Version': version,
                'Workflow_Type': self.lookup_term("Workflow_Type", workflow_type)}
            workflow_rid = self.ml_schema_path.Workflow.insert([workflow_record])[0]['RID']

        return workflow_rid

    def validate_rids(self, rid_list: list[RID]) -> tuple[list[ResolveRidResult], set[Table], list[AssociatedTable]]:
        """
        Given a table, and a list of RIDS, verify that each RID exists. For each RID, check to see if a binary
        association exists to another table.  Return the list of RID results, the set of tables that the RIDs come from
        and a list of the association tables that are connected to the RID tables.
        :param rid_list: A list of RIDs to check.
        :return: tuple of list of resolved RIDS, a list of tables associated with the rids and a list of
                all  the associated tables.
        """
        # Get the tables associated with every rid.
        rid_info = [self.resolve_rid(rid) for rid in rid_list]
        rid_tables = set(r.table for r in rid_info)
        assoc_tables = [a for t in rid_tables for a in self.find_association_tables(t)]

        return rid_info, rid_tables, assoc_tables

    # delete
    def add_attributes(self, object_rids: list[RID], attribute_rids: list[RID],
                       values: list[dict[str, Any]] = None,
                       validate: bool = True) -> int:
        """
        Add an attribute to the specified object.

        :param object_rids: A list of the rids to which the new attributes will be attached.  Every RID in the list
        must come from the same table.
        :param attribute_rids: A list of the rids of the attributes to be added.
        :param values: Additional attributes that are added to the linkage.
        :param validate: Flag indicating whether to validate the arguments
        :return: Number of attributed added
        """

        if len(object_rids) != len(attribute_rids):
            raise DerivaMLException(f"Must have the name number of objects and attributes")
        if values:
            if len(object_rids) != len(values):
                raise DerivaMLException(f"Must have the name number of values and attributes")
        else:
            values = [{}] * len(object_rids)

        # We assume that all the RIDs come from the same table.
        object_table = self.resolve_rid(object_rids[0]).table
        attribute_table = self.resolve_rid(attribute_rids[0]).table

        if validate:
            _, object_tables, _ = self.validate_rids(object_rids)
            if len(object_tables) != 1:
                raise DerivaMLException(f"object_rid list contains more than one table {object_tables}")
            _, attribute_tables, _ = self.validate_rids(attribute_rids)
            if len(attribute_tables) != 1:
                raise DerivaMLException(f"object_rid list contains more than one table: {attribute_tables}")

        if len(association_tables := self.find_association_tables(object_table, attribute_table)) != 1:
            print(association_tables)
            raise DerivaMLException(f"Ambiguous association table from {object_table.name} to {attribute_table.name}.")
        elif not association_tables:
            raise DerivaMLException(f"No association between {object_table.name} and {attribute_table.name}")
        association_table = association_tables[0]

        entries = []
        for object_rid, attribute_rid, value in zip(object_rids, attribute_rids, values):
            if set(value.keys()) != set(association_table.attributes):
                raise DerivaMLException(f"Missing attribute values: {set(association_table.attributes)}")
            entries.append(
                {association_table.left_column.name: object_rid,
                 association_table.right_column.name: attribute_rid} | value
            )

        at = association_table.association_table
        self._batch_insert(at.name, entries, schema_name=at.schema.schema_name)
        return len(entries)

    def define_feature(self, feature_name: str, table: Table, target: Table, comment: str = "") -> None:
        execution_instance = self.model.schemas[self.ml_schema].tables["Execution"]
        table.define_association(self.model.schemas[self.domain_schema],
                                 [execution_instance],
                                 comment=comment
                                 )

    def add_features(self, feature_name: str,
                     object_rids: Iterable[RID],
                     execution_rids: Iterable[RID], values: Iterable[RID],
                       metadata: list[dict[str, Any]]= None,
                       validate: bool = True) -> int:
        """
        Add an attribute to the specified object.

        :param object_rids: A list of the rids to which the new attributes will be attached.  Every RID in the list
        must come from the same table.
        :param attribute_rids: A list of the rids of the attributes to be added.
        :param execution_rids: A list of the executables to be associated with the rids.
        :param values: Additional attributes that are added to the linkage.
        :param validate: Flag indicating whether to validate the arguments
        :return: Number of attributed added
        """

        self.find_association_tables()
        try:
            entities = { {object_name: obj, "Execution": exe} for obj, exe in zip(object_rids, execution_rids, strict=True)}
        except ValueError:
            raise DerivaMLException(f"Must have the name number of objects and execution RIDS")

        if not values:
            values = [{}] * len(entities)
        if len(object_rids) != len(attribute_rids):
            raise DerivaMLException(f"Must have the name number of objects and attributes")


        # We assume that all the RIDs come from the same table.
        object_table = self.resolve_rid(object_rids[0]).table
        attribute_table = self.resolve_rid(attribute_rids[0]).table

        if validate:
            _, object_tables, _ = self.validate_rids(object_rids)
            if len(object_tables) != 1:
                raise DerivaMLException(f"object_rid list contains more than one table {object_tables}")
            _, attribute_tables, _ = self.validate_rids(attribute_rids)
            if len(attribute_tables) != 1:
                raise DerivaMLException(f"object_rid list contains more than one table: {attribute_tables}")

        if len(association_tables := self.find_association_tables(object_table, attribute_table)) != 1:
            print(association_tables)
            raise DerivaMLException(f"Ambiguous association table from {object_table.name} to {attribute_table.name}.")
        elif not association_tables:
            raise DerivaMLException(f"No association between {object_table.name} and {attribute_table.name}")
        association_table = association_tables[0]

        entries = []
        for object_rid, attribute_rid, value in zip(object_rids, attribute_rids, values):
            if set(value.keys()) != set(association_table.attributes):
                raise DerivaMLException(f"Missing attribute values: {set(association_table.attributes)}")
            entries.append(
                {association_table.left_column.name: object_rid,
                 association_table.right_column.name: attribute_rid} | value
            )

        self._batch_insert(self.schema.Diagnosis,
                           [{'Execution': execution_rid, 'Diagnosis_Tag': diagtag_rid, **e} for e in entities])
        at = association_table.association_table
        self._batch_insert(at.name, entries, schema_name=at.schema.schema_name)
        return len(entries)

    def create_dataset(self, description: str, members: list[RID]) -> RID:
        """
        Create a new dataset from the specified list of RIDs.
        :param description:  Description of the dataset.
        :param members:  List of RIDs to include in the dataset.
        :return: New dataset RID.
        """

        return self.extend_dataset(None, members, description=description, validate=False)

    def delete_dataset(self, dataset_rid: RID) -> None:
        """
        Delete a dataset from the catalog.
        :param dataset_rid:  RID of the dataset to delete.
        :return:
        """
        # Get association table entries for this dataset
        # Delete association table entries

        for assoc_table in self.find_association_tables(self.dataset_table):
            schema_path = self.pb.schemas[assoc_table.association_table.schema.name]
            table_path = schema_path.tables[assoc_table.association_table.name]
            if assoc_table.left_table == self.dataset_table:
                dataset_column, element_column = assoc_table.left_column, assoc_table.right_column
            else:
                dataset_column, element_column = assoc_table.right_column, assoc_table.left_column
            dataset_column_path = table_path.columns[dataset_column.name]
            dataset_entries = table_path.filter(dataset_column_path == dataset_rid)
            try:
                dataset_entries.delete()
            except DataPathException:
                pass

        # Delete dataset.
        dataset_path = self.pb.schemas[self.dataset_table.schema.name].tables[self.dataset_table.name]
        dataset_path.filter(dataset_path.columns['RID'] == dataset_rid).delete()

    def add_element_type(self, element: str) -> Table:
        """
        Add a new element type to a dataset.
        :param element:
        :return:
        """
        # Add table to map
        element_table = self.model.schemas[self.domain_schema].tables[element]
        assoc_table = self.model.schemas[self.domain_schema].create_association(self.dataset_table, element_table)
        return assoc_table

    def dataset_members(self, dataset_rid: RID) -> dict[Table, RID]:
        """
        Return a list of RIDs associated with a specific dataset.
        :param dataset_rid:
        :return:
        """

        dataset_path = self.pb.schemas[self.dataset_table.schema.name].tables[self.dataset_table.name]
        dataset_exists = list(dataset_path.filter(dataset_path.columns['RID'] == dataset_rid).entities().fetch())

        if len(dataset_exists) != 1:
            raise DerivaMLException(f'Invalid RID: {dataset_rid}')

        # Look at each of the element types that might be in the dataset and get the list of rid for them from
        # the appropriate association table.
        rid_list = {}
        for assoc_table in self.find_association_tables(self.dataset_table):
            schema_path = self.pb.schemas[assoc_table.association_table.schema.name]
            table_path = schema_path.tables[assoc_table.association_table.name]
            if assoc_table.left_table == self.dataset_table:
                dataset_column, element_column = assoc_table.left_column, assoc_table.right_column
                element_table = assoc_table.right_table
            else:
                dataset_column, element_column = assoc_table.right_column, assoc_table.left_column
                element_table = assoc_table.left_table
            dataset_path = table_path.columns[dataset_column.name]
            element_path = table_path.columns[element_column.name]
            assoc_rids = table_path.filter(dataset_path == dataset_rid).attributes(element_path).fetch()
            rid_list.setdefault(element_table.name, []).extend([e[element_column.name] for e in assoc_rids])
        return rid_list

    def extend_dataset(self, dataset_rid: Optional[RID], members: list[RID],
                       description="",
                       validate: bool = True) -> RID:
        """
        Add additional elements to an existing dataset.
        :param dataset_rid: RID of dataset to extend or None if new dataset is to be created.
        :param members: List of RIDs of members to add to the  dataset.
        :param description: Description of the dataset if new entry is created.
        :param validate: Check rid_list to make sure elements are not already in the dataset.
        :return:
        """

        rid_info, _tables, assoc_tables = self.validate_rids(members)
        if self.dataset_table not in [a for a in assoc_tables]:
            pass
        pb = self.pb
        if dataset_rid:
            if self.resolve_rid(dataset_rid).table != self.dataset_table:
                raise DerivaMLException(f"RID: {dataset_rid} is not a dataset.")
        else:
            # Create the entry for the new dataset and get its RID.
            dataset_table_path = pb.schemas[self.dataset_table.schema.name].tables[self.dataset_table.name]
            dataset_rid = dataset_table_path.insert([{'Description': description}])[0]['RID']

        if validate:
            existing_rids = set(
                m
                for ms in self.dataset_members(dataset_rid).values()
                for m in ms
            )
            if overlap := set(existing_rids).intersection(members):
                raise DerivaMLException(f"Attempting to add existing member to dataset {dataset_rid}: {overlap}")

        # Now go through every rid to be added to the data set and sort them based on what association table entries
        # need to be made.
        dataset_elements = {}
        for r in rid_info:
            dataset_elements.setdefault(r.table, []).append(r.rid)
        # Now make the entries into the association tables.
        for elements in dataset_elements.values():
            if len(elements):
                self.add_attributes([dataset_rid] * len(elements), elements)
        return dataset_rid

    def add_execution(self, workflow_rid: str = "", datasets: List[str] = None,
                      description: str = "") -> RID:
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
            execution_rid = self.ml_schema_path.Execution.insert([{'Description': description,
                                                                   'Workflow': workflow_rid}])[0]['RID']
        else:
            execution_rid = self.ml_schema_path.Execution.insert([{'Description': description}])[0]['RID']
        if datasets:
            self._batch_insert("Dataset_Execution",
                               [{"Dataset": d, "Execution": execution_rid} for d in datasets])
        return execution_rid

    def update_execution(self, execution_rid: RID, workflow_rid: RID = "", datasets: List[str] = None,
                         description: str = "") -> RID:
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
        self._batch_update("Execution",
                           [{"RID": execution_rid, "Workflow": workflow_rid, "Description": description}])
        if datasets:
            self._batch_insert("Dataset_Execution",
                               [{"Dataset": d, "Execution": execution_rid} for d in datasets])
        return execution_rid

    def add_term(self,
                 table_name: str,
                 name: str,
                 description: str,
                 synonyms: Optional[List[str]] = None,
                 exist_ok: bool = False) -> RID:
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

    def list_vocabularies(self) -> list:
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

    def resolve_rid(self, rid: RID) -> ResolveRidResult:
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

    def _batch_insert(self, table: str, entities: Sequence[dict], schema_name: str = "") -> None:
        """
        Batch insert entities into a table.

        Args:
        - table (datapath._TableWrapper): Table wrapper object.
        - entities (Sequence[dict]): Sequence of entity dictionaries to insert.

        """
        schema_path = self.pb.schemas[schema_name] if schema_name else self.ml_schema_path
        table_path = schema_path.tables[table]

        it = iter(entities)
        while chunk := list(islice(it, 2000)):
            table_path.insert(chunk)

    def _batch_update(self, table: str, entities: Sequence[dict], schema_name: str = ""):
        """
        Batch update entities in a table.

        Args:
        - table (datapath._TableWrapper): Table wrapper object.
        - entities (Sequence[dict]): Sequence of entity dictionaries to update, must include RID.
        - update_cols (List[datapath._ColumnWrapper]): List of columns to update.

        """
        schema_path = self.pb.schemas[schema_name] if schema_name else self.ml_schema_path
        table_path = schema_path.tables[table]

        it = iter(entities)
        while chunk := list(islice(it, 2000)):
            columns = [table_path.columns[c] for e in chunk for c in e.keys() if c != "RID"]
            table_path.update(chunk, [table_path.RID], columns)

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

    def materialize_bdbag(self, minid: str, execution_rid: Optional[RID] = None) -> tuple[Path, RID]:
        """
        Materialize a BDBag into the cache directory. Validate its contents and return the path to the bag, and its RID.

        Args:
        - minid (str): Minimum viable identifier (minid) of the bag.
        - execution_rid (str): Resource Identifier (RID) of the execution to report status to.  If None, status is
                                not updated.

        Returns:
        - tuple: Tuple containing the path to the bag and the RID of the associated dataset.

        Raises:
        - DerivaMLException: If there is an issue materializing the bag.

        """

        def fetch_progress_callback(current, total):
            msg = f"Materializing bag: {current} of {total} file(s) downloaded."
            if execution_rid:
                self.update_status(Status.running, msg, execution_rid)
            logging.info(msg)
            return True

        def validation_progress_callback(current, total):
            msg = f"Validating bag: {current} of {total} file(s) validated."
            if execution_rid:
                self.update_status(Status.running, msg, execution_rid)
            logging.info(msg)
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

    def download_asset(self, asset_url: str, dest_filename: str) -> Path:
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
        return Path(dest_filename)

    def upload_assets(self, assets_dir: str) -> dict[str, FileUploadState]:
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
        results = {
            path: FileUploadState(state=result["State"], status=result["Status"], result=result["Result"])
            for path, result in uploader.uploadFiles().items()
        }
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
        self._batch_update("Execution",
                           [{"RID": execution_rid, "Status": self.status, "Status_Detail": status_detail}])

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
        table = self.ml_schema_path.tables[table_name]
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
            ass_table = table_name + '_Execution'
            ass_table_path = self.ml_schema_path.tables[ass_table]
            exec_file_exec_entities = ass_table_path.filter(ass_table_path.columns[table_name] == file_rid).entities()
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
            execution_metadata_type_rid = self.lookup_term(self.ml_schema_path.Execution_Metadata_Type,
                                                           "Execution Config")
            self._batch_insert("Execution_Metadata",
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

    def upload_execution_metadata(self, execution_rid: RID) -> dict[str, FileUploadState]:
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
            meta_exec_entities = self.ml_schema_path.Execution_Metadata_Execution.filter(
                self.ml_schema_path.Execution_Metadata_Execution.Execution == execution_rid).entities()
            meta_list = [e['Execution_Metadata'] for e in meta_exec_entities]
            entities = []
            for metadata in results.values():
                if metadata["State"] == 0 and metadata["Result"] is not None:
                    rid = metadata["Result"].get("RID")
                    if (rid is not None) and (rid not in meta_list):
                        entities.append({"Execution_Metadata": rid, "Execution": execution_rid})
        self._batch_insert("Execution_Metadata_Execution", entities)

        return results

    def upload_execution_assets(self, execution_rid: RID) -> dict[str, dict[str, FileUploadState]]:
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
                    asset_exec_entities = self.ml_schema_path.Execution_Assets_Execution.filter(
                        self.ml_schema_path.Execution_Assets_Execution.Execution == execution_rid).entities()
                    assets_list = [e['Execution_Assets'] for e in asset_exec_entities]
                    entities = []
                    for asset in result.values():
                        if asset["State"] == 0 and asset["Result"] is not None:
                            rid = asset["Result"].get("RID")
                            if (rid is not None) and (rid not in assets_list):
                                entities.append({"Execution_Assets": rid, "Execution": execution_rid})
                    self._batch_insert("Execution_Assets_Execution", entities)
                    results[str(folder_path)] = result
        return results

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

        self.update_status(Status.running, "Algorithm execution ended.", execution_rid)
        self._batch_update("Execution", [{"RID": execution_rid, "Duration": duration}])

    def execution_init(self, configuration_rid: RID) -> ConfigurationRecord:
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

    def execution(self, execution_rid: RID) -> DerivaMlExec:
        """
        Start the execution by initializing the context manager DerivaMlExec.

        Args:
        - execution_rid (str): Resource Identifier (RID) of the execution.

        Returns:
        - DerivaMlExec: Execution object.

        """
        return DerivaMlExec(self, execution_rid)

    def _clean_folder_contents(self, folder_path: Path, execution_rid: RID):
        try:
            shutil.rmtree(folder_path)
        except OSError as e:
            error = format_exception(e)
            self.update_status(Status.failed, error, execution_rid)

    def execution_upload(self, execution_rid: RID, clean_folder: bool = True) -> dict[str, dict[str, FileUploadState]]:
        """
        Upload all the assets and metadata associated with the current execution.

        Args:
        - execution_rid (str): Resource Identifier (RID) of the execution.

        Returns:
        - dict: Uploaded assets with key as assets' suborder name,
        values as an ordered dictionary with RID and metadata in the Execution_Assets table.

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
