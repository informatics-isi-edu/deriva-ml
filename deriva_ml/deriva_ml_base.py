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
from itertools import repeat
from pathlib import Path
from requests import HTTPError
from tempfile import TemporaryDirectory
from typing import List, Optional, Any, NewType, Iterable

import pandas as pd
import pkg_resources
import requests
from bdbag import bdbag_api as bdb
from copy import deepcopy
from deriva.core import ErmrestCatalog, get_credential, format_exception, urlquote, DEFAULT_SESSION_CONFIG
from deriva.core.datapath import DataPathException, _ResultSet
from deriva.core.ermrest_catalog import ResolveRidResult
from deriva.core.ermrest_model import FindAssociationResult
#from deriva.chisel import Model, Table, Column, ForeignKey, Key, builtin_types
from deriva.core.ermrest_model import Model, Table, Column, ForeignKey, Key, builtin_types
from deriva.core.hatrac_store import HatracStore
from deriva.core.utils import hash_utils, mime_utils
from deriva.transfer.upload.deriva_upload import GenericUploader
from pydantic import BaseModel, ValidationError, model_serializer

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


class BuiltinTypes(Enum):
    text = builtin_types.text
    int2 = builtin_types.int2
    jsonb = builtin_types.json
    float8 = builtin_types.float8
    timestamp = builtin_types.timestamp
    int8 = builtin_types.int8
    boolean = builtin_types.boolean
    json = builtin_types.json
    float4 = builtin_types.float4
    int4 = builtin_types.int4
    timestamptz = builtin_types.timestamptz
    date = builtin_types.date
    ermrest_rid = builtin_types.ermrest_rid
    ermrest_rcb = builtin_types.ermrest_rcb
    ermrest_rmb = builtin_types.ermrest_rmb
    ermrest_rct = builtin_types.ermrest_rct
    ermrest_rmt = builtin_types.ermrest_rmt
    markdown = builtin_types.markdown
    longtext = builtin_types.longtext
    ermrest_curie = builtin_types.ermrest_curie
    ermrest_uri = builtin_types.ermrest_uri
    color_rgb_hex = builtin_types.color_rgb_hex
    serial2 = builtin_types.serial2
    serial4 = builtin_types.serial4
    serial8 = builtin_types.serial8


class ColumnDefinition(BaseModel):
    name: str
    type: BuiltinTypes
    nullok: bool = True
    default: Any = None
    comment: str = None
    acls: dict = {}
    acl_bindings: dict = {}
    annotations: dict = {}

    @model_serializer()
    def serialize_column_definition(self):
        return Column.define(
            cname=self.name,
            ctype=self.type.value,
            nullok=self.nullok,
            default=self.default,
            comment=self.comment,
            acls=self.acls,
            acl_bindings=self.acl_bindings,
            annotations=self.annotations)


class KeyDefinition(BaseModel):
    colnames: Iterable[str]
    constraint_names: Iterable[str]
    comment: str = None
    annotations: dict = {}

    @model_serializer()
    def serialize_key_definition(self):
        return Key.define(
            colnames=self.colnames,
            constraint_names=self.constraint_names,
            comment=self.comment,
            annotations=self.annotations
        )


class ForeignKeyDefinition(BaseModel):
    colnames: Iterable[str]
    pk_sname: str
    pk_tname: str
    pk_colnames: Iterable[str]
    constraint_names: Iterable[str] = []
    on_update: str = 'NO ACTION'
    on_delete: str = 'NO ACTION'
    comment: str = None
    acls: dict[str, Any] = {}
    acl_bindings: dict[str, Any] = {}
    annotations: dict[str, Any] = {}

    @model_serializer()
    def serialize_fk_definition(self):
        return ForeignKey.define(
            fk_colnames=self.colnames,
            pk_sname=self.pk_sname,
            pk_tname=self.pk_tname,
            pk_colnames=self.pk_colnames,
            on_update=self.on_update,
            on_delete=self.on_delete,
            comment=self.comment,
            acls=self.acls,
            acl_bindings=self.acl_bindings,
            annotations=self.annotations
        )


class TableDefinition(BaseModel):
    name: str
    column_defs: Iterable[ColumnDefinition]
    key_defs: Iterable[KeyDefinition] = []
    fkey_defs: Iterable[ForeignKeyDefinition] = []
    comment: str = None
    acls: dict = {}
    acl_bindings: dict = {}
    annotations: dict = {}

    @model_serializer()
    def serialize_table_definition(self):
        return Table.define(
            tname=self.name,
            column_defs=self.column_defs,
            key_defs=self.key_defs,
            fkey_defs=self.fkey_defs,
            comment=self.comment,
            acls=self.acls,
            acl_bindings=self.acl_bindings,
            annotations=self.annotations)


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


class FindFeatureResult(FindAssociationResult):
    """Wrapper for results of Table.find_associations()"""

    def __init__(self, feature_name, table, self_fkey, other_fkeys):
        self.feature_name = feature_name
        super().__init__(table, self_fkey, other_fkeys)


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

    def __init__(self,
                 hostname: str,
                 catalog_id: str,
                 domain_schema: str,
                 cache_dir: str,
                 working_dir: str,
                 model_version: str,
                 ml_schema='deriva-ml'):
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
        self.ml_schema = ml_schema
        self.version = model_version

        self.credential = get_credential(hostname)
        self.catalog = ErmrestCatalog('https', hostname, catalog_id,
                                      self.credential,
                                      session_config=self._get_session_config())
        self.model = self.catalog.getCatalogModel()
        self.dataset_table = self.model.schemas[self.ml_schema].tables['Dataset']
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

    def _get_table(self, table: str | Table) -> Table:
        """
        Return the table object corresponding to the given table name. If the table name appears in more
        than one schema, return the first one you find.
        :param table:
        :return: Table object.
        """
        if isinstance(table, Table):
            return table
        for s in self.model.schemas.values():
            if table in s.tables.keys():
                return s.tables[table]
        raise DerivaMLException(f"The table {table} doesn't exist.")

    def create_vocabulary(self, vocab_name: str, comment="", schema=None) -> Table:
        """
        Create a controlled vocabulary table with the given vocab name.
        :param vocab_name: Name of the controlled vocabulary table.
        :param comment:
        :param schema: Schema in which to create the controlled vocabulary table.  Defaults to domain_schema.
        :return:
        """
        schema = schema or self.domain_schema
        return self.model.schemas[self.domain_schema].create_table(
            Table.define_vocabulary(vocab_name, f'{schema}:{{RID}}',
                                    comment=comment,
                                    key_defs=[Key.define(["Name"])])
        )

    def is_vocabulary(self, table_name: str) -> Table:
        """
        Check if a given table is a controlled vocabulary table.

        Args:
        - table_name (str): The name of the table.

        Returns:
        - Table: Table object if the table is a controlled vocabulary, False otherwise.

        """
        vocab_columns = {'NAME', 'URI', 'SYNONYMS', 'DESCRIPTION', 'ID'}
        table = self._get_table(table_name)
        return vocab_columns.issubset({c.name.upper() for c in table.columns}) and table

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
        ml_schema_path = self.catalog.getPathBuilder().schemas[self.ml_schema]
        try:
            url_column = ml_schema_path.Workflow.URL
            workflow_record = list(ml_schema_path.Workflow.filter(url_column == url).entities())[0]
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
            workflow_rid = ml_schema_path.Workflow.insert([workflow_record])[0]['RID']

        return workflow_rid

    def create_feature(self,
                       feature_name: str,
                       table: Table | str,
                       metadata: Iterable[ColumnDefinition | Table | Key | str] = None,
                       comment: str = "") -> None:

        def normalize_metadata(m: Key | Table | ColumnDefinition | str):
            if isinstance(m, str):
                return self._get_table(m)
            elif isinstance(m, ColumnDefinition):
                return m.model_dump()
            else:
                return m

        execution = self.model.schemas[self.ml_schema].tables["Execution"]
        feature = self.model.schemas[self.ml_schema].tables["Feature_Name"]
        self.lookup_term("Feature_Name", feature_name)
        table = self._get_table(table)
        self.model.schemas[self.domain_schema].create_table(
            table.define_association(
                associates=[execution, table, feature],
                table_name=f"{table.name}_Execution_Feature_Name_{feature_name}",
                metadata=[normalize_metadata(m) for m in metadata] if metadata else [],
                comment=comment
            )
        )

    def drop_feature(self, feature_name: str, table: Table | str) -> bool:
        table = self._get_table(table)
        try:
            feature = next(f for f in self.find_features(table) if f.feature_name == feature_name)
            feature.table.drop()
            return True
        except StopIteration:
            return False

    def find_features(self, table: Table | str) -> Iterable[FindFeatureResult]:
        """
        List the names of the features in the specified table.
        """
        table = self._get_table(table)

        def is_feature(a: FindAssociationResult) -> bool:
            try:
                return a.table.columns['Feature_Name']
            except KeyError:
                return False
        return [
            FindFeatureResult(
                feature_name=a.name.replace(f"{table.name}_Execution_Feature_Name_", ""),
                table=a.table,
                self_fkey=a.self_fkey, other_fkeys=a.other_fkeys
            ) for a in table.find_associations(min_arity=3, max_arity=3, pure=False) if is_feature(a)
        ]

    def add_features(self,
                     table: str,
                     feature_name: str,
                     object_rids: Iterable[RID],
                     execution_rids: Iterable[RID],
                     metadata: list[dict[str, Any]] = None) -> int:
        """
        Add an attribute to the specified object.
        :param table: The table to which the attribute is added.
        :param feature_name: The name of the feature.
        :param object_rids: A list of the rids to which the new attributes will be attached.  Every RID in the list
        must come from the same table.
        :param execution_rids: A list of the executables to be associated with the rids.
        :param metadata: Additional attributes that are added to the linkage.
        :return: Number of attributed added
        """

        table = self._get_table(table)
        feature_name_rid = self.lookup_term("Feature_Name", feature_name)
        feature = next(f for f in self.find_features(table) if f.feature_name == feature_name)
        object_table = feature.self_fkey.pk_table.name
        skip_columns = {"RID", "RCB", "RMB", "RCT", "RMT", "Execution", "Feature_Name", table.name}
        metadata_columns = {c.name for c in feature.table.columns if c.name not in skip_columns}
        required_metadata = {c.name for c in feature.table.columns if c.name not in skip_columns and c.nullok is False}

        def feature_entity(object_rid: RID, exe_rid: RID, meta: dict[str, Any]) -> dict[str, Any]:
            if self.resolve_rid(object_rid).table.name != object_table:
                raise DerivaMLException(f"object_rid {object_rid} is not in {object_table} table.")
            if self.resolve_rid(exe_rid).table.name != "Execution":
                raise DerivaMLException(f"execution_rid {exe_rid} is not in Execution table.")
            if not metadata_columns >= set(meta.keys()):
                raise DerivaMLException(f"Bad column values: {set(meta.keys())} not in {metadata_columns}")
            if not set(meta.keys()) >= required_metadata:
                raise DerivaMLException(f"Missing non-null column: {required_metadata}")

            return {object_table: object_rid, 'Execution': exe_rid, 'Feature_Name': feature_name_rid} | meta

        try:
            entries = list(object_rids)
            entries = [
                feature_entity(object_rid, execution_rid, md)
                for object_rid, execution_rid, md in
                zip(entries, execution_rids, metadata or repeat({}, times=len(entries)), strict=True)
            ]
        except ValueError:
            raise DerivaMLException(f"Length of object_rid, execution_rid, and metadata must be equal.")
        self.catalog.getPathBuilder().schemas[feature.table.schema.name].tables[feature.name].insert(entries)
        return len(entries)

    def list_feature(self, table: Table | str, feature_name: str) -> _ResultSet:
        """
        Return a dataframe containing all values of a feature associated with a table.
        :param table:
        :param feature_name:
        :return:
        """
        feature_name_rid = self.lookup_term("Feature_Name", feature_name)
        feature = next(f for f in self.find_features(table) if f.feature_name == feature_name)
        pb = self.catalog.getPathBuilder()
        return pb.schemas[feature.table.schema.name].tables[feature.name].entities().fetch()

    def create_dataset(self, description: str, **kwargs) -> RID:
        """
        Create a new dataset from the specified list of RIDs.
        :param description:  Description of the dataset.
        :return: New dataset RID.
        """
        # Create the entry for the new dataset and get its RID.
        dataset_table_path = (
            self.catalog.getPathBuilder().schemas[self.dataset_table.schema.name].tables)[self.dataset_table.name]
        return dataset_table_path.insert([{'Description': description}])[0]['RID']

    def find_datasets(self) -> Iterable[dict[str, Any]]:
        """
        Returns a list of currently available datasets.
        :return:
        """
        return list(
            self.catalog.getPathBuilder().schemas[self.ml_schema].tables[self.dataset_table.name].entities().fetch())

    def delete_dataset(self, dataset_rid: RID) -> None:
        """
        Delete a dataset from the catalog.
        :param dataset_rid:  RID of the dataset to delete.
        :return:
        """
        # Get association table entries for this dataset
        # Delete association table entries
        pb = self.catalog.getPathBuilder()
        for assoc_table in self.dataset_table.find_associations(self.dataset_table):
            schema_path = pb.schemas[assoc_table.table.schema.name]
            table_path = schema_path.tables[assoc_table.name]
            dataset_column_path = table_path.columns[assoc_table.self_fkey.columns[0].name]
            dataset_entries = table_path.filter(dataset_column_path == dataset_rid)
            try:
                dataset_entries.delete()
            except DataPathException:
                pass

        # Delete dataset.
        dataset_path = pb.schemas[self.dataset_table.schema.name].tables[self.dataset_table.name]
        dataset_path.filter(dataset_path.columns['RID'] == dataset_rid).delete()

    def add_element_type(self, element: str | Table) -> Table:
        """
        Add a new element type to a dataset.
        :param element:
        :return:
        """
        # Add table to map
        element_table = self._get_table(element)
        assoc_table = self.model.schemas[self.domain_schema].create_association(self.dataset_table, element_table)
        return assoc_table

    def list_dataset_members(self, dataset_rid: RID) -> dict[Table, RID]:
        """
        Return a list of RIDs associated with a specific dataset.
        :param dataset_rid:
        :return:
        """
        pb = self.catalog.getPathBuilder()
        dataset_path = pb.schemas[self.dataset_table.schema.name].tables[self.dataset_table.name]
        dataset_exists = list(dataset_path.filter(dataset_path.columns['RID'] == dataset_rid).entities().fetch())

        if len(dataset_exists) != 1:
            raise DerivaMLException(f'Invalid RID: {dataset_rid}')

        # Look at each of the element types that might be in the dataset and get the list of rid for them from
        # the appropriate association table.
        rid_list = {}
        for assoc_table in self.dataset_table.find_associations():
            schema_path = pb.schemas[assoc_table.table.schema.name]
            table_path = schema_path.tables[assoc_table.name]
            dataset_column, element_column = assoc_table.self_fkey.columns[0], assoc_table.other_fkeys.pop().columns[0]
            element_table = element_column.table
            dataset_path = table_path.columns[dataset_column.name]
            element_path = table_path.columns[element_column.name]
            assoc_rids = table_path.filter(dataset_path == dataset_rid).attributes(element_path).fetch()
            rid_list.setdefault(element_table.name, []).extend([e[element_column.name] for e in assoc_rids])
        return rid_list

    def insert_dataset(self, dataset_rid: Optional[RID], members: list[RID],
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

        # Get a list of all of the types of objects that can be linked to a dataset.
        dataset_objects = [a.other_fkeys.pop().pk_table.name for a in self.dataset_table.find_associations(pure=False)]
        pb = self.catalog.getPathBuilder()
        dataset_table_path = pb.schemas[self.dataset_table.schema.name].tables[self.dataset_table.name]

        if validate:
            existing_rids = set(
                m
                for ms in self.list_dataset_members(dataset_rid).values()
                for m in ms
            )
            if overlap := set(existing_rids).intersection(members):
                raise DerivaMLException(f"Attempting to add existing member to dataset {dataset_rid}: {overlap}")

        # Now go through every rid to be added to the data set and sort them based on what association table entries
        # need to be made.
        dataset_elements = self.list_dataset_members(dataset_rid)
        dataset_elements = {}
        for m in members:
            rid_info = self.resolve_rid(m)
            dataset_elements.setdefault(rid_info.table, []).append(rid_info.rid)
        # Now make the entries into the association tables.
        for elements in dataset_elements.values():
            if len(elements):
                #       [ {'Dataset': dataset_rid, element_name: e} for e in elements]
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
        ml_schema_path = self.catalog.getPathBuilder().schemas[self.ml_schema]
        if workflow_rid:
            execution_rid = (
                ml_schema_path.Execution.insert([{'Description': description, 'Workflow': workflow_rid}]))[0]['RID']
        else:
            execution_rid = ml_schema_path.Execution.insert([{'Description': description}])[0]['RID']
        if datasets:
            ml_schema_path.Dataset_Execution.insert([{"Dataset": d, "Execution": execution_rid} for d in datasets])
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
        schema_path = self.catalog.getPathBuilder().schemas[self.ml_schema]
        schema_path.Execution.update([{"RID": execution_rid, "Workflow": workflow_rid, "Description": description}])
        if datasets:
            schema_path.Dataset_Execution.insert([{"Dataset": d, "Execution": execution_rid} for d in datasets])
        return execution_rid

    def add_term(self,
                 table: str | Table,
                 term_name: str,
                 description: str,
                 synonyms: Optional[List[str]] = None,
                 exists_ok: bool = True) -> RID:
        """
        Creates a new control vocabulary term in the control vocabulary table.

        Args:
        - table_name (str): The name of the control vocabulary table.
        - term_name (str): The name of the new control vocabulary.
        - description (str): The description of the new control vocabulary.
        - synonyms (List[str]): Optional list of synonyms for the new control vocabulary. Defaults to an empty list.
        - exist_ok (bool): Optional flag indicating whether to allow creation if the control vocabulary name
          already exists. Defaults to True.

        Returns:
        - str: The RID of the newly created control vocabulary.

        Raises:
        - EyeAIException: If the control vocabulary name already exists and exist_ok is False.
        """
        synonyms = synonyms or []
        pb = self.catalog.getPathBuilder()
        if not (table := self.is_vocabulary(table)):
            raise DerivaMLException(f"The table {table} is not a controlled vocabulary")

        schema_name = table.schema.name
        table_name = table.name

        try:
            col_map = {col.upper(): col for col in pb.schemas[schema_name].tables[table_name].columns.keys()}
            term_rid = pb.schemas[schema_name].tables[table_name].insert(
                [{col_map['NAME']: term_name, col_map['DESCRIPTION']: description, col_map['SYNONYMS']: synonyms}],
                defaults={col_map['ID'], col_map['URI']})[0]['RID']
        except DataPathException as e:
            if "already exists" in str(e):
                term_rid = self.lookup_term(table, term_name)
                if not exists_ok:
                    raise DerivaMLException(f"{term_name} existed with RID {term_rid}")
            else:
                raise e
            # Check vocabulary
        return term_rid

    def lookup_term(self, table: str | Table, term_name: str) -> str:
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
        vocab_table = self.is_vocabulary(table)
        if not vocab_table:
            raise DerivaMLException(f"The table {table} is not a controlled vocabulary")
        schema_name, table_name = vocab_table.schema.name, vocab_table.name
        schema_path = self.catalog.getPathBuilder().schemas[schema_name]
        for term in schema_path.tables[table_name].entities():
            term_upper = {key.upper(): value for key, value in term.items()}
            if term_name == term_upper['NAME'] or (term_upper['SYNONYMS'] and term_name in term_upper['SYNONYMS']):
                return term['RID']

        raise DerivaMLException(f"Term {term_name} is not in vocabulary {table_name}")

    def find_vocabularies(self) -> Iterable[Table]:
        """
        Return a list of all the controlled vocabulary tables in the domain schema.

        Returns:
         - List[str]: A list of table names representing controlled vocabulary tables in the schema.

        """
        return [t for s in self.model.schemas.values() for t in s.tables.values() if self.is_vocabulary(t)]

    def list_vocabulary_terms(self, table_name: str) -> Iterable[dict[str, Any]]:
        """
        Return the dataframe of terms that are in a vocabulary table.

        Args:
        - table_name (str): The name of the controlled vocabulary table.

        Returns:
        - Iterable: A iterable containing the terms in the specified controlled vocabulary table.

        Raises:
        - EyeAIException: If the schema or vocabulary table doesn't exist, or if the table is not
          a controlled vocabulary.
        """
        pb = self.catalog.getPathBuilder()
        if not (table := self.is_vocabulary(table_name)):
            raise DerivaMLException(f"The table {table_name} is not a controlled vocabulary")

        return list(pb.schemas[table.schema.name].tables[table.name].entities().fetch())

    def resolve_rid(self, rid: RID) -> ResolveRidResult:
        """
        Return a named tuple with information about the specified RID.
        :param rid:
        :return:
        """
        try:
            return self.catalog.resolve_rid(rid, self.model)
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
        user_path = self.catalog.getPathBuilder().schemas['public'].users.ERMrest_Client.path
        return pd.DataFrame(user_path.entities().fetch())[['ID', 'Full_Name']]

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
        results = deepcopy(uploader.uploadFiles())
        # results = {
        #     path: FileUploadState(state=result["State"], status=result["Status"], result=result["Result"])
        #     for path, result in uploader.uploadFiles().items()
        # }
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
        self.catalog.getPathBuilder().schemas[self.ml_schema].Execution.update(
            [{"RID": execution_rid, "Status": self.status, "Status_Detail": status_detail}]
        )

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
        ml_schema_path = self.catalog.getPathBuilder().schemas[self.ml_schema]
        table = ml_schema_path.tables[table_name]
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
            ass_table_path = ml_schema_path.tables[ass_table]
            exec_file_exec_entities = ass_table_path.filter(ass_table_path.columns[table_name] == file_rid).entities()
            exec_list = [e['Execution'] for e in exec_file_exec_entities]
            if execution_rid not in exec_list:
                table_path = self.catalog.getPathBuilder().schemas[self.ml_schema].tables[ass_table]
                table_path.insert([{table_name: file_rid, "Execution": execution_rid}])
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
            ml_schema_path = self.catalog.getPathBuilder().schemas[self.ml_schema]
            execution_metadata_type_rid = self.lookup_term(ml_schema_path.Execution_Metadata_Type,
                                                           "Execution Config")
            ml_schema_path.tables["Execution_Metadata"].insert(
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
        ml_schema_path = self.catalog.getPathBuilder().schemas[self.ml_schema]
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
            meta_exec_entities = ml_schema_path.Execution_Metadata_Execution.filter(
                ml_schema_path.Execution_Metadata_Execution.Execution == execution_rid).entities()
            meta_list = [e['Execution_Metadata'] for e in meta_exec_entities]
            entities = []
            for metadata in results.values():
                if metadata["State"] == 0 and metadata["Result"] is not None:
                    rid = metadata["Result"].get("RID")
                    if (rid is not None) and (rid not in meta_list):
                        entities.append({"Execution_Metadata": rid, "Execution": execution_rid})
        self.catalog.getPathBuilder().schemas[self.ml_schema].Execution_Metadata_Execution.insert(entities)
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
        ml_schema_path = self.catalog.getPathBuilder().schemas[self.ml_schema]
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
                    asset_exec_entities = ml_schema_path.Execution_Assets_Execution.filter(
                        ml_schema_path.Execution_Assets_Execution.Execution == execution_rid).entities()
                    assets_list = [e['Execution_Assets'] for e in asset_exec_entities]
                    entities = []
                    for asset in result.values():
                        if asset["State"] == 0 and asset["Result"] is not None:
                            rid = asset["Result"].get("RID")
                            if (rid is not None) and (rid not in assets_list):
                                entities.append({"Execution_Assets": rid, "Execution": execution_rid})
                    ml_schema_path.Execution_Assets_Execution.insert(entities)
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
        self.catalog.getPathBuilder().schemas[self.ml_schema].Execution.update(
            [{"RID": execution_rid, "Duration": duration}])

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
