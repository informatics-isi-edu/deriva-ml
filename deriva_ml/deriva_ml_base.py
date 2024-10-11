
from bdbag import bdbag_api as bdb
from copy import deepcopy
from datetime import datetime
from deriva.core import ErmrestCatalog, get_credential, format_exception, urlquote, DEFAULT_SESSION_CONFIG
from deriva.core.datapath import DataPathException, _ResultSet
from deriva.core.ermrest_catalog import ResolveRidResult
from deriva.core.datapath import _CatalogWrapper
from deriva.core.ermrest_model import FindAssociationResult
from deriva.core.ermrest_model import Table, Column, ForeignKey, Key, builtin_types
from deriva.core.hatrac_store import HatracStore
from deriva.core.utils import hash_utils, mime_utils
from deriva.core.utils.core_utils import tag as deriva_tags
from deriva_ml.execution_configuration import ExecutionConfiguration
from deriva_ml.schema_setup.dataset_annotations import generate_dataset_annotations
from deriva_ml.schema_setup.dataset_annotations import generate_dataset_download_spec
from deriva.transfer.upload.deriva_upload import GenericUploader
from deriva.transfer.download.deriva_download import GenericDownloader
from deriva.core.utils.hash_utils import compute_file_hashes
from enum import Enum
import hashlib
from itertools import chain
import json
import logging
import pkg_resources
from pydantic import BaseModel, ValidationError, model_serializer, Field, create_model, field_validator
import os
from pathlib import Path
import re
import requests
import shutil
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import List, Optional, Any, NewType, Iterable
import warnings


RID = NewType('RID', str)

# We are going to use schema as a field name and this collides with method in pydantic base class
warnings.filterwarnings('ignore',
                        message='Field name "schema"',
                        category=Warning,
                        module='pydantic')


# For some reason, deriva-py doesn't use the proper enum class!!
class UploadState(str, Enum):
    success = 'Success'
    failed = 'Failed'
    pending = 'Pending'
    running = 'Running'
    paused = 'Paused'
    aborted = 'Aborted'
    cancelled = 'Cancelled'
    timeout = 'Timeout'


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
            column_defs=[c.model_dump() for c in self.column_defs],
            key_defs=[k.model_dump() for k in self.key_defs],
            fkey_defs=[fk.model_dump() for fk in self.fkey_defs],
            comment=self.comment,
            acls=self.acls,
            acl_bindings=self.acl_bindings,
            annotations=self.annotations)


class VocabularyTerm(BaseModel):
    """
    An entry in a vocabulary table.
    """
    name: str = Field(alias='Name')
    synonyms: Optional[list[str]] = Field(alias='Synonyms')
    id: str = Field(alias='ID')
    uri: str = Field(alias='URI')
    description: str = Field(alias='Description')
    rid: str = Field(alias='RID')

    class Config:
        extra = 'ignore'


class Feature(BaseModel):
    pass


class FindFeatureResult(FindAssociationResult):
    """Wrapper for results of Table.find_associations()"""

    def __init__(self, feature_name, table, self_fkey, other_fkeys):
        self.feature_name = feature_name
        super().__init__(table, self_fkey, other_fkeys)

    def __repr__(self) -> str:
        return (f'FeatureResult({self.self_fkey.pk_table.name}, feature_name={self.feature_name}, '
                f'table={self.table.name})')


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

    def __init__(self, msg=''):
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
    running = 'Running'
    pending = 'Pending'
    completed = 'Completed'
    failed = 'Failed'


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
    vocabs: dict[str, list[VocabularyTerm]]
    execution_rid: RID
    workflow_rid: RID
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
                 domain_schema: str = None,
                 project_name: str = None,
                 cache_dir: Optional[str] = None,
                 working_dir: Optional[str] = None,
                 model_version: str = '1',
                 ml_schema='deriva-ml'):
        """

        :param hostname: Hostname of the Deriva server.
        :param catalog_id: Catalog ID.
        :param domain_schema: Schema name for domain specific tables and relationships.
        :param cache_dir: Directory path for caching data downloaded from the Deriva server as bdbag.
        :param working_dir: Directory path for storing data used by or generated by any computations.
        :param model_version:
        """
        self.host_name = hostname
        self.catalog_id = catalog_id
        self.ml_schema = ml_schema
        self.version = model_version

        self.credential = get_credential(hostname)
        self.catalog = ErmrestCatalog('https', hostname, catalog_id,
                                      self.credential,
                                      session_config=self._get_session_config())
        self.model = self.catalog.getCatalogModel()
        self.configuration = None

        builtin_schemas = ['public', self.ml_schema, 'www']
        self.domain_schema = domain_schema or [s for s in self.model.schemas.keys() if s not in builtin_schemas].pop()
        self.project_name = project_name or self.domain_schema

        self.start_time = datetime.now()
        self.status = Status.pending.value
        tdir = None
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            tdir = TemporaryDirectory(delete=False)
            self.cache_dir = Path(tdir.name)
        default_workdir = self.__class__.__name__ + '_working'
        if working_dir:
            self.working_dir = Path(working_dir)
        else:
            tdir = tdir or TemporaryDirectory(delete=False)
            self.working_dir = Path(tdir.name) / default_workdir
        self.execution_assets_path = self.working_dir / 'Execution_Assets/'
        self.execution_metadata_path = self.working_dir / 'Execution_Metadata/'
        self.execution_assets_path.mkdir(parents=True, exist_ok=True)
        self.execution_metadata_path.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        if 'dirty' in self.version:
            logging.info(f'Loading dirty model.  Consider commiting and tagging: {self.version}')

    @staticmethod
    def _get_session_config():
        session_config = DEFAULT_SESSION_CONFIG.copy()
        session_config.update({
            # our PUT/POST to ermrest is idempotent
            'allow_retry_on_all_methods': True,
            # do more retries before aborting
            'retry_read': 8,
            'retry_connect': 5,
            # increase delay factor * 2**(n-1) for Nth retry
            'retry_backoff_factor': 5,
        })
        return session_config

    @property
    def pathBuilder(self) -> _CatalogWrapper:
        return self.catalog.getPathBuilder()

    @property
    def dataset_table(self) -> Table:
        return self.model.schemas[self.ml_schema].tables['Dataset']

    @property
    def domain_path(self):
        return self.catalog.getPathBuilder().schemas[self.domain_schema]

    def _get_table(self, table: str | Table) -> Table:
        """
        Return the table object corresponding to the given table name. If the table name appears in more
        than one schema, return the first one you find.

        :param table: A ERMRest table object or a sting that is the name of the table.
        :return: Table object.
        """
        if isinstance(table, Table):
            return table
        for s in self.model.schemas.values():
            if table in s.tables.keys():
                return s.tables[table]
        raise DerivaMLException(f"The table {table} doesn't exist.")

    def chaise_url(self, table: str | Table) -> str:
        table = self._get_table(table)
        uri = self.catalog.get_server_uri().replace('ermrest/catalog/', 'chaise/recordset/#')
        return f'{uri}/{table.schema.name}:{table.name}'

    def create_vocabulary(self, vocab_name: str, comment='', schema=None) -> Table:
        """
        Create a controlled vocabulary table with the given vocab name.
        :param vocab_name: Name of the controlled vocabulary table.
        :param comment:
        :param schema: Schema in which to create the controlled vocabulary table.  Defaults to domain_schema.
        :return:
        """
        schema = schema or self.domain_schema
        return self.model.schemas[schema].create_table(
            Table.define_vocabulary(vocab_name, f'{self.project_name}:{{RID}}', comment=comment)
        )

    def is_vocabulary(self, table_name: str | Table) -> bool:
        """
        Check if a given table is a controlled vocabulary table.

        param: table_name: A ERMRest table object or the name of the table.

        returns: Table object if the table is a controlled vocabulary, False otherwise.

        """
        vocab_columns = {'NAME', 'URI', 'SYNONYMS', 'DESCRIPTION', 'ID'}
        table = self._get_table(table_name)
        return vocab_columns.issubset({c.name.upper() for c in table.columns})

    def create_asset(self, asset_name: str, comment='', schema: str = None, upload_spec: bool = True) -> Table:
        """
        Create an asset table with the given asset name.  Optionally modify the upload spec so upload_assets
        will work with this table.
        :param asset_name:
        :param comment:
        :param schema:
        :param upload_spec:
        :return:
        """
        schema = schema or self.domain_schema
        asset_table = self.model.schemas[schema].create_table(
            Table.define_asset(schema, asset_name, f'{schema}:{{RID}}', comment=comment))
        if upload_spec:
            self.model.annotations[deriva_tags.bulk_upload]['asset_mappings'].append(
                {
                    'column_map': {
                        'MD5': '{md5}',
                        'URL': '{URI}',
                        'Length': '{file_size}',
                        'Filename': '{file_name}'
                    },
                    'file_pattern': f'(?i)^.*/{asset_name}/(?P<filename>[A-Za-z0-9_]*)[.](?P<file_ext>[a-z0-9]*)$',
                    'target_table': [schema, asset_name],
                    'checksum_types': ['sha256', 'md5'],
                    'hatrac_options': {'versioned_urls': True},
                    'hatrac_templates': {
                        'hatrac_uri': f'/hatrac/{asset_name}/{{md5}}.{{file_name}}',
                        'content-disposition': "filename*=UTF-8''{file_name}"
                    },
                    'record_query_template': '/entity/{target_table}/MD5={md5}&Filename={file_name}',
                    'create_record_before_upload': False
                }
            )
            self.model.apply()
        return asset_table

    def is_association(self, table_name: str | Table, unqualified=True, pure=True) -> bool | set | int:
        table = self._get_table(table_name)
        return table.is_association(unqualified=unqualified, pure=pure)

    def is_asset(self, table_name: str | Table) -> bool:
        asset_columns = {'Filename', 'URL', 'Length', 'MD5', 'Description'}
        table = self._get_table(table_name)
        return asset_columns.issubset({c.name for c in table.columns})

    def find_assets(self) -> list[Table]:
        return [t for s in self.model.schemas.values() for t in s.tables.values() if self.is_asset(t)]

    def add_workflow(self, workflow_name: str, url: str, workflow_type: str,
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

        # Check to make sure that the workflow is not already in the table. If its not, add it.
        ml_schema_path = self.pathBuilder.schemas[self.ml_schema]
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
                'Workflow_Type': self.lookup_term('Workflow_Type', workflow_type).name}
            workflow_rid = ml_schema_path.Workflow.insert([workflow_record])[0]['RID']

        return workflow_rid

    def create_feature(self,
                       feature_name: str,
                       table: Table | str,
                       terms: list[Table | str] = None,
                       assets: list[Table | str] = None,
                       metadata: Iterable[ColumnDefinition | Table | Key | str] = None,
                       comment: str = '') -> type[Feature]:
        """
        Create a new feasture that can be associated with a table. The feature can assocate a controlloed
        vocabulary term, an asset, or any other values with a specific instance of a object and and execution.
        :param feature_name:
        :param table:
        :param terms:
        :param assets:
        :param metadata:
        :param comment:
        :return:
        """

        terms = terms or []
        assets = assets or []
        metadata = metadata or []

        def normalize_metadata(m: Key | Table | ColumnDefinition | str):
            if isinstance(m, str):
                return self._get_table(m)
            elif isinstance(m, ColumnDefinition):
                return m.model_dump()
            else:
                return m

        # Make sure that the provided assets or terms are actually assets or terms.
        if not all(map(self.is_asset, assets)):
            raise DerivaMLException(f'Invalid create_feature asset table.')
        if not all(map(self.is_vocabulary, terms)):
            raise DerivaMLException(f'Invalid create_feature asset table.')

        # Get references to the necessary tables and make sure that the
        # provided feature name exists.
        table = self._get_table(table)
        execution = self.model.schemas[self.ml_schema].tables['Execution']
        feature_name_table = self.model.schemas[self.ml_schema].tables['Feature_Name']
        feature_name_term = self.add_term('Feature_Name', feature_name, description=comment)
        atable_name = f'Execution_{table.name}_{feature_name_term.name}'
        # Now create the association table that implements the feature.
        atable = self.model.schemas[self.domain_schema].create_table(
            table.define_association(
                table_name=atable_name,
                associates=[execution, table, feature_name_table],
                metadata=[normalize_metadata(m) for m in chain(assets, terms, metadata)],
                comment=comment
            )
        )
        atable.columns['Feature_Name'].alter(default=feature_name_term.name)
        return self.feature_record_class(table, feature_name)

    def feature_record_class(self, table: str | Table, feature_name: str) -> type[Feature]:
        """"
        Create a pydantic model for entries into the specified feature table
        """

        def validate_rid(rid):
            try:
                self.resolve_rid(rid)
            except DerivaMLException as e:
                raise ValidationError(str(e))
            return rid

        def map_type(x: builtin_types) -> type:
            match x.typename:
                case 'text':
                    return str
                case 'int2' | 'int4' | 'int8':
                    return int
                case 'float4', 'float8':
                    return float
                case _:
                    return str

        table = self._get_table(table)
        # Get the association table that implements the feature.
        if len(assoc_table := [a.table for a in self.find_features(table) if a.feature_name == feature_name]):
            assoc_table = assoc_table.pop()
        else:
            raise DerivaMLException(f"Table {table.name} doesn't have feature named {feature_name}.")

        # Create feature class
        validators = {'execution_validator': field_validator('Execution', mode='after')(validate_rid),
                      'feature_name_validator': field_validator('Feature_Name', mode='after')(validate_rid)}
        system_columns = {'RID', 'RMB', 'RCB', 'RCT', 'RMT'}
        feature_columns = {
            c.name: (map_type(c.type), c.default or ...) for c in assoc_table.columns if c.name not in system_columns
        }
        featureclass_name = f'{table.name}Feature{feature_name}'
        return create_model(featureclass_name, __base__=Feature, __validators__=validators, **feature_columns)

    def _feature_table(self, f: Feature) -> Table:
        """
        Return the feature table associated with the specified feature value class instance.
        :param f: An instance of a feature class
        :return:
        """
        col_names = set(f.model_dump().keys())  # All of the column names in the feature record.
        feature_name = f.model_dump()['Feature_Name']

        for a in self.model.schemas[self.ml_schema].tables['Execution'].find_associations(
                min_arity=3, max_arity=3, pure=False):
            #  The column names in the keys of the association table are fully contained in the feature value
            if col_names > {k.pk_table.name for k in a.other_fkeys} and feature_name in a.name:
                break
        return a.table

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
                feature_name=a.name.replace(f'Execution_{table.name}_', ''),
                table=a.table,
                self_fkey=a.self_fkey, other_fkeys=a.other_fkeys
            ) for a in table.find_associations(min_arity=3, max_arity=3, pure=False) if is_feature(a)
        ]

    def add_features(self, features: Iterable[Feature]) -> int:
        """
        Add an attribute to the specified object.
        :return: Number of attributed added
        """
        features = list(features)
        feature_table = self._feature_table(features[0])
        feature_path = self.catalog.getPathBuilder().schemas[feature_table.schema.name].tables[feature_table.name]
        entries = feature_path.insert(f.model_dump() for f in features)
        return len(entries)

    def list_feature(self, table: Table | str, feature_name: str) -> _ResultSet:
        """
        Return a dataframe containing all values of a feature associated with a table.
        :param table:
        :param feature_name:
        :return:
        """
        table = self._get_table(table)
        feature = next(f for f in self.find_features(table) if
                       f.feature_name == feature_name)
        pb = self.catalog.getPathBuilder()
        return pb.schemas[feature.table.schema.name].tables[feature.name].entities().fetch()

    def create_dataset(self, ds_type: str | list[str], description: str) -> RID:
        """
        Create a new dataset from the specified list of RIDs.
        :param ds_type: One or more dataset types.  Must be a term from the Dataset_Type controlled vocabulary.
        :param description:  Description of the dataset.
        :return: New dataset RID.
        """
        # Create the entry for the new dataset and get its RID.
        ds_types = [ds_type] if isinstance(ds_type, str) else ds_type
        for ds_type in ds_types:
            if not self.lookup_term('Dataset_Type', ds_type):
                raise DerivaMLException(f'Dataset type must be a vocabulary term.')
        dataset_table_path = (
            self.pathBuilder.schemas[self.dataset_table.schema.name].tables)[self.dataset_table.name]
        dataset = dataset_table_path.insert([{'Description': description, 'Dataset_Type': ds_type}])[0]['RID']

        # Get the name of the association table between dataset and dataset_type.
        atable = next(self.model.schemas[self.ml_schema].tables['Dataset_Type'].find_associations()).name
        self.pathBuilder.schemas[self.ml_schema].tables[atable].insert(
            [{'Dataset_Type': ds_type, 'Dataset': dataset} for ds_type in ds_types])
        return dataset

    def find_datasets(self) -> Iterable[dict[str, Any]]:
        """
        Returns a list of currently available datasets.
        :return:
        """
        # Get datapath to all the tables we will need: Dataset, Dataset_Type and the association table.
        pb = self.pathBuilder
        dataset_path = pb.schemas[self.dataset_table.schema.name].tables[self.dataset_table.name]
        atable = next(self.model.schemas['deriva-ml'].tables['Dataset_Type'].find_associations()).name
        ml_path = pb.schemas[self.ml_schema]
        dataset_type_path = ml_path.Dataset_Type
        atable_path = ml_path.tables[atable]

        # Get a list of all the dataset_type values associated with this dataset.
        ds_type = [ds['Name'] for ds in dataset_path.link(atable_path).link(dataset_type_path).entities().fetch()]
        return [e | {'Dataset_Type': ds_type} for e in dataset_path.entities().fetch()]

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

    def list_dataset_element_types(self) -> Iterable[Table]:
        """
        Return the list of tables that can be included as members of a dataset.
        :return:
        """
        def domain_table(table: Table) -> bool:
            return table.schema.name == self.domain_schema or table.name == self.dataset_table.name

        return [t for a in self.dataset_table.find_associations() if domain_table(t := a.other_fkeys.pop().pk_table)]

    def add_dataset_element_type(self, element: str | Table) -> Table:
        """
        Add a new element type to a dataset.
        :param element:
        :return:
        """
        # Add table to map
        element_table = self._get_table(element)
        table = self.model.schemas[self.domain_schema].create_table(
            Table.define_association([self.dataset_table, element_table]))
        self.model = self.catalog.getCatalogModel()
        self.dataset_table.annotations.update(
            generate_dataset_annotations(self.model))
        self.model.apply()
        return table

    def list_dataset_members(self, dataset_rid: RID) -> dict[Table, RID]:
        """
        Return a list of entities associated with a specific dataset.
        :param dataset_rid:
        :return:
        """

        try:
            if self.resolve_rid(dataset_rid).table != self.dataset_table:
                raise DerivaMLException(f'RID is not for a dataset: {dataset_rid}')
        except DerivaMLException:
            raise DerivaMLException(f'Invalid RID: {dataset_rid}')

        # Look at each of the element types that might be in the dataset and get the list of rid for them from
        # the appropriate association table.
        member_list = {}
        pb = self.pathBuilder
        for assoc_table in self.dataset_table.find_associations():
            other_fkey = assoc_table.other_fkeys.pop()
            self_fkey = assoc_table.self_fkey
            target_table = other_fkey.pk_table
            member_table = assoc_table.table

            if target_table.schema.name != self.domain_schema and target_table.name != "Dataset":
                # Look at domain tables and nested datasets.
                continue
            if target_table.name == "Dataset":
                # find_assoc gives us the keys in the wrong position.
                self_fkey, other_fkey = other_fkey, self_fkey

            target_path = pb.schemas[target_table.schema.name].tables[target_table.name]
            member_path = pb.schemas[member_table.schema.name].tables[member_table.name]
            # Get the names of the columns that we are going to need for linking
            member_link = tuple(c.name for c in next(iter(other_fkey.column_map.items())))

            path = pb.schemas[member_table.schema.name].tables[member_table.name].path
            path.filter(member_path.Dataset == dataset_rid)
            path.link(target_path,
                      on=(member_path.columns[member_link[0]] == target_path.columns[member_link[1]]))
            target_entities = path.entities().fetch()
            member_list.setdefault(target_table.name, []).extend(target_entities)
        return member_list

    def add_dataset_members(self, dataset_rid: Optional[RID], members: list[RID],
                            validate: bool = True) -> RID:
        """
        Add additional elements to an existing dataset.
        :param dataset_rid: RID of dataset to extend or None if new dataset is to be created.
        :param members: List of RIDs of me
        mbers to add to the  dataset.
        :param validate: Check rid_list to make sure elements are not already in the dataset.
        :return:
        """

        def check_dataset_cycle(member_rid, path=None):
            path = path or set(dataset_rid)
            return member_rid in path

        if validate:
            existing_rids = set(
                m
                for ms in self.list_dataset_members(dataset_rid).values()
                for m in ms
            )
            if overlap := set(existing_rids).intersection(members):
                raise DerivaMLException(f'Attempting to add existing member to dataset {dataset_rid}: {overlap}')

        # Now go through every rid to be added to the data set and sort them based on what association table entries
        # need to be made.
        dataset_elements = {}
        association_map = {a.other_fkeys.pop().pk_table.name: a.table.name for a in
                           self.dataset_table.find_associations()}
        # Get a list of all of the types of objects that can be linked to a dataset.
        for m in members:
            rid_info = self.resolve_rid(m)
            if rid_info.table.name not in association_map:
                raise DerivaMLException(f'RID table: {rid_info.table.name} not part of dataset')
            if rid_info.table == self.dataset_table and check_dataset_cycle(rid_info.rid):
                raise DerivaMLException("Creating cycle of datasets is not allowed")
            dataset_elements.setdefault(rid_info.table.name, []).append(rid_info.rid)
        # Now make the entries into the association tables.
        pb = self.pathBuilder
        for table, elements in dataset_elements.items():
            schema_path = pb.schemas[self.ml_schema if table == 'Dataset' else self.domain_schema]
            fk_column = 'Nested_Dataset' if table == 'Dataset' else table
            if len(elements):
                # Find out the name of the column in the association table.
                schema_path.tables[association_map[table]].insert(
                    [{'Dataset': dataset_rid, fk_column: e} for e in elements])
        return dataset_rid

    def add_execution(self, workflow_rid: str = '', datasets: List[str] = None,
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
        ml_schema_path = self.catalog.getPathBuilder().schemas[self.ml_schema]
        if workflow_rid:
            execution_rid = (
                ml_schema_path.Execution.insert([{'Description': description, 'Workflow': workflow_rid}]))[0]['RID']
        else:
            execution_rid = ml_schema_path.Execution.insert([{'Description': description}])[0]['RID']
        if datasets:
            ml_schema_path.Dataset_Execution.insert([{'Dataset': d, 'Execution': execution_rid} for d in datasets])
        return execution_rid

    def update_execution(self, execution_rid: RID, workflow_rid: RID = '', datasets: List[str] = None,
                         description: str = '') -> RID:
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
        schema_path.Execution.update([{'RID': execution_rid, 'Workflow': workflow_rid, 'Description': description}])
        if datasets:
            schema_path.Dataset_Execution.insert([{'Dataset': d, 'Execution': execution_rid} for d in datasets])
        return execution_rid

    def add_term(self,
                 table: str | Table,
                 term_name: str,
                 description: str,
                 synonyms: Optional[List[str]] = None,
                 exists_ok: bool = True) -> VocabularyTerm:
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
        table = self._get_table(table)
        pb = self.catalog.getPathBuilder()
        if not (self.is_vocabulary(table)):
            raise DerivaMLException(f'The table {table} is not a controlled vocabulary')

        schema_name = table.schema.name
        table_name = table.name
        try:
            term_id = VocabularyTerm.model_validate(
                pb.schemas[schema_name].tables[table_name].insert(
                    [{'Name': term_name, 'Description': description, 'Synonyms': synonyms}],
                    defaults={'ID', 'URI'})[0])
        except DataPathException:
            term_id = self.lookup_term(table, term_name)
            if not exists_ok:
                raise DerivaMLException(f'{term_name} already exists')
            # Check vocabulary
        return term_id

    def lookup_term(self, table: str | Table, term_name: str) -> VocabularyTerm:
        """
        Given a term name, return the the vocabolary record.  Can provide either the term name
         or a synaonom for the term.
        Args:
        - table_name (str): The name of the controlled vocabulary table.
        - term_name (str): The name of the term to look up.

        Returns:
        - str: The entry the associated term or synonym.

        Raises:
        - EyeAIException: If the schema or vocabulary table doesn't exist, or if the term is not
          found in the vocabulary.

        """
        vocab_table = self._get_table(table)
        if not self.is_vocabulary(vocab_table):
            raise DerivaMLException(f'The table {table} is not a controlled vocabulary')
        schema_name, table_name = vocab_table.schema.name, vocab_table.name
        schema_path = self.catalog.getPathBuilder().schemas[schema_name]
        for term in schema_path.tables[table_name].entities():
            if term_name == term['Name'] or (term['Synonyms'] and term_name in term['Synonyms']):
                return VocabularyTerm.model_validate(term)
        raise DerivaMLException(f'Term {term_name} is not in vocabulary {table_name}')

    def find_vocabularies(self) -> Iterable[Table]:
        """
        Return a list of all the controlled vocabulary tables in the domain schema.

        Returns:
         - List[str]: A list of table names representing controlled vocabulary tables in the schema.

        """
        return [t for s in self.model.schemas.values() for t in s.tables.values() if self.is_vocabulary(t)]

    def list_vocabulary_terms(self, table_name: str) -> Iterable[VocabularyTerm]:
        """
        Return the dataframe of terms that are in a vocabulary table.

        Args:
        - table_name (str): The name of the controlled vocabulary table.

        Returns:
        - Iterable: A iterable containing the terms in the specified controlled vocabulary table.

        Raises:
        - DerivaMLException: If the schema or vocabulary table doesn't exist, or if the table is not
          a controlled vocabulary.
        """
        pb = self.catalog.getPathBuilder()
        table = self._get_table(table_name)
        if not (self.is_vocabulary(table_name)):
            raise DerivaMLException(f'The table {table_name} is not a controlled vocabulary')

        return [VocabularyTerm(**v) for v in pb.schemas[table.schema.name].tables[table.name].entities().fetch()]

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

    def cite(self, entity: dict | str) -> str:
        """
        Return a citation URL for the provided entity.
        :param entity: A dict that contains the column values for a specific entity.
        :return:  The PID for the provided entity.
        """
        try:
            self.resolve_rid(rid := entity if isinstance(entity, str) else entity['RID'])
            return f"https://{self.host_name}/id/{self.catalog_id}/{rid}"
        except KeyError as e:
            raise DerivaMLException(f"Entity {e} does not have RID column")
        except DerivaMLException as _e:
            raise DerivaMLException("Entity RID does not exist")

    def user_list(self) -> list[dict[str, str]]:
        """
        Return a DataFrame containing user information of current catalog.

        Returns:
        - pd.DataFrame: DataFrame containing user information.

        """
        user_path = self.pathBuilder.public.ERMrest_Client.path
        return [{'ID': u['ID'], 'Full Name': u['Full_Name']} for u in user_path.entities().fetch()]

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
            raise DerivaMLException(f'Invalid URL: {url}')
        else:
            sha256_hash = hashlib.sha256()
            sha256_hash.update(response.content)
            checksum = 'SHA-256: ' + sha256_hash.hexdigest()
        return checksum

    def download_dataset_bag(self, dataset_rid: RID) -> tuple[str, str]:
        if not any([dataset_rid == ds['RID'] for ds in self.find_datasets()]):
            raise DerivaMLException(f'RID {dataset_rid} is not a dataset')

        # Put current download spec into a file
        with TemporaryDirectory() as tmp_dir:
            with open('download_spec.json', 'w+') as ds:
                json.dump(generate_dataset_download_spec(self.model), ds)
            downloader = GenericDownloader(
                server={"catalog_id": self.catalog_id, "protocol": "https", "host": self.host_name},
                config_file='download_spec.json',
                output_dir=tmp_dir,
                envars={"Dataset_RID": dataset_rid})
            result = downloader.download()
            bag_path = list(result.values())[0]["local_path"]
            checksum_value = compute_file_hashes('download_spec.json', hashes=['sha256'])['sha256'][0]
            bag_dir = self.cache_dir / f'{dataset_rid}_{checksum_value}'
            bag_dir.mkdir(parents=True, exist_ok=True)
            bag_file = f'{bag_dir}/Dataset_{dataset_rid}.zip'
            shutil.move(bag_path, bag_file)
        return bag_file, checksum_value

    def materialize_bdbag(self, bag: str | RID, execution_rid: Optional[RID] = None) -> tuple[Path, RID]:
        """
        Materialize a BDBag into the cache directory. Validate its contents and return the path to the bag, and its RID.

        Args:
        - minid (str): RID or minimum viable identifier (minid) of the bag.
        - execution_rid (str): Resource Identifier (RID) of the execution to report status to.  If None, status is
                                not updated.

        Returns:
        - tuple: Tuple containing the path to the bag and the RID of the associated dataset.

        Raises:
        - DerivaMLException: If there is an issue materializing the bag.

        """

        def fetch_progress_callback(current, total):
            msg = f'Materializing bag: {current} of {total} file(s) downloaded.'
            if execution_rid:
                self.update_status(Status.running, msg, execution_rid)
            logging.info(msg)
            return True

        def validation_progress_callback(current, total):
            msg = f'Validating bag: {current} of {total} file(s) validated.'
            if execution_rid:
                self.update_status(Status.running, msg, execution_rid)
            logging.info(msg)
            return True

        # request metadata
        if bag.upper().startswith('MINID'):
            r = requests.get(f'https://identifiers.org/{bag}', headers={'accept': 'application/json'})
            metadata = r.json()['metadata']
            dataset_rid = metadata['Dataset_RID'].split('@')[0]
            checksum_value = ''
            for checksum in r.json().get('checksums', []):
                if checksum.get('function') == 'sha256':
                    checksum_value = checksum.get('value')
                    break
            bag_path = bag
        else:
            dataset_rid = bag
            bag_path, checksum_value = self.download_dataset_bag(dataset_rid)
            print(f"downloaded bag {bag_path, checksum_value}")

        bag_dir = self.cache_dir / f'{bag}_{checksum_value}'
        bag_dir.mkdir(parents=True, exist_ok=True)
        validated_check = bag_dir / 'validated_check.txt'
        bags = [str(item) for item in bag_dir.iterdir() if item.is_dir()]
        if not bags:
            bag_path = bdb.materialize(bag_path,
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
        hs = HatracStore('https', self.host_name, self.credential)
        hs.get_obj(path=asset_url, destfilename=dest_filename)
        return Path(dest_filename)

    def upload_assets(self, assets_dir: str | Path) -> dict[str, FileUploadState]:
        """
        Upload assets from a directory.

        Args:
        - assets_dir (str): Directory containing the assets to upload.

        Returns:
        - dict: Results of the upload operation.

        Raises:
        - DerivaMLException: If there is an issue uploading the assets.

           """
        uploader = GenericUploader(server={'host': self.host_name, 'protocol': 'https', 'catalog_id': self.catalog_id})
        uploader.getUpdatedConfig()
        uploader.scanDirectory(assets_dir)
        results = deepcopy(uploader.uploadFiles())
        # results = {
        #     path: FileUploadState(state=result['State'], status=result['Status'], result=result['Result'])
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
            [{'RID': execution_rid, 'Status': self.status, 'Status_Detail': status_detail}]
        )

    def download_execution_files(self, table_name: str, file_rid: str, execution_rid='', dest_dir: str = '') -> Path:
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
            self.update_status(Status.running, f'Downloading {table_name}...', execution_rid)
            file_path = self.download_asset(file_url, str(dest_dir) + '/' + file_name)
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error, execution_rid)
            raise DerivaMLException(f'Failed to download the file {file_rid}. Error: {error}')

        if execution_rid != '':
            ass_table = table_name + '_Execution'
            ass_table_path = ml_schema_path.tables[ass_table]
            exec_file_exec_entities = ass_table_path.filter(ass_table_path.columns[table_name] == file_rid).entities()
            exec_list = [e['Execution'] for e in exec_file_exec_entities]
            if execution_rid not in exec_list:
                table_path = self.catalog.getPathBuilder().schemas[self.ml_schema].tables[ass_table]
                table_path.insert([{table_name: file_rid, 'Execution': execution_rid}])
        self.update_status(Status.running, f'Successfully download {table_name}...', execution_rid)
        return Path(file_path)

    def upload_execution_configuration(self, config: ExecutionConfiguration) -> RID:
        """
        Upload execution configuration to Execution_Metadata table with Execution Metadata Type = Execution_Config.

        Args:
        - config_file (str): Path to the configuration file.
        - desc (str): Description of the configuration.

        Raises:
        - DerivaMLException: If there is an issue uploading the configuration.

        """
        try:
            with NamedTemporaryFile('w', prefix='exec_config',
                                    suffix='.json',
                                    delete_on_close=False,
                                    delete=True) as fp:
                json.dump(config.model_dump_json(), fp)
                fp.close()
                configuration_rid = self._upload_execution_configuration_file(fp.name, description=config.description)
        except Exception as _e:
            raise DerivaMLException(f'Error in execution configuration upload')
        return configuration_rid

    def download_execution_configuration(self, configuration_rid: RID) -> ExecutionConfiguration:
        configuration = self.retrieve_rid(configuration_rid)
        with NamedTemporaryFile('w+', delete_on_close=False, suffix='.json') as dest_file:
            hs = HatracStore('https', self.host_name, self.credential)
            hs.get_obj(path=configuration['URL'], destfilename=dest_file.name)
            return ExecutionConfiguration.load_configuration(dest_file.name)

    def _upload_execution_configuration_file(self, config_file: str, description: str) -> RID:
        file_path = Path(config_file)
        file_name = file_path.name
        file_size = file_path.stat().st_size
        try:
            hs = HatracStore('https', self.host_name, self.credential)
            md5 = hash_utils.compute_file_hashes(config_file, ['md5'])['md5'][1]
            sanitized_filename = urlquote(re.sub('[^a-zA-Z0-9_.-]', '_', md5 + '.' + file_name))
            hatrac_path = f'/hatrac/execution_metadata/{sanitized_filename}'
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
            return list(ml_schema_path.tables['Execution_Metadata'].insert(
                [{'URL': hatrac_uri,
                  'Filename': file_name,
                  'Length': file_size,
                  'MD5': md5,
                  'Description': description,
                  'Execution_Metadata_Type': 'Execution Config'}]))[0]['RID']
        except Exception as e:
            error = format_exception(e)
            raise DerivaMLException(
                f'Failed to update Execution_Asset table with configuration file metadata. Error: {error}')

    def upload_execution_metadata(self, execution_rid: RID) -> dict[str, FileUploadState]:
        """
        Upload execution metadata at working_dir/Execution_metadata.

        Args:
        - execution_rid (str): Resource Identifier (RID) of the execution.

        Raises:
        - DerivaMLException: If there is an issue uploading the metadata.

        """
        self.update_status(Status.running, 'Uploading assets...', execution_rid)
        try:
            results = self.upload_assets(str(self.execution_metadata_path))
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error, execution_rid)
            raise DerivaMLException(
                f'Fail to upload the files in {self.execution_metadata_path}'
                f' to Execution_Metadata table. Error: {error}')

        else:
            ml_schema_path = self.catalog.getPathBuilder().schemas[self.ml_schema]
            a_table = list(self.model.schemas[self.ml_schema].tables['Execution_Metadata'].find_associations())[0].name
            meta_exec_entities = list(ml_schema_path.tables[a_table].filter(
                ml_schema_path.tables[a_table].Execution == execution_rid).entities().fetch())
            meta_list = [e['Execution_Metadata'] for e in meta_exec_entities]
            entities = []
            for metadata in results.values():
                if metadata['State'] == 0 and metadata['Result'] is not None:
                    rid = metadata['Result'].get('RID')
                    if (rid is not None) and (rid not in meta_list):
                        entities.append({'Execution_Metadata': rid, 'Execution': execution_rid})
        self.catalog.getPathBuilder().schemas[self.ml_schema].tables[a_table].insert(entities)
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
            self.update_status(Status.running, f'Uploading assets {folder_path}...', execution_rid)
            if folder_path.is_dir():
                try:
                    result = self.upload_assets(str(folder_path))
                    self.update_status(Status.running, 'Uploading assets...', execution_rid)
                except Exception as e:
                    error = format_exception(e)
                    self.update_status(Status.failed, error, execution_rid)
                    raise DerivaMLException(
                        f'Fail to upload the files in {folder_path} to Execution_Assets table. Error: {error}')
                else:
                    asset_exec_entities = ml_schema_path.Execution_Assets_Execution.filter(
                        ml_schema_path.Execution_Assets_Execution.Execution == execution_rid).entities()
                    assets_list = [e['Execution_Assets'] for e in asset_exec_entities]
                    entities = []
                    for asset in result.values():
                        if asset['State'] == 0 and asset['Result'] is not None:
                            rid = asset['Result'].get('RID')
                            if (rid is not None) and (rid not in assets_list):
                                entities.append({'Execution_Assets': rid, 'Execution': execution_rid})
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

        self.update_status(Status.running, 'Algorithm execution ended.', execution_rid)
        self.catalog.getPathBuilder().schemas[self.ml_schema].Execution.update(
            [{'RID': execution_rid, 'Duration': duration}])

    def initialize_execution(self, configuration: ExecutionConfiguration) -> ConfigurationRecord:
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
        configuration_rid = self.upload_execution_configuration(configuration)

        # Download configuration json file
        configuration_path = self.download_execution_files('Execution_Metadata', configuration_rid,
                                                           dest_dir=str(self.working_dir))
        # Check input configuration
        try:
            self.configuration = ExecutionConfiguration.model_validate(configuration)
            logging.info('Configuration validation successful!')
        except ValidationError as e:
            raise DerivaMLException(f'configuration validation failed: {e}')
        # Insert Execution
        execution_rid = self.add_execution(description=self.configuration.execution.description)
        # Insert terms
        self.update_status(Status.running, 'Inserting tags... ', execution_rid)
        vocabs = {}
        for term in configuration.workflow_terms:
            term_record = self.add_term(table=term.term,
                                        term_name=term.name,
                                        description=term.description,
                                        exists_ok=True)
            term_records = vocabs.get(term.term, [])
            term_records.append(term_record)
            vocabs[term.term] = term_records
        # Materialize bdbag
        dataset_rids = []
        bag_paths = []
        for url in self.configuration.bdbag_url:
            self.update_status(Status.running, f'Inserting bag {url}... ', execution_rid)
            bag_path, dataset_rid = self.materialize_bdbag(url, execution_rid)
            dataset_rids.append(dataset_rid)
            bag_paths.append(bag_path)
        # Insert workflow
        self.update_status(Status.running, 'Inserting workflow... ', execution_rid)
        try:
            workflow_rid = self.add_workflow(self.configuration.workflow.name,
                                             self.configuration.workflow.url,
                                             self.configuration.workflow.workflow_type,
                                             self.configuration.workflow.version,
                                             self.configuration.workflow.description)
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error, execution_rid)
            raise DerivaMLException(f'Failed to insert workflow. Error: {error}')
        # Update execution info
        execution_rid = self.update_execution(execution_rid, workflow_rid, dataset_rids,
                                              self.configuration.execution.description)
        self.update_status(Status.running, 'Execution created ', execution_rid)

        # Download model
        self.update_status(Status.running, 'Downloading models ...', execution_rid)
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
                file.write(str(package) + '\n')
        self.start_time = datetime.now()
        self.update_status(Status.running, 'Initialize status finished.', execution_rid)
        return configuration_records

    def execution(self, configuration: ConfigurationRecord) -> 'DerivaMlExec':
        """
        Start the execution by initializing the context manager DerivaMlExec.

        Args:
        - execution_rid (str): Resource Identifier (RID) of the execution.

        Returns:
        - DerivaMlExec: Execution object.

        """
        return DerivaMlExec(self, configuration)

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
            self.update_status(Status.failed, error, execution_rid)

    def upload_execution(self, configuration: ConfigurationRecord, clean_folder: bool = True) -> (
            dict)[str, dict[str, FileUploadState]]:
        """
        Upload all the assets and metadata associated with the current execution.

        Args:
        - execution_rid (str): Resource Identifier (RID) of the execution.

        Returns:
        - dict: Uploaded assets with key as assets' suborder name,
        values as an ordered dictionary with RID and metadata in the Execution_Assets table.

        """
        execution_rid = configuration.execution_rid
        try:
            uploaded_assets = self.upload_execution_assets(execution_rid)
            self.upload_execution_metadata(execution_rid)
            self.update_status(Status.completed,
                               'Successfully end the execution.',
                               execution_rid)
            if clean_folder:
                self._clean_folder_contents(self.execution_assets_path, execution_rid)
                self._clean_folder_contents(self.execution_metadata_path, execution_rid)
            return uploaded_assets
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error, execution_rid)


class DerivaMlExec:
    """
    Context manager for managing DerivaML execution.

    Args:
    - catalog_ml: Instance of DerivaML class.
    - execution_rid (str): Execution resource identifier.

    """

    def __init__(self, catalog_ml: DerivaML, configuration: ConfigurationRecord):
        self.configuration = configuration
        self.execution_rid = configuration.execution_rid
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
                                      'Start ML algorithm.',
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
                                          'Successfully run Ml.',
                                          self.execution_rid)
            self.catalog_ml.execution_end(self.execution_rid)
        else:
            self.catalog_ml.update_status(Status.failed,
                                          f'Exception type: {exc_type}, Exception value: {exc_value}',
                                          self.execution_rid)
            logging.error(f'Exception type: {exc_type}, Exception value: {exc_value}, Exception traceback: {exc_tb}')
            return False
