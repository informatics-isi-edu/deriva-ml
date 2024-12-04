import csv
from collections import defaultdict
from datetime import datetime
from bdbag import bdbag_api as bdb
from bdbag.fetch.fetcher import fetch_single_file
from deriva.core import ErmrestCatalog, get_credential, format_exception, urlquote, DEFAULT_SESSION_CONFIG
from deriva.core.datapath import DataPathException, _ResultSet
from deriva.core.datapath import _CatalogWrapper
from deriva.core.ermrest_catalog import ResolveRidResult
from deriva.core.ermrest_model import FindAssociationResult, Column, Table
from deriva.core.hatrac_store import HatracStore
from deriva.core.utils import hash_utils, mime_utils
from deriva.core.utils.hash_utils import compute_file_hashes
from deriva.transfer.download.deriva_download import GenericDownloader
from deriva.transfer.upload.deriva_upload import GenericUploader
from .deriva_definitions import Key, ColumnDefinition, RID, UploadState
from .dataset_bag import generate_dataset_download_spec
from .execution_configuration import ExecutionConfiguration
from .schema_setup.dataset_annotations import generate_dataset_annotations
from .schema_setup.system_terms import MLVocab, ExecMetadataVocab
from .upload import is_feature_dir, is_feature_asset_dir
from .upload import execution_metadata_dir, execution_assets_dir, asset_dir, execution_root
from .upload import table_path, bulk_upload_configuration
from .upload import feature_root, feature_asset_dir, feature_value_path

# from enum import Enum, StrEnum
try:
    from enum import Enum, StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass

import getpass
import hashlib
from importlib.metadata import distributions
from itertools import chain
import json
import logging
import os
from pydantic import BaseModel, ValidationError, Field, create_model, computed_field
from pathlib import Path
import re
import requests
import shutil
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import List, Optional, Any, Iterable, Iterator, Type, ClassVar
from types import UnionType
import warnings

# We are going to use schema as a field name and this collides with method in pydantic base class
warnings.filterwarnings('ignore',
                        message='Field name "schema"',
                        category=Warning,
                        module='pydantic')

warnings.filterwarnings('ignore',
                        message='fields may not start with an underscore',
                        category=Warning,
                        module='pydantic')


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


class FeatureRecord(BaseModel):
    # model_dump of this feature should be compatible with feature table columns.
    Execution: str
    Feature_Name: str
    feature: ClassVar[Optional['Feature']] = None

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def feature_columns(cls) -> set[Column]:
        """
        Return the names of all  the columns that define the value of a feature.
        :return: set of feature column names.
        """
        return cls.feature.feature_columns

    @classmethod
    def asset_columns(cls) -> set[Column]:
        """
        Return the names of all  the columns of a feature that are assets.
        :return:  set of asset column names.
        """
        return cls.feature.asset_columns

    @classmethod
    def term_columns(cls) -> set[Column]:
        """
        Return the names of all  the columns of a feature that are controlled vocabulary terms.
        :return: set of term column names.
        """
        return cls.feature.term_columns

    @classmethod
    def value_columns(cls) -> set[Column]:
        """
        Return the names of all  the columns of a feature that are scale  values.
        :return: set of value column names.
        """
        return cls.feature.value_columns

class ConfigurationRecord(BaseModel):
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
    caching_dir: Path
    working_dir: Path
    execution_rid: RID
    workflow_rid: RID
    bag_paths: list[Path]
    asset_paths: list[Path]
    configuration: ExecutionConfiguration
    _ml_object: 'DerivaML'

    class Config:
        protected_namespaces = ()
        arbitrary_types_allowed = True

    @staticmethod
    def ConfigurationRecordFactory(ml_object: 'DerivaML', **kwargs) -> 'ConfigurationRecord':
        cr = ConfigurationRecord(**kwargs)
        cr._ml_object = ml_object
        cr.model_config['frozen'] = True
        return cr

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
        Given a collection of Feature records, write out a CSV file is the appropriate assets directory so that this
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




class Feature:
    """
    Wrapper for results of Table.find_associations()
    """

    def __init__(self, atable: FindAssociationResult):
        self.feature_table = atable.table
        self.target_table = atable.self_fkey.pk_table
        self.feature_name = atable.table.columns['Feature_Name'].default

        def is_asset(table):
            asset_columns = {'Filename', 'URL', 'Length', 'MD5', 'Description'}
            return asset_columns.issubset({c.name for c in table.columns})

        def is_vocabulary(table):
            vocab_columns = {'NAME', 'URI', 'SYNONYMS', 'DESCRIPTION', 'ID'}
            return vocab_columns.issubset({c.name.upper() for c in table.columns})

        skip_columns = {'RID', 'RMB', 'RCB', 'RCT', 'RMT', 'Feature_Name', self.target_table.name, 'Execution' }
        self.feature_columns =  {c for c in self.feature_table.columns if c.name not in skip_columns}

        assoc_fkeys = {atable.self_fkey} |  atable.other_fkeys

        # Determine the role of each column in the feature outside of the FK columns.
        self.asset_columns = {fk.foreign_key_columns[0] for fk in self.feature_table.foreign_keys if
                 fk not in assoc_fkeys  and is_asset(fk.pk_table)}

        self.term_columns =  {fk.foreign_key_columns[0] for fk in self.feature_table.foreign_keys if
                              fk not in assoc_fkeys and is_vocabulary(fk.pk_table)}

        self.value_columns = self.feature_columns - (self.asset_columns | self.term_columns)

    def feature_record_class(self) -> type[FeatureRecord]:
        """"
        Create a pydantic model for entries into the specified feature table
        :return: A Feature class that can be used to create instances of the feature.

        """

        def map_type(c: Column) -> UnionType | Type[str] | Type[int] | Type[float]:
            """
            Map a dervia type into a pydantic model type.
            :param c:  column to be mapped
            :return: pydantic model type
            """
            if c.name in {c.name for c in self.asset_columns}:
                return str | Path
            match c.type.typename:
                case 'text':
                    return str
                case 'int2' | 'int4' | 'int8':
                    return int
                case 'float4' | 'float8':
                    return float
                case _:
                    return str

        featureclass_name = f'{self.target_table.name}Feature{self.feature_name}'

        # Create feature class. To do this, we must determine the python type for each column and also if the
        # column is optional or not based on its nulliblity.
        feature_columns = ({c.name: (Optional[map_type(c)] if c.nullok else map_type(c), c.default or None)
                              for c in self.feature_columns} |
                           {
                               'Feature_Name': (str, self.feature_name),  # Set default value for Feature_Name
                               self.target_table.name: (str, ...)
                            }
        )
        docstring = f'Class to capture fields in a feature {self.feature_name} on table {self.target_table}. Feature columns include:\n'
        docstring += '\n'.join([f'    {c.name}' for c in self.feature_columns])

        model = create_model(featureclass_name, __base__=FeatureRecord, __doc__=docstring, **feature_columns)
        model.feature = self  # Set value of class variable within the feature class definition.

        return model

    def __repr__(self) -> str:
        return (f'Feature(target_table={self.target_table.name}, feature_name={self.feature_name}, '
                f'feature_table={self.feature_table.name})')


class FileUploadState(BaseModel):
    state: UploadState
    status: str
    result: Any

    @computed_field
    @property
    def rid(self) -> Optional[RID]:
        return self.result and self.result['RID']


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
                 ml_schema='deriva-ml',
                 logging_level=logging.WARNING):
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
            tdir = TemporaryDirectory()
            self.cache_dir = Path(tdir.name)
        default_workdir = self.__class__.__name__ + '_working'
        if working_dir:
            self.working_dir = Path(working_dir).joinpath(getpass.getuser(), default_workdir)
        else:
            tdir = tdir or TemporaryDirectory()
            self.working_dir = Path(tdir.name) / default_workdir
        self.working_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
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
        """
        Get a new instance of a pathbuilder object.
        :return: pathbuilder object.
        """
        return self.catalog.getPathBuilder()

    @property
    def dataset_table(self) -> Table:
        return self.model.schemas[self.ml_schema].tables['Dataset']

    @property
    def domain_path(self):
        """
        Get a new instance of a pathBuilder object to the domain schema.

        :return: A new instance of a pathBuilder path to the domain schema.
        """

        return self.pathBuilder.schemas[self.domain_schema]

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

    def model_dir(self, execution_rid: RID) -> Path:
        """
        Return the directory in which models downloaded as part of initializing an execution are placed.

        :param execution_rid: Execution RID for the current execution.
        :return: PathLib path object to model directory.
        """
        path = self.working_dir / execution_rid / 'models'
        path.mkdir(parents=True, exist_ok=True)
        return path

    def table_path(self, table: str | Table) -> Path:
        """
        Return a local file path in which to place a CSV to add values to a table on upload.

        :param table:
        :return:
        """
        return table_path(self.working_dir, schema=self.domain_schema, table=self._get_table(table).name)

    def asset_directory(self, table: str | Table, prefix: str | Path = None) -> Path:
        """
        Return a local file path in which to place a files for an asset table.  This needs to be kept in sync with
        bulk_upload specification
        :param table:
        :param prefix: Location of where to place files.  Defaults to execution_assets_path.
        """
        table = self._get_table(table)
        prefix = Path(prefix) or self.working_dir
        return asset_dir(prefix, table.schema.name, table.name)

    def download_dir(self, cached: bool = False) -> Path:
        return self.cache_dir if cached else self.working_dir

    def chaise_url(self, table: str | Table) -> str:
        """
        Return a Chaise URL to the specified table.

        :param table: Table to be browsed
        :return: URL to the table in Chaise format.
        """
        table = self._get_table(table)
        uri = self.catalog.get_server_uri().replace('ermrest/catalog/', 'chaise/recordset/#')
        return f'{uri}/{urlquote(table.schema.name)}:{urlquote(table.name)}'

    def cite(self, entity: dict | str) -> str:
        """
        Return a citation URL for the provided entity.

        :param entity: A dict that contains the column values for a specific entity.
        :return:  The PID for the provided entity.
        """
        try:
            self.resolve_rid(rid := entity if isinstance(entity, str) else entity['RID'])
            return f"https://{self.host_name}/id/{self.catalog_id}/{rid}@{self.catalog.latest_snapshot().snaptime}"
        except KeyError as e:
            raise DerivaMLException(f"Entity {e} does not have RID column")
        except DerivaMLException as _e:
            raise DerivaMLException("Entity RID does not exist")

    def user_list(self) -> list[dict[str, str]]:
        """
        Return a list containing user information of current catalog.

        :return: a list of dictionaries containing user information.

        """
        user_path = self.pathBuilder.public.ERMrest_Client.path
        return [{'ID': u['ID'], 'Full_Name': u['Full_Name']} for u in user_path.entities().fetch()]

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

    def add_page(self, title: str, content: str) -> None:
        self.pathBuilder.www.tables[self.domain_schema].insert([{'Title': title, 'Content': content}])

    def create_vocabulary(self, vocab_name: str, comment='', schema=None) -> Table:
        """
        Create a controlled vocabulary table with the given vocab name.
        :param vocab_name: Name of the controlled vocabulary table.
        :param comment: Description of the vocabulary table.
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

    def create_asset(self, asset_name: str, comment='', schema: str = None) -> Table:
        """
        Create an asset table with the given asset name.

        :param asset_name: Name of the asset table.
        :param comment: Description of the asset table.
        :param schema: Schema in which to create the asset table.  Defaults to domain_schema.
        :return: Table object for the asset table.
        """
        schema = schema or self.domain_schema
        asset_table = self.model.schemas[schema].create_table(
            Table.define_asset(schema, asset_name, comment=comment))
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

        # Check to make sure that the workflow is not already in the table. If it's not, add it.
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
                MLVocab.workflow_type: self.lookup_term(MLVocab.workflow_type, workflow_type).name}
            workflow_rid = ml_schema_path.Workflow.insert([workflow_record])[0]['RID']

        return workflow_rid

    def create_feature(self,
                       target_table: Table | str,
                       feature_name: str,
                       terms: list[Table | str] = None,
                       assets: list[Table | str] = None,
                       metadata: Iterable[ColumnDefinition | Table | Key | str] = None,
                       optional: Optional[list[str]] = None,
                       comment: str = '') -> type[FeatureRecord]:
        """
        Create a new feature that can be associated with a table. The feature can associate a controlled
        vocabulary term, an asset, or any other values with a specific instance of an object and  execution.

        :param feature_name: Name of the new feature to be defined
        :param target_table: table name or object on which the feature is to be associated
        :param terms: List of controlled vocabulary terms that will be part of the feature value
        :param assets: List of asset table names or objects that will be part of the feature value
        :param metadata: List of other value types that are associated with the feature
        :param optional: List of columns that are optional in the feature
        :param comment:
        :return: A Feature class that can be used to create instances of the feature.
        :raise DerivaException: If the feature cannot be created.
        """

        terms = terms or []
        assets = assets or []
        metadata = metadata or []
        optional = optional or []

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
        target_table = self._get_table(target_table)
        execution = self.model.schemas[self.ml_schema].tables['Execution']
        feature_name_table = self.model.schemas[self.ml_schema].tables['Feature_Name']
        feature_name_term = self.add_term('Feature_Name', feature_name, description=comment)
        atable_name = f'Execution_{target_table.name}_{feature_name_term.name}'
        
        # Now create the association table that implements the feature.
        atable = self.model.schemas[self.domain_schema].create_table(
            target_table.define_association(
                table_name=atable_name,
                associates=[execution, target_table, feature_name_table],
                metadata=[normalize_metadata(m) for m in chain(assets, terms, metadata)],
                comment=comment
            )
        )
        # Now set optional terms.
        for c in optional:
            atable.columns[c].alter(nullok=True)
        atable.columns['Feature_Name'].alter(default=feature_name_term.name)
        return self.feature_record_class(target_table, feature_name)

    def feature_record_class(self, table: str | Table, feature_name: str) -> type[FeatureRecord]:
        """"
        Create a pydantic model for entries into the specified feature table.  For information on how to
        See the pydantic documentation for more details about the pydantic model.

        :param table: table name or object on which the feature is to be associated
        :param feature_name: name of the feature to be created
        :return: A Feature class that can be used to create instances of the feature.

        """
        return self.lookup_feature(table,  feature_name).feature_record_class()

    def drop_feature(self, table: Table | str, feature_name: str) -> bool:
        table = self._get_table(table)
        try:
            feature = next(f for f in self.find_features(table) if f.feature_name == feature_name)
            feature.feature_table.drop()
            return True
        except StopIteration:
            return False

    def lookup_feature(self, table: str | Table, feature_name: str) -> Feature:
        """
        Lookup the named feature associated with the provided table.
        :param table:
        :param feature_name:
        :return: A Feature class that represents the requested feature.
        :raise DerivaMLException: If the feature cannot be found.
        """
        table = self._get_table(table)
        try:
            return [f for f in self.find_features(table) if f.feature_name == feature_name][0]
        except IndexError:
            raise DerivaMLException(f"Feature {table.name}:{feature_name} doesn't exist.")

    def find_features(self, table: Table | str) -> Iterable[Feature]:
        """
        List the names of the features in the specified table.

        :param table: The table to find features for.
        :return: An iterable of FeatureResult instances that describe the current features in the table.
        """
        table = self._get_table(table)

        def is_feature(a: FindAssociationResult) -> bool:
            # return {'Feature_Name', 'Execution'}.issubset({c.name for c in a.table.columns})
            return {'Feature_Name',
                    'Execution',
                    a.self_fkey.foreign_key_columns[0].name}.issubset({c.name for c in a.table.columns})

        return [
            Feature(a) for a in table.find_associations(min_arity=3, max_arity=3, pure=False) if is_feature(a)
        ]

    def add_features(self, features: Iterable[FeatureRecord]) -> int:
        """
        Add a set of new feature values to the catalog.
        :return: Number of attributes added
        """
        features = list(features)
        feature_table = features[0].feature.feature_table
        feature_path = self.pathBuilder.schemas[feature_table.schema.name].tables[feature_table.name]
        entries = feature_path.insert(f.model_dump() for f in features)
        return len(entries)

    def list_feature_values(self, table: Table | str, feature_name: str) -> _ResultSet:
        """
        Return a dataframe containing all values of a feature associated with a table.
        :param table:
        :param feature_name:
        :return:
        """
        table = self._get_table(table)
        feature = self.lookup_feature(table, feature_name)
        pb = self.catalog.getPathBuilder()
        return pb.schemas[feature.feature_table.schema.name].tables[feature.feature_table.name].entities().fetch()

    def create_dataset(self, ds_type: str | list[str], description: str) -> RID:
        """
        Create a new dataset from the specified list of RIDs.
        :param ds_type: One or more dataset types.  Must be a term from the DatasetType controlled vocabulary.
        :param description:  Description of the dataset.
        :return: New dataset RID.
        """
        # Create the entry for the new dataset and get its RID.
        ds_types = [ds_type] if isinstance(ds_type, str) else ds_type

        pb = self.pathBuilder
        for ds_type in ds_types:
            if not self.lookup_term(MLVocab.dataset_type, ds_type):
                raise DerivaMLException(f'Dataset type must be a vocabulary term.')
        dataset_table_path = (
            pb.schemas[self.dataset_table.schema.name].tables)[self.dataset_table.name]
        dataset = dataset_table_path.insert([{'Description': description,
                                              MLVocab.dataset_type: ds_type}])[0]['RID']

        # Get the name of the association table between dataset and dataset_type.
        atable = next(self.model.schemas[self.ml_schema].tables[MLVocab.dataset_type].find_associations()).name
        pb.schemas[self.ml_schema].tables[atable].insert(
            [{MLVocab.dataset_type: ds_type, 'Dataset': dataset} for ds_type in ds_types])
        return dataset

    def find_datasets(self) -> Iterable[dict[str, Any]]:
        """
        Returns a list of currently available datasets.
        :return:
        """
        # Get datapath to all the tables we will need: Dataset, DatasetType and the association table.
        pb = self.pathBuilder
        dataset_path = pb.schemas[self.dataset_table.schema.name].tables[self.dataset_table.name]
        atable = next(self.model.schemas[self.ml_schema].tables[MLVocab.dataset_type].find_associations()).name
        ml_path = pb.schemas[self.ml_schema]
        atable_path = ml_path.tables[atable]

        # Get a list of all the dataset_type values associated with this dataset.
        datasets = []
        for dataset in dataset_path.entities().fetch():
            ds_types = atable_path.filter(atable_path.Dataset == dataset['RID']).attributes(
                atable_path.Dataset_Type).fetch()
            datasets.append(dataset | {MLVocab.dataset_type: [ds[MLVocab.dataset_type] for ds in ds_types]})
        return datasets

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
            tpath = schema_path.tables[assoc_table.name]
            dataset_column_path = tpath.columns[assoc_table.self_fkey.columns[0].name]
            dataset_entries = tpath.filter(dataset_column_path == dataset_rid)
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
        :return: An iterable of Table objects that can be included as an element of a dataset.
        """

        def domain_table(table: Table) -> bool:
            return table.schema.name == self.domain_schema or table.name == self.dataset_table.name

        return [t for a in self.dataset_table.find_associations() if domain_table(t := a.other_fkeys.pop().pk_table)]

    def add_dataset_element_type(self, element: str | Table) -> Table:
        """
        A dataset is a heterogeneous collection of object, each of which comes from a different table. This
        routine makes it possible to add objects from the specified table to a dataset.

        :param element: Name or the table or table object that is to be added to the dataset.
        :return: The table object that was added to the dataset.
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

    def list_dataset_parent(self, dataset_rid: RID) -> Optional[RID]:
        """
        Given a dataset RID, return a RID of the parent dataset.
        :param dataset_rid:
        :return: RID of the parent dataset.
        """
        rid_record = self.resolve_rid(dataset_rid)
        if rid_record.table.name != self.dataset_table.name:
            raise DerivaMLException(f'RID: {dataset_rid} does not belong to dataset {self.dataset_table.name}')
        return self.retrieve_rid(dataset_rid)['Dataset_Parent']


    def list_dataset_children(self, dataset_rid: RID) -> list[RID]:
        """
        Given a dataset RID, return a RID of the parent dataset.
        :param dataset_rid:
        :return: RID of the parent dataset.
        """
        return self.list_dataset_members(dataset_rid)['Dataset']

    def is_nested_dataset(self, dataset_rid) -> dict[str, list[dict[str, list]]]:
        """
        Return the structure of a nested dataset, the result is a dictionary whose key is a dataset rid and whose
        value is the list of children datasets.
        :param dataset_rid:
        :return:
        """
        return {dataset_rid: [self.is_nested_dataset(d) for d in self.list_dataset_children(dataset_rid)]}

    def list_dataset_members(self, dataset_rid: RID) -> dict[str, list[RID]]:
        """
        Return a list of entities associated with a specific dataset.
        :param dataset_rid:

        :return: Dictionary of entities associated with a specific dataset.  Key is the table from which the elements
                 were taken.
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
                # find_assoc gives us the keys in the wrong position, so swap.
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
        :param members: List of RIDs of members to add to the  dataset.
        :param validate: Check rid_list to make sure elements are not already in the dataset.
        :return:
        """

        members = set(members)

        def check_dataset_cycle(member_rid, path=None):
            path = path or set(dataset_rid)
            return member_rid in path

        if validate:
            existing_rids = set(
                m['RID']
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
        # Get a list of all the types of objects that can be linked to a dataset.
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

    def _add_execution(self, workflow_rid: str = '', datasets: List[str] = None,
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

    def _update_execution(self, execution_rid: RID, workflow_rid: RID = '', datasets: List[str] = None,
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
        schema_path = self.pathBuilder.schemas[self.ml_schema]
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
        Given a term name, return the vocabulary record.  Can provide either the term name
         or a synonym for the term.
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

    def list_vocabulary_terms(self, table_name: str) -> list[VocabularyTerm]:
        """
        Return an list of terms that are in a vocabulary table.

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

    def download_dataset_bag(self, dataset_rid: RID | str) -> tuple[Path, RID]:
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

    def materialize_dataset_bag(self, bag: str | RID, execution_rid: Optional[RID] = None) -> tuple[Path, RID]:
        """
        Materialize a dataset bag into a local directory
        :param bag: A MINID to an existing bag or a RID of the dataset that should be downloaded.
        :param execution_rid:  RID of the execution for which this bag should be materialized. Used to update status.
        :return:
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
        bag_path, dataset_rid = self.download_dataset_bag(bag)
        bag_dir = bag_path.parent
        validated_check = bag_dir / 'validated_check.txt'

        # If this bag has already been validated, our work is done.  Otherwise, materialize the bag.
        if not validated_check.exists():
            bdb.materialize(bag_path.as_posix(),
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

    def upload_asset(self, file: str | Path, table: Table | str, **kwargs) -> dict:
        """
        Upload the specified file into Hatrac and update the associated asset table.
        :param file: path to the file to upload.
        :param table: Name of the asset table
        :param kwargs: Keyword arguments for values of additional columns to be added to the asset table.
        :return:
        """
        table = self._get_table(table)
        if not self.is_asset(table):
            raise DerivaMLException(f'Table {table} is not an asset table.')

        credential = self.model.catalog.deriva_server.credentials
        file_path = Path(file)
        file_name = file_path.name
        file_size = file_path.stat().st_size
        # Get everything up to the filename  part of the
        hatrac_path = f'/hatrac/{table.name}/'
        hs = HatracStore('https', self.host_name, credential)
        md5_hashes = hash_utils.compute_file_hashes(file, ['md5'])['md5']
        sanitized_filename = urlquote(re.sub('[^a-zA-Z0-9_.-]', '_', md5_hashes[0] + '.' + file_name))
        hatrac_path = f'{hatrac_path}{sanitized_filename}'

        try:
            # Upload the file to hatrac.
            hatrac_uri = hs.put_obj(hatrac_path,
                                    file,
                                    md5=md5_hashes[1],
                                    content_type=mime_utils.guess_content_type(file),
                                    content_disposition="filename*=UTF-8''" + file_name)
        except Exception as e:
            raise e
        try:
            # Now update the asset table.
            ipath = self.pathBuilder.schemas[table.schema.name].tables[table.name]
            return list(ipath.insert(
                [{'URL': hatrac_uri,
                  'Filename': file_name,
                  'Length': file_size,
                  'MD5': md5_hashes[0]} | kwargs]))[0]
        except Exception as e:
            raise e

    def upload_assets(self, assets_dir: str | Path) -> dict[str, FileUploadState]:
        """
        Upload assets from a directory. This routine assumes that the current upload specification includes a
        configuration for the specified directory.  Every asset in the specified directory is uploaded

        :param:
        - assets_dir (str): Directory containing the assets to upload.

        :returns:
        - dict: Results of the upload operation.

        raises:
        - DerivaMLException: If there is an issue uploading the assets.

           """
        with TemporaryDirectory() as temp_dir:
            spec_file = f'{temp_dir}/config.json'
            with open(spec_file, 'w+') as cfile:
                json.dump(bulk_upload_configuration, cfile)
            uploader = GenericUploader(server={'host': self.host_name, 'protocol': 'https', 'catalog_id': self.catalog_id},
                                       config_file=spec_file)
            uploader.getUpdatedConfig()
            uploader.scanDirectory(assets_dir)
            results = {
                 path: FileUploadState(state=UploadState(result['State']), status=result['Status'], result=result['Result'])
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
        self.pathBuilder.schemas[self.ml_schema].Execution.update(
            [{'RID': execution_rid, 'Status': self.status, 'Status_Detail': status_detail}]
        )

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
            self.update_status(Status.running, f'Downloading {table.name}...', execution_rid)
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
        self.update_status(Status.running, f'Successfully download {table.name}...', execution_rid)
        return Path(file_path)

    def _upload_execution_asset(self, file, asset_type: str) -> dict[str, Any]:
        return self.upload_asset(file, "Execution_Assets", Execution_Asset_Type=asset_type)

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
            fp = NamedTemporaryFile('w+', prefix='exec_config', suffix='.json', delete=False)
            json.dump(config.model_dump_json(), fp)
            fp.close()
            configuration_rid = self._upload_execution_configuration_file(fp.name, description=config.description)
            os.remove(fp.name)
        except Exception as _e:
            raise DerivaMLException(f'Error in execution configuration upload')
        return configuration_rid

    def download_execution_configuration(self, configuration_rid: RID) -> ExecutionConfiguration:
        """
        Create an ExecutionConfiguration object from a catalog RID that points to a JSON representation of that
        configuration in hatrac

        :param configuration_rid:  RID that should be to an asset table that refers to an execution configuration
        :return: A ExecutionConfiguration object for configured by the parameters in the configuration file.
        """
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
                  'Execution_Metadata_Type': ExecMetadataVocab.execution_config}]))[0]['RID']
        except Exception as e:
            error = format_exception(e)
            raise DerivaMLException(
                f'Failed to update Execution_Asset table with configuration file metadata. Error: {error}')

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

    def _update_execution_asset_table(self, execution_rid: RID, assets: dict[str, FileUploadState]) -> None:
        """
        Assets associated with an execution must be linked to an execution entity after they are uploaded into
        the catalog. This routine takes a list of uploaded assets and makes that association.
        :param execution_rid:
        :param assets:
        :return:
        """
        ml_schema_path = self.pathBuilder.schemas[self.ml_schema]
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

    def _upload_execution_dirs(self, configuration: ConfigurationRecord) -> dict[str, FileUploadState]:
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
            self.update_status(Status.running, 'Uploading execution assets...', configuration.execution_rid)
            execution_asset_files = self.upload_assets(configuration._execution_assets_dir)
            self._update_execution_asset_table(configuration.execution_rid, execution_asset_files)
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
            self.update_status(Status.failed, error, configuration.execution_rid)
            raise DerivaMLException(
                f'Fail to upload execution metadata. Error: {error}')

        self.update_status(Status.running, f'Updating features...', configuration.execution_rid)
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
                    self.update_status(Status.running, f'Uploading feature {m["feature_name"]}...',
                                       configuration.execution_rid)
                    feature_assets[m['target_table'], m['feature_name']] = self.upload_assets(p)
                    results |= feature_assets[m['target_table'], m['feature_name']]
                except Exception as e:
                    error = format_exception(e)
                    self.update_status(Status.failed, error, configuration.execution_rid)
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

        self.update_status(Status.running, f'Upload assets complete', configuration.execution_rid)
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
        Initialize the execution by a configuration  in the Execution_Metadata table.
        Setup working directory and download all the assets and data.

        :param configuration: Configuration to initialize the execution with.
        :return: ConfigurationRecord:
        :raise DerivaMLException: If there is an issue initializing the execution.
        """

        try:
            self.configuration = ExecutionConfiguration.model_validate(configuration)
            logging.info('Configuration validation successful!')
        except ValidationError as e:
            raise DerivaMLException(f'configuration validation failed: {e}')

        # Insert Execution
        execution_rid = self._add_execution(description=self.configuration.execution.description)
        # Insert workflow
        try:
            self.update_status(Status.running, 'Inserting workflow... ', execution_rid)
            workflow_rid = self._add_workflow(self.configuration.workflow.name,
                                              self.configuration.workflow.url,
                                              self.configuration.workflow.workflow_type,
                                              self.configuration.workflow.version,
                                              self.configuration.workflow.description)
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error, execution_rid)
            raise DerivaMLException(f'Failed to insert workflow. Error: {error}')

        # Materialize bdbag
        dataset_rids = []
        bag_paths = []
        for bag in self.configuration.bdbags:
            self.update_status(Status.running, f'Materialize bag {bag}... ', execution_rid)
            bag_path, dataset_rid = self.materialize_dataset_bag(bag, execution_rid)
            dataset_rids.append(dataset_rid)
            bag_paths.append(bag_path)

        # Download model
        self.update_status(Status.running, 'Downloading models ...', execution_rid)
        model_path = self.model_dir(execution_rid).as_posix()
        model_paths = [self._download_execution_file(
            file_rid=m,
            execution_rid=execution_rid,
            dest_dir=model_path) for m in configuration.models]
        configuration_record = ConfigurationRecord.ConfigurationRecordFactory(
            self,
            execution_rid=execution_rid,
            caching_dir=self.cache_dir,
            working_dir=self.working_dir,
            workflow_rid=workflow_rid,
            bag_paths=bag_paths,
            asset_paths=model_paths,
            configuration=configuration)

        # Save configuration details for later upload
        exec_config_path = ExecMetadataVocab.execution_config.value
        cfile = configuration_record.execution_metadata_path(exec_config_path) / "configuration.json"
        with open(cfile, 'w',  encoding="utf-8") as config_file:
            json.dump(self.configuration.model_dump_json(), config_file)

        # Update execution info
        execution_rid = self._update_execution(execution_rid, workflow_rid, dataset_rids,
                                              self.configuration.execution.description)
        self.update_status(Status.running, 'Execution created ', execution_rid)

        # save runtime env
        runtime_env_path = ExecMetadataVocab.runtime_env.value
        runtime_env_dir = configuration_record.execution_metadata_path(runtime_env_path)
        with NamedTemporaryFile('w+', dir=runtime_env_dir, prefix='environment_snapshot_',
                                suffix='.txt', delete=False) as fp:
            for dist in distributions():
                fp.write(f"{dist.metadata['Name']}=={dist.version}\n")
        self.start_time = datetime.now()
        self.update_status(Status.running, 'Initialize status finished.', execution_rid)
        return configuration_record

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

    def upload_execution(self, configuration: ConfigurationRecord, clean_folder: bool = True) -> None:
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
            #uploaded_assets =
            self._upload_execution_dirs(configuration)
            self.update_status(Status.completed,
                               'Successfully end the execution.',
                               execution_rid)
            if clean_folder:
                self._clean_folder_contents(configuration.execution_root, execution_rid)
        except Exception as e:
            error = format_exception(e)
            self.update_status(Status.failed, error, execution_rid)
            raise e




class DerivaMlExec:
    """
    Context manager for managing DerivaML execution.  Provides status updates.

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
