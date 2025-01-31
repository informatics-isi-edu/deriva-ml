"""
`deriva_ml_base.py` is the core module for the Deriva ML project.  This module implements the DeriaML class, which is
the primary interface to the Deriva based catalogs.  The module also implements the Feature and Vocabulary functions
in the DerivaML.

DerivaML and its associated classes all depend on a catalog that implements a `deriva-ml` schema with tables and
relationships that follow a specific data model.

"""

import getpass
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from enum import Enum
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
from types import UnionType
from typing import Optional, Any, Iterable, Type, ClassVar, TYPE_CHECKING

import requests
from bdbag import bdbag_api as bdb
from bdbag.fetch.fetcher import fetch_single_file
from deriva.core import (
    ErmrestCatalog,
    get_credential,
    format_exception,
    urlquote,
    DEFAULT_SESSION_CONFIG,
)
from deriva.core.datapath import DataPathException, _ResultSet
from deriva.core.datapath import _CatalogWrapper
from deriva.core.ermrest_catalog import ResolveRidResult
from deriva.core.ermrest_model import FindAssociationResult, Column, Key, Table
from deriva.core.hatrac_store import HatracStore
from deriva.core.utils import hash_utils, mime_utils
from deriva.core.utils.hash_utils import compute_file_hashes
from deriva.transfer.download.deriva_download import GenericDownloader
from deriva.transfer.upload.deriva_upload import GenericUploader
from pydantic import BaseModel, Field, create_model, validate_call, ConfigDict

from .execution_configuration import ExecutionConfiguration
from .schema_setup.dataset_annotations import generate_dataset_annotations
from .upload import asset_dir
from .upload import table_path, bulk_upload_configuration
from .dataset import Dataset
from .deriva_definitions import ColumnDefinition
from .deriva_definitions import MLVocab, ExecMetadataVocab
from .deriva_definitions import (
    RID,
    UploadState,
    Status,
    FileUploadState,
    DerivaMLException,
    ML_SCHEMA,
)

if TYPE_CHECKING:
    from .execution import Execution


class VocabularyTerm(BaseModel):
    """An entry in a vocabulary table.

    Attributes:
       name: Name of vocabulary term
       synonyms: List of alternative names for the term
       id: CURI identifier for the term
       uri: Unique URI for the term.
       description: A description of the meaning of the term
       rid: Resource identifier assigned to the term

    Args:

    Returns:

    """

    name: str = Field(alias="Name")
    synonyms: Optional[list[str]] = Field(alias="Synonyms")
    id: str = Field(alias="ID")
    uri: str = Field(alias="URI")
    description: str = Field(alias="Description")
    rid: str = Field(alias="RID")

    class Config:
        """ """

        extra = "ignore"


class SemanticVersion(Enum):
    """Simple enumeration for semantic versioning."""

    major = "major"
    minor = "minor"
    patch = "patch"


class FeatureRecord(BaseModel):
    """Base class for feature records.  Feature records are pydantic models which are dynamically generated and
    describe all of the columns of a feature.

    Args:

    Returns:

    """

    # model_dump of this feature should be compatible with feature table columns.
    Execution: str
    Feature_Name: str
    feature: ClassVar[Optional["Feature"]] = None

    class Config:
        """ """

        arbitrary_types_allowed = True

    @classmethod
    def feature_columns(cls) -> set[Column]:
        """

        Args:

        Returns:
          :return: set of feature column names.

        """
        return cls.feature.feature_columns

    @classmethod
    def asset_columns(cls) -> set[Column]:
        """

        Args:

        Returns:
          A set of asset column names.

        """
        return cls.feature.asset_columns

    @classmethod
    def term_columns(cls) -> set[Column]:
        """

        Args:

        Returns:
          :return: set of term column names.

        """
        return cls.feature.term_columns

    @classmethod
    def value_columns(cls) -> set[Column]:
        """

        Args:

        Returns:
          A set of value column names.

        """
        return cls.feature.value_columns


class Feature:
    """Wrapper for results of Table.find_associations()"""

    def __init__(self, atable: FindAssociationResult):
        self.feature_table = atable.table
        self.target_table = atable.self_fkey.pk_table
        self.feature_name = atable.table.columns["Feature_Name"].default

        def is_asset(table):
            """

            Args:
              table:

            Returns:

            """
            asset_columns = {"Filename", "URL", "Length", "MD5", "Description"}
            return asset_columns.issubset({c.name for c in table.columns})

        def is_vocabulary(table):
            """

            Args:
              table:

            Returns:

            """
            vocab_columns = {"NAME", "URI", "SYNONYMS", "DESCRIPTION", "ID"}
            return vocab_columns.issubset({c.name.upper() for c in table.columns})

        skip_columns = {
            "RID",
            "RMB",
            "RCB",
            "RCT",
            "RMT",
            "Feature_Name",
            self.target_table.name,
            "Execution",
        }
        self.feature_columns = {
            c for c in self.feature_table.columns if c.name not in skip_columns
        }

        assoc_fkeys = {atable.self_fkey} | atable.other_fkeys

        # Determine the role of each column in the feature outside the FK columns.
        self.asset_columns = {
            fk.foreign_key_columns[0]
            for fk in self.feature_table.foreign_keys
            if fk not in assoc_fkeys and is_asset(fk.pk_table)
        }

        self.term_columns = {
            fk.foreign_key_columns[0]
            for fk in self.feature_table.foreign_keys
            if fk not in assoc_fkeys and is_vocabulary(fk.pk_table)
        }

        self.value_columns = self.feature_columns - (
                self.asset_columns | self.term_columns
        )

    def feature_record_class(self) -> type[FeatureRecord]:
        """ "Create a pydantic model for entries into the specified feature table

        Returns:
            A Feature class that can be used to create instances of the feature.
        """

        def map_type(c: Column) -> UnionType | Type[str] | Type[int] | Type[float]:
            """Map a deriva type into a pydantic model type.

            Args:
                c: column to be mapped
                c: Column:

            Returns:
                A pydantic model type
            """
            if c.name in {c.name for c in self.asset_columns}:
                return str | Path

            match c.type.typename:
                case "text":
                    return str
                case "int2" | "int4" | "int8":
                    return int
                case "float4" | "float8":
                    return float
                case _:
                    return str

        featureclass_name = f"{self.target_table.name}Feature{self.feature_name}"

        # Create feature class. To do this, we must determine the python type for each column and also if the
        # column is optional or not based on its nulliblity.
        feature_columns = {
                              c.name: (
                                  Optional[map_type(c)] if c.nullok else map_type(c),
                                  c.default or None,
                              )
                              for c in self.feature_columns
                          } | {
                              "Feature_Name": (
                                  str,
                                  self.feature_name,
                              ),  # Set default value for Feature_Name
                              self.target_table.name: (str, ...),
                          }
        docstring = f"Class to capture fields in a feature {self.feature_name} on table {self.target_table}. Feature columns include:\n"
        docstring += "\n".join([f"    {c.name}" for c in self.feature_columns])

        model = create_model(
            featureclass_name,
            __base__=FeatureRecord,
            __doc__=docstring,
            **feature_columns,
        )
        model.feature = (
            self  # Set value of class variable within the feature class definition.
        )

        return model

    def __repr__(self) -> str:
        return (
            f"Feature(target_table={self.target_table.name}, feature_name={self.feature_name}, "
            f"feature_table={self.feature_table.name})"
        )


class DerivaML:
    """Base class for ML operations on a Deriva catalog.

    This class is intended to be used as a base class on which more domain specific interfaces are built.

    Attributes:
        host_name: Hostname of the Deriva server.
        catalog_id: Catalog ID. Either and identifier, or a catalog name.
        domain_schema: Schema name for domain specific tables and relationships.
        model: ERMRest model for the catalog
    """

    def __init__(
            self,
            hostname: str,
            catalog_id: str | int,
            domain_schema: str = None,
            project_name: str = None,
            cache_dir: Optional[str] = None,
            working_dir: Optional[str] = None,
            model_version: str = "1",
            ml_schema: str = ML_SCHEMA,
            logging_level=logging.WARNING,
    ):
        """Create and initialize a DerivaML instance.

        This method will connect to a catalog, and initialize local configuration for the ML execution.
        This class is intended to be used as a base class on which domain-specific interfaces are built.

        Args:
            hostname: Hostname of the Deriva server.
            catalog_id: Catalog ID. Either and identifier, or a catalog name.
            domain_schema: Schema name for domain specific tables and relationships.
            cache_dir: Directory path for caching data downloaded from the Deriva server as bdbag.
            working_dir: Directory path for storing data used by or generated by any computations.
            model_version: A string that indicates the version model.  Typically passed in via
        """
        self.host_name = hostname
        self.catalog_id = catalog_id
        self.ml_schema = ml_schema
        self.version = model_version

        self.credential = get_credential(hostname)
        self.catalog = ErmrestCatalog(
            "https",
            hostname,
            catalog_id,
            self.credential,
            session_config=self._get_session_config(),
        )
        self.model = self.catalog.getCatalogModel()
        self.configuration = None
        self.dataset_table = self.model.schemas[self.ml_schema].tables["Dataset"]

        builtin_schemas = ["public", self.ml_schema, "www"]
        self.domain_schema = (
                domain_schema
                or [s for s in self.model.schemas.keys() if s not in builtin_schemas].pop()
        )
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
        default_workdir = self.__class__.__name__ + "_working"
        if working_dir:
            self.working_dir = Path(working_dir).joinpath(
                getpass.getuser(), default_workdir
            )
        else:
            tdir = tdir or TemporaryDirectory()
            self.working_dir = Path(tdir.name) / default_workdir
        self.working_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        if "dirty" in self.version:
            logging.info(
                f"Loading dirty model.  Consider commiting and tagging: {self.version}"
            )

    @staticmethod
    def _get_session_config():
        """ """
        session_config = DEFAULT_SESSION_CONFIG.copy()
        session_config.update(
            {
                # our PUT/POST to ermrest is idempotent
                "allow_retry_on_all_methods": True,
                # do more retries before aborting
                "retry_read": 8,
                "retry_connect": 5,
                # increase delay factor * 2**(n-1) for Nth retry
                "retry_backoff_factor": 5,
            }
        )
        return session_config

    @property
    def pathBuilder(self) -> _CatalogWrapper:
        """Get a new instance of a pathBuilder object."""
        return self.catalog.getPathBuilder()

    def dataset_version(self, dataset_rid: RID) -> tuple[int, ...]:
        """Retrieve the version of the specified dataset.

        Args:
            dataset_rid: return: A tuple with the semantic version of the dataset.
            dataset_rid: RID:

        Returns:
            A tuple with the semantic version of the dataset.
        """
        rid_record = self.resolve_rid(dataset_rid)
        if rid_record.table.name != self.dataset_table.name:
            raise DerivaMLException(
                f"RID: {dataset_rid} does not belong to dataset {self.dataset_table.name}"
            )
        return tuple(map(int, self.retrieve_rid(dataset_rid)["Version"].split(".")))

    def increment_dataset_version(
            self, dataset_rid: RID, component: SemanticVersion
    ) -> tuple[int, ...]:
        """Increment the version of the specified dataset.

        Args:
          dataset_rid: RID to a dataset
          component: Which version of the dataset to increment.
          dataset_rid: RID:
          component: SemanticVersion:

        Returns:
          new vsemantic ersion of the dataset as a 3-tuple

        Raises:
          DerivaMLException: if provided RID is not to a dataset.
        """
        major, minor, patch = self.dataset_version(dataset_rid)
        match component:
            case SemanticVersion.major:
                major += 1
            case SemanticVersion.minor:
                minor += 1
            case SemanticVersion.patch:
                patch += 1
        dataset_path = self.pathBuilder.schemas[self.ml_schema].tables[
            self.dataset_table.name
        ]
        dataset_path.update(
            [{"RID": dataset_rid, "Version": f"{major}.{minor}.{patch}"}]
        )
        return major, minor, patch

    @property
    def domain_path(self):
        """Get a new instance of a pathBuilder object to the domain schema"""

        return self.pathBuilder.schemas[self.domain_schema]

    def _get_table(self, table: str | Table) -> Table:
        """Return the table object corresponding to the given table name.

        If the table name appears in more than one schema, return the first one you find.

        Args:
          table: A ERMRest table object or a string that is the name of the table.
          table: str | Table:

        Returns:
          Table object.
        """
        if isinstance(table, Table):
            return table
        for s in self.model.schemas.values():
            if table in s.tables.keys():
                return s.tables[table]
        raise DerivaMLException(f"The table {table} doesn't exist.")

    def table_path(self, table: str | Table) -> Path:
        """Return a local file path in which to place a CSV to add values to a table on upload.

        Args:
          table: return:
          table: str | Table:

        Returns:
            Path to a CSV file in which to add values to a table on upload.
        """
        return table_path(
            self.working_dir,
            schema=self.domain_schema,
            table=self._get_table(table).name,
        )

    def asset_directory(self, table: str | Table, prefix: str | Path = None) -> Path:
        """Return a local file path in which to place a files for an asset table.  T

        Args:
            table: Location of where to place files.  Defaults to execution_assets_path.
            prefix: Root path to asset directory.

        Returns:
            Path to the directory in which asset files should be placed.
        """
        table = self._get_table(table)
        if not self.is_asset(table):
            raise DerivaMLException(f"The table {table} is not an asset table.")

        prefix = Path(prefix) or self.working_dir
        return asset_dir(prefix, table.schema.name, table.name)

    def download_dir(self, cached: bool = False) -> Path:
        """Location where downloaded files are placed.

        Args:
          cached: bool:  (Default value = False)

        Returns:

        """
        return self.cache_dir if cached else self.working_dir

    def chaise_url(self, table: RID | Table) -> str:
        """Return a Chaise URL to the specified table.

        Args:
            table: Table or RID to be visited
            table: str | Table:

        Returns:
            URL to the table in Chaise format.
        """
        try:
            table = self._get_table(table)
            uri = self.catalog.get_server_uri().replace(
                "ermrest/catalog/", "chaise/recordset/#"
            )
        except DerivaMLException:
            # Perhaps we have a RID....
            uri = self.cite(table)
        return f"{uri}/{urlquote(table.schema.name)}:{urlquote(table.name)}"

    def cite(self, entity: dict | str) -> str:
        """Return a citation URL for the provided entity.

        Args:
            entity: A dict that contains the column values for a specific entity or a RID.

        Returns:
            The URI for the provided entity.

        Raises:
            DerivaMLException: if provided RID does not exist.
        """
        if entity.startswith(f"https://{self.host_name}/id/{self.catalog_id}/"):
            # Already got a citation...
            return entity
        try:
            self.resolve_rid(
                rid := entity if isinstance(entity, str) else entity["RID"]
            )
            return f"https://{self.host_name}/id/{self.catalog_id}/{rid}@{self.catalog.latest_snapshot().snaptime}"
        except KeyError as e:
            raise DerivaMLException(f"Entity {e} does not have RID column")
        except DerivaMLException as _e:
            raise DerivaMLException("Entity RID does not exist")

    def user_list(self) -> list[dict[str, str]]:
        """List of users in the catalog

        Args:

        Returns:
          A list of dictionaries containing user information.

        """
        user_path = self.pathBuilder.public.ERMrest_Client.path
        return [
            {"ID": u["ID"], "Full_Name": u["Full_Name"]}
            for u in user_path.entities().fetch()
        ]

    def resolve_rid(self, rid: RID) -> ResolveRidResult:
        """Return a named tuple with information about the specified RID.

        Args:
            rid: RID of the object of interest

        Returns:
            ResolveRidResult which has information about the specified RID.

        Raises:
          DerivaMLException: if the RID doesn't exist.
        """
        try:
            return self.catalog.resolve_rid(rid, self.model)
        except KeyError as _e:
            raise DerivaMLException(f"Invalid RID {rid}")

    def retrieve_rid(self, rid: RID) -> dict[str, Any]:
        """Return a dictionary that represents the values of the specified RID.

        Args:
            rid: RID of the object of interest

        Returns:
          A dictionary that represents the values of the specified RID.

        Raises:
          DerivaMLException: if the RID doesn't exist.'
        """
        return self.resolve_rid(rid).datapath.entities().fetch()[0]

    def add_page(self, title: str, content: str) -> None:
        """

        Args:
          title: str:
          content: str:

        Returns:

        """
        self.pathBuilder.www.tables[self.domain_schema].insert(
            [{"Title": title, "Content": content}]
        )

    def create_vocabulary(
            self, vocab_name: str, comment: str = "", schema: Optional[str] = None
    ) -> Table:
        """Create a controlled vocabulary table with the given vocab name.

        Args:
            vocab_name: Name of the controlled vocabulary table.
            comment: Description of the vocabulary table. (Default value = '')
            schema: Schema in which to create the controlled vocabulary table.  Defaults to domain_schema.
            vocab_name: str:

        Returns:
            An ERMRest table object for the newly created vocabulary table.
        """
        schema = schema or self.domain_schema
        return self.model.schemas[schema].create_table(
            Table.define_vocabulary(
                vocab_name, f"{self.project_name}:{{RID}}", comment=comment
            )
        )

    def is_vocabulary(self, table_name: str | Table) -> bool:
        """Check if a given table is a controlled vocabulary table.

        Args:
          table_name: A ERMRest table object or the name of the table.
          table_name: str | Table:

        Returns:
          Table object if the table is a controlled vocabulary, False otherwise.

        Raises:
          DerivaMLException: if the table doesn't exist.

        """
        vocab_columns = {"NAME", "URI", "SYNONYMS", "DESCRIPTION", "ID"}
        table = self._get_table(table_name)
        return vocab_columns.issubset({c.name.upper() for c in table.columns})

    def create_asset(
            self, asset_name: str, comment: str = "", schema: str = None
    ) -> Table:
        """Create an asset table with the given asset name.

        Args:
            asset_name: Name of the asset table.
            comment: Description of the asset table. (Default value = '')
            schema: Schema in which to create the asset table.  Defaults to domain_schema.
            asset_name: str:
            schema: str:  (Default value = None)

        Returns:
            Table object for the asset table.
        """
        schema = schema or self.domain_schema
        asset_table = self.model.schemas[schema].create_table(
            Table.define_asset(schema, asset_name, comment=comment)
        )
        return asset_table

    def is_association(
            self, table_name: str | Table, unqualified: bool = True, pure: bool = True
    ) -> bool | set | int:
        """Check the specified table to see if it is an association table.

        Args:
            table_name: param unqualified:
            pure: return: (Default value = True)
            table_name: str | Table:
            unqualified:  (Default value = True)

        Returns:


        """
        table = self._get_table(table_name)
        return table.is_association(unqualified=unqualified, pure=pure)

    def is_asset(self, table_name: str | Table) -> bool:
        """True if the specified table is an asset table.

        Args:
            table_name: str | Table:

        Returns:
            True if the specified table is an asset table, False otherwise.

        """
        asset_columns = {"Filename", "URL", "Length", "MD5", "Description"}
        table = self._get_table(table_name)
        return asset_columns.issubset({c.name for c in table.columns})

    def find_assets(self) -> list[Table]:
        """ """
        return [
            t
            for s in self.model.schemas.values()
            for t in s.tables.values()
            if self.is_asset(t)
        ]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def create_feature(
            self,
            target_table: Table | str,
            feature_name: str,
            terms: list[Table | str] = None,
            assets: list[Table | str] = None,
            metadata: Iterable[ColumnDefinition | Table | Key | str] = None,
            optional: Optional[list[str]] = None,
            comment: str = "",
    ) -> type[FeatureRecord]:
        """Create a new feature that can be associated with a table.

        The feature can associate a controlled vocabulary term, an asset, or any other values with a s
        pecific instance of an object and  execution.

        Args:
            feature_name: Name of the new feature to be defined
            target_table: table name or object on which the feature is to be associated
            terms: List of controlled vocabulary terms that will be part of the feature value
            assets: List of asset table names or objects that will be part of the feature value
            metadata: List of other value types that are associated with the feature
            optional: List of columns that are optional in the feature
            comment: return: A Feature class that can be used to create instances of the feature.

        Returns:
            A Feature class that can be used to create instances of the feature.

        Raises:
            DerivaException: If the feature cannot be created.
        """

        terms = terms or []
        assets = assets or []
        metadata = metadata or []
        optional = optional or []

        def normalize_metadata(m: Key | Table | ColumnDefinition | str):
            """

            Args:
              m: Key | Table | ColumnDefinition | str:

            Returns:

            """
            if isinstance(m, str):
                return self._get_table(m)
            elif isinstance(m, ColumnDefinition):
                return m.model_dump()
            else:
                return m

        # Make sure that the provided assets or terms are actually assets or terms.
        if not all(map(self.is_asset, assets)):
            raise DerivaMLException(f"Invalid create_feature asset table.")
        if not all(map(self.is_vocabulary, terms)):
            raise DerivaMLException(f"Invalid create_feature asset table.")

        # Get references to the necessary tables and make sure that the
        # provided feature name exists.
        target_table = self._get_table(target_table)
        execution = self.model.schemas[self.ml_schema].tables["Execution"]
        feature_name_table = self.model.schemas[self.ml_schema].tables["Feature_Name"]
        feature_name_term = self.add_term(
            "Feature_Name", feature_name, description=comment
        )
        atable_name = f"Execution_{target_table.name}_{feature_name_term.name}"

        # Now create the association table that implements the feature.
        atable = self.model.schemas[self.domain_schema].create_table(
            target_table.define_association(
                table_name=atable_name,
                associates=[execution, target_table, feature_name_table],
                metadata=[
                    normalize_metadata(m) for m in chain(assets, terms, metadata)
                ],
                comment=comment,
            )
        )
        # Now set optional terms.
        for c in optional:
            atable.columns[c].alter(nullok=True)
        atable.columns["Feature_Name"].alter(default=feature_name_term.name)
        return self.feature_record_class(target_table, feature_name)

    def feature_record_class(
            self, table: str | Table, feature_name: str
    ) -> type[FeatureRecord]:
        """ "Create a pydantic model for entries into the specified feature table.

        For information on how to
        See the pydantic documentation for more details about the pydantic model.

        Args:
            table: table name or object on which the feature is to be associated
            feature_name: name of the feature to be created
            table: str | Table:
            feature_name: str:

        Returns:
            A Feature class that can be used to create instances of the feature.
        """
        return self.lookup_feature(table, feature_name).feature_record_class()

    def drop_feature(self, table: Table | str, feature_name: str) -> bool:
        """

        Args:
          table: Table | str:
          feature_name: str:

        Returns:
        """
        table = self._get_table(table)
        try:
            feature = next(
                f for f in self.find_features(table) if f.feature_name == feature_name
            )
            feature.feature_table.drop()
            return True
        except StopIteration:
            return False

    def lookup_feature(self, table: str | Table, feature_name: str) -> Feature:
        """Lookup the named feature associated with the provided table.

        Args:
            table: param feature_name:
            table: str | Table:
            feature_name: str:

        Returns:
            A Feature class that represents the requested feature.

        Raises:
          DerivaMLException: If the feature cannot be found.
        """
        table = self._get_table(table)
        try:
            return [
                f for f in self.find_features(table) if f.feature_name == feature_name
            ][0]
        except IndexError:
            raise DerivaMLException(
                f"Feature {table.name}:{feature_name} doesn't exist."
            )

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def find_features(self, table: Table | str) -> Iterable[Feature]:
        """List the names of the features in the specified table.

        Args:
            table: The table to find features for.
            table: Table | str:

        Returns:
            An iterable of FeatureResult instances that describe the current features in the table.
        """
        table = self._get_table(table)

        def is_feature(a: FindAssociationResult) -> bool:
            """

            Args:
              a: FindAssociationResult:

            Returns:

            """
            # return {'Feature_Name', 'Execution'}.issubset({c.name for c in a.table.columns})
            return {
                "Feature_Name",
                "Execution",
                a.self_fkey.foreign_key_columns[0].name,
            }.issubset({c.name for c in a.table.columns})

        return [
            Feature(a)
            for a in table.find_associations(min_arity=3, max_arity=3, pure=False)
            if is_feature(a)
        ]

    @validate_call
    def add_features(self, features: Iterable[FeatureRecord]) -> int:
        """Add a set of new feature values to the catalog.

        Args:
          features: Iterable[FeatureRecord]:

        Returns:
            Number of attributes added
        """
        features = list(features)
        feature_table = features[0].feature.feature_table
        feature_path = self.pathBuilder.schemas[feature_table.schema.name].tables[
            feature_table.name
        ]
        entries = feature_path.insert(f.model_dump() for f in features)
        return len(entries)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def list_feature_values(self, table: Table | str, feature_name: str) -> _ResultSet:
        """Return a datapath resultset containing all values of a feature associated with a table.

        Args:
            table: param feature_name:
            table: Table | str:
            feature_name: str:

        Returns:

        """
        table = self._get_table(table)
        feature = self.lookup_feature(table, feature_name)
        pb = self.catalog.getPathBuilder()
        return (
            pb.schemas[feature.feature_table.schema.name]
            .tables[feature.feature_table.name]
            .entities()
            .fetch()
        )

    @validate_call
    def create_dataset(
            self,
            ds_type: str | list[str],
            description: str,
            execution_rid: Optional[RID] = None,
            version: tuple[int, int, int] = (1, 0, 0),
    ) -> RID:
        """Create a new dataset from the specified list of RIDs.

        Args:
            ds_type: One or more dataset types.  Must be a term from the DatasetType controlled vocabulary.
            description: Description of the dataset.
            execution_rid: Execution under which the dataset will be created.
            version: Version of the dataset.
            ds_type: str | list[str]:
            description: str:
            execution_rid: Optional[RID]:  (Default value = None)
            version: tuple[int: int: int]

        Returns:
            New dataset RID.

        """
        # Create the entry for the new dataset and get its RID.
        ds_types = [ds_type] if isinstance(ds_type, str) else ds_type

        pb = self.pathBuilder
        for ds_type in ds_types:
            if not self.lookup_term(MLVocab.dataset_type, ds_type):
                raise DerivaMLException(f"Dataset type must be a vocabulary term.")
        dataset_table_path = pb.schemas[self.dataset_table.schema.name].tables[
            self.dataset_table.name
        ]
        dataset = dataset_table_path.insert(
            [
                {
                    "Description": description,
                    MLVocab.dataset_type: ds_type,
                    "Version": f"{version[0]}.{version[1]}.{version[2]}",
                }
            ]
        )[0]["RID"]

        # Get the name of the association table between dataset and dataset_type.
        atable = next(
            self.model.schemas[self.ml_schema]
            .tables[MLVocab.dataset_type]
            .find_associations()
        ).name
        pb.schemas[self.ml_schema].tables[atable].insert(
            [
                {MLVocab.dataset_type: ds_type, "Dataset": dataset}
                for ds_type in ds_types
            ]
        )
        if execution_rid is not None:
            pb.schemas[self.ml_schema].Dataset_Execution.insert(
                [{"Dataset": dataset, "Execution": execution_rid}]
            )
        return dataset

    def find_datasets(self) -> Iterable[dict[str, Any]]:
        """Returns a list of currently available datasets.

        Returns:
             list of currently available datasets.
        """
        # Get datapath to all the tables we will need: Dataset, DatasetType and the association table.
        pb = self.pathBuilder
        dataset_path = pb.schemas[self.dataset_table.schema.name].tables[
            self.dataset_table.name
        ]
        atable = next(
            self.model.schemas[self.ml_schema]
            .tables[MLVocab.dataset_type]
            .find_associations()
        ).name
        ml_path = pb.schemas[self.ml_schema]
        atable_path = ml_path.tables[atable]

        # Get a list of all the dataset_type values associated with this dataset.
        datasets = []
        for dataset in dataset_path.entities().fetch():
            ds_types = (
                atable_path.filter(atable_path.Dataset == dataset["RID"])
                .attributes(atable_path.Dataset_Type)
                .fetch()
            )
            datasets.append(
                dataset
                | {MLVocab.dataset_type: [ds[MLVocab.dataset_type] for ds in ds_types]}
            )
        return datasets

    @validate_call
    def delete_dataset(self, dataset_rid: RID, recurse: bool = False) -> None:
        """Delete a dataset from the catalog.

        Args:
            dataset_rid: RID of the dataset to delete.
            recurse: If True, delete the dataset along with any nested datasets. (Default value = False)
            dataset_rid: RID:
        """
        # Get association table entries for this dataset
        # Delete association table entries
        pb = self.pathBuilder
        for assoc_table in self.dataset_table.find_associations(self.dataset_table):
            other_fkey = assoc_table.other_fkeys.pop()
            self_fkey = assoc_table.self_fkey
            target_table = other_fkey.pk_table
            member_table = assoc_table.table

            schema_path = pb.schemas[member_table.schema.name]
            tpath = schema_path.tables[assoc_table.name]
            dataset_column_path = tpath.columns[self_fkey.columns[0].name]
            dataset_entries = tpath.filter(dataset_column_path == dataset_rid)
            if recurse and target_table == self.dataset_table:
                # Nested table
                for dataset in dataset_entries:
                    self.delete_dataset(dataset["RID"], recurse)
            try:
                dataset_entries.delete()
            except DataPathException:
                pass

        # Delete dataset.
        dataset_path = pb.schemas[self.dataset_table.schema.name].tables[
            self.dataset_table.name
        ]
        dataset_path.filter(dataset_path.columns["RID"] == dataset_rid).delete()

    def list_dataset_element_types(self) -> Iterable[Table]:
        """List the types of entities that can be added to a dataset.

        Returns:
          :return: An iterable of Table objects that can be included as an element of a dataset.
        """

        def domain_table(table: Table) -> bool:
            return (
                    table.schema.name == self.domain_schema
                    or table.name == self.dataset_table.name
            )

        return [
            t
            for a in self.dataset_table.find_associations()
            if domain_table(t := a.other_fkeys.pop().pk_table)
        ]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def add_dataset_element_type(self, element: str | Table) -> Table:
        """A dataset is a heterogeneous collection of objects, each of which comes from a different table. This
        routine makes it possible to add objects from the specified table to a dataset.

        Args:
            element: Name or the table or table object that is to be added to the dataset.
            element: str | Table:

        Returns:
            The table object that was added to the dataset.
        """
        # Add table to map
        element_table = self._get_table(element)
        table = self.model.schemas[self.domain_schema].create_table(
            Table.define_association([self.dataset_table, element_table])
        )

        # self.model = self.catalog.getCatalogModel()
        self.dataset_table.annotations.update(generate_dataset_annotations(self.model))
        self.model.apply()
        return table

    @validate_call
    def list_dataset_parents(self, dataset_rid: RID) -> list[RID]:
        """Given a dataset RID, return a list of RIDs of the parent datasets.

        Args:
            dataset_rid: return: RID of the parent dataset.
            dataset_rid: RID:

        Returns:
            RID of the parent dataset.
        """
        rid_record = self.resolve_rid(dataset_rid)
        if rid_record.table.name != self.dataset_table.name:
            raise DerivaMLException(
                f"RID: {dataset_rid} does not belong to dataset {self.dataset_table.name}"
            )
        # Get association table for nested datasets
        atable_path = self.pathBuilder.schemas[self.ml_schema].Dataset_Dataset
        return [
            p["Dataset"]
            for p in atable_path.filter(atable_path.Nested_Dataset == dataset_rid)
            .entities()
            .fetch()
        ]

    @validate_call
    def list_dataset_children(self, dataset_rid: RID) -> list[RID]:
        """Given a dataset RID, return a list of RIDs of any nested datasets.

        Args:
            dataset_rid: A dataset RID.

        Returns:
          list of RIDs of nested datasets.

        """
        return [d["RID"] for d in self.list_dataset_members(dataset_rid)["Dataset"]]

    @validate_call
    def list_dataset_members(
            self, dataset_rid: RID, recurse: bool = False
    ) -> dict[str, list[dict[str, Any]]]:
        """Return a list of entities associated with a specific dataset.

        Args:
            dataset_rid: param recurse: If this is a nested dataset, list the members of the contained datasets
            dataset_rid: RID:
            recurse:  (Default value = False)

        Returns:
            Dictionary of entities associated with a specific dataset.  Key is the table from which the elements
            were taken.
        """

        try:
            if self.resolve_rid(dataset_rid).table != self.dataset_table:
                raise DerivaMLException(f"RID is not for a dataset: {dataset_rid}")
        except DerivaMLException:
            raise DerivaMLException(f"Invalid RID: {dataset_rid}")

        # Look at each of the element types that might be in the dataset and get the list of rid for them from
        # the appropriate association table.
        members = defaultdict(list)
        pb = self.pathBuilder
        for assoc_table in self.dataset_table.find_associations():
            other_fkey = assoc_table.other_fkeys.pop()
            self_fkey = assoc_table.self_fkey
            target_table = other_fkey.pk_table
            member_table = assoc_table.table

            if (
                    target_table.schema.name != self.domain_schema
                    and target_table != self.dataset_table
            ):
                # Look at domain tables and nested datasets.
                continue
            if target_table == self.dataset_table:
                # find_assoc gives us the keys in the wrong position, so swap.
                self_fkey, other_fkey = other_fkey, self_fkey

            target_path = pb.schemas[target_table.schema.name].tables[target_table.name]
            member_path = pb.schemas[member_table.schema.name].tables[member_table.name]
            # Get the names of the columns that we are going to need for linking
            member_link = tuple(
                c.name for c in next(iter(other_fkey.column_map.items()))
            )
            path = pb.schemas[member_table.schema.name].tables[member_table.name].path
            path.filter(member_path.Dataset == dataset_rid)
            path.link(
                target_path,
                on=(
                        member_path.columns[member_link[0]]
                        == target_path.columns[member_link[1]]
                ),
            )
            target_entities = path.entities().fetch()
            members[target_table.name].extend(target_entities)
            if recurse and target_table == self.dataset_table:
                # Get the members for all the nested datasets and add to the member list.
                nested_datasets = [d["RID"] for d in target_entities]
                for ds in nested_datasets:
                    for k, v in self.list_dataset_members(ds, recurse=False).items():
                        members[k].extend(v)
        return members

    @validate_call
    def add_dataset_members(
            self, dataset_rid: Optional[RID], members: list[RID], validate: bool = True
    ) -> None:
        """Add additional elements to an existing dataset.

        Args:
            dataset_rid: RID of dataset to extend or None if new dataset is to be created.
            members: List of RIDs of members to add to the  dataset.
            validate: Check rid_list to make sure elements are not already in the dataset.
            dataset_rid: Optional[RID]:
            members: list[RID]:
            validate: bool:  (Default value = True)
        """

        members = set(members)

        def check_dataset_cycle(member_rid, path=None):
            """

            Args:
              member_rid:
              path:  (Default value = None)

            Returns:

            """
            path = path or set(dataset_rid)
            return member_rid in path

        if validate:
            existing_rids = set(
                m["RID"]
                for ms in self.list_dataset_members(dataset_rid).values()
                for m in ms
            )
            if overlap := set(existing_rids).intersection(members):
                raise DerivaMLException(
                    f"Attempting to add existing member to dataset {dataset_rid}: {overlap}"
                )

        # Now go through every rid to be added to the data set and sort them based on what association table entries
        # need to be made.
        dataset_elements = {}
        association_map = {
            a.other_fkeys.pop().pk_table.name: a.table.name
            for a in self.dataset_table.find_associations()
        }
        # Get a list of all the types of objects that can be linked to a dataset.
        for m in members:
            rid_info = self.resolve_rid(m)
            if rid_info.table.name not in association_map:
                raise DerivaMLException(
                    f"RID table: {rid_info.table.name} not part of dataset"
                )
            if rid_info.table == self.dataset_table and check_dataset_cycle(
                    rid_info.rid
            ):
                raise DerivaMLException("Creating cycle of datasets is not allowed")
            dataset_elements.setdefault(rid_info.table.name, []).append(rid_info.rid)
        # Now make the entries into the association tables.
        pb = self.pathBuilder
        for table, elements in dataset_elements.items():
            schema_path = pb.schemas[
                self.ml_schema if table == "Dataset" else self.domain_schema
            ]
            fk_column = "Nested_Dataset" if table == "Dataset" else table

            if len(elements):
                # Find out the name of the column in the association table.
                schema_path.tables[association_map[table]].insert(
                    [{"Dataset": dataset_rid, fk_column: e} for e in elements]
                )

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def add_term(
            self,
            table: str | Table,
            term_name: str,
            description: str,
            synonyms: Optional[list[str]] = None,
            exists_ok: bool = True,
    ) -> VocabularyTerm:
        """Creates a new control vocabulary term in the control vocabulary table.

        Args:

        Args:
            table: The name of the control vocabulary table.
            term_name: The name of the new control vocabulary.
            description: The description of the new control vocabulary.
            synonyms: Optional list of synonyms for the new control vocabulary. Defaults to an empty list.
            exists_ok: Optional flag indicating whether to allow creation if the control vocabulary name
                already exists. Defaults to True.

        Returns:
          The RID of the newly created control vocabulary.

        Raises:
          DerivaException: If the control vocabulary name already exists and exist_ok is False.
        """
        synonyms = synonyms or []
        table = self._get_table(table)
        pb = self.catalog.getPathBuilder()
        if not (self.is_vocabulary(table)):
            raise DerivaMLException(f"The table {table} is not a controlled vocabulary")

        schema_name = table.schema.name
        table_name = table.name
        try:
            term_id = VocabularyTerm.model_validate(
                pb.schemas[schema_name]
                .tables[table_name]
                .insert(
                    [
                        {
                            "Name": term_name,
                            "Description": description,
                            "Synonyms": synonyms,
                        }
                    ],
                    defaults={"ID", "URI"},
                )[0]
            )
        except DataPathException:
            term_id = self.lookup_term(table, term_name)
            if not exists_ok:
                raise DerivaMLException(f"{term_name} already exists")
            # Check vocabulary
        return term_id

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def lookup_term(self, table: str | Table, term_name: str) -> VocabularyTerm:
        """Given a term name, return the vocabulary record.  Can provide either the term name
         or a synonym for the term.  Generate an exception if the term is not in the vocabulary.

        Args:
            table: The name of the controlled vocabulary table or a ERMrest table object..
            term_name: The name of the term to look up.

        Returns:
          The entry the associated term or synonym.

        Raises:
          DerivaException: If the schema or vocabulary table doesn't exist, or if the term is not
            found in the vocabulary.
        """
        vocab_table = self._get_table(table)
        if not self.is_vocabulary(vocab_table):
            raise DerivaMLException(f"The table {table} is not a controlled vocabulary")
        schema_name, table_name = vocab_table.schema.name, vocab_table.name
        schema_path = self.catalog.getPathBuilder().schemas[schema_name]
        for term in schema_path.tables[table_name].entities():
            if term_name == term["Name"] or (
                    term["Synonyms"] and term_name in term["Synonyms"]
            ):
                return VocabularyTerm.model_validate(term)
        raise DerivaMLException(f"Term {term_name} is not in vocabulary {table_name}")

    def find_vocabularies(self) -> Iterable[Table]:
        """Return a list of all the controlled vocabulary tables in the domain schema."""
        return [
            t
            for s in self.model.schemas.values()
            for t in s.tables.values()
            if self.is_vocabulary(t)
        ]

    def list_vocabulary_terms(self, table: str | Table) -> list[VocabularyTerm]:
        """Return an list of terms that are in a vocabulary table.

        Args:
            table: The name of the controlled vocabulary table or a ERMrest table object.
            table: str | Table:

        Returns:
            The list of terms that are in a vocabulary table.

        Raises:
            DerivaMLException: If the schema or vocabulary table doesn't exist, or if the table is not
                a controlled vocabulary.
        """
        pb = self.catalog.getPathBuilder()
        table = self._get_table(table)
        if not (self.is_vocabulary(table)):
            raise DerivaMLException(f"The table {table} is not a controlled vocabulary")

        return [
            VocabularyTerm(**v)
            for v in pb.schemas[table.schema.name].tables[table.name].entities().fetch()
        ]

    @validate_call
    def download_dataset_bag(
            self,
            bag: RID | str,
            materialize: bool = True,
            execution_rid: Optional[RID] = None,
    ) -> tuple[Path, RID]:
        """Given a RID to a dataset, or a MINID to an existing bag, download the bag file, extract it and validate
        that all the metadata is correct

        Args:
            bag: The RID of a dataset or a minid to an existing bag.
            materialize: Materalize the bag, rather than just downloading it.
            execution_rid: RID of execution object requesting the download.  Used to update status.

        Returns:
            the location of the unpacked and validated dataset bag and the RID of the bag
        """
        return (
            self._materialize_dataset_bag(bag, execution_rid)
            if materialize
            else self._download_dataset_bag(bag)
        )

    def _download_dataset_bag(self, dataset_rid: RID | str) -> tuple[Path, RID]:
        """Given a RID to a dataset, or a MINID to an existing bag, download the bag file, extract it and validate
        that all the metadata is correct

        Args:
            dataset_rid: The RID of a dataset or a minid to an existing bag.
             dataset_rid: RID | str:

        Returns:
            the location of the unpacked and validated dataset bag and the RID of the bag
        """
        if not any([dataset_rid == ds["RID"] for ds in self.find_datasets()]):
            raise DerivaMLException(f"RID {dataset_rid} is not a dataset")

        with TemporaryDirectory() as tmp_dir:
            if dataset_rid.startswith("minid"):
                # If provided a MINID, use the MINID metadata to get the checksum and download the bag.
                r = requests.get(
                    f"https://identifiers.org/{dataset_rid}",
                    headers={"accept": "application/json"},
                )
                metadata = r.json()["metadata"]
                dataset_rid = metadata["Dataset_RID"].split("@")[0]
                checksum_value = ""
                for checksum in r.json().get("checksums", []):
                    if checksum.get("function") == "sha256":
                        checksum_value = checksum.get("value")
                        break
                archive_path = fetch_single_file(dataset_rid, tmp_dir)
            else:
                # We are given the RID to a dataset, so we are going to have to export as a bag and place into
                # local file system.  The first step is to generate a downloadspec to create the bag, put the sped
                # into a local file and then use the downloader to create and download the desired bdbag.
                spec_file = f"{tmp_dir}/download_spec.json"
                with open(spec_file, "w", encoding="utf-8") as ds:
                    json.dump(Dataset(self.model).generate_dataset_download_spec(), ds)
                downloader = GenericDownloader(
                    server={
                        "catalog_id": self.catalog_id,
                        "protocol": "https",
                        "host": self.host_name,
                    },
                    config_file=spec_file,
                    output_dir=tmp_dir,
                    envars={"Dataset_RID": dataset_rid},
                )
                result = downloader.download()
                archive_path = list(result.values())[0]["local_path"]
                checksum_value = compute_file_hashes(archive_path, hashes=["sha256"])[
                    "sha256"
                ][0]

            # Check to see if we have an existing idempotent materialization of the desired bag. If so, then just reuse
            # it.  If not, then we need to extract the contents of the archive into our cache directory.
            bag_dir = self.cache_dir / f"{dataset_rid}_{checksum_value}"
            if bag_dir.exists():
                bag_path = (bag_dir / f"Dataset_{dataset_rid}").as_posix()
            else:
                bag_dir.mkdir(parents=True, exist_ok=True)
                bag_path = bdb.extract_bag(archive_path, bag_dir)
            bdb.validate_bag_structure(bag_path)
            return Path(bag_path), dataset_rid

    def _materialize_dataset_bag(
            self, bag: str | RID, execution_rid: Optional[RID] = None
    ) -> tuple[Path, RID]:
        """Materialize a dataset bag into a local directory

        Args:
            bag: A MINID to an existing bag or a RID of the dataset that should be downloaded.
            execution_rid: RID of the execution for which this bag should be materialized. Used to update status.
            bag: str | RID:
            execution_rid: Optional[RID]:  (Default value = None)

        Returns:

        """

        def fetch_progress_callback(current, total):
            """

            Args:
              current:
              total:

            Returns:

            """
            msg = f"Materializing bag: {current} of {total} file(s) downloaded."
            if execution_rid:
                self._update_status(Status.running, msg, execution_rid)
            logging.info(msg)
            return True

        def validation_progress_callback(current, total):
            """

            Args:
              current:
              total:

            Returns:

            """
            msg = f"Validating bag: {current} of {total} file(s) validated."
            if execution_rid:
                self._update_status(Status.running, msg, execution_rid)
            logging.info(msg)
            return True

        if execution_rid and self.resolve_rid(execution_rid).table.name != "Execution":
            raise DerivaMLException(f"RID {execution_rid} is not an execution")

        # request metadata
        bag_path, dataset_rid = self._download_dataset_bag(bag)
        bag_dir = bag_path.parent
        validated_check = bag_dir / "validated_check.txt"

        # If this bag has already been validated, our work is done.  Otherwise, materialize the bag.
        if not validated_check.exists():
            bdb.materialize(
                bag_path.as_posix(),
                fetch_callback=fetch_progress_callback,
                validation_callback=validation_progress_callback,
            )
            validated_check.touch()
        return Path(bag_path), dataset_rid

    def download_asset(self, asset_url: str, dest_filename: str) -> Path:
        """Download an asset from a URL and place it in a local directory.

        Args:
            asset_url: URL of the asset.
            dest_filename: Destination filename.

        Returns:
            A  Path object to the downloaded asset.
        """
        hs = HatracStore("https", self.host_name, self.credential)
        hs.get_obj(path=asset_url, destfilename=dest_filename)
        return Path(dest_filename)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def upload_asset(self, file: Path | str, table: Table | str, **kwargs: Any) -> dict:
        """Upload the specified file into Hatrac and update the associated asset table.

        Args:
            file: path to the file to upload.
            table: Name of the asset table
            kwargs: Keyword arguments for values of additional columns to be added to the asset table.

        Returns:

        """
        table = self._get_table(table)
        if not self.is_asset(table):
            raise DerivaMLException(f"Table {table} is not an asset table.")

        credential = self.model.catalog.deriva_server.credentials
        file_path = Path(file)
        file_name = file_path.name
        file_size = file_path.stat().st_size
        # Get everything up to the filename  part of the
        hatrac_path = f"/hatrac/{table.name}/"
        hs = HatracStore("https", self.host_name, credential)
        md5_hashes = hash_utils.compute_file_hashes(file, ["md5"])["md5"]
        sanitized_filename = urlquote(
            re.sub("[^a-zA-Z0-9_.-]", "_", md5_hashes[0] + "." + file_name)
        )
        hatrac_path = f"{hatrac_path}{sanitized_filename}"

        try:
            # Upload the file to hatrac.
            hatrac_uri = hs.put_obj(
                hatrac_path,
                file,
                md5=md5_hashes[1],
                content_type=mime_utils.guess_content_type(file),
                content_disposition="filename*=UTF-8''" + file_name,
            )
        except Exception as e:
            raise e
        try:
            # Now update the asset table.
            ipath = self.pathBuilder.schemas[table.schema.name].tables[table.name]
            return list(
                ipath.insert(
                    [
                        {
                            "URL": hatrac_uri,
                            "Filename": file_name,
                            "Length": file_size,
                            "MD5": md5_hashes[0],
                        }
                        | kwargs
                    ]
                )
            )[0]
        except Exception as e:
            raise e

    def upload_assets(self, assets_dir: str | Path) -> dict[str, FileUploadState]:
        """Upload assets from a directory. This routine assumes that the current upload specification includes a
        configuration for the specified directory.  Every asset in the specified directory is uploaded

        Args:
            assets_dir: Directory containing the assets to upload.

        Returns:
            Results of the upload operation.

        Raises:
            DerivaMLException: If there is an issue uploading the assets.
        """
        with TemporaryDirectory() as temp_dir:
            spec_file = f"{temp_dir}/config.json"
            with open(spec_file, "w+") as cfile:
                json.dump(bulk_upload_configuration, cfile)
            uploader = GenericUploader(
                server={
                    "host": self.host_name,
                    "protocol": "https",
                    "catalog_id": self.catalog_id,
                },
                config_file=spec_file,
            )
            uploader.getUpdatedConfig()
            uploader.scanDirectory(assets_dir)
            results = {
                path: FileUploadState(
                    state=UploadState(result["State"]),
                    status=result["Status"],
                    result=result["Result"],
                )
                for path, result in uploader.uploadFiles().items()
            }
            uploader.cleanup()
        return results

    def _update_status(
            self, new_status: Status, status_detail: str, execution_rid: RID
    ):
        """Update the status of an execution in the catalog.

        Args:
            new_status: New status.
            status_detail: Details of the status.
            execution_rid: Resource Identifier (RID) of the execution.
            new_status: Status:
            status_detail: str:
             execution_rid: RID:

        Returns:

        """
        self.status = new_status.value
        self.pathBuilder.schemas[self.ml_schema].Execution.update(
            [
                {
                    "RID": execution_rid,
                    "Status": self.status,
                    "Status_Detail": status_detail,
                }
            ]
        )

    def upload_execution_configuration(self, config: ExecutionConfiguration) -> RID:
        """Upload execution configuration to Execution_Metadata table with Execution Metadata Type = Execution_Config.

        Args:
            config: A execution configuration.

        Returns:
            RID of the newly created configuration file.

        Raises:
             DerivaMLException: If there is an issue uploading the configuration.
        """
        try:
            fp = NamedTemporaryFile(
                "w+", prefix="exec_config", suffix=".json", delete=False
            )
            json.dump(config.model_dump_json(), fp)
            fp.close()
            configuration_rid = self._upload_execution_configuration_file(
                fp.name, description=config.description
            )
            os.remove(fp.name)
        except Exception as _e:
            raise DerivaMLException(f"Error in execution configuration upload")
        return configuration_rid

    def download_execution_configuration(
            self, configuration_rid: RID
    ) -> ExecutionConfiguration:
        """Create an ExecutionConfiguration object from a catalog RID that points to a JSON representation of that
        configuration in hatrac

        Args:
            configuration_rid: RID that should be to an asset table that refers to an execution configuration

        Returns:
            A ExecutionConfiguration object for configured by the parameters in the configuration file.
        """
        configuration = self.retrieve_rid(configuration_rid)
        with NamedTemporaryFile("w+", delete=False, suffix=".json") as dest_file:
            hs = HatracStore("https", self.host_name, self.credential)
            hs.get_obj(path=configuration["URL"], destfilename=dest_file.name)
            return ExecutionConfiguration.load_configuration(dest_file.name)

    def _upload_execution_configuration_file(
            self, config_file: str, description: str
    ) -> RID:
        """

        Args:
            config_file: str:
            description: str:

        Returns:

        """
        file_path = Path(config_file)
        file_name = file_path.name
        file_size = file_path.stat().st_size
        try:
            hs = HatracStore("https", self.host_name, self.credential)
            md5 = hash_utils.compute_file_hashes(config_file, ["md5"])["md5"][1]
            sanitized_filename = urlquote(
                re.sub("[^a-zA-Z0-9_.-]", "_", md5 + "." + file_name)
            )
            hatrac_path = f"/hatrac/execution_metadata/{sanitized_filename}"
            hatrac_uri = hs.put_obj(
                hatrac_path,
                config_file,
                md5=md5,
                content_type=mime_utils.guess_content_type(config_file),
                content_disposition="filename*=UTF-8''" + file_name,
            )
        except Exception as e:
            error = format_exception(e)
            raise DerivaMLException(
                f"Failed to upload execution configuration file {config_file} to object store. Error: {error}"
            )
        try:
            ml_schema_path = self.pathBuilder.schemas[self.ml_schema]
            return list(
                ml_schema_path.tables["Execution_Metadata"].insert(
                    [
                        {
                            "URL": hatrac_uri,
                            "Filename": file_name,
                            "Length": file_size,
                            "MD5": md5,
                            "Description": description,
                            "Execution_Metadata_Type": ExecMetadataVocab.execution_config,
                        }
                    ]
                )
            )[0]["RID"]
        except Exception as e:
            error = format_exception(e)
            raise DerivaMLException(
                f"Failed to update Execution_Asset table with configuration file metadata. Error: {error}"
            )

    # @validate_call
    def create_execution(self, configuration: ExecutionConfiguration) -> "Execution":
        """Create an execution object

        Args:
            configuration: ExecutionConfiguration:

        Returns:
            An execution object.
        """
        from .execution import Execution

        return Execution(configuration, self)
