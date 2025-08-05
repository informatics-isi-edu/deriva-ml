"""Core module for the Deriva ML project.

This module implements the DerivaML class, which is the primary interface to Deriva-based catalogs. It provides
functionality for managing features, vocabularies, and other ML-related operations.

The module requires a catalog that implements a 'deriva-ml' schema with specific tables and relationships.

Typical usage example:
    >>> ml = DerivaML('deriva.example.org', 'my_catalog')
    >>> ml.create_feature('my_table', 'new_feature')
    >>> ml.add_term('vocabulary_table', 'new_term', description='Description of term')
"""

from __future__ import annotations  # noqa: I001

# Standard library imports
from collections import defaultdict
import getpass
import logging
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, List, cast, TYPE_CHECKING, Any
from urllib.parse import urlsplit


# Third-party imports
import requests
from pydantic import ConfigDict, validate_call

# Deriva imports
from deriva.core import (
    DEFAULT_SESSION_CONFIG,
    format_exception,
    get_credential,
    urlquote,
)

import deriva.core.datapath as datapath
from deriva.core.datapath import DataPathException, _SchemaWrapper as SchemaWrapper
from deriva.core.deriva_server import DerivaServer
from deriva.core.ermrest_catalog import ResolveRidResult
from deriva.core.ermrest_model import Key, Table
from deriva.core.utils.globus_auth_utils import GlobusNativeLogin

from deriva_ml.core.exceptions import DerivaMLInvalidTerm
from deriva_ml.core.definitions import (
    ML_SCHEMA,
    RID,
    ColumnDefinition,
    FileSpec,
    MLVocab,
    MLTable,
    Status,
    TableDefinition,
    VocabularyTerm,
)
from deriva_ml.core.exceptions import DerivaMLTableTypeError, DerivaMLException
from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.dataset.dataset import Dataset
from deriva_ml.dataset.dataset_bag import DatasetBag
from deriva_ml.dataset.upload import asset_file_path, execution_rids, table_path

# Local imports
from deriva_ml.execution.execution_configuration import ExecutionConfiguration
from deriva_ml.execution.workflow import Workflow
from deriva_ml.feature import Feature, FeatureRecord
from deriva_ml.model.catalog import DerivaModel
from deriva_ml.schema.annotations import asset_annotation

# Optional debug imports
try:
    from icecream import ic

    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

if TYPE_CHECKING:
    from deriva_ml.execution.execution import Execution

# Stop pycharm from complaining about undefined references.
ml: DerivaML


class DerivaML(Dataset):
    """Core class for machine learning operations on a Deriva catalog.

    This class provides core functionality for managing ML workflows, features, and datasets in a Deriva catalog.
    It handles data versioning, feature management, vocabulary control, and execution tracking.

    Attributes:
        host_name (str): Hostname of the Deriva server (e.g., 'deriva.example.org').
        catalog_id (Union[str, int]): Catalog identifier or name.
        domain_schema (str): Schema name for domain-specific tables and relationships.
        model (DerivaModel): ERMRest model for the catalog.
        working_dir (Path): Directory for storing computation data and results.
        cache_dir (Path): Directory for caching downloaded datasets.
        ml_schema (str): Schema name for ML-specific tables (default: 'deriva_ml').
        configuration (ExecutionConfiguration): Current execution configuration.
        project_name (str): Name of the current project.
        start_time (datetime): Timestamp when this instance was created.
        status (str): Current status of operations.

    Example:
        >>> ml = DerivaML('deriva.example.org', 'my_catalog')
        >>> ml.create_feature('my_table', 'new_feature')
        >>> ml.add_term('vocabulary_table', 'new_term', description='Description of term')
    """

    def __init__(
        self,
        hostname: str,
        catalog_id: str | int,
        domain_schema: str | None = None,
        project_name: str | None = None,
        cache_dir: str | Path | None = None,
        working_dir: str | Path | None = None,
        ml_schema: str = ML_SCHEMA,
        logging_level=logging.INFO,
        credential=None,
        use_minid: bool = True,
    ):
        """Initializes a DerivaML instance.

        This method will connect to a catalog and initialize local configuration for the ML execution.
        This class is intended to be used as a base class on which domain-specific interfaces are built.

        Args:
            hostname: Hostname of the Deriva server.
            catalog_id: Catalog ID. Either an identifier or a catalog name.
            domain_schema: Schema name for domain-specific tables and relationships. Defaults to the name of the
                schema that is not one of the standard schemas.  If there is more than one user-defined schema, then
                this argument must be provided a value.
            ml_schema: Schema name for ML schema. Used if you have a non-standard configuration of deriva-ml.
            project_name: Project name. Defaults to name of domain schema.
            cache_dir: Directory path for caching data downloaded from the Deriva server as bdbag.
            working_dir: Directory path for storing data used by or generated by any computations.
            use_minid: Use the MINID service when downloading dataset bags.
        """
        # Get or use provided credentials for server access
        self.credential = credential or get_credential(hostname)

        # Initialize server connection and catalog access
        server = DerivaServer(
            "https",
            hostname,
            credentials=self.credential,
            session_config=self._get_session_config(),
        )
        self.catalog = server.connect_ermrest(catalog_id)
        self.model = DerivaModel(self.catalog.getCatalogModel(), domain_schema=domain_schema)

        # Set up working and cache directories
        default_workdir = self.__class__.__name__ + "_working"
        self.working_dir = (
            Path(working_dir) / getpass.getuser() if working_dir else Path.home() / "deriva-ml"
        ) / default_workdir

        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir) if cache_dir else self.working_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize dataset functionality from the parent class
        super().__init__(self.model, self.cache_dir, self.working_dir, use_minid=use_minid)

        # Set up logging
        self._logger = logging.getLogger("deriva_ml")
        self._logger.setLevel(logging_level)

        # Store instance configuration
        self.host_name = hostname
        self.catalog_id = catalog_id
        self.ml_schema = ml_schema
        self.configuration = None
        self._execution: Execution | None = None
        self.domain_schema = self.model.domain_schema
        self.project_name = project_name or self.domain_schema
        self.start_time = datetime.now()
        self.status = Status.pending.value

        # Configure logging format
        logging.basicConfig(
            level=logging_level,
            format="%(asctime)s - %(name)s.%(levelname)s - %(message)s",
        )

        # Set Deriva library logging level
        deriva_logger = logging.getLogger("deriva")
        deriva_logger.setLevel(logging_level)

    def __del__(self):
        """Cleanup method to handle incomplete executions."""
        try:
            # Mark execution as aborted if not completed
            if self._execution and self._execution.status != Status.completed:
                self._execution.update_status(Status.aborted, "Execution Aborted")
        except (AttributeError, requests.HTTPError):
            pass

    @staticmethod
    def _get_session_config():
        """Returns customized HTTP session configuration.

        Configures retry behavior and connection settings for HTTP requests to the Deriva server. Settings include:
        - Idempotent retry behavior for all HTTP methods
        - Increased retry attempts for read and connect operations
        - Exponential backoff for retries

        Returns:
            dict: Session configuration dictionary with retry and connection settings.

        Example:
            >>> config = DerivaML._get_session_config()
            >>> print(config['retry_read']) # 8
        """
        # Start with a default configuration
        session_config = DEFAULT_SESSION_CONFIG.copy()

        # Customize retry behavior for robustness
        session_config.update(
            {
                # Allow retries for all HTTP methods (PUT/POST are idempotent)
                "allow_retry_on_all_methods": True,
                # Increase retry attempts for better reliability
                "retry_read": 8,
                "retry_connect": 5,
                # Use exponential backoff for retries
                "retry_backoff_factor": 5,
            }
        )
        return session_config

    @property
    def pathBuilder(self) -> SchemaWrapper:
        """Returns catalog path builder for queries.

        The path builder provides a fluent interface for constructing complex queries against the catalog.
        This is a core component used by many other methods to interact with the catalog.

        Returns:
            datapath._CatalogWrapper: A new instance of the catalog path builder.

        Example:
            >>> path = ml.pathBuilder.schemas['my_schema'].tables['my_table']
            >>> results = path.entities().fetch()
        """
        return self.catalog.getPathBuilder()

    @property
    def domain_path(self) -> datapath.DataPath:
        """Returns path builder for domain schema.

        Provides a convenient way to access tables and construct queries within the domain-specific schema.

        Returns:
            datapath._CatalogWrapper: Path builder object scoped to the domain schema.

        Example:
            >>> domain = ml.domain_path
            >>> results = domain.my_table.entities().fetch()
        """
        return self.pathBuilder.schemas[self.domain_schema]

    def table_path(self, table: str | Table) -> Path:
        """Returns a local filesystem path for table CSV files.

        Generates a standardized path where CSV files should be placed when preparing to upload data to a table.
        The path follows the project's directory structure conventions.

        Args:
            table: Name of the table or Table object to get the path for.

        Returns:
            Path: Filesystem path where the CSV file should be placed.

        Example:
            >>> path = ml.table_path("experiment_results")
            >>> df.to_csv(path) # Save data for upload
        """
        return table_path(
            self.working_dir,
            schema=self.domain_schema,
            table=self.model.name_to_table(table).name,
        )

    def download_dir(self, cached: bool = False) -> Path:
        """Returns the appropriate download directory.

        Provides the appropriate directory path for storing downloaded files, either in the cache or working directory.

        Args:
            cached: If True, returns the cache directory path. If False, returns the working directory path.

        Returns:
            Path: Directory path where downloaded files should be stored.

        Example:
            >>> cache_dir = ml.download_dir(cached=True)
            >>> work_dir = ml.download_dir(cached=False)
        """
        # Return cache directory if cached=True, otherwise working directory
        return self.cache_dir if cached else self.working_dir

    @staticmethod
    def globus_login(host: str) -> None:
        """Authenticates with Globus for accessing Deriva services.

        Performs authentication using Globus Auth to access Deriva services. If already logged in, notifies the user.
        Uses non-interactive authentication flow without a browser or local server.

        Args:
            host: The hostname of the Deriva server to authenticate with (e.g., 'deriva.example.org').

        Example:
            >>> DerivaML.globus_login('deriva.example.org')
            'Login Successful'
        """
        gnl = GlobusNativeLogin(host=host)
        if gnl.is_logged_in([host]):
            print("You are already logged in.")
        else:
            gnl.login(
                [host],
                no_local_server=True,
                no_browser=True,
                refresh_tokens=True,
                update_bdbag_keychain=True,
            )
            print("Login Successful")

    def chaise_url(self, table: RID | Table | str) -> str:
        """Generates Chaise web interface URL.

        Chaise is Deriva's web interface for data exploration. This method creates a URL that directly links to
        the specified table or record.

        Args:
            table: Table to generate URL for (name, Table object, or RID).

        Returns:
            str: URL in format: https://{host}/chaise/recordset/#{catalog}/{schema}:{table}

        Raises:
            DerivaMLException: If table or RID cannot be found.

        Examples:
            Using table name:
                >>> ml.chaise_url("experiment_table")
                'https://deriva.org/chaise/recordset/#1/schema:experiment_table'

            Using RID:
                >>> ml.chaise_url("1-abc123")
        """
        # Get the table object and build base URI
        table_obj = self.model.name_to_table(table)
        try:
            uri = self.catalog.get_server_uri().replace("ermrest/catalog/", "chaise/recordset/#")
        except DerivaMLException:
            # Handle RID case
            uri = self.cite(cast(str, table))
        return f"{uri}/{urlquote(table_obj.schema.name)}:{urlquote(table_obj.name)}"

    def cite(self, entity: Dict[str, Any] | str) -> str:
        """Generates permanent citation URL.

        Creates a versioned URL that can be used to reference a specific entity in the catalog. The URL includes
        the catalog snapshot time to ensure version stability.

        Args:
            entity: Either a RID string or a dictionary containing entity data with a 'RID' key.

        Returns:
            str: Permanent citation URL in format: https://{host}/id/{catalog}/{rid}@{snapshot_time}

        Raises:
            DerivaMLException: If an entity doesn't exist or lacks a RID.

        Examples:
            Using a RID string:
                >>> url = ml.cite("1-abc123")
                >>> print(url)
                'https://deriva.org/id/1/1-abc123@2024-01-01T12:00:00'

            Using a dictionary:
                >>> url = ml.cite({"RID": "1-abc123"})
        """
        # Return if already a citation URL
        if isinstance(entity, str) and entity.startswith(f"https://{self.host_name}/id/{self.catalog_id}/"):
            return entity

        try:
            # Resolve RID and create citation URL with snapshot time
            self.resolve_rid(rid := entity if isinstance(entity, str) else entity["RID"])
            return f"https://{self.host_name}/id/{self.catalog_id}/{rid}@{self.catalog.latest_snapshot().snaptime}"
        except KeyError as e:
            raise DerivaMLException(f"Entity {e} does not have RID column")
        except DerivaMLException as _e:
            raise DerivaMLException("Entity RID does not exist")

    def user_list(self) -> List[Dict[str, str]]:
        """Returns catalog user list.

        Retrieves basic information about all users who have access to the catalog, including their
        identifiers and full names.

        Returns:
            List[Dict[str, str]]: List of user information dictionaries, each containing:
                - 'ID': User identifier
                - 'Full_Name': User's full name

        Examples:

            >>> users = ml.user_list()
            >>> for user in users:
            ...     print(f"{user['Full_Name']} ({user['ID']})")
        """
        # Get the user table path and fetch basic user info
        user_path = self.pathBuilder.public.ERMrest_Client.path
        return [{"ID": u["ID"], "Full_Name": u["Full_Name"]} for u in user_path.entities().fetch()]

    def resolve_rid(self, rid: RID) -> ResolveRidResult:
        """Resolves RID to catalog location.

        Looks up a RID and returns information about where it exists in the catalog, including schema,
        table, and column metadata.

        Args:
            rid: Resource Identifier to resolve.

        Returns:
            ResolveRidResult: Named tuple containing:
                - schema: Schema name
                - table: Table name
                - columns: Column definitions
                - datapath: Path builder for accessing the entity

        Raises:
            DerivaMLException: If RID doesn't exist in catalog.

        Examples:
            >>> result = ml.resolve_rid("1-abc123")
            >>> print(f"Found in {result.schema}.{result.table}")
            >>> data = result.datapath.entities().fetch()
        """
        try:
            # Attempt to resolve RID using catalog model
            return self.catalog.resolve_rid(rid, self.model.model)
        except KeyError as _e:
            raise DerivaMLException(f"Invalid RID {rid}")

    def retrieve_rid(self, rid: RID) -> dict[str, Any]:
        """Retrieves complete record for RID.

        Fetches all column values for the entity identified by the RID.

        Args:
            rid: Resource Identifier of the record to retrieve.

        Returns:
            dict[str, Any]: Dictionary containing all column values for the entity.

        Raises:
            DerivaMLException: If the RID doesn't exist in the catalog.

        Example:
            >>> record = ml.retrieve_rid("1-abc123")
            >>> print(f"Name: {record['name']}, Created: {record['creation_date']}")
        """
        # Resolve RID and fetch the first (only) matching record
        return self.resolve_rid(rid).datapath.entities().fetch()[0]

    def add_page(self, title: str, content: str) -> None:
        """Adds page to web interface.

        Creates a new page in the catalog's web interface with the specified title and content. The page will be
        accessible through the catalog's navigation system.

        Args:
            title: The title of the page to be displayed in navigation and headers.
            content: The main content of the page can include HTML markup.

        Raises:
            DerivaMLException: If the page creation fails or the user lacks necessary permissions.

        Example:
            >>> ml.add_page(
            ...     title="Analysis Results",
            ...     content="<h1>Results</h1><p>Analysis completed successfully...</p>"
            ... )
        """
        # Insert page into www tables with title and content
        self.pathBuilder.www.tables[self.domain_schema].insert([{"Title": title, "Content": content}])

    def create_vocabulary(self, vocab_name: str, comment: str = "", schema: str | None = None) -> Table:
        """Creates a controlled vocabulary table.

        A controlled vocabulary table maintains a list of standardized terms and their definitions. Each term can have
        synonyms and descriptions to ensure consistent terminology usage across the dataset.

        Args:
            vocab_name: Name for the new vocabulary table. Must be a valid SQL identifier.
            comment: Description of the vocabulary's purpose and usage. Defaults to empty string.
            schema: Schema name to create the table in. If None, uses domain_schema.

        Returns:
            Table: ERMRest table object representing the newly created vocabulary table.

        Raises:
            DerivaMLException: If vocab_name is invalid or already exists.

        Examples:
            Create a vocabulary for tissue types:

                >>> table = ml.create_vocabulary(
                ...     vocab_name="tissue_types",
                ...     comment="Standard tissue classifications",
                ...     schema="bio_schema"
                ... )
        """
        # Use domain schema if none specified
        schema = schema or self.domain_schema

        # Create and return vocabulary table with RID-based URI pattern
        try:
            vocab_table = self.model.schemas[schema].create_table(
                Table.define_vocabulary(vocab_name, f"{self.project_name}:{{RID}}", comment=comment)
            )
        except ValueError:
            raise DerivaMLException(f"Table {vocab_name} already exist")
        return vocab_table

    def create_table(self, table: TableDefinition) -> Table:
        """Creates a new table in the catalog.

        Creates a table using the provided TableDefinition object, which specifies the table structure including
        columns, keys, and foreign key relationships.

        Args:
            table: A TableDefinition object containing the complete specification of the table to create.

        Returns:
            Table: The newly created ERMRest table object.

        Raises:
            DerivaMLException: If table creation fails or the definition is invalid.

        Example:

            >>> table_def = TableDefinition(
            ...     name="experiments",
            ...     column_definitions=[
            ...         ColumnDefinition(name="name", type=BuiltinTypes.text),
            ...         ColumnDefinition(name="date", type=BuiltinTypes.date)
            ...     ]
            ... )
            >>> new_table = ml.create_table(table_def)
        """
        # Create table in domain schema using provided definition
        return self.model.schemas[self.domain_schema].create_table(table.model_dump())

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def create_asset(
        self,
        asset_name: str,
        column_defs: Iterable[ColumnDefinition] | None = None,
        fkey_defs: Iterable[ColumnDefinition] | None = None,
        referenced_tables: Iterable[Table] | None = None,
        comment: str = "",
        schema: str | None = None,
    ) -> Table:
        """Creates an asset table.

        Args:
            asset_name: Name of the asset table.
            column_defs: Iterable of ColumnDefinition objects to provide additional metadata for asset.
            fkey_defs: Iterable of ForeignKeyDefinition objects to provide additional metadata for asset.
            referenced_tables: Iterable of Table objects to which asset should provide foreign-key references to.
            comment: Description of the asset table. (Default value = '')
            schema: Schema in which to create the asset table.  Defaults to domain_schema.

        Returns:
            Table object for the asset table.
        """
        # Initialize empty collections if None provided
        column_defs = column_defs or []
        fkey_defs = fkey_defs or []
        referenced_tables = referenced_tables or []
        schema = schema or self.domain_schema

        # Add an asset type to vocabulary
        self.add_term(MLVocab.asset_type, asset_name, description=f"A {asset_name} asset")

        # Create the main asset table
        asset_table = self.model.schemas[schema].create_table(
            Table.define_asset(
                schema,
                asset_name,
                column_defs=[c.model_dump() for c in column_defs],
                fkey_defs=[fk.model_dump() for fk in fkey_defs],
                comment=comment,
            )
        )

        # Create an association table between asset and asset type
        self.model.schemas[self.domain_schema].create_table(
            Table.define_association(
                [
                    (asset_table.name, asset_table),
                    ("Asset_Type", self.model.name_to_table("Asset_Type")),
                ]
            )
        )

        # Create references to other tables if specified
        for t in referenced_tables:
            asset_table.create_reference(self.model.name_to_table(t))

        # Create an association table for tracking execution
        atable = self.model.schemas[self.domain_schema].create_table(
            Table.define_association(
                [
                    (asset_name, asset_table),
                    (
                        "Execution",
                        self.model.schemas[self.ml_schema].tables["Execution"],
                    ),
                ]
            )
        )
        atable.create_reference(self.model.name_to_table("Asset_Role"))

        # Add asset annotations
        asset_annotation(asset_table)
        return asset_table

    def list_assets(self, asset_table: Table | str) -> list[dict[str, Any]]:
        """Lists contents of an asset table.

        Returns a list of assets with their types for the specified asset table.

        Args:
            asset_table: Table or name of the asset table to list assets for.

        Returns:
            list[dict[str, Any]]: List of asset records, each containing:
                - RID: Resource identifier
                - Type: Asset type
                - Metadata: Asset metadata

        Raises:
            DerivaMLException: If the table is not an asset table or doesn't exist.

        Example:
            >>> assets = ml.list_assets("tissue_types")
            >>> for asset in assets:
            ...     print(f"{asset['RID']}: {asset['Type']}")
        """
        # Validate and get asset table reference
        asset_table = self.model.name_to_table(asset_table)
        if not self.model.is_asset(asset_table):
            raise DerivaMLException(f"Table {asset_table.name} is not an asset")

        # Get path builders for asset and type tables
        pb = self._model.catalog.getPathBuilder()
        asset_path = pb.schemas[asset_table.schema.name].tables[asset_table.name]
        (
            asset_type_table,
            _,
            _,
        ) = self._model.find_association(asset_table, MLVocab.asset_type)
        type_path = pb.schemas[asset_type_table.schema.name].tables[asset_type_table.name]

        # Build a list of assets with their types
        assets = []
        for asset in asset_path.entities().fetch():
            # Get associated asset types for each asset
            asset_types = (
                type_path.filter(type_path.columns[asset_table.name] == asset["RID"])
                .attributes(type_path.Asset_Type)
                .fetch()
            )
            # Combine asset data with its types
            assets.append(
                asset | {MLVocab.asset_type.value: [asset_type[MLVocab.asset_type.value] for asset_type in asset_types]}
            )
        return assets

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def create_feature(
        self,
        target_table: Table | str,
        feature_name: str,
        terms: list[Table | str] | None = None,
        assets: list[Table | str] | None = None,
        metadata: list[ColumnDefinition | Table | Key | str] | None = None,
        optional: list[str] | None = None,
        comment: str = "",
    ) -> type[FeatureRecord]:
        """Creates a new feature definition.

        A feature represents a measurable property or characteristic that can be associated with records in the target
        table. Features can include vocabulary terms, asset references, and additional metadata.

        Args:
            target_table: Table to associate the feature with (name or Table object).
            feature_name: Unique name for the feature within the target table.
            terms: Optional vocabulary tables/names whose terms can be used as feature values.
            assets: Optional asset tables/names that can be referenced by this feature.
            metadata: Optional columns, tables, or keys to include in a feature definition.
            optional: Column names that are not required when creating feature instances.
            comment: Description of the feature's purpose and usage.

        Returns:
            type[FeatureRecord]: Feature class for creating validated instances.

        Raises:
            DerivaMLException: If a feature definition is invalid or conflicts with existing features.

        Examples:
            Create a feature with confidence score:
                >>> feature_class = ml.create_feature(
                ...     target_table="samples",
                ...     feature_name="expression_level",
                ...     terms=["expression_values"],
                ...     metadata=[ColumnDefinition(name="confidence", type=BuiltinTypes.float4)],
                ...     comment="Gene expression measurement"
                ... )
        """
        # Initialize empty collections if None provided
        terms = terms or []
        assets = assets or []
        metadata = metadata or []
        optional = optional or []

        def normalize_metadata(m: Key | Table | ColumnDefinition | str):
            """Helper function to normalize metadata references."""
            if isinstance(m, str):
                return self.model.name_to_table(m)
            elif isinstance(m, ColumnDefinition):
                return m.model_dump()
            else:
                return m

        # Validate asset and term tables
        if not all(map(self.model.is_asset, assets)):
            raise DerivaMLException("Invalid create_feature asset table.")
        if not all(map(self.model.is_vocabulary, terms)):
            raise DerivaMLException("Invalid create_feature asset table.")

        # Get references to required tables
        target_table = self.model.name_to_table(target_table)
        execution = self.model.schemas[self.ml_schema].tables["Execution"]
        feature_name_table = self.model.schemas[self.ml_schema].tables["Feature_Name"]

        # Add feature name to vocabulary
        feature_name_term = self.add_term("Feature_Name", feature_name, description=comment)
        atable_name = f"Execution_{target_table.name}_{feature_name_term.name}"
        # Create an association table implementing the feature
        atable = self.model.schemas[self.domain_schema].create_table(
            target_table.define_association(
                table_name=atable_name,
                associates=[execution, target_table, feature_name_table],
                metadata=[normalize_metadata(m) for m in chain(assets, terms, metadata)],
                comment=comment,
            )
        )
        # Configure optional columns and default feature name
        for c in optional:
            atable.columns[c].alter(nullok=True)
        atable.columns["Feature_Name"].alter(default=feature_name_term.name)

        # Return feature record class for creating instances
        return self.feature_record_class(target_table, feature_name)

    def feature_record_class(self, table: str | Table, feature_name: str) -> type[FeatureRecord]:
        """Returns a pydantic model class for feature records.

        Creates a typed interface for creating new instances of the specified feature. The returned class includes
        validation and type checking based on the feature's definition.

        Args:
            table: The table containing the feature, either as name or Table object.
            feature_name: Name of the feature to create a record class for.

        Returns:
            type[FeatureRecord]: A pydantic model class for creating validated feature records.

        Raises:
            DerivaMLException: If the feature doesn't exist or the table is invalid.

        Example:
            >>> ExpressionFeature = ml.feature_record_class("samples", "expression_level")
            >>> feature = ExpressionFeature(value="high", confidence=0.95)
        """
        # Look up a feature and return its record class
        return self.lookup_feature(table, feature_name).feature_record_class()

    def delete_feature(self, table: Table | str, feature_name: str) -> bool:
        """Removes a feature definition and its data.

        Deletes the feature and its implementation table from the catalog. This operation cannot be undone and
        will remove all feature values associated with this feature.

        Args:
            table: The table containing the feature, either as name or Table object.
            feature_name: Name of the feature to delete.

        Returns:
            bool: True if the feature was successfully deleted, False if it didn't exist.

        Raises:
            DerivaMLException: If deletion fails due to constraints or permissions.

        Example:
            >>> success = ml.delete_feature("samples", "obsolete_feature")
            >>> print("Deleted" if success else "Not found")
        """
        # Get table reference and find feature
        table = self.model.name_to_table(table)
        try:
            # Find and delete the feature's implementation table
            feature = next(f for f in self.model.find_features(table) if f.feature_name == feature_name)
            feature.feature_table.drop()
            return True
        except StopIteration:
            return False

    def lookup_feature(self, table: str | Table, feature_name: str) -> Feature:
        """Retrieves a Feature object.

        Looks up and returns a Feature object that provides an interface to work with an existing feature
        definition in the catalog.

        Args:
            table: The table containing the feature, either as name or Table object.
            feature_name: Name of the feature to look up.

        Returns:
            Feature: An object representing the feature and its implementation.

        Raises:
            DerivaMLException: If the feature doesn't exist in the specified table.

        Example:
            >>> feature = ml.lookup_feature("samples", "expression_level")
            >>> print(feature.feature_name)
            'expression_level'
        """
        return self.model.lookup_feature(table, feature_name)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def list_feature_values(self, table: Table | str, feature_name: str) -> datapath._ResultSet:
        """Retrieves all values for a feature.

        Returns all instances of the specified feature that have been created, including their associated
        metadata and references.

        Args:
            table: The table containing the feature, either as name or Table object.
            feature_name: Name of the feature to retrieve values for.

        Returns:
            datapath._ResultSet: A result set containing all feature values and their metadata.

        Raises:
            DerivaMLException: If the feature doesn't exist or cannot be accessed.

        Example:
            >>> values = ml.list_feature_values("samples", "expression_level")
            >>> for value in values:
            ...     print(f"Sample {value['RID']}: {value['value']}")
        """
        # Get table and feature references
        table = self.model.name_to_table(table)
        feature = self.lookup_feature(table, feature_name)

        # Build and execute query for feature values
        pb = self.catalog.getPathBuilder()
        return pb.schemas[feature.feature_table.schema.name].tables[feature.feature_table.name].entities().fetch()

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def add_term(
        self,
        table: str | Table,
        term_name: str,
        description: str,
        synonyms: list[str] | None = None,
        exists_ok: bool = True,
    ) -> VocabularyTerm:
        """Adds a term to a vocabulary table.

        Creates a new standardized term with description and optional synonyms in a vocabulary table.
        Can either create a new term or return an existing one if it already exists.

        Args:
            table: Vocabulary table to add term to (name or Table object).
            term_name: Primary name of the term (must be unique within vocabulary).
            description: Explanation of term's meaning and usage.
            synonyms: Alternative names for the term.
            exists_ok: If True, return the existing term if found. If False, raise error.

        Returns:
            VocabularyTerm: Object representing the created or existing term.

        Raises:
            DerivaMLException: If a term exists and exists_ok=False, or if the table is not a vocabulary table.

        Examples:
            Add a new tissue type:
                >>> term = ml.add_term(
                ...     table="tissue_types",
                ...     term_name="epithelial",
                ...     description="Epithelial tissue type",
                ...     synonyms=["epithelium"]
                ... )

            Attempt to add an existing term:
                >>> term = ml.add_term("tissue_types", "epithelial", "...", exists_ok=True)
        """
        # Initialize an empty synonyms list if None
        synonyms = synonyms or []

        # Get table reference and validate if it is a vocabulary table
        table = self.model.name_to_table(table)
        pb = self.catalog.getPathBuilder()
        if not (self.model.is_vocabulary(table)):
            raise DerivaMLTableTypeError("vocabulary", table.name)

        # Get schema and table names for path building
        schema_name = table.schema.name
        table_name = table.name

        try:
            # Attempt to insert a new term
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
            # Term exists - look it up or raise an error
            term_id = self.lookup_term(table, term_name)
            if not exists_ok:
                raise DerivaMLInvalidTerm(table.name, term_name, msg="term already exists")
        return term_id

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def lookup_term(self, table: str | Table, term_name: str) -> VocabularyTerm:
        """Finds a term in a vocabulary table.

        Searches for a term in the specified vocabulary table, matching either the primary name
        or any of its synonyms.

        Args:
            table: Vocabulary table to search in (name or Table object).
            term_name: Name or synonym of the term to find.

        Returns:
            VocabularyTerm: The matching vocabulary term.

        Raises:
            DerivaMLVocabularyException: If the table is not a vocabulary table, or term is not found.

        Examples:
            Look up by primary name:
                >>> term = ml.lookup_term("tissue_types", "epithelial")
                >>> print(term.description)

            Look up by synonym:
                >>> term = ml.lookup_term("tissue_types", "epithelium")
        """
        # Get and validate vocabulary table reference
        vocab_table = self.model.name_to_table(table)
        if not self.model.is_vocabulary(vocab_table):
            raise DerivaMLException(f"The table {table} is not a controlled vocabulary")

        # Get schema and table paths
        schema_name, table_name = vocab_table.schema.name, vocab_table.name
        schema_path = self.catalog.getPathBuilder().schemas[schema_name]

        # Search for term by name or synonym
        for term in schema_path.tables[table_name].entities().fetch():
            if term_name == term["Name"] or (term["Synonyms"] and term_name in term["Synonyms"]):
                return VocabularyTerm.model_validate(term)

        # Term not found
        raise DerivaMLInvalidTerm(table_name, term_name)

    def list_vocabulary_terms(self, table: str | Table) -> list[VocabularyTerm]:
        """Lists all terms in a vocabulary table.

        Retrieves all terms, their descriptions, and synonyms from a controlled vocabulary table.

        Args:
            table: Vocabulary table to list terms from (name or Table object).

        Returns:
            list[VocabularyTerm]: List of vocabulary terms with their metadata.

        Raises:
            DerivaMLException: If table doesn't exist or is not a vocabulary table.

        Examples:
            >>> terms = ml.list_vocabulary_terms("tissue_types")
            >>> for term in terms:
            ...     print(f"{term.name}: {term.description}")
            ...     if term.synonyms:
            ...         print(f"  Synonyms: {', '.join(term.synonyms)}")
        """
        # Get path builder and table reference
        pb = self.catalog.getPathBuilder()
        table = self.model.name_to_table(table.value if isinstance(table, MLVocab) else table)

        # Validate table is a vocabulary table
        if not (self.model.is_vocabulary(table)):
            raise DerivaMLException(f"The table {table} is not a controlled vocabulary")

        # Fetch and convert all terms to VocabularyTerm objects
        return [VocabularyTerm(**v) for v in pb.schemas[table.schema.name].tables[table.name].entities().fetch()]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def download_dataset_bag(
        self,
        dataset: DatasetSpec,
        execution_rid: RID | None = None,
    ) -> DatasetBag:
        """Downloads a dataset to the local filesystem and creates a MINID if needed.

        Downloads a dataset specified by DatasetSpec to the local filesystem. If the dataset doesn't have
        a MINID (Minimal Viable Identifier), one will be created. The dataset can optionally be associated
        with an execution record.

        Args:
            dataset: Specification of the dataset to download, including version and materialization options.
            execution_rid: Optional execution RID to associate the download with.

        Returns:
            DatasetBag: Object containing:
                - path: Local filesystem path to downloaded dataset
                - rid: Dataset's Resource Identifier
                - minid: Dataset's Minimal Viable Identifier

        Examples:
            Download with default options:
                >>> spec = DatasetSpec(rid="1-abc123")
                >>> bag = ml.download_dataset_bag(dataset=spec)
                >>> print(f"Downloaded to {bag.path}")

            Download with execution tracking:
                >>> bag = ml.download_dataset_bag(
                ...     dataset=DatasetSpec(rid="1-abc123", materialize=True),
                ...     execution_rid="1-xyz789"
                ... )
        """
        if not self._is_dataset_rid(dataset.rid):
            raise DerivaMLTableTypeError("Dataset", dataset.rid)
        return self._download_dataset_bag(
            dataset=dataset,
            execution_rid=execution_rid,
            snapshot_catalog=DerivaML(self.host_name, self._version_snapshot(dataset)),
        )

    def _update_status(self, new_status: Status, status_detail: str, execution_rid: RID):
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

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def add_files(
        self,
        files: Iterable[FileSpec],
        dataset_types: str | list[str] | None = None,
        description: str = "",
        execution_rid: RID | None = None,
    ) -> RID:
        """Adds files to the catalog with their metadata.

        Registers files in the catalog along with their metadata (MD5, length, URL) and associates them with
        specified file types. Optionally links files to an execution record.

        Args:
            files: File specifications containing MD5 checksum, length, and URL.
            dataset_types: One or more dataset type terms from File_Type vocabulary.
            description: Description of the files.
            execution_rid: Optional execution RID to associate files with.

        Returns:
            RID: Resource of dataset that represents the newly added files.

        Raises:
            DerivaMLException: If file_types are invalid or execution_rid is not an execution record.

        Examples:
            Add a single file type:
                >>> files = [FileSpec(url="path/to/file.txt", md5="abc123", length=1000)]
                >>> rids = ml.add_files(files, file_types="text")

            Add multiple file types:
                >>> rids = ml.add_files(
                ...     files=[FileSpec(url="image.png", md5="def456", length=2000)],
                ...     file_types=["image", "png"],
                ...     execution_rid="1-xyz789"
                ... )
        """
        if execution_rid and self.resolve_rid(execution_rid).table.name != "Execution":
            raise DerivaMLTableTypeError("Execution", execution_rid)

        filespec_list = list(files)

        # Get a list of all defined file types and their synonyms.
        defined_types = set(
            chain.from_iterable([[t.name] + t.synonyms for t in self.list_vocabulary_terms(MLVocab.asset_type)])
        )

        # Get a list of al of the file types used in the filespec_list
        spec_types = set(chain.from_iterable(filespec.file_types for filespec in filespec_list))

        # Now make sure that all of the file types and dataset_types in the spec list are defined.
        if spec_types - defined_types:
            raise DerivaMLInvalidTerm(MLVocab.asset_type.name, f"{spec_types - defined_types}")

        # Normalize dataset_types, make sure FIle type is included.
        if isinstance(dataset_types, list):
            dataset_types = ["File"] + dataset_types if "File" not in dataset_types else dataset_types
        else:
            dataset_types = ["File", dataset_types] if dataset_types else ["File"]
        for ds_type in dataset_types:
            self.lookup_term(MLVocab.dataset_type, ds_type)

        # Add files to the file table, and collect up the resulting entries by directory name.
        pb = self._model.catalog.getPathBuilder()
        file_records = list(
            pb.schemas[self.ml_schema].tables["File"].insert([f.model_dump(by_alias=True) for f in filespec_list])
        )

        # Get the name of the association table between file_table and file_type and add file_type records
        atable = self.model.find_association(MLTable.file, MLVocab.asset_type)[0].name
        # Need to get a link between file record and file_types.
        type_map = {
            file_spec.md5: file_spec.file_types + ([] if "File" in file_spec.file_types else [])
            for file_spec in filespec_list
        }
        file_type_records = [
            {MLVocab.asset_type.value: file_type, "File": file_record["RID"]}
            for file_record in file_records
            for file_type in type_map[file_record["MD5"]]
        ]
        pb.schemas[self._ml_schema].tables[atable].insert(file_type_records)

        if execution_rid:
            # Get the name of the association table between file_table and execution.
            pb.schemas[self._ml_schema].File_Execution.insert(
                [
                    {"File": file_record["RID"], "Execution": execution_rid, "Asset_Role": "Output"}
                    for file_record in file_records
                ]
            )

        # Now create datasets to capture the original directory structure of the files.
        dir_rid_map = defaultdict(list)
        for e in file_records:
            dir_rid_map[Path(urlsplit(e["URL"]).path).parent].append(e["RID"])

        nested_datasets = []
        path_length = 0
        dataset = None
        # Start with the longest path so we get subdirectories first.
        for p, rids in sorted(dir_rid_map.items(), key=lambda kv: len(kv[0].parts), reverse=True):
            dataset = self.create_dataset(
                dataset_types=dataset_types, execution_rid=execution_rid, description=description
            )
            members = rids
            if len(p.parts) < path_length:
                # Going up one level in directory, so Create nested dataset
                members = nested_datasets + rids
                nested_datasets = []
            self.add_dataset_members(dataset_rid=dataset, members=members, execution_rid=execution_rid)
            nested_datasets.append(dataset)
            path_length = len(p.parts)

        return dataset

    def list_files(self, file_types: list[str] | None = None) -> list[dict[str, Any]]:
        """Lists files in the catalog with their metadata.

        Returns a list of files with their metadata including URL, MD5 hash, length, description,
        and associated file types. Files can be optionally filtered by type.

        Args:
            file_types: Filter results to only include these file types.

        Returns:
            list[dict[str, Any]]: List of file records, each containing:
                - RID: Resource identifier
                - URL: File location
                - MD5: File hash
                - Length: File size
                - Description: File description
                - File_Types: List of associated file types

        Examples:
            List all files:
                >>> files = ml.list_files()
                >>> for f in files:
                ...     print(f"{f['RID']}: {f['URL']}")

            Filter by file type:
                >>> image_files = ml.list_files(["image", "png"])
        """

        asset_type_atable, file_fk, asset_type_fk = self.model.find_association("File", "Asset_Type")
        ml_path = self.pathBuilder.schemas[self._ml_schema]
        file = ml_path.File
        asset_type = ml_path.tables[asset_type_atable.name]

        path = file.path
        path = path.link(asset_type.alias("AT"), on=file.RID == asset_type.columns[file_fk], join_type="left")
        if file_types:
            path = path.filter(asset_type.columns[asset_type_fk] == datapath.Any(*file_types))
        path = path.attributes(
            path.File.RID,
            path.File.URL,
            path.File.MD5,
            path.File.Length,
            path.File.Description,
            path.AT.columns[asset_type_fk],
        )

        file_map = {}
        for f in path.fetch():
            entry = file_map.setdefault(f["RID"], {**f, "File_Types": []})
            if ft := f.get("Asset_Type"):  # assign-and-test in one go
                entry["File_Types"].append(ft)

        # Now get rid of the File_Type key and return the result
        return [(f, f.pop("Asset_Type"))[0] for f in file_map.values()]

    def list_workflows(self) -> list[Workflow]:
        """Lists all workflows in the catalog.

        Retrieves all workflow definitions, including their names, URLs, types, versions,
        and descriptions.

        Returns:
            list[Workflow]: List of workflow objects, each containing:
                - name: Workflow name
                - url: Source code URL
                - workflow_type: Type of workflow
                - version: Version identifier
                - description: Workflow description
                - rid: Resource identifier
                - checksum: Source code checksum

        Examples:
            >>> workflows = ml.list_workflows()
            >>> for w in workflows:
                    print(f"{w.name} (v{w.version}): {w.description}")
                    print(f"  Source: {w.url}")
        """
        # Get a workflow table path and fetch all workflows
        workflow_path = self.pathBuilder.schemas[self.ml_schema].Workflow
        return [
            Workflow(
                name=w["Name"],
                url=w["URL"],
                workflow_type=w["Workflow_Type"],
                version=w["Version"],
                description=w["Description"],
                rid=w["RID"],
                checksum=w["Checksum"],
            )
            for w in workflow_path.entities().fetch()
        ]

    def add_workflow(self, workflow: Workflow) -> RID:
        """Adds a workflow to the catalog.

        Registers a new workflow in the catalog or returns the RID of an existing workflow with the same URL.
        Each workflow represents a specific computational process or analysis pipeline.

        Args:
            workflow: Workflow object containing name, URL, type, version, and description.

        Returns:
            RID: Resource Identifier of the added or existing workflow.

        Raises:
            DerivaMLException: If workflow insertion fails or required fields are missing.

        Examples:
            >>> workflow = Workflow(
            ...     name="Gene Analysis",
            ...     url="https://github.com/org/repo/workflows/gene_analysis.py",
            ...     workflow_type="python_script",
            ...     version="1.0.0",
            ...     description="Analyzes gene expression patterns"
            ... )
            >>> workflow_rid = ml.add_workflow(workflow)
        """
        # Check if a workflow already exists by URL
        if workflow_rid := self.lookup_workflow(workflow.url):
            return workflow_rid

        # Get an ML schema path for the workflow table
        ml_schema_path = self.pathBuilder.schemas[self.ml_schema]
        try:
            # Create a workflow record
            workflow_record = {
                "URL": workflow.url,
                "Name": workflow.name,
                "Description": workflow.description,
                "Checksum": workflow.checksum,
                "Version": workflow.version,
                MLVocab.workflow_type: self.lookup_term(MLVocab.workflow_type, workflow.workflow_type).name,
            }
            # Insert a workflow and get its RID
            workflow_rid = ml_schema_path.Workflow.insert([workflow_record])[0]["RID"]
        except Exception as e:
            error = format_exception(e)
            raise DerivaMLException(f"Failed to insert workflow. Error: {error}")
        return workflow_rid

    def lookup_workflow(self, url: str) -> RID | None:
        """Finds a workflow by URL.

        Args:
            url: URL of the workflow to find.

        Returns:
            RID: Resource Identifier of the workflow if found, None otherwise.

        Example:
            >>> rid = ml.lookup_workflow("https://github.com/org/repo/workflow.py")
            >>> if rid:
            ...     print(f"Found workflow: {rid}")
        """
        # Get a workflow table path
        workflow_path = self.pathBuilder.schemas[self.ml_schema].Workflow
        try:
            # Search for workflow by URL
            url_column = workflow_path.URL
            return list(workflow_path.filter(url_column == url).entities())[0]["RID"]
        except IndexError:
            return None

    def create_workflow(self, name: str, workflow_type: str, description: str = "") -> Workflow:
        """Creates a new workflow definition.

        Creates a Workflow object that represents a computational process or analysis pipeline. The workflow type
        must be a term from the controlled vocabulary. This method is typically used to define new analysis
        workflows before execution.

        Args:
            name: Name of the workflow.
            workflow_type: Type of workflow (must exist in workflow_type vocabulary).
            description: Description of what the workflow does.

        Returns:
            Workflow: New workflow object ready for registration.

        Raises:
            DerivaMLException: If workflow_type is not in the vocabulary.

        Examples:
            >>> workflow = ml.create_workflow(
            ...     name="RNA Analysis",
            ...     workflow_type="python_notebook",
            ...     description="RNA sequence analysis pipeline"
            ... )
            >>> rid = ml.add_workflow(workflow)
        """
        # Validate workflow type exists in vocabulary
        self.lookup_term(MLVocab.workflow_type, workflow_type)

        # Create and return a new workflow object
        return Workflow.create_workflow(name, workflow_type, description)

    def create_execution(self, configuration: ExecutionConfiguration, dry_run: bool = False) -> "Execution":
        """Creates an execution environment.

        Given an execution configuration, initialize the local compute environment to prepare for executing an
        ML or analytic routine.  This routine has a number of side effects.

        1. The datasets specified in the configuration are downloaded and placed in the cache-dir. If a version is
        not specified in the configuration, then a new minor version number is created for the dataset and downloaded.

        2. If any execution assets are provided in the configuration, they are downloaded
        and placed in the working directory.


        Args:
            configuration: ExecutionConfiguration:
            dry_run: Do not create an execution record or upload results.

        Returns:
            An execution object.
        """
        # Import here to avoid circular dependency
        from deriva_ml.execution.execution import Execution

        # Create and store an execution instance
        self._execution = Execution(configuration, self, dry_run=dry_run)
        return self._execution

    def restore_execution(self, execution_rid: RID | None = None) -> Execution:
        """Restores a previous execution.

        Given an execution RID, retrieves the execution configuration and restores the local compute environment.
        This routine has a number of side effects.

        1. The datasets specified in the configuration are downloaded and placed in the cache-dir. If a version is
        not specified in the configuration, then a new minor version number is created for the dataset and downloaded.

        2. If any execution assets are provided in the configuration, they are downloaded and placed
        in the working directory.

        Args:
            execution_rid: Resource Identifier (RID) of the execution to restore.

        Returns:
            Execution: An execution object representing the restored execution environment.

        Raises:
            DerivaMLException: If execution_rid is not valid or execution cannot be restored.

        Example:
            >>> execution = ml.restore_execution("1-abc123")
        """
        # Import here to avoid circular dependency
        from deriva_ml.execution.execution import Execution

        # If no RID provided, try to find single execution in working directory
        if not execution_rid:
            e_rids = execution_rids(self.working_dir)
            if len(e_rids) != 1:
                raise DerivaMLException(f"Multiple execution RIDs were found {e_rids}.")
            execution_rid = e_rids[0]

        # Try to load configuration from a file
        cfile = asset_file_path(
            prefix=self.working_dir,
            exec_rid=execution_rid,
            file_name="configuration.json",
            asset_table=self.model.name_to_table("Execution_Metadata"),
            metadata={},
        )

        # Load configuration from a file or create from an execution record
        if cfile.exists():
            configuration = ExecutionConfiguration.load_configuration(cfile)
        else:
            execution = self.retrieve_rid(execution_rid)
            configuration = ExecutionConfiguration(
                workflow=execution["Workflow"],
                description=execution["Description"],
            )

        # Create and return an execution instance
        return Execution(configuration, self, reload=execution_rid)
