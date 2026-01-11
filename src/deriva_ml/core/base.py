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
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, cast, TYPE_CHECKING, Any
from typing_extensions import Self

# Third-party imports
import requests

# Deriva imports
from deriva.core import DEFAULT_SESSION_CONFIG, get_credential, urlquote
from deriva.core.deriva_server import DerivaServer
from deriva.core.ermrest_catalog import ErmrestCatalog, ErmrestSnapshot
from deriva.core.ermrest_model import Table
from deriva.core.utils.core_utils import DEFAULT_LOGGER_OVERRIDES
from deriva.core.utils.globus_auth_utils import GlobusNativeLogin

from deriva_ml.core.definitions import ML_SCHEMA, RID, Status, TableDefinition
from deriva_ml.core.config import DerivaMLConfig
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.interfaces import DerivaMLCatalog
from deriva_ml.core.mixins import (
    VocabularyMixin,
    RidResolutionMixin,
    PathBuilderMixin,
    WorkflowMixin,
    FeatureMixin,
    DatasetMixin,
    AssetMixin,
    ExecutionMixin,
    FileMixin,
)

# Optional debug imports
try:
    from icecream import ic

    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

if TYPE_CHECKING:
    from deriva_ml.execution.execution import Execution
    from deriva_ml.model.catalog import DerivaModel

# Stop pycharm from complaining about undefined references.
ml: DerivaML


class DerivaML(
    PathBuilderMixin,
    RidResolutionMixin,
    VocabularyMixin,
    WorkflowMixin,
    FeatureMixin,
    DatasetMixin,
    AssetMixin,
    ExecutionMixin,
    FileMixin,
    DerivaMLCatalog,
):
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

    # Class-level type annotations for DerivaMLCatalog protocol compliance
    ml_schema: str
    domain_schema: str
    model: DerivaModel
    cache_dir: Path
    working_dir: Path
    catalog: ErmrestCatalog | ErmrestSnapshot
    catalog_id: str | int

    @classmethod
    def instantiate(cls, config: DerivaMLConfig) -> Self:
        return cls(**config.model_dump())

    def __init__(
        self,
        hostname: str,
        catalog_id: str | int,
        domain_schema: str | None = None,
        project_name: str | None = None,
        cache_dir: str | Path | None = None,
        working_dir: str | Path | None = None,
        hydra_runtime_output_dir: str | Path | None = None,
        ml_schema: str = ML_SCHEMA,
        logging_level: int = logging.WARNING,
        deriva_logging_level: int = logging.WARNING,
        credential: dict | None = None,
        use_minid: bool = True,
        check_auth: bool = True,
    ) -> None:
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
            cache_dir: Directory path for caching data downloaded from the Deriva server as bdbag. If not provided,
                will default to working_dir.
            working_dir: Directory path for storing data used by or generated by any computations. If no value is
                provided, will default to  ${HOME}/deriva_ml
            use_minid: Use the MINID service when downloading dataset bags.
            check_auth: Check if the user has access to the catalog.
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
        try:
            if check_auth and server.get_authn_session():
                pass
        except Exception:
            raise DerivaMLException(
                "You are not authorized to access this catalog. "
                "Please check your credentials and make sure you have logged in."
            )
        self.catalog = server.connect_ermrest(catalog_id)
        # Import here to avoid circular imports
        from deriva_ml.model.catalog import DerivaModel
        self.model = DerivaModel(self.catalog.getCatalogModel(), domain_schema=domain_schema)
        self.use_minid = use_minid

        # Set up working and cache directories
        self.working_dir = DerivaMLConfig.compute_workdir(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.hydra_runtime_output_dir = hydra_runtime_output_dir

        self.cache_dir = Path(cache_dir) if cache_dir else self.working_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self._logger = logging.getLogger("deriva_ml")
        self._logger.setLevel(logging_level)
        self._logging_level = logging_level
        self._deriva_logging_level = deriva_logging_level

        # Configure deriva logging level
        logger_config = DEFAULT_LOGGER_OVERRIDES
        # allow for reconfiguration of module-specific logging levels
        [logging.getLogger(name).setLevel(level) for name, level in logger_config.items()]
        logging.getLogger("root").setLevel(deriva_logging_level)
        logging.getLogger("bagit").setLevel(deriva_logging_level)
        logging.getLogger("bdbag").setLevel(deriva_logging_level)

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

    def __del__(self) -> None:
        """Cleanup method to handle incomplete executions."""
        try:
            # Mark execution as aborted if not completed
            if self._execution and self._execution.status != Status.completed:
                self._execution.update_status(Status.aborted, "Execution Aborted")
        except (AttributeError, requests.HTTPError):
            pass

    @staticmethod
    def _get_session_config() -> dict:
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

    def is_snapshot(self) -> bool:
        return hasattr(self.catalog, "_snaptime")

    def catalog_snapshot(self, version_snapshot: str) -> Self:
        """Returns a DerivaML instance for a specific snapshot of the catalog."""
        return DerivaML(
            self.host_name,
            version_snapshot,
            logging_level=self._logging_level,
            deriva_logging_level=self._deriva_logging_level,
        )

    @property
    def _dataset_table(self) -> Table:
        return self.model.schemas[self.model.ml_schema].tables["Dataset"]

    # pathBuilder, domain_path, table_path moved to PathBuilderMixin

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
        user_path = self.pathBuilder().public.ERMrest_Client.path
        return [{"ID": u["ID"], "Full_Name": u["Full_Name"]} for u in user_path.entities().fetch()]

    # resolve_rid, retrieve_rid moved to RidResolutionMixin

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
        self.pathBuilder().www.tables[self.domain_schema].insert([{"Title": title, "Content": content}])

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


    # Methods moved to mixins:
    # - create_asset, list_assets -> AssetMixin
    # - create_feature, feature_record_class, delete_feature, lookup_feature, list_feature_values -> FeatureMixin
    # - find_datasets, create_dataset, lookup_dataset, delete_dataset, list_dataset_element_types,
    #   add_dataset_element_type, download_dataset_bag -> DatasetMixin
    # - _update_status, create_execution, restore_execution -> ExecutionMixin
    # - add_files, list_files, _bootstrap_versions, _synchronize_dataset_versions, _set_version_snapshot -> FileMixin

