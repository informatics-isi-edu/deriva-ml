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

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib
_deriva_core = importlib.import_module("deriva.core")
_deriva_server = importlib.import_module("deriva.core.deriva_server")
_ermrest_catalog = importlib.import_module("deriva.core.ermrest_catalog")
_ermrest_model = importlib.import_module("deriva.core.ermrest_model")
_core_utils = importlib.import_module("deriva.core.utils.core_utils")
_globus_auth_utils = importlib.import_module("deriva.core.utils.globus_auth_utils")

DEFAULT_SESSION_CONFIG = _deriva_core.DEFAULT_SESSION_CONFIG
get_credential = _deriva_core.get_credential
urlquote = _deriva_core.urlquote
DerivaServer = _deriva_server.DerivaServer
ErmrestCatalog = _ermrest_catalog.ErmrestCatalog
ErmrestSnapshot = _ermrest_catalog.ErmrestSnapshot
Table = _ermrest_model.Table
DEFAULT_LOGGER_OVERRIDES = _core_utils.DEFAULT_LOGGER_OVERRIDES
deriva_tags = _core_utils.tag
GlobusNativeLogin = _globus_auth_utils.GlobusNativeLogin

from deriva_ml.core.config import DerivaMLConfig
from deriva_ml.core.definitions import ML_SCHEMA, RID, Status, TableDefinition, VocabularyTableDef
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.core.logging_config import apply_logger_overrides, configure_logging
from deriva_ml.dataset.upload import bulk_upload_configuration
from deriva_ml.interfaces import DerivaMLCatalog
from deriva_ml.core.mixins import (
    AnnotationMixin,
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
    AnnotationMixin,
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
    domain_schemas: frozenset[str]
    default_schema: str | None
    model: DerivaModel
    cache_dir: Path
    working_dir: Path
    catalog: ErmrestCatalog | ErmrestSnapshot
    catalog_id: str | int

    @classmethod
    def instantiate(cls, config: DerivaMLConfig) -> Self:
        """Create a DerivaML instance from a configuration object.

        This method is the preferred way to instantiate DerivaML when using hydra-zen
        for configuration management. It accepts a DerivaMLConfig (Pydantic model) and
        unpacks it to create the instance.

        This pattern allows hydra-zen's `instantiate()` to work with DerivaML:

        Example with hydra-zen:
            >>> from hydra_zen import builds, instantiate
            >>> from deriva_ml import DerivaML
            >>> from deriva_ml.core.config import DerivaMLConfig
            >>>
            >>> # Create a structured config using hydra-zen
            >>> DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)
            >>>
            >>> # Configure for your environment
            >>> conf = DerivaMLConf(
            ...     hostname='deriva.example.org',
            ...     catalog_id='42',
            ...     domain_schema='my_domain',
            ... )
            >>>
            >>> # Instantiate the config to get a DerivaMLConfig object
            >>> config = instantiate(conf)
            >>>
            >>> # Create the DerivaML instance
            >>> ml = DerivaML.instantiate(config)

        Args:
            config: A DerivaMLConfig object containing all configuration parameters.

        Returns:
            A new DerivaML instance configured according to the config object.

        Note:
            The DerivaMLConfig class integrates with Hydra's configuration system
            and registers custom resolvers for computing working directories.
            See `deriva_ml.core.config` for details on configuration options.
        """
        return cls(**config.model_dump())

    def __init__(
        self,
        hostname: str,
        catalog_id: str | int,
        domain_schemas: set[str] | None = None,
        default_schema: str | None = None,
        project_name: str | None = None,
        cache_dir: str | Path | None = None,
        working_dir: str | Path | None = None,
        hydra_runtime_output_dir: str | Path | None = None,
        ml_schema: str = ML_SCHEMA,
        logging_level: int = logging.WARNING,
        deriva_logging_level: int = logging.WARNING,
        credential: dict | None = None,
        s3_bucket: str | None = None,
        use_minid: bool | None = None,
        check_auth: bool = True,
        clean_execution_dir: bool = True,
    ) -> None:
        """Initializes a DerivaML instance.

        This method will connect to a catalog and initialize local configuration for the ML execution.
        This class is intended to be used as a base class on which domain-specific interfaces are built.

        Args:
            hostname: Hostname of the Deriva server.
            catalog_id: Catalog ID. Either an identifier or a catalog name.
            domain_schemas: Optional set of domain schema names. If None, auto-detects all
                non-system schemas. Use this when working with catalogs that have multiple
                user-defined schemas.
            default_schema: The default schema for table creation operations. If None and
                there is exactly one domain schema, that schema is used. If there are multiple
                domain schemas, this must be specified for table creation to work without
                explicit schema parameters.
            ml_schema: Schema name for ML schema. Used if you have a non-standard configuration of deriva-ml.
            project_name: Project name. Defaults to name of default_schema.
            cache_dir: Directory path for caching data downloaded from the Deriva server as bdbag. If not provided,
                will default to working_dir.
            working_dir: Directory path for storing data used by or generated by any computations. If no value is
                provided, will default to  ${HOME}/deriva_ml
            s3_bucket: S3 bucket URL for dataset bag storage (e.g., 's3://my-bucket'). If provided,
                enables MINID creation and S3 upload for dataset exports. If None, MINID functionality
                is disabled regardless of use_minid setting.
            use_minid: Use the MINID service when downloading dataset bags. Only effective when
                s3_bucket is configured. If None (default), automatically set to True when s3_bucket
                is provided, False otherwise.
            check_auth: Check if the user has access to the catalog.
            clean_execution_dir: Whether to automatically clean up execution working directories
                after successful upload. Defaults to True. Set to False to retain local copies.
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
        self.model = DerivaModel(
            self.catalog.getCatalogModel(),
            ml_schema=ml_schema,
            domain_schemas=domain_schemas,
            default_schema=default_schema,
        )

        # Store S3 bucket configuration and resolve use_minid
        self.s3_bucket = s3_bucket
        if use_minid is None:
            # Auto mode: enable MINID if s3_bucket is configured
            self.use_minid = s3_bucket is not None
        elif use_minid and s3_bucket is None:
            # User requested MINID but no S3 bucket configured - disable MINID
            self.use_minid = False
        else:
            self.use_minid = use_minid

        # Set up working and cache directories
        self.working_dir = DerivaMLConfig.compute_workdir(working_dir, catalog_id)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.hydra_runtime_output_dir = hydra_runtime_output_dir

        self.cache_dir = Path(cache_dir) if cache_dir else self.working_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging using centralized configuration
        # This configures deriva_ml, Hydra, and deriva-py loggers without
        # affecting the root logger or calling basicConfig()
        self._logger = configure_logging(
            level=logging_level,
            deriva_level=deriva_logging_level,
        )
        self._logging_level = logging_level
        self._deriva_logging_level = deriva_logging_level

        # Apply deriva's default logger overrides for fine-grained control
        apply_logger_overrides(DEFAULT_LOGGER_OVERRIDES)

        # Store instance configuration
        self.host_name = hostname
        self.catalog_id = catalog_id
        self.ml_schema = ml_schema
        self.configuration = None
        self._execution: Execution | None = None
        self.domain_schemas = self.model.domain_schemas
        self.default_schema = self.model.default_schema
        self.project_name = project_name or self.default_schema or "deriva-ml"
        self.start_time = datetime.now()
        self.status = Status.pending.value
        self.clean_execution_dir = clean_execution_dir

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

    def cite(self, entity: Dict[str, Any] | str, current: bool = False) -> str:
        """Generates citation URL for an entity.

        Creates a URL that can be used to reference a specific entity in the catalog.
        By default, includes the catalog snapshot time to ensure version stability
        (permanent citation). With current=True, returns a URL to the current state.

        Args:
            entity: Either a RID string or a dictionary containing entity data with a 'RID' key.
            current: If True, return URL to current catalog state (no snapshot).
                     If False (default), return permanent citation URL with snapshot time.

        Returns:
            str: Citation URL. Format depends on `current` parameter:
                - current=False: https://{host}/id/{catalog}/{rid}@{snapshot_time}
                - current=True: https://{host}/id/{catalog}/{rid}

        Raises:
            DerivaMLException: If an entity doesn't exist or lacks a RID.

        Examples:
            Permanent citation (default):
                >>> url = ml.cite("1-abc123")
                >>> print(url)
                'https://deriva.org/id/1/1-abc123@2024-01-01T12:00:00'

            Current catalog URL:
                >>> url = ml.cite("1-abc123", current=True)
                >>> print(url)
                'https://deriva.org/id/1/1-abc123'

            Using a dictionary:
                >>> url = ml.cite({"RID": "1-abc123"})
        """
        # Return if already a citation URL
        if isinstance(entity, str) and entity.startswith(f"https://{self.host_name}/id/{self.catalog_id}/"):
            return entity

        try:
            # Resolve RID and create citation URL
            self.resolve_rid(rid := entity if isinstance(entity, str) else entity["RID"])
            base_url = f"https://{self.host_name}/id/{self.catalog_id}/{rid}"
            if current:
                return base_url
            return f"{base_url}@{self.catalog.latest_snapshot().snaptime}"
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

    def apply_catalog_annotations(
        self,
        navbar_brand_text: str = "ML Data Browser",
        head_title: str = "Catalog ML",
    ) -> None:
        """Apply catalog-level annotations including the navigation bar and display settings.

        This method configures the Chaise web interface for the catalog. Chaise is Deriva's
        web-based data browser that provides a user-friendly interface for exploring and
        managing catalog data. This method sets up annotations that control how Chaise
        displays and organizes the catalog.

        **Navigation Bar Structure**:
        The method creates a navigation bar with the following menus:
        - **User Info**: Links to Users, Groups, and RID Lease tables
        - **Deriva-ML**: Core ML tables (Workflow, Execution, Dataset, Dataset_Version, etc.)
        - **WWW**: Web content tables (Page, File)
        - **{Domain Schema}**: All domain-specific tables (excludes vocabularies and associations)
        - **Vocabulary**: All controlled vocabulary tables from both ML and domain schemas
        - **Assets**: All asset tables from both ML and domain schemas
        - **Features**: All feature tables with entries named "TableName:FeatureName"
        - **Catalog Registry**: Link to the ermrest registry
        - **Documentation**: Links to ML notebook instructions and Deriva-ML docs

        **Display Settings**:
        - Underscores in table/column names displayed as spaces
        - System columns (RID) shown in compact and entry views
        - Default table set to Dataset
        - Faceting and record deletion enabled
        - Export configurations available to all users

        **Bulk Upload Configuration**:
        Configures upload patterns for asset tables, enabling drag-and-drop file uploads
        through the Chaise interface.

        Call this after creating the domain schema and all tables to initialize the catalog's
        web interface. The navigation menus are dynamically built based on the current schema
        structure, automatically organizing tables into appropriate categories.

        Args:
            navbar_brand_text: Text displayed in the navigation bar brand area.
            head_title: Title displayed in the browser tab.

        Example:
            >>> ml = DerivaML('deriva.example.org', 'my_catalog')
            >>> # After creating domain schema and tables...
            >>> ml.apply_catalog_annotations()
            >>> # Or with custom branding:
            >>> ml.apply_catalog_annotations("My Project Browser", "My ML Project")
        """
        catalog_id = self.model.catalog.catalog_id
        ml_schema = self.ml_schema

        # Build domain schema menu items (one menu per domain schema)
        domain_schema_menus = []
        for domain_schema in sorted(self.domain_schemas):
            if domain_schema not in self.model.schemas:
                continue
            domain_schema_menus.append({
                "name": domain_schema,
                "children": [
                    {
                        "name": tname,
                        "url": f"/chaise/recordset/#{catalog_id}/{domain_schema}:{tname}",
                    }
                    for tname in self.model.schemas[domain_schema].tables
                    # Don't include controlled vocabularies, association tables, or feature tables.
                    if not (
                        self.model.is_vocabulary(tname)
                        or self.model.is_association(tname, pure=False, max_arity=3)
                    )
                ],
            })

        # Build vocabulary menu items (ML schema + all domain schemas)
        vocab_children = [{"name": f"{ml_schema} Vocabularies", "header": True}]
        vocab_children.extend([
            {
                "url": f"/chaise/recordset/#{catalog_id}/{ml_schema}:{tname}",
                "name": tname,
            }
            for tname in self.model.schemas[ml_schema].tables
            if self.model.is_vocabulary(tname)
        ])
        for domain_schema in sorted(self.domain_schemas):
            if domain_schema not in self.model.schemas:
                continue
            vocab_children.append({"name": f"{domain_schema} Vocabularies", "header": True})
            vocab_children.extend([
                {
                    "url": f"/chaise/recordset/#{catalog_id}/{domain_schema}:{tname}",
                    "name": tname,
                }
                for tname in self.model.schemas[domain_schema].tables
                if self.model.is_vocabulary(tname)
            ])

        # Build asset menu items (ML schema + all domain schemas)
        asset_children = [
            {
                "url": f"/chaise/recordset/#{catalog_id}/{ml_schema}:{tname}",
                "name": tname,
            }
            for tname in self.model.schemas[ml_schema].tables
            if self.model.is_asset(tname)
        ]
        for domain_schema in sorted(self.domain_schemas):
            if domain_schema not in self.model.schemas:
                continue
            asset_children.extend([
                {
                    "url": f"/chaise/recordset/#{catalog_id}/{domain_schema}:{tname}",
                    "name": tname,
                }
                for tname in self.model.schemas[domain_schema].tables
                if self.model.is_asset(tname)
            ])

        catalog_annotation = {
            deriva_tags.display: {"name_style": {"underline_space": True}},
            deriva_tags.chaise_config: {
                "headTitle": head_title,
                "navbarBrandText": navbar_brand_text,
                "systemColumnsDisplayEntry": ["RID"],
                "systemColumnsDisplayCompact": ["RID"],
                "defaultTable": {"table": "Dataset", "schema": "deriva-ml"},
                "deleteRecord": True,
                "showFaceting": True,
                "shareCiteAcls": True,
                "exportConfigsSubmenu": {"acls": {"show": ["*"], "enable": ["*"]}},
                "resolverImplicitCatalog": False,
                "navbarMenu": {
                    "newTab": False,
                    "children": [
                        {
                            "name": "User Info",
                            "children": [
                                {
                                    "url": f"/chaise/recordset/#{catalog_id}/public:ERMrest_Client",
                                    "name": "Users",
                                },
                                {
                                    "url": f"/chaise/recordset/#{catalog_id}/public:ERMrest_Group",
                                    "name": "Groups",
                                },
                                {
                                    "url": f"/chaise/recordset/#{catalog_id}/public:ERMrest_RID_Lease",
                                    "name": "ERMrest RID Lease",
                                },
                            ],
                        },
                        {  # All the primary tables in deriva-ml schema.
                            "name": "Deriva-ML",
                            "children": [
                                {
                                    "url": f"/chaise/recordset/#{catalog_id}/{ml_schema}:Workflow",
                                    "name": "Workflow",
                                },
                                {
                                    "url": f"/chaise/recordset/#{catalog_id}/{ml_schema}:Execution",
                                    "name": "Execution",
                                },
                                {
                                    "url": f"/chaise/recordset/#{catalog_id}/{ml_schema}:Execution_Metadata",
                                    "name": "Execution Metadata",
                                },
                                {
                                    "url": f"/chaise/recordset/#{catalog_id}/{ml_schema}:Execution_Asset",
                                    "name": "Execution Asset",
                                },
                                {
                                    "url": f"/chaise/recordset/#{catalog_id}/{ml_schema}:Dataset",
                                    "name": "Dataset",
                                },
                                {
                                    "url": f"/chaise/recordset/#{catalog_id}/{ml_schema}:Dataset_Version",
                                    "name": "Dataset Version",
                                },
                            ],
                        },
                        {  # WWW schema tables.
                            "name": "WWW",
                            "children": [
                                {
                                    "url": f"/chaise/recordset/#{catalog_id}/WWW:Page",
                                    "name": "Page",
                                },
                                {
                                    "url": f"/chaise/recordset/#{catalog_id}/WWW:File",
                                    "name": "File",
                                },
                            ],
                        },
                        *domain_schema_menus,  # One menu per domain schema
                        {  # Vocabulary menu with all controlled vocabularies.
                            "name": "Vocabulary",
                            "children": vocab_children,
                        },
                        {  # List of all asset tables.
                            "name": "Assets",
                            "children": asset_children,
                        },
                        {  # List of all feature tables in the catalog.
                            "name": "Features",
                            "children": [
                                {
                                    "url": f"/chaise/recordset/#{catalog_id}/{f.feature_table.schema.name}:{f.feature_table.name}",
                                    "name": f"{f.target_table.name}:{f.feature_name}",
                                }
                                for f in self.model.find_features()
                            ],
                        },
                        {
                            "url": "/chaise/recordset/#0/ermrest:registry@sort(RID)",
                            "name": "Catalog Registry",
                        },
                        {
                            "name": "Documentation",
                            "children": [
                                {
                                    "url": "https://github.com/informatics-isi-edu/deriva-ml/blob/main/docs/ml_workflow_instruction.md",
                                    "name": "ML Notebook Instruction",
                                },
                                {
                                    "url": "https://informatics-isi-edu.github.io/deriva-ml/",
                                    "name": "Deriva-ML Documentation",
                                },
                            ],
                        },
                    ],
                },
            },
            deriva_tags.bulk_upload: bulk_upload_configuration(model=self.model),
        }
        self.model.annotations.update(catalog_annotation)
        self.model.apply()

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
        # Use default schema or first domain schema for www tables
        schema = self.default_schema or (sorted(self.domain_schemas)[0] if self.domain_schemas else None)
        if schema is None:
            raise DerivaMLException("No domain schema available for adding pages")
        self.pathBuilder().www.tables[schema].insert([{"Title": title, "Content": content}])

    def create_vocabulary(
        self, vocab_name: str, comment: str = "", schema: str | None = None, update_navbar: bool = True
    ) -> Table:
        """Creates a controlled vocabulary table.

        A controlled vocabulary table maintains a list of standardized terms and their definitions. Each term can have
        synonyms and descriptions to ensure consistent terminology usage across the dataset.

        Args:
            vocab_name: Name for the new vocabulary table. Must be a valid SQL identifier.
            comment: Description of the vocabulary's purpose and usage. Defaults to empty string.
            schema: Schema name to create the table in. If None, uses domain_schema.
            update_navbar: If True (default), automatically updates the navigation bar to include
                the new vocabulary table. Set to False during batch table creation to avoid
                redundant updates, then call apply_catalog_annotations() once at the end.

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

            Create multiple vocabularies without updating navbar until the end:

                >>> ml.create_vocabulary("Species", update_navbar=False)
                >>> ml.create_vocabulary("Tissue_Type", update_navbar=False)
                >>> ml.apply_catalog_annotations()  # Update navbar once
        """
        # Use default schema if none specified
        schema = schema or self.model._require_default_schema()

        # Create and return vocabulary table with RID-based URI pattern
        try:
            vocab_table = self.model.schemas[schema].create_table(
                VocabularyTableDef(
                    name=vocab_name,
                    curie_template=f"{self.project_name}:{{RID}}",
                    comment=comment,
                )
            )
        except ValueError:
            raise DerivaMLException(f"Table {vocab_name} already exist")

        # Update navbar to include the new vocabulary table
        if update_navbar:
            self.apply_catalog_annotations()

        return vocab_table

    def create_table(self, table: TableDefinition, schema: str | None = None, update_navbar: bool = True) -> Table:
        """Creates a new table in the domain schema.

        Creates a table using the provided TableDefinition object, which specifies the table structure
        including columns, keys, and foreign key relationships. The table is created in the domain
        schema associated with this DerivaML instance.

        **Required Classes**:
        Import the following classes from deriva_ml to define tables:

        - ``TableDefinition``: Defines the complete table structure
        - ``ColumnDefinition``: Defines individual columns with types and constraints
        - ``KeyDefinition``: Defines unique key constraints (optional)
        - ``ForeignKeyDefinition``: Defines foreign key relationships to other tables (optional)
        - ``BuiltinTypes``: Enum of available column data types

        **Available Column Types** (BuiltinTypes enum):
        ``text``, ``int2``, ``int4``, ``int8``, ``float4``, ``float8``, ``boolean``,
        ``date``, ``timestamp``, ``timestamptz``, ``json``, ``jsonb``, ``markdown``,
        ``ermrest_uri``, ``ermrest_rid``, ``ermrest_rcb``, ``ermrest_rmb``,
        ``ermrest_rct``, ``ermrest_rmt``

        Args:
            table: A TableDefinition object containing the complete specification of the table to create.
            update_navbar: If True (default), automatically updates the navigation bar to include
                the new table. Set to False during batch table creation to avoid redundant updates,
                then call apply_catalog_annotations() once at the end.

        Returns:
            Table: The newly created ERMRest table object.

        Raises:
            DerivaMLException: If table creation fails or the definition is invalid.

        Examples:
            **Simple table with basic columns**:

                >>> from deriva_ml import TableDefinition, ColumnDefinition, BuiltinTypes
                >>>
                >>> table_def = TableDefinition(
                ...     name="Experiment",
                ...     column_defs=[
                ...         ColumnDefinition(name="Name", type=BuiltinTypes.text, nullok=False),
                ...         ColumnDefinition(name="Date", type=BuiltinTypes.date),
                ...         ColumnDefinition(name="Description", type=BuiltinTypes.markdown),
                ...         ColumnDefinition(name="Score", type=BuiltinTypes.float4),
                ...     ],
                ...     comment="Records of experimental runs"
                ... )
                >>> experiment_table = ml.create_table(table_def)

            **Table with foreign key to another table**:

                >>> from deriva_ml import (
                ...     TableDefinition, ColumnDefinition, ForeignKeyDefinition, BuiltinTypes
                ... )
                >>>
                >>> # Create a Sample table that references Subject
                >>> sample_def = TableDefinition(
                ...     name="Sample",
                ...     column_defs=[
                ...         ColumnDefinition(name="Name", type=BuiltinTypes.text, nullok=False),
                ...         ColumnDefinition(name="Subject", type=BuiltinTypes.text, nullok=False),
                ...         ColumnDefinition(name="Collection_Date", type=BuiltinTypes.date),
                ...     ],
                ...     fkey_defs=[
                ...         ForeignKeyDefinition(
                ...             colnames=["Subject"],
                ...             pk_sname=ml.default_schema,  # Schema of referenced table
                ...             pk_tname="Subject",          # Name of referenced table
                ...             pk_colnames=["RID"],         # Column(s) in referenced table
                ...             on_delete="CASCADE",         # Delete samples when subject deleted
                ...         )
                ...     ],
                ...     comment="Biological samples collected from subjects"
                ... )
                >>> sample_table = ml.create_table(sample_def)

            **Table with unique key constraint**:

                >>> from deriva_ml import (
                ...     TableDefinition, ColumnDefinition, KeyDefinition, BuiltinTypes
                ... )
                >>>
                >>> protocol_def = TableDefinition(
                ...     name="Protocol",
                ...     column_defs=[
                ...         ColumnDefinition(name="Name", type=BuiltinTypes.text, nullok=False),
                ...         ColumnDefinition(name="Version", type=BuiltinTypes.text, nullok=False),
                ...         ColumnDefinition(name="Description", type=BuiltinTypes.markdown),
                ...     ],
                ...     key_defs=[
                ...         KeyDefinition(
                ...             colnames=["Name", "Version"],
                ...             constraint_names=[["myschema", "Protocol_Name_Version_key"]],
                ...             comment="Each protocol name+version must be unique"
                ...         )
                ...     ],
                ...     comment="Experimental protocols with versioning"
                ... )
                >>> protocol_table = ml.create_table(protocol_def)

            **Batch creation without navbar updates**:

                >>> ml.create_table(table1_def, update_navbar=False)
                >>> ml.create_table(table2_def, update_navbar=False)
                >>> ml.create_table(table3_def, update_navbar=False)
                >>> ml.apply_catalog_annotations()  # Update navbar once at the end
        """
        # Use default schema if none specified
        schema = schema or self.model._require_default_schema()

        # Create table in domain schema using provided definition
        # Handle both TableDefinition (dataclass with to_dict) and plain dicts
        table_dict = table.to_dict() if hasattr(table, 'to_dict') else table
        new_table = self.model.schemas[schema].create_table(table_dict)

        # Update navbar to include the new table
        if update_navbar:
            self.apply_catalog_annotations()

        return new_table

    # =========================================================================
    # Cache and Directory Management
    # =========================================================================

    def clear_cache(self, older_than_days: int | None = None) -> dict[str, int]:
        """Clear the dataset cache directory.

        Removes cached dataset bags from the cache directory. Can optionally filter
        by age to only remove old cache entries.

        Args:
            older_than_days: If provided, only remove cache entries older than this
                many days. If None, removes all cache entries.

        Returns:
            dict with keys:
                - 'files_removed': Number of files removed
                - 'dirs_removed': Number of directories removed
                - 'bytes_freed': Total bytes freed
                - 'errors': Number of removal errors

        Example:
            >>> ml = DerivaML('deriva.example.org', 'my_catalog')
            >>> # Clear all cache
            >>> result = ml.clear_cache()
            >>> print(f"Freed {result['bytes_freed'] / 1e6:.1f} MB")
            >>>
            >>> # Clear cache older than 7 days
            >>> result = ml.clear_cache(older_than_days=7)
        """
        import shutil
        import time

        stats = {'files_removed': 0, 'dirs_removed': 0, 'bytes_freed': 0, 'errors': 0}

        if not self.cache_dir.exists():
            return stats

        cutoff_time = None
        if older_than_days is not None:
            cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)

        try:
            for entry in self.cache_dir.iterdir():
                try:
                    # Check age if filtering
                    if cutoff_time is not None:
                        entry_mtime = entry.stat().st_mtime
                        if entry_mtime > cutoff_time:
                            continue  # Skip recent entries

                    # Calculate size before removal
                    if entry.is_dir():
                        entry_size = sum(f.stat().st_size for f in entry.rglob('*') if f.is_file())
                        shutil.rmtree(entry)
                        stats['dirs_removed'] += 1
                    else:
                        entry_size = entry.stat().st_size
                        entry.unlink()
                        stats['files_removed'] += 1

                    stats['bytes_freed'] += entry_size
                except (OSError, PermissionError) as e:
                    self._logger.warning(f"Failed to remove cache entry {entry}: {e}")
                    stats['errors'] += 1

        except OSError as e:
            self._logger.error(f"Failed to iterate cache directory: {e}")
            stats['errors'] += 1

        return stats

    def get_cache_size(self) -> dict[str, int | float]:
        """Get the current size of the cache directory.

        Returns:
            dict with keys:
                - 'total_bytes': Total size in bytes
                - 'total_mb': Total size in megabytes
                - 'file_count': Number of files
                - 'dir_count': Number of directories

        Example:
            >>> ml = DerivaML('deriva.example.org', 'my_catalog')
            >>> size = ml.get_cache_size()
            >>> print(f"Cache size: {size['total_mb']:.1f} MB ({size['file_count']} files)")
        """
        stats = {'total_bytes': 0, 'total_mb': 0.0, 'file_count': 0, 'dir_count': 0}

        if not self.cache_dir.exists():
            return stats

        for entry in self.cache_dir.rglob('*'):
            if entry.is_file():
                stats['total_bytes'] += entry.stat().st_size
                stats['file_count'] += 1
            elif entry.is_dir():
                stats['dir_count'] += 1

        stats['total_mb'] = stats['total_bytes'] / (1024 * 1024)
        return stats

    def list_execution_dirs(self) -> list[dict[str, any]]:
        """List execution working directories.

        Returns information about each execution directory in the working directory,
        useful for identifying orphaned or incomplete execution outputs.

        Returns:
            List of dicts, each containing:
                - 'execution_rid': The execution RID (directory name)
                - 'path': Full path to the directory
                - 'size_bytes': Total size in bytes
                - 'size_mb': Total size in megabytes
                - 'modified': Last modification time (datetime)
                - 'file_count': Number of files

        Example:
            >>> ml = DerivaML('deriva.example.org', 'my_catalog')
            >>> dirs = ml.list_execution_dirs()
            >>> for d in dirs:
            ...     print(f"{d['execution_rid']}: {d['size_mb']:.1f} MB")
        """
        from datetime import datetime
        from deriva_ml.dataset.upload import upload_root

        results = []
        exec_root = upload_root(self.working_dir) / "execution"

        if not exec_root.exists():
            return results

        for entry in exec_root.iterdir():
            if entry.is_dir():
                size_bytes = sum(f.stat().st_size for f in entry.rglob('*') if f.is_file())
                file_count = sum(1 for f in entry.rglob('*') if f.is_file())
                mtime = datetime.fromtimestamp(entry.stat().st_mtime)

                results.append({
                    'execution_rid': entry.name,
                    'path': str(entry),
                    'size_bytes': size_bytes,
                    'size_mb': size_bytes / (1024 * 1024),
                    'modified': mtime,
                    'file_count': file_count,
                })

        return sorted(results, key=lambda x: x['modified'], reverse=True)

    def clean_execution_dirs(
        self,
        older_than_days: int | None = None,
        exclude_rids: list[str] | None = None,
    ) -> dict[str, int]:
        """Clean up execution working directories.

        Removes execution output directories from the local working directory.
        Use this to free up disk space from completed or orphaned executions.

        Args:
            older_than_days: If provided, only remove directories older than this
                many days. If None, removes all execution directories (except excluded).
            exclude_rids: List of execution RIDs to preserve (never remove).

        Returns:
            dict with keys:
                - 'dirs_removed': Number of directories removed
                - 'bytes_freed': Total bytes freed
                - 'errors': Number of removal errors

        Example:
            >>> ml = DerivaML('deriva.example.org', 'my_catalog')
            >>> # Clean all execution dirs older than 30 days
            >>> result = ml.clean_execution_dirs(older_than_days=30)
            >>> print(f"Freed {result['bytes_freed'] / 1e9:.2f} GB")
            >>>
            >>> # Clean all except specific executions
            >>> result = ml.clean_execution_dirs(exclude_rids=['1-ABC', '1-DEF'])
        """
        import shutil
        import time
        from deriva_ml.dataset.upload import upload_root

        stats = {'dirs_removed': 0, 'bytes_freed': 0, 'errors': 0}
        exclude_rids = set(exclude_rids or [])

        exec_root = upload_root(self.working_dir) / "execution"
        if not exec_root.exists():
            return stats

        cutoff_time = None
        if older_than_days is not None:
            cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)

        for entry in exec_root.iterdir():
            if not entry.is_dir():
                continue

            # Skip excluded RIDs
            if entry.name in exclude_rids:
                continue

            try:
                # Check age if filtering
                if cutoff_time is not None:
                    entry_mtime = entry.stat().st_mtime
                    if entry_mtime > cutoff_time:
                        continue

                # Calculate size before removal
                entry_size = sum(f.stat().st_size for f in entry.rglob('*') if f.is_file())
                shutil.rmtree(entry)
                stats['dirs_removed'] += 1
                stats['bytes_freed'] += entry_size

            except (OSError, PermissionError) as e:
                self._logger.warning(f"Failed to remove execution dir {entry}: {e}")
                stats['errors'] += 1

        return stats

    def get_storage_summary(self) -> dict[str, any]:
        """Get a summary of local storage usage.

        Returns:
            dict with keys:
                - 'working_dir': Path to working directory
                - 'cache_dir': Path to cache directory
                - 'cache_size_mb': Cache size in MB
                - 'cache_file_count': Number of files in cache
                - 'execution_dir_count': Number of execution directories
                - 'execution_size_mb': Total size of execution directories in MB
                - 'total_size_mb': Combined size in MB

        Example:
            >>> ml = DerivaML('deriva.example.org', 'my_catalog')
            >>> summary = ml.get_storage_summary()
            >>> print(f"Total storage: {summary['total_size_mb']:.1f} MB")
            >>> print(f"  Cache: {summary['cache_size_mb']:.1f} MB")
            >>> print(f"  Executions: {summary['execution_size_mb']:.1f} MB")
        """
        cache_stats = self.get_cache_size()
        exec_dirs = self.list_execution_dirs()

        exec_size_mb = sum(d['size_mb'] for d in exec_dirs)

        return {
            'working_dir': str(self.working_dir),
            'cache_dir': str(self.cache_dir),
            'cache_size_mb': cache_stats['total_mb'],
            'cache_file_count': cache_stats['file_count'],
            'execution_dir_count': len(exec_dirs),
            'execution_size_mb': exec_size_mb,
            'total_size_mb': cache_stats['total_mb'] + exec_size_mb,
        }

    # =========================================================================
    # Schema Validation
    # =========================================================================

    def validate_schema(self, strict: bool = False) -> "SchemaValidationReport":
        """Validate that the catalog's ML schema matches the expected structure.

        This method inspects the catalog schema and verifies that it contains all
        the required tables, columns, vocabulary terms, and relationships that are
        created by the ML schema initialization routines in create_schema.py.

        The validation checks:
        - All required ML tables exist (Dataset, Execution, Workflow, etc.)
        - All required columns exist with correct types
        - All required vocabulary tables exist (Asset_Type, Dataset_Type, etc.)
        - All required vocabulary terms are initialized
        - All association tables exist for relationships

        In strict mode, the validator also reports errors for:
        - Extra tables not in the expected schema
        - Extra columns not in the expected table definitions

        Args:
            strict: If True, extra tables and columns are reported as errors.
                   If False (default), they are reported as informational items.
                   Use strict=True to verify a clean ML catalog matches exactly.
                   Use strict=False to validate a catalog that may have domain extensions.

        Returns:
            SchemaValidationReport with validation results. Key attributes:
                - is_valid: True if no errors were found
                - errors: List of error-level issues
                - warnings: List of warning-level issues
                - info: List of informational items
                - to_text(): Human-readable report
                - to_dict(): JSON-serializable dictionary

        Example:
            >>> ml = DerivaML('localhost', 'my_catalog')
            >>> report = ml.validate_schema(strict=False)
            >>> if report.is_valid:
            ...     print("Schema is valid!")
            ... else:
            ...     print(report.to_text())

            >>> # Strict validation for a fresh ML catalog
            >>> report = ml.validate_schema(strict=True)
            >>> print(f"Found {len(report.errors)} errors, {len(report.warnings)} warnings")

            >>> # Get report as dictionary for JSON/logging
            >>> import json
            >>> print(json.dumps(report.to_dict(), indent=2))

        Note:
            This method validates the ML schema (typically 'deriva-ml'), not the
            domain schema. Domain-specific tables and columns are not checked
            unless they are part of the ML schema itself.

        See Also:
            - deriva_ml.schema.validation.SchemaValidationReport
            - deriva_ml.schema.validation.validate_ml_schema
        """
        from deriva_ml.schema.validation import SchemaValidationReport, validate_ml_schema
        return validate_ml_schema(self, strict=strict)

    # Methods moved to mixins:
    # - create_asset, list_assets -> AssetMixin
    # - create_feature, feature_record_class, delete_feature, lookup_feature, list_feature_values -> FeatureMixin
    # - find_datasets, create_dataset, lookup_dataset, delete_dataset, list_dataset_element_types,
    #   add_dataset_element_type, download_dataset_bag -> DatasetMixin
    # - _update_status, create_execution, restore_execution -> ExecutionMixin
    # - add_files, list_files, _bootstrap_versions, _synchronize_dataset_versions, _set_version_snapshot -> FileMixin

