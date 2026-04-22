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

from deriva_ml.core.catalog_stub import CatalogStub
from deriva_ml.core.config import DerivaMLConfig
from deriva_ml.core.connection_mode import ConnectionMode
from deriva_ml.core.definitions import ML_SCHEMA, RID, TableDefinition, VocabularyTableDef
from deriva_ml.core.exceptions import (
    DerivaMLConfigurationError,
    DerivaMLException,
    DerivaMLReadOnlyError,
    DerivaMLSchemaPinned,
    DerivaMLSchemaRefreshBlocked,
)
from deriva_ml.core.logging_config import apply_logger_overrides, configure_logging
from deriva_ml.core.mixins import (
    AnnotationMixin,
    AssetMixin,
    DatasetMixin,
    ExecutionMixin,
    FeatureMixin,
    FileMixin,
    PathBuilderMixin,
    RidResolutionMixin,
    VocabularyMixin,
    WorkflowMixin,
)
from deriva_ml.core.schema_cache import PinStatus, SchemaCache
from deriva_ml.dataset.upload import bulk_upload_configuration
from deriva_ml.interfaces import DerivaMLCatalog

if TYPE_CHECKING:
    from deriva_ml.catalog.clone import CatalogProvenance
    from deriva_ml.core.schema_diff import SchemaDiff
    from deriva_ml.execution.execution import Execution
    from deriva_ml.model.catalog import DerivaModel
    from deriva_ml.schema.validation import SchemaValidationReport

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

    @classmethod
    def from_context(cls, path: Path | str | None = None) -> Self:
        """Create a DerivaML instance from a .deriva-context.json file.

        Searches for .deriva-context.json starting from ``path`` (default: cwd),
        walking up parent directories. This enables scripts generated by Claude
        to connect to the same catalog without hardcoding connection details.

        The context file is written by the MCP server's ``connect_catalog`` tool
        and contains hostname, catalog_id, and default_schema.

        Args:
            path: Starting directory to search for the context file.
                Defaults to the current working directory.

        Returns:
            A new DerivaML instance configured from the context file.

        Raises:
            FileNotFoundError: If no .deriva-context.json is found.

        Example::

            # In a script generated by Claude:
            from deriva_ml import DerivaML
            ml = DerivaML.from_context()
            subjects = ml.cache_table("Subject")
        """
        import json

        start = Path(path) if path else Path.cwd()
        context_file = _find_context_file(start)
        with open(context_file) as f:
            ctx = json.load(f)

        kwargs: dict[str, Any] = {
            "hostname": ctx["hostname"],
            "catalog_id": ctx["catalog_id"],
        }
        if ctx.get("default_schema"):
            kwargs["default_schema"] = ctx["default_schema"]
        if ctx.get("working_dir"):
            kwargs["working_dir"] = ctx["working_dir"]

        return cls(**kwargs)

    def __init__(
        self,
        hostname: str,
        catalog_id: str | int,
        domain_schemas: str | set[str] | None = None,
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
        mode: ConnectionMode | str = ConnectionMode.online,
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
            mode: Connection mode for this instance. ``ConnectionMode.online`` (default)
                sends writes to the catalog eagerly; ``ConnectionMode.offline`` stages
                writes into local SQLite for later upload. Accepts the string
                literals ``"online"`` or ``"offline"``; any other value raises
                ``ValueError``. See spec §2.1.
        """
        # Store connection mode (see spec §2.1).
        # Done before catalog connection so subclasses/mixins can read
        # ``self._mode`` during their own setup if needed.
        # ``ConnectionMode(x)`` is idempotent on enum members and coerces
        # strings ("online"/"offline") uniformly; unknown strings raise ValueError.
        self._mode = ConnectionMode(mode)

        # Get or use provided credentials for server access.
        # get_credential() reads ~/.deriva/credential.json; no network.
        self.credential = credential or get_credential(hostname)

        # Set up working and cache directories. Done BEFORE catalog/
        # schema setup so SchemaCache can be constructed for either
        # mode branch below.
        # If working_dir is already provided (e.g. from DerivaMLConfig.instantiate()),
        # use it directly; otherwise compute the default path.
        if working_dir is not None:
            self.working_dir = Path(working_dir).absolute()
        else:
            self.working_dir = DerivaMLConfig.compute_workdir(None, catalog_id, hostname)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.hydra_runtime_output_dir = hydra_runtime_output_dir

        self.cache_dir = Path(cache_dir) if cache_dir else self.working_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Mode-branched init: online connects to the catalog and
        # verifies (or populates) the schema cache; offline reads
        # the cache and skips all network calls.
        cache = SchemaCache(self.working_dir)
        if self._mode is ConnectionMode.online:
            self._init_online(
                hostname=hostname,
                catalog_id=catalog_id,
                check_auth=check_auth,
                cache=cache,
                ml_schema=ml_schema,
                domain_schemas=domain_schemas,
                default_schema=default_schema,
            )
        else:
            self._init_offline(
                hostname=hostname,
                catalog_id=catalog_id,
                cache=cache,
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
        self.clean_execution_dir = clean_execution_dir

        # Reconcile any pending_rows stuck in 'leasing' from a prior
        # crash. Workspace-wide sweep; per-execution reconciliation
        # runs additionally on resume_execution (Group D).
        if self._mode is ConnectionMode.online:
            from deriva_ml.execution.lease_orchestrator import reconcile_pending_leases

            try:
                reconcile_pending_leases(
                    store=self.workspace.execution_state_store(),
                    catalog=self.catalog,
                    execution_rid=None,
                )
            except Exception as exc:
                # Best-effort. If reconciliation itself fails, log and
                # move on — the user's operation can still proceed;
                # the next acquire_leases call will retry.
                logging.getLogger("deriva_ml").warning(
                    "startup lease reconciliation failed (%s); continuing", exc,
                )

    def __del__(self) -> None:
        """Cleanup method to handle incomplete executions.

        Best-effort abort on DerivaML shutdown. The previous implementation
        used the legacy `Status` enum; the new `ExecutionStatus` lifecycle
        separates Stopped/Uploaded/Aborted/Failed. Here we only attempt to
        abort if the execution hasn't already reached a terminal state —
        InvalidTransitionError from the state machine covers the rest.
        """
        # Inline import to avoid a circular (core.base ↔ execution.state_store) import.
        try:
            from deriva_ml.execution.state_store import ExecutionStatus

            if self._execution and self._execution.status is not ExecutionStatus.Aborted:
                self._execution.update_status(ExecutionStatus.Aborted, error="Execution Aborted")
        except Exception:
            # Any failure here (catalog unreachable, InvalidTransition, etc.)
            # is swallowed — __del__ must not raise.
            pass

    def _init_online(
        self,
        *,
        hostname: str,
        catalog_id: str | int,
        check_auth: bool,
        cache: "SchemaCache",
        ml_schema: str,
        domain_schemas: "str | set[str] | None",
        default_schema: "str | None",
    ) -> None:
        """Online init: connect to server, resolve schema via cache
        or fresh fetch with a drift warning."""
        from deriva_ml.model.catalog import DerivaModel

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

        # GET / returns {snaptime: ..., ...} — cheap way to learn
        # the catalog's current snapshot id.
        live_snapshot_id = self.catalog.get("/").json()["snaptime"]

        if cache.exists():
            cached = cache.load()
            if cached["snapshot_id"] != live_snapshot_id:
                logging.getLogger("deriva_ml").warning(
                    "schema cache is at snapshot %s; live catalog is at %s. "
                    "Using cached schema. Call ml.refresh_schema() to update.",
                    cached["snapshot_id"], live_snapshot_id,
                )
            self.model = DerivaModel.from_cached(
                cached["schema"],
                catalog=self.catalog,
                ml_schema=ml_schema,
                domain_schemas=domain_schemas,
                default_schema=default_schema,
            )
        else:
            # First-time online init — fetch live schema and populate cache.
            live_schema = self.catalog.get("/schema").json()
            cache.write(
                snapshot_id=live_snapshot_id,
                hostname=hostname,
                catalog_id=str(catalog_id),
                ml_schema=ml_schema,
                schema=live_schema,
            )
            self.model = DerivaModel.from_cached(
                live_schema,
                catalog=self.catalog,
                ml_schema=ml_schema,
                domain_schemas=domain_schemas,
                default_schema=default_schema,
            )

    def _init_offline(
        self,
        *,
        hostname: str,
        catalog_id: str | int,
        cache: "SchemaCache",
        ml_schema: str,
        domain_schemas: "str | set[str] | None",
        default_schema: "str | None",
    ) -> None:
        """Offline init: read cache, skip all network. Raises if the
        cache is missing or belongs to a different (host, catalog)."""
        from deriva_ml.model.catalog import DerivaModel

        if not cache.exists():
            raise DerivaMLConfigurationError(
                f"offline mode requires a cached schema at {cache._path}; "
                f"run online once first (with the same working_dir) to populate the cache."
            )
        cached = cache.load()
        if cached["hostname"] != hostname or cached["catalog_id"] != str(catalog_id):
            raise DerivaMLConfigurationError(
                f"cached schema at {cache._path} is for "
                f"{cached['hostname']}/{cached['catalog_id']}, "
                f"but __init__ was called with {hostname}/{catalog_id}. "
                f"Use the matching working_dir or run online to refresh."
            )
        self.catalog = CatalogStub()
        self.model = DerivaModel.from_cached(
            cached["schema"],
            catalog=self.catalog,
            ml_schema=ml_schema,
            domain_schemas=domain_schemas,
            default_schema=default_schema,
        )

    def refresh_schema(self, *, force: bool = False) -> None:
        """Fetch the current catalog schema and overwrite the workspace cache.

        Online mode only. Refuses when the workspace has pending
        rows unless ``force=True`` is passed; a forced refresh may
        leave staged rows whose metadata references columns or types
        no longer in the new schema, causing catalog-insert failures
        on the next upload.

        Args:
            force: If True, refresh even when the workspace has
                pending rows (status staged/leasing/leased/uploading/
                failed). Default False refuses in that case with
                :class:`DerivaMLSchemaRefreshBlocked`.

        Raises:
            DerivaMLReadOnlyError: If called in offline mode.
            DerivaMLSchemaRefreshBlocked: If ``force=False`` and the
                workspace has pending rows.
        """
        from deriva_ml.model.catalog import DerivaModel

        if self._mode is not ConnectionMode.online:
            raise DerivaMLReadOnlyError(
                "refresh_schema requires online mode"
            )
        store = self.workspace.execution_state_store()
        count = store.count_pending_rows()
        if count > 0 and not force:
            raise DerivaMLSchemaRefreshBlocked(
                f"refresh_schema requires a drained workspace; "
                f"{count} pending rows. Run ml.upload_pending() first, "
                f"or call refresh_schema(force=True) to discard local "
                f"state (staged rows may become inconsistent with the "
                f"new schema)."
            )
        live_snapshot_id = self.catalog.get("/").json()["snaptime"]
        live_schema = self.catalog.get("/schema").json()
        cache = SchemaCache(self.working_dir)
        old_snapshot_id = cache.snapshot_id()
        cache.write(
            snapshot_id=live_snapshot_id,
            hostname=self.host_name,
            catalog_id=str(self.catalog_id),
            ml_schema=self.model.ml_schema,
            schema=live_schema,
        )
        # Reload the in-memory model so this session sees the new schema.
        self.model = DerivaModel.from_cached(
            live_schema,
            catalog=self.catalog,
            ml_schema=self.model.ml_schema,
            domain_schemas=self.model.domain_schemas,
            default_schema=self.model.default_schema,
        )
        logging.getLogger("deriva_ml").info(
            "schema cache refreshed from %s to %s",
            old_snapshot_id, live_snapshot_id,
        )

    def pin_schema(self, reason: str | None = None) -> "SchemaDiff | None":
        """Freeze the local schema cache at its current snapshot.

        While pinned, :meth:`refresh_schema` refuses to update the
        cache (even with ``force=True``). Call :meth:`unpin_schema`
        to clear the pin.

        Online mode additionally checks for structural drift: if the
        live catalog has moved on and its ``/schema`` payload differs
        from the cached one (columns, tables, foreign keys, etc.),
        a :class:`SchemaDiff` describing the drift is returned, and
        a WARNING is logged. The pin is still persisted.

        Offline mode always returns ``None`` — the cache is pinned,
        but no live comparison is possible.

        Args:
            reason: Free-text explanation stored alongside the pin.
                Useful for reporting (``pin_status().pin_reason``).

        Returns:
            A :class:`SchemaDiff` when the pin is applied online and
            the live catalog's schema differs structurally from the
            cache. ``None`` otherwise (offline, no drift, or snapshot
            bumped without schema change).

        Raises:
            FileNotFoundError: If the workspace has no cache yet.
                Run an online ``DerivaML.__init__`` or
                :meth:`refresh_schema` first.
        """
        from deriva_ml.core.schema_diff import compute_diff, SchemaDiff

        cache = SchemaCache(self.working_dir)
        drift: SchemaDiff | None = None
        if self._mode is ConnectionMode.online:
            live_snapshot_id = self.catalog.get("/").json()["snaptime"]
            cached_payload = cache.load()
            if cached_payload["snapshot_id"] != live_snapshot_id:
                live_schema = self.catalog.get("/schema").json()
                diff = compute_diff(cached_payload["schema"], live_schema)
                if not diff.is_empty():
                    logging.getLogger("deriva_ml").warning(
                        "pin_schema: cache at %s, live at %s; "
                        "structural drift detected (see returned SchemaDiff)",
                        cached_payload["snapshot_id"], live_snapshot_id,
                    )
                    drift = diff
        cache.pin(reason=reason)
        return drift

    def unpin_schema(self) -> None:
        """Clear the schema-cache pin. No-op if not pinned.

        Works in any mode. After unpinning, :meth:`refresh_schema`
        is allowed again (subject to the pending-rows guard).

        Raises:
            FileNotFoundError: If the workspace has no cache file.
        """
        SchemaCache(self.working_dir).unpin()

    def pin_status(self) -> "PinStatus":
        """Return the current pin state of the local schema cache.

        Works in any mode.

        Returns:
            A :class:`PinStatus` snapshot: ``pinned`` flag, UTC
            ``pinned_at`` timestamp (or None), caller-supplied
            ``pin_reason`` (or None), and the cache's current
            ``pinned_snapshot_id``.

        Raises:
            FileNotFoundError: If the workspace has no cache file.
        """
        return SchemaCache(self.working_dir).pin_status()

    def diff_schema(self) -> "SchemaDiff":
        """Return the structural diff between the cached and live schemas.

        Online mode only. Fetches the live catalog's ``/schema``
        payload, compares it against the cached copy with
        :func:`~deriva_ml.core.schema_diff.compute_diff`, and returns
        the result. The returned :class:`SchemaDiff` may be empty
        (no drift) — callers should check ``diff.is_empty()`` rather
        than truthiness.

        Unlike :meth:`pin_schema`, this method never modifies the
        cache and never logs a warning; it is a pure inspection
        operation.

        Returns:
            A :class:`SchemaDiff`, possibly empty.

        Raises:
            DerivaMLReadOnlyError: If called in offline mode.
            FileNotFoundError: If the workspace has no cache file.
        """
        from deriva_ml.core.schema_diff import compute_diff

        if self._mode is not ConnectionMode.online:
            raise DerivaMLReadOnlyError("diff_schema requires online mode")
        cache = SchemaCache(self.working_dir)
        cached_payload = cache.load()
        live_schema = self.catalog.get("/schema").json()
        return compute_diff(cached_payload["schema"], live_schema)

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
        """Check whether this DerivaML instance is connected to a catalog snapshot.

        Returns:
            True if the underlying catalog has a snapshot timestamp, False otherwise.
        """
        return hasattr(self.catalog, "_snaptime")

    def catalog_snapshot(self, version_snapshot: str) -> Self:
        """Return a new DerivaML instance connected to a specific catalog snapshot.

        Catalog snapshots provide a read-only, point-in-time view of the catalog.
        The snapshot identifier is typically obtained from a dataset version record.

        Args:
            version_snapshot: Snapshot identifier string (e.g., ``"2T-SXEH-JH4A"``),
                usually the ``snapshot`` field from a :class:`DatasetHistory` entry.

        Returns:
            A new DerivaML instance connected to the specified catalog snapshot.
        """
        return DerivaML(
            self.host_name,
            version_snapshot,
            logging_level=self._logging_level,
            deriva_logging_level=self._deriva_logging_level,
        )

    @property
    def mode(self) -> ConnectionMode:
        """Current connection mode.

        Returns:
            The ConnectionMode this DerivaML instance was constructed
            with. Drives whether writes go live to the catalog (online)
            or stage in SQLite for later upload (offline). See spec §2.1.

        Example:
            >>> ml.mode is ConnectionMode.online
            True
        """
        return self._mode

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

    @property
    def workspace(self) -> "Workspace":
        """Per-catalog Workspace for local caching, denormalization, and asset manifests.

        Backed by ``Workspace`` under ``{working_dir}/catalogs/{host}__{cat}/
        working.db``. Shared across invocations of scripts that use the same
        working directory.

        Example::

            # Cache a full table
            df = ml.cache_table("Subject")

            # Check what's cached
            ml.workspace.list_cached_results()
        """
        from deriva_ml.local_db.workspace import Workspace

        if not hasattr(self, "_workspace") or self._workspace is None:
            self._workspace = Workspace(
                working_dir=self.working_dir,
                hostname=self.host_name,
                catalog_id=self.catalog_id,
            )
            # Import any legacy JSON manifests
            try:
                n = self._workspace.import_legacy_manifests()
                if n:
                    import logging

                    logging.getLogger("deriva_ml").info(
                        "Migrated %d legacy asset manifests into workspace",
                        n,
                    )
            except Exception as exc:
                import logging

                logging.getLogger("deriva_ml").warning(
                    "Legacy manifest migration failed: %s",
                    exc,
                )
            # Build the local schema so the ORM is available. In online
            # mode, refresh the catalog model first so the ORM reflects
            # the actual catalog state at the time workspace is lazily
            # first accessed — the model object captured at
            # DerivaML.__init__ may have been constructed before later
            # catalog mutations (e.g. the test harness calling
            # ``add_dataset_element_type`` to create association tables).
            # Without this refresh, the local schema misses tables that
            # already exist in the catalog.
            #
            # In offline mode, the schema cache IS the authoritative
            # model and refresh_model() would attempt a network call
            # that CatalogStub refuses. Skip it — the cached model was
            # loaded at __init__ time and is what offline callers want.
            if self._mode is ConnectionMode.online:
                self.model.refresh_model()
            self._workspace.build_local_schema(
                model=self.model.model,  # the ERMrest Model object
                schemas=[self.ml_schema, *self.domain_schemas],
            )
        return self._workspace

    @property
    def working_data(self):
        """Deprecated: use ``workspace`` instead."""
        import warnings

        warnings.warn(
            "DerivaML.working_data is deprecated; use DerivaML.workspace instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.workspace

    def cache_table(self, table_name: str, force: bool = False) -> "pd.DataFrame":
        """Fetch a table from the catalog and cache locally as SQLite.

        On first call, fetches all rows from the catalog and stores in the
        working data cache. Subsequent calls return the cached data without
        contacting the catalog. Use ``force=True`` to re-fetch.

        Args:
            table_name: Name of the table to fetch (e.g., "Subject", "Image").
            force: If True, re-fetch even if already cached.

        Returns:
            DataFrame with the table contents.

        Example::

            subjects = ml.cache_table("Subject")
            print(f"{len(subjects)} subjects")

            # Second call returns cached data instantly
            subjects = ml.cache_table("Subject")
        """
        result = self.workspace.cached_table_read(
            table=table_name,
            source="catalog",
            refresh=force,
        )
        return result.to_dataframe()

    def cache_features(
        self,
        table_name: str,
        feature_name: str,
        force: bool = False,
        **kwargs,
    ) -> "pd.DataFrame":
        """Fetch feature values from the catalog and cache locally.

        On first call, fetches all feature values and stores in the working
        data cache. Subsequent calls return cached data.

        Args:
            table_name: Table the feature is attached to (e.g., "Image").
            feature_name: Name of the feature (e.g., "Classification").
            force: If True, re-fetch even if already cached.
            **kwargs: Additional arguments passed to ``fetch_table_features``
                (e.g., ``selector``, ``workflow``, ``execution``).

        Returns:
            DataFrame with feature value records.

        Example::

            labels = ml.cache_features("Image", "Classification")
            print(labels["Diagnosis_Type"].value_counts())
        """
        import time

        import pandas as pd

        from deriva_ml.local_db.result_cache import CachedResultMeta, ResultCache

        rc = self.workspace._get_result_cache()
        key = ResultCache.cache_key("features", table=table_name, feature=feature_name)

        if not force and rc.has(key):
            cached = rc.get(key)
            if cached is not None:
                return cached.to_dataframe()

        features = self.fetch_table_features(table_name, feature_name=feature_name, **kwargs)
        records = [r.model_dump(mode="json") for r in features.get(feature_name, [])]
        df = pd.DataFrame(records)
        if not df.empty:
            columns = list(df.columns)
            rows = df.to_dict(orient="records")
            meta = CachedResultMeta(
                cache_key=key,
                source="catalog",
                tool_name="features",
                params={"table": table_name, "feature": feature_name},
                columns=columns,
                row_count=len(rows),
                created_at=time.time(),
            )
            rc.store(key, columns, rows, meta)
        return df

    @staticmethod
    def globus_login(host: str) -> None:
        """Authenticate with Globus to obtain credentials for a Deriva server.

        Initiates a Globus Native Login flow to obtain OAuth2 tokens required
        by the Deriva server.  The flow uses a device-code grant (no browser
        or local server), and stores refresh tokens so that subsequent calls
        can re-authenticate silently.  The BDBag keychain is also updated so
        that bag downloads can use the same credentials.

        If the user is already logged in for the given host, a message is
        printed and no further action is taken.

        Args:
            host: Hostname of the Deriva server to authenticate with
                (e.g., ``"www.eye-ai.org"``).

        Example:
            >>> DerivaML.globus_login('www.eye-ai.org')
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

    @property
    def catalog_provenance(self) -> "CatalogProvenance | None":
        """Get the provenance information for this catalog.

        Returns provenance information if the catalog has it set. This includes
        information about how the catalog was created (clone, create, schema),
        who created it, when, and any workflow information.

        For cloned catalogs, additional details about the clone operation are
        available in the `clone_details` attribute.

        Returns:
            CatalogProvenance if available, None otherwise.

        Example:
            >>> ml = DerivaML('localhost', '45')
            >>> prov = ml.catalog_provenance
            >>> if prov:
            ...     print(f"Created: {prov.created_at} by {prov.created_by}")
            ...     print(f"Method: {prov.creation_method.value}")
            ...     if prov.is_clone:
            ...         print(f"Cloned from: {prov.clone_details.source_hostname}")
        """
        from deriva_ml.catalog.clone import get_catalog_provenance

        return get_catalog_provenance(self.catalog)

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
            domain_schema_menus.append(
                {
                    "name": domain_schema,
                    "children": [
                        {
                            "name": tname,
                            "url": f"/chaise/recordset/#{catalog_id}/{domain_schema}:{tname}",
                        }
                        for tname in self.model.schemas[domain_schema].tables
                        # Don't include controlled vocabularies, association tables, or feature tables.
                        if not (
                            self.model.is_vocabulary(tname) or self.model.is_association(tname, pure=False, max_arity=3)
                        )
                    ],
                }
            )

        # Build vocabulary menu items (ML schema + all domain schemas)
        vocab_children = [{"name": f"{ml_schema} Vocabularies", "header": True}]
        vocab_children.extend(
            [
                {
                    "url": f"/chaise/recordset/#{catalog_id}/{ml_schema}:{tname}",
                    "name": tname,
                }
                for tname in self.model.schemas[ml_schema].tables
                if self.model.is_vocabulary(tname)
            ]
        )
        for domain_schema in sorted(self.domain_schemas):
            if domain_schema not in self.model.schemas:
                continue
            vocab_children.append({"name": f"{domain_schema} Vocabularies", "header": True})
            vocab_children.extend(
                [
                    {
                        "url": f"/chaise/recordset/#{catalog_id}/{domain_schema}:{tname}",
                        "name": tname,
                    }
                    for tname in self.model.schemas[domain_schema].tables
                    if self.model.is_vocabulary(tname)
                ]
            )

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
            asset_children.extend(
                [
                    {
                        "url": f"/chaise/recordset/#{catalog_id}/{domain_schema}:{tname}",
                        "name": tname,
                    }
                    for tname in self.model.schemas[domain_schema].tables
                    if self.model.is_asset(tname)
                ]
            )

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
        table_dict = table.to_dict() if hasattr(table, "to_dict") else table
        new_table = self.model.schemas[schema].create_table(table_dict)

        # Update navbar to include the new table
        if update_navbar:
            self.apply_catalog_annotations()

        return new_table

    def define_association(
        self,
        associates: list,
        metadata: list | None = None,
        table_name: str | None = None,
        comment: str | None = None,
        **kwargs,
    ) -> dict:
        """Build an association table definition with vocab-aware key selection.

        Creates a table definition that links two or more tables via an association
        (many-to-many) table. Non-vocabulary tables automatically use RID as the
        foreign key target, while vocabulary tables use their Name key.

        Use with ``create_table()`` to create the association table in the catalog.

        Args:
            associates: Tables to associate. Each item can be:
                - A Table object
                - A (name, Table) tuple to customize the column name
                - A (name, nullok, Table) tuple for nullable references
                - A Key object for explicit key selection
            metadata: Additional metadata columns or reference targets.
            table_name: Name for the association table. Auto-generated if omitted.
            comment: Comment for the association table.
            **kwargs: Additional arguments passed to Table.define_association.

        Returns:
            Table definition dict suitable for ``create_table()``.

        Example::

            # Associate Image with Subject (many-to-many)
            image_table = ml.model.name_to_table("Image")
            subject_table = ml.model.name_to_table("Subject")
            assoc_def = ml.define_association(
                associates=[image_table, subject_table],
                comment="Links images to subjects",
            )
            ml.create_table(assoc_def)
        """
        return self.model._define_association(
            associates=associates,
            metadata=metadata,
            table_name=table_name,
            comment=comment,
            **kwargs,
        )

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

        stats = {"files_removed": 0, "dirs_removed": 0, "bytes_freed": 0, "errors": 0}

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
                        entry_size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
                        shutil.rmtree(entry)
                        stats["dirs_removed"] += 1
                    else:
                        entry_size = entry.stat().st_size
                        entry.unlink()
                        stats["files_removed"] += 1

                    stats["bytes_freed"] += entry_size
                except (OSError, PermissionError) as e:
                    self._logger.warning(f"Failed to remove cache entry {entry}: {e}")
                    stats["errors"] += 1

        except OSError as e:
            self._logger.error(f"Failed to iterate cache directory: {e}")
            stats["errors"] += 1

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
        stats = {"total_bytes": 0, "total_mb": 0.0, "file_count": 0, "dir_count": 0}

        if not self.cache_dir.exists():
            return stats

        for entry in self.cache_dir.rglob("*"):
            if entry.is_file():
                stats["total_bytes"] += entry.stat().st_size
                stats["file_count"] += 1
            elif entry.is_dir():
                stats["dir_count"] += 1

        stats["total_mb"] = stats["total_bytes"] / (1024 * 1024)
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
                size_bytes = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
                file_count = sum(1 for f in entry.rglob("*") if f.is_file())
                mtime = datetime.fromtimestamp(entry.stat().st_mtime)

                results.append(
                    {
                        "execution_rid": entry.name,
                        "path": str(entry),
                        "size_bytes": size_bytes,
                        "size_mb": size_bytes / (1024 * 1024),
                        "modified": mtime,
                        "file_count": file_count,
                    }
                )

        return sorted(results, key=lambda x: x["modified"], reverse=True)

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

        stats = {"dirs_removed": 0, "bytes_freed": 0, "errors": 0}
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
                entry_size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
                shutil.rmtree(entry)
                stats["dirs_removed"] += 1
                stats["bytes_freed"] += entry_size

            except (OSError, PermissionError) as e:
                self._logger.warning(f"Failed to remove execution dir {entry}: {e}")
                stats["errors"] += 1

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

        exec_size_mb = sum(d["size_mb"] for d in exec_dirs)

        return {
            "working_dir": str(self.working_dir),
            "cache_dir": str(self.cache_dir),
            "cache_size_mb": cache_stats["total_mb"],
            "cache_file_count": cache_stats["file_count"],
            "execution_dir_count": len(exec_dirs),
            "execution_size_mb": exec_size_mb,
            "total_size_mb": cache_stats["total_mb"] + exec_size_mb,
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
        from deriva_ml.schema.validation import validate_ml_schema

        return validate_ml_schema(self, strict=strict)

    # Methods moved to mixins:
    # - create_asset, list_assets -> AssetMixin
    # - create_feature, feature_record_class, delete_feature, lookup_feature, list_feature_values -> FeatureMixin
    # - find_datasets, create_dataset, lookup_dataset, delete_dataset, list_dataset_element_types,
    #   add_dataset_element_type, download_dataset_bag -> DatasetMixin
    # - _update_status, create_execution, resume_execution -> ExecutionMixin
    # - add_files, list_files, _bootstrap_versions, _synchronize_dataset_versions, _set_version_snapshot -> FileMixin


# =============================================================================
# Module-level helpers
# =============================================================================

CONTEXT_FILENAME = ".deriva-context.json"


def _find_context_file(start: Path) -> Path:
    """Search for .deriva-context.json starting from start, walking up.

    Args:
        start: Directory to start searching from.

    Returns:
        Path to the context file.

    Raises:
        FileNotFoundError: If no context file is found.
    """
    current = start.resolve()
    while True:
        candidate = current / CONTEXT_FILENAME
        if candidate.exists():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    raise FileNotFoundError(
        f"No {CONTEXT_FILENAME} found in {start} or any parent directory. "
        "Connect to a catalog first using the MCP 'connect_catalog' tool, "
        "or create the file manually with hostname, catalog_id, and default_schema."
    )
