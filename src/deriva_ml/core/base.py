"""Core module for the Deriva ML project.

This module implements the DerivaML class, which is the primary interface to Deriva-based catalogs. It provides
functionality for managing features, vocabularies, and other ML-related operations.

The module requires a catalog that implements a 'deriva-ml' schema with specific tables and relationships.

Typical usage example:
    >>> ml = DerivaML('deriva.example.org', 'my_catalog')  # doctest: +SKIP
    >>> ml.create_feature('my_table', 'new_feature')  # doctest: +SKIP
    >>> ml.add_term('vocabulary_table', 'new_term', description='Description of term')  # doctest: +SKIP
"""

from __future__ import annotations  # noqa: I001

# Standard library imports
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Dict, cast, TYPE_CHECKING, Any
from typing_extensions import Self

# Third-party imports

# Deriva imports - use importlib to avoid shadowing by local 'deriva.py' files
import importlib

_deriva_core = importlib.import_module("deriva.core")
_deriva_server = importlib.import_module("deriva.core.deriva_server")
_ermrest_catalog = importlib.import_module("deriva.core.ermrest_catalog")
_ermrest_model = importlib.import_module("deriva.core.ermrest_model")
_core_utils = importlib.import_module("deriva.core.utils.core_utils")
DEFAULT_SESSION_CONFIG = _deriva_core.DEFAULT_SESSION_CONFIG
get_credential = _deriva_core.get_credential
urlquote = _deriva_core.urlquote
DerivaServer = _deriva_server.DerivaServer
ErmrestCatalog = _ermrest_catalog.ErmrestCatalog
ErmrestSnapshot = _ermrest_catalog.ErmrestSnapshot
Table = _ermrest_model.Table
DEFAULT_LOGGER_OVERRIDES = _core_utils.DEFAULT_LOGGER_OVERRIDES

from deriva_ml.core.catalog_stub import CatalogStub
from deriva_ml.core.config import DerivaMLConfig
from deriva_ml.core.connection_mode import ConnectionMode
from deriva_ml.core.definitions import (
    DRY_RUN_RID,
    ML_SCHEMA,
    RID,
    ColumnDefinition,
    TableDefinition,
    VocabularyTableDef,
)
from deriva_ml.core.exceptions import (
    DerivaMLConfigurationError,
    DerivaMLException,
    DerivaMLOfflineError,
    DerivaMLSchemaPinned,
    DerivaMLSchemaRefreshBlocked,
)
from deriva_ml.core.logging_config import _apply_logger_overrides, configure_logging, get_logger
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
from deriva_ml.interfaces import DerivaMLCatalog

if TYPE_CHECKING:
    from deriva_ml.catalog.provenance import CatalogProvenance
    from deriva_ml.core.schema_diff import SchemaDiff
    from deriva_ml.core.storage import CachedAsset, CachedBag
    from deriva_ml.execution.execution import Execution
    from deriva_ml.model.catalog import DerivaModel

logger = get_logger(__name__)

# Stop pycharm from complaining about undefined references.
ml: DerivaML


__all__ = ["DerivaML"]


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

    Method naming convention:
        - ``find_*`` methods search the catalog for entities of a kind, optionally filtered.
          Examples: ``find_features(table=None)``, ``find_datasets()``, ``find_workflows()``,
          ``find_executions()``, ``find_experiments()``, ``find_assets()``. ``find_*`` returns
          everything that matches; pass arguments to narrow the search.
        - ``list_*`` methods enumerate things scoped to a specific entity passed as the first
          argument. Examples: ``list_assets(asset_table)``, ``list_dataset_members(dataset)``,
          ``list_dataset_children(dataset)``, ``list_workflow_executions(workflow)``,
          ``list_vocabulary_terms(table)``. ``list_*`` always has a "scope" argument; there is
          no scope-less ``list_*`` flavor for entities of a given kind — use ``find_*`` for that.

        So: "all features on the catalog" → ``find_features()``; "all features on table T" →
        ``find_features(T)`` (scoping is a filter); "all members of dataset D" →
        ``list_dataset_members(D)`` (scoping is the parent entity itself). There is no
        ``list_features()`` because features aren't scoped to a parent entity in the way
        dataset members are scoped to a dataset.

    Attributes:
        host_name (str): Hostname of the Deriva server (e.g., 'deriva.example.org').
        catalog_id (Union[str, int]): Catalog identifier or name.
        domain_schemas (frozenset[str]): Schema names for domain-specific tables and relationships.
        model (DerivaModel): ERMRest model for the catalog.
        working_dir (Path): Directory for storing computation data and results.
        cache_dir (Path): Directory for caching downloaded datasets.
        ml_schema (str): Schema name for ML-specific tables (default: 'deriva_ml').
        configuration (ExecutionConfiguration): Current execution configuration.
        project_name (str): Name of the current project.
        start_time (datetime): Timestamp when this instance was created.

    Example:
        >>> ml = DerivaML('deriva.example.org', 'my_catalog')  # doctest: +SKIP
        >>> ml.create_feature('my_table', 'new_feature')  # doctest: +SKIP
        >>> ml.add_term('vocabulary_table', 'new_term', description='Description of term')  # doctest: +SKIP
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
            >>> from hydra_zen import builds, instantiate  # doctest: +SKIP
            >>> from deriva_ml import DerivaML  # doctest: +SKIP
            >>> from deriva_ml.core.config import DerivaMLConfig  # doctest: +SKIP
            >>>
            >>> # Create a structured config using hydra-zen
            >>> DerivaMLConf = builds(DerivaMLConfig, populate_full_signature=True)  # doctest: +SKIP
            >>>
            >>> # Configure for your environment
            >>> conf = DerivaMLConf(  # doctest: +SKIP
            ...     hostname='deriva.example.org',
            ...     catalog_id='42',
            ...     domain_schemas={'my_domain'},
            ... )
            >>>
            >>> # Instantiate the config to get a DerivaMLConfig object
            >>> config = instantiate(conf)  # doctest: +SKIP
            >>>
            >>> # Create the DerivaML instance
            >>> ml = DerivaML.instantiate(config)  # doctest: +SKIP

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
        clean_execution_dir: bool = True,
        mode: ConnectionMode | str = ConnectionMode.online,
        reuse_schema_json: dict | None = None,
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
            clean_execution_dir: Whether to automatically clean up execution working directories
                after successful upload. Defaults to True. Set to False to retain local copies.
            mode: Connection mode for this instance. ``ConnectionMode.online`` (default)
                sends writes to the catalog eagerly; ``ConnectionMode.offline`` stages
                writes into local SQLite for later upload. Accepts the string
                literals ``"online"`` or ``"offline"``; any other value raises
                ``ValueError``. See spec §2.1.
            reuse_schema_json: Internal. A pre-parsed ermrest ``/schema``
                dict to build the model from, skipping the live
                ``getCatalogSchema()`` fetch. Used by
                :meth:`catalog_snapshot` to avoid re-introspecting a
                schema already held in memory (a snapshot's schema is
                structurally identical to the live catalog's). Not for
                general use.
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
                cache=cache,
                ml_schema=ml_schema,
                domain_schemas=domain_schemas,
                default_schema=default_schema,
                reuse_schema_json=reuse_schema_json,
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
        _apply_logger_overrides(DEFAULT_LOGGER_OVERRIDES)

        # Store instance configuration
        self.host_name = hostname
        self.catalog_id = catalog_id
        self.ml_schema = ml_schema
        self.configuration = None
        self._execution: Execution | None = None
        # Memoize snapshot DerivaML instances by snapshot id so the
        # many Dataset call sites that build a snapshot view within one
        # operation share a single instance (and its cached schema).
        # Snapshots are immutable, so entries never go stale.
        self._snapshot_cache: dict[str, "DerivaML"] = {}
        self.domain_schemas = self.model.domain_schemas
        self.default_schema = self.model.default_schema
        self.project_name = project_name or self.default_schema or "deriva-ml"
        self.start_time = datetime.now()
        self.clean_execution_dir = clean_execution_dir

    def __del__(self) -> None:
        """Cleanup method to handle incomplete executions.

        Best-effort abort on DerivaML shutdown — only for executions that
        died mid-flight (i.e., still in ``Created`` or ``Running``). Any
        post-Running status (``Stopped``, ``Failed``, ``Pending_Upload``,
        ``Uploaded``, ``Aborted``) is treated as terminal here: the user
        has either committed cleanly via the context manager or
        explicitly transitioned the execution, and a forced abort would
        either be a no-op or a wrongful state change.

        Forcing a transition during ``__del__`` is also unsafe at object-
        teardown time: Python's GC ordering means the underlying
        ``ErmrestCatalog`` HTTP session may already be finalized, in
        which case the catalog PUT would crash with ``'NoneType' object
        has no attribute 'get'`` (the catalog's ``_session`` reads as
        ``None``). Limiting the abort to non-terminal states avoids the
        common case where ``__exit__`` already moved the execution to
        ``Stopped`` and ``__del__`` would otherwise re-transition to
        ``Aborted`` against a dead session.
        """
        # Inline import to avoid a circular (core.base ↔ execution.state_store) import.
        try:
            from deriva_ml.execution.state_store import ExecutionStatus

            non_terminal = {ExecutionStatus.Created, ExecutionStatus.Running}
            if self._execution and self._execution.status in non_terminal:
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
        cache: "SchemaCache",
        ml_schema: str,
        domain_schemas: "str | set[str] | None",
        default_schema: "str | None",
        reuse_schema_json: dict | None = None,
    ) -> None:
        """Online init: connect to server, fetch the live schema, build the model.

        Schema freshness is handled entirely by deriva-py's
        ``ErmrestCatalog``: it caches the parsed ``/schema`` dict on
        the catalog instance, auto-invalidates on any same-instance
        schema-mutating POST/PUT/DELETE, and uses HTTP ETags
        (``If-None-Match``) for cross-instance freshness on every
        read. ``_init_online`` does not maintain its own schema cache.

        The disk cache write below is **only** for offline-mode
        bootstrapping: a later ``DerivaML(mode=offline)`` in the same
        working directory reads the JSON we write here. Online mode
        never reads it back.

        No pre-flight auth probe *here*: ``_init_online`` does not gate
        construction on a session — authentication failures surface as
        401s on the first real ermrest call (``getCatalogSchema()``
        below), source-true rather than a synthetic wrapper. (Callers who
        want an explicit, friendly check before doing work can use
        :meth:`is_authenticated` / :meth:`whoami`, which DO hit
        ``GET /authn/session`` and return a clean result — verified
        returning 200 + the client identity on current servers.)
        """
        from deriva_ml.model.catalog import DerivaModel

        server = DerivaServer(
            "https",
            hostname,
            credentials=self.credential,
            session_config=self._get_session_config(),
        )
        self.catalog = server.connect_ermrest(catalog_id)

        if reuse_schema_json is not None:
            # Caller (catalog_snapshot) handed us a schema parsed by the live
            # instance. It is USUALLY identical to this catalog's schema, but
            # NOT always: a snapshot pinned to a snaptime that predates a
            # schema migration has a different schema than the live instance
            # the reuse came from. Trusting the reused dict blindly would put
            # tables in the model that this (snapshot) catalog does not have,
            # and a later pathBuilder query against the snapshot would 409.
            #
            # So VALIDATE: fetch this catalog's own /schema. getCatalogSchema
            # is ETag-revalidated (a cheap 304 when the snapshot schema equals
            # live — the common case), and a real fetch only when they genuinely
            # differ — exactly when reuse would be incorrect. Use the catalog's
            # real schema so the model always matches the catalog it queries.
            actual_schema = self.catalog.getCatalogSchema()
            if actual_schema == reuse_schema_json:
                schema_json = reuse_schema_json
            else:
                logger.info(
                    "reuse_schema_json differs from the connected catalog's "
                    "schema (likely a snapshot predating a schema change); "
                    "using the catalog's own schema instead"
                )
                schema_json = actual_schema
        else:
            # Fetch the live schema. deriva-py caches the parsed dict on
            # the catalog instance and auto-invalidates on schema-mutating
            # POST/PUT/DELETE through the same catalog, so subsequent
            # reads in the same process are O(1) and always current.
            # The disk cache write below is purely for offline mode.
            schema_json = self.catalog.getCatalogSchema()
            live_snapshot_id = self.catalog.get("/").json()["snaptime"]
            cache.write(
                snapshot_id=live_snapshot_id,
                hostname=hostname,
                catalog_id=str(catalog_id),
                ml_schema=ml_schema,
                schema=schema_json,
            )
        # Retain the parsed schema so catalog_snapshot() can reuse it.
        self._schema_json = schema_json

        self.model = DerivaModel.from_cached(
            schema_json,
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
        """Force-refetch the live catalog schema; rebuild the model and disk cache.

        Online mode only. For normal in-process use this method is
        rarely needed: deriva-py's ``ErmrestCatalog`` already handles
        schema freshness automatically (auto-invalidation on same-
        instance mutations, ``If-None-Match`` revalidation on every
        read). Use ``refresh_schema()`` when you specifically need to
        bypass the in-process cache and re-fetch from the live
        catalog -- e.g. after a known out-of-band mutation by another
        process, or before overwriting the offline-mode disk cache.

        The disk cache (``SchemaCache``) is rewritten with the fresh
        ``/schema`` JSON so subsequent offline-mode reads see the new
        snapshot. The in-memory ``self.model`` is rebuilt from the
        same fresh JSON.

        Refuses in two cases:

        1. The disk cache is pinned (via :meth:`pin_schema`). Raises
           :class:`DerivaMLSchemaPinned`. ``force=True`` does NOT
           bypass a pin — call :meth:`unpin_schema` first.
        2. The workspace has pending rows (staged/leasing/leased/
           uploading/failed). Raises
           :class:`DerivaMLSchemaRefreshBlocked` unless ``force=True``
           is passed; a forced refresh may leave staged rows whose
           metadata references columns or types no longer in the
           new schema, causing catalog-insert failures on the next
           upload.

        Args:
            force: If True, refresh even when the workspace has
                pending rows. Does NOT bypass a pin.

        Raises:
            DerivaMLOfflineError: If called in offline mode.
            DerivaMLSchemaPinned: If the disk cache is pinned (any
                ``force`` value).
            DerivaMLSchemaRefreshBlocked: If ``force=False`` and the
                workspace has pending rows (and the cache is not
                pinned).
        """
        from deriva_ml.model.catalog import DerivaModel

        if self._mode is not ConnectionMode.online:
            raise DerivaMLOfflineError("refresh_schema requires online mode")
        cache = SchemaCache(self.working_dir)
        if cache.exists() and cache.pin_status().pinned:
            pin_info = cache.pin_status()
            raise DerivaMLSchemaPinned(
                f"refresh_schema refused: cache is pinned at snapshot "
                f"{pin_info.pinned_snapshot_id}"
                + (f" (reason: {pin_info.pin_reason})" if pin_info.pin_reason else "")
                + ". Call ml.unpin_schema() first."
            )
        store = self.workspace.execution_state_store()
        count = store.count_pending_rows()
        if count > 0 and not force:
            raise DerivaMLSchemaRefreshBlocked(
                f"refresh_schema requires a drained workspace; "
                f"{count} pending rows. Run ml.commit_pending_executions() first, "
                f"or call refresh_schema(force=True) to discard local "
                f"state (staged rows may become inconsistent with the "
                f"new schema)."
            )
        # Force a refetch through deriva-py's binding cache; rebuilds
        # the parsed-dict and path-builder caches on the catalog.
        # ``getCatalogSchema()`` is conditionally revalidated via
        # ``If-None-Match`` on every call, so an external mutation is
        # naturally observed; purging the prefix here guarantees the
        # refetch even when the ETag has not changed (defensive: covers
        # cases where the server lies about ETag stability or where the
        # caller explicitly wants the parsed-dict cache invalidated).
        self.catalog.purge_cache_by_prefix("/schema")
        live_schema = self.catalog.getCatalogSchema()
        live_snapshot_id = self.catalog.get("/").json()["snaptime"]
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
        logger.info("schema refreshed to snapshot %s", live_snapshot_id)

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
        from deriva_ml.core.schema_diff import _compute_diff

        cache = SchemaCache(self.working_dir)
        drift: SchemaDiff | None = None
        if self._mode is ConnectionMode.online:
            live_snapshot_id = self.catalog.get("/").json()["snaptime"]
            cached_payload = cache.load()
            if cached_payload["snapshot_id"] != live_snapshot_id:
                # See refresh_schema for the purge+get rationale.
                self.catalog.purge_cache_by_prefix("/schema")
                live_schema = self.catalog.getCatalogSchema()
                diff = _compute_diff(cached_payload["schema"], live_schema)
                if not diff.is_empty():
                    logger.warning(
                        "pin_schema: cache at %s, live at %s; structural drift detected (see returned SchemaDiff)",
                        cached_payload["snapshot_id"],
                        live_snapshot_id,
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
        :func:`~deriva_ml.core.schema_diff._compute_diff`, and returns
        the result. The returned :class:`SchemaDiff` may be empty
        (no drift) — callers should check ``diff.is_empty()`` rather
        than truthiness.

        Unlike :meth:`pin_schema`, this method never modifies the
        cache and never logs a warning; it is a pure inspection
        operation.

        Returns:
            A :class:`SchemaDiff`, possibly empty.

        Raises:
            DerivaMLOfflineError: If called in offline mode.
            FileNotFoundError: If the workspace has no cache file.
        """
        from deriva_ml.core.schema_diff import _compute_diff

        if self._mode is not ConnectionMode.online:
            raise DerivaMLOfflineError("diff_schema requires online mode")
        cache = SchemaCache(self.working_dir)
        cached_payload = cache.load()
        # See refresh_schema for the purge+get rationale.
        self.catalog.purge_cache_by_prefix("/schema")
        live_schema = self.catalog.getCatalogSchema()
        return _compute_diff(cached_payload["schema"], live_schema)

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
            >>> config['retry_read']
            8
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

    def whoami(self) -> dict | None:
        """Return the authenticated client identity, or ``None`` if not logged in.

        Asks the **server** for the current session via
        ``GET /authn/session`` (deriva-py's
        :meth:`~deriva.core.ErmrestCatalog.get_authn_session`). On a valid
        session the server returns the logged-in client, and this returns that
        ``client`` dict (``id``, ``display_name``, ``email``, ``full_name``,
        ``identities``). When there is no session the endpoint returns 401/404
        and this returns ``None``.

        This makes a network call. A successful (non-``None``) result is a safe
        bet that catalog operations will work — it proves the credential is
        present, unexpired, and accepted by the server's auth layer (the thing
        that otherwise blanket-401s). It confirms *authentication* (the server
        knows who you are), not *authorization* for any specific operation — a
        write to a read-only table can still be refused with a valid session.

        Returns:
            dict | None: The ``client`` identity dict, or ``None`` if there is
                no authenticated session.

        Raises:
            requests.exceptions.HTTPError: For HTTP errors other than 401/404
                (e.g. a 5xx server error) — these are real failures, not a
                "no session" answer, so they propagate.

        Example:
            >>> who = ml.whoami()  # doctest: +SKIP
            >>> who["display_name"] if who else "not logged in"  # doctest: +SKIP
            'user@example.org'
        """
        from requests.exceptions import HTTPError

        try:
            session = self.catalog.get_authn_session()
        except HTTPError as e:
            # 401 (classic webauthn) and 404 (no session on the current backend)
            # both mean "not authenticated". Anything else is a real error.
            if e.response is not None and e.response.status_code in (401, 404):
                return None
            raise
        return session.json().get("client")

    def is_authenticated(self) -> bool:
        """Whether there is a valid authenticated session for this catalog.

        Thin boolean over :meth:`whoami`: ``True`` if the server returns a
        client identity (``GET /authn/session`` → 200), ``False`` if there is no
        session (401/404). Makes one network call.

        A ``True`` result is a safe bet that catalog operations will work (the
        credential is accepted by the server). It means "I am logged in," not
        "every privileged operation will pass ACLs" — see :meth:`whoami` for the
        authentication-vs-authorization distinction.

        Returns:
            bool: True if authenticated, False otherwise.

        Raises:
            requests.exceptions.HTTPError: For non-auth HTTP errors (propagated
                from :meth:`whoami`).

        Example:
            >>> if not ml.is_authenticated():  # doctest: +SKIP
            ...     raise SystemExit("Log in first: deriva-globus-auth-utils login --host ...")
        """
        return self.whoami() is not None

    def catalog_snapshot(self, version_snapshot: str) -> Self:
        """Return a new DerivaML instance connected to a specific catalog snapshot.

        Catalog snapshots provide a read-only, point-in-time view of the catalog.
        The snapshot identifier is typically obtained from a dataset version record.

        Every connection-shaping kwarg the original instance was
        constructed with (``working_dir``, ``cache_dir``,
        ``domain_schemas``, ``default_schema``, ``s3_bucket``,
        ``use_minid``, ``credential``, ``mode``, ``ml_schema``,
        ``project_name``, ``clean_execution_dir``, plus the two
        logging levels) is forwarded to the snapshot instance.
        Without this forwarding, the snapshot would silently default
        ``working_dir`` to ``~/deriva_ml`` even when the user
        constructed ``self`` with an explicit shared-tree path, and
        would re-fetch credentials and re-detect domain schemas
        (which can pick differently from the snapshot than the live
        catalog did) — both observable behaviour drifts.

        The snapshot reuses this instance's already-parsed schema
        (``self._schema_json``) rather than re-fetching ``/schema`` — a
        snapshot's schema is structurally identical to the live
        catalog's. The constructed instance is memoized by
        ``version_snapshot`` so repeated calls share one object.

        Precondition: the snapshot's schema must match the live
        catalog's. This holds for deriva-ml's use (pinning a recent
        dataset-version snaptime on a catalog whose schema has not been
        migrated since). Do not use for a snapshot taken *before* a
        schema migration — its structure would differ from live.

        Args:
            version_snapshot: Snapshot identifier string (e.g., ``"2T-SXEH-JH4A"``),
                usually the ``snapshot`` field from a :class:`DatasetHistory` entry.

        Returns:
            A new DerivaML instance connected to the specified catalog snapshot,
            inheriting every connection-shaping kwarg from ``self``.

        Note:
            The reused ``schema_json`` can lag the live catalog within a
            session — e.g. tables created after this connection was opened
            are absent from the held model. Any caller that mixes this
            snapshot's ``pathBuilder()`` (held, possibly-stale model) with a
            freshly-fetched deriva-py model (e.g. a ``CatalogBagBuilder``
            walk that calls ``_get_model()``) must build its path builder
            from the **same** fresh model via
            ``datapath.from_model(snapshot.catalog, model)`` — otherwise a
            ``KeyError`` is raised on tables the held model lacks.
        """
        cached = self._snapshot_cache.get(version_snapshot)
        if cached is not None:
            return cached

        snapshot = DerivaML(
            self.host_name,
            version_snapshot,
            domain_schemas=self.domain_schemas,
            default_schema=self.default_schema,
            project_name=self.project_name,
            cache_dir=self.cache_dir,
            working_dir=self.working_dir,
            ml_schema=self.ml_schema,
            logging_level=self._logging_level,
            deriva_logging_level=self._deriva_logging_level,
            credential=self.credential,
            s3_bucket=self.s3_bucket,
            use_minid=self.use_minid,
            clean_execution_dir=self.clean_execution_dir,
            mode=self._mode,
            reuse_schema_json=self._schema_json,
        )
        self._snapshot_cache[version_snapshot] = snapshot
        return snapshot

    @property
    def mode(self) -> ConnectionMode:
        """Current connection mode.

        Returns:
            The ConnectionMode this DerivaML instance was constructed
            with. Drives whether writes go live to the catalog (online)
            or stage in SQLite for later upload (offline). See spec §2.1.

        Example:
            >>> ml.mode is ConnectionMode.online  # doctest: +SKIP
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
            >>> cache_dir = ml.download_dir(cached=True)  # doctest: +SKIP
            >>> work_dir = ml.download_dir(cached=False)  # doctest: +SKIP
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
                    logger.info(
                        "Migrated %d legacy asset manifests into workspace",
                        n,
                    )
            except Exception as exc:
                logger.warning(
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

    def _cache_features(
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
            **kwargs: Additional arguments passed to ``feature_values``
                (e.g., ``selector``, ``workflow``, ``execution``).

        Returns:
            DataFrame with feature value records.

        Example::

            labels = ml._cache_features("Image", "Classification")
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

        records = [
            r.model_dump(mode="json") for r in self.feature_values(table_name, feature_name=feature_name, **kwargs)
        ]
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
                >>> ml.chaise_url("experiment_table")  # doctest: +SKIP
                'https://deriva.org/chaise/recordset/#1/schema:experiment_table'

            Using RID:
                >>> ml.chaise_url("1-abc123")  # doctest: +SKIP
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
                >>> url = ml.cite("1-abc123")  # doctest: +SKIP
                >>> print(url)  # doctest: +SKIP
                'https://deriva.org/id/1/1-abc123@2024-01-01T12:00:00'

            Current catalog URL:
                >>> url = ml.cite("1-abc123", current=True)  # doctest: +SKIP
                >>> print(url)  # doctest: +SKIP
                'https://deriva.org/id/1/1-abc123'

            Using a dictionary:
                >>> url = ml.cite({"RID": "1-abc123"})  # doctest: +SKIP

            Dry-run sentinel — no catalog round-trip, no clickable link:
                >>> url = ml.cite("0000")  # doctest: +SKIP
                >>> print(url)  # doctest: +SKIP
                'dry-run (rid=0000)'
        """
        # Dry-run sentinel: ``run_notebook(dry_run=True)`` and friends
        # hand back ``DRY_RUN_RID`` as the execution RID because no row
        # was created on the catalog. Resolving it would 404. Return a
        # bare, non-link string so notebook templates that embed the
        # output in ``[label]({url})`` markdown render it as plain text
        # rather than a clickable link to a 404.
        rid_value = entity if isinstance(entity, str) else entity.get("RID")
        if rid_value == DRY_RUN_RID:
            return f"dry-run (rid={DRY_RUN_RID})"

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
            >>> ml = DerivaML('localhost', '45')  # doctest: +SKIP
            >>> prov = ml.catalog_provenance  # doctest: +SKIP
            >>> if prov:  # doctest: +SKIP
            ...     print(f"Created: {prov.created_at} by {prov.created_by}")
            ...     print(f"Method: {prov.creation_method}")
            ...     if prov.is_clone:
            ...         print(f"Cloned from: {prov.clone_details.source_hostname}")
        """
        from deriva_ml.catalog.provenance import get_catalog_provenance

        return get_catalog_provenance(self.catalog)

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
            >>> ml = DerivaML('deriva.example.org', 'my_catalog')  # doctest: +SKIP
            >>> # After creating domain schema and tables...
            >>> ml.apply_catalog_annotations()  # doctest: +SKIP
            >>> # Or with custom branding:
            >>> ml.apply_catalog_annotations("My Project Browser", "My ML Project")  # doctest: +SKIP
        """
        # Single source of truth lives in
        # :mod:`deriva_ml.schema.annotations`. Delegate to it.
        from deriva_ml.schema.annotations import catalog_annotation

        catalog_annotation(
            self.model,
            navbar_brand_text=navbar_brand_text,
            head_title=head_title,
        )

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

                >>> table = ml.create_vocabulary(  # doctest: +SKIP
                ...     vocab_name="tissue_types",
                ...     comment="Standard tissue classifications",
                ...     schema="bio_schema"
                ... )

            Create multiple vocabularies without updating navbar until the end:

                >>> ml.create_vocabulary("Species", update_navbar=False)  # doctest: +SKIP
                >>> ml.create_vocabulary("Tissue_Type", update_navbar=False)  # doctest: +SKIP
                >>> ml.apply_catalog_annotations()  # Update navbar once  # doctest: +SKIP
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
        # Invalidate the pathBuilder cache: the model was mutated in-place
        # so the identity hasn't changed, but the cached wrapper is stale.
        self._path_builder_cache = None

        # Curate the vocabulary's Chaise presentation (row-name, compact columns,
        # Name facet) so a runtime-created vocab is annotated consistently with
        # asset/feature tables rather than left on raw Chaise defaults.
        from deriva_ml.schema.annotations import vocabulary_annotation

        vocabulary_annotation(vocab_table)

        # Update navbar to include the new vocabulary table
        if update_navbar:
            self.apply_catalog_annotations()

        return vocab_table

    def create_asset_table(
        self,
        asset_name: str,
        additional_columns: Sequence[ColumnDefinition] = (),
        comment: str | None = None,
        schema: str | None = None,
        use_hatrac: bool = True,
        update_navbar: bool = True,
    ) -> Table:
        """Creates an asset table with the canonical DerivaML asset shape.

        One call builds everything an asset table needs, so the shape can
        never drift from the canonical form (validation-by-construction --
        the result always satisfies ``model.is_asset``):

        - The five standard hatrac columns: ``URL``, ``Filename``,
          ``Length``, ``MD5``, ``Description`` (with the standard NOT NULL
          constraints and the ``asset`` annotation on ``URL``).
        - The ``<asset_name>_Asset_Type`` association to the deriva-ml
          ``Asset_Type`` vocabulary (an asset can carry multiple type tags,
          e.g. ``Model_File`` + ``Output_File``).
        - The ``<asset_name>_Execution`` association to ``Execution``,
          carrying the ``Asset_Role`` FK (``Input`` / ``Output``) that the
          execution upload machinery writes.
        - The standard Chaise display annotations for asset tables.

        This replaces the manual recipe (generic ``create_table`` with
        hand-written hatrac columns) that was verbose and easy to get
        subtly wrong (issue #74).

        Args:
            asset_name: Name for the new asset table. Must be a valid SQL
                identifier.
            additional_columns: Optional domain-specific columns appended to
                the standard hatrac shape (e.g. a ``Scanner_Model`` text
                column on a ``Scan_File`` table). Use ``ColumnDefinition``
                with ``BuiltinTypes`` from ``deriva_ml``.
            comment: Description of the asset table's purpose.
            schema: Schema name to create the table in. If None, uses the
                domain schema.
            use_hatrac: When True (default) the ``URL`` column is wired with
                the Hatrac upload template so Chaise's file-upload UI
                deposits bytes into Hatrac. When False the ``URL`` column is
                a plain string (for assets whose bytes live elsewhere).
            update_navbar: If True (default), refresh the navigation bar to
                include the new table. Set to False during batch creation,
                then call ``apply_catalog_annotations()`` once at the end.

        Returns:
            Table: ERMrest table object for the newly created asset table.

        Raises:
            DerivaMLException: If ``asset_name`` already exists.

        Examples:
            Create an asset table for raw scanner output:

                >>> from deriva_ml import ColumnDefinition, BuiltinTypes
                >>> table = ml.create_asset_table(  # doctest: +SKIP
                ...     "Scan_File",
                ...     additional_columns=[
                ...         ColumnDefinition(name="Scanner_Model", type=BuiltinTypes.text),
                ...     ],
                ...     comment="Raw scanner output files.",
                ... )
                >>> ml.model.is_asset("Scan_File")  # doctest: +SKIP
                True
        """
        # Lazy import: schema.create_schema owns the canonical wiring (it
        # builds the same shape at catalog bootstrap); importing it at
        # module load would cycle through the schema package.
        from deriva_ml.core.definitions import MLTable, MLVocab
        from deriva_ml.schema.create_schema import create_asset_table as _create_asset_table

        schema = schema or self.model._require_default_schema()
        try:
            asset_table = _create_asset_table(
                self.model.schemas[schema],
                asset_name,
                execution_table=self.model.name_to_table(MLTable.execution),
                asset_type_table=self.model.name_to_table(MLVocab.asset_type),
                asset_role_table=self.model.name_to_table(MLVocab.asset_role),
                use_hatrac=use_hatrac,
                comment=comment,
                additional_columns=additional_columns,
            )
        except ValueError:
            raise DerivaMLException(f"Table {asset_name} already exist")
        # Invalidate the pathBuilder cache: the model was mutated in-place
        # so the identity hasn't changed, but the cached wrapper is stale.
        self._path_builder_cache = None

        if update_navbar:
            self.apply_catalog_annotations()

        return asset_table

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

                >>> from deriva_ml import TableDefinition, ColumnDefinition, BuiltinTypes  # doctest: +SKIP
                >>>
                >>> table_def = TableDefinition(  # doctest: +SKIP
                ...     name="Experiment",
                ...     column_defs=[
                ...         ColumnDefinition(name="Name", type=BuiltinTypes.text, nullok=False),
                ...         ColumnDefinition(name="Date", type=BuiltinTypes.date),
                ...         ColumnDefinition(name="Description", type=BuiltinTypes.markdown),
                ...         ColumnDefinition(name="Score", type=BuiltinTypes.float4),
                ...     ],
                ...     comment="Records of experimental runs"
                ... )
                >>> experiment_table = ml.create_table(table_def)  # doctest: +SKIP

            **Table with foreign key to another table**:

                >>> from deriva_ml import (  # doctest: +SKIP
                ...     TableDefinition, ColumnDefinition, ForeignKeyDefinition, BuiltinTypes
                ... )
                >>>
                >>> # Create a Sample table that references Subject
                >>> sample_def = TableDefinition(  # doctest: +SKIP
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
                >>> sample_table = ml.create_table(sample_def)  # doctest: +SKIP

            **Table with unique key constraint**:

                >>> from deriva_ml import (  # doctest: +SKIP
                ...     TableDefinition, ColumnDefinition, KeyDefinition, BuiltinTypes
                ... )
                >>>
                >>> protocol_def = TableDefinition(  # doctest: +SKIP
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
                >>> protocol_table = ml.create_table(protocol_def)  # doctest: +SKIP

            **Batch creation without navbar updates**:

                >>> ml.create_table(table1_def, update_navbar=False)  # doctest: +SKIP
                >>> ml.create_table(table2_def, update_navbar=False)  # doctest: +SKIP
                >>> ml.create_table(table3_def, update_navbar=False)  # doctest: +SKIP
                >>> ml.apply_catalog_annotations()  # Update navbar once at the end  # doctest: +SKIP
        """
        # Use default schema if none specified
        schema = schema or self.model._require_default_schema()

        # Create table in domain schema using provided definition
        # Handle both TableDefinition (dataclass with to_dict) and plain dicts
        table_dict = table.to_dict() if hasattr(table, "to_dict") else table
        new_table = self.model.schemas[schema].create_table(table_dict)
        # Invalidate the pathBuilder cache: the model was mutated in-place
        # so the identity hasn't changed, but the cached wrapper is stale.
        self._path_builder_cache = None

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

        Removes cached dataset bags and assets. Bags are removed
        *through the bag-cache index* (index row and on-disk directory
        together), so the index never references a removed bag.

        Args:
            older_than_days: If provided, only remove cache entries older than this
                many days (bags age by their recorded ``built_at``; assets and
                stray entries by mtime). If None, removes all cache entries.

        Returns:
            dict: Statistics about the cleanup:
                - 'files_removed': Number of files removed
                - 'dirs_removed': Number of directories removed
                - 'bytes_freed': Total bytes freed
                - 'errors': Number of errors encountered

        Example:
            >>> ml = DerivaML('deriva.example.org', 'my_catalog')  # doctest: +SKIP
            >>> stats = ml.clear_cache(older_than_days=30)  # doctest: +SKIP
            >>> print(f"Freed {stats['bytes_freed'] / 1e6:.1f} MB")  # doctest: +SKIP
        """
        from deriva_ml.core.storage import clear_cache as _clear_cache

        return _clear_cache(self.cache_dir, older_than_days, self._logger)

    def get_cache_size(self) -> dict[str, int | float]:
        """Get the current size of the cache directory.

        Returns:
            dict with keys:
                - 'total_bytes': Total size in bytes
                - 'total_mb': Total size in megabytes
                - 'file_count': Number of files
                - 'dir_count': Number of directories

        Example:
            >>> ml = DerivaML('deriva.example.org', 'my_catalog')  # doctest: +SKIP
            >>> size = ml.get_cache_size()  # doctest: +SKIP
            >>> print(f"Cache size: {size['total_mb']:.1f} MB ({size['file_count']} files)")  # doctest: +SKIP
        """
        # Walk with the error-tolerant helper: a single unreadable file
        # (e.g. a permission-denied file in a stale execution dir under the
        # cache root) must not crash the whole size computation, which a bare
        # ``Path.rglob`` would do on the first ``PermissionError``.
        from deriva_ml.core.storage import _dir_stats

        total_bytes, file_count, dir_count = _dir_stats(self.cache_dir)
        return {
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "file_count": file_count,
            "dir_count": dir_count,
        }

    def list_execution_dirs(self) -> list[dict[str, Any]]:
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
            >>> ml = DerivaML('deriva.example.org', 'my_catalog')  # doctest: +SKIP
            >>> dirs = ml.list_execution_dirs()  # doctest: +SKIP
            >>> for d in dirs:  # doctest: +SKIP
            ...     print(f"{d['execution_rid']}: {d['size_mb']:.1f} MB")
        """
        from datetime import datetime

        from deriva_ml.core.storage import _dir_stats
        from deriva_ml.core.upload_layout import upload_root

        results = []
        exec_root = upload_root(self.working_dir) / "execution"

        if not exec_root.exists():
            return results

        for entry in exec_root.iterdir():
            if entry.is_dir():
                # Error-tolerant walk: a single unreadable/permission-denied
                # file in one execution dir must not crash the whole listing
                # (which a bare ``rglob`` + ``stat`` would do). One walk yields
                # both size and file count.
                size_bytes, file_count, _ = _dir_stats(entry)
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
            >>> ml = DerivaML('deriva.example.org', 'my_catalog')  # doctest: +SKIP
            >>> # Clean all execution dirs older than 30 days
            >>> result = ml.clean_execution_dirs(older_than_days=30)  # doctest: +SKIP
            >>> print(f"Freed {result['bytes_freed'] / 1e9:.2f} GB")  # doctest: +SKIP
            >>>
            >>> # Clean all except specific executions
            >>> result = ml.clean_execution_dirs(exclude_rids=['1-ABC', '1-DEF'])  # doctest: +SKIP
        """
        import shutil
        import time

        from deriva_ml.core.upload_layout import upload_root

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

    def list_cached_bags(self) -> "list[CachedBag]":
        """List every dataset bag in the local cache.

        Answers "what bags are currently cached?" without needing to
        know any dataset RID up front. One record per
        (bag, dataset-anchor) pair, most-recently-built first.

        Returns:
            List of :class:`~deriva_ml.core.storage.CachedBag` records
            (empty when nothing is cached).

        Example:
            >>> ml = DerivaML('deriva.example.org', 'my_catalog')  # doctest: +SKIP
            >>> for bag in ml.list_cached_bags():  # doctest: +SKIP
            ...     print(bag.dataset_rid, bag.version, bag.status.value)  # doctest: +SKIP
        """
        from deriva_ml.dataset.bag_cache import BagCache

        with BagCache(self.cache_dir) as cache:
            return cache.list_bags()

    def delete_cached_bag(self, dataset_rid: str, version: str | None = None) -> dict[str, int]:
        """Delete a dataset's cached bag(s).

        Args:
            dataset_rid: Dataset RID whose cached bags to remove.
            version: When given, only the bag for that version;
                ``None`` removes every cached version.

        Returns:
            ``{"bags_removed": n, "bytes_freed": n}``; zeros when the
            dataset isn't cached (idempotent).

        Example:
            >>> ml = DerivaML('deriva.example.org', 'my_catalog')  # doctest: +SKIP
            >>> ml.delete_cached_bag('1-ABC', version='1.2.0')  # doctest: +SKIP
        """
        from deriva_ml.dataset.bag_cache import BagCache

        with BagCache(self.cache_dir) as cache:
            return cache.purge_dataset(dataset_rid, version=version)

    def list_cached_assets(self) -> "list[CachedAsset]":
        """List cached input assets (``assets/{rid}_{md5}`` entries).

        These are written by ``Execution.download_asset(use_cache=True)``
        / ``AssetSpec(cache=True)``.

        Returns:
            List of :class:`~deriva_ml.core.storage.CachedAsset`
            records (empty when no assets are cached).

        Example:
            >>> ml = DerivaML('deriva.example.org', 'my_catalog')  # doctest: +SKIP
            >>> for a in ml.list_cached_assets():  # doctest: +SKIP
            ...     print(a.rid, a.size_bytes)  # doctest: +SKIP
        """
        from deriva_ml.core.storage import list_cached_assets as _list

        return _list(self.cache_dir)

    def delete_cached_asset(self, rid: str, md5: str | None = None) -> dict[str, int]:
        """Delete cached copies of an input asset.

        Args:
            rid: Asset RID whose cache entries to remove.
            md5: When given, only the copy with that checksum;
                ``None`` removes every cached copy.

        Returns:
            ``{"assets_removed": n, "bytes_freed": n}``; zeros when
            nothing matched (idempotent).

        Example:
            >>> ml = DerivaML('deriva.example.org', 'my_catalog')  # doctest: +SKIP
            >>> ml.delete_cached_asset('2-XYZ')  # doctest: +SKIP
        """
        from deriva_ml.core.storage import delete_cached_asset as _delete

        return _delete(self.cache_dir, rid, md5=md5)

    def get_storage_summary(self) -> dict[str, Any]:
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
                - 'bag_count': Number of cached dataset bags
                - 'bag_size_mb': Total size of cached bags in MB
                - 'asset_count': Number of cached input assets
                - 'asset_size_mb': Total size of cached assets in MB

            Note: ``bag_size_mb`` and ``asset_size_mb`` break down
            ``cache_size_mb`` by species — they are subsets of it, not
            additive with it; ``total_size_mb`` remains
            ``cache_size_mb + execution_size_mb``. ``cache_file_count``
            counts only files outside the ``bags/`` subtree (bag file
            counts are not tracked in the index); it is a floor, not a
            full tally.

        Example:
            >>> ml = DerivaML('deriva.example.org', 'my_catalog')  # doctest: +SKIP
            >>> summary = ml.get_storage_summary()  # doctest: +SKIP
            >>> print(f"Total storage: {summary['total_size_mb']:.1f} MB")  # doctest: +SKIP
            >>> print(f"  Cache: {summary['cache_size_mb']:.1f} MB")  # doctest: +SKIP
            >>> print(f"  Executions: {summary['execution_size_mb']:.1f} MB")  # doctest: +SKIP
        """
        from deriva_ml.core.storage import _PROTECTED_CACHE_ENTRIES, _dir_stats

        exec_dirs = self.list_execution_dirs()
        exec_size_mb = sum(d["size_mb"] for d in exec_dirs)

        bags = self.list_cached_bags()
        assets = self.list_cached_assets()
        # A multi-anchor bag appears once per dataset RID in the
        # listing (bag_count counts those entries); size it once per
        # checksum, at the largest anchor's size — an anchor whose
        # Dataset_{rid} directory doesn't exist reports 0/None and
        # must not mask the real on-disk size.
        sizes_by_checksum: dict[str, int] = {}
        for b in bags:
            sizes_by_checksum[b.checksum] = max(sizes_by_checksum.get(b.checksum, 0), b.size_bytes or 0)
        bag_bytes = sum(sizes_by_checksum.values())
        asset_bytes = sum(a.size_bytes for a in assets)

        # Cache total = index-known bag bytes (O(1), no re-walk) + asset bytes
        # (from the asset listing) + any stray content under the cache root
        # that is neither ``bags/``, ``assets/``, nor the bag-index SQLite
        # files. The "other" slice is tiny, so walking it is cheap; this avoids
        # the full-tree ``get_cache_size`` walk that re-measured the multi-GB
        # ``bags/`` subtree on every call. The index files are cache machinery,
        # not cached content, so they are excluded — matching the
        # ``_PROTECTED_CACHE_ENTRIES`` set ``clear_cache`` uses, and keeping the
        # total stable whether or not a listing has lazily created the index.
        other_bytes = other_files = 0
        if self.cache_dir.exists():
            for entry in self.cache_dir.iterdir():
                if entry.name in _PROTECTED_CACHE_ENTRIES:
                    continue
                if entry.is_dir():
                    sub_bytes, sub_files, _ = _dir_stats(entry)
                    other_bytes += sub_bytes
                    other_files += sub_files
                else:
                    try:
                        other_bytes += entry.stat().st_size
                        other_files += 1
                    except (OSError, PermissionError):
                        continue
        cache_bytes = bag_bytes + asset_bytes + other_bytes
        cache_size_mb = cache_bytes / (1024 * 1024)
        # File count excludes the bags/ subtree (not tracked per-bag in the
        # index); assets/ + everything else is counted.
        cache_file_count = sum(a.file_count for a in assets) + other_files

        return {
            "working_dir": str(self.working_dir),
            "cache_dir": str(self.cache_dir),
            "cache_size_mb": cache_size_mb,
            "cache_file_count": cache_file_count,
            "execution_dir_count": len(exec_dirs),
            "execution_size_mb": exec_size_mb,
            "total_size_mb": cache_size_mb + exec_size_mb,
            # Per-species breakdown (spec 2026-06-11)
            "bag_count": len(bags),
            "bag_size_mb": bag_bytes / (1024 * 1024),
            "asset_count": len(assets),
            "asset_size_mb": asset_bytes / (1024 * 1024),
        }

    # Schema integrity validation runs at CI time via the
    # ``deriva-ml-validate-schema`` console script, which compares
    # ``docs/reference/schema.md`` (the canonical source of truth)
    # against the Python definitions in
    # ``deriva_ml.schema.create_schema``. There is no live-catalog
    # validator; one is unnecessary because clones / fresh catalogs
    # are produced from the same Python definitions the doc check
    # validates.

    # Methods moved to mixins:
    # - create_asset, list_assets -> AssetMixin
    # - create_feature, feature_record_class, delete_feature, lookup_feature, feature_values -> FeatureMixin
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
