"""Custom exceptions for the DerivaML package.

This module defines the exception hierarchy for DerivaML. All DerivaML-specific
exceptions inherit from DerivaMLException, making it easy to catch all library
errors with a single except clause.

Exception Hierarchy:
    DerivaMLException (base class for all DerivaML errors)
    │
    ├── DerivaMLConfigurationError (configuration and initialization)
    │   ├── DerivaMLSchemaError (schema/catalog structure issues)
    │   ├── DerivaMLAuthenticationError (authentication failures)
    │   ├── DerivaMLOfflineError (online-only operation in offline mode)
    │   └── DerivaMLNoExecutionContext (write attempted on read-only handle)
    │
    ├── DerivaMLDataError (data access and validation)
    │   ├── DerivaMLNotFoundError (entity not found)
    │   │   ├── DerivaMLDatasetNotFound (dataset lookup failures)
    │   │   ├── DerivaMLTableNotFound (table lookup failures)
    │   │   └── DerivaMLInvalidTerm (vocabulary term not found)
    │   ├── DerivaMLTableTypeError (wrong table type)
    │   ├── DerivaMLValidationError (data validation failures)
    │   ├── DerivaMLCycleError (cycle detected in relationships)
    │   └── DerivaMLStateInconsistency (SQLite/catalog state disagreement)
    │
    ├── DerivaMLExecutionError (execution lifecycle)
    │   ├── DerivaMLWorkflowError (workflow issues)
    │   │   └── DerivaMLDirtyWorkflowError (uncommitted changes)
    │   └── DerivaMLUploadError (asset upload failures)
    │
    ├── DerivaMLReadOnlyError (write operation on read-only resource)
    │
    └── DerivaMLDenormalizeError (denormalization planning errors)
        ├── DerivaMLDenormalizeMultiLeaf
        ├── DerivaMLDenormalizeNoSink
        ├── DerivaMLDenormalizeDownstreamLeaf
        ├── DerivaMLDenormalizeAmbiguousPath
        └── DerivaMLDenormalizeUnrelatedAnchor

Example:
    >>> from deriva_ml.core.exceptions import DerivaMLException, DerivaMLNotFoundError
    >>> try:
    ...     dataset = ml.lookup_dataset("invalid_rid")
    ... except DerivaMLDatasetNotFound as e:
    ...     print(f"Dataset not found: {e}")
    ... except DerivaMLNotFoundError as e:
    ...     print(f"Entity not found: {e}")
    ... except DerivaMLException as e:
    ...     print(f"DerivaML error: {e}")
"""


class DerivaMLException(Exception):
    """Base exception class for all DerivaML errors.

    This is the root exception for all DerivaML-specific errors. Catching this
    exception will catch any error raised by the DerivaML library.

    Attributes:
        _msg: The error message stored for later access.

    Args:
        msg: Descriptive error message. Defaults to empty string.

    Example:
        >>> raise DerivaMLException("Failed to connect to catalog")
        DerivaMLException: Failed to connect to catalog
    """

    def __init__(self, msg: str = "") -> None:
        super().__init__(msg)
        self._msg = msg


# =============================================================================
# Configuration and Initialization Errors
# =============================================================================


class DerivaMLConfigurationError(DerivaMLException):
    """Exception raised for configuration and initialization errors.

    Raised when there are issues with DerivaML configuration, catalog
    initialization, or schema setup.

    Example:
        >>> raise DerivaMLConfigurationError("Invalid catalog configuration")
    """

    pass


class DerivaMLSchemaError(DerivaMLConfigurationError):
    """Exception raised for schema or catalog structure issues.

    Raised when the catalog schema is invalid, missing required tables,
    or has structural problems that prevent normal operation.

    Example:
        >>> raise DerivaMLSchemaError("Ambiguous domain schema: ['Schema1', 'Schema2']")
    """

    pass


class DerivaMLSchemaRefreshBlocked(DerivaMLConfigurationError):
    """Raised when ``refresh_schema()`` is called with staged work in the workspace.

    The caller should drain the workspace first (``ml.upload_pending()``)
    or call ``refresh_schema(force=True)`` to discard local state.
    Draining is the safer choice — a forced refresh may leave rows
    whose metadata references columns or types no longer in the new
    schema, causing catalog-insert failures on the next upload.

    Example:
        >>> raise DerivaMLSchemaRefreshBlocked(
        ...     "refresh_schema requires a drained workspace; 3 pending rows"
        ... )
    """

    pass


class DerivaMLSchemaPinned(DerivaMLConfigurationError):
    """Raised when ``refresh_schema()`` is called on a pinned cache.

    The cache has been explicitly pinned via ``pin_schema()``. Call
    ``unpin_schema()`` first if you really want to refresh. Note:
    ``force=True`` does NOT bypass a pin — it only bypasses the
    pending-rows guard.

    Example:
        >>> raise DerivaMLSchemaPinned(
        ...     "refresh_schema refused: cache is pinned at snapshot s0"
        ... )
    """

    pass


class DerivaMLAuthenticationError(DerivaMLConfigurationError):
    """Exception raised for authentication failures.

    Raised when authentication with the catalog fails or credentials are invalid.

    Example:
        >>> raise DerivaMLAuthenticationError("Failed to authenticate with catalog")
    """

    pass


class DerivaMLOfflineError(DerivaMLConfigurationError):
    """Exception raised when an online-only operation is attempted in offline mode.

    The DerivaML instance was constructed with ``mode=ConnectionMode.offline``
    but the caller invoked an operation that requires server contact — most
    commonly ``create_execution``, which needs a server-assigned Execution RID.

    Example:
        Creating an execution requires an online mode because the
        Execution RID must be server-assigned::

            >>> ml = DerivaML(..., mode=ConnectionMode.offline)
            >>> ml.create_execution(config)
            Traceback (most recent call last):
                ...
            DerivaMLOfflineError: create_execution requires online mode
    """

    pass


class DerivaMLNoExecutionContext(DerivaMLConfigurationError):
    """Exception raised when an execution-scoped operation is attempted without an execution context.

    Handles returned by ``ml.table(name)`` are read-only — useful for schema
    introspection — but their ``.insert(...)`` and asset-file methods raise
    this exception. Use ``exe.table(name)`` to get a handle bound to an
    execution that permits writes.

    Example:
        Calling a write method on a read-only handle raises this error::

            >>> handle = ml.table("Subject")
            >>> handle.record_class()              # OK
            >>> handle.insert({"Name": "x"})       # raises
            Traceback (most recent call last):
                ...
            DerivaMLNoExecutionContext: ml.table() handles are read-only; use exe.table() for writes
    """

    pass


# =============================================================================
# Data Access and Validation Errors
# =============================================================================


class DerivaMLDataError(DerivaMLException):
    """Exception raised for data access and validation issues.

    Base class for errors related to data lookup, validation, and integrity.

    Example:
        >>> raise DerivaMLDataError("Invalid data format")
    """

    pass


class DerivaMLNotFoundError(DerivaMLDataError):
    """Exception raised when an entity cannot be found.

    Raised when a lookup operation fails to find the requested entity
    (dataset, table, term, etc.) in the catalog or bag.

    Example:
        >>> raise DerivaMLNotFoundError("Entity '1-ABC' not found in catalog")
    """

    pass


class DerivaMLDatasetNotFound(DerivaMLNotFoundError):
    """Exception raised when a dataset cannot be found.

    Raised when attempting to look up a dataset that doesn't exist in the
    catalog or downloaded bag.

    Args:
        dataset_rid: The RID of the dataset that was not found.
        msg: Additional context. Defaults to "Dataset not found".

    Example:
        >>> raise DerivaMLDatasetNotFound("1-ABC")
        DerivaMLDatasetNotFound: Dataset 1-ABC not found
    """

    def __init__(self, dataset_rid: str, msg: str = "Dataset not found") -> None:
        super().__init__(f"{msg}: {dataset_rid}")
        self.dataset_rid = dataset_rid


class DerivaMLTableNotFound(DerivaMLNotFoundError):
    """Exception raised when a table cannot be found.

    Raised when attempting to access a table that doesn't exist in the
    catalog schema or downloaded bag.

    Args:
        table_name: The name of the table that was not found.
        msg: Additional context. Defaults to "Table not found".

    Example:
        >>> raise DerivaMLTableNotFound("MyTable")
        DerivaMLTableNotFound: Table not found: MyTable
    """

    def __init__(self, table_name: str, msg: str = "Table not found") -> None:
        super().__init__(f"{msg}: {table_name}")
        self.table_name = table_name


class DerivaMLInvalidTerm(DerivaMLNotFoundError):
    """Exception raised when a vocabulary term is not found or invalid.

    Raised when attempting to look up or use a term that doesn't exist in
    a controlled vocabulary table, or when a term name/synonym cannot be resolved.

    Args:
        vocabulary: Name of the vocabulary table being searched.
        term: The term name that was not found.
        msg: Additional context about the error. Defaults to "Term doesn't exist".

    Example:
        >>> raise DerivaMLInvalidTerm("Diagnosis", "unknown_condition")
        DerivaMLInvalidTerm: Invalid term unknown_condition in vocabulary Diagnosis: Term doesn't exist.
    """

    def __init__(self, vocabulary: str, term: str, msg: str = "Term doesn't exist") -> None:
        super().__init__(f"Invalid term {term} in vocabulary {vocabulary}: {msg}.")
        self.vocabulary = vocabulary
        self.term = term


class DerivaMLTableTypeError(DerivaMLDataError):
    """Exception raised when a RID or table is not of the expected type.

    Raised when an operation requires a specific table type (e.g., Dataset,
    Execution) but receives a RID or table reference of a different type.

    Args:
        table_type: The expected table type (e.g., "Dataset", "Execution").
        table: The actual table name or RID that was provided.

    Example:
        >>> raise DerivaMLTableTypeError("Dataset", "1-ABC123")
        DerivaMLTableTypeError: Table 1-ABC123 is not of type Dataset.
    """

    def __init__(self, table_type: str, table: str) -> None:
        super().__init__(f"Table {table} is not of type {table_type}.")
        self.table_type = table_type
        self.table = table


class DerivaMLValidationError(DerivaMLDataError):
    """Exception raised when data validation fails.

    Raised when input data fails validation, such as invalid RID format,
    mismatched metadata, or constraint violations.

    Example:
        >>> raise DerivaMLValidationError("Invalid RID format: ABC")
    """

    pass


class DerivaMLCycleError(DerivaMLDataError):
    """Exception raised when a cycle is detected in relationships.

    Raised when creating dataset hierarchies or other relationships that
    would result in a circular dependency.

    Args:
        cycle_nodes: List of nodes involved in the cycle.
        msg: Additional context. Defaults to "Cycle detected".

    Example:
        >>> raise DerivaMLCycleError(["Dataset1", "Dataset2", "Dataset1"])
    """

    def __init__(self, cycle_nodes: list[str], msg: str = "Cycle detected") -> None:
        super().__init__(f"{msg}: {cycle_nodes}")
        self.cycle_nodes = cycle_nodes


class DerivaMLStateInconsistency(DerivaMLDataError):
    """Exception raised when workspace SQLite state and catalog state disagree in an unresolvable way.

    The six disagreement cases enumerated in spec §2.2 are handled automatically
    by the reconciliation logic (see ``state_machine.reconcile_with_catalog``);
    anything outside those rules surfaces as this exception with enough
    information for a human to intervene.

    Example:
        A catalog-side delete of an in-flight execution produces this error::

            >>> exe = ml.resume_execution("EXE-A")
            Traceback (most recent call last):
                ...
            DerivaMLStateInconsistency: Execution EXE-A: SQLite status 'running' but catalog returned no Execution row
    """

    pass


# =============================================================================
# Execution Lifecycle Errors
# =============================================================================


class DerivaMLExecutionError(DerivaMLException):
    """Exception raised for execution lifecycle issues.

    Base class for errors related to workflow execution, asset management,
    and provenance tracking.

    Example:
        >>> raise DerivaMLExecutionError("Execution failed to initialize")
    """

    pass


class DerivaMLWorkflowError(DerivaMLExecutionError):
    """Exception raised for workflow-related issues.

    Raised when there are problems with workflow lookup, creation, or
    Git integration for workflow tracking.

    Example:
        >>> raise DerivaMLWorkflowError("Not executing in a Git repository")
    """

    pass


class DerivaMLDirtyWorkflowError(DerivaMLWorkflowError):
    """Exception raised when workflow code has uncommitted changes.

    DerivaML requires code to be committed before execution for provenance
    tracking. Running with uncommitted changes means the execution record
    cannot reliably link back to the source code.

    Use ``allow_dirty=True`` in the API or ``--allow-dirty`` on the CLI
    to override this check when debugging or iterating.

    Args:
        path: Path to the file with uncommitted changes.

    Example:
        >>> raise DerivaMLDirtyWorkflowError("src/models/train.py")
        DerivaMLDirtyWorkflowError: File src/models/train.py has uncommitted changes. ...
    """

    def __init__(self, path: str) -> None:
        super().__init__(
            f"File {path} has uncommitted changes. Commit before running, or use --allow-dirty to override."
        )
        self.path = path


class DerivaMLUploadError(DerivaMLExecutionError):
    """Exception raised for asset upload failures.

    Raised when uploading assets to the catalog fails, including file
    uploads, metadata insertion, and provenance recording.

    Example:
        >>> raise DerivaMLUploadError("Failed to upload execution assets")
    """

    pass


# =============================================================================
# Read-Only Resource Errors
# =============================================================================


class DerivaMLReadOnlyError(DerivaMLException):
    """Exception raised when attempting write operations on read-only resources.

    Raised when attempting to modify data in a downloaded bag or other
    read-only context where write operations are not supported.

    Example:
        >>> raise DerivaMLReadOnlyError("Cannot create datasets in a downloaded bag")
    """

    pass


# =============================================================================
# Denormalization Planning Errors
# =============================================================================


class DerivaMLDenormalizeError(DerivaMLException):
    """Base class for denormalization errors.

    All errors raised by :class:`~deriva_ml.local_db.denormalizer.Denormalizer`
    and related planning functions are instances of this class.

    Example:
        >>> raise DerivaMLDenormalizeError("Planner failed")
    """


class DerivaMLDenormalizeMultiLeaf(DerivaMLDenormalizeError):
    """Multiple candidate tables for ``row_per`` — ambiguous leaf.

    Raised when Rule 2 auto-inference finds more than one sink in
    ``include_tables`` — i.e., multiple tables tie for "deepest in the
    FK graph." The user must specify ``row_per`` explicitly to resolve.

    Attributes:
        candidates: list of table names that all qualify as sinks.
        include_tables: the ``include_tables`` argument that triggered
            the ambiguity, for reference.

    Example:
        >>> try:
        ...     d.as_dataframe(["Dataset", "Subject"])
        ... except DerivaMLDenormalizeMultiLeaf as e:
        ...     print(f"Pick one of {e.candidates} as row_per")
        ...     # Then retry: d.as_dataframe(..., row_per="Subject")
    """

    def __init__(self, candidates: list[str], include_tables: list[str]) -> None:
        self.candidates = list(candidates)
        self.include_tables = list(include_tables)
        super().__init__(
            f"Multiple candidates for row_per: {candidates}. "
            f"Specify row_per=... explicitly. "
            f"(include_tables={include_tables})"
        )


class DerivaMLDenormalizeNoSink(DerivaMLDenormalizeError):
    """No sink found in the FK subgraph — cycle detected.

    Raised when every table in ``include_tables`` has an outbound FK to
    another table in the set, forming a cycle. Pathological — rare in
    real schemas.

    Args:
        msg: Descriptive error message. Should identify the tables
            forming the cycle.

    Example:
        >>> raise DerivaMLDenormalizeNoSink(
        ...     "Cycle in FK graph between tables A, B, C"
        ... )
    """


class DerivaMLDenormalizeDownstreamLeaf(DerivaMLDenormalizeError):
    """Explicit ``row_per`` conflicts with a downstream table in ``include_tables``.

    Raised when the user specifies ``row_per=X`` but another table in
    ``include_tables`` is downstream of X via FK (would require aggregation).

    Attributes:
        row_per: the explicit row_per value.
        downstream_tables: tables downstream of row_per that can't be hoisted.
    """

    def __init__(self, row_per: str, downstream_tables: list[str]) -> None:
        self.row_per = row_per
        self.downstream_tables = list(downstream_tables)
        super().__init__(
            f"Table(s) {downstream_tables} are downstream of row_per={row_per!r}. "
            f"One row per {row_per} would require aggregating multiple rows of "
            f"{downstream_tables} — aggregation is not yet supported. "
            f"Drop row_per to get one row per {downstream_tables}, or remove "
            f"{downstream_tables} from include_tables."
        )


class DerivaMLDenormalizeAmbiguousPath(DerivaMLDenormalizeError):
    """Multiple FK paths between two requested tables — can't silently choose.

    Raised when Rule 6 detects two or more distinct FK paths between
    ``row_per`` and another requested / via table. Silent path selection
    is rejected by design — the result shape would be materially
    different depending on which path is chosen, and callers should be
    explicit. Disambiguate by adding intermediates to ``include_tables``
    (their columns are included) or to ``via=`` (path-only, columns
    excluded).

    Attributes:
        from_table: the ``row_per`` table name (the "anchor" of the
            ambiguity).
        to_table: the requested table with multiple paths.
        paths: list of path descriptions — each is a list of table
            names from ``from_table`` to ``to_table``.
        suggested_intermediates: tables that appear in at least one
            path but not in ``include_tables`` — any of these could be
            named in ``include_tables`` or ``via`` to force a choice.

    Example:
        >>> try:
        ...     d.as_dataframe(["Image", "Subject"])  # diamond schema
        ... except DerivaMLDenormalizeAmbiguousPath as e:
        ...     for p in e.paths:
        ...         print(" → ".join(p))
        ...     # Retry routing explicitly through Observation:
        ...     df = d.as_dataframe(
        ...         ["Image", "Subject"], via=e.suggested_intermediates[:1]
        ...     )
    """

    def __init__(
        self,
        from_table: str,
        to_table: str,
        paths: list[list[str]],
        suggested_intermediates: list[str],
    ) -> None:
        self.from_table = from_table
        self.to_table = to_table
        self.paths = [list(p) for p in paths]
        self.suggested_intermediates = list(suggested_intermediates)
        path_strs = ["\n    " + " → ".join(p) for p in paths]
        super().__init__(
            f"Multiple FK paths between {from_table!r} and {to_table!r}:"
            f"{''.join(path_strs)}\n"
            f"Resolve by one of:\n"
            f"  • Add an intermediate to include_tables "
            f"(its columns will be in output): {suggested_intermediates}\n"
            f"  • Add an intermediate to via= (path-only, no columns): "
            f"{suggested_intermediates}\n"
            f"  • Narrow include_tables so only one path is valid."
        )


class DerivaMLDenormalizeUnrelatedAnchor(DerivaMLDenormalizeError):
    """Anchor has no FK path to any table in ``include_tables ∪ via``.

    Raised when Rule 8 detects anchors whose table has no FK
    relationship to any requested table — those anchors would
    contribute nothing to the output, which is almost always a mistake
    (wrong dataset passed, stale table name, etc.). Pass
    ``ignore_unrelated_anchors=True`` to silently drop them if the
    heterogeneity is intentional.

    Note: this is distinct from Rule 7 case 5 (table has an FK path
    into ``include_tables ∪ via`` but the specific anchor RIDs don't
    reach ``row_per``). Case 5 anchors are silently dropped regardless
    of the flag — only case 6 (no path at all) raises this error.

    Attributes:
        unrelated_tables: tables of the unrelated anchors.
        include_tables: the ``include_tables`` argument for reference.

    Example:
        >>> try:
        ...     d.as_dataframe(["Image", "Subject"])  # dataset has stray types
        ... except DerivaMLDenormalizeUnrelatedAnchor as e:
        ...     print(f"Dataset has unrelated members: {e.unrelated_tables}")
        ...     # Retry, dropping them:
        ...     df = d.as_dataframe(
        ...         ["Image", "Subject"], ignore_unrelated_anchors=True
        ...     )
    """

    def __init__(
        self,
        unrelated_tables: list[str],
        include_tables: list[str],
    ) -> None:
        self.unrelated_tables = list(unrelated_tables)
        self.include_tables = list(include_tables)
        super().__init__(
            f"Anchors of table(s) {unrelated_tables} have no FK path to any "
            f"table in include_tables={include_tables}. They would contribute "
            f"nothing to the output.\n"
            f"Options:\n"
            f"  • Remove these anchors from the anchor set.\n"
            f"  • Add {unrelated_tables} (or a linking table) to include_tables.\n"
            f"  • Pass ignore_unrelated_anchors=True to silently drop them."
        )
