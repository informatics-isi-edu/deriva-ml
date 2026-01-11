"""Custom exceptions for the DerivaML package.

This module defines the exception hierarchy for DerivaML. All DerivaML-specific
exceptions inherit from DerivaMLException, making it easy to catch all library
errors with a single except clause.

Exception Hierarchy:
    DerivaMLException (base class for all DerivaML errors)
    │
    ├── DerivaMLConfigurationError (configuration and initialization)
    │   ├── DerivaMLSchemaError (schema/catalog structure issues)
    │   └── DerivaMLAuthenticationError (authentication failures)
    │
    ├── DerivaMLDataError (data access and validation)
    │   ├── DerivaMLNotFoundError (entity not found)
    │   │   ├── DerivaMLDatasetNotFound (dataset lookup failures)
    │   │   ├── DerivaMLTableNotFound (table lookup failures)
    │   │   └── DerivaMLInvalidTerm (vocabulary term not found)
    │   ├── DerivaMLTableTypeError (wrong table type)
    │   ├── DerivaMLValidationError (data validation failures)
    │   └── DerivaMLCycleError (cycle detected in relationships)
    │
    ├── DerivaMLExecutionError (execution lifecycle)
    │   ├── DerivaMLWorkflowError (workflow issues)
    │   └── DerivaMLUploadError (asset upload failures)
    │
    └── DerivaMLReadOnlyError (write operation on read-only resource)

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


class DerivaMLAuthenticationError(DerivaMLConfigurationError):
    """Exception raised for authentication failures.

    Raised when authentication with the catalog fails or credentials are invalid.

    Example:
        >>> raise DerivaMLAuthenticationError("Failed to authenticate with catalog")
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
