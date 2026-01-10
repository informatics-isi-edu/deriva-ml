"""Custom exceptions for the DerivaML package.

This module defines the exception hierarchy for DerivaML. All DerivaML-specific
exceptions inherit from DerivaMLException, making it easy to catch all library
errors with a single except clause.

Exception Hierarchy:
    DerivaMLException (base class)
    ├── DerivaMLInvalidTerm - Invalid vocabulary term
    └── DerivaMLTableTypeError - Wrong table type for operation

Example:
    >>> from deriva_ml.core.exceptions import DerivaMLException, DerivaMLInvalidTerm
    >>> try:
    ...     term = ml.lookup_term("Diagnosis", "invalid_term")
    ... except DerivaMLInvalidTerm as e:
    ...     print(f"Term not found: {e}")
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

    def __init__(self, msg: str = ""):
        super().__init__(msg)
        self._msg = msg


class DerivaMLInvalidTerm(DerivaMLException):
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

    def __init__(self, vocabulary: str, term: str, msg: str = "Term doesn't exist"):
        super().__init__(f"Invalid term {term} in vocabulary {vocabulary}: {msg}.")


class DerivaMLTableTypeError(DerivaMLException):
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

    def __init__(self, table_type: str, table: str):
        super().__init__(f"Table {table} is not of type {table_type}.")