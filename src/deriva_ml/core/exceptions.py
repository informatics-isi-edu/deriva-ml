"""
Custom exceptions used throughout the DerivaML package.
"""


class DerivaMLException(Exception):
    """Exception class specific to DerivaML module.

    Args:
        msg (str): Optional message for the exception.
    """

    def __init__(self, msg=""):
        super().__init__(msg)
        self._msg = msg


class DerivaMLInvalidTerm(DerivaMLException):
    """Exception class for invalid terms in DerivaML controlled vocabulary."""
    def __init__(self, vocabulary, term: str, msg: str = "Term doesn't exist"):
        """Exception indicating undefined term type"""
        super().__init__(f"Invalid term {term} in vocabulary {vocabulary}: {msg}.")

class DerivaMLTableTypeError(DerivaMLException):
    """RID for table is not of correct type."""
    def __init__(self, table_type, table: str):
        """Exception indicating undefined term type"""
        super().__init__(f"Table  {table} is not of type {table_type}.")