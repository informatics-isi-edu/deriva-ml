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