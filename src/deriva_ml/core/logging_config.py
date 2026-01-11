"""Centralized logging configuration for DerivaML.

This module provides a consistent logging setup for all DerivaML components.
It configures the 'deriva_ml' logger namespace and provides utilities for
adjusting log levels throughout the library.

The module provides:
    - get_logger(): Get the standard DerivaML logger
    - configure_logging(): Set up logging with specified levels
    - LoggerMixin: Mixin class providing _logger attribute

Example:
    >>> from deriva_ml.core.logging_config import configure_logging, get_logger
    >>> import logging
    >>>
    >>> configure_logging(level=logging.DEBUG)
    >>> logger = get_logger()
    >>> logger.info("DerivaML initialized")
"""

import logging
from typing import Any

# The standard logger name used throughout DerivaML
LOGGER_NAME = "deriva_ml"

# Default logging format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Related library loggers that should be configured together
RELATED_LOGGERS = [
    "root",
    "bagit",
    "bdbag",
    "deriva",
]


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a DerivaML logger.

    Args:
        name: Optional sub-logger name. If provided, returns a child logger
              under the deriva_ml namespace (e.g., 'deriva_ml.dataset').
              If None, returns the main deriva_ml logger.

    Returns:
        The configured logger instance.

    Example:
        >>> logger = get_logger()  # Main deriva_ml logger
        >>> dataset_logger = get_logger("dataset")  # deriva_ml.dataset
    """
    if name is None:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(
    level: int = logging.WARNING,
    deriva_level: int | None = None,
    format_string: str = DEFAULT_FORMAT,
    handler: logging.Handler | None = None,
) -> logging.Logger:
    """Configure logging for DerivaML and related libraries.

    This function sets up the logging configuration for DerivaML and
    related libraries (bagit, bdbag, deriva). It should be called once
    during application initialization.

    Args:
        level: Log level for the deriva_ml logger. Defaults to WARNING.
        deriva_level: Log level for related libraries (bagit, bdbag, deriva).
                     If None, uses the same level as 'level'.
        format_string: Format string for log messages.
        handler: Optional handler to add to the logger. If None, uses
                StreamHandler with the specified format.

    Returns:
        The configured deriva_ml logger.

    Example:
        >>> import logging
        >>> configure_logging(level=logging.DEBUG)
        >>> configure_logging(
        ...     level=logging.INFO,
        ...     deriva_level=logging.WARNING,  # Less verbose for libs
        ... )
    """
    if deriva_level is None:
        deriva_level = level

    # Configure main DerivaML logger
    logger = get_logger()
    logger.setLevel(level)

    # Add handler if not already present
    if not logger.handlers:
        if handler is None:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(handler)

    # Configure related library loggers
    for logger_name in RELATED_LOGGERS:
        logging.getLogger(logger_name).setLevel(deriva_level)

    return logger


def apply_logger_overrides(overrides: dict[str, Any]) -> None:
    """Apply logger level overrides from a configuration dictionary.

    This is used for compatibility with deriva's DEFAULT_LOGGER_OVERRIDES
    pattern, allowing fine-grained control over specific loggers.

    Args:
        overrides: Dictionary mapping logger names to log levels.

    Example:
        >>> apply_logger_overrides({
        ...     "deriva": logging.WARNING,
        ...     "bdbag": logging.ERROR,
        ... })
    """
    for name, level in overrides.items():
        logging.getLogger(name).setLevel(level)


class LoggerMixin:
    """Mixin class that provides a _logger attribute.

    Classes that inherit from this mixin get a _logger property that
    returns a child logger under the deriva_ml namespace, named after
    the class.

    Example:
        >>> class MyProcessor(LoggerMixin):
        ...     def process(self):
        ...         self._logger.info("Processing started")
        ...
        >>> # Logs to 'deriva_ml.MyProcessor'
    """

    @property
    def _logger(self) -> logging.Logger:
        """Get the logger for this class."""
        return get_logger(self.__class__.__name__)


__all__ = [
    "LOGGER_NAME",
    "get_logger",
    "configure_logging",
    "apply_logger_overrides",
    "LoggerMixin",
]
