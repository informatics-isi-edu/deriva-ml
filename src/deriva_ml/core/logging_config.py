"""Centralized logging configuration for DerivaML.

This module provides a consistent logging setup for all DerivaML components.
It configures the 'deriva_ml' logger namespace and related library loggers
(deriva-py, bdbag, bagit) without impacting the calling application's
logging configuration.

Key design principles:
    - DerivaML configures only its own logger and related library loggers
    - Never calls logging.basicConfig() to avoid affecting the root logger
    - Respects Hydra's logging configuration when running under Hydra
    - Hydra loggers follow the deriva_ml logging level

Public API:
    - :func:`get_logger`: Return the main ``deriva_ml`` logger (or a child).
    - :func:`configure_logging`: Set log levels for DerivaML and related libs.
    - :func:`is_hydra_initialized`: Test whether a Hydra context is active.

Internal helpers (not part of the public API):
    - :func:`_apply_logger_overrides`: Apply per-logger level overrides dict.

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

# Default logging format (only used when adding handlers outside Hydra context)
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Related library loggers whose levels should follow deriva_level
# These are libraries that DerivaML uses internally
DERIVA_LOGGERS = [
    "deriva",
    "bagit",
    "bdbag",
]

# Hydra loggers whose levels should follow the deriva_ml level
HYDRA_LOGGERS = [
    "hydra",
    "hydra.core",
    "hydra.utils",
    "omegaconf",
]


def is_hydra_initialized() -> bool:
    """Check if running within an initialized Hydra context.

    This is used to determine whether Hydra is managing logging configuration.
    When Hydra is initialized, we avoid adding handlers or calling basicConfig
    since Hydra has already configured logging via dictConfig.

    Returns:
        True if Hydra's GlobalHydra is initialized, False otherwise.

    Example:
        >>> if is_hydra_initialized():
        ...     # Hydra is managing logging
        ...     pass
    """
    try:
        from hydra.core.global_hydra import GlobalHydra

        return GlobalHydra.instance().is_initialized()
    except (ImportError, Exception):
        return False


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a DerivaML logger.

    Three name forms are accepted:

    1. ``None`` — returns the main ``deriva_ml`` logger.
    2. A short suffix (no dots), e.g. ``get_logger("dataset")`` —
       returns ``deriva_ml.dataset``.
    3. A full module ``__name__`` (with or without the
       ``deriva_ml.`` prefix), e.g.
       ``get_logger(__name__)`` from inside
       ``deriva_ml/dataset/dataset.py`` — returns
       ``deriva_ml.dataset.dataset``. Names that already start
       with ``deriva_ml`` are used as-is; the bare string
       ``"deriva_ml"`` is treated identically to ``None``.

    Form (3) is the canonical project-wide pattern: every module
    writes ``logger = get_logger(__name__)`` so log messages
    carry their source module in the hierarchy.

    Args:
        name: Sub-logger name. See above for accepted forms.

    Returns:
        The configured logger instance.

    Example:
        >>> logger = get_logger()                       # deriva_ml
        >>> get_logger("dataset").name                  # deriva_ml.dataset
        'deriva_ml.dataset'
        >>> get_logger("deriva_ml.dataset").name        # already-prefixed
        'deriva_ml.dataset'
        >>> get_logger("deriva_ml").name                # bare root
        'deriva_ml'
    """
    if name is None or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(
    level: int = logging.WARNING,
    deriva_level: int | None = None,
    format_string: str = DEFAULT_FORMAT,
    handler: logging.Handler | None = None,
) -> logging.Logger:
    """Configure logging for DerivaML and related libraries.

    This function sets up logging levels for DerivaML, related libraries
    (deriva-py, bdbag, bagit), and Hydra loggers. It is designed to:

    1. Configure only specific logger namespaces, not the root logger
    2. Respect Hydra's logging configuration when running under Hydra
    3. Allow deriva-py libraries to have a separate logging level

    The logging level hierarchy:
        - deriva_ml logger: uses `level`
        - Hydra loggers: follow `level` (deriva_ml level)
        - Deriva/bdbag/bagit loggers: use `deriva_level` (defaults to `level`)

    When running under Hydra:
        - Only sets log levels on specific loggers
        - Does NOT add handlers (Hydra has already configured them)
        - Does NOT call basicConfig()

    When running standalone (no Hydra):
        - Sets log levels on specific loggers
        - Adds a StreamHandler to deriva_ml logger if none exists
        - Still does NOT touch the root logger or call basicConfig()

    Args:
        level: Log level for deriva_ml and Hydra loggers. Defaults to WARNING.
        deriva_level: Log level for deriva-py libraries (deriva, bagit, bdbag).
                     If None, uses the same level as `level`.
        format_string: Format string for log messages (used only when adding
                      handlers outside Hydra context).
        handler: Optional handler to add to the deriva_ml logger. If None and
                not running under Hydra, uses StreamHandler with format_string.

    Returns:
        The configured deriva_ml logger.

    Example:
        >>> import logging
        >>> # Same level for everything
        >>> configure_logging(level=logging.DEBUG)
        >>>
        >>> # Verbose DerivaML, quieter deriva-py libraries
        >>> configure_logging(
        ...     level=logging.INFO,
        ...     deriva_level=logging.WARNING,
        ... )
    """
    if deriva_level is None:
        deriva_level = level

    # Configure main DerivaML logger
    logger = get_logger()
    logger.setLevel(level)

    # Configure Hydra loggers to follow deriva_ml level
    for logger_name in HYDRA_LOGGERS:
        logging.getLogger(logger_name).setLevel(level)

    # Configure deriva-py and related library loggers
    for logger_name in DERIVA_LOGGERS:
        logging.getLogger(logger_name).setLevel(deriva_level)

    # Only add handlers if not running under Hydra
    # Hydra configures handlers via dictConfig, we don't want to duplicate
    if not is_hydra_initialized():
        if not logger.handlers:
            if handler is None:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter(format_string))
            logger.addHandler(handler)

    return logger


def _apply_logger_overrides(overrides: dict[str, Any]) -> None:
    """Apply logger level overrides from a configuration dictionary.

    This is an internal helper used for compatibility with deriva's
    DEFAULT_LOGGER_OVERRIDES pattern, allowing fine-grained control over
    specific loggers. Not part of the public API.

    Args:
        overrides: Dictionary mapping logger names to log levels.
    """
    for name, level_value in overrides.items():
        logging.getLogger(name).setLevel(level_value)


__all__ = [
    "LOGGER_NAME",
    "get_logger",
    "configure_logging",
    "is_hydra_initialized",
]
