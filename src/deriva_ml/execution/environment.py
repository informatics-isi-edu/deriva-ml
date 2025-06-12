"""Environment information collection for DerivaML executions.

This module captures detailed information about the execution environment for DerivaML
processes. It collects:

- Python environment: Installed packages, sys.path, site configuration
- Operating system: Platform details, user info, environment variables
- System settings: Locale, encoding, file system details
- Runtime state: Command line arguments, Python interpreter info

This information is stored as execution metadata to ensure reproducibility and
debugging capabilities.

Typical usage example:
    >>> env = get_execution_environment()
    >>> print(f"Python version: {env['platform']['python_version']}")
    >>> print(f"Operating system: {env['os']['name']}")
"""

import importlib.metadata
import locale
import os
import platform
import site
import sys
from typing import Any, Dict


def get_execution_environment() -> Dict[str, Any]:
    """Collects comprehensive information about the execution environment.

    Gathers information about the Python environment, operating system, and runtime
    configuration. This provides a complete snapshot of the execution context.

    Returns:
        Dict[str, Any]: Environment information including:
            - imports: Installed Python packages and versions
            - os: Operating system details and user info
            - sys: Python interpreter configuration
            - sys_path: Module search paths
            - site: Python site configuration
            - platform: Detailed platform information

    Example:
        >>> env = get_execution_environment()
        >>> print(f"Python: {env['platform']['python_version']}")
        >>> print(f"OS: {env['platform']['system']}")
    """
    return dict(
        imports=get_loaded_modules(),
        os=get_os_info(),
        sys=get_sys_info(),
        sys_path=sys.path,
        site=get_site_info(),
        platform=get_platform_info(),
    )


def get_loaded_modules() -> Dict[str, str]:
    """Gets information about installed Python packages.

    Returns a mapping of package names to their installed versions using
    Python's importlib.metadata.

    Returns:
        Dict[str, str]: Mapping of package names to version strings.

    Example:
        >>> modules = get_loaded_modules()
        >>> print(f"NumPy version: {modules.get('numpy')}")
    """
    return {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}


def get_site_info() -> Dict[str, Any]:
    """Gets Python site configuration information.

    Returns information about Python's site configuration, including paths
    for user-specific packages and site-packages directories.

    Returns:
        Dict[str, Any]: Site configuration including:
            - PREFIXES: Installation prefixes
            - ENABLE_USER_SITE: Whether user site-packages is enabled
            - USER_SITE: Path to user site-packages
            - USER_BASE: Base directory for user site-packages

    Example:
        >>> info = get_site_info()
        >>> print(f"User site-packages: {info['USER_SITE']}")
    """
    return {attr: getattr(site, attr) for attr in ["PREFIXES", "ENABLE_USER_SITE", "USER_SITE", "USER_BASE"]}


def get_platform_info() -> Dict[str, Any]:
    """Gets detailed platform information.

    Collects all available platform information using Python's platform module.
    This includes details about the operating system, Python version, and
    hardware architecture.

    Returns:
        Dict[str, Any]: Platform information including:
            - system: Operating system name
            - release: OS release version
            - version: OS version details
            - machine: Hardware architecture
            - processor: Processor type
            - python_version: Python version
            Additional fields vary by platform.

    Example:
        >>> info = get_platform_info()
        >>> print(f"OS: {info['system']} {info['release']}")
        >>> print(f"Architecture: {info['machine']}")
    """
    attributes: list[str] = [
        attr for attr in dir(platform) if (not attr.startswith("_")) and callable(getattr(platform, attr))
    ]
    platform_info: Dict[str, Any] = {}
    for attr in attributes:
        try:
            platform_info[attr] = getattr(platform, attr)()
        except Exception:
            # Not all attributes are available on all platforms.
            continue
    return platform_info


def get_os_info() -> Dict[str, Any]:
    """Gets operating system information.

    Collects information about the operating system environment, including
    user details, process information, and environment variables.

    Returns:
        Dict[str, Any]: OS information including:
            - cwd: Current working directory
            - egid: Effective group ID
            - euid: Effective user ID
            - gid: Real group ID
            - groups: Supplemental group IDs
            - login: User's login name
            - pgrp: Process group ID
            - uid: Real user ID
            - umask: File creation mask
            - name: Operating system name
            - environ: Environment variables

    Example:
        >>> info = get_os_info()
        >>> print(f"User: {info.get('login')}")
        >>> print(f"Working directory: {info['cwd']}")
    """
    values: Dict[str, Any] = {}
    for func in [
        "cwd",
        "egid",
        "euid",
        "gid",
        "groups",
        "login",
        "pgrp",
        "uid",
    ]:
        try:
            values[func] = getattr(os, "get" + func)()
        except (OSError, AttributeError):
            pass
    values["umask"] = oct(get_umask())
    values["name"] = os.name
    values["environ"] = {e: v for e, v in os.environ.items()}
    return values


def get_umask() -> int:
    """Gets the current file creation mask.

    Returns the current umask value in a thread-safe manner by temporarily
    setting it to 0 and then restoring it.

    Returns:
        int: Current umask value.

    Note:
        This implementation is based on the thread-safe approach described at:
        https://stackoverflow.com/questions/53227072/reading-umask-thread-safe
    """
    current_value: int = os.umask(0)
    os.umask(current_value)
    return current_value


def get_sys_info() -> Dict[str, Any]:
    """Gets Python interpreter information.

    Collects information about the Python interpreter configuration and
    runtime environment.

    Returns:
        Dict[str, Any]: System information including:
            - argv: Command line arguments
            - byteorder: Native byte order
            - exec_prefix: Platform-specific Python files location
            - executable: Python interpreter path
            - flags: Python compiler flags
            - float_info: Floating point number info
            - maxsize: Maximum integer size
            - maxunicode: Maximum Unicode code point
            - encoding settings and recursion limits

    Example:
        >>> info = get_sys_info()
        >>> print(f"Python path: {info['executable']}")
        >>> print(f"Arguments: {info['argv']}")
    """
    values: Dict[str, Any] = {}
    for attr in [
        "argv",
        "byteorder",
        "exec_prefix",
        "executable",
        "flags",
        "float_info",
        "maxsize",
        "maxunicode",
    ]:
        values[attr] = getattr(sys, attr)
    for func in [
        "getdefaultencoding",
        "getfilesystemencoding",
        "getrecursionlimit",
    ]:
        try:
            values[func] = getattr(sys, func)()
        except (OSError, AttributeError) as exc:
            values[func] = exc
    return values


def localeconv() -> list[str]:
    """Gets locale convention information.

    Returns formatted strings containing locale-specific formatting information
    for numbers, currency, and other locale-dependent values.

    Returns:
        List[str]: Formatted strings with locale convention information.

    Example:
        >>> info = localeconv()
        >>> for item in info:
        ...     print(item)  # e.g., "decimal_point: ."
    """
    values: list[str] = []
    for key, value in sorted(locale.localeconv().items()):
        if isinstance(value, bytes):
            value = value.decode("ascii", errors="replace")
        if key == "currency_symbol":
            value = repr(value)
        values.append("%s: %s" % (key, value))
    return values


def locale_module() -> list[str]:
    """Gets locale settings information.

    Returns formatted strings containing information about the current locale
    settings for various categories (e.g., numeric formatting, time, etc.).

    Returns:
        List[str]: Formatted strings with locale settings.

    Example:
        >>> info = locale_module()
        >>> for item in info:
        ...     print(item)  # e.g., "LC_NUMERIC: en_US.UTF-8"
    """
    values: list[str] = []
    values.append("getdefaultlocale(): {}".format(locale.getdefaultlocale()))
    for category in [
        "LC_CTYPE",
        "LC_COLLATE",
        "LC_TIME",
        "LC_MONETARY",
        "LC_MESSAGES",
        "LC_NUMERIC",
    ]:
        values.append("getlocale(locale.{}): {}".format(category, locale.getlocale(getattr(locale, category))))
    return values
