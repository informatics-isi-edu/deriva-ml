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
    >>> env = get_execution_environment()  # doctest: +SKIP
    >>> print(f"Python version: {env['platform']['python_version']}")  # doctest: +SKIP
    >>> print(f"Operating system: {env['os']['name']}")  # doctest: +SKIP
"""

import importlib.metadata
import importlib.util
import os
import platform
import shutil
import site
import subprocess
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
        >>> env = get_execution_environment()  # doctest: +SKIP
        >>> print(f"Python: {env['platform']['python_version']}")  # doctest: +SKIP
        >>> print(f"OS: {env['platform']['system']}")  # doctest: +SKIP
    """
    return dict(
        imports=get_loaded_modules(),
        os=get_os_info(),
        sys=get_sys_info(),
        sys_path=sys.path,
        site=get_site_info(),
        platform=get_platform_info(),
        gpu=get_gpu_info(),
    )


def get_gpu_info() -> Dict[str, Any]:
    """Best-effort, dependency-free GPU / accelerator snapshot.

    The CPU and architecture are already captured by ``get_platform_info``;
    ``platform`` does not see the GPU. This probes for GPU details without
    adding a hard dependency and without ever raising — a failed probe must
    never break the environment snapshot.

    Probe order:
      1. ``nvidia-smi`` via subprocess (no Python dependency) — model(s),
         count, total memory, driver version.
      2. ``torch.cuda`` *only if torch is importable* (guarded by
         ``importlib.util.find_spec`` so torch is never imported into a
         non-ML process) — device names plus the CUDA / cuDNN versions torch
         was built against, which ``nvidia-smi`` does not report.

    Returns:
        Dict[str, Any]: Always contains ``available`` (bool). When a GPU is
        found, also carries ``source`` and probe-specific fields. When
        nothing succeeds, ``{"available": False}`` (optionally with a
        ``probe_error``). Never raises.

    Example:
        >>> info = get_gpu_info()  # doctest: +SKIP
        >>> info["available"]  # doctest: +SKIP
        True
    """
    info: Dict[str, Any] = {"available": False}

    # --- Probe 1: nvidia-smi (no added dependency) ---------------------
    try:
        smi = shutil.which("nvidia-smi")
        if smi:
            result = subprocess.run(
                [
                    smi,
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                gpus = []
                for line in result.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        gpus.append({"name": parts[0], "memory_total": parts[1], "driver_version": parts[2]})
                if gpus:
                    info = {
                        "available": True,
                        "source": "nvidia-smi",
                        "count": len(gpus),
                        "devices": gpus,
                    }
    except Exception as exc:  # noqa: BLE001 — best-effort, never raise
        info.setdefault("probe_error", f"nvidia-smi: {exc}")

    # --- Probe 2: torch.cuda (only if torch is already installed) ------
    # Adds the CUDA/cuDNN build versions, which nvidia-smi does not report.
    try:
        if importlib.util.find_spec("torch") is not None:
            import torch  # local import — never pulled into a non-ML process

            if torch.cuda.is_available():
                names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                info["available"] = True
                info.setdefault("source", "torch.cuda")
                info["torch"] = {
                    "device_count": torch.cuda.device_count(),
                    "device_names": names,
                    "cuda_version": torch.version.cuda,
                    "cudnn_version": (
                        torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
                    ),
                }
    except Exception as exc:  # noqa: BLE001 — best-effort, never raise
        info.setdefault("probe_error", f"torch: {exc}")

    return info


def get_loaded_modules() -> Dict[str, str]:
    """Gets information about installed Python packages.

    Returns a mapping of package names to their installed versions using
    Python's importlib.metadata.

    Returns:
        Dict[str, str]: Mapping of package names to version strings.

    Example:
        >>> modules = get_loaded_modules()  # doctest: +SKIP
        >>> print(f"NumPy version: {modules.get('numpy')}")  # doctest: +SKIP
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
        >>> info = get_site_info()  # doctest: +SKIP
        >>> print(f"User site-packages: {info['USER_SITE']}")  # doctest: +SKIP
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
        >>> info = get_platform_info()  # doctest: +SKIP
        >>> print(f"OS: {info['system']} {info['release']}")  # doctest: +SKIP
        >>> print(f"Architecture: {info['machine']}")  # doctest: +SKIP
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
        >>> info = get_os_info()  # doctest: +SKIP
        >>> print(f"User: {info.get('login')}")  # doctest: +SKIP
        >>> print(f"Working directory: {info['cwd']}")  # doctest: +SKIP
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
        >>> info = get_sys_info()  # doctest: +SKIP
        >>> print(f"Python path: {info['executable']}")  # doctest: +SKIP
        >>> print(f"Arguments: {info['argv']}")  # doctest: +SKIP
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
