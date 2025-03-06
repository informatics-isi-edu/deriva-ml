""" Module that captures details of execution environment for use as execution metadata"""

import locale
import os
import platform
import sys

import site
import importlib


def get_execution_environment() -> dict:
    return dict(
        imports=get_loaded_modules(),
        os=get_os_info(),
        sys=get_sys_info(),
        sys_path=sys.path,
        site=get_site_info(),
        platform=get_platform_info(),
    )


def get_loaded_modules():
    return {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
    }


def get_site_info():
    return {
        attr: getattr(site, attr)
        for attr in ["PREFIXES", "ENABLE_USER_SITE", "USER_SITE", "USER_BASE"]
    }


def get_platform_info():
    """
    Returns all available attributes from the platform module.
    """
    attributes = [
        attr
        for attr in dir(platform)
        if (not attr.startswith("_")) and callable(getattr(platform, attr))
    ]
    platform_info = {}
    for attr in attributes:
        try:
            platform_info[attr] = getattr(platform, attr)()
        except Exception:
            # Not all attributes are available on all platforms.
            continue
    return platform_info


def get_os_info():
    values = {}
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


def get_umask():
    # https://stackoverflow.com/questions/53227072/reading-umask-thread-safe
    current_value = os.umask(0)
    os.umask(current_value)
    return current_value


def get_sys_info():
    values = {}
    for attr in [
        "argv",
        "byteorder",
        "exec_prefix",
        "executable",
        "flags",
        "float_info",
        #      "maxint",
        "maxsize",
        "maxunicode",
        #   "meta_path",
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


def localeconv():
    values = []
    for key, value in sorted(locale.localeconv().items()):
        if isinstance(value, bytes):
            value = value.decode("ascii", errors="replace")
        if key == "currency_symbol":
            value = repr(value)
        values.append("%s: %s" % (key, value))
    return values


def locale_module():
    values = []
    values.append("getdefaultlocale(): {}".format(locale.getdefaultlocale()))
    for category in [
        "LC_CTYPE",
        "LC_COLLATE",
        "LC_TIME",
        "LC_MONETARY",
        "LC_MESSAGES",
        "LC_NUMERIC",
    ]:
        values.append(
            "getlocale(locale.{}): {}".format(
                category, locale.getlocale(getattr(locale, category))
            )
        )
    return values
