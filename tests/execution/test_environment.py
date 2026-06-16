"""Smoke tests for ``deriva_ml.execution.environment`` (audit P1 coverage gap).

The functions in this module are called on every execution and
their output is serialized as a metadata asset. A platform-
specific edge case (e.g., ``os.getlogin()`` raising on a
container with no login session) would crash every execution.
Pre-fix, none of these helpers were tested at all.

These tests pin the basic shape invariants: each helper returns
a dict / int with the documented keys, and the aggregate
``get_execution_environment`` is JSON-serializable end-to-end.

Pure-Python tests; no live catalog required.
"""

from __future__ import annotations

import json

import pytest

from deriva_ml.execution.environment import (
    get_execution_environment,
    get_loaded_modules,
    get_os_info,
    get_platform_info,
    get_site_info,
    get_sys_info,
    get_umask,
)


class TestGetLoadedModules:
    def test_returns_dict_of_str_to_str(self):
        modules = get_loaded_modules()
        assert isinstance(modules, dict)
        for name, version in modules.items():
            assert isinstance(name, str)
            assert isinstance(version, str)

    def test_includes_deriva_ml_itself(self):
        """The function lists installed distributions; deriva-ml is one."""
        modules = get_loaded_modules()
        # Name match is exact-case as importlib.metadata reports it.
        # ``deriva-ml`` is the distribution name in pyproject.toml.
        assert "deriva-ml" in modules


class TestGetOsInfo:
    def test_returns_dict_with_basic_keys(self):
        info = get_os_info()
        assert isinstance(info, dict)
        # The keys vary by platform but the dict should be
        # non-empty on every supported runtime.
        assert info, f"get_os_info returned empty dict — likely a platform edge case: {info}"


class TestGetPlatformInfo:
    def test_returns_dict_with_documented_keys(self):
        info = get_platform_info()
        assert isinstance(info, dict)
        # Documented in the docstring as always-present:
        for key in ("system", "release", "version", "machine", "processor"):
            assert key in info, f"get_platform_info missing key {key!r}: {info}"


class TestGetSiteInfo:
    def test_returns_documented_keys(self):
        info = get_site_info()
        assert isinstance(info, dict)
        for key in ("PREFIXES", "ENABLE_USER_SITE", "USER_SITE", "USER_BASE"):
            assert key in info, f"get_site_info missing key {key!r}: {info}"


class TestGetSysInfo:
    def test_returns_dict(self):
        """``get_sys_info`` returns a non-empty dict of interpreter state."""
        info = get_sys_info()
        assert isinstance(info, dict)
        assert info, "get_sys_info returned empty dict"


class TestGetUmask:
    def test_returns_non_negative_int(self):
        """``get_umask`` returns the current umask without leaving the
        process umask permanently changed.

        Implementation uses ``os.umask(0); os.umask(saved)`` — if a
        regression dropped the restore, every execution would
        permanently clobber the process umask. We can't observe
        side-effects perfectly here (the test runs after the
        function), but we can check the return contract.
        """
        before = get_umask()
        after = get_umask()
        assert isinstance(before, int)
        assert before >= 0
        # Two consecutive calls should return the same value
        # (function doesn't leak the umask).
        assert before == after


class TestGetExecutionEnvironment:
    def test_returns_top_level_keys(self):
        """``get_execution_environment`` aggregates the six sources."""
        env = get_execution_environment()
        assert isinstance(env, dict)
        for key in ("imports", "os", "sys", "sys_path", "site", "platform"):
            assert key in env, (
                f"get_execution_environment missing aggregate key {key!r}: keys present = {sorted(env.keys())}"
            )

    def test_is_json_serializable(self):
        """The aggregate must serialize to JSON — it's saved as a metadata asset.

        ``json.dumps(env, default=str)`` is the actual serialization
        path used downstream; pin that it doesn't raise. Using
        ``default=str`` is the function-under-test's documented
        coping mechanism for non-JSON-serializable values (e.g.,
        Path objects); we mirror it here.
        """
        env = get_execution_environment()
        try:
            json.dumps(env, default=str)
        except TypeError as e:
            pytest.fail(f"get_execution_environment is not JSON-serializable: {e}")
