"""Tests for ``deriva_ml.execution.multirun_config`` (audit P1 coverage gap).

Pre-fix, none of the public functions in this module were tested.
A typo or re-architecting would land silently. This file pins the
register / lookup / list / get-all surface plus the per-name
isolation contract.

Pure-Python tests; no live catalog required.
"""

from __future__ import annotations

import pytest

from deriva_ml.execution.multirun_config import (
    MultirunSpec,
    get_all_multirun_configs,
    get_multirun_config,
    list_multirun_configs,
    multirun_config,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    """Clear the module-level registry between tests.

    ``multirun_config`` registers into a process-global dict;
    without resetting between tests, a registration from an
    earlier test leaks into later tests (silent name collisions,
    flaky asserts). The autouse fixture clears both the
    pre-test state (so tests start from empty) and the
    post-test state (so leftover registrations don't bleed into
    the wider test run).
    """
    # NOTE: ``from deriva_ml.execution import multirun_config``
    # at module top imports the FUNCTION (the package __init__
    # re-exports it under the same name as the module). The
    # function shadows the submodule on the package namespace,
    # so we have to reach into ``sys.modules`` to get the
    # actual module object that carries the private registry.
    import sys

    mod = sys.modules["deriva_ml.execution.multirun_config"]

    saved = dict(mod._multirun_registry)
    mod._multirun_registry.clear()
    yield
    mod._multirun_registry.clear()
    mod._multirun_registry.update(saved)


class TestMultirunConfigRegister:
    """``multirun_config()`` registers a spec and returns it."""

    def test_register_returns_a_multirun_spec(self):
        spec = multirun_config(
            "lr_sweep",
            overrides=["model.lr=0.01,0.1"],
            description="LR sweep",
        )
        assert isinstance(spec, MultirunSpec)
        assert spec.name == "lr_sweep"
        assert spec.overrides == ["model.lr=0.01,0.1"]
        assert spec.description == "LR sweep"

    def test_register_lands_in_lookup(self):
        """A registered spec is retrievable by name."""
        multirun_config("epoch_sweep", overrides=["model.epochs=5,10"])
        assert get_multirun_config("epoch_sweep") is not None
        assert get_multirun_config("epoch_sweep").overrides == ["model.epochs=5,10"]

    def test_register_overwrites_existing_name(self):
        """Re-registering the same name replaces the previous spec.

        Pin the semantic so a future change that silently warns
        instead of overwriting (or vice versa) doesn't ship
        unnoticed.
        """
        multirun_config("dup", overrides=["a=1"])
        multirun_config("dup", overrides=["a=2"])
        result = get_multirun_config("dup")
        assert result is not None
        assert result.overrides == ["a=2"]

    def test_register_with_empty_description(self):
        """``description`` defaults to empty string."""
        spec = multirun_config("no_desc", overrides=["x=1"])
        assert spec.description == ""


class TestMultirunConfigLookup:
    """``get_multirun_config()`` returns the spec or None."""

    def test_unknown_name_returns_none(self):
        """Lookup of an unregistered name returns ``None``, doesn't raise."""
        assert get_multirun_config("never_registered") is None

    def test_known_name_returns_spec(self):
        multirun_config("known", overrides=["x=1"])
        spec = get_multirun_config("known")
        assert isinstance(spec, MultirunSpec)
        assert spec.name == "known"


class TestMultirunConfigList:
    """``list_multirun_configs()`` lists registered names."""

    def test_empty_registry_returns_empty_list(self):
        """With an empty registry, the listing is ``[]``."""
        assert list_multirun_configs() == []

    def test_lists_all_registered_names(self):
        multirun_config("a", overrides=["x=1"])
        multirun_config("b", overrides=["y=2"])
        multirun_config("c", overrides=["z=3"])
        assert set(list_multirun_configs()) == {"a", "b", "c"}


class TestMultirunConfigGetAll:
    """``get_all_multirun_configs()`` returns a defensive copy."""

    def test_returns_dict_of_specs(self):
        multirun_config("a", overrides=["x=1"])
        multirun_config("b", overrides=["y=2"])
        all_configs = get_all_multirun_configs()
        assert set(all_configs.keys()) == {"a", "b"}
        for spec in all_configs.values():
            assert isinstance(spec, MultirunSpec)

    def test_returned_dict_is_a_copy(self):
        """Mutating the returned dict doesn't affect the registry.

        The function calls ``dict(self._multirun_registry)`` to
        return a defensive copy; callers shouldn't be able to
        scribble into the registry through the returned reference.
        """
        multirun_config("a", overrides=["x=1"])
        all_configs = get_all_multirun_configs()
        all_configs.pop("a")
        # Registry still has "a".
        assert get_multirun_config("a") is not None
