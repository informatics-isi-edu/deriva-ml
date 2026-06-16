"""Tests for shared FK-traversal constants."""

from __future__ import annotations


def test_provenance_terminal_tables_value():
    from deriva_ml.core.constants import ML_SCHEMA, PROVENANCE_TERMINAL_TABLES

    assert PROVENANCE_TERMINAL_TABLES == frozenset({(ML_SCHEMA, "Execution"), (ML_SCHEMA, "Workflow")})


def test_provenance_terminal_tables_exported():
    import deriva_ml.core.constants as c

    assert "PROVENANCE_TERMINAL_TABLES" in c.__all__


def test_clone_and_bag_builder_share_terminal_tables():
    """clone_via_bag and bag_builder must use the SAME terminal set —
    a divergence is what let the dataset-export path miss this guard.

    Note: importlib.import_module is required here because
    ``deriva_ml.catalog.__init__`` re-exports the ``clone_via_bag``
    *function* under the same name, which shadows the submodule in the
    ``deriva_ml.catalog`` namespace.  ``import_module`` bypasses that
    namespace and always returns the actual module object.
    """
    import importlib

    from deriva_ml.core.constants import PROVENANCE_TERMINAL_TABLES

    cvb = importlib.import_module("deriva_ml.catalog.clone_via_bag")
    assert cvb._DEFAULT_TERMINAL_TABLES == set(PROVENANCE_TERMINAL_TABLES)
