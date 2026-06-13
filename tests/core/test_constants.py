"""Tests for shared FK-traversal constants."""

from __future__ import annotations


def test_provenance_terminal_tables_value():
    from deriva_ml.core.constants import ML_SCHEMA, PROVENANCE_TERMINAL_TABLES

    assert PROVENANCE_TERMINAL_TABLES == frozenset(
        {(ML_SCHEMA, "Execution"), (ML_SCHEMA, "Workflow")}
    )


def test_provenance_terminal_tables_exported():
    import deriva_ml.core.constants as c

    assert "PROVENANCE_TERMINAL_TABLES" in c.__all__
