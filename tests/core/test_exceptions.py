"""Tests for DerivaML exception classes."""

from __future__ import annotations


def test_offline_error_is_configuration_error():
    from deriva_ml.core.exceptions import (
        DerivaMLConfigurationError,
        DerivaMLOfflineError,
    )
    err = DerivaMLOfflineError("create_execution requires online mode")
    assert isinstance(err, DerivaMLConfigurationError)
    assert "create_execution" in str(err)


def test_no_execution_context_error_is_configuration_error():
    from deriva_ml.core.exceptions import (
        DerivaMLConfigurationError,
        DerivaMLNoExecutionContext,
    )
    err = DerivaMLNoExecutionContext("ml.table(...) handles are read-only")
    assert isinstance(err, DerivaMLConfigurationError)


def test_state_inconsistency_error_is_data_error():
    from deriva_ml.core.exceptions import (
        DerivaMLDataError,
        DerivaMLStateInconsistency,
    )
    err = DerivaMLStateInconsistency(
        "SQLite says running, catalog says aborted for EXE-A"
    )
    assert isinstance(err, DerivaMLDataError)
