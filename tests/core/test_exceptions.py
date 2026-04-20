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
