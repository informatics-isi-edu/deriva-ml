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


def test_state_inconsistency_error_is_data_error():
    from deriva_ml.core.exceptions import (
        DerivaMLDataError,
        DerivaMLStateInconsistency,
    )

    err = DerivaMLStateInconsistency("SQLite says running, catalog says aborted for EXE-A")
    assert isinstance(err, DerivaMLDataError)


def test_derivaml_schema_pinned_inherits_configuration_error():
    from deriva_ml.core.exceptions import (
        DerivaMLConfigurationError,
        DerivaMLSchemaPinned,
    )

    err = DerivaMLSchemaPinned("refresh_schema refused: cache is pinned")
    assert isinstance(err, DerivaMLConfigurationError)
    assert "pinned" in str(err)


def test_no_association_exception_inherits_not_found_and_base():
    """``NoAssociationException`` is a ``DerivaMLNotFoundError`` so callers
    that already ``except DerivaMLException:`` continue to work, but the
    typed subclass enables narrow catches that don't string-match the
    exception message."""
    from deriva_ml.core.exceptions import (
        DerivaMLException,
        DerivaMLNotFoundError,
        NoAssociationException,
    )

    err = NoAssociationException("Image", "Subject")
    assert isinstance(err, DerivaMLNotFoundError)
    assert isinstance(err, DerivaMLException)
    assert err.table1 == "Image"
    assert err.table2 == "Subject"
    assert "No association tables found" in str(err)
    assert "Image" in str(err)
    assert "Subject" in str(err)


def test_ambiguous_association_exception_inherits_data_error_and_base():
    """``AmbiguousAssociationException`` is a ``DerivaMLDataError`` so
    ``except DerivaMLException:`` still catches it. The ``count`` field
    is structured rather than buried in the message."""
    from deriva_ml.core.exceptions import (
        AmbiguousAssociationException,
        DerivaMLDataError,
        DerivaMLException,
    )

    err = AmbiguousAssociationException("Image", "Dataset", 2)
    assert isinstance(err, DerivaMLDataError)
    assert isinstance(err, DerivaMLException)
    assert err.table1 == "Image"
    assert err.table2 == "Dataset"
    assert err.count == 2
    assert "2 association tables" in str(err)
    assert "Image" in str(err)
    assert "Dataset" in str(err)
