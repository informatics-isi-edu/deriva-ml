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


def test_definitions_reexports_every_exception_from_exceptions():
    """``deriva_ml.core.definitions`` must re-export every public
    exception class declared in ``deriva_ml.core.exceptions``.

    The module-level docstring of ``definitions`` recommends users
    import exceptions from there ("From: deriva_ml.core.definitions
    or deriva_ml.exceptions"). Before this test, ``definitions.
    __all__`` lagged behind: every class added since Phase 2
    (``DerivaMLFeatureNotFound``, ``DerivaMLOfflineError``,
    ``DerivaMLStateInconsistency``, ``DerivaMLDirtyWorkflowError``,
    ``NoAssociationException``, ``AmbiguousAssociationException``,
    ``DerivaMLSchemaRefreshBlocked``, ``DerivaMLSchemaPinned``,
    ``DerivaMLMaterializeLimitExceeded``, ``DerivaMLDenormalizeError``
    + 5 subclasses, ``DerivaMLRidsNotFound``) was missing. Users
    following the recommended import path couldn't actually import
    them from there.

    This test pins the parity contract: the set of exception names
    in ``definitions.__all__`` must equal the set in
    ``exceptions.__all__``. Any new exception class added to
    exceptions.py and forgotten in definitions.py fails this test.
    """
    from deriva_ml.core import definitions, exceptions

    exception_names = set(exceptions.__all__)
    definitions_exception_names = {
        name
        for name in definitions.__all__
        if name in exception_names
    }
    missing = exception_names - definitions_exception_names
    assert not missing, (
        f"deriva_ml.core.definitions does not re-export these "
        f"exception classes: {sorted(missing)}. Add them to "
        f"``definitions.__all__`` and the corresponding import "
        f"block so the recommended import path works."
    )


def test_dirty_workflow_error_with_paths():
    """When dirty_paths is provided, the message lists each path on its own line."""
    from deriva_ml.core.exceptions import DerivaMLDirtyWorkflowError

    exc = DerivaMLDirtyWorkflowError(
        "src/models/train.py",
        dirty_paths=["?? findings/run_output.txt", " M src/models/train.py"],
    )
    msg = str(exc)
    assert "src/models/train.py" in msg
    assert "?? findings/run_output.txt" in msg
    assert " M src/models/train.py" in msg
    assert exc.path == "src/models/train.py"
    assert exc.dirty_paths == ["?? findings/run_output.txt", " M src/models/train.py"]


def test_dirty_workflow_error_without_paths_is_backward_compatible():
    """The single-argument form still works (no dirty_paths passed)."""
    from deriva_ml.core.exceptions import DerivaMLDirtyWorkflowError

    exc = DerivaMLDirtyWorkflowError("src/models/train.py")
    msg = str(exc)
    assert "src/models/train.py" in msg
    assert "--allow-dirty" in msg
    assert exc.path == "src/models/train.py"
    assert exc.dirty_paths == []
