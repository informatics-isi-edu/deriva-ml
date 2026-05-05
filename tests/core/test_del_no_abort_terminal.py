"""Tests for DerivaML.__del__'s status-check logic.

The finalizer must only force-abort executions that died mid-flight
(``Created`` / ``Running``). For any post-Running status — ``Stopped``,
``Failed``, ``Pending_Upload``, ``Uploaded``, ``Aborted`` — it must do
nothing. Otherwise it re-transitions a cleanly-stopped execution to
``Aborted``, against an HTTP session that may already be torn down by
GC ordering, producing the ``'NoneType' object has no attribute 'get'``
crash that ``__del__`` then silently swallows but ``state_machine``
logs as ``catalog sync FAILED``.

These tests call ``__del__`` directly via the bound method rather than
relying on GC, since CPython's GC ordering is undefined.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from deriva_ml.core.base import DerivaML
from deriva_ml.execution.state_store import ExecutionStatus


@pytest.mark.parametrize(
    "terminal_status",
    [
        ExecutionStatus.Stopped,
        ExecutionStatus.Failed,
        ExecutionStatus.Pending_Upload,
        ExecutionStatus.Uploaded,
        ExecutionStatus.Aborted,
    ],
)
def test_del_does_not_abort_post_running_states(terminal_status):
    """__del__ must be a no-op for any state past Running.

    Re-transitioning a Stopped/Uploaded/etc. execution to Aborted is a
    state-machine violation and historically (before this fix) crashed
    against a torn-down catalog session — the symptom users saw as a
    spurious "catalog sync FAILED" warning on every clean exit.
    """
    fake_exe = SimpleNamespace(status=terminal_status, update_status=MagicMock())
    dml = SimpleNamespace(_execution=fake_exe)

    DerivaML.__del__(dml)

    fake_exe.update_status.assert_not_called()


@pytest.mark.parametrize(
    "non_terminal_status",
    [ExecutionStatus.Created, ExecutionStatus.Running],
)
def test_del_aborts_non_terminal_states(non_terminal_status):
    """__del__ must abort executions that died mid-flight.

    The original cleanup intent — catching abandoned ``with``-block
    exits — still applies for ``Created``/``Running`` so the audit
    trail reflects "this execution was killed, not committed."
    """
    fake_exe = SimpleNamespace(status=non_terminal_status, update_status=MagicMock())
    dml = SimpleNamespace(_execution=fake_exe)

    DerivaML.__del__(dml)

    fake_exe.update_status.assert_called_once_with(ExecutionStatus.Aborted, error="Execution Aborted")


def test_del_swallows_update_status_errors():
    """__del__ must never raise — finalizers that raise crash the interpreter."""
    fake_exe = SimpleNamespace(
        status=ExecutionStatus.Running,
        update_status=MagicMock(side_effect=RuntimeError("boom")),
    )
    dml = SimpleNamespace(_execution=fake_exe)

    # Should not raise.
    DerivaML.__del__(dml)


def test_del_no_execution_is_noop():
    """__del__ does nothing when no execution is attached (e.g. catalog-only use)."""
    dml = SimpleNamespace(_execution=None)

    # Should not raise.
    DerivaML.__del__(dml)
