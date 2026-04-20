"""Tests for the execution state machine."""

from __future__ import annotations

import pytest

from deriva_ml.execution.state_machine import (
    ALLOWED_TRANSITIONS,
    InvalidTransitionError,
    validate_transition,
)
from deriva_ml.execution.state_store import ExecutionStatus


def test_allowed_transitions_cover_all_happy_paths():
    # created → running → stopped → pending_upload → uploaded
    assert (ExecutionStatus.created, ExecutionStatus.running) in ALLOWED_TRANSITIONS
    assert (ExecutionStatus.running, ExecutionStatus.stopped) in ALLOWED_TRANSITIONS
    assert (ExecutionStatus.stopped, ExecutionStatus.pending_upload) in ALLOWED_TRANSITIONS
    assert (ExecutionStatus.pending_upload, ExecutionStatus.uploaded) in ALLOWED_TRANSITIONS


def test_allowed_transitions_cover_failure_paths():
    assert (ExecutionStatus.running, ExecutionStatus.failed) in ALLOWED_TRANSITIONS
    assert (ExecutionStatus.pending_upload, ExecutionStatus.failed) in ALLOWED_TRANSITIONS


def test_allowed_transitions_cover_abort():
    # Abort legal from created, running, stopped, failed
    for start in [ExecutionStatus.created, ExecutionStatus.running,
                  ExecutionStatus.stopped, ExecutionStatus.failed]:
        assert (start, ExecutionStatus.aborted) in ALLOWED_TRANSITIONS


def test_retry_from_failed_back_to_pending_upload():
    # retry_failed → pending_upload is legal (upload retry path)
    assert (ExecutionStatus.failed, ExecutionStatus.pending_upload) in ALLOWED_TRANSITIONS


def test_validate_transition_accepts_allowed():
    validate_transition(
        current=ExecutionStatus.running,
        target=ExecutionStatus.stopped,
    )  # must not raise


def test_validate_transition_rejects_disallowed():
    with pytest.raises(InvalidTransitionError) as exc:
        validate_transition(
            current=ExecutionStatus.uploaded,
            target=ExecutionStatus.running,  # can't go back to running
        )
    msg = str(exc.value)
    assert "uploaded" in msg
    assert "running" in msg


def test_invalid_transition_error_is_deriva_ml_exception():
    from deriva_ml.core.exceptions import DerivaMLException
    assert issubclass(InvalidTransitionError, DerivaMLException)
