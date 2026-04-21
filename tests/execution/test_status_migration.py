"""Tests for the title-case ExecutionStatus migration (Phase 2 Subsystem 1)."""

from __future__ import annotations

import pytest


def test_execution_status_values_are_title_case():
    from deriva_ml.execution.state_store import ExecutionStatus
    assert ExecutionStatus.Created.value == "Created"
    assert ExecutionStatus.Running.value == "Running"
    assert ExecutionStatus.Stopped.value == "Stopped"
    assert ExecutionStatus.Failed.value == "Failed"
    assert ExecutionStatus.Pending_Upload.value == "Pending_Upload"
    assert ExecutionStatus.Uploaded.value == "Uploaded"
    assert ExecutionStatus.Aborted.value == "Aborted"


def test_execution_status_parses_title_case_from_catalog():
    """A catalog row with {'Status': 'Running'} parses directly."""
    from deriva_ml.execution.state_store import ExecutionStatus
    assert ExecutionStatus("Running") is ExecutionStatus.Running
    assert ExecutionStatus("Pending_Upload") is ExecutionStatus.Pending_Upload


def test_execution_status_rejects_lowercase():
    from deriva_ml.execution.state_store import ExecutionStatus
    with pytest.raises(ValueError):
        ExecutionStatus("running")
    with pytest.raises(ValueError):
        ExecutionStatus("pending_upload")


def test_execution_status_member_count():
    """Canonical set is exactly 7."""
    from deriva_ml.execution.state_store import ExecutionStatus
    assert len(list(ExecutionStatus)) == 7
