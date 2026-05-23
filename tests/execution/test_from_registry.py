"""Tests for ``Execution.from_registry`` (audit P1 coverage gap).

Pre-coverage, ``from_registry`` was only exercised indirectly
through ``DerivaML.resume_execution`` integration tests; the
classmethod itself — minimum-init construction, the
SQLite-row ``workflow_rid`` population, the absent
``_execution_record`` — was never asserted.

These tests pin the construction contract directly with a
``MagicMock`` ``ml_object``:

1. ``execution_rid`` is set from the arg.
2. ``_ml_object``, ``_model`` are bound.
3. ``_dry_run = False``.
4. ``configuration is None`` (Group E loads from ``config_json``).
5. ``_execution_record is None`` (read-through machinery covers
   lifecycle fields).
6. ``workflow_rid`` is populated from the SQLite row when the
   registry returns one.
7. ``workflow_rid`` falls back to ``None`` when the SQLite
   read raises (offline-mode tests, partial fixtures).
8. ``workflow_rid`` falls back to ``None`` when the registry
   returns ``None`` for the row.

Pure-Python tests; no live catalog required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from deriva_ml.execution.execution import Execution


def _build_ml_object(workflow_row: dict | None = None, raise_on_get: bool = False):
    """Build a MagicMock ``DerivaML`` shaped for ``from_registry``.

    Args:
        workflow_row: The dict ``get_execution`` should return
            (with ``workflow_rid`` key). ``None`` means the
            registry has no row.
        raise_on_get: When ``True``, ``get_execution`` raises —
            simulates offline-mode / partial-fixture scenarios.
    """
    ml = MagicMock()
    ml.model = MagicMock()
    ml.working_dir = "/tmp/wdir"
    ml.cache_dir = "/tmp/cache"
    ml._logger = MagicMock()
    store = ml.workspace.execution_state_store.return_value
    if raise_on_get:
        store.get_execution.side_effect = RuntimeError("registry down")
    else:
        store.get_execution.return_value = workflow_row
    return ml


class TestFromRegistryConstruction:
    """Pin the minimum-init construction contract."""

    def test_binds_execution_rid_from_arg(self):
        ml = _build_ml_object()
        exe = Execution.from_registry(ml_object=ml, execution_rid="EX-42")
        assert exe.execution_rid == "EX-42"

    def test_binds_ml_object_and_model(self):
        ml = _build_ml_object()
        exe = Execution.from_registry(ml_object=ml, execution_rid="EX-1")
        assert exe._ml_object is ml
        assert exe._model is ml.model

    def test_dry_run_is_false(self):
        ml = _build_ml_object()
        exe = Execution.from_registry(ml_object=ml, execution_rid="EX-1")
        assert exe._dry_run is False

    def test_configuration_is_none(self):
        """Group E loads config from the SQLite ``config_json`` field; the
        minimum-init constructor leaves it ``None``.
        """
        ml = _build_ml_object()
        exe = Execution.from_registry(ml_object=ml, execution_rid="EX-1")
        assert exe.configuration is None

    def test_execution_record_is_none(self):
        """No ExecutionRecord on resume — read-through machinery handles lifecycle."""
        ml = _build_ml_object()
        exe = Execution.from_registry(ml_object=ml, execution_rid="EX-1")
        assert exe._execution_record is None

    def test_empty_lists_and_dicts_are_initialized(self):
        """Fields the existing class expects to exist."""
        ml = _build_ml_object()
        exe = Execution.from_registry(ml_object=ml, execution_rid="EX-1")
        assert exe._datasets_list == []
        assert exe.dataset_rids == []
        assert exe.asset_paths == {}

    def test_working_and_cache_dirs_come_from_ml_object(self):
        ml = _build_ml_object()
        exe = Execution.from_registry(ml_object=ml, execution_rid="EX-1")
        assert exe._working_dir == ml.working_dir
        assert exe._cache_dir == ml.cache_dir


class TestFromRegistryWorkflowRid:
    """Pin the SQLite-row ``workflow_rid`` population (audit Ex-init's
    third item — the original bug was that ``workflow_rid`` was always
    ``None``)."""

    def test_populates_from_registry_row(self):
        ml = _build_ml_object(workflow_row={"workflow_rid": "WF-7"})
        exe = Execution.from_registry(ml_object=ml, execution_rid="EX-1")
        assert exe.workflow_rid == "WF-7"

    def test_falls_back_to_none_when_row_missing(self):
        """Registry returns ``None`` → ``workflow_rid is None``."""
        ml = _build_ml_object(workflow_row=None)
        exe = Execution.from_registry(ml_object=ml, execution_rid="EX-1")
        assert exe.workflow_rid is None

    def test_falls_back_to_none_when_row_has_no_workflow_rid_key(self):
        """Registry row exists but the ``workflow_rid`` field is missing.

        Defensive — ``row.get("workflow_rid")`` returns ``None``.
        """
        ml = _build_ml_object(workflow_row={"other_field": "x"})
        exe = Execution.from_registry(ml_object=ml, execution_rid="EX-1")
        assert exe.workflow_rid is None

    def test_falls_back_to_none_when_registry_read_raises(self):
        """Offline-mode / partial-fixture scenarios — the try/except in
        ``from_registry`` swallows the failure.

        Pre-fix the function had ``workflow_rid = None`` with no
        registry lookup at all; the audit (Ex-init cluster) asked for
        the lookup with a defensive fallback. Pin both:
        (1) ``workflow_rid`` is loaded when possible, (2) failure
        falls back to ``None`` rather than refusing to construct.
        """
        ml = _build_ml_object(raise_on_get=True)
        exe = Execution.from_registry(ml_object=ml, execution_rid="EX-1")
        assert exe.workflow_rid is None
        # The instance still constructs successfully.
        assert exe.execution_rid == "EX-1"
