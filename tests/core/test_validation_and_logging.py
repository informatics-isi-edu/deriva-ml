"""Focused tests for ``deriva_ml.core.validation`` and ``deriva_ml.core.logging_config``.

Closes audit P1 V1 (validation has zero direct tests in tests/core/)
and L1 (configure_logging has no test coverage). Scope is bounded:
the audit named specific paths that were untested; this file
covers those paths plus a couple of close neighbors. Broader
coverage of these two modules is a follow-up task.

Pure-Python tests; no live catalog required.
"""

from __future__ import annotations

import logging

import pytest

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.core.logging_config import (
    DERIVA_LOGGERS,
    HYDRA_LOGGERS,
    configure_logging,
    get_logger,
    is_hydra_initialized,
)
from deriva_ml.core.validation import ValidationResult, validate_rids
from deriva_ml.dataset.aux_classes import DatasetHistory, DatasetVersion

# ---------------------------------------------------------------------------
# ValidationResult — the missing direct coverage
# ---------------------------------------------------------------------------


class TestValidationResult:
    """Direct unit tests for ``ValidationResult``.

    The audit's V1 finding called out that ``ValidationResult`` had
    no direct tests; coverage was incidental through
    ``tests/dataset/test_validate_execution_configuration_unit.py``.
    These tests pin the structural contract.
    """

    def test_default_state_is_valid_empty(self):
        result = ValidationResult()
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.validated_rids == {}

    def test_add_error_flips_to_invalid_and_records_message(self):
        result = ValidationResult()
        result.add_error("Bad RID")
        assert result.is_valid is False
        assert result.errors == ["Bad RID"]

    def test_add_warning_keeps_valid(self):
        """Warnings do not invalidate the result — only errors do."""
        result = ValidationResult()
        result.add_warning("Suspicious version")
        assert result.is_valid is True
        assert result.warnings == ["Suspicious version"]

    def test_merge_combines_errors_warnings_and_invalidates(self):
        """Merging an invalid result makes ``self`` invalid; lists concat."""
        a = ValidationResult()
        a.add_warning("a-warn")

        b = ValidationResult()
        b.add_error("b-err")

        a.merge(b)
        assert a.is_valid is False
        assert "a-warn" in a.warnings
        assert "b-err" in a.errors

    def test_merge_validated_rids_combines_both_dicts(self):
        """The ``validated_rids`` dict merges via dict.update.

        The audit specifically called this out as an uncovered
        path. Two results carrying different RID sets must produce
        the union after merge.
        """
        a = ValidationResult()
        a.validated_rids["RID-A"] = {"rid": "RID-A", "table": "Image", "schema": "x"}

        b = ValidationResult()
        b.validated_rids["RID-B"] = {"rid": "RID-B", "table": "Subject", "schema": "x"}

        a.merge(b)
        assert set(a.validated_rids) == {"RID-A", "RID-B"}

    def test_merge_validated_rids_later_overwrites_earlier_on_key_conflict(self):
        """Last-merge-wins on key conflict — pin dict.update semantics."""
        a = ValidationResult()
        a.validated_rids["RID-X"] = {"rid": "RID-X", "table": "Image", "schema": "x"}

        b = ValidationResult()
        b.validated_rids["RID-X"] = {"rid": "RID-X", "table": "Subject", "schema": "y"}

        a.merge(b)
        assert a.validated_rids["RID-X"]["table"] == "Subject"

    def test_merge_valid_into_invalid_keeps_invalid(self):
        """Merging a valid result into an invalid one stays invalid."""
        a = ValidationResult()
        a.add_error("a-err")  # makes a invalid

        b = ValidationResult()  # default-valid

        a.merge(b)
        assert a.is_valid is False

    def test_repr_summary_for_valid(self):
        result = ValidationResult()
        result.validated_rids["RID-A"] = {"rid": "RID-A", "table": "Image", "schema": "x"}
        s = repr(result)
        assert "OK" in s
        assert "Validated 1 RID(s)" in s

    def test_repr_summary_for_invalid(self):
        result = ValidationResult()
        result.add_error("oops")
        result.add_warning("hmm")
        s = repr(result)
        assert "FAIL" in s
        assert "1 error" in s
        assert "oops" in s
        assert "hmm" in s


# ---------------------------------------------------------------------------
# configure_logging — focused coverage
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_deriva_ml_logger():
    """Provide ``deriva_ml`` logger with all handlers stripped, restored on exit.

    ``configure_logging`` adds a ``StreamHandler`` when the logger
    has no handlers; tests need a clean baseline to observe that
    behaviour, and must not leak handlers into the rest of the
    test session.
    """
    logger = get_logger()
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    for h in saved_handlers:
        logger.removeHandler(h)
    yield logger
    # Restore.
    for h in list(logger.handlers):
        logger.removeHandler(h)
    for h in saved_handlers:
        logger.addHandler(h)
    logger.setLevel(saved_level)


class TestConfigureLogging:
    """Direct unit tests for ``configure_logging``.

    The audit's L1 finding called out three behaviours that the
    function decides:

      * Whether to add a StreamHandler (only when not under Hydra
        and no handler exists already).
      * Which loggers (deriva-ml + hydra + deriva-py + bdbag +
        bagit) to set levels on.
      * Whether to call ``basicConfig`` (it explicitly doesn't —
        touching the root logger would clobber the caller's setup).

    These tests pin each of those.
    """

    def test_sets_level_on_deriva_ml_logger(self, clean_deriva_ml_logger):
        """The ``deriva_ml`` logger gets the requested level."""
        configure_logging(level=logging.DEBUG)
        assert clean_deriva_ml_logger.level == logging.DEBUG

    def test_sets_level_on_hydra_loggers(self, clean_deriva_ml_logger):
        """Every documented Hydra logger follows ``level``."""
        configure_logging(level=logging.ERROR)
        for name in HYDRA_LOGGERS:
            assert logging.getLogger(name).level == logging.ERROR, f"Hydra logger {name!r} did not follow level=ERROR"

    def test_sets_deriva_level_on_deriva_py_loggers(self, clean_deriva_ml_logger):
        """The deriva-py / bdbag / bagit loggers follow ``deriva_level``."""
        configure_logging(level=logging.DEBUG, deriva_level=logging.WARNING)
        for name in DERIVA_LOGGERS:
            assert logging.getLogger(name).level == logging.WARNING, (
                f"Deriva-py logger {name!r} did not follow deriva_level=WARNING"
            )

    def test_deriva_level_defaults_to_level_when_omitted(self, clean_deriva_ml_logger):
        """``deriva_level=None`` reuses ``level``."""
        configure_logging(level=logging.INFO, deriva_level=None)
        for name in DERIVA_LOGGERS:
            assert logging.getLogger(name).level == logging.INFO

    def test_adds_stream_handler_when_none_present_outside_hydra(self, clean_deriva_ml_logger):
        """Outside Hydra, the deriva_ml logger gets a StreamHandler.

        Pre-test fixture strips all handlers, so this exercises the
        "logger has no handlers" branch.
        """
        # The non-Hydra path is taken only when ``is_hydra_initialized()``
        # returns False; in a plain pytest invocation that's the case.
        if is_hydra_initialized():
            pytest.skip("Test runs under Hydra; non-Hydra path unreachable.")

        configure_logging(level=logging.WARNING)
        assert any(isinstance(h, logging.StreamHandler) for h in clean_deriva_ml_logger.handlers), (
            "Expected configure_logging to add a StreamHandler when the "
            "logger had no handlers and is_hydra_initialized() is False."
        )

    def test_does_not_duplicate_handler_when_one_already_exists(self, clean_deriva_ml_logger):
        """A pre-existing handler is preserved; no second one is added."""
        sentinel = logging.NullHandler()
        clean_deriva_ml_logger.addHandler(sentinel)

        configure_logging(level=logging.WARNING)

        # Sentinel still present, no StreamHandler appended.
        assert sentinel in clean_deriva_ml_logger.handlers
        assert not any(isinstance(h, logging.StreamHandler) for h in clean_deriva_ml_logger.handlers), (
            "configure_logging must not add a StreamHandler when the deriva_ml logger already has handlers."
        )

    def test_does_not_call_basicConfig(self, clean_deriva_ml_logger, monkeypatch):
        """``configure_logging`` must NOT touch the root logger via basicConfig.

        Calling ``logging.basicConfig`` would install a handler on
        the root logger, which is the caller's domain (applications,
        notebooks, test runners may have configured root themselves).
        """
        calls: list[tuple] = []

        def fake_basicConfig(*args, **kwargs):
            calls.append((args, kwargs))

        monkeypatch.setattr(logging, "basicConfig", fake_basicConfig)
        configure_logging(level=logging.INFO)
        assert calls == [], (
            f"configure_logging called logging.basicConfig with {calls}; it must never touch the root logger."
        )

    def test_returns_the_deriva_ml_logger(self, clean_deriva_ml_logger):
        """The return value is the configured ``deriva_ml`` logger."""
        returned = configure_logging(level=logging.INFO)
        assert returned is clean_deriva_ml_logger


# ---------------------------------------------------------------------------
# validate_rids — dataset version-history branch
# ---------------------------------------------------------------------------
#
# Regression coverage for the alignment-audit Batch A finding
# (core/validation.py:285): the version-checking branch called a
# nonexistent ``dataset.list_versions()`` inside a broad ``except
# Exception``. The AttributeError was silently swallowed, so the
# intended hard ERROR for a required version that does not exist in
# history NEVER ran — every bad version requirement degraded to a
# warning. The fix calls ``dataset.dataset_history()`` and narrows the
# except to ``DerivaMLException`` so programming errors propagate.


def _history(*versions: str, dataset_rid: str = "1-DSAA") -> list[DatasetHistory]:
    """Build a ``DatasetHistory`` list with real ``DatasetVersion`` labels.

    Mirrors what ``Dataset.dataset_history()`` returns so the validator
    exercises ``h.dataset_version`` exactly as it does in production.

    Args:
        *versions: PEP 440 version strings, e.g. ``"1.0.0"``.
        dataset_rid: RID stamped on each history entry.

    Returns:
        A list of :class:`DatasetHistory`, one entry per version.

    Example:
        >>> hist = _history("0.1.0", "1.0.0")
        >>> [str(h.dataset_version) for h in hist]
        ['0.1.0', '1.0.0']
    """
    return [
        DatasetHistory(
            dataset_version=DatasetVersion.parse(v),
            dataset_rid=dataset_rid,
            version_rid=f"{i}-V{i:03d}",
            snapshot=f"snap{i}",
        )
        for i, v in enumerate(versions)
    ]


class _ResolvedInfo:
    """Stand-in for ``BatchRidResult`` — validator reads only these two attrs."""

    def __init__(self, table_name: str, schema_name: str) -> None:
        self.table_name = table_name
        self.schema_name = schema_name


class _StubDataset:
    """Stand-in for the ``Dataset`` returned by ``ml.lookup_dataset``.

    Carries a ``current_version`` and a ``dataset_history()`` whose
    return value (or raised exception) is scripted per test.
    """

    def __init__(
        self,
        current_version: str,
        history: list[DatasetHistory] | None = None,
        history_exc: BaseException | None = None,
    ) -> None:
        self.current_version = DatasetVersion.parse(current_version)
        self._history = history or []
        self._history_exc = history_exc

    def dataset_history(self) -> list[DatasetHistory]:
        if self._history_exc is not None:
            raise self._history_exc
        return self._history


class _StubML:
    """Minimal ``DerivaML`` stand-in for ``validate_rids`` version checks.

    ``validate_rids`` touches only ``resolve_rids`` (for batch RID
    resolution) and ``lookup_dataset`` (for the version-history branch);
    this stub scripts both.
    """

    def __init__(self, dataset_rid: str, dataset: _StubDataset) -> None:
        self._dataset_rid = dataset_rid
        self._dataset = dataset

    def resolve_rids(self, rids):
        return {rid: _ResolvedInfo(table_name="Dataset", schema_name="deriva-ml") for rid in rids}

    def lookup_dataset(self, rid):
        return self._dataset


class TestValidateRidsVersionHistory:
    """The version-history branch of ``validate_rids``.

    These tests pin the FIXED behaviour: a required version absent
    from ``dataset_history()`` is a hard ERROR (not a warning), while a
    required version present in history validates cleanly. The bug being
    regressed is that the branch called a nonexistent ``list_versions()``
    whose ``AttributeError`` was swallowed, downgrading every bad
    version to a warning.
    """

    def test_required_version_not_in_history_is_a_hard_error(self):
        """A required version absent from history must produce an ERROR.

        Pre-fix this only produced a warning (the swallowed
        AttributeError fell through to the ``add_warning`` path).
        """
        ds = _StubDataset(current_version="1.0.0", history=_history("0.1.0", "1.0.0"))
        ml = _StubML("1-DSAA", ds)

        result = validate_rids(
            ml,
            dataset_rids=["1-DSAA"],
            dataset_versions={"1-DSAA": "9.9.9"},  # not in history
        )

        assert result.is_valid is False, (
            "A required dataset version that does not exist in history must be a hard error, not a warning."
        )
        assert any("does not have version '9.9.9'" in e for e in result.errors)
        # And it must NOT have been silently degraded to a warning.
        assert not any("Could not verify version history" in w for w in result.warnings)

    def test_required_version_present_in_history_validates_ok(self):
        """A non-current version that DOES exist in history is accepted."""
        ds = _StubDataset(current_version="1.0.0", history=_history("0.1.0", "1.0.0"))
        ml = _StubML("1-DSAA", ds)

        result = validate_rids(
            ml,
            dataset_rids=["1-DSAA"],
            dataset_versions={"1-DSAA": "0.1.0"},  # older but real
        )

        assert result.is_valid is True
        assert result.errors == []
        assert result.validated_rids["1-DSAA"]["version"] == "0.1.0"
        assert result.validated_rids["1-DSAA"]["current_version"] == "1.0.0"

    def test_history_read_failure_degrades_to_warning(self):
        """A genuine catalog-read failure (DerivaMLException) still warns.

        The narrowed except must preserve the original intent: when
        history truly cannot be read, degrade to a warning rather than
        crash.
        """
        ds = _StubDataset(
            current_version="1.0.0",
            history_exc=DerivaMLException("catalog unavailable"),
        )
        ml = _StubML("1-DSAA", ds)

        result = validate_rids(
            ml,
            dataset_rids=["1-DSAA"],
            dataset_versions={"1-DSAA": "9.9.9"},
        )

        # Degraded, not crashed: a warning, and is_valid stays True.
        assert result.is_valid is True
        assert any("Could not verify version history" in w for w in result.warnings)

    def test_programming_error_in_history_propagates(self):
        """A non-DerivaML error (e.g. a typo'd method) must NOT be swallowed.

        This is the heart of the regression: the original
        ``except Exception`` hid the AttributeError from the misnamed
        ``list_versions()`` call. With the narrowed except, an
        ``AttributeError`` raised while reading history propagates
        loudly instead of degrading to a warning.
        """
        ds = _StubDataset(
            current_version="1.0.0",
            history_exc=AttributeError("'Dataset' object has no attribute 'list_versions'"),
        )
        ml = _StubML("1-DSAA", ds)

        # validate_rids wraps the whole per-dataset block in a broad
        # ``except Exception`` that turns the propagated AttributeError
        # into a hard ERROR — it is decisively NOT a silent warning.
        result = validate_rids(
            ml,
            dataset_rids=["1-DSAA"],
            dataset_versions={"1-DSAA": "9.9.9"},
        )

        assert result.is_valid is False
        assert any("Failed to validate dataset '1-DSAA' version" in e for e in result.errors)
        assert not any("Could not verify version history" in w for w in result.warnings)
