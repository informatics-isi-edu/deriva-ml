"""Config-file validation for DerivaML hydra-zen config files.

This module exposes the read-side helpers that
``deriva-ml-skills/write-hydra-config`` documents: parse a
``src/configs/*.py`` file via AST, find every
``DatasetSpecConfig``/``AssetSpecConfig``/``Workflow``/``DerivaMLConfig``
constructor call, and validate each against the catalog. The validator
never executes the file -- AST-only -- so it's safe against arbitrary
user code in the configs/ dir.

The write-side helpers (the "after I created this dataset, want me to
add it to datasets.py?" offer) live in the dataset-lifecycle,
work-with-assets, and execution-lifecycle skills as per-context skill
prose. See
``deriva-ml-skills/docs/superpowers/plans/2026-05-22-config-validator-bootstrap-design.md``
for the full design rationale.

Public surface:

- :class:`ConfigEntry` -- one ``*Config(...)`` constructor call found in
  a file, with location info.
- :class:`ConfigEntryResult` -- per-entry validation result, used in
  the report.
- :class:`ConfigValidationReport` -- top-level report from
  :meth:`DerivaML.validate_config_file` /
  :meth:`DerivaML.validate_config_directory`.
- :func:`parse_config_file` -- AST-only parser; returns the
  :class:`ConfigEntry` list for one file. Useful for tests and for
  callers that want to inspect what would be validated without making
  catalog calls.

The :class:`DerivaML` validator methods live on
:class:`~deriva_ml.core.mixins.dataset.DatasetMixin` alongside the
existing :meth:`DerivaML.validate_dataset_specs` and
:meth:`DerivaML.validate_execution_configuration`.

Example:
    Parse a config file without touching the catalog::

        >>> from deriva_ml.config import parse_config_file
        >>> entries, parse_errors = parse_config_file("src/configs/datasets.py")  # doctest: +SKIP
        >>> for e in entries:  # doctest: +SKIP
        ...     print(e.line, e.entry_kind, e.rid, e.version)

    Validate against a catalog::

        >>> report = ml.validate_config_file("src/configs/datasets.py")  # doctest: +SKIP
        >>> if not report.all_valid:  # doctest: +SKIP
        ...     for r in report.results:
        ...         if not r.valid:
        ...             print(r.file, r.line, r.rid, r.reasons)
"""

from __future__ import annotations

from deriva_ml.config.ast_walker import parse_config_file
from deriva_ml.config.validation import (
    ConfigEntry,
    ConfigEntryReason,
    ConfigEntryResult,
    ConfigValidationReport,
)

__all__ = [
    "ConfigEntry",
    "ConfigEntryResult",
    "ConfigEntryReason",
    "ConfigValidationReport",
    "parse_config_file",
]
