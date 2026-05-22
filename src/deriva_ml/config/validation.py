"""Pydantic models for the config-file validation API.

These models are returned by :meth:`DerivaML.validate_config_file` /
:meth:`DerivaML.validate_config_directory`. The models live in their
own module so they can cross a boundary: the deriva-ml-mcp tool
wrapper serializes them with ``.model_dump()`` and downstream agents
consume the JSON.

The validation methods themselves are metadata-only catalog queries
(no bag download, no source execution). Parsing is AST-only -- the
file is never executed. This makes the surface safe to point at
arbitrary user code in ``src/configs/``.

Example:
    Inspect a validation report::

        >>> report = ml.validate_config_file("src/configs/datasets.py")  # doctest: +SKIP
        >>> for r in report.results:  # doctest: +SKIP
        ...     if not r.valid:
        ...         print(f"{r.file}:{r.line} {r.entry_kind}({r.rid!r}) -> {r.reasons}")
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

ConfigEntryKind = Literal[
    "DatasetSpecConfig",
    "AssetSpecConfig",
    "Workflow",
    "DerivaMLConfig",
]

ConfigEntryReason = Literal[
    # RID + entity-kind problems (one of these means the entry doesn't
    # resolve in the catalog at all).
    "rid_not_found",
    "not_a_dataset",
    "not_an_asset",
    "not_a_workflow",
    # Version problems (datasets only).
    "version_missing",
    "version_not_found",
    # AST extraction problems -- the parser found the constructor but
    # couldn't pull the RID. Doesn't mean the entry is bad; means the
    # validator can't reason about it (it skips the catalog call).
    "rid_unresolvable",
    # Connection problems (DerivaMLConfig entries only).
    "catalog_unreachable",
    "catalog_hostname_mismatch",
    "catalog_id_mismatch",
]


# ---------------------------------------------------------------------------
# Parsed entry -- AST-only, no catalog hit yet
# ---------------------------------------------------------------------------


class ConfigEntry(BaseModel):
    """One ``*Config(...)`` (or ``Workflow(...)``) constructor call found in a file.

    Produced by :func:`parse_config_file`. Used as input to the
    catalog-side validators and as one half of :class:`ConfigEntryResult`.

    Attributes:
        file: Path to the config file the entry was found in.
            Relative paths are accepted; the validator does not
            resolve them.
        line: 1-based line number of the constructor call.
        col: 0-based column number of the constructor call.
        entry_kind: The constructor name. One of ``DatasetSpecConfig``,
            ``AssetSpecConfig``, ``Workflow``, ``DerivaMLConfig``.
            Matching is by name (not by import); identical names from
            unrelated modules will also be picked up. This is
            intentional -- callers can use the line + column to
            diagnose false matches.
        rid: The ``rid=`` kwarg's value as a string. ``None`` when the
            kwarg is missing, when it's not a string literal (e.g.
            assigned from an unresolved local variable), or when the
            entry uses positional args this parser does not handle.
            ``None`` triggers ``rid_unresolvable`` at validation time
            rather than an error -- bad RIDs in configs are bugs the
            user wants to see surfaced, not exceptions during the
            walk.
        version: The ``version=`` kwarg's value as a string. ``None``
            for entries that don't take a version (assets, workflows,
            connection configs) or for entries where the kwarg is
            missing.
        hostname: The ``hostname=`` kwarg's value (DerivaMLConfig only).
            ``None`` for other entry kinds.
        catalog_id: The ``catalog_id=`` kwarg's value (DerivaMLConfig
            only). ``None`` for other entry kinds.
        cache: The ``cache=`` kwarg's value (AssetSpecConfig only).
        snippet: A short text excerpt around the call site, included
            for human-readable error messages. Not parsed.
    """

    model_config = ConfigDict(extra="forbid")

    file: str
    line: int
    col: int
    entry_kind: ConfigEntryKind
    rid: str | None = None
    version: str | None = None
    hostname: str | None = None
    catalog_id: str | None = None
    cache: bool | None = None
    snippet: str | None = None


# ---------------------------------------------------------------------------
# Validation result models
# ---------------------------------------------------------------------------


class ConfigEntryResult(BaseModel):
    """Result of validating one :class:`ConfigEntry`.

    Attributes:
        entry: The parsed entry that was validated. Echoed back so
            callers can match results to inputs.
        valid: True when ``reasons`` is empty.
        reasons: List of failure reasons. Empty when ``valid`` is True.
        actual_table: When ``not_a_*`` is set, the name of the table
            the RID actually points at. Useful for diagnosing copy-
            paste between configs (e.g. an asset RID pasted into
            datasets.py).
        available_versions: When ``version_not_found`` is set, up to
            20 known versions for the dataset (newest first). None
            otherwise.
        resolved_name: Friendly name from the catalog -- dataset
            description, asset filename, or workflow name. Only set
            when ``valid``.
    """

    model_config = ConfigDict(extra="forbid")

    entry: ConfigEntry
    valid: bool
    reasons: list[ConfigEntryReason] = Field(default_factory=list)
    actual_table: str | None = None
    available_versions: list[str] | None = None
    resolved_name: str | None = None


class ConfigFileParseError(BaseModel):
    """A file the validator couldn't parse.

    Attributes:
        file: Path that failed to parse.
        line: 1-based line of the syntax error (best-effort -- the
            ``SyntaxError`` location is used directly).
        message: ``SyntaxError`` message verbatim.
    """

    model_config = ConfigDict(extra="forbid")

    file: str
    line: int | None
    message: str


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------


class ConfigValidationReport(BaseModel):
    """Result returned by :meth:`DerivaML.validate_config_file` /
    :meth:`DerivaML.validate_config_directory`.

    Attributes:
        file_count: Number of files processed (parsed successfully).
        entry_count: Total number of constructor entries found across
            all files. Includes entries that were unresolvable at
            parse time.
        all_valid: True iff every entry in ``results`` is valid AND
            ``parse_errors`` is empty. An empty ``results`` list with
            no parse errors is reported as ``all_valid=True``.
        results: Per-entry results. Ordered first by file (input
            order), then by line within each file.
        parse_errors: Files that failed to parse. A single broken file
            does not abort the walk -- the validator keeps going so
            issues in the other files still surface.
    """

    model_config = ConfigDict(extra="forbid")

    file_count: int = 0
    entry_count: int = 0
    all_valid: bool = True
    results: list[ConfigEntryResult] = Field(default_factory=list)
    parse_errors: list[ConfigFileParseError] = Field(default_factory=list)
