"""Pydantic models for the validation methods on :class:`DerivaML`.

These models are returned by :meth:`DerivaML.validate_dataset_specs`
and :meth:`DerivaML.validate_execution_configuration`. The models live
in their own module so they can cross a boundary: the deriva-ml-mcp
Round 6 follow-up serializes them with ``.model_dump()`` from a tool
wrapper, and downstream agents (notebook, skill, web app) consume the
JSON.

The validation methods themselves are metadata-only catalog queries
(no bag download). For the heavier full-path test see
:meth:`Execution.dry_run` — and ADR-0002 for the rationale on keeping
the two surfaces distinct.

Example:
    Inspect a singular validation report::

        >>> report = ml.validate_dataset_specs(specs=[  # doctest: +SKIP
        ...     DatasetSpec(rid="2-B4C8", version="0.4.0"),
        ... ])
        >>> if not report.all_valid:  # doctest: +SKIP
        ...     for r in report.results:
        ...         if not r.valid:
        ...             print(r.spec.rid, r.reasons, r.available_versions)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from deriva_ml.asset.aux_classes import AssetSpec
from deriva_ml.core.definitions import RID
from deriva_ml.dataset.aux_classes import DatasetSpec

# ---------------------------------------------------------------------------
# Reason vocabularies
# ---------------------------------------------------------------------------

DatasetReason = Literal[
    "rid_not_found",
    "not_a_dataset",
    "version_not_found",
]

DatasetWarning = Literal["dataset_deleted",]

AssetReason = Literal[
    "rid_not_found",
    "not_an_asset",
]

WorkflowReason = Literal[
    "rid_not_found",
    "not_a_workflow",
]

CrossSpecIssueKind = Literal[
    "duplicate_rid",
    "version_conflict",
    "role_conflict",
]


# ---------------------------------------------------------------------------
# Per-spec result models
# ---------------------------------------------------------------------------


class DatasetSpecResult(BaseModel):
    """Result of validating one :class:`DatasetSpec`.

    Attributes:
        spec: The spec that was validated. Echoed back so callers can
            match results to inputs even if the input was reordered or
            coerced from a shorthand.
        valid: True when ``reasons`` is empty (warnings do not affect
            validity).
        reasons: List of failure reasons. Empty when ``valid`` is True.
        warnings: List of non-fatal observations (e.g. the dataset
            exists but is soft-deleted).
        actual_table: When ``not_a_dataset`` is set, the name of the
            table the RID actually points at. None otherwise.
        available_versions: When ``version_not_found`` is set, up to
            20 known versions for the dataset (newest first). None
            otherwise.
        resolved_version: The canonical normalized version string
            (e.g. ``"0.4.0"`` even if the user wrote ``"0.4"``). Only
            set when ``valid``.
        dataset_name: Dataset description, when available. Only set
            when ``valid``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    spec: DatasetSpec
    valid: bool
    reasons: list[DatasetReason] = Field(default_factory=list)
    warnings: list[DatasetWarning] = Field(default_factory=list)
    actual_table: str | None = None
    available_versions: list[str] | None = None
    resolved_version: str | None = None
    dataset_name: str | None = None


class AssetSpecResult(BaseModel):
    """Result of validating one :class:`AssetSpec`.

    Attributes:
        spec: The spec that was validated.
        valid: True when ``reasons`` is empty.
        reasons: List of failure reasons. Empty when ``valid`` is True.
        actual_table: When ``not_an_asset`` is set, the name of the
            table the RID actually points at. None otherwise.
        asset_table: Name of the asset table the row lives in. Only
            set when ``valid``.
        filename: Asset filename. Only set when ``valid``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    spec: AssetSpec
    valid: bool
    reasons: list[AssetReason] = Field(default_factory=list)
    actual_table: str | None = None
    asset_table: str | None = None
    filename: str | None = None


class WorkflowSpecResult(BaseModel):
    """Result of validating the workflow on an :class:`ExecutionConfiguration`.

    Attributes:
        rid: The workflow RID that was validated.
        valid: True when ``reasons`` is empty.
        reasons: List of failure reasons. Empty when ``valid`` is True.
        actual_table: When ``not_a_workflow`` is set, the name of the
            table the RID actually points at. None otherwise.
        workflow_name: Workflow name from the catalog. Only set when
            ``valid``.
    """

    model_config = ConfigDict(extra="forbid")

    rid: RID
    valid: bool
    reasons: list[WorkflowReason] = Field(default_factory=list)
    actual_table: str | None = None
    workflow_name: str | None = None


class CrossSpecIssue(BaseModel):
    """A consistency issue spanning two or more specs.

    Attributes:
        issue: The kind of inconsistency. ``duplicate_rid`` for the
            same RID listed twice; ``version_conflict`` for the same
            dataset RID with two different versions; ``role_conflict``
            for the same asset RID with both ``Input`` and ``Output``
            roles.
        rids: The RIDs involved. For ``duplicate_rid`` and
            ``role_conflict`` this is a single-element list (the RID
            that appears multiple times); for ``version_conflict`` it
            is also a single-element list (the dataset RID with the
            conflict). Kept as a list so future issue kinds spanning
            two distinct RIDs can use the same shape.
        detail: Human-readable description, suitable for showing the
            user.
    """

    model_config = ConfigDict(extra="forbid")

    issue: CrossSpecIssueKind
    rids: list[RID]
    detail: str


# ---------------------------------------------------------------------------
# Top-level reports
# ---------------------------------------------------------------------------


class DatasetSpecValidationReport(BaseModel):
    """Result returned by :meth:`DerivaML.validate_dataset_specs`.

    Attributes:
        all_valid: True iff every entry in ``results`` has ``valid``
            True. An empty ``results`` list is reported as
            ``all_valid=True``.
        results: Per-spec results, in the same order as the input
            list (duplicates preserved — the singular method does not
            deduplicate; cross-spec duplicate detection is reserved
            for the composite).
    """

    model_config = ConfigDict(extra="forbid")

    all_valid: bool
    results: list[DatasetSpecResult] = Field(default_factory=list)


class ExecutionConfigurationValidationReport(BaseModel):
    """Result returned by :meth:`DerivaML.validate_execution_configuration`.

    Attributes:
        all_valid: True iff every nested ``valid`` is True AND
            ``cross_spec_issues`` is empty.
        dataset_results: Per-dataset-spec results.
        asset_results: Per-asset-spec results.
        workflow_result: Workflow validation result, or None if the
            config has no workflow set (a config with no workflow is
            allowed; the workflow can be supplied later via
            ``create_execution(workflow=...)``).
        cross_spec_issues: Consistency issues spanning two or more
            specs (duplicate RIDs, version conflicts, role conflicts).
    """

    model_config = ConfigDict(extra="forbid")

    all_valid: bool
    dataset_results: list[DatasetSpecResult] = Field(default_factory=list)
    asset_results: list[AssetSpecResult] = Field(default_factory=list)
    workflow_result: WorkflowSpecResult | None = None
    cross_spec_issues: list[CrossSpecIssue] = Field(default_factory=list)
