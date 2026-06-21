"""Read-only provenance audit (DerivaML provenance contract §Audit).

A catalog-wide, **read-only** scan that surfaces every violation of the
complete-provenance predicate in one place, plus a separate *known-degraded*
report of compliant-but-thin provenance (sentinel attributions). It mutates
nothing — it detects; humans decide and resolve (Goal 4, no autonomous sweep).
See ``docs/reference/provenance-contract.md`` §Audit for the normative spec.

The audit tests the complete-provenance predicate clause by clause; each clause
maps to one finding category. Sentinel-seeded substrate (the unknown-provenance
Workflow / File / Execution rows) is **exempt** — it is the catalog's substrate,
not data, and carries no producer/input obligation.

Result shape (consumed by callers and the contract tests):

- :attr:`ProvenanceAuditReport.violations` — must-fix findings.
- :attr:`ProvenanceAuditReport.known_degraded` — compliant, surfaced for
  visibility (sentinel-attributed producers, sentinel-File-only inputs).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from deriva_ml.core.constants import SENTINEL_EXECUTION_DESCRIPTION, SENTINEL_FILE_URL, SENTINEL_WORKFLOW_URL
from deriva_ml.core.logging_config import get_logger

logger = get_logger(__name__)


class ProvenanceFinding(BaseModel):
    """One audit finding — a single artifact/execution that failed (or thinned)
    a complete-provenance clause.

    Attributes:
        category: Short machine key for the clause, e.g. ``"null_producer"``,
            ``"zero_input"``, ``"stranded_non_terminal"``,
            ``"sentinel_producer"``.
        rid: RID of the offending artifact or execution.
        kind: The row's kind — ``"Execution"``, ``"Dataset"``, ``"Asset"``,
            ``"Feature"``.
        detail: Human-readable explanation of what is wrong / thin.
    """

    category: str
    rid: str
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.category}] {self.kind} {self.rid}: {self.detail}"


class ProvenanceAuditReport(BaseModel):
    """Result of :func:`audit_provenance`.

    Separates **violations** (must fix) from **known_degraded** (compliant but
    visible) so the two are never conflated — a sentinel attribution is
    compliant, not a violation.

    Attributes:
        violations: Findings that violate the complete-provenance predicate.
        known_degraded: Compliant-but-thin findings (sentinel attributions).
    """

    violations: list[ProvenanceFinding] = Field(default_factory=list)
    known_degraded: list[ProvenanceFinding] = Field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True when there are no violations (known-degraded does not count)."""
        return not self.violations

    def summary(self) -> str:
        """One-line summary: counts of each bucket."""
        return f"{len(self.violations)} violation(s), {len(self.known_degraded)} known-degraded"


def _sentinel_rids(ml_instance: Any) -> dict[str, str | None]:
    """Resolve the three sentinel RIDs (or None when a catalog predates them).

    Returns a dict with keys ``workflow``, ``file``, ``execution``. A missing
    sentinel maps to None — the audit still runs; rows simply cannot be exempted
    against an absent sentinel.
    """
    pb = ml_instance.pathBuilder().schemas[ml_instance.ml_schema]

    def _first(table, col, value):
        rows = list(table.filter(col == value).entities().fetch())
        return rows[0]["RID"] if rows else None

    wf = pb.tables["Workflow"]
    fl = pb.tables["File"]
    ex = pb.tables["Execution"]
    return {
        "workflow": _first(wf, wf.URL, SENTINEL_WORKFLOW_URL),
        "file": _first(fl, fl.URL, SENTINEL_FILE_URL),
        "execution": _first(ex, ex.Description, SENTINEL_EXECUTION_DESCRIPTION),
    }


def audit_provenance(ml_instance: Any) -> ProvenanceAuditReport:
    """Run the read-only provenance audit over the whole catalog.

    Scans every dataset version's producer edge and classifies it:

    - **null producer** (no ``Dataset_Version.Execution`` and not attributed to
      the unknown-provenance Execution sentinel) → **violation**;
    - **sentinel producer** (attributed to the unknown-provenance Execution
      sentinel) → **known-degraded** (compliant, but the producer is the "no
      real producer" marker);
    - real producer → conformant, not reported.

    The sentinel-seeded substrate (unknown-provenance Workflow/File/Execution)
    is exempt and never reported.

    This is the initial audit covering the null-producer / sentinel-producer
    clauses (the producer side of the predicate, exercised by the contract's
    G-series tests). The remaining predicate clauses — stranded non-terminal
    producers, degenerate workflow URL/checksum, zero declared input, untagged
    output, ``Failed`` missing a reason — extend this same report.

    Args:
        ml_instance: The bound DerivaML instance.

    Returns:
        ProvenanceAuditReport with ``violations`` and ``known_degraded``. Never
        mutates the catalog (Goal 4).
    """
    report = ProvenanceAuditReport()
    sentinels = _sentinel_rids(ml_instance)
    sentinel_exec_rid = sentinels["execution"]

    pb = ml_instance.pathBuilder().schemas[ml_instance.ml_schema]
    version_path = pb.tables["Dataset_Version"]

    # Group version rows by Dataset so we examine each dataset's CURRENT
    # (highest) version — the producer the lineage walk reports. A null on the
    # current version is the violation; older versions are history.
    rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in version_path.entities().fetch():
        ds = row.get("Dataset")
        if ds:
            rows_by_dataset.setdefault(ds, []).append(row)

    def _semver_key(row: dict[str, Any]) -> tuple[int, ...]:
        v = row.get("Version") or "0.0.0"
        try:
            return tuple(int(p) for p in v.split("."))
        except ValueError:
            return (0,)

    for dataset_rid, version_rows in rows_by_dataset.items():
        current = max(version_rows, key=_semver_key)
        producer = current.get("Execution")

        if producer is None:
            report.violations.append(
                ProvenanceFinding(
                    category="null_producer",
                    rid=dataset_rid,
                    kind="Dataset",
                    detail=(
                        "Current Dataset_Version has a null producer (no producing "
                        "execution and not attributed to the unknown-provenance "
                        "Execution sentinel). Run the adoption backfill to attribute it."
                    ),
                )
            )
        elif sentinel_exec_rid is not None and producer == sentinel_exec_rid:
            report.known_degraded.append(
                ProvenanceFinding(
                    category="sentinel_producer",
                    rid=dataset_rid,
                    kind="Dataset",
                    detail=(
                        "Attributed to the unknown-provenance Execution sentinel — "
                        "compliant (the gap is recorded explicitly), but the producer "
                        "is the 'no real producer' marker."
                    ),
                )
            )

    logger.info("Provenance audit complete: %s", report.summary())
    return report
