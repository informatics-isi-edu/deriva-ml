"""One-time adoption backfill for the DerivaML provenance contract.

Bringing an *existing* catalog to conformance is a migration, not optional
cleanup (the contract is whole-catalog). This module performs the two
conformance-restoring moves the contract's *Adoption and backfill* section
defines — **neither fabricates provenance**:

1. **Attribute orphan dataset versions** — every ``Dataset_Version`` row with a
   null ``Execution`` producer is set to point at the seeded unknown-provenance
   Execution sentinel. This records them as honestly *"unknown origin"*
   (compliant, known-degraded), not as fabricated provenance. (Assets and
   feature values with null producers are a documented follow-up; this first
   version covers ``Dataset_Version``, the ~95%-of-rows bulk on an established
   catalog.)
2. **Abort stranded non-terminal executions** — a ``Created`` / ``Running`` /
   ``Pending_Upload`` row that was abandoned cannot be honestly "completed"; its
   only honest terminal state is ``Aborted``. The backfill transitions these to
   ``Aborted`` with a migration reason.

The backfill **records its own provenance** on the catalog provenance
annotation (when it ran, the counts, a sample of touched RIDs) so the run is
auditable. After it applies, a catalog-wide :func:`audit_provenance` should
report zero null-producer violations and zero stranded non-terminal producers.

**Safety model.** :func:`backfill_provenance` is **dry-run by default** — it
reports exactly what it *would* change and writes nothing. Pass ``apply=True``
to mutate. The CLI wrapper (``scripts/provenance_backfill.py``) defaults to the
same and requires an explicit ``--apply`` flag, so the intended workflow is:
run against **dev** first (dry-run, then ``--apply``), verify with the audit,
then run against **prod**.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from deriva_ml.core.logging_config import get_logger

logger = get_logger(__name__)

# Non-terminal statuses an abandoned execution can be stranded in. The contract
# lists Created / Running / Pending_Upload as the stranded set ("its only honest
# terminal state is Aborted").
_STRANDED_STATUSES = ("Created", "Running", "Pending_Upload")

_BACKFILL_ABORT_REASON = (
    "Aborted by the provenance-contract adoption backfill: this execution was "
    "stranded in a non-terminal state and could not be honestly completed."
)


class BackfillResult(BaseModel):
    """Outcome of :func:`backfill_provenance`.

    Attributes:
        applied: True if the run mutated the catalog; False for a dry-run.
        orphan_datasets: RIDs of Datasets whose version row(s) had a null
            producer and were (or would be) attributed to the unknown-provenance
            Execution sentinel. (The fix is written to the underlying
            ``Dataset_Version`` rows; this reports the owning Dataset identity —
            matching how the audit reports ``null_producer``.)
        stranded_executions: RIDs of executions that were (or would be) aborted.
        sentinel_execution_rid: The Execution sentinel orphans point at.
        notes: Any non-fatal observations (e.g. sentinel missing, skips).
    """

    applied: bool
    orphan_datasets: list[str] = Field(default_factory=list)
    stranded_executions: list[str] = Field(default_factory=list)
    sentinel_execution_rid: str | None = None
    notes: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        verb = "attributed" if self.applied else "would attribute"
        verb2 = "aborted" if self.applied else "would abort"
        return (
            f"{'APPLIED' if self.applied else 'DRY-RUN'}: "
            f"{verb} {len(self.orphan_datasets)} orphan dataset(s) "
            f"to sentinel {self.sentinel_execution_rid}; "
            f"{verb2} {len(self.stranded_executions)} stranded execution(s)."
        )


def _iso_now(ml_instance: Any) -> str:
    """Server-side 'now' (the catalog snaptime) — avoids client clock skew and
    the scripts-can't-call-Date constraint. Returns an ISO timestamp string."""
    return ml_instance.catalog.get("/").json()["snaptime"]


def backfill_provenance(ml_instance: Any, *, apply: bool = False) -> BackfillResult:
    """Run the one-time adoption backfill against ``ml_instance``'s catalog.

    Dry-run by default (``apply=False``): scans and reports, writes nothing.
    With ``apply=True``, attributes orphan ``Dataset_Version`` rows to the
    unknown-provenance Execution sentinel and aborts stranded non-terminal
    executions, then records the run on the catalog provenance annotation.

    Args:
        ml_instance: The bound DerivaML instance (pointed at the target
            catalog — dev first, then prod).
        apply: If False (default), perform a dry-run — report what would change
            without mutating. If True, apply the changes.

    Returns:
        :class:`BackfillResult` describing what was (or would be) changed.

    Raises:
        Nothing for the no-op/empty case. If the Execution sentinel is missing
        (catalog not seeded), orphan attribution is skipped with a note rather
        than fabricating a target.

    Example:
        >>> from deriva_ml import DerivaML  # doctest: +SKIP
        >>> ml = DerivaML(hostname="dev.eye-ai.org", catalog_id="eye-ai")  # doctest: +SKIP
        >>> print(backfill_provenance(ml).summary())  # dry-run  # doctest: +SKIP
        >>> backfill_provenance(ml, apply=True)  # mutate  # doctest: +SKIP
    """
    result = BackfillResult(applied=apply)
    pb = ml_instance.pathBuilder().schemas[ml_instance.ml_schema]

    # Resolve the Execution sentinel (the target for orphan attribution).
    try:
        sentinel_exec_rid = ml_instance.unknown_provenance_execution_rid()
        result.sentinel_execution_rid = sentinel_exec_rid
    except Exception as e:
        sentinel_exec_rid = None
        result.notes.append(
            f"Unknown-provenance Execution sentinel not found ({e}); orphan "
            "attribution skipped. Seed the catalog (initialize_ml_schema) first."
        )

    # ── Move 1: orphan Dataset_Version rows (null Execution producer) ──────
    # We WRITE the fix to the Dataset_Version rows (that's where the producer FK
    # lives), but REPORT by the owning Dataset RID — the artifact identity a user
    # and the audit reason about (the audit flags null_producer by Dataset RID).
    # All orphan version rows are attributed, not just the current one, so even
    # historical versions stop being null.
    version_path = pb.tables["Dataset_Version"]
    orphan_version_rids: list[str] = []
    orphan_dataset_rids: set[str] = set()
    for r in version_path.entities().fetch():
        if not r.get("Execution"):
            orphan_version_rids.append(r["RID"])
            if r.get("Dataset"):
                orphan_dataset_rids.add(r["Dataset"])
    result.orphan_datasets = sorted(orphan_dataset_rids)

    if apply and sentinel_exec_rid and orphan_version_rids:
        version_path.update(
            [{"RID": rid, "Execution": sentinel_exec_rid} for rid in orphan_version_rids]
        )
        logger.info(
            "Backfill: attributed %d orphan Dataset_Version row(s) across %d dataset(s) to sentinel %s",
            len(orphan_version_rids),
            len(orphan_dataset_rids),
            sentinel_exec_rid,
        )

    # ── Move 2: stranded non-terminal executions → Aborted ────────────────
    exe_path = pb.tables["Execution"]
    stranded_rids: list[str] = []
    for r in exe_path.entities().fetch():
        rid = r["RID"]
        # Never abort the sentinel execution itself (it is intentionally
        # Aborted-status substrate, already terminal — and not stranded).
        if rid == sentinel_exec_rid:
            continue
        if r.get("Status") in _STRANDED_STATUSES:
            stranded_rids.append(rid)
    result.stranded_executions = stranded_rids

    if apply and stranded_rids:
        # Direct catalog write: a migration reconciles existing rows rather than
        # driving a live state machine, and Pending_Upload → Aborted is not in
        # the live transition table (which only governs in-process runs).
        exe_path.update(
            [
                {"RID": rid, "Status": "Aborted", "Status_Detail": _BACKFILL_ABORT_REASON}
                for rid in stranded_rids
            ]
        )
        logger.info("Backfill: aborted %d stranded execution(s)", len(stranded_rids))

    # ── Record the backfill's own provenance on the catalog annotation ────
    if apply and (orphan_version_rids or stranded_rids):
        _record_backfill_provenance(ml_instance, result)

    logger.info("Provenance backfill complete: %s", result.summary())
    return result


# Catalog provenance-annotation tag for backfill self-provenance.
_BACKFILL_ANNOTATION_TAG = "tag:deriva-ml,provenance-backfill"


def _record_backfill_provenance(ml_instance: Any, result: BackfillResult) -> None:
    """Record when the backfill ran and what it touched, on the catalog
    provenance annotation — so a sentinel-attributed row is distinguishable as
    'backfilled on date X' rather than 'always unknown'.

    Best-effort: a failure to record self-provenance must not undo the
    already-applied conformance changes (warn, don't raise).
    """
    try:
        when = _iso_now(ml_instance)
        record = {
            "when": when,
            "orphan_datasets_attributed": len(result.orphan_datasets),
            "stranded_executions_aborted": len(result.stranded_executions),
            "sentinel_execution_rid": result.sentinel_execution_rid,
            # A bounded sample so the annotation does not balloon on a catalog
            # with tens of thousands of orphans.
            "sample_datasets": result.orphan_datasets[:50],
            "sample_stranded_executions": result.stranded_executions[:50],
        }
        model = ml_instance.catalog.getCatalogModel()
        existing = model.annotations.get(_BACKFILL_ANNOTATION_TAG, [])
        if not isinstance(existing, list):
            existing = [existing]
        existing.append(record)
        model.annotations[_BACKFILL_ANNOTATION_TAG] = existing
        model.apply()
        logger.info("Backfill self-provenance recorded on catalog annotation (%s).", when)
    except Exception as e:
        result.notes.append(f"Could not record backfill self-provenance annotation: {e}")
        logger.warning("Backfill self-provenance annotation failed (changes still applied): %s", e)
