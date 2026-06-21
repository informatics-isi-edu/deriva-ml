"""One-time adoption backfill for the DerivaML provenance contract.

Bringing an *existing* catalog to conformance is a migration, not optional
cleanup (the contract is whole-catalog). This module performs the
conformance-restoring moves the contract's *Adoption and backfill* section
defines — **none fabricates provenance**:

0. **Ensure the sentinels exist.** An established catalog predates the
   sentinel-seeding schema-init, so the sentinels are typically absent. Under
   ``apply``, the backfill seeds them first (idempotent) so adoption is a single
   self-contained operation. (A dry-run can't resolve a sentinel RID and reports
   that ``--apply`` would seed.)
1. **Attribute orphan dataset versions** — every ``Dataset_Version`` row with a
   null ``Execution`` producer is set to point at the unknown-provenance
   Execution sentinel. This records them as honestly *"unknown origin"*
   (compliant, known-degraded), not as fabricated provenance. (Assets and
   feature values with null producers are a documented follow-up; this first
   version covers ``Dataset_Version``, the ~95%-of-rows bulk on an established
   catalog.)
2. **Abort stranded non-terminal executions** — a ``Created`` / ``Running`` /
   ``Pending_Upload`` row that was abandoned cannot be honestly "completed"; its
   only honest terminal state is ``Aborted``. The backfill transitions these to
   ``Aborted`` with a migration reason.
3. **Unify the prior 'BlackBox' backfill onto the sentinel.** An earlier
   convention attributed legacy datasets to per-dataset "BlackBox" executions
   (under placeholder ``UnknownDatasetProducer`` / ``UnknownAssetProducer``
   workflows). The backfill re-points those ``Dataset_Version`` rows to the single
   unknown-provenance Execution sentinel, then deletes the now-unreferenced
   BlackBox executions and workflows (only when truly unreferenced — checked via
   inbound FKs). One canonical "unknown origin" marker going forward.

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

# Markers that identify the *prior* backfill convention's "BlackBox" rows on an
# established catalog (e.g. eye-ai): two placeholder Workflows
# (UnknownDatasetProducer / UnknownAssetProducer) and per-legacy-dataset
# executions described "BlackBox: reconstructed provenance for legacy dataset ...".
# Matched case-insensitively on Name/Description text rather than by RID, because
# the RIDs differ per catalog (dev vs prod). The new backfill unifies these onto
# the single unknown-provenance Execution sentinel.
_BLACKBOX_WORKFLOW_MARKERS = ("blackbox workflow for legacy", "unknowndatasetproducer", "unknownassetproducer")
_BLACKBOX_EXECUTION_MARKER = "blackbox"


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
        repointed_blackbox_datasets: RIDs of Datasets whose version row(s) were
            (or would be) re-pointed from a legacy BlackBox execution to the new
            unknown-provenance Execution sentinel — unifying the prior backfill
            convention onto the single sentinel.
        deleted_blackbox_executions: RIDs of legacy BlackBox executions that
            were (or would be) deleted after re-pointing left them unreferenced.
        deleted_blackbox_workflows: RIDs of legacy BlackBox workflows that were
            (or would be) deleted after their executions were removed.
        retained_blackbox_executions: BlackBox execution RIDs NOT deleted because
            something still references them (reported, never force-deleted).
        notes: Any non-fatal observations (e.g. sentinel missing, skips).
    """

    applied: bool
    orphan_datasets: list[str] = Field(default_factory=list)
    stranded_executions: list[str] = Field(default_factory=list)
    sentinel_execution_rid: str | None = None
    repointed_blackbox_datasets: list[str] = Field(default_factory=list)
    deleted_blackbox_executions: list[str] = Field(default_factory=list)
    deleted_blackbox_workflows: list[str] = Field(default_factory=list)
    retained_blackbox_executions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        verb = "attributed" if self.applied else "would attribute"
        verb2 = "aborted" if self.applied else "would abort"
        verb3 = "re-pointed" if self.applied else "would re-point"
        verb4 = "deleted" if self.applied else "would delete"
        return (
            f"{'APPLIED' if self.applied else 'DRY-RUN'}: "
            f"{verb} {len(self.orphan_datasets)} orphan dataset(s) "
            f"to sentinel {self.sentinel_execution_rid}; "
            f"{verb2} {len(self.stranded_executions)} stranded execution(s); "
            f"{verb3} {len(self.repointed_blackbox_datasets)} BlackBox dataset(s) "
            f"and {verb4} {len(self.deleted_blackbox_executions)} BlackBox exec(s) "
            f"+ {len(self.deleted_blackbox_workflows)} BlackBox workflow(s)."
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

    # Adoption step 0: ensure the sentinels exist. An ESTABLISHED catalog (the
    # backfill's whole reason to exist) predates the sentinel-seeding schema-init,
    # so the sentinels are typically absent. Seed them here (idempotent) so the
    # backfill is a single self-contained adoption operation — no separate
    # re-run-schema-init step. Seeding is a WRITE, so it only happens under
    # ``apply``; a dry-run reports that it would seed.
    try:
        sentinel_exec_rid = ml_instance.unknown_provenance_execution_rid()
    except Exception:
        sentinel_exec_rid = None

    if sentinel_exec_rid is None:
        if apply:
            from deriva_ml.schema.create_schema import _ensure_sentinels

            _ensure_sentinels(pb)
            sentinel_exec_rid = ml_instance.unknown_provenance_execution_rid()
            result.notes.append("Seeded the unknown-provenance sentinels (catalog predated sentinel schema-init).")
            logger.info("Backfill: seeded sentinels on a catalog that lacked them.")
        else:
            result.notes.append(
                "Unknown-provenance sentinels are absent; --apply would seed them first "
                "(idempotent). Dry-run cannot resolve a sentinel RID, so orphan/BlackBox "
                "counts below are what WOULD be attributed once seeded."
            )

    result.sentinel_execution_rid = sentinel_exec_rid

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

    # ── Move 3: unify the prior 'BlackBox' backfill onto the sentinel ──────
    bb_changed = _unify_blackbox(ml_instance, result, apply=apply, sentinel_exec_rid=sentinel_exec_rid)

    # ── Record the backfill's own provenance on the catalog annotation ────
    if apply and (orphan_version_rids or stranded_rids or bb_changed):
        _record_backfill_provenance(ml_instance, result)

    logger.info("Provenance backfill complete: %s", result.summary())
    return result


def _execution_fk_columns(ml_instance: Any) -> list[tuple[str, str, str]]:
    """Discover every (schema, table, column) that holds an FK to Execution.RID.

    Walks all tables' foreign keys and selects those whose ``column_map`` targets
    the Execution table — the schema-agnostic way to enumerate inbound references
    (deriva-py's ermrest_model has no ``Table.referenced_by``). Cached nowhere;
    cheap enough to recompute per backfill run.
    """
    model = ml_instance.model.model
    refs: list[tuple[str, str, str]] = []
    for schema in model.schemas.values():
        for table in schema.tables.values():
            for fk in table.foreign_keys:
                for local_col, ref_col in fk.column_map.items():
                    if ref_col.table.name == "Execution" and ref_col.name == "RID":
                        refs.append((schema.name, table.name, local_col.name))
    return refs


def _execution_is_referenced(
    ml_instance: Any,
    execution_rid: str,
    *,
    exclude_self_rid: str | None = None,
    ignore_version_rids: set[str] | None = None,
) -> bool:
    """True if any catalog row (other than the execution itself) still points at
    ``execution_rid`` via an inbound FK.

    Enumerates every (table, column) with an FK to Execution.RID and checks each
    for a row referencing this RID. The 'fully unreferenced?' gate the delete
    step depends on, so a BlackBox execution is never deleted while anything
    still attributes to it.

    ``ignore_version_rids`` (dry-run use): Dataset_Version rows in this set are
    treated as already re-pointed away — used to predict the post-re-point state
    before the writes happen.
    """
    pb = ml_instance.pathBuilder()
    ignore = ignore_version_rids or set()
    for schema_name, table_name, col in _execution_fk_columns(ml_instance):
        tp = pb.schemas[schema_name].tables[table_name]
        try:
            rows = list(tp.filter(tp.columns[col] == execution_rid).entities().fetch())
        except Exception:
            # If a referencing table can't be queried, be conservative: treat as
            # referenced (do not delete) rather than risk an orphaned FK.
            return True
        for row in rows:
            if exclude_self_rid is not None and row.get("RID") == exclude_self_rid:
                continue
            if table_name == "Dataset_Version" and row.get("RID") in ignore:
                continue  # this reference will be (or was) cleared by the re-point
            return True
    return False


def _unify_blackbox(
    ml_instance: Any,
    result: BackfillResult,
    *,
    apply: bool,
    sentinel_exec_rid: str | None,
) -> bool:
    """Unify the prior 'BlackBox' backfill convention onto the single sentinel.

    The prior convention created per-legacy-dataset BlackBox executions (under
    placeholder Workflows ``UnknownDatasetProducer`` / ``UnknownAssetProducer``)
    and pointed legacy ``Dataset_Version`` rows at them. This:

    1. Re-points every ``Dataset_Version`` whose producer is a BlackBox execution
       to the unknown-provenance Execution sentinel (one canonical marker).
    2. Deletes each BlackBox execution that is fully unreferenced afterwards.
    3. Deletes each BlackBox workflow that has no remaining executions.

    Deletes only when truly unreferenced (checked via inbound FKs); anything
    still referenced is reported in ``retained_blackbox_executions``, never
    force-deleted. BlackBox rows are identified by Name/Description markers (not
    RID), so this works on any catalog.

    Returns True if anything changed (or, in dry-run, would change).
    """
    pb = ml_instance.pathBuilder().schemas[ml_instance.ml_schema]
    wf_path = pb.tables["Workflow"]
    exe_path = pb.tables["Execution"]
    version_path = pb.tables["Dataset_Version"]

    # Identify BlackBox workflows by marker text.
    def _is_bb_workflow(r: dict) -> bool:
        blob = " ".join(str(r.get(k) or "") for k in ("Name", "Description")).lower()
        return any(m in blob for m in _BLACKBOX_WORKFLOW_MARKERS)

    bb_workflow_rids = {r["RID"] for r in wf_path.entities().fetch() if _is_bb_workflow(r)}
    if not bb_workflow_rids:
        return False  # No prior BlackBox convention on this catalog.

    # BlackBox executions = executions under a BlackBox workflow, OR executions
    # whose description carries the BlackBox marker (belt-and-suspenders).
    bb_exec_rids = {
        r["RID"]
        for r in exe_path.entities().fetch()
        if r.get("Workflow") in bb_workflow_rids
        or _BLACKBOX_EXECUTION_MARKER in str(r.get("Description") or "").lower()
    }

    # Always REPORT the BlackBox footprint (so a dry-run on an unseeded catalog
    # still shows what unification would do). Only the WRITES are gated on having
    # a sentinel — which, under --apply, step 0 has already seeded.
    # 1. Re-point Dataset_Version rows from BlackBox execs → sentinel.
    repointed_version_rids: list[str] = []
    repointed_dataset_rids: set[str] = set()
    for r in version_path.entities().fetch():
        if r.get("Execution") in bb_exec_rids:
            repointed_version_rids.append(r["RID"])
            if r.get("Dataset"):
                repointed_dataset_rids.add(r["Dataset"])
    result.repointed_blackbox_datasets = sorted(repointed_dataset_rids)

    if apply and sentinel_exec_rid and repointed_version_rids:
        version_path.update(
            [{"RID": rid, "Execution": sentinel_exec_rid} for rid in repointed_version_rids]
        )
        logger.info(
            "BlackBox unify: re-pointed %d Dataset_Version row(s) across %d dataset(s) to sentinel %s",
            len(repointed_version_rids),
            len(repointed_dataset_rids),
            sentinel_exec_rid,
        )

    # 2. Delete BlackBox executions that are now fully unreferenced.
    #    In dry-run, the re-point has NOT happened, so the Dataset_Version rows
    #    still reference the exec — report what WOULD be deletable assuming the
    #    re-point lands (i.e. ignore references that the re-point will clear).
    deletable_execs: list[str] = []
    repoint_set = set(repointed_version_rids)
    for ex_rid in sorted(bb_exec_rids):
        # After apply, the re-point has landed, so a plain reference check is
        # exact. In dry-run, predict the post-re-point state by ignoring the
        # Dataset_Version rows we are about to re-point away from this exec.
        referenced = _execution_is_referenced(
            ml_instance,
            ex_rid,
            exclude_self_rid=ex_rid,
            ignore_version_rids=None if apply else repoint_set,
        )
        if referenced:
            result.retained_blackbox_executions.append(ex_rid)
        else:
            deletable_execs.append(ex_rid)

    if apply and deletable_execs:
        # Delete per-RID (portable across datapath versions).
        for ex_rid in deletable_execs:
            exe_path.filter(exe_path.RID == ex_rid).delete()
        logger.info("BlackBox unify: deleted %d unreferenced BlackBox execution(s)", len(deletable_execs))
    result.deleted_blackbox_executions = deletable_execs

    # 3. Delete BlackBox workflows that have no remaining executions.
    deletable_workflows: list[str] = []
    for wf_rid in sorted(bb_workflow_rids):
        # After exec deletion (apply) or predicted deletion (dry-run), does any
        # execution still reference this workflow?
        remaining = [
            r for r in exe_path.entities().fetch()
            if r.get("Workflow") == wf_rid and (apply or r["RID"] not in set(deletable_execs))
        ]
        if remaining:
            result.notes.append(
                f"BlackBox workflow {wf_rid} retained: {len(remaining)} execution(s) still reference it."
            )
        else:
            deletable_workflows.append(wf_rid)

    if apply and deletable_workflows:
        for wf_rid in deletable_workflows:
            wf_path.filter(wf_path.RID == wf_rid).delete()
        logger.info("BlackBox unify: deleted %d unreferenced BlackBox workflow(s)", len(deletable_workflows))
    result.deleted_blackbox_workflows = deletable_workflows

    return bool(repointed_version_rids or deletable_execs or deletable_workflows)


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
            "blackbox_datasets_repointed": len(result.repointed_blackbox_datasets),
            "blackbox_executions_deleted": len(result.deleted_blackbox_executions),
            "blackbox_workflows_deleted": len(result.deleted_blackbox_workflows),
            "sentinel_execution_rid": result.sentinel_execution_rid,
            # A bounded sample so the annotation does not balloon on a catalog
            # with tens of thousands of orphans.
            "sample_datasets": result.orphan_datasets[:50],
            "sample_stranded_executions": result.stranded_executions[:50],
            "sample_blackbox_datasets": result.repointed_blackbox_datasets[:50],
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
