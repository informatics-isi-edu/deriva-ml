"""Runtime enforcement of the DerivaML provenance contract.

This module holds the framework-internal checks that fire at the moment an
execution becomes an **artifact-producer** — i.e. when it writes its first
durable artifact (a dataset version, a feature value, or an uploaded asset).
See ``docs/reference/provenance-contract.md`` for the normative contract.

Currently implemented:

- **No-input check** (:func:`ensure_artifact_producer_has_input`): an
  artifact-producer that has declared **no** input — no ``datasets=`` dataset,
  no ``assets=`` asset, no registered input ``File`` — gets an explicit
  ``Input`` edge to the seeded *unknown-provenance File* sentinel, so the gap
  is recorded as "input is explicitly unknown" rather than left absent. Per the
  contract's one-rule handling of violations, this is a **warn-and-mark**, never
  a hard failure: the work already happened and refusing to record it would lose
  provenance, not improve it (Goal 4).

The check is **idempotent** — it is safe to call at every durable-artifact write
path, and calling it more than once for the same execution links the sentinel at
most once (the association insert uses ``on_conflict_skip``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deriva_ml.core.enums import ExecAssetType
from deriva_ml.core.logging_config import get_logger

if TYPE_CHECKING:
    from deriva_ml.core.definitions import RID

logger = get_logger(__name__)


def _execution_has_input(ml_instance: Any, execution_rid: str) -> bool:
    """Return True if the execution has any declared input edge.

    An input is any of: a ``Dataset_Execution`` row (a ``datasets=`` input), or
    a ``{Asset}_Execution`` row with ``Asset_Role="Input"`` (an ``assets=`` /
    ``File`` input, including a previously-linked sentinel). This is the
    predicate the no-input check gates on.

    Args:
        ml_instance: The bound DerivaML instance.
        execution_rid: RID of the execution to inspect.

    Returns:
        True if at least one input edge exists, else False.
    """
    from deriva_ml.execution._helpers import list_assets, list_input_datasets

    if list_input_datasets(ml_instance=ml_instance, execution_rid=execution_rid):
        return True
    if list_assets(ml_instance=ml_instance, execution_rid=execution_rid, asset_role="Input"):
        return True
    return False


def _link_file_sentinel_as_input(ml_instance: Any, execution_rid: str) -> "RID":
    """Link the unknown-provenance File sentinel to the execution as an Input.

    Writes the ``File_Execution`` association row (``Asset_Role="Input"``) and
    the directional ``Input_File`` Asset_Type tag on the sentinel — the same
    pair :func:`asset_upload.update_asset_execution_table` writes for a
    downloaded input asset, but for the already-existing sentinel File row
    rather than a freshly-uploaded one. Idempotent via ``on_conflict_skip``.

    Args:
        ml_instance: The bound DerivaML instance.
        execution_rid: RID of the artifact-producing execution.

    Returns:
        The sentinel File's RID (the input now linked to the execution).
    """
    sentinel_file_rid = ml_instance.unknown_provenance_file_rid()

    pb = ml_instance.pathBuilder()
    assoc, file_fk, execution_fk = ml_instance.model.find_association("File", "Execution")
    assoc_path = pb.schemas[assoc.schema.name].tables[assoc.name]
    assoc_path.insert(
        [
            {
                file_fk: sentinel_file_rid,
                execution_fk: execution_rid,
                "Asset_Role": "Input",
            }
        ],
        on_conflict_skip=True,
    )

    # Directional Input_File tag on the sentinel (queryable as "ever an input").
    type_assoc, _, _ = ml_instance.model.find_association("File", "Asset_Type")
    type_path = pb.schemas[type_assoc.schema.name].tables[type_assoc.name]
    type_path.insert(
        [{"File": sentinel_file_rid, "Asset_Type": ExecAssetType.input_file.value}],
        on_conflict_skip=True,
    )

    return sentinel_file_rid


def ensure_artifact_producer_has_input(ml_instance: Any, execution_rid: str) -> None:
    """No-input check: an artifact-producer with no declared input gets the
    unknown-provenance File sentinel as an explicit Input edge.

    Call this at any durable-artifact write path (dataset-version authorship,
    feature-value write, asset commit) — whichever fires first is the moment the
    execution becomes an artifact-producer (contract §"Timing of the
    artifact-producer rules"). Idempotent and safe to call at every path.

    If the execution already has an input, this is a no-op. If it has none, the
    sentinel is linked and a **loud, structured warning** is emitted (the
    durable mark is the sentinel edge itself, which the audit can later find).
    Never raises on the no-input condition and never fails the caller's work —
    the warn-and-mark rule (Goal 4: never fake, never destroy).

    Args:
        ml_instance: The bound DerivaML instance.
        execution_rid: RID of the execution that just wrote a durable artifact.
    """
    # The whole no-input check is provenance bookkeeping layered on top of the
    # caller's already-committed work. Per the contract's one-rule handling of
    # violations (warn-and-mark, never fail the commit or roll back — Goal 4),
    # any failure here (missing sentinel, schema without File_Execution, catalog
    # blip) must NOT propagate and take down a successful upload. Swallow with a
    # loud warning so the lapse is visible without breaking the run.
    try:
        # The unknown-provenance Execution sentinel is exempt bootstrap substrate
        # — it carries no input obligation. (It is the producer that producerless
        # artifacts attribute to; linking it a sentinel input would be circular.)
        if execution_rid == ml_instance.unknown_provenance_execution_rid():
            return

        if _execution_has_input(ml_instance, execution_rid):
            return

        sentinel_file_rid = _link_file_sentinel_as_input(ml_instance, execution_rid)
        logger.warning(
            "Provenance no-input check: execution %s produced a durable artifact "
            "but declared no input (no datasets=, no assets=, no input File). "
            "Linked the unknown-provenance File sentinel %s as an explicit Input "
            "edge so the gap is recorded as 'input unknown', not absent. To record "
            "real provenance, declare the run's inputs in its ExecutionConfiguration.",
            execution_rid,
            sentinel_file_rid,
        )
    except Exception as e:
        logger.warning(
            "Provenance no-input check could not be applied to execution %s "
            "(%s). The run's work is unaffected; the input-provenance gap was "
            "not marked. This is recoverable later by the provenance audit/backfill.",
            execution_rid,
            e,
        )
