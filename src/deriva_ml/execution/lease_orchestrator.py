"""Orchestrator for the two-phase RID lease protocol.

Composes ExecutionStateStore's lease helpers with rid_lease's POST
machinery. One entry point: acquire_leases_for_execution. Called by
handle.rid property and by the upload-engine drain.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deriva_ml.execution.rid_lease import generate_lease_token, post_lease_batch
from deriva_ml.execution.state_store import PendingRowStatus

if TYPE_CHECKING:
    from deriva.core import ErmrestCatalog

    from deriva_ml.execution.state_store import ExecutionStateStore

logger = logging.getLogger(__name__)


def acquire_leases_for_execution(
    *,
    store: "ExecutionStateStore",
    catalog: "ErmrestCatalog",
    execution_rid: str,
    pending_ids: list[int],
) -> None:
    """Transition the given pending rows from 'staged' to 'leased',
    assigning server-issued RIDs.

    Skips rows already in status='leased' (idempotent). Rows in
    other intermediate states (leasing, uploading, uploaded, failed)
    are also skipped — the orchestrator only promotes staged→leased.

    Two-phase protocol:
      1. Generate tokens, mark rows 'leasing' in SQLite (committed
         before the POST).
      2. POST batch to ERMrest_RID_Lease.
      3. On success: finalize each row with its assigned RID
         (status → 'leased').
      4. On POST failure: revert all rows we marked in step 1
         (status → 'staged'; token cleared).

    Crash recovery is handled in Task F4 (reconcile at startup).

    Args:
        store: The ExecutionStateStore holding SQLite state.
        catalog: Live ErmrestCatalog for POSTing to ERMrest_RID_Lease.
        execution_rid: For logging + scoping; all pending_ids must
            belong to this execution (not enforced here; caller's
            concern).
        pending_ids: pending_rows.id values to lease.

    Raises:
        Exception: Whatever the catalog POST raises. Before
            propagating, the orchestrator reverts any rows it had
            marked 'leasing' back to 'staged'.

    Example:
        >>> acquire_leases_for_execution(
        ...     store=store, catalog=ml.catalog,
        ...     execution_rid="EXE-A",
        ...     pending_ids=[1, 2, 3],
        ... )
    """
    if not pending_ids:
        return

    # Filter to rows actually in 'staged'. Build a (pending_id, token)
    # list; the order maps to the POST body order, which maps to the
    # response order in _MockLeaseCatalog and in real ERMrest.
    rows_to_lease: list[tuple[int, str]] = []
    all_rows = {r["id"]: r for r in store.list_pending_rows(execution_rid=execution_rid)}
    for pid in pending_ids:
        row = all_rows.get(pid)
        if row is None:
            logger.warning(
                "acquire_leases: pending_id %d not in execution %s; skipping",
                pid, execution_rid,
            )
            continue
        if row["status"] != str(PendingRowStatus.staged):
            # Already leased or past; skip silently.
            continue
        rows_to_lease.append((pid, generate_lease_token()))

    if not rows_to_lease:
        return

    # Phase 1: write 'leasing' + token to SQLite, committed.
    # This MUST happen before the POST so that if we crash, the token
    # is in SQLite and we can look it up on the server at reconcile.
    for pid, token in rows_to_lease:
        store.mark_pending_leasing(pid, lease_token=token)

    # Phase 2: POST the batch. On failure, revert all.
    tokens = [t for _, t in rows_to_lease]
    try:
        assigned = post_lease_batch(catalog=catalog, tokens=tokens)
    except Exception:
        logger.warning(
            "acquire_leases: POST failed for execution %s; reverting %d rows to staged",
            execution_rid, len(rows_to_lease),
        )
        for _, token in rows_to_lease:
            store.revert_pending_leasing(lease_token=token)
        raise

    # Phase 3: finalize each row with its assigned RID.
    for _, token in rows_to_lease:
        assigned_rid = assigned.get(token)
        if assigned_rid is None:
            # Server response missing this token. Revert just this
            # row; leave the others (they did succeed).
            logger.warning(
                "acquire_leases: token %s missing from server response "
                "for execution %s; reverting that row",
                token, execution_rid,
            )
            store.revert_pending_leasing(lease_token=token)
        else:
            store.finalize_pending_lease(lease_token=token, assigned_rid=assigned_rid)

    logger.debug(
        "acquire_leases: %d rows leased for execution %s",
        len(rows_to_lease), execution_rid,
    )
