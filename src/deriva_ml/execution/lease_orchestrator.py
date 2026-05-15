"""Crash-recovery entry point for the two-phase RID lease protocol.

Exposes :func:`reconcile_pending_leases` — workspace-open and
``resume_execution`` call sites — for the case where a prior
process crashed between writing ``leasing`` rows to SQLite and
finalizing them after the ERMrest_RID_Lease POST. With the
production-dead acquire path retired (audit §1.6), the function
exists for crash-recovery completeness; in practice the
``leasing`` table never grows because no production writer
populates it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import quote

from deriva_ml.core.logging_config import get_logger
from deriva_ml.execution.rid_lease import PENDING_ROWS_LEASE_CHUNK

if TYPE_CHECKING:
    from deriva.core import ErmrestCatalog

    from deriva_ml.execution.state_store import ExecutionStateStore

logger = get_logger(__name__)


def reconcile_pending_leases(
    *,
    store: "ExecutionStateStore",
    catalog: "ErmrestCatalog",
    execution_rid: str | None = None,
) -> None:
    """Recover from a crash during the two-phase lease protocol.

    Finds pending_rows in status='leasing' (the intermediate state
    between SQLite write and POST response) and asks ERMrest_RID_Lease
    whether each token made it to the server.

    Per-token outcomes:
    - Token exists on server → adopt the server RID, status → 'leased'.
    - Token doesn't exist → POST never landed, revert to 'staged' so
      the next acquire_leases reissues.

    Call sites:
    - On workspace open (no execution_rid arg: sweep all executions) — F5.
    - On resume_execution of a specific rid (pass execution_rid) — F6.

    Args:
        store: ExecutionStateStore.
        catalog: Live ErmrestCatalog.
        execution_rid: If None, reconcile across the whole workspace.
            Otherwise scope to one execution (cheaper — typical for
            resume_execution JIT reconciliation).

    Example:
        >>> # Workspace-wide startup reconciliation:
        >>> reconcile_pending_leases(store=store, catalog=ml.catalog)
        >>> # Per-execution on resume:
        >>> reconcile_pending_leases(
        ...     store=store, catalog=ml.catalog,
        ...     execution_rid="EXE-A",
        ... )
    """
    leasing_rows = store.list_leasing_rows(execution_rid=execution_rid)
    if not leasing_rows:
        return

    tokens = [r["lease_token"] for r in leasing_rows if r["lease_token"]]
    if not tokens:
        # Shouldn't happen — leasing rows always carry tokens — but
        # be defensive.
        return

    # Index leasing rows by token for per-chunk apply.
    rows_by_token = {r["lease_token"]: r for r in leasing_rows if r["lease_token"]}

    # Query ERMrest_RID_Lease for the tokens we expect to find there.
    # Use a filter clause: ID=t1;ID=t2;... (ERMrest's in-list syntax).
    # Chunked to stay under URL length limits.
    #
    # Apply outcomes PER CHUNK so partial progress is durable: if the
    # GET fails midway, the chunks already applied are committed in
    # SQLite, and the next reconcile cycle retries the remainder from
    # a clean state.
    adopted = 0
    reverted = 0
    try:
        for i in range(0, len(tokens), PENDING_ROWS_LEASE_CHUNK):
            chunk = tokens[i : i + PENDING_ROWS_LEASE_CHUNK]
            filter_clause = ";".join(f"ID={quote(t, safe='')}" for t in chunk)
            path = f"/entity/public:ERMrest_RID_Lease/{filter_clause}"
            response = catalog.get(path)
            found_by_token = {row["ID"]: row["RID"] for row in response.json()}

            # Apply outcomes for THIS chunk's tokens only.
            for token in chunk:
                if token not in rows_by_token:
                    continue
                if token in found_by_token:
                    store.finalize_pending_lease(
                        lease_token=token,
                        assigned_rid=found_by_token[token],
                    )
                    adopted += 1
                else:
                    store.revert_pending_leasing(lease_token=token)
                    reverted += 1
    except Exception:  # noqa: BLE001 — catalog GET can raise anything; we swallow to preserve partial progress
        logger.warning(
            "reconcile_pending_leases aborted partway through "
            "(adopted=%d, reverted=%d of %d rows); next reconcile will retry",
            adopted,
            reverted,
            len(leasing_rows),
            exc_info=True,
        )
        return

    logger.info(
        "lease reconciliation: %d rows, %d adopted, %d reverted (execution_rid=%s)",
        len(leasing_rows),
        adopted,
        reverted,
        execution_rid or "all",
    )
