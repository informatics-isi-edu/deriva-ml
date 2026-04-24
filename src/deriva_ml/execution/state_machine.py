"""Execution-lifecycle state machine.

Per spec §2.2. All transitions go through this module; direct updates
to executions.status from elsewhere are a bug. The module:

- Defines the allowed (from, to) pairs as a set-based table.
- Validates transitions at call time.
- Owns the SQLite-write + catalog-sync path (with sync_pending
  soft-fail on catalog failure).
- Provides the disagreement-resolution logic used by just-in-time
  reconciliation in resume_execution.

Why a module and not a class: the state machine is functional —
transitions take (store, catalog, rid, target, metadata). The
ExecutionStateStore and ErmrestCatalog live elsewhere; this module
wires them together without owning lifecycle of either.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from deriva_ml.core.exceptions import (
    DerivaMLDataError,
    DerivaMLException,
    DerivaMLStateInconsistency,
)
from deriva_ml.execution.state_store import ExecutionStatus

if TYPE_CHECKING:
    from deriva.core import ErmrestCatalog  # noqa: F401  (string-annotation only)

    from deriva_ml.core.connection_mode import ConnectionMode  # noqa: F401  (string-annotation only)
    from deriva_ml.execution.state_store import ExecutionStateStore  # noqa: F401  (string-annotation only)

logger = logging.getLogger(__name__)


__all__ = [
    "ALLOWED_TRANSITIONS",
    "InvalidTransitionError",
    "validate_transition",
    "transition",
    "flush_pending_sync",
    "reconcile_with_catalog",
    "create_catalog_execution",
]


class InvalidTransitionError(DerivaMLException):
    """Raised when a requested status transition is not in the
    allowed table.

    This is a programming error, not a runtime-data error — allowed
    transitions are a compile-time decision and something outside the
    state machine tried to bypass the rules.
    """

    pass


# Allowed transitions. Kept explicit (not derived) so the table is
# the single source of truth and easy to read.
#
# The state diagram (spec §2.2):
#
#     created → running → {stopped, failed} → pending_upload → {uploaded, failed}
#                                                      ↑             │
#                                                      └──── retry ──┘
#     created / running / stopped / failed → aborted (terminal)
#     failed → pending_upload (retry_failed path)

ALLOWED_TRANSITIONS: frozenset[tuple[ExecutionStatus, ExecutionStatus]] = frozenset({
    # Happy path
    (ExecutionStatus.Created, ExecutionStatus.Running),
    (ExecutionStatus.Running, ExecutionStatus.Stopped),
    (ExecutionStatus.Stopped, ExecutionStatus.Pending_Upload),
    (ExecutionStatus.Pending_Upload, ExecutionStatus.Uploaded),

    # Failure paths
    (ExecutionStatus.Running, ExecutionStatus.Failed),
    (ExecutionStatus.Pending_Upload, ExecutionStatus.Failed),

    # Retry from upload failure back into upload
    (ExecutionStatus.Failed, ExecutionStatus.Pending_Upload),

    # Abort is legal from any pre-terminal state. 'uploaded' is
    # terminal — we don't allow abort after successful upload.
    (ExecutionStatus.Created, ExecutionStatus.Aborted),
    (ExecutionStatus.Running, ExecutionStatus.Aborted),
    (ExecutionStatus.Stopped, ExecutionStatus.Aborted),
    (ExecutionStatus.Failed, ExecutionStatus.Aborted),
})


def validate_transition(
    *,
    current: ExecutionStatus,
    target: ExecutionStatus,
) -> None:
    """Verify that (current → target) is in the allowed table.

    Args:
        current: The execution's current status (as read from SQLite).
        target: The requested new status.

    Raises:
        InvalidTransitionError: If the pair is not in
            ALLOWED_TRANSITIONS. Message includes both states.

    Example:
        >>> validate_transition(
        ...     current=ExecutionStatus.Running,
        ...     target=ExecutionStatus.Stopped,
        ... )  # returns None, no raise
    """
    if (current, target) not in ALLOWED_TRANSITIONS:
        raise InvalidTransitionError(
            f"Illegal execution transition {current} → {target}. "
            f"See spec §2.2 for the allowed transition graph."
        )


def transition(
    *,
    store: "ExecutionStateStore",
    catalog: "ErmrestCatalog | None",
    execution_rid: str,
    current: ExecutionStatus,
    target: ExecutionStatus,
    mode: "ConnectionMode",
    extra_fields: dict | None = None,
) -> None:
    """Transition an execution's status, writing SQLite and syncing
    the catalog when online.

    This is the single entry point for all lifecycle status changes.
    Direct writes to executions.status bypass validation and catalog
    sync; don't do it.

    Args:
        store: The ExecutionStateStore owning the SQLite row.
        catalog: The ErmrestCatalog for syncing. Pass None in offline
            mode (attempting to pass a non-None catalog in offline
            mode is a programming error and raises).
        execution_rid: Which execution to transition.
        current: The status we believe the execution is in. The state
            machine does NOT re-read SQLite to determine `current`;
            the caller passed it, typically from a just-prior read.
            This lets the caller do its own consistency check if
            needed.
        target: The status to transition to.
        mode: ConnectionMode. Online → also PUT catalog row; offline
            → only update SQLite, set sync_pending=True.
        extra_fields: Additional executions columns to update in the
            same transaction (start_time, stop_time, error, etc.).

    Raises:
        InvalidTransitionError: If (current, target) is not in
            ALLOWED_TRANSITIONS.
        ValueError: If mode=offline but catalog is not None, or
            mode=online but catalog is None. These are caller bugs.
        NotImplementedError: Online-mode path is implemented in Task C3.

    Example:
        >>> transition(  # doctest: +SKIP
        ...     store=store, catalog=ml.catalog,
        ...     execution_rid="EXE-A",
        ...     current=ExecutionStatus.Running,
        ...     target=ExecutionStatus.Stopped,
        ...     mode=ConnectionMode.online,
        ...     extra_fields={"stop_time": datetime.now(timezone.utc)},
        ... )
    """
    # Runtime import of ConnectionMode — the TYPE_CHECKING import is for
    # annotations only, but we need the class object for `is` comparisons
    # and isinstance checks at the function boundary.
    from deriva_ml.core.connection_mode import ConnectionMode

    validate_transition(current=current, target=target)

    # Consistency: offline must pass catalog=None, online must pass
    # a real catalog. Mismatches indicate a caller bug.
    if mode is ConnectionMode.offline and catalog is not None:
        raise ValueError("offline mode must pass catalog=None")
    if mode is ConnectionMode.online and catalog is None:
        raise ValueError("online mode requires a catalog")

    now = datetime.now(timezone.utc)
    extra_fields = dict(extra_fields or {})
    extra_fields.setdefault("last_activity", now)

    if mode is ConnectionMode.offline:
        # Offline: only SQLite. Set sync_pending so that the next
        # online opportunity will push this status to the catalog.
        store.update_execution(
            execution_rid,
            status=target,
            sync_pending=True,
            **extra_fields,
        )
        logger.debug(
            "offline transition %s: %s → %s (sync_pending)",
            execution_rid, current, target,
        )
        return

    # Online: SQLite first, then catalog PUT. If PUT fails, leave
    # sync_pending=True so a later call (or resume_execution)
    # flushes. We never let catalog failure roll back SQLite — the
    # local view is the source of truth; the catalog catches up.
    #
    # Ordering note: we commit SQLite BEFORE the catalog PUT. If we
    # crashed between the commit and the PUT, sync_pending would stay
    # True (we set it preemptively) and the next online operation
    # would push. The reverse ordering (catalog first) creates an
    # unrecoverable window where the catalog has moved but SQLite
    # hasn't — a later crash would lose the catalog transition.
    store.update_execution(
        execution_rid,
        status=target,
        sync_pending=True,  # preemptively True; cleared after successful PUT
        **extra_fields,
    )

    # Compose the catalog PUT body from the SQLite row we just wrote.
    # Only the columns the catalog Execution row knows about go here —
    # Status and lifecycle timestamps — not SQLite-only fields like
    # sync_pending or config_json.
    body = _catalog_body_for_execution(
        store=store,
        execution_rid=execution_rid,
    )
    # Use the datapath API for the update. Raw catalog.put on ERMrest
    # requires specific URL forms that vary by server version; the
    # pathBuilder abstracts this and is what the rest of the codebase
    # uses for Execution.update.
    try:
        pb = catalog.getPathBuilder()
        pb.schemas["deriva-ml"].tables["Execution"].update(body)
    except Exception as exc:  # network blip, 5xx, etc.
        logger.warning(
            "execution %s: catalog sync FAILED (%s); SQLite committed, "
            "sync_pending stays True for later flush",
            execution_rid, exc,
        )
        return

    # PUT succeeded — clear sync_pending.
    store.update_execution(execution_rid, sync_pending=False)
    logger.debug(
        "online transition %s: %s → %s (synced)",
        execution_rid, current, target,
    )


def _catalog_body_for_execution(
    *,
    store: "ExecutionStateStore",
    execution_rid: str,
) -> list[dict]:
    """Build the ERMrest PUT body for an execution's catalog row.

    Reads the current SQLite state and projects to the catalog's
    column set. Kept as a helper so transition() stays focused on
    orchestration and so tests can assert on body contents.

    Args:
        store: The ExecutionStateStore to read from.
        execution_rid: Which execution's row to project.

    Returns:
        A list of one dict suitable as the ``json=`` body for a
        catalog PUT on ``/entity/deriva-ml:Execution``.

    Raises:
        DerivaMLStateInconsistency: If the SQLite row vanished
            between the preceding update and this read (concurrent
            delete — shouldn't happen in practice but we fail
            loudly rather than PUT a partial body).
    """
    row = store.get_execution(execution_rid)
    if row is None:
        # Caller just updated SQLite; this would only happen on a
        # concurrent delete. Surface clearly rather than putting a
        # partial body to the catalog.
        raise DerivaMLStateInconsistency(
            f"executions row {execution_rid} vanished between write and PUT"
        )
    # Catalog Execution schema has: Workflow, Description, Duration,
    # Status, Status_Detail (see src/deriva_ml/schema/create_schema.py).
    # Start/stop times are NOT catalog columns — they live in SQLite
    # only. Duration is computed elsewhere (in execution_stop) and
    # written directly; don't echo it here.
    return [{
        "RID": row["rid"],
        "Status": row["status"],
        # Status_Detail: prefer error if present, else description.
        "Status_Detail": row["error"] or row["description"],
    }]


def flush_pending_sync(
    *,
    store: "ExecutionStateStore",
    catalog: "ErmrestCatalog",
    execution_rid: str,
) -> None:
    """Push a single execution's SQLite state to the catalog.

    Called when we've opened online and notice this execution has
    sync_pending=True (accumulated from offline transitions, or from
    a previous online transition whose PUT failed).

    Idempotent: no-op if sync_pending is already False. If the PUT
    fails, sync_pending stays True for the next attempt.

    Args:
        store: ExecutionStateStore holding the row.
        catalog: Live ErmrestCatalog.
        execution_rid: Which execution to flush.

    Raises:
        DerivaMLStateInconsistency: If the execution row has vanished
            from the SQLite store (concurrent delete or missing row).

    Example:
        >>> # After resuming an execution online that last ran offline:
        >>> flush_pending_sync(store=store, catalog=ml.catalog,  # doctest: +SKIP
        ...                    execution_rid="EXE-A")
    """
    row = store.get_execution(execution_rid)
    if row is None:
        raise DerivaMLStateInconsistency(
            f"flush_pending_sync: execution {execution_rid} not in SQLite"
        )
    if not row["sync_pending"]:
        return

    body = _catalog_body_for_execution(store=store, execution_rid=execution_rid)
    # Use the datapath API (same fix pattern as transition()): raw
    # catalog.put on /entity/... is rejected by ERMrest with 409
    # "Entity PUT requires at least one client-managed key for input
    # correlation."
    try:
        pb = catalog.getPathBuilder()
        pb.schemas["deriva-ml"].tables["Execution"].update(body)
    except Exception as exc:
        logger.warning(
            "flush_pending_sync %s: catalog sync failed (%s); will retry later",
            execution_rid, exc,
        )
        return

    store.update_execution(execution_rid, sync_pending=False)
    logger.debug("flush_pending_sync %s: synced", execution_rid)


# Disagreement resolution table (spec §2.2 — six cases).
#
# Rows are keyed by (sqlite_status, catalog_status) tuples. Value is
# a literal action name:
#   'adopt'        — SQLite adopts the catalog's status
#   'push'         — SQLite state is newer; set sync_pending=True
#   (missing)      — outside the rule table; caller raises DerivaMLStateInconsistency
#
# Sync-pending handling is layered on top: if sqlite.sync_pending was
# True we generally 'push' regardless of catalog state.

_DISAGREEMENT_RULES: dict[tuple[ExecutionStatus, ExecutionStatus], str] = {
    # Externally aborted while we thought we were running.
    (ExecutionStatus.Running, ExecutionStatus.Aborted): "adopt",
    # Another process completed the upload.
    (ExecutionStatus.Pending_Upload, ExecutionStatus.Uploaded): "adopt",
    # External failure signal.
    (ExecutionStatus.Running, ExecutionStatus.Failed): "adopt",
    # We stopped cleanly; catalog still says running (our earlier PUT
    # never landed).
    (ExecutionStatus.Stopped, ExecutionStatus.Running): "push",
    # Same story at other cleanly-terminal SQLite states.
    (ExecutionStatus.Failed, ExecutionStatus.Running): "push",
    (ExecutionStatus.Uploaded, ExecutionStatus.Pending_Upload): "push",
    (ExecutionStatus.Uploaded, ExecutionStatus.Running): "push",
    (ExecutionStatus.Aborted, ExecutionStatus.Running): "push",
}


def reconcile_with_catalog(
    *,
    store: "ExecutionStateStore",
    catalog: "ErmrestCatalog",
    execution_rid: str,
) -> None:
    """Compare a single execution's SQLite state with the catalog and
    apply the disagreement rules (spec §2.2).

    Called on resume_execution when online, before returning the
    Execution to the user. Keeps startup fast by acting per-execution
    rather than workspace-wide.

    Behavior:
    - If SQLite and catalog agree: no-op.
    - If sqlite.sync_pending is True: respect it, leave alone (the
      caller's flush_pending_sync will handle it).
    - Otherwise look up (sqlite_status, catalog_status) in the
      disagreement table:
        * 'adopt' → SQLite adopts the catalog's status.
        * 'push' → mark sync_pending=True (caller will flush).
        * (missing) → raise DerivaMLStateInconsistency.
    - If the catalog GET fails transiently: log a warning and return.
    - If the catalog row is missing: raise DerivaMLStateInconsistency.

    Args:
        store: The ExecutionStateStore.
        catalog: Live ErmrestCatalog.
        execution_rid: Which execution to reconcile.

    Raises:
        DerivaMLStateInconsistency: Catalog row missing (orphan), or
            disagreement is outside the known rule table.

    Example:
        >>> # On resume_execution in online mode:
        >>> reconcile_with_catalog(  # doctest: +SKIP
        ...     store=ws.execution_state_store(),
        ...     catalog=ml.catalog,
        ...     execution_rid="EXE-A",
        ... )
    """
    sqlite_row = store.get_execution(execution_rid)
    if sqlite_row is None:
        raise DerivaMLStateInconsistency(
            f"reconcile: execution {execution_rid} not in SQLite"
        )
    sqlite_status = ExecutionStatus(sqlite_row["status"])

    try:
        # URL filter on RID — returns a list of 0 or 1 rows.
        response = catalog.get(
            f"/entity/deriva-ml:Execution/RID={execution_rid}"
        )
        rows = response.json()
    except Exception as exc:
        logger.warning(
            "reconcile %s: catalog GET failed (%s); leaving SQLite as-is",
            execution_rid, exc,
        )
        return

    if not rows:
        # Orphan: SQLite has the row, catalog doesn't. This is
        # usually a clone/copy gone wrong or a catalog-side delete.
        # Don't guess; ask the user to resolve.
        raise DerivaMLStateInconsistency(
            f"Execution {execution_rid} exists in SQLite (status={sqlite_status}) "
            f"but has no row in the catalog. Either the catalog was "
            f"re-initialized, or the workspace was copied from elsewhere. "
            f"To adopt SQLite state, manually insert the catalog row; "
            f"to discard, call ml.gc_executions(status='aborted')."
        )

    catalog_row = rows[0]
    # Catalog Status is a vocab term; its string value matches our enum.
    try:
        catalog_status = ExecutionStatus(catalog_row.get("Status", ""))
    except ValueError:
        # Catalog has a Status we don't recognize. Surface rather than guess.
        raise DerivaMLStateInconsistency(
            f"Execution {execution_rid}: catalog Status="
            f"{catalog_row.get('Status')!r} is not a recognized "
            f"ExecutionStatus value"
        )

    # Happy path: they agree.
    if sqlite_status == catalog_status:
        return

    # SQLite was waiting to push — this disagreement is expected.
    if sqlite_row["sync_pending"]:
        # We'll flush later; don't treat the catalog as authoritative.
        logger.debug(
            "reconcile %s: disagreement (SQLite=%s, catalog=%s) is "
            "expected because sync_pending=True; leaving for flush",
            execution_rid, sqlite_status, catalog_status,
        )
        return

    rule = _DISAGREEMENT_RULES.get((sqlite_status, catalog_status))
    if rule == "adopt":
        # Catalog is authoritative. Include any error/timing from the
        # catalog row so the user's Execution.error reflects reality.
        store.update_execution(
            execution_rid,
            status=catalog_status,
            error=catalog_row.get("Status_Detail"),
            sync_pending=False,
        )
        logger.info(
            "reconcile %s: adopted catalog state %s (was %s in SQLite)",
            execution_rid, catalog_status, sqlite_status,
        )
    elif rule == "push":
        # SQLite is newer; mark for flush. The resume flow will
        # invoke flush_pending_sync after reconcile.
        store.update_execution(execution_rid, sync_pending=True)
        logger.info(
            "reconcile %s: SQLite ahead (SQLite=%s, catalog=%s); "
            "marked sync_pending for flush",
            execution_rid, sqlite_status, catalog_status,
        )
    else:
        raise DerivaMLStateInconsistency(
            f"Execution {execution_rid}: unexpected state disagreement "
            f"(SQLite={sqlite_status}, catalog={catalog_status}) not "
            f"covered by reconciliation rules. Human intervention required."
        )


def create_catalog_execution(
    *,
    catalog: "ErmrestCatalog",
    workflow_rid: str | None,
    description: str | None,
) -> str:
    """POST a new row to the catalog's Execution table and return
    its server-assigned RID.

    This is the one place in the state machine that actually creates a
    new execution — all other transitions modify an existing row. It
    is callable only in online mode (the caller enforces).

    Args:
        catalog: Live ErmrestCatalog.
        workflow_rid: Workflow FK. May be None only if the catalog's
            Execution.Workflow column is nullable (Deriva-ML's
            schema requires it, but other catalogs may differ).
        description: Human-readable description. Passes through to the
            Execution.Description column.

    Returns:
        The RID assigned by the server.

    Raises:
        DerivaMLDataError: If the catalog's POST response lacks a RID.
        Exception: On HTTP failure (caller may want to retry).

    Example:
        >>> rid = create_catalog_execution(  # doctest: +SKIP
        ...     catalog=ml.catalog,
        ...     workflow_rid="WFL-1",
        ...     description="first training run",
        ... )
        >>> rid
        'EXE-NEW'
    """
    body = [{
        "Workflow": workflow_rid,
        "Description": description,
        "Status": str(ExecutionStatus.Created),
    }]
    response = catalog.post("/entity/deriva-ml:Execution", json=body)
    inserted = response.json()
    if not inserted or "RID" not in inserted[0]:
        raise DerivaMLDataError(
            "catalog POST to Execution returned no RID; unable to continue"
        )
    return inserted[0]["RID"]
