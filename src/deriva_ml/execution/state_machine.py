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
from datetime import datetime, timezone  # noqa: F401  (used in C2-C6)
from typing import TYPE_CHECKING

from deriva_ml.core.exceptions import (
    DerivaMLDataError,  # noqa: F401  (used in C2-C6)
    DerivaMLException,
    DerivaMLStateInconsistency,
)
from deriva_ml.execution.state_store import ExecutionStatus

if TYPE_CHECKING:
    from deriva.core import ErmrestCatalog  # noqa: F401  (used in C2-C6)

    from deriva_ml.core.connection_mode import ConnectionMode  # noqa: F401  (used in C2-C6)
    from deriva_ml.execution.state_store import ExecutionStateStore  # noqa: F401  (used in C2-C6)

logger = logging.getLogger(__name__)


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
    (ExecutionStatus.created, ExecutionStatus.running),
    (ExecutionStatus.running, ExecutionStatus.stopped),
    (ExecutionStatus.stopped, ExecutionStatus.pending_upload),
    (ExecutionStatus.pending_upload, ExecutionStatus.uploaded),

    # Failure paths
    (ExecutionStatus.running, ExecutionStatus.failed),
    (ExecutionStatus.pending_upload, ExecutionStatus.failed),

    # Retry from upload failure back into upload
    (ExecutionStatus.failed, ExecutionStatus.pending_upload),

    # Abort is legal from any pre-terminal state. 'uploaded' is
    # terminal — we don't allow abort after successful upload.
    (ExecutionStatus.created, ExecutionStatus.aborted),
    (ExecutionStatus.running, ExecutionStatus.aborted),
    (ExecutionStatus.stopped, ExecutionStatus.aborted),
    (ExecutionStatus.failed, ExecutionStatus.aborted),
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
        ...     current=ExecutionStatus.running,
        ...     target=ExecutionStatus.stopped,
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
        >>> transition(
        ...     store=store, catalog=ml.catalog,
        ...     execution_rid="EXE-A",
        ...     current=ExecutionStatus.running,
        ...     target=ExecutionStatus.stopped,
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
    try:
        catalog.put(
            "/entity/deriva-ml:Execution",
            json=body,
        )
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
    # Catalog Execution schema: RID (PK), Status, Start_Time, End_Time,
    # Duration, ... — see src/deriva_ml/schema/create_schema.py for
    # the canonical column list. We update only the columns we own
    # here; the catalog is responsible for RCB/RMT etc.
    return [{
        "RID": row["rid"],
        "Status": row["status"],
        "Start_Time": row["start_time"],
        "End_Time": row["stop_time"],
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
        >>> flush_pending_sync(store=store, catalog=ml.catalog,
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
    try:
        catalog.put("/entity/deriva-ml:Execution", json=body)
    except Exception as exc:
        logger.warning(
            "flush_pending_sync %s: catalog PUT failed (%s); will retry later",
            execution_rid, exc,
        )
        return

    store.update_execution(execution_rid, sync_pending=False)
    logger.debug("flush_pending_sync %s: synced", execution_rid)
