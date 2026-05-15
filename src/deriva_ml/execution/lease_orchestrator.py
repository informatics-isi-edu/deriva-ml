"""Vestigial crash-recovery stub for the retired pending-rows lease protocol.

The pending-rows / directory-rules architecture (per
``2026-04-18-sqlite-execution-state-design.md``) provided durable
RID leases for an upload-engine that was superseded by the
bag-commit path before its writer shipped. The audit at
``docs/design/deriva-ml-audit-2026-05-phase3-execution.md`` §1.5 /
§1.6 retired the entire surface in the Phase 3 cleanup.

:func:`reconcile_pending_leases` survives **only** so the two
production call sites in ``core/base.py`` and
``core/mixins/execution.py`` continue to compile without an
import-removal sweep. The body is a no-op: there are no
``leasing`` rows because there is no writer, so there is nothing
to reconcile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deriva_ml.core.logging_config import get_logger

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
    """No-op crash-recovery stub. Retained for call-site stability.

    The pending-rows lease protocol was retired in Phase 3 cleanup
    (audit §1.5 / §1.6). There is no writer of ``leasing`` rows in
    production, so this reconciliation has nothing to do. The
    function exists so the workspace-open and ``resume_execution``
    call sites continue to compile.

    Args:
        store: Ignored; kept for signature stability.
        catalog: Ignored; kept for signature stability.
        execution_rid: Ignored; kept for signature stability.
    """
    # Intentional no-op — see module docstring.
    return
