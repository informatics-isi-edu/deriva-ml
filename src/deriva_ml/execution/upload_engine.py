"""Upload engine — drives SQLite-queued pending rows through deriva-py's
uploader.

Per spec §2.11.2. Phase 1 implementation: generic drain loop that
leases RIDs, topologically sorts by FK, and hands each group to
deriva-py's existing upload machinery. Provisional step 6
(feature-aware pre-insert validation) is NOT included in Phase 1 —
see §2.13.

Key idempotency properties (spec §1.1 item 4):
- Server-side Hatrac hash-dedup rejects duplicate uploads cheaply.
- deriva-py's uploader writes per-chunk resume state to
  .deriva-upload-state-*.json with fsync; a killed mid-chunk upload
  resumes from the next chunk on re-run.
- SQLite status tracking means a re-run of upload_pending skips
  already-uploaded rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from deriva_ml.execution.state_store import PendingRowStatus

if TYPE_CHECKING:
    from deriva_ml.core.base import DerivaML

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _WorkItem:
    """One (execution, target_table) grouping of pending items."""
    execution_rid: str
    target_schema: str
    target_table: str
    pending_ids: list[int]
    is_asset: bool


def _enumerate_work(
    *,
    ml: "DerivaML",
    execution_rids: "list[str] | None",
    retry_failed: bool,
) -> list[_WorkItem]:
    """Collect pending items grouped by (execution_rid, target_table).

    Args:
        ml: DerivaML instance for workspace access.
        execution_rids: If None, scan every execution in the registry.
            Otherwise scope to the listed RIDs.
        retry_failed: If True, include rows in status='failed' in the
            work set. If False (default), only include non-terminal
            rows.

    Returns:
        List of _WorkItem — one per (execution, table) grouping.
        Empty list if nothing to do.
    """
    store = ml.workspace.execution_state_store()

    if execution_rids is None:
        candidate_rids = [row["rid"] for row in store.list_executions()]
    else:
        candidate_rids = list(execution_rids)

    statuses_to_take = [
        PendingRowStatus.staged,
        PendingRowStatus.leasing,
        PendingRowStatus.leased,
        PendingRowStatus.uploading,
    ]
    if retry_failed:
        statuses_to_take.append(PendingRowStatus.failed)

    items: list[_WorkItem] = []
    for rid in candidate_rids:
        rows = store.list_pending_rows(
            execution_rid=rid, status=statuses_to_take,
        )
        if not rows:
            continue
        by_key: dict[tuple[str, str], list[dict]] = {}
        for r in rows:
            key = (r["target_schema"], r["target_table"])
            by_key.setdefault(key, []).append(r)

        for (schema, table), group in by_key.items():
            items.append(_WorkItem(
                execution_rid=rid,
                target_schema=schema, target_table=table,
                pending_ids=[r["id"] for r in group],
                is_asset=any(r["asset_file_path"] is not None for r in group),
            ))
    return items
