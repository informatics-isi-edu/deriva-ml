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


def topo_sort_work_items(
    items: list[_WorkItem],
    *,
    fk_edges: "dict[tuple[str, str], list[tuple[str, str]]]",
) -> list[_WorkItem]:
    """Topologically sort work items by FK dependencies.

    Args:
        items: Work items to sort.
        fk_edges: Adjacency map: (schema, table) → [(schema, parent_table), ...].
            An entry means "this table has FKs to these parents"; parents
            must drain first. Missing entries mean no FKs; equivalent
            to [].

    Returns:
        Items in drain order: all parents before all children.

    Raises:
        DerivaMLCycleError: If fk_edges contains a cycle.
    """
    from collections import deque

    from deriva_ml.core.exceptions import DerivaMLCycleError

    # Kahn's algorithm.
    by_key = {(i.target_schema, i.target_table): i for i in items}
    indeg: dict[tuple[str, str], int] = {k: 0 for k in by_key}
    # We only care about edges between tables that have work to do.
    filtered_edges: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for child, parents in fk_edges.items():
        if child not in by_key:
            continue
        real_parents = [p for p in parents if p in by_key]
        filtered_edges[child] = real_parents
        indeg[child] = len(real_parents)

    # Stable queue: keep input order for tables of equal in-degree.
    queue: deque = deque(k for k in by_key if indeg[k] == 0)
    output: list[_WorkItem] = []
    while queue:
        k = queue.popleft()
        output.append(by_key[k])
        for child, parents in filtered_edges.items():
            if k in parents:
                indeg[child] -= 1
                if indeg[child] == 0:
                    queue.append(child)

    if len(output) != len(by_key):
        seen = {(o.target_schema, o.target_table) for o in output}
        remaining = [k for k in by_key if k not in seen]
        raise DerivaMLCycleError(
            f"FK cycle detected in pending tables: {remaining}. "
            "Split into multiple executions, or write rows that break "
            "the cycle in a prior run."
        )
    return output
