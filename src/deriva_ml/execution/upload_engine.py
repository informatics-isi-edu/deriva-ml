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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deriva_ml.execution.lease_orchestrator import acquire_leases_for_execution
from deriva_ml.execution.state_machine import transition
from deriva_ml.execution.state_store import ExecutionStatus, PendingRowStatus

if TYPE_CHECKING:
    from deriva.core.ermrest_catalog import ErmrestCatalog

    from deriva_ml.core.base import DerivaML
    from deriva_ml.execution.state_store import ExecutionStateStore

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


@dataclass
class UploadReport:
    """Result of a run_upload_engine call.

    Attributes:
        execution_rids: Executions attempted.
        total_uploaded: Count of rows+files in status='uploaded' after
            the run.
        total_failed: Count of rows+files in status='failed' after the
            run.
        per_table: Map of "schema:table" → dict {uploaded, failed}.
        errors: List of human-readable error lines from failed items.
    """
    execution_rids: list[str]
    total_uploaded: int
    total_failed: int
    per_table: dict[str, dict]
    errors: list[str] = field(default_factory=list)


def run_upload_engine(
    *,
    ml: "DerivaML",
    execution_rids: "list[str] | None",
    retry_failed: bool = False,
    bandwidth_limit_mbps: "int | None" = None,
    parallel_files: int = 4,
) -> UploadReport:
    """Drain pending rows/files for the given executions.

    Phase 1 implementation per spec §2.11.2 (omitting provisional
    step 6). Steps:

    1. Enumerate work items (exe, table) with pending/failed-if-retry.
    2. Re-validate metadata against catalog schema (Phase 2; skipped
       here — dependency on the provisional TableHandle surface).
    3. Acquire RID leases for any status='staged' rows.
    4. Topologically sort work items by FK.
    5. For each level in topo order, drain each item via
       _drain_work_item (which wraps deriva-py's uploader).
    6. On the first level with failures, abort the drain but leave
       the rest of the work intact for a later re-run.
    7. Return an UploadReport.

    Args:
        ml: DerivaML instance.
        execution_rids: Which executions to drain; None = all pending.
        retry_failed: Include status='failed' rows.
        bandwidth_limit_mbps: Cap uploader egress. None = unlimited.
            Passed to deriva-py's uploader config.
        parallel_files: Concurrent file uploads per table. Bounded.

    Returns:
        UploadReport summarizing the run.
    """
    store = ml.workspace.execution_state_store()

    rids = execution_rids or [row["rid"] for row in store.list_executions()]

    # Step 1: enumerate.
    items = _enumerate_work(ml=ml, execution_rids=rids, retry_failed=retry_failed)
    if not items:
        return UploadReport(
            execution_rids=rids, total_uploaded=0, total_failed=0,
            per_table={},
        )

    # Step 3: lease any staged rows first (per-execution).
    by_exe: dict[str, list[int]] = {}
    for item in items:
        staged = [
            r["id"]
            for r in store.list_pending_rows(
                execution_rid=item.execution_rid,
                status=PendingRowStatus.staged,
            )
            if r["id"] in item.pending_ids
        ]
        if staged:
            by_exe.setdefault(item.execution_rid, []).extend(staged)
    for exe_rid, pending_ids in by_exe.items():
        acquire_leases_for_execution(
            store=store, catalog=ml.catalog,
            execution_rid=exe_rid, pending_ids=pending_ids,
        )

    # Step 4: topo sort.
    fk_edges = _fk_edges_for_work(ml=ml, items=items)
    sorted_items = topo_sort_work_items(items, fk_edges=fk_edges)

    # Step 5: drain, aborting after the first level with failures.
    per_table: dict[str, dict] = {}
    errors: list[str] = []
    total_uploaded = 0
    total_failed = 0

    for item in sorted_items:
        try:
            row = store.get_execution(item.execution_rid)
            current_status = ExecutionStatus(row["status"])
            if current_status != ExecutionStatus.pending_upload:
                transition(
                    store=store,
                    catalog=ml.catalog if ml._mode.value == "online" else None,
                    execution_rid=item.execution_rid,
                    current=current_status,
                    target=ExecutionStatus.pending_upload,
                    mode=ml._mode,
                )
        except Exception as exc:
            logger.warning(
                "upload: could not set pending_upload for %s: %s",
                item.execution_rid, exc,
            )

        try:
            n = _drain_work_item(store=store, catalog=ml.catalog, work_item=item)
            total_uploaded += n
            fqn = f"{item.target_schema}:{item.target_table}"
            per_table.setdefault(fqn, {"uploaded": 0, "failed": 0})
            per_table[fqn]["uploaded"] += n
        except Exception as exc:
            errors.append(f"{item.target_table}: {exc}")
            failed_rows = store.list_pending_rows(
                execution_rid=item.execution_rid,
                status=PendingRowStatus.failed,
                target_table=item.target_table,
            )
            total_failed += len(failed_rows)
            fqn = f"{item.target_schema}:{item.target_table}"
            per_table.setdefault(fqn, {"uploaded": 0, "failed": 0})
            per_table[fqn]["failed"] += len(failed_rows)
            break

    # Final execution status transitions.
    for exe_rid in rids:
        counts = store.count_pending_by_kind(execution_rid=exe_rid)
        row = store.get_execution(exe_rid)
        if row is None:
            continue
        current_status = ExecutionStatus(row["status"])
        if current_status == ExecutionStatus.pending_upload:
            if counts["pending_rows"] == 0 and counts["pending_files"] == 0:
                if counts["failed_rows"] == 0 and counts["failed_files"] == 0:
                    target = ExecutionStatus.uploaded
                else:
                    target = ExecutionStatus.failed
            else:
                target = ExecutionStatus.failed
            try:
                transition(
                    store=store,
                    catalog=ml.catalog if ml._mode.value == "online" else None,
                    execution_rid=exe_rid,
                    current=current_status,
                    target=target,
                    mode=ml._mode,
                    extra_fields={"error": errors[0]} if errors and target == ExecutionStatus.failed else {},
                )
            except Exception as exc:
                logger.warning(
                    "upload: final status transition failed for %s: %s", exe_rid, exc,
                )

    return UploadReport(
        execution_rids=rids,
        total_uploaded=total_uploaded,
        total_failed=total_failed,
        per_table=per_table,
        errors=errors,
    )


def _fk_edges_for_work(
    *,
    ml: "DerivaML",
    items: list[_WorkItem],
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Extract FK edges for the tables involved in `items`.

    Consults ml.model (the ERMrest model) for each table's outgoing
    foreign keys; an edge is added for each FK whose target is also
    among the items. Edges to tables without pending work are pruned
    (irrelevant to this drain).

    Args:
        ml: DerivaML instance providing ml.model.
        items: Work items — only their target tables matter.

    Returns:
        Adjacency dict: (schema, table) → [(schema, parent_table), ...].
    """
    edges: dict[tuple[str, str], list[tuple[str, str]]] = {}
    table_keys = {(i.target_schema, i.target_table) for i in items}
    for item in items:
        try:
            schema = ml.model.schemas[item.target_schema]
            table = schema.tables[item.target_table]
        except (KeyError, AttributeError):
            # Schema/table not in the model (test fixtures sometimes
            # register pending rows for tables that don't exist in the
            # deployed catalog). Treat as no outgoing FKs.
            edges[(item.target_schema, item.target_table)] = []
            continue
        parents = []
        for fk in getattr(table, "foreign_keys", []):
            pk_table = fk.pk_table
            parent = (pk_table.schema.name, pk_table.name)
            if parent in table_keys and parent != (item.target_schema, item.target_table):
                parents.append(parent)
        edges[(item.target_schema, item.target_table)] = parents
    return edges


def _drain_work_item(
    *,
    store: "ExecutionStateStore",
    catalog: "ErmrestCatalog",
    work_item: _WorkItem,
) -> int:
    """Phase-1 stub — the concrete deriva-py-uploader invocation lives
    in Task G7. Tests monkeypatch this in G6; G7 replaces the body.

    Returns the number of rows uploaded.
    """
    raise NotImplementedError("G7 implements the deriva-py invocation")
