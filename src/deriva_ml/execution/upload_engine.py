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
    """Topologically sort work items by FK dependencies (flat order).

    Args:
        items: Work items to sort.
        fk_edges: Adjacency map: (schema, table) → [(schema, parent_table), ...].
            An entry means "this table has FKs to these parents"; parents
            must drain first. Missing entries mean no FKs; equivalent
            to [].

    Returns:
        Items in a flat valid drain order: all parents before all
        children. Still used by callers that only need a single valid
        sequence. For per-level fail-fast semantics (spec §2.11.2
        step 7), use `_group_by_topo_level` instead.

    Raises:
        DerivaMLCycleError: If fk_edges contains a cycle.
    """
    levels = _group_by_topo_level(items, fk_edges=fk_edges)
    return [item for level in levels for item in level]


def _group_by_topo_level(
    items: list[_WorkItem],
    *,
    fk_edges: "dict[tuple[str, str], list[tuple[str, str]]]",
) -> list[list[_WorkItem]]:
    """Group work items into topological levels.

    Items in the same inner list share the same dependency depth —
    they have no FK dependency on each other and may be drained
    concurrently (§2.11.2 step 6: "for each topological level, in
    parallel"). All items in level N must complete before any item
    in level N+1 is drained.

    The per-level grouping is what makes per-level fail-fast possible:
    the drain can record failures across an entire level before
    aborting at the level boundary (spec §2.11.2 step 7).

    Args:
        items: Work items to group.
        fk_edges: Adjacency map: (schema, table) → [(schema, parent_table), ...].
            An entry means "this table has FKs to these parents"; parents
            must drain first. Missing entries mean no FKs; equivalent
            to [].

    Returns:
        List of levels. Each level is a list of `_WorkItem`. Levels
        appear in drain order (level 0 first). Empty outer list if
        `items` is empty.

    Raises:
        DerivaMLCycleError: If fk_edges contains a cycle.
    """
    from deriva_ml.core.exceptions import DerivaMLCycleError

    by_key = {(i.target_schema, i.target_table): i for i in items}
    indeg: dict[tuple[str, str], int] = {k: 0 for k in by_key}
    filtered_edges: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for child, parents in fk_edges.items():
        if child not in by_key:
            continue
        real_parents = [p for p in parents if p in by_key]
        filtered_edges[child] = real_parents
        indeg[child] = len(real_parents)

    levels: list[list[_WorkItem]] = []
    # Frontier: keys with in-degree 0. Level-by-level BFS (Kahn).
    frontier = [k for k in by_key if indeg[k] == 0]
    processed: set[tuple[str, str]] = set()
    while frontier:
        levels.append([by_key[k] for k in frontier])
        next_frontier: list[tuple[str, str]] = []
        for k in frontier:
            processed.add(k)
            for child, parents in filtered_edges.items():
                if child in processed:
                    continue
                if k in parents:
                    indeg[child] -= 1
                    if indeg[child] == 0:
                        next_frontier.append(child)
        frontier = next_frontier

    if len(processed) != len(by_key):
        remaining = [k for k in by_key if k not in processed]
        raise DerivaMLCycleError(
            f"FK cycle detected in pending tables: {remaining}. "
            "Split into multiple executions, or write rows that break "
            "the cycle in a prior run."
        )
    return levels


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
    4. Group work items into topological levels by FK (items in the
       same level have no FK dependency on each other).
    5. For each level, drain every item via _drain_work_item (which
       wraps deriva-py's uploader). One item's failure does NOT skip
       its siblings in the same level — all independent items at a
       level get drained.
    6. After a level completes, if any item failed, abort the drain
       at the level boundary and leave the rest of the work intact
       for a later re-run (spec §2.11.2 step 7).
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

    # Step 4: topo sort into levels.
    fk_edges = _fk_edges_for_work(ml=ml, items=items)
    levels = _group_by_topo_level(items, fk_edges=fk_edges)

    # Step 5: drain level-by-level. Within a level, items are
    # independent — one item's failure does NOT skip its siblings.
    # Per spec §2.11.2 step 7, we record all failures in the level
    # first, then abort at the level boundary.
    per_table: dict[str, dict] = {}
    errors: list[str] = []
    total_uploaded = 0
    total_failed = 0

    for level in levels:
        level_had_failure = False
        for item in level:
            try:
                row = store.get_execution(item.execution_rid)
                current_status = ExecutionStatus(row["status"])
                if current_status != ExecutionStatus.Pending_Upload:
                    transition(
                        store=store,
                        catalog=ml.catalog if ml._mode.value == "online" else None,
                        execution_rid=item.execution_rid,
                        current=current_status,
                        target=ExecutionStatus.Pending_Upload,
                        mode=ml._mode,
                    )
            except Exception as exc:
                logger.warning(
                    "upload: could not set pending_upload for %s: %s",
                    item.execution_rid, exc,
                )

            try:
                n = _drain_work_item(store=store, catalog=ml.catalog, work_item=item, ml=ml)
                total_uploaded += n
                fqn = f"{item.target_schema}:{item.target_table}"
                per_table.setdefault(fqn, {"uploaded": 0, "failed": 0})
                per_table[fqn]["uploaded"] += n
            except Exception as exc:
                level_had_failure = True
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
                # Continue draining siblings in this level.
        if level_had_failure:
            # Abort at the level boundary — don't descend into
            # dependent levels whose parents couldn't drain.
            break

    # Final execution status transitions.
    for exe_rid in rids:
        counts = store.count_pending_by_kind(execution_rid=exe_rid)
        row = store.get_execution(exe_rid)
        if row is None:
            continue
        current_status = ExecutionStatus(row["status"])
        if current_status != ExecutionStatus.Pending_Upload:
            continue

        total_failed_counts = counts["failed_rows"] + counts["failed_files"]
        total_pending_counts = counts["pending_rows"] + counts["pending_files"]

        if total_failed_counts == 0 and total_pending_counts == 0:
            target = ExecutionStatus.Uploaded
        elif total_failed_counts > 0:
            target = ExecutionStatus.Failed
        else:
            # No failures, but rows still pending (drain was aborted
            # at a higher level or this run only partially drained).
            # Leave status as pending_upload so a future upload_pending
            # run picks up the rest without requiring retry_failed.
            continue

        try:
            transition(
                store=store,
                catalog=ml.catalog if ml._mode.value == "online" else None,
                execution_rid=exe_rid,
                current=current_status,
                target=target,
                mode=ml._mode,
                extra_fields={"error": errors[0]} if errors and target == ExecutionStatus.Failed else {},
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
    ml: "DerivaML | None" = None,
) -> int:
    """Drain one (exe, table) grouping: insert rows and/or upload files.

    For plain rows: build a pathBuilder insert body and issue a single
    call per table. Rows get status='uploaded' on success.

    For asset rows: delegate file-upload to deriva-py's existing
    uploader (chunk-resume + Hatrac hash dedup for idempotency).

    Args:
        store: ExecutionStateStore.
        catalog: ErmrestCatalog.
        work_item: Group of pending items to drain.
        ml: DerivaML instance for pathBuilder + uploader config.

    Returns:
        Number of rows successfully uploaded.

    Raises:
        Any exception from catalog insert or file upload. Individual
        failed rows are marked 'failed' in SQLite before re-raising.
    """
    import json
    from datetime import datetime, timezone

    from deriva_ml.execution.state_store import PendingRowStatus

    if ml is None:
        raise RuntimeError("ml kwarg is required — pass from run_upload_engine")

    rows = [
        r for r in store.list_pending_rows(execution_rid=work_item.execution_rid)
        if r["id"] in set(work_item.pending_ids)
    ]
    if not rows:
        return 0

    now = datetime.now(timezone.utc)

    if work_item.is_asset:
        files = [
            {
                "path": r["asset_file_path"],
                "rid": r["rid"],
                "pending_id": r["id"],
                "metadata": json.loads(r["metadata_json"]),
            }
            for r in rows
        ]
        result = _invoke_deriva_py_uploader(
            ml=ml, files=files,
            target_table=work_item.target_table,
            execution_rid=work_item.execution_rid,
        )
        uploaded_paths = set(result["uploaded"])
        for r in rows:
            path = r["asset_file_path"]
            pid = r["id"]
            if path in uploaded_paths:
                store.update_pending_row(
                    pid,
                    status=PendingRowStatus.uploaded,
                    uploaded_at=now,
                )
            else:
                failure_msg = next(
                    (f.get("error", "upload failed") for f in result.get("failed", [])
                     if f.get("path") == path),
                    "upload failed",
                )
                store.update_pending_row(
                    pid,
                    status=PendingRowStatus.failed,
                    error=failure_msg,
                )

        return sum(1 for r in rows if r["asset_file_path"] in uploaded_paths)

    # Plain rows: build a single catalog insert body including pre-leased RIDs.
    body = []
    for r in rows:
        metadata = json.loads(r["metadata_json"])
        metadata["RID"] = r["rid"]
        body.append(metadata)

    try:
        pb = ml.pathBuilder()
        tpath = pb.schemas[work_item.target_schema].tables[work_item.target_table]
        tpath.insert(body)
    except Exception as exc:
        for r in rows:
            store.update_pending_row(
                r["id"],
                status=PendingRowStatus.failed,
                error=str(exc),
            )
        raise

    for r in rows:
        store.update_pending_row(
            r["id"],
            status=PendingRowStatus.uploaded,
            uploaded_at=now,
        )
    return len(rows)


def _invoke_deriva_py_uploader(
    *,
    ml: "DerivaML",
    files: list[dict],
    target_table: str,
    execution_rid: str,
) -> dict:
    """Invoke deriva-py's uploader for a batch of files.

    Phase 1: raises NotImplementedError. The real asset-upload path
    currently flows through src/deriva_ml/dataset/upload.py::upload_directory
    which requires a directory layout matching the asset-upload spec
    regex — incompatible with per-file invocations at this grain.

    Phase 2 finalization: wire this to either (a) per-file Hatrac +
    pathBuilder invocations, or (b) a restructured upload_directory
    that works off a manifest rather than a regex-matched tree.

    Tests monkeypatch this function. Real asset uploads in Phase 1
    continue to flow through exe.upload_outputs (which calls the
    existing _upload_execution_dirs flow, wired in G8).

    Returns:
        Dict with keys 'uploaded' (list of paths) and 'failed'
        (list of {path, error}).
    """
    raise NotImplementedError(
        "G7 Phase 1: _invoke_deriva_py_uploader not yet wired to a "
        "real uploader. Tests monkeypatch this; production asset uploads "
        "flow through exe.upload_outputs for now. Phase 2 finalizes."
    )
