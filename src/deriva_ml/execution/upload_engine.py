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

import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from deriva.core import DEFAULT_SESSION_CONFIG
from deriva.transfer.upload.deriva_upload import GenericUploader, UploadState

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.dataset.upload import DEFAULT_UPLOAD_TIMEOUT, bulk_upload_configuration
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
    cancel_event: "threading.Event | None" = None,
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
        cancel_event: If provided and .is_set(), the engine stops
            dispatching new batches before each batch and signals any
            in-flight GenericUploader via its cancel() primitive. If
            None, the engine runs to completion.

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
            if cancel_event is not None and cancel_event.is_set():
                logger.info("upload: cancel_event set — stopping drain before next batch")
                break
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
                n = _drain_work_item(
                    store=store, catalog=ml.catalog,
                    work_item=item, ml=ml,
                    cancel_event=cancel_event,
                )
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
        if level_had_failure or (cancel_event is not None and cancel_event.is_set()):
            # Abort at the level boundary — don't descend into
            # dependent levels whose parents couldn't drain, and honor
            # cancel requests at level boundaries.
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
    cancel_event: "threading.Event | None" = None,
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
            cancel_event=cancel_event,
        )
        # NOTE: _invoke_deriva_py_uploader has already written per-row
        # SQLite status via its callbacks. We only need the aggregate
        # count here. Rows whose status is still Pending at this point
        # fall through and will be retried on the next run.
        uploaded_paths = set(result["uploaded"])
        return sum(
            1 for r in rows
            if r["asset_file_path"] in uploaded_paths
        )

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
    cancel_event: "threading.Event | None" = None,
) -> dict:
    """Invoke deriva-py's uploader for a batch of files.

    Builds a per-batch symlink-farm scan root whose layout matches
    ``asset_table_upload_spec``'s regex, constructs a fresh
    ``GenericUploader``, and drives ``scanDirectory + uploadFiles``.

    Two callbacks are wired:

    - ``status_callback()``: fires at each file boundary with no args.
      Walks ``uploader.file_status``, writes newly-terminal rows to the
      SQLite store, observes ``cancel_event`` and calls
      ``uploader.cancel()``.
    - ``file_callback(**kw)``: fires during in-flight uploads with
      byte progress. Observes ``cancel_event``; returns ``-1`` to
      signal hatrac to abort the current transfer when cancelled.

    After ``uploadFiles`` returns, a reconciliation pass over
    ``uploader.file_status`` catches any file the callback missed and
    writes its terminal state.

    Args:
        ml: DerivaML instance (for model, host, catalog_id, workspace).
        files: List of dicts with keys 'path', 'rid', 'pending_id',
            'metadata'. All entries share target_table and execution_rid.
        target_table: Name of the target asset table (no schema).
        execution_rid: Execution RID these files belong to.
        cancel_event: Optional cancellation signal.

    Returns:
        ``{"uploaded": list[str], "failed": list[dict]}`` where
        ``uploaded`` lists absolute input paths that succeeded and
        ``failed`` lists ``{"path": str, "error": str}`` for failures.
    """
    if not files:
        return {"uploaded": [], "failed": []}

    from datetime import datetime, timezone

    store = ml.workspace.execution_state_store()

    # Resolve schema for the target table (scan root layout needs it).
    try:
        table_obj = ml.model.name_to_table(target_table)
        schema_name = table_obj.schema.name
        metadata_cols = sorted(ml.model.asset_metadata(target_table))
    except Exception as exc:
        raise DerivaMLException(
            f"Unable to resolve asset table {target_table!r}: {exc}"
        ) from exc

    # Map absolute input path → the file dict (for callback writes).
    rows_by_path: dict[str, dict] = {str(Path(f["path"]).resolve()): f for f in files}
    written_paths: set[str] = set()

    with TemporaryDirectory(prefix="deriva-ml-upload-") as scan_root_str:
        # Resolve symlinks in the temp dir path so the scan-path ↔
        # input-path map lines up with what GenericUploader records
        # (which uses canonical absolute paths after Path.rglob/resolve).
        scan_root = Path(scan_root_str).resolve()

        # Build the symlink farm:
        # <scan_root>/<schema>/<table>/<md1>/.../<filename>
        for f in files:
            src = Path(f["path"]).resolve()
            metadata = f.get("metadata") or {}
            target_dir = scan_root / schema_name / target_table
            for col in metadata_cols:
                target_dir = target_dir / str(metadata.get(col, "None"))
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / src.name
            if target.exists():
                continue
            try:
                os.link(src, target)  # hardlink where possible
            except (OSError, NotImplementedError):
                try:
                    target.symlink_to(src)
                except OSError:
                    shutil.copy2(src, target)

        # Build config file (same shape upload_directory uses).
        spec_file = scan_root / "config.json"
        spec_file.write_text(json.dumps(bulk_upload_configuration(ml.model)))

        session_config = DEFAULT_SESSION_CONFIG.copy()
        session_config["timeout"] = DEFAULT_UPLOAD_TIMEOUT

        uploader = GenericUploader(
            server={
                "host": ml.model.hostname,
                "protocol": "https",
                "catalog_id": ml.model.catalog.catalog_id,
                "session": session_config,
            },
            config_file=spec_file,
            dcctx_cid="deriva-ml/upload_engine",
        )

        # Map scan-root path (what uploader.file_status uses as keys)
        # back to original input path for SQLite attribution.
        scan_path_to_input: dict[str, str] = {}
        for abs_input, row_dict in rows_by_path.items():
            src = Path(abs_input)
            metadata = row_dict.get("metadata") or {}
            target_dir = scan_root / schema_name / target_table
            for col in metadata_cols:
                target_dir = target_dir / str(metadata.get(col, "None"))
            scan_path_to_input[str(target_dir / src.name)] = abs_input

        def _apply_state_to_sqlite(scan_path: str, state_info: dict) -> None:
            """Idempotent: translate one uploader status dict to a SQLite write."""
            if scan_path in written_paths:
                return
            input_path = scan_path_to_input.get(scan_path)
            if input_path is None:
                return
            row = rows_by_path.get(input_path)
            if row is None:
                return
            state = state_info.get("State")
            status = state_info.get("Status", "")
            now = datetime.now(timezone.utc)
            # Map deriva-py UploadState codes → SQLite status.
            if state == UploadState.Success:
                store.update_pending_row(
                    row["pending_id"],
                    status=PendingRowStatus.uploaded,
                    uploaded_at=now,
                )
                written_paths.add(scan_path)
            elif state == UploadState.Failed:
                store.update_pending_row(
                    row["pending_id"],
                    status=PendingRowStatus.failed,
                    error=status or "upload failed",
                )
                written_paths.add(scan_path)
            # UploadState.Cancelled/Aborted/Paused/Timeout: leave Pending.

        def status_callback() -> None:
            # Observe cancel first.
            if cancel_event is not None and cancel_event.is_set():
                uploader.cancel()
            # Walk file_status, write any newly-terminal rows.
            for scan_path, info in list(uploader.file_status.items()):
                _apply_state_to_sqlite(scan_path, info)

        def file_callback(**kwargs) -> bool | int:
            if cancel_event is not None and cancel_event.is_set():
                uploader.cancel()
                return -1  # hatrac abort signal
            return True

        try:
            uploader.initialize(cleanup=False)
            uploader.getUpdatedConfig()
            uploader.scanDirectory(scan_root, abort_on_invalid_input=True)
            uploader.uploadFiles(
                status_callback=status_callback,
                file_callback=file_callback,
            )

            # Reconciliation pass — catch anything the callback missed.
            for scan_path, info in uploader.file_status.items():
                _apply_state_to_sqlite(scan_path, info)
        finally:
            try:
                uploader.cleanup()
            except Exception:
                pass

        # Build return dict by walking final file_status one more time.
        uploaded: list[str] = []
        failed: list[dict] = []
        for scan_path, info in uploader.file_status.items():
            input_path = scan_path_to_input.get(scan_path)
            if input_path is None:
                continue
            state = info.get("State")
            if state == UploadState.Success:
                uploaded.append(input_path)
            elif state == UploadState.Failed:
                failed.append({
                    "path": input_path,
                    "error": info.get("Status") or "upload failed",
                })
        return {"uploaded": uploaded, "failed": failed}
