# Changelog

All notable changes to this project are documented here.

## Unreleased — Phase 1: SQLite-backed execution state

### Breaking changes

Per spec §1.3 and R5.1 (aggressive deprecation), the following APIs
have been removed without a shim. Update call sites before upgrading.

| Removed | Replacement |
|---|---|
| `ml.restore_execution(rid)` | `ml.resume_execution(rid)` |
| `exe.upload_execution_outputs(...)` | `exe.upload_outputs(...)` |
| `exe.retry_failed()` | `exe.upload_outputs(retry_failed=True)` |
| `Execution.list_nested_executions()` | `ExecutionRecord.list_execution_children(recurse=...)` |
| `ExecutionRecord.list_nested_executions(recurse=...)` | `ExecutionRecord.list_execution_children(recurse=...)` |
| `ExecutionRecord.list_parent_executions(recurse=...)` | `ExecutionRecord.list_execution_parents(recurse=...)` |
| `Execution.datasets` as `list[DatasetBag]` | `Execution.datasets` as `DatasetCollection` (RID-keyed mapping + iterable) |

Additional behavior changes that may affect callers:

- `Execution.__exit__` now **propagates** exceptions (`return False`).
  Prior versions returned `True` and silently swallowed errors raised
  inside the `with` block.
- `Execution.status` is now an `ExecutionStatus` enum (lowercase string
  values: `created`, `running`, `stopped`, `failed`, `pending_upload`,
  `uploaded`, `aborted`) rather than the legacy title-case `Status`
  enum. Code that compared `exe.status == "Completed"` must be updated.

### New — execution state

- `ConnectionMode` enum (`online`, `offline`) with string coercion.
- `DerivaML(..., mode=ConnectionMode.online | offline)` — default online.
- `DerivaMLOfflineError` raised when a create-execution is attempted
  in offline mode.
- `DerivaMLNoExecutionContext` raised when writes are attempted on an
  `ml.table()` handle (reserved for Phase 2 TableHandle surface).
- `DerivaMLStateInconsistency` raised when workspace SQLite and
  catalog disagree in ways outside the six reconciliation rules
  (spec §2.2).
- Three new SQLite tables in the workspace engine:
  `execution_state__executions`, `execution_state__pending_rows`,
  `execution_state__directory_rules`. See
  `ExecutionStateStore` in `deriva_ml.execution.state_store`.
- Execution lifecycle module: `deriva_ml.execution.state_machine`
  with `transition()`, `flush_pending_sync()`,
  `reconcile_with_catalog()`, `create_catalog_execution()`.
- RID leasing against `public:ERMrest_RID_Lease` via
  `deriva_ml.execution.lease_orchestrator`
  (`acquire_leases_for_execution`, `reconcile_pending_leases`).
- Workspace-wide lease reconciliation on `DerivaML.__init__` (online
  only); per-execution reconciliation on `resume_execution`.

### New — public APIs

- `ml.list_executions(status=, workflow_rid=, mode=, since=)` —
  SQLite-only registry query returning `ExecutionRecord` dataclasses.
- `ml.find_incomplete_executions()` — sugar for the non-terminal
  status set.
- `ml.resume_execution(execution_rid)` — re-hydrate from registry;
  runs just-in-time reconciliation when online.
- `ml.gc_executions(status=, older_than=, delete_working_dir=)` —
  cleanup registry state (and optionally on-disk files).
- `ml.pending_summary()` — workspace-wide `WorkspacePendingSummary`.
- `ml.upload_pending(execution_rids=, retry_failed=, ...)` — blocking
  upload for selected executions; None = drain all pending.
- `ml.start_upload(...)` — non-blocking; returns `UploadJob`.
- `exe.upload_outputs(...)`, `ExecutionRecord.upload_outputs(ml=, ...)`
  — sugar for `ml.upload_pending(execution_rids=[self.rid], ...)`.
- `exe.pending_summary()`, `ExecutionRecord.pending_summary(ml=)` —
  per-execution `PendingSummary`.
- `exe.abort()` — transition an execution to `aborted`.
- `create_execution` accepts both a pre-built `ExecutionConfiguration`
  and kwargs (`datasets=[...]`, `workflow=...`, `description=...`);
  mixing raises `TypeError`. Dataset shorthand `"RID@version"` is
  coerced via `DatasetSpec.from_shorthand`.
- `UploadReport` (from `deriva_ml.execution.upload_engine`) —
  returned by all upload paths; exposes `total_uploaded`,
  `total_failed`, `per_table`, `errors`.
- `UploadJob` / `UploadProgress` (from
  `deriva_ml.execution.upload_job`) — thread-backed non-blocking
  handle with `wait(timeout=)`, `cancel()`, `progress()`,
  `pause()`/`resume()` stubs.

### Behavior changes

- `Execution.status`, `.start_time`, `.stop_time`, `.error` are now
  read-through SQLite properties — no in-memory caching. Mutations
  from other processes are visible on the next read.
- `Execution.__enter__`/`__exit__`/`abort` route through the state
  machine (validated transitions, SQLite write, catalog sync).
- Context-manager exit emits an INFO log with pending counts if any
  rows are staged. Upload does NOT auto-run on exit (per R6.3).
- Workspace open triggers a best-effort lease reconciliation
  (`reconcile_pending_leases`) to recover from mid-flight crashes.

### New CLI

- `deriva-ml-upload --host --catalog [--execution RID...]
  [--retry-failed] [--bandwidth-mbps N] [--parallel N]
  [--working-dir DIR] [--mode online|offline]` — operator-driven
  upload wrapping `ml.upload_pending`. Exit codes: 0 clean,
  1 partial failures, 2 fatal.

### Phase 2 deferred

Per spec §2.13, these sections remain provisional pending the
feature-consistency follow-on:

- Full `TableHandle` / `AssetTableHandle` surface
  (`handle.insert(...)`, `asset_file(...)`, `asset_directory(...)`,
  `handle.record_class()`, etc.). Phase 1 engine tests stage rows
  directly via `ExecutionStateStore.insert_pending_row`.
- `_drain_work_item` feature-aware pre-insert validation
  (§2.11.2 step 6).
- `Metric` / `Param` as first-class concepts parallel to features
  (spec §8).
- `_invoke_deriva_py_uploader` real body — Phase 1 delegates to a
  `NotImplementedError`-raising seam that tests monkeypatch; real
  asset uploads continue through the existing `exe.upload_outputs`
  path until the per-file uploader is finalized.
- `UploadJob` cancellation does not interrupt in-flight work items
  (Python threading limitation + deriva-py uploader lacks mid-chunk
  cancellation). `_cancel_event` is plumbed but not consumed by the
  engine yet.
- `parallel_files` and `bandwidth_limit_mbps` are accepted by
  `run_upload_engine` but not yet threaded through the uploader.
- `get_upload_job(id)` / `list_upload_jobs()` — require a persisted
  `upload_jobs` SQLite table, not in Phase 1 scope.
- `ConnectionMode.offline` currently does not fully skip network I/O
  in `DerivaML.__init__` (credential/catalog-model fetch still runs).
  Per spec §2.1, offline should be a distinct code path; Phase 2
  gates these calls on `_mode`.
- Legacy `Status` (title-case) vs `ExecutionStatus` (lowercase)
  reconciliation: `Execution._initialize_execution` and
  `execution_start` / `execution_stop` write legacy title-case values
  that `reconcile_with_catalog` does not map. Integration tests work
  around this by reading SQLite directly. Phase 2 unifies on
  `ExecutionStatus` (or adds a bidirectional mapping).
- Centralize UTC-naive→aware coercion in
  `ExecutionStateStore.list_executions` (currently handled ad hoc
  in `gc_executions`).
- ISO-format datetime serialization in
  `state_machine._catalog_body_for_execution` (currently passes
  raw datetimes which `catalog.put(json=)` cannot serialize; silently
  caught by soft-fail).

### Minor follow-ups

- Extract shared `_make_workflow` test helper to a conftest fixture
  (currently duplicated across test_pending_summary.py,
  test_upload_engine.py, test_upload_public_api.py).
- Prefer `ml._mode is ConnectionMode.online` over
  `ml._mode.value == "online"` for enum-safety.
- Add `_state_lock` around `UploadJob.status` transitions; `cancel()`
  races with the worker thread's final assignment.
- Update `Execution.upload_outputs` docstring — example uses the
  non-existent `with exe.execute():` pattern.
- Add tests: `UploadJob.wait(timeout=0.01)` raising `TimeoutError`;
  `UploadJob.progress()` populated after completion; ordering
  assertion for `reconcile_with_catalog` before
  `reconcile_pending_leases` in `resume_execution`; the "continue"
  branch of `run_upload_engine` final-transition (pending rows
  remain but no failures).
- `_enumerate_work`: assert homogeneous `is_asset` per group, or key
  the group on `(schema, table, is_asset)`; add a `logger.debug`
  summary line.
- Subprocess-invoked tests (e.g., `tests/cli/test_upload_cli.py`)
  can be flaky under certain intra-pytest orderings due to the
  Python 3.13 + `fair_identifiers_client` + `distutils` shim
  interaction. Tighten `deriva_ml/__init__.py` eager imports or
  pin/upgrade `fair_identifiers_client`.
