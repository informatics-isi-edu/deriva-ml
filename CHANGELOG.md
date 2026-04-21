# Changelog

All notable changes to this project are documented here.

## Unreleased — Phase 2 Subsystem 1b: Execution_Status vocabulary

### New

- **`Execution_Status` vocabulary table** in the `deriva-ml` schema. Seeded with 7 canonical terms (Created, Running, Stopped, Failed, Pending_Upload, Uploaded, Aborted) matching the `ExecutionStatus` StrEnum from S1a.
- **FK on `Execution.Status` → `Execution_Status.Name`.** Catalog now rejects any `Execution` insert whose `Status` value isn't one of the seeded canonical terms.
- **`MLVocab.execution_status`** enum member for canonical access to the new vocabulary-table name.
- **`tests/schema/test_vocab_fk_convention.py`** — live-catalog audit test that asserts every FK targeting a vocabulary table in the `deriva-ml` schema references the `Name` column.
- **Conventions section in `docs/reference/schema.md`** documenting the vocabulary-FK-on-Name rule, its rationale, and the enforcing test.

### Changed

- **S0 validator** (`src/deriva_ml/tools/validate_schema_doc.py`): `_extract_fk` now uses `_extract_ast_name_or_enum` on `referenced_table=` values so that `MLVocab.xxx` references in `ForeignKeyDef(...)` resolve to their StrEnum values (Title_Case) rather than the Python identifier (lowercase). Without this fix the validator reported spurious mismatches for FKs whose `referenced_table` was an enum reference.

### Audit notes

- Audit of the pre-S1b schema found **zero violations** of the Name-not-RID rule. Deriva-py's `Table.define_association` default `key_column_search_order = ['Name', 'name', 'ID', 'id']` already picks `Name` for vocabulary targets and falls back to `RID` only for entity targets. The new convention makes this implicit behavior explicit and test-enforced.

### Migration notes

- Existing deployed catalogs MAY need a one-time migration to add the `Execution_Status` vocabulary table + FK if they were created before S1b. A separate cleanup task (outside S1b scope) verifies compliance on any live deployments. New catalogs created via `create_ml_catalog` have everything set up.
- Historical `Execution` rows whose `Status` value was `"Initializing"` or `"Pending"` (pre-S1a) will violate the new FK constraint if the catalog is migrated to include the FK. Users should clean those rows via `gc_executions` or manual updates.

---

## Unreleased — Phase 2 Subsystem 1a: Status-enum reconciliation

### Breaking changes

- **`ExecutionStatus` values are title-case.** `ExecutionStatus.Running` (was `.running`); `.value == "Running"` (was `"running"`). Identifiers and values both changed. Catalog reads via `ExecutionStatus(row["Status"])` now work without translation.
- **Legacy `Status` enum deleted.** `deriva_ml.core.enums.Status` and `deriva_ml.core.definitions.Status` re-export are gone. Use `deriva_ml.execution.state_store.ExecutionStatus` instead.
- **`Execution.update_status(...)` signature changed.** Was `update_status(status: Status, message: str)`. Now `update_status(target: ExecutionStatus, *, error: str | None = None)`. The positional `message` parameter is gone; use `logger.info(...)` for progress chatter. The keyword-only `error=` argument is meaningful only on `Failed` / `Aborted` transitions; on a non-terminal transition the kwarg is ignored with a warning log.
- **`ExecutionRecord.update_status(...)` is new.** Same signature as `Execution.update_status` plus a required `ml: DerivaML` kwarg.
- **`ExecutionMixin._update_status` deleted.** Was dead code (no callers in src/).
- **`DerivaML.status` attribute deleted.** Was vestigial; no readers post-`_update_status` deletion.
- **Legacy lifecycle states `Initializing` and `Pending` no longer exist.** Code that wrote `Status.initializing` is now `logger.info(...)`. `Status.pending` callsites now write `ExecutionStatus.Created`.
- **`Status.completed` split.** Maps to `ExecutionStatus.Stopped` (algorithm finished, before upload) or `ExecutionStatus.Uploaded` (after upload finishes), depending on call-site context.

### Fixed

- **Catalog `Execution.Status` is now actually written.** `Execution.__init__` previously inserted only `Description` and `Workflow`, leaving `Status` NULL on every newly created Execution row. Now writes `Status: "Created"`.
- **`state_machine.transition()` catalog sync now works.** Was using `catalog.put('/entity/...')` which ERMrest rejects with `409 Conflict: Entity PUT requires at least one client-managed key for input correlation.` Switched to `catalog.getPathBuilder().schemas['deriva-ml'].tables['Execution'].update(body)` — the same datapath pattern used elsewhere in the codebase.
- **`_catalog_body_for_execution` no longer sends `Start_Time` / `End_Time`.** The catalog `Execution` table has no such columns; the body shape was off. Lifecycle timestamps live in SQLite only.
- **Phase 1 H2 integration test workaround removed.** With `Execution.Status` now correctly synced, the H2 test no longer needs to read SQLite directly to bypass the reconciliation bug. Assertions tightened.

### Migration notes

- Existing catalogs with historical Execution rows containing `Status = "Initializing"` or `Status = "Pending"` will raise `DerivaMLStateInconsistency` on `resume_execution`. Users with such rows either clean them via `gc_executions` or update the `Status` field manually.
- The catalog `Execution.Status_Detail` column is no longer written by `update_status` (no equivalent in the new signature). Columns persist in the schema; new rows have `NULL` here.
- This subsystem (S1a) does NOT add `Execution_Status` as a controlled vocabulary table — the column remains free-text. Adding the vocabulary table is deferred to a future S1b subsystem that exercises the S0 doc-first workflow.

### Tests

- 11 new unit tests across `tests/execution/test_status_migration.py` and `tests/execution/test_update_status.py` verify the new enum + new method API.
- `tests/test_migration_complete.py` is a grep gate that fails CI if any legacy `Status.xxx` references reappear in `src/deriva_ml/`.
- 254/258 tests in `tests/execution/` pass. (3 pre-existing legacy lifecycle tests in `test_execution.py` that were stuck on the catalog-sync bug now pass after the fix.)

---

## Unreleased — Phase 2 Subsystem 0: Schema-doc source of truth

### New

- **`docs/reference/schema.md`** — authoritative description of the `deriva-ml` schema (tables, columns, FKs, vocabulary seeded terms). Edit this file **first** when changing the schema, then update `src/deriva_ml/schema/create_schema.py` to match.
- **`deriva-ml-validate-schema`** CLI (`src/deriva_ml/tools/validate_schema_doc.py`) — asserts the doc and code agree on structure and seeded terms. Exit 0 on match, 1 on mismatch, 2 on parse error.
- **CI workflow** `.github/workflows/validate-schema.yml` — runs the validator on every PR and push to main.
- **`docs/reference/README.md`** — developer workflow instructions for doc-first schema changes.

### Changed

- Schema changes now **require editing two files together**: `docs/reference/schema.md` and `src/deriva_ml/schema/create_schema.py`. CI enforces they agree.

### Known limitations

- The validator's static AST walker cannot extract dynamically-named tables created via parameterized calls like `create_asset_table(asset_name=...)`. These tables (`Asset`, `File`, `Execution_Execution`, `Execution_Metadata`, `Execution_Asset`) exist at runtime but aren't cross-validated against the doc. The exemption is documented in `tests/tools/test_schema_doc_structure.py`.
- `referenced_schema` in FKs is often written as a parameter (`sname`) or attribute (`schema.name`) in the code. The validator resolves these to a `<dynamic>` sentinel and treats them as matching any doc-side value.

### Not yet supported (filed follow-ups)

- Direction 2: `deriva-ml-validate-schema --against=catalog` for live-catalog drift detection.
- Description validation (table/column prose on doc vs `comment=` on code).
- Annotation validation (display configs, `curie_template`, indexes).
- Ordering-convention test (warn-not-fail on section ordering violations in schema.md).

---

## Unreleased — Phase 1: SQLite-backed execution state

### Breaking changes

Per spec §1.3 and R5.1 (aggressive deprecation), the following APIs
have been removed without a shim. Update call sites before upgrading.

| Removed | Replacement |
|---|---|
| `ml.restore_execution(rid)` | `ml.resume_execution(rid)` |
| `exe.upload_execution_outputs(...)` *(still present; superseded)* | `exe.upload_outputs(...)` |
| `exe.retry_failed()` | `exe.upload_outputs(retry_failed=True)` |
| `Execution.list_nested_executions()` | `ExecutionRecord.list_execution_children(recurse=...)` |
| `ExecutionRecord.list_nested_executions(recurse=...)` | `ExecutionRecord.list_execution_children(recurse=...)` |
| `ExecutionRecord.list_parent_executions(recurse=...)` | `ExecutionRecord.list_execution_parents(recurse=...)` |
| `Execution.datasets` as `list[DatasetBag]` | `Execution.datasets` as `DatasetCollection` (RID-keyed mapping + iterable) |

Note: `exe.upload_execution_outputs(...)` is retained alongside the
new `exe.upload_outputs(...)` Phase-1 entry point. Several internal
call sites (`demo_catalog.py`, `run_notebook.py`, `dataset/split.py`,
`base_config.py`, `runner.py`) still use the legacy method; migration
is a Phase-2 cleanup.

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
- `ml.list_pending_uploads()` (spec §2.7) — flat `list[UploadTarget]`
  view of pending work across executions. Information is already
  reachable via `ml.pending_summary().per_execution[i].rows/assets`;
  a flat sugar method is Phase 2.
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
