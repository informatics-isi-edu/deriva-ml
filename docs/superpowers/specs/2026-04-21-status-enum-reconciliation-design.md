# Status Enum Reconciliation — Design Spec

**Date:** 2026-04-21
**Author:** Claude (with Carl)
**Status:** Approved, ready for implementation-plan
**Phase:** Phase 2, Subsystem 1 (of 4)
**Parent spec:** `2026-04-18-sqlite-execution-state-design.md` (Phase 1)

## 1. Motivation

Phase 1 shipped with two parallel execution-status vocabularies:

- **Legacy** `Status` enum (`src/deriva_ml/core/enums.py`): title-case values `Initializing`, `Created`, `Pending`, `Running`, `Aborted`, `Completed`, `Failed`. Written to the catalog `Execution.Status` field and consumed by the catalog's `Execution_Status` vocabulary (terms match).
- **Phase-1 new** `ExecutionStatus` enum (`src/deriva_ml/execution/state_store.py`): lowercase values `created`, `running`, `stopped`, `failed`, `pending_upload`, `uploaded`, `aborted`. Used by the state-machine module, SQLite registry, and Phase 1's public API (`list_executions`, `resume_execution`, etc.).

`Execution._initialize_execution` and several other internal call sites still write the legacy title-case values to the catalog. Phase 1's `reconcile_with_catalog` reads those values and tries `ExecutionStatus(catalog_row["Status"])`, which raises `ValueError` (wrapped as `DerivaMLStateInconsistency`). The Phase-1 integration test (H2) had to work around this by reading SQLite directly instead of using `resume_execution`.

The CHANGELOG flagged this as the "critical Phase 2 item." This spec is the fix.

## 2. Decisions (already made during brainstorming)

The brainstorming sessions settled these:

- **Option A (full migration).** Delete the legacy `Status` enum and its consumers; rewrite every internal call site. Backwards compatibility is not a constraint because `update_status` is not called by users outside the library.
- **W3 (title-case values).** Change `ExecutionStatus` values from lowercase to title-case to match the catalog `Execution_Status` vocabulary directly. No translation layer at the wire boundary; the in-memory enum *is* the wire format.
- **Method name preserved: `Execution.update_status(...)`.** Keep the legacy method name — it reads cleanly on an Execution instance and existing call sites barely change. The Phase-1 state-machine module function `transition()` stays internal; `update_status` wraps it.
- **M2 (drop status_message).** Legacy messages are split across three categories: progress chatter (→ `logger.info`), error context (→ existing `error` column), completion notes (redundant with the status value itself). No new `status_message` column on SQLite or the catalog.
- **B1 (delete `DerivaML.status`).** The `self.status = Status.pending.value` line in `core/base.py:353` is vestigial. Grep confirmed no readers; delete.
- **E1 (strict reconciliation).** Unknown legacy status values in catalog rows raise `DerivaMLStateInconsistency` — consistent with Phase 1. Users with historical rows migrate their data; the library doesn't silently accept unknown values.

## 3. Semantic mappings

The legacy `Status` → new `ExecutionStatus` mappings. These are binding for every call site:

| Legacy                    | New action                              | Rationale                                                         |
|---------------------------|-----------------------------------------|-------------------------------------------------------------------|
| `Status.initializing`     | **Delete.** Replace with `logger.info`. | Not a lifecycle state — progress chatter during setup.            |
| `Status.created`          | `ExecutionStatus.Created`               | 1:1.                                                              |
| `Status.pending`          | **Delete.** Collapse into `Created`.    | Phase 1's state machine doesn't track a separate "pending" state. |
| `Status.running`          | `ExecutionStatus.Running`               | 1:1.                                                              |
| `Status.completed`        | **Split per call site.** See §4.        | Legacy conflated "algorithm done" with "data uploaded."           |
| `Status.aborted`          | `ExecutionStatus.Aborted`               | 1:1.                                                              |
| `Status.failed`           | `ExecutionStatus.Failed`                | 1:1.                                                              |

### 3.1 The `Status.completed` split

Each of the ~3 legacy `Status.completed` call sites maps to one of:

- `ExecutionStatus.Stopped` — algorithm finished successfully, data not yet uploaded. The `Execution.__exit__` path.
- `ExecutionStatus.Pending_Upload` — in-flight upload. Mid-`upload_execution_outputs` progress.
- `ExecutionStatus.Uploaded` — upload finished successfully. Tail of `upload_execution_outputs`.

Each call site gets its mapping decided explicitly during implementation. No blanket rule.

## 4. API changes

### 4.1 `ExecutionStatus` enum (value change)

```python
class ExecutionStatus(StrEnum):
    """Lifecycle status for an Execution (see Phase 1 spec §2.2).

    Values are title-case to match the catalog's Execution_Status
    vocabulary — a catalog row's Status field can be parsed directly
    via ExecutionStatus(row["Status"]).
    """
    Created = "Created"
    Running = "Running"
    Stopped = "Stopped"
    Failed = "Failed"
    Pending_Upload = "Pending_Upload"
    Uploaded = "Uploaded"
    Aborted = "Aborted"
```

Python identifier casing matches the enum's string value (so `ExecutionStatus.Running.name == "Running"` and `.value == "Running"`). This is a deviation from PEP 8's lowercase convention for enum members, but it optimizes for readability at call sites (`ExecutionStatus.Running` matches what appears in the catalog) and keeps identifier == value. Precedent: `http.HTTPStatus` in the stdlib uses uppercase identifiers matching HTTP-status names.

### 4.2 `Execution.update_status`

```python
def update_status(
    self,
    target: ExecutionStatus,
    *,
    error: str | None = None,
) -> None:
    """Transition this execution to a new status.

    Thin wrapper around state_machine.transition() that validates
    against ALLOWED_TRANSITIONS, updates the SQLite registry, and
    syncs to the catalog when online.

    Args:
        target: Target status.
        error: For Failed/Aborted transitions, a human-readable
            error message. Written to the `error` column. Ignored
            (with a logger.warning) for non-terminal transitions.

    Raises:
        InvalidTransitionError: If the (current, target) pair is
            not in ALLOWED_TRANSITIONS.
        DerivaMLStateInconsistency: If state_machine reconciliation
            detects catalog/SQLite divergence during the sync step.
    """
```

### 4.3 `ExecutionRecord.update_status`

```python
def update_status(
    self,
    target: ExecutionStatus,
    *,
    ml: "DerivaML",
    error: str | None = None,
) -> None:
    """Parallel to Execution.update_status; Records require an ml= kwarg."""
```

### 4.4 Deletions

- `src/deriva_ml/core/enums.py::Status` — entire class deleted.
- `Execution._initialize_execution` — legacy method deleted (its lifecycle work was already taken over by `create_catalog_execution` in Phase 1).
- `DerivaML.status` attribute — the `self.status = Status.pending.value` in `core/base.py:353`. Verify no readers first.
- Legacy `update_status(...)` signature accepting `Status` — the whole method body gets replaced per §4.2; no separate deletion.

## 5. Internal call sites

Identified by grep (non-exhaustive — implementation will do a full sweep):

| File                                   | Line      | Legacy call                                         | New call                                              |
|----------------------------------------|-----------|-----------------------------------------------------|-------------------------------------------------------|
| `src/deriva_ml/core/base.py`           | 353       | `self.status = Status.pending.value`                | **Delete.**                                           |
| `src/deriva_ml/core/base.py`           | 380–381   | `update_status(Status.aborted, "…")`                | `exe.update_status(Ex.Aborted, error="…")`            |
| `src/deriva_ml/execution/execution.py` | 307       | `status=Status.created`                             | `status=Ex.Created`                                   |
| `src/deriva_ml/execution/execution.py` | 514       | `update_status(Status.initializing, …)`             | `logger.info("Materialize bag %s…", dataset.rid)`     |
| `src/deriva_ml/execution/execution.py` | 526       | `update_status(Status.running, "Downloading…")`     | `logger.info("Downloading assets…")` (progress)       |
| `src/deriva_ml/execution/execution.py` | 583       | `update_status(Status.pending, "Init finished.")`   | **Delete.** (Already in `Created` post-Phase-1.)      |
| `src/deriva_ml/execution/execution.py` | 980       | `update_status(Status.initializing, …)`             | `logger.info("Start execution…")`                     |
| `src/deriva_ml/execution/execution.py` | 1020      | `update_status(Status.completed, "Algo ended.")`    | `update_status(Ex.Stopped)`                           |
| `src/deriva_ml/execution/execution.py` | 1122      | `update_status(Status.running, "Uploading files…")` | `update_status(Ex.Pending_Upload)` (enter upload)     |
| `src/deriva_ml/execution/execution.py` | 1134      | `update_status(Status.failed, error)`               | `update_status(Ex.Failed, error=error)`               |
| `src/deriva_ml/execution/execution.py` | 1166      | `update_status(Status.running, "Updating feat…")`   | `logger.info("Updating features…")` (progress)        |
| `src/deriva_ml/execution/execution.py` | 1177      | `update_status(Status.running, "Upload complete")`  | `logger.info("Upload assets complete")` (progress)    |
| `src/deriva_ml/execution/execution.py` | 1378      | `update_status(Status.completed, "Success end.")`   | `update_status(Ex.Uploaded)`                          |
| `src/deriva_ml/execution/execution.py` | 1384      | `update_status(Status.failed, error)`               | `update_status(Ex.Failed, error=error)`               |
| `src/deriva_ml/dataset/dataset.py`     | 2666, 2672| `update_status(Status.running, msg)` via callback   | `logger.info(msg)` or keep callback with `Ex.Running` |

`src/deriva_ml/execution/execution_record.py` — the field's type annotation is `Status`; rewrite to `ExecutionStatus`. The PrivateAttr default of `Status.created` becomes `ExecutionStatus.Created`.

`src/deriva_ml/core/mixins/execution.py` — docstring examples mentioning `Status.completed` etc. update to `ExecutionStatus.Uploaded` where relevant.

## 6. Schema seeding

**Correction from earlier draft:** `Execution.Status` is a plain `text` column (`src/deriva_ml/schema/create_schema.py:149`), NOT a foreign key to an `Execution_Status` vocabulary table. There is no vocab table to seed. The column holds whatever string the library writes.

This simplifies §6 to nothing:
- No vocabulary table exists.
- No schema-init routine seeds `Execution_Status` terms.
- No term-migration helper needed.

**The only discipline:** post-migration, `Execution.Status` values written to the catalog are constrained by the library to the 7 canonical title-case strings (the `ExecutionStatus` enum values). Historical rows with `"Initializing"` / `"Pending"` remain as free-text until the user manually updates them or they age out via `gc_executions`.

`Execution.Status_Detail` (a sibling text column on the catalog Execution table) stays unused by the library — it was a pre-existing column that legacy `update_status` would populate with the human-readable message. With M2 (drop status_message), this column is simply not written to. It may be removed from the schema in a future migration; not in scope here.

## 7. Error handling

(Covered in §5 of brainstorming; summarized here.)

- **Invalid transition**: existing `InvalidTransitionError` from `state_machine.transition`. `update_status` surfaces it.
- **Catalog sync failure (online mode)**: existing `sync_pending=True` soft-fail. No change.
- **Offline mode**: writes SQLite only; sets `sync_pending=True`. No change.
- **Unknown status from catalog during reconciliation** (E1): raises `DerivaMLStateInconsistency` with a message that names the offending value and the execution RID. Users clean up historical rows (documented in CHANGELOG).

## 8. Testing

### 8.1 New test files

- `tests/execution/test_status_migration.py` — enum value correctness:
  - All 7 `ExecutionStatus` members present with title-case values.
  - `ExecutionStatus("Running")` succeeds.
  - `ExecutionStatus("running")` raises `ValueError`.
  - Round-trip: catalog row `{"Status": "Running"}` parses to `ExecutionStatus.Running`.

- `tests/execution/test_update_status.py` — new public method:
  - Valid transition updates SQLite status + syncs catalog (online).
  - Invalid transition raises `InvalidTransitionError`.
  - `error="…"` kwarg writes the `error` column on `Failed` / `Aborted`.
  - `error="…"` on a non-terminal transition logs a warning via `logger.warning` and is otherwise ignored. (The kwarg is accepted silently to keep call-site patterns uniform; misuse is a logged-only soft error, not a raise.)
  - Offline mode: SQLite updates, `sync_pending=True`, no catalog call.
  - `ExecutionRecord.update_status(target, *, ml, error=None)` parallel coverage.

- `tests/test_migration_complete.py` — grep gate:
  - Greps `src/deriva_ml/` for remaining references to the legacy `Status` enum or its members. Fails if any match. Catches missed migrations before they ship.
  - Specifically greps for: `class Status`, `Status.pending`, `Status.running`, `Status.completed`, `Status.initializing`, `Status.aborted`, `Status.failed`, `Status.created`, `from deriva_ml.core.enums import Status`, `from .enums import Status`.

### 8.2 Existing test updates

- `tests/execution/test_state_machine.py` — lowercase `"running"` literals → title-case `"Running"` or `ExecutionStatus.Running.value`.
- `tests/execution/test_state_store.py` — same sweep.
- `tests/execution/test_execution_registry.py` — same.
- `tests/execution/test_execution_readthrough.py` — same.
- `tests/execution/test_execution_hierarchy.py` — same.
- `tests/integration/test_phase1_end_to_end.py`:
  - `test_online_create_offline_stage_online_upload`: remove the "workaround: read SQLite directly" comment; tighten the final assertion to `exe_final.status is ExecutionStatus.Uploaded` via `resume_execution`.
  - Tighten the `with exe.execute(): pass` post-assertion from `in (Stopped, Running)` to `is ExecutionStatus.Stopped`.

### 8.3 Regression

The existing 627-test Phase-1 suite must pass post-migration. Any test that hardcoded a lowercase value gets swept; any that incidentally checked `Status.something` gets rewritten.

## 9. Delivery order (for the plan)

Rough task sequence; the implementation plan will refine into bite-sized steps:

1. Update `ExecutionStatus` enum values (the 7-character strings).
2. Sweep SQLite-written status string literals (if any hardcoded) — ensure they use the enum.
3. Implement `Execution.update_status(target, *, error=None)`.
4. Implement `ExecutionRecord.update_status(target, *, ml, error=None)`.
5. Migrate every call site per §5's table.
6. Delete `Status` enum + `DerivaML.status` attribute + `_initialize_execution` (verify no readers first).
7. Add test files (§8.1).
8. Update existing tests (§8.2).
9. Run the grep gate; address any remaining hits.
10. Full regression: 627 unit + 2 integration must pass.
11. CHANGELOG: add a "Phase 2 Subsystem 1" entry noting the breaking change.

## 10. Non-goals

Explicitly out of scope; separate subsystems handle them:

- Real `_invoke_deriva_py_uploader` implementation (Subsystem 3).
- `parallel_files` / `bandwidth_limit_mbps` threading (Subsystem 3).
- Offline-mode `DerivaML.__init__` network-I/O skip (Subsystem 4).
- `UploadJob._state_lock` race-fixing (Subsystem 4).
- Full `TableHandle` / `AssetTableHandle` surface (Subsystem 2, the real Phase 2).
- Metric / Param as first-class concepts (Subsystem 2).
- Deprecation warnings for the legacy `Status` enum. We're deleting it outright; there are no external users to warn.

## 11. Open questions

None at design freeze. Decisions made during brainstorming that might surface as follow-ups:

- If an unexpected external consumer of the legacy `Status` enum surfaces post-merge, revert to Option C (shim). Unlikely given the current architecture, but noted.
- (The earlier-drafted bullet about seeding an `Execution_Status` vocabulary is removed — §6 documents the correction: `Execution.Status` is a free-text column, not a vocab FK.)
