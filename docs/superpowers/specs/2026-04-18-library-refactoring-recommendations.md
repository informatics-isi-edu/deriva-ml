# DerivaML library refactoring recommendations

**Status:** Recommendations document.
**Date:** 2026-04-18.
**Scope:** improvements to existing DerivaML code that are NOT required
for the rev-5 stage-and-upload spec to ship, but should be tracked
explicitly so they don't get lost.

Source: findings from the API-consistency analysis
(`2026-04-18-api-consistency-analysis.md`) that apply to existing
library code rather than to the new rev-5 surface. Each entry records
the problem, proposed fix, estimated scope, and suggested priority.

Entries are grouped by concern area. Every entry is standalone —
shippable independently of the others.

---

## 1. Status / lifecycle management

### R1.1 — Single state-machine module (high priority)

**Problem.** The `Status` enum in `src/deriva_ml/core/enums.py` is used
by `Execution.update_status` to write the catalog's `Execution.Status`
column. Rev 5 introduces new status values for the SQLite `executions`
and `pending_rows` tables. There is no shared module, no transition
table, no validation that a given `(from_status, to_status)` pair is
legal, and no documented mapping between SQLite statuses and catalog
statuses.

**Fix.** Extract lifecycle definitions into a single module (e.g.
`src/deriva_ml/execution/lifecycle.py`) containing:
- `ExecutionStatus` enum (superset of today's `Status` plus rev-5
  additions)
- `PendingRowStatus` enum
- `TRANSITIONS: dict[Enum, set[Enum]]` — legal transitions
- `transition(obj, new_status)` helper that validates and applies
- Documented mapping from SQLite `ExecutionStatus` to catalog
  `Execution.Status` column values

All status writes — `Execution.update_status`, the new SQLite writes,
the pending-rows writes — go through this module.

**Scope.** 1-2 days. Includes touching every call site of
`update_status` or direct `Execution.Status` writes.

**Priority.** High. This is called out as Blocking 2.1 in the analysis
because rev 5 introduces new status values without reconciliation.
Finishing rev 5 without this module almost guarantees drift.

### R1.2 — `Execution` Python object reads status from SQLite, not from cached fields (high priority)

**Problem.** `Execution._status` and `Execution._execution_record._status`
are two in-memory caches of status that rev-5 "SQLite is ground truth"
does not eliminate. Status reads can return stale values across
processes.

**Fix.** Refactor `Execution.status` (and related `start_time`,
`stop_time`, `error`) to be read-through properties against SQLite.
No in-memory caching. Writes go through the state-machine helper
(R1.1).

**Scope.** Half day for the getter/setter refactor + tests.

**Priority.** High (Blocking 2.4 in the analysis).

---

## 2. `Execution` class scope

### R2.1 — Move hierarchy queries off `Execution` (high priority)

**Problem.** `Execution` currently carries `add_nested_execution`,
`list_nested_executions`, `list_parent_executions`, `is_nested`,
`is_parent` — these are read-only queries about relationships between
executions, not operations on the running execution itself.
`ExecutionRecord` already has fallback implementations for some.

**Fix.** Consolidate hierarchy queries on `ExecutionRecord`. On
`Execution`, either delegate to the record (`self._record.is_nested`)
or remove entirely — users call `ml.list_executions()` → filter by
`.is_nested`.

**Scope.** 1 day. Non-breaking if `Execution` retains thin shims.

**Priority.** High (Blocking 2.2). Prerequisite for not growing
`Execution` further in rev 5.

### R2.2 — Move `add_files` off `Execution` (medium priority)

**Problem.** `exe.add_files` delegates to `ml.` already. It lives on
`Execution` for historical reasons, not because it's execution-specific
behavior.

**Fix.** Move canonical implementation to `ml.`; `exe.add_files`
becomes a deprecation shim. Align with `ml.create_dataset`,
`ml.add_dataset_members`, etc.

**Scope.** Half day.

**Priority.** Medium. Makes `Execution`'s surface more focused.

### R2.3 — Decide responsibility for `exe.create_dataset` (medium priority)

**Problem.** Creating a dataset is a catalog operation, not an
execution-lifecycle operation. But `exe.create_dataset` provides
automatic execution-provenance attachment (`Dataset_Execution` row).
Today it lives on `Execution`.

**Fix options:**
- **A.** Leave on `Execution`; accept the responsibility smear.
- **B.** Move to `ml.create_dataset(..., execution=exe)` with
  execution as a kwarg. Automatic detection if called within a
  `with exe.execute():` context.
- **C.** Both exist, `exe.create_dataset` is sugar for
  `ml.create_dataset(execution=self)`.

**Recommendation.** C. Symmetric with `exe.table(name)` (which is
sugar over `ml.table(name, scope=exe)` conceptually). Users get both
call patterns.

**Scope.** 1 day.

**Priority.** Medium. Not blocking, but cleans up a long-standing
scope smear.

---

## 3. Handle types and records

### R3.1 — Replace `AssetFilePath` `Path`-subclass hack with composition (medium priority, V2)

**Problem.** `AssetFilePath` subclasses `Path`. The existing code
comments that `with_segments` has to return a plain `Path` because
the subclass constructor can't be called with a single argument. This
is fragile — any new `Path` method added to the stdlib that uses the
"construct another Path" pattern will break.

**Fix.** Replace subclass with composition. `AssetFilePath` becomes a
regular class with a `.path` attribute (a `Path`) and implements
`__fspath__` so it's still usable anywhere a path is expected. The
`RowHandle` protocol (Important 3.1 in the analysis) subsumes the
remaining overlap with `PendingRow`.

**Scope.** 2-3 days. Touches every place that does
`isinstance(handle, Path)` or `isinstance(handle, AssetFilePath)`.

**Priority.** Medium. Not urgent, but the subclass is a known smell.

### R3.2 — Merge `AssetRecord` into `RowRecord` (medium priority, V2)

**Problem.** `AssetRecord` and the rev-5-proposed `RowRecord` are
parallel hierarchies built on the same Pydantic `create_model`
machinery. Both have near-identical docstrings and near-identical
factory code.

**Fix.** Make `AssetRecord` a subclass of `RowRecord` (asset-metadata
column filter). Today's `asset_record_class` becomes a thin wrapper
over `record_class(..., column_filter=model.asset_metadata(table))`.
Single source of truth for dynamic record construction.

**Scope.** Half day.

**Priority.** Medium. Natural cleanup during rev 5 implementation if
the implementer touches the factory code.

### R3.3 — `RowHandle` Protocol (shipped with rev 5)

**Problem.** `PendingRow` and `AssetFilePath` share a handle contract
(`.rid`, `.status`, `.metadata`, `.error`) but aren't related by
inheritance. User code doing `isinstance(x, PendingRow)` will be
wrong for asset handles.

**Fix.** Define a `RowHandle` Protocol (structural). Both classes
conform. Document: don't check `isinstance` of the class; check
conformance to the protocol. This was added as Important 3.1 in the
analysis and should land with rev 5 itself.

**Scope.** Half day.

**Priority.** Ship with rev 5.

---

## 4. File-tree vs SQLite

### R4.1 — Remove `execution_rids()` file-tree scan after rev 5 ships (medium priority)

**Problem.** `src/deriva_ml/dataset/upload.py::execution_rids()`
walks `{working_dir}/deriva-ml/execution/` to enumerate executions.
Rev 5's SQLite-backed registry supersedes this. After rev 5,
`restore_execution` falls back to `execution_rids` only for the
single-RID "no argument" path.

**Fix.** Once rev 5 is shipped and `restore_execution` is fully a shim
for `resume_execution`, delete `execution_rids()` and its single
caller. Any script depending on it today should be migrated to
`ml.list_executions()`.

**Scope.** Half day. Prerequisite: rev 5 is in production.

**Priority.** Medium. Technical debt cleanup.

### R4.2 — `configuration.json` on disk becomes derived (ship with rev 5)

**Problem.** Two config sources after rev 5: `configuration.json` on
disk (for catalog upload as an Execution_Metadata asset) and
`executions.config_json` in SQLite (for local resume). Sync burden.

**Fix.** SQLite is the authoritative write path. The disk file is
regenerated from SQLite at upload time only. Existing restore fallback
that reads `configuration.json` handles workspaces without the SQLite
table (e.g., upgrades from older code) but is not the primary source.

**Scope.** Small; part of rev 5 implementation.

**Priority.** Ship with rev 5 (Important 3.5).

---

## 5. Shim management

### R5.1 — Explicit deprecation calendar (ship with rev 5)

**Problem.** Rev 5 adds at least five deprecation shims (§2.10):
`exe.asset_file_path`, `ml.asset_record_class`, `ml.restore_execution`,
`exe.table_path`, plus the `ml.find_executions` / `list_executions`
split. Pre-existing shims already include `working_data` → `workspace`
(with a runtime `DeprecationWarning` but no removal date) and
`AssetRID` back-compat. Without a calendar, shims accumulate
indefinitely.

**Fix.** Each shim gets a "remove in" version in its docstring and in
a tracked `DEPRECATIONS.md` file in the repo root. Suggested calendar:
- V1 (this release): add shims, doc-only deprecation.
- V2: emit `DeprecationWarning` at import/call time.
- V3: remove shims.

**Scope.** Half day to write the calendar document + docstrings.

**Priority.** Ship with rev 5. Prevents uncontrolled shim growth.

### R5.2 — `AssetRID` dataclass removal (low priority, V2+)

**Problem.** `AssetRID` in `execution_configuration.py` is a
back-compat dataclass, handled explicitly in `validate_assets`. No
current tests cover it as a primary path.

**Fix.** Audit callers. If none remain in user-facing docs/tests,
delete in V2.

**Scope.** Half day audit + removal.

**Priority.** Low. Clean-up; nothing breaks if left in.

### R5.3 — `working_data` shim removal (low priority, V2)

**Problem.** `DerivaML.working_data` already has a
`DeprecationWarning` pointing to `DerivaML.workspace` — but no
removal date.

**Fix.** Remove in V2 per the calendar in R5.1.

**Scope.** 1 line of code, plus any test updates.

**Priority.** Low. Cosmetic cleanup.

---

## 6. ML-practitioner ergonomics

### R6.1 — `create_execution` kwargs convenience (high priority, ship with rev 5 or immediately after)

**Problem.** Today users write:
```python
wf = ml.lookup_workflow_by_url("https://github.com/me/repo/blob/<sha>/train.py")
config = ExecutionConfiguration(
    datasets=[DatasetSpec(rid="D-001", version="1.0.0")],
    workflow=wf,
    description="Train",
)
exe = ml.create_execution(config)
```

Four nested constructor calls; three Deriva-specific concepts to hold
in mind.

**Fix.** Accept kwargs on `create_execution`:
```python
exe = ml.create_execution(
    datasets=["D-001"],
    workflow="https://.../train.py",
    description="Train",
)
```

Strings autoresolve to the right underlying objects. The explicit
`ExecutionConfiguration(...)` form stays for hydra-zen pipelines and
fine-grained control.

**Scope.** 1 day.

**Priority.** High. Ship with or immediately after rev 5. Saves 3+
lines from every new ML script (ML review F2, daily friction).

### R6.2 — Metric / param logging primitives (high priority, ship with rev 5 or immediately after)

**Problem.** ML users expect:
```python
exe.log_metric("val_loss", 0.23, step=10)
exe.log_param("lr", 0.001)
```

The rev-5 `exe.table("X").insert(...)` is semantically correct but
expects the user to provision canonical `Metric` / `Param` tables.

**Fix.** Provision canonical tables automatically in the ML schema:
- `Execution_Metric(Execution, Name, Value, Step, Timestamp)`
- `Execution_Param(Execution, Name, Value)`

Surface sugar on `Execution`:
```python
exe.log_metric(name, value, step=None)
exe.log_param(name, value)
```

Sugar delegates to `exe.table(...).insert(...)`.

**Scope.** 1-2 days for schema + sugar methods + migration note for
existing catalogs.

**Priority.** High. Single biggest "MLflow expectations unmet" gap
(ML review recommendation #2).

### R6.3 — Auto-upload on context-manager exit (medium priority)

**Problem.** Existing `upload_execution_outputs` docstring says in
ALL CAPS: "must be called AFTER exiting the context manager, not
inside it." That's a warning the shape is wrong. Users forget; nothing
gets uploaded.

**Fix.** Default behavior on clean exit in online mode: auto-upload.
```python
with exe.execute() as e:
    ...
# online + clean exit → auto-upload happens here
# failed exit or offline mode → no auto-upload
```

Opt-out: `with exe.execute(auto_upload=False):`.

**Scope.** Half day; modify `Execution.__exit__`.

**Priority.** Medium. Eliminates a daily footgun (ML review F7).

### R6.4 — `ml_run` decorator (lower priority, V2 candidate)

**Problem.** Users writing one-off training/inference scripts want a
one-liner that handles workflow resolution, execution creation,
context management, and upload. MLflow autologging and
`wandb.init()` provide this.

**Fix.** Provide a decorator:
```python
@ml_run(datasets=["D-001"], description="Train seg model")
def main(exe):
    ...
```

Decorator internally: looks up the calling script's git info for
workflow, creates execution, enters context, passes `exe` in, handles
exit + upload.

**Scope.** 2-3 days. Needs thought about how workflow auto-resolution
from `sys.argv[0]` interacts with interactive (Jupyter) sessions.

**Priority.** Lower. High user value, but not urgent while the kwargs
form (R6.1) covers most of the need.

### R6.5 — Workflow auto-resolution from git state (medium priority)

**Problem.** Users creating a new execution must explicitly look up
or create a `Workflow`. ML mental model is "this run is tied to my
git commit," not "look up a Workflow object by URL."

**Fix.** If no `workflow` kwarg is passed to `create_execution`, the
library auto-resolves from:
1. The calling script's file (`sys.argv[0]` or the caller's `__file__`)
2. Its git repo (if any)
3. The current commit SHA
4. Constructs a URL like `https://<remote>/blob/<sha>/<relpath>`
5. Looks up or creates a Workflow for that URL

Users with unusual setups (interactive notebooks, non-git repos) pass
`workflow=` explicitly.

**Scope.** 2 days. Needs the git-detection logic (most of it exists
for the `--allow-dirty` check today) plus the URL-construction logic.

**Priority.** Medium. Big reproducibility story unlock.

---

## 7. Naming polish

### R7.1 — `gc_executions` → `clean_executions` (low priority)

Align with existing `clean_*` / `clear_*` convention in the codebase.

### R7.2 — `DerivaMLOfflineError` under `DerivaMLConfigurationError` (ship with rev 5)

Add to exception hierarchy, mirror the taxonomy convention.

### R7.3 — `handle.pending()` → `handle.list_pending()` (low priority)

Matches `ml.list_executions()`. `pending` reads as a property name.

### R7.4 — `handle.rid` vs `handle.rid_or_none` (low priority)

Currently `.rid` is lazy — reading it may trigger a network call.
Consider adding `.rid_or_none` as a pure read (returns None if not
yet leased) for scripts that want to inspect without side effects.

---

## 8. Priority summary

**Ship with rev 5:**
- R1.1 Single state-machine module (blocking)
- R1.2 Execution status via SQLite (blocking)
- R2.1 Move hierarchy queries off Execution (blocking)
- R3.3 RowHandle Protocol
- R4.2 configuration.json as derived
- R5.1 Deprecation calendar
- R7.2 DerivaMLOfflineError in hierarchy

**Ship with rev 5 or immediately after (high ML value):**
- R6.1 create_execution kwargs
- R6.2 log_metric / log_param primitives
- R6.3 Auto-upload on context-manager exit

**V2 targets:**
- R2.2 Move add_files to ml.
- R2.3 exe.create_dataset on ml. too
- R3.1 AssetFilePath composition over Path subclass
- R3.2 Merge AssetRecord into RowRecord
- R4.1 Remove execution_rids file-tree scan
- R6.4 ml_run decorator
- R6.5 Workflow auto-resolution from git
- R7.1 gc_executions → clean_executions
- R7.3 pending → list_pending
- R7.4 rid_or_none

**V3+:**
- R5.2 AssetRID removal
- R5.3 working_data removal
