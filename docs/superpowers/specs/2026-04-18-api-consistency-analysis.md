# DerivaML API consistency analysis (execution + row-write subset)

**Status:** Analysis document.
**Date:** 2026-04-18.
**Scope:** the DerivaML public API surface that touches creating / managing
executions and their data, as it stands after the rev-5 stage-and-upload
spec is applied. Explicitly excludes denormalization, catalog cloning,
and hydra-zen config internals.

This document collates two independent reviews:
- a software-engineering review focused on consistency, DRY, structural
  metrics, separation of concerns, and naming
- an ML-practitioner review focused on alignment with ML experiment
  workflows, ecosystem fit (MLflow / W&B / HuggingFace), and
  scenario-by-scenario ergonomics

Where the reviews overlap, a finding is reinforced. Where they diverge,
both perspectives are recorded.

The **companion document** `2026-04-18-library-refactoring-recommendations.md`
extracts findings that apply to existing library code outside the
stage-and-upload spec's scope. **Inline updates** to the rev-5 spec
itself are tracked in a `rev 6` revision of that spec.

## 1. What the combined API gets right

Both reviews identified the same core strengths. Keep these properties
as non-negotiable invariants.

### 1.1 Architectural strengths (SE)

- **The rev-5 one-verb `.insert()` collapse** is the right move.
  Identical code online and offline, mode as a property of the
  instance rather than a per-call choice.
- **SQLite as authoritative state** inverts today's filesystem-scan
  model (`execution_rids()` walking `os.listdir`) and fixes cross-
  process, cross-machine resumption.
- **Mixin decomposition on `DerivaML`** is clean; individual mixins
  have clear responsibilities.
- **The `Workflow` writable-proxy pattern** (setattr intercept → catalog
  write) is elegant and should be the model for other mutable objects.
- **`ManifestStore` + `AssetManifest` provide most of what
  `pending_rows` needs.** Rev 5 builds on this foundation, not a
  rewrite.
- **Typed record classes** (`feature_record_class`, `asset_record_class`,
  new `record_class`) are a good idiom that rev 5 generalizes well.

### 1.2 ML-workflow strengths (ML)

- **Dataset-bag + versioning is genuinely differentiated.** Versioned,
  snapshot-isolated, materializable datasets with catalog provenance
  outperform MLflow, W&B, and HuggingFace on reproducibility.
- **Provenance by construction.** The `Workflow → Execution →
  {Dataset_Execution, Asset_Execution, Feature values}` chain is
  airtight. Every catalog row knows which execution produced it.
- **Offline mode (rev 5) is a first-class differentiator.** No
  competitor offers create-online / work-offline / upload-online as a
  native workflow.
- **1-line DataFrame bulk insert** (`exe.table("X").insert(df)`) is
  competitive with HuggingFace `Dataset.add_item` and better than
  MLflow for this case.
- **Resume-a-failed-run in 3-4 lines** (`find_incomplete_executions` +
  `resume_execution` + `retry_failed`) is genuinely better than
  MLflow's "just re-run" model.

## 2. Blocking issues (must resolve before rev 5 merges)

These are the four findings the SE review classed as blocking. Each
has been cross-checked against the ML review; none is dismissed.

### 2.1 Status vocabulary is triplicated with no shared definition

**Finding.** Three status value sets exist or will exist:

1. **Existing `Status` enum** (`src/deriva_ml/core/enums.py`, used by
   `Execution.update_status` to write the catalog's `Execution.Status`
   column): `pending | running | completed | failed | aborted`.
2. **Proposed `executions.status` in SQLite** (rev-5 §2.3.1): `created
   | running | stopped | failed | pending_upload | uploaded | aborted`.
3. **Proposed `pending_rows.status` in SQLite** (rev-5 §2.3.2):
   `staged | leasing | leased | uploading | uploaded | failed`.

There is no shared helper, no central enum, no documented mapping
between the SQLite `executions.status` and the catalog's
`Execution.Status` column — even though a write to one implies a
concurrent write to the other.

**Why blocking.** `Execution.update_status` is in production, writing
`Status.value` to the catalog. Rev 5 silently redefines the set of
values applied to the same column (`pending_upload`, `uploaded` are
new). Existing catalog consumers reading `Execution.Status` will see
values they don't understand. No shim can paper over this.

**Fix.** Before implementation starts:

1. Define the full lifecycle state machine once — an enum with valid
   transitions — in a single module (e.g. `deriva_ml.execution.status`
   or extending the existing `enums.py`).
2. Make SQLite writes and catalog `Execution.Status` updates go through
   it.
3. Decide explicitly: does the catalog `Execution.Status` column gain
   `pending_upload` / `uploaded`, or are those SQLite-only with a
   mapping to `running` / `completed` on the catalog side? Document
   the choice.
4. Factor the `pending_rows.status` state machine as a separate enum
   in the same module so the parallel lifecycle is visible but not
   conflated.

**Scope.** 1-2 days; mostly writing the state-machine module and its
transition table, plus a migration note about what existing callers
will see.

### 2.2 `Execution` is a god-object and rev 5 adds more without subtraction

**Finding.** Today `Execution` carries: configuration, workflow
linkage, dataset materialization, asset download, asset upload, status
tracking, manifest management, feature management, dataset creation,
`add_files`, nested-execution hierarchy queries, context manager.
>20 public methods already. Rev 5 adds `.table()`, `.abort()`,
`.retry_failed()` plus implicit registry interactions and the lazy
drain. Public method count heads for ~25+.

**Why blocking.** Adding to a god-object is a technical-debt
anti-pattern, and it compounds the ML-review finding below (F-D.ml,
"`ml.` and `exe.` tab-complete dumps too many verbs"). Users can't
discover what's relevant; maintainers can't reason about responsibility
boundaries.

**Fix.** Before rev 5 lands, extract at least:

1. **Hierarchy queries** (`add_nested_execution`, `list_nested_executions`,
   `list_parent_executions`, `is_nested`, `is_parent`) to
   `ExecutionRecord` where they already have fallbacks. These are
   read-only queries; they don't need the full Execution context.
2. **`add_files`** to `ml.` since it already delegates upward.

Then add the rev-5 surface (`table`, `abort`, `retry_failed`) to the
slimmed class. Public methods drop to ~20-22 after the new additions,
not grow to 25+.

**Scope.** 1 day refactor, no user-visible breakage (shim the moved
methods back on the old receiver).

### 2.3 `create_execution` has different failure semantics in online vs offline mode, signature unchanged

**Finding.** Rev 5 §2.1 says `create_execution` in `mode="offline"`
raises `DerivaMLOfflineError`. The method signature in
`ExecutionMixin.create_execution` doesn't change. A notebook configured
with `mode="offline"` will see a runtime error for what feels like a
silent state mismatch.

**Why blocking.** This is a contract change on a method used everywhere
in existing user scripts. Split-script workflows require users to
remember that the first script must be online — not something the
signature conveys.

**Fix.** Not a code change — a documentation-and-error-message
requirement:

1. Prominent note in `create_execution`'s docstring that it requires
   online mode.
2. `DerivaMLOfflineError` message must be actionable: "create_execution
   requires online mode; reconnect or resume an existing execution
   with `ml.resume_execution(rid)`."
3. The rev-5 spec's §2.1 needs a single sentence making this explicit
   rather than implicit.

**Scope.** Documentation only, but pre-merge.

### 2.4 `Execution.status` would have four sources of truth

**Finding.** Today: `Execution._status`, `Execution._execution_record._status`,
and catalog `Execution.Status` column — three. The existing `status`
property already has a conditional read path (`if self._execution_record
is not None: return self._execution_record.status`). After rev 5 adds
SQLite `executions.status`, that's four sources.

**Why blocking.** The rev-5 design principle is "SQLite is ground
truth." That principle is violated the moment the Python object keeps
its own `_status` field that could diverge across processes.

**Fix.** Make `Execution.status` a pure read-through property against
SQLite. Eliminate `_status` and `_execution_record._status` as
independent state. This is what the spec implies but doesn't spell
out. The rev-5 spec needs an explicit §2.2.X commitment: "`Execution`
Python objects are thin views over the SQLite `executions` row; they
do not cache status."

**Scope.** 2-4 hours; touches only the `status` getter/setter.

## 3. Important issues (resolve during rev 5 implementation)

### 3.1 Three handle types with overlapping but distinct roles

**SE finding.** `PendingRow`, `AssetFilePath`, `AssetDirectoryHandle`.
The first two share a handle contract (`.rid`, `.status`, `.metadata`,
`.error`); only `AssetFilePath` is also a `Path`. Users may write
`isinstance(x, PendingRow)` and be wrong for asset rows.

**Fix.** Define a `RowHandle` Protocol (structural). Both `PendingRow`
and `AssetFilePath` conform; document that users should check the
protocol, not the class. No inheritance — the `Path` subclass hack in
`AssetFilePath` is already load-bearing and can't cleanly inherit from
a shared base without breaking `Path` semantics.

**Scope.** Half day — one Protocol declaration + docstring updates.

### 3.2 `record_class()` on `TableHandle` couples schema to execution

**SE finding.** `asset_record_class` is a pure schema operation — no
execution needed. Placing `record_class()` only on `TableHandle`
(which requires `exe.table(name)`) means users need an execution
context to inspect a table's Pydantic schema. That's backwards for
schema introspection.

**Fix.** Add `ml.table(name).record_class()` as a sibling to
`exe.table(name)`. `ml.table(name)` returns a `TableHandle` that can
do read-only / schema operations (including `record_class`) but
raises on write operations (`insert`, `asset_file`) since there's no
execution context.

**Implication for rev 5.** This reverses rev-4's decision to drop
`ml.table(name)`. The re-introduction is narrower: only read-only /
schema operations, not a second write path.

**Scope.** 1 day in implementation; reopens and resolves a design
question the spec had closed.

### 3.3 `list_executions` vs `find_executions` silently divergent semantics

**SE finding.** `find_executions` (existing) is server-side; returns
all catalog executions. `list_executions` (new) is local-only. The
rev-5 §2.10 shim that says "`find_executions` delegates to
`list_executions` when called locally" is a silent semantics change
for existing callers.

**Fix.** Keep them semantically distinct:
- `find_executions(...)` = server-side, unchanged behavior.
- `list_executions(...)` = local SQLite only, no fallback.
- Remove the shim sentence from the rev-5 spec.

Users who need the union write `set(ml.find_executions()) |
set(ml.list_executions())` explicitly.

**Scope.** Spec edit + one test. No code change beyond keeping the
existing `find_executions` untouched.

### 3.4 `retry_failed` and `upload_execution_outputs` are two drain paths

**SE finding.** Both drain pending rows. `retry_failed` is "reset
failed → leased, then upload." That's a parameter on upload, not a
second method.

**Fix (option A).** Single method:
```python
exe.upload_execution_outputs(retry_failed: bool = False)
```

**Fix (option B).** Split into pure operations:
- `exe.reset_failed()` — resets failed → leased, no upload
- `exe.upload_execution_outputs()` — drains current leased/staged

User composes as needed: `exe.reset_failed(); exe.upload_execution_outputs()`.

**Recommendation.** Option A — the common case is "retry the failures
along with any new staged rows." Option B forces extra boilerplate.

**Scope.** API decision + doc change.

### 3.5 Config persistence duplicated (disk JSON + SQLite)

**SE finding.** `configuration.json` on disk (asset for upload) AND
`executions.config_json` in SQLite. Both are "the config." Sync
burden; stale-one-vs-the-other divergence possible.

**Fix.** SQLite is the single write-source-of-truth. The disk file is
regenerated from SQLite at upload time only. Existing restore path
that reads `configuration.json` becomes a fallback for workspaces
without the SQLite table (e.g., migrating from older code); it's not
the primary source.

**Scope.** Small — touches `_initialize_execution` and upload drain.

### 3.6 Online-mode drain inconsistency between plain rows and assets

**Both reviews flag this.** Rev-5 §2.8: online mode drains plain-row
`.insert()` but not `.asset_file()`. Defensible (Hatrac two-phase) but
user-visible inconsistency. ML reviewer names it F6 (weekly friction):
"user calls `asset_file()`, queries catalog, sees no row, thinks it
failed."

**Fix.** Not a code change — a user-experience change:

1. Rev-5 §2.1 mode table needs explicit "assets always defer, even in
   online mode" note.
2. `AssetFilePath.__repr__` and `PendingRow.__repr__` should convey
   status: `<AssetFilePath staged; upload deferred until
   upload_execution_outputs()>` vs `<PendingRow uploaded rid=5-ABC>`.
3. Consider an `eager_upload=True` option on `.asset_file()` for the
   rare case where a user wants per-file immediate upload. V2 if ever.

**Scope.** Rev-5 spec edit + 4-hour repr implementation.

### 3.7 ML workflow: `create_execution` ceremony costs 3-4 lines per script

**ML finding F2.** Every script today writes:
```python
wf = ml.lookup_workflow_by_url("https://github.com/me/repo/blob/<sha>/train.py")
config = ExecutionConfiguration(
    datasets=[DatasetSpec(rid="D-001", version="1.0.0")],
    workflow=wf,
    description="Train",
)
exe = ml.create_execution(config)
```

Four nested constructor calls. Three unique Deriva-specific concepts
to hold in head (Workflow, DatasetSpec, ExecutionConfiguration).

**Fix.** Add a kwargs convenience path on `create_execution`:
```python
exe = ml.create_execution(
    datasets=["D-001"],                        # or [DatasetSpec(...)]
    workflow="https://.../train.py",           # or Workflow(...)
    description="Train",
)
```

Strings autoresolve to the existing objects. Keep the explicit
`ExecutionConfiguration(...)` form for hydra-zen pipelines and
reproducibility-critical paths. This saves 3 lines per new script and
reduces Deriva-specific concepts visible at the call site from 3 to 0
in the common case.

**Scope.** 1 day; adds a kwargs constructor to `ExecutionConfiguration`
and a dispatch path in `create_execution`.

### 3.8 ML workflow: no metric / param logging primitive

**ML finding F5 (weekly) and top-5 recommendation #2.** Practitioners
expect:
```python
exe.log_metric("val_loss", 0.23, step=10)
exe.log_param("lr", 0.001)
```

Today's DerivaML handles these via features (for annotations) or by
creating a user-defined table. Rev 5 adds `exe.table("X").insert(...)`
which is semantically correct but expects the user to have provisioned
a `Metric` table with the right schema.

**Fix (not rev-5-blocking, but worth scoping).** Provision canonical
tables automatically:
- `Execution_Metric(Execution, Name, Value, Step, Timestamp)`
- `Execution_Param(Execution, Name, Value)`

Surface as:
```python
exe.log_metric(name, value, step=None)   # → exe.table("Execution_Metric").insert(...)
exe.log_param(name, value)               # → exe.table("Execution_Param").insert(...)
```

Implementation is just sugar over the new handle API. MLflow-style
logging becomes 1 line per call with schema validation and full
provenance, for free.

This is the single biggest "MLflow expectations unmet" gap. Consider
shipping as part of V1 or immediately after.

**Scope.** 1-2 days for the canonical tables + sugar methods +
schema-migration note for catalogs that don't yet have them.

### 3.9 ML workflow: upload-after-context-manager is a footgun

**ML finding F7 (weekly).** Today's `execution.py` docstring says in
ALL CAPS: "This method must be called AFTER exiting the context
manager, not inside it." That's a warning that the shape is wrong.

**Fix.** Change the default to auto-upload on clean exit:
```python
with exe.execute() as e:
    ...
# online mode + clean exit → auto-upload happens here
# failed exit or offline mode → no auto-upload
```

Let advanced users opt out: `with exe.execute(auto_upload=False): ...`.

**Scope.** Half day; modify `__exit__` to call `upload_execution_outputs`
conditionally.

## 4. Technical debt (V2+)

Findings that don't block rev 5 but should be explicitly tracked in a
V2+ backlog. See the companion `library-refactoring-recommendations.md`
for detail — this list is the pointer to issue-level work.

- **T.1** — Three uses of "pending" (method, table, status). Rename
  `pending_upload` status value in V2.
- **T.2** — `AssetFilePath` as `Path` subclass is fragile. Replace
  with composition in V2.
- **T.3** — `AssetRecord` / `RowRecord` parallel hierarchies. Merge
  under a single base in V2.
- **T.4** — `AssetRID` back-compat dataclass. Deletion candidate once
  tests don't use it.
- **T.5** — Shim accumulation. Need an explicit deprecation calendar.
- **T.6** — `execution_rids()` file-tree scan lingers in `dataset/upload.py`.
  Delete once rev 5 is shipped.
- **T.7** — `exe.add_files` has unclear semantic overlap with
  `exe.asset_file_path` / new `exe.table().asset_file()`. Consolidate.
- **T.8** — `exe.create_dataset` lives on Execution but arguably
  belongs on `ml` with an execution kwarg. Mixing
  data-model-management with execution-lifecycle.
- **T.9** — The existing `Status` enum should absorb the rev-5
  additions (see Blocking 2.1).

## 5. Stylistic issues

Low-cost fixes worth doing as part of rev 5 implementation:

- **S.1** — `gc_executions` is the only `gc_` prefix. Align to
  `clean_executions` or `clear_executions`.
- **S.2** — `DerivaMLOfflineError` needs adding under
  `DerivaMLConfigurationError` in the exception hierarchy.
- **S.3** — `handle.pending()` reads like a property. Consider
  `handle.list_pending()` to match `ml.list_executions`.
- **S.4** — `handle.rid` (lazy, may network) vs `handle.rid_or_none`
  (pure read) — consider adding the latter for scripts that want to
  inspect without triggering leases.

## 6. Cross-cutting recommendations

### 6.1 One state-machine module

All status-carrying things (catalog `Execution.Status`, SQLite
`executions.status`, SQLite `pending_rows.status`) should share a
single module with enums, transition tables, and helper functions.
Covers the Blocking-2.1 finding and prevents future drift.

### 6.2 A `RowHandle` Protocol, not a hierarchy

Structural typing over inheritance. `PendingRow` and `AssetFilePath`
conform to `.rid / .status / .metadata / .error`. Don't try to unify
via subclassing.

### 6.3 Two-tier table entry points

- `ml.table(name)` — schema introspection, read-only helpers
- `exe.table(name)` — execution-scoped writes (inserts, asset files)

Both return `TableHandle` / `AssetTableHandle`; the execution-scoped
one allows write methods. See Important 3.2.

### 6.4 SQLite-as-truth means Python objects are thin views

`Execution.status`, `Execution.stop_time`, etc. should all be
read-through properties. In-memory Python state is a cache, not the
authority. See Blocking 2.4.

### 6.5 Explicit deprecation calendar

Shims listed in rev-5 §2.10 each get a "remove in" version.
`working_data` → `workspace` shim (already in code) gets scheduled
too. Rev-5 spec should include this as §2.10.X.

### 6.6 ML-ergonomics features to ship early

The ML review's top-5 list suggests three changes that could ship with
or shortly after rev 5 and meaningfully improve user experience:

1. **`create_execution` kwargs form** (§3.7 above).
2. **`log_metric` / `log_param` primitives** (§3.8 above).
3. **Auto-upload on context-manager exit** (§3.9 above).

If scheduling permits, consider an `ml_run` decorator (ML
recommendation #4) for one-stop-shop script scaffolding:
```python
@ml_run(datasets=["D-001"], description="Train seg model")
def main(exe):
    ...
```

Decorator handles workflow resolution, execution creation, context
management, upload. This is the MLflow-autologging / W&B-`wandb.init`
equivalent — users expect it.

## 7. Updates to the rev-5 spec (to land as rev 6)

The following findings require changes to the rev-5 spec directly:

1. §2.1 — explicit statement that `create_execution` requires online
   mode; error message text (Blocking 2.3).
2. §2.1 — mode comparison table must explicitly call out that assets
   always defer (Important 3.6).
3. §2.2 — explicit statement that `Execution` Python objects are thin
   views over SQLite, don't cache status (Blocking 2.4).
4. §2.3.1 — reference a single state-machine module for
   `executions.status` values; cross-reference to catalog
   `Execution.Status` mapping (Blocking 2.1).
5. §2.5 — add `ml.table(name)` as schema-introspection sibling
   (Important 3.2).
6. §2.7 — unify `retry_failed` into `upload_execution_outputs(retry_failed=True)`
   (Important 3.4).
7. §2.9 — update error-handling section to reflect the above.
8. §2.10 — drop the shim that conflates `find_executions` with
   `list_executions` (Important 3.3); add explicit deprecation
   calendar (§6.5 above).
9. §6 — add the `RowHandle` Protocol recommendation; add the
   state-machine recommendation.
10. §7 — add `auto_upload` on context-manager exit as planned for V1
    or V2 depending on schedule.
11. §7 — document what happens if workspace SQLite is missing (fresh
    install, or user deleted it): fall back to the disk
    `configuration.json` once to rebuild a minimal registry row.

## 8. Summary

**Does the rev-5 direction hold up?** Yes. The ML-practitioner review
is positive about the one-verb `.insert()` and the offline-mode
story; the SE review is positive about the SQLite-as-truth inversion.
Neither review suggested reversing any rev-5 decision.

**Must-fix blocking items** (for rev 5 to merge safely):
1. Unify status vocabularies (Blocking 2.1)
2. Slim `Execution` before adding to it (Blocking 2.2)
3. Document online-requirement of `create_execution` (Blocking 2.3)
4. `Execution.status` as SQLite read-through (Blocking 2.4)

**High-value ML improvements that could ship with rev 5 or immediately
after** (each reduces lines-per-script or eliminates a common footgun):
1. `create_execution` kwargs form
2. `log_metric` / `log_param` primitives
3. Auto-upload on context-manager exit

**Tech debt to track explicitly** (companion document):
- Handle-type consolidation, shim removal calendar, `AssetFilePath`
  `Path` subclass fragility, file-tree scan lingering in upload code,
  `exe.add_files` / `exe.create_dataset` responsibility smears.

The rev-6 update to the stage-and-upload spec addresses items that
affect the new API; the library-refactoring-recommendations document
tracks items that affect existing code not in the rev-5 scope.
