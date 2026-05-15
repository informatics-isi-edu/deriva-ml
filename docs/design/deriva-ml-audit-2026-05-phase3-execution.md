# deriva-ml audit 2026-05 — Phase 3: execution/

Reviewed `/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/execution/`
(10 230 LoC across 23 Python files) and the corresponding test suite
`/Users/carl/GitHub/DerivaML/deriva-ml/tests/execution/` (8 883 LoC,
29 test files, ~363 tests) at the tip of
`fix/catalog-manager-state-guards` (HEAD `4442f82`). Cross-workspace
references were grepped against
`/Users/carl/GitHub/DerivaML/{deriva-mcp,deriva-mcp-core,deriva-ml-mcp,deriva-ml-model-template,deriva-skills,deriva-ml-skills}/`.

The Phase 1 audit flagged `execution/` as the worst-offender candidate
("11 260 LoC, 24 files" — "Largest, mostly load-bearing") and deferred
it. The deferral was correct: nothing in `execution/` is a façade over
a deleted upstream class the way `dataset/` had `CatalogGraph` as a
load-bearing relic. The dragging tail here is different — **the
subsystem carries two parallel halves of an unfinished cutover**.
The bag-commit pipeline (`bag_commit.py`, `manifest_lease.py`,
`AssetManifest`) is the live production path. Alongside it, the
SQLite-pending-rows / RID-lease / state-store machinery built for
the upload-engine cutover (per spec `2026-04-18-sqlite-execution-state-design.md`)
lives at full size with substantial test coverage **but no production
writer** for the `pending_rows` table. The `_engine_harness.py` module
is a doc-only file documenting that the engine path is "Phase 2."
Phase 2 is the bag commit; the engine never shipped. The lease
machinery has two parallel implementations — `lease_orchestrator`
(SQLite-pending-row based, used only by tests) and `manifest_lease`
(manifest-based, used by production).

## Executive summary

Overall posture: **two-thirds load-bearing, one-third unfinished cutover
with a parallel test-only API surface.** The execution lifecycle
(`Execution`, `state_machine`, `state_store.executions`,
`execution_record`, `execution_snapshot`) is healthy and correct. The
bag-commit upload path is clean. But the SQLite pending-rows /
directory-rules / RID-lease machinery — ~1 800 LoC of `state_store.py`,
all of `lease_orchestrator.py`, plus 1 000+ LoC of tests — is
*production-dead*: no code in `src/` inserts a row into
`execution_state__pending_rows`. Production writes the asset manifest
instead. The lifecycle properties (`status`, `start_time`, `stop_time`,
`error`) read through SQLite's `executions` table; nothing else does.

Top themes ranked by impact:

1. **The SQLite pending-rows / directory-rules / engine machinery is
   production-dead.** `state_store.py` defines three tables
   (`executions`, `pending_rows`, `directory_rules`); only the first
   is written by production code. Grep across `src/` for
   `insert_pending_row`, `insert_directory_rule`,
   `mark_pending_leasing`, `revert_pending_leasing`,
   `finalize_pending_lease`, `list_pending_rows`,
   `list_leasing_rows`, `pending_summary_rows` — every hit is inside
   `state_store.py` itself, `lease_orchestrator.py`, the `executions`
   bookkeeping in `Execution.__init__`, or tests. The
   `pending_summary` machinery is the one exception that *reads*
   pending_rows (from the test-only inserters). This is the largest
   single LoC liability in `execution/`. `_engine_harness.py` is a
   12-line module that exists solely to document this state.

2. **Two parallel lease implementations.** `manifest_lease.py`
   (production) leases RIDs against the asset manifest;
   `lease_orchestrator.py` (test-only in prod, ~248 LoC of orchestrator
   + ~366 LoC of tests) leases against `pending_rows`. The two phases
   are conceptually identical (`generate_lease_token` → `post_lease_batch`
   → finalize/revert) but implemented twice with diverged failure
   semantics. `reconcile_pending_leases` is the only function in
   `lease_orchestrator.py` with production callers (2: `core/base.py:367`,
   `core/mixins/execution.py:481`); both fire during `_open_for_use`
   and find zero rows to reconcile because no production caller
   transitions a row to `leasing`.

3. **`Execution._flush_staged_features` is retained-by-docstring but
   has no callers.** `execution/execution.py:1569` carries a 110-LoC
   `_flush_staged_features` whose docstring says "retained only for
   the few remaining direct-flush callers (tests)." Grep across `src/`
   and `tests/` shows zero callers other than the docstring itself and
   comments inside other functions describing what the legacy version
   did. The bag-commit path
   (`bag_commit.py::_add_staged_feature_rows_to_bag`) carries the
   feature rows inline and is the only production path.

4. **`workflow.py::setup_url_checksum` is a `@model_validator` whose
   docstring describes a `Workflow.create_workflow(...)` factory that
   doesn't exist.** `workflow.py:382-469` is decorated `@model_validator(mode="after")`
   but its docstring example shows `>>> workflow = Workflow.create_workflow(...)`.
   `Workflow` has no `create_workflow` method anywhere in the
   workspace (the canonical entry point is `DerivaML.create_workflow`
   in `core/mixins/workflow.py:361`). The Args section is empty.
   This docstring will mislead anyone reading the public Workflow API.

5. **`base_config.py` types `datasets`/`assets` as `Any`.** The
   `BaseConfig` dataclass at `base_config.py:115-116` declares
   `datasets: Any = None` and `assets: Any = None`. The docstring at
   `base_config.py:95-96` describes the runtime types
   (`list[DatasetSpec]`, `list[str]`) but the class declares neither.
   The comment justifies it ("OmegaConf compatibility") but
   `ExecutionConfiguration` solves the same problem via a
   `@field_validator("assets", mode="before")` coercer
   (`execution_configuration.py:87-115`). The two configs diverge on
   how they handle hydra-zen's structured-config tax.

Worst-offending modules within `execution/`:

1. **`state_store.py`** (1 072 LoC). Defines three tables; only one
   is written by production. The pending_rows / directory_rules
   surface is ~700 LoC of unused-by-prod CRUD + 250 LoC of accurate
   tests. Cleanest deletion target in `execution/`.

2. **`execution.py`** (2 525 LoC). Healthy core (lifecycle, asset
   register, bag upload) wrapped around three dead-or-vestigial
   methods: `_flush_staged_features` (110 LoC),
   `_from_registry` classmethod (40 LoC, only called by
   `resume_execution` which now has the full reconstruction logic
   inline), and `upload_assets` (35 LoC, 1 callsite in tests, 0 in
   `src/`).

3. **`lease_orchestrator.py`** (248 LoC). `acquire_leases_for_execution`
   has zero production callers; `reconcile_pending_leases` runs but
   finds no work because (1) is also production-dead. The whole
   module is bookkeeping for an unshipped path.

4. **`workflow.py`** (786 LoC). Healthy intent but burdened by the
   misleading `setup_url_checksum` docstring (theme 4), three optional
   imports for IPython/Jupyter that duplicate the same machinery
   `find_caller.py` already implements (`workflow.py:39-72` vs.
   `find_caller.py:1-35`), and a static `get_dynamic_version` that
   imports `setuptools_scm` lazily in the call path — fine, but
   wrapped in 18 lines of historical commentary justifying the
   removal of a long-deleted env-var mutation.

---

## Subsystem inventory

| File | LoC | Posture |
|---|---:|---|
| `__init__.py` | 99 | Public surface; explicit `__all__` with lazy `Execution` import. **Healthy.** |
| `_engine_harness.py` | 12 | Doc-only module about the unshipped engine. **Dead.** |
| `bag_commit.py` | 746 | The production upload path. `build_execution_bag` → `BagCatalogLoader`. **Healthy.** |
| `base_config.py` | 698 | `BaseConfig`, `notebook_config`, `run_notebook`, `load_configs`. Two top-level docstrings (lines 1-39, 1-39) overlap. `datasets`/`assets` typed `Any`. |
| `dataset_collection.py` | 93 | RID-keyed mapping + iterable wrapper. **Healthy** but violates `Mapping.__iter__` contract (theme 6.5). |
| `environment.py` | 290 | `get_execution_environment()` for provenance. Carries two dead helpers `localeconv` / `locale_module`. |
| `execution.py` | 2 525 | God module. Lifecycle is clean. `_flush_staged_features` (1 569-1 679) is dead. `_from_registry` (361-401) is vestigial. |
| `execution_configuration.py` | 141 | Pydantic config. `assets` field has a `@field_validator` coercer. **Healthy.** |
| `execution_record.py` | 646 | Live catalog-bound execution row. `list_execution_children` / `parents` with `_visited` recursion guard (same fig-leaf pattern Phase 2 §1.6 flagged). |
| `execution_snapshot.py` | 248 | Frozen Pydantic value object. **Healthy.** |
| `find_caller.py` | 312 | Module/notebook-path resolution. **Healthy** but duplicates 3 optional-import blocks from `workflow.py` (theme 4.2). |
| `lease_orchestrator.py` | 248 | `acquire_leases_for_execution` production-dead. `reconcile_pending_leases` production-live but finds nothing. |
| `lineage.py` | 210 | Pydantic models for `lookup_lineage`. **Healthy.** |
| `manifest_lease.py` | 64 | Production lease path against the asset manifest. **Healthy.** |
| `model_protocol.py` | 175 | `DerivaMLModel` protocol — consumed by `deriva-ml-model-template`. **Healthy.** |
| `multirun_config.py` | 153 | Multirun spec registry. **Healthy.** |
| `pending_summary.py` | 200 | `PendingSummary` rendering. Consumes test-inserted pending_rows. **Healthy** (but depends on theme 1's dead writer surface). |
| `rid_lease.py` | 146 | `post_lease_batch` + `_validate_pending_asset_leases`. **Healthy.** |
| `runner.py` | 698 | `run_model` + multirun. Global `_multirun_state` is the only mutable global in `execution/`. |
| `state_machine.py` | 616 | Status transitions. **Healthy** but `validate_transition` example claims "returns None"; the function returns nothing and raises. |
| `state_store.py` | 1 072 | Executions table is load-bearing. Pending-rows + directory-rules tables (~700 LoC) are test-only writers. |
| `upload_report.py` | 52 | Return type for `upload_pending`. **Healthy.** |
| `workflow.py` | 786 | `Workflow` Pydantic model + script-URL resolution. `setup_url_checksum` docstring is misleading. |

Cross-module dependencies worth naming:

- `Execution.__init__` (`execution.py:165-359`) reads
  `workspace.execution_state_store()` and writes the `executions`
  row. That's the single production writer to `state_store`.
- `bag_commit.py` is the only consumer of `manifest_lease.py`. Every
  asset RID assignment in production flows through this pair.
- `lease_orchestrator.py` and `manifest_lease.py` both import
  `rid_lease.generate_lease_token` and `rid_lease.post_lease_batch`
  — those two helpers are the only shared lease primitives.
- `runner.py`'s `_multirun_state` global is read by every `run_model`
  invocation and written by `_create_parent_execution`. Tests must
  call `reset_multirun_state()` to avoid cross-test pollution.
- `execution.py` imports nothing from `lease_orchestrator.py` —
  the production lease path is entirely through `bag_commit.py` →
  `manifest_lease.py`.

---

## Lens 1 — Dead code

### 1.1 `_engine_harness.py` is a doc-only file with no contents

`_engine_harness.py:1-12` — twelve lines describing why the upload
engine is "Phase 2." The Phase 2 it describes never landed; the
production path is the bag-commit pipeline. No production code
imports the module; grep confirms zero references in `src/` or
`tests/`.

**Fix:** delete the file. The doc content, if anyone cares,
belongs in a one-line note inside `state_store.py` (or, more
realistically, in the obsolete spec document that already
describes this design — `2026-04-18-sqlite-execution-state-design.md`).
**Risk: trivial. LoC: −12.** **Severity: low** but symbolically
high — it advertises the dead surface.

### 1.2 `Execution._flush_staged_features` has no callers

`execution.py:1569-1679` — 110-LoC method retained per docstring
("retained only for the few remaining direct-flush callers (tests)").
Reality check:

```
$ grep -rn "_flush_staged_features\b" src/ tests/
src/deriva_ml/execution/execution.py:1472:  # comment about legacy path
src/deriva_ml/execution/execution.py:1569:  # the definition itself
src/deriva_ml/execution/execution.py:1950:  # docstring on _manifest_store
src/deriva_ml/local_db/manifest_store.py:55:  # docstring reference
src/deriva_ml/local_db/manifest_store.py:658: # docstring reference
src/deriva_ml/execution/bag_commit.py:405:    # comment "same as _flush..."
src/deriva_ml/execution/bag_commit.py:430:    # comment "same as _flush..."
tests/execution/test_staged_features.py:150: # comment in a test docstring
tests/execution/test_staged_features.py:491: # comment in a test docstring
```

Every hit is a comment or docstring; nothing actually *calls*
`_flush_staged_features`. The bag-commit path is the canonical
flusher. The method's `try/except` swallows asset-rewriting bugs
that the bag path treats correctly via `_rewrite_fks`. Risk of
keeping it: a copy-paste from this method into a maintenance fix
would reintroduce the legacy per-table-failure semantics the
bag-commit path replaced.

**Fix:** delete `_flush_staged_features` and update the four
docstring/comment references to say "see `bag_commit._add_staged_feature_rows_to_bag`."
**Risk: low** (no production callers). **LoC: −110 + a few
comment touch-ups.** **Severity: medium** — it confuses the
maintenance picture; the docstring lies about retention purpose.

### 1.3 `Execution._from_registry` is a vestigial classmethod with no callers

`execution.py:361-401` — 40-LoC classmethod whose docstring says
"Distinct from create_execution: does NOT contact the catalog and
does NOT POST a new row. Called by ml.resume_execution.
Temporary implementation for Group D — Group E replaces the body
to wire up read-through lifecycle properties."

Group E has shipped — the read-through properties (`status`,
`start_time`, `stop_time`, `error`) are at
`execution.py:636-779`. `resume_execution` lives in
`core/mixins/execution.py:444-500` and constructs an `Execution`
the canonical way (`reload=execution_rid`); `_from_registry` is
not called.

**Fix:** delete the method. **Risk: low** (no callers).
**LoC: −40.** **Severity: low.**

### 1.4 `Execution.upload_assets` is a dataset-style upload with no production callers

`execution.py:1264-1299` — 35-LoC public method that scans a
directory and calls `upload_directory(self._model, assets_dir)`.
The public flow is `asset_file_path()` → `upload_execution_outputs()`;
`upload_assets()` is a legacy "upload this directory directly"
shortcut. Grep across `src/`: zero callers. Tests:
`test_execution.py::test_upload_assets_invalid_directory`
(line 1362) is the only caller, and it asserts the failure path.

**Fix:** delete the method. The single negative test deletes with
it. **Risk: low.** **LoC: −40.** **Severity: low.**

### 1.5 `state_store.PendingRowStatus`, `DirectoryRuleStatus`, and all pending-row CRUD have no production writers

`state_store.py:75-99` defines `PendingRowStatus` (6 members) and
`DirectoryRuleStatus` (2 members). The full CRUD surface:

- `insert_pending_row` (line 417) — production callers: zero.
- `update_pending_row` / `update_pending_rows_batch` (line 475/492)
  — production callers: zero.
- `list_pending_rows` (line 539) — production callers: zero.
- `mark_pending_leasing*` / `finalize_pending_leases*` /
  `revert_pending_leasing` (lines 795-927) — production callers:
  zero.
- `list_leasing_rows` (line 929) — called by `reconcile_pending_leases`
  (`lease_orchestrator.py:187`) which is production-live but
  finds nothing because nobody writes a `leasing` row.
- `insert_directory_rule` / `update_directory_rule` /
  `list_directory_rules` (lines 959-1049) — production callers:
  zero.
- `count_pending_rows` (line 644) — production callers: 1
  (`core/base.py:561`, the schema-refresh guard); always returns 0
  in production.
- `pending_summary_rows` (line 670) — called by
  `Execution.pending_summary` (`execution.py:2467-2496`) which
  itself has only test callers in `src/`.

Test callers: extensive
(`test_state_store.py`, `test_lease_orchestrator.py`,
`test_pending_summary.py`, `test_offline_init.py`,
`test_lookup_lineage_unit.py`).

The pending_rows / directory_rules architecture is the
"upload-engine, sqlite-handle-based" design spelled out in
`2026-04-18-sqlite-execution-state-design.md`. That design was
superseded by the bag-commit path before the
pending_rows-writing handle API shipped (see
`_engine_harness.py:1-12`). The schemas, the enum, and the CRUD
methods landed; the writer never did.

**Fix:** four options, in order of preference:

(a) **Delete the dead surface.** Remove `PendingRowStatus`,
`DirectoryRuleStatus`, `insert_pending_row`, `update_pending_row`,
`list_pending_rows`, `mark_pending_leasing*`,
`finalize_pending_leases*`, `revert_pending_leasing`,
`insert_directory_rule`, `update_directory_rule`,
`list_directory_rules`, the `pending_rows` and `directory_rules`
Table definitions. Keep `executions`, `count_pending_rows`
(returns 0), `pending_summary_rows` (returns empty).
**LoC: −600 to −700.** **Risk: medium** — the test suite covering
this surface goes with it (~700 LoC of tests).

(b) **Ship the writer.** Wire `Execution.add_features` and the
plain-row write path through `state_store.insert_pending_row`
to give pending-row durability for free. Requires a bag-commit
refactor to read from `pending_rows` instead of from the asset
manifest. **High risk, high reward.** Out of audit scope.

(c) **Document the dead surface as an intentional reserve API.**
Add a `state_store.py` module docstring section explaining that
the pending_rows machinery is a future-work hook, kept tested to
preserve invariants while production uses the asset manifest.
This is the do-nothing-now answer.

(d) **Carve the dead surface into a separate `state_store_legacy.py`
or `state_store_engine.py`.** Keeps the surface area visible
without polluting the production file. Mechanical, low risk.

**Recommended:** (a) for the pending_rows write surface but keep
`pending_summary_rows` and `count_pending_rows` (they degrade
gracefully and would be needed by option (b) if it ever ships).
Alternative: (c) plus a redesign in a future spec. The current
state — full implementation + full test coverage + no production
writer — is the worst of every option.

**Severity: high** (largest LoC liability in `execution/`;
biggest divergence between docs and reality).

### 1.6 `lease_orchestrator.acquire_leases_for_execution` is production-dead

`lease_orchestrator.py:29-147` — entry point per the docstring:
"Called by handle.rid property and by the upload-engine drain."
Neither caller exists: there is no `handle.rid` property and the
upload-engine drain was replaced by `bag_commit`.

**Fix:** delete `acquire_leases_for_execution`. Keep
`reconcile_pending_leases` (it has production callers, even if
they find no work today). The deletion takes ~120 LoC of source
and a parallel ~250 LoC of `test_lease_orchestrator.py` (8 of 9
tests exercise the acquire path). **Risk: medium** (cross-cuts
with §1.5; ship together). **LoC: −120 src + −250 tests.**
**Severity: medium.**

### 1.7 `environment.localeconv` and `environment.locale_module` are dead helpers

`environment.py:241-291` defines `localeconv()` and `locale_module()`.
Neither is exported from `__init__.py`. Neither is called by
`get_execution_environment()` (line 49-56) — the dict returned to
the catalog does not include locale info. Grep across the
workspace shows zero callers.

**Fix:** delete both. **Risk: trivial. LoC: −50.** **Severity: low.**

### 1.8 `Execution.__str__` references `self.asset_paths` and `self.configuration` after `_from_registry`

`execution.py:2254-2263` builds a string from `self.asset_paths`,
`self.configuration`, `self.workflow_rid`. For an Execution
constructed via `_from_registry` (theme 1.3) `self.configuration`
is `None` and `self.workflow_rid` is `None`. Since
`_from_registry` itself is dead, this isn't a runtime bug today,
but if anything ever calls `_from_registry`, `str(exe)` includes
`configuration: None`.

**Fix:** subsumed by §1.3. **Severity: low.**

### 1.9 `execution.py:99-107` defines an IPython display fallback that's no longer used

The `from IPython.display import Markdown, display` block with the
fallback `def display(s): print(s)` and `def Markdown(s): return s`
is defined at module load but never invoked: grep across
`execution/execution.py` shows zero call sites for `display` or
`Markdown`.

**Fix:** delete the optional-import block. **Risk: trivial.
LoC: −10.** **Severity: low.**

### 1.10 Stale `Group D / Group E / Task F2 / Task C3 / WI2` references in docstrings

Multiple module docstrings reference internal task IDs from the
SQLite-execution-state design and the upload-engine cutover:

- `state_machine.py:207` — "NotImplementedError: Online-mode path
  is implemented in Task C3." Online mode is implemented.
- `execution.py:368` — "Temporary implementation for Group D —
  Group E replaces the body…". Group E shipped.
- `rid_lease.py:5-7` — "Pure helpers — no SQLite awareness here.
  The acquire_leases_for_pending function in state_store
  composition (Task F2) wires these into the two-phase SQLite
  protocol." Task F2 shipped (as lease_orchestrator).
- `upload_report.py:5-7` — "lives in its own module (rather than
  alongside the now-retired ``upload_engine.py``) so the type can
  survive independent of the engine's life cycle. Post-WI2 the
  engine is gone."

These are useful archaeology but misleading to maintainers —
"implemented in Task C3" reads as "not implemented yet."

**Fix:** rewrite each to describe the current state without
referencing task IDs. **LoC: ±0 doc-only.** **Severity: low.**

---

## Lens 2 — deriva-py interface usage

### 2.1 `state_machine.flush_pending_sync` and `transition` use the same `pb.schemas[...].tables[...].update(body)` shape — clean

`state_machine.py:284-302` (in `transition`) and
`state_machine.py:391-402` (in `flush_pending_sync`) both use the
datapath API for the catalog PUT, with identical inline comments
about why raw `catalog.put` doesn't work. This is correct usage of
deriva-py; the duplication is small enough (2 lines each) that
factoring would be over-engineering. **No fix.**

### 2.2 `state_machine.reconcile_with_catalog` uses raw ERMrest path

`state_machine.py:484` — `catalog.get(f"/entity/deriva-ml:Execution/RID={execution_rid}")`.
This works but is the only raw-URL ERMrest call in `execution/`.
The datapath equivalent (`pb.schemas["deriva-ml"].Execution.filter(...).entities().fetch()`)
matches the pattern used everywhere else. The reconcile call is
read-only and runs at resume time so the performance cost is
negligible, but the inconsistency stands out.

**Fix:** swap to the datapath form. **Risk: low. LoC: ±0.**
**Severity: low.**

### 2.3 `rid_lease.post_lease_batch` uses raw `catalog.post`

`rid_lease.py:79` — `catalog.post("/entity/public:ERMrest_RID_Lease", json=body)`.
This is the correct shape — the lease table is in the `public`
schema (not under any deriva-ml schema) and the body is a list of
`{"ID": token}` dicts, which the datapath API doesn't make easier.
**No fix.**

### 2.4 `execution.py::download_asset` uses `HatracStore` directly instead of `ml.download_asset`

`execution.py:1221-1222` constructs a `HatracStore` and calls
`hs.get_obj(path=asset_url, destfilename=asset_filename.as_posix())`.
This is the canonical way to fetch a Hatrac object; the alternative
would be `ml.catalog`'s aggregated helpers, but those don't exist
for raw Hatrac. **No fix.**

### 2.5 `lease_orchestrator.reconcile_pending_leases` uses raw ERMrest URL with `urllib.quote`

`lease_orchestrator.py:212-215` constructs a filter clause by
hand:

```python
filter_clause = ";".join(f"ID={quote(t, safe='')}" for t in chunk)
path = f"/entity/public:ERMrest_RID_Lease/{filter_clause}"
response = catalog.get(path)
```

The datapath equivalent is `lease_table.filter(lease_table.ID.in_(chunk)).fetch()`
(matching `rid_lease._validate_pending_asset_leases:127`). The
existing pattern in the same subsystem is the datapath form —
`reconcile_pending_leases` is the outlier.

**Fix:** swap to the datapath form. **Risk: low. LoC: −5.**
**Severity: low.**

### 2.6 `execution.py::_update_asset_execution_table` does the right thing with `on_conflict_skip`

`execution.py:1705-1715` uses `path.insert([...], on_conflict_skip=True)`
to handle the case where an asset is registered against the same
execution twice. This is the correct deriva-py idiom; the bag-commit
path matches it via the loader's `match_by_columns`. **No fix;
flagged as well-shaped for the audit record.**

### 2.7 `workflow.py` re-implements optional IPython/Jupyter imports that `find_caller.py` already has

`workflow.py:39-72` defines three fallback blocks for IPython,
jupyter_server, and ipykernel. `find_caller.py:9-32` defines the
same three blocks (more concisely, with `# type: ignore` and
`# pragma: no cover - optional` markers). `_get_python_script`
(`workflow.py:670-674`) delegates to `find_caller._get_calling_module`,
but `_get_notebook_path` (`workflow.py:583-597`) builds its own
session lookup via the optional imports defined at module top.

**Fix:** delete the optional-import block in `workflow.py:39-72`
and route `_get_notebook_path` and `_get_notebook_session`
through helpers in `find_caller.py` (lift them up since both
files have the same need). **Risk: medium** (notebook path is
non-trivial to test without a live kernel). **LoC: −50 net.**
**Severity: low.**

---

## Lens 4 — Inconsistencies / duplication

### 4.1 Two parallel lease implementations (the headline duplication)

`manifest_lease.py:25-65` (production) and
`lease_orchestrator.py:29-147` (production-dead per §1.6) implement
the same two-phase protocol over different state stores.

Both:

1. Collect items needing RIDs.
2. Generate a token per item via `generate_lease_token`.
3. `post_lease_batch(catalog=..., tokens=...)`.
4. Write the assigned RIDs back to the originating state.

Differences:

- **Mark intermediate state.** `lease_orchestrator` writes
  `status='leasing'` to SQLite *before* the POST so a crash
  between commit and POST is recoverable.
  `manifest_lease` writes nothing intermediate — manifest entries
  go from "no RID" directly to "leased RID" inside one POST
  round-trip; a crash mid-POST leaves entries without RIDs and
  the manifest's idempotent re-leasing handles it on retry.

- **Failure handling.** `lease_orchestrator` reverts marked rows
  on POST failure (`revert_pending_leasing`); `manifest_lease`
  simply raises.

- **Concurrency.** The SQLite path is built to be safe against
  workspace-wide concurrent acquires (the two-phase lease, the
  `revert_pending_leasing` rollback, the `reconcile_pending_leases`
  startup sweep). The manifest path assumes the asset manifest is
  single-writer.

If the SQLite engine path ever ships, the asset manifest's
single-writer assumption breaks down and the lease logic needs to
be unified. If it doesn't ship (theme 1's status quo), the SQLite
lease orchestrator is dead code.

**Fix:** subsumed by §1.5 / §1.6 — delete the SQLite lease
orchestrator until/unless the engine ships, at which point bring
both lease paths under a shared interface. **Severity: medium**
(silent divergence; tests pin both shapes).

### 4.2 `Execution.execution_start` and `Execution.__enter__` do nearly identical work

`execution.py:1069-1119` (`execution_start`) and
`execution.py:2293-2337` (`__enter__`) both:

1. Skip if `self._dry_run`.
2. Read `current = self.status` from SQLite.
3. Call `transition(... target=Running, extra_fields={"start_time": now})`.

`execution_start` is documented as "non-context-manager
counterpart for code paths that can't use ``with``" — the multirun
parent and `run_notebook` callers. `__enter__` is the
context-manager path. The bodies are 12 lines apart and they
differ only in trivial respects (an unused `self.uploaded_assets = None`
assignment in `execution_start`, an extra `self._logger.info("Start execution...")`).

**Fix:** delegate `__enter__` to `execution_start()` (or vice
versa). **Risk: low** (mechanical). **LoC: −10.** **Severity: low.**

### 4.3 `Execution._initialize_execution` builds the manifest before `__enter__` would

`execution.py:531-634` runs in `Execution.__init__` and calls
`self._bag_commit_upload()` at line 627 *before* the execution
ever enters the `Running` state. This uploads the
`configuration.json`, `uv.lock`, runtime environment, and Hydra
configs as Execution_Metadata. The comment at line 596-597
("Save configuration details and upload (skip in dry_run mode)")
treats this as setup, not output, but the upload pipeline doesn't
distinguish.

Consequence: `create_execution(config)` returns a context manager
whose status is still `Created` but which has already done a
catalog write. If the user never calls `__enter__` (the
test-without-`with` case used in some legacy tests), the metadata
assets are committed against a never-Running execution.

**Fix:** debatable. The current behavior is what the test suite
asserts; changing it would ripple through the multirun harness.
Document the early upload at the API surface
(`create_execution`'s docstring should mention that initial
metadata uploads happen at construction time, not at `__enter__`).
**Risk: low (doc-only).** **LoC: ±0.** **Severity: low** — but
the user-facing surprise is real.

### 4.4 Three places construct `ExecutionConfiguration` from scratch with different argument styles

`execution.py:165-208` (`Execution.__init__` accepts a config
already), `runner.py:539-545` (`run_model` builds one), and
`base_config.py:559-564` (`run_notebook` builds one).

`runner.py:539-545` passes `config_choices` as a kwarg;
`base_config.py:559-564` does not (the `_captured_hydra_output_dir`
goes through a different channel). Tests in `tests/catalog_manager.py`
build `ExecutionConfiguration()` with zero arguments and rely on
the model's defaults.

**Fix:** factor the construction into a helper, or accept that
the three callers genuinely differ. **Severity: low** — the
duplication is shallow and the inputs differ.

### 4.5 `Execution.execution_stop` does the catalog write directly; the state machine does not

`execution.py:1121-1164` calls `self.update_status(ExecutionStatus.Stopped)`
which routes through `transition()`, **then** at line 1162-1164
does a *second* catalog write to update `Duration`:

```python
self._ml_object.pathBuilder().schemas[self._ml_object.ml_schema].Execution.update(
    [{"RID": self.execution_rid, "Duration": duration_str}]
)
```

The state machine's `_catalog_body_for_execution` (state_machine.py:305-348)
explicitly omits `Duration` per its comment ("Duration is computed
elsewhere (in execution_stop) and written directly; don't echo it
here"). So we have two writers to the same catalog row, one
synced through SQLite and one bypassing it. If the first PUT
fails (sync_pending stays True) but the second succeeds, the
catalog has a `Duration` that doesn't match the catalog's `Status`
nor SQLite's `status`. The probability is low but the design is
fragile.

**Fix:** Either add `duration` to SQLite's `executions` table and
let the state machine sync it (cleaner), or fold the Duration
write into the `transition()` `extra_fields` and let
`_catalog_body_for_execution` include it. **Risk: medium** (a
column add ripples). **LoC: −10 + schema change.** **Severity: medium.**

### 4.6 `find_caller.py` and `workflow.py` redundantly detect the Jupyter notebook path

Theme 2.7 covered the optional-import duplication.
`find_caller._get_notebook_path` (`find_caller.py:192-252`) and
`workflow.Workflow._get_notebook_path` (`workflow.py:583-597`) both
walk the running-Jupyter-servers list and query for sessions.
`workflow._get_python_script` (line 670-674) calls
`find_caller._get_calling_module` for everything else but
re-implements the notebook path itself.

**Fix:** consolidate. **Risk: low. LoC: −60.** **Severity: low.**

### 4.7 `DatasetCollection.__iter__` violates the `Mapping` contract

`dataset_collection.py:72-75` overrides `__iter__` to yield
**values**, not keys, with the comment "Mapping's default would
iterate keys; we override to iterate values (bags) because that's
the overwhelmingly common use." This violates the `Mapping`
abstract contract: every code that types `def f(m: Mapping[str, X])`
and calls `for k in m: m[k]` will get `for v in m: m[v]` instead,
indexing a `DatasetBag` into a mapping that expects a `str` key.

This is a deliberate ergonomic choice the docstring documents
clearly, but the class inherits from `Mapping` precisely to
advertise the contract. Downstream type-checkers and any
`isinstance(x, Mapping)` check that then iterates will hit the
bug.

**Fix:** either drop the `Mapping` inheritance and make
`DatasetCollection` a custom value-iterable (with separate
`keys()` / `values()` / `items()`), or restore the Mapping
contract and rename the iteration ergonomics to a separate method
(e.g. `for bag in exe.datasets.values():`). **Risk: medium**
(test suite asserts `for bag in exe.datasets:`). **LoC: ±0 to +10.**
**Severity: medium** — silent contract violation.

### 4.8 `Execution.status` / `error` / `start_time` / `stop_time` are four near-identical read-through properties

`execution.py:636-779` defines four properties, each with
identical boilerplate:

```python
store = self._ml_object.workspace.execution_state_store()
row = store.get_execution(self.execution_rid)
if row is None:
    raise DerivaMLStateInconsistency(...)
return ExecutionStatus(row["status"])  # or row["error"], row["start_time"], row["stop_time"]
```

`start_time` and `stop_time` additionally do the UTC coercion. The
four implementations each have an 18-line docstring; the bodies are
6 lines each.

**Fix:** factor the SQLite read into a single private
`_get_registry_row()` that the four properties call. **Risk: low.
LoC: −40.** **Severity: low** — quality-of-life maintainability.

---

## Lens 5 — Simplification opportunities

### 5.1 Delete the SQLite pending-rows machinery (§1.5)

Net: **~−600 LoC src + ~−700 LoC tests**. Largest single
simplification available in `execution/`. See §1.5 for the option
matrix; recommended path is (a).

### 5.2 Delete `_flush_staged_features` (§1.2)

Net: **−110 LoC** plus comment cleanup.

### 5.3 Delete `_engine_harness.py` (§1.1)

Net: **−12 LoC**. Symbolic but worthwhile.

### 5.4 Delete `_from_registry` (§1.3)

Net: **−40 LoC**.

### 5.5 Delete `Execution.upload_assets` (§1.4)

Net: **−40 LoC src + −10 LoC tests** (one negative test goes).

### 5.6 Delete `lease_orchestrator.acquire_leases_for_execution` (§1.6)

Net: **−120 LoC src + −250 LoC tests**. Bundle with §5.1.

### 5.7 Consolidate the IPython/Jupyter notebook-path detection (§2.7 + §4.6)

Net: **−60 LoC**.

### 5.8 Delete dead `environment.locale*` helpers (§1.7)

Net: **−50 LoC**.

### 5.9 Factor `Execution.status`/`error`/`start_time`/`stop_time` to share one fetcher (§4.8)

Net: **−40 LoC**.

### 5.10 Delegate `Execution.__enter__` to `execution_start()` (§4.2)

Net: **−10 LoC**.

### 5.11 Scrub stale `Task / Group / WI` references from docstrings (§1.10)

Net: **±0 doc-only**, but maintainability win.

---

## Lens 6 — Maintainability

### 6.1 `execution.py` carries three responsibilities

`Execution` class methods cluster into:

- **Lifecycle** (~600 LoC): `__init__`, `__enter__`, `__exit__`,
  `execution_start`, `execution_stop`, `update_status`, `abort`,
  status / error / start_time / stop_time properties,
  `pending_summary`, `upload_outputs`, `_from_registry`.
- **Asset register/upload** (~900 LoC): `asset_file_path`,
  `metrics_file`, `download_asset`, `_bag_commit_upload`,
  `_update_asset_execution_table`, `_set_asset_descriptions`,
  `_clean_folder_contents`, `upload_execution_outputs`,
  `upload_assets`, `_get_manifest`, `_manifest_store`, the
  `_flush_staged_features` legacy.
- **Provenance / nesting / metadata** (~400 LoC):
  `_save_runtime_environment`, `_upload_hydra_config_assets`,
  `_get_metadata_description`, `_initialize_execution`,
  `create_dataset`, `add_files`, `add_features`,
  `add_nested_execution`, `is_nested`, `is_parent`,
  `list_input_datasets`, `list_assets`, `database_catalog`,
  `catalog`, `working_dir`, `_execution_root`, `_asset_root`.

The lifecycle cluster is the bone-deep correct concern of
`Execution`. The asset cluster could conceivably move into
`bag_commit.py` or a sibling `execution_assets.py`. The provenance
cluster is mostly thin sugar over `ExecutionRecord` and
`bag_commit`.

**Severity: medium** (file size; mixed concerns). Phase 4 scope.

### 6.2 `state_store.py` carries an unshipped second architecture

The `executions` table CRUD (~300 LoC) is the production half;
the pending_rows / directory_rules CRUD (~700 LoC) is the
unshipped half. Splitting into `state_store.py` (executions only,
~400 LoC including imports + base) and either deleting or
isolating the pending_rows machinery would make the prod-vs-spec
distinction maintainable. **Severity: high** — see §1.5.

### 6.3 Docstring quality

Sampled the public methods on `Execution`, `ExecutionRecord`,
`Workflow`, `state_machine`, `bag_commit`. The Google-format
skeleton is consistently filled in; `Example:` blocks are
correctly marked `# doctest: +SKIP` for catalog-dependent paths.
**Healthy overall.** Specific issues:

- **`workflow.py:382-414`** — `setup_url_checksum` decorated
  `@model_validator(mode="after")`. The docstring describes a
  "factory" and the example shows `Workflow.create_workflow(...)`
  which doesn't exist. The Args section is empty. The Returns
  section says "Workflow: New workflow instance with detected Git
  information" — the method actually returns `self` after mutating
  it. **Misleading.** Fix: rewrite the docstring as a Pydantic
  model-validator description ("Mutates the freshly-constructed
  Workflow to fill in URL, checksum, and version from the calling
  context").
- **`state_machine.py:141-165`** — `validate_transition`'s
  `Example` says "returns None, no raise" but the function returns
  `None` implicitly (no explicit return). Fine in Python; the
  doctest shape is `validate_transition(...)  # returns None, no raise`
  which is correct but oddly phrased. **Minor.**
- **`state_machine.py:204-207`** — `transition`'s `Raises` section
  says "NotImplementedError: Online-mode path is implemented in
  Task C3." Online mode is fully implemented. **Stale.**
- **`execution.py:1166-1192`** — `download_asset`'s docstring
  mentions the `_asset_table: Any` parameter with the warning
  "Internal — pre-resolved Table object… callers should not rely
  on this; pass through the public ``resolve_rid`` path
  otherwise." Good practice; the leading underscore signals
  internal use but it's still a public method parameter. **Fine.**
- **`bag_commit.py:35-44`** — module docstring describes a
  "progress reporting" section that the code partially implements.
  Per-asset events during staging fire; byte-level streaming
  does not. The docstring explicitly says so ("Byte-level
  streaming progress would require a new hook on `BagCatalogLoader`
  (tracked as deriva-py follow-up)"). **Healthy.**

**LoC: ±0 (doc-only).** **Severity: low.**

### 6.4 `__all__` discipline

`execution/__init__.py:63-99` declares an explicit `__all__` —
the right pattern. Per-module:

- `state_machine.py:39-47` — explicit `__all__`. Good.
- `upload_report.py:52` — explicit `__all__`. Good.
- Every other module lacks `__all__`. Anything imported from those
  modules' namespaces (e.g., `from deriva_ml.execution.workflow
  import logger, get_logger, RequestException`) is reachable.

**Recommendation:** add `__all__` to `bag_commit.py`,
`base_config.py`, `dataset_collection.py`, `environment.py`,
`execution.py`, `execution_configuration.py`,
`execution_record.py`, `execution_snapshot.py`, `find_caller.py`,
`lease_orchestrator.py`, `lineage.py`, `manifest_lease.py`,
`model_protocol.py`, `multirun_config.py`, `pending_summary.py`,
`rid_lease.py`, `runner.py`, `state_store.py`, `workflow.py`.
**LoC: +75.** **Severity: low.**

### 6.5 `DatasetCollection` violates `Mapping.__iter__` contract

Covered under §4.7. The class advertises itself as a `Mapping`
but iterates values. A `Mapping`-typed parameter elsewhere will
double-dispatch incorrectly. **Severity: medium** (silent
correctness gap, not just style).

### 6.6 Logger usage is consistent — good

Every module uses `get_logger(__name__)` from
`deriva_ml.core.logging_config`. No `logging.getLogger(...)`
direct calls in `execution/`. Phase 1 §1.4 flagged the logger
fragmentation across the codebase; the `execution/` subsystem is
already clean.

### 6.7 Global mutable state in `runner._multirun_state`

`runner.py:196` constructs a module-level `MultirunState`
singleton. Tests must call `reset_multirun_state()` to clean it
up; `tests/execution/test_runner.py` does this in 9 of 15 tests.
A single test that forgets to reset leaves cross-test pollution.

The global is unavoidable in the multirun architecture (Hydra's
`--multirun` mode invokes the runner once per job within the same
process). It is correctly fenced — the global is the only place
in `execution/` that needs explicit test isolation.

**Recommendation:** add a `pytest` autouse fixture in
`tests/execution/conftest.py` (currently absent) that resets
`_multirun_state` between tests. **LoC: +10 (test only).**
**Severity: low.**

### 6.8 Naming: `workspace`, `working_dir`, `_working_dir`, `cache_dir`, `_cache_dir`, `execution_root`, `_execution_root`, `_asset_root`

A 100-line block in `Execution.__init__` (`execution.py:226-302`)
juggles eight related path concepts. The naming is consistent
across the codebase (Phase 1 §6.5 already flagged the
inter-module variant) but the local density makes the lifecycle
hard to read.

**Severity: low.**

### 6.9 `Execution.add_features` docstring promises "DerivaMLValidationError" but the production path is staging-only

`execution.py:925-927` documents:

```
Raises:
    DerivaMLValidationError: Records do not share a single feature
        definition.
```

The function does raise correctly (line 951-954). But the
`DerivaMLDataError: SQLite staging write failed` clause is harder
to surface — the SQLite write goes through
`_manifest_store.stage_feature_records` which wraps SQLAlchemy
errors as `DerivaMLException`, not `DerivaMLDataError`. The
docstring is wishful, not accurate.

**Severity: low.**

---

## Lens A — Legacy-user removal

There are **no legacy users**. Items that exist to support them
are liabilities.

### A.1 `_engine_harness.py` — full-file legacy reservation

12 lines describing the future direction the codebase didn't
take. See §1.1. **Delete.**

### A.2 `Execution._flush_staged_features` — "retained for tests" but tests don't call it

Docstring `execution.py:1576-1577` claims test retention. Reality
shows zero test callers. See §1.2. **Delete.**

### A.3 `Execution._from_registry` — "Temporary implementation for Group D"

Group E shipped. See §1.3. **Delete.**

### A.4 `Execution.upload_assets` — predates `asset_file_path` + `upload_execution_outputs`

Documented in `execution.py:1265-1289` as if it's a current
public API. The canonical flow doesn't go through it; users who
call it are running the legacy pre-manifest path. See §1.4.
**Delete.**

### A.5 `state_store.pending_rows`, `directory_rules` and their CRUD — pre-bag-commit architecture

See §1.5. The largest single legacy carryover. **Delete the
production-dead surface.** Keep `pending_summary_rows`,
`count_pending_rows` because the schema-refresh guard still
calls into them and they degrade to zero in the absence of
writers.

### A.6 `lease_orchestrator.acquire_leases_for_execution` — pre-bag-commit lease path

See §1.6. **Delete.** Keep `reconcile_pending_leases` (it has
prod callers, even if they no-op today; the call is cheap).

### A.7 Stale Group/Task references in docstrings

See §1.10. **Rewrite or delete.**

### A.8 The "uploaded_assets" attribute on `Execution`

`execution.py:219` initializes `self.uploaded_assets: dict[str, list[AssetFilePath]] | None = None`,
`upload_execution_outputs` writes it (line 1392, 1399), and
`execution_start` reassigns it to `None` (line 1096) before the
state machine transitions. The third call to
`upload_execution_outputs` in §test_additive_upload_after_uploaded
relies on the cached value to short-circuit. The "no-op return
on `Uploaded` with no pending manifest entries" path
(`execution.py:1378-1381`) reads `self.uploaded_assets or {}`.

This isn't legacy per se — but the comment at
`execution.py:213-218` describing the removal of in-memory
status/start_time/stop_time fields explicitly does not justify
keeping `uploaded_assets` as an instance field. After the bag-
commit refactor, the manifest is authoritative; reading the
manifest's `uploaded` rows is the source of truth.

**Recommendation:** make `uploaded_assets` a computed property
that reads from `self._get_manifest().assets`. The "cached return
value" pattern in additive upload becomes obsolete.
**Risk: medium** (cross-cuts the test suite). **LoC: −20.**
**Severity: low** — but the field is the last in-memory
remnant of the pre-bag-commit lifecycle.

---

## Lens B — Privatization

Cross-workspace grep performed against
`/Users/carl/GitHub/DerivaML/{deriva-mcp,deriva-mcp-core,deriva-ml-mcp,deriva-ml-model-template,deriva-skills,deriva-ml-skills}/`.

### B.1 `ExecutionStateStore` class — internal, currently public via lazy module

`state_store.py:107`. Constructed inside `Workspace.execution_state_store()`
(in `local_db/workspace.py`); not re-exported from
`execution/__init__.py`. External grep: zero hits.

**Recommendation:** rename `ExecutionStateStore` →
`_ExecutionStateStore` once the pending-row surface deletions
(§1.5) land. The "executions" CRUD remains its sole public role;
that's a `Workspace` internal helper.
**Severity: low.**

### B.2 `PendingRowStatus`, `DirectoryRuleStatus`, `EXECUTIONS_TABLE`, `PENDING_ROWS_TABLE`, `DIRECTORY_RULES_TABLE` — table-name constants

`state_store.py:75-104`. All module-level. External grep:
zero hits. Module table names should be `_PENDING_ROWS_TABLE`
etc. The enums are part of the dead pending-rows surface.

**Recommendation:** delete the enums (§1.5). Privatize the table
names with leading underscore. **LoC: ±0.** **Severity: low.**

### B.3 `lease_orchestrator.acquire_leases_for_execution` — public but tests-only

`lease_orchestrator.py:29`. External grep: zero. **Already
production-dead — delete (§1.6).**

### B.4 `lease_orchestrator.reconcile_pending_leases` — public, used by core

`lease_orchestrator.py:150`. Called from
`core/base.py:367` and `core/mixins/execution.py:481`. Should
remain public.

### B.5 `rid_lease.generate_lease_token`, `post_lease_batch` — public, used cross-module

`rid_lease.py:29` (`generate_lease_token`) and
`rid_lease.py:44` (`post_lease_batch`). Both used by
`manifest_lease.py`, `bag_commit.py`, `lease_orchestrator.py`,
and tests. External grep: zero. **Keep public** within
`execution/` (cross-module internal API); not exposed from
`__init__.py`.

### B.6 `rid_lease._validate_pending_asset_leases` — already underscore-private

`rid_lease.py:86`. Already `_`-prefixed. **Correct.**

### B.7 `rid_lease.PENDING_ROWS_LEASE_CHUNK` — public constant

`rid_lease.py:26`. Imported by `lease_orchestrator.py:14`. External
grep: zero. Should be `_PENDING_ROWS_LEASE_CHUNK` once
`lease_orchestrator` either consumes it as a module-private import
or `lease_orchestrator` is deleted (§1.6).
**Severity: low.**

### B.8 `manifest_lease.lease_manifest_pending_assets` — public, used by bag_commit

`manifest_lease.py:25`. Called from `bag_commit.py:128`. Tests
hit it directly. External grep: zero. **Keep public** within
`execution/` (it's the canonical production lease primitive); not
in `__init__.py`. Future plan: surface as `_lease_manifest_pending_assets`
and bury bag_commit's import — currently public is harmless.

### B.9 `state_machine.transition`, `validate_transition`,
`flush_pending_sync`, `reconcile_with_catalog`, `create_catalog_execution` —
public, exported via `__all__`

`state_machine.py:39-47`. Used by `Execution.update_status`,
`Execution.__enter__`/`__exit__`, `core/mixins/execution.py` for
the resume path, and `execution_snapshot.py`. Tests exercise them
directly. External grep: zero. **Keep public** as the
state-machine's contract.

### B.10 `state_machine._catalog_body_for_execution`,
`state_machine._DISAGREEMENT_RULES` — already private

`state_machine.py:305, 417`. Already `_`-prefixed. **Correct.**

### B.11 `state_machine.InvalidTransitionError` — public exception class

`state_machine.py:50-59`. Exported via `__all__`. External grep:
zero hits (the legacy `deriva-mcp` references it via a comment in
`deriva-ml-mcp/docs/`), but it's a `DerivaMLException` subclass
in the publicly-documented `Raises:` clause of
`Execution.update_status`. **Keep public.**

### B.12 `execution.Execution._initialize_execution`,
`Execution._save_runtime_environment`,
`Execution._upload_hydra_config_assets`,
`Execution._get_metadata_description`,
`Execution._set_asset_descriptions`,
`Execution._bag_commit_upload`,
`Execution._clean_folder_contents`,
`Execution._update_asset_execution_table`,
`Execution._get_manifest`,
`Execution._manifest_store`,
`Execution._execution_root`,
`Execution._asset_root` — already private

All `_`-prefixed. **Correct.**

### B.13 `Execution.execute()` — public, returns self

`execution.py:1985-2002`. Exists per spec §2.8 so usage reads
`with exe.execute() as e:`. Tested. External grep: every test
file under `deriva-ml/tests/` and `deriva-ml-mcp/tests/` uses it.
**Keep public.** Slight design oddity: it's a no-op that returns
self; consumers could just `with exe as e:`. Removing it would
break MCP code and skill snippets.

### B.14 `find_caller._get_calling_module`, `_get_notebook_path`, `_top_user_frame`, etc.

`find_caller.py:35-313`. All `_`-prefixed (private). Used by
`workflow.py:30` and `runner.py`. External grep: zero. **Correct.**

### B.15 `workflow.Workflow._check_writable_catalog`,
`_update_description_in_catalog`, `_update_workflow_types_in_catalog`,
`_get_workflow_type_association_table`,
`_get_python_script`, `_get_notebook_path`,
`_get_notebook_session`, `_get_git_root`, `_github_url`,
`_check_nbstrip_status`, `_in_repl` — already private

All `_`-prefixed. **Correct.**

### B.16 `runner._multirun_state`, `_atexit_registered`,
`_complete_parent_execution`, `_create_parent_execution`,
`_resolve_model_source`, `_is_multirun`, `_get_job_num`

All `_`-prefixed. **Correct.**

### B.17 `multirun_config._multirun_registry`

`multirun_config.py:63`. `_`-prefixed. **Correct.**

### B.18 `base_config._captured_hydra_output_dir`,
`_notebook_configs`, `_make_described_list`,
`_DescribedListConfig`

`base_config.py:61, 275, 630, 640`. All `_`-prefixed. **Correct.**

### B.19 `pending_summary._humanize_bytes`

`pending_summary.py:189`. `_`-prefixed. **Correct.**

### B.20 `DatasetCollection` — public, but only constructed by `Execution.datasets`

`dataset_collection.py:18`. Only constructor caller in `src/` is
`execution.py:812-814`. External grep: zero. Tests in
`test_dataset_collection.py` do `from deriva_ml.execution.dataset_collection import DatasetCollection`
(test-only).

**Recommendation:** keep public — it's the return-type of a public
property and users will reasonably want to type-annotate
`exe.datasets` as `DatasetCollection`. **Severity: low.**

### B.21 `execution_record.ExecutionRecord._update_status_in_catalog`,
`_update_description_in_catalog`, `_check_writable_catalog`

`execution_record.py:262, 281, 236`. All `_`-prefixed. **Correct.**

### Lens B summary

Privatization candidates with concrete impact:

- `PENDING_ROWS_TABLE`, `DIRECTORY_RULES_TABLE` constants → `_`-prefix.
- `PendingRowStatus`, `DirectoryRuleStatus` enums → delete (§1.5).
- `acquire_leases_for_execution` → delete (§1.6).
- `PENDING_ROWS_LEASE_CHUNK` → `_`-prefix.
- `ExecutionStateStore` → `_ExecutionStateStore` (once §1.5
  retirement makes its surface internal-only).

**Severity overall: low.** No external consumer of any privatized
symbol; the cleanup is mechanical.

---

## Lens C — Test coverage

### Per-file posture

| File | LoC | # tests | Posture | State-leakage risk |
|---|---:|---:|---|---|
| `test_execution.py` | 1 893 | 62 | Live-catalog (uses `workflow_terms`, `basic_execution`). 152 fixture refs. | High — long; uses session catalog. |
| `test_state_store.py` | 797 | 31 | Pure SQLite, no catalog. Builds engine + `ExecutionStateStore`. | None. |
| `test_staged_features.py` | 703 | 20 | Live-catalog. 4 fixture refs. | Medium. |
| `test_state_machine.py` | 624 | 23 | Pure unit + tiny mocks (3). | None. |
| `test_lookup_lineage_unit.py` | 465 | 15 | Pure unit, mocked services. | None. |
| `test_execution_registry.py` | 459 | 20 | Live-catalog; 86 fixture refs. | High. |
| `test_find_caller.py` | 375 | 20 | Pure unit, frame stack mocked. | None. |
| `test_lease_orchestrator.py` | 366 | 9 | Pure SQLite + 2 MagicMocks (no real catalog). | None. |
| `test_runner.py` | 310 | 15 | Heavy mocking (98 mock refs). | Multirun-state pollution. |
| `test_bug_c_live_smoke.py` | 286 | 3 | Live smoke. 24 fixture refs. | High. |
| `test_storage.py` | 269 | 11 | Live-catalog; 26 fixture refs. | High. |
| `test_pending_summary.py` | 244 | 12 | SQLite + Workspace (no catalog). | None. |
| `test_base_config.py` | 202 | 10 | Pure unit + mocks (hydra_zen stores). | None. |
| `test_bag_commit_poc.py` | 186 | 1 | "POC" name — single test only. | TBD. |
| `test_bug_e2_live_smoke.py` | 172 | 2 | Live smoke. 15 fixture refs. | High. |
| `test_execution_readthrough.py` | 168 | 8 | Live-catalog; 25 fixture refs. | Medium. |
| `test_upload_public_api.py` | 149 | 4 | Live-catalog; 22 fixture refs. | Medium. |
| `test_dirty_workflow.py` | 138 | 8 | Subprocess + git fixtures. | None. |
| `test_update_status.py` | 137 | 6 | Live-catalog; 25 fixture refs. | Medium. |
| `test_manifest_lease.py` | 131 | 4 | Pure mock (6 mock refs). | None. |
| `test_pending_asset_lease_validator.py` | 128 | 5 | Pure mock (12 mock refs). | None. |
| `test_lookup_lineage_live.py` | 116 | 2 | Live smoke. 11 fixture refs. | Medium. |
| `test_execution_snapshot.py` | 111 | 4 | Pure unit. | None. |
| `test_find_executions_sort.py` | 93 | 4 | Live-catalog. | Medium. |
| `test_rid_lease.py` | 90 | 5 | Pure unit. | None. |
| `test_find_workflows_sort.py` | 86 | 4 | Live-catalog. | Medium. |
| `test_dataset_collection.py` | 85 | 7 | Pure unit, no catalog. | None. |
| `test_status_migration.py` | 61 | 5 | Pure unit. | None. |
| `test_execution_hierarchy.py` | 19 | 1 | One test. | TBD. |

**Mock-vs-live mix per module:**

- **Lifecycle / state machine** (`state_machine.py`,
  `state_store.py`): excellent — pure unit + SQLite tests, no
  catalog dependency, 54 tests across two files.
- **Bag-commit pipeline** (`bag_commit.py`): minimal —
  `test_bag_commit_poc.py` is a single "POC" test; everything else
  comes through `test_execution.py`'s end-to-end paths. No
  isolated test for `_add_asset_rows_to_bag`, `_add_staged_feature_rows_to_bag`,
  `load_execution_bag`, `report_to_asset_map`. **Coverage gap.**
- **Lease primitives** (`rid_lease.py`, `manifest_lease.py`,
  `lease_orchestrator.py`): good mocked coverage. Production lease
  path (`manifest_lease`) has 4 mocked tests; orchestrator
  (production-dead, §1.6) has 9 tests. Imbalanced.
- **Runner / multirun** (`runner.py`): heavy mocking (98 mock
  refs in test_runner.py); zero live execution. Reasonable —
  Hydra integration tests would require a non-trivial harness.
- **Workflow detection** (`workflow.py`): split between
  `test_dirty_workflow.py` (subprocess + git fixtures, real but
  isolated) and `test_execution.py::TestWorkflow*` (live catalog).
  Docker scenarios fully unit-tested via `monkeypatch`. **Healthy.**
- **Asset upload / register** (`execution.py::asset_file_path`,
  `upload_execution_outputs`): exercised by every live
  `test_execution.py` test that creates assets. No isolated mock
  tests for the manifest write-through path; that lives in
  `tests/asset/`. **Acceptable.**

### Coverage gaps

**C.1 No isolated test for `bag_commit._add_asset_rows_to_bag`,
`_add_staged_feature_rows_to_bag`, `load_execution_bag`.**

The bag-commit pipeline is the production upload path. Tests
exercise it through `Execution.upload_execution_outputs()` end-to-end
(every "test_asset_*" in `test_execution.py`). When the upload
fails, the failure surface is the full pipeline, not the
individual function. Bug locality is poor.

**Recommendation:** add `tests/execution/test_bag_commit_unit.py`
with mocked `BagBuilder` and `BagCatalogLoader`. Target:
URL-dedup behavior, type-pair leasing, asset-RID rewriting in
feature payloads. **LoC: +400 of tests.** **Severity: medium**
(prevents future regressions, doesn't fix today's coverage hole).

**C.2 No test for `Execution.__exit__`'s exception-propagation contract.**

`execution.py:2339-2432` — `__exit__` is documented to return
False (propagate the exception) per spec §2.12 / R6.3. The
specifications point at "the legacy ``__exit__`` which returned
True to suppress." Searching `test_execution.py` for `with pytest.raises(.*) as exc.*\n.*with .*execute()` (the exception-in-context test) returns
zero hits. Each "exception" test catches the exception from outside the
context manager (the implicit shape) but no test verifies that an
exception inside the `with` block actually re-raises through `__exit__`
to the caller.

**Recommendation:** add three tests — exception during work,
exception in the upload, double-exception (work raised, then
__exit__'s state-machine transition raises). **LoC: +50 of tests.**
**Severity: medium.**

**C.3 No test for `state_machine.reconcile_with_catalog` when
catalog has unrecognized Status.**

`state_machine.py:509-515` catches `ValueError` from the
`ExecutionStatus(...)` constructor and raises
`DerivaMLStateInconsistency`. `test_state_machine.py` has 23
tests; none exercises a catalog row with an unknown status string.
This branch is reachable in field-deployed catalogs that drift
ahead of the deriva-ml install.

**Recommendation:** add one parametrized test feeding bogus
status strings. **LoC: +20.** **Severity: low.**

**C.4 No test for `Execution.upload_execution_outputs` retry semantics.**

`execution.py:1305-1339` documents `max_retries=3`, `retry_delay=5.0`,
"Doubles on each successive retry." Grep across `execution.py`
for `max_retries` shows zero use after the parameter is accepted —
the value is **not propagated** to any retry mechanism. The
parameter is a no-op.

**Recommendation:** either delete the parameter and the docstring
(if no retry is intended in the bag-commit pipeline) or
implement the retry loop wrapping `_bag_commit_upload`.
**LoC: −10 (delete) or +30 (implement).** **Severity: medium** —
the docstring promises retry behavior the code doesn't deliver.

**C.5 `DatasetCollection` test coverage is healthy.**

`test_dataset_collection.py` covers 7 tests: keyed access,
iteration order, missing-key error message, length, contains,
`keys`/`values`/`items`. Does NOT verify the `Mapping` contract
violation (§4.7); a test asserting "for k in dc: assert dc[k]" works
correctly would fail and force the contract decision.

**Recommendation:** add the explicit contract test. **LoC: +15.**
**Severity: low.**

**C.6 `find_executions_sort.py` / `find_workflows_sort.py` parallel structure**

Both files have 4 tests of similar shape (insert N records, fetch
with various sort orders, assert order). Identical helper code
across files. The two files exist because the find-paths
themselves are duplicated (`find_executions` and `find_workflows`).

**Recommendation:** parametrize. **LoC: −60.** **Severity: low.**

### State-leakage warning (load-bearing finding)

**The execution-test suite's risk profile:**

`tests/execution/test_execution.py` (1 893 LoC, 62 tests) uses
`workflow_terms` and `basic_execution` fixtures that depend on
`test_ml`, which calls `catalog_manager.reset()` before each test
and sets `state = CatalogState.POPULATED` after (to force the
next test to reset again). The state-flag-guards landed in
`catalog_manager.py` per PR #99 (commit `4442f82`'s parent) and
the existence guards in `ensure_features` / `ensure_datasets`
catch the case where state is stale.

**Risks specific to `tests/execution/`:**

(a) **`runner._multirun_state` pollution.** Not handled by
`catalog_manager` — that resets catalog tables but not
deriva-ml module-level globals. `test_runner.py` calls
`reset_multirun_state()` in fixtures (line 14 of `tests/execution/test_runner.py`
uses an autouse fixture pattern via context). Other tests that
trigger the multirun path (especially `test_execution.py::test_multirun_parent_lifecycle`)
do not. **Risk:** one test creates a parent execution and stores
it in `_multirun_state`; the next test that calls `run_model`
finds a stale parent, links a new child to a deleted parent
catalog row.

**Recommendation:** add an autouse fixture in
`tests/execution/conftest.py` (currently missing) that calls
`reset_multirun_state()` between tests. **LoC: +10.**
**Severity: medium.**

(b) **`Workspace` SQLite registry pollution.** Each `test_ml`
returns a fresh DerivaML instance with `working_dir=tmp_path`.
The SQLite registry is per-working-dir, so `tmp_path`'s
randomization gives test isolation. Confirmed by reading
`tests/conftest.py:117-130`. **No risk.**

(c) **Catalog state — long execution chains.** Tests in
`TestExecutionAssets`, `TestAssetCaching` upload multiple assets
into the same execution, then assume that the next test sees an
empty `Execution_Asset` table. The `catalog_manager.reset()` at
test start clears `Execution`, `Workflow`, `Dataset_Execution`,
but NOT `Execution_Asset` — that's a dynamic table managed by
the schema and dropped by `_drop_dynamic_tables`. Verified
by reading `catalog_manager.py:142-219`. **No risk** for
`tests/execution/`; `Execution_Asset` is in the permanent set.

(d) **`add_features` staging in SQLite outlives the test.**
`Execution.add_features` writes to the workspace's
`execution_state__feature_records` table. The workspace is
per-`tmp_path`, so each test's workspace is fresh.
**No risk.**

(e) **Hatrac uploads accumulate across tests.** The test catalog
is destroyed at session end (`catalog_manager.destroy()`). Hatrac
content for the test catalog is also destroyed. **No risk** within
a single session; cross-session leakage is solved by the catalog
being session-scoped.

(f) **Test order sensitivity.** `test_execution.py` is the largest
file in the suite. If pytest runs in alphabetical order
(default), `test_bag_commit_poc.py` runs before
`test_execution.py`, but the POC test creates an execution that
isn't cleaned up. The state-flag-guards in
`catalog_manager.ensure_populated` and friends catch the empty-
table case but not the dirty-table case (an execution-row that
the next test's `basic_execution` fixture creates a *second*
of). This is the most likely source of test flakiness.

**Recommendation:** make `test_bag_commit_poc.py` use `test_ml`
(which calls `catalog_manager.reset()`) — currently it doesn't
read from the file content shown. (Verify by inspection — if it
doesn't use fixtures, mark it for cleanup. The "POC" in the name
suggests it's exploratory and may be deletable.)

### Cost trade-off per test file

| File | Choice | Cost | Right? |
|---|---|---|---|
| `test_state_store.py` (31 tests, no catalog) | Fast SQLite-only | < 1s | Yes |
| `test_state_machine.py` (23 tests, no catalog) | Pure unit + mocks | < 1s | Yes |
| `test_lookup_lineage_unit.py` (15 tests, mocked) | Mocked services | < 1s | Yes |
| `test_find_caller.py` (20 tests, frame mocks) | Pure unit | < 1s | Yes |
| `test_rid_lease.py` (5 tests, no catalog) | Pure unit | < 1s | Yes |
| `test_manifest_lease.py` (4 tests, mocked) | Pure mock | < 1s | Yes |
| `test_pending_asset_lease_validator.py` (5, mocked) | Pure mock | < 1s | Yes |
| `test_base_config.py` (10 tests, hydra mocks) | Pure unit | < 1s | Yes |
| `test_lease_orchestrator.py` (9 tests, SQLite + mocks) | SQLite-only | < 1s | Yes — but tests dead surface (§1.6) |
| `test_pending_summary.py` (12 tests, SQLite) | SQLite-only | < 1s | Yes — but tests dead-writer surface (§1.5) |
| `test_runner.py` (15 tests, heavy mock) | Mock-only | < 5s | Yes |
| `test_dataset_collection.py` (7 tests) | Pure unit | < 1s | Yes |
| `test_status_migration.py` (5 tests) | Pure unit | < 1s | Yes |
| `test_execution_snapshot.py` (4 tests) | Pure unit | < 1s | Yes |
| `test_dirty_workflow.py` (8 tests, subprocess) | Subprocess + git | 5-10s | Yes — subprocess isolation is appropriate |
| `test_execution.py` (62 tests, live) | Live-catalog | 5-10min | **Mostly right but oversized — could split.** |
| `test_execution_registry.py` (20, live) | Live-catalog | 1-3min | Yes |
| `test_staged_features.py` (20, live) | Live-catalog | 1-3min | Yes |
| `test_storage.py` (11, live) | Live-catalog | < 1min | Yes |
| `test_execution_readthrough.py` (8, live) | Live-catalog | < 1min | Yes |
| `test_upload_public_api.py` (4, live) | Live-catalog | < 1min | Yes |
| `test_update_status.py` (6, live) | Live-catalog | < 1min | Yes |
| `test_lookup_lineage_live.py` (2, live smoke) | Live smoke | < 1min | Yes |
| `test_bug_c_live_smoke.py` (3, live smoke) | Live smoke | < 1min | Yes |
| `test_bug_e2_live_smoke.py` (2, live smoke) | Live smoke | < 1min | Yes |
| `test_find_executions_sort.py` (4, live) | Live-catalog | < 1min | Yes |
| `test_find_workflows_sort.py` (4, live) | Live-catalog | < 1min | Yes |
| `test_bag_commit_poc.py` (1 test) | Live? | TBD | "POC" suggests exploratory; review for retention |
| `test_execution_hierarchy.py` (1 test) | TBD | TBD | One test — verify it's not orphaned |

**Test-runtime headline:** `test_execution.py` and the 7 other
live-catalog files dominate runtime. Splitting `test_execution.py`
into per-class files (`test_workflow.py`,
`test_execution_lifecycle.py`, `test_execution_assets.py`, …) is a
mechanical refactor that improves debuggability of failures but
doesn't change runtime.

### Streamlining recommendations

**C.S1 Delete the `lease_orchestrator` tests when `lease_orchestrator`
is deleted (§1.6).** 9 tests / 366 LoC.

**C.S2 Delete the pending-rows half of `test_state_store.py` when
the pending-rows surface is deleted (§1.5).** Approximately 18 of
31 tests / ~500 LoC.

**C.S3 Delete `test_pending_summary.py`'s pending-row tests** —
keep the executions-only tests. ~150 LoC.

**C.S4 Consolidate `test_find_executions_sort.py` and
`test_find_workflows_sort.py`** via parametrization. ~60 LoC.

**C.S5 Investigate `test_execution_hierarchy.py` (19 LoC, 1 test)
and `test_bag_commit_poc.py` (186 LoC, 1 test).** Verify they
aren't orphans of deleted features.

**C.S6 Add `tests/execution/conftest.py` with an autouse
`reset_multirun_state()` fixture.** Solves a latent leakage hole.
**LoC: +15.**

**Net test LoC delta from §1 + §C cleanups:**
- Delete: ~1 100 LoC of tests covering dead production surface.
- Add: ~500 LoC of bag-commit unit coverage + multirun fixture +
  Mapping contract + retry-semantics test.

---

## Lens D — Docs/spec/ADR/docstring sync

### D.1 Orphan spec: `2026-04-21-upload-engine-deriva-py-integration-design.md`

This spec describes the production path for `upload_engine.py` →
`GenericUploader` integration. `upload_engine.py` has been
retired; the production path is `bag_commit.py`. The spec
references functions (`run_upload_engine`, `_invoke_deriva_py_uploader`)
that no longer exist.

**Reality check** (`upload_report.py:4-7`): "Post-WI2 the engine
is gone and `upload_pending` drives `Execution._bag_commit_upload`
per execution."

**Fix:** mark the spec deprecated with a one-line preamble
("Superseded by ADR-0006 / bag-commit path; retained for
archaeology"), or move to `docs/superpowers/specs/archive/`.
**Severity: low.**

### D.2 Orphan spec: `2026-04-18-sqlite-execution-state-design.md` partially superseded

This spec describes the `executions`, `pending_rows`,
`directory_rules` schema. The `executions` table shipped and is
load-bearing. The `pending_rows` + `directory_rules` schema
shipped but has no production writer (§1.5). The spec describes
them as the architecture; production has a different one
(asset manifest).

**Fix:** the spec is correct historical record; add a top-of-
document note that the pending_rows / directory_rules halves are
implemented-but-not-used and the actual production write path
is via `AssetManifest`. **Severity: medium** — newcomers reading
this spec will believe the writer surface is alive.

### D.3 `docs/user-guide/executions.md` matches code

The user-facing description of the execution lifecycle (Created
→ Running → Stopped → Pending_Upload → Uploaded), with the four
non-happy-path edges (`Running → Pending_Upload`,
`Failed → Pending_Upload`, `Uploaded → Pending_Upload`,
`Created/Running/Stopped/Failed → Aborted`), matches
`state_machine.ALLOWED_TRANSITIONS` exactly.

**Drift checked:** the user guide names `execution_stop()` /
`execution_start()` for imperative paths. Both exist
(`execution.py:1069`, `1121`). **No drift.**

### D.4 `docs/configuration/overview.md` includes `DerivaMLModel` snippet

The class snippet at `docs/configuration/overview.md:384`
matches `execution/model_protocol.py:93-176`. **No drift.**

### D.5 `workflow.py::setup_url_checksum` docstring lies

Covered in §6.3 / theme 4. The docstring describes a
`Workflow.create_workflow` factory that doesn't exist. The
example uses a non-existent method. **High-impact doc drift.**

### D.6 `state_machine.py::transition` claims `NotImplementedError`

`state_machine.py:207` lists "NotImplementedError: Online-mode
path is implemented in Task C3" in `Raises:`. Task C3 shipped.
**Stale.** See §1.10.

### D.7 `_engine_harness.py` is the canonical reference for the dead surface

It is a 12-line module whose entire content is documentation
about the engine path being "Phase 2." Phase 2 = bag commit, which
shipped. The doc itself is incorrect. See §1.1.

### D.8 ADR-0006 (bag pipeline) accurately describes the
production path

ADR-0006 defines `BagCatalogLoader` as the unified producer for
both clone and commit. `bag_commit.py:1-45` (the module
docstring) matches. **No drift.**

### D.9 ADR-0007 (annotation builders public API) — not relevant to execution/

Skipping.

### D.10 ADR-0008 (estimate_bag_size opt-out) — not relevant to execution/

Skipping.

### D.11 `Execution.upload_execution_outputs` docstring promises
retry params that aren't wired

`execution.py:1305-1339` accepts `max_retries`, `retry_delay` —
both unused. See §C.4. **Drift between docstring and behavior.**

### D.12 `Execution.add_features` `Raises` clause inaccurate

See §6.9.

### D.13 `pending_summary.py` references "WorkspacePendingSummary"
correctly

Cross-checked module docstring (line 1-7) and class definitions.
**No drift.**

### D.14 deriva-mcp legacy README references `AssetRIDConfig`

`deriva-mcp/README.md:900` and `deriva-mcp/src/deriva_mcp/resources.py:130-139`
import `AssetRIDConfig` from `deriva_ml.execution`. That symbol
**does not exist** in `execution/__init__.py` or anywhere else
in deriva-ml's source. Per CLAUDE.md the legacy `deriva-mcp` is
being phased out; this is a known drift not requiring action in
deriva-ml. **Out of scope; noted for completeness.**

---

## Persona summaries

### Senior software engineer

The execution lifecycle is correct and the bag-commit pipeline is
clean. The two big quality issues are (1) a large parallel API
surface — pending_rows, directory_rules, the lease orchestrator —
that no production code writes to but that has full SQLAlchemy
table definitions and extensive test coverage, and (2) a handful
of vestigial methods on `Execution` itself (`_flush_staged_features`,
`_from_registry`, `upload_assets`) that survived because their
delete-cost looked unclear at the time. Both are eminently
recoverable in a single targeted PR.

### Testing engineer

Coverage is strong for what matters (live lifecycle + state
machine + state-store CRUD). The notable holes are bag-commit
unit isolation, `__exit__` exception propagation, and the
multirun-state cross-test pollution risk. The biggest savings
available come from deleting the test coverage that
mirrors dead production surface (§C.S1 + §C.S2); this is
~1 100 LoC of tests gone with the production deletion. Add a
single `tests/execution/conftest.py` with the multirun-state reset.

### Technical writer

Three pieces of documentation lie: the `setup_url_checksum`
docstring on `Workflow` (claims a non-existent factory), the
`transition` `Raises:` clause (claims `NotImplementedError`), and
the `upload_execution_outputs` `max_retries`/`retry_delay`
parameter docs (the retry behavior they describe isn't
implemented). Spec
`2026-04-18-sqlite-execution-state-design.md` describes a
production architecture that isn't production today; add a header
noting the pending_rows half is shelf-warmer.

### ML-developer user (the workflow author)

The happy path — `ml.create_execution(config)`, `with exe.execute()`,
`exe.upload_execution_outputs()` — is well-documented in
`docs/user-guide/executions.md` and matches the code. Discoverable
warts: (a) `create_execution(config)` does a catalog write at
construction time (early metadata upload, §4.3); the docstring
doesn't say so; (b) the `assets: Any` typing on `BaseConfig` makes
config-file authoring harder than it needs to be; (c) the retry
params on `upload_execution_outputs` look like they configure
retry but don't. None of these is critical; each is a quality-of-
life paper-cut.

### DBA

Lifecycle writes to the catalog `Execution` table are routed
through `state_machine.transition()` and the datapath API
consistently. The one bypass is `Execution.execution_stop` writing
`Duration` directly (§4.5) — minor inconsistency but the
intent is documented. The lease POST against
`public:ERMrest_RID_Lease` is chunked at 500 and batched, which
is fine for typical workloads. Concurrent-execution safety
analysis is light: the asset manifest assumes single-writer per
execution, and there is no test exercising two processes touching
the same execution_rid simultaneously. If multi-process
upload-from-multiple-machines is on the roadmap, the manifest
needs concurrency review. The SQLite pending-rows / directory-
rules path was designed for this scenario but doesn't ship.

---

## Ranked actions (1–N)

Ranked by `(impact × confidence) / cost`.

| # | Action | Risk | LoC | Files | Rationale |
|---|---|---|---:|---|---|
| 1 | **§1.5 + §C.S2 + §C.S3** Retire `state_store.py`'s pending-rows / directory-rules surface (or formally document the reserve). Includes `PendingRowStatus`, `DirectoryRuleStatus`, the table definitions, and all related CRUD. Delete the corresponding `test_state_store.py` / `test_pending_summary.py` / `test_lease_orchestrator.py` tests covering the dead surface. | medium | −600 src, −800 tests | `state_store.py`, `lease_orchestrator.py`, `pending_summary.py`, tests | Largest single LoC liability in `execution/`. Removes the worst-of-both-worlds split (full implementation, full tests, no production writer). |
| 2 | **§1.6 + §C.S1** Delete `lease_orchestrator.acquire_leases_for_execution` and its 8 tests; preserve `reconcile_pending_leases` (production-live, returns zero-work). | medium | −120 src, −250 tests | `lease_orchestrator.py`, `test_lease_orchestrator.py` | Ships with action 1. The orchestrator is the dead second lease implementation. |
| 3 | **§1.2** Delete `Execution._flush_staged_features` and its docstring/comment references in `bag_commit.py`, `manifest_store.py`. | low | −110 src, −0 tests | `execution.py` and refs | The "retained for tests" justification is false; no test calls the function. |
| 4 | **§1.1 + §1.3 + §1.4 + §1.7 + §1.9 + §1.10 + §6.3 stale-task refs** Pure dead-code + stale-comment sweep. Includes `_engine_harness.py`, `Execution._from_registry`, `Execution.upload_assets`, `environment.locale*`, the dead IPython display fallback, the stale Group/Task/WI references. | low | −150 | several | Mechanical; each item small but they accumulate. |
| 5 | **§6.3 (workflow docstring) + §D.5 + §D.6 + §D.11** Fix the three lying docstrings: `Workflow.setup_url_checksum`, `state_machine.transition.Raises`, `Execution.upload_execution_outputs.max_retries/retry_delay`. Rewrite or implement. | low | ±0 to +20 | 3 files | High user-impact: each is read by ML-developer users; each promises behavior the code doesn't deliver. |
| 6 | **§C.4** Implement or remove the unused `max_retries` / `retry_delay` parameters on `upload_execution_outputs`. | low | −10 or +30 | `execution.py` | The API surface lies to callers. Choose: implement the wrapping retry, or rip the parameters. |
| 7 | **§C.S6** Add `tests/execution/conftest.py` with autouse `reset_multirun_state()` fixture. | low | +15 | new file | Closes a real test-isolation hole. |
| 8 | **§4.7 + §C.5** Resolve `DatasetCollection`'s `Mapping.__iter__` contract violation. Either drop the inheritance (rename to a custom class) or restore the keys-default iteration. | medium | ±0 to +10 src, +15 tests | `dataset_collection.py`, `test_dataset_collection.py` | Silent contract violation that breaks any `Mapping`-typed consumer. |
| 9 | **§4.8 + §6.4** Factor `Execution.status` / `error` / `start_time` / `stop_time` to share one `_get_registry_row` helper. Add `__all__` to the modules that lack it. | low | −40 src, +75 (`__all__`) | several | Maintainability cleanup. |
| 10 | **§C.1 + §C.2** Add bag-commit unit tests (`_add_asset_rows_to_bag`, `_add_staged_feature_rows_to_bag`, `load_execution_bag` with mocked `BagBuilder` + `BagCatalogLoader`) and `__exit__` exception-propagation tests. | low | +400 tests | new file | Coverage gap; localizes bag-commit failures. |
| 11 | **§4.2** Delegate `Execution.__enter__` to `execution_start()` (or vice versa). | low | −10 | `execution.py` | Twin near-identical transitions. |
| 12 | **§2.7 + §4.6** Consolidate IPython/Jupyter notebook-path detection in `find_caller.py`; remove the duplicate block from `workflow.py`. | medium | −60 | `workflow.py`, `find_caller.py` | Two parallel implementations of the same Jupyter session lookup. |
| 13 | **§4.5** Address the `Duration` double-write in `Execution.execution_stop`. Either fold `Duration` into SQLite's `executions` table + state-machine sync, or add explicit error handling around the bypass write. | medium | +10 src, +schema | `execution.py`, `state_store.py`, `state_machine.py` | Subtle FK-of-correctness gap between the SQLite-truth and the Duration column. |
| 14 | **§A.8** Make `Execution.uploaded_assets` a computed property reading from the manifest. | medium | −20 | `execution.py` | Last in-memory remnant of the pre-bag-commit lifecycle. |
| 15 | **§2.2 + §2.5** Swap raw ERMrest URLs in `state_machine.reconcile_with_catalog` and `lease_orchestrator.reconcile_pending_leases` for datapath calls. | low | −5 | 2 files | Restore datapath consistency. |

Items 1+2 ship together (the lease orchestrator's deletion is
gated on the pending-rows machinery's deletion). Items 3–6 are a
single clean-up PR. Items 7+10 should bundle into a "test
strengthening" PR. Items 8, 13, 14 each warrant their own PR
with focused review.

---

## Follow-up scope (Phase 4 candidates)

### 4.A `execution.py` structural split

`Execution` carries three concerns (§6.1): lifecycle (~600 LoC),
asset register/upload (~900 LoC), provenance/nesting (~400 LoC).
After actions 1–6 remove dead weight, the class is ~2 100 LoC
across the same three concerns. Splitting into `execution.py`
(lifecycle), `execution_assets.py` (the asset-register +
bag-commit-call orchestration), and leaving provenance helpers
spread between `execution_record.py` and `execution.py` is the
natural shape. Out of audit scope; flagged as motivation.

### 4.B `state_store.py` split: executions only vs. engine machinery

If theme 1 retains the pending-rows surface (option (c) /
"document the reserve API"), split it into `state_store.py`
(executions only) and `state_store_engine.py` (pending_rows +
directory_rules + the lease machinery). Makes the production /
reserve boundary visible.

### 4.C Bag-commit byte-streaming progress callback

`bag_commit.py:35-44` calls out a deriva-py follow-up for
byte-level progress streaming. Currently per-asset events fire
at staging completion; users uploading a 50GB asset see no
progress for the full upload duration. Coordinate with deriva-py
to add a `BagCatalogLoader` byte-streaming hook.

### 4.D Concurrent-execution safety review

The asset manifest assumes single-writer per execution_rid. If
multiple processes (or threads) ever touch the same execution
simultaneously, the manifest's idempotency relies on the SQLite
WAL transaction ordering. The pending_rows architecture was
designed for this scenario but doesn't ship; if multi-process
upload is on the roadmap, the manifest needs a concurrency
audit and either a lock or a migration to the SQLite pending-rows
path. Coupled with theme 1's "should we keep this surface?"
question.

### 4.E Unify the lease primitives

After theme 1 retires the SQLite pending-rows path, the lease
primitives are `rid_lease.generate_lease_token` and
`rid_lease.post_lease_batch`. `manifest_lease` becomes the only
consumer. Consider folding `manifest_lease.lease_manifest_pending_assets`
into a method on `AssetManifest` itself.
