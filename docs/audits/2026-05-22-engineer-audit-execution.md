# Engineer audit — execution/ subsystem (2026-05-22)

Release target: **v1.37.1**
Scope: `src/deriva_ml/execution/*.py` (21 modules, ~9,451 LOC) and
`tests/execution/*.py` (34 files, ~10,668 LOC).

## Summary

- **21 modules audited** (8 large: execution.py 2,676 LOC; runner.py
  720; bag_commit.py 761; base_config.py 698; workflow.py 706;
  execution_record.py 688; state_machine.py 626; state_store.py 420).
- **65 findings: 1 P0, 24 P1, 31 P2, 9 P3.**
- Top themes:
  1. **`execution.py` is a god module (2,676 LOC, ~30 public/protected
     methods on one class).** Asset-tagging, lifecycle transitions, asset
     download, dataset attach, feature staging, hierarchy, and cleanup
     all live in one place. Heaviest method (`_update_asset_execution_table`)
     mixes Input/Output branches over 130 lines with documented
     dead Output-path. Split candidates exist in `bag_commit.py` and
     `execution_record.py` patterns.
  2. **Real bug in dirty-tree detection (`Workflow._github_url`).** The
     check `"M " in result.stdout.strip()` misses every case where the
     change is unstaged (` M`), both-modified (`MM`), renamed (`R `),
     deleted (`D `), or untracked (`??`), and falsely flags clean
     files whenever any unrelated file in the repo has a staged
     modification. Combined with `cwd=executable_path.parent` rather
     than the repo root, dirty-detection is unreliable. This silently
     allows provenance-violating runs through. Tests only exercise the
     happy path (`# modified` → modified-not-staged → ` M`); they
     pass today only because the test fixture stages-then-modifies in
     a way that doesn't reproduce the substring.
  3. **Duplicate `list_input_datasets` / `list_assets` between
     `Execution` and `ExecutionRecord`.** Both walk the same catalog
     tables with the same Producer filter and the same try/except.
     `Execution`'s "fallback for dry_run" branch can never fire (a
     dry-run execution is created with `_execution_record=None`, so the
     delegate path uses the fallback unconditionally) — but the
     `_execution_record` guard implies otherwise.
  4. **Docstring-style inconsistency in `runner.py`** (NumPy section
     headers `Parameters / ----------` inside an otherwise Google-style
     codebase — CLAUDE.md mandates Google). Affects readability and
     tooling that parses doctrings.
  5. **`_update_asset_execution_table` Output branch is dead in
     production.** All Output assets now flow through
     `bag_commit._add_asset_rows_to_bag`. The Output branch only runs
     from tests (`test_asset_role_auto_tag.py`). Tests preserve a code
     path the production pipeline no longer exercises — risks divergence
     between bag-commit reality and the tested behaviour.

## Findings by module

### src/deriva_ml/execution/workflow.py

#### [P0] Dirty-tree detection is wrong — silently accepts dirty workflows
**Location:** `src/deriva_ml/execution/workflow.py:627-637`
**Issue:** Two compounding bugs in `_github_url`:
1. `is_dirty = bool("M " in result.stdout.strip())` matches only the
   first-column-`M` staged-modified status. `git status --porcelain`
   emits ` M` (unstaged modify), `MM` (both), `??` (untracked), `D ` /
   ` D`, `R ` (rename), `A ` (added), etc. None of these set
   `is_dirty=True`, so a user editing a workflow file and saving (but
   not staging) will have provenance recorded with the LAST-COMMITTED
   checksum — defeating the entire dirty-workflow guard.
2. `cwd=executable_path.parent` plus `--porcelain` (no path argument)
   walks the whole repo. A clean workflow file in a repo with any
   staged modification elsewhere will be flagged dirty (false positive).
**Evidence:**
```python
result = subprocess.run(
    ["git", "status", "--porcelain"],
    cwd=executable_path.parent,
    capture_output=True, text=True, check=False,
)
is_dirty = bool("M " in result.stdout.strip())
```
**Suggested fix:** Scope to the file and check the porcelain line's
first two columns properly. The robust form is `git status --porcelain
-- <relative-path>` and treat any non-empty output as dirty:
```python
result = subprocess.run(
    ["git", "status", "--porcelain", "--", str(executable_path)],
    cwd=repo_root, capture_output=True, text=True, check=False,
)
is_dirty = bool(result.stdout.strip())
```
Add tests for ` M`, `MM`, `??`, `D `, and the "clean file in dirty
repo" case. This is the only P0 in the audit and warrants a hold-and-fix
before tagging v1.37.1.

#### [P1] Triple-quoted "docstring" mid-function is a stray comment, not a docstring
**Location:** `src/deriva_ml/execution/workflow.py:639`
**Issue:** The line `"""Get SHA-1 hash of latest commit of the file in
the repository"""` sits inside `_github_url`, not at function head, and
behaves as a no-op string expression that the bytecode evaluator
discards. Looks like a misplaced docstring; should be `# Get SHA-1 ...`.
**Suggested fix:** Convert to a regular `#` comment. Trivial.

#### [P1] `_github_url` has no error handling for `git remote get-url` exit code
**Location:** `src/deriva_ml/execution/workflow.py:612-621`
**Issue:** `subprocess.run([... "git", "remote", "get-url", "origin"])` is
called without `check=True`. If `origin` doesn't exist, `result.stdout`
is empty and `github_url` becomes `""`. The `except CalledProcessError`
clause is unreachable without `check=True`. Returned URL silently looks
like `/blob/<sha>/relative/path.py` (no host) and gets recorded as
provenance.
**Suggested fix:** Either add `check=True` (so the except clause fires)
or test `result.returncode` explicitly. Same pattern fix applies to the
`git log` block at line 641.

#### [P1] `get_url_and_checksum` raises on non-git directories even when `allow_dirty=True`
**Location:** `src/deriva_ml/execution/workflow.py:472-480`
**Issue:** The "Not executing in a Git repository" guard fires before
`allow_dirty` is consulted. A dry-run / one-off script run from
outside a checkout cannot benefit from `DERIVA_ML_ALLOW_DIRTY=true`.
Combined with the `DERIVA_ML_DRY_RUN=true` env flag flowing into
`allow_dirty=True` in the model validator, the intent ("allow dirty
provenance") is broken at the precondition stage.
**Suggested fix:** Bypass the git-presence check when `allow_dirty` is
True; emit a warning instead and leave URL/checksum empty.

#### [P2] `_check_nbstrip_status` is defined but never called
**Location:** `src/deriva_ml/execution/workflow.py:540-556`
**Issue:** Static method appears unreferenced inside this module and is
not exported. Dead code per CLAUDE "no backwards-compat shims" rule.
**Suggested fix:** Delete or wire into `_github_url` for `.ipynb`
files.

#### [P2] `get_dynamic_version` raises `RuntimeError`, not a `DerivaML*` exception
**Location:** `src/deriva_ml/execution/workflow.py:697-700`
**Issue:** Departs from the documented hierarchy
(`DerivaMLConfigurationError`/`DerivaMLException`). Callers can't catch
this with the project's exception umbrella.
**Suggested fix:** Wrap as `DerivaMLConfigurationError("setuptools_scm
is not available")`.

#### [P3] `__setattr__` validator does heavy work on every assignment
**Location:** `src/deriva_ml/execution/workflow.py:136-175`
**Issue:** For each `workflow.description = ...` assignment, the
override checks `__pydantic_private__`, dispatches to
`_update_description_in_catalog`, then re-enters Pydantic. Fine for
intentional updates, but the same path triggers for internal
re-assignment paths (e.g. `setup_url_checksum` mutating `self.url`
inside a model validator — guarded only by the `_ml_instance is not
None` test). Document the invariant in the class docstring.

### src/deriva_ml/execution/execution.py

#### [P1] God class — Execution carries 30+ methods and 2,676 LOC
**Location:** entire file
**Issue:** One class manages lifecycle transitions (`__enter__`,
`__exit__`, `execution_start`, `execution_stop`, `abort`,
`update_status`), asset download (`download_asset`,
`_initialize_execution`), asset upload (`upload_execution_outputs`,
`_bag_commit_upload`, `_update_asset_execution_table`,
`_set_asset_descriptions`), asset staging (`asset_file_path`,
`metrics_file`), feature staging (`add_features`), dataset attach
(`download_dataset_bag`, `create_dataset`, `add_files`,
`list_input_datasets`), hierarchy (`add_nested_execution`, `is_nested`,
`is_parent`), and cleanup (`_clean_folder_contents`). Cognitive load is
high and changes ripple. Suggest extracting `AssetUploadPipeline` /
`AssetStaging` mixins, leaving lifecycle on `Execution`.
**Suggested fix:** Phase out by extracting the asset-staging and
upload methods into `execution/asset_upload.py` (mirroring the
`bag_commit.py` extraction pattern). Not a blocker for v1.37.1; ledger
for the next minor.

#### [P1] `_update_asset_execution_table` Output branch is dead in production
**Location:** `src/deriva_ml/execution/execution.py:1864-1899` (Output
branch); only call site at line 1462 always passes `asset_role="Input"`.
**Issue:** The Output flow now lives in
`bag_commit._add_asset_rows_to_bag` (which writes
`{Asset}_Execution` + `{Asset}_Asset_Type` rows during bag build —
explicitly noted in `bag_commit.py:322-326`). Only tests still call
`_update_asset_execution_table` with `asset_role="Output"`
(`test_asset_role_auto_tag.py`). The tests are pinning a code path
that production no longer exercises — silent risk of bag-commit
diverging from the tested behavior.
**Evidence:**
```python
# execution.py:1462 (only production caller)
self._update_asset_execution_table(
    {f"{asset_table.schema.name}/{asset_table.name}": [asset_path]},
    asset_role="Input",  # always Input here
)
```
**Suggested fix:** Either (a) drop the Output branch from
`_update_asset_execution_table` and rewrite the auto-tag tests to
exercise `bag_commit._add_asset_rows_to_bag` directly, or (b) keep the
branch but route the bag-commit Output path through it as the single
source of truth. Option (a) is the smaller change and aligns with the
documented "bag is now authoritative" stance.

#### [P1] `asset_file_path` falsy-`asset_types` bug — empty list collapses to asset_name
**Location:** `src/deriva_ml/execution/execution.py:1950`
**Issue:** `asset_types = asset_types or kwargs.pop("Asset_Type", None)
or asset_name`. If the user explicitly passes `asset_types=[]` to
register an asset with no content tags, the empty list is falsy and the
code substitutes `asset_name` (the table name as a tag) which then
triggers a `lookup_term` on a name that may not exist in
`Asset_Type` vocabulary. The user's intent ("no tags") becomes "guess
a tag from the table name."
**Suggested fix:** Use `if asset_types is None:` instead of `or` so an
explicit empty list is honored. Add a test:
`asset_file_path(..., asset_types=[])` should not call `lookup_term`.

#### [P1] `_initialize_execution` mixes init-side-effects with download work
**Location:** `src/deriva_ml/execution/execution.py:654-772`
**Issue:** 118-line method does (1) dataset materialization,
(2) `Dataset_Execution` association inserts, (3) batched asset
download, (4) per-asset destination-dir construction, (5) configuration
JSON serialization, (6) uv.lock attachment, (7) Hydra config asset
registration, (8) runtime-env snapshot, (9) the first
`_bag_commit_upload`. Each step has its own failure mode but they
share one log message ("Initialize status finished.") and one
exception umbrella (`DerivaMLException`). Hard to reason about
recovery — which steps already succeeded if upload at step 9 raises?
**Suggested fix:** Split into `_materialize_datasets()`,
`_download_inputs()`, `_register_init_metadata()`, and
`_upload_init_assets()` private helpers. Each one log-bracketed and
individually catchable.

#### [P1] `Execution.__init__` performs catalog inserts before SQLite registry write — partial-failure window
**Location:** `src/deriva_ml/execution/execution.py:378-482`
**Issue:** The catalog `Execution.insert` (line 378) happens BEFORE
the SQLite registry insert (line 449). If the SQLite write raises
(disk full, schema mismatch), the catalog row exists but the workspace
has no record. The `except Exception` at line 468 logs and re-raises,
but the orphaned catalog row stays. The error message says "Execution
can be recovered via ml.lookup_execution(rid) + manual adoption" — but
that path requires implementing manual adoption, which is not a documented
public API.
**Suggested fix:** Either (a) insert SQLite first (so a catalog
failure can be retried without orphan), or (b) document and ship the
adopt-orphan path that the error message references.

#### [P1] `from_registry` skips workflow lookup, leaving `workflow_rid=None`
**Location:** `src/deriva_ml/execution/execution.py:484-524`
**Issue:** `from_registry` minimally hydrates an Execution for
`resume_execution`, but sets `instance.workflow_rid = None` (line 523)
without loading it from SQLite (the registry row carries it). Any
downstream code that relies on `execution.workflow_rid` (e.g.,
`add_workflow_executions` linkage during nested-execution attach) will
see None on a resumed execution.
**Evidence:**
```python
instance.workflow_rid = None  # never repopulated from store.get_execution
```
**Suggested fix:** Populate from the SQLite row in `from_registry`:
`instance.workflow_rid = store.get_execution(rid)["workflow_rid"]`.

#### [P2] `_clean_folder_contents` has duplicate retry logic with magic constants
**Location:** `src/deriva_ml/execution/execution.py:1721-1764`
**Issue:** Inline `MAX_RETRIES = 3`, `RETRY_DELAY = 1`, and a nested
`remove_with_retry` closure. Same retry pattern appears in
`asset/manifest.py` and `catalog/clone.py`. Magic constants live in
each call site rather than `core/constants.py`.
**Suggested fix:** Move to a `core/file_utils.py` retry helper. Not a
blocker.

#### [P2] `download_asset` cache code has two `expected_md5 = asset_record.get("MD5")` lookups
**Location:** `src/deriva_ml/execution/execution.py:1399, 1404, 1430`
**Issue:** `expected_md5` is fetched once at line 1399, then re-fetched
as `asset_record.get("MD5")` at line 1404 (`if use_cache: md5 =
expected_md5`) and again at line 1430. Three lookups, two locals
(`md5` and `expected_md5`) holding the same value. Reads cleaner with
one local.
**Suggested fix:** Use `expected_md5` throughout; drop the `md5` local.

#### [P2] Inconsistent logger usage (`logger`, `self._logger`, `logging`)
**Location:** `src/deriva_ml/execution/execution.py:208, 474, 549, 673,
685, 772, 1229, 1272, 1417, 1441, 1646, 1659, 1668, 1683, 1714, 1745,
1764, 2505, 2567, 2575`
**Issue:** Three different logger handles in one file:
- `logger` (module-level, line 90)
- `self._logger` (instance handle from `ml_object._logger`)
- `logging.warning(...)` / `logging.error(...)` (root logger calls)
The module-level `logger` is `get_logger(__name__)` (deriva_ml.*),
while `logging.error` bypasses that and lands on the root logger.
Inconsistent observability and filter behavior.
**Suggested fix:** Use module-level `logger` everywhere; remove the
`self._logger` indirection.

#### [P2] `Execution.execute()` is a sugar method that doesn't add value
**Location:** `src/deriva_ml/execution/execution.py:2141-2158`
**Issue:** Returns `self`. Docstring says "use as `with exe.execute()
as e:`". Same semantics as `with exe as e:`. Adds a method the user
must know about without buying anything.
**Suggested fix:** Either delete (and update docs to use bare
`with exe:`) or document the legacy-compat rationale.

#### [P2] `list_input_datasets` dry-run fallback is unreachable
**Location:** `src/deriva_ml/execution/execution.py:2174-2191`
**Issue:** The check `if self._execution_record is not None: return
self._execution_record.list_input_datasets()` delegates the persistent
path. The "Fallback for dry_run mode" branch below requires
`_execution_record is None`, but dry-run executions also have
`_dry_run=True` and never insert into `Dataset_Execution` (per
`_initialize_execution` line 679-682). The fallback queries
`Dataset_Execution` for an execution_rid that's never been written —
returning an empty list. The fallback is dead code in practice.
**Suggested fix:** Either remove the fallback (raising
`DerivaMLException("not available in dry-run")` like `is_nested()` /
`is_parent()`) or fix the gap by populating
`Dataset_Execution` in dry-run.

#### [P2] `list_assets` swallows lookup failures with bare `except Exception: pass`
**Location:** `src/deriva_ml/execution/execution.py:2218-2226`
**Issue:** The dry-run fallback iterates `Execution_Asset_Execution`,
calls `lookup_asset(r["Execution_Asset"])`, and swallows every
exception with `pass`. Hides real bugs (e.g., a typo in the column
name); user sees an empty list and assumes "no assets" when the lookup
is just broken.
**Suggested fix:** Log at `WARNING` with the swallowed exception. The
`ExecutionRecord.list_assets` (line 660-661) does it correctly with
`logger.debug`; mirror that.

#### [P3] `asset_file_path` accepts arbitrary `**kwargs` as metadata — silent typos
**Location:** `src/deriva_ml/execution/execution.py:1911, 1963`
**Issue:** `**kwargs` are merged into `metadata_dict` without
column-name validation. A typo like `Lenght=12345` silently lands in
the asset's metadata column dict and gets dropped by the asset_table
filter later (or rejected at insert time with a confusing message).
**Suggested fix:** Validate kwargs against
`self._model.asset_metadata(asset_name)` before merging; raise
`DerivaMLValidationError` on unknown columns.

#### [P3] `__str__` and `__repr__` both define output but `__str__` doesn't include status
**Location:** `src/deriva_ml/execution/execution.py:2410-2419`
**Issue:** `__str__` lists working_dir, execution_rid, workflow_rid,
asset_paths, configuration — no status. `__repr__` includes status.
Calling `print(exe)` is less informative than `repr(exe)`.
**Suggested fix:** Either include `status` in `__str__` or have
`__str__` delegate to `__repr__`.

### src/deriva_ml/execution/execution_record.py

#### [P1] `list_assets` is a quadratic schema walk with bare `except Exception`
**Location:** `src/deriva_ml/execution/execution_record.py:609-664`
**Issue:** For each schema (domain + ml), iterates every table and
filters by `endswith("_Execution")`. For a 200-table catalog that's
400 catalog touches per call. Plus the inner try/except swallows every
filter-and-fetch error at `logger.debug`. A real catalog connectivity
problem becomes "execution has no assets." Same anti-pattern as
`Execution.list_assets` (see above).
**Suggested fix:** Index association tables once (per ml_instance
lifetime) into `self._ml_instance._asset_execution_tables` and iterate
that. Drop the bare-except to surface real errors.

#### [P1] Duplicate `list_input_datasets` / `list_assets` between Execution and ExecutionRecord
**Location:** `execution.py:2160-2227` vs `execution_record.py:568-664`
**Issue:** Both classes implement the same Producer-filter walk
(`Dataset_Execution` filter then `_producer_of_dataset` exclusion).
Even the error message style differs ("ExecutionRecord is not bound to
a catalog" vs "Execution has no bound ExecutionRecord"). Two places to
keep in sync.
**Suggested fix:** Move the implementation into a free function
`lineage._list_input_datasets(ml, execution_rid)` and have both classes
call it. Removes ~60 lines.

#### [P2] Pydantic `_workflow` PrivateAttr violates the project's "user-facing → BaseModel field" rule
**Location:** `src/deriva_ml/execution/execution_record.py:131-141`
**Issue:** CLAUDE.md ("Class idiom choice — Pydantic vs `@dataclass`")
says user-facing fields should be Pydantic fields with validation, not
PrivateAttr-backed properties. `_workflow`, `_status`, `_description`
are user-visible (read via `record.workflow`, set via
`record.description = ...`) but bypass Pydantic validation entirely.
The catalog setter is in `__setattr__`-like wrappers instead.
**Suggested fix:** Restructure so `workflow`, `status`, `description`
are real Pydantic fields with `@field_validator` for catalog write-back.
Matches the audit's "single serialization story" rationale.

#### [P2] `ExecutionRecord.__init__` overrides Pydantic's auto-init for no clear reason
**Location:** `src/deriva_ml/execution/execution_record.py:143-185`
**Issue:** Hand-rolled `__init__` that calls `super().__init__(...)`
with a partial kwargs set then manually sets `_workflow`, `_status`,
etc. Pydantic v2 supports `PrivateAttr(default=...)` initialization
through the constructor; the manual override is redundant.
**Suggested fix:** Use Pydantic v2's `model_post_init` hook, or fold
the private-attr initialization into a `@model_validator(mode="after")`.

#### [P2] `is_nested()` / `is_parent()` materialize the full iterator just to count
**Location:** `src/deriva_ml/execution/execution_record.py:337-359`
**Issue:** `return len(list(self.list_execution_parents())) > 0` walks
every parent (recursively if there's a cycle) just to ask "does at
least one parent exist?" Use the iterator and short-circuit.
**Suggested fix:** `return next(iter(self.list_execution_parents()),
None) is not None`.

### src/deriva_ml/execution/runner.py

#### [P1] Docstrings use NumPy style, project mandates Google style
**Location:** `src/deriva_ml/execution/runner.py:380-450, 633-707`
**Issue:** Sections like `Parameters\n----------\n` and `Examples\n--------\n`
are NumPy-style. CLAUDE.md ("Docstrings: Google style") requires
`Args:` / `Returns:` / `Raises:` / `Example:`. The mismatch breaks
mkdocstrings parsing if the project enforces Google in its mkdocs
config (the module docstring uses RST `:class:` directives too — also
non-Google).
**Suggested fix:** Convert `run_model` and `create_model_config`
docstrings to Google sections. Roughly 80 lines of docstring rewrite.

#### [P1] `_complete_parent_execution` catches and warns on every exception, no test coverage for failure
**Location:** `src/deriva_ml/execution/runner.py:259-263`
**Issue:** `except Exception as e: logging.warning(...)`. Atexit runs
in a teardown context; a swallowed upload failure means the user
loses their multirun parent. Only the dry-run-skip path
(`test_dry_run_placeholder_does_not_warn`) is tested. No coverage for
"parent upload raises mid-atexit" — and no observability beyond a
single WARN line.
**Suggested fix:** At minimum, log the exception with stack
(`logger.exception("...")`) instead of formatting `{e}`. Better: write
the failure to the workspace so `ml.find_incomplete_executions()`
surfaces it on next run.

#### [P1] `MultirunState` uses a class with class-level mutable defaults instead of an instance
**Location:** `src/deriva_ml/execution/runner.py:175-198`
**Issue:** `parent_execution_rid: str | None = None` etc. are
class-level attributes, but `_multirun_state = MultirunState()` is a
module-level singleton. Subclassing or instantiating elsewhere would
silently share state through the class. Should be an instance or a
plain `@dataclass`.
**Suggested fix:** `@dataclass` with default factory, or move the
state to module-level globals. The current shape misleads.

#### [P2] Global singleton state across all multirun calls — concurrency hazard
**Location:** `src/deriva_ml/execution/runner.py:198, 451`
**Issue:** `_multirun_state` is module-global; a single Python process
running two multirun sweeps concurrently (via subprocess or asyncio)
shares parent-execution state. The atexit hook compounds: register
once, runs once, even if multiple multiruns happen in sequence in a
notebook.
**Suggested fix:** Document the single-process-single-sweep invariant
explicitly. Long-term, scope state by Hydra job name.

#### [P2] `_resolve_model_source` swallows `(TypeError, OSError)` silently
**Location:** `src/deriva_ml/execution/runner.py:362-364`
**Issue:** When source-resolution fails, the workflow URL stays at the
CLI entry point and the user has no signal. Especially relevant for
`deriva-ml-run` users — they expect the workflow URL to reflect the
model file, and `_resolve_model_source` is the only path that fixes
it.
**Suggested fix:** Log at `WARNING` when resolution fails, naming the
config.

#### [P2] `run_model` is 270 lines with 13 sectional comments — split it
**Location:** `src/deriva_ml/execution/runner.py:369-630`
**Issue:** One function does Hydra-logging-reset, ML-instance
construction, RID validation, workflow URL correction, multirun parent
creation, choices capture, execution config build, parent linking, the
actual `callable_config(...)` call, and the upload. Sections are
visually separated by `# ----` banners, but the same 13 sections could
be 5 helper functions.
**Suggested fix:** Extract `_connect_ml`, `_correct_workflow_url`,
`_capture_choices`, `_run_callable_in_execution`, `_upload_outputs`
helpers.

#### [P3] `Returns -------\nNone` docstring sections will not render
**Location:** `src/deriva_ml/execution/runner.py:432-435`
**Issue:** Even in NumPy style, "Returns" should describe the return
value semantics. Saying `None` literally is fine, but pair with the
side-effects line.

### src/deriva_ml/execution/bag_commit.py

#### [P1] `_add_asset_rows_to_bag` and `_add_staged_feature_rows_to_bag` each lease RIDs in a separate batch
**Location:** `bag_commit.py:337-339, 363-372, 490-491`
**Issue:** Three separate `post_lease_batch` calls per commit (one for
asset_execution_assoc, one for asset_type rows, one for feature rows).
Each is a serialized round trip. For a 1,000-asset commit with 3 types
each + features, that's 3 sequential POSTs that could be one batch
(or one POST per logical group, properly amortized).
**Suggested fix:** Either batch into one `post_lease_batch` with
typed-token-prefix-routing, or accept the multi-batch cost and
document it.

#### [P1] `_add_staged_feature_rows_to_bag` silently skips rows on lookup failure
**Location:** `bag_commit.py:444-452`
**Issue:** When `ml.lookup_feature(target_table, feature_name)` raises,
the entire feature group is skipped with `continue` and a `logger.error`.
The user's staged feature records vanish from this commit. They stay
"pending" in SQLite (no `mark_feature_records_failed`), so the next
commit re-tries — which will fail the same way. Silent loss of work
that needs operator intervention to discover.
**Suggested fix:** Call `mark_feature_records_failed` for the group so
the user sees the failure via `pending_summary()`.

#### [P2] `match_by_columns` construction walks every table in every schema
**Location:** `bag_commit.py:561-576`
**Issue:** Nested `for schema in model.schemas: for table in
model.schemas[schema].tables.values():` iterates the full model on
every load. Cheap for small catalogs, but a deriva-py model can carry
hundreds of tables.
**Suggested fix:** Cache on the model object (or compute once per
DerivaML instance lifetime).

#### [P2] `_read_asset_type_map` ignores duplicate filename writes
**Location:** `bag_commit.py:725-760`
**Issue:** Iterates JSONL lines and calls `out.update(json.loads(line))`.
If two lines carry the same filename with different type lists, the
later one wins silently. Should at least log when a filename is
overwritten.
**Suggested fix:** Detect collision and union or warn.

#### [P3] `total_assets` denominator silently zeroes when manifest is empty
**Location:** `bag_commit.py:195-196`
**Issue:** `total_assets = sum(len(es) for es in by_table.values())`.
When empty, progress callback's `100.0 * staged_so_far / total_assets`
would be `0/0`. Code guards with `if total_assets > 0`, but the user
sees no `Staging` events for a feature-only commit. Document that
behavior, or emit a single "no assets" event.

### src/deriva_ml/execution/state_machine.py

#### [P1] `transition` writes SQLite preemptively with `sync_pending=True` even on success
**Location:** `state_machine.py:264-295`
**Issue:** The flow is: (1) write SQLite with `sync_pending=True`, (2) PUT
catalog, (3) write SQLite again to clear `sync_pending`. Two SQLite
writes per online transition. The doc-comment explains the rationale
(crash safety), but the cost is a 2× write amplification for every
status change. For a fast happy path (Created → Running → Stopped →
Pending_Upload → Uploaded), that's 8 SQLite writes per execution. WAL
mode amortizes, but 8× fsync is still real.
**Suggested fix:** Use a savepoint or single-transaction-with-deferred-flush
pattern: write the row optimistically with `sync_pending=False`, do
the catalog PUT, on failure set `sync_pending=True`. Trade-off
documented in spec §2.2 but worth re-checking against actual fsync
cost.

#### [P2] `reconcile_with_catalog` doesn't cover `Stopped ↔ {Uploaded, Pending_Upload}` disagreements
**Location:** `state_machine.py:424-439` (`_DISAGREEMENT_RULES`)
**Issue:** The rule table covers 8 pairs but misses several plausible
ones: `(Stopped, Uploaded)`, `(Stopped, Pending_Upload)`, `(Pending_Upload,
Aborted)`. These fall into the "unexpected" else branch (line 570) and
raise `DerivaMLStateInconsistency` — but the user's recourse is to
manually edit SQLite. Common case for resuming a workspace from a
catalog where another process has advanced the execution.
**Suggested fix:** Extend `_DISAGREEMENT_RULES` to "adopt" the catalog
state in these cases.

#### [P3] `validate_transition` example renders as `assert <value> is None`
**Location:** `state_machine.py:156-161`
**Issue:** Docstring shows `>>> validate_transition(...)` with the
comment "returns None, no raise". This is pure-Python and runs as
doctest — but the call requires `ExecutionStatus` to be in the local
namespace. Without conftest.py wiring it, the doctest collects and
fails on `ExecutionStatus` not being defined.
**Verification needed:** Run `uv run pytest --collect-only --doctest-modules
src/deriva_ml/execution/state_machine.py` and confirm.
**Suggested fix:** Mark the example `# doctest: +SKIP` or expose
`ExecutionStatus` in `src/deriva_ml/conftest.py`.

### src/deriva_ml/execution/state_store.py

#### [P1] `pending_summary_rows`, `count_pending_by_kind`, `count_pending_rows` are no-op stubs
**Location:** `state_store.py:336-399`
**Issue:** Three reader methods always return zero/empty
("retired in Phase 3 cleanup"). Production code paths still call them
(``Execution.pending_summary``, ``find_executions``,
``__repr__`` pending suffix). Documented, but live as silent
zero-truthers. Any caller expecting "what's pending in this execution"
gets a misleading "nothing pending" when there genuinely is staged
work in the asset manifest (`pending_assets`).
**Evidence:** `ml.pending_summary()` and `exe.pending_summary()` both
return empty PendingSummary objects in production today, regardless of
manifest state.
**Suggested fix:** Either (a) implement against the asset manifest
(`AssetManifest.pending_assets()` is the real source of pending data),
or (b) delete the stubs and update the callers to query the manifest
directly. Option (a) restores the documented surface.

#### [P2] `update_execution` accepts arbitrary `**fields` then validates against column set
**Location:** `state_store.py:258-286`
**Issue:** Runtime KeyError check is fine, but a typed signature
(`**fields: object`) gives no IDE hint about valid kwargs. A typed
TypedDict overload would catch typos statically.
**Suggested fix:** Define `class ExecutionFields(TypedDict, total=False):
status: ExecutionStatus; error: str | None; ...` and overload.

#### [P3] `delete_execution` doesn't return whether a row was deleted
**Location:** `state_store.py:403-420`
**Issue:** Silent no-op when rid doesn't exist. Caller has to call
`get_execution` before to know whether the delete actually fired.
**Suggested fix:** Return `bool` (whether a row was deleted).

### src/deriva_ml/execution/execution_configuration.py

#### [P2] `validate_assets` `else` branch fabricates an AssetSpec from `str(v)`
**Location:** `execution_configuration.py:113-114`
**Issue:** "Unknown type — try string coercion" is too lenient. A
mistaken `None` becomes `AssetSpec(rid="None")` which is later
"resolved" against the catalog and raises a confusing not-found error.
**Suggested fix:** Raise `DerivaMLValidationError` listing the
unsupported type instead.

#### [P3] `argv` default uses `Field(default_factory=lambda: sys.argv)` — captures import-time argv
**Location:** `execution_configuration.py:82`
**Issue:** `default_factory` is called at instance construction (correct),
but `sys.argv` at construction time may not match the argv of the
process that originally launched the execution (e.g., a notebook
kernel restart). The semantic is "the argv that created this
ExecutionConfiguration object" rather than "the argv that started the
job."
**Suggested fix:** Document the semantics explicitly.

### src/deriva_ml/execution/find_caller.py

#### [P2] `_top_user_frame` walks the entire stack twice
**Location:** `find_caller.py:111-189`
**Issue:** The function records `last_user_frame` and "keeps going back
to find an even higher one." Then `_get_calling_module` calls it
twice (line 292 and line 306), each walking the full stack. For deep
notebooks this is measurable.
**Suggested fix:** Cache the result for the lifetime of one
`_get_calling_module` invocation.

#### [P2] `_get_notebook_path` queries Jupyter Server API with `timeout=3`
**Location:** `find_caller.py:239`
**Issue:** A non-responsive Jupyter server blocks workflow construction
for 3 seconds × N servers. In a notebook session this is hit on every
`create_workflow`/`create_execution` call.
**Suggested fix:** Cache the resolved notebook path per process. The
mapping `(kernel_id) → notebook_path` is stable for the lifetime of the
kernel.

### src/deriva_ml/execution/base_config.py

#### [P2] `_captured_hydra_output_dir` is a module-global mutable
**Location:** `base_config.py:61, 264-266, 527`
**Issue:** Captures Hydra's output dir at `get_notebook_configuration`
call time, then `run_notebook` reads it. Two notebooks running
concurrently in the same process (Jupyter Lab kernels in the same
host?) would share-state-and-clobber. Stack-trace recoverable but a
multi-tenant footgun.
**Suggested fix:** Return the path from `get_notebook_configuration`
explicitly as a second tuple element; remove the module global.

#### [P2] `load_configs` "experiments must load last" rule is undocumented invariant
**Location:** `base_config.py:425-428`
**Issue:** Magic name "experiments" is special-cased without
explanation. Future devs adding a config module won't know why.
**Suggested fix:** Document the dependency direction explicitly in the
function's `Note:` block (it's noted at the end, but the *why* isn't).

### src/deriva_ml/execution/environment.py

#### [P2] `get_os_info["environ"]` captures every environment variable verbatim
**Location:** `environment.py:171`
**Issue:** Includes secrets-bearing variables like
`AWS_SECRET_ACCESS_KEY`, `DERIVA_CLIENT_AUTH_TOKEN`, etc. The captured
snapshot gets uploaded as an Execution_Metadata asset. Anyone with
catalog read access to the asset table can extract these.
**Evidence:** No allow/deny-list applied.
**Suggested fix:** Filter against an explicit allow-list or redact
keys matching common secret-name patterns (`*_TOKEN`, `*_KEY`,
`*_SECRET`, `*_PASSWORD`). This is a confidentiality issue but not
P0 because access to the asset already implies catalog read which
is itself privileged.

### Test coverage gaps (cross-module)

#### [P1] Zero tests for `multirun_config.py` public surface
**Location:** `src/deriva_ml/execution/multirun_config.py` (153 LOC)
**Issue:** None of `multirun_config()`, `get_multirun_config()`,
`list_multirun_configs()`, `get_all_multirun_configs()`, or
`MultirunSpec` are exercised by any test. They're public symbols
re-exported in `__init__.py`. A typo or re-architecting would land
silently.
**Suggested fix:** Add `tests/execution/test_multirun_config.py`
covering register/lookup/list/get-all and registry isolation.

#### [P1] Zero tests for `environment.py` helpers
**Location:** `src/deriva_ml/execution/environment.py` (237 LOC)
**Issue:** `get_execution_environment`, `get_loaded_modules`,
`get_platform_info`, `get_os_info`, `get_umask`, `get_sys_info` are all
untested. They run on every execution (saved as a metadata asset). A
platform-specific edge case (e.g., `os.getlogin()` raising on a
container with no login session) would crash every execution.
**Suggested fix:** At least smoke tests that the dict has the
documented keys and serializes to JSON.

#### [P1] `Execution.metrics_file`, `database_catalog`, and `catalog` property paths untested
**Location:** `src/deriva_ml/execution/execution.py:1025-1070, 2016-2091`
**Issue:** No tests for `metrics_file()` (the convenience metric
write path), `database_catalog` property (returns DerivaMLBagView for
downloaded bags), or `catalog` property.
**Suggested fix:** Unit tests with the existing `basic_execution`
fixture pattern.

#### [P1] `from_registry` classmethod not directly tested
**Location:** `src/deriva_ml/execution/execution.py:484-524`
**Issue:** Only exercised indirectly through `resume_execution` calls
in `test_execution_registry.py`. The classmethod itself — minimum-init
construction, the explicit `workflow_rid=None` assignment, the
absent `_execution_record` — isn't asserted.
**Suggested fix:** Test that the constructed instance answers
`status` (via read-through) and raises on `is_nested()` /
`is_parent()` (the documented dry-run failure modes).

#### [P2] `lineage.py` Pydantic models tested at the lookup layer, not as standalone shapes
**Location:** `src/deriva_ml/execution/lineage.py`
**Issue:** `LineageResult`, `LineageNode`, `RootDescriptor`,
`WorkflowSummary`, etc. are exposed publicly (re-exported from
`__init__.py`) for downstream MCP consumers. No direct
schema/serialization tests for the model boundary —
`test_lookup_lineage_unit.py` exercises them via the lookup path but
doesn't pin the JSON shape.
**Suggested fix:** Add `test_lineage_models.py` with `.model_dump()` /
`.model_validate_json()` round-trip tests.

#### [P2] `_clean_folder_contents` retry path not tested
**Location:** `src/deriva_ml/execution/execution.py:1721-1764`
**Issue:** The retry-on-PermissionError logic (for Windows compat) is
never triggered in tests. A regression that turns retries off would
go unnoticed.
**Suggested fix:** Mock `Path.unlink` to raise PermissionError once,
then succeed; assert retry count.

#### [P2] `DescribedList` runtime behavior covered via `test_ast_walker`, but `with_description` not directly tested
**Location:** `src/deriva_ml/execution/base_config.py:580-698`
**Issue:** Only consumer is `tests/config/test_ast_walker.py` (which
runs at the AST level). No test verifies that `instantiate(
with_description([...]))` actually produces a `DescribedList` with the
expected `.description` attribute.
**Suggested fix:** Add `test_with_description_instantiates.py` with a
hydra-zen instantiate round-trip.

### Duplication candidates (cross-module)

#### [P1] `_update_description_in_catalog` / `_update_status_in_catalog` duplicated across Workflow, ExecutionRecord
**Location:** `workflow.py:201-219`, `execution_record.py:282-314`
**Issue:** Both classes implement near-identical `pathBuilder().schemas
[ml_schema].<Table>.update([{"RID": ..., "<Col>": ...}])` helpers.
Pattern is "look up the path, update one column on one RID." Could be
a shared `_catalog_set_column(ml, schema, table, rid, column, value)`
helper.

#### [P1] `pathBuilder().schemas[ml_schema].Execution_Execution` query for parent/child appears 4× across the codebase
**Location:** `execution.py:2358-2368`, `execution_record.py:412-447,
500-535`
**Issue:** Same "filter `Execution_Execution` on parent/child, fetch
related Execution rows, instantiate ExecutionRecord" walked 4 times.
Each has minor variations (recurse flag, visited set, RID kwarg).
**Suggested fix:** Extract `lineage._walk_execution_hierarchy(ml,
rid, direction)` (direction = "children" or "parents") with the
recursion / visited tracking centralized.

#### [P2] `_check_writable_catalog` duplicated between Workflow and ExecutionRecord
**Location:** `workflow.py:177-199`, `execution_record.py:256-280`
**Issue:** Same `ErmrestSnapshot` isinstance check + same RID-presence
check + same exception message format. The only difference is the
"operation" string interpolation.
**Suggested fix:** Move to a free function
`_assert_writable_catalog(catalog, rid, operation)` in
`execution_record.py` or `core/checks.py`. Both classes import it.

#### [P2] `update_status` implementation pattern duplicated 3×
**Location:** `execution.py:1189-1243`, `execution_snapshot.py:199-251`,
`execution_record.py:316-335`
**Issue:** Three classes implement `update_status` differently:
- `Execution.update_status` goes through the state machine.
- `ExecutionRecord.update_status` goes through
  `_update_status_in_catalog` (catalog-only, no state machine).
- `ExecutionSnapshot.update_status` goes through the state machine.
The mismatch: `ExecutionRecord` skips state-machine validation, so
`record.update_status(ExecutionStatus.Failed)` from any starting
status will succeed without `ALLOWED_TRANSITIONS` enforcement.
**Suggested fix:** Either (a) route ExecutionRecord through the state
machine (preferred — consistent validation), or (b) document the
"no state-machine guard" intent and add a warning to the docstring.

#### [P2] `_format_duration` is a module-private but its logic is duplicated by `transition` callers
**Location:** `execution.py:110-138`
**Issue:** `execution_start` / `execution_stop` / `__exit__` /
`upload_execution_outputs` all call `_format_duration` — five sites.
The state machine sees only the formatted string. Could move the
formatting INTO the state machine's `extra_fields` post-processing so
callers pass `{stop_time: dt}` and the state machine derives
`duration`.
**Suggested fix:** Centralize in `state_machine.transition` once the
read-through pattern stabilizes.

#### [P2] `lookup_term(MLVocab.asset_role, ...)` called repeatedly across asset paths
**Location:** `execution.py:1812, 1953`, `bag_commit.py:168`
**Issue:** `Output` / `Input` / individual asset-type lookups happen
on every asset operation. Each is a small catalog GET. Could be cached
once on the ml_instance.
**Suggested fix:** Memoize `lookup_term` results per
`(vocab, term)` pair on the ml_instance for the session lifetime.

#### [P3] Two identical `from datetime import timezone` local imports inside `Execution`
**Location:** `execution.py:127, 426, 1270, 1317, 1560, 2499`
**Issue:** Six in-function local imports of the same name. Pulled
locally to avoid potential import-time circulars, but the module
already imports `from datetime import datetime` at top level — the
`timezone` import could join it without breaking anything.
**Suggested fix:** Lift to top-level import.

## Top action items for v1.37.1 hold-gate

1. **P0 fix:** `Workflow._github_url` dirty-detection — replace the
   substring check with proper file-scoped porcelain parsing. Add
   tests for ` M`, `MM`, `??`, `D `, untracked, and the
   "clean file in dirty repo" false positive case.
2. **P1 backports worth doing pre-release** (cheap, contained):
   - `asset_file_path` `or asset_name` falsy-asset_types bug (one
     line + one test).
   - `_initialize_execution` setting `instance.workflow_rid = None`
     in `from_registry` (one line + one test).
   - Mid-function triple-quoted string at workflow.py:639 (one line).
3. **Defer to v1.38.x:**
   - God-class refactor of `Execution`.
   - `runner.py` Google-style docstring conversion.
   - `_update_asset_execution_table` dead-Output-branch removal.
   - State-store pending-summary stubs replaced with manifest-backed
     readers.
   - `multirun_config` and `environment` test gaps.
