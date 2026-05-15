# deriva-ml audit 2026-05 — Phase 3: core/

Reviewed `/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/core/`
(11 577 LoC across 30 Python files, including 10 mixins and the
`DerivaML` class) and the corresponding test suite
`/Users/carl/GitHub/DerivaML/deriva-ml/tests/core/` (2 748 LoC,
19 test files, ~120 tests) at the tip of
`fix/catalog-manager-state-guards` (HEAD `4442f82`). Cross-workspace
references were grepped against
`/Users/carl/GitHub/DerivaML/{deriva-mcp,deriva-mcp-core,deriva-ml-mcp,deriva-ml-model-template,deriva-skills,deriva-ml-skills}/`.

`core/` is the foundation. `core/base.py` (1 718 LoC) defines
`DerivaML`, which is the user's primary entry point. The mixins under
`core/mixins/` (5 100 LoC across 10 files) compose `DerivaML`'s
surface — every `create_*`, `find_*`, `lookup_*`, `list_*`,
`add_*`, and `delete_*` a user calls is in either `base.py` or a
mixin. This makes `core/` the highest-leverage uncovered surface
after the Phase 3 execution sweep — and unlike `execution/` it does
not carry an unfinished cutover. The dragging tail here is
different: **`core/` is mostly load-bearing, but pre-cutover legacy
aliases, stale docstrings, dead helpers in `upload_layout.py`,
and a thin layer of post-Phase-3 follow-ups (now-stub callsites,
stale CLAUDE.md references) accumulate around an otherwise healthy
foundation.**

## Executive summary

Overall posture: **healthy with three concentrated cleanups
available.** `core/base.py`, `core/mixins/*.py`, `core/schema_cache.py`,
`core/schema_diff.py`, `core/catalog_stub.py`, `core/connection_mode.py`,
`core/sort.py`, and `core/exceptions.py` are all in good shape. The
liabilities are concentrated in three places: (1) the explicit
"backwards compatibility" alias surface in `core/ermrest.py` and
`core/enums.py`, (2) ~150 LoC of dead helpers in `core/upload_layout.py`,
and (3) post-Phase-3 follow-ups where the `reconcile_pending_leases`
stub callsites in `base.py` / `mixins/execution.py` and the
`validate_vocabulary_terms` broken helper in `core/validation.py` should
have been pulled along with the stub.

Top themes ranked by impact:

1. **`core/validation.validate_vocabulary_terms` is broken.** At
   `core/validation.py:374` it calls `ml.list_terms(vocabulary_name)`
   — but `list_terms` does not exist on `DerivaML`. The real method
   is `list_vocabulary_terms`. The function is exported in
   `__all__` (line 74) and documented with a docstring example
   that would crash on first use. No production callers. Cross-
   workspace grep returns zero hits. The function is dead and
   buggy; either fix the method name or delete the function.

2. **Three callsites still invoke the retired `reconcile_pending_leases`
   stub.** `core/base.py:364-379` and `core/mixins/execution.py:478-491`
   continue to import and call `reconcile_pending_leases` from
   `execution/lease_orchestrator.py`, which after Phase 3 audit #2 is
   a documented no-op (`lease_orchestrator.py:32-52`). The
   `reconcile_pending_leases` body is literally `return`. The
   import + try/except machinery around it is now ~30 LoC of
   ceremony around a no-op. The audit closing inventory called out
   "Phase-4 candidate" cleanups for this; these specific callsites
   are mechanical to remove.

3. **`core/upload_layout.py` carries ~150 LoC of dead helpers.**
   `asset_file_path` (the function at line 595), `upload_asset`
   (line 535), `upload_directory` (line 343), `upload_staging_root`
   (line 196), `normalize_asset_dir` (line 105) have zero callers in
   `src/`, `tests/`, or any of the six sibling workspaces. The names
   `asset_file_path` and `upload_directory` survive in docstrings and
   external docs but the function-form `asset_file_path` is dead —
   `Execution.asset_file_path` is the live entry point and uses
   different logic (manifest-based, not directory-based). Deleting
   the module-level versions removes 150+ LoC of "looks-alive"
   bait that doesn't ship.

4. **Mixin docstrings describe a singular `domain_schema` attribute
   that doesn't exist.** Five mixins
   (`asset.py:44`, `dataset.py:47`, `feature.py:40`, `file.py:34`,
   `path_builder.py:37`) list "`domain_schema`: str - name of the
   domain schema" in their "This mixin requires the host class to
   have:" section. The real attribute is `domain_schemas:
   frozenset[str]` (plural), declared in `DerivaML` at `base.py:120`
   and accessed via `for s in self.domain_schemas: ...`. The
   singular form predates the multi-schema cutover (PR #100 introduced
   `domain_schemas`); the docstring carries the legacy convention.

5. **`core/ermrest.py` and `core/enums.py` are full of "backwards
   compatibility" aliases that have no external consumer.**
   `ColumnDefinition`, `KeyDefinition`, `ForeignKeyDefinition`,
   `TableDefinition` aliases at `ermrest.py:86-112` (annotated
   "Alias for ColumnDef from deriva.core.typed. This maintains
   backwards compatibility with existing DerivaML code. New code
   should use ColumnDef directly."). Cross-workspace grep confirms
   `deriva-ml-model-template/src/scripts/_cifar10_schema.py` and
   `deriva-mcp/src/deriva_mcp/tools/schema.py` both still import
   the legacy names. The aliases are **load-bearing for external
   consumers** — not delete candidates, but the "new code should
   use ColumnDef directly" docstring needs to flip: external code
   does not migrate when the library prefers it, so deriva-ml's
   own internal code (e.g. `core/mixins/dataset.py:25`,
   `core/mixins/asset.py:28`) should standardize on the alias the
   external world uses. Similar story for `enums.BuiltinTypes` →
   `deriva.core.typed.BuiltinType` (line 22).

6. **`DatasetMixin._dataset_table` is a dead `raise NotImplementedError`
   property.** `core/mixins/dataset.py:73-75` declares
   `_dataset_table` as a property that raises NotImplementedError,
   with a docstring saying "Must be provided by host class". The
   host class (`DerivaML`) does provide it (`base.py:778-780`), so
   the mixin's raise-NotImplementedError property is structurally
   dead: it would only fire if someone instantiated `DatasetMixin`
   alone, which is impossible because the mixin demands too many
   other host attributes. The pattern is over-engineered — the
   real "host class must have this" contract is enforced by class
   annotation, not by a stub property.

7. **`Execution._from_registry` is a private method, called by
   `core/mixins/execution.py:495` — the Phase 3 audit nearly
   deleted it.** The Phase 3 execution audit (§1.3, ranked action
   #3) initially proposed deleting `Execution._from_registry`. The
   closing inventory rejected the action ("§1.3 — proposed
   `Execution._from_registry` deletion — rejected after re-verification.
   The audit claimed zero callers, but `core/mixins/execution.py:495`
   (`resume_execution`) clearly calls it"). The caller is in this
   subsystem; the lesson is that `core/`'s use of cross-module
   private API surfaces (`Execution._from_registry`) is one of the
   reasons grep-based audits can miss callers. Worth flagging that
   `_from_registry` should probably be renamed to a non-underscore
   name (or `resume_execution` should inline the body) so the
   "private + cross-module" pattern doesn't recur.

Worst-offending modules within `core/`:

1. **`base.py`** (1 718 LoC). `DerivaML.__init__` carries 100+
   lines of branching for online/offline init; the lifecycle is
   correct but the file is the largest single source of
   maintenance friction. The `__del__` cleanup, `mode` property,
   `workspace` lazy-init, `cache_table`/`_cache_features`,
   `clear_cache`/`clean_execution_dirs`/`get_storage_summary` (the
   "storage management" trio, ~250 LoC), and `validate_schema`
   are all single-call entry points layered onto a class that
   already inherits from 10 mixins.

2. **`validation.py`** (468 LoC). Healthy `VALIDATION_CONFIG` /
   `STRICT_VALIDATION_CONFIG` constants at the top. `ValidationResult`
   is a `@dataclass` that uses emojis in `__repr__` (✓/✗/⚠) — a
   user-facing class that should be Pydantic per CLAUDE.md's
   "Class idiom choice" guidance. `validate_rids` is healthy and
   externally consumed; `validate_vocabulary_terms` is broken
   (theme 1); `validate_execution_config` is shallow sugar around
   `validate_rids`.

3. **`upload_layout.py`** (649 LoC). The biggest LoC concentration
   of dead module-level helpers (theme 3). `bulk_upload_configuration`
   and `asset_table_upload_spec` are the live ones — everything
   else has zero callers or only docstring callers.

4. **`mixins/annotation.py`** (977 LoC). 17 methods, all live.
   Healthy. The `STRICT_PREALLOCATED_RID_TAG` constant for Bug E.2
   lives here without an `__all__`.

5. **`mixins/execution.py`** (1 262 LoC). Healthy. `lookup_lineage`
   and its helpers (`_classify_rid`, `_producer_of_dataset`,
   `_producer_of_asset`, `_walk_node` — ~280 LoC) is the bulk;
   the rest is the create/list/find/resume surface.

---

## Subsystem inventory

| File | LoC | Posture |
|---|---:|---|
| `__init__.py` | 67 | Re-exports + explicit `__all__`. **Healthy.** |
| `async_helpers.py` | 71 | `run_async` for notebook/non-notebook async bridge. **Healthy.** |
| `base.py` | 1 718 | `DerivaML` class. Lifecycle correct; `__init__` carries 100+ lines of mode branching. Three Phase-3 leftovers: dead `reconcile_pending_leases` call (theme 2), one stale-task docstring, one private cross-module method (`Execution._from_registry`). |
| `catalog_stub.py` | 41 | Offline-mode stand-in for `ErmrestCatalog`. **Healthy.** |
| `config.py` | 233 | `DerivaMLConfig` Pydantic model + hydra-zen integration. **Healthy.** |
| `connection_mode.py` | 32 | `ConnectionMode` StrEnum. **Healthy.** |
| `constants.py` | 142 | `RID`, `ML_SCHEMA`, RID regex, system schema helpers. **Healthy.** |
| `definitions.py` | 181 | Pure re-export module. **Healthy.** |
| `enums.py` | 171 | `UploadState`, `MLVocab`, `MLAsset`, `MLTable`, `ExecMetadataType`, `ExecAssetType`. `BuiltinTypes` is a legacy alias for `BuiltinType` (theme 5). **Healthy** but the alias purpose is muddy. |
| `ermrest.py` | 325 | `ColumnDef`/`KeyDef`/`ForeignKeyDef`/`TableDef` re-exports + legacy aliases (theme 5). `VocabularyTerm`, `VocabularyTermHandle`. Uses `warnings.filterwarnings` at module load to silence pydantic. **Healthy** with documented compat concerns. |
| `exceptions.py` | 645 | The full exception hierarchy. **Healthy.** |
| `filespec.py` | 187 | `FileSpec` Pydantic model with hash computation. **Healthy.** |
| `logging_config.py` | 221 | `get_logger`, `configure_logging`, `is_hydra_initialized`. **Healthy.** CLAUDE.md references a non-existent `LoggerMixin` (D.2). |
| `pd_utils.py` | 36 | `rows_to_dataframe`. **Healthy.** |
| `schema_cache.py` | 204 | Workspace cache + `PinStatus`. Atomic writes via `os.replace`. **Healthy.** |
| `schema_diff.py` | 309 | `SchemaDiff` Pydantic + `_compute_diff`. **Healthy.** |
| `sort.py` | 94 | `resolve_sort` helper. **Healthy.** |
| `upload_layout.py` | 649 | Mix of live (`asset_table_upload_spec`, `bulk_upload_configuration`, `execution_root`, `flat_asset_dir`, `manifest_path`, `asset_root`, `asset_type_path`, `table_path`) and dead helpers (theme 3: `asset_file_path`, `upload_asset`, `upload_directory`, `upload_staging_root`, `normalize_asset_dir`). |
| `validation.py` | 468 | `VALIDATION_CONFIG` constants + `validate_rids` (used externally) + `validate_vocabulary_terms` (broken, theme 1) + `validate_execution_config` (sugar). `ValidationResult` is a dataclass that ought to be Pydantic. |
| `mixins/__init__.py` | 42 | Re-exports the 10 mixins. **Healthy.** |
| `mixins/annotation.py` | 977 | 17 methods for table/column display annotations. **Healthy.** |
| `mixins/asset.py` | 447 | `create_asset`, `list_assets`, `find_assets`, `lookup_asset`, etc. **Healthy.** Mixin docstring says singular `domain_schema` (theme 4). |
| `mixins/dataset.py` | 953 | `find_datasets`, `lookup_dataset`, `validate_dataset_specs`, denorm helpers. Dead `_dataset_table` stub property (theme 6). |
| `mixins/execution.py` | 1 262 | `create_execution`, `lookup_execution`, `list_executions`, `find_executions`, `resume_execution`, `gc_executions`, `upload_pending`, `lookup_lineage`. Dead `reconcile_pending_leases` import (theme 2). |
| `mixins/feature.py` | 636 | `create_feature`, `lookup_feature`, `feature_values`. Three "Retired —" stubs that raise (`add_features`, `fetch_table_features`, `list_feature_values`, `select_by_workflow`) — kept as redirect points; legitimate. |
| `mixins/file.py` | 265 | `add_files`, `list_files`, `_bootstrap_versions`, `_synchronize_dataset_versions`, `_set_version_snapshot`. **Healthy** but the three private versioning helpers look like they belong in `DatasetMixin`. |
| `mixins/path_builder.py` | 162 | `pathBuilder`, `_domain_path`, `_table_path`, `get_table_as_dataframe`, `get_table_as_dict`. **Healthy.** |
| `mixins/rid_resolution.py` | 207 | `resolve_rid`, `resolve_rids`, `_retrieve_rid` + `BatchRidResult` dataclass. **Healthy.** |
| `mixins/vocabulary.py` | 430 | `add_term`, `lookup_term`, `list_vocabulary_terms`, `delete_term`, vocabulary cache machinery. **Healthy.** |
| `mixins/workflow.py` | 402 | `find_workflows`, `lookup_workflow`, `lookup_workflow_by_url`, `create_workflow`, `_add_workflow`. **Healthy.** |

Cross-module dependencies worth naming:

- `DerivaML.__init__` (`base.py:218-379`) is the single
  construction entry point. After init, every mixin reads
  `self.model`, `self.catalog`, `self.ml_schema`,
  `self.domain_schemas`, `self.default_schema`, `self.working_dir`,
  `self.cache_dir`, and `self._mode`. The mixins **type-hint these
  as class attributes** in their bodies (e.g.
  `mixins/dataset.py:64-70`), but only as documentation —
  Python's typing system doesn't enforce that the host class
  provides them.

- `ExecutionMixin` reads `self.workspace` (a lazy property on
  `DerivaML`) and `self._mode` (a `__init__` attribute). The mixin
  does not type-hint either — `self.workspace` is not declared in
  `ExecutionMixin`'s class body. This works today because
  `DerivaML` inherits from the mixin and provides both, but a
  reader of `ExecutionMixin` alone has no signal that `workspace`
  is required. Same for `self._mode` (the mixin reads it at line
  166 and 458 without a declaration).

- `AssetMixin.create_asset` calls `self.add_term(...)` (line 127)
  — depends on `VocabularyMixin`. `WorkflowMixin._add_workflow`
  calls `self.lookup_term(...)` (line 204) — depends on
  `VocabularyMixin`. `WorkflowMixin.create_workflow` calls
  `self.lookup_term` (line 399) — same. `FileMixin.add_files`
  calls `self.lookup_term`, `self.list_vocabulary_terms`,
  `self.resolve_rid` — depends on `VocabularyMixin` and
  `RidResolutionMixin`. The mixin web has clear precedence:
  `VocabularyMixin` ⊂ `RidResolutionMixin` ⊂ `PathBuilderMixin`,
  with `AssetMixin`/`WorkflowMixin`/`FileMixin`/`DatasetMixin`/
  `FeatureMixin`/`ExecutionMixin` all depending on the lower
  three.

- `core/base.py:364-379` and `core/mixins/execution.py:478-491`
  import `reconcile_pending_leases` from
  `execution/lease_orchestrator.py`. After Phase 3 retirement that
  function is a no-op stub.

- `core/upload_layout.py` is imported from `execution/execution.py`
  (5 helpers), `execution/bag_commit.py` (2 helpers), `core/base.py`
  (1 helper `upload_root`), `core/mixins/path_builder.py` (1 helper
  `table_path`), `asset/null_sentinel_processor.py` (1 constant),
  and `schema/annotations.py` (1 helper). Of the 30 module-level
  names defined in `upload_layout.py`, only 12 have callers.

---

## Lens 1 — Dead code

### 1.1 `core/validation.validate_vocabulary_terms` is dead AND broken

`core/validation.py:350-391` declares the function. At line 374:

```python
existing_terms = ml.list_terms(vocabulary_name)
```

`ml.list_terms` does not exist on `DerivaML`. Grep across the
workspace shows the only declarations of a similar name are
`VocabularyMixin.list_vocabulary_terms` (`mixins/vocabulary.py:295`).
Calling `validate_vocabulary_terms(ml, "Dataset_Type", [...])` per
the docstring example (line 366) would raise `AttributeError` on
first attempt.

Cross-workspace grep:

```
$ grep -rn "validate_vocabulary_terms" /Users/carl/GitHub/DerivaML
deriva-ml/tests/schema/test_validation.py:130   # tests schema/validation._validate_vocabulary_terms (different)
deriva-ml/src/deriva_ml/core/validation.py:74   # __all__ export
deriva-ml/src/deriva_ml/core/validation.py:350  # def
deriva-ml/src/deriva_ml/core/validation.py:366  # docstring
deriva-ml/src/deriva_ml/schema/validation.py:400  # different method (_validate_vocabulary_terms)
deriva-ml/src/deriva_ml/schema/validation.py:540  # ditto
```

Zero callers anywhere. The function is exported in
`__all__` (`validation.py:74`) so it advertises an external surface
it cannot deliver.

**Fix:** delete the function plus its `__all__` entry. **Risk:
trivial.** No callers. **LoC: −42.** **Severity: medium** — the
function is a tripwire if any user reads the export list and tries
to use it.

### 1.2 `reconcile_pending_leases` call sites are now ceremony around a no-op

`core/base.py:364-379`:

```python
if self._mode is ConnectionMode.online:
    from deriva_ml.execution.lease_orchestrator import reconcile_pending_leases

    try:
        reconcile_pending_leases(
            store=self.workspace.execution_state_store(),
            catalog=self.catalog,
            execution_rid=None,
        )
    except Exception as exc:
        # Best-effort. If reconciliation itself fails, log and
        # move on — the user's operation can still proceed;
        # the next acquire_leases call will retry.
        logger.warning(
            "startup lease reconciliation failed (%s); continuing",
            exc,
        )
```

`core/mixins/execution.py:478-491` — same pattern, scoped per
execution at `resume_execution`.

Both call `reconcile_pending_leases`, which (per
`execution/lease_orchestrator.py:32-52`) is documented as a no-op:

```python
def reconcile_pending_leases(*, store, catalog, execution_rid=None) -> None:
    """No-op crash-recovery stub. Retained for call-site stability.
    ...
    """
    # Intentional no-op — see module docstring.
    return
```

So 16 LoC at `base.py` and 13 LoC at `mixins/execution.py` are
calling a function whose entire body is `return`. The try/except
wrapper around it adds further overhead.

**Fix:** delete the imports and call sites. Optionally retire
`lease_orchestrator.py` itself (after deletion, nothing imports
from it). **Risk: low** — verified the no-op semantics with a
direct read of the stub. **LoC: −30 src** + `lease_orchestrator.py`
(53 LoC) becomes removable. **Severity: medium** — the ceremony is
ten times the size of the function it wraps.

### 1.3 Five module-level helpers in `core/upload_layout.py` are dead

Per `grep -rn` across all six sibling workspaces plus `src/` and
`tests/`:

- `asset_file_path(prefix, exec_rid, asset_table, file_name,
  metadata)` (`upload_layout.py:595-633`) — zero callers. The
  live method with the same name is `Execution.asset_file_path`
  (`execution/execution.py:1638`) which has different signature
  and behavior (manifest-based, not directory-segment-based).
- `upload_asset(model, file, table, **kwargs)`
  (`upload_layout.py:535-592`) — zero callers. Predates the
  bag-commit pipeline.
- `upload_directory(model, directory, progress_callback, ...)`
  (`upload_layout.py:343-531`) — zero callers. Predates the
  bag-commit pipeline. The Phase 3 audit closing summary says
  "One asset-upload path (`bag_commit` via
  `Execution._bag_commit_upload`)" — this module-level function is
  not on the live path.
- `upload_staging_root(prefix, exec_rid)`
  (`upload_layout.py:196-205`) — zero callers. The doc says
  "ephemeral upload-staging directory, created at upload time
  only" — but no current caller creates it.
- `normalize_asset_dir(path)` (`upload_layout.py:105-117`) —
  zero callers.

Live module-level helpers in `upload_layout.py` (kept):
`upload_root`, `execution_rids` (function form, used by base.py),
`execution_root`, `asset_root`, `flat_asset_dir`, `manifest_path`,
`table_path`, `asset_table_upload_spec`,
`bulk_upload_configuration`, `asset_type_path`, the
`NULL_SENTINEL` constant, the various regex constants.

**Fix:** delete the five dead helpers. The remaining 12 live
helpers stay. **Risk: low.** **LoC: ~−260** (including their
docstrings). **Severity: medium** — biggest single LoC liability
in `core/`; the names mislead users to think there's an alternate
public-function path to upload assets, when the live path is
`Execution.asset_file_path()` + `upload_execution_outputs()`.

### 1.4 `DatasetMixin._dataset_table` raise-NotImplementedError property is dead

`mixins/dataset.py:72-75`:

```python
@property
def _dataset_table(self) -> Table:
    """Get the Dataset table. Must be provided by host class."""
    raise NotImplementedError
```

`DerivaML.__init__` inherits `DatasetMixin`, but `DerivaML` also
defines `_dataset_table` as a real property in `base.py:778-780`:

```python
@property
def _dataset_table(self) -> Table:
    return self.model.schemas[self.model.ml_schema].tables["Dataset"]
```

The mixin's NotImplementedError property fires only if someone
constructs a class that inherits from `DatasetMixin` but not from
`DerivaML` — which is impossible in practice because the mixin
demands `self.model`, `self.ml_schema`, `self.pathBuilder`,
`self.s3_bucket`, `self.use_minid`, none of which are init'd by
the mixin. The pattern was meant to advertise the "host class
must provide" contract, but the body never fires.

**Fix:** delete the property body, keep only the class-body
type annotation `_dataset_table: Table` if a static signal is
wanted (matches the pattern used in the same file for `model`,
`ml_schema`, etc. at lines 64-70). **Risk: trivial.** **LoC: −5.**
**Severity: low.**

### 1.5 `Execution.asset_file_path` docstring drift in `core/mixins/asset.py:443`

`asset_record_class`'s docstring example at line 443:

```python
>>> path = exe.asset_file_path("Image", "scan.jpg", metadata=record)  # doctest: +SKIP
```

This is a docstring example, not a function call — the function
itself is fine. It references a method that exists. **No fix.**
Listed for completeness.

### 1.6 `core/validation.ValidationResult.__repr__` uses Unicode emojis

`core/validation.py:147, 150, 162` — `__repr__` and `__str__`
include the emojis `✓`, `✗`, `⚠`. Per workspace CLAUDE.md ("Only
use emojis if the user explicitly requests it"), these should be
plain text characters or ASCII equivalents.

**Fix:** swap to `OK / FAIL / WARNING` strings or similar. **Risk:
trivial** (only affects `__repr__` output). **LoC: ±0.**
**Severity: low.**

---

## Lens 2 — deriva-py interface usage

### 2.1 `core/` mostly uses the datapath API correctly

Every mixin uses `pb = self.pathBuilder(); pb.schemas[schema].tables[table]`
consistently. No raw ERMrest URL building in `core/` (unlike
`execution/state_machine.reconcile_with_catalog` flagged in Phase
3 §2.2 — which is in `execution/`, not `core/`). **Healthy.**

### 2.2 `core/mixins/vocabulary.py::lookup_term` uses three-tier server-side fallback

`vocabulary.py:243-268` — when a term isn't cached, the method
does:

1. Server-side filtered query for the name (`_server_lookup_term`,
   line 270-293).
2. If found, populate the full cache and return.
3. If not found, populate the full cache and check synonyms.
4. If still not found, raise.

This is correct shape — the server-side filter avoids materializing
the whole vocabulary when a single term is needed. **Healthy.**

### 2.3 `core/mixins/workflow.py::_find_workflow_rid_by_url` walks every workflow row

`workflow.py:278-295`:

```python
for w in workflow_path.path.entities().fetch():
    if w["URL"] == url_or_checksum or w["Checksum"] == url_or_checksum:
        workflow_rid = w["RID"]
        break
```

A linear scan of every Workflow row. For catalogs with hundreds
of workflows this is the same N+1-style cost the audit flagged in
Phase 2 dataset. The fix is server-side filter:

```python
matches = list(
    workflow_path.filter(
        (workflow_path.URL == url_or_checksum) |
        (workflow_path.Checksum == url_or_checksum)
    ).entities().fetch()
)
```

**Fix:** server-side filter. **Risk: low** (the catalog filter
syntax is well-trod). **LoC: −5.** **Severity: low** but compounds
when called per execution at `create_execution` time.

### 2.4 `core/mixins/asset.py::lookup_asset` issues two filtered fetches

`asset.py:299-359` — for each asset lookup:
1. `resolve_rid(asset_rid)` → server round-trip
2. `filter(RID == asset_rid).fetch()` → second round-trip
3. Look up association table for asset_type → third round-trip

`resolve_rid` already returns `ResolveRidResult` with a `datapath`
attribute that points at the entity; the second fetch is
redundant. Compare to `lookup_dataset` which uses `find_datasets()`
(also imperfect but at least one less round-trip).

**Fix:** use `rid_info.datapath.entities().fetch()` instead of a
new filter. **Risk: low.** **LoC: −3.** **Severity: low.**

### 2.5 `core/mixins/workflow.py::find_workflows` correctly pre-fetches association rows

`workflow.py:123-148` — pre-fetches the full
`Workflow_Workflow_Type` table once via
`_get_workflow_types_index` (line 59-76) to avoid an N+1 pattern.
This is the right shape; called out as **well-shaped** for the
audit record.

### 2.6 `core/mixins/asset.py::list_assets` pre-fetches asset_type associations

`asset.py:220-244` — uses one `attributes()` call to pre-fetch
the `{asset_rid → [type_name, ...]}` map, then a single asset
fetch, then in-memory join. **Healthy.**

---

## Lens 4 — Inconsistencies / duplication

### 4.1 `_dataset_table` defined in both `DerivaML` and `DatasetMixin`

Covered in §1.4. The mixin defines a raise-NotImplementedError
stub; `DerivaML` overrides it with a real property. Mostly a
double-definition issue.

### 4.2 The mixin "host class contract" pattern is inconsistent

`DatasetMixin` (`mixins/dataset.py:72-75`) defines `_dataset_table`
as `raise NotImplementedError`. Every other mixin uses class-body
type annotations only:

```python
class AssetMixin:
    model: "DerivaModel"
    ml_schema: str
    domain_schemas: frozenset[str]
    default_schema: str | None
    pathBuilder: Callable[[], Any]
    add_term: Callable[..., VocabularyTerm]
    apply_catalog_annotations: Callable[[], None]
```

Either pattern works in Python — the annotations are not enforced
at runtime, the `NotImplementedError` only fires when called.
Mixing the two patterns within the same subsystem is confusing.
The annotation form is lighter and more consistent.

**Fix:** drop the `_dataset_table` stub property, keep only the
annotation `_dataset_table: Table` in the class body. **LoC: −5.**
**Severity: low.**

### 4.3 `DerivaML.from_context` uses different docstring example syntax than other DerivaML constructors

`core/base.py:174-216` (`from_context`) uses a `Example::` block
with literal-block syntax (indented code block, no `>>> `
prompts):

```python
        Example::

            # In a script generated by Claude:
            from deriva_ml import DerivaML
            ml = DerivaML.from_context()
```

while every other method on `DerivaML` uses Google-style `Example:`
+ `>>> ` doctest blocks (which the project runs via
`--doctest-modules`). The literal-block form is a different
restructured-text directive and doctest cannot collect it.

**Fix:** convert to the `Example: >>> ...` doctest form
(catalog-dependent → `# doctest: +SKIP`). **LoC: ±0.** **Severity: low.**

### 4.4 `DerivaML.workspace` uses lazy `_workspace` initialization; nothing else in `base.py` does

`core/base.py:802-859` — `workspace` is a property that lazily
constructs a `Workspace` instance on first access. Everything else
in `__init__` (catalog, model, cache_dir, working_dir) is set
eagerly. The lazy pattern is appropriate (workspace construction
touches SQLite and builds a local schema), but the asymmetry isn't
documented in the class docstring. The lazy import on line 818
re-imports `Workspace` on every property access — a minor cost,
but unnecessary.

**Fix:** move `from deriva_ml.local_db.workspace import Workspace`
to module top (or class-level), guarded by `TYPE_CHECKING` for the
forward reference. **LoC: ±0.** **Severity: low.**

### 4.5 Per-mixin docstring drift on `domain_schema` vs. `domain_schemas`

Five mixins say "`domain_schema`: str - name of the domain schema"
in their "This mixin requires the host class to have:" block:

- `mixins/asset.py:44`
- `mixins/dataset.py:47`
- `mixins/feature.py:40`
- `mixins/file.py:34`
- `mixins/path_builder.py:37`

The actual attribute is `domain_schemas: frozenset[str]` (plural,
declared in `DerivaML` at `base.py:120` and in each mixin's class
body at the corresponding `frozenset[str]` annotation). The
singular form is documentation drift from a pre-multi-schema era.

**Fix:** rewrite to "`domain_schemas`: frozenset[str] - names of
the domain schemas" in all five mixins. **LoC: ±0.** **Severity:
low.**

### 4.6 `DerivaML.__init__` accepts `domain_schemas: str | set[str] | None` but stores `frozenset[str]`

`base.py:222` — parameter is `str | set[str] | None`. After
`__init__` the attribute is `frozenset[str]` (assigned at line 354
via `self.model.domain_schemas`). The conversion happens inside
`DerivaModel.from_cached` (see `model/catalog.py`). The mixin
class-body annotation matches the stored type (frozenset). This
isn't drift, just a clear param-vs-attribute type difference; the
docstring at line 246 describes the input shape only. Worth
explicit doc: "Stored as `frozenset[str]`. Use `ml.domain_schemas`
to read."

**Severity: low.**

### 4.7 `Execution._from_registry` is "private" but is the canonical re-hydration entry point

`core/mixins/execution.py:495`:

```python
return Execution._from_registry(
    ml_object=self,
    execution_rid=execution_rid,
)
```

`Execution._from_registry` is the canonical way to rebuild an
`Execution` from the SQLite registry. The underscore prefix
suggests private, but the only caller is in a different module
(`core/mixins/execution.py` calling `execution/execution.py`).
Phase 3 audit §1.3 nearly deleted the method assuming it had no
callers (the audit closing inventory notes the bug:
"§1.3 — rejected after re-verification").

This is a private cross-module method whose name violates the
"private API stays in its module" convention. The name should be
either:

- `Execution.from_registry` (no underscore, declares cross-module
  contract), or
- `Execution._resume_helper` (keeps the underscore but renames
  to something less generic), or
- inline the body into `resume_execution`.

**Fix:** rename to `from_registry` and document as the canonical
re-hydration entry point. **Risk: low** (rename only).
**LoC: ±0.** **Severity: medium** — the false-negative that
escaped the Phase 3 audit was specifically caused by this naming
choice.

### 4.8 `core/mixins/workflow.py::lookup_workflow_by_url` linear scan

Covered in §2.3.

---

## Lens 5 — Simplification opportunities

### 5.1 Delete `core/validation.validate_vocabulary_terms` (§1.1)

Net: **−42 LoC**.

### 5.2 Delete `reconcile_pending_leases` call sites (§1.2)

Net: **−30 LoC src** + `execution/lease_orchestrator.py` (53 LoC)
becomes deletable.

### 5.3 Delete dead `core/upload_layout.py` helpers (§1.3)

Net: **~−260 LoC**. Five module-level functions whose names
overlap with live methods on `Execution`. Removes a tripwire
without affecting any live path.

### 5.4 Fix `domain_schema` → `domain_schemas` in mixin docstrings (§4.5)

Net: **±0 LoC**, doc-only. Five files touched.

### 5.5 Delete `_dataset_table` stub property in `DatasetMixin` (§1.4 + §4.2)

Net: **−5 LoC**.

### 5.6 Server-side filter in `_find_workflow_rid_by_url` (§2.3)

Net: **−5 LoC**, performance win.

### 5.7 Use `rid_info.datapath` in `lookup_asset` (§2.4)

Net: **−3 LoC**, one fewer catalog round-trip per asset lookup.

### 5.8 `core/validation.ValidationResult` should be Pydantic

`core/validation.py:90-168` declares `ValidationResult` as a
`@dataclass`. Per CLAUDE.md ("Class idiom choice"), classes that
are returned from user-facing methods (`validate_rids`) and that
might cross a serialization boundary (logged, printed) should be
Pydantic. The dataclass works today, but a future caller that
wants `result.model_dump()` for JSON output won't find it; the
current `__repr__` uses `print()`-friendly emojis but
`json.dumps(result.__dict__)` ignores the lists' types.

**Fix:** rewrite as `BaseModel` with the same field set. **LoC: ±0
to +5.** **Severity: low** but improves consistency with the rest
of the codebase.

### 5.9 Move lazy `Workspace` import in `DerivaML.workspace` (§4.4)

Net: **±0 LoC**, microscopic perf win.

### 5.10 Add `__all__` to `core/` modules that lack it

Currently with `__all__`:
- `core/__init__.py`, `core/mixins/__init__.py`
- `core/definitions.py`, `core/ermrest.py`, `core/logging_config.py`
- `core/validation.py`

Lacking `__all__`:
- `core/base.py`, `core/config.py`, `core/connection_mode.py`,
  `core/constants.py`, `core/enums.py`, `core/exceptions.py`,
  `core/filespec.py`, `core/schema_cache.py`, `core/schema_diff.py`,
  `core/sort.py`, `core/upload_layout.py`, `core/async_helpers.py`,
  `core/pd_utils.py`, `core/catalog_stub.py`
- All 10 mixins under `core/mixins/`

**Recommendation:** add explicit `__all__` so `from
deriva_ml.core.X import *` is predictable. **LoC: +180** roughly.
**Severity: low** — quality-of-life win.

### 5.11 Rename `Execution._from_registry` → `Execution.from_registry` (§4.7)

Net: **±0 LoC**, ergonomics + audit safety win.

---

## Lens 6 — Maintainability

### 6.1 `DerivaML.__init__` is the long pole

`core/base.py:218-379` — 161 LoC, two private branches
(`_init_online`, `_init_offline`), 13 instance attributes set.
The class is correctly designed; the file is large because it
houses every method `DerivaML` adds beyond the mixins. Worth
splitting `apply_catalog_annotations`, `create_vocabulary`,
`create_table`, `define_association`, `chaise_url`, `cite`,
`refresh_schema`/`pin_schema`/`unpin_schema`/`pin_status`/
`diff_schema`, `clear_cache`/`get_cache_size`/`list_execution_dirs`/
`clean_execution_dirs`/`get_storage_summary`, and `validate_schema`
out of `base.py`.

The cleanest split would be:

- `core/schema_management.py` (new) — `refresh_schema`,
  `pin_schema`, `unpin_schema`, `pin_status`, `diff_schema`.
  These five methods are tightly coupled to `SchemaCache` and
  `SchemaDiff`; today they live in `base.py` because they need
  `self.catalog` / `self.working_dir` / `self._mode`. Either move
  them to a new mixin (`SchemaCacheMixin`) or to a sibling file
  imported into `base.py`.
- `core/storage_management.py` (new) — `clear_cache`,
  `get_cache_size`, `list_execution_dirs`, `clean_execution_dirs`,
  `get_storage_summary`. Same shape.

This isn't urgent; flagged as Phase 4 candidate. **Severity:
medium** (file size, not correctness).

### 6.2 The "import deriva via importlib" pattern is dense

`core/base.py:25-40` and `core/upload_layout.py:48-60` and four
mixin files use:

```python
import importlib
_deriva_core = importlib.import_module("deriva.core")
_deriva_server = importlib.import_module("deriva.core.deriva_server")
DerivaServer = _deriva_server.DerivaServer
...
```

instead of the normal `from deriva.core import DerivaServer`. The
comment at `base.py:25` explains: "use importlib to avoid
shadowing by local 'deriva.py' files". The motivation is real
(a stray `deriva.py` in the user's project would shadow), but the
pattern produces 8-12 lines of import ceremony per file and a
private `_deriva_core` underscore-prefixed module-level name that
isn't a constant. Worth factoring into a single
`deriva_ml.core.deriva_imports` module that does the importlib
dance once and re-exports the names normally.

**Fix:** consolidate into one helper module. **LoC: −60 across
the six files.** **Risk: low.** **Severity: low.**

### 6.3 Docstring quality

Sampled 30 public methods across `DerivaML`, the 10 mixins, and the
data classes. The Google-format skeleton is consistently filled
in; `Example:` blocks are correctly marked `# doctest: +SKIP` for
catalog-dependent paths. **Healthy overall.** Specific issues:

- **`DerivaML.from_context` uses `Example::` literal-block
  instead of `Example:` + doctest syntax** (§4.3). One-off drift.
- **`AssetMixin.list_asset_executions`'s `asset_role` parameter
  accepts arbitrary string, not enum** (`mixins/asset.py:247`).
  The legal values are `"Input"` and `"Output"` (per
  `MLVocab.asset_role`'s rows in the catalog). Passing
  `"Output"` works; passing `"output"` silently returns zero
  records. Worth declaring as `Literal["Input", "Output"] | None`
  in the type hint.
- **`AssetMixin.list_assets` docstring says "Asset types are
  pre-fetched in a single query"** (line 184). True; this is one
  of the well-shaped methods. **No drift.**
- **`DatasetMixin._dataset_table.NotImplementedError` lies** —
  covered in §1.4.
- **`core/exceptions.py` docstrings carry `>>> raise X(...)
  # doctest: +SKIP` patterns** (lines 65, 86, 98, etc.). These
  doctest-skip the raise statement but the surrounding text and
  Args sections are accurate.

### 6.4 `__all__` discipline (covered in §5.10)

Only 4 of 30 `core/` modules have `__all__`.

### 6.5 Logger usage is consistent — good

Every `core/` module uses `get_logger(__name__)` or imports
`logger` from a module-level `get_logger` call. **Healthy.**

### 6.6 `core/validation.py` `__all__` lists a broken function

`validation.py:68-77` exports `validate_vocabulary_terms` which
is dead and buggy (§1.1). Removing it removes a tripwire.

### 6.7 `core/upload_layout.py` is large and partly dead

649 LoC; ~30% of the surface is the dead module-level functions
(§1.3). The live surface (12 helpers) is healthy. **Severity:
medium** (covered under §1.3 / §5.3).

### 6.8 Naming: `_workspace` (private cached) vs. `workspace` (property), `domain_schemas` (plural attribute), `_mode` (private), `mode` (public property)

The base class consistently uses the pattern
`_x` (internal) + `x` (public property). `_mode`/`mode`,
`_workspace`/`workspace`, `_execution`/`<no public form>`. The
mixins read `_mode` directly (e.g.
`mixins/execution.py:166, 458`) without going through the
public property — minor inconsistency but harmless.

**Severity: low.**

---

## Lens A — Legacy-user removal

### A.1 `core/ermrest.py` legacy aliases — KEEP (external consumers)

`core/ermrest.py:86-112` — `ColumnDefinition`, `KeyDefinition`,
`ForeignKeyDefinition`, `TableDefinition`. Each carries a "Alias
for ... from deriva.core.typed. This maintains backwards
compatibility with existing DerivaML code. New code should use
ColumnDef directly." docstring.

Cross-workspace grep confirms external consumers exist:

- `deriva-mcp/src/deriva_mcp/tools/schema.py:84, 195, 223` —
  imports `ColumnDefinition`, `ForeignKeyDefinition`,
  `TableDefinition` from `deriva_ml`.
- `deriva-ml-model-template/src/scripts/_cifar10_schema.py:45-46`
  imports `ColumnDefinition`, `BuiltinTypes`.
- `deriva-ml-model-template/src/scripts/_cifar10_assets.py:46`
  imports `UploadProgress`.

So the **aliases must remain**. But the "new code should use
ColumnDef directly" advice in the docstrings is misleading —
external code does not migrate, and `deriva-ml`'s own internal
files (`core/mixins/asset.py:28`, `core/mixins/dataset.py:38`,
`core/mixins/feature.py:25`) use `ColumnDefinition` consistently.
The aliases are the canonical names from external code's
perspective.

**Recommendation:** flip the docstrings to read "Canonical name
for `ColumnDef` in `deriva-ml`'s public API. External code uses
this name; internal `deriva-ml` code uses it via
`deriva_ml.core.definitions.ColumnDefinition`. `deriva.core.typed.ColumnDef`
remains the underlying ERMrest type for advanced/typed use."
**LoC: ±0.** **Severity: low** (purely cosmetic, but reduces
confusion for someone reading the alias section).

### A.2 `core/enums.BuiltinTypes` — KEEP, same as A.1

`core/enums.py:22-27` — `BuiltinTypes = BuiltinType`. Same
"backwards compatibility" justification, same external consumers.
Same recommended docstring flip.

### A.3 The "Status was deleted in Phase 2 Subsystem 1a" comment

`core/enums.py:56-60` — a docstring-comment block:

```python
# Note: the legacy `Status` StrEnum was deleted in Phase 2 Subsystem 1a.
# Use `deriva_ml.execution.state_store.ExecutionStatus` instead — it carries
# the canonical 7-state lifecycle (Created, Running, Stopped, Failed,
# Pending_Upload, Uploaded, Aborted) with title-case values that match the
# catalog Execution.Status column directly.
```

This is a legitimate "removed; here's where to go" marker. It
guides anyone grepping for `Status` in the codebase. But it's also
a smell that external code (specifically
`deriva-mcp/src/deriva_mcp/tools/execution.py:194` and
`deriva-ml-skills/skills/troubleshoot-execution/references/execution-lifecycle.md:310`)
still tries to `from deriva_ml.core.definitions import Status` and
will fail at import time.

**Recommendation:** the comment itself is fine. Phase 4 work item:
update `deriva-mcp/src/deriva_mcp/tools/execution.py` to use
`ExecutionStatus` (or accept that legacy `deriva-mcp` is being
phased out per CLAUDE.md). **Severity: low** (out of audit scope
for `deriva-ml`).

### A.4 `Execution._from_registry` (§4.7) — not legacy, but private-named

The Phase 3 audit closing inventory called this out: the underscore
suggests "private and deletable" but the method is the canonical
cross-module re-hydration entry. Recommend rename to a public name.

### A.5 `validate_vocabulary_terms` (§1.1) — broken, no consumers, delete

Per §1.1, this is exported-but-broken-but-unused. Delete.

### A.6 `reconcile_pending_leases` call sites (§1.2)

The stub itself is a legitimate Phase 3 retirement-with-marker.
The two call sites are now ceremony. Delete them.

### A.7 `core/base.py:1059` comment "resolve_rid, retrieve_rid moved to RidResolutionMixin"

A comment marker noting that two methods moved out of the file.
Useful archaeology, but the empty marker line "`# resolve_rid,
retrieve_rid moved to RidResolutionMixin`" plus the "Methods moved
to mixins:" block at `base.py:1677-1683` are accumulated
breadcrumbs from the original mixin extraction. Worth pruning once
the codebase has stabilized.

**Recommendation:** delete the trailing marker comments at
`base.py:1059, 1677-1683, 782`. **LoC: −10.** **Severity: trivial.**

### A.8 `core/base.py:1059` `pathBuilder, domain_path, table_path moved to PathBuilderMixin` comment

Same shape as A.7. **LoC: counted above.**

---

## Lens B — Privatization

Cross-workspace grep performed against
`/Users/carl/GitHub/DerivaML/{deriva-mcp,deriva-mcp-core,deriva-ml-mcp,deriva-ml-model-template,deriva-skills,deriva-ml-skills}/`.

### B.1 `BatchRidResult` (`mixins/rid_resolution.py:34-48`) — keep public

Used in `tests/core/test_rid_resolution.py` and re-exported via
`interfaces.py:81`. External grep: zero hits. Return type of
public `resolve_rids` method. **Keep public.**

### B.2 `STRICT_PREALLOCATED_RID_TAG` constant (`mixins/annotation.py:45`)

Tested directly (`tests/core/test_strict_preallocated_rid.py:11`).
External grep: zero. The constant is what the Bug E.2 annotation
machinery uses to flag asset tables. It is the annotation tag URI
— a string constant that's part of the catalog metadata wire
format. **Keep public** (annotation tag URIs are documented
external constants).

### B.3 `DISPLAY_TAG`, `VISIBLE_COLUMNS_TAG`, `VISIBLE_FOREIGN_KEYS_TAG`,
`TABLE_DISPLAY_TAG`, `COLUMN_DISPLAY_TAG` (`mixins/annotation.py:35-39`)

Same shape as B.2 — annotation tag URIs. Used internally by the
mixin. External grep: zero. The annotation builders in
`model/annotations.py` (ADR-0007) re-export these. **Keep public.**

### B.4 `VocabularyTermHandle` (`ermrest.py:249-326`) — keep public

Used as a return type from public `lookup_term` / `add_term`
methods. External grep: zero, but the type is a public return.
**Keep public.**

### B.5 `ValidationResult` (`validation.py:90-168`) — keep public

Return type of public `validate_rids`. Cross-workspace grep: zero
in src code. **Keep public** but covered under §5.8 (should be
Pydantic).

### B.6 `validate_rids`, `validate_vocabulary_terms`,
`validate_execution_config` (`validation.py:171, 350, 394`) —
mixed

- `validate_rids` — external consumer at
  `deriva-mcp/src/deriva_mcp/tools/catalog.py:716`. **Keep
  public.**
- `validate_vocabulary_terms` — broken (§1.1), zero consumers.
  **Delete.**
- `validate_execution_config` — used by `execution/runner.py` and
  `execution/base_config.py`. **Keep public** (internal use, but
  imported across subsystems).

### B.7 `PinStatus` (`schema_cache.py:39-58`) — keep public

Return type of public `pin_status` method on `DerivaML`. External
grep: zero. **Keep public.**

### B.8 `SchemaCache` (`schema_cache.py:61-204`) — keep public

Constructed in `core/base.py:303` (online init), `core/base.py:551`
(`refresh_schema`), `core/base.py:627` (`pin_schema`),
`core/base.py:697` (`diff_schema`), and `core/base.py:654`
(`unpin_schema`). Also constructed in `tests/core/test_offline_init.py`
and `tests/core/test_schema_pin.py`. External grep: zero outside
deriva-ml. **Keep public** — its construction signature
(`SchemaCache(workspace_root: Path)`) is a clean API that
consumers might want.

### B.9 `SchemaDiff` and the `AddedTable` / `RemovedTable` /
`AddedColumn` / `RemovedColumn` / `ColumnTypeChange` /
`AddedForeignKey` / `RemovedForeignKey` Pydantic models
(`schema_diff.py:18-160`) — keep public

Return type of public `diff_schema` and `pin_schema`. Tested
in `tests/core/test_schema_diff.py`. External grep: zero.
**Keep public.**

### B.10 `_compute_diff` (`schema_diff.py:204-310`) — already private

Underscore-prefixed. Used by `pin_schema` and `diff_schema` in
`base.py`. **Correct.**

### B.11 `CatalogStub` (`catalog_stub.py:25-41`) — keep public

Tested directly. Replaces `ErmrestCatalog` in offline mode. External
grep: zero. The class is a public no-op stand-in; users may
`isinstance(ml.catalog, CatalogStub)` check it. **Keep public.**

### B.12 `ConnectionMode` (`connection_mode.py:12-32`) — keep public

Re-exported from `deriva_ml.__init__`. Public param of
`DerivaML.__init__`. **Keep public.**

### B.13 `_is_system_schema`, `_get_domain_schemas`
(`constants.py:54-96`) — already private

Underscore-prefixed. **Correct.**

### B.14 `DRY_RUN_RID` (`constants.py:47`)

Used by `run_notebook.py:61` and internally. External grep: zero.
**Keep public** (it's a meaningful sentinel value).

### B.15 `MLTable` enum (`enums.py:101-127`)

Used internally in `core/mixins/file.py:127`. External grep: zero
in any of the six sibling workspaces; only `tools/validate_schema_doc.py`
imports it.

**Recommendation:** consider `_MLTable` (underscore-prefix) — it
is purely an internal enum. **Risk: low** (only one internal
caller). **LoC: ±0.** **Severity: low.**

### B.16 `LOGGER_NAME` (`logging_config.py:35`) — public constant

Listed in `__all__`. External grep: zero. **Keep public** (it's a
documented value, even if not externally consumed today).

### B.17 `_apply_logger_overrides` (`logging_config.py:202-213`) — already private

Underscore-prefixed. Used by `core/base.py:346`. **Correct.**

### B.18 `is_hydra_initialized` (`logging_config.py:57-77`)

Listed in `__all__`. External grep: zero. Used internally by
`configure_logging`. The function is genuinely useful as a
condition predicate.

**Recommendation:** Keep public; it's a meaningful public predicate.

### B.19 `_table_path`, `_domain_path` (`mixins/path_builder.py:75, 98`) — already private

Underscore-prefixed. **Correct.**

### B.20 `BuiltinTypes` alias (`enums.py:22-27`) — KEEP, external consumers

External consumers in `deriva-mcp/src/deriva_mcp/tools/schema.py`
and `deriva-ml-model-template/src/scripts/_cifar10_schema.py`.
**Keep public.**

### B.21 `rid_part`, `snapshot_part`, `rid_regex`
(`constants.py:105-112`) — module-level regex constants

External grep: zero. Used internally by `RID` type validation.

**Recommendation:** keep public — they are reusable regex
constants for anyone parsing RID strings. **LoC: ±0.**

### Lens B summary

Privatization candidates with concrete impact:

- `MLTable` enum (`enums.py:101`) → `_MLTable` (one internal
  caller, no external).

Everything else in `core/` is either already private or has
external consumers (or a strong "Keep public" signal as the return
type of a public method).

**Severity overall: low.** No external consumer of any privatized
symbol; the cleanup is mechanical.

---

## Lens C — Test coverage

### Per-file posture

| File | LoC | # tests | Posture | State-leakage risk |
|---|---:|---:|---|---|
| `test_hydra_zen_config.py` | 538 | ~30 | Pure unit + mocks (Hydra config patched). | None. |
| `test_vocabulary.py` | 289 | ~13 | Live-catalog; 24 `test_ml` refs. | Medium. |
| `test_schema_cache.py` | 217 | 14 | Pure unit (uses `tmp_path`). | None. |
| `test_file.py` | 199 | 5 | Live-catalog. Uses `deriva_catalog` fixture. | High (creates File rows, no fixture-level cleanup). |
| `test_schema_pin.py` | 195 | 8 | Mix: 4 offline + 4 live. | Low (offline) / Medium (live). |
| `test_offline_init.py` | 197 | 5 | Mix: 2 offline + 3 live. | Low / Medium. |
| `test_catalog_annotations.py` | 186 | 8 | Live-catalog; 16 `test_ml` refs. Idempotent. | Low — annotations replace cleanly. |
| `test_schema_diff.py` | 190 | 11 | Pure unit. No catalog. | None. |
| `test_connection_mode.py` | 133 | 7 | Mix: 4 pure unit + 3 live (catalog_manager.reset). | Low. |
| `test_rid_resolution.py` | 107 | 4 | Live-catalog (uses `catalog_with_datasets`). | Low (read-only). |
| `test_del_no_abort_terminal.py` | 89 | 5 | Pure unit (SimpleNamespace mocks). | None. |
| `test_sort.py` | 88 | 7 | Pure unit. | None. |
| `test_offline_mode_smoke.py` | 84 | 1 | Live smoke. | Low. |
| `test_strict_preallocated_rid.py` | 76 | 7 | Live-catalog. Self-cleaning. | Low. |
| `test_enums_modernize.py` | 70 | 9 | Pure unit. | None. |
| `test_catalog_stub.py` | 42 | 4 | Pure unit. | None. |
| `test_exceptions.py` | 34 | 3 | Pure unit. | None. |
| `test_basic_tables.py` | 14 | 1 | Live-catalog. One assertion. | Low. |
| `__init__.py` | 1 | — | Empty. | — |

**Mock-vs-live mix per module:**

- **Pure unit:** `schema_cache.py`, `schema_diff.py`, `enums.py`,
  `exceptions.py`, `sort.py`, `catalog_stub.py`, `del_no_abort_terminal`.
  All in `tests/core/`. **Excellent.**
- **Live catalog smoke:** `connection_mode.py`, `offline_init.py`
  (some), `offline_mode_smoke.py`, `schema_pin.py` (some).
  **Healthy.**
- **Live catalog full:** `vocabulary.py`, `file.py`,
  `catalog_annotations.py`, `rid_resolution.py`,
  `strict_preallocated_rid.py`, `basic_tables.py`. **Healthy.**
- **Hydra-zen config:** `test_hydra_zen_config.py` is dense and
  covers config integration without a real catalog.

### Coverage gaps

**C.1 `core/base.py`'s `clean_execution_dirs` / `clear_cache` /
`list_execution_dirs` / `get_storage_summary` have no tests.**

These four storage-management methods (256 LoC, `base.py:1355-1609`)
have zero test coverage. The methods walk the local filesystem and
call `shutil.rmtree` (destructive). The lack of coverage is a real
gap — a bug in `clean_execution_dirs(older_than_days=30)` could
silently delete data.

**Recommendation:** add `tests/core/test_storage_management.py`
with `tmp_path` fixtures covering happy path, exclude_rids,
older_than_days filter, and the error path (PermissionError on
unlink). **LoC: +200 tests.** **Severity: medium** — destructive
operations with zero test coverage.

**C.2 `DerivaML.from_context` has no tests.**

`base.py:172-216` reads a `.deriva-context.json` file. The
`_find_context_file` helper at `base.py:1693-1718` walks parent
directories. The "no context file found" error path is reachable
but untested.

**Recommendation:** add unit tests in `tests/core/test_base.py`
(currently absent) — happy path with `tmp_path` planted JSON,
walking-up-parents behavior, FileNotFoundError. **LoC: +60.**
**Severity: low.**

**C.3 `DerivaML.cite` and `DerivaML.chaise_url` have no tests.**

`base.py:950-1031` — URL construction methods. The
`isinstance(entity, str) and entity.startswith(...)` short-circuit
at line 1017 is non-obvious. The URL replacement logic at line 976
(`replace("ermrest/catalog/", "chaise/recordset/#")`) is brittle.

**Recommendation:** add unit tests with a mocked `catalog`
returning a known `server_uri`. **LoC: +40.** **Severity: low.**

**C.4 `core/mixins/feature.py::feature_values` `materialize_limit`
exception path has tests at `tests/feature/test_feature_values.py`**

Spot-checked the test file (not in `tests/core/`, but in the
sibling subsystem tests). Coverage exists. **No gap.**

**C.5 `core/mixins/asset.py::find_assets` is barely tested.**

The function (`asset.py:380-422`) has zero direct tests in
`tests/core/test_basic_tables.py` (the asset-related core test
file). Coverage may live in `tests/asset/`. Spot-check:

```
$ grep -rn "find_assets" tests/
tests/asset/...
```

Confirmed coverage in `tests/asset/`. **No gap in core, but the
gap-checking exercise found the `tests/core/test_basic_tables.py`
file has only **1 test** for the entire asset/feature/file
surface that `tests/core/` is supposed to cover.

**Recommendation:** rename `tests/core/test_basic_tables.py` to
something narrower (it covers only `create_asset`), or expand it
to cover the rest of the mixin surface that lives in
`tests/asset/`, `tests/feature/`, `tests/dataset/`. **LoC: ±0**
(rename or move).

### State-leakage warning (load-bearing finding)

**The `tests/core/` suite's risk profile:**

`tests/core/` has **no `conftest.py`**. All fixtures come from
the root `tests/conftest.py`. This is fine — the fixtures
(`test_ml`, `catalog_manager`, etc.) are session-scoped and the
existence-guard pattern added in PR #99 (the parent of this
branch) catches state-stale issues.

**Risks specific to `tests/core/`:**

(a) **`test_vocabulary.py` adds vocabularies named `CV1`, `CV2`,
`CV3`, `CV_Cache`, `CV_Syn`, `CV_RemSyn`, `CV_Handle`, etc. without
cleanup.** Each test calls `ml_instance.create_vocabulary(...)`
and creates 5-10 terms. The test_ml fixture resets the catalog
between tests via `catalog_manager.reset()`, so the vocabs are
gone — but if `reset()` ever fails to clear the new vocabularies
(they're in the domain schema, not the ML schema), the next test
sees them. The existence guards in `catalog_manager.ensure_*`
catch the case where the ML schema's tables are empty, but they
don't catch leftover domain-schema vocabulary tables.

**Recommendation:** add an autouse fixture in `tests/core/conftest.py`
that calls `catalog_manager.reset()` between tests in
`test_vocabulary.py`, or move to a per-test catalog reset. The
existence-guard system was added precisely for this kind of
"cleanup might fail silently" case. **LoC: +20.** **Severity:
medium** — the leakage risk is real if `reset()` ever has a bug.

(b) **`test_catalog_annotations.py` mutates the catalog-level
annotation surface on the test catalog.** Each test calls
`ml.apply_catalog_annotations(...)`. The annotations replace
cleanly, so test ordering doesn't matter — but the **test catalog
is left with a different `chaise_config` than a fresh catalog
would have.** Subsequent test files using `test_ml` see the
modified annotations. The existence-guard pattern doesn't track
annotation state.

**Recommendation:** unlikely to bite, but worth a comment in
the test file noting "annotations are session-state; this file
runs late in the test order intentionally." **LoC: ±0** (comment
only).

(c) **`test_file.py` creates File rows and asset types without
cleanup-on-failure.** The `file_table_setup` fixture has a
`clean_up()` method that does `tables["File"].delete()` — but if
that itself fails (DataPathException), the exception is caught
and printed. The catalog has stale File rows after a failed
cleanup; the next test sees them.

**Recommendation:** strengthen `file_table_setup` cleanup to
explicitly reset the catalog manager on cleanup failure. **LoC:
+5.** **Severity: low** (rare to hit in CI, but it does happen).

### Cost trade-off per test file

| File | Choice | Cost | Right? |
|---|---|---|---|
| `test_schema_cache.py` (14 tests, tmp_path) | Pure unit | < 1s | Yes |
| `test_schema_diff.py` (11 tests, no catalog) | Pure unit | < 1s | Yes |
| `test_enums_modernize.py` (9 tests, pure) | Pure unit | < 1s | Yes |
| `test_sort.py` (7 tests, pure) | Pure unit | < 1s | Yes |
| `test_catalog_stub.py` (4 tests, pure) | Pure unit | < 1s | Yes |
| `test_exceptions.py` (3 tests, pure) | Pure unit | < 1s | Yes |
| `test_del_no_abort_terminal.py` (5 tests, mocked) | Pure unit | < 1s | Yes |
| `test_hydra_zen_config.py` (~30 tests, mocked) | Pure unit | < 1s | Yes |
| `test_basic_tables.py` (1 test, live) | Live | < 10s | Yes |
| `test_connection_mode.py` (7 tests, mix) | Mix | < 10s | Yes |
| `test_offline_init.py` (5 tests, mix) | Mix | < 10s | Yes |
| `test_offline_mode_smoke.py` (1 test, live) | Live smoke | < 10s | Yes |
| `test_schema_pin.py` (8 tests, mix) | Mix | < 10s | Yes |
| `test_rid_resolution.py` (4 tests, live) | Live | < 10s | Yes |
| `test_strict_preallocated_rid.py` (7 tests, live) | Live | < 30s | Yes |
| `test_catalog_annotations.py` (8 tests, live) | Live | < 30s | Yes |
| `test_vocabulary.py` (~13 tests, live) | Live | 30s-60s | Yes — but uses many vocabs (a) |
| `test_file.py` (5 tests, live) | Live | 30s-60s | Yes — but cleanup-on-fail issue (c) |

**Test-runtime headline:** `tests/core/` is overall **fast** —
~70% of the test files run without a live catalog. The
state-leakage risks are limited to four files and are tractable.

### Streamlining recommendations

**C.S1 Move `test_basic_tables.py` (only 1 test) into
`tests/asset/` or `tests/core/test_asset_creation.py`.** 14 LoC,
1 test that arguably belongs in `tests/asset/`.

**C.S2 Add `tests/core/conftest.py`** with autouse fixtures for
vocabulary cleanup (a) and an explicit ordering hint for
annotation-mutating tests (b). **LoC: +30.**

**C.S3 Add `tests/core/test_storage_management.py`** for the four
storage-management methods (§C.1). **LoC: +200.**

**C.S4 Add `tests/core/test_base.py`** for `from_context`,
`cite`, `chaise_url`. **LoC: +100.**

**Net test LoC delta from §C cleanups:**
- Add: ~+330 LoC of test coverage for currently-uncovered surface.
- Move: ~−15 LoC (rename `test_basic_tables.py`).

---

## Lens D — Docs/spec/ADR/docstring sync

### D.1 Mixin docstrings say `domain_schema` (singular); attribute is `domain_schemas` (plural)

Covered in §4.5. Five files. Pure doc drift.

### D.2 Workspace-level `CLAUDE.md` mentions `LoggerMixin` that doesn't exist

`deriva-ml/CLAUDE.md:234`:

```
- `LoggerMixin`: Mixin providing `_logger` attribute
```

Cross-check:

```
$ grep -rn "class LoggerMixin\|LoggerMixin" src/
(no output)
```

The `LoggerMixin` class does not exist anywhere in the codebase.
The convention is to use `get_logger(__name__)` at module level.
The CLAUDE.md reference is stale documentation.

**Fix:** delete the `LoggerMixin` bullet from `CLAUDE.md`. **LoC:
−1.** **Severity: low** (CLAUDE.md is meta-doc; a small lie there
misleads only readers, not callers).

### D.3 Skill plugins reference deleted `Status` enum

External documentation drift:
- `deriva-ml-skills/skills/troubleshoot-execution/references/execution-lifecycle.md:310`
  shows `from deriva_ml.core.definitions import Status`.
- `deriva-mcp/src/deriva_mcp/tools/execution.py:194` does the
  same import and accesses `Status.pending`, `Status.running`,
  etc. — all non-existent.

The `core/enums.py:56-60` comment block correctly points to
`ExecutionStatus` as the replacement. The external skill plugins
need to be updated.

**Fix:** out of scope for this `deriva-ml` audit; flag for the
respective repositories (legacy `deriva-mcp` is being phased out
per workspace `CLAUDE.md`). **LoC: ±0 in deriva-ml.**
**Severity: low** (out of scope).

### D.4 `core/validation.py` docstring example for `validate_vocabulary_terms` would crash

Covered in §1.1.

### D.5 `core/base.py` carries an out-of-date docstring referring to "domain_schema (str)" as an attribute

`core/base.py:103`:

```
domain_schema (str): Schema name for domain-specific tables and relationships.
```

The actual attribute is `domain_schemas: frozenset[str]` (line
120). Same drift pattern as the mixin docstrings (§4.5). The
docstring at line 246 (the `__init__` Args section) does say
"`domain_schemas`: Optional set of domain schema names" — but the
**class** docstring at line 100-115 (under `Attributes:`) still
says "`domain_schema (str)`".

**Fix:** update the class Attributes block to `domain_schemas`.
**LoC: ±0.** **Severity: low** (consistency with mixin fix).

### D.6 `core/__init__.py` docstring example shows non-existent default catalog

`core/__init__.py:14-16`:

```python
>>> from deriva_ml.core import DerivaML, DerivaMLConfig
>>> ml = DerivaML('deriva.example.org', 'my_catalog')
>>> datasets = ml.find_datasets()
```

This `Example:` block isn't `# doctest: +SKIP` marked — but
catalog-construction can't succeed in doctest without
network/auth. The example would fail if `pytest --doctest-modules`
collected it. Check whether the module is excluded from doctest:

```
$ grep -n "doctest" pyproject.toml | head -10
```

The example needs `# doctest: +SKIP` per workspace CLAUDE.md
("Catalog-dependent examples must carry `# doctest: +SKIP` on the
first interactive line"). **Fix.** **LoC: +3 (one annotation per
line).** **Severity: low.**

### D.7 ADR-0006 (bag pipeline) accurately describes the production path

ADR-0006 (`docs/adr/0006-bag-oriented-data-movement.md`) defines
`BagCatalogLoader` as the unified producer. `core/` does not
write to bags; it reads via `pathBuilder()`. **No drift.**

### D.8 ADR-0007 (annotation builders public API) is consistent with `core/mixins/annotation.py`

`apply_annotations()` exists at `mixins/annotation.py:433-448` and
takes no required args (per ADR-0007's pinned contract). **No drift.**

### D.9 ADR-0008 (estimate_bag_size opt-out) — not relevant to core/

Skipping.

### D.10 `docs/user-guide/exploring.md` is accurate

Spot-checked: it says `ml.domain_schemas` (plural) is a
`frozenset[str]`. Matches code. **No drift.**

### D.11 `docs/configuration/overview.md` is accurate

Says `domain_schemas` (plural). Matches code. **No drift.**

### D.12 `core/validation.py::ValidationResult.__repr__` uses emojis

Covered in §1.6. CLAUDE.md ("Only use emojis if the user
explicitly requests it") drift; the file violates the project
convention.

### D.13 `core/mixins/path_builder.py::pathBuilder` return type docstring mismatch

`mixins/path_builder.py:62-63`:

```
Returns:
    datapath._CatalogWrapper: A new instance of the catalog path builder.
```

The annotated return type is `SchemaWrapper` (line 56). The
docstring says `_CatalogWrapper`. The actual return value is
`self.catalog.getPathBuilder()` which returns `_CatalogWrapper`.
The annotation lies; the docstring is correct. Fix the annotation
to match.

**Fix:** annotation `-> SchemaWrapper` → `-> _CatalogWrapper`,
or fix the import. **LoC: −1 + 1.** **Severity: low.**

---

## Persona summaries

### Senior software engineer

`core/` is healthy at its bones: the lifecycle of `DerivaML`,
`SchemaCache`, `CatalogStub`, the mixin composition, the
exception hierarchy, and `ConnectionMode` are all clean. The
mixins compose correctly because every mixin reads only state
that `DerivaML` initializes — there's no cross-mixin state
mutation. The two real concerns are (1) the post-Phase-3
follow-ups that should ride along with this audit (the
`reconcile_pending_leases` ceremony, the broken
`validate_vocabulary_terms`, the dead `upload_layout.py` helpers,
the singular-vs-plural `domain_schema` docstrings) and (2) the
size of `base.py` (1 718 LoC). The latter is a Phase 4
candidate; the former is a single targeted PR.

### Testing engineer

Test posture is strong: 70% of `tests/core/` runs without a live
catalog, the schema_cache / schema_diff / catalog_stub /
connection_mode / sort / del_no_abort_terminal tests are all
pure unit. The gaps are concentrated:
`base.py::clean_execution_dirs` and friends (256 LoC) have zero
coverage and do destructive filesystem operations; `from_context`,
`cite`, `chaise_url` have zero direct unit tests. The state-leakage
risk is small but real in `test_vocabulary.py` (creates many
domain-schema vocabularies without explicit cleanup-on-failure).
Adding a single `tests/core/conftest.py` with an autouse vocab
cleanup fixture would close the leakage hole.

### Technical writer

The big drift is in five mixin docstrings (`domain_schema` →
`domain_schemas`) — singular survived a plural cutover. The
workspace `CLAUDE.md` references a non-existent `LoggerMixin`.
`core/__init__.py`'s doctest example needs `# doctest: +SKIP`.
The "Alias for ... New code should use ColumnDef directly"
language in `core/ermrest.py` is misleading — external code uses
the aliases as the canonical names, and internal `deriva-ml` code
follows suit. The broken `validate_vocabulary_terms` docstring
example would crash on first call.

### ML-developer user (the workflow author)

The happy path discovers easily: `ml = DerivaML(host, cat)`,
`ml.find_datasets()`, `ml.find_workflows()`, `ml.create_execution(...)`.
Five `find_*` methods follow the same pattern (datasets,
workflows, executions, features, assets) — discoverable, consistent.
The `sort=` parameter is uniform across `find_*` per the
`core/sort.py` helper. Method naming follows the
`find_/lookup_/list_/create_/add_/delete_` convention without
gaps — the audit's earlier worry about discoverability is
unfounded; the mixins are well-named.

Two annoyances: (a) the `from_context` constructor's docstring
example uses different syntax than every other method's example
(literal block vs. `>>> ` doctest); (b) the `ValidationResult.__repr__`
returns emoji characters that don't render cleanly in non-UTF8
terminals.

### DBA

The catalog interaction patterns in `core/` are clean: the
datapath API is used everywhere, no raw ERMrest URL building
(the one offender in `execution/state_machine.reconcile_with_catalog`
is in a different subsystem). The `lookup_workflow_by_url` linear
scan (§2.3) is the only place a server-side filter would clearly
help — for catalogs with hundreds of workflows it's a real perf
cost at every `create_execution` call. The schema-cache /
schema-diff pin machinery is well-designed: atomic writes, drift
detection, drained-workspace pre-condition for refresh, all
testable. Concurrent-access safety is bounded by `SchemaCache`'s
single-writer assumption (the file is per-workspace; one process
per workspace is the deployed shape). The `reconcile_pending_leases`
no-op stub is harmless after Phase 3 retirement but the
unnecessary calls in `base.py` and `mixins/execution.py` could
add a small per-Workspace.execution_state_store() cost on
`DerivaML.__init__`.

---

## Ranked actions (1–N)

Ranked by `(impact × confidence) / cost`.

| # | Action | Risk | LoC | Files | Rationale |
|---|---|---|---:|---|---|
| 1 | **§1.1** Delete `validate_vocabulary_terms` from `core/validation.py` + `__all__` entry. The function is exported, documented, broken (calls non-existent `ml.list_terms`), and has no callers. | low | −42 | `core/validation.py` | Tripwire removal: any caller would hit AttributeError on first use. |
| 2 | **§1.2** Delete the two `reconcile_pending_leases` call sites in `core/base.py:363-379` and `core/mixins/execution.py:478-491`. After Phase 3 audit closure, the function is documented as a no-op. The try/except ceremony around it is ten times the size of its body. Optionally delete `execution/lease_orchestrator.py` if no other callers remain. | low | −30 src, −53 if `lease_orchestrator.py` also goes | `core/base.py`, `core/mixins/execution.py`, optionally `execution/lease_orchestrator.py` | Post-Phase-3 follow-up; the stub exists to keep call sites compiling, but the call sites themselves are now dead weight. |
| 3 | **§1.3** Delete the five dead module-level helpers in `core/upload_layout.py`: `asset_file_path` (function), `upload_asset`, `upload_directory`, `upload_staging_root`, `normalize_asset_dir`. All five have zero callers across all six workspaces. The names `asset_file_path` and `upload_directory` survive only in docstrings and external docs; they confuse users into thinking there's a function-form path to upload assets. | low | ~−260 | `core/upload_layout.py` | Biggest single LoC liability in `core/`; removes a name-collision tripwire. |
| 4 | **§4.5** Fix the singular `domain_schema` → plural `domain_schemas` in five mixin docstrings (asset, dataset, feature, file, path_builder) and in `core/base.py:103` class-level Attributes docstring. | low | ±0 | 6 files | High user-impact: every mixin reader sees this; the singular form is wrong since multi-schema support landed. |
| 5 | **§4.7** Rename `Execution._from_registry` → `Execution.from_registry`. The underscore signals private; the only caller is cross-module in `core/mixins/execution.py:495`. The Phase 3 audit nearly deleted the method due to this naming. | low | ±0 | `execution/execution.py`, `core/mixins/execution.py` | Audit-safety improvement; reduces false-negative risk in future grep-based audits. |
| 6 | **§1.4 + §4.2** Delete the `raise NotImplementedError` stub property `_dataset_table` in `DatasetMixin`. The base class provides the real property; the stub never fires. Standardize on the class-body annotation pattern used by every other mixin. | trivial | −5 | `core/mixins/dataset.py` | Consistency cleanup. |
| 7 | **§5.8** Convert `ValidationResult` from `@dataclass` to Pydantic `BaseModel`. Provides `model_dump()` for JSON output and aligns with the CLAUDE.md "Class idiom choice" guidance for user-facing return types. | low | ±0 to +5 | `core/validation.py` | Quality-of-life consistency. |
| 8 | **§1.6** Remove the Unicode emojis (`✓`, `✗`, `⚠`) from `ValidationResult.__repr__`. Swap to plain text (`OK`, `FAIL`, `WARN`). Aligns with workspace CLAUDE.md "Only use emojis if the user explicitly requests it". | trivial | ±0 | `core/validation.py` | Project-convention compliance. |
| 9 | **§5.10** Add `__all__` declarations to the 26 `core/` modules that lack it. Currently only 4 of 30 modules declare an explicit export surface. | trivial | +180 | many | Public-surface clarity. |
| 10 | **§A.1 + §A.2** Flip the "Alias for ColumnDef. New code should use ColumnDef directly." docstrings in `core/ermrest.py:86-112` and `core/enums.py:22-27` to clarify that the aliases ARE the canonical public API; external code uses them, and `deriva-ml`'s internal code follows suit. | trivial | ±0 | 2 files | Docstring honesty about the surface external code depends on. |
| 11 | **§C.1** Add `tests/core/test_storage_management.py` covering `clear_cache`, `clean_execution_dirs`, `list_execution_dirs`, `get_storage_summary`. These are 256 LoC of destructive filesystem operations with zero coverage. | low | +200 tests | new file | Closes the biggest concrete coverage gap. |
| 12 | **§D.2** Delete the `LoggerMixin` line from workspace-level `CLAUDE.md`. The class doesn't exist; the line is stale documentation. | trivial | −1 | `CLAUDE.md` | Meta-doc honesty. |
| 13 | **§D.6** Add `# doctest: +SKIP` annotations to the catalog-construction example in `core/__init__.py:14-17`. The example references a network catalog and would fail if collected. | trivial | +3 | `core/__init__.py` | Doctest-collection hygiene. |
| 14 | **§2.3** Replace the linear scan in `WorkflowMixin._find_workflow_rid_by_url` with a server-side filter. For catalogs with hundreds of workflows this is a per-`create_execution` cost. | low | −5 | `core/mixins/workflow.py` | Performance win; consistent with the well-shaped pattern in `find_workflows`. |
| 15 | **§2.4** Replace `lookup_asset`'s second filter+fetch with `resolve_rid.datapath.entities().fetch()`. Saves one catalog round-trip per asset lookup. | low | −3 | `core/mixins/asset.py` | Performance + consistency win. |
| 16 | **§C.S2** Add `tests/core/conftest.py` with autouse vocabulary cleanup fixture (closes state-leakage risk in `test_vocabulary.py`). | low | +30 tests | new file | Closes a latent state-leakage hole. |
| 17 | **§C.4 + §C.2** Add `tests/core/test_base.py` for the `DerivaML.from_context`, `cite`, `chaise_url`, `is_snapshot`, `catalog_snapshot` methods. Currently zero direct coverage. | low | +100 tests | new file | Coverage gap for the most-likely-called methods. |
| 18 | **§B.15** Privatize `MLTable` enum (`core/enums.py:101`) → `_MLTable`. Only one internal caller (`core/mixins/file.py:127`); no external consumers. | low | ±0 | 2 files | Surface-minimization cleanup. |
| 19 | **§6.2** Factor the importlib-shadowing dance into a single `core/_deriva_imports.py` module that re-exports the deriva symbols. Currently the same 8-12 lines of import ceremony are copy-pasted across six `core/` files. | medium | −60 | 6 files | Reduces boilerplate; one place to maintain when deriva-py adds/removes symbols. |
| 20 | **§D.13** Fix `pathBuilder` return-type annotation `-> SchemaWrapper` vs. docstring `_CatalogWrapper` mismatch in `core/mixins/path_builder.py:56-73`. | trivial | ±0 | `core/mixins/path_builder.py` | Type-vs-docstring drift fix. |

Items 1+2+3 are the highest-leverage bundle — a single cleanup PR
removing ~330 LoC of post-Phase-3 cruft plus dead helpers. Items
4+5+6+7+8+10+12+13+20 are a single docstring/cosmetic PR. Item 9
is mechanical. Items 11+16+17 should bundle into a "core
test-strengthening" PR. Item 14, 15, 18 each warrant their own
narrow PR with focused review. Item 19 is Phase-4-scope.

---

## Follow-up scope (Phase 4 candidates)

### 4.A `core/base.py` structural split

`DerivaML` is 1 718 LoC carrying 4 concerns: (a) construction and
lifecycle (~400 LoC), (b) schema-cache management
(`refresh_schema` / `pin_schema` / `unpin_schema` / `pin_status` /
`diff_schema`, ~200 LoC), (c) storage management (`clear_cache` /
`clean_execution_dirs` / `list_execution_dirs` /
`get_storage_summary`, ~260 LoC), (d) catalog-level operations
(`create_vocabulary` / `create_table` / `define_association` /
`chaise_url` / `cite` / `apply_catalog_annotations` /
`validate_schema`, ~500 LoC). After actions 1-3 trim post-Phase-3
cruft, the natural split is into `core/base.py` (a), a new
`SchemaCacheMixin` or `core/schema_management.py` (b), and a new
`StorageManagementMixin` or `core/storage_management.py` (c).
(d) probably stays in `base.py` because each method is small and
the concern is narrow.

### 4.B `core/upload_layout.py` is in the wrong subsystem

After deleting the five dead module-level helpers (§1.3), the
remaining 12 live helpers in `upload_layout.py` are all consumed
by `execution/` and `schema/`. The module belongs in `execution/`
or in a new `upload/` subsystem. Moving it out of `core/` clarifies
the "what is foundational vs. what is execution-specific"
boundary.

### 4.C The mixin "host class contract" is implicit

Mixins declare class-body type annotations for the attributes
and methods they expect the host class to provide (e.g.
`mixins/asset.py:55-61`), but Python doesn't enforce the
contract. A `Protocol` based on the
existing `interfaces.DerivaMLCatalog` could be the canonical
mixin contract — every mixin asserts
`def __init_subclass__(cls): assert issubclass(host, DerivaMLCatalog)`
or similar. This is Phase-4-scope (protocol-based mixin contracts
need careful design).

### 4.D Cross-workspace `Status` migration in legacy `deriva-mcp`

`deriva-mcp/src/deriva_mcp/tools/execution.py:194` imports
`Status` from `deriva_ml.core.definitions`. The import fails
because `Status` was deleted in Phase 2. The skill plugin
`deriva-ml-skills/skills/troubleshoot-execution/references/execution-lifecycle.md:310`
shows the same pattern in documentation. Legacy `deriva-mcp` is
being phased out per workspace `CLAUDE.md`, so this is "wait for
the cutover to complete" rather than a fix. Worth flagging
explicitly so the migration doesn't ship broken.

### 4.E Workspace SchemaCache concurrency

`SchemaCache` writes are atomic via `os.replace` (good), but
reads are not coordinated with writes. Two processes opening the
same workspace simultaneously will both load whichever cache file
exists at their respective `load()` time; a write between them
isn't fatal (the loser's view is just stale), but the drift-warning
log message may fire spuriously. The current single-writer
assumption is documented; if multi-process workspace access is on
the roadmap, the cache needs a lock or version stamp.
