Migration notes — corrections (2026-06-03)

**Documentation backfill for previously under-documented public-API breakages.** The
changes below already shipped in earlier releases but were missing from (or only
partially covered by) these notes. They are recorded here because they break
external callers — notably domain subclasses of `DerivaML` such as `EyeAI` — and a
consumer migrating across these versions needs them.

| Change | Shipped in | Migration |
|--------|-----------|-----------|
| `DatasetBag.denormalize_as_dataframe(...)` renamed to **`DatasetBag.get_denormalized_as_dataframe(...)`** | ~v1.30.6 (the denormalize sugar-method refactor; old names removed) | Rename the call. Arguments are unchanged: the positional `include_tables` list and the `include_tables=`, `row_per=`, `via=`, `selector=` keywords all carry over. The dict variant is likewise `get_denormalized_as_dict(...)`. There is no deprecation shim — the old name raises `AttributeError`. |
| Public `DerivaML.domain_path` property removed; replaced by the **`_domain_path()` method** | mixin refactor (path-builder accessors moved to `PathBuilderMixin`) | Change `self.domain_path.<Table>...` to `self._domain_path().<Table>...` (note the parentheses — it is now a method). `_domain_path(schema=None)` returns the path builder for the domain schema (defaulting to `default_schema`); pass a schema name for a non-default one. |
| `check_auth` removed — **also from the `DerivaML.__init__` constructor**, not only from `DerivaMLConfig` | v1.37.x auth refactor (see the existing 1.37.0 note) | The existing note says to drop `check_auth` from hydra-zen configs / `DerivaMLConfig`. It is *also* gone from the `DerivaML(...)` constructor signature: subclasses that accepted `check_auth` and forwarded it via `super().__init__(check_auth=...)` now raise `TypeError`. Remove the parameter and the forwarded keyword. |
| `DerivaML.__init__` gained a **`mode` parameter** (`ConnectionMode`, default `online`) | the mode-branched `__init__` change | Non-breaking addition. `mode=ConnectionMode.online` (the default) preserves prior behavior; `mode=ConnectionMode.offline` (or the string `"online"`/`"offline"`) stages writes locally. Subclasses overriding `__init__` may forward `mode` through to `super().__init__(...)` to expose offline mode. |

Version 1.39.0

**Breaking-API change shipped as a minor bump.** The four ways to upload execution outputs (`Execution.upload_execution_outputs`, `Execution.upload_outputs`, `ExecutionSnapshot.upload_outputs`, `DerivaML.upload_pending`) collapse into one per-execution method and one batch method. Callers of the removed methods must migrate at upgrade — there are no deprecation shims. Major-version (`v2.0.0`) is deferred until the unified surface has more bake time. See ADR-0009 for the rationale and the two latent bugs fixed.

Migration table:

| Old                                                       | New                                                                              |
|-----------------------------------------------------------|----------------------------------------------------------------------------------|
| `exe.upload_execution_outputs(clean_folder=, progress_callback=)` | `exe.commit_output_assets(clean_folder=, progress_callback=)` (returns UploadReport now, not dict) |
| `exe.upload_outputs(retry_failed=)`                       | `exe.commit_output_assets()` (retry_failed was a no-op; removed)                 |
| `snap.upload_outputs(ml=, retry_failed=)`                 | `ml.resume_execution(snap.rid).commit_output_assets()`                           |
| `ml.upload_pending(execution_rids=, retry_failed=)`       | `ml.commit_pending_executions(execution_rids=, clean_folder=False)`              |
| `deriva-ml-upload --retry-failed`                         | (removed; flag was a no-op)                                                      |
| `deriva-ml-upload` (default: no folder cleanup)           | `deriva-ml-upload --clean` (explicit opt-in to clean working folder)             |

**Bugs fixed by the unification:**
- CLI-uploaded executions now correctly transition to `Uploaded` status (were stuck `Stopped`).
- `exe.upload_outputs()` callers now get asset descriptions written and Upload_Duration recorded (were silently skipped).

Both bugs were present in v1.37.x but only reachable via the legacy methods that v1.39.0 removes.

**Also in this release** (post-v1.37.14, folded into v1.39):

- **`feat(asset,dataset)`: write-through description setters** (#221, closes #70). `Asset.description` and `Dataset.description` now use a `@property` / `@setter` pair that persists assignments to the catalog row, mirroring the symmetric pattern already in place on `Workflow` and `ExecutionRecord`. `update_field_in_catalog` in `execution/_helpers.py` gained an optional `schema_name` parameter so the same helper now serves both ML-schema (Workflow / Execution / Dataset) and domain-schema (Asset) callers.
- **`fix(execution)`: Output_File directional tag + Execution_Execution exclusion** (#220). `bag_commit._add_asset_rows_to_bag` now auto-adds the `Output_File` directional Asset_Type to every asset uploaded after an execution — restores the public-API contract that every execution-linked asset carries an Input or Output role. `find_asset_execution_tables` excludes `Execution_Execution` (which is an execution-to-execution association, not an asset). Adds a "How execution-asset roles work" section to `docs/user-guide/executions.md` and a 5-test `test_asset_role_contract.py` regression suite.
- **`refactor(execution)`: public-API surface extracted to `asset_upload.py`** (#219, audit P1 Ex-god, third sweep). `upload_execution_outputs` and several sibling methods moved out of `execution.py` into the asset-upload module. Brings the `execution.py` running total to ~600 LOC removed across the three Ex-god sweeps.
- **`docs(audits)`: 2026-05-22 audit status summary** (#218). New `docs/audits/2026-05-22-audit-status.md` running ledger — tracks which P0/P1/P2/P3 items from the 9 audit reports have shipped vs. remain open.
- **`refactor(execution)`: bag_commit_upload + update_asset_execution_table extracted** (#217, audit P1 Ex-god, second sweep). Second pass on the execution god-class — moves `_bag_commit_upload` and `update_asset_execution_table` out of `Execution` into `asset_upload.py`.

Version 1.37.14

- **`chore(execution)`: P1 sweep (part 1) — 9 audit findings** (#203). Correctness + coverage fixes across `execution/` subsystem; details in `docs/audits/2026-05-22-engineer-audit-execution.md`.

Version 1.37.13

- **`chore(model)`: P1 sweep — 10 audit findings** (#202). Correctness + cleanup + coverage fixes across `model/` subsystem.

Version 1.37.12

- **`chore(asset)`: P1 sweep — 7 audit findings** (#201). Correctness + cleanup + coverage fixes across `asset/` subsystem.

Version 1.37.11

- **`chore(catalog)`: P1 sweep — 6 audit findings** (#200). Docs + correctness + coverage fixes across `catalog/` subsystem.

Version 1.37.10

- **`chore(dataset)`: P1 sweep — 2 correctness bugs + 1 cleanup** (#199). Fixes a typo in `_version_snapshot_catalog` and rebuilds the `add_dataset_members` cycle check on a real graph walk; deletes two retired `DatasetBag` tombstones.

Version 1.37.9

- **`chore(schema)`: P1 sweep — 7 audit findings** (#198). Threads the `schema` parameter through `generate_annotation`, sorts iteration, honours `use_hatrac`; backfills docstrings on 3 public functions; adds direct unit tests for `asset_annotation` and `generate_annotation`.

Version 1.37.8

- **`chore(feature)`: P1 sweep — 3 audit findings** (#197). FK-classification coverage and docstring fixes.

Version 1.37.7

- **`chore(core)`: P1 sweep — 15 audit findings** (#196). Sweep across `core/` covering exceptions, type hints, dead-code removal, and coverage.

Version 1.37.6

- **fix-pack 2: eight documentation P0s** (#195). Eight documentation-only P0s from the technical-writer audit — corrected examples, missing args, drifted references.

Version 1.37.5

- **Six correctness P0 fixes from the 2026-05-22 pre-release audit** (#193). (1) `Workflow._github_url` now treats any non-empty `git status --porcelain` output as dirty, not just staged-modified files — fixes silent provenance corruption for repos with unstaged, untracked, deleted, or renamed files. (2) `FileSpec.create_filespecs` now reports each file's actual length under a directory walk (was reporting the parent directory's stat size). (3) `DatasetBag.find_features(None)` no longer crashes with TypeError; now delegates to the deduped catalog walk. (4) `FeatureRecord.select_majority_vote(column=None)` auto-detect works on the single-term-feature happy path (was crashing with `TypeError: 'set' object is not subscriptable`). (5) `AssetSpecConfig` now mirrors `AssetSpec` field-for-field — the hydra-zen surface had been silently missing `asset_role`. (6) Adds integration coverage for `localize_assets` (350-LoC public function, previously had zero end-to-end tests).
- Adds nine pre-release audit reports under `docs/audits/` documenting the 382 findings across the four lenses (test coverage, code duplication, logic clarity, docstring completeness) and a follow-on technical-writer audit.

Version 1.37.4

- **New: `validate_config_file` and `bootstrap_config` config APIs** (#192). Reusable validation surface plus a bootstrap helper that resolves the workspace `.config.yaml` if it exists.

Version 1.37.3

- **`split_dataset` takes the caller's Execution** (breaking, #191). The function no longer opens its own execution internally; callers must wrap the call in their own `ml.create_execution(...)` context and pass `exe` as the third positional argument. The CLI (`deriva-ml-split-dataset`) opens its execution on the caller's behalf and is unaffected. Rationale: deriva-ml never invents a workflow on the caller's behalf — the caller's workflow is the durable provenance record for the splitting decision. See `src/deriva_ml/dataset/split.py` module docstring for the canonical call shape.
- **`fix(local_db)`: honest `estimated_row_count` when `row_per` is downstream of anchor** (#190).

Version 1.37.2

- **`fix(local_db)`: stateless `PagedFetcher`; consolidate denormalize design doc** (#189).

Version 1.37.1

- **`fix(local_db)`: hydrate `PagedFetcher` dedup state from engine on first access** (#188).
- **Audit fix-pack: test-suite resync + cross-repo bug-fix workflow docs** (#187). Resyncs 19 tests against the current deriva-py pin (drops removed `getCatalogSchema(refresh=True)` calls, replaces removed `BagDataSource(asset_localization=...)` kwarg, deletes stale `test_online_drift_warning`), migrates 3 denormalize tests to the new `row_per=` contract, and captures a new "Cross-repo bug fixes (deriva-py ↔ deriva-ml)" subsection in CLAUDE.md.

Version 1.37.0

- **`fix(schema)`: apply `acl_config` AFTER `create_ml_schema`** (#186). Production bug — every catalog created by deriva-ml since this ordering took shape had been silently missing `row_owner_guard` on its deriva-ml tables. Effect: non-curator users hit HTTP 403 the first time they PATCHed `Execution_Metadata` (asset uploads do this). Masked because tests run as catalog-owner with full ACLs.
- **Typed exceptions for `find_association` failure modes** (breaking for `except DerivaMLException:` catchers, #180). New: `DerivaMLAssociationAmbiguous` and `DerivaMLAssociationNotFound` replace bare `DerivaMLException` raises in `Table.find_associations`. Existing `except DerivaMLException:` blocks still work; narrower handlers should switch to the new types.
- **`fix(execution)`: warn before `download_asset` overwrites different bytes** (#184/#181).
- **`fix(execution)`: silence misleading dry-run exit warning** (#183/#177).
- **`fix(execution)`: move upload staging out of cache root** (#182/#178). Staging now lives at `<workspace>/staging/<execution_rid>/`, not in the cache root — prevents stale staging files from leaking across runs.
- **`fix(execution)`: RID-key the per-asset download path** (#179).
- **`fix(denormalize)`: feature-assoc tables are transparent** (#176/#174). Denormalization now treats feature association tables (Execution_*) as transparent links rather than terminal sinks. Closes the "feature columns missing from denormalized output" class of bugs.
- **`fix`: `SchemaCache` race when two processes write concurrently** (#175/#173).
- **`fix(execution)`: `list_execution_children`/`parents` propagate duration fields** (#172).
- **`fix(feature)`: `select_by_execution` returns `None` on no match, not raise** (#171). Mirrors `select_by_workflow`'s no-match semantics; `feature_values` silently skips target rows whose record group has no match rather than aborting the whole query.
- **`feat(dataset)`: adapters always surface element RID in return tuple** (#169). Both `as_torch_dataset` and `as_tf_dataset` now yield `(image, target, rid)` rather than `(image, target)`. Lets training loops correlate per-batch predictions back to dataset rows.
- **`fix(dataset)`: adapters resolve asset paths against canonical BDBag layout** (#167). Plus a new "RID opacity rule" added to CLAUDE.md: a RID's only valid operation is equality comparison; never parse, slice, regex, or `startswith` on it, never compare across catalogs, never sort client-side by RID.
- **`docs(reference)`: backfill `schema.md` to match `create_schema.py`** (#170). The reference is now described as canonical per `reference/README.md`.
- **`fix(execution)`: drop unsupported `upload_timeout` / `upload_chunk_size` from `run_model`** (#168).
- **`fix(execution)`: `__exit__` writes `Duration`; split into three phase columns** (#166). Execution duration is now broken into `Download_Duration`, `Execution_Duration`, `Upload_Duration` so the three phases are independently observable in catalog rows.
- **`refactor(auth)`: drop `check_auth` and the `get_authn_session` probe**. The `check_auth` keyword no longer exists on `DerivaMLConfig`; remove it from any hydra-zen configs.
- **B1 + B2: simplify schema cache; fix `find_features` duplicates**. `DerivaML._init_online` now delegates schema-freshness entirely to deriva-py's `ErmrestCatalog` (ETag revalidation + auto-invalidation on schema mutations through the same instance). `find_features()` no longer returns each feature multiple times when its association table has FKs to multiple tables.
- **T1: route all schema reads through deriva-py's `getCatalogSchema`**. Removes deriva-ml's local schema cache plumbing; the binding-layer cache is now the single source.
- **`docs(api)`: document `find_*` / `list_*` method naming convention** (#163). The convention is now load-bearing — `find_*` is schema-introspection/discovery with filtering; `list_*` is straightforward enumeration of "what's there" inside a known scope.

Version 1.36.5

- **`fix(execution)`: auto-tag input/output assets with `Input_File` / `Output_File`** (#165). Every asset linked to an execution is now tagged with `Asset_Role` automatically based on the direction of the link, restoring a regression in input-side tagging that had been silently broken since the previous lifecycle refactor.
- **`fix(schema)`: remove domain-specific terms from default `Workflow_Type`** (E5, #164). The bootstrap vocabulary set no longer seeds opinionated domain-specific terms like `VGG19`, `RETFound`, `Multimodal`, or `Embedding` — those belong to downstream projects, not the framework.

Version 1.36.4

- Docs: Phase 3 audit of `core/` subsystem and follow-on cleanup (#153, #154). 10 cleanup actions across `core/`.

Version 1.36.3

- **`refactor(schema)`: Phase 3 audit cleanup** (#158). 11 commits / 13 actions; −11.6K LoC. Stripped legacy schema-generation paths now superseded by the bag pipeline. No user-facing API surface change.

Version 1.36.2

- **`refactor(catalog)`: Phase 3 audit cleanup** (#156). 10 commits / 13 actions plus provenance wiring on the clone path.

Version 1.36.1

- **`refactor(core)`: Phase 3 audit cleanup** (#154). 10 commits / 10 actions; tightens core module structure ahead of the next round of feature work.

Version 1.36.0

- **Bag-pipeline migration complete** (#111, #109, #110). The catalog clone path now routes entirely through deriva-py's bag pipeline (`CatalogBagBuilder`, `BagCatalogLoader`); the legacy upload path is retired. Net: −1479 LoC across deriva-ml.
- **`feat(execution)`: bag-based `commit_execution` (per-execution path)** (#103). Plus URL-dedup bag commit and the deletion of the legacy upload path (#104).
- **`feat(clone-via-bag)`: terminal-tables for Execution/Workflow + policy-default merging** (#101).
- **`feat(catalog)`: nested-Dataset anchor expansion + vocab FULL default + schema-clone fixture** (#100).
- **`fix(dataset)`: `DatasetMinid` stops constructing `rid@None` when snapshot absent** (#106).
- **`refactor(execution)`: lean on upstream bag helpers, drop ~330 LoC** (#107). Plus #108 Phase-1 cleanup (~1479 LoC removed across dataset-bag cutover and logger consolidation).
- **Phase 2 cleanup sprint (Steps 1–10)** (#109) and **Phase 2 polish (Steps 13–16)** (#110): docstrings, naming, ADR-0007 (annotation-builders public API contract), apply-annotation guards.

Version 1.35.0

- **`feat(bag-migration)`: migrate deriva-ml onto `deriva.bag`** (#96). Foundation for the bag-pipeline arc completed in v1.36.0.
- **ADR-0006 + CONTEXT.md additions for bag-oriented data movement** (#95).
- **`refactor(dataset)`: rewire callers from `CatalogGraph` to `DatasetBagBuilder`** (#97).
- **`test(catalog)`: live-catalog integration tests for `clone_via_bag` (xfail-marked)** (#98).
- **`fix(local_db)`: turn off SQLite FK enforcement during `_populate_from_catalog`** (#99).
- **`fix(schema)`: regenerate `deriva-ml-reference.json` from fresh catalog dump** (closes #83, #92).

Version 1.34.0

- **Dataset dev versioning** (breaking). Datasets now use a two-state versioning model: released versions (citable, snapshot-pinned) and dev versions (mutable, between-release labels of the form `<release>.post1.devN`). Every mutation lands on a dev version; `Dataset.release()` is the only path to a released version. See `docs/adr/0003-dataset-dev-versioning-model.md` and `docs/user-guide/migration.md` for the full migration story.
- **`increment_dataset_version` renamed to `release`** (breaking). New signature: `Dataset.release(bump, description, execution=None)`. The old method is preserved as a private `_increment_dataset_version` for system-internal use only (e.g., catalog clone reinitialization).
- **`add_dataset_members`, `delete_dataset_members`, `add_dataset_type`, `add_dataset_types`, `remove_dataset_type` now land on dev** (breaking). Each call advances `.devN` rather than producing a released version. To mint a release after mutations, call `dataset.release(...)`.
- **`DatasetVersion` rebased on PEP 440** (breaking for some equality assertions). The wire format for released versions is unchanged (`"0.4.0"`); dev labels use PEP 440 post-release form (`"0.4.0.post1.dev1"`). String equality (`current_version == "1.0.0"`) no longer works — coerce explicitly: `str(current_version) == "1.0.0"`.
- **New: `Dataset.mark_dev`, `is_dirty`, `release_diff`, `compare_versions`.** Mark drift explicitly, detect whether the catalog has drifted since the last release, and compare any two versions.
- Documentation: new ADRs (0003 dev-versioning model, 0004 PEP 440 vocabulary, 0005 delivery sequence) and a CONTEXT.md vocabulary file.

Version 1.2.0

- Dataset versioning with semantic versioning. Note that the current dataset version does *NOT* have the current catalog values, but rather the values at the time the dataset was created. 
To get the current values you must increment the dataset version number.  Please consult online documentation for more information on dataset and versioning.
- Streamlined create_execution.  Now all datasets are automatically downloaded and instance variable has databag classes. You no longer need to explictly create dataset_bdbag. 
- Significant performance improvement on cached dataset access and initial download
- Automatic creation of MINID for every dataset download
- Added method to restore an existing execution from local disk.

Version 1.1.4
- Fixed error when creating DatasetBag on windows platform.

Version 1.1.1

- Removed restriction on nested datasets so that now any level of nesting can be accomidated.
- Fixed bug in nested dataset download.
- Added additional methods to DatasetBag to make it easear to explore datasets.
- Added `datasets` instance variable to Execution object which has Dataset objects for all of the datasets listed in the configuration.
- Added option to DatasetBag init to provide a dataset RID or a path.  If the dataset has already been loaded, or the dataset is nested, this will return the assocated DatasetBag object.

