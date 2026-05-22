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

