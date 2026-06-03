# deriva-ml Alignment Audit — 2026-05-30

Read-only, four-dimension alignment audit of the `deriva-ml` package:
docstring accuracy + examples, docstring/implementation consistency,
spec alignment, and docs currency. Findings come from 12 per-package
auditors plus one cross-cutting docs auditor. **No source or existing
docs were changed** — this report is the deliverable.

Totals: **31 HIGH**, **40 MEDIUM**, **63 LOW** (134 findings).

---

## 1. Headline

The codebase is structurally healthy — the implementations behind these
docstrings are correct, and most findings are *documentation* drifting
behind code that already shifted, not bugs. But the documentation
surface has a recurring, high-impact failure mode: **copy-pasteable
examples that will not run.** The HIGH bucket is dominated by examples
and protocol contracts that name parameters, methods, or column formats
the implementation no longer (or never did) accept — `split_dataset`
examples omitting the required `execution` argument, `Execution.add_files`
examples using `file_types=` instead of `dataset_types=`, the entire
`annotations.py` "Quick Start" calling a non-existent
`handle.set_annotation(...)`, the offline user-guide passing
`group_by=` to `restructure_assets`, and the executions user-guide
pointing readers at the renamed-away `list_nested_executions`. A
new user following the published docs would hit `TypeError`,
`AttributeError`, or `KeyError` on first contact.

The single most dangerous finding is not a doc bug at all but a latent
defect surfaced by the docstring/impl check:
**`core/validation.py:285` calls `dataset.list_versions()`, a method
that does not exist** (the real method is `dataset_history()`), inside a
broad `except Exception` that silently swallows the `AttributeError`.
The documented "validate the required version exists, raise on miss"
behavior therefore never runs — a bad `dataset_versions` requirement is
downgraded from a hard error to a warning. Fix that first.

The second-order theme is **renamed APIs leaving stale references
behind** (`.rid`→`dataset_rid`/`workflow_rid`, `list_nested_executions`→
`list_execution_children`, `add_workflow`→`_add_workflow`) and
**a denormalizer column-naming change** (underscore → dot `Table.column`)
that the Python docstrings were updated for but the CLI, the
`SelectionFunction` Protocol, and several examples were not. Three
interface-protocol contracts in `interfaces.py` describe behavior the
concrete implementations contradict (return types, a retired
`add_features` stub) — these are worth treating as correctness issues
because they are the published contract.

What's genuinely clean: the dataset, execution, and model
*implementations* themselves; the ADR set is mostly accurate (two
exceptions below); and the LOW bucket is overwhelmingly cosmetic
(repr-vs-str, key-count off-by-one, missing `Example:` blocks) rather
than misleading.

---

## 2. Findings by severity

### HIGH (31) — wrong or misleading

#### Docstring accuracy + examples

- **`dataset/split.py` — every `split_dataset` function-level Example omits the required `execution` arg.** Signature is `split_dataset(ml, source_dataset_rid, execution, *, ...)`; examples at lines 1319, 1325, 1357, 1438, 1449, 1461, 1470 call `split_dataset(ml, "28D0", test_size=0.2, ...)`. Copy-pasting any raises `TypeError: missing required positional argument 'execution'`. The module preamble (lines 41–50) correctly wraps it in `with ml.create_execution(config) as exe:`, so the file is internally inconsistent.
- **`dataset/split.py:154-159` — `SelectionFunction` Protocol docstring documents underscore columns; denormalizer emits dot-notation.** Says columns are `Image_RID`, `Image_Classification_Image_Class`; actual columns use `Table.column` (`dataset.py:1724`), and the split code reads `f"{element_table}.RID"` (split.py:853). A selector written against the documented names would `KeyError`.
- **`dataset/split.py:1423-1429` — concrete custom-selector example indexes a non-existent column.** `label_col = "Image_Class_Name"` then `df[label_col]` against dot-notation columns (`Image_Class.Name`) → `KeyError`.
- **`dataset/dataset_bag.py:~1367` — `as_tf_dataset` `targets=None` doc contradicts itself and the impl.** Args says the unlabeled element is "just the sample (not a tuple)"; the same docstring's Returns and `tf_adapter.py:152-153` both yield `(sample, rid)`.
- **`model/annotations.py:19-39, 358-359, 768` — module "Quick Start" calls `handle.set_annotation(Display(...))`, which doesn't exist.** `TableHandle` has no `set_annotation`; `hasattr(...)` is `False`. ADR-0007's real apply path is `table.annotations[Display.tag] = display.to_dict(); ml.apply_annotations()`. This is the canonical pattern for an externally-consumed public API.
- **`model/catalog.py:484` — `find_association` return type lies.** Annotated `-> tuple[Table, Column, Column]` and doc says "the two FK columns," but the body returns column *names* (strings) via `.name` (lines 514–518). Callers trusting the hint treat strings as `Column` objects.
- **`feature.py:30-32` — module "Typical usage" constructs `DiagnosisRecord(Diagnosis="benign", ...)`, but the generated FeatureRecord subclass has no `Diagnosis` field.** (Finding truncated in source data; the constructed field name does not exist on the record.)
- **`execution/execution.py:1995-2002` — `Execution.add_files` Examples use `file_types=`, not the real `dataset_types=`.** Underlying mixin (`core/mixins/file.py:67`) also uses `dataset_types`; examples raise `TypeError: unexpected keyword argument 'file_types'`.

#### Docstring / impl consistency

- **`core/validation.py:285` — `validate_rids` calls `dataset.list_versions()`, which does not exist (real method: `dataset_history()`).** The call sits in `try/except Exception`, so the `AttributeError` is swallowed and the function falls through to a generic "Could not verify version history" *warning*. The documented hard-error-on-missing-version behavior never runs. `list_versions` is grepped nowhere else in `src/`. **Highest-priority finding — a real latent defect, not just doc drift.**
- **`interfaces.py:986-998` — `DerivaMLCatalog.add_features` protocol advertises a working batch insert returning an int count, but the only concrete impl (`FeatureMixin.add_features`) is a retired stub that raises unconditionally.** A reader believes `ml.add_features(records)` works.
- **`interfaces.py:613-622` — `AssetLike.list_executions` protocol declares `-> list[dict[str, Any]]` ("records with Execution RID and Asset_Role"), but `Asset.list_executions` returns `list[ExecutionRecord]`.** Both the return type and the prose are wrong vs. the implementation.
- **`execution/execution.py:2152-2155` — `Execution.__enter__` doc claims dry-run `start_time` reads back as `None`.** The `start_time` property has no dry-run carve-out and raises `DerivaMLStateInconsistency` on the missing SQLite row (dry-runs never insert one). `runner.py:237-241` documents the opposite.
- **`execution/bag_commit.py:33-34` — module doc says the transient bag "is discarded after a successful load."** No code path deletes it post-load; the only `rmtree` (`asset_upload.py:656`) wipes a *pre-existing partial* bag before a fresh build. The sibling driver `bag_commit_upload()` (`asset_upload.py:614-619`) explicitly says the bag is *left in place* for post-mortem inspection.
- **`local_db/denormalize.py:30-32` — module doc assigns `source="slice"` to `DatasetBag.get_denormalized_as_dataframe`; the bag path actually uses `source="local"`.** `Denormalizer.__init__` resolves a bag to `self._source = "local"` (`denormalizer.py:164`); no code path sets `"slice"` for a bag. Confirmed by the design spec §2.

#### Spec / docs alignment

- **`docs/user-guide/executions.md:637` — tells readers to call `list_nested_executions`; it was renamed to `list_execution_children` in the R5.1 hard cutover with no alias.** Calling it raises `AttributeError`. (`execution_record.py:372`.)
- **`docs/user-guide/offline.md:333` — `restructure_assets` example passes `group_by=[...]`; the parameter is `targets=`.** `TypeError`. Every other example in the same doc uses `targets=`.
- **`docs/user-guide/offline.md:415, 431` — torch-adapter return shape documented as `(row_dict, target)` / `(image_tensor, {dict})`, omitting the trailing RID.** The adapter always returns a 3-tuple `(sample, target, rid)` (`torch_adapter.py:155`).
- **`docs/user-guide/offline.md:113 (and 258)** — doc describes a per-bag `feature_values()` cache that makes "subsequent calls fast."** That `BagFeatureCache` path was deleted in Stage 3b; the impl now calls `Denormalizer(self).feature_records(...)` fresh on every call (re-runs the full SQL join). Line 258's "reads from the bag's SQLite cache" is the same stale concept. (Line 258 is MEDIUM; 113 is HIGH.)
- **`docs/adr/0007-annotation-builders-public-api.md:73-75` — ADR pins the canonical Display tag as `tag:misd.isi.edu,2015:display`; the real `TAG_DISPLAY` is `tag:isrd.isi.edu,2015:display`.** The ADR says this value changes "only when deriva-py renames the tag," so a maintainer trusting the ADR writes the wrong URI. The same `misd` typo recurs at `schema/create_schema.py:641`.
- **`interfaces.py:986-998` — keeping a functional-looking `add_features` on the catalog protocol contradicts the feature-API-consistency spec**, which explicitly retires `ml.add_features` ("One write path: `exe.add_features(records)`") and lists `interfaces.py` as modified "to the three-method surface."
- **`dataset/dataset.py:2390-2418` — `Dataset.list_executions` declares `-> list["Execution"]` and doc says "Execution objects," but it returns `ExecutionRecord` objects** (via `lookup_execution`, annotated `-> "ExecutionRecord"`).
- **`dataset/dataset.py:~1619 — `Dataset.list_dataset_members` Example passes a RID positionally**, but the first positional param is `recurse: bool` (the RID is bound to `self`). `ml.list_dataset_members("1-abc123", recurse=True)` would map the RID to `recurse` *and* duplicate the keyword → `TypeError`. Looks copied from a DerivaML-mixin form.

### MEDIUM (40) — material but narrower

#### Docstring accuracy

- **`dataset/dataset.py:2037-2042` — `add_dataset_members` cycle case: doc/user-guide promise `DerivaMLCycleError`; code raises plain `DerivaMLException`.** The dedicated `DerivaMLCycleError` (exceptions.py:429) exists but is unused; `docs/user-guide/datasets.md:176` is therefore wrong.
- **`dataset/split.py:246-252` — `stratified_split` doctest uses underscore stratify columns** (`Image_Classification_Image_Class`, `Diagnosis_Label`); impl compares against dot-notation columns and would raise "Column not found."
- **`execution/execution.py:1216-1220` — `Execution.download_dataset_bag` Returns documents `.rid` and `.minid` attrs on the `DatasetBag`; neither exists** (the attr is `dataset_rid`; no `minid`). Same stale `.rid` pattern as `Workflow.rid`.
- **`execution/workflow.py:11` — module doc references `DerivaML.add_workflow()`, which doesn't exist;** dedup lives in private `_add_workflow` (`core/mixins/workflow.py:156`).
- **`model/catalog.py:748-751` — `lookup_feature` doc claims `DerivaMLTableNotFound`; `name_to_table` raises the broader `DerivaMLException`** (line 393). Distinct classes.
- **`model/denormalize_planner.py:162-167` — "Consumers" section names `local_db/denormalize.py:denormalize()`; the real entry point is `_denormalize_impl(...)`.** No `denormalize()` exists.
- **`local_db/denormalizer.py:816` — `describe()` Returns says "12 keys"; the return literal has 13** (incl. `warnings`). The method's own summary line and the user-guide say 13. The "design spec §5" cross-ref is also stale (now `docs/user-guide/denormalization.md §8.3.2`).
- **`dataset/bag_download.py:641, 667` — `get_dataset_minid` doc says cache key is `{rid}_{spec_hash[:16]}_{snapshot}`; code computes `{spec_hash[:16]}_{snapshot}` with no rid prefix.** Every other tier helper omits the rid.
- **`core/exceptions.py:7-40` — the ASCII hierarchy diagram omits two exported classes:** `DerivaMLFeatureNotFound` (under `DerivaMLNotFoundError`) and `DerivaMLMaterializeLimitExceeded` (under `DerivaMLValidationError`). Readers using the diagram as the catchable-exception catalog miss them.
- **`model/database.py:198` — `rid_lookup` annotated `-> DatasetVersion | None`, but no `None` path exists** (returns on hit, raises `DerivaMLDatasetNotFound` on miss). The doc even admits the `| None` is dead.

#### Docstring / impl consistency

- **`dataset/dataset.py:2495-2507` — `download_dataset_bag` doc describes `version` as optional with a "version set in the dataset" fallback; the signature makes it a required positional with no default and no fallback path.**
- **`dataset/dataset.py:2189-2190` — `delete_dataset_members` Raises claims an error "If any RID is invalid or not part of this dataset," but a valid element-type RID that isn't a member is a benign DELETE no-op** (the code comment at 2253–2258 says so).
- **`dataset/restructure.py:136-142` — `_build_dataset_type_path_map` doc says default selector returns `"unknown"` for no-types; code returns `"Testing"`.**
- **`execution/execution.py:1949-1952` — `Execution.create_dataset` Raises claims `DerivaMLExecutionError` "if the context is no longer active"; no such check exists in the call chain.**
- **`execution/runner.py:450-453` — `run_model` Returns says "Results are uploaded to the catalog" unconditionally; upload runs only when `not dry_run`.** The `dry_run` skip is broader than the param doc states.
- **`execution/workflow.py:363-364` — `setup_url_checksum` doc claims fallback "to a `file://` path"; no code path produces a `file://` URL** (git path builds a `/blob/` URL; no-git allow_dirty returns `''`).
- **`execution/workflow.py:367 vs 426-448` — version-resolution bullet only credits `DERIVA_MCP_VERSION` (Docker branch); the dominant local-git path sets version from `get_dynamic_version()`, unmentioned.**
- **`docs/user-guide/offline.md:258` — `restructure_assets` "reads feature values from the bag's SQLite cache"; that cache was deleted in Stage 3b** (now goes through the Denormalizer wide-table join). Same stale concept as offline.md:113.
- **`dataset/dataset_bag.py:195-219` — `DatasetBag.path` doc says the dir contains `data/`, `manifest-md5.txt`, *and the SQLite db*; the db lives in a separate `databases/` dir, sibling to the bag** (ADR-0006 confirms `bags/{checksum}/db/main.db`).

#### Spec / docs alignment

- **`dataset/dataset.py:1100-1106` — ADR-0003 says `release()` on a no-dev-row dataset raises `DerivaMLValidationError`; code (and doc) raise the broader base `DerivaMLException`.** A caller catching `DerivaMLValidationError` per the ADR misses it.
- **`dataset/split.py:1644-1648, 1717-1719 — the `split_dataset` CLI epilog and `--stratify-by-column` help still use underscore columns and the `Image_Classification` shorthand;** the Python API was updated (task #38) to `Image_Class.Name` / vocab table. CLI surface left behind → column-not-found / Rule-5 rejection.
- **`docs/user-guide/offline.md:59-72, 45 — documented bag layout puts the `.db` inside the bag root and shows a rid/version path (`.../bags/1-ABC1/2.0.0/`);** actual layout is checksum-based `{cache_root}/bags/{checksum}/Dataset_{rid}/` with a separate `databases/` dir (ADR-0006).
- **`docs/user-guide/denormalization.md:672-674 — contract section cites `docs/audits/2026-05-26-denormalize-audit.md` and dozens of SC-NN/RB-NN/TC-NN IDs; that file does not exist.** The audits dir only has 2026-05-22 engineer files. No cited identifier is resolvable.

#### Missing docstrings (convention)

- **`execution/runner.py:410-449 — `run_model`'s load-bearing `script_config` param is undocumented** (it takes precedence over `model_config` and changes dry-run handling). The whole docstring is also NumPy-style, violating the mandated Google style.
- **`execution/workflow.py:53-67 — `Workflow` class Attributes block omits public model fields `git_root` and `allow_dirty`.**
- **`model/catalog.py:232-233 — `DerivaModel.refresh_model` has no docstring at all** and silently mutates `self.model` (invalidating two caches).
- **`model/catalog.py — ~19 of ~21 public `DerivaModel` methods lack the required `Example:` block** (only `find_asset_execution_tables` and `asset_metadata_sorted` have one).

### LOW (63) — cosmetic / convention / low-risk

Representative cluster (full set covered by area auditors; grouped here by kind):

#### Doc accuracy — stale names / values, harmless

- `dataset/aux_classes.py:309` — version attr says "semantic versioning"; project moved to PEP 440 (ADR-0004).
- `dataset/dataset_bag.py:1061-1075` & `local_db/denormalizer.py:816` — "12-key" describe dict that actually has 13 keys (`warnings`). Recurs in three places.
- `local_db/denormalize.py:117` — `cache_age_seconds` doc lists a `source='bag'` mode that doesn't exist (only `local`/`catalog`/`slice`).
- `dataset/bag_builder.py:728,736` — references `DatasetLike.dataset_children`; real method is `list_dataset_children`.
- `execution/runner.py:118-120` — lists `list_execution_children(parent_rid)` as taking a RID arg; it's an instance method on `ExecutionRecord` taking only `recurse`.
- `execution/execution_record.py:15-17,103` — Example prints `ExecutionStatus.Running`; a `StrEnum` `print()` emits the bare value `Running`.
- `execution/execution.py:229` — class Attributes still says `datasets (list[DatasetBag])`; the property returns a `DatasetCollection` (R5.1 cutover).
- `execution/execution.py:1986-1988` — `add_files` Returns labeled `RID:` but returns a `Dataset`.
- `core/definitions.py:13-15,168` — frames `ColumnDefinition` et al. as "legacy/backwards-compat"; `ermrest.py:85-89` says both forms are canonical.
- `model/catalog.py:354-356` — calls `apply`/`catalog` "properties"; they're a method and an instance attr.
- `local_db/result_cache.py:334,280` — class doc calls keys "SHA-256 digests"; `cache_key()` truncates to the first 16 hex chars (64 bits) — material for collision reasoning.

#### Doc/impl consistency — narrow carve-outs

- `dataset/restructure.py:624-625, 686 — `missing="error"` and `type_selector` default docs omit the feature-vs-column path split and the `type_to_dir_map` indirection.
- `execution/asset_upload.py:1062-1069 — `download_asset` sets `AssetFilePath.file_name` to a full `Path`, violating the "bare name string" contract (harmless today: downloads aren't in the uploaded manifest).
- `dataset/dataset.py:172-186 vs 92 — class-level `description` attribute doc omits the setter's catalog write-through side effect.
- `local_db/workspace.py:489-493 — `cache_denormalized`'s `ignore_unrelated_anchors` is folded into the cache key but never forwarded; two calls differing only in this flag cache separately yet produce identical results. Same for `selector` (not accepted/forwarded).

#### Spec alignment — stale cross-refs

- `model/denormalize_planner.py:181-183` & `catalog.py:938` — cite `docs/design/deriva-ml-audit-2026-05-phase2-model.md`; moved to `docs/archive/2026-05-audit-working-drafts/`.
- `local_db/manifest_store.py:45` — spec §2.14 names table `execution_state__features`; code uses `execution_state__feature_records`.
- `docs/adr/0009-...md:79-87` — free-function `commit_output_assets` (`-> dict[...]`) shadows the public delegate (`-> UploadReport`); behavior reconciled, readability hazard only.

#### Missing docstrings / convention

- `dataset/aux_classes.py` — `DatasetSpec` Attributes omits `description` + `fetch_concurrency`; `DatasetHistory` omits `execution_rid`, `description`, `spec_hash`.
- `execution/base_config.py:115-121` — `BaseConfig.script_config` field undocumented.
- `execution/asset_upload.py` — five exported helpers (`set_asset_descriptions`, `save_runtime_environment`, `upload_hydra_config_assets`, `clean_folder_contents`, `update_asset_execution_table`) lack the required `Example:` block.
- `core/base.py:257-291` — `__init__` documents "raises ValueError" with no `Raises:` section; also misses `DerivaMLConfigurationError` from `_init_offline`. `DerivaML` class Attributes lists `configuration (ExecutionConfiguration)` but `__init__` always sets it to `None`.
- `core/validation.py:360-387` — `validate_execution_config` uses NumPy-style headers (repo mandates Google; mkdocstrings may misrender).

#### Doctest / rendering hygiene

- `dataset/dataset.py:2192-2198 & 829-831` — `current_version` shown as repr `<Version(...)>` vs str elsewhere; `@validate_call` stacked on a zero-arg property getter is a no-op.
- `execution/workflow.py:91-109,152-160` — only the first line of multi-line examples carries `# doctest: +SKIP`; follow-on lines run live with `workflow` undefined.
- `execution/workflow.py:391-394` & `bag_commit.py:128-130` — `Raises:` lists base `DerivaMLException` where the common/actual failure is the more specific `DerivaMLDirtyWorkflowError` / `DerivaMLValidationError`.
- `dataset/dataset_bag.py:351,1498-1505` — dangling internal-audit refs ("§6 inline comment gap #4", "Phase 3 §3.B split (audit-flagged)") leaking into published API docstrings.
- `execution/base_config.py:300-305` — `notebook_config` doc lists `workflow`/`model_config` default groups it doesn't actually seed.
- `docs/api-reference/deriva_model.md` & `deriva_ml_base.md` — page intros mismatch what mkdocstrings renders; `deriva_ml_base.md:4` has a "tha tthe" typo and targets `::: deriva_ml.core` instead of `deriva_ml.core.base`.

---

## 3. By-area summary

| Area | High | Medium | Low | Health note |
|------|------|--------|-----|-------------|
| `dataset/dataset.py` | 2 | 3 | 3 | Two examples/returns flat wrong (`list_dataset_members`, `list_executions`); rest are ADR/exception-class narrowing. |
| `dataset/split + bag_builder + restructure + aux` | 3 | 3 | 5 | Worst examples cluster: every `split_dataset` example unrunnable; underscore-vs-dot column drift in API, CLI, and Protocol. |
| `dataset/dataset_bag + bag_download + adapters + offline docs` | 4 | 4 | 4 | Offline user-guide is the most stale doc: wrong param, wrong tuple shape, deleted cache, wrong layout. |
| `local_db` (denormalize/denormalizer/cache/manifest/workspace) | 1 | 1 | 4 | One real source-mode mislabel; rest are key-count and stale-spec-name cosmetics. |
| `execution` (execution/runner/record/state) | 3 | 3 | 4 | `__enter__` dry-run claim + `add_files` example wrong; runner docstring style + undocumented `script_config`. |
| `execution` (asset_upload/bag_commit/base_config/workflow) | 1 | 3 | 6 | `bag_commit` "discarded after load" is false; workflow resolution doc drift; many missing `Example:` blocks. |
| `model` (catalog/denormalize_planner/annotations/database) | 2 | 4 | 4 | `annotations` Quick Start unrunnable; `find_association` return-type lie; ~19 methods missing examples; ADR-0007 wrong tag URI. |
| `core` (base/mixins/exceptions/validation/definitions) | 1 | 1 | 5 | **One real defect** (`validate_rids` swallowed `AttributeError`); rest cosmetic. |
| `feature.py + interfaces.py` | 4 | 0 | 0 | Smallest area, densest HIGH rate: three protocol contracts contradict their implementations + a non-existent feature field in the module example. |
| Cross-cutting docs (user-guides, ADRs, API ref) | — | — | — | Counted within areas above; offline.md and the missing denormalize-audit reference are the standouts. |

---

## 4. Cross-cutting patterns (highest-value synthesis)

These recur across modules and are the real story — fixing the
*pattern* is higher-leverage than fixing each instance.

1. **Examples that name a parameter/method/field the impl doesn't have.**
   The single biggest source of HIGH findings. `split_dataset` (missing
   `execution`), `add_files` (`file_types` vs `dataset_types`),
   `annotations` Quick Start (`handle.set_annotation`), offline.md
   (`group_by` vs `targets`), `list_dataset_members` (RID positional),
   `feature.py` (`Diagnosis=` field). These all fail at the *first line*
   a user copies. A doctest run (or `# doctest: +SKIP` discipline) would
   have caught most of them — and the inconsistent `+SKIP` placement in
   `workflow.py` shows the doctest harness isn't actually exercised.

2. **The underscore → dot `Table.column` denormalizer rename, applied
   unevenly.** The denormalizer now emits `Image_Class.Name`, and the
   *primary* `split_dataset` docstrings were updated (task #38). But the
   `SelectionFunction` Protocol doc, the `stratified_split` doctests,
   the concrete custom-selector example, and the *entire CLI surface*
   (`--stratify-by-column` epilog + help) still show underscore names
   and the `Image_Classification` shorthand. Same logical change, four
   stragglers — a classic partial-migration signature.

3. **Renamed/`.rid`-style API references left dangling.** `.rid`→
   `dataset_rid`/`workflow_rid` (download_dataset_bag returns,
   DatasetBag attrs), `list_nested_executions`→`list_execution_children`
   (user-guide + runner module doc), `add_workflow`→`_add_workflow`,
   `datasets: list[DatasetBag]`→`DatasetCollection`. Hard cutovers (R5.1)
   updated the code and the property docstrings but missed prose
   references and class-level Attributes blocks.

4. **The deleted `BagFeatureCache` / "SQLite feature cache" still
   documented as live.** `offline.md:113` and `:258`, plus a historical
   reference in `dataset_bag.py:564`, describe a per-bag feature cache
   that was removed in Stage 3b. The current path re-runs the full join
   every call — so the docs promise a performance characteristic the
   code no longer has.

5. **Exception-class narrowing: docs/code use the base
   `DerivaMLException` where a dedicated subclass exists (or where the
   ADR/user-guide promises one).** `DerivaMLCycleError`,
   `DerivaMLValidationError` (release no-dev-row, per ADR-0003),
   `DerivaMLTableNotFound`, `DerivaMLDirtyWorkflowError`. Callers writing
   `except DerivaMLValidationError` per the documented contract silently
   miss these. The exception *hierarchy diagram* (`exceptions.py`) is
   also missing two exported classes — the same theme from the catalog
   side.

6. **"13 keys vs 12 keys" describe-dict off-by-one in three places** —
   `denormalizer.describe`, `dataset_bag.describe_denormalized`, and the
   summary lines — all trail the addition of the `warnings` key. Trivial
   individually; a tell that the describe contract changed without a
   doc sweep.

7. **Systematic missing `Example:` blocks and Google-style violations.**
   `DerivaModel` (~19 methods), five `asset_upload` helpers,
   `run_model` and `validate_execution_config` (NumPy style). The
   CLAUDE.md convention ("Examples are required, Google style") is
   widely under-enforced in the model and execution layers.

8. **Stale doc/spec cross-references to moved or non-existent files.**
   The missing `2026-05-26-denormalize-audit.md` (cited dozens of
   times), the moved `phase2-model.md` audit, the stale "design spec §5",
   and the recurring `misd.isi.edu` tag typo (ADR-0007 *and*
   `create_schema.py`). Authority anchored on files that don't resolve.

9. **`interfaces.py` Protocol contracts diverging from concrete impls.**
   Three of the four `feature.py + interfaces.py` HIGHs are the
   *published protocol* (return types, a retired method) contradicting
   the only implementation. Because the protocol is the contract, treat
   these as correctness, not docs.

---

## 5. Recommended fix batches (report-only — ordered by leverage)

A follow-up could land these as coherent PRs, roughly highest-leverage
first.

**Batch A — the one real defect (do first, smallest, highest risk).**
Fix `core/validation.py:285` `dataset.list_versions()` →
`dataset.dataset_history()` and narrow or remove the `except Exception`
so the version-existence check actually raises instead of degrading to a
warning. Add a regression test for a bad `dataset_versions` requirement.
This is the only finding that changes runtime behavior.

**Batch B — make the published examples runnable (the HIGH example
cluster).** `split_dataset`/`stratified_split`/custom-selector examples
(add `execution`, switch to dot-notation columns), `add_files`
(`file_types`→`dataset_types`), `annotations` Quick Start
(`handle.set_annotation` → the ADR-0007 `table.annotations[...] =
...; ml.apply_annotations()` path), `list_dataset_members` (instance
form), `feature.py` module example (real field name), and the
`as_tf_dataset` `targets=None` self-contradiction. Pair with turning on
doctest execution (or fixing the `+SKIP` discipline in `workflow.py`) so
this class can't silently regress.

**Batch C — sweep the underscore→dot denormalizer rename to completion.**
`SelectionFunction` Protocol doc, `stratified_split` doctests, the
`split_dataset` CLI epilog + `--stratify-by-column` help. One logical
change, four call sites; leaves the migration finished.

**Batch D — `interfaces.py` protocol/impl reconciliation.** Fix
`add_features` (retire or correctly document the stub),
`AssetLike.list_executions` (`-> list[ExecutionRecord]`), and
`find_association`'s `-> tuple[Table, Column, Column]` return type. These
are contract-level and align the protocol with the feature-API spec and
ADR-0009.

**Batch E — the offline.md / dataset_bag doc refresh.** `group_by`→
`targets`, the torch/tf 3-tuple `(sample, target, rid)` shape, remove the
deleted-`BagFeatureCache` "fast cache" claims (lines 113, 258, and
`dataset_bag.py:564`), and correct the bag layout / `bag.path` /
`DatasetBag.path` SQLite-location description to the checksum-based
ADR-0006 layout.

**Batch F — renamed-API prose + class-Attributes sweep.**
`executions.md` `list_nested_executions`, runner module doc call shapes,
`workflow.py` `add_workflow`→`_add_workflow`, `download_dataset_bag`
`.rid`/`.minid`, and the `datasets: list[DatasetBag]` class Attributes.

**Batch G — exception contracts.** Decide per-site whether to raise the
dedicated subclass (`DerivaMLCycleError`, `DerivaMLValidationError` per
ADR-0003, `DerivaMLTableNotFound`) or to relax the doc/ADR to the base
class — then make doc, code, and ADR agree. Add the two missing classes
to the `exceptions.py` hierarchy diagram.

**Batch H — convention cleanup (lowest risk, largest count).** Add the
missing `Example:` blocks (`DerivaModel`, `asset_upload` helpers),
convert `run_model` / `validate_execution_config` to Google style,
document the undocumented fields (`script_config`, `git_root`,
`allow_dirty`, `DatasetSpec`/`DatasetHistory` fields), `refresh_model`
docstring, the "13 keys" off-by-one, and the stale cross-references
(missing denormalize-audit citation, moved phase2 audit, the
`misd`→`isrd` tag typo in ADR-0007 and `create_schema.py`).
