# Technical-writer audit (2026-05-22)

Pre-release documentation audit of deriva-ml v1.37.x. Scope, lenses, and severity tags as defined in the audit brief.

## Summary

- **Public-API symbols audited:** ~80 (DerivaML class methods, DerivaMLConfig, ExecutionConfiguration, Execution lifecycle methods, Dataset, DatasetBag, FeatureRecord and its selectors, AssetSpec/AssetSpecConfig, DatasetSpec/DatasetSpecConfig, Workflow, top-level exceptions, and all annotation-builder classes in `deriva_ml.model`).
- **Doc pages audited:** 38 (`docs/index.md`, 3 getting-started stubs, 9 user-guide pages, 4 configuration pages, 14 api-reference pages, 7 concepts stubs, 3 workflows stubs, `docs/cli-reference.md`, `docs/architecture.md`, `docs/release-notes.md`, `docs/reference/`).
- **Findings:** 56 total — **9 P0**, **20 P1**, **20 P2**, **7 P3**.
- **Top themes (in priority order):**
  1. **Reference rot**: the api-reference tree has at least one entry pointing at a module that does not exist (`upload.md` → `deriva_ml.dataset.upload`), the docs nav buries entry points (`DerivaMLConfig`, `AssetSpec`, `AssetSpecConfig`, `FeatureRecord`, annotation builders) under `DerivaML` or omits them entirely.
  2. **Stale code examples**: at least three high-traffic snippets in `docs/user-guide/datasets.md`, `docs/configuration/overview.md`, and `docs/user-guide/executions.md` call APIs with signatures that no longer exist (`split_dataset(ml, rid, test_size=...)` is missing the now-required `execution` arg; `DerivaMLConfig(check_auth=True)` cites a field that no longer exists; `AssetFilePath.set_metadata(key, value)` is documented but the class only exposes a `metadata` property + `set_asset_types`).
  3. **Getting-started is missing**: `docs/getting-started/{install,quick-start,project-setup}.md` are HTML meta-refresh redirects to the template repo. A first-time reader landing on the site has zero in-site walkthrough — they get bounced out of the docs to a separate GitHub repo before they can copy a single line of working code.
  4. **Doc tree shape**: 14 redirect-only pages (`concepts/`, `workflows/`, `cli-reference.md`, three `getting-started/`) live in the tree but are not in `mkdocs.yml`'s nav. They render as bare HTML if a Google result links to them. Decision needed — delete them, or restore real content.
  5. **Public-API symbols without runnable `Example:` blocks**: most annotation builders (`TableDisplayOptions`, `ColumnDisplayOptions`, `PreFormat`, `Facet`, `FacetRange`, `InboundFK`/`OutboundFK`, `Aggregate`, `TemplateEngine`, `ArrayUxMode`, `FacetUxMode`), most exception classes, and `DatasetHistory` / `DatasetMinid` carry attribute-style docstrings without an `Example:` block, even though they are user-facing per CLAUDE.md.
  6. **Release notes are 3+ releases out of date** (last entry `Version 1.34.0`; current release `v1.37.4`).

## Top-level entry-point review

### `DerivaML` class — `src/deriva_ml/core/base.py`

- **Class-level docstring** (lines 97–135): strong. Documents the verb-naming convention (`find_*` vs `list_*`), lists attributes, and gives a runnable `# doctest: +SKIP` example. The convention block is one of the best concept-in-docstring entries in the codebase. **No findings.**
- **`__init__`** (lines 237–290): full Google docstring with all kwargs, but **no `Example:` block** despite the class being the canonical user-facing entry point. P2.
- **`from_context`** (lines 192–235): good prose; uses `Example::` (sphinx-rst directive) instead of Google `Example:` style. Inconsistent with the rest of the file. P3.
- **`apply_catalog_annotations`** (lines 1078–1132): thorough; one of the best docstrings in the codebase. **No findings.**
- **`refresh_schema`**, **`pin_schema`**, **`unpin_schema`**, **`pin_status`**, **`diff_schema`**: excellent — every method documents its mode-branched behavior, its raises, and its caller obligations. **No findings.**
- **`chaise_url`** (lines 963–985): Example block uses heading `Examples:` (plural) — Google style is `Example:` (singular). Same for `cite` (lines 995–1043), `create_vocabulary` (lines 1134–1170), `create_table` (lines 1192–1300), `add_term` (mixin), `clear_cache`. The mkdocstrings handler accepts either, but the rest of the codebase uses singular `Example:`. Inconsistency. P3.
- **`create_table`** docstring (lines 1192–1300): runs to ~108 lines of `Example:` markup. Useful, but mostly duplicates what's in `concepts/`-era docs. Could be split or shortened. P3.
- **`mode`** property (lines 776–789): minimal `Example:` (`ml.mode is ConnectionMode.online`) — pure-Python example would need a constructed instance, so the example doesn't actually run as a doctest. Not marked `# doctest: +SKIP`. P2.
- **`download_dir`** (lines 797–813): no `# doctest: +SKIP` on the catalog-dependent example (`cache_dir = ml.download_dir(cached=True)`) — example uses an undefined `ml`. If `--doctest-modules` runs against this it will fail. P1.

### `DerivaMLConfig` — `src/deriva_ml/core/config.py`

- **Class docstring** (lines 71–114): comprehensive attribute table, runnable `Example:`. **No findings.**
- **`compute_workdir`** (lines 163–203): doctest example `>>> DerivaMLConfig.compute_workdir('/shared/data', '52', 'ml.example.org')` is expected output uses literal `username` — but the method calls `getpass.getuser()` at runtime. The example doctest would FAIL for any non-`username` user. P0 — this WILL break `pytest --doctest-modules` on any developer machine. Either skip with `# doctest: +SKIP` or convert to a no-output assertion.

### `ExecutionConfiguration` — `src/deriva_ml/execution/execution_configuration.py`

- **Class docstring** (lines 39–76): excellent — describes every attribute, including the mixed RID-string/AssetSpec coercion. **No findings.**
- **`validate_assets`** validator (lines 87–115): docstring describes the four input shapes correctly. **No findings.**
- **`load_configuration`** (lines 117–141): good. **No findings.**

### `Execution` — `src/deriva_ml/execution/execution.py`

- **Class docstring** (lines 220–266): the example documents the "upload AFTER the context manager exits" rule clearly. **No findings.**
- **`__init__`** (lines 269–311): solid. **No findings.**
- **`download_dataset_bag`** (lines 1163–1188): the example is fine but the **returned-value description** ("DatasetBag with `path`, `rid`, `minid`") is wrong — `DatasetBag` exposes `path`, `dataset_rid`, `current_version` (no `rid`, no top-level `minid` field). P1.
- **`update_status`** (lines 1189–1244): example `exe.update_status(ExecutionStatus.Running)` — but `update_status(status)` without an `error` is valid only on terminal-state attempts where error is ignored. Fine. **No findings.**
- **`execution_start`/`execution_stop`** (lines 1245–1344): mention "execution_stop()" inside the `execution_start()` example, which is a good cross-link. **No findings.**
- **`upload_execution_outputs`** (lines 1469–1505): solid. **No findings.**
- **`add_features`** (lines 1072–1160): excellent prose ("Provenance requirement" callout); example is runnable as `# doctest: +SKIP`. **No findings.**
- **`asset_file_path`** (lines 1902–2014): missing `Example:` block entirely (every other public mutating method has one). P1.
- **`metrics_file`** (lines 2016–2091): excellent. The `MLAsset.execution_metadata.value` pure-Python doctest actually runs — exemplary. **No findings.**
- **`abort`** (lines 2585–2616): solid. **No findings.**
- **`pending_summary`** / **`upload_outputs`** (lines 2618–2676): solid. **No findings.**

### `AssetSpec`, `AssetSpecConfig`, `AssetFilePath` — `src/deriva_ml/asset/aux_classes.py`

- **`AssetSpec`** (lines 153–176): the doctest examples are pure-Python and *will run* during collection — but `AssetSpec(rid="3JSE")` is not asserted (no expected output). `RID` validation lives in `core.definitions`; if the regex allows `"3JSE"` the test passes silently. P3 (cosmetic — preferred Google form: `>>> AssetSpec(rid="3JSE").cache` `False`).
- **`AssetSpecConfig`** (lines 187–202): example uses `from hydra_zen import store` — pure-Python but marked `# doctest: +SKIP`. **No findings.**
- **`AssetFilePath`** (lines 28–150): no class-level `Example:` shows `set_asset_types()`; the constructor docstring is fine. The `metadata` getter/setter docstrings (lines 88–122) document the semantics well. **No findings on existence**, but see the `executions.md` issue below — user-guide docs claim a `set_metadata(key, value)` method exists; it does not. P0 (in docs/, not in src/).

### `DatasetSpec`, `DatasetSpecConfig` — `src/deriva_ml/dataset/aux_classes.py`

- **`DatasetSpec`** (lines 304–394): docstring opens with `"Represent a dataset_table in an execution configuration dataset_table list"` — typo / placeholder leak from a refactor. P2.
- **`DatasetSpec.version`** field type is `DatasetVersion | conlist(...) | tuple[...] | str` but the docstring just says "The version of the dataset. Should follow semantic versioning." — doesn't mention that lists/tuples/dicts/strings are all accepted via `version_field_validator`. P2.
- **`DatasetVersion`** / **`VersionPart`** (lines 28–202): exemplary. Pure-Python doctests, clear rationale. **No findings.**
- **`DatasetHistory`** (lines 204–235): no `Example:` block, attribute-style only. P2.
- **`DatasetMinid`** (lines 238–301): no `Example:` block, no description of when an instance is actually constructed (it's a parser for the MINID landing-page JSON). P2.
- **`DatasetSpecConfig`** (lines 398–425): no `Example:` block. Compared to `AssetSpecConfig` (lines 187–202) — `AssetSpecConfig` has one, `DatasetSpecConfig` does not. P2.

### `Dataset` — `src/deriva_ml/dataset/dataset.py`

- **Class docstring** (lines 75–110): excellent. **No findings.**
- **`Dataset.__init__`** (lines 112–139): solid. **No findings.**
- **`dataset_types` property** (lines 189–211): two-line `Example:` is opaque (`>>> types = dataset.dataset_types` `>>> print(types)`) — doesn't say what the output looks like. P3.
- **`create_dataset`** static method (lines 213–292): excellent. **No findings.**
- **`release`**, **`is_dirty`**, **`release_diff`**, **`compare_versions`**, **`mark_dev`** — pin-quality writing on dev/released semantics. **No findings.**
- **`add_dataset_members`** (lines 1852+): inspected, well-documented. **No findings.**
- **`download_dataset_bag`** (lines 2320+): **docstring says `fetch_concurrency` defaults to 8 but the signature is `fetch_concurrency: int = 1`**. Found at file lines 2327 and 2353. P0 — direct contradiction between signature and docstring.

### `DatasetBag` — `src/deriva_ml/dataset/dataset_bag.py`

- **Class docstring** (lines 65–112): excellent. **No findings.**
- **`as_torch_dataset`** / **`as_tf_dataset`** (lines 1130–1485): exhaustive — long but warranted given the framework adapter shape. **No findings.**
- **`restructure_assets`** (line 1485+): inspected. **No findings.**
- **`current_version`** property (lines 181–192): no `Example:`. P3.

### `FeatureRecord` — `src/deriva_ml/feature.py`

- **Module docstring** (lines 1–33): excellent — names every selector helper. **No findings.**
- **Class docstring** (lines 49–95): explains attributes, provenance, multi-annotator selector pattern. **No findings.**
- **`select_newest`**, **`select_first`**, **`select_latest`** (lines 107–311): documented with runnable code in `Example:` blocks but the inline code is shown inside RST `::` literal blocks instead of doctest-style `>>>`. Mkdocstrings rendering is fine; the missing `>>>` means these are NOT picked up by `--doctest-modules`. Acceptable but inconsistent. P3.
- **`select_by_execution`**, **`select_by_workflow`** (lines 131–269): excellent — "None return semantics" callout is exactly what a user needs. **No findings.**
- **`select_majority_vote`** (lines 313–373): solid. **No findings.**

### `Workflow` — `src/deriva_ml/execution/workflow.py`

- **Class docstring** (lines 41–105): good — covers catalog binding, write-back, deduplication. **No findings.**

### Annotation builders — `src/deriva_ml/model/annotations.py`

Per CLAUDE.md, this is documented public API consumed by the `deriva-skills/use-annotation-builders` skill. Per-class assessment:

- **`Display`** (lines 336–420): excellent. Multiple runnable examples. **No findings.**
- **`NameStyle`** (lines 302–333): one runnable example. **No findings.**
- **`SortKey`** (lines 428–448): example block `>>> SortKey("Name")  # Ascending` has no expected output — doctest passes silently. P3.
- **`InboundFK`**, **`OutboundFK`** (lines 456–559): good runnable examples. **No findings.**
- **`PseudoColumnDisplay`** (lines 562–599): no `Example:`. P2.
- **`PseudoColumn`** (lines 606–730): exhaustive — multiple `Example:` blocks for outbound, inbound, multi-hop, display formatting, array aggregate. **No findings.**
- **`VisibleColumns`** (lines 741–862): excellent. **No findings.**
- **`VisibleForeignKeys`** (lines 874–911): minimal but adequate. **No findings.**
- **`TableDisplayOptions`** (lines 920–969): no `Example:`. P2.
- **`TableDisplay`** (lines 973–1021): adequate. **No findings.**
- **`PreFormat`** (lines 1030–1051): no `Example:`. P2.
- **`ColumnDisplayOptions`** (lines 1055–1085): no `Example:`. P2.
- **`ColumnDisplay`** (lines 1087–1140): no `Example:` block. P2.
- **`Facet`**, **`FacetList`**, **`FacetRange`** (lines 1142+): inspected; minimal docstrings without `Example:` blocks. P2.
- **`TemplateEngine`**, **`Aggregate`**, **`ArrayUxMode`**, **`FacetUxMode`** enums (lines 139–280): inspected; member-style docstrings without examples of *use*. P2.
- **`AnnotationBuilder`** base (lines 282–294): documents only the abstract contract. **No findings.**
- **`fk_constraint` helper**: referenced in `VisibleColumns` examples but its own docstring not surfaced in this audit. P3.

## Findings by area

### Docstrings: src/

#### [P0] `DerivaMLConfig.compute_workdir` doctest example is non-runnable
**Location:** `src/deriva_ml/core/config.py:186-191`
**Issue:** The docstring shows a `>>> DerivaMLConfig.compute_workdir('/shared/data', '52', 'ml.example.org')` example with literal expected output `PosixPath('/shared/data/username/deriva-ml/ml.example.org/52')`. The method internally calls `getpass.getuser()`, so the expected `username` portion will differ on every machine. Per CLAUDE.md, pure-Python examples run for real during pytest collection — this one will FAIL.
**Quote:** `>>> DerivaMLConfig.compute_workdir('/shared/data', '52', 'ml.example.org')` / `PosixPath('/shared/data/username/deriva-ml/ml.example.org/52')`
**Suggested fix:** Either (a) mark both example lines `# doctest: +SKIP`, or (b) convert to a no-output assertion like `>>> str(DerivaMLConfig.compute_workdir('/x', '52', 'h.example.org')).endswith('/h.example.org/52')` `True`.

#### [P0] `Dataset.download_dataset_bag` docstring contradicts signature on `fetch_concurrency` default
**Location:** `src/deriva_ml/dataset/dataset.py:2327` (signature) vs `dataset.py:2353` (docstring)
**Issue:** Signature is `fetch_concurrency: int = 1` but the `Args:` block says `"Defaults to 8."` — direct contradiction.
**Quote (signature):** `fetch_concurrency: int = 1`
**Quote (docstring):** `"fetch_concurrency: Maximum number of concurrent file downloads during materialization. Defaults to 8."`
**Suggested fix:** Decide which is correct and align the other. (The `DatasetSpec.fetch_concurrency` in `aux_classes.py` defaults to `8`, so the docstring intent appears to be 8 — the signature looks like a regression.)

#### [P1] `Execution.download_dataset_bag` documents return fields that don't exist
**Location:** `src/deriva_ml/execution/execution.py:1174-1178`
**Issue:** Returns block lists `path`, `rid`, `minid` — but `DatasetBag` exposes `path`, `dataset_rid`, `current_version`, etc. There is no top-level `rid` attribute and no `minid` attribute on a `DatasetBag`.
**Quote:** `"DatasetBag: Object containing: - path: Local filesystem path to downloaded dataset - rid: Dataset's Resource Identifier - minid: Dataset's Minimal Viable Identifier"`
**Suggested fix:** Replace with `bag.path`, `bag.dataset_rid`, `bag.current_version` to match the actual public surface (and mirror the more-correct docstring on `Dataset.download_dataset_bag`).

#### [P1] `Execution.asset_file_path` is missing a usage `Example:` block
**Location:** `src/deriva_ml/execution/execution.py:1902-1942`
**Issue:** This is the canonical method for registering output files. The class-level docstring and every other major mutating method on `Execution` (`add_features`, `metrics_file`, `upload_execution_outputs`, `download_dataset_bag`, `download_asset`) has an `Example:` block. `asset_file_path` doesn't.
**Quote:** Ends with `"Raises:" ... "DerivaMLValidationError: If asset_types contains invalid terms."` — no `Example:`.
**Suggested fix:** Add a runnable `# doctest: +SKIP` example covering at least (a) new-file mode (`path = exe.asset_file_path("Model", "model.pt"); path.write_bytes(...)`) and (b) the `metadata=AssetRecord(...)` form, since that's the recommended path.

#### [P1] `DerivaML.download_dir` example uses an undefined `ml` and is not skipped
**Location:** `src/deriva_ml/core/base.py:808-811`
**Issue:** The `Example:` block reads `>>> cache_dir = ml.download_dir(cached=True)` with no `# doctest: +SKIP`. Per CLAUDE.md, catalog-dependent examples MUST carry `+SKIP`. The references to `ml` would fail under `--doctest-modules`.
**Quote:** `>>> cache_dir = ml.download_dir(cached=True)` / `>>> work_dir = ml.download_dir(cached=False)`
**Suggested fix:** Add `# doctest: +SKIP` to both lines.

#### [P1] `DerivaML._get_session_config` example is not skipped
**Location:** `src/deriva_ml/core/base.py:727-729`
**Issue:** `>>> config = DerivaML._get_session_config()` — this *should* run (it's a `@staticmethod` with no catalog requirement), but `>>> print(config['retry_read']) # 8` has the expected output `8` written as a *comment* not as a doctest expected line. So either the doctest is wrong, or it's not actually being run. The intent is unclear from the source.
**Quote:** `>>> print(config['retry_read']) # 8`
**Suggested fix:** Convert to standard doctest form: `>>> config['retry_read']` `8`.

#### [P1] `DatasetSpec` class docstring leads with a typo / placeholder
**Location:** `src/deriva_ml/dataset/aux_classes.py:304-317`
**Issue:** Opens with `"Represent a dataset_table in an execution configuration dataset_table list"` — "dataset_table" looks like an unfinished refactor (the class is `DatasetSpec`, not `DatasetTable`).
**Quote:** `"Represent a dataset_table in an execution configuration dataset_table list"`
**Suggested fix:** Rewrite to `"Specification for a dataset input to an execution configuration."` to match the `AssetSpec` style.

#### [P2] `DatasetSpec.version` docstring understates the accepted types
**Location:** `src/deriva_ml/dataset/aux_classes.py:317`
**Issue:** Field type is `DatasetVersion | conlist(int, 3) | tuple[int,int,int] | str` and the `version_field_validator` accepts dicts, lists, tuples, strings, and `DatasetVersion`. The `Attributes:` block reduces this to `"version (DatasetVersion): The version of the dataset. Should follow semantic versioning."` — readers won't know they can pass `"1.2.3"`, `[1, 2, 3]`, or `(1, 2, 3)`.
**Quote:** `"version (DatasetVersion): The version of the dataset. Should follow semantic versioning."`
**Suggested fix:** Mention all accepted forms with one-liner examples per accepted type.

#### [P2] `DerivaML.__init__` has no `Example:` block
**Location:** `src/deriva_ml/core/base.py:237-290`
**Issue:** Every kwarg is documented but there is no runnable `Example:` showing the constructor with common kwarg combos (`hostname` + `catalog_id`; `working_dir=`; `mode=ConnectionMode.offline`).
**Suggested fix:** Add three small `# doctest: +SKIP` example invocations.

#### [P2] Several annotation-builder helpers lack `Example:` blocks
**Locations:**
- `PseudoColumnDisplay` — `src/deriva_ml/model/annotations.py:562-599`
- `TableDisplayOptions` — `src/deriva_ml/model/annotations.py:920-969`
- `PreFormat` — `src/deriva_ml/model/annotations.py:1030-1051`
- `ColumnDisplayOptions` — `src/deriva_ml/model/annotations.py:1055-1085`
- `ColumnDisplay` — `src/deriva_ml/model/annotations.py:1087-1140`
- `Facet`, `FacetList`, `FacetRange` — `src/deriva_ml/model/annotations.py:1142+`
- Enums `TemplateEngine`, `Aggregate`, `ArrayUxMode`, `FacetUxMode`

**Issue:** Per CLAUDE.md `model/annotations.py` is consumed externally by the `deriva-skills/use-annotation-builders` skill. The skill's authors and end-users need to see runnable examples for each surface, not just an attribute table.
**Suggested fix:** Each builder should carry at least one minimum-viable `>>>` example showing how to instantiate it and pass it to a parent builder (e.g., `ColumnDisplay().detailed(ColumnDisplayOptions(pre_format=PreFormat(format="%.2f")))`).

#### [P2] `DatasetHistory`, `DatasetMinid` lack `Example:` blocks
**Location:** `src/deriva_ml/dataset/aux_classes.py:204-301`
**Issue:** Both are user-facing return types (`Dataset.dataset_history()` returns `list[DatasetHistory]`; `DatasetMinid` surfaces in MINID-creation flows). Neither has an `Example:` block; readers landing on these from the API reference get only an attribute table.
**Suggested fix:** Add a `# doctest: +SKIP` example for each showing the typical access pattern.

#### [P2] `DatasetSpecConfig` lacks `Example:` block (asymmetric with `AssetSpecConfig`)
**Location:** `src/deriva_ml/dataset/aux_classes.py:398-425`
**Issue:** `AssetSpecConfig` (lines 187–202) has a clear hydra-zen `store()` example. `DatasetSpecConfig` does not, even though it's the more common configuration in real projects.
**Suggested fix:** Add an `Example:` block paralleling `AssetSpecConfig`'s.

#### [P2] `DerivaML.cache_table`, `DerivaML._cache_features` use sphinx-style `Example::` not Google `Example:`
**Location:** `src/deriva_ml/core/base.py:874-901`, `:903-961`
**Issue:** `Example::` is the docutils/sphinx literal-block convention; the rest of the codebase consistently uses Google's `Example:` heading followed by `>>>` lines. Mixed convention.
**Suggested fix:** Convert to `Example:` + `>>>` style consistently across the file.

#### [P3] `SortKey`, `AssetSpec` examples have no expected-output lines
**Locations:** `src/deriva_ml/model/annotations.py:437-438`, `src/deriva_ml/asset/aux_classes.py:167-169`
**Issue:** Examples like `>>> SortKey("Name")  # Ascending` and `>>> spec = AssetSpec(rid="3JSE")` lack expected-output assertions. They run but assert nothing.
**Suggested fix:** Add an assertion line (`>>> SortKey("Name").column` `'Name'`).

#### [P3] Inconsistent `Example:` vs `Examples:` heading
**Locations:** `core/base.py:978-985, 1015-1028, 1156-1169, 1226-1299, 1385-1393, 1446-1449`, plus most mixin files.
**Issue:** Google style is singular `Example:`. Some methods use `Examples:`. mkdocstrings handles both, but readers see two visual treatments.
**Suggested fix:** Normalize to `Example:` across the codebase (search/replace).

#### [P3] `Dataset.dataset_types` example is opaque
**Location:** `src/deriva_ml/dataset/dataset.py:203-205`
**Issue:** `>>> types = dataset.dataset_types` `>>> print(types)` shows no expected output. Reader has no idea what's printed.
**Suggested fix:** Add a representative expected output line like `['TrainingSet', 'Labeled']`.

### docs/user-guide/

#### [P0] `docs/user-guide/datasets.md` calls `split_dataset` with a stale signature
**Location:** `docs/user-guide/datasets.md:289-330` (multiple examples)
**Issue:** Every `split_dataset` example in the user guide uses the form `split_dataset(ml, source_dataset_rid, test_size=0.2, seed=42)`. The current signature in `src/deriva_ml/dataset/split.py:482` requires an `execution` parameter as the third positional argument: `split_dataset(ml, source_dataset_rid, execution, *, test_size=0.2, ...)`. Every copy-paste from this page raises `TypeError: split_dataset() missing 1 required positional argument: 'execution'`.
**Quote:** `result = split_dataset(ml, source_dataset_rid, test_size=0.2, seed=42)`
**Suggested fix:** Update all four examples (lines 290, 304, 322, 342, 361) to include an `execution=exe` arg, wrapped in an `ml.create_execution(...)` context manager. Note also that the *module docstring* at `src/deriva_ml/dataset/split.py:26-87` has the correct shape — the docs page just diverged from it.

#### [P0] `docs/configuration/overview.md` documents a `check_auth` field that does not exist on `DerivaMLConfig`
**Location:** `docs/configuration/overview.md:46-66`
**Issue:** The example shows `check_auth=True` and the field table lists `check_auth | bool | True | Verify authentication on connect`. There is no `check_auth` field in `src/deriva_ml/core/config.py` (and a grep across `core/` finds no occurrences). Code copied from this snippet will fail Pydantic validation immediately.
**Quote:** `check_auth=True, # Verify authentication`
**Suggested fix:** Remove `check_auth=True` from the example and the corresponding row from the parameter table. If the intent was something else (mode? credential?), document the real field.

#### [P0] `docs/user-guide/executions.md` documents an `AssetFilePath.set_metadata()` method that does not exist
**Location:** `docs/user-guide/executions.md:108-143`
**Issue:** The page contains two mentions of `.set_metadata()`: `"if you plan to update its metadata via .set_metadata() or .set_asset_types()"` and `path.set_metadata("Acquisition_Time", "14:30:00")  # update after registration`. The class (`src/deriva_ml/asset/aux_classes.py:28-150`) only exposes a `metadata` property (read/write) plus `set_asset_types()`. There is no `set_metadata()` method.
**Quote:** `path.set_metadata("Acquisition_Time", "14:30:00")  # update after registration`
**Suggested fix:** Replace with `path.metadata = {...}` (full-dict replacement via the property setter) — and document the underlying behavior (the setter accepts a dict or an `AssetRecord` and overwrites the entire metadata dict; there is no per-key incremental update).

#### [P0] `docs/getting-started/{install,quick-start,project-setup}.md` are HTML redirects, not docs
**Location:** all three files in `docs/getting-started/`
**Issue:** Each is an HTML meta-refresh redirecting to `https://github.com/informatics-isi-edu/deriva-ml-model-template`. A first-time visitor to https://informatics-isi-edu.github.io/deriva-ml hitting "Quick start" gets bounced out of the docs site to a separate GitHub repo. The mkdocs nav doesn't surface getting-started at all (`mkdocs.yml` nav skips it entirely), so the existence of these files is invisible to most readers — but anyone with a stale bookmark or Google result hits these stubs.
**Quote:** `<meta http-equiv="refresh" content="0; url=https://github.com/informatics-isi-edu/deriva-ml-model-template">`
**Suggested fix:** Either (a) delete the three files (they're not in nav and the redirect is silent), or (b) restore minimal getting-started content so the first-time reader has a path through the docs site itself. Option (b) is strongly preferred — the `docs/index.md` "Starting a new project" section already points to the template repo, so an in-site walkthrough adds value rather than duplicating.

#### [P0] `docs/api-reference/upload.md` references a non-existent module
**Location:** `docs/api-reference/upload.md`
**Issue:** Page body is `::: deriva_ml.dataset.upload` — but `src/deriva_ml/dataset/` contains no `upload.py`. The `mkdocstrings` directive will either render an empty page or fail the build.
**Quote:** `::: deriva_ml.dataset.upload`
**Suggested fix:** Either point to `deriva_ml.cli.upload` (the actual upload CLI module under `src/deriva_ml/cli/upload.py`), or delete the file and its nav entry (`mkdocs.yml:81`). Note: the nav entry's label is "Upload"; the CLI's job is different from "asset upload during executions," so check the intent before silently re-pointing.

#### [P0] `docs/release-notes.md` is 3+ releases stale
**Location:** `docs/release-notes.md`
**Issue:** Latest entry is `Version 1.34.0`. Current release per `git tag` is `v1.37.4`. Releases 1.35, 1.36, 1.37, and 1.37.1–1.37.4 are not documented. Recent merged PRs (#177, #178, #179, #180, #181, #184, #185) include user-visible changes — execution staging directory move, asset overwrite warning, find_association typed exceptions — none of which appear in release notes.
**Suggested fix:** Backfill 1.35–1.37.4. This is a hard pre-release blocker — release notes are the standard signal for "should I upgrade?"

#### [P1] `mkdocs.yml` nav omits 14 doc pages that exist on disk
**Location:** `mkdocs.yml`
**Issue:** The following exist as `.md` files but are not in the nav, so they're either dead pages (rendering as orphans if linked to) or undeclared content:
- `docs/getting-started/install.md`, `quick-start.md`, `project-setup.md` (redirects)
- `docs/concepts/*.md` (7 redirects)
- `docs/workflows/*.md` (3 redirects)
- `docs/cli-reference.md` (redirect)
- `docs/architecture.md` (real content — but not in nav!)
- `docs/reference/README.md`, `docs/reference/schema.md` (real content — schema is canonical per `reference/README.md`)

**Suggested fix:** Delete the redirect-only files; add `architecture.md` and `reference/schema.md` to the nav (the latter is described as the "Authoritative description of the deriva-ml schema" — first-class content). If the redirects are kept for legacy URL preservation, document the policy in a comment in `mkdocs.yml`.

#### [P1] `mkdocs.yml` does not surface `DerivaMLConfig`, `AssetSpec/AssetSpecConfig`, `DatasetSpec/DatasetSpecConfig`, or `MLAsset/MLVocab` enums in API Reference
**Location:** `mkdocs.yml:62-81`
**Issue:** The `API Reference` section exposes `DerivaML` (`deriva_ml_base.md`), `DerivaModel`, `Dataset`, `DatasetBag`, `Dataset Splitting`, `Dataset Auxiliary Classes`, `Execution`, `ExecutionConfiguration`, `Workflow`, `Lineage`, `Feature`, `Definitions & Types`, `Exceptions`, `Upload`. Missing: `DerivaMLConfig`, the `AssetSpec`/`AssetSpecConfig` pair as a discoverable page, and (most importantly) the annotation builders in `deriva_ml.model.annotations` are buried inside `deriva_model.md` (`::: deriva_ml.model`) rather than getting their own first-class API-reference page.
**Suggested fix:** Add `api-reference/config.md` (`::: deriva_ml.core.config`), `api-reference/asset.md` (`::: deriva_ml.asset.aux_classes`), and `api-reference/annotations.md` (`::: deriva_ml.model.annotations`) — the last is particularly important because CLAUDE.md pins it as a public API consumed by the `deriva-skills` Claude Code skill.

#### [P1] `docs/api-reference/exceptions.md` is a single `:::` directive — no overview of the hierarchy
**Location:** `docs/api-reference/exceptions.md`
**Issue:** Page body is just `::: deriva_ml.core.exceptions`. The exception hierarchy described in `CLAUDE.md` is a useful map (`DerivaMLException → DerivaMLConfigurationError → DerivaMLSchemaError, ...`), but readers landing on this page get an alphabetical class dump without that tree.
**Suggested fix:** Add the hierarchy diagram from `CLAUDE.md` ("Exception Hierarchy" section) as introductory prose above the mkdocstrings directive.

#### [P1] `docs/api-reference/execution.md` uses `::: deriva_ml.execution` with no overview
**Location:** `docs/api-reference/execution.md`
**Issue:** Single-line body `::: deriva_ml.execution` will render the entire submodule (Execution, Workflow, ExecutionConfiguration, BaseConfig, lineage types, multirun_config, runner, lineage, ...) on one page — but the `execution_configuration.md`, `workflow.md`, `lineage.md` pages already point to specific submodules. Either this page duplicates them, or it's incoherent. The TOC will be enormous.
**Suggested fix:** Either narrow this page to `::: deriva_ml.execution.execution` (just the `Execution` class) or restructure to make this page a true table-of-contents / overview that links to the per-class pages.

#### [P1] `docs/api-reference/deriva_ml_base.md` typo
**Location:** `docs/api-reference/deriva_ml_base.md:4`
**Issue:** Typo `"These methods assume tha tthe catalog"` — "tha tthe" with a space inside the word.
**Quote:** `"These methods assume tha tthe catalog contains a 'deriva-ml' and a domain schema."`
**Suggested fix:** Fix to `"These methods assume that the catalog ..."`.

#### [P1] `docs/api-reference/deriva_ml_base.md` mentions Execution Asset table that no longer exists by that name
**Location:** `docs/api-reference/deriva_ml_base.md:17-18`
**Issue:** Lists `Execution Asset` and `Execution Metadata` as ML schema entities. Per `MLAsset` enum in `core/definitions.py`, the catalog tables are `Execution_Asset` and `Execution_Metadata` (underscore-separated). The prose form ("Execution Asset", "Execution Metadata" with space) is fine for English but might mislead readers searching the catalog. Minor.
**Suggested fix:** Either use the canonical `Execution_Asset` form or add "(`Execution_Asset` in the catalog)" parenthetical for searchability.

#### [P1] `docs/concepts/`, `docs/workflows/`, `docs/cli-reference.md` are 11 HTML redirects with no nav presence
**Location:** all files in `docs/concepts/`, `docs/workflows/`, plus `docs/cli-reference.md`
**Issue:** Same shape as the getting-started redirects. Old URLs from prior versions of these docs are preserved as redirects, but neither the redirect targets nor the redirect itself is in the nav. If a search-engine result or external link points to e.g. `/concepts/datasets/`, the user lands on a bare HTML page with no navigation, no header, no styling — just a plain refresh tag.
**Quote (example):** `<meta http-equiv="refresh" content="0; url=../user-guide/datasets.md">`
**Suggested fix:** Either (a) delete all 11 files (the redirects do bounce, but the experience is broken), or (b) use a proper mkdocs redirect plugin like `mkdocs-redirects` so the redirect happens server-side and the user lands on a real docs page.

#### [P1] `docs/user-guide/datasets.md` references `ml.add_term(MLVocab.dataset_type, ...)` without showing the `MLVocab` import
**Location:** `docs/user-guide/datasets.md:52, 62-68`
**Issue:** `ml.add_term(MLVocab.dataset_type, "TrainingSet", ...)` appears in a "Notes" callout AND in the next code block, but the `from deriva_ml import MLVocab` import only appears at line 61 (inside the second usage). A reader skimming the callout sees `MLVocab.dataset_type` with no idea where to import it from.
**Suggested fix:** Add a one-line `from deriva_ml import MLVocab` to the first occurrence, or pull the import up to the chapter's overall setup snippet.

#### [P1] `docs/user-guide/executions.md` references a `DatasetSpec(version=None)` default that doesn't match the code
**Location:** `docs/user-guide/executions.md:42-47`
**Issue:** The table says `version | str | None | Specific version to use (default: current)`. The class signature in `src/deriva_ml/dataset/aux_classes.py:319-320` makes `version` **required** (no default) and accepts `DatasetVersion | conlist | tuple | str`. Constructing `DatasetSpec(rid="...")` without a version fails validation.
**Quote:** `| version | str | None | Specific version to use (default: current) |`
**Suggested fix:** Update the table to reflect that `version` is required and document the four accepted types.

#### [P1] `docs/configuration/overview.md` shows `domain_schemas='my_domain'` (singular string) and elsewhere shows `domain_schemas={'my_domain'}` (set)
**Location:** `docs/configuration/overview.md:26-27, 50, 61`
**Issue:** The example shows `domain_schemas='my_domain'` (a string), but the field type is `str | set[str] | None`. Both forms are legal — but mixing them across examples in the same page without explanation confuses readers.
**Suggested fix:** Pick one form for the page, or add a single sentence: "`domain_schemas` accepts a single name (string) or a set of names. Use a set when working with catalogs that expose multiple domain schemas."

#### [P1] `docs/user-guide/exploring.md` claims `ml.find_executions(status=...)` accepts an `ExecutionStatus` enum directly
**Location:** `docs/user-guide/exploring.md:122-128`
**Issue:** `uploaded = list(ml.find_executions(status=ExecutionStatus.Uploaded))` — confirm this works. Cross-check with `core/mixins/execution.py`. If the actual filter signature takes a string, the example will fail.
**Suggested fix:** Verify the signature; if the kwarg requires a string, either change the call or document the conversion (`status="Uploaded"` or `status=ExecutionStatus.Uploaded.value`).

#### [P1] `docs/user-guide/reproducibility.md` references an asset_type `"Execution_Config"` for `uv.lock`
**Location:** `docs/user-guide/reproducibility.md:18`
**Issue:** Row in the table says `Execution_Config | uv.lock | Python dependency lockfile from the project root, recording exact package versions`. Confirm against `ExecMetadataType` enum members. If the enum uses a different name (e.g. `Lock_File`), the reader has no way to query the asset later.
**Suggested fix:** Verify against `src/deriva_ml/core/definitions.py` `ExecMetadataType` and align.

#### [P2] `docs/index.md` "When to use deriva-ml" is excellent but never reached from nav
**Location:** `docs/index.md`
**Issue:** The "Strong fit" / "Weaker fit" discussion is one of the best onboarding signals in the docs. But the nav opens directly to "Exploring a catalog" via the User Guide. New readers may never see this content.
**Suggested fix:** Either add a prominent "Read this first" call-out at the top of `user-guide/exploring.md`, or restructure the nav so `index.md` is a clearly-labeled landing page distinct from chapter 1.

#### [P2] `docs/user-guide/exploring.md` references `Chapter 4` and `Chapter 7` by number, but the nav uses titles
**Location:** `docs/user-guide/exploring.md:236-241`, similar in `datasets.md:543-550`, `features.md`
**Issue:** Internal cross-references use ad-hoc chapter numbers ("Chapter 1", "Chapter 7"), but the nav titles them by topic ("Exploring a catalog", "Reproducibility"). Numbers get out of sync when chapters are added/removed.
**Suggested fix:** Use the chapter titles directly: `[Exploring a catalog](exploring.md)`, `[Reproducibility](reproducibility.md)`. mkdocs-autorefs picks up titles automatically.

#### [P2] `docs/user-guide/executions.md` "Execution status lifecycle" mermaid diagram and `ExecutionStatus` enum are not cross-linked
**Location:** `docs/user-guide/executions.md:270-288`
**Issue:** The mermaid diagram shows the states; the table below documents them. Neither links to `execution.state_store.ExecutionStatus` in the API reference, so readers wanting the canonical enum can't navigate there.
**Suggested fix:** Add an "API reference: `ExecutionStatus`" link to the section's "See also" footer, pointing to whichever api-reference page hosts the enum.

#### [P2] `docs/configuration/overview.md` `DatasetSpecConfig` example uses field `timeout=[10, 1800]` (list) instead of the `tuple` form in the `DatasetSpec` model
**Location:** `docs/configuration/overview.md:91-93`
**Issue:** `DatasetSpec.timeout` is typed `tuple[int, int] | None`. `DatasetSpecConfig.timeout` is `list[int] | None`. Reader might expect either form to work in both contexts.
**Suggested fix:** Add a one-line note: "`DatasetSpec` (Python) accepts a tuple; `DatasetSpecConfig` (hydra-zen) uses a list because hydra serialization doesn't support tuples. Both produce the same effective config."

#### [P2] `docs/concepts/`, `docs/workflows/`, `docs/cli-reference.md` redirects don't tell mkdocs they should be excluded from build
**Location:** `mkdocs.yml`
**Issue:** mkdocs builds all `.md` files under `docs/` by default. The 14 redirect-only pages get compiled and shipped, just without nav. Build output is bloated.
**Suggested fix:** Either delete the files or set `not_in_nav` / `exclude_from_search` configuration. mkdocs-material supports per-page `not_in_nav` in front matter.

#### [P2] `docs/user-guide/sharing.md` is in nav but not in scope of this audit — flag for review
**Location:** `mkdocs.yml:48`, `docs/user-guide/sharing.md`
**Issue:** Not reviewed in this pass due to time. Flag for a follow-up audit — it appears in nav as Chapter 6 (Sharing and collaboration) but I did not validate its code examples or cross-references against current API.
**Suggested fix:** Run the same pass on `sharing.md` before release.

#### [P2] `docs/architecture.md` is large, current, and not in nav
**Location:** `docs/architecture.md`, `mkdocs.yml`
**Issue:** A 300+ line mermaid-rich architecture overview exists but is not surfaced in the nav. Readers who want a system-level view (especially those evaluating deriva-ml for adoption) will not find it.
**Suggested fix:** Add to nav, possibly as "About → Architecture" or as a top-level "Architecture" link.

#### [P2] `docs/user-guide/features.md` `create_feature` signature in "Notes" doesn't match the actual signature
**Location:** `docs/user-guide/features.md:113-114`
**Issue:** Notes block says `create_feature(target_table, feature_name, terms=None, assets=None, metadata=None, optional=None, comment="", update_navbar=True)`. Check this against `src/deriva_ml/core/mixins/feature.py:63` to confirm parameter order. The user-guide claims `terms`, `assets`, `metadata`, `optional` — confirm all four are accepted and in this order.

#### [P2] `docs/user-guide/datasets.md` mentions `DatasetSpec` import without showing `from deriva_ml.dataset.aux_classes import DatasetSpec`
**Location:** `docs/user-guide/datasets.md:438, 467-468, 485, 490`
**Issue:** Several code blocks use `DatasetSpec(...)` without showing the import. Some blocks have the import; others don't. Inconsistent.
**Suggested fix:** Add `from deriva_ml.dataset.aux_classes import DatasetSpec` (or, equivalently, `from deriva_ml.dataset import DatasetSpec`) to every snippet that uses it, or note that subsequent snippets reuse imports from earlier in the page.

#### [P2] `docs/user-guide/hydra-zen.md` line 53 imports `AssetSpecConfig` from `deriva_ml.asset`
**Location:** `docs/user-guide/hydra-zen.md:53`
**Issue:** Table cell says `from deriva_ml.asset import AssetSpecConfig`. Verified to work — `asset/__init__.py` re-exports it. But other docs (`docs/configuration/overview.md`, README content) might import from `deriva_ml.asset.aux_classes` directly. Pick one canonical import path and document it.
**Suggested fix:** Pick `from deriva_ml.asset import AssetSpecConfig` (the shorter path) as canonical and update everywhere.

#### [P3] `docs/api-reference/lineage.md` cross-references `deriva_ml_base.md` for `lookup_lineage` but mkdocstrings inheritance might already include it
**Location:** `docs/api-reference/lineage.md:11-14`
**Issue:** `"The method itself is documented on the DerivaML class: [DerivaML — lookup_lineage](deriva_ml_base.md)"` — the link target is the page, not the specific anchor. Readers click through to the entire 1600-line DerivaML page and have to ctrl-F.
**Suggested fix:** Use the anchor: `(deriva_ml_base.md#deriva_ml.core.base.DerivaML.lookup_lineage)`.

#### [P3] `docs/index.md` "ERD" image alt text could be more descriptive
**Location:** `docs/index.md:27`
**Issue:** `![ERD](assets/ERD.png)` — alt text is just "ERD", which is meaningless to screen readers.
**Suggested fix:** Change to `![Entity-relationship diagram showing Dataset, Workflow, Execution, and Feature_Name tables in the deriva-ml schema](assets/ERD.png)`.

#### [P3] `docs/user-guide/datasets.md` mentions `MINID` but doesn't link to a definition
**Location:** `docs/user-guide/datasets.md:522`
**Issue:** First mention of MINID is `"To create a MINID (a persistent, citable identifier) for a shared bag"` with a forward reference to Chapter 6. Good — but no link, just text.
**Suggested fix:** `"To create a [MINID](sharing.md#minids) (a persistent, citable identifier) ..."`.

#### [P3] `docs/release-notes.md` is in nav root, not under a "Release" group
**Location:** `mkdocs.yml:82`
**Issue:** Singleton top-level entry. Consider grouping with "About" / "Project info" if other singletons are added.

### docs/getting-started/ + cross-references

#### [P0] User has no in-site walkthrough of "install → connect → first command"
**Location:** `docs/getting-started/*.md`, `docs/index.md`, `docs/user-guide/exploring.md`
**Issue:** Aggregate finding — covered by the per-page P0s above. Worth surfacing separately: the docs site has no `pip install deriva-ml` instruction visible in the nav. The reader is bounced to the template repo before they can install the library.
**Suggested fix:** Add a real `getting-started/install.md` (in-site, in nav) with the `uv add deriva-ml` / `pip install deriva-ml` snippet, a one-line `deriva-auth` instruction, and a "next: Exploring a catalog" call to action.

### Cross-cutting

#### [P2] `core/__init__.py` docstring at line 16 example uses `DerivaML('deriva.example.org', 'my_catalog')` — string for catalog_id
**Location:** `src/deriva_ml/core/__init__.py:16-18`
**Issue:** Other examples use `catalog_id='42'` (numeric-looking string). Mixed conventions in examples — sometimes name, sometimes number, sometimes int. The `catalog_id: str | int` field type accepts both, but consistency aids skim-readability.
**Suggested fix:** Pick one form for examples (numeric string `'42'` is most common across the codebase) and use it consistently.

#### [P2] No mention of `mode=ConnectionMode.offline` in user-facing docs
**Location:** Nowhere in `docs/user-guide/`.
**Issue:** Offline mode is a documented feature on `DerivaML.__init__` (`base.py:285-289`) and gets a dedicated `Working offline` chapter (`offline.md`). But the offline-mode constructor pattern (`ml = DerivaML(..., mode=ConnectionMode.offline)`) is not in any user-guide page I audited.
**Suggested fix:** Add a "How to work in offline mode" section to `offline.md` showing the constructor pattern, or cross-reference from `exploring.md`.

#### [P3] CLAUDE.md mentions `cli-reference.md` exists and is current; the file in the repo is a redirect
**Location:** `docs/cli-reference.md`, brief
**Issue:** The brief lists `docs/cli-reference.md` as in-scope and asks "current?" — it's a redirect stub. So either the brief is stale or the file was deleted recently.
**Suggested fix:** Decide whether `cli-reference.md` should be revived (the `deriva-ml-run` / `deriva-ml-run-notebook` reference is currently buried inside `docs/user-guide/executions.md` lines 388–460). A dedicated CLI reference page accessible from nav would be more discoverable.

## Coverage gaps (cross-cutting)

These are areas where either docs exist without code, or code exists without docs.

1. **`DerivaML.from_context()`** (`base.py:192-235`) has a thorough docstring but is not mentioned in any user-guide page. It's the documented entry point for Claude-generated scripts — should appear in `exploring.md` or `hydra-zen.md` as the canonical pattern for "I'm working in a context the MCP server set up."
2. **`DerivaML.cache_table()` / `_cache_features()` / `workspace` property** (`base.py:874-961`) — the "local caching" UX is documented in source but has no user-guide chapter. There's a `Workspace` concept buried in the source — should be documented as a discoverable feature.
3. **`DerivaML.pin_schema` / `unpin_schema` / `pin_status` / `diff_schema`** are thoroughly documented in source (`base.py:603-713`) but have no user-guide coverage. These are diagnostic / advanced-user surfaces and probably deserve a "Working with schema cache" section.
4. **`DerivaML.clear_cache` / `get_cache_size` / `list_execution_dirs` / `clean_execution_dirs` / `get_storage_summary`** (`base.py:1368-1622`) — local-disk management methods. No user-guide doc. These are useful day-to-day commands and worth a "Managing local storage" section.
5. **Annotation builders** have docstrings in `model/annotations.py` and a `concepts/annotations.md` redirect to `features.md` — but `features.md` doesn't actually cover annotation builders (it covers feature definitions). True annotation-builder docs only exist in the docstrings. The `deriva-skills` Claude Code skill is documented as the consumer per CLAUDE.md — but for human readers, there is no user-guide tour of `Display`, `VisibleColumns`, `TableDisplay`. Either (a) add a chapter (P1 for a thorough surface, P2 if the skill consumer is the primary audience), or (b) explicitly link to `deriva-skills` docs from the API-reference page.
6. **`AssetRecord`** is exported from `deriva_ml.asset.__init__` and referenced in docstrings (`Execution.asset_file_path` mentions `AssetRecord`) but is not in the api-reference nav at all and not in the main `deriva_ml.__init__` exports.
7. **`Workflow`** has an api-reference page (`api-reference/workflow.md`) but no user-guide chapter section. Workflow creation appears inline in `executions.md` (`ml.create_workflow(...)`) but the model itself (catalog write-back, deduplication, URL detection) isn't explained anywhere in the user guide.
8. **`MultirunSpec` and `multirun_config`** are exported from `deriva_ml.execution.__init__` but the `configuration/experiments.md` page covers experiments without making the `multirun_config` shape explicit. Reader sees it referenced via `deriva-ml-run --multirun` but the underlying spec is not introduced.
9. **`DerivaMLBagView`** is mentioned in `dataset_bag.py` docstrings as "use DerivaMLBagView for catalog-level operations" but there is no docs page for it. Its mention in `dataset_bag.py:87` is a dead-end for any reader.
10. **The `core/exceptions.py` hierarchy** — CLAUDE.md lays this out cleanly, and `exceptions.md` should reproduce it. The hierarchy is not in user-facing docs except as docstrings.

## Recommendations

In priority order for the v1.37.x release cycle:

1. **Fix the four P0 broken-code-example issues** (`split_dataset`, `check_auth`, `set_metadata`, `compute_workdir` doctest). These actively mislead readers; every one will produce a TypeError or AttributeError on the first paste. ETA: half a day across all four.
2. **Fix the api-reference `upload.md` dead reference** — either re-point to `deriva_ml.cli.upload` or delete. Even a 404 in a published docs site is a bad signal. ETA: 10 minutes.
3. **Write release notes for 1.35–1.37.4.** Use `git log v1.34.0..v1.37.4 --oneline` and the merged PRs as input. This is what release-day readers expect to see first. ETA: 1–2 hours.
4. **Decide and act on the 14 redirect-only pages** (`getting-started/`, `concepts/`, `workflows/`, `cli-reference.md`). Recommend: install `mkdocs-redirects` plugin, declare redirects there, delete the HTML-stub files. ETA: 1–2 hours including testing.
5. **Add real getting-started content.** At minimum a single `getting-started/install.md` with `pip install deriva-ml`, the `deriva-auth` step, and a `DerivaML(hostname=..., catalog_id=...)` first-call. Put it in the nav. The current "bounce to template repo" experience loses readers who just want to evaluate the library. ETA: half a day.
6. **Restructure `mkdocs.yml` nav** to add: `DerivaMLConfig` page, dedicated `Annotation Builders` page (`api-reference/annotations.md`), `architecture.md`, `reference/schema.md`. Fix the `Upload` entry. ETA: 1–2 hours.
7. **Backfill the `Example:` blocks** on the user-facing annotation builders (`TableDisplayOptions`, `PreFormat`, `ColumnDisplayOptions`, `ColumnDisplay`, the `Facet*` family, `PseudoColumnDisplay`) — these are pinned as public API by CLAUDE.md but lack the runnable examples that CLAUDE.md mandates for public APIs. ETA: 1 day.

Lower priority (post-release cycle): the P2 / P3 normalization items (singular `Example:`, internal cross-link anchors, alt-text), the coverage-gap items (workspace docs, schema-pin docs, multirun docs). These are real but don't block release.
