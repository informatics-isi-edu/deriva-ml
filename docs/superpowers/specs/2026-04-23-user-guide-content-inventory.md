# User-Guide Content Inventory

**Purpose:** migration map from cut-list pages to User Guide chapters.
Every source sentence has a destination (or a retirement rationale).
Used as source-of-truth during chapter drafting.

---

## Destination: Introduction (docs/index.md rewrite)

### From docs/concepts/overview.md
- Four-object mental model table: Dataset / Workflow / Execution / Execution Asset (adapt to Catalog / Dataset / Execution / Feature)
- ERD image reference (`docs/assets/ERD.png`) — keep on Introduction page
- Domain schema vs ML schema explanation (two-schema architecture)
- `deriva-ml` library description: execution lifecycle management, context manager, output upload with provenance
- Domain-specific library pattern (DerivaML subclass, e.g., EyeAI)
- RETIRED from overview: "Next Steps" nav links — replaced by new nav structure

### From docs/getting-started/quick-start.md
- RETIRED: all content — onramp story (install, template repo, run model, run notebook) is owned by the template repo. Introduction will contain a single pointer to the template repo.

### From docs/getting-started/install.md
- RETIRED: all content — install instructions (uv, `uv add deriva-ml`, Globus auth, Jupyter kernel) are owned by the template repo. One pointer from Introduction suffices.

### From docs/getting-started/project-setup.md
- RETIRED: all content — project structure, config layout, DerivaML subclass pattern, dependency groups, notebook setup, `uv.lock` are owned by the template repo. The DerivaML subclass pattern (a short code sketch) may be re-homed to Chapter 1 as a "what you'll see in real projects" note.

---

## Destination: Chapter 1 — Exploring a catalog (docs/user-guide/exploring.md)

### From docs/concepts/overview.md
- Connecting to a catalog: `DerivaML(hostname, catalog_id)` constructor
- Brief description of domain-specific DerivaML subclasses as what users encounter in real projects (see also project-setup.md)
- ML schema tables list (Dataset, Workflow, Execution, Execution Asset, Execution Metadata) — use as "what you can find" orientation

### From docs/concepts/identifiers.md
- ALL content: RID concept — global unique identifiers, FAIR principle motivation
- RID format: dash-separated four-character blocks (e.g., `1-000C`)
- Unqualified RID refers to current catalog values
- Snapshot-qualified RID: `1-000C@32S-W6DS-GAMG` — refers to a specific point in time
- URI form: `https://www.eye-ai.org/id/1-000C@32S-W6DS-GAMG` — for cross-catalog references
- `DerivaML.cite()` method — obtains URI form of a RID (cross-ref to API Reference)
- Fix spelling errors in source ("seperated", "unqualified") when re-homing

### From docs/getting-started/project-setup.md
- DerivaML subclass pattern — short code sketch showing `class EyeAI(DerivaML)` and `get_oct_images()` as what users encounter in domain catalogs

### From docs/workflows/execution-lifecycle.md
- `find_workflows()` — listing all workflows in the catalog (lines 196-201)
- `lookup_workflow()` by RID (lines 189-190)
- `lookup_workflow_by_url()` (lines 192-193)
- Workflow_Type vocabulary terms (Training, Inference, Preprocessing, Evaluation, Annotation table)
- `ml.add_term(table="Workflow_Type", ...)` for custom types

### From docs/concepts/datasets.md
- `ml.find_datasets()` — listing all datasets with RID, description, version (lines 747-754)
- `ml.lookup_dataset(rid)` — finding a specific dataset by RID (line 755)

### From docs/concepts/features.md
- `ml.find_features(target_table)` — listing feature definitions for a table (lines 274-283)
- `ml.find_features()` — listing all features in the catalog (line 284)
- `ml.lookup_feature(target_table, feature_name)` — inspecting a feature's schema (lines 286-298)

### From docs/cli-reference.md
- `deriva-ml-check-catalog-schema` — for checking catalog schema validity (administration tool, note as such)
- `deriva-ml-create-schema` — for setting up a new catalog (administration tool, note as such)
- NOTE: `get_chaise_url()` is referenced in the spec's Chapter 1 outline but not in any source page; flag for chapter author to add from API source reading.

---

## Destination: Chapter 2 — Working with datasets (docs/user-guide/datasets.md)

### From docs/concepts/datasets.md (primary — 770 lines)

**Concept setup:**
- What a dataset is: versioned collection of objects, heterogeneous types, identified by RID
- Datasets can reference objects via multiple paths (subject + observation that references subjects)
- Dataset types assigned from `Dataset_Type` controlled vocabulary (orthogonal tagging)

**Creating and populating datasets:**
- Code example: `ml.add_term(MLVocab.dataset_type, ...)` for defining dataset types (lines 13-17)
- Code example: create dataset within execution context using `exe.create_dataset(dataset_types=[...], description=...)` (lines 30-58)
- Code example: create dataset directly via `ml.create_dataset(...)` (lines 64-69)
- `dataset.list_dataset_element_types()` — what element types are available (lines 75-80)
- `ml.add_dataset_element_type("Subject")` — registering a new element type (lines 82-87)
- Code example: `dataset.add_dataset_members(members=[...])` by RID list (lines 93-101)
- `dataset.add_dataset_members(members=..., execution_rid=...)` — with execution tracking

**Listing and browsing:**
- `dataset.list_dataset_members()` — all members of current version (lines 107-112)
- `dataset.list_dataset_members(version="1.0.0")` — members at a specific version (lines 113-116)

**Nested datasets:**
- Code example: parent/child dataset hierarchy (lines 119-139)
- `parent_dataset.add_dataset_members(members=[child.dataset_rid, ...])` — nesting via member addition
- `parent_dataset.list_dataset_children()` — navigating down
- `training_dataset.list_dataset_parents()` — navigating up

**Splitting datasets:**
- `split_dataset` conceptual hierarchy diagrams (two-way and three-way, lines 149-167)
- `SplitResult` and `PartitionInfo` return types
- Code example: simple random 80/20 split (lines 174-184)
- Code example: fixed-count split (lines 186-194)
- Code example: three-way train/validation/test split (lines 198-213)
- Code example: labeled splits with `training_types=["Labeled"]` (lines 219-234)
- Code example: stratified split with `stratify_by_column` (lines 237-252)
- `!!! note`: scikit-learn required for stratified splitting (line 252-254)
- Code example: custom selection function with `SelectionFunction` protocol (lines 257-283)
- Code example: dry run with `dry_run=True` (lines 289-302)
- `stratify_by_column` format: `{TableName}_{ColumnName}` — document explicitly as a pitfall
- Auto-detection of element table when only one type present (lines 330-339)
- `element_table` parameter required when dataset has multiple element types

**Versioning:**
- Semantic versioning: major.minor.patch semantics (schema change / new elements / minor alterations)
- Auto-assignment of `0.1.0` on creation, auto-increment of minor on member add
- Code example: `dataset.current_version`, `dataset.dataset_history()` (lines 352-368)
- Code example: `dataset.increment_dataset_version(component=VersionPart.minor, ...)` (lines 363-369)
- Version snapshots: each version tied to a catalog snapshot for reproducibility
- Code example: `dataset.set_version("1.0.0")` — dataset bound to that snapshot (lines 375-381)

**Downloading datasets:**
- Code example: `dataset.download_dataset_bag()` — current version (line 390)
- Code example: `dataset.download_dataset_bag(version="1.0.0")` — specific version (line 393)
- Code example: `dataset.download_dataset_bag(materialize=True)` — with asset files (line 396)
- Download timeouts — `timeout=(10, 1800)` for large datasets (lines 403-424)
- `DatasetSpec` with `timeout` for execution configurations (lines 415-424)
- `dataset.estimate_bag_size(version=...)` — row counts and asset size before downloading (lines 428-445)
- Bag export FK traversal behavior: stops at other dataset element types, traverses non-element tables normally (lines 447-455, important edge case)
- Code example: automatic download in executions with `DatasetSpec` in `ExecutionConfiguration` (lines 458-477)
- `materialize=False` flag — metadata only, faster for large datasets

**Working with DatasetBag:**
- `bag.dataset_rid`, `bag.version` — metadata access (lines 483-487)
- `bag.get_table_as_dataframe("Subject")` — tables as DataFrames (lines 488-490)
- `bag.path` — local filesystem path (line 492)

**Assets in datasets:**
- Asset characteristics: versioned URL, Length, MD5, Asset_Type (lines 499-503)
- `bag.get_table_as_dataframe("Image")` — asset table access (lines 507-513)
- Asset organization after materialization: same-type assets in same directory (lines 519-525)
- Symlinks as efficient reorganization option (line 525)

**restructure_assets:**
- Code example: `bag.restructure_assets(asset_table, output_dir, group_by)` (lines 531-543)
- Output directory structure diagram (`ml_data/Complete/Training/positive/image1.jpg`) (lines 545-561)
- `group_by` with column names and feature names (lines 563-588)
- `use_symlinks=True/False` — symlinks vs copies (lines 591-609)
- `type_selector` lambda for custom type selection (lines 612-622)
- `Unknown` folder for missing grouping values (lines 624-635)
- Prediction scenarios: datasets without types treated as Testing (lines 638-668)
- FK path traversal: `bag.restructure_assets` finds assets via indirect FK paths (lines 671-689)
- `value_selector` for multiple feature values per asset (lines 691-723)
- `FeatureValueRecord` attributes: `target_rid`, `feature_name`, `value`, `execution_rid`, `raw_record` (lines 713-722)
- `enforce_vocabulary=True/False` — vocabulary-based grouping validation (lines 726-744)

**Delete:**
- `ml.delete_dataset(dataset)` — soft delete (lines 763-770)
- `ml.delete_dataset(dataset, recurse=True)` — recursive delete

### From docs/workflows/execution-lifecycle.md
- Dataset versioning interplay with executions: `exe.create_dataset(...)` for provenance-tracked dataset creation (lines 627-644)
- Resuming executions to access previously created datasets (lines 648-657)

### From docs/cli-reference.md
- `deriva-ml-split-dataset` CLI — full synopsis and arguments table (lines 275-329)
- All `deriva-ml-split-dataset` examples (lines 310-329)
- `deriva-ml-split-dataset --dry-run` flag
- `--val-size` argument for three-way split (NOTE: check if this is present in CLI; source concepts/datasets.md shows it for Python API; CLI reference shows `--test-size`, `--train-size` but not `--val-size` explicitly — flag for chapter author verification)

---

## Destination: Chapter 3 — Defining and using features (docs/user-guide/features.md)

### From docs/concepts/features.md (primary — 460 lines)

**What a feature is:**
- Four properties: tracks provenance, uses controlled vocabularies, supports multiple values, versioned (lines 9-13)
- Common use cases table: ground truth labels, model predictions, quality scores, derived measurements, related assets (lines 16-22)

**Feature types:**
- Term-based features — controlled vocabulary, consistency, hierarchy queries (lines 30-45)
- Code example: `ml.create_vocabulary(...)` + `ml.add_term(...)` + `ml.create_feature(target_table, feature_name, terms=[...])` (lines 33-45)
- Asset-based features — segmentation masks, embeddings, computed files (lines 48-64)
- Code example: `ml.create_asset(...)` + `ml.create_feature(..., assets=[...])` (lines 52-64)
- FeatureRecord class auto-generated: accepts file path or asset RID; paths replaced with RIDs on upload (lines 63-65)
- Mixed features: terms + assets (lines 67-68)

**Creating feature values:**
- Workflow: get FeatureRecord class → create instances → `execution.add_features()` (lines 73-75)
- Code example: full term-based feature value creation in execution context (lines 79-113)
- `ml.feature_record_class("Image", "Diagnosis")` — getting the FeatureRecord class (line 87)
- Code example: asset-based feature values with `exe.asset_file_path()` + file path in FeatureRecord (lines 120-153)
- Upload process: 1) upload asset files 2) replace file paths with RIDs 3) insert feature records (lines 155-161)

**Querying feature values:**
- `ml.feature_values("Image", "Diagnosis")` — streaming iterator of typed FeatureRecord instances (lines 165-176)
- FeatureRecord attributes: `Image`, `Diagnosis_Type`, `Execution`, `RCT` (lines 182-188)
- Converting to pandas DataFrame with `model_dump()` (lines 190-196)
- Selector pattern: `(list[FeatureRecord]) -> FeatureRecord` callable signature (lines 200-206)
- `FeatureRecord.select_newest` — most common selector, picks latest RCT (lines 208-220)
- Custom selector example: `select_highest_confidence` (lines 222-237)
- `FeatureRecord.select_by_workflow(workflow_or_type, container=ml)` — selector factory (lines 239-268)
- `select_by_workflow` accepts workflow RID or Workflow_Type name (lines 261-267)
- Silent omission when no records match the workflow (lines 268-270)
- `ml.find_features("Image")` — feature definitions for a table (lines 273-283)
- `ml.find_features()` — all features in catalog (line 284)
- `ml.lookup_feature("Image", "Diagnosis")` — schema descriptor (lines 286-298)
- Feature.`term_columns`, `asset_columns`, `value_columns` inspection (lines 294-297)
- `feature.feature_record_class()` — get FeatureRecord class from Feature object (lines 299-302)

**Feature tables structure:**
- Association table columns: TargetTable RID, Feature_Name, Execution RID, VocabTable RID, AssetTable RID (lines 311-319)
- Enables: querying all values, finding producing execution, joining vocab, many-to-many (lines 320-325)

**Best practices:**
- Descriptive feature names (lines 331-333)
- Feature names are Feature_Name vocabulary terms (line 333)
- Same feature name can be used across tables (line 334)
- Create vocabularies before features (lines 337-340)
- Always create feature values within an Execution context (lines 343-346)

**Multiple values / multi-annotator scenarios:**
- Common cases: multiple annotators, model runs at different times, different analysis versions (lines 349-353)
- Code example: querying multiple values for a specific image (lines 355-363)
- Code example: `FeatureRecord.select_newest` deduplication (lines 366-370)
- Code example: `FeatureRecord.select_by_workflow("Manual_Annotation", ...)` (lines 371-381)
- `value_selector` in `restructure_assets` (lines 384-401)
- `FeatureValueRecord` attributes table (lines 403-411)
- Majority vote custom selector example (lines 413-425)

**Deleting features:**
- `ml.delete_feature("Image", "Diagnosis")` — warning: permanent, removes feature table + all values + provenance (lines 429-434)

**Features in datasets:**
- Feature tables exported as CSVs in BDBag (line 443)
- Feature values loaded into local SQLite on bag download (line 444)
- Versioning via catalog snapshots (line 445)
- `bag.feature_values(...)` — same API as live catalog (lines 449-460)
- `FeatureRecord.select_by_workflow(..., container=bag)` — workflow lookup scoped to bag (lines 459-460)

### From docs/concepts/denormalization.md
- ALL content re-homed here as a "When to use Denormalization vs. feature_values" subsection
- Mental model: one row per leaf, star-schema denormalization (lines 18-32)
- Example table output showing hoisted columns (lines 34-50)
- Dataset membership as filter (lines 52-59)
- API: `ds.get_denormalized_as_dataframe(include_tables=[...])` (lines 62-76)
- `ds.get_denormalized_as_dict(...)` — streaming rows (line 73)
- Preview methods: `ds.list_denormalized_columns(...)`, `ds.describe_denormalized(...)`, `ds.list_schema_paths()` (lines 78-99)
- `Denormalizer.from_rids(rids, ml=ml_instance)` — arbitrary RID anchors without a dataset (lines 101-121)
- Parameters: `include_tables`, `row_per`, `via`, `ignore_unrelated_anchors` (lines 123-162)
- Rules 1-7 covering edge cases: row cardinality, auto-inference, column projection, column hoisting, downstream rejection, path ambiguity, orphan anchors (lines 164-226)
- Examples by use case: hoisting upward, all diagnoses per subject, diamond path with `via`, feature values on images, heterogeneous datasets with orphan members, arbitrary anchors (lines 229-287)
- Exploration workflow: `list_schema_paths` → `list_denormalized_columns` → `describe_denormalized` → `get_denormalized_as_dataframe` (lines 290-315)
- API rename table (earlier names vs. current names) (lines 317-329)
- Two behavioral changes: ambiguous FK paths now error; feature tables fully supported as leaves (lines 331-338)

---

## Destination: Chapter 4 — Running an experiment (docs/user-guide/executions.md)

### From docs/workflows/execution-lifecycle.md (primary)

**Workflow / Execution / Workflow_Type hierarchy:**
- Conceptual hierarchy diagram: Workflow_Type → Workflow → Execution (lines 19-34)
- `ml.create_workflow(name, workflow_type, description)` (lines 38-48)
- Deduplication: same URL/checksum reuses existing workflow RID (lines 49-64)
- Implicit workflow creation via hydra-zen (lines 67-94) — pointer to Chapter 8
- Automatic source code detection: Python scripts (lines 108-126)
- Automatic source code detection: Jupyter notebooks + nbstripout (lines 128-147)
- Automatic source code detection: Docker containers + env vars table (lines 148-163)
- Overriding detection with explicit `url` and `checksum` (lines 165-181)
- `ml.lookup_workflow(rid)`, `ml.lookup_workflow_by_url(url)`, `ml.find_workflows()` (lines 186-201)
- Updating workflow `description` and `workflow_type` on bound workflows (lines 204-210)
- Workflow types: built-in categories + `ml.add_term("Workflow_Type", ...)` for custom (lines 213-238)
- Two ways to provide workflow to execution (Option 1 in config, Option 2 as arg) (lines 241-263)

**Execution status lifecycle:**
- ASCII state diagram: created → initializing → pending → running → completed / failed (lines 267-301)
- Status values table with when-set and description (lines 302-315)
- Context manager handles transitions automatically (lines 316-317)
- `exe.update_status(Status.running, "message")` for progress reporting (lines 318-325)

**Execution metadata:**
- Catalog records created: Execution, Dataset_Execution, Execution_Asset_Execution (lines 329-342)
- Automatic metadata files table: Deriva_Config, Hydra_Config, Execution_Config, Runtime_Env (lines 344-353)
- Metadata uploaded during initialization (before model runs) (lines 355-356)
- `config_choices` field in ExecutionConfiguration for Hydra group selections (lines 358-371)

**ExecutionConfiguration:**
- Code example: `ExecutionConfiguration(workflow, description, datasets, assets)` (lines 381-404)
- `DatasetSpec` options table: `rid`, `version`, `materialize` (lines 406-419)
- DatasetSpec examples: materialize, metadata only, specific version (lines 420-424)
- `AssetSpec` with `cache=True/False` — MD5-based caching for large immutable assets (lines 425-469)
- Asset caching behavior table (lines 451-454)
- `AssetSpecConfig` for hydra-zen configurations (lines 462-469)

**Running an execution:**
- Code example: full context manager pattern with bag download, processing, asset registration (lines 471-502)
- Context manager behavior: entry sets status "running", exit records stop time, exception sets "failed" (lines 504-510)
- Why upload is separate from context manager: async, inspection, retry, partial results (lines 512-518)
- `upload_execution_outputs()` tuning parameters: `timeout`, `chunk_size`, `max_retries`, `retry_delay` (lines 520-558)
- `!!! note`: urllib3 uses connect_timeout for write socket timeout (lines 548-551)
- Upload parameters table (lines 553-558)

**Registering output files:**
- Code example: five `asset_file_path()` methods (new path, stage existing, rename, apply types, provide metadata) (lines 561-603)
- `path.set_metadata(key, value)` — post-registration metadata update (line 601)
- `path.set_asset_types([...])` — post-registration type update

**Status updates:**
- Code example: `exe.update_status(Status.running, "message")` in training loop (lines 606-623)

**Creating output datasets:**
- Code example: `exe.create_dataset(description, dataset_types)` + `add_dataset_members` (lines 625-644)

**Resuming executions:**
- `ml.resume_execution("1-XYZ")` — resume by RID (lines 646-657)

**Nested executions:**
- Conceptual hierarchy: parent as container for parameter sweeps/cross-validation (lines 659-679)
- Automatic nesting with multirun: `multirun_config()` in `configs/multiruns.py` (lines 681-718)
- Manual nesting: `parent.add_nested_execution(child, sequence=fold)` (lines 719-746)
- `parent.list_nested_executions()`, `list_nested_executions(recurse=True)` (lines 747-764)

**Complete example:**
- Code example: full training run from connect to upload (lines 766-817) — preserve in chapter

### From docs/workflows/running-models.md (CLI subsection)

- `deriva-ml-run` basic usage: run defaults, override group, override param, dry run, --info (lines 135-155)
- `--host` / `--catalog` override from command line (lines 157-163)
- Model function protocol: `ml_instance` + `execution` injected, configurable params from hydra (lines 55-92)
- `execution.datasets`, `execution.asset_paths` — accessing downloaded inputs inside model function (lines 70-79)
- Troubleshooting section: "Config directory not found", "Uncommitted changes" warning, "Kernel not found", override syntax errors, "Notebook ignores Hydra overrides" (lines 288-305) — keep as common pitfalls

### From docs/cli-reference.md

- `deriva-ml-run` full synopsis, arguments table, all examples (lines 31-90)
- `deriva-ml-run-notebook` full synopsis, arguments table, environment variables table, all examples (lines 92-165)
- `deriva-ml-install-kernel` synopsis (lines 225-261) — may belong in an appendix note
- `bump-version` synopsis + examples (lines 170-221) — belongs in Chapter 7 (reproducibility); cross-reference here

---

## Destination: Chapter 5 — Working offline (docs/user-guide/offline.md)

### From docs/concepts/datasets.md (bag sections)
- `DatasetBag` object: `bag.dataset_rid`, `bag.version`, `bag.path` (lines 483-492)
- `bag.get_table_as_dataframe("Subject")` — tables as DataFrames (lines 488-490)
- `dataset.download_dataset_bag()` / `download_dataset_bag(version=..., materialize=True/False)` (lines 388-398)
- Download timeouts (`timeout` tuple) (lines 403-424)
- `dataset.estimate_bag_size()` (lines 428-445)
- Bag FK traversal behavior (boundary-aware export) (lines 447-455)
- `restructure_assets()` return type: `dict[Path, Path]` — manifest, not a Path (CLAUDE.md)
- All `restructure_assets()` subsections: grouping options, symlinks vs copies, custom type selection, missing values, prediction scenarios, FK path traversal, multiple feature values, enforce_vocabulary (lines 528-744)

### From docs/concepts/features.md
- `bag.feature_values(...)` — same API as live catalog (lines 449-460)
- `FeatureRecord.select_by_workflow(..., container=bag)` — workflow lookup scoped to bag (lines 459-460)
- `bag.lookup_feature(...)` — returns Feature whose `feature_record_class()` works offline (implied by API parity)

### From docs/workflows/execution-lifecycle.md
- `exe.download_dataset_bag(DatasetSpec(...))` — download inside execution (lines 471-502, partial)

---

## Destination: Chapter 6 — Sharing and collaboration (docs/user-guide/sharing.md)

### From docs/concepts/file-assets.md (partial — bag-related and provenance sections)
- Asset characteristics: URL (Hatrac), Filename, Length, MD5, Description columns (lines 7-17)
- `Asset_Type` vocabulary: filtering, Chaise UI, consistent categorization (lines 20-39)
- Assets-and-provenance section: `{Asset}_Execution` table linking assets to executions with Input/Output role (lines 228-248)
- Querying assets produced by an execution via `pathBuilder()` (lines 239-248)
- Assets in datasets: `ml.add_dataset_element_type("Image")`, `dataset.add_dataset_members(image_rids)` (lines 251-261)
- Bag download with `materialize=True/False` (lines 262-276)
- Working directory layout: `asset-manifest.json`, `assets/{Table}/` (lines 279-292)
- Crash recovery with `ml.resume_execution(execution_rid)` + `upload_execution_outputs()` (lines 293-300)

### From docs/concepts/datasets.md
- MINID creation: `use_minid=True` when downloading — pointer to this as citable identifier (mentioned implicitly via `download_dataset_bag` parameters; verify in source code / CLAUDE.md)
- BDBag format context: archives for offline use and sharing (lines 384-398)

### Net-new content (from CLAUDE.md and source code reading)
- `create_ml_workspace` — partial catalog clones, three-stage approach (schema + data + FKs), orphan handling
- Orphan strategies: FAIL, DELETE, NULLIFY
- `localize_assets` — copying Hatrac assets after `asset_mode=REFERENCES` clone
- `create_ml_schema` CASCADE warning: never call on populated catalog
- Access control: brief note + link to Deriva/Globus docs

---

## Destination: Chapter 7 — Reproducibility (docs/user-guide/reproducibility.md)

### From docs/workflows/git-and-versioning.md (primary)
- Core principle: always commit before running (lines 7-20)
- Warning behavior: uncommitted changes causes warning; execution record may not reflect actual code (lines 21-24)
- Debugging workflow: dry run pattern + commit before real run (lines 26-43)
- `dry_run=true` — downloads data, skips catalog writes (lines 30-32)
- Branching strategy guidance (lines 47-70)
- Semantic versioning: major/minor/patch semantics table (lines 75-84)
- `bump-version patch/minor/major` command (lines 86-98)
- Version check: `uv run python -m setuptools_scm` (lines 100-113)
- When to version: before important runs, before sweeps, during development (lines 115-128)
- `nbstripout --install` — strip notebook output cells for reproducibility (lines 137-144)
- Notebook structure guidelines: sequential cells, Papermill parameters cell, restart+run-all test (lines 146-154)
- Commit before running notebooks (lines 156-163)
- Don't commit large files to Git — use DerivaML assets and datasets instead (lines 165-177)
- What DerivaML records per execution: git commit hash, version tag, repository URL, branch name (lines 179-184)

### From docs/workflows/execution-lifecycle.md
- Automatic source code detection full description (scripts, notebooks, Docker) (lines 108-163)
- Docker env vars: `DERIVA_MCP_IMAGE_DIGEST`, `DERIVA_MCP_IMAGE_NAME`, `DERIVA_MCP_GIT_COMMIT`, `DERIVA_MCP_VERSION` (lines 151-163)
- `configuration.json` content: fully resolved ExecutionConfiguration as JSON (line 348-350)
- `uv.lock` capture as `Execution_Config` asset type (line 352)
- `environment_snapshot_<timestamp>.txt` — Python version, packages, OS, GPU, env vars (line 353-354)
- `config_choices` field — Hydra group selections for exact run reproduction (lines 358-371)

### From docs/concepts/datasets.md (version pinning sections)
- Version snapshots: catalog snapshot tied to each version ensures same rows always (lines 373-381)
- `dataset.set_version("1.0.0")` — bind dataset to specific version (lines 375-381)
- `DatasetSpec(rid, version="1.0.0")` — pinning in execution configuration (lines 416-419)

### From docs/cli-reference.md
- `bump-version` synopsis and examples (re-home here, cross-reference from Chapter 4) (lines 170-221)

---

## Destination: Chapter 8 — Integrating with hydra-zen (docs/user-guide/hydra-zen.md)

### From docs/workflows/running-models.md (primary)
- What hydra-zen does: composable Python-first config, no YAML (lines 1-10)
- Project layout with `src/configs/` structure (lines 17-37)
- How configuration discovery works: `load_configs()`, `pkgutil.iter_modules()`, alphabetical order, `experiments.py` last (lines 42-51)
- Model function protocol: `ml_instance` + `execution` injected at runtime (lines 53-92)
- `base.py` with `create_model_config()` wiring (lines 97-128)
- `create_model_config(EyeAI)` for domain subclasses (lines 126-129)
- Complete walkthrough steps 1-6 (lines 168-285) — condensed for chapter; full detail stays in template repo
- Troubleshooting section (lines 288-305) — include as common pitfalls / See also

### From docs/workflows/execution-lifecycle.md
- Implicit workflow creation via hydra-zen config in `configs/workflow.py` (lines 67-94)
- Hydra multirun nesting: `multirun_config()`, parent+child execution creation (lines 681-718)
- `config_choices` captured in ExecutionConfiguration (lines 358-371)

### From docs/cli-reference.md
- `deriva-ml-run` full CLI reference (re-home partial here; full detail in Chapter 4) (lines 31-90)
- `deriva-ml-run-notebook` full CLI reference (lines 92-165)

### From docs/getting-started/project-setup.md
- Config modules list: `deriva.py`, `datasets.py`, `assets.py`, `workflow.py`, `my_model.py`, `experiments.py`, `multiruns.py` (lines 31-43) — orients reader to config group names
- NOTE: full project setup belongs to template repo; this chapter only orients to the config surface

---

## Retirements (content deliberately dropped with rationale)

### From docs/getting-started/quick-start.md
- All content (Steps 1-5: uv install, template repo setup, Globus auth, `deriva-ml-run`, `deriva-ml-run-notebook`, Next Steps links) — template repo owns the onramp story; Introduction contains the pointer.

### From docs/getting-started/install.md
- All content (Python 3.12 prereq, `uv add deriva-ml`, `uv pip install`, Globus auth, `deriva-ml-install-kernel`, `nbstripout --install`, version check, upgrade) — template repo owns install. Python 3.12 requirement may be worth one line in Introduction but is not a user-guide chapter concern.

### From docs/getting-started/project-setup.md
- Project structure layout with `src/configs/`, `src/models/`, `notebooks/` directories — belongs to template repo README; user guide just orients to the config group names.
- Dependency groups: `uv sync --group=jupyter/pytorch/docs` — template repo concern.
- `default-groups` in `pyproject.toml` — template repo concern.
- Notebook setup commands (install kernel, nbstripout, verify kernelspec) — covered in Chapter 4 (notebook note) and Chapter 7 (reproducibility); not worth a standalone section in user guide.
- `uv sync --upgrade-package`, `uv lock --upgrade` — template repo concern; not deriva-ml-specific.
- "Next Steps" nav links — replaced by new nav.

### From docs/concepts/overview.md
- "Next Steps" nav links — replaced by new nav.
- The existing table of ML schema tables (Dataset, Workflow, Execution, Execution Asset, Execution Metadata) is partially stale relative to S2 (Feature is now a first-class object, not in the table); incorporate accurate version into Introduction and Chapter 1, drop the stale table.

### From docs/concepts/identifiers.md
- Spelling errors ("seperated", "unqualified" used incorrectly) — fix when re-homing, do not preserve errors.
- Cross-reference link to `../code-docs/deriva_ml_base.md#...` — rewrite to `../api-reference/deriva_ml_base.md#...` when re-homing.

### From docs/concepts/annotations.md
- ALL content — the annotation builder API (`TableHandle`, `Display`, `VisibleColumns`, `TableDisplay`, `ColumnDisplay`, `VisibleForeignKeys`, `PseudoColumn`, `Facet`, Handlebars templates) is a **schema administration** concern, not a user-guide concern. This belongs in a developer/admin reference section, not the user guide. The annotation builders are not part of the ML workflow. RETIRE to API Reference or a future "Schema Administration" section. Not re-homed into any chapter. Rationale: the spec's Chapter outlines make no mention of catalog annotations; this is Chaise configuration work that ML practitioners do not need to do themselves.
- The `!!! note` about `ml.get_chaise_url("TableName")` — the *method call* is worth a one-liner in Chapter 1 ("Jumping to Chaise"); the annotation configuration context is not.
- API Reference cross-links to `annotations.md` and `handles.md` — those code-docs pages will be renamed to `api-reference/`; links will be corrected in Task 11.

### From docs/concepts/denormalization.md
- "Related concepts" links at the bottom (lines 340-345) — internal links rewritten per Task 12.
- Link to `../superpowers/specs/2026-04-15-unified-local-db-design.md` — this is an internal design doc, not user-facing; drop from chapter.
- Earlier API name table (lines 317-329) — include in Chapter 3 as a historical note for users upgrading from pre-S2 (one paragraph, not a full section).

### From docs/workflows/running-models.md
- "See Also" cross-reference section at the bottom — rewritten to point at new chapter locations.
- "How Configuration Discovery Works" — belongs in Chapter 8 (hydra-zen) as an implementation detail note, not a standalone section in Chapter 4.
- Complete walkthrough (Steps 1-6) content — condensed in Chapter 8; the template repo owns the full project-setup walkthrough.

### From docs/workflows/git-and-versioning.md
- "See Also" section — rewritten to point at new chapter locations.
- Branching strategy guidance (lines 47-70) — RETIRE. Generic Git workflow advice is not deriva-ml-specific. One sentence in Chapter 7 ("use feature branches; all runs on a branch share the same tagged version") is sufficient. The detailed branching diagram is out of scope for the user guide.

### From docs/workflows/execution-lifecycle.md
- "See Also" cross-reference sections — rewritten to point at new chapter locations.
- The ASCII execution lifecycle overview diagram (lines 267-288) — preserved in Chapter 4 as the execution flow overview; the status state diagram (lines 300-315) is a more precise replacement for the ASCII art and should be rendered as a Mermaid diagram.

### From docs/cli-reference.md
- `create-demo-catalog` command — RETIRE to a developer/testing note. This is a testing utility for DerivaML development itself, not for end users. A brief mention in CLAUDE.md or developer docs is sufficient.
- `deriva-ml-table-comments-utils` — RETIRE to developer/admin reference. This is a schema maintenance utility for catalog administrators, not for ML practitioners.
- `deriva-ml-create-schema` and `deriva-ml-check-catalog-schema` — RETIRE from main user-guide flow; brief mention in Chapter 1 as admin tools (if a user needs to set up a catalog from scratch, they are an admin, not a typical user of these docs).
- `deriva-ml-install-kernel` — RETIRE from user guide. Belongs in project setup (template repo) and is already covered in the Getting Started section being dropped. One-line mention in Chapter 4 CLI subsection is sufficient.
- The "Command Overview" table (lines 13-26) — re-home the execution-relevant commands into Chapters 4 and 8; admin commands into a footnote.

---

## Spot-Check Verification

### Spot-check 1: docs/concepts/identifiers.md (19 lines)
- RID concept, format, snapshot-qualified form, URI form, `DerivaML.cite()` — all mapped to Chapter 1.
- Spelling errors noted for correction. Every paragraph accounted for. PASS.

### Spot-check 2: docs/concepts/features.md (460 lines)
- What a feature is (Chapter 3) ✓
- Feature types: term-based, asset-based, mixed (Chapter 3) ✓
- Creating feature values in execution context (Chapter 3) ✓
- Asset-based feature values + upload process (Chapter 3) ✓
- `feature_values()` iterator + selectors (Chapter 3) ✓
- `select_by_workflow` factory (Chapter 3) ✓
- `find_features`, `lookup_feature` (Chapter 1 for discovery; Chapter 3 for creation context) ✓
- Feature tables structure (Chapter 3) ✓
- Best practices (Chapter 3) ✓
- Multiple values / multi-annotator (Chapter 3) ✓
- `value_selector` in `restructure_assets` (Chapter 5) ✓
- `FeatureValueRecord` (Chapter 5) ✓
- Majority vote selector (Chapter 3) ✓
- `ml.delete_feature()` (Chapter 3) ✓
- Features in datasets / bag API parity (Chapter 5) ✓
- Every paragraph accounted for. PASS.

### Spot-check 3: docs/workflows/git-and-versioning.md (197 lines)
- Core principle + commit before running (Chapter 7) ✓
- Dry run debugging workflow (Chapter 7) ✓
- Branching strategy (RETIRED) ✓ with rationale
- Semantic versioning table (Chapter 7) ✓
- `bump-version` command (Chapter 7) ✓
- Version check with setuptools_scm (Chapter 7) ✓
- When to version (Chapter 7) ✓
- nbstripout (Chapter 7) ✓
- Notebook structure guidelines (Chapter 7) ✓
- Commit before running notebooks (Chapter 7) ✓
- Large files to DerivaML assets (Chapter 7) ✓
- What DerivaML records per execution (Chapter 7) ✓
- "See Also" links (RETIRED/rewritten) ✓
- Every paragraph accounted for. PASS.

---

## Content items requiring chapter-author attention (ambiguous or needs verification)

1. **`get_chaise_url()` method** — referenced in spec's Chapter 1 outline but absent from all 14 source pages. Chapter author must add from API source reading (`src/deriva_ml/core/mixins/`) or API reference.

2. **`ml.pathBuilder()` usage** — referenced in spec's Chapter 1 outline and in `file-assets.md` (one example, lines 239-248), but not deeply covered in concepts pages. Chapter author should draw from source code and CLAUDE.md patterns.

3. **MINID creation parameter** — `use_minid=True` in `download_dataset_bag` is referenced in Chapter 6 plan but not explicitly shown in any of the 14 source pages. Chapter author must verify parameter name from source code.

4. **`bag.lookup_feature()` offline behavior** — implied by API parity principle (CLAUDE.md) but not demonstrated in any source page. Chapter author should verify from source and tests.

5. **`deriva-ml-split-dataset --val-size` flag** — the Python API supports `val_size` but the CLI reference does not show this flag explicitly. Chapter author should verify CLI completeness from source.

6. **`find_executions()` method** — referenced in spec's Chapter 1 outline ("find_datasets, find_features, find_workflows, find_executions") but not in any source page. Chapter author must add from API source reading.

7. **`docs/concepts/annotations.md` retirement** — the full page is retired as admin-only content. If a future "Schema Administration" section is added to the docs, this entire page maps cleanly to it. Flag for the nav rewrite step.
