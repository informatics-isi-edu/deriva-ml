# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DerivaML is a Python library (requires Python ≥3.12) for creating and executing reproducible machine learning workflows using a Deriva catalog. It provides:
- Dataset versioning and management with BDBag support
- Execution tracking with provenance
- Feature management for ML experiments
- Controlled vocabulary management
- Asset tracking and upload
- Catalog cloning with partial data subsetting

## Environment Setup

**PATH**: `uv` is installed at `/Users/carl/.local/bin/uv`. If `uv` is not found, prefix commands with:
```bash
export PATH="/Users/carl/.local/bin:$PATH"
```

**Dirty workflow check & test catalog**: DerivaML refuses to run
workflows with uncommitted changes; tests set
`DERIVA_ML_ALLOW_DIRTY=true` to bypass. Tests also need a live
Deriva catalog at `DERIVA_HOST` (defaults to `localhost`). See
the workspace-level [`CLAUDE.md`](../CLAUDE.md) "Dirty-tree
override for testing" for the rule rationale; full suite is
30–90 minutes.

**Running long test suites from Claude Code**: The full test suite exceeds Claude Code's default 2-minute Bash timeout. Use these strategies:
1. **Run test subsets incrementally** rather than the full suite at once:
   ```bash
   # Unit tests (fast, no catalog needed, ~3 seconds)
   DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q
   
   # Dataset integration tests (needs DERIVA_HOST, ~10-30 minutes)
   DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/ -q --timeout=600
   
   # Execution tests (needs DERIVA_HOST, ~10-20 minutes)  
   DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/ -q --timeout=600
   
   # Remaining integration tests
   DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/catalog/ tests/feature/ tests/schema/ tests/core/ -q --timeout=600
   ```
2. **Use the Monitor tool** for streaming test output on long runs, or **Bash with `run_in_background=true`** and check progress via Read on the output file.
3. **Never use a bare `uv run pytest tests/`** with a 2-minute timeout — it will be killed mid-run. Always set `timeout` on the Bash call (up to 600000ms / 10 minutes) or break into subsets.
4. **Per-test timeout**: Install `pytest-timeout` and use `--timeout=300` to prevent individual tests from hanging.

## Build and Development Commands

> **CWD:** every command in this section assumes you are in
> `/Users/carl/GitHub/DerivaML/deriva-ml`. The Bash tool's cwd is **not**
> reliably persistent across turns — always chain `cd` into a single
> call, e.g. `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest`.
> See the workspace-level `CLAUDE.md` ("CWD discipline") for the rule.

```bash
# Install dependencies
uv sync

# Run all tests
DERIVA_ML_ALLOW_DIRTY=true uv run pytest

# Run a single test file
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_datasets.py

# Run a specific test
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_datasets.py::test_function_name -v

# Run tests with coverage
DERIVA_ML_ALLOW_DIRTY=true uv run pytest --cov=deriva_ml --cov-report=term-missing

# Lint and format
uv run ruff check src/
uv run ruff format src/

# Build documentation
uv run mkdocs serve
```

## Architecture

### Core Classes

**DerivaML** (`src/deriva_ml/core/base.py`): Main entry point for catalog operations. Uses mixin composition:
- `VocabularyMixin`, `FeatureMixin`, `DatasetMixin`, `AssetMixin`, `ExecutionMixin`
- `WorkflowMixin`, `FileMixin`, `AnnotationMixin`, `RidResolutionMixin`, `PathBuilderMixin`

Each mixin lives in `src/deriva_ml/core/mixins/` and handles one concern.

**Execution** (`src/deriva_ml/execution/execution.py`): Manages ML workflow lifecycle:
- Downloads/materializes datasets specified in configuration
- Tracks execution status and provenance
- Handles asset upload after execution completes
- Used as context manager: `with ml.create_execution(config) as exe:`

**Dataset** (`src/deriva_ml/dataset/dataset.py`): Versioned dataset management:
- Semantic versioning (major.minor.patch)
- BDBag export with optional MINID creation
- Nested dataset support
- Version history tracking via catalog snapshots

**DatasetBag** (`src/deriva_ml/dataset/dataset_bag.py`): Downloaded dataset representation:
- Provides same interface as Dataset via `DatasetLike` protocol
- Works with local BDBag directories (no catalog connection needed)
- Supports nested dataset traversal and member listing
- Use `as_torch_dataset()` / `as_tf_dataset()` to feed the bag to PyTorch / TensorFlow / Keras training loops
- Use `restructure_assets()` to reorganize files by dataset type/features for `ImageFolder`-style third-party trainers
- All three share the same `targets` / `target_transform` / `missing` vocabulary; lazy import of torch/tensorflow inside each adapter so the base library stays importable without either framework

**ExecutionConfiguration** (`src/deriva_ml/execution/execution_configuration.py`): Pydantic model for execution setup:
- Dataset specifications with version and materialization options
- Input asset RIDs
- Workflow reference
- Execution parameters

### Key Patterns

**Catalog Path Builder**: Most catalog queries use the fluent path builder API:
```python
pb = ml.pathBuilder()
results = pb.schemas[schema_name].tables[table_name].entities().fetch()
```

**Dataset Versioning**: Datasets use catalog snapshots for version isolation:
- Each version records a catalog snapshot timestamp
- `dataset.set_version(version)` returns a Dataset bound to that snapshot
- Version increments propagate to parent/child datasets via topological sort

**Asset Management**: Assets are tracked via association tables:
- `Asset_Type` vocabulary controls asset categorization
- `{Asset}_Execution` tables link assets to executions with Input/Output roles
- File uploads use Hatrac object store

**Asset_Role contract**: Every execution-linked asset carries an
`Asset_Role` of `Input` or `Output` on its `{Asset}_Execution` row
**and** a directional `Input_File` / `Output_File` Asset_Type tag.
deriva-ml assigns both — never the caller. Input role goes on
assets materialized at execution start; Output role goes on
assets uploaded via `bag_commit`. Pinned by
`tests/execution/test_asset_role_contract.py`; user-facing
description in `docs/user-guide/executions.md` ("How
execution-asset roles work").

**Verb naming (`find_*` vs `list_*`)**: deriva-ml's public-API methods
follow a predictable verb-then-noun convention. Knowing the rule
means you don't have to guess which method to reach for:

- `find_*` — schema-introspection / discovery that walks the
  catalog model with filtering or traversal logic.
  Examples: `find_features`, `find_datasets`, `find_workflows`,
  `find_executions`, `find_assets`, `find_associations`. These do
  non-trivial work to identify matching entities (predicates,
  association detection, FK traversal).
- `list_*` — straightforward enumeration of "what's there" inside
  a known scope.
  Examples: `list_assets`, `list_executions`,
  `list_dataset_members`, `list_dataset_children`,
  `list_dataset_parents`, `list_vocabulary_terms`. These are thin
  wrappers over a table read.
- `get_*` — single-RID detail read; takes a RID, returns the
  bundled record.
- `lookup_*` — RID-or-name resolution helper (e.g.
  `lookup_dataset`); a sibling of `get_*` that accepts either form.
- Mutating verbs (`create_*` / `update_*` / `delete_*` /
  `add_*`) — write operations.

If you reach for `ml.list_features` you will get an `AttributeError`;
the correct call is `ml.find_features(table)`. Features require an
association walk to identify, so they live under `find_*` not
`list_*`.

This convention applies to the Python library. The MCP wire
surface uses a slightly different `find_*` rule (catalog-wide
search by non-RID identifier — see the
`deriva_ml_getting_started` prompt for the MCP-side definition).

**RIDs are opaque: equality only.** A RID's only valid operation
is equality comparison. The format (`<catalog-prefix>-<encoded-counter>`,
e.g. `4CY`, `BR0`) is a server implementation detail. In particular:

- **Never hard-code a RID as a literal** in tests, fixtures,
  scripts, configs, or docstring examples. Obtain RIDs from a
  fresh catalog lookup or fixture call — never from a string
  written by a human. The failure mode is silent: a literal like
  `"1-IMG1"` round-trips through a test that mocks the layer
  where the RID matters, the test passes, and production breaks
  the first time a real RID flows through the same code path
  (see the 2026-05-19 torch-adapter bag-layout bug for an
  illustrative case — every adapter test stubbed the path
  argument so the wrong on-disk layout went undetected for
  three weeks).
- **Never parse, slice, regex, or `startswith`** on a RID. The
  format is not stable contract.
- **RID ordering is server-internal mechanics, not semantics.**
  The server sorts rows by RID to provide stable iteration order
  for cursor pagination — `after_rid=X` returns rows with
  `RID > X`, walking the table in the server's RID collation.
  That ordering is what makes pagination work; **it carries no
  semantic meaning**. Don't sort by RID client-side to surface
  "newest" or "related" rows; use `RCT` for newest-first ordering
  and follow the relevant FK / association table for relatedness.
- **Never compare RIDs across catalogs.** RID `4CY` in catalog 46
  has nothing to do with RID `4CY` in catalog 42. Cross-catalog
  identity is `(host, catalog_id, RID)`.
- **Never infer column values from a RID across snaptimes.** The
  same RID at two snaptimes refers to the same row, but its
  column values can differ — RID identity is constant; row data
  is not.

For tests specifically: every RID a test touches must come from
a fixture-produced catalog row (e.g. via `CatalogManager`
populated datasets, or a fresh `create_*` call inside the test).
The test asserts shape and equality against values it pulled
from the catalog — never against values it wrote into a string
literal.

### Testing

Tests require a running Deriva catalog. Set `DERIVA_HOST` environment variable to specify the test server (defaults to `localhost`).

**Core test infrastructure** (`tests/catalog_manager.py`):
- `CatalogManager`: Session-scoped owner of the test catalog. Tracks state via `CatalogState` (EMPTY → POPULATED → WITH_FEATURES → WITH_DATASETS, monotonic) and provides `ensure_populated()`, `ensure_features()`, `ensure_datasets()`, `reset()`, and `get_ml_instance()`.

**Key fixtures** (`tests/conftest.py`):
- `catalog_manager`: Session-scoped `CatalogManager` instance (primary entry point)
- `test_ml`: Function-scoped DerivaML instance, resets catalog between tests
- `catalog_with_datasets`: Provides a catalog with populated demo data (datasets, splits)
- `deriva_catalog`: Legacy session-scoped fixture (uses `catalog_manager` internally)

**Gotcha**: All three `CatalogManager.ensure_*` methods (`ensure_populated`, `ensure_features`, `ensure_datasets`) validate data actually exists before trusting their state flag. Function-scoped fixtures may empty tables during teardown without resetting the manager's state, leaving the flag stale; the existence guards catch that and re-populate the necessary level.

### Docstring Examples (Doctest)

`Example:` blocks in public-method docstrings are run as doctests during
the main pytest collection (via `--doctest-modules`).

- **Catalog-dependent examples** must carry `# doctest: +SKIP` on the
  first interactive line. Doctest collection happens without a live
  catalog.
- **Pure-Python examples** (selector factories, type coercion, config
  construction from literals) run for real and catch regressions.
- Common symbols (e.g., `FeatureRecord`) are available in the doctest
  namespace via `src/deriva_ml/conftest.py`.

Example annotation pattern:

```python
    Example:
        >>> from deriva_ml import DerivaML  # doctest: +SKIP
        >>> ml = DerivaML(hostname="example.org", catalog_id="42")  # doctest: +SKIP
        >>> datasets = ml.find_datasets()  # doctest: +SKIP

        >>> # Pure-Python example — runs for real:
        >>> from deriva_ml.feature import FeatureRecord
        >>> selector = FeatureRecord.select_newest
        >>> callable(selector)
        True
```

## Schema Structure

The library uses two schemas:
- **deriva-ml** (`ML_SCHEMA`): Core ML tables (Dataset, Execution, Workflow, Feature_Name, etc.)
- **Domain schema**: Application-specific tables created by users

Controlled vocabularies: Dataset_Type, Asset_Type, Workflow_Type, Asset_Role, Feature_Name

## Exception Hierarchy

DerivaML uses a structured exception hierarchy for error handling:

```
DerivaMLException (base class)
├── DerivaMLConfigurationError (configuration/initialization)
│   ├── DerivaMLSchemaError (schema structure issues)
│   └── DerivaMLAuthenticationError (auth failures)
├── DerivaMLDataError (data access/validation)
│   ├── DerivaMLNotFoundError (entity not found)
│   │   ├── DerivaMLDatasetNotFound
│   │   ├── DerivaMLTableNotFound
│   │   └── DerivaMLInvalidTerm
│   ├── DerivaMLTableTypeError (wrong table type)
│   ├── DerivaMLValidationError (validation failures)
│   └── DerivaMLCycleError (relationship cycles)
├── DerivaMLExecutionError (execution lifecycle)
│   ├── DerivaMLWorkflowError
│   └── DerivaMLUploadError
└── DerivaMLReadOnlyError (writes on read-only)
```

Import from: `from deriva_ml.core.exceptions import ...`

## Protocol Hierarchy

The library uses protocols for type-safe polymorphism:

**Dataset Protocols:**
- `DatasetLike`: Read-only operations (Dataset and DatasetBag)
- `WritableDataset`: Write operations (Dataset only)

**Catalog Protocols:**
- `DerivaMLCatalogReader`: Read-only catalog operations
- `DerivaMLCatalog`: Full catalog operations with writes

Import from: `from deriva_ml.interfaces import ...`

## Shared Utilities

**Validation** (`deriva_ml.core.validation`):
- `VALIDATION_CONFIG`: Standard ConfigDict for `@validate_call`
- `STRICT_VALIDATION_CONFIG`: ConfigDict that forbids extra fields

**Logging** (`deriva_ml.core.logging_config`):
- `get_logger(name)`: Get a deriva_ml logger
- `configure_logging(level)`: Configure logging for all components

## Catalog Cloning

**`clone_via_bag`** (`src/deriva_ml/catalog/clone_via_bag.py`): The current catalog-clone path. Builds a BDBag from the source catalog (using the export pipeline + an explicit traversal policy), then loads the bag into a schema-ready destination catalog via deriva-py's `BagCatalogLoader`. The legacy three-stage approach (schema without FKs → async data copy → FK application) was retired in v1.36.0 in favor of this single-call pipeline.

Key parameters:
- `root_rid` — anchor for the bag's content slice. Translated to a `RIDAnchor` and merged with the policy's existing anchors.
- `policy` — `FKTraversalPolicy` controlling traversal scope and per-table behaviour. The caller's policy is merged with deriva-ml defaults (vocab export FULL, terminal tables {Execution, Workflow}, dangling FK strategy DELETE) — caller-provided non-default values win.
- `asset_mode` — `AssetMode.UPLOAD` (default) copies asset bytes into the destination Hatrac; `AssetMode.ROWS_ONLY` skips asset upload (rows are still inserted with their source-Hatrac URLs); `AssetMode.REFERENCES` preserves source URLs for later localization.
- `dangling_fk_strategy` — `FAIL` (refuse), `DELETE` (drop the row), or `NULLIFY` (set the FK column NULL).

**`create_ml_workspace`** (`src/deriva_ml/catalog/clone.py`): Legacy wrapper preserved for callers that haven't migrated. Routes to `clone_via_bag` under the hood; several historical parameters (e.g., `reinitialize_dataset_versions`, `prune_hidden_fkeys`, `truncate_oversized`, `table_concurrency`, `copy_annotations`) are accepted-but-ignored and logged as deprecation warnings. New code should call `clone_via_bag` directly.

**`localize_assets`** (`src/deriva_ml/catalog/localize.py`): Phase-2 leg of the split-phase clone. After a `clone_via_bag(asset_mode=REFERENCES)` (or `ROWS_ONLY`) leaves the destination with source-Hatrac URLs, `localize_assets` copies the bytes server-to-server (download from source, upload to dest) and rewrites the catalog rows to point at the destination Hatrac. Records phase-2 stats on the catalog provenance annotation.

### create_ml_schema CASCADE Warning

`create_ml_schema()` in `src/deriva_ml/schema/create_schema.py` **drops the existing `deriva-ml` schema with CASCADE** if it already exists, destroying all data. The clone code guards against this — never call `create_ml_schema` on a catalog that already has data in `deriva-ml`.

## Hydra-zen Configuration

DerivaML integrates with hydra-zen for reproducible configuration. Key config classes:

**DerivaMLConfig** (`deriva_ml.core.config`): Main connection configuration
```python
from deriva_ml import DerivaMLConfig
config = DerivaMLConfig(hostname="example.org", catalog_id="42")
ml = DerivaML.instantiate(config)
```

**DatasetSpecConfig** (`deriva_ml.dataset`): Dataset specification for executions
```python
from deriva_ml.dataset import DatasetSpecConfig
spec = DatasetSpecConfig(rid="XXXX", version="1.0.0", materialize=True)
```

**AssetSpec / AssetSpecConfig** (`deriva_ml.execution`): Input asset specification. Use
`AssetSpec` in regular Python code and `AssetSpecConfig` (the hydra-zen interface) inside
hydra-zen stores. Bare RID strings are accepted where a list of assets is expected.
```python
from deriva_ml.execution import AssetSpec, AssetSpecConfig
asset = AssetSpec(rid="YYYY", cache=True)
cfg = AssetSpecConfig(rid="YYYY", cache=True)
```

**ExecutionConfiguration** (`deriva_ml.execution`): Full execution setup
```python
from deriva_ml.execution import ExecutionConfiguration
config = ExecutionConfiguration(
    datasets=[DatasetSpecConfig(rid="DATA", version="1.0.0")],
    assets=["WGTS"],
    description="Training run"
)
```

Use `builds()` with `populate_full_signature=True` for hydra-zen integration.
Use `zen_partial=True` for model functions that receive execution context at runtime.

See `docs/configuration/overview.md` for complete documentation.

## User Preferences

### Opening markdown files for review

When the user asks to open a spec, plan, or other markdown file for review, use MarkEdit:
```bash
open -a MarkEdit /absolute/path/to/file.md
```
This applies to spec/plan reviews during the brainstorming → spec → plan cycle as well as ad-hoc markdown review requests.

### Input-required notifications

When Claude needs user input to proceed, prefix the question with `⏸ NEED INPUT:` on its own line so it stands out in long sessions. One question at a time is preferred over batched lists.

### Class idiom choice — Pydantic vs `@dataclass`

Use **Pydantic `BaseModel`** when ANY of these apply — the goal is a single serialization/validation story for anything user-facing:

1. **Users construct instances directly** (config, specs). They expect field validation.
2. **Users assign to mutable fields** (`record.status = ...`). Validation should fire at the assignment site, not at next sync.
3. **Public method parameters**. Use `@validate_call` (or `@pydantic.validate_call`) so bad args fail with a clear message at the call boundary.
4. **The class may be serialized or cross a boundary** (JSON I/O, logs, cache, API, bag metadata). Users should reach for one API (`.model_dump()`) rather than juggling `dataclasses.asdict()` depending on type.

Use **`@dataclass`** only when NONE of those apply — purely internal value objects with no user-facing surface. Examples in this codebase: `catalog/clone.py` internal orchestration records, `tools/validate_schema_doc.py` parse results.

When in doubt, pick Pydantic — "too many interfaces" is the failure mode to avoid. A user-facing return type that's `@dataclass` forces every serialization caller to know its exact type.

## Best Practices & Patterns

### Pull-Request Workflow (required)

**All changes to `deriva-ml` go through a pull request — including
spawned-agent work.** Do not commit directly to `main`. The standard
sequence:

```bash
git checkout -b feature/<short-name>
# ... commit work on the branch ...
git push -u origin feature/<short-name>
gh pr create --title "..." --body "..."
# wait for review or self-merge with:
gh pr merge --squash --delete-branch
```

The PR is part of the deliverable; "shipped without a PR" is
incomplete work. This applies even for solo developers and even for
spawned subagents — the PR creates the durable review record that
`git log` alone doesn't.

The `bump-version` step happens AFTER merge, on `main`, in a clean
working tree (so the bump-commit lands on `main` directly and the
tag points at it). Never bump from a feature branch.

### Cross-repo bug fixes (deriva-py ↔ deriva-ml)

When a deriva-ml bug traces back to a deriva-py contract violation,
fix the bug **and** pin the contract at the layer that promises it:

- **The contract test belongs upstream**, in deriva-py's test suite,
  alongside the implementation that makes the promise. A future
  refactor that breaks the contract must fail deriva-py's own CI
  before a release ships — not deriva-ml's CI after the pin advances.
- **The contract fix lands in the same upstream PR as the contract
  test.** Don't defer the test to "we'll add it later" — the
  fix-without-test PR creates a regression window in deriva-py's main
  branch.
- **A reproduction harness in deriva-ml is OK, but it's not the pin.**
  An `xfail(strict=True)` test in this repo is fine as a reproduction
  while the upstream fix is scoped — it proves the bug is reachable
  from a real user flow and gives a visible "lockstep ratchet" when
  the pin advances. Once upstream ships with its own test, **delete
  the downstream xfail**; do not "unxfail" it. The contract test now
  lives upstream; whatever downstream tests remain are
  consumer-integration coverage and should be framed that way.

Reference case: bag-cache multi-anchor erasure (issue #142). Fix
landed in deriva-py PR #254 with upstream tests
(`test_cache_index_record_accumulates_anchors`,
`test_cache_index_record_dedupes_repeated_anchor`); deriva-ml PR #146
bumped the pin and the downstream coverage at
`tests/dataset/test_multi_anchor_bag_cache.py` reframed as
integration. See that file's module docstring for the consumer-side
framing.

### Version Bumping

Use the `bump-version` script for releases - it handles the complete workflow:
```bash
uv run bump-version patch  # or minor, major
```
This fetches tags, bumps the version, creates a tag, and pushes everything in one command.
Don't use `bump-my-version` directly as it doesn't push changes.

Run `bump-version` only on `main`, after the feature PR has merged.

### Asset Upload

Use `asset_file_path()` API to register files for upload:
```python
path = execution.asset_file_path(
    MLAsset.execution_metadata,
    "my-file.yaml",
    asset_types=ExecMetadataType.hydra_config.value,
)
with path.open("w") as f:
    f.write(content)
```
Don't manually create files in `working_dir / "Execution_Metadata"` - they won't be uploaded.

### Asset Manifest Architecture

Assets use a manifest-first design (`asset-manifest.json`):
- Files stored in flat `assets/{Table}/` directories, metadata in the manifest JSON
- Write-through + fsync on every mutation for crash safety
- Upload builds ephemeral symlinks into the regex-expected tree at upload time
- Metadata directory order must match `sorted(metadata_columns)` — both `asset_table_upload_spec()` and `_build_upload_staging()` sort alphabetically
- Per-asset status tracking enables upload resume after crashes

### Upload Network Configuration

`upload_directory()` has two network configuration parameters:
- `timeout`: HTTP session timeout (connect, read) - passed to session config
- `chunk_size`: Hatrac chunk upload size in bytes - passed through upload spec

### Workflow Deduplication

Workflows are deduplicated by checksum. When the same script runs multiple times, `add_workflow()` returns the existing workflow's RID rather than creating a new one. Tests that need distinct workflows must account for this.

### Testing find_experiments

The `find_experiments()` function finds executions with Hydra config files (matching `*-config.yaml` in Execution_Metadata). Test fixtures must use `asset_file_path()` to properly register config files - see `execution_with_hydra_config` fixture.

### Association Tables

Use `Table.define_association()` for creating association tables instead of manually defining columns, keys, and foreign keys:
```python
Table.define_association(
    associates=[("Execution", execution), ("Nested_Execution", execution)],
    comment="Description",
    metadata=[Column.define("Sequence", builtin_types.int4, nullok=True)]
)
```

### API Priority

When writing new code, prefer APIs in this order:
1. **deriva-ml routines** (highest level, most convenient)
2. **deriva-py routines** with preference for **datapath** API over raw ERMrest URLs
3. **BDBag/fair-data routines** (for bag creation and materialization)

For catalog queries, prefer `dp.attributes(dp.RID)` and `dp.aggregates(Cnt(dp.RID))` over
constructing raw ERMrest URL strings.

### Bags Should Behave Like Catalog Connections

DatasetBag should provide the same interface as live catalog operations wherever possible.
Users should not need to learn a separate API for working with downloaded data vs live data.
This means: same method names, same record types (FeatureRecord everywhere, not a separate
FeatureValueRecord), same selector signatures, same return types. When a bag can't support
a feature (e.g., workflow-based selection requires catalog access), it should raise a clear
error, not silently provide a different type.

### BDBag Remote File Manifest

When creating bags with remote asset references, **never write fetch.txt directly**. Instead,
write a remote file manifest JSON file and pass it to `bdb.make_bag(remote_file_manifest=...)`.
This is because `make_bag(update=True)` destroys any hand-written fetch.txt by regenerating it
from its internal `remote_entries` dict (which is empty if no manifest was provided).

The manifest format is one JSON object per line:
```json
{"url": "https://...", "length": 12345, "filename": "asset/RID/Image/file.txt", "md5": "abc123"}
```
Note: `filename` omits the `data/` prefix — the bdbag API prepends it automatically.
A hash value (md5 or sha256) is required for each entry.

See `deriva_download.py` in deriva-py for the reference implementation.

### Dataset bag-builder traversal

`src/deriva_ml/dataset/bag_builder.py` drives dataset export and size
estimation via `deriva.bag.catalog_builder.CatalogBagBuilder`, per ADR-0006.
`CatalogBagBuilder` walks FK paths from the Dataset RID to discover all
reachable tables; deriva-ml supplies a `_SnapshotAwareCatalogBagBuilder`
subclass that pins to a catalog snapshot and applies the per-table
traversal policy.

`estimate_bag_size` uses RID-union semantics: it fetches RID lists from
all FK paths to each table and computes `set.union()` for exact counts.
This handles the case where the same table is reachable via multiple
FK paths with overlapping or disjoint rows.

The previous `CatalogGraph` (`src/deriva_ml/dataset/catalog_graph.py`)
that owned this logic was retired in the 2026-05 bag-client cutover —
see `docs/archive/2026-05-bag-cutover/dataset-bag-cutover-2026-05.md`
for the design that drove the replacement.

### Dataset Download Flow

`download_dataset_bag` → `get_dataset_minid` → (free-function pipeline
in `dataset/bag_download.py`):
1. Creates a bag directory structure
2. Processes query_processors from the export spec (env, json, csv, fetch)
3. Writes remote file manifest for assets (passed to `make_bag(remote_file_manifest=...)`)
4. Archives the bag as a zip
5. `materialize_dataset_bag` then calls `bdb.materialize()` to fetch remote assets

(`get_dataset_minid` and `materialize_dataset_bag` are the
free-function successors to the Dataset-method versions
`_get_dataset_minid` / `_materialize_dataset_bag`, extracted in the
2026-05 audit cycle.)

### restructure_assets Return Type

`DatasetBag.restructure_assets()` returns `dict[Path, Path]` (manifest mapping source → dest),
not a `Path`. Tests should assert `isinstance(result, dict)`.

### ExecutionConfiguration Assets

`ExecutionConfiguration.assets` accepts plain RID strings but coerces them to `AssetSpec` objects
via a Pydantic model validator. Tests comparing assets should use `[a.rid for a in config.assets]`
rather than comparing directly to string lists.

### `model/annotations.py` is a public API for deriva-skills

`src/deriva_ml/model/annotations.py` (and its re-exports through
`src/deriva_ml/model/__init__.py`) is the documented public API for the
`deriva-skills/use-annotation-builders` Claude Code skill. Every builder
class — `Display`, `VisibleColumns`, `VisibleForeignKeys`,
`TableDisplay`, `ColumnDisplay`, `PseudoColumn`, `FacetList`, plus
supporting enums/records/context constants — is consumed externally and
must preserve its `.tag` class attribute and `.to_dict()` method.
`DerivaML.apply_annotations()` is the documented apply entry point and
must keep its zero-required-arg signature.

A workspace-wide grep for these names returns zero hits inside
`src/deriva_ml/` itself; this is expected and not a deletion signal.
See `docs/adr/0007-annotation-builders-public-api.md` for the full
contract and `tests/model/test_annotations.py::TestExternalConsumerContract`
for the regression coverage that pins the surface in CI.

### Reference manual for traversal / export / denormalization

The exact behavior of FK traversal, bag export, and denormalization is
documented formally in `docs/reference/`:

- `docs/reference/fk-traversal.md` — `FKTraversalPolicy` fields
  (walk-phase vs load-phase) + the walker rules (bidirectional walk,
  terminal tables, vocab leaves, `max_depth`).
- `docs/reference/bag-export.md` — how `DatasetBagBuilder` drives the
  engine: nested-descendant anchors, empty-association pruning, the
  `{Execution, Workflow}` terminal-tables guard, and the export-spec
  shape.
- `docs/reference/denormalization.md` — the wide-table
  (`get_denormalized_*`) rules (`row_per` / `via` / selector / return
  types).

Each rule is tagged `[engine: deriva-py]` (enforced upstream in
`deriva/bag/`) or `[deriva-ml]`, and every worked example is verified
against a demo catalog. These docs are RAG-indexed as `ml-docs`
(searchable via `rag_search(doc_type="ml-docs")`) once on `main` —
consult them when answering questions about what a bag includes or how
denormalization shapes a frame.