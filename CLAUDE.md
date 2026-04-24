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

**Dirty workflow check**: DerivaML checks for uncommitted git changes before running workflows. For tests, set:
```bash
export DERIVA_ML_ALLOW_DIRTY=true
```

**Test catalog**: Tests require a running Deriva catalog server. Set `DERIVA_HOST` to specify the server (defaults to `localhost`). Tests take 30-90 minutes due to real catalog operations.

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
- Use `restructure_assets()` to reorganize files by dataset type/features

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

### Testing

Tests require a running Deriva catalog. Set `DERIVA_HOST` environment variable to specify the test server (defaults to `localhost`).

**Core test infrastructure** (`tests/catalog_manager.py`):
- `CatalogManager`: Session-scoped owner of the test catalog. Tracks state (EMPTY/POPULATED/DATASETS) and provides `ensure_populated()`, `ensure_datasets()`, `reset()`, and `get_ml_instance()`.

**Key fixtures** (`tests/conftest.py`):
- `catalog_manager`: Session-scoped `CatalogManager` instance (primary entry point)
- `test_ml`: Function-scoped DerivaML instance, resets catalog between tests
- `catalog_with_datasets`: Provides a catalog with populated demo data (datasets, splits)
- `deriva_catalog`: Legacy session-scoped fixture (uses `catalog_manager` internally)

**Gotcha**: `CatalogManager.ensure_populated()` validates data actually exists before trusting its state flag. Other test modules (function-scoped fixtures) may empty tables during teardown, making the state flag stale.

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
- `LoggerMixin`: Mixin providing `_logger` attribute

## Catalog Cloning

**`create_ml_workspace`** (`src/deriva_ml/catalog/clone.py`): Creates partial catalog clones:
- Three-stage approach: schema without FKs → async data copy → FK application with orphan handling
- Export annotation parsing for guided table discovery
- Orphan strategies: FAIL, DELETE, NULLIFY for incoherent row-level policies
- Dataset version reinitialization with proper catalog snapshots

**`localize_assets`** (`src/deriva_ml/catalog/localize.py`): Copies assets from source to local Hatrac after cloning with `asset_mode=REFERENCES`.

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

### Version Bumping

Use the `bump-version` script for releases - it handles the complete workflow:
```bash
uv run bump-version patch  # or minor, major
```
This fetches tags, bumps the version, creates a tag, and pushes everything in one command.
Don't use `bump-my-version` directly as it doesn't push changes.

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

### CatalogGraph and Dataset Export

`CatalogGraph` (`src/deriva_ml/dataset/catalog_graph.py`) traverses FK paths from the Dataset
table to discover all reachable tables. It builds:
- **Export specs** (`generate_dataset_download_spec`): Query processor definitions for bag export
- **Aggregate queries** (`_aggregate_queries`): Datapath objects for `estimate_bag_size`
- **Table paths** (`_table_paths`): ERMrest paths for export

`estimate_bag_size` uses RID-union semantics: it fetches RID lists from all FK paths to each
table and computes `set.union()` for exact counts. This handles the case where the same table
is reachable via multiple FK paths with overlapping or disjoint rows.

### Dataset Download Flow

`download_dataset_bag` → `_get_dataset_minid` → `_create_dataset_bag_client` (or MINID path):
1. Creates a bag directory structure
2. Processes query_processors from the export spec (env, json, csv, fetch)
3. Writes remote file manifest for assets (passed to `make_bag(remote_file_manifest=...)`)
4. Archives the bag as a zip
5. `_materialize_dataset_bag` then calls `bdb.materialize()` to fetch remote assets

### restructure_assets Return Type

`DatasetBag.restructure_assets()` returns `dict[Path, Path]` (manifest mapping source → dest),
not a `Path`. Tests should assert `isinstance(result, dict)`.

### ExecutionConfiguration Assets

`ExecutionConfiguration.assets` accepts plain RID strings but coerces them to `AssetSpec` objects
via a Pydantic model validator. Tests comparing assets should use `[a.rid for a in config.assets]`
rather than comparing directly to string lists.