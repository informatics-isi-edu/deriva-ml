# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DerivaML is a Python library for reproducible machine-learning workflows on a Deriva catalog.

## Environment Setup

**Dirty workflow check & test catalog**: DerivaML refuses to run
workflows with uncommitted changes; tests set
`DERIVA_ML_ALLOW_DIRTY=true` to bypass. Tests also need a live
Deriva catalog at `DERIVA_HOST` (defaults to `localhost`). See
the workspace-level [`CLAUDE.md`](../CLAUDE.md) "Dirty-tree
override for testing" for the rule rationale; full suite is
30–90 minutes.

**Running long test suites**: The full suite takes 30–90 minutes, longer than one foreground command allows. Run it in the subsets below, and run the integration subsets in the background rather than as one bare `uv run pytest tests/`:
1. **Run test subsets incrementally** rather than the full suite at once:
   ```bash
   # Unit tests (fast, no catalog needed, ~3 seconds)
   DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q
   
   # Dataset integration tests (needs DERIVA_HOST, ~10-30 minutes)
   DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/ -q
   
   # Execution tests (needs DERIVA_HOST, ~10-20 minutes)  
   DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/ -q
   
   # Remaining integration tests
   DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/catalog/ tests/feature/ tests/schema/ tests/core/ -q
   ```
2. **Per-test timeout**: `pytest-timeout` is not a dev dependency. To stop a hanging test, add it with `uv add --dev pytest-timeout`, then pass `--timeout=300`.

## Build and Development Commands

> **CWD:** every command in this section assumes you are in
> `/Users/carl/GitHub/DerivaML/deriva-ml`. The Bash tool's cwd is **not**
> reliably persistent across turns — always chain `cd` into a single
> call, e.g. `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run pytest`.
> See the workspace-level `CLAUDE.md` ("CWD discipline") for the rule.

Standard `uv sync` / `uv run pytest` / `uv run ruff` invocations; prefix
test runs with `DERIVA_ML_ALLOW_DIRTY=true` (see Environment Setup).
Docs: `uv run mkdocs serve`.

## Architecture

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
  `find_executions`, `find_assets`. These do
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
  the first time a real RID flows through the same code path.
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

## Catalog Cloning

**`clone_via_bag`** (`src/deriva_ml/catalog/clone_via_bag.py`): The current catalog-clone path. Builds a BDBag from the source catalog (using the export pipeline + an explicit traversal policy), then loads the bag into a schema-ready destination catalog via deriva-py's `BagCatalogLoader`.

New code calls `clone_via_bag` directly; `create_ml_workspace` is a legacy
wrapper around it, and `localize_assets` is the phase-2 byte copy after a
`REFERENCES`/`ROWS_ONLY` clone.

### create_ml_schema CASCADE Warning

`create_ml_schema()` in `src/deriva_ml/schema/create_schema.py` **drops the existing `deriva-ml` schema with CASCADE** if it already exists, destroying all data. The clone code guards against this — never call `create_ml_schema` on a catalog that already has data in `deriva-ml`.

## Hydra-zen Configuration

Full documentation: `docs/configuration/overview.md`. Use `builds()` with
`populate_full_signature=True`, and `zen_partial=True` for model functions
that receive execution context at runtime.

## Code Conventions

### Class idiom choice — Pydantic vs `@dataclass`

Use **Pydantic `BaseModel`** when ANY of these apply — the goal is a single serialization/validation story for anything user-facing:

1. **Users construct instances directly** (config, specs). They expect field validation.
2. **Users assign to mutable fields** (`record.status = ...`). Validation should fire at the assignment site, not at next sync.
3. **Public method parameters**. Use `@validate_call` (or `@pydantic.validate_call`) so bad args fail with a clear message at the call boundary.
4. **The class may be serialized or cross a boundary** (JSON I/O, logs, cache, API, bag metadata). Users should reach for one API (`.model_dump()`) rather than juggling `dataclasses.asdict()` depending on type.

Use **`@dataclass`** only when NONE of those apply — purely internal value objects with no user-facing surface. Example in this codebase: `tools/validate_schema_doc.py` parse results.

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

Example: `tests/dataset/test_multi_anchor_bag_cache.py` — its module
docstring shows how the downstream tests read once the contract test
lives upstream.

### Version Bumping

Use the `bump-version` script for releases - it handles the complete workflow:
```bash
uv run bump-version patch  # or minor, major
```
This fetches tags, bumps the version, creates a tag, and pushes everything in one command.
Don't use `bump-my-version` directly as it doesn't push changes.

Run `bump-version` only on `main`, after the feature PR has merged.

Verify the deriva-py pin is current before bumping (see "Updating the
deriva-py pin" below). A release with a stale pin freezes every
downstream consumer on an old deriva-py, which can bring back deriva-py
bugs that are already fixed.

### Updating the deriva-py pin

deriva-py is pinned to an **immutable commit SHA**, not the
`deriva-ml` branch, in **two** places that must stay in lockstep:

- `project.dependencies` in `pyproject.toml`:
  `"deriva @ git+...deriva-py@<sha>"`
- `[tool.uv.sources]` in `pyproject.toml`:
  `deriva = { git = "...deriva-py", rev = "<sha>" }`

**Why a SHA and not the branch.** A git-*branch* dependency carries no
"newer-than" version semantics, and `[tool.uv.sources]` is **not
transitive** (uv only reads it from the workspace-root project). So a
downstream project (e.g. eye-ai-vgg19) that bumps `deriva-ml` will
**not** advance deriva-py — its own lock stays frozen on whatever
deriva-py commit it last resolved, even though deriva-ml moved. A
**commit SHA in `project.dependencies` is transitive and immutable**:
uv records it in deriva-ml's published metadata, so bumping deriva-ml
*forces* every consumer's resolver to pull deriva-py at exactly that
commit. This is the mechanism that keeps the two repos in sync
automatically.

**To advance the pin** (when deriva-ml needs a newer deriva-py — e.g.
after merging a deriva-py PR on the `deriva-ml` branch):

1. Get the target commit (usually `origin/deriva-ml` HEAD after the
   deriva-py PR merged):
   ```bash
   git -C ../../deriva-py rev-parse origin/deriva-ml
   ```
2. Replace the SHA in **both** the `project.dependencies` URL and the
   `[tool.uv.sources]` `rev` (they must match).
3. `uv lock && uv sync` and run the suite to confirm the new deriva-py
   works.
4. Land via PR like any change, then `bump-version` on `main`.

**The pin advance is part of the bump, not separate.** When a release
depends on a deriva-py fix, advancing the SHA and bumping the version
go together — never bump with a stale pin.

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
- Metadata directory order must match `sorted(metadata_columns)` — `upload_layout.asset_table_upload_spec()` and `bag_commit._add_asset_rows_to_bag()` both read columns through `model.asset_metadata_sorted()`
- Per-asset status tracking enables upload resume after crashes

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