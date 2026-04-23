# Feature API Consistency — Design

**Status:** Draft
**Date:** 2026-04-22
**Scope:** S2 from the sqlite-execution-state roadmap — reframed from "add Metric/Param" to "audit + unify Feature API for consistency across online/offline containers and the new caching/denormalization layer."

## Problem

The feature-access API is divergent across the three container surfaces (`DerivaML`, `Dataset`, `DatasetBag`) and internally inconsistent. Symptoms:

- Duplicate methods: `fetch_table_features` and `list_feature_values` appear on both `DerivaML` and `DatasetBag` with partially overlapping behavior.
- Asymmetric surfaces: `Dataset` has `find_features` but no read method for feature values; `DatasetBag` and `DerivaML` have both but their return semantics differ in edge cases.
- `ml.select_by_workflow(records, workflow)` is an outlier — it takes records and a workflow but is invoked as an ml-level function rather than a selector that composes with the rest of the selector ecosystem.
- `ml.add_features` exists as a write-direct-to-catalog escape hatch that bypasses provenance, allowing feature values to be created without an execution.
- Bag-side feature reads do not flow through the new denormalization caching layer.

The result is three near-duplicate APIs the user has to learn, with subtle differences, and no single answer to the three canonical ML-developer questions:

1. What features exist on this table?
2. Give me feature values shaped per target RID for model input.
3. Let me explore feature values with their provenance.

## Goals

- **One read surface** across all three containers with identical signatures and return types for corresponding methods.
- **Three methods** aligned with the three user questions above.
- **One write path** that requires provenance (no escape hatches).
- **Offline parity** — bags serve feature reads from the denormalization cache so online and offline return matching data.
- **No deprecation shims** — retired APIs raise a clear error pointing at the replacement.

## Non-goals

- No new concept of Metric or Param. Metrics are execution assets; parameters live in Hydra config + execution assets.
- No new general wide-form method. `Denormalizer` already handles general structural wide-joins and is feature-agnostic; S2 leaves it unchanged.
- No changes to `create_feature`, `FeatureRecord` dynamic class generation, or the vocabulary subsystem.

## Architecture

### Three-method read surface, symmetric across containers

Each of `DerivaML`, `Dataset`, `DatasetBag` exposes:

| Method | Returns | Purpose |
|---|---|---|
| `find_features(table)` | `list[Feature]` | Discovery — what features exist on the table |
| `feature_values(table, feature_name, selector=...)` | `Iterable[FeatureRecord]` | Wide-shaped typed records per target RID (selector-reduced) |
| `lookup_feature(table, feature_name)` | `Feature` | Metadata — feature definition, enables `feature_record_class()` |

`feature_values` returns an iterator of `FeatureRecord` instances. Each record is already wide in shape — it has the target RID, all value columns (vocab term, asset references, metadata columns), and provenance columns (Execution, RCT) as typed attributes. The iterator is single-feature; after selector reduction, one record per target RID is yielded. Callers convert to pandas/polars/numpy as they wish — `pd.DataFrame(r.model_dump() for r in records)` is the idiomatic pandas form.

Rationale for `Iterable[FeatureRecord]` over `pd.DataFrame`: pandas is a heavy optional dependency; iterator return is stream-friendly for large bags; typed attribute access catches typos at authoring time; `FeatureRecord` is already the currency of the write path so the read and write APIs share types.

Single-feature only. Multi-feature wide-form is `Denormalizer`'s job; duplicating it here adds complexity without simplifying any real use case. A user who wants two features merged on target RID writes two iterations and a `dict` merge (or uses `Denormalizer`).

### `Denormalizer` — unchanged

The existing `Denormalizer` class handles general structural wide-joins via FK traversal, feature-agnostic. Feature tables participate in it like any other table:

```python
d = Denormalizer(dataset)
df = d.as_dataframe(["Image", "Image_Glaucoma"])
```

S2 adds no feature-specific code to `Denormalizer`. Selector-style multi-value reduction stays on `feature_values`, where it is feature-aware.

### Container helper: `list_workflow_executions`

Each container exposes:

```python
container.list_workflow_executions(workflow) -> list[RID]
```

Returns the list of execution RIDs in the container's scope that ran the given workflow. Entries are unique (a given execution runs one workflow). Consumers that need O(1) membership testing (such as `FeatureRecord.select_by_workflow`) convert to `set` at the call site.

- `DerivaML`: all executions in the catalog for this workflow.
- `Dataset`: executions in scope of the dataset.
- `DatasetBag`: same as Dataset, read from bag SQLite.

Public because users may want it for custom selectors. Its primary consumer is the new workflow selector factory.

### Selector factory: `FeatureRecord.select_by_workflow`

`ml.select_by_workflow(records, workflow)` is retired. Replacement:

```python
selector = FeatureRecord.select_by_workflow(workflow, *, container)
```

- Classmethod factory on `FeatureRecord`.
- `container` is a **required kwarg** (no ambient state, no guessing).
- At construction, resolves the execution list once via `container.list_workflow_executions(workflow)` and converts to a `set` internally for O(1) membership tests.
- Returns a closure `(list[FeatureRecord]) -> FeatureRecord | None` matching the existing selector signature.
- When no record in the group matches the workflow, returns `None`; `feature_values` treats `None` as "absent for this target RID" and omits the target from its iterator.

Usage:

```python
selector = FeatureRecord.select_by_workflow(training_workflow, container=bag)
for rec in bag.feature_values("Image", "Glaucoma", selector=selector):
    ...
```

The selector composes naturally with `find_features`-discovered features and the rest of the selector ecosystem (`select_newest`, `select_first`, `select_latest`, `select_by_execution`, `select_majority_vote`). Unknown-workflow failures surface at factory-construction time, not at selection time.

### Write surface — execution-staged, provenance-required

**One write path: `exe.add_features(records)`.** `ml.add_features` is retired.

Feature writes integrate with the SQLite execution-state design (§8 of `2026-04-18-sqlite-execution-state-design.md`). `exe.add_features(records)`:

1. Auto-fills the `Execution` field on each record if unset.
2. Validates records share one feature definition.
3. Appends rows to a new `staged_features` table in the execution's SQLite with status `Pending`.

On successful execution completion, `_flush_staged_features()` runs as part of `upload_execution_outputs()`, **after** staged-asset upload (so any feature row referencing an asset has the asset's catalog-side row available). Flush:

1. Reads `Pending` rows, groups by feature table.
2. Batch-inserts each group via path builder.
3. Marks rows `Uploaded` on success, `Failed` with error populated on failure.
4. Raises `DerivaMLUploadError` summarizing any failed groups; other groups continue.

Crash-resume piggybacks on the §8 staged-asset mechanism — a resumed execution finds `Pending` rows and re-flushes. Durability boundary is the SQLite write-through in `add_features`.

Escape hatch retirement: `ml.add_features` is removed entirely. Admin/fixup scenarios that previously used it now create a short-lived execution with an appropriate workflow (e.g., `Manual_Correction`). This is three lines more than the bypass and produces a real audit trail — the motivating reason to retire the shortcut.

### Bag denormalization cache — feature read path

`DatasetBag.feature_values` reads from a per-feature denormalization cache in bag SQLite. The cache:

- Is populated on first access per feature.
- Is immutable after population (bags are immutable).
- Mirrors ermrest semantics so online and offline `feature_values` return the same records (modulo RCT/RMT floating precision).

No new tables on the live catalog — this is bag-local.

### `lookup_feature` works offline

`DatasetBag.lookup_feature(table, feature_name)` reads feature definition metadata from bag SQLite. The returned `Feature` object's `feature_record_class()` constructs a Pydantic class identical to the online form, enabling users to build `FeatureRecord` instances while offline. These records can be handed to a later `exe.add_features(records)` call when back online — the execution provides provenance at stage time; the records themselves are portable.

### Retired APIs

Each raises `DerivaMLException` with a message pointing at the replacement:

| Retired | Replacement |
|---|---|
| `ml.add_features(records)` | `exe.add_features(records)` within an execution context |
| `ml.fetch_table_features(...)` | `feature_values(table, name)` for single-feature; `Denormalizer` for multi-table |
| `bag.fetch_table_features(...)` | `feature_values(table, name)` for single-feature; `Denormalizer` for multi-table |
| `ml.list_feature_values(...)` | `feature_values(...)` — same signature, renamed |
| `bag.list_feature_values(...)` | `feature_values(...)` — same signature, renamed |
| `ml.select_by_workflow(records, workflow)` | `FeatureRecord.select_by_workflow(workflow, container=...)` as `selector=` |

No deprecation shims. Messages are specific about the substitution.

## Components

### New

- `src/deriva_ml/dataset/bag_feature_cache.py` — bag per-feature denormalization cache: populate-on-first-access, read path for `DatasetBag.feature_values` and `DatasetBag.lookup_feature`.

### Modified

- `core/mixins/feature.py`
  - Add: `feature_values(table, feature_name, selector=...)`, `list_workflow_executions(workflow)`
  - Remove: `add_features`, `fetch_table_features`, `list_feature_values`, `select_by_workflow`
  - Keep: `create_feature`, `lookup_feature`, `delete_feature`, `find_features`
- `feature.py`
  - Add: `FeatureRecord.select_by_workflow(workflow, *, container)` classmethod factory
  - Keep existing selectors unchanged
- `dataset/dataset.py`
  - Add: `feature_values`, `lookup_feature`, `list_workflow_executions` (delegates to `ml`)
- `dataset/dataset_bag.py`
  - Rewrite feature read path through the bag cache
  - Add: `lookup_feature`, `list_workflow_executions`
  - Remove: `fetch_table_features`, `list_feature_values`
- `execution/execution.py`
  - Change: `add_features(records)` stages to SQLite instead of write-through
  - Add: `_flush_staged_features()` called during completion flush (after assets)
  - Add: resume detection for `Pending` staged features (piggybacks on §8 staged-asset resume)
- `interfaces.py`
  - Update `DatasetLike` and related protocols to the three-method surface plus `lookup_feature` and `list_workflow_executions`

### SQLite schema addition (§8 integration)

```sql
CREATE TABLE staged_features (
    stage_id         INTEGER PRIMARY KEY,
    feature_table    TEXT NOT NULL,        -- schema.table
    feature_name     TEXT NOT NULL,
    record_json      TEXT NOT NULL,        -- FeatureRecord.model_dump_json()
    created_at       TEXT NOT NULL,
    status           TEXT NOT NULL,        -- Pending | Uploaded | Failed
    uploaded_at      TEXT,
    error            TEXT
);
CREATE INDEX idx_staged_features_feature ON staged_features(feature_table, status);
```

## Data flow

### Read — online

`ml.feature_values("Image", "Glaucoma", selector=select_newest)`:

1. Resolve `Image` table and `Glaucoma` feature definition.
2. Query feature association table via path builder, producing raw `FeatureRecord` instances.
3. Group by target RID, apply selector per group (one record per target RID surviving).
4. Yield the surviving records.

### Read — dataset-scoped

`dataset.feature_values(...)`:

1. Same as online, but the underlying query filters target RIDs to dataset members.

### Read — offline (bag)

`bag.feature_values(...)`:

1. On first access, populate bag-local denormalization cache for this feature from bag SQLite source tables.
2. Read records from cache table.
3. Group, apply selector, yield — identical pipeline to online.

### Write — normal path

`exe.add_features(records)`:

1. Auto-fill `Execution` on each record if unset.
2. Validate single feature definition across records.
3. Write rows to `staged_features` with status `Pending`.

On execution completion:
1. Flush staged assets (existing §8 behavior).
2. Flush staged features: group by feature table, batch-insert, mark `Uploaded`/`Failed`.

### Write — crash-resume

1. Execution restarts.
2. §8 resume path detects `Pending` rows in both staged-asset and staged-feature tables.
3. Flush re-runs. SQLite write-through is the durability boundary.

### Write — offline-compute, online-flush

1. User constructs records via `bag.lookup_feature().feature_record_class()(...)` offline.
2. Records are plain Pydantic objects, portable.
3. Later, back online, `exe.add_features(records)` stages and flushes.

## Error handling

**Discovery / lookup:** unknown table → `DerivaMLTableNotFound`; unknown feature → `DerivaMLNotFoundError`. Same messages across containers.

**Reads:**
- Unknown table or feature → `DerivaMLNotFoundError`.
- Selector raises → propagates unchanged.
- Selector returns `None` → target RID omitted from iterator (documented as "feature absent for this RID").
- Bag cache corrupt/missing → `DerivaMLDataError` with bag path and re-extract guidance.

**Selector factory:**
- Unknown workflow → `DerivaMLNotFoundError` at factory construction.
- Missing `container` kwarg → Python `TypeError`.
- Wrong `container` type → `TypeError` with expected protocol.

**Writes:**
- Mixed feature definitions → `DerivaMLValidationError` at stage time (before SQLite write).
- Empty records list → `ValueError`.
- SQLite staging failure → `DerivaMLDataError`, execution state `Failed`.
- Flush failure for a group → group marked `Failed`, others continue, overall `DerivaMLUploadError` with summary; retry via resume.
- Asset-referencing feature, referenced asset upload failed → FK rejection at insert time → `DerivaMLUploadError` identifying the offending asset RID.

**Retired-API calls:** each raises `DerivaMLException` with a message naming the replacement (table above).

## Testing

**Symmetry compliance suite (new, high-value):** parametrized over `DerivaML`, `Dataset`, `DatasetBag`. Asserts that for a fixture feature on a fixture catalog + its downloaded bag:

- `find_features(table)` returns matching metadata.
- `feature_values(table, name)` yields matching records (modulo timestamp tolerance).
- `lookup_feature(table, name).feature_record_class()` returns a class with matching field shape.
- `list_workflow_executions(workflow)` returns the same list (order-independent comparison).

This suite is the contract: any future container claiming feature capability must pass it.

**Selector tests:**
- Existing selector tests adapted to the new `feature_values` return type.
- New: `FeatureRecord.select_by_workflow` with each container type; unknown-workflow raises at construction.

**Bag cache:**
- First access populates.
- Subsequent access reads from cache (no recompute).
- Corrupt/missing cache raises with recovery pointer.

**Execution-staged writes:**
- Valid records stage with auto-filled Execution and status `Pending`.
- Mixed definitions fail at stage time, nothing in SQLite.
- Successful completion transitions all `Pending` → `Uploaded`.
- Simulated mid-flush failure: affected group `Failed`, others `Uploaded`.
- Crash-before-flush + resume: `Pending` rows re-flushed, no ermrest duplicates.
- Flush-after-assets ordering: feature flush attempts occur only after asset upload completes.
- Asset-referencing feature with pre-flushed asset succeeds; with missing asset raises clear error.

**Offline-to-online write cycle:**
- Construct records from bag without catalog.
- Hand to later `exe.add_features`.
- Assert round-trip correctness.

**Retired APIs:**
- Each retired method raises `DerivaMLException` with the expected replacement pointer.

**Unchanged (not re-tested beyond current coverage):**
- `Denormalizer` — untouched by S2.
- `create_feature` — untouched.
- `FeatureRecord` dynamic class generation — untouched.

## Documentation requirements

Every public method touched by S2 must ship with complete docstrings including runnable examples. This is a hard requirement, not a nice-to-have — the API is being redesigned, so the docstrings are the canonical reference until docs/mkdocs catch up.

### Per-method docstring contract

Every public method in the new surface (`find_features`, `feature_values`, `lookup_feature`, `list_workflow_executions`, `add_features`, `FeatureRecord.select_by_workflow`, and the retired-API shims with their error-path pointers) must include:

1. **One-line summary.**
2. **Extended description** — what it does, when to use it, how it relates to sibling methods.
3. **`Args:` section** — every parameter typed and described. Required vs. optional. Default value rationale. Cross-references to related methods.
4. **`Returns:` section** — the return type and semantic meaning. For iterators, mention streaming characteristics. For writes, mention staging vs. flush.
5. **`Raises:` section** — every exception the method can raise, keyed to the conditions in §Error handling. For retired APIs: the exact `DerivaMLException` message.
6. **`Example:` section** — at least one runnable example, typically two:
   - The common happy path.
   - A second example showing an interesting variation (selector use, cross-container use, offline use, etc.).
7. **Cross-container note** where applicable — if the method exists on `DerivaML`, `Dataset`, and `DatasetBag` with identical signatures, the docstring says so and notes any behavior differences (e.g., bag-side cache population on first access).

### Example docstring (canonical shape)

```python
def feature_values(
    self,
    table: Table | str,
    feature_name: str,
    selector: Callable[[list[FeatureRecord]], FeatureRecord | None] | None = None,
) -> Iterable[FeatureRecord]:
    """Yield feature values for a single feature, one record per target RID.

    Returns an iterator of typed ``FeatureRecord`` instances. Each record is
    wide in shape — target RID, all value columns (vocab terms, asset
    references, metadata columns), and provenance columns (``Execution``,
    ``RCT``) — exposed as typed attributes.

    When a ``selector`` is provided, records are grouped by target RID and
    the selector collapses each group to a single survivor. Target RIDs
    whose group's selector returns ``None`` are omitted. When no selector
    is provided, every raw record is yielded — multiple records per target
    RID are possible.

    This method has identical signatures and semantics across ``DerivaML``,
    ``Dataset``, and ``DatasetBag``. The bag implementation reads from a
    per-feature denormalization cache populated on first access; subsequent
    calls are cheap.

    Args:
        table: Target table the feature is defined on (name or Table).
        feature_name: Name of the feature to read.
        selector: Optional callable ``(list[FeatureRecord]) -> FeatureRecord | None``
            used to reduce multi-value groups. See ``FeatureRecord.select_newest``,
            ``FeatureRecord.select_by_workflow``, etc. for built-ins.

    Returns:
        Iterator of ``FeatureRecord`` — one record per target RID after
        selector reduction (or all raw records if no selector).

    Raises:
        DerivaMLTableNotFound: ``table`` does not exist.
        DerivaMLNotFoundError: ``feature_name`` is not a feature on ``table``.
        DerivaMLDataError: (bag only) the feature cache is corrupt or missing.

    Example:
        Get the newest Glaucoma label per image::

            >>> from deriva_ml.feature import FeatureRecord
            >>> for rec in ml.feature_values(
            ...     "Image", "Glaucoma", selector=FeatureRecord.select_newest,
            ... ):
            ...     print(f"{rec.Image}: {rec.Glaucoma} (by {rec.Execution})")

        Filter by a specific workflow, works identically on a downloaded bag::

            >>> workflow = ml.lookup_workflow("Glaucoma_Training_v2")
            >>> sel = FeatureRecord.select_by_workflow(workflow, container=bag)
            >>> labels = [r.Glaucoma for r in bag.feature_values(
            ...     "Image", "Glaucoma", selector=sel,
            ... )]
    """
```

### Inline comments

For the non-trivial machinery — `staged_features` lifecycle, bag cache population, selector factory resolution, flush-after-assets ordering — inline comments explain *why*, not just *what*. Specifically:

- `_flush_staged_features` — comment the ordering contract (assets first, features second) with a pointer to the §8 design.
- `bag_feature_cache` — comment the immutability invariant and the populate-on-first-access trigger.
- `FeatureRecord.select_by_workflow` factory body — comment that workflow resolution is eager (at construction) and why (fail-fast on unknown workflow).
- Retired-API error shims — comment linking to this spec as the source of truth for the replacement.

### Module-level docstrings

Any new module (`bag_feature_cache.py`, expanded sections of `core/mixins/feature.py`, etc.) gets a module-level docstring naming this spec and stating the module's responsibility in one paragraph.

## Testing (expanded — complete coverage, live catalog included)

The §Testing section above names the test categories; this section states the coverage standard.

### Coverage requirement

Every public method and every retired-API shim added, modified, or removed by S2 must have explicit test coverage. No method ships without at least one test exercising its happy path, its parameter variations, and its documented error paths.

### Test organization

Organized by the existing `tests/feature/`, `tests/execution/`, `tests/dataset/` structure. New modules when a category has no existing home:

- `tests/feature/test_feature_values.py` — the three-container parametrized compliance suite.
- `tests/feature/test_select_by_workflow.py` — selector factory, including unknown-workflow-at-construction.
- `tests/feature/test_retired_apis.py` — each retired method raises the expected exception with the expected replacement pointer.
- `tests/execution/test_staged_features.py` — staging, flush ordering, crash-resume, error paths.
- `tests/dataset/test_bag_feature_cache.py` — first-access population, immutability, corrupt-cache error path.

### Live catalog coverage

Integration coverage against a live Deriva catalog is **required**, not optional, for these flows:

- `feature_values` on `DerivaML` (live ermrest query path).
- `feature_values` on `Dataset` (live ermrest + dataset member filter).
- `exe.add_features` stage + real flush to a live catalog (round-trip).
- Flush ordering: assets flush to live Hatrac + ermrest, features flush against live ermrest referencing those asset rows.
- Crash-resume: SQLite `Pending` rows flushed to live catalog on resume, no duplicate rows in the catalog after resume.
- Retired-API error path: `ml.add_features` on a live catalog raises the expected message.
- Offline-to-online cycle: construct records from a downloaded bag (no catalog), hand to `exe.add_features` against a live catalog, assert round-trip.

These live tests run under the existing `tests/conftest.py` fixtures (`test_ml`, `catalog_with_datasets`, `execution_with_hydra_config`, or equivalents). They require `DERIVA_HOST` and are part of the long-run suite (see CLAUDE.md for incremental invocation).

### Offline / unit coverage

- Three-container symmetry suite covers bag-side reads without a live catalog once the bag is materialized.
- Selector factory with unknown workflow raises synchronously — unit-testable.
- SQLite `staged_features` transitions (`Pending` → `Uploaded`, `Pending` → `Failed`, resume detection) unit-testable against a temp SQLite.
- Retired-API shims raise synchronously — unit-testable.

### Parametrized symmetry suite — the contract test

`tests/feature/test_feature_values.py` contains a parametrized test class that runs the same assertions across `DerivaML`, `Dataset`, `DatasetBag`:

```python
@pytest.fixture(params=["ml", "dataset", "bag"])
def feature_container(request, test_ml, catalog_with_datasets, materialized_bag):
    return {"ml": test_ml, "dataset": catalog_with_datasets, "bag": materialized_bag}[request.param]

class TestFeatureValuesSymmetry:
    def test_find_features_matches_definition(self, feature_container): ...
    def test_feature_values_yields_expected_records(self, feature_container): ...
    def test_feature_values_with_selector(self, feature_container): ...
    def test_lookup_feature_returns_usable_record_class(self, feature_container): ...
    def test_list_workflow_executions_matches(self, feature_container): ...
```

Any future container claiming feature capability must pass this suite as its acceptance test.

### Regression guards

Specific bugs that have bitten before in this API area get explicit regression tests:

- Auto-filled `Execution` RID on `exe.add_features` — test covers the record-with-Execution-None and record-with-Execution-set cases.
- Empty records list raises `ValueError` (existing behavior preserved).
- Bag `lookup_feature` works without a live catalog connection (offline test uses a pre-materialized bag, no `DERIVA_HOST`).

### Coverage target

When the test suite runs with coverage enabled, every public method added or modified by S2 must show non-trivial coverage (branches for each `Raises:` path, happy path, selector-present and selector-absent). A coverage report is part of the final review gate before merging.

## Open dependencies

- `staged_features` SQLite table lives alongside the §8 staged-asset tables; the §8 design must be in place before the write-path changes can ship.
- Bag feature cache depends on existing bag SQLite structure (feature tables already mirrored).
- No changes to the live catalog schema.

## Migration

- No data migration — this is an API consistency change, not a schema change.
- Retired APIs raise at first call with a clear pointer to the replacement; there is no silent period.
- Client code using `ml.add_features` must move to `exe.add_features` within an execution context.
- Client code using `fetch_table_features` or `list_feature_values` must move to `feature_values` (renamed, same signature).
- Client code using `ml.select_by_workflow(records, workflow)` must move to the selector-factory form.

## Out of scope

- Metric or Param as first-class concepts (dropped during brainstorming).
- Wide-form feature DataFrame return type (kept as `Iterable[FeatureRecord]`; pandas is a caller concern).
- Multi-feature merged reads on `feature_values` (use `Denormalizer`).
- Changes to `Denormalizer`, `create_feature`, `FeatureRecord` class generation.
- Deprecation shims or silent fallbacks.
