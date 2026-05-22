# Engineer audit — dataset/ subsystem (2026-05-22)

Pre-release audit of `src/deriva_ml/dataset/` and its tests against
release v1.37.1. Four lenses: test coverage, duplication, logic
clarity, docstring quality.

## Summary

- 14 source modules audited (22k+ LoC counting tests).
- 42 findings: **0 P0**, **6 P1**, **22 P2**, **14 P3**.
- Top themes:
  1. **Two correctness bugs in `Dataset.add_dataset_members` cycle check** — `set(self.dataset_rid)` produces character-set, not a singleton; recursion path is dead. P1.
  2. **God-functions** in `Dataset.estimate_bag_size` (220 lines), `split_dataset` (590 lines), `restructure_assets` (~250 lines body), `get_dataset_minid` (220 lines). Each does I/O, planning, and result assembly in a single body. P1/P2.
  3. **Parallel torch/tf adapters** — `_bag_element_is_asset`, `_build_row_lookup`, `_resolve_asset_path` are duplicated character-for-character across `torch_adapter.py` and `tf_adapter.py`. The selector-grouping in `Dataset.feature_values` / `DatasetBag.feature_values` is also parallel. P2.
  4. **Retired-API shims** (`fetch_table_features`, `list_feature_values` on `DatasetBag`) violate the workspace "no backwards-compat shims" rule. P2.
  5. **Test gaps** for public methods: `Dataset.to_markdown`, `Dataset.display_markdown`, `Dataset.get_chaise_url`, `DatasetBag.list_tables`, `DatasetBagBuilder.build_bag` (the heaviest method in the class). P2.

---

## Findings by module

### src/deriva_ml/dataset/dataset.py

#### [P1] `add_dataset_members` cycle check is structurally broken
**Location:** `src/deriva_ml/dataset/dataset.py:1922-1933, 1965`
**Issue:** Two bugs in `check_dataset_cycle`:

1. `path = path or set(self.dataset_rid)` calls `set()` on a string,
   producing `{"4", "H", "M"}` from RID `"4HM"` instead of the
   intended `{"4HM"}` singleton. Cycle detection therefore false-
   positives on any single-character match with the dataset RID's
   characters and misses every real cycle.
2. The `path` parameter is never recursively threaded — the only
   call site (`check_dataset_cycle(rid_info.rid)`) supplies one
   argument, so the second-arg branch is dead code; there's no
   actual graph walk.

The "Creating cycle of datasets is not allowed" path is effectively
never hit on real inputs, and a cycle check with this shape was
clearly never working correctly. The CLAUDE.md "RIDs are opaque:
equality only" rule applies — `set(rid_string)` is parsing.

**Evidence:**
```python
def check_dataset_cycle(member_rid, path=None):
    path = path or set(self.dataset_rid)   # bug: char-set, not RID-set
    return member_rid in path
# Caller:
if rid_info.table == self._dataset_table and check_dataset_cycle(rid_info.rid):
    raise DerivaMLException("Creating cycle of datasets is not allowed")
```

**Suggested fix:** Either implement a real graph walk (traverse
`list_dataset_children(recurse=True)` on each candidate Dataset
member and reject if `self.dataset_rid` appears), or delete the
nested helper and document that cycles must be prevented downstream.
Add a regression test that nests A→B→A and asserts the raise.

#### [P1] `_version_snapshot_catalog` has a dead `and str:` clause
**Location:** `src/deriva_ml/dataset/dataset.py:2779`
**Issue:** Truthiness check appends the literal `str` class to the
condition. `str` is always truthy, so the clause is dead; the
intended guard was presumably `isinstance(dataset_version, str) and
dataset_version` (non-empty). Mostly benign — `DatasetVersion.parse("")`
would raise — but it's a clear bug that signals the codepath wasn't
exercised when written.

**Evidence:**
```python
if isinstance(dataset_version, str) and str:
    dataset_version = DatasetVersion.parse(dataset_version)
```

**Suggested fix:** `if isinstance(dataset_version, str) and dataset_version:`,
or drop the second clause entirely.

#### [P1] `Dataset.estimate_bag_size` is a 220-line god-function
**Location:** `src/deriva_ml/dataset/dataset.py:2415-2633`
**Issue:** Single method does (1) snapshot resolution, (2)
async catalog construction with manual URL parsing, (3) URI-to-path
extraction via a nested helper, (4) query-item list construction,
(5) async query orchestration, (6) RID/length/sample
post-processing, (7) per-table aggregation, (8) final dict
assembly. Two nested helpers (`_extract_path`, `_run_query`,
`_run_all_queries`) live inside the method.

Tests rely entirely on integration-level fixtures
(`test_estimate_bag_size.py` is 600 lines of demo-catalog
exercises). The fine-grained steps — URI extraction, sample CSV
estimation, asset/non-asset detection — are not unit-testable
without standing up a live catalog.

**Suggested fix:** Extract three helpers: `_build_estimate_queries(table_queries) -> list[_QueryItem]`, `_run_estimate_queries(catalog, items) -> tuple[rids_by_table, lengths_by_table, samples_by_table]`, `_assemble_estimate(...) -> dict`. The async machinery already lives behind `run_async`; pulling it out makes the orchestration testable with a mock catalog.

#### [P1] `Dataset` has retired-API surface that should be deleted
**Location:** `src/deriva_ml/dataset/dataset_bag.py:654-690`
**Issue:** `DatasetBag.fetch_table_features` and
`DatasetBag.list_feature_values` are stubs that raise with a
"renamed to feature_values" message. Workspace `CLAUDE.md`:
"No backwards-compat shims — if something is unused, delete it."
There's an entire `tests/feature/test_retired_apis.py` keeping
them alive. Same shape appears for `DerivaML.fetch_table_features`
/ `list_feature_values` in `core/mixins/feature.py` and
`Asset.list_feature_values` — but the dataset audit owns only the
bag pair.

**Suggested fix:** Delete both methods and the corresponding
`test_retired_apis.py::test_bag_*` entries. Anyone still calling
the old names will get the standard `AttributeError`, which is
the workspace's documented stance.

#### [P2] Dataset.list_workflow_executions is misleadingly named
**Location:** `src/deriva_ml/dataset/dataset.py:629-660`
**Issue:** The method's own docstring acknowledges: "Despite the name,
the live Dataset implementation does **not** filter the returned
execution list to executions whose outputs touch the dataset's
member set — it returns every execution of the workflow." Meanwhile
`DatasetBag.list_workflow_executions` *is* dataset-scoped. The two
implementations of the same protocol method have divergent semantics
that callers have to read the docstring to discover.

**Suggested fix:** Either (a) implement the dataset-scoping filter
in the live `Dataset` (preferred — symmetry with the bag), or
(b) rename to `list_workflow_executions_unscoped` and add a
scoped variant. Continuing to ship two methods with the same name
and different semantics is a footgun.

#### [P2] `feature_values` selector grouping is duplicated between Dataset and DatasetBag
**Location:** `src/deriva_ml/dataset/dataset.py:591-604` and
`src/deriva_ml/dataset/dataset_bag.py:637-652`
**Issue:** Both implementations end with the identical pattern:
group by target RID, call selector once per group, yield non-None
choices. The grouping step is verbatim parallel code.

**Suggested fix:** Move the selector-application loop into
`deriva_ml.feature` (alongside `FeatureRecord.select_newest`) as a
free function `apply_selector(records, target_col, selector)`.
Both call sites become a one-liner. Reduces drift risk if
selector semantics change.

#### [P2] `add_dataset_types` redoes lookups already done by `add_dataset_type`
**Location:** `src/deriva_ml/dataset/dataset.py:414-424`
**Issue:** Loop calls `lookup_term` to get the term name, then
`add_dataset_type` re-runs `lookup_term` internally. Double
catalog round-trip per type.

**Evidence:**
```python
for term in types_to_add:
    if isinstance(term, VocabularyTerm):
        term_name = term.name
    else:
        term_name = self._ml_instance.lookup_term(MLVocab.dataset_type, term).name
    if term_name not in self.dataset_types:
        self.add_dataset_type(term, _skip_version_increment=True)  # re-looks up
        added_types.append(term_name)
```

**Suggested fix:** Resolve all terms once at the top, then call a
shared internal `_add_dataset_type_unchecked(vocab_term, ...)` that
takes a `VocabularyTerm` and skips the lookup.

#### [P2] `_create_or_advance_dev_row` is a 120-line method that mixes lifecycle paths
**Location:** `src/deriva_ml/dataset/dataset.py:1303-1421`
**Issue:** Single method handles both create-new-dev-row and
advance-existing-dev-row, with the branch determined inline at
the top. Two distinct write paths with different conditional
logic for RMT/Snapshot/Version share one body. The
fault-injection comment on line 1418 also references the *wrong*
exception message variable substitution — the f-string starts at
`f"Concurrent modification of dev row {current_dev.version_rid} "`
and then a regular string `"for dataset {self.dataset_rid}: ..."`
follows it on lines 1417-1420 (`self.dataset_rid` is not
interpolated).

**Evidence (interpolation bug):**
```python
raise DerivaMLException(
    f"Concurrent modification of dev row {current_dev.version_rid} "
    "for dataset {self.dataset_rid}: another writer advanced the "  # not an f-string
    "row between this call's read and write. Re-read the dataset "
    "and retry if the new state is still what you intended."
)
```
The literal text `{self.dataset_rid}` is shown to the user
verbatim. Compare with the same message in `release()` (line 1043)
which uses `f"... {self.dataset_rid}: ..."` correctly. P1
candidate for the interpolation bug specifically — file as P2 since
the user can still tell which dataset (the `version_rid` resolves
it).

**Suggested fix:** Split into `_insert_dev_row` and `_advance_dev_row`;
share the concurrency-detection raise via a helper.

#### [P2] `DatasetHistory` model docstring drifts from current attributes
**Location:** `src/deriva_ml/dataset/aux_classes.py:204-235`
**Issue:** Class docstring lists `dataset_version`, `dataset_rid`,
`version_rid`, `minid`, `snapshot` but the actual model has
`execution_rid`, `description`, `spec_hash` too. Critical detail:
the docstring says "Catalog snapshot ID of when the version
record was created" but doesn't mention that `snapshot` is
`None` for dev rows — this is the load-bearing semantic from
ADR-0003 and callers need to know it.

**Suggested fix:** Add `execution_rid`, `description`, `spec_hash`
to the Attributes list. Annotate `snapshot` with "None for dev
rows; only released rows pin a snapshot."

#### [P2] `Dataset.cache_denormalized` lacks `Raises:` and exception docs
**Location:** `src/deriva_ml/dataset/dataset.py:1774-1849`
**Issue:** Docstring documents Args/Returns/Example but no
`Raises:` block. The denormalizer raises
`DerivaMLDenormalizeUnrelatedAnchor` (referenced in arg
description but not in `Raises:`), and the underlying paged
client can raise on snapshot resolution failure. Mirrors the same
gap in `get_denormalized_as_dataframe` / `get_denormalized_as_dict`.

**Suggested fix:** Add `Raises:` with the documented exceptions
(`DerivaMLDenormalizeUnrelatedAnchor`, `DerivaMLException`).

#### [P2] `_increment_dataset_version` docs say "Internal" but it's the catalog-clone path
**Location:** `src/deriva_ml/dataset/dataset.py:877-924`
**Issue:** The docstring labels this "Internal helper preserved
from the pre-dev-versioning model... Not for user-facing release
work; use release() for that." The leading underscore in the name
matches that. But the only caller is `catalog/clone.py`, which is
a public surface. The leading-underscore convention typically
means "package-private", which is correct here, but the docstring
says "Internal" and `release()` is the user path — the boundary
is fuzzy. Worth promoting to a public-named helper with a clear
"clone-only" caveat, or moving it to `dataset.aux_classes` /
`catalog.clone` to make ownership obvious.

#### [P2] `release()` 120-line body could thin out
**Location:** `src/deriva_ml/dataset/dataset.py:926-1049`
**Issue:** Method does (1) dev-entry validation, (2) released-entry
anchor lookup, (3) catalog snapshot stamp, (4) RMT-correlated
update, (5) raise on contention. The `_create_or_advance_dev_row`
helper already encapsulates a similar pattern but isn't reused —
extract `_promote_dev_to_release(...)` symmetric with
`_create_or_advance_dev_row` so the dev-vs-release lifecycle reads
as parallel methods.

#### [P3] `Dataset.to_markdown` / `display_markdown` / `get_chaise_url` have no tests
**Location:** `src/deriva_ml/dataset/dataset.py:757-821`
**Issue:** Three public methods with no test coverage. Each is
small and pure (no catalog write) — easy to test with a fake
`_ml_instance`.

#### [P3] `Dataset.__init__` description arg is stored but `dataset_types` is fetched live
**Location:** `src/deriva_ml/dataset/dataset.py:113-139`
**Issue:** `description` is cached on the instance; `dataset_types`
is a property that re-queries the catalog every access. The
asymmetry isn't documented anywhere, and the description on the
object can drift from the catalog (e.g. if another writer updates
the Dataset row). Tests should at minimum acknowledge this.

#### [P3] `_get_dataset_type_association_table` returns tuple but caller usually only uses second element
**Location:** `src/deriva_ml/dataset/dataset.py:173-187`
**Issue:** Returns `(atable_name, atable_path)`. Callers
(`dataset_types`, `add_dataset_type`, `remove_dataset_type`) use only
`atable_path`. The name is computed via `find_associations()[0].name`
in case of multiple associations — but that pop-the-first behavior
is suspect; if there are multiple Dataset_Type associations, the
choice is arbitrary.

#### [P3] `_estimate_csv_bytes` static helper has no test
**Location:** `src/deriva_ml/dataset/dataset.py:2713-2749`
**Issue:** Pure function — sample rows → estimated byte count.
Could be unit-tested with synthetic rows. Currently exercised
only through `estimate_bag_size` integration tests.

### src/deriva_ml/dataset/dataset_bag.py

#### [P1] `DatasetBag` re-implements selector + grouping rather than sharing with Dataset
**Location:** `src/deriva_ml/dataset/dataset_bag.py:532-652`
**Issue:** See dataset.py finding above — selector grouping is
parallel. Bag also has extra logic (over-reach filtering,
execution_rids Python-side filter, materialize_limit check) that
makes the parallel less clean but the selector tail is identical.

**Suggested fix:** Same — shared helper in `deriva_ml.feature`.

#### [P2] Retired-API shims on DatasetBag
**Location:** `src/deriva_ml/dataset/dataset_bag.py:654-690`
**Issue:** Delete per workspace convention.

#### [P2] `_dataset_table_view` SQL is opaque
**Location:** `src/deriva_ml/dataset/dataset_bag.py:307-352`
**Issue:** 45-line method builds a SQL UNION across all paths
from Dataset to the target table by introspecting
`self.model._schema_to_paths()`. The dataset RID list construction
includes "all nested datasets" silently — there's no path filter
to scope nested-dataset inclusion. Comments are sparse, and the
method is consumed only by `restructure._get_reachable_assets`
(restructure.py:200) — moving it to `restructure.py` as a private
helper would tighten ownership.

#### [P2] DatasetBag.list_tables has no tests, no Raises section
**Location:** `src/deriva_ml/dataset/dataset_bag.py:221-230`
**Issue:** Public method, undocumented exceptions, no tests in
`tests/dataset/test_bag_api_coverage.py`.

#### [P2] `as_torch_dataset` / `as_tf_dataset` docstrings are 130+ lines each
**Location:** `src/deriva_ml/dataset/dataset_bag.py:1130-1302`,
`1304-1483`
**Issue:** The two docstrings are near-identical prose (target
arity, missing policy, RID-last contract) — drift risk is high.
The `as_torch_dataset` Returns section and the `as_tf_dataset`
Returns section describe the same shape contract twice.

**Suggested fix:** Either accept the duplication and add a
docstring-style invariant test (test that both methods'
parameters and Args texts share the matching prose), or extract
the shared sections into a module-level constant referenced by
both. Given how big they are, a shared "design contract" link
in each docstring plus per-method specifics would be cleaner.

#### [P3] DatasetBag fetches execution_rid at init time, not lazily
**Location:** `src/deriva_ml/dataset/dataset_bag.py:151-153`
**Issue:** `self.execution_rid = execution_rid or (self._catalog._get_dataset_execution(self.dataset_rid) or {}).get("Execution")` runs an unconditional SQLite query at `__init__`, accessing a private method on the catalog (`_get_dataset_execution`). Lazy property would avoid the cost when callers don't read it.

#### [P3] `current_version` property docstring claims immutability but doesn't enforce
**Location:** `src/deriva_ml/dataset/dataset_bag.py:181-192`
**Issue:** "Unlike the live Dataset class, this value is immutable" — `self._current_version` is set in `__init__` but a writer could `bag._current_version = ...`. Add `@property` setter that raises, or document the convention more honestly.

### src/deriva_ml/dataset/aux_classes.py

#### [P2] `DatasetHistory` model fields don't all appear in docstring
**Location:** `src/deriva_ml/dataset/aux_classes.py:204-215`
**Issue:** See above.

#### [P3] `DatasetVersion.parse` re-runs `Version.__init__` redundantly
**Location:** `src/deriva_ml/dataset/aux_classes.py:138-165`
**Issue:** `Version(version)` parses; then a new `cls` instance has `Version.__init__(instance, str(v))` called on it (re-parses). Could `__new__` and copy `__dict__` directly. Minor perf, but the two-parse round-trip is a code smell.

#### [P3] `DatasetMinid.dataset_snapshot` docstring missing
**Location:** `src/deriva_ml/dataset/aux_classes.py:268-276`
**Issue:** Computed-field property has only an inline comment ("`version_rid` is `{rid}` or `{rid}@{snapshot}`") — no docstring. Same for `dataset_rid` at line 262-266.

### src/deriva_ml/dataset/bag_builder.py

#### [P2] `DatasetBagBuilder.build_bag` has no direct unit/integration test
**Location:** `src/deriva_ml/dataset/bag_builder.py:240-334`
**Issue:** The heaviest method in the class (95 lines), responsible for the snapshot-aware CatalogBagBuilder dance and zip-path recovery. `tests/dataset/test_bag_builder.py` only covers spec generation, annotations, aggregate queries, and policy. The `_SnapshotAwareCatalogBagBuilder._run_export` override and the timeout-restoration `try/finally` are entirely uncovered.

**Suggested fix:** Add at least one test that calls `build_bag` against a small demo dataset and asserts (a) returned path is a `.zip` that exists, (b) bag name is `Dataset_{rid}`, (c) timeout restoration happens on exception.

#### [P2] `_dataset_nesting_depth` is computed but only annotation flow uses it
**Location:** `src/deriva_ml/dataset/bag_builder.py:686-717`
**Issue:** Method exists and is documented but isn't called anywhere in `bag_builder.py` or under `src/deriva_ml/`. Likely dead.

**Evidence:**
```bash
$ grep -rn "_dataset_nesting_depth" src/deriva_ml/
# only the definition matches
```

**Suggested fix:** Delete unless an external skill uses it. Per workspace rule "if something is unused, delete it."

#### [P3] `_SnapshotAwareCatalogBagBuilder` is documented as workaround for upstream #114 — should track resolution
**Location:** `src/deriva_ml/dataset/bag_builder.py:71-119`
**Issue:** Inline note "See issue #114" — track upstream so this can be retired when the upstream fix lands.

#### [P3] `anchors_for` walks descendants twice
**Location:** `src/deriva_ml/dataset/bag_builder.py:723-754`
**Issue:** `_iter_descendant_rids` walks; later `_exclude_empty_associations` also walks descendants (`rids_to_scan`). Both walks are full traversals via `list_dataset_children`. Cache the descendant set per build call.

### src/deriva_ml/dataset/bag_cache.py

#### [P3] `CacheStatus.cached_incomplete` aliased but never tested
**Location:** `src/deriva_ml/dataset/bag_cache.py:67-71`
**Issue:** Back-compat alias for `cached_holey`. No test exercises the alias (only the canonical value). If it's a load-bearing back-compat surface, pin a test; otherwise delete per workspace convention.

#### [P3] `BagCache._is_fully_materialized` is a static method but reaches via class
**Location:** `src/deriva_ml/dataset/bag_cache.py:174-206`
**Issue:** Called as `BagCache._is_fully_materialized(bag_path)` from `bag_download.py:657`. Promote to a module-level free function (since it has no class state) — easier to test, easier to import.

### src/deriva_ml/dataset/bag_download.py

#### [P1] `get_dataset_minid` is a 220-line monolith
**Location:** `src/deriva_ml/dataset/bag_download.py:389-605`
**Issue:** Single function does (a) version-record resolution, (b) spec hash computation, (c) Tier-1 cache lookup, (d) Tier-2 MINID branch, (e) Tier-3 client-side bag branch, (f) DatasetMinid construction with shared snapshot-vs-None RID guard repeated twice (lines 531, 599). The three-tier comment headers help reading but the cyclomatic complexity is high.

**Suggested fix:** Extract `_resolve_version_record`, `_tier1_local_cache_lookup`, `_tier2_minid_path`, `_tier3_client_path`, and `_build_dataset_minid(rid, snapshot, version, location, checksum)` helper for the duplicated RID-vs-snapshot construction.

#### [P2] `create_dataset_minid` `spec` arg is "vestigial on the non-MINID arm"
**Location:** `src/deriva_ml/dataset/bag_download.py:340-352`
**Issue:** The comment at 343-352 explicitly says the spec parameter is unused on the `use_minid=False` branch ("CatalogBagBuilder recomputes its own spec from the same anchors/policy"). Carrying a parameter that's silently ignored on one branch is a footgun.

**Suggested fix:** Either pass the precomputed spec into `build_bag` so both branches honor it, or drop `spec` from this function's signature on the `use_minid=False` path (route through a different helper).

#### [P2] `fetch_minid_metadata` `dataset` arg is unused
**Location:** `src/deriva_ml/dataset/bag_download.py:101-119`
**Issue:** `del dataset` on line 116. The parameter is kept "for future logging / auth context" per the inline comment. Per workspace "no over-engineering" rule, drop it; re-add when actually needed.

#### [P3] `download_dataset_minid` mixes three source types with no dispatch helper
**Location:** `src/deriva_ml/dataset/bag_download.py:184-194`
**Issue:** Three `if/elif/else` branches choose archive source by sniffing `minid.bag_url` (`use_minid`, `file://`, else). The dispatch is fine but each branch contributes to a method that's already 130 lines; extract `_fetch_archive(minid, use_minid, tmp_dir) -> Path` to keep this method focused on cache placement.

### src/deriva_ml/dataset/restructure.py

#### [P1] `restructure_assets` is a 500-line function (with docstring)
**Location:** `src/deriva_ml/dataset/restructure.py:250-746`
**Issue:** Pure function but its body (post-docstring) is still ~250 lines covering: input validation, asset-table detection, type-path map, asset-to-dataset map, feature/column target classification, per-asset processing with branch on `missing` policy, file placement (symlink/copy/transform).

**Suggested fix:** Extract `_resolve_grouping_path(asset, targets, ..., missing)`, `_place_asset(source, target_path, file_transformer, use_symlinks)`. The current shape works but is hard to navigate.

#### [P2] Asset source-path fallback to `_catalog._database_model.bag_path` is private-API reach
**Location:** `src/deriva_ml/dataset/restructure.py:611-614`
**Issue:** `bag._catalog._database_model.bag_path` — reaching through two private attributes on the catalog. The bag already exposes `bag.path` (per dataset_bag.py:195). Use that.

**Evidence:**
```python
try:
    bag_root = Path(bag._catalog._database_model.bag_path)
    source_path = bag_root / "data" / "asset" / asset.get("RID", "") / asset_table / Path(filename).name
except AttributeError:
    pass  # catalog doesn't have _database_model (e.g. in tests)
```

**Suggested fix:** Replace with `bag_root = bag.path`. The `try/except AttributeError` becomes unnecessary.

#### [P2] `_default_dir_name_from_target` skip-cols set is hardcoded
**Location:** `src/deriva_ml/dataset/restructure.py:104-105`
**Issue:** `_skip_cols = {"RID", "RCT", "RMT", "RCB", "RMB", "Feature_Name", "Execution"}` — the catalog's system columns are listed in
`deriva_ml.core.definitions.SYSTEM_COLUMNS` or similar; duplicating
the set here means it drifts if Deriva adds another (e.g. a future
`RVT` or system tag). Pull from the canonical constant.

#### [P3] `_get_reachable_assets` dict-conversion is fragile
**Location:** `src/deriva_ml/dataset/restructure.py:220-221`
**Issue:** `rows = [dict(row._mapping) for row in result]` — accesses
SQLAlchemy's private `_mapping`. Use `.mappings().all()` instead
(public API; same shape).

### src/deriva_ml/dataset/split.py

#### [P1] `split_dataset` is a 590-line function
**Location:** `src/deriva_ml/dataset/split.py:482-1069`
**Issue:** Single function does validation, member listing, size resolution, denormalization, partition computation, dry-run early return, vocab provisioning, dataset hierarchy creation, member assignment, and result construction. The bulk (200 lines) is the docstring; the body is ~250 lines and contains seven inline log-info calls, a 5-section commented-block structure, and a nested loop for member batching.

**Suggested fix:** Extract `_compute_partitions(source_ds, ...) -> (partition_rids, strategy_desc)` and `_create_split_hierarchy(exe, source_dataset_rid, partition_rids, ...) -> SplitResult`. The dry-run path returns before the hierarchy creation, so the split is clean.

#### [P2] `split_dataset` CLI `main()` is 320 lines
**Location:** `src/deriva_ml/dataset/split.py:1077-1402`
**Issue:** Argparse setup (170 lines), dry-run vs real-run branching, two large duplicated parameter blocks (the same `split_dataset` call appears twice with `dry_run=True` and `dry_run=False`), and print formatting.

**Suggested fix:** Build kwargs dict once, pass `dry_run=args.dry_run` through it. Halves the body.

#### [P2] `stratified_split` three-way path hardcodes "Testing"/"Training"/"Validation" partition names
**Location:** `src/deriva_ml/dataset/split.py:338-364`
**Issue:** The three-way branch accesses `partition_sizes["Testing"]` directly by string. Two-way path keys by ordered list. Inconsistency makes it harder to extend (e.g. four-way splits). At minimum document the keying convention; ideally use the same key-by-list pattern in both branches.

#### [P2] `_resolve_sizes` raises `ValueError` but type hints don't say so
**Location:** `src/deriva_ml/dataset/split.py:374-438`
**Issue:** Function raises on negative/zero sizes and on totals exceeding the dataset. Docstring documents Args/Returns/Raises correctly, but the more general issue is that the validation happens during normal execution, not at the `validate_call` boundary. Consider a Pydantic model for the partition spec so the validation surface is one named class.

#### [P3] `_ensure_dataset_types` silently swallows errors
**Location:** `src/deriva_ml/dataset/split.py:472-474`
**Issue:** `if "already exists" not in str(e).lower(): logger.warning(...)` — string-matching on the exception message is brittle. Use exception type instead, or catch a specific `DerivaMLException` subclass.

### src/deriva_ml/dataset/torch_adapter.py and tf_adapter.py

#### [P2] Adapter helpers `_bag_element_is_asset`, `_build_row_lookup`, `_resolve_asset_path` duplicated character-for-character
**Location:** `src/deriva_ml/dataset/torch_adapter.py:160-195` and
`src/deriva_ml/dataset/tf_adapter.py:185-220`
**Issue:** Three helpers (35 lines per file) are verbatim identical
across the two adapter modules. Validation block (element_type in
bag, asset-table sample_loader check) and target-resolution
(`_resolve_targets` + RID-list construction) are also duplicated.

**Suggested fix:** Move helpers and the validation block into
`deriva_ml.dataset.target_resolution` (already shared) or a new
sibling `adapter_common.py`. Each adapter becomes ~30 lines of
framework-specific dataset construction.

#### [P3] Both adapters catch `Exception` in `_bag_element_is_asset`
**Location:** `src/deriva_ml/dataset/torch_adapter.py:167, 173` and same in tf_adapter
**Issue:** Broad `except Exception` silently swallows. Narrow to `(AttributeError, KeyError)` (already listed alongside) and let other exceptions propagate.

### src/deriva_ml/dataset/bag_feature_cache.py

#### [P2] `_ensure_cache_populated` parameter `feat` lacks type annotation and `record_class: type`
**Location:** `src/deriva_ml/dataset/bag_feature_cache.py:121-180`
**Issue:** `feat` is untyped, `record_class: type` is overly broad. The actual types are `Feature` and `type[FeatureRecord]`. Add them.

#### [P3] Cache table stores everything as TEXT — type round-tripping relies on Pydantic
**Location:** `src/deriva_ml/dataset/bag_feature_cache.py:159-163`
**Issue:** Comment "Store everything as TEXT; Pydantic reifies types at FeatureRecord construction time" — fine, but if a feature column has e.g. JSON-stringified array values, the SQLite layer would store the str representation and Pydantic would re-parse. Document the constraint that all FeatureRecord field types must accept str-deserialization, or pin a regression test.

### src/deriva_ml/dataset/target_resolution.py

(No findings — module is well-scoped, single-purpose, well-documented, and tested via `tests/dataset/test_target_resolution.py`.)

### src/deriva_ml/dataset/validation.py

#### [P3] Validation models duplicate the dual-track shape across Dataset/Asset/Workflow
**Location:** `src/deriva_ml/dataset/validation.py:71-150`
**Issue:** `DatasetSpecResult`, `AssetSpecResult`, `WorkflowSpecResult` all carry `valid: bool`, `reasons: list[...]`, `actual_table: str | None`. A base class would tighten the contract. Low-priority — current shape works.

### src/deriva_ml/dataset/__init__.py

(No findings — minimal re-exports, correctly documented.)

---

## Test coverage gaps (cross-module)

1. **`Dataset.to_markdown` / `Dataset.display_markdown` / `Dataset.get_chaise_url`** — no tests.
2. **`DatasetBag.list_tables`** — no tests in `test_bag_api_coverage.py`.
3. **`DatasetBagBuilder.build_bag`** — no direct test (only spec/annotation/aggregate-query coverage).
4. **`Dataset._estimate_csv_bytes`** — pure helper, only exercised through end-to-end `estimate_bag_size` tests.
5. **`BagCache.cached_incomplete`** alias — no test exercises the back-compat alias.
6. **`Dataset.list_workflow_executions`** unscoped semantic — no test asserts that it returns all executions (not just dataset-scoped ones). Risk: future drift toward scoped semantics goes undetected.
7. **`test_prefetch_is_alias_for_cache`** — tests `cache()` (not `prefetch()`); since `prefetch` doesn't exist, the test is meaningless. Either delete the test or implement `prefetch()` as an alias.
8. **`add_dataset_members` cycle path** — no test catches the broken `set(self.dataset_rid)` bug; covered with a real A→B→A cycle test.
9. **`_create_or_advance_dev_row` concurrent-write race** — error path is documented but no test injects an RMT mismatch. Add with a model-side fixture that mutates the row between `dataset_history()` and the update call.
10. **Snapshot-suffix preservation in `_SnapshotAwareCatalogBagBuilder._run_export`** — overrides upstream for issue #114 but no regression test pins the snapshot suffix on the export URL.

## Duplication candidates (cross-module)

1. **`feature_values` selector + grouping pattern** — `dataset.py:591-604` and `dataset_bag.py:637-652`. Extract to `deriva_ml.feature`.
2. **`_bag_element_is_asset`, `_build_row_lookup`, `_resolve_asset_path`** — `torch_adapter.py:160-195` and `tf_adapter.py:185-220` are verbatim parallel.
3. **`DatasetMinid` construction with conditional snapshot suffix** — `bag_download.py:531-537` and `bag_download.py:599-605`. Extract to `_build_dataset_minid(...)`.
4. **`as_torch_dataset` / `as_tf_dataset` docstring prose** — same target-arity, missing-policy, RID-last contract repeated. Extract to module docstring or shared constant.
5. **Retired-API shim shape** — `DatasetBag.fetch_table_features` / `list_feature_values` repeat the same pattern as the DerivaML / Asset retired pairs. Delete all three per workspace convention.

---

## Severity tag legend

- **P0** — blocks release.
- **P1** — should-fix this cycle. Correctness bugs or extraction needed for sustainable maintenance.
- **P2** — nice-to-have. Real findings but the system functions correctly today.
- **P3** — note-only. Style/clarity nudges.
