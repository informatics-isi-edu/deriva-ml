# Engineer audit — feature subsystem (v1.37.1)

**Date:** 2026-05-22
**Scope:**

- `src/deriva_ml/feature.py` — `Feature`, `FeatureRecord`, selectors
- `src/deriva_ml/core/mixins/feature.py` — `FeatureMixin` (CRUD + reads)
- `src/deriva_ml/dataset/dataset.py` — `Dataset.feature_values`, `Dataset.find_features`, `Dataset.lookup_feature`
- `src/deriva_ml/dataset/dataset_bag.py` — `DatasetBag.feature_values`, `DatasetBag.find_features`, `DatasetBag.lookup_feature`, `DatasetBag.list_workflow_executions`
- `src/deriva_ml/dataset/bag_feature_cache.py` — bag denormalization read path
- `src/deriva_ml/model/catalog.py` — `DerivaModel.find_features`, `DerivaModel.lookup_feature`
- `src/deriva_ml/model/deriva_ml_bag_view.py` — bag-view `find_features`
- `tests/feature/` — entire test directory

## Summary

Smaller, recently-refactored subsystem with reasonably tight test coverage of the unit-level
selector contract and the three-container symmetry suite. Two real correctness issues stand
out:

- **P0 — `DatasetBag.find_features(None)` will return duplicate `Feature` objects** because it
  bypasses the dedup logic the catalog model added for the no-arg branch.
- **P1 — `FeatureRecord.select_majority_vote(column=None)` indexes a `set`** when auto-detecting
  the single term column, which raises `TypeError` instead of returning a record. No test covers
  the auto-detect happy path, so this latent bug isn't caught.

The rest is largely polish: three parallel implementations of the
"group-by-target-RID-then-apply-selector" pattern, a few docstring/annotation inconsistencies,
and several coverage gaps around the multi-target dedup story, edge cases of the selectors, and
the FK-walk classification in `Feature.__init__`.

Total findings: **21** (3 × P0, 7 × P1, 8 × P2, 3 × P3).

---

## Findings by module

### `src/deriva_ml/feature.py`

#### F-1 [P1] `select_majority_vote(column=None)` auto-detect indexes a set
`feature.py:354-355`
```python
if len(record_cls.feature.term_columns) == 1:
    col = record_cls.feature.term_columns[0].name
```
`Feature.term_columns` is a `set[Column]` (constructed via a set comprehension in
`Feature.__init__` at line 552). Subscripting a set raises `TypeError`. The only existing test
that exercises `column=None` (`test_select_majority_vote_raises_without_column_when_no_metadata`)
hits the *no* metadata branch and never reaches line 355, so this is uncovered. Replace with
`next(iter(record_cls.feature.term_columns)).name` (or sort by name for deterministic output).

#### F-2 [P2] `select_majority_vote._selector` IndexErrors on empty `records`
`feature.py:352` (`record_cls = type(records[0])`) and `feature.py:368`
(`max_count = max(counts.values())`). Both `records[0]` and `max(empty)` raise on
`records == []`. Other selectors return `None` for empty input (see
`test_select_by_workflow_returns_none_on_empty_records`,
`test_select_by_execution_returns_none_on_empty_records`); `select_majority_vote` silently
breaks the symmetry. Guard at the top with `if not records: return None` and update the return
annotation to `FeatureRecord | None`.

#### F-3 [P2] `map_type` return-type annotation drops `bool`
`feature.py:596` annotates the return as
`UnionType | Type[str] | Type[int] | Type[float]`, but line 628 returns `bool`. Docstring
line 611 says "str for all other types" but `bool` is a distinct branch. Add `Type[bool]` to
the annotation and correct the docstring.

#### F-4 [P3] Class-docstring header lists `select_by_execution` as classmethod
`feature.py:21` describes `select_by_execution(execution_rid)` next to
`select_by_workflow(workflow, *, container)`, but the implementation at line 132 is
`@staticmethod`. The asymmetry is harmless but visible — `select_by_execution` should either be
a classmethod for consistency with `select_by_workflow` / `select_majority_vote`, or the docs
shouldn't suggest classmethod symmetry.

#### F-5 [P2] `Feature._model` is an internal attribute with no leading underscore in `__init__` docstring
`feature.py:523` stores `self._model = model`. The `__init__` Args block uses `model` (no
underscore) and the rest of the class doesn't otherwise touch `_model` — it's used inside
`__init__` only. The instance attribute can be dropped entirely (assign into a local, drop the
`self._model = model` line). Saves one strong reference per Feature.

#### F-6 [P1] No test exercises FK classification (`asset_columns` / `term_columns` / `value_columns`) on a real `Feature` from the catalog
The classification logic at `feature.py:546-558` is the heart of `Feature.__init__`. Existing
tests only assert *counts* on demo features (`test_create_feature`,
`test_create_feature_with_non_asset_table_raises`) without ever inspecting which specific
columns landed in which bucket, and `TestFeatureRecord.test_feature_record_column_methods` uses
fully mocked columns that never run `Feature.__init__`. Add a test that:
1. Creates a feature with a known mix of asset/term/value/metadata columns.
2. Inspects each bucket by column name (not just `len`).
3. Verifies the structural FKs (`Execution`, `Feature_Name`, target table FK) are NOT in any
   bucket — this is the `assoc_fkeys` subtraction the comment at line 537 warns against.

---

### `src/deriva_ml/core/mixins/feature.py`

#### F-7 [P1] `select_by_workflow` docstring example passes a `Workflow` object where the signature requires `str`
`core/mixins/feature.py:459-460`:
```python
>>> workflow = ml.lookup_workflow("Glaucoma_Training_v2")  # doctest: +SKIP
>>> sel = FeatureRecord.select_by_workflow(workflow, container=ml)  # doctest: +SKIP
```
`FeatureRecord.select_by_workflow(workflow: str, ...)` calls
`container.list_workflow_executions(workflow)` which is `@validate_call`'d to `str` — passing
the `Workflow` object would fail Pydantic validation. The example is misleading. Use either
`"Glaucoma_Training_v2"` directly (Workflow_Type name) or `workflow.rid`. The same correction
applies to the module-header example in `feature.py:21` if it's ever materialized.

#### F-8 [P1] `feature_values` and `Dataset.feature_values` and `DatasetBag.feature_values` each re-implement the same group-by-RID-and-apply-selector pattern
Three near-identical blocks:
- `core/mixins/feature.py:513-523`
- `dataset/dataset.py:594-604`
- `dataset/dataset_bag.py:641-652`

```python
grouped: dict[str, list[FeatureRecord]] = defaultdict(list)
for rec in records:
    target_rid = getattr(rec, target_col, None)
    if target_rid is not None:
        grouped[target_rid].append(rec)
for group in grouped.values():
    chosen = selector(group)
    if chosen is not None:
        yield chosen
```
Extract once into a module-level helper (e.g. `feature._reduce_with_selector(records, target_col,
selector)`), then call from all three sites. Three places drift independently as more selector
options accumulate. The bag's import of `defaultdict` at line 591 inside the method (lazy) and
the dataset's at the top (eager) is an existing inconsistency that hints at this duplication
being intentional but unaudited.

#### F-9 [P2] `delete_feature` swallows underlying failures behind `False`
`core/mixins/feature.py:269-277`:
```python
try:
    feature = next(f for f in self.model.find_features(table) if f.feature_name == feature_name)
    feature.feature_table.drop()
    return True
except StopIteration:
    return False
```
Only `StopIteration` from the `next(...)` becomes `False`. Any catalog-side error during
`feature.feature_table.drop()` propagates — fine in principle, but the docstring's
`Raises: DerivaMLException` says "If deletion fails due to constraints or permissions" which
implies wrapped errors. Either: (a) wrap the drop in a try-except that converts to
`DerivaMLException` with the table name in context, or (b) reword the docstring to admit that
non-existence is `False` and any other failure is the raw underlying exception (deriva-py's
`HTTPError` etc.).

#### F-10 [P2] `find_features` return-type drift: `Iterable` in `interfaces.py`, `list` in mixin
`interfaces.py:215` declares `find_features(...) -> Iterable[Feature]` (and the same for
Dataset / DatasetBag in interfaces.py:763), but `core/mixins/feature.py:328` and `:355` return
`list[Feature]`. `DatasetBag.find_features` is a generator (yields), so it really is an
`Iterable` — meaning callers cannot assume `len(...)` or random-access on the result.
`tests/feature/test_features.py:221` does `len(list(subject_features)) + len(list(image_features))`,
which works defensively but the rest of the codebase consumes via comprehension — the
inconsistency hasn't bitten yet. Either pin the protocol to `list` (and make the bag eager) or
pin all three to `Iterable` and stop returning a `list` in the mixin. Picking `Iterable` keeps
the bag streaming.

#### F-11 [P2] `lookup_feature` not-found type is documented inconsistently across containers
- `mixins/feature.py:316-317`: `Raises: DerivaMLFeatureNotFound`
- `dataset/dataset.py:621`: `Raises: DerivaMLException` (parent class)
- `dataset/dataset_bag.py:709`: `Raises: DerivaMLException`
- `model/catalog.py:664-666` (actual implementation): raises `DerivaMLFeatureNotFound`

All three containers delegate to the same `model.lookup_feature`, so they all raise the same
exception. Pin the docstrings to `DerivaMLFeatureNotFound` everywhere (sibling of
`DerivaMLException`, so existing `except DerivaMLException` callers still work) — otherwise
users writing precise `except` blocks against the bag/dataset surface won't know they can use
the narrower type.

#### F-12 [P3] `list_workflow_executions` docstring claims "Entries are unique by construction"
`mixins/feature.py:578-579`. True today (each execution runs one workflow), but the live
`Dataset.list_workflow_executions` pass-through (`dataset.py:629`) explicitly documents that
dataset scoping is deferred. If the dataset path is ever changed to union per-member-execution
contributions, uniqueness could break. The "unique by construction" claim is a load-bearing
invariant for `select_by_workflow`'s `set(...)` conversion — pin it with a regression test
asserting `len(rids) == len(set(rids))` on a multi-execution workflow. Test
`test_list_workflow_executions_returns_matching_rids` at `test_features.py:498` does assert
uniqueness, so this is well-covered for the catalog path; the dataset/bag paths are not.

---

### `src/deriva_ml/dataset/dataset.py`

#### F-13 [P1] `Dataset.feature_values` documents `materialize_limit` as forwarding to the upstream, but the upstream's check fires on the **unfiltered** row count
`dataset/dataset.py:579-589`: the dataset filters `raw_in_scope` after fetching from
`self._ml_instance.feature_values(...)`. If `materialize_limit=N` and the feature has 10N rows
in the catalog but only N/2 dataset members, the upstream raises *before* the dataset filter
even sees the rows. The docstring (lines 545-548) says "forwarded to `DerivaML.feature_values`;
raises `DerivaMLMaterializeLimitExceeded` if exceeded" — accurate, but a user reading
"materialize_limit caps the number of rows" naturally expects the post-filter count to be the
gate. Clarify the docstring: the cap applies to the **catalog query**, not the
dataset-filtered result.

---

### `src/deriva_ml/dataset/dataset_bag.py`

#### F-14 [P0] `DatasetBag.find_features(None)` returns duplicate `Feature` objects
`dataset/dataset_bag.py:522-530`:
```python
if table is None:
    for schema in sorted(self.model.schemas):
        for t in sorted(self.model.schemas[schema].tables):
            yield from self.model.find_features(t)
    return
yield from self.model.find_features(table)
```
This deliberately walks every table and calls the `table-specific` branch of
`model.find_features`, which **does not** apply the dedup logic that the no-arg branch added
(`catalog.py:603-645`). Each feature association is reachable from multiple FK targets (target
table, `Execution`, `Feature_Name` vocab, every term vocab the feature references), so a feature
on `Image` with two term columns produces ~4 duplicate `Feature` objects from the bag walk.

This is the same bug `docs/bugs/2026-05-19-find-features-duplicates.md` calls out for the
catalog path; the bag path was not similarly fixed. Replace the no-arg branch with a single
call to `self.model.find_features(None)` (which the bag-view model inherits from `DerivaModel`),
or apply the same `seen_feature_tables` dedup inline.

#### F-15 [P0] No regression test covers `DatasetBag.find_features(None)`
The catalog's dedup test (`test_features.py:198-216`) only exercises `ml_instance.find_features()`.
The three-container symmetry suite (`test_feature_values.py:379-383`) always calls
`find_features(target_table)` with a specific table, so the bag's no-arg branch is uncovered.
Add a `test_bag_find_features_no_arg_dedups` test that asserts uniqueness on
`bag.find_features()` and matches the catalog's invariants in `test_find_features`.

#### F-16 [P2] `DatasetBag.feature_values` materialize_limit semantics differ from the catalog
`dataset/dataset_bag.py:629-635` checks the limit *after* fetching from the bag cache and
applying the in-bag target/execution filters (lines 614-624). The catalog backend checks
*before* record construction (`mixins/feature.py:497-501`). For symmetric API expectations this
should fire before filtering, or the docstring (line 562-565: "mainly for API parity") should
be more explicit that the bag enforces the cap on the **post-filter** count, not the source
row count.

#### F-17 [P2] `DatasetBag.feature_values` swallows `Exception` when probing `get_table_contents(target_col)`
`dataset/dataset_bag.py:607-613`:
```python
try:
    target_rids = {r["RID"] for r in self.model.get_table_contents(target_col)}
except Exception:
    target_rids = None
```
Bare `except Exception` is over-broad. The comment claims "If the target table is missing from
the bag entirely, fall through with the unfiltered set" — but a `KeyError` / `DerivaMLTableNotFound`
is the only legitimate signal here. Tighten to the specific exception class so genuine bag-IO
failures (sqlite locked, corrupt row) aren't silently masked as "target table missing".

#### F-18 [P2] `BagFeatureCache._ensure_cache_populated` does not handle a partially-populated cache
`dataset/bag_feature_cache.py:140-180`: the existence check at line 141 returns the existing
cache table. If a prior call crashed midway through the insert loop at lines 173-177, the next
caller sees the partially-populated table and silently returns short data. Bags are immutable
*after* materialization, but the cache is built on demand inside the running process. Either
build inside a single `BEGIN ... COMMIT` (so a crash leaves the table empty and the existence
check fails) or write a sentinel row that the existence check inspects.

---

### `src/deriva_ml/dataset/bag_feature_cache.py`

#### F-19 [P3] Cache table column types are uniformly `Text`
`bag_feature_cache.py:160-163` stores every field as `Text` ("Pydantic reifies types at
FeatureRecord construction time"). This is fine but loses SQL-side filtering capability — a
future enhancement that lets `bag.feature_values` push `execution_rids` into a SQL `WHERE`
clause (instead of the Python-side filter at `dataset_bag.py:620-624`) would benefit from a
properly-typed `Execution` column. Not blocking for v1.37.1; flag for the cache-design review.

---

### Tests (`tests/feature/`)

#### F-20 [P1] `select_majority_vote` test coverage misses the auto-detect happy path
`tests/feature/test_select_majority_vote.py` covers (a) explicit-column, (b) no-metadata raise,
and (c) explicit-overrides-auto-detect. Missing:
1. Auto-detect from a real `feature_record_class()` with **exactly one** term column —
   exercises `feature.py:354-355` (the set-indexing bug F-1).
2. Auto-detect raise when the feature has **multiple** term columns —
   exercises `feature.py:356-361`.
3. Empty `records` input — exercises F-2.

#### F-21 [P2] `test_feature_record_column_methods` uses mocks rather than a real `Feature`
`tests/feature/test_features.py:60-90` builds the `FeatureRecord.feature` attribute from
`mocker.Mock()` objects. This pins the classmethod *call-through* (does `cls.feature_columns()`
return `cls.feature.feature_columns`?) but not the *construction* — the actual classification
into `asset_columns` / `term_columns` / `value_columns` is never exercised by this test. See
F-6 — the same gap applies in two layers (Feature construction and classmethod readback).

#### F-22 [P3] `test_create_feature` (`test_features.py:106-129`) doesn't assert `value_columns` count for `image_quality_feature`
Line 129 stops after `feature_columns()`. Add `assert len(image_quality_feature.value_columns()) == 0`
for parity with the other two assertions in the same test.

---

## Cross-module coverage gaps

| Gap | P |
|-----|---|
| `DatasetBag.find_features(None)` dedup (F-14/F-15) | P0 |
| `Feature.__init__` FK classification — no test inspects which specific columns land in `asset_columns` / `term_columns` / `value_columns` from a real catalog Feature (F-6, F-21) | P1 |
| `select_majority_vote` auto-detect happy path + multi-term raise (F-20) | P1 |
| Empty-records behavior of `select_majority_vote` (F-2) | P2 |
| `materialize_limit` interaction with `Dataset.feature_values`'s post-fetch member filter — no test verifies what the cap counts (F-13) | P2 |
| `Dataset.list_workflow_executions` uniqueness invariant (F-12) | P3 |

## Cross-module duplication candidates

| Sites | Pattern | P |
|-------|---------|---|
| `mixins/feature.py:513-523`, `dataset/dataset.py:594-604`, `dataset/dataset_bag.py:641-652` | Group-by-target-RID + apply-selector + drop-None survivors (F-8) | P1 |
| `mixins/feature.py:482-483`, `dataset/dataset.py:579-589`, `dataset/dataset_bag.py:620-624` | `execution_rids` empty-list short-circuit + non-empty filter | P2 |
| `dataset/dataset_bag.py:489-530` walking schemas/tables manually vs `catalog.py:625-645` dedup walk — both want "every feature in the model" (F-14) | P0 |
| `mixins/feature.py:560-626` (catalog `list_workflow_executions`) and `dataset/dataset_bag.py:720-781` (bag `list_workflow_executions`) — same two-phase RID-then-type resolution implemented twice against different storage layers | P3 |

## Notes

- `delete_feature_term` does not exist in this codebase. The audit prompt mentioned it; the
  closest is `VocabularyMixin.delete_term` (`mixins/vocabulary.py:395`), which is out of scope
  for this audit.
- `add_features` on `DerivaML` is intentionally retired (raises with pointer to
  `exe.add_features`). Tests in `test_retired_apis.py` pin the retirement contract; nothing to
  flag.
- The selector docstrings are otherwise in good shape and the unit-test coverage for the pure-
  Python selectors (`select_newest`, `select_first`, `select_latest`, `select_by_execution`,
  `select_by_workflow`) is solid — the None-on-no-match contract is pinned in three places
  (test_select_by_execution.py, test_select_by_workflow.py, conftest of the symmetry suite).
