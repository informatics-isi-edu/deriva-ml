# Phase 3 — Denormalize Planner Extraction Plan

**Status:** Plan only. Source files unchanged. This doc captures the
*precise* mechanical steps so the move can be done in one focused PR
without surprises.

## Motivation

`src/deriva_ml/model/catalog.py` is **2193 LoC** with two distinct
responsibilities:

1. **Catalog introspection** (~1300 LoC) — `DerivaModel.__init__`,
   `name_to_table`, `is_vocabulary`, `is_asset`, `is_association`,
   `find_assets`, `find_vocabularies`, `find_features`,
   `lookup_feature`, `asset_metadata_columns`, schema helpers,
   `apply`, `is_dataset_rid`, `list_dataset_element_types`,
   `is_topological_association`. **Fan-out: every mixin.**
2. **Denormalization planner** (~880 LoC) — `JoinNode` dataclass +
   `denormalize_column_name` + 6 `_`-prefixed planner methods + 3
   reachability primitives. **Fan-out: `local_db/denormalize*.py` +
   one line in `dataset/dataset_bag.py` + a couple in
   `core/mixins/dataset.py`.**

The two fan-outs are fundamentally different — catalog introspection
is touched by every layer; the planner is a specialty subsystem with
a narrow consumer set. Phase 1 §6.1 + Phase 2 model audit §5.2 both
flagged this; the audit's recommendation stands after a fresh read.

## Exact LoC accounting

Lines in `catalog.py` that move to the new file:

| Range | Symbol | Notes |
|---:|---|---|
| 51–93 | `class JoinNode` + `walk()` + `walk_edges()` | dataclass |
| 96–119 | `def denormalize_column_name` | module-level helper |
| 822–876 | `is_topological_association` | reachability primitive |
| 878–928 | `_fk_neighbors` | helper for planner |
| 929–1167 | `_build_join_tree` (+ nested `_intermediates_covered`) | main planner |
| 1169–1224 | `_downstream_fk_sources` | reachability primitive |
| 1225–1320 | `_outbound_reachable` | reachability composer |
| 1321–1369 | `_find_sinks` | sink finder |
| 1370–1451 | `_determine_row_per` | row-per resolver |
| 1452–1516 | `_enumerate_paths` | path enumerator |
| 1517–1720 | `_find_path_ambiguities` (+ nested `_edge_direction`, `_is_downstream_chain`, `_is_signaled`) | ambiguity detector |
| 1722–1948 | `_prepare_wide_table` (+ nested `_is_standard_assoc`) | top-level entry |
| 1949–2000 | `_table_relationship` | column-pair helper |
| 2001–2192 | `_schema_to_paths` (+ nested `is_nested_dataset_loopback`) | path DFS |

Total: ~1100 LoC move (audit said ~880; the discrepancy is the
`is_topological_association`, `_table_relationship`, and
`_schema_to_paths` helpers that the audit missed).

## Target layout

```
src/deriva_ml/local_db/
    denormalize.py                 # existing — top-level Denormalizer
    denormalize_impl.py            # existing — execution path
    denormalize_planner.py         # NEW — extracted planner + helpers
    denormalizer.py                # existing — main API
```

Or alternatively, sibling to `model/catalog.py`:

```
src/deriva_ml/model/
    catalog.py                     # ~1100 LoC — pure introspection
    denormalize_planner.py         # NEW — extracted planner + helpers
```

The audit prefers the `model/` location because the planner is
`DerivaModel`-anchored (takes a model, walks its schemas). I agree;
keeping the import path `deriva_ml.model.denormalize_planner` is the
right shape.

## API shape

The planner methods are currently `DerivaModel` instance methods.
Two options for the extraction:

### Option A: Free functions taking a model

```python
def prepare_wide_table(model: DerivaModel, ...): ...
def build_join_tree(model: DerivaModel, ...): ...
```

Pros: simplest signature change; tests easy.
Cons: every existing call site changes from `model._prepare_wide_table(...)`
to `prepare_wide_table(model, ...)`.

### Option B: `DenormalizePlanner` class wrapping a model

```python
class DenormalizePlanner:
    def __init__(self, model: DerivaModel): ...
    def prepare_wide_table(self, ...): ...
    def build_join_tree(self, ...): ...
```

Pros: state can be cached (e.g., reachability sets); call sites
become `planner.prepare_wide_table(...)`; future enrichments live
on the class.
Cons: callers must construct the planner; risk of holding onto a
stale model.

**Recommendation: Option B.** The audit prefers it too (line 569 of
the audit doc). The class can be instantiated once and reused.

`DerivaModel` keeps a thin shim that constructs a planner on demand
for backward compat, deletable in a follow-up:

```python
class DerivaModel:
    @property
    def _planner(self) -> DenormalizePlanner:
        if not hasattr(self, "_cached_planner"):
            self._cached_planner = DenormalizePlanner(self)
        return self._cached_planner
```

## Callsite migration

Every `model._<method>(...)` call inside the planner methods becomes
`self._<method>(...)` (planner instance methods).

Every external `model._<planner-method>(...)` call site rewrites to
`model._planner.<method>(...)` initially, then deletes the underscore
in a follow-up cleanup pass.

External call sites (from the search above):

- `local_db/denormalize.py:274` — main entry
- `local_db/denormalizer.py:525, 621, 625, 638, 654, 828, 832, 856, 920, 928, 1147, 1153, 1182, 1250` — 14 sites
- `core/mixins/dataset.py:449` — 1 site
- `dataset/dataset_bag.py:381` — 1 site (calls `_schema_to_paths`)

Total: **~17 external call sites** to rewrite. All `s/model\._foo/model._planner.foo/` mechanical.

## Test surface

Tests that import or reference planner methods directly:

```bash
grep -rn "_build_join_tree\|_prepare_wide_table\|_enumerate_paths\|_find_path_ambiguities\|_determine_row_per\|_find_sinks\|_downstream_fk_sources\|_outbound_reachable\|_schema_to_paths\|denormalize_column_name\|JoinNode" tests/ --include="*.py"
```

Run this before the move to enumerate them; rewrite each with the
new import path.

## Sequencing

1. **Create `model/denormalize_planner.py`** with the new
   `DenormalizePlanner` class and the moved helpers + dataclass.
   Copy-not-cut so the module compiles and tests still pass against
   the old impl.
2. **Add the `_planner` property shim** to `DerivaModel`.
3. **Migrate external callsites** one by one (or with sed), running
   the targeted test files after each batch.
4. **Cut the old methods out of `catalog.py`** once all callsites
   are migrated.
5. **Drop the underscore from method names** in a follow-up cleanup
   PR (they were prefix-`_` because they were DerivaModel
   internals; on the new class they're the public API).

## Risk

**Medium-high.** The planner is well-tested but the move touches a
load-bearing import that 14+ sites depend on. The compile-fail mode
is loud (ImportError, AttributeError) but a stray reference is
easy to miss. Run the full integration suite between steps 3 and 4.

## Estimated effort

- Step 1 (create new module): 1-2 hours.
- Step 2 (shim): 5 minutes.
- Step 3 (migrate callsites): 1-2 hours.
- Step 4 (cut from catalog.py): 30 minutes.
- Step 5 (rename): separate small PR.

Total: **half a day of focused work**, single PR for steps 1-4.

## Out of scope

- Renaming `_is_association_table` (Lens 4 of the audit) — a
  separate cleanup, can ride along or stay independent.
- `dataset.py` / `dataset_bag.py` structural splits (audit's other
  Phase 3 candidates) — independent work, no dependency on this.
- Performance / async-loop alignment (`estimate_bag_size`) —
  independent, can be done after.
