# Denormalize Refactoring Plan: Join Tree Approach

## Problem Statement

The denormalization subsystem (`_prepare_wide_table` + `_denormalize`) has 3 root causes
for 6 of 61 failing tests, all stemming from the current approach of merging FK paths into
a flat dependency graph and topologically sorting. This destroys path-specific structure
and causes unwanted JOINs.

See: `docs/denormalization-analysis.md` for the complete analysis.

## Root Causes

1. **Topological sort contamination**: Tables not in `include_tables` appear in the join
   order, causing unwanted INNER JOINs that drop rows
2. **Association table elision**: M:N association tables excluded from JOIN chain
   (correctly excluded from columns, incorrectly excluded from joins)
3. **INNER JOIN for nullable FKs**: Should use LEFT OUTER JOIN when FK is nullable

## Solution: JoinTree

Replace the flat topological sort with a tree structure that preserves FK path relationships.

### Data Structure

```python
@dataclass
class JoinNode:
    table: Table              # The table to join
    parent: JoinNode | None   # Parent in the tree (None for root)
    join_conditions: list[tuple[Column, Column]]  # FK column pairs
    join_type: str            # "inner" or "left" (based on FK nullability)
    is_association: bool      # True = join through but exclude columns
    children: list[JoinNode]  # Child nodes
```

### Algorithm

**Phase 1: Path Discovery** — `_schema_to_paths()` (keep as-is)
- DFS from Dataset through FK graph
- No changes needed

**Phase 2: Path Selection** — new `_select_paths()` function
- Filter paths to only those relevant to `include_tables`
- Group by element type
- For each element, find the MINIMAL subtree connecting the element to all `include_tables`
- Association tables are included as interior nodes when they connect two requested tables
- Ambiguity: if multiple distinct minimal subtrees exist, raise error with suggestions

**Phase 3: Build JoinTree** — new `_build_join_tree()` function
- For each element type, build a JoinNode tree from the selected path
- Root = element table (e.g., Image)
- Children = tables reachable via one FK hop that are in the selected path
- Set `join_type="left"` when FK column is nullable
- Set `is_association=True` for association/linking tables

**Phase 4: Generate Joins** — refactored `_denormalize()` and `_denormalize_datapath()`
- Walk JoinTree depth-first to generate SQL JOINs (bag) or BFS lookups (catalog)
- Both paths consume the same JoinTree plan
- Association nodes: joined but columns excluded from SELECT/output

---

## Implementation Steps

### Step 1: Add new test cases for missing FK patterns
**File**: `tests/dataset/test_denormalize.py`, `tests/catalog_manager.py`

Add tests (expected to fail initially):
- Diamond pattern: A→B, A→C, B→D, C→D with `["A", "D"]` → ambiguity error
- Diamond resolved: `["A", "B", "D"]` → no ambiguity
- Feature table denorm: `["Image", "Execution_Image_Quality"]` → feature values joined
- Nullable intermediate: Image→Observation(nullable)→ClinicalRecord with some null Observations
- Association as mandatory intermediate: must JOIN through ClinicalRecord_Observation

**Test**: New tests fail (they define correct behavior not yet implemented)
**Commit**: "Add test cases for missing FK patterns (diamond, feature, nullable intermediate)"

### Step 2: Add JoinNode dataclass and _build_join_tree()
**File**: `src/deriva_ml/model/catalog.py`

- Add `JoinNode` dataclass near top of file
- Add `_build_join_tree(element_table, selected_paths, include_tables)` → `JoinNode`
- The function walks the selected path and builds a tree:
  - Root = element table
  - For each table in the path after the element, add as child of its FK parent
  - Determine join_type from FK nullability
  - Mark association tables

**Test**: Unit test for `_build_join_tree` with known paths
**Commit**: "Add JoinNode dataclass and _build_join_tree() function"

### Step 3: Add _select_paths() for minimal subtree selection
**File**: `src/deriva_ml/model/catalog.py`

- Add `_select_paths(paths_by_element, include_tables)` → `dict[str, list[list[Table]]]`
- For each element type:
  - Find all paths that end at a table in `include_tables`
  - For each endpoint, pick the shortest path (fewest intermediates)
  - If tie: prefer path with all intermediates in `include_tables`
  - If still ambiguous: raise DerivaMLException with both paths listed
- Key rule: association tables that connect two requested tables are automatically included

**Test**: Unit test with known FK graph
**Commit**: "Add _select_paths() for minimal subtree FK path selection"

### Step 4: Refactor _prepare_wide_table() to use JoinTree
**File**: `src/deriva_ml/model/catalog.py`

- Replace the 100+ line ambiguity detection block (lines 654-758) with call to `_select_paths()`
- Replace the topological sort + consecutive-pair join generation with `_build_join_tree()`
- Return type changes: instead of `(join_tables, join_conditions)` per element, return `JoinNode` tree per element
- Column collection walks the JoinTree, skipping `is_association` nodes

**Test**: All existing passing tests still pass (regression check)
**Commit**: "Refactor _prepare_wide_table() to use JoinTree — fixes topological sort contamination"

### Step 5: Update bag _denormalize() to walk JoinTree
**File**: `src/deriva_ml/dataset/dataset_bag.py`

- Replace flat path iteration (line 841-848) with JoinTree walk
- For each JoinNode child, generate `.join()` or `.outerjoin()` based on `join_type`
- Association nodes: joined but columns excluded from SELECT
- This fixes Root Cause 2 (association elision) and Root Cause 3 (INNER→LEFT JOIN)

**Test**: `test_full_chain_with_association` and `test_association_table_single_path` now pass
**Commit**: "Update bag _denormalize() to walk JoinTree — fixes association elision and LEFT JOIN"

### Step 6: Update catalog _denormalize_datapath() to use shared plan
**File**: `src/deriva_ml/dataset/dataset.py`

- Replace the custom BFS with JoinTree consumption
- Walk JoinTree to determine fetch order and FK chain
- LEFT JOIN semantics: if FK is null, fill with None columns

**Test**: `test_catalog_non_member_fk_join`, `test_bag_catalog_multihop_consistency` pass
**Commit**: "Update catalog denormalize to consume shared JoinTree plan"

### Step 7: Modify demo data to expose nullable FK bug
**File**: `src/deriva_ml/demo_catalog.py`

- In `populate_demo_catalog`, create some Images WITHOUT Observation FK set
- This makes the LEFT JOIN vs INNER JOIN distinction testable

**Test**: `test_row_count_matches_members` and `test_null_fk_outer_join` pass correctly
**Commit**: "Add images with null Observation FK to demo data for LEFT JOIN testing"

### Step 8: Run full suite, fix any regressions
- Run all 61+ tests
- Fix any regressions from the refactoring
- Ensure 0 failures

**Test**: Full suite green
**Commit**: "Fix regressions from denormalize refactoring"

### Step 9: Documentation
**File**: `src/deriva_ml/model/catalog.py` (docstrings), `docs/denormalization-design.md` (new)

- Comprehensive docstrings on JoinNode, _build_join_tree, _select_paths, _prepare_wide_table
- Design document explaining:
  - The FK pattern taxonomy (direct, multi-hop, association, composite, diamond, etc.)
  - The JoinTree algorithm
  - How ambiguity detection works
  - How bag and catalog paths share the plan
  - How to add new FK patterns to tests

**Commit**: "Add denormalization design documentation"

---

## Success Criteria

1. All 61 existing tests pass (currently 55/61)
2. New test cases for missing FK patterns pass
3. No code duplication between bag and catalog ambiguity/path logic
4. _prepare_wide_table is < 100 lines (currently ~250)
5. Clear documentation of the algorithm
6. The 3 root causes are eliminated, not worked around

## Risk Mitigation

- **Regression risk**: Run full suite after every step. If any previously passing test fails, investigate before proceeding.
- **Performance risk**: JoinTree walk is O(n) in tree size, same as topological sort. No performance regression expected.
- **Catalog path divergence**: The catalog path currently uses BFS + pre-fetch. Switching to JoinTree consumption may change behavior for edge cases. Test thoroughly.
