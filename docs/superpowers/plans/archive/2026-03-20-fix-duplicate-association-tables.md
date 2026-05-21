# Fix Denormalize Failure with Duplicate Association Tables

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix bag-based denormalization returning empty results when a schema has multiple association tables linking Dataset to the same element table.

**Architecture:** The root cause is in `_prepare_wide_table()` (catalog.py). When two association tables connect Dataset to the same element (e.g., `Image_Dataset` and `Dataset_Image` both connecting Dataset→Image), `_schema_to_paths()` discovers both paths. `_prepare_wide_table()` then merges them into a single graph, producing a join path that includes BOTH association tables. Since only one has data, the INNER JOIN returns 0 rows. The fix: deduplicate paths that reach the same element via different association tables, keeping only one per element. Add a test schema with dual association tables to prevent regression.

**Tech Stack:** Python, SQLAlchemy, pytest, Deriva ERMrest model API

---

## Root Cause Analysis

In the eye-ai catalog:
- `Image_Dataset` (pure association: Image FK + Dataset FK) — has 343 rows
- `Dataset_Image` (association with extra RCB/RMB FKs) — has 0 rows

Both are structurally valid association tables between Dataset and Image. `_schema_to_paths()` finds paths through both. `_prepare_wide_table()` groups paths by element table (position 2), then at lines 644-648 builds a graph from ALL paths for that element. The topological sort produces a merged path containing both association tables. The SQL generator (`_denormalize` in dataset_bag.py) then JOINs through both, and the empty one kills all results.

## Fix Strategy

In `_prepare_wide_table()`, after filtering paths at line 574-578, deduplicate paths that differ only in which association table they use (position 1 in path). Two paths that go `[Dataset, assoc_A, Element, ...]` and `[Dataset, assoc_B, Element, ...]` are redundant — keep one. Use structural detection (`is_association`) rather than naming conventions.

Use `is_association(pure=False)` to catch both pure and impure association tables. First-encountered path wins; both paths lead to the same element data so the choice is arbitrary.

---

### Task 1: Add duplicate association table to test schema

**Files:**
- Modify: `src/deriva_ml/demo_catalog.py:366-474` (create_domain_schema function)
- Modify: `tests/catalog_manager.py` (add new table to reset list)

- [ ] **Step 1: Add a second Image-Dataset association table to demo schema**

In `create_domain_schema()`, after the existing Image asset creation (which creates `Dataset_Image` automatically), add a second association table `Image_Dataset_Legacy` that also links Image to Dataset. This simulates the eye-ai situation where both `Image_Dataset` and `Dataset_Image` exist.

```python
# After ml_instance.apply_catalog_annotations() at line 404, add:

# Create a second (redundant) association table between Dataset and Image.
# This simulates real-world catalogs (e.g., eye-ai) where both Image_Dataset
# and Dataset_Image exist as association tables linking Dataset to Image.
domain_schema = catalog.getCatalogModel().schemas[sname]
domain_schema.create_table(
    TableDef(
        name="Image_Dataset_Legacy",
        columns=[
            ColumnDef("Image", BuiltinType.text, nullok=False),
            ColumnDef("Dataset_Ref", BuiltinType.text, nullok=False),
        ],
        foreign_keys=[
            ForeignKeyDef(
                columns=["Image"],
                referenced_schema=sname,
                referenced_table="Image",
                referenced_columns=["RID"],
            ),
            ForeignKeyDef(
                columns=["Dataset_Ref"],
                referenced_schema="deriva-ml",
                referenced_table="Dataset",
                referenced_columns=["RID"],
            ),
        ],
    )
)
```

- [ ] **Step 2: Add the new table to catalog_manager.py reset list**

In `catalog_manager.py`, find the data deletion list in `reset()` and add `Image_Dataset_Legacy` before the existing Dataset_Image entry (FK dependency order).

- [ ] **Step 3: Run existing tests to confirm schema creation still works**

Run: `uv run pytest tests/dataset/test_denormalize.py::TestDenormalize::test_denormalize_single_table -v`
Expected: PASS (existing test still works with the extra table in schema)

- [ ] **Step 4: Commit**

```bash
git add src/deriva_ml/demo_catalog.py tests/catalog_manager.py
git commit -m "test: add duplicate association table to demo schema for regression testing"
```

---

### Task 2: Write failing test for duplicate association table denormalization

**Files:**
- Modify: `tests/dataset/test_denormalize.py`

- [ ] **Step 1: Write failing test**

Add to `TestDenormalize` class (or create a new class `TestDuplicateAssociationTables`):

```python
class TestDuplicateAssociationTables:
    """Test denormalization when multiple association tables connect Dataset to the same element."""

    def test_denormalize_with_duplicate_association_tables(self, dataset_test, tmp_path):
        """Denormalize should work when schema has two association tables linking Dataset to Image.

        Regression test for eye-ai bug where Image_Dataset and Dataset_Image both exist.
        Only one has data; the other is empty. Merging both into the join path causes
        INNER JOIN to return 0 rows.
        """
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(["Image"])
        # Should return rows even though Image_Dataset_Legacy has no data
        assert len(df) > 0, (
            "denormalize_as_dataframe returned empty DataFrame — "
            "duplicate association table likely caused empty INNER JOIN"
        )

    def test_denormalize_dict_with_duplicate_association_tables(self, dataset_test, tmp_path):
        """denormalize_as_dict should also work with duplicate association tables."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        rows = list(bag.denormalize_as_dict(["Image"]))
        assert len(rows) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/dataset/test_denormalize.py::TestDuplicateAssociationTables -v`
Expected: FAIL — empty DataFrame due to duplicate association table join

- [ ] **Step 3: Commit failing test**

```bash
git add tests/dataset/test_denormalize.py
git commit -m "test: add failing test for duplicate association table denormalization"
```

---

### Task 3: Fix `_prepare_wide_table` to deduplicate association paths

**Files:**
- Modify: `src/deriva_ml/model/catalog.py:564-641` (_prepare_wide_table method)

- [ ] **Step 1: Add path deduplication after path filtering**

After line 578 (the `table_paths` list comprehension) and before line 579 (`paths_by_element`), add deduplication logic:

```python
        # Deduplicate paths that reach the same element via different association tables.
        # In some catalogs (e.g., eye-ai), both Image_Dataset and Dataset_Image exist as
        # association tables linking Dataset to Image. _schema_to_paths() discovers paths
        # through both. If we merge them into a single join graph, the SQL JOINs through
        # both association tables, and the empty one produces 0 rows via INNER JOIN.
        #
        # Fix: For each (element, endpoint) pair, if multiple paths differ only in the
        # association table (path[1]), keep only one path per association table group.
        # First-encountered path wins (both lead to same data).
        deduplicated_paths = []
        seen_element_endpoint = {}  # (element_name, endpoint_name) -> best path
        for path in table_paths:
            if len(path) < 3:
                deduplicated_paths.append(path)
                continue
            assoc_table = path[1]  # Association table between Dataset and element
            element = path[2]     # Element table
            endpoint = path[-1]   # Final table
            key = (element.name, endpoint.name)

            # Check if assoc_table is actually an association table linking Dataset to element
            if not self.is_association(assoc_table, pure=False):
                deduplicated_paths.append(path)
                continue

            if key not in seen_element_endpoint:
                seen_element_endpoint[key] = (path, assoc_table)
                deduplicated_paths.append(path)
            else:
                existing_path, existing_assoc = seen_element_endpoint[key]
                # Check if this is a different association table for the same element
                if existing_assoc.name != assoc_table.name:
                    # Same element via different association table — skip this duplicate path.
                    # Keep the one already chosen (first wins; both lead to same data).
                    pass
                else:
                    # Same association table, different endpoint path — keep both
                    deduplicated_paths.append(path)

        table_paths = deduplicated_paths
```

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest tests/dataset/test_denormalize.py::TestDuplicateAssociationTables -v`
Expected: PASS

- [ ] **Step 3: Run all denormalize tests**

Run: `uv run pytest tests/dataset/test_denormalize.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/deriva_ml/model/catalog.py
git commit -m "fix: deduplicate paths through redundant association tables in _prepare_wide_table"
```

---

### Task 4: Add catalog-side regression test

**Files:**
- Modify: `tests/dataset/test_denormalize.py`

- [ ] **Step 1: Add catalog-side test**

Add to `TestDuplicateAssociationTables`:

```python
    def test_catalog_denormalize_with_duplicate_association(self, catalog_with_datasets, tmp_path):
        """Catalog-side denormalize should also handle duplicate association tables."""
        ml_instance, dataset_description = catalog_with_datasets
        dataset = dataset_description.dataset

        # Test denormalizing with just the Image table
        df = dataset.denormalize_as_dataframe(include_tables=["Image"])
        assert len(df) > 0, (
            "Catalog-side denormalize returned empty DataFrame — "
            "duplicate association table likely caused empty result"
        )
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/dataset/test_denormalize.py::TestDuplicateAssociationTables::test_catalog_denormalize_with_duplicate_association -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/dataset/test_denormalize.py
git commit -m "test: add catalog-side regression test for duplicate association tables"
```

---

### Task 5: Verify fix against eye-ai dataset

This is a manual verification step, not an automated test.

- [ ] **Step 1: Run bag-based denormalization against eye-ai 4-411G**

```python
from deriva_ml import DerivaML
from deriva_ml.dataset.aux_classes import DatasetSpec

ml = DerivaML('www.eye-ai.org', 'eye-ai')
spec = DatasetSpec(rid='4-411G', version='2.10.0', materialize=False)
bag = ml.download_dataset_bag(spec)
df = bag.denormalize_as_dataframe(['Subject', 'Clinical_Records'])
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df.head(3))
```

Expected: Non-empty DataFrame with Subject and Clinical_Records columns.

- [ ] **Step 2: Verify results look correct**

Expected output: DataFrame with ~347 rows (one per clinical record), columns like `Subject.RID`, `Subject.Subject_ID`, `Clinical_Records.Date_of_Encounter`, etc.
