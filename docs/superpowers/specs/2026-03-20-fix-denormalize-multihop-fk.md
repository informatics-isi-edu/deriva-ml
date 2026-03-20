# Fix Denormalize: Multi-hop FK Joins and Dead Code Cleanup

**Date:** 2026-03-20
**Status:** Draft

## Problem

`denormalize_as_dataframe()` and `denormalize_as_dict()` return null values for all joined tables when those tables are not explicit dataset members. This affects both the bag-side and catalog-side implementations.

**Example:** In EyeAI, dataset 4-411G has 343 Image members. Calling `denormalize(["Image", "Observation", "Subject"])` returns nulls for Observation and Subject columns because neither table has dataset members — they're only reachable via FK chains (`Image → Observation → Subject`).

**Root cause (bag, `dataset_bag.py`):** The public API methods call `_denormalize_from_members()` (line 830), which only joins tables that appear in `list_dataset_members()`. The guard at line 909 (`if other_table_name in members:`) skips FK lookups entirely for non-member tables. A correct SQL-based implementation (`_denormalize()`, line 764) already exists but is orphaned — never called by any public method.

**Root cause (catalog, `dataset.py`):** `_denormalize_datapath()` has the same single-hop, members-only limitation. The guard at line 861 (`if other_table_name in members:`) skips FK lookups for non-member tables.

## Fix

### Bag-side

Switch `denormalize_as_dataframe()` and `denormalize_as_dict()` to use the existing `_denormalize()` method, which builds proper multi-hop SQL JOINs via `_prepare_wide_table()` and `_schema_to_paths()`. The `_denormalize()` method returns a SQLAlchemy `Select` — execute it against the bag's SQLite engine and format results.

`denormalize_as_dict()` should remain a true generator, streaming rows from the SQLite cursor one at a time rather than materializing all results. This preserves memory efficiency for large datasets.

### Catalog-side

Rewrite `_denormalize_datapath()` to follow FK chains between tables using the datapath API, rather than requiring all joined tables to have dataset members. Use the schema model to discover multi-hop FK paths between tables in `include_tables`, traversing intermediate tables (including association tables) automatically.

### Dead code cleanup

Remove internal methods that are no longer called after the fix:

- `_denormalize_from_members()` in `dataset_bag.py` (line 830)
- The equivalent `_denormalize_datapath()` in `dataset.py` will be rewritten, not removed
- Assess whether `_find_relationship_attr()` in `dataset_bag.py` (line 238) is still used after the switch — if only called by `_denormalize_from_members()`, remove it

Audit for other orphaned private methods in both files while making changes.

### Files requiring changes

- `src/deriva_ml/dataset/dataset_bag.py` — switch public methods, remove dead code
- `src/deriva_ml/dataset/dataset.py` — rewrite `_denormalize_datapath()`
- `src/deriva_ml/model/catalog.py` — may need updates to `_prepare_wide_table()` or `_schema_to_paths()`
- `src/deriva_ml/demo_catalog.py` — add new test tables
- `tests/catalog_manager.py` — update `_clear_domain_data` for new tables (Observation, ClinicalRecord, ClinicalRecord_Observation)
- `tests/dataset/test_denormalize.py` — add new test cases, update existing tests affected by schema changes

## Schema Extension for Tests

Add to `demo_catalog.py` to mirror the EyeAI pattern that exposed the bug. **Important: keep the existing `Image → Subject` direct FK.** Add new tables alongside existing ones to avoid breaking existing tests.

New tables:

- **`Observation`** table with FK to `Subject` (RID) and a data column (e.g., `Observation_Date`)
- **`Image` gets an additional FK** to `Observation` (RID, nullable) — the existing `Subject` FK is preserved
- **`ClinicalRecord`** table with data columns (e.g., `Diagnosis`, `Notes`)
- **`ClinicalRecord_Observation`** association table with FK to `ClinicalRecord` (RID) and FK to `Observation` (RID)

This creates the FK graph (new edges shown, existing `Image → Subject` preserved):
```
Image → Subject                    (existing, preserved)
Image → Observation → Subject      (new multi-hop path)
                 ↑
  ClinicalRecord_Observation       (new association, FKs to both sides)
                 ↓
          ClinicalRecord
```

Population: When populating test data, create Observation records linked to existing Subjects, set Image.Observation FKs, and create ClinicalRecords linked via the association table. Observations and ClinicalRecords are NOT added as dataset members — they are only FK-reachable.

**Note on ambiguous paths:** With both `Image → Subject` and `Image → Observation → Subject`, there are now two FK paths from Image to Subject. This is intentional — it exercises test case 5 (ambiguous paths). The ambiguity should be raised by `_table_relationship()` (or `_prepare_wide_table()`) with a clear error listing both paths. The user resolves it by including `Observation` in `include_tables`.

## Test Cases

### Ambiguous path tests (comprehensive)

These tests verify that ambiguous FK paths produce actionable errors and that providing intermediate tables resolves them.

**Test A1: Ambiguous paths produce error with both paths listed**
- Tables: `["Image", "Subject"]`
- Schema: Image has direct FK to Subject AND FK to Observation which has FK to Subject
- Expected: `DerivaMLException` raised
- Error message MUST contain:
  - Both paths: `Image → Subject` (direct) and `Image → Observation → Subject` (multi-hop)
  - Suggestion to include an intermediate table to disambiguate
- Verify: Parse the exception message, confirm both paths are present

**Test A2: Including intermediate table resolves ambiguity**
- Tables: `["Image", "Observation", "Subject"]`
- Same schema as A1
- Expected: No error. All three tables have populated columns.
- Verify: Observation columns are non-null, Subject columns are non-null, FK relationships are correct (`Image.Observation` matches `Observation.RID`, `Observation.Subject` matches `Subject.RID`)

**Test A3: Direct FK still works when no ambiguity exists**
- Tables: `["Image", "Subject"]` but on a dataset where Image has NO Observation FK set (null)
- Expected: Falls back to direct FK path. Subject columns populated via `Image.Subject`.
- This tests that the direct path works when the multi-hop path has no data.

**Test A4: Ambiguous error for association table paths**
- Tables: `["Image", "ClinicalRecord"]`
- Schema: Image → Observation ← ClinicalRecord_Observation → ClinicalRecord
- If multiple paths exist (e.g., a second association table linking Image to ClinicalRecord directly), error should list all paths.
- If only one path exists, no error — the single path is used.

**Test A5: Three-way ambiguity**
- If the schema allows three paths between two tables, the error must list all three.
- This can be constructed by adding a third FK path (if practical with the test schema).

### Multi-hop FK join tests

**Test M1: Non-member FK join (the original bug)**
- Tables: `["Image", "Observation"]`
- Image is a dataset member, Observation is NOT
- Expected: Observation columns are populated (non-null) for every Image that has an Observation FK
- Verify: For each row, if `Image.Observation` is non-null, then `Observation.RID` equals `Image.Observation`

**Test M2: Multi-hop chain through intermediate**
- Tables: `["Image", "Observation", "Subject"]`
- Only Image has dataset members
- Expected: All three tables have populated columns
- Verify: `Image.Observation` = `Observation.RID`, `Observation.Subject` = `Subject.RID`

**Test M3: Association table traversal (M:N)**
- Tables: `["Observation", "ClinicalRecord"]`
- Neither is a dataset member (reached via Image → Observation)
- Join goes through ClinicalRecord_Observation association table
- Expected: ClinicalRecord columns populated for Observations that have linked ClinicalRecords
- Verify: Association table is traversed transparently — its columns do NOT appear in output

**Test M4: Reverse FK direction**
- Tables: `["Observation", "Image"]`
- Image has FK to Observation (so Observation is referenced_by Image)
- Observation listed first but Image is the member
- Expected: Both tables populated. The implementation handles the reversed direction.

**Test M5: Full chain with association**
- Tables: `["Image", "Observation", "ClinicalRecord"]`
- Path: Image → Observation ← ClinicalRecord_Observation → ClinicalRecord
- Expected: All columns populated where data exists

### Data integrity tests

**Test D1: FK value integrity across joins**
- Tables: `["Image", "Observation"]`
- For every row: assert `row["Image.Observation"] == row["Observation.RID"]` (where non-null)

**Test D2: Row count matches dataset members**
- Tables: `["Image", "Observation"]`
- Row count must equal number of Image dataset members (not Observation count)
- No duplication from one-to-many or many-to-many joins at this level

**Test D3: Null handling for missing FK values**
- Some Images may have null Observation FK
- Expected: Those rows have null for all Observation columns (outer join semantics)

**Test D4: No data leakage from non-member records**
- Only records FK-reachable from dataset members should appear
- Observations not linked to any Image in this dataset should NOT appear

### Edge cases

**Test E1: Empty intermediate table**
- Tables: `["Image", "Observation"]` but Observation table has zero rows in bag
- Expected: Image rows returned, all Observation columns null

**Test E2: Single-table regression**
- Tables: `["Subject"]` alone, then `["Image"]` alone
- Expected: Same results as before schema changes — no interference from new tables

**Test E3: All non-member tables**
- Tables: `["Observation", "ClinicalRecord"]` with no dataset members for either
- Expected: Empty result or error (no primary table with members)

### Consistency tests

**Test C1: Bag and catalog produce same results**
- Run same denormalize query on bag and catalog
- Compare sorted RID sets for each table (ignoring column prefix style: dots vs underscores)
- Both should return identical data

**Test C2: Denormalize matches list_dataset_members for member tables**
- Tables: `["Image"]`
- Denormalized RIDs must exactly match `list_dataset_members()["Image"]` RIDs

### Existing tests to verify still pass

All tests in `TestDenormalize`, `TestDenormalizeSchemaGraph`, `TestDenormalizeDataIntegrity`, `TestDenormalizeEdgeCases`, `TestDenormalizeSqlGeneration`, and `TestCatalogDenormalize` must continue to pass. The existing `Image → Subject` direct FK is preserved specifically for this backward compatibility.

## Implementation Notes

- Bag column naming uses dots (`Image.Filename`), catalog uses underscores (`Image_Filename`)
- `_prepare_wide_table()` in `catalog.py` already handles topological sort, cycle detection, and association table traversal
- `_schema_to_paths()` walks the full FK graph from Dataset, follows both `foreign_keys` and `referenced_by`, stops at vocabulary tables
- `_find_relationship_attr()` prefers MANYTOONE direction when multiple ORM relationships exist
- Test case 9 (bag/catalog consistency) should compare sorted sets of RID values rather than DataFrame equality, since row ordering may differ between SQLite and ERMrest
- `catalog_manager.py` must be updated to clear new tables (Observation, ClinicalRecord, ClinicalRecord_Observation) during test reset, respecting FK dependency order
