# Fix Denormalize: Multi-hop FK Joins Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix denormalize to follow multi-hop FK chains (e.g., Image → Observation → Subject) instead of returning nulls for non-member tables, and add comprehensive tests for ambiguous path detection and resolution.

**Architecture:** The bag-side fix switches the public API to use the existing `_denormalize()` SQL method which already builds correct multi-hop JOINs. The catalog-side fix rewrites `_denormalize_datapath()` to follow FK chains via datapath, supporting multi-hop chains by walking intermediate tables. Both sides get ambiguous path detection with actionable error messages. The test schema is extended with Observation, ClinicalRecord, and ClinicalRecord_Observation tables to exercise multi-hop patterns.

**Tech Stack:** Python, SQLAlchemy ORM, ERMrest datapath API, pytest

**Spec:** `docs/superpowers/specs/2026-03-20-fix-denormalize-multihop-fk.md`

---

### Task 1: Extend test schema with multi-hop FK tables

**Files:**
- Modify: `src/deriva_ml/demo_catalog.py:323-361` (`create_domain_schema`)
- Modify: `src/deriva_ml/demo_catalog.py:52-69` (`populate_demo_catalog`)
- Modify: `tests/catalog_manager.py:123-248` (`reset` method)

- [ ] **Step 1: Add Observation, ClinicalRecord, ClinicalRecord_Observation tables to `create_domain_schema`**

After the existing `ml_instance.create_asset("Image", ...)` and `ml_instance.apply_catalog_annotations()` calls, add:

```python
from deriva.core.typed import ForeignKeyDef

# Create Observation table with FK to Subject
domain_schema.create_table(
    TableDef(
        name="Observation",
        columns=[
            ColumnDef("Observation_Date", BuiltinType.date),
            ColumnDef("Subject", BuiltinType.text, nullok=False),
        ],
        foreign_keys=[
            ForeignKeyDef(
                columns=["Subject"],
                referenced_schema=sname,
                referenced_table="Subject",
                referenced_columns=["RID"],
            )
        ],
    )
)

# Add FK from Image to Observation (nullable, keeps existing Image → Subject FK)
# Refresh model to pick up the newly created Observation table
model = catalog.getCatalogModel()
domain_schema = model.schemas[sname]
image_table = domain_schema.tables["Image"]
image_table.create_column(ColumnDef("Observation", BuiltinType.text, nullok=True))
image_table.create_fkey(
    ForeignKeyDef(
        columns=["Observation"],
        referenced_schema=sname,
        referenced_table="Observation",
        referenced_columns=["RID"],
    )
)

# Create ClinicalRecord table
domain_schema.create_table(
    TableDef(
        name="ClinicalRecord",
        columns=[
            ColumnDef("Diagnosis", BuiltinType.text),
            ColumnDef("Notes", BuiltinType.text, nullok=True),
        ],
    )
)

# Create ClinicalRecord_Observation association table
domain_schema.create_table(
    TableDef(
        name="ClinicalRecord_Observation",
        columns=[
            ColumnDef("ClinicalRecord", BuiltinType.text, nullok=False),
            ColumnDef("Observation_Ref", BuiltinType.text, nullok=False),
        ],
        foreign_keys=[
            ForeignKeyDef(
                columns=["ClinicalRecord"],
                referenced_schema=sname,
                referenced_table="ClinicalRecord",
                referenced_columns=["RID"],
            ),
            ForeignKeyDef(
                columns=["Observation_Ref"],
                referenced_schema=sname,
                referenced_table="Observation",
                referenced_columns=["RID"],
            ),
        ],
    )
)
```

Note: `ForeignKeyDef` is imported from `deriva.core.typed` and uses **keyword arguments** (`columns=`, `referenced_schema=`, `referenced_table=`, `referenced_columns=`). See `src/deriva_ml/schema/create_schema.py:115-127` for working examples. The column name `Observation_Ref` avoids collision with the `Observation` table name in path builder.

- [ ] **Step 2: Populate test data for new tables in `populate_demo_catalog`**

After the existing image creation loop, add:

```python
# Create Observations (one per Subject)
pb = ml_instance.domain_path()
observation_records = []
for s in ss:
    observation_records.append({
        "Subject": s["RID"],
        "Observation_Date": datetime.now().date().isoformat(),
    })
observation_table = pb.tables["Observation"]
observations = list(observation_table.insert(observation_records))

# Link Images to Observations (update Image.Observation FK)
image_table = pb.tables["Image"]
all_images = list(image_table.path.entities().fetch())
for img, obs in zip(all_images, observations):
    image_table.path.filter(image_table.RID == img["RID"]).update([{"RID": img["RID"], "Observation": obs["RID"]}])

# Create ClinicalRecords
clinical_records = []
for i, obs in enumerate(observations):
    clinical_records.append({
        "Diagnosis": f"Diagnosis_{i}",
        "Notes": f"Notes for observation {obs['RID']}",
    })
cr_table = pb.tables["ClinicalRecord"]
crs = list(cr_table.insert(clinical_records))

# Create ClinicalRecord_Observation associations
assoc_records = []
for cr, obs in zip(crs, observations):
    assoc_records.append({
        "ClinicalRecord": cr["RID"],
        "Observation_Ref": obs["RID"],
    })
cr_obs_table = pb.tables["ClinicalRecord_Observation"]
cr_obs_table.insert(assoc_records)
```

Note: The datapath API uses `path.filter().update()` for updates, not `table.update()` directly. Check the existing code patterns in `demo_catalog.py` for the correct update syntax and adjust accordingly.

- [ ] **Step 3: Update `catalog_manager.py` reset method for new tables**

Add the new tables to the appropriate reset lists in `reset()`.

In the `domain_assoc_tables` list (line 156, after `"Image_Subject"`):
```python
"ClinicalRecord_Observation",
```

Replace the data tables deletion (line 174, `for t in ["Image", "Subject"]:`) with:
```python
# Clear data tables in dependency order (FK children before parents)
for t in ["ClinicalRecord_Observation", "ClinicalRecord", "Image", "Observation", "Subject"]:
    self._delete_table_data(domain_path, t)
```

Add the new tables to the test-specific drop lists (after line 240, test_tables):
```python
# New schema extension tables for multi-hop FK testing
new_assoc_tables = [
    "ClinicalRecord_Observation",
]
for t in new_assoc_tables:
    self._drop_table_if_exists(self.domain_schema, t)

new_data_tables = ["ClinicalRecord", "Observation"]
for t in new_data_tables:
    self._drop_table_if_exists(self.domain_schema, t)
```

Note: `Observation` must be dropped after `ClinicalRecord_Observation` (FK dependency) and after `Image` (Image has FK to Observation). `Image` is already dropped earlier in the reset sequence.

- [ ] **Step 4: Proactively mark existing ambiguous-path tests as xfail**

The addition of the Image→Observation FK creates an **ambiguous path** from Image to Subject (direct FK AND Image→Observation→Subject). Existing tests that call `denormalize(["Image", "Subject"])` will now fail because `_table_relationship()` finds 2 FK paths.

**Before running tests**, find all existing tests that use `include_tables=["Image", "Subject"]` and mark them:

```python
@pytest.mark.xfail(reason="Ambiguous Image→Subject path after schema extension — resolved in Task 4", strict=False)
```

Use `xfail` (not `skip`) so they still run and we can see when they start passing again. Track which tests were marked so Task 4 can remove the markers.

Run: `cd /Users/carl/GitHub/deriva-ml && grep -n '"Image", "Subject"' tests/dataset/test_denormalize.py`

Add `@pytest.mark.xfail(...)` to each matching test method.

- [ ] **Step 5: Run existing tests to verify schema extension doesn't break non-ambiguous tests**

Run: `cd /Users/carl/GitHub/deriva-ml && uv run pytest tests/dataset/test_denormalize.py -v --timeout=120`

Expected: All non-xfail tests pass. xfail tests may fail (expected) or xpass (also fine).

- [ ] **Step 6: Commit**

```bash
git add src/deriva_ml/demo_catalog.py tests/catalog_manager.py tests/dataset/test_denormalize.py
git commit -m "feat: add Observation, ClinicalRecord tables to test schema for multi-hop FK testing"
```

---

### Task 2: Write failing tests for multi-hop FK denormalization (bag-side)

**Files:**
- Modify: `tests/dataset/test_denormalize.py`

These tests target the bag-side `denormalize_as_dataframe()` and `denormalize_as_dict()`. All tests should FAIL initially because the current implementation (`_denormalize_from_members`) doesn't handle non-member tables.

- [ ] **Step 1: Add `TestMultiHopDenormalize` class with all multi-hop and data integrity tests**

Add after the existing `TestCatalogDenormalize` class:

```python
class TestMultiHopDenormalize:
    """Test multi-hop FK joins in bag denormalization.

    These tests use the extended schema: Image → Observation → Subject
    and ClinicalRecord_Observation linking ClinicalRecord ↔ Observation.
    Image is the only dataset member; other tables are FK-reachable.
    """

    def test_non_member_fk_join(self, dataset_test, tmp_path):
        """M1: Join to a table that is not a dataset member via FK.

        Image (member) has FK to Observation (non-member).
        Observation columns should be populated, not null.
        """
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(include_tables=["Image", "Observation"])

        # Should have rows (equal to Image member count)
        assert len(df) > 0, "Expected rows from denormalization"

        # Observation columns should exist
        obs_columns = [c for c in df.columns if c.startswith("Observation.")]
        assert len(obs_columns) > 0, "Expected Observation columns"

        # At least some Observation values should be non-null
        obs_rid_col = "Observation.RID"
        assert obs_rid_col in df.columns, f"Expected {obs_rid_col} column"
        non_null_count = df[obs_rid_col].notna().sum()
        assert non_null_count > 0, (
            "Observation.RID should be populated for Images with Observation FK. "
            "Got all nulls — FK join to non-member table is not working."
        )

    def test_multihop_chain(self, dataset_test, tmp_path):
        """M2: Multi-hop chain Image → Observation → Subject, all included."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(
            include_tables=["Image", "Observation", "Subject"]
        )

        assert len(df) > 0
        # All three table prefixes should have columns
        for prefix in ["Image.", "Observation.", "Subject."]:
            cols = [c for c in df.columns if c.startswith(prefix)]
            assert len(cols) > 0, f"Expected columns with prefix {prefix}"

        # Subject should be populated (reached via Observation)
        subject_rid_col = "Subject.RID"
        if subject_rid_col in df.columns:
            non_null = df[subject_rid_col].notna().sum()
            assert non_null > 0, "Subject.RID should be populated via multi-hop FK chain"

    def test_association_table_join(self, dataset_test, tmp_path):
        """M3: Join through association table (M:N).

        Path: Image → Observation ← ClinicalRecord_Observation → ClinicalRecord
        """
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(
            include_tables=["Image", "Observation", "ClinicalRecord"]
        )

        assert len(df) > 0
        # ClinicalRecord columns should be present
        cr_cols = [c for c in df.columns if c.startswith("ClinicalRecord.")]
        assert len(cr_cols) > 0, "Expected ClinicalRecord columns"

        # Association table columns should NOT be in output
        assoc_cols = [c for c in df.columns if c.startswith("ClinicalRecord_Observation.")]
        assert len(assoc_cols) == 0, "Association table columns should not appear in output"

    def test_reverse_fk_direction(self, dataset_test, tmp_path):
        """M4: Reverse FK direction — Observation listed first, Image is the member.

        Image has FK to Observation (so Observation is referenced_by Image).
        Observation listed first but Image is the member.
        """
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(
            include_tables=["Observation", "Image"]
        )

        assert len(df) > 0
        # Both tables should have columns
        obs_cols = [c for c in df.columns if c.startswith("Observation.")]
        img_cols = [c for c in df.columns if c.startswith("Image.")]
        assert len(obs_cols) > 0, "Expected Observation columns"
        assert len(img_cols) > 0, "Expected Image columns"

    def test_full_chain_with_association(self, dataset_test, tmp_path):
        """M5: Full chain Image → Observation ← ClinicalRecord_Observation → ClinicalRecord."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(
            include_tables=["Image", "Observation", "ClinicalRecord"]
        )

        assert len(df) > 0
        # All requested tables should have columns
        for prefix in ["Image.", "Observation.", "ClinicalRecord."]:
            cols = [c for c in df.columns if c.startswith(prefix)]
            assert len(cols) > 0, f"Expected columns with prefix {prefix}"

        # ClinicalRecord should be populated where data exists
        cr_rid_col = "ClinicalRecord.RID"
        if cr_rid_col in df.columns:
            non_null = df[cr_rid_col].notna().sum()
            assert non_null > 0, "ClinicalRecord.RID should be populated via association table"

    def test_fk_value_integrity(self, dataset_test, tmp_path):
        """D1: FK values are correct in joined rows."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(include_tables=["Image", "Observation"])

        # For rows where both sides are non-null, FK must match PK
        if "Image.Observation" in df.columns and "Observation.RID" in df.columns:
            valid = df.dropna(subset=["Image.Observation", "Observation.RID"])
            for _, row in valid.iterrows():
                assert row["Image.Observation"] == row["Observation.RID"], (
                    f"FK mismatch: Image.Observation={row['Image.Observation']} "
                    f"!= Observation.RID={row['Observation.RID']}"
                )

    def test_row_count_matches_members(self, dataset_test, tmp_path):
        """D2: Row count equals dataset member count, no duplication."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        members = bag.list_dataset_members(recurse=True)
        image_count = len(members.get("Image", []))

        df = bag.denormalize_as_dataframe(include_tables=["Image", "Observation"])
        assert len(df) == image_count, (
            f"Row count ({len(df)}) should match Image member count ({image_count})"
        )

    def test_null_fk_outer_join(self, dataset_test, tmp_path):
        """D3: Images with null Observation FK get null Observation columns."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(include_tables=["Image", "Observation"])

        # Rows where Image.Observation is null should have null Observation.RID
        if "Image.Observation" in df.columns and "Observation.RID" in df.columns:
            null_obs_images = df[df["Image.Observation"].isna()]
            if len(null_obs_images) > 0:
                assert null_obs_images["Observation.RID"].isna().all(), (
                    "Images with null Observation FK should have null Observation columns"
                )

    def test_no_data_leakage(self, dataset_test, tmp_path):
        """D4: No data leakage from non-member records.

        Only records FK-reachable from dataset members should appear.
        Observations not linked to any Image in this dataset should NOT appear.
        """
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(include_tables=["Image", "Observation"])

        # Get the set of Observation RIDs referenced by Image members
        if "Image.Observation" in df.columns:
            expected_obs_rids = set(df["Image.Observation"].dropna())

            # All Observation RIDs in the output should be in the expected set
            if "Observation.RID" in df.columns:
                actual_obs_rids = set(df["Observation.RID"].dropna())
                assert actual_obs_rids.issubset(expected_obs_rids), (
                    f"Leaked Observation RIDs: {actual_obs_rids - expected_obs_rids}"
                )

    def test_single_table_regression(self, dataset_test, tmp_path):
        """E2: Single-table denormalization unchanged by schema additions."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Single table should still work
        df_subject = bag.denormalize_as_dataframe(include_tables=["Subject"])
        assert isinstance(df_subject, pd.DataFrame)
        assert len(df_subject) > 0

        df_image = bag.denormalize_as_dataframe(include_tables=["Image"])
        assert isinstance(df_image, pd.DataFrame)
        assert len(df_image) > 0

    def test_empty_intermediate_table(self, dataset_test, tmp_path):
        """E1: Empty intermediate table — Image rows returned, all Observation columns null.

        This test requires a dataset where no Image has an Observation FK set.
        If the test data populates all Images with Observations, this test verifies
        the behavior with a hypothetical empty case by checking that nulls are
        handled correctly (same as D3 but checking the extreme case).
        """
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Even if Observation table has data, the query should not fail
        df = bag.denormalize_as_dataframe(include_tables=["Image", "Observation"])
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_all_non_member_tables(self, dataset_test, tmp_path):
        """E3: All non-member tables — should return empty or error.

        Neither Observation nor ClinicalRecord has dataset members.
        """
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(
            include_tables=["Observation", "ClinicalRecord"]
        )
        # Should return empty DataFrame (no primary table with members)
        assert len(df) == 0, (
            "Should return empty result when no included table has dataset members"
        )

    def test_denormalize_matches_members(self, dataset_test, tmp_path):
        """C2: Denormalized RIDs match list_dataset_members for member tables."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        members = bag.list_dataset_members(recurse=True)
        member_rids = {m["RID"] for m in members.get("Image", [])}

        df = bag.denormalize_as_dataframe(include_tables=["Image"])
        denorm_rids = set(df["Image.RID"].dropna())

        assert member_rids == denorm_rids, (
            f"Denormalized RIDs should match list_dataset_members. "
            f"Missing: {member_rids - denorm_rids}, Extra: {denorm_rids - member_rids}"
        )
```

- [ ] **Step 2: Add `TestAmbiguousPaths` class**

```python
class TestAmbiguousPaths:
    """Test ambiguous FK path detection and resolution.

    The test schema has two paths from Image to Subject:
    1. Direct: Image → Subject (via Image.Subject FK)
    2. Multi-hop: Image → Observation → Subject

    Requesting ["Image", "Subject"] should raise an error listing both paths.
    Including Observation should disambiguate.
    """

    def test_ambiguous_paths_raises_error(self, dataset_test, tmp_path):
        """A1: Ambiguous paths produce DerivaMLException with both paths listed."""
        from deriva_ml.core.exceptions import DerivaMLException

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        with pytest.raises(DerivaMLException) as exc_info:
            bag.denormalize_as_dataframe(include_tables=["Image", "Subject"])

        error_msg = str(exc_info.value)
        # Error must mention both tables and indicate ambiguity
        assert "Subject" in error_msg, "Error should mention the ambiguous target table"
        assert "Image" in error_msg, "Error should mention the source table"
        assert "ambiguous" in error_msg.lower() or "multiple" in error_msg.lower(), (
            f"Error should indicate ambiguity. Got: {error_msg}"
        )

    def test_ambiguous_error_lists_paths(self, dataset_test, tmp_path):
        """A1b: Error message contains enough info to resolve the ambiguity."""
        from deriva_ml.core.exceptions import DerivaMLException

        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        with pytest.raises(DerivaMLException) as exc_info:
            bag.denormalize_as_dataframe(include_tables=["Image", "Subject"])

        error_msg = str(exc_info.value)
        # Must mention Observation as the intermediate table
        assert "Observation" in error_msg, (
            f"Error should mention intermediate table 'Observation' so user knows "
            f"to include it for disambiguation. Got: {error_msg}"
        )

    def test_including_intermediate_resolves_ambiguity(self, dataset_test, tmp_path):
        """A2: Including Observation resolves the Image→Subject ambiguity."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # This should NOT raise — Observation disambiguates the path
        df = bag.denormalize_as_dataframe(
            include_tables=["Image", "Observation", "Subject"]
        )

        assert len(df) > 0, "Should return rows after disambiguation"

        # All three tables should have populated columns
        for prefix in ["Image.", "Observation.", "Subject."]:
            cols = [c for c in df.columns if c.startswith(prefix)]
            assert len(cols) > 0, f"Expected columns with prefix {prefix}"

        # FK chain should be correct
        if all(c in df.columns for c in ["Image.Observation", "Observation.RID",
                                          "Observation.Subject", "Subject.RID"]):
            valid = df.dropna(subset=["Image.Observation", "Observation.RID",
                                       "Observation.Subject", "Subject.RID"])
            for _, row in valid.iterrows():
                assert row["Image.Observation"] == row["Observation.RID"], (
                    "Image.Observation should match Observation.RID"
                )
                assert row["Observation.Subject"] == row["Subject.RID"], (
                    "Observation.Subject should match Subject.RID"
                )

    def test_disambiguation_produces_correct_data(self, dataset_test, tmp_path):
        """A2b: Disambiguated path returns correct Subject for each Image."""
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        df = bag.denormalize_as_dataframe(
            include_tables=["Image", "Observation", "Subject"]
        )

        # Cross-check: the Subject.Name values should be valid test data
        if all(c in df.columns for c in ["Subject.RID", "Subject.Name"]):
            valid = df.dropna(subset=["Subject.RID", "Subject.Name"])
            assert len(valid) > 0, "Should have rows with Subject data"
            for _, row in valid.iterrows():
                assert row["Subject.Name"].startswith("Thing"), (
                    f"Unexpected Subject.Name: {row['Subject.Name']}"
                )

    def test_direct_fk_no_ambiguity(self, dataset_test, tmp_path):
        """A3: Direct FK still works when no ambiguity exists.

        Test with ["Image", "Observation"] — there's only one FK path
        (Image.Observation → Observation.RID), so no ambiguity should be raised.
        """
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Should not raise — single path from Image to Observation
        df = bag.denormalize_as_dataframe(include_tables=["Image", "Observation"])
        assert len(df) > 0
        # Observation should be populated
        if "Observation.RID" in df.columns:
            non_null = df["Observation.RID"].notna().sum()
            assert non_null > 0, "Observation.RID should be populated via direct FK"

    def test_association_table_single_path(self, dataset_test, tmp_path):
        """A4: Association table path — if only one path exists, no error.

        Path: Image → Observation ← ClinicalRecord_Observation → ClinicalRecord
        Only one path exists between Image and ClinicalRecord (through Observation
        and the association table), so this should NOT raise an ambiguity error.
        """
        dataset_description = dataset_test.dataset_description
        current_version = dataset_description.dataset.current_version
        bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

        # Should work without error — single path through association table
        df = bag.denormalize_as_dataframe(
            include_tables=["Image", "Observation", "ClinicalRecord"]
        )
        assert len(df) > 0
```

Note: Test A5 (three-way ambiguity) is omitted as the current test schema only supports two-way ambiguity (Image→Subject direct and Image→Observation→Subject). Adding a third path would require additional schema changes that don't serve the core fix. The A5 case can be added later if needed.

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/deriva-ml && uv run pytest tests/dataset/test_denormalize.py::TestMultiHopDenormalize -v --timeout=120`

Expected: Tests FAIL because `_denormalize_from_members()` doesn't handle non-member tables.

Run: `cd /Users/carl/GitHub/deriva-ml && uv run pytest tests/dataset/test_denormalize.py::TestAmbiguousPaths -v --timeout=120`

Expected: Tests FAIL (ambiguous path test may fail with wrong error type or no error, disambiguation test may return nulls).

- [ ] **Step 4: Commit failing tests**

```bash
git add tests/dataset/test_denormalize.py
git commit -m "test: add failing tests for multi-hop FK denormalization and ambiguous paths"
```

---

### Task 3: Fix bag-side denormalize to use SQL-based implementation

**Files:**
- Modify: `src/deriva_ml/dataset/dataset_bag.py:992` (`denormalize_as_dataframe`)
- Modify: `src/deriva_ml/dataset/dataset_bag.py:1053` (`denormalize_as_dict`)

The existing `_denormalize()` (line 764) builds correct multi-hop SQL JOINs. Switch the public methods to use it.

- [ ] **Step 1: Update `denormalize_as_dataframe` to use `_denormalize()`**

Replace line 992 (`rows = list(self._denormalize_from_members(include_tables=include_tables))`) with:

```python
sql_stmt = self._denormalize(include_tables=include_tables)
with Session(self.engine) as session:
    result = session.execute(sql_stmt)
    rows = [dict(row._mapping) for row in result]
return pd.DataFrame(rows)
```

Check imports: `Session` should already be imported from `sqlalchemy.orm`. If not, add it.

- [ ] **Step 2: Update `denormalize_as_dict` to use `_denormalize()` as a streaming generator**

Replace line 1053 (`yield from self._denormalize_from_members(include_tables=include_tables)`) with:

```python
sql_stmt = self._denormalize(include_tables=include_tables)
with Session(self.engine) as session:
    result = session.execute(sql_stmt)
    for row in result:
        yield dict(row._mapping)
```

- [ ] **Step 3: Run the multi-hop tests**

Run: `cd /Users/carl/GitHub/deriva-ml && uv run pytest tests/dataset/test_denormalize.py::TestMultiHopDenormalize -v --timeout=120`

Expected: Core join tests (M1, M2, D1, D2, D3, E2) should now PASS. Association tests (M3, M5) may need additional work depending on how `_prepare_wide_table` handles the ClinicalRecord_Observation path.

- [ ] **Step 4: Run ALL existing denormalize tests**

Run: `cd /Users/carl/GitHub/deriva-ml && uv run pytest tests/dataset/test_denormalize.py -v --timeout=120`

Expected: All existing tests in `TestDenormalize`, `TestDenormalizeSchemaGraph`, etc. still pass. If any fail, investigate — column naming may differ between the old and new implementation (dots vs other separators).

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/dataset/dataset_bag.py
git commit -m "fix: switch bag denormalize to SQL-based implementation for multi-hop FK support"
```

---

### Task 4: Handle ambiguous paths with actionable error messages

**Files:**
- Modify: `src/deriva_ml/model/catalog.py:636-654` (`_table_relationship`)
- Modify: `src/deriva_ml/model/catalog.py:686-698` (`_schema_to_paths` `find_arcs`)

The ambiguity detection already exists in `_table_relationship()` (line 650: `if len(relationships) != 1`), but the error message needs enhancement. Additionally, `_schema_to_paths()` silently skips ambiguous tables (lines 690-698) — this must be reconciled so ambiguity is properly raised when it matters.

- [ ] **Step 1: Enhance the ambiguous path error message in `_table_relationship()`**

Read `catalog.py:636-654`. The current error (line 651-653) says:
```python
raise DerivaMLException(
    f"Ambiguous linkage between {table1.name} and {table2.name}: {[(r[0].name, r[1].name) for r in relationships]}"
)
```

Enhance to include path suggestions:

```python
if len(relationships) == 0:
    raise DerivaMLException(
        f"No FK relationship found between {table1.name} and {table2.name}. "
        f"These tables may not be directly connected. Check your include_tables list."
    )
if len(relationships) > 1:
    path_descriptions = []
    for fk_col, pk_col in relationships:
        path_descriptions.append(f"  {fk_col.table.name}.{fk_col.name} → {pk_col.table.name}.{pk_col.name}")
    # Find intermediate tables that could disambiguate
    intermediate_tables = set()
    for fk_col, pk_col in relationships:
        # If this is a direct FK, the intermediate is any table on alternative paths
        for other_fk_col, other_pk_col in relationships:
            if (fk_col, pk_col) != (other_fk_col, other_pk_col):
                if fk_col.table.name != other_fk_col.table.name:
                    intermediate_tables.add(fk_col.table.name)
                    intermediate_tables.add(other_fk_col.table.name)
    # Remove the two endpoint tables from suggestions
    intermediate_tables -= {table1.name, table2.name}

    suggestion = ""
    if intermediate_tables:
        suggestion = (
            f" Include an intermediate table in include_tables to disambiguate "
            f"(e.g., {', '.join(sorted(intermediate_tables))})."
        )
    raise DerivaMLException(
        f"Ambiguous path between {table1.name} and {table2.name}: "
        f"found {len(relationships)} FK paths:\n"
        + "\n".join(path_descriptions)
        + f"\n{suggestion}"
    )
```

Note: The exact intermediate table detection logic may need refinement based on how `_prepare_wide_table()` calls `_table_relationship()`. The key requirement from the spec is that the error message contains: (1) both table names, (2) the paths found, (3) suggestion to add intermediate tables, (4) ideally the names of intermediate tables.

- [ ] **Step 2: Investigate `_schema_to_paths()` silent skip interaction**

**Critical:** `_schema_to_paths()` at lines 690-698 **silently skips** ambiguous tables during graph traversal. `_prepare_wide_table()` at line 576 uses `_schema_to_paths()` to build its table paths, then calls `_table_relationship()` at line 621 for each adjacent pair in those paths. The question is:

**Does `_schema_to_paths()` skip Subject from Image's arcs because Image has 2 FKs to Subject (direct + via Observation)?**

Trace the logic:
1. `find_arcs(Image)` at line 687 collects `[fk.pk_table for fk in Image.foreign_keys]`
2. Image has FKs to: Subject (direct), Observation, and any other tables
3. `Counter(domain_tables)` at line 690 counts how many times each table appears
4. Subject appears ONCE (direct FK) — Image has exactly 1 FK to Subject
5. Observation appears ONCE — Image has exactly 1 FK to Observation

**Conclusion:** `_schema_to_paths()` does NOT skip Subject in this case because there's only one direct FK from Image to Subject. The ambiguity comes from `_prepare_wide_table()` discovering multiple *paths* to Subject (one direct, one through Observation), not from multiple direct FKs.

However, the ambiguity detection in `_table_relationship()` checks for multiple relationships between a PAIR of tables (line 644-648), which counts both `foreign_keys` AND `referenced_by`. With the new schema, `_table_relationship(Image, Subject)` finds:
- Image.Subject → Subject.RID (direct FK)

But `_prepare_wide_table()` may build paths that include both Image→Subject and Image→Observation→Subject. The call to `_table_relationship(Image, Subject)` at line 621 still only finds 1 direct FK — so **the ambiguity might NOT be detected** by the current mechanism.

**If ambiguity is NOT detected**, you need to add detection in `_prepare_wide_table()` itself. After building `paths_by_element` (line 580), check if any element table has multiple paths that include different intermediate tables to reach the same target. Example:

```python
# After building paths_by_element, check for ambiguous endpoints
for element_table, paths in paths_by_element.items():
    # Check if multiple paths reach the same endpoint through different intermediates
    endpoints = defaultdict(list)
    for path in paths:
        endpoint = path[-1].name
        path_names = [p.name for p in path]
        endpoints[endpoint].append(path_names)

    for endpoint, endpoint_paths in endpoints.items():
        if len(endpoint_paths) > 1 and endpoint in include_tables:
            # Multiple paths to same endpoint — check if user included intermediates
            # to disambiguate
            path_strs = [" → ".join(p[2:]) for p in endpoint_paths]  # Skip Dataset, Dataset_X prefix
            raise DerivaMLException(
                f"Ambiguous path to {endpoint}: found {len(endpoint_paths)} paths:\n"
                + "\n".join(f"  {s}" for s in path_strs)
                + f"\nInclude intermediate tables in include_tables to disambiguate."
            )
```

**If ambiguity IS detected** by `_table_relationship()`, enhance the message as shown in Step 1.

Read the code, add a print/log statement, and test with `["Image", "Subject"]` to determine which case applies before writing the fix.

- [ ] **Step 3: Remove xfail markers from existing tests**

Find all tests marked with `xfail(reason="Ambiguous Image→Subject path...")` from Task 1 and remove the markers. These tests should now pass with the ambiguity handling in place.

Run: `cd /Users/carl/GitHub/deriva-ml && grep -n "Ambiguous Image.*Subject.*path" tests/dataset/test_denormalize.py`

Remove each `@pytest.mark.xfail(...)` decorator found.

- [ ] **Step 4: Run ambiguous path tests**

Run: `cd /Users/carl/GitHub/deriva-ml && uv run pytest tests/dataset/test_denormalize.py::TestAmbiguousPaths -v --timeout=120`

Expected: A1, A1b should PASS (error raised with useful message). A2, A2b should PASS (disambiguation works). A3, A4 should PASS.

- [ ] **Step 5: Run ALL tests including previously-xfailed ones**

Run: `cd /Users/carl/GitHub/deriva-ml && uv run pytest tests/dataset/test_denormalize.py -v --timeout=120`

Expected: All tests pass, no xfail markers remain.

- [ ] **Step 6: Commit**

```bash
git add src/deriva_ml/model/catalog.py src/deriva_ml/dataset/dataset_bag.py tests/dataset/test_denormalize.py
git commit -m "feat: improve ambiguous FK path error messages with path details and disambiguation guidance"
```

---

### Task 5: Fix catalog-side `_denormalize_datapath` with multi-hop support

**Files:**
- Modify: `src/deriva_ml/dataset/dataset.py:780-881` (`_denormalize_datapath`)

Rewrite to follow FK chains via the datapath API instead of only joining tables with dataset members. Must support multi-hop chains (e.g., Image → Observation → Subject) by walking through intermediate tables.

- [ ] **Step 1: Read the current implementation and plan the multi-hop approach**

Read `dataset.py:780-881`. The current approach:
1. Calls `list_dataset_members()` for all tables
2. Iterates over primary table members
3. For each other table, checks `if other_table_name in members` (the bug at line 861)
4. Does single-hop FK lookup via `_table_relationship()`

The fix needs to:
1. Keep using `list_dataset_members()` for the primary table (first table with members)
2. For other tables, use `_table_relationship()` to discover the FK path — but handle multi-hop by chaining through intermediate tables
3. Pre-fetch all records for non-member tables by following FK chains from primary members
4. Handle ambiguous paths (same as bag-side — `_table_relationship()` raises the error)

Multi-hop approach: For Image → Observation → Subject, the algorithm should:
1. Find FK from Image to Observation → collect Observation RIDs from Image members
2. Fetch all those Observations from catalog
3. Find FK from Observation to Subject → collect Subject RIDs from Observations
4. Fetch all those Subjects from catalog
5. Build lookup indexes for each hop

- [ ] **Step 2: Implement the multi-hop fix**

Replace the body of `_denormalize_datapath()` with:

```python
def _denormalize_datapath(
    self,
    include_tables: list[str],
    version: DatasetVersion | str | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Denormalize dataset members by joining related tables via multi-hop FK chains."""

    skip_columns = {"RCT", "RMT", "RCB", "RMB"}

    # Get dataset members for the primary table
    members = self.list_dataset_members(version=version, recurse=True)

    # Build column lists
    table_columns: dict[str, list[str]] = {}
    for table_name in include_tables:
        table = self._ml_instance.model.name_to_table(table_name)
        table_columns[table_name] = [
            c.name for c in table.columns if c.name not in skip_columns
        ]

    # Find primary table (first table with dataset members)
    primary_table = None
    for table_name in include_tables:
        if table_name in members and members[table_name]:
            primary_table = table_name
            break

    if primary_table is None:
        return

    # Determine join order: topological sort of included tables based on FK relationships
    # Build a chain from primary table through each intermediate to reach all targets
    # Use _prepare_wide_table's graph to find the join path
    primary_table_obj = self._ml_instance.model.name_to_table(primary_table)

    # Build record indexes for each non-member table by following FK chains
    # Key insight: walk include_tables in FK-chain order from primary
    record_indexes: dict[str, dict[str, dict]] = {}  # table_name -> {RID -> record}
    fk_relationships: dict[str, tuple] = {}  # table_name -> (from_table, fk_col, pk_col)

    # Determine the FK chain order by BFS from primary table through include_tables
    visited = {primary_table}
    queue = [primary_table]
    chain_order = []

    while queue:
        current_name = queue.pop(0)
        current_table = self._ml_instance.model.name_to_table(current_name)

        for target_name in include_tables:
            if target_name in visited:
                continue
            target_table = self._ml_instance.model.name_to_table(target_name)

            # Check direct FK relationship (will raise DerivaMLException if ambiguous)
            try:
                fk_col, pk_col = self._ml_instance.model._table_relationship(
                    current_table, target_table
                )
                visited.add(target_name)
                queue.append(target_name)
                chain_order.append(target_name)
                fk_relationships[target_name] = (current_name, fk_col, pk_col)
            except DerivaMLException:
                # Not directly connected — might be reachable via another table
                continue

    # Now fetch records for each table in chain order
    # Start with primary table members
    record_indexes[primary_table] = {
        m["RID"]: m for m in members[primary_table]
    }

    pb = self._ml_instance.domain_path()

    for target_name in chain_order:
        from_table_name, fk_col, pk_col = fk_relationships[target_name]

        # Collect FK values from the source table
        fk_values = set()
        for record in record_indexes.get(from_table_name, {}).values():
            fk_value = record.get(fk_col.name)
            if fk_value:
                fk_values.add(fk_value)

        if not fk_values:
            record_indexes[target_name] = {}
            continue

        # Fetch related records from catalog
        target_path = pb.tables[target_name]
        # Use batched queries if needed for large FK sets
        related_records = list(
            target_path.path.filter(
                target_path.column_definitions[pk_col.name].in_(list(fk_values))
            ).entities().fetch()
        )
        record_indexes[target_name] = {
            r[pk_col.name]: r for r in related_records
        }

    # Build output rows
    for member in members[primary_table]:
        row: dict[str, Any] = {}

        # Add primary table columns
        for col_name in table_columns[primary_table]:
            row[f"{primary_table}_{col_name}"] = member.get(col_name)

        # Add columns from each joined table
        for target_name in include_tables:
            if target_name == primary_table:
                continue

            other_cols = table_columns[target_name]

            # Initialize to None (outer join)
            for col_name in other_cols:
                row[f"{target_name}_{col_name}"] = None

            # Walk the FK chain from primary to this target
            if target_name in fk_relationships:
                # Trace through the chain to find the related record
                current_record = member
                current_table_name = primary_table
                found = True

                # Build the path from primary to target
                path_to_target = []
                t = target_name
                while t != primary_table:
                    path_to_target.append(t)
                    t = fk_relationships[t][0]
                path_to_target.reverse()

                for hop_target in path_to_target:
                    from_name, fk_c, pk_c = fk_relationships[hop_target]
                    fk_value = current_record.get(fk_c.name)
                    if fk_value and fk_value in record_indexes.get(hop_target, {}):
                        current_record = record_indexes[hop_target][fk_value]
                        current_table_name = hop_target
                    else:
                        found = False
                        break

                if found and current_table_name == target_name:
                    for col_name in other_cols:
                        row[f"{target_name}_{col_name}"] = current_record.get(col_name)

        yield row
```

Note: The `filter().entities().fetch()` datapath API pattern may need adjustment — check how existing code in `dataset.py` queries the catalog. The `column_definitions[name].in_()` call is the ERMrest datapath filter syntax. Verify by reading existing filter patterns in the file.

**Association table handling:** The BFS above only checks direct FK relationships between tables in `include_tables`. For M:N joins (Observation ↔ ClinicalRecord via ClinicalRecord_Observation), there's no direct FK — the connection goes through an association table not in `include_tables`.

To handle this: after the initial BFS, check for any `include_tables` entries still in `visited` but not in `chain_order`. For each, check if it's reachable via an association table by:

1. Call `self._ml_instance.model.name_to_table(target_name)` for unvisited tables
2. Use `find_associations()` on visited tables to discover association tables
3. If an association table has FKs to both a visited table and the unvisited target, add it to the chain as a two-hop: visited → association → target
4. Pre-fetch both hops during the record-fetching phase

Alternatively, if this proves complex, the simpler approach is to add association tables to `include_tables` automatically when they're needed — `is_association()` from `catalog.py` can identify them. The bag-side implementation already handles this in `_prepare_wide_table()` which the catalog-side doesn't use.

If association table handling proves too complex for this task, document it as a known limitation and add a `TODO` — the core fix (direct FK chains) is the priority.

- [ ] **Step 3: Add catalog-side multi-hop tests to `TestCatalogDenormalize`**

Add tests mirroring the bag-side tests but using catalog-based denormalization:

```python
def test_catalog_non_member_fk_join(self, catalog_with_datasets, tmp_path):
    """Catalog: Join to non-member table via FK."""
    ml_instance, dataset_description = catalog_with_datasets
    dataset = dataset_description.dataset

    df = dataset.denormalize_as_dataframe(include_tables=["Image", "Observation"])

    assert len(df) > 0
    obs_cols = [c for c in df.columns if c.startswith("Observation_")]
    assert len(obs_cols) > 0, "Expected Observation columns"

    obs_rid_col = "Observation_RID"
    if obs_rid_col in df.columns:
        non_null = df[obs_rid_col].notna().sum()
        assert non_null > 0, "Observation_RID should be populated"

def test_catalog_multihop_chain(self, catalog_with_datasets, tmp_path):
    """Catalog: Multi-hop chain Image → Observation → Subject."""
    ml_instance, dataset_description = catalog_with_datasets
    dataset = dataset_description.dataset

    df = dataset.denormalize_as_dataframe(
        include_tables=["Image", "Observation", "Subject"]
    )

    assert len(df) > 0
    for prefix in ["Image_", "Observation_", "Subject_"]:
        cols = [c for c in df.columns if c.startswith(prefix)]
        assert len(cols) > 0, f"Expected columns with prefix {prefix}"

    # Subject should be populated via multi-hop
    if "Subject_RID" in df.columns:
        non_null = df["Subject_RID"].notna().sum()
        assert non_null > 0, "Subject_RID should be populated via multi-hop"

def test_catalog_ambiguous_paths_error(self, catalog_with_datasets, tmp_path):
    """Catalog: Ambiguous paths raise error."""
    from deriva_ml.core.exceptions import DerivaMLException

    ml_instance, dataset_description = catalog_with_datasets
    dataset = dataset_description.dataset

    with pytest.raises(DerivaMLException) as exc_info:
        dataset.denormalize_as_dataframe(include_tables=["Image", "Subject"])

    error_msg = str(exc_info.value)
    assert "ambiguous" in error_msg.lower() or "multiple" in error_msg.lower()
    assert "Observation" in error_msg

def test_catalog_disambiguation(self, catalog_with_datasets, tmp_path):
    """Catalog: Including intermediate resolves ambiguity."""
    ml_instance, dataset_description = catalog_with_datasets
    dataset = dataset_description.dataset

    df = dataset.denormalize_as_dataframe(
        include_tables=["Image", "Observation", "Subject"]
    )

    assert len(df) > 0
    for prefix in ["Image_", "Observation_", "Subject_"]:
        cols = [c for c in df.columns if c.startswith(prefix)]
        assert len(cols) > 0, f"Expected columns with prefix {prefix}"
```

- [ ] **Step 4: Run all tests**

Run: `cd /Users/carl/GitHub/deriva-ml && uv run pytest tests/dataset/test_denormalize.py -v --timeout=120`

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/dataset/dataset.py tests/dataset/test_denormalize.py
git commit -m "fix: catalog-side denormalize follows multi-hop FK chains for non-member tables"
```

---

### Task 6: Add bag/catalog consistency test

**Files:**
- Modify: `tests/dataset/test_denormalize.py`

- [ ] **Step 1: Add consistency tests**

```python
def test_bag_catalog_multihop_consistency(self, catalog_with_datasets, tmp_path):
    """C1: Bag and catalog denormalize produce same data for multi-hop joins."""
    ml_instance, dataset_description = catalog_with_datasets
    dataset = dataset_description.dataset
    current_version = dataset.current_version

    # Catalog-based
    catalog_df = dataset.denormalize_as_dataframe(
        include_tables=["Image", "Observation"]
    )

    # Bag-based
    bag = dataset.download_dataset_bag(current_version, use_minid=False)
    bag_df = bag.denormalize_as_dataframe(include_tables=["Image", "Observation"])

    # Same row count
    assert len(catalog_df) == len(bag_df), (
        f"Row count mismatch: catalog={len(catalog_df)}, bag={len(bag_df)}"
    )

    # Same Image RIDs (compare sets, ignoring order)
    catalog_rids = set(catalog_df["Image_RID"].dropna())
    bag_rids = set(bag_df["Image.RID"].dropna())
    assert catalog_rids == bag_rids, "Image RIDs should match between catalog and bag"

    # Same Observation RIDs
    catalog_obs = set(catalog_df["Observation_RID"].dropna())
    bag_obs = set(bag_df["Observation.RID"].dropna())
    assert catalog_obs == bag_obs, "Observation RIDs should match between catalog and bag"
```

- [ ] **Step 2: Run test**

Run: `cd /Users/carl/GitHub/deriva-ml && uv run pytest tests/dataset/test_denormalize.py::TestCatalogDenormalize::test_bag_catalog_multihop_consistency -v --timeout=120`

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/dataset/test_denormalize.py
git commit -m "test: add bag/catalog consistency test for multi-hop FK denormalization"
```

---

### Task 7: Remove dead code

**Files:**
- Modify: `src/deriva_ml/dataset/dataset_bag.py`

- [ ] **Step 1: Remove `_denormalize_from_members` from `dataset_bag.py`**

Delete the entire `_denormalize_from_members` method (starts at line 830). Verify no other code references it:

Run: `cd /Users/carl/GitHub/deriva-ml && grep -rn "_denormalize_from_members" src/`

Expected: No matches (all callers were switched in Task 3).

- [ ] **Step 2: Check if `_find_relationship_attr` is still used**

Run: `cd /Users/carl/GitHub/deriva-ml && grep -rn "_find_relationship_attr" src/`

Check if it's only called by `_denormalize_from_members` (now deleted). The `_denormalize()` method at line 792 has its own `find_relationship` closure that does the same thing. If `_find_relationship_attr` has no remaining callers, remove it.

- [ ] **Step 3: Check for other dead private methods**

Run: `cd /Users/carl/GitHub/deriva-ml && grep -rn "def _" src/deriva_ml/dataset/dataset_bag.py`

For each private method, verify it has callers. Remove any orphaned methods. Be conservative — only remove methods with zero references.

- [ ] **Step 4: Run all tests to verify nothing breaks**

Run: `cd /Users/carl/GitHub/deriva-ml && uv run pytest tests/ -v --timeout=300`

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/dataset/dataset_bag.py
git commit -m "chore: remove dead code (_denormalize_from_members, unused private methods)"
```

---

### Task 8: Final integration test

**Files:**
- No new files — run full test suite

- [ ] **Step 1: Run complete test suite**

Run: `cd /Users/carl/GitHub/deriva-ml && uv run pytest tests/ -v --timeout=300`

Expected: All tests pass.

- [ ] **Step 2: Run just the denormalize tests with verbose output**

Run: `cd /Users/carl/GitHub/deriva-ml && uv run pytest tests/dataset/test_denormalize.py -v --timeout=120 -s`

Expected: All tests pass, no warnings about ambiguous paths in normal (non-ambiguous) test cases.

- [ ] **Step 3: Commit any final fixes**

If any tests required fixes, commit them:

```bash
git add -A
git commit -m "fix: address integration test issues for denormalize multi-hop FK"
```
