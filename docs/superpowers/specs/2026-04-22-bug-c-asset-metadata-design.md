# Bug C: Asset Metadata None-Stringification — Design

**Status:** Draft · **Date:** 2026-04-22 · **Bug:** Pre-S3 regression, affects all asset tables with non-string metadata columns

## 1. Problem

Three call-sites in the asset-upload pipeline substitute the literal string `"None"` for missing metadata values when building the staging directory tree:

- `src/deriva_ml/execution/execution.py:1074` (`_build_upload_staging`)
- `src/deriva_ml/execution/upload_engine.py:644` (`_invoke_deriva_py_uploader`)
- `src/deriva_ml/dataset/upload.py:691` (`asset_file_path` path-builder)

Each looks like:
```python
target_dir = target_dir / str(metadata.get(col, "None"))
```

deriva-py's `GenericUploader` walks that tree, captures the directory segment with its `(?P<col>[-:._ \w]+)` regex (which matches `"None"` happily), and substitutes it via `column_map` into the catalog insert. For non-string columns — timestamp, date, int4, etc. — the server rejects the insert:

```
invalid input syntax for type timestamp: "None"
```

When metadata is `nullok=True`, the user expects SQL `NULL` in the catalog. When `nullok=False`, they expect a clear error pointing at the missing value. Today they get neither — just an opaque 400 from the catalog mid-upload.

## 2. Goals & non-goals

**In scope:**
- Pre-upload validation that fails fast with an actionable error when NOT-NULL metadata columns are missing.
- Correct SQL `NULL` emission when nullable metadata columns are missing.
- A single pre-upload checkpoint that covers all three staging sites (`upload_outputs`, `upload_pending`, `run_upload_engine`).
- A pre-processor pattern that plugs into deriva-py's documented extension point.

**Out of scope:**
- Adding per-column defaults that the validator would fill in automatically. (YAGNI; users can do this in their own code.)
- Renaming or restructuring existing metadata APIs (`asset_file_path`, `AssetFilePath.metadata`).
- Any upstream changes to deriva-py itself.
- Bug E.2 (deriva-py upserts by MD5+Filename, separate subsystem).

## 3. Design summary

Three units, one pre-upload checkpoint, one deriva-py extension point:

1. **Validator** — `_validate_pending_asset_metadata(model, manifest)` in `src/deriva_ml/asset/manifest.py`. Runs once at the top of every upload entry point. Raises `DerivaMLValidationError` listing all missing NOT-NULL columns across all pending assets.

2. **Sentinel** — `NULL_SENTINEL = "__NULL__"` constant in `src/deriva_ml/dataset/upload.py`. The three staging sites use this instead of `"None"` for nullable missing metadata.

3. **Pre-processor** — `NullSentinelProcessor(BaseProcessor)` in `src/deriva_ml/asset/null_sentinel_processor.py`. Registered in `asset_table_upload_spec`'s `pre_processors` list. Mutates `self.metadata`, replacing `"__NULL__"` with `None` before deriva-py's `interpolateDict` runs. deriva-py's existing None-pruning (at `deriva_upload.py:356-358`) drops those keys, and the catalog insert receives SQL `NULL`.

### 3.1 Why pre-upload (not per-mutation) validation

Metadata can enter the manifest at two points: `asset_file_path(metadata=...)` at registration, and `AssetFilePath.metadata = ...` post-registration via the setter. Validating at either of those individually would miss the case where the user registers incomplete, adds some metadata, then uploads still-incomplete. A single pass at the top of the upload entry point inspects the **final** manifest state and catches any inconsistency regardless of how it arose.

### 3.2 Why the sentinel + pre-processor (not "skip the dir level")

deriva-py's `asset_path_regex` is built from `sorted(metadata_columns)` at spec-generation time and is a static regex — there's no per-file conditional path. Making some directory levels optional would require forking `asset_table_upload_spec` per-file or post-processing the scan results. The sentinel is a minimal in-band marker that lets us reuse the existing regex + column_map plumbing with one small in-place transformation.

### 3.3 Why `"__NULL__"` specifically

- Matches deriva-py's `(?P<col>[-:._ \w]+)` regex (`\w` includes underscores).
- Unambiguous — no natural metadata value looks like `__NULL__`.
- Short enough to inspect in a staging-directory listing.
- Collision risk documented as a known constraint.

## 4. Components

### 4.1 `_validate_pending_asset_metadata`

**Location:** module-level private function in `src/deriva_ml/asset/manifest.py`.

**Signature:**
```python
def _validate_pending_asset_metadata(
    model: "DerivaModel",
    manifest: "AssetManifest",
) -> None:
    """Raise DerivaMLValidationError if any pending asset entry is
    missing a NOT-NULL metadata column value.

    Iterates manifest entries where status == pending and the asset
    table has metadata columns. For each NOT-NULL column absent from
    the entry's metadata dict, records (manifest_key, schema, table,
    column). If any errors collected, raises a single
    DerivaMLValidationError whose message lists every failure in
    sorted order.

    Nullable columns may be absent without error; downstream staging
    substitutes ``NULL_SENTINEL`` which the upload pre-processor
    translates to SQL NULL.
    """
```

**Error message shape:**
```
Missing required metadata for 3 pending asset(s):
  - Image/acq001.png: missing columns Acquisition_Date, Acquisition_Time
  - Image/acq002.png: missing columns Acquisition_Time
  - Plate/p1.json: missing column Well_Position
Supply these values before calling upload_outputs(), either via
the ``metadata=`` arg to asset_file_path(...) or by assigning to
the returned AssetFilePath's ``metadata`` property.
```

**Call sites (3):**
- `Execution.upload_outputs()` — first step, before `_build_upload_staging`.
- `DerivaML.upload_pending()` — first step, before engine dispatch.
- `run_upload_engine()` — first step, before `_enumerate_work`.

### 4.2 `DerivaModel.asset_metadata_columns`

**Location:** new method on `DerivaModel` in `src/deriva_ml/model/catalog.py` (or wherever the existing `asset_metadata(table) -> set[str]` lives).

**Signature:**
```python
def asset_metadata_columns(
    self, table: "str | Table"
) -> list["Column"]:
    """Return Column objects (not just names) for the asset-metadata
    columns of ``table``. Ordered deterministically (sorted by name)
    for stable iteration in callers.
    """
```

Used by the validator to inspect `nullok`. The existing `asset_metadata(table) -> set[str]` method remains unchanged for backward compatibility.

### 4.3 `NULL_SENTINEL` constant

**Location:** module-level constant at the top of `src/deriva_ml/dataset/upload.py`.

```python
NULL_SENTINEL = "__NULL__"
"""Directory-segment marker for nullable asset-metadata columns with
no value. Translated to Python None by NullSentinelProcessor before
deriva-py builds the catalog insert. See Bug C design doc."""
```

Imported by the three staging sites.

### 4.4 `NullSentinelProcessor`

**Module:** `src/deriva_ml/asset/null_sentinel_processor.py`.

**Class:**
```python
from deriva.transfer.upload.processors import BaseProcessor


class NullSentinelProcessor(BaseProcessor):
    """Pre-upload metadata pre-processor for deriva-py's GenericUploader.

    Translates the sentinel string ``"__NULL__"`` (see
    :data:`deriva_ml.dataset.upload.NULL_SENTINEL`) in the uploader's
    metadata dict to Python ``None``. deriva-py's
    :func:`interpolateDict` then drops None-valued keys, causing the
    resulting catalog insert to send SQL ``NULL`` for those columns.

    Not part of the end-user API — configured automatically by
    :func:`deriva_ml.dataset.upload.asset_table_upload_spec`.
    """

    def process(self):
        for k, v in list(self.metadata.items()):
            if v == "__NULL__":
                self.metadata[k] = None
```

### 4.5 Upload spec wiring

In `asset_table_upload_spec()` at `src/deriva_ml/dataset/upload.py`, when `metadata_columns` is non-empty, the returned spec dict gains:

```python
spec["pre_processors"] = [
    {
        "processor": "NullSentinelProcessor",
        "processor_type": (
            "deriva_ml.asset.null_sentinel_processor.NullSentinelProcessor"
        ),
    }
]
```

deriva-py's `find_processor(..., bypass_whitelist=True)` imports the class by its fully-qualified path and instantiates it per file upload.

### 4.6 Three staging-site edits

Each site replaces one string literal:

```diff
- target_dir = target_dir / str(metadata.get(col, "None"))
+ target_dir = target_dir / str(metadata.get(col, NULL_SENTINEL))
```

and adds `from deriva_ml.dataset.upload import NULL_SENTINEL` at the top.

## 5. Data flow

### 5.1 Happy path (all metadata supplied)

```
exe.upload_outputs()
  → _validate_pending_asset_metadata — no errors
  → staging: directory per metadata col, all values real
  → deriva-py scanner → regex → column_map → catalog insert with real values
```

### 5.2 NOT-NULL missing — the validator catches it

```
exe.upload_outputs()
  → _validate_pending_asset_metadata — collects errors
  → raise DerivaMLValidationError
  → user sees aggregated message, fixes, retries
```

No staging or upload work performed.

### 5.3 Nullable missing — sentinel path

```
exe.upload_outputs()
  → _validate_pending_asset_metadata — no errors (nullable OK)
  → staging: "__NULL__" written as directory segment for missing values
  → deriva-py scanner → regex captures "__NULL__" into self.metadata
  → _execute_processors(PRE_PROCESSORS_KEY)
      → NullSentinelProcessor.process() — "__NULL__" → None
  → interpolateDict — drops None-valued keys
  → catalog insert with SQL NULL for those columns
```

### 5.4 Interaction with other subsystems

- **Offline mode:** `upload_outputs` is already online-only. The validator runs purely against local manifest + cached model — works offline, though the subsequent upload would fail with `DerivaMLReadOnlyError`. No regression.
- **Schema pin:** Orthogonal. Pinned cache still lets the validator inspect `nullok` (it reads from the in-memory model, not the live catalog).
- **Workspace reset:** Orthogonal. Validator reads whatever manifest exists.

## 6. Error handling summary

| Scenario | Exception / behavior |
|---|---|
| NOT-NULL column missing on any pending asset | `DerivaMLValidationError` with aggregated message |
| Nullable column missing on any pending asset | (none — sentinel path) |
| All metadata supplied | (none — clean upload) |
| Asset table has zero metadata columns | Validator fast-paths (no errors); staging skips sentinel logic |
| User sets `entry.metadata["X"] = None` explicitly (Pydantic `model_dump` exclude_none idiom) | Treated identically to absence — same validator + sentinel path |
| `"__NULL__"` is a legitimate metadata value | **Corrupted to SQL NULL by the pre-processor.** Known constraint; documented in the processor's docstring. |
| Pre-processor import fails (class renamed/moved) | `DerivaUploadConfigurationError` from deriva-py at upload time — surfaces as upload failure, not silent corruption |

## 7. Testing plan

### 7.1 Unit tests — no catalog needed

**`tests/asset/test_metadata_validator.py`** (new, ~8 tests):
1. `test_empty_manifest_returns_none`
2. `test_asset_with_no_metadata_columns_passes`
3. `test_all_required_metadata_present_passes`
4. `test_missing_single_not_null_column_raises`
5. `test_missing_multiple_columns_aggregated`
6. `test_missing_across_multiple_assets_aggregated`
7. `test_nullable_missing_is_not_an_error`
8. `test_error_message_is_deterministic`

**`tests/asset/test_null_sentinel_processor.py`** (new, ~4 tests):
9. `test_single_sentinel_converted_to_none`
10. `test_multiple_sentinels_in_metadata_all_converted`
11. `test_non_sentinel_values_unchanged`
12. `test_empty_metadata_is_no_op`

### 7.2 Integration tests — live-catalog, gated on `DERIVA_HOST`

Resurrect and extend `tests/execution/test_upload_engine_live_smoke.py` (deleted after the S3 branch merge; original at commit `df620c3`).

13. `test_upload_engine_live_end_to_end` — unchanged from original; zero-metadata `Execution_Asset` path.
14. `test_upload_asset_with_missing_required_metadata_raises_validation_error` — Bug C reproducer now passing as `DerivaMLValidationError`, not xfail.
15. `test_upload_asset_with_missing_nullable_metadata_succeeds_with_null` — the sentinel path, end-to-end, asserts catalog row has SQL NULL.

### 7.3 Regression

- Existing `tests/execution/` upload tests pass unchanged — validation + sentinel are no-ops when metadata is complete.
- Schema-pin, S4, and S3 regression suites pass unchanged (orthogonal).
- Old xfail reproducer becomes redundant — deleted or repurposed as test #14.

### 7.4 Test catalog requirements

The validator test (#14) needs an asset table with at least one `nullok=False` metadata column. The test fixtures already include `Image` with `Acquisition_Time` and `Acquisition_Date` — nullability needs confirmation at implementation time. If both are nullable, the plan adds a fixture tweak (add a NOT-NULL column to one asset table for the test).

## 8. Risks

1. **`"__NULL__"` collision with a legitimate string value.** Mitigation: double-underscore framing + docstring note.

2. **deriva-py `self.metadata` semantics change.** We rely on `process()` receiving a live mutable reference. Current behavior verified at `deriva_upload.py:1141`. Integration test #15 is the canary — a silent drift would fail it.

3. **`find_processor(bypass_whitelist=True)` behavior drift.** Low risk; this is the documented path for asset-mapping pre_processors.

4. **Validator false positives.** Reading `nullok` from the parsed ermrest schema (via `DerivaModel`) avoids any translation-layer bugs.

5. **Users who relied on broken `None`-stringification.** Those uploads were already failing in production. New behavior replaces a 400 with a `DerivaMLValidationError` — better UX, same net outcome (the upload still cannot succeed).

## 9. Rollout

Single PR against `main`. No feature flag. Breaking change is limited to callers whose uploads were already broken.

CHANGELOG entry under a new `## Unreleased — Bug C: asset metadata None-stringification` section documenting:
- The validator and its failure mode.
- The sentinel protocol.
- The `NullSentinelProcessor` as an opaque implementation detail.
- That callers whose uploads succeeded before continue to succeed.

## 10. Open questions

None at spec time. Implementation-time checks:
- Confirm `Image.Acquisition_Time`/`Acquisition_Date` nullability in the test fixture.
- Confirm `model.name_to_table(t).columns[c].nullok` is the right accessor (vs. `nullok()` method or similar).
