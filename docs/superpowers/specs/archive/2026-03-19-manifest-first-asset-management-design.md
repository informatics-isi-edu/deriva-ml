# Manifest-First Asset Management for DerivaML

## Context

DerivaML's current asset management encodes metadata as directory names in a shadow tree that mirrors the GenericUploader's regex patterns. This creates four problems:

1. **Crash fragility** — If a model crashes between `asset_file_path()` and upload, the shadow directory state may be inconsistent
2. **Metadata inflexibility** — Column values encoded as directory names are brittle (special chars, ordering, renaming)
3. **Upload coupling** — The staging format is tightly bound to GenericUploader's regex conventions
4. **Code complexity** — The shadow tree creation, regex generation, and FK fixup are entangled

This design replaces the eager shadow-tree approach with a persistent JSON manifest as single source of truth, flat per-table file storage, and ephemeral symlink staging at upload time.

## Design Decisions

- **Manifest persistence**: Write-through + fsync on every mutation for crash safety
- **API compatibility**: `asset_file_path()` still returns `AssetFilePath` (a `Path` subclass), same signature
- **Metadata addition**: `AssetFilePath` gains `.set_metadata(col, val)` and `.set_asset_types([...])` methods
- **Validation**: Column names validated immediately; FK values warn at set time, error at upload
- **Duplicate handling**: Last-write-wins with warning log
- **Upload resume**: Per-asset status tracking in manifest (pending → uploaded with RID)
- **Upload staging**: Ephemeral symlinks from flat storage into regex-expected tree at upload time
- **Backward compatibility**: Clean break — same API surface, new internals

## Architecture

### Directory Structure (New)

```
{working_dir}/deriva-ml/execution/{exec_rid}/
├── asset-manifest.json              # Single source of truth (write-through + fsync)
├── assets/                          # Flat per-table storage
│   ├── Image/
│   │   ├── scan001.jpg
│   │   └── scan002.jpg
│   ├── Model/
│   │   └── weights.pt
│   └── Execution_Metadata/
│       └── config.yaml
├── upload-staging/                   # Ephemeral, created at upload time only
│   └── asset/{schema}/{table}/{metadata_dirs}/...  # Symlinks into assets/
├── features/                        # Unchanged from current
│   └── {schema}/{target_table}/{feature_name}/...
└── table/                           # Unchanged from current
    └── {schema}/{table}/{table}.csv
```

### Manifest Schema

```json
{
  "version": 1,
  "execution_rid": "4SP",
  "created_at": "2026-03-19T...",
  "assets": {
    "Image/scan001.jpg": {
      "asset_table": "Image",
      "schema": "test-schema",
      "asset_types": ["Training_Data"],
      "metadata": {
        "Subject": "2-DEF",
        "Acquisition_Date": "2026-01-15"
      },
      "status": "pending",
      "rid": null,
      "uploaded_at": null,
      "error": null
    }
  },
  "features": {
    "Diagnosis": {
      "feature_name": "Diagnosis",
      "target_table": "Image",
      "schema": "test-schema",
      "values_path": "features/test-schema/Image/Diagnosis/Diagnosis.jsonl",
      "asset_columns": {},
      "status": "pending"
    }
  }
}
```

- Asset keys are `{AssetTable}/{filename}` (unique per table)
- `status` transitions: `pending` → `uploaded` (with RID) | `failed` (with error)
- `metadata` dict maps column names to values — no directory encoding
- Features tracked separately for different upload flow

### asset_file_path() Three Modes (Preserved)

The current method has three modes based on whether `file_name` exists. All three are preserved:

1. **New file** — `file_name` doesn't exist: returns `assets/{Table}/filename.ext`. Caller writes to it.
2. **Symlink** — `file_name` exists, `copy_file=False` (default): creates symlink `assets/{Table}/filename.ext` → original
3. **Copy** — `file_name` exists, `copy_file=True`: copies file to `assets/{Table}/filename.ext`

In all cases, a manifest entry is created immediately. The `rename_file` parameter works unchanged.

**Key difference from current**: Files land in `assets/{Table}/` (flat) instead of `asset/{schema}/{Table}/{meta1}/{meta2}/.../` (deep tree). Metadata is in the manifest, not the path.

### AssetRecord Class (Dynamically Generated)

Following the `FeatureRecord` / `feature_record_class()` pattern, each asset table gets a dynamically generated Pydantic model:

```python
# Generate typed record class from catalog schema (cached per table)
ImageAsset = ml.asset_record_class("Image")
# → class ImageAsset(AssetRecord):
#       Subject: str           # FK to Subject table (required if non-nullable)
#       Acquisition_Date: str  # metadata column
#       Acquisition_Time: str | None = None  # nullable metadata

# Use at registration time
record = ImageAsset(Subject="2-DEF", Acquisition_Date="2026-01-15")
path = exe.asset_file_path("Image", "scan001.jpg", metadata=record)

# Or set metadata after registration
path = exe.asset_file_path("Image", "scan001.jpg")
path.metadata = ImageAsset(Subject="2-DEF", Acquisition_Date="2026-01-15")
```

**Implementation**: Reuse the same pattern as `Feature.feature_record_class()` in `feature.py:391-468`:
- `DerivaModel.asset_metadata(table)` already returns the set of metadata column names
- Use `pydantic.create_model()` with column types mapped from ERMrest types
- Required vs optional based on column nullability
- FK columns validated against referenced table at upload time

**Base class**: `AssetRecord(BaseModel)` in `src/deriva_ml/asset/aux_classes.py`
- Provides `model_dump()` for manifest serialization
- Provides column name introspection for validation

### AssetFilePath API Changes

```python
class AssetFilePath(Path):
    # Existing attributes preserved
    asset_table: str
    file_name: str
    asset_metadata: AssetRecord | None  # Changed from dict to AssetRecord
    asset_types: list[str]
    asset_rid: str | None

    # NEW: Typed metadata via AssetRecord
    @property
    def metadata(self) -> AssetRecord | None: ...

    @metadata.setter
    def metadata(self, record: AssetRecord) -> None:
        """Set metadata from an AssetRecord. Updates manifest (write-through + fsync)."""

    def set_asset_types(self, types: list[str]) -> None:
        """Set asset types. Validates terms exist in Asset_Type vocabulary.
        Raises DerivaMLValidationError for invalid terms.
        Updates manifest (write-through + fsync)."""
```

### Validation Timeline

| When | What | On failure |
|------|------|------------|
| `set_metadata(col, val)` | Column exists on asset table | Raise `DerivaMLValidationError` |
| `set_metadata(col, val)` | FK target exists (if FK column) | Warn (log) |
| `set_asset_types([...])` | Terms exist in Asset_Type vocab | Raise `DerivaMLValidationError` |
| `set_metadata(col, val)` | Same column set twice | Warn (log), last-write-wins |
| `upload_execution_outputs()` | Non-nullable columns have values | Raise with full report |
| `upload_execution_outputs()` | All FK references resolve | Raise with full report |

### Upload Flow

1. **Read manifest** — load `asset-manifest.json`
2. **Filter** — skip entries with `status: "uploaded"` (resume support)
3. **Validate** — check all required metadata present, FK refs valid
4. **Build symlink tree** — for each pending asset:
   - Read metadata from manifest
   - Create symlink: `upload-staging/asset/{schema}/{table}/{meta1}/{meta2}/.../file` → `assets/{table}/file`
5. **Upload** — feed `upload-staging/` to GenericUploader with existing bulk upload config
6. **Update manifest** — for each successful upload, write RID + "uploaded" + timestamp (fsync)
7. **FK fixup** — link assets to execution, asset types, features (same as current)
8. **Cleanup** — remove `upload-staging/` directory

### Crash Recovery

- Model crashes during training: manifest has all registered assets with `status: pending`
- Re-create execution, call `upload_execution_outputs()` — it reads manifest, skips uploaded, resumes
- Upload crashes mid-way: some assets marked `uploaded`, rest still `pending`
- Re-call `upload_execution_outputs()` — picks up where it left off

## Files to Modify

### Core changes
| File | Change |
|------|--------|
| `src/deriva_ml/asset/aux_classes.py` | Add `set_metadata()`, `set_asset_types()` to `AssetFilePath` |
| `src/deriva_ml/execution/execution.py` | Rewrite `asset_file_path()` to use flat storage + manifest |
| `src/deriva_ml/execution/execution.py` | Rewrite `_upload_execution_dirs()` to build symlinks from manifest |
| `src/deriva_ml/execution/execution.py` | Rewrite `upload_execution_outputs()` with resume logic |
| `src/deriva_ml/dataset/upload.py` | Keep `upload_directory()` and `bulk_upload_configuration()` unchanged |
| `src/deriva_ml/dataset/upload.py` | Rewrite path helpers: `asset_file_path()` → flat, remove deep tree creation |

### New files
| File | Purpose |
|------|---------|
| `src/deriva_ml/asset/manifest.py` | `AssetManifest` class — load/save/query/update manifest with fsync |
| `src/deriva_ml/asset/asset_record.py` | `AssetRecord` base class + `asset_record_class()` factory (like `feature_record_class()`) |

### Key existing utilities to reuse
| File:Line | Function | How used |
|-----------|----------|----------|
| `model/catalog.py:500` | `DerivaModel.asset_metadata(table)` | Returns metadata column names for asset table |
| `model/catalog.py:405` | `DerivaModel.is_asset(table)` | Validates table is an asset table |
| `feature.py:391-468` | `Feature.feature_record_class()` | Pattern to follow for `asset_record_class()` — uses `pydantic.create_model()` |
| `dataset/upload.py:321-508` | `upload_directory()` | Reused unchanged for GenericUploader integration |
| `dataset/upload.py:245-307` | `bulk_upload_configuration()` | Reused unchanged for upload config |

### Test changes
| File | Change |
|------|--------|
| `tests/execution/test_execution.py` | Update asset staging assertions for flat structure |
| `tests/asset/test_asset.py` | Add tests for `set_metadata()`, `set_asset_types()`, validation |
| `tests/asset/test_manifest.py` | New: manifest persistence, crash recovery, resume tests |

### Unchanged (reused as-is)
- `src/deriva_ml/dataset/upload.py: upload_directory()` — GenericUploader integration
- `src/deriva_ml/dataset/upload.py: bulk_upload_configuration()` — regex-based upload config
- `src/deriva_ml/schema/annotations.py` — bulk upload annotation generation
- `src/deriva_ml/execution/execution.py: _update_asset_execution_table()` — FK fixup
- `src/deriva_ml/execution/execution.py: _update_feature_table()` — feature remapping

## Verification

1. **Unit tests**: `AssetManifest` — create, append, update status, fsync, load after simulated crash
2. **Unit tests**: `AssetFilePath.set_metadata()` — valid columns, invalid columns, FK warnings, duplicates
3. **Integration tests**: Full execution cycle — register assets, set metadata, upload, verify catalog
4. **Resume test**: Register assets, upload half, simulate crash, resume, verify all uploaded
5. **Existing test suite**: Run full `uv run pytest` to verify no regressions
