# File Assets

File assets are tables that manage files (images, models, data files, etc.) in DerivaML. They provide automatic tracking of file metadata, integration with the Hatrac object store, and provenance linking through executions.

## What is an Asset Table?

An **asset table** is a special table type that includes standard columns for file management:

| Column | Type | Purpose |
|--------|------|---------|
| `URL` | text | Hatrac object store URL for the file |
| `Filename` | text | Original filename |
| `Length` | int8 | File size in bytes |
| `MD5` | text | MD5 checksum for integrity verification |
| `Description` | text | Optional description |

You can add additional columns for domain-specific metadata (e.g., `Width`, `Height` for images).

## Asset Types

Assets are categorized using the `Asset_Type` controlled vocabulary. This enables:

- Filtering assets by type in queries
- Organizing assets in the Chaise UI
- Consistent categorization across the catalog

```python
# List available asset types
types = ml.list_vocabulary_terms("Asset_Type")
for t in types:
    print(f"{t.name}: {t.description}")

# Add a new asset type
ml.add_term(
    table="Asset_Type",
    term_name="Segmentation_Mask",
    description="Binary mask images for segmentation tasks"
)
```

## Creating Asset Tables

### Basic Asset Table

```python
from deriva_ml import DerivaML

ml = DerivaML(hostname, catalog_id)

# Create a simple asset table
ml.create_asset(
    asset_name="Model",
    comment="Trained ML model files"
)
```

### Asset Table with Metadata Columns

```python
from deriva_ml import ColumnDefinition, BuiltinTypes

# Create an image asset table with additional metadata
ml.create_asset(
    asset_name="Image",
    column_defs=[
        ColumnDefinition(name="Width", type=BuiltinTypes.int4, comment="Image width in pixels"),
        ColumnDefinition(name="Height", type=BuiltinTypes.int4, comment="Image height in pixels"),
        ColumnDefinition(name="Format", type=BuiltinTypes.text, comment="Image format (PNG, JPEG, etc.)"),
    ],
    comment="Training and evaluation images"
)
```

### Asset Table with References

Link assets to other domain tables:

```python
# Create image asset that references Subject
ml.create_asset(
    asset_name="Image",
    referenced_tables=["Subject"],  # Creates foreign key to Subject table
    comment="Medical images linked to subjects"
)
```

## Uploading Assets in Executions

Assets are uploaded through the execution workflow using `asset_file_path()`. This:

1. Registers the file in a persistent JSON manifest (crash-safe)
2. Stores the file in flat per-table storage (`assets/{Table}/`)
3. Tracks metadata, asset types, and upload status per file
4. At upload time, builds ephemeral symlinks for the GenericUploader

### Basic Upload Pattern

```python
from deriva_ml.execution import ExecutionConfiguration

config = ExecutionConfiguration(
    workflow=ml.create_workflow("Training", "Training"),
    datasets=[DatasetSpec(rid=dataset_rid)],
)

with ml.create_execution(config) as exe:
    # Train your model...
    model = train_model(data)

    # Register the model file for upload
    model_path = exe.asset_file_path(
        asset_name="Model",           # Target asset table
        file_name="best_model.pt"     # Filename to use
    )

    # Save the model to the registered path
    torch.save(model.state_dict(), model_path)

# Upload all registered assets to the catalog
exe.upload_execution_outputs()
```

### Registering with Metadata

Provide column values for the asset table at registration time:

```python
with ml.create_execution(config) as exe:
    # Register an image with metadata columns
    path = exe.asset_file_path(
        asset_name="Image",
        file_name="scan001.jpg",
        metadata={"Subject": subject_rid, "Acquisition_Date": "2026-01-15"}
    )

    # Or update metadata after registration
    path.set_metadata("Acquisition_Time", "14:30:00")
```

Metadata is stored in the manifest JSON, not encoded in directory names.

### Registering Existing Files

```python
with ml.create_execution(config) as exe:
    # Stage an existing file for upload (symlink by default)
    exe.asset_file_path(
        asset_name="Image",
        file_name="/path/to/existing/image.png",  # Source file path
        copy_file=True,                           # Copy instead of symlink
        rename_file="processed_image.png"         # Optional: rename during upload
    )

exe.upload_execution_outputs()
```

### Applying Asset Types

```python
with ml.create_execution(config) as exe:
    # Register with specific asset types
    path = exe.asset_file_path(
        asset_name="Image",
        file_name="mask.png",
        asset_types=["Segmentation_Mask", "Derived"]  # Multiple types
    )

    # Or update types after registration
    path.set_asset_types(["Segmentation_Mask", "Derived", "QA_Reviewed"])

exe.upload_execution_outputs()
```

## Asset Manifest

Each execution tracks its assets in a persistent JSON manifest (`asset-manifest.json`). The manifest:

- Is written with fsync on every mutation for crash safety
- Tracks per-asset status: `pending` → `uploaded` (with RID) or `failed`
- Enables upload resume after crashes — already-uploaded assets are skipped
- Stores metadata as key-value pairs, not encoded in directory paths

```
{working_dir}/deriva-ml/execution/{exec_rid}/
├── asset-manifest.json     # Single source of truth
├── assets/                 # Flat per-table storage
│   ├── Image/
│   │   └── scan001.jpg
│   └── Model/
│       └── weights.pt
└── ...
```

## Listing Assets

```python
# List all assets in a table
assets = ml.list_assets("Image")
for asset in assets:
    print(f"RID: {asset['RID']}")
    print(f"  Filename: {asset['Filename']}")
    print(f"  URL: {asset['URL']}")
    print(f"  Size: {asset['Length']} bytes")
    print(f"  MD5: {asset['MD5']}")
    print(f"  Types: {asset.get('Asset_Type', [])}")
```

## Assets and Provenance

Every asset uploaded through an execution is linked via an association table:

| Table | Purpose |
|-------|---------|
| `{Asset}_Execution` | Links assets to executions with role (Input/Output) |

This enables:
- Finding which execution produced an asset
- Tracking which assets were used as inputs
- Full lineage from raw data to final outputs

```python
# Query assets produced by an execution
pb = ml.pathBuilder()
asset_execution = pb.schemas[ml.ml_schema].Image_Execution
results = asset_execution.filter(
    asset_execution.Execution == execution_rid
).filter(
    asset_execution.Asset_Role == "Output"
).entities().fetch()
```

## Assets in Datasets

Assets can be included in datasets as dataset elements:

```python
# Enable Image table as dataset element type
ml.add_dataset_element_type("Image")

# Add images to a dataset
dataset = ml.lookup_dataset(dataset_rid)
dataset.add_dataset_members(image_rids)
```

When you download a dataset bag, asset files are fetched based on the `materialize` option:

```python
# Download with all asset files
bag = exe.download_dataset_bag(
    DatasetSpec(rid=dataset_rid, materialize=True)
)

# Download metadata only (faster, smaller)
bag = exe.download_dataset_bag(
    DatasetSpec(rid=dataset_rid, materialize=False)
)
```

## Working Directory

Each execution has a working directory for asset storage and manifests:

```python
with ml.create_execution(config) as exe:
    # Get the working directory path
    work_dir = exe.working_dir
    print(f"Working directory: {work_dir}")

    # Files registered with asset_file_path are stored in assets/{Table}/
    # The manifest tracks all registered files and their metadata
    # upload_execution_outputs() builds staging symlinks and uploads
```

The working directory is cleaned up after `upload_execution_outputs(clean_folder=True)` (default).

### Crash Recovery

If a model crashes between `asset_file_path()` and `upload_execution_outputs()`, the manifest preserves the registration state. On retry:

1. Restore the execution: `exe = ml.restore_execution(execution_rid)`
2. Call `upload_execution_outputs()` — it reads the manifest and uploads all pending assets
3. Assets already marked as uploaded are skipped automatically

## Best Practices

### File Organization

- Use descriptive filenames that include relevant identifiers
- Apply consistent asset types for filtering
- Include meaningful descriptions in asset records

### Large Files

- Consider compression for large files
- Use `materialize=False` when you only need metadata
- Monitor disk space in working directories

### Provenance

- Always upload assets through executions for provenance
- Use meaningful workflow types and descriptions
- Include input datasets for complete lineage
