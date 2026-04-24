# Working offline

A downloaded bag is a self-contained, read-only snapshot of a catalog dataset that lives on your laptop and needs no network connection. By the end of this chapter you will know how to materialize asset files into a bag, read features and tables from it, organize those assets for standard ML frameworks, build new feature records offline, and commit them back to the catalog when you reconnect.

This chapter builds on Chapter 2 (datasets), Chapter 3 (features), and Chapter 4 (executions). After reading it you will be able to work entirely from a bag file without a catalog connection and then upload results in a single execution when you come back online.

## Bags and live catalogs

A live `Dataset` object talks to the server on every operation. A `DatasetBag` wraps the same data stored locally as a BDBag — a directory of CSV files, asset files, a SQLite database, and integrity manifests. The two classes implement the same `DatasetLike` protocol, so most read operations use identical method calls on either object.

The key differences are scope and direction. A bag is a point-in-time snapshot: it reflects exactly the rows and files that existed when it was downloaded at a specific version. You cannot add members, increment the version, or write feature values into the bag itself. Writes go back through a live `Execution` object after you reconnect.

## How to materialize assets for offline use

Chapter 2 covers downloading a bag in the context of an execution configuration. This section focuses on the `materialize` flag and what it gives you for offline work.

`materialize=True` (the default) tells DerivaML to fetch every asset file referenced in `fetch.txt` into `data/asset/` inside the bag directory. `materialize=False` downloads only the CSV metadata; asset files remain as remote references and cannot be read offline.

```python
from deriva_ml.dataset.aux_classes import DatasetSpec

# Full offline use: download metadata + all asset files
spec = DatasetSpec(rid="1-ABC1", version="2.0.0", materialize=True)
bag = ml.download_dataset_bag(spec)

# Metadata only: faster for cataloging, but bag.path / asset files are absent
spec_meta_only = DatasetSpec(rid="1-ABC1", version="2.0.0", materialize=False)
bag_meta = ml.download_dataset_bag(spec_meta_only)
```

Inside an execution the same `DatasetSpec` object works via `exe.download_dataset_bag(spec)`.

**Notes**

- If a bag was previously downloaded and fully materialized, DerivaML reuses the cached copy rather than re-downloading. Delete the bag directory if you want a fresh download.
- For large datasets, set a generous read timeout: `DatasetSpec(rid=..., version=..., timeout=(10, 1800))`.
- `estimate_bag_size()` on the live `Dataset` gives row counts and asset byte totals before you commit to a download — see Chapter 2.

## How to inspect bag structure

`bag.path` is a `pathlib.Path` pointing to the root of the materialized bag directory on disk.

```python
print(bag.path)
# /home/user/.deriva/bags/1-ABC1/2.0.0/

# Integrity manifest
manifest = (bag.path / "manifest-md5.txt").read_text()

# CSV data tables
images_csv = bag.path / "data" / "domain" / "Image.csv"

# Materialized asset files
asset_dir = bag.path / "data" / "asset"
```

The directory layout after materialization:

```
<bag-root>/
  manifest-md5.txt          # BDBag integrity manifest
  bag-info.txt              # bag metadata
  data/
    deriva-ml/              # core ML schema tables (CSV)
    domain/                 # domain-schema tables (CSV)
    asset/
      <RID>/
        <Table>/
          <filename>        # materialized asset files
  *.db                      # SQLite database (bag queries go here)
  schema.json               # catalog schema snapshot
```

The `bag.path` property was added in PR #64 and is available on the current branch. Treat the directory contents as read-only — bags are immutable by contract.

## How to read from a bag

`DatasetBag` mirrors the live `DerivaML` and `Dataset` APIs so you can use the same code paths offline.

```python
# List all members of the dataset, grouped by table name
members = bag.list_dataset_members()
for table_name, rows in members.items():
    print(f"{table_name}: {len(rows)} records")

# Access a full table as a DataFrame
df = bag.get_table_as_dataframe("Image")

# Iterate feature values — same signature as ml.feature_values()
from deriva_ml.feature import FeatureRecord

for rec in bag.feature_values("Image", "Glaucoma"):
    print(rec.Image, rec.Glaucoma)

# Deduplicate with a selector
records = list(bag.feature_values(
    "Image", "Glaucoma",
    selector=FeatureRecord.select_newest,
))

# Discover features on a table
for feat in bag.find_features("Image"):
    print(feat.feature_name, feat.term_columns)

# Look up a specific feature definition
feat = bag.lookup_feature("Image", "Glaucoma")
```

**Notes**

- `bag.list_dataset_members()` returns members of this dataset only. Pass `recurse=True` to include nested datasets.
- `bag.get_table_as_dataframe("Subject")` returns all rows in that table, not just those belonging to the dataset. For dataset-scoped rows, combine with `list_dataset_members()`.
- Feature cache: the first call to `bag.feature_values()` for a given `(table, feature)` pair populates a per-bag cache. Subsequent calls are fast. The cache avoids re-scanning the source CSV on subsequent calls, but every call loads all matching records into memory before yielding the first one — this is not a streaming iterator, same as the live-catalog `feature_values`.
- `bag.list_workflow_executions(workflow)` may return an empty list if the `Execution` rows were not exported into the bag. This is a known bag-export limitation; see "What bags can't do" below.

## What bags can't do

Bags are read-only snapshots. The following operations raise errors or return incomplete results against a bag:

| Operation | Behavior |
|---|---|
| Adding dataset members | `DerivaMLReadOnlyError` |
| Incrementing version | `DerivaMLReadOnlyError` |
| Uploading assets | Not supported; use a live `Execution` |
| `bag.list_workflow_executions()` | Returns empty list if `Execution` is not a dataset element type in the bag export |
| Some workflow queries | Workflow-based selectors require `Workflow_Workflow_Type` rows; these may be absent if the FK path wasn't exported |

For any write operation, create a fresh `Execution` on the live catalog when you reconnect. The offline-to-online write cycle in the next two sections covers this pattern.

## How to construct feature records offline

`bag.lookup_feature()` returns a `Feature` object whose `feature_record_class()` method generates a typed Pydantic model. That method works entirely offline — no catalog connection is needed after the bag is downloaded.

```python
# Offline: no ml or catalog object required after this point
feat = bag.lookup_feature("Image", "Glaucoma")
RecordClass = feat.feature_record_class()

# Build records from bag data
members = bag.list_dataset_members()
image_rids = [row["RID"] for row in members.get("Image", [])]

pending_records = [
    RecordClass(Image=rid, Glaucoma="Normal")
    for rid in image_rids
]
```

The `Execution` field on each record is left `None`. When you pass these records to `exe.add_features()` later, the execution fills in its own RID automatically.

**Notes**

- `feature_record_class()` returns a Pydantic model, so field names are typed and validated. Pass the wrong type and Pydantic raises a validation error before any catalog interaction.
- The model attributes match the feature definition in the bag: one attribute for the target table RID (e.g., `Image`), one for each vocabulary column (e.g., `Glaucoma`), plus `Feature_Name` and `Execution`.

## How to commit offline-built records when back online

Reconnect to the catalog and create a new `Execution`. Pass the list of offline-built `FeatureRecord` objects to `exe.add_features()`. The execution records provenance for every staged value.

```python
from deriva_ml import DerivaML
from deriva_ml.execution import ExecutionConfiguration

# Back online
ml = DerivaML(hostname="example.org", catalog_id="42")

workflow = ml.find_workflows()[0]
cfg = ExecutionConfiguration(
    description="Offline inference results",
    workflow=workflow,
)

with ml.create_execution(cfg) as exe:
    count = exe.add_features(pending_records)
    print(f"Staged {count} records")

# Records are flushed to the catalog after the context manager exits
exe.upload_execution_outputs()
```

`exe.add_features()` stages records in the execution's local SQLite state. They are flushed to ERMrest in a single batch after asset upload, when the execution completes. If the process crashes before upload, call `ml.resume_execution(execution.execution_rid)` and re-run `upload_execution_outputs()`.

**Notes**

- All records passed to a single `exe.add_features()` call must share the same feature definition (same table and feature name). Mixing features raises `DerivaMLValidationError`.
- `exe.add_features()` returns the count of staged records, not a list of RIDs. RIDs are assigned when the records land in ERMrest.
- The offline-built records are plain Pydantic objects. You can serialize them with `.model_dump()`, write them to a file, and reload them later — the catalog does not know about them until `exe.add_features()` is called.

## How to restructure assets for ML frameworks

PyTorch `ImageFolder` and Keras `image_dataset_from_directory` both expect a `class/image.jpg` directory tree. `bag.restructure_assets()` builds that tree from a downloaded bag, using dataset types and feature labels as directory names.

```python
from deriva_ml.feature import FeatureRecord

manifest = bag.restructure_assets(
    output_dir="./ml_data",
    group_by=["Glaucoma"],
)
# Produces:
# ./ml_data/training/Normal/retina_001.jpg  -> symlink
# ./ml_data/training/Severe/retina_002.jpg  -> symlink
# ./ml_data/testing/Normal/retina_099.jpg   -> symlink
```

The method returns `dict[Path, Path]` — a manifest mapping each source path to the output path that was written. It does **not** return a `Path`.

```python
# Correct: manifest is a dict
manifest = bag.restructure_assets(output_dir="./ml_data", group_by=["Glaucoma"])
for src, dst in manifest.items():
    print(f"{src.name} -> {dst}")

# Wrong: restructure_assets does not return a Path
output_path = bag.restructure_assets(...)  # this is a dict, not a Path
```

### Full signature

```python
manifest = bag.restructure_assets(
    output_dir="./ml_data",
    asset_table="Image",           # auto-detected if omitted and only one asset table exists
    group_by=["Glaucoma"],         # column names or feature names; creates subdirectory levels
    use_symlinks=True,             # True (default) = symlinks; False = copies
    type_selector=None,            # callable(list[str]) -> str for multi-type datasets
    type_to_dir_map={              # map dataset type names to directory names
        "Training": "training",
        "Testing": "testing",
    },
    enforce_vocabulary=True,       # only allow vocabulary-based features in group_by
    value_selector=None,           # callable(list[FeatureRecord]) -> FeatureRecord for multi-value features
    file_transformer=None,         # callable(src: Path, dest: Path) -> Path for format conversion
)  # returns dict[Path, Path]
```

### Directory structure

Top-level subdirectories come from dataset types (`Training` → `training`, `Testing` → `testing`). Below that, each entry in `group_by` adds one level of subdirectories named after the label value.

```
./ml_data/
  training/
    Normal/
      retina_001.jpg
    Severe/
      retina_002.jpg
    Unknown/            # assets with no label for this feature
      retina_003.jpg
  testing/
    Normal/
      retina_099.jpg
```

### Using features as group_by keys

If a `group_by` name matches a feature defined on the asset table (or a table it references via FK), `restructure_assets` reads the feature values from the bag's SQLite cache and uses the vocabulary term as the directory name.

```python
# Group by a feature defined on Image
manifest = bag.restructure_assets(
    output_dir="./ml_data",
    group_by=["Glaucoma"],          # feature name
)

# Group by a specific column of a multi-term feature
manifest = bag.restructure_assets(
    output_dir="./ml_data",
    group_by=["Classification.Label"],  # FeatureName.column_name
)
```

Column names on the asset record are checked first; feature names are checked if no column matches.

### Handling multiple feature values

When an asset has more than one value for the same feature (e.g., two annotators disagreed), `restructure_assets` raises `DerivaMLException` by default. Provide a `value_selector` to resolve the conflict:

```python
manifest = bag.restructure_assets(
    output_dir="./ml_data",
    group_by=["Glaucoma"],
    value_selector=FeatureRecord.select_newest,
)
```

`FeatureRecord.select_newest` picks the record with the latest creation timestamp. Other built-in selectors: `FeatureRecord.select_first`. For majority voting, use `FeatureRecord.select_majority_vote("column_name")` — it is a `@classmethod` factory that takes a column name and returns a selector; pass the result as `value_selector=FeatureRecord.select_majority_vote("Glaucoma")`.

Set `enforce_vocabulary=False` to silently use the first value when multiple exist without raising.

### Prediction scenarios

Datasets with no type defined are treated as `Testing`. Assets with no label for a `group_by` key land in `Unknown/`. This allows `restructure_assets` to run on unlabeled inference data:

```python
# Unlabeled prediction dataset — no types, no Glaucoma labels
manifest = bag.restructure_assets(
    output_dir="./predictions",
    group_by=["Glaucoma"],
)
# ./predictions/testing/Unknown/retina_new_001.jpg
```

### Using symlinks vs copies

`use_symlinks=True` (the default) creates symbolic links back to the files inside the bag directory. This saves disk space but requires the bag directory to remain in place. If you need a fully portable output tree, pass `use_symlinks=False` to copy the files instead.

### Format conversion with file_transformer

Pass a `file_transformer` callable to convert files during restructuring. The transformer receives the source path and the suggested destination path; it must write the output and return the actual path written (which may differ in extension).

```python
from PIL import Image as PILImage

def dicom_to_png(src: Path, dest: Path) -> Path:
    img = load_dicom(str(src))
    out = dest.with_suffix(".png")
    PILImage.fromarray(img).save(out)
    return out

manifest = bag.restructure_assets(
    output_dir="./ml_data",
    group_by=["Diagnosis"],
    file_transformer=dicom_to_png,
)
# manifest maps: Path(".../bag/.../scan.dcm") -> Path("./ml_data/training/Normal/scan.png")
```

When `file_transformer` is provided, `use_symlinks` is ignored.

## Common pitfalls

!!! warning "restructure_assets returns a dict, not a Path"
    `bag.restructure_assets()` returns `dict[Path, Path]` — a manifest mapping source paths to output paths. It does **not** return the output directory as a `Path`. Iterating the result without checking will silently operate on the dict keys (source paths) rather than the directory you expect.

!!! warning "materialize=False means no asset files"
    A bag downloaded with `materialize=False` contains only CSV metadata. Calling `bag.path` succeeds, but asset files under `data/asset/` are absent. `restructure_assets` will log warnings and produce an empty manifest. If you need to read actual files offline, re-download with `materialize=True`.

!!! warning "Bags are immutable snapshots"
    If the source dataset changes in the catalog — new members added, version incremented — your bag does not update automatically. Re-download with the new version to get fresh data. Holding a stale bag while comparing feature values from a live catalog will produce silent mismatches.

## See also

- [Working with datasets](datasets.md) — Chapter 2: creating datasets, downloading bags, `estimate_bag_size`, version pinning.
- [Defining and using features](features.md) — Chapter 3: creating features, `feature_values()` selectors, multi-annotator patterns.
- [Running an experiment](executions.md) — Chapter 4: `ExecutionConfiguration`, `exe.add_features()`, `upload_execution_outputs()`.
- Chapter 7 — Reproducibility (coming soon): `DatasetSpec` version pinning, catalog snapshots.
- API reference: [`DatasetBag`](../api-reference/dataset_bag.md)
