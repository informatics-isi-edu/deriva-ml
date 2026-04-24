# Chapter 2: Working with Datasets

This chapter covers dataset creation, membership, versioning, splitting, and downloading as BDBag archives. By the end you will know how to build a versioned dataset hierarchy, split it for training and evaluation, and download it for local use.

## What a dataset is

A dataset is a versioned, named collection of RIDs. It does not copy data — it points at existing rows in the catalog. When you add a `Subject` or an `Image` to a dataset, the catalog records the association between the dataset and that row's RID. The underlying record stays in its own table; the dataset is just a named view over a subset of those records.

Datasets can be heterogeneous. A single dataset can contain members from multiple tables — for example, subjects and the images linked to those subjects. DerivaML manages the FK relationships and exports all reachable data when you download.

Every dataset has a semantic version (`major.minor.patch`) and a catalog snapshot tied to each version. The snapshot is the key to reproducibility: loading version `1.2.0` of a dataset always returns the same rows it contained when that version was created, regardless of what has changed in the catalog since then. See Chapter 7 ("Reproducibility") for the full version-pinning story.

## How to create a dataset

Creating a dataset within an execution context gives you full provenance tracking. The execution records which workflow produced the dataset and when.

```python
from deriva_ml import DerivaML
from deriva_ml.execution import ExecutionConfiguration

ml = DerivaML(hostname="catalog.example.org", catalog_id="1")

# Create a workflow to anchor the execution
workflow = ml.create_workflow(
    name="Data Preparation",
    workflow_type="Data Processing",
    description="Prepares training and validation splits",
)

config = ExecutionConfiguration(
    workflow=workflow,
    description="Create training dataset",
)

with ml.create_execution(config) as exe:
    training_dataset = exe.create_dataset(
        dataset_types=["TrainingSet"],
        description="Images for model training",
    )
    validation_dataset = exe.create_dataset(
        dataset_types=["ValidationSet"],
        description="Images for model evaluation",
    )

exe.upload_execution_outputs()
```

`exe.create_dataset()` accepts `dataset_types` (a string or list of strings from the `Dataset_Type` vocabulary) and a `description`. It returns a `Dataset` object bound to the live catalog connection.

**Notes**

- `dataset_types` must be terms already present in the `Dataset_Type` vocabulary. Use `ml.add_term(MLVocab.dataset_type, "TrainingSet", ...)` to register new terms before creating datasets that use them.
- Omit `dataset_types` or pass `None` to create a dataset with no type — useful for prediction/inference datasets where no category applies yet.
- The returned `Dataset` object holds a live connection. Keep the `DerivaML` instance alive as long as you need to call methods on it.

## How to design dataset types

Dataset types come from the `Dataset_Type` controlled vocabulary. They are labels, not a classification hierarchy — a single dataset can have more than one type at the same time.

```python
from deriva_ml import MLVocab

ml.add_term(MLVocab.dataset_type, "TrainingSet",
            description="Dataset used for model training")
ml.add_term(MLVocab.dataset_type, "Labeled",
            description="Members have ground-truth annotations")
ml.add_term(MLVocab.dataset_type, "ValidationSet",
            description="Dataset used during training for hyperparameter selection")
```

Because types are orthogonal, you can create a labeled training set by combining both types:

```python
with ml.create_execution(config) as exe:
    labeled_train = exe.create_dataset(
        dataset_types=["TrainingSet", "Labeled"],
        description="Labeled training images",
    )
```

Design types around the questions you will ask later — for example, "give me all datasets that are both `TrainingSet` and `Labeled`" — rather than trying to encode a rigid taxonomy.

**Notes**

- Types are checked against the `Dataset_Type` vocabulary at insert time. Misspellings raise `DerivaMLInvalidTerm`.
- There is no constraint requiring a dataset to have exactly one type. Use as many (or as few) as describe the dataset's role.

## How to add members to a dataset

Use `dataset.add_dataset_members()` to associate records with a dataset. The method accepts two forms:

**List of RIDs** — simpler, but slower for large datasets because DerivaML must resolve each RID to determine which table it belongs to:

```python
# Assumes: subject_rids is a list of RID strings from the Subject table
dataset.add_dataset_members(members=subject_rids)
```

**Dict mapping table name to RIDs** — faster for large datasets; no resolution step needed:

```python
dataset.add_dataset_members(
    members={
        "Subject": subject_rids,
        "Image": image_rids,
    }
)
```

Both forms trigger a minor version increment on the dataset. To link the addition to the execution that created the data, pass `execution_rid`:

```python
dataset.add_dataset_members(
    members={"Image": image_rids},
    execution_rid=exe.execution_rid,
)
```

Before members can be added from a table, that table must be registered as a dataset element type. Check what types are available:

```python
element_types = ml.list_dataset_element_types()
for table in element_types:
    print(table.name)
```

To register a new element type:

```python
ml.add_dataset_element_type("Subject")
```

**Notes**

- Use the dict form whenever you know which table the RIDs come from. The list form issues an extra catalog query per batch to identify tables.
- Each call to `add_dataset_members` automatically increments the dataset's minor version. If you need to add members in multiple batches without creating intermediate versions, consider collecting all RIDs first and calling `add_dataset_members` once.
- Members can only come from tables registered with `add_dataset_element_type`. Attempting to add a RID from an unregistered table raises `DerivaMLException`.

## Parent and child datasets

Datasets can contain other datasets as members, forming a hierarchy. This is the standard way to group related splits or sub-collections under a parent.

```python
# Create the parent that will hold both partitions
parent = exe.create_dataset(
    dataset_types=["ExperimentCollection"],
    description="Full experiment — all partitions",
)

# Create two child datasets
train = exe.create_dataset(dataset_types=["TrainingSet"],
                           description="Training partition")
test = exe.create_dataset(dataset_types=["TestingSet"],
                          description="Testing partition")

# Add image members to each child
train.add_dataset_members(members={"Image": train_image_rids})
test.add_dataset_members(members={"Image": test_image_rids})

# Nest the children under the parent by adding their RIDs as members
parent.add_dataset_members(
    members=[train.dataset_rid, test.dataset_rid]
)
```

Navigate the hierarchy with `list_dataset_children()` and `list_dataset_parents()`:

```python
children = parent.list_dataset_children()
parents = train.list_dataset_parents()
```

Bag export honors the hierarchy: downloading a parent dataset includes all records reachable from every child. See "How to download a dataset as a bag" below.

!!! note
    DerivaML detects and prevents cycles in the parent/child graph. Attempting to add a dataset as a member of one of its own descendants raises `DerivaMLCycleError`.

## How to version a dataset

Every dataset starts at version `0.1.0` and increments its minor version each time `add_dataset_members` is called. You can also increment a version explicitly and control which component changes.

```python
from deriva_ml.dataset import VersionPart

# Read the current version
print(dataset.current_version)  # e.g., DatasetVersion(0, 3, 0)

# View the version history
for entry in dataset.dataset_history():
    print(f"v{entry.dataset_version}: {entry.description} ({entry.snapshot})")

# Bump to 1.0.0 when the dataset is stable for a training run
new_version = dataset.increment_dataset_version(
    component=VersionPart.major,
    description="Stable release for experiment 1",
)
print(new_version)  # DatasetVersion(1, 0, 0)
```

`increment_dataset_version` propagates the bump through the parent/child graph using a topological sort, so all related datasets move to consistent version numbers together.

Each version is tied to a catalog snapshot. To read members as they existed at a specific version, pass `version=` to `list_dataset_members()` or download the versioned bag:

```python
# List members as they existed at v1.0.0 (uses the catalog snapshot for that version)
members = dataset.list_dataset_members(version="1.0.0")

# Or download the versioned bag for fully offline use
versioned_bag = dataset.download_dataset_bag(version="1.0.0")
```

This is the guarantee that makes dataset downloads reproducible. Downloading a versioned dataset always returns the same rows.

**Notes**

- Every call to `add_dataset_members` unconditionally bumps the minor version once. There is no parameter to suppress this. If you want fewer version bumps, **batch all members into a single `add_dataset_members` call** rather than making multiple small calls.
- To mark a milestone (major or patch version) without adding members, call `increment_dataset_version` explicitly.
- `VersionPart` lives in `deriva_ml.dataset.aux_classes` alongside `DatasetVersion` and related types, and is re-exported from `deriva_ml.dataset` for convenience.
- Version pinning for executions (`DatasetSpec(rid=..., version="1.0.0")`) is covered in Chapter 7 ("Reproducibility").

## How to split a dataset

`split_dataset` divides a dataset into training, testing, and optionally validation subsets, tracking the entire operation within an execution for provenance.

The result is a parent `Split` dataset with child datasets for each partition:

```
Split (parent, type: "Split")
├── Training (child, type: "Training")
├── Validation (child, type: "Validation")  ← only when val_size is set
└── Testing (child, type: "Testing")
```

### Simple random split

```python
from deriva_ml.dataset.split import split_dataset

# 80/20 split, reproducible with seed=42
result = split_dataset(ml, source_dataset_rid, test_size=0.2, seed=42)

print(f"Split RID:    {result.split.rid} (v{result.split.version})")
print(f"Training:     {result.training.rid} ({result.training.count} samples)")
print(f"Testing:      {result.testing.rid} ({result.testing.count} samples)")
```

### Three-way split

Add `val_size` to create a validation partition:

```python
# 70/10/20 train/val/test
result = split_dataset(
    ml, source_dataset_rid,
    test_size=0.2,
    val_size=0.1,
    seed=42,
)

print(f"Training:   {result.training.count} samples")
print(f"Validation: {result.validation.count} samples")
print(f"Testing:    {result.testing.count} samples")
```

When `val_size` is `None` (the default), `result.validation` is `None` and only two child datasets are created.

### Stratified split

Stratified splitting preserves class distribution across all partitions. Pass `stratify_by_column` using the denormalized dot-notation format `{TableName}.{ColumnName}`:

```python
result = split_dataset(
    ml, source_dataset_rid,
    test_size=0.2,
    val_size=0.1,
    seed=42,
    stratify_by_column="Image_Classification.Image_Class",
    include_tables=["Image", "Image_Classification"],
)
```

The `stratify_by_column` format uses a **period as separator** between table name and column name (`Image_Classification.Image_Class`), not an underscore. The `include_tables` parameter controls which tables are joined during denormalization to access the column.

!!! note
    Stratified splitting requires scikit-learn. It is imported lazily — `split_dataset` works without it for random splits.

### Labeled partitions

When all partitions need ground-truth labels (for example, ROC-curve evaluation on the test set), add the `"Labeled"` type to each partition:

```python
result = split_dataset(
    ml, source_dataset_rid,
    test_size=0.2,
    val_size=0.1,
    training_types=["Labeled"],
    validation_types=["Labeled"],
    testing_types=["Labeled"],
)
```

Each partition gets its default type (e.g., `"Training"`) plus the additional types you specify.

### Dry run

Preview split sizes without writing to the catalog:

```python
result = split_dataset(
    ml, source_dataset_rid,
    test_size=0.2,
    val_size=0.1,
    dry_run=True,
)
print(f"Would create: train={result.training.count}, "
      f"val={result.validation.count}, test={result.testing.count}")
print(f"Strategy: {result.strategy}")
```

### Command-line interface

The same functionality is available as `deriva-ml-split-dataset`:

```bash
# Simple two-way split
deriva-ml-split-dataset --hostname localhost --catalog-id 9 \
    --dataset-rid 28D0 --test-size 0.2

# Three-way split
deriva-ml-split-dataset --hostname localhost --catalog-id 9 \
    --dataset-rid 28D0 --test-size 0.2 --val-size 0.1

# Stratified split
deriva-ml-split-dataset --hostname localhost --catalog-id 9 \
    --dataset-rid 28D0 --test-size 0.2 \
    --stratify-by-column Image_Classification.Image_Class \
    --include-tables Image,Image_Classification

# Dry run
deriva-ml-split-dataset --hostname localhost --catalog-id 9 \
    --dataset-rid 28D0 --dry-run
```

**Notes**

- When the source dataset has members in only one element table, `split_dataset` auto-detects which table to split. If the dataset has multiple element types, specify `element_table="Image"` explicitly.
- `split_dataset` creates a new execution internally for provenance tracking. You do not need to wrap it in your own `create_execution` context.
- `result.split.rid` is the RID of the parent `Split` dataset. Use this RID in `DatasetSpec` when you want an execution to download the full hierarchy.

## How to download a dataset as a bag

DerivaML downloads datasets as BDBag archives — a standard format for reproducible data packaging. The download captures all rows reachable from the dataset's members (by following FK relationships), plus remote file references for any associated assets.

```python
# Download current version, materialize asset files
bag = dataset.download_dataset_bag(
    version=dataset.current_version,
    materialize=True,
)
print(f"Downloaded to {bag.path}")
print(f"RID: {bag.dataset_rid}, version: {bag.current_version}")
```

`materialize=True` (the default) fetches the actual asset files from the Hatrac object store in addition to downloading the table data. `materialize=False` downloads only the table rows and a reference manifest — useful when you only need metadata, not the files themselves:

```python
# Metadata only — fast, no file downloads
bag = dataset.download_dataset_bag(
    version="1.0.0",
    materialize=False,
)
```

Use `materialize=False` when:

- You are validating that a dataset version has the expected members before committing to a full download.
- You are running on a machine where the asset files will be fetched by a separate step.
- Asset files are large and you only need the tabular data.

!!! warning
    With `materialize=False`, the bag contains remote references but no local copies of asset files. Attempting to read asset file paths from such a bag will fail. Materialize the bag first, or use the bag's `path` attribute to locate the fetch manifest and download assets separately.

### Estimating download size

Before downloading, check the size to decide whether you need a longer timeout:

```python
from deriva_ml.dataset.aux_classes import DatasetSpec

spec = DatasetSpec(rid=dataset.dataset_rid, version="1.0.0")
estimate = ml.estimate_bag_size(spec)

print(f"Total rows: {estimate['total_rows']}")
print(f"Total asset size: {estimate['total_asset_size']}")  # e.g., "3.4 GB"
for table_name, info in estimate["tables"].items():
    print(f"  {table_name}: {info['row_count']} rows", end="")
    if info["is_asset"]:
        print(f", {info['asset_bytes']} bytes", end="")
    print()
```

This queries the catalog snapshot without downloading any data.

### Timeouts for large datasets

Deep FK joins can exceed the server's default timeout. Override with `timeout=(connect_seconds, read_seconds)`:

```python
bag = dataset.download_dataset_bag(
    version="1.0.0",
    timeout=(10, 1800),  # 30-minute read timeout
)
```

You can also set the timeout in a `DatasetSpec` for execution configurations:

```python
from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.execution import ExecutionConfiguration

config = ExecutionConfiguration(
    datasets=[
        DatasetSpec(rid="1-abc123", version="1.0.0", timeout=(10, 1800)),
    ],
    workflow=workflow,
)
```

### Downloading inside an execution

When datasets are specified in `ExecutionConfiguration`, download them with `exe.download_dataset_bag()`:

```python
config = ExecutionConfiguration(
    datasets=[DatasetSpec(rid="1-abc123", version="1.0.0")],
    workflow=workflow,
)

with ml.create_execution(config) as exe:
    bag = exe.download_dataset_bag(DatasetSpec(rid="1-abc123", version="1.0.0"))
    # bag.path is now a local directory with CSV exports and asset references
```

### What the bag contains

After download, the `DatasetBag` object provides immediate access to metadata:

```python
print(f"RID: {bag.dataset_rid}")
print(f"Version: {bag.current_version}")
print(f"Local path: {bag.path}")

# Load a table as a DataFrame
subjects_df = bag.get_table_as_dataframe("Subject")
images_df = bag.get_table_as_dataframe("Image")
```

For reading the full bag offline — `feature_values()`, `list_dataset_members()`, and `restructure_assets()` — see Chapter 5 ("Working offline").

### How bag export traverses foreign keys

The exporter follows FK relationships outward from each member table to include related rows (vocabulary terms, device records, and so on). It stops when it reaches another **dataset element type** — a table that has its own `Dataset_X` association — if that element type has no members in this dataset.

For example, if the schema has `CGM_Blood_Glucose → Observation → Image`, and your dataset contains only `CGM_Blood_Glucose` records, the exporter does not follow the path through `Observation → Image` because `Observation` is itself an element type with no members in this dataset. This boundary-aware traversal avoids expensive joins that would return empty results.

Non-element-type tables (such as `Device`) are always traversed normally.

**Notes**

- `download_dataset_bag` writes the bag to a temporary directory inside the working directory. The path is stable for the lifetime of the `DatasetBag` object.
- To exclude a table that causes timeout issues, use `DatasetSpec(exclude_tables={"ProcessStep"})`.
- To create a MINID (a persistent, citable identifier) for a shared bag, pass `use_minid=True` to `download_dataset_bag`. MINIDs are covered in depth in Chapter 6 ("Sharing and collaboration").

## Common pitfalls

!!! warning
    **`stratify_by_column` uses dot notation, not underscore.**

    The column name must be in the format `TableName.ColumnName` (e.g., `Image_Classification.Image_Class`). A common mistake is using the underscore-joined form (`Image_Classification_Image_Class`), which is the denormalized DataFrame column name — not the argument to `stratify_by_column`.

!!! warning
    **Version numbers do not update automatically when members change.**

    `add_dataset_members` auto-increments the minor version as a convenience, but if you never call `add_dataset_members` (for example, you added members in a previous session and now want to mark the dataset as stable), you must call `increment_dataset_version()` explicitly. The version counter is not driven by catalog state — it only advances when you tell it to.

!!! warning
    **`materialize=False` gives you table metadata, not asset files.**

    Downloading with `materialize=False` is fast but produces a bag whose asset paths are remote references, not local files. Any code that reads asset file content — including `restructure_assets()` — will fail until the bag is materialized.

## See also

- Chapter 1 ("Exploring a catalog") — `find_datasets()`, `lookup_dataset()`, and RID basics.
- [Chapter 3 ("Defining and using features")](features.md) — attaching provenance-linked annotations to dataset members.
- Chapter 4 ("Running an experiment") — `ExecutionConfiguration`, `DatasetSpec`, and downloading datasets inside an execution.
- Chapter 5 ("Working offline") — `DatasetBag` API, `get_table_as_dataframe()`, `feature_values()`, and `restructure_assets()`.
- Chapter 7 ("Reproducibility") — version pinning via catalog snapshots; `DatasetSpec(version=...)` in execution configs.
- API reference: [`deriva_ml.dataset.dataset.Dataset`](../api-reference/dataset.md), [`deriva_ml.dataset.split.split_dataset`](../api-reference/dataset_split.md), [`deriva_ml.dataset.aux_classes.DatasetSpec`](../api-reference/dataset_aux_classes.md).
