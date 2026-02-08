# Datasets

When working with ML models, it is often convenient to collect various input data into named, identifiable collections called datasets. DerivaML provides a flexible set of mechanisms for creating and manipulating datasets.

A dataset is a versioned collection of objects within a DerivaML catalog. Datasets can be heterogeneous, containing sets of different object types. It's possible to have an object referenced in multiple ways - for example, a collection of subjects and a collection of observations that reference those subjects. DerivaML manages these relationships and makes it possible to view them from all paths.

As with any other object in DerivaML, each dataset is identified by its *Resource Identifier* (RID). In addition, a dataset may have one or more dataset types and a version.

## Dataset Types

Dataset types are assigned from a controlled vocabulary called `Dataset_Type`. You can define new dataset types as needed:

```python
from deriva_ml import MLVocab

ml.add_term(MLVocab.dataset_type, "TrainingSet", description="Dataset for model training")
ml.add_term(MLVocab.dataset_type, "ValidationSet", description="Dataset for model validation")
```

When you create a dataset, you can provide as many dataset types as required to streamline organizing and discovering them in your code.

## Creating Datasets

The most common way to create a dataset is within an execution, which provides provenance tracking:

```python
from deriva_ml import DerivaML
from deriva_ml.execution import ExecutionConfiguration

# Connect to the catalog
ml = DerivaML("deriva.example.org", "my_catalog")

# Create a workflow
workflow = ml.create_workflow(
    name="Data Preparation Workflow",
    workflow_type="Data Processing",
    description="Prepares training and validation datasets"
)

# Create an execution configuration
config = ExecutionConfiguration(
    workflow=workflow,
    description="Create datasets for experiment"
)

# Create datasets within the execution context
with ml.create_execution(config) as exe:
    training_dataset = exe.create_dataset(
        dataset_types=["TrainingSet", "Image"],
        description="Training images for classification model"
    )

    validation_dataset = exe.create_dataset(
        dataset_types=["ValidationSet", "Image"],
        description="Validation images for model evaluation"
    )

# Upload any outputs after context exits
exe.upload_execution_outputs()
```

You can also create datasets directly through the DerivaML instance:

```python
dataset = ml.create_dataset(
    dataset_types=["ExperimentData"],
    description="Raw experimental data"
)
```

## Dataset Element Types

A dataset may consist of many different types of objects. In general, any element from the domain model may be included in a dataset. To see what element types are available:

```python
# List available element types
element_types = dataset.list_dataset_element_types()
for table in element_types:
    print(table.name)
```

To add a new element type that can be included in datasets:

```python
# Add Subject table as a valid dataset element type
ml.add_dataset_element_type("Subject")
```

## Adding Members to a Dataset

Once you have a dataset, you can add members using their RIDs:

```python
# Add individual members by RID
dataset.add_dataset_members(members=["1-abc123", "1-def456", "1-ghi789"])

# Add members with execution tracking
dataset.add_dataset_members(
    members=subject_rids,
    execution_rid=execution.execution_rid
)
```

## Listing Dataset Members

To see what's in a dataset:

```python
# List all members of current version
members = dataset.list_dataset_members()
for member in members:
    print(f"Table: {member['table']}, RID: {member['rid']}")

# List members of a specific version
members_v1 = dataset.list_dataset_members(version="1.0.0")
```

## Nested Datasets

Datasets can contain other datasets, forming hierarchies:

```python
# Create a parent dataset
parent_dataset = ml.create_dataset(
    dataset_types=["ExperimentCollection"],
    description="Collection of experiment datasets"
)

# Add child datasets as members
parent_dataset.add_dataset_members(
    members=[training_dataset.dataset_rid, validation_dataset.dataset_rid]
)

# List child datasets
children = parent_dataset.list_dataset_children()

# List parent datasets
parents = training_dataset.list_dataset_parents()
```

## Splitting Datasets

A common ML workflow is splitting a dataset into training and testing subsets. DerivaML provides the [`split_dataset`][deriva_ml.dataset.split.split_dataset] function for this, with full provenance tracking. The API follows scikit-learn conventions (`test_size`, `train_size`, `shuffle`, `seed`, `stratify`) while creating a proper dataset hierarchy in the catalog.

### How Splitting Works

`split_dataset` creates a three-level dataset hierarchy:

```
Split (parent, type: "Split")
├── Training (child, type: "Training")
└── Testing (child, type: "Testing")
```

The entire operation is performed within an execution context, so the split is fully traceable back to the source dataset, the parameters used, and the code that ran it.

### Simple Random Split

The simplest case splits a dataset into training and testing subsets by randomly shuffling members:

```python
from deriva_ml.dataset.split import split_dataset

# 80/20 random split (default)
result = split_dataset(ml, source_dataset_rid, test_size=0.2, seed=42)

print(f"Split:    {result['split']}")
print(f"Training: {result['training']} ({result['train_count']} samples)")
print(f"Testing:  {result['testing']} ({result['test_count']} samples)")
```

You can also specify absolute counts instead of fractions:

```python
# Fixed-count split
result = split_dataset(
    ml, source_dataset_rid,
    train_size=400,
    test_size=100,
    seed=42,
)
```

### Labeled Splits

When your experiment needs ground truth labels in both training and testing sets (for evaluation, ROC curves, etc.), add the `"Labeled"` dataset type:

```python
result = split_dataset(
    ml, source_dataset_rid,
    test_size=0.2,
    seed=42,
    training_types=["Labeled"],
    testing_types=["Labeled"],
)
```

This creates Training and Testing datasets with both their default type and the additional `"Labeled"` type, making them easy to discover and distinguish from unlabeled splits.

### Stratified Splitting

Stratified splitting maintains the class distribution of a column across both splits. This requires denormalizing the dataset to access the column values:

```python
result = split_dataset(
    ml, source_dataset_rid,
    test_size=0.2,
    seed=42,
    stratify_by_column="Image_Classification_Image_Class",
    include_tables=["Image", "Image_Classification"],
)
```

The `stratify_by_column` uses the denormalized column name format: `{TableName}_{ColumnName}`. The `include_tables` parameter specifies which tables to join during denormalization.

!!! note
    Stratified splitting requires scikit-learn to be installed. It is imported lazily, so the base `split_dataset` function works without it for random splits.

### Custom Selection Functions

For advanced splitting logic (balanced sampling, filtered subsets, etc.), provide a custom selection function:

```python
import numpy as np

def balanced_selector(df, train_size, test_size, seed):
    """Select equal numbers from each class."""
    rng = np.random.default_rng(seed)
    label_col = "Image_Classification_Image_Class"
    classes = df[label_col].unique()
    train_idx, test_idx = [], []
    for cls in classes:
        cls_indices = df.index[df[label_col] == cls].to_numpy()
        rng.shuffle(cls_indices)
        per_class_train = train_size // len(classes)
        per_class_test = test_size // len(classes)
        train_idx.extend(cls_indices[:per_class_train])
        test_idx.extend(cls_indices[per_class_train:per_class_train + per_class_test])
    return np.array(train_idx), np.array(test_idx)

result = split_dataset(
    ml, source_dataset_rid,
    test_size=100,
    selection_fn=balanced_selector,
    include_tables=["Image", "Image_Classification"],
)
```

A selection function must conform to the [`SelectionFunction`][deriva_ml.dataset.split.SelectionFunction] protocol: it receives a DataFrame, train/test sizes, and a seed, and returns `(train_indices, test_indices)` as numpy arrays.

### Dry Run

Use `dry_run=True` to preview what would happen without modifying the catalog:

```python
result = split_dataset(
    ml, source_dataset_rid,
    test_size=0.2,
    dry_run=True,
)
print(f"Would create: {result['train_count']} train, {result['test_count']} test")
print(f"Strategy: {result['strategy']}")
```

### Command-Line Interface

The `deriva-ml-split-dataset` CLI provides the same functionality from the command line:

```bash
# Simple random split
deriva-ml-split-dataset --hostname localhost --catalog-id 9 \
    --dataset-rid 28D0 --test-size 0.2

# Stratified split
deriva-ml-split-dataset --hostname localhost --catalog-id 9 \
    --dataset-rid 28D0 --test-size 0.2 \
    --stratify-by-column Image_Classification_Image_Class \
    --include-tables Image,Image_Classification

# Dry run
deriva-ml-split-dataset --hostname localhost --catalog-id 9 \
    --dataset-rid 28D0 --dry-run
```

### Auto-Detection

When the source dataset has members in only one element table, `split_dataset` auto-detects which table to split. If the dataset has members in multiple tables, you must specify `element_table`:

```python
result = split_dataset(
    ml, source_dataset_rid,
    test_size=0.2,
    element_table="Image",  # Required when dataset has multiple element types
)
```

## Dataset Versioning

Every dataset is assigned a version number using semantic versioning (`major.minor.patch`):

- **Major**: Changes when there is a schema change to any object in the dataset
- **Minor**: Changes when new elements are added to a dataset
- **Patch**: Changes for minor alterations, such as adding comments or data cleaning

DerivaML automatically assigns version `0.1.0` when a dataset is first created and increments the minor part whenever new elements are added.

### Working with Versions

```python
# Get current version
current = dataset.current_version
print(f"Current version: {current}")  # e.g., "1.2.3"

# Get version history
history = dataset.dataset_history()
for entry in history:
    print(f"Version {entry.version}: {entry.description} at {entry.timestamp}")

# Increment version
from deriva_ml.dataset.aux_classes import VersionPart

new_version = dataset.increment_dataset_version(
    component=VersionPart.minor,
    description="Added new training samples"
)
```

### Version Snapshots

Each version is tied to a catalog snapshot, ensuring that the values in the dataset are the values that were present when the version was created. This provides reproducibility for ML experiments.

```python
# Get a dataset bound to a specific version
versioned_dataset = dataset.set_version("1.0.0")

# Access members at that version
members = versioned_dataset.list_dataset_members()
```

## Downloading Datasets

Datasets can be downloaded as BDBag archives for offline use or sharing:

```python
from deriva_ml.dataset.aux_classes import DatasetSpec

# Download the current version
bag = dataset.download_dataset_bag()

# Download a specific version
bag = dataset.download_dataset_bag(version="1.0.0")

# Download with materialization (fetches all referenced files)
bag = dataset.download_dataset_bag(materialize=True)
```

### Automatic Download in Executions

When creating an execution with dataset specifications, you can download datasets within the execution context:

```python
from deriva_ml.dataset import DatasetSpec

config = ExecutionConfiguration(
    datasets=[
        DatasetSpec(rid="1-abc123", version="1.0.0"),
        DatasetSpec(rid="1-def456", materialize=True),
    ],
    workflow=workflow,
    description="Process datasets"
)

with ml.create_execution(config) as exe:
    # Download datasets as needed
    bag = exe.download_dataset_bag(DatasetSpec(rid="1-abc123"))
    print(f"Dataset available at {bag.bag_path}")
```

## Working with DatasetBag

Once downloaded, a dataset is represented as a [`DatasetBag`][deriva_ml.dataset.dataset_bag.DatasetBag] object:

```python
# Access dataset metadata
print(f"RID: {bag.dataset_rid}")
print(f"Version: {bag.version}")

# Get tables as DataFrames
subjects_df = bag.get_table_as_dataframe("Subject")
images_df = bag.get_table_as_dataframe("Image")

# Access the local path
print(f"Dataset path: {bag.path}")
```

## Assets in Datasets

Assets are files stored in the DerivaML object store (Hatrac). Each asset is characterized by:

- A versioned URL
- Length (file size)
- MD5 checksum
- Asset type (from the `Asset_Type` vocabulary)

### Accessing Assets

```python
# Get asset table from a downloaded dataset bag
assets = bag.get_table_as_dataframe("Image")

# Assets have a local path after materialization
for _, asset in assets.iterrows():
    local_path = asset["Filename"]
    print(f"Asset: {local_path}")
```

### Asset Organization

When datasets are materialized:

- All assets of the same type are placed in the same directory
- The directory is named by the asset type
- Use metadata to identify specific assets rather than relying on filenames

If you need to reorganize assets for your application, using symbolic links is efficient for both time and disk space.

### Restructuring Assets for ML Workflows

Many ML frameworks expect training data organized in a specific directory structure (e.g., `train/class1/`, `train/class2/`). The `restructure_assets()` method reorganizes downloaded assets into a hierarchical directory structure based on dataset types and metadata or feature values.

```python
from pathlib import Path

# Download a dataset with nested structure
bag = dataset.download_dataset_bag()

# Restructure images by dataset type and a label column/feature
bag.restructure_assets(
    asset_table="Image",
    output_dir=Path("./ml_data"),
    group_by=["label"],
)
```

This creates a directory structure like:

```
ml_data/
  Complete/           # Parent dataset type
    Training/         # Nested dataset type
      positive/       # Label value
        image1.jpg
        image2.jpg
      negative/
        image3.jpg
    Testing/
      positive/
        image4.jpg
      negative/
        image5.jpg
```

#### Grouping Options

The `group_by` parameter accepts column names from the asset table or feature names:

```python
# Group by a column in the asset table
bag.restructure_assets(
    asset_table="Image",
    output_dir=Path("./by_subject"),
    group_by=["Subject"],  # Foreign key column
)

# Group by a feature attached to the asset
bag.restructure_assets(
    asset_table="Image",
    output_dir=Path("./by_quality"),
    group_by=["Quality"],  # Feature name
)

# Multiple grouping levels
bag.restructure_assets(
    asset_table="Image",
    output_dir=Path("./multi_level"),
    group_by=["Subject", "Quality"],  # Creates Subject/Quality/file structure
)
```

#### Symlinks vs Copies

By default, `restructure_assets()` creates symbolic links to save disk space:

```python
# Default: create symlinks (efficient, but requires original bag to remain)
bag.restructure_assets(
    asset_table="Image",
    output_dir=Path("./linked"),
    use_symlinks=True,  # Default
)

# Create copies instead (uses more disk space, but independent of original)
bag.restructure_assets(
    asset_table="Image",
    output_dir=Path("./copied"),
    use_symlinks=False,
)
```

#### Custom Type Selection

When a dataset has multiple types, you can control which type is used for the directory name:

```python
# Use a custom function to select the type
bag.restructure_assets(
    asset_table="Image",
    output_dir=Path("./custom"),
    group_by=["label"],
    type_selector=lambda types: types[-1] if types else "unknown",
)
```

#### Handling Missing Values

When a grouping value is missing or `None`, assets are placed in an `Unknown` folder:

```
ml_data/
  Training/
    positive/
      image1.jpg
    Unknown/          # Assets with missing label values
      image2.jpg
```

#### Prediction Scenarios (Datasets Without Types)

When a dataset has no type defined (empty `dataset_types` list), it is treated as a Testing dataset. This is common for prediction/inference scenarios where you want to apply a trained model to new unlabeled data:

```python
# Create a dataset for prediction (no type)
prediction_dataset = ml.create_dataset(
    dataset_types=[],  # No type - will be treated as Testing
    description="Unlabeled images for prediction"
)

# Add images and download
prediction_dataset.add_dataset_members({"Image": image_rids})
bag = prediction_dataset.download_dataset_bag()

# Restructure for prediction - ends up in testing/Unknown/
bag.restructure_assets(
    asset_table="Image",
    output_dir=Path("./prediction_data"),
    group_by=["Diagnosis"],  # No labels assigned yet
)
```

This creates:

```
prediction_data/
  testing/            # Dataset without type treated as Testing
    Unknown/          # No labels assigned
      image1.jpg
      image2.jpg
```

#### Finding Assets Through Foreign Key Paths

Assets are found by traversing all foreign key paths from the dataset, not just direct associations. For example, if a dataset contains Subjects and the schema has Subject → Encounter → Image relationships, `restructure_assets()` will find all Images reachable through those paths even though they are not directly in a Dataset_Image association table:

```python
# Dataset contains only Subjects
subject_dataset = ml.create_dataset(
    dataset_types=["Training"],
    description="Training subjects"
)
subject_dataset.add_dataset_members({"Subject": subject_rids})

# But we want to restructure Images connected via FK path
bag = subject_dataset.download_dataset_bag()
bag.restructure_assets(
    asset_table="Image",  # Finds Images via Subject -> Encounter -> Image
    output_dir=Path("./ml_data"),
    group_by=["Quality"],
)
```

#### Handling Multiple Feature Values

When an asset has multiple values for the same feature (e.g., labeled by different annotators or different model runs), you can provide a `value_selector` function to choose which value to use:

```python
from deriva_ml.dataset.dataset_bag import FeatureValueRecord

def select_latest_execution(records: list[FeatureValueRecord]) -> FeatureValueRecord:
    """Select the feature value from the most recent execution."""
    return max(records, key=lambda r: r.execution_rid or "")

def select_by_confidence(records: list[FeatureValueRecord]) -> FeatureValueRecord:
    """Select the feature value with highest confidence from raw record."""
    return max(records, key=lambda r: r.raw_record.get("Confidence", 0))

# Use value_selector to resolve multiple values
bag.restructure_assets(
    asset_table="Image",
    output_dir=Path("./ml_data"),
    group_by=["Diagnosis"],
    value_selector=select_latest_execution,
)
```

The `FeatureValueRecord` contains:
- `target_rid`: RID of the asset this feature value applies to
- `feature_name`: Name of the feature
- `value`: The feature value (typically a vocabulary term name)
- `execution_rid`: RID of the execution that created this value (for provenance)
- `raw_record`: The complete feature table row as a dictionary

If no `value_selector` is provided and an asset has multiple different values for the same feature, an error is raised when `enforce_vocabulary=True` (the default). Set `enforce_vocabulary=False` to use the first value found instead.

#### Enforcing Vocabulary-Based Grouping

By default, `enforce_vocabulary=True` ensures that feature-based grouping only uses vocabulary-controlled features. This prevents accidental grouping by asset-based features (which would create cryptic directory names):

```python
# This will raise an error if BoundingBox is an asset-based feature
bag.restructure_assets(
    asset_table="Image",
    output_dir=Path("./ml_data"),
    group_by=["BoundingBox"],  # Asset-based feature
    enforce_vocabulary=True,   # Default - will error
)

# Allow non-vocabulary features
bag.restructure_assets(
    asset_table="Image",
    output_dir=Path("./ml_data"),
    group_by=["BoundingBox"],
    enforce_vocabulary=False,  # Allows asset-based features
)
```

## Finding Datasets

To discover datasets in the catalog:

```python
# List all datasets
all_datasets = ml.find_datasets()
for ds in all_datasets:
    print(f"{ds.dataset_rid}: {ds.description} (v{ds.current_version})")

# Look up a specific dataset
dataset = ml.lookup_dataset("1-abc123")
```

## Deleting Datasets

Datasets can be soft-deleted (marked as deleted but retained in the catalog):

```python
# Delete a single dataset
ml.delete_dataset(dataset)

# Delete a dataset and all nested datasets
ml.delete_dataset(dataset, recurse=True)
```
