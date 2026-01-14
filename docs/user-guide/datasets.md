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
from deriva_ml import DerivaML, ExecutionConfiguration, Workflow

# Connect to the catalog
ml = DerivaML("deriva.example.org", "my_catalog")

# Create a workflow
workflow = ml.create_workflow(
    name="Data Preparation Workflow",
    workflow_type="Data Processing",
    description="Prepares training and validation datasets"
)

# Create an execution
config = ExecutionConfiguration(
    workflow=workflow,
    description="Create datasets for experiment"
)
execution = ml.create_execution(config)

# Create datasets within the execution
with execution.execute():
    training_dataset = execution.create_dataset(
        dataset_types=["TrainingSet", "Image"],
        description="Training images for classification model"
    )

    validation_dataset = execution.create_dataset(
        dataset_types=["ValidationSet", "Image"],
        description="Validation images for model evaluation"
    )
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

When creating an execution with dataset specifications, datasets are automatically downloaded:

```python
config = ExecutionConfiguration(
    datasets=[
        DatasetSpec(rid="1-abc123", version="1.0.0"),
        DatasetSpec(rid="1-def456", materialize=True),
    ],
    workflow=workflow,
    description="Process datasets"
)

execution = ml.create_execution(config)

with execution.execute() as exe:
    # Access downloaded datasets
    for dataset in exe.datasets:
        print(f"Dataset {dataset.dataset_rid} available at {dataset.path}")
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

When a grouping value is missing or `None`, assets are placed in an `unknown` folder:

```
ml_data/
  Training/
    positive/
      image1.jpg
    unknown/          # Assets with missing label values
      image2.jpg
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
