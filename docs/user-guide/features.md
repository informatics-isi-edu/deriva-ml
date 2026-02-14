# Features

Features are a core concept in DerivaML for ML data engineering. They enable you to associate metadata with domain objects (like Images, Subjects, or any table in your schema) to support machine learning workflows.

## What is a Feature?

A **feature** associates metadata values with records in your domain tables. Unlike regular table columns, features:

1. **Track Provenance**: Every feature value records which Execution produced it
2. **Use Controlled Vocabularies**: Categorical features use vocabulary terms for consistency
3. **Support Multiple Values**: An object can have multiple values for the same feature
4. **Are Versioned**: Feature values are included in dataset versions for reproducibility

## Common Use Cases

| Use Case | Example | Feature Type |
|----------|---------|--------------|
| Ground truth labels | "Normal" vs "Abnormal" classification | Term-based |
| Model predictions | Inference results from a classifier | Term-based |
| Quality scores | Image quality ratings (1-5) | Term-based |
| Derived measurements | Computed metrics from analysis | Value-based |
| Related assets | Segmentation masks, embeddings | Asset-based |

## Feature Types

Features can reference different types of values:

### Term-Based Features

The most common type. Values come from controlled vocabulary tables, ensuring consistency and enabling queries across the vocabulary hierarchy.

```python
# Create a vocabulary for diagnosis labels
ml.create_vocabulary("Diagnosis_Type", "Clinical diagnosis categories")
ml.add_term("Diagnosis_Type", "Normal", "No abnormality detected")
ml.add_term("Diagnosis_Type", "Abnormal", "Abnormality present")

# Create a feature that uses this vocabulary
ml.create_feature(
    target_table="Image",
    feature_name="Diagnosis",
    terms=["Diagnosis_Type"],
    comment="Clinical diagnosis for this image"
)
```

### Asset-Based Features

Link derived assets (files) to domain objects. Useful for segmentation masks, embeddings, or any computed files.

```python
# Create an asset table for segmentation masks
ml.create_asset("Segmentation_Mask", comment="Binary segmentation mask images")

# Create a feature linking masks to images
ml.create_feature(
    target_table="Image",
    feature_name="Segmentation",
    assets=["Segmentation_Mask"],
    comment="Segmentation mask for this image"
)
```

When you create an asset-based feature, the generated `FeatureRecord` class accepts either a file path or an asset RID for the asset column. During execution upload, file paths are automatically replaced with the RIDs of the uploaded assets.

### Mixed Features

Features can reference both terms and assets for complex annotations.

## Creating Feature Values

Feature values are created during **Executions** to maintain provenance. Every value knows which workflow produced it.

The workflow for adding feature values is:
1. Get the FeatureRecord class for your feature (via `create_feature()` or `feature_record_class()`)
2. Create instances of the FeatureRecord with your data
3. Add the records within an execution using `execution.add_features()`

```python
from deriva_ml import DerivaML
from deriva_ml.execution import ExecutionConfiguration
from deriva_ml.dataset import DatasetSpec

ml = DerivaML(hostname, catalog_id)

# Get the FeatureRecord class for the Diagnosis feature
DiagnosisFeature = ml.feature_record_class("Image", "Diagnosis")

# Set up execution
config = ExecutionConfiguration(
    workflow=ml.create_workflow("Labeling", "Annotation"),
    datasets=[DatasetSpec(rid=dataset_rid)],
)

with ml.create_execution(config) as exe:
    # Get images to label
    bag = exe.download_dataset_bag(DatasetSpec(rid=dataset_rid))

    # Create feature records (provenance tracked automatically)
    feature_records = []
    for image in bag.list_dataset_members()["Image"]:
        record = DiagnosisFeature(
            Image=image["RID"],       # Target record RID
            Diagnosis_Type="Normal",  # Vocabulary term name
        )
        feature_records.append(record)

    # Add all feature records to the execution
    exe.add_features(feature_records)

# Upload after execution context exits
exe.upload_execution_outputs()
```

### Asset-Based Feature Values

For asset-based features, you provide file paths instead of vocabulary terms. The execution handles uploading the asset files and linking them to the feature records.

```python
# Get the FeatureRecord class for an asset-based feature
SegmentationFeature = ml.feature_record_class("Image", "Segmentation")

config = ExecutionConfiguration(
    workflow=ml.create_workflow("Segmentation", "Model_Inference"),
    datasets=[DatasetSpec(rid=dataset_rid)],
)

with ml.create_execution(config) as exe:
    bag = exe.download_dataset_bag(DatasetSpec(rid=dataset_rid))

    feature_records = []
    for image in bag.list_dataset_members()["Image"]:
        # Create the asset file using asset_file_path
        mask_path = exe.asset_file_path(
            "Segmentation_Mask",
            f"mask_{image['RID']}.png",
        )

        # Write the asset file (e.g., a segmentation mask)
        generate_segmentation_mask(image, output_path=mask_path)

        # Reference the file path in the feature record
        record = SegmentationFeature(
            Image=image["RID"],
            Segmentation_Mask=mask_path,  # File path, not an RID
        )
        feature_records.append(record)

    exe.add_features(feature_records)

# Upload assets and feature values
exe.upload_execution_outputs()
```

During upload, the execution automatically:

1. Uploads each asset file to the catalog's object store
2. Replaces the file paths in the feature records with the RIDs of the uploaded assets
3. Inserts the feature records into the catalog

After upload, querying the feature values returns asset RIDs rather than file paths.

## Querying Feature Values

### List All Values for a Feature

```python
# Get all diagnosis values across all images
values = ml.list_feature_values("Image", "Diagnosis")
for v in values:
    print(f"Image {v['Image']}: {v['Diagnosis']} (by Execution {v['Execution']})")
```

### Find Features on a Table

```python
# What features are defined for images?
features = ml.find_features("Image")
for f in features:
    print(f"  {f.feature_name}: {f.feature_table.name}")

# List all features in the catalog
all_features = ml.find_features()
```

### Get Feature Structure

```python
# Examine a specific feature's structure
feature = ml.lookup_feature("Image", "Diagnosis")
print(f"Target: {feature.target_table.name}")
print(f"Feature table: {feature.feature_table.name}")
print(f"Columns: {[c.name for c in feature.feature_table.columns]}")
```

## Feature Tables

When you create a feature, DerivaML creates an association table with:

| Column | Purpose |
|--------|---------|
| `{TargetTable}` | RID of the domain object (e.g., Image RID) |
| `Feature_Name` | The feature name (from Feature_Name vocabulary) |
| `Execution` | RID of the execution that produced this value |
| `{VocabTable}` | RID of the vocabulary term (for term-based features) |
| `{AssetTable}` | RID of the asset (for asset-based features) |

This structure enables:
- Querying all values for a feature
- Finding which execution produced a value
- Joining with vocabulary tables for term labels
- Multiple values per object (many-to-many relationship)

## Best Practices

### Feature Naming

- Use descriptive names: `Diagnosis`, `Quality_Score`, `Predicted_Class`
- Feature names are controlled vocabulary terms in `Feature_Name` table
- Same feature name can be used across different tables

### Vocabulary Design

- Create vocabularies before features that use them
- Include synonyms for flexible matching
- Add descriptions to help users understand term meanings

### Provenance

- Always create feature values within an Execution context
- Use meaningful workflow types: "Manual_Annotation", "Model_Inference", etc.
- Include dataset versions for reproducibility

## Working with Multiple Values

A single object can have multiple values for the same feature. This is common when:

- Multiple annotators label the same image
- A model produces predictions at different times
- Different versions of analysis are run

### Querying Multiple Values

```python
# Get all values for a specific image
values = ml.list_feature_values("Image", "Diagnosis")
image_values = [v for v in values if v["Image"] == image_rid]

for v in image_values:
    print(f"Value: {v['Diagnosis_Type']} from Execution {v['Execution']}")
```

### Resolving Multiple Values in restructure_assets

When restructuring assets by a feature that has multiple values, you can provide a `value_selector` function to choose which value to use:

```python
from deriva_ml.dataset.dataset_bag import FeatureValueRecord

def select_latest(records: list[FeatureValueRecord]) -> FeatureValueRecord:
    """Select the value from the most recent execution."""
    return max(records, key=lambda r: r.execution_rid or "")

bag.restructure_assets(
    asset_table="Image",
    output_dir=Path("./ml_data"),
    group_by=["Diagnosis"],
    value_selector=select_latest,
)
```

The `FeatureValueRecord` provides:

| Attribute | Description |
|-----------|-------------|
| `target_rid` | RID of the object this value applies to |
| `feature_name` | Name of the feature |
| `value` | The feature value (e.g., vocabulary term name) |
| `execution_rid` | RID of the execution that created this value |
| `raw_record` | Complete feature table row as a dictionary |

### Consensus or Aggregation

For more complex scenarios, you might aggregate multiple values:

```python
from collections import Counter

def select_majority_vote(records: list[FeatureValueRecord]) -> FeatureValueRecord:
    """Select the most common value (majority vote)."""
    counts = Counter(r.value for r in records)
    most_common_value = counts.most_common(1)[0][0]
    return next(r for r in records if r.value == most_common_value)
```

## Deleting Features

```python
# WARNING: This permanently removes the feature and all its values
ml.delete_feature("Image", "Diagnosis")
```

Deletion removes:
- The feature table
- All feature values
- All provenance information for this feature

## Features in Datasets

Feature values are included when you:

1. **Export a dataset**: Feature tables are exported as CSVs in the BDBag
2. **Download a dataset bag**: Feature values are loaded into the local SQLite database
3. **Version a dataset**: Feature values at that version are preserved via catalog snapshots

This ensures ML workflows have access to the labels and annotations associated with dataset elements.
