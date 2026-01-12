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

### Mixed Features

Features can reference both terms and assets for complex annotations.

## Creating Feature Values

Feature values are created during **Executions** to maintain provenance. Every value knows which workflow produced it.

```python
from deriva_ml import DerivaML
from deriva_ml.execution import ExecutionConfiguration

ml = DerivaML(hostname, catalog_id)

# Set up execution
config = ExecutionConfiguration(
    workflow=ml.create_workflow("Labeling", "Annotation"),
    datasets=[DatasetSpec(rid=dataset_rid)],
)

with ml.create_execution(config) as exe:
    # Get images to label
    bag = exe.download_dataset_bag(DatasetSpec(rid=dataset_rid))

    # Add feature values (provenance tracked automatically)
    for image in bag.list_dataset_members()["Image"]:
        ml.add_feature_value(
            table="Image",
            feature_name="Diagnosis",
            target_rid=image["RID"],
            value="Normal",  # or RID of vocabulary term
        )

exe.upload_execution_outputs()
```

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
features = ml.model.find_features(ml.model.name_to_table("Image"))
for f in features:
    print(f"  {f.feature_name}: {f.feature_table.name}")
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
