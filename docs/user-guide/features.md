# Chapter 3: Defining and using features

Features are structured, provenance-linked annotations on existing catalog records. By the end of this chapter you will know how to define features and vocabularies, record values inside an execution, read them back using the three-method S2 surface, reduce multi-annotator data with selectors, and work with asset-based features — both online and from a downloaded bag. You will also understand when to choose each reading method and how selectors differ in their error-handling contracts.

## What is a feature?

A **feature** associates metadata values with records in your domain tables. The key distinction from a plain table column is provenance: every feature value records the execution that produced it, which workflow ran that execution, and when.

Four properties distinguish features from columns:

1. **Provenance.** Every value is linked to an execution, so you always know which model or annotator produced it.
2. **Controlled vocabularies.** Categorical values come from a vocabulary table rather than free text, enforcing consistency across runs.
3. **Multiple values.** The same record can carry several values for the same feature — one per annotator, one per model version, or one per time window.
4. **Versioned.** Feature values travel with the dataset version they were captured under, so an older dataset snapshot returns the labels that existed then.

A feature is backed by an **association table** created automatically by `create_feature`. The table has columns for the target record's RID, the feature name, the execution RID, and one column per vocabulary term or asset reference you declared.

### Common use cases

| Use case | Example | Type |
|---|---|---|
| Ground truth labels | Normal / Abnormal classification | Term-based |
| Model predictions | Classifier inference results | Term-based |
| Quality scores | Image quality rating 1–5 | Term-based |
| Derived measurements | Computed signal-to-noise ratio | Value-based (metadata column) |
| Related assets | Segmentation masks, embeddings | Asset-based |

## How to decide between a feature and a column

Add a column to your domain table when the value is a single, permanent, schema-level property that does not change across runs and does not need provenance (for example, a patient's date of birth, or the pixel dimensions of an image).

Use a feature when any of these is true:

- The value is produced by a workflow (model, annotator, analysis script).
- Multiple sources or versions may produce different values for the same record.
- You want to query "which execution produced this value?" or compare values across runs.
- The value depends on a controlled vocabulary.
- The value is a derived asset (mask, embedding) that needs to be tracked as a file in the object store.

A good rule of thumb: if you will ever write `SELECT * FROM ... WHERE execution = ?`, use a feature.

## How to create a vocabulary

Term-based features require a vocabulary table — a catalog-managed list of terms. Create the vocabulary before the feature that uses it.

```python
ml.create_vocabulary("Diagnosis_Type", "Clinical diagnosis categories")

ml.add_term("Diagnosis_Type", "Normal",   "No pathology detected")
ml.add_term("Diagnosis_Type", "Mild",     "Mild pathology")
ml.add_term("Diagnosis_Type", "Severe",   "Severe pathology")
```

`create_vocabulary` creates the table; `add_term` adds rows to it. Both operations are idempotent by default (`exists_ok=True` on `add_term`), so re-running setup code is safe.

**Notes**

- `add_term` signature: `add_term(table, term_name, description, synonyms=None, exists_ok=True)`.
- `create_vocabulary` signature: `create_vocabulary(vocab_name, comment="", schema=None, update_navbar=True)`.
- The same vocabulary can be used by multiple features and across target tables.
- Feature names are themselves vocabulary terms (in the `Feature_Name` table). `create_feature` adds the name automatically.

## How to create a feature

`create_feature` returns the `FeatureRecord` subclass you will use to construct values. Calling it is a schema-modifying operation — it creates the backing association table — so it belongs in setup code, not inside the execution loop.

### Term-based feature

```python
DiagnosisRecord = ml.create_feature(
    target_table="Image",
    feature_name="Diagnosis",
    terms=["Diagnosis_Type"],
    comment="Clinical diagnosis label for this image",
)
```

### Feature with metadata columns

```python
from deriva_ml import ColumnDefinition, BuiltinTypes

DiagnosisRecord = ml.create_feature(
    target_table="Image",
    feature_name="Diagnosis",
    terms=["Diagnosis_Type"],
    metadata=[ColumnDefinition(name="Confidence", type=BuiltinTypes.float4)],
    optional=["Confidence"],
    comment="Diagnosis with optional confidence score",
)
```

### Asset-based feature

```python
ml.create_asset("Segmentation_Mask", comment="Binary segmentation mask images")

SegmentationRecord = ml.create_feature(
    target_table="Image",
    feature_name="Segmentation",
    assets=["Segmentation_Mask"],
    comment="Segmentation mask for this image",
)
```

The returned `SegmentationRecord` class accepts either a `pathlib.Path` (file to upload) or a catalog RID for the asset column. During upload the library replaces file paths with the RIDs of the uploaded assets automatically.

### Mixed feature

Pass both `terms` and `assets` to create a feature whose values include a vocabulary term and an associated asset in a single record.

**Notes**

- `create_feature` signature: `create_feature(target_table, feature_name, terms=None, assets=None, metadata=None, optional=None, comment="", update_navbar=True)`.
- All columns in `metadata` are required by default. Add their names to `optional` to allow `None`.
- When creating many features in a batch, pass `update_navbar=False` to each call and run `ml.apply_catalog_annotations()` once at the end.
- You can retrieve the record class later without re-creating the feature: `ml.feature_record_class("Image", "Diagnosis")`.

## How to record feature values

Feature values must be created inside an execution context. This is a hard requirement — there is no bypass. The execution provides the provenance link that makes feature values meaningful.

The write path is: get the record class → create instances → call `exe.add_features(records)` inside the `with` block.

```python
from deriva_ml.execution import ExecutionConfiguration
from deriva_ml.dataset import DatasetSpec

config = ExecutionConfiguration(
    workflow=ml.create_workflow("Labeling", "Annotation"),
    datasets=[DatasetSpec(rid=dataset_rid)],
    description="Human annotation pass 1",
)

# Retrieve the record class (or use the return value from create_feature)
DiagnosisRecord = ml.feature_record_class("Image", "Diagnosis")

with ml.create_execution(config) as exe:
    bag = exe.download_dataset_bag(DatasetSpec(rid=dataset_rid))

    records = []
    for image in bag.list_dataset_members()["Image"]:
        records.append(DiagnosisRecord(
            Image=image["RID"],         # target record RID
            Diagnosis_Type="Normal",    # vocabulary term name
        ))

    exe.add_features(records)           # staged, not yet in the catalog

exe.upload_execution_outputs()          # flush staged features to the catalog
```

`exe.add_features` stages records to a local SQLite table with status `Pending`. They are not visible in the catalog until `upload_execution_outputs()` runs. The flush happens after asset upload, so feature rows that reference assets are guaranteed to find those asset rows already present.

### Asset-based feature values

For asset features, provide a `pathlib.Path` where you would normally provide a term name. The upload step resolves the path to a catalog RID.

```python
SegmentationRecord = ml.feature_record_class("Image", "Segmentation")

with ml.create_execution(config) as exe:
    bag = exe.download_dataset_bag(DatasetSpec(rid=dataset_rid))

    records = []
    for image in bag.list_dataset_members()["Image"]:
        mask_path = exe.asset_file_path(
            "Segmentation_Mask",
            f"mask_{image['RID']}.png",
        )
        generate_mask(image, output_path=mask_path)   # your model writes here

        records.append(SegmentationRecord(
            Image=image["RID"],
            Segmentation_Mask=mask_path,   # Path object, replaced with RID on upload
        ))

    exe.add_features(records)

exe.upload_execution_outputs()
```

**Notes**

- `exe.add_features(features)` signature: takes `list[FeatureRecord]`, returns the count staged.
- All records in a single `add_features` call must share one feature definition. Pass records for different features in separate calls.
- `Execution` is auto-filled if unset. You do not need to set it on each record.
- Staged records survive a crash: a resumed execution re-flushes `Pending` rows. See Chapter 4 on `resume_execution`.

## How to read feature values

The S2 surface provides three methods, consistent across `DerivaML`, `Dataset`, and `DatasetBag`.

| Method | Returns | Purpose |
|---|---|---|
| `find_features(table)` | `list[Feature]` | Discover what features exist on a table |
| `feature_values(table, feature_name, selector=...)` | `Iterable[FeatureRecord]` | Read typed values, optionally selector-reduced |
| `lookup_feature(table, feature_name)` | `Feature` | Inspect a feature's structure, get its record class |

### `find_features` — discovery

```python
# Features defined on a specific table
for f in ml.find_features("Image"):
    print(f.feature_name)

# All features in the catalog
for f in ml.find_features():
    print(f"{f.target_table.name}.{f.feature_name}")
```

### `feature_values` — read typed records

```python
from deriva_ml.feature import FeatureRecord

# All values, unfiltered (multiple records per image are possible)
for rec in ml.feature_values("Image", "Diagnosis"):
    print(f"Image {rec.Image}: {rec.Diagnosis_Type} (execution {rec.Execution})")
```

Each record is a typed Pydantic model. Attributes match the feature's columns: the target RID (`rec.Image`), term columns (`rec.Diagnosis_Type`), asset columns, any metadata columns, plus provenance (`rec.Execution`, `rec.RCT`).

To convert to a pandas DataFrame:

```python
import pandas as pd

df = pd.DataFrame(r.model_dump() for r in ml.feature_values("Image", "Diagnosis"))
```

!!! note
    `feature_values` fetches all rows for the feature from the catalog before yielding the first record. It is iterator-shaped for composability, not for streaming very large tables. For tables with millions of rows, apply a selector to reduce the output before collecting.

### `lookup_feature` — structure inspection

```python
feature = ml.lookup_feature("Image", "Diagnosis")
print(f"Term columns:  {[c.name for c in feature.term_columns]}")
print(f"Asset columns: {[c.name for c in feature.asset_columns]}")
print(f"Value columns: {[c.name for c in feature.value_columns]}")

# Get the record class without calling create_feature
DiagnosisRecord = feature.feature_record_class()
```

**Notes**

- `find_features(table=None)` returns a `list[Feature]`. Omit `table` to discover features across all domain tables.
- `feature_values` returns an iterator but materializes all rows in memory before yielding the first record — it is not a true streaming cursor. Apply a `selector` to reduce memory usage when the feature table is large.
- `lookup_feature` does not fetch values; it returns the schema-level `Feature` object and its `feature_record_class()`. Use it when you need column metadata or want a typed record class without running `create_feature` again.
- All three methods work identically on `DerivaML` (live catalog), `Dataset` (catalog-pinned to a snapshot), and `DatasetBag` (offline local bag). The only difference is scope: `Dataset` and `DatasetBag` are limited to data included in that version's bag.

## How to use selectors to reduce multi-annotator data

When the same record has multiple feature values — from different annotators, model runs, or time windows — a **selector** collapses the group to a single value per target RID.

A selector is any callable: `(list[FeatureRecord]) -> FeatureRecord | None`. Pass it as `selector=` to `feature_values`. A selector that returns `None` causes that target RID to be omitted from the output.

### `select_newest` — most recent by creation time

The most common selector. Picks the record with the latest `RCT` (Row Creation Time).

```python
newest = list(ml.feature_values(
    "Image", "Diagnosis",
    selector=FeatureRecord.select_newest,
))
```

### `select_first` — earliest creation time

Preserves the original annotation and ignores later revisions.

```python
originals = list(ml.feature_values(
    "Image", "Diagnosis",
    selector=FeatureRecord.select_first,
))
```

(`select_latest` has the same behavior as `select_newest`.)

### `select_by_execution` — specific execution

Filters to records produced by one known execution RID. Raises if no record in a group matches.

```python
from_run = list(ml.feature_values(
    "Image", "Diagnosis",
    selector=FeatureRecord.select_by_execution("3-WY2A"),
))
```

!!! note
    `select_by_execution` raises `DerivaMLException` when no record in a group matches the given execution RID; `select_by_workflow` returns `None` and silently omits that target RID from the iterator. Use `select_by_workflow` when "no match" is a valid state (some targets may not have been processed by that workflow); use `select_by_execution` when a missing match is a bug.

### `select_by_workflow` — workflow-scoped factory

`FeatureRecord.select_by_workflow` is a **classmethod factory** — it takes the workflow name and a `container`, resolves the execution list eagerly at construction time, and returns a selector closure. Pass the result as `selector=`.

```python
# Keep only values from any execution of "Manual_Annotation" workflow
selector = FeatureRecord.select_by_workflow("Manual_Annotation", container=ml)
from_humans = list(ml.feature_values("Image", "Diagnosis", selector=selector))

# Scoped to a specific dataset (only executions that processed that dataset)
selector = FeatureRecord.select_by_workflow("Model_Inference", container=dataset)
model_preds = list(dataset.feature_values("Image", "Diagnosis", selector=selector))
```

`container` is **required** and keyword-only. It must be a `DerivaML`, `Dataset`, or `DatasetBag` instance — any object that implements `list_workflow_executions(workflow) -> list[str]`. The container determines the scope: `DerivaML` considers all catalog executions; `Dataset` and `DatasetBag` consider only executions within that dataset's scope.

Unknown-workflow errors surface at factory construction time, not during iteration.

### `select_majority_vote` — consensus labeling

For multi-annotator scenarios, pick the most common value. The factory auto-detects the term column for single-term features.

```python
selector = DiagnosisRecord.select_majority_vote()   # auto-detects Diagnosis_Type
consensus = list(ml.feature_values("Image", "Diagnosis", selector=selector))

# For multi-term features, specify the column explicitly
selector = SomeRecord.select_majority_vote(column="MyTerm")
```

## How to define asset features

Asset features link computed files — segmentation masks, embeddings, depth maps — to target records. They differ from term features in two ways:

1. The feature column holds an asset RID, not a vocabulary term.
2. During the execution write path, you provide a `Path` object instead of a string. The library uploads the file to Hatrac and replaces the path with the catalog-side asset RID before inserting the feature row.

### Setup

```python
# Create the asset table (once, at schema setup time)
ml.create_asset("Embedding", comment="512-dim image embedding vectors")

# Create the feature
EmbeddingRecord = ml.create_feature(
    target_table="Image",
    feature_name="ResNet50_Embedding",
    assets=["Embedding"],
    comment="ResNet-50 embedding for contrastive learning",
)
```

### Writing asset features

```python
with ml.create_execution(config) as exe:
    records = []
    for image in images:
        emb_path = exe.asset_file_path("Embedding", f"emb_{image['RID']}.npy")
        np.save(emb_path, compute_embedding(image))

        records.append(EmbeddingRecord(
            Image=image["RID"],
            Embedding=emb_path,      # Path → resolved to RID on upload
        ))
    exe.add_features(records)

exe.upload_execution_outputs()       # assets uploaded first, then feature rows
```

### Reading asset features

After upload, the `Embedding` column holds the asset's catalog RID. Use it to retrieve the file:

```python
for rec in ml.feature_values("Image", "ResNet50_Embedding"):
    print(f"Image {rec.Image} → asset RID {rec.Embedding}")
```

**Notes**

- The upload ordering contract — assets before features — is enforced by the library. You do not need to manage it.
- Mixed features (terms + assets) work the same way: put a `Path` in the asset column and a term name in the term column; both are resolved at upload time.

## How to read features from a downloaded bag

`DatasetBag` implements the same three-method surface as `DerivaML`. You can call `find_features`, `feature_values`, and `lookup_feature` on a bag without a catalog connection.

```python
bag = dataset.download_dataset_bag(version="2.0.0", materialize=True)

# Identical API to the live catalog
for rec in bag.feature_values("Image", "Diagnosis",
                               selector=FeatureRecord.select_newest):
    print(f"{rec.Image}: {rec.Diagnosis_Type}")
```

`select_by_workflow` also works offline — pass `container=bag`:

```python
selector = FeatureRecord.select_by_workflow("Manual_Annotation", container=bag)
human_labels = list(bag.feature_values("Image", "Diagnosis", selector=selector))
```

The bag reads from a per-feature denormalization cache in its local SQLite database. The first call for a feature populates the cache; subsequent calls are cheap. The cache is immutable — bags are read-only.

`lookup_feature` on a bag returns a `Feature` whose `feature_record_class()` works fully offline. This means you can construct `FeatureRecord` instances from a bag's data, then hand them to `exe.add_features` when you are back online:

```python
# Offline: build records from bag data
feature = bag.lookup_feature("Image", "QualityScore")
RecordClass = feature.feature_record_class()
pending = [RecordClass(Image=r["RID"], QualityScore_Type="Good") for r in rows]

# Online: stage and upload
with ml.create_execution(online_config) as exe:
    exe.add_features(pending)
exe.upload_execution_outputs()
```

!!! note
    `restructure_assets` for organizing bag files into ML framework directory layouts is covered in Chapter 5 (Working offline). That chapter also covers the `value_selector` parameter for resolving multi-value features during asset restructuring.

## When to reach for `feature_values` vs `Denormalizer`

Both tools produce tabular output from catalog data, but they answer different questions.

**Use `feature_values(table, feature_name, selector=...)`** when:

- You want the values for one specific feature.
- You want typed `FeatureRecord` objects with provenance attributes.
- You want selector reduction (one value per target RID after filtering by workflow, annotator, or recency).
- You want to iterate over records and process them in Python before converting to a DataFrame.

```python
# feature_values: typed records for one feature, selector-reduced
selector = FeatureRecord.select_by_workflow("Manual_Annotation", container=ml)
labels = {
    rec.Image: rec.Diagnosis_Type
    for rec in ml.feature_values("Image", "Diagnosis", selector=selector)
}
```

**Use `Denormalizer`** when:

- You want a wide DataFrame combining columns from multiple tables in a single call.
- You want structural joins (e.g., `Subject → Observation → Image`) that are not feature-based.
- You want feature tables to participate as leaves in a multi-table join alongside domain tables.
- You want pandas output directly.

```python
# Denormalizer: wide DataFrame joining Subject, Image, and a feature table
from deriva_ml.local_db.denormalize import Denormalizer

d = Denormalizer(dataset)
df = d.as_dataframe(["Subject", "Image", "Execution_Image_Diagnosis"])
# df has Subject columns, Image columns, and Diagnosis columns — one row per feature value
```

The same question — "give me the diagnosis for each image, with subject context" — answered both ways:

```python
# feature_values approach: start from the feature, join subject in Python
records = {r.Image: r.Diagnosis_Type for r in ml.feature_values(
    "Image", "Diagnosis", selector=FeatureRecord.select_newest
)}
# Then join with a subject lookup if needed

# Denormalizer approach: one call, pandas output with all context columns
df = dataset.get_denormalized_as_dataframe(
    ["Subject", "Observation", "Image", "Execution_Image_Diagnosis"]
)
```

Use `feature_values` when you care about provenance, selectors, or typed access. Use `Denormalizer` when you care about wide joins and pandas integration.

## Migrating from pre-S2 APIs

If your code uses any of the following, it will raise `DerivaMLException` with a migration pointer:

| Retired API | Replacement |
|---|---|
| `ml.add_features(records)` | `exe.add_features(records)` inside an execution context |
| `ml.fetch_table_features(...)` | `ml.feature_values(table, name)` or `Denormalizer` |
| `bag.fetch_table_features(...)` | `bag.feature_values(table, name)` |
| `ml.list_feature_values(...)` | `ml.feature_values(...)` (renamed, same signature) |
| `bag.list_feature_values(...)` | `bag.feature_values(...)` (renamed, same signature) |
| `ml.select_by_workflow(records, workflow)` | `FeatureRecord.select_by_workflow(workflow, container=ml)` |

The retired APIs raise immediately with a specific message naming the replacement. There are no silent fallbacks.

## Common pitfalls

!!! warning "feature_values materializes all rows before yielding"
    Despite being iterator-shaped, `feature_values` fetches every row for the feature from the catalog in one call before yielding the first record. For features with very large value tables, apply a selector to reduce the output, or filter in your caller. This behavior is documented in the method's docstring and is by design — stream-safe is a future enhancement.

!!! warning "exe.add_features stages to SQLite, not to the catalog"
    Records passed to `exe.add_features` are not visible in the catalog until `upload_execution_outputs()` completes. If you query `feature_values` on the live catalog before calling `upload_execution_outputs()`, the records from the current execution will not appear. This is intentional: partial execution output should not be visible to other readers.

!!! warning "Workflow deduplication affects select_by_workflow"
    Workflows are deduplicated by checksum. If you run the same script multiple times, `create_workflow` returns the same workflow RID each time. `FeatureRecord.select_by_workflow("My_Workflow", container=ml)` will therefore match feature values from all runs of that script. If you need to distinguish runs, use `select_by_execution` with a specific execution RID, or create a new workflow for each run by changing the script or passing an explicit `checksum`.

## See also

- **API reference:** `DerivaML.create_feature`, `DerivaML.feature_values`, `DerivaML.find_features`, `DerivaML.lookup_feature` in the Library Documentation.
- **[Chapter 2 (Datasets)](datasets.md):** how dataset versioning interacts with feature values via catalog snapshots.
- **[Chapter 4 (Executions)](executions.md):** `create_execution`, `upload_execution_outputs`, `resume_execution`, and the full execution lifecycle.
- **[Chapter 5 (Working offline)](offline.md):** `restructure_assets` with `value_selector` for feature-based file organization in offline ML workflows.
