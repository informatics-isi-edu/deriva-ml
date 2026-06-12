# Schema-evolution impact analysis

Before dropping a column, renaming a table, or restructuring a schema,
ask the catalog what breaks. Two methods walk the DerivaML domain model
and report the artifacts that reference the thing you are about to
change:

```python
# About to change the Image table -- what breaks?
ml.find_datasets_referencing("Image")
# -> [DatasetReference(dataset_rid='1-AAAA', element_table='Image', member_count=550), ...]

ml.find_features_referencing("Image")
# -> [FeatureReference(feature_name='Quality', target_table='Image',
#                      feature_table='Execution_Image_Quality',
#                      referencing_columns=['Image']), ...]
```

## What each method checks

**`find_datasets_referencing(table, column=None)`** — datasets reference
tables through their member associations (`Dataset_<Table>`). Every
dataset currently holding rows of the table is reported with its member
count. Impact is **table-granular**: membership is row-level, so any
column change affects every dataset holding rows of that table — the
`column` argument is accepted for symmetry but does not narrow the
result.

**`find_features_referencing(table, column=None)`** — a feature's
association table carries FKs to its target table (the self-FK), to
vocabulary tables (term columns), and to asset tables (asset columns).
Any feature whose FK lands on the table is reported, with the FK column
names doing the referencing. `column=` narrows by the *referenced*
column name (FKs reference key columns, almost always `RID`).

This catches the non-obvious cases: dropping a **vocabulary** table
breaks every feature with a term column into it; dropping an **asset**
table breaks every feature that attaches files from it.

## The workflow

1. Run both methods for the table you plan to change.
2. Empty results from both → the change is invisible to the DerivaML
   domain layer (raw catalog consumers may still care — check FKs with
   the schema tools).
3. Non-empty `find_datasets_referencing` → those datasets' next bag
   download changes shape; coordinate with their consumers and plan a
   version bump.
4. Non-empty `find_features_referencing` → those features break
   structurally; migrate or delete them (`delete_feature`) before the
   schema change.

Both methods are read-only and cheap (one bulk query for datasets; a
schema walk for features) — run them freely.
