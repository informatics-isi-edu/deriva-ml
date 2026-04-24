# User Guide — Task 14 Code-Example Verification

**Date:** 2026-04-23
**Scope:** Primary code examples in Chapters 2–5.

## Method

Since chapters went through per-chapter review (spec + quality), method signatures were already
verified by grep during drafting. Task 14 is a spot-check that:

1. Primary examples in each key chapter match the library source as of `main` HEAD.
2. Every major field access and method call uses the correct attribute names.

Source files checked: `src/deriva_ml/dataset/dataset.py`, `src/deriva_ml/dataset/dataset_bag.py`,
`src/deriva_ml/dataset/split.py`, `src/deriva_ml/dataset/aux_classes.py`,
`src/deriva_ml/core/mixins/{feature,vocabulary,workflow,dataset,execution}.py`,
`src/deriva_ml/execution/execution.py`, `src/deriva_ml/feature.py`.

## Results

### Chapter 2 — Datasets

| Example | Location | Status |
|---|---|---|
| `exe.create_dataset(dataset_types=..., description=...)` | datasets.md:35 | ✓ matches execution.py:1920 |
| `dataset.add_dataset_members(members={...}, execution_rid=...)` | datasets.md:113 | ✓ matches dataset.py:1194 |
| `split_dataset(ml, rid, test_size=..., seed=...)` | datasets.md:238 | ✓ matches split.py:483 |
| `dataset.download_dataset_bag(version=..., materialize=...)` | datasets.md:354 | ✓ matches dataset.py:1612 |
| `dataset.increment_dataset_version(component=VersionPart.major, description=...)` | datasets.md:193 | ✓ matches dataset.py:782 |
| `dataset.dataset_history()` field access | datasets.md:190 | FIXED (see below) |
| `bag.version` attribute | datasets.md:359,448 | FIXED (see below) |
| `dataset.set_version("1.0.0")` | datasets.md:206 | FIXED (see below) |

**Fixes applied to datasets.md:**

- `entry.version` → `entry.dataset_version` (`DatasetHistory.dataset_version` is the field name)
- `entry.timestamp` → `entry.snapshot` (`DatasetHistory` has no `timestamp` field; the catalog snapshot ID is `snapshot`)
- Replaced the non-existent `dataset.set_version("1.0.0")` example with the correct pattern: `dataset.list_dataset_members(version="1.0.0")` (live) and `dataset.download_dataset_bag(version="1.0.0")` (offline). No `set_version` method exists on `Dataset`.
- `bag.version` → `bag.current_version` (the `DatasetBag` property is `current_version`, not `version`)

### Chapter 3 — Features

| Example | Location | Status |
|---|---|---|
| `ml.create_vocabulary(name, comment)` | features.md:47 | ✓ matches base.py:1390 |
| `ml.add_term(table, term, description)` | features.md:49 | ✓ matches vocabulary.py:105 |
| `ml.create_feature(target_table, feature_name, terms=..., metadata=..., ...)` | features.md:70 | ✓ matches feature.py:62 |
| `exe.add_features(records)` / `exe.upload_execution_outputs()` | features.md:148 | ✓ matches execution.py:835, 1379 |
| `ml.find_features("Image")` | features.md:204 | ✓ matches feature.py:315 |
| `ml.feature_values("Image", "Diagnosis", selector=...)` | features.md:218 | ✓ matches feature.py:366 |
| `FeatureRecord.select_newest`, `select_first`, `select_by_execution`, `select_by_workflow`, `select_majority_vote` | features.md:265–324 | ✓ all present in feature.py |
| `from deriva_ml.feature import FeatureRecord` | features.md:215 | ✓ feature.py exists and exports FeatureRecord |
| `from deriva_ml import ColumnDefinition, BuiltinTypes` | features.md:82 | ✓ re-exported via __init__.py |

### Chapter 4 — Executions

| Example | Location | Status |
|---|---|---|
| `ml.create_workflow(name, workflow_type, description)` | executions.md:22 | ✓ matches workflow.py:318 |
| `ExecutionConfiguration(workflow, datasets=[DatasetSpec(...)], assets=[...])` | executions.md:27 | ✓ matches execution_configuration.py |
| `ml.create_execution(config)` context manager | executions.md:64 | ✓ matches execution.py:64 |
| `exe.download_dataset_bag(DatasetSpec(rid=...))` | executions.md:68 | ✓ matches execution.py:922 |
| `exe.asset_file_path(asset_name, file_name, ...)` | executions.md:103 | ✓ matches execution.py:1691 |
| `exe.upload_execution_outputs(timeout=..., chunk_size=..., ...)` | executions.md:195 | ✓ matches execution.py:1379 |
| `exe.update_status(ExecutionStatus.Running, message)` | executions.md:273 | ✓ matches execution.py:948 |
| `ml.resume_execution(rid)` | executions.md:305 | ✓ matches execution.py:410 |
| `ml.find_workflows()` | offline.md:170 | ✓ matches workflow.py:64 |

### Chapter 5 — Offline

| Example | Location | Status |
|---|---|---|
| `ml.download_dataset_bag(DatasetSpec(rid=..., version=..., materialize=...))` | offline.md:24 | ✓ matches dataset mixin:239 |
| `bag.path` | offline.md:43 | ✓ matches dataset_bag.py:191 |
| `bag.list_dataset_members()` | offline.md:82 | ✓ matches DatasetBag |
| `bag.get_table_as_dataframe("Image")` | offline.md:87 | ✓ present in DatasetBag |
| `bag.feature_values(table, feature, selector=...)` | offline.md:92 | ✓ matches dataset_bag.py:519 |
| `bag.find_features("Image")` | offline.md:102 | ✓ matches dataset_bag.py:489 |
| `bag.lookup_feature(table, feature_name)` | offline.md:106 | ✓ matches dataset_bag.py:590 |
| `bag.restructure_assets(output_dir=..., group_by=..., value_selector=...)` | offline.md:196 | ✓ matches dataset_bag.py:1452 |
| `FeatureRecord.select_majority_vote` as `value_selector` factory | offline.md:287 | ✓ present in feature.py:294 |

## Conclusion

All primary examples in Chapters 3, 4, and 5 verified correct against current source. Chapter 2
had four broken examples: `entry.version`/`entry.timestamp` used wrong `DatasetHistory` field
names (`dataset_version` and `snapshot` are correct), `dataset.set_version()` referenced a
non-existent method (replaced with `list_dataset_members(version=...)` and
`download_dataset_bag(version=...)`), and `bag.version` used the wrong property name
(`current_version` is correct). All four have been fixed in `datasets.md`. No broken examples
remain in Chapters 3–5.
