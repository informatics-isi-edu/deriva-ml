Version 1.35.0

- **`download_dataset_bag` accepts dev-version labels** (closes [#89](https://github.com/informatics-isi-edu/deriva-ml/issues/89)). Calling `download_dataset_bag(current_version)` on a dataset that's currently in a dev period now returns a bag generated against live catalog state, rather than crashing with a Pydantic `ValidationError`. The dev-version `DatasetMinid.RID` uses the bare dataset RID (no `@<snaptime>` suffix); `DatasetMinid.dataset_snapshot` returns `None`. Local caching is disabled for dev versions because the catalog can drift between calls. `use_minid=True` is rejected for dev versions with a clear error pointing at `release()` — dev labels are not citable.
- **Catalog clone uses the public dev → release lifecycle** (closes [#90](https://github.com/informatics-isi-edu/deriva-ml/issues/90)). The post-clone version-reinitialisation path now goes through `mark_dev` + `release` instead of the private force-bump primitive.
- **Removed `Dataset._increment_dataset_version`** (breaking for any caller that reached for the underscored API). The private force-bump primitive is gone now that no callers remain. Public callers always use `release()`.
- **Removed unused dataset-graph helpers** (`_build_dataset_graph`, `_build_dataset_graph_1`, `_list_dataset_parents_current`, `_list_dataset_children_current`, plus the `graphlib` import). These existed only to support the cascading-version behavior of the deleted `_increment_dataset_version`. Per ADR-0003 / Branch C, `release()` operates on a single dataset; no public API path cascades through the parent/child graph.

Version 1.34.0

- **Dataset dev versioning** (breaking). Datasets now use a two-state versioning model: released versions (citable, snapshot-pinned) and dev versions (mutable, between-release labels of the form `<release>.post1.devN`). Every mutation lands on a dev version; `Dataset.release()` is the only path to a released version. See `docs/adr/0003-dataset-dev-versioning-model.md` and `docs/user-guide/migration.md` for the full migration story.
- **`increment_dataset_version` renamed to `release`** (breaking). New signature: `Dataset.release(bump, description, execution=None)`.
- **`add_dataset_members`, `delete_dataset_members`, `add_dataset_type`, `add_dataset_types`, `remove_dataset_type` now land on dev** (breaking). Each call advances `.devN` rather than producing a released version. To mint a release after mutations, call `dataset.release(...)`.
- **`DatasetVersion` rebased on PEP 440** (breaking for some equality assertions). The wire format for released versions is unchanged (`"0.4.0"`); dev labels use PEP 440 post-release form (`"0.4.0.post1.dev1"`). String equality (`current_version == "1.0.0"`) no longer works — coerce explicitly: `str(current_version) == "1.0.0"`.
- **New: `Dataset.mark_dev`, `is_dirty`, `release_diff`, `compare_versions`.** Mark drift explicitly, detect whether the catalog has drifted since the last release, and compare any two versions.
- Documentation: new ADRs (0003 dev-versioning model, 0004 PEP 440 vocabulary, 0005 delivery sequence) and a CONTEXT.md vocabulary file.

Version 1.2.0

- Dataset versioning with semantic versioning. Note that the current dataset version does *NOT* have the current catalog values, but rather the values at the time the dataset was created. 
To get the current values you must increment the dataset version number.  Please consult online documentation for more information on dataset and versioning.
- Streamlined create_execution.  Now all datasets are automatically downloaded and instance variable has databag classes. You no longer need to explictly create dataset_bdbag. 
- Significant performance improvement on cached dataset access and initial download
- Automatic creation of MINID for every dataset download
- Added method to restore an existing execution from local disk.

Version 1.1.4
- Fixed error when creating DatasetBag on windows platform.

Version 1.1.1

- Removed restriction on nested datasets so that now any level of nesting can be accomidated.
- Fixed bug in nested dataset download.
- Added additional methods to DatasetBag to make it easear to explore datasets.
- Added `datasets` instance variable to Execution object which has Dataset objects for all of the datasets listed in the configuration.
- Added option to DatasetBag init to provide a dataset RID or a path.  If the dataset has already been loaded, or the dataset is nested, this will return the assocated DatasetBag object.

