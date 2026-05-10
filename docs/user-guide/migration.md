# Migrating from previous deriva-ml versions

If you are upgrading from the previously-published deriva-ml version to the post-S2 release, several public methods changed signature or were renamed, and a small number of new additive methods became available. This chapter enumerates every user-visible change and shows the before/after code for each, so you can update a downstream project in one pass.

## At a glance

| Category | What changed | Action |
|---|---|---|
| **`DatasetBag.restructure_assets`** | `group_by` → `targets`; `value_selector` merged into `targets` dict form; `"Feature.column"` dotted syntax removed | Rename kwargs; `target_transform` replaces dotted syntax |
| **12 private-named methods** | Public methods that were internal helpers are now `_`-prefixed | Switch to the public alternative where one exists; otherwise your calls break with `AttributeError` |
| **5 deleted methods** | `prefetch_dataset`, `list_foreign_keys`, `add_page`, `user_list`, `globus_login` removed | Use replacements (below) or delete the calls |
| **`AssetRIDConfig` (doc-only)** | The class was never actually named `AssetRIDConfig`; real name is `AssetSpec` / `AssetSpecConfig` | Update imports |
| **`Execution.metrics_file()`** | New recommended path for training-metric logs | Additive; replace ad-hoc `asset_file_path()` calls for metrics if you want the cleaner API |
| **`DatasetBag.as_torch_dataset()` / `as_tf_dataset()`** | New framework adapters | Additive; consider replacing hand-rolled `Dataset` subclasses |
| **Crash recovery** | `Running → Pending_Upload` is now a legal state transition | Additive; drop the `update_status(Failed)` workaround if you had one |
| **`find_*(sort=...)`** | `find_executions`, `find_datasets`, `find_workflows` accept an optional `sort` parameter | Additive; existing callers unaffected (default preserves backend order) |
| **`DerivaMLCatalog` / `DatasetLike` protocols** | Extended with optional `sort=`, `materialize_limit=`, `execution_rids=` kwargs | If you subclass these protocols, accept the new kwargs (or `**kwargs`) so static type-checks still pass |
| **`DerivaMLMaterializeLimitExceeded`** | New exception class exported from `deriva_ml` | Additive; catch it once you start passing `materialize_limit=` to `feature_values()` |

## Breaking changes

### DatasetBag.restructure_assets signature (`TypeError` on call)

This is the largest break in the release. `restructure_assets` now shares the `targets` / `target_transform` / `missing` vocabulary with the new framework adapters (`as_torch_dataset`, `as_tf_dataset`). The old `group_by` / `value_selector` kwargs and the `"Feature.column"` dotted-string syntax are **removed**. Passing them raises `TypeError` (standard Python "unexpected keyword argument" behavior).

**Simple rename — single feature:**

```python
# Before
bag.restructure_assets(
    output_dir="./ml_data",
    group_by=["Diagnosis"],
)

# After
bag.restructure_assets(
    output_dir="./ml_data",
    targets=["Diagnosis"],
)
```

**Dotted column syntax → `target_transform`:**

```python
# Before
bag.restructure_assets(
    output_dir="./ml_data",
    group_by=["Classification.Label"],  # picks the Label column of the Classification feature
)

# After
bag.restructure_assets(
    output_dir="./ml_data",
    targets=["Classification"],
    target_transform=lambda rec: rec.Label,  # the transform extracts the column you want
)
```

`target_transform` must return a string (used as the directory name). Non-string returns raise `DerivaMLValidationError` at construction.

**`value_selector` → per-feature selector dict:**

```python
# Before
from deriva_ml.feature import FeatureRecord
bag.restructure_assets(
    output_dir="./ml_data",
    group_by=["Glaucoma"],
    value_selector=FeatureRecord.select_newest,
)

# After
bag.restructure_assets(
    output_dir="./ml_data",
    targets={"Glaucoma": FeatureRecord.select_newest},
)
```

The dict form is strictly more expressive — you can attach a different selector per feature, which `value_selector` (single callable for all features) could not do.

**New `missing` parameter:**

```python
# Default preserves pre-D2 behavior: unlabeled assets go to unknown/
bag.restructure_assets(
    output_dir="./ml_data",
    targets=["Diagnosis"],
    missing="unknown",  # default; same tree as before
)

# Strict mode: raise if any asset is unlabeled
bag.restructure_assets(
    output_dir="./ml_data",
    targets=["Diagnosis"],
    missing="error",
)

# Drop mode: omit unlabeled assets from the output tree
bag.restructure_assets(
    output_dir="./ml_data",
    targets=["Diagnosis"],
    missing="skip",
)
```

**Unchanged parameters:** `output_dir`, `asset_table`, `use_symlinks`, `type_selector`, `type_to_dir_map`, `enforce_vocabulary`, `file_transformer` work exactly as before. FK-reachability for finding assets through parent datasets is unchanged. Type-derived top-level directory naming (`Training` → `training` etc.) is unchanged.

### Renamed private methods (`AttributeError` on call)

Twelve methods that were public-named but actually internal helpers are now `_`-prefixed. If your code called them by the old name, switch to the underscored version or to the public alternative where one exists.

| Old name | New name | Recommended replacement (if different) |
|---|---|---|
| `ml.domain_path()` | `ml._domain_path()` | — |
| `ml.table_path()` | `ml._table_path()` | — |
| `ml.is_system_schema()` | `ml._is_system_schema()` | — |
| `ml.get_domain_schemas()` | `ml._get_domain_schemas()` | — |
| `ml.apply_logger_overrides()` | `ml._apply_logger_overrides()` | — |
| `ml.compute_diff()` | `ml._compute_diff()` | — |
| `ml.retrieve_rid()` | `ml._retrieve_rid()` | **`ml.resolve_rid()`** is the user-facing API |
| `ml.cache_features()` | `ml._cache_features()` | — (legacy workspace-cache shortcut) |
| `ml.add_workflow()` | `ml._add_workflow()` | **`ml.create_workflow()`** is the user-facing factory |
| `ml.start_upload()` | `ml._start_upload()` | — (internal upload plumbing) |
| `asset_record_class()` (module-level factory) | `_asset_record_class()` | **`ml.asset_record_class(table)`** mixin method stays public |

The underscored versions still work and behave identically — they are simply no longer part of the library's documented public surface. If your code calls one of them, the recommended fix is the public alternative where the table above names one; otherwise, just add the underscore.

### Deleted methods (`AttributeError` on call)

Five methods with no callers in any sibling repo were removed entirely.

| Deleted | Replacement |
|---|---|
| `ml.prefetch_dataset(rid)` | `ml.cache_dataset(rid)` — `prefetch_dataset` was a one-line deprecated shim |
| `ml.list_foreign_keys(...)` | None needed — the method had no users |
| `ml.add_page(...)` | None — stale web-app helper |
| `ml.user_list()` | None — stale web-app helper |
| `ml.globus_login()` | None — handled transparently by deriva-py |

### AssetRIDConfig class name (`ImportError` on copy-paste)

The previously-published configuration documentation referred to a class named `AssetRIDConfig`. That class never existed under that name — the real public classes are `AssetSpec` (runtime Pydantic model) and `AssetSpecConfig` (hydra-zen interface). If you copied example code from the old configuration docs, you hit `ImportError`.

```python
# Before (from old docs — never actually worked)
from deriva_ml.execution import AssetRIDConfig
asset = AssetRIDConfig(rid="YYYY", description="Pretrained weights")

# After (the real classes)
from deriva_ml.execution import AssetSpec, AssetSpecConfig

# For direct construction in non-hydra code:
asset = AssetSpec(rid="YYYY", cache=True)

# For hydra-zen store definitions:
cfg = AssetSpecConfig(rid="YYYY", cache=True)
```

Note that the real classes do not accept a `description` parameter (that kwarg was fabricated in the old docs).

## New recommended patterns (additive; no migration required)

You are not required to adopt these, but they replace common hand-rolled patterns with cleaner API surface.

### Training metrics

Before, users wrote metrics to an `Execution_Metadata` asset via `asset_file_path` with a generic type:

```python
# Before — works but the asset type is misleading
path = exe.asset_file_path(
    MLAsset.execution_metadata,
    "metrics.jsonl",
    asset_types=ExecMetadataType.execution_config.value,  # not really a config
)
with path.open("a") as f:
    f.write(json.dumps({"epoch": 0, "val_loss": 0.23}) + "\n")
```

Now there is a dedicated method and asset type:

```python
# After
with exe.metrics_file().open("a") as f:
    f.write(json.dumps({"epoch": 0, "val_loss": 0.23}) + "\n")
```

The file uploads as an `Execution_Metadata` asset with `asset_types=Metrics_File`, honestly describing its purpose. See [Chapter 4 — Running an experiment](executions.md#how-to-record-training-metrics).

### PyTorch / TensorFlow / Keras training

Before, users hand-rolled `torch.utils.data.Dataset` subclasses (~35 lines of join code) or reached for `restructure_assets` + `ImageFolder` for image classification only. Now both frameworks have native adapters:

```python
# After — PyTorch
ds = bag.as_torch_dataset(
    element_type="Image",
    sample_loader=PIL.Image.open,
    targets=["Glaucoma"],
    target_transform=lambda rec: CLASS_TO_IDX[rec.Glaucoma],
)
loader = DataLoader(ds, batch_size=32, shuffle=True)

# After — TensorFlow
ds = bag.as_tf_dataset(
    element_type="Image",
    sample_loader=lambda p, row: tf.io.decode_image(tf.io.read_file(str(p))),
    targets=["Glaucoma"],
    target_transform=lambda rec: CLASS_TO_IDX[rec.Glaucoma],
).batch(32).prefetch(tf.data.AUTOTUNE)

# Keras works with either — choose the adapter matching your backend.
```

See [Chapter 5 — Working offline](offline.md#how-to-feed-a-bag-to-a-training-framework) for the full walkthrough covering PyTorch, TensorFlow, Keras (all three backends), and the `restructure_assets` alternative for third-party trainers.

`restructure_assets` still exists and is still the right tool when your downstream trainer expects the `ImageFolder` class-folder directory layout (e.g., RetFound fine-tuning scripts). The new adapters and `restructure_assets` share the same `targets` / `target_transform` / `missing` vocabulary (that is the whole point of the D2 rename) — but they are alternatives, not a pipeline.

### Sorted `find_*` results

`find_executions`, `find_datasets`, and `find_workflows` now accept an optional `sort=` keyword. The same three-state spec works on all three:

```python
# Backend-determined order — existing default, no behavior change.
all_execs = ml.find_executions()

# Newest-first by record creation time (RCT desc) — recommended for
# "show me what's new" queries.
recent = ml.find_executions(sort=True)
recent_ds = ml.find_datasets(sort=True)
recent_wfs = ml.find_workflows(sort=True)

# Custom sort — receives the path-builder path and returns one or more
# sort keys. Useful when you want to sort by a domain column.
by_name = ml.find_workflows(sort=lambda p: p.Name)
```

`sort=` defaults to `None`, so existing call sites are unaffected. Pass `True` to opt into the method's documented default (RCT desc). Pass a callable to compose your own ordering using path-builder column expressions.

The shared semantics live in [`deriva_ml.core.sort.resolve_sort`](../api/index.md) (a one-line helper) and the matching `SortSpec` type alias, both available from `deriva_ml.core.sort` if you are writing your own `find_*`-shaped helpers in downstream code.

### Subclasser note: `DerivaMLCatalog` / `DatasetLike` protocols

If you implement `DatasetLike`, `DerivaMLCatalogReader`, or `DerivaMLCatalog` yourself (custom storage backend, mock, test double), the protocol declarations now include three optional kwargs:

- `sort=` (on `find_executions`, `find_datasets`, `find_workflows`, and reserved on `list_dataset_*` for forward-compat)
- `materialize_limit=` (on `feature_values`)
- `execution_rids=` (on `feature_values`)

All three default to `None`, so live behavior is unchanged. But strict static type-checking (`mypy --strict` on a `Protocol` `runtime_checkable=True`) will flag your concrete class as no longer conforming if its method signatures don't expose the kwargs. The minimum-effort fix is to add `**kwargs` to your impls, or accept the new kwargs and ignore them.

```python
# Before — was protocol-conformant.
class MyCustomCatalog(DerivaMLCatalogReader):
    def find_executions(self) -> list[Execution]: ...

# After — accept the new kwarg (ignore if you don't use it).
class MyCustomCatalog(DerivaMLCatalogReader):
    def find_executions(self, sort=None) -> list[Execution]: ...
```

### `DerivaMLMaterializeLimitExceeded`

A new exception class is exported from the top-level package:

```python
from deriva_ml import DerivaMLMaterializeLimitExceeded
```

It is raised when a query result set would exceed the caller-supplied `materialize_limit=`. The first user-facing call site will be `feature_values(materialize_limit=...)`; the exception class is available now so downstream code can prepare its handler.

```python
try:
    rows = ml.feature_values(
        feature_name="Glaucoma",
        materialize_limit=10_000,
    )
except DerivaMLMaterializeLimitExceeded as e:
    # Narrow the query (e.g. pass execution_rids=...) or raise the limit.
    ...
```

### Crash recovery

Before, after a hard process crash (OOM, SIGKILL) in the middle of an execution, you had to manually mark the execution `Failed` so `upload_execution_outputs` could proceed:

```python
# Before — workaround that polluted the audit trail
exe = ml.resume_execution(rid)
exe.update_status(ExecutionStatus.Failed, "Resumed after crash")  # spurious Failed marker
exe.upload_execution_outputs()  # Failed → Pending_Upload → Uploaded
```

Now `Running → Pending_Upload` is a legal state transition; the crash-recovery path is explicit and does not lie about the execution's outcome:

```python
# After — clean path
exe = ml.resume_execution(rid)
exe.update_status(ExecutionStatus.Pending_Upload)  # honest: "advance the state machine, please upload"
exe.upload_execution_outputs()  # Pending_Upload → Uploaded
```

See [Chapter 4 — Running an experiment](executions.md#how-to-handle-a-crash-resume) for the full crash-recovery flow.

## Environment and tooling

### Python version

deriva-ml requires Python ≥3.12. This is unchanged from the previous release; stated here for completeness.

### Optional dependencies

Two new optional dependency extras are available for the framework adapters:

```bash
# For PyTorch users:
pip install 'deriva-ml[torch]'

# For TensorFlow / Keras-on-TF users:
pip install 'deriva-ml[tf]'

# Both (uncommon but valid):
pip install 'deriva-ml[torch,tf]'
```

The base install remains lean — neither framework is a hard dependency. Library imports continue to work without either installed; only the corresponding `as_*_dataset` method raises `ImportError` with an install hint when called.

**Platform note for TensorFlow:** on macOS (Apple Silicon), the conventional wheel is `tensorflow-macos` rather than vanilla `tensorflow`. On CUDA-enabled Linux, it is `tensorflow[and-cuda]`. Our extra pins `tensorflow>=2.15` as the package name; if you want a different variant, install it separately and skip the `[tf]` extra — the import guard inside `as_tf_dataset` checks only that `tensorflow` is importable.

### SETUPTOOLS_USE_DISTUTILS env var

A small internal fix: `Workflow.get_dynamic_version()` used to set `os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"` as a defensive measure for an older setuptools bug. That mutation leaked into every subsequent `subprocess.Popen`, which on Python 3.13 crashed third-party packages (e.g., `fair_identifiers_client`) that still import `distutils.version`. The env mutation is gone. No user action required; noted here because observers may have seen the env var appear process-wide before and wondered why.

### `SchemaORM.__del__` shutdown noise

Short-lived scripts that constructed a `DerivaML` instance and exited used to print a multi-line `Exception ignored in: <function SchemaORM.__del__>` traceback ending in `AttributeError: 'NoneType' object has no attribute '_dispose_registries'`. This was the well-known interpreter-shutdown ordering issue — SQLAlchemy module globals torn down before the finalizer ran. The runtime swallowed it, so nothing actually broke, but the traceback made successful runs look failed. The finalizer now catches and ignores the late-shutdown error explicitly. No user action required; the explicit `dispose()` and context-manager exit paths still raise normally.

### Internal performance fixes (no API change)

Several catalog-load and feature-flush hot paths were rewritten to batch SQLite transactions and HTTP round-trips that previously fired per-row. On a 10K-asset CIFAR-10 load this collapses the wall-clock from ~50 minutes to ~11 minutes. The fixes are internal: no API surface changed for callers. Specific paths affected: lease-token writeback, feature-record staging and flush, upload-engine status callbacks, asset-RID resolution at execution start, `delete_dataset_members`, `find_workflows`, and `_load_hydra_config`.

If you previously worked around the slow asset-load time with shell-side parallelism or chunking, you can drop those workarounds.

## Finding affected code in your project

Run these greps against your project directory to enumerate call sites that need updates:

```bash
# restructure_assets old kwargs
grep -rn "group_by=\|value_selector=" your_project/ --include="*.py"
grep -rnE 'restructure_assets.*"[A-Za-z_]+\.[A-Za-z_]+"' your_project/ --include="*.py"

# Renamed private methods — find calls by the old public name
grep -rnE "\b(domain_path|table_path|is_system_schema|get_domain_schemas|apply_logger_overrides|compute_diff|retrieve_rid|cache_features|add_workflow|start_upload)\(" your_project/ --include="*.py" \
  | grep -v -E "\._\(?(domain_path|table_path|is_system_schema|get_domain_schemas|apply_logger_overrides|compute_diff|retrieve_rid|cache_features|add_workflow|start_upload)"

# Deleted methods
grep -rnE "\.(prefetch_dataset|list_foreign_keys|add_page|user_list|globus_login)\(" your_project/ --include="*.py"

# AssetRIDConfig (doc-only break)
grep -rn "AssetRIDConfig" your_project/ --include="*.py"

# DerivaMLCatalog / DatasetLike subclassers (only relevant if you
# implement these protocols — the kwargs default to None so live
# behavior is unchanged, but mypy --strict will flag the protocol gap)
grep -rnE "class .*\(.*DerivaMLCatalog(Reader)?\)|class .*\(.*DatasetLike\)" your_project/ --include="*.py"
```

For known downstream projects in the deriva-ml ecosystem, the sibling-repo grep at the time this guide was written found the following call sites:

```
~/GitHub/deriva-ml-template-test/src/models/cifar10_cnn.py:288:        dataset.restructure_assets(
~/GitHub/deriva-ml-template-test/src/models/cifar10_cnn.py:291:            group_by=["Image_Classification"],  # Group by feature to create class subdirs
```

Template maintainers should update the `group_by=` to `targets=` in lockstep with upgrading.

## Migrating to 2.0: dev versioning

The 2.0 release introduces a two-state versioning model for datasets — see [ADR-0003](../adr/0003-dataset-dev-versioning-model.md) and the ["How to version a dataset"](datasets.md#how-to-version-a-dataset) chapter for the full picture. This is the largest semantic break in the 2.0 release.

**Headline change:** `add_dataset_members`, `delete_dataset_members`, and the dataset-type mutation methods now flip the dataset to a *dev* version (a mutable label of the form `<last_release>.post1.devN`) instead of bumping to a released version. The new `Dataset.release()` method is the only operation that produces a released version.

### At-a-glance

| Before (1.x) | After (2.0) |
|---|---|
| `dataset.increment_dataset_version(VersionPart.minor, "...")` | `dataset.release(bump=VersionPart.minor, description="...")` |
| `add_dataset_members` → `0.4.0` → `0.5.0` (released bump) | `add_dataset_members` → `0.4.0` → `0.4.0.post1.dev1` (dev flip) |
| Multiple `add_dataset_members` calls = multiple released versions | Multiple `add_dataset_members` calls = one mutable dev row, advancing `.devN` |
| No way to record indirect drift | `dataset.mark_dev(description)` records drift explicitly |
| No drift detection | `dataset.is_dirty()`, `dataset.release_diff()`, `dataset.compare_versions(a, b)` |
| `DatasetVersion` extends `semver.Version` | `DatasetVersion` extends `packaging.Version` (PEP 440) |

### `increment_dataset_version` → `release()`

The most mechanical break. Every `increment_dataset_version` call site needs to switch to `release()`:

```python
# Before (1.x)
new_version = dataset.increment_dataset_version(
    component=VersionPart.minor,
    description="Stable for experiment 1",
    execution_rid=exe.execution_rid,
)

# After (2.0)
new_version = dataset.release(
    bump=VersionPart.minor,
    description="Stable for experiment 1",
    execution=exe,            # the Execution object, not the RID
)
```

Three changes in the signature:

- Method name: `increment_dataset_version` → `release`.
- Argument name: `component` → `bump`.
- Argument type: `execution_rid: RID | None` → `execution: Execution | None` (pass the object, not the RID).

**`release()` errors if the dataset has no dev period to promote.** This is intentional — every release must come from a real working state. If you want to mint a release without any actual change (for example, to attach release notes), call `mark_dev(description)` first to declare a dev period, then `release()` to close it.

A private `_increment_dataset_version` is preserved as an internal force-bump primitive. **Don't use it for application code** — it bypasses the dev-versioning model and exists for system-internal use cases (catalog clone version reinitialization). Public callers should always use `release()`.

### Member mutations now land on dev

Code that calls `add_dataset_members` and assumes the result is a released version needs to call `release()` afterward to mint a stable reference:

```python
# Before (1.x): one call, one released version
dataset.add_dataset_members(members={"Image": image_rids})
# dataset.current_version is now e.g. "0.5.0" (released)

# After (2.0): one call → dev label; release explicitly
dataset.add_dataset_members(members={"Image": image_rids})
# dataset.current_version is now "0.4.0.post1.dev1" (dev)
dataset.release(bump=VersionPart.minor, description="Add training images")
# dataset.current_version is now "0.5.0" (released)
```

Code that only consumes `current_version` for display works without changes; code that uses `current_version` as a stable reference (passing it to `DatasetSpec`, citing it externally, etc.) needs to verify it's a released label.

### Version label format change

`DatasetVersion` is now a `packaging.Version` (PEP 440) rather than a `semver.Version`. The wire format for *released* versions is unchanged: `"0.4.0"` parses and serializes identically. Two corner cases to watch:

- **String equality with `DatasetVersion` no longer works.** Under semver, `Version.parse("1.0.0") == "1.0.0"` was True. Under `packaging.Version`, it's False. If you have assertions like `dataset.current_version == "1.0.0"`, change them to `str(dataset.current_version) == "1.0.0"`.
- **Dev versions** print as `"0.4.0.post1.dev3"`, not `"0.4.0-dev3"`. Choice of post-release form (vs. semver's pre-release form) was made so the dev label sorts *after* its anchor release, matching `setuptools-scm`'s convention.

### New methods (additive)

Four methods are new in 2.0; you don't need to migrate to them but they're available:

- `Dataset.mark_dev(description, execution=None)` — explicitly flip the dataset to dev when the catalog drifted via a path that didn't go through the dataset API (a separate execution recorded a feature value, etc.).
- `Dataset.is_dirty() -> bool` — fast predicate for "has anything reachable changed since the last release?"
- `Dataset.release_diff() -> dict[str, int]` — per-table change counts since the last release.
- `Dataset.compare_versions(v_a, v_b) -> dict[str, int]` — per-table change counts between two versions.

### Find every callsite

```bash
# Method renames
grep -rn "increment_dataset_version" your_project/ --include="*.py"

# String-equality assertions on DatasetVersion (most common silent break)
grep -rnE "current_version == |== current_version|\.version == \"" your_project/ --include="*.py"

# Code that relies on add_dataset_members producing a released version
grep -rn "add_dataset_members" your_project/ --include="*.py"
```

### Known limitation: download requires released versions

`download_dataset_bag()` does not yet accept a dev label. If your test fixtures or scripts call `add_dataset_members(...)` and then `download_dataset_bag(current_version)` without releasing in between, the download will raise `ValidationError` (the dev row has no snapshot). Add an explicit `release()` between the mutation and the download:

```python
dataset.add_dataset_members(...)
dataset.release(VersionPart.minor, "Cut for download")
bag = dataset.download_dataset_bag(dataset.current_version)
```

Tracked as [issue #89](https://github.com/informatics-isi-edu/deriva-ml/issues/89). Resolving downloads against live dev state is on the roadmap.

### Out-of-repo follow-ups

The deriva-ml-mcp tool surface is updated separately. The `deriva_ml_increment_dataset_version` MCP tool will be renamed to `deriva_ml_release` in a coordinated MCP-server release; until that lands, MCP-mediated callers should use the bundle resource directly rather than the renamed tool.

## Version compatibility

| deriva-ml version | Python | Torch | TensorFlow | Notes |
|---|---|---|---|---|
| Previous published release | ≥3.12 | n/a | n/a | Pre-S2 baseline |
| This release (post-S2) | ≥3.12 | ≥2.0 optional (`[torch]` extra) | ≥2.15 optional (`[tf]` extra) | This migration guide applies. Includes the batched-SQLite/HTTP perf fixes (~5× speedup on 10K-asset loads) and the additive `find_*(sort=...)`, `materialize_limit=`, `execution_rids=` kwargs. |

## Support

If you hit a break not documented in this chapter, please open an issue. In particular, if you have code that worked on the previous release and fails after upgrading with an error that is not listed here, that is likely a bug on our side — the clean-break renames are all enumerated above, and everything else in the release is intended to be additive.
