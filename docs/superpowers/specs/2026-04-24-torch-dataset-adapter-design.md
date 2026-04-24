# PyTorch Dataset Adapter Design

**Date:** 2026-04-24
**Status:** Approved design; implementation plan to follow.
**Scope label:** Post-S2 item D2.
**Source context:** Reviewer #5 fit-for-purpose item #2 in
`docs/superpowers/specs/2026-04-23-post-s2-findings.md` §5 — "the path
from `DatasetBag` to `torch.utils.data.DataLoader` is 30 lines of
boilerplate every user rewrites."

## 1. Problem statement

A deriva-ml user who downloads a `DatasetBag` and wants to train a
PyTorch model today writes ~35 lines of glue code per project:
enumerate dataset members, pull feature values for labels, join with
the asset table to resolve file paths, build `(path, label)` pairs,
build a label-to-int encoder, subclass `torch.utils.data.Dataset`,
implement `__len__` and `__getitem__`, and finally instantiate a
`DataLoader`. The same boilerplate appears in every downstream project
that targets this library, with minor variations in the label column
name and the sample loader. It's error-prone (asset-path resolution is
fragile, feature joins need `recurse=True` knowledge, label encoding
is easy to get wrong across train/test splits) and every rewrite
re-encodes decisions that deriva-ml already has typed answers for.

`DatasetBag.restructure_assets()` covers the narrow case of image
classification with a single scalar feature as the label, by emitting
the `ImageFolder` directory layout. That shape breaks down for
tabular datasets, multi-modal inputs, regression targets, multi-label
classification, non-image data, and lazy streaming. It also leaves
TensorFlow users unserved.

Users want a one-line path from bag to `torch.utils.data.Dataset` that
respects the library's existing feature-access API, stays consistent
with the `DatasetBag`-vs-catalog parity the rest of the library promises,
and doesn't force the user to materialize or copy their data.

## 2. Design anchors

Non-negotiables that constrain all subsequent decisions.

**Anchor 1: Labels come from features.** The existing feature-access
API is the source of truth for labels. The adapter calls
`bag.feature_values(table, feature_name, selector=...)` internally; it
does not invent a separate label-reading path. This keeps one mental
model: a feature value is "a typed record describing an annotation on
a domain-table row," whether the consumer is a user inspecting the
dataset or a PyTorch training loop.

**Anchor 2: User owns the label-to-integer mapping.** The library
surfaces `FeatureRecord` instances (typed with the feature's declared
column types); the user's `target_transform` callable does the
numeric encoding. The library does not hold a class-to-index table,
does not auto-fit one on first pass, does not persist one. This avoids
the worst failure mode of other ML data libraries — train/test label
encoders silently diverging — and respects deriva-ml's broader
convention of not opining on user numeric shapes.

**Anchor 3: PyTorch and TensorFlow are optional dependencies.** Each
adapter lives behind a runtime framework-import guard (`import torch`
for `as_torch_dataset`, `import tensorflow` for `as_tf_dataset`). The
base library never imports either at module-load time. A user who
installs deriva-ml without one or both frameworks can call every
non-adapter API normally; calls to the corresponding adapter method
raise a clean `ImportError` with install hints. Package extras
`deriva-ml[torch]` and `deriva-ml[tf]` exist as conveniences.

**Anchor 4: Bag-only in v1.** Live-catalog adapter variants
(`Dataset.as_torch_dataset`, `Dataset.as_tf_dataset`) are
intentionally out of scope. Live-catalog datasets require
download-during-`__getitem__`, which introduces network IO on hot
paths, credential lifetime concerns, retry policy, cache eviction —
all of which deserve their own spec. The bag case covers the primary
training workflow (download the bag, then train locally) without any
of that complexity. Live-catalog support is a named follow-up for
both framework adapters.

**Anchor 5: Feature-parallel consistency.** The adapter's signature,
error behavior, and selector handling mirror
`bag.feature_values(...)`'s existing shape. A user who knows the
feature API should be able to read the adapter's signature without
surprise.

**Anchor 6: Composition over replacement; no feature overlap.**
deriva-ml already answers several questions the adapter could try to
re-solve:

- **Train/val/test splitting** lives in `split_dataset` (with
  stratification via `stratify_by_column`). Each partition becomes its
  own catalog dataset and its own bag; the adapter handles each bag
  independently. This is the one real **composition** pattern — see
  §7.4.1.
- **Filesystem-layout rewrites for third-party tools** (`ImageFolder`
  et al.) live in `restructure_assets`. This is an **alternative** to
  the adapter, not a pipeline step: the two tools solve the same
  problem with different output shapes (lazy in-place read vs.
  eager on-disk rewrite). See §7.4.2 for why they don't chain.
- **Catalog-level annotation joins** live in
  `bag.get_denormalized_as_dataframe` and `bag.feature_values`. The
  adapter reads feature values through the existing API (anchor 1),
  it doesn't invent a second path.

The adapter's v1 job is exactly one thing: lazy iteration over an
already-defined bag. It does not duplicate these neighboring tools,
does not wrap them in competing convenience methods, and does not
offer knobs that replicate theirs.

**Anchor 7: Shared target-specification vocabulary across
`as_torch_dataset` and `restructure_assets`.** Both methods answer
the same question — "given a bag element, which feature value becomes
the label?" — so their inputs describing that question use the same
parameter names, types, and semantics. The divergence is output-only:
one returns a torch `Dataset`; the other writes a directory tree.
This is NOT composition (§7.4.2 explains why they don't chain), it is
*vocabulary alignment*: users who learn one method's label-selection
arguments can transfer that knowledge directly. Concretely, both
methods share the same `targets`, `target_transform`, and `missing`
parameters with matching semantics. Achieving this requires small
non-breaking changes to `restructure_assets`'s signature, which ship
in the same PR as the adapter — see §8 for the alignment scope.

## 3. Public API

### 3.1 Signatures

Two adapter methods, with identical shapes for their six shared
keyword arguments:

```python
class DatasetBag:
    def as_torch_dataset(
        self,
        element_type: str,
        *,
        sample_loader: Callable[[Path, dict[str, Any]], Any] | None = None,
        transform: Callable[[Any], Any] | None = None,
        targets: list[str] | dict[str, FeatureSelector] | None = None,
        target_transform: Callable[
            [FeatureRecord | dict[str, FeatureRecord] | list[FeatureRecord] | None],
            Any,
        ] | None = None,
        missing: Literal["error", "skip", "unknown"] = "error",
    ) -> "torch.utils.data.Dataset":
        """..."""

    def as_tf_dataset(
        self,
        element_type: str,
        *,
        sample_loader: Callable[[Path, dict[str, Any]], Any] | None = None,
        transform: Callable[[Any], Any] | None = None,
        targets: list[str] | dict[str, FeatureSelector] | None = None,
        target_transform: Callable[
            [FeatureRecord | dict[str, FeatureRecord] | list[FeatureRecord] | None],
            Any,
        ] | None = None,
        missing: Literal["error", "skip", "unknown"] = "error",
        output_signature: "tf.TensorSpec | tuple[tf.TensorSpec, ...] | None" = None,
    ) -> "tf.data.Dataset":
        """..."""
```

Both adapters share six keyword-only parameters with matching names,
types, and semantics (`element_type`, `sample_loader`, `transform`,
`targets`, `target_transform`, `missing`). `as_tf_dataset` adds one
TF-specific parameter (`output_signature`) that has no counterpart in
the torch shape; see §3.2 for its semantics. All arguments after
`element_type` are keyword-only by intent: they are semantically
independent and positional order would make future additions
(e.g., `sampler=` on torch, `prefetch=` on tf) awkward.

### 3.2 Argument semantics

**`element_type`** — Name of the domain table whose rows become the
dataset's samples. Must be a table present in the bag (covered by the
same error path `list_dataset_members` uses). Whether `sample_loader`
is required depends on whether the element_type is an asset table or
not — see the `sample_loader` entry below.

**`sample_loader`** — `Callable[[Path | None, dict], Any]`. Invoked
once per `__getitem__` call. Receives:
- `Path | None`: absolute filesystem path under
  `bag.path / "data/assets/<element_type>/<rid>/<filename>"` when
  `element_type` is an asset table; `None` otherwise.
- `dict[str, Any]`: the raw row dict from the element table (all
  columns, not just those the framework needs). Lets the loader pull
  non-asset columns, FK values, timestamps, etc.

Return value is the sample (`PIL.Image`, tensor, array, dict of
modalities, whatever the user's model expects). The library does not
inspect it.

Default: asymmetric by element-type kind.

- **Asset-table element_type**: no default — the adapter raises
  `DerivaMLException` at construction if `sample_loader` is `None`.
  Returning raw bytes isn't useful (users can't train on bytes; they
  need a decoded PIL Image / tensor / array), so demanding the loader
  explicitly surfaces the decision the user needs to make. The error
  message names common loaders the user might reach for
  (`PIL.Image.open`, `nibabel.load`, `h5py.File`) as hints. Being
  domain-specific about decoding is the user's job; the library stays
  domain-agnostic.
- **Non-asset-table element_type**: default returns `row_dict`
  unchanged. Useful for tabular training where the element IS the row
  data and no file-level decoding exists. Can be overridden.

Either default can be overridden by passing any callable with the
required signature.

**`transform`** — `Callable[[Any], Any]`. Applied to the sample after
`sample_loader` returns. Standard torchvision-style transform pipeline
goes here. No-op default.

**`targets`** — Source of label data. Three shapes accepted:
- `None` (default) — unlabeled dataset. `__getitem__` returns just the
  sample (not a tuple). Useful for inference loops and for
  self-supervised pretext tasks.
- `list[str]` — feature names to read via
  `bag.feature_values(element_type, name)` with no selector. The
  library aggregates one feature record per element.
- `dict[str, FeatureSelector]` — feature names → selector callables,
  passed verbatim to `bag.feature_values(element_type, name,
  selector=...)`. The selector resolves multi-annotator cases. When a
  selector returns `None` for a group, that element is treated as
  "unlabeled for this feature" and handled per `missing=`.

**`target_transform`** — `Callable` consuming the raw feature-record
shape (see §3.3 for arity) and returning whatever the user's loss
function expects (typically an `int` or a `torch.Tensor`). No-op
default returns the feature record(s) as-is.

**`missing`** — Behavior when a feature value is absent for an element:
- `"error"` (default) — raise `DerivaMLException` at adapter
  construction time, before any `__getitem__` call. Message includes
  the list of unlabeled RIDs. Explicit-over-silent: users should know
  if their dataset is partially labeled.
- `"skip"` — drop unlabeled elements from the dataset entirely. The
  resulting dataset's `__len__` reflects only labeled elements. Index
  mapping is stable across the dataset's lifetime.
- `"unknown"` — keep all elements; pass `None` to the user's
  `target_transform` for unlabeled ones. The user's
  `target_transform` must handle `None` (typically returning a
  sentinel class index or an ignore-mask value). Useful for
  semi-supervised / self-training work.

  Note the naming: `"unknown"` is the **policy name** (shared across
  `as_torch_dataset`, `as_tf_dataset`, and `restructure_assets` —
  see §3.7), while the runtime value the adapter hands to
  `target_transform` is Python `None`. The same literal policy value
  drives restructure's `unknown/` directory and the adapter's
  `None` target — one name, two honest realizations of the same
  "keep it, flag it" intent.

**`output_signature`** (`as_tf_dataset` only) — the `tf.TensorSpec`
structure describing the `(sample, target)` tuple shape and dtype.
TensorFlow's `tf.data.Dataset.from_generator` path requires this
signature to do graph-mode tracing over the dataset.

- `None` (default) — the adapter **infers** the signature from the
  first `(sample, target)` tuple produced at construction time. The
  adapter pulls one sample eagerly, builds the appropriate
  `tf.TensorSpec` via `tf.type_spec_from_value`, and re-wraps the
  generator so that eagerly-consumed first sample is not lost on
  iteration. Cost: one extra iteration at adapter-construction. This
  mirrors the behavior of `keras.utils.timeseries_dataset_from_array`
  and other Keras data-loading helpers that auto-infer when they
  can.
- `tf.TensorSpec(...)` or nested tuple of same — the user provides
  the signature explicitly. Recommended for production pipelines that
  want deterministic startup (no eager first-sample read) or where
  the first sample's shape isn't representative (e.g., ragged
  tensors, variable-length sequences).

Inference limitations: the inferred signature uses `tf.TensorSpec`
with the first sample's exact shape. If later samples have different
shapes (variable-height images, variable-length sequences), the user
must pass `output_signature` explicitly with `shape=[None, ...]`
markers where ragged. The docstring example covers this case.

### 3.3 Target arity

Single-target (`targets=["Feature_A"]`): `target_transform` receives a
`FeatureRecord` directly.

Multi-target (`targets=["Feature_A", "Feature_B"]`): `target_transform`
receives `dict[str, FeatureRecord]` keyed by feature name.

Multi-valued feature (single-target where the feature is multi-valued
— e.g., an `Image → list[Diagnosis]` feature): `target_transform`
receives `list[FeatureRecord]`. Users who want "just pick the first"
flatten in their `target_transform`.

The arity is determined by the `targets` argument shape at
construction time; `target_transform` always sees a stable, predictable
type. No "do what I mean" inference at `__getitem__`.

### 3.4 `__getitem__` contract

Returns:
- `sample` when `targets=None` (unlabeled)
- `(sample, target)` when `targets` is set

Where:
- `sample = transform(sample_loader(path, row_dict))`
- `target = target_transform(feature_record_shape)` per §3.3

This matches the torchvision convention exactly, keeping the adapter
drop-in compatible with standard PyTorch training loops.

### 3.5 `__len__` contract

Equals the count of labeled elements when `missing="skip"`, the total
count otherwise.

### 3.6 Error behavior summary

| Situation | Behavior |
|---|---|
| `element_type` not in bag | `DerivaMLException` at construction |
| feature named in `targets` does not exist | `DerivaMLException` at construction (same message as `bag.feature_values` itself) |
| `missing="error"` + sparse labels | `DerivaMLException` at construction listing up to 20 unlabeled RIDs |
| `import torch` fails | `ImportError` at first call to `as_torch_dataset` with install hint |
| asset-table element_type, no `sample_loader` supplied | `DerivaMLException` at construction — message suggests common loaders (PIL.Image.open, nibabel.load, h5py.File) |
| non-asset element_type, no `sample_loader` supplied | No error — default returns row_dict unchanged |
| asset file missing on disk (bag corrupted) | `FileNotFoundError` on `__getitem__` (torch convention) |

All statically-detectable errors surface at construction time (rows 1-5
above), so the dataset is valid as soon as it's returned for the
overwhelmingly common cases. The `__getitem__`-time `FileNotFoundError`
is unavoidable: the filesystem can change between construction and
access (asset file removed, bag directory renamed, etc.), and torch's
own convention is to propagate `OSError` subclasses from dataset
access. No library-level retry or fallback.

### 3.7 Shared target-specification vocabulary

Per anchor 7, the two adapter methods (`as_torch_dataset`,
`as_tf_dataset`) and `restructure_assets` share the vocabulary for
describing how to resolve feature values into labels. The same three
parameters appear on all three methods with identical names, types,
and semantics:

**`targets: list[str] | dict[str, FeatureSelector] | None`**

Describes which features supply labels for a bag element.

- `None` (adapter only; meaningless for restructure) — yields
  unlabeled samples.
- `list[str]` — list of feature names on `element_type` (or on
  tables FK-reachable from it; see `restructure_assets`'s
  "Finding assets through foreign key relationships" docstring).
  Each feature resolves via `bag.feature_values(element_type, name)`
  with no per-feature selector.
- `dict[str, FeatureSelector]` — mapping from feature name to a
  `Callable[[list[FeatureRecord]], FeatureRecord | None]` that
  resolves multi-annotator cases. Passed through verbatim to
  `bag.feature_values(element_type, name, selector=...)`. Built-in
  selectors: `FeatureRecord.select_newest`, `select_first`,
  `select_latest`, `select_majority_vote(column)`.

Single-key vs multi-key behavior matches each method's output shape
(see §3.3 for the adapter; §8 for restructure).

**`target_transform: Callable[[FeatureRecord | dict[str, FeatureRecord] | list[FeatureRecord]], Any]`**

User-supplied callable that maps the resolved feature record(s) into
the final label value.

- For the adapters (both torch and tf): return value goes to the
  training loop unchanged. Any type (int, `torch.Tensor`, `tf.Tensor`,
  dict of tensors, etc.). The library does not inspect it.
- For restructure: return value MUST be a string (used as a
  filesystem directory name). A non-string return raises
  `DerivaMLValidationError` at construction with a message explaining
  the string constraint.

The input-shape contract is identical across methods:
- Single-target (`targets=["A"]`): receives `FeatureRecord`.
- Multi-target (`targets=["A", "B"]`): receives `dict[str, FeatureRecord]`.
- Multi-valued single-target: receives `list[FeatureRecord]`.

This means a user's `target_transform` written for one method works
for the other with a return-type adjustment (e.g., `return CLASS_IDX[rec.Grade]`
for adapter becomes `return rec.Grade` for restructure).

**`missing: Literal["error", "skip", "unknown"]`**

Policy for elements whose feature value is absent:

- `"error"` — raise `DerivaMLException` at construction listing
  unlabeled RIDs. Default for **both adapters** (torch and tf).
- `"skip"` — drop unlabeled elements entirely. For restructure, the
  asset file is not placed in the output tree. For the adapters, the
  element is absent from `__len__` / iteration.
- `"unknown"` — keep unlabeled elements. For restructure, the asset
  goes into an `unknown/` directory (matches today's behavior).
  For the adapters, the target is `None` (user's `target_transform`
  must handle it). Default for **restructure**.

The default differs by method because the failure modes differ:
silent mislabeling during training is a correctness hazard (adapter
defaults are `"error"`); silently placing an asset in `unknown/`
during a filesystem lay-out is recoverable by inspection (restructure
default is `"unknown"`, which preserves pre-alignment behavior and
avoids breakage for unparameterized calls).

Users who want the adapter strictness in restructure pass
`missing="error"`; users who want restructure's leniency in an
adapter pass `missing="unknown"` and handle `None` in their
`target_transform`.

**Per-method unique arguments.** The required positional argument
differs: both adapter methods take `element_type` (a table name);
`restructure_assets` takes `output_dir` (a path). These can't unify
without hiding what each method's input actually is. Additionally:
`sample_loader` and `transform` apply only to the adapters
(restructure doesn't load in-memory samples; its on-disk rewrite
hook is `file_transformer`, which operates on paths).
`output_signature` applies only to `as_tf_dataset`
(TF-specific, no counterpart on torch or restructure).

## 4. Package structure

Three new modules: two framework adapters and a shared helper:

- `src/deriva_ml/dataset/torch_adapter.py` — builds
  `torch.utils.data.Dataset`. Imports torch lazily inside the builder
  function body.
- `src/deriva_ml/dataset/tf_adapter.py` — builds `tf.data.Dataset`.
  Imports tensorflow lazily inside the builder function body. Handles
  the `from_generator` + `output_signature` inference plumbing.
- `src/deriva_ml/dataset/target_resolution.py` — private helper
  `_resolve_targets(...)` consumed by both adapter modules AND by
  `restructure_assets`. See §8.5.

Public entry points are methods on `DatasetBag` that delegate to the
module-level builders:

```python
# dataset_bag.py (class body):
def as_torch_dataset(self, element_type, **kwargs):
    """..."""
    from deriva_ml.dataset.torch_adapter import build_torch_dataset
    return build_torch_dataset(self, element_type, **kwargs)

def as_tf_dataset(self, element_type, **kwargs):
    """..."""
    from deriva_ml.dataset.tf_adapter import build_tf_dataset
    return build_tf_dataset(self, element_type, **kwargs)
```

This keeps the public signatures discoverable from the class
docstring and IDE autocomplete, while isolating each framework's
coupling to one module. Users who install deriva-ml without torch
or without tensorflow can still import `DatasetBag` and call every
other method; only the adapter method corresponding to the missing
framework raises on first call, with an install-hint message.

## 5. Package extras

`pyproject.toml` gains:

```toml
[project.optional-dependencies]
torch = ["torch>=2.0"]
tf    = ["tensorflow>=2.15"]
```

Installation:
- `pip install 'deriva-ml[torch]'` — for PyTorch users.
- `pip install 'deriva-ml[tf]'` — for TensorFlow / Keras-on-TF users.
- `pip install 'deriva-ml[torch,tf]'` — both (uncommon but valid).

The existing `dependency-groups` table for dev / lint stays where
it is — the new `optional-dependencies` section documents runtime
extras users can opt into, which is distinct from dev tooling.

**Version-floor rationale:**

- **PyTorch `>=2.0`** — modern torch; torchvision users typically
  already have this. Specific-version pinning is the user's concern.
- **TensorFlow `>=2.15`** — recent-enough to cover Keras 3 migration
  (Keras 3 requires TF 2.15+), and cleanly supports the modern
  `tf.data.Dataset.from_generator(output_signature=...)` path this
  adapter uses. Older TF versions had bugs in the generator
  output-signature inference that we'd have to work around.

**Platform note (tensorflow wheel selection):**

On macOS (Apple Silicon) the conventional install is
`tensorflow-macos` rather than vanilla `tensorflow`. On CUDA-enabled
Linux it's `tensorflow[and-cuda]`. Our extra pins `tensorflow>=2.15`
as the package name; users on special platforms should install their
preferred tf wheel separately and install `deriva-ml` without the
`[tf]` extra (the import guard inside `as_tf_dataset` just checks
that `tensorflow` is importable — it doesn't care which wheel). The
UG section documents this.

## 6. Testing strategy

Three test tiers per adapter (torch and TF), plus shared-helper tests
and the existing catalog fixture.

### 6.1 Module-import tests (framework not installed)

Verify the library still imports when a given framework is absent.
Two parallel tests:

- `tests/dataset/test_torch_adapter_no_torch.py` — removes torch via
  `sys.modules` manipulation, confirms `DatasetBag` is importable and
  every method except `as_torch_dataset` works, confirms
  `as_torch_dataset` raises `ImportError` with install-hint message.
- `tests/dataset/test_tf_adapter_no_tf.py` — same shape for
  tensorflow. The two tests are independent (removing torch does not
  touch the TF import path and vice versa).

### 6.2 Pure-Python logic (framework stubbed)

Exercises the join logic, selector pass-through, `missing=` branches,
target arity, and error paths without a real framework install. Two
parallel tests:

- `tests/dataset/test_torch_adapter_logic.py` — stubs
  `torch.utils.data.Dataset`.
- `tests/dataset/test_tf_adapter_logic.py` — stubs `tf.data.Dataset`
  plus `tf.TensorSpec` (the `output_signature` inference is tested
  against the stub).

Both tiers share the same test matrix:
- `missing="error"` raises at construction with RID list.
- `missing="skip"` drops unlabeled elements.
- `missing="unknown"` passes `None` to `target_transform`.
- Selector dict form vs list form yield equivalent results when no
  selector is provided.
- `target_transform` is called on the right shape (FeatureRecord /
  dict / list per target arity).
- Asset-table element with no `sample_loader` raises at construction.

### 6.3 End-to-end with real frameworks

Runs only when the framework is importable. Two parallel tests:

- `tests/dataset/test_torch_adapter_e2e.py` — gated by
  `pytest.importorskip("torch")`. Builds a dataset from the
  `catalog_with_datasets` fixture, iterates under a real
  `DataLoader`, verifies tuple shape, label-encoding round-trip,
  `missing="skip"` filtering, and (where applicable) multi-worker
  behavior.
- `tests/dataset/test_tf_adapter_e2e.py` — gated by
  `pytest.importorskip("tensorflow")`. Same flow using
  `tf.data.Dataset.batch(...)` / `.prefetch(...)`. Additionally
  verifies `output_signature=None` inference works (caller passes
  no signature; first `.take(1)` produces the expected shape).

### 6.4 Shared-helper tests

`tests/dataset/test_target_resolution.py` tests the
`_resolve_targets` helper in isolation from any framework. Covers
the contract enumerated in §8.5 (returns-dict-keyed-by-RID semantics,
three `missing=` branches, selector dict behavior). Independent of
torch and tf — pure-Python.

### 6.5 Test catalog fixture

Leverages the existing `catalog_with_datasets` fixture. The test
doesn't need new catalog setup — the demo catalog already has
subjects, images, and a Glaucoma_Grade-style feature via
`create_demo_features`. The spec assumes that fixture is stable;
implementation will verify.

## 7. Documentation deliverables

**User-facing documentation is a first-class deliverable**, not
optional work to punt to a follow-up.

### 7.1 Docstrings

Every public surface gets the full library docstring contract (one-line
summary + extended description + Args + Returns + Raises + Example).

Covered public surfaces:
- `DatasetBag.as_torch_dataset(...)` — the method docstring
- `build_torch_dataset(...)` — the module-level builder (if exposed
  publicly; likely `_build_torch_dataset` private)

For each argument with non-obvious semantics (`targets` shapes,
`missing` behavior, `sample_loader` signature), include a dedicated
paragraph in the Args section that shows what the user hands in and
what they get back. The catalog-dependent end-to-end `Example:` block
uses `# doctest: +SKIP`; a smaller pure-Python assertion (e.g.,
enum-value check or import-guard behavior) runs at doctest collection.

### 7.2 User guide section

New section in `docs/user-guide/offline.md` (Chapter 5, which already
covers `DatasetBag` and `restructure_assets`) titled **"How to feed a
bag to a training framework"**. Structure follows the UG contract,
with sub-subsections for each supported framework plus Keras routing:

1. **Motivation.** Two sentences explaining why these adapters exist
   and when a user would reach for them instead of
   `restructure_assets`.
2. **Choosing your path.** Short decision paragraph:
   - `as_torch_dataset` — if you're using PyTorch (native) or Keras
     on the PyTorch backend.
   - `as_tf_dataset` — if you're using TensorFlow (native) or Keras
     on the TensorFlow backend.
   - `restructure_assets` — if you're using a third-party trainer
     that expects `ImageFolder` class-folder layout (e.g., RetFound
     fine-tuning scripts, `torchvision.datasets.ImageFolder`,
     `tf.keras.utils.image_dataset_from_directory`).

#### 7.2.1 Using with PyTorch

Structure:
- **Simple example.** Image classification with one feature as
  label, `PIL.Image.open` as loader, torchvision transform
  pipeline. ~15 lines end-to-end.
- **Explanation.** How labels come from features, how the selector
  pattern handles multi-annotator cases, how `missing=` resolves
  sparse-label choices.
- **Worked variations:**
  - Tabular regression (non-asset element_type, no `sample_loader`
    override needed, continuous-valued feature).
  - Multi-target (two features → dict target).
  - Multi-annotator resolution (selector-dict form).
- **Notes:** pointer to `deriva-ml[torch]` extra; explicit note that
  `Dataset.as_torch_dataset` (live-catalog variant) is not yet
  shipped.

#### 7.2.2 Using with TensorFlow

Structure:
- **Simple example.** Same image-classification task but returning
  `tf.data.Dataset`, showing `.batch().prefetch()` pipeline. ~15
  lines end-to-end.
- **`output_signature` guidance.** When to let inference happen
  (default; one-shot research code), when to pass it explicitly
  (production pipelines with deterministic startup; ragged tensors).
- **Worked variations:** the same three from the PyTorch section,
  translated to TF shape.
- **Notes:** pointer to `deriva-ml[tf]` extra and to the platform
  note (§5) about `tensorflow-macos` / `tensorflow[and-cuda]` for
  users who want a specific wheel.

#### 7.2.3 Using with Keras

Keras 3 accepts data from either backend's native format. Users
don't need a separate adapter; they pick the method matching their
Keras backend:

- **Keras on PyTorch backend:** wrap `as_torch_dataset(...)` in a
  `DataLoader` and pass to `model.fit(...)`. Keras consumes
  `torch.utils.data.DataLoader` natively.
- **Keras on TensorFlow backend:** pass `as_tf_dataset(...)` directly
  to `model.fit(...)`. Keras consumes `tf.data.Dataset` natively.
- **Keras on JAX backend:** use either adapter; Keras 3 accepts
  Python iterables of `(x, y)` tuples regardless of origin.

A ~20-line worked example shows the backend choice via
`os.environ["KERAS_BACKEND"] = "torch"` or `"tensorflow"` and then
the appropriate adapter call. Explicit note: no `as_keras_dataset`
method exists because Keras is a consumer of both adapters, not a
third parallel one.

#### 7.2.4 Using with `ImageFolder` / third-party tools

Pointer to §7.4.2: `restructure_assets` is the tool when your
downstream consumer wants the class-folder directory layout
(including RetFound fine-tuning scripts,
`torchvision.datasets.ImageFolder`,
`tf.keras.utils.image_dataset_from_directory`, and similar).
Aligned vocabulary (§3.7) means the `targets` / `target_transform`
/ `missing` kwargs you'd use with `as_torch_dataset` work
identically with `restructure_assets`.

#### 7.2.5 See also

Chapter 3 (features API for selector pattern); the
`DatasetBag.feature_values` docstring for the selector shape;
the `split_dataset` docstring for the stratified-split flow that
composes upstream of all three framework paths (see §7.4.1).

### 7.3 Cross-references

- The new UG section gets linked from Chapter 1 (exploring) where
  users first learn about bags.
- `DatasetBag.restructure_assets`'s docstring gets one sentence
  pointing at `as_torch_dataset` as the general-case companion.
- `split_dataset` (in `src/deriva_ml/dataset/split.py`) gets one
  sentence in its docstring's Examples / See Also section pointing
  at the composition pattern in §7.4 below. No signature change;
  existing behavior preserved.
- The `DatasetBag` class-level docstring gets one bullet mentioning
  the adapter alongside the existing `restructure_assets()` mention.
  (The corresponding paragraph in `CLAUDE.md` — the project-level
  codebase notes — gets the same one-sentence addition for future
  agentic work, tracked as part of the plan's UG-integration task.)

### 7.4 Composition with existing dataset APIs

The adapter **composes** with the other data-shaping primitives; it
does not replace them. Three composition patterns need dedicated
coverage in the UG section and in docstrings because they are the
most common production flows.

#### 7.4.1 Stratified train/val/test split → per-partition torch datasets

`split_dataset` creates a parent `Split` catalog dataset with child
`Training`, `Testing`, and optionally `Validation` datasets. Each
partition is a standalone dataset — download each as its own bag,
then turn each bag into its own `torch.utils.data.Dataset` via the
adapter. The adapter does not need split-awareness; each bag is
independent.

```python
from deriva_ml.dataset.split import split_dataset

# Upstream (online): create stratified split hierarchy in catalog
split = split_dataset(
    ml,
    source_dataset_rid="28D0",
    test_size=0.2,
    val_size=0.1,
    stratify_by_column="Image_Classification.Image_Class",
    seed=42,
)
# split.training.rid, split.testing.rid, split.validation.rid are
# three new catalog dataset RIDs.

# Later (offline training): one bag per partition
train_bag = ml.lookup_dataset(split.training.rid).download_dataset_bag(
    version="1.0.0"
)
val_bag   = ml.lookup_dataset(split.validation.rid).download_dataset_bag(
    version="1.0.0"
)
test_bag  = ml.lookup_dataset(split.testing.rid).download_dataset_bag(
    version="1.0.0"
)

# Build torch datasets from each bag independently.
# Shared adapter config — everything except the per-partition transform:
shared = dict(
    element_type="Image",
    sample_loader=lambda p, row: PIL.Image.open(p).convert("RGB"),
    targets=["Image_Classification"],
    target_transform=lambda rec: CLASS_TO_IDX[rec.Image_Class],
)
train_ds = train_bag.as_torch_dataset(**shared, transform=train_transform)
val_ds   = val_bag.as_torch_dataset(**shared, transform=eval_transform)
test_ds  = test_bag.as_torch_dataset(**shared, transform=eval_transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=4)
```

**Why this works cleanly:** each partition inherits the full feature
coverage of the source dataset (the split only selects RIDs; feature
values for those RIDs remain attached). Stratification happens at
split time against the denormalized catalog DataFrame, so the label
distribution is correct by construction; the adapter just reads it
back through `feature_values(...)` the same way it would for any bag.

#### 7.4.2 `restructure_assets` vs `as_torch_dataset` — these are alternatives, not a pipeline

A natural question: "can I call `restructure_assets` first and then feed
the result into `as_torch_dataset`?" The short answer is **no, and you
wouldn't want to**. These two tools solve the same problem with
different shapes, not two halves of one workflow.

| Need | Tool |
|---|---|
| Lazy streaming, no disk rewrite, full `FeatureRecord` access, any element type | `as_torch_dataset` |
| Compatibility with `torchvision.datasets.ImageFolder` or a third-party trainer that expects the `ImageFolder` directory convention | `restructure_assets` + `ImageFolder` |

**Why they don't chain:** `restructure_assets` creates a *new directory*
with the `ImageFolder` layout by copying or symlinking from the bag's
flat `data/assets/<type>/<rid>/<file>` tree. It doesn't modify the
bag. `as_torch_dataset`'s path resolution is hard-coded to
`bag.path / "data/assets/<type>/<rid>/<file>"` — it reads from the
bag, not from the restructured sibling tree. Restructuring first and
then calling the adapter buys nothing: the adapter still reads the
unchanged bag, and the restructured tree sits unused.

**Picking between them:**

- If your downstream consumer is a `torchvision.datasets.ImageFolder`
  instance (or any code that expects a class-folder directory layout),
  use `restructure_assets`. The adapter is the wrong tool.
- If your downstream consumer is your own `torch.utils.data.DataLoader`
  and you want the full `FeatureRecord` for each sample, use the
  adapter. `restructure_assets` is the wrong tool.
- If you need **both** — e.g., you want an `ImageFolder` for a baseline
  comparison and a custom adapter for your main model — run
  `restructure_assets` once and call `as_torch_dataset` independently
  on the same bag. They don't conflict; they just don't collaborate.

The adapter deliberately does **not** offer a `layout="image_folder"`
shortcut to drive `restructure_assets` under the hood. Keeping the two
primitives separate means `restructure_assets` has a clear contract
(it rewrites disk), the adapter has a clear contract (it reads disk
lazily), and users see in their own code which tool they chose.

**But they do share the target-specification vocabulary.** Per
anchor 7 and §3.7, both methods accept the same `targets`,
`target_transform`, and `missing` parameters with matching semantics.
A user who writes `targets={"Diagnosis": FeatureRecord.select_newest}`
for the adapter can use the same dict for `restructure_assets`. The
alignment is *input-shape* only; output shapes remain distinct. §8
documents the specific changes to `restructure_assets` that achieve
this alignment — a clean-break rename, not a deprecation cycle — and
the migration guidance for any callers on the old parameter names.

## 8. `restructure_assets` alignment scope (same PR)

Per anchor 7 and §3.7, the adapter work lands together with a
coordinated signature update to `restructure_assets`. This section
enumerates what changes on the restructure side and why.

This is a **clean break** — the old parameter names are removed, not
deprecated. Rationale: deriva-ml is not yet at the scale where
maintaining a deprecation-warning surface pays for itself, and
carrying legacy kwargs through the rename would bake in the exact
naming inconsistency this alignment is meant to remove. Downstream
impact is explicitly accepted and called out in §11 (risks).

### 8.1 Parameter renames

| Today | After | Notes |
|---|---|---|
| `group_by: list[str] | None` | `targets: list[str] | dict[str, FeatureSelector] | None` | "group_by" was SQL-flavored and misleading in the ML context. "targets" reads consistently across both methods: what determines the output, directory-name or label. |
| `value_selector: Callable | None` | merged into `targets` dict form | Single-selector was strictly less expressive than per-feature selectors. The dict form subsumes it: `targets={"A": FeatureRecord.select_newest}` covers `value_selector=FeatureRecord.select_newest` for the single-feature case. |

Passing the old kwargs after this PR lands raises `TypeError`
("unexpected keyword argument") — the standard Python signal for a
removed keyword. No `DeprecationWarning` shim, no alias.

### 8.2 New parameter: `target_transform`

Restructure today derives directory names directly from feature values
(or via the `"Feature.column"` dotted-string syntax). After alignment,
`target_transform` is the canonical way to customize what goes into
the directory name:

```python
# Before this PR:
bag.restructure_assets(output_dir="./out", group_by=["Classification.Label"])
# → directories named after the Label column value

# After this PR:
bag.restructure_assets(
    output_dir="./out",
    targets=["Classification"],
    target_transform=lambda rec: rec.Label,   # returns str → directory name
)
```

The `"Feature.column"` dotted-string syntax is **removed**. Calls that
pass it raise `DerivaMLValidationError` at construction explaining the
new form. Users rewrite their one-line `group_by=["Feature.column"]`
as a two-line `targets=["Feature"]` + `target_transform=lambda rec: rec.column`.

Runtime constraint: `target_transform`'s return type is checked at
the first directory-naming call. A non-string return raises
`DerivaMLValidationError` with a message explaining that restructure's
target_transform must return a filesystem-name-compatible string.
(The adapter version has no such constraint.)

### 8.3 New parameter: `missing`

Currently, `restructure_assets` silently places unlabeled assets in an
`Unknown/` directory. This is effectively `missing="unknown"` in the
unified vocabulary.

After alignment:

- `missing: Literal["error", "skip", "unknown"] = "unknown"`
- Default `"unknown"` preserves today's behavior for callers that
  don't explicitly pass the kwarg. This is the one concession to
  continuity: the typical `bag.restructure_assets(output_dir="./out",
  targets=["Diagnosis"])` call produces the same directory tree as
  before.
- `"error"` gives users the strict-mode option (raise at construction
  listing unlabeled RIDs) for when they want to ensure their training
  data is complete before committing to disk.
- `"skip"` drops the asset from the output tree entirely (new behavior;
  no directory for it).

### 8.4 Unchanged restructure parameters

Everything else on `restructure_assets` stays as-is:

- `output_dir`, `asset_table`, `use_symlinks`, `type_selector`,
  `type_to_dir_map`, `enforce_vocabulary`, `file_transformer`.
- The FK-reachability behavior (find assets through foreign-key
  relationships from the dataset).
- The `type_to_dir_map` default (`{"Training": "training",
  "Testing": "testing", "Unknown": "unknown"}`).

These are restructure-specific and don't map to adapter concepts.

### 8.5 Internal refactor: shared resolution helper

The feature-resolution logic (lookup-features-by-name, apply-selector,
handle-missing) gets extracted into a private helper in
`src/deriva_ml/dataset/target_resolution.py` (or similar). Both
`as_torch_dataset` and `restructure_assets` call into it. This makes
the "same semantics" claim in anchor 7 enforceable by shared code
rather than by parallel re-implementation.

Shape of the helper (sketch; implementation-plan will firm up):

```python
def resolve_targets(
    bag: "DatasetBag",
    element_type: str,
    targets: list[str] | dict[str, FeatureSelector] | None,
    missing: Literal["error", "skip", "unknown"],
) -> dict[str, FeatureRecord | dict[str, FeatureRecord] | list[FeatureRecord] | None]:
    """Resolve feature values per element_rid for the requested targets.

    Returns a dict keyed by element RID. The value shape matches §3.3:
    single FeatureRecord, dict of FeatureRecords, or list per target arity.
    Values are None for elements where missing="unknown" / "skip" left
    the label absent.

    Raises DerivaMLException at construction when missing="error" and
    any target is absent, or when a target feature doesn't exist.
    """
```

This helper is private (`_resolve_targets`) — callers reach it through
the public methods, not directly. Naming inside the module is an
implementation-plan detail.

### 8.6 Testing impact

New tests verify the aligned semantics on restructure:
- `targets=[...]` produces the expected directory layout.
- `targets={"A": selector}` applies the selector per-feature.
- `missing="error"` raises on sparse-labeled bags.
- `missing="skip"` omits assets from the output tree.
- `missing="unknown"` places unlabeled assets in `unknown/` (the
  preserved default behavior).
- `target_transform` returning a non-string raises
  `DerivaMLValidationError`.
- Passing `group_by=` raises `TypeError` (unexpected keyword).
- Passing `value_selector=` raises `TypeError` (unexpected keyword).
- Passing a `"Feature.column"` dotted string as a target raises
  `DerivaMLValidationError` with the new-form hint.

**Existing restructure tests that use `group_by` / `value_selector`
must be updated as part of this PR** — the break is intentional.
The implementation plan's verification step includes converting the
existing test suite to the new signature and confirming it still
covers the same behavior.

### 8.7 Documentation impact

Docstring for `restructure_assets` is rewritten to the new signature
— no references to the old parameter names in the docstring body.
A short "Migration note" section near the bottom of the docstring
maps the old call shapes (`group_by=`, `value_selector=`,
`"Feature.column"`) onto the new equivalents so users who find the
docstring while debugging a `TypeError` can translate their code.
The UG Chapter 5 section on `restructure_assets` gets updated
examples in the same style as the adapter section — same vocabulary,
same selector examples, same `missing` callouts. A one-paragraph
"Vocabulary alignment with `as_torch_dataset`" sidebar explains the
shared parameter names.

## 9. Non-goals (explicit)

The following are intentionally **out of scope** to keep the v1
surface focused:

- **Live-catalog adapter variants** (`Dataset.as_torch_dataset`,
  `Dataset.as_tf_dataset`). Covered in anchor 4 above. Named
  follow-up; not v1.
- **DataLoader / `.batch()` construction.** Both adapters return a
  raw `Dataset` (torch) or `tf.data.Dataset` (tf). The user wraps in
  `torch.utils.data.DataLoader` with their own `batch_size`,
  `shuffle`, `num_workers`, or chains `.batch().prefetch()` on the
  TF side. The library does not opine on batching.
- **Train/val/test splits.** Already covered by
  `split_dataset` / `stratify_by_column`. The adapters take whatever
  bag you hand them; splitting happens upstream.
- **In-memory caching across epochs.** Both frameworks have standard
  idioms (`torch.utils.data.DataLoader` persistent workers,
  `tf.data.Dataset.cache()`). The library doesn't layer on top.
- **Inference-result writeback.** That's feature-record territory
  (D1 + existing `add_features` path), orthogonal to the data-loading
  adapters.
- **Label encoding helpers.** Explicit non-goal per anchor 2.
  Users own their encoder; the library surfaces typed records.

## 10. Deliverables checklist

Implementation plan (separate document at
`docs/superpowers/plans/2026-04-24-torch-dataset-adapter-plan.md`)
will enumerate specific tasks; this spec's checklist names the
deliverables the plan must cover.

**Adapter deliverables (PyTorch):**

- [ ] `src/deriva_ml/dataset/torch_adapter.py` — new module
- [ ] `src/deriva_ml/dataset/dataset_bag.py` — `as_torch_dataset`
  method added with full docstring
- [ ] `tests/dataset/test_torch_adapter_no_torch.py`
- [ ] `tests/dataset/test_torch_adapter_logic.py`
- [ ] `tests/dataset/test_torch_adapter_e2e.py` (pytest.importorskip)

**Adapter deliverables (TensorFlow):**

- [ ] `src/deriva_ml/dataset/tf_adapter.py` — new module
- [ ] `src/deriva_ml/dataset/dataset_bag.py` — `as_tf_dataset`
  method added with full docstring (incl. `output_signature`
  inference behavior)
- [ ] `tests/dataset/test_tf_adapter_no_tf.py`
- [ ] `tests/dataset/test_tf_adapter_logic.py`
- [ ] `tests/dataset/test_tf_adapter_e2e.py` (pytest.importorskip;
  verifies `output_signature=None` inference)

**Shared infrastructure (per §8.5):**

- [ ] `src/deriva_ml/dataset/target_resolution.py` — private helper
  `_resolve_targets(bag, element_type, targets, missing)` shared
  among `as_torch_dataset`, `as_tf_dataset`, and `restructure_assets`
- [ ] `tests/dataset/test_target_resolution.py` — tests for the
  shared helper covering the three `missing=` branches and
  selector-dict vs list forms
- [ ] `pyproject.toml` — `[project.optional-dependencies]` table
  with `torch = [...]` and `tf = [...]` extras

**User guide:**

- [ ] `docs/user-guide/offline.md` — new "How to feed a bag to a
  training framework" section per §7.2 covering PyTorch, TensorFlow,
  Keras (both backends), and `ImageFolder`-style layout

**Restructure alignment deliverables (per §8):**

- [ ] `src/deriva_ml/dataset/dataset_bag.py` — `restructure_assets`
  signature updated:
  - Rename `group_by` → `targets` (clean break; old kwarg removed)
  - Remove `value_selector` (replaced by `targets` dict form)
  - Add `target_transform` parameter with string-return check
  - Add `missing: Literal["error", "skip", "unknown"] = "unknown"`
  - Remove `"Feature.column"` dotted syntax in targets
- [ ] `restructure_assets` docstring rewritten to the new signature.
  No references to the old parameter names beyond a single migration
  note (see cross-cutting checklist below).
- [ ] `docs/user-guide/offline.md` — update existing restructure
  section to use new parameter names; add short "Vocabulary
  alignment with `as_torch_dataset`" sidebar
- [ ] `tests/dataset/test_restructure_*` — update existing tests to
  use the new signature (intentional break, not a migration path);
  add new tests per §8.6 covering the aligned semantics
- [ ] Grep sibling repos (`deriva-ml-model-template`, `deriva-ml-apps`,
  `deriva-ml-demo`, `deriva-ml-template-test`) for callers of the
  removed kwargs and flag them in the PR description so template
  maintainers can update in lockstep

**Cross-cutting:**

- [ ] Cross-references per §7.3
- [ ] mkdocs --strict clean (no new warnings)
- [ ] Fast suite green (`tests/local_db/ tests/asset/ tests/model/
  tests/dataset/test_torch_adapter_* tests/dataset/test_restructure_*`)
- [ ] Ruff check + format clean (parity with main baseline)
- [ ] PR description includes a "Breaking changes" section naming
  the removed `restructure_assets` kwargs and the required migration
  (`group_by=` → `targets=`, `value_selector=` → `targets={feature:
  selector}`, `"Feature.column"` → explicit `target_transform`)

## 11. Risks and mitigations

**Risk: PyTorch API changes between versions.** Using only stable APIs
(`torch.utils.data.Dataset`, no tensor ops in the adapter) keeps the
surface insulated from PyTorch point-release churn. Pinning `torch>=2.0`
draws the line at a clear stability boundary.

**Risk: Bag path resolution breaks if a future bag format changes the
`data/assets/<type>/<rid>/...` layout.** Mitigation: the adapter reads
path structure via `bag.path` + the asset table's URL column rather
than hard-coding the layout; if the layout ever changes, the same
code path that breaks `restructure_assets` breaks the adapter, so
they're in sync.

**Risk: Selector-pattern inconsistency between bag and Dataset
variants.** Mitigation: bag-only v1 eliminates this by construction.
When the `Dataset.as_torch_dataset` follow-up lands, the selector
signature is lifted verbatim from the bag variant, preserving the
parity promise.

**Risk: Users expect MLflow-compat `log_metric`-style automatic
metric writeback.** Mitigation: docstring Notes section points at D1's
`exe.metrics_file()` for that use case. The adapter does not write
anything; it reads.

**Risk: Torch dependency bloat for users who install
`deriva-ml[torch]` accidentally.** Mitigation: named extra; pip install
without `[torch]` leaves torch out entirely. The default install
remains lean. Same reasoning applies to `deriva-ml[tf]`.

**Risk: TensorFlow wheel selection is platform-dependent and confuses
users.** On macOS (Apple Silicon) the canonical install is
`tensorflow-macos`, not vanilla `tensorflow`. On CUDA-enabled Linux
it's `tensorflow[and-cuda]`. Mitigation: document the wheel choice in
both the package-extras section of pyproject.toml's PyPI description
and in the UG TensorFlow sub-section. The import guard inside
`as_tf_dataset` just checks that `tensorflow` is importable, so any
properly-installed tf wheel satisfies it regardless of which variant
the user chose.

**Risk: `tf.data.Dataset.from_generator` is slower than
`from_tensor_slices`.** `from_generator` can't be traced into a
tf.graph, so tf.data optimizations (autograph, XLA) don't apply
inside our iteration. Mitigation: anchor 4 (no eager materialization)
is a deliberate choice — the cost is acceptable for research-scale
datasets, and the alternative (eager materialization) would violate
our lazy-loading contract for any bag larger than RAM. Users with
performance-critical pipelines and smaller datasets can use
`.cache()` on the returned `tf.data.Dataset` as a standard TF
escape hatch.

**Risk: `output_signature=None` inference consumes one sample eagerly
at adapter construction.** For pipelines where even the first sample
is expensive (e.g., huge 4D medical volumes), this adds measurable
startup cost. Mitigation: documented in §3.2 and in the docstring;
users with this concern pass `output_signature` explicitly to skip
the eager-infer path.

**Risk: `restructure_assets` parameter rename is a hard break for any
downstream caller using `group_by=`, `value_selector=`, or the
`"Feature.column"` dotted-string syntax in targets.** This is an
explicit choice (§8) — deriva-ml is not yet at the scale where a
deprecation surface pays for itself, and keeping the old names as
aliases would bake in the inconsistency that motivated the rename.
Mitigation is twofold: (1) the implementation plan includes a grep of
sibling repos (`deriva-ml-model-template`, `deriva-ml-apps`,
`deriva-ml-demo`, `deriva-ml-template-test`) to enumerate callers and
flag them in the PR description, so template maintainers can update
in lockstep with this PR landing; (2) the PR description carries a
"Breaking changes" section with a concrete migration mapping so
downstream users who pull the new version see exactly what to
substitute. The `missing` default stays at `"unknown"` so the
most common unparameterized call (`bag.restructure_assets(output_dir=
"./out", targets=["Diagnosis"])`) produces the same tree as before
the PR — the break is contained to callers that were exercising the
renamed surface.

**Risk: Shared `_resolve_targets` helper introduces coupling that
makes future per-method changes harder.** Mitigation: the helper has a
narrow contract (resolve feature values → dict keyed by element RID;
apply missing policy). Method-specific concerns (directory-naming for
restructure, label-in-tuple for adapter) stay in their respective
methods. If a future change genuinely needs divergence, the helper
can be inlined back per-method with a single commit.

## 12. Future work (named for future-spec continuity)

- **Live-catalog adapter variants** (separate future spec):
  `Dataset.as_torch_dataset` and `Dataset.as_tf_dataset` with
  live-download-during-iteration semantics. Requires design work on
  credential lifetime, retry policy, cache eviction, and
  download-during-`__getitem__` performance. The bag-only v1 covers
  the primary training workflow (download once, train many times)
  without any of that complexity.
- **Numpy/Arrow adapters** (lower priority): for users who want
  framework-agnostic iteration over (sample, target) tuples.
- **Streaming / iterable datasets** (lower priority): `IterableDataset`
  variant for bags that exceed local disk.
- **Sampler hooks** (lower priority): `WeightedRandomSampler` +
  class-balance computation from feature values could be a helper,
  but users can do this themselves from the dataset length + target
  list today.
