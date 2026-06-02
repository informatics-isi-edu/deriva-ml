# Split_Partition tag + subsample primitive + CIFAR migration — Design

**Date:** 2026-06-01
**Status:** Draft for review.
**Subproject:** `deriva-ml` (and `deriva-ml-model-template` for CIFAR migration).
**Companion CONTEXT.md update:** new `### Datasets — types and partitions` subsection.
**Follow-up PR (separate):** `characterize_dataset` + `compare_datasets` + `validate_split` / `validate_subsample` — design sketched in task #48, not implemented here.

## 1. Problem statement

A small but real catalog-vocabulary ambiguity has been hiding in
plain sight. Today a downstream consumer that finds a dataset
tagged `Dataset_Type=["Training"]` cannot tell, from the dataset's
type set alone, whether the dataset is:

- a **training corpus** (`Complete` + `Training` + `Labeled`, hand-built or imported), or
- the **training partition of a split** (`Training` + `Labeled`, the Training child of a `split_dataset` call).

Both are legitimately tagged `Training`. The role is the same; the
*origin* differs. The 100K vs. 8K example illustrates how this
matters: `find_datasets(dataset_type="Training")` today returns both
and treats them as equivalent. A user (or agent) asking *"find every
training partition of a split"* or *"find every training corpus that
isn't itself a partition"* has no 1-hop filter to express the
question.

Two further smells surface when this is examined alongside the
existing CIFAR-10 dataset hierarchy in
`deriva-ml-model-template/src/scripts/_cifar10_datasets.py`:

- The canonical CIFAR train/test partition is built **by hand**
  (`exe.create_dataset` + `add_dataset_members`) even though
  `split_dataset(selection_fn=...)` can express the predicate-based
  partition.
- The `Small_Training` / `Small_Testing` family is also built by
  hand via `stratified_sample_rids`, parented under a hand-rolled
  `Small_Split`, because there is no primitive for "stratified
  subsample of a single dataset."

Both smells perpetuate the type ambiguity: every hand-built
hierarchy contributes corpus-role `Training` and partition-role
`Training` datasets to the same vocabulary tag with no
discriminator.

A third concern, surfaced during code review: `split_dataset`'s
selector dispatch has a documented public `random_split` selector
that is **never called from `src/`**. The random shuffle-and-slice
logic is reimplemented inline inside `_compute_partitions`. A
duplication audit is overdue.

## 2. Goals and non-goals

### Goals

- A clean discriminator between corpus-role and partition-role
  dataset_types — one new vocab term, one extra association row
  per split child.
- A new primitive `subsample(source, size, stratify_by_column=...)`
  that mirrors sklearn's `resample(stratify=, replace=False,
  n_samples=...)` semantics but speaks DerivaML's denormalization +
  partition vocabulary.
- The CIFAR-10 reference template uses the library's API for every
  partition it builds. No hand-rolled `Split`-typed datasets.
- One source of truth for the random-selector logic. `random_split`
  is the default selector; the inlined shuffle-and-slice is
  deleted.
- The role-types-don't-inherit-and-don't-propagate rule is made
  explicit in the docstring and vocabulary descriptions.
- Test coverage closes the contracts listed above (parent Split is
  not nested under source; new `Split_Partition` tag lands on each
  child; source `dataset_types` are not mutated; row-mode leakage
  matrix coverage).

### Non-goals

- **Catalog-side enforcement.** No CHECK constraints, no
  triggers, no FK from child_member to source_member, no
  persistent "validated" tag. Existing write-time disjointness
  assert in `split_dataset` stays.
- **A `subsample_split` operation** (parallel-subsample each child
  of an existing Split, returning a mirror Split hierarchy).
  Rejected during grilling: `subsample` is the more fundamental
  primitive; if `subsample_split` is ever needed, it composes
  trivially on top.
- **A `selection_fn` parameter on `subsample`.** Rejected: the
  `SelectionFunction` Protocol returns a multi-key partition dict;
  a single-key degeneration is awkward. Users who need
  hand-curated subsampling logic build the dataset manually via
  `create_dataset` + `add_dataset_members`.
- **A change to the `partition_by` defaulting behavior** in
  `split_dataset`. The current behavior is surprising but
  defensible; it sits on top of a real bug fix
  (curator/02 leakage). Leave alone.
- **A `validate_split` / `validate_subsample` / `compare_datasets`
  / `characterize_dataset` family.** Sketched in task #48 as a
  follow-up PR. Out of scope here.
- **Skill content updates in deriva-ml-skills.** Tracked separately
  in task #49 as a post-PR review.

## 3. Approach

The work breaks into five independent edits, each landable in its
own commit:

### 3.1. Vocabulary seeds

Add two new vocab terms to `_ensure_dataset_types`' seed dict in
`src/deriva_ml/dataset/split.py`:

```python
required_types = {
    # ... existing terms unchanged ...
    "Split_Partition": (
        "A child partition of a Split — set by ``split_dataset`` on "
        "every Training/Testing/Validation child. The discriminator "
        "that distinguishes a split-partition role tag from a "
        "corpus role tag."
    ),
    "Subsample": (
        "A dataset produced by ``subsample()`` as a stratified "
        "sample of another dataset. Source relationship is recorded "
        "in execution provenance, not in Dataset_Dataset edges."
    ),
}
```

`_ensure_dataset_types` runs idempotently inside `split_dataset`
(and will inside `subsample`). Catalogs created against an older
deriva-ml release pick up the new terms on first call.

### 3.2. Apply `Split_Partition` to every child of `split_dataset`

In `_create_split_hierarchy` (`split.py:962+`):

```python
# Before:
train_types = ["Training"] + (training_types or [])
test_types  = ["Testing"]  + (testing_types or [])
val_types   = ["Validation"] + (validation_types or []) if val_size is not None else []

# After:
train_types = ["Training",   "Split_Partition"] + (training_types or [])
test_types  = ["Testing",    "Split_Partition"] + (testing_types or [])
val_types   = ["Validation", "Split_Partition"] + (validation_types or []) if val_size is not None else []
```

The parent Split's `dataset_types` is **unchanged** — it stays
`["Split"]`. The Split is the container, not a partition.

The `split_params` config artifact also receives the updated
`train_types`/`test_types`/`val_types` (already does, via lines
1628-1650 — no separate change).

### 3.3. Document the role-types non-propagation rule

Update `split_dataset`'s docstring (around line 1170-1200, in the
"Provenance" paragraph) to add an explicit rule:

> **Role types do not inherit from the source and do not
> propagate to children.** The Training/Testing/Validation tags
> on the partition children are assigned based on the partition's
> position in the split, **not** copied from the source's
> `dataset_types`. A source tagged `Testing` (because it is a
> testing corpus) produces a Training partition tagged `Training`
> (because that partition is the training half of the split).
> This is intentional: role-axis types describe a dataset's role
> in its *immediate context*, not a property the operation should
> preserve.

The same rule is captured in CONTEXT.md as the **role-axis** vs
**content-axis** vs **origin-axis** decomposition (see CONTEXT.md
update in this PR).

### 3.4. New primitive: `subsample()`

Add to `src/deriva_ml/dataset/split.py` (or a new
`src/deriva_ml/dataset/subsample.py` — see §5):

```python
def subsample(
    ml: DerivaML,
    source_dataset_rid: str,
    execution: Execution,
    *,
    size: int | float,
    seed: int = 42,
    stratify_by_column: str | None = None,
    stratify_missing: Literal["error", "drop", "include"] = "error",
    element_table: str | None = None,
    include_tables: list[str] | None = None,
    via: list[str] | None = None,
    row_per: str | None = None,
    partition_by: Literal["element", "row"] | None = None,
    dataset_types: list[str] | None = None,
    description: str | None = None,
    dry_run: bool = False,
) -> Dataset:
    """Create a stratified subsample of ``source_dataset_rid``.

    Returns one new dataset whose member set is a stratified
    random subset of the source's members. The source relationship
    is recorded as **execution provenance only** — the source is an
    input of ``execution``; the subsample is an output. No
    ``Dataset_Dataset`` edge is created between source and
    subsample (mirroring ``split_dataset``'s design call).

    Mirrors sklearn's ``resample(stratify=y, replace=False,
    n_samples=N)`` semantics: stratified sample without
    replacement.

    See ``split_dataset`` for the meaning of ``stratify_by_column``,
    ``element_table``, ``include_tables``, ``via``, ``row_per``,
    and ``partition_by`` — they pass through to the same
    denormalization machinery.

    Args:
        ml: Connected DerivaML instance.
        source_dataset_rid: The dataset to sample from.
        execution: The caller's open Execution; the subsample is
            attributed to it for provenance.
        size: If float in (0, 1), fraction of source to sample.
            If int, absolute sample count. Mirrors sklearn
            ``train_test_split``'s shape for ``test_size``.
        seed: Random seed for reproducibility.
        stratify_by_column: Optional column for stratified
            sampling (preserves class proportions). When None, the
            subsample is a uniform random sample.
        stratify_missing: How to handle nulls in the stratify
            column. Same semantics as ``split_dataset``.
        element_table, include_tables, via, row_per, partition_by:
            Denormalization / partition controls — same semantics
            as ``split_dataset``.
        dataset_types: Caller-supplied additional dataset types.
            ``"Subsample"`` is always appended; do not include it
            in this list.
        description: Description for the output dataset. When
            None, an auto-description is generated.
        dry_run: If True, return the planned outputs without
            mutating the catalog.

    Returns:
        The new :class:`Dataset` instance.

    Raises:
        ValueError: argument-shape errors (mutually exclusive
            args, missing required args, size <= 0 or >= total).

    Example:
        >>> # Take 400 stratified samples from a Training dataset.  # doctest: +SKIP
        >>> with ml.create_execution(cfg) as exe:  # doctest: +SKIP
        ...     small = subsample(  # doctest: +SKIP
        ...         ml, training_rid, exe,  # doctest: +SKIP
        ...         size=400,  # doctest: +SKIP
        ...         stratify_by_column="Image_Class.Name",  # doctest: +SKIP
        ...         element_table="Image",  # doctest: +SKIP
        ...         include_tables=["Image", "Image_Class"],  # doctest: +SKIP
        ...         dataset_types=["Training", "Labeled"],  # doctest: +SKIP
        ...     )  # doctest: +SKIP
        >>> exe.commit_output_assets()  # doctest: +SKIP
    """
```

#### Implementation strategy

`subsample` reuses the **same denormalization + selector
machinery** as `split_dataset`. The implementation lives behind a
shared helper:

```python
def _compute_subsample(
    source_ds: Dataset,
    *,
    size: int | float,
    seed: int,
    stratify_by_column: str | None,
    stratify_missing: str,
    include_tables: list[str] | None,
    row_per: str | None,
    via: list[str] | None,
    element_table: str | None,
    partition_by: Literal["element", "row"] = "element",
    ignore_unrelated_anchors: bool = False,
) -> tuple[list[str], int, str, str]:
    """Resolve the source members to a single stratified sample.

    Reuses the denormalize → dedupe → stratified-or-random
    selector pipeline from ``_compute_partitions``, but with a
    single-partition shape (``{"Subsample": size}``).
    """
```

Then `subsample()` calls `_compute_subsample` and creates one new
dataset (with `["Subsample"]` + caller-supplied types) — no parent
Split, no children, no `Dataset_Dataset` edges.

`subsample` writes its own config artifact
(`subsample_config.json`) to `execution.working_dir` for
reproducibility — same pattern as `split_dataset`'s
`split_config.json`.

### 3.5. Dispatch refactor in `_compute_partitions`

Today (`split.py:888-895`):

```python
if stratify_by_column:
    selector = stratified_split(stratify_by_column, missing=stratify_missing)
else:
    logger.info("Using custom selection function")
    selector = selection_fn

partition_indices = selector(df, partition_sizes, seed)
```

This branch is reached only when `use_denormalization` is true
(stratify or selection_fn). When **both** are None, the code falls
through to an inlined random shuffle-and-slice
(`split.py:920-930`):

```python
if shuffle:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(all_rids))
    rng.shuffle(indices)
    all_rids = [all_rids[i] for i in indices]

partition_rids = {}
offset = 0
for name, size in partition_sizes.items():
    partition_rids[name] = all_rids[offset : offset + size]
    offset += size
```

This is functionally `random_split` — and `random_split` is the
public, exported, tested helper that the function never calls.

**Refactor:** make `random_split` the default selector. The
dispatch becomes uniform:

```python
if stratify_by_column:
    selector = stratified_split(stratify_by_column, missing=stratify_missing)
elif selection_fn is not None:
    selector = selection_fn
else:
    selector = random_split

# Single call path:
partition_indices = selector(df, partition_sizes, seed)
partition_rids = {
    name: df.iloc[indices][rid_column].tolist()
    for name, indices in partition_indices.items()
}
```

The inlined random branch (`split.py:920-930` and the preceding
non-denormalization fallback) is **deleted**. `use_denormalization`
becomes unconditionally True; the dataframe is always built. (See
§7 for the cost analysis — denormalizing a flat member list is
cheap.)

**Subtle consequence:** the random path now goes through the same
denormalization step that stratified/custom paths use. For the
no-stratify-no-selection-fn case this is a small extra cost — one
denormalize call returning the element table's member rows. In
exchange, we get one selector pipeline instead of two.

If the denormalize cost is unacceptable (TBD, see §7), an
alternative is a fast-path:

```python
if selection_fn is None and stratify_by_column is None:
    # Avoid denormalization. Use random_split directly on the
    # member RID list (no dataframe needed).
    ...
else:
    # Build the dataframe; use selector.
    ...
```

The fast-path keeps `random_split` as the named operation in
both branches. The decision between unified-pipeline and
fast-path is taken in implementation review.

`random_split`'s test class (`TestRandomSplit` in
`tests/dataset/test_split.py`) is unchanged. The new wiring is
covered by gap **D** in §6.

### 3.6. CIFAR-10 migration

In `deriva-ml-model-template/src/scripts/_cifar10_datasets.py`:

#### 3.6.1. Canonical `Split` / `Training` / `Testing` — use `split_dataset` with a predicate selector

Replace lines 445-465 (hand-built Split + Training + Testing +
`add_dataset_members` calls plus the subsequent batched RID
assignment for those three datasets):

```python
def cifar_canonical_partition(
    df: pd.DataFrame,
    partition_sizes: dict[str, int],
    seed: int,
) -> dict[str, np.ndarray]:
    """Partition CIFAR-10 images by the curators' fixed prefix:
    ``train_*`` → Training, ``test_*`` → Testing.

    Ignores ``partition_sizes`` — the canonical partition is
    fully predicate-determined.
    """
    # The denormalized DataFrame must include Image.filename; the
    # caller specifies include_tables=["Image"] accordingly.
    is_train = df["Image.filename"].str.startswith("train_")
    return {
        "Training": np.flatnonzero(is_train.values),
        "Testing":  np.flatnonzero(~is_train.values),
    }

with ml.create_execution(...) as exe:
    canonical = split_dataset(
        ml,
        datasets["complete"],
        exe,
        # test_size is required by split_dataset's signature but is
        # effectively ignored when selection_fn is set; pass any
        # nonzero value just to satisfy validation.
        test_size=len(test_rids),
        selection_fn=cifar_canonical_partition,
        element_table="Image",
        include_tables=["Image"],
        training_types=["Labeled"],
        testing_types=["Labeled"],
        partition_by="element",
        split_description=descriptions["split"],
    )
    datasets["split"]    = canonical.split.rid
    datasets["training"] = canonical.training.rid
    datasets["testing"]  = canonical.testing.rid
```

Note that with `split_dataset` doing the work, the subsequent
`_batched_add` calls for the `training` and `testing` datasets
(lines 506-509) are **redundant** — `split_dataset` already
assigns members. Those calls go away.

The `complete` dataset (line 439-443) **stays hand-built** — it is
the input to the split, not a Split itself.

#### 3.6.2. `Small_Training` / `Small_Testing` — use `subsample`

Replace lines 467-487 (hand-built Small_Split + Small_Training +
Small_Testing) and lines 517-522 (`stratified_sample_rids` calls)
with two `subsample()` calls inside the same execution:

```python
with ml.create_execution(...) as exe:
    # ...the split_dataset call from §3.6.1 above...

    small_training = subsample(
        ml,
        datasets["training"],
        exe,
        size=SMALL_TRAIN_SIZE,
        seed=42,
        stratify_by_column=STRATIFY_COLUMN,
        element_table="Image",
        include_tables=["Image", "Image_Class"],
        dataset_types=["Training", "Labeled"],
        description=descriptions["small_training"],
        partition_by="element",
    )
    small_testing = subsample(
        ml,
        datasets["testing"],
        exe,
        size=SMALL_TEST_SIZE,
        seed=43,
        stratify_by_column=STRATIFY_COLUMN,
        element_table="Image",
        include_tables=["Image", "Image_Class"],
        dataset_types=["Testing", "Labeled"],
        description=descriptions["small_testing"],
        partition_by="element",
    )
    datasets["small_training"] = small_training.dataset_rid
    datasets["small_testing"]  = small_testing.dataset_rid
```

`Small_Split` parent **is dropped entirely**. The two subsamples
are siblings under the same execution's outputs; that's where
the "they go together" relationship lives. The
`stratified_sample_rids` import and helper become unused (delete).

#### 3.6.3. `_require_small_variant_distinct` stays

The pool-size guard at the top of `create_dataset_hierarchy`
(lines 410-412) still runs. The `subsample()` calls will raise if
asked for more samples than the source contains, but failing fast
at the top of the function is still useful operator UX.

#### 3.6.4. `Labeled_Split` and `Small_Labeled_Split` — no change

These already use `split_dataset()`. They pick up the new
`Split_Partition` tag automatically (deriva-ml side) once the pin
moves.

### 3.7. The CIFAR migration's pyproject pin bump

deriva-ml-model-template's `pyproject.toml` pins
`deriva-ml>=1.39,<2.0`. The new `subsample` symbol means the
template needs a pin bump to whatever release this PR ships in
(`>=1.40,<2.0` if the deriva-ml side gets a minor bump, or higher
patch). Coordinated like the v1.39 cycle — deriva-ml PR ships
first, then deriva-ml-model-template PR with the migration plus
pin bump.

## 4. Output format / changed surface

### Vocab terms (`Dataset_Type`)

| Term | New? | Tagged on |
|---|---|---|
| `Split` | existing | Parent Split dataset |
| `Training`, `Testing`, `Validation` | existing | Partition children of a Split (role); also corpora |
| `Labeled`, `Unlabeled`, domain-specific | existing | Content axis |
| **`Split_Partition`** | **new** | Every child of `split_dataset` |
| **`Subsample`** | **new** | Output of `subsample()` |

### Public Python API additions / changes

| Symbol | Status |
|---|---|
| `deriva_ml.dataset.subsample` | NEW — primitive |
| `deriva_ml.dataset.split_dataset` | unchanged signature; new behavior (`Split_Partition` tagging, default selector dispatch) |
| `deriva_ml.dataset.random_split` | unchanged surface; now load-bearing (becomes default selector) |
| `deriva_ml.dataset.stratified_split` | unchanged |
| `deriva_ml.dataset.SelectionFunction` Protocol | unchanged |
| `deriva_ml.dataset.SplitResult` / `PartitionInfo` | unchanged |
| Catalog schema | **unchanged** — no new tables, no new columns. Two new `Dataset_Type` vocab rows seeded by `_ensure_dataset_types`. |

### CLI

No new CLI verbs. (`deriva-ml-split-dataset` already exists;
`subsample` does not get its own CLI because the migration goal
is API-driven; an operator-driven `deriva-ml-subsample` could
follow if demand surfaces.)

## 5. Components and module layout

```
src/deriva_ml/dataset/
    split.py              # split_dataset, subsample (or extract subsample to subsample.py)
    __init__.py           # re-export subsample
tests/dataset/
    test_split.py         # gap A, B, C, D, G mandatory tests added here
    test_subsample.py     # NEW — full subsample coverage
    test_split_partition_by.py  # row-mode leakage matrix tests added here

deriva-ml-model-template/src/scripts/
    _cifar10_datasets.py  # canonical Split and Small Training/Testing migrations

CONTEXT.md                # new ### Datasets — types and partitions subsection
```

### `subsample.py` extraction question (open)

`split.py` is already 2020 lines. Adding ~150 lines of
`subsample` + shared helpers pushes it toward 2200. Extracting
`subsample` to a sibling module
(`src/deriva_ml/dataset/subsample.py`) keeps each file focused on
one operation; the shared `_compute_subsample` and the existing
`_compute_partitions` would share helpers from a third module
(`_split_internals.py` or similar).

**Decision deferred to implementation review.** The cheapest
landing pattern is to add `subsample` to `split.py` first, see
how the helpers actually shape up, and extract if file size or
testability suffers. If the audit cycle has guidance on
2000-line-files (it did — there's a god-file refactor pattern in
the recent audit reports), follow it; otherwise inline is fine.

## 6. Testing strategy

Coverage analysis during grilling identified eight gaps. Five are
mandatory pins (A, B, C, D, G), one is the full subsample suite
(E), one is row-mode leakage coverage promoted from nice-to-have
(F), and one is redundant with the new dispatch (H, skipped).

| Gap | New test | File | LOC |
|---|---|---|---|
| **A** | `test_source_does_not_list_split_as_child` — `source.list_dataset_children()` does NOT include the Split | `test_split.py` | ~15 |
| **B** | `test_split_children_carry_split_partition_tag` — every child's `dataset_types` contains `Split_Partition` | `test_split.py` | ~20 |
| **C** | `test_role_types_dont_inherit_from_source` — source tagged `Testing`, children correctly tagged per role | `test_split.py` | ~25 |
| **D** | `test_split_dataset_default_selector_is_random` — verify the new dispatch wiring | `test_split.py` | ~15 |
| **G** | `test_split_does_not_mutate_source_dataset_types` — defensive | `test_split.py` | ~10 |
| **F** | row-mode leakage matrix: `(element, row) × (random, stratify, custom_fn)` cartesian product — what is and isn't disjoint under each combination | `test_split_partition_by.py` | ~80 |
| **E** | New `TestSubsample` suite: happy path, stratified, dry-run, single-output (no parent), provenance (source = exe input, subsample = exe output, no Dataset_Dataset edge), `Subsample` tag check, role-types-don't-propagate, dataset_types argument honored, deterministic by seed | `test_subsample.py` (new file) | ~250 |

Total: ~12-14 tests, ~400 LOC of test additions.

Doctests on the new `subsample` Example block run as part of
pytest collection (deriva-ml's standard pattern) — all
catalog-dependent lines marked `# doctest: +SKIP`.

## 7. Risks and mitigations

**R1. Random path performance regression.** The dispatch refactor
makes the random selector go through the same denormalization
step as stratified/custom selectors. For very large datasets
where the random path is the hot path, this is a real cost. The
denormalization for the random case is just "give me the
element_table's RIDs," which is one path-builder query — not the
multi-table join the stratified path needs. **Mitigation:** the
fast-path branch sketched in §3.5 is an explicit out if perf
measurement shows a regression. Implementer measures on a 100K
dataset before deciding.

**R2. Hand-built CIFAR canonical may have surprises.** The
predicate `df["Image.filename"].str.startswith("train_")` requires
the denormalized DataFrame to include `Image.filename`. The
existing `include_tables=["Image"]` covers it, but tests should
verify the column exists by name on a real catalog before the
migration lands. **Mitigation:** wrap the CIFAR predicate in a
test that builds a tiny fixture catalog with mixed `train_` /
`test_` prefixes and asserts the partition splits them correctly.

**R3. `Split_Partition` tag missing on old catalogs.** Pre-existing
Splits in long-lived catalogs predate this PR and won't carry the
new tag. `find_datasets(dataset_type="Split_Partition")` will not
return historical split children. **Mitigation:** documented
limitation in the PR description; a one-off backfill script can
be written if a catalog operator needs it (cheap — walk every
`Dataset_Type=Split` parent, find its children via
`Dataset_Dataset`, add the tag to each child). Not provided in
this PR.

**R4. `dataset_types` mutation surprise.** `subsample`'s
`dataset_types` parameter is documented as "caller-supplied
additional types; `Subsample` is always appended." If a caller
passes `dataset_types=["Subsample"]` thinking they need to be
explicit, the result is double-tagging. **Mitigation:**
`_ensure_dataset_types` already deduplicates; defensively dedupe
in `subsample` too.

**R5. Coordinated PR window.** Same risk as the v1.39 cycle —
deriva-ml ships, deriva-ml-model-template needs the new symbol +
pin bump. **Mitigation:** open the deriva-ml-model-template PR
against the deriva-ml feature branch via direct-git pin, the same
pattern that worked for v1.39 / v1.40.

## 8. Rollout

Single PR against deriva-ml `main`, followed within the same
session by a companion PR against deriva-ml-model-template (the
CIFAR migration + pin bump). No feature flag — the new behavior
is additive at the dataset-types level and the dispatch refactor
is invisible to existing callers.

Version bump on deriva-ml: **minor**
(`v1.39.x → v1.40.0`). Justification: new public symbol
(`subsample`), new vocab terms (additive), new dispatch behavior
(behavior change for the no-stratify-no-fn random path — same
results modulo edge-case shuffle determinism, but goes through a
new code path). Not a breaking change; not a patch.

If the dispatch refactor (§3.5) is dropped or scoped down during
implementation, the bump can be patch. The implementer's call.

## 9. Open questions

None blocking. Two minor items the implementer can decide:

1. **Inline vs. extract `subsample` into its own module.** See §5.
2. **Fast-path or unified-pipeline for the no-stratify random
   case.** See §3.5 + §7 R1.

## 10. References

- `CONTEXT.md` — `### Datasets — types and partitions` subsection
  added in the same PR.
- `docs/superpowers/specs/archive/2026-04-30-association-index-sql-generator-design.md`
  — note: unrelated; archived in this same docs tree.
- ADR-0001 — *Lineage walks data flow, not orchestration*.
  Justifies the "source recorded as execution input, not as
  `Dataset_Dataset` parent" call for both `split_dataset` and
  `subsample`.
- Task #48 — sketch of the follow-up PR for
  `characterize_dataset` / `compare_datasets` /
  `validate_split` / `validate_subsample`. Out of scope here.
- Task #49 — `deriva-ml-skills/dataset-lifecycle` review after
  this PR ships. Out of scope here.
- sklearn `resample(stratify=, replace=False, n_samples=N)` —
  conceptual analog for `subsample`. The new function is named
  `subsample` rather than `resample` to avoid the
  bootstrap-resampling connotation.
- Hugging Face `Dataset.train_test_split(stratify_by_column=)` —
  naming inspiration for the keyword.
