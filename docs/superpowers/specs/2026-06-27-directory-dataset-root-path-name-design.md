# Root directory dataset records its name in `Directory_Dataset.Path` (not `"."`) — design

**Date:** 2026-06-27
**Status:** Approved
**Component:** `deriva_ml.core.mixins.file` (`add_files` writer),
`deriva_ml.dataset.dataset` / `dataset_bag` (new `is_source_root` accessor),
`deriva_ml.schema.create_schema` (column comment).
**Relates to:** `docs/superpowers/plans/2026-06-24-directory-dataset-path.md`
(the original `Directory_Dataset.Path` design this amends), tk-024 (the
`add_files` dataset-input edge). Cross-repo: `deriva-ml-cifar-example` and
`deriva-ml-skills` consume the new accessor. Releases as a deriva-ml **minor**
bump.

## Problem

`add_files` builds a nested File-dataset tree and records each node's source
folder in `Directory_Dataset.Path` as a path **relative to the ingest root**.
The ingest root stores the literal `"."` (`core/mixins/file.py:382`):

```python
"Path": "." if directory == ingest_root else directory.relative_to(ingest_root).as_posix(),
```

Children store meaningful names (`train`, `test`); the root stores `"."`. When a
user browses the catalog, the Dataset record page's "Folder" column (Chaise
annotation, `schema/annotations.py:889`) shows the root as `.` — uninformative.
The root's actual name *is* recorded, but in a different column
(`Dataset.Description`, set by `_root_description` to the basename, e.g.
`cifar10_source`). So "what folder is this?" is split across two columns, and the
prominent one shows `.`.

**`"."` is not merely cosmetic — it is a load-bearing root-identification key.**
An audit (2026-06-27) found five functional readers across three repos that use
`source_directory == "."` to *find the root* of the tree (not just display it),
plus four deriva-ml tests asserting the root Path is exactly `"."`, plus
docstrings and the schema column comment. Naively changing the stored value
would silently break root identification everywhere.

### Audit — every reader/writer of `Directory_Dataset.Path`

**Writer (1):** `deriva-ml/src/deriva_ml/core/mixins/file.py:378-386` —
`add_files` inserts the rows; line 382 is the `"."`-for-root conditional.

**Functional readers that identify the root via `== "."` (must migrate):**
- `deriva-ml-cifar-example/src/scripts/load_cifar10.py:177` —
  `_find_latest_source_dataset_rid` filters `source_directory == "."`.
- `deriva-ml-cifar-example/src/scripts/_cifar10_upload.py:319` — comment + logic
  that the root holds `labels.csv` and children are `train`/`test`.
- `deriva-ml-cifar-example/tests/test_lineage_connected.py:214` — `== "."` to
  locate the root.
- `deriva-ml-skills/skills/setup-ml-catalog/scripts/loader_orchestrator_template.py:75`
  — `source_directory == "."` (shipped to users).
- `deriva-ml-skills/skills/setup-ml-catalog/scripts/upload_phase_template.py:102`
  — `partition == "."` root check (shipped to users).

**Value-agnostic readers (NO change needed):** `Dataset.source_directory` /
`Dataset.is_directory` (`dataset.py:172-216`), the `DatasetBag` equivalents
(`dataset_bag.py:228-283`), the Chaise "Folder" annotation
(`annotations.py:889`) — all return/display whatever Path is stored without
assuming `"."`. The child-partition filter `source_directory in {"train","test"}`
(`_cifar10_upload.py:344`) is also unaffected.

**Tests asserting root Path `"."` (must update):**
`deriva-ml/tests/core/test_file.py:173,278`,
`deriva-ml/tests/dataset/test_bag_api_coverage.py:519,530`,
`deriva-ml-cifar-example/tests/test_find_source_dataset.py:78` (stub).

**Docs/comments documenting the `"."` convention (must update):**
`create_schema.py:69`, both `source_directory` docstrings
(`dataset.py:177`, `dataset_bag.py:232`) and their `'.'` examples,
`docs/superpowers/plans/2026-06-24-directory-dataset-path.md`, and the
cifar-example/skills comments listed above.

## Goal

1. The root directory dataset records its **directory basename** (e.g.
   `cifar10_source`) in `Directory_Dataset.Path`, so the catalog "Folder" column
   is self-describing for every node.
2. Root identification no longer depends on a magic Path string. A **structural,
   name-independent** rule — exposed as a reusable accessor — replaces every
   `== "."` reader, and works on **both old and new catalogs**.

## Design

### 1. Structural root rule (`is_source_root`)

**Definition.** A dataset is the *source root* of an `add_files` tree iff it has
a `Directory_Dataset` row **and** none of its parents (via `Dataset_Dataset`,
where `Dataset` = parent and `Nested_Dataset` = child) is itself a directory
dataset (has a `Directory_Dataset` row).

This is the dataset's position in the parent graph, independent of the Path
string — so it identifies the root whether the stored Path is the legacy `"."`
or the new basename. Verified live on catalog 328: the rule matches exactly the
root `AT4`, excludes the `train`/`test` children, and (critically) does **not**
match the later `split_dataset()` datasets (`ZB8`, `Y7J`, `T18`) — those are
parents of their own children but have **no** `Directory_Dataset` row, so the
"is a directory dataset" clause excludes them. Scoping to directory datasets is
what keeps the rule from mis-firing on non-`add_files` nesting.

**New accessor** — mirror the existing `source_directory` / `is_directory`
shape, on both classes:
- `Dataset.is_source_root` (property, live) — `dataset.py`.
- `DatasetBag.is_source_root` (property, offline bag) — `dataset_bag.py`.

Returns `True` for the tree root, `False` for child directory datasets, `False`
for datasets with no `Directory_Dataset` row (non-directory datasets and
`split_dataset()` outputs).

Implementation: `is_directory` (already exists) gives the "has a
`Directory_Dataset` row" half. For the "no directory-dataset parent" half, fetch
this dataset's parents (`list_dataset_parents` exists on `Dataset`; the bag has
its own membership tables) and test whether any parent is itself a directory
dataset. Keep the parent lookup scoped so a single small query suffices (the
tree is shallow; a root has no directory-dataset parents by definition).

The five readers replace `source_directory == "."` with `is_source_root`. Where
a reader needs *the* root RID from a set of candidates
(`_find_latest_source_dataset_rid`), it filters its candidate datasets by
`is_source_root`.

### 2. Writer change (`add_files`, `core/mixins/file.py`)

The root stores its basename instead of `"."`:

```python
"Path": _root_path_name(ingest_root, root_name)
        if directory == ingest_root
        else directory.relative_to(ingest_root).as_posix(),
```

where the root's Path name follows the **same precedence as the root
Description** so the two columns agree:

```python
def _root_path_name(ingest_root: Path, root_name: str | None) -> str:
    # Mirrors _root_description's precedence (minus the description fallback):
    # explicit root_name, else the basename, else the "root" sentinel — never empty.
    return root_name or ingest_root.name or "root"
```

- **Empty-basename edge** (`Path("/").name == ""`, when file specs span
  top-level dirs): falls through to `"root"`, never empty. (`_root_description`
  already guards this for the Description; we reuse the same precedence.)
- **`root_name` consistency:** today Description respects a caller-supplied
  `root_name` but Path is always `"."` — they can disagree. After this change
  both derive the root's name from the same precedence, so Path and Description
  name the root identically.
- Children are **unchanged** — still `directory.relative_to(ingest_root).as_posix()`.

Factor `_root_path_name` alongside `_root_description` so the shared precedence
lives in one place (DRY); `_root_description` keeps its extra `description`
fallback for the prose field.

### 3. Schema + docs

- `create_schema.py:69` column comment: replace "The ingest root stores '.'."
  with language describing that the root stores its directory basename (the same
  name as its Description), children store paths relative to the root.
- `source_directory` docstrings (`dataset.py`, `dataset_bag.py`): drop the
  "ingest root stores '.'" sentence and the `'.'` doctest examples; document the
  new behavior and point to `is_source_root` for root identification.
- `is_source_root`: full Google-style docstring with a runnable `Example:`.
- Amend `docs/superpowers/plans/2026-06-24-directory-dataset-path.md` with a note
  that the root Path convention changed from `"."` to the basename, and why.

## Cross-repo migration & ordering

deriva-ml ships first (defines `is_source_root`); consumers then pin to it.

**deriva-ml** (minor bump): writer change, `is_source_root` (×2 classes),
schema-comment + docstring updates, tests (below).

**deriva-ml-cifar-example** (lock bump to new deriva-ml):
`load_cifar10.py:177` and `test_lineage_connected.py:214` switch
`source_directory == "."` → `is_source_root`; `_cifar10_upload.py:319` comment +
root reasoning updated (its child filter `in {"train","test"}` is unchanged).

**deriva-ml-skills** (template fixes): `loader_orchestrator_template.py:75` and
`upload_phase_template.py:102` switch `== "."` → `is_source_root`, with their
surrounding comments.

### Existing catalogs — no backfill

`is_source_root` keys off the parent graph, not the Path string, so it identifies
the root correctly on **pre-change catalogs** (whose root Path is still `"."`)
*and* new ones. Therefore root identification keeps working everywhere with no
migration. The only residue on an already-loaded catalog is cosmetic: its root
"Folder" still displays `"."`. Rewriting old roots' Path is **out of scope** — not
worth a migration for a display nit; new loads get the descriptive name. (If a
backfill is ever wanted, it is a separate, optional one-shot: for each dataset
where `is_source_root` and `source_directory == "."`, set Path to the root's
Description/basename.)

## Testing

### deriva-ml — offline (`tests/core/test_file.py`)
- After `add_files`, the root row's Path equals the ingest-root **basename**
  (not `"."`); children unchanged (`{"d1","d2"}`).
- A `root_name="Custom Name"` case: the root row's Path **and** its Description
  both equal `"Custom Name"` (Path/Description agree).
- The four existing `== "."` assertions
  (`test_file.py:173,278`, `test_bag_api_coverage.py:519,530`) become
  basename assertions.

### deriva-ml — `is_source_root` (live + bag)
- `is_source_root` is `True` for the tree root, `False` for `train`/`test`
  children, `False` for a non-directory dataset and for a `split_dataset()`
  output (the parent-but-not-directory case the rule must exclude).
- **Identity-not-name test:** locate the root via `is_source_root` and assert it
  is the expected dataset *without* referencing any Path string — so root
  identification can never silently regress to depending on `"."` or on the
  basename value.
- **Legacy-catalog test (offline/bag):** a fixture whose root Path is `"."`
  (the pre-change shape) still resolves `is_source_root == True` for the root —
  proves no backfill is required.

### deriva-ml-cifar-example
- `test_find_source_dataset.py`: stub/filter switches to `is_source_root`; the
  test asserts the root is found regardless of its Path value.
- `test_lineage_connected.py`: root located via `is_source_root`; lineage still
  connects end-to-end.
- The existing live end-to-end load confirms the real root's Folder reads its
  basename (e.g. `cifar10_source`).

## Acceptance criteria

1. A fresh `add_files`/`load-cifar10` catalog shows the root's "Folder" as its
   directory basename (e.g. `cifar10_source`); children show `train`/`test`.
2. All five former `== "."` readers identify the root via `is_source_root` and
   work on both new and pre-change catalogs.
3. Root Path and Description name the root identically (incl. when `root_name` is
   supplied).
4. End-to-end lineage still connects source → split → run.
5. No schema migration and no catalog backfill required.

## Out of scope

- Backfilling existing catalogs' root Path from `"."` to the basename.
- Changing child Path semantics (they remain relative to the ingest root).
- Any change to `Dataset.Description` derivation beyond sharing the name
  precedence with the new root Path.
