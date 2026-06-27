# add_files: O(1) dataset-Input edge instead of O(N) per-file edges — design

**Date:** 2026-06-27
**Status:** Approved
**Component:** `deriva_ml.core.mixins.file` (`add_files`), `deriva_ml.execution.execution` (`Execution.add_files` wrapper)
**Relates to:** tk-005 (the by-reference Input provenance pattern), tk-022 (the lineage self-parent guard this relies on), tk-024 (the analysis). Releases as deriva-ml **1.53.0** (behavior change to a released API).

## Problem

`add_files` registers external files by reference and links **each file** to the
execution as an Input: it inserts one `File_Execution` row with
`Asset_Role="Input"` **per file** (`core/mixins/file.py:303-308`, a batched
insert, but O(N) rows in file count). Two costs at scale:

1. **Write/storage:** N extra association rows per `add_files` call (2M files →
   2M extra rows on top of the File + File_Asset_Type rows).
2. **Lineage read/render:** `lookup_lineage` pulls all N File Inputs into a single
   `LineageNode.consumed_assets`. At millions of files this is the same
   large-fetch class tk-023 just fixed for member-producers, and an enormous node
   to serialize. (Confirmed live: the CIFAR source-registration execution `4AP`
   in catalog 278 carries 2001 `File` Inputs — 2000 images + `labels.csv`.)

These per-file edges are redundant for lineage **traversal**: `lookup_lineage`
reaches the registration execution via `Dataset_Version.Execution` (the dataset's
producer edge), and the files are reachable via dataset membership — exactly like
a regular dataset. Verified live: `_producer_of_dataset('M0J') == '4AP'`;
`_producers_of_dataset_members('M0J') == []`; the 2001 File_Execution rows are
read only to *populate* `4AP.consumed_assets` for display.

## Goal

Make `add_files` record input provenance in **O(1)** — one association row,
independent of file count — while keeping provenance truthful and complete, with
no public-model change and no lineage-side cap.

## Design

`add_files` stops inserting per-file `File_Execution` Input rows and instead
inserts **one** `Dataset_Execution` row with `Asset_Role="Input"`: the created
source dataset (the root of the nested tree) declared as the registration
execution's **Input**. The dataset remains the execution's **Output**
(`Dataset_Version.Execution`, written by `create_dataset(execution_rid=...)`,
unchanged). So the execution both *produces* the source dataset and *declares it
as a consumed input* — at dataset granularity, the registration both defines and
consumes the source set.

### What is preserved (verified live, all independent of the per-file edges)

- **Producer of the dataset:** `Dataset_Version.Execution(root) = execution_rid`
  — written by `create_dataset(execution_rid=...)` inside `add_files`
  (`file.py:338`). Untouched.
- **Files in the dataset:** `Dataset_File` membership. Independent of
  `File_Execution`. Untouched. (You walk the dataset to get its files.)
- **Lineage traversal:** unchanged — the walk reaches the registration execution
  via the producer edge, and `lookup_lineage(<source dataset>)` is unaffected.

### Why the dataset-Input self-loop is safe (lineage)

The new edge makes the source dataset both an Output (`Dataset_Version.Execution`)
and an Input (`Dataset_Execution`) of the same execution. When `lookup_lineage`
walks the registration execution and lists this consumed dataset, it computes
`_producer_of_dataset(dataset) == execution_rid` — i.e. producer == the execution
being expanded. The tk-022 self-parent guard in `_walk_node`
(`if producer and producer != execution_rid: parent_rids.add(producer)`) **skips
adding the execution as its own parent**, so there is **no false cycle** and no
re-expansion. The dataset appears once as a `consumed_dataset`; the walk does not
loop. (Verified by trace against catalog 278.)

### Why provenance enforcement needs no change

`ensure_artifact_producer_has_input` (`provenance_enforcement.py`, fired from
`dataset.py:2613` when an execution authors a dataset) calls `_execution_has_input`,
which returns True if there is **any** `Dataset_Execution` input row **or** any
`Asset_Role="Input"` asset. The single `Dataset_Execution` Input row satisfies
this. So: no "unknown-provenance" sentinel is linked, no warning is emitted, and
**no enforcement logic changes.**

### Components

**1. `add_files` (`core/mixins/file.py`).**
- Remove the per-file `File_Execution` Input insert (the `file_execution_path.insert([...])`
  block, ~lines 300-308) and the `file_execution_path = ...` lookup if it becomes
  unused (~line 270).
- After the dataset tree is built and the root dataset is known (the node whose
  directory == `ingest_root`, i.e. `node_dataset[ingest_root]`), insert **one**
  `Dataset_Execution` row: `{Dataset: <root dataset RID>, Execution: execution_rid,
  Asset_Role: "Input"}`. Resolve the association/columns the same way the rest of
  the codebase does (`self.model.find_association("Dataset", "Execution")` →
  `(assoc, dataset_fk, execution_fk)`, then `pb.schemas[...].tables[...].insert([...],
  on_conflict_skip=True)`), to match `_link_file_sentinel_as_input`'s idempotent
  pattern. The plan pins the exact Dataset_Execution column names + whether a
  `Dataset_Version` FK column must be set (mirror how `Execution.execute`/the
  `datasets=` input path writes a `Dataset_Execution` Input row — see
  `execution.py` around the `Dataset_Execution` insert that records `datasets=`
  inputs).
- The `File` rows, the `File_Asset_Type` tags, the nested `Dataset` tree, the
  `Directory_Dataset` rows, and `Dataset_Version.Execution` are all UNCHANGED.

**2. Docstrings (both `add_files` signatures).**
Update `core/mixins/file.py:add_files` and `execution/execution.py:Execution.add_files`
docstrings: replace "links each file to the execution as an Input
(`File_Execution.Asset_Role="Input"`)" with language describing that the
registered **source dataset** is recorded as the execution's Input (one
`Dataset_Execution` Input edge), at dataset granularity — and note that
per-file consumption edges are intentionally not written (find consumers via the
dataset). Keep the "files are references, not Hatrac uploads; produced files use
asset_file_path" framing.

### No public-model / signature change

`LineageNode` / `AssetSummary` (`execution/lineage.py`) are unchanged — no
lineage cap is needed because the N-asset bloat is eliminated at the source. The
`add_files` signatures (both) are unchanged (no new parameter — this is a default
behavior change, per the design decision to change the default rather than gate
it behind a flag).

## Behavior change & migration

This changes a released API's behavior: after 1.53.0, `add_files`-registered
files are no longer individually linked to the registering execution as Inputs.
- `find_executions_consuming(<a single File RID>)` returns empty for such files
  (it depends on the `File_Execution` Input edge). Find consumers via the
  **dataset** instead (`find_executions_consuming(<dataset RID>)` / dataset
  membership). Document in the changelog.
- Existing catalogs are not rewritten; only new `add_files` calls take the new
  shape. (A backfill/migration is out of scope.)

Minor version bump (1.52.x → 1.53.0): new behavior, backward-compatible at the
API-signature level, behavior change documented.

## Testing

### Offline / unit (`tests/core/test_file.py` and/or a focused new test)

The existing tests assert the per-file `File_Execution` Input rows exist; update
them to the new contract:
- **O(1) input edge:** after `add_files` of N files (use a small N, e.g. 5),
  assert **exactly one** `Dataset_Execution` Input row exists for the execution
  (Dataset == root dataset RID, Asset_Role == "Input"), and **zero**
  `File_Execution` Input rows were written. The "O(1), not O(N)" property is the
  point — assert the File_Execution Input count is 0 regardless of N.
- **`_execution_has_input` True:** assert `ensure_artifact_producer_has_input`
  does NOT link the unknown-provenance sentinel for an `add_files` execution
  (i.e. `_execution_has_input` returns True from the Dataset_Execution row), so no
  "input unknown" warning/sentinel appears.
- **Producer + membership intact:** assert `Dataset_Version.Execution(root) ==
  execution_rid` and that the dataset's File members are unchanged (the existing
  membership assertions stay green).
- Update any other deriva-ml test that asserted the per-file File_Execution Input
  rows (grep `File_Execution` under `tests/` — e.g. `test_file.py`,
  `test_local_file_input.py`, `test_asset_table.py` if they touch the add_files
  Input edges) to the new dataset-edge contract. Do NOT weaken — re-express the
  same provenance assertion at dataset granularity.

### Live (`tests/execution/test_lookup_lineage_live.py`, DERIVA_HOST-gated)

- **Large add_files lineage:** register a dataset via `add_files` with a
  meaningful number of files (e.g. ≥50 — enough that the OLD code would attach
  ≥50 consumed_assets), then `lookup_lineage(<the source dataset>)` and assert:
  (a) the registration execution appears, (b) its `consumed_assets` is empty (no
  per-file File Inputs), (c) its `consumed_datasets` includes the source dataset
  (the one Input edge), (d) `cycle_detected is False` (the self-loop is guarded),
  (e) the walk completes. This proves the render bloat is gone AND the self-loop
  is safe end-to-end.

## Out of scope

- Backfilling existing catalogs' per-file edges into dataset edges.
- Any lineage-side `consumed_assets` cap (unnecessary once the source stops
  writing N edges).
- Changing how *produced* assets (asset_file_path / commit_output_assets) record
  Output edges — only the `add_files` by-reference Input path changes.
