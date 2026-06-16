# tf/torch adapters: enumerate FK-reachable elements (match restructure_assets)

**Date:** 2026-06-16
**Status:** Design — approved approach (reachable-by-default + shared helper)

## Problem

`DatasetBag.as_tf_dataset(element_type="Image")` and `as_torch_dataset(...)`
return an **empty generator** for datasets whose elements are FK-*reachable*
but not *direct* members — e.g. a subject-partitioned dataset (members are
`Subject`s; images reachable via `Subject → Observation → Image`). The older
`restructure_assets(asset_table="Image")` path handles these correctly. This
is a behavioral regression for anyone migrating from
`restructure_assets`/`ImageDataGenerator` to the framework adapters, and it
blocks every eye-ai LAC/kyle training dataset (all subject-partitioned).

## Root cause (verified in source)

The two paths enumerate elements differently:

- `restructure_assets` → `_get_reachable_assets(bag, table)`
  (`dataset/restructure.py:207`) → `bag._dataset_table_view(table)`
  (`dataset/dataset_bag.py:365`): a SQL UNION over **all FK paths** from
  `Dataset` to the target table, scoped to the dataset RID + nested-dataset
  RIDs. Finds FK-reachable rows.
- `as_tf_dataset` → `tf_adapter.build_tf_dataset`
  (`dataset/tf_adapter.py:102`): `members_by_type =
  bag.list_dataset_members(recurse=True)` then `all_rids = [m["RID"] for m in
  members_by_type[element_type]]` (line 126). `list_dataset_members(recurse=True)`
  recurses **nested Datasets only** — it does not traverse the
  `Subject→Observation→Image` FK chain. So `members_by_type["Image"]` is empty
  → empty generator → raises at `tf_adapter.py:166`.
- `torch_adapter.build_torch_dataset` (lines 91/115) has the identical
  structure.

**Reproduction** (eye-ai `2-277G` v4.8.0): `list_dataset_members(recurse=True)`
→ `{Subject: 4212, Dataset: 49}`, 0 Images. `_get_reachable_assets(bag,
"Image")` → 28,546 Images. `as_tf_dataset(element_type="Image", ...)` raises;
`restructure_assets(asset_table="Image", ...)` finds 28,546.

## Key finding that scopes the fix tightly

Only the **RID set** is wrong. The surrounding machinery already operates on
the whole table, not direct members:

- `_build_row_lookup(bag, element_type)` (`tf_adapter.py:202`) uses
  `bag.get_table_as_dict(element_type)` — the **whole element table** from the
  bag's local SQLite, keyed by RID. So the row dict for every reachable RID is
  *already present* in `row_lookup`.
- `_resolve_targets(bag, element_type, ...)` (`target_resolution.py`) joins
  feature values by RID — RID-keyed, works for any RID.
- `_resolve_asset_path`, `missing=` handling — operate per-RID, source-agnostic.

So the fix replaces *where `all_rids` comes from* (direct members →
reachability) and the *existence validation*. Everything downstream is
unchanged — exactly as the report requested.

## Design

### Architecture: ONE shared enumeration core, THREE consumers

The three paths have two layers. Only the first is currently divergent (and
buggy); the second is legitimately different and must NOT be merged:

| Layer | restructure | tf | torch | share? |
|---|---|---|---|---|
| **(1) "which rows of `table` are reachable from this dataset?"** | `_get_reachable_assets` (FK) | `list_dataset_members` ✗ | `list_dataset_members` ✗ | **YES** |
| **(2) what to do per element** | symlink/copy into `output_dir` tree | `_resolve_asset_path` + `sample_loader` + target → lazy `yield` | same as tf | **NO** |

Layer (2) is each consumer's actual job (materialize an on-disk ImageFolder
tree vs. lazily yield `(sample, label, rid)` into a streaming dataset) —
different return types, different laziness. Forcing them together would be
wrong. The bug is entirely in layer (1): all three need the same FK-reachable
row set, but restructure computes it right and the adapters compute it wrong.

So: extract **one** layer-(1) core that all three route through.

### 1a. The shared core (new) — returns full rows

```python
def resolve_reachable_rows(bag: "DatasetBag", table: str) -> list[dict]:
    """All rows of `table` reachable from this dataset via any FK path.

    The single source of truth for dataset element reachability. Wraps
    bag._dataset_table_view(table) (Dataset -> ... -> table UNION over all FK
    paths, scoped to this dataset + nested datasets). Returns full row dicts so
    restructure (needs Filename etc.) and the adapters (project RID) share one
    traversal. Rows are NOT RID-deduped here: a UNION over multiple FK paths can
    surface the same RID twice, and restructure collapses by filename — callers
    that iterate by RID must dedup (see resolve_element_rids).
    """
```

Place it in `dataset/target_resolution.py` (already imported by both adapters)
or a new `dataset/_reachability_view.py`. `_get_reachable_assets` in
`restructure.py` becomes a thin wrapper over (or is replaced by) this — so
restructure's existing live coverage also protects the adapters' enumeration.

### 1b. The adapter projection (new) — RIDs, deduped

```python
def resolve_element_rids(
    bag: "DatasetBag", element_type: str, *, reachable: bool = True
) -> list[str]:
    """RIDs of `element_type` rows belonging to this dataset, order-preserving,
    RID-deduped (a UNION can surface a RID via two FK paths → don't double-yield).

    reachable=True (default): resolve_reachable_rows (FK reachability — same
    semantics as restructure_assets / bag_info).
    reachable=False: direct members only (list_dataset_members(recurse=True)).
    Raises DerivaMLException if element_type is not resolvable in the bag.
    """
```

- **reachable=True**: `resolve_reachable_rows(bag, element_type)`, project
  `row["RID"]`, **order-preserving dedup** (the bug-prevention point — restructure
  doesn't dedup because it keys by filename; the adapters iterate RIDs so a
  path-duplicate RID would yield a sample twice). Works for asset AND non-asset
  element types (`_dataset_table_view` is general).
- **reachable=False**: `[m["RID"] for m in
  bag.list_dataset_members(recurse=True).get(element_type, [])]` — old behavior,
  preserved as the opt-out.
- **Validation**: replace the current `element_type not in members_by_type`
  check. Reuse the bag's existing table-existence check (the one
  `get_table_as_dict` relies on) so an unknown `element_type` still raises a
  clear error listing available types.

### 2. Wire both adapters through it

In `build_tf_dataset` and `build_torch_dataset`, replace:

```python
members_by_type = bag.list_dataset_members(recurse=True)
if element_type not in members_by_type: raise ...
...
all_rids = [m["RID"] for m in members_by_type[element_type]]
```

with:

```python
all_rids = resolve_element_rids(bag, element_type, reachable=reachable)
```

(the existence check moves into the helper). Everything else in each adapter —
`is_asset` / `sample_loader` validation, `_resolve_targets`, the
`targets`/`missing` RID filtering, `_build_row_lookup`, the generator body — is
**unchanged**.

### 3. Surface `reachable` on the public methods

Add `reachable: bool = True` to `DatasetBag.as_tf_dataset` and
`as_torch_dataset` (`dataset_bag.py:1373` / `:1199`), forwarded to the
builders. Default True = the fix (reachable). Pass `reachable=False` to force
direct-members-only (the old behavior, for callers who want it).

**Default is reachable** (the user's choice): the adapters then agree with
`bag_info`, `restructure_assets`, and `_get_reachable_assets` on "what's in the
dataset." For a dataset with no FK indirection, the reachable set == the direct
set, so default-True is a no-op there; it only *adds* the currently-missing
elements for subject-partitioned datasets.

## Testing

1. **Unit (catalog-free where possible):** `resolve_element_rids` with a stub
   bag — `reachable=True` calls `_dataset_table_view`; `reachable=False` calls
   `list_dataset_members`. Assert the right source is used and RIDs returned.
2. **Adapter wiring (source-level):** assert both `build_tf_dataset` and
   `build_torch_dataset` call `resolve_element_rids` (not
   `list_dataset_members` directly) and expose `reachable`.
3. **Live regression (demo catalog):** build a subject-partitioned dataset on
   the demo catalog (Subjects as members, Images via Subject→...→Image) and
   assert `as_tf_dataset(element_type="Image")` yields the FK-reachable images,
   not empty — and that the count matches `_get_reachable_assets` /
   `restructure_assets` on the same bag.
4. **Acceptance (live eye-ai, this session has access):** on `2-277G` v4.8.0,
   `as_tf_dataset(element_type="Image", targets={"Image_Diagnosis":
   select_initial_diagnosis}, missing="skip")` yields ~28,546 `(image, label,
   rid)` triples (matching `restructure_assets` / `_get_reachable_assets`), not
   an empty generator. Same count for `as_torch_dataset`.
5. **Opt-out preserved:** `reachable=False` reproduces the old direct-members
   set (asserts the escape hatch works and didn't change).

## Risks / things to watch

- **Row-shape parity:** `_dataset_table_view` rows and `get_table_as_dict` rows
  both come from the same bag SQLite table, so `row_lookup` (keyed by RID)
  resolves every reachable RID. Verify in the live test that
  `_resolve_asset_path` finds the file for FK-reachable RIDs (it uses
  `row["Filename"]` from `row_lookup`, which is whole-table → present).
- **Duplicate RIDs from the UNION:** `_dataset_table_view` UNIONs multiple FK
  paths; the same element RID may appear via two paths. `_get_reachable_assets`
  returns rows as-is. The helper should **dedupe RIDs** (preserve order) so the
  adapter doesn't yield the same sample twice. (restructure may not dedupe
  because it writes by filename; the adapters iterate RIDs, so dedupe matters
  here — confirm against the live count, which should be distinct images.)
- **Non-asset element types:** default-reachable must still work for a
  non-asset `element_type` (e.g. `Subject`). `_dataset_table_view` handles it;
  the test matrix should include one non-asset case.
