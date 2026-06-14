# Fast descendant-tree walk for nested datasets — Design

**Date:** 2026-06-14
**Status:** Approved in brainstorming; spec for implementation.
**Subproject:** `deriva-ml`

## 1. Problem statement

`estimate_bag_size(2-277G)` takes ~464 s. The just-shipped `/schema` fix
(model-built pathBuilder) cut `/schema` GETs 849 → 8 but barely moved
wall-clock (510 s → 464 s) — proving `/schema` was *not* the time sink.

Profiling the post-fix run (2026-06-14, live www.eye-ai.org, 2-277G, 84
descendants / 85 anchors) shows the cost is **client-side descendant-tree
enumeration**: 1,543 serial catalog GETs, ~0.3 s each.

**Decisive comparison.** `generate_dataset_download_spec(2-277G)` (the
client-side work a *real* bag download does before handing the export to the
server) costs **1,529 GETs / 366 s** — essentially identical to the estimate.
So "knowing the bag size" costs ~as much as "preparing to build the bag," and
the shared cost is the descendant walk via `DatasetBagBuilder.anchors_for`.
**Fixing the walk speeds up both `estimate_bag_size` and every real
`download_dataset_bag` on a nested dataset.**

### Measured breakdown of the ~1,543 GETs (85-node tree)

Two redundancies plus one genuine-but-reducible cost:

| Source | GETs | Cause |
|---|---|---|
| `lookup_dataset` (total) | 592 | per-node, 4–7× over-called |
| `list_dataset_children` | 170 | called per-node (manual re-recursion); **each fetches the entire `Dataset_Dataset` table** |
| per-descendant member scan | ~765 | `_exclude_empty_associations` calls `list_dataset_members()` for each of 85 RIDs; each loops every Dataset association table (`85 × ~9 assoc`) |
| `/schema` | 8 | already fixed |

Root structural facts (source-confirmed):

- **`_iter_descendant_rids` (`bag_builder.py:892`) manually recurses**,
  calling `Dataset.list_dataset_children()` (non-recursive) **once per node
  (170×)**. But `list_dataset_children(recurse=True)` *already* walks the whole
  subtree from a **single** `Dataset_Dataset` fetch (`dataset.py:2391` fetches
  the full table; `find_children` recurses in-memory). The manual re-recursion
  throws that away → 170 full-table fetches instead of 1.
- **`list_dataset_children` hydrates a `Dataset` per child** via
  `lookup_dataset` (`dataset.py:2405`) — one GET per descendant. The bag walk
  only needs **RIDs**, not `Dataset` objects.
- **The whole walk runs twice** — `anchors_for` and
  `_exclude_empty_associations` each call `_iter_descendant_rids`.
- **`_exclude_empty_associations` (`bag_builder.py:824`)** then scans members
  per-descendant: for each of 85 RIDs, `list_dataset_members()`
  (`dataset.py:1602`) loops `dataset_table.find_associations()` and does a
  filtered `.entities().fetch()` per association — `85 × ~9 ≈ 765` queries —
  only to compute a **boolean per element type** ("does the tree contain any
  member of type X?").

## 2. Goals

- Cut the descendant-walk GET count from ~1,500 to a few hundred, dropping
  `estimate_bag_size(2-277G)` and the bag-download spec-gen from ~460 s toward
  **well under a minute** (target: re-measure; aim < 60 s, ideally < 30 s).
- **No change to results.** Same anchor RID set, same `exclude_tables`
  decisions, same estimate dict. Pure de-duplication / query-shape change.
- Benefit is **shared**: `estimate_bag_size` and `download_dataset_bag` both
  improve (they share `anchors_for` / `build_policy`).
- The estimate stays **version-pinned** (it already takes `version` /
  `DatasetSpec`); this change must not regress to current-catalog.

## 3. The fix — three coordinated changes, all in deriva-ml

### Part A — `_iter_descendant_rids`: one recursive fetch, RIDs only

`_iter_descendant_rids` should obtain the full descendant-RID set from a
**single** `Dataset_Dataset` fetch, without per-node `lookup_dataset`.

- Add a RID-only descendant accessor. Two clean options (plan picks one):
  - **A1 (recommended):** a `Dataset.list_dataset_children_rids(recurse=True)`
    that runs the same `find_children` logic over the one full-table fetch but
    returns **RIDs** (skips the `lookup_dataset` hydration at `dataset.py:2405`).
  - **A2:** a `lookup: bool = True` (or `rids_only`) flag on
    `list_dataset_children` that, when False, returns RIDs instead of `Dataset`
    objects. (More API surface on a public method; A1 keeps the public method
    unchanged.)
- `_iter_descendant_rids` calls the recursive RID accessor **once** and returns
  its result, instead of manual per-node recursion.

Effect: 170 `Dataset_Dataset` fetches → **1**; ~592 `lookup_dataset` → **~0**
for the walk.

### Part B — `_exclude_empty_associations`: aggregate member-presence, not per-descendant lists

The method only needs, per Dataset association table `Dataset_X`, a **boolean**:
"does any dataset in the tree (root + descendants) have ≥1 member of element
type X?" It currently computes this by materializing every member of every
association for every descendant (765 queries).

Replace with **one membership query per Dataset association table**, scoped to
the descendant-RID set:

- For each `assoc` in `dataset_table.find_associations()`, issue a single
  query: "does `assoc.table` have any row whose `Dataset` FK is in
  `{root} ∪ descendants`?" using `Dataset.in_(rid_set)` + a `limit=1` (or a
  `cnt(RID)` aggregate). One round-trip per association table (~9), independent
  of descendant count.
- Vocabulary-linked associations stay always-included (unchanged rule).

Effect: ~765 per-descendant member queries → **~9** (one per Dataset
association table). This is the **largest** single reduction.

> Correctness note: the existing logic adds an element type to
> `member_element_types` if *any* descendant has a member of it; the aggregate
> "any row with Dataset in rid_set" computes exactly the same set. The
> `limit=1`/`cnt` form returns presence, which is all the boolean needs.

### Part C — memoize the descendant set per `DatasetBagBuilder` op

`anchors_for` and `_exclude_empty_associations` both need the descendant-RID
set. Compute it **once** and reuse:

- Memoize on the `DatasetBagBuilder` instance (constructed fresh per
  `aggregate_queries` / `build` in `_catalog_bag_builder`, so lifetime = one
  logical operation; no cross-session staleness). Key by `dataset.dataset_rid`.
- `anchors_for` and `_exclude_empty_associations` both read the memoized set.

Effect: the walk runs **once**, not twice.

### Out of scope (separate follow-ups, not this change)

- **Asset-size-fast estimate mode** (the user's "primary use case is asset
  size"): a future, separate path that skips exact metadata-row counts and uses
  server-side `Sum(Length)` over asset tables. Recorded as a follow-up; the walk
  fix lands first because it helps *every* nested-dataset operation, including
  real downloads.
- Replacing the estimate's per-path RID-union `csv` queries with server-side
  `cnt(RID)` aggregates (ADR-0008 territory; the walk is the dominant cost, not
  these).
- The `list_dataset_children` full-table `Dataset_Dataset` fetch
  (`dataset.py:2391`) — Part A reduces it to once per walk; making it a
  filtered/recursive server query is a further optimization left for later.

## 4. Deliverables

- `src/deriva_ml/dataset/dataset.py`:
  - a RID-only recursive descendant accessor (Part A1) —
    `list_dataset_children_rids(recurse=True, version=...)` (or the chosen A2
    flag), sharing `find_children` with `list_dataset_children`.
- `src/deriva_ml/dataset/bag_builder.py`:
  - `_iter_descendant_rids` uses the recursive RID accessor once (Part A);
  - a memoized descendant-set helper used by both `anchors_for` and
    `_exclude_empty_associations` (Part C);
  - `_exclude_empty_associations` uses one aggregate membership query per
    Dataset association table over the descendant-RID set (Part B).
- Tests (below).
- Patch/minor version bump after merge.

## 5. Testing / verification

### 5.1 Correctness — identical results

- The descendant-RID set from the new RID-only accessor equals the set of
  `.dataset_rid` from the existing `list_dataset_children(recurse=True)` on a
  nested demo dataset.
- `_exclude_empty_associations` returns the **same** excluded-set before and
  after Part B (capture on a nested demo dataset, compare). This is the
  load-bearing correctness pin.
- `estimate_bag_size` returns an **identical** dict (total_rows,
  total_asset_bytes, per-table) before and after, on a nested demo dataset.
- `anchors_for` returns the same anchor list (same RIDs, order may be asserted
  loosely as a set).

### 5.2 GET-count regression guard

On a nested demo dataset (≥2 descendants; skip with a clear message if the
fixture isn't nested), count catalog GETs during `estimate_bag_size`:

- Assert `list_dataset_children` / its RID variant is invoked **once** (or a
  small constant), **not** O(descendants).
- Assert the total catalog GET count does **not scale linearly with descendant
  count** — compare a deeper-nested vs shallow dataset if the fixture supports
  it, else assert an absolute small ceiling and document it.
- Specifically assert the per-descendant member-scan pattern is gone: the
  number of `Dataset_X` association `.fetch()` calls is ~O(associations), not
  O(descendants × associations).

### 5.3 Live re-measure (manual, recorded in PR)

Re-run `estimate_bag_size(2-277G)` against www.eye-ai.org; record wall-clock +
GET count (was 1,543 GETs / 464 s). Confirm totals unchanged (360756 rows /
18.0 GB / 80 tables / incomplete False). Also re-measure
`generate_dataset_download_spec(2-277G)` to confirm the real-download path
improved too (was 1,529 GETs / 366 s).

## 6. Risks

- **Aggregate membership semantics (Part B):** the `Dataset.in_(rid_set)` +
  presence query must match the old "any descendant has a member" set exactly.
  Guard: the §5.1 before/after equality test on `_exclude_empty_associations`.
  Edge: a very large `rid_set` could make the URL long — for the catalogs in
  scope (≤ low hundreds of descendants) this is fine; if a dataset had
  thousands of descendants the `in_` list could need chunking (note it, don't
  pre-build it — YAGNI until a real case appears).
- **RID-only accessor correctness (Part A):** must produce the same RID set and
  respect the version snapshot (it shares `find_children` + the snapshot
  catalog, so it does). Guard: §5.1 set-equality test.
- **Cycle guard:** `find_children` already tracks `_visited`; the RID-only path
  must keep it (it shares the function).
- **Public API:** prefer A1 (new method) over A2 (flag on the public
  `list_dataset_children`) to avoid changing a documented signature; the plan
  finalizes which, but defaults to A1.
