# Dataset-Export Terminal Tables — Design

**Date:** 2026-06-13
**Status:** Approved in brainstorming; spec for implementation.
**Subproject:** `deriva-ml`

## 1. Problem statement

Downloading a dataset bag for a large nested dataset
(e.g. eye-ai `2-277G`, a 50-dataset nested training pool) **never
completes the bag-formation phase**. The export's deepest
query-processor — the path
`Dataset → Dataset_Image → Image → Annotation → Execution → Execution_Asset_Execution → Execution_Asset`
— ran for **18 minutes** before dying on a connection error for a
single large child dataset (`5-WEBG`, 9,511 images); the aggregate
of all 50 anchors is hopeless.

### Root cause (empirically confirmed, 2026-06-13)

`DatasetBagBuilder.build_policy()`
(`src/deriva_ml/dataset/bag_builder.py`) constructs its
`FKTraversalPolicy` with **no `terminal_tables`**. The
`CatalogBagBuilder` walker therefore follows both outbound and
**inbound** FKs through every reached table — including `Execution`.

`Execution` is a **provenance hub**: it has 21 inbound foreign keys
(`referenced_by`), including back-edges to `Annotation` and
`Dataset_Execution`, a self-loop via `Execution_Execution`, and an
edge to every `*_Execution` association in the catalog
(`Execution_Asset_Execution`, `Model_Artifact_Execution`,
`File_Execution`, …). Entering `Execution` inbound makes the walk
fan out across the catalog's **entire provenance DAG**, reachable
from any dataset that shares an execution with anything else.

The explosion was located hop-by-hop on a 20-image dataset that
*does* complete (`4-N9C8`):

```
Dataset_Image:            20 rows
+Image:                   20 rows
+Annotation:             100 rows   (5 annotations/image)
+Execution:                5 rows → 1 DISTINCT execution
+Exec_Asset_Execution: 9,066 rows   ← ×1,800 blow-up
+Execution_Asset:      9,066 rows
```

20 images reach **one** execution, and that one execution has
**9,066 output assets** — so the bag pulls *every output asset of
every execution that produced any feature/annotation in the
dataset* (model weights, the full 257 GB artifact arm), plus, via
`Execution`'s other inbound edges, the executions and datasets
*those* touched, transitively. The walk is bounded only by the size
of the connected provenance component.

### Why `clone_via_bag` doesn't have this bug

`src/deriva_ml/catalog/clone_via_bag.py` already sets
`terminal_tables = {("deriva-ml","Execution"), ("deriva-ml","Workflow")}`
with a comment explaining exactly this: provenance tables describe
*how* rows came to be, aggregate across many anchor scopes, and must
be **entered but not traversed outward**. The dataset-export path
(`bag_builder.build_policy`) simply omits the same protection.

## 2. The fix

Make `DatasetBagBuilder.build_policy()` set the same
`terminal_tables` as `clone_via_bag`:

```python
terminal_tables = {
    ("deriva-ml", "Execution"),
    ("deriva-ml", "Workflow"),
}
```

`terminal_tables` semantics (from `deriva.bag.traversal`): the walker
**enters** a terminal table — so the bag still carries the
provenance rows referenced by the dataset's `*_Execution` /
`*_Workflow` associations (the link "this annotation was made by
execution X") — but does **not** follow that table's outbound or
inbound FKs. This severs the fan-out and the loop at the provenance
boundary.

### Empirical validation (before writing this spec)

With the change applied in-process against `5-WEBG` (the
18-minute-killer dataset):

- `Execution_Asset` reached? **False** (explosion arm severed)
- `Execution_Asset_Execution` reached? **False**
- `Execution` reached? **True** (still entered — provenance link kept)
- `aggregate_queries` built in **1.9 s** (was: 18-minute timeout)
- `Annotation`, `Dataset_Image`, feature tables still reached — the
  dataset's actual content is unaffected.

## 3. Scope & shared-constant extraction

`{Execution, Workflow}` is now needed in **two** places
(`clone_via_bag` and `bag_builder`). To avoid a silent divergence (a
future edit to one and not the other re-opens this bug in the path
that wasn't updated), define the set **once** and import it in both.

- Add `DATASET_PROVENANCE_TERMINAL_TABLES` (or similar) to
  `src/deriva_ml/core/constants.py`, next to `INTENTIONAL_FK_CYCLES`
  (already the home for FK-traversal constants shared by these paths).
- `clone_via_bag.default_terminal_tables` and
  `bag_builder.build_policy` both import it.

This is a small, in-scope improvement justified by the bug itself
(the divergence is what allowed the two paths to differ).

## 4. User-supplied terminal tables

`build_policy` merges user `exclude_tables` over derived exclusions
today. The terminal set is a fixed default (not currently user-
tunable), matching `clone_via_bag`. **Per-export configurability of
`terminal_tables` is a non-goal** (§7) — YAGNI until a real use case
appears; the default is correct for every dataset bag.

## 5. Architecture / data flow

No structural change. `build_policy` already returns the
`FKTraversalPolicy` consumed by `_catalog_bag_builder` →
`CatalogBagBuilder` → the export engine. The change adds one field
to the returned policy. The deriva-py export engine, anchors, and
`anchors_for` nested-traversal are all unchanged.

The earlier-considered **deriva-py query-staging rewrite** (pre-resolve
intermediate RID sets, batch shallow queries) is **rejected**: it
would have made deriva-py efficiently fetch a catalog-sized *wrong*
result. The problem was never query mechanics; it was traversal
scope. (Recorded here so the rejected path isn't re-proposed.)

## 6. Testing

- **Unit (no catalog):** a test asserting
  `build_policy(dataset).terminal_tables` contains
  `("deriva-ml","Execution")` and `("deriva-ml","Workflow")`. Guards
  the regression directly.
- **Shared-constant test:** assert `clone_via_bag` and `bag_builder`
  resolve the *same* object/set (import-identity or value-equality),
  pinning the no-divergence intent.
- **Walk-scope test (mocked model or live, as the existing bag_builder
  tests do):** assert that for a dataset whose members reach
  `Execution`, the reached-table set from `aggregate_queries`
  **excludes** `Execution_Asset` / `Execution_Asset_Execution` while
  still **including** `Execution` and the member/feature tables.
- **Integration (live catalog, marked):** the existing dataset-bag
  download tests already exercise the policy; confirm they pass
  unchanged (the demo catalog's executions are tiny, so behavior is
  observable but not slow). The eye-ai `5-WEBG` 18-min reproduction
  is the manual before/after, not a CI test (needs the eye-ai
  catalog + credentials).

## 7. Non-goals

- deriva-py changes (the export engine, query staging, batching).
- Per-export-configurable `terminal_tables` / depth caps.
- Changing what `_exclude_empty_associations` prunes (it is correct —
  it kept `Dataset_Image` because the dataset genuinely has image
  members; verified).
- The separate `DatasetSpec.timeout`-not-plumbed gap (issue #287) —
  independent; this fix removes the need for a timeout bump on this
  path but doesn't close that issue.

## 8. Follow-on (recorded, not in this PR)

The 18-minute non-timeout-then-ConnectionError behavior also shows
the bag export had **no effective ceiling** on a runaway walk — a
provenance-hub mistake produced an 18-minute hang rather than a fast
failure. Whether the export should bound its own walk size (refuse a
traversal that fans past N tables/rows) is a deriva-py resilience
question worth raising upstream, separate from this scope fix.
