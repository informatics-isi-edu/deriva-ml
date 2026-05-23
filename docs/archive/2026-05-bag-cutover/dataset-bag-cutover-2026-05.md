# Dataset-bag cutover — `CatalogGraph` → `CatalogBagBuilder`

**Date:** 2026-05-13
**Status:** Plan accepted; implementation pending.
**Closes:** ADR-0006's final deferred follow-up
("`Dataset.download_dataset_bag` is rewired to construct a
`DatasetBagBuilder` internally and return a `DatasetMinid` from
its output. The public signature does not change.").

This document records the design decisions that drive the
cutover of `Dataset.download_dataset_bag` (and `Dataset.is_dirty()`)
from the legacy `CatalogGraph` machinery to the bag pipeline
established by ADR-0006. It is the spec the cutover PR
implements.

## Context

After ADR-0006 landed:

- Clone (`catalog/clone.py` → `clone_via_bag.py`) runs on
  `CatalogBagBuilder + BagCatalogLoader`.
- `Execution.commit_execution` runs on
  `BagBuilder(LocalDBDataSource) + BagCatalogLoader`.
- Dataset download — **still uses `CatalogGraph`**. The cutover
  was deferred ("once live-catalog tests confirm byte-for-byte
  spec equivalence", see `bag_builder.py:32–36`).

`DatasetBagBuilder` exists already as a *facade*: its public
methods (`generate_dataset_download_spec`,
`generate_dataset_download_annotations`, `aggregate_queries`)
delegate to a `CatalogGraph` instance held privately. The
bag-pipeline-shaped helpers (`anchors_for`, `build_policy`,
`_exclude_empty_associations`, `_iter_descendant_rids`) are
wired but unused.

The five `dataset.py` call sites that produce / consume dataset
bags already route through the facade, so the cutover is local:
flip what's inside `DatasetBagBuilder` without touching callers.

## Decisions

### D1 — Equivalence shape: behavioural, not byte-for-byte

Old MINIDs are not load-bearing. Pre-cutover cached MINIDs may
become unresolvable; users re-download once. Bag *contents*
(rows + assets) must match what `CatalogGraph` produces; bag
*spec internals* (output_path strings, query_processor ordering)
are explicitly allowed to differ.

This unblocks the "byte-for-byte spec equivalence" gate that
held up the original cutover.

### D2 — `is_dirty()` cuts over in lockstep

CONTEXT.md ("Dirty"): *"The drift walk **is** the bag walk plus an
RMT filter."* If `download_dataset_bag` flips to
`CatalogBagBuilder`'s walk and `is_dirty()` doesn't, the
invariant breaks: drift detection reports against the old paths
while bags assemble along the new paths. A "clean" dataset could
have uncommitted drift.

The two flips therefore ship together in one PR. The two
mechanically share a `CatalogBagBuilder` instance — they don't
have to share commits, but they cannot ship in different PRs.

### D3 — Test gate: bag-content equivalence + facebase smoke

Two tiers:

- **Structured (α):** New `TestBagEquivalence` harness. Produce
  a bag two ways (legacy `CatalogGraph` + new `CatalogBagBuilder`),
  open both via `BagDatabase`, assert *row sets per table* and
  *asset RID/hash sets per asset table* are equal. Specs are
  allowed to differ. Runs against the `catalog_with_datasets`
  fixture.
- **Smoke (γ):** Manual end-to-end run against `facebase.org`.
  Download one real dataset both ways, confirm member counts +
  vocab terms + asset hashes match. Not a unit test.

The harness is **load-bearing for the cutover** — it gates the
merge — and **disposable** after merge.

### D4 — `CatalogGraph` deleted; equivalence harness deleted with it

After the harness goes green and the facade flips, `CatalogGraph`
has no callers. Carrying it as a "test oracle" past merge means
~800 LoC of dead production code maintained for nothing.

The cutover PR:
1. Adds the harness.
2. Iterates `_generate_spec_via_bag()` until the harness passes
   on `catalog_with_datasets`.
3. Flips the facade body.
4. **Deletes** both `src/deriva_ml/dataset/catalog_graph.py` and
   `TestSpecEquivalence` + `TestBagEquivalence` (the harness) in
   the same commit.

After merge, `download_dataset_bag` is covered by the existing
dataset integration tests — download → materialize → query
members → assert correctness against the live catalog. Those
tests assert *correct* behaviour, not *equivalent-to-legacy*
behaviour, so no oracle is needed.

### D5 — `iter_table_datapaths()` on `CatalogBagBuilder` (deriva-py)

`aggregate_queries()` (the `is_dirty()` driver) returns datapath
objects, one per FK path per reached table. Today it lives in
`CatalogGraph._aggregate_queries`. The cutover needs an
equivalent on `CatalogBagBuilder`.

Three options considered (`CONTEXT.md` style):

| Option | Where | Verdict |
|---|---|---|
| (A) `CatalogBagBuilder.iter_table_datapaths()` (deriva-py) | Public on the bag builder | **Adopted.** |
| (B) deriva-ml reaches into `_reached_tables` / `_table_paths` | Brittle private-attr access | Rejected. |
| (C) Independent walk in deriva-ml | Violates D2 invariant | Rejected. |

The new method is **instance-scoped**: anchors are fixed at
construction time, matching the existing `build()` /
`get_export_spec()` convention. To cover the "catalog-wide"
case (today: `_aggregate_queries(dataset=None)`), deriva-ml
constructs a fresh `CatalogBagBuilder` per call — once with
`anchors_for(dataset)` for the per-dataset path, once with
`[TableAnchor("Dataset")]` for the catalog-wide path. Per-call
construction is fine: `is_dirty()` runs maybe once per release.

Return shape:

```python
def iter_table_datapaths(
    self,
) -> dict[tuple[str, str], list[tuple[Any, Any, bool]]]:
    """For each reached table, return (datapath, pb_table, is_asset) per FK path.

    Caller evaluates each datapath against the live catalog (with
    its own filters layered on, e.g. ``RMT > T_release``) and computes
    set-unions across paths for cross-route deduplication. Matches
    the structure deriva-ml's ``aggregate_queries`` returns.
    """
```

### D6 — Deriva-py change shipped in the same package

The deriva-py-side change (`iter_table_datapaths` + unit tests)
amends into the existing squash commit `59399c6` on the
`deriva-ml` branch of deriva-py. No new deriva-py PR; the bag
module ships as one commit with this method included.

The deriva-ml cutover PR pins to the resulting deriva-py SHA.

### D7 — Vocab handling is structurally equivalent (no edge case)

Domain confirmation: a vocabulary is never referenced by
another vocabulary. The theoretical "chained vocab outbound FK"
case the walker's "FKs into vocab, never out" rule would miss
**does not exist** in any real schema. The walker's rule is
exact, not approximate.

### D8 — Annotation generators stay in deriva-ml; `_dataset_nesting_depth()` moves to `DatasetBagBuilder`

`generate_dataset_download_annotations()` produces a Chaise
export annotation — a static template baked into the catalog
that describes nested-dataset paths *symbolically*. It needs to
enumerate paths up to the maximum nesting depth that exists in
the catalog, computed by `_dataset_nesting_depth()`.

That logic is dataset-domain-specific (Chaise annotation; the
`Dataset → Dataset_Dataset → Dataset` chain by name) and doesn't
fit `CatalogBagBuilder`'s model ("given anchors and a policy,
build a bag"). Pushing it into deriva-py would force a generic
class to know about Chaise; bad fit.

Resolution: lift the four annotation-related methods from
`CatalogGraph` into `DatasetBagBuilder` as private methods.
~100 LoC. Methods affected:

- `_dataset_nesting_depth()`
- `_export_annotation()`
- `_export_annotation_dataset_element()`
- `generate_dataset_download_annotations()` (already on the facade)

Everything else in `CatalogGraph` is deleted.

### D9 — Feature tables reached naturally by the generic walker

ADR-0006 predicts (and `bag_builder.py` lines 99–108 echo): "Feature
tables are reached naturally by FK-following from member element
rows; no separate force-include mechanism is needed."

Code-level verification: a feature table is itself an *association
table* (`Feature.__init__` takes a `FindAssociationResult`,
`Feature.feature_table = atable.table`). Its FKs:
one back to the target table (e.g., `Image`),
one to the `Feature_Name` vocabulary,
optional asset / term / value columns.

`CatalogBagBuilder.find_associations()`-equivalent walking on
inbound FKs into the target table will naturally encounter the
feature association table. Its endpoints, vocab terms, and asset
references all come along under the existing walker rules.

`CatalogGraph`'s explicit feature-table inclusion
(`_collect_paths` lines 499–505) is redundant with what the
generic walker does. No special handling in the cutover.

The bet is **unverified by code today**, but the equivalence
harness (D3) checks it on the test fixture. Empty feature tables
(no member has feature values for this dataset) produce empty
CSV files on both sides — equivalent.

## Implementation plan

### Step A — deriva-py amend into squash commit `59399c6`

1. `deriva/bag/catalog_builder.py`:
   - Add `iter_table_datapaths(self) -> dict[tuple[str, str], list[tuple[Any, Any, bool]]]`.
   - Internal: `_ensure_walked()` is the existing
     `_compute_reached_tables()` (already idempotent — already
     used by `get_export_spec()` and `reached_tables`).
   - Build datapaths from `_reached_tables` + `_table_paths` +
     `self.catalog.getPathBuilder()`.
2. `tests/deriva/bag/test_catalog_builder.py`:
   - `test_iter_table_datapaths_returns_one_entry_per_reached_table`
   - `test_iter_table_datapaths_one_path_per_fk_route`
   - `test_iter_table_datapaths_marks_asset_tables_correctly`
   - `test_iter_table_datapaths_filterable_with_in_for_dirty_detection`
3. Update commit message: append bullet under `catalog_builder.py` in "## What this commit lands".
4. Amend `59399c6`. Force-push.

Sizing: ~80 LoC source + ~150 LoC tests.

### Step B — deriva-ml cutover PR

1. Pin deriva-py to the resulting commit SHA.
2. `src/deriva_ml/dataset/bag_builder.py`:
   - Add `_dataset_nesting_depth()`, `_export_annotation()`,
     `_export_annotation_dataset_element()` as private methods
     (lifted from `CatalogGraph`).
   - Rewrite `generate_dataset_download_annotations()` to call
     the lifted methods directly.
   - Add `_generate_spec_via_bag(dataset)`: builds
     `CatalogBagBuilder(catalog=..., anchors=self.anchors_for(dataset),
     policy=self.build_policy(dataset))`, calls
     `get_export_spec()`, layers on dataset-specific top-level
     keys (`env`, `bag.bag_name = "Dataset_{RID}"`, preamble
     `query_processors`, optional MINID `post_processors`).
   - Flip `generate_dataset_download_spec()` to call
     `_generate_spec_via_bag()`.
   - Rewrite `aggregate_queries(dataset=None)`: construct a
     `CatalogBagBuilder` with the appropriate anchors, call
     `iter_table_datapaths()`, return as-is.
   - Drop the `_catalog_graph` attribute.
3. `tests/dataset/test_bag_builder.py`:
   - **Add** `TestBagEquivalence` — produces two bags, opens via
     `BagDatabase`, asserts row-set equality per table and
     asset-RID equality per asset table.
   - Run until green on `catalog_with_datasets`.
4. `tests/dataset/test_datasets.py:155` — fix the lingering
   `CatalogGraph(...)._dataset_nesting_depth()` call. Migrate to
   `DatasetBagBuilder(ml_instance=ml)._dataset_nesting_depth()`.
5. Manual smoke check (γ): script that builds two bags from
   `facebase.org`, compares row counts + asset hashes. Not
   checked into the test suite; documented in the PR
   description.
6. **Delete** `src/deriva_ml/dataset/catalog_graph.py` (~800 LoC).
7. **Delete** `TestSpecEquivalence` + `TestBagEquivalence` from
   `tests/dataset/test_bag_builder.py`. Keep `TestAnchorsAndPolicy`
   (it tests the helpers that survived).
8. Update `bag_builder.py` module docstring: drop the "scope note"
   about the deferred cutover; it's done.

Net LoC: roughly **−1100** (delete `CatalogGraph` + delete
equivalence-scaffolding tests + flip facade body + add ~100 LoC
of lifted annotation methods).

### Step C — open the PR

Single PR into `main` (or the appropriate base branch) of
deriva-ml. PR body cross-references this doc and the smoke-check
results.

## Out of scope

- Migration of `Dataset.download_dataset_bag` callers in
  application code (model template, notebooks). Public signature
  is unchanged; callers don't need to change.
- Backwards compatibility shims for old MINIDs (D1).
- Performance comparison between the two pipelines. The cutover
  is correctness-driven; performance is a follow-up if needed.

## Open questions deferred to implementation

- **`paged_query: True`**: `CatalogGraph` sets it on every CSV
  processor; `CatalogBagBuilder` does too (line 629 of
  `catalog_builder.py`). No expected divergence, but the
  equivalence harness will spot it if I'm wrong.
- **Asset row hash invariants**: `BagCatalogLoader` does
  Hatrac MD5 dedup. The equivalence harness should compare
  MD5 sets, not bytes — that's what dedup actually preserves.
