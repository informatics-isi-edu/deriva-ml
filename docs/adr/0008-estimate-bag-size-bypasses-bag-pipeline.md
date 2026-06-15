# ADR-0008: `Dataset.estimate_bag_size` deliberately bypasses `CatalogBagBuilder`

Date: 2026-05-15
Status: Accepted

## Context

ADR-0006 made `deriva.bag.catalog_builder.CatalogBagBuilder` the
canonical "catalog → BDBag" producer for DerivaML. Every code path
that materialises a bag (download, drift check, clone-via-bag,
upload staging) now routes through it.

`Dataset.estimate_bag_size` is the **one place** in
`src/deriva_ml/dataset/` that does not. The function predicts the
on-disk size of a bag *before* downloading it — used by the MCP
plugin to gate large downloads, by `bag_info` to give users a
heads-up, and by tooling that decides between materialised vs.
metadata-only fetches.

Its implementation
(`src/deriva_ml/dataset/dataset.py::Dataset.estimate_bag_size`):

1. Builds per-table aggregate datapaths via
   `DatasetBagBuilder.aggregate_queries(self)` — the same walker
   `CatalogBagBuilder` uses internally, so the **path computation**
   is shared.
2. Then bypasses `CatalogBagBuilder.build()` entirely and runs the
   queries through `AsyncErmrestCatalog` + `asyncio.gather` to fetch
   RID lists, asset-size pairs, and sample rows concurrently.

The Phase 2 dataset audit
(`docs/design/deriva-ml-audit-2026-05-phase2-dataset.md` §2.6 and
§3.D) flagged this as architectural drift:

> The async-query loop in `estimate_bag_size` (action 8) is the only
> piece of `dataset/` that bypasses `CatalogBagBuilder` for
> performance. Decision point: lift the optimisation upstream to
> `CatalogBagBuilder` (giving every consumer free parallelism), or
> formalise `estimate_bag_size` as an opt-out path. The current
> "opportunistic copy-paste" is the worst of both worlds.

The audit's framing — *"the worst of both worlds"* — is precise:
neither alternative is strictly chosen, so a casual reader sees code
that looks like copy-paste from `CatalogBagBuilder` without any
signal that the bypass is deliberate.

## Decision

**Formalise `estimate_bag_size` as an opt-out path from the bag
pipeline.** Keep the parallel async implementation; document the
bypass as a deliberate architectural choice, not an oversight.

Specifically:

1. The function continues to use `AsyncErmrestCatalog` +
   `asyncio.gather` directly. The perf headroom (~30–100× on
   datasets with many tables) is real and load-bearing for the MCP
   gate.
2. `Dataset.estimate_bag_size`'s docstring carries a `Note:` block
   pointing at this ADR so future maintainers understand the bypass
   is deliberate.
3. Path resolution is **not** duplicated: `aggregate_queries`
   continues to drive a `CatalogBagBuilder` for the walk, so any
   change to the walker semantics propagates to the estimator
   automatically.
4. The URI-parsing trick in `_extract_path` (audit §2.5) is
   recorded as a known technical-debt point but **not fixed in this
   ADR's scope**. It's a follow-up: either deriva-py exposes a
   "give me the catalog-relative path" API on its datapath
   objects, or `estimate_bag_size` keeps the trick. See "Open
   questions" below.

## Consequences

### Accepted

- The estimator and the bag builder share the **walk** but not the
  **execution**. Walker changes propagate; transport-layer changes
  don't.
- One module in `dataset/` legitimately uses `AsyncErmrestCatalog`
  directly. New async-catalog uses outside `estimate_bag_size`
  should still go through `CatalogBagBuilder` unless a future ADR
  amends this.
- A future "lift parallelism upstream" project (see "Open
  questions") would deprecate this ADR. The opt-out is a stable
  position, not a permanent one.

### Rejected alternatives

#### Alternative A: Lift parallelism into `CatalogBagBuilder`

Add an `async` execution mode to `CatalogBagBuilder.build()`. Every
consumer (download, drift, clone, estimate) would get parallelism
for free. Both estimator and builder would share both the walk and
the execution.

Rejected for now because:

- Cross-repo coordination. `CatalogBagBuilder` lives in
  `deriva-py`; the design discussion needs that repo's maintainers
  to weigh in on async-catalog churn, connection-pool tuning, and
  shared-catalog rate-limit risk.
- Behavioural breadth. `CatalogBagBuilder.build()` does more than
  parallelisable reads — it writes to disk, applies the BagIt
  profile, manages temp directories. Parallelising the read phase
  without breaking the write phase is non-trivial.
- The estimator uses a different query shape than the builder. The
  builder's queries hydrate full table contents; the estimator's
  queries fetch only RID lists, RID+Length pairs, and 100-row
  samples. Sharing the transport layer is harder than sharing the
  walker.

If/when the upstream project lands, this ADR is superseded and the
estimator collapses back into the builder. The opt-out is a
stable-but-not-permanent position.

#### Alternative B: Leave the bypass undocumented

Was the status quo before this ADR. The audit's critique is exactly
that no signal distinguishes a deliberate opt-out from accidental
divergence. Future contributors will keep "fixing" the duplication
without realising the perf consequences.

### Practical guidance for contributors

- **Adding a new estimator-like operation** (e.g. "estimate dataset
  cost before commit"): go through `CatalogBagBuilder.build()`
  first. Only fall back to direct async-catalog usage if the same
  measurement shows the bag-pipeline path is significantly slower
  *and* the optimisation is load-bearing.
- **Changing the walker** (`DatasetBagBuilder.aggregate_queries`,
  `CatalogBagBuilder.iter_table_datapaths`, table-anchor policy):
  the estimator picks the change up automatically. No special
  handling needed.
- **Refactoring `Dataset.estimate_bag_size`**: feel free to break
  out helpers, but preserve the async transport — this ADR is the
  signal that the perf path is deliberate.

## Open questions (tracked as follow-up)

1. **URI parsing in `_extract_path`** (audit §2.5). The function
   parses datapath URIs back into ERMrest paths because the
   datapath API doesn't expose a "give me the catalog-relative
   path" accessor. Should we:
   - File an upstream issue against `deriva-py` asking for a
     `datapath.relative_path` property?
   - Keep the string-parse trick (small, contained, fast)?
   - Build a thin helper in deriva-ml and live with it?

   No decision yet. The audit raised it; this ADR records it as a
   known sharp edge but does not resolve it.

2. **Long-term: lift parallelism upstream** (Alternative A). A
   future ADR may revisit if/when the deriva-py async-catalog
   surface stabilises and the cross-repo discussion is ready.

## Update — 2026-06-14: bypass mechanism changed (decision unchanged)

The **decision** this ADR records — `estimate_bag_size` deliberately
bypasses the `CatalogBagBuilder` *export engine* — is **unchanged**.
What changed is the bypass *mechanism*, an implementation evolution:

- **Before:** the estimator issued per-FK-path live aggregate queries
  executed async against the snapshot
  (`build_estimate_queries` + `run_estimate_queries`, fanned out via
  `AsyncErmrestCatalog` + `asyncio.gather`). Each reached table cost one
  or more deep server-side FK-join queries.
- **After:** the estimator runs a **client-side FK-reachability** engine
  (`src/deriva_ml/dataset/_reachability.py::compute_reachability`). It
  still shares the export *walk* (`iter_reached_paths()` + descendant
  anchors), but instead of asking the server to join, it fetches each
  reached table's edge columns **once** (a projected whole-table scan)
  and reconstructs FK reachability **in memory** (BFS over the symbolic
  FK paths), computing exact per-table RID-union counts. The deep
  server joins — and the async transport — are gone; the
  `build_estimate_queries` / `run_estimate_queries` functions were
  deleted.

The "share the walk, not the execution" consequence above still holds:
walker changes propagate; the estimator's transport is now in-memory
reachability rather than `AsyncErmrestCatalog`. Alternative A ("lift
parallelism upstream") is moot — there is no longer a parallel async
query phase to lift. The Open-question about `_extract_path` URI parsing
is likewise resolved by removal: the reachability engine works on
fetched rows, not parsed datapath URIs.

This change is part of the connected portable-bag CSV-contract design —
see `docs/superpowers/specs/2026-06-14-portable-bag-csv-contract.md`.

## References

- ADR-0006 — bag-oriented data movement (the producer/consumer
  model this ADR opts out of).
- `docs/design/deriva-ml-audit-2026-05-phase2-dataset.md` §2.6,
  §3.D — the audit findings that motivated this ADR.
- `src/deriva_ml/dataset/dataset.py::Dataset.estimate_bag_size` —
  the function whose design is captured here.
- `src/deriva_ml/dataset/bag_builder.py::DatasetBagBuilder.aggregate_queries` —
  the shared walker that both the estimator and the bag builder
  drive off.
