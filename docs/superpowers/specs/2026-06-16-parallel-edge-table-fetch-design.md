# Parallelize edge-table fetches in compute_reachability

**Date:** 2026-06-16
**Status:** Design — approved approach (opt-in bounded concurrency)

## Problem

`compute_reachability` ([`_reachability.py`](../../../src/deriva_ml/dataset/_reachability.py))
fetches each reached edge table **sequentially** (the loop at ~line 252):

```python
for seg in edge_tables:
    s, t = seg
    fetched_rows[seg] = fetch(s, t, _needed_columns(seg, model))
```

Each iteration is an independent HTTP round-trip (a projected whole-table
scan). On a large dataset (eye-ai 2-277G: ~80 edge tables, ~1.4M rows) this
sequential phase is a meaningful share of the ~68.5s estimate wall-clock —
the `estimate-bag-size-client-side-join` memory notes the production engine
"fetches the FULL projected column set ... and runs **sequentially**" as the
reason live 68.5s lags the 27s prototype. The fetches are pure, independent
reads writing to distinct `fetched_rows[seg]` keys — a textbook parallelism
candidate.

**This is not estimate-only.** `_compute_rid_sets` (which wraps
`compute_reachability`) is shared by the estimate AND the bag-generation
paths — `generate_dataset_download_spec` and `build_bag` both call it to
compute the Format-B `rid_sets`. So the same sequential fetch loop runs when
generating a download spec or building a bag, and parallelizing it speeds up
**bag generation**, not just the estimate (see "This affects bag generation
too" below).

## Why this is the right one of the three screenshot suggestions

- **Fetch-processor RID filter** — already fixed (deriva-py #275 / deriva-ml #308).
- **Server-side aggregate for asset sizes** — rejected. The whole redesign
  exists *because* server aggregates pay the same deep-join cost the
  client-side engine escapes (`CntD`/aggregates "do NOT help" — the design
  memory). The raw projected rows are fetched once and reused for BOTH
  row-count and Length; a server aggregate would *add* a round-trip, not
  remove the fetch (still needed for reachability + CSV-byte sampling).
  Strictly slower.
- **Parallelize edge-table fetches** — legitimate, and already flagged as the
  known sequential bottleneck in our own measurements.

## Constraint: shared-session thread-safety

The production `fetch` closure issues `tpb.attributes(...).fetch()` through
deriva-py's `ErmrestCatalog`, which holds **one** `requests.Session`
(`self._session`, created once per binding). `requests.Session` is **not
documented thread-safe** for concurrent use across threads. So we cannot
blindly fan out unbounded GETs on the shared session.

In practice, a *bounded* number of concurrent GETs on one session works (the
risk is connection-pool / cookie-jar races, low for read-only GETs), but it
is technically unsupported. The design must therefore:

1. Keep the `fetch` callable's contract unchanged (injected, single-table).
2. Make concurrency **opt-in and bounded**, defaulting to the current
   sequential behavior so nothing regresses by default.
3. Preserve **exact** output — parallelism must not change `fetched_rows`,
   `rids_by_table`, or `asset_lengths_by_table` (the live differential
   exactness test is the gate).

## Design

Add a `max_workers: int = 1` parameter to `compute_reachability`. When `> 1`,
run the edge-table fetch loop through a `concurrent.futures.ThreadPoolExecutor`
capped at `min(max_workers, len(edge_tables))`; when `== 1` (default), keep the
plain sequential loop verbatim (no thread overhead, identical behavior).

```python
def compute_reachability(*, reached, anchor_rids, model, fetch, max_workers=1):
    ...
    edge_tables = ...  # unchanged
    fetched_rows = {}
    if max_workers > 1 and len(edge_tables) > 1:
        from concurrent.futures import ThreadPoolExecutor
        workers = min(max_workers, len(edge_tables))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(fetch, s, t, _needed_columns((s, t), model)): (s, t)
                for (s, t) in edge_tables
            }
            for fut in futures:
                seg = futures[fut]
                fetched_rows[seg] = fut.result()  # re-raises fetch errors
    else:
        for seg in edge_tables:
            s, t = seg
            fetched_rows[seg] = fetch(s, t, _needed_columns(seg, model))
    # ... phases 3 & 4 unchanged (operate on the complete fetched_rows dict)
```

Only the fetch phase is touched. The union (phase 3) and asset-length (phase 4)
phases are unchanged — they read the fully-populated `fetched_rows` dict, so
they are insensitive to fetch ordering. `fut.result()` re-raises any fetch
exception, preserving the current "a fetch failure aborts the estimate"
semantics (no silent partial results).

### This affects bag generation too, not just the estimate

`_compute_rid_sets` ([`bag_builder.py`](../../../src/deriva_ml/dataset/bag_builder.py))
is the shared chokepoint, called by **three** paths — so the fetch-loop change
speeds up all of them:

1. `estimate_bag_size` / `bag_info` (dataset.py:2735) — the estimate.
2. `generate_dataset_download_spec` (bag_builder.py:296) — the MINID /
   download spec (#305). Computes `rid_sets` to emit the Format-B processors.
3. `build_bag` (bag_builder.py:397) — the client-side Format-B bag build.

So this is a **bag-generation** improvement, not only an estimate one — which
is the more valuable target, since bag generation on a large dataset is the
slow operation that motivated the work. The knob therefore belongs on
`_compute_rid_sets` (where all three converge), threaded down to
`compute_reachability(max_workers=...)`.

### Naming: do NOT reuse `fetch_concurrency`

`download_dataset_bag` / `cache` already have a `fetch_concurrency` (default
**8**) — but it controls **concurrent asset *file* downloads during
materialization**, a different operation from the reachability **edge-table**
fetches. Reusing that name would conflate two unrelated concurrency knobs.

Use a distinct name — **`reachability_concurrency: int = 1`** — for the
edge-table fetch parallelism:

- `compute_reachability(..., max_workers=...)` — internal.
- `DatasetBagBuilder._compute_rid_sets(dataset, reachability_concurrency=1)`
  → passes `max_workers=reachability_concurrency`.
- Surface `reachability_concurrency: int = 1` on the three public entry
  points: `estimate_bag_size` / `bag_info` (and the DerivaML mixin wrappers),
  and on `download_dataset_bag` / `cache` / `build_bag` **alongside** the
  existing `fetch_concurrency` (the two coexist: one parallelizes the
  reachability fetch that builds the spec, the other parallelizes the asset
  download that materializes the bag).

**Default stays sequential (`1`).** Deliberate: the parallel path is opt-in
because (a) shared-session concurrency is technically unsupported, and (b) the
win only materializes at scale (80+ tables); the demo catalog (32 tables, 280
rows, 0.34s) sees no benefit. Production callers on large catalogs opt in.

To keep the surface from sprawling, the public wiring can be staged: land the
engine + `_compute_rid_sets` knob first (internal, fully tested), then thread
the public params. The MVP that delivers the value is the engine knob reaching
all three internal callers; the public-arg ergonomics are a thin follow-on.

## Testing

1. **Equivalence (unit, injected fetch):** run `compute_reachability` with
   `max_workers=1` and `max_workers=8` against the same in-memory `fetch`
   stub; assert byte-identical `rids_by_table` / `asset_lengths_by_table` /
   `fetched_rows`. The injected fetch makes this catalog-free.
2. **Concurrency actually used:** a fetch stub that records the set of thread
   idents it was called on; assert >1 distinct thread when `max_workers>1`
   and exactly 1 when `max_workers=1`.
3. **Error propagation:** a fetch stub that raises on one table; assert
   `compute_reachability(max_workers=8)` re-raises (no silent partial).
4. **Exactness gate (live):** the existing differential test
   (`test_reachability_matches_server_union` / the estimate exactness test)
   re-run with `fetch_concurrency>1` to confirm the parallel path still
   matches the server-union oracle exactly.
5. **Live 2-277G re-measure** (if an eye-ai token is available): sequential vs
   `fetch_concurrency=8` wall-clock, exactness preserved. If no token,
   document that the at-scale speedup is unverified and the change is
   behavior-preserving by default.

## Risks

- **Shared-session races.** Mitigated by bounded pool + read-only GETs +
  opt-in default. If races surface at high concurrency, the cap is the knob;
  a follow-up could move to a session-per-thread, but that is out of scope
  here (and heavier).
- **No at-scale measurement locally.** The design is *correct and
  behavior-preserving by default* regardless; the speedup claim is validated
  only when a large catalog is available. We do not silently claim a speedup
  we haven't measured — the default is unchanged and the parallel path is
  opt-in.
