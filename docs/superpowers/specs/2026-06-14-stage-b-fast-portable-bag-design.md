# Stage B — Fast Portable Bag Generation (Design)

> **Status:** approved design, 2026-06-14. Stage B of the fast-estimate +
> portable-bag effort. Stage A (the client-side FK-reachability engine +
> fast estimate) shipped in deriva-ml PR #300. Stage B makes bag
> *generation* fast and the bag portable, by reusing Stage A's
> `compute_reachability`. Builds on the contract spec
> `docs/superpowers/specs/2026-06-14-portable-bag-csv-contract.md` and the
> proven prototype recorded in memory `csv-ridset-chunk-append-proto.md`.
>
> **Plan B2 SHIPPED, 2026-06-15.** The deriva-ml wiring is complete: the
> client-side bag-build path now produces Format-B bags. `DatasetBagBuilder`
> factors the shared reachability assembly into `_compute_rid_sets` (reused
> by `estimate_bag_size`), and `build_bag` feeds the resulting per-table
> RID sets to `CatalogBagBuilder(rid_sets=...)`, which emits one rid-set
> `csv` processor per reached non-vocab table — one clean
> `data/{schema}/{table}.csv` each. Vocabulary tables keep their full
> unfiltered export (FULL, not a rid-set), so a vocab CSV's row count can
> exceed the estimate's reachable-subset count by design. The live test
> `tests/dataset/test_format_b_bag.py::test_format_b_bag_one_csv_per_table_matches_estimate`
> pins one-CSV-per-table and the non-vocab count-match (vocab exempted),
> and passes. The reference doc `docs/reference/bag-export.md` (rule B4)
> documents the Format-B client-side build path. Plan B1 (the deriva-py
> `get_as_file` rid-set chunk-append + `CatalogBagBuilder(rid_sets=)`
> emission + loader simplification) shipped upstream first, per D4.
> **Live 2-277G bag-gen wall-clock: <to be filled by the controller's
> re-measure>** (target: tens of seconds vs ~280s deep-join).

## Problem

Bag generation pays the same deep server-side FK-join cost the estimate
used to: ~280s for eye-ai 2-277G, because `CatalogBagBuilder` emits one
deep-join `csv` query_processor **per FK path** to each reached table. The
bag is also **not portable** to plain `bdbag`-CLI consumers: a table's
rows are split across multiple path-encoded `data/**/{table}.csv` files,
and the complete set requires an out-of-band union-by-basename +
dedup-by-RID that only deriva-ml's SQLite loader implements.

Stage A already computes, client-side and fast, the exact set of
reachable RIDs per table (`compute_reachability` in
`src/deriva_ml/dataset/_reachability.py`). Stage B feeds those RID sets to
the bag exporter so generation becomes a set of cheap direct
`/entity/{table}/RID=any(...)` fetches instead of deep joins, and emits
**one clean CSV per table**.

## Goals

- Bag generation drops from ~280s toward tens of seconds (join
  elimination, same as the estimate win).
- One clean `data/{schema}/{table}.csv` per reached table — complete,
  RID-distinct, flat. Trivially consumable by plain `bdbag` CLI (no
  union/dedup contract to know).
- The SQLite loader simplifies (one file → one table; no `ON CONFLICT`
  union needed).
- deriva-py stays domain-agnostic — it consumes RID sets, it does not
  learn FK-reachability. The scope/mechanics split (ADR-0006/0008) holds.

## Non-goals

- Concurrency tuning of the RID-set fetch is a follow-up, not Stage B's
  bar. Stage B's correctness + portability win stands at sequential speed
  (the prototype's full-Image-table fetch was 55.6s sequential — already
  faster than the 280s deep-join path; concurrency is the lever that
  closes the rest, and is a known, proven technique from the estimate's
  async path).
- Chaise's annotation-driven export is untouched. The new behavior is
  additive (a new optional `rid_set` param / processor mode); existing
  `query_path` `csv` processors behave exactly as before.

## The four locked decisions

### D1 — Engine layer: extend `get_as_file` with `rid_set`

Add an optional `rid_set` (list of RIDs) + `rid_table` to
`ErmrestCatalog.get_as_file`
(`deriva/core/ermrest_catalog.py`). When `rid_set` is present:

- Chunk the RIDs into URL-safe batches (~500, matching
  `rid_lease.py`'s `PENDING_ROWS_LEASE_CHUNK`).
- For each chunk, fetch `/entity/{rid_table}/RID=any(q1,q2,...)` where
  **each RID is individually URL-quoted** and joined with literal commas
  (the load-bearing gotcha: `,` is `any()` syntax, not a value —
  `quote(",".join(rids))` silently returns zero rows; the prototype
  caught this).
- **Append every chunk to one output CSV** by reusing the existing
  per-page `first_page`-skips-header append logic already in the paged
  loop: open `'w+b'` once (truncate), the first chunk writes
  header+body, every later chunk skips its header line and appends body
  only. The chunk loop and the page loop are the **same** append
  mechanism — a chunk is just an outer iteration around the existing
  paged fetch.

This is the only genuinely new engine machinery, and it reuses the
proven page-append. Solves the URL-length 414 (proven: a 225k-RID single
URL = 1.9 MB → 414 Request-URI Too Long).

### D2 — Bag format: one clean CSV per table (Format B)

`CatalogBagBuilder` emits **one** `csv` query_processor per reached
table (carrying that table's RID set), producing one flat
`data/{schema}/{table}.csv`. Replaces the per-FK-path emission
(`for fk_path in self._fk_paths_for(key)`). Vocabulary full-export keeps
its single unfiltered `/entity/{schema}:{table}` query (unchanged). The
asset `fetch` processor is **unchanged** — it reads
`URL/Length/Filename/MD5` from the asset table, and the RID-set query
returns those columns just as the join query did.

The SQLite loader (`deriva/bag/database.py` / `loader.py`) simplifies:
one CSV → one table, no `ON CONFLICT` union needed (the file is already
RID-distinct). Clean break — the per-path emission is deleted, not kept
behind a flag (no known consumer needs the path-encoded directory
provenance; it was always "provenance, not semantics").

### D3 — Integration seam: deriva-ml computes, passes RID sets in

deriva-ml's `DatasetBagBuilder` calls `compute_reachability` to get
`{table: rid_set}`, then builds the export spec with one rid-set `csv`
processor per table (each `processor_params` carries `rid_table` + the
RID list). `CatalogBagBuilder` gains a path that emits rid-set processors
from a supplied `{table: rids}` map. deriva-py never learns
FK-reachability — it consumes RID sets and fetches them. deriva-ml owns
the "what's reachable" decision, consistent with the ADR-0006/0008
scope/mechanics split.

### D4 — PR sequencing: deriva-py first, then deriva-ml

Two plans, two PRs, in dependency order:

1. **Plan B1 (deriva-py):** `get_as_file` rid_set chunk-append +
   `CSVQueryProcessor` passthrough of `rid_set`/`rid_table` from
   `processor_params` + `CatalogBagBuilder` rid-set spec emission +
   loader simplification + **upstream contract tests** (the contract
   test belongs in deriva-py's CI, per the workspace cross-repo rule).
   Ships as its own deriva-py PR on the `deriva-ml` branch.
2. **Plan B2 (deriva-ml):** bump the deriva-py pin, wire
   `DatasetBagBuilder` → `compute_reachability` → Format-B export spec,
   rewire the bag download path, live 2-277G bag-gen re-measure. Ships as
   its own deriva-ml PR after B1 merges.

## Architecture

```
deriva-ml                                    deriva-py
─────────                                    ─────────
DatasetBagBuilder
  compute_reachability(...)  ──► {table: rid_set}
  build Format-B export spec ──────────────► CatalogBagBuilder
    (one rid-set csv processor                 emit rid-set csv processors
     per table)                                (one per table)
                                                   │
                                             export engine runs spec
                                                   │
                                             CSVQueryProcessor
                                               passes rid_set/rid_table ──► ErmrestCatalog.get_as_file
                                                                              chunk RIDs @500, quote
                                                                              each, append to ONE CSV
                                                   │
                                             data/{schema}/{table}.csv  (one clean file per table)
                                             + fetch.txt (assets, unchanged)
                                                   │
                                             BagDatabase loader: one CSV → one table (no union)
```

## CSV contract (Format B — the consumer-facing guarantee)

Per the contract spec, the bag now satisfies:

- **One CSV per table:** `data/{schema}/{table}.csv`. `Image.csv` is THE
  complete reachable Image set.
- **No union/dedup/glob:** the file IS the union, already RID-distinct.
  `read_csv("data/eye-ai/Image.csv")` is correct and complete.
- **`RID` is the PK** (in `data/schema.json`); asset bytes out-of-line
  via `fetch.txt` (`sum(Length)` for size).

## Testing

**deriva-py (Plan B1) — contract tests upstream:**
- `get_as_file` with `rid_set`: a RID set larger than the URL limit
  produces one CSV, header once, RID-distinct, complete (against a
  fixture catalog or the existing bag test harness). This is the pin for
  the engine contract.
- The per-RID-quoting correctness (commas are syntax): a test that a
  multi-RID `rid_set` returns all rows (would fail under
  `quote(",".join(...))`).
- `CatalogBagBuilder` rid-set emission: the export spec has one csv
  processor per table with the RID set, not one-per-FK-path.
- Loader: one-CSV-per-table loads correctly.

**deriva-ml (Plan B2) — integration + the headline:**
- `DatasetBagBuilder` builds a Format-B spec from `compute_reachability`
  output (the RID sets match the reachable sets).
- A live bag-generation run produces one CSV per table, RID-distinct,
  with correct asset `fetch.txt`.
- **Live 2-277G bag-gen re-measure** — the headline number (target: tens
  of seconds vs ~280s) for the PR.
- A bag round-trips through the SQLite loader with correct per-table row
  counts (matching the estimate).

## Risks

- **deriva-py is a shared library.** The `get_as_file` change is additive
  (optional param), but `get_as_file` is widely used — the contract test
  upstream is the guard. Chaise's existing exports use `query_path` and
  are unaffected.
- **Concurrency gap.** Sequential RID-set fetch is already faster than
  the deep-join path but not as fast as the estimate's 27s prototype
  (which fetched RID+FK cols only, concurrently). Stage B ships
  sequential-correct; concurrency is a tracked follow-up, not a blocker.
  `log()`/document the sequential nature so the speed expectation is
  honest.

## See also
- `docs/superpowers/specs/2026-06-14-portable-bag-csv-contract.md` — the
  CSV rules + export-spec shapes (Format A vs B) this design realizes.
- Stage A: `docs/superpowers/plans/2026-06-14-reachability-engine-fast-estimate.md`
  + deriva-ml PR #300 (the shipped `compute_reachability` engine B2 reuses).
- Memory `csv-ridset-chunk-append-proto.md` — the proven prototype + the
  per-RID-quoting gotcha.
- ADR-0006 (bag-oriented data movement), ADR-0008 (estimate bypass) — the
  scope/mechanics split this design preserves.
