# Portable Bag — CSV Contract & Export-Spec Shape (Format B design)

> **Status:** design artifact for the fast-estimate + portable-bag
> effort. Describes the **proposed** Format B (one clean CSV per table,
> produced by a RID-set query processor), contrasted against **Format A**
> (the per-FK-path CSVs deriva-py emits today). Grounded in the on-disk
> 2-277G eye-ai bag (100 CSVs / ~80 tables, captured 2026-06-14) and the
> deriva-py `deriva/bag/catalog_builder.py` + `deriva/bag/loader.py`
> source. Nothing here ships until the deriva-py RID-set processor lands;
> this is the contract the design commits to.

## Why this document exists

The bag may be consumed three ways, and **two of them are not
deriva-ml**:

1. **deriva-ml's own loader** (`BagDatabase` → SQLite) — tolerant; it
   already unions multi-file tables by RID.
2. **A plain `bdbag` CLI user** — runs `bdbag --materialize`, then reads
   `data/**/*.csv` with pandas / DuckDB / `csvkit`. Has **no** deriva-ml
   union logic. This consumer is why the contract must be explicit.
3. **Chaise** — emits the *same export-spec format* from a catalog
   `tag:isrd.isi.edu,2016:export` annotation (see
   [§ Chaise relationship](#chaise-relationship)). Anything we change in
   the `csv` processor shape is a shape Chaise also generates.

The CSV interpretation rules and the export-spec shape are one contract.
This document states both, for Format A (today) and Format B (proposed).

---

## Part 1 — CSV interpretation rules

### Invariants (true in every format)

These are the BagIt / deriva-py engine guarantees. A consumer can rely
on them regardless of which generation path produced the bag.

- **INV1 — One CSV is one RFC-4180 table.** Header row + N data rows,
  engine-quoted. Never hand-written. `[engine: deriva-py]`
- **INV2 — Basename = catalog table name.** `Image.csv` holds
  `eye-ai:Image` rows. The mapping from file to table is the basename,
  never the directory. `[engine: deriva-py]`
- **INV3 — `RID` is the primary key.** Declared in `data/schema.json`
  (which the engine writes as a leading `json` processor). Every row is
  uniquely identified by `RID` within its table. `[engine: deriva-py]`
- **INV4 — Columns = the table's projected columns.** `RID` + scalar
  columns + FK columns. FK columns hold the *referenced* value — usually
  the parent's `RID`, but for a vocabulary FK the referenced `Name`
  (e.g. `Asset_Role` holds `"Input"`, not a RID). `data/schema.json`'s
  FK definitions are authoritative for which column references what.
  `[engine: deriva-py]`
- **INV5 — Asset bytes are not inline.** An asset table's CSV
  (`Image.csv`) carries `URL / Length / Filename / MD5`; the bytes live
  under `data/assets/...`, fetched from `fetch.txt`. To compute download
  size without materializing, `sum(Length)` over the asset CSV. `[deriva-ml]`

### Format A — per-FK-path CSVs (TODAY)

What's on disk now. Verified against the 2-277G bag: **100 CSV files for
~80 tables**; `Execution.csv` and `Workflow.csv` each appear in **10**
path-encoded directories, ~13 other tables in 2 each.

> **A1 — A table's rows may be split across multiple CSV files** in
> different path-encoded directories
> (`data/{schema}/{path_chain}/{table}.csv`). There is no single
> `Image.csv`. `[engine: deriva-py]`

> **A2 — To get the complete row set for table T, you MUST union and
> dedup:**
> ```
> glob   data/**/T.csv          # all files whose basename is T.csv
> concat all matches
> dedup  by RID                 # same RID recurs across FK routes
> ```
> deriva-ml's loader does this via SQLite `ON CONFLICT` (loader.py:498).
> A plain `bdbag` consumer must implement it themselves. `[engine: deriva-py]`

> **A3 — A single T.csv is a PARTIAL, path-scoped slice** — only the rows
> reachable via *that* FK path. Reading one `Image.csv` in isolation
> silently undercounts. This is the dangerous-by-default case: it looks
> like it worked. `[engine: deriva-py]`

> **A4 — The directory name is provenance, not semantics.** The chain
> (e.g. `Dataset_Subject_Dataset_Subject_Observation_Image/`) records the
> FK route the engine walked. Do **not** parse it to reconstruct joins —
> use `schema.json`'s FK definitions. `[engine: deriva-py]`

**The portability tax:** A2 + A3 mean a non-deriva-ml consumer who does
`read_csv("data/eye-ai/Image.csv")` against *one* of the two `Image.csv`
files gets a silently partial answer. The union contract is real but
hidden.

### Format B — one clean CSV per table (PROPOSED)

What the RID-set query processor produces. The rules collapse:

> **B-CSV1 — Exactly one CSV per table:** `data/{schema}/{table}.csv`.
> `Image.csv` is THE complete reachable Image set. `[engine: deriva-py, new]`

> **B-CSV2 — No union, no dedup, no glob.** The file *is* the union,
> already RID-distinct. `read_csv("data/eye-ai/Image.csv")` is correct
> and complete by construction. `[engine: deriva-py, new]`

> **B-CSV3 — Completeness in both directions.** Every row in `T.csv` is
> reachable from the dataset; every reachable row of T is in `T.csv`. The
> file is the canonical reachable set for T, path-independent.
> `[engine: deriva-py, new]`

> **B-CSV4 — Flat layout.** `data/{schema}/{table}.csv` — directory is
> the schema namespace only, no path-encoded subdirectories. `[engine: deriva-py, new]`

**What B buys the plain consumer:** the dangerous A3 case disappears.
`read_csv` of any table file is correct. The bag is self-describing via
`schema.json` (INV3/INV4) with no out-of-band union rule to know.

### Format comparison

| | Format A (today) | Format B (proposed) |
|---|---|---|
| CSVs for 2-277G | 100 files | ~80 files |
| Per table | 1..N files, path-encoded dirs | exactly 1 file, flat |
| Consumer post-processing | glob + union + **dedup by RID** | none — read the file |
| Reading one file in isolation | silently partial (A3) | correct (B-CSV2) |
| Generation cost (2-277G) | ~280s (deep FK joins) | ~10–30s (RID-set fetches) |
| Loader (`BagDatabase`) | unions via `ON CONFLICT` | one file → one table, trivial |
| deriva-py work | none (documentation only) | the RID-set processor |
| `schema.json` (RID = PK) | required (enables A2 dedup) | present (already distinct) |

---

## Part 2 — Export-spec shape

The export spec is a `tag:isrd.isi.edu,2016:export`-format document: a
`catalog` block with a list of `query_processors`. deriva-py's ERMrest
export engine executes it. The shape is what changes between A and B.

### Format A — export spec TODAY

Per-FK-path emission (`catalog_builder.py:558-600`,
`for fk_path in self._fk_paths_for(key)`). Shape:

```jsonc
{
  "catalog": {
    "query_processors": [
      // 1. leading schema dump
      { "processor": "json",
        "processor_params": { "query_path": "/schema",
                              "output_path": "schema" } },

      // 2. anchor table — filtered on its RID list
      { "processor": "csv",
        "processor_params": {
          "query_path":  "/entity/deriva-ml:Dataset/RID=any(...)",
          "output_path": "deriva-ml/Dataset",
          "paged_query": true } },

      // 3. NON-anchor table reached via ONE FK path — a deep chained join.
      //    ONE such processor PER FK ROUTE to the table. A table reachable
      //    via 10 routes => 10 processors, 10 output_paths, 10 CSVs.
      { "processor": "csv",
        "processor_params": {
          "query_path":  "/entity/deriva-ml:Dataset/RID=any(...)/Dataset_Subject/Subject/Observation/Image",
          "output_path": "eye-ai/Dataset_Subject_Dataset_Subject_Observation_Image/Image",
          "paged_query": true } },
      // ... 9 more Image processors for the other 9 routes ...

      // 4. full-vocabulary table — single unfiltered query (the exception)
      { "processor": "csv",
        "processor_params": {
          "query_path":  "/entity/deriva-ml:Asset_Role",
          "output_path": "deriva-ml/Asset_Role",
          "paged_query": true } },

      // 5. one fetch processor PER ASSET TABLE — reads URL/Length/Filename/MD5
      { "processor": "fetch",
        "processor_params": {
          "query_path":  "/attribute/eye-ai:Image/url:=URL,length:=Length,filename:=Filename,md5:=MD5,RID",
          "output_path": "assets/Image" } }
    ]
  }
}
```

**The cost driver:** processor #3's `query_path` is a multi-hop FK join
the *server* evaluates — 16–60s each on 2-277G, ~477 of them. That's the
~280s. The per-route fan-out is also what makes Format A's CSVs
multi-file.

### Format B — export spec PROPOSED

One RID-set processor per table. deriva-ml's client-side reachability
engine has already computed `{table: set(reachable_RIDs)}`, so the spec
no longer asks the server to join — it asks for explicit RID sets:

```jsonc
{
  "catalog": {
    "query_processors": [
      { "processor": "json",
        "processor_params": { "query_path": "/schema", "output_path": "schema" } },

      // ONE processor per reached table. No FK path in the query — just the
      // table and its precomputed reachable RID set. The NEW processor mode
      // chunks the RID list into URL-safe batches internally and APPENDS all
      // batches to ONE output CSV.
      { "processor": "csv-ridset",                  // <-- new processor mode
        "processor_params": {
          "table":       "eye-ai:Image",
          "rids":        ["<rid>", "<rid>", ...],    // engine chunks @ ~500
          "output_path": "eye-ai/Image",             // ONE flat path
          "paged_query": true } },

      // vocabulary full-export is unchanged — still a single unfiltered query
      { "processor": "csv",
        "processor_params": { "query_path":  "/entity/deriva-ml:Asset_Role",
                              "output_path": "deriva-ml/Asset_Role",
                              "paged_query": true } },

      // fetch is UNCHANGED — asset rows come back from the RID-set query with
      // URL/Length/Filename/MD5 exactly as from the join query
      { "processor": "fetch",
        "processor_params": { "query_path":  "/attribute/eye-ai:Image/url:=URL,length:=Length,filename:=Filename,md5:=MD5,RID",
                              "output_path": "assets/Image" } }
    ]
  }
}
```

**Three changes, scoped:**

1. **New `csv-ridset` processor mode** (`base_query_processor.py`) —
   takes `{table, rids}`, chunks `rids` into URL-safe batches (~500,
   matching `rid_lease.py`'s `PENDING_ROWS_LEASE_CHUNK = 500`), fetches
   `/entity/{table}/RID=any(batch)` per chunk, and **appends every chunk
   to one CSV** reusing the existing per-page `first_page`-skips-header
   append logic in `get_as_file` (`ermrest_catalog.py`). *This is the
   only genuinely new machinery.* `[engine: deriva-py, new]`

2. **`CatalogBagBuilder` emits one `csv-ridset` per table** instead of
   `for fk_path in self._fk_paths_for(key)`. FK-path *discovery* stays
   (still need the reachable-table set); only query *emission* changes.
   deriva-ml feeds the per-table RID sets. `[engine: deriva-py, new]` +
   `[deriva-ml]` (supplies the RID sets)

3. **`fetch` and `json` processors unchanged.** Verified: fetch reads
   asset columns from the table directly — the RID-set query returns the
   same columns as the join query. `[engine: deriva-py]`

### Why the URL limit is handled, not hit

A naive `RID=any(rid1,...,rid29000)` filter exceeds the URL length limit
(ERMrest is GET-only — no POST-body query). The `csv-ridset` processor
avoids this by chunking *inside* the processor and appending across
chunks — the same way `get_as_file` already appends across *pages* of
one paged query. The consumer sees one clean CSV; the URL-safety is an
engine-internal concern. `[engine: deriva-py, new]`

---

## Chaise relationship

**The bag export spec and Chaise's export annotation are the same wire
format.** Both are `tag:isrd.isi.edu,2016:export` query-processor
documents consumed by the same deriva-py download engine
(`core_utils.py:772`). The difference is only the *source*:

| | Bag builder | Chaise |
|---|---|---|
| Spec source | generated by `CatalogBagBuilder` | catalog `tag:isrd.isi.edu,2016:export` annotation |
| Trigger | `download_dataset_bag` / `estimate_bag_size` | user clicks "Export → BDBag" in the UI |
| Processor format | identical | identical |
| Engine | deriva-py ERMrest export engine | same |

**Implication for Format B — two distinct surfaces, do not conflate:**

- **The new `csv-ridset` processor is opt-in.** Adding a processor
  *mode* to the engine does **not** change how existing `csv` (query_path)
  processors behave. Chaise's hand-authored / annotation-driven export
  specs use `query_path` and keep working unchanged. We are *adding* a
  capability, not altering the existing one.
- **Chaise export annotations are author-controlled, RID-set is
  programmatic.** Chaise specs are written by a catalog admin to describe
  "what a user can export from this table/row" — they don't have a
  precomputed client-side reachability set, so they stay `query_path`.
  The RID-set mode is for *programmatic* generation where the caller
  (deriva-ml) already holds the reachable RIDs. The two coexist.
- **The new processor SHOULD be usable from a Chaise annotation too** if
  an author ever wants it, but that's not required for this work — and
  an annotation author rarely has a RID set to inline. Treat
  Chaise-annotation use of `csv-ridset` as out of scope for v1.
- **`schema.json` / RID-PK / fetch.txt are shared guarantees.** A Chaise
  BDBag and a deriva-ml bag carry the same `data/schema.json` and the
  same `fetch.txt` mechanics, so INV1–INV5 hold for both regardless of
  format. A consumer's reading rules don't need to know which tool
  produced the bag — only which *format* (A vs B) it's in, and that's
  discoverable from the directory shape (path-encoded dirs ⇒ A; flat
  one-per-table ⇒ B).

**Net:** Format B is additive at the engine layer. Chaise's existing
export behavior is untouched; deriva-ml opts into the new processor mode
for its programmatic bags. The cross-cutting risk ("we'd break Chaise
exports") does not materialize because we add a mode rather than change
the `csv` processor's contract.

---

## Open questions for the design

1. **Processor name.** `csv-ridset` vs. extending `csv` with an optional
   `rids` param (when `rids` present, switch to chunk-append mode). The
   latter is fewer concepts; the former is clearer in a spec dump.
   *Leaning: extend `csv` with optional `rids`* — keeps one CSV processor,
   the presence of `rids` selects the mode.
2. **Chunk size.** 500 (match `rid_lease.py`) vs. derive from a URL-length
   budget. *Leaning: start at 500, make it a param.*
3. **Empty-table handling.** A reached table with zero reachable RIDs —
   emit an empty CSV (header only) or skip? Format A emits the file;
   Format B should match (header-only CSV) so the table set is stable.
4. **Migration window.** Do existing cached Format-A bags need to be
   re-fetched, or does the loader read both? The loader reads both today
   (basename-keyed); a Format-B bag is a strict subset of what the loader
   already handles. *Leaning: no migration needed; loader is
   format-agnostic.*

## See also

- `docs/reference/bag-export.md` — the shipped Format A behavior (rules
  B1–B5). This document's Format A section is the consumer-facing
  restatement of B4.
- `docs/reference/fk-traversal.md` — the FK walk that discovers the
  reachable-table set (unchanged by Format B; only query *emission*
  changes).
- `docs/adr/0008-estimate-bag-size-bypasses-bag-pipeline.md` — the
  estimate's deliberate bypass; the reachability engine generalizes this
  bypass into a shared core that *also* feeds bag generation.
- deriva-py `deriva/bag/catalog_builder.py::_build_export_spec` — where
  the per-table emission lives (the change site for Format B).
- deriva-py `deriva/transfer/download/processors/query/base_query_processor.py`
  — where the `csv-ridset` chunk-append mode would live.
