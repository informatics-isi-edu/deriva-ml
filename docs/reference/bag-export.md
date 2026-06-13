# Bag Export Reference

> Verified against a populated demo catalog
> (`create_demo_catalog(create_features=True, create_datasets=True)`,
> deriva-ml v1.46.x, catalog `106`, captured 2026-06-13). Rules tagged
> `[engine: deriva-py]` are enforced upstream in the `deriva.bag`
> package (the export-spec generator and the server-side ERMrest export
> engine); rules tagged `[deriva-ml]` are deriva-ml's own configuration
> of that engine. The "Worked examples" section pastes verified
> excerpts from `docs/reference/.examples/` — every numeric/structural
> claim traces to a capture there.

Bag export is how deriva-ml turns a **dataset** into a downloadable
[BDBag](https://github.com/fair-research/bdbag): it computes the
*anchors* (where the FK walk starts) and the *traversal policy* (what
the walk is allowed to reach), hands both to deriva-py's
`CatalogBagBuilder`, which generates a server-side **export spec** that
the ERMrest export engine runs to produce the bag's CSV files and
fetched asset bytes.

The behavior splits across two layers, the same split as in
[FK Traversal](fk-traversal.md):

- **deriva-ml decides *scope*.** `DatasetBagBuilder`
  (`src/deriva_ml/dataset/bag_builder.py`) computes the anchor set
  (`anchors_for`) and the `FKTraversalPolicy` (`build_policy`) for one
  dataset — which RIDs to start from, which `Dataset_X` associations to
  prune, which tables are terminal.
- **deriva-py decides *fetch mechanics*.** `CatalogBagBuilder`
  (`.venv/.../deriva/bag/catalog_builder.py`) runs the FK walk under
  that policy and translates the reachable-table set into an export
  spec — one CSV `query_processor` per FK path, one `fetch` processor
  per asset table. The ERMrest export engine executes that spec
  server-side.

For the FK walk itself — the bidirectional traversal, terminal tables,
vocabulary leaves, depth bounds — see [FK Traversal](fk-traversal.md);
its rules T1–T7 are the mechanism this page configures. This page
documents the dataset-export *scope decisions* layered on top.

## Quick answers

| Question | Rule |
|---|---|
| Where does the walk start for a nested dataset? | [B1](#b1) (root + every descendant RID) |
| Why does my flat dataset's bag omit `Dataset_Image`? | [B2](#b2) (empty-association pruning) |
| My dataset has no images — why is `Dataset_Image` in the bag anyway? | [B2](#b2) (a nested child has images) |
| Why doesn't the bag carry the producing executions' assets? | [B3](#b3) (provenance terminal) → [T3](fk-traversal.md#t3) |
| What does the export spec actually look like? | [B4](#b4) (one processor per FK path + fetch) |
| What lands in the bag, and what doesn't? | [B5](#b5) |
| Does `estimate_bag_size` use this same walk? | [Same engine](#same-engine-different-consumer) (ADR-0008) |

## Mental model

A bag export answers one question: *"give me everything reachable from
this dataset, as files."* Producing the bag is a pipeline with a clean
**scope / mechanics** division.

```
DatasetLike  ──► DatasetBagBuilder ──► CatalogBagBuilder ──► ERMrest export engine ──► BDBag
  (deriva-ml)      anchors_for          _build_export_spec      (server-side)            .zip
                   build_policy         (FK walk + spec)
                   ───── scope ─────    ──── mechanics ────     ───── execution ─────
```

1. **Scope (deriva-ml).** `DatasetBagBuilder` turns the dataset into
   an **anchor set** ([B1](#b1)) and a **policy** that prunes
   member-less `Dataset_X` associations ([B2](#b2)) and marks the
   provenance hub terminal ([B3](#b3)). Nothing is fetched yet — this
   is pure scope computation.
2. **Mechanics (deriva-py).** `CatalogBagBuilder` runs the
   [FK walk](fk-traversal.md) under that policy to discover the
   reachable `(schema, table)` set, then emits an **export spec**: a
   list of `query_processors`, one CSV query per FK path to each
   reached table, plus a `fetch` processor per asset table
   ([B4](#b4)).
3. **Execution (server-side).** The ERMrest export engine runs the
   spec, writing one CSV per processor under `data/{schema}/` and
   downloading asset bytes under `asset/{rid}/`. The result is
   archived as a `.zip` BDBag.

The two builders are the load-bearing distinction. **deriva-ml never
fetches rows** — it only decides scope. **deriva-py never decides what's
in scope for a dataset** — it consumes the anchors and policy deriva-ml
hands it and turns them into fetch mechanics. This is the same
two-layer contract that ADR-0006 established for all bag-oriented data
movement.

## Formal rules

<a id="b1"></a>
**B1 — The anchor set is the root RID plus every recursive descendant
dataset RID.** `DatasetBagBuilder.anchors_for` builds one
`RIDAnchor(table="Dataset")` for the dataset's own RID, then appends one
more `RIDAnchor` for **every** descendant dataset RID, discovered by
walking `list_dataset_children` recursively
(`bag_builder.py::anchors_for` + `_iter_descendant_rids`). A flat
dataset (no children) anchors at exactly one RID; a nested dataset
anchors at its whole subtree. Each anchor RID becomes a BFS root for the
[FK walk](fk-traversal.md#t1) ([T1](fk-traversal.md#t1)), so the walk
reaches the members of *every* dataset in the tree, not just the root's.
`[deriva-ml]` `src/deriva_ml/dataset/bag_builder.py::anchors_for`,
`::_iter_descendant_rids`.

Verified on the demo catalog (`docs/reference/.examples/`): the flat
dataset `5D0` anchors at **1 RID** (`export.txt`:
`ANCHORS ["RIDAnchor:['5D0']"]`). The nested dataset `5CM` — whose own
members are just `{Dataset: 2}` (two child datasets, no direct
content) — anchors at **7 RIDs**: its own `5CM` plus the six recursive
descendants `5CT, 5D0, 5DA, 5DY, 5EE, 5E4` (`nested_b2.txt`):

```
RID=5CM own={'Dataset': 2} children=['5CT', '5DY'] descendant_members={'Dataset': 4, 'Image': 4}
  anchors=["RIDAnchor:['5CM']", "RIDAnchor:['5CT']", "RIDAnchor:['5D0']",
           "RIDAnchor:['5DA']", "RIDAnchor:['5DY']", "RIDAnchor:['5EE']", "RIDAnchor:['5E4']"]
```

`5CM`'s direct children are `5CT` and `5DY`, but the anchor list is
seven RIDs deep because `_iter_descendant_rids` recurses: `5CT`'s own
children (`5D0`, `5DA`) and `5DY`'s descendants (`5EE`, `5E4`) all
become anchors too. The walk therefore starts from every dataset in the
subtree simultaneously.

<a id="b2"></a>
**B2 — A `Dataset_X` association is included iff the dataset *or any
descendant* has a member of element-type X (or X is a vocabulary).**
`DatasetBagBuilder._exclude_empty_associations` scans the member counts
of the root **and every recursive descendant** (the same RID set
[B1](#b1) anchors at). For each `Dataset_X` association table, it keeps
the table only when at least one of those datasets has a member whose
element type is X, *or* when the association links to a vocabulary
table (those carry dataset metadata and always come along). Associations
that are empty across the whole tree are added to the policy's
`exclude_tables`, so the [generic walker prunes them before the edge is
walked](fk-traversal.md#t5) ([T5](fk-traversal.md#t5)).
`[deriva-ml]` `src/deriva_ml/dataset/bag_builder.py::_exclude_empty_associations`
(feeding `build_policy`'s `exclude_tables`).

**The descendant clause is the subtle, load-bearing part.** Scoping the
member scan to the root alone would silently drop element types that
only appear under a nested child (regression #94: child `Image` members
disappearing). Verified on the demo catalog:

- **Flat `5D0`** has members `{Subject: 2, Image: 0, ...}` — zero
  images. Its `Dataset_Image` association is therefore **excluded**
  (`export.txt`):

  ```
  EXCLUDED_EMPTY_ASSOCIATIONS ['demo-schema.Dataset_Image',
                               'deriva-ml.Dataset_Dataset',
                               'deriva-ml.Dataset_File']
  ```

- **Nested `5CM`** has **zero direct Image members** (its own members
  are `{Dataset: 2}`) — yet `Dataset_Image` is **not** excluded
  (`nested_b2.txt`: `Dataset_Image_excluded=False`; the only
  excluded-empty entry is `['deriva-ml.Dataset_File']`). The reason:
  `5CM`'s descendants *do* have Image members
  (`descendant_members={'Dataset': 4, 'Image': 4}`), so the descendant
  clause keeps `Dataset_Image` in scope and the child datasets' image
  rows reach the bag.

This is the eye-ai subtlety reproduced on the demo catalog: a dataset
with no images of its own still includes `Dataset_Image` — and hence its
children's images — because a nested child has them. Without the
descendant clause, downloading `5CM` would yield a bag missing every
image in its subtree.

<a id="b3"></a>
**B3 — The provenance hub is terminal: `{Execution, Workflow}` are
entered but their inbound FKs are not followed.** `build_policy` sets
`terminal_tables=set(PROVENANCE_TERMINAL_TABLES)`
(`bag_builder.py:812`), sourcing the set from the shared
`src/deriva_ml/core/constants.py::PROVENANCE_TERMINAL_TABLES` =
`frozenset({("deriva-ml", "Execution"), ("deriva-ml", "Workflow")})`.
The walker still *reaches* `Execution` and `Workflow` (their rows are
emitted, so other rows' FKs into them resolve at load) and still follows
their *outbound* FKs, but it does **not** follow their *inbound* FKs —
see [T3](fk-traversal.md#t3) for the mechanism.
`[deriva-ml]` chooses the set
(`src/deriva_ml/dataset/bag_builder.py::build_policy`,
`src/deriva_ml/core/constants.py::PROVENANCE_TERMINAL_TABLES`);
`[engine: deriva-py]` enforces the inbound-skip
(`deriva/bag/path_walker.py`, the `is_terminal` check —
[T3](fk-traversal.md#t3)).

> **This is the post-#297 default.** Before that fix, `build_policy`
> did not set `terminal_tables`, so a dataset export walked *inbound*
> from `Execution` and `Workflow` into the entire provenance closure.
> `build_policy` and `clone_via_bag` now both source the same constant.

**Consequence for the bag.** The bag carries the provenance **link**
rows — `Dataset_Execution`, `*_Execution` feature associations, the
`Execution` and `Workflow` rows themselves — so you can see *which*
execution produced a dataset's features. But it does **not** carry the
producing executions' *asset closure*: the `Execution_Asset` /
`Execution_Metadata` / `File` tables and their `*_Asset_Type` /
`*_Execution` associations are reached only by inbound traversal from
`Execution`, which [B3](#b3) cuts. The verified reached-set contrast for
this rule — **30 tables with the hub terminal vs 42 without**, and the
exact 12-table severed set — is documented in
[FK Traversal, T3 and Worked examples](fk-traversal.md#t3); that contrast
was captured on dataset `5D0` of this same demo catalog.

<a id="b4"></a>
**B4 — The export spec is one CSV `query_processor` per FK path, plus a
`fetch` processor per asset table.** After the walk discovers the
reachable `(schema, table)` set, `CatalogBagBuilder._build_export_spec`
translates it into a deriva-py export spec — a list of `query_processors`
under `catalog`:

- A leading **`json`** processor writes `data/schema.json` (the catalog
  schema).
- For each reached table, **one `csv` processor per FK path** that the
  walker found to it (`_fk_paths_for` → `_table_query_path`). Each
  carries `"paged_query": True` (cursor pagination over RID) and an
  `output_path` that encodes the path's table chain, so multiple FK
  routes to the same table land at distinct on-disk locations; the
  bag's loader unions their rows by RID. A `vocab_export=FULL` vocabulary
  table is the exception — it gets a single unfiltered
  `/entity/{schema}:{table}` query (the whole controlled vocabulary),
  not one-per-path.
- For each reached table that **`is_asset()`**, one additional
  **`fetch`** processor that reads `URL` / `Length` / `Filename` / `MD5`
  / `RID` from each asset row and downloads the bytes to
  `asset/{asset_rid}/{table}`.

`[engine: deriva-py]`
`deriva/bag/catalog_builder.py::_build_export_spec`,
`::_table_query_path` (the spec is retrievable without running via
`::get_export_spec`).

The spec is *declarative* — it describes the queries; the ERMrest export
engine executes them server-side. `_table_query_path` scopes each query
to rows reachable from the anchor RIDs: an anchor table filters on its
RID list (`/entity/{schema}:{table}/RID=any(...)`), and a non-anchor
table reached via FK gets a chained path
(`/entity/{anchor}/RID=any(...)/{step2}/...`) that ERMrest joins along
its natural FKs.

<a id="b5"></a>
**B5 — What lands in the bag.** Combining [B1](#b1)–[B4](#b4), a dataset
bag contains:

- **Members** — every member row of the root *and* every nested
  descendant ([B1](#b1)), plus the domain rows those members reference
  by FK ([T2](fk-traversal.md#t2)).
- **Features and annotations** — feature tables and their `*_Execution`
  associations, reached by FK-following from member rows; no separate
  policy field is needed.
- **Referenced (and, by deriva-ml's choice, *full*) vocabularies** —
  `build_policy` defaults to `vocab_export=VocabExport.FULL`, so each
  reached vocabulary table exports *every* term ([B4](#b4)), and the
  walk stops at vocab tables ([T4](fk-traversal.md#t4)).
- **Provenance *link* rows** — `Dataset_Execution`, the `Execution` and
  `Workflow` rows, and the feature `*_Execution` associations, so the
  bag records which executions produced its contents ([B3](#b3)).
- **Asset bytes** — for every reached asset table, fetched by the
  `fetch` processors ([B4](#b4)).

A dataset bag does **not** contain the **execution asset-closure** — the
`Execution_Asset` / `Execution_Metadata` / `File` tables of the
producing executions — because the provenance hub is terminal
([B3](#b3); the severed set is the 12 tables documented in
[FK Traversal, T3](fk-traversal.md#t3)). The bag tells you *that* an
execution produced a feature; it does not pull in *that execution's
other outputs and metadata*.

## Same engine, different consumer

Three other deriva-ml operations drive the **same** `CatalogBagBuilder`
walk with a different consumer or different anchors. The shared walker
is what makes "the bag walk *is* the drift walk *is* the size walk" a
load-bearing invariant rather than an aspiration
(`aggregate_queries` docstring: *"the drift walk is the bag walk"*).

- **`estimate_bag_size`** shares the *walk* but bypasses the *export
  engine*. It calls `DatasetBagBuilder.aggregate_queries(dataset)` — the
  same anchors and policy as the export — but instead of generating an
  export spec it runs live datapath aggregate queries against the
  reached tables and computes exact RID-union counts client-side. It
  never builds a bag or contacts the export engine. This deliberate
  bypass is ADR-0008
  (`docs/adr/0008-estimate-bag-size-bypasses-bag-pipeline.md`).

- **`clone_via_bag`** uses the *same policy shape* — including the same
  `terminal_tables={Execution, Workflow}` and `vocab_export=FULL` — but
  with **different anchors** (a single `RIDAnchor` on the clone's
  `root_rid`, not a dataset's descendant subtree) and a different
  load-phase choice: `dangling_fk_strategy=DanglingFKStrategy.DELETE`
  (drop orphan rows) rather than the dataset export's defaults. The
  bag-as-unifying-artifact decision is ADR-0006
  (`docs/adr/0006-bag-oriented-data-movement.md`).

- **Dataset drift (`is_dirty`)** shares the walk too. `Dataset.is_dirty`
  walks "the same foreign-key paths used to build the dataset's bag"
  (its docstring) via `aggregate_queries`, short-circuiting on the first
  reached table with a row modified since the last released version's
  snapshot. Because it reuses the export walker, drift is defined over
  *exactly* the rows the bag would contain — change anything the bag
  would carry and the dataset is dirty.

## See also

- [FK Traversal](fk-traversal.md) — the walk this page configures
  (anchors [T1](fk-traversal.md#t1), bidirectional traversal
  [T2](fk-traversal.md#t2), terminal tables [T3](fk-traversal.md#t3),
  vocabulary leaves [T4](fk-traversal.md#t4), pruning
  [T5](fk-traversal.md#t5), depth [T6](fk-traversal.md#t6),
  load-phase fields [T7](fk-traversal.md#t7)).
- [Denormalization](denormalization.md) — how the bag's per-table CSVs
  become a single denormalized frame.
- ADR-0006 — bag-oriented data movement; the unifying-artifact decision
  behind the two-builder split (`docs/adr/0006-*`).
- ADR-0008 — `estimate_bag_size` deliberately bypasses the bag pipeline
  (`docs/adr/0008-*`).
- ADR-0002 — `validate_*` (metadata-only pre-flight) vs `dry_run`
  (full-path), for where bag-walk cost shows up in the execution
  lifecycle (`docs/adr/0002-*`).
