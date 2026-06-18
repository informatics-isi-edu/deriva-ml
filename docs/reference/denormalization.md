# Denormalization Reference

> Verified against a populated demo catalog
> (`create_demo_catalog(create_features=True, create_datasets=True)`,
> deriva-ml v1.46.x, catalog `106`, captured 2026-06-13). Every rule on
> this page is tagged `[deriva-ml]` — denormalization is **pure local
> computation** inside deriva-ml's own code, with no deriva-py engine
> involvement and no catalog access. The "Worked examples" section
> pastes verified excerpts from `docs/reference/.examples/denorm.txt`.
>
> **For a gentle, example-led introduction**, read the
> [denormalization tutorial](../user-guide/denormalization.md) first.
> This page is the formal counterpart — it states the rules precisely
> and names the code locations, but does not re-teach the concepts.

Denormalization turns an **already-downloaded** dataset bag into a
single **wide table** — one row per ML observation, with columns pulled
side-by-side from related tables along the bag's foreign-key graph. The
four entry points are methods on `DatasetBag`:

| Method | Returns | Fetches data? |
|---|---|---|
| `get_denormalized_as_dataframe` | `pandas.DataFrame` | yes (from local SQLite) |
| `get_denormalized_as_dict` | **`Generator[dict, None, None]`** | yes (yields rows) |
| `list_denormalized_columns` | `list[tuple[str, str]]` | no — model-only |
| `describe_denormalized` | `dict` (13-key plan) | no — dry-run |

All four live in `src/deriva_ml/dataset/dataset_bag.py` and delegate to
`src/deriva_ml/local_db/denormalizer.py::Denormalizer`, which executes
against the bag's local SQLite database.

## Quick answers

| Question | Rule |
|---|---|
| How is the row grain (one-row-per-what) chosen? | [Row grain](#row-grain) (`row_per`, default = furthest-downstream table) |
| Why do my `Subject` columns repeat across rows? | [Column hoisting](#column-hoisting) |
| Which tables contribute columns vs. just a join path? | [Column projection](#column-projection) (`include_tables` vs `via`) |
| I have two annotators per image — how do I pick one? | [Column hoisting](#column-hoisting) (the `selector`) |
| Why was my explicit `row_per` rejected? | [Downstream-leaf rejection](#downstream-leaf-rejection) |
| Why did two FK paths raise instead of picking one? | [Path ambiguity](#path-ambiguity) |
| My anchors have no FK path — error or skip? | [Anchor disposition](#anchor-disposition) (case 6, `ignore_unrelated_anchors`) |
| Some members reach no `row_per` row — dropped or NULL? | [Anchor disposition](#anchor-disposition) (case 3, orphan rows) |
| How do I keep `RCB`/`RCT` provenance columns? | [Return shapes](#return-shapes) (`system_columns`, **dataframe only**) |
| Does `_as_dict` return a list I can index? | No — [Return shapes](#return-shapes) (it's a **generator**) |
| My dataset's members are `Subject`s but I read `Image` features — why does it work? | [FK-reachable scoping](#fk-reachable-scoping) (the planner unions FK-reachable routes, not just direct membership) |
| What does denormalization have to do with FK-traversal policy? | Nothing — see [Mental model](#mental-model) |

## Mental model

Denormalization is **star-schema flattening over a downloaded bag.**
You name the tables whose columns you want; deriva-ml reads the bag's
SQLite, finds the table furthest *downstream* in the FK chain, and
emits **one row per instance of that table**. Columns from *upstream*
tables (the ones it references) are hoisted and **duplicated** across
every row that shares the same upstream row — the classic
fact-table-plus-dimensions shape.

```
include_tables = ["Subject", "Observation", "Image"]
        Subject ◄──── Observation ◄──── Image          (FK chain)
        (upstream)                      (furthest downstream = row_per)

result: one row per Image; Subject/Observation columns repeated
```

Two things this is **not**:

- **It is not a catalog operation.** Everything runs against the bag's
  local SQLite — no network, no `DerivaML` connection, no snapshot. A
  `DatasetBag` produced offline denormalizes offline. (A live `Dataset`
  *can* also be denormalized, via the same `Denormalizer` class fed by
  a paged catalog client, but the four `DatasetBag` methods documented
  here are the pure-local path.)

- **It is NOT the FK-traversal-policy operation.** This is the single
  most important distinction on this page. [Bag Export](bag-export.md)
  and [FK Traversal](fk-traversal.md) describe a *different* operation:
  a bidirectional FK walk, governed by `FKTraversalPolicy`, that
  decides **which rows of which tables land in the bag**. Denormalization
  runs **after** that — it takes the rows the bag already contains and
  reshapes them into a wide frame. There is no `FKTraversalPolicy` here,
  no terminal-tables, no `max_depth`, no anchors-as-BFS-roots; the bag's
  membership is simply a **filter** on what's in scope. Don't conflate
  the two: one *produces* the bag, the other *reshapes* it.

  (Within that bag-local scope, the planner reaches an `include_tables`
  element through **every** `Dataset → element` route — direct membership
  *and* FK-reachable chains — unioned RID-distinct; see
  [FK-reachable scoping](#fk-reachable-scoping).)

The denormalizer is governed by **six named semantic rules**. This page
is their canonical definition; the
[tutorial](../user-guide/denormalization.md) gives the example-led
version and the
[implementation contract](../design/denormalization-contract.md) §6.2
gives the full planner-level language. The six rules answer, in order:

| Rule | Answers |
|---|---|
| [Row grain](#row-grain) | *How many rows, keyed on what?* (`row_per`, default = furthest-downstream table) |
| [Column projection](#column-projection) | *Which tables contribute columns vs. just a join path?* (`include_tables` vs `via`) |
| [Column hoisting](#column-hoisting) | *Why do upstream columns repeat across rows?* |
| [Downstream-leaf rejection](#downstream-leaf-rejection) | *Why was my explicit `row_per` rejected?* |
| [Path ambiguity](#path-ambiguity) | *Why did two FK paths raise instead of picking one?* |
| [Anchor disposition](#anchor-disposition) | *What happens to each dataset member — row, filter, orphan, or error?* |

Two further behaviors are not caller-tunable "rules" but you need them to
predict a result: [**FK-reachable scoping**](#fk-reachable-scoping) (how
the dataset's members define which rows are in scope) and
[**return shapes**](#return-shapes) (what each of the four methods hands
back). Both follow the rules table.

## Formal rules

<a id="row-grain"></a>
### Row grain

**The output has one row per `row_per`-table instance in scope;
`row_per` defaults to the furthest-downstream requested table.**

"In scope" means **FK-reachable from a dataset member**, not *member-of*
(see [FK-reachable scoping](#fk-reachable-scoping)). By default `row_per`
is **auto-inferred**: among `include_tables`, the one that no other
requested table points to via FK — the "deepest" / furthest-downstream
table — becomes the grain. Auto-inference **raises** if two candidates
tie (`DerivaMLDenormalizeMultiLeaf`) or the FK subgraph has a cycle with
no sink (`DerivaMLDenormalizeNoSink`); pass `row_per=` explicitly to
resolve. (Setting `row_per` to a table *downstream* of another requested
table is rejected — see [Downstream-leaf rejection](#downstream-leaf-rejection).)
`[deriva-ml]`
`src/deriva_ml/local_db/denormalizer.py::Denormalizer.as_dataframe`
(`row_per` argument; sink-finding is
`src/deriva_ml/model/denormalize_planner.py::DenormalizePlanner._find_sinks`);
surfaced through each `DatasetBag` method's `row_per` keyword.

<a id="column-projection"></a>
### Column projection

**`include_tables` names the column sources; `via` adds a join hop with
no columns.** Columns come **only** from tables named in
`include_tables`. Tables that appear only as transitive intermediates —
association tables bridging two requested tables — are joined *through*
but contribute **no columns** ("transparent intermediates"). `via` is
the explicit "route through this table but don't project it" knob, used
to resolve a [path ambiguity](#path-ambiguity) without cluttering the
output. Feature names (e.g. `"Image_Classification"`) in `include_tables`
are resolved to the underlying feature-association table before
projection (see [the contract §8.4](../design/denormalization-contract.md)).
`[deriva-ml]` `src/deriva_ml/local_db/denormalizer.py` (`include_tables`
/ `via` arguments; transparent-intermediate detection in the planner).

<a id="column-hoisting"></a>
### Column hoisting

**Upstream columns are hoisted and duplicated across rows that share an
upstream row.** For each `row_per` row, columns from the tables it
references (transitively, up the FK graph) are **repeated verbatim**
across every output row that shares that upstream row — the classic
star-schema shape. A table reached through an N-to-N link produces one
output row per `(row_per, linked-row)` combination, so the `row_per`
count can grow. Output columns use `Table.column` notation (e.g.
`Subject.Name`).

This is also where a **feature `selector`** acts: when `include_tables`
contains **exactly one** feature-association table, the optional
`selector` callable reduces each feature's multi-row group (e.g. several
annotators' labels for one image) to a single chosen row before
hoisting. Its signature is
`Callable[[list[FeatureRecord]], FeatureRecord | None]` — identical to
`DerivaML.feature_values`'s `selector`; built-ins on `FeatureRecord`
include `select_newest`, `select_first`, `select_by_workflow`, and a
custom callable may return `None` to drop the group. With no `selector`,
every feature row is kept, multiplying the `row_per` count. Passing
`selector` without exactly one feature-association table raises
`ValueError`. `selector` is accepted by `get_denormalized_as_dataframe`
and `get_denormalized_as_dict` only.
`[deriva-ml]` `src/deriva_ml/local_db/denormalize.py::_denormalize_impl`
(the SQL join + projection); `src/deriva_ml/feature/__init__.py::FeatureRecord`
for the built-in selectors.

<a id="downstream-leaf-rejection"></a>
### Downstream-leaf rejection

**An explicit `row_per` with a downstream requested table is rejected.**
If you pass `row_per=` and another table in `include_tables` is
*downstream* of it (i.e. `row_per` has an outbound FK path to that
table), the planner raises `DerivaMLDenormalizeDownstreamLeaf` rather
than silently aggregating. One row per `row_per` would require
collapsing many downstream rows per `row_per` row, and aggregation is
not a denormalize concern. Resolve by dropping `row_per` (let
auto-inference pick the downstream table as the grain) or removing the
downstream table from `include_tables`.
`[deriva-ml]`
`src/deriva_ml/model/denormalize_planner.py::DenormalizePlanner._determine_row_per`.

<a id="path-ambiguity"></a>
### Path ambiguity

**Multiple distinct FK paths between `row_per` and a requested table
raise rather than guess.** When two or more distinct FK paths connect
`row_per` to another requested table, the planner raises
`DerivaMLDenormalizeAmbiguousPath` listing the paths and suggesting
intermediates — the result would differ between paths, and deriva-ml
won't pick one silently. Resolve by adding the disambiguating
intermediate to `include_tables` (its columns appear) or to `via`
(path-only, no columns), or by narrowing the request so only one path is
valid. (This is about *distinct* paths to the **same** table; it is
**not** the same as the membership-vs-FK-reachable *route* union in
[FK-reachable scoping](#fk-reachable-scoping), which unions routes to the
same element and never raises.)
`[deriva-ml]`
`src/deriva_ml/model/denormalize_planner.py::DenormalizePlanner._find_path_ambiguities`.

<a id="anchor-disposition"></a>
### Anchor disposition

**Every dataset member (anchor) contributes in exactly one way,
determined by its table and reachability.** The six cases:

| Case | Anchor table | Reaches a `row_per` row? | Contribution |
|---|---|---|---|
| 1 | `== row_per` | — | one output row (itself, upstream hoisted) |
| 2 | `∈ include_tables`, upstream of `row_per` | yes | filter-only (appears via the `row_per` row that hoists it) |
| 3 | `∈ include_tables`, upstream of `row_per` | no | one **orphan** row — its columns populated, `row_per`-side columns `NULL` (LEFT-JOIN shape) |
| 4 | `∉ include_tables`, upstream of `row_per` | yes | filter-only (no row of its own — its columns aren't projected) |
| 5 | `∉ include_tables`, upstream of `row_per` | no | silently dropped (would not affect output) |
| 6 | no FK path to any `include_tables ∪ via` table | — | raises `DerivaMLDenormalizeUnrelatedAnchor` unless `ignore_unrelated_anchors=True` (then silently dropped) |

`ignore_unrelated_anchors` (default `False`) controls **only** case 6;
it does **not** affect orphan (case 3) emission. Empty anchor sets are
skipped and never trigger case 6. After the main join, a per-RID orphan
scan catches case-3 anchors whose specific RIDs didn't appear in the
result (e.g. a `Subject` with an empty `Image` set) and emits their
orphan rows.
`[deriva-ml]`
`src/deriva_ml/local_db/denormalizer.py::Denormalizer._classify_anchors`
(+ `_emit_orphan_rows`); `ignore_unrelated_anchors` accepted by
`get_denormalized_as_dataframe` / `get_denormalized_as_dict`.

## Behaviors (not caller knobs)

<a id="fk-reachable-scoping"></a>
### FK-reachable scoping

**The dataset's members define scope by FK-reachability: every
`Dataset → … → element` route is unioned, RID-distinct.** To reach an
`include_tables` element from the dataset, the planner does **not** pick
a single "best" route — it **unions every** route it can find: the
**direct-membership association** (`Dataset_{Element}`, e.g.
`Dataset_Image`) **and** **FK-reachable chains** (e.g.
`Dataset → Dataset_Subject → Subject → … → Image`). The union is
**RID-distinct** (SQL `UNION`, not `UNION ALL`), so an element reachable
several ways contributes a single row.

- On a **directly-populated** dataset the membership route is non-empty
  and the FK routes add no new leaf RIDs, so the union equals a
  membership-only join — directly-populated reads are unchanged.
- On a **subject-partitioned** dataset (members are an upstream element
  like `Subject`; the target is reachable **only** via the FK chain,
  with an empty membership association) this union is the **only** way
  the target's rows appear — a membership-only join returns **0**. This
  is what makes `feature_values` / `get_denormalized_*` work on
  subject-partitioned datasets.

**Only the `row_per` element's routes drive rows.** The union is applied
**per element**, but the table is [`row_per`-grained](#row-grain): only
routes that *end at the `row_per` table* produce rows. Routes ending at a
non-`row_per` include-table are **not** unioned into the row set — that
table's columns are [hoisted](#column-hoisting) into the `row_per`
routes' join, not emitted as their own grain. (Emitting them was the
#322 cartesian: every route is projected with the *full* column set, so
a non-`row_per` route that never joins the `row_per` table cross-joins
its columns — `[Image, Observation]` with `row_per=Image` produced 8×8
rows instead of 8. The planner now keeps only the `row_per` element's
routes.)
`[deriva-ml]`
`src/deriva_ml/model/denormalize_planner.py::DenormalizePlanner._prepare_wide_table`
(per-`row_per`-element route discovery + UNION-distinct emission);
shared by `feature_values`, `get_denormalized_*`, and `describe`.
Pinned by
`tests/local_db/test_denormalize_fk_reachable_paths.py::test_only_row_per_element_drives_routes`
and
`tests/dataset/test_denormalize.py::TestDenormalize::test_denormalize_row_per_grain_no_cartesian`.
Spec:
`docs/superpowers/specs/2026-06-16-denormalize-fk-reachable-paths-design.md`.

<a id="return-shapes"></a>
### Return shapes

The four entry points return distinct types — pick by what you need:

- **`get_denormalized_as_dataframe(...)` → `pandas.DataFrame`.** One row
  per `row_per` instance, `Table.column` headers. Materializes the whole
  frame in memory. Only this method takes `system_columns=` (a list of
  `RCT`/`RMT`/`RCB`/`RMB` to **retain**; they are dropped by default).
- **`get_denormalized_as_dict(...)` → `Generator[dict, None, None]`.** A
  **generator**, *not* a list — iterate it (`for row in ...`), don't
  index it. Each `dict` keys on `Table.column` labels. (The full result
  is materialized internally before the first row is yielded — a
  streaming *interface*, not a lower-memory algorithm; audit finding
  SC-07.)
- **`list_denormalized_columns(...)` → `list[tuple[str, str]]`.**
  Model-only, no data fetch: the `(column_name, column_type)` pairs the
  frame *would* have. Runs the same Row-grain / Downstream-leaf /
  Path-ambiguity validation as the real call, so planner errors surface
  early.
- **`describe_denormalized(...)` → `dict`.** A **dry-run** plan with
  **13 keys** that never raises (ambiguities are reported, not thrown):
  `row_per`, `row_per_source`, `row_per_candidates`, `columns`,
  `include_tables`, `via`, `join_path`, `transparent_intermediates`,
  `ambiguities`, `estimated_row_count`, `anchors`, `source`, `warnings`.

`[deriva-ml]`
`src/deriva_ml/dataset/dataset_bag.py::get_denormalized_as_dataframe`
/ `::get_denormalized_as_dict` / `::list_denormalized_columns` /
`::describe_denormalized`; the 13-key shape is detailed in
[the contract §8.3.2](../design/denormalization-contract.md).
`[deriva-ml]`
`src/deriva_ml/model/denormalize_planner.py::DenormalizePlanner`
(route discovery + per-`row_per`-element UNION-distinct emission in
`_prepare_wide_table`); shared by `feature_values`,
`get_denormalized_*`, and `describe`.

## Worked examples

### A single-table denormalization (verified)

The simplest denormalization names one table. On dataset `5D0`'s
downloaded bag (catalog `106`), asking for just `Subject` columns
yields **two rows** — one per subject member of the dataset. Source:
`docs/reference/.examples/denorm.txt`.

```python
bag = ds.download_dataset_bag(version=ds.current_version, use_minid=False)

bag.list_denormalized_columns(["Subject"])
# → [('Subject.RID', 'ermrest_rid'), ('Subject.Name', 'text')]

df = bag.get_denormalized_as_dataframe(["Subject"])
df.shape      # → (2, 2)
list(df.columns)  # → ['Subject.RID', 'Subject.Name']
```

`df` as records (verified, `DENORM_HEAD` in `denorm.txt`):

| Subject.RID | Subject.Name |
|---|---|
| `4CE` | Thing1 |
| `4CG` | Thing2 |

Two members in, two rows out, `row_per` auto-inferred as `Subject`
(the only requested table, so trivially the sink —
[Row grain](#row-grain)). The two columns are exactly `Subject`'s own
columns ([Column projection](#column-projection)); there is no upstream
table to hoist ([Column hoisting](#column-hoisting)).

> The `RID` values `4CE` / `4CG` above come straight from the
> capture file — they are catalog-assigned and opaque. Never read
> meaning into a RID or hard-code one; obtain it from a real lookup.

### A multi-table star-schema flatten (illustrative)

*Illustrative* — described on a small hand-drawn schema; the real demo
catalog wasn't captured for this multi-table shape. With the FK chain
`Subject ◄── Observation ◄── Image` (Image references Observation,
Observation references Subject):

```python
bag.get_denormalized_as_dataframe(["Subject", "Observation", "Image"])
```

auto-infers `row_per=Image` ([Row grain](#row-grain) — Image is furthest
downstream), emitting **one row per image**. Each row carries the
image's own columns plus the `Observation` and `Subject` columns hoisted
from its FK ancestors ([Column hoisting](#column-hoisting)), so a subject
with three images appears three times with its `Subject.*` values
repeated verbatim:

| Subject.RID | Subject.Name | Observation.RID | Image.RID | Image.Filename |
|---|---|---|---|---|
| S1 | Alice | O1 | I1 | a.png |
| S1 | Alice | O1 | I2 | b.png |
| S2 | Bob | O3 | I4 | d.png |

If two FK paths existed between `Image` and `Subject` (e.g. via
`Observation` *and* directly), the planner would raise rather than guess
([Path ambiguity](#path-ambiguity)); adding `via=["Observation"]`
selects the route without adding columns. To reduce a per-image
multi-annotator feature to one row, pass
`selector=FeatureRecord.select_newest` with the single
feature-association table in `include_tables`
([Column hoisting](#column-hoisting)).

## See also

- [Denormalization tutorial](../user-guide/denormalization.md) — the
  gentle, example-led introduction to the six rules (this reference is
  its formal counterpart).
- [Denormalization implementation contract](../design/denormalization-contract.md)
  — the maintainer-facing design contract (state model, fetcher/INSERT
  contracts, fragility map, test matrix).
- [Bag Export](bag-export.md) — **how the bag this page reshapes was
  produced** (anchors, traversal policy, the export spec). Denormalization
  runs *after* export, on the rows the bag already contains.
- [FK Traversal](fk-traversal.md) — the bidirectional FK walk and
  `FKTraversalPolicy` that decide bag membership. A **different**
  operation from denormalization — see the [Mental model](#mental-model).
- `DatasetBag.restructure_assets()` — the sibling local-bag operation
  that reorganizes asset *files* on disk (by dataset type / feature) for
  `ImageFolder`-style trainers, rather than flattening *rows* into a
  frame.
