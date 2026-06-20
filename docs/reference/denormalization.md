# Denormalization Reference

> Verified against a populated demo catalog
> (`create_demo_catalog(create_features=True, create_datasets=True)`,
> deriva-ml v1.46.x, catalog `106`, captured 2026-06-13). Every rule on
> this page is tagged `[deriva-ml]` — the denormalization **rules and
> SQL shaping** are pure local computation inside deriva-ml's own code,
> with no deriva-py engine involvement. (The *shaping* is always local;
> a live `Dataset` additionally **fetches its rows from the catalog**
> into local SQLite first — see [Mental model](#mental-model). A
> `DatasetBag` reads only the bag's local SQLite and touches no
> catalog.) The "Worked
> examples" section pastes verified excerpts from
> `docs/reference/.examples/denorm.txt`.
>
> **For a gentle, example-led introduction**, read the
> [denormalization tutorial](../user-guide/denormalization.md) first.
> This page is the formal counterpart — it states the rules precisely
> and names the code locations, but does not re-teach the concepts.

## What denormalization returns (the goal)

Denormalization flattens a dataset (a downloaded `DatasetBag`, or a live
`Dataset`) into a single **wide table**. Precisely, it returns:

> **One row per `row_per`-table instance that is FK-reachable from the
> dataset's (recursive) members, with the requested columns projected and
> upstream context duplicated across rows that share it. It does no
> aggregation; an explicitly-requested upstream table that reaches no
> `row_per` row contributes a NULL-filled orphan row.**

That is the contract to hold the result against. "One row per ML
observation" is the *intuition* — but the observation is whatever table
becomes `row_per` ([Row grain](#row-grain)): `Image`, `Observation`, a
`file`, or a feature-association row. It is **not** "one row per dataset
member" (scope is FK-reachable, not membership —
[FK-reachable scoping](#fk-reachable-scoping)) and it is **not**
aggregated (N feature rows per image stay N rows unless a `selector`
collapses them — [Column hoisting](#column-hoisting)).

**Scope: deriva-ml catalogs only.** Denormalization is anchored on a
deriva-ml **`Dataset`** — it reads a dataset's members and walks the FK
graph from there. It therefore runs only on a catalog that has the
deriva-ml `Dataset` machinery (e.g. eye-ai). A plain Deriva catalog
without it (e.g. facebase, whose `isa.dataset` is an ordinary domain
table, not the deriva-ml membership model) is **not** a denormalize
target — there is no dataset to anchor the walk. The *rules* below
(grain, hoisting, ambiguity) are general FK-graph logic, but the entry
points require a `Dataset`.

The four entry points are methods on `DatasetBag`:

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

## Terminology

These terms are used precisely and consistently across this page, the
[tutorial](../user-guide/denormalization.md), and the
[contract](../design/denormalization-contract.md):

| Term | Meaning |
|---|---|
| **`row_per` table** / **grain** / **sink** / **leaf** | The table whose rows become output rows — one row per instance of it. "Sink" and "leaf" name the same table from the FK-graph side (it has no outbound FK to another requested table). |
| **upstream / downstream** | Relative to FK direction. In `Subject ◄── Observation ◄── Image` (Image references Observation references Subject), `Image` is **downstream**, `Subject` is **upstream**. A table is downstream of another if you reach it by following FKs *outward* (`A.fk → B.RID`). |
| **anchor** | A dataset member — a specific row of a member table — that scopes the output. The anchor *set* is the dataset's members, taken recursively across nested datasets. (Used interchangeably with "member"; "anchor" is the precise term once it's serving as a scope.) |
| **in scope** | A `row_per` row is in scope if it is **FK-reachable from some anchor**, in either direction — not necessarily a direct dataset member. |
| **transparent intermediate** | An association table on a join path between two requested tables. It is joined *through* but contributes **no columns** (a "bridge"). Feature-association tables are transparent intermediates too. |
| **orphan row** | A row produced for an anchor that is in `include_tables` but reaches no `row_per` row: its own columns are populated, the `row_per`-side columns are `NULL` (a LEFT-JOIN-shaped row). |
| **route** vs **path** | A **route** is a `Dataset → … → element` chain (one of possibly several ways the dataset reaches an element); routes to the *same* element are unioned ([FK-reachable scoping](#fk-reachable-scoping)). A **path** is a `row_per → … → table` FK chain between two requested tables; *distinct* paths to the same table are an error ([Path ambiguity](#path-ambiguity)). They are different concepts — see each rule. |

## Formal rules

<a id="row-grain"></a>
### Row grain

**The output has one row per `row_per`-table instance in scope;
`row_per` defaults to the furthest-downstream requested table.**

"In scope" means **FK-reachable from a dataset member**, not *member-of*
(see [FK-reachable scoping](#fk-reachable-scoping)).

**Auto-inference (the default).** Build a directed graph over
`include_tables ∪ via` whose edges are **direct** FK references between
domain tables (`A.fk_col → B.RID`). **Read the arrow as "A holds the FK,
so A is *downstream* of B"** — the edge points from the table that *has*
the foreign key toward the table it *references*. (e.g. `Image.Subject →
Subject.RID` means `Image` is downstream of `Subject`; in
`Subject ◄ Observation ◄ Image`, `Image` is the most-downstream table.)
Association tables — M:N bridges and feature-association tables — are
**not** edges here: two tables linked only *through* an association
establish no downstream direction between them (the bridge is a 1:1:1
hop, not a fan-out). The **sinks** of this graph (tables with **no
outbound edge** — i.e. they hold no FK to another requested table, so
nothing is downstream of them) are the `row_per` candidates,
restricted to `include_tables` — a `via` table participates in the graph
but can never be the grain, since its columns aren't projected. Then:

- **exactly one sink** → that table is `row_per`;
- **two or more sinks** → `DerivaMLDenormalizeMultiLeaf`. The requested
  tables aren't on one FK chain. Either pass `row_per=` to pick a grain,
  **or add a connecting table** so they form a chain — the exception's
  `bridge_suggestions` field (and message) names the intermediate table(s)
  on a path between the candidates, when one exists. (e.g. on eye-ai,
  `["Subject", "Clinical_Records"]` suggests adding `Observation` /
  `Clinical_Records_Observation` — the tables that bridge them.) Empty when
  the candidates share no path.
- **no sink** (an FK cycle) → `DerivaMLDenormalizeNoSink`.

**Explicit `row_per`.** Must name a table in `include_tables` (it has no
meaning if its columns aren't in the output) — otherwise `ValueError`.
Setting it to a table *downstream* of another requested table is
rejected — see [Downstream-leaf rejection](#downstream-leaf-rejection).
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

**"Transparent" is a topology heuristic, not a purity judgment — and you
can override it.** A table is treated as a transparent intermediate when
it has **exactly two domain FKs** (`_is_topological_association`) — *not*
when it is "pure." This is the right default (it skips `Dataset_Image`-style
link tables), but it has a real-catalog edge: a two-FK table that *also*
carries meaningful columns (a `Role`, `Ordinal`, `Comment`, or — on
catalogs like **facebase** — a genuine domain entity that happens to have
two FKs) is **still treated as transparent and its columns are dropped**
unless you opt in. **The escape hatch:** name that table explicitly in
`include_tables` and its columns appear (it loses transparent status for
that call). So the rule is: *two-FK tables are hidden by default; request
them by name to project their columns.* (Feature-association tables are a
separate, marker-based case — one FK to `Feature_Name` + one to
`Execution` — and are always transparent unless named; see the contract.)
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

**Join type follows FK nullability.** Each hop to an upstream table is an
`INNER JOIN` when the FK column is `NOT NULL` and a `LEFT OUTER JOIN`
when the FK column is nullable. This matters: a `row_per` row with a
`NULL` FK survives the join (its columns for that upstream table come
back `NULL`) rather than being dropped — so a nullable FK never reduces
the `row_per` row count. An `INNER JOIN` is safe only because the
`NOT NULL` constraint guarantees a matching upstream row.

This is also where a **feature `selector`** acts: when `include_tables`
contains **exactly one** feature-association table, the optional
`selector` callable reduces each feature's multi-row group (e.g. several
annotators' labels for one image) to a single chosen row. The reduction
runs **after the SQL join materializes** — the full N-row-per-anchor
frame is produced first, then grouped by the feature's target RID and
collapsed in Python. (So without a `selector` the result is *not* one
row per `row_per` instance when features fan out; *with* one it is. This
is also why `describe_denormalized`'s pre-run row estimate counts the
un-collapsed rows.) Its signature is
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
determined by its table and reachability.**

**"Reaches a `row_per` row" is direction-agnostic.** An anchor reaches
`row_per` if there is *any* FK chain connecting the anchor's table and
`row_per`, **in either direction** — the anchor either sits upstream of
`row_per` (its columns get hoisted onto the `row_per` rows it points to)
or downstream (it points *at* `row_per`, scoping which `row_per` rows are
in play). Both serve as a filter on the output. A reader who assumes
"reaches" means strictly-downstream-only would wrongly drop the upstream
case. In the table below, **"FK-connected to `row_per`"** means exactly
this either-direction relationship — the anchor sits upstream of
`row_per`, downstream of it, or *is* it. The six cases:

| Case | Anchor table | Reaches a `row_per` row? | Contribution |
|---|---|---|---|
| 1 | `== row_per` | — | one output row (itself, upstream hoisted) |
| 2 | `∈ include_tables`, FK-connected to `row_per` | yes | filter-only (appears via the `row_per` row that hoists or points at it) |
| 3 | `∈ include_tables`, FK-connected to `row_per` | no | one **orphan** row — its columns populated, `row_per`-side columns `NULL` (LEFT-JOIN shape) |
| 4 | `∉ include_tables`, FK-connected to `row_per` | yes | filter-only (no row of its own — its columns aren't projected) |
| 5 | `∉ include_tables`, FK-connected to `row_per` | no | silently dropped (would not affect output) |
| 6 | no FK path to any `include_tables ∪ via` table | — | raises `DerivaMLDenormalizeUnrelatedAnchor` unless `ignore_unrelated_anchors=True` (then silently dropped) |

(An orphan row — case 3 — only arises for an anchor that is *upstream*
of `row_per` and reaches none; a *downstream* anchor that reaches no
`row_per` row simply scopes nothing and falls under case 5. "Connected
but unreached" is the orphan trigger only on the upstream side.)

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

Full method signatures (parameters, defaults, raised exceptions) are
auto-generated from the docstrings in the
[DatasetBag API reference](../api-reference/dataset_bag.md); the
behavioral contracts for each method are in
[the contract §8](../design/denormalization-contract.md), and the 13-key
`describe()` shape in [§8.3.2](../design/denormalization-contract.md).

`[deriva-ml]`
`src/deriva_ml/dataset/dataset_bag.py::get_denormalized_as_dataframe`
/ `::get_denormalized_as_dict` / `::list_denormalized_columns` /
`::describe_denormalized`.
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

!!! note "Illustrative — not captured output"
    The schema and the result table below are **hand-drawn** to show the
    star-schema *shape*; they are not pasted from a real catalog capture
    (unlike the single-table example above, which is verified against demo
    catalog `106`). Treat the row counts and values as schematic, not as
    ground truth — verify against your own schema before relying on them.

With the FK chain `Subject ◄── Observation ◄── Image` (Image references
Observation, Observation references Subject):

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

### Corner cases (each rule, by example)

These illustrate the rules that fire only on specific shapes — what they
look like when you hit them, and how to recover. All are described on the
demo schema (`Subject ◄── Observation ◄── Image`, plus a direct
`Image.Subject` FK that makes Image↔Subject ambiguous).

**FK-reachable scoping — more rows than direct members.** A dataset with
8 `Subject` members and 4 `Image` members:

```python
df = bag.get_denormalized_as_dataframe(["Image"])
len(df)   # → 8, not 4
```

The 4 direct Image members plus the 4 Images reachable through the 8
member Subjects (`Image.Subject`) are all in scope
([FK-reachable scoping](#fk-reachable-scoping)). If you wanted only the
direct members, scope the dataset differently — denormalize always
follows FK-reachability.

**Downstream-leaf rejection — explicit `row_per` upstream of a requested
table.** `Subject` is upstream of `Image`:

```python
bag.get_denormalized_as_dataframe(["Subject", "Image"], row_per="Subject")
# raises DerivaMLDenormalizeDownstreamLeaf:
#   Table Image is downstream of row_per=Subject. One row per Subject
#   would require aggregating multiple Image rows per Subject row —
#   aggregation is not yet supported.
#   Options:
#     • Drop row_per to get one row per Image.
#     • Remove Image from include_tables.
```

Drop `row_per` (auto-infers `Image`, one row per image with `Subject.*`
hoisted) or drop `Image`.

**Path ambiguity — two FK routes to the same table.** `Image` reaches
`Subject` directly *and* via `Observation`:

```python
bag.get_denormalized_as_dataframe(["Image", "Subject"])
# raises DerivaMLDenormalizeAmbiguousPath:
#   Multiple FK paths between Image and Subject:
#     Image → Subject
#     Image → Observation → Subject
#   Options:
#     • Add an intermediate table to include_tables (its columns will be in output).
#     • Add an intermediate table to via= (routing only, no columns).
#     • Narrow the requested set to eliminate one path.
#   Suggested intermediates: Observation
```

Resolve by routing explicitly — `via=["Observation"]` (no Observation
columns) or `include_tables=["Image", "Observation", "Subject"]`
(Observation columns included).

**Anchor disposition case 3 — orphan rows.** A dataset whose members
include a `Subject` with no in-scope `Image`. With `row_per=Image`, that
Subject still appears, as a LEFT-JOIN-shaped orphan row:

| Subject.RID | Subject.Name | Image.RID | Image.Filename |
|---|---|---|---|
| S1 | Alice | I1 | a.png |
| S2 | Bob | `NaN` | `NaN` |

S2 reaches no Image, so its `Image.*` columns are `NaN` but the row is
not dropped ([Anchor disposition](#anchor-disposition) case 3).

**Anchor disposition case 6 — unrelated anchor.** A dataset member whose
table has no FK path to anything requested:

```python
# Dataset has File members; request is Image-only.
bag.get_denormalized_as_dataframe(["Image"])
# raises DerivaMLDenormalizeUnrelatedAnchor (File has no FK path to Image).
bag.get_denormalized_as_dataframe(["Image"], ignore_unrelated_anchors=True)
# OK — the File members are silently dropped.
```

## Known limitations (FK shapes)

The rules above assume the **single-column, RID-targeted FKs** that
DerivaML schemas in active use (CSA, CFDE, GPCR, eye-ai) rely on. Two FK
shapes found on other Deriva catalogs are not fully supported:

- **Composite (multi-column) FKs, catalog-side.** A FK spanning several
  columns — e.g. facebase's `file(biosample, dataset) → biosample(RID,
  dataset)` — is handled on the **bag** path (the local SQL emits a
  multi-column `AND` join) but **not** on the live-`Dataset` (catalog
  fetch) path: `_collect_fk_values` raises `NotImplementedError` rather
  than under-scoping the fetch (RB-08). So `Dataset.get_denormalized_*`
  across a composite FK fails loudly; `DatasetBag.get_denormalized_*` on
  a downloaded bag works. Tracked in
  [issue #327](https://github.com/informatics-isi-edu/deriva-ml/issues/327).

- **Self-referential FKs (a table referencing itself).** A hierarchy like
  `HierNode.Parent_Node → HierNode.RID` is **not** denormalized as a
  parent/child relationship. The planner deduplicates join paths by table
  *name* and labels output columns `Table.column`, so it cannot alias the
  same table in two roles — parent-`HierNode.*` and child-`HierNode.*`
  would collide. Single-table requests (`["HierNode"]`) work and path
  discovery terminates without looping; but you **cannot** project a
  node's parent columns alongside its own in one call. To flatten a
  hierarchy, denormalize the single table and resolve the parent links in
  pandas.

Both are about FK *shape*, not dataset content; a schema with only
single-column RID FKs (the common case) hits neither.

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
