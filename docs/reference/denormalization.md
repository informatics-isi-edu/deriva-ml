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
| How is the row grain (one-row-per-what) chosen? | [D1](#d1) (`row_per`, default = furthest-downstream table) |
| Why do my `Subject` columns repeat across rows? | [D2](#d2) (upstream hoisting) |
| Which tables contribute columns vs. just a join path? | [D3](#d3) (`include_tables` vs `via`) |
| I have two annotators per image — how do I pick one? | [D4](#d4) (`selector`) |
| My anchors have no FK path — error or skip? | [D5](#d5) (`ignore_unrelated_anchors`) |
| How do I keep `RCB`/`RCT` provenance columns? | [D5](#d5) (`system_columns`, **dataframe only**) |
| Does `_as_dict` return a list I can index? | No — [D6](#d6) (it's a **generator**) |
| My dataset's members are `Subject`s but I read `Image` features — why does it work? | [D7](#d7) (the planner unions FK-reachable routes, not just direct membership) |
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
  *and* FK-reachable chains — unioned RID-distinct; see [D7](#d7).)

The denormalizer is governed by **nine semantic rules** (the canonical
contract list is `Denormalizer`'s class docstring and the
[tutorial](../user-guide/denormalization.md)'s "Rules" section). The
formal rules below — **D1–D6** — group those nine by the knob a caller
actually turns; each cites the underlying rule numbers. **D7** is not a
caller knob — it documents how the planner reaches an element from the
dataset (union of all FK-reachable routes), the behavior that makes
denormalization work on subject-partitioned datasets.

## Formal rules

<a id="d1"></a>
**D1 — `row_per` selects the row grain; the default is the
furthest-downstream requested table.** The output has **one row per
`row_per`-table instance in scope** (tutorial Rule 1). By default
`row_per` is **auto-inferred** (tutorial Rule 2): among the
`include_tables`, the one that no other requested table points to via FK
— the "deepest" / furthest-downstream table — becomes the grain. If two
candidates tie (multi-leaf) or the FK subgraph has a cycle (no sink),
auto-inference raises and you must pass `row_per=` explicitly. Setting
`row_per` to a table that is *downstream* of another requested table is
rejected (tutorial Rule 5: that would require aggregation).
`[deriva-ml]`
`src/deriva_ml/local_db/denormalizer.py::Denormalizer.as_dataframe`
(`row_per` argument; the sink-finding logic is
`src/deriva_ml/model/denormalize_planner.py::DenormalizePlanner._find_sinks`,
invoked through `Denormalizer`); surfaced through each `DatasetBag`
method's `row_per` keyword.

<a id="d2"></a>
**D2 — Upstream columns are hoisted and duplicated across rows that
share an upstream row.** For each `row_per` row, columns from the tables
it references (transitively, up the FK graph) are **repeated verbatim**
across every output row that shares that upstream row (tutorial Rule 4 —
star-schema hoisting). A table reached through an N-to-N link produces
one output row per `(row_per, linked-row)` combination, so the
`row_per` count can grow. Output columns use `Table.column` notation
(e.g. `Subject.Name`).
`[deriva-ml]` `src/deriva_ml/local_db/denormalize.py::_denormalize_impl`
(the SQL join + projection the `Denormalizer` drives).

<a id="d3"></a>
**D3 — `include_tables` names the column sources; `via` disambiguates
the join path.** Columns come **only** from tables named in
`include_tables` (tutorial Rule 3). Tables that appear only as
transitive intermediates — association tables bridging two requested
tables — are joined *through* but contribute **no columns**. When there
are **multiple FK paths** between `row_per` and another requested table,
the planner refuses to guess and raises (tutorial Rule 6); you resolve
the ambiguity by either adding the disambiguating intermediate to
`include_tables` (its columns appear) or to `via` (path-only, **no
columns**). `via` is exactly the "route through this table but don't
project it" knob.
`[deriva-ml]` `src/deriva_ml/local_db/denormalizer.py` (the `via`
argument; ambiguity detection raises
`DerivaMLDenormalizeAmbiguousPath`).

<a id="d4"></a>
**D4 — Feature groups are reduced by a `selector`.** When
`include_tables` contains **exactly one** feature-association table, the
optional `selector` callable reduces each feature's multi-row group
(e.g. several annotators' labels for one image) to a single chosen row.
Its signature is
`Callable[[list[FeatureRecord]], FeatureRecord | None]` — identical to
`DerivaML.feature_values`'s `selector` argument. Built-ins on
`FeatureRecord` include `select_newest`, `select_first`, and
`select_by_workflow`; a custom callable may return `None` to drop the
group. Passing `selector` while `include_tables` does **not** contain
exactly one feature-association table raises `ValueError`. With no
`selector`, every feature row is kept (the group is *not* collapsed),
which multiplies the `row_per` count per [D2](#d2). `selector` is
accepted by `get_denormalized_as_dataframe` and
`get_denormalized_as_dict`; it is **not** a parameter of
`list_denormalized_columns` or `describe_denormalized` (those are
model-only / dry-run).
`[deriva-ml]`
`src/deriva_ml/dataset/dataset_bag.py::get_denormalized_as_dataframe`
(`selector` argument; delegated to `Denormalizer`). See
`src/deriva_ml/feature/__init__.py::FeatureRecord` for the built-in
selectors.

<a id="d5"></a>
**D5 — `ignore_unrelated_anchors` controls unrelated-anchor handling;
`system_columns` (dataframe only) opts audit columns back in.** Two
precise behaviors:

- **`ignore_unrelated_anchors`** (default `False`) — an anchor whose
  table has **no FK path at all** to any table in
  `include_tables ∪ via` raises `DerivaMLDenormalizeUnrelatedAnchor` by
  default (tutorial Rule 8). Set `ignore_unrelated_anchors=True` to
  **silently drop** such anchors instead of raising. (This is distinct
  from an *orphan* anchor — one that *is* upstream of `row_per` but
  reaches no `row_per` row — which always emits a LEFT-JOIN-style row
  with the `row_per`-side columns `NULL`, per tutorial Rule 7;
  `ignore_unrelated_anchors` does not affect orphan emission.) Accepted
  by `get_denormalized_as_dataframe` and `get_denormalized_as_dict`.
- **`system_columns`** (default `None`) — the per-table audit columns
  `RCT` / `RMT` / `RCB` / `RMB` are **dropped by default**. Pass a list
  of any of those four to **retain** them in the output, labeled
  `Table.RCB` like any other column — e.g. `system_columns=["RCB"]`
  keeps each row's creating-user id for a grader join. `system_columns`
  is a parameter of **`get_denormalized_as_dataframe` only**; it is
  **not** present on `get_denormalized_as_dict`,
  `list_denormalized_columns`, or `describe_denormalized`.

`[deriva-ml]`
`src/deriva_ml/dataset/dataset_bag.py` (both keywords on
`get_denormalized_as_dataframe`; `ignore_unrelated_anchors` also on
`get_denormalized_as_dict`).

<a id="d6"></a>
**D6 — Return shapes differ by method.** The four entry points return
distinct types — pick by what you need:

- **`get_denormalized_as_dataframe(...)` → `pandas.DataFrame`.** One row
  per `row_per` instance, `Table.column` headers. Materializes the
  whole frame in memory.
- **`get_denormalized_as_dict(...)` → `Generator[dict, None, None]`.**
  A **generator**, *not* a list — you iterate it (`for row in ...`) and
  cannot index it directly. Each yielded `dict` keys on `Table.column`
  labels with raw Python values. (Note: per the method's own docstring
  / audit finding SC-07, the full result is still materialized
  internally before the first row is yielded, so this is a streaming
  *interface*, not a lower-memory algorithm.)
- **`list_denormalized_columns(...)` → `list[tuple[str, str]]`.**
  Model-only, no data fetch: `(column_name, column_type)` pairs the
  frame *would* have. Runs the same Rule 2/5/6 validation as the real
  call, so planner errors surface early.
- **`describe_denormalized(...)` → `dict`.** A **dry-run** plan with
  **13 keys** that never raises (ambiguities are reported, not thrown):
  `row_per`, `row_per_source`, `row_per_candidates`, `columns`,
  `include_tables`, `via`, `join_path`, `transparent_intermediates`,
  `ambiguities`, `estimated_row_count`, `anchors`, `source`,
  `warnings`. Use it to inspect the resolved grain, join chain, and any
  path ambiguity *before* committing to a real call.

`[deriva-ml]`
`src/deriva_ml/dataset/dataset_bag.py::get_denormalized_as_dataframe`
/ `::get_denormalized_as_dict` / `::list_denormalized_columns` /
`::describe_denormalized`; the 13-key shape is spelled out in
`Denormalizer.describe`'s docstring.

<a id="d7"></a>
**D7 — The planner unions ALL reachable `Dataset → element` routes,
RID-distinct.** To reach an `include_tables` element from the dataset,
the planner does **not** pick a single "best" route. It **unions every**
`Dataset → … → element` path it can find — both the **direct-membership
association** route (`Dataset_{Element}`, e.g. `Dataset_Image`) **and**
**FK-reachable chains** (e.g.
`Dataset → Dataset_Subject → Subject → … → Image`). The unioned result
is **RID-distinct**: SQL `UNION` (not `UNION ALL`) dedups identical
output rows, so when the same element is reachable via several routes it
contributes a single row — for a feature read, exactly one row per
feature-association row. On a **directly-populated** dataset the
membership route is non-empty and the FK routes add no new leaf RIDs (or
duplicate existing ones), so the union yields exactly the same row set as
a membership-only join — directly-populated reads are unchanged. On a
**subject-partitioned** dataset (members are an upstream element like
`Subject`; the target table is reachable **only** via the FK chain, with
an **empty** direct-membership association) this union is the **only**
way the target's rows appear at all — a membership-only join would return
**0**. This is what makes `feature_values` / `get_denormalized_*` work on
subject-partitioned datasets. (Multiple *distinct* FK paths between
`row_per` and a requested table still raise per [D3](#d3) — D7 is about
unioning the membership and FK-reachable routes to the *same* element,
not about silently guessing among ambiguous join paths.) Spec:
`docs/superpowers/specs/2026-06-16-denormalize-fk-reachable-paths-design.md`.
`[deriva-ml]`
`src/deriva_ml/model/denormalize_planner.py::DenormalizePlanner`
(route discovery + UNION-distinct emission in `_prepare_wide_table`);
shared by `feature_values`, `get_denormalized_*`, and `describe`.

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
(the only requested table, so trivially the sink — [D1](#d1)). The two
columns are exactly `Subject`'s own columns ([D3](#d3)); there is no
upstream table to hoist ([D2](#d2)).

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

auto-infers `row_per=Image` ([D1](#d1) — Image is furthest downstream),
emitting **one row per image**. Each row carries the image's own
columns plus the `Observation` and `Subject` columns hoisted from its
FK ancestors ([D2](#d2)), so a subject with three images appears three
times with its `Subject.*` values repeated verbatim:

| Subject.RID | Subject.Name | Observation.RID | Image.RID | Image.Filename |
|---|---|---|---|---|
| S1 | Alice | O1 | I1 | a.png |
| S1 | Alice | O1 | I2 | b.png |
| S2 | Bob | O3 | I4 | d.png |

If two FK paths existed between `Image` and `Subject` (e.g. via
`Observation` *and* directly), the planner would raise rather than guess
([D3](#d3)); adding `via=["Observation"]` selects the route without
adding columns. To reduce a per-image multi-annotator feature to one row,
pass `selector=FeatureRecord.select_newest` with the single
feature-association table in `include_tables` ([D4](#d4)).

## See also

- [Denormalization tutorial](../user-guide/denormalization.md) — the
  gentle, example-led introduction and the canonical nine-rule contract
  (this reference is its formal counterpart).
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
