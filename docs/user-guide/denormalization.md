# Denormalization

Denormalization transforms normalized relational data into a single
**wide table** (a pandas DataFrame or equivalent) where each row
contains all related information for one observation, with columns
from related tables joined side-by-side. This is the standard input
format for machine-learning frameworks.

DerivaML's denormalization is designed around one principle:

> **Give me a table where each row is one ML observation, with every
> column I asked for filled in from related rows via FK traversal.**

!!! note "Looking for exact rules?"
    This page is the example-led introduction. For the **formal rules**
    — precise `row_per` / `via` / selector behavior and exact return
    types — see the
    [Denormalization Reference](../reference/denormalization.md). (The
    "FK traversal" here means the local column-hoisting join over a
    downloaded bag, which is a *different* operation from the
    catalog-side [FK traversal](../reference/fk-traversal.md) that
    produces a bag.)

The user names the tables they want columns from; DerivaML decides
the shape of the output from the schema's FK graph.

---

## The mental model

### One row per leaf

If you request tables `[A, B, C]` where `C` has an FK to `B` and `B`
has an FK to `A`, you get one row per `C` instance. `B` and `A`
columns are **hoisted** from `C`'s foreign keys and repeated across
rows that share the same `B` or `A`.

This is **star-schema denormalization**: the table furthest
downstream in the FK chain becomes the `row_per` table — the thing
you get one row of each — and upstream context is duplicated across
rows as needed.

### Example

Schema: `Subject ← Observation ← Image` (Image has FK to
Observation; Observation has FK to Subject).

```python
ds.get_denormalized_as_dataframe(["Subject", "Observation", "Image"])
```

Produces:

| Subject.RID | Subject.Name | Observation.RID | Observation.Date | Image.RID | Image.Filename |
|-------------|--------------|-----------------|------------------|-----------|----------------|
| S1          | Alice        | O1              | 2024-01-01       | I1        | a.png          |
| S1          | Alice        | O1              | 2024-01-01       | I2        | b.png          |
| S1          | Alice        | O2              | 2024-02-01       | I3        | c.png          |
| S2          | Bob          | O3              | 2024-01-15       | I4        | d.png          |

One row per Image. Subject columns repeat across all images for a
subject. Observation columns repeat across all images for an
observation.

### Dataset membership acts as a filter

If you call `ds.get_denormalized_as_dataframe(...)` on a `Dataset`,
the dataset's members scope which rows are in scope. Only rows
reachable from a dataset member via FK appear in the output.

If the dataset's members are Images (say 4 of them), you get 4
rows — one per member Image, with Observations and Subjects hoisted.

Nested datasets (datasets whose members include other datasets) are
followed transparently: members of nested datasets contribute their
own rows, just as if they were direct members of the root dataset.

---

## The API

### On a Dataset or DatasetBag

```python
df = ds.get_denormalized_as_dataframe(
    include_tables=["Subject", "Observation", "Image"]
)
```

Returns a pandas DataFrame. Use `ds.get_denormalized_as_dict(...)`
to iterate the rows as dicts.

The method names follow the same conventions as the existing
`get_table_as_dataframe` / `get_table_as_dict` methods on `Dataset`
and `DatasetBag` — the stem `denormalized` describes the (virtual)
table being retrieved.

### Preview without fetching

```python
ds.list_denormalized_columns(["Subject", "Image"])
# → [("Subject.RID", "text"), ("Subject.Name", "text"),
#    ("Image.RID", "text"), ...]

ds.describe_denormalized(["Subject", "Image"])
# → {
#     "row_per": "Image",
#     "row_per_source": "auto-inferred",
#     "row_per_candidates": ["Image"],
#     "columns": [...],
#     "include_tables": ["Subject", "Image"],
#     "via": [],
#     "join_path": [...],
#     "transparent_intermediates": [],
#     "ambiguities": [],
#     "estimated_row_count": {...},
#     "anchors": {...},
#     "source": "catalog",
#     "warnings": [],
#   }

ds.list_schema_paths()
# → {"member_types": [...], "reachable_tables": {...},
#    "schema_paths": {...}, ...}
```

`describe_denormalized` is the **dry-run** entry point: it never
raises, even if the request would fail at run time. Inspect the
returned dict's `ambiguities` and `warnings` lists to find out
what's wrong before committing to a real call. (See the contract
section below for the full envelope and the dry-run invariant.)

### Arbitrary RID anchors (no dataset required)

When you don't have a Dataset but you do have a set of RIDs you
want to denormalize from, use the `Denormalizer.from_rids`
constructor:

```python
from deriva_ml.local_db.denormalize import Denormalizer

d = Denormalizer.from_rids(
    ["1-ABCD", "1-EFGH"],     # bare RIDs — tables looked up via catalog
    ml=ml_instance,
    dataset_rid="42-DATASET",  # required against a live catalog
)
df = d.as_dataframe(["Subject", "Image"])
```

Or skip the lookup if you already know the tables:

```python
d = Denormalizer.from_rids(
    [("Image", "1-ABCD"), ("Image", "1-EFGH")],
    ml=ml_instance,
    dataset_rid="42-DATASET",
)
```

Mixed forms (bare RIDs and `(table, RID)` tuples in the same
list) are supported.

The `dataset_rid` argument is **required against a live catalog**.
The underlying SQL scopes everything by `Dataset.RID IN
(dataset_rid, ...)`, so without a real dataset RID `from_rids`
would silently return zero rows. Local-only / fixture mode (no
catalog) falls back to using the first anchor as a placeholder
scope, with a warning logged.

For most use cases, prefer the `Denormalizer(dataset)` constructor
on an existing Dataset — `from_rids` is a power-user escape hatch
for flows that don't start from a Dataset object.

### Version-pinned denormalization

Pass `version=` to the `Denormalizer` constructor (or to the sugar
methods on `Dataset`) to fetch from a specific catalog snapshot:

```python
df = ds.get_denormalized_as_dataframe(
    ["Image", "Subject"],
    version=ds.version,  # or any DatasetVersion
)
```

Snapshot resolution follows the same pattern as
`Dataset.list_dataset_members(version=...)`. Bags ignore `version`
because they are already pinned to whatever version they were
built from.

---

## Parameters

### `include_tables: list[str]` — required

The tables whose columns appear in the output. Also determines
which table is `row_per` (see below).

You can pass either table names (e.g. `"Image"`) or feature names
(e.g. `"Image_Classification"`) — feature names are translated to
the underlying feature-association table (e.g.
`"Execution_Image_Image_Classification"`) automatically. This
matches the rest of the DerivaML API (`find_features`,
`feature_values`, `lookup_feature`).

### `row_per: str | None` — optional

Names the table whose rows become output rows (one row per
instance of this table). If omitted, auto-inferred as the unique
"deepest" table in `include_tables` — the one no other requested
table points to via FK.

You only need `row_per` when:

- Auto-inference finds more than one candidate (multi-leaf
  ambiguity).
- You want a non-default anchor for a specific ML task.

### `via: list[str] | None` — optional

Names tables to force into the join chain without contributing
columns to the output. Useful when two paths exist between your
requested tables and you want to specify routing without
cluttering your output columns.

Example: `ds.get_denormalized_as_dataframe(["Image", "Subject"],
via=["Observation"])` routes through Observation but doesn't
include Observation columns in the output.

### `ignore_unrelated_anchors: bool = False`

By default, if an anchor has no FK path to any table in
`include_tables`, DerivaML raises an error. Set `True` to silently
drop such anchors. Useful when you know some dataset member types
are out of scope for this particular denormalization.

---

## Examples by use case

### "I want Subject columns on each Image row" (hoist upward)

```python
ds.get_denormalized_as_dataframe(["Subject", "Image"])
# row_per = Image (auto). Each row = one image, Subject columns repeated.
```

### "I want all diagnoses for each subject"

```python
ds.get_denormalized_as_dataframe(["Subject", "Diagnosis"])
# row_per = Diagnosis (auto — Diagnosis points to Subject).
# One row per diagnosis; Subject columns repeated across all
# diagnoses for a subject. Group by Subject.RID in pandas to
# aggregate.
```

### Diamond path — explicit routing

```python
# Schema has Image→Subject direct AND Image→Observation→Subject.

ds.get_denormalized_as_dataframe(["Image", "Subject"])
# ERROR: multiple FK paths.

# Force the multi-hop with columns included:
ds.get_denormalized_as_dataframe(["Image", "Observation", "Subject"])

# Force the multi-hop without Observation columns:
ds.get_denormalized_as_dataframe(["Image", "Subject"], via=["Observation"])
```

### Feature values on images

```python
ds.get_denormalized_as_dataframe(["Image", "Image_Classification"])
# row_per = Execution_Image_Image_Classification (auto — feature
# association table; points to Image).
# One row per feature observation; Image columns repeated for
# multi-execution images.
```

The feature-name shorthand (`"Image_Classification"`) is resolved
to the underlying feature-association table
(`"Execution_Image_Image_Classification"`) by the same helper that
backs `find_features` / `feature_values`. The full table name
works too.

**Setting `row_per=<target_table>` with a feature shorthand in
`include_tables` is rejected (Rule 5).** The feature-association
table is downstream of the target table, and the denormalizer does
not aggregate. The two intents the caller might have in mind are
spelled differently:

```python
# Intent A: "one row per Image with the feature value projected
# as a column." Pass the value table (vocabulary) directly, not
# the feature-name shorthand. The feature-association table
# becomes a transparent bridge in the join.
ds.get_denormalized_as_dataframe(
    ["Image", "Image_Class"],
    row_per="Image",
)
# OK: one row per Image, Image_Class.Name projected.

# Intent B: "one row per feature observation." Let auto-inference
# pick the feature-association table as row_per (the default
# behavior shown at the top of this section), or name it
# explicitly. Image RIDs repeat across multi-execution annotations.
ds.get_denormalized_as_dataframe(
    ["Image", "Image_Classification"],
    row_per="Execution_Image_Image_Classification",
)
# OK: one row per feature observation.

# Forbidden combination: feature shorthand with row_per=<target>.
ds.get_denormalized_as_dataframe(
    ["Image", "Image_Classification"],
    row_per="Image",
)
# RAISES DerivaMLDenormalizeDownstreamLeaf: the shorthand resolves
# to Execution_Image_Image_Classification, which is downstream of
# Image. To project the feature value, use Intent A's shape.
```

### Heterogeneous dataset with orphan members

```python
# Dataset has Image members AND Subject members
# (some subjects with no images in the dataset)

ds.get_denormalized_as_dataframe(["Subject", "Image"])
# row_per = Image.
# Output = |member Images| rows (Subjects hoisted)
#       + |orphan Subjects| rows (Image columns NULL).
```

### Arbitrary anchors (no dataset)

```python
from deriva_ml.local_db.denormalize import Denormalizer

d = Denormalizer.from_rids(
    ["1-ABCD", "2-WXYZ"],
    ml=ml,
    dataset_rid="42-DATASET",
)
df = d.as_dataframe(["Subject", "Image"])
```

---

## Exploring before denormalizing

When you don't know the schema well, use the exploration methods
in order. Each step is more expensive but more definitive:

```python
# 1. What can I put in include_tables?
info = ds.list_schema_paths()
print(info["member_types"])        # dataset element types
print(info["reachable_tables"])    # FK-reachable tables from each type
print(info["schema_paths"])        # available FK paths between tables

# 2. What columns will I get for this request?
ds.list_denormalized_columns(["Image", "Subject"])

# 3. Will this request succeed? How big will the output be?
plan = ds.describe_denormalized(["Image", "Subject"])
if plan["ambiguities"]:
    # Inspect `ambiguities` to see what to add/change
    ...
if plan["warnings"]:
    # describe never raises — warnings reports anything that was
    # silently swallowed
    for w in plan["warnings"]:
        print(w)
print(plan["estimated_row_count"])

# 4. Run it for real.
df = ds.get_denormalized_as_dataframe(["Image", "Subject"])
```

The workflow: `list_schema_paths` → `list_denormalized_columns` →
`describe_denormalized` → `get_denormalized_as_dataframe`.

---

## Rules (for when you need to reason about edge cases)

DerivaML's denormalizer is governed by **nine semantic Rules**.
This section gives the user-facing summary. The full contract
language (including the planner-level guarantees) is in the
implementation-contract section below.

### Rule 1 — Row cardinality

One output row per **`row_per`-table instance in scope**, plus
one orphan row per unreached anchor (see Rule 7). "In scope"
means reachable from some anchor via the FK graph.

### Rule 2 — Auto-inference of `row_per`

Among the requested tables, the one that no other requested
table points to via FK is `row_per`. Intuition: the "deepest"
table in your request.

If two or more candidates tie (multi-leaf), you get an error
asking you to specify `row_per` explicitly. If the FK subgraph
has a cycle (no sink), you get an error too.

### Rule 3 — Column projection

Columns come from `include_tables`. Tables in `via` are path-only,
no columns. Tables that only appear as transitive intermediates
(e.g., association tables bridging two requested tables) are also
path-only, no columns.

### Rule 4 — Column hoisting

For each `row_per` row, upstream columns are hoisted from the FK
graph and **repeated verbatim** across rows sharing the same
upstream row. This is the classic star-schema denormalization
shape.

Tables reached through an N-to-N link produce one output row per
(row_per, linked-row) combination — the `row_per` count grows
accordingly.

### Rule 5 — Downstream-to-`row_per` is rejected

If you set `row_per` explicitly and another requested table is
downstream of it (i.e., `row_per` points to it via FK), you get
an error. This would require aggregation (collapsing N downstream
rows per `row_per` row), which is a future feature.

Workaround: drop `row_per` and let auto-inference pick the
downstream table, or remove the downstream table from
`include_tables`.

### Rule 6 — Path ambiguity requires resolution

If there are multiple FK paths between `row_per` and another
requested table, you get an error listing the paths. Resolve by:

- Adding an intermediate to `include_tables` (its columns are
  included), or
- Adding an intermediate to `via` (path-only, no columns), or
- Narrowing your request so only one path is valid.

Silent path selection is rejected by design — the result would
be different between the two paths, and DerivaML won't guess.

### Rule 7 — Orphan anchors (LEFT-JOIN semantics)

If an anchor's table is upstream of `row_per` but the anchor
doesn't reach any `row_per` row (no rows of `row_per` type point
back to it), the anchor produces an **orphan row** — its columns
populated, `row_per`-side columns NULL. Upstream FK columns are
still hoisted for the orphan row.

This is the LEFT-OUTER-JOIN interpretation: every anchor
contributes at least one row; unreachable ones contribute with
NULL leaf-side data.

Anchors of types not in `include_tables` are dropped silently
when they have no path to `row_per` (no warning — they wouldn't
contribute anyway). Anchors with no FK path **at all** are
handled by Rule 8.

### Rule 8 — Unrelated anchors

An anchor whose table has no FK path to any table in
`include_tables ∪ via` raises by default. Pass
`ignore_unrelated_anchors=True` to silently drop them.

### Rule 9 — Nested datasets

When constructed from a `Dataset` (or `DatasetBag`), the anchor
set is the dataset's members **recursively**, including all
members of nested datasets. The `Denormalizer.__init__(dataset)`
constructor handles the recursive walk; `from_rids` takes
anchors directly and does no recursion.

Descendant dataset RIDs are also threaded into the SQL filter
(`Dataset.RID IN (root, ...children)`), so members of nested
datasets show up correctly under the root.

> **Note on `version`.** A `version` kwarg pins denormalization
> to a catalog snapshot (see "Version-pinned denormalization"
> above). This was deferred in earlier specs as "Rule 10"; the
> kwarg is implemented for live `Dataset` inputs and ignored for
> bags. There is no separate semantic Rule 10 in the current
> design.

---

## When to reach for `feature_values` vs `Denormalizer`

Both tools produce tabular output from catalog data, but they
answer different questions.

**Use `feature_values(table, feature_name, selector=...)`** when:

- You want the values for one specific feature.
- You want typed `FeatureRecord` objects with provenance
  attributes.
- You want selector reduction (one value per target RID after
  filtering by workflow, annotator, or recency).
- You want to iterate over records and process them in Python
  before converting to a DataFrame.

```python
# feature_values: typed records for one feature, selector-reduced
selector = FeatureRecord.select_by_workflow("Manual_Annotation", container=ml)
labels = {
    rec.Image: rec.Diagnosis_Type
    for rec in ml.feature_values("Image", "Diagnosis", selector=selector)
}
```

**Use `Denormalizer`** when:

- You want a wide DataFrame combining columns from multiple
  tables in a single call.
- You want structural joins (e.g., `Subject → Observation →
  Image`) that are not feature-based.
- You want feature tables to participate as leaves in a
  multi-table join alongside domain tables.
- You want pandas output directly.

```python
# Denormalizer: wide DataFrame joining Subject, Image, and a feature table
df = dataset.get_denormalized_as_dataframe(
    ["Subject", "Image", "Execution_Image_Diagnosis"]
)
# df has Subject columns, Image columns, and Diagnosis columns —
# one row per feature value
```

The same question — "give me the diagnosis for each image, with
subject context" — answered both ways:

```python
# feature_values approach: start from the feature, join subject in Python
records = {r.Image: r.Diagnosis_Type for r in ml.feature_values(
    "Image", "Diagnosis", selector=FeatureRecord.select_newest
)}
# Then join with a subject lookup if needed

# Denormalizer approach: one call, pandas output with all context columns
df = dataset.get_denormalized_as_dataframe(
    ["Subject", "Observation", "Image", "Execution_Image_Diagnosis"]
)
```

Use `feature_values` when you care about provenance, selectors,
or typed access. Use `Denormalizer` when you care about wide
joins and pandas integration.

---

## Known limitations and pitfalls

A handful of failure modes are worth knowing about as a user. Each
is covered in more depth in the contract section below.

### `as_dict` is not streaming

Despite the "memory-efficient" wording in some older docstrings,
`get_denormalized_as_dict(...)` and `Denormalizer.as_dict(...)`
**materialise the full result before yielding any row**. They
expose an iteration interface, not a streaming cursor — peak
memory equals the full result set.

Use `as_dict` when downstream code naturally processes one row at
a time. Don't rely on it to bound memory for very large results.

True streaming is a known gap (audit finding SC-07); see the
contract section §8.2 for the design discussion.

### Stale local cache (cross-process)

The local SQLite cache is a write-through history of past
catalog fetches. It survives the process and is shared across
sessions. This is by design — it's what makes re-running
denormalize fast — but it has one consequence:

- **Server-side inserts between calls are observed correctly**
  (Step 3 picks up new rows).
- **Server-side deletes between calls are NOT observed** — the
  old rows remain in the cache and participate in subsequent
  joins.
- **Server-side updates between calls are NOT observed** for the
  same reason.

In practice Deriva data is largely append-only, so this is rarely
visible. If you need a stable snapshot, use
`download_dataset_bag(version=...)` or pass `version=` to the
denormalizer (see "Version-pinned denormalization" above).

There is no caller-visible signal today that a result was
affected by a stale-cache window. Tracked as audit finding SC-03.

### `from_rids` and the placeholder `dataset_rid` (SC-02 / TC-05)

If you construct a `Denormalizer` via `from_rids` against a live
catalog **without** passing `dataset_rid=`, the call now raises
`ValueError` with a clear message — the underlying SQL scopes by
`Dataset.RID IN (dataset_rid, ...)` and a missing scope produces
silent zero rows.

Local-only / fixture flows (no live catalog) fall back to using
the first anchor's RID as a placeholder scope, with a warning
logged. Pre-populated engines treat the RID as an opaque scoping
key.

### Silent fallback to `source="local"` (RB-05)

When `Denormalizer.__init__` cannot build an `ErmrestPagedClient`
(offline tests, mock catalog, transient auth race), it falls
back to `source="local"` and the live-catalog fetch path is
disabled. A warning is logged and stashed on the Denormalizer;
in this revision, the warning surfaces via
`describe()["warnings"]` as a future improvement (see §8.3.1 in
the contract section).

User-visible effect: a `Denormalizer` constructed in an
auth-race window may return whatever rows are already in the
local engine — possibly zero — rather than fresh data.

---

## Relationship to earlier APIs

This is the only denormalization API. The earlier method names
have been removed. The replacements use the same `get_*_as_*` /
`list_*` / `describe_*` conventions as the existing
`get_table_as_dataframe` and `list_dataset_members` methods on
Dataset and DatasetBag:

| Earlier name | Current name |
|--------------|--------------|
| `denormalize_as_dataframe(include_tables)` | `get_denormalized_as_dataframe(include_tables, ...)` |
| `denormalize_as_dict(include_tables)` | `get_denormalized_as_dict(include_tables, ...)` |
| `denormalize_columns(include_tables)` | `list_denormalized_columns(include_tables, ...)` |
| `denormalize_info(include_tables)` | `describe_denormalized(include_tables, ...)` |

Two behavioral changes from the earlier implementations:

- **Ambiguous FK paths now error** instead of silently picking
  the shortest one. Use `via` or add an intermediate to
  `include_tables` to resolve.
- **Feature tables are fully supported** as leaves (`row_per`).
  Previously they were not handled specially and some requests
  failed.

---

## Related concepts

- [Datasets](datasets.md) — how dataset membership defines the
  anchor set.
- [Defining and using features](features.md) — `feature_values`
  is the other half of the picture.

---

# Implementation contract

The material above is the user-facing guide. The material below
is the authoritative design contract — what the code is required
to do. Audience shifts here from "user understanding the API" to
"engineer or auditor verifying behavior." If the two sections ever
disagree on observable behavior, the contract section is
authoritative; file an issue against the user-facing prose.

This section is the single design reference for engineers
working on `deriva_ml.local_db.denormalize`,
`local_db.denormalizer`, `local_db.paged_fetcher`, and
`model.denormalize_planner`. Findings are tagged inline with
stable SC-NN / TC-NN / RB-NN identifiers used as internal
labels within this document (the original standalone audit
report they came from is no longer maintained separately).

---

## 1. What denormalization is, in one paragraph (engineer-facing restatement)

A user has a `Dataset` (live, on the catalog) or a `DatasetBag`
(downloaded snapshot). They ask: "give me a wide table joining
these N tables in the dataset's scope, with one row per X." The
denormalization pipeline plans the join (which FK chains to
follow, which table is the "leaf" producing one row per output
row, what columns each table contributes), populates the local
SQLite cache from whichever source applies (live catalog,
downloaded bag, or "local" for tests), runs the join in SQL
against the local cache, and returns the rows. The user sees a
pandas DataFrame or a row iterator.

The nine semantic Rules summarised in the user-facing section
are reproduced as Rules 1–9 in §6 below with their full contract
language.

---

## 2. Pipeline architecture

```
┌────────────────────────────────────────────────────────────────────┐
│ Caller                                                             │
├────────────────────────────────────────────────────────────────────┤
│ Dataset.get_denormalized_as_dataframe(...)        ← live catalog   │
│ DatasetBag.get_denormalized_as_dataframe(...)     ← downloaded bag │
└────────────────────────────────────────────────────────────────────┘
              │                                           │
              ▼                                           ▼
       Denormalizer(dataset_like)              Denormalizer(bag)
       source = "catalog"                      source = "local"
       paged_client = ErmrestPagedClient                  │
              │                                           │
              └──────────┬────────────────────────────────┘
                         ▼
                   _denormalize_impl(...)                          ← src/deriva_ml/local_db/denormalize.py
                         │
                         ├─ _prepare_wide_table(...)               ← src/deriva_ml/model/denormalize_planner.py
                         │    pure model code; no I/O
                         │    returns:  join_tables, column_specs, multi_schema
                         │    FK-graph walking delegated to
                         │      deriva.bag.path_walker.SchemaPathWalker
                         │      (shared with deriva-py's CatalogBagBuilder;
                         │       DerivaML layers skip-tables + nested-dataset
                         │       loopback as an edge_filter hook)
                         │
                         ├─ if source == "catalog":
                         │    _populate_from_catalog(...)
                         │      └─ new PagedFetcher(client, engine)         ← single-use; one per call
                         │         └─ _populate_from_catalog_inner(...)
                         │            ├─ fetch_by_rids("deriva-ml:Dataset", rids=[dataset_rids], rid_column="RID")
                         │            └─ for each (path, conditions) in join_tables:
                         │                 for each table_name in path[1:]:
                         │                   _collect_fk_values(...) → (rids_to_fetch, fk_column_on_target)
                         │                   fetcher.fetch_by_rids(qualified, rids_to_fetch,
                         │                                          target_table=target_orm.__table__,
                         │                                          rid_column=fk_column_on_target)
                         │
                         ├─ Step 4: build SQLAlchemy SELECT per element path
                         ├─ Step 5: UNION → execute against engine → rows
                         └─ return DenormalizeResult
```

Three source modes are supported by `_denormalize_impl`:

- **`source="catalog"`** — the live path. A `PagedClient`
  (`ErmrestPagedClient` by default) fetches rows from the
  ERMrest catalog into the workspace's local SQLite. Used by
  `Dataset.get_denormalized_as_dataframe`.
- **`source="local"`** — the bag / unit-test path. Rows are
  already in the engine (a downloaded bag's SQLite files, or a
  test fixture's pre-populated tables). No HTTP. Used by
  `DatasetBag.get_denormalized_as_dataframe`.
- **`source="slice"`** — a slice (a subset of a workspace) has
  been ATTACH'd into the engine and rows are visible that way.
  Used by some bag-style flows.

The SQL emission code (Steps 4 & 5) is identical across the
three modes; only the row-population side differs.

`_denormalize_impl` validates the `source` / `paged_client`
combination at entry: `source="catalog"` with `paged_client=None`
raises `ValueError` before any work is done.

### What "planner is pure; no I/O" actually means

`_prepare_wide_table` takes a `DatasetLike` and a `dataset_rid`
as parameters but **does not call any methods on them** in the
current implementation — anchor enumeration
(`list_dataset_members`) and nested-dataset enumeration
(`list_dataset_children`) happen in the `Denormalizer` layer
above the planner (see §6). "Pure; no I/O" therefore means:

1. **No HTTP**: the planner does not consult ERMrest. It walks
   the in-memory `DerivaModel` only (FK graph, table existence,
   sink/path enumeration).
2. **No engine reads**: the planner does not read from the local
   SQLite engine. It plans purely against schema.
3. **No dataset I/O**: today the planner accepts the `dataset`
   parameter past type-checking but does not invoke methods on
   it. If a future change adds dataset reads (e.g. to specialize
   the plan on dataset state), the planner becomes impure and
   this contract has to be revisited.

The takeaway: callers may safely invoke `_prepare_wide_table`
from a `describe`-style dry-run path on an offline catalog
handle, and the only failure modes are model-level (table not
found, ambiguous path, no sink). SC-08 in the 2026-05-26 audit
named the "no I/O" claim as ambiguous; this section is the
tightened wording.

---

## 3. State, ownership, and lifetime

Read this table carefully. Every denormalize bug we have shipped
has been a state-ownership confusion.

| State | Type | Owner | Lifetime | Visible to |
|---|---|---|---|---|
| `model` (catalog schema + FK graph) | in-memory | `DerivaML` / `DatasetBag` | session | all calls in the session |
| `engine` (SQLAlchemy `Engine` over a per-catalog SQLite directory) | sqlite-backed | `Workspace` | **persists to disk; survives the process** | every call against the same catalog, across processes and sessions |
| `orm_resolver` (table-name → SQLAlchemy ORM class) | in-memory | `LocalSchema` | session | all calls |
| `PagedClient` (`ErmrestPagedClient`) | in-memory | `Denormalizer` instance | one Denormalizer | one denormalize call |
| `PagedFetcher` instance | in-memory | `_populate_from_catalog` | **one call to `_denormalize_impl`** | only that fetcher's own methods |
| `PagedFetcher._counts` (per-table row count memo) | in-memory | `PagedFetcher` instance | one fetcher lifetime (== one `_denormalize_impl` call) | only that fetcher's own cardinality heuristic |
| Local SQLite rows (`Image`, `Dataset`, `Execution_Image_*`, …) | sqlite-backed | engine | **persists to disk; survives the process** | every later call, every later session |

### The two truths to remember

1. **`PagedFetcher` is single-use.** A fresh one is constructed
   every time `_populate_from_catalog` is called. Any cache that
   lives on a `PagedFetcher` instance is empty at the start of
   each denormalize call. **In-memory state on a `PagedFetcher`
   cannot carry information across denormalize calls.**

   The fetcher does carry one piece of state **within** its
   lifetime: the `_counts` dict memoises per-table row counts so
   the cardinality heuristic (`fetch_by_rids_or_predicate`) can
   route without repeated `COUNT(*)` round-trips. The memo is
   populated lazily by the first call that needs the count for a
   given table, and it does *not* invalidate if the catalog
   mutates between two `fetch_*` calls inside the same
   denormalize. That window is rare-but-not-impossible against a
   write-active catalog, and acceptable because the heuristic
   only chooses between two correct strategies — a stale count
   produces suboptimal HTTP, never wrong rows. (SC-04 in the
   2026-05-26 audit named the prior "memoised row counts only"
   phrasing as misleading; this section is the clarification.)

2. **The local SQLite is the only durable, shared state.** It
   accumulates rows from every prior fetch. It persists across
   processes, across DerivaML instances, across days. A
   denormalize call doesn't start from a blank engine; it starts
   from whatever the engine has accumulated.

These two facts together generate every subtle denormalize bug.

---

## 4. The fetcher contract

`PagedFetcher` is a **thin transport adapter** with three jobs:

1. **Fetch.** Given `(table, rids, rid_column)`, issue HTTP
   requests to ERMrest and return rows. The `rid_column` may be
   a primary key (most often `"RID"`) *or* a foreign-key column
   (e.g. `"Image"` on `Execution_Image_<Feature>`). When
   `rid_column` is an FK, the server may return many rows per
   filter value; `PagedFetcher` must not assume one-to-one.
2. **Insert.** Write rows into the engine's local SQLite via
   `_insert_rows`. Handle conflicts cleanly: a row whose RID
   already exists must not crash the insert and must not be
   overwritten.
3. **Count.** Memoise per-table row counts within one fetcher
   lifetime so the cardinality heuristic
   (`fetch_by_rids_or_predicate`) doesn't re-issue `COUNT(*)`
   for repeated visits to the same table.

### What it does NOT do

- **Network dedup based on engine state.** Do not consult the
  local SQLite to decide whether to issue a fetch. The local
  cache is a write-through history of past fetches, not an
  authoritative answer to "is the server's row for this value
  already represented here?" This is the bug-class that produced
  finding **A01** (2026-05-21): v1.37.2 of this library
  hydrated a "seen" set from the engine keyed by the caller's
  `rid_column`, and skipped fetches whenever the FK values
  appeared in the engine — silently dropping all but one
  feature row per anchor.
- **Make idempotency claims via row dedup.** Use the database's
  built-in UNIQUE constraint as the authority. Conflict
  handling belongs at the INSERT, not at the fetch.
- **Outlive a single denormalize call.** No long-lived
  `PagedFetcher` instances exist or are intended to exist.

### Public API summary

```python
PagedFetcher(client: PagedClient, engine: Engine)

  .fetch_by_rids(table, rids, target_table, rid_column="RID",
                 batch_size=DEFAULT_BATCH_SIZE,
                 max_url_bytes=DEFAULT_MAX_URL_BYTES) -> int
  .fetch_predicate(table, predicate, target_table, sort=("RID",),
                   page_size=DEFAULT_PAGE_SIZE) -> int
  .fetch_by_rids_or_predicate(...)             # cardinality heuristic
  .fetched_rids(table, target_table=None) -> set[str]
```

Both `fetch_*` methods return the number of rows actually
inserted (after conflict skipping). `fetched_rids` returns the
set of RIDs in the target_table after the call.

---

## 5. The INSERT contract

`_insert_rows(target_table, rows)` is **the only place that
mutates engine state**. Its contract:

- **Input.** A list of dicts. Each dict has at least the columns
  the target_table declares; extras are silently dropped. Each
  dict has a `"RID"` key (every Deriva-managed table has `RID`
  as its primary key — this is invariant across the Deriva
  catalog schema).
- **Behavior.** For each row: if a row with that RID is already
  in the target_table, **skip** (do not update, do not crash).
  Otherwise insert. Use dialect-native upsert when available
  (`sqlite_insert(...).on_conflict_do_nothing(index_elements=["RID"])`
  for SQLite; equivalent for any other engine added later).
- **Output.** Number of rows actually written (skipped rows do
  not count).
- **Invariant after call.** Every RID from `rows` appears in
  `target_table` exactly once.
- **Idempotent.** Calling twice with the same `rows` yields the
  same engine state and second call returns `0`.

### Missing-RID behavior — engine-enforced

A row whose dict lacks a `"RID"` key (or has `"RID": None`) is a
**programming error** and must surface as an exception. The
implementation does NOT include an explicit `if "RID" not in
row: raise ValueError(...)` guard — instead it relies on the
engine's `NOT NULL` constraint on the `RID` primary key, which
fires as `sqlalchemy.exc.IntegrityError`. This contract is
therefore implemented by the *dialect*, not by the function. It
works today on every supported engine (SQLite, per
`local_db/README.md`); a future engine with different RID-column
constraints (e.g. server-generated RIDs) would silently break
the contract. The dialect coupling is acknowledged here as a
known seam (SC-05 in the 2026-05-26 audit) — if and when a
non-SQLite engine is added, this function gains an explicit
pre-insert guard.

### Foreign-key enforcement is OFF during catalog fetch

`_populate_from_catalog` wraps the entire row-fetch loop in
`_foreign_keys_off` (`PRAGMA foreign_keys = OFF` on every
connection checkout). Reason: the join-path walk legitimately
inserts a referencing row before its referent — `Dataset_Image`
rows arrive before `Image` rows because we read
`Dataset_Image.Image` to find out which `Image` RIDs to fetch.
Real integrity comes from the source ERMrest catalog the rows
arrived from; the local engine is a **transport mirror**, not
the authoritative store. FK enforcement is restored on exit
from the `_foreign_keys_off` context.

This is **stateless** — no cache to hydrate, no per-table set
to track. The database's UNIQUE constraint is the authority.
This is the only sustainable design: any client-side cache that
mirrors the engine's content will eventually disagree with it,
and that disagreement is what we keep tripping on.

---

## 6. The denormalize contract

`Denormalizer(dataset).as_dataframe(include_tables, *, row_per,
via, ignore_unrelated_anchors)`:

- **Input.** A `DatasetLike` (live `Dataset`, `DatasetBag`, or
  test fixture), a list of table names to include, and the four
  semantic-rule knobs documented in §6.2 below.
- **Behavior.**
  1. Resolves source mode from the dataset type (see §6.1).
  2. Plans the join via `_prepare_wide_table` (planner is pure;
     no I/O — see §2). The FK-graph walker is
     `deriva.bag.path_walker.SchemaPathWalker`, shared with
     deriva-py's `CatalogBagBuilder`; DerivaML-specific rules
     (the default `Dataset_Dataset` / `Execution` skip set and
     the nested-dataset loopback guard) plug in via the
     walker's `edge_filter` hook so the generic walker stays
     domain-free.
  3. If `source="catalog"`: populates the local SQLite via
     `_populate_from_catalog`. This step issues catalog fetches
     and must satisfy the **row-completeness invariant**
     (§6.4): when the SQL JOIN runs in Step 4, the local cache
     must contain the union of rows every path's `(table,
     rid_column, rid_set)` tuple would fetch. A naïve dedup
     keyed on table name only violates the invariant when two
     element paths converge on the same table from different
     angles (see §7 row F5).
  4. Emits SQL against the local SQLite, runs it, materialises
     rows.
  5. Returns a `DenormalizeResult` (rows + column metadata).
- **Output row count.** Determined by the nine Rules
  (§6.2). The pipeline is *correct* iff that row count matches
  the server's reality, modulo the freshness caveat below.

### 6.1 Source-mode selection (`Denormalizer.__init__`)

`Denormalizer.__init__(dataset, *, version=None)` derives every
dependency from the `dataset` argument:

- **Live `Dataset`** (has `_ml_instance` pointing at a
  `DerivaML`): `model`, `catalog`, `engine`, and `orm_resolver`
  come from the ML instance's `workspace.local_schema`.
  `source` defaults to `"catalog"` because a `PagedClient`
  (`ErmrestPagedClient`) can be built against the catalog.
- **`DatasetBag` / canned test fixture** (has `model`, `engine`,
  `_orm_resolver` as direct attributes): `source` defaults to
  `"local"`. The engine is assumed pre-populated.

`version` (optional `DatasetVersion | str | None`): when given
AND the dataset is a live `Dataset`, the constructor resolves
the version to a catalog snapshot via the dataset's
`_version_snapshot_catalog` resolver, then builds the
`ErmrestPagedClient` against that snapshot. Member enumeration
(`_anchors_as_dict`) also threads `version` through to
`list_dataset_members` so the anchor set is read from the same
snapshot the SQL join will use. For `DatasetBag` and fixtures
the kwarg is silently ignored — the bag is already pinned to
whatever version it was built from.

**Failure-handling caveats** (silent fallbacks worth knowing):

- If `ErmrestPagedClient` construction raises (offline tests,
  mock catalog, auth not yet established), `__init__` falls
  back to `source="local"` and `paged_client=None`. A `WARNING`
  is logged and an `_init_warning` string is stashed on the
  Denormalizer instance for future surfacing via
  `describe()["warnings"]`. A user re-running denormalize on a
  `Dataset` whose ML instance was constructed before auth
  completed will get whatever rows are in the engine —
  possibly zero. (RB-05 in the 2026-05-26 audit named this
  silent-zero hazard; the logging + stash is the current
  mitigation, with full `describe()["warnings"]` surfacing
  tracked as a future improvement.)
- If snapshot resolution fails (bad version string, history
  lookup error), the constructor **re-raises** — version-pinned
  construction is explicit enough that a silent fallback would
  be worse than the failure.

### 6.2 The nine semantic Rules

Each Rule below has a one-line summary; full contract language
follows. Rule 1, 3, 4 are descriptive (shape rules); Rules 2,
5, 6, 7, 8 are operational (the planner enforces them); Rule 9
governs nested-dataset anchor enumeration.

#### Rule 1: Row cardinality = `|row_per rows in scope|` + `|orphan rows|`

Each output row corresponds to one of:

- A **`row_per` row in scope**: a row of the `row_per` table
  that is reachable from at least one anchor via the FK graph.
- An **orphan row**: a row of some anchor whose table is in
  `include_tables` and upstream of `row_per`, when no `row_per`
  row is reachable from that anchor. Its columns are populated
  from the anchor and its own upstream FKs; `row_per` and
  downstream columns are NULL.

#### Rule 2: Auto-inference of `row_per`

Given `include_tables = {T₁, …, Tₙ}`, build the directed graph
on `include_tables ∪ via` where an edge `A → B` means A has an
outbound FK to B (directly or via non-member association-table
intermediates). Find sinks — tables with no outbound edges in
this subgraph.

- Exactly one sink → `row_per = that sink`.
- Zero sinks → cycle; raise `DerivaMLDenormalizeNoSink`.
- Multiple sinks → multi-leaf ambiguity; raise
  `DerivaMLDenormalizeMultiLeaf` with candidates listed.

#### Rule 3: Column projection

- Columns come **only** from tables in `include_tables`.
- `via` tables contribute to the join path but not to columns.
- Association tables on a necessary chain between two requested
  tables are **transparent intermediates**: joined through,
  columns omitted.
- Non-requested tables that are neither `via` nor transparent
  intermediates are not joined at all.

#### Rule 4: Upstream hoisting (star schema)

For each `row_per` row R:

- Walk outbound FKs from R to every table T in `include_tables`.
- The columns of the reached row of T are repeated verbatim in
  R's output row, labeled with `{T}.{column_name}` prefix.
- If T is reached via an N-to-N link (through an association
  table or M-to-N linking table), one output row is emitted per
  (R, T-row) combination. The `row_per` count grows accordingly
  — the output then has `|row_per × (average T-links per
  row_per)|` rows.

#### Rule 5: Explicit `row_per` with downstream table → error

If `row_per` is explicitly specified and any table in
`include_tables` is **downstream** of `row_per` (i.e., `row_per`
has an outbound FK path to it), raise
`DerivaMLDenormalizeDownstreamLeaf`:

```
Table {T} is downstream of row_per={row_per}. One row per
{row_per} would require aggregating multiple {T} rows per
{row_per} row — aggregation is not yet supported.

Options:
  • Drop row_per to get one row per {T}.
  • Remove {T} from include_tables.
```

#### Rule 6: Multiple FK paths → error

If between `row_per` and another table T in `include_tables ∪
via` there exist two or more distinct FK paths, raise
`DerivaMLDenormalizeAmbiguousPath`:

```
Multiple FK paths between {row_per} and {T}:
  {path1}
  {path2}
  ...

Options:
  • Add an intermediate table to include_tables (its columns will be in output).
  • Add an intermediate table to via= (routing only, no columns).
  • Narrow the requested set to eliminate one path.

Suggested intermediates: {list of candidates}
```

The "suggested intermediates" list is the set of tables that
appear in at least one path but not in `include_tables`.

#### Rule 7: Anchor contribution

Every anchor must contribute to the output in exactly one way,
determined by its table and reachability:

1. **Anchor table == `row_per`**: contributes one output row
   (the anchor row itself, upstream cols hoisted).
2. **Anchor table ∈ `include_tables`, upstream of `row_per`,
   reaches at least one `row_per` row**: filter-only
   contribution (the `row_per` rows reachable from this anchor
   are in scope). Does not produce its own row — that row
   appears via the `row_per` row that hoists this anchor.
3. **Anchor table ∈ `include_tables`, upstream of `row_per`,
   reaches no `row_per` row**: produces one orphan row (Rule 1
   §6.2 Rule 1).
4. **Anchor table ∉ `include_tables`, upstream of `row_per`,
   reaches at least one `row_per` row**: filter-only
   contribution. No orphan row because the anchor's columns
   aren't in the output.
5. **Anchor table ∉ `include_tables`, upstream of `row_per`,
   reaches no `row_per` row**: contributes nothing. Silently
   dropped (would not affect output either way).
6. **Anchor table has no FK path to or from any table in
   `include_tables ∪ via`**: raise
   `DerivaMLDenormalizeUnrelatedAnchor` unless
   `ignore_unrelated_anchors=True` (Rule 8).

Empty anchor sets (e.g. `{"File": []}` returned by
`list_dataset_members` for an association table whose row
count is zero) are skipped entirely and never trigger Rule 8.

After the main SQL join runs, `_run` performs a **per-RID
orphan scan** (Step 4a) to catch upstream scoping anchors whose
specific RIDs didn't appear in the main result — a Subject
whose Image set is empty, for example. Those RIDs are added to
the orphan set and emitted via `_emit_orphan_rows`, producing
LEFT-JOIN-shaped output rows with the row_per-side columns set
to `None`.

#### Rule 8: Unrelated anchors

If any anchor falls under Rule 7 case 6 and
`ignore_unrelated_anchors=False` (the default), raise
`DerivaMLDenormalizeUnrelatedAnchor`:

```
Anchors of table {T} have no FK path to any table in
include_tables={list}. They will not affect the output.

Options:
  • Remove these anchors from the anchor set.
  • Add {T} (or a linking table) to include_tables.
  • Pass ignore_unrelated_anchors=True to silently drop them.
```

When `ignore_unrelated_anchors=True`, unrelated anchors are
silently ignored (they contribute no row and no filter).

Anchors of Rule 7 case 5 (table ∉ include_tables, unreachable)
are silently dropped regardless of the flag — they contribute
no output either way, so there's nothing to warn about.

#### Rule 9: Nested datasets

When constructed from a `Dataset`/`DatasetBag`, the anchor set
is the dataset's members **recursively**, including all
members of nested datasets. This matches current
`Dataset.list_dataset_members(recurse=True)` behavior. The
`Denormalizer.__init__(dataset)` constructor is responsible
for materializing the recursive member list; `from_rids` takes
anchors directly and does no recursion.

`_run` also pulls **descendant dataset RIDs** via
`dataset.list_dataset_children(recurse=True)` when the dataset
supports it. These RIDs are passed to `_denormalize_impl` as
`dataset_children_rids` and end up in the SQL `WHERE
Dataset.RID IN (dataset_rid, ...children)` clause. Without
them, members of nested datasets (whose `Dataset_X.Dataset`
points at a descendant rather than the root) never pass the
filter and the result comes back empty. If
`list_dataset_children` is not implemented or raises, the
helper falls back to root-only scoping silently —
fixture-shaped datasets often don't implement it. (RB-06 in
the 2026-05-26 audit flags this silent-fallback against
transient network errors as a known robustness gap.)

> **On Rule 10 (`version`).** Earlier drafts treated `version`
> as a deferred Rule 10. The current implementation accepts a
> `version` kwarg on `Denormalizer.__init__` (and the
> Dataset-side sugar methods) and resolves it to a catalog
> snapshot via `_version_snapshot_catalog`. There is no
> separate semantic Rule 10 in this design — version-pinning
> is a constructor-level concern, not a Rule.

### 6.3 Anchor classification, nested datasets, and the SQL filter

The Denormalizer's `_run` method partitions the anchor set
(`_anchors_as_dict`, derived from
`dataset.list_dataset_members(recurse=True, version=...)`)
into three buckets via `_classify_anchors`:

- **Scoping** — anchors whose RIDs filter the row_per side.
  Cases 1, 2, 4 in Rule 7 above.
- **Orphan** — case 3: anchor is in include_tables but has no
  FK path to row_per. Emits LEFT-JOIN-shaped rows via
  `_emit_orphan_rows`.
- **Ignored** — case 5 (anchor connected to subgraph but
  doesn't reach row_per; silent drop) and case 6 with the
  `ignore_unrelated_anchors` flag set (no FK path at all;
  would otherwise raise `DerivaMLDenormalizeUnrelatedAnchor`).

### 6.4 The row-completeness invariant (§6 step 3 in detail)

The fetch step must leave the local cache in a state where the
SQL JOIN in Step 4 returns every row a *fully-fetched*
execution against the server would return. Concretely:

> For every distinct `(table, rid_column, rid_set)` tuple that
> the planner emits across the join paths, the local cache
> must contain the rows the server would return for that
> tuple.

This is the **row-completeness invariant**. It implies — but
is strictly stronger than — "every table appears in the
cache." Two paths can target the same table with different
`rid_column`s or different RID sets; the cache must hold the
**union** of all rows those parametrizations would fetch, not
just one path's worth.

The current implementation enforces the invariant by
deduplicating fetches on the full tuple `(table_name,
rid_column_on_target, frozenset(rids_to_fetch))`. Only
**true** duplicates (identical table, identical rid column,
identical rid set) are skipped; every distinct parametrization
fires its own catalog fetch. Until 2026-05-26 the dedup key
was the table name only, which only *coincidentally* satisfied
the invariant under the planner's current output. See §7 row
F5 for the full incident write-up.

### 6.5 Freshness caveat

The local SQLite is a **write-through cache of past fetches**,
not a live view of the server. If the catalog mutates between
two denormalize calls in the same process, the engine still
holds the old rows in addition to whatever the new fetch adds.
Specifically:

- **Insertions on the server are observed correctly.** Step
  3's fetch picks up new rows and `_insert_rows` adds them.
- **Deletions on the server are NOT observed.** A row deleted
  server-side after we cached it remains in the engine and
  participates in subsequent JOINs.
- **Updates on the server are NOT observed.** Same reason.

This is a real design limitation, not a bug. It is acceptable
because (a) Deriva data is largely append-only in practice,
(b) the cache is per-catalog-snapshot conceptually (callers
who need a snapshot should `download_dataset_bag(version=...)`
or pass `version=` to the Denormalizer), and (c) the fix is
invasive (we'd need invalidation hooks tied to every server
mutation).

Documented; tracked as a known limitation. The deletion/update
regression test promised in §9 (row C.5x, marked `live,
xfail`) is **planned but not yet written** — searching the
test suite as of 2026-05-26 finds no matching `xfail` (TC-04
in the audit).

There is also no caller-visible signal that a given result
was affected by a stale-cache window — no last-fetch
timestamp, no warning on `DenormalizeResult`, no `reason`
field. (SC-03 in the 2026-05-26 audit named this gap;
surfacing it is a future contract change.)

---

## 7. Fragility map — known bug patterns and how the contract prevents them

| # | Pattern | Cause | Status |
|---|---|---|---|
| **F1** | Re-denormalize crashes with `UNIQUE constraint failed: Dataset.RID` | Plain `INSERT` against engine that has prior rows. | Fixed by the INSERT-OR-IGNORE contract in §5. Originally surfaced as 2026-05-21 finding 05 in the model-template e2e run. |
| **F2** | Re-denormalize silently drops rows | `_get_seen` (v1.37.2) hydrated a dedup map from the engine keyed by the caller's `rid_column`. For FK columns with N rows per value, this collapsed the fetch to one row per FK. | Fixed by removing the engine-hydrated seen-set entirely. The fetcher does not dedup based on engine state; conflict handling belongs at INSERT. Originally surfaced as 2026-05-21 finding A01. |
| F3 | Stale local data when server mutates between calls (deletes, updates) | Cache is write-through, never invalidated. | Documented as a known limitation. See §6.5 freshness caveat. The promised `xfail` regression test (row C.5x in §9) is planned but unwritten as of 2026-05-26 (TC-04 in the audit). |
| F4 | `_collect_fk_values` walks "values currently present in the engine" to decide what to fetch from the server. If a parent table's membership was updated server-side after the engine cached it, downstream fetches use the stale parent set. | Same root cause as F3. | Same status — known limitation, tracked. |
| **F5** | Path-walk order silently determines which rows get loaded when two element tables share intermediate tables | The contract (§6 step 3) requires the local cache to contain the union of rows every path's `(table, rid_column, rids)` tuple would fetch — the *row-completeness invariant*. Until 2026-05-26 `_populate_from_catalog_inner` keyed its `processed` set on table name only, which only *coincidentally* satisfied the invariant under the current planner's output. A future planner change (new element type, FK refactor, split datasets) could silently break the invariant without any code in `_populate_from_catalog_inner` changing. | **Fixed** (2026-05-26): the dedup key is now `(table_name, rid_column, frozenset(rids))`, which implements the invariant directly — each distinct parametrization fires its own fetch; only true duplicates are deduped. Regression test pinned at §8 row D.3 / C.8. Originally surfaced as 2026-05-26 audit finding SC-06 / RB-02. |
| **F6** | `describe()` / `preflight_count` reports `estimated_row_count.total = 0` while the actual fetch returns rows | The estimator counted anchors whose table literally equals `row_per`. When `row_per` is downstream of the anchor table (the common feature-table case), no anchor matches and the sum is silently 0. Mathematically the cardinality is N rows per anchor for an unknown-from-anchor-data N. | Fixed by honest "unknown" semantics: when anchors are downstream of `row_per`, `in_scope_row_per_rows` and `total` return `None` and a `reason` field tells the caller why. The case-1 path (anchor == row_per) still returns an exact integer. Originally surfaced as 2026-05-21 finding A02 (Analyst arc). |

---

## 8. The public Denormalizer surface

`Denormalizer` (in `src/deriva_ml/local_db/denormalizer.py`) is
the contracted public class. Every method below has a stable
input/output shape; the docstrings on the implementation carry
the parameter-level detail. This section pins the **behavioral
contracts** — what is guaranteed, what can fail, and how the
methods relate to each other.

### 8.1 Constructors

#### `Denormalizer(dataset, *, version=None)`

Construct from a `DatasetLike`. Source mode and dependencies are
derived from the dataset's shape (§6.1). `version` snapshots
the catalog for live `Dataset` inputs and is ignored for
`DatasetBag`.

#### `Denormalizer.from_rids(anchors, *, ml=None, dataset_rid=None, ...)`

Construct from an explicit RID anchor set without a `Dataset`
context. Anchors may be bare RIDs (table looked up via catalog)
or `(table_name, RID)` tuples. Mixed forms supported. Bare-RID
lookup prefers `ml.resolve_rids` (batched, O(tables)
round-trips) over `catalog.resolve_rid` (per-RID, O(N)
round-trips).

**`dataset_rid` is required against a live catalog.** The
underlying `_denormalize_impl` scopes its SQL via `Dataset.RID
IN (dataset_rid, ...)`. `from_rids` therefore needs a real
dataset RID. When the caller omits `dataset_rid`:

- **Live catalog** (an `ErmrestPagedClient` can be wrapped
  around the given `catalog`): the constructor raises
  `ValueError` with a message explaining that
  `dataset_rid=<existing dataset RID>` must be passed, or
  suggesting `Denormalizer(dataset)` if the caller has a
  `Dataset` in hand. This guards against the SC-02 / TC-05
  silent-zero failure mode.
- **Local / fixture mode** (no catalog, or a mock that can't be
  wrapped): the constructor falls back to using the first
  anchor's RID as a placeholder scope and **logs a warning**.
  This keeps fixture-shaped flows working where the engine has
  been pre-populated with arbitrary RIDs.

`from_rids` always sets `source="local"` and
`paged_client=None`. Callers who want catalog-side fetching
must pre-populate the engine or use the
`Denormalizer(dataset)` constructor with a real `Dataset`.

**Raises:** `ValueError` for missing `model`/`ml`, unresolvable
bare RIDs, missing catalog for lookup, malformed `(table, RID)`
tuples (arity ≠ 2), or `dataset_rid=None` against a live
catalog.

### 8.2 Materialization methods

#### `as_dataframe(include_tables, *, row_per=None, via=None, ignore_unrelated_anchors=False) -> pandas.DataFrame`

Run the full 4-phase pipeline (planner decisions → anchor
classification → main SQL join → orphan-row combine) and
return a DataFrame. One row per `row_per` instance in scope,
plus any orphan rows whose `row_per`-side columns are `NaN`.

**Raises:**
`DerivaMLDenormalizeMultiLeaf`, `DerivaMLDenormalizeNoSink`,
`DerivaMLDenormalizeDownstreamLeaf`,
`DerivaMLDenormalizeAmbiguousPath`,
`DerivaMLDenormalizeUnrelatedAnchor` — the semantic-rule
exceptions. Also `DerivaMLTableNotFound` and
`DerivaMLDenormalizeError` from feature-name resolution
(§8.4). See §6.2 for per-Rule details.

#### `as_dict(include_tables, *, row_per=None, via=None, ignore_unrelated_anchors=False) -> Generator[dict, None, None]`

Same planner, same rules, same exceptions as `as_dataframe`.
Yields one `dict[str, Any]` per row keyed by `Table.column` /
`schema.Table.column` labels.

**Materialization, not streaming.** The implementation builds
the full row list in `_denormalize_impl` before any row is
yielded (rows are eagerly materialised by
`session.execute(final_query)` into a Python list). `as_dict`
then yields from that list via `DenormalizeResult.iter_rows`.
There is no streaming path today. The "yields one at a time"
interface is preserved (the caller's loop iterates rows one by
one) but peak memory equals the full result set.

This is acknowledged as a known gap (SC-07 in the 2026-05-26
audit). Treat `as_dict` as "iteration interface, materialised
internals." True streaming (replacing the materialise step
with a cursor-driven generator) is tracked as future work.

**Raises:** same as `as_dataframe`. All planner validation
runs before any row is yielded (the pipeline materialises up
front), so exceptions surface on the first `next()`.

### 8.3 Inspection methods

#### `columns(include_tables, *, row_per=None, via=None) -> list[tuple[str, str]]`

Preview the `(column_name, column_type)` pairs the matching
`as_dataframe` call would produce. Planner-only — no data
fetch, no catalog query, no anchor classification. Useful as a
cheap validator of `include_tables` before committing to a
full run.

**Raises:** same planner-rule exceptions as `as_dataframe`
(Rules 2/5/6) plus the resolver exceptions from §8.4. Rule 7
and Rule 8 do NOT fire here — anchor classification runs only
when rows are materialised.

#### `describe(include_tables, *, row_per=None, via=None) -> dict[str, Any]`

**The dry-run inspection method.** Returns a 13-key dict
describing what a corresponding `as_dataframe` call would do,
**without raising**. Every failure mode (planner-rule
violation, catalog access error, network timeout, ambiguous
resolution) is **swallowed** and represented in the returned
dict as `None` / `[]` / `{}` in the affected positions, with
an entry appended to the `warnings` list explaining what was
swallowed. Ambiguities are reported in the `ambiguities` list
so the caller can inspect before committing to a real call.

##### 8.3.1 The dry-run invariant

`describe()` **never raises**. This is a contract, not an
implementation detail — callers may safely wrap a `describe()`
call in code that does not handle exceptions, knowing that any
failure inside the planner / catalog / schema stack collapses
to a well-formed dict with sensible defaults.

The invariant is enforced by wrapping every internal call
(planner hooks, anchor enumeration, ambiguity finder,
column-spec builder, classifier) in a broad `try/except`. The
**information-preservation invariant** complements the dry-run
invariant: every broad-except site appends a one-line
diagnostic to `plan["warnings"]` naming the failing call, what
defaulted, and the exception type + truncated message. This
closes SC-01 / RB-01 / TC-09 in the 2026-05-26 audit — callers
can distinguish "genuine empty result" from "swallowed
failure" by inspecting `plan["warnings"]`.

##### 8.3.2 The 13-key return shape

The returned dict has these 13 keys. Keys are present in
**every** call (never omitted), even if their value is the
empty default.

| Key | Type | Meaning when populated | Meaning when empty |
|---|---|---|---|
| `row_per` | `str \| None` | Resolved leaf table name. | `None` — planner couldn't resolve (multi-leaf, no sink, bad explicit value). |
| `row_per_source` | `str` | `"explicit"` if the caller passed `row_per`, else `"auto-inferred"`. | Always present. |
| `row_per_candidates` | `list[str]` | Sink tables from Rule 2 sink-finding (what auto-inference considered). | `[]` — sink-finding raised. |
| `columns` | `list[tuple[str, str]]` | `(name, type)` pairs `as_dataframe` would produce. | `[]` — planner raised before columns could be computed. |
| `include_tables` | `list[str]` | Echo of the caller's input (post feature-name resolution; see §8.4). On resolver failure, falls back to the original input. | Always populated. |
| `via` | `list[str]` | Echo of the caller's input. | `[]` if not supplied. |
| `join_path` | `list[str]` | Ordered table names on the join chain (excludes `Dataset` root). | `[]` — no element tables resolved. |
| `transparent_intermediates` | `list[str]` | Subset of `join_path` not named in `include_tables` (joined through but not projected). | `[]` if no transparent tables. |
| `ambiguities` | `list[dict]` | Per-Rule-6 entries: `{type, from, to, paths, suggestions}`. | `[]` if plan is unambiguous OR if ambiguity detection raised. |
| `estimated_row_count` | `dict` | `{in_scope_row_per_rows, orphan_rows, total[, reason]}`. See §8.3.3. | All three counts `None` if classification raised. |
| `anchors` | `dict` | `{total, by_type}` — RID counts grouped by table. | `{total: 0, by_type: {}}` if anchor enumeration raised. |
| `source` | `str` | `"catalog"` for live Datasets, `"local"` for bags/fixtures, `"slice"` for attached slices. | Always populated. |
| `warnings` | `list[str]` | One short human-readable entry per swallowed exception. Each entry names the failing call, what defaulted, and the exception type + truncated message. | `[]` — describe ran clean. |

##### 8.3.3 `estimated_row_count` semantics (the A02 fix)

Three cases drive the row-count estimate:

1. **Anchor table == row_per** → exact: 1 row per anchor.
   `in_scope_row_per_rows = anchor_count`, `total =
   in_scope_row_per_rows + orphan_rows`. No `reason` key.
2. **Anchor table reaches row_per via FK chain (downstream or
   upstream) but is NOT row_per itself** → unknown without a
   catalog query. `in_scope_row_per_rows = None`, `total =
   None`, and a `reason` string names the downstream anchor
   tables so the caller knows why. `orphan_rows` is still
   exact (orphans contribute exactly one row regardless of
   row_per cardinality).
3. **Anchor table has no FK path to row_per** → orphan (Rule
   7 case 3). Contributes to `orphan_rows` (exact), not to
   `in_scope_row_per_rows`.

Mixed case-1 and case-2 anchors collapse to the case-2
behavior: `None` with a `reason`. This is the honest answer —
we can't sum a known integer with an unknown integer.

The pre-A02 implementation only honored case 1 and silently
returned 0 for case 2, which produced false "0 rows" estimates
for every feature-table denormalize (the common case). See §7
row F6 for the incident.

##### 8.3.4 The describe-vs-run agreement contract

`describe(include_tables=X)` and `as_dataframe(include_tables=X)`
must agree on whether `X` is a valid input. If `describe`
accepts a name and returns a plan, the matching `as_dataframe`
call must either succeed with a result of the shape `describe`
predicted or raise on a semantic-rule violation `describe`
reported in `ambiguities` — never raise on an input-validation
failure `describe` silently accepted.

The validation is shared via the `_resolve_table_names`
helper (introduced by PR #228, the analyst/01 fix; described
in §8.4). Both paths now recognize **feature names** (e.g.
`"Image_Classification"`) as shorthand for the underlying
feature-association table (e.g.
`"Execution_Image_Image_Classification"`).

The current test suite pins this contract at one place — the
`{name for name, _ in plan["columns"]} == set(df.columns)`
check in
`TestFeatureNameResolution::test_describe_and_run_agree`.
**Every other key in the 13-key envelope has the same
analyst/01-shaped asymmetry risk and no parity test today**
(TC-02 in the audit). Future work: a parity test per key.

#### `list_paths(tables=None) -> dict[str, Any]`

Describe the FK graph reachable from the anchor set. Model-only
analysis: no catalog query, no data fetch. Useful for schema
exploration — answers "what tables could I reasonably include
in `include_tables` given my anchor set?"

Returns a 6-key dict:

- `member_types`: dataset element types (same as `anchor_types`
  for `from_rids`-constructed Denormalizers).
- `anchor_types`: sorted list of distinct anchor table names.
- `reachable_tables`: `{anchor_table: [reachable downstream
  tables, sorted]}`.
- `association_tables`: pure M:N association tables in the
  schema.
- `feature_tables`: feature tables discovered via
  `DerivaModel.find_features`. Empty if the model doesn't
  expose `find_features` or has no features.
- `schema_paths`: `{(source, target): [{path, direct}]}` — FK
  paths between reachable pairs, with a `direct` flag for
  single-hop paths.

Failures inside the model walk (e.g. catalog unreachable when
inspecting `model.schemas`) collapse to empty defaults in the
affected keys — same dry-run posture as `describe`.

### 8.4 Feature-name resolution (`_resolve_table_names`)

`describe`, `as_dataframe`, `as_dict`, and `columns` all share
the same `include_tables` / `via` / `row_per` validation via
the private `_resolve_table_names` helper (introduced by PR
#228). The helper translates **feature names** (e.g.
`"Image_Classification"`) into the underlying
**feature-association table name** (e.g.
`"Execution_Image_Image_Classification"`) so callers can pass
either form symmetrically with the rest of the DerivaML API
(`find_features`, `feature_values`, `lookup_feature`).

Resolution algorithm for each input name:

1. If `model.name_to_table(t)` succeeds → keep `t` as-is.
2. Otherwise consult `model.find_features()`. If exactly one
   feature has `feature_name == t` → substitute with
   `feature.feature_table.name`. If multiple matches across
   different target tables → raise `DerivaMLDenormalizeError`
   (ambiguous — caller must name the feature-association
   table directly).
3. If neither path works → raise `DerivaMLTableNotFound`. The
   error message includes the list of known feature names so
   the user can tell whether they typo'd a feature or pointed
   at the wrong table.

On the **dry-run path** (`describe`), the entire resolver call
is wrapped in `try/except Exception` to preserve the dry-run
invariant (§8.3.1). On resolver failure, the original input is
passed through, a `warnings` entry is appended, and the
downstream planner calls produce their own empty fields.

On the **run path** (`as_dataframe` / `as_dict` / `columns`),
the resolver raises immediately on ambiguous or unknown names,
so the user gets a clear error before any planner work fires.

The contract: **if `describe(X)` accepts X, the matching run
path accepts X too.** (§8.3.4.)

---

## 9. Test matrix

The matrix below pins the contracts in §4–§8 against concrete
pytest cases. Each row is one test. Cases against a live
catalog are marked `live`; cases against an in-memory engine +
`FakePagedClient` are marked `unit`.

### Layer A — `_insert_rows` (the heart of the new contract)

| # | Scenario | Kind | Assert |
|---|---|---|---|
| A.1 | Insert N rows into empty target | unit | engine has N rows; returned count == N |
| A.2 | Insert overlapping rows twice into one target | unit | engine has N rows; 2nd call returns 0 |
| A.3 | Insert rows where some RIDs already exist | unit | engine = pre ∪ new; returned count == \|new only\| |
| A.4 | Insert rows missing RID — pin behavior | unit | raise `IntegrityError` (engine NOT NULL constraint; see §5 missing-RID note) |
| A.5 | Insert rows with extra columns | unit | extras silently dropped (existing contract) |
| A.6 | Two `PagedFetcher` instances inserting overlapping rows into one engine | unit | no crash; engine = union |
| A.7 | Fresh fetcher against pre-populated engine (cross-session simulation) | unit | no crash; existing rows preserved |
| A.8 | Engine has rows with `RID=NULL` (defensive) | unit | treated as never-seen; insert proceeds (or raise — pin one) |

### Layer B — `fetch_by_rids` / `fetch_predicate`

| # | Scenario | Kind | Assert |
|---|---|---|---|
| B.1 | `fetch_by_rids(rid_column="RID")`, fresh engine | unit | all rows fetched and inserted |
| B.2 | Same call repeated within session | unit | no crash; engine final state correct |
| B.3 | Same call against pre-populated engine | unit | no crash; new RIDs inserted; existing untouched |
| B.4 | **`fetch_by_rids(rid_column=<FK>)`, multiple rows per FK value** | unit | **all matching rows fetched and inserted (A01)** |
| B.5 | `fetch_by_rids(rid_column=<FK>)`, partial population (some FK values have rows; server has more) | unit | all server rows fetched; engine = union |
| B.6 | `fetch_predicate` over populated engine | unit | no crash; new rows inserted; existing untouched |
| B.7 | Empty input list | unit | returns 0; no HTTP requests |
| B.8 | `fetch_by_rids` with `rid_column` not in target table | unit | clear error |

### Layer C — End-to-end through the full stack

| # | Scenario | Kind | Assert |
|---|---|---|---|
| C.1 | Live denormalize on a fresh workspace | live | correct row count |
| C.2 | Live denormalize twice in one session | live | both succeed; same result both times |
| C.3 | Live denormalize across sessions (two processes, same workspace dir) | live | both succeed; same result both times |
| C.4 | **Live denormalize with N feature rows per anchor (A01 reproduction)** | live | N × anchor-count rows; all N executions present |
| C.5 | Mutation → re-denormalize. **Insertion** server-side between calls. | live | second call observes the new row |
| C.5x | Mutation → re-denormalize. **Deletion or update** server-side between calls. | live, `xfail` — **planned, not yet written (TC-04)** | freshness limitation — see §6.5 |
| C.6 | `split_dataset` then live denormalize of the parent | live | parent's feature rows visible |
| C.7 | Live denormalize a Split parent containing children | live | members from children appear |
| **C.8** | **Two element paths converge on the same target table with disjoint rid sets (F5 / SC-06)** | unit | **every Image RID reachable via either path is fetched; local cache equals the union (covered by `tests/local_db/test_denormalize_impl.py::TestRowCompletenessInvariant`)** |

### Layer D — Cross-channel parity

| # | Scenario | Kind | Assert |
|---|---|---|---|
| D.1 | Same dataset, same include_tables/row_per, fetched via DatasetBag (source="local") AND Dataset (source="catalog") | live | identical row count and column set |
| D.2 | Same as D.1 but with multi-feature-per-anchor data (A01 shape) | live | identical row count |
| **D.3** | **Same as D.1 but with two element paths converging on one table (F5 / SC-06 shape)** | live | **identical row count and column set; both sources see every reachable row** |

### Layer E — `describe()` / preflight estimated_row_count

| # | Scenario | Kind | Assert |
|---|---|---|---|
| E.1 | Anchor table == `row_per` (case 1) | live | exact integer estimate == anchor count |
| E.2 | **Anchor table is downstream of `row_per` (case 2 — feature-table common case)** | live | **`in_scope_row_per_rows` and `total` are `None`; a `reason` field is present (A02 regression)** |
| E.3 | Mixed scoping anchors (some at `row_per`, some downstream) | live | honest `None` with reason (the case-2 contribution can't be added to the case-1 count) |
| E.4 | All anchors orphan (no FK path) | live | `orphan_rows` is exact, `in_scope_row_per_rows == 0`, `total == orphan_rows` |
| E.5 | `describe(<feature_name>)` and `as_dataframe(<feature_name>)` agree on `columns` (analyst/01 regression) | live | `{name for name, _ in plan["columns"]} == set(df.columns)` |
| E.6 | `describe(<unknown_name>)` never raises (dry-run invariant) | unit | returns a 13-key dict with empty defaults; non-empty `warnings`; no exception |

### Layer F — `from_rids` constructor

| # | Scenario | Kind | Assert |
|---|---|---|---|
| F.1 | `from_rids` with `(table, RID)` tuples and explicit `dataset_rid` | unit | correct anchor partitioning; row count matches |
| F.2 | `from_rids` with bare RIDs (catalog lookup) | unit | resolves to (table, RID) via `ml.resolve_rids` or `catalog.resolve_rid` |
| F.3 | `from_rids` rejects malformed inputs (no model, no catalog for bare RIDs, bad tuple arity) | unit | `ValueError` with clear message |
| F.4 | `from_rids` without `dataset_rid` against a real-shaped catalog (TC-05) | live | `ValueError` raised; message explains the SQL scoping constraint |

### Coverage status as of 2026-05-26

- **Layer A**: covered by `tests/local_db/test_paged_fetcher.py`
  (A.1–A.7 in `TestFetchByRids`; A.8 is the new defensive case).
- **Layer B**: B.1, B.2, B.7 covered. B.3, B.4, B.5, B.6, B.8
  are fix-pass additions for A01.
- **Layer C**: C.1, C.2, C.6 covered by
  `tests/dataset/test_split.py` and `test_denormalize.py`.
  C.3, C.4, C.5, C.7 are fix-pass additions. **C.5x is
  planned but not yet written** (TC-04).
- **Layer D**: D.1 is partially covered by
  `test_catalog_and_bag_denormalize_consistency` (single-table
  trivial case only); **D.2 is not yet written** (TC-01,
  TC-10).
- **Layer E**: E.1–E.4 covered by the A02 fix-pass tests in
  `test_denormalizer.py::TestDescribe`. E.5 covered by
  `TestFeatureNameResolution::test_describe_and_run_agree`
  (one key, one input; TC-02 names the gap for the other 12
  keys). E.6 covered by `test_describe_never_raises_on_*`.
- **Layer F**: F.1–F.3 covered by `TestFromRids` in
  `test_denormalizer.py`. F.4 closed by PR #239's
  `dataset_rid` guard; the new `ValueError`-shaped test pins
  the contract.

---

## 10. Process commitments

Future engineering on this subsystem should respect:

1. **Reference this doc**, not the implementation plans. Plans
   are point-in-time. This doc is current. If a piece of
   behavior is not in this doc, it is not part of the
   contract.
2. **Update this doc** when changing any of: the
   `_insert_rows` semantics, the `PagedFetcher`
   lifetime/contract, the source-mode selection in
   `Denormalizer.__init__`, the freshness model in §6, the
   `describe()` envelope shape in §8.3, the
   `_resolve_table_names` algorithm in §8.4, or any of the
   nine semantic Rules in §6.2.
3. **Add a test row** to §9 when fixing any denormalize bug.
   The minimum-failing repro becomes the regression test; the
   contract it exposed gets restated in §4–§8. If the test is
   not yet written, add the row anyway and mark it "planned,
   not yet written" so the gap is visible.
4. **The Rules live here.** This is the single source of
   record. Earlier drafts split the Rules into a separate
   "semantics companion" document; that companion has been
   merged into §6.2 of this doc and removed. If you change a
   Rule's semantics, update §6.2 and the user-facing summary
   above the contract divider — single-file edit, no
   cross-doc sync.
5. **Plans go to `docs/superpowers/plans/archive/` when
   superseded.** Don't delete them — they're historical
   context — but don't leave them in `plans/` where future
   readers will mistake them for current design.
6. **Silent zero is a bug class, not an acceptable failure
   mode.** The audit's cross-cutting observation §5.1 names a
   half-dozen sites where a code path returns an empty result
   with no diagnostic — `from_rids`'s placeholder
   `dataset_rid` (closed by PR #239), `describe`'s
   broad-except resolver (closed by the `warnings` envelope),
   `__init__`'s fallback to `source="local"` on
   `ErmrestPagedClient` construction failure (logging +
   `_init_warning` stash; full `describe()["warnings"]`
   surfacing pending), `_run`'s broad-except around
   `list_dataset_children`. New code in this subsystem must
   either raise on its silent-zero paths or surface a
   `reason` / `warnings` field. The honest-`None`-plus-
   `reason` pattern `describe.estimated_row_count` already
   uses (§8.3.3) is the canonical template.
