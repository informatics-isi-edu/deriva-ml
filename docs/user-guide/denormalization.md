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

### Dataset membership scopes by FK-reachability, not by direct membership

When you denormalize a `Dataset`, its members define the **scope** —
but "in scope" means **FK-reachable from a member**, not
**a member of**. A row of the `row_per` table appears whenever it can
be reached from *some* dataset member by following foreign keys, even
if that row was never added to the dataset directly.

This matters most for **upstream-partitioned** datasets. Suppose a
dataset has **8 `Subject` members and 4 `Image` members**, and `Image`
has an FK to `Subject` (`Image.Subject`). Denormalizing `["Image"]`
returns **every Image reachable from a member** — the 4 direct Image
members *plus* every Image belonging to one of the 8 member Subjects.
On the demo catalog that's **8 rows, not 4**: the member Subjects pull
their Images into scope through the FK. (This is the behavior that lets
`feature_values` and `get_denormalized_*` work on datasets whose members
are `Subject`s but whose features live on `Image` — see the
[reference, D7](../reference/denormalization.md#d7).)

If a dataset's members already *are* the `row_per` table, the two
notions coincide and you get one row per member. The general rule is the
FK-reachable one; direct-membership is just its simplest case.

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
`include_tables` is rejected ([Downstream-leaf rejection](../reference/denormalization.md#downstream-leaf-rejection)).**
The feature-association table is downstream of the target table, and the
denormalizer does not aggregate. The two intents the caller might have in mind are
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
# Output = |FK-reachable Images| rows (Subjects hoisted)
#       + |orphan Subjects with no reachable Image| rows (Image columns NULL).
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

## The six rules (for when you need to reason about edge cases)

DerivaML's denormalizer is governed by **six named semantic rules**.
The [reference](../reference/denormalization.md#formal-rules) gives the
formal definitions and the
[implementation contract](../design/denormalization-contract.md) §6.2
the planner-level language; this is the user-facing summary, grouped by
how often you'll meet each rule.

### Basic behavior (every call)

**Row grain.** One output row per **`row_per`-table instance in scope**.
By default `row_per` is auto-inferred as the "deepest" requested table —
the one no other requested table points to via FK. (Ties or FK cycles
raise; pass `row_per=` to resolve.) "In scope" means **FK-reachable from
a dataset member**, not member-of — see
[Dataset membership scopes by FK-reachability](#dataset-membership-scopes-by-fk-reachability-not-by-direct-membership).

**Column projection.** Columns come from `include_tables`. Tables in
`via`, and association tables that only bridge two requested tables, are
joined *through* but contribute no columns.

**Column hoisting.** For each `row_per` row, upstream columns are hoisted
from the FK graph and **repeated verbatim** across rows sharing the same
upstream row (the classic star-schema shape). A table reached through an
N-to-N link produces one row per `(row_per, linked-row)` combination, so
the `row_per` count grows. (This is also where a feature `selector`
reduces a multi-row feature group to one row — see
[Feature values on images](#feature-values-on-images).)

### Constraints (cause an error you must resolve)

**Downstream-leaf rejection.** If you set `row_per` explicitly and
another requested table is *downstream* of it (i.e. `row_per` points to
it via FK), you get `DerivaMLDenormalizeDownstreamLeaf` — one row per
`row_per` would require aggregating the downstream rows, which
denormalize doesn't do. Drop `row_per` (let auto-inference pick the
downstream table) or remove that table from `include_tables`.

**Path ambiguity.** If two or more distinct FK paths connect `row_per`
to another requested table, you get `DerivaMLDenormalizeAmbiguousPath`
listing the paths — the result would differ between them, so DerivaML
won't guess. Add the disambiguating intermediate to `include_tables`
(its columns appear) or to `via` (path-only), or narrow the request so
only one path is valid.

### Anchor disposition (what happens to each member)

Every dataset member (anchor) contributes in exactly one way, by its
table and reachability:

- **Member's table is `row_per`** → it's one output row.
- **Member is upstream of `row_per` and reaches a `row_per` row** →
  filter-only: it appears via the `row_per` row that hoists it (no row
  of its own).
- **Member is upstream of `row_per` but reaches no `row_per` row** → an
  **orphan row**: its columns populated, `row_per`-side columns NULL
  (LEFT-JOIN shape). Example: a `Subject` with no Images in scope.
- **Member's table has no FK path at all** to any `include_tables ∪ via`
  table → raises `DerivaMLDenormalizeUnrelatedAnchor`. Pass
  `ignore_unrelated_anchors=True` to silently drop these instead.

(`ignore_unrelated_anchors` affects only the last case, never orphan
emission. The reference's
[Anchor disposition](../reference/denormalization.md#anchor-disposition)
table spells out all six sub-cases.)

### Anchor-set definition (not a rule)

The denormalizer's anchor set is the dataset's members **recursively** —
members of nested datasets are followed transparently, and descendant
dataset RIDs are threaded into the SQL scope so their members show up
under the root. (`from_rids` takes anchors directly and does no
recursion.) This is *how anchors enter the system*, not a denormalize
behavior — so it's a definition, not one of the six rules.

> **On `version`.** A `version=` kwarg pins denormalization to a catalog
> snapshot (see "Version-pinned denormalization" above): implemented for
> live `Dataset` inputs, ignored for bags. It's a constructor concern,
> not a semantic rule.

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
is covered in more depth in the
[implementation contract](../design/denormalization-contract.md).

### `as_dict` is not streaming

Despite the "memory-efficient" wording in some older docstrings,
`get_denormalized_as_dict(...)` and `Denormalizer.as_dict(...)`
**materialise the full result before yielding any row**. They
expose an iteration interface, not a streaming cursor — peak
memory equals the full result set.

Use `as_dict` when downstream code naturally processes one row at
a time. Don't rely on it to bound memory for very large results.

True streaming is a known gap (audit finding SC-07); see §8.2 of the
[implementation contract](../design/denormalization-contract.md) for
the design discussion.

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
the [implementation contract](../design/denormalization-contract.md)).

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

## For maintainers

The full **implementation contract** — pipeline architecture, state
ownership, the `PagedFetcher` / `_insert_rows` contracts, the
row-completeness invariant, the fragility map, and the test matrix —
lives in the
[Denormalization implementation contract](../design/denormalization-contract.md).
That document is the authoritative design reference for engineers
working on the denormalize subsystem; this page is the user-facing guide.
