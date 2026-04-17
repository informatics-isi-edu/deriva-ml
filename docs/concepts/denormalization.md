# Denormalization

Denormalization transforms normalized relational data into a single **wide
table** (a pandas DataFrame or equivalent) where each row contains all related
information for one observation, with columns from related tables joined
side-by-side. This is the standard input format for machine-learning
frameworks.

DerivaML's denormalization is designed around one principle:

> **Give me a table where each row is one ML observation, with every column
> I asked for filled in from related rows via FK traversal.**

The user names the tables they want columns from; DerivaML decides the shape
of the output from the schema's FK graph.

---

## The mental model

### One row per leaf

If you request tables `[A, B, C]` where `C` has an FK to `B` and `B` has an FK
to `A`, you get one row per `C` instance. `B` and `A` columns are **hoisted**
from `C`'s foreign keys and repeated across rows that share the same `B` or
`A`.

This is **star-schema denormalization**: the table furthest downstream in the
FK chain becomes the "row_per" table — the thing you get one row of each — and
upstream context is duplicated across rows as needed.

### Example

Schema: `Subject ← Observation ← Image` (Image has FK to Observation;
Observation has FK to Subject).

```python
ds.denormalize(["Subject", "Observation", "Image"])
```

Produces:

| Subject.RID | Subject.Name | Observation.RID | Observation.Date | Image.RID | Image.Filename |
|-------------|--------------|-----------------|------------------|-----------|----------------|
| S1          | Alice        | O1              | 2024-01-01       | I1        | a.png          |
| S1          | Alice        | O1              | 2024-01-01       | I2        | b.png          |
| S1          | Alice        | O2              | 2024-02-01       | I3        | c.png          |
| S2          | Bob          | O3              | 2024-01-15       | I4        | d.png          |

One row per Image. Subject columns repeat across all images for a subject.
Observation columns repeat across all images for an observation.

### Dataset membership acts as a filter

If you call `ds.get_denormalized_as_dataframe(...)` on a Dataset, the
dataset's members scope which rows are in scope. Only rows reachable from a
dataset member via FK appear in the output.

If the dataset's members are Images (say 4 of them), you get 4 rows — one per
member Image, with Observations and Subjects hoisted.

---

## The API

### On a Dataset or DatasetBag

```python
df = ds.get_denormalized_as_dataframe(
    include_tables=["Subject", "Observation", "Image"]
)
```

Returns a pandas DataFrame. Use `ds.get_denormalized_as_dict(...)` to stream
rows one at a time.

The method names follow the same conventions as the existing
`get_table_as_dataframe` / `get_table_as_dict` methods on Dataset and
DatasetBag — the stem `denormalized` describes the (virtual) table being
retrieved.

### Preview without fetching

```python
ds.list_denormalized_columns(["Subject", "Image"])
# → [("Subject.RID", "text"), ("Subject.Name", "text"), ("Image.RID", "text"), ...]

ds.describe_denormalized(["Subject", "Image"])
# → {
#     "row_per": "Image",
#     "columns": [...],
#     "join_path": [...],
#     "estimated_row_count": {...},
#     "ambiguities": [],
#   }

ds.list_schema_paths()
# → {"member_types": [...], "reachable_tables": {...}, "schema_paths": {...}}
```

### Arbitrary RID anchors (no dataset required)

```python
from deriva_ml.local_db.denormalize import Denormalizer

d = Denormalizer.from_rids(
    ["1-ABCD", "1-EFGH"],           # bare RIDs — tables looked up via catalog
    ml=ml_instance,
)
df = d.as_dataframe(["Subject", "Image"])
```

Or skip the lookup if you already know the tables:

```python
d = Denormalizer.from_rids(
    [("Image", "1-ABCD"), ("Image", "1-EFGH")],
    ml=ml_instance,
)
```

Mixed forms are supported in one call.

---

## Parameters

### `include_tables: list[str]` — required

The tables whose columns appear in the output. Also determines which table is
`row_per` (see below).

### `row_per: str | None` — optional

Names the table whose rows become output rows (one row per instance of this
table). If omitted, auto-inferred as the unique "deepest" table in
`include_tables` — the one no other requested table points to via FK.

You only need `row_per` when:

- Auto-inference finds more than one candidate (multi-leaf ambiguity).
- You want a non-default anchor for a specific ML task.

### `via: list[str] | None` — optional

Names tables to force into the join chain without contributing columns to the
output. Useful when two paths exist between your requested tables and you want
to specify routing without cluttering your output columns.

Example: `ds.denormalize(["Image", "Subject"], via=["Observation"])` routes
through Observation but doesn't include Observation columns in the output.

### `ignore_unrelated_anchors: bool = False`

By default, if an anchor has no FK path to any table in `include_tables`,
DerivaML raises an error. Set `True` to silently drop such anchors. Useful
when you know some dataset member types are out of scope for this particular
denormalization.

---

## Rules (for when you need to reason about edge cases)

### Rule 1 — Row cardinality

One output row per **`row_per`-table instance in scope**. "In scope" means
reachable from some anchor via the FK graph.

### Rule 2 — Auto-inference of `row_per`

Among the requested tables, the one that no other requested table points to
via FK is `row_per`. Intuition: the "deepest" table in your request.

If two or more candidates tie (multi-leaf), you get an error asking you to
specify `row_per` explicitly.

### Rule 3 — Column projection

Columns come from `include_tables`. Tables in `via` are path-only, no columns.
Tables that only appear as transitive intermediates (e.g., association tables
bridging two requested tables) are also path-only, no columns.

### Rule 4 — Column hoisting

For each `row_per` row, upstream columns are hoisted from the FK graph and
**repeated verbatim** across rows sharing the same upstream row. This is the
classic star-schema denormalization shape.

Tables reached through an N-to-N link produce one output row per
(row_per, linked-row) combination.

### Rule 5 — Downstream-to-row_per is rejected

If you set `row_per` explicitly and another requested table is downstream of
it (i.e., `row_per` points to it via FK), you get an error. This would
require aggregation (collapsing N downstream rows per `row_per` row), which
is a future feature.

Workaround: drop `row_per` and let auto-inference pick the downstream table,
or remove the downstream table from `include_tables`.

### Rule 6 — Path ambiguity requires resolution

If there are multiple FK paths between `row_per` and another requested table,
you get an error listing the paths. Resolve by:

- Adding an intermediate to `include_tables` (its columns are included), or
- Adding an intermediate to `via` (path-only, no columns), or
- Narrowing your request so only one path is valid.

Silent path selection is rejected by design — the current behavior would be
different between the two paths, and DerivaML won't guess.

### Rule 7 — Orphan anchors (LEFT-JOIN semantics)

If an anchor's table is upstream of `row_per` but the anchor doesn't reach
any `row_per` row (no rows of `row_per` type point back to it), the anchor
produces an **orphan row** — its columns populated, `row_per`-side columns
NULL. Upstream FK columns are still hoisted for the orphan row.

This is the LEFT-OUTER-JOIN interpretation: every anchor contributes at
least one row; unreachable ones contribute with NULL leaf-side data.

Anchors of types not in `include_tables` are rejected by default (see
`ignore_unrelated_anchors`).

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
# One row per diagnosis; Subject columns repeated across all diagnoses for a subject.
# Group by Subject.RID in pandas to aggregate.
```

### Diamond path — explicit routing

```python
# Schema has Image→Subject direct AND Image→Observation→Subject
ds.get_denormalized_as_dataframe(["Image", "Subject"])
# ERROR: multiple FK paths.

# Force the multi-hop with columns included:
ds.get_denormalized_as_dataframe(["Image", "Observation", "Subject"])

# Force the multi-hop without Observation columns:
ds.get_denormalized_as_dataframe(["Image", "Subject"], via=["Observation"])
```

### Feature values on images

```python
ds.get_denormalized_as_dataframe(["Image", "Execution_Image_Quality"])
# row_per = Execution_Image_Quality (auto — it points to Image).
# One row per feature observation; Image columns repeated for multi-execution images.
```

### Heterogeneous dataset with orphan members

```python
# Dataset has Image members AND Subject members
# (some subjects with no images in the dataset)
ds.get_denormalized_as_dataframe(["Subject", "Image"])
# row_per = Image.
# Output = |member Images| rows (Subjects hoisted) + |orphan Subjects| rows
# (Image columns NULL).
```

### Arbitrary anchors (no dataset)

```python
from deriva_ml.local_db.denormalize import Denormalizer

d = Denormalizer.from_rids(["1-ABCD", "2-WXYZ"], ml=ml)
df = d.as_dataframe(["Subject", "Image"])
```

---

## Exploring before denormalizing

When you don't know the schema well, use the exploration methods:

```python
# What can I put in include_tables?
info = ds.list_schema_paths()
print(info["member_types"])        # dataset element types
print(info["reachable_tables"])    # FK-reachable tables from each type
print(info["schema_paths"])        # available FK paths between tables

# What columns will I get for this request?
ds.list_denormalized_columns(["Image", "Subject"])

# Will this request succeed? How big will the output be?
plan = ds.describe_denormalized(["Image", "Subject"])
if plan["ambiguities"]:
    # Inspect `ambiguities` to see what to add/change
    ...
print(plan["estimated_row_count"])
```

The workflow: `list_schema_paths` → `list_denormalized_columns` →
`describe_denormalized` → `get_denormalized_as_dataframe`. Each step is more
expensive but more definitive.

---

## Relationship to earlier APIs

This is the only denormalization API. The earlier method names have been
removed. The replacements use the same `get_*_as_*` / `list_*` /
`describe_*` conventions as the existing `get_table_as_dataframe` and
`list_dataset_members` methods on Dataset and DatasetBag:

| Earlier name | Current name |
|--------------|--------------|
| `denormalize_as_dataframe(include_tables)` | `get_denormalized_as_dataframe(include_tables, ...)` |
| `denormalize_as_dict(include_tables)` | `get_denormalized_as_dict(include_tables, ...)` |
| `denormalize_columns(include_tables)` | `list_denormalized_columns(include_tables, ...)` |
| `denormalize_info(include_tables)` | `describe_denormalized(include_tables, ...)` |

Two behavioral changes from the earlier implementations:

- **Ambiguous FK paths now error** instead of silently picking the shortest
  one. Use `via` or add an intermediate to `include_tables` to resolve.
- **Feature tables are fully supported** as leaves (`row_per`). Previously
  they were not handled specially and some requests failed.

---

## Related concepts

- [Datasets](datasets.md) — how dataset membership defines the anchor set.
- [Local SQLite layer](../superpowers/specs/2026-04-15-unified-local-db-design.md)
  — how denormalization uses the local working DB under the hood.
