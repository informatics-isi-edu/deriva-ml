# Brainstorm: Unify feature-listing methods across catalog and bag

**Status:** paused — awaiting resumption.

**Started:** 2026-04-18 (during denormalization refactor work).

**Motivation:** The recent changes to denormalization and table caching
(Workspace + Denormalizer + unified `_denormalize_impl`) gave the
library a uniform way to abstract over "read rows from live catalog vs
downloaded bag vs attached slice." The feature-listing methods on
`DerivaML`, `Dataset`, and `DatasetBag` predate that unification and
still carry duplicated fetch logic. Worth revisiting.

---

## Context captured so far

### Where feature-listing methods live today

| Class | `find_features(table)` | `fetch_table_features` | `list_feature_values` |
|---|---|---|---|
| `DerivaModel` (`model/catalog.py`) | ✅ schema-only, no values | — | — |
| `DerivaML` (via `FeatureMixin`) | delegates to `model.find_features` | **catalog-mode** (pathBuilder + HTTP) | delegates to `fetch_table_features` |
| `Dataset` | delegates to `_ml_instance.find_features` | — (inherits from DerivaML) | — |
| `DatasetBag` | delegates to `self.model.find_features` | **bag-mode** (SQLAlchemy + local SQLite) | delegates to `fetch_table_features` |
| `Asset` | delegates to `_ml_instance.find_features` | — | delegates to ml's `list_feature_values` |

Also: `DerivaML.cache_features` (in `core/base.py:525`) hits
`fetch_table_features` and writes the DataFrame to the Workspace
result cache. No bag equivalent.

### The core duplication

`DerivaML.fetch_table_features` (~60 lines) and
`DatasetBag.fetch_table_features` (~50 lines) are **structurally
identical**. Same algorithm:

1. `find_features(table)` → list of Feature objects
2. For each Feature, dynamically build the Pydantic `FeatureRecord` class
3. Fetch raw rows from the feature table
4. Field-filter each row against the record class's fields
5. Optionally group by target RID and apply a selector

Only step 3 differs:
- DerivaML: `pb.schemas[X].tables[Y].entities().fetch()` (HTTP)
- DatasetBag: `select(feature_table); session.execute(...)` (SQLAlchemy)

### Cache asymmetry — issue C (explained to the user)

Three caching paths on Workspace / DerivaML:

| Method | What it fetches | Cache key |
|---|---|---|
| `Workspace.cached_table_read(table)` | full table | `table_read + table_name` |
| `Workspace.cache_denormalized(...)` | denormalized wide table | `denormalize + dataset_rid + tables + version + row_per + via + ignore_unrelated_anchors` |
| `DerivaML.cache_features(table, feat)` | one feature's values | `features + table + feature` |

**Three asymmetries:**

1. `cache_features` exists only on `DerivaML`, not `Dataset` or `DatasetBag`. A
   user switching from catalog to bag workflow gets different APIs and
   different caching behavior.

2. `cache_features` is **catalog-wide**, not dataset-scoped — the cache key
   doesn't include `dataset_rid`. Two datasets sharing a catalog can't cache
   different scoped views. Compare: `cache_denormalized` IS dataset-scoped.

3. Bag has no `cache_*` methods — "already local" is assumed to mean caching
   is redundant. True for row-fetch cost, but `fetch_table_features` does
   meaningful work above the row read (Feature discovery, dynamic
   FeatureRecord class construction, field filtering, selector grouping)
   that re-runs every call.

---

## Decisions captured so far

- **User confirmed scope includes A + B** (code-size DRY + interface consistency)
  — with constraint "as long as there's no performance consequence."

- **User did NOT yet decide on C** (cache asymmetry). I explained the issue
  and recommended **Option 1**: add scoped `cache_features` to both `Dataset`
  and `DatasetBag`, keep DerivaML's version for catalog-wide use. ~30 lines
  per class, no perf regression. **User hasn't answered yet — resume here.**

- **Row-fetch strategy** — three options proposed, user hasn't answered yet:
  - **(i)** Always populate workspace DB first (catalog side uses
    `PagedClient` like `_denormalize_impl` does with `source="catalog"`),
    then run SQLAlchemy query. One code path.
  - **(ii)** Two internal backends (HTTP pathBuilder, local SQLite) behind a
    shared interface. Keeps two fetch paths.
  - **(iii)** Reuse `_denormalize_impl` with `include_tables=[feature_table]`
    — features become a special case of denormalization.
  - **My recommendation: (i)** — uniform engine, single query path, small
    one-shot fetch overhead on catalog side; rows cached as a side benefit.
    (ii) defeats A. (iii) couples features to Rules 2/5/6 which don't
    really apply to single-table queries.

---

## Where to resume

1. Get user's answer on issue C (include in scope or not).
2. Get user's answer on row-fetch strategy (i / ii / iii).
3. Propose concrete design based on answers.
4. Present design sections incrementally for approval.
5. Write the spec doc to `docs/superpowers/specs/2026-04-18-unify-feature-listing-design.md`.
6. Invoke `writing-plans` to produce the implementation plan.

## Related context

- Design spec for the denormalization refactor that enabled this:
  `docs/superpowers/specs/2026-04-17-denormalization-semantics-design.md`
- Implementation plan for that refactor:
  `docs/superpowers/plans/2026-04-17-denormalization-semantics.md`
- The Workspace / Denormalizer unification landed in commits `0c032be` →
  `80b6987` (April 17-18, 2026). The `cached_table_read` and
  `cache_denormalized` entry points on `Workspace` are the existing
  patterns this brainstorm is aligning with.
