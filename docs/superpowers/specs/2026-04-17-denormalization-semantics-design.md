# Denormalization Semantics — Implementation Spec

**Status:** Draft, pending review.
**Date:** 2026-04-17.
**Companion doc:** `docs/concepts/denormalization.md` (user-facing).
**Scope:** Defines the semantics and implementation contract for the
`Denormalizer` class and the denormalization primitive. Supersedes the
prior ad-hoc behavior of `_prepare_wide_table` path selection.

## 1. Goals and scope

### 1.1 Goals

1. **Deterministic, predictable denormalization.** Users can reason about
   row count, column set, and orphan handling without understanding the FK
   graph internals.
2. **ML-native default shape.** One row per observation, upstream context
   hoisted and repeated — the classic star-schema denormalization.
3. **Clear errors for ambiguous cases.** When the request is
   under-specified, produce actionable errors that list the exact options
   for resolution.
4. **Unambiguous `row_per` and `via` semantics.** Give users explicit knobs
   for cases auto-inference cannot resolve.
5. **Generalize beyond Dataset.** Support arbitrary RID anchor sets for
   workflows that don't start from a Dataset.
6. **Preserve the LEFT-JOIN-like "all anchors contribute" property.**
   Orphan anchors emit rows rather than being silently dropped.

### 1.2 Non-goals

- Downstream-to-leaf aggregation (collapsing N rows per leaf). Explicitly
  deferred; rejected with clear error in v1.
- New caching behavior. The existing `Workspace.cache_denormalized` and
  `ResultCache` machinery continues to serve caching concerns.
- Upload-back semantics. Out of scope.
- Version-pinned denormalization (the `version` parameter remains a
  deprecated no-op with a UserWarning, as in current code).

### 1.3 Migration posture

Direct replacement. `Denormalizer` is introduced as the new public class.
The old methods `Dataset.denormalize_as_dataframe`, `denormalize_as_dict`,
`denormalize_columns`, and `denormalize_info` are **removed outright** and
replaced with the new sugar methods in §2.4. Same for `DatasetBag`.

No deprecation window — callers are updated to the new API in the same
change. This is safe because the new API is a superset of the old API
(every legitimate previous call has an equivalent new form with better
semantics).

## 2. The `Denormalizer` class

### 2.1 Location

`src/deriva_ml/local_db/denormalize.py` (existing module). The public
interface becomes the `Denormalizer` class.

The existing free `denormalize()` function is renamed to the private
`_denormalize_impl()` and continues to host the SQL-generation and
fetch-orchestration work, now driven by `Denormalizer` methods rather
than called directly. Its previous callers in `Dataset` and `DatasetBag`
are rewired to use `Denormalizer` via the sugar methods described in §2.4.

The `DenormalizeResult` dataclass (already exported from this module)
remains the return type of the SQL-generation step and is surfaced as the
internal representation that `Denormalizer.as_dataframe()` /
`.as_dict()` convert into their user-facing return types.

### 2.2 Constructors

```python
class Denormalizer:
    def __init__(self, dataset: DatasetLike):
        """Construct from a Dataset or DatasetBag.

        The dataset's members (recursively, including nested datasets) become
        the anchor set. Catalog, workspace, and model are derived from the
        dataset's ML instance (for Dataset) or the bag's DatabaseModel (for
        DatasetBag).
        """

    @classmethod
    def from_rids(
        cls,
        anchors: list[str | tuple[str, str]],
        *,
        ml: "DerivaML | None" = None,
        catalog: "ErmrestCatalog | None" = None,
        workspace: "Workspace | None" = None,
        model: "DerivaModel | None" = None,
        ignore_unrelated_anchors: bool = False,
    ) -> "Denormalizer":
        """Construct from an explicit anchor set.

        Anchors may be bare RIDs (table looked up via catalog) or
        ``(table_name, RID)`` tuples (lookup skipped). Mixed forms supported.

        Pass either ``ml=`` (common) or all three of ``catalog``,
        ``workspace``, ``model`` (escape hatch for testing / custom setups).
        """
```

### 2.3 Methods

```python
def as_dataframe(
    self,
    include_tables: list[str],
    *,
    row_per: str | None = None,
    via: list[str] | None = None,
    ignore_unrelated_anchors: bool = False,
) -> pd.DataFrame:
    """Materialize the wide table as a pandas DataFrame."""

def as_dict(
    self,
    include_tables: list[str],
    *,
    row_per: str | None = None,
    via: list[str] | None = None,
    ignore_unrelated_anchors: bool = False,
) -> Generator[dict[str, Any], None, None]:
    """Stream the wide table row-by-row as dicts."""

def columns(
    self,
    include_tables: list[str],
    *,
    row_per: str | None = None,
    via: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Preview (column_name, type_name) pairs. No data fetch. Model-only."""

def plan(
    self,
    include_tables: list[str],
    *,
    row_per: str | None = None,
    via: list[str] | None = None,
) -> dict[str, Any]:
    """Dry-run the call. Returns planning metadata (see §5)."""

def explore_schema(
    self,
    tables: list[str] | None = None,
) -> dict[str, Any]:
    """Describe FK graph from the dataset's/anchors' reachable tables."""
```

### 2.4 Dataset-side sugar

Add to `Dataset` (in `src/deriva_ml/dataset/dataset.py`) and `DatasetBag`
(in `src/deriva_ml/dataset/dataset_bag.py`):

```python
def denormalize(
    self,
    include_tables: list[str],
    *,
    row_per: str | None = None,
    via: list[str] | None = None,
    ignore_unrelated_anchors: bool = False,
) -> pd.DataFrame:
    """Shortcut for ``Denormalizer(self).as_dataframe(...)``."""

def denormalize_as_dict(self, ...) -> Generator[dict, None, None]:
    """Shortcut for ``Denormalizer(self).as_dict(...)``."""

def denormalize_columns(self, ...) -> list[tuple[str, str]]:
    """Shortcut for ``Denormalizer(self).columns(...)``."""

def denormalize_plan(self, ...) -> dict[str, Any]:
    """Shortcut for ``Denormalizer(self).plan(...)``."""

def explore_schema(self, ...) -> dict[str, Any]:
    """Shortcut for ``Denormalizer(self).explore_schema(...)``."""
```

### 2.5 Removed methods

The following methods are **removed outright** from `Dataset` and
`DatasetBag` and replaced with the §2.4 sugar:

| Removed | Replacement |
|---------|-------------|
| `denormalize_as_dataframe(...)` | `denormalize(...)` |
| `denormalize_info(...)` | `denormalize_plan(...)` |
| `denormalize_as_dict(...)` | kept — same name, new body |
| `denormalize_columns(...)` | kept — same name, new body |

The `DatasetLike` protocol in `src/deriva_ml/interfaces.py` is updated to
match: `denormalize_as_dataframe` and `denormalize_info` are removed from
the abstract surface; `denormalize`, `denormalize_as_dict`,
`denormalize_columns`, `denormalize_plan`, and `explore_schema` are the
documented protocol methods.

## 3. Semantic rules

### 3.1 Rule 1: Row cardinality = `|row_per rows in scope|` + `|orphan rows|`

Each output row corresponds to one of:

- A **`row_per` row in scope**: a row of the `row_per` table that is
  reachable from at least one anchor via the FK graph.
- An **orphan row**: a row of some anchor whose table is in
  `include_tables` and upstream of `row_per`, when no `row_per` row is
  reachable from that anchor. Its columns are populated from the anchor
  and its own upstream FKs; `row_per` and downstream columns are NULL.

### 3.2 Rule 2: Auto-inference of `row_per`

Given `include_tables = {T₁, …, Tₙ}`, build the directed graph on
`include_tables ∪ via` where an edge `A → B` means A has an outbound FK to
B (directly or via non-member association-table intermediates). Find
sinks — tables with no outbound edges in this subgraph.

- Exactly one sink → `row_per = that sink`.
- Zero sinks → cycle; raise `DerivaMLException`.
- Multiple sinks → multi-leaf ambiguity; raise with candidates listed.

### 3.3 Rule 3: Column projection

- Columns come **only** from tables in `include_tables`.
- `via` tables contribute to the join path but not to columns.
- Association tables on a necessary chain between two requested tables are
  **transparent intermediates**: joined through, columns omitted.
- Non-requested tables that are neither `via` nor transparent
  intermediates are not joined at all.

### 3.4 Rule 4: Upstream hoisting (star schema)

For each `row_per` row R:

- Walk outbound FKs from R to every table T in `include_tables`.
- The columns of the reached row of T are repeated verbatim in R's output
  row, labeled with `{T}.{column_name}` prefix.
- If T is reached via an N-to-N link (through an association table or
  M-to-N linking table), one output row is emitted per (R, T-row)
  combination. The `row_per` count grows accordingly — the output then has
  `|row_per × (average T-links per row_per)|` rows.

### 3.5 Rule 5: Explicit `row_per` with downstream table → error

If `row_per` is explicitly specified and any table in `include_tables` is
**downstream** of `row_per` (i.e., `row_per` has an outbound FK path to
it), raise:

```
DerivaMLException:
  Table {T} is downstream of row_per={row_per}. One row per {row_per}
  would require aggregating multiple {T} rows per {row_per} row —
  aggregation is not yet supported.

  Options:
    • Drop row_per to get one row per {T}.
    • Remove {T} from include_tables.
```

### 3.6 Rule 6: Multiple FK paths → error

If between `row_per` and another table T in `include_tables ∪ via` there
exist two or more distinct FK paths, raise:

```
DerivaMLException:
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

The "suggested intermediates" list is the set of tables that appear in at
least one path but not in `include_tables`.

### 3.7 Rule 7: Anchor contribution

Every anchor must contribute to the output in exactly one way, determined
by its table and reachability:

1. **Anchor table == `row_per`**: contributes one output row (the anchor
   row itself, upstream cols hoisted).
2. **Anchor table ∈ `include_tables`, upstream of `row_per`, reaches at
   least one `row_per` row**: filter-only contribution (the `row_per` rows
   reachable from this anchor are in scope). Does not produce its own row
   — that row appears via the `row_per` row that hoists this anchor.
3. **Anchor table ∈ `include_tables`, upstream of `row_per`, reaches no
   `row_per` row**: produces one orphan row (Rule 1 §3.1).
4. **Anchor table ∉ `include_tables`, upstream of `row_per`, reaches at
   least one `row_per` row**: filter-only contribution. No orphan row
   because the anchor's columns aren't in the output.
5. **Anchor table ∉ `include_tables`, upstream of `row_per`, reaches no
   `row_per` row**: contributes nothing. Would be silently dropped except
   see §3.8.
6. **Anchor table has no FK path to or from any table in `include_tables
   ∪ via`**: raise `DerivaMLException` unless `ignore_unrelated_anchors=
   True`.

### 3.8 Rule 8: Unrelated anchors

If any anchor falls under case 6 above and `ignore_unrelated_anchors=False`
(the default), raise:

```
DerivaMLException:
  Anchors of table {T} have no FK path to any table in include_tables=
  {list}. They will not affect the output.

  Options:
    • Remove these anchors from the anchor set.
    • Add {T} (or a linking table) to include_tables.
    • Pass ignore_unrelated_anchors=True to silently drop them.
```

When `ignore_unrelated_anchors=True`, unrelated anchors are silently
ignored (they contribute no row and no filter).

Anchors of case 5 (table ∉ include_tables, unreachable) are silently
dropped regardless of the flag — they contribute no output either way, so
there's nothing to warn about.

### 3.9 Rule 9: Nested datasets

When constructed from a `Dataset`/`DatasetBag`, the anchor set is the
dataset's members **recursively**, including all members of nested
datasets. This matches current
`Dataset.list_dataset_members(recurse=True)` behavior. The
`Denormalizer.__init__(dataset)` constructor is responsible for
materializing the recursive member list; `from_rids` takes anchors
directly and does no recursion.

### 3.10 Rule 10: Version parameter (deferred)

A `version` parameter is accepted by the deprecated aliases for backward
compatibility but emits a `UserWarning` and has no effect. Proper
version-pinned denormalization is deferred to a future spec.

## 4. Algorithm

### 4.1 High-level flow

```
Denormalizer(anchors, ml/catalog/workspace/model)
  .as_dataframe(include_tables, row_per=?, via=?, ignore_unrelated=?)

1. Resolve anchor tables
     For any bare-RID anchor, look up its owning table via catalog.
     Produce a uniform list[(table_name, RID)].

2. Validate the request
     - include_tables must be non-empty and all names must exist in model.
     - via tables, if any, must exist in model.
     - row_per, if explicit, must be in include_tables.

3. Build FK subgraph
     Restricted to include_tables ∪ via. Treat association tables that
     bridge two requested/via tables as transparent transitive edges.

4. Determine row_per
     If explicit: validate per Rule 5 (downstream-to-row_per check).
     Else: run sink-finding per Rule 2.

5. Check path ambiguity (Rule 6)
     For each (row_per, T) where T ∈ include_tables ∪ via, enumerate FK
     paths. If >1, raise with suggestions.

6. Validate anchors (Rule 7 + Rule 8)
     Partition anchors by their relationship to include_tables and
     row_per. Raise if any are unrelated unless ignore_unrelated_anchors.

7. Build join plan
     Produce a list of join edges (fk_col, pk_col) + join_types
     (inner/left) per FK along the row_per→... paths.

8. Populate local SQLite
     If source is "catalog": use PagedFetcher to populate rows per table.
     If source is "slice": use already-attached slice rows.

9. Emit main SQL
     SELECT projected columns FROM row_per LEFT JOIN ... WHERE
     row_per rows are in scope of anchors.

10. Emit orphan rows (Rule 7 case 3)
     For each anchor in include_tables that's upstream of row_per and has
     no reachable row_per row, emit one orphan row with NULL row_per cols.

11. Combine via UNION ALL
     Main result + orphan rows.

12. Return materialized DataFrame / iterator.
```

### 4.2 FK subgraph construction

The subgraph edges are:

- **Direct FK from A to B** where A, B ∈ include_tables ∪ via: direct
  edge A → B.
- **Transitive path A → C₁ → ... → Cₙ → B** where all `Cᵢ` are pure
  association tables not in `include_tables`: adds a single transitive
  edge A → B.
- No other edges.

Pure association tables are detected via a predicate: a table with only
system columns (RID, RCT, RMT, RCB, RMB) and exactly two domain FKs (not
counting system FKs to ERMrest_Client etc.) is a pure association table.

### 4.3 Sink-finding (Rule 2)

Standard graph algorithm: a sink is a node with no outgoing edges.
Complexity: O(|E|) where E is the edge count of the subgraph.

### 4.4 Path enumeration (Rule 6)

For each pair (row_per, T) where T ∈ include_tables ∪ via, enumerate all
distinct simple paths. Distinctness is by sequence of intermediate tables.
Complexity: O(paths × length). For typical schemas (10s of tables, small
branching), this is trivially fast.

### 4.5 Orphan row emission

For anchors classified as Rule 7 case 3:

1. For each such anchor, emit a single row.
2. Populate columns for tables upstream of the anchor's table (along the
   subgraph) by running a small SQL query per orphan anchor to fetch the
   upstream row chain.
3. `row_per` and downstream columns are left NULL.

The orphan rows are UNION-ALL'd with the main result.

## 5. `plan()` output structure

```python
{
    "row_per": "Image",
    "row_per_source": "auto-inferred" | "explicit",
    "row_per_candidates": ["Image"],   # from sink-finding
    "columns": [("Image.RID", "text"), ...],
    "include_tables": ["Image", "Subject"],
    "via": [],
    "join_path": ["Image", "Observation", "Subject"],
    "transparent_intermediates": ["Observation"],
    "ambiguities": [
        # Empty if no ambiguities, else:
        {
            "type": "multiple_paths",
            "from": "Image",
            "to": "Subject",
            "paths": ["Image → Subject (direct)", "Image → Observation → Subject"],
            "suggestions": {
                "add_to_include_tables": ["Observation"],
                "add_to_via": ["Observation"],
            },
        },
    ],
    "estimated_row_count": {
        "in_scope_row_per_rows": 42,
        "orphan_rows": 3,
        "total": 45,
    },
    "anchors": {
        "total": 20,
        "by_type": {"Image": 12, "Subject": 8},
        "unrelated": [],    # populated if any case-6 anchors exist
    },
    "source": "catalog" | "slice",
}
```

`estimated_row_count` requires a catalog `COUNT(*)` per table involved —
still much cheaper than a full denormalize. Can be omitted / marked
`None` if estimation is costly (e.g., for very large schemas with many
joins).

## 6. `explore_schema()` output structure

```python
{
    "member_types": ["Image", "Subject"],   # if constructed from a dataset
    "anchor_types": ["Image", "Subject"],   # all distinct anchor tables
    "reachable_tables": {
        "Image": ["Subject", "Observation", "Diagnosis", "Execution_Image_Quality"],
        "Subject": ["Observation", "Diagnosis"],
    },
    "association_tables": ["Dataset_Image", "ClinicalRecord_Observation"],
    "feature_tables": ["Execution_Image_Quality"],  # detected via naming/structure
    "schema_paths": {
        ("Image", "Subject"): [
            {"path": ["Image", "Subject"], "direct": True},
            {"path": ["Image", "Observation", "Subject"], "direct": False},
        ],
        ("Image", "ClinicalRecord"): [
            {"path": ["Image", "Observation", "ClinicalRecord_Observation",
                     "ClinicalRecord"], "direct": False},
        ],
    },
}
```

If `tables=[...]` is given, the output is filtered to paths involving at
least one of those tables.

## 7. Errors

All errors raised by `Denormalizer` are `DerivaMLException` or subclasses:

- `DerivaMLDenormalizeAmbiguousPath` — Rule 6.
- `DerivaMLDenormalizeMultiLeaf` — Rule 2 (multiple sinks).
- `DerivaMLDenormalizeNoSink` — Rule 2 (cycle).
- `DerivaMLDenormalizeDownstreamLeaf` — Rule 5.
- `DerivaMLDenormalizeUnrelatedAnchor` — Rule 8.

Each error message includes:

1. A plain-English description of the problem.
2. A list of the exact option(s) to fix it.
3. The concrete suggestions (e.g., the list of intermediate tables to add).

## 8. Testing

### 8.1 Unit tests (no live catalog)

Location: `tests/local_db/test_denormalize.py` (existing file, extend).

New test classes:

- **`TestRowPerInference`**: sink-finding on all supported topologies —
  linear chain, diamond, fork, cycle, feature-table, association-only.
- **`TestPathAmbiguity`**: detect and report every variant of Rule 6
  — direct + indirect, two indirect, three-way.
- **`TestOrphanRows`**: Rule 7 cases 1–5 exhaustively.
- **`TestUnrelatedAnchors`**: Rule 8 both with and without the flag.
- **`TestExplicitRowPer`**: downstream-leaf rejection (Rule 5); leaf
  override that agrees with auto-inference; leaf override that forces
  non-default sink.
- **`TestVia`**: path routing without column projection; via combined
  with include_tables.
- **`TestExploreSchema`**: schema reachability, association/feature table
  detection, path enumeration.
- **`TestPlan`**: plan output structure, ambiguity reporting, row-count
  estimation.
- **`TestFromRids`**: bare RIDs, `(table, RID)` tuples, mixed forms; with
  `ml=` vs separate deps; unrelated anchor rejection.

Each test uses fixtures that pre-populate a minimal schema (no live
catalog). The `canned_bag_model` pattern from Phase 1 is reused.

### 8.2 Integration tests (live catalog)

Location: `tests/dataset/test_denormalize.py`.

The three currently-xfailed tests become expected passes:

- `test_diamond_resolved_with_intermediate` — Rule 2 selects Image,
  intermediate Observation disambiguates path.
- `test_association_mandatory_intermediate` — association table ACTS as
  transparent intermediate.
- `test_feature_table_included` — Rule 2 selects the feature table as
  `row_per`.

New integration tests:

- Heterogeneous dataset with orphan Subject members.
- `Denormalizer.from_rids` against a live catalog.
- `denormalize_plan` on a real dataset.
- `explore_schema` on a real dataset.

### 8.3 Regression safety

All existing passing tests in `tests/dataset/test_denormalize.py` must
continue to pass. The two `pytest.mark.skip` tests (for the deleted
`bag._denormalize()` private method) remain skipped.

## 9. Implementation order

1. Introduce `Denormalizer` class with core `as_dataframe`, `as_dict`,
   `columns`, `plan` methods. Implement new rule system.
2. Add `explore_schema` method.
3. Add `from_rids` constructor + RID lookup.
4. Add sugar methods on `Dataset` and `DatasetBag` (`denormalize`,
   `denormalize_plan`, `explore_schema`).
5. Rewrite `denormalize_as_dict` and `denormalize_columns` bodies to
   delegate to `Denormalizer` (same names, new semantics).
6. **Remove** `denormalize_as_dataframe` and `denormalize_info` from
   `Dataset`, `DatasetBag`, and the `DatasetLike` protocol.
7. Update every in-repo caller of the removed methods to the new names:
   - `denormalize_as_dataframe(...)` → `denormalize(...)`
   - `denormalize_info(...)` → `denormalize_plan(...)`
   Use `grep -rn` to find all callers; expected locations include
   `tests/dataset/`, possibly `src/deriva_ml/dataset/split.py`, and any
   notebooks under `docs/`.
8. Extend unit tests per §8.1.
9. Remove `xfail` markers from the three integration tests per §8.2.
10. Run full test suite (unit + live); confirm no regressions.

### 9.1 Breaking-change summary

The assumption is that no external code uses the old interface, so
removal is clean. Two semantic changes are in play regardless:

1. **Ambiguous paths now error.** Previously silent path selection (the
   direct FK would be picked in diamond cases) now raises. Callers must
   disambiguate via `include_tables` or `via`.
2. **Row counts may change in diamond cases.** Calls that happened to
   work before may produce different (more correct) row counts under the
   new rules.

Neither is a migration concern given the "no external users" assumption
— they are correctness improvements that propagate automatically to
tests, which is where regressions would surface.

## 10. Open questions / future work

- **Aggregation**: future spec for downstream-to-row_per with aggregation
  rules (`count`, `first`, `list`, `sum`, ...). Would relax Rule 5.
- **Version pinning**: future spec for `version` parameter + `ErmrestSnapshot`.
- **Feature table column projection**: if a feature has asset columns
  (e.g., pointers to files), how should they be serialized in the output?
- **Caching interaction**: `cache_denormalized` uses the old call
  signature. Needs updating to pass `row_per` and `via` through, and to
  include them in the cache key.
