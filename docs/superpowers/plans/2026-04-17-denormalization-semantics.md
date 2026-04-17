# Denormalization Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `Denormalizer` class with deterministic star-schema denormalization semantics, replacing the ad-hoc path-selection behavior in the current free `denormalize()` function and the existing `Dataset.denormalize_*` methods.

**Architecture:** Rewrite the planner in `model/catalog.py` to implement sink-finding (Rule 2), downstream rejection (Rule 5), and ambiguity detection (Rule 6). Introduce `Denormalizer` class as the public API in `local_db/denormalize.py` with `as_dataframe`, `as_dict`, `columns`, `describe`, `list_paths` methods and `from_rids` constructor for non-dataset anchor sets. Rename `Dataset`/`DatasetBag` methods to follow `get_*_as_*` / `list_*` / `describe_*` conventions. Old method names are removed outright (no deprecation window). The three currently-xfailed integration tests become passing by design.

**Tech Stack:** Python ≥3.12, SQLAlchemy 2.x, SQLite (WAL), deriva-py (ErmrestCatalog, ermrest_model), pandas, pytest.

**Spec:**
- Implementation: `docs/superpowers/specs/2026-04-17-denormalization-semantics-design.md`
- User-facing: `docs/concepts/denormalization.md`

---

## Environment & conventions

- Work in worktree `/Users/carl/GitHub/deriva-ml/.claude/worktrees/compassionate-visvesvaraya/` on branch `claude/compassionate-visvesvaraya`.
- `uv` at `/Users/carl/.local/bin/uv`. Prepend to PATH if not found.
- `DERIVA_ML_ALLOW_DIRTY=true uv run pytest ...` for tests.
- `uv run ruff format src/ tests/` and `uv run ruff check src/` before each commit.
- Live integration tests require `DERIVA_HOST=localhost` and take 10–30 minutes — run them at the end, not between every task.
- Commit after each task. Messages use `feat(denormalize):`, `refactor(denormalize):`, `test(denormalize):` prefixes.

## File structure

### New files

| Path | Responsibility |
|------|----------------|
| `src/deriva_ml/local_db/denormalizer.py` | New module hosting the `Denormalizer` class. Imports the existing `_denormalize_impl` (renamed from today's `denormalize()` function) from `denormalize.py`. |
| `tests/local_db/test_denormalizer.py` | Unit tests for the `Denormalizer` class public API against the canned fixtures. |
| `tests/local_db/test_planner_rules.py` | Unit tests for the planner: sink-finding (Rule 2), downstream rejection (Rule 5), path ambiguity (Rule 6), orphan classification (Rule 7), unrelated-anchor rejection (Rule 8). Uses canned models, no catalog. |

### Modified files

| Path | Change |
|------|--------|
| `src/deriva_ml/model/catalog.py` | **Extract** two nested functions to module-level methods on `DerivaModel`: `_is_likely_association` (was nested in `_build_join_tree` at lines 727-751) → `_is_association_table`; `find_arcs` (was nested in `_schema_to_paths` at lines 1173-1188) → `_fk_neighbors`. **Compose** new helpers on top of those + existing `_schema_to_paths` / `_table_relationship`: `_outbound_reachable`, `_enumerate_paths`, `_find_sinks`, `_determine_row_per`, `_find_path_ambiguities`. Update `_prepare_wide_table` to accept `row_per` / `via` and invoke the new rules before the existing tree-building flow. |
| `src/deriva_ml/core/exceptions.py` | Add `DerivaMLDenormalizeAmbiguousPath`, `DerivaMLDenormalizeMultiLeaf`, `DerivaMLDenormalizeNoSink`, `DerivaMLDenormalizeDownstreamLeaf`, `DerivaMLDenormalizeUnrelatedAnchor` exception classes. |
| `src/deriva_ml/local_db/denormalize.py` | Rename public `denormalize()` → `_denormalize_impl()`. Keep `DenormalizeResult`, `_MinimalDatasetMock`, helpers. `_denormalize_impl` becomes the private implementation called by `Denormalizer`. |
| `src/deriva_ml/local_db/__init__.py` | Export `Denormalizer` (new), remove export of free `denormalize` function (replaced by class). |
| `src/deriva_ml/dataset/dataset.py` | Remove `denormalize_as_dataframe` (~165 lines), `denormalize_as_dict` (~100 lines), `denormalize_columns` (~45 lines), `denormalize_info` (~155 lines). Add sugar methods: `get_denormalized_as_dataframe`, `get_denormalized_as_dict`, `list_denormalized_columns`, `describe_denormalized`, `list_schema_paths`. |
| `src/deriva_ml/dataset/dataset_bag.py` | Same — remove old methods, add the same sugar methods. |
| `src/deriva_ml/interfaces.py` | Remove old methods from `DatasetLike` protocol (~180 lines of docstrings). Add new methods to protocol. |
| `src/deriva_ml/core/mixins/dataset.py` | Rename `DerivaML.denormalize_info` → `DerivaML.describe_denormalized`. Update the body to use `Denormalizer` (no dataset required — same pattern: catalog-wide counts). |
| `src/deriva_ml/dataset/split.py:738` | Update `source_ds.denormalize_as_dataframe(...)` → `source_ds.get_denormalized_as_dataframe(...)`. |
| `src/deriva_ml/local_db/workspace.py` | Update `cache_denormalized` to pass `row_per`, `via`, `ignore_unrelated_anchors` through to the planner, and include them in the cache key. |
| `tests/dataset/test_denormalize.py` | Rewrite: restructure around new rules, rename calls to new API, remove `xfail` markers from three tests (diamond-resolved, association-mandatory, feature-table), add new test classes for rules. |
| `tests/dataset/test_denormalize_info.py` | Rename to `tests/dataset/test_describe_denormalized.py`, update to new method name and new return-dict structure. |
| `tests/local_db/test_denormalize.py` | Rename to `tests/local_db/test_denormalize_impl.py` (covers the private `_denormalize_impl` function). Update for new signature. |

### Unchanged files

| Path | Why |
|------|-----|
| `src/deriva_ml/local_db/paths.py` | Phase 1–2, no changes. |
| `src/deriva_ml/local_db/sqlite_helpers.py` | Phase 1, no changes. |
| `src/deriva_ml/local_db/schema.py` | Phase 1, no changes. |
| `src/deriva_ml/local_db/paged_fetcher.py` | Phase 1, no changes. |
| `src/deriva_ml/local_db/paged_fetcher_ermrest.py` | Phase 1, no changes. |
| `src/deriva_ml/local_db/manifest_store.py` | Phase 1, no changes. |
| `src/deriva_ml/local_db/result_cache.py` | Phase 2, no changes. |

---

## Reuse inventory (DRY — do NOT reimplement)

The codebase already contains the primitives this refactor needs. Every new
helper must either **reuse as-is**, **extract from where it's currently
nested**, or **wrap** an existing routine. If you find yourself writing a new
BFS/DFS over the FK graph, stop — it already exists.

### Must-reuse (as-is)

| Existing routine | Location | Use in new code |
|------------------|----------|-----------------|
| `DerivaModel._schema_to_paths(root, max_depth=...)` | `src/deriva_ml/model/catalog.py:1120` | Authoritative FK-path enumerator (DFS, handles cycles, terminates at vocabs). Base for `_enumerate_paths` and `_outbound_reachable`. |
| `DerivaModel._table_relationship(fk_src, pk_dst)` | `src/deriva_ml/model/catalog.py:1079` | Resolves FK column pairs between two tables. Handles composite FKs + ambiguity detection. Use wherever pairs of joined tables need their FK edge. |
| `DerivaModel.name_to_table(name)` | `src/deriva_ml/model/catalog.py:371` | Name → Table resolver. Single source of truth — never walk schemas manually. |
| `DerivaModel.is_vocabulary(tbl)` | `src/deriva_ml/model/catalog.py:399` | Vocabulary detection for path termination. |
| `DerivaModel.is_asset(tbl)` | `src/deriva_ml/model/catalog.py:482` | Asset-table detection when formatting output columns. |
| `denormalize_column_name(table, column, multi_schema)` | `src/deriva_ml/model/catalog.py:100` | `{Table}.{col}` / `{schema}.{Table}.{col}` naming. **Do not invent new naming.** |
| `Dataset.list_dataset_members(...)` | `src/deriva_ml/dataset/dataset.py:701` | Dataset-member retrieval (grouped by element type) — the anchor source when `Denormalizer` is constructed from a dataset. |
| `Dataset._denormalize_datapath(...)` | `src/deriva_ml/dataset/dataset.py:782` | LEFT-JOIN / NULL-init pattern for row emission. Study before writing `Denormalizer._run()` — the NULL-initialization orphan pattern is already implemented here. |

### Must-extract (currently nested — promote to methods on `DerivaModel`)

| Currently nested | Promote to | Why |
|------------------|------------|-----|
| `_is_likely_association(tbl)` — nested inside `_build_join_tree` at `catalog.py:727-751` | `DerivaModel._is_association_table(name_or_tbl)` | Both `_prepare_wide_table` and the new `Denormalizer` need it. Nested function cannot be reused. |
| `find_arcs(table)` — nested inside `_schema_to_paths` at `catalog.py:1173-1188` | `DerivaModel._fk_neighbors(tbl)` | Returns FK-reachable tables (both outbound and inbound, deduplicated by target). The building block for `_outbound_reachable`. |

### Wrap / compose (new thin helpers delegate to existing)

| New helper | Composes | Notes |
|------------|----------|-------|
| `_outbound_reachable(from_table, tables_in_set)` | `_fk_neighbors` + association-transparent filter | BFS over already-available `_fk_neighbors`. ~15 lines. |
| `_enumerate_paths(from_t, to_t, tables_in_set, max_depth)` | `_schema_to_paths(root=from_t, max_depth=...)` + filter | Filter full path list for paths ending at `to_t`. Do NOT re-implement DFS. |
| `_find_sinks(include_tables, via)` | `_outbound_reachable` | Sink = table with empty `_outbound_reachable` (minus self) within the set. |
| `_determine_row_per(include_tables, via, row_per)` | `_find_sinks` + `_outbound_reachable` | Explicit-row_per validation also delegates to `_outbound_reachable`. |
| `_find_path_ambiguities(row_per, include_tables, via)` | `_enumerate_paths` | For each target, count distinct paths; assemble ambiguity dicts. |

### Must-study (reference implementation, may be wrappable)

- `Dataset.denormalize_as_dataframe` (`src/deriva_ml/dataset/dataset.py:974`) — wraps `_denormalize_datapath` to produce a DataFrame with `denormalize_column_name`-prefixed columns. The new `Denormalizer.as_dataframe` should produce the **same column-naming convention**. Consider whether `Denormalizer` can call `_denormalize_datapath` (after the planner guards) rather than replicate its generator logic.
- `Dataset.denormalize_info` (`src/deriva_ml/core/mixins/dataset.py:312`) — already produces a plan-dict with columns, join_path, per-table counts. `describe_denormalized` should **extend** this shape with `row_per`, `row_per_source`, `ambiguities`, `orphan_rows`, `anchors` fields — not rewrite it.

### Red-flag checklist

Before every helper function you add, ask:
- [ ] Am I walking FK edges? → use `_schema_to_paths` or `_fk_neighbors`.
- [ ] Am I checking if a table is an association? → use extracted `_is_association_table`.
- [ ] Am I resolving FK columns between two tables? → use `_table_relationship`.
- [ ] Am I naming an output column? → use `denormalize_column_name`.
- [ ] Am I fetching dataset member rows? → use `list_dataset_members`.
- [ ] Am I walking table records to build the wide table? → extend `_denormalize_datapath` or call it directly.

---

## Task 1: Add denormalize-specific exceptions

**Files:**
- Modify: `src/deriva_ml/core/exceptions.py`
- Test: `tests/model/test_exceptions.py` (may not exist — create if needed).

- [ ] **Step 1: Find the existing exception hierarchy and add new classes under a `DenormalizeError` sub-hierarchy.**

First read the file to understand the existing hierarchy:

```bash
grep -n "class DerivaML" src/deriva_ml/core/exceptions.py | head -20
```

Expected output should show a base `DerivaMLException` and a data error subtype like `DerivaMLDataError`. The new exceptions inherit from `DerivaMLException` (or from `DerivaMLDataError` if it exists and fits; pick the closest established convention).

- [ ] **Step 2: Write the new exception classes.**

Add to `src/deriva_ml/core/exceptions.py` (at the end of the existing exception definitions, before the module-level `__all__` if there is one):

```python
class DerivaMLDenormalizeError(DerivaMLException):
    """Base class for denormalization errors.

    All errors raised by :class:`~deriva_ml.local_db.denormalizer.Denormalizer`
    and related planning functions are instances of this class.
    """


class DerivaMLDenormalizeMultiLeaf(DerivaMLDenormalizeError):
    """Multiple candidate tables for ``row_per`` — ambiguous leaf.

    Raised when auto-inference finds more than one table in
    ``include_tables`` that could serve as the leaf. The user must specify
    ``row_per`` explicitly.

    Attributes:
        candidates: list of table names that all qualify as sinks.
    """

    def __init__(self, candidates: list[str], include_tables: list[str]) -> None:
        self.candidates = list(candidates)
        self.include_tables = list(include_tables)
        super().__init__(
            f"Multiple candidates for row_per: {candidates}. "
            f"Specify row_per=... explicitly. "
            f"(include_tables={include_tables})"
        )


class DerivaMLDenormalizeNoSink(DerivaMLDenormalizeError):
    """No sink found in the FK subgraph — cycle detected.

    Raised when every table in ``include_tables`` has an outbound FK to
    another table in the set, forming a cycle. Pathological — rare in
    real schemas.
    """


class DerivaMLDenormalizeDownstreamLeaf(DerivaMLDenormalizeError):
    """Explicit ``row_per`` conflicts with a downstream table in ``include_tables``.

    Raised when the user specifies ``row_per=X`` but another table in
    ``include_tables`` is downstream of X via FK (would require aggregation).

    Attributes:
        row_per: the explicit row_per value.
        downstream_tables: tables downstream of row_per that can't be hoisted.
    """

    def __init__(self, row_per: str, downstream_tables: list[str]) -> None:
        self.row_per = row_per
        self.downstream_tables = list(downstream_tables)
        super().__init__(
            f"Table(s) {downstream_tables} are downstream of row_per={row_per!r}. "
            f"One row per {row_per} would require aggregating multiple rows of "
            f"{downstream_tables} — aggregation is not yet supported. "
            f"Drop row_per to get one row per {downstream_tables}, or remove "
            f"{downstream_tables} from include_tables."
        )


class DerivaMLDenormalizeAmbiguousPath(DerivaMLDenormalizeError):
    """Multiple FK paths between two requested tables — can't silently choose.

    Raised when two or more distinct FK paths exist between ``row_per`` and
    another requested/via table. Silent path selection is rejected by
    design; the user must disambiguate by adding tables to ``include_tables``
    or ``via``.

    Attributes:
        from_table: the row_per table name.
        to_table: the table with ambiguous path.
        paths: list of path descriptions (each a list of table names).
        suggested_intermediates: tables that appear in at least one path but
            not in ``include_tables``.
    """

    def __init__(
        self,
        from_table: str,
        to_table: str,
        paths: list[list[str]],
        suggested_intermediates: list[str],
    ) -> None:
        self.from_table = from_table
        self.to_table = to_table
        self.paths = [list(p) for p in paths]
        self.suggested_intermediates = list(suggested_intermediates)
        path_strs = ["\n    " + " → ".join(p) for p in paths]
        super().__init__(
            f"Multiple FK paths between {from_table!r} and {to_table!r}:"
            f"{''.join(path_strs)}\n"
            f"Resolve by one of:\n"
            f"  • Add an intermediate to include_tables "
            f"(its columns will be in output): {suggested_intermediates}\n"
            f"  • Add an intermediate to via= (path-only, no columns): "
            f"{suggested_intermediates}\n"
            f"  • Narrow include_tables so only one path is valid."
        )


class DerivaMLDenormalizeUnrelatedAnchor(DerivaMLDenormalizeError):
    """Anchor has no FK path to any table in ``include_tables``.

    Raised when the caller passes anchors whose table has no FK relationship
    to any table in ``include_tables ∪ via``. The anchor would contribute
    nothing to the output.

    Pass ``ignore_unrelated_anchors=True`` to silently drop them.

    Attributes:
        unrelated_tables: tables of the unrelated anchors.
        include_tables: the include_tables argument for reference.
    """

    def __init__(
        self,
        unrelated_tables: list[str],
        include_tables: list[str],
    ) -> None:
        self.unrelated_tables = list(unrelated_tables)
        self.include_tables = list(include_tables)
        super().__init__(
            f"Anchors of table(s) {unrelated_tables} have no FK path to any "
            f"table in include_tables={include_tables}. They would contribute "
            f"nothing to the output.\n"
            f"Options:\n"
            f"  • Remove these anchors from the anchor set.\n"
            f"  • Add {unrelated_tables} (or a linking table) to include_tables.\n"
            f"  • Pass ignore_unrelated_anchors=True to silently drop them."
        )
```

- [ ] **Step 3: Add tests for the exception classes.**

Create `tests/model/test_exceptions.py` (or append if it exists):

```python
"""Unit tests for denormalization-specific exception classes."""

from __future__ import annotations

import pytest

from deriva_ml.core.exceptions import (
    DerivaMLDenormalizeAmbiguousPath,
    DerivaMLDenormalizeDownstreamLeaf,
    DerivaMLDenormalizeError,
    DerivaMLDenormalizeMultiLeaf,
    DerivaMLDenormalizeNoSink,
    DerivaMLDenormalizeUnrelatedAnchor,
    DerivaMLException,
)


class TestDenormalizeExceptionHierarchy:
    """All denormalize exceptions inherit from DerivaMLDenormalizeError."""

    def test_multi_leaf_inherits(self) -> None:
        err = DerivaMLDenormalizeMultiLeaf(["A", "B"], ["A", "B", "C"])
        assert isinstance(err, DerivaMLDenormalizeError)
        assert isinstance(err, DerivaMLException)

    def test_ambiguous_path_inherits(self) -> None:
        err = DerivaMLDenormalizeAmbiguousPath("X", "Y", [["X", "Y"], ["X", "Z", "Y"]], ["Z"])
        assert isinstance(err, DerivaMLDenormalizeError)

    def test_downstream_leaf_inherits(self) -> None:
        err = DerivaMLDenormalizeDownstreamLeaf("Subject", ["Image"])
        assert isinstance(err, DerivaMLDenormalizeError)

    def test_unrelated_anchor_inherits(self) -> None:
        err = DerivaMLDenormalizeUnrelatedAnchor(["Foo"], ["Image", "Subject"])
        assert isinstance(err, DerivaMLDenormalizeError)

    def test_no_sink_inherits(self) -> None:
        err = DerivaMLDenormalizeNoSink("cycle detected in FK graph")
        assert isinstance(err, DerivaMLDenormalizeError)


class TestDenormalizeExceptionMessages:
    """Exception messages include the specific fields needed to fix the problem."""

    def test_multi_leaf_message_includes_candidates(self) -> None:
        err = DerivaMLDenormalizeMultiLeaf(["Subject", "Diagnosis"], ["Subject", "Diagnosis"])
        assert "Subject" in str(err)
        assert "Diagnosis" in str(err)
        assert "row_per" in str(err)

    def test_ambiguous_path_message_includes_paths(self) -> None:
        err = DerivaMLDenormalizeAmbiguousPath(
            "Image", "Subject",
            [["Image", "Subject"], ["Image", "Observation", "Subject"]],
            ["Observation"],
        )
        assert "Image" in str(err)
        assert "Subject" in str(err)
        assert "Observation" in str(err)
        assert "include_tables" in str(err)
        assert "via" in str(err)

    def test_downstream_leaf_message_includes_tables(self) -> None:
        err = DerivaMLDenormalizeDownstreamLeaf("Subject", ["Image", "Diagnosis"])
        assert "Subject" in str(err)
        assert "Image" in str(err)
        assert "Diagnosis" in str(err)
        assert "aggregation" in str(err).lower()

    def test_unrelated_anchor_message_includes_ignore_option(self) -> None:
        err = DerivaMLDenormalizeUnrelatedAnchor(["Foo"], ["Image"])
        assert "Foo" in str(err)
        assert "ignore_unrelated_anchors" in str(err)


class TestDenormalizeExceptionAttributes:
    """Exceptions expose structured fields for programmatic consumers."""

    def test_multi_leaf_fields(self) -> None:
        err = DerivaMLDenormalizeMultiLeaf(["A", "B"], ["A", "B"])
        assert err.candidates == ["A", "B"]
        assert err.include_tables == ["A", "B"]

    def test_ambiguous_path_fields(self) -> None:
        err = DerivaMLDenormalizeAmbiguousPath(
            "X", "Y", [["X", "Y"], ["X", "Z", "Y"]], ["Z"]
        )
        assert err.from_table == "X"
        assert err.to_table == "Y"
        assert err.paths == [["X", "Y"], ["X", "Z", "Y"]]
        assert err.suggested_intermediates == ["Z"]

    def test_downstream_leaf_fields(self) -> None:
        err = DerivaMLDenormalizeDownstreamLeaf("Subject", ["Image"])
        assert err.row_per == "Subject"
        assert err.downstream_tables == ["Image"]

    def test_unrelated_anchor_fields(self) -> None:
        err = DerivaMLDenormalizeUnrelatedAnchor(["Foo"], ["Image"])
        assert err.unrelated_tables == ["Foo"]
        assert err.include_tables == ["Image"]
```

- [ ] **Step 4: Run the exception tests.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/model/test_exceptions.py -v
```

Expected: all PASS.

- [ ] **Step 5: Lint + commit.**

```bash
uv run ruff format src/deriva_ml/core/exceptions.py tests/model/test_exceptions.py
uv run ruff check src/deriva_ml/core/exceptions.py
git add src/deriva_ml/core/exceptions.py tests/model/test_exceptions.py
git commit -m "feat(denormalize): add denormalize-specific exception hierarchy"
```

---

## Task 2: Implement the new planner rules in `model/catalog.py`

**Files:**
- Modify: `src/deriva_ml/model/catalog.py` (`_prepare_wide_table`, `_build_join_tree`, helpers).
- Create: `tests/local_db/test_planner_rules.py`.

This is the largest single task because the planner is the semantic core. The rules to implement:

- **Rule 2 (sink-finding):** Given `include_tables ∪ via`, find the unique sink (table with no outbound FK to any other table in the set via transparent-association links). Multiple sinks → `DerivaMLDenormalizeMultiLeaf`. Zero sinks (cycle) → `DerivaMLDenormalizeNoSink`.
- **Rule 5 (downstream-leaf detection):** If `row_per` is explicit and any table in `include_tables` is downstream of it, raise `DerivaMLDenormalizeDownstreamLeaf`.
- **Rule 6 (path ambiguity):** For every pair (`row_per`, T) where T ∈ `include_tables ∪ via`, enumerate all simple paths through the FK graph (with association tables as transparent edges). If >1 path, raise `DerivaMLDenormalizeAmbiguousPath` with the path list and suggested intermediates (tables in at least one path but not in `include_tables`).
- **Association tables as transparent intermediates:** pure association tables on a required chain between two requested tables are joined through without contributing columns.

### DRY strategy for this task

The existing planner (`_build_join_tree` at `catalog.py:632`, `_schema_to_paths` at
`catalog.py:1120`) already does 80% of what these rules need. The task is to:

1. **Extract** two nested functions so they can be reused:
   - `_is_likely_association` (nested in `_build_join_tree` at lines 727-751)
     → `DerivaModel._is_association_table`
   - `find_arcs` (nested in `_schema_to_paths` at lines 1173-1188)
     → `DerivaModel._fk_neighbors`

2. **Compose** the new helpers (`_outbound_reachable`, `_enumerate_paths`,
   `_find_sinks`, `_determine_row_per`, `_find_path_ambiguities`) by calling
   the extracted + existing routines. No new FK traversal code.

3. **Extend** `_prepare_wide_table` with the three guard calls
   (`_determine_row_per`, `_find_path_ambiguities`) in front of the existing
   tree-building logic. The existing path-selection / flattening is preserved
   for the legacy `_denormalize_impl` code path.

- [ ] **Step 1: Write failing planner unit tests.**

Create `tests/local_db/test_planner_rules.py`. These tests use a canned ERMrest `Model` from the existing `denorm_deriva_model` fixture in `tests/local_db/conftest.py`. No live catalog.

```python
"""Unit tests for _prepare_wide_table / _build_join_tree planner rules.

These tests verify the semantic rules independently of any catalog fetching
or SQL execution. They check the pure model-analysis behavior:

- Rule 2: sink-finding for row_per auto-inference
- Rule 5: downstream-leaf rejection when row_per is explicit
- Rule 6: per-pair path ambiguity detection
"""

from __future__ import annotations

from typing import Any

import pytest

from deriva_ml.core.exceptions import (
    DerivaMLDenormalizeAmbiguousPath,
    DerivaMLDenormalizeDownstreamLeaf,
    DerivaMLDenormalizeMultiLeaf,
)


class TestSinkFinding:
    """Rule 2: row_per is the unique sink in the FK subgraph on include_tables."""

    def test_linear_chain_sink(self, denorm_deriva_model) -> None:
        """Subject ← Observation ← Image: Image is the sink."""
        # Observation in canned model — check denorm_deriva_model actually has it.
        # If not, skip until test fixture is extended in Task 8.
        # For now, this test documents expected behavior; extend fixture as needed.
        model = denorm_deriva_model
        if "Observation" not in model.name_to_schema_and_table:
            pytest.skip("Canned model needs Observation/Image chain fixture")

        sinks = model._find_sinks(
            include_tables=["Subject", "Observation", "Image"],
            via=[],
        )
        assert sinks == ["Image"], f"expected ['Image'], got {sinks}"

    def test_single_sink_simple(self, denorm_deriva_model) -> None:
        """Image points to Subject. Image is the sink."""
        model = denorm_deriva_model
        sinks = model._find_sinks(include_tables=["Subject", "Image"], via=[])
        assert sinks == ["Image"]

    def test_multi_leaf_raises(self, denorm_deriva_model) -> None:
        """If two requested tables have no FK between them, both are sinks."""
        model = denorm_deriva_model
        # "Dataset" and "Subject" don't point to each other in canned schema.
        sinks = model._find_sinks(include_tables=["Dataset", "Subject"], via=[])
        assert len(sinks) >= 2
        assert "Dataset" in sinks or "Subject" in sinks
        # And _determine_row_per raises multi-leaf
        with pytest.raises(DerivaMLDenormalizeMultiLeaf) as excinfo:
            model._determine_row_per(
                include_tables=["Dataset", "Subject"], via=[], row_per=None
            )
        assert "Dataset" in str(excinfo.value)
        assert "Subject" in str(excinfo.value)


class TestDownstreamLeafRejection:
    """Rule 5: explicit row_per with downstream table in include_tables → error."""

    def test_downstream_leaf_rejected(self, denorm_deriva_model) -> None:
        """row_per=Subject with Image (downstream) → error."""
        model = denorm_deriva_model
        with pytest.raises(DerivaMLDenormalizeDownstreamLeaf) as excinfo:
            model._determine_row_per(
                include_tables=["Subject", "Image"],
                via=[],
                row_per="Subject",
            )
        assert "Subject" in str(excinfo.value)
        assert "Image" in str(excinfo.value)

    def test_downstream_leaf_accepted_if_no_downstream(
        self, denorm_deriva_model
    ) -> None:
        """row_per=Image is fine because Image IS the sink (nothing downstream)."""
        model = denorm_deriva_model
        result = model._determine_row_per(
            include_tables=["Subject", "Image"],
            via=[],
            row_per="Image",
        )
        assert result == "Image"


class TestPathAmbiguity:
    """Rule 6: multiple FK paths between row_per and a requested table → error."""

    def test_no_ambiguity_single_path(self, denorm_deriva_model) -> None:
        """Simple chain: no ambiguity."""
        model = denorm_deriva_model
        # Image → Subject (direct only in canned schema). No ambiguity.
        result = model._find_path_ambiguities(
            row_per="Image",
            include_tables=["Image", "Subject"],
            via=[],
        )
        assert result == []

    def test_ambiguity_raises(self, denorm_deriva_model) -> None:
        """If canned schema has diamond Image→Subject, raise."""
        model = denorm_deriva_model
        # This test requires a diamond schema (Image→Subject direct and
        # Image→Observation→Subject). The canned model may need extension.
        if "Observation" not in model.name_to_schema_and_table:
            pytest.skip("Canned model needs diamond fixture for this test")
        # When both paths exist and Observation is NOT in include_tables:
        result = model._find_path_ambiguities(
            row_per="Image",
            include_tables=["Image", "Subject"],
            via=[],
        )
        # Result is a list of ambiguities; expect at least one
        assert len(result) >= 1
        amb = result[0]
        assert amb["from_table"] == "Image"
        assert amb["to_table"] == "Subject"
        assert len(amb["paths"]) >= 2

    def test_ambiguity_resolved_by_intermediate(self, denorm_deriva_model) -> None:
        """Including the intermediate in include_tables removes ambiguity."""
        model = denorm_deriva_model
        if "Observation" not in model.name_to_schema_and_table:
            pytest.skip("Canned model needs diamond fixture for this test")
        result = model._find_path_ambiguities(
            row_per="Image",
            include_tables=["Image", "Observation", "Subject"],
            via=[],
        )
        # With Observation in include_tables, the multi-hop is the only valid path
        assert result == []

    def test_ambiguity_resolved_by_via(self, denorm_deriva_model) -> None:
        """via=[Observation] removes ambiguity without adding Observation columns."""
        model = denorm_deriva_model
        if "Observation" not in model.name_to_schema_and_table:
            pytest.skip("Canned model needs diamond fixture for this test")
        result = model._find_path_ambiguities(
            row_per="Image",
            include_tables=["Image", "Subject"],
            via=["Observation"],
        )
        assert result == []
```

- [ ] **Step 2: Run the new tests — expect failures.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_planner_rules.py -v
```

Expected: most fail with `AttributeError` because `_find_sinks`, `_determine_row_per`, `_find_path_ambiguities` don't exist yet. Some may skip (diamond schema not yet in fixture).

- [ ] **Step 3: Read the current `_prepare_wide_table` and `_build_join_tree` to understand the existing structure.**

Read `src/deriva_ml/model/catalog.py:887-1100` for `_prepare_wide_table` and `src/deriva_ml/model/catalog.py:633-830` for `_build_join_tree`. Note the existing single-method interface.

- [ ] **Step 4a: Extract the two nested helpers to module-level methods on `DerivaModel`.**

These are currently nested functions that cannot be reused by the new planner code. Promote each to a method on `DerivaModel`.

**First**, extract `_is_likely_association` from `_build_join_tree` (currently at `catalog.py:727-751`). Replace the nested `def _is_likely_association(tbl)` with a call to the new method, and add the method at module-level alongside other `DerivaModel` helpers:

```python
def _is_association_table(self, name_or_table: str | Table) -> bool:
    """Check if a table is a pure association (M:N link) table.

    An association table has only system columns (RID/RCT/RMT/RCB/RMB)
    plus exactly two domain FK columns. ERMrest's built-in
    ``Table.is_association()`` counts system FKs (RCB/RMB →
    ERMrest_Client), so we use our own check that ignores them.

    Previously nested inside ``_build_join_tree``; promoted so that
    denormalization planning can also use it.

    Args:
        name_or_table: table name (looked up via :meth:`name_to_table`) or
            a :class:`Table` instance.

    Returns:
        True if the table is a pure association table.
    """
    system_cols = {"RID", "RCT", "RMT", "RCB", "RMB"}
    try:
        tbl = (
            name_or_table
            if hasattr(name_or_table, "foreign_keys")
            else self.name_to_table(name_or_table)
        )
        cols = {c.name for c in tbl.columns}
        fks = list(tbl.foreign_keys)
        domain_fks = [
            fk for fk in fks
            if fk.pk_table.name not in ("ERMrest_Client", "ERMrest_Group")
        ]
        fk_col_names: set[str] = set()
        for fk in domain_fks:
            for col in fk.columns:
                fk_col_names.add(col.name if hasattr(col, "name") else str(col))
        user_cols = cols - system_cols - fk_col_names
        return len(domain_fks) == 2 and len(user_cols) == 0
    except Exception:
        return False
```

Then in `_build_join_tree`, replace the nested `def _is_likely_association(tbl): ...` and the `_is_likely_association(tbl)` call site with:

```python
# (delete the nested def here — it's now self._is_association_table)
```

and update the call site in `_intermediates_covered`:

```python
if tbl is not None and self._is_association_table(tbl):
```

**Second**, extract `find_arcs` from `_schema_to_paths` (currently at `catalog.py:1173-1188`). The new method follows both outbound and inbound FKs, returning reachable tables deduplicated by target.

```python
def _fk_neighbors(self, table: str | Table) -> set[Table]:
    """Return FK-neighbor tables of *table* (outbound + inbound, deduplicated).

    Follows both ``table.foreign_keys`` (outbound) and
    ``table.referenced_by`` (inbound), filters to valid schemas
    (``domain_schemas ∪ {ml_schema}``), and deduplicates multi-FK
    targets — if two FKs both point to Foo, Foo appears once.

    Previously nested inside ``_schema_to_paths``; promoted so that
    denormalization planning can also use it as the FK-traversal primitive.

    Args:
        table: table name or Table instance.

    Returns:
        Set of :class:`Table` objects reachable from *table* via one FK arc.
    """
    tbl = table if hasattr(table, "foreign_keys") else self.name_to_table(table)
    valid_schemas = self.domain_schemas | {self.ml_schema}
    arc_list = (
        [fk.pk_table for fk in tbl.foreign_keys]
        + [fk.table for fk in tbl.referenced_by]
    )
    arc_list = [t for t in arc_list if t.schema.name in valid_schemas]
    # Deduplicate: when multiple FKs point to the same target table, keep
    # only one. Downstream code handles FK selection via _table_relationship.
    seen: set[Table] = set()
    deduped: list[Table] = []
    for t in arc_list:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return set(deduped)
```

In `_schema_to_paths`, replace the nested `def find_arcs(table)` and the `find_arcs(root)` call with:

```python
# (delete the nested def here — it's now self._fk_neighbors)
for child in self._fk_neighbors(root):
```

- [ ] **Step 4b: Add the new planner helpers — each composes the extracted primitives.**

These helpers are all thin compositions of `_fk_neighbors`, `_is_association_table`, `_schema_to_paths`, and `_table_relationship`. No new FK traversal — just filter/orchestrate calls into the existing routines.

Add these methods to `DerivaModel` (in `src/deriva_ml/model/catalog.py`, near `_prepare_wide_table` around line 887):

```python
def _find_sinks(
    self,
    include_tables: list[str],
    via: list[str] | None = None,
) -> list[str]:
    """Find sinks in the FK subgraph on include_tables ∪ via.

    A sink is a table in ``include_tables`` with no outbound FK to any
    other table in the set, considering association tables not in the set
    as transparent edges.

    Composes :meth:`_outbound_reachable` — does not traverse FKs itself.

    Args:
        include_tables: tables whose FK edges are considered.
        via: additional tables that are part of the subgraph (routing only).

    Returns:
        List of sink table names, sorted alphabetically. Empty if a cycle
        in the subgraph means no sink exists.
    """
    via = via or []
    all_tables = set(include_tables) | set(via)
    return sorted(
        t for t in all_tables
        if t in include_tables
        and not (self._outbound_reachable(t, all_tables) - {t})
    )


def _outbound_reachable(
    self,
    from_table: str,
    tables_in_set: set[str],
) -> set[str]:
    """Tables in ``tables_in_set`` reachable from ``from_table`` via FKs.

    BFS over :meth:`_fk_neighbors` — association tables NOT in
    ``tables_in_set`` are transparent hops (we traverse past them to see
    the tables they link). Tables in ``tables_in_set`` are recorded.

    Args:
        from_table: starting table.
        tables_in_set: the subgraph — only edges to/from these tables count
            as "destinations." Association tables outside this set are
            transparent.

    Returns:
        Set of names in ``tables_in_set`` reachable from from_table
        (excluding from_table itself).
    """
    seen: set[str] = set()
    stack: list[str] = [from_table]
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        try:
            tbl = self.name_to_table(t)
        except Exception:
            continue
        # Use the extracted _fk_neighbors primitive — no direct FK walking here.
        for neighbor in self._fk_neighbors(t):
            target_name = neighbor.name
            if target_name == from_table:
                continue
            if target_name in tables_in_set:
                seen.add(target_name)
                # Continue through it only if association — so we can see
                # past it to further requested targets.
                if self._is_association_table(neighbor):
                    stack.append(target_name)
            elif self._is_association_table(neighbor):
                # Transparent hop through a non-requested association.
                stack.append(target_name)
            # else: non-requested, non-association — dead end
    seen.discard(from_table)
    return {t for t in seen if t in tables_in_set and t != from_table}


# Note: _is_association_table is added in Step 4a above (extracted from
# _build_join_tree). Do NOT re-define it here.


def _determine_row_per(
    self,
    include_tables: list[str],
    via: list[str] | None,
    row_per: str | None,
) -> str:
    """Resolve the row_per table per Rule 2 and Rule 5.

    - If ``row_per`` is explicit, validate it's in ``include_tables`` and
      that no table in ``include_tables`` is downstream of it. Raise
      :class:`DerivaMLDenormalizeDownstreamLeaf` otherwise.
    - If ``row_per`` is None, auto-infer by finding sinks. Exactly one sink
      → return it. Zero → :class:`DerivaMLDenormalizeNoSink`. Multiple →
      :class:`DerivaMLDenormalizeMultiLeaf`.

    Returns:
        The resolved ``row_per`` table name.

    Raises:
        DerivaMLDenormalizeMultiLeaf: auto-inference finds multiple sinks.
        DerivaMLDenormalizeNoSink: no sink (cycle).
        DerivaMLDenormalizeDownstreamLeaf: explicit row_per with downstream
            table in include_tables.
    """
    from deriva_ml.core.exceptions import (
        DerivaMLDenormalizeDownstreamLeaf,
        DerivaMLDenormalizeMultiLeaf,
        DerivaMLDenormalizeNoSink,
    )

    via = via or []
    all_tables = set(include_tables) | set(via)

    if row_per is not None:
        if row_per not in include_tables:
            raise ValueError(
                f"row_per={row_per!r} must be in include_tables={include_tables}"
            )
        # Rule 5: check no include_tables entry is downstream of row_per.
        downstream = self._outbound_reachable(row_per, all_tables)
        downstream_in_inc = [t for t in include_tables if t in downstream and t != row_per]
        if downstream_in_inc:
            raise DerivaMLDenormalizeDownstreamLeaf(
                row_per=row_per,
                downstream_tables=sorted(downstream_in_inc),
            )
        return row_per

    # Auto-infer via sink-finding.
    sinks = self._find_sinks(include_tables, via)
    if not sinks:
        raise DerivaMLDenormalizeNoSink(
            f"No sink found in include_tables={include_tables}. "
            f"The FK subgraph may contain a cycle."
        )
    if len(sinks) > 1:
        raise DerivaMLDenormalizeMultiLeaf(
            candidates=sinks,
            include_tables=list(include_tables),
        )
    return sinks[0]


def _find_path_ambiguities(
    self,
    row_per: str,
    include_tables: list[str],
    via: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Enumerate path ambiguities per Rule 6.

    For each T in include_tables ∪ via (T != row_per), enumerate all
    simple FK paths between T and row_per, considering association tables
    as transparent edges. If >1 distinct path, collect an ambiguity entry.

    Args:
        row_per: the leaf table.
        include_tables: tables whose paths to row_per are checked.
        via: additional tables whose paths are checked (but their columns
            aren't in the output).

    Returns:
        List of ambiguity dicts. Each dict has:
          - from_table: row_per
          - to_table: the T with multiple paths
          - paths: list of path lists (each path a list of table names)
          - suggested_intermediates: tables in at least one path but not
            in include_tables
    """
    via = via or []
    all_tables = set(include_tables) | set(via)
    ambiguities: list[dict[str, Any]] = []

    for t in all_tables:
        if t == row_per:
            continue
        paths = self._enumerate_paths(row_per, t, all_tables)
        # Deduplicate by tuple signature.
        unique = list({tuple(p): p for p in paths}.values())
        if len(unique) > 1:
            # Tables in at least one path but not in include_tables
            all_intermediates: set[str] = set()
            for p in unique:
                for node in p[1:-1]:  # skip endpoints
                    if node not in include_tables:
                        all_intermediates.add(node)
            ambiguities.append(
                {
                    "from_table": row_per,
                    "to_table": t,
                    "paths": unique,
                    "suggested_intermediates": sorted(all_intermediates),
                }
            )
    return ambiguities


def _enumerate_paths(
    self,
    from_table: str,
    to_table: str,
    tables_in_set: set[str],
    max_depth: int = 6,
) -> list[list[str]]:
    """Enumerate simple FK paths from ``from_table`` to ``to_table``.

    **Implementation:** delegates the FK-graph DFS to the existing
    :meth:`_schema_to_paths` (which handles cycle detection, vocabulary
    termination, schema filtering, and multi-FK deduplication) and then
    filters / trims the result. Do NOT write a fresh DFS here.

    Association tables not in ``tables_in_set`` are kept as transparent
    intermediates in the returned path so callers can report them as
    suggestions.

    Args:
        from_table: path start.
        to_table: path end.
        tables_in_set: ``include_tables ∪ via``. Paths passing through
            tables NOT in this set are accepted only if every intermediate
            is a pure association table (transparent hop).
        max_depth: forwarded to ``_schema_to_paths`` as safety cap.

    Returns:
        List of paths, each a list of table-name strings from
        ``from_table`` to ``to_table``.
    """
    from_tbl = self.name_to_table(from_table)

    # Get every FK path from from_table (and its prefixes) via the existing
    # authoritative enumerator.
    all_paths = self._schema_to_paths(root=from_tbl, max_depth=max_depth)

    result: list[list[str]] = []
    for path in all_paths:
        if not path or path[-1].name != to_table:
            continue
        names = [t.name for t in path]
        if names[0] != from_table:
            continue
        # Accept if every intermediate (between endpoints) is either in the
        # set or a transparent association table.
        ok = True
        for mid in names[1:-1]:
            if mid in tables_in_set:
                continue
            if self._is_association_table(mid):
                continue
            ok = False
            break
        if ok:
            result.append(names)
    return result
```

- [ ] **Step 5: Run the planner tests.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_planner_rules.py -v
```

Expected: tests that don't require the diamond fixture pass. Tests that need the diamond fixture skip with a message.

- [ ] **Step 6: Extend the canned model fixture to support diamond tests.**

Edit `tests/local_db/conftest.py`. In the canned schema (look for the fixture that creates the `isa.Image` table with the Subject FK), add:

1. A new `isa.Observation` table with columns `RID`, `Subject` (FK to Subject), `Date`.
2. A new FK on `isa.Image`: `Observation` (nullable FK to Observation). This creates the diamond.

Look at the existing fixture definition around the `isa` / `deriva-ml` schema setup in `conftest.py` and extend it.

After editing, re-run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_planner_rules.py -v
```

All tests in the file should now PASS (no skips).

- [ ] **Step 7: Wire the new helpers into `_prepare_wide_table`.**

Update `_prepare_wide_table` in `src/deriva_ml/model/catalog.py` to:

1. Accept new `row_per` and `via` parameters:

```python
def _prepare_wide_table(
    self,
    dataset,
    dataset_rid: RID,
    include_tables: list[str],
    *,
    row_per: str | None = None,
    via: list[str] | None = None,
) -> tuple[dict[str, Any], list[tuple], bool]:
    """..."""
```

2. Before any existing logic, call `self._determine_row_per(...)` and `self._find_path_ambiguities(...)`. If ambiguities are non-empty, raise the first one as a `DerivaMLDenormalizeAmbiguousPath`.

3. Use the resolved `row_per` as the element type for the JoinTree (instead of the current "paths_by_element" discovery).

Keep the rest of the implementation (join-tree building, flattening) — it still produces the same output shape. The new logic is the guard in front.

- [ ] **Step 8: Run all existing planner tests + new tests.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/test_manifest.py tests/dataset/test_denormalize.py -x --tb=short 2>&1 | tail -40
```

Expected: the new planner tests pass; integration tests in `tests/dataset/test_denormalize.py` that used to silently select a path now raise. This is expected — they will be updated in Task 9 below.

Note: don't fix the integration test failures yet. Just confirm the planner-level unit tests pass.

- [ ] **Step 9: Lint + commit.**

```bash
uv run ruff format src/deriva_ml/model/catalog.py tests/local_db/test_planner_rules.py tests/local_db/conftest.py
uv run ruff check src/deriva_ml/model/catalog.py
git add src/deriva_ml/model/catalog.py tests/local_db/test_planner_rules.py tests/local_db/conftest.py
git commit -m "feat(denormalize): implement Rule 2/5/6 planner helpers with unit tests

Extract two nested functions to module-level methods on DerivaModel:
  - _is_likely_association (was nested in _build_join_tree)
      → DerivaModel._is_association_table
  - find_arcs (was nested in _schema_to_paths)
      → DerivaModel._fk_neighbors
Both _build_join_tree and _schema_to_paths now call the extracted methods.

Add thin composing helpers that reuse the extracted primitives and the
existing _schema_to_paths / _table_relationship:
  - _outbound_reachable (BFS on _fk_neighbors)
  - _enumerate_paths (filter on _schema_to_paths output)
  - _find_sinks / _determine_row_per / _find_path_ambiguities

_prepare_wide_table accepts row_per and via kwargs and invokes the new
rules before the existing tree-building flow.

Extends canned test fixture with Observation table to support diamond-path
test coverage.
"
```

---

## Task 3: Rename `denormalize()` → `_denormalize_impl()`

The existing free `denormalize()` function becomes the private implementation detail called by the `Denormalizer` class. Rename it, update callers.

**Files:**
- Modify: `src/deriva_ml/local_db/denormalize.py`
- Modify: `src/deriva_ml/local_db/__init__.py`
- Modify: `src/deriva_ml/local_db/workspace.py:452` (the `cache_denormalized` method calls it)
- Modify: `src/deriva_ml/dataset/dataset.py` (the delegate methods call it)
- Modify: `src/deriva_ml/dataset/dataset_bag.py` (same)
- Rename: `tests/local_db/test_denormalize.py` → `tests/local_db/test_denormalize_impl.py`

- [ ] **Step 1: Rename the function in `denormalize.py`.**

In `src/deriva_ml/local_db/denormalize.py`, find `def denormalize(` and rename to `def _denormalize_impl(`. Keep the signature exactly the same. Also update the module docstring to reflect that the public API is now `Denormalizer`.

- [ ] **Step 2: Update in-repo callers.**

Find every caller:

```bash
grep -rn "from deriva_ml.local_db.denormalize import denormalize\|from deriva_ml.local_db import denormalize\|denormalize(" src/deriva_ml/ --include="*.py" | grep -v __pycache__ | grep -v "def denormalize"
```

Update each import and each call to use `_denormalize_impl`. Expected locations:

- `src/deriva_ml/local_db/workspace.py` — `cache_denormalized` method.
- `src/deriva_ml/dataset/dataset.py` — delegate methods.
- `src/deriva_ml/dataset/dataset_bag.py` — delegate methods.

Each call site just needs the import line updated and the call changed.

- [ ] **Step 3: Remove `denormalize` from package exports.**

In `src/deriva_ml/local_db/__init__.py`, remove the `denormalize` name from the `__all__` list and from the imports block. Keep `DenormalizeResult` exported (it's a public return type).

- [ ] **Step 4: Rename the test file.**

```bash
git mv tests/local_db/test_denormalize.py tests/local_db/test_denormalize_impl.py
```

Update the module docstring at the top of the renamed file to say "tests for `_denormalize_impl` — the low-level primitive called by `Denormalizer`."

- [ ] **Step 5: Update the test file's imports and function references.**

```bash
sed -i '' 's/from deriva_ml.local_db.denormalize import DenormalizeResult, denormalize/from deriva_ml.local_db.denormalize import DenormalizeResult, _denormalize_impl/' tests/local_db/test_denormalize_impl.py
sed -i '' 's/\<denormalize(/_denormalize_impl(/g' tests/local_db/test_denormalize_impl.py
```

Verify the edits:

```bash
grep -n "denormalize\b\|_denormalize_impl" tests/local_db/test_denormalize_impl.py | head -10
```

Expected: all references are `_denormalize_impl`, no bare `denormalize(`.

- [ ] **Step 6: Run the renamed tests.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_denormalize_impl.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Run the full local_db test suite.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/test_manifest.py --tb=short 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 8: Lint + commit.**

```bash
uv run ruff format src/deriva_ml/ tests/local_db/
uv run ruff check src/deriva_ml/
git add -A
git commit -m "refactor(denormalize): rename denormalize() → _denormalize_impl()"
```

---

## Task 4: Create the `Denormalizer` class skeleton + core `as_dataframe`, `as_dict`, `columns`

Introduce the class in a new module `denormalizer.py` to avoid making `denormalize.py` too large. The class constructor accepts a `DatasetLike` for the common case; `from_rids` is added in a later task.

**Files:**
- Create: `src/deriva_ml/local_db/denormalizer.py`
- Create: `tests/local_db/test_denormalizer.py`

**DRY reminders (see Reuse inventory above):**
- `as_dataframe` / `as_dict` delegate to `_denormalize_impl` — do NOT
  re-implement the SQL execution loop. `_denormalize_impl` already handles
  the JOIN construction, row emission with NULL-init, and schema-prefixed
  column names via `denormalize_column_name`.
- `columns` invokes `self._model._prepare_wide_table(...)` (model-only
  planning, no data fetch) and formats with `denormalize_column_name`.
  Do NOT invent a new schema-prefixing scheme.
- Anchor retrieval uses `dataset.list_dataset_members(recurse=True)` —
  `_anchors_as_dict` in Task 6 wraps this.
- Study `Dataset.denormalize_as_dataframe` (`dataset/dataset.py:974`)
  before writing `Denormalizer.as_dataframe` — the public shape (column
  prefixing, row dicts, DataFrame construction) is already defined there.
  The new class preserves that shape so downstream tools keep working.

- [ ] **Step 1: Write failing tests for the class skeleton.**

Create `tests/local_db/test_denormalizer.py`:

```python
"""Unit tests for the Denormalizer class public API.

These tests use the canned bag-model fixtures from conftest.py — no live
catalog required. They verify the class wraps _denormalize_impl correctly
with the new rules applied.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from deriva_ml.core.exceptions import (
    DerivaMLDenormalizeAmbiguousPath,
    DerivaMLDenormalizeDownstreamLeaf,
    DerivaMLDenormalizeMultiLeaf,
)
from deriva_ml.local_db.denormalizer import Denormalizer


class TestDenormalizerConstruction:
    """Denormalizer(dataset) derives catalog/workspace/model from the dataset."""

    def test_construct_from_dataset_like(self, populated_denorm) -> None:
        """Minimal construction: wrap a dataset-like object."""
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        assert d is not None


class TestAsDataframe:
    """Denormalizer.as_dataframe returns a pd.DataFrame with expected shape."""

    def test_simple_star_schema(self, populated_denorm) -> None:
        """One row per Image, Subject columns hoisted."""
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        df = d.as_dataframe(["Image", "Subject"])
        assert isinstance(df, pd.DataFrame)
        # 3 Images in the fixture (one with NULL Subject = LEFT JOIN preserved).
        assert len(df) == 3
        assert any(c.startswith("Image.") for c in df.columns)
        assert any(c.startswith("Subject.") for c in df.columns)

    def test_empty_dataset(self, populated_denorm) -> None:
        """Nonexistent dataset RID returns empty DataFrame with correct columns."""
        ds = _FakeDataset(populated_denorm, dataset_rid="NO-SUCH-DS")
        d = Denormalizer(ds)
        df = d.as_dataframe(["Image", "Subject"])
        assert len(df) == 0
        assert len(df.columns) > 0  # schema preserved


class TestAsDict:
    """Denormalizer.as_dict streams rows as dicts."""

    def test_yields_dicts(self, populated_denorm) -> None:
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        rows = list(d.as_dict(["Image", "Subject"]))
        assert len(rows) == 3
        for r in rows:
            assert isinstance(r, dict)


class TestColumns:
    """Denormalizer.columns previews column names and types — no data fetch."""

    def test_columns_returns_tuples(self, populated_denorm) -> None:
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        cols = d.columns(["Image", "Subject"])
        assert isinstance(cols, list)
        for entry in cols:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            name, typ = entry
            assert isinstance(name, str)
            assert isinstance(typ, str)


class TestRowPerAutoInference:
    """Verify Rule 2 auto-inference through the Denormalizer."""

    def test_image_is_sink(self, populated_denorm) -> None:
        """include_tables=[Subject, Image] → row_per auto = Image."""
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        plan = d.describe(["Subject", "Image"])
        assert plan["row_per"] == "Image"
        assert plan["row_per_source"] == "auto-inferred"


class TestExplicitRowPer:
    """Verify explicit row_per honored; Rule 5 downstream rejection."""

    def test_explicit_matching_auto(self, populated_denorm) -> None:
        """Explicit row_per=Image (same as auto) works."""
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        df = d.as_dataframe(["Image", "Subject"], row_per="Image")
        assert len(df) == 3

    def test_downstream_leaf_rejected(self, populated_denorm) -> None:
        """row_per=Subject with Image downstream → DerivaMLDenormalizeDownstreamLeaf."""
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        with pytest.raises(DerivaMLDenormalizeDownstreamLeaf):
            d.as_dataframe(["Image", "Subject"], row_per="Subject")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Minimal DatasetLike for Denormalizer construction in tests.

    Wraps the populated_denorm fixture with the interface bits the
    Denormalizer needs: model, engine, orm_resolver, dataset_rid,
    list_dataset_members, list_dataset_children.
    """

    def __init__(self, populated_denorm: dict[str, Any], dataset_rid: str | None = None):
        self._pd = populated_denorm
        self.dataset_rid = dataset_rid or populated_denorm["dataset_rid"]
        # Attributes the Denormalizer uses:
        self.model = populated_denorm["model"]
        self.engine = populated_denorm["local_schema"].engine
        self._orm_resolver = populated_denorm["local_schema"].get_orm_class

    # Attributes exposed as "ml instance" pseudo-shim for Denormalizer(ds)
    # construction: the class pulls workspace/catalog from ds. For unit
    # tests against the canned fixture, we supply only what's actually
    # dereferenced — see Denormalizer.__init__.
    @property
    def _ml_instance(self):
        return None  # sentinel: tests use the fixture's engine directly

    def list_dataset_members(self, **kwargs: Any) -> dict[str, list[dict]]:
        # All member types in the fixture
        return {
            "Image": [{"RID": r} for r in self._pd["image_rids"]],
            "Subject": [{"RID": r} for r in self._pd["subject_rids"]],
        }

    def list_dataset_children(self, **kwargs: Any) -> list:
        return []
```

Note: the `_FakeDataset` fixture is a workaround — once `Denormalizer(ds)` is fully implemented, we'll discover the real cleaner way. For now, the class's constructor needs enough flexibility to accept this shim. Document that the test-only shim is intentional and tests will be refactored when the integration is complete.

- [ ] **Step 2: Run tests — expect ImportError.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_denormalizer.py -v
```

Expected: `ImportError: cannot import name 'Denormalizer' from 'deriva_ml.local_db.denormalizer'` (module doesn't exist yet).

- [ ] **Step 3: Create the `Denormalizer` class.**

Create `src/deriva_ml/local_db/denormalizer.py`:

```python
"""Denormalizer — public API for producing wide tables from Deriva data.

Wraps the lower-level ``_denormalize_impl`` primitive in a class-based API
with support for auto-inferred ``row_per``, explicit ``via`` path routing,
orphan-row handling, and arbitrary RID anchor sets.

See ``docs/superpowers/specs/2026-04-17-denormalization-semantics-design.md``
for the semantic rules this class implements.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Generator

import pandas as pd

from deriva_ml.local_db.denormalize import DenormalizeResult, _denormalize_impl

if TYPE_CHECKING:
    from deriva_ml.interfaces import DatasetLike

logger = logging.getLogger(__name__)


class Denormalizer:
    """Produce wide-table denormalizations from Deriva datasets or anchor sets.

    Construction:
      - ``Denormalizer(dataset_like)`` — use the dataset's members as anchors
        and derive catalog/workspace/model from the dataset's bindings.
      - ``Denormalizer.from_rids(rids, ml=...)`` — arbitrary RID anchors
        (see classmethod; implemented in a later task).

    Methods:
      - :meth:`as_dataframe` — materialize as pd.DataFrame.
      - :meth:`as_dict` — stream rows as dicts.
      - :meth:`columns` — preview column schema without fetching.
      - :meth:`describe` — dry-run the call; returns planning metadata.
      - :meth:`list_paths` — describe the FK graph (for exploration).
    """

    def __init__(self, dataset: "DatasetLike") -> None:
        """Construct from a ``DatasetLike`` object.

        The dataset's members (recursively via ``list_dataset_members``)
        become the anchor set. The underlying model, engine, and
        orm_resolver are derived from the dataset.

        Args:
            dataset: A :class:`Dataset` or :class:`DatasetBag` (or any
                object satisfying the ``DatasetLike`` protocol plus the
                attributes ``model``, ``engine``, ``_orm_resolver``).
        """
        self._dataset = dataset
        self._dataset_rid = dataset.dataset_rid
        self._model = dataset.model
        # engine / orm_resolver / paged_client are extracted lazily so
        # test fixtures can inject their own.
        self._engine = getattr(dataset, "engine", None)
        self._orm_resolver = getattr(dataset, "_orm_resolver", None)
        if self._orm_resolver is None:
            # Fall back to model's get_orm_class_by_name if available
            gocbn = getattr(self._model, "get_orm_class_by_name", None)
            self._orm_resolver = gocbn

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def as_dataframe(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
        ignore_unrelated_anchors: bool = False,
    ) -> pd.DataFrame:
        """Materialize the denormalized table as a pandas DataFrame.

        Args:
            include_tables: Tables whose columns appear in the output.
                Also determines ``row_per`` unless overridden.
            row_per: Explicit leaf table. Must be in ``include_tables``.
                If None, auto-inferred (Rule 2).
            via: Tables forced into the join chain without contributing
                columns. Useful for disambiguating path ambiguity without
                cluttering the output.
            ignore_unrelated_anchors: If True, silently drop anchors whose
                table has no FK path to any requested table. Default False
                raises :class:`DerivaMLDenormalizeUnrelatedAnchor`.

        Returns:
            A :class:`pandas.DataFrame` with one row per ``row_per`` instance
            in scope. See the semantic rules (Rules 1–8 in the spec) for
            the full cardinality and column-projection semantics.
        """
        result = self._run(
            include_tables,
            row_per=row_per,
            via=via,
            ignore_unrelated_anchors=ignore_unrelated_anchors,
        )
        return result.to_dataframe()

    def as_dict(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
        ignore_unrelated_anchors: bool = False,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream the denormalized table row-by-row as dicts.

        Same semantics as :meth:`as_dataframe` but yields one dict per row.
        Use for large datasets where a full DataFrame won't fit in memory.
        """
        result = self._run(
            include_tables,
            row_per=row_per,
            via=via,
            ignore_unrelated_anchors=ignore_unrelated_anchors,
        )
        yield from result.iter_rows()

    def columns(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        """Preview (column_name, type_name) pairs for the denormalized table.

        Model-only — no data fetch, no catalog query. Runs the same path
        validation as :meth:`as_dataframe` so ambiguity errors surface
        here too.
        """
        from deriva_ml.model.catalog import denormalize_column_name

        # Invokes the planner; raises on ambiguity.
        element_tables, column_specs, multi_schema = self._model._prepare_wide_table(
            self._dataset,
            self._dataset_rid,
            list(include_tables),
            row_per=row_per,
            via=via,
        )
        return [
            (denormalize_column_name(s, t, c, multi_schema), tp)
            for s, t, c, tp in column_specs
        ]

    def describe(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return a planning-metadata dict describing what would happen.

        Stub implementation in this task — extended in Task 7 below.
        """
        # Minimal implementation: return just the row_per + columns.
        resolved_row_per = self._model._determine_row_per(
            include_tables=list(include_tables),
            via=list(via or []),
            row_per=row_per,
        )
        cols = self.columns(include_tables, row_per=row_per, via=via)
        return {
            "row_per": resolved_row_per,
            "row_per_source": "explicit" if row_per else "auto-inferred",
            "columns": cols,
            "include_tables": list(include_tables),
            "via": list(via or []),
            "ambiguities": [],
        }

    def list_paths(
        self,
        tables: list[str] | None = None,
    ) -> dict[str, Any]:
        """Describe the FK graph for exploration. Stub — filled in Task 8."""
        # Stub: full implementation in Task 8.
        return {
            "member_types": [],
            "reachable_tables": {},
            "association_tables": [],
            "feature_tables": [],
            "schema_paths": {},
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run(
        self,
        include_tables: list[str],
        *,
        row_per: str | None,
        via: list[str] | None,
        ignore_unrelated_anchors: bool,
    ) -> DenormalizeResult:
        """Dispatch to the low-level primitive."""
        # For now, delegate unchanged — Task 5+ wires row_per/via through.
        return _denormalize_impl(
            model=self._model,
            engine=self._engine,
            orm_resolver=self._orm_resolver,
            dataset_rid=self._dataset_rid,
            include_tables=list(include_tables),
            dataset=self._dataset,
            source="local",  # fixture tests pre-populate the DB
        )
```

- [ ] **Step 4: Run the tests.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_denormalizer.py -v
```

Expected: most pass. The `TestExplicitRowPer::test_downstream_leaf_rejected` and `TestRowPerAutoInference` tests depend on the planner passing `row_per` through correctly — they should pass because we added row_per to `_prepare_wide_table` in Task 2.

- [ ] **Step 5: Add `Denormalizer` to the package exports.**

In `src/deriva_ml/local_db/__init__.py`, add:

```python
from deriva_ml.local_db.denormalizer import Denormalizer
```

And add `"Denormalizer"` to `__all__`.

- [ ] **Step 6: Run all local_db tests.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/test_manifest.py -q --tb=short 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 7: Lint + commit.**

```bash
uv run ruff format src/deriva_ml/local_db/denormalizer.py tests/local_db/test_denormalizer.py src/deriva_ml/local_db/__init__.py
uv run ruff check src/deriva_ml/local_db/
git add src/deriva_ml/local_db/denormalizer.py tests/local_db/test_denormalizer.py src/deriva_ml/local_db/__init__.py
git commit -m "feat(denormalize): add Denormalizer class with as_dataframe/as_dict/columns"
```

---

## Task 5: Wire `row_per` and `via` through to `_denormalize_impl`

Currently `_denormalize_impl` accepts these kwargs but ignores `row_per`/`via` — Task 2 added the guards in the planner but the impl doesn't forward them. Connect the flow so explicit `row_per`/`via` affect the join plan and SQL.

**Files:**
- Modify: `src/deriva_ml/local_db/denormalize.py` (`_denormalize_impl`).
- Modify: `src/deriva_ml/local_db/denormalizer.py` (`_run` method).

- [ ] **Step 1: Update `_denormalize_impl` signature to accept `row_per` and `via`.**

In `src/deriva_ml/local_db/denormalize.py`, modify the signature:

```python
def _denormalize_impl(
    model: DerivaModel,
    engine: Engine,
    orm_resolver: Callable[[str], Any],
    dataset_rid: str,
    include_tables: list[str],
    dataset: Any = None,
    dataset_children_rids: list[str] | None = None,
    source: str = "local",
    paged_client: PagedClient | None = None,
    *,
    row_per: str | None = None,
    via: list[str] | None = None,
) -> DenormalizeResult:
```

Inside the body, pass `row_per=row_per, via=via` to the `model._prepare_wide_table(...)` call.

- [ ] **Step 2: Update `Denormalizer._run` to forward the params.**

In `src/deriva_ml/local_db/denormalizer.py`:

```python
def _run(
    self,
    include_tables: list[str],
    *,
    row_per: str | None,
    via: list[str] | None,
    ignore_unrelated_anchors: bool,
) -> DenormalizeResult:
    return _denormalize_impl(
        model=self._model,
        engine=self._engine,
        orm_resolver=self._orm_resolver,
        dataset_rid=self._dataset_rid,
        include_tables=list(include_tables),
        dataset=self._dataset,
        source="local",
        row_per=row_per,
        via=list(via) if via else None,
    )
```

- [ ] **Step 3: Add a test that explicit via= resolves an ambiguity.**

In `tests/local_db/test_denormalizer.py`, add:

```python
class TestViaParameter:
    """Verify via= is forwarded to the planner and resolves path ambiguity."""

    def test_via_resolves_diamond(self, populated_denorm) -> None:
        """Diamond schema: via=['Observation'] should prevent ambiguity error."""
        # This test requires the canned model to include the diamond fixture
        # from Task 2 step 6.
        if "Observation" not in populated_denorm["model"].name_to_schema_and_table:
            pytest.skip("Canned model needs diamond fixture (Task 2 step 6)")

        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        # Without via, diamond raises
        with pytest.raises(DerivaMLDenormalizeAmbiguousPath):
            d.as_dataframe(["Image", "Subject"])
        # With via, ambiguity resolved
        df = d.as_dataframe(["Image", "Subject"], via=["Observation"])
        assert isinstance(df, pd.DataFrame)
        # Observation columns should NOT be present
        assert not any(c.startswith("Observation.") for c in df.columns)
```

- [ ] **Step 4: Run tests.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_denormalizer.py tests/local_db/test_denormalize_impl.py tests/local_db/test_planner_rules.py -v --tb=short 2>&1 | tail -15
```

Expected: all pass.

- [ ] **Step 5: Lint + commit.**

```bash
uv run ruff format src/deriva_ml/local_db/
uv run ruff check src/deriva_ml/local_db/
git add src/deriva_ml/local_db/denormalize.py src/deriva_ml/local_db/denormalizer.py tests/local_db/test_denormalizer.py
git commit -m "feat(denormalize): wire row_per and via through Denormalizer to planner"
```

---

## Task 6: Anchor validation + unrelated-anchor rejection (Rule 7, Rule 8)

Implement the anchor-classification logic that rejects unrelated anchors unless `ignore_unrelated_anchors=True`. Also implement orphan-row emission (Rule 7 case 3) — upstream anchors with no reachable `row_per` produce LEFT-JOIN-style rows with `row_per`-side NULL.

**Files:**
- Modify: `src/deriva_ml/local_db/denormalizer.py` — add `_classify_anchors`, `_emit_orphan_rows`.

**DRY reminders (see Reuse inventory above):**
- `_classify_anchors` uses `self._model._outbound_reachable` — do NOT walk FKs here.
- `_emit_orphan_rows` uses `denormalize_column_name` for column labels — do NOT reinvent the `{Table}.{col}` convention.
- The NULL-init / LEFT-JOIN pattern is already established in
  `Dataset._denormalize_datapath` (`src/deriva_ml/dataset/dataset.py:941-970`).
  Read it before writing `_emit_orphan_rows` — your implementation follows
  the same shape (init everything to None, populate only the anchor's
  columns from its own row).
- The orphan anchor's own row is looked up from the local engine via the
  orm_resolver. The `_populate_from_catalog` helper in `denormalize.py:255`
  (and `PagedFetcher.fetch_by_rids` in `paged_fetcher.py:174`) is the
  catalog-side analog if source="catalog" — the existing
  `_denormalize_impl` already fetches these rows, so in practice the orm
  table is populated by the time `_emit_orphan_rows` runs.

- [ ] **Step 1: Write failing tests.**

Add to `tests/local_db/test_denormalizer.py`:

```python
class TestUnrelatedAnchors:
    """Rule 8: anchors with no FK path to include_tables → error by default."""

    def test_unrelated_anchor_rejected(self, populated_denorm) -> None:
        # A dataset whose members include an unrelated type.
        class _HeteroDataset(_FakeDataset):
            def list_dataset_members(self, **kwargs):
                return {
                    "Image": [{"RID": "IMG-1"}],
                    # Dataset_Version is in the ML schema but has no FK path
                    # to Subject alone. This is contrived, but exercises the rule.
                    "Dataset_Version": [{"RID": "DV-1"}],
                }

        ds = _HeteroDataset(populated_denorm)
        d = Denormalizer(ds)
        with pytest.raises(DerivaMLDenormalizeUnrelatedAnchor):
            d.as_dataframe(["Image", "Subject"])

    def test_unrelated_anchor_ignored_with_flag(self, populated_denorm) -> None:
        class _HeteroDataset(_FakeDataset):
            def list_dataset_members(self, **kwargs):
                return {
                    "Image": [{"RID": "IMG-1"}],
                    "Dataset_Version": [{"RID": "DV-1"}],
                }

        ds = _HeteroDataset(populated_denorm)
        d = Denormalizer(ds)
        # With the flag, no error
        df = d.as_dataframe(
            ["Image", "Subject"],
            ignore_unrelated_anchors=True,
        )
        assert isinstance(df, pd.DataFrame)


class TestOrphanRows:
    """Rule 7 case 3: upstream anchor with no row_per reachable → orphan row."""

    def test_orphan_subject_emits_row(self, populated_denorm) -> None:
        """Subject member with no Image in the dataset → one orphan row."""
        class _WithOrphan(_FakeDataset):
            def list_dataset_members(self, **kwargs):
                members = super().list_dataset_members()
                # Add an orphan Subject that no Image points to
                members["Subject"].append({"RID": "ORPHAN-SUBJ"})
                return members

        ds = _WithOrphan(populated_denorm)
        d = Denormalizer(ds)
        df = d.as_dataframe(["Image", "Subject"])
        # 3 Image rows + 1 orphan Subject row = 4
        # The orphan row has Image cols NULL and Subject cols populated.
        orphans = df[df["Image.RID"].isna()]
        assert len(orphans) == 1
        # The orphan's Subject.RID should be "ORPHAN-SUBJ".
        assert orphans.iloc[0]["Subject.RID"] == "ORPHAN-SUBJ"
```

Add import at top of the test file:

```python
from deriva_ml.core.exceptions import DerivaMLDenormalizeUnrelatedAnchor
```

- [ ] **Step 2: Run the tests — expect failures.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_denormalizer.py::TestUnrelatedAnchors tests/local_db/test_denormalizer.py::TestOrphanRows -v --tb=short
```

Expected: fails. Either:
- Unrelated anchors are silently included (producing empty result rows).
- Orphans are dropped.

- [ ] **Step 3: Add `_classify_anchors` method to `Denormalizer`.**

In `src/deriva_ml/local_db/denormalizer.py`, add:

```python
def _classify_anchors(
    self,
    anchors: dict[str, list[str]],
    *,
    include_tables: list[str],
    via: list[str],
    row_per: str,
    ignore_unrelated_anchors: bool,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    """Classify anchors by their relationship to row_per (Rule 7 + Rule 8).

    Args:
        anchors: table_name -> list of RIDs.
        include_tables, via, row_per: the planner inputs.
        ignore_unrelated_anchors: if False, raise on Rule-8 cases.

    Returns:
        Tuple of (scoping_anchors, orphan_anchors, ignored_anchors). Each
        is a table_name -> list of RIDs dict.

        - scoping_anchors: anchors that scope the main SQL query (their
          reachable row_per rows go in the output).
        - orphan_anchors: anchors whose table is in include_tables but can't
          reach row_per — they emit orphan rows.
        - ignored_anchors: anchors whose table isn't in include_tables and
          can't reach row_per — contribute nothing; only populated when
          ignore_unrelated_anchors=True.

    Raises:
        DerivaMLDenormalizeUnrelatedAnchor: if any anchor's table has no FK
            path to include_tables and ignore_unrelated_anchors=False.
    """
    from deriva_ml.core.exceptions import DerivaMLDenormalizeUnrelatedAnchor

    scoping: dict[str, list[str]] = {}
    orphans: dict[str, list[str]] = {}
    ignored: dict[str, list[str]] = {}
    unrelated: list[str] = []

    all_tables = set(include_tables) | set(via)

    for table, rids in anchors.items():
        if table == row_per:
            scoping[table] = list(rids)
            continue
        # Does this table have any FK path into the subgraph?
        reachable = self._model._outbound_reachable(table, all_tables | {table})
        if row_per in reachable:
            # Upstream of row_per and reaches it → scoping
            scoping[table] = list(rids)
        elif table in include_tables:
            # Upstream of row_per (in the subgraph) but no row_per reachable
            # from this specific anchor — we emit an orphan row per anchor.
            # (Actual reachability-from-the-anchor is per-RID, evaluated in
            # _emit_orphan_rows.)
            orphans[table] = list(rids)
        else:
            # Anchor's table is not in include_tables — and has no path to
            # row_per. It would contribute nothing.
            unrelated.append(table)

    if unrelated and not ignore_unrelated_anchors:
        raise DerivaMLDenormalizeUnrelatedAnchor(
            unrelated_tables=sorted(set(unrelated)),
            include_tables=list(include_tables),
        )
    if unrelated:
        for table in unrelated:
            ignored[table] = list(anchors[table])

    return scoping, orphans, ignored
```

- [ ] **Step 4: Wire `_classify_anchors` into `_run`.**

Modify `Denormalizer._run`:

```python
def _run(
    self,
    include_tables: list[str],
    *,
    row_per: str | None,
    via: list[str] | None,
    ignore_unrelated_anchors: bool,
) -> DenormalizeResult:
    # Step 1: planner decisions (row_per, ambiguity checks)
    resolved_row_per = self._model._determine_row_per(
        include_tables=list(include_tables),
        via=list(via or []),
        row_per=row_per,
    )
    ambiguities = self._model._find_path_ambiguities(
        row_per=resolved_row_per,
        include_tables=list(include_tables),
        via=list(via or []),
    )
    if ambiguities:
        from deriva_ml.core.exceptions import DerivaMLDenormalizeAmbiguousPath

        a = ambiguities[0]
        raise DerivaMLDenormalizeAmbiguousPath(
            from_table=a["from_table"],
            to_table=a["to_table"],
            paths=a["paths"],
            suggested_intermediates=a["suggested_intermediates"],
        )

    # Step 2: anchor classification (Rule 7, Rule 8)
    anchors = self._anchors_as_dict()
    scoping, orphans, ignored = self._classify_anchors(
        anchors,
        include_tables=list(include_tables),
        via=list(via or []),
        row_per=resolved_row_per,
        ignore_unrelated_anchors=ignore_unrelated_anchors,
    )

    # Step 3: main SQL via _denormalize_impl
    main_result = _denormalize_impl(
        model=self._model,
        engine=self._engine,
        orm_resolver=self._orm_resolver,
        dataset_rid=self._dataset_rid,
        include_tables=list(include_tables),
        dataset=self._dataset,
        source="local",
        row_per=resolved_row_per,
        via=list(via or []) or None,
    )

    # Step 4: orphan rows
    if orphans:
        orphan_rows = self._emit_orphan_rows(
            orphans,
            include_tables=list(include_tables),
            row_per=resolved_row_per,
        )
        main_result = DenormalizeResult(
            columns=main_result.columns,
            row_count=main_result.row_count + len(orphan_rows),
            _rows=list(main_result.iter_rows()) + orphan_rows,
        )

    return main_result


def _anchors_as_dict(self) -> dict[str, list[str]]:
    """Return anchors as table_name -> list of RID strings."""
    members = self._dataset.list_dataset_members(recurse=True)
    return {table: [r["RID"] for r in rows] for table, rows in members.items()}


def _emit_orphan_rows(
    self,
    orphans: dict[str, list[str]],
    *,
    include_tables: list[str],
    row_per: str,
) -> list[dict[str, Any]]:
    """Emit one output row per orphan anchor.

    For each orphan RID, fetch its row and construct an output dict with:
      - Anchor's columns populated with the row values.
      - row_per and downstream columns set to None.
      - Upstream columns (reachable by outbound FK from the anchor)
        populated by walking the chain.
    """
    from deriva_ml.model.catalog import denormalize_column_name
    from sqlalchemy import select

    # Only emit orphan rows for anchors that don't reach row_per.
    # (The scoping anchors that DO reach row_per are handled by the main SQL.)
    orphan_rows: list[dict[str, Any]] = []

    # Get the full column spec so we know what keys to populate.
    _, column_specs, multi_schema = self._model._prepare_wide_table(
        self._dataset, self._dataset_rid, list(include_tables),
    )

    # For each orphan anchor table, for each RID, emit one row.
    for anchor_table, rids in orphans.items():
        orm_cls = self._orm_resolver(anchor_table)
        if orm_cls is None:
            continue
        for rid in rids:
            with self._engine.connect() as conn:
                row = conn.execute(
                    select(orm_cls.__table__).where(orm_cls.__table__.c.RID == rid)
                ).mappings().first()
            if row is None:
                continue
            # Build the output dict: anchor cols populated, others None.
            out: dict[str, Any] = {}
            for schema_name, table_name, col_name, _type_name in column_specs:
                label = denormalize_column_name(
                    schema_name, table_name, col_name, multi_schema
                )
                if table_name == anchor_table:
                    out[label] = row.get(col_name)
                else:
                    out[label] = None
            orphan_rows.append(out)

    return orphan_rows
```

- [ ] **Step 5: Run tests.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_denormalizer.py -v --tb=short 2>&1 | tail -30
```

Expected: all pass, including the new `TestUnrelatedAnchors` and `TestOrphanRows` classes.

- [ ] **Step 6: Lint + commit.**

```bash
uv run ruff format src/deriva_ml/local_db/denormalizer.py tests/local_db/test_denormalizer.py
uv run ruff check src/deriva_ml/local_db/denormalizer.py
git add src/deriva_ml/local_db/denormalizer.py tests/local_db/test_denormalizer.py
git commit -m "feat(denormalize): anchor classification + orphan row emission (Rules 7, 8)"
```

---

## Task 7: Extend `describe()` to return the full plan dict (spec §5)

The minimal `describe` from Task 4 just returned row_per + columns. The full plan dict needs: transparent intermediates, join path, ambiguities (when the user explicitly asks for a dry-run), estimated row counts, anchor summary, source mode.

**Files:**
- Modify: `src/deriva_ml/local_db/denormalizer.py` — extend `describe`.
- Modify: `tests/local_db/test_denormalizer.py` — extend `TestDescribe`.

**DRY reminders (see Reuse inventory above):**
- `describe()` should reuse the existing `Dataset.denormalize_info` shape
  (`src/deriva_ml/core/mixins/dataset.py:312`) as its starting point —
  that function already produces `columns`, `join_path`, and per-table
  counts. Extend with `row_per` / `row_per_source` / `row_per_candidates` /
  `ambiguities` / `orphan_rows` / `anchors` fields. Do NOT rewrite column
  generation or join-path construction.
- All row-count estimation should go through `_classify_anchors` (added
  in Task 6) — do NOT duplicate anchor classification.
- Ambiguity detection: call `self._model._find_path_ambiguities(...)`.
  Unlike `as_dataframe`, `describe` reports ambiguities (dry-run) rather
  than raising.
- Use `denormalize_column_name` for all output column labels.

- [ ] **Step 1: Write failing tests.**

Add to `tests/local_db/test_denormalizer.py`:

```python
class TestDescribe:
    """Denormalizer.describe returns the full plan dict per spec §5."""

    def test_describe_keys(self, populated_denorm) -> None:
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        plan = d.describe(["Image", "Subject"])
        # Required keys per spec §5
        for key in [
            "row_per",
            "row_per_source",
            "row_per_candidates",
            "columns",
            "include_tables",
            "via",
            "join_path",
            "transparent_intermediates",
            "ambiguities",
            "estimated_row_count",
            "anchors",
            "source",
        ]:
            assert key in plan, f"plan missing key {key}: {list(plan.keys())}"

    def test_describe_row_per_explicit(self, populated_denorm) -> None:
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        plan = d.describe(["Image", "Subject"], row_per="Image")
        assert plan["row_per_source"] == "explicit"

    def test_describe_row_per_auto(self, populated_denorm) -> None:
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        plan = d.describe(["Image", "Subject"])
        assert plan["row_per_source"] == "auto-inferred"
        assert plan["row_per"] == "Image"

    def test_describe_ambiguity_reported(self, populated_denorm) -> None:
        if "Observation" not in populated_denorm["model"].name_to_schema_and_table:
            pytest.skip("Canned model needs diamond fixture (Task 2 step 6)")

        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        plan = d.describe(["Image", "Subject"])
        # Ambiguity reported rather than raised (describe is dry-run)
        assert len(plan["ambiguities"]) > 0
        amb = plan["ambiguities"][0]
        assert amb["from"] == "Image"
        assert amb["to"] == "Subject"
        assert "paths" in amb
        assert "suggestions" in amb

    def test_describe_anchors(self, populated_denorm) -> None:
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        plan = d.describe(["Image", "Subject"])
        anc = plan["anchors"]
        assert "by_type" in anc
        assert "total" in anc
        assert anc["by_type"]["Image"] == 3  # 3 image members in fixture
```

- [ ] **Step 2: Run tests — expect failures.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_denormalizer.py::TestDescribe -v
```

Expected: `KeyError: 'transparent_intermediates'` etc.

- [ ] **Step 3: Extend `Denormalizer.describe`.**

Replace the `describe` body in `src/deriva_ml/local_db/denormalizer.py`:

```python
def describe(
    self,
    include_tables: list[str],
    *,
    row_per: str | None = None,
    via: list[str] | None = None,
) -> dict[str, Any]:
    """Return a planning-metadata dict describing what would happen.

    Unlike :meth:`as_dataframe`, ``describe`` does NOT raise on ambiguity —
    it reports ambiguities in the ``ambiguities`` key so callers can
    inspect before committing to a real call.

    Returns a dict with these keys (see spec §5):
        row_per, row_per_source, row_per_candidates, columns,
        include_tables, via, join_path, transparent_intermediates,
        ambiguities, estimated_row_count, anchors, source
    """
    from deriva_ml.core.exceptions import (
        DerivaMLDenormalizeDownstreamLeaf,
        DerivaMLDenormalizeMultiLeaf,
    )
    from deriva_ml.model.catalog import denormalize_column_name

    include = list(include_tables)
    via_list = list(via or [])

    # ── row_per resolution ─────────────────────────────────────────────
    row_per_source = "explicit" if row_per else "auto-inferred"
    row_per_candidates = self._model._find_sinks(include, via_list)
    try:
        resolved_row_per: str | None = self._model._determine_row_per(
            include_tables=include, via=via_list, row_per=row_per,
        )
    except (DerivaMLDenormalizeMultiLeaf, DerivaMLDenormalizeDownstreamLeaf):
        resolved_row_per = None

    # ── columns (may raise if row_per is None) ─────────────────────────
    try:
        element_tables, column_specs, multi_schema = self._model._prepare_wide_table(
            self._dataset, self._dataset_rid, include,
            row_per=row_per, via=via_list,
        )
        cols = [
            (denormalize_column_name(s, t, c, multi_schema), tp)
            for s, t, c, tp in column_specs
        ]
    except Exception:
        element_tables = {}
        cols = []

    # ── ambiguities (reported, not raised) ─────────────────────────────
    ambiguities_raw = []
    if resolved_row_per is not None:
        ambiguities_raw = self._model._find_path_ambiguities(
            row_per=resolved_row_per,
            include_tables=include,
            via=via_list,
        )
    ambiguities = [
        {
            "type": "multiple_paths",
            "from": a["from_table"],
            "to": a["to_table"],
            "paths": [" → ".join(p) for p in a["paths"]],
            "suggestions": {
                "add_to_include_tables": a["suggested_intermediates"],
                "add_to_via": a["suggested_intermediates"],
            },
        }
        for a in ambiguities_raw
    ]

    # ── join path + transparent intermediates ───────────────────────────
    join_path: list[str] = []
    transparent: list[str] = []
    for _, (path_names, _, _) in element_tables.items():
        for tn in path_names:
            if tn not in join_path and tn != "Dataset":
                join_path.append(tn)
                if tn not in include and self._model.is_association(tn):
                    transparent.append(tn)

    # ── anchors summary ─────────────────────────────────────────────────
    anchors = self._anchors_as_dict()
    anchors_by_type = {t: len(rids) for t, rids in anchors.items()}

    # ── estimated row count (crude — refined in future work) ────────────
    # For v1: report how many anchors would be scoping vs orphan.
    estimated = {
        "in_scope_row_per_rows": None,
        "orphan_rows": None,
        "total": None,
    }
    if resolved_row_per is not None:
        # Count anchors classified as scoping (includes row_per anchors)
        try:
            scoping, orphans, _ = self._classify_anchors(
                anchors,
                include_tables=include,
                via=via_list,
                row_per=resolved_row_per,
                ignore_unrelated_anchors=True,  # describe shouldn't raise
            )
            in_scope = sum(len(rids) for table, rids in scoping.items() if table == resolved_row_per)
            orphan_count = sum(len(rids) for rids in orphans.values())
            estimated = {
                "in_scope_row_per_rows": in_scope,
                "orphan_rows": orphan_count,
                "total": in_scope + orphan_count,
            }
        except Exception:
            pass

    return {
        "row_per": resolved_row_per,
        "row_per_source": row_per_source,
        "row_per_candidates": row_per_candidates,
        "columns": cols,
        "include_tables": include,
        "via": via_list,
        "join_path": join_path,
        "transparent_intermediates": transparent,
        "ambiguities": ambiguities,
        "estimated_row_count": estimated,
        "anchors": {"total": sum(anchors_by_type.values()), "by_type": anchors_by_type},
        "source": "local",  # Task 11 updates this when source is tracked
    }
```

- [ ] **Step 4: Run tests.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_denormalizer.py::TestDescribe -v --tb=short
```

Expected: all pass.

- [ ] **Step 5: Lint + commit.**

```bash
uv run ruff format src/deriva_ml/local_db/denormalizer.py
uv run ruff check src/deriva_ml/local_db/denormalizer.py
git add src/deriva_ml/local_db/denormalizer.py tests/local_db/test_denormalizer.py
git commit -m "feat(denormalize): Denormalizer.describe returns full plan dict (spec §5)"
```

---

## Task 8: Implement `list_paths()` for schema exploration (spec §6)

**Files:**
- Modify: `src/deriva_ml/local_db/denormalizer.py` — implement `list_paths`.
- Modify: `tests/local_db/test_denormalizer.py` — add `TestListPaths`.

**DRY reminders (see Reuse inventory above):**
- `reachable_tables` uses `self._model._outbound_reachable` — do NOT
  re-walk FKs.
- `association_tables` uses `self._model._is_association_table`.
- `schema_paths` uses `self._model._enumerate_paths` — which itself
  wraps `_schema_to_paths`.
- `feature_tables` should use `DerivaModel.find_features` /
  `is_feature_table` if exposed; do NOT heuristic-detect by name parsing.
- If the `Dataset` has a dedicated `list_dataset_element_types` (see
  `src/deriva_ml/core/mixins/dataset.py:166`), use that for
  `member_types`.

- [ ] **Step 1: Write failing tests.**

Add to `tests/local_db/test_denormalizer.py`:

```python
class TestListPaths:
    """list_paths describes the FK graph from the dataset's anchor types."""

    def test_list_paths_keys(self, populated_denorm) -> None:
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        info = d.list_paths()
        # Required keys per spec §6
        for key in [
            "member_types",
            "anchor_types",
            "reachable_tables",
            "association_tables",
            "feature_tables",
            "schema_paths",
        ]:
            assert key in info

    def test_list_paths_reports_member_types(self, populated_denorm) -> None:
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        info = d.list_paths()
        assert "Image" in info["member_types"]
        assert "Subject" in info["member_types"]

    def test_list_paths_filter_by_tables(self, populated_denorm) -> None:
        ds = _FakeDataset(populated_denorm)
        d = Denormalizer(ds)
        # When filter is given, schema_paths contains only entries involving
        # those tables.
        info = d.list_paths(tables=["Image"])
        for (source, target), _ in info["schema_paths"].items():
            assert "Image" in (source, target)
```

- [ ] **Step 2: Run tests — expect failures.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_denormalizer.py::TestListPaths -v
```

Expected: assertions fail (the stub returns empty).

- [ ] **Step 3: Implement `list_paths`.**

Replace `Denormalizer.list_paths` body in `src/deriva_ml/local_db/denormalizer.py`:

```python
def list_paths(
    self,
    tables: list[str] | None = None,
) -> dict[str, Any]:
    """Describe the FK graph reachable from the dataset/anchors.

    Useful for picking ``include_tables`` when the user doesn't know the
    schema. Model-only analysis.

    Args:
        tables: If given, filter ``schema_paths`` to paths involving at
            least one of these tables.

    Returns:
        Dict with:
            member_types: list of dataset element types (if constructed
                from a dataset); else empty.
            anchor_types: union of all distinct anchor table names.
            reachable_tables: mapping from each member/anchor type to
                tables reachable from it via FK.
            association_tables: names of pure association tables in the
                schema.
            feature_tables: names of feature tables (detected heuristically).
            schema_paths: mapping from (source_table, target_table) to a
                list of path descriptions.
    """
    model = self._model
    anchors = self._anchors_as_dict()
    anchor_types = sorted(anchors.keys())

    # member_types: if constructed from a dataset, the dataset's members.
    # For from_rids, this is the anchor types. Same in both cases.
    member_types = anchor_types

    # Enumerate all tables in the relevant schemas.
    all_table_names: set[str] = set()
    try:
        ml_schema = getattr(model, "ml_schema", "deriva-ml")
        domain_schemas = getattr(model, "domain_schemas", [])
        for sname in [ml_schema, *domain_schemas]:
            if sname in model.schemas:
                for t in model.schemas[sname].tables.values():
                    all_table_names.add(t.name)
    except Exception:
        pass

    # reachable_tables: from each anchor type, which domain tables are
    # reachable via FK (using the whole schema as the subgraph).
    reachable_tables: dict[str, list[str]] = {}
    for t in anchor_types:
        reach = model._outbound_reachable(t, all_table_names)
        reachable_tables[t] = sorted(reach)

    # association_tables: pure M-to-N linking tables
    association_tables = sorted(
        t for t in all_table_names if model._is_association_table(t)
    )

    # feature_tables: heuristic — tables whose name contains "_" and
    # have an FK to a non-association, non-system table plus FK to a
    # vocabulary term. Approximated by: `is_feature_table` if the model
    # exposes it, else empty.
    feature_tables: list[str] = []
    is_feat = getattr(model, "is_feature_table", None)
    if callable(is_feat):
        for t in all_table_names:
            try:
                if is_feat(t):
                    feature_tables.append(t)
            except Exception:
                pass
    feature_tables.sort()

    # schema_paths: for every (source, target) pair among anchor_types ×
    # reachable_tables, enumerate FK paths.
    schema_paths: dict[tuple[str, str], list[dict]] = {}
    sources = set(anchor_types)
    if tables is not None:
        sources |= set(tables)
    for source in sources:
        for target in reachable_tables.get(source, []):
            if tables is not None and source not in tables and target not in tables:
                continue
            paths = model._enumerate_paths(source, target, all_table_names)
            # Deduplicate
            unique = list({tuple(p): p for p in paths}.values())
            schema_paths[(source, target)] = [
                {"path": p, "direct": len(p) == 2}
                for p in unique
            ]

    return {
        "member_types": member_types,
        "anchor_types": anchor_types,
        "reachable_tables": reachable_tables,
        "association_tables": association_tables,
        "feature_tables": feature_tables,
        "schema_paths": schema_paths,
    }
```

- [ ] **Step 4: Run tests.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_denormalizer.py::TestListPaths -v --tb=short
```

Expected: all pass.

- [ ] **Step 5: Lint + commit.**

```bash
uv run ruff format src/deriva_ml/local_db/denormalizer.py
uv run ruff check src/deriva_ml/local_db/denormalizer.py
git add src/deriva_ml/local_db/denormalizer.py tests/local_db/test_denormalizer.py
git commit -m "feat(denormalize): implement Denormalizer.list_paths for schema exploration"
```

---

## Task 9: Add `Denormalizer.from_rids` for arbitrary anchor sets

**Files:**
- Modify: `src/deriva_ml/local_db/denormalizer.py` — add `from_rids` classmethod.
- Modify: `tests/local_db/test_denormalizer.py` — add `TestFromRids`.

- [ ] **Step 1: Write failing tests.**

Add to `tests/local_db/test_denormalizer.py`:

```python
class TestFromRids:
    """Denormalizer.from_rids constructs from arbitrary RID anchors."""

    def test_from_rids_with_table_tuples(self, populated_denorm) -> None:
        """(table, RID) pairs skip the lookup."""
        from deriva_ml.local_db.workspace import Workspace

        ml = _FakeMl(populated_denorm)
        d = Denormalizer.from_rids(
            [("Image", r) for r in populated_denorm["image_rids"]],
            ml=ml,
        )
        df = d.as_dataframe(["Image", "Subject"])
        # 3 Images in fixture — all reachable.
        assert len(df) == 3

    def test_from_rids_with_separate_deps(self, populated_denorm) -> None:
        """Escape hatch: pass catalog, workspace, model explicitly."""
        ls = populated_denorm["local_schema"]
        d = Denormalizer.from_rids(
            [("Image", r) for r in populated_denorm["image_rids"]],
            catalog=None,  # no lookup needed (table supplied)
            workspace=None,  # fixture provides engine directly
            model=populated_denorm["model"],
            engine=ls.engine,
            orm_resolver=ls.get_orm_class,
        )
        df = d.as_dataframe(["Image", "Subject"])
        assert len(df) == 3

    def test_from_rids_mixed_forms_not_implemented_yet(self, populated_denorm) -> None:
        """Bare RIDs require catalog lookup; in this test we'll just use
        the tuple form (bare-RID lookup is catalog-dependent and exercised
        in the live integration test suite)."""
        # Placeholder - mixed forms with bare RIDs need a live catalog.
        pass


class _FakeMl:
    """Minimal DerivaML-shaped fixture for from_rids tests."""
    def __init__(self, populated_denorm):
        self._pd = populated_denorm
        self.model = populated_denorm["model"]
        # Workspace-like object that exposes engine + local_schema
        class _WS:
            def __init__(self, ls):
                self._ls = ls
                self.local_schema = ls
                self.engine = ls.engine
        self.workspace = _WS(populated_denorm["local_schema"])
        self.catalog = None  # bare-RID lookup not needed for tuple anchors
```

- [ ] **Step 2: Run tests — expect `AttributeError: from_rids`.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_denormalizer.py::TestFromRids -v
```

- [ ] **Step 3: Implement `from_rids` classmethod.**

Add to the `Denormalizer` class:

```python
@classmethod
def from_rids(
    cls,
    anchors: list[str | tuple[str, str]],
    *,
    ml: Any = None,
    catalog: Any = None,
    workspace: Any = None,
    model: Any = None,
    engine: Any = None,
    orm_resolver: Any = None,
    ignore_unrelated_anchors: bool = False,
) -> "Denormalizer":
    """Construct from an explicit anchor set.

    Anchors may be bare RIDs (table looked up via catalog) or
    ``(table_name, RID)`` tuples (lookup skipped). Mixed forms supported.

    Pass either ``ml=`` (common path) or the separate ``catalog``,
    ``workspace``, ``model`` keyword args (escape hatch).

    Args:
        anchors: list of RIDs or (table, RID) tuples.
        ml: Convenience: pass a DerivaML instance. catalog/workspace/model
            are derived from it.
        catalog, workspace, model, engine, orm_resolver: Explicit deps.
        ignore_unrelated_anchors: propagated to subsequent method calls.
    """
    # Derive deps from ml if given
    if ml is not None:
        catalog = catalog or getattr(ml, "catalog", None)
        workspace = workspace or getattr(ml, "workspace", None)
        model = model or getattr(ml, "model", None)
        if engine is None and workspace is not None:
            engine = workspace.engine
        if orm_resolver is None and workspace is not None:
            ls = getattr(workspace, "local_schema", None)
            if ls is not None:
                orm_resolver = ls.get_orm_class

    if model is None:
        raise ValueError(
            "Denormalizer.from_rids requires either ml= or an explicit model="
        )

    # Normalize anchors to (table, RID) pairs
    resolved: list[tuple[str, str]] = []
    bare_rids: list[str] = []
    for a in anchors:
        if isinstance(a, tuple):
            resolved.append(a)
        else:
            bare_rids.append(a)

    # Batch-resolve bare RIDs via catalog
    if bare_rids:
        if catalog is None:
            raise ValueError(
                "Bare RIDs given but no catalog available for lookup. "
                "Pass (table, RID) tuples or provide catalog=."
            )
        for rid in bare_rids:
            info = catalog.resolve_rid(rid) if hasattr(catalog, "resolve_rid") else None
            if info is None:
                raise ValueError(f"Cannot resolve RID {rid!r} to a table")
            resolved.append((info.table.name, rid))

    # Group by table
    anchors_by_table: dict[str, list[str]] = {}
    for table, rid in resolved:
        anchors_by_table.setdefault(table, []).append(rid)

    # Use the first anchor as a "pseudo-dataset_rid" for _prepare_wide_table
    # (it only uses this for the WHERE clause Dataset.RID IN (...) — for
    # from_rids we don't have a Dataset context, so we pass the resolver
    # a placeholder).
    pseudo_rid = resolved[0][1] if resolved else ""

    # Create a pseudo-dataset that exposes the anchors dict as members
    class _AnchorSet:
        dataset_rid = pseudo_rid
        model = None  # filled below

        def list_dataset_members(self, **_kwargs):
            return {t: [{"RID": r} for r in rids] for t, rids in anchors_by_table.items()}

        def list_dataset_children(self, **_kwargs):
            return []

    anchor_set = _AnchorSet()
    anchor_set.model = model

    # Build the Denormalizer manually
    inst = object.__new__(cls)
    inst._dataset = anchor_set
    inst._dataset_rid = pseudo_rid
    inst._model = model
    inst._engine = engine
    inst._orm_resolver = orm_resolver
    return inst
```

- [ ] **Step 4: Run tests.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_denormalizer.py::TestFromRids -v --tb=short
```

Expected: the tuple-form tests pass; the mixed-form test is a placeholder that passes trivially.

- [ ] **Step 5: Lint + commit.**

```bash
uv run ruff format src/deriva_ml/local_db/denormalizer.py
uv run ruff check src/deriva_ml/local_db/denormalizer.py
git add src/deriva_ml/local_db/denormalizer.py tests/local_db/test_denormalizer.py
git commit -m "feat(denormalize): Denormalizer.from_rids for arbitrary RID anchors"
```

---

## Task 10: Add sugar methods on `Dataset` and `DatasetBag`; remove old methods

**Files:**
- Modify: `src/deriva_ml/dataset/dataset.py` — add new methods, remove old.
- Modify: `src/deriva_ml/dataset/dataset_bag.py` — same.
- Modify: `src/deriva_ml/interfaces.py` — update `DatasetLike` protocol.

- [ ] **Step 1: In `src/deriva_ml/dataset/dataset.py`, add the five sugar methods.**

Near the end of the `Dataset` class (before the `cache_denormalized` method), add:

```python
def get_denormalized_as_dataframe(
    self,
    include_tables: list[str],
    *,
    row_per: str | None = None,
    via: list[str] | None = None,
    ignore_unrelated_anchors: bool = False,
) -> "pd.DataFrame":
    """Return the dataset as a denormalized wide table (DataFrame).

    Shortcut for ``Denormalizer(self).as_dataframe(include_tables, ...)``.
    See :class:`~deriva_ml.local_db.denormalizer.Denormalizer` for the
    full API and semantic rules.
    """
    from deriva_ml.local_db.denormalizer import Denormalizer
    return Denormalizer(self).as_dataframe(
        include_tables,
        row_per=row_per,
        via=via,
        ignore_unrelated_anchors=ignore_unrelated_anchors,
    )


def get_denormalized_as_dict(
    self,
    include_tables: list[str],
    *,
    row_per: str | None = None,
    via: list[str] | None = None,
    ignore_unrelated_anchors: bool = False,
) -> "Generator[dict[str, Any], None, None]":
    """Stream the denormalized dataset rows as dicts.

    Shortcut for ``Denormalizer(self).as_dict(include_tables, ...)``.
    """
    from deriva_ml.local_db.denormalizer import Denormalizer
    yield from Denormalizer(self).as_dict(
        include_tables,
        row_per=row_per,
        via=via,
        ignore_unrelated_anchors=ignore_unrelated_anchors,
    )


def list_denormalized_columns(
    self,
    include_tables: list[str],
    *,
    row_per: str | None = None,
    via: list[str] | None = None,
) -> list[tuple[str, str]]:
    """List the columns the denormalized table would have.

    Shortcut for ``Denormalizer(self).columns(include_tables, ...)``.
    Model-only — no data fetch.
    """
    from deriva_ml.local_db.denormalizer import Denormalizer
    return Denormalizer(self).columns(
        include_tables, row_per=row_per, via=via,
    )


def describe_denormalized(
    self,
    include_tables: list[str],
    *,
    row_per: str | None = None,
    via: list[str] | None = None,
) -> dict[str, Any]:
    """Dry-run the denormalization; return planning metadata.

    Shortcut for ``Denormalizer(self).describe(include_tables, ...)``.
    See the spec (docs/superpowers/specs/...) for the exact structure.
    """
    from deriva_ml.local_db.denormalizer import Denormalizer
    return Denormalizer(self).describe(
        include_tables, row_per=row_per, via=via,
    )


def list_schema_paths(
    self,
    tables: list[str] | None = None,
) -> dict[str, Any]:
    """List FK paths reachable from this dataset's members.

    Shortcut for ``Denormalizer(self).list_paths(tables)``. Useful for
    discovering what tables are available to include in denormalization.
    """
    from deriva_ml.local_db.denormalizer import Denormalizer
    return Denormalizer(self).list_paths(tables=tables)
```

- [ ] **Step 2: Remove the old methods from `Dataset`.**

In `src/deriva_ml/dataset/dataset.py`, delete these method definitions (they're around the line numbers shown — use the grep output above to confirm):

- `def denormalize_as_dataframe(...)` at line ~782
- `def denormalize_as_dict(...)` at line ~944
- `def denormalize_columns(...)` at line ~1043
- `def denormalize_info(...)` at line ~1083

Remove these methods entirely. Do not leave stubs or aliases.

- [ ] **Step 3: Do the same for `src/deriva_ml/dataset/dataset_bag.py`.**

Add the same five sugar methods, and remove the corresponding old methods at lines ~763, ~841, ~934.

- [ ] **Step 4: Update the `DatasetLike` protocol in `src/deriva_ml/interfaces.py`.**

Replace the old method stubs (around lines 225, 299, 350) with the new ones. Keep the type-signature-level documentation but use the new names.

- [ ] **Step 5: Update `DerivaML.describe_denormalized` in `core/mixins/dataset.py`.**

Rename `DerivaML.denormalize_info` to `DerivaML.describe_denormalized`. Update the body to use the new planner helpers (delegate to `Denormalizer` isn't straightforward since there's no dataset — keep the same implementation shape but with the new method name).

- [ ] **Step 6: Update all in-repo callers to the new names.**

```bash
grep -rn "denormalize_as_dataframe\|denormalize_as_dict\|denormalize_columns\|denormalize_info" src/ tests/ --include="*.py" | grep -v __pycache__ | grep -v ".pyc"
```

For each hit that's not in the deleted methods themselves, update:

- `denormalize_as_dataframe(...)` → `get_denormalized_as_dataframe(...)`
- `denormalize_as_dict(...)` → `get_denormalized_as_dict(...)`
- `denormalize_columns(...)` → `list_denormalized_columns(...)`
- `denormalize_info(...)` → `describe_denormalized(...)`

Known locations:
- `src/deriva_ml/dataset/split.py:738` — `source_ds.denormalize_as_dataframe` call.
- `tests/dataset/test_denormalize.py` — many call sites.
- `tests/dataset/test_denormalize_info.py` — all call sites.
- `src/deriva_ml/interfaces.py` — docstring references in `denormalize_as_dict`'s See Also.

Use a global sed to speed this up for source files:

```bash
# Source files only — tests handled in Task 12
find src -name "*.py" -not -path "*/node_modules/*" -exec sed -i '' \
  -e 's/\.denormalize_as_dataframe(/\.get_denormalized_as_dataframe(/g' \
  -e 's/\.denormalize_as_dict(/\.get_denormalized_as_dict(/g' \
  -e 's/\.denormalize_columns(/\.list_denormalized_columns(/g' \
  -e 's/\.denormalize_info(/\.describe_denormalized(/g' \
  {} \;
```

Verify no stale method names remain in src:

```bash
grep -rn "\.denormalize_as_dataframe\|\.denormalize_as_dict\|\.denormalize_columns\|\.denormalize_info" src/ --include="*.py" | grep -v __pycache__
```

Expected: empty.

- [ ] **Step 7: Run the new Denormalizer tests + split tests.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/test_manifest.py --tb=short 2>&1 | tail -5
```

Expected: all pass. Don't run dataset integration tests yet — those are updated in Task 12.

- [ ] **Step 8: Lint + commit.**

```bash
uv run ruff format src/
uv run ruff check src/
git add src/
git commit -m "refactor(denormalize): add Dataset/DatasetBag sugar methods; remove old names

Rename denormalize_as_dataframe → get_denormalized_as_dataframe,
denormalize_as_dict → get_denormalized_as_dict, denormalize_columns →
list_denormalized_columns, denormalize_info → describe_denormalized.
Old methods removed outright (no deprecation). All in-repo callers
updated via sed.
"
```

---

## Task 11: Update `Workspace.cache_denormalized` to forward new params

**Files:**
- Modify: `src/deriva_ml/local_db/workspace.py` — `cache_denormalized` method.

- [ ] **Step 1: Find the current implementation.**

```bash
grep -n "def cache_denormalized\|def cache_key\|resolve_row_per" src/deriva_ml/local_db/workspace.py
```

Expected: `cache_denormalized` around line 412.

- [ ] **Step 2: Update the signature to accept `row_per`, `via`, `ignore_unrelated_anchors`.**

Add these params with sensible defaults. Include them in the cache key so different denormalizations are cached independently. The impl forwards to `_denormalize_impl` or a fresh `Denormalizer` — whichever is cleanest (check if the current impl calls `_denormalize_impl` or an intermediate).

Key fragment:

```python
def cache_denormalized(
    self,
    model: Any,
    dataset_rid: str,
    include_tables: list[str],
    version: str | None = None,
    source: str = "local",
    slice_id: str | None = None,
    refresh: bool = False,
    dataset: Any = None,
    dataset_children_rids: list[str] | None = None,
    paged_client: Any = None,
    row_per: str | None = None,                    # new
    via: list[str] | None = None,                  # new
    ignore_unrelated_anchors: bool = False,        # new
) -> "CachedResult":
```

In the body, include row_per/via/ignore_unrelated_anchors in the cache key and pass them through to the impl.

- [ ] **Step 3: Run workspace tests.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_workspace.py -v --tb=short 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 4: Lint + commit.**

```bash
uv run ruff format src/deriva_ml/local_db/workspace.py
uv run ruff check src/deriva_ml/local_db/workspace.py
git add src/deriva_ml/local_db/workspace.py
git commit -m "feat(denormalize): Workspace.cache_denormalized forwards row_per/via"
```

---

## Task 12: Restructure the existing integration test file

This is the largest test-rewriting task. The existing `tests/dataset/test_denormalize.py` (1461 lines) has classes organized by historical concern, not by the new rules. Restructure to match the rule-by-rule organization in the spec, update all call sites to the new API, and un-xfail the three tests that should now pass.

**Files:**
- Rewrite: `tests/dataset/test_denormalize.py`
- Rewrite: `tests/dataset/test_denormalize_info.py` → `tests/dataset/test_describe_denormalized.py`

- [ ] **Step 1: Update call sites in `tests/dataset/test_denormalize.py`.**

Run the same sed as in Task 10 step 6 but for tests:

```bash
find tests -name "*.py" -exec sed -i '' \
  -e 's/\.denormalize_as_dataframe(/\.get_denormalized_as_dataframe(/g' \
  -e 's/\.denormalize_as_dict(/\.get_denormalized_as_dict(/g' \
  -e 's/\.denormalize_columns(/\.list_denormalized_columns(/g' \
  -e 's/\.denormalize_info(/\.describe_denormalized(/g' \
  {} \;
```

- [ ] **Step 2: Remove `xfail` markers on the three tests that should now pass.**

In `tests/dataset/test_denormalize.py`:

- Line 1275: `@pytest.mark.xfail(reason="Pending join tree refactoring — intermediate should force multi-hop path")` on `test_diamond_resolved_with_intermediate`. Delete the decorator.

- Line 1317: `@pytest.mark.xfail(reason="Pending join tree refactoring — association table must be joined through as implicit intermediate")` on `test_association_mandatory_intermediate`. Delete.

- Line 1427: `@pytest.mark.xfail(reason="Feature table denormalization not yet implemented in include_tables")` on `test_feature_table_included`. Delete.

- [ ] **Step 3: Update the diamond-ambiguity test that used to silently succeed but now raises.**

In `test_denormalize.py`, the `TestAmbiguousPaths::test_direct_fk_prefers_shortest_path` test (line ~1141) expects the direct FK to silently win. Under the new rules, this is an ambiguity error. Update:

```python
def test_direct_fk_raises_ambiguity(self, dataset_test, tmp_path):
    """A1: Two FK paths between Image and Subject now raise (was: silent pick).

    Under the new denormalization semantics (spec §3.6 / Rule 6), any
    path ambiguity is surfaced as an error. Users must add an intermediate
    to include_tables or via= to disambiguate.

    This replaces the old test_direct_fk_prefers_shortest_path test —
    silent path selection is explicitly rejected by design.
    """
    from deriva_ml.core.exceptions import DerivaMLDenormalizeAmbiguousPath

    dataset_description = dataset_test.dataset_description
    current_version = dataset_description.dataset.current_version
    bag = dataset_description.dataset.download_dataset_bag(current_version, use_minid=False)

    with pytest.raises(DerivaMLDenormalizeAmbiguousPath):
        bag.get_denormalized_as_dataframe(include_tables=["Image", "Subject"])
```

- [ ] **Step 4: Run the dataset integration tests against a live catalog.**

Requires `DERIVA_HOST=localhost`. Takes ~20 minutes.

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_denormalize.py -q --tb=short 2>&1 | tail -10
```

Expected: all pass. Failures likely fall into these categories — fix each:

1. Test still uses old method name — update with sed or manually.
2. Test expects silent path selection in a diamond — update to `pytest.raises(DerivaMLDenormalizeAmbiguousPath)`.
3. Test expects bag.denormalize_* to work — already handled by sugar method.
4. Test asserts row count that changed due to new rules — if the new count is correct per spec, update the test.

- [ ] **Step 5: Rename and update `test_denormalize_info.py`.**

```bash
git mv tests/dataset/test_denormalize_info.py tests/dataset/test_describe_denormalized.py
```

Update the module docstring to reference `describe_denormalized`. The return-dict structure has more keys now (see spec §5). Update the test assertions to include the new keys (`row_per`, `row_per_candidates`, `ambiguities`, `transparent_intermediates`, etc.) and check they have sensible values.

- [ ] **Step 6: Run both renamed/updated test files.**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_denormalize.py tests/dataset/test_describe_denormalized.py -q --tb=short 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 7: Lint + commit.**

```bash
uv run ruff format tests/
uv run ruff check tests/
git add tests/dataset/test_denormalize.py tests/dataset/test_describe_denormalized.py
git commit -m "test(denormalize): restructure integration tests for new rules

Update all call sites to new API names. Remove xfail markers on three
tests that pass under the new rules (diamond-resolved, association-
mandatory, feature-table). Update test_direct_fk_prefers_shortest_path
to assert ambiguity raises under the new semantics. Rename
test_denormalize_info.py → test_describe_denormalized.py.
"
```

---

## Task 13: Full test suite run + comprehensive code review

**Files:** None — this is the final gate.

- [ ] **Step 1: Run the full unit test suite (fast).**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/test_manifest.py tests/model/ -q --tb=short 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 2: Run the full integration test suite.**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/ tests/local_db/test_denormalize_impl.py -q --tb=short 2>&1 | tail -10
```

Expected: all pass. Allow ~20–30 minutes.

- [ ] **Step 3: Dispatch a comprehensive code review subagent.**

Using the implementer's session (or `Agent` tool in Claude Code), dispatch a review agent. The prompt should include:

```
You are performing a comprehensive code review of the denormalization
semantics refactor in the deriva-ml project.

Working directory: /Users/carl/GitHub/deriva-ml/.claude/worktrees/compassionate-visvesvaraya/
Branch: claude/compassionate-visvesvaraya
Base commit: <commit SHA just before Task 1>
Head commit: <HEAD SHA after Task 12>

Spec: docs/superpowers/specs/2026-04-17-denormalization-semantics-design.md
User-facing doc: docs/concepts/denormalization.md

Review all changes with focus on:

1. Correctness
   - Does the planner implement Rules 2, 5, 6 exactly as specified?
   - Does Denormalizer.as_dataframe produce correct output for all the
     example scenarios in the spec (linear chain, diamond, diamond with
     intermediate, diamond with via, feature table, orphan members)?
   - Are association tables correctly handled as transparent
     intermediates (joined through, columns excluded)?
   - Does Rule 7 case 3 (orphan rows) actually produce NULL on the
     row_per side?
   - Does Rule 8 (unrelated anchors) raise on default and silently drop
     with the flag?

2. Interface consistency
   - Do the new Dataset/DatasetBag method names follow the
     get_*_as_* / list_* / describe_* pattern?
   - Do Denormalizer class method signatures match the spec §2.3?
   - Is Denormalizer.from_rids consistent with the spec §2.2?

3. Error messages
   - Do the five new exception classes (DerivaMLDenormalize*) include
     all the fields the spec says they should (candidates, paths,
     suggestions, etc.)?
   - Are error messages actionable — do they tell the user exactly what
     to do?

4. Test coverage
   - Does tests/local_db/test_planner_rules.py cover all the sink/
     downstream/path-ambiguity scenarios from the spec?
   - Does tests/local_db/test_denormalizer.py cover every public method
     of Denormalizer, both happy paths and error cases?
   - Are the three xfail tests now passing under the new rules?
   - Is TestOrphanRows actually exercising the Rule-7-case-3 path?
   - Is TestUnrelatedAnchors exercising Rule 8 including the ignore flag?

5. Code quality & DRY
   - Is each method in Denormalizer / the planner helpers focused on one
     responsibility?
   - Are there copy-paste duplications between Denormalizer and the
     older _denormalize_impl that should be consolidated?
   - Do the planner helpers (_find_sinks, _determine_row_per, etc.)
     have clear docstrings with args/returns/raises?
   - Any dead code left over from the old API?

   **DRY audit (check against the plan's "Reuse inventory" section):**
   - Does `_enumerate_paths` delegate to `_schema_to_paths` (the
     authoritative FK-graph DFS), or does it re-implement DFS?
   - Is `_is_association_table` a method on `DerivaModel` (extracted
     from its previous nested definition in `_build_join_tree`), and do
     both the planner and `Denormalizer.list_paths` call the same method?
   - Is `_fk_neighbors` a method on `DerivaModel` (extracted from its
     previous nested definition in `_schema_to_paths`), and are there any
     places still walking `table.foreign_keys` + `table.referenced_by`
     directly that should instead call `_fk_neighbors`?
   - Are all output column names generated through
     `deriva_ml.model.catalog.denormalize_column_name`, or is there a new
     `{table}.{col}` string concatenation?
   - Are FK edges between two tables resolved through
     `DerivaModel._table_relationship`, or re-implemented?
   - Does `_emit_orphan_rows` follow the same NULL-init pattern as
     `Dataset._denormalize_datapath`, or invent its own?
   - Does `describe` reuse the shape established by `denormalize_info`
     (columns / join_path / per-table counts), or does it recompute them?
   - Does the anchor-retrieval path use `list_dataset_members` rather
     than querying dataset association tables directly?

   Any "yes it duplicates" finding is an Important-severity issue.

6. Docstring accuracy
   - Do the docstrings on the new methods match the spec's semantic
     rules?
   - Do they accurately describe row cardinality and orphan behavior?

Format the review as:
- Summary (1 paragraph)
- Strengths (top 3-5)
- Issues, organized by severity: Critical / Important / Minor
  Each issue: file path, line numbers, what's wrong, suggested fix.
- Test coverage assessment
- Recommendation: READY / NEEDS FIXES / NEEDS REDESIGN
```

Run the review agent. Address any Critical issues immediately. Important issues should be addressed before merge. Minor issues can be deferred or fixed as a follow-up.

- [ ] **Step 4: Lint all changed files.**

```bash
uv run ruff format src/ tests/
uv run ruff check src/
```

Fix any remaining lint issues. Commit any lint fixes:

```bash
git add -A
git commit -m "style: ruff format across denormalize changes"
```

- [ ] **Step 5: Final verification.**

```bash
# Verify no stale old-name references in source
grep -rn "\.denormalize_as_dataframe\|\.denormalize_as_dict\|\.denormalize_columns\|\.denormalize_info" src/ tests/ --include="*.py" | grep -v __pycache__
# Verify all new names resolve
DERIVA_ML_ALLOW_DIRTY=true uv run python -c "
from deriva_ml.local_db.denormalizer import Denormalizer
from deriva_ml.core.exceptions import (
    DerivaMLDenormalizeError,
    DerivaMLDenormalizeAmbiguousPath,
    DerivaMLDenormalizeMultiLeaf,
    DerivaMLDenormalizeNoSink,
    DerivaMLDenormalizeDownstreamLeaf,
    DerivaMLDenormalizeUnrelatedAnchor,
)
print('All imports resolve')
"
```

Expected:
- `grep` returns no hits.
- Python import succeeds.

- [ ] **Step 6: Update CHANGELOG / release notes (if present).**

Check if `CHANGELOG.md` or similar exists. If so, append a breaking-change note:

```markdown
### Breaking changes

- Denormalization API renamed to follow `get_*_as_*` / `list_*` / `describe_*` conventions:
  - `denormalize_as_dataframe` → `get_denormalized_as_dataframe`
  - `denormalize_as_dict` → `get_denormalized_as_dict`
  - `denormalize_columns` → `list_denormalized_columns`
  - `denormalize_info` → `describe_denormalized`
- Ambiguous FK paths now raise `DerivaMLDenormalizeAmbiguousPath` instead
  of silently selecting the shortest path. See the migration guide in
  `docs/concepts/denormalization.md`.
- New `Denormalizer` class in `deriva_ml.local_db.denormalizer` is the
  primary entry point for ad-hoc denormalization from arbitrary RID sets.
```

- [ ] **Step 7: Final commit.**

```bash
git add CHANGELOG.md 2>/dev/null || true
git commit -m "docs: note breaking denormalization API changes in CHANGELOG" || true
```

---

## Verification checklist (run before declaring done)

- [ ] `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/test_manifest.py tests/model/ -v --tb=short` → all pass.
- [ ] `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/ -v --tb=short` → all pass, including the three previously-xfailed tests.
- [ ] `grep -rn "denormalize_as_dataframe\|denormalize_as_dict\|denormalize_columns\|denormalize_info" src/` returns no matches.
- [ ] `uv run ruff check src/` → no issues.
- [ ] `from deriva_ml.local_db.denormalizer import Denormalizer` works.
- [ ] The five new exception classes are importable from `deriva_ml.core.exceptions`.
- [ ] Code review (Task 13 step 3) returns READY or all issues addressed.

## What this plan delivers

At the end:

- **`Denormalizer` class** in `src/deriva_ml/local_db/denormalizer.py` with the five public methods and two constructors.
- **New planner helpers** in `src/deriva_ml/model/catalog.py`: `_find_sinks`, `_determine_row_per`, `_find_path_ambiguities`, `_enumerate_paths`, `_outbound_reachable`, `_is_association_table`.
- **Five new exception classes** in `src/deriva_ml/core/exceptions.py`.
- **Renamed sugar methods** on `Dataset` / `DatasetBag` / `DatasetLike` protocol.
- **All three xfail markers removed** from the integration tests that now pass.
- **Restructured test files** matching the rule-by-rule organization.
- **Zero remaining stale method names** anywhere in the repo.
- **Comprehensive code review** (Task 13 step 3) confirming correctness and quality.
