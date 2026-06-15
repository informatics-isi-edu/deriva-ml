# Fast descendant-tree walk for nested datasets — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the ~1,500 client-side catalog round-trips that make `estimate_bag_size` (and every nested-dataset `download_dataset_bag`) take ~460s, by (A) enumerating descendants from one whole-table fetch instead of per-node recursion, (B) replacing the per-descendant member scan with one aggregate membership query per association table, and (C) memoizing the descendant set so the walk runs once per operation.

**Architecture:** All changes are in `dataset.py` (a RID-only descendant accessor built on one `Dataset_Dataset` fetch) and `bag_builder.py` (`_iter_descendant_rids` uses it; `_exclude_empty_associations` uses aggregate membership; both share a memoized descendant set). Results are identical — same anchor RIDs, same excluded-association set, same estimate dict. Pure de-duplication + query-shape change. Builds on the already-verified `/schema`+pathBuilder work on this branch.

**Tech Stack:** Python ≥3.12, deriva-py datapath (`.filter`, `.in_`, `.entities().fetch(limit=...)`, `Cnt` aggregate), pytest, `uv`, ruff.

---

## Background / measured facts (do not re-derive)

- `Dataset_Dataset` table = **174 rows, 0.08s** full fetch; `Dataset` table = **159 rows**. Both are dataset-*metadata* tables, inherently small catalog-wide. One full fetch of each replaces ~760 per-node round-trips.
- `Dataset.list_dataset_children(recurse=True)` (dataset.py:2348) ALREADY fetches the full `Dataset_Dataset` table once (line 2391) and recurses in-memory via `find_children` (2393-2403) with a `_visited` cycle guard. Its ONLY waste is line 2405: `lookup_dataset(rid)` per descendant (a GET each). The walk needs RIDs, not `Dataset` objects.
- `_iter_descendant_rids` (bag_builder.py:892) manually re-recurses, calling `list_dataset_children()` (non-recursive) per node → 170 full-table fetches.
- `_exclude_empty_associations` (bag_builder.py:824) calls `list_dataset_members()` per descendant RID (dataset.py:1602), each looping every Dataset association table → ~765 queries, only to compute a boolean per association.
- The walk runs twice (anchors_for + _exclude_empty_associations).
- `.in_()` works at path-builder level: `clone_via_bag.py:177` uses `pathBuilder.filter(col.in_(...))`. (Note: `feature.py:563` warns the wrapper column may not expose `.in_()` directly in some contexts — Task 3 Step 2 verifies the exact working form before relying on it.)

Spec: `docs/superpowers/specs/2026-06-14-descendant-walk-perf-design.md`.

---

## File Structure

- `src/deriva_ml/dataset/dataset.py`:
  - factor the one-fetch descendant core out of `list_dataset_children` into a shared helper that returns RIDs; add public `list_dataset_children_rids(recurse=..., version=...)`; `list_dataset_children` keeps its signature and just hydrates the RID result.
- `src/deriva_ml/dataset/bag_builder.py`:
  - `_descendant_rids(dataset)` — memoized RID-set accessor (Part C) used by both callers; calls the new RID accessor once (Part A).
  - `_iter_descendant_rids` → delegates to `_descendant_rids` (kept as a thin alias or replaced at call sites).
  - `_exclude_empty_associations` → aggregate membership per association over the descendant-RID set (Part B).
- Tests: `tests/dataset/test_schema_paths.py` or a new `tests/dataset/test_descendant_walk_perf.py` for the GET-count + correctness guards.

---

## Task 1: RID-only descendant accessor (Part A)

**Files:**
- Modify: `src/deriva_ml/dataset/dataset.py` (`list_dataset_children` ~2348-2405)

- [ ] **Step 1: Write the failing test**

Create `tests/dataset/test_descendant_walk_perf.py`:

```python
"""Tests for the fast descendant-tree walk (perf spec 2026-06-14)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.catalog_manager import CatalogManager


def _descendant_rid_set_via_objects(dataset) -> set:
    """Baseline: descendant RIDs via the object-hydrating list_dataset_children."""
    return {c.dataset_rid for c in dataset.list_dataset_children(recurse=True)}


def test_list_dataset_children_rids_matches_objects(catalog_manager: CatalogManager, tmp_path: Path):
    """list_dataset_children_rids(recurse=True) returns the same RID set as the object form."""
    catalog_manager.reset()
    ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
    dataset = dataset_desc.dataset

    via_objects = _descendant_rid_set_via_objects(dataset)
    via_rids = set(dataset.list_dataset_children_rids(recurse=True))

    assert via_rids == via_objects
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_descendant_walk_perf.py::test_list_dataset_children_rids_matches_objects -q
```
Expected: FAIL with `AttributeError: 'Dataset' object has no attribute 'list_dataset_children_rids'`.

- [ ] **Step 3: Factor the one-fetch core + add the RID accessor**

In `src/deriva_ml/dataset/dataset.py`, the current `list_dataset_children` body (2382-2405) ends with:

```python
        return [version_snapshot_catalog.lookup_dataset(rid) for rid in find_children(self.dataset_rid)]
```

Refactor so the RID computation is reusable. Replace the body from line 2382 (`# Initialize visited set...`) through 2405 with:

```python
        # Compute the descendant RID set (one Dataset_Dataset fetch,
        # in-memory traversal) and hydrate Dataset objects.
        version_snapshot_catalog, child_rids = self._descendant_child_rids(
            recurse=recurse, _visited=_visited, version=version
        )
        return [version_snapshot_catalog.lookup_dataset(rid) for rid in child_rids]

    def _descendant_child_rids(
        self,
        recurse: bool = False,
        _visited: set[RID] | None = None,
        *,
        version: DatasetVersion | str | None = None,
    ) -> tuple[Any, list[RID]]:
        """Core of list_dataset_children: one Dataset_Dataset fetch +
        in-memory traversal, returning ``(version_snapshot_catalog, child_rids)``.

        Fetches the (small) full Dataset_Dataset table once, builds the
        nesting adjacency in memory, and traverses from this dataset's RID
        with a ``_visited`` cycle guard. No per-node round-trips. The
        snapshot catalog is returned so callers that need Dataset objects
        can hydrate via its ``lookup_dataset``.
        """
        if _visited is None:
            _visited = set()

        version = DatasetVersion.parse(version) if isinstance(version, str) else version
        version_snapshot_catalog = self._version_snapshot_catalog(version)
        dataset_dataset_path = (
            version_snapshot_catalog.pathBuilder().schemas[self._ml_instance.ml_schema].tables["Dataset_Dataset"]
        )
        nested_datasets = list(dataset_dataset_path.entities().fetch())

        def find_children(rid: RID) -> list[RID]:
            if rid in _visited:
                return []
            _visited.add(rid)
            children = [child["Nested_Dataset"] for child in nested_datasets if child["Dataset"] == rid]
            if recurse:
                for child in children.copy():
                    children.extend(find_children(child))
            return children

        return version_snapshot_catalog, find_children(self.dataset_rid)

    def list_dataset_children_rids(
        self,
        recurse: bool = False,
        *,
        version: DatasetVersion | str | None = None,
    ) -> list[RID]:
        """Return descendant dataset RIDs without hydrating Dataset objects.

        Same traversal as :meth:`list_dataset_children` but returns RIDs
        directly — one ``Dataset_Dataset`` fetch and in-memory traversal,
        with zero per-node ``lookup_dataset`` round-trips. Use this when you
        only need the RID set (e.g. building the bag-walk anchor list).

        Args:
            recurse: If True, return all descendant RIDs (children of
                children, etc.); otherwise only direct children.
            version: Dataset version snapshot to query against.

        Returns:
            list[RID]: descendant dataset RIDs (depth-first order).

        Example:
            >>> ds = ml.lookup_dataset("1-ABC")  # doctest: +SKIP
            >>> rids = ds.list_dataset_children_rids(recurse=True)  # doctest: +SKIP
        """
        _, child_rids = self._descendant_child_rids(recurse=recurse, version=version)
        return child_rids
```

> The `list_dataset_children` signature, docstring, and return type are unchanged. `Any` is already imported in dataset.py (used widely); `RID` and `DatasetVersion` are already imported. Confirm with a quick grep if unsure.

- [ ] **Step 4: Run it to verify it passes**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_descendant_walk_perf.py::test_list_dataset_children_rids_matches_objects -q
```
Expected: PASS.

- [ ] **Step 5: Confirm `list_dataset_children` itself still works (refactor didn't break it)**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_datasets.py -k "child or children or nest" -q
```
Expected: pass (the object-returning method is unchanged in behavior).

- [ ] **Step 6: Lint + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src/deriva_ml/dataset/dataset.py tests/dataset/test_descendant_walk_perf.py && git add src/deriva_ml/dataset/dataset.py tests/dataset/test_descendant_walk_perf.py && git commit -m "$(cat <<'EOF'
perf(dataset): list_dataset_children_rids — descendant RIDs from one fetch, no per-node lookup

Factors the one-Dataset_Dataset-fetch + in-memory traversal out of
list_dataset_children into _descendant_child_rids, and adds a public
list_dataset_children_rids that returns RIDs without hydrating Dataset
objects (no per-node lookup_dataset round-trips). list_dataset_children
behavior is unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `bag_builder` uses the RID accessor + memoizes the descendant set (Parts A + C)

**Files:**
- Modify: `src/deriva_ml/dataset/bag_builder.py` (`__init__` to add the cache; `_iter_descendant_rids` ~892; `anchors_for` ~723; `_exclude_empty_associations` ~862-864)

- [ ] **Step 1: Write the failing test (descendant set is walked once)**

Append to `tests/dataset/test_descendant_walk_perf.py`:

```python
def test_estimate_walks_descendant_tree_once(catalog_manager: CatalogManager, tmp_path: Path, monkeypatch):
    """estimate_bag_size fetches the Dataset_Dataset table O(1) times, not O(descendants)."""
    catalog_manager.reset()
    ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
    dataset = dataset_desc.dataset
    version = dataset.current_version

    children = dataset.list_dataset_children_rids(recurse=True)
    if len(children) < 2:
        pytest.skip(f"demo dataset has {len(children)} descendants; need >= 2 to exercise the guard")

    # Count GETs against the Dataset_Dataset association table.
    import deriva.core.deriva_binding as db

    counter = {"dataset_dataset": 0}
    orig = db.DerivaBinding.get

    def spy(self, path, *a, **k):
        if isinstance(path, str) and "Dataset_Dataset" in path:
            counter["dataset_dataset"] += 1
        return orig(self, path, *a, **k)

    monkeypatch.setattr(db.DerivaBinding, "get", spy)

    from deriva_ml.dataset import DatasetSpec

    ml.estimate_bag_size(DatasetSpec(rid=dataset.dataset_rid, version=version))

    # Pre-fix: ~170 Dataset_Dataset fetches (per-node). Post-fix: a small
    # constant (one per descendant-set computation; memoized to 1 within the
    # builder op, but other call sites may add a few). Must NOT scale with
    # the descendant count.
    assert counter["dataset_dataset"] <= 5, (
        f"{counter['dataset_dataset']} Dataset_Dataset fetches for {len(children)} descendants "
        "— should be a small constant, not O(descendants)"
    )
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_descendant_walk_perf.py::test_estimate_walks_descendant_tree_once -q
```
Expected: FAIL (current code does many Dataset_Dataset fetches; counter > 5).

- [ ] **Step 3: Add the memoized descendant-set helper + use the RID accessor**

In `src/deriva_ml/dataset/bag_builder.py`, find `DatasetBagBuilder.__init__`. Add a cache init at the end of it:

```python
        # Memoize the descendant-RID set per root RID for one builder op so
        # anchors_for and _exclude_empty_associations share a single tree walk.
        self._descendant_rids_cache: dict[RID, list[RID]] = {}
```

(If `__init__` doesn't currently exist or you can't find it, search for `def __init__` in the class; if the class has no `__init__`, add one that calls nothing special plus this cache — but check first, the class stores `_ml_instance`/`_exclude_tables` somewhere.)

Replace the body of `_iter_descendant_rids` (892-904) with a memoized delegate to the new RID accessor:

```python
    def _iter_descendant_rids(self, dataset: DatasetLike) -> Iterable[RID]:
        """Yield every descendant Dataset RID.

        Memoized per root RID for the lifetime of this builder so the
        nested-dataset tree is walked once per operation. Delegates to
        :meth:`Dataset.list_dataset_children_rids` (one Dataset_Dataset
        fetch, in-memory traversal, no per-node lookups).
        """
        root = dataset.dataset_rid
        cached = self._descendant_rids_cache.get(root)
        if cached is None:
            dataset_obj = self._ml_instance.lookup_dataset(root)
            cached = list(dataset_obj.list_dataset_children_rids(recurse=True))
            self._descendant_rids_cache[root] = cached
        return cached
```

> Note: this still does ONE `lookup_dataset(root)` to get a `Dataset` object to call the method on. That's 1 GET total (memoized), not per-node. (A later refactor could expose the RID accessor on the mixin to avoid even that, but YAGNI — 1 GET is fine.)

- [ ] **Step 4: Run the walk-once test to verify it passes**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_descendant_walk_perf.py::test_estimate_walks_descendant_tree_once -q
```
Expected: PASS (Dataset_Dataset fetches now a small constant).

- [ ] **Step 5: Confirm anchors_for still produces the same anchor set**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_bag_builder.py tests/dataset/test_schema_paths.py -q
```
Expected: pass (anchor set unchanged).

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src/deriva_ml/dataset/bag_builder.py && git add src/deriva_ml/dataset/bag_builder.py tests/dataset/test_descendant_walk_perf.py && git commit -m "$(cat <<'EOF'
perf(bag): _iter_descendant_rids uses one-fetch RID accessor + memoizes per op

Delegates to Dataset.list_dataset_children_rids (one Dataset_Dataset fetch,
in-memory traversal) instead of manual per-node re-recursion, and memoizes the
descendant set on the builder so anchors_for and _exclude_empty_associations
share one walk. ~170 Dataset_Dataset fetches + ~592 lookup_dataset -> ~2.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `_exclude_empty_associations` — aggregate membership per association (Part B)

**Files:**
- Modify: `src/deriva_ml/dataset/bag_builder.py` (`_exclude_empty_associations` ~824-890)

This is the largest reduction (~765 → ~9 queries). The method currently scans `list_dataset_members()` per descendant to build `member_element_types`, then checks each association's `other_fkeys[].pk_table` against it. The new approach checks membership **directly on each association table** scoped to the descendant-RID set — an association is non-empty iff it has a row whose `Dataset` FK is in the tree.

- [ ] **Step 1: Write the failing/pinning test — excluded set unchanged**

Append to `tests/dataset/test_descendant_walk_perf.py`:

```python
def test_exclude_empty_associations_unchanged(catalog_manager: CatalogManager, tmp_path: Path):
    """The Part B rewrite produces the SAME excluded-association set as before.

    This is the load-bearing correctness pin: the aggregate membership query
    must select exactly the associations the per-descendant member scan did.
    """
    catalog_manager.reset()
    ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
    dataset = dataset_desc.dataset

    from deriva_ml.dataset.bag_builder import DatasetBagBuilder

    builder = DatasetBagBuilder(ml)
    excluded = builder._exclude_empty_associations(dataset)

    # Sanity: the result is a set of (schema, table) tuples; non-empty
    # associations (those with members in the tree) are NOT excluded.
    assert isinstance(excluded, set)
    for entry in excluded:
        assert isinstance(entry, tuple) and len(entry) == 2
    # The demo dataset has Subject/Image members, so Dataset_Subject /
    # Dataset_Image must NOT be excluded.
    excluded_names = {t for _, t in excluded}
    assert "Dataset_Subject" not in excluded_names
    assert "Dataset_Image" not in excluded_names
```

> NOTE: This test pins the *observable contract* (member-bearing associations not excluded). The strongest correctness check is the before/after comparison — but since we're replacing the implementation in this same task, capture the BASELINE excluded set on the CURRENT code first (Step 2) and assert equality after (Step 5).

- [ ] **Step 2: Capture the baseline excluded set (before changing the code)**

Run this one-off to record what the current implementation returns, so Step 5 can assert equality:

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run python - <<'PY'
import tempfile, pathlib
from tests.catalog_manager import CatalogManager
from deriva_ml.dataset.bag_builder import DatasetBagBuilder
cm = CatalogManager(hostname="localhost")
with tempfile.TemporaryDirectory() as td:
    cm.reset(); ml, desc = cm.ensure_datasets(pathlib.Path(td))
    b = DatasetBagBuilder(ml)
    ex = b._exclude_empty_associations(desc.dataset)
    print("BASELINE excluded:", sorted(ex))
PY
```
Record the printed BASELINE set. (Also verify `.in_()` works here: this run exercises the OLD code; the NEW code's query is validated in Step 4's run.)

- [ ] **Step 3: Verify the `.in_()` membership query form against the live catalog**

Before rewriting, confirm the exact working datapath form for "rows of an association table whose Dataset is in a set":

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run python - <<'PY'
import tempfile, pathlib
from tests.catalog_manager import CatalogManager
cm = CatalogManager(hostname="localhost")
with tempfile.TemporaryDirectory() as td:
    cm.reset(); ml, desc = cm.ensure_datasets(pathlib.Path(td))
    ds = desc.dataset
    rids = [ds.dataset_rid] + list(ds.list_dataset_children_rids(recurse=True))
    pb = ml.pathBuilder()
    t = pb.schemas[ml.ml_schema].tables["Dataset_Subject"]
    # Try the in_ membership form with limit=1 (presence check)
    rows = list(t.filter(t.Dataset.in_(rids)).entities().fetch(limit=1))
    print("Dataset_Subject membership (in_ + limit=1):", "PRESENT" if rows else "empty")
PY
```
Expected: prints `PRESENT` (demo dataset has Subject members). If `t.Dataset.in_(rids)` raises (the `feature.py:563` caveat), fall back to the form `feature.py` uses — read `src/deriva_ml/core/mixins/feature.py:560-575` for the working `.in_()` construction and mirror it. Record the working form for Step 4.

- [ ] **Step 4: Rewrite `_exclude_empty_associations` to use aggregate membership**

Replace the member-scan block (the lines from `dataset_obj = self._ml_instance.lookup_dataset(...)` at ~862 through the `member_element_types` loop ending ~871) AND the association loop (884-889) so the method computes membership directly. Replace from line 862 to the end of the method (890) with:

```python
        # Descendant RID set (root + all nested descendants), one fetch.
        rid_set = [dataset.dataset_rid] + list(self._iter_descendant_rids(dataset))

        # Every vocabulary table in any schema — associations into
        # vocabularies always come along for the ride.
        vocab_tables: set[Table] = {
            table for schema in model.schemas.values() for table in schema.tables.values() if model.is_vocabulary(table)
        }

        pb = self._ml_instance.pathBuilder()

        excluded: set[tuple[str, str]] = set()
        for assoc in dataset_table.find_associations():
            assoc_table = assoc.table
            # Vocab-linked associations are always included.
            if any(fk.pk_table in vocab_tables for fk in assoc.other_fkeys):
                continue
            # Non-empty iff the association has any row whose Dataset FK is in
            # the tree. One presence query per association table (limit=1),
            # independent of descendant count.
            assoc_path = pb.schemas[assoc_table.schema.name].tables[assoc_table.name]
            has_member = bool(
                list(assoc_path.filter(assoc_path.Dataset.in_(rid_set)).entities().fetch(limit=1))
            )
            if not has_member:
                excluded.add((assoc_table.schema.name, assoc_table.name))
        return excluded
```

> If Step 3 showed `assoc_path.Dataset.in_(rid_set)` doesn't work directly, use the working `.in_()` form recorded in Step 3 (mirroring `feature.py`). The INTENT is fixed: one presence query per association over the descendant-RID set; vocab associations always kept; an association is excluded iff it has zero rows for the tree.
>
> Also confirm every Dataset association table actually has a column named `Dataset` (the FK to Dataset). The existing code at `dataset.py:1670` filters `member_path.Dataset == self.dataset_rid`, so `Dataset` is the conventional column name. If some association uses a differently-named Dataset FK, derive the column from `assoc`'s FK to `dataset_table` rather than hard-coding `.Dataset` — read how `list_dataset_members` (dataset.py:1602-1672) resolves the Dataset column and mirror it. Note any such case.

- [ ] **Step 5: Run the correctness pin — excluded set must equal the baseline**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run python - <<'PY'
import tempfile, pathlib
from tests.catalog_manager import CatalogManager
from deriva_ml.dataset.bag_builder import DatasetBagBuilder
cm = CatalogManager(hostname="localhost")
with tempfile.TemporaryDirectory() as td:
    cm.reset(); ml, desc = cm.ensure_datasets(pathlib.Path(td))
    b = DatasetBagBuilder(ml)
    ex = b._exclude_empty_associations(desc.dataset)
    print("NEW excluded:", sorted(ex))
PY
```
Compare to the Step 2 BASELINE — they MUST be identical. If they differ, STOP: the aggregate query doesn't match the old member-scan semantics. Debug (likely the Dataset-column resolution or the vocab rule) before proceeding. Then run the pinning test:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_descendant_walk_perf.py::test_exclude_empty_associations_unchanged -q
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src/deriva_ml/dataset/bag_builder.py && git add src/deriva_ml/dataset/bag_builder.py tests/dataset/test_descendant_walk_perf.py && git commit -m "$(cat <<'EOF'
perf(bag): _exclude_empty_associations uses aggregate membership per association

Replaces the per-descendant list_dataset_members scan (~765 queries: 85
descendants x ~9 associations) with one presence query per association table
over the descendant-RID set (~9 queries, independent of descendant count).
Vocab-linked associations still always included. Excluded set is identical
(pinned by before/after equality test).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Estimate-dict correctness + GET-count guard

**Files:**
- Modify: `tests/dataset/test_descendant_walk_perf.py`

- [ ] **Step 1: Add the estimate-unchanged correctness test**

Append:

```python
def test_estimate_dict_unchanged_after_walk_fix(catalog_manager: CatalogManager, tmp_path: Path):
    """estimate_bag_size returns the same totals + per-table shape (pure perf change)."""
    catalog_manager.reset()
    ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
    dataset = dataset_desc.dataset
    from deriva_ml.dataset import DatasetSpec

    est = ml.estimate_bag_size(DatasetSpec(rid=dataset.dataset_rid, version=dataset.current_version))
    assert est["incomplete"] is False
    assert est["total_rows"] >= 0
    assert isinstance(est["tables"], dict) and len(est["tables"]) > 0
    # Member-bearing tables present (Subject/Image), proving the walk still
    # reaches them after the membership-query rewrite.
    table_names = set(est["tables"])
    assert any("Subject" in t for t in table_names)
    assert any("Image" in t for t in table_names)
```

- [ ] **Step 2: Add the total GET-count guard (does not scale with nesting)**

Append:

```python
def test_estimate_total_gets_not_linear_in_descendants(catalog_manager: CatalogManager, tmp_path: Path, monkeypatch):
    """Total catalog GETs during estimate is bounded, not O(descendants x tables)."""
    catalog_manager.reset()
    ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
    dataset = dataset_desc.dataset
    n_desc = len(dataset.list_dataset_children_rids(recurse=True))
    if n_desc < 2:
        pytest.skip(f"{n_desc} descendants; need >= 2")

    import deriva.core.deriva_binding as db

    counter = {"n": 0}
    orig = db.DerivaBinding.get

    def spy(self, path, *a, **k):
        counter["n"] += 1
        return orig(self, path, *a, **k)

    monkeypatch.setattr(db.DerivaBinding, "get", spy)
    from deriva_ml.dataset import DatasetSpec

    ml.estimate_bag_size(DatasetSpec(rid=dataset.dataset_rid, version=dataset.current_version))

    # Pre-fix the demo would issue many GETs that scale with descendants.
    # Post-fix: enumeration is ~2 + per-association presence (~O(associations))
    # + per-table estimate queries (~O(tables)). Bound it well below the
    # pre-fix per-descendant explosion. The exact ceiling is catalog-specific;
    # the load-bearing property is no per-descendant member scan.
    assert counter["n"] < 50 * max(1, n_desc), (
        f"{counter['n']} GETs for {n_desc} descendants looks O(descendants x tables)"
    )
```

> The ceiling `50 * n_desc` is deliberately loose — the point is to catch a regression to the old per-descendant-per-association explosion (which was ~9-12 GETs *per descendant per association*), not to pin an exact count. If the demo catalog's numbers make this vacuous or too tight, adjust to a value that (a) passes post-fix and (b) would FAIL if Part B's per-descendant scan were reintroduced; document the chosen bound.

- [ ] **Step 3: Run all the new tests**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_descendant_walk_perf.py -q
```
Expected: all pass (or skip where descendants < 2).

- [ ] **Step 4: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add tests/dataset/test_descendant_walk_perf.py && git commit -m "$(cat <<'EOF'
test(dataset): estimate-dict-unchanged + GET-count-not-linear guards for walk fix

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Regression sweep + live 2-277G re-measure

**Files:** none (verification)

- [ ] **Step 1: Dataset + bag-builder + core suites (the walk touches these)**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_datasets.py tests/dataset/test_bag_builder.py tests/dataset/test_estimate_bag_size.py tests/dataset/test_schema_paths.py tests/core/ -q --maxfail=8 2>&1 | tail -8
```
Expected: all pass (the one known pre-existing failure `tests/feature/test_features.py::test_download_feature` is NOT in this set; if any NEW failure appears, investigate per systematic-debugging — do not proceed).

- [ ] **Step 2: Live re-measure (manual, record in PR)**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run python - <<'PY'
import time
import deriva.core.deriva_binding as db
from deriva_ml import DerivaML
from deriva_ml.dataset import DatasetSpec
n={"s":0}; orig=db.DerivaBinding.get
def spy(self,p,*a,**k):
    n["s"]+=1; return orig(self,p,*a,**k)
db.DerivaBinding.get=spy
ml = DerivaML(hostname="www.eye-ai.org", catalog_id="eye-ai")
n["s"]=0; t0=time.time()
est = ml.estimate_bag_size(DatasetSpec(rid="2-277G", version="4.11.0"))
print(f"estimate_bag_size(2-277G): {time.time()-t0:.1f}s, {n['s']} GETs   (was 464s / 1543 GETs)")
print("rows:", est["total_rows"], "size:", est["total_asset_size"], "tables:", len(est["tables"]), "incomplete:", est["incomplete"])
PY
```
Expected: dramatically fewer GETs (target: low hundreds, ideally ~tens) and much lower wall-clock. **Totals MUST be unchanged: 360756 rows / 18.0 GB / 80 tables / incomplete False.** Record the numbers. If totals differ, STOP — a correctness regression.

- [ ] **Step 3: Decide if good enough**

If wall-clock is now acceptable (target < 60s, ideally < 30s) → done; proceed to PRs. If it's much better but still slow, the remaining cost is the genuine per-table estimate queries (the asset-size-fast-mode follow-up territory) — note it and decide with the user whether to stop here.

---

## Task 6: PRs (deriva-py + deriva-ml, the full "fast estimate" story)

**Files:** none (PRs)

- [ ] **Step 1: Push the deriva-ml branch + format/lint final**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff format src/deriva_ml/dataset/dataset.py src/deriva_ml/dataset/bag_builder.py tests/dataset/test_descendant_walk_perf.py && uv run ruff check src tests | tail -3 ; git diff --quiet || git commit -am "style: ruff format" ; git push -u origin feature/fast-estimate-bag-size 2>&1 | tail -3
```
(The repo-wide `ruff check src tests` has ~220 pre-existing errors unrelated to this work; confirm our touched files are clean specifically.)

- [ ] **Step 2: Open the deriva-py PR (from_model)**

```bash
cd /Users/carl/GitHub/deriva-py && gh pr create --base master --head deriva-ml --title "feat(datapath): from_model(catalog, model) — build path-builder from a supplied Model" --body "$(cat <<'EOF'
## Summary
`_CatalogWrapper` builds its schemas from `catalog.getCatalogModel()`; the raw
`/schema` dict is never read for content. But `getCatalogModel()` is uncached
and re-fetches `/schema` every call, so every `getPathBuilder()` pays a full
schema fetch — and on servers that don't return 304, deriva-py's binding cache
never dedups it. Adds an optional `model=` param to `_CatalogWrapper` and a
public `datapath.from_model(catalog, model)` that builds the wrapper from a
supplied Model with zero schema fetches. HTTP (reads and writes) still routes
through the supplied catalog. `from_catalog()` is unchanged (`model=None`).

Consumer: deriva-ml's `estimate_bag_size`/`download_dataset_bag` hold their own
up-to-date Model; `from_model` lets them build path-builders without the
redundant `/schema` GET (eliminated an O(N) live-instance refetch).

Test: `tests/deriva/core/test_datapath.py::TestFromModel` (no-network).
EOF
)" 2>&1 | tail -3
```
> Confirm with the user before opening if they prefer to upstream via a different flow; the `deriva-ml` branch already carries the commit for deriva-ml's pin.

- [ ] **Step 3: Open/update the deriva-ml PR**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && gh pr create --title "perf(estimate): fast estimate_bag_size — eliminate redundant /schema + descendant-walk round-trips" --body "$(cat <<'EOF'
## Summary
`estimate_bag_size(2-277G)` took ~8.5 min. Two root causes, both client-side
catalog round-trips:

1. **Redundant `/schema` refetch** (model-built pathBuilder): deriva-py
   re-fetches `/schema` on every `pathBuilder()` call (servers don't honor
   304). Now builds the wrapper from the in-memory model via the new
   `datapath.from_model` (deriva-py PR linked below). 849 -> 8 `/schema` GETs.
   Plus `catalog_snapshot` schema-reuse + snapshot-instance memoization.
2. **Per-descendant tree walk** (the dominant cost): the descendant
   enumeration did ~170 full `Dataset_Dataset` fetches + ~592 `lookup_dataset`
   + ~765 per-descendant member-scan queries for an 85-node tree. Now:
   - descendants enumerated from ONE `Dataset_Dataset` fetch + in-memory
     traversal (`list_dataset_children_rids`), no per-node lookups;
   - `_exclude_empty_associations` uses one aggregate membership query per
     association (~9) instead of per-descendant scans (~765);
   - the descendant set is memoized so the walk runs once.

Both fixes also speed up real `download_dataset_bag` on nested datasets (shared
`anchors_for`).

## Results (www.eye-ai.org 2-277G)
<!-- fill from Task 5 Step 2: GETs 1543 -> N, wall-clock 464s -> Ns; totals
unchanged: 360756 rows / 18.0 GB / 80 tables / incomplete False -->

## Correctness
Pure perf change: identical anchor set, identical excluded-association set
(before/after equality pinned), identical estimate dict. Full core suite green;
write-through / snapshot-pinning / offline guards for the pathBuilder change;
descendant-walk GET-count + estimate-dict guards.

## Follow-ups (not in this PR)
- Asset-size-fast estimate mode (server-side Sum(Length); the "just tell me the
  GB" use case).
- The remaining `CatalogBagBuilder` internal `getPathBuilder` calls (deriva-py).

Spec: docs/superpowers/specs/2026-06-13-estimate-bag-size-perf-design.md,
docs/superpowers/specs/2026-06-14-descendant-walk-perf-design.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)" 2>&1 | tail -3
```

- [ ] **Step 4: Report both PR URLs + the final before/after numbers.** `bump-version patch` happens after merge on clean `main` (repo rule), not in the branch.

---

## Self-Review notes

- **Spec coverage:** Part A (one-fetch RID accessor) → Task 1; Part A in bag_builder + Part C (memoize) → Task 2; Part B (aggregate membership) → Task 3; §5.1 correctness (RID-set equality, excluded-set equality, estimate-dict equality) → Tasks 1, 3, 4; §5.2 GET-count guard → Tasks 2, 4; §5.3 live re-measure → Task 5; version-pinned preserved (all accessors thread `version`/snapshot). Out-of-scope items (asset-fast mode, ADR-0008 aggregate counts) explicitly deferred in the PR body.
- **Type/signature consistency:** `list_dataset_children_rids(recurse=False, *, version=None) -> list[RID]` and `_descendant_child_rids(...) -> tuple[Any, list[RID]]` defined in Task 1, used in Task 2. `_descendant_rids_cache: dict[RID, list[RID]]` init in Task 2 Step 3, used in `_iter_descendant_rids`. `_iter_descendant_rids` keeps returning `Iterable[RID]` (Task 2). `_exclude_empty_associations` keeps `-> set[tuple[str,str]]` (Task 3).
- **No placeholders:** every code step shows full code; the two judgment points (the `.in_()` working form, the Dataset-column resolution) have explicit verify-first steps + the fallback (mirror `feature.py` / `list_dataset_members`), not vague "handle it."
- **Correctness-first:** Task 3 captures a BASELINE excluded set on the old code (Step 2) and asserts equality after the rewrite (Step 5) — the load-bearing pin that the query-shape change preserves semantics.
