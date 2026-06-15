# Reachability Engine + Fast Estimate — Implementation Plan (Stage A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `estimate_bag_size`'s ~377s of server-side deep-FK-join queries with a client-side FK-reachability engine that fetches each reached table's edge columns once and computes exact per-table RID-union counts in memory (~27s, same exact answer).

**Architecture:** A new pure-Python module `dataset/_reachability.py` owns the engine: given the bag walk's `{(schema,table): [fk_path, ...]}` reached-paths map + anchor RIDs + the catalog model, it (1) fetches each involved edge/target table once projecting only RID + FK-join columns + asset `Length`, (2) reconstructs reachability by in-memory BFS over the symbolic FK paths, hash-indexed, and (3) returns the same `rids_by_table` / `asset_lengths_by_table` dicts the existing `assemble_estimate` already consumes. `estimate_bag_size` swaps its query layer (`build_estimate_queries` + `run_estimate_queries`) for one `compute_reachability(...)` call; `assemble_estimate` is unchanged. The deep-join path is deleted (no backwards-compat shim, per CLAUDE.md).

**Tech Stack:** Python ≥3.12, deriva-py datapath/model API, deriva-ml `DatasetBagBuilder`, pytest. The engine is pure (no async, no httpx) — its only catalog contact is whole-table projected fetches via the path builder.

---

## Why this is Stage A of two

This plan is pure deriva-ml and ships the estimate win standalone. Stage B
(separate plan, written after this lands) productionizes the
`csv-ridset` chunk-append processor in deriva-py and rewires bag
*generation* to consume the same reachability engine. The reachability
engine built here is the shared core both stages use — Stage B imports
`compute_reachability` to get its per-table RID sets. Design context:
- `docs/superpowers/specs/2026-06-14-portable-bag-csv-contract.md` (the bag format / processor design)
- `docs/adr/0008-estimate-bag-size-bypasses-bag-pipeline.md` (the estimate's deliberate engine bypass — this plan deepens that bypass, it does not undo it)
- Memory: `estimate-bag-size-client-side-join.md` (the proven approach + the vocab-FK `Name` subtlety) and `csv-ridset-chunk-append-proto.md` (Stage B's de-risked piece).

The proven prototype this plan productionizes is `/tmp/cside_estimate_proto.py` — **27s EXACT match on all 80 tables + 18 GB** for eye-ai 2-277G, vs ~280s server. The plan turns that throwaway into tested, documented production code.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/deriva_ml/dataset/_reachability.py` | The engine: edge resolution, projected fetch, in-memory BFS, RID-union per table. Pure module — input is reached-paths + anchors + model + a fetch callable; output is the two result dicts. | **Create** |
| `tests/dataset/test_reachability.py` | Unit tests for the pure engine pieces (FK-column resolution, BFS over a synthetic edge graph, vocab-FK `Name` join, multi-path union) with a synthetic in-memory catalog — no live server. | **Create** |
| `src/deriva_ml/dataset/dataset.py` | `estimate_bag_size` swaps the query/orchestration layer for one `compute_reachability(...)` call; keeps snapshot resolution + `assemble_estimate`. | **Modify** (~2655–2790) |
| `src/deriva_ml/dataset/_estimate.py` | Delete `build_estimate_queries` + `run_estimate_queries` + `QueryItem` + `QueryType` + `DEFAULT_ESTIMATE_CONCURRENCY` (the deep-join path). Keep `assemble_estimate` (still consumed). | **Modify** |
| `tests/dataset/test_estimate_helpers.py` | Drop tests of the deleted functions; keep `assemble_estimate` tests. | **Modify** |
| `tests/dataset/test_estimate_bag_size.py` (or wherever the live estimate integration test lives) | Add a correctness assertion: client-side engine result == reference server-union for the demo catalog. | **Modify** |
| `docs/reference/bag-export.md` | Update the "Same engine, different consumer" section: the estimate now shares the *walk* AND uses a client-side reachability engine instead of live aggregate queries. | **Modify** (~277–285) |
| `docs/adr/0008-estimate-bag-size-bypasses-bag-pipeline.md` | Note the engine change: bypass is now via client-side reachability, not per-path aggregate queries. | **Modify** |

---

## Engine interface (locked here so all tasks agree)

```python
# src/deriva_ml/dataset/_reachability.py

ReachedPaths = dict[tuple[str, str], list[tuple[tuple[str, str], ...]]]
"""{(schema, table): [ fk_path, ... ]} from CatalogBagBuilder.iter_reached_paths().
Each fk_path is a tuple of (schema, table) segments, anchor-first."""

FetchFn = Callable[[str, str, set[str]], list[dict]]
"""(schema, table, columns) -> list of row dicts. Injected so the engine
is testable without a live catalog. Production binds it to the path builder."""


def compute_reachability(
    *,
    reached: ReachedPaths,
    anchor_rids: list[str],
    model: Any,                 # deriva-py Model (cb._get_model())
    fetch: FetchFn,
) -> tuple[dict[str, set[str]], dict[str, dict[str, int]]]:
    """Return (rids_by_table, asset_lengths_by_table).

    rids_by_table:          table_name -> set of reachable RIDs (exact union
                            across all FK paths to that table).
    asset_lengths_by_table: table_name -> {RID: Length} for asset tables.

    Shapes match run_estimate_queries' first two outputs so assemble_estimate
    consumes them unchanged. (sample_rows + failed_by_table are dropped — the
    engine fetches whole tables, so CSV-byte sampling uses the fetched rows
    directly and there are no per-query partial failures.)
    """
```

Note the deliberate shape match: `assemble_estimate` already takes
`rids_by_table` and `asset_lengths_by_table` as its first two result dicts.
This plan removes the `sample_rows_by_table` / `failed_by_table` inputs (see
Task 6 for how `assemble_estimate`'s signature adapts).

---

### Task 1: Engine module scaffold + FK-column resolution

**Files:**
- Create: `src/deriva_ml/dataset/_reachability.py`
- Test: `tests/dataset/test_reachability.py`

The first pure piece: given two adjacent `(schema, table)` segments and a model, return the column pairs that join them and which side holds the FK. This mirrors the prototype's `fk_join_columns` (lines 84–113) and `CatalogBagBuilder._composite_fk_on_clause`.

- [ ] **Step 1: Write the failing test**

Use a synthetic model with a hand-built FK so no catalog is needed. The test model has `Parent(RID)` and `Child(RID, parent_fk -> Parent.RID)`.

```python
# tests/dataset/test_reachability.py
"""Unit tests for the client-side FK-reachability engine (no live catalog)."""
from types import SimpleNamespace

from deriva_ml.dataset._reachability import _fk_join_columns


def _col(name):
    return SimpleNamespace(name=name)


def _make_model():
    """Minimal model: Child.parent_fk -> Parent.RID (FK lives on Child)."""
    parent = SimpleNamespace(foreign_keys=[], referenced_by=[])
    fk = SimpleNamespace(
        foreign_key_columns=[_col("parent_fk")],
        referenced_columns=[_col("RID")],
    )
    child = SimpleNamespace(foreign_keys=[fk], referenced_by=[])
    fk.pk_table = parent
    parent.referenced_by = [fk]
    schemas = {"S": SimpleNamespace(tables={"Parent": parent, "Child": child})}
    return SimpleNamespace(schemas=schemas)


def test_fk_join_columns_fk_on_child():
    model = _make_model()
    # Following Parent -> Child: the FK lives on Child (cur).
    constraints = _fk_join_columns(("S", "Parent"), ("S", "Child"), model)
    assert constraints == [{"fk_on": "cur", "pairs": [("RID", "parent_fk")]}]


def test_fk_join_columns_fk_on_parent():
    model = _make_model()
    # Following Child -> Parent: the FK lives on Child (prev).
    constraints = _fk_join_columns(("S", "Child"), ("S", "Parent"), model)
    assert constraints == [{"fk_on": "prev", "pairs": [("parent_fk", "RID")]}]


def test_fk_join_columns_no_edge_returns_empty():
    model = _make_model()
    # Parent has no FK to itself.
    assert _fk_join_columns(("S", "Parent"), ("S", "Parent"), model) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_reachability.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'deriva_ml.dataset._reachability'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/deriva_ml/dataset/_reachability.py
"""Client-side FK-reachability engine for fast bag-size estimation.

Replaces the deep-server-join estimate path (which issued ~477 multi-hop
FK-JOIN queries the server evaluated at 16-60s each) with: fetch each
reached table's edge columns ONCE (whole-table projected scan, ~0.3s for
240k rows), then reconstruct FK reachability by in-memory BFS over the
symbolic FK paths the bag walker discovered. Exact union counts, ~10x
faster. See docs/adr/0008 and the estimate-bag-size-client-side-join memo.

The engine is pure: its only catalog contact is the injected ``fetch``
callable, so the BFS and FK-resolution logic are unit-testable against a
synthetic in-memory catalog with no live server.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

ReachedPaths = dict[tuple[str, str], list[tuple[tuple[str, str], ...]]]
FetchFn = Callable[[str, str, set[str]], list[dict]]


def _fk_join_columns(
    prev_seg: tuple[str, str],
    cur_seg: tuple[str, str],
    model: Any,
) -> list[dict]:
    """Return the FK constraints linking two adjacent path segments.

    Each constraint dict is ``{"fk_on": "prev"|"cur", "pairs": [(prev_col,
    cur_col), ...]}`` where ``pairs[i]`` are column NAMES that must be equal
    to follow ``prev -> cur``. ``fk_on`` records which table physically
    holds the FK, which decides the in-memory join direction.

    Args:
        prev_seg: ``(schema, table)`` of the source segment.
        cur_seg: ``(schema, table)`` of the destination segment.
        model: deriva-py Model.

    Returns:
        List of constraint dicts (one per FK constraint between the tables,
        in either direction); empty if no FK links them.

    Example:
        >>> # See tests/dataset/test_reachability.py for synthetic-model
        >>> # exercises; this needs a Model object.
        >>> ...  # doctest: +SKIP
    """
    ps, pt = prev_seg
    cs, ct = cur_seg
    prev_tbl = model.schemas[ps].tables[pt]
    cur_tbl = model.schemas[cs].tables[ct]
    constraints: list[dict] = []
    # FK on prev pointing to cur.
    for fk in prev_tbl.foreign_keys:
        if fk.pk_table is cur_tbl:
            pairs = [
                (fkc.name, rc.name)
                for fkc, rc in zip(fk.foreign_key_columns, fk.referenced_columns)
            ]
            constraints.append({"fk_on": "prev", "pairs": pairs})
    # FK on cur pointing to prev.
    for fk in cur_tbl.foreign_keys:
        if fk.pk_table is prev_tbl:
            pairs = [
                (rc.name, fkc.name)  # (prev_col, cur_col)
                for fkc, rc in zip(fk.foreign_key_columns, fk.referenced_columns)
            ]
            constraints.append({"fk_on": "cur", "pairs": pairs})
    return constraints
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_reachability.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/dataset/_reachability.py tests/dataset/test_reachability.py
git commit -m "feat(estimate): FK-column resolution for reachability engine"
```

---

### Task 2: Needed-columns projection (incl. vocab-FK Name subtlety)

**Files:**
- Modify: `src/deriva_ml/dataset/_reachability.py`
- Test: `tests/dataset/test_reachability.py`

Compute the minimal column set to fetch per table: `RID` + outbound-FK columns + **inbound-FK referenced columns** (the load-bearing vocab-FK `Name` subtlety) + asset `Length`. Mirrors prototype `needed_columns` (lines 121–148).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/dataset/test_reachability.py
from deriva_ml.dataset._reachability import _needed_columns


def _make_vocab_model():
    """Asset_Role(Name) <- Child.role_fk -> Asset_Role.Name (vocab FK to Name)."""
    role = SimpleNamespace(foreign_keys=[], column_definitions=SimpleNamespace(elements={}))
    vocab_fk = SimpleNamespace(
        foreign_key_columns=[_col("role_fk")],
        referenced_columns=[_col("Name")],  # references Name, NOT RID
    )
    child = SimpleNamespace(
        foreign_keys=[vocab_fk],
        referenced_by=[],
        column_definitions=SimpleNamespace(elements={}),
    )
    vocab_fk.pk_table = role
    role.referenced_by = [vocab_fk]
    role.foreign_keys = []
    child.is_asset = lambda: False
    role.is_asset = lambda: False
    schemas = {"S": SimpleNamespace(tables={"Asset_Role": role, "Child": child})}
    return SimpleNamespace(schemas=schemas)


def test_needed_columns_includes_inbound_referenced_name():
    model = _make_vocab_model()
    # Asset_Role is the FK TARGET of an inbound vocab FK on Name. The engine
    # must project Name or the in-memory join silently drops rows.
    cols = _needed_columns(("S", "Asset_Role"), model)
    assert "Name" in cols
    assert "RID" in cols


def test_needed_columns_includes_outbound_fk_and_length():
    """Asset table with an outbound FK and a Length column."""
    length_col = SimpleNamespace(name="Length")
    fk = SimpleNamespace(foreign_key_columns=[_col("parent_fk")], referenced_columns=[_col("RID")])
    asset = SimpleNamespace(
        foreign_keys=[fk],
        referenced_by=[],
        column_definitions=SimpleNamespace(elements={"Length": length_col}),
    )
    asset.is_asset = lambda: True
    model = SimpleNamespace(schemas={"S": SimpleNamespace(tables={"A": asset})})
    cols = _needed_columns(("S", "A"), model)
    assert cols == {"RID", "parent_fk", "Length"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_reachability.py::test_needed_columns_includes_inbound_referenced_name -v`
Expected: FAIL — `ImportError: cannot import name '_needed_columns'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/deriva_ml/dataset/_reachability.py
def _needed_columns(seg: tuple[str, str], model: Any) -> set[str]:
    """Minimal column projection to fetch for one table in the walk.

    Returns ``RID`` plus every column the in-memory join needs:

    - **Outbound FK columns** (this table -> parent): needed when this table
      is the FK-holder for a hop.
    - **Inbound-FK *referenced* columns** (child -> this table): needed when
      this table is the FK TARGET. Critically, a vocabulary FK references the
      target's ``Name`` (e.g. ``*_Execution.Asset_Role -> Asset_Role.Name``),
      not ``RID`` — omitting ``Name`` silently undercounts vocab-leaf tables
      (the prototype was off by 817 rows until this was added).
    - **Asset ``Length``**: for asset-byte summation.

    Args:
        seg: ``(schema, table)``.
        model: deriva-py Model.

    Returns:
        Set of column names to project in the table fetch.

    Example:
        >>> ...  # doctest: +SKIP  (needs a Model)
    """
    s, t = seg
    tbl = model.schemas[s].tables[t]
    cols = {"RID"}
    for fk in tbl.foreign_keys:
        for c in fk.foreign_key_columns:
            cols.add(c.name)
    for fk in tbl.referenced_by:
        for c in fk.referenced_columns:
            cols.add(c.name)
    if tbl.is_asset() and "Length" in tbl.column_definitions.elements:
        cols.add("Length")
    return cols
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_reachability.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/dataset/_reachability.py tests/dataset/test_reachability.py
git commit -m "feat(estimate): minimal column projection incl. vocab-FK Name"
```

---

### Task 3: Single-path BFS reachability

**Files:**
- Modify: `src/deriva_ml/dataset/_reachability.py`
- Test: `tests/dataset/test_reachability.py`

Given fetched rows, one FK path, and anchor RIDs, return the set of terminal-table RIDs reachable along that path. Mirrors prototype `reached_rids_for_path` (lines 213–285) with the hash-indexed FK probe. This is the algorithmic core.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/dataset/test_reachability.py
from deriva_ml.dataset._reachability import _reached_rids_for_path


def test_reached_rids_single_hop_fk_on_cur():
    """Anchor Parent={p1}; Child rows c1,c2 point to p1, c3 to p2.
    Following Parent -> Child should reach {c1, c2}."""
    model = _make_model()  # Child.parent_fk -> Parent.RID
    fetched = {
        ("S", "Parent"): [{"RID": "p1"}, {"RID": "p2"}],
        ("S", "Child"): [
            {"RID": "c1", "parent_fk": "p1"},
            {"RID": "c2", "parent_fk": "p1"},
            {"RID": "c3", "parent_fk": "p2"},
        ],
    }
    fk_path = (("S", "Parent"), ("S", "Child"))
    result = _reached_rids_for_path(
        fk_path, anchor_rids={"p1"}, fetched_rows=fetched, model=model
    )
    assert result == {"c1", "c2"}


def test_reached_rids_anchor_table_only():
    """A length-1 path (the anchor table itself) returns anchor RIDs that exist."""
    model = _make_model()
    fetched = {("S", "Parent"): [{"RID": "p1"}, {"RID": "p2"}, {"RID": "p3"}]}
    result = _reached_rids_for_path(
        (("S", "Parent"),), anchor_rids={"p1", "p2", "p99"}, fetched_rows=fetched, model=model
    )
    assert result == {"p1", "p2"}  # p99 not in the table


def test_reached_rids_no_fk_returns_none():
    """A path with no FK between adjacent segments is unfollowable -> None."""
    model = _make_model()
    fetched = {("S", "Parent"): [{"RID": "p1"}], ("S", "Parent2"): []}
    # Parent -> Parent (no self-FK in the synthetic model)
    result = _reached_rids_for_path(
        (("S", "Parent"), ("S", "Parent")), anchor_rids={"p1"}, fetched_rows=fetched, model=model
    )
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_reachability.py::test_reached_rids_single_hop_fk_on_cur -v`
Expected: FAIL — `ImportError: cannot import name '_reached_rids_for_path'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/deriva_ml/dataset/_reachability.py
def _reached_rids_for_path(
    fk_path: tuple[tuple[str, str], ...],
    *,
    anchor_rids: set[str],
    fetched_rows: dict[tuple[str, str], list[dict]],
    model: Any,
) -> set[str] | None:
    """RIDs of the terminal table reachable along one FK path.

    Walks the path hop-by-hop from the anchor RIDs, maintaining the set of
    in-scope RIDs of the current table. Each hop advances the scope using the
    FK constraint(s) between adjacent segments, hash-indexed for O(n) probes.

    Args:
        fk_path: Anchor-first tuple of ``(schema, table)`` segments.
        anchor_rids: RIDs of ``fk_path[0]`` that seed the walk.
        fetched_rows: ``{(schema, table): [row dict, ...]}`` — the once-fetched
            projected rows for every involved table.
        model: deriva-py Model (for FK resolution).

    Returns:
        Set of reachable terminal-table RIDs, or ``None`` if any hop has no
        resolvable FK (an unfollowable path — caller treats as no contribution).

    Example:
        >>> ...  # doctest: +SKIP  (needs fetched rows + Model)
    """
    cur_seg = fk_path[0]
    if len(fk_path) == 1:
        all_rids = {r["RID"] for r in fetched_rows.get(cur_seg, [])}
        return all_rids & anchor_rids

    cur_scope: set[str] = set(anchor_rids)
    prev_seg = fk_path[0]
    # Local FK-value index cache: (seg, col) -> {fk_value: {RID, ...}}.
    fk_index_cache: dict[tuple[tuple[str, str], str], dict] = {}

    def _fk_index(seg: tuple[str, str], col: str) -> dict:
        keyc = (seg, col)
        idx = fk_index_cache.get(keyc)
        if idx is None:
            idx = defaultdict(set)
            for r in fetched_rows.get(seg, []):
                v = r.get(col)
                if v is not None:
                    idx[v].add(r["RID"])
            fk_index_cache[keyc] = idx
        return idx

    for nxt_seg in fk_path[1:]:
        constraints = _fk_join_columns(prev_seg, nxt_seg, model)
        if not constraints:
            return None
        new_scope: set[str] = set()
        for con in constraints:
            pairs = con["pairs"]
            if con["fk_on"] == "cur":
                if len(pairs) == 1 and pairs[0][0] == "RID":
                    cur_col = pairs[0][1]
                    idx = _fk_index(nxt_seg, cur_col)
                    for pv in cur_scope:
                        hit = idx.get(pv)
                        if hit:
                            new_scope |= hit
                else:
                    prev_rows = fetched_rows.get(prev_seg, [])
                    allowed = {
                        tuple(r.get(pc) for pc, _cc in pairs)
                        for r in prev_rows
                        if r["RID"] in cur_scope
                    }
                    for r in fetched_rows.get(nxt_seg, []):
                        if tuple(r.get(cc) for _pc, cc in pairs) in allowed:
                            new_scope.add(r["RID"])
            else:  # fk_on == "prev"
                prev_rows = fetched_rows.get(prev_seg, [])
                if len(pairs) == 1 and pairs[0][1] == "RID":
                    prev_col = pairs[0][0]
                    for r in prev_rows:
                        if r["RID"] in cur_scope:
                            v = r.get(prev_col)
                            if v is not None:
                                new_scope.add(v)
                else:
                    prev_col, cur_col = pairs[0]
                    idx = _fk_index(nxt_seg, cur_col)
                    for r in prev_rows:
                        if r["RID"] in cur_scope:
                            v = r.get(prev_col)
                            if v is not None:
                                hit = idx.get(v)
                                if hit:
                                    new_scope |= hit
        cur_scope = new_scope
        prev_seg = nxt_seg
    return cur_scope
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_reachability.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/dataset/_reachability.py tests/dataset/test_reachability.py
git commit -m "feat(estimate): single-path BFS reachability walk"
```

---

### Task 4: compute_reachability — orchestration + multi-path union + asset bytes

**Files:**
- Modify: `src/deriva_ml/dataset/_reachability.py`
- Test: `tests/dataset/test_reachability.py`

The public entry point. Determines edge tables, fetches each once via the injected `fetch`, unions per-table RIDs across all FK paths, and sums asset `Length` over the reached RIDs. Mirrors prototype main body (lines 61–329) minus the timing/reference scaffolding.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/dataset/test_reachability.py
from deriva_ml.dataset._reachability import compute_reachability


def test_compute_reachability_unions_paths_and_sums_assets():
    """Two FK paths reach Child; union the RID sets. Child is an asset table;
    sum Length over reached RIDs only."""
    # Model: Parent <- Child (parent_fk), and an Alt parent <- Child (alt_fk).
    parent = SimpleNamespace(foreign_keys=[], column_definitions=SimpleNamespace(elements={}))
    alt = SimpleNamespace(foreign_keys=[], column_definitions=SimpleNamespace(elements={}))
    fk1 = SimpleNamespace(foreign_key_columns=[_col("parent_fk")], referenced_columns=[_col("RID")])
    fk2 = SimpleNamespace(foreign_key_columns=[_col("alt_fk")], referenced_columns=[_col("RID")])
    length_col = SimpleNamespace(name="Length")
    child = SimpleNamespace(
        foreign_keys=[fk1, fk2],
        referenced_by=[],
        column_definitions=SimpleNamespace(elements={"Length": length_col}),
    )
    fk1.pk_table = parent; fk2.pk_table = alt
    parent.referenced_by = [fk1]; alt.referenced_by = [fk2]
    parent.is_asset = lambda: False; alt.is_asset = lambda: False
    child.is_asset = lambda: True
    model = SimpleNamespace(schemas={"S": SimpleNamespace(
        tables={"Parent": parent, "Alt": alt, "Child": child})})

    rows = {
        ("S", "Parent"): [{"RID": "p1"}],
        ("S", "Alt"): [{"RID": "a1"}],
        ("S", "Child"): [
            {"RID": "c1", "parent_fk": "p1", "alt_fk": None, "Length": 100},
            {"RID": "c2", "parent_fk": None, "alt_fk": "a1", "Length": 200},
            {"RID": "c3", "parent_fk": None, "alt_fk": None, "Length": 999},  # unreachable
        ],
    }

    def fake_fetch(schema, table, columns):
        return rows[(schema, table)]

    reached = {
        ("S", "Child"): [
            (("S", "Parent"), ("S", "Child")),
            (("S", "Alt"), ("S", "Child")),
        ],
    }
    rids_by_table, asset_lengths_by_table = compute_reachability(
        reached=reached, anchor_rids=["p1", "a1"], model=model, fetch=fake_fetch,
    )
    # c1 reached via Parent, c2 via Alt; c3 via neither. Union = {c1, c2}.
    assert rids_by_table["Child"] == {"c1", "c2"}
    # Asset bytes only over reached RIDs: 100 + 200 = 300 (c3's 999 excluded).
    assert asset_lengths_by_table["Child"] == {"c1": 100, "c2": 200}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_reachability.py::test_compute_reachability_unions_paths_and_sums_assets -v`
Expected: FAIL — `ImportError: cannot import name 'compute_reachability'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/deriva_ml/dataset/_reachability.py
def compute_reachability(
    *,
    reached: ReachedPaths,
    anchor_rids: list[str],
    model: Any,
    fetch: FetchFn,
) -> tuple[dict[str, set[str]], dict[str, dict[str, int]]]:
    """Compute exact per-table reachable RID sets + asset byte maps.

    The client-side replacement for the deep-server-join estimate path.
    Fetches each involved edge/target table once (projected to the columns
    the in-memory join needs), then for every reached table unions the RID
    sets contributed by each FK path. For asset tables, builds ``{RID:
    Length}`` over the reached RIDs only.

    Args:
        reached: ``CatalogBagBuilder.iter_reached_paths()`` output —
            ``{(schema, table): [fk_path, ...]}``.
        anchor_rids: Dataset RID + every recursive descendant dataset RID.
        model: deriva-py Model (``cb._get_model()``).
        fetch: ``(schema, table, columns) -> rows`` — injected so the engine
            is testable. Production binds it to the path builder.

    Returns:
        ``(rids_by_table, asset_lengths_by_table)`` — shapes matching the
        first two outputs of the retired ``run_estimate_queries`` so
        :func:`assemble_estimate` consumes them unchanged.

    Example:
        >>> ...  # doctest: +SKIP  (needs a live walk + Model)
    """
    # 1. Every distinct (schema, table) that appears as a hop in any path.
    edge_tables: set[tuple[str, str]] = set()
    for fk_paths in reached.values():
        for fk_path in fk_paths:
            edge_tables.update(fk_path)

    # 2. Fetch each edge table ONCE, projected.
    fetched_rows: dict[tuple[str, str], list[dict]] = {}
    for seg in edge_tables:
        s, t = seg
        fetched_rows[seg] = fetch(s, t, _needed_columns(seg, model))

    anchor_set = set(anchor_rids)

    # 3. Union RIDs per reached table across its FK paths.
    rids_by_table: dict[str, set[str]] = {}
    client_rids: dict[tuple[str, str], set[str]] = {}
    for key, fk_paths in reached.items():
        union: set[str] = set()
        for fk_path in fk_paths:
            rr = _reached_rids_for_path(
                fk_path, anchor_rids=anchor_set, fetched_rows=fetched_rows, model=model
            )
            if rr:
                union |= rr
        client_rids[key] = union
        rids_by_table[key[1]] = union

    # 4. Asset Length sums over reached RIDs only.
    asset_lengths_by_table: dict[str, dict[str, int]] = {}
    for key, rids in client_rids.items():
        s, t = key
        tbl = model.schemas[s].tables[t]
        if not tbl.is_asset():
            continue
        by_rid = {r["RID"]: r for r in fetched_rows.get(key, [])}
        lengths: dict[str, int] = {}
        for rid in rids:
            r = by_rid.get(rid)
            if r and r.get("Length") is not None:
                lengths[rid] = int(r["Length"])
        asset_lengths_by_table[t] = lengths

    return rids_by_table, asset_lengths_by_table


__all__ = ["ReachedPaths", "FetchFn", "compute_reachability"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_reachability.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/dataset/_reachability.py tests/dataset/test_reachability.py
git commit -m "feat(estimate): compute_reachability orchestration + multi-path union"
```

---

### Task 5: CSV-byte sampling from fetched rows

**Files:**
- Modify: `src/deriva_ml/dataset/_reachability.py`
- Test: `tests/dataset/test_reachability.py`

The old path took a 100-row `sample` query per table. The engine already has every table's rows in memory, so derive the sample directly — no extra query. Return a `sample_rows_by_table` so `assemble_estimate` keeps its CSV-byte estimation unchanged.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/dataset/test_reachability.py
from deriva_ml.dataset._reachability import sample_rows_from_fetched


def test_sample_rows_from_fetched_caps_at_100():
    fetched = {("S", "T"): [{"RID": f"r{i}", "x": i} for i in range(250)]}
    reached = {("S", "T"): [(("S", "T"),)]}
    samples = sample_rows_from_fetched(reached=reached, fetched_rows=fetched, limit=100)
    assert len(samples["T"]) == 100
    assert samples["T"][0] == {"RID": "r0", "x": 0}


def test_sample_rows_from_fetched_empty_table_absent():
    fetched = {("S", "T"): []}
    reached = {("S", "T"): [(("S", "T"),)]}
    samples = sample_rows_from_fetched(reached=reached, fetched_rows=fetched, limit=100)
    assert "T" not in samples
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_reachability.py::test_sample_rows_from_fetched_caps_at_100 -v`
Expected: FAIL — `ImportError: cannot import name 'sample_rows_from_fetched'`

- [ ] **Step 3: Write minimal implementation**

Note: `compute_reachability` must expose the fetched rows for this. Refactor so the fetch map is reusable — change `compute_reachability` to also return nothing new (keep its 2-tuple contract) but extract sampling as a sibling that re-reads from a shared fetch. Simplest: have the production caller hold the fetched rows. To keep the engine's public surface clean, `sample_rows_from_fetched` takes the same `reached` + a `fetched_rows` map. The caller (Task 6) obtains `fetched_rows` by having `compute_reachability` optionally return it.

Update `compute_reachability` to return a 3-tuple `(rids_by_table, asset_lengths_by_table, fetched_rows)` and adjust Task 4's test accordingly (the test unpacks 3 values, ignoring the third). Then:

```python
# add to src/deriva_ml/dataset/_reachability.py
def sample_rows_from_fetched(
    *,
    reached: ReachedPaths,
    fetched_rows: dict[tuple[str, str], list[dict]],
    limit: int = 100,
) -> dict[str, list[dict]]:
    """Take up to ``limit`` sample rows per reached table from fetched rows.

    The CSV-byte estimator needs a few representative rows per table to gauge
    serialised size. Because the engine already fetched every table's rows,
    the sample is a slice — no extra query (the old path issued one
    ``?limit=100`` query per table).

    Args:
        reached: The reached-paths map (keys identify the tables to sample).
        fetched_rows: ``{(schema, table): rows}`` from the engine fetch.
        limit: Max sample rows per table.

    Returns:
        ``{table_name: [row, ...]}`` — tables with zero fetched rows are omitted.

    Example:
        >>> rows = {("S", "T"): [{"RID": "r0"}]}
        >>> sample_rows_from_fetched(reached={("S", "T"): [(("S", "T"),)]},
        ...                          fetched_rows=rows, limit=10)
        {'T': [{'RID': 'r0'}]}
    """
    samples: dict[str, list[dict]] = {}
    for key in reached:
        rows = fetched_rows.get(key, [])
        if rows:
            samples[key[1]] = rows[:limit]
    return samples
```

Also update `compute_reachability`'s return statement and `__all__`:

```python
    return rids_by_table, asset_lengths_by_table, fetched_rows
```

```python
__all__ = ["ReachedPaths", "FetchFn", "compute_reachability", "sample_rows_from_fetched"]
```

And update the Task 4 test's unpacking line to:

```python
    rids_by_table, asset_lengths_by_table, _fetched = compute_reachability(
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_reachability.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/dataset/_reachability.py tests/dataset/test_reachability.py
git commit -m "feat(estimate): derive CSV-byte samples from fetched rows"
```

---

### Task 6: Wire estimate_bag_size to the engine; adapt assemble_estimate

**Files:**
- Modify: `src/deriva_ml/dataset/dataset.py` (~2655–2790)
- Modify: `src/deriva_ml/dataset/_estimate.py` (`assemble_estimate` signature)
- Test: existing live estimate integration test (see Task 8)

Replace the query-build + async-orchestration block with: build the walk, get `reached` + `anchor_rids` + `model`, build a `fetch` closure over the snapshot path builder, call `compute_reachability` + `sample_rows_from_fetched`, then `assemble_estimate`. `assemble_estimate` loses its `failed_by_table` param (the engine has no per-query failures).

- [ ] **Step 1: Write the failing test**

A unit test that `estimate_bag_size` no longer imports the deleted functions. (Behavioral correctness is the live test in Task 8; this guards the wiring shape.)

```python
# add to tests/dataset/test_reachability.py
def test_estimate_no_longer_imports_deep_join_helpers():
    """The deep-join path is deleted; estimate must not import it."""
    import inspect
    from deriva_ml.dataset import dataset as dataset_mod
    src = inspect.getsource(dataset_mod.Dataset.estimate_bag_size)
    assert "build_estimate_queries" not in src
    assert "run_estimate_queries" not in src
    assert "compute_reachability" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_reachability.py::test_estimate_no_longer_imports_deep_join_helpers -v`
Expected: FAIL — current source still references `build_estimate_queries`

- [ ] **Step 3: Write minimal implementation**

Replace the body of `estimate_bag_size` from the import block through the `assemble_estimate` call with:

```python
        from deriva_ml.dataset._estimate import assemble_estimate
        from deriva_ml.dataset._reachability import (
            compute_reachability,
            sample_rows_from_fetched,
        )

        if isinstance(version, str):
            version = DatasetVersion.parse(version)

        # Build a DatasetBagBuilder on the version snapshot. The walk (which
        # tables are reachable, via which FK paths) is shared with the bag
        # export; only the *execution* differs — see docs/adr/0008.
        version_snapshot_catalog = self._version_snapshot_catalog(version)
        builder = DatasetBagBuilder(
            ml_instance=version_snapshot_catalog,
            exclude_tables=exclude_tables,
        )
        cb = builder._catalog_bag_builder(dataset=self)
        reached = cb.iter_reached_paths()
        anchor_rids = [self.dataset_rid] + list(builder._iter_descendant_rids(self))
        model = cb._get_model()

        # Fetch closure: whole-table projected scan via the snapshot path
        # builder. Cheap relative to deep FK joins (a 240k-row scan is ~0.3s;
        # the join through it was 16-60s). Bound to the version snapshot.
        pb = version_snapshot_catalog.pathBuilder()

        def _fetch(schema: str, table: str, columns: set[str]) -> list[dict]:
            tpb = pb.schemas[schema].tables[table]
            try:
                attrs = [getattr(tpb, c) for c in sorted(columns)]
                return list(tpb.attributes(*attrs).fetch())
            except Exception:  # noqa: BLE001
                # Defensive fallback: a projection naming a column the table
                # lacks (model/data skew) degrades to a full-entity fetch
                # rather than dropping the table from the estimate.
                return list(tpb.entities().fetch())

        rids_by_table, asset_lengths_by_table, fetched_rows = compute_reachability(
            reached=reached, anchor_rids=anchor_rids, model=model, fetch=_fetch
        )
        sample_rows_by_table = sample_rows_from_fetched(
            reached=reached, fetched_rows=fetched_rows
        )

        return assemble_estimate(
            table_queries={
                key[1]: [(None, None, model.schemas[key[0]].tables[key[1]].is_asset())]
                for key in reached
            },
            rids_by_table=rids_by_table,
            asset_lengths_by_table=asset_lengths_by_table,
            sample_rows_by_table=sample_rows_by_table,
            estimate_csv_bytes=self._estimate_csv_bytes,
            human_readable_size=self._human_readable_size,
        )
```

Then in `_estimate.py`, change `assemble_estimate`'s signature to drop `failed_by_table` (remove the param, the `failed_by_table = failed_by_table or {}` line, and all `incomplete` logic that referenced it — set every table's `incomplete` to `False` and top-level `incomplete` to `False`, `incomplete_tables` to `[]`). The `table_queries` param keeps the same shape `{name: [(_, _, is_asset)]}`; the engine supplies `(None, None, is_asset)` triples since assembly only reads `is_asset`.

> **Note on `table_queries` shape:** `assemble_estimate` only uses
> `table_queries` to compute `asset_tables` via
> `any(is_asset for _, _, is_asset in entries)`. The `None, None`
> placeholders for datapath/target_table are never read. This is verified
> by reading `assemble_estimate` — its only `table_queries` access is the
> asset-detection comprehension at the top.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_reachability.py::test_estimate_no_longer_imports_deep_join_helpers tests/dataset/test_estimate_helpers.py -v`
Expected: PASS (the import-shape test passes; `assemble_estimate` tests pass after their `failed_by_table` args are dropped — fix those in Task 7)

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/dataset/dataset.py src/deriva_ml/dataset/_estimate.py tests/dataset/test_reachability.py
git commit -m "feat(estimate): wire estimate_bag_size to reachability engine"
```

---

### Task 7: Delete the deep-join path; fix dependent tests

**Files:**
- Modify: `src/deriva_ml/dataset/_estimate.py`
- Modify: `tests/dataset/test_estimate_helpers.py`

Remove the now-dead deep-join functions per CLAUDE.md's "no backwards-compat shims". Keep `assemble_estimate`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/dataset/test_reachability.py
def test_deep_join_helpers_are_deleted():
    """The retired deep-join estimate functions must be gone."""
    import deriva_ml.dataset._estimate as est
    assert not hasattr(est, "build_estimate_queries")
    assert not hasattr(est, "run_estimate_queries")
    assert not hasattr(est, "QueryItem")
    assert hasattr(est, "assemble_estimate")  # kept
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_reachability.py::test_deep_join_helpers_are_deleted -v`
Expected: FAIL — `build_estimate_queries` still present

- [ ] **Step 3: Delete the dead code**

From `src/deriva_ml/dataset/_estimate.py` remove: `QueryType`, `QueryItem`, `_extract_path`, `build_estimate_queries`, `DEFAULT_ESTIMATE_CONCURRENCY`, `run_estimate_queries`, and the `TYPE_CHECKING` import of `AsyncErmrestCatalog`/`AsyncErmrestSnapshot` if now unused. Trim `__all__` to `["assemble_estimate"]`. Update the module docstring to describe only `assemble_estimate`.

In `tests/dataset/test_estimate_helpers.py` delete every test exercising the removed functions; keep and fix the `assemble_estimate` tests (drop any `failed_by_table=` kwargs from their calls, and drop assertions about `incomplete`/`incomplete_tables` being populated by failures — they are now always `False`/`[]`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_reachability.py tests/dataset/test_estimate_helpers.py -v`
Expected: PASS (all reachability + remaining assemble_estimate tests)

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/dataset/_estimate.py tests/dataset/test_estimate_helpers.py tests/dataset/test_reachability.py
git commit -m "refactor(estimate): delete retired deep-join query path"
```

---

### Task 8: Live correctness test — engine == server-union on demo catalog

**Files:**
- Modify: the live estimate integration test file (find via `grep -rln "estimate_bag_size" tests/`)

Prove the engine produces the exact same per-table counts as the slow server-union path, on the demo catalog (needs `DERIVA_HOST`). This is the regression guard that makes the "exact" claim enforceable.

- [ ] **Step 1: Write the failing test**

```python
# in the live estimate test file
import os
import pytest


@pytest.mark.skipif(
    os.environ.get("DERIVA_HOST") in (None, ""),
    reason="needs a live catalog",
)
def test_reachability_matches_server_union(catalog_with_datasets):
    """The client-side engine's per-table counts must equal the server-side
    RID-union for every reached table (exactness guard)."""
    ml = catalog_with_datasets
    # Pick a nested dataset from the fixture (has multi-path tables).
    datasets = ml.find_datasets()
    nested = next(d for d in datasets if ml.lookup_dataset(d["RID"]).list_dataset_children())
    ds = ml.lookup_dataset(nested["RID"])
    version = ds.dataset_version  # current version

    # Client-side result.
    est = ds.estimate_bag_size(version)
    client_counts = {t: d["row_count"] for t, d in est["tables"].items()}

    # Server-union reference via aggregate_queries (the retired engine's math,
    # computed inline here purely as an oracle).
    from deriva_ml.dataset.bag_builder import DatasetBagBuilder
    snap = ds._version_snapshot_catalog(version)
    builder = DatasetBagBuilder(ml_instance=snap)
    tq = builder.aggregate_queries(ds)
    for tname, entries in tq.items():
        union = set()
        for dp, pb_tbl, _is_asset in entries:
            for row in dp.attributes(pb_tbl.RID).fetch():
                union.add(row["RID"])
        assert client_counts.get(tname, 0) == len(union), (
            f"{tname}: client={client_counts.get(tname)} server={len(union)}"
        )
```

- [ ] **Step 2: Run test to verify it fails (or errors without catalog)**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest <test_file>::test_reachability_matches_server_union -v --timeout=600`
Expected: PASS if the engine is correct (this test validates Tasks 1–7 end-to-end). If it FAILS, the mismatch names the offending table — debug the FK resolution for that table before proceeding.

- [ ] **Step 3: (No new impl — this test validates prior tasks)**

If the test fails, the fix is in `_reachability.py` (likely a missing inbound-referenced column or an unhandled composite FK). Do not weaken the test.

- [ ] **Step 4: Confirm pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest <test_file>::test_reachability_matches_server_union -v --timeout=600`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add <test_file>
git commit -m "test(estimate): pin engine == server-union exactness on demo catalog"
```

---

### Task 9: Update docs + ADR + lint + suites + PR

**Files:**
- Modify: `docs/reference/bag-export.md` (~277–285)
- Modify: `docs/adr/0008-estimate-bag-size-bypasses-bag-pipeline.md`

- [ ] **Step 1: Update bag-export.md "Same engine, different consumer"**

Replace the `estimate_bag_size` bullet (currently "runs live datapath aggregate queries against the reached tables and computes exact RID-union counts client-side") with text describing the client-side reachability engine: it shares the *walk* (`iter_reached_paths` + descendant anchors) but, instead of issuing per-FK-path aggregate queries, fetches each reached table's edge columns once and reconstructs reachability in memory — exact union counts, no deep server joins. Reference `dataset/_reachability.py::compute_reachability`.

- [ ] **Step 2: Update ADR-0008**

Add a dated note: the bypass mechanism changed from "per-FK-path live aggregate queries (async, concurrency-bounded)" to "client-side FK-reachability over once-fetched edge tables". The *decision* (estimate bypasses the bag export engine) is unchanged; the *implementation* of the bypass is now the reachability engine. Cross-link `docs/superpowers/specs/2026-06-14-portable-bag-csv-contract.md`.

- [ ] **Step 3: Lint + format**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check --fix src/deriva_ml/dataset/_reachability.py src/deriva_ml/dataset/_estimate.py src/deriva_ml/dataset/dataset.py tests/dataset/test_reachability.py && uv run ruff format src/deriva_ml/dataset/_reachability.py tests/dataset/test_reachability.py
```
Expected: clean

- [ ] **Step 4: Run the dataset unit + doctest suite**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_reachability.py tests/dataset/test_estimate_helpers.py --doctest-modules src/deriva_ml/dataset/_reachability.py -q
```
Expected: all pass

- [ ] **Step 5: Live re-measure on eye-ai 2-277G + branch, push, PR**

Run a timed estimate against eye-ai 2-277G (needs a fresh www.eye-ai.org token) to confirm the wall-clock win and the totals (80 tables / 360756 rows / ~18 GB asset). Capture the number for the PR body.

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git checkout -b feature/reachability-engine-fast-estimate
# (commits from Tasks 1-9 already on this branch if started here; else cherry-pick)
git push -u origin feature/reachability-engine-fast-estimate
gh pr create --title "feat(estimate): client-side FK-reachability engine (10x faster, exact)" \
  --body "Replaces ~377s of server-side deep-FK-join estimate queries with a client-side reachability engine: fetch each reached table's edge columns once, reconstruct FK reachability in memory, exact RID-union counts. eye-ai 2-277G: <N>s vs ~280s, identical totals (80 tables / 360756 rows / 18 GB). Stage A of the fast-estimate + portable-bag effort; Stage B (deriva-py csv-ridset processor + bag-gen rewire) reuses this engine. See docs/superpowers/specs/2026-06-14-portable-bag-csv-contract.md.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Self-Review

**1. Spec coverage** (against the connected investigation + memory `estimate-bag-size-client-side-join`):
- Client-side reachability replaces deep joins → Tasks 1–6. ✓
- Vocab-FK `Name` subtlety (inbound-referenced columns) → Task 2 (explicit test + impl). ✓
- Multi-path RID-union for exact counts → Task 4. ✓
- Asset `Length` sum over reached RIDs → Task 4. ✓
- CSV-byte sampling preserved → Task 5. ✓
- Exactness guard vs server-union → Task 8. ✓
- Delete deep-join path (no shim, per CLAUDE.md) → Task 7. ✓
- Engine reusable by Stage B → `compute_reachability` is the public, injectable entry point. ✓
- Docs/ADR updated → Task 9. ✓

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". Each code step shows complete code. Task 8 Step 3 intentionally has no new impl (it validates prior tasks) and says so explicitly. ✓

**3. Type consistency:**
- `compute_reachability` returns a 3-tuple `(rids_by_table, asset_lengths_by_table, fetched_rows)` — set in Task 4, amended in Task 5, consumed in Task 6. The Task 4 test unpacks 3 (amended in Task 5 Step 3). ✓
- `_reached_rids_for_path` signature (keyword-only `anchor_rids`/`fetched_rows`/`model`) consistent across Tasks 3, 4. ✓
- `_fk_join_columns` / `_needed_columns` signatures consistent across Tasks 1, 2, 3, 4. ✓
- `assemble_estimate` loses `failed_by_table` (Task 6) and its tests are fixed (Task 7) — consistent. ✓
- `FetchFn = (schema, table, columns) -> rows` consistent in the interface block, Task 4, Task 6. ✓

**Caveat flagged for the implementer:** Task 5 changes `compute_reachability`'s return arity (2-tuple → 3-tuple). The implementer must update the Task 4 test's unpacking line when doing Task 5 (called out in Task 5 Step 3). If executing strictly task-by-task, expect the Task 4 test to need a one-line edit during Task 5 — this is intentional, not a defect.
