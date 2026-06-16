# Parallel edge-table fetch in compute_reachability — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize the sequential edge-table fetch loop in `compute_reachability` (opt-in, bounded), speeding up the estimate AND bag generation (spec + build) which share `_compute_rid_sets`.

**Architecture:** Add `max_workers: int = 1` to `compute_reachability`; when >1, run the fetch phase through a bounded `ThreadPoolExecutor`. Default 1 = current sequential behavior, byte-identical output. Thread a distinct `reachability_concurrency` knob through `_compute_rid_sets` (the shared chokepoint reaching all three callers), then surface it on the public entry points alongside — not conflated with — the existing materialization `fetch_concurrency`.

**Tech Stack:** Python 3.12, `uv`, pytest, `concurrent.futures.ThreadPoolExecutor`.

**Spec:** `docs/superpowers/specs/2026-06-16-parallel-edge-table-fetch-design.md`

---

### Task 1: Engine — `max_workers` on `compute_reachability` (opt-in parallel fetch)

**Files:**
- Modify: `src/deriva_ml/dataset/_reachability.py` (`compute_reachability`, the fetch loop ~line 250)
- Test: `tests/dataset/test_reachability.py` (add cases; create if absent)

- [ ] **Step 1: Write failing tests (equivalence + concurrency + error propagation)**

```python
# tests/dataset/test_reachability_concurrency.py
"""compute_reachability's opt-in parallel fetch is exact, actually concurrent,
and propagates fetch errors. Catalog-free: the fetch fn is injected."""
from __future__ import annotations

import threading

import pytest

from deriva_ml.dataset._reachability import compute_reachability


class _Model:
    """Minimal model stub: every table non-asset, no inbound FK columns."""

    class _T:
        def is_asset(self):
            return False

        column_definitions = type("C", (), {"elements": []})()
        referenced_by = []

    class _S:
        tables = {}

        def __getitem__(self, name):
            return _Model._T()

    schemas = {}

    def __init__(self, segs):
        # every (schema, table) resolves to a non-asset table
        schemas = {}
        for s, t in segs:
            schemas.setdefault(s, _Model._S())
            schemas[s].tables[t] = _Model._T()
        self.schemas = _DictSchemas(schemas)


class _DictSchemas(dict):
    def __getitem__(self, k):
        return super().__getitem__(k)


def _reached(segs):
    # one single-hop path per table: reached[(s,t)] = [ (s,t), ]
    return {seg: [(seg,)] for seg in segs}


def _make_fetch(seen_threads=None):
    def fetch(s, t, cols):
        if seen_threads is not None:
            seen_threads.add(threading.get_ident())
        # deterministic rows keyed by table; RID equals "<t>-<i>"
        return [{"RID": f"{t}-{i}"} for i in range(3)]

    return fetch


SEGS = [("demo", f"T{i}") for i in range(6)]


def test_parallel_matches_sequential_exactly():
    model = _Model(SEGS)
    reached = _reached(SEGS)
    anchors = []

    seq = compute_reachability(
        reached=reached, anchor_rids=anchors, model=model, fetch=_make_fetch(), max_workers=1
    )
    par = compute_reachability(
        reached=reached, anchor_rids=anchors, model=model, fetch=_make_fetch(), max_workers=8
    )
    assert seq[0] == par[0]  # rids_by_table
    assert seq[1] == par[1]  # asset_lengths_by_table
    assert seq[2] == par[2]  # fetched_rows


def test_parallel_actually_uses_threads():
    model = _Model(SEGS)
    threads_par: set[int] = set()
    compute_reachability(
        reached=_reached(SEGS), anchor_rids=[], model=model,
        fetch=_make_fetch(threads_par), max_workers=4,
    )
    assert len(threads_par) > 1, "parallel path did not use multiple threads"

    threads_seq: set[int] = set()
    compute_reachability(
        reached=_reached(SEGS), anchor_rids=[], model=model,
        fetch=_make_fetch(threads_seq), max_workers=1,
    )
    assert len(threads_seq) == 1, "sequential path used more than one thread"


def test_parallel_propagates_fetch_error():
    model = _Model(SEGS)

    def boom(s, t, cols):
        if t == "T3":
            raise RuntimeError("fetch failed")
        return [{"RID": f"{t}-0"}]

    with pytest.raises(RuntimeError, match="fetch failed"):
        compute_reachability(
            reached=_reached(SEGS), anchor_rids=[], model=model, fetch=boom, max_workers=8
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_reachability_concurrency.py -q`
Expected: FAIL — `compute_reachability()` got an unexpected keyword argument `max_workers`.

- [ ] **Step 3: Implement `max_workers`**

In `compute_reachability` add `max_workers: int = 1` to the signature, and replace the fetch loop (currently the `for seg in edge_tables: fetched_rows[seg] = fetch(...)` block):

```python
def compute_reachability(
    *,
    reached: ReachedPaths,
    anchor_rids: list[str],
    model: Any,
    fetch: FetchFn,
    max_workers: int = 1,
) -> tuple[...]:
    ...
    # 2. Fetch each edge table ONCE, projected. Opt-in bounded parallelism:
    # the fetches are independent reads writing distinct keys, so order does
    # not matter; max_workers=1 keeps the exact sequential behavior.
    fetched_rows: dict[tuple[str, str], list[dict]] = {}
    if max_workers > 1 and len(edge_tables) > 1:
        from concurrent.futures import ThreadPoolExecutor

        workers = min(max_workers, len(edge_tables))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(fetch, s, t, _needed_columns((s, t), model)): (s, t)
                for (s, t) in edge_tables
            }
            for fut, seg in futures.items():
                fetched_rows[seg] = fut.result()  # re-raises fetch errors
    else:
        for seg in edge_tables:
            s, t = seg
            fetched_rows[seg] = fetch(s, t, _needed_columns(seg, model))
```

Update the docstring `Args:` to document `max_workers` (opt-in bounded parallelism; default 1 = sequential, exact).

- [ ] **Step 4: Run tests**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_reachability_concurrency.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/dataset/_reachability.py tests/dataset/test_reachability_concurrency.py
git commit -m "perf(reachability): opt-in bounded-parallel edge-table fetch in compute_reachability"
```

---

### Task 2: Thread `reachability_concurrency` through `_compute_rid_sets`

`_compute_rid_sets` is the shared chokepoint for the estimate + the two bag-gen paths. Add the knob here so all three benefit.

**Files:**
- Modify: `src/deriva_ml/dataset/bag_builder.py` (`_compute_rid_sets` ~line 604, and its three internal callers within bag_builder.py if they should pass it)
- Test: `tests/dataset/test_format_b_bag.py` (extend the existing source-inspection guard)

- [ ] **Step 1: Write failing test**

```python
def test_compute_rid_sets_threads_reachability_concurrency():
    """_compute_rid_sets accepts reachability_concurrency and forwards it to
    compute_reachability as max_workers."""
    import inspect

    from deriva_ml.dataset.bag_builder import DatasetBagBuilder

    sig = inspect.signature(DatasetBagBuilder._compute_rid_sets)
    assert "reachability_concurrency" in sig.parameters
    assert sig.parameters["reachability_concurrency"].default == 1

    src = inspect.getsource(DatasetBagBuilder._compute_rid_sets)
    assert "max_workers=reachability_concurrency" in src
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_format_b_bag.py::test_compute_rid_sets_threads_reachability_concurrency -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

Add `reachability_concurrency: int = 1` to `_compute_rid_sets`'s signature and pass `max_workers=reachability_concurrency` into the `compute_reachability(...)` call (~line 655). Document the param.

- [ ] **Step 4: Run test + the broader format-b suite**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_format_b_bag.py -q`
Expected: PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/dataset/bag_builder.py tests/dataset/test_format_b_bag.py
git commit -m "perf(bag): thread reachability_concurrency through _compute_rid_sets (estimate + spec + build)"
```

---

### Task 3: Surface `reachability_concurrency` on the public entry points

Thread the knob to the user-facing methods. Keep it distinct from the materialization `fetch_concurrency`.

**Files:**
- Modify: `src/deriva_ml/dataset/dataset.py` — `estimate_bag_size`, `bag_info`, `download_dataset_bag`, `cache` (add `reachability_concurrency: int = 1`, pass to `_compute_rid_sets` / through to the bag-gen path)
- Modify: `src/deriva_ml/dataset/bag_builder.py` — `build_bag`, `generate_dataset_download_spec` (accept + forward to `_compute_rid_sets`)
- Modify: `src/deriva_ml/core/mixins/dataset.py` — the `DerivaML.estimate_bag_size` / `bag_info` / `download_dataset_bag` wrappers, if they re-declare the args
- Test: `tests/dataset/test_format_b_bag.py` + a signature test

- [ ] **Step 1: Write failing signature tests**

```python
def test_public_entry_points_expose_reachability_concurrency():
    import inspect

    from deriva_ml.dataset.dataset import Dataset

    for name in ("estimate_bag_size", "bag_info"):
        sig = inspect.signature(getattr(Dataset, name))
        assert "reachability_concurrency" in sig.parameters, name
        assert sig.parameters["reachability_concurrency"].default == 1, name
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_format_b_bag.py::test_public_entry_points_expose_reachability_concurrency -q`
Expected: FAIL.

- [ ] **Step 3: Implement the wiring**

- `Dataset.estimate_bag_size(version, exclude_tables=None, reachability_concurrency=1)` → `builder._compute_rid_sets(self, reachability_concurrency=reachability_concurrency)`.
- `Dataset.bag_info(...)` → pass through to `estimate_bag_size`.
- `Dataset.build_bag` / `generate_dataset_download_spec` → accept `reachability_concurrency`, forward to `_compute_rid_sets`.
- `Dataset.download_dataset_bag` / `cache` → accept `reachability_concurrency` **alongside** the existing `fetch_concurrency`; forward the former to the bag-gen `_compute_rid_sets` and leave the latter on the asset-download path. Update docstrings to make the distinction explicit (reachability fetch vs asset file download).
- Mirror on the `DerivaML` mixin wrappers in `core/mixins/dataset.py` where they re-declare args (and on `DatasetSpec` if it carries `fetch_concurrency` — check and add `reachability_concurrency` there too for parity).

- [ ] **Step 4: Run signature + full format-b + download suites**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_format_b_bag.py tests/dataset/test_download.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add -A
git commit -m "perf(bag): surface reachability_concurrency on estimate/bag_info/download/cache (distinct from fetch_concurrency)"
```

---

### Task 4: Exactness gate + lint + (optional) live re-measure + PR

**Files:**
- Test: existing live exactness test (`tests/dataset/test_estimate_bag_size.py` differential test), re-run with `reachability_concurrency>1`

- [ ] **Step 1: Add a live exactness assertion under concurrency**

Add (or parametrize) a live test that runs the estimate with `reachability_concurrency=8` and asserts it equals the `reachability_concurrency=1` result on the demo catalog (per-table counts + asset bytes identical):

```python
@pytest.mark.skipif(os.environ.get("DERIVA_HOST") in (None, ""), reason="needs a live catalog")
def test_estimate_parallel_matches_sequential_live(catalog_with_datasets):
    ml, _ = catalog_with_datasets
    datasets = list(ml.find_datasets())
    nested = next((d for d in datasets if d.list_dataset_children()), None)
    if nested is None:
        pytest.skip("need a nested dataset")
    v = nested.current_version
    seq = nested.estimate_bag_size(v, reachability_concurrency=1)
    par = nested.estimate_bag_size(v, reachability_concurrency=8)
    # Compare the stable, order-insensitive fields.
    assert seq["tables"] == par["tables"]
    assert seq["total_asset_bytes"] == par["total_asset_bytes"]
    assert seq["total_rows"] == par["total_rows"]
```

- [ ] **Step 2: Run it + the estimate exactness suite**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_estimate_bag_size.py tests/dataset/test_format_b_bag.py -q`
Expected: PASS.

- [ ] **Step 3: Lint + format**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check --fix src/deriva_ml/dataset/_reachability.py src/deriva_ml/dataset/bag_builder.py src/deriva_ml/dataset/dataset.py src/deriva_ml/core/mixins/dataset.py tests/dataset/test_reachability_concurrency.py tests/dataset/test_format_b_bag.py && uv run ruff format <same files>`
Expected: clean. (Scope format args to the touched files only — do NOT `ruff format tests/dataset/` wholesale; it reformats unrelated files.)

- [ ] **Step 4: (Optional) live 2-277G re-measure**

If an eye-ai bearer token is available: measure `estimate_bag_size` on 2-277G with `reachability_concurrency=1` vs `8`, exactness preserved, record the wall-clock delta in the PR. If no token, state in the PR that the at-scale speedup is unverified and the change is behavior-preserving by default (sequential default).

- [ ] **Step 5: Branch, commit, PR**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add -A
git commit -m "test(reachability): live exactness under parallel fetch + plan/spec"
git push -u origin perf/parallel-reachability-fetch
gh pr create --title "perf(bag): opt-in parallel edge-table fetch for estimate + bag generation" --body "<summary + spec link + the three-caller scope + default-sequential + exactness gate + measurement caveat>"
```

---

## Self-Review notes

- **Spec coverage:** Task 1 = engine knob; Task 2 = `_compute_rid_sets` chokepoint (reaches all 3 callers); Task 3 = public surface (distinct from `fetch_concurrency`); Task 4 = exactness gate + PR. All spec sections covered.
- **Default-preserving:** every new param defaults to 1 → byte-identical current behavior; the parallel path is opt-in. The Task-1 equivalence test + Task-4 live exactness test pin "parallel == sequential."
- **Naming:** `reachability_concurrency` everywhere public; `max_workers` only inside `compute_reachability`. Never reuse `fetch_concurrency` (materialization file downloads).
- **Scope discipline:** lint/format only the touched files (the prior PR's wholesale `ruff format tests/dataset/` swept 17 unrelated files — do not repeat).
