# Format-B Bag Generation (deriva-ml, Plan B2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire deriva-ml's bag generation to the client-side reachability engine so a dataset bag is built from per-table RID sets (Format B — one clean CSV per table) instead of deep-FK-join per-path CSVs, dropping bag-gen wall-clock from ~280s toward tens of seconds for eye-ai 2-277G.

**Architecture:** Factor the estimate's reachability assembly (build walk → `iter_reached_paths` → anchors → model → `from_model` fetch closure → `compute_reachability`) into a shared `DatasetBagBuilder._compute_rid_sets(dataset)` helper, reused by both `estimate_bag_size` and the bag-build path. `build_bag` calls `_compute_rid_sets`, passes the resulting `{(schema,table): [RID,...]}` map to its `_SnapshotAwareCatalogBagBuilder(rid_sets=...)` (deriva-py B1's new param), and the upstream engine emits one rid-set csv processor per non-vocab table. `_catalog_bag_builder` gains an opt-in `rid_sets=None` param for the spec-generation path; the estimate/annotation/aggregate_queries callers pass nothing and are unchanged.

**Tech Stack:** Python ≥3.12, deriva-ml (`src/deriva_ml/dataset/bag_builder.py`, `dataset.py`, `bag_download.py`), deriva-py (the merged B1 `CatalogBagBuilder(rid_sets=)` + `get_as_file(rid_set=)`), pytest. The deriva-py dependency is pinned to the `deriva-ml` branch, which now contains B1.

---

## Scope: this is Plan B2 of two (deriva-ml side)

Stage B1 (deriva-py — RID-set fetch + `CatalogBagBuilder(rid_sets=)` emission) is **merged** (deriva-py PR #269 → `deriva-ml` branch). Stage A (the `compute_reachability` engine) is **merged** (deriva-ml PR #300). This plan is the consumer wiring: deriva-ml computes the RID sets and hands them to the merged upstream engine.

Design: `docs/superpowers/specs/2026-06-14-stage-b-fast-portable-bag-design.md` (decisions D1–D4) + `2026-06-14-portable-bag-csv-contract.md` (the CSV contract). The reachability engine: `src/deriva_ml/dataset/_reachability.py::compute_reachability` (returns `(rids_by_table, asset_lengths_by_table, fetched_rows)`). Memory: `csv-ridset-chunk-append-proto.md`, `snapshot-held-model-staleness-hazard.md`.

**The load-bearing reuse:** `estimate_bag_size` already assembles the exact reachability inputs (`dataset.py` ~2722–2746), including the `datapath.from_model(snapshot.catalog, model)` fix for the [[snapshot-held-model-staleness-hazard]]. B2 factors that into one helper so estimate and bag-gen share it — and the bag path inherits the from_model fix for free (it would hit the same KeyError otherwise).

**Key shape difference from the estimate:** `compute_reachability` returns `rids_by_table` keyed by **bare table name** (`{table: set}`). The upstream `CatalogBagBuilder(rid_sets=...)` is keyed by **`(schema, table)` tuple** (`{(schema,table): [RID]}`). The helper must return the tuple-keyed form (it has the schema from `reached`'s keys). See Task 1.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/deriva_ml/dataset/bag_builder.py` | New `_compute_rid_sets(dataset)` helper (factored reachability assembly, tuple-keyed). `_catalog_bag_builder` gains opt-in `rid_sets=None`. `build_bag` computes rid_sets + passes to `_SnapshotAwareCatalogBagBuilder(rid_sets=...)`. | **Modify** (~243 build_bag, ~449 _catalog_bag_builder, new helper) |
| `src/deriva_ml/dataset/dataset.py` | `estimate_bag_size` calls the shared `_compute_rid_sets` instead of inlining the assembly (DRY; keeps the from_model fix in one place). | **Modify** (~2722–2746) |
| `tests/dataset/test_format_b_bag.py` | Unit: `_compute_rid_sets` returns tuple-keyed RID sets matching `compute_reachability`; the spec carries rid-set processors. Live (DERIVA_HOST): a Format-B bag has one CSV per table, RID-distinct, with correct asset fetch.txt, and round-trips through the loader with counts matching the estimate. | **Create** |
| `docs/reference/bag-export.md` | Update "Same engine, different consumer" + B4: dataset bags now use Format B (one rid-set csv processor per table) when built client-side. | **Modify** |
| `docs/superpowers/specs/2026-06-14-stage-b-fast-portable-bag-design.md` | Mark B2 shipped; record the live bag-gen number. | **Modify** |

No new module — the reachability engine and the upstream emission already exist; B2 is wiring + one shared helper.

---

## Shared helper interface (locked here)

```python
# src/deriva_ml/dataset/bag_builder.py  (method on DatasetBagBuilder)

def _compute_rid_sets(self, dataset: DatasetLike) -> dict[tuple[str, str], list[str]]:
    """Compute per-table reachable RID sets for a dataset, client-side.

    Factored from estimate_bag_size's reachability assembly so the bag-build
    path and the estimate share one implementation (and one copy of the
    from_model snapshot-staleness fix). Returns a ``{(schema, table): [RID,...]}``
    map suitable for ``CatalogBagBuilder(rid_sets=...)`` — tuple-keyed (the
    upstream engine's key), unlike compute_reachability's bare-table-name
    rids_by_table. Vocab tables are EXCLUDED (the upstream rid-set branch
    skips vocab; including them would be harmless but wasteful).
    """
```

The helper builds `cb = self._catalog_bag_builder(dataset)`, `reached = cb.iter_reached_paths()`, `anchor_rids`, `model = cb._get_model()`, the `from_model` fetch closure, calls `compute_reachability`, then maps `rids_by_table` (bare name) back to `(schema, table)` keys using `reached`'s keys, dropping vocab tables. `self._ml_instance` is the snapshot catalog (the builder is constructed on the snapshot).

---

### Task 1: `_compute_rid_sets` shared helper

**Files:**
- Modify: `src/deriva_ml/dataset/bag_builder.py` (add `_compute_rid_sets` method)
- Test: `tests/dataset/test_format_b_bag.py` (create)

- [ ] **Step 1: Write the failing test**

A unit test with a synthetic builder is awkward (the helper drives a real walk). Instead, test the **key-mapping logic** in isolation with a small pure helper, plus a live test in Task 4. First, the pure mapping: given `compute_reachability`'s bare-name `rids_by_table` and `reached`'s `(schema,table)` keys, produce the tuple-keyed map dropping vocab. Extract that mapping as a module-level pure function `_rid_sets_from_reachability` and test it:

```python
# tests/dataset/test_format_b_bag.py
"""Tests for Format-B (rid-set) bag generation."""

from deriva_ml.dataset.bag_builder import _rid_sets_from_reachability


def test_rid_sets_from_reachability_tuple_keys_and_drops_vocab():
    # reached: (schema, table) -> [fk_path, ...]  (paths irrelevant here)
    reached = {
        ("eye-ai", "Image"): [()],
        ("eye-ai", "Subject"): [()],
        ("deriva-ml", "Asset_Role"): [()],  # vocab
    }
    rids_by_table = {
        "Image": {"r1", "r2"},
        "Subject": {"s1"},
        "Asset_Role": {"Input", "Output"},  # bare names; vocab
    }
    vocab_tables = {("deriva-ml", "Asset_Role")}
    result = _rid_sets_from_reachability(reached, rids_by_table, vocab_tables)
    # Tuple-keyed, vocab dropped, RID lists sorted for determinism.
    assert result == {
        ("eye-ai", "Image"): ["r1", "r2"],
        ("eye-ai", "Subject"): ["s1"],
    }


def test_rid_sets_from_reachability_missing_table_is_empty_list():
    """A reached non-vocab table with no RID-set entry maps to []."""
    reached = {("S", "T"): [()]}
    result = _rid_sets_from_reachability(reached, {}, set())
    assert result == {("S", "T"): []}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_format_b_bag.py -v`
Expected: FAIL — `ImportError: cannot import name '_rid_sets_from_reachability'`

- [ ] **Step 3: Write minimal implementation**

Add the pure mapping function at module level in `src/deriva_ml/dataset/bag_builder.py` (near the other module-level helpers, after the imports):

```python
def _rid_sets_from_reachability(
    reached: dict[tuple[str, str], Any],
    rids_by_table: dict[str, set[str]],
    vocab_tables: set[tuple[str, str]],
) -> dict[tuple[str, str], list[str]]:
    """Map compute_reachability output to the upstream rid_sets shape.

    ``compute_reachability`` returns ``rids_by_table`` keyed by bare table
    name; ``CatalogBagBuilder(rid_sets=...)`` wants ``{(schema, table):
    [RID,...]}``. This re-keys using ``reached``'s ``(schema, table)`` keys,
    drops vocab tables (the upstream rid-set branch skips them — vocab keeps
    its full/per-path query), and sorts each RID list for deterministic specs.

    Args:
        reached: ``{(schema, table): [fk_path, ...]}`` — the reached-paths map
            (its keys enumerate the non-vocab tables to emit).
        rids_by_table: ``{table_name: set(RID)}`` from compute_reachability.
        vocab_tables: ``(schema, table)`` pairs that are vocabulary tables.

    Returns:
        ``{(schema, table): [RID, ...]}`` for every non-vocab reached table.
    """
    result: dict[tuple[str, str], list[str]] = {}
    for key in reached:
        if key in vocab_tables:
            continue
        schema_name, table_name = key
        result[key] = sorted(rids_by_table.get(table_name, set()))
    return result
```

(`Any` is already imported in this module — verify; if not, add `from typing import Any`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_format_b_bag.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/dataset/bag_builder.py tests/dataset/test_format_b_bag.py && git commit -m "feat(bag): rid-sets-from-reachability key-mapping helper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `_compute_rid_sets` method + factor estimate onto it

**Files:**
- Modify: `src/deriva_ml/dataset/bag_builder.py` (add `_compute_rid_sets` method)
- Modify: `src/deriva_ml/dataset/dataset.py` (`estimate_bag_size` calls it)

- [ ] **Step 1: Write the failing test**

A wiring-shape guard: `estimate_bag_size`'s source should call `_compute_rid_sets`-shared logic rather than inlining `compute_reachability` + the from_model closure. We assert the shared method exists and that estimate no longer inlines its own `datapath.from_model` call (it delegates).

```python
# add to tests/dataset/test_format_b_bag.py
def test_compute_rid_sets_method_exists():
    from deriva_ml.dataset.bag_builder import DatasetBagBuilder

    assert hasattr(DatasetBagBuilder, "_compute_rid_sets")


def test_estimate_delegates_reachability_assembly():
    """estimate_bag_size should not inline its own from_model fetch closure
    once the shared helper exists — it reuses _compute_rid_sets' assembly."""
    import inspect

    from deriva_ml.dataset.bag_builder import DatasetBagBuilder

    src = inspect.getsource(DatasetBagBuilder._compute_rid_sets)
    # The from_model fix lives in the shared helper.
    assert "from_model" in src
    assert "compute_reachability" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_format_b_bag.py::test_compute_rid_sets_method_exists -v`
Expected: FAIL — `_compute_rid_sets` doesn't exist yet.

- [ ] **Step 3: Write minimal implementation**

READ `estimate_bag_size` in `dataset.py` (~2722–2790) to copy its exact reachability assembly. Add `_compute_rid_sets` to `DatasetBagBuilder` in `bag_builder.py`. It must return BOTH the rid_sets map AND the auxiliary data the estimate still needs (asset lengths, fetched rows for sampling) — OR be scoped to just rid_sets and have a sibling for the estimate's extra outputs. To keep the estimate working unchanged, make `_compute_rid_sets` return a richer result the estimate can use:

```python
    def _compute_rid_sets(
        self, dataset: DatasetLike
    ) -> tuple[
        dict[tuple[str, str], list[str]],
        dict[str, set[str]],
        dict[str, dict[str, int]],
        dict[tuple[str, str], list[dict]],
        dict[str, list[dict]],
    ]:
        """Compute per-table reachable RID sets for a dataset, client-side.

        Factored from estimate_bag_size's reachability assembly so the
        bag-build path and the estimate share one implementation (and one
        copy of the from_model snapshot-staleness fix — see
        docs memory snapshot-held-model-staleness-hazard).

        Returns:
            A 5-tuple:
            - ``rid_sets``: ``{(schema, table): [RID,...]}`` for non-vocab
              reached tables — the shape ``CatalogBagBuilder(rid_sets=...)``
              consumes (vocab excluded).
            - ``rids_by_table``: bare-name ``{table: set(RID)}`` (estimate
              row counts).
            - ``asset_lengths_by_table``: ``{table: {RID: Length}}`` (estimate
              asset bytes).
            - ``fetched_rows``: ``{(schema, table): rows}`` (estimate CSV-byte
              sampling).
            - ``sample_rows_by_table``: ``{table: [row,...]}`` derived sample.
        """
        from deriva_ml.dataset._reachability import (
            compute_reachability,
            sample_rows_from_fetched,
        )

        cb = self._catalog_bag_builder(dataset=dataset)
        reached = cb.iter_reached_paths()
        anchor_rids = [dataset.dataset_rid] + list(self._iter_descendant_rids(dataset))
        model = cb._get_model()

        # Build the path builder from the SAME model the walk used (not the
        # held-model pathBuilder()) — the snapshot-staleness fix. See
        # docs memory snapshot-held-model-staleness-hazard.
        pb = datapath.from_model(self._ml_instance.catalog, model)

        def _fetch(schema: str, table: str, columns: set[str]) -> list[dict]:
            tpb = pb.schemas[schema].tables[table]
            try:
                attrs = [getattr(tpb, c) for c in sorted(columns)]
                return list(tpb.attributes(*attrs).fetch())
            except Exception:  # noqa: BLE001
                return list(tpb.entities().fetch())

        rids_by_table, asset_lengths_by_table, fetched_rows = compute_reachability(
            reached=reached, anchor_rids=anchor_rids, model=model, fetch=_fetch
        )
        sample_rows_by_table = sample_rows_from_fetched(
            reached=reached, fetched_rows=fetched_rows
        )

        vocab_tables = {
            key for key in reached
            if model.schemas[key[0]].tables[key[1]].is_vocabulary()
        }
        rid_sets = _rid_sets_from_reachability(reached, rids_by_table, vocab_tables)

        return (
            rid_sets,
            rids_by_table,
            asset_lengths_by_table,
            fetched_rows,
            sample_rows_by_table,
        )
```

**`datapath` is NOT currently imported in `bag_builder.py`** (verified — it only appears in docstrings). Add the import using the SAME importlib form `dataset.py` uses to avoid shadowing by local `deriva.py` files. At the top of `bag_builder.py`, after the existing imports, add:
```python
import importlib

# Import datapath via importlib to avoid shadowing by local 'deriva.py' files
# (mirrors dataset.py's pattern).
datapath = importlib.import_module("deriva.core.datapath")
```
(If `importlib` is already imported, reuse it. Check the existing imports first.)

Now refactor `estimate_bag_size` in `dataset.py` to delegate. Replace its inlined assembly (the block from `version_snapshot_catalog = ...` through the `compute_reachability`/`sample_rows_from_fetched` calls) with:

```python
        version_snapshot_catalog = self._version_snapshot_catalog(version)
        builder = DatasetBagBuilder(
            ml_instance=version_snapshot_catalog,
            exclude_tables=exclude_tables,
        )
        (
            _rid_sets,
            rids_by_table,
            asset_lengths_by_table,
            _fetched_rows,
            sample_rows_by_table,
        ) = builder._compute_rid_sets(self)

        # asset_tables for assemble_estimate: any reached table flagged asset.
        model = builder._catalog_bag_builder(dataset=self)._get_model()
        ...
```

Wait — recomputing the model/reached for `asset_tables` would re-walk. Instead, have `estimate_bag_size` derive `asset_tables` from the data it already has. The current estimate computes `asset_tables` from `reached` + model. To avoid a second walk, `_compute_rid_sets` should ALSO return the asset-table set. ADD a sixth return element `asset_tables: set[str]` (bare names) computed inside `_compute_rid_sets`:

```python
        asset_tables = {
            key[1] for key in reached
            if model.schemas[key[0]].tables[key[1]].is_asset()
        }
```

and return it as the final tuple element. Update the helper's return type/docstring to a 6-tuple and the estimate's unpacking accordingly. Then `estimate_bag_size`'s `assemble_estimate` call uses that `asset_tables` directly (it already takes `asset_tables` after RE-Task 7).

The estimate's final assembly becomes:
```python
        (
            _rid_sets,
            rids_by_table,
            asset_lengths_by_table,
            _fetched_rows,
            sample_rows_by_table,
            asset_tables,
        ) = builder._compute_rid_sets(self)

        return assemble_estimate(
            asset_tables=asset_tables,
            rids_by_table=rids_by_table,
            asset_lengths_by_table=asset_lengths_by_table,
            sample_rows_by_table=sample_rows_by_table,
            estimate_csv_bytes=self._estimate_csv_bytes,
            human_readable_size=self._human_readable_size,
        )
```

> **Implementer note:** READ the current `estimate_bag_size` body first — RE-Task 6/7 already wired it to `compute_reachability` + `assemble_estimate(asset_tables=...)`. You are MOVING that assembly into `_compute_rid_sets` and having the estimate call it. Preserve the exact behavior: the live estimate test (`tests/dataset/test_estimate_bag_size.py::test_reachability_matches_server_union`) must still pass. The 6-tuple is unwieldy but keeps one walk; that's the right tradeoff over re-walking.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_format_b_bag.py tests/dataset/test_reachability.py tests/dataset/test_estimate_helpers.py -v
```
Expected: PASS (the new helper tests + the existing reachability/estimate-helper units).

Run the live estimate exactness test (needs DERIVA_HOST) to confirm the refactor preserved estimate behavior:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_estimate_bag_size.py::test_reachability_matches_server_union -v
```
Expected: PASS (estimate counts still == server union — the assembly moved but didn't change).

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/dataset/bag_builder.py src/deriva_ml/dataset/dataset.py tests/dataset/test_format_b_bag.py && git commit -m "refactor(bag): factor reachability assembly into _compute_rid_sets; estimate delegates

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `build_bag` produces a Format-B bag

**Files:**
- Modify: `src/deriva_ml/dataset/bag_builder.py` (`build_bag` ~243, `_catalog_bag_builder` ~449)
- Test: `tests/dataset/test_format_b_bag.py`

- [ ] **Step 1: Write the failing test**

A wiring-shape guard that `build_bag` computes rid_sets and constructs its builder with them. We assert `build_bag`'s source references `_compute_rid_sets` and passes `rid_sets=` to the builder.

```python
# add to tests/dataset/test_format_b_bag.py
def test_build_bag_uses_rid_sets():
    """build_bag computes rid_sets and passes them to the CatalogBagBuilder."""
    import inspect

    from deriva_ml.dataset.bag_builder import DatasetBagBuilder

    src = inspect.getsource(DatasetBagBuilder.build_bag)
    assert "_compute_rid_sets" in src
    assert "rid_sets=" in src


def test_catalog_bag_builder_accepts_rid_sets():
    """_catalog_bag_builder forwards an opt-in rid_sets to CatalogBagBuilder."""
    import inspect

    from deriva_ml.dataset.bag_builder import DatasetBagBuilder

    sig = inspect.signature(DatasetBagBuilder._catalog_bag_builder)
    assert "rid_sets" in sig.parameters
    assert sig.parameters["rid_sets"].default is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_format_b_bag.py::test_build_bag_uses_rid_sets tests/dataset/test_format_b_bag.py::test_catalog_bag_builder_accepts_rid_sets -v`
Expected: FAIL — neither wiring exists yet.

- [ ] **Step 3: Write minimal implementation**

(a) Add opt-in `rid_sets` to `_catalog_bag_builder` (`bag_builder.py` ~449). Change the signature to `def _catalog_bag_builder(self, dataset: DatasetLike | None, rid_sets: dict[tuple[str, str], list[str]] | None = None) -> CatalogBagBuilder:` and pass it to the constructor:

```python
        builder = CatalogBagBuilder(
            catalog=self._ml_instance.catalog,
            anchors=anchors,
            output_dir=output_dir,
            policy=policy,
            rid_sets=rid_sets,
        )
```

Document the new param in the method's docstring Args (opt-in; only the download path supplies it; None keeps per-path emission). The three existing callers (`generate_dataset_download_spec`, `generate_dataset_download_annotations`, `aggregate_queries`) call `_catalog_bag_builder(dataset=...)` without `rid_sets` → None → unchanged.

> **Watch:** `_compute_rid_sets` itself calls `self._catalog_bag_builder(dataset=dataset)` (no rid_sets) to run the walk. That's correct — it needs the walk to COMPUTE rid_sets; it must NOT pass rid_sets there (would be circular). Only `build_bag` and the spec path pass rid_sets to a SEPARATE builder used for emission.

(b) Wire `build_bag` (`bag_builder.py` ~243) to compute rid_sets and pass them to its `_SnapshotAwareCatalogBagBuilder`. READ the current `build_bag` — it constructs `_SnapshotAwareCatalogBagBuilder(catalog=, anchors=, output_dir=, policy=)` directly (~line 311). Before that construction, compute rid_sets:

```python
        rid_sets, *_ = self._compute_rid_sets(dataset)
```

and add `rid_sets=rid_sets` to the `_SnapshotAwareCatalogBagBuilder(...)` constructor call.

> **Confirmed (low risk):** `_SnapshotAwareCatalogBagBuilder` (`bag_builder.py` ~71) overrides ONLY `_run_export` — it does NOT override `__init__`. So it inherits `CatalogBagBuilder.__init__` unchanged, which (post-B1) accepts `rid_sets`. The new `rid_sets=` kwarg flows straight through the inherited constructor — no subclass change needed. Just add `rid_sets=rid_sets` to the `_SnapshotAwareCatalogBagBuilder(...)` call in `build_bag`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_format_b_bag.py -v`
Expected: PASS (the wiring-shape guards).

Also confirm the spec-generation path is unbroken (it still passes no rid_sets):
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run python -c "from deriva_ml.dataset.bag_builder import DatasetBagBuilder; print('import OK')"
```

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/dataset/bag_builder.py tests/dataset/test_format_b_bag.py && git commit -m "feat(bag): build_bag produces Format-B bags via rid_sets

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Live Format-B bag round-trip + correctness

**Files:**
- Modify: `tests/dataset/test_format_b_bag.py`

Prove on a live catalog (DERIVA_HOST) that a built bag is genuinely Format B: one CSV per table, RID-distinct, asset fetch.txt correct, and that loading it yields per-table row counts matching the estimate.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/dataset/test_format_b_bag.py
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("DERIVA_HOST") in (None, ""),
    reason="needs a live catalog",
)
def test_format_b_bag_one_csv_per_table_matches_estimate(catalog_with_datasets, tmp_path):
    """A built bag has one clean CSV per table; loading it reproduces the
    estimate's per-table row counts."""
    from deriva_ml.dataset.bag_builder import DatasetBagBuilder

    ml, _desc = catalog_with_datasets
    datasets = list(ml.find_datasets())
    nested = next(
        (d for d in datasets if d.list_dataset_children()),
        datasets[0],
    )
    version = nested.current_version

    # Estimate (the oracle for per-table counts).
    est = nested.estimate_bag_size(version)
    est_counts = {t: d["row_count"] for t, d in est["tables"].items() if d["row_count"] > 0}

    # Build a Format-B bag.
    snap = nested._version_snapshot_catalog(version)
    builder = DatasetBagBuilder(ml_instance=snap)
    zip_path = builder.build_bag(nested, output_dir=tmp_path)
    assert zip_path.exists()

    # Unzip and inspect data/ CSVs.
    import zipfile

    extract = tmp_path / "extracted"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract)

    # Find every {table}.csv under data/ and assert ONE per table basename
    # (Format B: no per-path fragmentation).
    csvs = list(extract.rglob("data/**/*.csv"))
    from collections import Counter

    by_basename = Counter(p.stem for p in csvs)
    multi = {name: n for name, n in by_basename.items() if n > 1}
    assert not multi, f"Format B violated — tables split across multiple CSVs: {multi}"

    # For each estimated table, the bag's CSV row count (minus header) matches.
    import csv as csvmod

    for table, est_count in est_counts.items():
        matches = [p for p in csvs if p.stem == table]
        if not matches:
            continue  # vocab tables use the full query; skip count match here
        with matches[0].open(encoding="utf-8") as fh:
            rows = list(csvmod.DictReader(fh))
        # RID-distinct.
        rids = [r["RID"] for r in rows if "RID" in r]
        assert len(set(rids)) == len(rids), f"{table}.csv has duplicate RIDs"
        assert len(rows) == est_count, f"{table}: bag={len(rows)} estimate={est_count}"
```

- [ ] **Step 2: Run test to verify it fails (or passes if wiring is correct)**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_format_b_bag.py::test_format_b_bag_one_csv_per_table_matches_estimate -v --timeout=600`
Expected: PASS if Tasks 1–3 wired correctly. If it FAILS:
- "tables split across multiple CSVs" → rid_sets weren't passed / per-path emission still active. Re-check Task 3's `rid_sets=` threading into the actual builder `build()` uses.
- a row-count mismatch → the rid_sets for that table don't match the estimate's reachable set; the bag and estimate share `_compute_rid_sets`, so a mismatch means the emission consumed a different map than computed — debug the key shape (tuple vs bare name).
Do NOT weaken the test; fix the wiring.

- [ ] **Step 3: (No new impl — validates Tasks 1–3)**

If failing, the fix is in Task 3's threading (rid_sets must reach the builder whose `build()` runs). Trace `build_bag` → `_SnapshotAwareCatalogBagBuilder(rid_sets=...)` → `build()` → `_build_export_spec` (upstream) → rid-set processors.

- [ ] **Step 4: Confirm pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_format_b_bag.py -v --timeout=600`
Expected: PASS (all — units + live).

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add tests/dataset/test_format_b_bag.py && git commit -m "test(bag): live Format-B bag one-CSV-per-table matches estimate counts

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: deriva-py pin advance, docs, lint, live 2-277G re-measure, PR

**Files:**
- Modify: `uv.lock` (advance the pinned deriva-py commit to include B1)
- Modify: `docs/reference/bag-export.md`, the Stage-B spec

- [ ] **Step 1: Advance the deriva-py pin to include B1**

The pin is `@deriva-ml` (branch). B1 is merged there, but `uv.lock` may point at an older commit. Advance it:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && uv lock --upgrade-package deriva && uv sync
```
Confirm the merged B1 is present:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run python -c "from deriva.core.ermrest_catalog import ErmrestCatalog; import inspect; assert 'rid_set' in inspect.signature(ErmrestCatalog.get_as_file).parameters; from deriva.bag.catalog_builder import CatalogBagBuilder; assert 'rid_sets' in inspect.signature(CatalogBagBuilder.__init__).parameters; print('B1 present')"
```
Expected: "B1 present". If `rid_set`/`rid_sets` are absent, the lock didn't advance — re-run `uv lock --upgrade-package deriva`.

- [ ] **Step 2: Update bag-export.md**

In the "Same engine, different consumer" section and B4: note that a **client-side-built** dataset bag (via `build_bag`) now uses Format B — one rid-set csv processor per reached non-vocab table, producing one clean `data/{schema}/{table}.csv` per table (no per-path fragmentation, no loader union needed). Vocab tables keep their full query. Reference deriva-py's `CatalogBagBuilder(rid_sets=)` and deriva-ml's `_compute_rid_sets`. Keep the MINID/server-export arm's description accurate (it still uses the server export engine; only the client-side `build()` arm is Format B — confirm which arm the user-facing `download_dataset_bag` takes by default and document accordingly).

- [ ] **Step 3: Update the Stage-B spec**

In `docs/superpowers/specs/2026-06-14-stage-b-fast-portable-bag-design.md`, mark B2 shipped and record the live 2-277G bag-gen wall-clock from Step 5.

- [ ] **Step 4: Lint + full unit sweep**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check --fix src/deriva_ml/dataset/bag_builder.py src/deriva_ml/dataset/dataset.py tests/dataset/test_format_b_bag.py && uv run ruff format src/deriva_ml/dataset/bag_builder.py src/deriva_ml/dataset/dataset.py tests/dataset/test_format_b_bag.py
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_format_b_bag.py tests/dataset/test_reachability.py tests/dataset/test_estimate_helpers.py -q
```
Expected: clean + all pass.

- [ ] **Step 5: Live 2-277G bag-gen re-measure + PR**

Run a timed `build_bag` (metadata-only, no asset materialization) against eye-ai 2-277G to capture the headline bag-gen number (needs a fresh www.eye-ai.org token). Target: tens of seconds vs the prior ~280s. Capture for the PR body.

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git push -u origin feature/format-b-bag-generation
gh pr create --base main --title "perf(bag): Format-B client-side bag generation (one clean CSV per table, ~Nx faster)" --body "Wires bag generation to the client-side reachability engine: build_bag computes per-table RID sets (compute_reachability) and hands them to deriva-py's CatalogBagBuilder(rid_sets=) (PR #269), which emits one rid-set csv processor per reached non-vocab table. Result: one clean data/{schema}/{table}.csv per table (no per-path fragmentation, no loader union), bag-gen <N>s vs ~280s on eye-ai 2-277G. Shares _compute_rid_sets with estimate_bag_size (and the from_model snapshot-staleness fix). Stage B2 of the fast-portable-bag effort; B1 = deriva-py #269, A = #300.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Self-Review

**1. Spec coverage** (against `2026-06-14-stage-b-fast-portable-bag-design.md` D1–D4):
- D1 (`get_as_file` rid_set) — shipped in B1; B2 consumes it. ✓
- D2 (Format B one-CSV-per-table) — Task 3 (build_bag passes rid_sets) + Task 4 (live one-CSV-per-table assertion). ✓
- D3 (deriva-ml computes, passes rid_sets in) — Task 1 (mapping) + Task 2 (`_compute_rid_sets`) + Task 3 (build_bag wiring). The opt-in `_catalog_bag_builder(rid_sets=None)` keeps the other 3 callers unchanged. ✓
- D4 (deriva-py first, then deriva-ml) — B1 merged; this is the deriva-ml PR (Task 5). ✓
- The from_model snapshot-staleness fix carried into the shared helper — Task 2. ✓
- Vocab exclusion (vocab keeps full query, not rid-set) — Task 1 (`_rid_sets_from_reachability` drops vocab) + matches B1's upstream `not is_vocabulary()` guard. ✓
- Live bag-gen re-measure (the headline) — Task 5. ✓

**2. Placeholder scan:** No "TBD"/"handle edge cases". Every code step shows complete code. Task 4 Step 3 has no new impl (validates prior tasks) and says so. The 6-tuple return in Task 2 is fully specified (each element named + typed). ✓

**3. Type consistency:**
- `_rid_sets_from_reachability(reached, rids_by_table, vocab_tables) -> {(schema,table): [RID]}` — Task 1, consumed by Task 2's `_compute_rid_sets`. ✓
- `_compute_rid_sets(dataset) -> 6-tuple (rid_sets, rids_by_table, asset_lengths_by_table, fetched_rows, sample_rows_by_table, asset_tables)` — defined Task 2, unpacked by estimate (Task 2) and by `build_bag` (`rid_sets, *_`, Task 3). Consistent. ✓
- `_catalog_bag_builder(dataset, rid_sets=None)` — Task 3; `_compute_rid_sets` calls it WITHOUT rid_sets (correct — avoids circularity, flagged). ✓
- `assemble_estimate(asset_tables=...)` — matches the post-RE-Task-7 signature (Stage A). ✓
- `build_bag` passes `rid_sets=` to `_SnapshotAwareCatalogBagBuilder`, which inherits/forwards to `CatalogBagBuilder(rid_sets=)` (B1). ✓

**Risk flagged for the implementer:** Task 2's 6-tuple is unwieldy but deliberate — it keeps ONE walk shared between the estimate and the rid_sets computation. The alternative (separate methods) re-walks. If the implementer finds the estimate's `asset_tables` derivation differs from what `_compute_rid_sets` returns, reconcile by reading the post-RE-Task-7 `estimate_bag_size` — do not change `assemble_estimate`'s contract. Task 3's `_SnapshotAwareCatalogBagBuilder.__init__` is the integration risk: verify it forwards `rid_sets` (inherits or explicitly passes to super) — a dropped `rid_sets` there silently reverts to per-path emission and Task 4's live test catches it.
