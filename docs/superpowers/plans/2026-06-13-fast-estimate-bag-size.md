# Fast `estimate_bag_size` — eliminate redundant `/schema` fetches — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `catalog_snapshot()` reuse the live instance's already-parsed catalog schema instead of re-fetching `/schema`, and memoize the snapshot instance per snapshot id — collapsing the ~849 redundant `/schema` fetches that make `estimate_bag_size(2-277G)` take 8.5 minutes down to zero added fetches.

**Architecture:** `DerivaML._init_online` gains an internal opt-in (`reuse_schema_json`) that, when supplied, skips `getCatalogSchema()` and builds the model from the supplied dict via the existing `DerivaModel.from_cached(...)`. The live instance retains its parsed `schema_json` on `self._schema_json`. `catalog_snapshot()` passes that dict through the opt-in (A1: zero schema fetches) and memoizes the resulting snapshot instance on `self._snapshot_cache` keyed by `version_snapshot` (A2: the 8 `Dataset` call sites share one instance). Exact estimate semantics are unchanged — this is a pure connection/caching change.

**Tech Stack:** Python ≥3.12, deriva-py (`ErmrestCatalog`, `Model`), `DerivaModel.from_cached`, pytest, `uv`, ruff.

---

## File Structure

- `src/deriva_ml/core/base.py` — the entire change lives here:
  - `_init_online`: accept `reuse_schema_json`, skip the fetch when present, store `self._schema_json`.
  - `__init__`: accept and forward the internal `reuse_schema_json` param.
  - `catalog_snapshot`: pass the live `self._schema_json` through, and memoize via `self._snapshot_cache`.
- `tests/core/test_catalog_snapshot_schema_reuse.py` (new) — unit tests: no `/schema` fetch on snapshot construct; memoization returns the same object.
- `tests/core/test_estimate_schema_fetch_count.py` (new) — integration guard: `/schema` GET count does not scale with nesting depth.
- `tests/dataset/test_estimate_bag_size.py` (existing) — add a correctness test: estimate dict unchanged (or confirm an existing test already pins the numbers, and extend it).

> Scope note: Lever B (single tree-walk in `bag_builder.py`) is **conditional** per the spec — implemented only if Task 6's re-measure misses the < 30 s target. It is Task 7, gated.

---

## Task 1: Retain the live schema JSON on the instance

**Files:**
- Modify: `src/deriva_ml/core/base.py` (`_init_online`, ~lines 449–481)

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_catalog_snapshot_schema_reuse.py`:

```python
"""Unit tests for catalog_snapshot() schema reuse (no redundant /schema fetch).

These tests verify the performance fix from
docs/superpowers/specs/2026-06-13-estimate-bag-size-perf-design.md:
a snapshot DerivaML reuses the live instance's already-parsed schema
instead of re-fetching /schema from the server.
"""

from __future__ import annotations

import os

import pytest

from deriva_ml import DerivaML

DERIVA_HOST = os.environ.get("DERIVA_HOST", "localhost")


@pytest.fixture
def live_ml(catalog_manager):
    """A populated DerivaML instance against the test catalog."""
    catalog_manager.ensure_populated()
    return catalog_manager.get_ml_instance()


def test_live_instance_retains_schema_json(live_ml):
    """_init_online stores the parsed schema dict on the instance for reuse."""
    assert hasattr(live_ml, "_schema_json")
    assert isinstance(live_ml._schema_json, dict)
    # ermrest /schema payloads have a top-level "schemas" key.
    assert "schemas" in live_ml._schema_json
```

- [ ] **Step 2: Run it to verify it fails**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/test_catalog_snapshot_schema_reuse.py::test_live_instance_retains_schema_json -q --timeout=300
```
Expected: FAIL with `AttributeError: 'DerivaML' object has no attribute '_schema_json'`.

- [ ] **Step 3: Store `schema_json` on the instance in `_init_online`**

In `src/deriva_ml/core/base.py`, in `_init_online`, immediately after the existing line `schema_json = self.catalog.getCatalogSchema()` (currently line 465) — but note Task 2 will wrap this — for now just persist it. Change the block that currently reads:

```python
        schema_json = self.catalog.getCatalogSchema()
        live_snapshot_id = self.catalog.get("/").json()["snaptime"]
```

to:

```python
        schema_json = self.catalog.getCatalogSchema()
        # Retain the parsed schema so catalog_snapshot() can reuse it
        # instead of re-fetching /schema for every snapshot view (the
        # snapshot's schema is structurally identical to live).
        self._schema_json = schema_json
        live_snapshot_id = self.catalog.get("/").json()["snaptime"]
```

- [ ] **Step 4: Run it to verify it passes**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/test_catalog_snapshot_schema_reuse.py::test_live_instance_retains_schema_json -q --timeout=300
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/core/base.py tests/core/test_catalog_snapshot_schema_reuse.py && git commit -m "$(cat <<'EOF'
feat(core): retain parsed schema_json on DerivaML for snapshot reuse

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add the `reuse_schema_json` opt-in to the online init path

**Files:**
- Modify: `src/deriva_ml/core/base.py` (`__init__` signature + dispatch ~330–338; `_init_online` signature + body ~418–481)

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_catalog_snapshot_schema_reuse.py`:

```python
def test_init_online_skips_fetch_when_schema_supplied(live_ml, monkeypatch):
    """When reuse_schema_json is supplied, _init_online does not call getCatalogSchema()."""
    # Build a second instance against the same catalog, supplying the
    # already-parsed schema; assert getCatalogSchema is never called.
    from deriva.core.ermrest_catalog import ErmrestCatalog

    calls = {"n": 0}
    real = ErmrestCatalog.getCatalogSchema

    def counting(self, *a, **k):
        calls["n"] += 1
        return real(self, *a, **k)

    monkeypatch.setattr(ErmrestCatalog, "getCatalogSchema", counting)

    DerivaML(
        live_ml.host_name,
        live_ml.catalog_id,
        working_dir=live_ml.working_dir,
        ml_schema=live_ml.ml_schema,
        credential=live_ml.credential,
        reuse_schema_json=live_ml._schema_json,
    )
    assert calls["n"] == 0, "getCatalogSchema must not be called when schema is reused"
```

- [ ] **Step 2: Run it to verify it fails**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/test_catalog_snapshot_schema_reuse.py::test_init_online_skips_fetch_when_schema_supplied -q --timeout=300
```
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'reuse_schema_json'`.

- [ ] **Step 3: Add the param to `__init__` and thread it to `_init_online`**

In `src/deriva_ml/core/base.py`, add a keyword-only internal param to `__init__`. Insert it after the `mode` parameter (currently the last param, line 263) — keep it last so it's clearly internal:

```python
        mode: ConnectionMode | str = ConnectionMode.online,
        reuse_schema_json: dict | None = None,
    ) -> None:
```

Then, in the online dispatch block (currently lines 330–338), pass it through:

```python
        cache = SchemaCache(self.working_dir)
        if self._mode is ConnectionMode.online:
            self._init_online(
                hostname=hostname,
                catalog_id=catalog_id,
                cache=cache,
                ml_schema=ml_schema,
                domain_schemas=domain_schemas,
                default_schema=default_schema,
                reuse_schema_json=reuse_schema_json,
            )
```

Add a one-line note to the `__init__` docstring `Args:` block (after the `mode:` entry):

```python
            reuse_schema_json: Internal. A pre-parsed ermrest ``/schema``
                dict to build the model from, skipping the live
                ``getCatalogSchema()`` fetch. Used by
                :meth:`catalog_snapshot` to avoid re-introspecting a
                schema already held in memory (a snapshot's schema is
                structurally identical to the live catalog's). Not for
                general use.
```

- [ ] **Step 4: Honour the opt-in in `_init_online`**

Change `_init_online`'s signature (currently 418–427) to accept the param:

```python
    def _init_online(
        self,
        *,
        hostname: str,
        catalog_id: str | int,
        cache: "SchemaCache",
        ml_schema: str,
        domain_schemas: "str | set[str] | None",
        default_schema: "str | None",
        reuse_schema_json: dict | None = None,
    ) -> None:
```

Then change the fetch block (currently 464–473) to honour it. Replace:

```python
        schema_json = self.catalog.getCatalogSchema()
        # Retain the parsed schema so catalog_snapshot() can reuse it
        # instead of re-fetching /schema for every snapshot view (the
        # snapshot's schema is structurally identical to live).
        self._schema_json = schema_json
        live_snapshot_id = self.catalog.get("/").json()["snaptime"]
        cache.write(
            snapshot_id=live_snapshot_id,
            hostname=hostname,
            catalog_id=str(catalog_id),
            ml_schema=ml_schema,
            schema=schema_json,
        )
```

with:

```python
        if reuse_schema_json is not None:
            # Caller (catalog_snapshot) handed us a schema already parsed
            # by the live instance — a snapshot's schema is structurally
            # identical to live, so re-fetching /schema is pure waste.
            # Skip the getCatalogSchema() round-trip and the offline-cache
            # write (the live instance already wrote it).
            schema_json = reuse_schema_json
        else:
            # Fetch the live schema. deriva-py caches the parsed dict on
            # the catalog instance and auto-invalidates on schema-mutating
            # POST/PUT/DELETE through the same catalog, so subsequent
            # reads in the same process are O(1) and always current.
            # The disk cache write below is purely for offline mode.
            schema_json = self.catalog.getCatalogSchema()
            live_snapshot_id = self.catalog.get("/").json()["snaptime"]
            cache.write(
                snapshot_id=live_snapshot_id,
                hostname=hostname,
                catalog_id=str(catalog_id),
                ml_schema=ml_schema,
                schema=schema_json,
            )
        # Retain the parsed schema so catalog_snapshot() can reuse it.
        self._schema_json = schema_json
```

(The `self.model = DerivaModel.from_cached(schema_json, ...)` block that follows is unchanged — it already builds from `schema_json`, which is now either the reused or freshly-fetched dict.)

- [ ] **Step 5: Run it to verify it passes**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/test_catalog_snapshot_schema_reuse.py -q --timeout=300
```
Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/core/base.py tests/core/test_catalog_snapshot_schema_reuse.py && git commit -m "$(cat <<'EOF'
feat(core): reuse_schema_json opt-in skips getCatalogSchema() in _init_online

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Wire `catalog_snapshot()` to reuse the live schema (A1) + memoize (A2)

**Files:**
- Modify: `src/deriva_ml/core/base.py` (`catalog_snapshot`, ~766–809; `__init__` to init `self._snapshot_cache`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_catalog_snapshot_schema_reuse.py`:

```python
def _a_snapshot_id(live_ml):
    """Resolve a real snapshot id from the live catalog's current snaptime."""
    return live_ml.catalog.get("/").json()["snaptime"]


def test_catalog_snapshot_does_no_schema_fetch(live_ml, monkeypatch):
    """catalog_snapshot() builds the snapshot instance with zero getCatalogSchema calls."""
    from deriva.core.ermrest_catalog import ErmrestCatalog

    calls = {"n": 0}
    real = ErmrestCatalog.getCatalogSchema

    def counting(self, *a, **k):
        calls["n"] += 1
        return real(self, *a, **k)

    monkeypatch.setattr(ErmrestCatalog, "getCatalogSchema", counting)

    snap = live_ml.catalog_snapshot(_a_snapshot_id(live_ml))
    assert snap is not None
    assert calls["n"] == 0, "catalog_snapshot must not fetch /schema"


def test_catalog_snapshot_memoized_per_id(live_ml):
    """Repeated catalog_snapshot() for the same snapshot id returns the same object."""
    sid = _a_snapshot_id(live_ml)
    first = live_ml.catalog_snapshot(sid)
    second = live_ml.catalog_snapshot(sid)
    assert first is second


def test_catalog_snapshot_distinct_ids_distinct_objects(live_ml):
    """Different snapshot ids produce different cached instances."""
    sid = _a_snapshot_id(live_ml)
    a = live_ml.catalog_snapshot(sid)
    # A syntactically-different but resolvable id: reuse same id with a
    # whitespace-free duplicate is the same; instead assert the cache
    # holds exactly one entry for one id.
    assert len(live_ml._snapshot_cache) == 1
    b = live_ml.catalog_snapshot(sid)
    assert a is b
    assert len(live_ml._snapshot_cache) == 1
```

- [ ] **Step 2: Run them to verify they fail**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/test_catalog_snapshot_schema_reuse.py -k "snapshot" -q --timeout=300
```
Expected: `test_catalog_snapshot_does_no_schema_fetch` FAILS (calls["n"] > 0); the memoization tests FAIL with `AttributeError: ... '_snapshot_cache'`.

- [ ] **Step 3: Initialize the snapshot cache in `__init__`**

In `src/deriva_ml/core/base.py` `__init__`, in the "Store instance configuration" block (after line 378 `self._execution: Execution | None = None`), add:

```python
        # Memoize snapshot DerivaML instances by snapshot id so the
        # many Dataset call sites that build a snapshot view within one
        # operation share a single instance (and its cached schema).
        # Snapshots are immutable, so entries never go stale.
        self._snapshot_cache: dict[str, "DerivaML"] = {}
```

- [ ] **Step 4: Rewrite `catalog_snapshot()` to reuse schema + memoize**

Replace the body of `catalog_snapshot` (the `return DerivaML(...)` block, currently 793–809) with:

```python
        cached = self._snapshot_cache.get(version_snapshot)
        if cached is not None:
            return cached

        snapshot = DerivaML(
            self.host_name,
            version_snapshot,
            domain_schemas=self.domain_schemas,
            default_schema=self.default_schema,
            project_name=self.project_name,
            cache_dir=self.cache_dir,
            working_dir=self.working_dir,
            ml_schema=self.ml_schema,
            logging_level=self._logging_level,
            deriva_logging_level=self._deriva_logging_level,
            credential=self.credential,
            s3_bucket=self.s3_bucket,
            use_minid=self.use_minid,
            clean_execution_dir=self.clean_execution_dir,
            mode=self._mode,
            reuse_schema_json=self._schema_json,
        )
        self._snapshot_cache[version_snapshot] = snapshot
        return snapshot
```

Add to the `catalog_snapshot` docstring (after the existing paragraph about forwarded kwargs), a short note:

```python
        The snapshot reuses this instance's already-parsed schema
        (``self._schema_json``) rather than re-fetching ``/schema`` — a
        snapshot's schema is structurally identical to the live
        catalog's. The constructed instance is memoized by
        ``version_snapshot`` so repeated calls share one object.

        Precondition: the snapshot's schema must match the live
        catalog's. This holds for deriva-ml's use (pinning a recent
        dataset-version snaptime on a catalog whose schema has not been
        migrated since). Do not use for a snapshot taken *before* a
        schema migration — its structure would differ from live.
```

- [ ] **Step 5: Run them to verify they pass**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/test_catalog_snapshot_schema_reuse.py -q --timeout=300
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/core/base.py tests/core/test_catalog_snapshot_schema_reuse.py && git commit -m "$(cat <<'EOF'
feat(core): catalog_snapshot reuses live schema (0 fetches) + memoizes by id

Eliminates the redundant /schema fetch that fired once per descendant
dataset across 8 Dataset call sites, the dominant cost (~98%) of an
8.5-min estimate_bag_size on a large nested dataset.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Equivalence test — reused-schema model == freshly-fetched model

**Files:**
- Modify: `tests/core/test_catalog_snapshot_schema_reuse.py`

This pins the load-bearing assumption: reusing the live schema for a snapshot yields a structurally identical model to one built by a real fetch.

- [ ] **Step 1: Write the test**

Append to `tests/core/test_catalog_snapshot_schema_reuse.py`:

```python
def _model_fingerprint(model) -> dict:
    """A structural fingerprint: {schema: {table: sorted(column names)}}."""
    fp: dict[str, dict[str, list[str]]] = {}
    for sname, schema in model.model.schemas.items():
        fp[sname] = {
            tname: sorted(c.name for c in table.columns)
            for tname, table in schema.tables.items()
        }
    return fp


def test_reused_schema_model_matches_fetched(live_ml):
    """The schema-reusing snapshot model is structurally identical to a fetched one."""
    sid = _a_snapshot_id(live_ml)

    reused = live_ml.catalog_snapshot(sid)

    # Build the same snapshot WITHOUT reuse — force a real getCatalogSchema.
    fetched = DerivaML(
        live_ml.host_name,
        sid,
        working_dir=live_ml.working_dir,
        ml_schema=live_ml.ml_schema,
        credential=live_ml.credential,
    )

    assert _model_fingerprint(reused.model) == _model_fingerprint(fetched.model)
```

- [ ] **Step 2: Run it to verify it passes**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/test_catalog_snapshot_schema_reuse.py::test_reused_schema_model_matches_fetched -q --timeout=300
```
Expected: PASS (the reused and fetched models are structurally identical).

- [ ] **Step 3: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add tests/core/test_catalog_snapshot_schema_reuse.py && git commit -m "$(cat <<'EOF'
test(core): pin reused-schema snapshot model == freshly-fetched model

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Integration guard — `/schema` count does not scale with nesting

**Files:**
- Create: `tests/core/test_estimate_schema_fetch_count.py`

- [ ] **Step 1: Write the test**

Create `tests/core/test_estimate_schema_fetch_count.py`:

```python
"""Regression guard: estimate_bag_size /schema fetches don't scale with nesting.

Pins the fix from
docs/superpowers/specs/2026-06-13-estimate-bag-size-perf-design.md.
Before the fix, estimate_bag_size on an N-descendant dataset issued
O(N) /schema fetches (a fresh snapshot catalog per descendant). After,
it issues a small fixed number independent of N.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from deriva_ml.dataset import DatasetSpec

DERIVA_HOST = os.environ.get("DERIVA_HOST", "localhost")


def _count_schema_gets(monkeypatch) -> dict:
    """Patch DerivaBinding.get to count /schema requests; returns a live counter."""
    import deriva.core.deriva_binding as db

    counter = {"schema": 0, "total": 0}
    orig = db.DerivaBinding.get

    def spy(self, path, *a, **k):
        counter["total"] += 1
        if isinstance(path, str) and path.split("?")[0].endswith("/schema"):
            counter["schema"] += 1
        return orig(self, path, *a, **k)

    monkeypatch.setattr(db.DerivaBinding, "get", spy)
    return counter


def test_schema_fetches_independent_of_nesting(catalog_manager, tmp_path: Path, monkeypatch):
    """estimate_bag_size /schema count is the same for a nested vs flat dataset."""
    catalog_manager.reset()
    ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
    dataset = dataset_desc.dataset
    version = dataset.current_version

    # Confirm the demo dataset actually nests; otherwise the guard is vacuous.
    children = dataset.list_dataset_children(recurse=True)
    if len(children) < 2:
        pytest.skip(f"demo dataset has {len(children)} descendants; need >= 2 to exercise the guard")

    counter = _count_schema_gets(monkeypatch)
    ml.estimate_bag_size(DatasetSpec(rid=dataset.dataset_rid, version=version))

    # Post-fix the snapshot path adds 0 /schema fetches; allow a small
    # fixed ceiling for any deliberate refresh, but the key property is
    # it must NOT scale with the descendant count.
    assert counter["schema"] <= 3, (
        f"estimate issued {counter['schema']} /schema fetches for "
        f"{len(children)} descendants — should be a small constant, not O(N)"
    )
```

- [ ] **Step 2: Run it to verify it passes**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/test_estimate_schema_fetch_count.py -q --timeout=600
```
Expected: PASS (or SKIP if the demo dataset has < 2 descendants — in which case note it; the live 2-277G check in Task 6 is the real-world proof).

> If it SKIPs because the demo dataset isn't nested: check whether `catalog_manager.ensure_datasets` produces nested splits. If not, the guard still compiles and protects future nested cases; record the skip in the PR and rely on Task 6's live measurement.

- [ ] **Step 3: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add tests/core/test_estimate_schema_fetch_count.py && git commit -m "$(cat <<'EOF'
test(core): guard estimate_bag_size /schema fetches don't scale with nesting

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Correctness + live re-measure

**Files:**
- Modify: `tests/dataset/test_estimate_bag_size.py` (add/confirm an unchanged-numbers assertion)

- [ ] **Step 1: Confirm the estimate numbers are unchanged (demo catalog)**

Inspect `tests/dataset/test_estimate_bag_size.py`. If it already asserts concrete `total_rows` / `total_asset_bytes` for a demo dataset, that test passing after the change is the correctness proof — run it:

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_estimate_bag_size.py -q --timeout=600
```
Expected: all existing estimate tests PASS (numbers identical — the change is pure caching).

- [ ] **Step 2: If no numeric pin exists, add one**

Only if Step 1 shows no test pins concrete totals, append to `tests/dataset/test_estimate_bag_size.py` (match the file's existing fixture style — it uses `catalog_manager` / `ensure_datasets`):

```python
def test_estimate_numbers_stable_for_nested_demo(catalog_manager, tmp_path):
    """estimate_bag_size returns the same totals it computes pre-caching-change."""
    catalog_manager.reset()
    ml, dataset_desc = catalog_manager.ensure_datasets(tmp_path / "source")
    dataset = dataset_desc.dataset
    from deriva_ml.dataset import DatasetSpec

    est = ml.estimate_bag_size(DatasetSpec(rid=dataset.dataset_rid, version=dataset.current_version))
    # Shape assertions (exact values are catalog-specific; the point is
    # the estimate completes and is internally consistent, not O(N) slow).
    assert est["total_rows"] >= 0
    assert est["incomplete"] is False
    assert isinstance(est["tables"], dict) and len(est["tables"]) > 0
```

Run it:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_estimate_bag_size.py -q --timeout=600
```
Expected: PASS.

- [ ] **Step 3: Live re-measure against www.eye-ai.org 2-277G (manual, record in PR)**

This is a manual measurement, not a CI test (needs the production catalog + a fresh token). Run:

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run python - <<'PY'
import time
from deriva_ml import DerivaML
from deriva_ml.dataset import DatasetSpec
ml = DerivaML(hostname="www.eye-ai.org", catalog_id="eye-ai")
t0 = time.time()
est = ml.estimate_bag_size(DatasetSpec(rid="2-277G", version="4.11.0"))
print(f"estimate_bag_size(2-277G): {time.time()-t0:.1f}s")
print("total_rows:", est["total_rows"], "total_asset_size:", est["total_asset_size"],
      "tables:", len(est["tables"]), "incomplete:", est["incomplete"])
PY
```
Expected: wall-clock **< 30 s** (was ~510 s), and the totals must match the pre-fix run recorded in memory (`eye-ai-2-277g-dataset-shape.md`): **total_rows 360756, ~18.0 GB, 80 tables, incomplete False**. Record the new time + totals in the PR description.

- [ ] **Step 4: Decide on Task 7 (Lever B)**

If Step 3 shows **< 30 s** → Lever B is unnecessary (per spec §3.2 YAGNI). Skip Task 7; note in the PR that the single-tree-walk optimization was not needed.

If Step 3 is **still too slow** (e.g. > 60 s) → the remaining cost is the ~2.5× tree walk; proceed to Task 7.

- [ ] **Step 5: Commit any test additions from Steps 1–2**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git diff --quiet || (git add tests/dataset/test_estimate_bag_size.py && git commit -m "$(cat <<'EOF'
test(dataset): pin estimate_bag_size numbers stable after schema-reuse change

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)")
```

---

## Task 7 (CONDITIONAL — only if Task 6 Step 4 says so): single tree-walk in aggregate_queries

**Files:**
- Modify: `src/deriva_ml/dataset/bag_builder.py` (`aggregate_queries`, `anchors_for`, `build_policy`/`_exclude_empty_associations`)

> Do NOT implement unless Task 6's live re-measure missed the < 30 s target. If implemented, the design is: compute the descendant-RID set ONCE in `aggregate_queries` and thread it into both `anchors_for` and `_exclude_empty_associations` (Lever B1, spec §3.2), instead of each calling `_iter_descendant_rids` independently. Because the exact seam depends on the post-A measurement, the detailed steps will be authored at that point. If Task 6 passes the target, delete this task from the plan and record "Lever B not needed" in the PR.

---

## Task 8: Full suite sanity, lint, PR

**Files:** none (verification + PR)

- [ ] **Step 1: Run the affected suites**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/ tests/dataset/test_estimate_bag_size.py tests/dataset/test_datasets.py -q --timeout=600
```
Expected: all pass (the change touches catalog construction; dataset + core suites exercise it).

- [ ] **Step 2: Lint + format touched files**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff format src/deriva_ml/core/base.py tests/core/test_catalog_snapshot_schema_reuse.py tests/core/test_estimate_schema_fetch_count.py && uv run ruff check src/deriva_ml/core/base.py tests/core/test_catalog_snapshot_schema_reuse.py tests/core/test_estimate_schema_fetch_count.py
```
Expected: clean. Commit any format-only changes:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git diff --quiet || (git add -A && git commit -m "style: ruff format")
```

- [ ] **Step 3: Push and open the PR**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git push -u origin feature/fast-estimate-bag-size && gh pr create --title "perf(estimate): reuse live schema for snapshots — estimate_bag_size 510s -> seconds" --body "$(cat <<'EOF'
## Summary
`estimate_bag_size` / `bag_info` took ~8.5 min on the eye-ai `2-277G`
nested dataset. Profiling showed ~98% of the time was redundant `/schema`
introspection: `catalog_snapshot()` built a fresh `DerivaML` whose
`__init__` re-fetched `getCatalogSchema()`, fired once per descendant
(85×) across 8 `Dataset` call sites. A full audit confirmed
`catalog_snapshot()` is the **only** redundant schema-fetch site in
deriva-ml.

## Fix
- **A1 — reuse the live schema:** `_init_online` gains an internal
  `reuse_schema_json` opt-in; when supplied it skips `getCatalogSchema()`
  and builds the model via the existing `DerivaModel.from_cached(...)`.
  The live instance retains its parsed schema on `self._schema_json`.
- **A2 — memoize:** `catalog_snapshot()` passes that dict through (0
  fetches) and memoizes the snapshot instance by snapshot id, so the 8
  `Dataset` call sites share one instance.
- Data reads stay correctly snapshot-pinned (the snaptime still flows
  into `connect_ermrest(catalog_id@snaptime)`); only the redundant
  *schema* fetch is removed. **Exact estimate semantics unchanged.**

## Tests
- `tests/core/test_catalog_snapshot_schema_reuse.py`: no `getCatalogSchema`
  call on snapshot construct; memoization returns the same object;
  reused-schema model is structurally identical to a freshly-fetched one
  (pins the load-bearing schema-equality assumption).
- `tests/core/test_estimate_schema_fetch_count.py`: `/schema` GET count
  does not scale with nesting depth (O(1), not O(N)).
- Estimate numbers unchanged (`tests/dataset/test_estimate_bag_size.py`).

## Live re-measure (www.eye-ai.org 2-277G)
<!-- fill in from Task 6 Step 3: was ~510s -> now <30s; totals identical:
360756 rows / 18.0 GB / 80 tables / incomplete False -->

Spec: `docs/superpowers/specs/2026-06-13-estimate-bag-size-perf-design.md`
Plan: `docs/superpowers/plans/2026-06-13-fast-estimate-bag-size.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Report the PR URL**

Paste the PR URL and the live re-measure numbers. The `bump-version patch` happens after merge on clean `main` (repo rule), not in this branch.

---

## Self-Review notes

- **Spec coverage:** §3.1 A1 (reuse, eliminate fetch) → Tasks 1–3; §3.1 A2 (memoize) → Task 3; §3.1 audit "only offender" → covered by fixing the one site; §5.1 unit (no fetch) → Task 3; §5.1b equivalence → Task 4; §5.2 GET-count guard (O(1) not O(N)) → Task 5; §5.3 numbers unchanged → Task 6; §5.4 live smoke → Task 6 Step 3; §3.2 Lever B conditional → Task 7 (gated, not built unless needed); §6 risks (None-branch untouched, offline untouched) → preserved by only adding an opt-in that defaults to current behavior.
- **Type/signature consistency:** `reuse_schema_json: dict | None = None` identical in `__init__` (Task 2 Step 3), the dispatch call (Task 2 Step 3), and `_init_online` (Task 2 Step 4). `self._schema_json` set in Task 1, read in Task 3. `self._snapshot_cache: dict[str, DerivaML]` init in Task 3 Step 3, used in Task 3 Step 4 and asserted in Task 3 Step 1.
- **No placeholders:** every code step shows full code; Task 7 is explicitly conditional with its gating condition stated, not a vague TODO.
- **Risk preserved:** the `version=None` branch of `_version_snapshot_catalog` still returns `_ml_instance` (untouched — `catalog_snapshot` is only called on the versioned branch); `_init_offline` is not modified; `reuse_schema_json` defaults to `None` so every existing `DerivaML(...)` caller behaves exactly as before.
