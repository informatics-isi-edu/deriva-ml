# tf/torch adapters: FK-reachable element enumeration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `as_tf_dataset` / `as_torch_dataset` enumerate FK-reachable elements (matching `restructure_assets`), via ONE shared reachability core consumed by all three paths.

**Architecture:** Extract a single layer-1 enumeration core (`resolve_reachable_rows`) in `target_resolution.py`; restructure's `_get_reachable_assets` becomes a thin wrapper over it; a `resolve_element_rids` projection (RIDs, order-preserving dedup, `reachable` opt-out) backs both adapters. Layer-2 (materialize-tree vs lazy-yield) stays separate per consumer. Default `reachable=True`.

**Tech Stack:** Python 3.12, `uv`, pytest, SQLAlchemy (bag SQLite), live demo + eye-ai catalogs.

**Spec:** `docs/superpowers/specs/2026-06-16-adapter-fk-reachable-elements-design.md`

---

### Task 1: Shared core `resolve_reachable_rows` + `resolve_element_rids`

**Files:**
- Modify: `src/deriva_ml/dataset/target_resolution.py`
- Test: `tests/dataset/test_reachable_enumeration.py` (new)

- [ ] **Step 1: Write failing unit tests (stub bag — no catalog)**

```python
"""resolve_reachable_rows / resolve_element_rids: the shared FK-reachable
element enumeration used by restructure_assets + the tf/torch adapters."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deriva_ml.dataset.target_resolution import (
    resolve_reachable_rows,
    resolve_element_rids,
)


def _stub_bag(rows, *, has_table=True):
    """Bag whose _dataset_table_view-backed reachable query yields `rows`."""
    bag = MagicMock()
    # resolve_reachable_rows reads rows via the same mechanism _get_reachable_assets
    # uses; stub the helper boundary so the test is catalog-free.
    bag._reachable_rows_impl = lambda table: list(rows)  # see Step 3 seam
    # table-existence check
    bag.model.name_to_table = {"Image": object()} if has_table else {}
    return bag


def test_resolve_reachable_rows_returns_rows_for_reachable_table():
    rows = [{"RID": "i1", "Filename": "a.png"}, {"RID": "i2", "Filename": "b.png"}]
    bag = _stub_bag(rows)
    out = resolve_reachable_rows(bag, "Image")
    assert out == rows  # full rows, not deduped here


def test_resolve_element_rids_reachable_dedups_preserving_order():
    # Same RID surfaced twice via two FK paths must yield once, in first-seen order.
    rows = [{"RID": "i1"}, {"RID": "i2"}, {"RID": "i1"}, {"RID": "i3"}]
    bag = _stub_bag(rows)
    assert resolve_element_rids(bag, "Image", reachable=True) == ["i1", "i2", "i3"]


def test_resolve_element_rids_direct_uses_list_members():
    bag = MagicMock()
    bag.list_dataset_members.return_value = {"Image": [{"RID": "d1"}, {"RID": "d2"}]}
    out = resolve_element_rids(bag, "Image", reachable=False)
    assert out == ["d1", "d2"]
    bag.list_dataset_members.assert_called_once_with(recurse=True)


def test_resolve_element_rids_unknown_type_raises():
    from deriva_ml.core.exceptions import DerivaMLException

    bag = _stub_bag([], has_table=False)
    with pytest.raises(DerivaMLException, match="not found|not resolvable"):
        resolve_element_rids(bag, "Nope", reachable=True)
```

- [ ] **Step 2: Run → fail (ImportError)**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_reachable_enumeration.py -q`
Expected: FAIL — `cannot import name 'resolve_reachable_rows'`.

- [ ] **Step 3: Implement the core + projection**

In `target_resolution.py` add (importing `Session` from `sqlalchemy.orm` and the
exceptions module):

```python
def resolve_reachable_rows(bag, table: str) -> list[dict]:
    """All rows of `table` reachable from this dataset via any FK path.

    Single source of truth for dataset element reachability. Wraps
    ``bag._dataset_table_view(table)`` (Dataset -> ... -> table UNION over all
    FK paths, scoped to this dataset + nested datasets). Returns full row dicts.
    NOT RID-deduped (a UNION can surface a RID via two paths; restructure
    collapses by filename, the adapters dedup in resolve_element_rids).
    """
    from sqlalchemy.orm import Session

    if not _bag_has_table(bag, table):
        raise DerivaMLException(
            f"Element type {table!r} not found in bag; available types: "
            f"{sorted(_bag_table_names(bag))}"
        )
    sql_query = bag._dataset_table_view(table)
    with Session(bag.engine) as session:
        return [dict(m) for m in session.execute(sql_query).mappings().all()]


def resolve_element_rids(bag, element_type: str, *, reachable: bool = True) -> list[str]:
    """RIDs of `element_type` rows belonging to this dataset (order-preserving,
    RID-deduped). reachable=True (default): FK reachability; reachable=False:
    direct members only."""
    if not reachable:
        members = bag.list_dataset_members(recurse=True)
        if element_type not in members:
            raise DerivaMLException(
                f"Element type {element_type!r} not found in bag; available "
                f"types: {sorted(members.keys())}"
            )
        rows = members[element_type]
    else:
        rows = resolve_reachable_rows(bag, element_type)
    seen, out = set(), []
    for r in rows:
        rid = r["RID"]
        if rid not in seen:
            seen.add(rid)
            out.append(rid)
    return out
```

Add small helpers `_bag_has_table(bag, table)` / `_bag_table_names(bag)` using
the bag's existing model surface (mirror whatever `get_table_as_dict` / the
adapters' current `_bag_element_is_asset` use to resolve a table — reuse, don't
invent). Adjust the Step-1 stub's `_reachable_rows_impl` seam to match the real
attribute the impl reads (likely patch `bag._dataset_table_view` + `bag.engine`
instead — update the test stub accordingly so it exercises the real path).

- [ ] **Step 4: Run → pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_reachable_enumeration.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/dataset/target_resolution.py tests/dataset/test_reachable_enumeration.py
git commit -m "feat(dataset): shared FK-reachable element enumeration (resolve_reachable_rows / resolve_element_rids)"
```

---

### Task 2: Route `restructure._get_reachable_assets` onto the shared core

Eliminate restructure's duplicate reachability impl so all paths share one.

**Files:**
- Modify: `src/deriva_ml/dataset/restructure.py` (`_get_reachable_assets`, ~line 207)
- Test: existing restructure tests must stay green (regression guard)

- [ ] **Step 1: Implement**

Replace `_get_reachable_assets`'s body (the inline `Session`/`_dataset_table_view`
block) with a delegation:

```python
def _get_reachable_assets(bag, asset_table: str) -> list[dict[str, Any]]:
    """Get all assets reachable from this dataset through any FK path.

    Thin wrapper over the shared reachability core so restructure and the
    tf/torch adapters use one traversal implementation.
    """
    from deriva_ml.dataset.target_resolution import resolve_reachable_rows

    return resolve_reachable_rows(bag, asset_table)
```

Keep the docstring's worked example. Both call sites (lines ~199, ~882) are
unchanged — they still get full rows.

- [ ] **Step 2: Run restructure tests (no behavior change)**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_restructure.py tests/dataset/test_restructure_helpers.py -q`
Expected: PASS (unchanged behavior; restructure now sources rows from the core).

- [ ] **Step 3: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/dataset/restructure.py
git commit -m "refactor(dataset): restructure._get_reachable_assets delegates to shared core"
```

---

### Task 3: Wire both adapters through `resolve_element_rids` + add `reachable`

**Files:**
- Modify: `src/deriva_ml/dataset/tf_adapter.py` (`build_tf_dataset`, lines ~102/126)
- Modify: `src/deriva_ml/dataset/torch_adapter.py` (`build_torch_dataset`, lines ~91/115)
- Test: `tests/dataset/test_format_b_bag.py` or a new adapter-wiring test (source-level)

- [ ] **Step 1: Write failing wiring test**

```python
def test_adapters_enumerate_via_resolve_element_rids():
    """Both builders enumerate via resolve_element_rids (FK-reachable), not
    list_dataset_members directly, and expose a `reachable` param."""
    import inspect
    from deriva_ml.dataset import tf_adapter, torch_adapter

    for mod, fn in ((tf_adapter, "build_tf_dataset"), (torch_adapter, "build_torch_dataset")):
        src = inspect.getsource(getattr(mod, fn))
        assert "resolve_element_rids" in src, fn
        assert "members_by_type[element_type]" not in src, fn  # old path gone
        sig = inspect.signature(getattr(mod, fn))
        assert "reachable" in sig.parameters, fn
        assert sig.parameters["reachable"].default is True, fn
```

- [ ] **Step 2: Run → fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_format_b_bag.py::test_adapters_enumerate_via_resolve_element_rids -q`
Expected: FAIL.

- [ ] **Step 3: Implement in BOTH adapters**

Add `reachable: bool = True` to `build_tf_dataset` / `build_torch_dataset`
signatures. Replace the enumeration+validation block:

```python
# OLD:
members_by_type = bag.list_dataset_members(recurse=True)
if element_type not in members_by_type:
    raise DerivaMLException(...)
...
all_rids = [m["RID"] for m in members_by_type[element_type]]

# NEW:
from deriva_ml.dataset.target_resolution import resolve_element_rids
all_rids = resolve_element_rids(bag, element_type, reachable=reachable)
```

Leave the `is_asset`/`sample_loader` check, `_resolve_targets`,
`targets`/`missing` filtering, `_build_row_lookup`, and the generator body
UNCHANGED. (Note `_build_row_lookup` already uses `get_table_as_dict` — whole
table — so reachable RIDs resolve.)

- [ ] **Step 4: Run wiring test + adapter no-catalog tests**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_format_b_bag.py::test_adapters_enumerate_via_resolve_element_rids tests/dataset/test_tf_adapter_no_tf.py tests/dataset/test_torch_adapter_no_torch.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/dataset/tf_adapter.py src/deriva_ml/dataset/torch_adapter.py tests/dataset/test_format_b_bag.py
git commit -m "fix(dataset): tf/torch adapters enumerate FK-reachable elements via shared core (reachable=True default)"
```

---

### Task 4: Surface `reachable` on the public DatasetBag methods

**Files:**
- Modify: `src/deriva_ml/dataset/dataset_bag.py` (`as_tf_dataset` ~1373, `as_torch_dataset` ~1199)
- Test: signature test

- [ ] **Step 1: Failing signature test**

```python
def test_public_dataset_methods_expose_reachable():
    import inspect
    from deriva_ml.dataset.dataset_bag import DatasetBag

    for name in ("as_tf_dataset", "as_torch_dataset"):
        sig = inspect.signature(getattr(DatasetBag, name))
        assert "reachable" in sig.parameters, name
        assert sig.parameters["reachable"].default is True, name
```

- [ ] **Step 2: Run → fail.** `pytest tests/dataset/test_format_b_bag.py::test_public_dataset_methods_expose_reachable -q`

- [ ] **Step 3: Implement**

Add `reachable: bool = True` to both public method signatures; forward
`reachable=reachable` into the `build_tf_dataset(...)` / `build_torch_dataset(...)`
calls (lines ~1363/1544). Document the param in each docstring (default True =
FK-reachable, matching restructure_assets / bag_info; pass False for
direct-members-only).

- [ ] **Step 4: Run signature test + full no-catalog adapter suite.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/dataset/dataset_bag.py tests/dataset/test_format_b_bag.py
git commit -m "feat(dataset): expose reachable on as_tf_dataset / as_torch_dataset"
```

---

### Task 5: Live regression (demo) + eye-ai acceptance + lint + PR

**Files:**
- Test: `tests/dataset/test_tf_adapter_e2e.py` / `test_torch_adapter_e2e.py` (add subject-partitioned case)

- [ ] **Step 1: Demo-catalog regression test (subject-partitioned)**

Add a live test: build (or use a fixture) dataset whose members are a non-asset
type with assets reachable only via FK (e.g. members = Subject, Images via
Subject→...→Image — mirror whatever the demo catalog supports). Assert:
- `as_tf_dataset(element_type="Image")` (default reachable) yields a NON-empty
  set whose RID set == `resolve_reachable_rows(bag, "Image")` RID set (deduped).
- `restructure_assets(asset_table="Image")` finds the same RID set.
- `as_tf_dataset(..., reachable=False)` reproduces the (empty or direct-only) set.
Mirror for torch.

```python
@pytest.mark.skipif(os.environ.get("DERIVA_HOST") in (None, ""), reason="needs catalog")
def test_as_tf_dataset_finds_fk_reachable_images(catalog_with_datasets):
    ml, _ = catalog_with_datasets
    # ... obtain/build a subject-partitioned dataset bag with FK-reachable Images ...
    from deriva_ml.dataset.target_resolution import resolve_element_rids
    expected = set(resolve_element_rids(bag, "Image", reachable=True))
    assert expected, "fixture must have FK-reachable images"
    yielded = {rid for *_rest, rid in bag.as_tf_dataset(element_type="Image", sample_loader=lambda p, r: p)}
    assert yielded == expected
    # opt-out reproduces direct-members
    direct = set(resolve_element_rids(bag, "Image", reachable=False))
    yielded_direct = {rid for *_rest, rid in bag.as_tf_dataset(element_type="Image", sample_loader=lambda p, r: p, reachable=False)} if direct else set()
    assert yielded_direct == direct
```

- [ ] **Step 2: Run demo regression**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_tf_adapter_e2e.py tests/dataset/test_torch_adapter_e2e.py -q`
Expected: PASS.

- [ ] **Step 3: eye-ai 2-277G acceptance (manual, this session has access — NOT a committed test)**

Run a throwaway script against `www.eye-ai.org`/`eye-ai`, dataset `2-277G`
v4.8.0: `as_tf_dataset(element_type="Image", targets={"Image_Diagnosis":
select_initial_diagnosis}, missing="skip")` → assert it yields ~28,546
`(image, label, rid)` triples (not empty), matching
`_get_reachable_assets(bag, "Image")` / `restructure_assets`. Same for torch.
Record the actual count in the PR. (Do NOT commit a live-eye-ai test — it needs
a token; the demo regression is the committed gate.)

- [ ] **Step 4: Lint + broad suite**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check --fix <touched files> && uv run ruff format <touched files>` (touched files only — do NOT `ruff format tests/dataset/` wholesale).
Then: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_reachable_enumeration.py tests/dataset/test_restructure.py tests/dataset/test_format_b_bag.py tests/dataset/test_tf_adapter_e2e.py tests/dataset/test_torch_adapter_e2e.py tests/dataset/test_tf_adapter_no_tf.py tests/dataset/test_torch_adapter_no_torch.py -q`
Expected: all PASS.

- [ ] **Step 5: Branch, commit, PR**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add -A
git commit -m "test(dataset): live regression — adapters yield FK-reachable elements"
git push -u origin fix/adapter-fk-reachable-elements
gh pr create --title "fix(dataset): tf/torch adapters enumerate FK-reachable elements (match restructure_assets)" --body "<summary + spec link + the shared-core architecture + eye-ai 2-277G acceptance count + opt-out>"
```

---

## Self-Review notes

- **Spec coverage:** T1 shared core+projection (with dedup); T2 restructure onto core (one impl); T3 both adapters wired + `reachable`; T4 public methods; T5 live regression + eye-ai acceptance + PR. All spec sections covered.
- **One impl of reachability:** after T2, `_dataset_table_view`-based enumeration exists in exactly one place (`resolve_reachable_rows`); restructure + both adapters consume it.
- **Dedup placement:** in `resolve_element_rids` (RID iteration), NOT the row core (restructure keeps path-duplicates, collapses by filename). Pinned by `test_resolve_element_rids_reachable_dedups_preserving_order`.
- **Behavior-preserving opt-out:** `reachable=False` reproduces the old `list_dataset_members` set everywhere; pinned in T1 + T5.
- **Default reachable** = the approved choice; no-op for datasets without FK indirection, fixes subject-partitioned datasets.
- **Scope discipline:** lint/format touched files only (avoid the wholesale-format trap from earlier PRs).
