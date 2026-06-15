# Model-built `pathBuilder()` — eliminate `/schema` refetch via deriva-py `from_model` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `DerivaML.pathBuilder()` build the deriva-py path-builder wrapper from the in-memory `Model` deriva-ml already holds (`self.model.model`) — with zero `/schema` GETs — by adding a public `datapath.from_model(catalog, model)` helper to deriva-py. This replaces the buggy "Lever C" identity-cache (which returned stale wrappers after in-place `create_table`) and delivers the O(1)-`/schema` estimate-bag-size win.

**Architecture:** The path-builder wrapper is built entirely from `catalog.getCatalogModel().schemas`; the `/schema` GET in deriva-py's `getPathBuilder()` is pure freshness ceremony (used only as an `is`-identity cache key, never read for content). deriva-ml already holds an up-to-date `Model` (mutated in place by `create_table`, rebound by `refresh_model`). So: (1) deriva-py gains `_CatalogWrapper(catalog, model=None)` + `from_model(catalog, model)` that builds from a supplied model with no fetch; (2) deriva-ml's `pathBuilder()` calls `from_model(self.catalog, self.model.model)`, caching the wrapper keyed on inner-model identity. HTTP (reads AND writes) still routes through `self.catalog` (the wrapper's `_wrapped_catalog`), so writes, joins, snapshot-pinning, and offline behavior are unchanged (verified by a 7-axis blast-radius audit, 2026-06-14).

**Tech Stack:** Python ≥3.12, deriva-py (`datapath`, `_CatalogWrapper`, `Model`), pytest, `uv`, ruff. Two repos: `~/GitHub/deriva-py` (branch `deriva-ml`) and `~/GitHub/DerivaML/deriva-ml` (branch `feature/fast-estimate-bag-size`).

---

## Context: why this replaces Lever C

The shipped "Lever C" (commit on this branch) cached the wrapper keyed on `self.model.model` identity. `create_table`/`create_vocabulary` mutate the deriva-py `Model` **in place** (same object identity), so the cache returned a **stale wrapper missing the new table** → `KeyError: 'SubjectHealth'` in `ensure_features`/`ensure_datasets` fixtures (3 core tests). Root cause confirmed empirically. The fix is NOT smarter invalidation — it's to build from the held model so there is no second cache to go stale, and to keep the wrapper cache keyed on inner-model identity (which rebinds on `refresh_model`). Writes via the wrapper still hit the live catalog because `_wrapped_catalog` is the real `ErmrestCatalog`. See spec §3.1b and the blast-radius audit.

Key source facts (verified):
- `datapath._CatalogWrapper.__init__` (deriva-py `datapath.py:137-151`) builds `self.schemas` from `catalog.getCatalogModel().schemas`; stores `self._wrapped_catalog = catalog`. The schema dict is never used.
- `getPathBuilder()` (deriva-py `ermrest_catalog.py:440-446`) uses `getCatalogSchema()` only as an `is`-identity cache key.
- `getCatalogModel()` is uncached and rebuilds the `Model` from `/schema` every call (`ermrest_catalog.py:379-380` → `Model.fromcatalog` → `getCatalogSchema`).
- deriva-ml's `self.model.model` is the authoritative `Model`: `create_table` mutates it in place; `refresh_model()` rebinds it.

---

## File Structure

**deriva-py** (`~/GitHub/deriva-py`):
- `deriva/core/datapath.py` — `_CatalogWrapper.__init__` gains `model=None`; new `from_model(catalog, model)` public helper.
- `tests/deriva/core/test_datapath.py` — upstream test that `from_model` builds a wrapper with zero `getCatalogModel`/`/schema` calls and that the wrapper is structurally correct.

**deriva-ml** (`~/GitHub/DerivaML/deriva-ml`):
- `src/deriva_ml/core/mixins/path_builder.py` — rewrite `pathBuilder()` to use `datapath.from_model(self.catalog, self.model.model)`, cache keyed on inner-model identity; fix the stale pin/unpin docstring.
- `pyproject.toml` / `uv.lock` — advance the deriva-py pin to the new commit.
- `tests/core/test_catalog_snapshot_schema_reuse.py` — rewrite the two pathbuilder tests (the `getPathBuilder`-counter ones break; switch to `/schema`-GET-count + identity) per the audit.
- `tests/core/test_estimate_schema_fetch_count.py` — tighten the ceiling (now ~0).
- new write-through + snapshot-pinning + offline assertions (audit-mandated guards).

---

## Task 1 (deriva-py): add `from_model` + `model=` param to `_CatalogWrapper`

**Files:**
- Modify: `~/GitHub/deriva-py/deriva/core/datapath.py` (`from_catalog` ~32-38; `_CatalogWrapper.__init__` ~137-151)
- Test: `~/GitHub/deriva-py/tests/deriva/core/test_datapath.py`

- [ ] **Step 1: Confirm the deriva-py checkout branch is clean and on `deriva-ml`**

```bash
cd /Users/carl/GitHub/deriva-py && git status --short && git branch --show-current
```
Expected: clean tree, branch `deriva-ml`. If dirty or on another branch, STOP and report.

- [ ] **Step 2: Write the failing upstream test**

Append to `~/GitHub/deriva-py/tests/deriva/core/test_datapath.py` (match the file's existing import/test style — read the top of the file first to mirror how it builds a catalog/model fixture). Add:

```python
def test_from_model_builds_without_fetching_model():
    """from_model(catalog, model) builds a wrapper from the supplied model,
    never calling catalog.getCatalogModel() (no /schema fetch)."""
    from deriva.core import datapath

    class _FakeColumn:
        def __init__(self, name):
            self.name = name

    class _FakeTable:
        def __init__(self, name):
            self.name = name
            self.column_definitions = [_FakeColumn("RID")]
            self.columns = self.column_definitions

    class _FakeSchema:
        def __init__(self, name, tables):
            self.name = name
            self.tables = {t: _FakeTable(t) for t in tables}

    class _FakeModel:
        def __init__(self):
            self.schemas = {"S": _FakeSchema("S", ["A", "B"])}

    class _FakeCatalog:
        getCatalogModel_calls = 0

        def getCatalogModel(self):
            type(self).getCatalogModel_calls += 1
            raise AssertionError("getCatalogModel must NOT be called by from_model")

    model = _FakeModel()
    cat = _FakeCatalog()
    wrapper = datapath.from_model(cat, model)

    # Built from the supplied model, no getCatalogModel call.
    assert _FakeCatalog.getCatalogModel_calls == 0
    assert set(wrapper.schemas.keys()) == {"S"}
    assert set(wrapper.schemas["S"].tables.keys()) == {"A", "B"}
    # HTTP object is the supplied catalog.
    assert wrapper._wrapped_catalog is cat
    assert wrapper._wrapped_model is model
```

> NOTE: deriva-py's `_SchemaWrapper`/`_TableWrapper` may access more attributes on the table/column objects than the fake provides (e.g. `sname`, `uname`, `_fqname`, key/fkey lists). If the test errors with an `AttributeError` on the fakes during wrapper construction, EXPAND the fakes to provide whatever `_SchemaWrapper.__init__` / `_TableWrapper.__init__` read (read `datapath.py` `_SchemaWrapper`/`_TableWrapper` constructors to see exactly which attributes are touched at build time). Do NOT change the test's intent: `from_model` must build from the supplied model with zero `getCatalogModel` calls. If faithfully faking the model is too brittle, instead build the test against a real `Model` constructed via `Model(catalog_stub, schema_dict)` from a small literal ermrest `/schema` dict — read how `tests/deriva/core/test_datapath.py` already builds its model fixture and reuse that, calling `from_model(catalog, that_model)` and asserting no network. Report which approach you used.

- [ ] **Step 3: Run it to verify it fails**

```bash
cd /Users/carl/GitHub/deriva-py && uv run pytest tests/deriva/core/test_datapath.py::test_from_model_builds_without_fetching_model -q 2>&1 | tail -5 || python -m pytest tests/deriva/core/test_datapath.py::test_from_model_builds_without_fetching_model -q 2>&1 | tail -5
```
Expected: FAIL with `AttributeError: module 'deriva.core.datapath' has no attribute 'from_model'`.
(deriva-py may not use `uv`; if `uv run` fails, fall back to the project's test runner — `python -m pytest` or `pytest` directly. Report which worked.)

- [ ] **Step 4: Add the `model=` param and `from_model` helper**

In `~/GitHub/deriva-py/deriva/core/datapath.py`, change `_CatalogWrapper.__init__` from:

```python
    def __init__(self, catalog):
        """Creates the _CatalogWrapper.

        :param catalog: ErmrestCatalog object
        """
        super(_CatalogWrapper, self).__init__()
        self._wrapped_catalog = catalog
        self._wrapped_model = catalog.getCatalogModel()
```

to:

```python
    def __init__(self, catalog, model=None):
        """Creates the _CatalogWrapper.

        :param catalog: ErmrestCatalog object (used for all HTTP I/O)
        :param model: optional pre-fetched Model to build the wrapper
            from. When None (default), the model is fetched via
            ``catalog.getCatalogModel()`` (the historical behavior).
            When supplied, the model is used as-is and no schema fetch
            occurs — the caller is asserting it holds an up-to-date
            model for this catalog. HTTP still routes through
            ``catalog``.
        """
        super(_CatalogWrapper, self).__init__()
        self._wrapped_catalog = catalog
        self._wrapped_model = model if model is not None else catalog.getCatalogModel()
```

Then add the public helper next to `from_catalog` (after line ~38):

```python
def from_model(catalog, model):
    """Wraps a catalog for datapath expressions using a supplied Model.

    Identical to :func:`from_catalog` except the schema structure is
    taken from ``model`` instead of being fetched via
    ``catalog.getCatalogModel()``. Use when the caller already holds an
    up-to-date :class:`~deriva.core.ermrest_model.Model` for ``catalog``
    and wants to avoid a redundant ``/schema`` fetch. All HTTP (reads
    and writes) still routes through ``catalog``.

    :param catalog: an ErmrestCatalog object (used for HTTP I/O)
    :param model: a Model object whose ``.schemas`` define the wrapper
    :return: a datapath._CatalogWrapper object
    """
    return _CatalogWrapper(catalog, model=model)
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
cd /Users/carl/GitHub/deriva-py && uv run pytest tests/deriva/core/test_datapath.py::test_from_model_builds_without_fetching_model -q 2>&1 | tail -5 || python -m pytest tests/deriva/core/test_datapath.py::test_from_model_builds_without_fetching_model -q 2>&1 | tail -5
```
Expected: PASS.

- [ ] **Step 6: Run the existing datapath tests — `from_catalog` must be unaffected (model=None default)**

```bash
cd /Users/carl/GitHub/deriva-py && uv run pytest tests/deriva/core/test_datapath.py -q 2>&1 | tail -8 || python -m pytest tests/deriva/core/test_datapath.py -q 2>&1 | tail -8
```
Expected: all pass (the `model=None` default preserves `from_catalog`'s behavior exactly). If the suite needs a live catalog and can't run here, run at least the collection + any no-catalog tests, and note that the full datapath suite requires a server.

- [ ] **Step 7: Commit + push to the `deriva-ml` branch**

```bash
cd /Users/carl/GitHub/deriva-py && git add deriva/core/datapath.py tests/deriva/core/test_datapath.py && git commit -m "$(cat <<'EOF'
feat(datapath): from_model(catalog, model) builds wrapper without /schema fetch

_CatalogWrapper already builds its schemas from catalog.getCatalogModel();
the schema dict is never read for content. Add an optional model= param and
a from_model() helper so callers holding an up-to-date Model can build a
path-builder without the redundant /schema GET that getCatalogModel() does
on every call. HTTP still routes through the supplied catalog, so reads and
writes are unchanged. from_catalog() is unaffected (model=None default).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)" && git push origin deriva-ml 2>&1 | tail -3
```
Record the new deriva-py commit SHA — Task 2 needs it for the pin bump.

---

## Task 2 (deriva-ml): advance the deriva-py pin

**Files:**
- Modify: `~/GitHub/DerivaML/deriva-ml/uv.lock` (via `uv lock`)

- [ ] **Step 1: Advance the pinned deriva-py commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && uv lock --upgrade-package deriva 2>&1 | tail -5
```
Expected: `uv.lock` updates the `deriva` git pin to the new `deriva-ml`-branch commit (the one from Task 1 Step 7).

- [ ] **Step 2: Sync and confirm `from_model` is importable**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && uv sync 2>&1 | tail -3 && uv run python -c "from deriva.core import datapath; print('from_model' , hasattr(datapath, 'from_model'))"
```
Expected: prints `from_model True`.

- [ ] **Step 3: Commit the pin bump**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add uv.lock && git commit -m "$(cat <<'EOF'
chore(deps): bump deriva-py pin for datapath.from_model

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 (deriva-ml): rewrite `pathBuilder()` to build from the held model

**Files:**
- Modify: `src/deriva_ml/core/mixins/path_builder.py` (`pathBuilder`, ~59-95)

This REPLACES the buggy Lever C cache. The new cache is still keyed on inner-model identity (correct for `refresh_model` rebind), but the wrapper is now built from the held model via `from_model` (no `/schema` fetch, and it always reflects in-place `create_table` mutations because it reads the current `self.model.model.schemas` at build time).

- [ ] **Step 1: Read the current `pathBuilder` to get the exact current body**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && sed -n '59,96p' src/deriva_ml/core/mixins/path_builder.py
```
Note the current Lever C body (the `inner_model = self.model.model` / `_path_builder_cache` block from the prior task) — you will replace it.

- [ ] **Step 2: Rewrite `pathBuilder`**

Replace the entire body of the `pathBuilder` method (everything after the docstring, i.e. the current `inner_model = ...` cache block OR the original `return self.catalog.getPathBuilder()`, whichever is present) with:

```python
        # Build the path-builder wrapper from the model deriva-ml
        # already holds (self.model.model) instead of letting deriva-py
        # re-fetch /schema on every call. The wrapper's schema structure
        # comes entirely from the model; HTTP (reads AND writes) still
        # routes through self.catalog (the wrapper's _wrapped_catalog),
        # so writes, joins, and snapshot-pinning are unchanged.
        #
        # Why this is correct under schema changes: self.model.model is
        # the authoritative in-memory Model. create_table mutates it in
        # place (a wrapper built afterward sees the new table), and
        # refresh_model()/refresh_schema() REBIND it to a new object.
        # We cache the wrapper keyed on the inner-model object identity,
        # so a rebind (refresh) invalidates the cache, while an in-place
        # mutation is reflected because we rebuild from the *current*
        # self.model.model whenever the cache misses. The previous
        # approach cached deriva-py's getPathBuilder() result and went
        # stale after an in-place create_table; building from the held
        # model via datapath.from_model avoids both the staleness and
        # the redundant /schema fetch. See the estimate-bag-size perf
        # spec (Lever C, revised: model-built pathBuilder).
        inner_model = self.model.model
        cached = getattr(self, "_path_builder_cache", None)
        if cached is not None and cached[0] is inner_model:
            return cached[1]
        wrapper = datapath.from_model(self.catalog, inner_model)
        self._path_builder_cache = (inner_model, wrapper)
        return wrapper
```

> IMPORTANT subtlety the cache does NOT cover: an in-place `create_table` followed by a `pathBuilder()` call WITHOUT a `refresh_model` returns the OLD cached wrapper (same `inner_model` identity), which would miss the new table. This matches the SHIPPED Lever C behavior and deriva-ml's documented "model is a snapshot until refresh" contract — the blast-radius audit (§6) verified NO deriva-ml flow does create-then-read-via-pathBuilder without an intervening refresh. The `SubjectHealth` fixture bug was NOT this case — it was the wrapper-snapshot staleness, which building-from-current-model on each cache-miss fixes (verified in Step 5's regression run). Do NOT add extra invalidation for the in-place case unless Step 5 surfaces a real failure; if it does, the correct fix is a `refresh_model()` at the offending mutation site, reported as a finding.

- [ ] **Step 3: Fix the stale pin/unpin docstring (audit §3 nit)**

In the same `pathBuilder` docstring region (the comment block that lists what invalidates the cache), if it claims `pin_schema()`/`unpin_schema()` "rebind the model," correct it: pin/unpin only touch the on-disk SchemaCache pin file and do NOT rebind `self.model`; they don't change schema content so the cached wrapper stays correct. (If your Step-2 replacement already omits that claim, skip this step. Read the surrounding docstring and ensure no false invalidation guarantee remains.)

- [ ] **Step 4: Confirm `datapath` is imported in path_builder.py**

`path_builder.py` imports `datapath` via `importlib.import_module("deriva.core.datapath")` at the top (line ~14). Confirm `datapath.from_model` resolves:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -c "import importlib; dp=importlib.import_module('deriva.core.datapath'); print(hasattr(dp,'from_model'))"
```
Expected: `True`.

- [ ] **Step 5: Run the full core suite — this is the regression gate (the SubjectHealth bug MUST be gone)**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/ -q 2>&1 | tail -6
```
Expected: the 3 previously-failing tests (`test_resolve_rid_single`, `test_resolve_rids_batch`, `test_resolve_rids_with_candidate_tables`, and the estimate guard) NO LONGER error with `KeyError: 'SubjectHealth'`. The two `test_live_pathbuilder_*` tests from the old Lever C WILL now fail (they count `getPathBuilder` calls, which we no longer make) — that is EXPECTED and fixed in Task 4. Note exactly which tests pass/fail. If any `SubjectHealth` KeyError remains, STOP and report (the model-built wrapper should see in-place creates).

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/core/mixins/path_builder.py && git commit -m "$(cat <<'EOF'
perf(core): build pathBuilder from held model via datapath.from_model

Replaces the buggy Lever C cache (which cached deriva-py's getPathBuilder
result and went stale after an in-place create_table -> KeyError on the new
table). Now builds the wrapper from self.model.model via the new
datapath.from_model helper: zero /schema fetches, and the wrapper reflects
in-place model mutations because it rebuilds from the current model on a
cache miss. HTTP still routes through self.catalog, so writes/joins/snapshot
pinning are unchanged. Cache keyed on inner-model identity (rebinds on
refresh_model/refresh_schema).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 (deriva-ml): rewrite the pathbuilder tests (audit-mandated)

**Files:**
- Modify: `tests/core/test_catalog_snapshot_schema_reuse.py` (the two `test_live_pathbuilder_*` tests)

The old tests counted `ErmrestCatalog.getPathBuilder` calls. The fix no longer calls `getPathBuilder`, so they break/become vacuous. Rewrite them to assert the REAL new guarantees: zero `/schema` GETs, and correct invalidation on `refresh_model`.

- [ ] **Step 1: Replace the two `test_live_pathbuilder_*` tests**

In `tests/core/test_catalog_snapshot_schema_reuse.py`, find `test_live_pathbuilder_cached_across_calls` and `test_live_pathbuilder_invalidated_on_refresh` and REPLACE both with:

```python
def _count_schema_gets(monkeypatch) -> dict:
    """Patch DerivaBinding.get to count /schema requests."""
    import deriva.core.deriva_binding as db

    counter = {"schema": 0}
    orig = db.DerivaBinding.get

    def spy(self, path, *a, **k):
        if isinstance(path, str) and path.split("?")[0].endswith("/schema"):
            counter["schema"] += 1
        return orig(self, path, *a, **k)

    monkeypatch.setattr(db.DerivaBinding, "get", spy)
    return counter


def test_live_pathbuilder_no_schema_fetch(live_ml, monkeypatch):
    """ml.pathBuilder() builds from the held model with zero /schema GETs."""
    counter = _count_schema_gets(monkeypatch)
    live_ml.pathBuilder()
    live_ml.pathBuilder()
    live_ml.pathBuilder()
    assert counter["schema"] == 0, (
        f"pathBuilder() issued {counter['schema']} /schema GETs; expected 0 "
        "(wrapper is built from the in-memory model)"
    )


def test_live_pathbuilder_cached_identity(live_ml):
    """Repeated pathBuilder() returns the same wrapper until the model rebinds."""
    pb1 = live_ml.pathBuilder()
    pb2 = live_ml.pathBuilder()
    assert pb1 is pb2  # cached on inner-model identity
    live_ml.model.refresh_model()  # rebinds inner model -> cache invalidates
    pb3 = live_ml.pathBuilder()
    assert pb3 is not pb1  # rebuilt after refresh


def test_live_pathbuilder_sees_in_place_create(live_ml):
    """A pathBuilder built after create_vocabulary sees the new table."""
    import uuid

    name = "PbVocab" + uuid.uuid4().hex[:6].upper()
    live_ml.create_vocabulary(name, "pathbuilder freshness test")
    pb = live_ml.pathBuilder()
    # The new vocab table is visible through the freshly-built wrapper.
    assert name in pb.schemas[live_ml.default_schema].tables
```

> NOTE on `test_live_pathbuilder_sees_in_place_create`: this works because `create_vocabulary` is a NEW pathBuilder() call after the mutation AND the cache key is the inner-model identity. If `create_vocabulary` does NOT rebind the model (in-place mutation, same identity) AND a prior test cached a wrapper for that identity, the cache could return a stale wrapper. To make this test robust and honest, call `live_ml.pathBuilder()` is invoked AFTER create. If it fails (stale cache hit on same identity), that is the real in-place-staleness edge the audit flagged — in that case the correct fix is for `create_vocabulary`/`create_table` to invalidate `self._path_builder_cache` (set it to None) right after mutating the model. If Step 2 shows this test failing, apply that one-line invalidation in `DerivaModel.create_table` / the DerivaML create methods and note it as a finding; otherwise leave it.

- [ ] **Step 2: Run the rewritten tests**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/test_catalog_snapshot_schema_reuse.py -q 2>&1 | tail -6
```
Expected: all pass. If `test_live_pathbuilder_sees_in_place_create` fails, follow the NOTE above (add cache invalidation at the create choke point) and report it.

- [ ] **Step 3: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add tests/core/test_catalog_snapshot_schema_reuse.py src/deriva_ml/core/mixins/path_builder.py src/deriva_ml/model/catalog.py 2>/dev/null; git commit -m "$(cat <<'EOF'
test(core): rewrite pathBuilder tests for model-built wrapper (0 /schema GETs)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 (deriva-ml): write-through + snapshot-pinning guards (audit-mandated)

**Files:**
- Modify: `tests/core/test_catalog_snapshot_schema_reuse.py` (append)

These lock in the two highest-value audit guarantees: (1) writes through a model-built wrapper land in the live catalog; (2) a snapshot instance's model-built wrapper reads from the `@snaptime` endpoint.

- [ ] **Step 1: Add the write-through and snapshot guards**

Append to `tests/core/test_catalog_snapshot_schema_reuse.py`:

```python
def test_model_built_wrapper_writes_reach_catalog(live_ml):
    """An insert via the model-built pathBuilder lands a real row."""
    import uuid

    # Use a vocabulary table (created fresh) to avoid disturbing fixtures.
    vname = "WriteVocab" + uuid.uuid4().hex[:6].upper()
    live_ml.create_vocabulary(vname, "write-through test")
    term = "term_" + uuid.uuid4().hex[:6]
    # add_term goes through pathBuilder().schemas[...].tables[...].insert(...)
    live_ml.add_term(vname, term, description="x")
    # Read it back through a fresh pathBuilder.
    pb = live_ml.pathBuilder()
    rows = list(pb.schemas[live_ml.default_schema].tables[vname].entities().fetch())
    names = {r.get("Name") for r in rows}
    assert term in names, f"inserted term {term!r} not found in {names}"


def test_snapshot_pathbuilder_reads_are_snapshot_pinned(live_ml):
    """A snapshot instance's pathBuilder routes data reads to the @snaptime URI."""
    raw = live_ml.catalog.get("/").json()["snaptime"]
    compound = f"{live_ml.catalog_id}@{raw}"
    snap = live_ml.catalog_snapshot(compound)
    pb = snap.pathBuilder()
    # The wrapped catalog is the snapshot, so its base URI carries @snaptime.
    base = pb._wrapped_catalog._server_uri
    assert "@" in base, f"snapshot pathBuilder base URI not pinned: {base}"
```

> NOTE: `pb._wrapped_catalog._server_uri` reaches a deriva-py internal — acceptable in a test that pins exactly this invariant. If `_server_uri` isn't the right attribute name in the installed deriva-py, inspect `pb._wrapped_catalog` for the attribute that carries the catalog base URL (it's what `DataPath._base_uri` reads) and assert `@` (the snaptime separator) is present. The intent: prove the snapshot wrapper's HTTP target is the pinned snapshot, not live.

- [ ] **Step 2: Run them**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/test_catalog_snapshot_schema_reuse.py -k "writes_reach or snapshot_pathbuilder" -q 2>&1 | tail -6
```
Expected: both pass.

- [ ] **Step 3: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add tests/core/test_catalog_snapshot_schema_reuse.py && git commit -m "$(cat <<'EOF'
test(core): guard writes + snapshot-pinning through model-built pathBuilder

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6 (deriva-ml): tighten the estimate /schema-count guard

**Files:**
- Modify: `tests/core/test_estimate_schema_fetch_count.py`

- [ ] **Step 1: Run the existing guard to get the new count**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/test_estimate_schema_fetch_count.py -q 2>&1 | tail -6
```
Expected: PASS (previously failed at 76; now the snapshot path adds 0 and the live-instance refetch is gone, so the count should be a small constant — likely 0–2 from the one construction-time fetch). If it SKIPs (demo dataset not nested), note it.

- [ ] **Step 2: Tighten the ceiling to lock in the win**

If Step 1 shows the count is now ≤1 (or whatever small constant), lower the assertion ceiling in `test_estimate_schema_fetch_count.py` from `<= 3` to the observed value + a tiny margin (e.g. `<= 1`), and update the message to reflect "model-built pathBuilder issues 0 schema GETs in the nesting walk." Use the actual observed number; do not guess. If the observed count is, say, 1, set `<= 1`.

- [ ] **Step 3: Re-run + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/test_estimate_schema_fetch_count.py -q 2>&1 | tail -3 && git add tests/core/test_estimate_schema_fetch_count.py && git commit -m "$(cat <<'EOF'
test(core): tighten estimate /schema-count ceiling after model-built pathBuilder

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7 (deriva-ml): offline-mode guard + full regression sweep

**Files:**
- Modify: `tests/core/test_catalog_snapshot_schema_reuse.py` or the existing offline test (append a small assertion)

- [ ] **Step 1: Confirm offline behavior is unchanged**

The audit found `pathBuilder()` already raises `DerivaMLReadOnlyError` offline (CatalogStub). Verify the model-built path preserves this — `from_model(self.catalog, self.model.model)` where `self.catalog` is a `CatalogStub`: building the wrapper itself doesn't call catalog methods (model is supplied), but any subsequent `.fetch()`/`.insert()` routes through the stub and raises. Confirm the existing offline tests still pass:

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_catalog_stub.py tests/core/test_offline_mode_smoke.py -q 2>&1 | tail -5
```
Expected: pass. If `pathBuilder()` offline now SUCCEEDS where it previously raised (because building from the supplied model no longer calls `getCatalogModel`), decide: is a usable offline pathBuilder a regression or an improvement? It's likely benign (reads/writes through the stub still raise), but if a test explicitly asserted `pathBuilder()` raises offline, that test's intent changed — report it and discuss rather than silently editing.

- [ ] **Step 2: Full core sweep — clean**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/ -q 2>&1 | tail -6
```
Expected: ALL pass (0 failures, 0 errors). This is the gate that the SubjectHealth regression is gone and nothing else broke.

- [ ] **Step 3: Broader sweep — dataset + execution paths that use pathBuilder heavily**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_datasets.py tests/dataset/test_estimate_bag_size.py tests/feature/ tests/execution/test_executions.py -q --maxfail=5 2>&1 | tail -8
```
Expected: pass (these exercise writes + reads through pathBuilder extensively — the real proof writes still work). Investigate any failure per systematic-debugging; do NOT proceed to the PR with failures.

- [ ] **Step 4: Commit any offline assertion added**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git diff --quiet || (git add -A && git commit -m "test(core): assert offline pathBuilder behavior unchanged")
```

---

## Task 8: live 2-277G re-measure + PR (both repos)

**Files:** none (measurement + PRs)

- [ ] **Step 1: Live re-measure (manual, record numbers)**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run python - <<'PY'
import time
from deriva_ml import DerivaML
from deriva_ml.dataset import DatasetSpec
ml = DerivaML(hostname="www.eye-ai.org", catalog_id="eye-ai")
t0 = time.time()
est = ml.estimate_bag_size(DatasetSpec(rid="2-277G", version="4.11.0"))
print(f"estimate_bag_size(2-277G): {time.time()-t0:.1f}s  (was ~510s)")
print("rows:", est["total_rows"], "size:", est["total_asset_size"],
      "tables:", len(est["tables"]), "incomplete:", est["incomplete"])
PY
```
Expected: dramatically faster (target the walk's network cost drops from ~2,375 GETs / ~510s toward seconds — the 849 `/schema` GETs are eliminated; the remaining ~170 `Dataset_Dataset`/`Dataset` GETs are the floor unless Lever B is added). Totals MUST match the recorded baseline: 360756 rows / 18.0 GB / 80 tables / incomplete False. Record actual time + totals.

- [ ] **Step 2: Decide on Lever B**

If Step 1 is < 30s → done; Lever B (tree-walk dedup) not needed. If still too slow (the ~170 descendant GETs dominate), note that Lever B is the remaining optimization and decide with the user whether to pursue it in this PR or defer.

- [ ] **Step 3: Open the deriva-py PR**

```bash
cd /Users/carl/GitHub/deriva-py && gh pr create --base master --title "feat(datapath): from_model() builds path-builder from a supplied Model (no /schema fetch)" --body "$(cat <<'EOF'
## Summary
`_CatalogWrapper` already builds its `schemas` from `catalog.getCatalogModel()`
— the raw `/schema` dict is never read for content. But `getCatalogModel()` is
uncached and re-fetches `/schema` on every call, so every `getPathBuilder()`
pays a full schema fetch even when the caller already holds an up-to-date Model
(and on servers that don't return 304, the binding cache never dedups it).

Adds an optional `model=` param to `_CatalogWrapper.__init__` and a public
`datapath.from_model(catalog, model)` helper that builds the wrapper from a
supplied Model with zero schema fetches. HTTP (reads and writes) still routes
through the supplied catalog, so behavior is otherwise identical.
`from_catalog()` is unchanged (`model=None` default).

Consumer: deriva-ml's estimate_bag_size walks 80+ tables across 80+ nested
datasets, each triggering pathBuilder()/lookup_dataset() — ~849 redundant
/schema GETs (~8.5 min on a large dataset). With from_model it holds its own
up-to-date Model and avoids them entirely.

## Test
`tests/deriva/core/test_datapath.py::test_from_model_builds_without_fetching_model`
asserts the wrapper is built from the supplied model with zero getCatalogModel
calls, wraps the supplied catalog for HTTP, and exposes the model's schemas.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)" 2>&1 | tail -3
```
> NOTE: confirm deriva-py's default branch name (`master` vs `main`) before running — adjust `--base`. If deriva-py uses a different contribution flow (not the `deriva-ml` working branch), ask the user how they want the upstream change landed; the `deriva-ml`-branch commit already lets deriva-ml consume it via the pin.

- [ ] **Step 4: Open/Update the deriva-ml PR**

If a PR for `feature/fast-estimate-bag-size` doesn't exist yet, create it; else it updates automatically on push. Push first:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git push -u origin feature/fast-estimate-bag-size 2>&1 | tail -3
```
Then create (or note the existing) PR with a body summarizing: A1/A2 (snapshot schema reuse), the model-built pathBuilder (replacing the reverted Lever C identity-cache), the deriva-py from_model dependency (link the deriva-py PR), the SubjectHealth regression fix, the 7-axis blast-radius audit, and the live 2-277G before/after numbers from Step 1.

- [ ] **Step 5: Report both PR URLs + the re-measure numbers.** The deriva-ml `bump-version patch` happens after merge on clean `main`; the deriva-py change lands first (deriva-ml's pin already points at the branch commit).

---

## Self-Review notes

- **Spec coverage:** §3.1a (correction: non-304 servers) → motivates the whole plan; §3.1b Lever C (revised to model-built) → Tasks 1–4; the "eliminate, don't memoize" principle → from_model (no /schema); §5.1c invalidation-on-refresh → Task 4 `test_live_pathbuilder_cached_identity`; §5.2 GET-count guard → Task 6; the blast-radius audit's 6 mandated tests → Tasks 4 (rewrite + identity + in-place-create), 5 (write-through + snapshot), 7 (offline); live re-measure → Task 8.
- **Type/signature consistency:** `from_model(catalog, model)` signature identical in deriva-py def (Task 1), deriva-ml call (Task 3), and the upstream test (Task 1). `_path_builder_cache = (inner_model, wrapper)` shape consistent between Task 3 impl and Task 4 identity test. Cache key is `self.model.model` (inner Model) throughout.
- **Cross-repo ordering:** Task 1 (deriva-py) MUST land/push before Task 2 (pin bump) before Task 3+ (deriva-ml uses from_model). The plan enforces this order.
- **The reverted Lever C:** Task 3 replaces the Lever C body; the old `test_live_pathbuilder_*` tests are replaced in Task 4. The prior Lever C commit stays in history (replaced, not git-reverted) — its net effect is fully superseded by Task 3. No separate revert commit needed since Task 3 overwrites the method body.
- **No placeholders:** every code step shows full code; the one judgment branch (in-place-create invalidation) has an explicit condition + the exact one-line fix to apply if the test fails.
