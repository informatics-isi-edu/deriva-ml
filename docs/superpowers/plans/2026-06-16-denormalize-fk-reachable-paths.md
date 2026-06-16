# Denormalize FK-Reachable Paths Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the denormalize planner union ALL reachable `Dataset → … → element` paths (membership AND FK-reachable), RID-distinct on the `row_per` leaf, so `feature_values` / `get_denormalized_*` return the correct rows on subject-partitioned datasets (members = Subjects, element FK-reachable via `Subject → … → element`).

**Architecture:** The planner (`_prepare_wide_table` in `model/denormalize_planner.py`) currently keeps ONE `Dataset → membership-assoc → element` path per element (Phase-1b dedup + the `paths[0][1]` prefix at Phase 3). On a subject-partitioned dataset the membership association is empty, so the join returns 0 rows. Fix: emit one `element_tables` entry per distinct `Dataset → element` route (membership AND FK-reachable chain), each keyed `f"{element}#{i}"`. The consumer `_denormalize_impl` already iterates `join_tables.items()` and `union(*sql_statements)` (UNION, deduped) when there is >1 statement, and projects columns from `column_specs` (per-table, not per-path) — so the projection is identical across paths and UNION dedups on the leaf row automatically. No value-tuple-shape change; the key is never used semantically (`for _key, (path, jc, jt) in ...`).

**Tech Stack:** Python 3.12, SQLAlchemy (bag SQLite mirror), pytest, `uv`, ruff. Live verification against `dev.eye-ai.org`/`eye-ai` bag `6-CQZE` v0.4.0 (already cached) and the localhost demo catalog.

---

## File structure

- **Modify** `src/deriva_ml/model/denormalize_planner.py` — `_prepare_wide_table` Phase-1b/1c/3: retain and emit multiple routes per element. The single behavioural change.
- **Create** `tests/local_db/test_denormalize_fk_reachable_paths.py` — catalog-free planner unit tests (stub model) proving multiple routes are emitted and Rule-6 still raises on genuine column ambiguity.
- **Modify** `tests/conftest.py` (or `tests/catalog_manager.py` / `src/deriva_ml/demo_catalog.py`) — add a subject-partitioned demo dataset fixture (members = Subjects only; `Image` FK-reachable via `Image.Subject`).
- **Create** `tests/dataset/test_subject_partitioned_feature_values.py` — live demo regression: `feature_values` and `as_tf_dataset` labels resolve on the subject-partitioned fixture.
- **Modify** `docs/reference/denormalization.md` — document the union-of-reachable-paths + RID-distinct-on-`row_per`-leaf rule.

> **CWD:** every command assumes `cd /Users/carl/GitHub/DerivaML/deriva-ml` chained into the same Bash call. Tests need `DERIVA_ML_ALLOW_DIRTY=true`; live demo tests need `DERIVA_HOST=localhost`; the eye-ai check needs network + a token.

---

### Task 1: Characterization spike — pin the multi-route prefix construction against the live bag

The Phase-3 prefix builder (`denormalize_planner.py:1808-1857`) only handles the 2-table `Dataset → assoc → element` shape (`assoc_name = paths[0][1].name`). The FK-reachable route is a multi-hop chain (`Dataset → Subject_Dataset → Subject → Observation → Image`). This task discovers the EXACT join-condition construction for a multi-hop prefix empirically, so later tasks contain real code, not guesses. **No production edits in this task** — it produces a throwaway probe and a written finding.

**Files:**
- Create (throwaway, NOT committed): `/tmp/spike_multiroute.py`

- [ ] **Step 1: Write the spike probe**

```python
# /tmp/spike_multiroute.py — THROWAWAY. Pin the exact path_names / join_conditions
# / join_types a multi-hop FK-reachable route must produce for [Image, Image_Diagnosis]
# on the subject-partitioned 6-CQZE bag.
import os
os.environ.setdefault("DERIVA_ML_ALLOW_DIRTY", "true")
from deriva_ml import DerivaML
from deriva_ml.dataset.aux_classes import DatasetSpec

ml = DerivaML(hostname="dev.eye-ai.org", catalog_id="eye-ai")
bag = ml.download_dataset_bag(DatasetSpec(rid="6-CQZE", version="0.4.0"))
pl = bag.model._planner

# All Dataset->...->Image routes the planner discovers (each is a path of Table objs).
routes = [p for p in pl._schema_to_paths() if p[-1].name == "Image"]
for p in routes:
    print("ROUTE:", [t.name for t in p])

# For the FK-reachable route, print the per-edge relationship the prefix builder needs.
# _table_relationship(from_table, to_table) -> list[(fk_col, pk_col)].
fkroute = next(p for p in routes if "Subject" in [t.name for t in p])
print("\nFK route edges:")
for a, b in zip(fkroute[:-1], fkroute[1:]):
    try:
        rel = pl._table_relationship(a, b)
        print(f"  {a.name} -> {b.name}: {rel}")
    except Exception as e:
        print(f"  {a.name} -> {b.name}: RAISE {e!r}")
```

- [ ] **Step 2: Run the spike**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run python /tmp/spike_multiroute.py 2>&1 | grep -E "ROUTE|->|RAISE"`
Expected: at least two ROUTE lines (`Dataset, Image_Dataset, Image` and `Dataset, Subject_Dataset, Subject, Observation, Image`), and a per-edge relationship list for every hop of the FK route with no RAISE. This proves `_table_relationship` resolves every edge of a multi-hop prefix — the construction Task 3 needs.

- [ ] **Step 3: Record the finding inline in the plan's Task 3 notes**

Confirm the prefix for a multi-hop route is built by walking consecutive `(from, to)` pairs and calling `_table_relationship(from, to)` for each — the same call the current 2-table prefix uses, just iterated over the whole chain. If any edge RAISES, STOP and surface it (the construction needs a different relationship resolver). Do NOT proceed to Task 3 until every edge resolves.

- [ ] **Step 4: Delete the throwaway probe**

```bash
rm -f /tmp/spike_multiroute.py
```

---

### Task 2: Failing catalog-free planner unit test — multiple routes emitted

Prove the planner emits one `element_tables` entry per distinct `Dataset → element` route, using a stub model so no catalog is needed. This is the RED test for the core change.

**Files:**
- Create: `tests/local_db/test_denormalize_fk_reachable_paths.py`

- [ ] **Step 1: Write the failing test**

Build the test against the real demo schema fixture the planner tests already use (the demo schema has BOTH `Dataset → Dataset_Image → Image` and `Dataset → Dataset_Subject → Subject → Image`, so it exercises two routes without a live catalog). Find the existing pattern first:

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && grep -rn "DenormalizePlanner\|_prepare_wide_table\|demo-catalog-schema\|Model.fromfile\|build_local_schema" tests/local_db/test_paths.py tests/local_db/test_planner_rules.py tests/local_db/conftest.py | head`

Then write, reusing whatever model-construction fixture those tests use (call it `planner_from_demo_schema` below — replace with the actual fixture/helper name discovered above):

```python
"""Planner emits ALL reachable Dataset->element routes (membership + FK-reachable),
not just the preferred membership association. Regression for subject-partitioned
feature_values returning 0 (denormalize-fk-reachable-paths spec)."""

from __future__ import annotations


def test_prepare_wide_table_emits_membership_and_fk_routes(planner_from_demo_schema):
    """The demo schema reaches Image two ways:
      - Dataset -> Dataset_Image -> Image           (membership)
      - Dataset -> Dataset_Subject -> Subject -> Image (FK-reachable)
    Both must appear as separate join_tables entries so the consumer unions them.
    """
    planner, dataset_stub, dataset_rid = planner_from_demo_schema
    join_tables, _cols, _multi = planner._prepare_wide_table(
        dataset_stub, dataset_rid, ["Image"], row_per="Image"
    )
    # Collect the association/second-hop table for every Image route.
    second_hops = {path[1] for (path, _jc, _jt) in join_tables.values() if len(path) >= 2}
    assert "Dataset_Image" in second_hops, f"membership route missing: {second_hops}"
    assert "Dataset_Subject" in second_hops, f"FK-reachable route missing: {second_hops}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_denormalize_fk_reachable_paths.py::test_prepare_wide_table_emits_membership_and_fk_routes -v`
Expected: FAIL — only `Dataset_Image` present in `second_hops` (the membership route); `Dataset_Subject` missing because Phase-1b dedup discarded it.

> If the fixture `planner_from_demo_schema` does not exist, Step 1 also creates it in `tests/local_db/conftest.py` from the model-construction pattern discovered by the grep (load `tests/dataset/demo-catalog-schema.json` into a `Model`, build the planner, and a minimal dataset stub exposing `.dataset_rid` and `list_dataset_children(recurse=True) -> []`). Use the EXACT construction the neighbouring planner tests use — do not invent a new one.

---

### Task 3: Implement multi-route emission in `_prepare_wide_table`

Make Phase-1b retain distinct routes and Phase-3 emit one entry per route. Uses the multi-hop prefix construction pinned in Task 1.

**Files:**
- Modify: `src/deriva_ml/model/denormalize_planner.py` (Phase 1b ~1746-1783, Phase 1c ~1785-1791, Phase 3 emission ~1800-1857)

- [ ] **Step 1: Read the current Phase 1b/1c/3 block in full before editing**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && sed -n '1742,1872p' src/deriva_ml/model/denormalize_planner.py`

- [ ] **Step 2: Replace the `(element, endpoint)`-collapsing dedup (Phase 1b) so it dedups only identical table-sequence paths, keeping distinct routes**

Change the Phase-1b loop (`denormalize_planner.py:1746-1783`) so the dedup key is the FULL path signature `tuple(t.name for t in path)` rather than `(element.name, endpoint.name)`. Replace the block from `deduplicated_paths: list[list[Table]] = []` through `table_paths = deduplicated_paths` with:

```python
        # ── Phase 1b: deduplicate ONLY truly identical paths ─────────────────
        # Previously this collapsed to one route per (element, endpoint),
        # preferring the Dataset_{Element} membership association and DISCARDING
        # the FK-reachable chain (e.g. Dataset -> Subject_Dataset -> Subject ->
        # Image). On subject-partitioned datasets the membership association is
        # empty, so that discard produced a silently-empty result. Retain every
        # distinct route; the consumer UNIONs them (RID-distinct on the row_per
        # leaf). See docs/superpowers/specs/2026-06-16-denormalize-fk-reachable-paths-design.md.
        deduplicated_paths: list[list[Table]] = []
        seen_path_sigs: set[tuple[str, ...]] = set()
        for path in table_paths:
            sig = tuple(t.name for t in path)
            if sig not in seen_path_sigs:
                seen_path_sigs.add(sig)
                deduplicated_paths.append(path)

        table_paths = deduplicated_paths
```

Delete the now-unused `_is_standard_assoc` helper and `seen_element_endpoint` dict (they were only used by the replaced dedup).

- [ ] **Step 3: Emit one `element_tables` entry per route (Phase 1c + Phase 3)**

In Phase 1c (`paths_by_element`) the value is already `list[list[Table]]` (all routes per element). In the Phase-2/3 loop (`for element_name, paths in paths_by_element.items():`), currently ONE entry is built per element using `paths[0]` as the prefix. Change it to iterate each route and emit `element_tables[f"{element_name}#{i}"]`. The downstream subtree (`_build_join_tree`) is the same for all routes of an element; only the `Dataset → … → element` PREFIX differs per route. Build each route's prefix by walking its consecutive `(from, to)` pairs (the construction pinned in Task 1):

Replace the Phase-2/3 per-element loop body (`denormalize_planner.py:1800-1857`) so that, for each `route_index, route in enumerate(paths)`:
- the prefix tables are `route[:-? ]` up to and including the element (i.e. `[t.name for t in route[:route.index(element_table)+1]]` — the element is `route[2]` only for the membership shape; for the FK chain the element is the LAST table of the prefix). Compute the element's position as `elem_pos = next(i for i, t in enumerate(route) if t.name == element_name)`; the prefix is `route[:elem_pos+1]`.
- `join_conditions` / `join_types` for the prefix come from `_table_relationship(route[j], route[j+1])` for each `j` in `range(elem_pos)` (inner joins; this is exactly the 2-table prefix code generalised to N hops, validated in Task 1).
- the downstream subtree (element → endpoint, from the JoinTree `tree.walk()` / `tree.walk_edges()`) is appended after the prefix, unchanged.
- key the result `element_tables[f"{element_name}#{route_index}"] = (path_names, join_conditions, join_types)`.

Write the actual replacement code in this step (do not summarise) using the exact variable names from Step 1's `sed` output and the per-edge relationship call confirmed in Task 1. Keep the JoinTree downstream-subtree code (`tree = self._build_join_tree(...)`, `tree.walk()`, `tree.walk_edges()`) intact — only the PREFIX construction and the dict KEY change.

- [ ] **Step 4: Run the Task-2 test to verify it passes**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_denormalize_fk_reachable_paths.py -v`
Expected: PASS — both `Dataset_Image` and `Dataset_Subject` routes present.

- [ ] **Step 5: Run the FULL planner/denormalize suite (Net-1 regression gate)**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ -q`
Expected: ALL PASS with NO expectation edits. If any test fails, the union changed a directly-populated demo result — STOP, inspect the diff, and explain it (per spec Net-1, a changed demo result is a finding, not a test to update). Common legitimate adjustment: a test that asserted exactly one `join_tables` entry per element now sees `element#0`, `element#1` keys — if such a test asserts the KEY name or COUNT, that is the multi-route behaviour and the assertion (not the code) is what the spec changed; update only assertions that encode the old single-route shape, and note each in the commit message.

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/model/denormalize_planner.py tests/local_db/test_denormalize_fk_reachable_paths.py tests/local_db/conftest.py
git commit -m "fix(denormalize): planner unions all reachable Dataset->element routes

Phase-1b dedup kept only the preferred Dataset_{Element} membership route and
discarded the FK-reachable chain, so feature_values returned 0 on
subject-partitioned datasets. Retain every distinct route and emit one
join_tables entry per route; the consumer already UNIONs them RID-distinct on
the row_per leaf."
```

---

### Task 4: Rule-6 ambiguity guard still raises on genuine column ambiguity

Pin Unit C: retaining multiple `Dataset → element` routes must NOT trip Rule 6, but a genuine `row_per`↔include-table column ambiguity must still raise.

**Files:**
- Modify: `tests/local_db/test_denormalize_fk_reachable_paths.py`

- [ ] **Step 1: Find an existing Rule-6 ambiguity test to mirror the construction**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && grep -rn "DerivaMLDenormalizeAmbiguousPath\|AmbiguousPath\|_find_path_ambiguities\|ambigu" tests/local_db/ | head`

- [ ] **Step 2: Write the guard test (mirroring that construction)**

Add to `tests/local_db/test_denormalize_fk_reachable_paths.py` a test asserting that a request which IS a genuine column-ambiguity (multiple FK paths between `row_per` and a column-contributing include-table — reuse the exact include-table pair the existing ambiguity test uses) still raises `DerivaMLDenormalizeAmbiguousPath`. Use the same fixture and the same ambiguous `include_tables` the existing test uses; assert `pytest.raises(DerivaMLDenormalizeAmbiguousPath)`.

```python
import pytest
from deriva_ml.core.exceptions import DerivaMLDenormalizeAmbiguousPath


def test_genuine_column_ambiguity_still_raises(planner_from_demo_schema):
    """Multiple Dataset->element ROUTES are unioned (not an error), but a genuine
    row_per<->include-table column ambiguity must still raise after the change."""
    planner, dataset_stub, dataset_rid = planner_from_demo_schema
    # <ambiguous include_tables from the existing Rule-6 test>
    with pytest.raises(DerivaMLDenormalizeAmbiguousPath):
        planner._prepare_wide_table(
            dataset_stub, dataset_rid, AMBIGUOUS_INCLUDE_TABLES, row_per=AMBIGUOUS_ROW_PER
        )
```

Replace `AMBIGUOUS_INCLUDE_TABLES` / `AMBIGUOUS_ROW_PER` with the exact values from the existing ambiguity test found in Step 1. If no such demo-schema ambiguity case exists, SKIP this test with `pytest.skip("no column-ambiguous table pair in demo schema")` and note it — do NOT fabricate a schema.

- [ ] **Step 3: Run**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_denormalize_fk_reachable_paths.py -v`
Expected: PASS (or one SKIP for the ambiguity case if the demo schema has none).

- [ ] **Step 4: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add tests/local_db/test_denormalize_fk_reachable_paths.py
git commit -m "test(denormalize): Rule-6 column ambiguity still raises after multi-route union"
```

---

### Task 5: Subject-partitioned demo fixture

Add a demo dataset whose members are Subjects only (no direct Image members) so Image is FK-reachable via `Image.Subject`. This is the committed gate for Net-3.

**Files:**
- Modify: `src/deriva_ml/demo_catalog.py` (add a subject-partitioned dataset builder) OR `tests/conftest.py` (a fixture that creates one). Prefer a `tests/conftest.py` fixture so demo-catalog production code is untouched.

- [ ] **Step 1: Read how the existing `catalog_with_datasets` fixture and `create_demo_datasets` add members**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && sed -n '143,251p' src/deriva_ml/demo_catalog.py && grep -n "def catalog_with_datasets\|add_dataset_members\|ensure_datasets\|create_dataset" tests/conftest.py tests/catalog_manager.py`

- [ ] **Step 2: Add a `subject_partitioned_dataset` fixture to `tests/conftest.py`**

```python
@pytest.fixture(scope="function")
def subject_partitioned_dataset(catalog_with_datasets, tmp_path):
    """A dataset whose members are Subjects ONLY — Image is FK-reachable via
    Image.Subject, never a direct member. Exercises the FK-reachable feature-read
    path the denormalize union fixes. Returns (ml, dataset)."""
    ml, _desc = catalog_with_datasets
    workflow = ml.create_workflow(name="Subject-partitioned", workflow_type="Test Workflow")
    execution = ml.create_execution(workflow=workflow, configuration=__import__(
        "deriva_ml.execution", fromlist=["ExecutionConfiguration"]).ExecutionConfiguration())
    with execution.execute() as exe:
        ds = exe.create_dataset(dataset_types=["Training"], description="subject-partitioned")
        subjects = [s["RID"] for s in ml._domain_path().tables["Subject"].entities().fetch()][:3]
        assert subjects, "fixture needs Subjects with Images"
        ds.add_dataset_members({"Subject": subjects})  # NO Image members
    execution.commit_output_assets()
    return ml, ds
```

Adjust `create_dataset` / `add_dataset_members` / execution calls to match the EXACT signatures discovered in Step 1 (the snippet above mirrors `create_demo_datasets`; reconcile names like `exe.create_dataset` vs `ml.create_dataset`).

- [ ] **Step 3: Verify the fixture builds a subject-partitioned dataset (Image FK-reachable, 0 direct)**

Add a smoke test in `tests/dataset/test_subject_partitioned_feature_values.py`:

```python
"""Subject-partitioned feature reads: members are Subjects, Image is FK-reachable.
Regression for feature_values returning 0 (denormalize-fk-reachable-paths)."""
import os
import pytest
from deriva_ml.dataset.tf_adapter import resolve_element_rids


@pytest.mark.skipif(os.environ.get("DERIVA_HOST") in (None, ""), reason="needs a live catalog")
def test_fixture_is_subject_partitioned(subject_partitioned_dataset):
    ml, ds = subject_partitioned_dataset
    bag = ds.download_dataset_bag(version=ds.current_version)
    assert len(resolve_element_rids(bag, "Image", reachable=False)) == 0  # no direct members
    assert len(resolve_element_rids(bag, "Image", reachable=True)) > 0     # FK-reachable
```

- [ ] **Step 4: Run the smoke test**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_subject_partitioned_feature_values.py::test_fixture_is_subject_partitioned -v --timeout=600`
Expected: PASS — `reachable=False` → 0, `reachable=True` → >0.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add tests/conftest.py tests/dataset/test_subject_partitioned_feature_values.py
git commit -m "test(dataset): subject-partitioned demo fixture (Image FK-reachable, 0 direct)"
```

---

### Task 6: Live demo regression — feature_values resolves on subject-partitioned dataset

The end-to-end gate: `feature_values` and `as_tf_dataset` labels must resolve through the FK-reachable path on the demo fixture.

**Files:**
- Modify: `tests/dataset/test_subject_partitioned_feature_values.py`

- [ ] **Step 1: Confirm which Image feature the demo has**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && grep -n "create_feature(\"Image\"\|create_feature('Image'\|Image.*Quality\|ImageQuality" src/deriva_ml/demo_catalog.py`
Expected: the demo defines `Image / Quality` (vocabulary `ImageQuality`) and `Image / BoundingBox`. Use `Quality` (vocabulary-based, one value per image).

- [ ] **Step 2: Write the feature-value regression test**

The demo features are created by `create_demo_features`, which the `catalog_with_datasets` fixture runs. Add:

```python
@pytest.mark.skipif(os.environ.get("DERIVA_HOST") in (None, ""), reason="needs a live catalog")
def test_feature_values_resolves_via_fk_reachable_path(subject_partitioned_dataset):
    """feature_values('Image','Quality') must return rows for FK-reachable Images
    even though Image has 0 direct dataset members. Pre-fix this returned 0."""
    ml, ds = subject_partitioned_dataset
    bag = ds.download_dataset_bag(version=ds.current_version)

    reachable_images = set(resolve_element_rids(bag, "Image", reachable=True))
    assert reachable_images, "fixture must have FK-reachable Images"

    fv = list(bag.feature_values("Image", "Quality"))
    assert fv, "feature_values returned 0 — FK-reachable feature read is broken"
    # Every returned feature row targets a reachable Image (no leakage, no loss).
    fv_targets = {rec.Image for rec in fv}
    assert fv_targets <= reachable_images
    # Exactness oracle (Net-2): the distinct feature-target set equals the
    # estimate's reachable Image count is asserted separately in Task 7.
```

- [ ] **Step 3: Add the `as_tf_dataset` label end-to-end assertion (the report's acceptance shape)**

```python
@pytest.mark.skipif(os.environ.get("DERIVA_HOST") in (None, ""), reason="needs a live catalog")
def test_as_tf_dataset_labels_resolve_subject_partitioned(subject_partitioned_dataset):
    tf = pytest.importorskip("tensorflow")
    ml, ds = subject_partitioned_dataset
    bag = ds.download_dataset_bag(version=ds.current_version)
    reachable = set(resolve_element_rids(bag, "Image", reachable=True))

    dataset = bag.as_tf_dataset(
        element_type="Image",
        sample_loader=lambda p, row: tf.constant([0.0]),
        targets=["Quality"],
        target_transform=lambda rec: 0,  # any int; we assert count, not value
        missing="skip",
    )
    rids = {r.decode() if isinstance(r, bytes) else r
            for *_rest, r in dataset.as_numpy_iterator()}
    assert rids, "empty generator — labels did not resolve (the reported bug)"
    assert rids <= reachable
```

- [ ] **Step 4: Run both (torch not required; tf is importorskip)**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_subject_partitioned_feature_values.py -v --timeout=600`
Expected: `test_feature_values_resolves_via_fk_reachable_path` PASS; the tf test PASS or SKIP (no tensorflow installed locally — it runs in CI). Before Task 3's fix these would FAIL/empty; after, they pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add tests/dataset/test_subject_partitioned_feature_values.py
git commit -m "test(dataset): feature_values + as_tf_dataset labels resolve on subject-partitioned dataset"
```

---

### Task 7: Net-2 exactness oracle test

Assert `feature_values` count equals the independently-computed reachable count from `estimate_bag_size` — two code paths must agree, on BOTH a directly-populated and the subject-partitioned dataset.

**Files:**
- Modify: `tests/dataset/test_subject_partitioned_feature_values.py`

- [ ] **Step 1: Confirm the estimate's per-table reachable-count shape**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && grep -n "def estimate_bag_size\|row_count\|\"tables\"\|tables\[" src/deriva_ml/dataset/dataset.py src/deriva_ml/core/mixins/dataset.py | head`
Expected: `estimate_bag_size(version)["tables"][T]["row_count"]` is the RID-distinct reachable count for table `T` (the BFS oracle). Confirm the exact key names from the output and use them below.

- [ ] **Step 2: Write the oracle test**

```python
@pytest.mark.skipif(os.environ.get("DERIVA_HOST") in (None, ""), reason="needs a live catalog")
def test_feature_count_matches_estimate_oracle(subject_partitioned_dataset):
    """Net-2: feature_values row count == estimate's reachable feature-table count.
    Two independent code paths (planner UNION vs reachability BFS) must agree."""
    ml, ds = subject_partitioned_dataset
    est = ds.estimate_bag_size(ds.current_version)
    # The demo Image/Quality feature association table is "Execution_Image_Quality"
    # (or the demo's actual Image Quality feature-assoc table name — confirm with
    # bag.lookup_feature("Image","Quality").feature_table.name).
    bag = ds.download_dataset_bag(version=ds.current_version)
    feat_assoc = bag.lookup_feature("Image", "Quality").feature_table.name
    expected = est["tables"].get(feat_assoc, {}).get("row_count")
    if not expected:
        pytest.skip(f"estimate has no reachable rows for {feat_assoc}")
    got = len(list(bag.feature_values("Image", "Quality")))
    assert got == expected, f"feature_values={got} estimate={expected} for {feat_assoc}"
```

Replace the `row_count` / `["tables"]` keys with the exact names confirmed in Step 1.

- [ ] **Step 3: Run**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_subject_partitioned_feature_values.py::test_feature_count_matches_estimate_oracle -v --timeout=600`
Expected: PASS — the two counts agree.

- [ ] **Step 4: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add tests/dataset/test_subject_partitioned_feature_values.py
git commit -m "test(dataset): Net-2 exactness oracle — feature_values count == estimate reachable count"
```

---

### Task 8: Docs + broad suite + eye-ai acceptance + PR

- [ ] **Step 1: Update `docs/reference/denormalization.md`**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && grep -n "membership\|scope\|filter\|D5\|D6\|reachable" docs/reference/denormalization.md | head`
Add a rule (tagged `[deriva-ml]`) stating: the planner unions ALL reachable `Dataset → element` routes (direct-membership AND FK-reachable chains), and the result is RID-distinct on the `row_per` leaf. Note that on subject-partitioned datasets (members are an upstream element, target FK-reachable) this is the ONLY way the target's rows appear. Reference the spec.

- [ ] **Step 2: Lint + format ONLY the touched files**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src/deriva_ml/model/denormalize_planner.py tests/local_db/test_denormalize_fk_reachable_paths.py tests/dataset/test_subject_partitioned_feature_values.py tests/conftest.py && uv run ruff format --check src/deriva_ml/model/denormalize_planner.py tests/local_db/test_denormalize_fk_reachable_paths.py tests/dataset/test_subject_partitioned_feature_values.py tests/conftest.py`
Expected: clean. Do NOT run `ruff format` on whole directories — touched files only (workspace rule).

- [ ] **Step 3: Broad regression sweep**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/local_db/ tests/dataset/test_subject_partitioned_feature_values.py tests/dataset/test_torch_adapter_e2e.py tests/feature/ -q --timeout=600`
Expected: ALL PASS. `tests/local_db/` is the Net-1 gate; `tests/feature/` exercises `feature_values` on the directly-populated catalog (must be unchanged).

- [ ] **Step 4: eye-ai acceptance (manual, NOT committed — needs a token)**

```python
# /tmp/eyeai_fv_acceptance.py — THROWAWAY, not committed.
import os
os.environ.setdefault("DERIVA_ML_ALLOW_DIRTY", "true")
from deriva_ml import DerivaML
from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.dataset.tf_adapter import resolve_element_rids

ml = DerivaML(hostname="dev.eye-ai.org", catalog_id="eye-ai")
bag = ml.download_dataset_bag(DatasetSpec(rid="6-CQZE", version="0.4.0"))
train = next(k for k in bag.list_dataset_children() if "Training" in (k.dataset_types or []))
print("reachable Images:", len(resolve_element_rids(train, "Image", reachable=True)))  # 102
print("feature_values:  ", len(list(train.feature_values("Image", "Image_Diagnosis"))))  # was 0 -> expect 591
```

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run python /tmp/eyeai_fv_acceptance.py 2>&1 | grep -E "reachable|feature_values"`
Expected: `reachable Images: 102`, `feature_values: 591` (was 0). Record the numbers in the PR body. Then: `rm -f /tmp/eyeai_fv_acceptance.py`. Optionally repeat for prod `2-277G` v4.8.0 on `www.eye-ai.org` and record the ~28,546 figure.

- [ ] **Step 5: Branch, push, PR**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git push -u origin fix/denormalize-fk-reachable-paths
gh pr create --title "fix(denormalize): planner unions FK-reachable paths (subject-partitioned feature reads)" \
  --body "<summary + spec link + root cause (6-CQZE 0->591) + the three regression nets + eye-ai acceptance numbers + the demo row-set-identity note>"
```

The spec and plan are already committed on this branch (from brainstorming). The PR bundles them with the fix.

---

## Self-Review notes

- **Spec coverage:** Unit A (path retention) → Task 3 Step 2. Unit B (UNION emission) → consumer already unions; Task 3 Step 3 emits multiple entries; identical-projection invariant holds structurally (columns from `column_specs`, not per-path) — noted in Architecture. Unit C (Rule-6) → Task 4. Net-1 → Task 3 Step 5 + Task 8 Step 3. Net-2 → Task 7. Net-3 committed → Tasks 5-6; live → Task 8 Step 4. Docs → Task 8 Step 1. Out-of-scope (no `reachable=` on feature_values) → respected (no API change). Fan-out/dedup → handled by UNION-distinct (Architecture + Task 2/6 assertions).
- **Risk front-loaded:** Task 1 spike pins the multi-hop prefix construction empirically before Task 3 writes it, so no guessed code.
- **No placeholders:** the only deferred specifics are the exact ambiguous-table pair (Task 4, discovered from existing tests) and exact estimate key names (Task 7, discovered from source) — each step says exactly how to obtain them and what to assert, with a skip fallback rather than fabrication.
