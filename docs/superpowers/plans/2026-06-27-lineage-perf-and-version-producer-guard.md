# Lineage Perf + Version-Producer Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `list_input_datasets_with_versions` fetch only the consumed `Dataset_Version` RIDs (not the whole table per lineage node), and stop an execution that produced the consumed version of a dataset it also consumed from being listed as its own parent (a confirmed false cycle).

**Architecture:** Two small, independent corrections to the merged consumed-version lineage fix, both from a second `/codex` review. (1) The helper builds its RID→version map with a chunked `.in_()` over only the consumed-version RIDs, skipping the fetch entirely when no edge is pinned. (2) `_walk_node`'s consumed-dataset loop excludes the currently-expanding execution from the version-producer it adds (mirroring the member-producer guard one line below).

**Tech Stack:** Python 3.13, `uv`, pytest, deriva-ml core (`deriva_ml.execution._helpers`, `deriva_ml.core.mixins.execution.ExecutionMixin`), ERMrest PathBuilder.

## Global Constraints

- Work in `../deriva-ml` (`/Users/carl/GitHub/DerivaML/deriva-ml`) on branch `feature/lineage-perf-version-guard` (already created; the spec is committed there).
- Use `uv` for everything: `uv run python -m pytest` (NOT `uv run pytest`), `uv run ruff check`, `uv run ruff format`.
- Always `cd /Users/carl/GitHub/DerivaML/deriva-ml && <cmd>` in every Bash call — the shell CWD is not persistent.
- Google-style docstrings on any new function/method.
- NO public-model change: `src/deriva_ml/execution/lineage.py` is NOT modified.
- NO contract change: `list_input_datasets_with_versions` keeps its signature and `list[tuple[Dataset, str | None]]` return (version STRING or None). `list_input_datasets` (the original) and `_producer_of_dataset` are NOT touched.
- The bounded fetch MUST chunk at 500 (the existing `_MEMBER_PRODUCER_CHUNK` convention) via the `.in_()` predicate; it must NEVER do an unfiltered full-table `Dataset_Version.entities().fetch()`. Use a local module constant `_VERSION_RID_CHUNK = 500` in `_helpers.py` to avoid a `_helpers.py` → `execution.py` import cycle.
- The `.in_()` predicate is the proven form: `column.in_(iterable)` (see `_distinct_member_output_producers` in `execution.py` and `bag_builder.py:1108`). It raises on an empty iterable, so only call it on a non-empty chunk.
- Live test gates on `DERIVA_HOST` (the convention in `test_lookup_lineage_live.py`).

---

### Task 1: bounded version fetch in `list_input_datasets_with_versions`

**Files:**
- Modify: `src/deriva_ml/execution/_helpers.py` (the `list_input_datasets_with_versions` body, ~lines 265-285; add the `_VERSION_RID_CHUNK` constant)
- Test: `tests/execution/test_input_datasets_with_versions.py` (extend the existing `_make_ml` mock + add tests)

**Interfaces:**
- Unchanged public surface: `list_input_datasets_with_versions(*, ml_instance, execution_rid) -> list[tuple[Dataset, str | None]]`.

**Background — the current body (the thing to fix):**
```python
    pb = ml_instance.pathBuilder()
    dataset_exec = pb.schemas[ml_instance.ml_schema].Dataset_Execution
    records = [
        record
        for record in dataset_exec.filter(dataset_exec.Execution == execution_rid).entities().fetch()
        if record.get("Dataset")
    ]
    if not records:
        return []

    # Dataset_Execution.Dataset_Version is an FK — the value is the
    # Dataset_Version row RID, not the version string. Resolve RID -> Version.
    version_path = pb.schemas[ml_instance.ml_schema].tables["Dataset_Version"]
    rid_to_version: dict[str, str | None] = {row["RID"]: row.get("Version") for row in version_path.entities().fetch()}

    result: list[tuple[Any, str | None]] = []
    for record in records:
        version_rid = record.get("Dataset_Version")
        consumed_version = rid_to_version.get(version_rid) if version_rid else None
        result.append((ml_instance.lookup_dataset(record["Dataset"]), consumed_version))
    return result
```
The `version_path.entities().fetch()` is the **full-table scan** (every `Dataset_Version` row in the catalog) issued on every lineage node. We replace it with a chunked `.in_()` over only the consumed-version RIDs.

**Background — the chunking precedent** (in `execution.py`, `_distinct_member_output_producers`):
```python
        for start in range(0, len(member_rids), _MEMBER_PRODUCER_CHUNK):
            chunk = member_rids[start : start + _MEMBER_PRODUCER_CHUNK]
            rows = (
                assoc_path.filter(assoc_path.columns[asset_fk].in_(chunk))
                .filter(assoc_path.Asset_Role == "Output")
                .entities()
                .fetch()
            )
```
The `Dataset_Version` table's RID column is accessed as `version_path.RID` (PathBuilder attribute access), so the filter is `version_path.filter(version_path.RID.in_(chunk))`.

- [ ] **Step 1: Extend the test mock to record the Dataset_Version filter**

In `tests/execution/test_input_datasets_with_versions.py`, the current `_make_ml` wires `dv_path.entities.return_value = dv_entities` (an UNFILTERED fetch). The bounded helper will instead call `dv_path.filter(dv_path.RID.in_(chunk)).entities().fetch()`. Rework the `Dataset_Version` half of `_make_ml` so it (a) supports the filtered call, (b) records the RIDs passed to `.in_()`, and (c) returns only the version rows whose RID is in that set — and ALSO record whether the UNFILTERED `dv_path.entities()` path was used (so a test can assert it is NOT).

Replace the `Dataset_Version path` block in `_make_ml` with:

```python
    # Dataset_Version path — the bounded helper calls
    # dv_path.filter(dv_path.RID.in_(chunk)).entities().fetch(). Record the
    # RIDs requested via .in_() and serve only matching version_rows. Also
    # expose a flag for whether the UNFILTERED entities() path was hit.
    version_by_rid = {r["RID"]: r for r in version_rows}
    calls = {"in_rids": [], "unfiltered_fetch": False}

    def _dv_in(rids):
        rids = list(rids)
        calls["in_rids"].append(rids)
        matched = [version_by_rid[r] for r in rids if r in version_by_rid]
        filtered_entities = MagicMock()
        filtered_entities.fetch = lambda: matched
        filtered_path = MagicMock()
        filtered_path.entities = lambda: filtered_entities
        return filtered_path

    dv_rid_col = MagicMock()
    dv_rid_col.in_ = _dv_in

    def _dv_unfiltered_entities():
        calls["unfiltered_fetch"] = True
        e = MagicMock()
        e.fetch = lambda: version_rows
        return e

    dv_path = MagicMock()
    dv_path.RID = dv_rid_col
    dv_path.filter = lambda predicate: predicate  # predicate IS the filtered_path from _dv_in
    dv_path.entities = _dv_unfiltered_entities
```

Then expose `calls` on the returned `ml` so tests can inspect it: at the end of `_make_ml`, before `return ml`, add `ml._dv_calls = calls`.

Note: `dv_path.filter(predicate)` returns `predicate` because `_dv_in` already returns the object whose `.entities().fetch()` yields the matched rows — i.e. the helper does `version_path.filter(version_path.RID.in_(chunk)).entities().fetch()`, and `version_path.RID.in_(chunk)` returns `filtered_path`, and `filter(filtered_path)` returns it unchanged. Keep this wiring exactly so the chain resolves.

- [ ] **Step 2: Update the existing three tests for the new mock + assert no full-table fetch**

The existing `test_pairs_dataset_with_consumed_version`, `test_version_none_when_edge_has_no_pin`, and `test_skips_rows_without_dataset` still pass the same `de_rows`/`version_rows` — the reworked mock serves them via the filtered path. After each result assertion, add:
```python
    assert ml._dv_calls["unfiltered_fetch"] is False
```
to lock in that the helper never full-table-scans. (In `test_version_none_when_edge_has_no_pin`, the single edge is unpinned, so the helper should skip the version fetch entirely — `ml._dv_calls["in_rids"]` should be `[]`; add `assert ml._dv_calls["in_rids"] == []` there.)

- [ ] **Step 3: Write the new failing tests**

Add to `tests/execution/test_input_datasets_with_versions.py`:

```python
def test_missing_fk_rid_resolves_to_none():
    """A consumed-version RID absent from Dataset_Version yields None, not a crash."""
    ml = _make_ml(
        de_rows=[{"Dataset": "1-DSAA", "Dataset_Version": "VR_MISSING", "Execution": "2-EXAA"}],
        version_rows=[{"RID": "VR1", "Version": "1.0.0"}],  # VR_MISSING not present
    )
    result = list_input_datasets_with_versions(ml_instance=ml, execution_rid="2-EXAA")
    assert result == [] or result  # non-empty; assert the pair below
    ds, version = result[0]
    assert ds.dataset_rid == "1-DSAA"
    assert version is None
    assert ml._dv_calls["unfiltered_fetch"] is False


def test_mixed_pinned_and_unpinned_inputs():
    """One pinned + one unpinned edge in the same execution both survive."""
    ml = _make_ml(
        de_rows=[
            {"Dataset": "1-DSAA", "Dataset_Version": "VR1", "Execution": "2-EXAA"},
            {"Dataset": "1-DSAB", "Execution": "2-EXAA"},  # no Dataset_Version key
        ],
        version_rows=[{"RID": "VR1", "Version": "1.0.0"}],
    )
    result = list_input_datasets_with_versions(ml_instance=ml, execution_rid="2-EXAA")
    pairs = {(ds.dataset_rid, v) for ds, v in result}
    assert pairs == {("1-DSAA", "1.0.0"), ("1-DSAB", None)}
    # Only the pinned RID was requested via .in_().
    assert ml._dv_calls["in_rids"] == [["VR1"]]
    assert ml._dv_calls["unfiltered_fetch"] is False


def test_empty_inputs_never_fetches_versions():
    """No input edges -> [] and the Dataset_Version table is never touched."""
    ml = _make_ml(de_rows=[], version_rows=[{"RID": "VR1", "Version": "1.0.0"}])
    result = list_input_datasets_with_versions(ml_instance=ml, execution_rid="2-EXAA")
    assert result == []
    assert ml._dv_calls["in_rids"] == []
    assert ml._dv_calls["unfiltered_fetch"] is False


def test_bounded_fetch_requests_only_consumed_rids():
    """The version fetch is filtered to exactly the distinct consumed RIDs."""
    ml = _make_ml(
        de_rows=[
            {"Dataset": "1-DSAA", "Dataset_Version": "VR1", "Execution": "2-EXAA"},
            {"Dataset": "1-DSAB", "Dataset_Version": "VR2", "Execution": "2-EXAA"},
        ],
        version_rows=[{"RID": "VR1", "Version": "1.0.0"}, {"RID": "VR2", "Version": "2.0.0"}],
    )
    list_input_datasets_with_versions(ml_instance=ml, execution_rid="2-EXAA")
    # One chunk, containing exactly the two consumed RIDs (order-insensitive).
    assert len(ml._dv_calls["in_rids"]) == 1
    assert set(ml._dv_calls["in_rids"][0]) == {"VR1", "VR2"}
    assert ml._dv_calls["unfiltered_fetch"] is False
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_input_datasets_with_versions.py -k "missing_fk or mixed_pinned or empty_inputs or bounded_fetch or pairs_dataset or version_none" -v`
Expected: the new tests + the updated existing ones FAIL — the current helper full-table-scans (`unfiltered_fetch` is True, `in_rids` is empty), so the `in_rids`/`unfiltered_fetch` assertions fail.

- [ ] **Step 5: Implement the bounded fetch**

Add the constant near the top of `src/deriva_ml/execution/_helpers.py` (with the other module constants, or just under the imports):

```python
# Max Dataset_Version RIDs per .in_() filter, matching the shared 500 chunk
# convention (cf. _MEMBER_PRODUCER_CHUNK in core/mixins/execution.py). Defined
# locally to avoid a _helpers -> execution import cycle.
_VERSION_RID_CHUNK = 500
```

Replace the version-map build (the `version_path = ...` line and the
`rid_to_version = {... version_path.entities().fetch()}` line) with:

```python
    # Dataset_Execution.Dataset_Version is an FK — the value is the
    # Dataset_Version row RID, not the version string. Resolve only the RIDs
    # actually referenced by these input edges (NOT the whole table — this
    # helper runs once per walked execution, so a full-table scan would be
    # O(walked-executions x total-versions)).
    wanted_rids = {r["Dataset_Version"] for r in records if r.get("Dataset_Version")}
    rid_to_version: dict[str, str | None] = {}
    if wanted_rids:
        version_path = pb.schemas[ml_instance.ml_schema].tables["Dataset_Version"]
        wanted = list(wanted_rids)
        for start in range(0, len(wanted), _VERSION_RID_CHUNK):
            chunk = wanted[start : start + _VERSION_RID_CHUNK]
            for row in version_path.filter(version_path.RID.in_(chunk)).entities().fetch():
                rid_to_version[row["RID"]] = row.get("Version")
```

Leave the `result`-building loop below unchanged (it already does
`rid_to_version.get(version_rid) if version_rid else None`, which yields `None`
for a missing RID). When `wanted_rids` is empty (all edges unpinned), the map
stays empty and the version table is never fetched.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_input_datasets_with_versions.py -v`
Expected: ALL pass (the 3 updated + 4 new helper tests, plus the unchanged `_producer_of_dataset` tests in the same file).

- [ ] **Step 7: Lint + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run ruff check src/deriva_ml/execution/_helpers.py tests/execution/test_input_datasets_with_versions.py
uv run ruff format --check src/deriva_ml/execution/_helpers.py tests/execution/test_input_datasets_with_versions.py
git add src/deriva_ml/execution/_helpers.py tests/execution/test_input_datasets_with_versions.py
git commit -m "perf(lineage): bounded Dataset_Version fetch (consumed RIDs only, chunked) — not a full-table scan per node"
```

---

### Task 2: version-producer self-parent guard + walk/seam tests

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py` (`_walk_node` consumed loop, the producer-add at ~line 1629-1631)
- Test: `tests/execution/test_lookup_lineage_unit.py` (add walk-level tests via `_FakeML`); `tests/execution/test_input_datasets_with_versions.py` (add the `_input_dataset_pairs` seam test)

**Interfaces:**
- Consumes: the `_FakeML` harness (`set_versioned_producer`, `set_member_producers`, `_StubDataset(consumed_version=...)`, `_input_dataset_pairs` override) from the consumed-version fix.

**Background — the unguarded producer add (the bug):**
```python
                producer = self._producer_of_dataset(ds.dataset_rid, version=consumed_version)
                if producer:
                    parent_rids.add(producer)
                # ... member_producers - {execution_rid}  (the member guard)
```
If `producer == execution_rid` (the execution produced the consumed version of a dataset it also consumed), it lands in `parent_rids`; the recursion re-enters `_walk_node(execution_rid)`, finds it `in_progress`, and sets `cycle_detected = True` (a false cycle). The member-producers below already subtract `execution_rid`; the producer does not.

- [ ] **Step 1: Write the failing walk-level tests**

Add to `tests/execution/test_lookup_lineage_unit.py` (the `_FakeML` harness already has `add_dataset`, `add_execution`, `set_versioned_producer`, `_StubDataset`, `_producer_of_dataset` override keyed by dataset+version):

```python
def test_self_parent_via_version_producer_no_false_cycle():
    """An execution that consumed D AND produced D's consumed version must not
    be its own parent (no false cycle). Guards execution.py:1631."""
    ml = _FakeML()
    ml.add_dataset("1-DSRC", producer=None)
    # EXSC consumes D-VER, and is ALSO the producer of D-VER's consumed version.
    ml.add_dataset("1-DVER", producer="2-EXSC")
    ml.set_versioned_producer("1-DVER", "1.0.0", "2-EXSC")  # consumed-version producer == consumer
    ml.add_execution("2-EXSC", input_datasets=[_StubDataset("1-DVER", consumed_version="1.0.0")])
    ml.add_dataset("1-DOUT", producer="2-EXSC")

    result = ml.lookup_lineage("1-DOUT")

    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXSC"
    # 2-EXSC must NOT be its own parent, and no false cycle.
    assert all(p.execution.rid != "2-EXSC" for p in result.lineage.parents)
    assert result.cycle_detected is False


def test_multiple_consumed_datasets_different_versions():
    """An execution consuming D1@1.0.0 (producer EXV1) and D2@2.0.0 (producer
    EXV2) surfaces BOTH version-specific producers and BOTH summary versions."""
    ml = _FakeML()
    ml.add_dataset("1-DSR1", producer=None)
    ml.add_dataset("1-DSR2", producer=None)
    ml.add_execution("2-EXV1", input_datasets=[_StubDataset("1-DSR1")])
    ml.add_execution("2-EXV2", input_datasets=[_StubDataset("1-DSR2")])
    ml.add_dataset("1-DD1", producer=None)
    ml.add_dataset("1-DD2", producer=None)
    ml.set_versioned_producer("1-DD1", "1.0.0", "2-EXV1")
    ml.set_versioned_producer("1-DD2", "2.0.0", "2-EXV2")
    ml.add_execution(
        "2-EXMID",
        input_datasets=[
            _StubDataset("1-DD1", consumed_version="1.0.0"),
            _StubDataset("1-DD2", consumed_version="2.0.0"),
        ],
    )
    ml.add_dataset("1-DEND", producer="2-EXMID")

    result = ml.lookup_lineage("1-DEND")

    parent_rids = {p.execution.rid for p in result.lineage.parents}
    assert {"2-EXV1", "2-EXV2"} <= parent_rids
    summary_versions = {s.rid: s.version for s in result.lineage.consumed_datasets}
    assert summary_versions["1-DD1"] == "1.0.0"
    assert summary_versions["1-DD2"] == "2.0.0"
```

- [ ] **Step 2: Run them to verify the self-parent one fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_lookup_lineage_unit.py -k "self_parent_via_version_producer or multiple_consumed_datasets" -v`
Expected: `test_self_parent_via_version_producer_no_false_cycle` FAILS (asserts no self-parent + no cycle, but the unguarded producer add makes `2-EXSC` its own parent → `cycle_detected` True). `test_multiple_consumed_datasets_different_versions` likely PASSES already (the version threading works) — that's fine; it locks in coverage.

- [ ] **Step 3: Implement the guard**

In `src/deriva_ml/core/mixins/execution.py`, change the producer add in `_walk_node`'s consumed loop:

```python
                producer = self._producer_of_dataset(ds.dataset_rid, version=consumed_version)
                if producer:
                    parent_rids.add(producer)
```
to:
```python
                producer = self._producer_of_dataset(ds.dataset_rid, version=consumed_version)
                # Never the execution we are currently expanding: if it produced
                # the consumed version of a dataset it also consumed, listing it
                # as its own parent re-enters `in_progress` and flags a false
                # cycle (same reason the member-producers below subtract it).
                if producer and producer != execution_rid:
                    parent_rids.add(producer)
```

- [ ] **Step 4: Run the walk tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_lookup_lineage_unit.py -k "self_parent_via_version_producer or multiple_consumed_datasets" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Add the `_input_dataset_pairs` real-seam test**

The seam `_input_dataset_pairs` wraps the module helper but is only exercised offline via the `_FakeML` override (the real wiring is only hit by the live test). Add a direct test that the real method forwards its args. Append to `tests/execution/test_input_datasets_with_versions.py`:

```python
def test_input_dataset_pairs_forwards_to_helper(monkeypatch):
    """ExecutionMixin._input_dataset_pairs calls the module helper with
    ml_instance=self and the given execution_rid (guards the real seam wiring
    that otherwise only the live test exercises)."""
    captured = {}

    def _fake_helper(*, ml_instance, execution_rid):
        captured["ml_instance"] = ml_instance
        captured["execution_rid"] = execution_rid
        return [("sentinel-ds", "1.0.0")]

    monkeypatch.setattr(
        "deriva_ml.execution._helpers.list_input_datasets_with_versions", _fake_helper
    )

    ml = ExecutionMixin.__new__(ExecutionMixin)
    result = ml._input_dataset_pairs("2-EXAA")

    assert captured["ml_instance"] is ml
    assert captured["execution_rid"] == "2-EXAA"
    assert result == [("sentinel-ds", "1.0.0")]
```

Note: `_input_dataset_pairs` imports the helper inside the method body
(`from deriva_ml.execution._helpers import list_input_datasets_with_versions`),
so `monkeypatch.setattr` on the `_helpers` module attribute takes effect at call
time. If the import is module-top instead, patch where it's looked up — verify by
reading `_input_dataset_pairs` and patch the name the method actually resolves.

- [ ] **Step 6: Run the seam test**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_input_datasets_with_versions.py -k input_dataset_pairs -v`
Expected: PASS (1 passed).

- [ ] **Step 7: Full no-regression run**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_lookup_lineage_unit.py tests/execution/test_producers_of_dataset_members.py tests/execution/test_input_datasets_with_versions.py -q`
Expected: all green (existing lineage tests + the new ones). The guard `producer != execution_rid` is a no-op for all existing tests (their producers are never the expanding execution).

- [ ] **Step 8: Lint + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run ruff check src/deriva_ml/core/mixins/execution.py tests/execution/test_lookup_lineage_unit.py tests/execution/test_input_datasets_with_versions.py
uv run ruff format --check src/deriva_ml/core/mixins/execution.py tests/execution/test_lookup_lineage_unit.py tests/execution/test_input_datasets_with_versions.py
git add src/deriva_ml/core/mixins/execution.py tests/execution/test_lookup_lineage_unit.py tests/execution/test_input_datasets_with_versions.py
git commit -m "fix(lineage): exclude self from version-producer parents (no false cycle) + seam/multi-version tests"
```

---

## Final verification (after both tasks)

- [ ] Full execution test dir: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/ -q` — all pass (live tests skip without `DERIVA_HOST`; the 2 pre-existing `test_workflow_creation_*` subprocess failures are unrelated).
- [ ] Live consumed-version test still passes end to end (the bounded fetch must still resolve the consumed version — that RID IS in the wanted set): `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_HOST=localhost DERIVA_ML_ALLOW_DIRTY=true uv run python -m pytest tests/execution/test_lookup_lineage_live.py::test_lookup_lineage_reflects_consumed_version_not_latest -v` — PASS. (If no container, report + confirm skip.)
- [ ] Lint the touched surface: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src tests`.
