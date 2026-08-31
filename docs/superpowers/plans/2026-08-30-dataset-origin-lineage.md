# Dataset Origin + Version-Attribution Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `lookup_lineage()` on a dataset reports the *origin* (first-recorded version author) instead of the last writer, plus a full version-attribution trace on the root — closing issue #367.

**Architecture:** No schema change. A new `_dataset_version_rows()` helper fetches+orders the version history once (RCT-primary total order); the dataset branch of `_classify_rid()` derives origin, `origin_recorded`, and `version_history` from that single fetch, with author summaries resolved by a new batched `_execution_summaries()` helper. Walk seeding is separated from origin attribution: the unknown-provenance sentinel never seeds the walk, and the walk-root overwrite of `root.producing_execution` no longer applies to Dataset roots.

**Tech Stack:** Python ≥3.12, Pydantic v2, deriva-py datapath, `packaging` (already a dependency), pytest with MagicMock offline fixtures.

**Spec:** `docs/superpowers/specs/2026-08-30-dataset-origin-lineage-design.md` (approved; survived two Codex review rounds — read it before starting).

## Global Constraints

- Run everything from `/Users/carl/GitHub/DerivaML/deriva-ml` with `cd` chained into the same Bash call (CWD is not persistent).
- All test/lint commands: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest ...` / `uv run ruff check ...` — never bare `pytest`/`python`.
- **No hand-written RID literals in new tests.** Generate RIDs programmatically in fixtures (see `_rid()` helper in Task 1) and assert only against values that flowed through the code under test. RID comparisons are equality-only.
- Every behavioral test must be shown to FAIL before its implementation lands (the plan's test-first steps do this; do not skip the "verify it fails" steps).
- Docstrings: Google style with Args/Returns/Raises/Example. Pure-Python doctest examples run for real; catalog-touching examples need `# doctest: +SKIP`.
- Existing behavior pinned: version-pinned resolution (`_producer_of_dataset(version=...)`), non-Dataset root behavior, and tk-018 member-producer walk seeding all stay unchanged except where the spec says otherwise.
- Branch: work continues on `feat/dataset-origin-lineage-367` (already exists, spec committed there). All changes land via one PR at the end.

---

### Task 1: `_dataset_version_rows()` — single fetch + total ordering

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py` (add module-level `_version_row_sort_key` and method `_dataset_version_rows`, near `_producer_of_dataset` ~line 1322)
- Test: `tests/execution/test_dataset_version_rows.py` (new)

**Interfaces:**
- Consumes: `self.pathBuilder()`, `self.ml_schema` (existing mixin surface).
- Produces: `_dataset_version_rows(self, dataset_rid: RID) -> list[dict[str, Any]]` — ALL `Dataset_Version` rows for the dataset, sorted earliest→latest by the total order `(RCT, parseable-flag, parsed PEP 440 Version, raw label)`. Later tasks index `rows[0]` for origin and iterate for the trace. Also `_version_row_sort_key(row: dict) -> tuple` (module-level, unit-testable).

- [ ] **Step 1: Write the failing tests**

Create `tests/execution/test_dataset_version_rows.py`:

```python
"""Unit tests for ExecutionMixin._dataset_version_rows ordering (issue #367).

Offline: the pathBuilder is a dict-backed mock. RIDs are generated
programmatically by _rid() — never hand-written literals — and assertions
compare only values that flowed through the helper.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from deriva_ml.core.mixins.execution import ExecutionMixin, _version_row_sort_key


def _rid(n: int) -> str:
    """Programmatic RID-shaped string (pattern [A-Z\\d]{1,4} segments)."""
    return f"1-{n:04X}"


def _mixin_with_version_rows(rows: list[dict]):
    """ExecutionMixin whose Dataset_Version fetch returns `rows`."""
    ml = ExecutionMixin.__new__(ExecutionMixin)
    ml.ml_schema = "deriva-ml"
    version_path = MagicMock()
    version_path.filter.return_value.entities.return_value.fetch.return_value = list(rows)
    schema = MagicMock()
    schema.tables = {"Dataset_Version": version_path}
    pb = MagicMock()
    pb.schemas = {"deriva-ml": schema}
    ml.pathBuilder = lambda: pb
    return ml


def _row(rct: str, version: str, exec_n: int | None, ver_rid: int = 0) -> dict:
    return {
        "RID": _rid(1000 + ver_rid),
        "RCT": rct,
        "Version": version,
        "Execution": _rid(exec_n) if exec_n is not None else None,
        "Description": f"notes for {version}",
    }


def test_rct_primary_beats_version_label():
    """A later-created row with a LOWER version label sorts after."""
    early = _row("2025-01-01T00:00:00Z", "1.0.0", 1, ver_rid=1)
    late_low_label = _row("2026-01-01T00:00:00Z", "0.9.0", 2, ver_rid=2)
    ml = _mixin_with_version_rows([late_low_label, early])
    out = ml._dataset_version_rows(_rid(500))
    assert [r["RID"] for r in out] == [early["RID"], late_low_label["RID"]]


def test_rct_tie_falls_back_to_pep440_label():
    t = "2025-01-01T00:00:00Z"
    v2 = _row(t, "0.2.0", 2, ver_rid=2)
    v10 = _row(t, "0.10.0", 3, ver_rid=3)
    ml = _mixin_with_version_rows([v10, v2])
    out = ml._dataset_version_rows(_rid(500))
    assert [r["Version"] for r in out] == ["0.2.0", "0.10.0"]  # numeric, not lexical


def test_unparseable_labels_sort_after_parseable_within_rct_tie():
    t = "2025-01-01T00:00:00Z"
    good = _row(t, "0.1.0", 1, ver_rid=1)
    bad = _row(t, "not-a-version", 2, ver_rid=2)
    ml = _mixin_with_version_rows([bad, good])
    out = ml._dataset_version_rows(_rid(500))
    assert out[0]["Version"] == "0.1.0"


def test_two_unparseable_labels_fall_back_to_raw_string_order():
    t = "2025-01-01T00:00:00Z"
    a = _row(t, "aaa-bad", 1, ver_rid=1)
    b = _row(t, "bbb-bad", 2, ver_rid=2)
    ml = _mixin_with_version_rows([b, a])
    out = ml._dataset_version_rows(_rid(500))
    assert [r["Version"] for r in out] == ["aaa-bad", "bbb-bad"]


def test_normalize_equal_labels_resolve_by_raw_string():
    """'1.0' and '1.0.0' parse equal — raw string breaks the tie, no crash."""
    t = "2025-01-01T00:00:00Z"
    short = _row(t, "1.0", 1, ver_rid=1)
    long = _row(t, "1.0.0", 2, ver_rid=2)
    ml = _mixin_with_version_rows([long, short])
    out = ml._dataset_version_rows(_rid(500))
    assert [r["Version"] for r in out] == ["1.0", "1.0.0"]


def test_missing_rct_sorts_first():
    """A row with no usable RCT is treated as epoch-minimum."""
    no_rct = {"RID": _rid(1001), "RCT": None, "Version": "5.0.0", "Execution": None, "Description": None}
    dated = _row("2025-01-01T00:00:00Z", "0.1.0", 1, ver_rid=2)
    ml = _mixin_with_version_rows([dated, no_rct])
    out = ml._dataset_version_rows(_rid(500))
    assert out[0]["RID"] == no_rct["RID"]


def test_single_fetch_per_call():
    ml = _mixin_with_version_rows([_row("2025-01-01T00:00:00Z", "0.1.0", 1)])
    pb = ml.pathBuilder()
    ml._dataset_version_rows(_rid(500))
    assert pb.schemas["deriva-ml"].tables["Dataset_Version"].filter.call_count == 1


def test_sort_key_is_module_level_and_total():
    """Direct key checks: flag separates parseable from not at equal RCT."""
    t = "2025-01-01T00:00:00Z"
    k_good = _version_row_sort_key({"RCT": t, "Version": "1.0.0"})
    k_bad = _version_row_sort_key({"RCT": t, "Version": "junk"})
    assert k_good < k_bad
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_dataset_version_rows.py -q -p no:randomly`
Expected: FAIL — `ImportError: cannot import name '_version_row_sort_key'`.

- [ ] **Step 3: Implement**

In `src/deriva_ml/core/mixins/execution.py`, add near the top of the module (after existing imports):

```python
from packaging.version import InvalidVersion, Version as _PEP440Version
```

Add a module-level function (place directly above the `ExecutionMixin` class or near `_producer_of_dataset`'s section):

```python
def _version_row_sort_key(row: dict[str, Any]) -> tuple:
    """Total sort key for a ``Dataset_Version`` row (spec: issue #367).

    Ordering is ``(RCT, parseable-flag, parsed PEP 440 version, raw label)``:

    - ``RCT`` primary: chronology of *recording*, which is the question the
      origin resolution asks. ERMrest emits RCT as ISO-8601 UTC text, so
      lexicographic comparison is chronological; a missing RCT sorts first
      (epoch-minimum) and defers to the tiebreaks.
    - Parseable labels sort before unparseable ones within an RCT tie; among
      parseable ones PEP 440 order applies (``0.2.0`` < ``0.10.0``).
    - The raw label string is the final tiebreak, making the order total even
      for two unparseable labels or labels that normalize equal
      (``1.0`` vs ``1.0.0``).

    Args:
        row: A ``Dataset_Version`` row dict (``RCT``, ``Version`` keys used).

    Returns:
        A tuple comparable against any other row's key.

    Example:
        >>> _version_row_sort_key({"RCT": "2025-01-01T00:00:00Z", "Version": "1.0.0"}) < \
        ...     _version_row_sort_key({"RCT": "2026-01-01T00:00:00Z", "Version": "0.1.0"})
        True
    """
    rct = row.get("RCT") or ""
    label = row.get("Version") or ""
    try:
        return (rct, 0, _PEP440Version(label), label)
    except InvalidVersion:
        # Constant Version("0") in slot 3: never compared against a parseable
        # row's key (flag differs), and equal among unparseable rows so the
        # raw label decides.
        return (rct, 1, _PEP440Version("0"), label)
```

Add the method to `ExecutionMixin` (directly above `_producer_of_dataset`):

```python
def _dataset_version_rows(self, dataset_rid: RID) -> list[dict[str, Any]]:
    """All ``Dataset_Version`` rows for ``dataset_rid``, earliest first.

    Single catalog fetch; sorted by :func:`_version_row_sort_key` (RCT-primary
    total order). The first row's ``Execution`` is the dataset's origin
    attribution; the full list becomes the lineage root's version history.

    Args:
        dataset_rid: Dataset whose version history to fetch.

    Returns:
        Row dicts (system columns included), sorted earliest → latest.
        Empty list if the dataset has no version rows.

    Example:
        >>> rows = ml._dataset_version_rows("1-DSAA")  # doctest: +SKIP
        >>> origin_rid = rows[0]["Execution"] if rows else None  # doctest: +SKIP
    """
    pb = self.pathBuilder()
    version_path = pb.schemas[self.ml_schema].tables["Dataset_Version"]
    rows = list(version_path.filter(version_path.Dataset == dataset_rid).entities().fetch())
    return sorted(rows, key=_version_row_sort_key)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_dataset_version_rows.py -q -p no:randomly`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/core/mixins/execution.py tests/execution/test_dataset_version_rows.py && git commit -m "feat(lineage): _dataset_version_rows with RCT-primary total ordering (#367)"
```

---

### Task 2: Models — `VersionAttribution`, `RootDescriptor` fields, exports

**Files:**
- Modify: `src/deriva_ml/execution/lineage.py` (add `VersionAttribution` after `AssetSummary` ~line 106; extend `RootDescriptor` ~line 139; update `RootDescriptor` and `LineageResult` docstrings)
- Modify: `src/deriva_ml/execution/__init__.py` (import ~line 40, `__all__` ~line 104)
- Test: `tests/execution/test_lineage_models.py` (new)

**Interfaces:**
- Consumes: existing `ExecutionSummary`, `RID`, Pydantic base pattern in `lineage.py`.
- Produces: `VersionAttribution(version: str, execution_rid: RID | None, execution: ExecutionSummary | None, description: str | None)`; `RootDescriptor.origin_recorded: bool | None = None`; `RootDescriptor.version_history: list[VersionAttribution]` (default empty). Task 5 constructs these.

- [ ] **Step 1: Write the failing tests**

Create `tests/execution/test_lineage_models.py`:

```python
"""Model + export pins for the #367 lineage additions."""

from __future__ import annotations


def _rid(n: int) -> str:
    return f"1-{n:04X}"


def test_version_attribution_importable_from_package():
    from deriva_ml.execution import VersionAttribution  # noqa: F401


def test_version_attribution_in_all():
    import deriva_ml.execution as pkg

    assert "VersionAttribution" in pkg.__all__


def test_root_descriptor_new_fields_default():
    from deriva_ml.execution.lineage import RootDescriptor

    root = RootDescriptor(rid=_rid(1), type="Asset", description=None)
    assert root.origin_recorded is None
    assert root.version_history == []


def test_version_attribution_carries_raw_rid_separately():
    from deriva_ml.execution import VersionAttribution

    entry = VersionAttribution(
        version="0.1.0", execution_rid=_rid(2), execution=None, description=None
    )
    assert entry.execution_rid == _rid(2)
    assert entry.execution is None  # recorded but unresolved


def test_model_dump_is_additive():
    """Serialized root carries the new keys alongside every old one."""
    from deriva_ml.execution.lineage import RootDescriptor

    d = RootDescriptor(rid=_rid(1), type="Dataset", description=None).model_dump()
    for key in ("rid", "type", "description", "producing_execution",
                "origin_recorded", "version_history"):
        assert key in d
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_lineage_models.py -q -p no:randomly`
Expected: FAIL — `ImportError: cannot import name 'VersionAttribution'`.

- [ ] **Step 3: Implement**

In `src/deriva_ml/execution/lineage.py`, after `AssetSummary` and before `LineageNode`:

```python
class VersionAttribution(BaseModel):
    """One entry in a dataset root's version-attribution trace.

    The trace lists every ``Dataset_Version`` row for the dataset,
    earliest recorded first, so a consumer can see who authored each
    version — distinguishing the origin (first entry) from later
    touchers such as migrations or backfills.

    Attributes:
        version: The version label as stored (e.g. ``"4.13.0"``).
        execution_rid: Raw ``Execution`` column value — the recorded
            author RID, or None if the row carries no author. Kept
            separate from ``execution`` so "no author recorded" and
            "author could not be resolved" are distinguishable.
        execution: Resolved summary of the author, or None when
            ``execution_rid`` is None or the lookup could not resolve it.
        description: The version's release notes.
    """

    model_config = ConfigDict(extra="forbid")

    version: str
    execution_rid: RID | None = None
    execution: ExecutionSummary | None = None
    description: str | None = None
```

Extend `RootDescriptor` — add after `producing_execution`:

```python
    origin_recorded: bool | None = None
    version_history: list[VersionAttribution] = Field(default_factory=list)
```

Replace the `RootDescriptor` docstring's `producing_execution` attribute line and add the new attributes (full docstring body):

```python
    """The artifact lineage was requested for.

    Attributes:
        rid: The root artifact's RID.
        type: Artifact kind — Dataset, Asset, Feature, or Execution.
        description: The artifact's description, if any.
        producing_execution: For datasets, the ORIGIN — the author of the
            first-recorded ``Dataset_Version`` row (the unknown-provenance
            sentinel included, when that is what the row records). For other
            types, the immediate producing execution. May be None when no
            producer is recorded or the recorded RID cannot be resolved; for
            datasets the raw recorded RID then remains available at
            ``version_history[0].execution_rid``. This is origin
            *attribution* and no longer necessarily equals the walk root
            ``LineageResult.lineage.execution`` (see that model's docstring).
        origin_recorded: Datasets only — True when a real (non-sentinel)
            origin execution is recorded (even if its summary could not be
            resolved); False when the origin is the unknown-provenance
            sentinel, the first version row carries no execution, or the
            dataset has no version rows; None for non-Dataset roots (not
            applicable).
        version_history: Datasets only — the full version-attribution trace,
            earliest recorded first. Empty for non-Dataset roots.
    """
```

Update `LineageResult`'s docstring: replace any sentence stating the lineage is rooted at the root's immediate producer with:

```
        lineage: The walked graph. Its root node is the walk seed — for
            datasets this is the origin execution when it is real and
            expandable, otherwise a member-producer representative (the
            unknown-provenance sentinel never seeds the walk). It therefore
            does not necessarily equal ``root.producing_execution``, which is
            origin attribution.
```

In `src/deriva_ml/execution/__init__.py`: add `VersionAttribution` to the `from deriva_ml.execution.lineage import (...)` block (alphabetical, after `RootDescriptor`) and to `__all__` (alphabetical).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_lineage_models.py tests/execution/test_lookup_lineage_unit.py -q -p no:randomly`
Expected: all pass (existing lineage unit tests must not break — new fields are optional with defaults).

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/execution/lineage.py src/deriva_ml/execution/__init__.py tests/execution/test_lineage_models.py && git commit -m "feat(lineage): VersionAttribution model + RootDescriptor origin fields (#367)"
```

---

### Task 3: `_execution_summaries()` — batched author resolution

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py` (new method + module constant, place after `_dataset_version_rows`)
- Test: `tests/execution/test_execution_summaries_batch.py` (new)

**Interfaces:**
- Consumes: `self.pathBuilder()`, `self.ml_schema`; `ExecutionSummary` / `WorkflowSummary` from `deriva_ml.execution.lineage`.
- Produces: `_execution_summaries(self, rids: Iterable[RID | None]) -> dict[RID, ExecutionSummary]` — deduped, chunked (`_SUMMARY_CHUNK_SIZE = 25`) Execution fetch plus one chunked Workflow-name fetch; missing RIDs simply absent from the dict. Task 5 calls this once per dataset root.

- [ ] **Step 1: Write the failing tests**

Create `tests/execution/test_execution_summaries_batch.py`:

```python
"""Query-count + shape pins for ExecutionMixin._execution_summaries (#367).

Offline. RIDs are programmatically generated; the mock pathBuilder records
filter() calls so the tests assert request counts, not URL syntax.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from deriva_ml.core.mixins.execution import _SUMMARY_CHUNK_SIZE, ExecutionMixin


def _rid(n: int) -> str:
    return f"1-{n:04X}"


class _Recorder:
    """Table mock: records filter() calls, returns scripted rows."""

    def __init__(self, rows_by_rid: dict):
        self.rows_by_rid = rows_by_rid
        self.filter_calls = 0
        # Predicate operands: table.RID == rid must yield an object carrying
        # the rid so fetch can resolve which rows were requested. A MagicMock
        # __eq__ returning a tagged object does this.
        self.RID = MagicMock()
        self._requested: list[list[str]] = []
        self.RID.__eq__ = lambda _self, other: _Pred([other])

    def filter(self, pred):
        self.filter_calls += 1
        self._requested.append(pred.rids)
        result = MagicMock()
        rows = [self.rows_by_rid[r] for r in pred.rids if r in self.rows_by_rid]
        result.entities.return_value.fetch.return_value = rows
        return result


class _Pred:
    def __init__(self, rids):
        self.rids = list(rids)

    def __or__(self, other):
        return _Pred(self.rids + other.rids)


def _mixin(exec_rows: dict, wf_rows: dict):
    ml = ExecutionMixin.__new__(ExecutionMixin)
    ml.ml_schema = "deriva-ml"
    exec_table = _Recorder(exec_rows)
    wf_table = _Recorder(wf_rows)
    schema = MagicMock()
    schema.tables = {"Execution": exec_table, "Workflow": wf_table}
    pb = MagicMock()
    pb.schemas = {"deriva-ml": schema}
    ml.pathBuilder = lambda: pb
    return ml, exec_table, wf_table


def _exec_row(n: int, wf_n: int | None = None) -> dict:
    return {
        "RID": _rid(n),
        "Description": f"exec {n}",
        "Status": "Completed",
        "Workflow": _rid(wf_n) if wf_n is not None else None,
    }


def test_empty_and_all_none_issue_zero_queries():
    ml, exec_table, wf_table = _mixin({}, {})
    assert ml._execution_summaries([]) == {}
    assert ml._execution_summaries([None, None]) == {}
    assert exec_table.filter_calls == 0
    assert wf_table.filter_calls == 0


def test_duplicates_dedupe_to_one_lookup_entry():
    row = _exec_row(1)
    ml, exec_table, _ = _mixin({row["RID"]: row}, {})
    out = ml._execution_summaries([row["RID"], row["RID"], row["RID"]])
    assert exec_table.filter_calls == 1
    assert list(out) == [row["RID"]]


def test_chunk_boundary_two_requests():
    rows = {(_r := _exec_row(i))["RID"]: _r for i in range(_SUMMARY_CHUNK_SIZE + 1)}
    ml, exec_table, _ = _mixin(rows, {})
    out = ml._execution_summaries(list(rows))
    assert exec_table.filter_calls == 2
    assert len(out) == _SUMMARY_CHUNK_SIZE + 1


def test_workflow_names_batched_not_per_execution():
    wf = {"RID": _rid(900), "Name": "training workflow"}
    rows = {(_r := _exec_row(i, wf_n=900))["RID"]: _r for i in range(1, 4)}
    ml, exec_table, wf_table = _mixin(rows, {wf["RID"]: wf})
    out = ml._execution_summaries(list(rows))
    assert wf_table.filter_calls == 1  # 3 executions, 1 distinct workflow, 1 query
    assert all(s.workflow is not None and s.workflow.name == "training workflow" for s in out.values())


def test_unresolvable_rid_absent_from_result():
    row = _exec_row(1)
    ml, _, _ = _mixin({row["RID"]: row}, {})
    out = ml._execution_summaries([row["RID"], _rid(999)])
    assert _rid(999) not in out
    assert row["RID"] in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_summaries_batch.py -q -p no:randomly`
Expected: FAIL — `ImportError: cannot import name '_SUMMARY_CHUNK_SIZE'`.

- [ ] **Step 3: Implement**

In `src/deriva_ml/core/mixins/execution.py`, module level:

```python
# Chunk size for batched RID-disjunction fetches. Bounds URL length (the
# tk-023 lesson: never one unbounded .in_()/disjunction over a caller-sized
# set) while keeping request count low for realistic author counts.
_SUMMARY_CHUNK_SIZE = 25
```

Method on `ExecutionMixin`:

```python
def _execution_summaries(self, rids: "Iterable[RID | None]") -> dict[RID, "ExecutionSummary"]:
    """Resolve ExecutionSummary objects for a set of execution RIDs, batched.

    Dedupes and drops None, then fetches Execution rows in chunks of
    ``_SUMMARY_CHUNK_SIZE`` (one RID-equality disjunction per chunk), then
    resolves the distinct Workflow names the same way — never a
    per-execution N+1. RIDs that resolve to no row are absent from the
    result; callers treat absence as "could not resolve".

    Args:
        rids: Execution RIDs (Nones ignored, duplicates collapsed).

    Returns:
        Mapping of execution RID to its ExecutionSummary.

    Example:
        >>> summaries = ml._execution_summaries(["2-ABC", "2-DEF"])  # doctest: +SKIP
        >>> summaries["2-ABC"].status  # doctest: +SKIP
        'Completed'
    """
    from deriva_ml.execution.lineage import ExecutionSummary, WorkflowSummary

    distinct = list(dict.fromkeys(r for r in rids if r))
    if not distinct:
        return {}

    pb = self.pathBuilder()

    def _chunked_rows(table_path: Any, wanted: list[RID]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for i in range(0, len(wanted), _SUMMARY_CHUNK_SIZE):
            chunk = wanted[i : i + _SUMMARY_CHUNK_SIZE]
            pred = table_path.RID == chunk[0]
            for r in chunk[1:]:
                pred = pred | (table_path.RID == r)
            rows.extend(table_path.filter(pred).entities().fetch())
        return rows

    exec_path = pb.schemas[self.ml_schema].tables["Execution"]
    exec_rows = _chunked_rows(exec_path, distinct)

    wf_rids = list(dict.fromkeys(row.get("Workflow") for row in exec_rows if row.get("Workflow")))
    wf_names: dict[RID, str | None] = {}
    if wf_rids:
        wf_path = pb.schemas[self.ml_schema].tables["Workflow"]
        for wrow in _chunked_rows(wf_path, wf_rids):
            wf_names[wrow["RID"]] = wrow.get("Name")

    out: dict[RID, ExecutionSummary] = {}
    for row in exec_rows:
        wf_rid = row.get("Workflow")
        wf_summary = WorkflowSummary(rid=wf_rid, name=wf_names.get(wf_rid)) if wf_rid else None
        out[row["RID"]] = ExecutionSummary(
            rid=row["RID"],
            description=row.get("Description"),
            workflow=wf_summary,
            status=row.get("Status") or "Unknown",
        )
    return out
```

(`Iterable` comes from `typing`/`collections.abc` — add to the module's imports if not present.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_summaries_batch.py -q -p no:randomly`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/core/mixins/execution.py tests/execution/test_execution_summaries_batch.py && git commit -m "feat(lineage): batched _execution_summaries with chunked workflow resolution (#367)"
```

---

### Task 4: Non-raising sentinel lookup with positive-only cache

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py` (new method `_sentinel_execution_rid_or_none`; reimplement `unknown_provenance_execution_rid` ~line 75 on top of it)
- Test: `tests/execution/test_sentinel_lookup.py` (new)

**Interfaces:**
- Consumes: `SENTINEL_EXECUTION_DESCRIPTION` from `deriva_ml.core.constants`; `self.pathBuilder()`.
- Produces: `_sentinel_execution_rid_or_none(self) -> RID | None` — None when the sentinel row is absent (catalog never adopted the contract), positive result cached on `self._sentinel_exec_rid`; transport errors propagate. `unknown_provenance_execution_rid()` keeps its exact public contract (raises `DerivaMLException` on absence).

- [ ] **Step 1: Write the failing tests**

Create `tests/execution/test_sentinel_lookup.py`:

```python
"""Pins for the non-raising sentinel lookup + positive-only cache (#367)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.core.mixins.execution import ExecutionMixin


def _rid(n: int) -> str:
    return f"1-{n:04X}"


def _mixin_with_sentinel_rows(rows: list[dict]):
    ml = ExecutionMixin.__new__(ExecutionMixin)
    ml.ml_schema = "deriva-ml"
    exe = MagicMock()
    exe.filter.return_value.entities.return_value = iter(rows)
    schema = MagicMock()
    schema.Execution = exe
    pb = MagicMock()
    pb.schemas = {"deriva-ml": schema}
    ml.pathBuilder = lambda: pb
    return ml, exe


def test_absent_returns_none_without_raising():
    ml, _ = _mixin_with_sentinel_rows([])
    assert ml._sentinel_execution_rid_or_none() is None


def test_absence_is_not_cached():
    ml, exe = _mixin_with_sentinel_rows([])
    ml._sentinel_execution_rid_or_none()
    exe.filter.return_value.entities.return_value = iter([{"RID": _rid(7)}])
    assert ml._sentinel_execution_rid_or_none() == _rid(7)  # re-queried


def test_positive_result_cached_no_second_query():
    ml, exe = _mixin_with_sentinel_rows([{"RID": _rid(7)}])
    first = ml._sentinel_execution_rid_or_none()
    second = ml._sentinel_execution_rid_or_none()
    assert first == second == _rid(7)
    assert exe.filter.call_count == 1


def test_transport_error_propagates():
    ml, exe = _mixin_with_sentinel_rows([])
    exe.filter.side_effect = ConnectionError("catalog unreachable")
    with pytest.raises(ConnectionError):
        ml._sentinel_execution_rid_or_none()


def test_public_raising_wrapper_unchanged():
    ml, _ = _mixin_with_sentinel_rows([])
    with pytest.raises(DerivaMLException, match="sentinel"):
        ml.unknown_provenance_execution_rid()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_sentinel_lookup.py -q -p no:randomly`
Expected: FAIL — `AttributeError: ... has no attribute '_sentinel_execution_rid_or_none'`.

- [ ] **Step 3: Implement**

In `src/deriva_ml/core/mixins/execution.py`, add above `unknown_provenance_execution_rid`:

```python
def _sentinel_execution_rid_or_none(self) -> RID | None:
    """RID of the unknown-provenance Execution sentinel, or None if absent.

    Non-raising variant used by lineage classification: absence (a catalog
    that never adopted the provenance contract) is a normal answer, not an
    error, so it returns None rather than raising — transport and auth
    failures still propagate. The positive result is cached on the instance
    (``self._sentinel_exec_rid``); absence is never cached, because contract
    adoption can happen during the instance lifetime.

    Returns:
        The sentinel Execution's RID, or None when no sentinel row exists.

    Example:
        >>> rid = ml._sentinel_execution_rid_or_none()  # doctest: +SKIP
    """
    cached = getattr(self, "_sentinel_exec_rid", None)
    if cached is not None:
        return cached
    from deriva_ml.core.constants import SENTINEL_EXECUTION_DESCRIPTION

    pb = self.pathBuilder()
    exe = pb.schemas[self.ml_schema].Execution
    rows = list(exe.filter(exe.Description == SENTINEL_EXECUTION_DESCRIPTION).entities())
    if not rows:
        return None
    self._sentinel_exec_rid = rows[0]["RID"]
    return self._sentinel_exec_rid
```

Reimplement the body of `unknown_provenance_execution_rid` (keep its docstring):

```python
    rid = self._sentinel_execution_rid_or_none()
    if rid is None:
        raise DerivaMLException(
            "Unknown-provenance Execution sentinel not found; catalog was not "
            "initialized with provenance-contract sentinels."
        )
    return rid
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_sentinel_lookup.py -q -p no:randomly`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/core/mixins/execution.py tests/execution/test_sentinel_lookup.py && git commit -m "feat(lineage): non-raising sentinel lookup with positive-only cache (#367)"
```

---

### Task 5: Origin resolution — `_producer_of_dataset` + `_classify_rid` dataset branch

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py` (`_producer_of_dataset` ~line 1322: replace the `max(rows)` unversioned branch; `_classify_rid` Dataset branch ~line 1277: build full origin state)
- Test: `tests/execution/test_lookup_lineage_unit.py` (extend — this module owns root-classification behavior)

**Interfaces:**
- Consumes: `_dataset_version_rows` (Task 1), `_execution_summaries` (Task 3), `_sentinel_execution_rid_or_none` (Task 4), `VersionAttribution` (Task 2).
- Produces: `_classify_rid` Dataset branch returns a fully-populated `RootDescriptor` (origin `producing_execution`, `origin_recorded`, `version_history`) and a **walk-seed** `producer_rid` that is None when the origin is absent or the sentinel. `_producer_of_dataset(rid)` (unversioned) returns the FIRST-recorded row's `Execution`; the `version=` path is untouched.

- [ ] **Step 1: Write the failing tests**

Append to `tests/execution/test_lookup_lineage_unit.py` (follow that module's existing stub conventions; RIDs via a local `_gen_rid(n)` helper, programmatically generated):

```python
# ---------------------------------------------------------------------------
# #367: origin resolution (first-recorded, not last-writer)
# ---------------------------------------------------------------------------


def _gen_rid(n: int) -> str:
    return f"1-{n:04X}"


_version_row_ids = iter(range(0x100, 0xFFF))


def _version_row(rct: str, version: str, exec_rid: str | None) -> dict:
    return {"RID": _gen_rid(next(_version_row_ids)), "RCT": rct,
            "Version": version, "Execution": exec_rid, "Description": None}


def _origin_ml(version_rows, sentinel_rid=None, summaries=None):
    """ExecutionMixin stub wired for the dataset branch of _classify_rid."""
    ml = ExecutionMixin.__new__(ExecutionMixin)
    ml.ml_schema = "deriva-ml"
    ml.resolve_rid = lambda rid: _StubResolved(table=_StubTable(name="Dataset", columns=[]))
    ml._retrieve_rid = lambda rid: {"RID": rid, "Description": "a dataset"}
    ml._dataset_version_rows = lambda rid: list(version_rows)
    ml._sentinel_execution_rid_or_none = lambda: sentinel_rid
    ml._execution_summaries = lambda rids: dict(summaries or {})
    return ml


def test_unversioned_producer_is_first_recorded_not_latest():
    origin_exec, migration_exec = _gen_rid(10), _gen_rid(11)
    rows = [
        _version_row("2025-01-01T00:00:00Z", "0.1.0", origin_exec),
        _version_row("2026-01-01T00:00:00Z", "4.13.0", migration_exec),
    ]
    ml = _origin_ml(rows)
    descriptor, walk_seed = ml._classify_rid(_gen_rid(1))
    assert walk_seed == origin_exec
    assert descriptor.origin_recorded is True
    assert descriptor.version_history[0].execution_rid == origin_exec
    assert descriptor.version_history[-1].execution_rid == migration_exec


def test_sentinel_origin_reported_but_never_walk_seed():
    sentinel = _gen_rid(66)
    rows = [_version_row("2025-01-01T00:00:00Z", "0.1.0", sentinel)]
    ml = _origin_ml(rows, sentinel_rid=sentinel)
    descriptor, walk_seed = ml._classify_rid(_gen_rid(1))
    assert walk_seed is None            # sentinel never seeds the walk
    assert descriptor.origin_recorded is False


def test_no_execution_on_first_row():
    rows = [_version_row("2025-01-01T00:00:00Z", "0.1.0", None)]
    ml = _origin_ml(rows)
    descriptor, walk_seed = ml._classify_rid(_gen_rid(1))
    assert walk_seed is None
    assert descriptor.origin_recorded is False
    assert descriptor.producing_execution is None
    assert descriptor.version_history[0].execution_rid is None


def test_no_version_rows():
    ml = _origin_ml([])
    descriptor, walk_seed = ml._classify_rid(_gen_rid(1))
    assert walk_seed is None
    assert descriptor.origin_recorded is False
    assert descriptor.version_history == []


def test_recorded_but_unresolvable_origin_keeps_recorded_true():
    origin_exec = _gen_rid(10)
    rows = [_version_row("2025-01-01T00:00:00Z", "0.1.0", origin_exec)]
    ml = _origin_ml(rows, summaries={})  # summary resolution came back empty
    descriptor, walk_seed = ml._classify_rid(_gen_rid(1))
    assert descriptor.origin_recorded is True
    assert descriptor.producing_execution is None
    assert descriptor.version_history[0].execution_rid == origin_exec
    assert walk_seed == origin_exec
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_lookup_lineage_unit.py -q -p no:randomly -k "367 or first_recorded or sentinel_origin or unresolvable or no_version_rows or no_execution_on_first"`
Expected: the five new tests FAIL (current code resolves via `_producer_of_dataset` → `max(rows)` and never sets `origin_recorded`/`version_history`).

- [ ] **Step 3: Implement**

In `_producer_of_dataset`, replace the unversioned branch (delete the local `_key` function and `latest = max(rows, key=_key)`):

```python
        # Unversioned: the ORIGIN — the first-recorded row's author (issue
        # #367). Last-writer-wins was the old behavior and reported whatever
        # touched the dataset most recently (e.g. a data migration) as "the
        # producer".
        ordered = sorted(rows, key=_version_row_sort_key)
        return ordered[0].get("Execution")
```

(Keep the single fetch already in the method; sorting reuses Task 1's key. The `version=` branch is untouched.)

Replace the `_classify_rid` Dataset branch (currently `producer_rid = self._producer_of_dataset(rid)` and a bare `RootDescriptor`) with:

```python
        if table_name == "Dataset":
            from deriva_ml.execution.lineage import VersionAttribution

            row = self._retrieve_rid(rid)
            version_rows = self._dataset_version_rows(rid)
            origin_rid = version_rows[0].get("Execution") if version_rows else None
            sentinel_rid = self._sentinel_execution_rid_or_none()
            origin_is_sentinel = origin_rid is not None and origin_rid == sentinel_rid
            author_summaries = self._execution_summaries(r.get("Execution") for r in version_rows)
            history = [
                VersionAttribution(
                    version=r.get("Version") or "",
                    execution_rid=r.get("Execution"),
                    execution=author_summaries.get(r.get("Execution")),
                    description=r.get("Description"),
                )
                for r in version_rows
            ]
            descriptor = RootDescriptor(
                rid=rid,
                type="Dataset",
                description=row.get("Description"),
                producing_execution=author_summaries.get(origin_rid) if origin_rid else None,
                origin_recorded=bool(origin_rid) and not origin_is_sentinel,
                version_history=history,
            )
            # Walk seed: the origin only when real and non-sentinel. The
            # sentinel never seeds the walk — lineage terminates at it, and
            # seeding from it would fabricate edges claiming it consumed the
            # member producers (spec: walk seed vs. origin attribution).
            walk_seed = origin_rid if (origin_rid and not origin_is_sentinel) else None
            return (descriptor, walk_seed)
```

Note: `_classify_rid`'s single caller is `lookup_lineage`; the sole other consumer of the unversioned `_producer_of_dataset` was this branch, so the two changes land together.

- [ ] **Step 4: Run tests to verify they pass, and that nothing else broke**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_lookup_lineage_unit.py tests/execution/test_dataset_version_rows.py tests/execution/test_producers_of_dataset_members.py tests/execution/test_forward_lineage.py -q -p no:randomly`
Expected: all pass. If an existing lookup_lineage unit test stubs `_producer_of_dataset` and asserts dataset-root behavior, update its stub to the new seam (`_dataset_version_rows` + `_execution_summaries` + `_sentinel_execution_rid_or_none`) — the walk-shape assertions themselves must keep passing unmodified.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/core/mixins/execution.py tests/execution/test_lookup_lineage_unit.py && git commit -m "feat(lineage): dataset root resolves origin, not last writer (#367)"
```

---

### Task 6: Walk seeding — sentinel exclusion, no overwrite, degradation

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py` (`lookup_lineage` walk-seeding block ~lines 1190–1232)
- Test: `tests/execution/test_lookup_lineage_unit.py` (extend)

**Interfaces:**
- Consumes: `(descriptor, walk_seed)` from Task 5's `_classify_rid`; existing `_producers_of_dataset_members`, `_walk_node`.
- Produces: final `lookup_lineage` behavior — `root.producing_execution` never overwritten for Dataset roots; member-producer seeding when `walk_seed` is None **or** the seed proves unexpandable; non-Dataset roots keep today's overwrite.

- [ ] **Step 1: Write the failing tests**

Append to `tests/execution/test_lookup_lineage_unit.py`. These drive `lookup_lineage` end-to-end over stubs, following the module's existing full-walk test pattern (stub `resolve_rid`, `_retrieve_rid`, `lookup_execution`, `_producers_of_dataset_members`, plus the Task-5 seams):

```python
def _walkable_ml(version_rows, member_producers, known_execs, sentinel_rid=None, summaries=None):
    """Full lookup_lineage stub: dataset root + walkable executions."""
    ml = _origin_ml(version_rows, sentinel_rid=sentinel_rid, summaries=summaries)
    ml._producers_of_dataset_members = lambda rid, version=None: set(member_producers)
    ml._input_dataset_pairs = lambda rid: []

    def fake_lookup_execution(rid):
        if rid not in known_execs:
            raise DerivaMLException(f"no execution {rid}")
        rec = MagicMock()
        rec.description = f"exec {rid}"
        rec.workflow = None
        rec.status = ExecutionStatus.completed
        rec.list_assets = lambda **kw: []  # consumed-asset iteration is on the record
        return rec

    ml.lookup_execution = fake_lookup_execution
    return ml


def test_member_fallback_seeds_walk_but_is_not_origin():
    member = _gen_rid(30)
    rows = [_version_row("2025-01-01T00:00:00Z", "0.1.0", None)]  # no origin
    ml = _walkable_ml(rows, {member}, {member})
    result = ml.lookup_lineage(_gen_rid(1))
    assert result.lineage is not None
    assert result.lineage.execution.rid == member          # walk seeded
    assert result.root.producing_execution is None         # origin NOT overwritten
    assert result.root.origin_recorded is False


def test_sentinel_origin_walk_seeds_from_member_producers():
    sentinel, member = _gen_rid(66), _gen_rid(30)
    rows = [_version_row("2025-01-01T00:00:00Z", "0.1.0", sentinel)]
    ml = _walkable_ml(rows, {member}, {member}, sentinel_rid=sentinel)
    result = ml.lookup_lineage(_gen_rid(1))
    assert result.lineage is not None
    assert result.lineage.execution.rid == member          # not the sentinel
    assert result.root.origin_recorded is False


def test_unexpandable_origin_degrades_to_member_seeding():
    origin, member = _gen_rid(10), _gen_rid(30)
    rows = [_version_row("2025-01-01T00:00:00Z", "0.1.0", origin)]
    ml = _walkable_ml(rows, {member}, known_execs={member})  # origin not expandable
    result = ml.lookup_lineage(_gen_rid(1))
    assert result.lineage is not None
    assert result.lineage.execution.rid == member
    assert result.root.origin_recorded is True             # origin still recorded


def test_normal_recorded_origin_walks_from_origin():
    origin, member = _gen_rid(10), _gen_rid(30)
    rows = [_version_row("2025-01-01T00:00:00Z", "0.1.0", origin)]
    ml = _walkable_ml(rows, {member}, {origin, member})
    result = ml.lookup_lineage(_gen_rid(1))
    assert result.lineage.execution.rid == origin
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_lookup_lineage_unit.py -q -p no:randomly -k "member_fallback or sentinel_origin_walk or unexpandable or normal_recorded"`
Expected: FAIL — `test_member_fallback...` fails on the overwrite (producing_execution set to the member); `test_unexpandable...` fails with `lineage is None`.

- [ ] **Step 3: Implement**

In `lookup_lineage`, restructure the block between classification and result assembly (current lines ~1190–1232):

```python
        # Member-producer discovery (tk-018): executions that produced member
        # assets are data-flow parents even when the dataset itself has no
        # usable walk seed.
        member_producers: set[RID] = set()
        extra_parent_rids: set[RID] = set()
        if root_descriptor.type == "Dataset":
            member_producers = self._producers_of_dataset_members(rid)
            if producer_rid is not None:
                extra_parent_rids = member_producers - {producer_rid}
            elif member_producers:
                ordered = sorted(member_producers)
                producer_rid = ordered[0]
                extra_parent_rids = set(ordered[1:])

        if producer_rid is None:
            return LineageResult(root=root_descriptor)

        visited_global: set[RID] = set()
        in_progress: set[RID] = set()
        flags = {"cycle_detected": False, "depth_capped": False, "walked_complete": True}

        lineage_root_node = self._walk_node(
            execution_rid=producer_rid,
            depth_remaining=depth,
            max_executions=max_executions,
            visited_global=visited_global,
            in_progress=in_progress,
            flags=flags,
            extra_parent_rids=extra_parent_rids or None,
        )

        # Degradation (spec: walk seed vs. origin attribution): a recorded
        # origin that cannot be expanded must not erase independently
        # resolvable member lineage.
        if (
            lineage_root_node is None
            and root_descriptor.type == "Dataset"
            and member_producers
            and producer_rid not in member_producers
        ):
            ordered = sorted(member_producers)
            lineage_root_node = self._walk_node(
                execution_rid=ordered[0],
                depth_remaining=depth,
                max_executions=max_executions,
                visited_global=visited_global,
                in_progress=in_progress,
                flags=flags,
                extra_parent_rids=set(ordered[1:]) or None,
            )

        # Non-Dataset roots keep the historical behavior: the root's
        # producing_execution is the walk-root summary. Dataset roots carry
        # origin attribution from _classify_rid and are never overwritten —
        # the walk root may legitimately differ (member-producer seeding).
        if lineage_root_node is not None and root_descriptor.type != "Dataset":
            root_descriptor = root_descriptor.model_copy(update={"producing_execution": lineage_root_node.execution})
```

(The `sorted(member_producers)` calls mirror the pre-existing deterministic-representative idiom — determinism only, no RID-order semantics.)

- [ ] **Step 4: Run the full lineage suite**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_lookup_lineage_unit.py tests/execution/test_forward_lineage.py tests/execution/test_producers_of_dataset_members.py tests/execution/test_dataset_version_rows.py tests/execution/test_execution_summaries_batch.py tests/execution/test_sentinel_lookup.py tests/execution/test_lineage_models.py -q -p no:randomly`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/core/mixins/execution.py tests/execution/test_lookup_lineage_unit.py && git commit -m "feat(lineage): separate walk seed from origin; sentinel never seeds (#367)"
```

---

### Task 7: `create_schema.py` comment correction

**Files:**
- Modify: `src/deriva_ml/schema/create_schema.py` (~line 286, `Dataset_Version.Execution` ColumnDef comment)

**Interfaces:**
- Consumes/Produces: nothing programmatic — a contract-surface comment fix. The schema-doc validator (`deriva-ml-validate-schema`) checks names/types/FKs, not comments, so `docs/reference/schema.md` needs no change — but run the validator to prove it.

- [ ] **Step 1: Fix the comment**

Replace:

```python
                comment=(
                    "RID of the execution that produced this version (NULL "
                    "for the initial release row, which has no producing "
                    "execution)."
                ),
```

with:

```python
                comment=(
                    "RID of the execution that authored this version. On the "
                    "initial release row this is the execution that created "
                    "the dataset (its origin); on later rows, whatever "
                    "execution triggered the version bump. NULL when no "
                    "authoring execution was recorded (pre-contract rows are "
                    "backfilled to the unknown-provenance sentinel)."
                ),
```

- [ ] **Step 2: Run the schema-doc validator**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run deriva-ml-validate-schema`
Expected: `schema.md and create_schema.py agree.`

- [ ] **Step 3: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/schema/create_schema.py && git commit -m "fix(schema): Dataset_Version.Execution comment contradicted provenance contract (#367)"
```

---

### Task 8: Documentation — `lookup_lineage` docstring + user guide

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py` (`lookup_lineage` docstring ~lines 1109–1175)
- Modify: `docs/user-guide/executions.md` (lineage result-shape prose around line 654's "No producer" paragraph)
- Modify: `docs/user-guide/reproducibility.md` (only if it describes the dataset-producer semantics — grep first; skip with a note if it merely mentions the method name)

**Interfaces:** none — documentation of Tasks 5–6 behavior.

- [ ] **Step 1: Update the `lookup_lineage` docstring**

In the docstring: remove/replace any sentence promising the latest version's author; add to the Returns section:

```
            For a Dataset root, ``root.producing_execution`` is the ORIGIN —
            the author of the first-recorded ``Dataset_Version`` row — not
            the latest writer. ``root.origin_recorded`` says whether that
            origin is a real recorded execution (False when it is the
            unknown-provenance sentinel or absent), and
            ``root.version_history`` lists every version's author, earliest
            first, so migrations and backfills appear as touchers rather
            than as "the producer". The walk (``result.lineage``) seeds from
            the origin when it is real and expandable, otherwise from member
            producers; the unknown-provenance sentinel never seeds the walk.
            ``depth=0`` bounds recursion but still resolves the root node
            and member producers — cheap, not free.
```

- [ ] **Step 2: Update `docs/user-guide/executions.md`**

In the lineage section (the "**No producer.**" paragraph at ~line 654 and the surrounding result-shape description): rewrite to state (a) dataset producer = origin (first-recorded author), (b) `origin_recorded` tri-state meaning, (c) `version_history` trace with the migration-as-toucher example from #367 (a data migration appears as the v4.13.0 author in the trace, not as the producer), (d) sentinel origins read "origin unrecorded" and never seed the walk. Keep the document's existing voice; do not paste RID literals from the issue — describe shapes generically.

- [ ] **Step 3: Check reproducibility.md**

Run: `grep -n "producing_execution\|producer" docs/user-guide/reproducibility.md`
If it describes producer semantics, align the wording the same way; if it only name-drops `lookup_lineage`, leave it.

- [ ] **Step 4: Doctest + docs build check**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest --doctest-modules src/deriva_ml/core/mixins/execution.py src/deriva_ml/execution/lineage.py -q -p no:randomly`
Expected: pass (skips are fine).

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/core/mixins/execution.py docs/user-guide/executions.md docs/user-guide/reproducibility.md && git commit -m "docs(lineage): origin semantics, version_history, walk-seed rules (#367)"
```

---

### Task 9: MCP compatibility verification (read-only)

**Files:**
- Read: `/Users/carl/GitHub/DerivaML/deriva-ml-mcp/` (the lineage tool wrapper — locate via `grep -rn "get_lineage\|lookup_lineage" ../deriva-ml-mcp/src`)

**Interfaces:** none — pre-release verification from the spec's Compatibility section.

- [ ] **Step 1: Inspect the MCP wrapper**

Run: `grep -rn "lookup_lineage\|get_lineage" /Users/carl/GitHub/DerivaML/deriva-ml-mcp/src | head -20`, then read the wrapper. Confirm it serializes via `model_dump()`/dict passthrough and does not re-validate the result against its own frozen Pydantic schema with `extra="forbid"`. Also grep its tests for lineage snapshot fixtures: `grep -rln "producing_execution" /Users/carl/GitHub/DerivaML/deriva-ml-mcp/tests`.

- [ ] **Step 2: Record the finding**

Write the outcome (compatible / needs a follow-up change in deriva-ml-mcp) into the PR description in Task 10. If the wrapper re-validates strictly, file an issue on deriva-ml-mcp — do not modify that repo in this PR.

---

### Task 10: Full verification + PR

**Files:** none new.

- [ ] **Step 1: Lint + format changed files**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src/deriva_ml/core/mixins/execution.py src/deriva_ml/execution/lineage.py src/deriva_ml/execution/__init__.py src/deriva_ml/schema/create_schema.py tests/execution/ && uv run ruff format --check src/deriva_ml/core/mixins/execution.py src/deriva_ml/execution/lineage.py src/deriva_ml/execution/__init__.py src/deriva_ml/schema/create_schema.py tests/execution/`
Expected: clean (run `ruff format` without `--check` if formatting drifted, then re-run).

- [ ] **Step 2: Offline unit suites**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/ tests/local_db/ tests/model/ tests/asset/ -q -p no:randomly --timeout=300`
Expected: no NEW failures versus the pre-change baseline (record the baseline first if unsure: `git stash && <same command> && git stash pop`). Catalog-connection errors from a stopped local Deriva are pre-existing and expected.

- [ ] **Step 3: Live smoke (only if `DERIVA_HOST` catalog is running)**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/test_lookup_lineage_live.py -q --timeout=600`
Expected: pass. Skip with a note in the PR if no catalog is available.

- [ ] **Step 4: Push and open the PR**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git push -u origin feat/dataset-origin-lineage-367
gh pr create --title "feat(lineage): dataset origin + version-attribution trace (#367)" --body "Closes #367. Implements docs/superpowers/specs/2026-08-30-dataset-origin-lineage-design.md (two Codex review rounds). Root producing_execution = first-recorded version author (RCT-primary total order); origin_recorded tri-state; version_history trace on the root; sentinel never seeds the walk; batched author summaries; create_schema.py comment contradiction fixed. MCP-compat verification: <Task 9 outcome>. Each behavioral test verified failing against the unfixed code.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

- [ ] **Step 5: Post-merge reminder (do not do this from the branch)**

After merge, on `main`: `uv run bump-version minor` (adds public API surface: `VersionAttribution`, new `RootDescriptor` fields). Verify the deriva-py pin is current first, per CLAUDE.md.

---

## Self-review notes (spec ↔ plan)

- Spec "Ordering rule (total)" → Task 1 (all four key components + missing-RCT rule + tests 1–8).
- Spec "Result shape" → Task 2 (model, tri-state, raw RID, exports, docstrings) + Task 5 (population).
- Spec "Walk seed vs. origin attribution" (all four seeding rules) → Tasks 5–6.
- Spec "Resolution mechanics" (single fetch, batching, chunking, reuse) → Tasks 1, 3, 5.
- Spec "Sentinel classification" → Task 4 (+ Task 5 usage).
- Spec "Compatibility" → Tasks 7 (schema comment), 8 (docs), 9 (MCP verification).
- Spec test list items 1–13 map to: 1→T5, 2/3→T1, 4→T5+T6, 5/6/7→T5, 8→T2+T3, 9→T6, 10→T3, 11→T4, 12→T4, 13→T2.
- Out-of-scope items (Dataset_Dataset pinning, clone RCT, ambiguity state) have no tasks — intentional.
