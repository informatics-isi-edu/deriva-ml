# Lineage Consumed-Version + Self-Parent Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `lookup_lineage`'s mid-walk consumed-dataset expansion reflect the version actually consumed (not current membership) and stop an execution that both consumed a dataset and produced some of its members from becoming its own parent.

**Architecture:** Add a new shared helper `list_input_datasets_with_versions` that surfaces the consumed `Dataset_Execution.Dataset_Version` (leaving `list_input_datasets`'s `list[Dataset]` contract untouched). Make `_producer_of_dataset` accept an optional `version`. Rewrite the `_walk_node` consumed-dataset loop to use the new helper, thread the consumed version into both producer lookups, report the consumed version in the summary, and subtract the currently-expanding execution from the member-producers it adds.

**Tech Stack:** Python 3.13, `uv`, pytest, deriva-ml core (`deriva_ml.core.mixins.execution.ExecutionMixin`, `deriva_ml.execution._helpers`), ERMrest PathBuilder.

## Global Constraints

- Work in `../deriva-ml` (`/Users/carl/GitHub/DerivaML/deriva-ml`) on branch `feature/lineage-consumed-version-fix` (already created; the spec is committed there).
- Use `uv` for everything: `uv run python -m pytest` (NOT `uv run pytest`), `uv run ruff check`, `uv run ruff format`.
- Always `cd /Users/carl/GitHub/DerivaML/deriva-ml && <cmd>` in every Bash call — the shell CWD is not persistent.
- Google-style docstrings (Args/Returns/Raises/Example) on every new function/method.
- NO change to `list_input_datasets()` (the existing `list[Dataset]` contract) or its four callers (`Execution`, `ExecutionRecord`, `provenance_enforcement`, `split.py`).
- NO public-model change: `src/deriva_ml/execution/lineage.py` is NOT modified. `consumed_datasets[].version` now reflects the consumed version (a correctness improvement to an existing field), not a shape change.
- The root path in `lookup_lineage` is UNCHANGED — a root dataset has no consumed-input edge, so no consumed version applies.
- `version=None` must preserve today's behavior exactly: `_producer_of_dataset(rid)` returns the latest-version producer; existing lineage unit tests stay green.
- Live tests gate on the `DERIVA_HOST` env var (the convention in `tests/execution/test_lookup_lineage_live.py`), NOT `DERIVA_ML_LIVE_LOCALHOST`.

---

### Task 1: `list_input_datasets_with_versions` shared helper

**Files:**
- Modify: `src/deriva_ml/execution/_helpers.py` (add a new function beside `list_input_datasets`, which ends ~line 226)
- Test: `tests/execution/test_input_datasets_with_versions.py` (new file)

**Interfaces:**
- Consumes (existing): `ml_instance.pathBuilder()`; `ml_instance.ml_schema`; `ml_instance.lookup_dataset(rid)`. The `Dataset_Execution` association table has columns `Execution`, `Dataset`, `Dataset_Version`.
- Produces (Task 3 relies on this exact signature): `list_input_datasets_with_versions(*, ml_instance, execution_rid) -> list[tuple[Dataset, str | None]]`.

**Background:** The existing `list_input_datasets` (same file) fetches `Dataset_Execution` rows for an execution and returns `[lookup_dataset(row["Dataset"]) for row ...]`, discarding the `Dataset_Version` column. The new helper does the same fetch but pairs each `Dataset` with its `Dataset_Version` value. `list_input_datasets` itself is NOT modified.

- [ ] **Step 1: Write the failing test**

Create `tests/execution/test_input_datasets_with_versions.py`:

```python
"""Unit test for list_input_datasets_with_versions.

Mocks the pathBuilder Dataset_Execution fetch and lookup_dataset so the
(Dataset, consumed_version) pairing is exercised offline.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from deriva_ml.execution._helpers import list_input_datasets_with_versions


def _make_ml(rows):
    """Build a fake ml_instance whose Dataset_Execution fetch returns `rows`."""
    entities = MagicMock()
    entities.fetch = lambda: rows
    de_path = MagicMock()
    de_path.filter.return_value = MagicMock(entities=lambda: entities)
    schema = MagicMock()
    schema.Dataset_Execution = de_path
    pb = MagicMock()
    pb.schemas = {"deriva-ml": schema}
    ml = MagicMock()
    ml.ml_schema = "deriva-ml"
    ml.pathBuilder.return_value = pb
    # lookup_dataset returns a stand-in carrying its rid so the test can assert.
    ml.lookup_dataset = lambda rid: SimpleNamespace(dataset_rid=rid)
    return ml


def test_pairs_dataset_with_consumed_version():
    ml = _make_ml([
        {"Dataset": "1-DSAA", "Dataset_Version": "1.0.0", "Execution": "2-EXAA"},
        {"Dataset": "1-DSAB", "Dataset_Version": "2.3.0", "Execution": "2-EXAA"},
    ])
    result = list_input_datasets_with_versions(ml_instance=ml, execution_rid="2-EXAA")
    pairs = {(ds.dataset_rid, v) for ds, v in result}
    assert pairs == {("1-DSAA", "1.0.0"), ("1-DSAB", "2.3.0")}


def test_version_none_when_edge_has_no_pin():
    ml = _make_ml([{"Dataset": "1-DSAA", "Execution": "2-EXAA"}])  # no Dataset_Version key
    result = list_input_datasets_with_versions(ml_instance=ml, execution_rid="2-EXAA")
    assert len(result) == 1
    ds, version = result[0]
    assert ds.dataset_rid == "1-DSAA"
    assert version is None


def test_skips_rows_without_dataset():
    ml = _make_ml([{"Dataset": None, "Dataset_Version": "1.0.0", "Execution": "2-EXAA"}])
    result = list_input_datasets_with_versions(ml_instance=ml, execution_rid="2-EXAA")
    assert result == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_input_datasets_with_versions.py -v`
Expected: FAIL — `ImportError: cannot import name 'list_input_datasets_with_versions'`.

- [ ] **Step 3: Implement the helper**

In `src/deriva_ml/execution/_helpers.py`, immediately after `list_input_datasets` (after ~line 226):

```python
def list_input_datasets_with_versions(
    *,
    ml_instance: Any,
    execution_rid: str,
) -> list[tuple[Any, str | None]]:
    """Input datasets of an execution paired with the consumed version.

    Like :func:`list_input_datasets`, but also returns the
    ``Dataset_Execution.Dataset_Version`` recorded on each input edge — the
    version of the dataset that was actually consumed. Lineage uses this to walk
    the consumed version rather than the dataset's current state. The existing
    :func:`list_input_datasets` ``list[Dataset]`` contract is intentionally left
    unchanged; lineage is the only caller that needs the consumed version.

    Args:
        ml_instance: The bound :class:`DerivaML` instance.
        execution_rid: The anchor execution RID.

    Returns:
        List of ``(Dataset, consumed_version)`` tuples. ``consumed_version`` is
        the version string from the input edge, or ``None`` when the edge has no
        version pin. Empty when the execution has no input datasets.

    Example:
        >>> pairs = list_input_datasets_with_versions(  # doctest: +SKIP
        ...     ml_instance=ml, execution_rid="2-EXAA"
        ... )
        >>> [(ds.dataset_rid, v) for ds, v in pairs]  # doctest: +SKIP
        [('1-DSAA', '1.0.0')]
    """
    pb = ml_instance.pathBuilder()
    dataset_exec = pb.schemas[ml_instance.ml_schema].Dataset_Execution
    records = dataset_exec.filter(dataset_exec.Execution == execution_rid).entities().fetch()
    return [
        (ml_instance.lookup_dataset(record["Dataset"]), record.get("Dataset_Version"))
        for record in records
        if record.get("Dataset")
    ]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_input_datasets_with_versions.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run ruff check src/deriva_ml/execution/_helpers.py tests/execution/test_input_datasets_with_versions.py
uv run ruff format --check src/deriva_ml/execution/_helpers.py tests/execution/test_input_datasets_with_versions.py
git add src/deriva_ml/execution/_helpers.py tests/execution/test_input_datasets_with_versions.py
git commit -m "feat(lineage): list_input_datasets_with_versions — surface consumed Dataset_Version"
```

---

### Task 2: version-aware `_producer_of_dataset`

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py` (`_producer_of_dataset`, currently ~lines 1325-1348)
- Test: `tests/execution/test_lookup_lineage_unit.py` (add a focused unit test using the `_FakeML` harness — but `_producer_of_dataset` is overridden there; see note → use a dedicated tiny test that drives the REAL method against a mock pathBuilder, in the new file from Task 1 or a new file)

**Interfaces:**
- Produces (Task 3 relies on): `_producer_of_dataset(self, dataset_rid, version=None) -> RID | None`.

**Background:** The current `_producer_of_dataset` fetches all `Dataset_Version` rows for the dataset and returns the `Execution` of the highest semver row. The version-aware path matches the row whose `Version` text equals `str(version)` and returns that row's `Execution`. The existing `_FakeML` harness OVERRIDES `_producer_of_dataset`, so the behavioral assertions live in Task 3; THIS task's test drives the real method against a mock pathBuilder.

- [ ] **Step 1: Write the failing test**

Add to `tests/execution/test_input_datasets_with_versions.py` (same file — it already mocks pathBuilder-style fetches; keep the producer test beside the helper test):

```python
from deriva_ml.core.mixins.execution import ExecutionMixin


def _ml_with_versions(version_rows):
    """Fake ExecutionMixin host whose Dataset_Version fetch returns version_rows."""
    entities = MagicMock()
    entities.fetch = lambda: version_rows
    vp = MagicMock()
    vp.filter.return_value = MagicMock(entities=lambda: entities)
    tables = {"Dataset_Version": vp}
    schema = MagicMock()
    schema.tables = tables
    pb = MagicMock()
    pb.schemas = {"deriva-ml": schema}
    ml = ExecutionMixin.__new__(ExecutionMixin)
    ml.pathBuilder = lambda: pb
    ml.ml_schema = "deriva-ml"
    return ml


def test_producer_of_dataset_latest_when_version_none():
    ml = _ml_with_versions([
        {"Version": "1.0.0", "Execution": "2-EXV1", "Dataset": "1-DSAA"},
        {"Version": "1.2.0", "Execution": "2-EXV2", "Dataset": "1-DSAA"},
    ])
    assert ml._producer_of_dataset("1-DSAA") == "2-EXV2"  # latest


def test_producer_of_dataset_specific_version():
    ml = _ml_with_versions([
        {"Version": "1.0.0", "Execution": "2-EXV1", "Dataset": "1-DSAA"},
        {"Version": "1.2.0", "Execution": "2-EXV2", "Dataset": "1-DSAA"},
    ])
    assert ml._producer_of_dataset("1-DSAA", version="1.0.0") == "2-EXV1"  # consumed


def test_producer_of_dataset_missing_version_returns_none():
    ml = _ml_with_versions([
        {"Version": "1.0.0", "Execution": "2-EXV1", "Dataset": "1-DSAA"},
    ])
    assert ml._producer_of_dataset("1-DSAA", version="9.9.9") is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_input_datasets_with_versions.py -k producer_of_dataset -v`
Expected: FAIL — `test_producer_of_dataset_specific_version` and `_missing_version` fail (the method ignores `version`); `_latest` passes already.

- [ ] **Step 3: Implement the version param**

Replace `_producer_of_dataset` in `src/deriva_ml/core/mixins/execution.py` with:

```python
    def _producer_of_dataset(self, dataset_rid: RID, version: Any | None = None) -> RID | None:
        """Return the Execution RID that produced a version of ``dataset_rid``.

        Args:
            dataset_rid: Dataset whose producing execution to find.
            version: When given, return the ``Execution`` recorded on that
                specific version's ``Dataset_Version`` row (the execution that
                produced the consumed version). When ``None`` (default), return
                the producer of the **latest** version — the historical
                behavior, unchanged for existing callers and the root path.

        Returns:
            The producing-execution RID, or ``None`` if the dataset has no
            ``Dataset_Version`` rows, the requested ``version`` has no row, or
            the matched row carries no ``Execution`` link.

        Example:
            >>> ml._producer_of_dataset("1-DSAA")  # doctest: +SKIP
            '2-EXV2'
            >>> ml._producer_of_dataset("1-DSAA", version="1.0.0")  # doctest: +SKIP
            '2-EXV1'
        """
        pb = self.pathBuilder()
        version_path = pb.schemas[self.ml_schema].tables["Dataset_Version"]
        rows = list(version_path.filter(version_path.Dataset == dataset_rid).entities().fetch())
        if not rows:
            return None

        if version is not None:
            want = str(version)
            for row in rows:
                if (row.get("Version") or "") == want:
                    return row.get("Execution")
            return None

        # Pick the row with the highest semver-style Version. The catalog
        # stores Version as text (e.g. "0.1.0"); sort lexically as a
        # tuple of ints so "1.10.0" beats "1.2.0".
        def _key(row: dict[str, Any]) -> tuple[int, ...]:
            v = row.get("Version") or "0.0.0"
            try:
                return tuple(int(p) for p in v.split("."))
            except ValueError:
                return (0,)

        latest = max(rows, key=_key)
        return latest.get("Execution")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_input_datasets_with_versions.py -k producer_of_dataset -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run ruff check src/deriva_ml/core/mixins/execution.py tests/execution/test_input_datasets_with_versions.py
uv run ruff format --check src/deriva_ml/core/mixins/execution.py tests/execution/test_input_datasets_with_versions.py
git add src/deriva_ml/core/mixins/execution.py tests/execution/test_input_datasets_with_versions.py
git commit -m "feat(lineage): _producer_of_dataset accepts version= (consumed-version producer)"
```

---

### Task 3: `_walk_node` consumed-loop rewrite + harness + behavioral tests

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py` (`_walk_node` consumed-dataset loop, ~lines 1564-1586)
- Modify: `tests/execution/test_lookup_lineage_unit.py` (extend the `_FakeML` harness + add behavioral tests)

**Interfaces:**
- Consumes: `list_input_datasets_with_versions` (Task 1); `_producer_of_dataset(dataset_rid, version=)` (Task 2); `_producers_of_dataset_members(dataset_rid, version=)` (already version-aware).

**Background:** `_walk_node` currently iterates `record.list_input_datasets()` (a method on the `ExecutionRecord` stub) and uses `ds.current_version` + version-less producer lookups. After this task it iterates `list_input_datasets_with_versions(ml_instance=self, execution_rid=execution_rid)` (the module helper), uses the consumed version for the summary and both producer lookups, and subtracts `execution_rid` from the member-producers it adds.

Because the walk now calls the MODULE-LEVEL helper (keyed by `execution_rid`) instead of `record.list_input_datasets()`, the `_FakeML` harness must intercept the new helper. The harness already stores `input_datasets` per execution on the `_StubExecutionRecord`; add a `_FakeML._input_dataset_versions` map (execution_rid -> list of (dataset_rid, consumed_version)) and override the module helper via monkeypatching OR via a method the helper resolves through. Simplest: monkeypatch `deriva_ml.core.mixins.execution.list_input_datasets_with_versions` is NOT possible if it's imported at call time inside `_walk_node` — so import it at call time inside `_walk_node` (the plan's implementation does `from ... import list_input_datasets_with_versions` inside the loop scope) and have `_FakeML` provide the inputs through a scripted map that a test-only override reads. See Step 1 for the exact harness shape.

- [ ] **Step 1: Extend the `_FakeML` harness**

In `tests/execution/test_lookup_lineage_unit.py`:

(a) The current `_StubExecutionRecord.list_input_datasets()` returns `_StubDataset` objects. We keep scripting inputs the same way (via `add_execution(input_datasets=[...])`), but ALSO let each `_StubDataset` optionally carry a consumed version. Extend `_StubDataset`:

```python
class _StubDataset:
    """Minimal Dataset stand-in for list_input_datasets()."""

    def __init__(
        self,
        dataset_rid: str,
        description: str = "",
        version: str = "0.1.0",
        consumed_version: str | None = None,
    ) -> None:
        self.dataset_rid = dataset_rid
        self.description = description
        self._version = version
        # The version recorded on the Dataset_Execution edge (what was consumed).
        # None means "no pin" -> the walk falls back to current_version.
        self.consumed_version = consumed_version

    @property
    def current_version(self) -> str:
        return self._version
```

(b) Add an override on `_FakeML` so the module helper resolves to the scripted inputs. `_walk_node` will call `list_input_datasets_with_versions(ml_instance=self, execution_rid=...)`; make `_FakeML` provide a method the implementation uses. To keep the implementation clean, the implementation calls the module function; the test overrides it by defining a method `list_input_datasets_with_versions` on `_FakeML` is NOT how a module function resolves. Instead, monkeypatch at test-collection time. Add this fixture-free helper at the top of `_FakeML`:

```python
    def _scripted_input_pairs(self, execution_rid: str):
        """(Dataset, consumed_version) pairs for an execution, from the stub record."""
        rec = self._executions.get(execution_rid)
        if rec is None:
            return []
        return [(ds, ds.consumed_version) for ds in rec.list_input_datasets()]
```

(c) Because `_walk_node` calls the module-level `list_input_datasets_with_versions`, the implementation in Task 3 Step 3 must call it through an overridable seam. Define a thin private method on `ExecutionMixin` that `_walk_node` calls, so `_FakeML` can override it:

In the IMPLEMENTATION (execution.py), `_walk_node` calls `self._input_dataset_pairs(execution_rid)`, and `ExecutionMixin` defines:

```python
    def _input_dataset_pairs(self, execution_rid: RID) -> list[tuple[Any, str | None]]:
        """(Dataset, consumed_version) pairs for an execution's input edges.

        Thin wrapper over
        :func:`deriva_ml.execution._helpers.list_input_datasets_with_versions`
        so the lineage walk has a single overridable seam for tests.
        """
        from deriva_ml.execution._helpers import list_input_datasets_with_versions

        return list_input_datasets_with_versions(ml_instance=self, execution_rid=execution_rid)
```

Then `_FakeML` overrides `_input_dataset_pairs` to return `self._scripted_input_pairs(execution_rid)`:

```python
    def _input_dataset_pairs(self, execution_rid: str):  # type: ignore[override]
        return self._scripted_input_pairs(execution_rid)
```

This seam keeps the real wiring (helper call) in `_input_dataset_pairs` and lets the offline harness script inputs without a catalog.

- [ ] **Step 2: Write the failing behavioral tests**

Add to `tests/execution/test_lookup_lineage_unit.py`. (`_FakeML` already has `set_member_producers`, `_producer_of_dataset` override, and now `consumed_version` on `_StubDataset`. The `_producer_of_dataset` override in `_FakeML` keys off `dataset_rid` only — extend it to honor a per-(dataset,version) map; add a `set_versioned_producer` helper.)

First extend `_FakeML` producer scripting:

```python
    # in __init__:  self._versioned_dataset_producers: dict[tuple[str, str], str] = {}
    def set_versioned_producer(self, dataset_rid: str, version: str, producer: str) -> None:
        """Script the producer of a SPECIFIC dataset version."""
        self._versioned_dataset_producers[(dataset_rid, version)] = producer

    def _producer_of_dataset(self, dataset_rid: str, version: Any = None) -> str | None:  # type: ignore[override]
        if version is not None:
            return self._versioned_dataset_producers.get((dataset_rid, str(version)))
        return self._dataset_producers.get(dataset_rid)
```

And extend the member-producer override to honor version (so version-aware member tests work):

```python
    # in __init__:  self._versioned_member_producers: dict[tuple[str, str], set[str]] = {}
    def set_versioned_member_producers(self, dataset_rid: str, version: str, producers: set[str]) -> None:
        self._versioned_member_producers[(dataset_rid, str(version))] = set(producers)

    def _producers_of_dataset_members(self, dataset_rid: str, version: Any = None) -> set[str]:  # type: ignore[override]
        if version is not None and (dataset_rid, str(version)) in self._versioned_member_producers:
            return set(self._versioned_member_producers[(dataset_rid, str(version))])
        return set(self._dataset_member_producers.get(dataset_rid, set()))
```

Now the tests:

```python
def test_mid_walk_self_parent_guard_no_false_cycle():
    """An execution that consumed D and also produced one of D's members must
    not become its own parent (no false cycle)."""
    ml = _FakeML()
    ml.add_dataset("1-DSAA", producer=None)
    # EXSC consumes D-CON and also produced a member of D-CON.
    ml.add_dataset("1-DCON", producer="2-EXOT")
    ml.add_execution("2-EXSC", input_datasets=[_StubDataset("1-DCON")])
    ml.add_dataset("1-DOUT", producer="2-EXSC")
    ml.set_member_producers("1-DCON", {"2-EXSC"})  # self-produced member
    ml.add_execution("2-EXOT", input_datasets=[])

    result = ml.lookup_lineage("1-DOUT")

    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXSC"
    # 2-EXSC must NOT appear among its own parents, and no false cycle.
    assert all(p.execution.rid != "2-EXSC" for p in result.lineage.parents)
    assert result.cycle_detected is False


def test_mid_walk_uses_consumed_version_producer():
    """When E consumed D@1.0.0 (produced by X) but D@2.0.0 (latest) is by Y,
    walking through E surfaces X, not Y."""
    ml = _FakeML()
    ml.add_dataset("1-DSRC", producer=None)
    ml.add_execution("2-EXVX", input_datasets=[_StubDataset("1-DSRC")])  # produced D@1.0.0
    ml.add_execution("2-EXVY", input_datasets=[])                         # produced D@2.0.0
    # D consumed at 1.0.0 by EXMID:
    ml.add_dataset("1-DVER", producer="2-EXVY")  # latest producer = Y
    ml.set_versioned_producer("1-DVER", "1.0.0", "2-EXVX")  # consumed-version producer = X
    ml.add_execution("2-EXMID", input_datasets=[_StubDataset("1-DVER", consumed_version="1.0.0")])
    ml.add_dataset("1-DEND", producer="2-EXMID")

    result = ml.lookup_lineage("1-DEND")

    parent_rids = {p.execution.rid for p in result.lineage.parents}
    assert "2-EXVX" in parent_rids   # consumed-version producer
    assert "2-EXVY" not in parent_rids  # latest-version producer must NOT appear


def test_mid_walk_uses_consumed_version_member_producers():
    """D@1.0.0 members by P1; D@2.0.0 adds members by P2. Walking E (consumed
    1.0.0) surfaces P1, not P2."""
    ml = _FakeML()
    ml.add_dataset("1-DSRC", producer=None)
    ml.add_execution("2-EXP1", input_datasets=[_StubDataset("1-DSRC")])
    ml.add_execution("2-EXP2", input_datasets=[])
    ml.add_dataset("1-DVER", producer=None)
    ml.set_versioned_member_producers("1-DVER", "1.0.0", {"2-EXP1"})
    ml.set_versioned_member_producers("1-DVER", "2.0.0", {"2-EXP1", "2-EXP2"})
    ml.add_execution("2-EXMID", input_datasets=[_StubDataset("1-DVER", consumed_version="1.0.0")])
    ml.add_dataset("1-DEND", producer="2-EXMID")

    result = ml.lookup_lineage("1-DEND")

    parent_rids = {p.execution.rid for p in result.lineage.parents}
    assert "2-EXP1" in parent_rids
    assert "2-EXP2" not in parent_rids


def test_mid_walk_consumed_dataset_summary_reports_consumed_version():
    """consumed_datasets[].version reflects the consumed version, not current."""
    ml = _FakeML()
    ml.add_dataset("1-DSRC", producer=None)
    ml.add_dataset("1-DVER", producer=None)
    ml.add_execution(
        "2-EXMID",
        input_datasets=[_StubDataset("1-DVER", version="9.9.9", consumed_version="1.0.0")],
    )
    ml.add_dataset("1-DEND", producer="2-EXMID")

    result = ml.lookup_lineage("1-DEND")

    consumed = result.lineage.consumed_datasets
    assert len(consumed) == 1
    assert consumed[0].version == "1.0.0"  # consumed, not current 9.9.9
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_lookup_lineage_unit.py -k "mid_walk_self_parent or consumed_version" -v`
Expected: FAIL — these exercise behavior not yet implemented (the loop still uses `record.list_input_datasets()` + version-less lookups + no self-parent guard). The harness seam `_input_dataset_pairs` may also need the implementation method to exist; if the import of `list_input_datasets_with_versions` is missing it errors — that's an expected red.

- [ ] **Step 4: Implement the `_walk_node` rewrite**

In `src/deriva_ml/core/mixins/execution.py`:

(a) Add the `_input_dataset_pairs` seam method (place it beside `_producer_of_dataset`, before `_walk_node`):

```python
    def _input_dataset_pairs(self, execution_rid: RID) -> list[tuple[Any, str | None]]:
        """(Dataset, consumed_version) pairs for an execution's input edges.

        Thin wrapper over
        :func:`deriva_ml.execution._helpers.list_input_datasets_with_versions`
        so the lineage walk has one overridable seam (tests stub this).

        Args:
            execution_rid: Execution whose input edges to read.

        Returns:
            List of ``(Dataset, consumed_version)`` tuples; ``consumed_version``
            is ``None`` for edges with no version pin.
        """
        from deriva_ml.execution._helpers import list_input_datasets_with_versions

        return list_input_datasets_with_versions(ml_instance=self, execution_rid=execution_rid)
```

(b) Replace the consumed-dataset loop in `_walk_node` (currently):

```python
            # Consumed inputs.
            consumed_datasets: list[DatasetSummary] = []
            parent_rids: set[RID] = set()
            for ds in record.list_input_datasets():
                ds_version = None
                try:
                    ds_version = str(ds.current_version)
                except Exception:
                    pass
                consumed_datasets.append(
                    DatasetSummary(
                        rid=ds.dataset_rid,
                        description=ds.description or None,
                        version=ds_version,
                    )
                )
                producer = self._producer_of_dataset(ds.dataset_rid)
                if producer:
                    parent_rids.add(producer)
                # Members of this consumed dataset may have been produced by a
                # different execution than the one that assembled the dataset;
                # those member-producers are data-flow parents too.
                parent_rids |= self._producers_of_dataset_members(ds.dataset_rid)
```

with:

```python
            # Consumed inputs. Walk the version that was ACTUALLY consumed
            # (Dataset_Execution.Dataset_Version), not the dataset's current
            # state, so lineage reflects the inputs as they were at consumption.
            consumed_datasets: list[DatasetSummary] = []
            parent_rids: set[RID] = set()
            for ds, consumed_version in self._input_dataset_pairs(execution_rid):
                version_str = consumed_version
                if version_str is None:
                    try:
                        version_str = str(ds.current_version)
                    except Exception:
                        version_str = None
                consumed_datasets.append(
                    DatasetSummary(
                        rid=ds.dataset_rid,
                        description=ds.description or None,
                        version=version_str,
                    )
                )
                producer = self._producer_of_dataset(ds.dataset_rid, version=consumed_version)
                if producer:
                    parent_rids.add(producer)
                # Member-producers of the CONSUMED version. Never the execution
                # we are currently expanding: an execution that both consumed
                # this dataset and produced some of its members must not become
                # its own parent (the mid-walk analogue of the root path's
                # version-producer subtraction).
                member_producers = self._producers_of_dataset_members(
                    ds.dataset_rid, version=consumed_version
                )
                parent_rids |= member_producers - {execution_rid}
```

Leave the consumed-ASSET loop, the `extra_parent_rids` merge, and the recursion UNCHANGED.

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_lookup_lineage_unit.py -k "mid_walk_self_parent or consumed_version" -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Run the FULL unit file for no-regression**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_lookup_lineage_unit.py -v`
Expected: ALL pass. The existing tests script inputs via `_StubDataset` with `consumed_version=None`, so the loop falls back to `current_version` and version-less producer lookups — identical to today's behavior. The member-producer override returns the unversioned set when no versioned entry is scripted, so existing member-producer tests are unchanged. The self-parent subtraction `- {execution_rid}` is a no-op for them (the expanding execution isn't in their member-producer sets).

- [ ] **Step 7: Run the producers + helper files too (full lineage surface)**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_lookup_lineage_unit.py tests/execution/test_producers_of_dataset_members.py tests/execution/test_input_datasets_with_versions.py -q`
Expected: all green.

- [ ] **Step 8: Lint + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run ruff check src/deriva_ml/core/mixins/execution.py tests/execution/test_lookup_lineage_unit.py
uv run ruff format --check src/deriva_ml/core/mixins/execution.py tests/execution/test_lookup_lineage_unit.py
git add src/deriva_ml/core/mixins/execution.py tests/execution/test_lookup_lineage_unit.py
git commit -m "fix(lineage): mid-walk uses consumed version + self-parent guard"
```

---

### Task 4: live versioned-mutation regression test

**Files:**
- Modify: `tests/execution/test_lookup_lineage_live.py` (add one DERIVA_HOST-gated test)

**Interfaces:**
- Consumes: the full `lookup_lineage` behavior (Tasks 1-3), the `test_ml` fixture, `tests/factories.py`, the `DERIVA_HOST` gate.

**Background:** This is the one scenario no offline mock fully proves: a real catalog where a consumed dataset is mutated to a new version after consumption, with the new version's members produced by a different execution. The test must assert `lookup_lineage` reflects the CONSUMED version's producers, not the latest. Read the existing live tests + `tests/factories.py` first; build the shape with the real catalog APIs.

- [ ] **Step 1: Write the live test**

Add to `tests/execution/test_lookup_lineage_live.py` (after the existing tests). Replace the `...` with real construction using the catalog APIs — study the existing live tests + factories first; the ASSERTION block is the contract:

```python
@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="lookup_lineage live smoke test requires DERIVA_HOST",
)
def test_lookup_lineage_reflects_consumed_version_not_latest(test_ml):
    """A dataset consumed at v1 must contribute v1's producers to lineage even
    after it is mutated to v2 by a different execution (tk-020 Gap 1).

    Shape:
        exec_v1 produces D@v1
        exec_mid consumes D@v1, produces DS_OUT
        exec_v2 later mutates D to v2 (new members / new version producer)
    Assert: lookup_lineage(DS_OUT) surfaces exec_v1 (the consumed-version
    producer of D), NOT exec_v2 (the later/latest producer).
    """
    # Build exec_v1 -> D@v1; exec_mid consumes D@v1 -> DS_OUT; then exec_v2
    # mutates D to v2. Use the catalog APIs / factories (see existing live
    # tests + tests/factories.py).
    #
    #   d_rid        = RID of D
    #   ds_out_rid   = RID of DS_OUT
    #   exec_v1_rid  = producer of D@v1 (consumed)
    #   exec_v2_rid  = producer/mutator of D@v2 (latest)
    ...

    result = test_ml.lookup_lineage(ds_out_rid)

    # Collect every execution RID anywhere in the lineage tree.
    seen: set[str] = set()

    def _collect(node):
        if node is None:
            return
        seen.add(node.execution.rid)
        for p in node.parents:
            _collect(p)

    _collect(result.lineage)

    assert exec_v1_rid in seen, (
        f"consumed-version producer {exec_v1_rid} missing; saw {seen}"
    )
    assert exec_v2_rid not in seen, (
        f"latest-version producer {exec_v2_rid} leaked into lineage; saw {seen}"
    )
    assert result.cycle_detected is False
```

The implementer MUST replace the `...` with real construction (no leftover ellipsis). If a faithful versioned-mutation shape cannot be built through the available public APIs, report exactly what blocked it and what partial shape was achievable — do NOT weaken the assertions or fake a pass.

- [ ] **Step 2: Run the live test (gated)**

Run (localhost container is up; the `test_ml` fixture defaults `DERIVA_HOST=localhost` and manages its own throwaway catalog; set `DERIVA_ML_ALLOW_DIRTY=true` if workflow creation complains about a dirty tree):
`cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_HOST=localhost DERIVA_ML_ALLOW_DIRTY=true uv run python -m pytest tests/execution/test_lookup_lineage_live.py::test_lookup_lineage_reflects_consumed_version_not_latest -v`
Expected: PASS. If a live host is unavailable, report that, confirm it SKIPS without `DERIVA_HOST`, and confirm collection is clean — do NOT fake a pass.

- [ ] **Step 3: Confirm skip path + collection**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_lookup_lineage_live.py -v`
Expected: all live tests SKIP (no `DERIVA_HOST`), no collection errors.

- [ ] **Step 4: Lint + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run ruff check tests/execution/test_lookup_lineage_live.py
uv run ruff format --check tests/execution/test_lookup_lineage_live.py
git add tests/execution/test_lookup_lineage_live.py
git commit -m "test(lineage): live consumed-version regression (tk-020 Gap 1)"
```

---

## Final verification (after all tasks)

- [ ] Full execution test directory: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/ -v` — all pass (live tests skip without `DERIVA_HOST`; the 2 pre-existing `test_workflow_creation_*` subprocess failures are unrelated).
- [ ] Lint the touched surface: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src tests`.
- [ ] Confirm `list_input_datasets` (the original helper) and its four callers are untouched: `cd /Users/carl/GitHub/DerivaML/deriva-ml && git diff main -- src/deriva_ml/execution/_helpers.py` shows ONLY the new function added, `list_input_datasets` unchanged.
