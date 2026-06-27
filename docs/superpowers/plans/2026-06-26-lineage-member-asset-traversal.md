# Lineage Member-Asset Traversal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `lookup_lineage(<dataset>)` follow the dataset's member assets back to their producing execution(s), so a dataset whose members were produced by a different execution than the one that assembled it (e.g. CIFAR images) surfaces the members' upstream lineage (the source File dataset).

**Architecture:** Add one private helper `_producers_of_dataset_members(dataset_rid)` that returns the distinct executions which produced a dataset's member assets. Feed those producing executions into the existing lineage walk as ordinary data-flow parents at two points: (a) the **root** dataset in `lookup_lineage` (via a new internal `extra_parent_rids` parameter on `_walk_node`), and (b) **mid-walk** when a dataset is reached as a consumed input. No change to the public Pydantic models; member-producers appear as ordinary `parents`. Stays inside ADR-0001's data-flow doctrine (an asset's producer is its `<AssetTable>_Execution` Output edge).

**Tech Stack:** Python 3.13, `uv`, pytest, deriva-ml core (`deriva_ml.core.mixins.execution.ExecutionMixin`), ERMrest PathBuilder.

## Global Constraints

- All work is in the `../deriva-ml` repo on branch `feature/lineage-member-asset-traversal` (already created; the design spec is committed there).
- Use `uv` for everything: `uv run python -m pytest`, `uv run ruff check`, `uv run ruff format`. Never invoke `pytest`/`ruff`/`python` directly.
- Google-style docstrings (Args/Returns/Raises/Example) on every new function/method.
- No public-model change: `LineageNode`, `LineageResult`, `RootDescriptor`, and the summary models in `src/deriva_ml/execution/lineage.py` are NOT modified.
- No `Execution_Execution`/orchestration traversal; no change to `find_executions_consuming`.
- The helper must be **O(member-asset-tables) queries, not O(members)** — never a single unbounded `IN (<thousands of RIDs>)` filter (HTTP 414 risk, cf. resolve_rids scale bug).
- Member-producer dedup: a dataset of N members sharing one producing execution yields exactly ONE producer RID.
- Offline unit tests mock the catalog (extend the existing `_FakeML(ExecutionMixin)` harness in `tests/execution/test_lookup_lineage_unit.py`). The live end-to-end test is gated on the `DERIVA_HOST` env var (the convention used by `tests/execution/test_lookup_lineage_live.py`), NOT `DERIVA_ML_LIVE_LOCALHOST`.
- All edits to the walk live in `src/deriva_ml/core/mixins/execution.py`.

---

### Task 1: `_producers_of_dataset_members` helper

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py` (add a new private method beside `_producer_of_dataset` / `_producer_of_asset`, which currently end around line 1369)
- Test: `tests/execution/test_producers_of_dataset_members.py` (new file)

**Interfaces:**
- Consumes (existing, already in `ExecutionMixin`/model): `self.lookup_dataset(rid)` → a `Dataset` with `.list_dataset_members(version=...) -> dict[str, list[dict]]`; `self.model.is_asset(table) -> bool`; `self.model.name_to_table(name) -> Table`; `self.model.find_association(asset_table, "Execution") -> tuple[Table, str, str]` (returns `(assoc_table, asset_fk_col_name, exec_fk_col_name)`); `self.pathBuilder()`; `NoAssociationException` (imported in this module already — confirm the existing import near `_producer_of_asset`).
- Produces (later tasks rely on this exact signature): `_producers_of_dataset_members(self, dataset_rid: RID, version: Any | None = None) -> set[RID]`.

**Background the implementer needs:**
- `list_dataset_members()` returns a dict keyed by member-type name → list of member row dicts (each row dict has at least `"RID"`). Member types include domain asset tables (e.g. `"Image"`), the `"File"` table, and nested `"Dataset"` members.
- `_producer_of_asset(asset_rid, asset_table)` (just above the insertion point) already shows the canonical way to get an asset's Output producer: `find_association(asset_table, "Execution")` → `(assoc_table, asset_fk, _exec_fk)`, then filter the assoc path on `columns[asset_fk] == asset_rid` and `Asset_Role == "Output"`, read `.get("Execution")`. The new helper does the same but for ALL members of a table at once and returns the DISTINCT set.
- Performance: do NOT loop `_producer_of_asset` per member. Instead, for each asset table, issue ONE association query filtered to that table's member RIDs and project distinct `Execution`. The member RID count per dataset is bounded; to avoid an over-long URL, filter the association in chunks of at most 500 member RIDs (matching the existing RID-chunk convention used elsewhere in the codebase) and union the results. (A pure server-side membership join is a possible optimization but the chunked-RID approach is simpler and provably correct; use it.)

- [ ] **Step 1: Write the failing test file**

Create `tests/execution/test_producers_of_dataset_members.py`:

```python
"""Unit tests for ``ExecutionMixin._producers_of_dataset_members``.

Mocks ``lookup_dataset``/``list_dataset_members`` and the per-table
association fetch so the helper's enumerate-members → distinct-producers
logic is exercised offline. The catalog-bound query path is covered by
the live test in ``tests/execution/test_lookup_lineage_live.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from deriva_ml.core.mixins.execution import ExecutionMixin


class _FakeMembersML(ExecutionMixin):
    """ExecutionMixin host scripting just what _producers_of_dataset_members needs.

    Scripts:
      * lookup_dataset(rid).list_dataset_members() -> members dict
      * model.is_asset(table) -> whether a member-type table is an asset table
      * the per-(asset_table, member_rids) Output-producer lookup
    """

    def __init__(self) -> None:
        # dataset_rid -> {member_type: [ {"RID": ...}, ... ]}
        self._members: dict[str, dict[str, list[dict[str, Any]]]] = {}
        # member_type name -> bool (is it an asset table?)
        self._asset_types: set[str] = set()
        # (asset_table, member_rid) -> producer execution RID or None
        self._member_producers: dict[tuple[str, str], str | None] = {}
        # asset tables that have NO <Asset>_Execution association at all
        self._no_assoc_tables: set[str] = set()

        self.model = MagicMock()
        self.model.is_asset = lambda table: getattr(table, "name", table) in self._asset_types
        self.model.name_to_table = lambda name: _T(name)

    # scripting helpers -----------------------------------------------------
    def set_members(self, dataset_rid: str, members: dict[str, list[dict[str, Any]]]) -> None:
        self._members[dataset_rid] = members

    def mark_asset_types(self, *names: str) -> None:
        self._asset_types.update(names)

    def set_member_producer(self, asset_table: str, member_rid: str, producer: str | None) -> None:
        self._member_producers[(asset_table, member_rid)] = producer

    def mark_no_association(self, asset_table: str) -> None:
        self._no_assoc_tables.add(asset_table)

    # mocked primitives -----------------------------------------------------
    def lookup_dataset(self, rid: str) -> Any:  # type: ignore[override]
        ds = MagicMock()
        ds.list_dataset_members = lambda version=None: self._members.get(rid, {})
        return ds

    def _distinct_member_output_producers(self, asset_table: str, member_rids: list[str]) -> set[str]:
        """Test seam the real helper calls once per asset table.

        The real implementation issues the chunked association query here;
        the test scripts its result directly.
        """
        if asset_table in self._no_assoc_tables:
            return set()
        out: set[str] = set()
        for rid in member_rids:
            p = self._member_producers.get((asset_table, rid))
            if p:
                out.add(p)
        return out


class _T:
    def __init__(self, name: str) -> None:
        self.name = name


def test_distinct_producers_dedup_across_many_members():
    """2000 members sharing one producer -> a single producer RID."""
    ml = _FakeMembersML()
    ml.mark_asset_types("Image")
    members = {"Image": [{"RID": f"4-IMG{i}"} for i in range(2000)]}
    ml.set_members("1-DSAA", members)
    for i in range(2000):
        ml.set_member_producer("Image", f"4-IMG{i}", "2-EXUP")

    result = ml._producers_of_dataset_members("1-DSAA")

    assert result == {"2-EXUP"}


def test_no_members_returns_empty_set():
    ml = _FakeMembersML()
    ml.set_members("1-DSAA", {})
    assert ml._producers_of_dataset_members("1-DSAA") == set()


def test_members_with_no_output_producer_returns_empty_set():
    ml = _FakeMembersML()
    ml.mark_asset_types("Image")
    ml.set_members("1-DSAA", {"Image": [{"RID": "4-IMG0"}]})
    ml.set_member_producer("Image", "4-IMG0", None)
    assert ml._producers_of_dataset_members("1-DSAA") == set()


def test_union_across_multiple_asset_tables():
    ml = _FakeMembersML()
    ml.mark_asset_types("Image", "File")
    ml.set_members(
        "1-DSAA",
        {"Image": [{"RID": "4-IMG0"}], "File": [{"RID": "4-FIL0"}]},
    )
    ml.set_member_producer("Image", "4-IMG0", "2-EXAA")
    ml.set_member_producer("File", "4-FIL0", "2-EXAB")
    assert ml._producers_of_dataset_members("1-DSAA") == {"2-EXAA", "2-EXAB"}


def test_nested_dataset_and_non_asset_members_skipped():
    ml = _FakeMembersML()
    ml.mark_asset_types("Image")  # "Dataset" is intentionally NOT an asset type
    ml.set_members(
        "1-DSAA",
        {
            "Image": [{"RID": "4-IMG0"}],
            "Dataset": [{"RID": "1-DSCH"}],  # nested dataset member -> skipped
        },
    )
    ml.set_member_producer("Image", "4-IMG0", "2-EXAA")
    # Even if a producer were scriptable for the nested dataset, it must be ignored.
    assert ml._producers_of_dataset_members("1-DSAA") == {"2-EXAA"}


def test_asset_table_without_execution_association_skipped():
    ml = _FakeMembersML()
    ml.mark_asset_types("Image")
    ml.mark_no_association("Image")
    ml.set_members("1-DSAA", {"Image": [{"RID": "4-IMG0"}]})
    ml.set_member_producer("Image", "4-IMG0", "2-EXAA")  # ignored: no assoc table
    assert ml._producers_of_dataset_members("1-DSAA") == set()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_producers_of_dataset_members.py -v`
Expected: FAIL — `AttributeError: 'ExecutionMixin' object has no attribute '_producers_of_dataset_members'` (and the seam `_distinct_member_output_producers` not yet referenced by the real code).

- [ ] **Step 3: Implement `_producers_of_dataset_members` and its query seam**

In `src/deriva_ml/core/mixins/execution.py`, add these two methods immediately after `_producer_of_asset` (after line ~1368). The public-facing helper enumerates member asset tables; the private seam `_distinct_member_output_producers` does the chunked association query (and is the method the unit test overrides).

```python
    def _producers_of_dataset_members(
        self, dataset_rid: RID, version: Any | None = None
    ) -> set[RID]:
        """Distinct executions that produced the member assets of a dataset.

        Enumerates the dataset's member asset tables and, for each, collects
        the distinct producing executions (the asset's ``<Asset>_Execution``
        association with ``Asset_Role="Output"``). Deduplicated across all
        members and tables, so a dataset of 2000 images that share one
        producing execution yields a single RID. Nested-``Dataset`` members and
        non-asset member kinds are skipped — dataset producers are handled by
        :meth:`_producer_of_dataset`, reached through the normal dataset-input
        path.

        The work is bounded by the number of member *asset tables* (typically
        1-2), not the number of members: one chunked association query per
        table (see :meth:`_distinct_member_output_producers`).

        Args:
            dataset_rid: RID of the dataset whose member assets to inspect.
            version: Optional dataset version to list members from. ``None``
                uses the current version.

        Returns:
            Set of distinct producing-execution RIDs. Empty when the dataset
            has no member assets or none have a recorded ``Output`` producer.

        Example:
            >>> producers = ml._producers_of_dataset_members("1-DSAA")  # doctest: +SKIP
            >>> sorted(producers)  # doctest: +SKIP
            ['2-EXUP']
        """
        members = self.lookup_dataset(dataset_rid).list_dataset_members(version=version)
        producers: set[RID] = set()
        for member_type, rows in members.items():
            if not rows:
                continue
            table = self.model.name_to_table(member_type)
            if not self.model.is_asset(table):
                # Nested-Dataset members and non-asset member kinds are not
                # asset-producer-shaped; skip them.
                continue
            member_rids = [r["RID"] for r in rows if r.get("RID")]
            if not member_rids:
                continue
            producers |= self._distinct_member_output_producers(member_type, member_rids)
        return producers

    def _distinct_member_output_producers(
        self, asset_table_name: str, member_rids: list[RID]
    ) -> set[RID]:
        """Distinct ``Output`` producing executions for a set of asset RIDs.

        Issues one chunked association query against the asset table's
        ``<Asset>_Execution`` table (``Asset_Role="Output"``), filtering by the
        given member RIDs in chunks of at most ``_MEMBER_PRODUCER_CHUNK`` to
        stay under URL length limits, and returns the distinct ``Execution``
        RIDs. Returns an empty set if the asset table has no execution
        association.

        Args:
            asset_table_name: Name of the member asset table (e.g. ``"Image"``).
            member_rids: RIDs of that table's members in this dataset.

        Returns:
            Set of distinct producing-execution RIDs (``Output`` role).
        """
        asset_table = self.model.name_to_table(asset_table_name)
        try:
            assoc_table, asset_fk, _exec_fk = self.model.find_association(asset_table, "Execution")
        except NoAssociationException:
            return set()

        pb = self.pathBuilder()
        assoc_path = pb.schemas[assoc_table.schema.name].tables[assoc_table.name]
        producers: set[RID] = set()
        for start in range(0, len(member_rids), _MEMBER_PRODUCER_CHUNK):
            chunk = member_rids[start : start + _MEMBER_PRODUCER_CHUNK]
            rows = (
                assoc_path.filter(assoc_path.columns[asset_fk].any(*chunk))
                .filter(assoc_path.Asset_Role == "Output")
                .entities()
                .fetch()
            )
            for row in rows:
                exec_rid = row.get("Execution")
                if exec_rid:
                    producers.add(exec_rid)
        return producers
```

Add the chunk-size module constant near the top of the file, beside the other module-level constants (search for an existing ``_MAX_RIDS_PER_QUERY`` or similar; if none, place it after the imports):

```python
# Max member RIDs per <Asset>_Execution association filter, to stay under
# ERMrest URL length limits (cf. the resolve_rids chunking).
_MEMBER_PRODUCER_CHUNK = 500
```

Note on `.any(*chunk)`: ERMrest PathBuilder column predicates expose `.any(...)` for "value in set" filtering. If the installed PathBuilder API spells this differently (verify against an existing multi-value filter in the codebase — search for `.any(` under `src/deriva_ml/`), use the form already used there. If no `.any` exists, build the predicate as an OR-chain over the chunk. The unit test does NOT exercise this method (it overrides the seam), so confirm the predicate shape against the live catalog in Task 4.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_producers_of_dataset_members.py -v`
Expected: PASS (6 passed). The unit tests override `_distinct_member_output_producers`, so they exercise the enumerate/dedup/skip logic of `_producers_of_dataset_members` without touching a catalog.

- [ ] **Step 4b: Add the query-count test (the spec's O(asset-tables) guarantee)**

The six tests above override the query seam, so they prove the enumerate/dedup
logic but NOT the "one query per asset table, not per member" property the spec
requires. Add ONE more test that drives the REAL `_distinct_member_output_producers`
against a mock pathBuilder and counts `.fetch()` calls — for 2000 members in one
table it must issue ⌈2000/500⌉ = 4 chunked fetches (NOT 2000), and the chunk math
is the only thing that scales with member count. Append to the same test file:

```python
def test_distinct_member_output_producers_chunks_not_per_member():
    """The real seam issues O(chunks) association fetches, never O(members)."""
    from unittest.mock import MagicMock

    ml = _FakeMembersML()
    # Real model.find_association + name_to_table: return stub table/assoc.
    assoc_tbl = MagicMock()
    assoc_tbl.schema.name = "deriva-ml"
    assoc_tbl.name = "Image_Execution"
    ml.model.find_association = lambda asset_table, target: (assoc_tbl, "Image", "Execution")
    ml.model.name_to_table = lambda name: _T(name)

    # Mock pathBuilder so .filter(...).filter(...).entities().fetch() returns []
    # and counts fetch() calls.
    fetch_calls = {"n": 0}

    def _fetch():
        fetch_calls["n"] += 1
        return []

    entities = MagicMock()
    entities.fetch = _fetch
    path = MagicMock()
    path.entities.return_value = entities
    path.filter.return_value = path  # chainable .filter(...).filter(...)
    # column access: assoc_path.columns[asset_fk] and assoc_path.Asset_Role
    path.columns = {"Image": MagicMock()}
    path.columns["Image"].any = lambda *rids: MagicMock()
    path.Asset_Role = MagicMock()

    schema = MagicMock()
    schema.tables = {"Image_Execution": path}
    pb = MagicMock()
    pb.schemas = {"deriva-ml": schema}
    ml.pathBuilder = lambda: pb

    member_rids = [f"4-IMG{i}" for i in range(2000)]
    ml._distinct_member_output_producers("Image", member_rids)

    # 2000 members / 500 chunk = 4 fetches. The point: it does NOT scale with
    # the number of members (would be 2000 if implemented per-asset).
    assert fetch_calls["n"] == 4, f"expected 4 chunked fetches, got {fetch_calls['n']}"
```

Note: this test depends on the exact PathBuilder predicate spelling chosen in
Step 3 (`.columns[asset_fk].any(...)`, `.Asset_Role == "Output"`). If you changed
the predicate form in Step 3 (e.g. no `.any`), adjust the mock to match the calls
your implementation actually makes — the assertion that matters is the fetch
COUNT (4, the chunk count), not the mock's internal shape. Run it:

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_producers_of_dataset_members.py::test_distinct_member_output_producers_chunks_not_per_member -v`
Expected: PASS (1 passed). If the mock can't faithfully model your predicate
calls, it's acceptable to instead assert via a lighter spy that `.fetch` was
called 4 times; do NOT weaken it to "called at least once" — the chunk count is
the guarantee.

- [ ] **Step 5: Lint**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src/deriva_ml/core/mixins/execution.py tests/execution/test_producers_of_dataset_members.py && uv run ruff format --check src/deriva_ml/core/mixins/execution.py tests/execution/test_producers_of_dataset_members.py`
Expected: clean (run `uv run ruff format <files>` and re-check if it would reformat).

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/core/mixins/execution.py tests/execution/test_producers_of_dataset_members.py
git commit -m "feat(lineage): _producers_of_dataset_members — distinct member-asset producers"
```

---

### Task 2: `_walk_node` gains `extra_parent_rids` + mid-walk consumed-dataset seeding

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py` (`_walk_node`, currently starts ~line 1370; the consumed-dataset loop is ~line 1455)
- Test: `tests/execution/test_lookup_lineage_unit.py` (extend the existing `_FakeML` harness and add tests)

**Interfaces:**
- Consumes: `_producers_of_dataset_members(dataset_rid) -> set[RID]` from Task 1.
- Produces: `_walk_node(..., extra_parent_rids: set[RID] | None = None)` — the new keyword-only parameter Task 3 uses for root seeding.

**Background the implementer needs:**
- `_walk_node` builds `parent_rids: set[RID]` from the execution's consumed datasets and input assets, then recurses on each parent RID. You will (a) add an `extra_parent_rids` keyword-only param merged into `parent_rids` before the recursion loop, and (b) in the consumed-dataset loop, also add the dataset's member-producers to `parent_rids`.
- The existing `_FakeML` harness in `tests/execution/test_lookup_lineage_unit.py` overrides `_producer_of_dataset`/`_producer_of_asset`. You must give it an override for `_producers_of_dataset_members` too (default: return `set()`), plus a scripting helper to set member-producers per dataset, so existing tests are unaffected (they get an empty set) and new tests can script member-producers.

- [ ] **Step 1: Extend the `_FakeML` harness (test infra)**

In `tests/execution/test_lookup_lineage_unit.py`, add to `_FakeML.__init__` a member-producer map and add an override + scripting helper. Insert the map init alongside the other maps (after `self._asset_producers = {}`):

```python
        # Map dataset_rid -> set of member-producing execution RIDs.
        self._dataset_member_producers: dict[str, set[str]] = {}
```

Add a scripting helper next to `add_dataset`:

```python
    def set_member_producers(self, dataset_rid: str, producers: set[str]) -> None:
        """Script the member-asset producing executions of a dataset."""
        self._dataset_member_producers[dataset_rid] = set(producers)
```

Add the override next to `_producer_of_dataset`:

```python
    def _producers_of_dataset_members(self, dataset_rid: str, version: Any = None) -> set[str]:  # type: ignore[override]
        return set(self._dataset_member_producers.get(dataset_rid, set()))
```

- [ ] **Step 2: Write the failing tests**

Add to `tests/execution/test_lookup_lineage_unit.py`:

```python
def test_walk_node_extra_parent_rids_attaches_as_parents():
    """extra_parent_rids passed to the root walk become parents of the root node."""
    ml = _FakeML()
    # EXE-UP produced some members; it consumed DS-SRC (no producer).
    ml.add_dataset("1-DSSR", producer=None)
    ml.add_execution("2-EXUP", input_datasets=[_StubDataset("1-DSSR")])
    # The dataset whose root walk we drive: produced by EXE-DS (version producer),
    # but its members were produced by EXE-UP.
    ml.add_dataset("1-DSIM", producer="2-EXDS")
    ml.add_execution("2-EXDS", input_datasets=[])
    ml.set_member_producers("1-DSIM", {"2-EXUP"})

    result = ml.lookup_lineage("1-DSIM")

    # Root node is the version-producer EXE-DS.
    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXDS"
    # EXE-UP (the member-producer) appears as a parent of the root.
    parent_rids = {p.execution.rid for p in result.lineage.parents}
    assert "2-EXUP" in parent_rids
    # And the walk continued into EXE-UP's consumed source dataset.
    up_node = next(p for p in result.lineage.parents if p.execution.rid == "2-EXUP")
    assert {d.rid for d in up_node.consumed_datasets} == {"1-DSSR"}


def test_mid_walk_consumed_dataset_member_producers_become_parents():
    """When an execution consumes a dataset whose members have a distinct
    producer, that producer is walked as a parent."""
    ml = _FakeML()
    # Source dataset consumed by the upload exec.
    ml.add_dataset("1-DSSR", producer=None)
    ml.add_execution("2-EXUP", input_datasets=[_StubDataset("1-DSSR")])
    # An intermediate image dataset: version-producer EXE-DS, members by EXE-UP.
    ml.add_dataset("1-DSIM", producer="2-EXDS")
    ml.set_member_producers("1-DSIM", {"2-EXUP"})
    ml.add_execution("2-EXDS", input_datasets=[])
    # A downstream execution that CONSUMES the image dataset as input.
    ml.add_execution("2-EXTR", input_datasets=[_StubDataset("1-DSIM")])
    ml.add_dataset("1-DSMO", producer="2-EXTR")

    result = ml.lookup_lineage("1-DSMO")

    # Root = EXE-TR; it consumed DS-IM. DS-IM's version producer EXE-DS AND its
    # member producer EXE-UP both appear among EXE-TR's parents.
    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXTR"
    parent_rids = {p.execution.rid for p in result.lineage.parents}
    assert "2-EXDS" in parent_rids  # version producer
    assert "2-EXUP" in parent_rids  # member producer (the new edge)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_lookup_lineage_unit.py -k "extra_parent_rids or mid_walk" -v`
Expected: FAIL — `_walk_node() got an unexpected keyword argument 'extra_parent_rids'` (first test) and the member-producer not appearing in parents (second test).

- [ ] **Step 4: Implement the `_walk_node` changes**

In `src/deriva_ml/core/mixins/execution.py`:

(a) Add the keyword-only parameter to the `_walk_node` signature:

```python
    def _walk_node(
        self,
        *,
        execution_rid: RID,
        depth_remaining: int | None,
        max_executions: int,
        visited_global: set[RID],
        in_progress: set[RID],
        flags: dict[str, bool],
        extra_parent_rids: set[RID] | None = None,
    ) -> "LineageNode | None":
```

(b) In the consumed-dataset loop, after the existing `producer = self._producer_of_dataset(ds.dataset_rid)` / `if producer: parent_rids.add(producer)` block, add member-producers:

```python
                producer = self._producer_of_dataset(ds.dataset_rid)
                if producer:
                    parent_rids.add(producer)
                # Members of this consumed dataset may have been produced by a
                # different execution than the one that assembled the dataset;
                # those member-producers are data-flow parents too.
                parent_rids |= self._producers_of_dataset_members(ds.dataset_rid)
```

(c) Immediately before the "Recurse on parents." block (where `parents: list[LineageNode] = []` is initialized), merge in `extra_parent_rids`:

```python
            # Root-seeded member-producers (and any other externally supplied
            # parents) are merged in before recursion so they get full
            # visited/cycle/depth handling.
            if extra_parent_rids:
                parent_rids |= extra_parent_rids
```

Make sure `extra_parent_rids` is merged AFTER `parent_rids` is fully built from inputs and BEFORE the `if depth_remaining is None or depth_remaining > 0:` recursion loop. Do not pass `extra_parent_rids` down into the recursive `_walk_node` calls (it applies only to the node it was given to).

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_lookup_lineage_unit.py -k "extra_parent_rids or mid_walk" -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Run the FULL unit file to confirm no regression**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_lookup_lineage_unit.py -v`
Expected: all pass (the original ~17 tests + the 2 new ones). The original tests get `set()` from the `_producers_of_dataset_members` override, so their parent sets are unchanged.

- [ ] **Step 7: Lint + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run ruff check src/deriva_ml/core/mixins/execution.py tests/execution/test_lookup_lineage_unit.py
uv run ruff format --check src/deriva_ml/core/mixins/execution.py tests/execution/test_lookup_lineage_unit.py
git add src/deriva_ml/core/mixins/execution.py tests/execution/test_lookup_lineage_unit.py
git commit -m "feat(lineage): _walk_node extra_parent_rids + mid-walk member-producer seeding"
```

---

### Task 3: Root-dataset seeding in `lookup_lineage`

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py` (`lookup_lineage`, the walk-dispatch block ~lines 1188-1200)
- Test: `tests/execution/test_lookup_lineage_unit.py` (add root-seeding tests)

**Interfaces:**
- Consumes: `_producers_of_dataset_members` (Task 1) and `_walk_node(..., extra_parent_rids=...)` (Task 2).
- Produces: the final `lookup_lineage` behavior for Dataset roots.

**Background the implementer needs:**
- `lookup_lineage` calls `root_descriptor, producer_rid = self._classify_rid(rid)`. For a Dataset root, `producer_rid` is the `Dataset_Version.Execution`. Today, if `producer_rid is None`, it returns an empty `LineageResult` immediately; otherwise it walks from `producer_rid`.
- You will add root-dataset member-producer seeding: compute `member_producers = self._producers_of_dataset_members(rid)` for Dataset roots, and:
  - **Both** version-producer and member-producers exist → walk from `producer_rid`, pass `extra_parent_rids = member_producers - {producer_rid}`.
  - **Only** member-producers (`producer_rid is None`) → walk from a deterministic representative `min(member_producers)`, pass the rest as `extra_parent_rids`.
  - **Neither** → empty result (unchanged).
- Only the Dataset root case is affected. Asset/Feature/Execution roots keep `extra_parent_rids=None` (do not compute member-producers for them).
- You need to know the root RID's type. `root_descriptor.type` is `"Dataset"` for dataset roots — use that to gate the seeding.

- [ ] **Step 1: Write the failing tests**

Add to `tests/execution/test_lookup_lineage_unit.py`:

```python
def test_root_dataset_surfaces_member_producer_when_both_exist():
    """lookup_lineage(image_dataset): version-producer is root, member-producer
    is a parent reaching the source — the tk-018 case."""
    ml = _FakeML()
    ml.add_dataset("1-DSSR", producer=None)  # source dataset
    ml.add_execution("2-EXUP", input_datasets=[_StubDataset("1-DSSR")])  # upload
    ml.add_execution("2-EXDS", input_datasets=[])  # datasets-phase (version producer)
    ml.add_dataset("1-DSIM", producer="2-EXDS")  # image dataset
    ml.set_member_producers("1-DSIM", {"2-EXUP"})

    result = ml.lookup_lineage("1-DSIM")

    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXDS"  # root = version producer
    up = next((p for p in result.lineage.parents if p.execution.rid == "2-EXUP"), None)
    assert up is not None, "member-producer must appear as a parent of the root"
    assert {d.rid for d in up.consumed_datasets} == {"1-DSSR"}  # reaches the source
    assert result.root.producing_execution is not None
    assert result.root.producing_execution.rid == "2-EXDS"  # contract preserved


def test_root_dataset_no_version_producer_walks_from_member_producers():
    """A dataset with NO version producer but WITH member producers yields a
    non-empty walk (previously this returned an empty LineageResult)."""
    ml = _FakeML()
    ml.add_dataset("1-DSSR", producer=None)
    ml.add_execution("2-EXUP", input_datasets=[_StubDataset("1-DSSR")])
    ml.add_dataset("1-DSIM", producer=None)  # no version producer
    ml.set_member_producers("1-DSIM", {"2-EXUP"})

    result = ml.lookup_lineage("1-DSIM")

    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXUP"  # representative root
    assert {d.rid for d in result.lineage.consumed_datasets} == {"1-DSSR"}
    assert result.root.producing_execution is not None
    assert result.root.producing_execution.rid == "2-EXUP"


def test_root_dataset_member_producer_equals_version_producer_no_dup():
    """If the version producer also produced the members, it is not listed as
    its own parent."""
    ml = _FakeML()
    ml.add_dataset("1-DSSR", producer=None)
    ml.add_execution("2-EXVP", input_datasets=[_StubDataset("1-DSSR")])
    ml.add_dataset("1-DSIM", producer="2-EXVP")
    ml.set_member_producers("1-DSIM", {"2-EXVP"})  # same exec

    result = ml.lookup_lineage("1-DSIM")

    assert result.lineage is not None
    assert result.lineage.execution.rid == "2-EXVP"
    # 2-EXVP must NOT appear as its own parent.
    assert all(p.execution.rid != "2-EXVP" for p in result.lineage.parents)


def test_root_dataset_no_producers_at_all_returns_empty_walk():
    """Neither version nor member producers -> empty walk (unchanged)."""
    ml = _FakeML()
    ml.add_dataset("1-DSIM", producer=None)
    # no member producers scripted -> empty set

    result = ml.lookup_lineage("1-DSIM")

    assert result.lineage is None
    assert result.root.producing_execution is None
    assert result.walked_complete is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_lookup_lineage_unit.py -k "root_dataset" -v`
Expected: FAIL — the member-producer is not surfaced (first two tests fail), because `lookup_lineage` does not yet seed member-producers for the root.

- [ ] **Step 3: Implement the root seeding in `lookup_lineage`**

In `src/deriva_ml/core/mixins/execution.py`, replace the walk-dispatch block (currently):

```python
        if producer_rid is None:
            # No producer — return a valid result with an empty walk.
            return LineageResult(root=root_descriptor)

        # 2. Walk iteratively from the producing execution.
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
        )
```

with:

```python
        # For a Dataset root, the members may have been produced by execution(s)
        # other than the one that assembled/versioned the dataset. Those
        # member-producers are data-flow parents and must be seeded into the
        # walk so e.g. lookup_lineage(image_dataset) reaches the source the
        # images were uploaded from. (See the lineage member-asset-traversal
        # design spec; tk-018.)
        extra_parent_rids: set[RID] = set()
        if root_descriptor.type == "Dataset":
            member_producers = self._producers_of_dataset_members(rid)
            if producer_rid is not None:
                extra_parent_rids = member_producers - {producer_rid}
            elif member_producers:
                # No version-producer but the members have producers: walk from
                # a deterministic representative; the rest become its parents.
                ordered = sorted(member_producers)
                producer_rid = ordered[0]
                extra_parent_rids = set(ordered[1:])

        if producer_rid is None:
            # No producer of any kind — return a valid result with an empty walk.
            return LineageResult(root=root_descriptor)

        # 2. Walk iteratively from the producing execution.
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
```

Leave the rest of `lookup_lineage` (the `root_descriptor.model_copy(...)` update and the `LineageResult(...)` return) unchanged.

- [ ] **Step 4: Run the root-seeding tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_lookup_lineage_unit.py -k "root_dataset" -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Run the FULL unit suite for execution to confirm no regression**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/ -v`
Expected: all pass (existing lineage/forward-lineage/provenance tests + the new ones). In particular `test_lineage_dataset_with_no_producer_returns_empty_walk` still passes (no member-producers scripted → empty set → empty walk).

- [ ] **Step 6: Lint + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run ruff check src/deriva_ml/core/mixins/execution.py tests/execution/test_lookup_lineage_unit.py
uv run ruff format --check src/deriva_ml/core/mixins/execution.py tests/execution/test_lookup_lineage_unit.py
git add src/deriva_ml/core/mixins/execution.py tests/execution/test_lookup_lineage_unit.py
git commit -m "feat(lineage): seed root-dataset member-asset producers into lookup_lineage"
```

---

### Task 4: Live end-to-end test + ADR consequences note

**Files:**
- Modify: `tests/execution/test_lookup_lineage_live.py` (add one gated test)
- Modify: `docs/adr/0001-lineage-walks-data-flow-not-orchestration.md` (append a one-line consequence)
- Test: the new live test is the deliverable.

**Interfaces:**
- Consumes: the full `lookup_lineage` behavior (Tasks 1-3), `test_ml` fixture, `tests/factories.py` helpers, and the `DERIVA_HOST` gate.

**Background the implementer needs:**
- The live test must construct the tk-018 shape on a real catalog: an execution that produces **asset members**, those assets added as members of a dataset, where that dataset's version-producer is a *different* execution. Then assert `lookup_lineage(<that dataset>)` surfaces, via a member-producer parent, the source the member-producing execution consumed.
- Read `tests/execution/test_lookup_lineage_live.py` (existing two tests) for the exact fixture/imports style, and `tests/factories.py` for member/asset/execution helpers (`make_execution`, `make_dataset`, `make_test_files`, `assert_dataset_has_members`). Use the highest-level APIs that produce the needed shape; mirror how the cifar two-execution flow does it (upload exec produces Image/File assets + consumes a source dataset; a separate exec assembles them into a dataset). If a faithful asset-member + distinct-version-producer shape is impractical to build through public APIs in this test harness, construct the minimal equivalent: (1) exec_src produces source dataset DS_SRC; (2) exec_up consumes DS_SRC and produces asset members; (3) build dataset DS_IM whose members are those assets and whose version-producer is a third exec_ds (or DS_IM created by exec_ds with the assets added as members). The essential invariant: `DS_IM`'s `Dataset_Version.Execution` != the asset members' Output producer.
- Gate exactly like the existing tests: `@pytest.mark.skipif(not os.environ.get("DERIVA_HOST"), reason="lookup_lineage live smoke test requires DERIVA_HOST")`.

- [ ] **Step 1: Write the live test**

Add to `tests/execution/test_lookup_lineage_live.py` (after the existing tests). Adapt the asset/member construction to the real helpers available — the assertion block is the contract:

```python
@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="lookup_lineage live smoke test requires DERIVA_HOST",
)
def test_lookup_lineage_descends_into_member_asset_producers(test_ml):
    """tk-018: lookup_lineage(dataset) reaches the source the dataset's MEMBER
    assets were produced from, even when a different execution versioned the
    dataset.

    Shape built on the catalog:
        exec_src  -> DS_SRC (source dataset)
        exec_up   consumes DS_SRC, produces asset members  (member producer)
        DS_IM     version-produced by exec_ds, members = exec_up's assets
    Assert: lookup_lineage(DS_IM) surfaces exec_up as a parent (reached via the
    member assets), and exec_up's consumed DS_SRC appears under it.
    """
    # Build exec_src -> DS_SRC, exec_up (consumes DS_SRC, produces asset
    # members), and DS_IM (versioned by exec_ds, members = those assets) using
    # the catalog APIs / factories. See test_lookup_lineage_two_execution_chain
    # above and tests/factories.py for the construction helpers.
    #
    # ds_im_rid       = RID of DS_IM
    # ds_src_rid      = RID of DS_SRC
    # exec_up_rid     = RID of exec_up (the member-asset producer)
    # exec_ds_rid     = RID of exec_ds (DS_IM's version producer)
    ...

    result = test_ml.lookup_lineage(ds_im_rid)

    assert isinstance(result, LineageResult)
    assert result.root.type == "Dataset"
    assert result.root.rid == ds_im_rid

    # The member-producer (exec_up) is reachable as a parent somewhere in the
    # tree (directly under the root version-producer node).
    assert result.lineage is not None
    parent_rids = {p.execution.rid for p in result.lineage.parents}
    assert exec_up_rid in parent_rids, (
        f"member-producer {exec_up_rid} not surfaced; parents were {parent_rids}"
    )

    # And from exec_up the walk reaches the source dataset DS_SRC it consumed.
    up_node = next(p for p in result.lineage.parents if p.execution.rid == exec_up_rid)
    consumed_src = {d.rid for d in up_node.consumed_datasets}
    assert ds_src_rid in consumed_src, (
        f"source dataset {ds_src_rid} not reached via member-producer; "
        f"consumed were {consumed_src}"
    )
```

The implementer MUST replace the `...` with real construction code using the catalog APIs (study the existing live test + factories first), so the test actually builds the shape and runs. Do not leave the ellipsis.

- [ ] **Step 2: Run the live test (gated)**

Run (with a live catalog host configured): `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_HOST=<host> uv run python -m pytest tests/execution/test_lookup_lineage_live.py::test_lookup_lineage_descends_into_member_asset_producers -v`
Expected: PASS. Without `DERIVA_HOST`: SKIPS.
If a live host is not available in the implementer's environment, report that the live test could not be executed, confirm it SKIPS cleanly without the env var, and confirm it imports/collects without error (`uv run python -m pytest tests/execution/test_lookup_lineage_live.py --collect-only`). Do NOT fake a pass.

- [ ] **Step 3: Confirm the skip path and collection**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_lookup_lineage_live.py -v`
Expected: all live tests SKIP (no `DERIVA_HOST`), no collection errors.

- [ ] **Step 4: Append the ADR consequence note**

In `docs/adr/0001-lineage-walks-data-flow-not-orchestration.md`, add one bullet to the Consequences (or a short "Update" note at the end):

```markdown
- **Update (2026-06-26):** the data-flow walk descends into a dataset's
  *member assets* — their `<AssetTable>_Execution` Output producers are
  data-flow parents of the dataset, surfaced by `lookup_lineage`. This stays
  within the data-flow doctrine (it is the same Output-edge rule applied to a
  dataset's members) and adds no orchestration traversal. See
  `docs/superpowers/specs/2026-06-26-lineage-member-asset-traversal-design.md`.
```

- [ ] **Step 5: Lint + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run ruff check tests/execution/test_lookup_lineage_live.py
uv run ruff format --check tests/execution/test_lookup_lineage_live.py
git add tests/execution/test_lookup_lineage_live.py docs/adr/0001-lineage-walks-data-flow-not-orchestration.md
git commit -m "test(lineage): live end-to-end member-asset traversal + ADR-0001 note"
```

---

## Final verification (after all tasks)

- [ ] Run the full execution test directory: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/ -v` — all pass (live tests skip without `DERIVA_HOST`).
- [ ] Run the broader suite to confirm no collateral breakage in dataset/lineage-adjacent tests: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/ -q` — green (modulo any pre-existing env-gated skips).
- [ ] Lint the whole touched surface: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src tests`.
