# Lineage Member-Producer Membership-Join Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `lookup_lineage` from 404ing on real datasets by replacing the client-side member-RID `.in_()` filter with a server-side membership join, so the request URL only ever carries the single dataset RID.

**Architecture:** Rewrite `_producers_of_dataset_members` to discover member asset tables via `find_associations` (no member-RID enumeration), and rewrite `_distinct_member_output_producers` to build a server-side join `Dataset_<member>(Dataset==rid) → <member> → <member>_Execution(Asset_Role="Output") → distinct Execution` on the dataset's **version-snapshot** pathBuilder (preserving consumed-version faithfulness). Delete `_MEMBER_PRODUCER_CHUNK`.

**Tech Stack:** Python 3.13, `uv`, pytest, deriva-ml core (`deriva_ml.core.mixins.execution.ExecutionMixin`), ERMrest PathBuilder.

## Global Constraints

- Work in `../deriva-ml` (`/Users/carl/GitHub/DerivaML/deriva-ml`) on branch `feature/lineage-member-producer-join` (already created; the spec is committed there).
- Use `uv` for everything: `uv run python -m pytest` (NOT `uv run pytest`), `uv run ruff check`, `uv run ruff format`.
- Always `cd /Users/carl/GitHub/DerivaML/deriva-ml && <cmd>` in every Bash call — the shell CWD is not persistent.
- Google-style docstrings on every changed method.
- NO public-model / contract change: `src/deriva_ml/execution/lineage.py` is NOT modified. `_producers_of_dataset_members`'s public signature `(dataset_rid, version=None) -> set[RID]` is unchanged.
- The member-producer query MUST NOT place a client-side member-RID list in the URL (no `.in_()` over member RIDs). The URL carries only `Dataset == dataset_rid`.
- Version faithfulness: build the join on `dataset._version_snapshot_catalog(version).pathBuilder()` (the same snapshot pathBuilder `list_dataset_members` uses), NOT the plain `self.pathBuilder()`.
- Delete `_MEMBER_PRODUCER_CHUNK` from `execution.py`. Do NOT touch `_VERSION_RID_CHUNK` in `_helpers.py` (different code path, not URL-length-bound at realistic scale).
- The live regression test must use a **≥200-member** dataset (a 1-member test cannot catch the URL-length class).

## Reference: the discovery + link pattern to mirror

`list_dataset_members` (`src/deriva_ml/dataset/dataset.py`, ~lines 1689-1717) already does the member-table discovery and the membership-link build. Mirror it:

```python
version_snapshot_catalog = self._version_snapshot_catalog(version)   # Dataset method
pb = version_snapshot_catalog.pathBuilder()
for assoc_table in self._dataset_table.find_associations():
    other_fkey = assoc_table.other_fkeys.pop()
    target_table = other_fkey.pk_table          # the member table (e.g. Image)
    member_table = assoc_table.table            # the membership table (e.g. Dataset_Image)
    member_column = other_fkey.foreign_key_columns[0].name   # FK col on membership -> member
    target_column = other_fkey.referenced_columns[0].name    # referenced col on member (often RID)
    target_path = pb.schemas[target_table.schema.name].tables[target_table.name]
    member_path = pb.schemas[member_table.schema.name].tables[member_table.name]
    # membership filtered by dataset, linked to the member table:
    path = member_path.filter(member_path.Dataset == self.dataset_rid).link(
        target_path, on=(member_path.columns[member_column] == target_path.columns[target_column])
    )
```

`find_association(member_table, "Execution") -> (assoc_exec_table, member_link_col, exec_link_col)` returns the `<member>_Execution` association table and the names (strings) of its two FK columns.

Facts for the implementer:
- `self._dataset_table` is available on the `ExecutionMixin` host (it's the deriva-ml `Dataset` table; `ml_instance._dataset_table` is referenced elsewhere in `dataset.py`).
- `self.lookup_dataset(dataset_rid)` returns a `Dataset` object whose `_version_snapshot_catalog(version)` returns a catalog (the live `ml_instance` when `version` is None, else a snapshot-bound catalog) exposing `.pathBuilder()`, and whose `dataset_rid` attribute is the RID. The `Dataset` also exposes `_dataset_table` and `find_associations` via its own `_dataset_table`.
- `self.model.is_asset(table)` / `self.model.find_association(table, "Execution")` / `NoAssociationException` are already imported/used in this module.

## File Structure

- `src/deriva_ml/core/mixins/execution.py` — rewrite `_producers_of_dataset_members` + `_distinct_member_output_producers`; delete `_MEMBER_PRODUCER_CHUNK`.
- `tests/execution/test_producers_of_dataset_members.py` — rework the offline harness + tests for the new structure; add a join-shape test asserting no member-RID `.in_()`.
- `tests/execution/test_lookup_lineage_live.py` — add the ≥200-member live regression.

---

### Task 1: membership-join member-producer query (rewrite + offline tests)

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py` (`_producers_of_dataset_members` ~1418-1462, `_distinct_member_output_producers` ~1464-1502, delete `_MEMBER_PRODUCER_CHUNK` ~line 52)
- Modify: `tests/execution/test_producers_of_dataset_members.py` (rework harness + tests)

**Interfaces:**
- Unchanged public: `_producers_of_dataset_members(self, dataset_rid, version=None) -> set[RID]`.
- New internal: `_distinct_member_output_producers(self, snapshot_pb, membership_table, member_table, member_link_column, target_column, dataset_rid) -> set[RID]` (a server-side join; NO member-RID list).

**Background:** see the "Reference" section above. The current implementation enumerates member RIDs via `list_dataset_members` and filters `<member>_Execution` by `.in_(member_rids)` — which blows the URL. The rewrite discovers member tables via `find_associations` and joins server-side.

- [ ] **Step 1: Rework the offline harness**

In `tests/execution/test_producers_of_dataset_members.py`, the `_FakeMembersML` harness currently scripts `list_dataset_members` + overrides `_distinct_member_output_producers(asset_table, member_rids)`. The rewrite changes both the discovery path and the seam signature. Replace the harness's relevant parts so it scripts the NEW structure:

```python
from types import SimpleNamespace
from unittest.mock import MagicMock

from deriva_ml.core.mixins.execution import ExecutionMixin


class _FakeAssoc:
    """Stand-in for a membership association table from find_associations()."""
    def __init__(self, membership_name, member_table_name, member_col="Image", target_col="RID"):
        self.table = SimpleNamespace(name=membership_name, schema=SimpleNamespace(name="domain"))
        member_tbl = SimpleNamespace(name=member_table_name, schema=SimpleNamespace(name="domain"))
        ofk = SimpleNamespace(
            pk_table=member_tbl,
            foreign_key_columns=[SimpleNamespace(name=member_col)],
            referenced_columns=[SimpleNamespace(name=target_col)],
        )
        # other_fkeys.pop() must return ofk
        self.other_fkeys = [ofk]


class _FakeMembersML(ExecutionMixin):
    """Scripts what the rewritten _producers_of_dataset_members needs:
    a dataset table whose find_associations() yields membership associations,
    is_asset() per member table, and a seam (_distinct_member_output_producers)
    that returns scripted producer sets per member table.
    """

    def __init__(self):
        # member_table_name -> set of producer RIDs (the seam result)
        self._producers_by_table: dict[str, set[str]] = {}
        # which member tables are asset tables
        self._asset_tables: set[str] = set()
        # membership associations the dataset table "has"
        self._assocs: list[_FakeAssoc] = []

        self.model = MagicMock()
        self.model.is_asset = lambda table: getattr(table, "name", table) in self._asset_tables

        # _dataset_table.find_associations() -> self._assocs
        self._dataset_table = SimpleNamespace(find_associations=lambda: list(self._assocs))

    # scripting --------------------------------------------------------------
    def add_member_table(self, membership_name, member_table_name, *, is_asset=True,
                         producers=frozenset(), member_col="Image", target_col="RID"):
        self._assocs.append(_FakeAssoc(membership_name, member_table_name, member_col, target_col))
        if is_asset:
            self._asset_tables.add(member_table_name)
        self._producers_by_table[member_table_name] = set(producers)

    # mocked primitives ------------------------------------------------------
    def lookup_dataset(self, rid):  # type: ignore[override]
        snap = MagicMock()
        snap.pathBuilder = lambda: MagicMock()
        ds = SimpleNamespace(
            dataset_rid=rid,
            _version_snapshot_catalog=lambda version: snap,
            _dataset_table=self._dataset_table,
        )
        return ds

    def _distinct_member_output_producers(self, snapshot_pb, membership_table, member_table,  # type: ignore[override]
                                          member_link_column, target_column, dataset_rid):
        # Seam: return scripted producers for this member table.
        return set(self._producers_by_table.get(getattr(member_table, "name", member_table), set()))
```

(Adjust the seam override's parameter list to EXACTLY match the new real signature you settle on in Step 3 — the harness override and the real method must agree.)

- [ ] **Step 2: Write the failing tests (logic via the seam)**

Rewrite the existing logic tests for the new harness and keep their intent (dedup across tables, skip non-asset, empty when none). Replace the file's test bodies with:

```python
def test_dedup_across_member_tables():
    ml = _FakeMembersML()
    ml.add_member_table("Dataset_Image", "Image", producers={"2-EXUP"})
    assert ml._producers_of_dataset_members("1-DSAA") == {"2-EXUP"}


def test_union_across_multiple_asset_tables():
    ml = _FakeMembersML()
    ml.add_member_table("Dataset_Image", "Image", producers={"2-EXAA"})
    ml.add_member_table("Dataset_File", "File", producers={"2-EXAB"}, member_col="File")
    assert ml._producers_of_dataset_members("1-DSAA") == {"2-EXAA", "2-EXAB"}


def test_non_asset_member_tables_skipped():
    ml = _FakeMembersML()
    ml.add_member_table("Dataset_Image", "Image", producers={"2-EXAA"})
    ml.add_member_table("Dataset_Dataset", "Dataset", is_asset=False, producers={"2-EXNO"})
    assert ml._producers_of_dataset_members("1-DSAA") == {"2-EXAA"}


def test_no_producers_returns_empty():
    ml = _FakeMembersML()
    ml.add_member_table("Dataset_Image", "Image", producers=set())
    assert ml._producers_of_dataset_members("1-DSAA") == set()


def test_no_member_tables_returns_empty():
    ml = _FakeMembersML()
    assert ml._producers_of_dataset_members("1-DSAA") == set()
```

- [ ] **Step 3: Run to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_producers_of_dataset_members.py -k "dedup or union or non_asset or no_producers or no_member_tables" -v`
Expected: FAIL — the current `_producers_of_dataset_members` calls `list_dataset_members` + the old `_distinct_member_output_producers(asset_table, member_rids)` seam, which the new harness no longer provides; the discovery-via-`find_associations` path doesn't exist yet.

- [ ] **Step 4: Rewrite `_producers_of_dataset_members`**

Replace the method in `src/deriva_ml/core/mixins/execution.py` with the discovery-via-associations version (no RID enumeration):

```python
    def _producers_of_dataset_members(self, dataset_rid: RID, version: Any | None = None) -> set[RID]:
        """Distinct executions that produced the member assets of a dataset.

        For each member ASSET table, collects the distinct producing executions
        (the asset's ``<Asset>_Execution`` association with
        ``Asset_Role="Output"``) via a server-side membership join — the request
        URL carries only the dataset RID, never a client-side member-RID list,
        so this is safe for datasets with thousands of members. Deduplicated
        across all member tables. Nested-``Dataset`` and non-asset member kinds
        are skipped — dataset producers are handled by :meth:`_producer_of_dataset`.

        The work is bounded by the number of member *asset tables* (typically
        1-2), independent of member count.

        Args:
            dataset_rid: RID of the dataset whose member assets to inspect.
            version: Optional dataset version. ``None`` uses the current version;
                a version resolves the membership join against that version's
                catalog snapshot (consumed-version faithfulness).

        Returns:
            Set of distinct producing-execution RIDs. Empty when the dataset has
            no member assets or none have a recorded ``Output`` producer.

        Example:
            >>> producers = ml._producers_of_dataset_members("1-DSAA")  # doctest: +SKIP
            >>> sorted(producers)  # doctest: +SKIP
            ['2-EXUP']
        """
        dataset = self.lookup_dataset(dataset_rid)
        snapshot_pb = dataset._version_snapshot_catalog(version).pathBuilder()
        producers: set[RID] = set()
        for assoc_table in dataset._dataset_table.find_associations():
            other_fkey = assoc_table.other_fkeys.pop()
            member_table = other_fkey.pk_table          # the member asset table (e.g. Image)
            membership_table = assoc_table.table        # the membership table (e.g. Dataset_Image)
            if not self.model.is_asset(member_table):
                # Nested-Dataset / non-asset member kinds are not asset-producer-shaped.
                continue
            member_link_column = other_fkey.foreign_key_columns[0].name
            target_column = other_fkey.referenced_columns[0].name
            producers |= self._distinct_member_output_producers(
                snapshot_pb,
                membership_table,
                member_table,
                member_link_column,
                target_column,
                dataset.dataset_rid,
            )
        return producers
```

- [ ] **Step 5: Rewrite `_distinct_member_output_producers` as a server-side join**

Replace the method with:

```python
    def _distinct_member_output_producers(
        self,
        snapshot_pb: Any,
        membership_table: Any,
        member_table: Any,
        member_link_column: str,
        target_column: str,
        dataset_rid: RID,
    ) -> set[RID]:
        """Distinct ``Output`` producing executions of a dataset's members of one
        asset table, via a server-side membership join.

        Builds the path
        ``<membership>(Dataset==dataset_rid) → <member> → <member>_Execution
        (Asset_Role="Output")`` on ``snapshot_pb`` and projects distinct
        ``Execution``. The request URL carries only the dataset RID — no
        client-side member-RID list — so it is safe at any member count.

        Args:
            snapshot_pb: Version-snapshot pathBuilder for the dataset.
            membership_table: The ``Dataset_<member>`` membership table.
            member_table: The member asset table (e.g. ``Image``).
            member_link_column: FK column on the membership table referencing
                the member.
            target_column: The referenced column on the member table (often
                ``"RID"``).
            dataset_rid: RID of the dataset whose members to inspect.

        Returns:
            Set of distinct producing-execution RIDs (``Output`` role). Empty if
            the member asset table has no ``<member>_Execution`` association.
        """
        try:
            exec_assoc, member_exec_fk, _exec_fk = self.model.find_association(member_table, "Execution")
        except NoAssociationException:
            return set()

        membership_path = snapshot_pb.schemas[membership_table.schema.name].tables[membership_table.name]
        member_path = snapshot_pb.schemas[member_table.schema.name].tables[member_table.name]
        exec_path = snapshot_pb.schemas[exec_assoc.schema.name].tables[exec_assoc.name]

        path = (
            membership_path.filter(membership_path.Dataset == dataset_rid)
            .link(
                member_path,
                on=(membership_path.columns[member_link_column] == member_path.columns[target_column]),
            )
            .link(
                exec_path,
                on=(member_path.columns[target_column] == exec_path.columns[member_exec_fk]),
            )
            .filter(exec_path.Asset_Role == "Output")
        )
        producers: set[RID] = set()
        for row in path.attributes(exec_path.Execution).fetch():
            exec_rid = row.get("Execution")
            if exec_rid:
                producers.add(exec_rid)
        return producers
```

Note on the second `.link`: it joins the member table to the `<member>_Execution` association on the member's referenced column (`target_column`, typically `RID`) matching the association's FK to the member (`member_exec_fk`). If the live test shows the link direction/columns need adjusting (e.g. the association references the member by a different column), fix the `on=` to match what the model exposes — the INVARIANT is that no member-RID list appears in the URL and the result equals the member-producer set. Verify against the live catalog in Task 2.

- [ ] **Step 6: Delete `_MEMBER_PRODUCER_CHUNK`**

Remove the `_MEMBER_PRODUCER_CHUNK = 500` constant (and its comment) near the top of `src/deriva_ml/core/mixins/execution.py` (~line 52). Confirm no other reference remains: `cd /Users/carl/GitHub/DerivaML/deriva-ml && grep -rn "_MEMBER_PRODUCER_CHUNK" src/` should return nothing.

- [ ] **Step 7: Run the logic tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_producers_of_dataset_members.py -v`
Expected: the 5 logic tests PASS (the seam override returns scripted producers; the discovery loop iterates the scripted associations, skips non-asset, dedups/unions).

- [ ] **Step 8: Add the join-shape test (drives the REAL join helper)**

This is the regression guard for the bug class — it must prove the real `_distinct_member_output_producers` builds a `Dataset==rid` filter and NO member-RID `.in_()`. Append to `tests/execution/test_producers_of_dataset_members.py`:

```python
def test_join_filters_by_dataset_not_member_rid_in():
    """The real join filters membership by Dataset==rid and never builds an
    .in_() over member RIDs (the tk-023 URL-length bug class)."""
    calls = {"dataset_filter": False, "member_in_called": False, "asset_role_output": False}

    # Column whose .in_() would be the BUG — flag it if ever called.
    member_col = MagicMock()
    member_col.in_ = lambda *a, **k: calls.__setitem__("member_in_called", True) or MagicMock()

    def _make_table(name):
        t = MagicMock()
        t.columns = {"RID": member_col, "Image": MagicMock()}
        # membership_path.Dataset == rid  -> record it
        def _eq(other):
            calls["dataset_filter"] = True
            return MagicMock()
        t.Dataset = MagicMock()
        t.Dataset.__eq__ = lambda self_, other: _eq(other)
        # Asset_Role == "Output"
        t.Asset_Role = MagicMock()
        t.Asset_Role.__eq__ = lambda self_, other: calls.__setitem__("asset_role_output", other == "Output") or MagicMock()
        t.Execution = MagicMock()
        # chainable path: filter/link return a path with attributes().fetch()
        path = MagicMock()
        path.filter.return_value = path
        path.link.return_value = path
        path.attributes.return_value.fetch.return_value = [{"Execution": "2-EXUP"}]
        t.filter = path.filter
        return t, path

    membership_t, mpath = _make_table("Dataset_Image")
    member_t, _ = _make_table("Image")
    exec_t, _ = _make_table("Image_Execution")

    schema = MagicMock()
    schema.name = "domain"
    schema.tables = {"Dataset_Image": membership_t, "Image": member_t, "Image_Execution": exec_t}
    pb = MagicMock()
    pb.schemas = {"domain": schema}

    ml = ExecutionMixin.__new__(ExecutionMixin)
    ml.model = MagicMock()
    # find_association(member, "Execution") -> (exec_assoc_table, member_fk, exec_fk)
    exec_assoc = SimpleNamespace(name="Image_Execution", schema=SimpleNamespace(name="domain"))
    ml.model.find_association = lambda mt, target: (exec_assoc, "Image", "Execution")

    membership_table = SimpleNamespace(name="Dataset_Image", schema=SimpleNamespace(name="domain"))
    member_table = SimpleNamespace(name="Image", schema=SimpleNamespace(name="domain"))

    result = ml._distinct_member_output_producers(
        pb, membership_table, member_table, "Image", "RID", "1-DSAA"
    )

    assert result == {"2-EXUP"}
    assert calls["dataset_filter"] is True, "must filter membership by Dataset==rid"
    assert calls["member_in_called"] is False, "must NOT build an .in_() over member RIDs (tk-023)"
```

Note: the mock column wiring is intricate; if a particular attribute path doesn't line up with your real implementation's exact calls, adjust the mock to match — but the two `assert`s that MUST hold are `dataset_filter is True` and `member_in_called is False`. If `__eq__` mocking proves brittle, an acceptable alternative is a lighter spy that records whether `.in_` was ever accessed on any member column and whether a `Dataset`-keyed filter was issued; do NOT weaken to "result is non-empty only".

- [ ] **Step 9: Run the join-shape test + full file**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_producers_of_dataset_members.py -v`
Expected: all PASS (5 logic + the join-shape test).

- [ ] **Step 10: No-regression across the lineage surface**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_lookup_lineage_unit.py tests/execution/test_input_datasets_with_versions.py -q`
Expected: green. The walk unit tests script member producers via the `_FakeML` override of `_producers_of_dataset_members`, which is mechanism-independent, so they are unaffected.

- [ ] **Step 11: Lint + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run ruff check src/deriva_ml/core/mixins/execution.py tests/execution/test_producers_of_dataset_members.py
uv run ruff format --check src/deriva_ml/core/mixins/execution.py tests/execution/test_producers_of_dataset_members.py
git add src/deriva_ml/core/mixins/execution.py tests/execution/test_producers_of_dataset_members.py
git commit -m "fix(lineage): member-producer query via server-side membership join (no member-RID URL list) — tk-023"
```

---

### Task 2: ≥200-member live regression test

**Files:**
- Modify: `tests/execution/test_lookup_lineage_live.py` (add one DERIVA_HOST-gated test)

**Interfaces:**
- Consumes: the rewritten `_producers_of_dataset_members` / `lookup_lineage` (Task 1), the `test_ml` fixture, `tests/factories.py`, the `DERIVA_HOST` gate.

**Background:** The bug only manifests at scale — a dataset with hundreds of asset members produces a URL too long for the OLD `.in_()` code. A 1-member test cannot catch it. This test builds a dataset with ≥200 (target ~500) asset members produced by an upstream execution and asserts `lookup_lineage` (or `_producers_of_dataset_members`) returns that producer WITHOUT a 404. Read the existing live tests + `tests/factories.py` first for the construction APIs.

- [ ] **Step 1: Write the live test**

Add to `tests/execution/test_lookup_lineage_live.py` (after the existing tests). Replace the `...` with real construction — study the existing live tests + factories; the ASSERTION block is the contract:

```python
@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="lookup_lineage live smoke test requires DERIVA_HOST",
)
def test_member_producer_query_scales_to_many_members(test_ml):
    """tk-023: the member-producer query must not 404 on a dataset with
    hundreds of members. Builds a dataset whose >=200 asset members are
    produced by one upstream execution, then asserts lookup_lineage surfaces
    that producer (the OLD client-side .in_() over 200+ RIDs would 404)."""
    N_MEMBERS = 250  # > the ~hundreds threshold where the old URL blew up
    # Build exec_up that produces N_MEMBERS asset members; assemble them into
    # DS_BIG (versioned by exec_ds). Use the catalog APIs / factories — see
    # the existing member-asset live test for how to create asset members and
    # attach them to a dataset, then loop to reach N_MEMBERS.
    #
    #   ds_big_rid   = RID of DS_BIG (the >=250-member dataset)
    #   exec_up_rid  = RID of the execution that produced the members
    ...

    # The core assertion: this must NOT raise (the old code 404'd here).
    result = test_ml.lookup_lineage(ds_big_rid)

    seen = set()
    def _collect(node):
        if node is None:
            return
        seen.add(node.execution.rid)
        for p in node.parents:
            _collect(p)
    _collect(result.lineage)

    assert exec_up_rid in seen, (
        f"member-producer {exec_up_rid} not surfaced for a {N_MEMBERS}-member "
        f"dataset; saw {seen}"
    )
    assert result.cycle_detected is False
```

The implementer MUST replace the `...` with real construction reaching ≥200 members (document the exact count). If building ≥200 members is impractical in the harness, build the largest count feasible that still exceeds the old URL limit (a few hundred short RIDs ≈ a few KB; aim for ≥200) and document it. Do NOT leave the ellipsis; do NOT weaken the no-404 assertion.

- [ ] **Step 2: Run the live test (gated)**

Run (localhost container is up; `test_ml` defaults `DERIVA_HOST=localhost`, manages its own catalog; set `DERIVA_ML_ALLOW_DIRTY=true` for workflow creation on a dirty tree):
`cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_HOST=localhost DERIVA_ML_ALLOW_DIRTY=true uv run python -m pytest tests/execution/test_lookup_lineage_live.py::test_member_producer_query_scales_to_many_members -v`
Expected: PASS (no 404; producer surfaced). This build creates ~250 members + uploads, so it may take a few minutes — do NOT kill it early.
If no live host is available, report that, confirm the test SKIPS without `DERIVA_HOST` and collects cleanly — do NOT fake a pass.

- [ ] **Step 3: Confirm the existing live tests still pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_HOST=localhost DERIVA_ML_ALLOW_DIRTY=true uv run python -m pytest tests/execution/test_lookup_lineage_live.py -v`
Expected: all live tests PASS — `test_lookup_lineage_descends_into_member_asset_producers` and `test_lookup_lineage_reflects_consumed_version_not_latest` must still produce the same producers via the new join (the join must return the same set for their small datasets, and the consumed-version test confirms version faithfulness through the snapshot pathBuilder).
If no container: confirm skip + clean collection, report.

- [ ] **Step 4: Confirm skip path + collection (offline)**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/test_lookup_lineage_live.py -v`
Expected: all live tests SKIP (no `DERIVA_HOST`), no collection errors.

- [ ] **Step 5: Lint + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
uv run ruff check tests/execution/test_lookup_lineage_live.py
uv run ruff format --check tests/execution/test_lookup_lineage_live.py
git add tests/execution/test_lookup_lineage_live.py
git commit -m "test(lineage): >=200-member live regression for the membership-join (tk-023)"
```

---

## Final verification (after both tasks)

- [ ] Full execution test dir: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run python -m pytest tests/execution/ -q` — pass (live tests skip without `DERIVA_HOST`; the 2 pre-existing `test_workflow_creation_*` failures are unrelated).
- [ ] No `_MEMBER_PRODUCER_CHUNK` remains: `cd /Users/carl/GitHub/DerivaML/deriva-ml && grep -rn "_MEMBER_PRODUCER_CHUNK" src/` — empty.
- [ ] Live smoke against the real small dataset (the original bug): with a localhost CIFAR catalog (e.g. 278) present, `lookup_lineage('<a 500-member Small_* dataset>')` returns WITHOUT a 404 and reaches the source File dataset. (If catalog 278 still exists, `ml.lookup_lineage('1MEP')` should now work at default settings.)
- [ ] Lint the touched surface: `cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src tests`.
