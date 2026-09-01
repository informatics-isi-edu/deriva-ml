# `lookup_provenance` Implementation Plan (issue #383)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One call — `ml.lookup_provenance(rid)` — returning the complete provenance closure of an artifact (`ProvenanceClosure`), built on a single arc-gated walk engine shared with `lookup_lineage`, whose observable contract stays byte-identical.

**Architecture:** Extract the existing `_walk_node` recursion into `_provenance_engine.WalkEngine` parameterized by a visitor; `lookup_lineage` becomes a TreeBuilder visitor over consumption arcs only (golden `model_dump()` captures prove byte-equivalence); `lookup_provenance` is a ClosureBuilder visitor with all arcs, expanding every discovered execution, following version-authorship (≤ walked version), member bindings (via a diagnostic-returning variant of the #385 machinery), and snapshot-strict dataset ancestry. Gaps are first-class results.

**Tech Stack:** Python ≥3.12, Pydantic v2 (`BaseModel`, `@validate_call`), `StrEnum`, pytest with the existing `_FakeML` seam-override harness.

**Spec:** `docs/superpowers/specs/2026-08-31-lookup-provenance-design.md` (approved v2.2). The six rulings in its §2 are non-negotiable constraints.

## Global Constraints

- Branch: `feat/383-lookup-provenance`; all work lands via ONE PR to `main`. Never commit to `main` directly.
- Every command: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run <cmd>` (chain `cd` in the same Bash call; never bare `pytest`).
- TDD: every behavioral change shows its test failing first.
- RIDs in tests are generated (`_rid(n)`-style) and asserted only against flow-through values — never human-written literals.
- `lookup_lineage` observable output must remain byte-identical (Task 1 goldens are the gate; they run in every subsequent task).
- Enums are `StrEnum` for vocabularies deriva-ml closes; catalog-sourced strings (`status`, `element_type`, `asset_table`) stay `str`.
- Public models are Pydantic; engine internals may be `@dataclass`.
- Google docstrings with `Example:` blocks; catalog-touching examples carry `# doctest: +SKIP`.
- Lint before every commit: `uv run ruff check --fix src tests && uv run ruff format src tests`.
- tk-023: no unbounded `.in_()`/disjunctions — reuse `_execution_summaries`' chunking (`_SUMMARY_CHUNK_SIZE`).

---

### Task 1: Golden byte-equivalence captures for `lookup_lineage`

Locks the pre-refactor behavior. The goldens are JSON files committed to the repo, produced by the CURRENT implementation, compared verbatim after every later task.

**Files:**
- Create: `tests/execution/test_lineage_goldens.py`
- Create: `tests/execution/goldens/` (7 JSON files, generated)

**Interfaces:**
- Consumes: `_FakeML` from `tests/execution/test_lookup_lineage_unit.py` (import it — do not copy).
- Produces: `pytest tests/execution/test_lineage_goldens.py` as the regression gate every later task runs.

- [ ] **Step 1: Write the golden module with scenario builders and a regeneration path**

```python
"""Golden byte-equivalence captures for lookup_lineage (issue #383).

The engine extraction (ruling 5) must leave lookup_lineage's observable
output byte-identical. These goldens are model_dump(mode="json") captures
of the PRE-refactor implementation across the pinned scenario matrix.
Regenerate ONLY before the extraction lands, never after:

    UPDATE_LINEAGE_GOLDENS=1 uv run pytest tests/execution/test_lineage_goldens.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.execution.test_lookup_lineage_unit import _FakeML

GOLDEN_DIR = Path(__file__).parent / "goldens"


def _rid(n: int) -> str:
    return f"1-{n:04X}"


def _scenario_simple_chain() -> tuple[_FakeML, str, dict]:
    ml = _FakeML()
    ml.add_workflow(_rid(900))
    ml.add_execution(_rid(10), workflow=_rid(900))
    ml.add_dataset(_rid(1), description="root ds", producer=_rid(10))
    return ml, _rid(1), {}


def _scenario_two_level() -> tuple[_FakeML, str, dict]:
    ml = _FakeML()
    ml.add_workflow(_rid(900))
    ml.add_execution(_rid(11), workflow=_rid(900))
    ml.add_dataset(_rid(2), description="mid ds", producer=_rid(11))
    ml.add_execution(_rid(10), workflow=_rid(900), input_datasets=[(_rid(2), "1.0.0")])
    ml.add_dataset(_rid(1), description="root ds", producer=_rid(10))
    return ml, _rid(1), {}


def _scenario_diamond() -> tuple[_FakeML, str, dict]:
    ml = _FakeML()
    ml.add_workflow(_rid(900))
    ml.add_execution(_rid(12), workflow=_rid(900))
    ml.add_dataset(_rid(3), producer=_rid(12))
    ml.add_dataset(_rid(4), producer=_rid(12))  # same producer via two inputs
    ml.add_execution(_rid(10), workflow=_rid(900), input_datasets=[(_rid(3), "1.0.0"), (_rid(4), "1.0.0")])
    ml.add_dataset(_rid(1), producer=_rid(10))
    return ml, _rid(1), {}


def _scenario_cycle() -> tuple[_FakeML, str, dict]:
    ml = _FakeML()
    ml.add_workflow(_rid(900))
    ml.add_execution(_rid(10), workflow=_rid(900), input_datasets=[(_rid(2), None)])
    ml.add_execution(_rid(11), workflow=_rid(900), input_datasets=[(_rid(1), None)])
    ml.add_dataset(_rid(1), producer=_rid(10))
    ml.add_dataset(_rid(2), producer=_rid(11))  # 10 -> 11 -> 10
    return ml, _rid(1), {}


def _scenario_depth_cap() -> tuple[_FakeML, str, dict]:
    ml, root, _ = _scenario_two_level()
    return ml, root, {"depth": 0}


def _scenario_member_fallback() -> tuple[_FakeML, str, dict]:
    ml = _FakeML()
    ml.add_workflow(_rid(900))
    ml.add_execution(_rid(20), workflow=_rid(900))
    ml.add_dataset(_rid(1), description="no origin")  # no producer recorded
    ml.set_member_producers(_rid(1), {_rid(20)})
    return ml, _rid(1), {}


def _scenario_cap_truncation() -> tuple[_FakeML, str, dict]:
    ml = _FakeML()
    ml.add_workflow(_rid(900))
    prev_ds = None
    for i in range(6):  # chain of 6 executions
        inputs = [(prev_ds, "1.0.0")] if prev_ds else []
        ml.add_execution(_rid(30 + i), workflow=_rid(900), input_datasets=inputs)
        ds = _rid(60 + i)
        ml.add_dataset(ds, producer=_rid(30 + i))
        prev_ds = ds
    return ml, prev_ds, {"max_executions": 3}


SCENARIOS = {
    "simple_chain": _scenario_simple_chain,
    "two_level": _scenario_two_level,
    "diamond": _scenario_diamond,
    "cycle": _scenario_cycle,
    "depth_cap": _scenario_depth_cap,
    "member_fallback": _scenario_member_fallback,
    "cap_truncation": _scenario_cap_truncation,
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_lookup_lineage_matches_golden(name: str):
    ml, root, kwargs = SCENARIOS[name]()
    dump = ml.lookup_lineage(root, **kwargs).model_dump(mode="json")
    golden_path = GOLDEN_DIR / f"{name}.json"
    if os.environ.get("UPDATE_LINEAGE_GOLDENS"):
        GOLDEN_DIR.mkdir(exist_ok=True)
        golden_path.write_text(json.dumps(dump, indent=2, sort_keys=True) + "\n")
    assert golden_path.exists(), f"golden missing — regenerate: UPDATE_LINEAGE_GOLDENS=1 (pre-refactor only)"
    assert dump == json.loads(golden_path.read_text()), f"lookup_lineage output diverged from pre-refactor golden '{name}'"
```

Adjust `add_execution` / `add_dataset` keyword names to the actual `_FakeML` signatures (read them in `tests/execution/test_lookup_lineage_unit.py:117-310` first); the scenario SHAPES above are the requirement, the builder call syntax follows the harness.

- [ ] **Step 2: Generate the goldens from the current implementation and verify tests pass**

Run: `UPDATE_LINEAGE_GOLDENS=1 DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_lineage_goldens.py -q -p no:randomly` then again WITHOUT the env var.
Expected: 7 passed both times; 7 JSON files exist in `tests/execution/goldens/`.

- [ ] **Step 3: Sanity-check a golden is meaningful**

Read `tests/execution/goldens/two_level.json` — it must contain two nested execution nodes and the consumed dataset with its version pin. If any golden is an empty walk, fix the scenario builder (the harness call was wrong), regenerate, and re-verify.

- [ ] **Step 4: Commit**

```bash
git checkout -b feat/383-lookup-provenance
git add tests/execution/test_lineage_goldens.py tests/execution/goldens/
git commit -m "test(lineage): golden byte-equivalence captures ahead of engine extraction (#383)"
```

---

### Task 2: `provenance.py` — enums and result models

**Files:**
- Create: `src/deriva_ml/execution/provenance.py`
- Test: `tests/execution/test_provenance_models.py`

**Interfaces:**
- Consumes: `ExecutionSummary`, `DatasetSummary`, `AssetSummary`, `VersionAttribution`, `RootDescriptor` from `deriva_ml.execution.lineage`; `FeatureProducerRecord` from `deriva_ml.feature`.
- Produces (used by all later tasks): `RootType`, `ArcKind`, `ArcInputType`, `AncestryState`, `GapKind` (StrEnums); `ParentLink`, `ProvenanceArc` (with `identity()` method), `ProvenanceExecution`, `DatasetVersionFacts`, `ProvenanceDataset`, `ProvenanceAsset`, `ProvenanceGap`, `ProvenanceClosure` — field names and types exactly as in spec §4.

**Import-cycle rule (locked here):** `provenance.py` imports from `lineage.py`, so `RootType` is DEFINED in `lineage.py` (Task 3) and RE-EXPORTED from `provenance.py`. All other enums live in `provenance.py`. Until Task 3 lands, `provenance.py` defines its own `RootType` placeholder — Task 3 moves it.

- [ ] **Step 1: Write failing model tests**

```python
"""Unit tests for the provenance closure models (issue #383)."""

from __future__ import annotations

import json

import pytest


def _rid(n: int) -> str:
    return f"1-{n:04X}"


def test_enums_are_strenums_with_string_equality():
    from deriva_ml.execution.provenance import AncestryState, ArcInputType, ArcKind, GapKind

    assert ArcKind.member_binding == "member_binding"
    assert GapKind.snapshot_chain_break == "snapshot_chain_break"
    assert ArcInputType.dataset == "dataset"
    assert AncestryState.chain_break == "chain_break"
    assert len(GapKind) == 12


def test_arc_identity_excludes_depth():
    from deriva_ml.execution.provenance import ArcInputType, ArcKind, ProvenanceArc

    a = ProvenanceArc(kind=ArcKind.consumption, consumed_by=_rid(1), input_rid=_rid(2),
                      input_type=ArcInputType.dataset, input_version="1.0.0", depth=1)
    b = ProvenanceArc(kind=ArcKind.consumption, consumed_by=_rid(1), input_rid=_rid(2),
                      input_type=ArcInputType.dataset, input_version="1.0.0", depth=5)
    assert a.identity() == b.identity()
    c = a.model_copy(update={"input_version": "2.0.0"})
    assert c.identity() != a.identity()


def test_closure_dumps_to_plain_json():
    from deriva_ml.execution.lineage import RootDescriptor
    from deriva_ml.execution.provenance import ProvenanceClosure

    closure = ProvenanceClosure(
        root=RootDescriptor(rid=_rid(1), type="Dataset", description=None),
        executions={}, datasets={}, assets={}, gaps=[],
        executions_visited=0, datasets_visited=0,
        traversal_complete=True, cap_hit=False,
    )
    dumped = json.dumps(closure.model_dump(mode="json"))  # must not raise
    assert '"traversal_complete": true' in dumped


def test_is_source_requires_resolved_ancestry():
    from deriva_ml.execution.provenance import AncestryState, DatasetVersionFacts

    with pytest.raises(Exception):
        DatasetVersionFacts(version="1.0.0", parents=[], ancestry_state=AncestryState.chain_break,
                            is_source=True, origin_recorded=None, version_authors=[])
    ok = DatasetVersionFacts(version="1.0.0", parents=[], ancestry_state=AncestryState.resolved,
                             is_source=True, origin_recorded=True, version_authors=[])
    assert ok.is_source is True
```

- [ ] **Step 2: Run to verify failure**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_provenance_models.py -q -p no:randomly`
Expected: FAIL — `ModuleNotFoundError: deriva_ml.execution.provenance`.

- [ ] **Step 3: Implement `provenance.py`**

The five enums and eight models exactly as spec §4, with these implementation specifics:

```python
class ProvenanceArc(BaseModel):
    kind: ArcKind
    consumed_by: str | None = None
    input_rid: str | None = None
    input_type: ArcInputType | None = None
    input_version: str | None = None
    evidence: list[FeatureProducerRecord] = Field(default_factory=list)
    depth: int

    def identity(self) -> tuple:
        """Dedup identity — every field except depth and evidence."""
        return (self.kind, self.consumed_by, self.input_rid, self.input_type, self.input_version)


class DatasetVersionFacts(BaseModel):
    version: str
    parents: list[ParentLink] = Field(default_factory=list)
    ancestry_state: AncestryState = AncestryState.not_walked
    is_source: bool | None = None
    origin_recorded: bool | None = None
    version_authors: list[VersionAttribution] = Field(default_factory=list)

    @model_validator(mode="after")
    def _source_needs_resolved_ancestry(self) -> "DatasetVersionFacts":
        if self.is_source is not None and self.ancestry_state != AncestryState.resolved:
            raise ValueError("is_source is only known when ancestry_state == 'resolved'")
        return self
```

Every class gets a Google docstring with a pure-Python runnable `Example:` block (construct from literals + generated RIDs). `ProvenanceClosure` fields per spec: `root: RootDescriptor`, `executions: dict[str, ProvenanceExecution]`, `datasets: dict[str, ProvenanceDataset]`, `assets: dict[str, ProvenanceAsset]`, `gaps: list[ProvenanceGap]`, `executions_visited: int`, `datasets_visited: int`, `traversal_complete: bool`, `cap_hit: bool`.

- [ ] **Step 4: Run tests to verify pass; run doctest collection**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_provenance_models.py src/deriva_ml/execution/provenance.py -q -p no:randomly`
Expected: PASS (models + doctests).

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/provenance.py tests/execution/test_provenance_models.py
git commit -m "feat(provenance): closure result models and StrEnum vocabularies (#383)"
```

---

### Task 3: `RootDescriptor.type` → `RootType`

**Files:**
- Modify: `src/deriva_ml/execution/lineage.py:215` (the `Literal` field) and its module imports
- Modify: `src/deriva_ml/execution/provenance.py` (drop placeholder, re-export)
- Test: extend `tests/execution/test_lineage_models.py`

**Interfaces:**
- Produces: `RootType` defined in `lineage.py`, re-exported by `provenance.py` (`from deriva_ml.execution.lineage import RootType`). Members: `dataset="Dataset"`, `asset="Asset"`, `feature="Feature"`, `execution="Execution"`.

- [ ] **Step 1: Write failing compat test** (append to `tests/execution/test_lineage_models.py`)

```python
def test_root_type_enum_is_string_compatible():
    from deriva_ml.execution.lineage import RootDescriptor, RootType

    r = RootDescriptor(rid="1-0001", type="Dataset", description=None)  # str input still validates
    assert r.type == "Dataset"                 # string comparison unchanged
    assert r.type == RootType.dataset          # enum comparison works
    assert r.model_dump(mode="json")["type"] == "Dataset"  # serialization unchanged
    from deriva_ml.execution.provenance import RootType as ReExported
    assert ReExported is RootType
```

- [ ] **Step 2: Run to verify it fails** (`ImportError: RootType`), implement (define `RootType(StrEnum)` in `lineage.py`, change the field annotation to `type: RootType`, re-export in `provenance.py`), re-run to green.

- [ ] **Step 3: Run the goldens + full lineage suites — must pass unmodified**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_lineage_goldens.py tests/execution/test_lookup_lineage_unit.py tests/execution/test_lineage_models.py tests/execution/test_forward_lineage.py tests/execution/test_lineage_html.py -q -p no:randomly`
Expected: all pass, zero golden diffs.

- [ ] **Step 4: Commit**

```bash
git add src/deriva_ml/execution/lineage.py src/deriva_ml/execution/provenance.py tests/execution/test_lineage_models.py
git commit -m "feat(lineage): RootType StrEnum replaces the type Literal — runtime-compatible (#383)"
```

---

### Task 4: `WorkflowSummary.checksum`

**Files:**
- Modify: `src/deriva_ml/execution/lineage.py` (`WorkflowSummary`, add `checksum: str | None = None`)
- Modify: `src/deriva_ml/core/mixins/execution.py:1510` (`_execution_summaries` — include `Checksum` in the Workflow row read and summary construction)
- Test: extend `tests/execution/test_lineage_models.py` + the `_FakeML` harness's `_execution_summaries` override

- [ ] **Step 1: Failing test** — construct `WorkflowSummary(rid=..., name=..., checksum="sha256:abc")`, assert the field round-trips through `model_dump`; and a seam test asserting `_execution_summaries` populates `checksum` from the fetched Workflow row (extend the existing `_execution_summaries` unit coverage in `tests/execution/test_lookup_lineage_unit.py`'s style — scripted Workflow rows gain a `Checksum` key).
- [ ] **Step 2: Run → fail (unexpected field / None), implement, run → pass.**
- [ ] **Step 3: Goldens must still pass.** New Optional field defaults to `None`; goldens contain no `checksum` key only if captured before this task — **regenerate goldens once here** (`UPDATE_LINEAGE_GOLDENS=1`), commit the diff (this is the sole permitted regeneration after Task 1, and its diff must consist ONLY of `"checksum": null` insertions — review it line by line).
- [ ] **Step 4: Commit** — `feat(lineage): WorkflowSummary.checksum — the identity workflows dedupe by (#383)`

---

### Task 5: `_producers_of_asset` — all Output rows

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py:1640` (`_producer_of_asset`)
- Test: `tests/execution/test_asset_producers.py` (new)

**Interfaces:**
- Produces: `_producers_of_asset(self, asset_rid: RID, asset_table: Any) -> list[RID]` — ALL executions with an Output association row for the asset, deterministic order (sorted). `_producer_of_asset` becomes `(next(iter(...), None))`-style wrapper returning the FIRST of the sorted list (lineage display behavior preserved, now deterministic instead of arbitrary).

- [ ] **Step 1: Failing tests** — mock the association path (same style as existing `_producer_of_asset` coverage; read its current tests first via `grep -rn "_producer_of_asset" tests/`): one asset with two Output rows → `_producers_of_asset` returns both sorted, `_producer_of_asset` returns the first; zero rows → `[]` / `None`.
- [ ] **Step 2: Run → fail, implement, run → pass. Goldens must pass (wrapper preserves single-producer display; sorted-first may differ from arbitrary-first only in multi-producer data, which no golden scenario has — verify).**
- [ ] **Step 3: Commit** — `feat(provenance): _producers_of_asset returns every recorded producer (#383)`

---

### Task 6: Diagnostic-returning binding primitive + `FeatureProducerRecord` doc fix

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py:1667` (`find_feature_producers` — split into `_find_feature_producers_impl` + thin public wrapper)
- Modify: `src/deriva_ml/feature.py` (docstring: "provenance candidates" → binding-fact language per ruling 3)
- Test: extend `tests/execution/test_find_feature_producers.py`

**Interfaces:**
- Produces: `_find_feature_producers_impl(self, dataset_rid, version=None) -> tuple[list[FeatureProducerRecord], list[BindingDiagnostic]]` where

```python
@dataclass(frozen=True)
class BindingDiagnostic:
    """One place the binding scan degraded instead of answering."""
    kind: str          # "ambiguous_hop" | "snapshot_absent" | "query_failed" | "discovery_failed"
    subject: str       # feature or table name
    detail: str
```

(module-level in `execution.py`, internal). Every `except` branch that currently `continue`s/degrades appends a diagnostic instead of discarding silently. The public `find_feature_producers` returns `records` only — behavior byte-identical.

- [ ] **Step 1: Failing tests** — reuse the existing `_Harness`/`_Source` seams: an ambiguous-hop scenario asserts `_find_feature_producers_impl` returns the surviving records AND a diagnostic with `kind="ambiguous_hop"`; a missing-table scenario yields `kind="snapshot_absent"`; a failing per-feature query yields `kind="query_failed"`; the happy path yields `(records, [])`. Assert the public wrapper's output is unchanged (equal to `impl(...)[0]`).
- [ ] **Step 2: Run → fail, implement the split, run → pass. Run the full `test_find_feature_producers.py` — the existing 12 tests must pass unmodified.**
- [ ] **Step 3: Reword the `FeatureProducerRecord` docstring** (facts-by-construction, cite the binding definition in `docs/reference/provenance-contract.md`), run doctest collection on `feature.py`.
- [ ] **Step 4: Commit** — `feat(provenance): binding scan diagnostics; FeatureProducerRecord doc says facts, not candidates (#383)`

---

### Task 7: Strict snapshot resolver + strict parents

**Files:**
- Modify: `src/deriva_ml/dataset/dataset.py` (new method near `_version_snapshot_catalog`, line 3011)
- Test: `tests/execution/test_strict_snapshot.py` (new)

**Interfaces:**
- Produces on `Dataset`:

```python
class SnapshotUnavailable(DerivaMLException):
    """A version row exists but records no catalog snapshot (ruling 6: never fall back to live)."""

def strict_version_snapshot_catalog(self, version: DatasetVersion | str) -> "DerivaMLCatalog":
    """Snapshot-bound catalog for `version`, or raise.

    Unlike _version_snapshot_catalog (which silently falls back to the
    LIVE catalog when the version row's Snapshot is empty), this raises
    SnapshotUnavailable — callers implementing snapshot-closed semantics
    (lookup_provenance ancestry, ruling 6) must convert the exception to
    a snapshot_chain_break gap, never read live state.
    """

def strict_parents_at(self, version: DatasetVersion | str) -> list[dict]:
    """Dataset_Dataset parent rows read AT this version's snapshot.

    Returns rows with keys 'parent_rid' and 'parent_version_then'
    (the parent's then-current version label, or None if the parent had
    no version rows at that snaptime). Raises SnapshotUnavailable.
    """
```

`strict_parents_at` reuses the read that `list_dataset_parents` (dataset.py:2363) performs, but through `strict_version_snapshot_catalog`; `parent_version_then` comes from the parent's version rows AT the same snapshot (max by the RCT-primary sort — reuse `_version_row_sort_key` semantics via a snapshot-bound `_dataset_version_rows`-equivalent read).

- [ ] **Step 1: Failing tests** — mock `dataset_history()` rows: version with `snapshot=None` → `strict_version_snapshot_catalog` raises `SnapshotUnavailable` (assert `_version_snapshot_catalog` on the same stub returns the live instance — pinning the behavioral difference); version with a snapshot → returns the snapshot-bound instance (assert `catalog_snapshot` called with `"<id>@<snap>"`); `strict_parents_at` happy path returns `parent_rid`/`parent_version_then`.
- [ ] **Step 2: Run → fail, implement, run → pass.**
- [ ] **Step 3: Commit** — `feat(dataset): strict snapshot resolver — no live fallback for provenance (#383)`

---

### Task 8: Engine extraction + TreeBuilder (the goldens gate)

**Files:**
- Create: `src/deriva_ml/core/mixins/_provenance_engine.py`
- Modify: `src/deriva_ml/core/mixins/execution.py` (`lookup_lineage` body + `_walk_node` moves)
- Test: goldens (Task 1) + entire existing lineage suite, unmodified

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class InputRef:
    """One concrete consumed input observed on an execution."""
    kind: ArcInputType
    rid: str
    version: str | None = None       # datasets: pinned version (None = unpinned)
    summary: Any = None              # DatasetSummary | AssetSummary (display object)


class WalkVisitor(Protocol[N]):
    def make_node(self, rid: str, *, summary: "ExecutionSummary", inputs: list[InputRef], depth: int) -> N: ...
    def make_cycle_node(self, rid: str, *, depth: int) -> N: ...
    def make_duplicate_node(self, rid: str, *, depth: int) -> N: ...
    def attach_parents(self, node: N, parents: list[N]) -> None: ...
    # Closure hooks — TreeBuilder implements them as no-ops:
    def on_consumption(self, *, consumer_rid: str, producer_rid: str | None, input_ref: InputRef, depth: int) -> None: ...
    def on_gap(self, kind: "GapKind", subject_rid: str, detail: str) -> None: ...


class WalkEngine(Generic[N]):
    def __init__(self, ml: Any, visitor: WalkVisitor[N], *, arcs: frozenset[ArcKind],
                 max_executions: int) -> None: ...
    flags: dict[str, bool]           # cycle_detected / depth_capped / walked_complete
    executions_visited: int
    def expand_execution(self, rid: str, *, depth_remaining: int | None, depth: int = 0) -> N | None: ...
```

**Extraction is a mechanical lift, not a rewrite.** Transformation table for `_walk_node` (execution.py:2030):

| current `_walk_node` code | engine code |
|---|---|
| `self._input_dataset_pairs(...)`, `self._producer_of_dataset(...)`, `self._producer_of_asset(...)`, `self._execution_summaries(...)`, `self.lookup_execution(...)`, `self._sentinel_execution_rid_or_none()` | same calls on `self._ml` (the mixin instance) — the `_FakeML` seam overrides keep working untouched |
| `LineageNode(execution=..., consumed_*=..., ...)` construction | `self._visitor.make_node(rid, summary=..., inputs=[InputRef(...)...], depth=depth)` |
| cycle-marker / duplicate-marker `LineageNode(..., already_shown=True)` | `make_cycle_node` / `make_duplicate_node` |
| recursion + `parents=` assignment | recurse `expand_execution(parent, depth_remaining-1, depth+1)`, then `attach_parents(node, parent_nodes)` |
| `visited_global` / `in_progress` / `flags` parameters | engine instance state |

`TreeBuilder` (also in `_provenance_engine.py`) implements the visitor to construct exactly today's `LineageNode`s — it receives `summary` and `inputs` and builds `LineageNode(execution=summary, consumed_datasets=[i.summary for i in inputs if i.kind==dataset], consumed_assets=[...], parents=...)`. `lookup_lineage`'s pre-walk logic (root classification, origin resolution, seed candidates, member fallback, sentinel filtering — execution.py:1198-1374) STAYS in the mixin unchanged; only the node-expansion recursion moves.

- [ ] **Step 1: Move the code per the table above.** No new tests first — the goldens ARE the failing-test discipline for this task: any behavioral drift fails them.
- [ ] **Step 2: Run the gate**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/ -q -p no:randomly -k "not live"`
Expected: identical pass/fail profile to pre-task baseline (record the baseline count first); `test_lineage_goldens.py` all green with ZERO golden edits in `git status`.

- [ ] **Step 3: Delete `_walk_node` from the mixin** (no-backwards-compat rule: nothing may still reference it — `grep -rn "_walk_node" src tests` must return only the engine).
- [ ] **Step 4: Commit** — `refactor(lineage): arc-gated WalkEngine + TreeBuilder — goldens prove byte-identical (#383)`

---

### Task 9: `lookup_provenance` — consumption closure skeleton

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py` (new public method)
- Modify: `src/deriva_ml/core/mixins/_provenance_engine.py` (`ClosureBuilder`)
- Test: `tests/execution/test_lookup_provenance_unit.py` (new; reuses `_FakeML`)

**Interfaces:**
- Produces: `lookup_provenance(self, rid: RID, *, version: str | None = None, max_executions: int = 500) -> ProvenanceClosure` with: root typology via `_classify_rid`; root-version resolution rule (spec §3 — resolved pin recorded in `root.version`, `version` on non-Dataset roots raises `DerivaMLValidationError`); `@validate_call` with `Field(ge=1)` on `max_executions`; consumption arcs with concrete inputs; assets map with ALL producers (`_producers_of_asset`) + `no_asset_producer` / `multiple_asset_producers` gaps; `unpinned_input` gap + `AncestryState.not_walked` quarantine; sentinel → `sentinel_origin` gap, never expanded; expansion loop: every discovered execution expands (pending queue drained under the cap).
- `ClosureBuilder` implements `WalkVisitor[str]` (node handle = execution RID), accumulating `executions` / `datasets` / `assets` / `gaps` dicts with arc-identity dedup (`ProvenanceArc.identity()`, minimum depth wins, evidence merged).

- [ ] **Step 1: Failing tests** (representative set — write all of these):

```python
def test_execution_root_two_level_closure_reaches_all_executions():
    # two_level scenario from goldens: closure of the ROOT DATASET must contain
    # BOTH executions with consumption arcs carrying concrete input RIDs+versions.

def test_arc_dedup_min_depth_on_diamond():
    # diamond scenario: producer discovered twice at equal identity -> ONE arc, depth == min.

def test_unpinned_input_quarantined_with_gap():
    # input dataset with version=None -> GapKind.unpinned_input, dataset present,
    # its facts entry has ancestry_state == AncestryState.not_walked.

def test_asset_with_two_producers_reports_all_plus_gap():
    # both producers in closure + GapKind.multiple_asset_producers.

def test_sentinel_never_expanded_gap_emitted():
    # origin == sentinel rid -> GapKind.sentinel_origin, sentinel not in closure.executions.

def test_cap_sets_traversal_complete_false():
    # cap_truncation scenario with max_executions=3 -> cap_hit=True, traversal_complete=False.

def test_version_on_execution_root_raises():
    with pytest.raises(DerivaMLValidationError): ml.lookup_provenance(exec_rid, version="1.0.0")

def test_max_executions_boundary_validation():
    for bad in (0, -1, True): pytest.raises(Exception, ml.lookup_provenance, rid, max_executions=bad)
```

- [ ] **Step 2: Run → fail (`AttributeError: lookup_provenance`), implement, run → pass. Goldens still green.**
- [ ] **Step 3: Commit** — `feat(provenance): lookup_provenance — consumption closure, assets, gaps, budget (#383)`

---

### Task 10: Version-authorship arc (bounded ≤ walked version)

**Files:**
- Modify: `_provenance_engine.py` (`ClosureBuilder.expand_dataset_authorship`)
- Test: extend `tests/execution/test_lookup_provenance_unit.py`

**Semantics (spec §6.3):** for each walked `(dataset, version)`: rows = `ml._dataset_version_rows(dataset_rid)` sorted by `_version_row_sort_key`; truncate AFTER the row whose `Version == version` (row absent → `GapKind.version_unresolvable`, no authorship arcs); each remaining row's `Execution` gets an `ArcKind.version_authorship` arc (`input_rid=dataset_rid`, `input_version=<that row's version>`) and is enqueued for expansion; rows with no `Execution` → `GapKind.no_version_author`; the sentinel author → `sentinel_origin` gap (not enqueued). Results land in `DatasetVersionFacts.version_authors` as `VersionAttribution`s. Memoized per `(dataset_rid, version)`.

- [ ] **Step 1: Failing tests** — dataset with versions 0.1.0 (author A), 0.2.0 (author B), 0.3.0 (author C), walked at 0.2.0: closure contains A and B (not C), each with a `version_authorship` arc; author-less row yields `no_version_author`; unknown walked version yields `version_unresolvable`; authors are themselves EXPANDED (give B a consumed input and assert its producer enters the closure).
- [ ] **Step 2: Run → fail, implement, run → pass. Goldens green.**
- [ ] **Step 3: Commit** — `feat(provenance): version-authorship arc bounded at the walked version (#383)`

---

### Task 11: Member-binding arc

**Files:**
- Modify: `_provenance_engine.py` (`ClosureBuilder.expand_dataset_bindings`)
- Test: extend `tests/execution/test_lookup_provenance_unit.py`

**Semantics (spec §6.4):** per walked pinned `(dataset, version)` (skip quarantined unpinned ones): `records, diagnostics = ml._find_feature_producers_impl(dataset_rid, version=version)`; each record with an execution → `ArcKind.member_binding` arc (evidence carries the record) + enqueue; `execution_rid=None` records → `GapKind.null_binding_execution`; every diagnostic → `GapKind.binding_scan_failed` (subject = diagnostic.subject, detail = kind + detail). Memoized per `(dataset_rid, version)`.

- [ ] **Step 1: Failing tests** — binding producers enter the closure with evidence attached and are expanded; null-execution record → gap not arc; a scripted diagnostic → `binding_scan_failed` gap while surviving records still produce arcs (degrade-with-honesty); two datasets sharing a binding execution → one `ProvenanceExecution` with two `member_binding` arcs (different `input_rid` ⇒ different identities).
- [ ] **Step 2: Run → fail, implement, run → pass. Goldens green.**
- [ ] **Step 3: Commit** — `feat(provenance): member-binding arc with diagnostics as gaps (#383)`

---

### Task 12: Snapshot-strict ancestry + dataset budget

**Files:**
- Modify: `_provenance_engine.py` (`ClosureBuilder.expand_dataset_ancestry`, dataset budget wiring)
- Test: extend `tests/execution/test_lookup_provenance_unit.py`

**Semantics (spec §6.5):** per walked pinned `(dataset, version)`: obtain `Dataset` via `ml.lookup_dataset(rid)` and call `strict_parents_at(version)`; `SnapshotUnavailable` → `GapKind.snapshot_chain_break`, `ancestry_state=chain_break`, `is_source=None`, branch stops. Success: `parents` → `ParentLink(parent_rid, child_version=version, parent_version_then=...)`; each parent recursed at `parent_version_then` (None → `snapshot_chain_break` on that link); `ancestry_state=resolved`; `is_source = (parents == [])`. Cycle safety: an active-ancestry-path set of dataset RIDs — revisit on the ACTIVE path → stop branch with `GapKind.unresolved_rid`? No: cycles in `Dataset_Dataset` across snapshots get a dedicated detail under `snapshot_chain_break` with `detail="ancestry cycle"`. Budget: `datasets_visited` increments per distinct `(dataset, version)` expansion across ALL dataset work (authorship/bindings/ancestry share the memo key); exceeding `4 * max_executions` sets `cap_hit=True`, `traversal_complete=False`, stops further dataset expansion.

- [ ] **Step 1: Failing tests** — parent chain of 3 resolves to a source (`is_source=True` at the top, `ParentLink` versions chained); `Snapshot=None` mid-chain → `snapshot_chain_break` + `chain_break` state + `is_source=None`; ancestry cycle terminates with the gap; parent datasets' AUTHORS enter the closure (integration across Tasks 10/12); dataset budget: a wide ancestry fan with tiny `max_executions` trips `cap_hit` without hanging.
- [ ] **Step 2: Run → fail, implement, run → pass. Goldens green.**
- [ ] **Step 3: Commit** — `feat(provenance): snapshot-strict ancestry to source, dataset budget (#383)`

---

### Task 13: Determinism finalize, exports, docs

**Files:**
- Modify: `_provenance_engine.py` (`ClosureBuilder.finalize()`)
- Modify: `src/deriva_ml/execution/__init__.py` (mirror the lineage model exports for the provenance names — read the existing lineage export block first and match its style)
- Modify: `docs/user-guide/executions.md` (new section "Complete provenance: lookup_provenance" — what closure/arcs/gaps mean, `lookup_lineage` positioned as the focused data-flow view; NOTE: primary-entry-point framing is the deferred §9.2 question — write the section neutrally, present both, flag for Carl's docs decision in the PR description)
- Test: extend `tests/execution/test_lookup_provenance_unit.py`

**Finalize sort keys (spec §4):** `executions`/`datasets`/`assets` dict insertion order rebuilt sorted by key; `gaps` by `(kind, subject_rid, detail)`; each `arcs` list by `(kind, input_rid or "", consumed_by or "")`; `evidence` by `(feature_name, element_type, execution_rid or "")`; `parents` by `parent_rid`; `version_authors` already in version order; `producers`/`consumed_by` sorted.

- [ ] **Step 1: Failing test** — build the same scenario twice with insertion orders shuffled (construct inputs in reversed order); `model_dump(mode="json")` must be byte-identical. Plus an export test: `from deriva_ml.execution import ProvenanceClosure, ArcKind, GapKind` works.
- [ ] **Step 2: Run → fail, implement finalize + exports, run → pass.**
- [ ] **Step 3: Write the docs section; verify with `uv run mkdocs build 2>&1 | tail -3` (no new warnings).**
- [ ] **Step 4: Commit** — `feat(provenance): deterministic serialization, public exports, user-guide docs (#383)`

---

### Task 14: Full regression, live reconciliation, PR

- [ ] **Step 1: Full offline suite**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/ -q -p no:randomly -k "not live" --timeout=600` (fall back to no `--timeout` flag if pytest-timeout is absent)
Expected: same pass/fail profile as the pre-branch baseline plus the new tests (record baseline numbers before comparing; the known #295 failure and no-catalog errors are pre-existing).

- [ ] **Step 2: Lint** — `uv run ruff check src tests && uv run ruff format --check src tests` clean.

- [ ] **Step 3: Live reconciliation (manual, NOT committed).** Against `www.eye-ai.org`/`eye-ai`: resolve the reference training execution by catalog lookup (find the VGG-19 training execution used throughout #383 — e.g. via `find_executions` filtered on its workflow description; never paste its RID as a literal into committed code), run `lookup_provenance` on it, and reconcile against the deploy inventory's ground truth (45 executions / 17 workflows / 14 datasets / 5 gaps). Acceptance: every count difference is explainable by a ruling (sentinel-by-identity, binding facts with evidence, snapshot-resolved ancestry versions, richer gap taxonomy). Record the reconciliation — numbers and explanations — as a dated entry in `tacit-knowledge.md` and commit that.

- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin feat/383-lookup-provenance
gh pr create --title "feat(provenance): lookup_provenance — the complete provenance closure (#383)" --body "<summary: spec link, six rulings honored, engine extraction with golden proof, live reconciliation numbers, the deferred §9.2 docs-framing question for Carl>"
```

- [ ] **Step 5: Codex pass** — run `codex review --base main` per the repo's pre-merge habit; adjudicate findings (the fabricated-RID standing ruling applies), fix accepted ones, re-run gate suites, push.

---

## Self-Review (performed at plan-writing time)

- **Spec coverage:** §3 API → Task 9; §4 models/enums/idioms → Tasks 2–3; §5 engine/budgets → Tasks 8, 9, 12; §6.1 → 9; §6.2 → 5+9; §6.3 → 10; §6.4 → 6+11; §6.5 → 7+12; §6.6 → 4; §6.7 → 3; §8.1 goldens → 1 (regeneration exception documented in 4); §8.2 → per-task tests; §8.3 → 14. Non-goals (§7) have no tasks — correct.
- **Known judgment calls encoded:** goldens regenerate exactly once (Task 4, additive-null field) with a reviewed diff; `RootType` lives in `lineage.py` re-exported from `provenance.py` (import-cycle resolution — deviation from spec letter, satisfies spec intent, noted in Task 2); ancestry-cycle gap rides `snapshot_chain_break` with a cycle detail rather than a 13th `GapKind`.
- **Type consistency:** `ProvenanceArc.identity()` defined in Task 2, used in Tasks 9–11; `InputRef`/`WalkVisitor`/`WalkEngine` defined in Task 8, consumed in 9–12; `_find_feature_producers_impl` defined in Task 6, consumed in 11; `strict_parents_at` defined in Task 7, consumed in 12; `_producers_of_asset` defined in Task 5, consumed in 9.
