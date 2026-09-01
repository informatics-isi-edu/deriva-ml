# `lookup_provenance` Implementation Plan (issue #383) — v2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** v2 after Codex plan review (19 P1s accepted, 1 accepted-with-amendment, 4 P2s accepted; disposition in §Codex round at bottom).

**Goal:** One call — `ml.lookup_provenance(rid)` — returning the complete provenance closure of an artifact (`ProvenanceClosure`), built on a single arc-gated walk engine shared with `lookup_lineage`, whose observable contract stays byte-identical.

**Architecture:** `_provenance_engine.WalkEngine` owns ALL traversal — execution expansion, root-seed candidate iteration, per-`(dataset, version)` arc expansion (authorship / bindings / ancestry, each gated by the requested arc set and memoized), the pending-execution queue, sentinel classification, per-domain visited tracking, and both budgets. Frontends are pure accumulators over visitor events: `TreeBuilder` rebuilds today's `LineageNode` tree (golden byte captures prove equivalence), `ClosureBuilder` accumulates the closure dicts. All per-`(dataset, version)` facts are read through a STRICT snapshot resolver — no live fallback anywhere in the closure (lineage's existing live-display behaviors are preserved behind its frontend for byte-compat).

**Tech Stack:** Python ≥3.12, Pydantic v2 (`BaseModel`, `@validate_call` with `Field(strict=True, ge=1)`), `StrEnum`, pytest with the extended `_FakeML` seam-override harness.

**Spec:** `docs/superpowers/specs/2026-08-31-lookup-provenance-design.md` (approved v2.2). The six rulings in its §2 are non-negotiable constraints.

## Global Constraints

- Branch: `feat/383-lookup-provenance`; all work lands via ONE PR to `main`. Never commit to `main` directly.
- Every command: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run <cmd>` (chain `cd` in the same Bash call; never bare `pytest`).
- TDD: every behavioral change shows its test failing first.
- RIDs in tests are generated (`_rid(n)`-style) and asserted only against flow-through values — never human-written literals. **RID ordering carries no semantics** — never let a sort over RIDs pick a "first" producer or "newest" anything; sorting is permitted ONLY as an output-determinism step at finalize.
- `lookup_lineage` observable output must remain BYTE-identical: goldens (Task 1) compare canonical serialized BYTES and run in every subsequent task. Sole permitted regeneration: Task 4 (reviewed diff).
- Enums are `StrEnum` for vocabularies deriva-ml closes; catalog-sourced strings (`status`, `element_type`, `asset_table`) stay `str`.
- Public models are Pydantic; engine internals may be `@dataclass`.
- Google docstrings with `Example:` blocks; catalog-touching examples carry `# doctest: +SKIP`.
- Lint before every commit: `uv run ruff check --fix src tests && uv run ruff format src tests`.
- tk-023: no unbounded `.in_()`/disjunctions. **Chunking ownership (documented spec deviation):** closure-side summary enrichment uses the chunked `_execution_summaries`; lineage's existing per-node `lookup_execution` fetches are preserved verbatim for byte-compat. The engine owns chunking wherever summaries are batch-fetched.
- **Snapshot strictness (rulings 3/6, extended by Codex):** every per-`(dataset, version)` fact in the closure — version authors, origin flag, bindings, parents — is read through `strict_version_snapshot_catalog`. `SnapshotUnavailable` ⇒ gap + skip that expansion; NEVER a live read. Lineage's live-display fallbacks stay lineage-frontend-only.

---

### Task 1: Golden byte-equivalence captures for `lookup_lineage`

Locks the pre-refactor behavior. Goldens are canonical-bytes JSON files committed to the repo, produced by the CURRENT implementation, byte-compared after every later task.

**Files:**
- Create: `tests/execution/test_lineage_goldens.py`
- Create: `tests/execution/goldens/` (8 JSON files, generated)

**Interfaces:**
- Consumes: `_FakeML` from `tests/execution/test_lookup_lineage_unit.py` (import — do not copy). Read its builder signatures (`tests/execution/test_lookup_lineage_unit.py:117-310`) before writing scenarios; the scenario SHAPES below are the requirement, call syntax follows the harness.
- Produces: `pytest tests/execution/test_lineage_goldens.py` as the gate every later task runs; `_canonical(dump) -> bytes` helper reused by Task 14's determinism tests.

- [ ] **Step 1: Write the golden module**

Eight scenarios: `simple_chain`, `two_level`, `diamond` (one producer reached via two distinct input datasets), `cycle`, `depth_cap` (`depth=0` on two_level), `member_fallback` (dataset with no recorded producer, member producers set), `cap_truncation` (6-chain, `max_executions=3`), **`failed_lookup`** (an execution whose `lookup_execution` raises / whose input references an unresolvable RID — construct via the harness's unresolved-RID path; read how `_FakeML.resolve_rid` signals unknown RIDs and script one mid-chain).

Byte-canonical comparison — this exact helper, used for BOTH writing and comparing:

```python
def _canonical(dump: dict) -> bytes:
    return (json.dumps(dump, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_lookup_lineage_matches_golden(name: str):
    ml, root, kwargs = SCENARIOS[name]()
    got = _canonical(ml.lookup_lineage(root, **kwargs).model_dump(mode="json"))
    golden_path = GOLDEN_DIR / f"{name}.json"
    if os.environ.get("UPDATE_LINEAGE_GOLDENS"):
        GOLDEN_DIR.mkdir(exist_ok=True)
        golden_path.write_bytes(got)
    assert golden_path.exists(), "golden missing — UPDATE_LINEAGE_GOLDENS=1 (pre-refactor only)"
    assert got == golden_path.read_bytes(), f"lookup_lineage BYTES diverged from golden '{name}'"
```

(`sort_keys=True` on both sides makes the comparison canonical yet still byte-strict for values, list order, and structure — a serialization-order regression in list content fails; dict-key order is normalized identically on both sides by construction.)

- [ ] **Step 2: Generate goldens; verify both modes**

Run: `UPDATE_LINEAGE_GOLDENS=1 DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_lineage_goldens.py -q -p no:randomly` then again WITHOUT the env var. Expected: 8 passed both times; 8 files exist.

- [ ] **Step 3: Sanity-check goldens are meaningful** — `two_level.json` has two nested execution nodes + version-pinned consumed dataset; `failed_lookup.json` shows the defensive path's actual output (whatever it is today — that IS the contract); `diamond.json` shows the duplicate-marker node. Empty-walk goldens mean a wrong builder call: fix, regenerate, re-verify.

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
- Produces (used by all later tasks): `ArcKind`, `ArcInputType`, `AncestryState`, `GapKind` (StrEnums; `RootType` placeholder here, moved by Task 3); `ParentLink`, `ProvenanceArc`, `ProvenanceExecution`, `DatasetVersionFacts`, `ProvenanceDataset`, `ProvenanceAsset`, `ProvenanceGap`, `ProvenanceClosure` — fields exactly as spec §4.
- **`ProvenanceArc.identity()` returns `(self.kind, self.consumed_by, self.input_rid, self.input_version)`** — the spec's exact tuple; `input_type` and `evidence` excluded (input_type is derivable from input_rid; evidence merges on identity match).

**Import-cycle rule (locked):** `provenance.py` imports from `lineage.py`, so `RootType` is DEFINED in `lineage.py` (Task 3) and RE-EXPORTED from `provenance.py`.

- [ ] **Step 1: Failing tests** — as plan v1 plus the corrected identity test:

```python
def test_arc_identity_is_spec_tuple():
    from deriva_ml.execution.provenance import ArcInputType, ArcKind, ProvenanceArc

    a = ProvenanceArc(kind=ArcKind.consumption, consumed_by=_rid(1), input_rid=_rid(2),
                      input_type=ArcInputType.dataset, input_version="1.0.0", depth=1)
    assert a.identity() == (ArcKind.consumption, _rid(1), _rid(2), "1.0.0")
    b = a.model_copy(update={"depth": 5, "input_type": None})
    assert a.identity() == b.identity()          # depth and input_type excluded
    assert a.model_copy(update={"input_version": "2.0.0"}).identity() != a.identity()
```

plus `test_enums_are_strenums_with_string_equality` (assert `len(GapKind) == 12`), `test_closure_dumps_to_plain_json`, `test_is_source_requires_resolved_ancestry` (all as v1).

- [ ] **Step 2: Run → fail (`ModuleNotFoundError`), implement per spec §4 (models exactly as spec; `DatasetVersionFacts` validator; Google docstrings with runnable pure-Python examples), run → pass incl. doctests** (`uv run pytest tests/execution/test_provenance_models.py src/deriva_ml/execution/provenance.py -q -p no:randomly`).
- [ ] **Step 3: Commit** — `feat(provenance): closure result models and StrEnum vocabularies (#383)`

---

### Task 3: `RootDescriptor.type` → `RootType`

**Files:** modify `src/deriva_ml/execution/lineage.py:215` + imports; `provenance.py` re-export; test in `tests/execution/test_lineage_models.py`.

- [ ] **Step 1: Failing compat test** (use generated RIDs — no literals):

```python
def test_root_type_enum_is_string_compatible():
    from deriva_ml.execution.lineage import RootDescriptor, RootType

    rid = f"1-{1:04X}"
    r = RootDescriptor(rid=rid, type="Dataset", description=None)   # str input validates
    assert r.type == "Dataset" and r.type == RootType.dataset
    assert r.model_dump(mode="json")["type"] == "Dataset"
    from deriva_ml.execution.provenance import RootType as ReExported
    assert ReExported is RootType
```

- [ ] **Step 2: Run → fail (`ImportError`), implement (`RootType(StrEnum)` in lineage.py, field `type: RootType`, provenance.py re-export replaces placeholder), run → pass.**
- [ ] **Step 3: Gate:** goldens + `test_lookup_lineage_unit.py` + `test_lineage_models.py` + `test_forward_lineage.py` + `test_lineage_html.py` all pass unmodified, zero golden diffs.
- [ ] **Step 4: Commit** — `feat(lineage): RootType StrEnum replaces the type Literal — runtime-compatible (#383)`

---

### Task 4: `WorkflowSummary.checksum` — BOTH construction paths

Checksum must flow wherever `WorkflowSummary` is built: (a) `_execution_summaries` (execution.py:1510, batched Workflow-row fetch — add `Checksum` to the fetched columns), and (b) **the walk's per-node path** (execution.py:2098 area — `_walk_node` builds the summary from `record.workflow`; find every `WorkflowSummary(` construction site via `grep -rn "WorkflowSummary(" src/` and populate `checksum` in each).

**Files:** modify `lineage.py` (`checksum: str | None = None`), `execution.py` (both paths), extend `_FakeML`'s workflow stubs with a `Checksum`/checksum attribute, tests in `test_lineage_models.py` + `test_lookup_lineage_unit.py`.

- [ ] **Step 1: Failing tests** — (a) `_execution_summaries` populates checksum from the scripted Workflow row; (b) a full `lookup_lineage` walk on the harness yields nodes whose `execution.workflow.checksum` equals the scripted value (this pins the per-node path Codex flagged — without (b), lineage nodes silently emit `None`).
- [ ] **Step 2: Run → fail, implement both paths, run → pass.**
- [ ] **Step 3: Regenerate goldens ONCE** (`UPDATE_LINEAGE_GOLDENS=1`), review the diff: ONLY `"checksum": ...` insertions (null or scripted values) — anything else is a regression; revert and fix. Commit the diff.
- [ ] **Step 4: Commit** — `feat(lineage): WorkflowSummary.checksum on both construction paths (#383)`

---

### Task 5: `_producers_of_asset` — all Output rows, fetched order

**Files:** modify `execution.py:1640`; test `tests/execution/test_asset_producers.py` (new).

**Interfaces:**
- Produces: `_producers_of_asset(self, asset_rid: RID, asset_table: Any) -> list[RID]` — ALL executions with an Output association row, **in fetched order** (deduped, first occurrence kept). `_producer_of_asset` becomes `next(iter(self._producers_of_asset(...)), None)` — **identical to today's rows[0] choice**, so lineage behavior is byte-preserved even for malformed multi-producer data. NO sorting here — RID order carries no semantics; the closure sorts its output lists only at finalize (Task 14).

- [ ] **Step 1: Failing tests** — two Output rows → both returned in fetched order and `_producer_of_asset` returns the fetched-first; zero rows → `[]`/`None`; duplicate rows deduped keeping first.
- [ ] **Step 2: Run → fail, implement, run → pass. Goldens green (behavior identical by construction).**
- [ ] **Step 3: Commit** — `feat(provenance): _producers_of_asset returns every recorded producer, fetched order (#383)`

---

### Task 6: Diagnostic-returning binding primitive + `FeatureProducerRecord` doc fix

**Files:** modify `execution.py:1667` (split `find_feature_producers` → `_find_feature_producers_impl` + thin wrapper), `feature.py` docstring; extend `tests/execution/test_find_feature_producers.py`.

**Interfaces:**
- Produces: `_find_feature_producers_impl(self, dataset_rid, version=None) -> tuple[list[FeatureProducerRecord], list[BindingDiagnostic]]`;

```python
@dataclass(frozen=True)
class BindingDiagnostic:
    kind: str          # "ambiguous_hop" | "snapshot_absent" | "query_failed" | "discovery_failed"
    subject: str
    detail: str
```

Every silent-degrade branch appends a diagnostic. Public wrapper returns `records` only — byte-identical.

- [ ] **Step 1: Failing tests** — ambiguous-hop → records + `ambiguous_hop` diag; missing table → `snapshot_absent`; per-feature query failure → `query_failed`; **discovery-level failure → `discovery_failed`** (the current function has two discovery failure branches — read `execution.py:1667-1798` and script one, e.g. `find_features` raising); happy path → `(records, [])`; wrapper output `== impl(...)[0]`.
- [ ] **Step 2: Run → fail, implement split, run → pass; existing 12 binding tests unmodified.**
- [ ] **Step 3: Reword `FeatureProducerRecord` docstring** (binding facts per ruling 3, cite `docs/reference/provenance-contract.md`); doctest collection green.
- [ ] **Step 4: Commit** — `feat(provenance): binding scan diagnostics; FeatureProducerRecord doc says facts (#383)`

---

### Task 7: Strict snapshot resolver + strict parents

**Files:** modify `src/deriva_ml/dataset/dataset.py` (near line 3011); test `tests/execution/test_strict_snapshot.py` (new).

**Current-behavior facts (verified — do not re-derive wrong):** with a NULL snapshot, `_version_snapshot_catalog_id()` returns the BARE catalog id (dataset.py:3064) and `_version_snapshot_catalog()` still calls `catalog_snapshot(<bare id>)` — the live fallback is via a bare-id snapshot handle, not by returning `self._ml_instance` on that path (the direct return happens only for falsy versions, dataset.py:3033).

**Interfaces:**
- Produces on `Dataset`: `SnapshotUnavailable(DerivaMLException)`; `strict_version_snapshot_catalog(self, version) -> DerivaMLCatalog` raising `SnapshotUnavailable` when (a) the version row's snapshot is empty/None, OR (b) `catalog_snapshot(...)` raises (unreadable snapshot — wrap and re-raise as `SnapshotUnavailable`); `strict_parents_at(self, version) -> list[dict]` with keys `parent_rid`, `parent_version_then`.
- **`parent_version_then` resolution rule:** mirror `list_dataset_parents`' existing version semantics (dataset.py:2363-2420) — the snapshot row's version FK/pointer first, with its documented fallback — NOT "max version-history row". Read `list_dataset_parents` before implementing and reuse its resolution helper (extract one if needed) so the two cannot drift.

- [ ] **Step 1: Failing tests** — NULL snapshot → `SnapshotUnavailable` (and a companion test pinning that `_version_snapshot_catalog` on the same stub calls `catalog_snapshot` with the BARE id — the behavioral difference, correctly stated); `catalog_snapshot` raising → `SnapshotUnavailable`; happy path → snapshot-bound instance with `"<id>@<snap>"`; `strict_parents_at` returns `parent_rid`/`parent_version_then` per the resolution rule (script a parent whose max-history row differs from its version-FK row and assert the FK row wins).
- [ ] **Step 2: Run → fail, implement, run → pass.**
- [ ] **Step 3: Commit** — `feat(dataset): strict snapshot resolver — no live fallback for provenance (#383)`

---

### Task 8: `_FakeML` harness extension

The closure tests (Tasks 10–13) need seams the harness lacks. Extend it HERE, with its own tests, so later tasks only consume.

**Files:** modify `tests/execution/test_lookup_lineage_unit.py` (`_FakeML` + stubs); smoke tests appended in the same file.

**Interfaces — produces on `_FakeML` (all additive; existing tests untouched):**
- Multi-row version history: `add_version_row(dataset_rid, version, execution, *, rct, snapshot)` feeding `_dataset_version_rows` AND a per-snapshot view; `set_snapshot_version_rows(dataset_rid, version, rows)` for snapshot-read authorship (Task 11).
- Snapshot seams: `lookup_dataset(rid)` returning a stub `Dataset` whose `strict_version_snapshot_catalog(version)` returns a marker or raises `SnapshotUnavailable` per scripting (`set_snapshot_available(dataset_rid, version, ok: bool)`), and whose `strict_parents_at(version)` returns scripted `[{parent_rid, parent_version_then}]` (`set_parents_at(dataset_rid, version, parents)`).
- Multi-producer assets: `add_asset(..., producers=[...])` backing `_producers_of_asset` (fetched order = list order); `_producer_of_asset` override derives first element (mirroring Task 5).
- Binding seam: `set_binding_scan(dataset_rid, version, records, diagnostics)` backing `_find_feature_producers_impl`.
- Workflow checksum on the stub workflow rows (consumed by Task 4's tests — coordinate: if Task 4 already added it, skip).
- **Call recording:** `ml.calls: list[tuple[str, tuple]]` — every seam override appends `(method_name, args)`. This powers the quarantine tests ("assert NO snapshot-dependent call happened").

- [ ] **Step 1: Write the extensions + smoke tests** (each seam scripted then read back through the override), run new + ALL existing lineage tests + goldens → green.
- [ ] **Step 2: Commit** — `test(harness): _FakeML closure seams — snapshots, multi-producers, bindings, call recording (#383)`

---

### Task 9: Engine extraction + TreeBuilder (the goldens gate)

**Files:**
- Create: `src/deriva_ml/core/mixins/_provenance_engine.py`
- Modify: `src/deriva_ml/core/mixins/execution.py` (`lookup_lineage` + `_walk_node` + seed iteration move)
- Test: goldens + entire existing lineage suite, unmodified

**Interfaces:**

```python
@dataclass(frozen=True)
class InputRef:
    kind: ArcInputType
    rid: str
    version: str | None = None       # datasets: pinned version (None = unpinned)
    summary: Any = None              # DatasetSummary | AssetSummary display object
    producer_rids: tuple[str, ...] = ()   # ALL recorded producers, fetched order (assets);
                                          # datasets: the resolved producer(s) for this pin


class WalkVisitor(Protocol[N]):
    # Tree construction:
    def make_node(self, rid: str, *, summary: "ExecutionSummary", inputs: list[InputRef], depth: int) -> N: ...
    def make_cycle_node(self, rid: str, *, depth: int) -> N: ...
    def make_duplicate_node(self, rid: str, *, depth: int) -> N: ...
    def attach_parents(self, node: N, parents: list[N]) -> None: ...
    # Seeding (spec §5 event surface — engine iterates candidates, visitor observes):
    def on_seed_candidate(self, rid: str, *, accepted: bool) -> None: ...
    # Closure hooks (TreeBuilder: no-ops):
    def on_execution(self, rid: str, *, summary: "ExecutionSummary", depth: int) -> None: ...
    def on_consumption(self, *, consumer_rid: str, input_ref: InputRef, depth: int) -> None: ...
    def on_version_author(self, *, dataset_rid: str, version: str, attribution: "VersionAttribution", depth: int) -> None: ...
    def on_binding_record(self, *, dataset_rid: str, version: str, record: "FeatureProducerRecord", depth: int) -> None: ...
    def on_parent_link(self, *, dataset_rid: str, link: "ParentLink", depth: int) -> None: ...
    def on_dataset_facts(self, *, dataset_rid: str, facts: "DatasetVersionFacts") -> None: ...
    def on_gap(self, kind: "GapKind", subject_rid: str, detail: str) -> None: ...


class WalkEngine(Generic[N]):
    def __init__(self, ml: Any, visitor: WalkVisitor[N], *, arcs: frozenset[ArcKind],
                 max_executions: int, dataset_budget: int | None = None) -> None: ...
    flags: dict[str, bool]           # cycle_detected / depth_capped / walked_complete
    executions_visited: int
    datasets_visited: int
    cap_hit: bool
    def expand_execution(self, rid: str, *, depth_remaining: int | None, depth: int = 0) -> N | None: ...
    def run_seed_candidates(self, candidates: list[str], *, depth_remaining: int | None) -> N | None:
        """Iterate seeds in order; on_seed_candidate per attempt; return first accepted tree."""
    def enqueue_execution(self, rid: str, *, depth: int) -> None: ...   # closure discoveries
    def drain(self, *, depth_remaining: int | None) -> None: ...        # expand queue under caps
    def expand_dataset(self, dataset_rid: str, version: str | None, *, depth: int) -> None:
        """Arc-gated, memoized per (rid, version): authorship / bindings / ancestry
        per the enabled arcs. version=None (unpinned) ⇒ unpinned_input gap + NO
        snapshot-dependent work (closure mode). No-op entirely when no dataset
        arcs are enabled (lineage mode) — zero cost inversion."""
```

**Extraction is a mechanical lift.** Transformation table for `_walk_node` (execution.py:2030) and the seed loop (execution.py:1288-1374):

| current code | engine code |
|---|---|
| `self._input_dataset_pairs / _producer_of_dataset / _producer_of_asset / _execution_summaries / lookup_execution / _sentinel_execution_rid_or_none / _producers_of_dataset_members` | same calls on `self._ml` — `_FakeML` overrides keep working untouched |
| per-node `lookup_execution(...)` fetch | preserved verbatim (chunk-ownership deviation, see Global Constraints) |
| asset producer: `self._producer_of_asset(...)` | `self._ml._producers_of_asset(...)` → `InputRef.producer_rids` (full tuple); the tree path uses `producer_rids[0] if producer_rids else None` — identical choice to today |
| `LineageNode(...)` construction / cycle / duplicate markers | `visitor.make_node / make_cycle_node / make_duplicate_node` |
| recursion + `parents=` | recurse + `attach_parents` |
| seed-candidate loop in `lookup_lineage` | `engine.run_seed_candidates(...)`; the CANDIDATE LIST is still computed in the mixin (origin resolution, member fallback, sentinel filtering — that logic is root classification, which stays) |
| unpinned-input live-display fallback (execution.py:2120 area) | preserved INSIDE the tree path (lineage byte-compat); the closure path never reaches it (`expand_dataset` handles closure-side unpinned semantics) |
| `visited_global` / `in_progress` / `flags` params | engine instance state |

`TreeBuilder` implements the visitor: `make_node` builds `LineageNode(execution=summary, consumed_datasets=[i.summary for i in inputs if i.kind == ArcInputType.dataset], consumed_assets=[...], ...)`; closure hooks are `pass`.

- [ ] **Step 1: Move code per the table.** The goldens are this task's failing-test discipline.
- [ ] **Step 2: Gate:** record the pre-task baseline profile of `uv run pytest tests/execution/ -q -p no:randomly -k "not live"`, run after, require identical profile; goldens green with zero file diffs (`git status` clean under `tests/execution/goldens/`).
- [ ] **Step 3: Delete `_walk_node` from the mixin;** `grep -rn "_walk_node" src tests` hits only the engine.
- [ ] **Step 4: Commit** — `refactor(lineage): arc-gated WalkEngine + TreeBuilder — goldens prove byte-identical (#383)`

---

### Task 10: `lookup_provenance` — consumption closure skeleton

**Files:** modify `execution.py` (public method), `_provenance_engine.py` (`ClosureBuilder`); create `tests/execution/test_lookup_provenance_unit.py`.

**Interfaces:**
- Produces: `lookup_provenance(self, rid: RID, *, version: str | None = None, max_executions: int = 500) -> ProvenanceClosure`.
  - `max_executions: Annotated[int, Field(strict=True, ge=1)]` under `@validate_call` — **strict**, because the repo's `VALIDATION_CONFIG` is non-strict and plain `Field(ge=1)` accepts `True` and `"2"` (verified empirically in review).
  - Root typology via `_classify_rid`; **root-version rule (spec §3)**: explicit `version` validated against `_dataset_version_rows` (absent → `DerivaMLValidationError`); omitted → latest recorded version (max by `_version_row_sort_key` — the EXISTING authorship-order helper, not a RID sort); resolved pin recorded in `root.version`; no version rows at all → `version_unresolvable` gap, non-snapshot arcs only. `version` on non-Dataset roots → `DerivaMLValidationError`.
  - **Root arcs:** the root execution (Execution root) or the root artifact's direct producer gets `ArcKind.root` (depth 0). Every other arc kind enters via later tasks' expansions.
  - **Feature-value roots (spec §3, v1):** `_classify_rid` already resolves them; closure = binding execution (root arc) + its expansion. Null binding execution → `null_binding_execution` gap + empty executions; sentinel binding execution → `sentinel_origin` gap; unresolvable → `unresolved_rid` gap.
  - Consumption arcs with concrete inputs; assets map with ALL producers (fetched order) + `no_asset_producer` / `multiple_asset_producers` gaps; executions with no workflow → `no_workflow` gap; RIDs that fail to resolve mid-walk → `unresolved_rid` gap (the tree path's defensive-None branch becomes a gap emission in closure mode); sentinel → `sentinel_origin`, never expanded; unpinned dataset input → `unpinned_input` gap + `AncestryState.not_walked` facts + zero snapshot-dependent calls; pending queue drained under the cap.
- `ClosureBuilder` implements `WalkVisitor[str]` (node handle = execution RID): accumulates dicts; arc-identity dedup (`ProvenanceArc.identity()`, min depth, evidence merged).

- [ ] **Step 1: Failing tests** (all of these; use Task 8 seams + `ml.calls` recording):

```text
test_dataset_root_two_level_closure_reaches_all_executions   # + each has a consumption arc w/ concrete input rid+version
test_root_arc_on_execution_root                              # ArcKind.root, depth 0
test_root_arc_on_dataset_root_producer
test_feature_value_root_closure                              # binding exec + its ancestry
test_feature_value_root_null_execution_gap
test_feature_value_root_sentinel_gap
test_root_version_explicit_historical                        # root.version == requested pin
test_root_version_omitted_resolves_latest                    # by _version_row_sort_key, recorded in root.version
test_root_version_unknown_raises_validation_error
test_dataset_with_no_version_rows_gap_and_degraded_walk      # version_unresolvable
test_version_kwarg_on_execution_root_raises
test_max_executions_strict_boundary                          # 0, -1, True, "2", 2.5 all rejected
test_diamond_two_distinct_consumption_arcs                   # different input_rid ⇒ TWO arcs on the shared producer
test_same_identity_rediscovery_dedups_min_depth              # same (kind, consumer, input, version) at depths 1 and 3 ⇒ one arc, depth 1
test_asset_two_producers_all_reported_plus_gap               # fetched order preserved; multiple_asset_producers gap
test_asset_zero_producers_gap                                # no_asset_producer
test_execution_without_workflow_gap                          # no_workflow
test_unresolved_rid_gap_mid_walk                             # failed_lookup-style scenario ⇒ unresolved_rid gap, walk continues
test_sentinel_never_expanded_gap_emitted
test_unpinned_input_quarantined_no_snapshot_calls            # gap + not_walked + ml.calls contains NO version-rows /
                                                             #   binding / parents / snapshot call for that dataset
test_cap_sets_traversal_complete_false
```

- [ ] **Step 2: Run → fail (`AttributeError: lookup_provenance`), implement, run → pass. Goldens green.**
- [ ] **Step 3: Commit** — `feat(provenance): lookup_provenance — consumption closure, roots, assets, gaps, strict budget (#383)`

---

### Task 11: Version-authorship arc — snapshot-read, bounded ≤ walked version

**Files:** modify `_provenance_engine.py` (`WalkEngine.expand_dataset` authorship leg); extend `tests/execution/test_lookup_provenance_unit.py`.

**Semantics (spec §6.3, snapshot-faithful per Codex):** for each walked pinned `(dataset, version)`:
1. Resolve the strict snapshot (`lookup_dataset(rid).strict_version_snapshot_catalog(version)`); `SnapshotUnavailable` → `GapKind.snapshot_chain_break` (detail `"authorship read skipped: no snapshot"`), no authorship facts, done.
2. Read the version rows AT that snapshot (harness seam: `set_snapshot_version_rows`), sort by `_version_row_sort_key`, truncate after the walked version's row (absent at snapshot → `version_unresolvable` gap).
3. Rows → `VersionAttribution`s into `DatasetVersionFacts.version_authors`; **`DatasetVersionFacts.origin_recorded` = whether the FIRST row's author is a real (non-sentinel, non-null) execution — evaluated from the snapshot rows** (this populates the field Codex flagged as orphaned); each author gets an `ArcKind.version_authorship` arc + enqueue; author-less row → `no_version_author` gap; sentinel author → `sentinel_origin` gap, not enqueued.
4. Memoized per `(dataset_rid, version)` across all dataset legs.

- [ ] **Step 1: Failing tests** — walked at 0.2.0 with snapshot rows 0.1.0/0.2.0/0.3.0 → authors of 0.1.0+0.2.0 only; snapshot rows DIFFER from live rows (script both; assert the snapshot values win — the snapshot-faithfulness pin); `origin_recorded` True/False/None cases from snapshot rows; author-less → `no_version_author`; unknown walked version at snapshot → `version_unresolvable`; `SnapshotUnavailable` → chain-break gap and NO authorship arcs; authors are themselves expanded.
- [ ] **Step 2: Run → fail, implement, run → pass. Goldens green.**
- [ ] **Step 3: Commit** — `feat(provenance): snapshot-read version-authorship arc bounded at the walked version (#383)`

---

### Task 12: Member-binding arc

**Files:** modify `_provenance_engine.py` (bindings leg); extend `tests/execution/test_lookup_provenance_unit.py`.

**Semantics (spec §6.4):** per walked pinned `(dataset, version)` with an available strict snapshot: `records, diagnostics = ml._find_feature_producers_impl(dataset_rid, version=version)` (already snapshot-scoped internally per #385); records with executions → `ArcKind.member_binding` arcs (evidence attached) + enqueue; `execution_rid=None` → `null_binding_execution` gap; each diagnostic → `binding_scan_failed` gap. Skipped (with the chain-break gap already emitted by Task 11's leg) when the strict snapshot is unavailable. Memoized.

- [ ] **Step 1: Failing tests** — binding producers enter with evidence and are expanded; null-execution → gap; scripted diagnostic → `binding_scan_failed` while surviving records still arc (degrade-with-honesty); two datasets sharing a binding execution → one `ProvenanceExecution`, two `member_binding` arcs (different `input_rid`); **evidence-merge**: same arc identity rediscovered with overlapping record lists → one arc, records deduped by equality, min depth; snapshot-unavailable dataset → NO binding scan call (`ml.calls`).
- [ ] **Step 2: Run → fail, implement, run → pass. Goldens green.**
- [ ] **Step 3: Commit** — `feat(provenance): member-binding arc with evidence merge and diagnostics as gaps (#383)`

---

### Task 13: Snapshot-strict ancestry + dataset budget

**Files:** modify `_provenance_engine.py` (ancestry leg + budget); extend `tests/execution/test_lookup_provenance_unit.py`.

**Semantics (spec §6.5):** per walked pinned `(dataset, version)`: `strict_parents_at(version)` via the stub/real Dataset; `SnapshotUnavailable` → `snapshot_chain_break`, `ancestry_state=chain_break`, `is_source=None`, stop branch. Success: rows → `ParentLink(parent_rid, child_version=version, parent_version_then)`; `parent_version_then=None` → `snapshot_chain_break` on that link (branch stops there); each resolved parent expanded via `expand_dataset(parent_rid, parent_version_then, depth+1)`; `ancestry_state=resolved`; `is_source = not parents`. Ancestry cycles: an active-ancestry-path set of dataset RIDs; revisit on the active path → `snapshot_chain_break` gap with `detail="ancestry cycle"`, branch stops. Budget: `datasets_visited` counts distinct `(dataset, version)` expansions; exceeding `dataset_budget` (default `4 * max_executions`) sets `cap_hit=True`, `traversal_complete=False`, stops dataset expansion.

- [ ] **Step 1: Failing tests** — 3-deep chain resolves to a source (`is_source=True`, `ParentLink` versions chained); snapshot-less mid-chain → chain break + `is_source=None`; `parent_version_then=None` link → chain break at that link; ancestry cycle terminates with the cycle-detail gap; parent datasets' authors enter the closure (Tasks 11+13 integration); wide ancestry fan with small `max_executions` trips the dataset budget without hanging.
- [ ] **Step 2: Run → fail, implement, run → pass. Goldens green.**
- [ ] **Step 3: Commit** — `feat(provenance): snapshot-strict ancestry to source, dataset budget (#383)`

---

### Task 14: Determinism finalize, gap-coverage checklist, exports, docs

**Files:** modify `_provenance_engine.py` (`ClosureBuilder.finalize()`), `src/deriva_ml/execution/__init__.py` (mirror the lineage export block for the provenance names), `docs/user-guide/executions.md`; extend `tests/execution/test_lookup_provenance_unit.py`.

**Finalize sort keys (spec §4):** `executions`/`datasets`/`assets` rebuilt sorted by key; **`ProvenanceDataset.versions` sorted by version label**; `gaps` by `(kind, subject_rid, detail)`; `arcs` by `(kind, input_rid or "", consumed_by or "")`; `evidence` by `(feature_name, element_type, execution_rid or "")`; `parents` by `parent_rid`; `producers` (output only) and `consumed_by` sorted.

- [ ] **Step 1: Failing tests** — (a) same scenario built with shuffled insertion orders AND a dataset walked at two versions inserted in reverse → `_canonical(model_dump)` bytes identical (reuse Task 1's `_canonical`); (b) export test `from deriva_ml.execution import ProvenanceClosure, ArcKind, GapKind, RootType`; (c) **gap-coverage checklist**: one parametrized test asserting every one of the 12 `GapKind` members is produced by at least one scenario in this file (collect `{g.kind for g in closure.gaps}` across the suite's scenarios; a member no scenario produces = a hole in the suite, fail with its name).
- [ ] **Step 2: Run → fail, implement finalize + exports, run → pass.**
- [ ] **Step 3: Docs:** new section "Complete provenance: `lookup_provenance`" in `docs/user-guide/executions.md` — closure/arcs/gaps semantics, `lookup_lineage` as the focused data-flow view, written neutrally (primary-entry-point framing = deferred spec §9.2, flagged for Carl in the PR). Verify `uv run mkdocs build 2>&1 | tail -3` adds no new warnings.
- [ ] **Step 4: Commit** — `feat(provenance): deterministic serialization, gap coverage, exports, docs (#383)`

---

### Task 15: Full regression, live reconciliation, PR

- [ ] **Step 1: Full offline suite** — `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/ -q -p no:randomly -k "not live"` (add `--timeout=600` only if pytest-timeout is installed). Compare against the pre-branch baseline profile recorded in Task 9 (the #295 failure and no-catalog errors are pre-existing).
- [ ] **Step 2: Lint** — `uv run ruff check src tests && uv run ruff format --check src tests` clean.
- [ ] **Step 3: Live reconciliation (manual, NOT committed).** Against `www.eye-ai.org`/`eye-ai`: resolve the reference VGG-19 training execution BY CATALOG LOOKUP (e.g. `find_executions` filtered on its workflow; never a RID literal in committed code), run `lookup_provenance`, reconcile against the deploy inventory ground truth (45 executions / 17 workflows / 14 datasets / 5 gaps). Acceptance: every difference explainable by a ruling (sentinel-by-identity, binding facts with evidence, snapshot-resolved ancestry versions, richer gap taxonomy). Record numbers + explanations as a dated tacit-knowledge.md entry; commit that.
- [ ] **Step 4: Push, open the PR** (`gh pr create` — body: spec link, rulings honored, golden proof, reconciliation numbers, the deferred §9.2 docs-framing question).
- [ ] **Step 5: Codex pass** — `codex review --base main`; adjudicate (fabricated-RID standing ruling applies), fix accepted findings, re-run gates, push.

---

## Codex plan-review round (2026-09-01) — disposition

20 P1 / 4 P2. **Accepted 19 P1s:** engine owns ALL arc traversal + seeding + memoization + queue (builders accumulate only); `InputRef.producer_rids` carries every producer; diamond = two arcs, dedup test rewritten for same-identity rediscovery; `identity()` = spec's exact tuple; `ArcKind.root` assignment specified; `origin_unrecorded`(→ per-snapshot facts)/`no_workflow`/`unresolved_rid`/`no_asset_producer` all tasked + tested + a 12-kind coverage checklist; feature-value-root tests; root-version tests (with `_classify_rid`'s live-derivation flagged for adaptation); authorship + `origin_recorded` read AT the strict snapshot; Task 7 current-behavior facts corrected (bare-id `catalog_snapshot`, not a live-instance return) + unreadable-snapshot case; `parent_version_then` mirrors `list_dataset_parents` resolution; `_producer_of_asset` keeps fetched-first (sorted-first would both break byte-compat and violate the repo's RID-ordering rule); checksum on BOTH construction paths; `Field(strict=True, ge=1)`; quarantine test asserts zero snapshot-dependent calls via harness call-recording; `failed_lookup` golden added; goldens compare canonical BYTES; evidence-merge test added. **Accepted-with-amendment (1):** chunked-summary ownership — closure paths chunk via `_execution_summaries`; lineage's per-node fetches preserved for byte-compat; deviation documented in Global Constraints. **Accepted 4 P2s:** `discovery_failed` test; nested `versions` ordering in finalize + test; `_rid(1)` not a literal; NEW Task 8 (harness extension with call recording).
