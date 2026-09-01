# `lookup_provenance`: the complete provenance closure (issue #383)

**Status:** approved (Carl, 2026-09-01) — v2.2 after Codex round 1
(16/17 P1s and 4/4 P2s accepted; see §10) plus idiom rulings (§4).
Open question §9.1 resolved to the stated leaning (proportional internal
dataset budget, no public knob); §9.2 (docs/CLI framing) deferred to the
documentation step of implementation.
**Depends on:** #385/#386 (shipped, v1.61.0) — `find_feature_producers` with
FK-reachable member scope and version-snapshot scoping.
**Contract basis:** `docs/reference/provenance-contract.md` (binding definition,
PR #384) and the six boundary rulings recorded on #383 and in
`tacit-knowledge.md` ("The boundary interview").

## 1. Problem

`lookup_lineage` answers *"what is this artifact's recorded data-flow
lineage?"* and stops at lineage edges. A provenance reviewer asks a bigger
question — *"everything that contributed to this artifact"* — which requires a
**transitive closure**: every discovered execution is itself expanded, the
feature arc is followed, and input-dataset ancestry is walked to its sources.
The gap is concrete: on eye-ai's reference training execution,
`lookup_lineage` reaches 2 executions; the downstream hand-built closure
(`deriva-ml-model-deploy/src/standalone/inventory.py`) reaches 45.

Every primitive that closure composes now lives in deriva-ml (lineage walk
with origin semantics, `find_feature_producers`, `list_dataset_parents`,
workflow identity, sentinel classification). The composition — structural
fact-finding over deriva-ml's own schema — belongs upstream too.

## 2. Constraints (the six rulings — none are open for re-litigation here)

1. **Capture, not mining.** The closure follows only schema-recorded facts.
   Hydra `key=RID` config mining is **not** an arc (it remains a downstream
   audit heuristic for legacy runs).
2. **Legacy is history.** Completeness is defined against contract-era
   records. Legacy holes surface as reported gaps (`origin_recorded=False`,
   sentinel terminations, null-execution bindings) — never compensated for.
3. **A feature value is the binding (member, value, execution).** A dataset
   at version vN *contains* its members' bindings at vN's snapshot, with
   FK-reachable member scope. Binding executions are provenance **by
   construction** — facts, not candidates.
4. **No arc-strength ordering.** Arc kinds form an unordered typology
   distinguished by attachment point, each with measurables (depth, counts)
   and orthogonal gap flags. Any ranking is downstream presentation.
5. **One arc-gated engine, two frontends, from day one.** `lookup_lineage`
   and `lookup_provenance` share a single traversal core; neither calls the
   other; the engine walks only the arcs its caller requests (no cost
   inversion); `lookup_lineage`'s observable contract stays byte-identical.
6. **Ancestry hops resolve at snapshots, unbounded to source.** Each
   `Dataset_Dataset` hop reads the parent as of the child version's snaptime,
   chained; snapshot-chain breaks are reported gaps. Depth runs to source
   under the global execution cap — no dedicated depth knob.

## 3. Public API

```python
closure = ml.lookup_provenance(rid, version=None, max_executions=500)
```

- `rid` — Dataset, Asset, Feature-value, or Execution root (same root
  typology as `lookup_lineage`, resolved by `_classify_rid`). Feature-value
  roots are in scope for v1: their closure is the binding's execution plus
  that execution's closure.
- `version` — Dataset roots only (rejected with `DerivaMLValidationError`
  for other root types). **Root version resolution rule:** if given, the
  closure is of `D@version`; if omitted, the root resolves to the dataset's
  **latest recorded version at call time** and the resolved pin is recorded
  in `root.version` so the output is self-describing. A root version that
  cannot be resolved (no version rows) is a `version_unresolvable` gap and
  the walk proceeds only on arcs that do not require a snapshot.
- `max_executions` — global execution budget shared with the lineage walk's
  cap semantics; validated via `@validate_call` with `Field(ge=1)` (zero,
  negative, bool, and non-int rejected at the boundary).
- Returns a **`ProvenanceClosure`** (Pydantic, `model_dump()`-serializable,
  same conventions as the lineage models).

Naming settled per the issue: `lookup_*` because it is a single-RID bundled
read (the `find_*`/`lookup_*` verb convention), sibling of `lookup_lineage`.

## 4. Result model

All models live in `execution/provenance.py` (new), importing shared
summaries (`ExecutionSummary`, `DatasetSummary`, `AssetSummary`,
`WorkflowSummary`, `VersionAttribution`) from `execution/lineage.py`.

**Idiom decisions (Carl, 2026-09-01):**

- **Public result models are Pydantic `BaseModel`** (per the CLAUDE.md class-
  idiom rule: serialized across a boundary, user-facing return types,
  composing with the all-Pydantic lineage models). Engine-internal
  bookkeeping (visited-set entries, walk events inside
  `_provenance_engine.py`) may be `@dataclass` — it has no public surface.
- **Closed vocabularies are `StrEnum`, not `Literal`** — consumers dispatch
  on them (filter arcs by kind, group gaps by kind), so the vocabulary must
  be importable and autocompletable, with typos failing at authoring time.
  `StrEnum` members compare equal to their strings and `model_dump()`
  serializes them as plain strings, so the JSON envelope is unaffected.
  Exported from `execution/provenance.py`:

  The rule has a principled stopping point: **enums are for vocabularies
  deriva-ml itself closes.** Catalog-sourced open vocabularies —
  `ExecutionSummary.status`, `element_type` / table names, `asset_table` —
  deliberately stay `str`: enum-validating them would make records from
  older or foreign catalogs fail model validation on values deriva-ml
  doesn't control.

```python
class RootType(StrEnum):        # replaces RootDescriptor.type's Literal (in scope, §6.7)
    dataset = "Dataset"
    asset = "Asset"
    feature = "Feature"
    execution = "Execution"

class ArcKind(StrEnum):
    root = "root"
    consumption = "consumption"
    version_authorship = "version_authorship"
    member_binding = "member_binding"

class ArcInputType(StrEnum):
    dataset = "dataset"
    asset = "asset"

class AncestryState(StrEnum):
    resolved = "resolved"
    chain_break = "chain_break"
    not_walked = "not_walked"

class GapKind(StrEnum):
    sentinel_origin = "sentinel_origin"
    origin_unrecorded = "origin_unrecorded"
    null_binding_execution = "null_binding_execution"
    no_workflow = "no_workflow"
    snapshot_chain_break = "snapshot_chain_break"
    unpinned_input = "unpinned_input"
    version_unresolvable = "version_unresolvable"
    no_version_author = "no_version_author"
    no_asset_producer = "no_asset_producer"
    multiple_asset_producers = "multiple_asset_producers"
    unresolved_rid = "unresolved_rid"
    binding_scan_failed = "binding_scan_failed"
```

```python
class ProvenanceArc(BaseModel):
    """One schema-recorded reason an execution is in the closure.

    Arc identity (for dedup) is the tuple of all fields EXCEPT depth:
    (kind, consumed_by, input_rid, input_version). A rediscovery with
    identical identity updates depth to the MINIMUM discovery depth and is
    not appended again; a rediscovery with a different identity is a new
    arc. `evidence` is merged (deduped by record equality) on identity match.
    """
    kind: ArcKind      # see enum above; comments per member:
                       #   root — the root artifact itself / its direct producer
                       #   consumption — produced an input some closure member consumed
                       #   version_authorship — authored a version (<= walked version) of a walked dataset
                       #   member_binding — bound feature values onto a walked dataset@version's members
    consumed_by: str | None = None    # consuming execution RID (consumption)
    input_rid: str | None = None      # the CONCRETE input: dataset or asset RID
    input_type: ArcInputType | None = None
    input_version: str | None = None  # dataset inputs: the pinned version (None = unpinned, see gaps)
    evidence: list[FeatureProducerRecord] = []   # member_binding only; sorted (feature_name, element_type, execution_rid)
    depth: int                        # MINIMUM hops from root at which this arc was found

class ProvenanceExecution(BaseModel):
    execution: ExecutionSummary       # incl. workflow identity (see §6.6)
    arcs: list[ProvenanceArc]         # unordered typology (ruling 4); sorted for determinism by (kind, input_rid or "", consumed_by or "")

class DatasetVersionFacts(BaseModel):
    """Facts observed AT one version's snapshot. Never merged across versions."""
    version: str
    parents: list[ParentLink]         # snapshot-resolved (ruling 6); sorted by parent_rid
    ancestry_state: AncestryState
    is_source: bool | None            # True/False only when ancestry_state == "resolved"; else None
    origin_recorded: bool | None      # tri-state, as observed at this snapshot
    version_authors: list[VersionAttribution]  # bounded at this version (§6.3)

class ProvenanceDataset(BaseModel):
    rid: str
    description: str | None           # live display metadata, labeled as such
    versions: dict[str, DatasetVersionFacts]   # keyed by version label — per-snapshot facts never conflated

class ParentLink(BaseModel):
    parent_rid: str
    child_version: str                # the version whose snaptime resolved the hop
    parent_version_then: str | None   # parent's then-current version (None = itself a gap)

class ProvenanceAsset(BaseModel):
    asset: AssetSummary
    producers: list[str]              # ALL producing execution RIDs (empty = gap; >1 = also a gap, see below)
    consumed_by: list[str]            # closure executions that consumed it

class ProvenanceGap(BaseModel):
    """First-class honest-gap record (ruling 2). Orthogonal to arcs."""
    kind: GapKind      # see enum above (12 kinds)
    subject_rid: str
    detail: str

class ProvenanceClosure(BaseModel):
    root: RootDescriptor              # reused from lineage.py; root.version = resolved pin (§3)
    executions: dict[str, ProvenanceExecution]   # keyed by execution RID
    datasets: dict[str, ProvenanceDataset]       # keyed by dataset RID; facts per-version inside
    assets: dict[str, ProvenanceAsset]           # keyed by asset RID
    gaps: list[ProvenanceGap]
    executions_visited: int
    datasets_visited: int
    traversal_complete: bool          # False iff any bound was hit (NOT a claim of gap-freedom;
                                      # gap-freedom is `not gaps`, deliberately separate)
    cap_hit: bool
```

Model notes:

- **Sentinel discipline:** the sentinel execution (and the unknown-provenance
  File sentinel, where present) is never a closure member and never expanded;
  encountering it emits a `sentinel_origin` gap naming what it terminated.
- **Determinism, fully specified:** `executions` / `datasets` / `assets`
  key order, `gaps`, every `arcs` list, `evidence`, `parents`,
  `version_authors`, `producers`, and `consumed_by` are all sorted at
  finalize with stated keys, so `model_dump()` output diffs cleanly.
- **"Strongest arc" does not exist** (ruling 4); arcs are a set with defined
  identity and minimum-depth semantics (see `ProvenanceArc` docstring).

## 5. Architecture: one engine, two frontends (ruling 5)

New internal module `core/mixins/_provenance_engine.py`. The engine owns
everything both walks share; frontends differ only in which arcs they enable
and how they accumulate.

```python
class _WalkEngine:
    """Arc-gated traversal over provenance arcs.

    Owns: root classification handoff, input resolution (pinned and
    unpinned), producer lookup (dataset-version authorship and asset Output
    rows — ALL of them), sentinel classification, per-domain visited
    tracking, the execution and dataset budgets, and chunked summary
    fetches (tk-023).

    The visitor receives, per event: the discovered node, the arc through
    which it was found, the CURRENT DEPTH, the ACTIVE PATH (for tree
    duplicate-marking and ancestry cycle detection), and — for root
    seeding — each candidate seed in order with accept/reject feedback
    (preserving lookup_lineage's candidate-retry and member-producer
    fallback behavior). This event surface is what the existing
    lookup_lineage walk needs to be reproduced exactly; the TreeBuilder
    is written against it first and validated against golden outputs
    (§8) before the ClosureBuilder is started.
    """
```

- **Visited/cycle bookkeeping is per-domain, not one set:** executions are
  keyed by RID; dataset expansion (bindings, authorship, ancestry) is keyed
  by `(dataset RID, version)`; ancestry cycle detection additionally checks
  the **active ancestry path** (a `Dataset_Dataset` cycle across versions
  must terminate the branch with a gap, not loop).
- **Two budgets:** `max_executions` bounds executions visited;
  `datasets_visited` is bounded by a proportional internal dataset budget
  (default `4 × max_executions`) so a large ancestry graph containing few
  executions cannot walk unboundedly. Hitting either sets
  `traversal_complete=False` and `cap_hit=True`.
- The engine's arc gate is typed: `arcs: frozenset[ArcKind]`, never bare
  strings.
- **`lookup_lineage`** = engine with `arcs={ArcKind.consumption}` feeding a
  **TreeBuilder** that reproduces today's `LineageNode` tree — including
  `already_shown` collapse, member-producer fallback seeding, candidate
  iteration, and unpinned-input display behavior. Byte-identical output is
  an acceptance criterion verified by golden captures (§8), not just by the
  existing suite passing.
- **`lookup_provenance`** = engine with all arcs feeding a **ClosureBuilder**
  accumulating the dict-shaped closure. Expansion rule: every execution
  entering through ANY arc is itself expanded. Feature and ancestry work
  runs once per `(dataset, version)` and is memoized within the call.
- Neither frontend calls the other. The engine has no public surface.

## 6. Semantics detail

### 6.1 Consumption arc
`Dataset_Execution` (input) and `{Asset}_Execution` Input rows. Every
consumption arc carries the **concrete input** (`input_rid`, `input_type`,
`input_version`). **Unpinned dataset inputs** (no `Dataset_Version`) emit an
`unpinned_input` gap, enter the closure as an execution+dataset with
`ancestry_state="not_walked"`, and are **not** expanded through any
snapshot-dependent arc (no bindings, no ancestry, no version authorship) —
never a silent live-state read.

### 6.2 Asset facts
Consumed assets are first-class closure members (`assets` map), including
producerless ones. Producer resolution follows **all** Output rows for the
asset — never first-match: zero producers ⇒ `no_asset_producer` gap; more
than one ⇒ all reported as producers **and** a `multiple_asset_producers`
gap (malformed multiplicity under the current writer contract). This
requires fixing `_producer_of_asset()`'s first-match behavior — that fix is
in scope and lands in the engine, with `lookup_lineage`'s frontend
preserving its current single-producer display choice to keep the contract
byte-identical.

### 6.3 Version-authorship arc — bounded at the walked version
For a walked `D@vN`, authors of versions **up to and including vN** in the
RCT-primary total order (#367) enter the closure; later versions' authors do
not (snapshot closure — live history must not leak future authors into a
pinned closure). A version row with no execution attribution emits
`no_version_author`. Attachment point is the version row (construction),
kept distinct from member bindings (content) per ruling 3's corollary.

### 6.4 Member-binding arc
Per walked `(dataset, version)`, the engine calls an **internal
diagnostic-returning variant** of the #385 machinery: it returns
`(records, diagnostics)` where diagnostics distinguish "no bindings" from
"failed to inspect" (planner ambiguity skips, snapshot-absent tables,
per-feature query failures). The public `find_feature_producers` keeps its
degrade-quietly contract and becomes a thin wrapper discarding diagnostics.
In the closure, failures surface as `binding_scan_failed` gaps; records are
carried verbatim as arc evidence; null-execution rows become
`null_binding_execution` gaps. **In-scope doc fix:** `FeatureProducerRecord`'s
docstring still says "provenance candidates" — stale against ruling 3;
reword to binding-fact language.

### 6.5 Ancestry — dataset discovery, not an execution arc
Ancestry hops discover **datasets**, not executions, so there is no
execution-level "ancestry" arc kind: a parent dataset's contribution enters
through its own `version_authorship` and `member_binding` arcs, and
`ParentLink` records the hop itself. Resolution per ruling 6: parents are
read at the child version's snaptime via a **strict snapshot resolver** —
a new internal `_strict_version_snapshot(dataset, version)` that raises
(rather than falling back to the live catalog) when the version row's
`Snapshot` is NULL or the snapshot is unreadable. (The existing
`_version_snapshot_catalog()` live-fallback behavior is unusable here and
must not be called by the engine.) Unresolvable hops emit
`snapshot_chain_break`, set `ancestry_state="chain_break"` /
`is_source=None`, and stop that branch. `is_source=True` requires
`ancestry_state="resolved"` with zero parents.

### 6.6 Workflow identity
`WorkflowSummary` today carries name/url/**version**, and version is
documented as possibly stale relative to what a given execution ran (#377
caveat). A provenance closure needs the identity deriva-ml actually
dedupes workflows by: **checksum**. In scope: add `checksum: str | None` to
`WorkflowSummary` (additive; populated by both frontends' summary fetch).
Executions with no workflow record emit `no_workflow` gaps. The spec's
completeness claim is correspondingly precise: the closure identifies the
workflow *record* and its checksum — not a git-state attestation beyond
what capture recorded.

### 6.7 `RootDescriptor.type` migration to `RootType`
The shipped `RootDescriptor.type` is a `Literal` with the same four values.
In scope: convert it to `RootType` (defined in `provenance.py`, imported by
`lineage.py`). The change is runtime-compatible — members carry the
identical string values, compare equal to the strings existing callers
use, and `model_dump()` output is unchanged — so `lookup_lineage`'s
byte-identical guarantee holds through it (the golden captures in §8
prove it rather than assert it).

## 7. Non-goals (stay downstream, per the issue's boundary section)

- Arc-strength ranking or any ordering of arcs (ruling 4).
- Config-asserted dependencies / Hydra mining (ruling 1).
- Bakeability judgments, BlackBox-workflow name policy, CI gating.
- Rendering: `deriva-ml-lineage --closure` HTML is a **follow-up issue** —
  drawing a closure (a DAG with typed arcs) is a different figure from the
  lineage tree and deserves its own design pass. This spec ships the model
  and API; the JSON envelope gains a `provenance` slot only when that
  follow-up lands.

## 8. Validation

1. **Golden byte-equivalence for the refactor:** before the engine
   extraction, capture golden `model_dump()` outputs of `lookup_lineage`
   for the pinned scenario matrix — simple root, diamond, cycle, failed
   lookup, member-producer fallback, depth cap, cap truncation — from the
   existing offline harnesses. After the extraction, the TreeBuilder output
   must compare equal to the goldens. The existing lineage suite passing
   unmodified is necessary but not sufficient.
2. **Offline closure tests:** seam-mocked unit tests per arc kind, per gap
   kind (all twelve), arc-identity dedup and minimum-depth, per-domain
   visited separation, ancestry cycle termination, both budgets,
   `traversal_complete` semantics, unpinned-input quarantine, strict
   snapshot resolver raising on NULL, determinism of every sorted
   collection — generated RIDs asserted only against flow-through values.
3. **Live reconciliation (manual/scripted, not a committed test):** run
   `lookup_provenance` on the eye-ai reference training execution — the
   artifact is **resolved by catalog lookup at run time** (per the RID
   rule, its identifier is never a literal in committed test code) — and
   reconcile against the deploy inventory's ground truth (45 executions /
   17 workflows / 14 datasets / 5 gaps), expecting **principled differences
   only**: sentinel by identity not name-matching, feature arcs as facts
   with binding evidence, ancestry versions snapshot-resolved, and the new
   gap kinds surfacing what the heuristic detector missed. Every difference
   must be explainable by a ruling; the reconciliation is recorded in
   tacit-knowledge.md.

## 9. Open questions for this review

1. **Dataset budget default:** is `4 × max_executions` the right proportional
   bound for `datasets_visited`, or should it be an independent parameter?
   (Leaning: proportional internal default; a second public knob is YAGNI
   until a real catalog proves otherwise.)
2. **Docs/CLI framing** (carried over, still unresolved from the interview):
   should the user guide and CLI lead with `lookup_provenance` as the
   primary entry point and present `lookup_lineage` as the focused
   data-flow view — or keep lineage primary? Pure documentation framing,
   but it decides the README examples.

## 10. Codex round 1 disposition (2026-08-31)

17 P1 / 4 P2 findings; 16 P1s and all P2s accepted and folded in above
(assets as members + concrete inputs, all-producers asset traversal, root
version rule, vN-bounded authorship, strict snapshot resolver, unpinned
quarantine, 12-kind gap taxonomy, diagnostic binding primitive,
FeatureProducerRecord doc fix, workflow checksum, ancestry de-arced,
tri-state ancestry/`is_source`, per-domain visited + dataset budget, arc
identity/min-depth, richer engine event surface, per-version dataset facts,
`traversal_complete` rename, full determinism, boundary validation, golden
captures). Partially rejected: "hard-coded RID in validation" — §8.3
describes a manual live reconciliation against a named production artifact
(documentation, like the repo's memory notes), but the legitimate kernel is
accepted: any committed/scripted form resolves the artifact by catalog
lookup, never a literal.
