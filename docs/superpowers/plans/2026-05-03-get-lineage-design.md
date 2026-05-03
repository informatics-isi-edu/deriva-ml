# `lookup_lineage` design (grill-with-docs refinement)

Date: 2026-05-03
Status: Approved → ready to implement
Companion ADR: `docs/adr/0001-lineage-walks-data-flow-not-orchestration.md`

## Goal

Add a `lookup_lineage(rid: str, *, depth: int | None = None,
max_executions: int = 500) -> LineageResult` method on `DerivaML`
that returns the provenance chain for any artifact (Dataset, Asset,
Feature value, Execution).

The method replaces what is today 5–15 client round-trips through
typed read methods with one call.

## Decisions carried in from the parent plan (already approved)

These were settled before grilling and are restated here for the
reader:

1. **Single `rid` parameter; auto-detect type.** Workflow RIDs are
   not lineage-shaped; raise `DerivaMLException` with a clear message
   if one is passed.
2. **Unbounded depth by default with mandatory cycle avoidance.**
   `depth=None` (default) walks to the root. `depth=0` returns just
   the immediate producing-execution node. `depth=N` (N>0) walks N
   levels of parents. Track visited execution RIDs in a set; detect
   diamond DAGs (same parent reached via two paths) and report
   without re-expanding.
3. **Tree response with summaries only.** Each execution node carries
   summary fields (RID, name, workflow name, status, input dataset
   summaries, input asset summaries). Caller drills into any node
   with existing typed methods (`ml.lookup_execution(rid)`, etc.).
4. **Top-level transparency fields**: `executions_visited`,
   `walked_complete`, `cycle_detected`, `depth_capped`.

## Decisions made during grilling

### Q1 — Method placement: `ExecutionMixin`

Add to the existing `ExecutionMixin` in
`src/deriva_ml/core/mixins/execution.py`. A new mixin for one method
would be over-engineering (per the workspace "no over-engineering"
rule), and every primitive `lookup_lineage` needs is already in or
reachable from this mixin (`lookup_execution`, `find_executions`,
`ExecutionRecord.list_input_datasets`, `ExecutionRecord.list_assets`).

### Q2 — RID-type detection: one `resolve_rid` call

`resolve_rid` already returns a `ResolveRidResult` with a `.table`
attribute (the `Table` object). Branch on `result.table.name` and
`self.model.is_asset(result.table)`:

| `table.name`              | Discriminator                     | Type           |
|---------------------------|-----------------------------------|----------------|
| `"Execution"`             | direct                            | Execution      |
| `"Dataset"`               | direct                            | Dataset        |
| anything else             | `model.is_asset(table)` is `True` | Asset          |
| anything else             | table has both `Feature_Name` and `Execution` columns | Feature value  |
| `"Workflow"`              | direct                            | error: not lineage-shaped |
| anything else             | none of the above                 | error: not lineage-shaped |

One round-trip total (`resolve_rid`); pure local model inspection
after that.

### Q3 — Implementation strategy: iterative client-side walk

The walk shape isn't known statically (we don't know how deep until
we walk), and ERMrest path queries can't traverse arbitrary-depth
recursive associations in one query. Compose existing typed methods
(`lookup_execution`, `list_input_datasets`,
`list_assets(asset_role="Input")`, `Dataset_Version.Execution`
producer lookup) and walk iteratively. Pay N round-trips for an
N-deep chain. Acceptable for an infrequent, deliberate provenance
request.

Use `resolve_rids` (the batch form already in `RidResolutionMixin`)
to resolve the input set of a node in one query when expanding.

### Q4 — Return type: Pydantic models

Per the deriva-ml CLAUDE.md "Class idiom choice" rule: use Pydantic
when the type may cross a boundary (the MCP wrapper will serialize
this with `.model_dump()`). Models live in a new file
`src/deriva_ml/execution/lineage.py` and are imported lazily inside
the method, mirroring the existing TYPE_CHECKING/local-import
pattern in `core/mixins/execution.py`.

Type sketch:

```python
class WorkflowSummary(BaseModel):
    rid: RID
    name: str | None

class DatasetSummary(BaseModel):
    rid: RID
    name: str | None
    version: str | None  # current version at the time the lineage was walked

class AssetSummary(BaseModel):
    rid: RID
    filename: str | None
    asset_table: str

class ExecutionSummary(BaseModel):
    rid: RID
    description: str | None
    workflow: WorkflowSummary | None
    status: str

class LineageNode(BaseModel):
    execution: ExecutionSummary
    consumed_datasets: list[DatasetSummary]
    consumed_assets: list[AssetSummary]
    parents: list["LineageNode"]
    already_shown: bool = False  # diamond marker

class RootDescriptor(BaseModel):
    rid: RID
    type: Literal["Dataset", "Asset", "Feature", "Execution"]
    name: str | None
    producing_execution: ExecutionSummary | None  # None if no producer link

class LineageResult(BaseModel):
    root: RootDescriptor
    lineage: LineageNode | None  # None when producing_execution is None
    executions_visited: int
    walked_complete: bool
    cycle_detected: bool
    depth_capped: bool
```

### Q5 — No-producer case: return a valid result, don't raise

If the artifact has no producing-execution link (manually-inserted
data, etc.), set `root.producing_execution = None` and `lineage =
None`. Return normally. The user asked "how did this come to exist?"
— "it has no recorded producer" is a valid answer.

For a Dataset: walk via the **current** version's
`Dataset_Version.Execution`. A `version=` parameter to query a
historical version's lineage is a future enhancement; document the
default in the docstring.

### Q6 — Execution RID input: producer is the input itself

When the input RID is an Execution RID, the lineage root is that
execution. Walk from there normally (its consumed inputs and their
producers). This is a natural special case of the general rule.

### Q7 — Parent semantics: data-flow only, not orchestration

**Lineage parents = producing executions of consumed inputs.**
`Execution_Execution` (nested-execution links) is **not** walked as
a lineage edge.

Rationale: provenance answers "how did this come to exist?" — a
question about data flow. Nested-execution links describe
orchestration topology (which execution called which). A nested
execution can be a sibling in data-flow terms. Conflating the two
would make the result harder to read and overstate the chain.

This is significant enough to record as
ADR-0001 (`docs/adr/0001-lineage-walks-data-flow-not-orchestration.md`).

For each execution node X:
- Inputs come from `ExecutionRecord.list_input_datasets()` and
  `ExecutionRecord.list_assets(asset_role="Input")`.
- For each input dataset D: the producer is the latest
  `Dataset_Version` row's `Execution` column for D.
- For each input asset A: the producer is the execution row in
  the matching `<AssetTable>_Execution` association where
  `Asset_Role="Output"` and the asset RID matches.
- Dedupe parents by execution RID (same producer for two inputs
  shows once).

### Q8 — Cycle detection: visited set + active-recursion set

- `visited_global: set[RID]` — every execution we've ever seen.
  When expanding, if a parent is already in this set, mark the
  child node `already_shown=True` and don't recurse. Diamond DAGs
  are not cycles.
- `in_progress: set[RID]` — currently being expanded on the active
  recursion path. If a parent appears here, that's a true cycle:
  set top-level `cycle_detected=True` and abort that branch.

A well-formed DerivaML graph never produces true cycles; this is
defensive.

### Q9 — Depth semantics

| `depth=`   | Meaning                                                |
|------------|--------------------------------------------------------|
| `None`     | Default. Walk to the root.                             |
| `0`        | Return only the immediate producing-execution node.    |
| `N` (>0)   | Walk N levels of parents from the producing execution. |

If depth is reached and parents remain, set `depth_capped=True`.

### Q10 — Hard cap: `max_executions: int = 500`

Defensive bound. If exceeded, set `walked_complete=False` and stop
expanding. Generous enough that no real ML pipeline hits it; tight
enough to fail fast on a corrupted catalog.

### Q11 — Naming: `lookup_lineage` (not `get_lineage`)

The deriva-ml convention is `lookup_*` for "given a RID, return the
unique X for that RID" (`lookup_execution`, `lookup_asset`,
`lookup_dataset`). Lineage is unique per RID, so the same prefix
fits.

The MCP wrapper on the `deriva-ml-mcp` side can keep
`deriva_ml_get_lineage` as the tool name (`get_*` is more
discoverable for LLMs that don't know deriva-ml's local naming) —
that's the wrapper repo's call.

### Q12 — Test strategy: unit + integration

- **Unit** (`tests/execution/test_lookup_lineage_unit.py`):
  mock the catalog primitives. Test cycle detection, diamond
  handling, depth cap, max_executions cap, RID-type detection,
  Workflow-RID error, no-producer case.
- **Integration** (`tests/execution/test_lookup_lineage_integration.py`):
  against the live catalog. Build a 3-execution chain (load →
  train → eval), call `lookup_lineage` on the eval output asset,
  assert the full chain.

### Q13 — One ADR

Only Q7 (data-flow vs orchestration) meets all three ADR criteria
(hard to reverse, surprising without context, real trade-off).
Record it as ADR-0001.

## Implementation plan

1. Create `src/deriva_ml/execution/lineage.py` with the Pydantic
   models above.
2. Add `lookup_lineage` to `ExecutionMixin` in
   `src/deriva_ml/core/mixins/execution.py`. Helper methods stay
   private (`_classify_rid`, `_producer_of_dataset`,
   `_producer_of_asset`, `_walk_node`).
3. Wire the new models into the package's public re-exports if
   pattern dictates (check `src/deriva_ml/execution/__init__.py`).
4. Write unit tests under `tests/execution/`.
5. Write integration test (gated on live catalog like the others).
6. Add `Example:` blocks in docstrings; verify doctest collection
   passes.
7. Run linter/format/full unit suite.
8. `uv run bump-version minor` (new feature).

## Out of scope (explicitly)

- Walking `Execution_Execution` (nested-execution links). See
  ADR-0001.
- Versioned-Dataset lineage (passing `version="0.4.0"` to query a
  historical version's lineage). Current-version only for v1.
- Forward / descendant traversal ("what was produced from this?").
  This is the inverse direction; if needed, ship as
  `lookup_descendants` later.
- An MCP tool wrapper. That's the deriva-ml-mcp Round 6 follow-up.
