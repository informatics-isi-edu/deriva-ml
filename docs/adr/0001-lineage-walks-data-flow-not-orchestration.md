# ADR-0001: Lineage walks data-flow parents, not orchestration parents

Date: 2026-05-03
Status: Accepted

## Context

DerivaML's catalog records two distinct execution-to-execution
relationships:

1. **Data-flow**: execution X consumed dataset D (or asset A), and
   D was produced by execution Z. X is "downstream" of Z in data-flow
   terms via D. The link is implicit — derived by walking
   `Dataset_Version.Execution` (for datasets) or the
   `<AssetTable>_Execution` association table with
   `Asset_Role="Output"` (for assets).
2. **Orchestration**: execution X spawned, called, or otherwise
   parented execution Y. The link is explicit, recorded in the
   `Execution_Execution` table (`Execution → Nested_Execution`).
   Exposed today as `add_nested_execution` /
   `list_nested_executions`.

The new `lookup_lineage(rid)` method must decide which of these two
edge sets defines a "parent" when walking from an artifact backwards
through the chain that produced it.

## Decision

**`lookup_lineage` walks data-flow parents only. It does not
traverse `Execution_Execution`.**

Concretely: when expanding an execution node X, the parents are the
producing executions of X's consumed inputs (datasets and assets).
Nested-execution links are ignored.

## Rationale

The lineage method answers "how did this come to exist?" — a
question about data flow. Specifically:

- A user staring at `predictions.csv` wants to know which dataset
  trained the model that produced it, and which dataset was loaded
  to produce that training set. That's a chain of data dependencies.
- A nested execution can be a *sibling* in data-flow terms: the
  outer execution and an inner sub-execution can consume the same
  inputs and produce different outputs. Walking the
  `Execution_Execution` edge would surface the inner as a "parent"
  of the outer's outputs, which is wrong — they're peers.
- Mixing the two link types in one tree response makes the result
  harder to read: a user can't tell whether a node is in the chain
  because it produced something or because it called something.

If a user wants the orchestration view, that's already available via
the existing `list_nested_executions` API (and the corresponding MCP
tool). The two views answer different questions and should stay
separate.

## Alternatives considered

### Walk `Execution_Execution` as well, with edge-typed links

Add an `edge_type: "data" | "orchestration"` field on each parent
edge so the consumer can filter. Rejected because:

- Doubles the response size for the common case.
- Forces every consumer (skill text, MCP tool docs, the user reading
  the JSON) to learn a distinction they likely don't want.
- The orchestration view is already covered by
  `list_nested_executions`.

### Walk `Execution_Execution` only

Define lineage as orchestration topology. Rejected because it
doesn't answer the actual question users ask ("how did this come to
exist?") and produces wrong results when sub-executions are siblings
of the outer in data-flow terms.

### Defer the decision; let the caller pass an `edge_types` param

Ship with an `edge_types: set[str] = {"data"}` parameter that
defaults to data-flow but allows orchestration to be opted in.
Rejected — at v1 we have no use case for orchestration-walking that
isn't already served by `list_nested_executions`. Add it later if
demand materializes; the field is additive.

## Consequences

- The lineage tree is data-flow only. Users who want orchestration
  topology use `list_nested_executions`.
- The docstring on `lookup_lineage` must explicitly state this and
  point readers at `list_nested_executions` for the other view.
- If we later add forward traversal (`lookup_descendants`), it
  inherits the same decision: descendants are data-flow children
  (executions that consumed this artifact's outputs as inputs).
