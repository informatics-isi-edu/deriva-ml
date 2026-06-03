# Find Executions by Dataset — Design

**Date:** 2026-06-02
**Status:** Approved design (pre-implementation)
**Repos touched:** `deriva-ml` (library), `deriva-ml-mcp-plugin` (MCP tools)

## Problem

There is no first-class way to answer **"which executions used dataset
X?"** — neither in the `deriva-ml` Python library nor in the
`deriva-ml-mcp-plugin` MCP surface.

The existing execution-query surface filters by **workflow**,
**workflow type**, or **status** only:

- Library: `DerivaML.find_executions(workflow, workflow_type, status, sort)`
  — no dataset parameter.
- Plugin tools: `deriva_ml_list_executions`,
  `deriva_ml_find_workflow_executions`, `deriva_ml_find_experiments`,
  `deriva_ml_list_execution_{children,parents}`,
  `deriva_ml_get_lineage` — none takes a dataset.
- Resources: `execution/{rid}` is per-execution (forward direction);
  `lineage/{rid}` walks **producing** executions only and is empty for
  curated root datasets (every version row has `execution_rid = null`).

Today the only way to get the answer is a hand-built ERMrest query
against the `Dataset_Execution` association table — the exact
"drop to `query_attribute`" anti-pattern the getting-started guide warns
against. The capability is one association join; it has simply never
been surfaced.

"Which experiments used this data?" is a headline provenance question
for a reproducibility platform — arguably more common than the
producer-direction question the lineage tools already answer well.

## Schema constraints (these drive the design)

`deriva-ml:Dataset_Execution` is a bare many-to-many association:
columns are just `Dataset` and `Execution` (plus system columns). The
table comment states it covers both "consume or produce". Critically:

1. **No role column.** Unlike assets — whose `{Asset}_Execution` rows
   carry an `Asset_Role` ("Input"/"Output") column that
   `list_assets(asset_role=...)` filters on directly — `Dataset_Execution`
   cannot natively distinguish inputs from outputs.

2. **No version column.** A link records `(Dataset, Execution)` but not
   *which version* of the dataset was involved.

The library already has a source of truth for role on the dataset side:
**`Dataset_Version.Execution` authorship.** An execution *produced*
(output) a dataset if a `Dataset_Version` row for that dataset points
its `Execution` FK back at the execution. `list_input_datasets()`
already relies on exactly this (see
`execution/_helpers.py::list_input_datasets`, which excludes
datasets the execution produced via `_producer_of_dataset`).

This is why the asset side has a clean `asset_role=` parameter while the
dataset side has a name-baked `list_input_datasets()` — role is *stored*
for assets but must be *derived* for datasets. Our design honors that:
the dataset filter derives role rather than pretending a column exists.

## Design

### Layer 1 — Library: extend `find_executions`

Rather than add a new method, extend the existing
`DerivaML.find_executions` (in `core/mixins/execution.py`) with three
optional, keyword-only dataset filters. This was chosen over a
standalone `find_dataset_executions` library method because:

- The dataset filter is structurally identical to the existing
  `workflow_type` filter — a join through an association table
  (`Dataset_Execution`, mirroring `Workflow_Workflow_Type`).
- It returns the same type (`ExecutionRecord`), unlike `find_experiments`
  (which returns `Experiment` and earned its own method).
- Keeps the library method count minimal; one method that already
  filters executions gains one more filter axis.

New signature:

```python
def find_executions(
    self,
    workflow=None,
    workflow_type=None,
    status=None,
    dataset=None,            # NEW: RID or Dataset — executions linked to this dataset
    dataset_role="any",      # NEW: "input" | "output" | "any"
    dataset_version=None,    # NEW: pin to a specific version (output side only)
    sort=None,
) -> Iterable["ExecutionRecord"]
```

**Filter mechanics** (mirrors the `workflow_type` association-join +
set-intersection pattern at execution.py:626-634):

1. When `dataset` is set, collect candidate `Execution` RIDs from
   `Dataset_Execution` where `Dataset == dataset_rid`.
2. Derive role for each candidate via a shared helper (Layer 2): an
   execution is **output** for the dataset if a `Dataset_Version` row
   for that dataset has `Execution ==` it; otherwise **input**.
3. `dataset_role` filters the candidate set:
   - `"any"` (default) — all linked executions, no role filter.
   - `"input"` — linked but not a producer of any version.
   - `"output"` — produced at least one version of the dataset.
4. `dataset_version`, when set, narrows the **output**-side match to the
   execution(s) that authored that specific `Dataset_Version`. It does
   **not** constrain the input side (no per-version link exists), and is
   a documented no-op when combined with `dataset_role="input"`.
5. The derived role and authored version travel with each result so the
   tool layer can surface them without recomputation.

**Guardrails:** passing `dataset_role` (≠ "any") or `dataset_version`
without `dataset` raises `ValueError` (fail fast on a nonsensical
combination).

The existing `workflow` / `workflow_type` / `status` filters compose
with the dataset filters (e.g. "Training-type executions that consumed
dataset X"), via the same intersect-into-result-set approach already
used for `workflow_type`.

### Layer 2 — Library: shared role-derivation helper

Factor the "classify a dataset↔execution link by role" logic into
`execution/_helpers.py`, consumed by **both**:

- the new `find_executions` dataset filter, and
- the existing `list_input_datasets()` (which currently inlines the
  "exclude produced datasets" logic via `_producer_of_dataset`).

This is an **internal DRY refactor — no public signature changes** to
`list_input_datasets()`. The helper returns, for a `(dataset, candidate
executions)` pair, each execution's derived role plus the authored
version (where it is a producer). Both call sites consume the same
classification so they cannot drift.

### Layer 3 — Plugin: two tool surfaces over one library method

Mirrors the existing precedent where `deriva_ml_list_executions(
workflow_rid=...)` and `deriva_ml_find_workflow_executions(...)` both
wrap `ml.find_executions` via the shared `_list_executions_impl` helper,
yet exist as two tools — tools are shaped by **LLM intent**, not by 1:1
correspondence to library methods.

**A. Extend the browser** `deriva_ml_list_executions`:
Add `dataset: str | None = None`, `dataset_role: str = "any"`,
`dataset_version: str | None = None` alongside the current filters.
They forward into the (extended) `_list_executions_impl` →
`ml.find_executions(...)`. Combinable with the existing filters. When
`dataset` is `None`, behavior and response shape are **identical to
today** — zero change for existing callers. The preflight branch picks
up the new filters for free.

**B. Add the intent tool** `deriva_ml_find_dataset_executions`:

```python
@ctx.tool(mutates=False)
async def deriva_ml_find_dataset_executions(
    hostname: str, catalog_id: str,
    dataset_rid: str,
    dataset_role: str = "any",          # "input" | "output" | "any"
    dataset_version: str | None = None,
    status: str | None = None,
    limit: int = 100,
    after_rid: str | None = None,
    preflight_count: bool = False,
    sort: bool = False,
) -> str
```

Body shape identical to `deriva_ml_find_workflow_executions`:
`asyncio.to_thread(_pkg.get_ml, …)` inside `with deriva_call():`, drain
the generator in the worker thread, `_paginate`,
`_error_envelope(operation="find_dataset_executions", audit=False)`.
Calls `_list_executions_impl(dataset=dataset_rid, dataset_role=...,
dataset_version=...)`. Preflight message:
`"Found N executions that <role> dataset {dataset_rid}. Choose a
limit…"`.

Docstring opens with the "Distinct from
`deriva_ml_list_executions(dataset=...)`" framing (mirroring the
workflow tool) and carries the two caveats below.

Both surfaces share `_list_executions_impl`, so their wire shapes cannot
drift — the same guarantee the workflow pair relies on.

### Layer 4 — Plugin: response shape

Introduce `DatasetExecutionSummary` extending the existing
`ExecutionSummary` (in `_response_models.py`) with two derived fields:

- `dataset_role: "input" | "output"` — the per-execution derived role.
- `dataset_version: str | None` — the dataset version the execution is
  associated with. **Populated on the output side** (authored version
  known via `Dataset_Version`); **`null` on the input side**
  (`Dataset_Execution` has no per-version link, so the consumed version
  cannot be determined without deeper inference).

Used **only** on the dataset-scoped paths:
- `deriva_ml_find_dataset_executions` always.
- `deriva_ml_list_executions` only when `dataset` is set.

When `dataset` is not set, `deriva_ml_list_executions` returns the plain
`ExecutionSummary` exactly as today. Wire shape:

```json
{"executions": [{"rid": "...", "workflow_rid": "...", "status": "...",
  "description": "...", "start_time": "...", "stop_time": "...",
  "duration": "...", "dataset_role": "input", "dataset_version": null}],
 "count": N, "truncated": false, "next_after_rid": null}
```

## Documented caveats (must appear in tool/method docstrings)

1. **Role is derived, not stored.** `dataset_role` is computed from
   `Dataset_Version` authorship — an execution is `output` if it authored
   a version of the dataset, otherwise `input`. There is no role column
   on `Dataset_Execution`.
2. **Version pinning is output-side only.** `dataset_version` constrains
   only the producing (output) side; on the input side it is a no-op,
   and the returned `dataset_version` is `null` for input executions.

## Out of scope (deferred)

**Unifying the `list_*` relation accessors** (`list_input_datasets`,
`list_assets`, `list_execution_parents`, `list_execution_children`).
These already share parameterized helpers underneath
(`fetch_nested_execution_rows(direction=...)`,
`list_assets(asset_role=...)`); the duplication is confined to thin
public façades. The `parents`/`children` split is a deliberate naming
mirror of `list_dataset_parents`/`list_dataset_children`, established in
a documented "R5.1 hard cutover" with no compat alias. Collapsing the
execution pair alone would break that symmetry and cascade into the
dataset API. This is a separate, breaking refactor — a candidate future
ADR, not part of this additive change.

An **asset-keyed twin** (`find_executions(asset=...)`) is the obvious
future symmetric capability; this design's parameter-naming convention
(`dataset_role` mirroring `asset_role`) is chosen to make that twin a
natural follow-on.

## Testing

- **Library unit tests** (`tests/`): `find_executions` with each
  `dataset_role` value against a fixture catalog with known
  input-only, output-only, and both-role executions; `dataset_version`
  output-side pinning; `ValueError` guardrails; composition with
  `status`/`workflow_type`. Assert the shared helper produces identical
  role classification as `list_input_datasets()` for the input case.
- **Plugin tests** (`tests/test_execution.py` patterns): both tool
  surfaces return shape-identical payloads via `_list_executions_impl`;
  `dataset_role`/`dataset_version` surfaced on `DatasetExecutionSummary`;
  plain `ExecutionSummary` retained when `dataset` is absent; preflight
  counts; `_error_envelope` on bad RID.

## Implementation order

1. Library: shared role-derivation helper in `execution/_helpers.py`;
   refactor `list_input_datasets()` to consume it (no API change).
2. Library: extend `find_executions` with the three params + guardrails.
3. Plugin: extend `_list_executions_impl` to forward the dataset params.
4. Plugin: `DatasetExecutionSummary` response model.
5. Plugin: extend `deriva_ml_list_executions`; add
   `deriva_ml_find_dataset_executions`.
6. Tests at both layers; update the getting-started guide's tool menu.
