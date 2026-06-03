# Find Executions by Dataset — Design

**Date:** 2026-06-02
**Status:** Design — pending review (revised to catalog-native role/version model)
**Repos touched:** `deriva-ml` (library + schema), `deriva-ml-mcp-plugin` (MCP tools)

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
against.

"Which experiments used this data?" is a headline provenance question
for a reproducibility platform — arguably more common than the
producer-direction question the lineage tools already answer well.

## Root cause: datasets never got the role/version model assets already have

Investigating the question exposed a deeper, structural asymmetry. The
catalog records dataset↔execution edges and asset↔execution edges very
differently:

| | Recorded in | Role stored? | Version stored? |
|---|---|---|---|
| **Asset** ↔ execution | `{Asset}_Execution` tables (`Image_Execution`, …) | **Yes** — `Asset_Role` FK (`Input`/`Output`) | n/a (assets aren't versioned) |
| **Output dataset** ↔ execution | `Dataset_Version.Execution` FK | implied (authorship) | **Yes** (the authored version row) |
| **Input dataset** ↔ execution | `Dataset_Execution` association | **No** | **No** (points at version-agnostic `Dataset`) |

Consequences of this split:

1. **Dataset role must be *derived*** (linked-but-not-author = input),
   instead of read from a column the way `Asset_Role` is.
2. **Input dataset version is lost** at the catalog level. It survives
   only inside the execution's `configuration.json` metadata asset
   (`DatasetSpec.version`, a required field) for config-declared inputs;
   for lightweight `add_input_dataset` links (e.g. `split_dataset`) it is
   recorded nowhere.
3. **Input and output datasets are modeled through two different tables**,
   so they can't be queried uniformly.

The principle this design follows: **provenance must be queryable
catalog data, not reconstructed by parsing a config file.** The
`configuration.json` asset records what was *requested at launch*; it is
not the catalog's authoritative model of what an execution used. The
read path must never fetch or parse it.

The fix is to give datasets the same first-class role/version model that
assets already have — then read everything relationally.

## Design

### Layer 0 — Schema: make `Dataset_Execution` role- and version-aware

Add two columns to the `deriva-ml:Dataset_Execution` association table:

- **`Role`** — FK to the existing `Asset_Role` vocabulary (reused, not a
  new vocab) carrying `Input` / `Output`. Mirrors `{Asset}_Execution`'s
  `Asset_Role`.
- **`Dataset_Version`** — nullable FK to `Dataset_Version.RID`, recording
  the *exact* version of the dataset involved in this edge.

The existing `Dataset` FK stays (the version-agnostic "which dataset"
link, needed for the "any version" query and preserved so every current
reader keeps working). Both new columns are nullable so pre-existing
rows remain valid.

This collapses the input/output dataset split: every dataset↔execution
edge becomes one uniform association row
`(Dataset, Dataset_Version, Execution, Role)` — structurally identical to
how `{Asset}_Execution` rows already work. Role stops being derived;
version becomes native on both sides.

The output side's existing `Dataset_Version.Execution` authorship FK is
retained as a convenience mirror (and to avoid a larger rewrite); it is
no longer the *only* source of output-edge truth.

### Layer 1 — Write path: populate Role + Version at link time

Both writers already have the role and version in hand at link time:

- **`create_dataset`** (output): writes `Role="Output"` and the authored
  `Dataset_Version` RID on the `Dataset_Execution` row it inserts.
- **`add_input_dataset(rid, version=…)`** (input): gains an optional
  `version` and writes `Role="Input"` plus the `Dataset_Version` RID when
  known. `split_dataset` (the canonical caller) passes the source
  version it already resolved.
- **Config-declared inputs** (`ExecutionConfiguration.datasets`): the
  `DatasetSpec.version` (required) is written to the association row at
  materialization time.

Backward compatibility: `add_input_dataset` keeps working without a
`version` argument (writes `Role="Input"`, `Dataset_Version=null`).

### Layer 2 — Library: extend `find_executions`

Extend the existing `DerivaML.find_executions` (in
`core/mixins/execution.py`) with dataset filters, rather than adding a
separate method — the dataset filter is structurally identical to the
existing `workflow_type` association-join filter, returns the same
`ExecutionRecord` type, and keeps the library method count minimal.

```python
def find_executions(
    self,
    workflow=None,
    workflow_type=None,
    status=None,
    dataset=None,            # NEW: RID or Dataset
    dataset_role="any",      # NEW: "input" | "output" | "any"
    dataset_version=None,    # NEW: pin to a specific version
    sort=None,
) -> Iterable["ExecutionRecord"]
```

**Filter mechanics** (now a pure relational read — no role derivation,
no config parsing):

1. `dataset` set → select `Dataset_Execution` rows where
   `Dataset == dataset_rid`, intersect their `Execution` RIDs into the
   result set (same pattern as `workflow_type`).
2. `dataset_role` → filters on the stored `Role` column directly
   (`"any"` = no filter).
3. `dataset_version` → filters on the stored `Dataset_Version` FK
   directly. Now meaningful on **both** sides (no longer output-only),
   because the column is populated for inputs too.

**Guardrails:** `dataset_role`/`dataset_version` without `dataset` →
`ValueError`.

Composes with `workflow` / `workflow_type` / `status` (e.g. "Training
executions that consumed dataset X at version 2.0.0").

### Layer 3 — Library: unify the role classification (cleanup)

With `Role` stored on the association, `list_input_datasets()` no longer
needs to *derive* role by excluding produced datasets — it filters
`Dataset_Execution` on `Role == "Input"`. Refactor it (and the
symmetric output accessor) to read the column. Internal change; public
signatures unchanged. Removes the `_producer_of_dataset` derivation as
the source of truth (kept only as a fallback for legacy null-`Role`
rows during the transition).

### Layer 4 — Plugin: two tool surfaces over one library method

Mirrors the existing precedent where
`deriva_ml_list_executions(workflow_rid=...)` and
`deriva_ml_find_workflow_executions(...)` both wrap `ml.find_executions`
via the shared `_list_executions_impl` helper, yet exist as two tools —
tools are shaped by LLM intent, not 1:1 with library methods.

**A. Extend** `deriva_ml_list_executions` with `dataset`,
`dataset_role="any"`, `dataset_version`. Forwarded into the extended
`_list_executions_impl`. When `dataset` is `None`, behavior and response
shape are identical to today — zero change for existing callers.

**B. Add** `deriva_ml_find_dataset_executions(dataset_rid,
dataset_role="any", dataset_version=None, status=None, limit, after_rid,
preflight_count, sort)`. Body shape identical to
`find_workflow_executions`; calls `_list_executions_impl`. Docstring
opens with the "Distinct from `deriva_ml_list_executions(dataset=...)`"
framing.

Both share `_list_executions_impl`, so wire shapes cannot drift.

### Layer 5 — Plugin: response shape

Introduce `DatasetExecutionSummary` extending `ExecutionSummary` with two
fields read **directly from the association row** (no extra fetch):

- `dataset_role: "input" | "output"`
- `dataset_version: str | None` — the stored version. Now populated on
  **both** sides for new rows; `null` only for legacy rows written before
  Layer 0, or for `add_input_dataset` calls that supplied no version.

Used only on the dataset-scoped paths (`find_dataset_executions` always;
`list_executions` only when `dataset` is set). Plain `ExecutionSummary`
retained otherwise. Wire shape:

```json
{"executions": [{"rid": "...", "workflow_rid": "...", "status": "...",
  "description": "...", "start_time": "...", "stop_time": "...",
  "duration": "...", "dataset_role": "input",
  "dataset_version": "2.0.0"}],
 "count": N, "truncated": false, "next_after_rid": null}
```

## Migration

There is **no migration framework** in `deriva-ml` (the `schema/` package
has only `create_schema.py` for fresh-catalog DDL and `annotations.py`).
Schema changes are applied imperatively via ERMrest column/FK calls.

- **New catalogs:** add the two columns + FKs in `create_schema.py`.
- **Existing catalogs** (e.g. `dev.eye-ai.org`): a one-shot upgrade
  script adds the `Role` + `Dataset_Version` columns and FKs, then
  **best-effort backfills**:
  - `Role`: derivable for every existing row via the current authorship
    rule (`Dataset_Version.Execution` authorship → `Output`, else
    `Input`). Backfill is complete.
  - `Dataset_Version`: backfillable for **output** rows (the authored
    version is known) and for **config-declared inputs** where the
    historical `configuration.json` is still readable. **Irreducibly
    `null`** for legacy `add_input_dataset` links that never recorded a
    version. This is a one-time historical gap; all new writes are clean.
- **Chaise annotations** (`annotations.py:525-526`): the existing
  `visible_foreign_keys` lists the two current FK constraint names; the
  new FKs are additive. Optionally add the `Role` / `Dataset_Version` FKs
  so they render in the Chaise UI (cosmetic).

## Blast radius (verified)

- **Writers (2):** `create_dataset` (`dataset.py:347`), input path
  (`execution.py:658`, `add_input_dataset` at `2007`). Both updated to
  set `Role` (+ `Dataset_Version`).
- **Readers (4):** `_helpers.list_input_datasets`,
  `Dataset.list_executions` (`dataset.py:2426`),
  `DatasetBag.list_executions` (`dataset_bag.py:906`), and the new
  filter. All currently select `Dataset`/`Execution` explicitly — none
  breaks on added columns; `list_input_datasets` is upgraded to read
  `Role`.
- **Bag export / offline ORM:** export traverses whole FK-connected
  tables (`dataset.py:2517`), and the offline SQLite ORM is
  reflection-based (SQLAlchemy `MetaData`), so the new columns round-trip
  automatically. Only new code that *reads* them needs writing.
- **`catalog.py`:** only *excludes* `Dataset_Execution` from asset-table
  discovery (`find_asset_execution_tables`) — untouched by added columns.

## Unification notes

- **Input datasets ↔ output datasets:** unified by Layer 0 — one
  association row with a `Role` column, exactly like `{Asset}_Execution`.
  This is the core of the design, not a side effect.
- **Output datasets ↔ output assets:** unified at the **API surface**,
  not storage. They remain different entity kinds (datasets are versioned
  collections; assets are files), so they keep separate tables. But once
  datasets carry `Role`, both edges read as `(thing, execution, role)`,
  making a future unified accessor (`record.list_outputs()`) or an
  asset-keyed `find_executions(asset=…)` twin a natural follow-on. Out of
  scope here; noted as the obvious next symmetry.
- **`list_*` accessor unification** (`list_execution_parents/children`
  etc.) remains deferred — deliberate naming symmetry with the dataset
  hierarchy API; a separate breaking refactor / future ADR.

## Testing

- **Schema/migration:** upgrade script adds columns + FKs idempotently;
  backfill assigns correct `Role` to every existing row and
  `Dataset_Version` where recoverable; `add_input_dataset` legacy rows
  end up `Role="Input"`, `Dataset_Version=null`.
- **Library:** `find_executions` with each `dataset_role`;
  `dataset_version` filtering on both sides; `ValueError` guardrails;
  composition with `status`/`workflow_type`; `list_input_datasets` reads
  `Role` and matches legacy derivation on null-`Role` rows.
- **Plugin:** both tool surfaces share-shape via `_list_executions_impl`;
  `DatasetExecutionSummary` carries `dataset_role` + `dataset_version`
  from the association row with no extra fetch; plain summary retained
  when `dataset` absent; preflight; `_error_envelope` on bad RID.

## Implementation order

1. Schema: add `Role` + `Dataset_Version` columns/FKs (`create_schema.py`
   + standalone upgrade/backfill script).
2. Library write path: `create_dataset`, `add_input_dataset(version=…)`,
   config-input writer populate the new columns.
3. Library read path: refactor `list_input_datasets` to read `Role`;
   extend `find_executions` with the dataset filters.
4. Plugin: extend `_list_executions_impl`; `DatasetExecutionSummary`;
   extend `deriva_ml_list_executions`; add
   `deriva_ml_find_dataset_executions`.
5. Tests at every layer; update the getting-started guide's tool menu.
