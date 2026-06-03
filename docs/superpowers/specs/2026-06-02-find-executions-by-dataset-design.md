# Find Executions by Dataset — Design

**Date:** 2026-06-02
**Status:** Design — pending review (authorship-canonical model)
**Branch:** `feature/find-executions-by-dataset`
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

Today the only way to get the answer is a hand-built ERMrest query
against the `Dataset_Execution` association table — the
"drop to `query_attribute`" anti-pattern the getting-started guide warns
against. "Which experiments used this data?" is a headline provenance
question for a reproducibility platform.

## Two edge types, two sources of truth (the model)

A dataset relates to an execution in exactly two ways, and the catalog
**already records each one in its own authoritative place**:

| Edge | Meaning | Sole source of truth |
|------|---------|----------------------|
| **Output** | execution *produced* version V of the dataset | `Dataset_Version.Execution` — the per-version author FK |
| **Input** | execution *consumed* the dataset | `Dataset_Execution` association row |

This is the key design decision (resolving an earlier dual-source-of-truth
concern): **we keep these as two separate, non-overlapping sources rather
than mirroring output edges into a `Role` column on `Dataset_Execution`.**

- `Dataset_Version.Execution` stays the **single** source of truth for
  "who produced version V." `get_lineage` already uses it
  (`execution.py:865`); `_producer_of_dataset` already reads it
  (`execution.py:1032`). The `DatasetHistory` value object
  (`dataset/aux_classes.py:206`) already models it: `dataset_rid`,
  `dataset_version`, `version_rid`, and `execution_rid` (nullable — `None`
  when the version was created outside an execution, e.g. curated
  datasets like `2-7P5P` whose 29 version rows are all null).
- `Dataset_Execution` becomes **input-only by definition.** Every row is a
  consume edge. It gains nothing but a nullable `Dataset_Version` FK to
  record *which version* was consumed.

No `Role` column. No overlap. No drift. No sync burden. The producer
question is answered by version authorship; the consumer question by the
association table; "any" is their union.

**Principle:** provenance is queryable catalog data, never reconstructed
by parsing a config file. The `configuration.json` asset records what was
*requested at launch*; it is not consulted on the read path. (It is read
exactly once, at migration time, as a best-effort backfill source — see
remediation.)

## How "what produced a dataset" is determined (precise definition)

`Dataset_Version` is a **per-version** table: one row per
`(dataset, version)`, each row's `Execution` FK = *"RID of the execution
that produced this version"* (nullable; `null` = produced outside an
execution / curated / dev row).

So "executions that produced dataset X" = the set of non-null
`Dataset_Version.Execution` values across X's version history (read via
`DatasetHistory`). It is naturally **per-version**:

- `dataset_role="output"`, no version pin → **every** execution that
  authored **any** version of X.
- version pinned (via `DatasetSpec`, below) → the execution that authored
  **that** version.

`_producer_of_dataset`'s "highest-semver version" rule stays an internal
convenience ("the current producer"); it is **not** the tool's output
semantics.

## Design

### Layer 0 — Schema: one nullable FK on `Dataset_Execution`

Add a single column to `deriva-ml:Dataset_Execution`:

- **`Dataset_Version`** — nullable FK to `Dataset_Version.RID`, recording
  the exact version consumed by this input edge.

The existing `Dataset` FK stays (version-agnostic "which dataset"). No
`Role` column. In `schema/create_schema.py`, add the column + FK to the
`Dataset_Execution` definition (near line 170); in `schema/annotations.py`
(`visible_foreign_keys`, ~line 525) add the new FK so it renders in
Chaise. `demo_catalog.py` inherits the change for tests.

### Layer 1 — Write path

- **`create_dataset`** (output): **stops** writing a `Dataset_Execution`
  row (`dataset.py:347`). Output provenance is recorded **only** via
  `Dataset_Version.Execution` (which `_insert_dataset_versions` already
  writes). This removes the redundant write that caused the dual-source
  concern. *(Implementation check: confirm no reader depends on output
  datasets appearing in a `Dataset_Execution` scan — see Blast radius;
  `Dataset.list_executions` is the one affected, and the change aligns it
  with its own docstring.)*
- **`add_input_dataset(rid, version=…)`** (input): gains an optional
  `version`; writes the `Dataset_Execution` row with the `Dataset_Version`
  FK populated when known. `split_dataset` (the canonical caller) passes
  the source version it already resolved.
- **Config-declared inputs** (`ExecutionConfiguration.datasets`): the
  required `DatasetSpec.version` is written to the `Dataset_Version` FK at
  materialization time.

Backward compatible: `add_input_dataset(rid)` still works, leaving
`Dataset_Version=null`.

### Layer 2 — Library: extend `find_executions`

Extend the existing `DerivaML.find_executions` (in
`core/mixins/execution.py`) rather than adding a method — keeps the
library minimal and the dataset filter composes with the existing ones.

```python
def find_executions(
    self,
    workflow=None,
    workflow_type=None,
    status=None,
    dataset=None,            # NEW: RID | DatasetSpec
    dataset_role="any",      # NEW: "input" | "output" | "any"
    sort=None,
) -> Iterable["ExecutionRecord"]
```

**The `dataset` argument is `RID | DatasetSpec`** — reusing the existing
`DatasetSpec` value object (`dataset/aux_classes.py:312`) that
`ExecutionConfiguration.datasets` already uses, so a spec from an
execution config can be passed straight in:

- **bare RID** → "this dataset, **any** version."
- **`DatasetSpec`** → pins the version (`spec.rid` + `spec.version`). The
  spec's bag-download fields (`materialize`, `timeout`, `exclude_tables`,
  `fetch_concurrency`, `description`) are **ignored** by the filter; only
  `rid` + `version` are read. Documented explicitly.

There is **no separate `dataset_version` scalar** — version travels
inside the `DatasetSpec`, eliminating incoherent scalar combinations.

**Filter mechanics** (pure relational reads — no role derivation, no
config parsing):

1. Resolve `dataset` to `(rid, version|None)`.
2. By role:
   - `"input"` → `Dataset_Execution` rows where `Dataset == rid`
     (and `Dataset_Version == version` when pinned).
   - `"output"` → executions from `Dataset_Version.Execution` across the
     dataset's history (filtered to the pinned version when set) — read
     via `DatasetHistory`.
   - `"any"` → union of the two.
3. Intersect the resulting execution set with the other active filters
   (`workflow` / `workflow_type` / `status`), same pattern as
   `workflow_type` today.

**Guardrails:** `dataset_role` ≠ `"any"` without `dataset` → `ValueError`.

### Layer 3 — Library: simplify `list_input_datasets`

With `Dataset_Execution` now input-only, `list_input_datasets()` no
longer needs to *subtract* produced datasets (its current
`_producer_of_dataset` exclusion) — every row is already an input. It
becomes a plain read of `Dataset_Execution`. Public signature unchanged;
internals simplified. (Legacy catalogs: the migration removes the stale
output rows so this holds for historical data too — see remediation.)

### Layer 4 — Plugin: two tool surfaces over one library method

Mirrors the precedent where `deriva_ml_list_executions(workflow_rid=...)`
and `deriva_ml_find_workflow_executions(...)` both wrap
`ml.find_executions` via the shared `_list_executions_impl` helper yet
exist as two tools — shaped by LLM intent, not 1:1 with methods.

The MCP wire surface can't pass a Python `DatasetSpec`, so the tools take
scalars and construct the spec internally:

**A. Extend** `deriva_ml_list_executions` with `dataset: str | None`,
`dataset_role: str = "any"`, `dataset_version: str | None`. When
`dataset_version` is set, the tool builds `DatasetSpec(rid=dataset,
version=dataset_version)`; otherwise passes the bare RID. When `dataset`
is `None`, behavior/shape are identical to today.

**B. Add** `deriva_ml_find_dataset_executions(dataset_rid,
dataset_role="any", dataset_version=None, status=None, limit, after_rid,
preflight_count, sort)`. Body shape identical to
`find_workflow_executions`; constructs the spec and calls
`_list_executions_impl`. Docstring opens with "Distinct from
`deriva_ml_list_executions(dataset=...)`".

Both share `_list_executions_impl`, so wire shapes cannot drift.

*(The scalar `dataset` + `dataset_version` on the MCP boundary is a
transport concession — the typed `DatasetSpec` lives in the Python API.
The tool layer immediately lifts the scalars into a spec, so the
"incoherent scalar combination" risk is confined to one well-tested
construction site, not spread across the API.)*

### Layer 5 — Plugin: response shape

Introduce `DatasetExecutionSummary` extending `ExecutionSummary` with two
derived fields:

- `dataset_role: "input" | "output"` — **which source the edge came
  from** (input = `Dataset_Execution`; output = `Dataset_Version`
  authorship), not a stored column.
- `dataset_version: str | None` — the version. For input edges, read from
  the `Dataset_Execution.Dataset_Version` FK (null for legacy/no-version
  `add_input_dataset` links). For output edges, the authored version from
  `DatasetHistory`. No extra catalog fetch beyond the rows already read.

Used only on the dataset-scoped paths (`find_dataset_executions` always;
`list_executions` only when `dataset` is set). Plain `ExecutionSummary`
otherwise. Wire shape:

```json
{"executions": [{"rid": "...", "workflow_rid": "...", "status": "...",
  "description": "...", "start_time": "...", "stop_time": "...",
  "duration": "...", "dataset_role": "input",
  "dataset_version": "2.0.0"}],
 "count": N, "truncated": false, "next_after_rid": null}
```

## Behavior change to call out

`Dataset.list_executions` (`dataset.py:2426`) is documented as *"all
executions that used this dataset as **input**"* but currently scans all
`Dataset_Execution` rows — which today **include** the output rows
`create_dataset` writes. Making the table input-only (Layer 1) **fixes
this method to match its own docstring**: it stops returning the
producing execution. This is an intentional, documented behavior change;
callers wanting producers use the output side of `find_executions` /
`get_lineage`.

## Schema creation update (fresh catalogs)

`create_schema.py`: add the nullable `Dataset_Version` column + FK to the
`Dataset_Execution` definition. `annotations.py`: add the FK to
`visible_foreign_keys`. Fresh catalogs are version-aware with no
migration; `demo_catalog.py` (test suite) inherits it.

## Catalog deployment & old-data remediation (existing catalogs)

No migration framework exists; changes are applied imperatively via
standalone scripts in `scripts/`. Ship one idempotent script,
`scripts/migrate_dataset_execution_version.py`, modeled on
`scripts/migrate_workflow_types.py` (same shape: `(hostname, catalog_id)`
args, `--dry-run`, `--schema`, `check_preconditions`, per-step `[SKIP]`
guards, verification, "already complete" short-circuit).

Steps:

1. **Preconditions:** report whether the `Dataset_Version` column/FK
   exists and count rows. Short-circuit if migrated.
2. **Add column + FK** (idempotent, additive, nullable — no reader breaks
   at this step).
3. **Remove stale output rows** — delete `Dataset_Execution` rows that
   correspond to an *output* edge (the dataset has a `Dataset_Version`
   authored by that execution). These are the redundant rows
   `create_dataset` historically wrote; output provenance is preserved in
   `Dataset_Version.Execution`, so deletion loses no information and
   de-duplicates the model. **This is the data-remediation core.**
4. **Backfill input `Dataset_Version`** — best-effort for the remaining
   (input) rows:
   - **Config-declared inputs:** resolve the consumed version from the
     execution's `configuration.json` (`DatasetSpec.version`) — read
     **at migration time only**, never on the steady-state read path.
   - **`add_input_dataset` links:** **irreducibly `null`** — no version
     was ever recorded. Reported in the summary as the known gap.
5. **Verify:** no output rows remain in `Dataset_Execution`; report the
   count of `Dataset_Version` nulls by cause (legacy `add_input_dataset`
   vs. unreadable config) so the residual is visible, not silent.

**Deployment order — dev first, then production:**

1. `--dry-run` on **`dev.eye-ai.org` / eye-ai** (the catalog with the
   48-link `2-7P5P` history we inspected); review.
2. Real run on **dev**; validate end-to-end with the new tools
   (`find_dataset_executions` on `2-7P5P`; confirm a `split_dataset`-origin
   input row shows `dataset_version=null`; confirm no output rows linger).
3. `--dry-run` then real run on **production** (`eye.rosci.org` / prod
   eye-ai) — same script, same idempotency.
4. Re-runnable: the precondition short-circuit makes a second run a no-op.

**Sequencing constraint:** deploy schema + remediation **before** the
library write-path upgrade on any client writing to these catalogs.
Nullable column ⇒ un-upgraded writers stay compatible during rollout.

**eye-ai-ml repo note:** eye-ai keeps catalog migrations under
`eye-ai-ml/scripts/catalog_management/`. The generic script lives in
`deriva-ml/scripts/`; a thin eye-ai wrapper pinning the two hosts/catalogs
may live in `catalog_management/` if the team prefers.

## Blast radius (verified)

- **Writers (2):** `create_dataset` (`dataset.py:347`) — *stops* writing
  the output row; input path (`execution.py:658`, `add_input_dataset` at
  `2007`) — writes the `Dataset_Version` FK.
- **Readers (4):** `_helpers.list_input_datasets` (simplified — no more
  authorship subtraction), `Dataset.list_executions` (`dataset.py:2426`,
  now correctly input-only — see Behavior change), `DatasetBag.list_executions`
  (`dataset_bag.py:906`, input-only offline mirror; unchanged query),
  the new filter. None breaks on an added column.
- **`get_lineage` / `_producer_of_dataset`:** already read
  `Dataset_Version.Execution` for producers — **unaffected** and now the
  sole producer source, exactly as intended.
- **Bag export / offline ORM:** whole-table FK traversal
  (`dataset.py:2517`) + reflection-based SQLite ORM ⇒ new column
  round-trips automatically.
- **`catalog.py`:** only *excludes* `Dataset_Execution` from asset-table
  discovery — untouched.

## Unification notes

- **Input ↔ output datasets:** unified at the *query* level —
  `find_executions(dataset=, dataset_role=)` answers both from their
  respective canonical sources, with `DatasetSpec` pinning version on
  either side. Storage stays appropriately separate (authorship vs.
  association).
- **Output datasets ↔ output assets:** API-surface symmetry only (an
  asset-keyed `find_executions(asset=…)` twin is the natural follow-on);
  different entity kinds keep different tables. Deferred.
- **`list_*` accessor unification:** deferred (deliberate naming symmetry
  with the dataset-hierarchy API; future ADR).

## Testing (new test cases)

**Schema creation:** fresh catalog's `Dataset_Execution` has the nullable
`Dataset_Version` column + FK to `Dataset_Version`.

**Write path:**
- `create_dataset` writes **no** `Dataset_Execution` row; output
  provenance present only via `Dataset_Version.Execution`.
- `add_input_dataset(rid, version=v)` writes the row with `Dataset_Version`
  set; `add_input_dataset(rid)` leaves it null (back-compat).
- `split_dataset` records its source input with the resolved version.

**Library read path:**
- `find_executions(dataset=rid)` (any version) returns inputs + outputs
  (union); `dataset_role="input"|"output"` each return the correct subset
  from the correct source.
- `find_executions(dataset=DatasetSpec(rid, version))` filters both sides
  to that version; bag-download fields on the spec are ignored.
- `ValueError` when `dataset_role` given without `dataset`.
- Composition with `workflow_type` / `status`.
- `list_input_datasets()` returns only inputs (regression: a dataset the
  execution produced does **not** appear), via the simplified read.

**Migration script** (seeded demo catalog):
- Idempotency (second run no-op); `--dry-run` makes no changes.
- After run: **no output rows remain** in `Dataset_Execution`; input rows
  from config have a `Dataset_Version`; an `add_input_dataset`-origin row
  is null and counted in the residual summary.
- Stale-output-row deletion targets exactly the authored-version rows and
  leaves input rows intact.

**Plugin:**
- Both surfaces share-shape via `_list_executions_impl`.
- `DatasetExecutionSummary` carries `dataset_role` + `dataset_version`
  with **no extra catalog/Hatrac fetch** (asserted).
- Plain `ExecutionSummary` when `dataset` absent (existing-caller guard).
- `dataset_version` scalar lifts into a `DatasetSpec` correctly; preflight;
  `_error_envelope` on bad RID.

## Implementation order

1. **Schema:** `create_schema.py` + `annotations.py` add the
   `Dataset_Version` column/FK. Schema-creation tests.
2. **Migration script:** `migrate_dataset_execution_version.py` (add FK →
   delete stale output rows → backfill input version → verify),
   `--dry-run`, idempotent. Migration tests.
3. **Catalog deployment:** dev dry-run → dev run → validate → production
   dry-run → production run. (Before client write-path upgrade.)
4. **Library write path:** `create_dataset` stops writing the output row;
   `add_input_dataset(version=…)` + config writer set the FK. Write tests.
5. **Library read path:** simplify `list_input_datasets`; extend
   `find_executions` with `dataset: RID|DatasetSpec` + `dataset_role`.
   Read tests.
6. **Plugin:** extend `_list_executions_impl`; `DatasetExecutionSummary`;
   extend `deriva_ml_list_executions`; add
   `deriva_ml_find_dataset_executions`. Plugin tests; update the
   getting-started guide's tool menu.
