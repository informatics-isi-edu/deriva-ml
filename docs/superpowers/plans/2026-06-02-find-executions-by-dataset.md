# Find Executions by Dataset — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make "which executions used dataset X?" a first-class query in the deriva-ml library and the deriva-ml-mcp-plugin MCP tools, with input/output role and version pinning, backed by catalog-native provenance (no config parsing).

**Architecture:** Two non-overlapping sources of truth — output edges live in `Dataset_Version.Execution` (per-version authorship), input edges in the `Dataset_Execution` association table (made input-only, gaining a nullable `Dataset_Version` FK). `find_executions(dataset=RID|DatasetSpec, dataset_role=...)` unions/filters the two. A one-shot idempotent migration adds the column, deletes redundant historical output rows, and best-effort backfills input versions.

**Tech Stack:** Python, deriva-py (ERMrest model + pathBuilder), Pydantic, pytest. Schema via `deriva.core.ermrest_model`. Plugin tools via `@ctx.tool` + `asyncio.to_thread`.

**Spec:** `docs/superpowers/specs/2026-06-02-find-executions-by-dataset-design.md`
**Branch:** `feature/find-executions-by-dataset` (already created)

**Repo roots:**
- Library: `/Users/carl/GitHub/DerivaML/deriva-ml`
- Plugin: `/Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin`

**Test convention note:** Library live/catalog tests use the demo catalog
via `tests/catalog_manager.py` fixtures and run with
`DERIVA_ML_ALLOW_DIRTY=true uv run pytest …`. Plugin tests use the
`mock_ml` (MagicMock) + `execution_ctx` + `capturing_mcp` fixtures in
`deriva-ml-mcp-plugin/tests/conftest.py` / `tests/test_execution.py`.

---

## File structure

**Library (`deriva-ml`):**
- `src/deriva_ml/schema/create_schema.py` — add `Dataset_Version` FK to `Dataset_Execution` (Task 1)
- `src/deriva_ml/schema/annotations.py` — surface the new FK in Chaise (Task 1)
- `scripts/migrate_dataset_execution_version.py` — **new**, idempotent migration (Task 2)
- `src/deriva_ml/dataset/dataset.py` — stop writing output row in `create_dataset` (Task 4)
- `src/deriva_ml/execution/execution.py` — `add_input_dataset(version=…)` (Task 4)
- `src/deriva_ml/execution/_helpers.py` — simplify `list_input_datasets`; add version to input-write helper (Tasks 4–5)
- `src/deriva_ml/core/mixins/execution.py` — extend `find_executions` (Task 5)

**Plugin (`deriva-ml-mcp-plugin`):**
- `src/deriva_ml_mcp_plugin/_response_models.py` — `DatasetExecutionSummary` (Task 6)
- `src/deriva_ml_mcp_plugin/tools/execution/read.py` — extend `_list_executions_impl`, `deriva_ml_list_executions`; add `deriva_ml_find_dataset_executions` (Task 6)

---

## Task 1: Schema — add `Dataset_Version` FK to `Dataset_Execution`

**Files:**
- Modify: `src/deriva_ml/schema/create_schema.py:168-177` (the `Dataset_Execution` association definition)
- Modify: `src/deriva_ml/schema/annotations.py` (visible_foreign_keys, ~line 525)
- Test: `tests/schema/test_dataset_execution_version_column.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/schema/test_dataset_execution_version_column.py`:

```python
"""Fresh-catalog schema: Dataset_Execution carries a nullable Dataset_Version FK."""
from tests.catalog_manager import ensure_demo_catalog  # see existing schema tests for the helper


def test_dataset_execution_has_dataset_version_fk(demo_model, ml_schema):
    """Dataset_Execution has a nullable Dataset_Version column FK'd to Dataset_Version."""
    de = demo_model.schemas[ml_schema].tables["Dataset_Execution"]
    cols = {c.name for c in de.columns}
    assert "Dataset_Version" in cols, "Dataset_Execution must have a Dataset_Version column"
    dv_col = next(c for c in de.columns if c.name == "Dataset_Version")
    assert dv_col.nullok is True, "Dataset_Version must be nullable"
    fk_targets = {
        fk.referenced_columns[0]["table_name"]
        for fk in de.foreign_keys
        if any(c["column_name"] == "Dataset_Version" for c in fk.foreign_key_columns)
    }
    assert "Dataset_Version" in fk_targets
```

(If `demo_model` / `ml_schema` fixtures don't already exist in
`tests/schema/conftest.py`, mirror the model-loading fixture used by the
other `tests/schema/` tests — they load the demo catalog model.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/schema/test_dataset_execution_version_column.py -v`
Expected: FAIL — `"Dataset_Execution must have a Dataset_Version column"`.

- [ ] **Step 3: Add the column + FK in create_schema.py**

The table is built with `Table.define_association`, which doesn't take
extra columns. Add the column and FK **after** creation. Replace the
block at `create_schema.py:168-177`:

```python
    dataset_execution_table = schema.create_table(
        Table.define_association(
            associates=[("Dataset", dataset_table), ("Execution", execution_table)],
            comment=(
                "Association linking datasets to executions that CONSUMED them "
                "as inputs. Output (producing) edges are NOT recorded here — they "
                "live in Dataset_Version.Execution. Each row optionally records "
                "the consumed version via the Dataset_Version FK."
            ),
        )
    )
    # Input edges may pin the consumed version. Nullable: legacy/no-version
    # links (e.g. add_input_dataset without a version) leave it NULL.
    dataset_execution_table.create_column(
        Column.define(
            "Dataset_Version",
            BuiltinType.text,
            nullok=True,
            comment="RID of the Dataset_Version consumed by this input edge (NULL if unknown).",
        )
    )
    dataset_execution_table.create_reference(dataset_version_table)
    return dataset_table
```

Notes for the implementer:
- `dataset_version_table` is the `Dataset_Version` table created earlier
  in this same function (the one `Dataset.Version` references via
  `create_reference(("Version", True, dataset_version))` at line ~154).
  If it isn't already bound to a local name, capture it from its
  `schema.create_table(...)` return value.
- `Column` / `BuiltinType` are already imported in this module (see the
  `ColumnDef`/`BuiltinType` usage elsewhere in the file). If the helper
  is `ColumnDef` rather than `Column.define`, match the surrounding
  style.

- [ ] **Step 4: Surface the FK in Chaise annotations**

In `src/deriva_ml/schema/annotations.py`, find the `visible_foreign_keys`
block for the Execution/Dataset area near line 525 (where
`Dataset_Execution_Dataset_fkey` is listed) and add the new FK so it
renders:

```python
                        {"outbound": [schema, "Dataset_Execution_Dataset_Version_fkey"]},
```

(Match the exact constraint name deriva-py generates for the new FK; if
it differs, read it from the created model in a quick REPL or from the
migration's precondition output once Task 2 exists.)

- [ ] **Step 5: Run test to verify it passes**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/schema/test_dataset_execution_version_column.py -v`
Expected: PASS.

- [ ] **Step 6: Run the broader schema-creation suite for regressions**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/schema/ -v`
Expected: PASS (no other schema test broken by the additive column).

- [ ] **Step 7: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/schema/create_schema.py src/deriva_ml/schema/annotations.py tests/schema/test_dataset_execution_version_column.py
git commit -m "feat(schema): add nullable Dataset_Version FK to Dataset_Execution"
```

---

## Task 2: Migration script — add column, delete stale output rows, backfill input versions

**Files:**
- Create: `scripts/migrate_dataset_execution_version.py`
- Test: `tests/test_migrate_dataset_execution_version.py` (create)

Model on `scripts/migrate_workflow_types.py` (same arg/`--dry-run`/
precondition/`[SKIP]`/verify/short-circuit shape).

- [ ] **Step 1: Write the failing test (idempotency + behavior on a seeded model)**

Create `tests/test_migrate_dataset_execution_version.py`. The migration
operates on an `ErmrestCatalog`; test against the demo catalog the other
live tests use (gate with `DERIVA_ML_ALLOW_DIRTY=true`).

```python
"""Migration: Dataset_Execution gains version FK, output rows removed, input backfilled."""
import pytest
from scripts.migrate_dataset_execution_version import (
    check_preconditions, run_migration,
)


@pytest.mark.integration
def test_migration_is_idempotent(demo_catalog, ml_schema):
    """Second run is a no-op; preconditions report 'already migrated'."""
    run_migration(demo_catalog, ml_schema, dry_run=False)
    pre = check_preconditions(demo_catalog, ml_schema)
    assert pre["has_version_column"] is True
    # Second run changes nothing.
    summary = run_migration(demo_catalog, ml_schema, dry_run=False)
    assert summary["columns_added"] == 0
    assert summary["output_rows_deleted"] == 0


@pytest.mark.integration
def test_migration_removes_output_rows_and_keeps_inputs(demo_catalog, ml_schema, seeded_edges):
    """Output rows (dataset has a Dataset_Version authored by the execution) are deleted;
    input rows survive."""
    run_migration(demo_catalog, ml_schema, dry_run=False)
    pb = demo_catalog.getPathBuilder()
    de = pb.schemas[ml_schema].Dataset_Execution
    rows = list(de.entities().fetch())
    # No row should correspond to an authored (output) version.
    for r in rows:
        assert not seeded_edges.is_output_edge(r["Dataset"], r["Execution"]), (
            "output edge should have been deleted from Dataset_Execution"
        )
```

(`seeded_edges` is a fixture you add in the test file that creates: one
dataset produced by an execution — i.e. a `Dataset_Version` row with that
`Execution` — plus a separate input link via `add_input_dataset`. Build it
with the demo `ml` instance. `is_output_edge` checks the
`Dataset_Version` authorship for the pair.)

- [ ] **Step 2: Run test to verify it fails**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/test_migrate_dataset_execution_version.py -v -m integration`
Expected: FAIL — `ModuleNotFoundError: scripts.migrate_dataset_execution_version`.

- [ ] **Step 3: Write the migration script**

Create `scripts/migrate_dataset_execution_version.py`:

```python
#!/usr/bin/env python3
"""Migrate Dataset_Execution to input-only with a Dataset_Version FK.

Steps:
1. Add a nullable Dataset_Version column + FK (idempotent).
2. Delete stale OUTPUT rows: Dataset_Execution rows where the dataset has
   a Dataset_Version authored by that same execution. Output provenance
   lives in Dataset_Version.Execution, so deletion loses nothing.
3. Backfill Dataset_Version on remaining (input) rows, best-effort, from
   the execution's configuration.json metadata (DatasetSpec.version).
   add_input_dataset links with no recorded version stay NULL.
4. Verify: no output rows remain; report NULL count by cause.

Usage:
    python scripts/migrate_dataset_execution_version.py HOST CATALOG_ID --dry-run
    python scripts/migrate_dataset_execution_version.py HOST CATALOG_ID
"""
from __future__ import annotations

import argparse
import sys

from deriva.core import ErmrestCatalog, get_credential
from deriva.core.ermrest_model import Column, builtin_types


def get_catalog(hostname: str, catalog_id: str) -> ErmrestCatalog:
    return ErmrestCatalog("https", hostname, catalog_id, credentials=get_credential(hostname))


def check_preconditions(catalog: ErmrestCatalog, ml_schema: str) -> dict:
    model = catalog.getCatalogModel()
    de = model.schemas[ml_schema].tables["Dataset_Execution"]
    has_col = "Dataset_Version" in {c.name for c in de.columns}
    pb = catalog.getPathBuilder()
    rows = list(pb.schemas[ml_schema].Dataset_Execution.entities().fetch())
    return {"has_version_column": has_col, "row_count": len(rows)}


def _authored_versions(catalog: ErmrestCatalog, ml_schema: str) -> set[tuple[str, str]]:
    """Set of (Dataset, Execution) pairs where the execution authored a version."""
    pb = catalog.getPathBuilder()
    dv = pb.schemas[ml_schema].Dataset_Version
    pairs = set()
    for row in dv.entities().fetch():
        if row.get("Execution") and row.get("Dataset"):
            pairs.add((row["Dataset"], row["Execution"]))
    return pairs


def step1_add_column(catalog, ml_schema, dry_run) -> int:
    model = catalog.getCatalogModel()
    de = model.schemas[ml_schema].tables["Dataset_Execution"]
    if "Dataset_Version" in {c.name for c in de.columns}:
        print("  [SKIP] Dataset_Version column already exists")
        return 0
    if dry_run:
        print("  [DRY-RUN] would add Dataset_Version column + FK")
        return 1
    de.create_column(Column.define("Dataset_Version", builtin_types.text, nullok=True))
    dv = model.schemas[ml_schema].tables["Dataset_Version"]
    de.create_reference(dv)
    print("  [OK] added Dataset_Version column + FK")
    return 1


def step2_delete_output_rows(catalog, ml_schema, dry_run) -> int:
    authored = _authored_versions(catalog, ml_schema)
    pb = catalog.getPathBuilder()
    de = pb.schemas[ml_schema].Dataset_Execution
    to_delete = [
        r["RID"] for r in de.entities().fetch()
        if (r["Dataset"], r["Execution"]) in authored
    ]
    if not to_delete:
        print("  [SKIP] no stale output rows")
        return 0
    if dry_run:
        print(f"  [DRY-RUN] would delete {len(to_delete)} output rows")
        return len(to_delete)
    for rid in to_delete:
        de.filter(de.RID == rid).delete()
    print(f"  [OK] deleted {len(to_delete)} output rows")
    return len(to_delete)


def step3_backfill_input_versions(catalog, ml_schema, dry_run) -> dict:
    """Best-effort: resolve consumed version from each execution's
    configuration.json. Rows that can't be resolved stay NULL.
    Returns {"filled": n, "null_config": n, "null_no_record": n}."""
    # Implementer: read each remaining row's Execution -> its
    # Execution_Metadata configuration.json asset -> DatasetSpec.version
    # for the matching dataset rid; set Dataset_Version via the version's
    # Dataset_Version RID. Skip+count rows whose config is missing.
    # This is the ONLY place configuration.json is read (migration-time only).
    raise NotImplementedError  # replace with the resolution loop below


def run_migration(catalog, ml_schema: str, dry_run: bool) -> dict:
    print("== migrate Dataset_Execution version ==")
    added = step1_add_column(catalog, ml_schema, dry_run)
    deleted = step2_delete_output_rows(catalog, ml_schema, dry_run)
    backfill = {"filled": 0, "null_config": 0, "null_no_record": 0}
    if not dry_run and added == 0:
        # column present -> safe to backfill
        backfill = step3_backfill_input_versions(catalog, ml_schema, dry_run)
    print(f"== summary: +{added} col, -{deleted} output rows, backfill={backfill} ==")
    return {"columns_added": added, "output_rows_deleted": deleted, "backfill": backfill}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("hostname")
    p.add_argument("catalog_id")
    p.add_argument("--schema", default="deriva-ml")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    catalog = get_catalog(a.hostname, a.catalog_id)
    pre = check_preconditions(catalog, a.schema)
    print(f"preconditions: {pre}")
    run_migration(catalog, a.schema, a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Then implement `step3_backfill_input_versions`'s resolution loop: for each
remaining `Dataset_Execution` row, fetch the `Execution`'s
`Execution_Metadata` rows, find the `configuration.json` asset, parse it
to a list of `DatasetSpec`, match the row's `Dataset` to a spec, look up
that dataset's `Dataset_Version` RID for `spec.version`, and write it to
the row's `Dataset_Version`. Count `null_config` when the asset is
missing/unreadable and `null_no_record` when no spec matches. Wrap each
row in try/except so one bad config never aborts the run.

- [ ] **Step 4: Run test to verify it passes**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/test_migrate_dataset_execution_version.py -v -m integration`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add scripts/migrate_dataset_execution_version.py tests/test_migrate_dataset_execution_version.py
git commit -m "feat(scripts): add Dataset_Execution version migration (idempotent, dry-run)"
```

---

## Task 3: Catalog deployment (MANUAL operational gate — not automated)

> This task is a manual, operator-run checkpoint. It mutates live
> catalogs and must be run by the user with valid credentials, dev before
> production. Do NOT script this into CI. Mark each box when the operator
> confirms completion.

- [ ] **Step 1: Dev dry-run**

Run (operator): `python scripts/migrate_dataset_execution_version.py dev.eye-ai.org eye-ai --dry-run`
Expected: preview shows column add + N output-row deletions, no changes made.

- [ ] **Step 2: Dev real run**

Run (operator): `python scripts/migrate_dataset_execution_version.py dev.eye-ai.org eye-ai`
Expected: column added, output rows deleted, input backfill summary printed.

- [ ] **Step 3: Validate on dev with the new tools (after Tasks 5–6 ship)**

Confirm `deriva_ml_find_dataset_executions` on `2-7P5P` returns its input
executions; a `split_dataset`-origin input row shows `dataset_version=null`;
no output rows remain in `Dataset_Execution`.

- [ ] **Step 4: Production dry-run then real run**

Run (operator): same script against the production eye-ai host (e.g.
`eye.rosci.org`), `--dry-run` first, then for real.

- [ ] **Step 5: Re-run safety check**

Re-run the script on dev; confirm it short-circuits (`[SKIP]` everywhere).

---

## Task 4: Library write path — stop writing output rows; record input version

**Files:**
- Modify: `src/deriva_ml/dataset/dataset.py:347-349` (remove the output `Dataset_Execution.insert`)
- Modify: `src/deriva_ml/execution/execution.py:1970-2014` (`add_input_dataset` gains `version`)
- Modify: `src/deriva_ml/execution/_helpers.py` (input-write helper sets `Dataset_Version`)
- Test: `tests/dataset/test_dataset_execution_write_path.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/dataset/test_dataset_execution_write_path.py`:

```python
"""create_dataset writes no Dataset_Execution row; add_input_dataset records version."""
import pytest


@pytest.mark.integration
def test_create_dataset_writes_no_dataset_execution_row(demo_ml, an_execution):
    """Output provenance lives only in Dataset_Version.Execution."""
    ds = an_execution.create_dataset(dataset_types=["Test"], description="out", version=(1, 0, 0))
    pb = demo_ml.pathBuilder()
    de = pb.schemas[demo_ml.ml_schema].Dataset_Execution
    rows = [r for r in de.entities().fetch() if r["Dataset"] == ds.dataset_rid]
    assert rows == [], "create_dataset must not write a Dataset_Execution row"
    # but the version row records the producer:
    assert demo_ml._producer_of_dataset(ds.dataset_rid) == an_execution.execution_rid


@pytest.mark.integration
def test_add_input_dataset_records_version(demo_ml, an_execution, an_existing_dataset):
    rid = an_existing_dataset.dataset_rid
    ver = an_existing_dataset.current_version
    an_execution.add_input_dataset(rid, version=ver)
    pb = demo_ml.pathBuilder()
    de = pb.schemas[demo_ml.ml_schema].Dataset_Execution
    row = next(r for r in de.entities().fetch()
               if r["Dataset"] == rid and r["Execution"] == an_execution.execution_rid)
    assert row["Dataset_Version"] is not None


@pytest.mark.integration
def test_add_input_dataset_without_version_is_null(demo_ml, an_execution, an_existing_dataset):
    an_execution.add_input_dataset(an_existing_dataset.dataset_rid)  # no version
    pb = demo_ml.pathBuilder()
    de = pb.schemas[demo_ml.ml_schema].Dataset_Execution
    row = next(r for r in de.entities().fetch()
               if r["Dataset"] == an_existing_dataset.dataset_rid)
    assert row["Dataset_Version"] is None
```

(`demo_ml`, `an_execution`, `an_existing_dataset` fixtures: build from
`tests/catalog_manager.py` — a connected `ml`, a created execution, and a
pre-existing dataset. Mirror fixtures in `tests/dataset/conftest.py`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_dataset_execution_write_path.py -v -m integration`
Expected: FAIL — `create_dataset` still writes the row; `add_input_dataset` has no `version` param.

- [ ] **Step 3: Remove the output-row insert in create_dataset**

In `src/deriva_ml/dataset/dataset.py`, delete lines 347-349:

```python
        pb.schemas[ml_instance.model.ml_schema].Dataset_Execution.insert(
            [{"Dataset": dataset_rid, "Execution": execution_rid}]
        )
```

(Leave the surrounding dataset-insert and `Dataset(...)` construction
intact. The `Dataset_Version` row written by `_insert_dataset_versions`
later in the same method already records the producer.)

- [ ] **Step 4: Add `version` to add_input_dataset**

In `src/deriva_ml/execution/execution.py`, change the signature and the
insert (around 1970/2014):

```python
    def add_input_dataset(self, dataset_rid: RID, version: "DatasetVersion | str | None" = None) -> None:
        # ... existing docstring; add a line documenting `version` (optional;
        # the consumed version, recorded on the Dataset_Execution row) ...
        if self._dry_run:
            return
        schema_path = self._ml_object.pathBuilder().schemas[self._ml_object.ml_schema]
        dataset_exec = schema_path.Dataset_Execution
        already_linked = {
            row["Dataset"]
            for row in dataset_exec.filter(dataset_exec.Execution == self.execution_rid).entities().fetch()
        }
        if dataset_rid in already_linked:
            return
        version_rid = None
        if version is not None:
            version_rid = self._ml_object._version_rid(dataset_rid, version)  # see Step 5
        dataset_exec.insert([{
            "Dataset": dataset_rid,
            "Execution": self.execution_rid,
            "Dataset_Version": version_rid,
        }])
```

- [ ] **Step 5: Add a `_version_rid` helper (dataset rid + version -> Dataset_Version RID)**

In `src/deriva_ml/core/mixins/execution.py` (near `_producer_of_dataset`):

```python
    def _version_rid(self, dataset_rid: RID, version) -> RID | None:
        """RID of the Dataset_Version row for (dataset_rid, version), or None."""
        pb = self.pathBuilder()
        vp = pb.schemas[self.ml_schema].tables["Dataset_Version"]
        want = str(version)
        for row in vp.filter(vp.Dataset == dataset_rid).entities().fetch():
            if (row.get("Version") or "") == want:
                return row["RID"]
        return None
```

Also have the config-input writer (where `ExecutionConfiguration.datasets`
materialization inserts the `Dataset_Execution` row — `execution.py:658`)
pass the resolved `Dataset_Version` RID the same way.

- [ ] **Step 6: Run tests to verify they pass**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_dataset_execution_write_path.py -v -m integration`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/dataset/dataset.py src/deriva_ml/execution/execution.py src/deriva_ml/core/mixins/execution.py tests/dataset/test_dataset_execution_write_path.py
git commit -m "feat(execution): record consumed version on input edge; stop writing output rows"
```

---

## Task 5: Library read path — simplify `list_input_datasets`; extend `find_executions`

**Files:**
- Modify: `src/deriva_ml/execution/_helpers.py:203-247` (`list_input_datasets`)
- Modify: `src/deriva_ml/core/mixins/execution.py:559-653` (`find_executions`)
- Test: `tests/execution/test_find_executions_by_dataset.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/execution/test_find_executions_by_dataset.py`:

```python
"""find_executions dataset filter: input/output/any, version pin via DatasetSpec."""
import pytest
from deriva_ml.dataset.aux_classes import DatasetSpec


@pytest.mark.integration
def test_find_input_executions(demo_ml, dataset_with_input_exec):
    ds_rid, input_exec = dataset_with_input_exec
    rids = {e.execution_rid for e in demo_ml.find_executions(dataset=ds_rid, dataset_role="input")}
    assert input_exec in rids


@pytest.mark.integration
def test_find_output_executions(demo_ml, dataset_with_output_exec):
    ds_rid, producer = dataset_with_output_exec
    rids = {e.execution_rid for e in demo_ml.find_executions(dataset=ds_rid, dataset_role="output")}
    assert producer in rids


@pytest.mark.integration
def test_find_any_is_union(demo_ml, dataset_with_both):
    ds_rid, producer, consumer = dataset_with_both
    rids = {e.execution_rid for e in demo_ml.find_executions(dataset=ds_rid, dataset_role="any")}
    assert {producer, consumer} <= rids


@pytest.mark.integration
def test_version_pin_via_datasetspec(demo_ml, dataset_with_versioned_input):
    ds_rid, version, exec_rid = dataset_with_versioned_input
    spec = DatasetSpec(rid=ds_rid, version=version)
    rids = {e.execution_rid for e in demo_ml.find_executions(dataset=spec, dataset_role="input")}
    assert exec_rid in rids


def test_role_without_dataset_raises(demo_ml):
    with pytest.raises(ValueError):
        list(demo_ml.find_executions(dataset_role="input"))


@pytest.mark.integration
def test_list_input_datasets_excludes_produced(demo_ml, an_execution):
    """Regression: a dataset the execution PRODUCED is not an input."""
    produced = an_execution.create_dataset(dataset_types=["Test"], description="o", version=(1, 0, 0))
    inputs = {d.dataset_rid for d in an_execution.list_input_datasets()}
    assert produced.dataset_rid not in inputs


@pytest.mark.integration
def test_dataset_list_executions_is_input_only(demo_ml, an_execution):
    """Documented behavior change: Dataset.list_executions no longer returns the producer."""
    produced = an_execution.create_dataset(dataset_types=["Test"], description="o", version=(1, 0, 0))
    exec_rids = {e.execution_rid for e in produced.list_executions()}
    assert an_execution.execution_rid not in exec_rids, (
        "Dataset.list_executions should be input-only (producer lives in Dataset_Version)"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_find_executions_by_dataset.py -v`
Expected: FAIL — `find_executions` has no `dataset`/`dataset_role` params.

- [ ] **Step 3: Simplify list_input_datasets (input-only table)**

In `src/deriva_ml/execution/_helpers.py`, replace the body of
`list_input_datasets` (203-247) so it no longer subtracts produced
datasets — every `Dataset_Execution` row is now an input:

```python
def list_input_datasets(*, ml_instance, execution_rid) -> list:
    pb = ml_instance.pathBuilder()
    dataset_exec = pb.schemas[ml_instance.ml_schema].Dataset_Execution
    records = list(
        dataset_exec.filter(dataset_exec.Execution == execution_rid).entities().fetch()
    )
    return [ml_instance.lookup_dataset(r["Dataset"]) for r in records if r.get("Dataset")]
```

(Keep the existing docstring's intent; drop the `_producer_of_dataset`
exclusion paragraph.)

- [ ] **Step 4: Extend find_executions with the dataset filter**

In `src/deriva_ml/core/mixins/execution.py`, add params and filter logic.
Add to the signature (559-565): `dataset=None, dataset_role="any"`.
After the existing workflow/status filters build `filtered_path`, and
before the fetch loop, insert:

```python
        # Dataset filter (resolve to an allowed Execution RID set, then intersect).
        from deriva_ml.dataset.aux_classes import DatasetSpec
        dataset_exec_rids: set[str] | None = None
        if dataset is not None:
            if isinstance(dataset, DatasetSpec):
                ds_rid, ds_version = dataset.rid, str(dataset.version)
            else:
                ds_rid, ds_version = dataset, None
            input_rids, output_rids = set(), set()
            pb2 = self.pathBuilder()
            if dataset_role in ("input", "any"):
                de = pb2.schemas[self.ml_schema].Dataset_Execution
                for r in de.filter(de.Dataset == ds_rid).entities().fetch():
                    if ds_version is None or self._version_label(r.get("Dataset_Version")) == ds_version:
                        input_rids.add(r["Execution"])
            if dataset_role in ("output", "any"):
                dv = pb2.schemas[self.ml_schema].tables["Dataset_Version"]
                for r in dv.filter(dv.Dataset == ds_rid).entities().fetch():
                    if r.get("Execution") and (ds_version is None or (r.get("Version") or "") == ds_version):
                        output_rids.add(r["Execution"])
            dataset_exec_rids = (
                input_rids if dataset_role == "input"
                else output_rids if dataset_role == "output"
                else input_rids | output_rids
            )
        elif dataset_role != "any":
            raise ValueError("dataset_role requires a dataset argument")
```

Then in the existing `for exec_record in entity_set.fetch():` loop, add an
early skip:

```python
            if dataset_exec_rids is not None and exec_record["RID"] not in dataset_exec_rids:
                continue
```

Add a small `_version_label(version_rid)` helper next to `_version_rid`
that maps a `Dataset_Version` RID back to its `Version` string (single
fetch), used to honor the version pin on input rows:

```python
    def _version_label(self, version_rid) -> str | None:
        if not version_rid:
            return None
        pb = self.pathBuilder()
        vp = pb.schemas[self.ml_schema].tables["Dataset_Version"]
        rows = list(vp.filter(vp.RID == version_rid).entities().fetch())
        return (rows[0].get("Version") if rows else None)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_find_executions_by_dataset.py -v`
Expected: PASS.

- [ ] **Step 6: Run the execution + dataset suites for regressions**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution tests/dataset -q`
Expected: PASS (notably `list_input_datasets` callers and `find_executions` callers unbroken).

- [ ] **Step 7: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml
git add src/deriva_ml/execution/_helpers.py src/deriva_ml/core/mixins/execution.py tests/execution/test_find_executions_by_dataset.py
git commit -m "feat(execution): find_executions(dataset=RID|DatasetSpec, dataset_role=...)"
```

---

## Task 6: Plugin — response model + two tool surfaces

**Files:**
- Modify: `src/deriva_ml_mcp_plugin/_response_models.py` (add `DatasetExecutionSummary`)
- Modify: `src/deriva_ml_mcp_plugin/tools/execution/read.py` (extend `_list_executions_impl`, `deriva_ml_list_executions`; add `deriva_ml_find_dataset_executions`)
- Test: `deriva-ml-mcp-plugin/tests/test_find_dataset_executions.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `deriva-ml-mcp-plugin/tests/test_find_dataset_executions.py`:

```python
"""deriva_ml_find_dataset_executions + list_executions(dataset=) tool surface."""
import json
import pytest
from unittest.mock import MagicMock


def _exec(rid, wf="1-WF"):
    e = MagicMock()
    e.execution_rid, e.workflow_rid = rid, wf
    e.status = None
    e.description = e.start_time = e.stop_time = e.duration = None
    return e


@pytest.mark.anyio
async def test_find_dataset_executions_returns_role_and_version(execution_ctx, capturing_mcp, mock_ml):
    mock_ml.find_executions.return_value = [_exec("1-EXEC")]
    # role/version come from the impl tagging; mock the dataset-edge lookups it uses.
    res = await capturing_mcp.call_tool(
        "deriva_ml_find_dataset_executions",
        {"hostname": "h", "catalog_id": "c", "dataset_rid": "2-DS", "dataset_role": "input"},
    )
    payload = json.loads(res)
    assert payload["executions"][0]["rid"] == "1-EXEC"
    assert "dataset_role" in payload["executions"][0]


@pytest.mark.anyio
async def test_list_executions_without_dataset_has_no_dataset_fields(execution_ctx, capturing_mcp, mock_ml):
    mock_ml.find_executions.return_value = [_exec("1-EXEC")]
    res = await capturing_mcp.call_tool(
        "deriva_ml_list_executions", {"hostname": "h", "catalog_id": "c"},
    )
    payload = json.loads(res)
    assert "dataset_role" not in payload["executions"][0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run pytest tests/test_find_dataset_executions.py -v`
Expected: FAIL — tool `deriva_ml_find_dataset_executions` not registered.

- [ ] **Step 3: Add the response model**

In `src/deriva_ml_mcp_plugin/_response_models.py`, add:

```python
class DatasetExecutionSummary(ExecutionSummary):
    """ExecutionSummary plus the dataset-edge role/version for dataset-scoped queries."""
    dataset_role: str  # "input" | "output"
    dataset_version: str | None = None
```

- [ ] **Step 4: Extend `_list_executions_impl` to accept + forward the dataset filter**

In `tools/execution/read.py`, add `dataset=None, dataset_role="any"` to
`_list_executions_impl`, forward them into `ml.find_executions(...)`, and
when `dataset` is set, wrap each summary as `DatasetExecutionSummary`
(role from which side matched; version from the association row already
fetched — no extra fetch). Keep returning plain `ExecutionSummary` when
`dataset` is None.

- [ ] **Step 5: Extend `deriva_ml_list_executions` and add the intent tool**

Add `dataset`, `dataset_role`, `dataset_version` params to
`deriva_ml_list_executions`; when `dataset_version` is set, construct
`DatasetSpec(rid=dataset, version=dataset_version)` and pass as `dataset`,
else pass the bare RID string. Then add, mirroring
`deriva_ml_find_workflow_executions`:

```python
    @ctx.tool(mutates=False)
    async def deriva_ml_find_dataset_executions(
        hostname: str, catalog_id: str, dataset_rid: str,
        dataset_role: str = "any", dataset_version: str | None = None,
        status: str | None = None, limit: int = 100,
        after_rid: str | None = None, preflight_count: bool = False, sort: bool = False,
    ) -> str:
        """Find executions that used a dataset (input, output, or any).

        Distinct from ``deriva_ml_list_executions(dataset=...)`` to surface
        the dataset-centric query as a first-class tool. Role is sourced
        relationally (input = Dataset_Execution; output = Dataset_Version
        authorship), never from configuration files.
        """
        try:
            with deriva_call():
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                from deriva_ml.dataset.aux_classes import DatasetSpec
                ds_arg = DatasetSpec(rid=dataset_rid, version=dataset_version) if dataset_version else dataset_rid
                # preflight + page via _list_executions_impl(dataset=ds_arg, dataset_role=dataset_role, ...)
                ...
            return payload.model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(exc, operation="find_dataset_executions",
                                   hostname=hostname, catalog_id=catalog_id, audit=False)
```

(Fill the preflight/page body by copying the exact structure of
`deriva_ml_find_workflow_executions` in the same file, substituting the
dataset args.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run pytest tests/test_find_dataset_executions.py -v`
Expected: PASS.

- [ ] **Step 7: Run the plugin execution suite for regressions**

Run: `uv run pytest tests/test_execution.py -q`
Expected: PASS (existing `list_executions` callers see unchanged shape when `dataset` absent).

- [ ] **Step 8: Update the getting-started guide's tool menu**

In `src/deriva_ml_mcp_plugin/` (the getting-started prompt/guide text;
grep for the execution-tool list), add `deriva_ml_find_dataset_executions`
to the `execution` domain's verb list and bump the tool count.

- [ ] **Step 9: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin
git add src/deriva_ml_mcp_plugin/_response_models.py src/deriva_ml_mcp_plugin/tools/execution/read.py tests/test_find_dataset_executions.py src/deriva_ml_mcp_plugin/prompts.py
git commit -m "feat(tools): add deriva_ml_find_dataset_executions + dataset filter on list_executions"
```

---

## Final verification

- [ ] Library suite: `cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/schema tests/execution tests/dataset -q`
- [ ] Plugin suite: `cd /Users/carl/GitHub/DerivaML/deriva-ml-mcp-plugin && uv run pytest -q`
- [ ] Confirm Task 3 (catalog deployment) boxes are checked by the operator for both dev and production.
