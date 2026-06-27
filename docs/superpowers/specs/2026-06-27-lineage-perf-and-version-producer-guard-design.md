# Lineage perf + version-producer guard — design

**Date:** 2026-06-27
**Status:** Approved
**Component:** `deriva_ml.execution._helpers` (`list_input_datasets_with_versions`)
and `deriva_ml.core.mixins.execution` (`_walk_node`)
**Relates to:** the consumed-version fix (merged `b0d5d6cb`), ADR-0001. Closes the
two findings from the second `/codex` pass (cifar-example tacit-knowledge `tk-022`).

## Problem

A second `/codex` review (one diff-gate pass + one test-completeness consult) of
the merged consumed-version lineage fix surfaced two issues our SDD review chain
(and the first codex pass) missed:

### Gap 1 — performance: full `Dataset_Version` table scan per lineage node

`list_input_datasets_with_versions` resolves the `Dataset_Execution.Dataset_Version`
**FK RID** to a version string by fetching the **entire** `Dataset_Version` table
and building a `{RID: Version}` map. But `_walk_node` calls this helper for **every
walked execution**, so a deep `lookup_lineage` walk is
O(walked-executions × total-dataset-versions). On a catalog with many dataset
versions this is slow and can hit ERMrest response/time limits. (The single-fetch
map was accepted as "simplest and correct"; it is correct but not scalable — scale
is the documented "reason not to" the original plan named.)

### Gap 2 — correctness: self-parent guard misses the version-producer (false cycle)

In `_walk_node`'s consumed-dataset loop (`execution.py:1629-1638`), the member-
producers are added as `member_producers - {execution_rid}` (the self-parent guard
from the consumed-version fix), but the dataset's **version-producer** —
`_producer_of_dataset(ds, version=...)` at line 1631 — is added with **no**
`execution_rid` exclusion.

Confirmed by code-trace: if an execution `E` both **consumed** dataset `D` and
**produced the consumed version** of `D` (a real pattern: consume `D`, add members,
re-version `D` in one run), `_producer_of_dataset` returns `E`; the recursion
re-enters `_walk_node(execution_rid=E)`, finds `E` in `in_progress`
(`execution.py:1554`), and sets `flags["cycle_detected"] = True` — a **false cycle**.
The member guard prevents exactly this for member-producers; the producer path is
unguarded. It went untested because no test made the consuming execution also the
version-producer.

## Goal

Make the consumed-version-RID resolution **bounded** (fetch only the version rows
actually referenced by the input edges), and extend the self-parent guard so an
execution that produced the consumed version of a dataset it also consumed is not
listed as its own parent (no false cycle). Close the test-coverage gaps the audit
named so these classes can't silently regress.

## Design

### Component 1 — bounded version fetch in `list_input_datasets_with_versions`

Rewrite the helper body (`src/deriva_ml/execution/_helpers.py`) to fetch only the
consumed `Dataset_Version` RIDs:

1. Fetch the `Dataset_Execution` rows for `execution_rid` (as today); keep only
   rows with a truthy `Dataset` → `records`.
2. **Early return `[]`** if `records` is empty (no input edges → never touch the
   version table).
3. Collect `wanted_rids = {r["Dataset_Version"] for r in records if r.get("Dataset_Version")}`
   (the distinct, non-null consumed-version RIDs).
4. Build `rid_to_version`:
   - If `wanted_rids` is empty (every edge unpinned), use an **empty map** (skip
     the fetch entirely).
   - Else fetch `Dataset_Version` filtered to those RIDs, **chunked** at
     `_MEMBER_PRODUCER_CHUNK` (500) via the `.in_()` predicate already used in the
     codebase (`_distinct_member_output_producers` in `execution.py` and
     `bag_builder.py:1108`). Build `{row["RID"]: row.get("Version")}` across chunks.
5. Map each edge: `consumed_version = rid_to_version.get(version_rid) if version_rid else None`.
   A `version_rid` not found in the map (deleted/missing version row) → `None`.

Chunk-size source: `_MEMBER_PRODUCER_CHUNK` is currently defined in
`execution.py`. To avoid a `_helpers.py` → `execution.py` import cycle, define a
local module constant `_VERSION_RID_CHUNK = 500` in `_helpers.py` (with a comment
pointing at the shared 500 convention). The plan decides the exact mechanism;
either is acceptable as long as it is 500 and not a per-row fetch.

The function signature and the docstring's contract are unchanged
(`list[tuple[Dataset, str | None]]`, version STRING or None). This is a pure
internal-efficiency rewrite plus the missing-RID fallback (which Gap-2-adjacent
tests will now cover).

### Component 2 — extend the self-parent guard to the version-producer

In `src/deriva_ml/core/mixins/execution.py`, `_walk_node` consumed loop, change:

```python
                producer = self._producer_of_dataset(ds.dataset_rid, version=consumed_version)
                if producer:
                    parent_rids.add(producer)
```

to:

```python
                producer = self._producer_of_dataset(ds.dataset_rid, version=consumed_version)
                # Never the execution we are currently expanding: if E produced
                # the consumed version of a dataset it also consumed, listing E
                # as its own parent would re-enter `in_progress` and flag a false
                # cycle (the same reason the member-producers below subtract E).
                if producer and producer != execution_rid:
                    parent_rids.add(producer)
```

Update the existing member-producer comment if helpful so the two guards read as a
matched pair. No other change to `_walk_node`.

### No public-model change

`lineage.py` unchanged. `list_input_datasets`, `_producer_of_dataset`, the root
path, the asset loop, `extra_parent_rids`, and the recursion are all unchanged.

## Testing

### Helper-level (`tests/execution/test_input_datasets_with_versions.py`)

- **Missing-FK-RID → None:** an edge has `"Dataset_Version": "VR_MISSING"`; the
  `Dataset_Version` rows omit it; assert the consumed version is `None`. (Gap-2
  adjacent: the fallback path.)
- **Mixed pinned + unpinned in one execution:** two `Dataset_Execution` rows, one
  with `VR1`, one with no `Dataset_Version` key; assert both pairs survive with
  the correct versions (`"1.0.0"` and `None`).
- **Empty input list:** no `Dataset_Execution` rows → assert `[]` AND assert the
  `Dataset_Version` table was NOT fetched (early-return). Spy on the version-table
  fetch.
- **Bounded fetch — only wanted RIDs:** with two pinned edges, assert the version
  fetch was issued with an `.in_()` over exactly the two wanted RIDs (or, if the
  mock can't introspect the predicate, assert the fetch happened once over a
  bounded set, NOT a full-table `.entities().fetch()` with no filter). The
  assertion that matters: the helper does NOT do an unfiltered full-table fetch.

### Walk-level (`tests/execution/test_lookup_lineage_unit.py`, via `_FakeML`)

- **Self-parent via version-producer:** `EXSC` consumes `D`;
  `_producer_of_dataset(D, version=...) == EXSC` (script via `set_versioned_producer`
  or the unversioned producer map so the consuming exec is the dataset's producer);
  assert `EXSC` is NOT among its own parents and `result.cycle_detected is False`.
  This is the test that would have caught the Gap-2 bug.
- **Multiple consumed datasets, different versions:** an execution consumes
  `D1@1.0.0` (producer `EXV1`) and `D2@2.0.0` (producer `EXV2`); assert both
  version-specific producers appear as parents AND both `consumed_datasets[].version`
  summaries are correct (`"1.0.0"`, `"2.0.0"`).

### Seam-level (`tests/execution/test_input_datasets_with_versions.py` or the unit file)

- **`_input_dataset_pairs` real wiring:** monkeypatch
  `deriva_ml.execution._helpers.list_input_datasets_with_versions`, call
  `ExecutionMixin._input_dataset_pairs(self_stub, "2-EXAA")`, and assert it
  forwards `ml_instance=self` and `execution_rid="2-EXAA"`. Today only the live
  test exercises the real seam; this guards against import/argument drift offline.

### No-regression + live

- The full `tests/execution/test_lookup_lineage_unit.py`,
  `test_producers_of_dataset_members.py`, and `test_input_datasets_with_versions.py`
  stay green.
- The existing live test (`test_lookup_lineage_reflects_consumed_version_not_latest`,
  `DERIVA_HOST`-gated) still passes — it exercises the bounded fetch end-to-end on a
  real catalog (the version it consumes IS in the wanted set, so the bounded fetch
  must still resolve it).

## Out of scope

- The remaining codex P3 items (self-parent guard interaction with `depth=0` /
  `max_executions` — low risk; a live self-parent test — keep as unit). May be
  added opportunistically but are not required.
- Any change to `list_input_datasets`, `_producer_of_dataset`'s contract, the root
  path, or the public lineage models.
