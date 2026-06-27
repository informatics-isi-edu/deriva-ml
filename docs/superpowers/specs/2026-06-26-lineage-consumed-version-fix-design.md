# Lineage consumed-version + self-parent fix — design

**Date:** 2026-06-26
**Status:** Approved
**Component:** `deriva_ml.core.mixins.execution` (`lookup_lineage` / `_walk_node` /
`_producer_of_dataset`) and `deriva_ml.execution._helpers`
**Relates to:** the 2026-06-26 member-asset-traversal change (merged `0a83b22b`),
ADR-0001 (lineage walks data-flow). Closes two `/codex`-found gaps (cifar-example
tacit-knowledge `tk-020`).

## Problem

An independent `/codex` review of the merged member-asset traversal found two
real correctness gaps in the lineage walk (no P1; both P2):

### Gap 1 — mid-walk uses CURRENT membership, not the CONSUMED version

When an execution `E` consumed dataset `D`, `_walk_node` walks `D`'s members and
producer from `D`'s **current** state. `list_input_datasets()` discards the
`Dataset_Execution.Dataset_Version` pin (`_helpers.py`), so:

- `_producer_of_dataset(ds.dataset_rid)` returns the producer of the **latest**
  version (`execution.py`), and
- `_producers_of_dataset_members(ds.dataset_rid)` lists the **current** members.

If `D` gained assets/versions after `E` consumed it, `lookup_lineage` reports
ancestors that were not actually inputs at consumption time. The `consumed_datasets`
summary also reports `ds.current_version` rather than the consumed version.

### Gap 2 — mid-walk has no self-parent guard → false cycle

If `D` (as walked) contains assets produced by the very execution `E` that
consumed it, `E` lands in its own `parent_rids`; the recursion hits `in_progress`
and reports a **false `cycle_detected` / self-parent**. The ROOT path already
guards this (`member_producers - {producer_rid}`, from the member-asset-traversal
change), but the **mid-walk path has no equivalent guard**.

Note on provenance: Gap 1 is partly a **pre-existing** limitation — `lookup_lineage`
already walks current dataset versions (its docstring notes historical-version
walking is a future enhancement); the member-asset change merely extended that
same current-version behavior to member assets. Gap 2's missing mid-walk guard
is introduced by the member-asset change.

## Goal

Make the mid-walk consumed-dataset expansion **version-faithful** (reflect the
version actually consumed, per `Dataset_Execution.Dataset_Version`) and add the
**self-parent guard** so an execution that both consumed a dataset and produced
some of its members is not reported as its own parent. Do this **without changing
the `list_input_datasets()` return contract** relied on by `Execution`,
`ExecutionRecord`, `provenance_enforcement`, and `split.py`.

## Design

### Component 1 — new shared helper `list_input_datasets_with_versions`

In `src/deriva_ml/execution/_helpers.py`, beside `list_input_datasets`:

```python
def list_input_datasets_with_versions(
    *, ml_instance: Any, execution_rid: str
) -> list[tuple[Any, str | None]]:
    """Input datasets of an execution paired with the consumed version.

    Like :func:`list_input_datasets`, but also returns the
    ``Dataset_Execution.Dataset_Version`` recorded on each input edge — the
    version of the dataset that was actually consumed. Returns a list of
    ``(Dataset, consumed_version)`` tuples; ``consumed_version`` is ``None``
    when the edge has no version pin.

    The existing :func:`list_input_datasets` return contract (``list[Dataset]``)
    is intentionally left unchanged; lineage is the only caller that needs the
    consumed version.
    """
```

Behavior: same `Dataset_Execution` filter as `list_input_datasets`
(`dataset_exec.Execution == execution_rid`), but project `Dataset_Version` too,
returning `(lookup_dataset(row["Dataset"]), row.get("Dataset_Version"))`.

`list_input_datasets` is untouched. The four existing callers are unaffected.

### Component 2 — `_producer_of_dataset` gains optional `version`

In `src/deriva_ml/core/mixins/execution.py`:

```python
def _producer_of_dataset(self, dataset_rid: RID, version: Any | None = None) -> RID | None:
```

- `version is None` (default): unchanged — return the `Execution` of the
  **latest** `Dataset_Version` row (current behavior; existing callers and the
  root path keep working identically).
- `version` given: return the `Execution` recorded on **that specific version's**
  `Dataset_Version` row (the execution that produced the consumed version).
  Resolve the version row via the existing `_version_rid(dataset_rid, version)`
  (or an equivalent filtered fetch on `Dataset_Version` matching the version
  string), and read its `Execution`. Return `None` if no such version row exists.

### Component 3 — `_walk_node` consumed-dataset loop, version-faithful + guarded

Replace the `for ds in record.list_input_datasets():` loop body. Iterate the
new pairs and use the consumed version everywhere:

```python
from deriva_ml.execution._helpers import list_input_datasets_with_versions
# execution_rid is the RID of the node being expanded (already a _walk_node arg);
# self is the bound ExecutionMixin (the ml_instance the helper expects).
for ds, consumed_version in list_input_datasets_with_versions(
    ml_instance=self, execution_rid=execution_rid
):
    version_str = consumed_version
    if version_str is None:
        try:
            version_str = str(ds.current_version)
        except Exception:
            version_str = None
    consumed_datasets.append(
        DatasetSummary(rid=ds.dataset_rid, description=ds.description or None, version=version_str)
    )
    producer = self._producer_of_dataset(ds.dataset_rid, version=consumed_version)
    if producer:
        parent_rids.add(producer)
    # Member-producers of the CONSUMED version; never the execution we are
    # currently expanding (an execution that both consumed D and produced some
    # of D's members must not become its own parent — the mid-walk analogue of
    # the root path's version-producer subtraction).
    member_producers = self._producers_of_dataset_members(ds.dataset_rid, version=consumed_version)
    parent_rids |= member_producers - {execution_rid}
```

Where the pairs come from: `_walk_node` works with an `ExecutionRecord`
(`record`). Add a tiny private accessor (or call the helper directly) that yields
`(Dataset, consumed_version)` for `record.execution_rid` via
`list_input_datasets_with_versions`. Keep the `consumed_datasets` summary's
`version` field reporting the **consumed** version (falling back to
`current_version` only when the edge has no pin, preserving today's output for
unpinned edges).

`_producers_of_dataset_members` already accepts `version=` and forwards it to
`list_dataset_members(version=...)` — no change needed there beyond passing the
consumed version.

### Component 4 — root path unchanged

`lookup_lineage`'s root-dataset seeding stays as-is. A root dataset has no
`Dataset_Execution` consumed-input edge, so there is no "consumed version" for it
— it is resolved at current version, consistent with the existing
`_classify_rid` / `_producer_of_dataset(rid)` root behavior. The root path's
existing `member_producers - {producer_rid}` subtraction is retained.

### No public-model change

`LineageNode` / `LineageResult` / `RootDescriptor` are unchanged. The
`consumed_datasets[].version` now reflects the consumed version (a correctness
improvement in an existing field), not a shape change.

## Data flow (after the fix)

```
E consumed D@v1   (Dataset_Execution: Dataset=D, Dataset_Version=v1)
  walk E:
    consumed_datasets: D @ v1                         (was: D @ current)
    parent = _producer_of_dataset(D, version=v1)      (was: producer of latest)
    member_producers = _producers_of_dataset_members(D, version=v1)  (was: current members)
                       minus {E}                       (self-parent guard; NEW)
```

## Testing

### Offline unit (extend `_FakeML` in `tests/execution/test_lookup_lineage_unit.py`)

- **Self-parent / false-cycle:** `E` consumes `D`; `D`'s (walked) members include
  an asset produced by `E`. Assert `E` is NOT among its own node's parents and
  `result.cycle_detected is False`.
- **Consumed-version producer:** `E` consumed `D@v1` produced by `X`; `D@v2`
  (latest) produced by `Y`. Walking through `E` surfaces `X`, not `Y`.
- **Version-aware member producers:** `D@v1` members produced by `P1`; `D@v2`
  adds members produced by `P2`. Walking `E` (consumed v1) surfaces `P1`, not `P2`.
- **No-regression (`version=None`):** all existing lineage unit tests still pass;
  `_producer_of_dataset(rid)` with no version returns the latest-version producer;
  the root path is unchanged. The `_FakeML` `list_input_datasets`-style stub gains
  a versions variant defaulting `consumed_version=None`, so existing scripted
  inputs behave exactly as before.
- **Helper unit:** `list_input_datasets_with_versions` returns the right
  `(Dataset, version)` pairs and `None` when the edge has no version pin (mock the
  `Dataset_Execution` fetch).

### Live (gated on `DERIVA_HOST`, in `tests/execution/test_lookup_lineage_live.py`)

- **Versioned-mutation scenario:** build `E` consuming `D@v1`; then mutate `D` to
  `v2` adding members produced by a different execution; assert
  `lookup_lineage` walked through `E` reflects v1's producers/members, not v2's.
  This is the real regression guard for Gap 1 (no offline mock fully proves the
  version-snapshot membership read).

## Out of scope

- Making `lookup_lineage` accept a `version=` to walk a **historical version of
  the root** (the root stays current-version; only consumed-input edges carry a
  consumed version). Tracked as a possible future enhancement.
- The P3 items from the codex audit (hard-coded `"Execution"` FK name in the
  asset-producer helpers; non-Dataset roots now surfacing member-producers deeper
  — the latter is the intended improvement, not a defect). The FK-name item is a
  pre-existing assumption in `_producer_of_asset` and is left for a separate
  change if a non-conventional schema ever needs it.
- Any change to `list_input_datasets` (the existing `list[Dataset]` contract) or
  its four callers.
