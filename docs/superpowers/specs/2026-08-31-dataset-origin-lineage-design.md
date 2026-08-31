# Dataset origin and version-attribution trace in lineage

**Date:** 2026-08-31
**Issue:** #367 — *Dataset producing_execution is last-writer-wins*
**Status:** Approved (design review with Carl, 2026-08-31)

## Problem

`lookup_lineage()` on a dataset reports the author of the **latest**
`Dataset_Version` row as `producing_execution`. The answer to "how did
this dataset come to exist?" is therefore overwritten by whatever
touched the dataset most recently — on the live eye-ai catalog, a
diagnosis/severity data migration (`6-FQ28`) is reported as the
producer of the LAC train/validation/test splits it merely modified.

The value is not a data bug: `Dataset_Version.Execution` means
*"the execution that authored this version"* per the provenance
contract (`docs/reference/provenance-contract.md`, authorship-canonical
model), and the 92 % sentinel population is the contract's documented
adoption backfill. The defect is in **lineage resolution and
presentation**: the unversioned root path answers "who last touched
it?" while presenting it as "who created it?".

The issue's semantics question is settled by the existing contract:
reading 1 (version authorship) is what the column means. This design
implements the presentation side of reading 2 — distinguishing
structural origin from incidental version bumps — **without any schema
change**, by deriving origin from data already recorded.

## Decision summary

| Question | Decision |
|---|---|
| Meaning of `Dataset_Version.Execution` | Unchanged: author of that version (contract) |
| Root `producing_execution` for a dataset | **Origin** = author of the *earliest* version row |
| Version-pinned resolution (consumed datasets) | Unchanged — already correct |
| Unknown origin (sentinel / None) | Reported explicitly via `origin_recorded=False`; never silently replaced by a later author |
| Full attribution history | New `version_history` trace on the **root descriptor only** |
| Schema change / backfill | None |

## Why earliest-row author is the origin

`create_dataset` writes the initial `Dataset_Version` row (v0.1.0,
table default) carrying the creating execution — so for every
post-contract dataset, **earliest-row author = structural origin by
construction**. Pre-contract datasets (e.g. `2-277G`, whose v0.1.0 row
already carries the sentinel `6-0B3J`) resolve to the
unknown-provenance sentinel — the honest answer: the true creator was
never recorded and cannot be recovered retroactively.

The version rows are already fetched in full by
`_producer_of_dataset` (which then keeps only `max(rows)`), so both the
origin and the trace come from the **same single query already being
made** — zero added catalog round-trips.

## Result shape (additive)

In `src/deriva_ml/execution/lineage.py`:

```python
class VersionAttribution(BaseModel):
    """One entry in a dataset's version-attribution trace."""
    model_config = ConfigDict(extra="forbid")

    version: str                          # PEP 440 label, e.g. "4.13.0"
    execution: ExecutionSummary | None    # author; None if row has no Execution
    description: str | None               # the version's release notes


class RootDescriptor(BaseModel):
    ...existing fields...
    producing_execution: ExecutionSummary | None   # datasets: the ORIGIN
    origin_recorded: bool = True
    version_history: list[VersionAttribution] = Field(default_factory=list)
```

- `producing_execution` carries whatever the earliest row says —
  **including the sentinel** (it is real catalog data). `origin_recorded`
  carries the interpretation: `False` when the origin is the sentinel,
  the row has no `Execution`, or the dataset has no version rows.
  Truth and judgment are separate fields.
- `version_history` is ordered **earliest → latest** and is populated
  only when the root is a Dataset. Non-dataset roots (Asset, Feature,
  Execution) keep an empty list and `origin_recorded=True` (the field
  makes no claim there).
- Additive fields flow through `.model_dump()` unchanged, so the
  deriva-ml-mcp serialization boundary is backward-compatible.

### Root-only trace (judgment call, approved)

Datasets appear in a lineage result in two roles: as the **root**
(the artifact asked about) and as **consumed mentions** inside walked
`LineageNode.consumed_datasets`. The trace attaches to the root only:

1. A consumed mention is version-pinned, and the pin *is* the
   provenance of that consumption — "who authored v4.11.0" is already
   answered correctly by the version-scoped path. The dataset's whole
   life story is context about the dataset, not about that consumption
   event.
2. A popular dataset appears in every walk that trained on it, and
   possibly several times within one walk. Trace-on-every-mention
   serializes the same N-entry history once per mention across the MCP
   boundary.
3. The escape hatch is one cheap call:
   `lookup_lineage(dataset_rid, depth=0)` returns any dataset's full
   trace with no graph walk.

## Resolution mechanics

`_producer_of_dataset(rid, version=None)` (`core/mixins/execution.py`):

- The unversioned path selects the **earliest** row instead of the
  latest. Its only caller is the dataset-root descriptor path of
  `lookup_lineage` (verified: one call site); consumed-dataset
  resolution always passes `version=` and is untouched.
- **Ordering trap:** the current `_key` collapses unparseable versions
  to `(0,)` — harmless for `max`, but under `min` a dev row like
  `0.4.0.post1.dev3` would sort before `0.1.0` and steal the origin.
  Fix: parse with `packaging.version.Version` (labels are PEP 440 by
  contract); rows that still fail to parse fall back to `RCT` order.
  The same corrected ordering produces `version_history`.
- Extract the fetch + ordering into a new helper
  `_dataset_version_rows(dataset_rid) -> list[dict]` (ordered
  earliest → latest). `_producer_of_dataset` and the root-descriptor
  path both consume it, so origin, `origin_recorded`, and
  `version_history` are all built from one fetch.
- `packaging>=24` is already a direct dependency (`pyproject.toml`);
  no dependency change needed for the PEP 440 parse.

## Sentinel classification

- Identify the sentinel via the existing
  `unknown_provenance_execution_rid()`; comparison is **RID equality
  only** (RIDs are opaque).
- Cache the sentinel RID per DerivaML instance — today's lookup is an
  uncached Description-filter query and lineage would otherwise pay it
  per call.
- If the sentinel lookup fails (catalog never adopted the contract),
  **degrade, don't raise**: classify nothing (`origin_recorded` stays
  `True` unless the row's `Execution` is None) and continue the walk.
  This is the #365 lesson — leaf helpers inside a lineage walk degrade
  to weaker answers; they never abort the walk.

## Out of scope

- **`Dataset_Dataset` version pinning** (parent links carry no version,
  so "derived from *which* of the 85 parent versions" is unanswerable).
  Real gap, separate schema decision — to be filed as its own issue.
- **Recovering true origins for pre-contract data** — impossible by
  construction; this design reports "unrecorded" honestly and stops.
- **Re-attributing or re-tagging existing `Dataset_Version` rows**
  (e.g. a Version_Kind column) — not needed for the use case; revisit
  only if origin-vs-migration tagging becomes necessary beyond
  presentation.

## Testing

Offline mocks in the established `test_producers_of_dataset_members.py`
style (dict-backed pathBuilder):

1. Unversioned resolution returns the **earliest** row's author; a
   later migration row no longer wins.
2. Dev-version trap: rows `["0.1.0", "0.4.0.post1.dev3", "4.13.0"]` —
   origin is `0.1.0`'s author, not the dev row's.
3. Unparseable version falls back to `RCT` ordering.
4. Sentinel origin → `origin_recorded=False`, `producing_execution`
   still carries the sentinel summary.
5. Row with `Execution=None` at origin → `origin_recorded=False`,
   `producing_execution=None`.
6. No version rows → `origin_recorded=False`, empty history.
7. `version_history` ordered earliest→latest with per-row execution
   and description; populated for dataset roots, empty for an
   execution root.
8. Sentinel-lookup failure degrades (no raise, walk completes).
9. `lookup_lineage`-level: version-pinned consumed-dataset resolution
   unchanged (regression pin on the one-call-site claim).

Each behavioral test verified to fail against the unfixed code before
landing (session convention).

## Acceptance (from the issue's motivating case)

Against eye-ai data shapes: `lookup_lineage("2-277G")` reports
`producing_execution` = sentinel summary with `origin_recorded=False`
(honest "origin unrecorded"), and `version_history` shows `6-FQ28`
as the author of v4.13.0 only — a toucher, not the producer.
