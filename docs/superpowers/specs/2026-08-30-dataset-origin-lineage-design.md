# Dataset origin and version-attribution trace in lineage

**Date:** 2026-08-30
**Issue:** #367 — *Dataset producing_execution is last-writer-wins*
**Status:** Approved (design review with Carl 2026-08-30; revised after
independent Codex review, same day — see "Codex review disposition")

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
| Root `producing_execution` for a dataset | **Origin** = author of the *first-recorded* version row (RCT order) |
| Version-pinned resolution (consumed datasets) | Unchanged — already correct |
| Unknown origin (sentinel / None) | Reported explicitly via `origin_recorded=False`; never silently replaced by a later author **or by a member-producer walk seed** |
| Full attribution history | New `version_history` trace on the **root descriptor only** |
| Schema change / backfill | None |

## Why first-recorded-row author is the origin

`create_dataset` writes the initial `Dataset_Version` row carrying the
creating execution, so for post-contract datasets the first-recorded
row's author is the structural origin **in the normal case**. Two
honest qualifications (Codex findings 1–2):

- `create_dataset` accepts an arbitrary initial `version=`, and
  imported/hand-inserted rows may be non-monotonic — so the version
  *label* is not creation order. Chronology comes from `RCT`.
- Dataset creation is not transactional (Dataset row, then types, then
  the version row); a partial failure plus repair can leave the first
  version row authored by someone other than the dataset's creator.
  The design claims "earliest **recorded** version author", not a
  guaranteed structural fact.

Pre-contract datasets (e.g. `2-277G`, whose first row already carries
the sentinel `6-0B3J`) resolve to the unknown-provenance sentinel — the
honest answer: the true creator was never recorded and cannot be
recovered retroactively.

## Ordering rule

**Primary key: `RCT` (row creation time). Tiebreak: PEP 440 parse of
`Version` (via `packaging`, already a dependency); rows whose labels
don't parse sort after parseable ones within an RCT tie.**

Rationale: `RCT` is a total order, needs no parsing, and directly
encodes "first recorded" — which is the question. Version labels are
only consulted when RCTs tie (e.g. bulk loads). This supersedes the
earlier PEP-440-primary rule and removes its two failure modes:
arbitrary initial version labels, and mixed parseable/unparseable
orderings with no defined total order.

Caveat (recorded, not solved): a catalog cloned without preserving
system columns re-stamps `RCT` at load time, degrading chronology to
load order. Clone provenance is out of scope here.

The same ordering produces `version_history` (earliest → latest).

## Result shape (additive)

In `src/deriva_ml/execution/lineage.py`:

```python
class VersionAttribution(BaseModel):
    """One entry in a dataset's version-attribution trace."""
    model_config = ConfigDict(extra="forbid")

    version: str                          # label as stored, e.g. "4.13.0"
    execution_rid: RID | None             # raw Execution column value
    execution: ExecutionSummary | None    # resolved summary; None if
                                          # execution_rid is None OR the
                                          # lookup could not resolve it
    description: str | None               # the version's release notes


class RootDescriptor(BaseModel):
    ...existing fields...
    producing_execution: ExecutionSummary | None   # datasets: the ORIGIN
    origin_recorded: bool | None = None
    version_history: list[VersionAttribution] = Field(default_factory=list)
```

- `origin_recorded` is tri-state: `True` (real origin recorded),
  `False` (origin is the sentinel, the first row has no `Execution`,
  or the dataset has no version rows), `None` (not applicable — root
  is not a Dataset). A plain bool cannot mean both "yes" and "not
  applicable" (Codex finding 7).
- `VersionAttribution` carries the raw `execution_rid` separately from
  the resolved summary, so a consumer can distinguish "row has no
  author" from "author could not be resolved" (Codex finding 8).
- `producing_execution` carries whatever the first-recorded row says —
  **including the sentinel** (real catalog data); `origin_recorded`
  carries the interpretation. Truth and judgment are separate fields.
- `version_history` is populated only when the root is a Dataset.
- `VersionAttribution` is exported from `execution/__init__.py`
  (`__all__` included), like the other lineage models (finding 12).

### Root-only trace (judgment call, approved)

Datasets appear in a lineage result in two roles: as the **root**
(the artifact asked about) and as **consumed mentions** inside walked
`LineageNode.consumed_datasets`. The trace attaches to the root only:

1. A consumed mention is version-pinned, and the pin *is* the
   provenance of that consumption — "who authored v4.11.0" is already
   answered correctly by the version-scoped path.
2. A popular dataset appears in every walk that trained on it;
   trace-on-every-mention serializes the same N-entry history once per
   mention across the MCP boundary.
3. The escape hatch is one cheap call:
   `lookup_lineage(dataset_rid, depth=0)` returns any dataset's full
   trace with no graph walk.

## Walk seed vs. origin attribution (Codex finding 6 — design change)

Today, when a dataset has no version-producer but its *members* have
producers, `lookup_lineage` seeds the walk from a deterministic member
producer (tk-018 behavior, kept) **and then overwrites
`root.producing_execution` with that walk-root execution**
(`execution.py:1226`). Under the new semantics that overwrite would
claim a member producer is the dataset's origin while
`origin_recorded=False` says otherwise.

Fix: separate the two pieces of state.

- **Walk seeding is unchanged** — member producers still seed the walk
  so member-asset lineage stays reachable.
- **`root.producing_execution` is built from origin resolution only**
  and is never overwritten from the walk root. When origin and walk
  root coincide (the normal recorded-origin case) the result is
  identical to today; when they differ, the root honestly shows
  origin (or sentinel/None + `origin_recorded=False`) while the walk
  still explores the member producers.

## Resolution mechanics

- New helper `_dataset_version_rows(dataset_rid) -> list[dict]`:
  fetches all `Dataset_Version` rows for the dataset **once**, sorted
  by the ordering rule, `RCT` included in the projection. The
  dataset-root path calls it **once** and derives all three outputs —
  origin (first row), `origin_recorded`, and `version_history` — from
  that single result. `_producer_of_dataset(version=None)` is
  reimplemented on top of it; the root path must not trigger a second
  fetch through `_classify_rid` (Codex finding 5).
- **Query cost (corrected — Codex finding 4):** one `Dataset_Version`
  fetch (already paid today) **plus at most one batched Execution
  lookup** for the distinct author RIDs (typically ≤ a handful even for
  85 versions), chunked to bound URL length (tk-023: never one giant
  `.in_()`/disjunction over an unbounded set), plus the workflow-name
  resolution those summaries already use. Root-only, so the cost is
  paid once per `lookup_lineage` call, not per walk node. The earlier
  "zero added round-trips" claim was wrong: version rows hold RIDs,
  not `ExecutionSummary` data.

## Sentinel classification

- Identify the sentinel via the existing
  `unknown_provenance_execution_rid()`; comparison is **RID equality
  only** (RIDs are opaque).
- Cache the **positive** result lazily on the instance
  (`self._sentinel_exec_rid` set on first success); never cache a
  failure — contract adoption can happen during the instance lifetime
  (Codex finding 10).
- Catch **only** the documented absence condition
  (`DerivaMLException` from the helper). Transport/auth/timeout errors
  propagate as usual — swallowing them would label an unknown origin
  as recorded (Codex finding 9). On absence: classify nothing
  (`origin_recorded` computed from row data alone: `False` when the
  first row has no `Execution`, else `True` with no sentinel claim).

## Compatibility

- Additive fields flow through `.model_dump()`; our models'
  `extra="forbid"` constrains what **we** accept, not what consumers
  parse. Verification task (not a design change): confirm the
  deriva-ml-mcp lineage tool wrapper and any snapshot tests tolerate
  the new keys before release (Codex finding 11).
- `lookup_lineage`'s docstring currently promises the latest-version
  author; it and the user-guide lineage section are updated in the
  same change (Codex finding 16).

## Out of scope

- **`Dataset_Dataset` version pinning** (parent links carry no
  version) — separate schema decision, own issue.
- **Recovering true origins for pre-contract data** — impossible by
  construction; reported honestly as unrecorded.
- **Re-attributing or tagging existing `Dataset_Version` rows** (e.g.
  a Version_Kind column) — revisit only if origin-vs-migration tagging
  becomes necessary beyond presentation.
- **Clone-time RCT preservation** — noted caveat, not addressed here.

## Testing

Locations (Codex finding 13): lineage-level behavior in
`tests/execution/test_lookup_lineage_unit.py` (root classification,
member-fallback, serialization mocks live there); ordering-helper tests
alongside the helper's unit tests. **No hand-written RID literals** —
all RIDs fixture-generated (repo rule; the existing literal-heavy
mocks are legacy, not the pattern to copy — Codex finding 14).

1. Unversioned resolution returns the **first-recorded** row's author;
   a later migration row no longer wins.
2. RCT-primary ordering: rows with out-of-order version labels (e.g.
   created at `1.0.0`, later row `0.9.0`) resolve by RCT, not label.
3. RCT tie falls back to PEP 440 label order; unparseable labels sort
   after parseable ones and never crash resolution.
4. Sentinel origin → `origin_recorded=False`, `producing_execution`
   still carries the sentinel summary.
5. First row `Execution=None` → `origin_recorded=False`,
   `producing_execution=None`, `execution_rid=None` in the trace.
6. No version rows → `origin_recorded=False`, empty history.
7. Non-dataset root → `origin_recorded=None`, empty history.
8. `version_history` ordered earliest→latest; entries carry
   `execution_rid` raw plus resolved summary (or None).
9. **Member-producer fallback does not become origin**: dataset with
   no version-producer but member producers → walk seeds from the
   member producer, `root.producing_execution` stays None/sentinel and
   `origin_recorded=False` (pins the finding-6 separation).
10. Query-count: one `Dataset_Version` fetch and one batched Execution
    fetch per root resolution (mock call-count assertions).
11. Sentinel absence (helper raises `DerivaMLException`) degrades as
    specified; a transport-shaped error propagates.
12. `VersionAttribution` importable from `deriva_ml.execution`
    (export pin).

Each behavioral test verified to fail against the unfixed code before
landing (session convention).

## Codex review disposition (2026-08-30)

Independent Codex review returned 17 findings. Accepted and folded in:
1/3 (RCT-primary total ordering), 2 (soften "by construction"),
4 (round-trip claim corrected + batching), 5 (single-fetch plumbing),
6 (walk-seed vs origin separation — the one behavioral design change),
7 (tri-state `origin_recorded`), 8 (raw `execution_rid` in trace),
9 (narrow sentinel catch), 10 (positive-only cache), 12 (export),
13/14/15 (test locations, no RID literals, added cases), 16 (doc
updates), 17 (date corrected to 2026-08-30). Finding 11 downgraded to
a verification task: `extra="forbid"` binds our validation, not
consumers'; MCP-side tolerance gets checked, not redesigned.

## Acceptance (from the issue's motivating case)

Against eye-ai data shapes: `lookup_lineage("2-277G")` reports
`producing_execution` = sentinel summary with `origin_recorded=False`
(honest "origin unrecorded"), and `version_history` shows `6-FQ28`
as the author of v4.13.0 only — a toucher, not the producer.
