# Dataset origin and version-attribution trace in lineage

**Date:** 2026-08-30
**Issue:** #367 — *Dataset producing_execution is last-writer-wins*
**Status:** Approved (design review with Carl 2026-08-30; revised after
two rounds of independent Codex review, same day — see "Codex review
disposition")

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
| Unknown origin (sentinel / None) | Reported explicitly via `origin_recorded=False`; never silently replaced by a later author or by a member-producer walk seed |
| Sentinel as walk root | Never — the sentinel terminates lineage; member producers seed the walk directly |
| Full attribution history | New `version_history` trace on the **root descriptor only** |
| Schema change / backfill | None (one **code-comment** correction in `create_schema.py`) |

## Why first-recorded-row author is the origin

`create_dataset` writes the initial `Dataset_Version` row carrying the
creating execution, so for post-contract datasets the first-recorded
row's author is the structural origin **in the normal case**. Two
honest qualifications:

- `create_dataset` accepts an arbitrary initial `version=`, and
  imported/hand-inserted rows may be non-monotonic — so the version
  *label* is not creation order. Chronology comes from `RCT`.
- Dataset creation is not transactional; a partial failure plus repair
  can leave the first version row authored by someone other than the
  dataset's creator. The design claims "earliest **recorded** version
  author", not a guaranteed structural fact.

Pre-contract datasets (e.g. `2-277G`, whose first row already carries
the sentinel `6-0B3J`) resolve to the unknown-provenance sentinel — the
honest answer: the true creator was never recorded and cannot be
recovered retroactively.

## Ordering rule (total)

Sort key, ascending, per row:

1. **`RCT`** (system column; ERMrest stamps it on every row — if a
   pathological import lacks it, treat as epoch-minimum and let the
   tiebreaks decide);
2. **parsed PEP 440 `Version`** (via `packaging`, already a
   dependency); rows whose labels do not parse sort **after** all
   parseable rows within the same RCT;
3. **raw `Version` string** — makes the order total even for two
   unparseable labels or labels that normalize equal (`1.0` vs
   `1.0.0`), since `(Dataset, Version)` labels are distinct per
   dataset in practice; equal raw labels at equal RCT would be
   duplicate rows, out of scope.

An RCT tie at the head of the history (bulk load) is resolved
deterministically by the tiebreaks; the design does **not** add an
"ambiguous origin" state (rejected as over-engineering — the tiebreak
is deterministic and the case is rare; revisit only if it misleads in
practice).

Caveat (recorded, explicitly out of scope): a catalog cloned without
preserving system columns re-stamps `RCT` at load time, degrading
chronology to load order. Clone provenance is a separate concern; a
follow-up may surface degraded chronology, but this design does not.

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

- `origin_recorded` is tri-state: `True` (a real, non-sentinel origin
  RID is recorded — even if its summary could not be resolved),
  `False` (origin is the sentinel, the first row has no `Execution`,
  or the dataset has no version rows), `None` (root is not a Dataset).
- `producing_execution` is the **resolved summary** of the
  first-recorded row's author when resolvable (sentinel included — it
  is real catalog data). When the recorded origin RID cannot be
  resolved, `producing_execution` is `None` while `origin_recorded`
  stays `True`; the raw RID remains available at
  `version_history[0].execution_rid`. (No separate
  `origin_execution_rid` field on the root — rejected as redundant
  with the trace.)
- `VersionAttribution` carries the raw `execution_rid` separately from
  the resolved summary, distinguishing "row has no author" from
  "author could not be resolved".
- `version_history` is populated only when the root is a Dataset.
- `VersionAttribution` is exported from `execution/__init__.py`
  (`__all__` included).
- The `LineageResult` / `LineageNode` model docstrings are updated in
  the same change: `root.producing_execution` is origin attribution
  and no longer necessarily equals `lineage.execution` (the walk
  root).

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
3. Any dataset's full trace is one **bounded** call away:
   `lookup_lineage(dataset_rid, depth=0)` — no *recursive* walk (it
   still resolves the root node and queries member producers; it is
   cheap, not free).

## Walk seed vs. origin attribution

Today, when a dataset has no version-producer but its *members* have
producers, `lookup_lineage` seeds the walk from a deterministic member
producer (tk-018 behavior, kept) and then overwrites
`root.producing_execution` with that walk-root execution
(`execution.py:1226`). Under the new semantics that overwrite would
claim a member producer is the dataset's origin while
`origin_recorded=False` says otherwise. The two pieces of state are
separated:

- **`root.producing_execution` is built from origin resolution only**
  and is never overwritten from the walk.
- **Walk seeding rules:**
  - Origin is a real (non-sentinel) resolvable execution → walk seeds
    from it; member producers become `extra_parent_rids` (unchanged).
  - Origin is **the sentinel** → the sentinel is **never a walk root**
    (the contract says lineage *terminates* at it; seeding from it
    would fabricate edges implying the unknown-provenance execution
    consumed the member producers). Seed from member producers
    directly, exactly as the no-origin branch does today.
  - Origin RID is recorded but its execution **cannot be expanded** →
    degrade to member-producer seeding rather than returning an empty
    walk; origin fields on the root are populated regardless.
  - No origin, no member producers → root-only result (unchanged).
- When origin and walk root coincide (the normal recorded-origin
  case), results are identical to today.

## Resolution mechanics

- New helper `_dataset_version_rows(dataset_rid) -> list[dict]`:
  fetches all `Dataset_Version` rows for the dataset **once**
  (projection includes `RCT`), sorted by the ordering rule. The
  dataset-root path calls it **once** and derives origin,
  `origin_recorded`, and `version_history` from that single result;
  `_producer_of_dataset(version=None)` is reimplemented on top of it,
  and the root path must not trigger a second fetch through
  `_classify_rid`.
- **Summary resolution (batched):** collect the **distinct** author
  RIDs across the history, then resolve them with batched queries —
  chunked to bound URL length (tk-023: never one unbounded
  `.in_()`/disjunction) — so the request count is
  `ceil(distinct_authors / chunk_size)` for executions, **zero when
  the history is empty or all authors are null**. Workflow names are
  resolved the same way: batch the distinct Workflow RIDs referenced
  by those executions (or join Workflow in the execution query via
  datapath link) — never per-execution N+1 lookups.
- `producing_execution` on the root reuses the summary from this
  batched fetch. `_walk_node` continues to resolve its own execution
  node independently (existing behavior); the two reads may observe
  different live catalog states — accepted, as snapshot consistency
  across a lineage walk is already not guaranteed today.

## Sentinel classification

- Identify the sentinel via a **non-raising** internal lookup
  (`_sentinel_execution_rid_or_none()`): query the sentinel row
  directly; zero rows → `None` (catalog never adopted the contract —
  classify nothing, compute `origin_recorded` from row data alone).
  This replaces catch-based handling: `DerivaMLException` is not an
  absence-specific type, so catching it cannot distinguish absence
  from transport failure. Transport/auth/timeout errors propagate.
- Comparison is **RID equality only** (RIDs are opaque).
- Cache the **positive** result lazily on the instance
  (`self._sentinel_exec_rid`, set on first success); never cache a
  miss — contract adoption can happen during the instance lifetime.
  Instances are effectively single-threaded per the library's usage
  model; no locking.

## Compatibility

- Additive fields flow through `.model_dump()`; our models'
  `extra="forbid"` constrains what **we** accept, not what consumers
  parse. Verification task (pre-release, not a design change): confirm
  the deriva-ml-mcp lineage tool wrapper and any snapshot tests
  tolerate the new keys.
- Documentation updated in the same change: `lookup_lineage`'s
  docstring (currently promises the latest-version author), the
  user-guide lineage section, and the `lineage.py` model docstrings.
- **`create_schema.py` comment correction:** the `Dataset_Version.
  Execution` column comment currently reads "NULL for the initial
  release row, which has no producing execution" — contradicting the
  provenance contract (the initial row carries the creating execution)
  and this design. Corrected in code; affects newly created catalogs'
  column comments only (no migration).

## Out of scope

- **`Dataset_Dataset` version pinning** (parent links carry no
  version) — separate schema decision, own issue.
- **Recovering true origins for pre-contract data** — impossible by
  construction; reported honestly as unrecorded.
- **Re-attributing or tagging existing `Dataset_Version` rows** —
  revisit only if origin-vs-migration tagging becomes necessary.
- **Clone-time RCT preservation / degraded-chronology signaling** —
  recorded caveat; candidate follow-up issue, not in this change.
- **Ambiguous-origin state for equal-RCT heads** — rejected
  (deterministic tiebreak suffices; revisit on evidence).

## Testing

Locations: lineage-level behavior in
`tests/execution/test_lookup_lineage_unit.py`; ordering-helper tests
alongside the helper's unit tests. **RIDs in offline tests are
programmatically generated inside fixtures and asserted only against
values that flowed through the code under test** — this satisfies the
repo rule (no human-written magic RID literals compared against by
hand); it does not require live-catalog fixtures.

1. Unversioned resolution returns the **first-recorded** row's author;
   a later migration row no longer wins.
2. RCT-primary ordering: rows with out-of-order version labels resolve
   by RCT, not label.
3. RCT tie → PEP 440 label order; unparseable labels sort after
   parseable ones; two unparseable labels fall back to raw-string
   order; `1.0` vs `1.0.0` (normalize-equal) resolve deterministically
   by raw string. No crash in any mix.
4. Sentinel origin → `origin_recorded=False`, `producing_execution`
   carries the sentinel summary, **and the walk seeds from member
   producers, not the sentinel** (no fabricated sentinel→producer
   edges).
5. First row `Execution=None` → `origin_recorded=False`,
   `producing_execution=None`, `execution_rid=None` in the trace.
6. No version rows → `origin_recorded=False`, empty history.
7. Non-dataset root → `origin_recorded=None`, empty history.
8. `version_history` ordered earliest→latest; entries carry raw
   `execution_rid` plus resolved summary (or None); duplicate authors
   across rows resolve to **one** batched lookup entry (distinct-RID
   dedup).
9. Member-producer fallback does not become origin: no
   version-producer + member producers → walk seeds from the member
   producer, `root.producing_execution` stays None and
   `origin_recorded=False`.
10. Recorded-but-unexpandable origin → root origin fields populated,
    walk degrades to member-producer seeding (not an empty walk).
11. Query-count: one `Dataset_Version` fetch;
    `ceil(distinct_authors/chunk)` batched execution fetches (assert
    both the 1-chunk and 2-chunk cases); **zero** execution fetches
    when history is empty or all-null; workflow resolution batched,
    never per-execution.
12. Sentinel absence (non-raising lookup returns None) degrades as
    specified; a transport-shaped error propagates. Positive cache:
    second resolution on the same instance issues no sentinel query.
13. `VersionAttribution` importable from `deriva_ml.execution`
    (export pin).

Each behavioral test verified to fail against the unfixed code before
landing (session convention).

## Codex review disposition

**Round 1 (17 findings):** accepted 1–10, 12–17 (RCT ordering,
softened claims, corrected query-cost, single-fetch plumbing,
walk-seed/origin separation, tri-state flag, raw RID in trace,
narrow sentinel handling, positive-only cache, exports, test
locations/cases, doc sequencing, date). Finding 11 downgraded to a
pre-release verification task.

**Round 2 (14 points):** accepted — total-order completion (raw-string
final tiebreak; RCT-absence rule), non-raising sentinel lookup
(replaces exception catching), batched-count wording + workflow-RID
batching (no N+1), root-reuses-batched-summary note,
`producing_execution`-vs-raw-RID contract clarification, **sentinel
never seeds the walk** (no fabricated edges), unexpandable-origin
degradation, model-docstring updates, `depth=0` wording ("no
recursive walk", not "no walk"), `create_schema.py` comment
contradiction (verified in code, fixed as part of this change), and
the added test cases (dedup, cache reuse, chunk boundaries).
Rejected: an "ambiguous origin" state for equal-RCT heads
(deterministic tiebreak suffices); clone-degraded-chronology
signaling (out of scope, candidate follow-up); live-catalog fixtures
for RID compliance (programmatically generated fixture RIDs satisfy
the repo rule — the rule bans human-written literals, not offline
tests).

## Acceptance (from the issue's motivating case)

Against eye-ai data shapes: `lookup_lineage("2-277G")` reports
`producing_execution` = sentinel summary with `origin_recorded=False`
(honest "origin unrecorded"), `version_history` shows `6-FQ28` as the
author of v4.13.0 only — a toucher, not the producer — and the walk
does not present the sentinel as having consumed the member
producers.
