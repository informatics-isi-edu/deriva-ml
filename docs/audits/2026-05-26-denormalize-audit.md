# Denormalize subsystem audit — 2026-05-26

**Auditor:** Claude (Opus 4.7) as audit-only agent, no code changes.
**Status:** Grilled. Verdicts inline.
**Triggering event:** analyst/01 fix landed as
`d1cf9346 fix(denormalize): resolve feature names in include_tables to feature tables`
on top of PR #228. That fix closed a describe-vs-run asymmetry the
existing test suite did not catch. The user asked for an in-depth
look at what else might be hiding in the denormalize subsystem.

---

## Status of findings after /grill-me review

After the original audit (this document, sections 2–6 below), every
finding was walked adversarially against the cited code, cited spec
text, and the existing tests. The grill produced verdicts that
either confirmed, demoted, dropped, or fixed each finding. Inline
**Grill verdict** blocks below each finding record what changed.

**Closed (fix landed):**

| Finding | PR | Notes |
|---|---|---|
| SC-06 / RB-02 / TC-03 | informatics-isi-edu/deriva-ml#230 | row-completeness invariant in `_populate_from_catalog_inner`; hermetic unit test `TestRowCompletenessInvariant` written failing-first, then fix; spec §7 F5 rewritten in the same PR to distinguish invariant from planner-output coincidence. |

**Closed (spec rewrite):**

| Finding | PR | Notes |
|---|---|---|
| SC-04, SC-05, SC-08, RB-09 | informatics-isi-edu/deriva-ml#231 | Doc-rot fixed in the spec rewrite, or finding determined not-a-bug after grilling. |
| Single-source-of-record consolidation | informatics-isi-edu/deriva-ml#242 | Design spec, recovered user guide, and semantics companion merged into `docs/user-guide/denormalization.md` (Mechanism D, 1,712 lines). Old paths removed: `docs/design/denormalization.md`, `docs/concepts/denormalization.md` (was a stub), `docs/superpowers/specs/2026-04-17-denormalization-semantics-design.md`. mkdocs nav + cross-refs updated. |

**Deferred — code work that the spec rewrite calls out:**

| Finding | What the spec says now | Code TODO | Status |
|---|---|---|---|
| SC-01 / RB-01 / TC-09 | §8.3 names a `warnings: list[str]` envelope field on `describe()` | implement the field; route every swallowed exception through it. | **Closed** by informatics-isi-edu/deriva-ml#239 — 13-key envelope with `warnings`, 6 broad-except sites instrumented. |
| SC-02 / TC-05 | §8.1 says `from_rids(dataset_rid=None)` must reject against live catalog (or surface a `reason`) | implement the guard. | **Closed** by informatics-isi-edu/deriva-ml#236 — `from_rids` rejects `dataset_rid=None` against live catalog with `ValueError`. |
| SC-03 | §6 Freshness caveat says `DenormalizeResult` should carry a freshness signal | add `cache_age_seconds: float \| None`. | **Closed** by informatics-isi-edu/deriva-ml#240 — field landed on `DenormalizeResult`. |
| SC-07 / TC-06 | §8.2 declares `as_dict` does NOT stream (matches today's eager materialisation) but flags streaming as the better implementation. | Pick A (correct the docstring — already aligned by spec) or B (implement streaming). Spec is honest either way. | **Closed (Pick A)** by informatics-isi-edu/deriva-ml#233 — docstring corrected to admit eager materialisation. Streaming remains a future enhancement. |

**Confirmed remaining test gaps:**

| Finding | Severity (post-grill) | Disposition | Status |
|---|---|---|---|
| TC-01 | Medium (was Blocker) | demoted — the load-bearing role TC-01 played for SC-06 is now covered by `TestRowCompletenessInvariant`. The cross-channel parity test is still worth writing; spec §9 D.2 marks it as "planned, not yet written." | **Closed** by informatics-isi-edu/deriva-ml#241 (cross-channel parity D.3 added). |
| TC-02 | Medium (was High) | demoted — per-key describe-vs-run parity test still worth writing for the 11 keys analyst/01's `test_describe_and_run_agree` doesn't cover. | **Closed** by informatics-isi-edu/deriva-ml#241 (`TestDescribeKeyParity` covers the remaining keys). |
| TC-04 | Medium (unchanged) | §8 C.5x xfail test for server-side delete still missing; spec rewrite reconciled it to "planned." | **Closed** by informatics-isi-edu/deriva-ml#241 (C.5x xfail test added). |
| TC-07 | Medium (unchanged) | live-catalog resolver test still missing. | **Closed** by informatics-isi-edu/deriva-ml#241 (`test_resolve_table_names_live_feature` added). |
| TC-08 | Medium (unchanged) | `_collect_fk_values` partial-engine-state behavior unpinned. | **Closed** by informatics-isi-edu/deriva-ml#241 (partial-engine-state coverage added). |
| TC-10 | Low (unchanged) | catalog↔bag parity test for A01-shape data — cheap extension of `test_feature_table_multiple_rows_per_anchor`. | **Closed** by informatics-isi-edu/deriva-ml#241 (parity assertion added). |

**Confirmed remaining robustness one-liners (code TODOs):**

| Finding | Severity | One-line fix | Status |
|---|---|---|---|
| RB-03 | Low | `if not rids: continue` filter in describe at line 721 (mirror `_classify_anchors` skip). | Pending. |
| RB-04 | Low | Multi-schema label hazard in `_run`'s per-RID orphan scan; use `denormalize_column_name` or assert single-schema. | Pending. |
| RB-05 | Medium | `Denormalizer.__init__` silent fallback to `source="local"` should log WARNING + attach `_init_warning`. | Pending. |
| RB-06 | Medium | `list_dataset_children` silent fallback to root-only should warn for `source="catalog"`. | Pending. |
| RB-07 | Low | Resolver should dedup after feature-name substitution so `["Image_Classification", "Execution_Image_Image_Classification"]` collapses to one entry. | **Closed** (post-PR-241 follow-up, this branch) — `_resolve_table_names` now dedups resolved `include_tables` / `via` in first-seen order, with `test_resolver_dedupes_after_feature_substitution` as regression. |
| RB-08 | Medium | `_collect_fk_values` composite-FK assumption — AND conditions or document single-column FK requirement. | **Closed** by informatics-isi-edu/deriva-ml#238 — `_collect_fk_values` raises `NotImplementedError` on composite FK rather than silently mis-fetching. |
| RB-10 | Low | Remove unused `model` parameter from `_populate_from_catalog`. | Pending. |

**Dropped after grilling:**

| Finding | Reason |
|---|---|
| SC-08 | The audit's `[inferred]` reasoning didn't hold against the actual planner code. `_prepare_wide_table` does not call `dataset.list_dataset_members()`; the "planner is pure; no I/O" claim is accurate (spec §2 in the rewrite tightens what "no I/O" means anyway). |
| RB-09 | The audit's own writeup explicitly said "Not a bug; flagged for completeness." Removed from findings list. |

**Overstated after grilling:**

| Finding | What changed |
|---|---|
| RB-07 | The auditor walked the code mid-finding and noted *"OK; orphan emission uses the resolved list. So this is correct"* — withdrawing the main claim themselves. What remains is a Low de-dup concern for callers naming both feature and feature-table in one include list. Recategorize as "resolver should dedup after substitution." |

**Cross-cutting patterns the grill confirmed:**

1. **Silent-zero family** (SC-01, SC-02, SC-03, RB-01, RB-05, RB-06) all reduce to two surfaces: a `warnings: list[str]` field on `describe()` envelopes + a `cache_age_seconds: float | None` on `DenormalizeResult`. One pattern, two cheap fields.
2. **Documentation honesty deficit.** The audit was rigorous about marking `[inferred]` (saved SC-08 from being a false-positive accusation) but sloppy about distinguishing spec citations from paraphrases (SC-01 confused code-comment language with spec text). Future audits: quote verbatim, never paraphrase.
3. **The spec was the bigger problem than the code.** Most "confirmed" findings have small or nil code fixes; the spec amendments are what make the behavior knowable. PR #231 (spec completeness) is the highest-leverage deliverable from this whole audit thread.
4. **Spec defense vs. test defense are different bug classes.**
   - **analyst/01-class** (silent describe-vs-run disagreement) would have been caught by a complete spec — PR #231's new "describe-vs-run agreement contract" §8.3.4 is exactly the contract analyst/01 violated.
   - **SC-06-class** (code violates an existing contract via wrong implementation choice) would NOT have been caught by the spec; the §6 step 3 language was already correct. SC-06-class bugs need test-level defense, which PR #230's `TestRowCompletenessInvariant` provides.

---

## 1. Scope and methodology

**Files read for this audit (canonical spec is the design doc):**

- Spec: `/Users/carl/GitHub/DerivaML/deriva-ml/docs/design/denormalization.md` (2026-05-21, authoritative).
- Companion spec: `/Users/carl/GitHub/DerivaML/deriva-ml/docs/superpowers/specs/2026-04-17-denormalization-semantics-design.md` (semantic Rules 1–10).
- Concept doc cross-check: `/Users/carl/GitHub/DerivaML/deriva-ml/docs/concepts/denormalization.md` (skimmed; user-facing only).
- Implementation:
  - `/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/local_db/denormalizer.py` (1348 lines)
  - `/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/local_db/denormalize.py` (681 lines)
  - `/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/local_db/paged_fetcher.py` (410 lines)
  - `/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/local_db/paged_fetcher_ermrest.py` (148 lines)
  - `/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/model/denormalize_planner.py` (1858 lines — skimmed for hooks the denormalizer reaches into; not audited end-to-end)
- Tests:
  - `/Users/carl/GitHub/DerivaML/deriva-ml/tests/local_db/test_denormalizer.py` (764)
  - `/Users/carl/GitHub/DerivaML/deriva-ml/tests/local_db/test_paged_fetcher.py` (887)
  - `/Users/carl/GitHub/DerivaML/deriva-ml/tests/local_db/test_paged_fetcher_live.py` (103)
  - `/Users/carl/GitHub/DerivaML/deriva-ml/tests/local_db/test_denormalize_impl.py` (357)
  - `/Users/carl/GitHub/DerivaML/deriva-ml/tests/dataset/test_denormalize.py` (1806)
- Recent fix patches: `d1cf9346` (analyst/01, this week), `b9bb6576` (A01 stateless refactor, A01 closure).

**Out of scope (deliberately, to stay focused on the denormalize surface):**

- The planner itself (`denormalize_planner.py`) beyond verifying which
  methods the Denormalizer reaches into. Rule 1–10 implementation
  correctness inside the planner would be its own audit.
- The Dataset / DatasetBag wrappers (`get_denormalized_as_dataframe`
  facade methods on `Dataset` / `DatasetBag`). The Denormalizer is the
  contracted surface; the facades just forward.
- `paged_fetcher_ermrest.py` HTTP URL construction beyond verifying
  the count/page/batch protocol exists and has unit coverage.
- Live-catalog tests that require a running ERMrest. Coverage status
  is read off the design doc's §8 status table and the existing test
  file names.

**Methodology:**

1. Walked the spec's §3 (state), §4 (fetcher), §5 (insert), §6
   (denormalize), §7 (fragility map), and §8 (test matrix) as the
   set of claims to verify.
2. For each claim, located the implementing code, then asked: does
   the code actually do this, and does a test pin it?
3. Cross-referenced against analyst/01's fix to look for nearby
   asymmetries the same class of test wouldn't catch.
4. Walked the code paths for explicit `try/except` blocks looking for
   swallowed exceptions (the analyst/01 root cause); flagged any that
   looked load-bearing.

Inferences (vs. things I directly observed) are marked `[inferred]`
per the tacit-knowledge convention.

---

## 2. Spec compliance findings

Format per finding:

```
SC-NN: Severity | Component | What | Reproduction | Impact | Suggested classification
```

### SC-01 — `describe()` swallows feature-name resolution failures, hiding them as silent absence

**Grill verdict:** confirmed (with wrong framing). The audit's spec citation — *"Spec §6 says ... every failure mode is swallowed and represented in the returned dict as `None`/`[]`/`{}` in the affected positions"* — is **invented**. Searched the spec for "swallow", "dry-run", "None/[]/{}", "sensible defaults" — none of those phrases exist in §6 or anywhere else. The "dry-run invariant" lives only in the code's own comment block; the audit conflated a code-design choice with a spec contract. The underlying observation (PR #228's `try/except Exception` on `_resolve_table_names` loses diagnostic information) is real; the spec-violation framing is not. **Reclassify as a PR #228 review comment / robustness concern, not a spec-compliance finding.** Identical content as RB-01 — collapse the two.
**Disposition:** Spec PR #231 §8.3 names the `warnings: list[str]` envelope field; code TODO to implement.
**Revised severity:** Medium (unchanged).

**Severity:** Medium
**Component:** `denormalizer.py::Denormalizer.describe`, lines ~640–657
**What:** Spec §6 says `describe` is dry-run — "every failure mode ...
is swallowed and represented in the returned dict as `None` / `[]` /
`{}` in the affected positions." After the analyst/01 fix, the
resolver block at the top of `describe` collapses **any** exception
from `_resolve_table_names` (including ambiguous-feature
`DerivaMLDenormalizeError` and missing-table `DerivaMLTableNotFound`)
to the original unresolved inputs. The dict still gets populated by
the downstream calls that fail later, but **the caller is not told
the resolver failed and which name was the problem.** This is the
same shape as analyst/01: describe silently accepts something that
run will reject — only this time it is the resolver failing on
describe and run will surface a clearer error.

**Reproduction:**
```python
# Two features sharing a name on different targets.
plan = d.describe(["Image", "AmbiguousName"])  # returns dict, no signal
df  = d.as_dataframe(["Image", "AmbiguousName"])  # raises DerivaMLDenormalizeError
```
`plan["row_per"]` will be `None` because the planner walks ran on the
raw `"AmbiguousName"`; the user has no way to know why.

**Impact:** Re-introduces an analyst/01-shaped asymmetry one level
down. The dry-run invariant is preserved literally (no raise) but the
information-preservation invariant ("describe tells you why run would
fail") is not.

**Suggested classification:** `unimplemented` — spec §6 frames
describe as a diagnostic; surfacing *why* the plan is empty is part
of "well-formed dict with sensible defaults."

---

### SC-02 — `from_rids` placeholder `dataset_rid` silently returns zero rows on production catalogs

**Grill verdict:** confirmed. Behavior matches docstring (lines 219–225); the silent-zero failure mode is real. Audit's analogy to A02 / "honest unknown" pattern is well-grounded (spec §7 F6 + §8 E.2 codify the pattern). Spec was silent on `from_rids`; PR #231 §8.1 documents the limitation and prescribes the fix.
**Disposition:** Spec PR #231 names the contract; code TODO to either reject `dataset_rid=None` against a live catalog or surface a `reason` in the empty-result envelope.
**Revised severity:** Medium (unchanged).

**Severity:** Medium
**Component:** `denormalizer.py::Denormalizer.from_rids`, lines 196–356
**What:** The docstring says: *"When `dataset_rid` is omitted, the
first anchor's RID is used as a pseudo-scope (which will currently
return zero rows against a production catalog — a known limitation)."*
This is acknowledged in the docstring but **not in the design doc**,
and **not surfaced as a `reason` field or warning** at call time. The
caller gets an empty DataFrame with no breadcrumb. This is exactly
the silent-zero failure mode A02 fixed in `describe`, repeated here.

**Reproduction:**
```python
d = Denormalizer.from_rids([("Image", "1-IMG-1"), ("Image", "1-IMG-2")], ml=ml)
df = d.as_dataframe(["Image"])  # 0 rows, no explanation
```

**Impact:** Discoverability hazard. Same class of bug as A02 (silent
zero) that the project already paid down once.

**Suggested classification:** `over-implemented` — the API accepts
calls it can't fulfill correctly. Either reject (raise ValueError if
`dataset_rid is None` against a non-local source) or warn with the
same "honest unknown" envelope `describe` now uses.

---

### SC-03 — Spec §3 says the engine survives the process; spec §6 freshness says deletions are not observed; no code path makes the staleness window visible to the caller

**Grill verdict:** confirmed; dual-severity reframed to single label. Spec §6 + §7 F3/F4 honestly document the limitation; `DenormalizeResult` carries no caller-visible signal — confirmed by inspection (no `cache_age_seconds`, no `stale`, no timestamp). Audit's dual "Low...but High" is itself diagnostic of audit uncertainty. The trade-off is real but the rights answer is Medium with a cheap field, not High.
**Disposition:** Spec PR #231 §6 Freshness caveat names the signal; code TODO to add `cache_age_seconds: float | None` (or equivalent) to `DenormalizeResult`.
**Revised severity:** Medium (was: Low/High dual).

**Severity:** Low (documented limitation) but **High** as a usability
trap.
**Component:** `denormalize.py::_populate_from_catalog`, lines 391–465
**What:** Spec §6 ("Freshness caveat") and the F3/F4 rows of §7
acknowledge that server-side deletes and updates are NOT observed
because the local SQLite cache is write-through-only. The code
implements this faithfully — `_insert_rows` uses
`INSERT OR IGNORE`, never `DELETE` or `UPDATE` to reconcile against
the server. **But there is no way to know whether your result was
affected.** No timestamp on the local cache, no last-fetch marker
exposed to the caller, no flag on `DenormalizeResult` that says "X
of the joined rows came from a fetch older than Y minutes."

**Reproduction:** Per F3/F4 — server-side deletion of an
`Execution_Image_Quality` row, then re-denormalize, still shows the
deleted row. Documented; not surfaced.

**Impact:** A user re-running denormalize in a long-lived process
trusts the row count. They have no signal to invalidate that trust.

**Suggested classification:** `unimplemented` — design says
"Documented; tracked as a known limitation. Tests assert the
positive behavior (new server rows show up) but `xfail` the
deletion/update freshness cases until we decide to fix." The
deletion/update test is not present in the suite as written (see
TC-04). The design doc itself names the gap but the code makes it
hard to act on.

---

### SC-04 — Spec §4 says PagedFetcher carries "memoised per-table row counts" only; the actual class also holds the `_counts` dict and the cardinality heuristic uses it across multiple `fetch_by_rids_or_predicate` calls within one fetcher lifetime

**Grill verdict:** confirmed Low doc-rot. Code matches the spec semantically; the spec's "only" word is the imprecise bit. Verified `_counts` at line 159 of paged_fetcher.py.
**Disposition:** Closed by PR #231 — §3 row count memo subsection now says "within one fetcher lifetime; lazy; stale-within-call possible."
**Revised severity:** Low (unchanged).

**Severity:** Low (correct per spec, but the spec's "only" word is
slightly misleading and the test `test_memoizes_count` cements
behavior that is more nuanced than "no state survives a call")
**Component:** `paged_fetcher.py::PagedFetcher._counts`
**What:** Spec §3 table row 5 says PagedFetcher state lifetime is
"one Denormalizer." Spec §4 says fetcher state is the "per-table
row-count memo (used by the cardinality heuristic) and nothing
else." These are consistent. But the `_populate_from_catalog`
function constructs **one** fetcher per `_denormalize_impl` call,
and *within that one call* the fetcher does cache `count(table)`
across `fetch_by_rids_or_predicate` invocations. That's a soft state
carry the audit reader should be aware of: if the catalog mutates
between two table fetches **inside one denormalize call** (rare but
not impossible), the cardinality heuristic uses a stale count.

**Reproduction:** [inferred] — concurrent mutation between two
sequential fetches inside one denormalize call against a write-active
catalog. No test simulates this.

**Impact:** Low. The heuristic only chooses between two correct
strategies (predicate vs. RID batch); a stale count doesn't produce
wrong rows, just suboptimal HTTP.

**Suggested classification:** `doc-rot` — strictly correct, but the
spec could be more explicit about "within one fetcher lifetime, counts
are memoized once."

---

### SC-05 — Spec §5 says `_insert_rows` "raise" on rows missing RID; the code does not raise explicitly — it lets SQLite's NOT NULL constraint fire

**Grill verdict:** confirmed Low doc-rot. Confirmed by reading lines 377–410 of paged_fetcher.py: no explicit guard, relies on engine NOT NULL.
**Disposition:** Closed by PR #231 — §5 "Missing-RID behavior — engine-enforced" subsection names the dialect coupling. Optional code TODO: add explicit `if "RID" not in row: raise ValueError(...)` belt-and-suspenders guard.
**Revised severity:** Low (unchanged).

**Severity:** Low (correct behavior, weak contract)
**Component:** `paged_fetcher.py::PagedFetcher._insert_rows`,
lines 377–410
**What:** Spec §8 row A.4 pins "Insert rows missing RID — raise (rows
without RID are a programming error)." The test
`test_insert_with_missing_rid_in_row` asserts `IntegrityError` (a
SQLAlchemy/SQLite error). The contract is implemented by the *engine*,
not by the function. This works today because every supported engine
(SQLite, per `local_db/README.md`) enforces NOT NULL on the RID
primary key. If a future engine has a different RID constraint (e.g.
generated RIDs), the contract silently drifts.

**Reproduction:** N/A — works correctly today.

**Impact:** Lurking dialect coupling — same place spec §5 calls out
("the dialect-aware seam").

**Suggested classification:** `doc-rot` — either say "we rely on the
RID column being declared NOT NULL on every supported engine" or add
a tiny `if "RID" not in row: raise ValueError` guard.

---

### SC-06 — Spec §6 step 3 says "must always re-issue the fetch for every (table, rid_column, rid) tuple in the plan"; the code does NOT re-issue per-tuple — it dedups via the `processed` set inside `_populate_from_catalog_inner`

**Grill verdict:** **CONFIRMED, FIXED.** Initial grill considered overstated (spec §7 F5 said "not currently a bug") but the user named the row-completeness invariant precisely: *"if two paths reach the same table with different (rid_column, rids), the local cache must contain the union of rows both would have fetched."* That invariant is exactly Spec §6 Clause B, and the code's table-name dedup only coincidentally satisfied it under today's planner. PR #230's `TestRowCompletenessInvariant` constructs a two-path plan with disjoint Image sets and shows main drops rows: `"Image RIDs requested across all fetches: {'IMG-A1', 'IMG-A2'}. Expected union: {'IMG-A1', 'IMG-A2', 'IMG-B1', 'IMG-B2'}. Missing: {'IMG-B1', 'IMG-B2'}."` Fix: dedup key changed to `(table_name, rid_column, frozenset(rids))`. F5's "not currently a bug" claim was wishful thinking — coincidence, not contract.
**Disposition:** Closed by informatics-isi-edu/deriva-ml#230 (code) + spec §7 F5 rewrite + new §8 test rows C.8 and D.3.
**Revised severity:** High (confirmed).

**Severity:** **High** (potential correctness gap mirroring A01)
**Component:** `denormalize.py::_populate_from_catalog_inner`,
lines 468–554; specifically the `processed: set[str] = {"Dataset"}`
guard and the `if table_name in processed: ... continue` early exit.
**What:** Spec §6 contract reads:
> "**must always re-issue the fetch for every (table, rid_column, rid)
> tuple in the plan** — because the server is the only authority for
> 'what rows match this query right now.'"

Code reality: `processed` is keyed on **table name only**, not on the
`(table, rid_column, rid)` tuple. If the same table appears on two
different join paths with **different rid_column / different RID
inputs** (e.g. one path fetches by PK `RID`, another by FK `Image`),
the second walk's fetch is **skipped** entirely. The first walk's
narrower fetch wins.

This is structurally the same shape as A01: a dedup based on
"already-seen at table granularity" that doesn't account for the rid
column being the real identifier of "what was asked."

**Reproduction (constructive, [inferred]):**
- A join plan where path 1 reaches `Image` via `Dataset_Image.Image`
  (rid_column=`RID`, scoped to a subset).
- A join plan where path 2 reaches `Image` via `Execution_Image_Quality.Image`
  (rid_column=`Image`, scoped to a different/broader set).
- Today: path 1 fetches `Image` first, `processed` adds `"Image"`,
  path 2 is skipped, and any Images that were members of path 2 but
  not path 1 are absent from the engine. SQL join silently drops
  rows.

The CSA e2e at N=50 didn't surface this because all paths happen to
converge on the same dataset-scoped Image set, [inferred].

**Impact:** Silent row loss on multi-element-path plans where two
paths visit the same table from different angles. Probability rises
with `len(include_tables) > 2`, with feature tables involved, and
with split datasets (children vs. root).

**Suggested classification:** `unimplemented` — the contract is in
the design doc; the code is one stale-pattern refactor old.

---

### SC-07 — `as_dict` materializes the full result then yields from it, contradicting its own docstring's "memory-efficient for large results" claim

**Grill verdict:** confirmed; severity **bumped Medium → High** for documentation integrity. The docstring is internally inconsistent — same docstring claims streaming AND admits eager materialisation in its Raises: block. A user who picks `as_dict` for the documented reason OOMs on a large result with no way to debug.
**Disposition:** Spec PR #231 §8.2 declares `as_dict` does NOT stream (matches today's eager materialisation) and flags streaming as the better implementation. Code TODO is binary: either correct the docstring (cheap, aligned with spec) or implement streaming (better but more work).
**Revised severity:** **High** (was Medium).

**Severity:** Medium
**Component:** `denormalizer.py::Denormalizer.as_dict`, lines 440–486
**What:** Docstring says:
> "Use this when the result set won't fit in memory or when downstream
> code processes rows one at a time."

Implementation: calls `self._run(...)` which calls `_denormalize_impl`,
which executes `session.execute(final_query)` and **eagerly
materialises** all rows into `rows = [dict(row._mapping) for row in
result]` (`denormalize.py:386`). Then `as_dict` yields from that fully
materialised list via `result.iter_rows()`.

There is no streaming path. The "memory-efficient" claim is false.

**Reproduction:** any denormalize result; profile the memory.

**Impact:** A user who picks `as_dict` because the docstring says it
streams will OOM on a million-row result. Worse, they'll have no
reason to debug — the docstring lied.

**Suggested classification:** `doc-rot` — the contract advertised by
the docstring is unimplemented. Either implement streaming (yield
inside the `Session.execute` loop) or correct the docstring.

---

### SC-08 — Spec §6 step 2 says "Plans the join via `_prepare_wide_table` (planner is pure; no I/O)" — the planner DOES make calls to the dataset (`list_dataset_members`, `list_dataset_children`) that can hit the network on a live Dataset

**Grill verdict:** **WRONG, dropped.** The audit's `[inferred]` honesty saved it from being a false-positive accusation: `_prepare_wide_table` in `denormalize_planner.py` takes a `dataset` parameter but **does not actually call `dataset.list_dataset_members()` or `dataset.list_dataset_children()`**. The grep returned no such calls. The planner's "no I/O" claim holds. PR #231 §2 tightens what "no I/O" means anyway (no HTTP / no engine reads / no dataset I/O) for future-proofing, but no code defect.
**Disposition:** DROPPED.
**Revised severity:** n/a.

**Severity:** Low (depends on what "I/O" means)
**Component:** `denormalize_planner.py::_prepare_wide_table` (called
from `denormalizer.py:535, 655, 1316`; from `denormalize.py:274`).
**What:** Spec §2 architecture diagram and §6 step 2 both label the
planner as "pure model code; no I/O." In practice, `_prepare_wide_table`
takes a `dataset` and a `dataset_rid` and may call
`dataset.list_dataset_members()` to discover element types. Against
a live `Dataset`, that's a network call. The planner is pure
*relative to the catalog schema* but not *relative to the dataset
state*.

[inferred] from reading the planner signature and how it's called;
have not traced every code path inside the planner.

**Impact:** Low — works correctly. The "no I/O" claim just isn't
literally true. Could become a problem if someone optimizes assuming
the planner is offline-safe.

**Suggested classification:** `doc-rot` — clarify "no fetch-step
I/O; may consult `DatasetLike` members."

---

## 3. Test coverage gaps

Format per finding: `TC-NN: Severity | What contract is unpinned | User-facing impact`.

Sorted by user-facing impact (highest first).

### TC-01 — No invariant test pins "the SAME plan produces the SAME row count across `source='catalog'` and `source='local'`"

**Grill verdict:** confirmed (still a gap); severity **demoted Blocker → Medium**. The Blocker justification was "cheapest test that would catch SC-06" — but PR #230's hermetic `TestRowCompletenessInvariant` catches SC-06 directly. Cross-channel parity test still worth writing as a smoke test for any future violation; spec §9 D.2 marks as "planned, not yet written."
**Disposition:** Spec PR #231 reconciled the §9 D.2 row to "planned." Test TODO.
**Revised severity:** Medium (was Blocker).

**Severity:** Blocker
**Contract:** Spec §8 Layer D — D.1 and D.2 are listed as "not
currently covered; new additions." Status as of design doc 2026-05-21
says they're additions. The current test
`test_catalog_and_bag_denormalize_consistency`
(`/Users/carl/GitHub/DerivaML/deriva-ml/tests/dataset/test_denormalize.py:753`)
checks row count parity for `include_tables=["Subject"]` only —
that's the trivial single-table case. **No test exercises D.2
(multi-feature-per-anchor, the A01 shape, on both sources).**

**Impact:** The A01 root cause was a behavior asymmetry between
"first run cached" and "subsequent run after server mutation." A
parity test between catalog and bag is the cheapest way to catch the
SC-06 class of bug (a path skip in `_populate_from_catalog_inner`
would manifest as catalog source missing rows the bag source has).

**Where the gap would surface:** SC-06 above.

---

### TC-02 — No test pins "describe and run agree on `row_per_candidates`" or any of the other 11 keys

**Grill verdict:** confirmed; severity **demoted High → Medium**. Audit's count of 12 keys verified by reading describe's return dict on the PR #228 branch. The 11 unpinned keys each carry analyst/01-shape asymmetry risk, but after SC-06 was fixed the load-bearing test gap was closed elsewhere. Per-key parity test still worth writing.
**Disposition:** Test TODO.
**Revised severity:** Medium (was High).

**Severity:** High
**Contract:** Implicit in the analyst/01 fix. The new test
`test_describe_and_run_agree` (in `TestFeatureNameResolution`) only
checks the `{name for name, _ in plan["columns"]} == set(df.columns)`
invariant on **one specific input**. Each other key in the 12-key
describe envelope (`row_per`, `row_per_source`, `row_per_candidates`,
`join_path`, `transparent_intermediates`, `ambiguities`,
`estimated_row_count`, `anchors`, `source`) has a separate analyst/01-
style asymmetry possibility. None has its own describe-vs-run
invariant.

**Impact:** Same class of bug as analyst/01 can hide behind any of
the other 11 keys. For instance: does `plan["join_path"]` match the
tables `_run` actually walked? Does `plan["row_per"]` match the
`resolved_row_per` `_run` uses internally? No test asserts.

---

### TC-03 — No live `_populate_from_catalog_inner` test exercising same-table-on-two-paths

**Grill verdict:** confirmed → **CLOSED by PR #230**. PR #230's hermetic unit test `TestRowCompletenessInvariant` is exactly this: two-path scenario with disjoint Image RIDs, failing on main, passing after the fix.
**Disposition:** Closed by informatics-isi-edu/deriva-ml#230.
**Revised severity:** n/a (fixed).

**Severity:** High
**Contract:** SC-06's failure mode. Spec §8 row F5 says path-walk
order *is* part of the contract; row F5 says "Pin to the test matrix
in §8 so a regression here would be caught" — but the only test
that exercises join-path ordering is the implicit assumption inside
`test_feature_table_multiple_rows_per_anchor` (A01 regression). That
test uses a SINGLE element path. F5's hazard requires TWO element
paths converging on one table.

**Impact:** SC-06 ships unobserved.

---

### TC-04 — Freshness/staleness `xfail` test for server-side delete is described in spec §8 row C.5x but is NOT in the test file

**Grill verdict:** confirmed. Grep for `xfail` in tests/local_db/ and tests/dataset/ found no denormalize-freshness xfail.
**Disposition:** Spec PR #231 reconciled the §8 C.5x row to "planned, not yet written." Test TODO.
**Revised severity:** Medium (unchanged).

**Severity:** Medium
**Contract:** Spec §8: "`C.5x | Mutation → re-denormalize. Deletion
or update server-side between calls. | live, xfail | freshness
limitation`." Searched the test suite for `xfail`; the only
denormalize-related xfail is none. The spec **claims** the
documentation is paired with an `xfail` test. There is no such
test.

**Impact:** F3/F4 (server-side deletion not observed) — the project
has nothing reminding it that this limitation exists. If a future
contributor adds invalidation in the wrong place, the regression
ships silently.

---

### TC-05 — `from_rids` with `dataset_rid=None` against a non-trivial fixture is not tested for the zero-row failure mode

**Grill verdict:** confirmed. Bundles with SC-02.
**Disposition:** Test TODO; lands with the SC-02 code fix.
**Revised severity:** Medium (unchanged).

**Severity:** Medium
**Contract:** SC-02's failure mode. `test_from_rids_with_table_tuples`
**always passes** an explicit `dataset_rid` (line 597 of
`test_denormalizer.py`). No test exercises the docstring's
"placeholder dataset_rid" caveat — i.e. nothing verifies what
happens when the placeholder is used. The behavior is whatever
SQLAlchemy returns when filtering `Dataset.RID IN (<image_rid>)` —
zero rows — but a test would pin this.

**Impact:** SC-02's silent zero. A user calling `from_rids` without
`dataset_rid` against a live catalog gets a wordless empty
DataFrame.

---

### TC-06 — `as_dict` streaming claim has no test that would catch the eager-materialization gap

**Grill verdict:** confirmed. Severity contingent on SC-07's resolution: if SC-07 fixes by correcting the docstring → Low (no test needed). If SC-07 implements streaming → load-bearing test.
**Disposition:** Bundles with SC-07.
**Revised severity:** Medium (unchanged, contingent).

**Severity:** Medium
**Contract:** SC-07's failure mode. The existing `test_yields_dicts`
just consumes the generator and counts rows. There is no test that:
(a) constructs a large result set, (b) measures peak memory before
the first `next()` returns, (c) asserts memory is bounded.

**Impact:** Performance/scaling cliff. Won't show up until a user
pulls a denormalized table with many millions of rows; then it OOMs.

---

### TC-07 — `_resolve_table_names` is exercised via stubbed `find_features` only; no test against a real feature-table-bearing fixture

**Grill verdict:** confirmed.
**Disposition:** Test TODO; live-catalog resolver test against a feature-bearing fixture.
**Revised severity:** Medium (unchanged).

**Severity:** Medium
**Contract:** The analyst/01 fix's tests
(`TestFeatureNameResolution`) all monkey-patch
`populated_denorm["model"].find_features = lambda ...`. There is no
test of the resolver against an actual catalog model with real
features and the actual ambiguity / not-found code paths. If
`find_features()` against a real model raises an exception type the
stub doesn't simulate, the `except Exception: feature_index = {}`
fallback silently swallows it (this is the analyst/01 pattern again).

**Impact:** A real catalog's `find_features` failure mode — e.g.
network timeout during describe — collapses to "feature name not
found" with no diagnostic.

---

### TC-08 — No test pins the `_collect_fk_values` behavior when the engine has partial data for the FK source table

**Grill verdict:** confirmed. Same F4 family as SC-03.
**Disposition:** Test TODO.
**Revised severity:** Medium (unchanged).

**Severity:** Medium
**Contract:** `denormalize.py::_collect_fk_values` (lines 557–620)
queries the local engine for distinct non-null values to feed the
next fetch. If the engine has only some of the prior table's rows
(because a previous denormalize call scoped it tighter), the new
fetch is scoped to that subset — silently. No test pins:
"engine has K rows of Subject; new denormalize needs all N>K
Subjects; the fetcher correctly broadens the fetch."

[inferred] this is what F4 in the fragility map names. The existing
suite has nothing that simulates the partial-state precondition.

**Impact:** Cross-session row loss when the workspace is shared
between two different dataset queries.

---

### TC-09 — No test pins behavior when `name_to_table` raises *inside* `_resolve_table_names` for a reason other than DerivaMLException

**Grill verdict:** confirmed. Verified by reading the PR #228 branch line 1062 — only `DerivaMLException` is caught narrowly; other exceptions propagate to describe's broad-except.
**Disposition:** Bundles with SC-01 / RB-01 `warnings` envelope fix.
**Revised severity:** Low (unchanged).

**Severity:** Low
**Contract:** `_resolve_table_names` catches `DerivaMLException`
narrowly (`denormalizer.py:1027ish`); other exception types (e.g.
network error during catalog introspection, AttributeError on a
malformed model) propagate. Describe wraps the whole call in a broad
`except Exception:` and reverts to original inputs — which then go
to the planner and either succeed (if it's a real table name) or
fail later.

**Impact:** Diagnostic loss. The user sees the *second* error, not
the first. Same class as SC-01.

---

### TC-10 — No test exercises the catalog→bag round-trip for an A01-shape dataset (multi-feature-per-anchor through DatasetBag.get_denormalized_as_dataframe)

**Grill verdict:** confirmed; cheap extension of existing `test_feature_table_multiple_rows_per_anchor` (line 1570).
**Disposition:** Test TODO.
**Revised severity:** Low (unchanged).

**Severity:** Low
**Contract:** Spec §8 D.2. The A01 regression test
`test_feature_table_multiple_rows_per_anchor` runs against the live
catalog only. After it adds the EXTRA_ANNOTATORS, no parallel call
fetches the same data via a downloaded bag and asserts equality.

**Impact:** A drift between bag and live paths could ship; would not
catch SC-06 either.

---

## 4. Robustness concerns

Format per finding: `RB-NN: Severity | Component | What | Reproduction | Impact | Suggested classification`.

### RB-01 — Broad `except Exception` in `describe` swallows resolver failures (the analyst/01 pattern)

**Grill verdict:** confirmed; **duplicate of SC-01.** Same finding restated from the robustness angle. Collapse into SC-01's disposition.
**Disposition:** Spec PR #231 §8.3 names the `warnings` envelope; code TODO bundles SC-01 + RB-01 + TC-09.
**Revised severity:** High (matches SC-01 bumped severity for the user-facing diagnosability claim).

**Severity:** High
**Component:** `denormalizer.py::Denormalizer.describe`, lines
~639–654.
**What:** After analyst/01's fix, the describe method wraps
`_resolve_table_names` in `try/except Exception` and silently
collapses to the original inputs. **Then** it makes 5 more
`try/except Exception` calls (lines 638–720) wrapping individual
planner / classifier hooks, each swallowing any error.

This is the analyst/01 fix's own design choice ("dry-run invariant:
describe never raises") taken to its logical limit. The cost is that
every failure mode becomes "an empty key in the dict" with no
diagnostic.

**Reproduction:** SC-01.

**Impact:** Diagnosability. The class of bugs analyst/01 surfaced are
exactly the ones describe's broad-except hides one level deeper.

**Suggested classification:** Add a `warnings` list to the describe
dict; on any swallowed exception, append a brief reason. The
dry-run-never-raises invariant is preserved; the
information-preservation invariant is restored. This is a much
smaller change than re-raising.

---

### RB-02 — `_populate_from_catalog_inner`'s `processed` set is keyed on table name only

**Grill verdict:** confirmed → **CLOSED by PR #230.** Duplicate of SC-06; same fix landed.
**Disposition:** Closed by informatics-isi-edu/deriva-ml#230.
**Revised severity:** Blocker → fixed.

**Severity:** **Blocker** for some plans, latent on CSA.
**Component:** `denormalize.py::_populate_from_catalog_inner`, line
501.
**What:** See SC-06. The dedup key is the wrong shape. Should be
`(table_name, rid_column, frozenset(rids))` or equivalent. Currently
the first walk wins.

**Reproduction:** SC-06 above; constructive `[inferred]`.

**Impact:** Silent row drop in any plan with two element paths
through the same table.

**Suggested classification:** Same class as A01. Fix the key
shape; add the test from TC-03.

---

### RB-03 — `_anchors_as_dict` and `_classify_anchors` silently include empty FK-edge case in `_classify_anchors` but `_anchors_as_dict` does not pre-filter

**Grill verdict:** confirmed. Verified — line 1188 has `if not rids: continue`; line 721 doesn't. Describe and run can report different anchor counts.
**Disposition:** Code TODO — one-line `if not rids: continue` at line 721.
**Revised severity:** Low (unchanged).

**Severity:** Low
**Component:** `denormalizer.py::_classify_anchors` line 1188
("if not rids: continue"), and `_anchors_as_dict` line 1124.
**What:** `_classify_anchors` includes a "skip empty anchor sets"
guard — the comment cites `list_dataset_members` returning
`{"File": []}` for empty association tables. Good. But the same
empty entry is included in `plan["anchors"]["by_type"]` in describe
(line 721: `anchors_by_type = {t: len(rids) for t, rids in
anchors.items()}` — will include `"File": 0`). Mild
inconsistency: describe reports the empty anchor; run skips it.

**Reproduction:**
```python
plan = d.describe(["Image", "Subject"])
plan["anchors"]["by_type"]  # may contain {"File": 0, "Image": 50}
```

**Impact:** Cosmetic; could confuse a careful reader.

**Suggested classification:** Add the same `if not rids: continue`
filter at line 721.

---

### RB-04 — `_run` step-4a's per-RID-orphan scan reads `f"{t}.RID"` keys from the materialised result with no validation

**Grill verdict:** confirmed; the `[inferred]` reasoning holds. Verified at line 1076 — `f"{t}.RID"` assumes single-schema labels; `denormalize_column_name` would produce `f"{schema}.{t}.RID"` under `multi_schema=True`. Multi-schema datasets would silently emit every anchor RID as an orphan.
**Disposition:** Code TODO — route through `denormalize_column_name` or assert single-schema.
**Revised severity:** Low (unchanged).

**Severity:** Low
**Component:** `denormalizer.py::Denormalizer._run`, lines 1069–1083.
**What:** The single-pass scan does `row.get(f"{t}.RID")`. If the
denormalize result for a multi-schema fixture uses `schema.Table.RID`
labels (which `denormalize_column_name` returns when `multi_schema`
is true — see `denormalize.py:289–322`), the scan looks up the wrong
key and silently returns `None` for every row. No per-RID orphans
are discovered, even when they should be.

[inferred] from reading both call sites; no explicit test covers the
multi-schema path through `_run`.

**Reproduction:** A `populated_denorm_multischema` fixture would
need to exist; it doesn't. The hazard is plausible from reading the
code.

**Impact:** Per-RID orphan emission silently incomplete on
multi-schema datasets.

**Suggested classification:** Either route through
`denormalize_column_name(...)` to construct the key, or assert
single-schema in `_run` and document the limitation.

---

### RB-05 — `Denormalizer.__init__` swallows exceptions from `ErmrestPagedClient` construction and silently falls back to `source="local"`

**Grill verdict:** confirmed. Verified line 185 broad-except. Silent-zero family member.
**Disposition:** Spec PR #231 §6.1 documents the fallback; code TODO — log WARNING + attach `_init_warning` for describe to report.
**Revised severity:** Medium (unchanged).

**Severity:** Medium
**Component:** `denormalizer.py::Denormalizer.__init__`, lines
180–189.
**What:** When a live `Dataset` is passed in, the init tries to
build an `ErmrestPagedClient`. On any exception, it sets
`self._source = "local"` silently. If the user is running against a
live catalog and the client construction fails for a transient
reason (e.g. auth not yet established), the denormalize call runs
against whatever is in the local engine — possibly an empty cache
— and returns zero rows with no warning.

**Reproduction:** Construct a Denormalizer right after instantiating
a DerivaML that hasn't fully authenticated yet. [inferred]

**Impact:** Silent zero-row result on auth/network setup races.

**Suggested classification:** Log at WARNING when falling back;
attach a `_init_warning` attribute that `describe` reports.

---

### RB-06 — `_run`'s `list_dataset_children(recurse=True)` is wrapped in `except Exception: pass`

**Grill verdict:** confirmed. Verified lines 1037–1046. Transient failure on live catalog silently falls back to root-only scoping.
**Disposition:** Spec PR #231 §6.2 documents the nested-dataset scoping; code TODO — log WARNING for `source="catalog"`, swallow for fixture-style DatasetLike.
**Revised severity:** Medium (unchanged).

**Severity:** Medium
**Component:** `denormalizer.py::Denormalizer._run`, lines 1036–1046.
**What:** Nested-dataset scoping pulls children RIDs from
`list_dataset_children(recurse=True)`. Any exception (network,
permissions, missing method) collapses to `dataset_children_rids =
None`, and the denormalize then runs root-only. If the user's data
lives in a child dataset and the children-list call has a transient
failure, the user gets zero rows from the children with no warning
— the silent-zero pattern again.

**Reproduction:** Mock `list_dataset_children` to raise; observe
denormalize returns root-only data.

**Impact:** Same class as RB-05.

**Suggested classification:** Log at WARNING; consider re-raising
for `source="catalog"` and only swallowing for fixture-style
`DatasetLike` objects that genuinely don't implement
`list_dataset_children`.

---

### RB-07 — `_emit_orphan_rows` does NOT use `_resolve_table_names`'s result; it uses raw `include_tables`

**Grill verdict:** **OVERSTATED** — auditor walked the code mid-finding and withdrew the main claim themselves (*"OK; orphan emission uses the resolved list. So this is correct"*). What remains is a Low de-dup concern for callers naming both `"Image_Classification"` (feature) and `"Execution_Image_Image_Classification"` (table) in one include list.
**Disposition:** Recategorize as "resolver should dedup after substitution" — one-liner in `_resolve_table_names`. Code TODO.
**Revised severity:** Low (unchanged for the residual concern).

**Severity:** Low
**Component:** `denormalizer.py::Denormalizer._emit_orphan_rows`,
lines 1271–1348.
**What:** Step 0 of `_run` (post-analyst/01 fix) resolves
`include_tables` from feature names to feature-association table
names. The resolved list is then passed to `_classify_anchors` and
to `_denormalize_impl`. But `_emit_orphan_rows` is called with the
already-resolved `include_tables` because `_run` shadows the
parameter. **Visual check:** `_run` reassigns `include_tables =
include_resolved`. OK; orphan emission uses the resolved list. So
this is correct.

However, `_emit_orphan_rows` itself calls
`self._model._planner._prepare_wide_table(...,
list(include_tables))` — that's the **resolved** list. Then it walks
the column specs and zips them with anchor RIDs. If a caller named
both `"Image_Classification"` (feature) and
`"Execution_Image_Image_Classification"` (table) in the same
include list, the resolver de-duplicates after substitution and the
orphan emission could see the same table twice with different
anchor RIDs. [inferred] hazard; no test pins it.

**Reproduction:** Construct a call with duplicate-after-resolution
input. None observed today because the common usage either uses
feature names or table names, not both.

**Impact:** Minor — duplicate orphan rows.

**Suggested classification:** De-duplicate the resolved list inside
`_resolve_table_names`, or assert no duplicates in `_run`.

---

### RB-08 — `_collect_fk_values`'s "first workable condition wins" loop silently picks one of multiple FK pairs

**Grill verdict:** confirmed; `[inferred]` reasoning holds for composite-FK schemas. Verified at lines 557–620 of denormalize.py — loop returns on the first non-empty value set. Latent on today's DerivaML schemas (single-column RID-based FKs dominate) but the failure mode (under-scoped fetch → wrong row count) is silent.
**Disposition:** Code TODO — either AND the conditions for multi-column predicate, or document the single-column FK assumption in spec.
**Revised severity:** Medium (unchanged).

**Severity:** Medium
**Component:** `denormalize.py::_collect_fk_values`, lines 557–620.
**What:** When the planner produces a join condition with multiple
`(fk_col, pk_col)` pairs (composite FK, multi-column join), the
loop returns on the **first** pair that produces non-empty values.
The other pairs are ignored. For single-column FKs (the dominant
case in DerivaML schemas) this is fine. For composite FKs it
produces an under-scoped fetch.

[inferred] DerivaML schemas largely use single-column RID-based
FKs, so this is latent.

**Reproduction:** A schema with a composite FK on the join path.

**Impact:** Under-scoped fetch → SQL join may produce wrong rows or
zero rows.

**Suggested classification:** Either AND the conditions together to
produce a multi-column predicate, or document that composite FKs are
unsupported.

---

### RB-09 — `fetch_predicate` page termination check `len(page) < page_size` is incorrect when the server returns a full page that happens to be the last

**Grill verdict:** **NOT A FINDING** — auditor explicitly said "Not a bug; flagged for completeness." Dropped from findings list.
**Disposition:** DROPPED.
**Revised severity:** n/a.

**Severity:** Low
**Component:** `paged_fetcher.py::fetch_predicate`, lines 197–204.
**What:** The loop breaks when a returned page is shorter than
`page_size`. If the server's last page is exactly `page_size` rows,
the loop fetches another page (which comes back empty) and breaks
on `if not page`. That's the `test_exact_page_boundary` test
intentionally pinning the 3-request behavior (lines 153–166 of
`test_paged_fetcher.py`). Correct, but the "extra request" cost is
unavoidable in this design — a minor performance concern at scale.

**Reproduction:** Not a bug; flagged for completeness.

**Impact:** Negligible.

**Suggested classification:** None.

---

### RB-10 — No dead code observed in the new contract; one piece of confused commentary remains

**Grill verdict:** confirmed. Dead `model` parameter on `_populate_from_catalog` (line 451 `_ = model`).
**Disposition:** Code TODO — one-line cleanup.
**Revised severity:** Low (unchanged).

**Severity:** Low (cleanliness)
**Component:** `denormalize.py::_populate_from_catalog`,
line 451: `_ = model`.
**What:** The function takes `model` as a parameter, doesn't use it,
and tags it with `_ = model` to silence the linter. Comment says
"part of the signature for future FK-aware insert ordering." Spec
§5 says FK enforcement is OFF for the whole load (via
`_foreign_keys_off`), so there is no need for FK-aware ordering. The
parameter is dead.

**Reproduction:** Read the code.

**Impact:** Cleanup, not a bug.

**Suggested classification:** Remove the parameter (and the comment).

---

## 5. Cross-cutting observations

1. **The "silent zero" pattern repeats.** SC-01, SC-02, RB-01, RB-05,
   RB-06, TC-05, and (historically) A02 all share the same shape:
   a code path that should produce data returns an empty result with
   no diagnostic. The project paid down A02 with the honest-`None`-
   plus-`reason` pattern in `describe.estimated_row_count`. That same
   pattern can be lifted out and applied to the other sites — every
   silent zero gets a `reason`. Cheap, high leverage.

2. **Three findings cluster around `_populate_from_catalog_inner`'s
   `processed` set.** SC-06 (the contract violation) and RB-02 (the
   robustness consequence) are the same bug; TC-01 and TC-03 are the
   missing tests. Fixing the key shape closes all four.

3. **The analyst/01 pattern hides at every dry-run/run boundary.**
   Every describe vs. run pair has its own potential asymmetry. TC-02
   suggests pinning ALL 12 keys of the describe envelope to a parity
   contract with what `_run` produces; the current suite pins only
   `plan["columns"]` and only on one input shape.

4. **Broad `except Exception` is the project's recurring footgun.**
   I counted at least 7 broad-except sites in `denormalizer.py`
   alone (lines 149–153, 184–189, 287, 297–303, 638, 652, 656, 668,
   677, 718, 796, 887, 909, 1042). Several swallow exceptions that
   should be diagnostic (e.g. `_resolve_table_names` failure in
   describe, ErmrestPagedClient construction in `__init__`, children
   enumeration in `_run`). The `describe` ones can be justified by
   the dry-run invariant; the `_run`-side ones cannot. The pattern
   needs a uniform replacement — `warnings.warn` or a `warnings: []`
   list on the returned envelope — applied across the file.

5. **The design doc is good; the code is several steps behind it on
   "must always re-issue the fetch."** Spec §6's exact language is
   in the doc; the implementation predates that wording. The
   `processed` set is the only material gap between spec and code I
   found beyond doc-rot. It's not yet a shipped bug only because
   CSA's plans happen to converge.

---

## 6. Recommendations

### Must fix before next release

- **R-1: Close SC-06 / RB-02 — fix `processed` set keying in
  `_populate_from_catalog_inner`.** Change the dedup key from
  `table_name` to `(table_name, rid_column, frozenset(rids))` (or
  equivalent). Add TC-03 (two-element-path live test) as the
  regression. *Addresses: SC-06, RB-02, TC-03, partly TC-01.*

- **R-2: Add a `warnings: list[str]` field to `describe()` and route
  every swallowed-exception site through it.** This is small, the
  dry-run-never-raises invariant is preserved, and the analyst/01-
  shape diagnostic loss is closed at every site at once. *Addresses:
  SC-01, RB-01, TC-09, partly TC-07.*

### High-leverage soon

- **R-3: Add the Layer D parity test (D.2) — multi-feature-per-anchor
  via both `Dataset.get_denormalized_as_dataframe` and
  `DatasetBag.get_denormalized_as_dataframe`.** This is the cheapest
  test that would catch SC-06 today and is on the spec's status
  list as a known gap. *Addresses: TC-01, TC-10, indirectly SC-06.*

- **R-4: Add the missing `xfail` test for spec §6's freshness
  caveat (server-side delete).** The design doc claims it exists;
  it doesn't. *Addresses: TC-04, partly SC-03.*

- **R-5: Either implement true streaming in `as_dict` or correct
  the docstring.** Pick one. Reading the docstring today is
  actively misleading. *Addresses: SC-07, TC-06.*

- **R-6: Make `from_rids` either reject `dataset_rid=None` against
  a live catalog or warn loudly.** Today the function smiles and
  hands back zero rows. *Addresses: SC-02, TC-05.*

### Nice to have

- **R-7: Pin all 12 describe keys to `_run`-side outputs.** A test
  per key that asserts "describe says X, run produces consistent
  Y." Cheap to write; deep coverage of the analyst/01 class.
  *Addresses: TC-02.*

- **R-8: Document or guard against the `_run` per-RID-orphan scan's
  hard-coded `f"{t}.RID"` key.** Either fail loudly on multi-schema
  or use `denormalize_column_name`. *Addresses: RB-04.*

- **R-9: Either AND `_collect_fk_values`'s composite-FK conditions
  or document the single-column FK assumption.** *Addresses: RB-08.*

- **R-10: Remove the unused `model` parameter from
  `_populate_from_catalog`** and its `_ = model` line. Trim the
  comment about future FK-aware ordering — spec §5 says the
  contract is no FK enforcement. *Addresses: RB-10.*

- **R-11: Tighten the docstrings on the planner-is-pure claim
  (SC-08) and the PagedFetcher-state claim (SC-04).** Doc-rot, not
  bugs. *Addresses: SC-04, SC-08.*

---

## Appendix — confidence and inference markers

- **Directly observed:** SC-01, SC-04, SC-05, SC-07, SC-08, RB-01,
  RB-03, RB-05, RB-06, RB-07, RB-09, RB-10; TC-01 through TC-04,
  TC-06, TC-07, TC-09, TC-10. Verified against file contents and
  spec text in this audit session.
- **[inferred] (reasoned from code shape, no repro built):** SC-02
  (production-catalog zero-row claim), SC-03 (no caller-visible
  staleness signal — read the entire code path looking for it),
  SC-06 / RB-02 (the constructed multi-path scenario — high
  confidence the dedup is wrong; have not built the live repro),
  RB-04 (multi-schema key shape), RB-08 (composite-FK behavior).
- **[inferred] from team context:** the "didn't surface on CSA at
  N=50" framing throughout (user supplied this; cross-channel parity
  was confirmed by the e2e arc).

End of audit.
