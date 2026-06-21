# Provenance Contract — Test Plan

Companion to [provenance-contract.md](provenance-contract.md). This is the
coverage matrix that makes "comprehensive" verifiable: every normative clause
of the contract maps to at least one test, with its substrate and status.

## Testing approach (settled)

- **TDD acceptance tests.** Tests are written ahead of (or alongside) the
  implementation and define done. Tests for not-yet-built features start as
  `xfail(strict=True)` and flip to pass when the feature lands.
- **Mixed substrate.**
  - **Live catalog** (`DERIVA_HOST`, fixtures `test_ml` /
    `catalog_with_datasets` / `populated_catalog`) for anything asserting an
    **edge, role, state transition, or attribution** — the contract is about
    what gets written to the catalog, and a mocked layer can pass while
    production breaks (the 2026-05-19 torch-adapter precedent).
  - **Pure-Python unit** for **shapes**: config classes, validator routing,
    `get_gpu_info()` degradation, the state-machine table.
- **No literals for RIDs or sentinels.** Every RID and every sentinel
  reference comes from a fixture/lookup, never a hard-coded string (CLAUDE.md
  RID discipline). Sentinel-dependent tests reference the sentinel via its
  accessor; until that accessor exists they are `xfail`.
- **Plan first, then code.** This matrix is reviewed before test code is
  committed; it is also the implementation's acceptance checklist.

## Status legend

- **EXISTS** — current test already covers this; reuse / extend, do not
  duplicate.
- **NEW-LIVE** — new test, live catalog.
- **NEW-UNIT** — new test, pure-Python.
- **XFAIL** — new test, written now, `xfail(strict=True)` until the
  implementation lands.
- **DONE** — implemented and passing in the suite (the feature landed; the
  test's `xfail` was removed).
- **PARTIAL** — the test's feature is implemented for the cases it asserts, but
  the spec clause has documented remaining extensions (named in the row's note).
- **TODO** — a spec clause with no test yet; a planned extension of an
  already-landed mechanism (e.g. an additional `audit_provenance` query).

Existing coverage was surveyed 2026-06-20; file references are to the current
test suite. Implementation status last updated 2026-06-20 after landing the
sentinel-seeding, no-input-check, and read-only-audit layers (F0, F2/F2b, G1,
G7, G8/G10).

---

## A. Execution state model & honest termination

| # | Clause (spec §) | Assertion | Substrate | Status | Notes |
|---|---|---|---|---|---|
| A1 | State machine transitions | `ALLOWED_TRANSITIONS` accepts every legal pair, rejects illegal | unit | EXISTS | `test_state_machine.py` — comprehensive |
| A2 | 7 status values, title-case, catalog round-trip | enum values + SQLite/catalog parsing | unit | EXISTS | `test_status_migration.py` |
| A3 | `update_status` valid/invalid + error kwarg | public API transitions; `error=` on Failed; warns on non-terminal | live+unit | EXISTS | `test_update_status.py` |
| A4 | Honest termination — blessed path reaches terminal | `with create_execution(...)` run ends Uploaded (or Stopped) | live | NEW-LIVE | end-to-end happy path; not currently pinned as a *contract* test |
| A5 | Stranded `Running`/`Pending_Upload` is a violation | a driven-then-abandoned exec is flagged by the audit (A-audit) | live | NEW-LIVE | depends on audit (G1) |
| A6 | Never-driven `Created` is a draft, NOT a violation | a `Created`-only exec is reported as draft/cleanup, not a contract breach | live | NEW-LIVE | the §Definitions ruling |
| A7 | `Stopped` tiebreaker | a `Stopped` exec that does not advance + idle is flagged; a `Stopped` probe is clean terminal | live | NEW-LIVE | heuristic on `last_activity`; assert audit treatment, not internal count |
| A8 | `__del__` best-effort abort | non-terminal exec GC'd in-process → Aborted (live session) | live | NEW-LIVE | hard to test deterministically; mark slow |

## B. Failed executions are first-class

| # | Clause | Assertion | Substrate | Status | Notes |
|---|---|---|---|---|---|
| B1 | Failure reason mandatory | a `Failed` exec has non-empty `Status_Detail`/error | live+unit | EXISTS (partial) | `test_update_status.py` pins `error=`; **NEW**: assert a `Failed` row with *empty* reason is an audit violation |
| B2 | Commit-path failure → Failed | `commit_output_assets` raising drives Pending_Upload→Failed with error | unit | EXISTS | `test_asset_upload.py::test_failure_transitions_to_failed` |
| B3 | Inputs recorded up to failure | a run that fails mid-materialization records `Dataset_Execution` edges for datasets that resolved before the failure | live | NEW-LIVE | the partial-input rule; needs a deliberately-failing input |
| B4 | Partial outputs recorded under Failed | outputs written before failure carry their role under the Failed exec | live | NEW-LIVE | |
| B5 | Failed/Aborted ≠ complete provenance | a Failed or Aborted artifact-producer does NOT satisfy the complete-provenance predicate | live | NEW-LIVE | the corrected predicate; assert audit/predicate result |

## C. "Consumed a dataset" = declared in `datasets=`

| # | Clause | Assertion | Substrate | Status | Notes |
|---|---|---|---|---|---|
| C1 | Each declared dataset → 1 pinned input edge | every `DatasetSpec` in `datasets=` yields exactly one version-pinned `Dataset_Execution` row | live | EXISTS | `test_dataset_execution_write_path.py`, `test_find_executions_by_dataset.py` |
| C2 | Undeclared dataset → no input edge | a dataset *not* in `datasets=` produces no `Dataset_Execution` row, even if its members were read | live | NEW-LIVE | the `6-09Z8` lesson — directly pins the motivating bug's correct behavior |
| C3 | Version pinning | `DatasetSpec(rid, version)` pins the edge to that `Dataset_Version` RID; bogus version excluded | live | EXISTS | `test_find_executions_by_dataset.py` |
| C4 | Role unambiguous | input lives only in `Dataset_Execution`; output only in `Dataset_Version.Execution`; never both | live | EXISTS | `test_dataset_execution_write_path.py::test_create_dataset_writes_no_dataset_execution_row` |
| C5 | Partial input on failure | (see B3) | live | NEW-LIVE | shared with B3 |

## D. Workflow reproducibility

| # | Clause | Assertion | Substrate | Status | Notes |
|---|---|---|---|---|---|
| D1 | Clean workflow → URL + checksum | committed file yields a well-formed URL and non-empty checksum | live | EXISTS | `test_dirty_workflow.py` |
| D2 | Dirty workflow refused (unless allow_dirty) | modified file raises; `allow_dirty` warns + captures anyway | live | EXISTS | `test_dirty_workflow.py` |
| D3 | Degenerate workflow on artifact-producer = violation | an artifact-producer whose workflow has an empty checksum is flagged by the audit; commit still warns-and-marks (not blocked) | live | NEW-LIVE | the enforcement-tone rule; needs a way to force a degenerate workflow |
| D4 | Probe may carry degenerate workflow | a probe (no artifact) with a degenerate workflow is NOT a violation | live | NEW-LIVE | |
| D5 | Checksum-based dedup | same URL+checksum resolves to the same Workflow row across two executions | live | NEW-LIVE | survey flagged this as zero-coverage |

## E. Inputs as assets / files / LocalFile

| # | Clause | Assertion | Substrate | Status | Notes |
|---|---|---|---|---|---|
| E1 | Input asset → Input edge | `assets=[AssetSpec(rid)]` yields `<Asset>_Execution` with `Asset_Role="Input"` + `Input_File` tag | live | EXISTS | `test_asset_role_contract.py` |
| E2 | Output asset → Output edge | uploaded output gets `Asset_Role="Output"` + `Output_File` tag | live | EXISTS | `test_asset_role_contract.py` |
| E3 | `add_files` can register an Input | `add_files(..., asset_role="Input")` writes a `File_Execution` Input edge | live | XFAIL | `add_files` hardcodes Output today; xfail until the param lands |
| E4 | `LocalFile` → registered File + Input edge | `LocalFile("/path")` in `assets=` mints a `File` row (computed MD5, tag: URL) and an Input edge; not downloaded | live | XFAIL | depends on `LocalFile` impl |
| E5 | Bare string is always a RID, never a path | a bare string in `assets=` resolves as a RID; a non-existent RID *fails*, never a path read | live+unit | XFAIL | the abuse-surface boundary; unit-test `validate_assets`, live-test resolution |
| E6 | `validate_assets` shape routing | `{path: ...}` dict/DictConfig → LocalFile; `{rid: ...}`/bare str → AssetSpec | unit | XFAIL | the hydra seam; pure-Python on the validator |
| E7 | `LocalFileConfig` round-trips via hydra | a `.yaml` with `{path: ...}` instantiates and flows to `ExecutionConfiguration.assets` | unit | XFAIL | mirror `AssetSpecConfig`; no `Literal` fields |
| E8 | File-RID input not downloaded | an `AssetSpec(rid=<file_rid>)` / tag:-URI input no-ops the Hatrac fetch | live | XFAIL | the download-skip |
| E9 | File MD5 computed from bytes | a registered `File`'s MD5 equals the hash of its bytes (not caller-supplied) | unit | EXISTS (partial) | `test_file.py::test_create_filespecs` covers FileSpec; extend for the input path |
| E10 | App reads input via `asset_paths` | a `LocalFile` input is readable inside the run through the same `asset_paths` surface as any asset | live | XFAIL | point-3 "looks like an asset" |

## F. Every artifact has a producer + the two sentinels

| # | Clause | Assertion | Substrate | Status | Notes |
|---|---|---|---|---|---|
| F0 | Sentinels seeded at catalog init | a fresh catalog carries the 3 sentinels; accessors resolve them idempotently; they survive `reset()` | live | **DONE** | `test_F0…`; seeded in `initialize_ml_schema._ensure_sentinels`, re-seeded by `CatalogManager.reset()` |
| F1 | New dataset version has a producer | `create_dataset` / new version sets a non-null `Dataset_Version.Execution` | live | EXISTS (partial) | implied by write-path tests; **NEW**: assert non-null explicitly |
| F2 | No-input artifact-producer → unknown-File sentinel input | an artifact-producer committing with no declared input gets an Input edge to the unknown-provenance File sentinel | live | **DONE** | `test_F2…` (dataset path) + `test_F2b…` (asset path); `provenance_enforcement.ensure_artifact_producer_has_input` wired at both write paths |
| F3 | Sentinel input is compliant, reported as known-degraded | the F2 exec satisfies the predicate (compliant) AND appears in the audit's known-degraded report (not violations) | live | PARTIAL | the audit's File-sentinel-only-input known-degraded clause is a documented remaining extension of `audit_provenance` |
| F4 | Producerless artifact → unknown-Execution sentinel | an artifact with no real producer attributes to the unknown-Execution sentinel; lineage from it returns "unknown origin", never null | live | **DONE** | `test_F4…` runs the adoption **backfill** then asserts `lookup_lineage` resolves to the sentinel; `provenance_backfill.py` + `scripts/provenance_backfill.py` |
| F4b | Backfill is dry-run-safe + idempotent | dry-run reports orphans but writes nothing; apply attributes + audit reclassifies violation→known-degraded; second apply is a no-op | live | **DONE** | `test_F4b…` |
| F4c | Backfill unifies prior BlackBox convention | a legacy BlackBox-attributed dataset version is re-pointed to the sentinel; the now-unreferenced BlackBox execution + workflow are deleted; the dataset reads as known-degraded | live | **DONE** | `test_F4c…`; `_unify_blackbox` (Move 3). Detects BlackBox by marker text, not RID |
| F5 | Sentinels are exempt bootstrap rows | the two sentinels are not themselves flagged as artifact-producers / null-producer artifacts by the audit | live | XFAIL | the exemption clause |
| F6 | Sentinel File MD5 exemption | the unknown-File sentinel's marker MD5 is accepted (not subject to computed-from-bytes) | unit | XFAIL | |
| F7 | Orphan dataset lineage (pre-sentinel) | a dataset with no producer returns an orphan lineage node | live+unit | EXISTS | `test_lookup_lineage_*` — current behavior; F4 supersedes once sentinels land |

## G. Audit & complete-provenance predicate

| # | Clause | Assertion | Substrate | Status | Notes |
|---|---|---|---|---|---|
| G1 | Audit exists & is read-only | the audit scans catalog-wide, returns findings, mutates nothing | live | **DONE** | `test_G1…`; `ml.audit_provenance()` → `ProvenanceAuditReport` (`provenance_audit.py`) |
| G2 | Predicate clause 1 — terminal state | audit flags an artifact-producer not in a terminal state | live | TODO | documented remaining clause; extends `audit_provenance` |
| G3 | Predicate clause 2 — workflow URL+checksum | audit flags an artifact-producer with degenerate workflow | live | TODO | shared with D3; remaining clause |
| G4 | Predicate clause 3 — ≥1 input or sentinel | audit flags an artifact-producer with zero inputs (no sentinel) | live | TODO | remaining clause |
| G5 | Predicate — output role-tagged | audit flags an artifact-producer with an untagged output | live | TODO | remaining clause |
| G6 | Predicate — Failed has reason | audit flags a `Failed` row with empty reason | live | TODO | shared with B1; remaining clause |
| G7 | Null-producer artifact flagged | audit flags an artifact with a null producer (not sentinel-attributed) | live | **DONE** | `test_G7…`; `category="null_producer"` |
| G8 | Known-degraded reported separately from violations | sentinel-attributed rows appear in a *separate* known-degraded list, not the violation list | live | **DONE** | `test_G8_G10…`; `category="sentinel_producer"` in `known_degraded` |
| G9 | Conformance is whole-catalog & falsifiable | a clean catalog → zero violations; introduce one violation → audit count increments | live | PARTIAL | falsifiability proven for the null-producer clause (G7); full whole-catalog guarantee tracks the remaining clauses |
| G10 | Sentinel-attributed state reads as conformant | a catalog whose orphan artifacts point at the unknown-Execution sentinel and whose stranded execs are Aborted → audit reports **zero violations** (sentinel/aborted state is compliant, known-degraded) | live | **DONE** (producer side) | `test_G8_G10…` covers the sentinel-attributed-producer half; the stranded-→-Aborted half tracks G2 |

## H. Lineage (data-flow, per ADR-0001)

| # | Clause | Assertion | Substrate | Status | Notes |
|---|---|---|---|---|---|
| H1 | Backward lineage walk | from an artifact, walk to producing exec → consumed inputs → their producers | live+unit | EXISTS | `test_lookup_lineage_unit.py`, `test_lookup_lineage_live.py` |
| H2 | Forward lineage | find executions that consumed an artifact as input | live | EXISTS | `test_forward_lineage.py` |
| H3 | Walks data-flow, not orchestration | lineage does NOT traverse `Execution_Execution` | unit | EXISTS (implicit) | **NEW**: add an explicit nested-execution case asserting the sibling is not surfaced as a parent |
| H4 | Depth / diamond / cap | depth control, diamond dedup, max-executions cap | unit | EXISTS | `test_lookup_lineage_unit.py` |
| H5 | Lineage terminates at sentinel | from a sentinel-attributed artifact, lineage reports "unknown origin" and stops | live | XFAIL | the F4 read-path |

## I. Execution metadata capture

| # | Clause | Assertion | Substrate | Status | Notes |
|---|---|---|---|---|---|
| I1 | `configuration.json` captured | resolved `ExecutionConfiguration` (+ argv) staged as Execution_Metadata | live+unit | EXISTS | `test_asset_upload.py`, migration tests |
| I2 | Environment snapshot captured | packages, CPU/arch/processor, OS, sys, env vars; JSON-serializable | unit | EXISTS | `test_environment.py` |
| I3 | `uv.lock` captured when present | lockfile staged when workflow git root has one | live | EXISTS (partial) | extend: assert conditional behavior |
| I4 | Hydra configs captured under hydra-zen | `config.yaml`/`overrides.yaml`/`hydra.yaml` staged | live+unit | EXISTS | `test_asset_upload.py::test_upload_hydra_config_assets` |
| I5 | Seeds captured via Hydra config | a seed set in config appears in the captured `config.yaml` | live | NEW-LIVE | the seeds-via-hydra ruling; lightweight |
| I6 | GPU info captured best-effort | env snapshot has a `gpu` section; degrades to `{available: false}` with no GPU/torch | unit | XFAIL | depends on `get_gpu_info()` |

## J. Adoption backfill (migration) — OUT OF THIS SUITE

The adoption backfill is a **one-off migration** (run once per catalog at
contract adoption, then never again). Its *mechanics* tests do **not** belong
in the permanent contract suite — re-running them on every CI pass exercises a
migration that has already happened. They live **with the backfill script**,
in their own file (the pattern the prior migration already follows:
`tests/test_migrate_dataset_execution_version.py` sits at `tests/` root, not
inside `tests/execution/`), and may be deleted once the migration has run on
all target catalogs.

Migration-mechanics tests (co-located with the script, NOT in the contract
suite): backfill attributes orphans to the unknown-Execution sentinel;
aborts stranded non-terminal executions with a migration reason; idempotent
re-run is a no-op; `--dry-run` mutates nothing; the migration records its own
provenance (date + touched rows). Prior art for all of these:
`test_migrate_dataset_execution_version.py`.

**What stays in the contract suite is the durable invariant, reframed as an
audit test:** that a catalog in the *post-backfill state* (orphans attributed
to the sentinel, stranded execs Aborted) reads as **conformant** — see
**G10**. That assertion is permanent: it guards against any future code path
re-introducing a null producer or leaving a sentinel-attributed row
mis-classified as a violation. It is reachable by seeding sentinel-attributed
and aborted rows directly, with no dependency on the migration script.

---

## Coverage summary

- **EXISTS (reuse/extend):** A1–A3, B1–B2, C1/C3/C4, D1–D2, E1–E2, E9, F7, H1–H4, I1–I4 — the low-level mechanism layer is well covered.
- **NEW-LIVE (writable now, no new impl):** A4–A8, B3–B5, C2, D3–D5, I5 — these test contract behaviors over the *existing* surface (state, edges, audit-of-current-state). Many depend on the audit (G), so sequence G first or assert raw catalog state directly.
- **XFAIL (write now, flip on impl):** E3–E8, E10, F2–F6, G1–G10, H5, I6 — every test tied to a not-yet-built deliverable (LocalFile, sentinels, audit, GPU).
- **Out of this suite:** adoption-backfill *mechanics* (one-off migration) live with the backfill script, not here; only the durable post-backfill conformance invariant (G10) stays.

## Sequencing

1. **Now:** write the EXISTS-extensions + NEW-LIVE that assert raw catalog
   state (don't need the audit) — A4–A8, B3–B5, C2, D5, F1, H3, I5. These run
   green against the current code where behavior already conforms, red where
   it doesn't (surfacing spec/impl gaps early).
2. **With the audit (G):** the predicate/audit tests, plus the NEW-LIVE that
   assert "is flagged" (A5, A7, B1-violation, D3).
3. **With each feature:** flip the matching XFAIL — LocalFile (E), sentinels
   (F), GPU (I6).

## File layout

- `tests/execution/test_provenance_contract.py` — A, B, C, D, F, H5 (the
  lifecycle/edge/sentinel contract, live).
- `tests/execution/test_local_file_input.py` — E (LocalFile, validate_assets
  routing, download-skip); split unit vs live within.
- `tests/execution/test_provenance_audit.py` — G (the audit predicate,
  including G10 the post-backfill conformance invariant).
- Extend existing files for EXISTS rows; do not duplicate.

**Not in this suite:** adoption-backfill mechanics ship with the backfill
script as a separate, one-off migration test file at `tests/` root (the
`test_migrate_dataset_execution_version.py` pattern) — deletable once the
migration has run. Only G10 (the durable invariant) lives here.
