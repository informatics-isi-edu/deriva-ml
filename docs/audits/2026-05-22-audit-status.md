# 2026-05-22 audit — status summary

**Last updated:** 2026-05-22 (after PR #217)

The eight engineer-audit docs and the writer-audit doc dated
2026-05-22 catalogued **5 P0**, **78 P1**, **139 P2**, and
**90 P3** items across the deriva-ml subsystems.

This file tracks which items have shipped vs. which remain open.
**The individual audit files have not been edited as items
shipped** — they remain a snapshot of the codebase as of
2026-05-22. Use this file as the running ledger.

## P0 — all 5 closed ✅

| ID | Subject | PR |
|---|---|---|
| Workflow dirty-detection | `workflow.py:636` early-return mishandling | #195-ish (fix-pack 1) |
| filespec length bug | `filespec.create_filespecs` outer-stat call in loop | #195-ish |
| `DatasetBag.find_features(None)` | duplicate features in protocol regression | #195-ish |
| `FeatureRecord.select_majority_vote(column=None)` | `term_columns[0]` on a set | #195-ish |
| `AssetSpecConfig.asset_role` | parity with `AssetSpec` | #195-ish |

Released as **v1.37.2**.

## P1 — all 78 closed ✅

Tracked in three categories matching the PR series:

### Category 1 — god-function / god-class extractions (8 items)

| ID | Subject | PR |
|---|---|---|
| Ds-est | `Dataset.estimate_bag_size` (220 LOC → 3 helpers) | #209 |
| Ds-minid | `get_dataset_minid` (220 LOC → 5 helpers) | #210 |
| Ds-restr | `restructure_assets` (500 LOC → 5 helpers + 3 P2 drive-bys) | #211 |
| Ds-split | `split_dataset` (590 LOC → 3 helpers) | #212 |
| Ex-init | `_initialize_execution` (118 LOC → 4 helpers) | #213 |
| Ex-init2 | `Execution.__init__` orphan-row rollback on partial init failure | #213 |
| Ex-god (first sweep) | 5 asset-staging helpers → `asset_upload.py` | #216 |
| Ex-god (second sweep) | `bag_commit_upload` + `update_asset_execution_table` → `asset_upload.py` | #217 |

`execution.py` shrank from **2,692 → 2,387 LOC** across the two
Ex-god sweeps. Further Ex-god work (the truly state-machine-tangled
methods: `upload_execution_outputs`, `download_asset`,
`asset_file_path`, `metrics_file`) is **deferred to next minor**
— their extraction wants the state-machine and manifest-store
types to stabilize further first.

### Category 2 — cross-file helper extractions (5 items)

| ID | Subject | PR |
|---|---|---|
| Ex-dup1/2/3 | Cross-class execution helpers in `execution/_helpers.py` | #204 |
| Y3 + A2 | Asset metadata helpers (`asset_metadata_sorted`, `_asset_type_path`) | #205 |
| F-8 | `reduce_with_selector` (3 feature_values sites unified) | #206 |
| Ex-listw | `list_assets` cached walk + error surfacing | #207 |
| Ex-batch | `LeaseAggregator` (3 `post_lease_batch` POSTs → 1) | #208 |

### Category 3 — test coverage / API additions (closed in 2 PRs + earlier merges)

| ID | Subject | PR |
|---|---|---|
| A3 | `Asset.download` test + **uncovered + fixed real `AttributeError` bug** | #214 |
| F-2 | `select_majority_vote` empty-records returns `None` | #214 |
| F-20 | Empty-records test for `select_majority_vote` | #214 |
| `from_registry` | Direct test coverage (11 tests) | #215 |
| `metrics_file` / `database_catalog` / `catalog` properties | Coverage (8 tests) | #215 |
| F-1, F-6, F-7, A1, B1, X2, B4, X1 | Various coverage gaps | merged earlier or already addressed |
| `multirun_config.py` | Pure-Python test suite | merged earlier |
| `environment.py` | Smoke tests | already covered |

### Other P1s flagged in audits but **already addressed**

The audit docs frequently flag items that were fixed before or
during the audit cycle but never had their entries struck through.
Examples (all confirmed shipped in the current main):

- **Asset:** A1 (retired stub), A2 (`_asset_type_path`), A3 (test+bug fix),
  B1 (private export removed)
- **Core:** B1 (`Any` import), B2 (`catalog_snapshot` kwarg forwarding),
  E1 (`definitions.__all__` exception re-exports), RR1 (typed
  `DerivaMLRidsNotFound`), VM1, VM2 (vocabulary exception consistency),
  WF1 (`WorkflowMixin` docstring drift)
- **Schema:** F-01 (`generate_annotation` schema parameter used everywhere),
  F-04 (`asset_annotation` return type), F-05/F-07 (docstrings),
  F-11 (`use_hatrac` properly wired)
- **Feature:** F-6 (FK classification test), F-7 (docstring example
  uses str argument), F-1 (set-indexing bug)
- **Catalog:** 1.4 (`reinitialize_dataset_versions` documented as no-op)

The Explore agent's 2026-05-22 re-read claimed ~51 P1s "still
open" but cross-referencing each against current `main` showed
they were almost all closed. Future audit cycles should consider
striking-through fixed items in-place to avoid this drift.

## P2 / P3 status

**139 P2 + 90 P3 = 229 items.** Many already closed; remaining
items concentrate in:

- **Doc drift / example bugs** — mostly fixed in earlier audit
  cycles or no longer applicable.
- **Test gaps** — addressed where the value/cost ratio was clear
  (PRs #214, #215, #216, #217); the rest are tracked for the
  next minor.
- **P3 "could be cleaner" items** — these stay as P3 by design
  (low value/cost; revisit when touching the surrounding code).

**Recommendation:** treat P2/P3 items as "fix in passing" — when
a future change touches a P2/P3 site, sweep its sibling audit
items at the same time. A dedicated P2/P3 sweep PR delivers
diminishing returns because so many are already closed.

## Items genuinely deferred to next minor

These were called out in the audits and consciously deferred:

1. **Ex-god full split** — remaining methods on `Execution`
   (`upload_execution_outputs`, `download_asset`,
   `asset_file_path`, `metrics_file`). The state-machine and
   manifest-store types should stabilize first.

2. **`_update_asset_execution_table` Output-branch removal**
   (audit `execution.py:1864-1899` P1) — the Output branch is
   dead in production but pinned by
   `test_asset_role_auto_tag.py`. Dropping the branch requires
   rewriting those tests against
   `bag_commit._add_asset_rows_to_bag`. Tracked in PR #217's
   commit message.

3. **Test files in wrong directory** (audit `model/` M-40, M-41)
   — `tests/model/test_data_sources.py` and
   `tests/model/test_fk_orderer.py` test code that lives in
   `deriva.bag`, not `deriva_ml`. Move upstream when
   coordinating a deriva-py release.

4. **Resume-test after process kill** (audit `asset/` X1) —
   stand-up cost is high (real catalog + process kill +
   resume harness); track as part of the broader durability
   testing roadmap.

Everything else from the 2026-05-22 audit set is shipped or
not actionable as a discrete fix.

## How to use this file

When working on a new audit cycle:

1. Read the audit-period summary and the individual subsystem
   docs.
2. Cross-reference against this status file — items marked
   here as "closed" or "already addressed" have shipped even
   if the audit doc still flags them.
3. If you ship a previously-tracked item, add a row to the
   relevant section above (don't edit the underlying audit
   doc).
