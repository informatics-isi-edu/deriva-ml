# Bag-based `upload_engine` — scoping note

## Goal

Replace the legacy `GenericUploader`-based batch upload pipeline (the
CLI flow under `deriva-ml-upload`) with the bag-based pipeline that
already backs per-execution `commit_execution`.

This is the **multi-execution counterpart** to PR #103/#104. Those
landed the per-execution commit path (`Execution.upload_execution_outputs`
→ `_bag_commit_upload` → `build_execution_bag` + `load_execution_bag`).
This note scopes the batch-CLI variant.

## Current state

The legacy batch flow (today, on `main`):

```
deriva-ml-upload (cli/upload.py:80)
  → DerivaML.upload_pending()  (core/mixins/execution.py:778)
    → run_upload_engine()      (execution/upload_engine.py:236)
      → _invoke_deriva_py_uploader()  (line 599)
        → GenericUploader + Hatrac
      → pathBuilder.insert()   (line 580, for plain rows)
```

Non-blocking variant via `UploadJob` wraps the engine in a daemon
thread (`upload_job.py:54`).

## What the engine does that the per-execution bag path doesn't

The bag-commit path (PR #103) reads `ManifestStore` — the per-execution
asset+feature manifest. The upload engine reads `ExecutionStateStore`
(`state_store.py:167`) — a separate, multi-execution registry with
three SQLite tables: `execution_state__executions`,
`execution_state__pending_rows`, `execution_state__directory_rules`.

`ExecutionStateStore` is the source of truth for "what executions are
pending upload across this whole workspace." `ManifestStore` is
per-execution and doesn't span that boundary. The two stores are
**orthogonal** — there's no inheritance, and the engine never reads
the per-execution manifest.

The engine also knows how to:

- enumerate work across N executions (`_enumerate_work()`, line 59);
- drain pending rows in **topological FK order** (`level` column in
  `pending_rows`);
- recover from crashes by reading SQLite status and resuming from the
  next unterminated row;
- buffer writes in 64-file batches (`pending_writes[]`, line 668) with
  a post-upload reconciliation pass (line 832);
- maintain per-execution status transitions independently (Pending_Upload
  at start, Uploaded/Failed at finish — line 366, 434).

## Mapping the bag pipeline onto this

Most of the engine's machinery has a natural bag-pipeline analogue:

| Engine concern | Bag pipeline equivalent |
|---|---|
| Per-execution lease acquisition (`acquire_leases_for_execution`) | `build_execution_bag` already leases pending RIDs internally |
| FK topological drain order | `BagCatalogLoader` does its own FK-aware ordering at load time |
| `_invoke_deriva_py_uploader` | `load_execution_bag` (which wraps `BagCatalogLoader` — FK-aware, atomic) |
| Per-execution status transitions | `_bag_commit_upload` already does these |
| Crash recovery (resume mid-row) | Bag path is atomic at load time — no mid-stream resume, only per-execution retry. **Loses granularity** — see below. |

So the obvious orchestration shape is:

```
for execution_rid in pending_executions:
    bag_dir = build_execution_bag(execution, ...)
    try:
        report = load_execution_bag(execution, bag_dir, ...)
        mark_execution_uploaded(execution_rid)
    except Exception:
        mark_execution_failed(execution_rid)
```

In other words: the engine becomes a thin orchestrator over
`_bag_commit_upload`, applied N times across whatever
`ExecutionStateStore.list_pending_executions()` returns.

## Wrinkles that will bite mid-implementation

1. **`ExecutionStateStore` vs `ManifestStore` divergence.**
   The bag-commit path was built to read `ManifestStore`. The batch
   engine reads `ExecutionStateStore`. These are **separate SQLite
   tables** with overlapping but non-identical contents.

   For each pending execution the engine processes, the per-execution
   `ManifestStore` may or may not have entries. The engine's
   `pending_rows` table is authoritative for "what bytes need
   uploading"; `ManifestStore` is authoritative for "what assets are
   declared by this execution's code path."

   **Open question:** does the engine just delegate to
   `build_execution_bag` (which reads `ManifestStore`), trusting that
   the two stores are in sync? Or does it need to pass row metadata
   directly into the bag builder, bypassing `ManifestStore`?

   This is the structural decision the rewrite hinges on. Probably:
   the cleanest answer is to **unify the stores** — make
   `ExecutionStateStore` write through to `ManifestStore` (or vice
   versa) so the bag path always sees the same view regardless of
   how it was invoked. But that's a bigger refactor.

2. **Loss of per-row resume granularity.**
   The legacy engine resumes mid-execution: if 200 of 500 files
   uploaded before a crash, the next run picks up at file 201. The
   bag pipeline is **atomic at load time** — the unit of restart is
   "the whole execution's bag," not "the next pending row." Whether
   this matters depends on the typical failure mode in practice; for
   large multi-GB executions it might.

   Mitigations: the bag-load step itself is idempotent
   (`UPLOAD_IF_MISSING` on Hatrac, plus the URL-dedup from PR #104),
   so re-running an interrupted execution doesn't re-PUT the bytes
   that already landed. But it does re-build the bag from scratch
   each retry. For an execution with a few thousand assets that's
   not free.

3. **Progress reporting fidelity.**
   The engine has no per-file progress today (called out as a Phase
   2 limitation). The bag path emits per-asset staging events and
   per-table load events (PR #103). If the CLI surfaces a progress
   bar, the bag-based variant has **better** granularity than today's
   engine, not worse — a non-blocker, but worth verifying the CLI
   wiring picks it up.

4. **`UploadJob` threading model.**
   The non-blocking variant runs the engine in a daemon thread. The
   bag path uses `asyncio.run` (with `nest_asyncio` fallback inside
   notebook contexts — see `bag_commit.py:559`). Calling `asyncio.run`
   from inside a thread is fine but worth a sanity test once the
   shape is in place.

5. **Test coverage stays valid.**
   Existing tests: `test_upload_engine.py` (unit), `test_upload_public_api.py`
   (integration), `test_upload_engine_live_smoke.py` (live),
   `test_phase1_end_to_end.py`, `test_upload_cli.py` (CLI). Most of
   these should continue to pass with a swap; the assertions are
   about catalog state post-upload, not engine internals. The
   `test_upload_engine_live_smoke.py::test_bug_c_none_stringification_corrupts_non_string_metadata`
   xfail will likely flip to pass once we're off `_invoke_deriva_py_uploader`
   (the bag path sends real `None`, not the literal `"None"` string).

## Estimated shape of the work

Roughly four phases:

1. **Decide store unification.** Either bridge `ExecutionStateStore`
   ↔ `ManifestStore` at write time, or refactor `build_execution_bag`
   to accept either a manifest or a list of pending rows. The latter
   is more invasive but cleaner.

2. **Rewrite `run_upload_engine` as a thin loop** over
   `_bag_commit_upload` (or equivalent). This is the bulk of the
   work, perhaps 200–300 LoC of orchestration replacing ~700 LoC of
   per-row drain logic.

3. **Delete the dead legacy code:** `_invoke_deriva_py_uploader`,
   `bulk_upload_configuration`, `asset_table_upload_spec`,
   `null_sentinel_processor.py`, `upload_directory`, the
   `GenericUploader` symbol re-export in `dataset/upload.py`. That's
   another ~600 LoC.

4. **Update tests** to exercise the new path. Most should be no-op
   changes; the `bug_c` xfail flips to pass.

Net: bag-based engine is **smaller** than the legacy engine (the
heavy lifting moves into `BagCatalogLoader`, which we already trust)
but the store-unification decision in Phase 1 is the hard part.

## Open questions

- Does the workspace ever have executions where `ManifestStore` is
  empty but `ExecutionStateStore.pending_rows` has work? (i.e., is
  there a code path that registers rows without going through
  `asset_file_path` / `add_features`?) If yes, the bag builder must
  accept row metadata directly. If no, simple delegation works.
- The CLI flag surface (`--execution`, `--retry`, etc.) — how much
  of this is intrinsic to the engine vs. how much can stay in the
  CLI wrapper?
- Is there a path to keep the legacy engine around as a fallback
  (env-var-gated) during the transition, like a feature flag?
  Probably not worth it — the two storage models would diverge fast.

## Status

Scoping only. Not scheduled. This is the largest of the remaining
bag-commit follow-ups; estimated 1–2 weeks once started.

## References

- `bag-based-commit-execution.md` — sibling design for the
  per-execution path.
- ADR-0006: bag-oriented data movement.
- deriva-py's `docs/design/column-construction-dedup.md` — related
  upstream cleanup (lives in the deriva-py repo because the
  implementation work is there; deriva-ml is a pure consumer).
- PR #103 (deriva-ml#103): per-execution bag-commit landed.
- PR #104 (deriva-ml#104): URL-dedup + legacy `_upload_execution_dirs`
  removal — merged.
