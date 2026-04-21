# Upload Engine — deriva-py Integration Design

**Status:** Draft · **Date:** 2026-04-21 · **Subsystem:** Phase 2 S3

## 1. Goal

Replace the `_invoke_deriva_py_uploader` stub in the new upload engine with a
real uploader that delegates to `deriva.transfer.upload.deriva_upload.GenericUploader`,
wire `UploadJob.cancel()` through to the running uploader, and remove the two
kwargs (`bandwidth_limit_mbps`, `parallel_files`) that deriva-py does not
implement. Retry / restart / transfer-state persistence stays inside deriva-py;
deriva-ml's engine is only a dispatcher and a per-file status tracker on top of
SQLite.

## 2. Scope

**In scope (D+(i)):**
1. Real body for `_invoke_deriva_py_uploader` using `GenericUploader`
2. Cancellation plumbed from `UploadJob._cancel_event` to `GenericUploader.cancel()`
3. Hard-drop of `bandwidth_limit_mbps` and `parallel_files` across all 7
   surfaces that currently thread them

**Out of scope:**
- Parallel upload (deriva-py's `uploadFiles` is a sequential for-loop — changing
  this is upstream work)
- Bandwidth throttling (same — upstream doesn't implement it)
- Live byte-level progress on `UploadJob.progress()` beyond what the per-file
  callback naturally provides (tracked for S4 or later)
- Pause/resume (`UploadJob.pause` / `resume` stay as no-ops — deriva-py has a
  `HatracJobPaused` exception but no public pause primitive on the uploader)

## 3. Design Decisions

### 3.1 Invocation model — directory-scan per batch

`_invoke_deriva_py_uploader` receives a batch of files that share a
`target_table` and an `execution_rid`. It arranges those files under a scan
root whose directory layout matches deriva-py's regex-expected tree (the same
symlink-farm pattern already used by the manifest upload path — see CLAUDE.md
"Asset Manifest Architecture"), then drives deriva-py's public API:

```python
uploader = GenericUploader(
    config_file=<asset-mapping config>,
    credential_file=<deriva credential file>,
    server={"host": ml.host, "protocol": "https"},
    dcctx_cid="deriva-ml/upload_engine",
)
uploader.initialize(cleanup=False)
uploader.scanDirectory(scan_root, abort_on_invalid_input=True)
results = uploader.uploadFiles(
    status_callback=_status_cb,
    file_callback=_file_cb,
)
```

**Rationale:** Keeps retry / transfer-state / hatrac resumability inside
deriva-py, where it already lives. deriva-ml does not re-implement any of it.
The batching boundary in deriva-ml's engine becomes "how many files live under
one scan root" — an internal detail that does not leak into deriva-py.

### 3.2 Scan-root construction

For each batch, the engine materializes a temporary directory:

```
<tmp_root>/
  <regex-expected-subpath-for-target-table>/
    <symlink-to-input-file-1>
    <symlink-to-input-file-2>
    ...
```

The regex-expected subpath is the same one `_build_upload_staging` already
computes for the manifest-based upload. Files are hard-linked where the
filesystem supports it (os.link), symlinked otherwise (os.symlink). The root
is cleaned up on exit regardless of outcome (try/finally around the uploader
call).

**Why not pass the real staging root directly?** Because the new engine
processes files across multiple executions and target tables; a per-batch
temporary root lets us scope one `scanDirectory` call to exactly the files
we want to attribute to this batch.

### 3.3 Cancellation

Two check points:

1. **Between batches** — `run_upload_engine` polls `cancel_event.is_set()`
   before each `_invoke_deriva_py_uploader` call. If set, it stops dispatching
   and returns the partial `UploadReport` with all currently-uploaded files
   reflected.

2. **Inside a batch** — two callbacks share the cancel-observation role
   because deriva-py has two distinct hook types:
   - `status_callback()` fires at each file-boundary (no args, return value
     ignored). Used by deriva-ml to observe cancel-event and call
     `uploader.cancel()`.
   - `file_callback(**kwargs)` fires during an in-flight upload with byte
     progress. Used by deriva-ml to observe cancel-event, call
     `uploader.cancel()`, and return `-1` — the value hatrac treats as an
     abort signal to tear down the current transfer.

   The `GenericUploader.uploadFiles` loop checks `self.cancelled` at group
   and entry boundaries and exits.

**Resulting latency:** Cancel takes effect at the next file boundary OR at the
next byte-progress callback inside a running upload, whichever comes first.
For large files this can still be seconds of work in flight — that is
deriva-py's granularity and we do not try to improve on it here.

**UploadJob side:** `UploadJob.cancel()` remains as-is — it sets
`_cancel_event`. The engine is what translates that into deriva-py calls.
`UploadJob.status` transitions to `"cancelled"` when `cancel()` is called
(existing behavior).

### 3.4 Per-file status attribution

Two update paths, both idempotent, to close the "row stuck in Pending" failure
mode:

1. **Live updates via callback.** The engine installs a `status_callback`
   that closes over (a) the batch's manifest rows indexed by absolute path,
   (b) a set of paths already written to SQLite this batch, and (c) the
   uploader instance. Because `status_callback()` takes no arguments, it
   cannot be told "file X just finished" — instead, each invocation walks
   `uploader.file_status` (a dict keyed by absolute path), diffs against
   paths-already-written, and writes any newly-terminal rows. This is the
   same pattern deriva-py's own UI clients use. Terminal states map as:
   - `UploadState.Success` → `Uploaded` (hatrac server-side hash-dedup
     means files that were already present also land here — deriva-py
     does not expose a distinct "skipped" state)
   - `UploadState.Failed` → `Failed` with the status string as the error
   - `UploadState.Cancelled` / `Aborted` → leave as `Pending` (so a later
     run resumes)
   - `UploadState.Paused` / `Timeout` → leave as `Pending`

2. **Post-hoc reconciliation.** After `uploadFiles` returns, the engine walks
   `uploader.getFileStatusAsArray()` once more and applies the same mapping
   for any manifest row the callback did not touch. Writes are no-ops if the
   row's current status already matches.

### 3.5 Return shape

```python
def _invoke_deriva_py_uploader(
    *, ml, files, target_table, execution_rid, cancel_event, store
) -> dict:
    """Returns:
        {
            "uploaded": list[Path],   # absolute input paths, State=Success
            "failed":   list[dict],   # [{"path": Path, "error": str}]
        }
    """
```

Matches the existing stub's documented contract. Files that deriva-py
internally recognized as already-present still come back as `Success` via
hatrac hash-dedup and are listed under `uploaded`.

### 3.6 Hard-drop of `bandwidth_limit_mbps` and `parallel_files`

All seven surfaces that currently accept these kwargs are updated to remove
them. The CLI gets explicit `argparse` removal — the flags `--bandwidth-limit`
and `--parallel` go away. No deprecation-warning period because:

1. deriva-ml is pre-1.0; the API is not yet stable.
2. The kwargs never actually did anything — they were plumbed from CLI to
   engine and ignored. Users who relied on them were relying on a no-op.
3. A `TypeError: unexpected keyword argument` on the very first call after
   upgrade is a loud, self-explaining failure mode that users will diagnose
   in seconds.

**Affected surfaces:**

| File | Function | Change |
|---|---|---|
| `src/deriva_ml/core/mixins/execution.py` | `DerivaML.upload_pending` | Remove both kwargs |
| `src/deriva_ml/core/mixins/execution.py` | `DerivaML.start_upload` | Remove both kwargs |
| `src/deriva_ml/execution/execution.py` | `Execution.upload_outputs_v2` | Remove both kwargs |
| `src/deriva_ml/execution/execution_record_v2.py` | `ExecutionRecord.start_upload` | Remove both kwargs |
| `src/deriva_ml/execution/upload_job.py` | `UploadJob.__init__` | Remove both kwargs + delete attributes |
| `src/deriva_ml/execution/upload_engine.py` | `run_upload_engine` | Remove both kwargs + docstring lines |
| `src/deriva_ml/cli/upload.py` | argparse | Remove `--bandwidth-limit` and `--parallel` flags and their dests |

Docstrings and CHANGELOG note the breaking change.

## 4. Architecture

### 4.1 Call flow

```
ml.start_upload() / ml.upload_pending() / cli/upload.py
         │
         ▼
   UploadJob._run() (background thread)
         │
         ▼
  run_upload_engine(ml, execution_rids, retry_failed)
         │
         │ for execution_rid in executions:
         │   for target_table, batch in batches(execution_rid):
         │     if cancel_event.is_set(): break
         ▼
  _invoke_deriva_py_uploader(ml, files=batch, target_table, execution_rid,
                              cancel_event, store)
         │
         │ 1. build symlink scan_root
         │ 2. construct GenericUploader
         │ 3. uploader.scanDirectory(scan_root)
         │ 4. uploader.uploadFiles(status_callback=_status_cb,
         │                          file_callback=_file_cb)
         │    _status_cb / _file_cb: observe cancel_event → uploader.cancel()
         │    _status_cb: write newly-terminal rows to SQLite store
         │ 5. reconcile uploader.file_status → store
         │ 6. return {uploaded, failed}
         │ (finally: rmtree scan_root)
         ▼
   UploadReport aggregation
```

### 4.2 SQLite store interaction

`_invoke_deriva_py_uploader` receives the `store` handle. Writes happen in
two places (live callback + reconciliation), both through the existing
`ExecutionStateStore.update_pending_row` API with `status=PendingRowStatus.uploaded`
or `PendingRowStatus.failed`. No new store methods are needed.

Writes are idempotent by the `written_paths` set the callback maintains: a
path that has been written once is not rewritten, whether via callback or
reconciliation.

### 4.3 GenericUploader configuration

- `config_file`: the asset-mapping config path already computed by
  `_build_upload_staging`. This is the per-catalog upload-spec JSON
  (`asset_table_upload_spec`).
- `credential_file`: deriva's default (`~/.deriva/credential.json`) unless
  the test env overrides. deriva-py reads this lazily from
  `DERIVA_CREDENTIAL_FILE` or the default path.
- `server`: `{"host": ml.host, "protocol": "https"}`. Derived from `ml.host`.
- `dcctx_cid`: `"deriva-ml/upload_engine"` for provenance in server logs.

The uploader is constructed fresh per batch — it is not reused across
`_invoke_deriva_py_uploader` calls. This matches deriva-py's design (state
lives on the instance) and makes cancellation unambiguous (cancelling one
batch does not affect future batches).

## 5. Testing

### 5.1 Unit tests (no network)

- **`_invoke_deriva_py_uploader` happy path** — monkeypatch `GenericUploader`
  class with a fake that sets `file_status` to all-Success. Assert return
  value lists all inputs as uploaded. Assert SQLite rows marked Uploaded.
- **`_invoke_deriva_py_uploader` mixed outcomes** — fake uploader produces
  Success and Failed. Assert return dict splits correctly and SQLite rows
  reflect each state.
- **Callback-driven live updates** — fake uploader invokes
  `status_callback` between file transitions; assert each callback fires a
  SQLite write before the uploader returns.
- **Reconciliation path** — fake uploader skips calling `status_callback`
  for one file but leaves `file_status[path] = Success`. Assert that row
  is marked Uploaded after reconciliation.
- **Cancellation mid-batch** — fake uploader's `uploadFiles` enters a loop
  that checks `self.cancelled`. Test sets `cancel_event` after N files;
  assert `uploader.cancel()` was called and remaining rows stay Pending.
- **Cancellation between batches** — `run_upload_engine` with two batches;
  set `cancel_event` after the first returns. Assert second batch is never
  dispatched.
- **Scan root cleanup on exception** — fake uploader raises from
  `scanDirectory`; assert the temp directory is removed and a clear error
  propagates.
- **Kwarg-drop TypeError** — calling `run_upload_engine(..., parallel_files=4)`
  raises `TypeError`. Same for `bandwidth_limit_mbps`. Same at each of the
  7 surfaces.
- **CLI kwarg-drop** — `deriva-ml upload --parallel 4` exits with a non-zero
  status and argparse's "unrecognized arguments" message.

### 5.2 Integration test (catalog required)

One end-to-end test (gated on `DERIVA_HOST`) that:
1. Creates an execution with a real file in its staging root
2. Calls `ml.upload_pending()` (no kwargs)
3. Asserts the file's catalog row exists
4. Asserts the SQLite manifest row shows Uploaded

This is already exercised by existing S3 tests in the Phase 1 layer — the
spec requires that those tests continue to pass after `_invoke_deriva_py_uploader`
is real (rather than being skipped / monkeypatched).

## 6. Risks and Mitigations

**R1: deriva-py API drift.** `GenericUploader.uploadFiles` signature and
`file_status` internal dict are not documented as public API. Mitigation:
the contract used here is narrow (constructor, `initialize`, `scanDirectory`,
`uploadFiles`, `getFileStatusAsArray`, `cancel`, and the state enum constants
on `UploadState`). A unit test asserts each of these exists at import time,
so a future deriva-py breakage is caught at test time rather than runtime.

**R2: Symlink vs hardlink behavior on upload.** deriva-py opens files by path;
it does not care whether the path is a symlink or a hardlink to the source.
Tested with both in 5.1.

**R3: Transfer-state file collisions between batches.** deriva-py writes a
`.transfer-state.json` inside the scan root. Per-batch temp roots make this
trivially unique. No shared transfer state across batches is fine — each
batch is a fresh uploader instance.

**R4: Cancelled rows stuck Pending.** If a user cancels and never retries,
Pending rows never reach a terminal state. This is correct behavior — a
subsequent `upload_pending(retry_failed=True)` call re-attempts them. The
`upload-status` CLI command already surfaces Pending counts.

## 7. Rollout

This is a single worktree landing on `main` as one PR. No feature flag — the
stub is simply replaced. The breaking change (kwarg removal) is called out
in CHANGELOG with a one-line upgrade note:

> **Breaking:** `bandwidth_limit_mbps` and `parallel_files` keyword arguments
> were removed from `ml.upload_pending`, `ml.start_upload`,
> `Execution.upload_outputs_v2`, `run_upload_engine`, and `UploadJob`.
> deriva-py's uploader does not implement bandwidth throttling or parallel
> file uploads; the kwargs were accepted but had no effect. The CLI flags
> `--bandwidth-limit` and `--parallel` are likewise removed.

## 8. Open Questions

None at spec time. Any surprises during implementation will surface as
subagent questions.
