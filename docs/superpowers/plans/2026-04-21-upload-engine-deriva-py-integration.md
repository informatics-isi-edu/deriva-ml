# Upload Engine — deriva-py Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `_invoke_deriva_py_uploader` stub with a real `GenericUploader`-driven implementation, wire cancellation from `UploadJob._cancel_event` through to the uploader, and remove the `bandwidth_limit_mbps` and `parallel_files` kwargs everywhere they appear (7 surfaces).

**Architecture:** Each call to `_invoke_deriva_py_uploader` materializes a per-batch symlink-farm scan root matching deriva-py's regex (same layout as `Execution._build_upload_staging`), constructs a fresh `GenericUploader`, and drives `scanDirectory + uploadFiles`. Two callbacks — `status_callback` (file-boundary, writes live SQLite status + observes cancel) and `file_callback` (byte-progress, observes cancel, returns -1 on abort) — plus a post-hoc reconciliation pass over `uploader.file_status`.

**Note on deriva-py's `UploadState`:** It is implemented as a `tuple` subclass, not a `Enum` — `UploadState.Success` evaluates to `0` (the tuple index) via `__getattr__ = tuple.index`. Comparisons like `state == UploadState.Success` still work because `state` is the numeric index. There is no `Skipped` value; hatrac's server-side hash-dedup makes already-present files succeed as `Success`.

**Tech Stack:** Python 3.12+, `deriva.transfer.upload.deriva_upload.GenericUploader`, SQLAlchemy Core (via `ExecutionStateStore`), existing deriva-ml dataset/upload.py helpers (`bulk_upload_configuration`, `DEFAULT_UPLOAD_TIMEOUT`).

## File Structure

| File | Role | Change |
|---|---|---|
| `src/deriva_ml/execution/upload_engine.py` | Engine entry + `_invoke_deriva_py_uploader` | Remove kwargs from `run_upload_engine`; replace stub with real body; add `cancel_event` param plumbing |
| `src/deriva_ml/execution/upload_job.py` | Background job wrapper | Remove kwargs; pass `self._cancel_event` into `run_upload_engine` |
| `src/deriva_ml/execution/execution.py` | `Execution.upload_outputs` | Remove kwargs |
| `src/deriva_ml/execution/execution_record_v2.py` | `ExecutionRecord.upload_outputs` | Remove kwargs |
| `src/deriva_ml/core/mixins/execution.py` | `DerivaML.upload_pending` and `start_upload` | Remove kwargs |
| `src/deriva_ml/cli/upload.py` | CLI entry point | Remove `--bandwidth-mbps` and `--parallel` argparse flags |
| `src/deriva_ml/execution/state_store.py` | SQLite store | (unchanged — existing `update_pending_row` handles all cases) |
| `tests/execution/test_upload_engine_deriva_py.py` | **NEW** | Unit tests for `_invoke_deriva_py_uploader` with a fake `GenericUploader` |
| `tests/execution/test_upload_engine.py` | Existing tests | Remove assertions about `bandwidth_limit_mbps` / `parallel_files`; update callers that pass them |
| `tests/cli/test_upload_cli.py` | CLI tests | Remove `--parallel`/`--bandwidth-mbps` test coverage; add "unrecognized arguments" test |
| `CHANGELOG.md` | Release notes | Breaking-change line |

## Task Order Rationale

1. **Task 1** plumbs a `cancel_event` parameter through to the engine. Must land first because later tasks need it.
2. **Task 2** adds the real `_invoke_deriva_py_uploader` body using `GenericUploader` with both callbacks, scan-root construction, reconciliation, and uses `cancel_event`. This is the bulk of the work.
3. **Task 3** removes the kwargs from all 7 surfaces as a single atomic commit — a half-removed state is a type-error cliff nobody wants to bisect across.
4. **Task 4** updates the existing test suite to match the new signatures and adds CLI-rejection coverage.
5. **Task 5** documents the breaking change in CHANGELOG.

---

### Task 1: Plumb `cancel_event` parameter into `run_upload_engine` and `_invoke_deriva_py_uploader`

**Why first:** Every subsequent task depends on `cancel_event` being a first-class parameter in the engine. Landing it before any other change keeps each commit clean.

**Files:**
- Modify: `src/deriva_ml/execution/upload_engine.py:225-398` (add `cancel_event` kwarg to `run_upload_engine`, plumb into `_drain_work_item`, plumb into `_invoke_deriva_py_uploader`)
- Modify: `src/deriva_ml/execution/upload_job.py:83-94` (pass `self._cancel_event` to `run_upload_engine`)
- Test: `tests/execution/test_upload_engine.py` (exercise new kwarg behavior)

- [ ] **Step 1.1: Write failing test — `run_upload_engine` accepts and honors `cancel_event`**

Add to `tests/execution/test_upload_engine.py`:

```python
import threading

def test_run_upload_engine_skips_batches_when_cancel_event_set(test_ml):
    """Cancel set before the run starts → no batches dispatched."""
    # Stage one asset row so there is work to do.
    exe = test_ml.create_execution(...)  # use existing fixture pattern
    # ... stage a pending asset row via the standard helper ...

    cancel_event = threading.Event()
    cancel_event.set()

    from deriva_ml.execution.upload_engine import run_upload_engine
    report = run_upload_engine(
        ml=test_ml,
        execution_rids=[exe.execution_rid],
        retry_failed=False,
        cancel_event=cancel_event,
    )
    # With cancel pre-set, no uploads should have occurred.
    assert report.total_uploaded == 0
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_upload_engine.py::test_run_upload_engine_skips_batches_when_cancel_event_set -v`

Expected: FAIL with `TypeError: run_upload_engine() got an unexpected keyword argument 'cancel_event'`.

- [ ] **Step 1.3: Add `cancel_event` parameter to `run_upload_engine`**

In `src/deriva_ml/execution/upload_engine.py` replace the `run_upload_engine` signature (lines 225-232) with:

```python
def run_upload_engine(
    *,
    ml: "DerivaML",
    execution_rids: "list[str] | None",
    retry_failed: bool = False,
    cancel_event: "threading.Event | None" = None,
) -> UploadReport:
```

Add `import threading` near the top of the file (after `import logging`).

Update the docstring Args section (replace the old `bandwidth_limit_mbps` / `parallel_files` lines with):

```
        cancel_event: If provided and .is_set(), the engine stops
            dispatching new batches before each batch and signals any
            in-flight GenericUploader via its cancel() primitive. If
            None, the engine runs to completion.
```

- [ ] **Step 1.4: Add between-batches cancel check**

In the drain loop starting around line 308 (`for level in levels:`) insert a cancel check at the start of each item dispatch. Replace:

```python
    for level in levels:
        level_had_failure = False
        for item in level:
            try:
                row = store.get_execution(item.execution_rid)
```

with:

```python
    for level in levels:
        level_had_failure = False
        for item in level:
            if cancel_event is not None and cancel_event.is_set():
                logger.info("upload: cancel_event set — stopping drain before next batch")
                break
            try:
                row = store.get_execution(item.execution_rid)
```

Also add a level-boundary check after the inner loop — replace:

```python
        if level_had_failure:
            # Abort at the level boundary — don't descend into
            # dependent levels whose parents couldn't drain.
            break
```

with:

```python
        if level_had_failure or (cancel_event is not None and cancel_event.is_set()):
            # Abort at the level boundary — don't descend into
            # dependent levels whose parents couldn't drain, and honor
            # cancel requests at level boundaries.
            break
```

- [ ] **Step 1.5: Thread `cancel_event` into `_drain_work_item` and `_invoke_deriva_py_uploader`**

Update `_drain_work_item` signature (around line 442) to accept `cancel_event`:

```python
def _drain_work_item(
    *,
    store: "ExecutionStateStore",
    catalog: "ErmrestCatalog",
    work_item: _WorkItem,
    ml: "DerivaML | None" = None,
    cancel_event: "threading.Event | None" = None,
) -> int:
```

Pass it into the `_invoke_deriva_py_uploader` call (around line 497-501):

```python
        result = _invoke_deriva_py_uploader(
            ml=ml, files=files,
            target_table=work_item.target_table,
            execution_rid=work_item.execution_rid,
            cancel_event=cancel_event,
        )
```

Update `_invoke_deriva_py_uploader` signature (around line 555-561) to accept `cancel_event` (the body still raises `NotImplementedError` at this step — Task 2 implements the body):

```python
def _invoke_deriva_py_uploader(
    *,
    ml: "DerivaML",
    files: list[dict],
    target_table: str,
    execution_rid: str,
    cancel_event: "threading.Event | None" = None,
) -> dict:
```

Update the drain loop call site to pass `cancel_event` into `_drain_work_item` (around line 330):

```python
                n = _drain_work_item(
                    store=store, catalog=ml.catalog,
                    work_item=item, ml=ml,
                    cancel_event=cancel_event,
                )
```

- [ ] **Step 1.6: Run test to verify it passes**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_upload_engine.py::test_run_upload_engine_skips_batches_when_cancel_event_set -v`

Expected: PASS.

- [ ] **Step 1.7: Wire `UploadJob._cancel_event` into `run_upload_engine`**

In `src/deriva_ml/execution/upload_job.py:83-94`, replace the `_run` method body:

```python
    def _run(self) -> None:
        try:
            self._report = run_upload_engine(
                ml=self._ml,
                execution_rids=self._execution_rids,
                retry_failed=self._retry_failed,
                cancel_event=self._cancel_event,
            )
            self.status = (
                "completed" if self._report.total_failed == 0 else "failed"
            )
        except BaseException as exc:  # noqa: BLE001 — surface via wait()
            logger.warning("upload job %s errored: %s", self.id, exc)
            self._exception = exc
            self.status = "failed"
```

(The `bandwidth_limit_mbps` and `parallel_files` kwargs are still on `UploadJob.__init__` at this point — Task 3 removes them. For this task we stop *using* them; they sit as unused attributes for one task.)

- [ ] **Step 1.8: Run full test suite to verify no regressions**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/ -v`

Expected: All tests pass. Any test that passed `cancel_event=...` before (there aren't any — it's new) will still pass; existing tests that don't pass it will keep working because the default is `None`.

- [ ] **Step 1.9: Commit**

```bash
git add src/deriva_ml/execution/upload_engine.py \
        src/deriva_ml/execution/upload_job.py \
        tests/execution/test_upload_engine.py
git commit -m "feat(upload): plumb cancel_event through run_upload_engine

UploadJob._cancel_event now reaches run_upload_engine, which
observes it at level and batch boundaries. _invoke_deriva_py_uploader
receives cancel_event in its signature; the stub body still raises
NotImplementedError — Task 2 replaces it with the real uploader that
also observes the event inside GenericUploader callbacks."
```

---

### Task 2: Implement `_invoke_deriva_py_uploader` with GenericUploader + callbacks + reconciliation

**Why second:** Task 1 plumbed the parameter; now we fill the body.

**Files:**
- Modify: `src/deriva_ml/execution/upload_engine.py:555-585` (replace stub body)
- Modify: `src/deriva_ml/execution/upload_engine.py` (add imports for `GenericUploader`, `UploadState`, `TemporaryDirectory`, `shutil`, `json`)
- Modify: `src/deriva_ml/execution/upload_engine.py` (simplify `_drain_work_item`'s asset branch to note that per-file SQLite writes already happened in the uploader callbacks)
- Test: `tests/execution/test_upload_engine_deriva_py.py` (NEW)

**Key constraints from spec:**
- Scan root layout must match `asset_table_upload_spec`'s `file_pattern` — same as `_build_upload_staging`: `<schema>/<AssetTable>/<sorted_metadata_cols_values>/<filename>`.
- Config file: use `bulk_upload_configuration(ml._model)` rendered to JSON in a temp directory (exactly like `upload_directory` does at `src/deriva_ml/dataset/upload.py:515-518`).
- Server dict: `{host, protocol, catalog_id, session}` (same as `upload_directory` at lines 530-538).
- Callbacks close over a `state` dict: `{manifest_rows_by_path, written_paths, uploader}`.

- [ ] **Step 2.1: Write failing unit test — happy path with fake GenericUploader**

Create `tests/execution/test_upload_engine_deriva_py.py`:

```python
"""Unit tests for _invoke_deriva_py_uploader using a fake GenericUploader.

Exercises the uploader integration without a real catalog. The fake
simulates GenericUploader's interface just deeply enough to test
scan-root layout, callback wiring, cancel propagation, and result
attribution.
"""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class FakeGenericUploader:
    """Test double matching GenericUploader's public surface used by S3."""

    instances: list["FakeGenericUploader"] = []

    def __init__(self, *, server=None, config_file=None,
                 credential_file=None, dcctx_cid=None):
        self.server = server
        self.config_file = config_file
        self.credential_file = credential_file
        self.dcctx_cid = dcctx_cid
        self.scanned_root: Path | None = None
        self.cancelled = False
        self.file_status: dict[str, dict] = {}
        # Test harness pokes these:
        self._scan_result_files: list[Path] = []
        self._per_file_states: dict[str, str] = {}  # path → "Success"/"Failed"
        self._on_upload: callable | None = None
        FakeGenericUploader.instances.append(self)

    def initialize(self, cleanup=False):
        pass

    def getUpdatedConfig(self):
        pass

    def scanDirectory(self, root, abort_on_invalid_input=False, purge_state=False):
        self.scanned_root = Path(root)
        # Walk the scan root and record the files the test said to expect.
        for p in Path(root).rglob("*"):
            if p.is_file() or p.is_symlink():
                self._scan_result_files.append(p.resolve())

    def cancel(self):
        self.cancelled = True

    def cleanup(self):
        pass

    def uploadFiles(self, status_callback=None, file_callback=None):
        """Simulate uploadFiles — iterate scanned files, apply pre-seeded
        per-file states, invoke callbacks before/after each."""
        for f in self._scan_result_files:
            if self.cancelled:
                self.file_status[str(f)] = {"State": 5, "Status": "Cancelled by user"}
                break
            # status_callback fires BEFORE each file
            if status_callback:
                status_callback()
            # Apply pre-seeded state
            state_name = self._per_file_states.get(str(f), "Success")
            # UploadState is a tuple: index 0 = Success, 1 = Failed,
            # 2 = Pending, 3 = Running, 4 = Paused, 5 = Aborted,
            # 6 = Cancelled, 7 = Timeout.
            state_code = {"Success": 0, "Failed": 1}[state_name]
            self.file_status[str(f)] = {
                "State": state_code,
                "Status": f"{state_name}",
                "Result": {"url": "mock"} if state_name == "Success" else None,
            }
            if self._on_upload:
                self._on_upload(f)
            # status_callback fires AFTER each file
            if status_callback:
                status_callback()
        return self.file_status

    def getFileStatusAsArray(self):
        return [{"File": k, **v} for k, v in self.file_status.items()]


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeGenericUploader.instances.clear()
    yield
    FakeGenericUploader.instances.clear()


def test_invoke_uploader_happy_path(monkeypatch, tmp_path, test_ml):
    """All files succeed → returned 'uploaded' lists them; SQLite marked Uploaded."""
    from deriva_ml.execution import upload_engine as ue
    monkeypatch.setattr(ue, "GenericUploader", FakeGenericUploader)

    # Stage two asset files on disk
    f1 = tmp_path / "a.txt"; f1.write_text("a")
    f2 = tmp_path / "b.txt"; f2.write_text("b")

    files = [
        {"path": str(f1), "rid": "R1", "pending_id": 1, "metadata": {}},
        {"path": str(f2), "rid": "R2", "pending_id": 2, "metadata": {}},
    ]

    result = ue._invoke_deriva_py_uploader(
        ml=test_ml, files=files,
        target_table="SomeAsset",
        execution_rid="EXE-X",
        cancel_event=None,
    )

    assert sorted(result["uploaded"]) == sorted([str(f1), str(f2)])
    assert result["failed"] == []
    # Exactly one fake uploader instance was constructed
    assert len(FakeGenericUploader.instances) == 1
    u = FakeGenericUploader.instances[0]
    # Scan root was under a temp dir (not tmp_path)
    assert u.scanned_root is not None
    assert u.scanned_root != tmp_path
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_upload_engine_deriva_py.py::test_invoke_uploader_happy_path -v`

Expected: FAIL with `NotImplementedError` (the stub body) or `AttributeError: module ... has no attribute 'GenericUploader'` (the import doesn't exist yet).

- [ ] **Step 2.3: Implement the real `_invoke_deriva_py_uploader` body**

Replace the stub body in `src/deriva_ml/execution/upload_engine.py:555-585` with the real implementation. First, add imports near the top of the file (with existing imports, after the `from deriva_ml.execution.state_store import ...` line):

```python
import json
import os
import shutil
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

from deriva.core import DEFAULT_SESSION_CONFIG
from deriva.transfer.upload.deriva_upload import GenericUploader
from deriva.transfer.upload.deriva_upload import UploadState

from deriva_ml.dataset.upload import bulk_upload_configuration, DEFAULT_UPLOAD_TIMEOUT
```

Now replace the stub:

```python
def _invoke_deriva_py_uploader(
    *,
    ml: "DerivaML",
    files: list[dict],
    target_table: str,
    execution_rid: str,
    cancel_event: "threading.Event | None" = None,
) -> dict:
    """Invoke deriva-py's uploader for a batch of files.

    Builds a per-batch symlink-farm scan root whose layout matches
    `asset_table_upload_spec`'s regex, constructs a fresh
    `GenericUploader`, and drives `scanDirectory + uploadFiles`.

    Two callbacks are wired:

    - status_callback(): fires at each file boundary with no args.
        Walks uploader.file_status, writes newly-terminal rows to the
        SQLite store, observes cancel_event and calls uploader.cancel().
    - file_callback(**kw): fires during in-flight uploads with byte
        progress. Observes cancel_event; returns -1 to signal hatrac to
        abort the current transfer when cancelled.

    After uploadFiles returns, a reconciliation pass over
    uploader.getFileStatusAsArray() catches any file the callback
    missed and writes its terminal state.

    Args:
        ml: DerivaML instance (for model, host, catalog_id, workspace).
        files: List of dicts with keys 'path', 'rid', 'pending_id',
            'metadata'. All entries share target_table and execution_rid.
        target_table: Name of the target asset table (no schema).
        execution_rid: Execution RID these files belong to.
        cancel_event: Optional cancellation signal.

    Returns:
        {
            "uploaded": list[str]       # absolute input paths (Success)
            "failed":   list[dict]      # [{"path": str, "error": str}]
        }
    """
    if not files:
        return {"uploaded": [], "failed": []}

    store = ml.workspace.execution_state_store()

    # Resolve schema for the target table (scan root layout needs it).
    try:
        table_obj = ml._model.name_to_table(target_table)
        schema_name = table_obj.schema.name
        metadata_cols = sorted(ml._model.asset_metadata(target_table))
    except Exception as exc:
        # Fall back: treat target_table as already schema-qualified or
        # propagate the failure with clear context.
        raise DerivaMLException(
            f"Unable to resolve asset table {target_table!r}: {exc}"
        ) from exc

    # Map absolute input path → the file dict (for callback writes).
    rows_by_path: dict[str, dict] = {str(Path(f["path"]).resolve()): f for f in files}
    written_paths: set[str] = set()

    with TemporaryDirectory(prefix="deriva-ml-upload-") as scan_root_str:
        scan_root = Path(scan_root_str)

        # Build the symlink farm: <scan_root>/<schema>/<table>/<md1>/.../<filename>
        for f in files:
            src = Path(f["path"]).resolve()
            metadata = f.get("metadata") or {}
            target_dir = scan_root / schema_name / target_table
            for col in metadata_cols:
                target_dir = target_dir / str(metadata.get(col, "None"))
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / src.name
            if target.exists():
                continue
            try:
                os.link(src, target)  # hardlink where possible
            except (OSError, NotImplementedError):
                try:
                    target.symlink_to(src)
                except OSError:
                    shutil.copy2(src, target)

        # Build config file (same shape upload_directory uses).
        spec_file = scan_root / "config.json"
        spec_file.write_text(json.dumps(bulk_upload_configuration(ml._model)))

        session_config = DEFAULT_SESSION_CONFIG.copy()
        session_config["timeout"] = DEFAULT_UPLOAD_TIMEOUT

        uploader = GenericUploader(
            server={
                "host": ml._model.hostname,
                "protocol": "https",
                "catalog_id": ml._model.catalog.catalog_id,
                "session": session_config,
            },
            config_file=spec_file,
            dcctx_cid="deriva-ml/upload_engine",
        )

        # Map scan-root path (what uploader.file_status uses as keys)
        # back to original input path for SQLite attribution.
        scan_path_to_input: dict[str, str] = {}
        for abs_input, _ in rows_by_path.items():
            src = Path(abs_input)
            # The uploader's file_status key is the path it scanned —
            # that's the staged hardlink/symlink, not the original.
            # We rebuild the expected key here.
            metadata = rows_by_path[abs_input].get("metadata") or {}
            target_dir = scan_root / schema_name / target_table
            for col in metadata_cols:
                target_dir = target_dir / str(metadata.get(col, "None"))
            scan_path_to_input[str(target_dir / src.name)] = abs_input

        def _apply_state_to_sqlite(scan_path: str, state_info: dict) -> None:
            """Idempotent: translate one uploader status dict to a SQLite write."""
            if scan_path in written_paths:
                return
            input_path = scan_path_to_input.get(scan_path)
            if input_path is None:
                return
            row = rows_by_path.get(input_path)
            if row is None:
                return
            state = state_info.get("State")
            status = state_info.get("Status", "")
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            # Map deriva-py UploadState codes → SQLite status.
            if state == UploadState.Success:
                store.update_pending_row(
                    row["pending_id"],
                    status=PendingRowStatus.uploaded,
                    uploaded_at=now,
                )
                written_paths.add(scan_path)
            elif state == UploadState.Failed:
                store.update_pending_row(
                    row["pending_id"],
                    status=PendingRowStatus.failed,
                    error=status or "upload failed",
                )
                written_paths.add(scan_path)
            # UploadState.Cancelled/Aborted/Paused/Timeout: leave Pending.

        def status_callback() -> None:
            # Observe cancel first.
            if cancel_event is not None and cancel_event.is_set():
                uploader.cancel()
            # Walk file_status, write any newly-terminal rows.
            for scan_path, info in list(uploader.file_status.items()):
                _apply_state_to_sqlite(scan_path, info)

        def file_callback(**kwargs) -> bool | int:
            if cancel_event is not None and cancel_event.is_set():
                uploader.cancel()
                return -1  # hatrac abort signal
            return True

        try:
            uploader.initialize(cleanup=False)
            uploader.getUpdatedConfig()
            uploader.scanDirectory(scan_root, abort_on_invalid_input=True)
            uploader.uploadFiles(
                status_callback=status_callback,
                file_callback=file_callback,
            )

            # Reconciliation pass — catch anything the callback missed.
            for scan_path, info in uploader.file_status.items():
                _apply_state_to_sqlite(scan_path, info)
        finally:
            try:
                uploader.cleanup()
            except Exception:
                pass

        # Build return dict by walking final file_status one more time.
        uploaded: list[str] = []
        failed: list[dict] = []
        for scan_path, info in uploader.file_status.items():
            input_path = scan_path_to_input.get(scan_path)
            if input_path is None:
                continue
            state = info.get("State")
            if state == UploadState.Success:
                uploaded.append(input_path)
            elif state == UploadState.Failed:
                failed.append({
                    "path": input_path,
                    "error": info.get("Status") or "upload failed",
                })
        return {"uploaded": uploaded, "failed": failed}
```

Also simplify `_drain_work_item`'s asset branch (around lines 487-524) — `_invoke_deriva_py_uploader` now writes per-row SQLite status via its callbacks, so the drain wrapper only needs to compute the aggregate uploaded count. Replace:

```python
        uploaded_paths = set(result["uploaded"])
        for r in rows:
            path = r["asset_file_path"]
            pid = r["id"]
            if path in uploaded_paths:
                store.update_pending_row(
                    pid,
                    status=PendingRowStatus.uploaded,
                    uploaded_at=now,
                )
            else:
                failure_msg = next(
                    (f.get("error", "upload failed") for f in result.get("failed", [])
                     if f.get("path") == path),
                    "upload failed",
                )
                store.update_pending_row(
                    pid,
                    status=PendingRowStatus.failed,
                    error=failure_msg,
                )

        return sum(1 for r in rows if r["asset_file_path"] in uploaded_paths)
```

with:

```python
        # NOTE: _invoke_deriva_py_uploader has already written per-row
        # SQLite status via its callbacks. We only need the aggregate
        # count here. Rows whose status is still Pending at this point
        # fall through and will be retried on the next run.
        uploaded_paths = set(result["uploaded"])
        return sum(
            1 for r in rows
            if r["asset_file_path"] in uploaded_paths
        )
```

- [ ] **Step 2.4: Add the other unit tests from spec §5.1**

Append to `tests/execution/test_upload_engine_deriva_py.py`:

```python
def test_invoke_uploader_mixed_outcomes(monkeypatch, tmp_path, test_ml):
    """Success and Failed files split correctly into return dict keys."""
    from deriva_ml.execution import upload_engine as ue
    monkeypatch.setattr(ue, "GenericUploader", FakeGenericUploader)

    f1 = tmp_path / "ok.txt"; f1.write_text("ok")
    f2 = tmp_path / "bad.txt"; f2.write_text("bad")

    files = [
        {"path": str(f1), "rid": "R1", "pending_id": 1, "metadata": {}},
        {"path": str(f2), "rid": "R2", "pending_id": 2, "metadata": {}},
    ]

    # Hook scanDirectory to assign per-file states by basename.
    original_scan = FakeGenericUploader.scanDirectory
    def scan_patch(self, root, **kw):
        original_scan(self, root, **kw)
        for p in self._scan_result_files:
            self._per_file_states[str(p)] = {
                "ok.txt": "Success",
                "bad.txt": "Failed",
            }.get(p.name, "Success")
    monkeypatch.setattr(FakeGenericUploader, "scanDirectory", scan_patch)

    result = ue._invoke_deriva_py_uploader(
        ml=test_ml, files=files,
        target_table="SomeAsset", execution_rid="EXE-X",
        cancel_event=None,
    )
    assert result["uploaded"] == [str(f1)]
    assert len(result["failed"]) == 1
    assert result["failed"][0]["path"] == str(f2)


def test_invoke_uploader_cancel_mid_batch(monkeypatch, tmp_path, test_ml):
    """cancel_event.set() during a batch → uploader.cancel() called; remaining files not attributed."""
    from deriva_ml.execution import upload_engine as ue
    monkeypatch.setattr(ue, "GenericUploader", FakeGenericUploader)

    cancel_event = threading.Event()

    f1 = tmp_path / "a.txt"; f1.write_text("a")
    f2 = tmp_path / "b.txt"; f2.write_text("b")
    f3 = tmp_path / "c.txt"; f3.write_text("c")

    files = [
        {"path": str(f1), "rid": "R1", "pending_id": 1, "metadata": {}},
        {"path": str(f2), "rid": "R2", "pending_id": 2, "metadata": {}},
        {"path": str(f3), "rid": "R3", "pending_id": 3, "metadata": {}},
    ]

    # After f1 uploads, fire cancel.
    original_scan = FakeGenericUploader.scanDirectory
    def scan_patch(self, root, **kw):
        original_scan(self, root, **kw)
        # Set up an on_upload hook that fires cancel after the first file.
        count = {"n": 0}
        def _after(_):
            count["n"] += 1
            if count["n"] == 1:
                cancel_event.set()
        self._on_upload = _after
    monkeypatch.setattr(FakeGenericUploader, "scanDirectory", scan_patch)

    result = ue._invoke_deriva_py_uploader(
        ml=test_ml, files=files,
        target_table="SomeAsset", execution_rid="EXE-X",
        cancel_event=cancel_event,
    )
    u = FakeGenericUploader.instances[0]
    assert u.cancelled is True
    # f1 succeeded before cancel; f2/f3 never dispatched.
    assert str(f1) in result["uploaded"]
    assert str(f2) not in result["uploaded"]
    assert str(f3) not in result["uploaded"]


def test_invoke_uploader_reconciliation(monkeypatch, tmp_path, test_ml):
    """If status_callback misses a file, the reconciliation pass catches it."""
    from deriva_ml.execution import upload_engine as ue
    monkeypatch.setattr(ue, "GenericUploader", FakeGenericUploader)

    f1 = tmp_path / "only.txt"; f1.write_text("only")
    files = [{"path": str(f1), "rid": "R1", "pending_id": 1, "metadata": {}}]

    # Make uploadFiles skip the status_callback entirely but still
    # populate file_status.
    def no_callback_upload(self, status_callback=None, file_callback=None):
        for f in self._scan_result_files:
            self.file_status[str(f)] = {
                "State": 0,  # Success
                "Status": "Complete",
                "Result": {"url": "mock"},
            }
        return self.file_status
    monkeypatch.setattr(FakeGenericUploader, "uploadFiles", no_callback_upload)

    result = ue._invoke_deriva_py_uploader(
        ml=test_ml, files=files,
        target_table="SomeAsset", execution_rid="EXE-X",
        cancel_event=None,
    )
    assert result["uploaded"] == [str(f1)]


def test_invoke_uploader_cleans_up_scan_root_on_exception(monkeypatch, tmp_path, test_ml):
    """Exception during scanDirectory → temp directory removed."""
    from deriva_ml.execution import upload_engine as ue
    monkeypatch.setattr(ue, "GenericUploader", FakeGenericUploader)

    f1 = tmp_path / "a.txt"; f1.write_text("a")
    files = [{"path": str(f1), "rid": "R1", "pending_id": 1, "metadata": {}}]

    recorded_roots: list[Path] = []
    def boom(self, root, **kw):
        recorded_roots.append(Path(root))
        raise RuntimeError("scan exploded")
    monkeypatch.setattr(FakeGenericUploader, "scanDirectory", boom)

    with pytest.raises(RuntimeError, match="scan exploded"):
        ue._invoke_deriva_py_uploader(
            ml=test_ml, files=files,
            target_table="SomeAsset", execution_rid="EXE-X",
            cancel_event=None,
        )

    # TemporaryDirectory context manager should have cleaned up.
    assert recorded_roots
    assert not recorded_roots[0].exists()


def test_invoke_uploader_empty_files_noop(test_ml):
    """Empty files list → no uploader constructed, empty result."""
    from deriva_ml.execution import upload_engine as ue
    result = ue._invoke_deriva_py_uploader(
        ml=test_ml, files=[],
        target_table="SomeAsset", execution_rid="EXE-X",
        cancel_event=None,
    )
    assert result == {"uploaded": [], "failed": []}
    assert FakeGenericUploader.instances == []
```

- [ ] **Step 2.5: Run all Task 2 tests to verify they pass**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_upload_engine_deriva_py.py -v`

Expected: All 5 tests PASS.

- [ ] **Step 2.6: Run full execution test suite to verify no regression**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/ -v`

Expected: All pass.

- [ ] **Step 2.7: Commit**

```bash
git add src/deriva_ml/execution/upload_engine.py \
        tests/execution/test_upload_engine_deriva_py.py
git commit -m "feat(upload): real _invoke_deriva_py_uploader via GenericUploader

Replaces the Phase 1 NotImplementedError stub with a real uploader
that drives deriva-py's GenericUploader per batch:

- Per-batch TemporaryDirectory scan root with a hardlink/symlink
  farm matching asset_table_upload_spec's regex layout.
- status_callback writes live per-file SQLite status (Uploaded /
  Failed) after each file boundary; idempotent.
- file_callback observes cancel_event and returns -1 to signal
  hatrac abort for in-flight chunk uploads.
- Post-run reconciliation pass over uploader.file_status catches
  any file the callback missed.
- Returns {uploaded, failed}.

Retry/transfer-state/hatrac resumability all stay in deriva-py."
```

---

### Task 3: Remove `bandwidth_limit_mbps` and `parallel_files` from all 7 surfaces

**Why atomic:** A half-removed state (e.g., engine signature changed but callers still pass the kwargs) causes `TypeError` cliffs during bisect and test runs. Do them in one commit.

**Files:**
- Modify: `src/deriva_ml/execution/upload_engine.py` (run_upload_engine — already signature-updated in Task 1, just delete docstring lines)
- Modify: `src/deriva_ml/execution/upload_job.py:54-71, 89-90` (UploadJob.__init__ and _run call site)
- Modify: `src/deriva_ml/execution/execution.py:2250-2272` (Execution.upload_outputs)
- Modify: `src/deriva_ml/execution/execution_record_v2.py:160-178` (ExecutionRecord.upload_outputs)
- Modify: `src/deriva_ml/core/mixins/execution.py:728-795` (DerivaML.upload_pending and start_upload)
- Modify: `src/deriva_ml/cli/upload.py:60-68, 114-115` (remove argparse flags and usage)

- [ ] **Step 3.1: Write failing test — calling with dropped kwargs raises TypeError**

Append to `tests/execution/test_upload_engine.py`:

```python
def test_run_upload_engine_rejects_dropped_kwargs(test_ml):
    """Dropped kwargs raise TypeError — this is the breaking-change contract."""
    from deriva_ml.execution.upload_engine import run_upload_engine
    import pytest
    with pytest.raises(TypeError, match="bandwidth_limit_mbps"):
        run_upload_engine(
            ml=test_ml, execution_rids=[], retry_failed=False,
            bandwidth_limit_mbps=100,
        )
    with pytest.raises(TypeError, match="parallel_files"):
        run_upload_engine(
            ml=test_ml, execution_rids=[], retry_failed=False,
            parallel_files=8,
        )
```

Append to `tests/cli/test_upload_cli.py` (create the file if it does not exist):

```python
def test_cli_rejects_parallel_flag():
    """--parallel is no longer supported — argparse should reject it."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "deriva_ml.cli.upload",
         "--host", "example.org", "--catalog", "1", "--parallel", "4"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr or "--parallel" in result.stderr


def test_cli_rejects_bandwidth_flag():
    """--bandwidth-mbps is no longer supported."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "deriva_ml.cli.upload",
         "--host", "example.org", "--catalog", "1", "--bandwidth-mbps", "100"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr or "--bandwidth-mbps" in result.stderr
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_upload_engine.py::test_run_upload_engine_rejects_dropped_kwargs tests/cli/test_upload_cli.py -v`

Expected: FAIL — `run_upload_engine` currently accepts neither kwarg (they were already stripped in Task 1), so the first test may already pass. The CLI tests will fail because `--parallel` and `--bandwidth-mbps` are currently valid flags.

- [ ] **Step 3.3: Remove kwargs from `run_upload_engine`'s docstring**

In `src/deriva_ml/execution/upload_engine.py`, the signature is already kwarg-free after Task 1. Now remove the stale docstring lines referencing removed kwargs (originally lines 257-259, adjust for current line numbers):

Delete the two docstring lines:

```
        bandwidth_limit_mbps: Cap uploader egress. None = unlimited.
            Passed to deriva-py's uploader config.
        parallel_files: Concurrent file uploads per table. Bounded.
```

- [ ] **Step 3.4: Remove kwargs from `UploadJob`**

In `src/deriva_ml/execution/upload_job.py:54-71`, replace the `__init__` signature:

```python
    def __init__(
        self,
        *,
        ml: "DerivaML",
        execution_rids: "list[str] | None",
        retry_failed: bool,
    ) -> None:
        self.id = f"upl_{uuid.uuid4().hex[:12]}"
        self.status: Literal[
            "running", "paused", "completed", "failed", "cancelled"
        ] = "running"
        self._ml = ml
        self._execution_rids = execution_rids
        self._retry_failed = retry_failed

        self._report: UploadReport | None = None
        self._exception: BaseException | None = None
        self._cancel_event = threading.Event()
        self._progress = UploadProgress()

        self._thread = threading.Thread(
            target=self._run, name=f"upload-{self.id}", daemon=True,
        )
        self._thread.start()
```

`_run` is already kwarg-free after Task 1 (it was updated to pass `cancel_event` only). No change needed.

- [ ] **Step 3.5: Remove kwargs from `Execution.upload_outputs`**

In `src/deriva_ml/execution/execution.py:2250-2272` replace the method:

```python
    def upload_outputs(
        self,
        *,
        retry_failed: bool = False,
    ) -> "UploadReport":
        """Upload this execution's pending rows and asset files.

        Sugar for ml.upload_pending(execution_rids=[self.execution_rid], **kwargs).
        See upload_pending for details.

        Example:
            >>> with exe.execute() as e:
            ...     ... work ...
            >>> exe.upload_outputs()
        """
        return self._ml_object.upload_pending(
            execution_rids=[self.execution_rid],
            retry_failed=retry_failed,
        )
```

- [ ] **Step 3.6: Remove kwargs from `ExecutionRecord.upload_outputs`**

In `src/deriva_ml/execution/execution_record_v2.py:160-178` replace:

```python
    def upload_outputs(
        self,
        *,
        ml: "DerivaML",
        retry_failed: bool = False,
    ) -> "UploadReport":
        """Sugar for ml.upload_pending(execution_rids=[self.rid], ...).

        Records are bare dataclasses — the caller provides the DerivaML
        instance that owns the workspace.
        """
        return ml.upload_pending(
            execution_rids=[self.rid],
            retry_failed=retry_failed,
        )
```

- [ ] **Step 3.7: Remove kwargs from `DerivaML.upload_pending` and `start_upload`**

In `src/deriva_ml/core/mixins/execution.py:728-795` replace:

```python
    def upload_pending(
        self,
        *,
        execution_rids: "list[RID] | None" = None,
        retry_failed: bool = False,
    ) -> "UploadReport":
        """Blocking upload of pending state for selected executions.

        Args:
            execution_rids: List of RIDs, or None to drain every execution
                that has pending work.
            retry_failed: Include rows in status='failed'.

        Returns:
            UploadReport with totals + per-table counts + error lines.

        Example:
            >>> report = ml.upload_pending()
            >>> print(f"{report.total_uploaded} uploaded, "
            ...       f"{report.total_failed} failed")
        """
        return run_upload_engine(
            ml=self,
            execution_rids=execution_rids,
            retry_failed=retry_failed,
        )

    def start_upload(
        self,
        *,
        execution_rids: "list[RID] | None" = None,
        retry_failed: bool = False,
    ) -> "UploadJob":
        """Non-blocking upload — returns an UploadJob to poll / wait.

        Spawns a daemon thread in the current process. If the process
        exits, the thread dies. For survive-process uploads, run
        ``deriva-ml upload`` from a shell (see CLI, Group H).

        Args: identical to upload_pending.

        Returns:
            An UploadJob; call job.wait() to block, job.progress() to
            poll, job.cancel() to stop.

        Example:
            >>> job = ml.start_upload()
            >>> while job.status == "running":
            ...     time.sleep(5)
            ...     print(job.progress())
            >>> report = job.wait()
        """
        from deriva_ml.execution.upload_job import UploadJob
        return UploadJob(
            ml=self,
            execution_rids=execution_rids,
            retry_failed=retry_failed,
        )
```

- [ ] **Step 3.8: Remove CLI flags**

In `src/deriva_ml/cli/upload.py:60-68` delete the two argparse blocks:

```python
    p.add_argument(
        "--bandwidth-mbps", type=int, default=None,
        dest="bandwidth_limit_mbps",
        help="Cap upload egress in Mbps. Unlimited if omitted.",
    )
    p.add_argument(
        "--parallel", type=int, default=4, dest="parallel_files",
        help="Concurrent file uploads per table (default 4).",
    )
```

In the same file lines 111-116 replace:

```python
        report = ml.upload_pending(
            execution_rids=args.execution_rids,
            retry_failed=args.retry_failed,
            bandwidth_limit_mbps=args.bandwidth_limit_mbps,
            parallel_files=args.parallel_files,
        )
```

with:

```python
        report = ml.upload_pending(
            execution_rids=args.execution_rids,
            retry_failed=args.retry_failed,
        )
```

Also update the module-level docstring example (lines 9-11) to drop the removed flags:

```python
"""Command-line interface for deriva-ml upload.

Wraps DerivaML.upload_pending so operator-driven uploads can be
scheduled, backgrounded, or run on a different host from the
compute. Typical invocations:

    deriva-ml-upload --host example.org --catalog 42

    deriva-ml-upload --host example.org --catalog 42 \\
        --execution EXE-A --execution EXE-B

    nohup deriva-ml-upload --host example.org --catalog 42 &

Per spec §2.11.4. Drives the same engine as ml.upload_pending.
"""
```

- [ ] **Step 3.9: Run tests to verify dropped-kwarg tests pass**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_upload_engine.py::test_run_upload_engine_rejects_dropped_kwargs tests/cli/test_upload_cli.py -v`

Expected: PASS.

- [ ] **Step 3.10: Run full test suite to verify no other callers break**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest -v 2>&1 | tail -50`

Expected: All tests pass. If any test still passes the removed kwargs, fix in Task 4. Record failures.

- [ ] **Step 3.11: Commit**

```bash
git add src/deriva_ml/execution/upload_engine.py \
        src/deriva_ml/execution/upload_job.py \
        src/deriva_ml/execution/execution.py \
        src/deriva_ml/execution/execution_record_v2.py \
        src/deriva_ml/core/mixins/execution.py \
        src/deriva_ml/cli/upload.py \
        tests/execution/test_upload_engine.py \
        tests/cli/test_upload_cli.py
git commit -m "feat(upload)!: drop bandwidth_limit_mbps and parallel_files kwargs

BREAKING CHANGE: The bandwidth_limit_mbps and parallel_files keyword
arguments were removed from ml.upload_pending, ml.start_upload,
Execution.upload_outputs, ExecutionRecord.upload_outputs,
run_upload_engine, and UploadJob. The CLI flags --bandwidth-mbps and
--parallel are likewise removed.

deriva-py's GenericUploader does not implement bandwidth throttling
or parallel file uploads. These kwargs were plumbed through all
surfaces in Phase 1 but never reached deriva-py — they were accepted
and ignored. Removing them makes the signatures match reality.

Callers passing these kwargs will get TypeError on upgrade; CLI
users passing --bandwidth-mbps or --parallel will get 'unrecognized
arguments'."
```

---

### Task 4: Update existing test suite to drop kwargs and fix any callers

**Why:** Task 3's `git grep bandwidth_limit_mbps` and `git grep parallel_files` may still turn up hits in fixtures, tests, or example code.

**Files:**
- Modify: any test or fixture that passes either kwarg

- [ ] **Step 4.1: Audit for stragglers**

Run: `git grep -n 'bandwidth_limit_mbps\|parallel_files' -- 'src/' 'tests/' 'docs/'`

Expected: no hits in `src/` after Task 3. Any hits in `tests/` or `docs/` need remediation.

- [ ] **Step 4.2: Fix each hit**

For each stray reference:
- If in a test: remove the kwarg from the call site. If the test's intent was to verify the kwarg's behavior, delete the test.
- If in a docstring or markdown: remove the line or update the example.

- [ ] **Step 4.3: Run full test suite**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest -v`

Expected: all tests pass.

- [ ] **Step 4.4: Run linter**

Run: `uv run ruff check src/ tests/`

Expected: no new violations introduced by the changes.

- [ ] **Step 4.5: Commit (only if changes were needed)**

```bash
git add <affected files>
git commit -m "test(upload): remove stale bandwidth_limit_mbps / parallel_files references

Follow-up to the kwarg-drop commit — cleans up references in tests
and docs that the previous commit did not touch."
```

If no stragglers were found, skip the commit.

---

### Task 5: CHANGELOG entry for the breaking change

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 5.1: Check for a CHANGELOG file**

Run: `ls CHANGELOG.md docs/CHANGELOG.md 2>/dev/null`

If it exists, use it. If not, create `CHANGELOG.md` at the repo root.

- [ ] **Step 5.2: Add the breaking-change entry**

Under an "Unreleased" section (creating one if needed), add:

```markdown
## Unreleased

### Breaking changes

- **Upload engine:** removed `bandwidth_limit_mbps` and `parallel_files`
  keyword arguments from `ml.upload_pending`, `ml.start_upload`,
  `Execution.upload_outputs`, `ExecutionRecord.upload_outputs`,
  `run_upload_engine`, and `UploadJob`. The CLI flags
  `--bandwidth-mbps` and `--parallel` are likewise removed.
  deriva-py's `GenericUploader` does not implement bandwidth
  throttling or parallel file uploads; the kwargs were plumbed
  but no-op. Callers must drop them on upgrade.

### Features

- **Upload engine:** `_invoke_deriva_py_uploader` now drives
  `GenericUploader` with real per-batch uploads, live SQLite per-file
  status via `status_callback`, and cancellation via
  `UploadJob.cancel()` → `GenericUploader.cancel()`. Retry/transfer-
  state/hatrac-resumability remain inside deriva-py.
```

- [ ] **Step 5.3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): S3 upload-engine deriva-py integration"
```

---

## Verification Checklist

After all 5 tasks land:

- [ ] `git grep -n bandwidth_limit_mbps src/` returns nothing
- [ ] `git grep -n parallel_files src/` returns nothing
- [ ] `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_upload_engine_deriva_py.py -v` — all 5 unit tests pass
- [ ] `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/cli/test_upload_cli.py -v` — argparse-rejection tests pass
- [ ] `DERIVA_ML_ALLOW_DIRTY=true uv run pytest` — full suite passes
- [ ] `uv run ruff check src/` — clean
- [ ] The schema-doc validator (from S0) still agrees with schema.md (no schema changes in this subsystem, so should be automatic)
