# Status Enum Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy `Status` enum (title-case, catalog-written) and lowercase `ExecutionStatus` enum with a single title-case `ExecutionStatus` that matches the catalog directly, eliminating the reconciliation bug Phase 1 shipped with.

**Architecture:** Change `ExecutionStatus` values from lowercase to title-case in one shot. Migrate every internal call site (~15 sites across 6 files) from `update_status(Status.xxx, "msg")` to `exe.update_status(ExecutionStatus.Xxx)` or `logger.info("msg")`. Delete the legacy `Status` enum, `_initialize_execution`, and `DerivaML.status` attribute. Add a grep-gate test to catch missed migrations before merge.

**Tech Stack:** Python 3.12+, StrEnum, SQLAlchemy Core (existing), pytest.

**Spec:** `docs/superpowers/specs/2026-04-21-status-enum-reconciliation-design.md`

---

## Conventions referenced throughout this plan

- **Worktree root:** `/Users/carl/github/deriva-ml/.claude/worktrees/phase2-s1-status-enum/`. All paths below are relative to it.
- **`Ex`** is shorthand for `ExecutionStatus` in narrative text; at call sites the code uses the full name `ExecutionStatus`.
- **Environment:** `export PATH="/Users/carl/.local/bin:$PATH"` if `uv` not found. `DERIVA_ML_ALLOW_DIRTY=true` for tests. Live catalog tests need `DERIVA_HOST=localhost`.
- **`status_detail` column on `Execution`**: pre-existing catalog column; we don't write to it post-M2. Ignore.
- **Ordering rule for the migration:** enum value change FIRST (Task 1), then sweep callers (Tasks 2–7), then delete legacy enum LAST (Task 8). Reversing this order would leave the tree in a failing state for every intermediate commit.

---

## Task Group overview

Seven task groups. Each group produces a working, testable increment.

| Group | Scope | Tasks |
|---|---|---|
| **A** | Enum value change + round-trip tests | 3 tasks |
| **B** | New `Execution.update_status` / `ExecutionRecord.update_status` API | 4 tasks |
| **C** | Library call-site migration (6 files) | 6 tasks |
| **D** | Test sweep — lowercase literals → title-case | 4 tasks |
| **E** | Deletions — `Status` enum, `_initialize_execution`, `DerivaML.status` | 3 tasks |
| **F** | Grep-gate + full regression + CHANGELOG | 3 tasks |
| **G** | Final review | 1 task |

Total: ~24 bite-sized tasks.

---

## Task Group A — `ExecutionStatus` enum value change

Change the 7-character string values in-place. This immediately breaks every test that hardcoded `"running"`, but those tests get fixed in Group D; between Groups A and D the suite will be red. That's the intended flow.

### Task A1: Change `ExecutionStatus` values to title-case

**Files:**
- Modify: `src/deriva_ml/execution/state_store.py:51-69`

- [ ] **Step 1: Write the failing test**

Create `tests/execution/test_status_migration.py`:

```python
"""Tests for the title-case ExecutionStatus migration (Phase 2 Subsystem 1)."""

from __future__ import annotations

import pytest


def test_execution_status_values_are_title_case():
    from deriva_ml.execution.state_store import ExecutionStatus
    assert ExecutionStatus.Created.value == "Created"
    assert ExecutionStatus.Running.value == "Running"
    assert ExecutionStatus.Stopped.value == "Stopped"
    assert ExecutionStatus.Failed.value == "Failed"
    assert ExecutionStatus.Pending_Upload.value == "Pending_Upload"
    assert ExecutionStatus.Uploaded.value == "Uploaded"
    assert ExecutionStatus.Aborted.value == "Aborted"


def test_execution_status_parses_title_case_from_catalog():
    """A catalog row with {'Status': 'Running'} parses directly."""
    from deriva_ml.execution.state_store import ExecutionStatus
    assert ExecutionStatus("Running") is ExecutionStatus.Running
    assert ExecutionStatus("Pending_Upload") is ExecutionStatus.Pending_Upload


def test_execution_status_rejects_lowercase():
    from deriva_ml.execution.state_store import ExecutionStatus
    with pytest.raises(ValueError):
        ExecutionStatus("running")
    with pytest.raises(ValueError):
        ExecutionStatus("pending_upload")


def test_execution_status_member_count():
    """Canonical set is exactly 7."""
    from deriva_ml.execution.state_store import ExecutionStatus
    assert len(list(ExecutionStatus)) == 7
```

- [ ] **Step 2: Run to verify it fails**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_status_migration.py -v
```

Expected: 4 FAIL — the enum currently has lowercase values (`created`, not `Created`), and Python identifiers are lowercase (`ExecutionStatus.created`, not `ExecutionStatus.Created`).

- [ ] **Step 3: Change the enum**

Replace the body of `class ExecutionStatus` in `src/deriva_ml/execution/state_store.py:51-69` with:

```python
class ExecutionStatus(StrEnum):
    """Lifecycle status for an Execution (see Phase 1 spec §2.2).

    Transitions are:
        Created → Running → {Stopped, Failed} →
            {Pending_Upload → {Uploaded, Failed}}
        Created → Aborted
        Running → Aborted

    Values are title-case to match the catalog Execution.Status field
    directly — ExecutionStatus(row["Status"]) works without translation.
    Python identifiers are title-case to match the values (precedent:
    stdlib http.HTTPStatus uses uppercase identifiers).
    """
    Created = "Created"
    Running = "Running"
    Stopped = "Stopped"
    Failed = "Failed"
    Pending_Upload = "Pending_Upload"
    Uploaded = "Uploaded"
    Aborted = "Aborted"
```

- [ ] **Step 4: Run tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_status_migration.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/state_store.py tests/execution/test_status_migration.py
git commit -m "$(cat <<'EOF'
feat(status): ExecutionStatus values → title-case

Changes the 7 StrEnum values from lowercase to title-case to match
the catalog Execution.Status column directly. Python identifiers
also title-case (precedent: http.HTTPStatus).

Breaks existing tests that hardcoded lowercase literals — those are
fixed in Group D's test sweep.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task A2: Update `ALLOWED_TRANSITIONS` references in state_machine

**Files:**
- Modify: `src/deriva_ml/execution/state_machine.py` — search for all `ExecutionStatus.xxx` references; Python identifiers changed from lowercase to title-case in A1.

- [ ] **Step 1: Run tests to see the breakage**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_machine.py -v 2>&1 | tail -20
```

Expected: many failures — `AttributeError: ExecutionStatus has no attribute 'running'` etc. The state_machine module's `ALLOWED_TRANSITIONS` tuple references lowercase names; after A1 those are title-case.

- [ ] **Step 2: Grep for all lowercase `ExecutionStatus.` references in src/**

```bash
grep -rn 'ExecutionStatus\.\(created\|running\|stopped\|failed\|pending_upload\|uploaded\|aborted\)' src/deriva_ml/ | grep -v __pycache__
```

Expected output: every hit is a case to rewrite. Most will be in `state_machine.py`.

- [ ] **Step 3: Rewrite each hit to title-case**

For every `ExecutionStatus.created` → `ExecutionStatus.Created`, `.running` → `.Running`, `.stopped` → `.Stopped`, `.failed` → `.Failed`, `.pending_upload` → `.Pending_Upload`, `.uploaded` → `.Uploaded`, `.aborted` → `.Aborted`.

Example: `src/deriva_ml/execution/state_machine.py` `ALLOWED_TRANSITIONS`:

```python
# Before
ALLOWED_TRANSITIONS = [
    (ExecutionStatus.created, ExecutionStatus.running),
    (ExecutionStatus.running, ExecutionStatus.stopped),
    ...
]

# After
ALLOWED_TRANSITIONS = [
    (ExecutionStatus.Created, ExecutionStatus.Running),
    (ExecutionStatus.Running, ExecutionStatus.Stopped),
    ...
]
```

Do the same sweep in every src/ file that has hits.

- [ ] **Step 4: Re-grep to confirm no lowercase member access remains**

```bash
grep -rn 'ExecutionStatus\.\(created\|running\|stopped\|failed\|pending_upload\|uploaded\|aborted\)' src/deriva_ml/ | grep -v __pycache__
```

Expected: empty.

- [ ] **Step 5: Run state_machine tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_machine.py -v 2>&1 | tail -15
```

Expected: still failures, but the ones about `ExecutionStatus.running` have disappeared. Remaining failures are now about lowercase string literals (`"running"`) in tests — those are fixed in Group D.

- [ ] **Step 6: Commit**

```bash
git add src/deriva_ml/
git commit -m "$(cat <<'EOF'
refactor(status): update all ExecutionStatus.xxx references to title-case

Follows A1's enum identifier change. Sweeps ExecutionStatus.running,
.created, .failed, etc. in all src/deriva_ml/ files to their title-case
equivalents (Running, Created, Failed, etc.).

Tests still red pending Group D — string literals in test assertions
need the same treatment.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task A3: Verify SQLite writes use title-case

**Files:**
- Verify: `src/deriva_ml/execution/state_store.py` (everywhere `str(status)` is used with `PendingRowStatus` or `ExecutionStatus`)

- [ ] **Step 1: Understand what Phase 1 wrote to SQLite**

Check what happens in `insert_execution` and `update_execution` when a status enum is passed:

```bash
grep -n "str(status)\|status=str\|PendingRowStatus\|ExecutionStatus" src/deriva_ml/execution/state_store.py | head -20
```

Phase 1 uses `str(status)` which for StrEnum subclasses returns the enum value. Since we changed the values in A1, `str(ExecutionStatus.Running)` now returns `"Running"` instead of `"running"`. **Nothing in state_store.py needs to change** — it correctly routes enum values through.

- [ ] **Step 2: Write a test that asserts SQLite stores title-case**

Append to `tests/execution/test_status_migration.py`:

```python
def test_sqlite_stores_title_case_status(tmp_path):
    """insert_execution + get_execution round-trips title-case status."""
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_store import ExecutionStateStore, ExecutionStatus

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-T1", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-T1",
        created_at=now, last_activity=now,
    )
    row = store.get_execution("EXE-T1")
    assert row is not None
    assert row["status"] == "Running"  # title-case stored verbatim
    assert ExecutionStatus(row["status"]) is ExecutionStatus.Running
```

- [ ] **Step 3: Run**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_status_migration.py::test_sqlite_stores_title_case_status -v
```

Expected: PASS (no implementation change needed — the enum change in A1 already routes through).

- [ ] **Step 4: Commit**

```bash
git add tests/execution/test_status_migration.py
git commit -m "$(cat <<'EOF'
test(status): SQLite round-trips title-case ExecutionStatus values

Verifies that after A1 the SQLite registry stores title-case values
(e.g. 'Running') and can read them back as ExecutionStatus enum members.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task Group B — New `update_status` API

Add the new `Execution.update_status` and `ExecutionRecord.update_status` methods. The legacy `update_status` on `Execution` still exists and still accepts the legacy `Status` enum; the two coexist during Groups B–C. After Group C (all callers migrated) and before Group E (legacy deletion), there's no functional overlap — legacy `Status` values are no longer written anywhere.

**Concurrency caveat:** the legacy `update_status` writes to `Execution_Status` via a catalog PUT that bypasses the state machine. The new method routes through `state_machine.transition()` which writes SQLite + catalog atomically. Groups B and C keep both methods active; the legacy one is only used by code that Group C rewrites, so no call site ever uses both in the same process.

### Task B1: Add `Execution.update_status` new method

**Files:**
- Modify: `src/deriva_ml/execution/execution.py` — replace the body of the existing `update_status` method.
- Test: `tests/execution/test_update_status.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/execution/test_update_status.py`:

```python
"""Tests for the new Execution.update_status / ExecutionRecord.update_status API."""

from __future__ import annotations


def _make_workflow(test_ml, name: str):
    from deriva_ml import MLVocab as vc

    test_ml.add_term(
        vc.workflow_type,
        "Test Workflow",
        description="for update_status tests",
    )
    return test_ml.create_workflow(
        name=name,
        workflow_type="Test Workflow",
        description="for update_status tests",
    )


def test_execution_update_status_valid_transition(test_ml):
    """Exe.update_status transitions via the state machine and persists to SQLite."""
    from deriva_ml.execution.state_store import ExecutionStatus

    wf = _make_workflow(test_ml, "B1 valid transition")
    exe = test_ml.create_execution(description="valid", workflow=wf)
    store = test_ml.workspace.execution_state_store()

    # Explicit transition Created → Running
    exe.update_status(ExecutionStatus.Running)

    row = store.get_execution(exe.execution_rid)
    assert row["status"] == "Running"


def test_execution_update_status_invalid_transition_raises(test_ml):
    """Violating ALLOWED_TRANSITIONS raises InvalidTransitionError."""
    from deriva_ml.execution.state_machine import InvalidTransitionError
    from deriva_ml.execution.state_store import ExecutionStatus

    wf = _make_workflow(test_ml, "B1 invalid")
    exe = test_ml.create_execution(description="invalid", workflow=wf)
    # Created → Uploaded is not an allowed direct transition.
    with pytest.raises(InvalidTransitionError):
        exe.update_status(ExecutionStatus.Uploaded)


def test_execution_update_status_error_kwarg_on_failed(test_ml):
    """error='...' is written to the error column on Failed."""
    from deriva_ml.execution.state_store import ExecutionStatus

    wf = _make_workflow(test_ml, "B1 error kwarg")
    exe = test_ml.create_execution(description="err", workflow=wf)
    exe.update_status(ExecutionStatus.Running)
    exe.update_status(ExecutionStatus.Failed, error="Network timeout")

    store = test_ml.workspace.execution_state_store()
    row = store.get_execution(exe.execution_rid)
    assert row["status"] == "Failed"
    assert row["error"] == "Network timeout"


def test_execution_update_status_error_kwarg_on_nonterminal_warns(test_ml, caplog):
    """error='...' on a non-terminal transition logs a warning; proceeds normally."""
    import logging
    from deriva_ml.execution.state_store import ExecutionStatus

    wf = _make_workflow(test_ml, "B1 warn")
    exe = test_ml.create_execution(description="warn", workflow=wf)
    with caplog.at_level(logging.WARNING):
        exe.update_status(ExecutionStatus.Running, error="this should warn")
    assert any(
        "error= ignored on non-terminal" in rec.message
        or "non-terminal transition" in rec.message
        for rec in caplog.records
    )
    store = test_ml.workspace.execution_state_store()
    row = store.get_execution(exe.execution_rid)
    assert row["status"] == "Running"
    # The error column is NOT populated for non-terminal transitions.
    assert row["error"] is None


import pytest  # keep at bottom so the per-test monkeypatches don't conflict
```

- [ ] **Step 2: Run to verify failures**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/test_update_status.py::test_execution_update_status_valid_transition -v
```

Expected: FAIL — the existing `update_status` takes a legacy `Status` argument, not `ExecutionStatus`, and doesn't route through `state_machine.transition`.

- [ ] **Step 3: Replace the `Execution.update_status` method body**

Find the existing `update_status` method in `src/deriva_ml/execution/execution.py` (search for `def update_status`). Replace the entire body with:

```python
def update_status(
    self,
    target: "ExecutionStatus",
    *,
    error: str | None = None,
) -> None:
    """Transition this execution to a new status.

    Thin wrapper around state_machine.transition() that validates
    against ALLOWED_TRANSITIONS, writes the SQLite registry, and syncs
    to the catalog when online.

    Args:
        target: Target ExecutionStatus enum member.
        error: For Failed/Aborted transitions, a human-readable error
            message written to the `error` column. On a non-terminal
            transition, error is ignored and a warning is logged.

    Raises:
        InvalidTransitionError: If the (current, target) pair is not in
            ALLOWED_TRANSITIONS.
        DerivaMLStateInconsistency: If state_machine's catalog sync
            detects divergence.

    Example:
        >>> exe.update_status(ExecutionStatus.Running)
        >>> exe.update_status(ExecutionStatus.Failed, error="Network timeout")
    """
    from deriva_ml.execution.state_machine import transition
    from deriva_ml.execution.state_store import ExecutionStatus

    store = self._ml_object.workspace.execution_state_store()
    row = store.get_execution(self.execution_rid)
    if row is None:
        raise DerivaMLException(
            f"Execution {self.execution_rid} not in workspace registry"
        )
    current = ExecutionStatus(row["status"])

    extra_fields: dict = {}
    # Terminal = Failed, Aborted.
    if target in (ExecutionStatus.Failed, ExecutionStatus.Aborted):
        if error is not None:
            extra_fields["error"] = error
    elif error is not None:
        import logging
        logging.getLogger(__name__).warning(
            "error= ignored on non-terminal transition to %s: %s",
            target.value, error,
        )

    transition(
        store=store,
        catalog=self._ml_object.catalog if self._ml_object._mode.value == "online" else None,
        execution_rid=self.execution_rid,
        current=current,
        target=target,
        mode=self._ml_object._mode,
        extra_fields=extra_fields,
    )
```

- [ ] **Step 4: Run tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/test_update_status.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/execution.py tests/execution/test_update_status.py
git commit -m "$(cat <<'EOF'
feat(exe): Execution.update_status(target, *, error=None) — new API

Replaces the legacy update_status(Status, message) signature with
one that takes an ExecutionStatus enum member and an optional error
kwarg (for Failed/Aborted transitions only).

The implementation routes through state_machine.transition() so
SQLite + catalog stay coordinated; ALLOWED_TRANSITIONS is enforced.
Invalid transitions raise InvalidTransitionError; non-terminal
error= calls log a warning and proceed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task B2: Add `ExecutionRecord.update_status`

**Files:**
- Modify: `src/deriva_ml/execution/execution_record_v2.py`
- Test: `tests/execution/test_update_status.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/execution/test_update_status.py`:

```python
def test_record_update_status_transitions(test_ml):
    """ExecutionRecord.update_status(target, *, ml, error=None) parallel."""
    from deriva_ml.execution.state_store import ExecutionStatus

    wf = _make_workflow(test_ml, "B2 record")
    exe = test_ml.create_execution(description="rec", workflow=wf)
    rec = next(r for r in test_ml.list_executions() if r.rid == exe.execution_rid)

    rec.update_status(ExecutionStatus.Running, ml=test_ml)
    store = test_ml.workspace.execution_state_store()
    row = store.get_execution(exe.execution_rid)
    assert row["status"] == "Running"

    rec.update_status(ExecutionStatus.Failed, ml=test_ml, error="boom")
    row = store.get_execution(exe.execution_rid)
    assert row["status"] == "Failed"
    assert row["error"] == "boom"
```

- [ ] **Step 2: Run — expect AttributeError**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/test_update_status.py::test_record_update_status_transitions -v
```

Expected: `AttributeError: 'ExecutionRecord' object has no attribute 'update_status'`.

- [ ] **Step 3: Add the method**

In `src/deriva_ml/execution/execution_record_v2.py`, add to the `ExecutionRecord` class (after `upload_outputs` or `pending_summary`):

```python
def update_status(
    self,
    target: "ExecutionStatus",
    *,
    ml: "DerivaML",
    error: str | None = None,
) -> None:
    """Transition this execution's status via the workspace state machine.

    Parallel to Execution.update_status. ExecutionRecord is a bare
    dataclass and doesn't carry an ml reference — caller passes one.

    Args:
        target: Target ExecutionStatus enum member.
        ml: The DerivaML instance whose workspace owns the registry.
        error: For Failed/Aborted, a human-readable message.

    Raises:
        InvalidTransitionError: If the transition is not allowed.
        DerivaMLStateInconsistency: If catalog sync detects divergence.

    Example:
        >>> rec.update_status(ExecutionStatus.Aborted, ml=ml, error="user cancel")
    """
    from deriva_ml.execution.state_machine import transition
    from deriva_ml.execution.state_store import ExecutionStatus

    store = ml.workspace.execution_state_store()
    row = store.get_execution(self.rid)
    if row is None:
        raise DerivaMLException(
            f"Execution {self.rid} not in workspace registry"
        )
    current = ExecutionStatus(row["status"])

    extra_fields: dict = {}
    if target in (ExecutionStatus.Failed, ExecutionStatus.Aborted):
        if error is not None:
            extra_fields["error"] = error
    elif error is not None:
        import logging
        logging.getLogger(__name__).warning(
            "error= ignored on non-terminal transition to %s: %s",
            target.value, error,
        )

    transition(
        store=store,
        catalog=ml.catalog if ml._mode.value == "online" else None,
        execution_rid=self.rid,
        current=current,
        target=target,
        mode=ml._mode,
        extra_fields=extra_fields,
    )
```

If `DerivaMLException` isn't already imported in this file, add:

```python
from deriva_ml.core.exceptions import DerivaMLException
```

(Check first — it may already be imported.)

Add `ExecutionStatus` and `DerivaML` to the file's TYPE_CHECKING block if not already there:

```python
if TYPE_CHECKING:
    from deriva_ml.core.base import DerivaML
    from deriva_ml.execution.state_store import ExecutionStatus
```

- [ ] **Step 4: Run tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/test_update_status.py -v
```

Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/execution_record_v2.py tests/execution/test_update_status.py
git commit -m "$(cat <<'EOF'
feat(exe): ExecutionRecord.update_status(target, *, ml, error=None)

Parallel to Execution.update_status but takes an ml= kwarg since
records are bare dataclasses with no DerivaML backref.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task B3: Offline-mode coverage for `update_status`

**Files:**
- Test: `tests/execution/test_update_status.py` (append)

- [ ] **Step 1: Write the test**

Append to `tests/execution/test_update_status.py`:

```python
def test_update_status_offline_writes_sqlite_sets_sync_pending(tmp_path, monkeypatch):
    """In offline mode, update_status writes SQLite + sync_pending=True, no catalog call."""
    from datetime import datetime, timezone
    from unittest.mock import MagicMock

    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_store import ExecutionStateStore, ExecutionStatus

    # Minimal setup: build a store + fake ml with just enough surface.
    from sqlalchemy import create_engine

    eng = create_engine(f"sqlite:///{tmp_path}/offline.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-OFF", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.Running,
        mode=ConnectionMode.offline, working_dir_rel="execution/EXE-OFF",
        created_at=now, last_activity=now,
    )

    # Run transition directly (we're testing state_machine behavior under offline mode).
    from deriva_ml.execution.state_machine import transition
    transition(
        store=store,
        catalog=None,
        execution_rid="EXE-OFF",
        current=ExecutionStatus.Running,
        target=ExecutionStatus.Stopped,
        mode=ConnectionMode.offline,
    )
    row = store.get_execution("EXE-OFF")
    assert row["status"] == "Stopped"
    assert row["sync_pending"] is True
```

- [ ] **Step 2: Run**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_update_status.py::test_update_status_offline_writes_sqlite_sets_sync_pending -v
```

Expected: PASS (this exercises existing Phase 1 behavior; the new API just wraps `transition`). If it fails, investigate why offline mode isn't setting `sync_pending=True`.

- [ ] **Step 3: Commit**

```bash
git add tests/execution/test_update_status.py
git commit -m "$(cat <<'EOF'
test(status): offline-mode update_status sets sync_pending=True

Covers the state_machine contract in offline mode: SQLite is the
authoritative store; catalog sync deferred until online reconnect.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task B4: `DerivaMLException` import audit

**Files:**
- Verify: `src/deriva_ml/execution/execution_record_v2.py`, `src/deriva_ml/execution/execution.py`

- [ ] **Step 1: Confirm both files have the import**

```bash
grep -n "DerivaMLException" src/deriva_ml/execution/execution.py src/deriva_ml/execution/execution_record_v2.py
```

Expected: both files already import `DerivaMLException`. If either is missing, add:

```python
from deriva_ml.core.exceptions import DerivaMLException
```

- [ ] **Step 2: Re-run Group B tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/test_update_status.py -v
```

Expected: all PASS.

- [ ] **Step 3: If any changes were needed, commit**

```bash
git add src/deriva_ml/
git commit -m "chore(exe): ensure DerivaMLException import in exe record module"
```

If no changes, skip the commit.

---

## Task Group C — Library call-site migration

For each legacy call site (see spec §5), rewrite to use the new API. One task per file to keep commits atomic.

### Task C1: Migrate `src/deriva_ml/core/base.py`

**Files:**
- Modify: `src/deriva_ml/core/base.py`

- [ ] **Step 1: Inspect current state**

```bash
grep -n "Status\.\|update_status\|self\.status" src/deriva_ml/core/base.py
```

Expected hits:
- `353: self.status = Status.pending.value` — delete (verify no readers first).
- `380: if self._execution and self._execution.status != Status.completed:` — rewrite.
- `381: self._execution.update_status(Status.aborted, "Execution Aborted")` — rewrite.

- [ ] **Step 2: Verify no readers of `DerivaML.status`**

```bash
grep -rn "\.status " src/deriva_ml/ | grep -v __pycache__ | grep -v "exe\.\|execution\.\|record\.\|_execution\." | grep "ml\.\|self\.\|derivaml\." | head -10
```

If any hit references `ml.status` or similar where `ml` is a DerivaML instance, the attribute IS read somewhere and we need to investigate before deletion.

If no readers, proceed.

- [ ] **Step 3: Edit `core/base.py`**

**Line 353**: delete `self.status = Status.pending.value`.

**Lines 380-381** (the destructor-style abort):

```python
# Before
if self._execution and self._execution.status != Status.completed:
    self._execution.update_status(Status.aborted, "Execution Aborted")

# After
if self._execution and self._execution.status is not ExecutionStatus.Aborted:
    # Best-effort abort on DerivaML shutdown; tolerate errors.
    try:
        self._execution.update_status(ExecutionStatus.Aborted, error="Execution Aborted")
    except Exception as exc:
        logger.warning("abort on shutdown failed for %s: %s",
                       self._execution.execution_rid, exc)
```

Note the logic inversion: the old code checked `!= Status.completed` (abort if NOT completed). In the new world, "completed" doesn't exist as a single state — the caller's real intent is "abort if the execution hasn't reached a clean terminal state." Simplest equivalent is "abort if not already aborted" — since `update_status(Aborted)` will raise InvalidTransition if the exe is in a state where aborting isn't allowed (e.g., `Uploaded`), the try/except covers the other terminal states. Document this rationale in the code comment.

Also update the import at the top of `core/base.py`:

```python
# Before
from deriva_ml.core.enums import Status

# After
from deriva_ml.execution.state_store import ExecutionStatus
```

(Keep `Status` imported until Group E deletes the enum. Actually — if `Status` is no longer referenced in this file after the edit, the import can go now.)

- [ ] **Step 4: Run tests to confirm nothing broke**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/ -q
```

Expected: all PASS. If failures reference the deleted `DerivaML.status` attribute, restore it and revisit Step 2.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/core/base.py
git commit -m "$(cat <<'EOF'
refactor(core): migrate base.py to ExecutionStatus.Aborted

- Delete self.status = Status.pending.value (vestigial, no readers).
- Rewrite destructor-style abort using ExecutionStatus.Aborted +
  InvalidTransition-tolerant try/except.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task C2: Migrate `src/deriva_ml/execution/execution.py`

**Files:**
- Modify: `src/deriva_ml/execution/execution.py`

Per spec §5 table, 10+ call sites. Walk through them one-by-one in the implementation.

- [ ] **Step 1: Grep current call sites**

```bash
grep -n "Status\.\|update_status(Status\." src/deriva_ml/execution/execution.py
```

- [ ] **Step 2: Rewrite each site per spec §5 table**

For each of the following line numbers in `src/deriva_ml/execution/execution.py`:

**Line 307** (`status=Status.created`):
```python
# Before
status=Status.created,
# After
status=ExecutionStatus.Created,
```

**Line 514** (`update_status(Status.initializing, f"Materialize bag {dataset.rid}... ")`):
```python
# Before
self.update_status(Status.initializing, f"Materialize bag {dataset.rid}... ")
# After
logger.info("Materialize bag %s...", dataset.rid)
```
(Delete the call entirely — replace with `logger.info`. Ensure `logger` is imported at module top. Phase 1 already uses `logger.warning` in this file, so `logger` exists.)

**Line 526** (`update_status(Status.running, "Downloading assets ...")`):
```python
# Before
self.update_status(Status.running, "Downloading assets ...")
# After
logger.info("Downloading assets ...")
```
(Progress chatter → log-only. The actual Running transition happens via `__enter__` / `execute()` elsewhere.)

**Line 583** (`update_status(Status.pending, "Initialize status finished.")`):
```python
# Before
self.update_status(Status.pending, "Initialize status finished.")
# After
# DELETE entirely. Execution is already in Created after create_catalog_execution;
# no further transition happens here.
```

**Line 980** (`update_status(Status.initializing, "Start execution  ...")`):
```python
# Before
self.update_status(Status.initializing, "Start execution  ...")
# After
logger.info("Start execution...")
```

**Line 1020** (`update_status(Status.completed, "Algorithm execution ended.")`):
This is the end of `execution_stop`. Algorithm finished; rows/assets staged but not uploaded.
```python
# Before
self.update_status(Status.completed, "Algorithm execution ended.")
# After
self.update_status(ExecutionStatus.Stopped)
```

**Line 1122** (`update_status(Status.running, "Uploading execution files...")`):
This is entering the upload phase (inside upload_execution_outputs).
```python
# Before
self.update_status(Status.running, "Uploading execution files...")
# After
# The transition to Pending_Upload has to come FROM Stopped per ALLOWED_TRANSITIONS.
# If upload_execution_outputs is called from an exe in Stopped state, transition.
# Otherwise just log.
if self.status is ExecutionStatus.Stopped:
    self.update_status(ExecutionStatus.Pending_Upload)
logger.info("Uploading execution files...")
```
(Check ALLOWED_TRANSITIONS. Phase 1 has `(Stopped, Pending_Upload)`. Without `Stopped`, the transition raises. So guard on `self.status is ExecutionStatus.Stopped`.)

**Line 1134** (`update_status(Status.failed, error)`):
```python
# Before
self.update_status(Status.failed, error)
# After
self.update_status(ExecutionStatus.Failed, error=error)
```

**Line 1166** (`update_status(Status.running, "Updating features...")`):
```python
# Before
self.update_status(Status.running, "Updating features...")
# After
logger.info("Updating features...")
```

**Line 1177** (`update_status(Status.running, "Upload assets complete")`):
```python
# Before
self.update_status(Status.running, "Upload assets complete")
# After
logger.info("Upload assets complete")
```

**Line 1378** (`update_status(Status.completed, "Successfully end the execution.")`):
End of `upload_execution_outputs` — the upload succeeded.
```python
# Before
self.update_status(Status.completed, "Successfully end the execution.")
# After
self.update_status(ExecutionStatus.Uploaded)
```

**Line 1384** (`update_status(Status.failed, error)`):
```python
# Before
self.update_status(Status.failed, error)
# After
self.update_status(ExecutionStatus.Failed, error=error)
```

Also update docstring examples that reference legacy `Status.xxx` — mechanical swap. Line numbers drift as you edit; find them by grep after each edit.

- [ ] **Step 3: Update imports**

At the top of `src/deriva_ml/execution/execution.py`:

```python
# Before
from deriva_ml.core.enums import ..., Status, ...

# After — remove Status from the import list.
```

Add (if not present):

```python
from deriva_ml.execution.state_store import ExecutionStatus
```

- [ ] **Step 4: Run focused tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/test_execution.py -q 2>&1 | tail -20
```

Expected: some failures — `test_execution.py` contains legacy string-literal assertions like `assert status == "Initializing"` (lines 415-424). These are handled in Group D; for now confirm the failures are assertion-level, not import-level.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/execution.py
git commit -m "$(cat <<'EOF'
refactor(exe): migrate execution.py to ExecutionStatus

- Status.completed → Stopped (on algorithm end, line 1020)
- Status.completed → Uploaded (on upload end, line 1378)
- Status.initializing calls → logger.info (progress chatter)
- Status.running progress chatter → logger.info
- Status.running "Uploading execution files" → Pending_Upload (guarded)
- Status.failed → Failed (with error= kwarg)
- Status.pending "Initialize status finished" → deleted

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task C3: Migrate `src/deriva_ml/execution/execution_record.py` (v1 legacy record)

**Files:**
- Modify: `src/deriva_ml/execution/execution_record.py`

Note: this is the **v1 legacy** `execution_record.py`, separate from `execution_record_v2.py` which Group B already updated. The v1 file uses `Status` as the field type and may also have `update_status` call sites.

- [ ] **Step 1: Inspect**

```bash
grep -n "Status\.\|update_status\|import.*Status" src/deriva_ml/execution/execution_record.py
```

- [ ] **Step 2: Rewrite**

**Type annotation of `_status` field** (line 105): `_status: Status = PrivateAttr(default=Status.created)` → `_status: ExecutionStatus = PrivateAttr(default=ExecutionStatus.Created)`.

**`status` parameter default in `__init__`** (line 118): `status: Status = Status.created,` → `status: ExecutionStatus = ExecutionStatus.Created,`.

**`update_status` method** (around line 299): the legacy v1 record has its own `update_status` method. If it's still called somewhere, port it to the new signature. If it's dead code (v2 is what Phase 1 uses), delete it.

- [ ] **Step 3: Check for callers of v1 record's `update_status`**

```bash
grep -rn "ExecutionRecord\b.*update_status\|record\.update_status" src/deriva_ml/ tests/ | grep -v __pycache__ | grep -v execution_record_v2 | head
```

If only v2 is referenced (via `list_executions` → ExecutionRecord which is v2) — v1 is dead or nearly dead. If so, leave it alone and let Group E sweep it; just fix the `Status` type annotation.

- [ ] **Step 4: Apply the minimal changes (type annotations only if v1 is unused)**

```python
# At the top: swap import
from deriva_ml.execution.state_store import ExecutionStatus
# Remove: from deriva_ml.core.enums import Status

# In the class:
_status: ExecutionStatus = PrivateAttr(default=ExecutionStatus.Created)

# In __init__:
status: ExecutionStatus = ExecutionStatus.Created,
```

Leave the `update_status` method body for now if v1 is unused — Group E deletes the whole v1 file.

- [ ] **Step 5: Run**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_record_v2.py -q 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/deriva_ml/execution/execution_record.py
git commit -m "$(cat <<'EOF'
refactor(exe): migrate execution_record.py (v1) type annotations

Type annotation of _status field and status kwarg default swapped
to ExecutionStatus. Method bodies of the legacy v1 file left for
Group E deletion if v1 is confirmed dead.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task C4: Migrate `src/deriva_ml/dataset/dataset.py`

**Files:**
- Modify: `src/deriva_ml/dataset/dataset.py`

- [ ] **Step 1: Inspect**

```bash
grep -n "Status\.\|update_status" src/deriva_ml/dataset/dataset.py | head
```

Expected hits: lines 2666 and 2672 (`update_status(Status.running, msg)` via a callback).

- [ ] **Step 2: Rewrite**

Read the surrounding context:

```bash
sed -n '2650,2680p' src/deriva_ml/dataset/dataset.py
```

If `update_status` is passed as a callback parameter, the easiest rewrite is: replace it with `logger.info` calls. Progress chatter for dataset materialization doesn't need to flip the execution state.

```python
# Before
update_status(Status.running, msg)

# After
logger.info(msg)
```

Remove `Status` from imports at the top of dataset.py if no other reference remains.

- [ ] **Step 3: Run dataset tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/ -q 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/deriva_ml/dataset/dataset.py
git commit -m "$(cat <<'EOF'
refactor(dataset): migrate dataset.py progress callbacks to logger.info

The two update_status(Status.running, msg) callback invocations in
dataset materialization were progress chatter, not lifecycle
transitions. Replace with logger.info.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task C5: Migrate `src/deriva_ml/core/mixins/execution.py`

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py`

- [ ] **Step 1: Inspect**

```bash
grep -n "Status\.\|update_status" src/deriva_ml/core/mixins/execution.py | head
```

Expected hits: docstring examples referencing `Status.completed` etc.

- [ ] **Step 2: Rewrite docstrings**

Each docstring example that shows `Status.completed` / `Status.running` etc. — swap to `ExecutionStatus.Uploaded`, `ExecutionStatus.Running`, etc. per spec §3.1.

- [ ] **Step 3: Run mixin tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/test_execution_registry.py -q 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/deriva_ml/core/mixins/execution.py
git commit -m "$(cat <<'EOF'
refactor(mixins): update docstring examples to ExecutionStatus

Find/list/gc_executions docstring examples showed legacy Status.xxx;
swap to ExecutionStatus.xxx.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task C6: Sweep for any remaining src/ call sites

**Files:** entire `src/deriva_ml/`

- [ ] **Step 1: Grep for any remaining legacy references**

```bash
grep -rn "Status\.\(pending\|running\|completed\|initializing\|aborted\|failed\|created\)\|from deriva_ml\.core\.enums import.*Status\|from \.enums import.*Status" src/deriva_ml/ 2>&1 | grep -v __pycache__ | grep -v "ExecutionStatus\|PendingRowStatus\|status_store"
```

Expected: empty, or only a few residual references (maybe in comments or `core/enums.py` itself, which is deleted in Group E).

- [ ] **Step 2: Rewrite any remaining hits**

Apply the same mapping rules as C1–C5.

- [ ] **Step 3: Commit (only if any changes)**

```bash
git add src/deriva_ml/
git commit -m "refactor(status): sweep residual Status. references post C1-C5"
```

If no changes, skip.

---

## Task Group D — Test sweep

Every test that hardcoded lowercase status literals (`"running"`, `"pending_upload"`) needs an update. Same for tests referencing legacy `Status.xxx`.

### Task D1: Sweep `tests/execution/`

**Files:** all `tests/execution/test_*.py`

- [ ] **Step 1: Grep for the bad literals and legacy enum refs**

```bash
cd tests/execution/
grep -n '"\(created\|running\|stopped\|failed\|pending_upload\|uploaded\|aborted\)"' test_*.py | head -30
grep -n "Status\." test_*.py | grep -v "ExecutionStatus\|PendingRowStatus" | head -20
```

- [ ] **Step 2: Rewrite lowercase string literals to title-case**

`"running"` → `"Running"`, `"pending_upload"` → `"Pending_Upload"`, etc.

- [ ] **Step 3: Rewrite legacy `Status.xxx` references**

Same mapping rules as Group C. Where tests asserted `exe.status == Status.completed`, decide per context:
- `"Completed"` string with meaning "algorithm done" → `ExecutionStatus.Stopped` or `"Stopped"`.
- `"Completed"` with meaning "fully uploaded" → `ExecutionStatus.Uploaded` or `"Uploaded"`.

Test case examples:
- `tests/execution/test_execution.py:416`: `assert get_execution_status(ml, execution.execution_rid) == "Initializing"` — this entire sequence was testing legacy behavior; the new flow doesn't have "Initializing". Decide: either DELETE the assertion (Initializing is gone) or change it to `"Created"` if it's really asserting "execution exists, not started."
- `tests/execution/test_execution.py:424`: `assert get_execution_status(ml, execution.execution_rid) == "Completed"` — in context this is after execution_stop, so → `"Stopped"`.

- [ ] **Step 4: Run**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/ -q --timeout=600 2>&1 | tail -15
```

Expected: all PASS (or reduced to pre-existing skips).

- [ ] **Step 5: Commit**

```bash
git add tests/execution/
git commit -m "$(cat <<'EOF'
test(status): sweep tests/execution/ for title-case literals

Replaces lowercase status string literals ("running", "pending_upload")
with title-case ("Running", "Pending_Upload") and legacy Status.xxx
references with ExecutionStatus.Xxx.

Also: test_execution.py's legacy "Initializing" checks are dropped
(no such state in the new lifecycle) and "Completed" mappings are
per-context-decided per spec §3.1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task D2: Sweep `tests/integration/`

**Files:** `tests/integration/test_phase1_end_to_end.py`

- [ ] **Step 1: Find the SQLite-read-direct workaround**

```bash
grep -n "workaround\|read SQLite\|Uploaded\|uploaded" tests/integration/test_phase1_end_to_end.py
```

- [ ] **Step 2: Tighten the assertions**

Replace:

```python
# Before (H2 workaround)
store = ml_upload.workspace.execution_state_store()
final_row = store.get_execution(exe_rid)
assert final_row["status"] == "uploaded"

# After
exe_final = ml_upload.resume_execution(exe_rid)
assert exe_final.status is ExecutionStatus.Uploaded
assert not exe_final.pending_summary().has_pending
```

Also tighten the intermediate assertion:

```python
# Before
assert exe.status in (ExecutionStatus.stopped, ExecutionStatus.running)

# After
assert exe.status is ExecutionStatus.Stopped
```

- [ ] **Step 3: Run**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/integration/test_phase1_end_to_end.py -v --timeout=600
```

Expected: 2 PASS. The H2 workaround is gone; `resume_execution` now works correctly.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_phase1_end_to_end.py
git commit -m "$(cat <<'EOF'
test(integration): remove H2 SQLite-read workaround — resume_execution works

Phase 1's H2 integration test read SQLite directly as a workaround
for the legacy Status vs ExecutionStatus reconciliation bug. With
Phase 2 Subsystem 1's title-case enum + full migration, reconcile
works end-to-end; tighten the assertion to resume_execution +
exe_final.status is ExecutionStatus.Uploaded.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task D3: Sweep `tests/` root-level

**Files:** `tests/test_factories.py` and any other top-level test file

- [ ] **Step 1: Grep**

```bash
grep -n "Initializing\|\"running\"\|\"completed\"\|Status\." tests/test_*.py | head
```

Specifically `tests/test_factories.py:159`: `assert exe.status.value in ("Initializing", "Running")` — this depended on legacy title-case values and the legacy transient `Initializing` state. Since Initializing doesn't exist anymore:

```python
# Before
# Status may be Initializing or Running depending on timing
assert exe.status.value in ("Initializing", "Running")

# After
# In Phase 2, Created is the only pre-running state.
assert exe.status in (ExecutionStatus.Created, ExecutionStatus.Running)
```

- [ ] **Step 2: Apply rewrites**

- [ ] **Step 3: Run**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/test_factories.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test(factories): Initializing→Created per Phase 2 enum migration"
```

---

### Task D4: Verify test_execution.py full migration

**Files:** `tests/execution/test_execution.py`

The grep earlier showed substantial legacy test code here (docstrings citing "catalog-side Status vocab (title-case Initializing/Running/Completed)"). This needs a careful sweep.

- [ ] **Step 1: Read the affected tests**

```bash
sed -n '380,430p' tests/execution/test_execution.py
sed -n '1100,1130p' tests/execution/test_execution.py
sed -n '1240,1270p' tests/execution/test_execution.py
```

- [ ] **Step 2: Per test, decide the new assertion**

The test at line 386-424 walks through a legacy lifecycle:
- `"Initializing"` (post-create) → no longer exists → assert `"Created"`.
- `"Running"` (post-start) → `"Running"`.
- `"Completed"` (post-stop) → `"Stopped"`.

Same pattern in later similar tests.

- [ ] **Step 3: Apply rewrites**

Straightforward per-test sweep.

- [ ] **Step 4: Run**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/test_execution.py -q --timeout=600
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/execution/test_execution.py
git commit -m "$(cat <<'EOF'
test(exe): migrate test_execution.py from legacy Status strings

Lifecycle assertions walked through Initializing→Running→Completed.
With Phase 2: Created→Running→Stopped. Updated every checkpoint
(lines 415, 420, 424, 1117, 1254).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task Group E — Deletions

Only after all call sites and tests are migrated. Delete in this order: `_initialize_execution` first (it's self-contained); then `Status` enum last.

### Task E1: Delete `Execution._initialize_execution`

**Files:**
- Modify: `src/deriva_ml/execution/execution.py`

- [ ] **Step 1: Find the method**

```bash
grep -n "def _initialize_execution\|_initialize_execution(" src/deriva_ml/execution/execution.py
```

- [ ] **Step 2: Confirm no callers after Group C**

```bash
grep -rn "_initialize_execution" src/deriva_ml/ tests/ 2>&1 | grep -v __pycache__
```

Phase 1's `create_catalog_execution` should have taken over the setup work. If callers exist, understand them — maybe a caller should be rewritten to use `create_catalog_execution` or the call is already dead.

- [ ] **Step 3: Delete the method body**

Remove the entire `def _initialize_execution(self, ...)` method from `execution.py`.

- [ ] **Step 4: Run regression**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/ -q --timeout=600
```

Expected: PASS. If any test fails with "no attribute _initialize_execution", the method was still called somewhere — restore it and investigate.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/execution.py
git commit -m "$(cat <<'EOF'
refactor(exe): delete Execution._initialize_execution (superseded)

The Phase 1 create_catalog_execution path does the work _initialize_execution
was responsible for. With status-enum migration complete, the legacy
method has no remaining callers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task E2: Delete `Status` enum

**Files:**
- Modify: `src/deriva_ml/core/enums.py`

- [ ] **Step 1: Final grep for any remaining references**

```bash
grep -rn "Status\.\(pending\|running\|completed\|initializing\|aborted\|failed\|created\)\|from deriva_ml\.core\.enums import.*Status\|from \.enums import.*Status\|class Status" src/deriva_ml/ tests/ 2>&1 | grep -v __pycache__ | grep -v "ExecutionStatus\|PendingRowStatus"
```

Expected: only the `class Status` definition in `core/enums.py`. If other hits remain, go back to Group C or D and fix.

- [ ] **Step 2: Delete the `Status` class**

In `src/deriva_ml/core/enums.py`, remove lines 56-77 (the `class Status(StrEnum)` definition).

Also remove any `__all__` / re-export entry for `Status` if present.

- [ ] **Step 3: Run full regression**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/ -q --timeout=600 --ignore=tests/experiment/ 2>&1 | tail -10
```

Expected: PASS. Any ImportError for `Status` means a caller was missed — grep, fix, retry.

- [ ] **Step 4: Commit**

```bash
git add src/deriva_ml/core/enums.py
git commit -m "$(cat <<'EOF'
refactor(enums): delete legacy Status enum

All callers migrated in Groups B–D. The legacy title-case Status
enum used "Initializing"/"Pending"/"Completed" values that don't
match Phase 2's canonical ExecutionStatus lifecycle. Deleted.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task E3: Delete v1 `execution_record.py` if dead

**Files:**
- Possibly delete: `src/deriva_ml/execution/execution_record.py`

- [ ] **Step 1: Check usage**

```bash
grep -rn "from deriva_ml\.execution\.execution_record\b\|from \.execution_record\b\|execution_record\.ExecutionRecord\b" src/deriva_ml/ tests/ 2>&1 | grep -v __pycache__ | grep -v "execution_record_v2"
```

If no callers reference the v1 file (only v2), it's dead code.

- [ ] **Step 2: Confirm with a broader grep**

```bash
grep -rn "execution_record[^_]" src/deriva_ml/ tests/ 2>&1 | grep -v __pycache__ | grep -v "execution_record_v2\|test_execution_record" | head
```

If empty or only matches are unrelated (comments, etc.), delete.

- [ ] **Step 3: If dead, delete**

```bash
git rm src/deriva_ml/execution/execution_record.py
```

- [ ] **Step 4: Run regression**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/ -q --timeout=600 2>&1 | tail -5
```

Expected: PASS.

- [ ] **Step 5: Commit (if file deleted)**

```bash
git commit -m "$(cat <<'EOF'
refactor(exe): delete dead execution_record.py (v1)

execution_record_v2.py is the active version (Phase 1+); v1 had
no callers after the Group C Status migration. Delete.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If v1 is still referenced, skip this task — E3 becomes a follow-up.

---

## Task Group F — Grep gate + regression + CHANGELOG

### Task F1: Add the migration-complete grep gate

**Files:**
- Create: `tests/test_migration_complete.py`

- [ ] **Step 1: Write the test**

Create `tests/test_migration_complete.py`:

```python
"""Grep gate: fail if any Phase-2-Subsystem-1 legacy Status references remain.

Fails loudly if the library still contains lowercase ExecutionStatus
member references, legacy Status enum usage, or legacy value strings
that Phase 2 Subsystem 1 was supposed to purge.

Catches missed migrations before merge.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_PATTERNS = [
    # Legacy Status enum identifiers
    r"from deriva_ml\.core\.enums import.*\bStatus\b",
    r"from \.enums import.*\bStatus\b",
    r"class Status\b",
    # Legacy Status member refs
    r"\bStatus\.pending\b",
    r"\bStatus\.running\b",
    r"\bStatus\.completed\b",
    r"\bStatus\.initializing\b",
    r"\bStatus\.aborted\b",
    r"\bStatus\.failed\b",
    r"\bStatus\.created\b",
    # Lowercase ExecutionStatus member refs
    r"ExecutionStatus\.created\b",
    r"ExecutionStatus\.running\b",
    r"ExecutionStatus\.stopped\b",
    r"ExecutionStatus\.failed\b",
    r"ExecutionStatus\.pending_upload\b",
    r"ExecutionStatus\.uploaded\b",
    r"ExecutionStatus\.aborted\b",
]

SCAN_DIRS = ["src/deriva_ml"]


def test_no_legacy_status_references_in_src():
    """Fails if any FORBIDDEN_PATTERNS match any file under src/deriva_ml/."""
    # Exclude the test file itself (it lists the patterns as strings).
    # Exclude __pycache__.
    hits: list[str] = []
    for scan_dir in SCAN_DIRS:
        for pattern in FORBIDDEN_PATTERNS:
            cmd = [
                "grep", "-rEn", "--include=*.py",
                pattern,
                str(REPO_ROOT / scan_dir),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False,
            )
            # grep returns 0 on hit, 1 on no-hit, 2 on error.
            if result.returncode == 0 and result.stdout.strip():
                hits.append(f"pattern={pattern!r}\n{result.stdout}")
    assert not hits, (
        "Legacy Status references found in src/deriva_ml/:\n\n"
        + "\n".join(hits)
    )
```

- [ ] **Step 2: Run**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/test_migration_complete.py -v
```

Expected: PASS (after Groups A–E). If it fails, the printed output shows exactly which file:line still references legacy Status — fix each one and re-run until green.

- [ ] **Step 3: Commit**

```bash
git add tests/test_migration_complete.py
git commit -m "$(cat <<'EOF'
test(gate): grep gate for legacy Status references

Fails if any src/deriva_ml/ .py file still contains legacy Status
enum identifiers or member references, or lowercase ExecutionStatus
member refs. Catches missed migrations before merge.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task F2: Full regression

**Files:** none (verification only)

- [ ] **Step 1: Unit + CLI + integration**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest \
  tests/core/ tests/local_db/ tests/asset/ tests/model/ \
  tests/execution/ tests/cli/ tests/integration/ \
  tests/test_factories.py tests/test_migration_complete.py \
  -q --timeout=600 2>&1 | tail -10
```

Expected: all PASS plus 1 flake (`test_cli_help_lists_flags`, pre-existing distutils-shim).

- [ ] **Step 2: Lint**

```bash
uv run ruff check src/deriva_ml/execution/ src/deriva_ml/core/ tests/execution/ tests/test_migration_complete.py
```

Expected: clean on the files touched in Phase 2 Subsystem 1.

- [ ] **Step 3: No commit — verification only**

If any regression fails, return to the failing Group and fix.

---

### Task F3: CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write the entry**

Insert a new section near the top of `CHANGELOG.md`, above (or below) the existing Phase 1 section:

```markdown
## Unreleased — Phase 2 Subsystem 1: Status enum reconciliation

### Breaking changes

- **`ExecutionStatus` enum values are title-case.** `ExecutionStatus.Running` (was `.running`), `.value == "Running"` (was `"running"`). Identifiers and values both changed. Any code that hardcoded lowercase status strings (`"running"`, `"pending_upload"`) must update to title-case.
- **Legacy `Status` enum deleted.** The `deriva_ml.core.enums.Status` title-case enum has been removed entirely. Use `ExecutionStatus` from `deriva_ml.execution.state_store` instead.
- **`Execution.update_status(...)` signature changed.** Old: `update_status(status: Status, message: str)`. New: `update_status(target: ExecutionStatus, *, error: str | None = None)`. The second positional `message` parameter is gone. Error messages use the keyword-only `error=` argument, and only land on Failed/Aborted transitions; non-terminal transitions log a warning if `error=` is passed.
- **`ExecutionRecord.update_status(...)` signature**: new method. `update_status(target, *, ml: DerivaML, error: str | None = None)`. Use on record instances returned by `list_executions()`.
- **Legacy transient states dropped.** `Initializing` and `Pending` no longer exist. `Initializing` becomes `logger.info(...)` output; `Pending` was semantically redundant with `Created`.
- **`Completed` replaced with two states.** `Stopped` (algorithm finished, rows staged locally) and `Uploaded` (rows successfully persisted to the catalog). Legacy callers that transitioned to `Completed` must now pick the right phase-1 or phase-2 target explicitly.
- **`Execution._initialize_execution` deleted.** Create flows go through `create_catalog_execution` (Phase 1).
- **`DerivaML.status` attribute deleted.** Vestigial; had no readers.

### Migration notes

- Existing catalogs with historical rows containing `Status = "Initializing"` or `Status = "Pending"` will raise `DerivaMLStateInconsistency` on `resume_execution`. Users with such rows either clean them via `gc_executions` or update the `Status` field manually.
- The catalog's `Execution.Status_Detail` column is now never written. It remains in the schema; a future cleanup can drop it.

### Fixed

- `resume_execution` now correctly returns an Execution with a valid status instead of raising `DerivaMLStateInconsistency` (Phase 1's H2 integration test workaround is removed).
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs(changelog): Phase 2 Subsystem 1 — Status enum reconciliation

Documents the breaking changes: title-case ExecutionStatus values,
legacy Status enum deletion, new update_status signature, dropped
transient states (Initializing/Pending), and the Completed split
into Stopped/Uploaded.

Also notes migration implications for existing catalogs with legacy
status rows.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task Group G — Final review

### Task G1: Dispatch final code reviewer

**Files:** none (review only)

- [ ] **Step 1: Dispatch a `superpowers:code-reviewer` subagent**

Per `superpowers:subagent-driven-development`, after the final task completes, run a fresh code-review pass on the entire Phase-2-Subsystem-1 diff:

```
Subagent: superpowers:code-reviewer
Task: review the entire Phase 2 Subsystem 1 diff (commits from A1
through F3) against spec
docs/superpowers/specs/2026-04-21-status-enum-reconciliation-design.md.

Check: spec coverage (every section has an implementation or is
correctly marked out-of-scope), call-site sweep completeness (spec §5
table), test coverage (spec §8), grep gate passes, CHANGELOG accurate.

Report any gaps as blocker / important / nit. Implementer addresses
blockers and important issues before merge.
```

- [ ] **Step 2: Address reviewer findings**

Follow the subagent-driven-development pattern: re-dispatch the implementer to fix any blocker / important items, re-run the code reviewer until approved.

- [ ] **Step 3: Finish-branch**

Once approved, invoke `superpowers:finishing-a-development-branch` to finalize (PR → review → merge to main).

---

*(End of Task Group G — Phase 2 Subsystem 1 complete.)*

---

## Post-Subsystem-1

After Subsystem 1 merges, move to **Subsystem 3 (Upload engine completions)** — real `_invoke_deriva_py_uploader`, parallel_files / bandwidth_limit_mbps threading, UploadJob cancellation. Then **Subsystem 4 (Hygiene)** then **Subsystem 2 (Feature-consistency, the real Phase 2)**.
