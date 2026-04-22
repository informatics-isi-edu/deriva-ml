# H3 — ExecutionRecord Disambiguation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the V2 `ExecutionRecord` class to `ExecutionSnapshot`, move it into `execution_snapshot.py`, convert it from `@dataclass(frozen=True)` to Pydantic `BaseModel(frozen=True)`, and rewrite live-vs-snapshot docstrings on the four execution-query methods.

**Architecture:** Pure rename + idiom-consistency cleanup. No functional changes. Field list and semantics unchanged; only the container type changes on `ExecutionSnapshot` (dataclass → Pydantic) and the class/module names change. Internal alias `_ExecutionRecordV2` goes away.

**Tech Stack:** Python 3.12+, Pydantic v2 (`BaseModel`, `ConfigDict`), pytest.

**Spec:** `docs/superpowers/specs/2026-04-22-h3-executionrecord-rename-design.md`

---

## File Structure

| File | Role | Action |
|---|---|---|
| `src/deriva_ml/execution/execution_record_v2.py` | Current V2 home | **DELETE** (after moving content) |
| `src/deriva_ml/execution/execution_snapshot.py` | New home for `ExecutionSnapshot` | **CREATE** |
| `src/deriva_ml/execution/execution_record.py` | Legacy live class | Modify — class docstring adds live-vs-snapshot disambiguation paragraph |
| `src/deriva_ml/core/mixins/execution.py` | Imports V2, defines query methods | Modify — replace alias import, update type annotations, rewrite 4 method docstrings |
| `tests/execution/test_execution_record_v2.py` | Tests for V2 class | **RENAME** → `test_execution_snapshot.py`; update imports + one test name |
| `tests/execution/test_execution_registry.py` | Tests list_executions etc. | Modify — update import + one test name |
| `CHANGELOG.md` | Release notes | Modify — "Renamed" entry under Unreleased |

---

## Task Order Rationale

Tasks 1→2→3→4→5:
- **Task 1** creates `execution_snapshot.py` with the new Pydantic class. Old file stays for now — tests still pass via the old path.
- **Task 2** swaps the internal importer in `core/mixins/execution.py` to use the new name; old file becomes unreferenced.
- **Task 3** renames the test files and updates imports. Tests still pass on the same behavior.
- **Task 4** rewrites the docstrings (both class docstrings and all 4 method docstrings). Pure documentation.
- **Task 5** deletes `execution_record_v2.py` (now unreferenced) and CHANGELOG.

If Task 2 goes wrong (broken import, type-annotation typo) it's caught before Task 5 — the old file is still there as a safety net until the end.

---

### Task 1: Create `ExecutionSnapshot` as a Pydantic class in the new module

**Files:**
- Create: `src/deriva_ml/execution/execution_snapshot.py`

- [ ] **Step 1.1: Write the failing test**

Since this task creates a new file that duplicates an existing class, the "test" is an import-and-construct check. Create a temporary test file `tests/execution/test_execution_snapshot_new.py` (will be deleted in Task 3 when the real rename happens):

```python
"""Temporary test for Task 1 of H3. Deleted in Task 3."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import BaseModel


def test_execution_snapshot_exists_and_is_pydantic():
    from deriva_ml.execution.execution_snapshot import ExecutionSnapshot
    assert issubclass(ExecutionSnapshot, BaseModel)


def test_execution_snapshot_is_frozen():
    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.execution_snapshot import ExecutionSnapshot
    from deriva_ml.execution.state_store import ExecutionStatus

    now = datetime.now(timezone.utc)
    snap = ExecutionSnapshot(
        rid="X", workflow_rid=None, description=None,
        status=ExecutionStatus.Created, mode=ConnectionMode.online,
        working_dir_rel="execution/X",
        start_time=None, stop_time=None, last_activity=now,
        error=None, sync_pending=False, created_at=now,
        pending_rows=0, failed_rows=0, pending_files=0, failed_files=0,
    )
    with pytest.raises(Exception):  # Pydantic raises ValidationError
        snap.rid = "Y"


def test_execution_snapshot_from_row():
    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.execution_snapshot import ExecutionSnapshot
    from deriva_ml.execution.state_store import ExecutionStatus

    now = datetime.now(timezone.utc)
    row = {
        "rid": "EXE-A", "workflow_rid": "WFL-1", "description": "test",
        "status": "Stopped", "mode": "online",
        "working_dir_rel": "execution/EXE-A",
        "start_time": now, "stop_time": now, "last_activity": now,
        "error": None, "sync_pending": False, "created_at": now,
        "config_json": "{}",
    }
    snap = ExecutionSnapshot.from_row(
        row,
        pending_rows=3, failed_rows=0,
        pending_files=1, failed_files=0,
    )
    assert snap.rid == "EXE-A"
    assert snap.status is ExecutionStatus.Stopped
    assert snap.mode is ConnectionMode.online
    assert snap.pending_rows == 3


def test_execution_snapshot_model_dump():
    """Pydantic conversion gives .model_dump()."""
    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.execution_snapshot import ExecutionSnapshot
    from deriva_ml.execution.state_store import ExecutionStatus

    now = datetime.now(timezone.utc)
    snap = ExecutionSnapshot(
        rid="X", workflow_rid=None, description=None,
        status=ExecutionStatus.Created, mode=ConnectionMode.online,
        working_dir_rel="execution/X",
        start_time=None, stop_time=None, last_activity=now,
        error=None, sync_pending=False, created_at=now,
        pending_rows=0, failed_rows=0, pending_files=0, failed_files=0,
    )
    d = snap.model_dump()
    assert d["rid"] == "X"
    assert d["status"] == "Created"
    assert d["mode"] == "online"
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `cd /Users/carl/GitHub/deriva-ml/.claude/worktrees/h3-execution-record-rename && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_snapshot_new.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'deriva_ml.execution.execution_snapshot'`.

- [ ] **Step 1.3: Create `execution_snapshot.py`**

Create `src/deriva_ml/execution/execution_snapshot.py` with this exact content:

```python
"""SQLite-backed ExecutionSnapshot — a registry row with derived counts.

Per spec §2.9 (originally drafted as execution_record_v2). A frozen
Pydantic snapshot of one ``execution_state__executions`` row plus
convenience counts from ``execution_state__pending_rows``. Returned
by ``DerivaML.list_executions`` and ``DerivaML.find_incomplete_executions``.

This is a VALUE OBJECT, not a live catalog record:

- Constructed from a SQLite registry row (``from_row``) or directly.
- Immutable (``ConfigDict(frozen=True)``) — the snapshot captures a
  moment in time; mutating it would be meaningless.
- Reads do not contact the catalog; works in offline mode.
- Behavior methods (``pending_summary``, ``upload_outputs``,
  ``update_status``) take ``ml: DerivaML`` as a kwarg because the
  snapshot itself doesn't carry a server connection.

For a live, catalog-bound record whose property setters write
through to the catalog, see
:class:`~deriva_ml.execution.execution_record.ExecutionRecord`,
returned by :meth:`~deriva_ml.DerivaML.lookup_execution` and
:meth:`~deriva_ml.DerivaML.find_executions`.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from deriva_ml.core.connection_mode import ConnectionMode
from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.execution.state_store import ExecutionStatus

if TYPE_CHECKING:
    from deriva_ml.core.base import DerivaML  # noqa: F401
    from deriva_ml.execution.pending_summary import PendingSummary
    from deriva_ml.execution.upload_engine import UploadReport


class ExecutionSnapshot(BaseModel):
    """Frozen snapshot of an execution's registry row plus pending counts.

    A Pydantic value object — no mutation, no server reads on property
    access. If you need lifecycle fields that change over time (live
    status, etc.), use the Execution object returned by
    ``resume_execution`` or the
    :class:`~deriva_ml.execution.execution_record.ExecutionRecord`
    returned by ``lookup_execution``/``find_executions``.

    Attributes:
        rid: Server-assigned Execution RID.
        workflow_rid: Workflow FK; None if not set.
        description: Free-form description from the configuration.
        status: Current lifecycle status as of this snapshot.
        mode: ConnectionMode the execution was last active under.
        working_dir_rel: Relative path to the execution root.
        start_time: Lifecycle start timestamp; None if not yet started.
        stop_time: Lifecycle stop timestamp; None if still running.
        last_activity: Last pending-row mutation time.
        error: Last error message if status == Failed.
        sync_pending: True if SQLite is ahead of the catalog.
        created_at: When the local registry first knew about this row.
        pending_rows: Count of non-asset pending rows not yet uploaded.
        failed_rows: Count of non-asset rows in status='failed'.
        pending_files: Count of asset-file rows not yet uploaded.
        failed_files: Count of asset-file rows in status='failed'.

    Example:
        >>> snapshots = ml.find_incomplete_executions()
        >>> for snap in snapshots:
        ...     print(snap.rid, snap.status, snap.pending_rows)
    """

    model_config = ConfigDict(frozen=True)

    rid: str
    workflow_rid: str | None
    description: str | None
    status: ExecutionStatus
    mode: ConnectionMode
    working_dir_rel: str
    start_time: datetime | None
    stop_time: datetime | None
    last_activity: datetime
    error: str | None
    sync_pending: bool
    created_at: datetime
    pending_rows: int
    failed_rows: int
    pending_files: int
    failed_files: int

    @classmethod
    def from_row(
        cls,
        row: dict,
        *,
        pending_rows: int = 0,
        failed_rows: int = 0,
        pending_files: int = 0,
        failed_files: int = 0,
    ) -> "ExecutionSnapshot":
        """Construct from a SQLite executions row + pending counts.

        Args:
            row: Dict returned by ``ExecutionStateStore.get_execution``
                or ``list_executions``. Must contain all the
                ``execution_state__executions`` columns.
            pending_rows: Count of non-asset pending rows. Defaults to
                0 when the caller hasn't queried pending_rows.
            failed_rows: Count of non-asset rows in ``status='failed'``.
            pending_files: Count of asset-file rows not yet uploaded.
            failed_files: Count of asset-file rows in ``status='failed'``.

        Returns:
            A frozen ``ExecutionSnapshot`` instance.

        Example:
            >>> row = store.get_execution("EXE-A")
            >>> counts = store.count_pending_by_kind(execution_rid="EXE-A")
            >>> snap = ExecutionSnapshot.from_row(row, **counts)
        """
        return cls(
            rid=row["rid"],
            workflow_rid=row["workflow_rid"],
            description=row["description"],
            status=ExecutionStatus(row["status"]),
            mode=ConnectionMode(row["mode"]),
            working_dir_rel=row["working_dir_rel"],
            start_time=row["start_time"],
            stop_time=row["stop_time"],
            last_activity=row["last_activity"],
            error=row["error"],
            sync_pending=bool(row["sync_pending"]),
            created_at=row["created_at"],
            pending_rows=pending_rows,
            failed_rows=failed_rows,
            pending_files=pending_files,
            failed_files=failed_files,
        )

    def pending_summary(self, *, ml: "DerivaML") -> "PendingSummary":
        """Return a PendingSummary via the DerivaML instance's workspace.

        Snapshot objects are frozen Pydantic models and don't carry a
        reference to DerivaML; the caller passes one.

        Args:
            ml: The DerivaML instance whose workspace to query.

        Returns:
            PendingSummary for this execution.

        Example:
            >>> for snap in ml.list_executions():
            ...     s = snap.pending_summary(ml=ml)
            ...     if s.has_pending:
            ...         print(s.render())
        """
        from deriva_ml.execution.pending_summary import (
            PendingAssetCount,
            PendingRowCount,
            PendingSummary,
        )

        store = ml.workspace.execution_state_store()
        data = store.pending_summary_rows(execution_rid=self.rid)
        return PendingSummary(
            execution_rid=self.rid,
            rows=[PendingRowCount(**r) for r in data["rows"]],
            assets=[PendingAssetCount(**a) for a in data["assets"]],
            diagnostics=data["diagnostics"],
        )

    def upload_outputs(
        self,
        *,
        ml: "DerivaML",
        retry_failed: bool = False,
    ) -> "UploadReport":
        """Sugar for ``ml.upload_pending(execution_rids=[self.rid], ...)``.

        Snapshots are frozen Pydantic models — the caller provides the
        DerivaML instance that owns the workspace.
        """
        return ml.upload_pending(
            execution_rids=[self.rid],
            retry_failed=retry_failed,
        )

    def update_status(
        self,
        target: ExecutionStatus,
        *,
        ml: "DerivaML",
        error: str | None = None,
    ) -> None:
        """Transition this execution's status via the workspace state machine.

        Parallel to ``Execution.update_status``. The snapshot is a
        frozen Pydantic model and doesn't carry an ml reference —
        caller passes one.

        Args:
            target: Target ExecutionStatus enum member.
            ml: The DerivaML instance whose workspace owns the registry.
            error: For Failed/Aborted, a human-readable message.

        Raises:
            InvalidTransitionError: If the transition is not allowed.
            DerivaMLStateInconsistency: If catalog sync detects divergence.

        Example:
            >>> snap.update_status(ExecutionStatus.Aborted, ml=ml, error="user cancel")
        """
        from deriva_ml.execution.state_machine import transition

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
            catalog=ml.catalog if ml._mode is ConnectionMode.online else None,
            execution_rid=self.rid,
            current=current,
            target=target,
            mode=ml._mode,
            extra_fields=extra_fields,
        )
```

- [ ] **Step 1.4: Run the new test**

Run: `cd /Users/carl/GitHub/deriva-ml/.claude/worktrees/h3-execution-record-rename && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_snapshot_new.py -v`

Expected: all 4 tests PASS. If `test_execution_snapshot_is_frozen` fails with an unexpected assertion error, Pydantic v2's frozen-enforcement may raise differently — update the `except Exception` to the specific exception type seen.

- [ ] **Step 1.5: Verify the old class still works (no regressions)**

Run: `cd /Users/carl/GitHub/deriva-ml/.claude/worktrees/h3-execution-record-rename && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_record_v2.py -v`

Expected: 3 tests PASS (the old class is untouched).

- [ ] **Step 1.6: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.claude/worktrees/h3-execution-record-rename
git add src/deriva_ml/execution/execution_snapshot.py tests/execution/test_execution_snapshot_new.py
git commit -m "$(cat <<'EOF'
feat(execution): add ExecutionSnapshot (Pydantic replacement for V2 ExecutionRecord)

Task 1 of H3. Introduces the renamed Pydantic version of the V2
ExecutionRecord alongside the old dataclass version. The old file
stays in place for now — Task 2 swaps the importer, Task 5 deletes
the old file after all references migrate.

Fields and methods are identical to the dataclass version, with
three deliberate changes:

- @dataclass(frozen=True) → Pydantic BaseModel with
  ConfigDict(frozen=True). Aligns with the project convention that
  user-facing return types are Pydantic (see CLAUDE.md).
- Class name: ExecutionRecord → ExecutionSnapshot. Removes the
  naming collision with the live-catalog ExecutionRecord.
- Docstrings explicitly state "frozen Pydantic model" and
  cross-reference the live ExecutionRecord for callers looking for
  mutable records.

Temporary test file tests/execution/test_execution_snapshot_new.py
verifies the new class imports, is frozen, from_row works, and
.model_dump() is available. Will be absorbed into
test_execution_snapshot.py in Task 3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Swap the internal importer to the new class

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py` (lines 19-20 import, lines 324/356/384 references)

- [ ] **Step 2.1: Update the import**

In `src/deriva_ml/core/mixins/execution.py`, find (around line 19-20):

```python
from deriva_ml.execution.execution_record_v2 import (
    ExecutionRecord as _ExecutionRecordV2,
)
```

Replace with:

```python
from deriva_ml.execution.execution_snapshot import ExecutionSnapshot
```

- [ ] **Step 2.2: Update references**

Three references to `_ExecutionRecordV2` exist in the file. Replace each:

At line ~324 (in `list_executions`):
```python
    ) -> list[_ExecutionRecordV2]:
```
→
```python
    ) -> list[ExecutionSnapshot]:
```

At line ~356 (inside `list_executions`'s return comprehension):
```python
            _ExecutionRecordV2.from_row(
```
→
```python
            ExecutionSnapshot.from_row(
```

At line ~384 (in `find_incomplete_executions`):
```python
    def find_incomplete_executions(self) -> list[_ExecutionRecordV2]:
```
→
```python
    def find_incomplete_executions(self) -> list[ExecutionSnapshot]:
```

- [ ] **Step 2.3: Verify no `_ExecutionRecordV2` references remain in the file**

Run: `cd /Users/carl/GitHub/deriva-ml/.claude/worktrees/h3-execution-record-rename && grep -n "_ExecutionRecordV2" src/deriva_ml/core/mixins/execution.py`

Expected: no output.

- [ ] **Step 2.4: Run related tests**

Run: `cd /Users/carl/GitHub/deriva-ml/.claude/worktrees/h3-execution-record-rename && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_record_v2.py tests/execution/test_execution_registry.py tests/execution/test_execution_snapshot_new.py -v`

Expected: all PASS. The old V2 class tests still work (we haven't deleted it yet); the new class tests still work (the importer now uses ExecutionSnapshot, which is a real class); the registry tests (which construct `ExecutionSnapshot` via the mixin) pass because the mixin now emits `ExecutionSnapshot` instances.

Note: `test_list_executions_returns_dataclass` in `test_execution_registry.py` asserts `isinstance(rows[0], ExecutionRecord)` importing from `execution_record_v2` — that still works (the V2 class exists). Task 3 updates this test to use `ExecutionSnapshot`.

- [ ] **Step 2.5: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.claude/worktrees/h3-execution-record-rename
git add src/deriva_ml/core/mixins/execution.py
git commit -m "$(cat <<'EOF'
refactor(mixins): execution mixin emits ExecutionSnapshot instead of V2 ExecutionRecord

Task 2 of H3. Swaps the internal import in core/mixins/execution.py
from the V2 dataclass to the new Pydantic ExecutionSnapshot. No
behavior change — the fields, methods, and return values are
identical; only the concrete class changes.

list_executions and find_incomplete_executions now return
list[ExecutionSnapshot]. Callers using attribute access
(snap.rid, snap.status, etc.) continue to work unchanged.

The old execution_record_v2 module is still imported by test
files — Task 3 migrates those, Task 5 deletes the module.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Rename and update test files

**Files:**
- Rename: `tests/execution/test_execution_record_v2.py` → `tests/execution/test_execution_snapshot.py`
- Delete: `tests/execution/test_execution_snapshot_new.py` (temporary from Task 1)
- Modify: `tests/execution/test_execution_registry.py` (one import + one test name)

- [ ] **Step 3.1: Rename the test file**

Run: `cd /Users/carl/GitHub/deriva-ml/.claude/worktrees/h3-execution-record-rename && git mv tests/execution/test_execution_record_v2.py tests/execution/test_execution_snapshot.py`

- [ ] **Step 3.2: Update imports and class refs in the renamed file**

In `tests/execution/test_execution_snapshot.py`, update:

- Line 1 docstring: `"""Tests for the SQLite-backed ExecutionRecord dataclass."""` → `"""Tests for the SQLite-backed ExecutionSnapshot Pydantic model."""`
- Every import `from deriva_ml.execution.execution_record_v2 import ExecutionRecord` (3 sites) → `from deriva_ml.execution.execution_snapshot import ExecutionSnapshot`
- Every constructor `ExecutionRecord(` → `ExecutionSnapshot(`
- Every classmethod `ExecutionRecord.from_row(` → `ExecutionSnapshot.from_row(`
- Rename the three test functions for clarity:
  - `test_execution_record_has_registry_fields` → `test_execution_snapshot_has_registry_fields`
  - `test_execution_record_is_frozen` → `test_execution_snapshot_is_frozen`
  - `test_from_row_constructs_from_sqlite_dict` → `test_execution_snapshot_from_row_constructs_from_sqlite_dict`

Concrete sed command to do all substitutions (run from the worktree root):

```bash
cd /Users/carl/GitHub/deriva-ml/.claude/worktrees/h3-execution-record-rename
python3 <<'PYEOF'
from pathlib import Path
p = Path("tests/execution/test_execution_snapshot.py")
s = p.read_text()
s = s.replace(
    '"""Tests for the SQLite-backed ExecutionRecord dataclass."""',
    '"""Tests for the SQLite-backed ExecutionSnapshot Pydantic model."""',
)
s = s.replace(
    "from deriva_ml.execution.execution_record_v2 import ExecutionRecord",
    "from deriva_ml.execution.execution_snapshot import ExecutionSnapshot",
)
s = s.replace("ExecutionRecord(", "ExecutionSnapshot(")
s = s.replace("ExecutionRecord.from_row(", "ExecutionSnapshot.from_row(")
s = s.replace(
    "def test_execution_record_has_registry_fields",
    "def test_execution_snapshot_has_registry_fields",
)
s = s.replace(
    "def test_execution_record_is_frozen",
    "def test_execution_snapshot_is_frozen",
)
s = s.replace(
    "def test_from_row_constructs_from_sqlite_dict",
    "def test_execution_snapshot_from_row_constructs_from_sqlite_dict",
)
p.write_text(s)
print("done")
PYEOF
```

Also, `test_execution_snapshot_is_frozen` currently does `with pytest.raises((AttributeError, TypeError)):` — Pydantic v2's frozen models raise `pydantic.ValidationError`. Update that exception tuple:

```bash
python3 <<'PYEOF'
from pathlib import Path
p = Path("tests/execution/test_execution_snapshot.py")
s = p.read_text()
s = s.replace(
    "    with pytest.raises((AttributeError, TypeError)):",
    "    # Pydantic v2 frozen models raise ValidationError on mutation\n"
    "    from pydantic import ValidationError\n"
    "    with pytest.raises(ValidationError):",
)
p.write_text(s)
print("done")
PYEOF
```

- [ ] **Step 3.3: Delete the temporary test file from Task 1**

Run: `cd /Users/carl/GitHub/deriva-ml/.claude/worktrees/h3-execution-record-rename && git rm tests/execution/test_execution_snapshot_new.py`

The coverage provided by the temporary file is now fully covered by the renamed `test_execution_snapshot.py` plus one addition — add the `.model_dump()` check to the renamed file by appending this function at the end:

```python
def test_execution_snapshot_model_dump():
    """Pydantic conversion gives .model_dump() — consistent with other library types."""
    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.execution_snapshot import ExecutionSnapshot
    from deriva_ml.execution.state_store import ExecutionStatus

    now = datetime.now(timezone.utc)
    snap = ExecutionSnapshot(
        rid="X", workflow_rid=None, description=None,
        status=ExecutionStatus.Created, mode=ConnectionMode.online,
        working_dir_rel="execution/X",
        start_time=None, stop_time=None, last_activity=now,
        error=None, sync_pending=False, created_at=now,
        pending_rows=0, failed_rows=0, pending_files=0, failed_files=0,
    )
    d = snap.model_dump()
    assert d["rid"] == "X"
    assert d["status"] == "Created"
    assert d["mode"] == "online"
```

- [ ] **Step 3.4: Update `test_execution_registry.py`**

In `tests/execution/test_execution_registry.py`:

- Line 31 function name: `def test_list_executions_returns_dataclass(test_ml):` → `def test_list_executions_returns_snapshot(test_ml):`
- Line 32 import: `from deriva_ml.execution.execution_record_v2 import ExecutionRecord` → `from deriva_ml.execution.execution_snapshot import ExecutionSnapshot`
- Line 39 isinstance: `assert isinstance(rows[0], ExecutionRecord)` → `assert isinstance(rows[0], ExecutionSnapshot)`

- [ ] **Step 3.5: Verify no old references remain in tests**

Run:

```bash
cd /Users/carl/GitHub/deriva-ml/.claude/worktrees/h3-execution-record-rename
grep -rn "execution_record_v2" tests/
```

Expected: no output.

- [ ] **Step 3.6: Run the renamed and updated tests**

Run: `cd /Users/carl/GitHub/deriva-ml/.claude/worktrees/h3-execution-record-rename && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/test_execution_snapshot.py tests/execution/test_execution_registry.py -v`

Expected: all PASS (4 in `test_execution_snapshot.py`, several in `test_execution_registry.py`).

- [ ] **Step 3.7: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.claude/worktrees/h3-execution-record-rename
git add -A tests/execution/
git commit -m "$(cat <<'EOF'
test(execution): rename test_execution_record_v2.py → test_execution_snapshot.py

Task 3 of H3. Follows the class + module rename: the dedicated
tests for the V2 class now live in test_execution_snapshot.py and
exercise ExecutionSnapshot. Four changes:

- git mv the old file to the new name
- update imports + constructor + classmethod references
- rename the three test functions with 'snapshot' in their names
- update the frozen test to catch Pydantic ValidationError (v2
  frozen models raise that, not AttributeError/TypeError)
- add a new test_execution_snapshot_model_dump verifying the
  .model_dump() API is available

Also updates test_execution_registry.py:
- test_list_executions_returns_dataclass →
  test_list_executions_returns_snapshot
- import + isinstance check use ExecutionSnapshot

Removes the temporary test file from Task 1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Rewrite method and class docstrings

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py` (4 method docstrings)
- Modify: `src/deriva_ml/execution/execution_record.py` (class docstring — add disambiguation paragraph)

- [ ] **Step 4.1: Rewrite `lookup_execution` docstring**

In `src/deriva_ml/core/mixins/execution.py`, find the `lookup_execution` method (around line 233). Replace its docstring (between the first `"""` and the closing `"""`) with:

```
Look up a single execution by RID in the live catalog.

Queries the ERMrest catalog for the Execution row with the given
RID and returns an ``ExecutionRecord`` — a live, catalog-bound
value whose mutable properties (``status``, ``description``)
write through to the catalog on assignment. Online mode only.

For enumerating executions from the local SQLite registry without
touching the catalog, see ``list_executions()``. For catalog-side
filter queries returning live records, see ``find_executions()``.

Args:
    execution_rid: Resource Identifier (RID) of the execution.

Returns:
    A live ``ExecutionRecord`` bound to the catalog. Property
    setters (``record.status = ...``) write through.

Raises:
    DerivaMLException: If execution_rid is not valid or doesn't
        refer to an Execution record.

Example:
    >>> record = ml.lookup_execution("1-abc123")
    >>> record.status = ExecutionStatus.Uploaded   # writes to catalog
```

- [ ] **Step 4.2: Rewrite `find_executions` docstring**

In the same file, find `find_executions` (around line 566). Replace its docstring with:

```
Search the live catalog for executions matching the given filters.

Queries the ERMrest catalog (online only) and yields live,
catalog-bound ``ExecutionRecord`` objects for each match. Each
returned record's mutable properties (``status``, ``description``)
write through to the catalog on assignment.

For enumerating locally-known executions from the SQLite registry
without touching the catalog (works in offline mode), see
``list_executions()`` and ``find_incomplete_executions()``.

Args:
    workflow: Optional Workflow or RID to filter by.
    workflow_type: Optional workflow type name (e.g., "python_script").
    status: Optional ExecutionStatus to filter by.

Returns:
    Iterable of live ``ExecutionRecord`` objects.

Example:
    >>> for record in ml.find_executions(status=ExecutionStatus.Uploaded):
    ...     print(record.execution_rid, record.status)
```

- [ ] **Step 4.3: Rewrite `list_executions` docstring**

Find `list_executions` (around line 317). Replace its docstring with:

```
Enumerate locally-known executions from the SQLite registry.

Reads from the workspace SQLite registry — **no server contact**.
Works in both online and offline mode. Each returned
``ExecutionSnapshot`` is a frozen Pydantic value object captured
at query time; it cannot mutate the catalog. Pending-row counts
are included in the same pass.

For live catalog queries that return mutable
:class:`~deriva_ml.execution.execution_record.ExecutionRecord`
objects bound to the catalog, see ``find_executions()`` and
``lookup_execution()``.

Args:
    status: Single ExecutionStatus or list to filter; None = all.
    workflow_rid: Match only executions tagged with this Workflow
        RID; None = all.
    mode: ConnectionMode the execution was last active under;
        None = all.
    since: Return only executions with last_activity >= this
        timestamp (timezone-aware). None = no time filter.

Returns:
    List of ``ExecutionSnapshot`` Pydantic models — one per matching
    row in the registry. Empty list if nothing matches.

Example:
    >>> from deriva_ml.execution.state_store import ExecutionStatus
    >>> failed = ml.list_executions(status=ExecutionStatus.Failed)
    >>> for snap in failed:
    ...     print(snap.rid, snap.error)
```

- [ ] **Step 4.4: Rewrite `find_incomplete_executions` docstring**

Find `find_incomplete_executions` (around line 384). Replace its docstring with:

```
Sugar over :meth:`list_executions` for everything not terminally done.

Reads from the workspace SQLite registry — no server contact.
Returns executions in status in (Created, Running, Stopped, Failed,
Pending_Upload) — the set of things a user would want to either
resume, retry, or clean up. Excludes Uploaded (terminal success)
and Aborted (terminal cleanup).

For live catalog queries returning mutable
:class:`~deriva_ml.execution.execution_record.ExecutionRecord`
objects, see ``find_executions(status=...)``.

Returns:
    List of ``ExecutionSnapshot`` Pydantic models for each incomplete
    execution known to the local registry.

Example:
    >>> for snap in ml.find_incomplete_executions():
    ...     print(snap.rid, snap.status, snap.pending_rows)
```

- [ ] **Step 4.5: Add disambiguation paragraph to `ExecutionRecord` (live) class docstring**

In `src/deriva_ml/execution/execution_record.py`, find the class docstring on `class ExecutionRecord(BaseModel):`. Right after its one-line summary `"""Represents a catalog record for an execution.`, and before the next paragraph, insert this:

```

    A live, catalog-bound record. Property setters (``record.status = ...``,
    ``record.description = ...``) write through to the catalog on
    assignment; requires online mode for mutations. Returned by
    :meth:`~deriva_ml.DerivaML.lookup_execution` and
    :meth:`~deriva_ml.DerivaML.find_executions`.

    For a frozen snapshot value object that reads from the local
    SQLite registry and works offline, see
    :class:`~deriva_ml.execution.execution_snapshot.ExecutionSnapshot`
    returned by :meth:`~deriva_ml.DerivaML.list_executions` and
    :meth:`~deriva_ml.DerivaML.find_incomplete_executions`.
```

- [ ] **Step 4.6: Quick syntax check**

Run:

```bash
cd /Users/carl/GitHub/deriva-ml/.claude/worktrees/h3-execution-record-rename
uv run python -c "
from deriva_ml.core.mixins.execution import ExecutionMixin
from deriva_ml.execution.execution_record import ExecutionRecord
from deriva_ml.execution.execution_snapshot import ExecutionSnapshot
# Docstrings must mention the other class in each direction:
assert 'ExecutionSnapshot' in ExecutionMixin.list_executions.__doc__
assert 'ExecutionSnapshot' in ExecutionMixin.find_incomplete_executions.__doc__
assert 'ExecutionRecord' in ExecutionMixin.list_executions.__doc__
assert 'list_executions' in ExecutionMixin.find_executions.__doc__
assert 'list_executions' in ExecutionMixin.lookup_execution.__doc__
assert 'ExecutionSnapshot' in ExecutionRecord.__doc__
print('ok')
"
```

Expected: `ok`.

- [ ] **Step 4.7: Run tests (regression check)**

Run: `cd /Users/carl/GitHub/deriva-ml/.claude/worktrees/h3-execution-record-rename && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/ -q --timeout=120`

Expected: all pass.

- [ ] **Step 4.8: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.claude/worktrees/h3-execution-record-rename
git add src/deriva_ml/core/mixins/execution.py src/deriva_ml/execution/execution_record.py
git commit -m "$(cat <<'EOF'
docs(execution): disambiguate live vs snapshot in docstrings

Task 4 of H3. Rewrites the four execution-query method docstrings
(lookup_execution, find_executions, list_executions,
find_incomplete_executions) and adds a disambiguation paragraph to
the ExecutionRecord (live) class docstring.

Each docstring now states whether it reads from the live catalog
(ERMrest, online-only, mutable) or the local SQLite registry
(offline-safe, frozen) and cross-references the sibling methods so
readers arriving from either direction find the full picture.

The ExecutionSnapshot class docstring (added in Task 1) already
cross-references ExecutionRecord in the other direction.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Delete old module + CHANGELOG

**Files:**
- Delete: `src/deriva_ml/execution/execution_record_v2.py`
- Modify: `CHANGELOG.md`

- [ ] **Step 5.1: Final check — no remaining references to `execution_record_v2`**

Run:

```bash
cd /Users/carl/GitHub/deriva-ml/.claude/worktrees/h3-execution-record-rename
grep -rn "execution_record_v2\|_ExecutionRecordV2" src/ tests/ docs/ 2>/dev/null | grep -v "docs/superpowers/specs/\|docs/superpowers/plans/\|CHANGELOG.md"
```

Expected: no output. (The specs and plans for H3 are historical records that DO mention `execution_record_v2` — that's intentional and not an offender.)

- [ ] **Step 5.2: Delete the old module**

Run: `cd /Users/carl/GitHub/deriva-ml/.claude/worktrees/h3-execution-record-rename && git rm src/deriva_ml/execution/execution_record_v2.py`

- [ ] **Step 5.3: Add CHANGELOG entry**

In `CHANGELOG.md`, under a new "Unreleased — H3" section at the top of the file (above existing entries), insert:

```markdown
## Unreleased — H3: ExecutionRecord disambiguation

### Renamed

- **`deriva_ml.execution.execution_record_v2.ExecutionRecord`** → **`deriva_ml.execution.execution_snapshot.ExecutionSnapshot`**. The V2 class-name collided with the live catalog-backed `ExecutionRecord` and was aliased internally as `_ExecutionRecordV2` to compensate. The rename makes the two distinct concepts — live catalog record (mutable, ERMrest-backed) vs. frozen snapshot (value object, SQLite-backed) — clear at the import site. Also converted from `@dataclass(frozen=True)` to Pydantic `BaseModel(frozen=True)` so users get `.model_dump()` for free and the class matches the project convention for user-facing return types.
- The following methods now return `ExecutionSnapshot` (was: aliased `_ExecutionRecordV2`):
  - `DerivaML.list_executions(...)`
  - `DerivaML.find_incomplete_executions()`
- Test module **`tests/execution/test_execution_record_v2.py`** → **`tests/execution/test_execution_snapshot.py`**.

### External-caller impact

If you were importing the V2 class directly by its module path:

```python
# Before
from deriva_ml.execution.execution_record_v2 import ExecutionRecord

# After
from deriva_ml.execution.execution_snapshot import ExecutionSnapshot
```

Callers using only the public method API (`ml.list_executions(...)`, `ml.find_incomplete_executions()`) see no change beyond the class name in the return type — attribute access (`snap.rid`, `snap.status`, `snap.pending_rows`, etc.) and the behavior methods (`snap.upload_outputs(ml=...)`, `snap.update_status(..., ml=...)`, `snap.pending_summary(ml=...)`) are unchanged.

### Docstrings

The four execution-query methods (`lookup_execution`, `find_executions`, `list_executions`, `find_incomplete_executions`) and both `ExecutionRecord` (live) and `ExecutionSnapshot` (local) class docstrings now explicitly state whether they use the live catalog or the local SQLite registry, and cross-reference the sibling methods/classes.
```

- [ ] **Step 5.4: Regression test run**

Run: `cd /Users/carl/GitHub/deriva-ml/.claude/worktrees/h3-execution-record-rename && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/ -q --timeout=300`

Expected: all pass (similar counts to the pre-H3 baseline of 413 passed, 3 skipped in the fast-unit subset, plus the new `test_execution_snapshot.py` tests).

- [ ] **Step 5.5: Ruff check on touched files**

Run: `cd /Users/carl/GitHub/deriva-ml/.claude/worktrees/h3-execution-record-rename && uv run ruff check src/deriva_ml/execution/execution_snapshot.py src/deriva_ml/execution/execution_record.py src/deriva_ml/core/mixins/execution.py tests/execution/test_execution_snapshot.py tests/execution/test_execution_registry.py`

Expected: no new violations. (Pre-existing violations on these files are out of scope.)

- [ ] **Step 5.6: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.claude/worktrees/h3-execution-record-rename
git add -A
git commit -m "$(cat <<'EOF'
chore(execution): remove execution_record_v2.py + CHANGELOG

Task 5 of H3. All callers now import from execution_snapshot; the
old V2 module is unreferenced and deletable. CHANGELOG documents
the rename for external users who imported V2 by module path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Checklist

### Spec coverage

- §2 in-scope items (rename class/module/test-file, convert to Pydantic, update importers, docstring rewrites, CHANGELOG): all have tasks ✓
- §3.1 Pydantic-vs-dataclass rationale: Task 1 commit message cites it ✓
- §4 rename table: Task 1 creates snapshot.py, Task 2 updates mixin imports, Task 3 renames tests, Task 5 deletes v2.py ✓
- §5 docstring rewrites (4 method + 2 class): Task 1 does both class docstrings (ExecutionSnapshot inline, ExecutionRecord in Task 4), Task 4 does all 4 methods + adds disambiguation to ExecutionRecord ✓
- §6 testing plan ("all existing tests pass unchanged"): Task 2.4, 3.6, 4.7, 5.4 all run the relevant suites ✓

### Placeholder scan

- No TBD/TODO/"handle appropriately"
- Every step has exact file paths + exact code
- `sed` / `python3` substitution scripts are complete and runnable

### Type consistency

- `ExecutionSnapshot` used in: Task 1 class def, Task 2 type annotations, Task 3 test imports, Task 4 docstrings
- `model_dump` used in Task 1 test, Task 3 new test, CHANGELOG
- `frozen=True` via `ConfigDict(frozen=True)` everywhere (not the old `@dataclass(frozen=True)`)
- Pydantic `ValidationError` (not `FrozenInstanceError`) is the exception raised on mutation, and Task 3.2 updates the test accordingly

---

## Execution Handoff

Plan complete. Two execution options (per project convention):

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks.

**2. Inline execution** — run tasks in the main session.

Given H3's small size (5 tasks, mostly mechanical, ~300 LoC total), inline is probably fine. The decision is yours.
