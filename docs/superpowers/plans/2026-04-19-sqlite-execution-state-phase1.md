# SQLite-Backed Execution State — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the DerivaML workspace the authoritative local source of truth for execution state — enabling offline workflows, crash resumption, and async upload — by adding SQLite-backed execution registry, pending-row staging, and a restartable upload engine that drives deriva-py's existing uploader.

**Architecture:** Three new SQLAlchemy Core tables (`executions`, `pending_rows`, `directory_rules`) in the existing Workspace `main.db`. A `state_machine` module owns status transitions with catalog sync + `sync_pending` reconciliation. A new `upload_engine` drives `deriva.transfer.upload.deriva_upload` over SQLite-queued items. Read-through properties on `Execution` eliminate in-memory caching of lifecycle fields. Phase 1 covers the non-provisional 85% of the spec; Phase 2 (not in this plan) finalizes `TableHandle` / `AssetTableHandle` full surface after the feature-consistency review.

**Tech Stack:** Python 3.12+, SQLAlchemy Core (WAL-mode SQLite), Pydantic v2, deriva-py (uploader, pathBuilder, ERMrest_RID_Lease), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-04-18-sqlite-execution-state-design.md` (revisions 6 + follow-ups through §2.17).

---

## Conventions referenced throughout this plan

- **§2.14 Impl technology:** SQLAlchemy Core for new tables, shared `Workspace` engine, `execution_state__` table-name prefix, `with engine.begin() as conn:` for transactions.
- **§2.15 Docstrings:** every new public method has complete docstring with Args, Returns, Raises, and runnable Example.
- **§2.16 DRY hierarchy:** check deriva-ml → deriva-py → new. Specifically: drive `deriva.transfer.upload.deriva_upload` for uploads; use pathBuilder for ERMrest queries; `Table.define_association` for association tables.
- **§2.17 Naming:** `execution_rid: RID`, `dataset_rid: RID`, `workflow_rid: RID`, `asset_types` (plural), `asset_type` (singular), `schema`, `status`, `recurse`, `since`, `older_than`, `parallel_files`, `bandwidth_limit_mbps`, `retry_failed`, `progress`, `chunk_size`, `copy_file`/`copy_files`, `description`. Ordering: positional subject → `*,` → scoping → behavior → `**kwargs`.
- **R5.1:** hard cutover on renames, no shims. Breaking changes listed in `CHANGELOG.md`.
- **Environment:** `export PATH="/Users/carl/.local/bin:$PATH"` if `uv` not found. `DERIVA_ML_ALLOW_DIRTY=true` for tests during development. Use per-test `--timeout=300`.

---

## Task Group overview

The plan is structured into eight task groups. Within each group, tasks are bite-sized (2–5 min each).

| Group | Scope | Est. size |
|---|---|---|
| **A** | `ConnectionMode` enum + DerivaML mode parameter | 4 tasks |
| **B** | SQLite schema: `executions`, `pending_rows`, `directory_rules` tables | 5 tasks |
| **C** | Execution state machine module + `sync_pending` reconciliation | 7 tasks |
| **D** | Execution registry public API (`list_executions`, `resume_execution`, `gc_executions`, `find_incomplete_executions`) | 6 tasks |
| **E** | Execution read-through lifecycle properties + `DatasetCollection` + hierarchy renames | 5 tasks |
| **F** | RID leasing (lazy, batched, crash-safe) against `public:ERMrest_RID_Lease` | 6 tasks |
| **G** | Upload engine + `PendingSummary` + `UploadJob` + `ml.upload_pending` / `ml.start_upload` | 8 tasks |
| **H** | `deriva-ml upload` CLI + integration tests + CHANGELOG | 5 tasks |

**Total ≈ 46 tasks. Estimated 3–4 weeks of focused work.**

Provisional sections (§2.10 full table-handle surface, §2.11.2 step 6 feature-aware drain) get minimal Phase 1 implementation — just enough to let Groups F/G work. Full finalization is Phase 2, after the feature-consistency spec.

---

## Task Group A — `ConnectionMode` enum

Introduces the mode enum and threads it through `DerivaML.__init__`. No SQLite yet; this is the smallest independent slice.

### Task A1: Define `ConnectionMode` enum

**Files:**
- Create: `src/deriva_ml/core/connection_mode.py`
- Modify: `src/deriva_ml/__init__.py` (add export)
- Test: `tests/core/test_connection_mode.py`

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_connection_mode.py`:

```python
"""Tests for ConnectionMode enum."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from deriva_ml import ConnectionMode


def test_enum_members():
    assert ConnectionMode.online.value == "online"
    assert ConnectionMode.offline.value == "offline"
    assert list(ConnectionMode) == [ConnectionMode.online, ConnectionMode.offline]


def test_coerce_from_string():
    adapter = TypeAdapter(ConnectionMode)
    assert adapter.validate_python("online") is ConnectionMode.online
    assert adapter.validate_python("offline") is ConnectionMode.offline


def test_invalid_string_raises():
    adapter = TypeAdapter(ConnectionMode)
    with pytest.raises(ValidationError):
        adapter.validate_python("hybrid")


def test_str_representation_is_value():
    assert str(ConnectionMode.online) == "online"
    assert str(ConnectionMode.offline) == "offline"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_connection_mode.py -v
```

Expected: FAIL with `ImportError: cannot import name 'ConnectionMode' from 'deriva_ml'`.

- [ ] **Step 3: Implement `ConnectionMode`**

Create `src/deriva_ml/core/connection_mode.py`:

```python
"""Connection mode enumeration for DerivaML.

See spec §2.1 — online mode talks to the catalog eagerly; offline mode
stages all work locally in SQLite until an upload operation drains it.
"""

from __future__ import annotations

from enum import StrEnum


class ConnectionMode(StrEnum):
    """How a DerivaML instance interacts with the catalog.

    Members:
        online: Writes reach the catalog by the time the call returns
            (plain rows). Asset files still stage and wait for upload.
            Execution status transitions sync to the catalog atomically.
        offline: Every write stages into the workspace SQLite and stays
            there until upload. No server contact except for RID leases
            and the final upload.

    Example:
        >>> from deriva_ml import ConnectionMode, DerivaML
        >>> ml = DerivaML(hostname="example.org", catalog_id="42",
        ...               mode=ConnectionMode.offline)
        >>> ml.mode is ConnectionMode.offline
        True
    """

    online = "online"
    offline = "offline"

    def __str__(self) -> str:
        return self.value
```

- [ ] **Step 4: Export from `deriva_ml` package**

Edit `src/deriva_ml/__init__.py` — add to the appropriate export group (alphabetical within the block that already exports core types like `DerivaML`, `DerivaMLException`):

```python
from deriva_ml.core.connection_mode import ConnectionMode
```

Add `"ConnectionMode"` to the module's `__all__` if it maintains one.

- [ ] **Step 5: Run test to verify it passes**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_connection_mode.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/deriva_ml/core/connection_mode.py src/deriva_ml/__init__.py tests/core/test_connection_mode.py
git commit -m "feat(core): add ConnectionMode enum (online/offline)

Per spec §2.1. StrEnum so string literals 'online'/'offline' coerce
cleanly in pydantic validation. Foundation for mode-aware writes in
Group B+.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task A2: Add `mode` parameter to `DerivaML.__init__`

**Files:**
- Modify: `src/deriva_ml/core/base.py` (around current `DerivaML.__init__`)
- Test: `tests/core/test_connection_mode.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_connection_mode.py`:

```python
def test_derivaml_default_mode_is_online(test_ml):
    from deriva_ml import ConnectionMode
    assert test_ml.mode is ConnectionMode.online


def test_derivaml_accepts_mode_enum(deriva_catalog):
    from deriva_ml import ConnectionMode, DerivaML
    ml = DerivaML(
        hostname=deriva_catalog.host,
        catalog_id=deriva_catalog.catalog_id,
        working_dir=deriva_catalog.working_dir,
        mode=ConnectionMode.offline,
    )
    assert ml.mode is ConnectionMode.offline


def test_derivaml_accepts_mode_string(deriva_catalog):
    from deriva_ml import ConnectionMode, DerivaML
    ml = DerivaML(
        hostname=deriva_catalog.host,
        catalog_id=deriva_catalog.catalog_id,
        working_dir=deriva_catalog.working_dir,
        mode="offline",
    )
    assert ml.mode is ConnectionMode.offline


def test_derivaml_rejects_invalid_mode(deriva_catalog):
    from pydantic import ValidationError
    from deriva_ml import DerivaML
    with pytest.raises((ValidationError, ValueError)):
        DerivaML(
            hostname=deriva_catalog.host,
            catalog_id=deriva_catalog.catalog_id,
            working_dir=deriva_catalog.working_dir,
            mode="hybrid",
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_connection_mode.py -v -k derivaml
```

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'mode'`.

- [ ] **Step 3: Add `mode` to `DerivaML.__init__`**

In `src/deriva_ml/core/base.py`, locate `DerivaML.__init__`. Add `mode` as a keyword-only argument with default `ConnectionMode.online`, coerce string input, and store as `self._mode`.

```python
# Near top of base.py imports:
from deriva_ml.core.connection_mode import ConnectionMode

# In DerivaML.__init__ signature (add this parameter; preserve existing args):
def __init__(
    self,
    # ... existing args ...
    *,
    mode: ConnectionMode | str = ConnectionMode.online,
    # ... remaining args ...
):
    # ... existing init body ...
    self._mode = ConnectionMode(mode) if isinstance(mode, str) else mode

# Add property:
@property
def mode(self) -> ConnectionMode:
    """Current connection mode.

    Returns:
        The ConnectionMode this DerivaML instance was constructed with.
        Drives whether writes go live to the catalog (online) or stage
        in SQLite for later upload (offline). See spec §2.1.

    Example:
        >>> ml.mode is ConnectionMode.online
        True
    """
    return self._mode
```

- [ ] **Step 4: Run test to verify it passes**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_connection_mode.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/core/base.py tests/core/test_connection_mode.py
git commit -m "feat(core): DerivaML accepts mode parameter (default online)

Per spec §2.1. String literals are coerced to ConnectionMode. Mode is
exposed as a read-only property. Foundation for mode-gated writes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task A3: `DerivaMLOfflineError` exception

**Files:**
- Modify: `src/deriva_ml/core/exceptions.py`
- Test: `tests/core/test_exceptions.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_exceptions.py`:

```python
def test_offline_error_is_configuration_error():
    from deriva_ml.core.exceptions import (
        DerivaMLConfigurationError,
        DerivaMLOfflineError,
    )
    err = DerivaMLOfflineError("create_execution requires online mode")
    assert isinstance(err, DerivaMLConfigurationError)
    assert "create_execution" in str(err)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_exceptions.py::test_offline_error_is_configuration_error -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add exception class**

In `src/deriva_ml/core/exceptions.py`, add:

```python
class DerivaMLOfflineError(DerivaMLConfigurationError):
    """Raised when an operation that requires online mode is attempted
    while the DerivaML instance is in offline mode.

    Example:
        Creating an execution requires an online mode because the
        Execution RID must be server-assigned::

            >>> ml = DerivaML(..., mode=ConnectionMode.offline)
            >>> ml.create_execution(config)
            Traceback (most recent call last):
                ...
            DerivaMLOfflineError: create_execution requires online mode
    """
```

- [ ] **Step 4: Run test to verify it passes**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_exceptions.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/core/exceptions.py tests/core/test_exceptions.py
git commit -m "feat(exceptions): add DerivaMLOfflineError

Raised when an online-only operation (create_execution) is attempted
in offline mode. Subclass of DerivaMLConfigurationError.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task A4: `DerivaMLNoExecutionContext` and `DerivaMLStateInconsistency` exceptions

**Files:**
- Modify: `src/deriva_ml/core/exceptions.py`
- Test: `tests/core/test_exceptions.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_exceptions.py`:

```python
def test_no_execution_context_error_is_configuration_error():
    from deriva_ml.core.exceptions import (
        DerivaMLConfigurationError,
        DerivaMLNoExecutionContext,
    )
    err = DerivaMLNoExecutionContext("ml.table(...) handles are read-only")
    assert isinstance(err, DerivaMLConfigurationError)


def test_state_inconsistency_error_is_data_error():
    from deriva_ml.core.exceptions import (
        DerivaMLDataError,
        DerivaMLStateInconsistency,
    )
    err = DerivaMLStateInconsistency(
        "SQLite says running, catalog says aborted for EXE-A"
    )
    assert isinstance(err, DerivaMLDataError)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_exceptions.py -v -k "no_execution or state_inconsistency"
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add exception classes**

Add to `src/deriva_ml/core/exceptions.py`:

```python
class DerivaMLNoExecutionContext(DerivaMLConfigurationError):
    """Raised when an execution-scoped operation is attempted on a
    handle that was obtained without an execution context.

    Handles returned by `ml.table(name)` are read-only — useful for
    schema introspection — but their `.insert(...)` and asset-file
    methods raise this exception. Use `exe.table(name)` to get a
    handle bound to an execution that permits writes.

    Example:
        >>> handle = ml.table("Subject")
        >>> handle.record_class()              # OK
        >>> handle.insert({"Name": "x"})       # raises
        DerivaMLNoExecutionContext: ml.table() handles are read-only;
        use exe.table() for writes
    """


class DerivaMLStateInconsistency(DerivaMLDataError):
    """Raised when the workspace SQLite state and the catalog disagree
    in a way that cannot be reconciled automatically.

    The six disagreement cases enumerated in spec §2.2 are handled
    automatically; anything outside those rules surfaces as this
    exception with enough information for a human to intervene.

    Example:
        >>> exe = ml.resume_execution("EXE-A")
        DerivaMLStateInconsistency: Execution EXE-A: SQLite status
        'running' but catalog returned no Execution row
    """
```

- [ ] **Step 4: Run test to verify it passes**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_exceptions.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/core/exceptions.py tests/core/test_exceptions.py
git commit -m "feat(exceptions): add NoExecutionContext and StateInconsistency

NoExecutionContext raised by ml.table() handle writes (read-only
introspection surface). StateInconsistency raised by just-in-time
reconciliation when catalog/SQLite disagree outside the six
documented rules.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

*(End of Task Group A — `ConnectionMode` enum + mode parameter + new exceptions. Task Groups B–H follow in the plan document.)*

---

**Note for continuation:** This document will grow to include Task Groups B (SQLite schema), C (state machine), D (registry API), E (read-through properties + DatasetCollection + hierarchy renames), F (RID leasing), G (upload engine + PendingSummary + UploadJob), and H (CLI + integration tests + CHANGELOG). Each group follows the same bite-sized TDD structure as Group A.

The plan is split across multiple commits during authoring to keep review sizes reasonable. The full implementation plan is tracked by this single file; each commit adds one task group.
