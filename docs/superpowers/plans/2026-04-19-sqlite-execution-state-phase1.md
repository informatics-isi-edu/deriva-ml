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

*(End of Task Group A — `ConnectionMode` enum + mode parameter + new exceptions.)*

---

## Task Group B — SQLite schema

Creates the three new tables (`execution_state__executions`,
`execution_state__pending_rows`, `execution_state__directory_rules`)
using SQLAlchemy Core on the existing `Workspace.engine`. Matches the
existing `ManifestStore` pattern (Core `Table` + `MetaData`,
`CREATE TABLE IF NOT EXISTS` via `metadata.create_all()`, table-name
prefix `execution_state__`).

No execution lifecycle logic yet — this group just installs the
storage layer so Groups C+ have somewhere to write.

### Task B1: `ExecutionStateStore` module scaffold + table definitions

**Files:**
- Create: `src/deriva_ml/execution/state_store.py`
- Test: `tests/execution/test_state_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/execution/test_state_store.py`:

```python
"""Tests for ExecutionStateStore — SQLite-backed execution state."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect

from deriva_ml.execution.state_store import (
    EXECUTIONS_TABLE,
    PENDING_ROWS_TABLE,
    DIRECTORY_RULES_TABLE,
    ExecutionStateStore,
)


def _engine(tmp_path: Path):
    db = tmp_path / "test.db"
    return create_engine(f"sqlite:///{db}")


def test_table_name_constants_use_prefix():
    assert EXECUTIONS_TABLE == "execution_state__executions"
    assert PENDING_ROWS_TABLE == "execution_state__pending_rows"
    assert DIRECTORY_RULES_TABLE == "execution_state__directory_rules"


def test_ensure_schema_creates_three_tables(tmp_path):
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    inspector = inspect(eng)
    tables = set(inspector.get_table_names())
    assert EXECUTIONS_TABLE in tables
    assert PENDING_ROWS_TABLE in tables
    assert DIRECTORY_RULES_TABLE in tables


def test_ensure_schema_is_idempotent(tmp_path):
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    store.ensure_schema()  # second call must not raise


def test_executions_table_columns(tmp_path):
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    inspector = inspect(eng)
    cols = {c["name"] for c in inspector.get_columns(EXECUTIONS_TABLE)}
    expected = {
        "rid", "workflow_rid", "description", "config_json",
        "status", "mode", "working_dir_rel",
        "start_time", "stop_time", "last_activity",
        "error", "sync_pending", "created_at",
    }
    assert expected.issubset(cols), f"missing: {expected - cols}"


def test_pending_rows_table_columns(tmp_path):
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    inspector = inspect(eng)
    cols = {c["name"] for c in inspector.get_columns(PENDING_ROWS_TABLE)}
    expected = {
        "id", "execution_rid", "key",
        "target_schema", "target_table",
        "rid", "lease_token", "metadata_json",
        "asset_file_path", "asset_types_json", "description",
        "status", "error",
        "created_at", "leased_at", "uploaded_at",
        "rule_id",
    }
    assert expected.issubset(cols), f"missing: {expected - cols}"


def test_directory_rules_table_columns(tmp_path):
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    inspector = inspect(eng)
    cols = {c["name"] for c in inspector.get_columns(DIRECTORY_RULES_TABLE)}
    expected = {
        "id", "execution_rid",
        "target_schema", "target_table",
        "source_dir",
        "glob", "recurse", "copy_files",
        "asset_types_json", "status", "created_at",
    }
    assert expected.issubset(cols), f"missing: {expected - cols}"


def test_indexes_created(tmp_path):
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    inspector = inspect(eng)
    pending_indexes = {idx["name"] for idx in inspector.get_indexes(PENDING_ROWS_TABLE)}
    # Expect at least indexes for (execution_rid, status) and (execution_rid, target_table)
    assert any("execution_rid" in n and "status" in n for n in pending_indexes), \
        f"missing (execution_rid, status) index; have: {pending_indexes}"
    assert any("execution_rid" in n and "target_table" in n for n in pending_indexes), \
        f"missing (execution_rid, target_table) index; have: {pending_indexes}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_store.py -v
```

Expected: FAIL with `ModuleNotFoundError: deriva_ml.execution.state_store`.

- [ ] **Step 3: Implement `ExecutionStateStore`**

Create `src/deriva_ml/execution/state_store.py`:

```python
"""SQLite-backed store for execution state.

Defines three tables in the workspace main.db:
- execution_state__executions: per-execution registry row
- execution_state__pending_rows: rows staged for catalog insert
- execution_state__directory_rules: registered asset directories

Uses SQLAlchemy Core (no ORM) matching the codebase pattern for
library-bookkeeping tables (ManifestStore, ResultCache). See spec
§2.14 for rationale. Schemas are stable; runtime discovery is not
needed here.

All writes use ``with engine.begin() as conn:`` for atomic
per-transaction commit. WAL mode + per-mutation fsync is inherited
from the Workspace engine configuration.
"""

from __future__ import annotations

import logging

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

EXECUTIONS_TABLE = "execution_state__executions"
PENDING_ROWS_TABLE = "execution_state__pending_rows"
DIRECTORY_RULES_TABLE = "execution_state__directory_rules"


class ExecutionStateStore:
    """SQLAlchemy Core wrapper for the three execution-state tables.

    Owns the MetaData and Table definitions but not the engine — the
    engine is provided by the caller (typically ``Workspace.engine``)
    so all library-bookkeeping tables live in a single main.db.

    Usage:
        >>> store = ExecutionStateStore(engine=workspace.engine)
        >>> store.ensure_schema()
        >>> # then use store.executions, store.pending_rows,
        >>> # store.directory_rules for queries.

    Attributes:
        engine: The shared SQLAlchemy Engine.
        metadata: MetaData object holding the three table definitions.
        executions: The sqlalchemy.Table for executions.
        pending_rows: The sqlalchemy.Table for pending_rows.
        directory_rules: The sqlalchemy.Table for directory_rules.
    """

    def __init__(self, engine: Engine) -> None:
        """Bind the store to an existing Engine.

        Args:
            engine: A SQLAlchemy Engine — typically obtained from
                ``Workspace.engine``. The store does not manage the
                engine's lifecycle; the caller disposes it.
        """
        self.engine = engine
        self.metadata = MetaData()

        # executions — see spec §2.5.1 for column purposes.
        # status values: created|running|stopped|failed|pending_upload|uploaded|aborted
        # mode values: online|offline
        self.executions = Table(
            EXECUTIONS_TABLE, self.metadata,
            Column("rid", String, primary_key=True),
            Column("workflow_rid", String, nullable=True),
            Column("description", Text, nullable=True),
            Column("config_json", Text, nullable=False),
            Column("status", String, nullable=False),
            Column("mode", String, nullable=False),
            Column("working_dir_rel", String, nullable=False),
            Column("start_time", DateTime(timezone=True), nullable=True),
            Column("stop_time", DateTime(timezone=True), nullable=True),
            Column("last_activity", DateTime(timezone=True), nullable=False),
            Column("error", Text, nullable=True),
            Column("sync_pending", Boolean, nullable=False, default=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Index("ix_executions_status", "status"),
            Index("ix_executions_workflow_rid", "workflow_rid"),
            Index("ix_executions_last_activity", "last_activity"),
            # Partial index: most rows have sync_pending=False, so a
            # filtered index keeps lookups of pending-sync rows fast
            # without bloating storage.
            Index(
                "ix_executions_sync_pending",
                "sync_pending",
                sqlite_where=Column("sync_pending"),
            ),
        )

        # pending_rows — see spec §2.5.2. status values:
        # staged|leasing|leased|uploading|uploaded|failed
        self.pending_rows = Table(
            PENDING_ROWS_TABLE, self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column(
                "execution_rid", String,
                ForeignKey(f"{EXECUTIONS_TABLE}.rid"),
                nullable=False,
            ),
            Column("key", String, nullable=False),
            Column("target_schema", String, nullable=False),
            Column("target_table", String, nullable=False),
            Column("rid", String, nullable=True),
            Column("lease_token", String, nullable=True),
            Column("metadata_json", Text, nullable=False),
            Column("asset_file_path", String, nullable=True),
            Column("asset_types_json", Text, nullable=True),
            Column("description", Text, nullable=True),
            Column("status", String, nullable=False),
            Column("error", Text, nullable=True),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("leased_at", DateTime(timezone=True), nullable=True),
            Column("uploaded_at", DateTime(timezone=True), nullable=True),
            Column(
                "rule_id", Integer,
                ForeignKey(f"{DIRECTORY_RULES_TABLE}.id"),
                nullable=True,
            ),
            Index("ix_pending_execution_status", "execution_rid", "status"),
            Index("ix_pending_execution_table", "execution_rid", "target_table"),
        )

        # directory_rules — see spec §2.5.3.
        # status values: active|closed
        self.directory_rules = Table(
            DIRECTORY_RULES_TABLE, self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column(
                "execution_rid", String,
                ForeignKey(f"{EXECUTIONS_TABLE}.rid"),
                nullable=False,
            ),
            Column("target_schema", String, nullable=False),
            Column("target_table", String, nullable=False),
            Column("source_dir", String, nullable=False),
            Column("glob", String, nullable=False),
            Column("recurse", Boolean, nullable=False, default=False),
            Column("copy_files", Boolean, nullable=False, default=False),
            Column("asset_types_json", Text, nullable=True),
            Column("status", String, nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Index("ix_directory_rules_execution", "execution_rid"),
        )

    def ensure_schema(self) -> None:
        """Create the three tables if they don't already exist.

        Idempotent — safe to call on every DerivaML construction. Uses
        SQLAlchemy's ``create_all`` which issues ``CREATE TABLE IF
        NOT EXISTS`` via dialect-specific SQL, matching the existing
        Workspace pattern (see ManifestStore.ensure_schema).

        Example:
            >>> store = ExecutionStateStore(engine=workspace.engine)
            >>> store.ensure_schema()
            >>> # Tables now exist; safe to insert/select.
        """
        self.metadata.create_all(self.engine)
        logger.debug("execution_state schema ensured on %s", self.engine.url)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_store.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/state_store.py tests/execution/test_state_store.py
git commit -m "feat(execution): add ExecutionStateStore with 3 SQLite tables

Per spec §2.5 + §2.14. SQLAlchemy Core tables on the workspace
engine, execution_state__ prefix matching ManifestStore/ResultCache
convention. Indexes on (execution_rid, status), (execution_rid,
target_table), executions.status, sync_pending partial index.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task B2: `Workspace.execution_state_store()` accessor

**Files:**
- Modify: `src/deriva_ml/local_db/workspace.py`
- Test: `tests/local_db/test_workspace.py` (extend) or `tests/execution/test_state_store.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/execution/test_state_store.py`:

```python
def test_workspace_exposes_execution_state_store(tmp_path):
    from deriva_ml.local_db.workspace import Workspace

    ws = Workspace(
        working_dir=tmp_path,
        hostname="test.example.org",
        catalog_id="1",
    )
    try:
        store = ws.execution_state_store()
        assert isinstance(store, ExecutionStateStore)
        # Tables must exist — ensure_schema ran.
        from sqlalchemy import inspect
        inspector = inspect(ws.engine)
        assert EXECUTIONS_TABLE in inspector.get_table_names()
        # Second call returns the same instance (cached).
        assert ws.execution_state_store() is store
    finally:
        ws.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_store.py::test_workspace_exposes_execution_state_store -v
```

Expected: FAIL with `AttributeError: 'Workspace' object has no attribute 'execution_state_store'`.

- [ ] **Step 3: Add accessor to `Workspace`**

In `src/deriva_ml/local_db/workspace.py`:

1. Add the import near the other local-db imports (around line 22–32):

   ```python
   if TYPE_CHECKING:
       from deriva_ml.execution.state_store import ExecutionStateStore
       # ... existing TYPE_CHECKING imports preserved ...
   ```

2. Add a private cache slot in `Workspace.__init__` (where other cached
   accessors like `_manifest_store` are initialized; see the existing
   pattern in this file around line 100–120):

   ```python
   self._execution_state_store: "ExecutionStateStore | None" = None
   ```

3. Add the accessor method (next to `manifest_store` / `result_cache`
   methods, matching their style):

   ```python
   def execution_state_store(self) -> "ExecutionStateStore":
       """Return the ExecutionStateStore for this workspace, creating
       schema on first access.

       Cached — subsequent calls return the same instance. The store
       lives on this Workspace's shared engine; the three
       ``execution_state__*`` tables coexist with ManifestStore and
       ResultCache tables in a single main.db, so cross-table
       transactions are atomic.

       Returns:
           The ExecutionStateStore bound to ``self.engine`` with
           schema ensured.

       Example:
           >>> ws = Workspace(working_dir=".", hostname="h", catalog_id="1")
           >>> store = ws.execution_state_store()
           >>> store.executions   # the sqlalchemy.Table
       """
       if self._execution_state_store is None:
           # Lazy import to avoid import cycle between workspace.py
           # and execution/state_store.py (state_store may later
           # reference workspace types).
           from deriva_ml.execution.state_store import ExecutionStateStore

           store = ExecutionStateStore(engine=self.engine)
           store.ensure_schema()
           self._execution_state_store = store
       return self._execution_state_store
   ```

- [ ] **Step 4: Run test to verify it passes**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_store.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/local_db/workspace.py tests/execution/test_state_store.py
git commit -m "feat(workspace): add execution_state_store() accessor

Cached per-Workspace ExecutionStateStore bound to the shared engine.
Schema ensured on first access. Matches existing manifest_store /
result_cache accessor pattern.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task B3: Status enum constants (`ExecutionStatus`, `PendingRowStatus`, `DirectoryRuleStatus`)

**Files:**
- Modify: `src/deriva_ml/execution/state_store.py`
- Test: `tests/execution/test_state_store.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/execution/test_state_store.py`:

```python
def test_execution_status_values():
    from deriva_ml.execution.state_store import ExecutionStatus
    assert ExecutionStatus.created.value == "created"
    assert ExecutionStatus.running.value == "running"
    assert ExecutionStatus.stopped.value == "stopped"
    assert ExecutionStatus.failed.value == "failed"
    assert ExecutionStatus.pending_upload.value == "pending_upload"
    assert ExecutionStatus.uploaded.value == "uploaded"
    assert ExecutionStatus.aborted.value == "aborted"


def test_pending_row_status_values():
    from deriva_ml.execution.state_store import PendingRowStatus
    for name in ["staged", "leasing", "leased",
                 "uploading", "uploaded", "failed"]:
        assert getattr(PendingRowStatus, name).value == name


def test_directory_rule_status_values():
    from deriva_ml.execution.state_store import DirectoryRuleStatus
    assert DirectoryRuleStatus.active.value == "active"
    assert DirectoryRuleStatus.closed.value == "closed"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_store.py -v -k status
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add the enums**

In `src/deriva_ml/execution/state_store.py`, add at the top (below
imports, above the table-name constants):

```python
from enum import StrEnum


class ExecutionStatus(StrEnum):
    """Lifecycle status for an Execution (see spec §2.2).

    Transitions are:
        created → running → {stopped, failed} →
            {pending_upload → {uploaded, failed}}
        created → aborted
        running → aborted

    Values are lowercase strings for direct storage in SQLite and for
    clean comparison against ERMrest's Status vocabulary terms.
    """
    created = "created"
    running = "running"
    stopped = "stopped"
    failed = "failed"
    pending_upload = "pending_upload"
    uploaded = "uploaded"
    aborted = "aborted"


class PendingRowStatus(StrEnum):
    """Per-pending-row status (see spec §2.5.2).

    Transitions are:
        staged → leasing → leased → uploading → {uploaded, failed}
    """
    staged = "staged"
    leasing = "leasing"
    leased = "leased"
    uploading = "uploading"
    uploaded = "uploaded"
    failed = "failed"


class DirectoryRuleStatus(StrEnum):
    """Per-directory-rule status (see spec §2.5.3).

    A rule is `active` until `close()` is called; closed rules reject
    further register/scan calls but their existing pending_rows can
    still drain.
    """
    active = "active"
    closed = "closed"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_store.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/state_store.py tests/execution/test_state_store.py
git commit -m "feat(execution): status enums (Execution, PendingRow, DirectoryRule)

StrEnums so string literals coerce in pydantic. Used by the state
machine, upload engine, and handle APIs in subsequent groups.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task B4: Basic CRUD helpers on `ExecutionStateStore`

These low-level helpers are the only thing Group C's state machine
needs to transact. Keep them narrow; richer APIs live higher up.

**Files:**
- Modify: `src/deriva_ml/execution/state_store.py`
- Test: `tests/execution/test_state_store.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/execution/test_state_store.py`:

```python
def test_insert_execution_row(tmp_path):
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A",
        workflow_rid="WFL-1",
        description="test",
        config_json='{"foo": "bar"}',
        status=ExecutionStatus.created,
        mode=ConnectionMode.online,
        working_dir_rel="execution/EXE-A",
        created_at=now,
        last_activity=now,
    )

    row = store.get_execution("EXE-A")
    assert row is not None
    assert row["rid"] == "EXE-A"
    assert row["status"] == "created"
    assert row["mode"] == "online"


def test_get_execution_missing_returns_none(tmp_path):
    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    assert store.get_execution("NOPE") is None


def test_update_execution_status(tmp_path):
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.created,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    store.update_execution(
        rid="EXE-A",
        status=ExecutionStatus.running,
        start_time=now,
        sync_pending=True,
    )

    row = store.get_execution("EXE-A")
    assert row["status"] == "running"
    assert row["sync_pending"] is True
    assert row["start_time"] is not None


def test_list_executions_filters_by_status(tmp_path):
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()

    now = datetime.now(timezone.utc)
    for rid, status in [
        ("A", ExecutionStatus.running),
        ("B", ExecutionStatus.stopped),
        ("C", ExecutionStatus.uploaded),
    ]:
        store.insert_execution(
            rid=rid, workflow_rid=None, description=None,
            config_json="{}", status=status,
            mode=ConnectionMode.online, working_dir_rel=f"execution/{rid}",
            created_at=now, last_activity=now,
        )

    rows = store.list_executions(status=[ExecutionStatus.running, ExecutionStatus.stopped])
    rids = {r["rid"] for r in rows}
    assert rids == {"A", "B"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_store.py -v -k "execution and (insert or get or update or list)"
```

Expected: FAIL with `AttributeError: 'ExecutionStateStore' object has no attribute 'insert_execution'`.

- [ ] **Step 3: Add CRUD methods**

In `src/deriva_ml/execution/state_store.py`, append methods to
`ExecutionStateStore`:

```python
    # ─── executions CRUD ────────────────────────────────────────────

    def insert_execution(
        self,
        *,
        rid: str,
        workflow_rid: str | None,
        description: str | None,
        config_json: str,
        status: ExecutionStatus,
        mode: "ConnectionMode",
        working_dir_rel: str,
        created_at: datetime,
        last_activity: datetime,
        sync_pending: bool = False,
        start_time: datetime | None = None,
        stop_time: datetime | None = None,
        error: str | None = None,
    ) -> None:
        """Insert a new row in the executions table.

        Idempotency is the caller's concern — this method fails if the
        rid already exists (PK constraint).

        Args:
            rid: Server-assigned Execution RID.
            workflow_rid: Workflow FK, or None if not yet attached.
            description: Human-readable description from config.
            config_json: Serialized ExecutionConfiguration.
            status: Initial status, typically ExecutionStatus.created.
            mode: ConnectionMode the execution is active under.
            working_dir_rel: Path relative to the workspace root.
            created_at: UTC timestamp when the row is written.
            last_activity: Starts equal to created_at; updated on every
                pending-row mutation.
            sync_pending: True if this row is ahead of the catalog.
            start_time / stop_time / error: Populated later by state
                transitions; None at insert time.

        Raises:
            sqlalchemy.exc.IntegrityError: If rid already exists.
        """
        from sqlalchemy import insert

        with self.engine.begin() as conn:
            conn.execute(
                insert(self.executions).values(
                    rid=rid, workflow_rid=workflow_rid,
                    description=description, config_json=config_json,
                    status=str(status), mode=str(mode),
                    working_dir_rel=working_dir_rel,
                    start_time=start_time, stop_time=stop_time,
                    last_activity=last_activity, error=error,
                    sync_pending=sync_pending, created_at=created_at,
                )
            )

    def get_execution(self, rid: str) -> dict | None:
        """Return the executions row as a dict, or None if absent.

        Args:
            rid: The execution RID to look up.

        Returns:
            A dict mapping column names to values, or None if no row
            matches. Datetime columns are returned as Python datetime
            objects (timezone-aware).

        Example:
            >>> row = store.get_execution("EXE-A")
            >>> row["status"] if row else None
            'running'
        """
        from sqlalchemy import select

        with self.engine.connect() as conn:
            result = conn.execute(
                select(self.executions).where(self.executions.c.rid == rid)
            ).mappings().first()
        return dict(result) if result is not None else None

    def update_execution(
        self,
        rid: str,
        **fields: object,
    ) -> None:
        """Partial update of an executions row.

        Any column name from the executions table may be passed as
        a kwarg. Status values are coerced to strings automatically.

        Args:
            rid: The execution to update.
            **fields: Columns to set. Missing columns are left alone.

        Raises:
            KeyError: If a kwarg doesn't match a column in the
                executions table.
        """
        valid_cols = {c.name for c in self.executions.columns}
        unknown = set(fields) - valid_cols
        if unknown:
            raise KeyError(f"unknown columns on executions: {unknown}")

        # Coerce enum values to strings — the table columns are plain
        # String, not Enum, so SQLAlchemy won't auto-coerce.
        coerced = {
            k: str(v) if isinstance(v, ExecutionStatus) else v
            for k, v in fields.items()
        }

        from sqlalchemy import update

        with self.engine.begin() as conn:
            conn.execute(
                update(self.executions)
                .where(self.executions.c.rid == rid)
                .values(**coerced)
            )

    def list_executions(
        self,
        *,
        status: "ExecutionStatus | list[ExecutionStatus] | None" = None,
        workflow_rid: str | None = None,
        mode: "ConnectionMode | None" = None,
        since: datetime | None = None,
    ) -> list[dict]:
        """Filter the executions table and return rows as dicts.

        Args:
            status: Single status or list of statuses to match, or
                None for all.
            workflow_rid: Match only executions attached to this
                workflow, or None for all.
            mode: Match only executions active under this mode.
            since: Return rows where last_activity >= this timestamp.

        Returns:
            List of dicts — one per matching execution row. Empty list
            if nothing matches.

        Example:
            >>> # All incomplete executions:
            >>> incomplete = [ExecutionStatus.created, ExecutionStatus.running,
            ...               ExecutionStatus.stopped, ExecutionStatus.failed,
            ...               ExecutionStatus.pending_upload]
            >>> rows = store.list_executions(status=incomplete)
        """
        from sqlalchemy import select

        stmt = select(self.executions)

        if status is not None:
            if isinstance(status, ExecutionStatus):
                statuses = [str(status)]
            else:
                statuses = [str(s) for s in status]
            stmt = stmt.where(self.executions.c.status.in_(statuses))
        if workflow_rid is not None:
            stmt = stmt.where(self.executions.c.workflow_rid == workflow_rid)
        if mode is not None:
            stmt = stmt.where(self.executions.c.mode == str(mode))
        if since is not None:
            stmt = stmt.where(self.executions.c.last_activity >= since)

        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]
```

Also ensure the imports at the top of the file include the datetime and
ConnectionMode references — add if missing:

```python
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deriva_ml.core.connection_mode import ConnectionMode
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_store.py -v
```

Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/state_store.py tests/execution/test_state_store.py
git commit -m "feat(state_store): basic CRUD on executions table

insert_execution, get_execution, update_execution (partial), and
list_executions with status/workflow_rid/mode/since filters. Low-
level helpers; richer API (resume_execution, find_incomplete, etc.)
lives on DerivaML in Group D.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task B5: CRUD helpers for `pending_rows` and `directory_rules`

**Files:**
- Modify: `src/deriva_ml/execution/state_store.py`
- Test: `tests/execution/test_state_store.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/execution/test_state_store.py`:

```python
def test_insert_pending_row(tmp_path):
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus, PendingRowStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)

    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    pending_id = store.insert_pending_row(
        execution_rid="EXE-A",
        key="k1",
        target_schema="deriva-ml",
        target_table="Subject",
        metadata_json='{"Name": "x"}',
        created_at=now,
    )
    assert isinstance(pending_id, int)
    assert pending_id > 0

    rows = store.list_pending_rows(execution_rid="EXE-A")
    assert len(rows) == 1
    assert rows[0]["status"] == str(PendingRowStatus.staged)
    assert rows[0]["target_table"] == "Subject"


def test_list_pending_rows_filter_by_status(tmp_path):
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus, PendingRowStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    id1 = store.insert_pending_row(
        execution_rid="EXE-A", key="k1",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )
    id2 = store.insert_pending_row(
        execution_rid="EXE-A", key="k2",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )

    store.update_pending_row(id1, status=PendingRowStatus.uploaded)
    staged = store.list_pending_rows(
        execution_rid="EXE-A", status=PendingRowStatus.staged,
    )
    assert {r["id"] for r in staged} == {id2}


def test_insert_directory_rule(tmp_path):
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus, DirectoryRuleStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    rule_id = store.insert_directory_rule(
        execution_rid="EXE-A",
        target_schema="deriva-ml",
        target_table="Mask",
        source_dir="/tmp/masks",
        glob="*.png",
        recurse=False,
        copy_files=False,
        asset_types_json=None,
        created_at=now,
    )
    assert isinstance(rule_id, int)

    rules = store.list_directory_rules(execution_rid="EXE-A")
    assert len(rules) == 1
    assert rules[0]["status"] == str(DirectoryRuleStatus.active)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_store.py -v -k "pending or directory"
```

Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Add CRUD methods**

Append to `ExecutionStateStore` in `src/deriva_ml/execution/state_store.py`:

```python
    # ─── pending_rows CRUD ──────────────────────────────────────────

    def insert_pending_row(
        self,
        *,
        execution_rid: str,
        key: str,
        target_schema: str,
        target_table: str,
        metadata_json: str,
        created_at: datetime,
        rid: str | None = None,
        lease_token: str | None = None,
        asset_file_path: str | None = None,
        asset_types_json: str | None = None,
        description: str | None = None,
        status: PendingRowStatus = PendingRowStatus.staged,
        rule_id: int | None = None,
    ) -> int:
        """Insert one pending_rows entry.

        Args:
            execution_rid: FK to executions.rid.
            key: Stable identifier for dedup (auto-hash for ad-hoc
                rows; rule_id+filename for directory-sourced rows).
            target_schema / target_table: Catalog target.
            metadata_json: Serialized column values.
            created_at: UTC timestamp.
            rid: Leased RID, None until leased.
            lease_token: Token for two-phase lease reconciliation.
            asset_file_path: Local file path, None for plain rows.
            asset_types_json: Serialized asset-type terms.
            description: Optional human-readable description.
            status: Initial status, defaults to 'staged'.
            rule_id: FK to directory_rules.id, None if not from a rule.

        Returns:
            The auto-assigned integer id of the new pending_rows row.
        """
        from sqlalchemy import insert

        with self.engine.begin() as conn:
            result = conn.execute(
                insert(self.pending_rows).values(
                    execution_rid=execution_rid, key=key,
                    target_schema=target_schema, target_table=target_table,
                    rid=rid, lease_token=lease_token,
                    metadata_json=metadata_json,
                    asset_file_path=asset_file_path,
                    asset_types_json=asset_types_json,
                    description=description,
                    status=str(status),
                    created_at=created_at,
                    rule_id=rule_id,
                )
            )
            # SQLite returns the auto-increment id via lastrowid.
            return int(result.inserted_primary_key[0])

    def update_pending_row(self, pending_id: int, **fields: object) -> None:
        """Partial update of a pending_rows entry.

        Status / token / rid / timestamps are the common callers. Enum
        values are coerced to strings.
        """
        valid_cols = {c.name for c in self.pending_rows.columns}
        unknown = set(fields) - valid_cols
        if unknown:
            raise KeyError(f"unknown columns on pending_rows: {unknown}")

        coerced = {
            k: str(v) if isinstance(v, PendingRowStatus) else v
            for k, v in fields.items()
        }

        from sqlalchemy import update

        with self.engine.begin() as conn:
            conn.execute(
                update(self.pending_rows)
                .where(self.pending_rows.c.id == pending_id)
                .values(**coerced)
            )

    def list_pending_rows(
        self,
        *,
        execution_rid: str,
        status: "PendingRowStatus | list[PendingRowStatus] | None" = None,
        target_table: str | None = None,
    ) -> list[dict]:
        """Return pending_rows entries scoped to one execution.

        Args:
            execution_rid: Required — pending rows are always scoped
                to a specific execution.
            status: Filter to a status or list of statuses.
            target_table: Filter to a single target table.

        Returns:
            List of dicts — empty if nothing matches.
        """
        from sqlalchemy import select

        stmt = select(self.pending_rows).where(
            self.pending_rows.c.execution_rid == execution_rid
        )
        if status is not None:
            if isinstance(status, PendingRowStatus):
                statuses = [str(status)]
            else:
                statuses = [str(s) for s in status]
            stmt = stmt.where(self.pending_rows.c.status.in_(statuses))
        if target_table is not None:
            stmt = stmt.where(self.pending_rows.c.target_table == target_table)

        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]

    # ─── directory_rules CRUD ───────────────────────────────────────

    def insert_directory_rule(
        self,
        *,
        execution_rid: str,
        target_schema: str,
        target_table: str,
        source_dir: str,
        glob: str,
        recurse: bool,
        copy_files: bool,
        asset_types_json: str | None,
        created_at: datetime,
        status: DirectoryRuleStatus = DirectoryRuleStatus.active,
    ) -> int:
        """Insert one directory_rules entry; return its auto id."""
        from sqlalchemy import insert

        with self.engine.begin() as conn:
            result = conn.execute(
                insert(self.directory_rules).values(
                    execution_rid=execution_rid,
                    target_schema=target_schema, target_table=target_table,
                    source_dir=source_dir,
                    glob=glob, recurse=recurse, copy_files=copy_files,
                    asset_types_json=asset_types_json,
                    status=str(status),
                    created_at=created_at,
                )
            )
            return int(result.inserted_primary_key[0])

    def update_directory_rule(self, rule_id: int, **fields: object) -> None:
        """Partial update of a directory_rules entry."""
        valid_cols = {c.name for c in self.directory_rules.columns}
        unknown = set(fields) - valid_cols
        if unknown:
            raise KeyError(f"unknown columns on directory_rules: {unknown}")
        coerced = {
            k: str(v) if isinstance(v, DirectoryRuleStatus) else v
            for k, v in fields.items()
        }
        from sqlalchemy import update

        with self.engine.begin() as conn:
            conn.execute(
                update(self.directory_rules)
                .where(self.directory_rules.c.id == rule_id)
                .values(**coerced)
            )

    def list_directory_rules(
        self,
        *,
        execution_rid: str,
        status: "DirectoryRuleStatus | None" = None,
    ) -> list[dict]:
        """List directory_rules for one execution, optionally filtered."""
        from sqlalchemy import select

        stmt = select(self.directory_rules).where(
            self.directory_rules.c.execution_rid == execution_rid
        )
        if status is not None:
            stmt = stmt.where(self.directory_rules.c.status == str(status))

        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_store.py -v
```

Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/state_store.py tests/execution/test_state_store.py
git commit -m "feat(state_store): pending_rows + directory_rules CRUD

Insert / update / list helpers matching the executions table's
pattern. Enum values coerced to strings at the boundary. These are
the primitive operations state machine (Group C) and upload engine
(Group G) will compose on top of.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

*(End of Task Group B — SQLite schema + CRUD primitives.)*

---

## Task Group C — Execution state machine

Implements the state-transition module (`state_machine.py`) that owns
execution lifecycle status. Every transition is an atomic SQLite
write; in online mode, the catalog `Execution` row is synced in the
same path with soft-fail handling (sets `sync_pending=True` on
catalog-PUT failure). Just-in-time reconciliation runs on
`resume_execution` (called in Group D).

### Task C1: Allowed-transition table + validator

**Files:**
- Create: `src/deriva_ml/execution/state_machine.py`
- Test: `tests/execution/test_state_machine.py`

- [ ] **Step 1: Write the failing test**

Create `tests/execution/test_state_machine.py`:

```python
"""Tests for the execution state machine."""

from __future__ import annotations

import pytest

from deriva_ml.execution.state_machine import (
    ALLOWED_TRANSITIONS,
    InvalidTransitionError,
    validate_transition,
)
from deriva_ml.execution.state_store import ExecutionStatus


def test_allowed_transitions_cover_all_happy_paths():
    # created → running → stopped → pending_upload → uploaded
    assert (ExecutionStatus.created, ExecutionStatus.running) in ALLOWED_TRANSITIONS
    assert (ExecutionStatus.running, ExecutionStatus.stopped) in ALLOWED_TRANSITIONS
    assert (ExecutionStatus.stopped, ExecutionStatus.pending_upload) in ALLOWED_TRANSITIONS
    assert (ExecutionStatus.pending_upload, ExecutionStatus.uploaded) in ALLOWED_TRANSITIONS


def test_allowed_transitions_cover_failure_paths():
    assert (ExecutionStatus.running, ExecutionStatus.failed) in ALLOWED_TRANSITIONS
    assert (ExecutionStatus.pending_upload, ExecutionStatus.failed) in ALLOWED_TRANSITIONS


def test_allowed_transitions_cover_abort():
    # Abort legal from created, running, stopped, failed
    for start in [ExecutionStatus.created, ExecutionStatus.running,
                  ExecutionStatus.stopped, ExecutionStatus.failed]:
        assert (start, ExecutionStatus.aborted) in ALLOWED_TRANSITIONS


def test_retry_from_failed_back_to_pending_upload():
    # retry_failed → pending_upload is legal (upload retry path)
    assert (ExecutionStatus.failed, ExecutionStatus.pending_upload) in ALLOWED_TRANSITIONS


def test_validate_transition_accepts_allowed():
    validate_transition(
        current=ExecutionStatus.running,
        target=ExecutionStatus.stopped,
    )  # must not raise


def test_validate_transition_rejects_disallowed():
    with pytest.raises(InvalidTransitionError) as exc:
        validate_transition(
            current=ExecutionStatus.uploaded,
            target=ExecutionStatus.running,  # can't go back to running
        )
    msg = str(exc.value)
    assert "uploaded" in msg
    assert "running" in msg


def test_invalid_transition_error_is_deriva_ml_exception():
    from deriva_ml.core.exceptions import DerivaMLException
    assert issubclass(InvalidTransitionError, DerivaMLException)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_machine.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the state machine primitives**

Create `src/deriva_ml/execution/state_machine.py`:

```python
"""Execution-lifecycle state machine.

Per spec §2.2. All transitions go through this module; direct updates
to executions.status from elsewhere are a bug. The module:

- Defines the allowed (from, to) pairs as a set-based table.
- Validates transitions at call time.
- Owns the SQLite-write + catalog-sync path (with sync_pending
  soft-fail on catalog failure).
- Provides the disagreement-resolution logic used by just-in-time
  reconciliation in resume_execution.

Why a module and not a class: the state machine is functional —
transitions take (store, catalog, rid, target, metadata). The
ExecutionStateStore and ErmrestCatalog live elsewhere; this module
wires them together without owning lifecycle of either.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from deriva_ml.core.exceptions import (
    DerivaMLDataError,
    DerivaMLException,
    DerivaMLStateInconsistency,
)
from deriva_ml.execution.state_store import ExecutionStatus

if TYPE_CHECKING:
    from deriva.core import ErmrestCatalog

    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_store import ExecutionStateStore

logger = logging.getLogger(__name__)


class InvalidTransitionError(DerivaMLException):
    """Raised when a requested status transition is not in the
    allowed table.

    This is a programming error, not a runtime-data error — allowed
    transitions are a compile-time decision and something outside the
    state machine tried to bypass the rules.
    """


# Allowed transitions. Kept explicit (not derived) so the table is
# the single source of truth and easy to read.
#
# The state diagram (spec §2.2):
#
#     created → running → {stopped, failed} → pending_upload → {uploaded, failed}
#                                                      ↑             │
#                                                      └──── retry ──┘
#     created / running / stopped / failed → aborted (terminal)
#     failed → pending_upload (retry_failed path)

ALLOWED_TRANSITIONS: frozenset[tuple[ExecutionStatus, ExecutionStatus]] = frozenset({
    # Happy path
    (ExecutionStatus.created, ExecutionStatus.running),
    (ExecutionStatus.running, ExecutionStatus.stopped),
    (ExecutionStatus.stopped, ExecutionStatus.pending_upload),
    (ExecutionStatus.pending_upload, ExecutionStatus.uploaded),

    # Failure paths
    (ExecutionStatus.running, ExecutionStatus.failed),
    (ExecutionStatus.pending_upload, ExecutionStatus.failed),

    # Retry from upload failure back into upload
    (ExecutionStatus.failed, ExecutionStatus.pending_upload),

    # Abort is legal from any pre-terminal state. 'uploaded' is
    # terminal — we don't allow abort after successful upload.
    (ExecutionStatus.created, ExecutionStatus.aborted),
    (ExecutionStatus.running, ExecutionStatus.aborted),
    (ExecutionStatus.stopped, ExecutionStatus.aborted),
    (ExecutionStatus.failed, ExecutionStatus.aborted),
})


def validate_transition(
    *,
    current: ExecutionStatus,
    target: ExecutionStatus,
) -> None:
    """Verify that (current → target) is in the allowed table.

    Args:
        current: The execution's current status (as read from SQLite).
        target: The requested new status.

    Raises:
        InvalidTransitionError: If the pair is not in
            ALLOWED_TRANSITIONS. Message includes both states.

    Example:
        >>> validate_transition(
        ...     current=ExecutionStatus.running,
        ...     target=ExecutionStatus.stopped,
        ... )  # returns None, no raise
    """
    if (current, target) not in ALLOWED_TRANSITIONS:
        raise InvalidTransitionError(
            f"Illegal execution transition {current} → {target}. "
            f"See spec §2.2 for the allowed transition graph."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_machine.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/state_machine.py tests/execution/test_state_machine.py
git commit -m "feat(state_machine): allowed-transition table + validator

Per spec §2.2. Explicit (from, to) pairs, not derived. InvalidTransitionError
raised when outside code tries to bypass the rules.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C2: `transition()` — SQLite-only path (catalog sync deferred to C3)

**Files:**
- Modify: `src/deriva_ml/execution/state_machine.py`
- Test: `tests/execution/test_state_machine.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/execution/test_state_machine.py`:

```python
def test_transition_writes_sqlite(tmp_path):
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_machine import transition
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.created,
        mode=ConnectionMode.offline, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    # Offline mode: no catalog argument, no sync attempt.
    transition(
        store=store,
        catalog=None,                # offline → skip catalog sync
        execution_rid="EXE-A",
        current=ExecutionStatus.created,
        target=ExecutionStatus.running,
        mode=ConnectionMode.offline,
        extra_fields={"start_time": now},
    )

    row = store.get_execution("EXE-A")
    assert row["status"] == "running"
    assert row["start_time"] is not None
    # Offline transitions always set sync_pending=True.
    assert row["sync_pending"] is True


def test_transition_rejects_invalid(tmp_path):
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_machine import (
        InvalidTransitionError, transition,
    )
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.uploaded,
        mode=ConnectionMode.offline, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    with pytest.raises(InvalidTransitionError):
        transition(
            store=store, catalog=None, execution_rid="EXE-A",
            current=ExecutionStatus.uploaded,
            target=ExecutionStatus.running,
            mode=ConnectionMode.offline,
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_machine.py -v -k transition
```

Expected: FAIL with `ImportError: cannot import name 'transition'`.

- [ ] **Step 3: Add `transition()` (offline-only for now)**

Append to `src/deriva_ml/execution/state_machine.py`:

```python
def transition(
    *,
    store: "ExecutionStateStore",
    catalog: "ErmrestCatalog | None",
    execution_rid: str,
    current: ExecutionStatus,
    target: ExecutionStatus,
    mode: "ConnectionMode",
    extra_fields: dict | None = None,
) -> None:
    """Transition an execution's status, writing SQLite and syncing
    the catalog when online.

    This is the single entry point for all lifecycle status changes.
    Direct writes to executions.status bypass validation and catalog
    sync; don't do it.

    Args:
        store: The ExecutionStateStore owning the SQLite row.
        catalog: The ErmrestCatalog for syncing. Pass None in offline
            mode (attempting to pass a non-None catalog in offline
            mode is a programming error and raises).
        execution_rid: Which execution to transition.
        current: The status we believe the execution is in. The state
            machine does NOT re-read SQLite to determine `current`;
            the caller passed it, typically from a just-prior read.
            This lets the caller do its own consistency check if
            needed.
        target: The status to transition to.
        mode: ConnectionMode. Online → also PUT catalog row; offline
            → only update SQLite, set sync_pending=True.
        extra_fields: Additional executions columns to update in the
            same transaction (start_time, stop_time, error, etc.).

    Raises:
        InvalidTransitionError: If (current, target) is not in
            ALLOWED_TRANSITIONS.
        ValueError: If mode=offline but catalog is not None, or
            mode=online but catalog is None. These are caller bugs.

    Example:
        >>> transition(
        ...     store=store, catalog=ml.catalog,
        ...     execution_rid="EXE-A",
        ...     current=ExecutionStatus.running,
        ...     target=ExecutionStatus.stopped,
        ...     mode=ConnectionMode.online,
        ...     extra_fields={"stop_time": datetime.now(timezone.utc)},
        ... )
    """
    from deriva_ml.core.connection_mode import ConnectionMode

    validate_transition(current=current, target=target)

    # Consistency: offline must pass catalog=None, online must pass
    # a real catalog. Mismatches indicate a caller bug.
    if mode is ConnectionMode.offline and catalog is not None:
        raise ValueError("offline mode must pass catalog=None")
    if mode is ConnectionMode.online and catalog is None:
        raise ValueError("online mode requires a catalog")

    now = datetime.now(timezone.utc)
    extra_fields = dict(extra_fields or {})
    extra_fields.setdefault("last_activity", now)

    if mode is ConnectionMode.offline:
        # Offline: only SQLite. Set sync_pending so that the next
        # online opportunity will push this status to the catalog.
        store.update_execution(
            execution_rid,
            status=target,
            sync_pending=True,
            **extra_fields,
        )
        logger.debug(
            "offline transition %s: %s → %s (sync_pending)",
            execution_rid, current, target,
        )
        return

    # Online path deferred to Task C3.
    raise NotImplementedError("online transition lands in Task C3")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_machine.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/state_machine.py tests/execution/test_state_machine.py
git commit -m "feat(state_machine): transition() with offline-only path

Validates the transition via ALLOWED_TRANSITIONS, writes SQLite with
sync_pending=True. Online path is a NotImplementedError placeholder
that Task C3 fills in. Consistency checks reject mode/catalog
mismatches as caller bugs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C3: Online-mode catalog sync in `transition()` (soft-fail on PUT)

**Files:**
- Modify: `src/deriva_ml/execution/state_machine.py`
- Test: `tests/execution/test_state_machine.py` (extend)

- [ ] **Step 1: Write the failing test (mocked catalog)**

Append to `tests/execution/test_state_machine.py`:

```python
class _MockCatalog:
    """Minimal mock for ErmrestCatalog exposing just what transition() uses."""
    def __init__(self, *, put_should_fail: bool = False):
        self.put_should_fail = put_should_fail
        self.put_calls: list[dict] = []

    def put(self, path: str, json: object = None, **_kw):
        if self.put_should_fail:
            raise RuntimeError("simulated network failure")
        self.put_calls.append({"path": path, "json": json})
        return None


def test_online_transition_syncs_catalog(tmp_path):
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_machine import transition
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.created,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    cat = _MockCatalog()
    transition(
        store=store, catalog=cat, execution_rid="EXE-A",
        current=ExecutionStatus.created, target=ExecutionStatus.running,
        mode=ConnectionMode.online, extra_fields={"start_time": now},
    )

    row = store.get_execution("EXE-A")
    assert row["status"] == "running"
    # Online: sync succeeded, no pending flag.
    assert row["sync_pending"] is False
    # Catalog was put-updated.
    assert len(cat.put_calls) == 1
    body = cat.put_calls[0]["json"]
    assert isinstance(body, list) and len(body) == 1
    assert body[0]["RID"] == "EXE-A"
    assert body[0]["Status"] == "running"


def test_online_transition_soft_fails_on_catalog_error(tmp_path):
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_machine import transition
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.created,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    cat = _MockCatalog(put_should_fail=True)
    # SQLite transition must still succeed; the catalog failure is
    # soft — user gets sync_pending=True for the next pass to flush.
    transition(
        store=store, catalog=cat, execution_rid="EXE-A",
        current=ExecutionStatus.created, target=ExecutionStatus.running,
        mode=ConnectionMode.online,
    )

    row = store.get_execution("EXE-A")
    assert row["status"] == "running"
    assert row["sync_pending"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_machine.py -v -k online
```

Expected: FAIL with `NotImplementedError` from the placeholder.

- [ ] **Step 3: Implement the online path**

In `src/deriva_ml/execution/state_machine.py`, replace the
`NotImplementedError` line in `transition()` with the online-sync
block:

```python
    # Online: SQLite first, then catalog PUT. If PUT fails, leave
    # sync_pending=True so a later call (or resume_execution)
    # flushes. We never let catalog failure roll back SQLite — the
    # local view is the source of truth; the catalog catches up.
    #
    # Ordering note: we commit SQLite BEFORE the catalog PUT. If we
    # crashed between the commit and the PUT, sync_pending would stay
    # True (we set it preemptively) and the next online operation
    # would push. The reverse ordering (catalog first) creates an
    # unrecoverable window where the catalog has moved but SQLite
    # hasn't — a later crash would lose the catalog transition.
    store.update_execution(
        execution_rid,
        status=target,
        sync_pending=True,  # preemptively True; cleared after successful PUT
        **extra_fields,
    )

    # Compose the catalog PUT body from the SQLite row we just wrote.
    # Only the columns the catalog Execution row knows about go here —
    # Status and lifecycle timestamps — not SQLite-only fields like
    # sync_pending or config_json.
    body = _catalog_body_for_execution(
        store=store,
        execution_rid=execution_rid,
    )
    try:
        catalog.put(
            f"/entity/deriva-ml:Execution",
            json=body,
        )
    except Exception as exc:  # network blip, 5xx, etc.
        logger.warning(
            "execution %s: catalog sync FAILED (%s); SQLite committed, "
            "sync_pending stays True for later flush",
            execution_rid, exc,
        )
        return

    # PUT succeeded — clear sync_pending.
    store.update_execution(execution_rid, sync_pending=False)
    logger.debug(
        "online transition %s: %s → %s (synced)",
        execution_rid, current, target,
    )


def _catalog_body_for_execution(
    *,
    store: "ExecutionStateStore",
    execution_rid: str,
) -> list[dict]:
    """Build the ERMrest PUT body for an execution's catalog row.

    Reads the current SQLite state and projects to the catalog's
    column set. Kept as a helper so transition() stays focused on
    orchestration and so tests can assert on body contents.
    """
    row = store.get_execution(execution_rid)
    if row is None:
        # Caller just updated SQLite; this would only happen on a
        # concurrent delete. Surface clearly rather than putting a
        # partial body to the catalog.
        raise DerivaMLStateInconsistency(
            f"executions row {execution_rid} vanished between write and PUT"
        )
    # Catalog Execution schema: RID (PK), Status, Start_Time, End_Time,
    # Duration, ... — see src/deriva_ml/schema/create_schema.py for
    # the canonical column list. We update only the columns we own
    # here; the catalog is responsible for RCB/RMT etc.
    return [{
        "RID": row["rid"],
        "Status": row["status"],
        "Start_Time": row["start_time"],
        "End_Time": row["stop_time"],
        # Status_Detail: prefer error if present, else description.
        "Status_Detail": row["error"] or row["description"],
    }]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_machine.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/state_machine.py tests/execution/test_state_machine.py
git commit -m "feat(state_machine): online catalog sync with soft-fail

SQLite commits first, then catalog PUT; on PUT failure sync_pending
stays True for a later flush. _catalog_body_for_execution projects
the SQLite row to catalog's Execution column set.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C4: `flush_pending_sync()` — push queued offline transitions

**Files:**
- Modify: `src/deriva_ml/execution/state_machine.py`
- Test: `tests/execution/test_state_machine.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/execution/test_state_machine.py`:

```python
def test_flush_pending_sync_pushes_catalog(tmp_path):
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_machine import flush_pending_sync, transition
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.created,
        mode=ConnectionMode.offline, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )
    # Do an offline transition: SQLite has sync_pending=True.
    transition(
        store=store, catalog=None, execution_rid="EXE-A",
        current=ExecutionStatus.created, target=ExecutionStatus.running,
        mode=ConnectionMode.offline,
    )
    assert store.get_execution("EXE-A")["sync_pending"] is True

    # Now flush it against a live (mock) catalog.
    cat = _MockCatalog()
    flush_pending_sync(store=store, catalog=cat, execution_rid="EXE-A")

    assert store.get_execution("EXE-A")["sync_pending"] is False
    assert len(cat.put_calls) == 1


def test_flush_pending_sync_noop_when_not_pending(tmp_path):
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_machine import flush_pending_sync
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.stopped,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
        sync_pending=False,
    )

    cat = _MockCatalog()
    flush_pending_sync(store=store, catalog=cat, execution_rid="EXE-A")
    assert len(cat.put_calls) == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_machine.py -v -k flush
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add `flush_pending_sync()`**

Append to `src/deriva_ml/execution/state_machine.py`:

```python
def flush_pending_sync(
    *,
    store: "ExecutionStateStore",
    catalog: "ErmrestCatalog",
    execution_rid: str,
) -> None:
    """Push a single execution's SQLite state to the catalog.

    Called when we've opened online and notice this execution has
    sync_pending=True (accumulated from offline transitions, or from
    a previous online transition whose PUT failed).

    Idempotent: no-op if sync_pending is already False. If the PUT
    fails, sync_pending stays True for the next attempt.

    Args:
        store: ExecutionStateStore holding the row.
        catalog: Live ErmrestCatalog.
        execution_rid: Which execution to flush.

    Raises:
        DerivaMLStateInconsistency: If the execution row has vanished.

    Example:
        >>> # After resuming an execution online that last ran offline:
        >>> flush_pending_sync(store=store, catalog=ml.catalog,
        ...                    execution_rid="EXE-A")
    """
    row = store.get_execution(execution_rid)
    if row is None:
        raise DerivaMLStateInconsistency(
            f"flush_pending_sync: execution {execution_rid} not in SQLite"
        )
    if not row["sync_pending"]:
        return

    body = _catalog_body_for_execution(store=store, execution_rid=execution_rid)
    try:
        catalog.put(f"/entity/deriva-ml:Execution", json=body)
    except Exception as exc:
        logger.warning(
            "flush_pending_sync %s: catalog PUT failed (%s); will retry later",
            execution_rid, exc,
        )
        return

    store.update_execution(execution_rid, sync_pending=False)
    logger.debug("flush_pending_sync %s: synced", execution_rid)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_machine.py -v
```

Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/state_machine.py tests/execution/test_state_machine.py
git commit -m "feat(state_machine): flush_pending_sync()

Idempotent flush of a single execution's sync_pending=True row to the
catalog. Called from resume_execution (Group D) when reopening online.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C5: Disagreement-resolution rules (`reconcile_with_catalog`)

**Files:**
- Modify: `src/deriva_ml/execution/state_machine.py`
- Test: `tests/execution/test_state_machine.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/execution/test_state_machine.py`:

```python
class _MockCatalogWithGet(_MockCatalog):
    """Extends the mock with a configurable GET response."""
    def __init__(self, *, get_row: dict | None | str = None, **kw):
        super().__init__(**kw)
        # get_row: dict = returned row, None = 404/no row, "raise" = raise
        self._get_row = get_row

    def get(self, path: str, **_kw):
        if self._get_row == "raise":
            raise RuntimeError("simulated 500")
        class _R:
            def __init__(self, row):
                self._row = row
            def json(self):
                return [self._row] if self._row is not None else []
            status_code = 200
        return _R(self._get_row)


def test_reconcile_no_disagreement(tmp_path):
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.execution.state_machine import reconcile_with_catalog
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.stopped,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now, sync_pending=False,
    )

    cat = _MockCatalogWithGet(get_row={"RID": "EXE-A", "Status": "stopped"})
    reconcile_with_catalog(store=store, catalog=cat, execution_rid="EXE-A")
    # Unchanged.
    assert store.get_execution("EXE-A")["status"] == "stopped"


def test_reconcile_catalog_says_aborted(tmp_path):
    """SQLite=running, catalog=aborted → SQLite flips to aborted."""
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.execution.state_machine import reconcile_with_catalog
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now, sync_pending=False,
    )

    cat = _MockCatalogWithGet(get_row={"RID": "EXE-A", "Status": "aborted"})
    reconcile_with_catalog(store=store, catalog=cat, execution_rid="EXE-A")
    assert store.get_execution("EXE-A")["status"] == "aborted"


def test_reconcile_catalog_says_uploaded(tmp_path):
    """SQLite=pending_upload, catalog=uploaded → SQLite flips to uploaded."""
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.execution.state_machine import reconcile_with_catalog
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.pending_upload,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now, sync_pending=False,
    )

    cat = _MockCatalogWithGet(get_row={"RID": "EXE-A", "Status": "uploaded"})
    reconcile_with_catalog(store=store, catalog=cat, execution_rid="EXE-A")
    assert store.get_execution("EXE-A")["status"] == "uploaded"


def test_reconcile_sqlite_stopped_catalog_running(tmp_path):
    """SQLite=stopped, catalog=running (stale) → push SQLite to catalog."""
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.execution.state_machine import reconcile_with_catalog
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.stopped,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now, sync_pending=False,
    )

    cat = _MockCatalogWithGet(get_row={"RID": "EXE-A", "Status": "running"})
    reconcile_with_catalog(store=store, catalog=cat, execution_rid="EXE-A")
    # SQLite unchanged; sync_pending set so next flush pushes.
    assert store.get_execution("EXE-A")["status"] == "stopped"
    assert store.get_execution("EXE-A")["sync_pending"] is True


def test_reconcile_catalog_missing_raises(tmp_path):
    """SQLite has the row; catalog doesn't → orphan, raise."""
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.core.exceptions import DerivaMLStateInconsistency
    from deriva_ml.execution.state_machine import reconcile_with_catalog
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.stopped,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    cat = _MockCatalogWithGet(get_row=None)
    with pytest.raises(DerivaMLStateInconsistency):
        reconcile_with_catalog(store=store, catalog=cat, execution_rid="EXE-A")


def test_reconcile_catalog_error_logs_and_returns(tmp_path, caplog):
    """Transient catalog error → reconcile skips cleanly (caller decides)."""
    from datetime import datetime, timezone
    from sqlalchemy import create_engine

    from deriva_ml.execution.state_machine import reconcile_with_catalog
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.stopped,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now, sync_pending=False,
    )

    cat = _MockCatalogWithGet(get_row="raise")
    import logging
    with caplog.at_level(logging.WARNING):
        reconcile_with_catalog(store=store, catalog=cat, execution_rid="EXE-A")
    # SQLite unchanged.
    assert store.get_execution("EXE-A")["status"] == "stopped"
    assert any("reconcile" in r.message.lower() for r in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_machine.py -v -k reconcile
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `reconcile_with_catalog()`**

Append to `src/deriva_ml/execution/state_machine.py`:

```python
# Disagreement resolution table (spec §2.2 — six cases).
#
# Rows are keyed by (sqlite_status, catalog_status) tuples. Value is
# a literal action name:
#   'noop'         — states agree, do nothing
#   'adopt'        — SQLite adopts the catalog's status
#   'push'         — SQLite state is newer; set sync_pending=True
#   'raise'        — unexpected; surface to the user for intervention
#
# Sync-pending handling is layered on top: if sqlite.sync_pending was
# True we generally 'push' regardless of catalog state.

_DISAGREEMENT_RULES: dict[tuple[ExecutionStatus, ExecutionStatus], str] = {
    # Externally aborted while we thought we were running.
    (ExecutionStatus.running, ExecutionStatus.aborted): "adopt",
    # Another process completed the upload.
    (ExecutionStatus.pending_upload, ExecutionStatus.uploaded): "adopt",
    # External failure signal.
    (ExecutionStatus.running, ExecutionStatus.failed): "adopt",
    # We stopped cleanly; catalog still says running (our earlier PUT
    # never landed).
    (ExecutionStatus.stopped, ExecutionStatus.running): "push",
    # Same story at other cleanly-terminal SQLite states.
    (ExecutionStatus.failed, ExecutionStatus.running): "push",
    (ExecutionStatus.uploaded, ExecutionStatus.pending_upload): "push",
    (ExecutionStatus.uploaded, ExecutionStatus.running): "push",
    (ExecutionStatus.aborted, ExecutionStatus.running): "push",
}


def reconcile_with_catalog(
    *,
    store: "ExecutionStateStore",
    catalog: "ErmrestCatalog",
    execution_rid: str,
) -> None:
    """Compare a single execution's SQLite state with the catalog and
    apply the disagreement rules (spec §2.2).

    Called on resume_execution when online, before returning the
    Execution to the user. Keeps startup fast by acting per-execution
    rather than workspace-wide.

    Args:
        store: The ExecutionStateStore.
        catalog: Live ErmrestCatalog.
        execution_rid: Which execution to reconcile.

    Raises:
        DerivaMLStateInconsistency: Catalog row missing (orphan), or
            disagreement is outside the known rule table.

    Example:
        >>> # On resume_execution in online mode:
        >>> reconcile_with_catalog(
        ...     store=ws.execution_state_store(),
        ...     catalog=ml.catalog,
        ...     execution_rid="EXE-A",
        ... )
    """
    sqlite_row = store.get_execution(execution_rid)
    if sqlite_row is None:
        raise DerivaMLStateInconsistency(
            f"reconcile: execution {execution_rid} not in SQLite"
        )
    sqlite_status = ExecutionStatus(sqlite_row["status"])

    try:
        # URL filter on RID — returns a list of 0 or 1 rows.
        response = catalog.get(
            f"/entity/deriva-ml:Execution/RID={execution_rid}"
        )
        rows = response.json()
    except Exception as exc:
        logger.warning(
            "reconcile %s: catalog GET failed (%s); leaving SQLite as-is",
            execution_rid, exc,
        )
        return

    if not rows:
        # Orphan: SQLite has the row, catalog doesn't. This is
        # usually a clone/copy gone wrong or a catalog-side delete.
        # Don't guess; ask the user to resolve.
        raise DerivaMLStateInconsistency(
            f"Execution {execution_rid} exists in SQLite (status={sqlite_status}) "
            f"but has no row in the catalog. Either the catalog was "
            f"re-initialized, or the workspace was copied from elsewhere. "
            f"To adopt SQLite state, manually insert the catalog row; "
            f"to discard, call ml.gc_executions(status='aborted')."
        )

    catalog_row = rows[0]
    # Catalog Status is a vocab term; its string value matches our enum.
    try:
        catalog_status = ExecutionStatus(catalog_row.get("Status", ""))
    except ValueError:
        # Catalog has a Status we don't recognize. Surface rather than guess.
        raise DerivaMLStateInconsistency(
            f"Execution {execution_rid}: catalog Status="
            f"{catalog_row.get('Status')!r} is not a recognized "
            f"ExecutionStatus value"
        )

    # Happy path: they agree.
    if sqlite_status == catalog_status:
        return

    # SQLite was waiting to push — this disagreement is expected.
    if sqlite_row["sync_pending"]:
        # We'll flush later; don't treat the catalog as authoritative.
        logger.debug(
            "reconcile %s: disagreement (SQLite=%s, catalog=%s) is "
            "expected because sync_pending=True; leaving for flush",
            execution_rid, sqlite_status, catalog_status,
        )
        return

    rule = _DISAGREEMENT_RULES.get((sqlite_status, catalog_status))
    if rule == "adopt":
        # Catalog is authoritative. Include any error/timing from the
        # catalog row so the user's Execution.error reflects reality.
        store.update_execution(
            execution_rid,
            status=catalog_status,
            error=catalog_row.get("Status_Detail"),
            sync_pending=False,
        )
        logger.info(
            "reconcile %s: adopted catalog state %s (was %s in SQLite)",
            execution_rid, catalog_status, sqlite_status,
        )
    elif rule == "push":
        # SQLite is newer; mark for flush. The resume flow will
        # invoke flush_pending_sync after reconcile.
        store.update_execution(execution_rid, sync_pending=True)
        logger.info(
            "reconcile %s: SQLite ahead (SQLite=%s, catalog=%s); "
            "marked sync_pending for flush",
            execution_rid, sqlite_status, catalog_status,
        )
    else:
        raise DerivaMLStateInconsistency(
            f"Execution {execution_rid}: unexpected state disagreement "
            f"(SQLite={sqlite_status}, catalog={catalog_status}) not "
            f"covered by reconciliation rules. Human intervention required."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_machine.py -v
```

Expected: 19 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/state_machine.py tests/execution/test_state_machine.py
git commit -m "feat(state_machine): reconcile_with_catalog disagreement rules

Six-case table per spec §2.2: adopt, push, raise. Respects
sync_pending as a 'leave alone, flush later' hint. Transient catalog
errors log and return (reconcile is best-effort, not a barrier).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C6: `create_catalog_execution()` helper (online-only row insert)

**Files:**
- Modify: `src/deriva_ml/execution/state_machine.py`
- Test: `tests/execution/test_state_machine.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/execution/test_state_machine.py`:

```python
class _MockCatalogWithInsert(_MockCatalog):
    """Mock that records POSTs to Execution and returns a fake RID."""
    def __init__(self, *, assigned_rid: str = "EXE-NEW", **kw):
        super().__init__(**kw)
        self.assigned_rid = assigned_rid
        self.post_calls: list[dict] = []

    def post(self, path: str, json=None, **_kw):
        self.post_calls.append({"path": path, "json": json})
        class _R:
            def __init__(self, rid): self._rid = rid
            def json(self): return [{"RID": self._rid, **(json[0] if json else {})}]
            status_code = 201
        return _R(self.assigned_rid)


def test_create_catalog_execution_posts_and_returns_rid():
    from deriva_ml.execution.state_machine import create_catalog_execution

    cat = _MockCatalogWithInsert(assigned_rid="EXE-NEW")
    rid = create_catalog_execution(
        catalog=cat,
        workflow_rid="WFL-1",
        description="a test run",
    )
    assert rid == "EXE-NEW"
    assert len(cat.post_calls) == 1
    body = cat.post_calls[0]["json"]
    assert body[0]["Workflow"] == "WFL-1"
    assert body[0]["Description"] == "a test run"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_machine.py -v -k create_catalog_execution
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `create_catalog_execution()`**

Append to `src/deriva_ml/execution/state_machine.py`:

```python
def create_catalog_execution(
    *,
    catalog: "ErmrestCatalog",
    workflow_rid: str | None,
    description: str | None,
) -> str:
    """POST a new row to the catalog's Execution table and return
    its server-assigned RID.

    This is the one place in the state machine that actually creates a
    new execution — all other transitions modify an existing row. It
    is callable only in online mode (the caller enforces).

    Args:
        catalog: Live ErmrestCatalog.
        workflow_rid: Workflow FK. May be None only if the catalog's
            Execution.Workflow column is nullable (Deriva-ML's
            schema requires it, but other catalogs may differ).
        description: Human-readable description. Passes through to the
            Execution.Description column.

    Returns:
        The RID assigned by the server.

    Raises:
        Exception: On HTTP failure (caller may want to retry).

    Example:
        >>> rid = create_catalog_execution(
        ...     catalog=ml.catalog,
        ...     workflow_rid="WFL-1",
        ...     description="first training run",
        ... )
        >>> rid
        'EXE-NEW'
    """
    body = [{
        "Workflow": workflow_rid,
        "Description": description,
        "Status": str(ExecutionStatus.created),
    }]
    response = catalog.post(f"/entity/deriva-ml:Execution", json=body)
    inserted = response.json()
    if not inserted or "RID" not in inserted[0]:
        raise DerivaMLDataError(
            "catalog POST to Execution returned no RID; unable to continue"
        )
    return inserted[0]["RID"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_machine.py -v
```

Expected: 20 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/state_machine.py tests/execution/test_state_machine.py
git commit -m "feat(state_machine): create_catalog_execution() helper

Online-only: POSTs to catalog Execution table, returns assigned RID.
Sole entry point for new execution creation; all other transitions
modify existing rows.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C7: Module-level re-exports + `__all__`

**Files:**
- Modify: `src/deriva_ml/execution/state_machine.py`
- Test: `tests/execution/test_state_machine.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/execution/test_state_machine.py`:

```python
def test_public_api_exported():
    import deriva_ml.execution.state_machine as sm
    expected = {
        "ALLOWED_TRANSITIONS",
        "InvalidTransitionError",
        "validate_transition",
        "transition",
        "flush_pending_sync",
        "reconcile_with_catalog",
        "create_catalog_execution",
    }
    assert expected.issubset(set(sm.__all__))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_machine.py -v -k public_api
```

Expected: FAIL with `AttributeError: module has no attribute '__all__'`.

- [ ] **Step 3: Add `__all__`**

At the top of `src/deriva_ml/execution/state_machine.py` (after imports), add:

```python
__all__ = [
    "ALLOWED_TRANSITIONS",
    "InvalidTransitionError",
    "validate_transition",
    "transition",
    "flush_pending_sync",
    "reconcile_with_catalog",
    "create_catalog_execution",
]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_machine.py -v
```

Expected: 21 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/state_machine.py tests/execution/test_state_machine.py
git commit -m "chore(state_machine): add __all__ for explicit exports

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

*(End of Task Group C — state machine with catalog sync + reconciliation.)*

---

## Task Group D — Execution registry public API

Wires the state machine and state store into `DerivaML` public methods:
`list_executions`, `find_incomplete_executions`, `resume_execution`,
`gc_executions`, plus the new `create_execution` with kwargs form.

The existing `src/deriva_ml/core/mixins/execution.py` has
`create_execution`, `restore_execution`, `lookup_execution` today.
This group replaces `restore_execution` with `resume_execution`
(hard cutover, R5.1), adds the new methods, and threads the state
machine through `create_execution`.

### Task D1: `ExecutionRecord` dataclass (registry row projection)

**Files:**
- Create: `src/deriva_ml/execution/execution_record_v2.py` (new; see Task D8 for merge-back plan)
- Test: `tests/execution/test_execution_record_v2.py`

**Note:** There's an existing `ExecutionRecord` at `src/deriva_ml/execution/execution_record.py`. That class is a catalog-backed wrapper (queries the server). The new dataclass is SQLite-backed and represents the registry row. To avoid a giant rename-and-refactor in one commit, we build the new one alongside in D1, then in Task D8 we merge them (the existing ExecutionRecord gets updated to subclass or replace its internals). Tests in Groups D–H reference the new dataclass; the old class stays untouched until D8.

- [ ] **Step 1: Write the failing test**

Create `tests/execution/test_execution_record_v2.py`:

```python
"""Tests for the SQLite-backed ExecutionRecord dataclass."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


def test_execution_record_has_registry_fields():
    from deriva_ml.execution.execution_record_v2 import ExecutionRecord
    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_store import ExecutionStatus

    now = datetime.now(timezone.utc)
    rec = ExecutionRecord(
        rid="EXE-A",
        workflow_rid="WFL-1",
        description="test",
        status=ExecutionStatus.stopped,
        mode=ConnectionMode.online,
        working_dir_rel="execution/EXE-A",
        start_time=now,
        stop_time=now,
        last_activity=now,
        error=None,
        sync_pending=False,
        created_at=now,
        pending_rows=0,
        failed_rows=0,
        pending_files=0,
        failed_files=0,
    )
    assert rec.rid == "EXE-A"
    assert rec.status is ExecutionStatus.stopped
    assert rec.mode is ConnectionMode.online
    assert rec.pending_rows == 0


def test_execution_record_is_frozen():
    from deriva_ml.execution.execution_record_v2 import ExecutionRecord
    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_store import ExecutionStatus

    now = datetime.now(timezone.utc)
    rec = ExecutionRecord(
        rid="X", workflow_rid=None, description=None,
        status=ExecutionStatus.created, mode=ConnectionMode.online,
        working_dir_rel="execution/X",
        start_time=None, stop_time=None, last_activity=now,
        error=None, sync_pending=False, created_at=now,
        pending_rows=0, failed_rows=0, pending_files=0, failed_files=0,
    )
    with pytest.raises((AttributeError, TypeError)):
        rec.rid = "Y"


def test_from_row_constructs_from_sqlite_dict():
    from deriva_ml.execution.execution_record_v2 import ExecutionRecord
    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_store import ExecutionStatus

    now = datetime.now(timezone.utc)
    row = {
        "rid": "EXE-A",
        "workflow_rid": "WFL-1",
        "description": "test",
        "status": "stopped",
        "mode": "online",
        "working_dir_rel": "execution/EXE-A",
        "start_time": now,
        "stop_time": now,
        "last_activity": now,
        "error": None,
        "sync_pending": False,
        "created_at": now,
        "config_json": "{}",
    }
    rec = ExecutionRecord.from_row(
        row,
        pending_rows=3, failed_rows=0,
        pending_files=1, failed_files=0,
    )
    assert rec.rid == "EXE-A"
    assert rec.status is ExecutionStatus.stopped
    assert rec.mode is ConnectionMode.online
    assert rec.pending_rows == 3
    assert rec.pending_files == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_record_v2.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the dataclass**

Create `src/deriva_ml/execution/execution_record_v2.py`:

```python
"""SQLite-backed ExecutionRecord — a registry row with derived counts.

Per spec §2.9. A frozen dataclass projection of one execution_state__
row plus convenience counts from pending_rows. Returned by
DerivaML.list_executions, ml.find_incomplete_executions, and as the
handle for resume_execution's just-in-time reconciliation input.

This class will eventually replace the catalog-backed ExecutionRecord
in execution_record.py (Task D8 merges). Built alongside to keep this
refactor reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from deriva_ml.core.connection_mode import ConnectionMode
from deriva_ml.execution.state_store import ExecutionStatus

if TYPE_CHECKING:
    from deriva_ml.core.base import DerivaML


@dataclass(frozen=True)
class ExecutionRecord:
    """Frozen snapshot of an execution's registry row plus pending counts.

    A value object — no mutation, no server reads on property access.
    If you need lifecycle fields that change over time (live status,
    etc.), use the Execution object returned by resume_execution.

    Attributes:
        rid: Server-assigned Execution RID.
        workflow_rid: Workflow FK; None if not set.
        description: Free-form description from the configuration.
        status: Current lifecycle status as of this snapshot.
        mode: ConnectionMode the execution was last active under.
        working_dir_rel: Relative path to the execution root.
        start_time / stop_time: Lifecycle timestamps, None if absent.
        last_activity: Last pending-row mutation time.
        error: Last error message if status in (failed,).
        sync_pending: True if SQLite is ahead of the catalog.
        created_at: When the local registry first knew about this row.
        pending_rows: Count of non-asset pending rows not yet uploaded.
        failed_rows: Count of non-asset rows in status='failed'.
        pending_files: Count of asset-file rows not yet uploaded.
        failed_files: Count of asset-file rows in status='failed'.

    Example:
        >>> records = ml.find_incomplete_executions()
        >>> for r in records:
        ...     print(r.rid, r.status, r.pending_rows)
    """

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
    ) -> "ExecutionRecord":
        """Construct from a SQLite executions row + pending counts.

        Args:
            row: Dict returned by ExecutionStateStore.get_execution or
                list_executions. Must contain all the executions
                columns.
            pending_rows / failed_rows: Non-asset row counts. Zero if
                the caller hasn't queried pending_rows.
            pending_files / failed_files: Asset-file row counts.

        Returns:
            A frozen ExecutionRecord instance.
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_record_v2.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/execution_record_v2.py tests/execution/test_execution_record_v2.py
git commit -m "feat(execution): add SQLite-backed ExecutionRecord dataclass

Per spec §2.9. Frozen dataclass projection of one execution_state__
row with pending/failed counts. from_row classmethod converts a
SQLite row dict to an instance. Lives in execution_record_v2.py;
existing catalog-backed ExecutionRecord in execution_record.py stays
until Task D8 merges.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task D2: Pending-counts query helper on `ExecutionStateStore`

**Files:**
- Modify: `src/deriva_ml/execution/state_store.py`
- Test: `tests/execution/test_state_store.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/execution/test_state_store.py`:

```python
def test_count_pending_by_kind(tmp_path):
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus, PendingRowStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )

    # Two plain rows: one staged, one failed.
    id1 = store.insert_pending_row(
        execution_rid="EXE-A", key="k1",
        target_schema="s", target_table="Subject",
        metadata_json="{}", created_at=now,
    )
    id2 = store.insert_pending_row(
        execution_rid="EXE-A", key="k2",
        target_schema="s", target_table="Subject",
        metadata_json="{}", created_at=now,
    )
    store.update_pending_row(id2, status=PendingRowStatus.failed)

    # Two asset rows: one staged, one uploaded.
    id3 = store.insert_pending_row(
        execution_rid="EXE-A", key="f1",
        target_schema="s", target_table="Image",
        metadata_json="{}", created_at=now,
        asset_file_path="/tmp/a.png",
    )
    id4 = store.insert_pending_row(
        execution_rid="EXE-A", key="f2",
        target_schema="s", target_table="Image",
        metadata_json="{}", created_at=now,
        asset_file_path="/tmp/b.png",
    )
    store.update_pending_row(id4, status=PendingRowStatus.uploaded)

    counts = store.count_pending_by_kind(execution_rid="EXE-A")
    # Plain rows: 1 pending (staged/leasing/leased/uploading), 1 failed.
    # Asset rows: 1 pending, 0 failed (the uploaded one doesn't count).
    assert counts == {
        "pending_rows": 1,
        "failed_rows": 1,
        "pending_files": 1,
        "failed_files": 0,
    }
```

- [ ] **Step 2: Run test to verify it fails**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_store.py -v -k count_pending
```

Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement `count_pending_by_kind`**

Append to `ExecutionStateStore` in `src/deriva_ml/execution/state_store.py`:

```python
    def count_pending_by_kind(
        self,
        *,
        execution_rid: str,
    ) -> dict[str, int]:
        """Return per-kind counts of non-terminal pending rows.

        A "pending" row is in one of staged/leasing/leased/uploading
        (not yet terminally uploaded or failed). A "failed" row is
        specifically in status='failed'. Rows in status='uploaded'
        are excluded from both counts.

        "plain" vs "asset" is determined by asset_file_path — non-null
        means it's an asset row.

        Args:
            execution_rid: Scoping. Required — pending rows are
                execution-scoped.

        Returns:
            A dict with keys pending_rows, failed_rows, pending_files,
            failed_files. Missing keys default to 0 (which is what the
            aggregate returns when nothing matches).

        Example:
            >>> store.count_pending_by_kind(execution_rid="EXE-A")
            {'pending_rows': 5, 'failed_rows': 0,
             'pending_files': 12, 'failed_files': 1}
        """
        from sqlalchemy import case, func, select

        pending_statuses = [
            str(PendingRowStatus.staged),
            str(PendingRowStatus.leasing),
            str(PendingRowStatus.leased),
            str(PendingRowStatus.uploading),
        ]
        failed_status = str(PendingRowStatus.failed)

        # A single aggregate query, branched by asset_file_path IS NULL.
        # case() produces 1 or 0 for each row matching the branch; sum
        # gives the count. This is ~4x faster than 4 separate queries
        # for large pending_rows tables.
        is_plain = self.pending_rows.c.asset_file_path.is_(None)
        is_asset = self.pending_rows.c.asset_file_path.isnot(None)
        status_col = self.pending_rows.c.status

        stmt = select(
            func.coalesce(
                func.sum(
                    case((is_plain & status_col.in_(pending_statuses), 1), else_=0)
                ),
                0,
            ).label("pending_rows"),
            func.coalesce(
                func.sum(
                    case((is_plain & (status_col == failed_status), 1), else_=0)
                ),
                0,
            ).label("failed_rows"),
            func.coalesce(
                func.sum(
                    case((is_asset & status_col.in_(pending_statuses), 1), else_=0)
                ),
                0,
            ).label("pending_files"),
            func.coalesce(
                func.sum(
                    case((is_asset & (status_col == failed_status), 1), else_=0)
                ),
                0,
            ).label("failed_files"),
        ).where(self.pending_rows.c.execution_rid == execution_rid)

        with self.engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        return {
            "pending_rows": int(row["pending_rows"]),
            "failed_rows": int(row["failed_rows"]),
            "pending_files": int(row["pending_files"]),
            "failed_files": int(row["failed_files"]),
        }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_store.py -v
```

Expected: 19 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/state_store.py tests/execution/test_state_store.py
git commit -m "feat(state_store): count_pending_by_kind() single-query aggregate

Returns {pending_rows, failed_rows, pending_files, failed_files} in
one aggregate query. asset_file_path IS NULL discriminates plain vs
asset rows. ~4x faster than four separate COUNT queries for large
pending_rows tables.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task D3: `DerivaML.list_executions()`

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py`
- Test: `tests/execution/test_execution_registry.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/execution/test_execution_registry.py`:

```python
"""Tests for DerivaML execution registry API (list, find_incomplete,
resume, gc, create with kwargs)."""

from __future__ import annotations

from datetime import datetime, timezone


def _insert_test_execution(ws, rid, status, mode="online", workflow_rid=None):
    """Helper to insert an executions row without going through
    catalog.  Used only in unit tests."""
    from deriva_ml.core.connection_mode import ConnectionMode
    from deriva_ml.execution.state_store import ExecutionStatus

    store = ws.execution_state_store()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid=rid, workflow_rid=workflow_rid, description=f"test {rid}",
        config_json="{}",
        status=ExecutionStatus(status) if isinstance(status, str) else status,
        mode=ConnectionMode(mode) if isinstance(mode, str) else mode,
        working_dir_rel=f"execution/{rid}",
        created_at=now, last_activity=now,
    )


def test_list_executions_empty(test_ml):
    assert test_ml.list_executions() == []


def test_list_executions_returns_dataclass(test_ml):
    from deriva_ml.execution.execution_record_v2 import ExecutionRecord
    from deriva_ml.execution.state_store import ExecutionStatus

    _insert_test_execution(test_ml.workspace, "EXE-A", ExecutionStatus.stopped)

    rows = test_ml.list_executions()
    assert len(rows) == 1
    assert isinstance(rows[0], ExecutionRecord)
    assert rows[0].rid == "EXE-A"
    assert rows[0].status is ExecutionStatus.stopped


def test_list_executions_status_filter(test_ml):
    from deriva_ml.execution.state_store import ExecutionStatus

    _insert_test_execution(test_ml.workspace, "A", ExecutionStatus.running)
    _insert_test_execution(test_ml.workspace, "B", ExecutionStatus.uploaded)
    _insert_test_execution(test_ml.workspace, "C", ExecutionStatus.failed)

    incomplete = test_ml.list_executions(
        status=[ExecutionStatus.running, ExecutionStatus.failed],
    )
    rids = {r.rid for r in incomplete}
    assert rids == {"A", "C"}


def test_find_incomplete_executions(test_ml):
    from deriva_ml.execution.state_store import ExecutionStatus

    _insert_test_execution(test_ml.workspace, "A", ExecutionStatus.running)
    _insert_test_execution(test_ml.workspace, "B", ExecutionStatus.uploaded)
    _insert_test_execution(test_ml.workspace, "C", ExecutionStatus.stopped)
    _insert_test_execution(test_ml.workspace, "D", ExecutionStatus.aborted)
    _insert_test_execution(test_ml.workspace, "E", ExecutionStatus.pending_upload)

    rows = test_ml.find_incomplete_executions()
    rids = {r.rid for r in rows}
    # Incomplete = anything not terminally uploaded/aborted.
    # That's {created, running, stopped, failed, pending_upload}.
    # C is stopped, E is pending_upload, A is running.
    assert rids == {"A", "C", "E"}


def test_list_executions_carries_pending_counts(test_ml):
    from deriva_ml.execution.state_store import ExecutionStatus

    _insert_test_execution(test_ml.workspace, "EXE-A", ExecutionStatus.stopped)

    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)
    store.insert_pending_row(
        execution_rid="EXE-A", key="k1", target_schema="s",
        target_table="Subject", metadata_json="{}", created_at=now,
    )

    rows = test_ml.list_executions()
    assert len(rows) == 1
    assert rows[0].pending_rows == 1
    assert rows[0].pending_files == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_registry.py -v
```

Expected: FAIL with `AttributeError: 'DerivaML' object has no attribute 'list_executions'`.

- [ ] **Step 3: Implement `list_executions` and `find_incomplete_executions`**

In `src/deriva_ml/core/mixins/execution.py`, add:

```python
# Near the top of the file, with the other imports:
from datetime import datetime
from deriva_ml.core.connection_mode import ConnectionMode
from deriva_ml.execution.execution_record_v2 import ExecutionRecord
from deriva_ml.execution.state_store import ExecutionStatus


# Inside the ExecutionMixin class, add methods. Place them near
# restore_execution / lookup_execution for locality.

def list_executions(
    self,
    *,
    status: "ExecutionStatus | list[ExecutionStatus] | None" = None,
    workflow_rid: "str | None" = None,
    mode: "ConnectionMode | None" = None,
    since: "datetime | None" = None,
) -> list[ExecutionRecord]:
    """Return known-local executions matching the filters.

    Reads from the workspace SQLite registry — no server contact.
    Works in both online and offline mode.

    Args:
        status: Single ExecutionStatus or list to filter; None = all.
        workflow_rid: Match only executions tagged with this Workflow
            RID; None = all.
        mode: ConnectionMode the execution was last active under;
            None = all.
        since: Return only executions with last_activity >= this
            timestamp (timezone-aware). None = no time filter.

    Returns:
        List of ExecutionRecord dataclasses — one per matching row.
        Empty list if nothing matches. Pending-row counts are derived
        in the same pass.

    Example:
        >>> from deriva_ml.execution.state_store import ExecutionStatus
        >>> failed = ml.list_executions(status=ExecutionStatus.failed)
        >>> for rec in failed:
        ...     print(rec.rid, rec.error)
    """
    store = self.workspace.execution_state_store()
    rows = store.list_executions(
        status=status, workflow_rid=workflow_rid,
        mode=mode, since=since,
    )
    return [
        ExecutionRecord.from_row(row, **store.count_pending_by_kind(execution_rid=row["rid"]))
        for row in rows
    ]


def find_incomplete_executions(self) -> list[ExecutionRecord]:
    """Sugar over list_executions for everything not terminally done.

    Returns executions in status in (created, running, stopped,
    failed, pending_upload) — the set of things a user would want to
    either resume, retry, or clean up. Excludes uploaded (terminal
    success) and aborted (terminal cleanup).

    Returns:
        List of ExecutionRecord for incomplete runs.

    Example:
        >>> for rec in ml.find_incomplete_executions():
        ...     print(rec.rid, rec.status, rec.pending_rows)
    """
    return self.list_executions(
        status=[
            ExecutionStatus.created,
            ExecutionStatus.running,
            ExecutionStatus.stopped,
            ExecutionStatus.failed,
            ExecutionStatus.pending_upload,
        ],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_registry.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/core/mixins/execution.py tests/execution/test_execution_registry.py
git commit -m "feat(execution): list_executions + find_incomplete_executions

SQLite-only registry queries — no server contact. Returns the new
ExecutionRecord dataclass with pending/failed counts derived in the
same pass via count_pending_by_kind.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task D4: `DerivaML.resume_execution()` with reconciliation

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py`
- Test: `tests/execution/test_execution_registry.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/execution/test_execution_registry.py`:

```python
def test_resume_execution_reads_from_sqlite(test_ml):
    from deriva_ml.execution.state_store import ExecutionStatus

    _insert_test_execution(test_ml.workspace, "EXE-A", ExecutionStatus.stopped)

    exe = test_ml.resume_execution("EXE-A")
    assert exe.execution_rid == "EXE-A"


def test_resume_execution_missing_raises(test_ml):
    from deriva_ml.core.exceptions import DerivaMLException

    import pytest
    with pytest.raises(DerivaMLException) as exc:
        test_ml.resume_execution("EXE-NOPE")
    assert "EXE-NOPE" in str(exc.value)


def test_resume_execution_offline_skips_reconcile(test_ml_offline):
    """Offline resume must not contact the server."""
    from deriva_ml.execution.state_store import ExecutionStatus

    _insert_test_execution(
        test_ml_offline.workspace, "EXE-A",
        ExecutionStatus.stopped, mode="offline",
    )

    # Just must not raise — offline reconcile is a no-op.
    exe = test_ml_offline.resume_execution("EXE-A")
    assert exe.execution_rid == "EXE-A"


def test_resume_execution_online_flushes_sync_pending(test_ml, monkeypatch):
    """If sync_pending=True on resume (online), flush to catalog."""
    from deriva_ml.execution.state_store import ExecutionStatus

    _insert_test_execution(test_ml.workspace, "EXE-A", ExecutionStatus.stopped)
    # Simulate prior offline transition.
    test_ml.workspace.execution_state_store().update_execution(
        "EXE-A", sync_pending=True,
    )

    flushed_calls = []
    reconcile_calls = []

    def _fake_flush(*, store, catalog, execution_rid):
        flushed_calls.append(execution_rid)
        store.update_execution(execution_rid, sync_pending=False)

    def _fake_reconcile(*, store, catalog, execution_rid):
        reconcile_calls.append(execution_rid)

    monkeypatch.setattr(
        "deriva_ml.core.mixins.execution.flush_pending_sync", _fake_flush,
    )
    monkeypatch.setattr(
        "deriva_ml.core.mixins.execution.reconcile_with_catalog", _fake_reconcile,
    )

    exe = test_ml.resume_execution("EXE-A")
    assert flushed_calls == ["EXE-A"]
    # Reconcile runs AFTER flush (so we compare catalog vs synced state).
    assert reconcile_calls == ["EXE-A"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_registry.py -v -k resume
```

Expected: FAIL with `AttributeError: 'DerivaML' object has no attribute 'resume_execution'`.

- [ ] **Step 3: Implement `resume_execution`**

In `src/deriva_ml/core/mixins/execution.py`, add imports:

```python
from deriva_ml.execution.state_machine import (
    flush_pending_sync,
    reconcile_with_catalog,
)
```

Add the method:

```python
def resume_execution(self, execution_rid: "RID") -> "Execution":
    """Re-hydrate an Execution from the workspace SQLite registry.

    Works in both online and offline modes. The execution's recorded
    mode is independent of the current DerivaML instance's mode — a
    user can create an execution online, run it offline, then upload
    online, all via the same RID.

    Before returning, runs just-in-time state reconciliation
    (spec §2.2): if online and sync_pending=True, flushes SQLite to
    the catalog; then checks for catalog/SQLite disagreement and
    applies the disagreement rules.

    Args:
        execution_rid: Server-assigned Execution RID returned by a
            prior create_execution call.

    Returns:
        An Execution object bound to this DerivaML instance, with
        lifecycle fields as SQLite read-through properties (see
        spec §2.3).

    Raises:
        DerivaMLException: If no matching executions row exists in
            the workspace registry.
        DerivaMLStateInconsistency: If just-in-time reconciliation
            surfaces a disagreement outside the six documented cases
            (see state_machine.reconcile_with_catalog).

    Example:
        >>> ml = DerivaML(hostname="example.org", catalog_id="42")
        >>> exe = ml.resume_execution("5-ABC")
        >>> exe.status
        <ExecutionStatus.stopped>
        >>> exe.upload_outputs()
    """
    from deriva_ml.core.exceptions import DerivaMLException
    from deriva_ml.execution.execution import Execution

    store = self.workspace.execution_state_store()
    row = store.get_execution(execution_rid)
    if row is None:
        raise DerivaMLException(
            f"Execution {execution_rid} is not in the workspace registry. "
            f"Either it was never created on this workspace, or it was "
            f"garbage-collected. Use ml.list_executions() to see what's "
            f"available locally."
        )

    # Just-in-time reconciliation. Online only — offline mode has no
    # catalog to compare against.
    if self._mode is ConnectionMode.online:
        # Order matters: flush first (push our newer state) before
        # reconcile (which would otherwise see stale catalog state
        # as a disagreement). See spec §4.6 step 3.
        if row["sync_pending"]:
            flush_pending_sync(
                store=store, catalog=self.catalog,
                execution_rid=execution_rid,
            )
        reconcile_with_catalog(
            store=store, catalog=self.catalog,
            execution_rid=execution_rid,
        )

    # Construct Execution bound to this DerivaML — it reads lifecycle
    # fields from SQLite via read-through properties (Group E).
    return Execution._from_registry(
        ml_object=self, execution_rid=execution_rid,
    )
```

**Note:** `Execution._from_registry` is created in Task E1 (read-through lifecycle fields). Until then, this test will fail with an AttributeError on `_from_registry`. To enable isolated testing of `resume_execution` before Group E lands, Group E's first task introduces `_from_registry` *before* modifying the Execution public properties. Adjust ordering of Groups D/E if desired, but the plan as written dispatches D4 after E1 for this reason — see ordering note at the top of Group E.

Alternative for a standalone D4: implement a temporary `Execution._from_registry` returning an existing-Execution-shaped object in D4 itself, to be replaced in E1. The test above checks only `exe.execution_rid`, which the existing class already supports.

- [ ] **Step 3a: Add temporary `_from_registry` classmethod on existing `Execution`**

In `src/deriva_ml/execution/execution.py`, add this classmethod on `Execution` (near the existing `__init__`). This minimal version lets Group D finish independently; Group E replaces the body:

```python
@classmethod
def _from_registry(cls, *, ml_object, execution_rid: str) -> "Execution":
    """Bind an Execution to an existing SQLite registry row.

    Distinct from create_execution: does NOT contact the catalog and
    does NOT POST a new row. Called by ml.resume_execution.

    Temporary implementation for Group D — Group E replaces the body
    to wire up read-through lifecycle properties.
    """
    # Minimal construction: skip the existing __init__'s catalog
    # interactions. Store the rid and ml_object so Group E has a
    # starting point.
    instance = cls.__new__(cls)
    instance._ml_object = ml_object
    instance.execution_rid = execution_rid
    instance._dry_run = False
    # Fields the existing class expects to exist:
    instance.datasets = []
    instance.dataset_rids = []
    instance.asset_paths = {}
    instance.configuration = None   # Group E loads from config_json
    instance._working_dir = ml_object.working_dir
    instance._cache_dir = ml_object.cache_dir
    return instance
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_registry.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/core/mixins/execution.py src/deriva_ml/execution/execution.py tests/execution/test_execution_registry.py
git commit -m "feat(execution): DerivaML.resume_execution with JIT reconcile

SQLite-registry read + online-mode flush_pending_sync +
reconcile_with_catalog. Temporary Execution._from_registry
classmethod enables Group D to finish standalone; Group E will
replace the body to wire read-through lifecycle properties.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task D5: `DerivaML.gc_executions()`

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py`
- Modify: `src/deriva_ml/execution/state_store.py` (add `delete_execution`)
- Test: `tests/execution/test_execution_registry.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/execution/test_execution_registry.py`:

```python
def test_gc_executions_deletes_matching(test_ml):
    from datetime import timedelta
    from deriva_ml.execution.state_store import ExecutionStatus

    # Three uploaded executions of different ages.
    _insert_test_execution(test_ml.workspace, "OLD", ExecutionStatus.uploaded)
    _insert_test_execution(test_ml.workspace, "NEW", ExecutionStatus.uploaded)
    _insert_test_execution(test_ml.workspace, "RUN", ExecutionStatus.running)

    # Backdate OLD so it matches older_than.
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)
    store.update_execution(
        "OLD",
        last_activity=now - timedelta(days=30),
        created_at=now - timedelta(days=30),
    )

    n = test_ml.gc_executions(
        status=ExecutionStatus.uploaded,
        older_than=timedelta(days=7),
    )
    assert n == 1
    rids = {r.rid for r in test_ml.list_executions()}
    assert rids == {"NEW", "RUN"}


def test_gc_executions_status_only(test_ml):
    from deriva_ml.execution.state_store import ExecutionStatus

    _insert_test_execution(test_ml.workspace, "A", ExecutionStatus.aborted)
    _insert_test_execution(test_ml.workspace, "B", ExecutionStatus.running)

    n = test_ml.gc_executions(status=ExecutionStatus.aborted)
    assert n == 1
    assert {r.rid for r in test_ml.list_executions()} == {"B"}


def test_gc_executions_delete_working_dir(test_ml, tmp_path):
    from deriva_ml.execution.state_store import ExecutionStatus
    from pathlib import Path

    _insert_test_execution(test_ml.workspace, "EXE-A", ExecutionStatus.uploaded)

    # Create the working directory files.
    work = Path(test_ml.working_dir) / "execution/EXE-A"
    work.mkdir(parents=True)
    (work / "scratch.txt").write_text("hi")
    assert work.exists()

    n = test_ml.gc_executions(
        status=ExecutionStatus.uploaded,
        delete_working_dir=True,
    )
    assert n == 1
    assert not work.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_registry.py -v -k gc
```

Expected: FAIL with `AttributeError: gc_executions`.

- [ ] **Step 3: Add `delete_execution` helper to `ExecutionStateStore`**

Append to `ExecutionStateStore` in `src/deriva_ml/execution/state_store.py`:

```python
    def delete_execution(self, execution_rid: str) -> None:
        """Delete an execution row and all its pending_rows /
        directory_rules.

        Foreign keys cascade via ON DELETE, but SQLite only honors
        that with PRAGMA foreign_keys=ON (which the workspace sets).
        Belt-and-suspenders: we explicitly delete children first, so
        the ORDER of deletions is predictable and so callers running
        without FK-on get sensible behavior.

        Args:
            execution_rid: Which execution to remove.
        """
        from sqlalchemy import delete

        with self.engine.begin() as conn:
            conn.execute(
                delete(self.pending_rows).where(
                    self.pending_rows.c.execution_rid == execution_rid
                )
            )
            conn.execute(
                delete(self.directory_rules).where(
                    self.directory_rules.c.execution_rid == execution_rid
                )
            )
            conn.execute(
                delete(self.executions).where(
                    self.executions.c.rid == execution_rid
                )
            )
```

- [ ] **Step 4: Add `gc_executions` on `DerivaML`**

In `src/deriva_ml/core/mixins/execution.py`, add:

```python
def gc_executions(
    self,
    *,
    older_than: "timedelta | None" = None,
    status: "ExecutionStatus | list[ExecutionStatus] | None" = None,
    delete_working_dir: bool = False,
) -> int:
    """Garbage-collect execution registry rows matching the filters.

    By default only removes registry state (SQLite rows and their
    pending_rows / directory_rules). Pass delete_working_dir=True to
    also `rm -rf` the on-disk execution root under the workspace.

    Does NOT touch the catalog. Executions uploaded to the catalog
    remain there regardless of local gc.

    Args:
        older_than: If set, only gc executions whose last_activity is
            older than this timedelta.
        status: Filter by status (single or list); None = any status.
            Typical: pass ExecutionStatus.uploaded to clean up after
            successful uploads.
        delete_working_dir: If True, remove the per-execution working
            directory from disk. Defaults to False (registry-only).

    Returns:
        The number of executions removed.

    Example:
        >>> from datetime import timedelta
        >>> from deriva_ml.execution.state_store import ExecutionStatus
        >>> n = ml.gc_executions(
        ...     status=ExecutionStatus.uploaded,
        ...     older_than=timedelta(days=30),
        ...     delete_working_dir=True,
        ... )
        >>> print(f"cleaned {n} old executions")
    """
    import shutil
    from datetime import datetime, timezone
    from pathlib import Path

    store = self.workspace.execution_state_store()

    # Build the filter list. If older_than is set, compute the cutoff.
    since = None  # inverse: we want older-than, not newer-than
    rows = store.list_executions(status=status)
    if older_than is not None:
        cutoff = datetime.now(timezone.utc) - older_than
        rows = [r for r in rows if r["last_activity"] < cutoff]

    for row in rows:
        if delete_working_dir:
            wd = Path(self.working_dir) / row["working_dir_rel"]
            if wd.exists():
                shutil.rmtree(wd)
        store.delete_execution(row["rid"])

    return len(rows)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_registry.py -v
```

Expected: 12 passed.

- [ ] **Step 6: Commit**

```bash
git add src/deriva_ml/core/mixins/execution.py src/deriva_ml/execution/state_store.py tests/execution/test_execution_registry.py
git commit -m "feat(execution): gc_executions + state_store.delete_execution

gc_executions removes registry rows matching (status, older_than).
delete_working_dir=True also rm -rf the per-execution working dir.
Delete_execution clears children explicitly (pending_rows,
directory_rules) then the executions row — belt and suspenders beyond
FK cascade.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task D6: `create_execution` accepts both config-object and kwargs forms

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py` (existing `create_execution`)
- Modify: `src/deriva_ml/execution/execution_configuration.py` (string shorthand parser)
- Test: `tests/execution/test_execution_registry.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/execution/test_execution_registry.py`:

```python
def test_create_execution_kwargs_form_accepts_string_shorthand(test_ml):
    """datasets=["RID@version"] should coerce to DatasetSpecConfig."""
    from deriva_ml.execution import ExecutionConfiguration
    from deriva_ml.dataset import DatasetSpecConfig

    # Build the config the kwargs form should produce.
    expected = ExecutionConfiguration(
        datasets=[DatasetSpecConfig(rid="1-XYZ", version="1.0.0")],
        workflow=test_ml.lookup_workflow_by_url("__test_workflow__"),
        description="kwargs form",
    )

    exe = test_ml.create_execution(
        datasets=["1-XYZ@1.0.0"],
        workflow="__test_workflow__",
        description="kwargs form",
    )
    assert exe.configuration.datasets[0].rid == expected.datasets[0].rid
    assert exe.configuration.datasets[0].version == expected.datasets[0].version


def test_create_execution_rejects_mixed_forms(test_ml):
    import pytest
    from deriva_ml.execution import ExecutionConfiguration

    cfg = ExecutionConfiguration(
        workflow=test_ml.lookup_workflow_by_url("__test_workflow__"),
        description="cfg",
    )

    with pytest.raises(TypeError) as exc:
        test_ml.create_execution(cfg, datasets=["X@1.0.0"])
    assert "cannot mix" in str(exc.value).lower() or "exactly one" in str(exc.value).lower()


def test_create_execution_offline_raises(test_ml_offline):
    from deriva_ml.core.exceptions import DerivaMLOfflineError
    import pytest

    with pytest.raises(DerivaMLOfflineError):
        test_ml_offline.create_execution(
            description="can't",
            workflow="__test_workflow__",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_registry.py -v -k create_execution
```

Expected: FAIL with `TypeError` (unexpected kwarg) or similar.

- [ ] **Step 3: Add string-shorthand parser to `DatasetSpecConfig`**

In `src/deriva_ml/execution/execution_configuration.py` (or wherever
`DatasetSpecConfig` is defined — from grounding it's in the dataset
module; search for it):

```python
@classmethod
def from_shorthand(cls, s: str) -> "DatasetSpecConfig":
    """Parse 'RID@version' into a DatasetSpecConfig.

    Accepts both 'RID' (no version, means 'latest') and 'RID@version'.
    Used by the create_execution kwargs form.

    Args:
        s: The shorthand string.

    Returns:
        A DatasetSpecConfig instance.

    Raises:
        ValueError: If the string contains more than one '@' or is empty.

    Example:
        >>> DatasetSpecConfig.from_shorthand("1-XYZ@2.0.0")
        DatasetSpecConfig(rid='1-XYZ', version='2.0.0', ...)
        >>> DatasetSpecConfig.from_shorthand("1-XYZ")
        DatasetSpecConfig(rid='1-XYZ', version=None, ...)
    """
    if not s:
        raise ValueError("empty dataset shorthand")
    parts = s.split("@")
    if len(parts) == 1:
        return cls(rid=parts[0])
    if len(parts) == 2:
        return cls(rid=parts[0], version=parts[1])
    raise ValueError(f"dataset shorthand has too many '@' separators: {s!r}")
```

- [ ] **Step 4: Add kwargs form + offline guard to `create_execution`**

In `src/deriva_ml/core/mixins/execution.py`, locate the existing
`create_execution` method (line 71 per grounding). Modify its signature
and body to accept both forms:

```python
def create_execution(
    self,
    configuration: "ExecutionConfiguration | None" = None,
    *,
    datasets: "list[DatasetSpecConfig | str] | None" = None,
    assets: "list[AssetRIDConfig | str] | None" = None,
    workflow: "Workflow | str | None" = None,
    description: "str | None" = None,
    dry_run: bool = False,
) -> "Execution":
    """Create a new execution and register it in SQLite.

    Accepts either a pre-built ExecutionConfiguration (the
    config-object form) OR kwargs that the method assembles into an
    ExecutionConfiguration (the kwargs form). Mixing is not allowed
    and raises TypeError — pick one or the other.

    Creating executions requires online mode because the Execution
    RID is server-assigned.

    Args:
        configuration: A pre-built ExecutionConfiguration. If this is
            provided, all the kwargs below must be None.
        datasets: Kwargs form only. List of DatasetSpecConfig or
            "RID@version" strings; strings are coerced.
        assets: Kwargs form only. List of AssetRIDConfig or "RID"
            strings.
        workflow: Kwargs form only. A Workflow object, or a string
            that the method looks up via lookup_workflow_by_url.
        description: Kwargs form only. Passes through to
            ExecutionConfiguration.
        dry_run: Pre-existing arg; does not write to the catalog.

    Returns:
        A new Execution object bound to this DerivaML instance with
        status=created in the workspace registry.

    Raises:
        TypeError: If configuration is given alongside kwargs.
        DerivaMLOfflineError: If the current mode is offline.

    Example (config form):
        >>> cfg = ExecutionConfiguration(
        ...     datasets=[DatasetSpecConfig(rid="1-XYZ", version="1.0.0")],
        ...     workflow=my_workflow,
        ...     description="First run",
        ... )
        >>> exe = ml.create_execution(cfg)

    Example (kwargs form):
        >>> exe = ml.create_execution(
        ...     datasets=["1-XYZ@1.0.0"],
        ...     workflow=my_workflow,
        ...     description="First run",
        ... )
    """
    from deriva_ml.core.exceptions import DerivaMLOfflineError
    from deriva_ml.execution import ExecutionConfiguration
    from deriva_ml.dataset import DatasetSpecConfig
    from deriva_ml.execution import AssetRIDConfig

    if self._mode is ConnectionMode.offline:
        raise DerivaMLOfflineError(
            "create_execution requires online mode — the Execution RID is "
            "server-assigned. Switch to ConnectionMode.online to create, "
            "then you can run offline scripts via resume_execution."
        )

    kwargs_provided = any(
        x is not None for x in (datasets, assets, workflow, description)
    )
    if configuration is not None and kwargs_provided:
        raise TypeError(
            "create_execution: cannot mix configuration= with "
            "datasets/assets/workflow/description kwargs; pass exactly one form."
        )

    if configuration is None:
        # Kwargs form: build ExecutionConfiguration.
        ds_specs = []
        for d in (datasets or []):
            if isinstance(d, str):
                ds_specs.append(DatasetSpecConfig.from_shorthand(d))
            else:
                ds_specs.append(d)
        as_specs = []
        for a in (assets or []):
            if isinstance(a, str):
                as_specs.append(AssetRIDConfig(rid=a))
            else:
                as_specs.append(a)
        wf = workflow
        if isinstance(wf, str):
            wf = self.lookup_workflow_by_url(wf)

        configuration = ExecutionConfiguration(
            datasets=ds_specs,
            assets=as_specs,
            workflow=wf,
            description=description,
        )

    # Existing body: create Execution, register in catalog, register
    # in SQLite workspace. The SQLite registry insert is Task D7.
    return Execution(
        ml_object=self,
        configuration=configuration,
        dry_run=dry_run,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_registry.py -v -k create_execution
```

Expected: 3 new passing tests.

- [ ] **Step 6: Commit**

```bash
git add src/deriva_ml/core/mixins/execution.py src/deriva_ml/execution/execution_configuration.py tests/execution/test_execution_registry.py
git commit -m "feat(execution): create_execution accepts kwargs + string shorthand

Per spec §2.7 / R6.1. Both forms supported: configuration=X OR
datasets=[...], workflow=..., description=... kwargs. Mixing raises
TypeError. Offline mode raises DerivaMLOfflineError. String shorthand
'RID@version' coerces to DatasetSpecConfig via from_shorthand.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task D7: `create_execution` writes SQLite registry row

**Files:**
- Modify: `src/deriva_ml/execution/execution.py` (existing `Execution.__init__`)
- Test: `tests/execution/test_execution_registry.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/execution/test_execution_registry.py`:

```python
def test_create_execution_writes_registry_row(test_ml):
    """After create_execution, the workspace SQLite must have the row."""
    from deriva_ml.execution.state_store import ExecutionStatus

    exe = test_ml.create_execution(
        workflow="__test_workflow__",
        description="registry test",
    )
    store = test_ml.workspace.execution_state_store()
    row = store.get_execution(exe.execution_rid)
    assert row is not None
    assert row["rid"] == exe.execution_rid
    # Initial status is 'created'.
    assert row["status"] == ExecutionStatus.created
    assert row["mode"] == "online"
    assert row["config_json"]  # non-empty
    assert row["working_dir_rel"].startswith("execution/")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_registry.py::test_create_execution_writes_registry_row -v
```

Expected: FAIL — row not in SQLite (create_execution doesn't know about the registry yet).

- [ ] **Step 3: Add SQLite insert to `Execution.__init__`**

In `src/deriva_ml/execution/execution.py`, at the end of `Execution.__init__`
(after the existing execution-root directory creation), add:

```python
        # Register the execution in the workspace SQLite registry.
        # Per spec §2.4, SQLite is authoritative for local state; the
        # file-tree exists but we do NOT rely on listing the filesystem
        # to enumerate executions anymore.
        if not self._dry_run:
            from datetime import datetime, timezone
            import json

            from deriva_ml.core.connection_mode import ConnectionMode
            from deriva_ml.execution.state_store import ExecutionStatus

            store = self._ml_object.workspace.execution_state_store()
            now = datetime.now(timezone.utc)

            # Serialize the ExecutionConfiguration. Pydantic v2 dumps to
            # a plain dict; json.dumps then serializes. Include the rid
            # + version info so a reconstructed configuration from a
            # resume_execution call is faithful.
            config_json = self.configuration.model_dump_json()

            try:
                store.insert_execution(
                    rid=self.execution_rid,
                    workflow_rid=self.workflow_rid,
                    description=self.configuration.description,
                    config_json=config_json,
                    status=ExecutionStatus.created,
                    mode=self._ml_object._mode,
                    working_dir_rel=f"execution/{self.execution_rid}",
                    created_at=now,
                    last_activity=now,
                )
            except Exception as exc:
                # The catalog row is already created at this point —
                # don't leave a catalog Execution with no SQLite sibling.
                # Log; the user can still restore via lookup_execution,
                # but workspace-based resume is impaired until the row
                # is re-registered manually or via a future adopt helper.
                logger = logging.getLogger("deriva_ml.execution")
                logger.error(
                    "create_execution %s: catalog POST succeeded but "
                    "SQLite registry write FAILED (%s). The execution "
                    "can be recovered via ml.lookup_execution(rid) and "
                    "manual adoption.",
                    self.execution_rid, exc,
                )
                raise
```

- [ ] **Step 4: Run test to verify it passes**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_registry.py -v
```

Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/execution.py tests/execution/test_execution_registry.py
git commit -m "feat(execution): create_execution writes SQLite registry row

Post-catalog-POST, the new executions row is written to the workspace
SQLite registry with status='created', mode from the DerivaML instance,
and config_json = ExecutionConfiguration.model_dump_json(). Dry-run
mode skips the registry write.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task D8: Hard-cutover rename `restore_execution` → `resume_execution`

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py`
- Modify: `src/deriva_ml/mcp/` (any MCP tool referencing restore_execution — check)
- Modify: `tests/execution/test_execution.py` (existing tests)
- Modify: any callers in `src/deriva_ml/` found by grep

- [ ] **Step 1: Find all callers of `restore_execution`**

```bash
grep -rn "restore_execution" /Users/carl/GitHub/deriva-ml/.claude/worktrees/compassionate-visvesvaraya/src/ /Users/carl/GitHub/deriva-ml/.claude/worktrees/compassionate-visvesvaraya/tests/
```

Expected: Listed in grounding — mixins/execution.py:194, plus test files.

- [ ] **Step 2: Write a guard test that the old name is removed**

Create/extend `tests/execution/test_execution_registry.py`:

```python
def test_restore_execution_symbol_removed(test_ml):
    """Per R5.1 aggressive deprecation, restore_execution is removed."""
    assert not hasattr(test_ml, "restore_execution"), (
        "restore_execution should have been removed in D8; "
        "use resume_execution (see CHANGELOG breaking changes)."
    )
```

- [ ] **Step 3: Run test to verify it fails**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_registry.py::test_restore_execution_symbol_removed -v
```

Expected: FAIL — `restore_execution` still present.

- [ ] **Step 4: Delete `restore_execution` from ExecutionMixin**

In `src/deriva_ml/core/mixins/execution.py` (line 194 per grounding), delete the entire `restore_execution` method definition. Do NOT leave a shim.

- [ ] **Step 5: Update every caller**

For each file yielded by Step 1, replace `restore_execution` with `resume_execution`. Representative sites (check grep output for the complete list):

- `tests/execution/test_execution.py` — callers update method name only (signature unchanged: `ml.restore_execution(rid)` → `ml.resume_execution(rid)`).
- Any `src/deriva_ml/mcp/*.py` tool mapping.
- Docstrings and docs that mention `restore_execution`.

Use a single mechanical replacement:

```bash
# Dry-run first to review:
grep -rln "restore_execution" src/ tests/ | xargs sed -n 's/restore_execution/resume_execution/gp'

# Apply:
grep -rln "restore_execution" src/ tests/ | xargs sed -i '' 's/restore_execution/resume_execution/g'
```

(`-i ''` is the macOS BSD sed incantation; on Linux use `-i` alone.)

- [ ] **Step 6: Run the full test-execution suite**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/ -q --timeout=300
```

Expected: all pass, including the new guard test.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(execution)!: rename restore_execution → resume_execution

BREAKING CHANGE: restore_execution is removed. Use resume_execution.
Per spec §1.3, R5.1 — hard cutover, no shim. See CHANGELOG breaking
changes section (added in Group H).

All callers in src/ and tests/ updated. MCP tool mappings (if any)
updated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

*(End of Task Group D — execution registry API on DerivaML.)*

---

## Task Group E — Read-through lifecycle + `DatasetCollection` + hierarchy renames

Converts `Execution`'s in-memory `status`, `start_time`, `stop_time`, and
`error` fields to SQLite read-through properties (spec §2.3). Wraps
`exe.datasets` in a `DatasetCollection` (RID-keyed mapping + iterable,
per spec §2.8 / R-datasets-mapping). Renames hierarchy traversal to
match the dataset template (R-hierarchy).

### Task E1: Read-through `status` property

**Files:**
- Modify: `src/deriva_ml/execution/execution.py`
- Test: `tests/execution/test_execution_readthrough.py`

- [ ] **Step 1: Write the failing test**

Create `tests/execution/test_execution_readthrough.py`:

```python
"""Tests for SQLite-backed read-through properties on Execution."""

from __future__ import annotations

from datetime import datetime, timezone


def test_status_reads_from_sqlite(test_ml):
    """Mutating SQLite via state-machine transition must be visible
    via exe.status without any cache invalidation."""
    from deriva_ml.execution.state_store import ExecutionStatus

    exe = test_ml.create_execution(
        description="readthrough test",
        workflow="__test_workflow__",
    )

    # Direct SQLite mutation (simulating state_machine.transition):
    store = test_ml.workspace.execution_state_store()
    store.update_execution(exe.execution_rid, status=ExecutionStatus.running)

    # exe.status must reflect the change on the very next read.
    assert exe.status is ExecutionStatus.running

    store.update_execution(exe.execution_rid, status=ExecutionStatus.stopped)
    assert exe.status is ExecutionStatus.stopped


def test_status_reflects_second_process(test_ml):
    """Two Execution instances bound to the same rid must see the
    same SQLite row — no per-instance caching."""
    from deriva_ml.execution.state_store import ExecutionStatus

    exe_a = test_ml.create_execution(
        description="two-views",
        workflow="__test_workflow__",
    )
    exe_b = test_ml.resume_execution(exe_a.execution_rid)

    store = test_ml.workspace.execution_state_store()
    store.update_execution(exe_a.execution_rid, status=ExecutionStatus.running)

    assert exe_a.status is ExecutionStatus.running
    assert exe_b.status is ExecutionStatus.running


def test_status_raises_if_registry_gone(test_ml):
    """If the SQLite registry row is deleted (gc), the read-through
    property surfaces clearly rather than returning stale cache."""
    import pytest
    from deriva_ml.core.exceptions import DerivaMLStateInconsistency

    exe = test_ml.create_execution(
        description="gone",
        workflow="__test_workflow__",
    )
    store = test_ml.workspace.execution_state_store()
    store.delete_execution(exe.execution_rid)

    with pytest.raises(DerivaMLStateInconsistency):
        _ = exe.status
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_readthrough.py -v
```

Expected: FAIL — existing Execution.status is an in-memory attribute; mutating SQLite doesn't reflect.

- [ ] **Step 3: Convert `status` to a property**

In `src/deriva_ml/execution/execution.py`:

1. Locate the existing `self._status = Status.created` assignment in `Execution.__init__` (around line 216 per grounding). Remove it — the state lives in SQLite now.

2. Locate the existing `status` property or attribute (if any). If `status` is a plain attribute, convert it; if it's already a property, replace the body.

3. Add the property:

```python
@property
def status(self) -> "ExecutionStatus":
    """Current execution status, read from SQLite on every access.

    No caching — a mutation from another process (e.g., `deriva-ml
    upload` running in a shell) is visible on the next read.

    Returns:
        The ExecutionStatus value from the workspace registry.

    Raises:
        DerivaMLStateInconsistency: If the executions row for this
            rid is missing (gc'd or never created).

    Example:
        >>> exe = ml.resume_execution("5-ABC")
        >>> exe.status
        <ExecutionStatus.stopped>
    """
    from deriva_ml.core.exceptions import DerivaMLStateInconsistency
    from deriva_ml.execution.state_store import ExecutionStatus

    store = self._ml_object.workspace.execution_state_store()
    row = store.get_execution(self.execution_rid)
    if row is None:
        raise DerivaMLStateInconsistency(
            f"Execution {self.execution_rid} no longer in workspace registry. "
            f"It may have been garbage-collected or the workspace was "
            f"recreated. Use ml.list_executions() to see current state."
        )
    return ExecutionStatus(row["status"])
```

4. Anywhere in the existing `Execution` body that assigns `self._status = ...` (e.g., inside `update_status`, `__enter__`, `__exit__`), replace the assignment with a call to `state_machine.transition(...)`. For this task, simplify by deleting the `self._status = new_status` lines and marking those call sites for Task E2 to complete. A comment placeholder:

```python
# TODO(E2): replace in-memory mutation with state_machine.transition()
# self._status = Status.running   # removed; status lives in SQLite now
```

**Note:** Several tests in `tests/execution/test_execution.py` may break temporarily because `self._status` is gone and the transitions in `__enter__`/`__exit__` are placeholders. Run the full execution test suite at end of Step 4 and expect those breaks; Task E2 fixes them.

- [ ] **Step 4: Run the new readthrough tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_readthrough.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/execution.py tests/execution/test_execution_readthrough.py
git commit -m "feat(execution): status is SQLite read-through property

No in-memory caching; every read hits the workspace registry. Two
Execution instances for the same RID see the same state
consistently; inter-process mutations visible on next read. Missing
registry row raises DerivaMLStateInconsistency.

Existing self._status = ... assignments in __enter__/__exit__/
update_status replaced with TODO comments; Task E2 wires
state_machine.transition() into those sites.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task E2: Wire `state_machine.transition` into lifecycle methods

**Files:**
- Modify: `src/deriva_ml/execution/execution.py`
- Test: `tests/execution/test_execution_readthrough.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/execution/test_execution_readthrough.py`:

```python
def test_execute_enter_transitions_to_running(test_ml):
    from deriva_ml.execution.state_store import ExecutionStatus

    exe = test_ml.create_execution(
        description="lifecycle",
        workflow="__test_workflow__",
    )
    assert exe.status is ExecutionStatus.created

    with exe.execute() as _e:
        assert exe.status is ExecutionStatus.running

    assert exe.status is ExecutionStatus.stopped


def test_execute_exit_with_exception_transitions_to_failed(test_ml):
    import pytest
    from deriva_ml.execution.state_store import ExecutionStatus

    exe = test_ml.create_execution(
        description="lifecycle-fail",
        workflow="__test_workflow__",
    )

    with pytest.raises(RuntimeError):
        with exe.execute():
            raise RuntimeError("boom")

    assert exe.status is ExecutionStatus.failed
    # error message captured:
    assert "boom" in (exe.error or "")


def test_abort_transitions_to_aborted(test_ml):
    from deriva_ml.execution.state_store import ExecutionStatus

    exe = test_ml.create_execution(
        description="abort-test",
        workflow="__test_workflow__",
    )
    exe.abort()
    assert exe.status is ExecutionStatus.aborted
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_readthrough.py -v -k "execute_enter or execute_exit or abort"
```

Expected: FAIL — transitions don't happen yet.

- [ ] **Step 3: Wire `transition()` into `__enter__` / `__exit__` / `abort`**

In `src/deriva_ml/execution/execution.py`:

1. Add imports at the top:

```python
from deriva_ml.execution.state_machine import transition
from deriva_ml.execution.state_store import ExecutionStatus
```

2. Locate `__enter__` (around line 1771 per grounding). Replace its body:

```python
def __enter__(self) -> "Execution":
    """Begin the execution: status → running (synced to catalog online)."""
    from datetime import datetime, timezone

    current = self.status  # reads SQLite
    transition(
        store=self._ml_object.workspace.execution_state_store(),
        catalog=(
            self._ml_object.catalog
            if self._ml_object._mode is ConnectionMode.online
            else None
        ),
        execution_rid=self.execution_rid,
        current=current,
        target=ExecutionStatus.running,
        mode=self._ml_object._mode,
        extra_fields={"start_time": datetime.now(timezone.utc)},
    )
    return self
```

3. Locate `__exit__` (around line 1782). Replace its body:

```python
def __exit__(self, exc_type, exc_val, exc_tb):
    """End the execution: status → stopped (clean) or failed (exception)."""
    from datetime import datetime, timezone

    current = self.status
    now = datetime.now(timezone.utc)

    if exc_val is None:
        target = ExecutionStatus.stopped
        extra = {"stop_time": now}
    else:
        target = ExecutionStatus.failed
        extra = {"stop_time": now, "error": f"{exc_type.__name__}: {exc_val}"}

    transition(
        store=self._ml_object.workspace.execution_state_store(),
        catalog=(
            self._ml_object.catalog
            if self._ml_object._mode is ConnectionMode.online
            else None
        ),
        execution_rid=self.execution_rid,
        current=current,
        target=target,
        mode=self._ml_object._mode,
        extra_fields=extra,
    )

    # Emit the pending-summary INFO log per §2.12 / R6.3. (Full
    # PendingSummary object lands in Group G — this task logs a
    # placeholder message; Group G replaces with the full render.)
    store = self._ml_object.workspace.execution_state_store()
    counts = store.count_pending_by_kind(execution_rid=self.execution_rid)
    if counts["pending_rows"] or counts["pending_files"]:
        logging.getLogger("deriva_ml.execution").info(
            "[Execution %s] exited with pending: "
            "%d rows, %d files. Call exe.upload_outputs() to flush.",
            self.execution_rid,
            counts["pending_rows"], counts["pending_files"],
        )

    # Propagate the exception if any.
    return False
```

4. Add `abort()` method:

```python
def abort(self) -> None:
    """Mark this execution as aborted.

    Legal from created/running/stopped/failed. Pending rows are not
    discarded; the user can inspect them and decide whether to
    recover via resume_execution or discard via gc.

    Raises:
        InvalidTransitionError: If the current status doesn't allow
            abort (e.g., status='uploaded' — terminal).

    Example:
        >>> exe = ml.resume_execution("EXE-A")
        >>> exe.abort()
        >>> exe.status
        <ExecutionStatus.aborted>
    """
    transition(
        store=self._ml_object.workspace.execution_state_store(),
        catalog=(
            self._ml_object.catalog
            if self._ml_object._mode is ConnectionMode.online
            else None
        ),
        execution_rid=self.execution_rid,
        current=self.status,
        target=ExecutionStatus.aborted,
        mode=self._ml_object._mode,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_readthrough.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Run the existing execution test suite for regressions**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/ -q --timeout=300
```

Expected: either all pass, or any failures relate to assertions on the old `self._status` attribute. Fix those assertions by replacing `exe._status` with `exe.status` (behavior identical from callers' perspective).

- [ ] **Step 6: Commit**

```bash
git add src/deriva_ml/execution/execution.py tests/execution/test_execution_readthrough.py
git commit -m "feat(execution): __enter__/__exit__/abort use state_machine.transition

Lifecycle methods now route through the state machine for validation,
SQLite write, and catalog sync. On exit, emits the pending-summary
INFO log (placeholder render — Group G wires the full PendingSummary
object). abort() is a new public method legal from any non-terminal
state.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task E3: `start_time`, `stop_time`, `error` as read-through properties

**Files:**
- Modify: `src/deriva_ml/execution/execution.py`
- Test: `tests/execution/test_execution_readthrough.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/execution/test_execution_readthrough.py`:

```python
def test_start_stop_time_readthrough(test_ml):
    exe = test_ml.create_execution(
        description="times",
        workflow="__test_workflow__",
    )
    assert exe.start_time is None
    assert exe.stop_time is None

    with exe.execute():
        assert exe.start_time is not None
        assert exe.stop_time is None

    assert exe.stop_time is not None
    assert exe.stop_time >= exe.start_time


def test_error_readthrough(test_ml):
    import pytest

    exe = test_ml.create_execution(
        description="err",
        workflow="__test_workflow__",
    )
    assert exe.error is None

    with pytest.raises(ValueError):
        with exe.execute():
            raise ValueError("kaboom")

    assert exe.error is not None
    assert "kaboom" in exe.error
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_readthrough.py -v -k "time or error"
```

Expected: FAIL — `start_time`, `stop_time`, `error` are stale instance attributes.

- [ ] **Step 3: Convert to properties**

In `src/deriva_ml/execution/execution.py`:

1. Remove the `self.start_time = ...` / `self.stop_time = ...` / `self._error = ...` assignments from `__init__`. The state lives in SQLite.

2. Add three properties:

```python
@property
def start_time(self) -> "datetime | None":
    """Start time from SQLite, or None if not yet started.

    Reads on every access (see status docstring for rationale).
    """
    from deriva_ml.core.exceptions import DerivaMLStateInconsistency

    store = self._ml_object.workspace.execution_state_store()
    row = store.get_execution(self.execution_rid)
    if row is None:
        raise DerivaMLStateInconsistency(
            f"Execution {self.execution_rid} not in workspace registry"
        )
    return row["start_time"]

@property
def stop_time(self) -> "datetime | None":
    """Stop time from SQLite, or None if not yet stopped/failed."""
    from deriva_ml.core.exceptions import DerivaMLStateInconsistency

    store = self._ml_object.workspace.execution_state_store()
    row = store.get_execution(self.execution_rid)
    if row is None:
        raise DerivaMLStateInconsistency(
            f"Execution {self.execution_rid} not in workspace registry"
        )
    return row["stop_time"]

@property
def error(self) -> "str | None":
    """Last error message from SQLite, or None if no error recorded."""
    from deriva_ml.core.exceptions import DerivaMLStateInconsistency

    store = self._ml_object.workspace.execution_state_store()
    row = store.get_execution(self.execution_rid)
    if row is None:
        raise DerivaMLStateInconsistency(
            f"Execution {self.execution_rid} not in workspace registry"
        )
    return row["error"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_readthrough.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/execution.py tests/execution/test_execution_readthrough.py
git commit -m "feat(execution): start_time / stop_time / error as read-through

Parallel to status — no caching, SQLite on every read. Raises
DerivaMLStateInconsistency if registry row vanishes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task E4: `DatasetCollection` — RID-keyed mapping over `DatasetBag`s

**Files:**
- Create: `src/deriva_ml/execution/dataset_collection.py`
- Modify: `src/deriva_ml/execution/execution.py` (expose `datasets` as `DatasetCollection`)
- Test: `tests/execution/test_dataset_collection.py`

- [ ] **Step 1: Write the failing test**

Create `tests/execution/test_dataset_collection.py`:

```python
"""Tests for DatasetCollection — RID-keyed mapping + iterable of
DatasetBags accessible via exe.datasets."""

from __future__ import annotations

import pytest


class _FakeBag:
    """Stand-in for DatasetBag for isolated unit tests."""
    def __init__(self, rid: str):
        self.dataset_rid = rid


def test_collection_rid_lookup():
    from deriva_ml.execution.dataset_collection import DatasetCollection

    bag_a = _FakeBag("1-AAA")
    bag_b = _FakeBag("1-BBB")
    coll = DatasetCollection([bag_a, bag_b])

    assert coll["1-AAA"] is bag_a
    assert coll["1-BBB"] is bag_b


def test_collection_missing_key_raises():
    from deriva_ml.execution.dataset_collection import DatasetCollection

    coll = DatasetCollection([_FakeBag("1-AAA")])
    with pytest.raises(KeyError) as exc:
        _ = coll["NOPE"]
    # Error message lists what IS available.
    assert "1-AAA" in str(exc.value)


def test_collection_iteration_yields_bags():
    from deriva_ml.execution.dataset_collection import DatasetCollection

    bag_a = _FakeBag("1-AAA")
    bag_b = _FakeBag("1-BBB")
    coll = DatasetCollection([bag_a, bag_b])

    assert list(coll) == [bag_a, bag_b]


def test_collection_len():
    from deriva_ml.execution.dataset_collection import DatasetCollection

    assert len(DatasetCollection([])) == 0
    assert len(DatasetCollection([_FakeBag("1")])) == 1
    assert len(DatasetCollection([_FakeBag("1"), _FakeBag("2")])) == 2


def test_collection_contains():
    from deriva_ml.execution.dataset_collection import DatasetCollection

    coll = DatasetCollection([_FakeBag("1-AAA")])
    assert "1-AAA" in coll
    assert "NOPE" not in coll


def test_collection_keys_values():
    from deriva_ml.execution.dataset_collection import DatasetCollection

    bags = [_FakeBag("A"), _FakeBag("B")]
    coll = DatasetCollection(bags)
    assert list(coll.keys()) == ["A", "B"]
    assert list(coll.values()) == bags
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_dataset_collection.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `DatasetCollection`**

Create `src/deriva_ml/execution/dataset_collection.py`:

```python
"""RID-keyed mapping + iterable over DatasetBag instances.

Per spec §2.8. Returned by Execution.datasets so users can write
`bag = exe.datasets["1-XYZ"]` (primary access pattern) or iterate
with `for bag in exe.datasets:`. Replaces the previous
list[DatasetBag] exposure (hard cutover per R5.1).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deriva_ml.dataset.dataset_bag import DatasetBag


class DatasetCollection(Mapping):
    """Immutable RID-keyed mapping plus iterable of DatasetBags.

    Backed by a list — the bags are already materialized when the
    collection is constructed. No lazy loading; no mutation after
    construction.

    Iteration yields bags (not keys, which is the Mapping default).
    This matches the intuition "iterate the datasets I materialized",
    which is overwhelmingly what callers want. Use ``.keys()`` for
    RIDs and ``.items()`` for (rid, bag) pairs.

    Attributes:
        (None public — use subscript, iter, len, keys, values, items.)

    Example:
        >>> for bag in exe.datasets:
        ...     print(bag.dataset_rid, len(bag.list_dataset_members()))
        >>> specific = exe.datasets["1-XYZ"]
        >>> "1-XYZ" in exe.datasets
        True
    """

    def __init__(self, bags: "list[DatasetBag]") -> None:
        """Build from a list of DatasetBag instances.

        Args:
            bags: Already-materialized bags, in the order the user
                declared them in ExecutionConfiguration.datasets.
                Multiple bags with the same dataset_rid are not
                supported; the last wins on __getitem__.
        """
        # Preserve order for iteration.
        self._bags = list(bags)
        # Dict for O(1) RID lookup.
        self._by_rid = {b.dataset_rid: b for b in self._bags}

    def __getitem__(self, rid: str) -> "DatasetBag":
        try:
            return self._by_rid[rid]
        except KeyError:
            # More helpful than KeyError('1-XYZ') alone.
            available = ", ".join(self._by_rid) or "(none)"
            raise KeyError(
                f"dataset {rid!r} not in this execution's inputs. "
                f"Available: {available}"
            ) from None

    def __iter__(self) -> "Iterator[DatasetBag]":
        # Mapping's default would iterate keys; we override to iterate
        # values (bags) because that's the overwhelmingly common use.
        return iter(self._bags)

    def __len__(self) -> int:
        return len(self._bags)

    def __contains__(self, rid: object) -> bool:
        return rid in self._by_rid

    def keys(self):
        """Dataset RIDs in declaration order."""
        return list(self._by_rid.keys())

    def values(self):
        """DatasetBag instances in declaration order."""
        return list(self._bags)

    def items(self):
        """(rid, bag) pairs in declaration order."""
        return [(b.dataset_rid, b) for b in self._bags]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_dataset_collection.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/dataset_collection.py tests/execution/test_dataset_collection.py
git commit -m "feat(execution): DatasetCollection RID-keyed mapping

Backed by list + dict; iteration yields DatasetBags (overriding
Mapping default). KeyError on missing RID lists what IS available.
No mutation after construction. Wired into exe.datasets in next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task E5: Hook `DatasetCollection` into `Execution.datasets` + hierarchy renames

**Files:**
- Modify: `src/deriva_ml/execution/execution.py` (replace `self.datasets = []` + `_initialize_execution`)
- Modify: `src/deriva_ml/execution/execution_record.py` (rename `list_parent_executions` / `list_nested_executions`)
- Test: `tests/execution/test_dataset_collection.py` (extend)
- Test: `tests/execution/test_execution_hierarchy.py` (new)

- [ ] **Step 1: Write the failing tests**

Append to `tests/execution/test_dataset_collection.py`:

```python
def test_exe_datasets_is_collection(test_ml_with_real_dataset):
    """Integration-style: after create_execution + enter, exe.datasets
    is a DatasetCollection with RID-keyed lookup."""
    from deriva_ml.execution.dataset_collection import DatasetCollection

    # test_ml_with_real_dataset is a fixture providing a DerivaML
    # with one DatasetSpec already materialized (see tests/conftest.py
    # — will need a new fixture if missing).
    exe = test_ml_with_real_dataset["execution"]
    rid = test_ml_with_real_dataset["dataset_rid"]

    with exe.execute():
        assert isinstance(exe.datasets, DatasetCollection)
        assert rid in exe.datasets
        bag = exe.datasets[rid]
        assert bag.dataset_rid == rid
```

Create `tests/execution/test_execution_hierarchy.py`:

```python
"""Tests for ExecutionRecord.list_execution_parents /
list_execution_children (renamed from list_parent_executions /
list_nested_executions)."""

from __future__ import annotations


def test_list_execution_parents_symbol(test_ml):
    """New name exists; old name is gone."""
    # Use the v2 dataclass lookup — or the old class once merged.
    # For Phase 1, verify on the new dataclass path.
    from deriva_ml.execution.execution_record_v2 import ExecutionRecord

    # v2 dataclass doesn't own list_* methods directly yet — in this
    # plan the methods live on the live-catalog-backed
    # ExecutionRecord (execution_record.py); we rename those. Grab
    # that class:
    from deriva_ml.execution.execution_record import ExecutionRecord as LiveER

    assert hasattr(LiveER, "list_execution_parents"), \
        "list_execution_parents should exist"
    assert hasattr(LiveER, "list_execution_children"), \
        "list_execution_children should exist"
    assert not hasattr(LiveER, "list_parent_executions"), \
        "list_parent_executions should be removed (R5.1 hard cutover)"
    assert not hasattr(LiveER, "list_nested_executions"), \
        "list_nested_executions should be removed (R5.1 hard cutover)"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_dataset_collection.py tests/execution/test_execution_hierarchy.py -v
```

Expected: both fail — `exe.datasets` is still a list; old hierarchy names still present.

- [ ] **Step 3: Replace `self.datasets = []` with `DatasetCollection`**

In `src/deriva_ml/execution/execution.py`:

1. Locate `self.datasets: list[DatasetBag] = []` in `__init__` (around line 222 per grounding). Remove it.

2. Locate `_initialize_execution` (around line 401). Replace `self.datasets.append(...)` pattern with list-then-wrap:

```python
def _initialize_execution(self, reload: "RID | None" = None) -> None:
    # ... existing docstring and code up to the dataset loop ...

    # Materialize dataset bags — kept as a list internally, exposed
    # as a DatasetCollection via the public .datasets property.
    bags = []
    for dataset in self.configuration.datasets:
        self.update_status(Status.initializing, f"Materialize bag {dataset.rid}... ")
        bag = self.download_dataset_bag(dataset)
        bags.append(bag)
        self.dataset_rids.append(dataset.rid)
    self._datasets_list = bags

    # ... remainder of existing body unchanged ...
```

3. Add the public `datasets` property. If a `datasets` attribute is still referenced via `self.datasets` anywhere in the existing body, rename those internal uses to `self._datasets_list`, and add the property:

```python
@property
def datasets(self) -> "DatasetCollection":
    """Input datasets as a RID-keyed mapping + iterable.

    Replaces the previous list[DatasetBag] exposure (hard cutover).
    Pattern:

        >>> bag = exe.datasets["1-XYZ"]          # RID lookup
        >>> for bag in exe.datasets: ...          # iterate bags
        >>> rids = exe.datasets.keys()            # list RIDs

    Returns:
        A DatasetCollection wrapping the materialized DatasetBags.

    Example:
        >>> for bag in exe.datasets:
        ...     print(bag.dataset_rid)
    """
    from deriva_ml.execution.dataset_collection import DatasetCollection
    return DatasetCollection(self._datasets_list)
```

- [ ] **Step 4: Rename hierarchy methods on `ExecutionRecord` (live-catalog class)**

In `src/deriva_ml/execution/execution_record.py`:

1. Rename `list_parent_executions` → `list_execution_parents`. Update the method signature, docstring, and any internal references.
2. Rename `list_nested_executions` → `list_execution_children`. Update the method signature, docstring, and any internal references.

Add updated Tier-1 docstrings per §2.15:

```python
def list_execution_parents(
    self,
    *,
    recurse: bool = False,
) -> "Iterable[ExecutionRecord]":
    """Parent executions that this execution is nested under.

    Mirrors Dataset.list_dataset_parents — same recurse semantics,
    same visited-set cycle guard.

    Args:
        recurse: If True, walk the full ancestor chain.

    Returns:
        Iterable of ExecutionRecord objects for parent executions.
        Empty iterable if this execution is not nested under any.

    Raises:
        DerivaMLException: If this record is not bound to a catalog.

    Example:
        >>> for parent in record.list_execution_parents():
        ...     print(parent.execution_rid)
    """
    # ... existing body, just renamed ...


def list_execution_children(
    self,
    *,
    recurse: bool = False,
) -> "Iterable[ExecutionRecord]":
    """Child executions nested under this one.

    Mirrors Dataset.list_dataset_children.

    Args:
        recurse: If True, walk the full descendant tree.

    Returns:
        Iterable of ExecutionRecord objects for child executions.

    Raises:
        DerivaMLException: If this record is not bound to a catalog.

    Example:
        >>> for child in record.list_execution_children(recurse=True):
        ...     print(child.execution_rid)
    """
    # ... existing body, just renamed ...
```

3. Also remove `Execution.list_nested_executions` if it exists in `execution.py` (around line 1644 per grounding) — per R2.1 the hierarchy queries live on `ExecutionRecord` only. Users who have a live `Execution` obtain the record via the existing path (e.g., `exe._execution_record`, or a new `exe.as_record()` method in Task E6 below).

- [ ] **Step 5: Update all callers**

```bash
grep -rln "list_parent_executions\|list_nested_executions" src/ tests/ | xargs -I{} echo "review {}"
```

For each hit, rename to the new method names. `add_nested_execution` keeps its name (it's a write verb, not a list method — §2.17 mapping).

- [ ] **Step 6: Run tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/ -q --timeout=300
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(execution)!: DatasetCollection + hierarchy method rename

BREAKING CHANGE:
- exe.datasets is now a DatasetCollection (Mapping[rid, DatasetBag]
  + Iterable[DatasetBag]), not list[DatasetBag]. Access by RID:
  exe.datasets['1-XYZ'] instead of exe.datasets[0].
- ExecutionRecord.list_parent_executions → list_execution_parents.
- ExecutionRecord.list_nested_executions → list_execution_children.
- Execution.list_nested_executions removed (per R2.1; use
  record.list_execution_children).

Renames mirror the Dataset template (list_dataset_parents /
list_dataset_children). Hard cutover per R5.1. Added in CHANGELOG
breaking changes in Group H.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

*(End of Task Group E — read-through properties + DatasetCollection + hierarchy renames.)*

---

## Task Group F — RID leasing

Implements lazy, batched, crash-safe RID acquisition against
`public:ERMrest_RID_Lease`. Six tasks at moderate density — the logic
is mostly a faithful transcription of spec §2.6 + §4.4 + §4.5, so the
plan spells out the test fixtures and the API shape but doesn't
repeat spec narrative.

### Task F1: `lease_module` — pure helpers for batched POST

**Files:**
- Create: `src/deriva_ml/execution/rid_lease.py`
- Test: `tests/execution/test_rid_lease.py`

- [ ] **Step 1: Write the failing test**

Create `tests/execution/test_rid_lease.py`:

```python
"""Tests for RID leasing against public:ERMrest_RID_Lease."""

from __future__ import annotations

import uuid

import pytest


class _MockLeaseCatalog:
    """Mock that records POSTs to ERMrest_RID_Lease and returns
    synthetic RIDs keyed by the lease tokens."""
    def __init__(self, *, prefix: str = "RID-", fail: bool = False):
        self.prefix = prefix
        self.fail = fail
        self.post_calls: list[list[dict]] = []

    def post(self, path: str, json=None, **_kw):
        if self.fail:
            raise RuntimeError("simulated lease failure")
        assert "ERMrest_RID_Lease" in path
        assert isinstance(json, list)
        self.post_calls.append(json)
        class _R:
            def __init__(self, bodies, prefix):
                self._bodies = bodies
                self._prefix = prefix
            def json(self):
                return [
                    {"RID": f"{self._prefix}{i}", "ID": b["ID"]}
                    for i, b in enumerate(self._bodies)
                ]
        return _R(json, self.prefix)


def test_generate_lease_token_is_uuid_string():
    from deriva_ml.execution.rid_lease import generate_lease_token

    t = generate_lease_token()
    # Must round-trip through UUID parser.
    uuid.UUID(t)


def test_post_lease_batch_sends_tokens_and_returns_rids():
    from deriva_ml.execution.rid_lease import post_lease_batch

    cat = _MockLeaseCatalog(prefix="RID-")
    tokens = ["T1", "T2", "T3"]
    rids_by_token = post_lease_batch(catalog=cat, tokens=tokens)

    # Every input token received a RID back.
    assert set(rids_by_token.keys()) == set(tokens)
    assert all(v.startswith("RID-") for v in rids_by_token.values())
    # Exactly one POST call with N entries.
    assert len(cat.post_calls) == 1
    assert len(cat.post_calls[0]) == 3


def test_post_lease_batch_chunks(monkeypatch):
    from deriva_ml.execution import rid_lease
    from deriva_ml.execution.rid_lease import post_lease_batch

    monkeypatch.setattr(rid_lease, "PENDING_ROWS_LEASE_CHUNK", 2)
    cat = _MockLeaseCatalog(prefix="X-")
    tokens = ["A", "B", "C", "D", "E"]
    rids_by_token = post_lease_batch(catalog=cat, tokens=tokens)

    # 5 tokens, chunk size 2 → 3 POSTs of 2, 2, 1.
    assert len(cat.post_calls) == 3
    assert len(cat.post_calls[0]) == 2
    assert len(cat.post_calls[1]) == 2
    assert len(cat.post_calls[2]) == 1
    assert set(rids_by_token.keys()) == set(tokens)


def test_post_lease_batch_empty_is_noop():
    from deriva_ml.execution.rid_lease import post_lease_batch

    cat = _MockLeaseCatalog()
    result = post_lease_batch(catalog=cat, tokens=[])
    assert result == {}
    assert cat.post_calls == []


def test_post_lease_batch_propagates_catalog_error():
    from deriva_ml.execution.rid_lease import post_lease_batch

    cat = _MockLeaseCatalog(fail=True)
    with pytest.raises(RuntimeError):
        post_lease_batch(catalog=cat, tokens=["T"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_rid_lease.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `rid_lease.py`**

Create `src/deriva_ml/execution/rid_lease.py`:

```python
"""RID leasing against public:ERMrest_RID_Lease.

Per spec §2.6. Pure helpers — no SQLite awareness here. The
acquire_leases_for_pending function in state_store composition (Task
F2) wires these into the two-phase SQLite protocol.

Why a dedicated module: the POST body format, chunking, and
error-handling choices are specific to the lease table and worth
isolating from the higher-level "take pending_rows with status=staged
and assign them RIDs" orchestration.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deriva.core import ErmrestCatalog

logger = logging.getLogger(__name__)

# Chunk size for batched POSTs. 500 keeps us comfortably under
# ERMrest URL and body-size limits while amortizing round-trip cost.
# See spec §2.6 — may be tuned by tests via monkeypatch.
PENDING_ROWS_LEASE_CHUNK = 500


def generate_lease_token() -> str:
    """Generate a fresh lease token.

    Returns:
        A UUID4 string. Used as the ERMrest_RID_Lease.ID column so
        we can look up what we leased after a mid-flight crash.

    Example:
        >>> token = generate_lease_token()
        >>> len(token) == 36
        True
    """
    return str(uuid.uuid4())


def post_lease_batch(
    *,
    catalog: "ErmrestCatalog",
    tokens: list[str],
) -> dict[str, str]:
    """POST to ERMrest_RID_Lease in chunks; return token→RID map.

    Args:
        catalog: Live ErmrestCatalog to POST against.
        tokens: Lease tokens (typically uuid4 strings from
            generate_lease_token). Empty list is a no-op.

    Returns:
        Dict mapping each input token to its server-assigned RID.

    Raises:
        Exception: Whatever the catalog raises on POST failure.
            Partial progress is NOT rolled back — the caller is
            responsible for recording which tokens landed (via the
            two-phase SQLite write in Task F2).

    Example:
        >>> tokens = [generate_lease_token() for _ in range(100)]
        >>> assigned = post_lease_batch(catalog=cat, tokens=tokens)
        >>> assigned[tokens[0]]
        'EXE-ABC'
    """
    if not tokens:
        return {}

    result: dict[str, str] = {}
    # Chunk to keep URL + body sizes bounded.
    for i in range(0, len(tokens), PENDING_ROWS_LEASE_CHUNK):
        chunk = tokens[i : i + PENDING_ROWS_LEASE_CHUNK]
        body = [{"ID": t} for t in chunk]
        response = catalog.post("/entity/public:ERMrest_RID_Lease", json=body)
        for row in response.json():
            # ERMrest echoes both ID (our token) and RID (assigned).
            result[row["ID"]] = row["RID"]
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_rid_lease.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/rid_lease.py tests/execution/test_rid_lease.py
git commit -m "feat(execution): rid_lease — generate_lease_token + post_lease_batch

Chunked POST to public:ERMrest_RID_Lease. Returns token→RID mapping.
Empty input is a no-op. Errors propagate; caller owns idempotency
(via the two-phase SQLite protocol in Task F2).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task F2: `acquire_leases_for_pending` — two-phase protocol

**Files:**
- Modify: `src/deriva_ml/execution/state_store.py` (add methods)
- Test: `tests/execution/test_state_store.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/execution/test_state_store.py`:

```python
def test_mark_leasing_sets_token_and_status(tmp_path):
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus, PendingRowStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )
    pid = store.insert_pending_row(
        execution_rid="EXE-A", key="k1",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )

    store.mark_pending_leasing(pid, lease_token="TOKEN-1")
    row = store.list_pending_rows(execution_rid="EXE-A")[0]
    assert row["status"] == str(PendingRowStatus.leasing)
    assert row["lease_token"] == "TOKEN-1"


def test_finalize_lease_sets_rid_and_status(tmp_path):
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus, PendingRowStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )
    pid = store.insert_pending_row(
        execution_rid="EXE-A", key="k1",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )
    store.mark_pending_leasing(pid, lease_token="T1")

    store.finalize_pending_lease(lease_token="T1", assigned_rid="1-NEW")
    row = store.list_pending_rows(execution_rid="EXE-A")[0]
    assert row["status"] == str(PendingRowStatus.leased)
    assert row["rid"] == "1-NEW"


def test_revert_leasing_to_staged(tmp_path):
    """Crash recovery: leasing rows with no matching server lease
    revert to staged so the next attempt reissues them."""
    from datetime import datetime, timezone
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus, PendingRowStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = _engine(tmp_path)
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )
    pid = store.insert_pending_row(
        execution_rid="EXE-A", key="k1",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )
    store.mark_pending_leasing(pid, lease_token="T-LOST")

    store.revert_pending_leasing(lease_token="T-LOST")
    row = store.list_pending_rows(execution_rid="EXE-A")[0]
    assert row["status"] == str(PendingRowStatus.staged)
    assert row["lease_token"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_store.py -v -k "mark_leasing or finalize_lease or revert_leasing"
```

Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Add helpers to `ExecutionStateStore`**

Append to `ExecutionStateStore` in `src/deriva_ml/execution/state_store.py`:

```python
    # ─── lease two-phase protocol ───────────────────────────────────

    def mark_pending_leasing(self, pending_id: int, *, lease_token: str) -> None:
        """Phase 1 of RID leasing: write lease_token + status='leasing'.

        Must be committed BEFORE the POST to ERMrest_RID_Lease so that
        a crash between this write and the POST is recoverable via
        revert_pending_leasing (token wasn't yet sent, so no server
        state to reconcile).

        Args:
            pending_id: pending_rows.id to transition.
            lease_token: UUID string. Same token goes in the POST body.
        """
        self.update_pending_row(
            pending_id,
            status=PendingRowStatus.leasing,
            lease_token=lease_token,
        )

    def finalize_pending_lease(
        self,
        *,
        lease_token: str,
        assigned_rid: str,
    ) -> None:
        """Phase 2: POST succeeded, assign the server RID and flip to
        'leased'.

        Identified by lease_token (not pending_id) so this works for
        batched lease responses without requiring the caller to hold
        pending_id↔token mappings.

        Args:
            lease_token: The token we POSTed and got a RID back for.
            assigned_rid: The server-assigned RID from the response.

        Raises:
            Nothing specific — silently no-ops if no row matches the
            token (e.g., another process already finalized). If
            silent no-op is undesirable, the caller should verify
            post-conditions via list_pending_rows.
        """
        from datetime import datetime, timezone
        from sqlalchemy import update

        with self.engine.begin() as conn:
            conn.execute(
                update(self.pending_rows)
                .where(self.pending_rows.c.lease_token == lease_token)
                .values(
                    rid=assigned_rid,
                    status=str(PendingRowStatus.leased),
                    leased_at=datetime.now(timezone.utc),
                )
            )

    def revert_pending_leasing(self, *, lease_token: str) -> None:
        """Rollback: clear lease_token and flip back to 'staged'.

        Called either:
        (a) right after a failed POST (token never landed on server), or
        (b) during startup reconciliation when the token query to
            ERMrest_RID_Lease returns nothing (POST failed silently
            or was dropped before persisting).

        Args:
            lease_token: The token to clear.
        """
        from sqlalchemy import update

        with self.engine.begin() as conn:
            conn.execute(
                update(self.pending_rows)
                .where(self.pending_rows.c.lease_token == lease_token)
                .values(
                    lease_token=None,
                    status=str(PendingRowStatus.staged),
                )
            )

    def list_leasing_rows(
        self,
        *,
        execution_rid: str | None = None,
    ) -> list[dict]:
        """Return rows currently in status='leasing' — candidates for
        startup reconciliation.

        Args:
            execution_rid: If set, scope to one execution; if None,
                return all leasing rows across all executions
                (workspace-wide reconciliation).

        Returns:
            List of pending_rows dicts.
        """
        from sqlalchemy import select

        stmt = select(self.pending_rows).where(
            self.pending_rows.c.status == str(PendingRowStatus.leasing)
        )
        if execution_rid is not None:
            stmt = stmt.where(self.pending_rows.c.execution_rid == execution_rid)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_state_store.py -v
```

Expected: 22 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/state_store.py tests/execution/test_state_store.py
git commit -m "feat(state_store): lease two-phase protocol helpers

mark_pending_leasing (phase 1), finalize_pending_lease (phase 2),
revert_pending_leasing (rollback on POST failure or missing server
state), list_leasing_rows (for startup reconciliation). Key helpers
for the lease orchestrator in F3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task F3: `acquire_leases_for_execution` orchestrator

Composes the state_store helpers and `rid_lease` module into a single
entry point: given a list of pending_ids in status='staged', transition
them to 'leased' with assigned RIDs.

**Files:**
- Create: `src/deriva_ml/execution/lease_orchestrator.py`
- Test: `tests/execution/test_lease_orchestrator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/execution/test_lease_orchestrator.py`:

```python
"""Tests for the lease orchestrator — composes rid_lease + state_store."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine


class _MockLeaseCatalog:
    def __init__(self, *, prefix="RID-", fail_after=None):
        self.prefix = prefix
        self.fail_after = fail_after  # After N successful posts, start failing
        self.post_calls: list[list[dict]] = []

    def post(self, path: str, json=None, **_kw):
        self.post_calls.append(json)
        if self.fail_after is not None and len(self.post_calls) > self.fail_after:
            raise RuntimeError("simulated post failure")
        class _R:
            def __init__(self, bodies, prefix, offset):
                self._bodies = bodies
                self._prefix = prefix
                self._offset = offset
            def json(self):
                return [
                    {"RID": f"{self._prefix}{self._offset + i}", "ID": b["ID"]}
                    for i, b in enumerate(self._bodies)
                ]
        offset = sum(len(p) for p in self.post_calls[:-1])
        return _R(json, self.prefix, offset)


def _setup_store(tmp_path):
    from deriva_ml.execution.state_store import (
        ExecutionStateStore, ExecutionStatus,
    )
    from deriva_ml.core.connection_mode import ConnectionMode

    eng = create_engine(f"sqlite:///{tmp_path}/t.db")
    store = ExecutionStateStore(engine=eng)
    store.ensure_schema()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-A", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-A",
        created_at=now, last_activity=now,
    )
    return store


def test_acquire_leases_happy_path(tmp_path):
    from deriva_ml.execution.lease_orchestrator import acquire_leases_for_execution

    store = _setup_store(tmp_path)
    now = datetime.now(timezone.utc)
    pending_ids = [
        store.insert_pending_row(
            execution_rid="EXE-A", key=f"k{i}",
            target_schema="s", target_table="t",
            metadata_json="{}", created_at=now,
        )
        for i in range(3)
    ]

    cat = _MockLeaseCatalog(prefix="R-")
    acquire_leases_for_execution(
        store=store, catalog=cat, execution_rid="EXE-A",
        pending_ids=pending_ids,
    )

    rows = store.list_pending_rows(execution_rid="EXE-A")
    assert len(rows) == 3
    for r in rows:
        assert r["status"] == "leased"
        assert r["rid"] is not None
        assert r["rid"].startswith("R-")
        assert r["leased_at"] is not None


def test_acquire_leases_skips_already_leased(tmp_path):
    from deriva_ml.execution.state_store import PendingRowStatus
    from deriva_ml.execution.lease_orchestrator import acquire_leases_for_execution

    store = _setup_store(tmp_path)
    now = datetime.now(timezone.utc)
    already_leased = store.insert_pending_row(
        execution_rid="EXE-A", key="already",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
        rid="EXISTING-RID",
        status=PendingRowStatus.leased,
    )
    to_lease = store.insert_pending_row(
        execution_rid="EXE-A", key="new",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )

    cat = _MockLeaseCatalog(prefix="R-")
    acquire_leases_for_execution(
        store=store, catalog=cat, execution_rid="EXE-A",
        pending_ids=[already_leased, to_lease],
    )

    # Only ONE token was POSTed — the already-leased row was skipped.
    assert len(cat.post_calls) == 1
    assert len(cat.post_calls[0]) == 1


def test_acquire_leases_reverts_on_post_failure(tmp_path):
    from deriva_ml.execution.state_store import PendingRowStatus
    from deriva_ml.execution.lease_orchestrator import acquire_leases_for_execution

    store = _setup_store(tmp_path)
    now = datetime.now(timezone.utc)
    ids = [
        store.insert_pending_row(
            execution_rid="EXE-A", key=f"k{i}",
            target_schema="s", target_table="t",
            metadata_json="{}", created_at=now,
        )
        for i in range(2)
    ]

    cat = _MockLeaseCatalog(fail_after=0)  # fail on first POST
    with pytest.raises(RuntimeError):
        acquire_leases_for_execution(
            store=store, catalog=cat, execution_rid="EXE-A",
            pending_ids=ids,
        )

    # Rows must be back in 'staged' — the leasing→staged revert ran.
    rows = store.list_pending_rows(execution_rid="EXE-A")
    for r in rows:
        assert r["status"] == str(PendingRowStatus.staged), \
            "orchestrator must revert on POST failure"
        assert r["lease_token"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_lease_orchestrator.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the orchestrator**

Create `src/deriva_ml/execution/lease_orchestrator.py`:

```python
"""Orchestrator for the two-phase RID lease protocol.

Composes ExecutionStateStore's lease helpers with rid_lease's POST
machinery. One entry point: acquire_leases_for_execution. Called by
handle.rid property and by the upload-engine drain.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deriva_ml.execution.rid_lease import generate_lease_token, post_lease_batch
from deriva_ml.execution.state_store import PendingRowStatus

if TYPE_CHECKING:
    from deriva.core import ErmrestCatalog

    from deriva_ml.execution.state_store import ExecutionStateStore

logger = logging.getLogger(__name__)


def acquire_leases_for_execution(
    *,
    store: "ExecutionStateStore",
    catalog: "ErmrestCatalog",
    execution_rid: str,
    pending_ids: list[int],
) -> None:
    """Transition the given pending rows from 'staged' to 'leased',
    assigning server-issued RIDs.

    Skips rows already in status='leased' (idempotent). Rows in
    other intermediate states (leasing, uploading, uploaded, failed)
    are also skipped — the orchestrator only promotes staged→leased.

    Two-phase protocol:
      1. Generate tokens, mark rows 'leasing' in SQLite (committed
         before the POST).
      2. POST batch to ERMrest_RID_Lease.
      3. On success: finalize each row with its assigned RID
         (status → 'leased').
      4. On POST failure: revert all rows we marked in step 1
         (status → 'staged'; token cleared).

    Crash recovery is handled in Task F4 (reconcile at startup).

    Args:
        store: The ExecutionStateStore holding SQLite state.
        catalog: Live ErmrestCatalog for POSTing to ERMrest_RID_Lease.
        execution_rid: For logging + scoping; all pending_ids must
            belong to this execution (not enforced here; caller's
            concern).
        pending_ids: pending_rows.id values to lease.

    Raises:
        Exception: Whatever the catalog POST raises. Before
            propagating, the orchestrator reverts any rows it had
            marked 'leasing' back to 'staged'.

    Example:
        >>> acquire_leases_for_execution(
        ...     store=store, catalog=ml.catalog,
        ...     execution_rid="EXE-A",
        ...     pending_ids=[1, 2, 3],
        ... )
    """
    if not pending_ids:
        return

    # Filter to rows actually in 'staged'. Build a (pending_id, token)
    # list; the order maps to the POST body order, which maps to the
    # response order in _MockLeaseCatalog and in real ERMrest.
    rows_to_lease: list[tuple[int, str]] = []
    all_rows = {r["id"]: r for r in store.list_pending_rows(execution_rid=execution_rid)}
    for pid in pending_ids:
        row = all_rows.get(pid)
        if row is None:
            logger.warning(
                "acquire_leases: pending_id %d not in execution %s; skipping",
                pid, execution_rid,
            )
            continue
        if row["status"] != str(PendingRowStatus.staged):
            # Already leased or past; skip silently.
            continue
        rows_to_lease.append((pid, generate_lease_token()))

    if not rows_to_lease:
        return

    # Phase 1: write 'leasing' + token to SQLite, committed.
    # This MUST happen before the POST so that if we crash, the token
    # is in SQLite and we can look it up on the server at reconcile.
    for pid, token in rows_to_lease:
        store.mark_pending_leasing(pid, lease_token=token)

    # Phase 2: POST the batch. On failure, revert all.
    tokens = [t for _, t in rows_to_lease]
    try:
        assigned = post_lease_batch(catalog=catalog, tokens=tokens)
    except Exception:
        logger.warning(
            "acquire_leases: POST failed for execution %s; reverting %d rows to staged",
            execution_rid, len(rows_to_lease),
        )
        for _, token in rows_to_lease:
            store.revert_pending_leasing(lease_token=token)
        raise

    # Phase 3: finalize each row with its assigned RID.
    for _, token in rows_to_lease:
        assigned_rid = assigned.get(token)
        if assigned_rid is None:
            # Server response missing this token. Revert just this
            # row; leave the others (they did succeed).
            logger.warning(
                "acquire_leases: token %s missing from server response "
                "for execution %s; reverting that row",
                token, execution_rid,
            )
            store.revert_pending_leasing(lease_token=token)
        else:
            store.finalize_pending_lease(lease_token=token, assigned_rid=assigned_rid)

    logger.debug(
        "acquire_leases: %d rows leased for execution %s",
        len(rows_to_lease), execution_rid,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_lease_orchestrator.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/lease_orchestrator.py tests/execution/test_lease_orchestrator.py
git commit -m "feat(execution): acquire_leases_for_execution orchestrator

Two-phase protocol: mark 'leasing' in SQLite, POST batch, finalize
with assigned RIDs. On POST failure, revert all marked rows back to
'staged'. Idempotent — already-leased rows are skipped.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task F4: Startup lease reconciliation

**Files:**
- Modify: `src/deriva_ml/execution/lease_orchestrator.py`
- Test: `tests/execution/test_lease_orchestrator.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/execution/test_lease_orchestrator.py`:

```python
class _MockCatalogWithGet:
    """Mock exposing GET for querying ERMrest_RID_Lease by token."""
    def __init__(self, *, rows_by_id: dict[str, str] | None = None):
        self.rows_by_id = rows_by_id or {}  # token → RID
        self.get_paths: list[str] = []

    def get(self, path: str, **_kw):
        self.get_paths.append(path)
        # Parse out the ID filter from the URL (ID=T1;ID=T2...).
        import re
        ids_param = re.search(r"ID=([^&]+)", path)
        tokens = ids_param.group(1).split(";") if ids_param else []
        tokens = [re.sub(r"^ID=", "", t) for t in tokens if t]
        rows = [
            {"ID": t, "RID": self.rows_by_id[t]}
            for t in tokens
            if t in self.rows_by_id
        ]
        class _R:
            def __init__(self, rows): self._rows = rows
            def json(self): return self._rows
        return _R(rows)


def test_reconcile_pending_leases_adopts_server_assigned(tmp_path):
    """Row status='leasing' whose token exists on the server →
    promoted to 'leased' with the server's RID."""
    from deriva_ml.execution.lease_orchestrator import reconcile_pending_leases
    from deriva_ml.execution.state_store import PendingRowStatus

    store = _setup_store(tmp_path)
    now = datetime.now(timezone.utc)
    pid = store.insert_pending_row(
        execution_rid="EXE-A", key="k1",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )
    store.mark_pending_leasing(pid, lease_token="T-FOUND")

    cat = _MockCatalogWithGet(rows_by_id={"T-FOUND": "R-LATE"})
    reconcile_pending_leases(store=store, catalog=cat, execution_rid="EXE-A")

    row = store.list_pending_rows(execution_rid="EXE-A")[0]
    assert row["status"] == str(PendingRowStatus.leased)
    assert row["rid"] == "R-LATE"


def test_reconcile_pending_leases_reverts_missing(tmp_path):
    """Row status='leasing' whose token doesn't exist on the server →
    reverted to 'staged'."""
    from deriva_ml.execution.lease_orchestrator import reconcile_pending_leases
    from deriva_ml.execution.state_store import PendingRowStatus

    store = _setup_store(tmp_path)
    now = datetime.now(timezone.utc)
    pid = store.insert_pending_row(
        execution_rid="EXE-A", key="k1",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )
    store.mark_pending_leasing(pid, lease_token="T-ORPHAN")

    cat = _MockCatalogWithGet(rows_by_id={})  # nothing on server
    reconcile_pending_leases(store=store, catalog=cat, execution_rid="EXE-A")

    row = store.list_pending_rows(execution_rid="EXE-A")[0]
    assert row["status"] == str(PendingRowStatus.staged)
    assert row["lease_token"] is None


def test_reconcile_pending_leases_workspace_wide(tmp_path):
    """execution_rid=None reconciles across all executions."""
    from deriva_ml.execution.lease_orchestrator import reconcile_pending_leases
    from deriva_ml.execution.state_store import PendingRowStatus, ExecutionStatus
    from deriva_ml.core.connection_mode import ConnectionMode

    store = _setup_store(tmp_path)  # creates EXE-A
    now = datetime.now(timezone.utc)
    # Add a second execution.
    store.insert_execution(
        rid="EXE-B", workflow_rid=None, description=None,
        config_json="{}", status=ExecutionStatus.running,
        mode=ConnectionMode.online, working_dir_rel="execution/EXE-B",
        created_at=now, last_activity=now,
    )
    pid_a = store.insert_pending_row(
        execution_rid="EXE-A", key="ka",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )
    pid_b = store.insert_pending_row(
        execution_rid="EXE-B", key="kb",
        target_schema="s", target_table="t",
        metadata_json="{}", created_at=now,
    )
    store.mark_pending_leasing(pid_a, lease_token="TA")
    store.mark_pending_leasing(pid_b, lease_token="TB")

    cat = _MockCatalogWithGet(rows_by_id={"TA": "R-A", "TB": "R-B"})
    reconcile_pending_leases(store=store, catalog=cat, execution_rid=None)

    for rid, expected_r in [("EXE-A", "R-A"), ("EXE-B", "R-B")]:
        row = store.list_pending_rows(execution_rid=rid)[0]
        assert row["status"] == str(PendingRowStatus.leased)
        assert row["rid"] == expected_r
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_lease_orchestrator.py -v -k reconcile
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `reconcile_pending_leases`**

Append to `src/deriva_ml/execution/lease_orchestrator.py`:

```python
def reconcile_pending_leases(
    *,
    store: "ExecutionStateStore",
    catalog: "ErmrestCatalog",
    execution_rid: str | None = None,
) -> None:
    """Recover from a crash during the two-phase lease protocol.

    Finds pending_rows in status='leasing' (the intermediate state
    between SQLite write and POST response) and asks ERMrest_RID_Lease
    whether each token made it to the server.

    Per-token outcomes:
    - Token exists on server → adopt the server RID, status → 'leased'.
    - Token doesn't exist → POST never landed, revert to 'staged' so
      the next acquire_leases reissues.

    Call sites:
    - On workspace open (no execution_rid arg: sweep all executions).
    - On resume_execution of a specific rid (pass execution_rid).

    Args:
        store: ExecutionStateStore.
        catalog: Live ErmrestCatalog.
        execution_rid: If None, reconcile across the whole workspace.
            Otherwise scope to one execution (cheaper — typical for
            resume_execution JIT reconciliation).

    Example:
        >>> # Workspace-wide startup reconciliation:
        >>> reconcile_pending_leases(store=store, catalog=ml.catalog)
        >>> # Per-execution on resume:
        >>> reconcile_pending_leases(
        ...     store=store, catalog=ml.catalog,
        ...     execution_rid="EXE-A",
        ... )
    """
    leasing_rows = store.list_leasing_rows(execution_rid=execution_rid)
    if not leasing_rows:
        return

    tokens = [r["lease_token"] for r in leasing_rows if r["lease_token"]]
    if not tokens:
        # Shouldn't happen — leasing rows always carry tokens — but
        # be defensive.
        return

    # Query ERMrest_RID_Lease for the tokens we expect to find there.
    # Use a filter clause: ID=t1;ID=t2;... (ERMrest's in-list syntax).
    # Chunked to stay under URL length limits.
    from deriva_ml.execution.rid_lease import PENDING_ROWS_LEASE_CHUNK

    found_by_token: dict[str, str] = {}
    for i in range(0, len(tokens), PENDING_ROWS_LEASE_CHUNK):
        chunk = tokens[i : i + PENDING_ROWS_LEASE_CHUNK]
        filter_clause = ";".join(f"ID={t}" for t in chunk)
        path = f"/entity/public:ERMrest_RID_Lease/{filter_clause}"
        response = catalog.get(path)
        for row in response.json():
            found_by_token[row["ID"]] = row["RID"]

    # Apply outcomes.
    for row in leasing_rows:
        token = row["lease_token"]
        if token in found_by_token:
            store.finalize_pending_lease(
                lease_token=token,
                assigned_rid=found_by_token[token],
            )
        else:
            store.revert_pending_leasing(lease_token=token)

    logger.info(
        "lease reconciliation: %d rows, %d adopted, %d reverted (execution_rid=%s)",
        len(leasing_rows),
        len(found_by_token),
        len(leasing_rows) - len(found_by_token),
        execution_rid or "all",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_lease_orchestrator.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/lease_orchestrator.py tests/execution/test_lease_orchestrator.py
git commit -m "feat(lease): reconcile_pending_leases crash-recovery

Finds status='leasing' rows, queries ERMrest_RID_Lease by token.
Found → adopt; missing → revert to 'staged'. Workspace-wide or
scoped. Called on workspace open (Task F5) and on resume_execution
(Group D already stubbed the call path).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task F5: Hook lease reconciliation into workspace startup

**Files:**
- Modify: `src/deriva_ml/core/base.py` (`DerivaML.__init__` or post-init)
- Test: `tests/execution/test_lease_orchestrator.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/execution/test_lease_orchestrator.py`:

```python
def test_workspace_open_reconciles_leases(test_ml, monkeypatch):
    """On DerivaML construction (online), startup sweep runs."""
    calls: list[str | None] = []

    def _spy(*, store, catalog, execution_rid):
        calls.append(execution_rid)

    from deriva_ml.execution import lease_orchestrator
    monkeypatch.setattr(lease_orchestrator, "reconcile_pending_leases", _spy)

    # Creating a DerivaML instance should trigger the sweep.
    # test_ml is already constructed; construct another one to observe.
    from deriva_ml import DerivaML
    DerivaML(
        hostname=test_ml.host_name,
        catalog_id=test_ml.catalog_id,
        working_dir=test_ml.working_dir,
    )
    # Sweep scoped workspace-wide (execution_rid=None).
    assert None in calls


def test_offline_workspace_skips_lease_reconcile(monkeypatch, tmp_path):
    """In offline mode there's no server to query — skip the sweep."""
    from deriva_ml.execution import lease_orchestrator
    calls: list[str | None] = []
    def _spy(**_kw): calls.append(_kw.get("execution_rid"))
    monkeypatch.setattr(lease_orchestrator, "reconcile_pending_leases", _spy)

    from deriva_ml import ConnectionMode, DerivaML
    DerivaML(
        hostname="offline.example",
        catalog_id="1",
        working_dir=str(tmp_path),
        mode=ConnectionMode.offline,
    )
    assert calls == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_lease_orchestrator.py -v -k workspace_open
```

Expected: FAIL — the sweep isn't wired in yet.

- [ ] **Step 3: Add the hook**

In `src/deriva_ml/core/base.py`, locate the end of `DerivaML.__init__`
(after the workspace is ready). Add:

```python
        # Reconcile any pending_rows stuck in 'leasing' from a prior
        # crash. Workspace-wide sweep; per-execution reconciliation
        # runs additionally on resume_execution (Group D).
        if self._mode is ConnectionMode.online:
            from deriva_ml.execution.lease_orchestrator import reconcile_pending_leases

            try:
                reconcile_pending_leases(
                    store=self.workspace.execution_state_store(),
                    catalog=self.catalog,
                )
            except Exception as exc:
                # Best-effort. If reconciliation itself fails, log and
                # move on — the user's operation can still proceed;
                # the next acquire_leases call will retry.
                logging.getLogger("deriva_ml").warning(
                    "startup lease reconciliation failed (%s); continuing", exc,
                )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_lease_orchestrator.py -v
```

Expected: all passing.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/core/base.py tests/execution/test_lease_orchestrator.py
git commit -m "feat(core): startup lease reconciliation on DerivaML init

Online-only workspace-wide sweep of status='leasing' pending rows.
Best-effort — failures log and continue. Offline mode skips (no
server to query).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task F6: Also reconcile per-execution on `resume_execution`

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py` (extend `resume_execution`)
- Test: `tests/execution/test_execution_registry.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/execution/test_execution_registry.py`:

```python
def test_resume_execution_per_rid_lease_reconcile(test_ml, monkeypatch):
    from deriva_ml.execution.state_store import ExecutionStatus

    _insert_test_execution(test_ml.workspace, "EXE-A", ExecutionStatus.stopped)

    calls: list[str | None] = []

    def _spy(*, store, catalog, execution_rid):
        calls.append(execution_rid)

    from deriva_ml.execution import lease_orchestrator
    monkeypatch.setattr(lease_orchestrator, "reconcile_pending_leases", _spy)

    test_ml.resume_execution("EXE-A")
    assert "EXE-A" in calls  # scoped reconciliation for this execution
```

- [ ] **Step 2: Run test to verify it fails**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_registry.py -v -k per_rid_lease
```

Expected: FAIL.

- [ ] **Step 3: Wire the scoped reconcile call**

In `src/deriva_ml/core/mixins/execution.py`, in `resume_execution` (from
Task D4), add after the `flush_pending_sync` / `reconcile_with_catalog`
calls but BEFORE returning the Execution:

```python
    if self._mode is ConnectionMode.online:
        from deriva_ml.execution.lease_orchestrator import reconcile_pending_leases
        # Scoped reconciliation. Cheaper than the workspace-wide sweep
        # we did at DerivaML init, and guarantees this specific
        # execution's pending rows are consistent before we hand out
        # the Execution object.
        try:
            reconcile_pending_leases(
                store=store, catalog=self.catalog,
                execution_rid=execution_rid,
            )
        except Exception as exc:
            logging.getLogger("deriva_ml").warning(
                "per-execution lease reconciliation failed for %s (%s); continuing",
                execution_rid, exc,
            )
```

- [ ] **Step 4: Run test**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_execution_registry.py -v
```

Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/core/mixins/execution.py tests/execution/test_execution_registry.py
git commit -m "feat(execution): resume_execution reconciles its own lease state

Per-execution lease reconcile runs on resume_execution. Belt-and-
suspenders on top of the workspace-wide sweep at DerivaML init —
guarantees this specific execution is consistent before the
Execution is handed to the caller.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

*(End of Task Group F — RID leasing, two-phase protocol, crash recovery.)*

---


