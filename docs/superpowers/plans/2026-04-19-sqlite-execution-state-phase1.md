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


