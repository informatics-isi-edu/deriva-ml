# SQLite-backed execution state: registry, pending rows, and stage-and-upload

**Status:** Draft, pending review (revision 6 — API consistency pass).
**Date:** 2026-04-18.
**Scope:** Design spec. Implementation plan follows separately.

## 0. Design guideline — friction in service of reproducibility

Before any specific design choice, the governing principle:

> When ergonomics conflict with reproducibility, robustness, or provenance,
> reproducibility wins. Small amounts of user friction — declaring a metric
> before recording it, naming a dataset version explicitly, explicitly
> uploading at a moment the user chooses — are acceptable costs when they
> produce a catalog record that is unambiguous, queryable, and
> reconstructable after the fact. DerivaML is not optimizing for "fastest
> to first result"; it is optimizing for "every result is a durable,
> reviewable artifact."

Concretely, this guideline means:

- Controlled vocabularies over free-form strings, even at the cost of a
  declaration step (features, metrics, params, asset types, dataset
  types).
- Explicit declaration over auto-creation, when auto-creation would admit
  typos into the provenance record.
- Explicit state transitions over implicit ones (no auto-upload on
  context-manager exit; the user decides when to commit results to the
  catalog).
- Version pinning over "latest" defaults, when ambiguity would break
  reproducibility of a past result.
- Named entities with provenance (datasets with RIDs, executions with
  workflow FKs) over anonymous data.

When proposing an ergonomic convenience, the check is: *does this erode
the catalog's guarantee that any past result can be reconstructed and
reviewed?* If yes, the convenience is rejected or modified until it
doesn't.

## 1. Goals and scope

### 1.1 Goals

Make the DerivaML workspace the authoritative local source of truth for
execution state — enabling robust offline workflows where scripts split
across processes, machines, and network conditions still reach a
correct upload.

Four capabilities, one coherent design:

1. **Execution registry in SQLite.** The workspace keeps a table of
   known-local executions. Enumeration (`list_executions`,
   `find_incomplete_executions`) and resumption
   (`resume_execution(rid)`) read from SQLite, not from scanning the
   filesystem tree. Status transitions are atomic SQLite writes with
   catalog sync in online mode.

2. **Pending-row staging in SQLite.** New rows destined for the
   catalog — both asset rows (file + metadata) and plain rows —
   accumulate in a per-execution pending-rows table until drained.
   RID leasing happens against `public:ERMrest_RID_Lease` so every
   pending row carries its final RID before upload.

3. **One write verb, two modes.** `handle.insert(records)` is the
   single write method. In online mode, rows reach the catalog by the
   time `.insert()` returns. In offline mode, rows stage and wait for
   upload. The user's code is identical in both modes; mode only
   changes timing.

4. **Upload as an async, restartable, bandwidth-aware operation.**
   Upload is a first-class long-running operation, not a terminal step.
   Scoped forms (`exe.upload_outputs()`, `record.upload_outputs()`),
   workspace-wide form (`ml.upload_pending(...)`), non-blocking form
   (`ml.start_upload(...)` → `UploadJob`), and a `deriva-ml upload`
   CLI all share one engine. Resumability is provided by the
   combination of SQLite queue state and deriva-py's per-chunk
   upload-state file.

Design principle: the code to write a row is the same whether the user
is connected or offline, and the same whether they're in the create
script, a compute-heavy middle script, or the final upload script.
SQLite carries state across processes.

### 1.2 Non-goals

- **Updates or deletes.** Insert-only. Modifying existing catalog rows
  or deleting them is a separate future problem.
- **Creating new tables or columns.** Target table must already exist
  in the catalog schema. `exe.table(name)` / `ml.table(name)` raises
  `DerivaMLTableNotFound` at call time if absent.
- **Full slice upload-back.** The broader "sync arbitrary local
  changes" problem remains deferred.
- **Filesystem watching.** Directory registration doesn't watch for
  file arrivals; files are surfaced via explicit `register(file)` or
  scan-at-upload.
- **Auto-detecting connection state.** Mode is explicit
  (`ConnectionMode.online` vs `ConnectionMode.offline`). The library
  does not silently defer writes on server timeout.
- **Cross-workspace sharing.** Executions are scoped to the workspace
  they were created in; the user copying a working directory between
  machines is their concern to handle.
- **Separate upload provenance.** Upload events are not recorded as a
  separate catalog entity. Evidence that upload happened manifests via
  Execution.status transitions and row creation timestamps already on
  every Deriva row.

### 1.3 Compatibility and deprecation posture

Per the project-wide deprecation policy, this refactor uses **hard
cutovers** for renamed / restructured APIs. No shim layer. Breaking
changes are enumerated in `CHANGELOG.md` under "Breaking changes" with
replacement pointers.

Renames introduced by this spec (all hard cutovers):

| Removed | Replaced by |
|---|---|
| `ml.restore_execution(rid)` | `ml.resume_execution(rid)` |
| `exe.upload_execution_outputs(...)` | `exe.upload_outputs(...)` |
| `exe.retry_failed()` | `exe.upload_outputs(retry_failed=True)` |
| `Execution.list_nested_executions()` | `ml.list_nested_executions(of=exe_rid)` |
| `Execution.find_ancestors()` | `ml.find_execution_ancestors(of=exe_rid)` |

No migration code. Clean slate — no in-flight `ManifestStore` entries
to preserve.

## 2. Architecture

### 2.1 Online mode vs offline mode

Connection mode is an enumeration:

```python
from deriva_ml import ConnectionMode

DerivaML(
    ...,
    mode: ConnectionMode | str = ConnectionMode.online,
)
# Accepts either the enum or the string literals "online" / "offline";
# strings are coerced. Default: ConnectionMode.online.
```

**Online mode (default).** Plain-row writes go live to the catalog by
the time `.insert()` returns — the library drains the pending entry
inline after staging. Asset-row writes still stage and wait for upload
(Hatrac is a two-phase pipeline; draining per file would defeat
batching). Execution status transitions also sync to the catalog's
Execution row atomically with the SQLite transition.

**Offline mode.** Every write stages into the workspace SQLite and
stays there until an upload operation drains it. The server is only
contacted for RID leases and final uploads. Status transitions update
SQLite only, marked with `sync_pending=True`; the next time the
workspace runs online, pending syncs flush (see §2.2).

**Creating executions requires online mode.** An Execution row has to
be created on the server (the RID is server-assigned), which can't
happen offline. The supported split-script workflow is:

1. Script 1 (online): `create_execution` → execution RID assigned,
   registry row written to SQLite, configuration persisted.
2. Scripts 2..N (any mode): `resume_execution(rid)` re-hydrates from
   SQLite; writes stage per the current mode.
3. Final script (online): `resume_execution(rid)` +
   `exe.upload_outputs()` drains everything.

Any `create_execution` call while `mode=ConnectionMode.offline` raises
`DerivaMLOfflineError` with a clear message.

### 2.2 Execution state machine with catalog sync

Execution lifecycle and status live in a dedicated state-machine
module (`deriva_ml/execution/state_machine.py`). SQLite is the
authoritative local state; the catalog is synced in online mode and
eventually-consistent in offline mode.

**States:**

```
created → running → {stopped, failed} → {pending_upload → {uploaded, failed}}
                                      ↘ aborted
```

**Sync rules:**

- **Online mode.** Every status transition is a single logical
  operation: update SQLite `executions.status` and PUT the catalog's
  Execution row in the same transition path. If the catalog PUT fails
  (network blip, server down), the SQLite transition still commits
  with `sync_pending=1` set. The next time a state transition or
  reconciliation runs successfully, pending syncs flush first.

- **Offline mode.** SQLite transitions commit with `sync_pending=1`
  unconditionally. The catalog is never contacted.

- **Just-in-time reconciliation.** When a workspace opens (or
  `resume_execution(rid)` is called), the library checks for
  disagreement between SQLite and catalog Execution state *only for
  the execution in question*, not workspace-wide. This keeps startup
  fast and surfaces disagreements at the moment the user is about to
  act on them.

**Disagreement handling** (six cases — rules are explicit, user is
prompted on unexpected mismatch):

| SQLite says | Catalog says | Interpretation | Action |
|---|---|---|---|
| `running` | `aborted` | Externally aborted | Set SQLite `aborted`, raise on resume |
| `pending_upload` | `uploaded` | Uploaded by another process | Set SQLite `uploaded`, clear pending rows |
| `running` | `failed` | Externally failed | Set SQLite `failed`, retain error from catalog |
| `stopped` | `running` | Stale catalog (pre-crash) | Set catalog `stopped` (sync forward) |
| `stopped` | (no row) | Orphaned | Raise `DerivaMLStateInconsistency` with guidance |
| anything else | anything else | Unexpected | Raise `DerivaMLStateInconsistency`; user intervention |

Scope for R1.1 implementation: 3–4 days.

### 2.3 Execution fields as SQLite read-through

The `Execution` object's user-visible lifecycle fields — `status`,
`start_time`, `stop_time`, `error` — become **read-through properties**
backed by SQLite. Every read issues a SELECT; no in-memory caching.
This keeps multiple processes holding the same execution RID consistent
with each other.

```python
class Execution:
    @property
    def status(self) -> ExecutionStatus: ...  # reads SQLite
    @property
    def start_time(self) -> datetime | None: ...
    @property
    def stop_time(self) -> datetime | None: ...
    @property
    def error(self) -> str | None: ...
```

No setter surface: state transitions go through the state machine.

Scope for R1.2 implementation: 2–4 hours.

### 2.4 Workspace SQLite as authoritative state

Everything that matters for resuming an execution lives in SQLite:

- Execution registry and configuration
- Pending-row entries (asset and plain)
- Directory-rule entries
- Lease tokens for crash-safe RID allocation
- Status for every row and every execution
- `sync_pending` flags for catalog reconciliation

The filesystem tree (`{working_dir}/deriva-ml/execution/{rid}/`) is a
**derived artifact**: asset files live there for the uploader to
consume, staging CSVs are materialized there at drain time, but the
SQLite workspace is the ground truth. If the filesystem is corrupted
or edited externally, SQLite is consulted, not the tree.

This inverts today's model, where file-tree scanning (`os.listdir`)
was the enumeration mechanism.

### 2.5 SQLite schema

Three tables in the workspace SQLite:

#### 2.5.1 `executions`

One row per known-local execution.

| Column | Purpose |
|---|---|
| `rid` | PK, server-assigned Execution RID |
| `workflow_rid` | FK concept (server RID string); null if not set |
| `description` | from config |
| `config_json` | full ExecutionConfiguration, serialized |
| `status` | `created \| running \| stopped \| failed \| pending_upload \| uploaded \| aborted` |
| `mode` | `online \| offline` — mode the execution was last active under |
| `working_dir_rel` | relative path to execution root |
| `start_time` / `stop_time` | lifecycle timestamps |
| `last_activity` | updated on every pending-row insert/upload |
| `error` | last error message if `status='failed'` |
| `sync_pending` | bool; catalog hasn't caught up with SQLite |
| `created_at` | when the local registry knew about it |

Indexes: `(status)`, `(workflow_rid)`, `(last_activity)`,
`(sync_pending) WHERE sync_pending=1`.

#### 2.5.2 `pending_rows`

One row per pending catalog-row insert, keyed to an execution.

| Column | Purpose |
|---|---|
| `id` | internal PK |
| `execution_rid` | FK to executions.rid |
| `key` | auto-hash; directory entries derive from `rule_id + filename` |
| `target_schema` / `target_table` | catalog target |
| `rid` | leased RID, null until leased |
| `lease_token` | idempotency token for crash-safe leasing |
| `metadata_json` | column values (serialized via `AssetManifest._json_default`) |
| `asset_file_path` | local file path, null for plain rows |
| `asset_types_json` | vocabulary terms, null for plain rows |
| `description` | optional |
| `status` | `staged \| leasing \| leased \| uploading \| uploaded \| failed` |
| `error` | set on failure |
| `created_at` / `leased_at` / `uploaded_at` | timestamps |
| `rule_id` | FK to directory_rules.id, null if not from a directory |

Indexes: `(execution_rid, status)`, `(execution_rid, target_table)`.

#### 2.5.3 `directory_rules`

One row per registered asset directory.

| Column | Purpose |
|---|---|
| `id` | internal PK |
| `execution_rid` | FK to executions.rid |
| `target_schema` / `target_table` | target |
| `source_dir` | absolute path |
| `glob` / `recurse` / `copy_files` | selection params |
| `asset_types_json` | applied to every registered file |
| `status` | `active \| closed` |
| `created_at` | timestamp |

### 2.6 RID leasing (lazy, batched, crash-safe)

Pending rows eventually carry a real, server-assigned RID from
`public:ERMrest_RID_Lease`. Leases are acquired **lazily** — deferred
to the earliest of:

- An explicit `handle.rid` read.
- The upload drain starting.

Until leased, `handle.rid == None` and `status='staged'`.

**Two-phase acquisition for crash safety.** Before the lease POST,
the library writes `status='leasing'` with a UUID `lease_token`. On
success, the response RIDs are written back. On workspace startup,
entries still in `status='leasing'` trigger reconciliation: query
`ERMrest_RID_Lease` by token; adopt if the lease exists, revert to
`staged` if it doesn't.

Lease POSTs are batched and chunked (`PENDING_ROWS_LEASE_CHUNK`,
default 500).

### 2.7 Public API — `DerivaML` top level

```python
class DerivaML:
    def __init__(self, ..., mode: ConnectionMode | str = ConnectionMode.online):
        ...

    # Connection / model
    @property
    def mode(self) -> ConnectionMode: ...

    def table(self, name: str, *, schema: str | None = None) -> TableHandle | AssetTableHandle:
        """Schema-introspection sibling of exe.table(). Same return
        types. Writes on a handle returned by ml.table() raise
        DerivaMLNoExecutionContext — there's no execution to scope
        them to. Supports: handle.record_class(), handle.name,
        handle.schema, and asset-type introspection on
        AssetTableHandle."""

    # Execution construction — kwargs form or config-object form (both supported)
    def create_execution(
        self,
        config: ExecutionConfiguration | None = None,
        *,
        datasets: list[DatasetSpecConfig | str] | None = None,
        assets: list[AssetRIDConfig | str] | None = None,
        workflow: Workflow | str | None = None,
        description: str | None = None,
    ) -> Execution:
        """Online-only. Kwargs form builds an ExecutionConfiguration
        internally; string shorthand 'RID@version' is accepted for
        datasets. Mixing `config` with kwargs raises TypeError.

        In offline mode raises DerivaMLOfflineError."""

    # Registry
    def list_executions(
        self, *,
        status: ExecutionStatus | list[ExecutionStatus] | None = None,
        workflow_rid: str | None = None,
        mode: ConnectionMode | None = None,
        since: datetime | None = None,
    ) -> list[ExecutionRecord]: ...

    def find_incomplete_executions(self) -> list[ExecutionRecord]: ...

    def resume_execution(self, execution_rid: str) -> Execution:
        """Re-hydrate an Execution from SQLite. Works in both modes;
        the execution's recorded mode is independent of the current
        DerivaML instance's mode. Runs just-in-time reconciliation
        (§2.2) for this execution before returning. Raises
        DerivaMLException if no matching registry row exists."""

    def gc_executions(
        self, *,
        older_than: timedelta | None = None,
        status: ExecutionStatus | list[ExecutionStatus] | None = None,
        delete_working_dir: bool = False,
    ) -> int: ...

    # Hierarchy queries (moved off Execution per R2.1)
    def list_nested_executions(
        self, *, of: str,                       # parent execution RID
    ) -> list[ExecutionRecord]: ...

    def find_execution_ancestors(self, *, of: str) -> list[ExecutionRecord]: ...

    # Pending / upload inspection
    def pending_summary(self) -> WorkspacePendingSummary:
        """Workspace-wide summary: per-execution pending row/file counts."""

    def list_pending_uploads(self) -> list[UploadTarget]:
        """Flat list of (execution_rid, table, row_count, file_count,
        bytes) tuples — one per (execution, table) group with pending
        items."""

    # Upload operations
    def upload_pending(
        self, *,
        execution_rids: list[str] | None = None,
        retry_failed: bool = False,
        bandwidth_limit_mbps: int | None = None,
        parallel_files: int = 4,
    ) -> UploadReport:
        """Blocking upload of pending state. `execution_rids=None`
        means all pending. Runs in-process. Idempotent — safe to
        re-run after crash; resumes via SQLite queue state and
        deriva-py's per-chunk resume file."""

    def start_upload(self, **kwargs) -> UploadJob:
        """Non-blocking form. Spawns a worker thread in the current
        process, returns an UploadJob handle. Same kwargs as
        upload_pending. The thread dies if the process exits; to
        survive process exit, use deriva-ml upload CLI from a shell."""

    def get_upload_job(self, job_id: str) -> UploadJob: ...
    def list_upload_jobs(self, *, status: str | None = None) -> list[UploadJob]: ...
```

### 2.8 Public API — Execution

```python
class Execution:
    # Identity / configuration
    @property
    def rid(self) -> str: ...
    @property
    def configuration(self) -> ExecutionConfiguration: ...
    @property
    def mode(self) -> ConnectionMode: ...

    # Lifecycle fields (SQLite read-through, §2.3)
    @property
    def status(self) -> ExecutionStatus: ...
    @property
    def start_time(self) -> datetime | None: ...
    @property
    def stop_time(self) -> datetime | None: ...
    @property
    def error(self) -> str | None: ...

    # Context manager for the running phase
    def execute(self) -> AbstractContextManager[Execution]:
        """Enter: status → running (synced to catalog online). Exit OK:
        status → stopped. Exit with exception: status → failed.
        On exit, emits INFO log with pending_summary() contents if
        anything is staged. Does NOT auto-upload (per R6.3, in service
        of restart-safety and offline mode)."""

    def abort(self) -> None: ...

    # Row-level writes
    def table(self, name: str, *, schema: str | None = None) -> TableHandle | AssetTableHandle:
        """Execution-scoped handle. Same class as ml.table() returns,
        but writes are scoped to this execution and are permitted."""

    # Pending / upload inspection & drain
    def pending_summary(self) -> PendingSummary: ...

    def upload_outputs(
        self, *,
        retry_failed: bool = False,
        bandwidth_limit_mbps: int | None = None,
        parallel_files: int = 4,
    ) -> UploadReport:
        """Sugar for ml.upload_pending(execution_rids=[self.rid], ...)."""

    # __repr__ includes pending counts (per R6.3)
    def __repr__(self) -> str:
        # "<Execution EXE-A status=stopped pending=15rows/2files>"
        ...
```

### 2.9 Public API — `ExecutionRecord`

`ExecutionRecord` is the restored-from-SQLite handle — a frozen
dataclass with the registry columns plus convenience counts derived
from `pending_rows`:

```python
@dataclass(frozen=True)
class ExecutionRecord:
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
    # Derived
    pending_rows: int
    failed_rows: int
    pending_files: int
    failed_files: int

    def pending_summary(self) -> PendingSummary: ...
    def upload_outputs(self, **kwargs) -> UploadReport:
        """Sugar for ml.upload_pending(execution_rids=[self.rid], ...)."""
```

### 2.10 Public API — Table handles (row-level writes)

```python
ml.table(name, schema=None) -> TableHandle | AssetTableHandle       # no execution scope
exe.table(name, schema=None) -> TableHandle | AssetTableHandle      # execution-scoped
```

Same class hierarchy for both entry points. The only difference: handles
returned by `ml.table(...)` have no execution to bind writes to, so
`.insert(...)` and asset-file methods raise `DerivaMLNoExecutionContext`
with a message pointing at `exe.table(...)`. Read-only methods
(`record_class()`, `name`, `schema`, asset-type introspection) work on
both.

Returns `AssetTableHandle` if the table is an asset table,
`TableHandle` otherwise. Asset-specific methods appear only on
`AssetTableHandle`, visible via IDE tab-completion.

Ambiguous `table_name` without `schema=` raises
`DerivaMLTableNotFound` listing candidates.

#### 2.10.1 `TableHandle` surface

```python
class TableHandle:
    name: str
    schema: str

    def record_class(self, *, include_system_columns: bool = False) -> type[RowRecord]:
        """Dynamically-generated Pydantic model for this table. Fields
        from non-nullable columns are required; nullable are Optional.
        extra='forbid' catches typos."""

    def insert(
        self,
        records: RowRecord | dict | Iterable[RowRecord | dict] | "pd.DataFrame",
        *,
        description: str | None = None,
        chunk_size: int = 1000,
        progress: Callable[[int, int], None] | None = None,
    ) -> PendingRow | list[PendingRow]:
        """Write rows. Raises DerivaMLNoExecutionContext on handles
        obtained via ml.table(). On exe.table() handles:
        - Online mode: rows reach the catalog by the time .insert()
          returns.
        - Offline mode: rows stage in workspace SQLite; upload drains
          them at exe.upload_outputs().
        Accepts scalar / Iterable / DataFrame; returns scalar or list
        accordingly. Validation via self.record_class() at call time;
        invalid records raise regardless of mode."""

    def pending(self, *, status: str | list[str] | None = None) -> list[PendingRow]: ...
    def discard_pending(self, *, rows: Iterable[PendingRow] | None = None,
                        status: str | list[str] | None = None) -> int: ...
```

#### 2.10.2 `AssetTableHandle` surface (extends `TableHandle`)

```python
class AssetTableHandle(TableHandle):

    def asset_file(
        self,
        file: str | Path,
        metadata: RowRecord | dict | None = None,
        *,
        asset_types: list[str] | str | None = None,
        copy_file: bool = False,
        rename_file: str | None = None,
        description: str | None = None,
        **kwargs,
    ) -> AssetFilePath:
        """Register a single file for upload. ALWAYS DEFERRED (even in
        online mode) because Hatrac + row insert is two-phase. Validates
        metadata at call time. Returns AssetFilePath bound to the
        pending entry."""

    def asset_directory(
        self,
        source_dir: str | Path,
        *,
        asset_types: list[str] | str | None = None,
        glob: str = "*",
        recurse: bool = False,
        copy_files: bool = False,
    ) -> AssetDirectoryHandle: ...
```

#### 2.10.3 `AssetDirectoryHandle`

```python
class AssetDirectoryHandle:
    path: Path                    # source_dir
    rule_id: int
    table: AssetTableHandle

    def register(self, file: str | Path) -> AssetFilePath: ...
    def scan(self) -> list[AssetFilePath]: ...
    def pending(self) -> list[AssetFilePath]: ...

    def set_metadata(
        self, source: "pd.DataFrame", *, filename_col: str = "Filename",
    ) -> None: ...

    def close(self) -> None: ...
```

Deleted files between scans are not auto-removed; upload surfaces them
as per-row failures.

#### 2.10.4 Handle read properties

Both `PendingRow` and `AssetFilePath` expose:

```python
handle.rid         # leased RID; None until first read (lazy lease)
handle.status      # staged | leasing | leased | uploading | uploaded | failed
handle.metadata    # dict; setter writes through to SQLite
handle.error       # error message if status == 'failed'
```

Handles are thin stateless views over SQLite — every property read
issues a SELECT, every write commits an UPDATE. Consistent across
multiple handles pointing at the same entry. Bulk operations use the
batch setters.

### 2.11 `PendingSummary` and upload operations

#### 2.11.1 PendingSummary dataclasses

```python
@dataclass(frozen=True)
class PendingRowCount:
    table: str                    # "deriva-ml:Image"
    pending: int
    failed: int
    uploaded: int

@dataclass(frozen=True)
class PendingAssetCount:
    table: str
    pending_files: int
    failed_files: int
    uploaded_files: int
    total_bytes_pending: int

@dataclass(frozen=True)
class PendingSummary:
    execution_rid: str
    rows: list[PendingRowCount]
    assets: list[PendingAssetCount]
    diagnostics: list[str]

    @property
    def has_pending(self) -> bool: ...
    @property
    def total_pending_rows(self) -> int: ...
    @property
    def total_pending_files(self) -> int: ...

    def render(self) -> str: ...     # human-readable multi-line

@dataclass(frozen=True)
class WorkspacePendingSummary:
    per_execution: list[PendingSummary]
    @property
    def total_executions_with_pending(self) -> int: ...
    def render(self) -> str: ...
```

#### 2.11.2 Upload engine

All upload surfaces (`exe.upload_outputs`, `ml.upload_pending`,
`ml.start_upload`, CLI) drive one internal engine:

```
_upload_engine(ml, execution_rids, retry_failed, bandwidth_mbps, parallel):
  1. Enumerate pending items from SQLite for the selected executions:
     - pending_rows in status staged/leasing/leased/failed-if-retry
     - asset files in status staged/uploading/failed-if-retry
  2. For each table, final scan of AssetDirectoryHandle rules.
  3. Re-validate each pending row's metadata against current catalog
     schema (catches drift between stage and upload).
  4. Batch-lease RIDs for any status='staged' rows.
  5. Build inter-table FK DAG; topological sort. Cycle →
     DerivaMLCycleError.
  6. For each topological level, in parallel (bounded by
     parallel_files):
     - For asset files: call deriva-py uploader (idempotent, resumable
       at chunk granularity).
     - For plain rows: materialize staging CSV, invoke upload pipeline.
     - Update SQLite row status per-item with fsync.
  7. On first level with failures, drain aborts after recording all
     failures in that level. Execution status → uploaded (all
     succeeded) or failed (any failed).
```

**Resumability.** Deriva-py's uploader writes per-chunk state to
`.deriva-upload-state-{hostname}.json` with fsync; killed mid-chunk,
a re-run resumes from the next chunk. Server-side Hatrac rejects
duplicate content by hash, so any item the client thinks needs upload
but which is already there is a cheap HEAD + skip. The engine combines
these with SQLite's queue-level state to make the whole pipeline
idempotent: re-running `upload_pending` after any kind of crash
converges to the same final state without duplication.

#### 2.11.3 UploadJob (non-blocking form)

```python
class UploadJob:
    id: str
    status: Literal["running", "paused", "completed", "failed", "cancelled"]

    def progress(self) -> UploadProgress: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def cancel(self) -> None: ...     # finishes current in-flight, stops new
    def wait(self, timeout: float | None = None) -> UploadReport: ...

@dataclass(frozen=True)
class UploadProgress:
    total_rows: int
    uploaded_rows: int
    total_bytes: int
    uploaded_bytes: int
    current_mbps: float
    eta_seconds: float | None
    current_file: str | None
    failures: list[str]
```

`UploadJob` state lives in SQLite (`upload_jobs` table); the worker
thread updates it. If the process dies, state survives and
`ml.get_upload_job(id).resume()` restarts work from where the queue
left off.

#### 2.11.4 CLI

```
deriva-ml upload [--execution RID]... [--retry-failed]
                 [--bandwidth-mbps N] [--parallel N]
                 [--host HOSTNAME] [--catalog CATALOG_ID]
```

Wraps `upload_pending`. Ships as a console-script entry point.
Supports shell backgrounding (`nohup deriva-ml upload &`,
launchd/systemd units) for operator-driven uploads on a
network-managed schedule.

### 2.12 Context-manager exit behavior (R6.3)

On `with exe.execute() as e:` exit:

1. Update execution status (`stopped` on clean exit, `failed` on
   exception).
2. Compute `pending_summary()`; if `has_pending`, emit INFO log:

   ```
   INFO [Execution EXE-A] exited with pending uploads:
     rows:    Image (12 pending, 0 failed)
              Subject (3 pending, 0 failed)
     assets:  Execution_Metadata (2 files, 4.2MB pending)
     Call exe.upload_outputs() to flush, or exe.pending_summary() for details.
   ```

3. `Execution.__repr__` reflects pending counts.

**No auto-upload on exit.** Rationale per guideline §0: auto-upload
at exit conflicts with two important properties:
- **Offline mode** — the user may not be connected.
- **Restart after failure** — an auto-upload that fails at exit
  destroys the ability to retry from a clean state, and blocks exit
  on a long-running operation.

An opt-in `create_execution(upload_on_exit=True)` is deferred to a
future version if usage patterns demand it. The default is explicit.

## 3. Walkthrough — three-script workflow

The motivating workflow: a user connects from home, creates an
execution, downloads a dataset bag. Next day on a laptop with no
connection, they run the model against the bag and write
predictions + asset masks locally. The following day, reconnecting,
they upload.

**Script 1 — create the execution (online):**

```python
ml = DerivaML(hostname="example.org", catalog_id="42")    # default: online
exe = ml.create_execution(
    datasets=["1-XYZ@1.0.0"],                              # kwargs form
    workflow=my_workflow,
    description="Segmentation run 2026-04-18",
)

with exe.execute() as e:
    bag = e.datasets["1-XYZ"].bag          # download to workspace
# SQLite executions row: rid=5-ABC, status='stopped', mode='online',
#   config_json=<serialized>, working_dir_rel='execution/5-ABC'.
# Catalog Execution row synced: status='stopped'.
```

**Script 2 — offline computation:**

```python
ml = DerivaML(hostname="example.org", catalog_id="42",
              mode=ConnectionMode.offline)

for record in ml.find_incomplete_executions():
    print(record.rid, record.status, record.description)
# Prints: 5-ABC  stopped  "Segmentation run 2026-04-18"

exe = ml.resume_execution("5-ABC")   # from SQLite, no server contact

with exe.execute() as e:
    bag = e.datasets["1-XYZ"].bag

    for image_rid in bag.list_dataset_members()["Image"]:
        img = bag.get_row("Image", image_rid)
        mask_arr, conf = my_model.predict(img.filename)

        mask_path = Path(exe.working_dir) / f"mask_{image_rid}.png"
        save_png(mask_arr, mask_path)
        mask = exe.table("Segmentation_Mask").asset_file(
            mask_path, Image=image_rid, Model_Version="v1.2",
        )

        exe.table("Prediction").insert({
            "Image": image_rid,
            "Mask": mask.rid,
            "Confidence": conf,
        })
# Exit INFO log:
#   [Execution 5-ABC] exited with pending uploads:
#     rows:    Prediction (N pending, 0 failed)
#     assets:  Segmentation_Mask (N files, ~M MB pending)
# SQLite state after script 2:
#   executions.5-ABC.status = 'stopped', sync_pending=1
#   pending_rows: 2N rows, status='staged' / 'leased'
```

**Script 3 — upload (online):**

```python
ml = DerivaML(hostname="example.org", catalog_id="42")   # online

print(ml.pending_summary().render())
# Workspace pending summary:
#   Execution 5-ABC (stopped):
#     rows:    Prediction (N pending)
#     assets:  Segmentation_Mask (N files, ~M MB pending)

ml.upload_pending(execution_rids=["5-ABC"])
# Drain runs; status transitions to 'uploaded'.
```

Or, operator-driven from a shell (say, an overnight schedule):

```
$ deriva-ml upload --host example.org --catalog 42 --execution 5-ABC \
                   --bandwidth-mbps 50 --parallel 4
```

Same engine; different driver.

## 4. Algorithms

### 4.1 Handle construction

```
ml.table(name, schema=None) / exe.table(name, schema=None):
  1. Resolve via model.name_to_table(name, schema=schema). Raise
     DerivaMLTableNotFound if missing or ambiguous-without-schema.
  2. If model.is_asset(table): return AssetTableHandle(self, table).
     Else:                      return TableHandle(self, table).
  Handles hold the owner (ml or exe) and table reference. No cached state.
  ml-bound handles set .execution = None; write methods check and raise
  DerivaMLNoExecutionContext.
```

### 4.2 `TableHandle.insert(records)`

```
1. If self.execution is None: raise DerivaMLNoExecutionContext.
2. Normalize records:
   - DataFrame → list of dicts.
   - Scalar → list of one (flag: scalar_input).
   - Iterable → materialize in chunks.
3. For each chunk:
   a. For each rec:
      - RowRecord: pass through.
      - dict: rec = self.record_class()(**rec). Raises on failure.
   b. Generate lease_token per row.
   c. BEGIN TRANSACTION.
   d. Insert N pending_rows: execution_rid, target_schema, target_table,
      status='staged', rid=NULL.
   e. COMMIT.
   f. progress(done, total) if provided.
4. Update executions.last_activity = now().
5. If self.execution.mode == ConnectionMode.online:
   _drain_pending(execution_rid, entry_ids=[...]).
6. Return scalar handle or list.
```

### 4.3 `AssetTableHandle.asset_file(file, ...)`

```
1. If self.execution is None: raise DerivaMLNoExecutionContext.
2. Validate file, determine staging path.
3. Symlink/copy file to assets/{AssetTable}/.
4. Construct metadata record via self.record_class()(**{metadata, **kwargs}).
5. Generate lease_token.
6. Write pending_rows with asset_file_path set, status='staged'.
7. Update executions.last_activity.
8. Return AssetFilePath bound to the entry. No drain.
```

### 4.4 Lazy `.rid` materialization

```
1. If entry.rid is not null, return it.
2. _acquire_leases(including_entries=[...]):
   a. Find all entries in status='staged' (coalesce pending reads).
   b. SET status='leasing' for them, commit.
   c. POST /entity/public:ERMrest_RID_Lease with [{"ID": token}, ...].
   d. UPDATE pending_rows SET rid=<assigned>, status='leased' by token.
3. Return entry.rid.
```

### 4.5 Workspace startup lease reconciliation

```
1. Query pending_rows WHERE status='leasing'.
2. For each group (batched by execution):
   a. Query ERMrest_RID_Lease by lease_tokens.
   b. Found → rid=<found>, status='leased'.
   c. Not found → status='staged' (POST never landed).
```

### 4.6 `ml.resume_execution(rid)`

```
1. Query executions WHERE rid=<rid>.
2. If not found: raise DerivaMLException("Execution {rid} not in workspace").
3. Just-in-time reconciliation:
   a. If executions.sync_pending=1 and mode=online:
      Flush SQLite state to catalog Execution row. Clear sync_pending.
   b. Else if mode=online:
      GET catalog Execution row. Compare status with SQLite. Apply
      disagreement rules (§2.2). Raise on unexpected mismatch.
4. Load configuration from executions.config_json.
5. Construct Execution object bound to self.
6. Return Execution.
```

### 4.7 Execution lifecycle transitions

Every transition is a single call to `state_machine.transition(rid,
target_status, **metadata)` which:
1. Opens SQLite transaction.
2. Validates current→target is legal.
3. Updates SQLite row (status, timestamp, error, etc.).
4. If online mode: PUTs catalog Execution row; on PUT failure sets
   sync_pending=1, commits SQLite anyway.
5. If offline mode: sets sync_pending=1, commits SQLite.
6. Commits transaction.

```
create_execution:
  1. Require online mode.
  2. POST Execution row to catalog → rid=<assigned>.
  3. Serialize ExecutionConfiguration to config_json.
  4. INSERT executions row: rid, status='created', mode, config_json,
     start_time=NULL, last_activity=now(), sync_pending=0.
  5. Return Execution.

.execute() context enter:
  transition(rid, 'running', start_time=now())

.execute() context exit OK:
  transition(rid, 'stopped', stop_time=now())
  + emit exit log; update __repr__ via pending_summary.

.execute() context exit w/ exception:
  transition(rid, 'failed', stop_time=now(), error=<msg>)

upload_outputs start:
  transition(rid, 'pending_upload')

upload_outputs OK:
  transition(rid, 'uploaded')

upload_outputs w/ failures:
  transition(rid, 'failed', error=<msg>)

exe.abort():
  transition(rid, 'aborted')
```

## 5. Testing

### 5.1 Unit tests (no live catalog)

Registry + state machine:
- `create_execution` writes SQLite row; catalog PUT mocked.
- `resume_execution(rid)` reads from SQLite; works without server.
- `resume_execution(missing_rid)` raises.
- `list_executions()` with each filter combination.
- `find_incomplete_executions()` returns expected status set.
- `gc_executions(older_than=...)` removes matching rows.
- Each lifecycle transition produces correct status sequence.
- State-machine validates current→target legality; illegal raises.
- `sync_pending` set on PUT failure; cleared on successful flush.
- Disagreement rules: each of the six cases produces expected outcome.
- Read-through properties: status/start_time/stop_time/error reflect
  SQLite mutations by another process.

Connection mode:
- `ConnectionMode.online` (default), `ConnectionMode.offline` accepted.
- String literals "online"/"offline" coerced.
- Invalid values raise pydantic validation error.

Kwargs form of create_execution:
- `create_execution(config=X)` works.
- `create_execution(datasets=[...], workflow=...)` works.
- Mixing `config` with other kwargs raises TypeError.
- String shorthand `"RID@version"` coerces to DatasetSpecConfig.

Handle dispatch:
- `exe.table(plain)` → `TableHandle`; no asset methods.
- `exe.table(asset)` → `AssetTableHandle`; asset methods present.
- `ml.table(plain).insert(...)` → `DerivaMLNoExecutionContext`.
- `ml.table(asset).asset_file(...)` → `DerivaMLNoExecutionContext`.
- `ml.table(plain).record_class()` works.
- Ambiguous name without schema raises.

Insert:
- `insert(dict)` validates and raises on bad field.
- `insert(record)` passes through.
- `insert(df)` converts correctly.
- Scalar → scalar; Iterable / DataFrame → list.
- Empty input is a no-op.
- Large batch is chunked with progress callbacks.
- Offline mode: `insert(row)` returns with `status='staged'`.
- Online mode: `insert(row)` drains inline; `status='uploaded'`.

Lazy lease:
- `.rid` read triggers batched POST; already-leased entries skipped.
- Two-phase: `status='leasing'` written before POST; POST failure
  reverts to `'staged'`.
- Startup reconciliation: POST success + simulated crash → lease
  adopted.

Directory handle:
- `asset_directory(dir).register(file)` eager.
- `scan()` idempotent.
- `set_metadata(df)` validates every resolved value.
- Deleted file → not auto-removed; upload surfaces failure.

PendingSummary & exit behavior:
- `pending_summary()` correct counts across tables.
- Context-manager exit logs INFO when `has_pending`.
- `__repr__` includes pending counts.
- No auto-upload on exit (regression guard).

Upload engine:
- `upload_pending(execution_rids=[...])` drives engine; correct final
  status.
- `retry_failed=True` picks up `status='failed'` entries.
- Crash mid-drain: re-run converges (SQLite status preserved).
- Topological order respected; FK cycle raises `DerivaMLCycleError`.
- Bandwidth limit honored (rate-limited egress).

UploadJob:
- `start_upload` returns running job.
- `progress()` reports per-item counts.
- `pause()` / `resume()` / `cancel()` transitions.
- Process-exit + `get_upload_job(id).resume()` picks up.

CLI:
- `deriva-ml upload --execution RID` runs to completion.
- `--bandwidth-mbps N` respected.

### 5.2 Integration tests (live catalog)

Online mode:
- `ml.create_execution(kwargs)` writes SQLite + server.
- `exe.table("Subject").insert({...})` reaches catalog before return.
- `exe.table("Image").asset_file(...)` + `exe.upload_outputs()`:
  file in Hatrac, row in catalog.
- Status sync: each transition visible in catalog Execution row.

Offline mode:
- Same code with `mode=ConnectionMode.offline`: rows reach catalog at
  `exe.upload_outputs()`, not at `.insert()` time.
- `create_execution` with `mode=ConnectionMode.offline` raises.
- On reconnect (online resume_execution), sync_pending flushed.

Split-script workflow:
- Script A (online): create_execution kwargs form, download bag.
- Script B (offline): resume_execution(rid), insert rows, exit.
- Script C (online, separate process):
  `ml.upload_pending(execution_rids=[rid])`.
- Verify: catalog Subject rows have correct `Execution` FK / RCB;
  RID assigned in script B matches catalog.

Separate-process upload:
- Script A (online): create execution, stage rows.
- Script B (subprocess `deriva-ml upload`): drains.
- Script A sees `uploaded` status after B completes.

Resumption robustness:
- Crash mid-insert: re-open workspace → `find_incomplete_executions`
  surfaces the execution; `resume_execution` works; pending_rows
  intact.
- Crash mid-upload: `find_incomplete_executions` surfaces it;
  `upload_outputs(retry_failed=True)` completes the remaining work.
- Corrupted configuration.json on disk: `resume_execution` still
  works (reads from SQLite).
- Kill during large-file upload: resume picks up at chunk boundary
  (deriva-py state + server dedupe).

Disagreement scenarios:
- External abort of execution: SQLite says running, catalog says
  aborted → SQLite updated, resume raises.
- External upload completion: SQLite says pending_upload, catalog says
  uploaded → SQLite updated, pending rows cleared.

Edge cases:
- Inter-table FK cycle raises `DerivaMLCycleError`.
- Schema drift: add required column between stage and upload → error.
- File deleted between register and upload: per-row failure with
  clear message.

### 5.3 Regression

After hard-cutover renames, all existing tests that referenced the old
names are updated as part of the rename PR (not shim-tested).

## 6. Evolution from earlier revisions

Revision highlights:

- **Rev 2/3:** Separate `stage` and `insert` verbs. Collapsed in rev 4.
- **Rev 3:** Workspace-scoped pending rows. Dropped in rev 4;
  everything is execution-scoped.
- **Rev 4:** `dedupe_key`, `best_effort` mode, handle-as-FK coercion,
  sidecar CSV metadata. All dropped for rev 5 simplicity.
- **Rev 5:** Reached 10-concept API with single write verb.
- **Rev 6 (this revision):** API-consistency pass against the
  cross-cutting analysis:
  - `ConnectionMode` enum (stricter than string).
  - Execution state machine as a dedicated module with catalog sync
    and `sync_pending` reconciliation.
  - Execution lifecycle fields as SQLite read-through.
  - Hierarchy queries moved from `Execution` to `ml.*`.
  - `ml.table()` as schema-introspection sibling of `exe.table()`.
  - `retry_failed` merged into `upload_outputs` / `upload_pending`.
  - `create_execution` kwargs form alongside config-object form.
  - Metrics and params as first-class concepts parallel to features
    (deferred to follow-on task, §8).
  - `pending_summary` on `Execution`, `ExecutionRecord`, and
    workspace-level.
  - Upload as async-capable operation with `ml.upload_pending`,
    `ml.start_upload` / `UploadJob`, CLI, all sharing one engine.
  - Hard deprecation policy (no shims).
  - Reproducibility-over-ergonomics guideline (§0) captured.

Concept count after rev 6: **13**.

1. `DerivaML(..., mode=ConnectionMode.online|offline)` — mode at
   construction.
2. `ml.list_executions() / find_incomplete_executions() / resume_execution()` — registry.
3. `ml.create_execution(...)` — online only, kwargs or config form.
4. `ml.table(name)` — schema-introspection (read-only).
5. `exe.table(name)` — row-level entry point, execution-scoped writes.
6. `TableHandle / AssetTableHandle` — asset variant adds file methods.
7. `handle.record_class()` — typed record.
8. `handle.insert(records)` — one write verb.
9. `handle.pending() / discard_pending()` — triage.
10. `handle.asset_file() / asset_directory()` — asset-specific.
11. `exe.pending_summary() / ml.pending_summary()` — inspection.
12. `exe.upload_outputs() / ml.upload_pending() / ml.start_upload() +
    UploadJob / deriva-ml upload` — drain.
13. `ml.list_nested_executions(of=...) / find_execution_ancestors(of=...)` — hierarchy.

## 7. Known limits

- **Updates / deletes:** insert-only.
- **Intra-table FK cycles:** unsupported. Users split into batches.
- **Binary columns (`bytea`):** unsupported; `insert()` raises
  `NotImplementedError`. Asset-table route is the supported path.
- **CSV-based insert roundtrip (§2.11.2 step 6):** transitional. V2
  can swap to direct `pathBuilder` insert without public-API change.
- **Orphaned server leases:** possible only if client loses its
  workspace state after a successful lease POST. Not reconcilable
  from the client; server-side GC is an ERMrest concern.
- **Creating new executions offline:** not supported. Execution RID
  is server-assigned and requires an online `create_execution` call.
- **best_effort upload mode, dedupe_key, handle-FK coercion:**
  deferred. V2 can add these when users hit their absence.
- **Cross-machine workspace sync:** user's concern; the library does
  not replicate `workspace.db` between machines.
- **Upload on context-manager exit:** intentionally not auto; an
  opt-in `upload_on_exit=True` is deferred to future revision.

## 8. Scoped follow-on: Feature system + Metric/Param + SQLite consistency

Deferred to a dedicated design task after this spec lands.

**Scope:** Revisit the feature system, introduce `Metric` and `Param`
as first-class concepts parallel to features, and ensure consistency
of all three (features, metrics, params) against the SQLite-backed
staging layer.

**Specifically:**

### 8.1 Metric and Param as first-class concepts

Row-form complement to execution metadata files. Subject is the run
(not data rows, so not features); content is scalar and queryable
(not files or metadata blobs).

**Parallel to features** to minimize cognitive load:

| Action | Feature | Metric | Param |
|---|---|---|---|
| Declare | `ml.create_feature(target_table, name, metadata)` | `ml.create_metric(name, unit, higher_is_better, description)` | `ml.create_param(name, unit, description)` |
| Record | `exe.add_feature_value(name, target, **v)` | `exe.record_metric(name, value, step=None)` | `exe.record_param(name, value)` |
| List | `ml.list_features()` | `ml.list_metrics()` | `ml.list_params()` |
| Readback | `bag.feature_values(name)` | `exe.metrics(name=None)` | `exe.params()` |

**Schema:**
- Vocabulary tables `Metric_Name`, `Param_Name` (curated, not
  auto-populated, parallel to `Feature_Name`).
- Single shared data tables `ExecutionMetric`, `ExecutionParam` (FK
  to vocabulary; fixed scalar shape for query efficiency).
- Metric vocab carries `Unit` and `Higher_Is_Better` to enable
  unit-aware queries and automatic best-run selection.

**Typo safety:** `record_metric` on an undeclared name raises
`DerivaMLInvalidTerm` — same behavior as `add_feature_value` on an
undeclared feature. Per guideline §0: the declaration step is
acceptable friction in service of vocabulary integrity.

### 8.2 SQLite integration

All three (features, metrics, params) stage through SQLite
`pending_rows` identically. Drain through the same upload engine.
Offline-safe. Typed via `record_class()`.

### 8.3 Readback shapes

Determine long-format vs wide-format conventions for
`exe.metrics(name=None)`, cross-run comparison helpers
(`ml.compare_metrics(...)`), and ensure the `DatasetBag` readback
surface stays consistent with the live-catalog surface.

### 8.4 Task deliverables

A separate spec at `docs/superpowers/specs/YYYY-MM-DD-feature-metric-param-consistency.md`
covering:
- Full schema for `Metric_Name`, `Param_Name`, `ExecutionMetric`,
  `ExecutionParam`.
- Full API surface (create/record/list/readback for each).
- SQLite staging integration.
- Cross-concept consistency review (shared naming conventions,
  parallel error handling, shared underlying machinery).
- Migration: no changes to existing feature API; metric/param are
  additive.
