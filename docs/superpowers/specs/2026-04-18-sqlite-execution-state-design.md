# SQLite-backed execution state: registry, pending rows, and stage-and-upload

**Status:** Draft, pending review (revision 5 — expanded scope).
**Date:** 2026-04-18.
**Scope:** Design spec. Implementation plan follows separately.

## 1. Goals and scope

### 1.1 Goals

Make the DerivaML workspace the authoritative local source of truth for
execution state — enabling robust offline workflows where scripts split
across processes, machines, and network conditions still reach a
correct upload.

Three capabilities, one coherent design:

1. **Execution registry in SQLite.** The workspace keeps a table of
   known-local executions. Enumeration (`list_executions`,
   `find_incomplete_executions`) and resumption
   (`resume_execution(rid)`) read from SQLite, not from scanning the
   filesystem tree. Status transitions are atomic SQLite writes.

2. **Pending-row staging in SQLite.** New rows destined for the
   catalog — both asset rows (file + metadata) and plain rows —
   accumulate in a per-execution pending-rows table until drained.
   RID leasing happens against `public:ERMrest_RID_Lease` so every
   pending row carries its final RID before upload.

3. **One write verb, two modes.** `handle.insert(records)` is the
   single write method. In online mode, rows reach the catalog by the
   time `.insert()` returns. In offline mode, rows stage and wait for
   `exe.upload_execution_outputs()`. The user's code is identical in
   both modes; mode only changes timing.

Design principle: the code to write a row is the same whether the user
is connected or offline, and the same whether they're in the create
script, a compute-heavy middle script, or the final upload script.
SQLite carries state across processes.

### 1.2 Non-goals

- **Updates or deletes.** Insert-only. Modifying existing catalog rows
  or deleting them is a separate future problem.
- **Creating new tables or columns.** Target table must already exist
  in the catalog schema. `exe.table(name)` raises `DerivaMLTableNotFound`
  at call time if absent.
- **Full slice upload-back.** The broader "sync arbitrary local
  changes" problem remains deferred.
- **Filesystem watching.** Directory registration doesn't watch for
  file arrivals; files are surfaced via explicit `register(file)` or
  scan-at-upload.
- **Auto-detecting connection state.** Mode is explicit
  (`mode="online"` vs `mode="offline"`). The library does not silently
  defer writes on server timeout.
- **Cross-workspace sharing.** Executions are scoped to the workspace
  they were created in; the user copying a working directory between
  machines is their concern to handle.

### 1.3 Migration posture

No migration. Clean slate — no in-flight `ManifestStore` entries to
preserve. External names (`exe.asset_file_path`, `ml.asset_record_class`,
`ml.restore_execution`, `upload_execution_outputs`, `exe.table_path`)
keep their signatures as thin shims delegating to the new APIs.

## 2. Architecture

### 2.1 Online mode vs offline mode

`DerivaML` gains a `mode` parameter:

```python
DerivaML(..., mode="online" | "offline")     # default: "online"
```

**Online mode.** Plain-row writes go live to the catalog by the time
`.insert()` returns — the library drains the pending entry inline
after staging. Asset-row writes still stage and wait for upload
(Hatrac is a two-phase pipeline; draining per file would defeat
batching).

**Offline mode.** Every write stages into the workspace SQLite and
stays there until `exe.upload_execution_outputs()` drains it. The
server is only contacted for RID leases and final uploads.

**Creating executions requires online mode.** An Execution row has to
be created on the server (the RID is server-assigned), which can't
happen offline. The supported split-script workflow is:

1. Script 1 (online): `create_execution` → execution RID assigned,
   registry row written to SQLite, configuration persisted.
2. Scripts 2..N (any mode): `resume_execution(rid)` re-hydrates from
   SQLite; writes stage per the current mode.
3. Final script (online): `resume_execution(rid)` +
   `upload_execution_outputs()` drains everything.

Any `create_execution` call while `mode="offline"` raises
`DerivaMLOfflineError` with a clear message.

### 2.2 Workspace SQLite as authoritative state

Everything that matters for resuming an execution lives in SQLite:

- Execution registry and configuration
- Pending-row entries (asset and plain)
- Directory-rule entries
- Lease tokens for crash-safe RID allocation
- Status for every row and every execution

The filesystem tree (`{working_dir}/deriva-ml/execution/{rid}/`) is a
**derived artifact**: asset files live there for the uploader to
consume, staging CSVs are materialized there at drain time, but the
SQLite workspace is the ground truth. If the filesystem is corrupted
or edited externally, SQLite is consulted, not the tree.

This inverts today's model, where file-tree scanning (`os.listdir`)
was the enumeration mechanism.

### 2.3 SQLite schema

Three tables in the workspace SQLite:

#### 2.3.1 `executions`

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
| `created_at` | when the local registry knew about it |

Indexes: `(status)`, `(workflow_rid)`, `(last_activity)`.

Status transitions:

```
create_execution (online only) → created
.execute() context entered     → running
.execute() context exited OK   → stopped
.execute() context exited w/ error → failed
upload_execution_outputs starts → pending_upload
upload_execution_outputs succeeds → uploaded
user calls exe.abort()         → aborted
```

Transitions are atomic SQLite writes. No half-written JSON files.

#### 2.3.2 `pending_rows`

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

#### 2.3.3 `directory_rules`

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

### 2.4 RID leasing (lazy, batched, crash-safe)

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

### 2.5 Public API — Execution registry

```python
ml.list_executions(
    *,
    status: str | list[str] | None = None,
    workflow_rid: str | None = None,
    mode: str | None = None,
    since: datetime | None = None,
) -> list[ExecutionRecord]
    """Return all known-local executions matching the filters. Reads
    from SQLite; no server contact required."""

ml.find_incomplete_executions() -> list[ExecutionRecord]
    """Sugar over list_executions(status=[...]) returning every
    execution not fully uploaded: status in
    (created, running, stopped, pending_upload, failed)."""

ml.resume_execution(execution_rid: str) -> Execution
    """Re-hydrate an Execution from SQLite registry state. Works in
    both online and offline modes; the execution's recorded mode is
    independent of the current DerivaML instance's mode. Raises
    DerivaMLException if no matching registry row exists. Does NOT
    contact the server — resume is purely local.

    New canonical name; `restore_execution` becomes a shim."""

ml.gc_executions(
    *,
    older_than: timedelta | None = None,
    status: str | list[str] | None = None,
    delete_working_dir: bool = False,
) -> int
    """Remove execution registry rows matching the filter. By default
    only deletes registry state; pass delete_working_dir=True to also
    remove the on-disk execution root."""
```

`ExecutionRecord` is a dataclass with the registry columns plus
convenience counts (`pending_rows`, `failed_rows`, etc.) derived from
`pending_rows`.

### 2.6 Public API — Table handle (row-level writes)

```python
exe.table(
    table_name: str,
    *,
    schema: str | None = None,
) -> TableHandle | AssetTableHandle
```

Returns `AssetTableHandle` if the table is an asset table,
`TableHandle` otherwise. Asset-specific methods appear only on
`AssetTableHandle`, visible via IDE tab-completion.

Ambiguous `table_name` without `schema=` raises
`DerivaMLTableNotFound` listing candidates.

#### 2.6.1 `TableHandle` surface

```python
class TableHandle:
    name: str
    schema: str

    def record_class(
        self, *, include_system_columns: bool = False,
    ) -> type[RowRecord]:
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
        """Write rows.
        - Online mode: rows reach the catalog by the time .insert()
          returns.
        - Offline mode: rows stage in workspace SQLite; upload drains
          them at exe.upload_execution_outputs().
        Accepts scalar / Iterable / DataFrame; returns scalar or list
        accordingly. Validation via self.record_class() at call time;
        invalid records raise regardless of mode."""

    def pending(
        self, *, status: str | list[str] | None = None,
    ) -> list[PendingRow]:
        """Pending rows for this table, filtered by status. Includes
        staged/leasing/leased and any in failed awaiting retry."""

    def discard_pending(
        self,
        *,
        rows: Iterable[PendingRow] | None = None,
        status: str | list[str] | None = None,
    ) -> int:
        """Remove pending rows matching the filter. Does not reclaim
        already-acquired server leases."""
```

#### 2.6.2 `AssetTableHandle` surface (extends `TableHandle`)

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
    ) -> AssetDirectoryHandle:
        """Register a directory of files for upload. Takes NO metadata
        kwargs; zero-ceremony default. For per-file metadata, use the
        returned handle's register() or set_metadata(), or iterate
        pending() and set .metadata per file."""
```

#### 2.6.3 `AssetDirectoryHandle`

```python
class AssetDirectoryHandle:
    path: Path                    # source_dir
    rule_id: int
    table: AssetTableHandle

    def register(self, file: str | Path) -> AssetFilePath:
        """Eagerly register a specific file. Use when you need the
        handle to reference its .rid from another row's insert."""

    def scan(self) -> list[AssetFilePath]:
        """Walk source_dir (respecting glob / recurse), register new
        files not yet seen under this rule. Idempotent."""

    def pending(self) -> list[AssetFilePath]:
        """All AssetFilePaths under this rule not yet uploaded."""

    def set_metadata(
        self,
        source: "pd.DataFrame",
        *,
        filename_col: str = "Filename",
    ) -> None:
        """Batch-set metadata across all pending files under this rule.
        One SQLite transaction. DataFrame indexed by filename; other
        columns become metadata. Each resolved value validated through
        record_class."""

    def close(self) -> None:
        """Final scan, mark rule closed. Further register/scan raise."""
```

Deleted files between scans are not auto-removed; upload surfaces
them as per-row failures.

#### 2.6.4 Handle read properties

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

### 2.7 Upload drain

`exe.upload_execution_outputs()`:

1. **Status transition.** Execution `status` → `pending_upload`.
2. **Final scan.** For every active `AssetDirectoryHandle` rule, run
   `scan()` to pick up files written since last register/scan.
3. **Re-validation.** Re-fetch the catalog model; re-validate each
   pending row's `metadata_json` against the typed record class.
   Catches schema drift (required column added, FK tightened).
4. **Batch-lease.** Remaining `status='staged'` rows have RIDs leased
   in chunked POSTs.
5. **Inter-table FK topological sort.** Build a DAG over pending
   rows. Cycle → `DerivaMLCycleError` listing the cycle's tables.
   Intra-table FK cycles unsupported (§7).
6. **Materialize staging tree + invoke existing upload pipeline** per
   topological level. Files to Hatrac, URL/MD5 patching, ERMrest row
   inserts via `GenericUploader`.
7. **Reconcile status.** Each row → `uploaded` or `failed`. On first
   level with failures, the drain aborts after recording all failures
   in that level. Execution `status` → `uploaded` (all succeeded) or
   `failed` (any failed).

**Transaction semantics.** Each row insert is its own transaction.
No batch rollback. Partial success after a crash is the expected
state; per-row `status` in SQLite enables resumability across
processes and machines.

**`retry_failed()`.** Convenience on `Execution` — resets
`status='failed'` entries to `status='leased'` and re-runs
`upload_execution_outputs`.

### 2.8 Online-mode inline drain

In online mode, `.insert(plain_rows)` also drains the entries it
created before returning. Implementation: after staging N rows,
`_drain_pending(execution_rid, entry_ids=[...])` runs steps 4–7 of
§2.7 restricted to those entries.

Asset-file calls (`.asset_file(...)`, `.asset_directory(...)`) do NOT
drain eagerly even in online mode — they always defer until
`upload_execution_outputs`.

So in practice:

- Online + `.insert(plain_row)` → row is in catalog by return.
- Online + `.asset_file(file, ...)` → staged; reaches catalog at
  `upload_execution_outputs`.
- Offline + anything → staged; nothing reaches catalog until
  `upload_execution_outputs`.

### 2.9 Error handling

V1 is fail-fast: the drain aborts after the first level with failures.
Pending rows not yet attempted keep their status for inspection and
retry. A best-effort mode is deferred (§7).

`handle.pending(status='failed')` for triage. `error` carries the
message. User edits metadata or fixes referenced rows, then
`exe.retry_failed()`.

### 2.10 Backward-compatibility shims

Existing public methods become thin shims delegating to the new
surface. No caller changes required.

```python
# exe.asset_file_path(asset_name, file, ...):
exe.table(asset_name).asset_file(file, ...)

# ml.asset_record_class(table):
exe.table(table).record_class()

# ml.restore_execution(rid):
ml.resume_execution(rid)

# exe.table_path(name):
exe.table(name).upload_csv_path()

# ml.find_executions(...):   # existing, server-side enumeration
# delegates to ml.list_executions(...) when called locally.
```

## 3. Walkthrough — three-script workflow

The motivating workflow: a user connects from home, creates an
execution, downloads a dataset bag. Next day on a laptop with no
connection, they run the model against the bag and write
predictions + asset masks locally. The following day, reconnecting,
they upload.

**Script 1 — create the execution (online):**

```python
ml = DerivaML(hostname="example.org", catalog_id="42")   # default: online
exe = ml.create_execution(
    ExecutionConfiguration(
        datasets=[DatasetSpecConfig(rid="1-XYZ", version="1.0.0")],
        workflow=my_workflow,
        description="Segmentation run 2026-04-18",
    )
)

with exe.execute() as e:
    bag = e.datasets["1-XYZ"].bag          # download to workspace
# SQLite executions row: rid=5-ABC, status='stopped', mode='online',
#   config_json=<serialized>, working_dir_rel='execution/5-ABC'.
```

User captures `exe.execution_rid` somehow — prints, notebook, copies
to laptop. They can also always look it up via
`ml.list_executions()`.

**Script 2 — offline computation:**

```python
ml = DerivaML(hostname="example.org", catalog_id="42", mode="offline")

for record in ml.find_incomplete_executions():
    print(record.rid, record.status, record.description)
# Prints: 5-ABC  stopped  "Segmentation run 2026-04-18"

exe = ml.resume_execution("5-ABC")   # from SQLite, no server contact

with exe.execute() as e:
    bag = e.datasets["1-XYZ"].bag    # already local

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
            "Mask": mask.rid,           # lazy-leased; batches on read
            "Confidence": conf,
        })
# SQLite state after script 2:
#   executions.5-ABC.status = 'stopped'
#   pending_rows: N masks + N predictions, status='staged' / 'leased'
```

In `mode="offline"`, `.insert()` stages Prediction rows rather than
going live. `.asset_file()` would have deferred anyway. Lease POSTs
for `mask.rid` reads need the server briefly; the user can batch
these by avoiding `.rid` reads until reconnection, in which case the
upload drain issues them all in one shot.

**Script 3 — upload (online):**

```python
ml = DerivaML(hostname="example.org", catalog_id="42")   # online

for record in ml.find_incomplete_executions():
    print(record.rid, record.status, record.pending_rows)
# Prints: 5-ABC  stopped  <N+N pending>

exe = ml.resume_execution("5-ABC")
exe.upload_execution_outputs()
# Drain:
#   - final scan (no-op; no directory rules)
#   - re-validate metadata against fresh catalog model
#   - lease any remaining RIDs
#   - FK topo-sort: Masks before Predictions
#   - materialize staging tree, invoke upload pipeline
#   - status: pending_rows → uploaded; executions.5-ABC → uploaded
```

The execution RID (`5-ABC`) is preserved across all three scripts
because it's stored in `executions.rid` in SQLite. Every
`pending_rows.execution_rid` points at it. The upload writes rows
with correct provenance.

## 4. Algorithms

### 4.1 Handle construction

```
exe.table(name, schema=None):
  1. Resolve via model.name_to_table(name, schema=schema). Raise
     DerivaMLTableNotFound if missing or ambiguous-without-schema.
  2. If model.is_asset(table): return AssetTableHandle(exe, table).
     Else:                      return TableHandle(exe, table).
  Handles hold exe + table references. No cached state.
```

### 4.2 `TableHandle.insert(records)`

```
1. Normalize records:
   - DataFrame → list of dicts.
   - Scalar → list of one (flag: scalar_input).
   - Iterable → materialize in chunks.
2. For each chunk:
   a. For each rec:
      - RowRecord: pass through.
      - dict: rec = self.record_class()(**rec). Raises on failure.
   b. Generate lease_token per row.
   c. BEGIN TRANSACTION.
   d. Insert N pending_rows: execution_rid, target_schema, target_table,
      status='staged', rid=NULL.
   e. COMMIT.
   f. progress(done, total) if provided.
3. Update executions.last_activity = now().
4. If exe.ml.mode == 'online':
   _drain_pending(execution_rid, entry_ids=[...]).
5. Return scalar handle or list.
```

### 4.3 `AssetTableHandle.asset_file(file, ...)`

```
1. Validate file, determine staging path.
2. Symlink/copy file to assets/{AssetTable}/.
3. Construct metadata record via self.record_class()(
   **{metadata, **kwargs}).
4. Generate lease_token.
5. Write pending_rows with asset_file_path set, status='staged'.
6. Update executions.last_activity.
7. Return AssetFilePath bound to the entry. No drain.
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
3. Load configuration from executions.config_json.
4. Construct Execution object bound to self.
5. Execution carries exe.execution_rid=<rid>. Any exe.table(...)
   calls scope pending rows to this rid.
6. Do NOT contact the server.
7. Return Execution.
```

### 4.7 Execution lifecycle transitions

```
create_execution:
  1. Require online mode.
  2. POST Execution row to catalog → rid=<assigned>.
  3. Serialize ExecutionConfiguration to config_json.
  4. INSERT executions row: rid, status='created', mode, config_json,
     start_time=NULL, last_activity=now().
  5. Return Execution.

.execute() context enter:
  UPDATE executions SET status='running', start_time=now()
     WHERE rid=<rid>.

.execute() context exit OK:
  UPDATE executions SET status='stopped', stop_time=now()
     WHERE rid=<rid>.

.execute() context exit w/ exception:
  UPDATE executions SET status='failed', stop_time=now(), error=<msg>
     WHERE rid=<rid>.

upload_execution_outputs start:
  UPDATE executions SET status='pending_upload' WHERE rid=<rid>.

upload_execution_outputs OK:
  UPDATE executions SET status='uploaded' WHERE rid=<rid>.

upload_execution_outputs w/ failures:
  UPDATE executions SET status='failed', error=<msg> WHERE rid=<rid>.
  (retry_failed resets individual rows back to 'leased'.)

exe.abort():
  UPDATE executions SET status='aborted' WHERE rid=<rid>.
  Pending rows remain; user can discard_pending or resume.
```

## 5. Testing

### 5.1 Unit tests (no live catalog)

Registry:
- `create_execution` writes SQLite row.
- `resume_execution(rid)` reads from SQLite; works without server.
- `resume_execution(missing_rid)` raises.
- `list_executions()` returns all rows; filter combinations work.
- `find_incomplete_executions()` returns expected status set.
- `gc_executions(older_than=...)` removes matching rows.
- Lifecycle transitions produce correct status sequences.
- Crash between `running` and `stopped`: restart → status still
  `running`; user intervention required (`abort()` or new `execute()`).

Handle dispatch:
- `exe.table(plain)` → `TableHandle`; no asset methods.
- `exe.table(asset)` → `AssetTableHandle`; asset methods present.
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

Compatibility shims:
- `exe.asset_file_path(...)` delegates.
- `ml.asset_record_class(...)` delegates.
- `ml.restore_execution(...)` delegates to `ml.resume_execution(...)`.

### 5.2 Integration tests (live catalog)

Online mode:
- `ml.create_execution(config)` writes SQLite + server.
- `exe.table("Subject").insert({...})` reaches catalog before return.
- `exe.table("Image").asset_file(...)` + `upload_execution_outputs()`:
  file in Hatrac, row in catalog.

Offline mode:
- Same code with `mode="offline"`: rows reach catalog at
  `upload_execution_outputs()`, not at `.insert()` time.
- `create_execution` with `mode="offline"` raises.

Split-script workflow:
- Script A (online): create_execution, download bag, print exe.rid.
- Script B (offline): resume_execution(rid), insert rows, exit.
- Script C (online): resume_execution(rid), upload_execution_outputs.
- Verify: catalog Subject rows have correct `Execution` FK / RCB;
  RID assigned in script B matches catalog.

Resumption robustness:
- Crash mid-insert: re-open workspace → `find_incomplete_executions`
  surfaces the execution; `resume_execution` works; pending_rows
  intact.
- Crash mid-upload: `find_incomplete_executions` surfaces it;
  `retry_failed` completes the remaining work.
- Corrupted configuration.json on disk: `resume_execution` still
  works (reads from SQLite).

Edge cases:
- Inter-table FK cycle raises `DerivaMLCycleError`.
- Schema drift: add required column between stage and upload → error.
- File deleted between register and upload: per-row failure with
  clear message.

### 5.3 Regression

- Existing `exe.asset_file_path` tests (via shim).
- Existing `ml.restore_execution` tests (via shim).
- Existing `ml.find_executions` tests (via shim).
- Existing `upload_execution_outputs` asset tests.

## 6. Key simplifications from earlier revisions

The spec reached rev 5 through several pivots. Summary of rejected
complexity:

- **Separate `stage` and `insert` verbs** (rev 2/3) — collapsed. One
  `insert` handles both, online vs offline is a mode.
- **Workspace-scoped pending rows** (rev 3) — dropped. Everything is
  execution-scoped.
- **`dedupe_key` for notebook idempotency** — dropped. Users
  `discard_pending` + re-run.
- **`best_effort` upload mode** — dropped. Fail-fast is V1.
- **Handle-as-FK implicit coercion** — dropped. Users pass `.rid`
  explicitly. Self-documenting at the call site.
- **`TableHandle.insert` as pathBuilder passthrough** — dropped. The
  one `insert` does the right thing for the current mode.
- **Forwarding catalog metadata through TableHandle** — dropped.
  Users reach `ml.model` for schema metadata; `TableHandle` is for
  operations only.
- **Sidecar CSV metadata for directories** — dropped. SQLite is the
  single metadata source; users who have a CSV convert it to a
  DataFrame and call `set_metadata(df)`.

Concept count after rev 5: **10**.

1. `DerivaML(..., mode="online"|"offline")` — mode at construction.
2. `ml.list_executions() / find_incomplete_executions() / resume_execution()` — registry.
3. `ml.create_execution()` — online only.
4. `exe.table(name)` — row-level entry point.
5. `TableHandle / AssetTableHandle` — asset variant adds file methods.
6. `handle.record_class()` — typed record.
7. `handle.insert(records)` — one write verb.
8. `handle.pending() / discard_pending()` — triage.
9. `handle.asset_file() / asset_directory()` — asset-specific.
10. `exe.upload_execution_outputs()` — drain.

## 7. Known limits

- **Updates / deletes:** insert-only.
- **Intra-table FK cycles:** unsupported. Users split into batches.
- **Binary columns (`bytea`):** unsupported; `insert()` raises
  `NotImplementedError`. Asset-table route is the supported path.
- **CSV-based insert roundtrip (§2.7 step 6):** transitional. V2 can
  swap to direct `pathBuilder` insert without public-API change.
- **Orphaned server leases:** possible only if client loses its
  workspace state after a successful lease POST. Not reconcilable
  from the client; server-side GC is an ERMrest concern.
- **Creating new executions offline:** not supported. Execution RID
  is server-assigned and requires an online `create_execution` call.
- **best_effort upload mode, dedupe_key, handle-FK coercion:**
  deferred. V2 can add these when users hit their absence.
- **Cross-machine workspace sync:** user's concern; the library does
  not replicate `workspace.db` between machines.
