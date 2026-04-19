# Stage & upload new rows — design spec

**Status:** Draft, pending review (revised 2026-04-18 from first-round review).
**Date:** 2026-04-18.
**Scope:** Design spec. Implementation plan follows separately.

## 1. Goals and scope

### 1.1 Goals

Generalize the existing `upload_assets` pathway so the same pipeline uploads:

1. **Asset rows** — a file in Hatrac plus a row in an asset table (today's
   `asset_file_path` behavior).
2. **Plain new rows** — row inserts into any catalog table, no file
   attached. Enables bulk annotation workflows, bulk dataset population,
   and any other "add N new records to table T" use case.
3. **Directory-registered asset rows** — one call registers every file in
   a local directory as a pending asset upload. Files may trickle in over
   time.

### 1.2 Non-goals

- **Updates or deletes.** V1 is insert-only. Modifying existing catalog
  rows and deleting them are separate design problems and out of scope.
- **Creating new tables or columns.** The target table must already exist
  in the catalog schema before staging begins. V1 is row-level only; no
  DDL, no schema evolution as part of the stage/upload pipeline. If the
  target table does not exist, `stage_row` / `stage_rows` /
  `asset_file_path` raise at call time.
- **Full slice upload-back.** The broader "sync arbitrary local changes
  to catalog" problem (Phase 1 §1.1 goal #4) remains deferred.
- **Offline editing across disconnected sessions.** Target "work locally
  while connected."
- **Filesystem watching.** Directory registration does NOT watch the
  source directory for new files. File arrival is surfaced via explicit
  `register(file)` calls or scan-at-upload-time; no inotify/fsevents
  dependency.
- **Auto-generating metadata from sidecar files.** Per-file metadata is
  set by iterating the directory handle's files (or a batch helper on the
  handle) and writing via the existing `.metadata` setter. See §2.4.4.
- **Network-required staging.** A `stage_*` or `asset_file_path` call
  must not fail because the server is briefly unreachable. RID leasing
  is deferred to the first server-touching operation on the entry; see
  §2.2.

### 1.3 Migration posture

No migration. This work ships on a clean slate — no in-flight
`ManifestStore` entries to preserve. The SQLite schema in §2.3 is the
starting shape. External API names (`asset_file_path`, `upload_assets`,
`upload_execution_outputs`) keep their signatures; the generalization
happens beneath those entry points.

## 2. Architecture

### 2.1 One store, one upload drain

The workspace holds one per-execution `PendingRowsStore` (the former
`ManifestStore`, renamed to reflect its broader role) carrying a single
pending-row entry shape that subsumes both asset uploads and plain row
inserts.

- **Entry shape:** a pending row has `target_schema`, `target_table`,
  `metadata` (the column values), and an **optional** `asset_file_path`
  pointing at a local file. When the file pointer is set, the entry is
  an asset upload (file → Hatrac → row with URL/MD5 patched in). When
  null, the entry is a plain-row insert.
- **One upload drain:** `upload_execution_outputs` iterates pending
  entries, handles file-attached entries via the existing Hatrac +
  row-insert pipe, handles file-less entries via plain row insert. Order
  is determined by FK dependencies (§2.5).

### 2.2 RID leasing lifecycle (deferred, reconcilable)

Every pending row eventually holds a real, server-assigned RID obtained
from `public:ERMrest_RID_Lease`. The lease is **not** acquired at stage
time — staging is pure local work and must not require network
connectivity. Instead:

**Lease timing.** A lease is acquired for a pending entry at the latest
of these moments (whichever comes first):

- The caller explicitly reads `.rid` on the handle (pulls on demand).
- The caller stages a second row whose metadata references this entry's
  handle as a FK value (the staging path resolves the handle by
  materializing its RID; see §2.4.3).
- The upload drain begins (batch-lease all still-unleased entries in one
  POST).

Until lease time, the handle carries `.rid == None` and `status='staged'`.

Rationale: deferring preserves "staging is offline-safe" and batches
network cost. Users who never read `.rid` before upload pay exactly one
batched POST for the whole execution's pending rows.

**Lease batching.** `ERMrest_RID_Lease` accepts a list body. One POST
allocates N RIDs for N entries. Lease batch size is capped at a
configurable chunk (`PENDING_ROWS_LEASE_CHUNK`, default 500) to stay
under ERMrest's request-size limits; larger batches are chunked.

**Crash-safe lease acquisition.** Leasing follows a two-phase pattern to
tolerate client crashes between the server POST and the SQLite write:

1. Before posting, write the entry with `status='leasing'` and a
   client-side idempotency token (UUID stored in the entry).
2. POST to `/entity/public:ERMrest_RID_Lease` with the batch; each body
   element carries the token in the `ID` column (the optional label
   field on `ERMrest_RID_Lease`).
3. On success, update the SQLite entries: `rid=<leased>`, `status='leased'`.
4. On workspace open, any entries still in `status='leasing'` trigger a
   reconciliation: query `ERMrest_RID_Lease` filtered by the stored
   idempotency tokens. If a lease exists for the token, adopt the RID
   and continue. If not, the POST never reached the server or failed;
   reset to `status='staged'` and let the next lease cycle retry.

Outcome: orphaned leases can only occur if the server accepted a POST
that the client never acknowledged AND the client later lost its SQLite
state (e.g. workspace deleted). In normal crash-recovery paths, no
leases are orphaned. The spec acknowledges that truly-orphaned leases
(catastrophic loss of workspace state) are accepted as a known limit —
server-side stale-lease GC is an ERMrest concern, not this spec's.

### 2.3 SQLite schema

Two tables in the workspace SQLite, both scoped to the running
execution:

**`pending_rows`** — one row per pending catalog-row insert.

| Column | Purpose |
|---|---|
| `id` | integer primary key, internal |
| `execution_rid` | scope |
| `key` | `"{target_table}/{file_name}"` for asset rows, or a stable hash for plain rows, or a caller-supplied `dedupe_key` (§2.4.2) |
| `target_schema` | catalog schema for insert |
| `target_table` | catalog table for insert |
| `rid` | leased RID, null until lease acquired (§2.2) |
| `lease_token` | client-side idempotency token for lease reconciliation |
| `metadata_json` | column values as JSON (see §2.3.1 for serialization) |
| `asset_file_path` | local file path, or null for plain rows |
| `asset_types_json` | vocabulary terms, or null for plain rows |
| `description` | optional |
| `status` | `staged \| leasing \| leased \| uploading \| uploaded \| failed` |
| `error` | set on failure |
| `created_at` | timestamp |
| `leased_at` | timestamp, null until lease acquired |
| `uploaded_at` | timestamp, null until uploaded |
| `rule_id` | fk to `directory_rules.id`, null for non-directory rows |

Indexes: `(execution_rid, status)`, `(execution_rid, target_table)`, and
`(execution_rid, key)` for dedupe lookups.

**`directory_rules`** — one row per registered directory.

| Column | Purpose |
|---|---|
| `id` | integer primary key, referenced by `pending_rows.rule_id` |
| `execution_rid` | scope |
| `target_schema` / `target_table` | catalog target for each registered file |
| `source_dir` | absolute path to the directory on disk |
| `glob` | file selection glob, default `"*"` |
| `recurse` | bool |
| `copy_files` | bool |
| `asset_types_json` | asset type terms applied to every registered file |
| `status` | `active \| closed` |
| `created_at` | timestamp |

#### 2.3.1 Serialization of `metadata_json`

Reuses `AssetManifest._json_default` from `src/deriva_ml/asset/manifest.py`
which already handles `datetime`, `date`, `Path`, and `Pydantic.BaseModel`.
Pending-rows writes MUST route through this helper to keep serialization
identical to today's asset manifest — preventing roundtrip drift between
the two code paths.

On reads, the raw dict is handed back to the appropriate `RowRecord`
subclass (§2.4.1) which coerces ISO-format strings back to
`datetime`/`date` via its Pydantic type schema.

### 2.4 Public API

#### 2.4.1 Typed row records

A single factory produces a Pydantic model for any insertable catalog
table:

```python
ml.record_class(
    table_name: str,
    *,
    schema: str | None = None,          # required if the table name is ambiguous
    include_system_columns: bool = False,  # RID, RCT, RMT, RCB, RMB
) -> type[RowRecord]
```

`RowRecord` is a new `BaseModel` subclass. Fields are derived from the
catalog column metadata: non-nullable columns become required; nullable
columns become `Optional` with their catalog default; `extra='forbid'`
catches typos at construction time. The existing `_map_column_type`
helper for ERMrest → Python types is reused.

**`AssetRecord` becomes a subclass of `RowRecord`** with a column filter
that exposes only the asset-metadata subset. Today's `asset_record_class`
is retained as an alias that calls `record_class(..., column_filter=
model.asset_metadata(table))`. Single factory, single drift point.

**Schema disambiguation.** If `table_name` appears in multiple schemas
and `schema` is not supplied, raise `DerivaMLTableNotFound` listing the
candidates — matches today's `name_to_table` behavior.

#### 2.4.2 Plain row staging

```python
exe.stage_row(
    target_table: str,
    record: RowRecord | dict | "handle",
    *,
    schema: str | None = None,
    description: str | None = None,
    dedupe_key: str | None = None,
) -> PendingRow

exe.stage_rows(
    target_table: str,
    records: Iterable[RowRecord | dict] | "pd.DataFrame",
    *,
    schema: str | None = None,
    dedupe_key: Callable[[dict], str] | None = None,
    chunk_size: int = 1000,
    progress: Callable[[int, int], None] | None = None,
) -> list[PendingRow]
```

**Accepts a `pandas.DataFrame` natively.** Internally calls
`df.to_dict(orient="records")`. Same contract as passing an iterable.

**Validation at staging time.** Every record is validated through
`ml.record_class(target_table)`:
- `RowRecord` instances are already validated at construction.
- `dict` inputs are passed to the typed record constructor, which will
  raise Pydantic errors for missing required fields, unknown fields
  (`extra='forbid'`), or type mismatches.
- A handle (`PendingRow` / `AssetFilePath`) is coerced to its `.rid`
  via `__get_pydantic_core_schema__` on the handle types — the typed
  record's FK column accepts either a RID string or a handle; handles
  materialize via the lazy-lease path (§2.2). Users write
  `stage_row("Image", {"Subject": subj_handle, ...})` and the right
  thing happens.

**Idempotency via `dedupe_key`.** Optional. If supplied, the entry's
`key` column is set to the dedupe key (instead of an auto-hash). On
`stage_row` with a matching dedupe_key for the same `(execution_rid,
target_table)` triple:
- If the existing entry is in `staged | leasing | leased` — return the
  existing handle unchanged.
- If the existing entry is in `uploading | uploaded` — raise
  `DerivaMLException` with a clear message.
- If `failed` — update metadata and reset to `staged`.

Notebook re-runs with explicit dedupe keys become idempotent.

**Behavior:**
1. Resolve `target_table` against the catalog model. Raise
   `DerivaMLTableNotFound` if not present.
2. Validate each record via `record_class(target_table)` (raises on
   failure).
3. Write pending entries to SQLite with `status='staged'`, `rid=None`,
   `lease_token` set, `metadata_json` serialized per §2.3.1.
4. Do NOT acquire RIDs yet (§2.2).
5. Return list of `PendingRow` handles.

#### 2.4.3 Handle FK coercion

A `PendingRow` or `AssetFilePath` passed as a column value in another
`stage_*` call is coerced by forcing a lease:

```python
subj = exe.stage_row("Subject", {"Name": "Alice"})   # status=staged, rid=None
exe.stage_row("Image", {"Subject": subj, "Filename": "a.jpg"})
# → subj's rid is materialized (batched server POST); Image record validates.
```

Handle-as-FK lookups are coalesced: if N pending rows reference handles
in the same `stage_rows` call, one batched lease POST services all of
them.

#### 2.4.4 Asset file registration (existing API, internals updated)

```python
exe.asset_file_path(
    asset_name, file_name, ...,
    metadata=None, ...,
) -> AssetFilePath
```

Signature unchanged. Internals updated: writes a pending entry with
`asset_file_path=<local path>`, `status='staged'`, `rid=None`. The lease
is deferred exactly as for `stage_row`; this preserves the
no-network-at-call-time contract today's callers depend on.

#### 2.4.5 Asset directory registration (new)

```python
exe.asset_directory_path(
    asset_name: str,
    source_dir: str | Path,
    *,
    asset_types: list[str] | str | None = None,
    glob: str = "*",
    recurse: bool = False,
    copy_files: bool = False,
) -> AssetDirectoryHandle
```

Deliberately takes **no metadata kwargs**. Two intents:

- **Zero-ceremony.** `asset_name` + `source_dir`. Every file becomes a
  pending row using only file-derived columns (MD5, URL, Length,
  Filename). If the asset table requires user-supplied columns, upload-
  time re-validation fails loudly listing what's missing.
- **Per-file metadata.** Caller registers the directory, then sets
  metadata — either one file at a time via `.metadata = ...` or in
  batch via `handle.set_metadata(...)` (§2.4.6).

```python
class AssetDirectoryHandle:
    path: Path
    rule_id: int
    asset_name: str

    def register(self, file: str | Path) -> AssetFilePath:
        """Eagerly register a specific file; return its handle."""

    def scan(self) -> list[AssetFilePath]:
        """Walk source_dir, register any new files. Idempotent."""

    def pending(self) -> list[AssetFilePath]:
        """All handles under this rule whose status is not uploaded/failed."""

    def set_metadata(
        self,
        source: dict | Callable[[Path], dict | RowRecord] | "pd.DataFrame",
        *,
        filename_col: str = "Filename",
    ) -> None:
        """Batch-set metadata across all pending entries under this rule.

        One SQLite transaction. Accepts:
        - dict — applied identically to every pending file
        - callable(Path) -> dict | RowRecord — per-file resolver
        - DataFrame — indexed by filename (``filename_col`` identifies the
          column to match, or the index if it's already the filename).
          Each row's remaining columns become the file's metadata.
        Each resolved value is validated through the typed record class."""

    def close(self) -> None:
        """Final scan, then mark rule status=closed."""
```

**Deleted files between scans.** `scan()` does not auto-remove pending
rows whose files no longer exist. At upload time, a missing file for an
asset-row entry is a per-row failure with a clear `error` message.

#### 2.4.6 Batch metadata setters

Iterate-and-set on `handle.pending()` is still supported (the §2.4.4
`AssetFilePath.metadata` setter) but `handle.set_metadata(...)` is the
preferred batch API. It runs one SQLite transaction across N files, one
set of typed-record validations, and one write path — avoiding N×
roundtrips for the common case.

#### 2.4.7 Pending-row inspection and discard (public)

```python
exe.list_pending(
    *,
    table: str | None = None,
    status: str | list[str] | None = None,
) -> list[PendingRow | AssetFilePath]

exe.discard_pending(
    *,
    table: str | None = None,
    rows: Iterable[PendingRow | AssetFilePath] | None = None,
    status: str | list[str] | None = None,
) -> int   # returns count discarded
```

`list_pending` returns handles for entries matching the filter. Used for
triage after a failed upload (`status='failed'`) or inspection before
upload.

`discard_pending` removes entries from SQLite. Does NOT reclaim any
already-acquired server-side leases (per §2.2 known-limit statement).

#### 2.4.8 Idempotent directory re-registration across sessions

`asset_directory_path` keyed by `(execution_rid, target_table,
source_dir)`:
- Identical parameters → return existing handle.
- Any parameter differs → raise `DerivaMLException` with the diff.

### 2.5 Upload drain

`Execution.upload_execution_outputs` adds a staging-materialization
prelude before delegating to today's `upload_directory`:

```python
upload_execution_outputs(
    mode: Literal["strict", "best_effort"] = "strict",
    ...   # existing kwargs
)
```

**Sequence:**

1. **Final scan.** For every `active` directory rule, `scan()` to pick up
   files written since last register/scan.
2. **Re-validation against fresh model.** The catalog model may have
   changed between stage and upload (added required column, tightened
   FK). Re-fetch the model and re-validate every pending row's
   `metadata_json` via the typed record class. In strict mode, any
   failure aborts before any inserts. In best-effort mode, failures are
   recorded and continue.
3. **Batch-lease unleased entries.** Any entries still in `status='staged'`
   have RIDs leased now, in one or more chunked POSTs per
   `PENDING_ROWS_LEASE_CHUNK` (§2.2).
4. **Inter-table FK topological sort.** Build a DAG where table A
   depends on table B iff some pending A-row references a pending
   B-row's leased RID. If the DAG contains a cycle, raise
   `DerivaMLCycleError` listing the cycle's tables. Intra-table FK
   cycles (e.g. a self-referencing Subject pointing at another new
   Subject) are not supported in V1 (per §6). User must split into
   separate batches.
5. **Materialize staging tree.** For each pending row in topo order:
   - **Asset rows:** symlink/copy the file into the metadata-path
     layout expected by the existing asset upload spec
     (`{exec_dir}/asset/{schema}/{asset_table}/{v1}/{v2}/.../{file_name}`).
     Mark `status='uploading'`.
   - **Plain rows:** append a CSV row to
     `{exec_dir}/table/{target_schema}/{target_table}/{target_table}.csv`.
     Mark `status='uploading'`.
   CSVs are written one table at a time in topological order so that
   when `upload_directory` is called multiple times (once per topo
   level), FK dependencies are already satisfied.
6. **Invoke `upload_directory` per topological level.** Hatrac uploads,
   URL/MD5/Length patching, ERMrest row inserts via `GenericUploader`.
7. **Reconcile status per row.** From `FileUploadState` dict:
   - `success` → `status='uploaded'`, `uploaded_at=now()`
   - `failure` → `status='failed'`, `error=<msg>`. In strict mode,
     abort after recording all failures in the current level. In
     best-effort mode, continue to next level.

**Transaction semantics.** Each individual row insert is its own
transaction. There is no batch rollback at the ERMrest layer. Partial
success after a crash is the expected state; SQLite reflects per-row
status so the next `upload_execution_outputs` invocation picks up where
it left off.

**`retry_failed()`.** Convenience that calls `upload_execution_outputs`
restricted to entries with `status='failed'`, resetting them to
`status='leased'` first.

### 2.6 CSV handoff (transitional)

Plain-row inserts currently route through
`GenericUploader`'s `"asset_type": "table"` path: serialize row →
write CSV → re-parse CSV → ERMrest insert. This reuses the existing
retry / progress / size-limit machinery without duplication, but
introduces a serialization round-trip.

**Column types that roundtrip cleanly through CSV:**
`text`, `markdown`, `longtext`, `int2/4/8`, `serial*`, `float4/8`,
`numeric`, `boolean` (as `"True"`/`"False"`), `date`, `timestamp`,
`timestamptz` (as ISO 8601 strings).

**Risk column types** — may need explicit normalization:
- `json` / `jsonb` — serialized as a JSON string; must be parsed
  correctly by the server's CSV importer.
- `bytea` or any binary — not CSV-friendly. If encountered, V1 raises
  `NotImplementedError`. Plain-row staging for binary columns is
  explicitly out of scope.

**V1 is transitional.** V2 can swap the CSV detour for a direct
`pathBuilder` insert once the pending-rows layer is stable. The public
API does not change; the swap is purely internal. This is called out
here (not in an FAQ) so reviewers and future maintainers see the
intended arc.

### 2.7 Error handling and recovery

- **Strict mode (default).** First failure of a given phase (validation,
  materialization, insert) aborts after the current level records its
  failures. Pending entries not yet attempted keep their `status`
  unchanged so the user can inspect and retry.
- **Best-effort mode.** Each row attempts independently. Failures are
  recorded but don't abort siblings. Returns a summary.

**Restart semantics.** `upload_execution_outputs` is resumable. Rows in
`status='staged' | 'leasing' | 'leased' | 'failed'` are processed;
`'uploading' | 'uploaded'` are skipped. `'uploading'` rows are treated
as recoverable — the drain checks Hatrac / catalog state for them and
advances them to `'uploaded'` if the server-side state already reflects
the insert.

**Lease validity.** Leased RIDs stay valid indefinitely server-side; no
client-side expiration to worry about.

**Inspection & triage.** `exe.list_pending(status='failed')` returns the
failed entries with their `error` messages. Users edit metadata or fix
referenced data, then call `exe.retry_failed()` or
`exe.discard_pending(...)`.

### 2.8 Relationship to workspace and executions

Pending-rows are scoped by `execution_rid`. They live in workspace
SQLite under the same execution context as today's manifest.
`Workspace.manifest_store()` returns an instance bound to a specific
execution; the V1 generalization widens the returned object's API
surface but does not change the outer `Workspace` API.

## 3. Details worth calling out

### 3.1 `PendingRow` vs `AssetFilePath`

`AssetFilePath` remains a `Path` subclass so all existing `Path`-based
callers keep working. `PendingRow` is a new non-Path handle for plain
rows. Both carry the same pending-row contract:

```python
class _PendingHandle(Protocol):
    rid: str | None                 # None until lease acquired
    status: str                     # see §2.3 status column
    metadata: dict[str, Any]        # read; setter writes through
    error: str | None
```

**Storage model.** Both handles are **thin stateless views** over the
SQLite row — every property read issues a SELECT, every write commits an
UPDATE. No in-memory caching. Rationale:

- Guarantees consistency across multiple handles pointing at the same
  underlying entry (common in directory workflows).
- Survives process restarts without invalidation logic.
- Performance cost: one SQLite read per property access. Workspaces are
  WAL-mode SQLite on local disk; typical read is ~100 µs. Batch setters
  (§2.4.6) avoid the per-row overhead for bulk operations.

Handles are cheap to construct (just hold a connection reference and
primary key); they do not hold their own state.

### 3.2 What exists today vs. what's new

| Component | Status |
|---|---|
| `ManifestStore` (will become `PendingRowsStore`) | Exists, extended |
| `AssetEntry` → `PendingRowEntry` | Generalized |
| `AssetRecord` factory | Retained as specialization of `RowRecord` |
| `GenericUploader` + `table_regex` CSV insert | Reused unchanged |
| `upload_directory` drain | Reused unchanged |
| `asset_file_path(...)` signature | Unchanged |
| `AssetFilePath` class | Extended with lazy-lease `rid` |
| Workspace SQLite (WAL) | Unchanged |
| `ERMrest_RID_Lease` server table | Accessed for leasing |
| `record_class(table)` factory | **New** |
| `RowRecord` base class | **New** |
| `stage_row` / `stage_rows` (DataFrame-native) | **New** |
| `asset_directory_path` + `AssetDirectoryHandle` | **New** |
| `handle.set_metadata(...)` batch setter | **New** |
| `exe.list_pending` / `exe.discard_pending` | **New** |
| `exe.retry_failed()` | **New** |
| Lazy + batched + crash-safe RID leasing | **New** |
| Handle-as-FK-value coercion | **New** |
| Inter-table FK topological sort + cycle detection | **New** |
| Stale-schema re-validation at upload time | **New** |

### 3.3 Validation timing

Three validation checkpoints:

1. **At staging time:** typed-record construction catches type errors,
   unknown columns, missing required fields. Fails immediately.
2. **At upload time (pre-lease):** re-validate against a fresh catalog
   model snapshot. Catches schema drift (required column added after
   staging, FK constraint tightened, column dropped).
3. **At upload time (post-insert):** `GenericUploader` surfaces any
   remaining server-side rejection (unique constraint, orphaned FK,
   permissions).

### 3.4 Logging and observability

All new paths use `get_logger("deriva_ml.local_db.pending_rows")` per the
project's shared-utilities convention. Events logged at INFO:

- pending row created (table, key, status)
- lease batch acquired (count, duration)
- upload level started/completed (table, counts)
- row insert succeeded/failed (rid, table, error)

Events logged at DEBUG: SQLite reads/writes for handle properties,
per-file scan results.

## 4. Algorithms

### 4.1 `exe.stage_row(table, record)`

```
1. Resolve table against model. Raise DerivaMLTableNotFound if missing.
2. If record is a dict: record = record_class(table)(**record)  # validates
   If record is a RowRecord: pass through (already validated)
   If record contains handle-valued columns: record is unchanged
     (handle coercion happens at write time via __get_pydantic_core_schema__)
3. Generate lease_token = uuid4().
4. Write pending_rows entry: execution_rid, target_schema, target_table,
   key=<hash or dedupe_key>, rid=NULL, lease_token, metadata_json,
   status='staged', created_at=now().
5. Return PendingRow(conn, row_id).
```

### 4.2 `exe.stage_rows(table, records_or_df)`

```
1. If records_or_df is DataFrame: convert to list of dicts via
   to_dict(orient='records').
2. For chunk of `chunk_size` records:
   a. Validate each: record_class(table)(**rec) or pass if RowRecord.
   b. BEGIN TRANSACTION.
   c. Insert N pending_rows entries in one transaction with
      lease_token per row, status='staged'.
   d. COMMIT.
   e. If progress callback provided: progress(done, total)
3. Return list of PendingRow handles.
```

### 4.3 Handle `.rid` materialization (lazy lease)

```
1. If entry.rid is not null, return it.
2. Call workspace._acquire_leases([self.row_id]) which:
   a. Find all entries with status='staged' or 'leasing'.
   b. For each in 'staged': set status='leasing', commit.
   c. POST to /entity/public:ERMrest_RID_Lease with body
      [{"ID": lease_token} for each].
   d. For each response: UPDATE pending_rows SET rid=<assigned>,
      status='leased', leased_at=now() WHERE lease_token=<token>.
3. Return entry.rid.
```

### 4.4 Workspace startup lease reconciliation

```
1. Query pending_rows WHERE status='leasing' for this workspace.
2. If empty: done.
3. Group by execution_rid. For each group:
   a. Query ERMrest_RID_Lease for the group's lease_tokens.
   b. For each found lease: UPDATE pending_rows SET rid=<lease.RID>,
      status='leased', leased_at=now() WHERE lease_token=<token>.
   c. For unfound tokens: UPDATE pending_rows SET status='staged'
      WHERE lease_token=<token>.  (POST never completed; retry on next lease.)
```

### 4.5 Upload drain

See §2.5 step-by-step.

## 5. Testing

### 5.1 Unit tests (no live catalog)

- `PendingRowsStore` schema creation, entry CRUD, status transitions.
- Directory rule registration: idempotent, parameter-drift rejection.
- `PendingRow.metadata` setter round-trips through SQLite.
- `record_class(table)` produces valid Pydantic model for every column
  type in `_map_column_type`.
- `record_class` schema disambiguation: ambiguous name raises.
- Handle FK coercion: typed record accepts handle, coerces to rid.
- Inter-table topological sort: DAG, multiple roots, cycle detection.
- Required-column validator identifies missing columns.
- Lazy-lease path with mocked server: unleased handle reads trigger
  batched POST; leased entries not re-leased.
- Lease reconciliation: simulated POST success + SQLite crash → on
  startup, lease is adopted.
- Serialization roundtrip for every type handled by `_json_default`.
- `handle.set_metadata(dict | callable | DataFrame)` batch path.
- `stage_row` with `dedupe_key`: second call returns existing handle;
  after upload, second call raises.
- `exe.list_pending` / `discard_pending` filter combinations.
- Empty batches: `stage_rows([])`, `stage_rows(empty_df)`.
- Large batches: `stage_rows(10000 records)` — throughput + memory.

### 5.2 Integration tests (live catalog)

- `stage_row` inserts a Subject; server returns matching RID.
- `stage_rows(df)` bulk-inserts 100 rows via DataFrame path.
- Intra-batch FK: stage Subject + Image referencing subject handle;
  upload; verify server-side Image.Subject = Subject.RID.
- `asset_directory_path` zero-ceremony: register dir of 10 files; upload
  fails with clear missing-column message when the asset table needs
  more; succeeds when it doesn't.
- Per-file metadata via `handle.set_metadata(df)`.
- `handle.register()` incremental + `handle.scan()` final sweep.
- Parameter-drift across sessions raises.
- Stale-schema: stage a row, add a required column to the catalog
  table, `upload_execution_outputs` surfaces the new required column
  in re-validation error.
- Inter-table cycle: stage rows A→B, B→A → `DerivaMLCycleError`.
- Partial failure best-effort mode: 1 of 10 FK-invalid → 9 upload, 1
  marked failed, no abort.
- Partial failure strict mode: same → level aborts, partial inserts
  remain, `list_pending(status='failed')` shows the failure.
- `retry_failed()` happy path.
- Lease timeout: stub server to delay POST; staging completes offline,
  lease acquired when `.rid` read or upload starts.
- File deleted between `register` and upload: per-row upload failure.
- Symlinked source directory: `scan()` follows symlinks consistently.

### 5.3 Regression

- Existing `asset_file_path` tests pass unchanged.
- Existing `upload_execution_outputs` asset-upload tests pass unchanged.
- Existing `ManifestStore` tests updated for renamed class + new columns
  (non-breaking).

## 6. Open questions and known limits

### 6.1 Known V1 limits

- **Updates / deletes:** insert-only. Full upload-back is a separate
  future spec.
- **Intra-table FK cycles:** unsupported. Users must split across
  batches.
- **Binary columns (`bytea`):** unsupported; `stage_row` raises
  `NotImplementedError`. Asset-table routes for binary payloads are the
  supported path.
- **CSV-based insert roundtrip:** see §2.6 risk-column list. V2 can
  swap to direct `pathBuilder` insert without public-API change.
- **Orphaned server-side leases:** possible only if client loses its
  entire workspace state after a successful lease POST. Not reconcilable
  from the client side in V1; server-side GC is an ERMrest concern.

### 6.2 Deferred

- **Fire-and-forget directory alias.** `exe.log_asset_directory(name,
  dir)` that calls `asset_directory_path(...).close()` in one shot,
  mirroring MLflow `log_artifacts`. Add if the call pattern proves
  common in practice.
- **Streaming upload.** `handle.flush()` to upload just one rule's
  pending rows (instead of the whole execution). Useful for inference
  loops writing one file per second. Not V1.
- **Cross-execution pending-rows.** Today entries are per-execution. A
  workspace-level "staging area" that persists across execution
  contexts is a future consideration.
- **Workspace-scoped `dedupe_key`.** V1's dedupe is per-execution; a
  cross-execution dedupe key would let a re-run across sessions be
  idempotent.
