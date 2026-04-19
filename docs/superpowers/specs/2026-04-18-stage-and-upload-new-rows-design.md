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
  target table does not exist, `ml.table(name)` / `exe.table(name)`
  raises `DerivaMLTableNotFound` at call time.
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

### 2.4 Public API — the `TableHandle` abstraction

The row-level entry point is a **`TableHandle`** object — a DerivaML
wrapper over a catalog table that unifies every row-level operation
(record typing, live insert, deferred stage, asset-file registration,
pending-row inspection) on one object. Users learn one abstraction and
one naming pattern; live vs deferred vs asset-attached differ only in
the verb they call.

#### 2.4.1 Entry points

```python
ml.table(
    table_name: str,
    *,
    schema: str | None = None,          # required if table name is ambiguous
) -> TableHandle | AssetTableHandle

exe.table(
    table_name: str,
    *,
    schema: str | None = None,
) -> TableHandle | AssetTableHandle      # same object, execution-scoped
```

- `ml.table(name)` returns a handle bound to the DerivaML instance. Any
  `stage(...)` or `asset_*` calls on it create pending rows at
  **workspace level** — scoped to the DerivaML / workspace but not to
  any specific execution. Useful for catalog-bootstrap workflows:
  adding vocabulary terms, bulk-inserting Subjects, seeding reference
  data. Unscoped pending rows are drained via
  `ml.workspace.flush_pending()` (see §2.4.6).

- `exe.table(name)` returns a handle bound to the current execution.
  Any `stage(...)` or `asset_*` calls record pending rows scoped to
  `execution_rid`; they drain as part of `upload_execution_outputs()`
  and pick up execution-provenance columns automatically (e.g.
  `RCB=current_user`, `Execution=<rid>` for association tables that
  track provenance).

- The handle **subclass** is determined at lookup time:
  - **`AssetTableHandle`** if the table is an asset table — adds
    `.asset_file(...)` and `.asset_directory(...)` methods.
  - **`TableHandle`** otherwise — plain rows only.

- **Schema disambiguation.** If `table_name` appears in multiple
  schemas and `schema` is not supplied, raise `DerivaMLTableNotFound`
  listing the candidates — matches today's `name_to_table`.

#### 2.4.2 `TableHandle` surface

```python
class TableHandle:
    schema: str
    name: str

    # ----- typed records -----

    def record_class(self, *, include_system_columns: bool = False) -> type[RowRecord]:
        """Return a dynamically-generated Pydantic model for this table.
        Fields are derived from catalog column metadata; non-nullable
        columns are required, nullable are Optional with catalog default,
        ``extra='forbid'`` catches typos at construction time."""

    # ----- live insert -----

    def insert(
        self,
        records: Iterable[RowRecord | dict] | "pd.DataFrame",
        *,
        defaults_ok: bool = True,
        chunk_size: int = 1000,
    ) -> list[str]:
        """Insert rows into the catalog now. Returns the assigned RIDs.
        This is a thin wrapper over deriva-py's pathBuilder ``insert`` so
        live and deferred inserts can be found side-by-side on the same
        object. Accepts the same input shapes as ``stage``."""

    # ----- deferred stage -----

    def stage(
        self,
        records: RowRecord | dict | Iterable[RowRecord | dict] | "pd.DataFrame",
        *,
        dedupe_key: str | Callable[[dict], str] | None = None,
        description: str | None = None,
        chunk_size: int = 1000,
        progress: Callable[[int, int], None] | None = None,
    ) -> PendingRow | list[PendingRow]:
        """Record one or more pending rows in the workspace SQLite. RID
        assignment is deferred (§2.2). Returns a single ``PendingRow`` if
        ``records`` is scalar; a list if iterable or DataFrame.

        Validates every record through ``self.record_class()``. See
        §2.4.4 for the typed-record contract, §2.2 for lease lifecycle,
        and §2.4.5 for dedupe_key semantics."""

    # ----- inspection / discard -----

    def pending(
        self,
        *,
        status: str | list[str] | None = None,
    ) -> list[PendingRow]:
        """Pending rows for this table (filtered by status if given)."""

    def discard_pending(
        self,
        *,
        rows: Iterable[PendingRow] | None = None,
        status: str | list[str] | None = None,
    ) -> int:
        """Remove pending rows matching the filter. Returns count."""
```

#### 2.4.3 `AssetTableHandle` surface (subclass)

```python
class AssetTableHandle(TableHandle):

    def asset_file(
        self,
        file_name: str | Path,
        metadata: RowRecord | dict | None = None,
        *,
        asset_types: list[str] | str | None = None,
        copy_file: bool = False,
        rename_file: str | None = None,
        description: str | None = None,
        **kwargs,            # legacy per-column metadata kwargs
    ) -> AssetFilePath:
        """Register a single file for upload. Returns an AssetFilePath
        bound to a pending row (same lifecycle as stage). Signature
        matches today's ``exe.asset_file_path`` minus the leading table-
        name positional argument, which is implicit in the handle.

        ``metadata=`` / ``**kwargs`` are validated through
        ``self.record_class()``."""

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
        kwargs — zero-ceremony by default; per-file metadata is set via
        the returned handle's ``set_metadata(...)`` batch API or
        AssetFilePath.metadata setter. See §2.4.7."""
```

#### 2.4.4 Records and validation

Typed records are obtained from the handle:

```python
Subject = ml.table("Subject").record_class()
record = Subject(Name="Alice", Species="Human")   # validates here
ml.table("Subject").stage(record)                  # stores validated record
```

`RowRecord` is a `BaseModel` subclass with fields derived from the
catalog's column metadata. The existing `_map_column_type` helper for
ERMrest → Python types is reused.

**`AssetRecord` is a `RowRecord` subclass** with a column filter
restricting to asset-metadata columns. An `AssetTableHandle` exposes
both:
- `.record_class()` — full schema, all non-system columns
- `.record_class(columns="asset_metadata")` — same filter as today's
  `ml.asset_record_class(table)`

Today's `ml.asset_record_class(table)` is retained as a thin alias
delegating to `ml.table(table).record_class(columns="asset_metadata")`.

**`stage` accepts three input shapes:**
1. A `RowRecord` instance (preferred; already validated).
2. A `dict` — passed through the typed record constructor; same
   validation result.
3. A `PendingRow` / `AssetFilePath` **handle** as a column value inside
   a dict or record — coerced to the handle's RID, triggering lazy
   lease (§2.2). Handle references across the same `stage` call are
   coalesced into one batched lease POST.

#### 2.4.5 Idempotency via `dedupe_key`

Optional kwarg on `stage(...)`. If provided, the entry's key column in
SQLite is set to the dedupe key instead of an auto-hash. Subsequent
`stage` calls with a matching key (same scope: `(execution_rid OR
None, schema, table)`) resolve as:

- Existing `staged | leasing | leased` → return existing handle.
- Existing `uploading | uploaded` → raise `DerivaMLException`.
- Existing `failed` → update metadata, reset to `staged`.

Notebook re-runs with explicit dedupe keys become idempotent.
`dedupe_key` can be a string (for `stage(record)`) or a callable
`(dict) -> str` (for `stage(records_iter)`).

#### 2.4.6 Workspace- vs execution-scoped pending rows

- `ml.table(name).stage(...)` — pending rows have `execution_rid=None`.
  They drain via `ml.workspace.flush_pending(mode=...)`. No execution-
  provenance columns auto-filled.
- `exe.table(name).stage(...)` — pending rows have
  `execution_rid=<current>`. They drain as part of
  `upload_execution_outputs(...)`. Execution-provenance columns
  auto-filled for association tables that track them.

Both share the same `pending_rows` SQLite table, the same drain logic
(§2.5), and the same RID-leasing path (§2.2). The scope is purely a
filter on which entries are drained by which call.

`ml.workspace.flush_pending(mode="strict" | "best_effort")` drains
workspace-scoped entries (entries with `execution_rid=None`). It does
not touch execution-scoped entries.

#### 2.4.7 Asset directory handle

Unchanged from revision 2's design, but rooted in the new abstraction:

```python
class AssetDirectoryHandle:
    path: Path
    rule_id: int
    table: AssetTableHandle       # the handle this rule belongs to

    def register(self, file: str | Path) -> AssetFilePath:
        """Eagerly register a specific file."""

    def scan(self) -> list[AssetFilePath]:
        """Walk source_dir, register new files. Idempotent."""

    def pending(self) -> list[AssetFilePath]:
        """Pending entries under this rule."""

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
        - DataFrame — indexed by filename (``filename_col`` identifies
          the column to match, or the index if already the filename).
          Remaining columns become the file's metadata.
        Each resolved value is validated through the table's record_class."""

    def close(self) -> None:
        """Final scan, mark rule status=closed."""
```

**Deleted files between scans.** `scan()` does not auto-remove pending
rows whose files no longer exist. At upload time, a missing file for
an asset-row entry surfaces as a per-row failure with a clear `error`
message.

**Idempotent re-registration across sessions.** Keyed by `(scope,
target_table, source_dir)` where `scope` is either `execution_rid` or
the literal `workspace`. Identical parameters → return existing
handle. Any parameter differs → raise `DerivaMLException` with the
diff.

#### 2.4.8 Workflow examples

**Live insert of a single row:**
```python
ml.table("Subject").insert([{"Name": "Alice"}])       # returns [rid]
```

**Stage a batch of rows from a DataFrame:**
```python
exe.table("Prediction").stage(predictions_df)         # deferred
```

**Stage a row referencing another staged row via handle coercion:**
```python
subj = exe.table("Subject").stage({"Name": "Alice"})
exe.table("Image").asset_file("scan.jpg", Subject=subj, Acquisition_Date="...")
```

**Zero-ceremony asset directory upload:**
```python
dir_handle = exe.table("Image").asset_directory("/outputs/scans")
# ... files accumulate ...
exe.upload_execution_outputs()
```

**Per-file metadata via batch setter:**
```python
dir_handle = exe.table("Image").asset_directory("/outputs/scans")
dir_handle.set_metadata(labels_df)   # labels_df indexed by filename
```

**Bulk vocabulary bootstrap (workspace-scoped, no execution):**
```python
asset_types = ml.table("Asset_Type")
asset_types.stage([{"Name": f"Label_{i}"} for i in range(100)])
ml.workspace.flush_pending()
```

**Triage after a failed upload:**
```python
for row in ml.table("Subject").pending(status="failed"):
    print(row.error, row.metadata)
    ml.table("Subject").discard_pending(rows=[row])
```

#### 2.4.9 Backward-compatibility shims

Existing public methods remain callable and delegate to the new
abstraction. No external caller needs to change to ship this.

```python
# exe.asset_file_path(asset_name, file, ...) — existing
# becomes equivalent to:
exe.table(asset_name).asset_file(file, ...)

# ml.asset_record_class(table) — existing
# becomes equivalent to:
ml.table(table).record_class(columns="asset_metadata")
```

The shim layer is a half-dozen lines; the real work lives on the new
handles. Once the handle API is documented and callers have migrated,
the shims can be deprecated in a future release. For V1 they stay with
a deprecation-documentation-only mark (no runtime warning).

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

**Inspection & triage.** `handle.pending(status='failed')` on any
`TableHandle` returns failed entries with their `error` messages; or
`ml.workspace.list_pending(status='failed')` across all tables.
Users edit metadata or fix referenced data, then call
`exe.retry_failed()` or `handle.discard_pending(...)`.

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
| `exe.asset_file_path(...)` signature | Unchanged (shim delegates to handle) |
| `ml.asset_record_class(...)` signature | Unchanged (shim delegates to handle) |
| `AssetFilePath` class | Extended with lazy-lease `rid` |
| Workspace SQLite (WAL) | Unchanged |
| `ERMrest_RID_Lease` server table | Accessed for leasing |
| `ml.table(name)` / `exe.table(name)` | **New** entry point |
| `TableHandle` / `AssetTableHandle` | **New** row-level abstraction |
| `TableHandle.insert(...)` | **New** (pathBuilder passthrough) |
| `TableHandle.stage(...)` | **New** (DataFrame-native, deferred) |
| `TableHandle.record_class(...)` | **New** (replaces `ml.record_class`) |
| `AssetTableHandle.asset_file(...)` | **New** (replaces `exe.asset_file_path`) |
| `AssetTableHandle.asset_directory(...)` | **New** |
| `AssetDirectoryHandle` + `set_metadata(...)` | **New** |
| `RowRecord` base class | **New** |
| `handle.pending(...)` / `discard_pending(...)` | **New** per-table triage |
| `ml.workspace.list_pending(...)` | **New** cross-table triage |
| `ml.workspace.flush_pending(...)` | **New** workspace-scoped drain |
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

### 4.1 `ml.table(name)` / `exe.table(name)` handle construction

```
1. Resolve name against model using name_to_table(name, schema=schema).
   - Raise DerivaMLTableNotFound if absent.
   - Raise DerivaMLTableNotFound (ambiguous) if multiple schemas match
     and schema= was not supplied.
2. If the table is an asset table (model.is_asset(table)):
     return AssetTableHandle(ml, table, scope=<execution_rid or None>).
   Else:
     return TableHandle(ml, table, scope=<execution_rid or None>).
3. Handles are lightweight: hold a DerivaML/Execution reference, a
   Table reference, and the scope. All row-level state lives in SQLite.
```

### 4.2 `TableHandle.stage(records, dedupe_key=...)`

```
1. Normalize records:
   - If DataFrame: convert via to_dict(orient='records').
   - If single RowRecord / dict: wrap in a list (scalar input).
   - If Iterable: materialize per chunk (see below).
2. For chunk of `chunk_size` records:
   a. For each rec:
      - If RowRecord: pass through (already validated).
      - If dict: rec = self.record_class()(**rec)  (validates, raises).
      - Handle-valued columns in rec remain as handles; coercion
        happens when handle.__get_pydantic_core_schema__ resolves
        during record construction or when writing to SQLite.
   b. If dedupe_key given: resolve per §2.4.5. Existing matching
      entries short-circuit into the return list.
   c. Generate lease_token = uuid4() for each new entry.
   d. BEGIN TRANSACTION.
   e. Insert N pending_rows entries: scope=self.scope (execution_rid
      or NULL), target_schema, target_table, key=<hash or dedupe_key>,
      rid=NULL, lease_token, metadata_json serialized per §2.3.1,
      status='staged', created_at=now().
   f. COMMIT.
   g. If progress callback: progress(done_so_far, total).
3. Return PendingRow (scalar input) or list[PendingRow] (iterable/DF).
```

### 4.3 `AssetTableHandle.asset_file(file, metadata=...)` /
### `AssetTableHandle.asset_directory(dir, ...)`

`asset_file` is equivalent to `stage` with an attached file path:

```
1. Validate file_name, determine staging path (existing logic from
   today's asset_file_path).
2. Symlink or copy file into assets/{AssetTable}/.
3. Construct metadata record via self.record_class(
      columns="asset_metadata")(**{metadata_dict, **kwargs}).
4. Generate lease_token.
5. Write pending_rows entry with asset_file_path set, rule_id=NULL,
   status='staged'. Same schema as §4.2.
6. Return AssetFilePath bound to the entry.
```

`asset_directory` creates a directory_rules row and returns a handle;
per-file registration is lazy (via register()/scan()) and follows §4.2
with the additional `rule_id` linkage.

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

Handle construction and dispatch:
- `ml.table(name)` returns `TableHandle` for plain tables,
  `AssetTableHandle` for asset tables.
- `ml.table(name, schema=s)` resolves ambiguous names; missing
  `schema=` for ambiguous names raises `DerivaMLTableNotFound`.
- `exe.table(name).scope == execution_rid`; `ml.table(name).scope is None`.
- `TableHandle` does not expose `asset_file` / `asset_directory` (only
  `AssetTableHandle` does).

Store and record:
- `PendingRowsStore` schema creation, entry CRUD, status transitions.
- Directory rule registration: idempotent, parameter-drift rejection.
- `TableHandle.record_class()` produces valid Pydantic model for every
  column type in `_map_column_type`.
- `AssetTableHandle.record_class(columns="asset_metadata")` filters to
  asset-metadata columns only; base class otherwise identical.

Stage and validation:
- `stage(dict)` validates through record_class and raises on missing
  required, unknown fields, type mismatch.
- `stage(record)` passes through without re-validation (already typed).
- `stage(df)` converts and preserves order.
- Empty inputs: `stage([])`, `stage(empty_df)` succeed as no-ops.
- Large batches: `stage(10000_record_iterable)` — chunked, progress
  callback fires, memory stays bounded.

Handle FK coercion:
- Handle passed as column value in another `stage` call → coerces to
  `.rid`, triggers batched lease POST.
- N handles in one `stage(batch)` call coalesce to one lease POST.
- Coerced handles materialize RID once; subsequent reads return cached
  value from SQLite.

dedupe_key:
- `stage(record, dedupe_key="k")` twice → second call returns existing
  handle unchanged.
- After upload, `stage(record, dedupe_key="k")` raises.
- `stage(rec, dedupe_key="k")` on a `status='failed'` entry resets to
  `status='staged'` and updates metadata.

Lazy lease + reconciliation:
- Lazy-lease path with mocked server: unleased handle `.rid` read
  triggers batched POST; already-leased entries not re-leased.
- Two-phase lease: SQLite `status='leasing'` written before POST; on
  POST failure (exception), status reverts to `'staged'`.
- Startup reconciliation: simulated POST success + crash before SQLite
  update → on workspace open, lease query adopts the RID.

Inter-table dependencies:
- Topological sort: DAG, multiple roots, single root with chain.
- Cycle detection: inter-table cycle raises `DerivaMLCycleError` with
  cycle listed.

Directory handle:
- `asset_directory(dir).register(file)` — eager.
- `asset_directory(dir).scan()` — idempotent, only new files registered.
- `set_metadata(dict | callable | DataFrame)` batch path; one SQLite
  transaction; each resolved value validated through record_class.
- Deleted files between scans: not auto-removed; upload surfaces as
  per-row failure.
- Symlinked source directory: `scan()` follows symlinks consistently.

Serialization:
- Roundtrip for every type handled by `AssetManifest._json_default`
  (datetime, date, Path, dict, Pydantic BaseModel).
- `bytea` column in record → `stage()` raises `NotImplementedError`.

Inspection / discard:
- `handle.pending(status=...)` filter combinations.
- `handle.discard_pending(...)` removes entries; server-side leases
  NOT reclaimed (acknowledged limit).
- Cross-table inspection via `ml.workspace.list_pending(...)`.

Compatibility shims:
- `exe.asset_file_path(asset_name, file, ...)` delegates to
  `exe.table(asset_name).asset_file(file, ...)`; behavior identical.
- `ml.asset_record_class(table)` delegates to
  `ml.table(table).record_class(columns="asset_metadata")`; same class.

### 5.2 Integration tests (live catalog)

- `ml.table("Subject").insert([{...}])` inserts and returns RIDs (thin
  pathBuilder passthrough).
- `exe.table("Subject").stage({...})` + `upload_execution_outputs()`
  inserts with matching RID after upload.
- `exe.table("Prediction").stage(df)` bulk-inserts 100 rows via
  DataFrame path.
- Intra-batch FK: stage Subject, stage Image referencing subject
  handle; upload; verify server-side Image.Subject = Subject.RID.
- `exe.table("Image").asset_directory("/outputs/scans")` zero-ceremony:
  register 10 files; upload fails with clear missing-column message
  when Image requires user-supplied columns, succeeds when it doesn't.
- Per-file metadata via `dir_handle.set_metadata(df)`.
- Workspace-scoped stage: `ml.table("Asset_Type").stage([...])` +
  `ml.workspace.flush_pending()` — rows inserted without an execution.
- Parameter-drift across sessions raises.
- Stale-schema: stage a row, server adds a required column, upload
  surfaces the new required column in re-validation error.
- Inter-table cycle: stage rows A→B, B→A → `DerivaMLCycleError`.
- Partial failure best-effort mode: 1 of 10 FK-invalid → 9 upload, 1
  marked failed, no abort.
- Partial failure strict mode: same → level aborts, partial inserts
  remain, `handle.pending(status='failed')` shows the failure.
- `exe.retry_failed()` happy path.
- Lease timeout: stub server to delay POST; staging completes offline,
  lease acquired when `.rid` read or upload starts.
- File deleted between `register` and upload: per-row upload failure
  with `FileNotFoundError` in `error` message.
- Execution-scoped vs workspace-scoped dedupe_key: same key in
  different scopes does not collide.

### 5.3 Regression

- Existing `exe.asset_file_path` tests pass unchanged (via shim).
- Existing `ml.asset_record_class` tests pass unchanged (via shim).
- Existing `upload_execution_outputs` asset-upload tests pass
  unchanged.
- Existing `ManifestStore` tests updated for renamed class + new
  columns (non-breaking).

## 6. Open questions and known limits

### 6.1 Known V1 limits

- **Updates / deletes:** insert-only. Full upload-back is a separate
  future spec.
- **Intra-table FK cycles:** unsupported. Users must split across
  batches.
- **Binary columns (`bytea`):** unsupported; `TableHandle.stage(...)`
  raises `NotImplementedError`. Asset-table routes for binary payloads
  are the supported path.
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
