# Stage & upload new rows — design spec

**Status:** Draft, pending review.
**Date:** 2026-04-18.
**Scope:** Design spec. Implementation plan follows separately.

## 1. Goals and scope

### 1.1 Goals

Generalize the existing `upload_assets` pathway so the same pipeline uploads:

1. **Asset rows** — a file in Hatrac plus a row in an asset table (today's
   `asset_file_path` behavior).
2. **Plain new rows** — row inserts into any catalog table, no file attached.
   Enables bulk annotation workflows, bulk dataset population, and any other
   "add N new records to table T" use case.
3. **Directory-registered asset rows** — one call registers every file in a
   local directory as a pending asset upload. Files may trickle in over time.

### 1.2 Non-goals

- **Updates or deletes.** V1 is insert-only. Modifying existing catalog rows
  and deleting them are separate design problems and out of scope.
- **Full slice upload-back.** The broader "sync arbitrary local changes to
  catalog" problem (Phase 1 §1.1 goal #4, §5 deferred) remains deferred. This
  spec lays the groundwork but only covers new-row inserts.
- **Offline editing across disconnected sessions.** Same stance as the
  Phase 1 spec: target "work locally while connected."
- **Filesystem watching.** Directory registration does NOT watch the source
  directory for new files. File arrival is surfaced via explicit
  `register(file)` calls or scan-at-upload-time; no inotify/fsevents
  dependency.
- **Auto-generating metadata from sidecar files.** If a user wants per-file
  metadata, they iterate the directory handle and call the existing
  `AssetFilePath.metadata` setter on each path. See §2.4.3 for the
  intended usage pattern.

### 1.3 Migration posture

Additive. `ManifestStore` is generalized into a single store tracking both
asset-row and plain-row pending writes; existing `AssetEntry` state
continues to work unchanged through the generalized API. Existing
`asset_file_path` and `upload_assets` / `upload_execution_outputs` keep
their signatures and behavior; directory registration and plain-row staging
are new surface area.

## 2. Architecture

### 2.1 One store, one upload drain

Today the workspace holds one per-execution `ManifestStore` with
`AssetEntry` rows describing pending/uploaded/failed asset uploads. Every
call to `asset_file_path(...)` appends an `AssetEntry`; upload walks the
entries, uploads files to Hatrac, inserts rows into the asset table.

The generalization:

- **Rename conceptually:** `ManifestStore` → `PendingRowsStore`. The SQL
  schema gains a couple of columns; the class gains a couple of methods.
  Existing behavior is preserved.
- **Entry shape:** a pending row has `target_schema`, `target_table`,
  `metadata` (the column values), and an **optional** `asset_file_path`
  pointing at a local file. When the file pointer is set, the entry is an
  asset upload (file → Hatrac → row with URL/MD5 patched in). When it's
  null, the entry is a plain-row insert.
- **One upload drain:** `upload_execution_outputs` iterates pending entries,
  handles file-attached entries via the existing Hatrac + row-insert pipe,
  handles file-less entries via direct row insert. Order is determined by
  FK dependencies (see §2.5).

### 2.2 RID leasing at stage time

Every call that creates a pending row (asset or plain) immediately reserves
a RID by inserting into `public:ERMrest_RID_Lease` on the server. The
leased RID is stored on the pending entry from the moment of registration.

This eliminates the need for client-side tokens, topological sorting for
RID assignment, or post-upload patching:

- Users see a real, server-assigned RID the moment they stage a row.
- Intra-batch FK references work by passing the leased RID as a string —
  e.g. `exe.stage_row("Image", {"Subject": subj_rid, ...})`.
- Upload still requires FK dependency ordering so that when row A has
  `FK_col = <B's leased RID>`, B is inserted before A. But the RID
  allocation step is fully decoupled from upload ordering.

Lease allocation is a single POST per batch of new entries (ERMrest
accepts a list), so N new rows cost one extra server round-trip, not N.

#### 2.2.1 Lease hygiene

Leased RIDs that never upload are orphaned in `ERMrest_RID_Lease`. V1
disposition:

- No automatic cleanup. Leases are cheap.
- Provide `PendingRowsStore.discard(execution_rid)` that drops all pending
  rows for an execution. Users who abandon batches deliberately can call it.
- Server-side garbage-collection of stale leases is a separate ERMrest
  concern, not this spec's.

### 2.3 SQLite schema

The generalized store has two tables in the workspace SQLite:

**`pending_rows`** — one row per pending catalog-row insert.

| Column | Purpose |
|---|---|
| `id` | integer primary key, internal |
| `execution_rid` | scope |
| `key` | `"{target_table}/{file_name}"` for asset rows, or a stable hash for plain rows |
| `target_schema` | catalog schema for insert |
| `target_table` | catalog table for insert |
| `rid` | leased RID (set at staging time) |
| `metadata_json` | column values as JSON |
| `asset_file_path` | local file path, or null for plain rows |
| `asset_types_json` | vocabulary terms, or null for plain rows |
| `description` | optional |
| `status` | `leased | uploading | uploaded | failed` |
| `error` | set on failure |
| `leased_at` | timestamp |
| `uploaded_at` | timestamp |
| `rule_id` | fk to `directory_rules.id`, null for non-directory rows |

**`directory_rules`** — one row per registered directory.

| Column | Purpose |
|---|---|
| `id` | integer primary key, referenced by `pending_rows.rule_id` |
| `execution_rid` | scope |
| `target_schema` | catalog schema for each file |
| `target_table` | catalog table for each file |
| `source_dir` | absolute path to the directory on disk |
| `glob` | file selection glob, default `"*"` |
| `recurse` | bool |
| `copy_files` | bool |
| `asset_types_json` | asset type terms applied to every registered file |
| `status` | `active | closed` |
| `created_at` | timestamp |

Indexes: `(execution_rid, status)` on both; `(execution_rid, target_table)`
on `pending_rows` for upload-drain lookup.

### 2.4 Public API

#### 2.4.1 Plain row staging

```python
exe.stage_row(
    target_table: str,                  # e.g. "Subject"
    metadata: dict | AssetRecord,       # the column values
    *,
    description: str | None = None,
) -> PendingRow
```

Leases a RID, validates required columns, creates a pending entry with
`asset_file_path=None`. Returns a lightweight `PendingRow` handle carrying
`.rid`, `.metadata`, `.status`. Metadata can be updated via the handle's
`.metadata` setter before upload (same write-through semantics as
`AssetFilePath.metadata`).

For bulk staging:

```python
exe.stage_rows(
    target_table: str,
    records: Iterable[dict | AssetRecord],
) -> list[PendingRow]
```

Convenience that loops and batches the lease request.

#### 2.4.2 Asset file registration (unchanged existing API)

```python
exe.asset_file_path(
    asset_name, file_name, ...,
    metadata=None, ...,
) -> AssetFilePath
```

Unchanged signature. Internally: lease a RID, register a pending entry
with `asset_file_path=<local path>` and `metadata=...`. Existing behavior
preserved.

#### 2.4.3 Asset directory registration (new)

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

Deliberately takes **no metadata kwargs.** The two intents:

- **Zero-ceremony.** Caller passes `asset_name` + `source_dir`. Every file
  in the directory becomes a pending row using only file-derived columns
  (MD5, URL, Length, Filename). If the asset table requires user-supplied
  columns, the upload-time validation surfaces an error listing which
  columns still need values and pointing the user at `.pending()` +
  `.metadata = {...}`.

- **Per-file metadata.** Caller registers the directory, then iterates:
  ```python
  for path in handle.pending():
      path.metadata = derive_metadata(path)
  ```
  Uses the existing `AssetFilePath.metadata` setter — write-through to
  SQLite, same contract users already know.

No `metadata=`, no `filename_pattern=`, no `metadata_fn=`, no sidecar
support. Reasons:
- Sidecar CSV would introduce a parallel metadata store and duplicate
  SQLite's role.
- `metadata_fn` callbacks don't survive process restarts, and the
  iterate-and-set loop is three lines of user code that's more explicit
  and greppable than a library callback.
- Keeping `asset_directory_path` scoped to "register every file as a
  pending row" makes the method's contract tight.

Returned handle:

```python
class AssetDirectoryHandle:
    path: Path                  # source_dir, for convenience
    rule_id: int
    asset_name: str

    def register(self, file: str | Path) -> AssetFilePath:
        """Eagerly register a specific file. Leases a RID, creates a
        pending row with no metadata beyond file-derived columns. Returns
        the AssetFilePath bound to the pending row — caller sets metadata
        via `.metadata = {...}` before upload if needed."""

    def scan(self) -> list[AssetFilePath]:
        """Walk source_dir (respecting glob / recurse), register any file
        not yet registered under this rule. Idempotent. Returns newly-
        registered paths only."""

    def pending(self) -> list[AssetFilePath]:
        """All AssetFilePath bound to pending rows under this rule.
        Includes rows created via register() and scan()."""

    def close(self) -> None:
        """Run a final scan, then mark the rule status=closed.
        Subsequent register() / scan() calls raise."""
```

#### 2.4.4 Idempotent re-registration across sessions

`asset_directory_path` is idempotent. Calling it again in a new session
with the same `(execution_rid, target_table, source_dir)` triple returns a
handle to the existing rule. If any other argument differs, the call
raises `DerivaMLException` listing what changed — the user has to either
close the old rule or update the source directory.

### 2.5 Upload drain

`Execution.upload_execution_outputs()` already walks the staging tree and
delegates to `upload_directory` / `GenericUploader` for the actual
ERMrest inserts. The new work sits in front of that existing pipeline —
the drain turns pending-rows SQLite state into staging-tree CSV files,
then lets the existing pipeline do its job. §2.6 describes the CSV
handoff in more detail.

Sequence at upload time:

1. **Final scan.** For every `active` directory rule in this execution,
   call `scan()` so files written since the last register-or-scan call
   are picked up as pending rows.
2. **Required-column validation.** For every pending row where
   `status=leased`, check that required columns are present in
   `metadata_json`. If not, either fail the whole upload (strict mode —
   default) or skip the offending row and continue (best-effort mode).
   List the offending `(table, key, missing_columns)` triples in the
   error or warning.
3. **FK topological sort at the table level.** Build a dependency graph
   over the target tables that have pending rows. Table A depends on
   table B iff some pending A-row has an FK column whose value equals
   some pending B-row's leased RID. (Intra-table cycles are not handled
   in V1 — they would require per-row ordering that the existing
   `GenericUploader` cannot deliver; see the note below. In practice
   most schemas are acyclic at the table level.) The topological order
   governs the sequence in which per-table CSVs are handed to
   `upload_directory` — we call `upload_directory` once per group of
   tables at the same topological depth (or once per table if we want
   strict ordering).

   **Caveat.** `GenericUploader` walks its input directory with
   `os.walk`, so within a single call it does not honor caller-supplied
   per-file ordering. FK ordering is enforced at the **granularity of
   separate `upload_directory` invocations**, not within one. V1
   consequence: plain-row CSVs are split per table and uploaded in
   topological order table-by-table; asset files (whose row inserts are
   driven by the asset upload spec's `target_table`) are grouped
   likewise. Intra-table FK cycles (a Subject whose `Parent_Subject`
   points at another new Subject) are not supported in V1 — the user
   must upload the referenced Subject in a separate prior batch.
4. **Materialize staging tree.** For each pending row in topo order:
   - **Asset rows** (`asset_file_path` non-null): the asset file already
     lives at `assets/{AssetTable}/{file_name}`; the drain ensures a
     symlink or copy exists at the metadata-path layout the existing
     asset upload spec expects
     (`{exec_dir}/asset/{schema}/{asset_table}/{v1}/{v2}/.../{file_name}`).
     Mark `status=uploading`.
   - **Plain rows** (`asset_file_path` null): append a CSV row to
     `{exec_dir}/table/{target_schema}/{target_table}/{target_table}.csv`
     grouped by target table. Mark `status=uploading`.
5. **Delegate to `upload_directory`.** The existing pipeline uploads
   files to Hatrac (for asset rows, patching in URL/MD5/Length/Filename
   per the upload spec's column_map) and inserts CSV-driven rows into
   the catalog. Progress callbacks, chunked uploads, and retry behavior
   are unchanged.
6. **Reconcile status.** After `upload_directory` returns its
   per-path `FileUploadState` dict, update each pending row:
   - `UploadState.success` → `status=uploaded`, `uploaded_at=now()`.
   - `UploadState.failure` → `status=failed`, `error=<message>`. In
     strict mode the drain raises after recording all failures; in
     best-effort mode it returns a summary.

### 2.6 CSV handoff to GenericUploader

The existing `upload_directory` / `GenericUploader` pipeline consumes files
from the staging tree (`deriva-ml/execution/{rid}/table/{schema}/{table}/{table}.csv`
for plain rows, `deriva-ml/execution/{rid}/asset/...` for asset files).
The upload spec already has a `"asset_type": "table"` entry that matches
`table_regex` and inserts CSV rows.

V1 keeps this pipeline as the actual insert mechanism. The upload drain
described in §2.5 serializes pending-rows SQLite state into the staging
tree CSVs immediately before calling `upload_directory`:

- Plain rows → written to `{exec_dir}/table/{schema}/{table}/{table}.csv`
  grouped by target table.
- Asset rows → the asset file already lives in
  `assets/{AssetTable}/{file_name}`; metadata goes into the same CSV
  layout the existing asset upload spec expects.
- After `upload_directory` returns, pending-rows status is updated based
  on the per-file `FileUploadState` results.

This keeps V1 changes concentrated in the staging / manifest layer, leaves
the actual ERMrest insert path unchanged, and reuses the retry / progress-
callback machinery of `GenericUploader`.

### 2.7 Error handling and recovery

**Failure semantics:**

- **Strict mode (default).** First failure (missing column, Hatrac upload
  error, FK violation, etc.) aborts the batch. Pending entries that were
  not yet attempted keep `status=leased` so the user can fix and retry.
  Entries that failed keep `status=failed` with `error` set.
- **Best-effort mode.** Each pending row attempts independently. Failures
  are recorded but don't abort siblings. Useful for large batches where
  partial success is acceptable.

**Restart semantics.** If a process dies mid-upload, the workspace SQLite
state is what it is — pending-rows entries hold their last-written status.
On next `upload_execution_outputs()` call, rows with `status=leased` or
`status=failed` are retried; `status=uploaded` are skipped. Users can
also call `exe.retry_failed()` to force-retry only the failed entries.

**Lease validity.** A leased RID stays valid indefinitely on the server.
There's no "lease expiration" to worry about across process restarts.

### 2.8 Relationship to workspace and executions

- Pending-rows are scoped by `execution_rid`. They live in the workspace
  SQLite under the same execution context as today's `ManifestStore`.
- Two executions in the same workspace see each other's leased RIDs
  (they're real server-side RIDs) but not each other's pending-row
  entries — scope is enforced at the query layer.
- `Workspace.manifest_store()` returns an instance bound to a specific
  `execution_rid`, same as today. The V1 generalization widens the
  returned object's API surface; the outer Workspace API is unchanged.

## 3. Details worth calling out

### 3.1 `PendingRow` vs `AssetFilePath`

`AssetFilePath` is a `Path` subclass and continues to serve its current
role: the object returned from `asset_file_path(...)` that the user writes
file bytes to. For plain rows, there's no file — so `stage_row` returns a
different, non-Path object (`PendingRow`). They share a common interface
for metadata read/write and status inspection:

```python
class _PendingHandle(Protocol):
    rid: str
    status: str
    metadata: dict[str, Any]          # read; setter updates SQLite
    error: str | None

class PendingRow(_PendingHandle): ...
class AssetFilePath(Path, _PendingHandle): ...
```

Internally, both wrap a `pending_rows` SQLite row and route mutations
through the same write-through path.

### 3.2 What exists today vs. what's new

| Component | Status |
|---|---|
| `ManifestStore` with `AssetEntry` | Exists — generalize SQL schema |
| `GenericUploader` with `table_regex` CSV insert | Exists — reuse as-is |
| `upload_directory` drains staging tree | Exists — reuse as-is |
| `asset_file_path(...)` single-file API | Exists — unchanged signature |
| Workspace SQLite with WAL | Exists — no changes needed |
| `ERMrest_RID_Lease` server table | Exists — POST to allocate |
| Generalize store: `pending_rows` table + `directory_rules` table | **New** |
| `exe.stage_row` / `exe.stage_rows` | **New** |
| `exe.asset_directory_path` + `AssetDirectoryHandle` | **New** |
| RID leasing at stage time | **New** |
| Final scan + FK topological sort in upload drain | **New** |
| `PendingRow` handle class | **New** |

### 3.3 Validation at registration vs. at upload

Two checks, two timings:

- **At registration time:** the asset table / target table must exist and
  be an asset table (for `asset_file_path` / `asset_directory_path`) or a
  domain table (for `stage_row`). Asset types, if supplied, must be valid
  vocabulary terms. Fails immediately.
- **At upload time:** required columns must have values, FKs must resolve
  to valid RIDs. Per §2.5, surfaced in strict mode as an abort, in
  best-effort mode as per-row failures.

Why the split: registration can succeed with incomplete metadata — the
user may be planning to fill it in via `.metadata = {...}` later, or via
`dir_handle.pending()` iteration. Refusing registration early would block
the zero-ceremony directory case where metadata arrives post-registration.

### 3.4 Directory-rule parameter changes across sessions

Rules are keyed by `(execution_rid, target_table, source_dir)`. When a
user calls `asset_directory_path` with the same triple:

- If all other parameters match — return existing handle.
- If any parameter differs (glob, recurse, copy_files, asset_types) —
  raise `DerivaMLException` with a diff: `"Directory rule for {table} at
  {source_dir} was previously registered with glob='*.jpg'; you passed
  glob='*.png'. Close the old rule first or use the original parameters."`

Prevents silent configuration drift.

## 4. Algorithm

### 4.1 `exe.stage_row(table, metadata)`

```
1. Validate `table` exists in model; resolve schema.
2. Lease a RID: POST /entity/public:ERMrest_RID_Lease -> rid
3. Create pending_rows entry:
     execution_rid=self.execution_rid,
     target_schema=schema, target_table=table,
     rid=rid, metadata_json=json(metadata),
     asset_file_path=None, asset_types_json=None,
     status='leased', leased_at=now()
4. Return PendingRow(rid=rid, ...).
```

### 4.2 `exe.asset_directory_path(asset_name, source_dir, ...)`

```
1. Validate asset_name is an asset table in model.
2. Validate asset_types, if any.
3. Canonicalize source_dir to absolute path.
4. Query directory_rules for (execution_rid, target_table, source_dir):
   - Existing match with same params -> return existing handle.
   - Existing match with different params -> raise.
   - No match -> INSERT new row, get rule_id.
5. Return AssetDirectoryHandle(rule_id=..., path=source_dir, ...).
```

### 4.3 `handle.register(file)`

```
1. Assert the rule's status is 'active'.
2. Check if file is already registered under this rule
   (pending_rows where rule_id=... and metadata_json->>'Filename' matches).
   If yes -> return existing AssetFilePath.
3. Symlink or copy file to assets/{AssetTable}/{file_name}
   per rule.copy_files.
4. Lease a RID.
5. Create pending_rows entry with asset_file_path set, rule_id set,
   metadata containing only file-derived defaults (URL is null until
   Hatrac upload, MD5/Length/Filename computed at upload time).
6. Return AssetFilePath bound to the pending entry.
```

### 4.4 `handle.scan()`

```
1. Assert rule's status is 'active'.
2. Walk source_dir (respecting glob, recurse).
3. For each file:
     if already registered under this rule -> skip
     else -> call self.register(file), collect the result
4. Return list of newly-registered AssetFilePaths.
```

### 4.5 Upload drain (within `upload_execution_outputs`)

```
1. For every directory_rules row with status='active' for this execution:
     call handle.scan() -- final safety-net scan.
2. Validate required columns on all status='leased' pending rows.
   In strict mode: abort on any violation.
3. Build FK dependency graph over status='leased' pending rows.
   Topologically sort.
4. For each row in topo order:
     if asset_file_path is not None:
       Upload file to Hatrac via existing asset pipeline.
       Patch URL/MD5/Length/Filename into metadata_json.
     Insert row into {schema}:{table} with leased RID.
     On success: status='uploaded', uploaded_at=now().
     On failure: status='failed', error=<msg>.
       In strict mode: re-raise.
       In best-effort mode: continue.
5. Return per-table summary of uploaded/failed counts.
```

## 5. Testing

### 5.1 Unit tests (no live catalog)

- `PendingRowsStore` schema creation, entry insert/update/status transitions.
- Directory rule registration: new, idempotent, parameter-drift rejection.
- `PendingRow.metadata` setter round-trips through SQLite.
- FK topological sort correctness with cycles (must raise), with
  independent components, with chains.
- Required-column validator identifies missing columns from a metadata dict.

### 5.2 Integration tests (live catalog)

- `stage_row` inserts a Subject and the returned RID matches server-side.
- `stage_rows` bulk-inserts 100 rows; all have distinct leased RIDs.
- Intra-batch FK: stage Subject, stage Image with `Subject=<leased_rid>`,
  upload, verify server-side Image.Subject equals the leased Subject.RID.
- `asset_directory_path` zero-ceremony: register dir of 10 files with no
  metadata; upload fails with clear message listing required columns when
  the asset table requires them; succeeds with the expected count when
  the asset table doesn't.
- `asset_directory_path` with per-file metadata: register dir, iterate
  `pending()`, set metadata per file, upload, verify catalog rows.
- `handle.register()` incremental: write a file, register it, write
  another, register it, upload → both appear.
- `handle.scan()` final-sweep safety net: write files, never call
  register, call upload → final scan picks them all up.
- Parameter-drift: register with glob='*', restart, re-register with
  glob='*.jpg' → raises.

### 5.3 Regression

- Existing `asset_file_path` tests pass unchanged.
- Existing `upload_execution_outputs` asset-upload tests pass unchanged.
- Existing `ManifestStore` tests updated to exercise the generalized
  schema where applicable.

## 6. Open questions (none blocking)

- **Cross-execution pending-rows.** Today pending entries are
  per-execution. A future iteration might want workspace-scoped pending
  rows (e.g. "staging area" that persists across execution contexts).
  Deferred — revisit if users hit the limit.
- **Update/delete support.** Entirely out of scope for V1. The broader
  Phase 1 goal #4 "upload-back from slice" is its own spec; this spec
  explicitly shapes new-row insert only.
- **Intra-table FK cycles.** Self-referencing tables (e.g. a Subject
  whose `Parent_Subject` points at another new Subject in the same
  batch) are not supported in V1 because `GenericUploader` does not
  honor caller-supplied per-file ordering within a single invocation
  (see §2.5 caveat). Users must stage the referenced row in an earlier
  batch or wait for a V2 that drives row inserts directly via ERMrest
  without the CSV/`GenericUploader` handoff.
- **Typed `AssetRecord` for plain rows.** `stage_row` currently accepts
  a `dict`. If typed records are useful (generate a `SubjectRecord`
  class from the model's Subject table, validate at stage time), that's
  an ergonomic addition that doesn't change the data-flow design.
