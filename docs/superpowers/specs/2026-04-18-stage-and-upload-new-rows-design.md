# Stage & upload new rows — design spec

**Status:** Draft, pending review (revision 4 from user feedback).
**Date:** 2026-04-18.
**Scope:** Design spec. Implementation plan follows separately.

## 1. Goals and scope

### 1.1 Goals

One API for writing new rows to a catalog, whether the user is connected
or offline. Covers three cases with the same vocabulary:

1. **Asset rows** — a file in Hatrac plus a row in an asset table.
2. **Plain new rows** — row inserts into any catalog table, no file.
3. **Directory-registered asset rows** — one call registers every file
   in a local directory.

Design principle: **the code to write a row is the same whether the
DerivaML instance is online or offline.** Only the library's mode
(online vs offline) changes when the row actually reaches the catalog.
This lets users write code once and run it either way, including in
intermittent-connectivity environments.

### 1.2 Non-goals

- **Updates or deletes.** V1 is insert-only.
- **Creating new tables or columns.** Target table must already exist
  in the catalog schema. ``exe.table(name)`` raises
  ``DerivaMLTableNotFound`` at call time if it doesn't.
- **Full slice upload-back.** The broader "sync arbitrary local changes
  to catalog" problem remains a separate future spec.
- **Filesystem watching.** Directory registration doesn't watch for new
  files; file arrival is surfaced via explicit ``register(file)`` or
  scan-at-upload-time.
- **Auto-generating metadata from sidecar files.** Per-file metadata is
  set by iterating the directory handle or by the handle's batch
  setter.

### 1.3 Migration posture

No migration. Clean slate — no in-flight pending entries to preserve.
External names (``exe.asset_file_path``, ``ml.asset_record_class``,
``upload_execution_outputs``, ``exe.table_path``) keep their signatures
as thin shims delegating to the new handle API.

## 2. Architecture

### 2.1 Online mode and offline mode

DerivaML gains a ``mode`` parameter:

```python
DerivaML(..., mode="online" | "offline")    # default: "online"
```

- **Online mode.** Writes go live to the catalog as soon as the call
  returns — the library drains the pending-row entry immediately after
  staging it. Asset rows still stage-then-upload because Hatrac is a
  two-phase pipeline (upload file, then insert row with URL). But plain
  rows are fire-and-forget: user calls ``insert(...)``, row is in the
  catalog by the time it returns.

- **Offline mode.** Every write stages into the workspace SQLite and
  stays there until ``exe.upload_execution_outputs()`` drains it.
  Server connectivity is only required at upload time.

Both modes share the same public API. The only user-visible difference
is the timing of when rows reach the catalog. Internally, both modes
use the same pending-row machinery (§2.3, §2.5); online mode is just
"stage + drain immediately."

### 2.2 Pending-row store

The workspace holds one per-execution `PendingRowsStore` (the former
`ManifestStore`, renamed) tracking pending-row entries uniformly for
asset rows and plain rows.

**Entry shape:** `target_schema`, `target_table`, `metadata_json` (the
column values), `rid` (leased), `status`, and an **optional**
`asset_file_path`. When non-null, the entry is an asset upload (file →
Hatrac → row with URL/MD5 patched in). When null, the entry is a
plain-row insert.

**One upload drain:** `upload_execution_outputs` iterates pending
entries. Order is determined by FK dependencies (§2.5). In online
mode, the drain also runs at the end of each `.insert()` call for
entries that haven't already been drained.

### 2.3 RID leasing (lazy, batched, crash-safe)

Every pending row eventually holds a real, server-assigned RID from
`public:ERMrest_RID_Lease`. Leases are acquired **lazily** — deferred
to the earliest of:

- An explicit `handle.rid` read.
- The upload drain starting.

Until lease time the handle carries `.rid == None` and
`status='staged'`. Users who never read `.rid` before upload pay one
batched POST for the entire execution's pending rows.

**Crash-safe two-phase acquisition.** Before the lease POST, the
library writes `status='leasing'` with a UUID idempotency token. On
success, the response RIDs are written back. On workspace startup,
entries still in `status='leasing'` trigger reconciliation: query
`ERMrest_RID_Lease` by token; adopt or revert to `staged`.

**Batching.** One POST per batch, chunked at
`PENDING_ROWS_LEASE_CHUNK` (default 500) to stay within ERMrest request
limits.

### 2.4 SQLite schema

Two tables in the workspace SQLite, scoped to execution:

**`pending_rows`**

| Column | Purpose |
|---|---|
| `id` | internal PK |
| `execution_rid` | scope |
| `key` | auto-hash; directory entries derive from rule_id + filename |
| `target_schema` / `target_table` | catalog target |
| `rid` | leased RID, null until leased |
| `lease_token` | idempotency token for crash recovery |
| `metadata_json` | column values as JSON (serialized via `AssetManifest._json_default`) |
| `asset_file_path` | local file path, or null for plain rows |
| `asset_types_json` | vocabulary terms, or null for plain rows |
| `description` | optional |
| `status` | `staged \| leasing \| leased \| uploading \| uploaded \| failed` |
| `error` | set on failure |
| `created_at` / `leased_at` / `uploaded_at` | timestamps |
| `rule_id` | fk to directory_rules, null if not from a directory |

**`directory_rules`**

| Column | Purpose |
|---|---|
| `id` | internal PK |
| `execution_rid` | scope |
| `target_schema` / `target_table` | target |
| `source_dir` | absolute path |
| `glob` / `recurse` / `copy_files` | selection params |
| `asset_types_json` | applied to every registered file |
| `status` | `active \| closed` |
| `created_at` | timestamp |

Indexes: `(execution_rid, status)`, `(execution_rid, target_table)`.

### 2.5 Public API

The row-level entry point is a `TableHandle`:

```python
exe.table(
    table_name: str,
    *,
    schema: str | None = None,   # required if name is ambiguous
) -> TableHandle | AssetTableHandle
```

Returns `AssetTableHandle` if the table is an asset table,
`TableHandle` otherwise. The subclass distinction makes asset-specific
methods (`asset_file`, `asset_directory`) visible only where they
apply, so IDE tab-completion shows exactly what's legal.

Handles are lightweight: they carry a reference to the ERMrest
`Table`, the execution RID, and a DerivaML reference. All row state
lives in the workspace SQLite.

**Schema disambiguation.** If `table_name` is ambiguous and `schema`
isn't provided, raise `DerivaMLTableNotFound` listing candidates.

#### 2.5.1 `TableHandle` surface

```python
class TableHandle:
    name: str
    schema: str

    def record_class(
        self,
        *,
        include_system_columns: bool = False,
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
        """Write rows. In online mode, rows reach the catalog by the
        time this returns. In offline mode, rows stage into the
        workspace and upload at `exe.upload_execution_outputs()`.
        Accepts scalar or iterable or DataFrame; returns scalar or
        list accordingly.

        Validation happens at call time via `self.record_class()`;
        invalid records raise immediately in either mode."""

    def pending(
        self,
        *,
        status: str | list[str] | None = None,
    ) -> list[PendingRow]:
        """Pending rows for this table, optionally filtered by status.
        Includes entries still in 'staged'/'leasing'/'leased' and any
        in 'failed' awaiting retry."""

    def discard_pending(
        self,
        *,
        rows: Iterable[PendingRow] | None = None,
        status: str | list[str] | None = None,
    ) -> int:
        """Remove pending rows matching the filter. Returns count
        discarded. Does not reclaim already-acquired server leases."""
```

#### 2.5.2 `AssetTableHandle` surface (extends TableHandle)

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
        **kwargs,   # per-column metadata, legacy compat
    ) -> AssetFilePath:
        """Register a single file for upload. ALWAYS DEFERRED
        (regardless of mode) because Hatrac upload + row insert is a
        two-phase pipeline. Validates metadata at call time.

        Returns an AssetFilePath (Path subclass) carrying the pending
        entry's metadata and (eventually) its leased RID."""

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
        kwargs; the zero-ceremony default is "register everything, no
        metadata beyond file-derived columns." For per-file metadata,
        use the returned handle's register() or set_metadata() or
        iterate pending() and set .metadata per file."""
```

#### 2.5.3 `AssetDirectoryHandle`

```python
class AssetDirectoryHandle:
    path: Path                   # source_dir
    rule_id: int
    table: AssetTableHandle

    def register(self, file: str | Path) -> AssetFilePath:
        """Eagerly register a specific file under this rule. Returns
        its pending-row handle. Use when you need the handle (e.g. to
        reference its .rid from another row's insert)."""

    def scan(self) -> list[AssetFilePath]:
        """Walk source_dir (respecting glob / recurse), register new
        files not yet seen under this rule. Idempotent."""

    def pending(self) -> list[AssetFilePath]:
        """All AssetFilePaths under this rule that are not yet
        uploaded."""

    def set_metadata(
        self,
        source: "pd.DataFrame",
        *,
        filename_col: str = "Filename",
    ) -> None:
        """Batch-set metadata for all pending files under this rule.
        One SQLite transaction. The DataFrame is indexed by filename
        (``filename_col`` identifies the column, or the index if it's
        already the filename); remaining columns become each file's
        metadata. Every resolved value is validated through the typed
        record class."""

    def close(self) -> None:
        """Final scan, then mark the rule closed. Subsequent register()
        or scan() calls raise."""
```

Deleted files between scans are not auto-removed; upload surfaces them
as per-row failures with clear error messages.

#### 2.5.4 Handle read properties

Both `PendingRow` and `AssetFilePath` expose:

```python
handle.rid         # leased RID; None until first read (lazy lease)
handle.status      # 'staged' | 'leasing' | 'leased' | 'uploading' | 'uploaded' | 'failed'
handle.metadata    # dict; setter writes through to SQLite
handle.error       # error message if status == 'failed'
```

Handles are thin stateless views over the SQLite row — every property
read issues a SELECT, every write commits an UPDATE. This guarantees
multiple handles to the same entry stay consistent; cost is ~100 µs
per read (WAL SQLite, local disk). Bulk operations use the
`set_metadata` batch path.

### 2.6 Upload drain

`Execution.upload_execution_outputs()`:

1. **Final scan.** For every active `AssetDirectoryHandle` rule, run
   `scan()` to pick up files written since the last register/scan.
2. **Re-validation.** Re-fetch the catalog model, re-validate each
   pending row's `metadata_json` against the typed record class to
   catch schema drift.
3. **Batch-lease.** Any `status='staged'` rows have RIDs leased now in
   chunked POSTs.
4. **Inter-table FK topological sort.** Build a DAG over pending rows.
   If it contains a cycle, raise `DerivaMLCycleError` listing the
   cycle's tables. Intra-table FK cycles are unsupported (see §6).
5. **Materialize staging tree.** For each pending row in topo order:
   - Asset rows: symlink/copy the file into the metadata-path layout
     expected by the existing asset upload spec; mark `uploading`.
   - Plain rows: append a CSV row to the table's staging CSV; mark
     `uploading`.
6. **Invoke existing upload pipeline** per topological level. Hatrac
   uploads, URL/MD5/Length patching, ERMrest row inserts via
   `GenericUploader`.
7. **Reconcile status.** Each row transitions to `uploaded` or
   `failed`. On first failure, the drain abort after recording all
   failures in the current level (fail-fast default; see §2.8).

**Transaction semantics.** Each row insert is its own transaction. No
batch rollback. Partial success after a crash is the expected state;
per-row `status` in SQLite enables resumability.

**`retry_failed()`.** Convenience on `Execution` that resets
`status='failed'` entries to `status='leased'` and re-runs
`upload_execution_outputs`.

### 2.7 Online-mode drain

In online mode, `.insert(...)` also drains the entries it created
before returning. Implementation: after staging N rows,
`_drain_pending(execution_rid, entry_ids=[...])` runs the same drain
logic (§2.6 steps 3-7) restricted to those entries.

Asset-file calls (`.asset_file(...)`) do NOT drain eagerly even in
online mode — Hatrac is a two-phase pipeline and draining per-file
defeats batching. Asset rows collect until `upload_execution_outputs`.

In practice this means:

- Online mode: `.insert(plain_row)` → returns when row is in catalog.
- Online mode: `.asset_file(file, ...)` → returns when staged; file
  and row hit catalog at `upload_execution_outputs` time.
- Offline mode: both return when staged; nothing reaches catalog
  until `upload_execution_outputs`.

### 2.8 Error handling

Default behavior: the drain **fails fast** on the first level that
has a failure. Pending rows not yet attempted keep their status for
inspection and retry. A future release may add a best-effort mode; V1
is strict.

`handle.pending(status='failed')` for triage. `error` carries the
message. User edits metadata or fixes referenced data, then
`exe.retry_failed()`.

### 2.9 Backward-compatibility shims

Existing public methods become thin shims delegating to the new
handle API. No caller changes required to ship this.

```python
# exe.asset_file_path(asset_name, file, ...) becomes:
exe.table(asset_name).asset_file(file, ...)

# ml.asset_record_class(table) becomes:
exe.table(table).record_class()

# exe.table_path(name) becomes:
exe.table(name).upload_csv_path()    # deferred; kept as compat only
```

Shims are documentation-level deprecations in V1 (no runtime warning).

## 3. Walkthrough

A training script that downloads a dataset bag, runs a model, and
uploads per-image assets plus predictions:

```python
ml = DerivaML(hostname="example.org", catalog_id="42")   # online (default)
exe = ml.create_execution(config_with_dataset_spec)

with exe.execute() as e:
    bag = e.datasets["1-XYZ"].bag

    for image_rid in bag.list_dataset_members()["Image"]:
        img = bag.get_row("Image", image_rid)

        mask_arr, confidence = my_model.predict(img.filename)

        # Save mask to disk, register as a pending asset row
        mask_path = Path(exe.working_dir) / f"mask_{image_rid}.png"
        save_png(mask_arr, mask_path)
        mask = exe.table("Segmentation_Mask").asset_file(
            mask_path,
            Image=image_rid,
            Model_Version="v1.2",
        )

        # Register a prediction row referencing the mask. In online
        # mode this goes live to the catalog before the loop moves on.
        # In offline mode it stages and waits for upload.
        exe.table("Prediction").insert({
            "Image": image_rid,
            "Mask": mask.rid,       # lazy-materialized lease on first read
            "Confidence": confidence,
        })

exe.upload_execution_outputs()
```

The same loop body works online or offline. Mode changes the timing of
when rows reach the catalog, not the code.

## 4. Algorithms

### 4.1 Handle construction

```
exe.table(name):
  1. Resolve name via model.name_to_table(name, schema=schema).
  2. If model.is_asset(table):   return AssetTableHandle(exe, table)
     else:                        return TableHandle(exe, table)
  Handle is cheap: holds exe + table + scope.
```

### 4.2 `TableHandle.insert(records)`

```
1. Normalize records:
   - DataFrame -> list of dicts.
   - scalar -> list of one.
   - Iterable -> materialize in chunks.
2. For each chunk of chunk_size:
   a. Validate each rec through self.record_class() (raises on failure).
   b. Generate lease_token per row.
   c. BEGIN TRANSACTION.
   d. Insert N pending_rows entries (status='staged', rid=NULL).
   e. COMMIT.
   f. progress(done, total) if provided.
3. If exe.ml.mode == 'online':
   a. _drain_pending(entry_ids=[...])   # steps 3-7 of §2.6 for these rows
4. Return handles (scalar if input was scalar; list otherwise).
```

### 4.3 `AssetTableHandle.asset_file(file, ...)`

```
1. Validate file_name, determine staging path.
2. Symlink or copy file into assets/{AssetTable}/.
3. Construct metadata record via self.record_class()(**metadata, **kwargs).
4. Generate lease_token; write pending_rows entry with
   asset_file_path set, status='staged'.
5. Return AssetFilePath bound to the entry.
   (No drain — asset rows always defer to upload_execution_outputs.)
```

### 4.4 Lazy `.rid` materialization

```
1. If entry.rid is not null, return it.
2. Call workspace._acquire_leases([entry]) which:
   a. Find all entries in status='staged' (coalesce other pending
      reads into this batch).
   b. Set status='leasing', commit.
   c. POST /entity/public:ERMrest_RID_Lease with body
      [{"ID": lease_token} for each].
   d. UPDATE pending_rows SET rid=<assigned>, status='leased',
      leased_at=now() WHERE lease_token=<token>.
3. Return entry.rid.
```

### 4.5 Workspace startup lease reconciliation

```
1. Query pending_rows WHERE status='leasing'.
2. For each group (batched by execution):
   a. Query ERMrest_RID_Lease for the group's lease_tokens.
   b. Found -> set rid=<found>, status='leased'.
   c. Not found -> set status='staged' (POST never landed).
```

### 4.6 Upload drain

See §2.6 step-by-step.

## 5. Testing

### 5.1 Unit tests (no live catalog)

Handle dispatch:
- `exe.table(plain)` returns `TableHandle`; `.asset_file` not available.
- `exe.table(asset)` returns `AssetTableHandle`; `.asset_file` available.
- Ambiguous name without `schema=` raises.

Insert semantics:
- `insert(dict)` validates via record_class; raises on bad field.
- `insert(record)` passes through.
- `insert(df)` converts to list of dicts.
- `insert(scalar)` returns scalar handle; `insert(iterable)` returns list.
- Empty input: `insert([])`, `insert(empty_df)` succeed as no-ops.
- Large batch: `insert(10000_records)` chunked with progress callbacks.

Online vs offline dispatch:
- Online mode: `.insert(plain_row)` drains immediately; `.status` is
  `'uploaded'` on return (mock server).
- Offline mode: `.insert(plain_row)` returns with `.status='staged'`;
  `upload_execution_outputs()` drains.

Lazy lease:
- `.rid` read triggers batched POST; already-leased entries skipped.
- Two-phase: `status='leasing'` written before POST; on POST failure,
  reverts to `'staged'`.
- Startup reconciliation: POST success + simulated crash → on startup,
  lease adopted.

Directory handle:
- `asset_directory(dir).register(file)` eager.
- `.scan()` idempotent.
- `.set_metadata(df)` one transaction, all files validated.
- Deleted file between scans: not auto-removed; upload surfaces
  failure.

Inspection / discard:
- `handle.pending(status=...)` filter combinations.
- `handle.discard_pending(...)` removes; server leases NOT reclaimed.

Compatibility shims:
- `exe.asset_file_path(...)` delegates to
  `exe.table(...).asset_file(...)`.
- `ml.asset_record_class(...)` delegates to handle.record_class.

### 5.2 Integration tests (live catalog)

Online mode:
- `exe.table("Subject").insert({...})` returns with live RID in catalog.
- `exe.table("Prediction").insert(df)` bulk-insert 100 rows.
- Asset upload via `exe.table("Image").asset_file(...)` +
  `upload_execution_outputs()`: file in Hatrac, row in catalog.

Offline mode:
- Same code, `mode="offline"`; rows only reach catalog at
  `upload_execution_outputs()`.

Mixed:
- Asset row referenced by plain row via `mask.rid`: lazy lease
  materializes before Prediction insert.

Edge cases:
- Inter-table FK cycle raises `DerivaMLCycleError`.
- Stale-schema drift: add required column between stage and upload →
  re-validation error.
- File deleted between register and upload: per-row failure with
  clear error message.

### 5.3 Regression

- Existing `exe.asset_file_path` tests pass (via shim).
- Existing `ml.asset_record_class` tests pass (via shim).
- Existing `upload_execution_outputs` asset tests pass.

## 6. Known limits

- **Updates / deletes:** insert-only.
- **Intra-table FK cycles:** unsupported.
- **Binary columns (`bytea`):** unsupported; `insert()` raises
  `NotImplementedError`. Asset-table routes for binary payloads are
  the supported path.
- **CSV-based insert roundtrip (§2.6 step 5-6):** transitional. V2 can
  swap to direct `pathBuilder` insert without public-API change.
- **Orphaned server leases:** possible only if client loses its entire
  workspace state after a successful lease POST. Not reconcilable from
  the client side; server-side GC is an ERMrest concern.
