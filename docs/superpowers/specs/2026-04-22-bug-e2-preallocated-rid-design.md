# Bug E.2: Asset Uploads Honor Pre-Leased RIDs — Design

**Status:** Draft · **Date:** 2026-04-22 · **Bug:** Pre-S3 architectural gap — deriva-py's asset uploader ignores pre-allocated RIDs

## 1. Problem

DerivaML's lease system (§2.6, `rid_lease.py` + `lease_orchestrator.py`) pre-allocates RIDs from `public:ERMrest_RID_Lease` so client code can reference those RIDs in FK columns **before** the target row exists in the catalog. The plain-row drain path honors pre-leased RIDs correctly: `metadata["RID"] = r["rid"]` + `tpath.insert(body)` (see `upload_engine.py` around line 573).

The **asset-row path** does not. It delegates to deriva-py's `GenericUploader`, which uses `record_query_template: "/entity/{target_table}/MD5={md5}&Filename={file_name}"` to either look up an existing row (by MD5+Filename) or `_catalogRecordCreate` a new one with a **server-generated RID**. The pre-leased RID in `files[i]["rid"]` is never threaded into the insert payload.

Consequences:
- Any client code that captured the pre-leased RID (e.g., to build an FK-referencing row elsewhere) ends up with a dangling reference — the row in the asset table has a different RID.
- SQLite's `pending_rows.rid` column becomes stale (diverged from catalog).
- The whole point of the lease system — "client has a valid RID immediately, no round-trip per row" — is silently broken for asset tables.

## 2. Goals & non-goals

**In scope:**
- Asset uploads must use the pre-leased RID for the catalog insert.
- Upstream fix in deriva-py (branch `2.0-dev`): add opt-in config flag `use_pre_allocated_rid` to asset mappings.
- Pre-flight validator in deriva-ml that confirms every pre-leased RID is still live in `ERMrest_RID_Lease` before the upload runs — raises an aggregated `DerivaMLValidationError` if any are missing.
- Idempotent retry: if a partial failure leaves a catalog row at the pre-leased RID, a subsequent upload detects and skips the duplicate insert rather than raising an RID-collision error.

**Explicitly out of scope:**
- Any change to the plain-row drain path (already works).
- Removing the MD5+Filename query for callers who don't opt into `use_pre_allocated_rid` (backward compatibility requirement).
- Implementing a separate lease-release-on-success workflow — a successfully consumed lease stays in `ERMrest_RID_Lease` indefinitely (existing behavior).
- Changes to the state-store's lease-acquisition protocol.
- Cross-repo tooling automation. The deriva-py and deriva-ml PRs are coordinated manually.

## 3. Design summary

Four units across two repos:

### 3.1 deriva-py (upstream, branch `2.0-dev`)

- **New config flag** `use_pre_allocated_rid: true` on any asset mapping in the upload spec. Default `false` (legacy MD5+Filename behavior).
- **Fast-fail validation** in `_initFileMetadata`: if flag is set and `match_groupdict` lacks `RID`, raise `DerivaUploadConfigurationError` before any hatrac I/O.
- **New method** `_createFileRecordWithRid` replaces `_getFileRecord` when the flag is set. Does a pre-check GET by RID for idempotency; if no row exists, creates one with the caller-supplied RID in the payload.
- Everything else (hatrac upload, regex, column_map, pre/post processors) unchanged.

### 3.2 deriva-ml (downstream)

- **New validator** `_validate_pending_asset_leases(catalog, entries)` in `rid_lease.py` — batch-queries `ERMrest_RID_Lease` and raises `DerivaMLValidationError` aggregating any missing RIDs.
- **Upload entry points** (`Execution._upload_execution_dirs` and `run_upload_engine`) call the new validator alongside the existing `_validate_pending_asset_metadata`.
- **Upload spec** (`asset_table_upload_spec`) emits `use_pre_allocated_rid=True`, adds `(?P<RID>[-A-Z0-9]+)` to `file_pattern`, and adds `"RID": "{RID}"` to `column_map`.
- **Staging path builders** (`_invoke_deriva_py_uploader._path_for` and `Execution._build_upload_staging`) append the pre-leased RID as the final directory segment after the metadata columns: `.../Table/<col1>/<col2>/<RID>/<file>`.
- **Dependency** bumps `deriva-py` in `pyproject.toml` to the release that ships the new flag.

## 4. Components

### 4.1 deriva-py — `use_pre_allocated_rid` flag

**File:** `deriva/transfer/upload/deriva_upload.py`

**Change A** — after `_initFileMetadata`'s `self._updateFileMetadata(match_groupdict)`:

```python
if stob(asset_mapping.get("use_pre_allocated_rid", False)):
    if not self.metadata.get("RID"):
        raise DerivaUploadConfigurationError(
            "Asset mapping has use_pre_allocated_rid=true but no RID "
            "was captured by the file_pattern regex. Ensure the pattern "
            "includes a (?P<RID>[A-Z0-9-]+) named group."
        )
```

**Change B** — in `_uploadAsset` Step 7 (`if not record:`):

```python
if not record:
    if stob(asset_mapping.get("use_pre_allocated_rid", False)):
        record, result = self._createFileRecordWithRid(asset_mapping)
    else:
        record, result = self._getFileRecord(asset_mapping)
```

**Change C** — new private method on `GenericUploader`:

```python
def _createFileRecordWithRid(self, asset_mapping):
    """Create a new record using a caller-supplied RID.

    Used when asset_mapping has ``use_pre_allocated_rid: true``.
    Skips the MD5+Filename lookup in ``_getFileRecord``; the caller
    (via the scan-path regex) must have supplied an RID that was
    pre-allocated from ``ERMrest_RID_Lease``.

    Idempotency: if a row with the supplied RID already exists
    (e.g., a prior upload landed the catalog row but the client
    crashed before recording success), returns that row rather
    than raising an RID-collision error. Callers can safely retry
    after partial failures.

    Returns:
        Tuple of (record, result) mirroring ``_getFileRecord``'s
        return shape.
    """
    column_map = asset_mapping.get("column_map", {})
    allow_none_col_list = asset_mapping.get("allow_empty_columns_on_update", [])
    target_table = self.metadata['target_table']
    rid = self.metadata["RID"]

    existing = self.catalog.get(f"/entity/{target_table}/RID={rid}").json()
    if existing:
        record = existing[0]
        self._updateFileMetadata(record, no_overwrite=True)
        return self.pruneDict(record, column_map, allow_none_col_list), record

    row = self.interpolateDict(self.metadata, column_map, allow_none_col_list)
    result = self._catalogRecordCreate(target_table, row)
    record = result[0] if result else row
    if record:
        self._updateFileMetadata(record)
    return self.interpolateDict(
        self.metadata, column_map, allow_none_column_list=allow_none_col_list
    ), record
```

### 4.2 deriva-ml — `_validate_pending_asset_leases`

**Location:** module-level private function at the end of `src/deriva_ml/execution/rid_lease.py`.

```python
def _validate_pending_asset_leases(
    catalog: "ErmrestCatalog",
    entries: "Iterable[tuple[str, str]]",
) -> None:
    """Confirm each (key, rid) pair's RID is still live in ERMrest_RID_Lease.

    Queries the lease table in batches of ``PENDING_ROWS_LEASE_CHUNK``.
    Aggregates missing RIDs and raises a single
    :class:`DerivaMLValidationError` listing every failure in sorted
    order. Returns ``None`` silently when every RID is present.

    Args:
        catalog: Live ErmrestCatalog for querying the lease table.
        entries: Iterable of (key, rid) tuples. Key is a
            human-readable identifier used in the error message.

    Raises:
        DerivaMLValidationError: If one or more RIDs are not found
            in ``ERMrest_RID_Lease``.
    """
```

Uses the existing `PENDING_ROWS_LEASE_CHUNK` constant for batch size.

**Error-message shape:**

```
Missing or invalid pre-allocated RIDs for 2 pending asset(s):
  - Image/acq001.png: RID 1-ABC not found in ERMrest_RID_Lease
  - Image/acq002.png: RID 1-DEF not found in ERMrest_RID_Lease
A pre-leased RID has become invalid (e.g., cleared from the lease
table or never successfully POSTed). Restart the execution to
re-lease, or investigate lease-table state.
```

### 4.3 deriva-ml — validator call sites

**Call site 1 — `Execution._upload_execution_dirs`:**
After the existing `_validate_pending_asset_metadata(self._model, self._get_manifest())` call, add:

```python
from deriva_ml.execution.rid_lease import _validate_pending_asset_leases
manifest = self._get_manifest()
lease_entries = [
    (key, entry.rid)
    for key, entry in manifest.pending_assets().items()
    if entry.rid
]
if lease_entries:
    _validate_pending_asset_leases(self._ml_object.catalog, lease_entries)
```

**Call site 2 — `run_upload_engine`:**
After the existing metadata-validator block, reuse the row-iteration to build lease entries:

```python
from deriva_ml.execution.rid_lease import _validate_pending_asset_leases
lease_entries = [
    (key_str, row["rid"])
    for key_str, row in <rows already iterated>
    if row.get("rid") and row.get("asset_file_path")
]
if lease_entries:
    _validate_pending_asset_leases(ml.catalog, lease_entries)
```

### 4.4 deriva-ml — upload spec changes

In `asset_table_upload_spec()` (`src/deriva_ml/dataset/upload.py`):

**Regex** — add RID capture group after metadata path, before filename:

```python
metadata_path = "/".join([rf"(?P<{c}>[-:._ \w]+)" for c in metadata_columns])
rid_path = r"(?P<RID>[-A-Z0-9]+)"
parts = [metadata_path, rid_path] if metadata_path else [rid_path]
asset_path = (
    f"{exec_dir_regex}/asset/{schema}/{asset_table.name}/"
    f"{'/'.join(parts)}/{asset_file_regex}"
)
```

**Column map** — add `"RID": "{RID}"`.

**Spec dict** — add `"use_pre_allocated_rid": True`.

### 4.5 deriva-ml — staging path changes

**In `_invoke_deriva_py_uploader._path_for`** (`upload_engine.py`):

```python
def _path_for(f: dict) -> Path:
    src = Path(f["path"]).resolve()
    metadata = f.get("metadata") or {}
    target_dir = asset_root / schema_name / target_table
    for col in metadata_cols:
        target_dir = target_dir / str(metadata.get(col, NULL_SENTINEL))
    target_dir = target_dir / f["rid"]
    return target_dir / src.name
```

**In `Execution._build_upload_staging`** (`execution.py`):

```python
metadata_parts = [
    str(entry.metadata.get(k, NULL_SENTINEL)) for k in all_metadata_cols
] if all_metadata_cols else []
target_dir = staging_root / entry.schema / asset_table_name
for part in metadata_parts:
    target_dir = target_dir / part
target_dir = target_dir / entry.rid
target_dir.mkdir(parents=True, exist_ok=True)
```

**In `dataset/upload.py:asset_file_path`** (bag-export helper, line 657):

```python
for m in asset_metadata:
    path = path / str(metadata.get(m, NULL_SENTINEL))
path = path / rid  # requires new rid argument; see §8
path.mkdir(parents=True, exist_ok=True)
return path / file_name
```

This helper is primarily used by `core/mixins/execution.py:229` for `Execution_Metadata` (zero metadata columns). It will need a new `rid: str` parameter. Callers that don't have a pre-leased RID (bag-export for already-catalog-resident rows) must pass the catalog row's existing RID. See §8.

### 4.6 deriva-ml — dependency pin

In `pyproject.toml`, bump `deriva-py` requirement to the version that includes the `use_pre_allocated_rid` flag. Target version TBD when the upstream release tag is chosen.

## 5. Data flow

### 5.1 Happy path

```
exe.upload_outputs()
  → _validate_pending_asset_metadata (Bug C validator, unchanged)
  → _validate_pending_asset_leases (NEW)
    [batch query /entity/public:ERMrest_RID_Lease/RID=r1;RID=r2;...]
    [all found → no error]
  → staging: .../Table/<col1>/<col2>/<RID>/<file>
  → GenericUploader.scanDirectory
  → _initFileMetadata captures RID from regex match_groupdict
  → [flag set + RID present → pass validation]
  → hatrac upload (Step 6, unchanged)
  → _createFileRecordWithRid
    → GET /entity/{table}/RID={rid}
    → [no existing row → _catalogRecordCreate with RID in payload]
  → Catalog row has pre-leased RID ✓
```

### 5.2 Missing-lease validator failure

```
exe.upload_outputs()
  → _validate_pending_asset_metadata — ok
  → _validate_pending_asset_leases
    [one or more RIDs not in ERMrest_RID_Lease]
    → raise DerivaMLValidationError (aggregated)
```

No hatrac upload, no catalog mutation.

### 5.3 Retry after partial failure

```
[Attempt 1: hatrac upload succeeds, catalog insert interrupted]
  → pending_rows.status = failed
[User retries: exe.upload_outputs(retry_failed=True)]
  → _validate_pending_asset_leases passes (lease still live)
  → staging rebuilt with same RID
  → GenericUploader.scanDirectory → _initFileMetadata ok
  → _createFileRecordWithRid
    → GET /entity/{table}/RID={rid}
    → [row EXISTS from attempt 1 → return existing row, skip create]
  → Upload reported successful; pending row marked uploaded
```

### 5.4 Flag-off path (backward compat)

```
[Asset tables without use_pre_allocated_rid=true — any caller not opted in]
  → Existing _getFileRecord flow runs (MD5+Filename lookup, server RID on create)
  → No behavior change vs. pre-Bug-E.2
```

### 5.5 Interaction with Bug C

Orthogonal. Bug C's validator runs first, then Bug E.2's. Staging path has metadata-column segments (from Bug C) followed by RID segment (from Bug E.2). `NULL_SENTINEL` + `NullSentinelProcessor` continue to work unchanged.

## 6. Error handling summary

| Scenario | Exception / behavior |
|---|---|
| Pre-leased RID not in `ERMrest_RID_Lease` (stale SQLite) | `DerivaMLValidationError` (aggregated) |
| `use_pre_allocated_rid=true` but regex didn't capture RID | `DerivaUploadConfigurationError` (deriva-py, fast-fail in `_initFileMetadata`) |
| Catalog row at pre-leased RID already exists (retry after partial success) | Treated as idempotent success; existing row returned |
| Catalog insert fails with RID-collision for some OTHER reason | Row marked `failed` in SQLite; retry behavior same as today |
| Hatrac upload fails | Same as today — retry re-attempts the upload |
| Flag not set on asset mapping | Legacy MD5+Filename flow, no validator runs |

## 7. Testing plan

### 7.1 deriva-py unit tests (no catalog)

1. `test_init_file_metadata_raises_when_flag_set_and_no_rid_captured`
2. `test_init_file_metadata_passes_when_flag_set_and_rid_captured`
3. `test_create_file_record_with_rid_happy_path` — no existing row → `_catalogRecordCreate` called with RID
4. `test_create_file_record_with_rid_idempotent_on_existing` — GET returns row → skip create
5. `test_flag_off_uses_existing_md5_filename_path`

### 7.2 deriva-ml unit tests (no catalog)

6. `test_empty_entries_returns_none`
7. `test_all_leases_valid_passes`
8. `test_single_missing_lease_raises`
9. `test_multiple_missing_leases_aggregated`
10. `test_batched_queries_use_chunk_size`

### 7.3 deriva-ml upload-spec tests (live catalog)

11. `test_asset_table_upload_spec_has_use_pre_allocated_rid_flag`
12. `test_asset_table_upload_spec_file_pattern_captures_rid`
13. `test_asset_table_upload_spec_column_map_includes_rid`

### 7.4 deriva-ml integration tests (DERIVA_HOST-gated)

14. `test_upload_asset_uses_pre_leased_rid` — happy path end-to-end, assert catalog row's RID equals the pre-leased RID.
15. `test_upload_asset_raises_validation_when_lease_missing` — RID not in lease table → `DerivaMLValidationError`.
16. `test_upload_asset_retry_after_hatrac_success_is_idempotent` — partial failure + retry → same catalog row.

### 7.5 Regression

- Bug C tests pass unchanged.
- Plain-row drain path tests pass unchanged.
- `Execution_Asset` (zero-metadata) smoke test passes — now exercises the new RID-segment path.

## 8. Open questions

- **`asset_file_path` signature.** The bag-export helper at `dataset/upload.py:657` currently takes `(prefix, exec_rid, file_name, asset_table, metadata)`. The new staging layout requires a RID segment. Callers split into two cases:
  - **Upload-time** (manifest-driven): caller has a pre-leased RID, passes it.
  - **Bag-export-time** (reading existing catalog rows): caller has the catalog's existing RID.
  Both cases can supply a RID. The signature gains a new `rid: str` parameter. Any existing callers without a RID at hand need to be updated; grep confirms only `core/mixins/execution.py:229` uses this helper, and that call is for an `Execution_Metadata` row being re-uploaded with known RID.
- **Retry MD5 mismatch edge case.** If a retry supplies a DIFFERENT file at the same pre-leased RID (e.g., the user re-staged), the idempotency pre-check returns the old row and skips insert. Not a ship blocker — pre-leased RIDs are tied to specific file identities in practice — but worth a follow-up hardening to compare MD5 and raise a clear error on mismatch.

## 9. Rollout

**Phase 1: deriva-py PR** against `2.0-dev`. Minimal: flag + fast-fail + `_createFileRecordWithRid`. Unit tests. Review + merge.
**Phase 2: deriva-py prerelease** tagged (e.g., `2.0.0.dev<N>`).
**Phase 3: deriva-ml PR** against `main`. Pin bump + validator + spec changes + staging changes + tests + CHANGELOG. Review + merge.

CHANGELOG entry under `## Unreleased — Bug E.2: asset uploads honor pre-leased RIDs` documenting:
- The new deriva-py `use_pre_allocated_rid` flag.
- deriva-ml's opt-in via `asset_table_upload_spec` (automatic — no caller changes needed).
- The new lease validator and its failure mode.
- That callers who already supplied correct metadata continue to succeed.
