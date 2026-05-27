# Bug E.2: Pre-Allocated RID Upload — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the pre-S3 architectural gap where deriva-py's asset uploader ignores pre-allocated RIDs from the lease system. Catalog inserts for asset rows must honor the caller-supplied RID.

**Architecture:** Two-repo change. **deriva-py** gains an opt-in `use_pre_allocated_rid` config flag on asset mappings and a new `_createFileRecordWithRid` method (with idempotency pre-check) that bypasses the MD5+Filename lookup. **deriva-ml** emits the flag in `asset_table_upload_spec`, appends the pre-leased RID as the final directory segment in the staging tree, and adds a pre-flight validator that confirms every pre-leased RID is live in `ERMrest_RID_Lease` before any upload work starts.

**Tech Stack:** Python ≥3.12, Pydantic v2 (deriva-ml validator output matches Bug C's aggregated-error shape), deriva-py's `GenericUploader` extension point, pytest. Built on the existing lease infrastructure (`rid_lease.py`, `lease_orchestrator.py`) and Bug C's validator pattern.

**Reference spec:** `docs/superpowers/specs/2026-04-22-bug-e2-preallocated-rid-design.md`

---

## Repo coordination

- **deriva-py** branch `2.0-dev` already tracked from deriva-ml's `pyproject.toml` as a git dep — no version bump needed.
- **deriva-py** tasks (Phase 1) land first; CI for the deriva-ml PR will fail until deriva-py merges, then automatically picks up the new commit.
- **deriva-ml** tasks (Phase 2) run against current `main` (which includes Bug C fixes).

Both repos are checked out side-by-side on the same machine:
- deriva-py: `/Users/carl/GitHub/deriva-py` (branch `2.0-dev`)
- deriva-ml worktree: `/Users/carl/GitHub/deriva-ml/.worktrees/bug-e` (branch `claude/bug-e`)

---

## File Structure

### deriva-py (Phase 1)

- Modify: `deriva/transfer/upload/deriva_upload.py` — add fast-fail validation in `_initFileMetadata`, branch in `_uploadAsset` Step 7, new method `_createFileRecordWithRid`.
- Create: `tests/deriva/transfer/upload/__init__.py` — test package marker.
- Create: `tests/deriva/transfer/upload/test_pre_allocated_rid.py` — unit tests for the new flag and method.

### deriva-ml (Phase 2)

- Modify: `src/deriva_ml/execution/rid_lease.py` — add `_validate_pending_asset_leases` at end of file.
- Modify: `src/deriva_ml/dataset/upload.py` — emit `use_pre_allocated_rid=True`, extend regex + column_map.
- Modify: `src/deriva_ml/execution/upload_engine.py` — append RID to staging path, call lease validator at `run_upload_engine`.
- Modify: `src/deriva_ml/execution/execution.py` — append RID to staging path in `_build_upload_staging`, call lease validator at `_upload_execution_dirs`.
- Modify: `src/deriva_ml/core/mixins/execution.py` — adapt caller of `asset_file_path` (line 229) to pass its known RID.
- Create: `tests/execution/test_pending_asset_lease_validator.py` — 5 unit tests for the lease validator.
- Modify: `tests/asset/test_null_sentinel_processor.py` — add 3 tests for the new upload-spec fields.
- Create: `tests/execution/test_bug_e2_live_smoke.py` — 3 live-catalog integration tests.
- Modify: `CHANGELOG.md` — Unreleased entry.

---

# Phase 1: deriva-py

## Task 1: Add fast-fail validation in `_initFileMetadata`

**Files:**
- Modify: `deriva/transfer/upload/deriva_upload.py:829-831` (append after `self._updateFileMetadata(match_groupdict)`)
- Create: `tests/deriva/transfer/upload/__init__.py` (empty, package marker)
- Test: `tests/deriva/transfer/upload/test_pre_allocated_rid.py` (new file)

Guard: if asset mapping has `use_pre_allocated_rid=true` but the regex didn't capture `RID`, raise `DerivaUploadConfigurationError` before any hatrac I/O.

- [ ] **Step 1: Create test package marker**

```bash
cd /Users/carl/GitHub/deriva-py && mkdir -p tests/deriva/transfer/upload && touch tests/deriva/transfer/upload/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/deriva/transfer/upload/test_pre_allocated_rid.py`:

```python
"""Unit tests for deriva-py's use_pre_allocated_rid asset-mapping flag.

These tests exercise two narrowly-scoped changes on GenericUploader:

1. A fast-fail validation in ``_initFileMetadata`` when the flag is
   set but the regex didn't capture a RID group.
2. A new ``_createFileRecordWithRid`` method that bypasses the
   MD5+Filename lookup in ``_getFileRecord`` and creates (or
   idempotently returns) a catalog row at the caller-supplied RID.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def uploader():
    """Construct a GenericUploader with mocked catalog + identity."""
    from deriva.transfer.upload.deriva_upload import GenericUploader
    inst = GenericUploader.__new__(GenericUploader)
    # Fill the attributes _initFileMetadata and _createFileRecordWithRid read.
    inst.metadata = {}
    inst.catalog = MagicMock()
    inst.identity = {"id": "anon", "display_name": "a", "full_name": "A", "email": "a@b"}
    inst.processor_output = {}
    inst.cancelled = False
    return inst


def test_init_file_metadata_raises_when_flag_set_and_no_rid_captured(uploader):
    from deriva.transfer.upload.deriva_upload import DerivaUploadConfigurationError
    asset_mapping = {
        "use_pre_allocated_rid": True,
        "target_table": ["S", "T"],
    }
    # match_groupdict has NO 'RID' key — regex didn't capture it.
    match_groupdict = {"schema": "S", "asset_table": "T"}
    # Stub getCatalogTable to avoid needing a real catalog model.
    uploader.getCatalogTable = MagicMock(return_value="S:T")
    uploader.getFileDisplayName = MagicMock(return_value="f.bin")
    uploader.getFileSize = MagicMock(return_value=123)

    with pytest.raises(DerivaUploadConfigurationError) as ei:
        uploader._initFileMetadata("/tmp/f.bin", asset_mapping, match_groupdict)
    msg = str(ei.value)
    assert "use_pre_allocated_rid" in msg
    assert "RID" in msg


def test_init_file_metadata_passes_when_flag_set_and_rid_captured(uploader):
    asset_mapping = {
        "use_pre_allocated_rid": True,
        "target_table": ["S", "T"],
    }
    match_groupdict = {"schema": "S", "asset_table": "T", "RID": "1-ABC"}
    uploader.getCatalogTable = MagicMock(return_value="S:T")
    uploader.getFileDisplayName = MagicMock(return_value="f.bin")
    uploader.getFileSize = MagicMock(return_value=123)

    # Should not raise.
    uploader._initFileMetadata("/tmp/f.bin", asset_mapping, match_groupdict)
    assert uploader.metadata["RID"] == "1-ABC"


def test_init_file_metadata_passes_when_flag_absent(uploader):
    """Legacy callers (flag not set) see no behavior change — no RID check."""
    asset_mapping = {"target_table": ["S", "T"]}
    match_groupdict = {"schema": "S", "asset_table": "T"}
    uploader.getCatalogTable = MagicMock(return_value="S:T")
    uploader.getFileDisplayName = MagicMock(return_value="f.bin")
    uploader.getFileSize = MagicMock(return_value=123)

    # Should not raise — legacy path doesn't require RID.
    uploader._initFileMetadata("/tmp/f.bin", asset_mapping, match_groupdict)
    assert "RID" not in uploader.metadata
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/deriva-py && python -m pytest tests/deriva/transfer/upload/test_pre_allocated_rid.py -v`

Expected: `test_init_file_metadata_raises_when_flag_set_and_no_rid_captured` FAILS with "DID NOT RAISE" because the validation doesn't exist yet. The other two tests pass trivially (no validation runs, so no raise).

- [ ] **Step 4: Add the fast-fail validation**

In `deriva/transfer/upload/deriva_upload.py`, locate `_initFileMetadata` (around line 829). After the line `self._updateFileMetadata(match_groupdict)` and before `self.metadata['target_table'] = self.getCatalogTable(...)`, insert:

```python
        # Fast-fail check for pre-allocated-RID asset mappings: the caller
        # opted in via ``use_pre_allocated_rid: true`` and must supply the
        # RID via a ``(?P<RID>...)`` named group in the file_pattern regex.
        if stob(asset_mapping.get("use_pre_allocated_rid", False)):
            if not self.metadata.get("RID"):
                raise DerivaUploadConfigurationError(
                    "Asset mapping has use_pre_allocated_rid=true but no RID "
                    "was captured by the file_pattern regex. Ensure the pattern "
                    "includes a (?P<RID>[A-Z0-9-]+) named group."
                )
```

Verify `stob` and `DerivaUploadConfigurationError` are already imported (they should be — `stob` appears in `_uploadAsset`; `DerivaUploadConfigurationError` in `_getFileRecord`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/deriva-py && python -m pytest tests/deriva/transfer/upload/test_pre_allocated_rid.py -v`

Expected: all 3 tests pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/deriva-py && git add tests/deriva/transfer/upload/__init__.py tests/deriva/transfer/upload/test_pre_allocated_rid.py deriva/transfer/upload/deriva_upload.py
git commit -m "feat(upload): fast-fail when use_pre_allocated_rid flag has no RID"
```

---

## Task 2: Add `_createFileRecordWithRid` with idempotency pre-check

**Files:**
- Modify: `deriva/transfer/upload/deriva_upload.py` — add new method after `_getFileRecord` (around line 820).
- Test: `tests/deriva/transfer/upload/test_pre_allocated_rid.py` — append 2 tests.

New private method that skips MD5+Filename lookup, does a RID-keyed existence check for idempotent retry, creates if absent.

- [ ] **Step 1: Write the failing tests**

Append to `tests/deriva/transfer/upload/test_pre_allocated_rid.py`:

```python
def test_create_file_record_with_rid_happy_path(uploader):
    """No existing row with RID → _catalogRecordCreate called with RID in payload."""
    asset_mapping = {
        "use_pre_allocated_rid": True,
        "column_map": {"MD5": "{md5}", "Filename": "{file_name}", "RID": "{RID}"},
    }
    uploader.metadata = {
        "RID": "1-NEW",
        "md5": "abc123",
        "file_name": "f.bin",
        "target_table": "S:T",
    }

    # Pre-check GET returns empty — row doesn't exist yet.
    get_response = MagicMock()
    get_response.json.return_value = []
    uploader.catalog.get.return_value = get_response

    # Stub _catalogRecordCreate to return a created-record shape.
    uploader._catalogRecordCreate = MagicMock(
        return_value=[{"RID": "1-NEW", "MD5": "abc123", "Filename": "f.bin"}]
    )
    uploader._updateFileMetadata = MagicMock()

    record, result = uploader._createFileRecordWithRid(asset_mapping)

    # Pre-check queried by RID.
    uploader.catalog.get.assert_called_once_with("/entity/S:T/RID=1-NEW")
    # Create called with RID in the row.
    create_call = uploader._catalogRecordCreate.call_args
    assert create_call[0][0] == "S:T"  # target_table
    assert create_call[0][1]["RID"] == "1-NEW"
    # Return shape mirrors _getFileRecord: (dict, record).
    assert isinstance(record, dict)
    assert result["RID"] == "1-NEW"


def test_create_file_record_with_rid_idempotent_on_existing(uploader):
    """RID already exists → skip create, return existing row."""
    asset_mapping = {
        "use_pre_allocated_rid": True,
        "column_map": {"MD5": "{md5}", "Filename": "{file_name}", "RID": "{RID}"},
    }
    uploader.metadata = {
        "RID": "1-OLD",
        "md5": "abc123",
        "file_name": "f.bin",
        "target_table": "S:T",
    }

    existing_row = {"RID": "1-OLD", "MD5": "abc123", "Filename": "f.bin"}
    get_response = MagicMock()
    get_response.json.return_value = [existing_row]
    uploader.catalog.get.return_value = get_response

    # _catalogRecordCreate MUST NOT be called.
    uploader._catalogRecordCreate = MagicMock()
    uploader._updateFileMetadata = MagicMock()

    record, result = uploader._createFileRecordWithRid(asset_mapping)

    uploader._catalogRecordCreate.assert_not_called()
    # Return shape: (pruned dict, existing record).
    assert isinstance(record, dict)
    assert result == existing_row
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/deriva-py && python -m pytest tests/deriva/transfer/upload/test_pre_allocated_rid.py -v`

Expected: the 2 new tests FAIL with `AttributeError: 'GenericUploader' object has no attribute '_createFileRecordWithRid'`.

- [ ] **Step 3: Add the method**

In `deriva/transfer/upload/deriva_upload.py`, locate `_getFileRecord` (around line 793). Immediately AFTER that method (before `_urlEncodeMetadata` or whatever method comes next), insert:

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
            Tuple of (row, record) mirroring ``_getFileRecord``'s
            return shape.
        """
        column_map = asset_mapping.get("column_map", {})
        allow_none_col_list = asset_mapping.get("allow_empty_columns_on_update", [])
        target_table = self.metadata['target_table']
        rid = self.metadata["RID"]

        # Pre-check: does a row with this RID already exist?
        existing = self.catalog.get("/entity/%s/RID=%s" % (target_table, rid)).json()
        if existing:
            record = existing[0]
            self._updateFileMetadata(record, no_overwrite=True)
            return self.pruneDict(record, column_map, allow_none_col_list), record

        # Fresh create — RID goes into the payload via column_map.
        row = self.interpolateDict(self.metadata, column_map, allow_none_col_list)
        result = self._catalogRecordCreate(target_table, row)
        record = result[0] if result else row
        if record:
            self._updateFileMetadata(record)
        return self.interpolateDict(
            self.metadata, column_map, allow_none_column_list=allow_none_col_list
        ), record
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/deriva-py && python -m pytest tests/deriva/transfer/upload/test_pre_allocated_rid.py -v`

Expected: all 5 tests pass (3 from Task 1 + 2 new).

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/deriva-py && git add deriva/transfer/upload/deriva_upload.py tests/deriva/transfer/upload/test_pre_allocated_rid.py
git commit -m "feat(upload): _createFileRecordWithRid bypasses MD5+Filename lookup"
```

---

## Task 3: Wire `_createFileRecordWithRid` into `_uploadAsset`

**Files:**
- Modify: `deriva/transfer/upload/deriva_upload.py:736-738` (Step 7 of `_uploadAsset`)
- Test: `tests/deriva/transfer/upload/test_pre_allocated_rid.py` — append 1 test.

Route `_uploadAsset`'s Step 7 to `_createFileRecordWithRid` when the flag is set.

- [ ] **Step 1: Write the failing test**

Append to `tests/deriva/transfer/upload/test_pre_allocated_rid.py`:

```python
def test_flag_off_uses_existing_md5_filename_path(uploader):
    """When flag is absent, _getFileRecord (MD5+Filename path) is used."""
    asset_mapping = {"target_table": ["S", "T"], "column_map": {"MD5": "{md5}"}}
    uploader.metadata = {"md5": "abc", "file_name": "f.bin", "target_table": "S:T"}

    uploader._getFileRecord = MagicMock(return_value=({}, {"RID": "SERVER-RID"}))
    uploader._createFileRecordWithRid = MagicMock()

    # Simulate what _uploadAsset Step 7 will do (see task step 2).
    from deriva.transfer.upload.deriva_upload import stob
    if stob(asset_mapping.get("use_pre_allocated_rid", False)):
        record, result = uploader._createFileRecordWithRid(asset_mapping)
    else:
        record, result = uploader._getFileRecord(asset_mapping)

    uploader._getFileRecord.assert_called_once()
    uploader._createFileRecordWithRid.assert_not_called()
    assert result["RID"] == "SERVER-RID"
```

- [ ] **Step 2: Update `_uploadAsset` Step 7**

In `deriva/transfer/upload/deriva_upload.py`, locate `_uploadAsset` (line 660). Find Step 7 (around line 736-738):

```python
        # 7. Check for an existing record and create a new one if necessary
        if not record:
            record, result = self._getFileRecord(asset_mapping)
```

Replace with:

```python
        # 7. Check for an existing record and create a new one if necessary.
        #    If use_pre_allocated_rid is set, skip the MD5+Filename lookup
        #    and go straight to _createFileRecordWithRid (with idempotency
        #    pre-check).
        if not record:
            if stob(asset_mapping.get("use_pre_allocated_rid", False)):
                record, result = self._createFileRecordWithRid(asset_mapping)
            else:
                record, result = self._getFileRecord(asset_mapping)
```

Also check Step 5 (around line 690-693) — it has a similar `record = self._getFileRecord(asset_mapping)` call in the `create_record_before_upload` branch. For safety and symmetry, update it too:

```python
        # 5. If "create_record_before_upload" specified in asset_mapping, check for an existing record, creating a new
        #    one if necessary. Otherwise, delay this logic until after the file upload.
        result = record = None
        if stob(asset_mapping.get("create_record_before_upload", False)):
            if stob(asset_mapping.get("use_pre_allocated_rid", False)):
                record = self._createFileRecordWithRid(asset_mapping)
            else:
                record = self._getFileRecord(asset_mapping)
```

Wait — the existing Step 5 returns only `record` (not a tuple). Let me leave Step 5 as is for now; `create_record_before_upload` isn't in deriva-ml's use case. Only update Step 7.

Revert any Step 5 change; only Step 7 should be modified.

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/deriva-py && python -m pytest tests/deriva/transfer/upload/test_pre_allocated_rid.py -v`

Expected: all 6 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /Users/carl/GitHub/deriva-py && git add deriva/transfer/upload/deriva_upload.py tests/deriva/transfer/upload/test_pre_allocated_rid.py
git commit -m "feat(upload): route _uploadAsset Step 7 to _createFileRecordWithRid when flag set"
```

---

## Task 4: Push deriva-py branch and open PR

- [ ] **Step 1: Push the branch**

```bash
cd /Users/carl/GitHub/deriva-py && git checkout -b claude/bug-e2-preallocated-rid && git push -u origin claude/bug-e2-preallocated-rid
```

- [ ] **Step 2: Open a PR against `2.0-dev`**

```bash
cd /Users/carl/GitHub/deriva-py && gh pr create \
  --base 2.0-dev \
  --title "feat(upload): add use_pre_allocated_rid flag for caller-supplied RIDs" \
  --body "$(cat <<'EOF'
## Summary

Add opt-in support for asset mappings that use caller-supplied RIDs instead of deriva-py's default MD5+Filename lookup/upsert.

When an asset mapping sets \`use_pre_allocated_rid: true\`:

- \`_initFileMetadata\` fast-fails with \`DerivaUploadConfigurationError\` if the file_pattern regex didn't capture a \`RID\` named group.
- \`_uploadAsset\` Step 7 routes to a new \`_createFileRecordWithRid\` method that does a RID-keyed existence check for idempotent retry, then creates with the caller-supplied RID in the payload.

Default (flag absent): legacy MD5+Filename behavior — no change.

## Motivation

deriva-ml's lease system (\`ERMrest_RID_Lease\`) pre-allocates RIDs so client code can reference them in FK columns before the target row exists. The existing upload path silently substitutes server-generated RIDs for asset rows, breaking this invariant. This flag lets opt-in callers honor pre-allocated RIDs.

## Test Plan

- [x] 6 unit tests in \`tests/deriva/transfer/upload/test_pre_allocated_rid.py\`:
  - Fast-fail validation when flag set without RID
  - Passthrough when flag set with RID captured
  - No change when flag absent
  - Happy path (no existing row → create with RID)
  - Idempotent retry (existing row → return without re-create)
  - Step 7 routing correctness

## Backward Compatibility

Strictly opt-in. No existing asset mapping has the flag, so no existing caller sees any behavior change.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Record the PR URL — deriva-ml Phase 2 can't merge until this one does.

---

# Phase 2: deriva-ml

## Task 5: Add `_validate_pending_asset_leases` validator

**Files:**
- Modify: `src/deriva_ml/execution/rid_lease.py` — append new function at end of file.
- Test: `tests/execution/test_pending_asset_lease_validator.py` (new file).

Pre-flight validator that batch-queries `ERMrest_RID_Lease` and aggregates missing RIDs into a single `DerivaMLValidationError`.

- [ ] **Step 1: Write the failing tests**

Create `tests/execution/test_pending_asset_lease_validator.py`:

```python
"""Unit tests for _validate_pending_asset_leases."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _fake_catalog(found_rids: set[str]):
    """Build a fake ErmrestCatalog whose .get() returns rows with
    RIDs from ``found_rids`` that match the query's filter."""
    catalog = MagicMock()

    def fake_get(path: str):
        # Parse "/entity/public:ERMrest_RID_Lease/RID=r1;RID=r2;..."
        response = MagicMock()
        assert path.startswith("/entity/public:ERMrest_RID_Lease/")
        filter_part = path.split("/entity/public:ERMrest_RID_Lease/", 1)[1]
        queried = []
        for clause in filter_part.split(";"):
            key, _, value = clause.partition("=")
            if key == "RID":
                queried.append(value)
        response.json.return_value = [
            {"RID": rid, "ID": f"token-{rid}"}
            for rid in queried
            if rid in found_rids
        ]
        return response

    catalog.get.side_effect = fake_get
    return catalog


def test_empty_entries_returns_none():
    from deriva_ml.execution.rid_lease import _validate_pending_asset_leases
    catalog = MagicMock()
    assert _validate_pending_asset_leases(catalog, []) is None
    catalog.get.assert_not_called()


def test_all_leases_valid_passes():
    from deriva_ml.execution.rid_lease import _validate_pending_asset_leases
    catalog = _fake_catalog({"1-ABC", "1-DEF"})
    entries = [("Image/a.png", "1-ABC"), ("Image/b.png", "1-DEF")]
    assert _validate_pending_asset_leases(catalog, entries) is None


def test_single_missing_lease_raises():
    from deriva_ml.execution.rid_lease import _validate_pending_asset_leases
    from deriva_ml.core.exceptions import DerivaMLValidationError
    catalog = _fake_catalog({"1-ABC"})  # 1-DEF NOT there
    entries = [("Image/a.png", "1-ABC"), ("Image/b.png", "1-DEF")]
    with pytest.raises(DerivaMLValidationError) as ei:
        _validate_pending_asset_leases(catalog, entries)
    msg = str(ei.value)
    assert "Image/b.png" in msg
    assert "1-DEF" in msg
    assert "Image/a.png" not in msg  # the valid one isn't listed


def test_multiple_missing_leases_aggregated():
    from deriva_ml.execution.rid_lease import _validate_pending_asset_leases
    from deriva_ml.core.exceptions import DerivaMLValidationError
    catalog = _fake_catalog(set())  # nothing found
    entries = [
        ("Image/z.png", "1-ZZZ"),
        ("Image/a.png", "1-AAA"),
    ]
    with pytest.raises(DerivaMLValidationError) as ei:
        _validate_pending_asset_leases(catalog, entries)
    msg = str(ei.value)
    assert "1-ZZZ" in msg
    assert "1-AAA" in msg
    # Sorted by (key, rid) — 'a.png' before 'z.png'.
    assert msg.index("Image/a.png") < msg.index("Image/z.png")


def test_batched_queries_use_chunk_size(monkeypatch):
    """More entries than chunk size → multiple catalog.get calls."""
    from deriva_ml.execution import rid_lease
    from deriva_ml.execution.rid_lease import _validate_pending_asset_leases

    # Set chunk size to 2 so 3 entries require 2 batched calls.
    monkeypatch.setattr(rid_lease, "PENDING_ROWS_LEASE_CHUNK", 2)
    catalog = _fake_catalog({"1-A", "1-B", "1-C"})
    entries = [("k1", "1-A"), ("k2", "1-B"), ("k3", "1-C")]
    _validate_pending_asset_leases(catalog, entries)
    # Expect 2 calls: first batch of 2, second batch of 1.
    assert catalog.get.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_pending_asset_lease_validator.py -v`

Expected: all 5 tests FAIL with `ImportError: cannot import name '_validate_pending_asset_leases'`.

- [ ] **Step 3: Implement the validator**

In `src/deriva_ml/execution/rid_lease.py`, append at the end of the file:

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
    from deriva_ml.core.exceptions import DerivaMLValidationError

    entries_list = list(entries)
    if not entries_list:
        return

    # Build a reverse map so we can attribute a missing RID back to
    # its caller-supplied key. If the same RID appears under two keys
    # (shouldn't happen in practice), the last one wins in the reverse
    # map — missing-list below iterates the forward list.
    rid_to_keys: dict[str, list[str]] = {}
    for key, rid in entries_list:
        rid_to_keys.setdefault(rid, []).append(key)

    all_rids = list(rid_to_keys.keys())
    found_rids: set[str] = set()

    for i in range(0, len(all_rids), PENDING_ROWS_LEASE_CHUNK):
        chunk = all_rids[i : i + PENDING_ROWS_LEASE_CHUNK]
        filter_clause = ";".join(f"RID={rid}" for rid in chunk)
        path = f"/entity/public:ERMrest_RID_Lease/{filter_clause}"
        response = catalog.get(path)
        for row in response.json():
            found_rids.add(row["RID"])

    missing: list[tuple[str, str]] = []
    for key, rid in entries_list:
        if rid not in found_rids:
            missing.append((key, rid))
    if not missing:
        return

    lines = [
        f"Missing or invalid pre-allocated RIDs for "
        f"{len(missing)} pending asset(s):"
    ]
    for key, rid in sorted(missing):
        lines.append(f"  - {key}: RID {rid} not found in ERMrest_RID_Lease")
    lines.append(
        "A pre-leased RID has become invalid (e.g., cleared from the "
        "lease table or never successfully POSTed). Restart the "
        "execution to re-lease, or investigate lease-table state."
    )
    raise DerivaMLValidationError("\n".join(lines))
```

Also add `Iterable` to the `typing` imports at the top of `rid_lease.py` if not already present. Current imports block:

```python
from typing import TYPE_CHECKING
```

becomes:

```python
from typing import TYPE_CHECKING, Iterable
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_pending_asset_lease_validator.py -v`

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && git add src/deriva_ml/execution/rid_lease.py tests/execution/test_pending_asset_lease_validator.py
git commit -m "feat(rid_lease): _validate_pending_asset_leases for pre-flight lease-table check"
```

---

## Task 6: Emit `use_pre_allocated_rid` in upload spec

**Files:**
- Modify: `src/deriva_ml/dataset/upload.py` — extend `asset_table_upload_spec()`.
- Test: `tests/asset/test_null_sentinel_processor.py` — append 3 tests.

Extend the regex with a RID capture group, add `RID` to `column_map`, emit the `use_pre_allocated_rid=True` flag.

- [ ] **Step 1: Write the failing tests**

Append to `tests/asset/test_null_sentinel_processor.py`:

```python
def test_asset_table_upload_spec_has_use_pre_allocated_rid_flag(test_ml):
    from deriva.core.ermrest_model import builtin_types
    from deriva_ml.core.definitions import ColumnDefinition
    from deriva_ml.dataset.upload import asset_table_upload_spec

    test_ml.create_asset(
        "UsePreAllocatedFlagTest",
        column_defs=[ColumnDefinition(name="foo", type=builtin_types.int4)],
    )
    spec = asset_table_upload_spec(test_ml.model, "UsePreAllocatedFlagTest")
    assert spec.get("use_pre_allocated_rid") is True


def test_asset_table_upload_spec_file_pattern_captures_rid(test_ml):
    from deriva.core.ermrest_model import builtin_types
    from deriva_ml.core.definitions import ColumnDefinition
    from deriva_ml.dataset.upload import asset_table_upload_spec

    test_ml.create_asset(
        "UsePreAllocatedRegexTest",
        column_defs=[ColumnDefinition(name="foo", type=builtin_types.int4)],
    )
    spec = asset_table_upload_spec(test_ml.model, "UsePreAllocatedRegexTest")
    pattern = spec["file_pattern"]
    # Regex must contain a (?P<RID>...) named group.
    assert "(?P<RID>" in pattern, f"file_pattern missing RID capture: {pattern}"


def test_asset_table_upload_spec_column_map_includes_rid(test_ml):
    from deriva.core.ermrest_model import builtin_types
    from deriva_ml.core.definitions import ColumnDefinition
    from deriva_ml.dataset.upload import asset_table_upload_spec

    test_ml.create_asset(
        "UsePreAllocatedColumnMapTest",
        column_defs=[ColumnDefinition(name="foo", type=builtin_types.int4)],
    )
    spec = asset_table_upload_spec(test_ml.model, "UsePreAllocatedColumnMapTest")
    assert spec["column_map"].get("RID") == "{RID}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/asset/test_null_sentinel_processor.py -v`

Expected: the 3 new tests FAIL (flag, regex group, and column_map entry all absent).

- [ ] **Step 3: Extend `asset_table_upload_spec`**

In `src/deriva_ml/dataset/upload.py`, locate `asset_table_upload_spec()` (around line 276). Find the block that builds `asset_path`:

```python
    metadata_path = "/".join([rf"(?P<{c}>[-:._ \w]+)" for c in metadata_columns])
    asset_path = f"{exec_dir_regex}/asset/{schema}/{asset_table.name}/{metadata_path}/{asset_file_regex}"
```

Replace with:

```python
    metadata_path = "/".join([rf"(?P<{c}>[-:._ \w]+)" for c in metadata_columns])
    # Bug E.2: capture pre-allocated RID as an additional path segment
    # after metadata columns and before the filename.
    rid_path = r"(?P<RID>[-A-Z0-9]+)"
    parts = [metadata_path, rid_path] if metadata_path else [rid_path]
    asset_path = (
        f"{exec_dir_regex}/asset/{schema}/{asset_table.name}/"
        f"{'/'.join(parts)}/{asset_file_regex}"
    )
```

Then find the `spec = {...}` dict. Extend `column_map` to include `RID`:

Current:
```python
        "column_map": {
            "MD5": "{md5}",
            "URL": "{URI}",
            "Length": "{file_size}",
            "Filename": "{file_name}",
        }
        | {c: f"{{{c}}}" for c in metadata_columns},
```

Change to:
```python
        "column_map": {
            "MD5": "{md5}",
            "URL": "{URI}",
            "Length": "{file_size}",
            "Filename": "{file_name}",
            "RID": "{RID}",  # Bug E.2: pre-allocated RID
        }
        | {c: f"{{{c}}}" for c in metadata_columns},
```

Finally, add the `use_pre_allocated_rid` flag to the spec dict. Find the end of the spec dict (after `record_query_template`) and add:

```python
    spec = {
        ...existing fields...
        "record_query_template": "/entity/{target_table}/MD5={md5}&Filename={file_name}",
        "use_pre_allocated_rid": True,  # Bug E.2: use caller-supplied RID
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/asset/test_null_sentinel_processor.py -v`

Expected: all tests pass (original 6 from Bug C + 3 new = 9).

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && git add src/deriva_ml/dataset/upload.py tests/asset/test_null_sentinel_processor.py
git commit -m "feat(dataset/upload): emit use_pre_allocated_rid + RID capture (Bug E.2)"
```

---

## Task 7: Append RID to staging path in `_build_upload_staging`

**Files:**
- Modify: `src/deriva_ml/execution/execution.py` — `_build_upload_staging` around line 1074.

Change the staging directory layout to append the pre-leased RID (from `entry.rid`) as the final segment before the filename.

- [ ] **Step 1: Update `_build_upload_staging`**

In `src/deriva_ml/execution/execution.py`, locate `_build_upload_staging` method (around line 1034). Find the block that builds `target_dir`:

```python
            all_metadata_cols = sorted(self._model.asset_metadata(asset_table_name))
            metadata_parts = [
                str(entry.metadata.get(k, NULL_SENTINEL)) for k in all_metadata_cols
            ] if all_metadata_cols else []
            target_dir = staging_root / entry.schema / asset_table_name
            for part in metadata_parts:
                target_dir = target_dir / part
            target_dir.mkdir(parents=True, exist_ok=True)
```

Change to:

```python
            all_metadata_cols = sorted(self._model.asset_metadata(asset_table_name))
            metadata_parts = [
                str(entry.metadata.get(k, NULL_SENTINEL)) for k in all_metadata_cols
            ] if all_metadata_cols else []
            target_dir = staging_root / entry.schema / asset_table_name
            for part in metadata_parts:
                target_dir = target_dir / part
            # Bug E.2: append pre-leased RID as the final path segment.
            # asset_table_upload_spec's file_pattern expects to capture
            # this as (?P<RID>[-A-Z0-9]+).
            target_dir = target_dir / entry.rid
            target_dir.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 2: Sanity import check**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && DERIVA_ML_ALLOW_DIRTY=true uv run python -c "from deriva_ml.execution.execution import Execution; print('ok')"`

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && git add src/deriva_ml/execution/execution.py
git commit -m "fix(execution): append pre-leased RID in _build_upload_staging (Bug E.2)"
```

---

## Task 8: Append RID to staging path in `_invoke_deriva_py_uploader`

**Files:**
- Modify: `src/deriva_ml/execution/upload_engine.py` — `_invoke_deriva_py_uploader._path_for` around line 680.

Same one-line addition for the engine-driven path.

- [ ] **Step 1: Update `_path_for`**

In `src/deriva_ml/execution/upload_engine.py`, locate `_invoke_deriva_py_uploader` (around line 562). Inside, find `_path_for`:

```python
        def _path_for(f: dict) -> Path:
            """Compute the regex-expected path for one input file."""
            src = Path(f["path"]).resolve()
            metadata = f.get("metadata") or {}
            target_dir = asset_root / schema_name / target_table
            for col in metadata_cols:
                target_dir = target_dir / str(metadata.get(col, NULL_SENTINEL))
            return target_dir / src.name
```

Change to:

```python
        def _path_for(f: dict) -> Path:
            """Compute the regex-expected path for one input file."""
            src = Path(f["path"]).resolve()
            metadata = f.get("metadata") or {}
            target_dir = asset_root / schema_name / target_table
            for col in metadata_cols:
                target_dir = target_dir / str(metadata.get(col, NULL_SENTINEL))
            # Bug E.2: pre-leased RID as the final segment before filename.
            target_dir = target_dir / f["rid"]
            return target_dir / src.name
```

- [ ] **Step 2: Sanity import check**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && DERIVA_ML_ALLOW_DIRTY=true uv run python -c "from deriva_ml.execution.upload_engine import run_upload_engine; print('ok')"`

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && git add src/deriva_ml/execution/upload_engine.py
git commit -m "fix(upload_engine): append pre-leased RID in _invoke_deriva_py_uploader (Bug E.2)"
```

---

## Task 9: Update `asset_file_path` helper in `dataset/upload.py`

**Files:**
- Modify: `src/deriva_ml/dataset/upload.py` — `asset_file_path` function around line 657.
- Modify: `src/deriva_ml/core/mixins/execution.py:229` — caller site.

The module-level `asset_file_path` helper (not the Execution method) is used only by `core/mixins/execution.py:229` for `Execution_Metadata` configuration. It needs a new `rid` parameter since the staging regex now requires it.

- [ ] **Step 1: Add `rid` parameter to `asset_file_path`**

In `src/deriva_ml/dataset/upload.py`, locate the `asset_file_path` function (around line 657):

```python
def asset_file_path(
    prefix: Path | str,
    exec_rid: RID,
    file_name: str | Path,
    asset_table: Table,
    metadata: dict | None = None,
) -> Path:
```

Add a `rid` parameter:

```python
def asset_file_path(
    prefix: Path | str,
    exec_rid: RID,
    file_name: str | Path,
    asset_table: Table,
    rid: RID,
    metadata: dict | None = None,
) -> Path:
```

And in the function body (around line 691), add the RID segment:

```python
    for m in asset_metadata:
        path = path / str(metadata.get(m, NULL_SENTINEL))
    # Bug E.2: append pre-allocated RID as final segment.
    path = path / rid
    path.mkdir(parents=True, exist_ok=True)
    return path / file_name
```

- [ ] **Step 2: Update the caller in `core/mixins/execution.py`**

In `src/deriva_ml/core/mixins/execution.py`, locate the call around line 229:

```python
        cfile = asset_file_path(
            prefix=self.working_dir,
            exec_rid=execution_rid,
            file_name="configuration.json",
            asset_table=self.model.name_to_table("Execution_Metadata"),
            metadata={},
        )
```

This call is reading an EXISTING configuration.json for an already-uploaded execution. The file's staging path on disk includes whatever RID it has — but we don't have that RID at this call site. We're trying to read it, not write it.

Read the function's purpose in context — it's looking up a config file that was created during a previous run. The current behavior was: path ends at `/configuration.json` after the metadata dirs. With the RID segment added, we need to either:

**A.** Glob under all RID dirs: `path / "*" / "configuration.json"` (ambiguous if multiple exist).
**B.** Pass the known RID from the caller's context.
**C.** Fall back to a wildcard-resolving helper specifically for this case.

Let me check the broader context. What's this function called from?

Reading lines around 229 in mixins/execution.py:

The function is `bind_execution` or similar — it loads a configuration for a RESUMED execution. The caller has `execution_rid` but not the `configuration.json`'s RID (which would be the RID of the Execution_Metadata row recording that file).

**The simplest compatible path:** for `Execution_Metadata` rows loaded this way, there's no metadata column and no pre-leased RID applicable — this is reading an existing on-disk file, not staging a new upload. Use a glob:

```python
        # Execution_Metadata for a resumed execution: the file lives under
        # a RID-segment directory that was written by a prior run. Glob
        # for it rather than requiring the caller to know the RID.
        em_table = self.model.name_to_table("Execution_Metadata")
        staging_base = (
            Path(self.working_dir) / "deriva-ml" / "execution" / execution_rid
            / "asset" / em_table.schema.name / em_table.name
        )
        candidates = list(staging_base.glob("*/configuration.json"))
        cfile = candidates[0] if candidates else (staging_base / "configuration.json")
```

Actually this is getting complex for a helper meant to just compute a path. Let me reconsider.

**Alternative simpler approach: don't change `asset_file_path`'s signature.** Instead, only callers that actually need the new RID segment (the two upload-side staging builders in Tasks 7 and 8) compute the path themselves. The `asset_file_path` helper stays as-is — it's no longer aligned with the upload regex, but it's not ON the upload path either.

Confirm by checking: is `asset_file_path` called anywhere else in the upload pipeline?

```bash
grep -rn "from deriva_ml.dataset.upload import asset_file_path\|dataset.upload\.asset_file_path\| asset_file_path(" src/deriva_ml/ tests/
```

Expected: only `core/mixins/execution.py:229` calls it.

If that's the only caller, and it's not part of the upload pipeline, we revert this task. The helper is then dead code with respect to the new regex, but still correctly finds the configuration.json it's looking for (since that file was saved BEFORE Bug E.2 with no RID segment).

**Decision:** skip this task. Leave `asset_file_path` unchanged. Verify in step 3 that no other callers exist.

- [ ] **Step 3: Verify `asset_file_path` has no other upload-path callers**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && grep -rn "asset_file_path" src/deriva_ml/ | grep -v "__pycache__" | grep -v "self\.asset_file_path\|exe\.asset_file_path\|execution\.asset_file_path"`

Expected output: only the definition at `src/deriva_ml/dataset/upload.py:657` and the one caller at `src/deriva_ml/core/mixins/execution.py:229`.

The `Execution.asset_file_path` method (different function, defined at `src/deriva_ml/execution/execution.py:1291`) uses the manifest + `_build_upload_staging` (Task 7) — that path IS covered.

- [ ] **Step 4: No code change needed — leave Task 9 as a verification-only task**

If step 3 confirms, this task is a no-op beyond the check. Commit nothing; log the decision:

```bash
cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && git log --oneline -1
```

(If step 3 surfaces other callers, this task expands to include those updates. Default plan assumes no other callers.)

---

## Task 10: Call lease validator at `_upload_execution_dirs`

**Files:**
- Modify: `src/deriva_ml/execution/execution.py` — `_upload_execution_dirs` around line 1100.

Add the lease validator call alongside the existing Bug C metadata validator.

- [ ] **Step 1: Add the validator call**

In `src/deriva_ml/execution/execution.py`, locate `_upload_execution_dirs` (around line 1100). Find the existing Bug C validator block:

```python
        # Bug C: refuse to upload if any pending asset is missing a
        # required (NOT-NULL) metadata column. This raises a single
        # DerivaMLValidationError that lists all failures at once.
        from deriva_ml.asset.manifest import _validate_pending_asset_metadata
        _validate_pending_asset_metadata(self._model, self._get_manifest())
```

Immediately after it, add:

```python
        # Bug E.2: confirm each pre-leased asset RID is still live in
        # ERMrest_RID_Lease. Catches stale SQLite state and out-of-band
        # lease-table edits before any hatrac I/O.
        from deriva_ml.execution.rid_lease import _validate_pending_asset_leases
        manifest = self._get_manifest()
        lease_entries = [
            (key, entry.rid)
            for key, entry in manifest.pending_assets().items()
            if entry.rid
        ]
        if lease_entries:
            _validate_pending_asset_leases(
                self._ml_object.catalog, lease_entries,
            )
```

- [ ] **Step 2: Sanity import check**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && DERIVA_ML_ALLOW_DIRTY=true uv run python -c "from deriva_ml.execution.execution import Execution; print('ok')"`

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && git add src/deriva_ml/execution/execution.py
git commit -m "feat(execution): validate pre-leased RIDs at _upload_execution_dirs (Bug E.2)"
```

---

## Task 11: Call lease validator at `run_upload_engine`

**Files:**
- Modify: `src/deriva_ml/execution/upload_engine.py` — `run_upload_engine` around line 290.

- [ ] **Step 1: Add the validator call**

In `src/deriva_ml/execution/upload_engine.py`, locate `run_upload_engine` (around line 236). The existing Bug C validator block iterates `list_pending_rows` once to build `validator_entries` and invokes `_validate_pending_asset_metadata_iter`. To avoid a second full iteration, restructure that block to collect both metadata entries AND lease entries in a single pass, then invoke both validators sequentially.

**Before** (existing Bug C code):

```python
    import json as _json
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata_iter
    store = ml.workspace.execution_state_store()
    _statuses_to_validate = [
        PendingRowStatus.staged,
        PendingRowStatus.leasing,
        PendingRowStatus.leased,
        PendingRowStatus.uploading,
    ]
    if retry_failed:
        _statuses_to_validate.append(PendingRowStatus.failed)
    if execution_rids is None:
        rids_for_validation = [row["rid"] for row in store.list_executions()]
    else:
        rids_for_validation = list(execution_rids)
    validator_entries: list[tuple[str, str, str, dict]] = []
    for rid in rids_for_validation:
        for row in store.list_pending_rows(
            execution_rid=rid, status=_statuses_to_validate,
        ):
            if not row.get("asset_file_path"):
                continue
            md = _json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            validator_entries.append((
                f"{row['execution_rid']}/{row['target_table']}/{row['key']}",
                row["target_schema"],
                row["target_table"],
                md,
            ))
    if validator_entries:
        _validate_pending_asset_metadata_iter(ml.model, validator_entries)
```

**After** (Bug C + Bug E.2 combined in one iteration):

```python
    import json as _json
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata_iter
    from deriva_ml.execution.rid_lease import _validate_pending_asset_leases
    store = ml.workspace.execution_state_store()
    _statuses_to_validate = [
        PendingRowStatus.staged,
        PendingRowStatus.leasing,
        PendingRowStatus.leased,
        PendingRowStatus.uploading,
    ]
    if retry_failed:
        _statuses_to_validate.append(PendingRowStatus.failed)
    if execution_rids is None:
        rids_for_validation = [row["rid"] for row in store.list_executions()]
    else:
        rids_for_validation = list(execution_rids)
    metadata_entries: list[tuple[str, str, str, dict]] = []
    lease_entries: list[tuple[str, str]] = []
    for rid in rids_for_validation:
        for row in store.list_pending_rows(
            execution_rid=rid, status=_statuses_to_validate,
        ):
            if not row.get("asset_file_path"):
                continue
            key_str = f"{row['execution_rid']}/{row['target_table']}/{row['key']}"
            md = _json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            metadata_entries.append((
                key_str,
                row["target_schema"],
                row["target_table"],
                md,
            ))
            if row.get("rid"):
                lease_entries.append((key_str, row["rid"]))
    if metadata_entries:
        _validate_pending_asset_metadata_iter(ml.model, metadata_entries)
    if lease_entries:
        _validate_pending_asset_leases(ml.catalog, lease_entries)
```

(`metadata_entries` is just a rename of `validator_entries` to be more specific.)

- [ ] **Step 2: Sanity import check**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && DERIVA_ML_ALLOW_DIRTY=true uv run python -c "from deriva_ml.execution.upload_engine import run_upload_engine; print('ok')"`

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && git add src/deriva_ml/execution/upload_engine.py
git commit -m "feat(upload_engine): validate pre-leased RIDs at run_upload_engine (Bug E.2)"
```

---

## Task 12: Live-catalog integration tests

**Files:**
- Create: `tests/execution/test_bug_e2_live_smoke.py` (new file).

Three tests, all gated on `DERIVA_HOST`.

- [ ] **Step 1: Write the tests**

Create `tests/execution/test_bug_e2_live_smoke.py`:

```python
"""Live-catalog integration tests for Bug E.2 (pre-allocated RID upload).

Gated on DERIVA_HOST. Three tests:

1. Happy path — upload uses pre-leased RID.
2. Missing-lease validation — RID not in ERMrest_RID_Lease raises.
3. Retry idempotent — partial failure + retry produces same catalog row.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import pytest


requires_catalog = pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="Bug E.2 live tests require DERIVA_HOST",
)


def _make_workflow(test_ml, name: str):
    from deriva_ml import MLVocab as vc
    test_ml.add_term(
        vc.workflow_type,
        "Test Workflow",
        description="for Bug E.2 live tests",
    )
    return test_ml.create_workflow(
        name=name,
        workflow_type="Test Workflow",
        description="for Bug E.2 live tests",
    )


def _lease_one_rid(test_ml) -> tuple[str, str]:
    """POST a single lease to ERMrest_RID_Lease; return (token, rid)."""
    from deriva_ml.execution.rid_lease import (
        generate_lease_token, post_lease_batch,
    )
    token = generate_lease_token()
    result = post_lease_batch(catalog=test_ml.catalog, tokens=[token])
    return token, result[token]


@requires_catalog
def test_upload_asset_uses_pre_leased_rid(test_ml, tmp_path):
    """End-to-end: stage an asset with a pre-leased RID; assert the
    catalog row has that RID (not a server-generated one)."""
    from deriva_ml.execution.state_store import PendingRowStatus

    f = tmp_path / "bug-e2-happy.bin"
    f.write_bytes(b"bug-e2 happy path " * 32)

    # Pre-lease a RID.
    token, leased_rid = _lease_one_rid(test_ml)

    wf = _make_workflow(test_ml, "Bug E2 happy path")
    exe = test_ml.create_execution(description="bug-e2-happy", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)

    with exe.execute():
        pass

    # Stage a pending row with the pre-leased RID.
    store.insert_pending_row(
        execution_rid=exe.execution_rid,
        key="k1",
        target_schema="deriva-ml",
        target_table="Execution_Asset",
        metadata_json=json.dumps({}),
        created_at=now,
        rid=leased_rid,
        status=PendingRowStatus.leased,
        lease_token=token,
        asset_file_path=str(f),
    )

    report = exe.upload_outputs()
    assert report.total_failed == 0, f"failures: {report.errors}"
    assert report.total_uploaded == 1

    # Catalog row must have our pre-leased RID.
    pb = test_ml.pathBuilder()
    asset_path = pb.schemas["deriva-ml"].tables["Execution_Asset"]
    rows = list(
        asset_path.filter(asset_path.RID == leased_rid)
        .entities().fetch()
    )
    assert len(rows) == 1, (
        f"Execution_Asset row with RID={leased_rid} not found; "
        f"got {len(rows)} results. Bug E.2 regression: server "
        f"substituted its own RID instead of honoring our lease."
    )
    # Sanity: MD5 matches what we uploaded.
    expected_md5 = hashlib.md5(f.read_bytes()).hexdigest()
    assert rows[0]["MD5"] == expected_md5


@requires_catalog
def test_upload_asset_raises_validation_when_lease_missing(test_ml, tmp_path):
    """Staging an asset with a bogus RID (never in ERMrest_RID_Lease)
    must raise DerivaMLValidationError before any hatrac I/O."""
    from deriva_ml.core.exceptions import DerivaMLValidationError
    from deriva_ml.execution.state_store import PendingRowStatus

    f = tmp_path / "bug-e2-missing.bin"
    f.write_bytes(b"bug-e2 missing lease " * 32)

    # A plausible-looking RID that was never leased.
    bogus_rid = "0-FAKEFAKE"

    wf = _make_workflow(test_ml, "Bug E2 missing lease")
    exe = test_ml.create_execution(description="bug-e2-missing", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)

    with exe.execute():
        pass

    store.insert_pending_row(
        execution_rid=exe.execution_rid,
        key="k2",
        target_schema="deriva-ml",
        target_table="Execution_Asset",
        metadata_json=json.dumps({}),
        created_at=now,
        rid=bogus_rid,
        status=PendingRowStatus.leased,
        lease_token="bogus-token",
        asset_file_path=str(f),
    )

    with pytest.raises(DerivaMLValidationError) as ei:
        exe.upload_outputs()
    msg = str(ei.value)
    assert bogus_rid in msg
    assert "ERMrest_RID_Lease" in msg


@requires_catalog
def test_upload_asset_retry_idempotent_with_existing_row(test_ml, tmp_path):
    """Simulate partial success: pre-insert a row at the pre-leased RID.
    A subsequent upload_outputs must detect the existing row and report
    success without re-insert."""
    from deriva_ml.execution.state_store import PendingRowStatus

    f = tmp_path / "bug-e2-retry.bin"
    f.write_bytes(b"bug-e2 retry " * 32)

    token, leased_rid = _lease_one_rid(test_ml)

    wf = _make_workflow(test_ml, "Bug E2 retry")
    exe = test_ml.create_execution(description="bug-e2-retry", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)

    with exe.execute():
        pass

    # Pre-populate a catalog row at the pre-leased RID, simulating a
    # prior successful insert that the client didn't record locally.
    expected_md5 = hashlib.md5(f.read_bytes()).hexdigest()
    pb = test_ml.pathBuilder()
    asset_path = pb.schemas["deriva-ml"].tables["Execution_Asset"]
    asset_path.insert([{
        "RID": leased_rid,
        "Filename": "bug-e2-retry.bin",
        "MD5": expected_md5,
        "Length": f.stat().st_size,
        "URL": "/hatrac/fake-pre-existing-uri",
    }])

    # Now stage the row as pending with the SAME RID.
    store.insert_pending_row(
        execution_rid=exe.execution_rid,
        key="k3",
        target_schema="deriva-ml",
        target_table="Execution_Asset",
        metadata_json=json.dumps({}),
        created_at=now,
        rid=leased_rid,
        status=PendingRowStatus.leased,
        lease_token=token,
        asset_file_path=str(f),
    )

    # Retry must succeed (idempotency pre-check finds the existing row).
    report = exe.upload_outputs()
    assert report.total_failed == 0, f"retry failures: {report.errors}"
    # At least one row reported uploaded (exact count semantics depend on
    # whether deriva-py counts "existing" as uploaded, which is fine).
    assert report.total_uploaded >= 1

    # Exactly ONE catalog row for this RID — no duplicates.
    rows = list(
        asset_path.filter(asset_path.RID == leased_rid)
        .entities().fetch()
    )
    assert len(rows) == 1
```

- [ ] **Step 2: Run the tests**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/test_bug_e2_live_smoke.py -v`

Expected: all 3 tests pass.

**If the tests fail with `DerivaUploadConfigurationError: ...use_pre_allocated_rid=true but no RID...` or similar:** the deriva-py PR (Tasks 1-4) hasn't landed yet. Check `uv.lock` to see which commit of deriva-py is pinned; if it predates the Phase 1 merge, run `uv lock --upgrade-package deriva` to pick up the latest `2.0-dev` head.

**If the tests fail with `Server substituted its own RID` in test #1:** deriva-ml's upload spec isn't emitting the flag OR the staging path doesn't include the RID segment. Verify Tasks 6-8 landed correctly.

- [ ] **Step 3: Broader regression**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/asset/ tests/execution/test_bug_c_live_smoke.py tests/execution/test_bug_e2_live_smoke.py tests/execution/test_pending_asset_lease_validator.py -q`

Expected: all tests pass (Bug C tests continue to pass — orthogonal fix).

- [ ] **Step 4: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && git add tests/execution/test_bug_e2_live_smoke.py
git commit -m "test(bug_e2): live-catalog integration (happy, missing-lease, retry-idempotent)"
```

---

## Task 13: CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md` — prepend new Unreleased section.

- [ ] **Step 1: Add CHANGELOG entry**

In `CHANGELOG.md`, immediately after the title/preamble and BEFORE the first existing `## Unreleased —` section, insert:

```markdown
## Unreleased — Bug E.2: asset uploads honor pre-leased RIDs

### Fixed

- **Asset-table catalog inserts now use the caller-supplied pre-leased RID** instead of silently accepting a server-generated one. Previously, the plain-row drain path honored `pending_rows.rid` but the asset-row path (via deriva-py's `GenericUploader`) looked up or created rows by MD5+Filename, discarding the pre-leased RID. Any client code that captured the pre-leased RID (e.g., for FK references) ended up with a dangling pointer. Fix is a two-repo change:
  - **deriva-py** (`2.0-dev`): new opt-in `use_pre_allocated_rid: true` flag on asset mappings. When set, `_uploadAsset` bypasses `_getFileRecord` (MD5+Filename lookup) and routes to a new `_createFileRecordWithRid` method that does a RID-keyed existence check for idempotent retry, then creates with the caller-supplied RID in the payload. Legacy callers (flag absent) see no change.
  - **deriva-ml**: `asset_table_upload_spec` emits the new flag automatically, adds `(?P<RID>[-A-Z0-9]+)` to the `file_pattern` regex, and includes `"RID": "{RID}"` in the `column_map`. The two staging path-builders (`Execution._build_upload_staging`, `_invoke_deriva_py_uploader`) append the pre-leased RID as the final directory segment.

### Added

- **`_validate_pending_asset_leases(catalog, entries)`** — pre-flight validator in `deriva_ml.execution.rid_lease`. Batch-queries `public:ERMrest_RID_Lease` (chunked by `PENDING_ROWS_LEASE_CHUNK`) and raises an aggregated `DerivaMLValidationError` listing every pre-leased RID that is no longer present on the server. Called at the top of `Execution._upload_execution_dirs` and `run_upload_engine`, right after Bug C's metadata validator.

### External-caller impact

No action required. Any caller using the standard `asset_file_path` → `upload_execution_outputs` flow is automatically opted into pre-allocated-RID semantics. Callers who had captured a pre-leased RID for cross-references will find that RID now matches what's in the catalog after upload.

Edge case: if a pre-leased RID has been cleared from `ERMrest_RID_Lease` (e.g., by manual cleanup or a dropped catalog), the upload is refused at pre-flight with a clear error message. Restart the execution to re-lease.
```

- [ ] **Step 2: Verify file still parses**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && head -40 CHANGELOG.md`

Expected: new section appears between preamble and the first existing Unreleased section.

- [ ] **Step 3: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && git add CHANGELOG.md
git commit -m "docs(changelog): Bug E.2 asset uploads honor pre-leased RIDs"
```

---

## Final verification

- [ ] **Step 1: deriva-py tests**

Run: `cd /Users/carl/GitHub/deriva-py && python -m pytest tests/deriva/transfer/upload/test_pre_allocated_rid.py -v`

Expected: 6/6 tests pass.

- [ ] **Step 2: deriva-ml unit tier**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_pending_asset_lease_validator.py tests/asset/test_null_sentinel_processor.py tests/asset/test_metadata_validator.py tests/core/test_schema_cache.py tests/core/test_schema_diff.py tests/core/test_exceptions.py -q`

Expected: all tests pass. (Some may skip without DERIVA_HOST — acceptable.)

- [ ] **Step 3: deriva-ml live tier**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/asset/ tests/execution/test_bug_c_live_smoke.py tests/execution/test_bug_e2_live_smoke.py tests/execution/test_pending_asset_lease_validator.py -q`

Expected: all tests pass.

- [ ] **Step 4: Ruff check**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-e && uv run ruff check src/deriva_ml/execution/rid_lease.py src/deriva_ml/execution/execution.py src/deriva_ml/execution/upload_engine.py src/deriva_ml/dataset/upload.py`

Expected: no new warnings on Bug E.2 changes. Pre-existing unrelated warnings are fine.

---

## Commit summary (expected)

### deriva-py side (Phase 1)

1. `docs(spec): Bug E.2 pre-allocated RID upload design` (already committed on deriva-ml worktree)
2. `feat(upload): fast-fail when use_pre_allocated_rid flag has no RID`
3. `feat(upload): _createFileRecordWithRid bypasses MD5+Filename lookup`
4. `feat(upload): route _uploadAsset Step 7 to _createFileRecordWithRid when flag set`

Then: PR against `2.0-dev`, merge.

### deriva-ml side (Phase 2)

5. `feat(rid_lease): _validate_pending_asset_leases for pre-flight lease-table check`
6. `feat(dataset/upload): emit use_pre_allocated_rid + RID capture (Bug E.2)`
7. `fix(execution): append pre-leased RID in _build_upload_staging (Bug E.2)`
8. `fix(upload_engine): append pre-leased RID in _invoke_deriva_py_uploader (Bug E.2)`
9. `feat(execution): validate pre-leased RIDs at _upload_execution_dirs (Bug E.2)`
10. `feat(upload_engine): validate pre-leased RIDs at run_upload_engine (Bug E.2)`
11. `test(bug_e2): live-catalog integration (happy, missing-lease, retry-idempotent)`
12. `docs(changelog): Bug E.2 asset uploads honor pre-leased RIDs`

Total: 3 deriva-py commits (after spec) + 8 deriva-ml commits = 11 implementation commits across the two repos.
