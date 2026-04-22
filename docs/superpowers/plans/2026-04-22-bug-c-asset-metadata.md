# Bug C: Asset Metadata None-Stringification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the pre-S3 regression where missing asset metadata becomes the literal string `"None"` in catalog inserts. Replace with pre-upload validation for NOT-NULL columns plus a sentinel-based SQL-NULL path for nullable columns.

**Architecture:** Three units. Add a private validator function in `asset/manifest.py` that inspects `col.nullok` via a new `DerivaModel.asset_metadata_columns()` helper. Define a `NULL_SENTINEL = "__NULL__"` constant in `dataset/upload.py` replacing the current `"None"` string at three staging sites. Add a `NullSentinelProcessor(BaseProcessor)` to deriva-py's upload pre-processor pipeline, wired via `asset_table_upload_spec`, to translate the sentinel to Python `None` before catalog insert.

**Tech Stack:** Python ≥3.12, Pydantic v2, deriva-py (`BaseProcessor`, `find_processor`, `GenericUploader`), pytest. Built on the existing upload pipeline without modifying deriva-py.

**Reference spec:** `docs/superpowers/specs/2026-04-22-bug-c-asset-metadata-design.md`

---

## File Structure

- Modify: `src/deriva_ml/model/catalog.py` — add `DerivaModel.asset_metadata_columns(table) -> list[Column]` alongside existing `asset_metadata()`.
- Modify: `src/deriva_ml/asset/manifest.py` — add private module-level `_validate_pending_asset_metadata(model, manifest)` function.
- Create: `src/deriva_ml/asset/null_sentinel_processor.py` — `NullSentinelProcessor(BaseProcessor)` class.
- Modify: `src/deriva_ml/dataset/upload.py` — add `NULL_SENTINEL` constant at module top, wire `pre_processors` into `asset_table_upload_spec`, replace `"None"` at line 691.
- Modify: `src/deriva_ml/execution/execution.py` — replace `"None"` at line 1074 (in `_build_upload_staging`); call validator at top of `_upload_execution_dirs`.
- Modify: `src/deriva_ml/execution/upload_engine.py` — replace `"None"` at line 644 (in `_invoke_deriva_py_uploader`); call validator at top of `run_upload_engine`.
- Create: `tests/asset/test_metadata_validator.py` — 8 unit tests for validator.
- Create: `tests/asset/test_null_sentinel_processor.py` — 4 unit tests for processor.
- Create: `tests/execution/test_bug_c_live_smoke.py` — 3 integration tests (live-catalog, gated on `DERIVA_HOST`).
- Modify: `CHANGELOG.md` — Unreleased entry.

---

## Task 1: Add `DerivaModel.asset_metadata_columns()` helper

**Files:**
- Modify: `src/deriva_ml/model/catalog.py` — add new method after existing `asset_metadata()` (line 580)
- Test: `tests/model/test_asset_metadata_columns.py` (new file)

Returns `list[Column]` (sorted by name) for asset-metadata columns. Used by the validator to inspect `nullok`. The existing `asset_metadata(table) -> set[str]` remains unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/model/test_asset_metadata_columns.py`:

```python
"""Unit tests for DerivaModel.asset_metadata_columns."""
from __future__ import annotations

import pytest


def test_asset_metadata_columns_returns_column_objects(test_ml):
    """For an asset table with metadata cols, returns sorted Column objects."""
    # Use Execution_Metadata which is a known asset table with known columns.
    # We're not checking specific columns here; we're checking shape.
    cols = test_ml.model.asset_metadata_columns("Execution_Metadata")
    assert isinstance(cols, list)
    for c in cols:
        assert hasattr(c, "name")
        assert hasattr(c, "nullok")


def test_asset_metadata_columns_sorted_by_name(test_ml):
    cols = test_ml.model.asset_metadata_columns("Execution_Metadata")
    names = [c.name for c in cols]
    assert names == sorted(names)


def test_asset_metadata_columns_excludes_standard_asset_columns(test_ml):
    """Must not include Filename, URL, Length, MD5, Description, or system columns."""
    cols = test_ml.model.asset_metadata_columns("Execution_Metadata")
    names = {c.name for c in cols}
    for forbidden in ("Filename", "URL", "Length", "MD5", "Description",
                      "RID", "RCT", "RMT", "RCB", "RMB"):
        assert forbidden not in names


def test_asset_metadata_columns_matches_asset_metadata_set(test_ml):
    """The column-object list names must equal asset_metadata()'s set."""
    cols = test_ml.model.asset_metadata_columns("Execution_Metadata")
    names = {c.name for c in cols}
    assert names == test_ml.model.asset_metadata("Execution_Metadata")


def test_asset_metadata_columns_raises_for_non_asset_table(test_ml):
    from deriva_ml.core.exceptions import DerivaMLTableTypeError
    with pytest.raises(DerivaMLTableTypeError):
        test_ml.model.asset_metadata_columns("Workflow")   # not an asset
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/model/test_asset_metadata_columns.py -v`

Expected: FAIL with `AttributeError: 'DerivaModel' object has no attribute 'asset_metadata_columns'`.

- [ ] **Step 3: Implement the helper**

In `src/deriva_ml/model/catalog.py`, immediately after the existing `asset_metadata()` method (ending around line 587), insert:

```python
    def asset_metadata_columns(self, table: str | Table) -> list[Column]:
        """Return Column objects for the asset-metadata columns of ``table``.

        Like :meth:`asset_metadata` but returns the :class:`Column`
        instances (not just names) so callers can inspect attributes
        such as ``nullok``. Results are sorted by column name for
        deterministic iteration.

        Args:
            table: Asset table name or Table object.

        Returns:
            Sorted list of Column objects.

        Raises:
            DerivaMLTableTypeError: If ``table`` is not an asset table.
        """
        table = self.name_to_table(table)
        if not self.is_asset(table):
            raise DerivaMLTableTypeError("asset table", table.name)
        return sorted(
            (c for c in table.columns if c.name not in DerivaAssetColumns),
            key=lambda c: c.name,
        )
```

Also verify `Column` is imported at the top of `catalog.py`. If not, add `from deriva.core.ermrest_model import Column, Table` to the existing import (most likely it's already imported).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/model/test_asset_metadata_columns.py -v`

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && git add src/deriva_ml/model/catalog.py tests/model/test_asset_metadata_columns.py
git commit -m "feat(model): add asset_metadata_columns() helper returning Column objects"
```

---

## Task 2: Add `_validate_pending_asset_metadata` to `asset/manifest.py`

**Files:**
- Modify: `src/deriva_ml/asset/manifest.py` — add private module-level function
- Test: `tests/asset/test_metadata_validator.py` (new file)

Validator iterates pending manifest entries, collects per-entry missing-NOT-NULL-column errors, raises aggregated `DerivaMLValidationError` if any.

- [ ] **Step 1: Write the failing tests**

Create `tests/asset/test_metadata_validator.py`:

```python
"""Unit tests for _validate_pending_asset_metadata."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _col(name: str, nullok: bool = True):
    """Build a mock Column with name + nullok."""
    c = MagicMock()
    c.name = name
    c.nullok = nullok
    return c


def _fake_model(columns_by_table: dict[str, list]):
    """Build a fake DerivaModel.asset_metadata_columns() for the given
    mapping of table_name -> list of mock Column objects."""
    model = MagicMock()
    model.asset_metadata_columns.side_effect = lambda t: columns_by_table.get(t, [])
    return model


def _fake_manifest(entries: dict):
    """Build a fake AssetManifest.pending_assets() that returns the
    given dict of {key: AssetEntry}."""
    from deriva_ml.asset.manifest import AssetEntry
    manifest = MagicMock()
    manifest.pending_assets.return_value = entries
    return manifest


def _entry(asset_table: str, schema: str = "test-schema", metadata: dict | None = None):
    from deriva_ml.asset.manifest import AssetEntry
    return AssetEntry(
        asset_table=asset_table,
        schema=schema,
        metadata=metadata or {},
    )


def test_empty_manifest_returns_none():
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata
    model = _fake_model({})
    manifest = _fake_manifest({})
    # Should not raise.
    assert _validate_pending_asset_metadata(model, manifest) is None


def test_asset_with_no_metadata_columns_passes():
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata
    model = _fake_model({"Execution_Asset": []})
    manifest = _fake_manifest({
        "Execution_Asset/f.bin": _entry("Execution_Asset"),
    })
    assert _validate_pending_asset_metadata(model, manifest) is None


def test_all_required_metadata_present_passes():
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata
    model = _fake_model({
        "Image": [_col("Acquisition_Time", nullok=False)],
    })
    manifest = _fake_manifest({
        "Image/a.png": _entry("Image", metadata={"Acquisition_Time": "2026-01-01"}),
    })
    assert _validate_pending_asset_metadata(model, manifest) is None


def test_missing_single_not_null_column_raises():
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata
    from deriva_ml.core.exceptions import DerivaMLValidationError
    model = _fake_model({
        "Image": [_col("Acquisition_Time", nullok=False)],
    })
    manifest = _fake_manifest({
        "Image/a.png": _entry("Image", metadata={}),
    })
    with pytest.raises(DerivaMLValidationError) as ei:
        _validate_pending_asset_metadata(model, manifest)
    msg = str(ei.value)
    assert "Image/a.png" in msg
    assert "Acquisition_Time" in msg


def test_missing_multiple_columns_aggregated():
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata
    from deriva_ml.core.exceptions import DerivaMLValidationError
    model = _fake_model({
        "Image": [
            _col("Acquisition_Date", nullok=False),
            _col("Acquisition_Time", nullok=False),
        ],
    })
    manifest = _fake_manifest({
        "Image/a.png": _entry("Image", metadata={}),
    })
    with pytest.raises(DerivaMLValidationError) as ei:
        _validate_pending_asset_metadata(model, manifest)
    msg = str(ei.value)
    # Both columns listed, sorted order
    assert "Acquisition_Date" in msg
    assert "Acquisition_Time" in msg
    assert msg.index("Acquisition_Date") < msg.index("Acquisition_Time")


def test_missing_across_multiple_assets_aggregated():
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata
    from deriva_ml.core.exceptions import DerivaMLValidationError
    model = _fake_model({
        "Image": [_col("Acquisition_Time", nullok=False)],
        "Plate": [_col("Well_Position", nullok=False)],
    })
    manifest = _fake_manifest({
        "Image/a.png": _entry("Image", metadata={}),
        "Plate/p.json": _entry("Plate", metadata={}),
    })
    with pytest.raises(DerivaMLValidationError) as ei:
        _validate_pending_asset_metadata(model, manifest)
    msg = str(ei.value)
    assert "Image/a.png" in msg
    assert "Plate/p.json" in msg
    assert "Acquisition_Time" in msg
    assert "Well_Position" in msg


def test_nullable_missing_is_not_an_error():
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata
    model = _fake_model({
        "Image": [_col("Description_Note", nullok=True)],  # nullable
    })
    manifest = _fake_manifest({
        "Image/a.png": _entry("Image", metadata={}),  # no value supplied
    })
    # Nullable missing is fine — sentinel path handles it.
    assert _validate_pending_asset_metadata(model, manifest) is None


def test_error_message_is_deterministic():
    """Same manifest twice → byte-identical error messages (sorted)."""
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata
    from deriva_ml.core.exceptions import DerivaMLValidationError
    model = _fake_model({
        "Image": [
            _col("Zeta", nullok=False),
            _col("Alpha", nullok=False),
        ],
    })
    manifest = _fake_manifest({
        "Image/z.png": _entry("Image", metadata={}),
        "Image/a.png": _entry("Image", metadata={}),
    })
    msgs = []
    for _ in range(2):
        try:
            _validate_pending_asset_metadata(model, manifest)
        except DerivaMLValidationError as e:
            msgs.append(str(e))
    assert msgs[0] == msgs[1]
    # Sorted: Alpha before Zeta, a.png before z.png
    assert msgs[0].index("Image/a.png") < msgs[0].index("Image/z.png")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/asset/test_metadata_validator.py -v`

Expected: all 8 tests FAIL with `ImportError: cannot import name '_validate_pending_asset_metadata'`.

- [ ] **Step 3: Implement `_validate_pending_asset_metadata`**

In `src/deriva_ml/asset/manifest.py`, append a new private module-level function at the end of the file:

```python
def _validate_pending_asset_metadata(
    model: "DerivaModel",
    manifest: "AssetManifest",
) -> None:
    """Raise DerivaMLValidationError if any pending asset entry is
    missing a NOT-NULL metadata column value.

    Iterates manifest entries where status == pending and the asset
    table has metadata columns. For each NOT-NULL column absent from
    the entry's metadata dict, records (manifest_key, schema, table,
    column). If any errors collected, raises a single
    :class:`DerivaMLValidationError` whose message lists every
    failure in sorted order.

    Nullable columns may be absent without error; downstream staging
    substitutes ``NULL_SENTINEL`` which the upload pre-processor
    translates to SQL NULL.
    """
    from deriva_ml.core.exceptions import DerivaMLValidationError

    # Map manifest_key -> sorted list of missing NOT-NULL column names
    missing_by_key: dict[str, list[str]] = {}

    for key, entry in sorted(manifest.pending_assets().items()):
        cols = model.asset_metadata_columns(entry.asset_table)
        if not cols:
            continue
        provided = set(entry.metadata.keys())
        missing: list[str] = []
        for col in cols:
            if not col.nullok and col.name not in provided:
                missing.append(col.name)
        if missing:
            missing_by_key[key] = sorted(missing)

    if not missing_by_key:
        return

    lines = [
        f"Missing required metadata for {len(missing_by_key)} pending asset(s):"
    ]
    for key in sorted(missing_by_key.keys()):
        cols = missing_by_key[key]
        noun = "column" if len(cols) == 1 else "columns"
        lines.append(f"  - {key}: missing {noun} {', '.join(cols)}")
    lines.append(
        "Supply these values before calling upload_outputs(), either via "
        "the ``metadata=`` arg to asset_file_path(...) or by assigning "
        "to the returned AssetFilePath's ``metadata`` property."
    )
    raise DerivaMLValidationError("\n".join(lines))
```

Also add the TYPE_CHECKING import block at the top of `manifest.py` if not already present:

```python
if TYPE_CHECKING:
    from deriva_ml.model.catalog import DerivaModel
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/asset/test_metadata_validator.py -v`

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && git add src/deriva_ml/asset/manifest.py tests/asset/test_metadata_validator.py
git commit -m "feat(asset/manifest): _validate_pending_asset_metadata for NOT-NULL columns"
```

---

## Task 3: Add `NullSentinelProcessor`

**Files:**
- Create: `src/deriva_ml/asset/null_sentinel_processor.py`
- Test: `tests/asset/test_null_sentinel_processor.py` (new file)

deriva-py `BaseProcessor` subclass that rewrites `"__NULL__"` → `None` in `self.metadata` before catalog insert.

- [ ] **Step 1: Write the failing tests**

Create `tests/asset/test_null_sentinel_processor.py`:

```python
"""Unit tests for NullSentinelProcessor."""
from __future__ import annotations


def test_single_sentinel_converted_to_none():
    from deriva_ml.asset.null_sentinel_processor import NullSentinelProcessor
    metadata = {"Acquisition_Time": "__NULL__", "Acquisition_Date": "2026-01-01"}
    proc = NullSentinelProcessor(metadata=metadata)
    proc.process()
    assert metadata["Acquisition_Time"] is None
    assert metadata["Acquisition_Date"] == "2026-01-01"


def test_multiple_sentinels_in_metadata_all_converted():
    from deriva_ml.asset.null_sentinel_processor import NullSentinelProcessor
    metadata = {"a": "__NULL__", "b": "__NULL__", "c": "real"}
    proc = NullSentinelProcessor(metadata=metadata)
    proc.process()
    assert metadata["a"] is None
    assert metadata["b"] is None
    assert metadata["c"] == "real"


def test_non_sentinel_values_unchanged():
    from deriva_ml.asset.null_sentinel_processor import NullSentinelProcessor
    metadata = {
        "x": "hello",
        "y": 42,
        "z": None,         # already None — stays None
        "q": "__NUL__",    # almost-sentinel — stays as-is
        "r": "__NULLISH__",
    }
    proc = NullSentinelProcessor(metadata=metadata)
    proc.process()
    assert metadata["x"] == "hello"
    assert metadata["y"] == 42
    assert metadata["z"] is None
    assert metadata["q"] == "__NUL__"
    assert metadata["r"] == "__NULLISH__"


def test_empty_metadata_is_no_op():
    from deriva_ml.asset.null_sentinel_processor import NullSentinelProcessor
    metadata: dict = {}
    proc = NullSentinelProcessor(metadata=metadata)
    proc.process()
    assert metadata == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/asset/test_null_sentinel_processor.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'deriva_ml.asset.null_sentinel_processor'`.

- [ ] **Step 3: Create the processor module**

Create `src/deriva_ml/asset/null_sentinel_processor.py`:

```python
"""Pre-upload metadata pre-processor for deriva-py's GenericUploader.

Part of the Bug C fix — see
``docs/superpowers/specs/2026-04-22-bug-c-asset-metadata-design.md``.
"""
from __future__ import annotations

from deriva.transfer.upload.processors import BaseProcessor


class NullSentinelProcessor(BaseProcessor):
    """Translate the ``"__NULL__"`` sentinel to Python ``None``.

    Runs before deriva-py's ``interpolateDict`` expansion. Mutates
    the in-flight ``self.metadata`` dict in place: any value equal
    to ``"__NULL__"`` (see
    :data:`deriva_ml.dataset.upload.NULL_SENTINEL`) is replaced
    with ``None``. deriva-py then drops None-valued keys, causing
    the resulting catalog insert to send SQL ``NULL`` for those
    columns.

    Not part of the end-user API — configured automatically by
    :func:`deriva_ml.dataset.upload.asset_table_upload_spec`.

    Note: if a user's legitimate metadata value equals the sentinel
    string, it will be corrupted to NULL. Known constraint; chosen
    sentinel is unlikely to collide naturally.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # BaseProcessor stores kwargs on self.kwargs; the metadata
        # dict is passed via kwargs["metadata"] by
        # deriva-py's _execute_processors.
        self.metadata = kwargs.get("metadata", {})

    def process(self):
        for k, v in list(self.metadata.items()):
            if v == "__NULL__":
                self.metadata[k] = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/asset/test_null_sentinel_processor.py -v`

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && git add src/deriva_ml/asset/null_sentinel_processor.py tests/asset/test_null_sentinel_processor.py
git commit -m "feat(asset): NullSentinelProcessor for SQL NULL emission on nullable cols"
```

---

## Task 4: Define `NULL_SENTINEL` and wire `pre_processors` into `asset_table_upload_spec`

**Files:**
- Modify: `src/deriva_ml/dataset/upload.py`

Add the shared constant at module top. Emit `pre_processors` in the upload spec when the asset table has metadata columns.

- [ ] **Step 1: Add `NULL_SENTINEL` constant at the top of `src/deriva_ml/dataset/upload.py`**

Locate the module docstring / import block at the top of `src/deriva_ml/dataset/upload.py`. After the imports, before the first function/constant, insert:

```python
NULL_SENTINEL = "__NULL__"
"""Directory-segment marker for nullable asset-metadata columns with
no value. Written into the staging tree by the three path-builders
(``Execution._build_upload_staging``, ``_invoke_deriva_py_uploader``,
``asset_file_path``) and translated back to Python ``None`` by
:class:`deriva_ml.asset.null_sentinel_processor.NullSentinelProcessor`
before deriva-py builds the catalog insert. See Bug C design doc."""
```

- [ ] **Step 2: Wire `pre_processors` into `asset_table_upload_spec`**

Find `asset_table_upload_spec()` in the same file (~line 276). Locate the spec dict being built; inside the spec dict add `pre_processors` entry, only when `metadata_columns` is non-empty.

Change:

```python
    spec = {
        # Upload assets into an asset table of an asset table.
        "column_map": {
            "MD5": "{md5}",
            "URL": "{URI}",
            "Length": "{file_size}",
            "Filename": "{file_name}",
        }
        | {c: f"{{{c}}}" for c in metadata_columns},
        "file_pattern": asset_path,
        ...
    }
    return spec
```

to:

```python
    spec = {
        # Upload assets into an asset table of an asset table.
        "column_map": {
            "MD5": "{md5}",
            "URL": "{URI}",
            "Length": "{file_size}",
            "Filename": "{file_name}",
        }
        | {c: f"{{{c}}}" for c in metadata_columns},
        "file_pattern": asset_path,
        ...
    }
    # Wire the NullSentinelProcessor only when the table has metadata
    # columns — otherwise no sentinel values can appear and the
    # processor would be a no-op.
    if metadata_columns:
        spec["pre_processors"] = [
            {
                "processor": "NullSentinelProcessor",
                "processor_type": (
                    "deriva_ml.asset.null_sentinel_processor."
                    "NullSentinelProcessor"
                ),
            }
        ]
    return spec
```

- [ ] **Step 3: Write a unit test that verifies the spec includes `pre_processors` for tables with metadata**

Append to `tests/asset/test_null_sentinel_processor.py`:

```python
def test_asset_table_upload_spec_includes_null_sentinel_processor_when_metadata_present(test_ml):
    """Upload spec for an asset table with metadata wires the processor."""
    from deriva_ml.dataset.upload import asset_table_upload_spec

    # Execution_Metadata has metadata columns — expect the processor wired.
    spec = asset_table_upload_spec(test_ml.model, "Execution_Metadata")
    pre = spec.get("pre_processors", [])
    types = [p.get("processor_type") for p in pre]
    assert any(
        t and t.endswith("NullSentinelProcessor") for t in types
    ), f"pre_processors must wire NullSentinelProcessor; got {pre}"


def test_asset_table_upload_spec_omits_pre_processors_when_no_metadata(test_ml):
    """Asset tables with zero metadata columns don't need the processor."""
    from deriva_ml.dataset.upload import asset_table_upload_spec

    # Execution_Asset has NO metadata columns (per the S3 live smoke).
    spec = asset_table_upload_spec(test_ml.model, "Execution_Asset")
    assert "pre_processors" not in spec or not spec["pre_processors"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/asset/test_null_sentinel_processor.py -v`

Expected: all 6 tests pass (4 original + 2 new).

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && git add src/deriva_ml/dataset/upload.py tests/asset/test_null_sentinel_processor.py
git commit -m "feat(dataset/upload): NULL_SENTINEL constant + wire NullSentinelProcessor into asset_table_upload_spec"
```

---

## Task 5: Replace `"None"` with `NULL_SENTINEL` at three staging sites

**Files:**
- Modify: `src/deriva_ml/dataset/upload.py:691` (staging-site A)
- Modify: `src/deriva_ml/execution/execution.py:1074` (staging-site B)
- Modify: `src/deriva_ml/execution/upload_engine.py:644` (staging-site C)

Three single-line changes. No new tests needed — the end-to-end integration tests (Task 8) verify the full round-trip.

- [ ] **Step 1: Edit `dataset/upload.py` staging site (line 691)**

In `src/deriva_ml/dataset/upload.py`, locate the `asset_file_path` function (~line 657). Change:

```python
    for m in asset_metadata:
        path = path / str(metadata.get(m, "None"))
```

to:

```python
    for m in asset_metadata:
        path = path / str(metadata.get(m, NULL_SENTINEL))
```

(`NULL_SENTINEL` is already defined at the top of this module — no import needed.)

- [ ] **Step 2: Edit `execution/execution.py` staging site (line 1074)**

In `src/deriva_ml/execution/execution.py`, locate `_build_upload_staging` (~line 1034). Find the existing line:

```python
            metadata_parts = [
                str(entry.metadata.get(k, "None")) for k in all_metadata_cols
            ] if all_metadata_cols else []
```

Change to:

```python
            metadata_parts = [
                str(entry.metadata.get(k, NULL_SENTINEL)) for k in all_metadata_cols
            ] if all_metadata_cols else []
```

Also add the import to the top of `execution.py` (look for the `from deriva_ml.dataset.upload import ...` block around line 71 and extend it):

```python
from deriva_ml.dataset.upload import (
    ...,
    NULL_SENTINEL,
    ...,
)
```

(Add `NULL_SENTINEL` to the existing import list alphabetically.)

Also remove or update the stale comment on line 1071:

```python
            # Missing metadata values get "None" (matching legacy asset_file_path).
```

Change to:

```python
            # Missing metadata values get NULL_SENTINEL which
            # NullSentinelProcessor translates to SQL NULL at insert time.
            # Only nullable columns reach here — the validator guards
            # NOT-NULL columns upstream.
```

- [ ] **Step 3: Edit `execution/upload_engine.py` staging site (line 644)**

In `src/deriva_ml/execution/upload_engine.py`, locate `_invoke_deriva_py_uploader` (~line 562). Around line 644 find:

```python
            for col in metadata_cols:
                target_dir = target_dir / str(metadata.get(col, "None"))
```

Change to:

```python
            for col in metadata_cols:
                target_dir = target_dir / str(metadata.get(col, NULL_SENTINEL))
```

Add the import near the top of `upload_engine.py`. Look for an existing import from `deriva_ml.dataset.upload`. If present, extend it; otherwise add a new line:

```python
from deriva_ml.dataset.upload import NULL_SENTINEL
```

- [ ] **Step 4: Run a sanity-check import plus the full unit test tier**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && DERIVA_ML_ALLOW_DIRTY=true uv run python -c "from deriva_ml.dataset.upload import NULL_SENTINEL; assert NULL_SENTINEL == '__NULL__'; print('ok')"`

Expected: prints `ok`.

Then run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/asset/ tests/core/test_schema_cache.py tests/core/test_schema_diff.py -q`

Expected: all tests pass (no regressions; none of these tests exercise the staging paths directly but they share the import graph).

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && git add src/deriva_ml/dataset/upload.py src/deriva_ml/execution/execution.py src/deriva_ml/execution/upload_engine.py
git commit -m "fix: use NULL_SENTINEL for missing metadata at three staging sites (Bug C)"
```

---

## Task 6: Call validator from `Execution._upload_execution_dirs`

**Files:**
- Modify: `src/deriva_ml/execution/execution.py` — `_upload_execution_dirs` at line 1100

Add validator call as the first step before any staging work.

- [ ] **Step 1: Add the validator call**

In `src/deriva_ml/execution/execution.py`, locate `_upload_execution_dirs` (~line 1100). Immediately after the method signature and docstring (before any existing code), insert:

```python
        # Bug C: refuse to upload if any pending asset is missing a
        # required (NOT-NULL) metadata column. This raises a single
        # DerivaMLValidationError that lists all failures at once.
        from deriva_ml.asset.manifest import _validate_pending_asset_metadata
        _validate_pending_asset_metadata(self._model, self._get_manifest())
```

- [ ] **Step 2: Sanity check**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && DERIVA_ML_ALLOW_DIRTY=true uv run python -c "from deriva_ml.execution.execution import Execution; print('ok')"`

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && git add src/deriva_ml/execution/execution.py
git commit -m "feat(execution): validate metadata at _upload_execution_dirs top (Bug C)"
```

---

## Task 7: Call validator from `run_upload_engine`

**Files:**
- Modify: `src/deriva_ml/execution/upload_engine.py` — `run_upload_engine` at line 236

Add validator call at the top of the engine's drain loop, before any work-item enumeration. The engine reads pending rows from SQLite (not a manifest), so we construct an AssetManifest from the same source to feed the validator.

Wait — let me think about this. The engine doesn't have an `AssetManifest`; it has SQLite pending rows. We need to adapt.

Actually the simpler and better approach: iterate the engine's own pending rows (which already include schema, table, metadata_json) directly. Instead of reusing `_validate_pending_asset_metadata`, extract a lower-level helper that takes an iterable of `(key, schema, table, metadata_dict)` tuples.

Let me refactor the approach:

- [ ] **Step 1: Refactor the validator to take an iterable of entries**

In `src/deriva_ml/asset/manifest.py`, replace `_validate_pending_asset_metadata` with two functions — the original public-ish manifest-based one plus a lower-level iterable-based one it delegates to:

```python
def _validate_pending_asset_metadata_iter(
    model: "DerivaModel",
    entries: "Iterable[tuple[str, str, str, dict]]",
) -> None:
    """Lower-level validator accepting (key, schema, asset_table, metadata_dict)
    tuples. Used by both the manifest-based wrapper and the upload
    engine (which reads pending rows from SQLite rather than a manifest).

    See :func:`_validate_pending_asset_metadata` for semantics.
    """
    from deriva_ml.core.exceptions import DerivaMLValidationError

    missing_by_key: dict[str, list[str]] = {}

    for key, _schema, asset_table, metadata in sorted(entries):
        cols = model.asset_metadata_columns(asset_table)
        if not cols:
            continue
        provided = set(metadata.keys())
        missing: list[str] = []
        for col in cols:
            if not col.nullok and col.name not in provided:
                missing.append(col.name)
        if missing:
            missing_by_key[key] = sorted(missing)

    if not missing_by_key:
        return

    lines = [
        f"Missing required metadata for {len(missing_by_key)} pending asset(s):"
    ]
    for key in sorted(missing_by_key.keys()):
        cols = missing_by_key[key]
        noun = "column" if len(cols) == 1 else "columns"
        lines.append(f"  - {key}: missing {noun} {', '.join(cols)}")
    lines.append(
        "Supply these values before calling upload_outputs(), either via "
        "the ``metadata=`` arg to asset_file_path(...) or by assigning "
        "to the returned AssetFilePath's ``metadata`` property."
    )
    raise DerivaMLValidationError("\n".join(lines))


def _validate_pending_asset_metadata(
    model: "DerivaModel",
    manifest: "AssetManifest",
) -> None:
    """Raise DerivaMLValidationError if any pending manifest entry is
    missing a NOT-NULL metadata column value.

    Thin wrapper over :func:`_validate_pending_asset_metadata_iter`
    that projects ``AssetManifest.pending_assets()`` into the
    iterable shape the lower-level function expects.
    """
    entries = (
        (key, entry.schema, entry.asset_table, dict(entry.metadata))
        for key, entry in manifest.pending_assets().items()
    )
    _validate_pending_asset_metadata_iter(model, entries)
```

Also make sure `from typing import Iterable` is imported at the top of `manifest.py`.

- [ ] **Step 2: Write a test for the iterable-based validator**

Append to `tests/asset/test_metadata_validator.py`:

```python
def test_iter_validator_accepts_tuples():
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata_iter
    from deriva_ml.core.exceptions import DerivaMLValidationError
    model = _fake_model({
        "Image": [_col("Acquisition_Time", nullok=False)],
    })
    entries = [
        ("Image/x.png", "test-schema", "Image", {}),
    ]
    with pytest.raises(DerivaMLValidationError) as ei:
        _validate_pending_asset_metadata_iter(model, entries)
    assert "Image/x.png" in str(ei.value)
    assert "Acquisition_Time" in str(ei.value)


def test_iter_validator_happy_path():
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata_iter
    model = _fake_model({
        "Image": [_col("Acquisition_Time", nullok=False)],
    })
    entries = [
        ("Image/x.png", "test-schema", "Image", {"Acquisition_Time": "now"}),
    ]
    assert _validate_pending_asset_metadata_iter(model, entries) is None
```

- [ ] **Step 3: Run tests to verify both pass**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/asset/test_metadata_validator.py -v`

Expected: all 10 tests pass (8 existing + 2 new).

- [ ] **Step 4: Call the iter validator from `run_upload_engine`**

In `src/deriva_ml/execution/upload_engine.py`, locate `run_upload_engine` (~line 236). After the docstring and before the first work-item enumeration, insert:

```python
    # Bug C: refuse to drain if any pending row is missing a required
    # (NOT-NULL) asset-metadata column. Looks up pending rows (all
    # executions or the subset requested) and feeds them to the
    # iter validator.
    import json as _json
    from deriva_ml.asset.manifest import _validate_pending_asset_metadata_iter

    store = ml.workspace.execution_state_store()
    rids = execution_rids if execution_rids else None
    entries = []
    for row in store.list_pending_asset_rows(execution_rids=rids):
        # pending_asset_rows returns rows with asset_file_path set;
        # key is "exeRID/schema/table/key" for disambiguation.
        md = _json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        entries.append((
            f"{row['execution_rid']}/{row['target_table']}/{row['key']}",
            row["target_schema"],
            row["target_table"],
            md,
        ))
    if entries:
        _validate_pending_asset_metadata_iter(ml.model, entries)
```

- [ ] **Step 5: Check that `list_pending_asset_rows` exists; if not adapt**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && DERIVA_ML_ALLOW_DIRTY=true uv run python -c "from deriva_ml.execution.state_store import ExecutionStateStore; m = [m for m in dir(ExecutionStateStore) if 'pending' in m.lower()]; print(m)"`

Expected: prints a list of methods. If `list_pending_asset_rows` doesn't exist, find the closest equivalent (e.g., `list_pending_rows`) and add a filter for `asset_file_path IS NOT NULL`.

If the method doesn't exist with that exact name, substitute the closest one in Step 4 and add a filter:

```python
    entries = []
    for row in store.list_pending_rows(execution_rids=rids):
        if not row.get("asset_file_path"):
            continue  # Skip non-asset rows
        md = _json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        entries.append((
            f"{row['execution_rid']}/{row['target_table']}/{row['key']}",
            row["target_schema"],
            row["target_table"],
            md,
        ))
```

- [ ] **Step 6: Sanity check**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && DERIVA_ML_ALLOW_DIRTY=true uv run python -c "from deriva_ml.execution.upload_engine import run_upload_engine; print('ok')"`

Expected: prints `ok`.

- [ ] **Step 7: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && git add src/deriva_ml/asset/manifest.py src/deriva_ml/execution/upload_engine.py tests/asset/test_metadata_validator.py
git commit -m "feat(upload_engine): validate metadata at run_upload_engine top (Bug C)"
```

---

## Task 8: Integration tests (live-catalog, gated on `DERIVA_HOST`)

**Files:**
- Create: `tests/execution/test_bug_c_live_smoke.py`

Restore the Bug C live reproducer (from commit `df620c3`) as a passing test, plus two new tests for the nullable-path and the end-to-end happy path.

- [ ] **Step 1: Confirm test fixture has an asset table with NOT-NULL metadata**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run python -c "
from deriva_ml import DerivaML, ConnectionMode
import tempfile
with tempfile.TemporaryDirectory() as tmp:
    ml = DerivaML(hostname='localhost', catalog_id='1', working_dir=tmp)
    for t in ['Image', 'Execution_Metadata', 'Model']:
        try:
            cols = ml.model.asset_metadata_columns(t)
            print(f'{t}: {[(c.name, c.nullok) for c in cols]}')
        except Exception as e:
            print(f'{t}: {e}')
"`

Expected: prints each table's asset-metadata columns and their `nullok` values. Pick an asset table with at least one `nullok=False` (NOT-NULL) column; typical candidates are `Image` or a user-defined table.

If NO table has `nullok=False` metadata, add a fixture tweak: in `tests/conftest.py` or the nearest relevant fixture, add a test-only asset table with a NOT-NULL metadata column (this is a well-trodden pattern in the codebase). Name the column `Required_Attr` of type `text`, `nullok=False`.

- [ ] **Step 2: Write the integration tests**

Create `tests/execution/test_bug_c_live_smoke.py`:

```python
"""Live-catalog integration tests for Bug C (asset metadata None-stringification).

Gated on DERIVA_HOST. Three tests:

1. End-to-end happy path with Execution_Asset (zero metadata cols).
2. Upload refused when required metadata missing — validator raises.
3. Upload succeeds with SQL NULL when nullable metadata missing.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import pytest


requires_catalog = pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="Bug C live tests require DERIVA_HOST",
)


def _make_workflow(test_ml, name: str):
    from deriva_ml import MLVocab as vc
    test_ml.add_term(
        vc.workflow_type,
        "Test Workflow",
        description="for Bug C live tests",
    )
    return test_ml.create_workflow(
        name=name,
        workflow_type="Test Workflow",
        description="for Bug C live tests",
    )


@requires_catalog
def test_upload_asset_with_full_metadata_end_to_end(test_ml, tmp_path):
    """Zero-metadata table (Execution_Asset) — upload succeeds, validator is no-op."""
    from deriva_ml.execution.state_store import PendingRowStatus

    f = tmp_path / "smoke.bin"
    f.write_bytes(b"bug-c full-metadata smoke" * 32)

    wf = _make_workflow(test_ml, "Bug C happy path")
    exe = test_ml.create_execution(description="bug-c-happy", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)

    with exe.execute():
        pass

    store.insert_pending_row(
        execution_rid=exe.execution_rid,
        key="k1",
        target_schema="deriva-ml",
        target_table="Execution_Asset",
        metadata_json=json.dumps({}),
        created_at=now,
        rid="EA-BUGC-HAPPY-1",
        status=PendingRowStatus.leased,
        lease_token="happy-lease",
        asset_file_path=str(f),
    )

    report = exe.upload_outputs()
    assert report.total_failed == 0, f"failures: {report.errors}"
    assert report.total_uploaded == 1

    # Catalog has the row with real URL + MD5.
    expected_md5 = hashlib.md5(f.read_bytes()).hexdigest()
    pb = test_ml.pathBuilder()
    asset_path = pb.schemas["deriva-ml"].tables["Execution_Asset"]
    rows = list(
        asset_path.filter(asset_path.MD5 == expected_md5)
        .filter(asset_path.Filename == "smoke.bin")
        .entities().fetch()
    )
    assert len(rows) == 1
    assert "/hatrac/" in rows[0]["URL"]


@requires_catalog
def test_upload_with_missing_required_metadata_raises_validation(test_ml, tmp_path):
    """Bug C reproducer — passing now. Staging an asset whose table has
    a NOT-NULL metadata column, with no metadata supplied, must raise
    DerivaMLValidationError and NOT attempt the upload."""
    from deriva_ml.core.exceptions import DerivaMLValidationError
    from deriva_ml.execution.state_store import PendingRowStatus

    # Pick an asset table with at least one NOT-NULL metadata column.
    # Use "Image" if its Acquisition_Time / Acquisition_Date is NOT NULL,
    # otherwise the test fixture should have been extended in Step 1.
    candidate_table = "Image"
    cols = test_ml.model.asset_metadata_columns(candidate_table)
    required_cols = [c for c in cols if not c.nullok]
    if not required_cols:
        pytest.skip(
            f"Test catalog has no asset table with NOT-NULL metadata "
            f"(tried {candidate_table}); add fixture per plan Step 1."
        )

    f = tmp_path / "bug-c-required.bin"
    f.write_bytes(b"bug-c required-missing" * 32)

    wf = _make_workflow(test_ml, "Bug C required missing")
    exe = test_ml.create_execution(description="bug-c-required", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)

    with exe.execute():
        pass

    # Pending row with EMPTY metadata — missing required columns.
    store.insert_pending_row(
        execution_rid=exe.execution_rid,
        key="k2",
        target_schema="deriva-ml",  # adjust if Image lives elsewhere
        target_table=candidate_table,
        metadata_json=json.dumps({}),
        created_at=now,
        rid="IMG-BUGC-REQ-1",
        status=PendingRowStatus.leased,
        lease_token="req-lease",
        asset_file_path=str(f),
    )

    with pytest.raises(DerivaMLValidationError) as ei:
        exe.upload_outputs()
    msg = str(ei.value)
    for c in required_cols:
        assert c.name in msg, f"expected column {c.name} in error message"


@requires_catalog
def test_upload_with_missing_nullable_metadata_succeeds_with_null(test_ml, tmp_path):
    """The sentinel path. Staging an asset with a nullable metadata col
    absent must upload successfully and write SQL NULL to the catalog."""
    from deriva_ml.execution.state_store import PendingRowStatus

    # Find an asset table with at least one nullable metadata column
    # AND NO NOT-NULL columns (so we can exercise the sentinel without
    # tripping the validator). Fall back to skipping if none matches.
    candidate_table = None
    nullable_col_name = None
    for t in ("Image", "Model", "Execution_Metadata"):
        try:
            cols = test_ml.model.asset_metadata_columns(t)
        except Exception:
            continue
        required = [c for c in cols if not c.nullok]
        nullable = [c for c in cols if c.nullok]
        if nullable and not required:
            candidate_table = t
            nullable_col_name = nullable[0].name
            break
    if not candidate_table:
        pytest.skip("No asset table with all-nullable metadata found in test fixture")

    f = tmp_path / "bug-c-null.bin"
    f.write_bytes(b"bug-c nullable-missing" * 32)

    wf = _make_workflow(test_ml, "Bug C nullable missing")
    exe = test_ml.create_execution(description="bug-c-nullable", workflow=wf)
    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)

    with exe.execute():
        pass

    # Supply metadata for no columns — all are nullable.
    store.insert_pending_row(
        execution_rid=exe.execution_rid,
        key="k3",
        target_schema="deriva-ml",
        target_table=candidate_table,
        metadata_json=json.dumps({}),
        created_at=now,
        rid="NULL-BUGC-1",
        status=PendingRowStatus.leased,
        lease_token="null-lease",
        asset_file_path=str(f),
    )

    report = exe.upload_outputs()
    assert report.total_failed == 0, f"failures: {report.errors}"
    assert report.total_uploaded == 1

    # The catalog row's nullable column must be actual SQL NULL
    # (Python None after fetch), not the string "__NULL__" and not
    # "None".
    expected_md5 = hashlib.md5(f.read_bytes()).hexdigest()
    pb = test_ml.pathBuilder()
    asset_path = pb.schemas["deriva-ml"].tables[candidate_table]
    rows = list(
        asset_path.filter(asset_path.MD5 == expected_md5)
        .entities().fetch()
    )
    assert len(rows) == 1
    assert rows[0][nullable_col_name] is None, (
        f"expected SQL NULL for {nullable_col_name}, "
        f"got {rows[0][nullable_col_name]!r}"
    )
```

- [ ] **Step 3: Run the tests**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/test_bug_c_live_smoke.py -v`

Expected: all 3 tests pass (or skip cleanly with a helpful reason if the test-fixture catalog lacks the required asset shapes — in that case, tweak per §7.4 of the spec).

- [ ] **Step 4: Broader regression**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/asset/ tests/model/test_asset_metadata_columns.py tests/execution/test_bug_c_live_smoke.py tests/core/test_schema_cache.py tests/core/test_schema_diff.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && git add tests/execution/test_bug_c_live_smoke.py
git commit -m "test(bug_c): live-catalog integration tests (happy, required-missing, nullable-null)"
```

---

## Task 9: CHANGELOG

**Files:**
- Modify: `CHANGELOG.md` — prepend new Unreleased section

- [ ] **Step 1: Add CHANGELOG entry**

In `CHANGELOG.md`, locate the existing first Unreleased section (currently `## Unreleased — Schema pin + diff`). Insert immediately above it (after the title + preamble):

```markdown
## Unreleased — Bug C: asset metadata None-stringification

### Fixed

- **Missing asset metadata no longer inserts the literal string `"None"`.** The three staging path-builders (`Execution._build_upload_staging`, `_invoke_deriva_py_uploader`, `asset_file_path` helper in `dataset/upload.py`) previously substituted `"None"` for any metadata value absent from the manifest entry. That string flowed through deriva-py's regex + `column_map` expansion into catalog inserts, causing 400 errors on non-string columns (e.g., `invalid input syntax for type timestamp: "None"`). Replaced with a two-branch approach:
  - **NOT-NULL columns missing → `DerivaMLValidationError` at upload time.** New `_validate_pending_asset_metadata` runs at the top of `Execution._upload_execution_dirs` and `run_upload_engine`, listing every missing required column across every pending asset in a single aggregated message.
  - **Nullable columns missing → SQL `NULL` in the catalog.** Staging writes the sentinel `"__NULL__"` as the directory segment; the new `NullSentinelProcessor` (a deriva-py `BaseProcessor` subclass wired into `asset_table_upload_spec`) translates the sentinel to Python `None` before deriva-py's `interpolateDict` drops the key from the insert row.

### Added

- **`DerivaModel.asset_metadata_columns(table) -> list[Column]`** — companion to the existing `asset_metadata(table) -> set[str]`, returning `Column` objects so callers can inspect `nullok`.
- **`deriva_ml.asset.null_sentinel_processor.NullSentinelProcessor`** — `BaseProcessor` subclass for sentinel→None translation.
- **`NULL_SENTINEL = "__NULL__"`** — module-level constant in `deriva_ml.dataset.upload`.

### External-caller impact

No action required for callers who supply complete metadata. Callers who previously relied on the broken `"None"` substitution now get a `DerivaMLValidationError` (with an actionable message) at upload time instead of an opaque 400 from the catalog. Net outcome unchanged — the upload still can't succeed — but failure mode is clearer and cheaper (no network round-trip).

A known constraint: if a legitimate metadata value is the string `"__NULL__"`, it will be corrupted to SQL `NULL` by the pre-processor. Highly unlikely in practice; documented in `NullSentinelProcessor`'s docstring.
```

- [ ] **Step 2: Verify file parses**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && head -40 CHANGELOG.md`

Expected: new section appears between the preamble and the Schema Pin section.

- [ ] **Step 3: Commit**

```bash
cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && git add CHANGELOG.md
git commit -m "docs(changelog): Bug C asset metadata None-stringification fix"
```

---

## Final verification

- [ ] **Step 1: Ruff**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && uv run ruff check src/deriva_ml/ tests/ 2>&1 | grep -v "pre-existing" | head -20`

Expected: no new warnings on Bug C files (`manifest.py`, `null_sentinel_processor.py`, `upload.py`, `execution.py`, `upload_engine.py`, `catalog.py`, and the new tests).

- [ ] **Step 2: Unit + model subset (offline)**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/asset/test_metadata_validator.py tests/asset/test_null_sentinel_processor.py tests/model/test_asset_metadata_columns.py -v`

Expected: all tests pass (model tests need `DERIVA_HOST=localhost` — skip otherwise).

- [ ] **Step 3: Live-catalog subset**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/asset/ tests/model/test_asset_metadata_columns.py tests/execution/test_bug_c_live_smoke.py -v`

Expected: all pass.

- [ ] **Step 4: Broader regression**

Run: `cd /Users/carl/GitHub/deriva-ml/.worktrees/bug-c && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/ -q`

Expected: all tests pass (Bug C adds a validation check but with complete metadata this is a no-op for existing tests).

---

## Commit summary (expected)

After implementation, the branch will have these commits on top of main:

1. `docs(spec): Bug C asset metadata None-stringification design` (already committed, `0506163`)
2. `feat(model): add asset_metadata_columns() helper returning Column objects`
3. `feat(asset/manifest): _validate_pending_asset_metadata for NOT-NULL columns`
4. `feat(asset): NullSentinelProcessor for SQL NULL emission on nullable cols`
5. `feat(dataset/upload): NULL_SENTINEL constant + wire NullSentinelProcessor into asset_table_upload_spec`
6. `fix: use NULL_SENTINEL for missing metadata at three staging sites (Bug C)`
7. `feat(execution): validate metadata at _upload_execution_dirs top (Bug C)`
8. `feat(upload_engine): validate metadata at run_upload_engine top (Bug C)`
9. `test(bug_c): live-catalog integration tests (happy, required-missing, nullable-null)`
10. `docs(changelog): Bug C asset metadata None-stringification fix`

Nine implementation commits on top of the spec, plus the earlier revert of the scope-correction commit (landed before this plan).
