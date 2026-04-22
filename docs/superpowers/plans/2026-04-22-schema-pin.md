# Schema Pin + Diff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add user-facing pin/unpin/diff controls on top of the existing `SchemaCache` so users can freeze the cache at a snapshot, inspect structural drift between cached and live catalog schemas, and safely block `refresh_schema` while pinned.

**Architecture:** Three units. Extend `SchemaCache` with pin methods and a `PinStatus` Pydantic model, add a new `schema_diff.py` module with a `SchemaDiff` Pydantic model + walker, and add four thin `DerivaML` public methods plus a new pin-guard branch in `refresh_schema`.

**Tech Stack:** Python ≥3.12, Pydantic v2 (BaseModel, ConfigDict(frozen=True)), `deriva-py` ErmrestCatalog (for live `/schema` + `/` snaptime fetches — online only), pytest. Built on S4 (`SchemaCache`, `ConnectionMode`, `DerivaMLReadOnlyError`, `DerivaMLSchemaRefreshBlocked`).

**Reference spec:** `docs/superpowers/specs/2026-04-22-schema-pin-design.md`

---

## File Structure

- Create: `src/deriva_ml/core/schema_diff.py` — `SchemaDiff` Pydantic model, fine-grained record classes, `compute_diff(cached, live)` function.
- Modify: `src/deriva_ml/core/schema_cache.py` — add `PinStatus`, add `pin/unpin/pin_status` methods, extract `_write_atomic` helper from `write()` so the three write paths share identical on-disk discipline.
- Modify: `src/deriva_ml/core/exceptions.py` — add `DerivaMLSchemaPinned`.
- Modify: `src/deriva_ml/core/base.py` — add `pin_schema`, `unpin_schema`, `pin_status`, `diff_schema` public methods; add pin-guard to `refresh_schema`.
- Create: `tests/core/test_schema_diff.py` — unit tests for diff walker + render.
- Create: `tests/core/test_schema_pin.py` — integration tests (gated on `DERIVA_HOST`).
- Modify: `tests/core/test_schema_cache.py` — add 9 unit tests for pin methods.
- Modify: `tests/core/test_exceptions.py` — add `DerivaMLSchemaPinned` inheritance test.
- Modify: `CHANGELOG.md` — add `## Unreleased — Schema pin + diff` section.

---

## Task 1: Extract `_write_atomic` helper in `SchemaCache`

**Files:**
- Modify: `src/deriva_ml/core/schema_cache.py:70-98`
- Test: `tests/core/test_schema_cache.py` (existing `test_write_is_atomic_no_partial_file_on_crash` still passes — no behavior change)

Pure refactor: extract the tmp + fsync + rename dance into a private helper so the upcoming `pin` and `unpin` can reuse it without copy-pasting the atomic-write logic.

- [ ] **Step 1: Add a failing-by-inspection test that locks in the extracted helper signature**

Append to `tests/core/test_schema_cache.py`:

```python
def test_write_atomic_helper_exists_and_writes_payload(tmp_path):
    """_write_atomic is the private helper pin/unpin will reuse."""
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    payload = {
        "snapshot_id": "s1",
        "hostname": "h",
        "catalog_id": "c",
        "ml_schema": "ml",
        "schema": {"k": "v"},
    }
    cache._write_atomic(payload)
    assert cache.exists()
    import json
    assert json.loads(cache._path.read_text()) == payload
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_schema_cache.py::test_write_atomic_helper_exists_and_writes_payload -v`

Expected: FAIL with `AttributeError: 'SchemaCache' object has no attribute '_write_atomic'`.

- [ ] **Step 3: Extract the helper**

In `src/deriva_ml/core/schema_cache.py`, replace the `write` method (lines 70-98) with:

```python
    def write(
        self,
        *,
        snapshot_id: str,
        hostname: str,
        catalog_id: str,
        ml_schema: str,
        schema: dict,
    ) -> None:
        """Atomically overwrite the cache.

        The new contents are written to a sibling ``.tmp`` file,
        ``fsync``'d, then moved over the original via ``os.replace``.
        If any step fails, the original file is unchanged.
        """
        payload = {
            "snapshot_id": snapshot_id,
            "hostname": hostname,
            "catalog_id": catalog_id,
            "ml_schema": ml_schema,
            "schema": schema,
        }
        self._write_atomic(payload)

    def _write_atomic(self, payload: dict) -> None:
        """Atomically write ``payload`` as JSON to the cache file.

        Writes to a sibling ``.tmp``, ``fsync``'s, then ``os.replace``'s
        over the target. On failure the original file is unchanged.
        Used by both ``write()`` (full cache refresh) and ``pin()``/
        ``unpin()`` (which rewrite an existing cache's payload with
        a tweaked ``"pin"`` key).
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w") as fp:
            json.dump(payload, fp, indent=2)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp, self._path)
```

- [ ] **Step 4: Run the new test and the existing atomic-write test to verify**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_schema_cache.py -v`

Expected: all 6 tests pass (5 old + 1 new).

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/core/schema_cache.py tests/core/test_schema_cache.py
git commit -m "refactor(schema_cache): extract _write_atomic helper for reuse by pin/unpin"
```

---

## Task 2: Add `PinStatus` + `pin`/`unpin`/`pin_status` methods to `SchemaCache`

**Files:**
- Modify: `src/deriva_ml/core/schema_cache.py` (add imports at top, add `PinStatus` class, add three methods)
- Test: `tests/core/test_schema_cache.py` (add 8 tests)

The cache gains a nested optional `"pin"` key in its JSON payload; presence means pinned. `pin_status()` projects that onto a frozen Pydantic snapshot with `pinned_at` (UTC datetime) and `pin_reason` fields.

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_schema_cache.py`:

```python
def _populate(cache, schema_payload=None):
    """Helper: write a minimal valid cache for pin tests."""
    cache.write(
        snapshot_id="s0",
        hostname="h",
        catalog_id="c",
        ml_schema="ml",
        schema=schema_payload or {"schemas": {}},
    )


def test_pin_on_unpinned_cache_sets_fields(tmp_path):
    from datetime import datetime, timezone
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    _populate(cache)
    before = datetime.now(timezone.utc)
    cache.pin(reason="paper repro")
    after = datetime.now(timezone.utc)
    status = cache.pin_status()
    assert status.pinned is True
    assert status.pin_reason == "paper repro"
    assert status.pinned_snapshot_id == "s0"
    assert status.pinned_at is not None
    assert before <= status.pinned_at <= after


def test_pin_without_reason(tmp_path):
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    _populate(cache)
    cache.pin()
    status = cache.pin_status()
    assert status.pinned is True
    assert status.pin_reason is None


def test_pin_idempotent_updates_metadata(tmp_path):
    import time
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    _populate(cache)
    cache.pin(reason="first")
    first = cache.pin_status()
    time.sleep(0.01)
    cache.pin(reason="second")
    second = cache.pin_status()
    assert second.pinned is True
    assert second.pin_reason == "second"
    assert second.pinned_at >= first.pinned_at


def test_unpin_clears_fields(tmp_path):
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    _populate(cache)
    cache.pin(reason="r")
    cache.unpin()
    status = cache.pin_status()
    assert status.pinned is False
    assert status.pinned_at is None
    assert status.pin_reason is None
    assert status.pinned_snapshot_id == "s0"


def test_unpin_on_unpinned_is_no_op(tmp_path):
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    _populate(cache)
    cache.unpin()  # should not raise
    status = cache.pin_status()
    assert status.pinned is False


def test_pin_status_on_missing_cache_raises(tmp_path):
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    import pytest
    with pytest.raises(FileNotFoundError):
        cache.pin_status()


def test_pin_persists_across_instances(tmp_path):
    from deriva_ml.core.schema_cache import SchemaCache
    a = SchemaCache(tmp_path)
    _populate(a)
    a.pin(reason="persist me")
    b = SchemaCache(tmp_path)
    status = b.pin_status()
    assert status.pinned is True
    assert status.pin_reason == "persist me"


def test_cache_file_format_has_nested_pin_object(tmp_path):
    """After pin, the JSON has a top-level ``pin`` object; unpin removes it."""
    import json
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    _populate(cache)
    cache.pin(reason="x")
    raw = json.loads(cache._path.read_text())
    assert "pin" in raw
    assert raw["pin"]["reason"] == "x"
    assert "at" in raw["pin"]
    cache.unpin()
    raw2 = json.loads(cache._path.read_text())
    assert "pin" not in raw2
```

- [ ] **Step 2: Run tests to verify they all fail**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_schema_cache.py -v`

Expected: the 8 new tests FAIL with `AttributeError: 'SchemaCache' object has no attribute 'pin'` (or similar). The old tests still pass.

- [ ] **Step 3: Implement `PinStatus`, `pin`, `unpin`, and `pin_status`**

In `src/deriva_ml/core/schema_cache.py`, replace the top of the file (imports) with:

```python
"""Workspace-backed cache of the catalog schema.

Offline mode reads from this cache; online mode detects drift by
comparing the live catalog's snapshot id to the cached one and
warns the user without auto-refreshing.

File layout on disk at ``<workspace>/schema-cache.json``::

    {
        "snapshot_id": "<ERMrest snapshot id (snaptime)>",
        "hostname": "example.org",
        "catalog_id": "42",
        "ml_schema": "deriva-ml",
        "schema": { ... full ermrest /schema payload ... },
        "pin": {                              # optional; presence = pinned
            "at": "2026-04-22T20:30:00Z",
            "reason": "reproducing 2025 paper analysis"
        }
    }

Writes are atomic: the new contents go to ``schema-cache.json.tmp``,
get ``fsync``'d, then ``os.replace`` moves the tmp over the
original. If anything crashes mid-write, the old file remains
intact.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from deriva_ml.core.exceptions import DerivaMLConfigurationError


class PinStatus(BaseModel):
    """Current pin state of a :class:`SchemaCache`. Frozen Pydantic snapshot.

    Attributes:
        pinned: True iff the cache's JSON payload has a ``"pin"`` key.
        pinned_at: UTC timestamp of the most recent ``pin()`` call,
            or ``None`` when unpinned.
        pin_reason: Caller-supplied reason, or ``None`` when unpinned
            or when ``pin()`` was called without a reason.
        pinned_snapshot_id: The cache's current ``snapshot_id``
            (always present, whether pinned or not). A pinned cache
            is guaranteed to stay at this snapshot until ``unpin()``.
    """

    model_config = ConfigDict(frozen=True)

    pinned: bool
    pinned_at: datetime | None
    pin_reason: str | None
    pinned_snapshot_id: str
```

Then replace the `class SchemaCache` body by appending three new methods after `write`:

```python
    def pin(self, reason: str | None = None) -> None:
        """Mark the cache pinned at its current snapshot.

        Idempotent: pinning an already-pinned cache updates ``pinned_at``
        and ``reason`` to reflect the most recent call. The on-disk
        write goes through :meth:`_write_atomic` so a crash mid-pin
        leaves the prior cache state intact.

        Args:
            reason: Free-text explanation stored alongside the pin.
                Optional; defaults to ``None`` (stored as JSON null).

        Raises:
            FileNotFoundError: If the cache file doesn't exist. Call
                an online ``DerivaML.__init__`` or ``refresh_schema()``
                first to populate the cache.
        """
        payload = self.load()
        payload["pin"] = {
            "at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "reason": reason,
        }
        self._write_atomic(payload)

    def unpin(self) -> None:
        """Clear pin state. No-op if already unpinned.

        Atomic write when a pin existed; no I/O otherwise. Does not
        alter the cache's schema payload or snapshot_id.

        Raises:
            FileNotFoundError: If the cache file doesn't exist.
        """
        payload = self.load()
        if "pin" not in payload:
            return
        del payload["pin"]
        self._write_atomic(payload)

    def pin_status(self) -> PinStatus:
        """Return current pin state as a frozen :class:`PinStatus`.

        Raises:
            FileNotFoundError: If the cache file doesn't exist.
        """
        payload = self.load()
        pin = payload.get("pin")
        if pin is None:
            return PinStatus(
                pinned=False,
                pinned_at=None,
                pin_reason=None,
                pinned_snapshot_id=payload["snapshot_id"],
            )
        # Parse the ISO string Pydantic-style; handle the trailing "Z".
        raw_at = pin["at"]
        if raw_at.endswith("Z"):
            raw_at = raw_at[:-1] + "+00:00"
        return PinStatus(
            pinned=True,
            pinned_at=datetime.fromisoformat(raw_at),
            pin_reason=pin.get("reason"),
            pinned_snapshot_id=payload["snapshot_id"],
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_schema_cache.py -v`

Expected: all 14 tests pass (5 original + 1 from Task 1 + 8 new).

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/core/schema_cache.py tests/core/test_schema_cache.py
git commit -m "feat(schema_cache): add pin/unpin/pin_status + PinStatus Pydantic model"
```

---

## Task 3: Create `schema_diff.py` with `SchemaDiff` model + records

**Files:**
- Create: `src/deriva_ml/core/schema_diff.py`
- Create: `tests/core/test_schema_diff.py`

Self-contained module. Seven fine-grained record classes + the parent `SchemaDiff` + `compute_diff(cached, live)`. Deterministic (sorted) walk over the ERMrest `/schema` payload shape.

ERMrest `/schema` shape (the keys we read):

```
{
  "schemas": {
    "<schema_name>": {
      "schema_name": "...",
      "tables": {
        "<table_name>": {
          "schema_name": "...",
          "table_name": "...",
          "column_definitions": [
            {"name": "col", "type": {"typename": "text", ...}, ...},
            ...
          ],
          "foreign_keys": [
            {
              "foreign_key_columns": [{"schema_name": "s", "table_name": "t", "column_name": "c"}, ...],
              "referenced_columns":  [{"schema_name": "s", "table_name": "t", "column_name": "c"}, ...]
            },
            ...
          ]
        }
      }
    }
  }
}
```

We key FKs by `(sorted fk columns, referenced schema, referenced table, sorted referenced columns)` for comparison.

- [ ] **Step 1: Write the failing tests**

Create `tests/core/test_schema_diff.py`:

```python
"""Unit tests for compute_diff + SchemaDiff rendering."""
from __future__ import annotations

import pytest


def _schema(tables: dict | None = None, schema_name: str = "deriva-ml") -> dict:
    """Minimal ERMrest /schema payload for tests."""
    return {
        "schemas": {
            schema_name: {
                "schema_name": schema_name,
                "tables": tables or {},
            }
        }
    }


def _table(columns=None, fkeys=None, name="T"):
    return {
        "schema_name": "deriva-ml",
        "table_name": name,
        "column_definitions": columns or [],
        "foreign_keys": fkeys or [],
    }


def _col(name, typename="text"):
    return {"name": name, "type": {"typename": typename}}


def _fkey(columns, ref_schema, ref_table, ref_columns):
    return {
        "foreign_key_columns": [
            {"schema_name": "deriva-ml", "table_name": "X", "column_name": c}
            for c in columns
        ],
        "referenced_columns": [
            {"schema_name": ref_schema, "table_name": ref_table, "column_name": c}
            for c in ref_columns
        ],
    }


def test_empty_diff_when_schemas_identical():
    from deriva_ml.core.schema_diff import compute_diff
    s = _schema({"T": _table(columns=[_col("a")])})
    diff = compute_diff(s, s)
    assert diff.is_empty()
    assert diff.added_schemas == []
    assert diff.removed_schemas == []


def test_added_schema():
    from deriva_ml.core.schema_diff import compute_diff
    cached = _schema()
    live = {
        "schemas": {
            "deriva-ml": {"schema_name": "deriva-ml", "tables": {}},
            "newsch":    {"schema_name": "newsch",    "tables": {}},
        }
    }
    diff = compute_diff(cached, live)
    assert diff.added_schemas == ["newsch"]
    assert diff.removed_schemas == []
    assert not diff.is_empty()


def test_removed_schema():
    from deriva_ml.core.schema_diff import compute_diff
    cached = {
        "schemas": {
            "deriva-ml": {"schema_name": "deriva-ml", "tables": {}},
            "gone":      {"schema_name": "gone",      "tables": {}},
        }
    }
    live = _schema()
    diff = compute_diff(cached, live)
    assert diff.removed_schemas == ["gone"]


def test_added_table():
    from deriva_ml.core.schema_diff import compute_diff
    cached = _schema({"T1": _table(name="T1")})
    live = _schema({
        "T1": _table(name="T1"),
        "T2": _table(name="T2"),
    })
    diff = compute_diff(cached, live)
    assert [t.table for t in diff.added_tables] == ["T2"]
    assert all(t.schema == "deriva-ml" for t in diff.added_tables)


def test_removed_table():
    from deriva_ml.core.schema_diff import compute_diff
    cached = _schema({
        "T1": _table(name="T1"),
        "T2": _table(name="T2"),
    })
    live = _schema({"T1": _table(name="T1")})
    diff = compute_diff(cached, live)
    assert [t.table for t in diff.removed_tables] == ["T2"]


def test_added_column():
    from deriva_ml.core.schema_diff import compute_diff
    cached = _schema({"T": _table(columns=[_col("a")])})
    live = _schema({"T": _table(columns=[_col("a"), _col("b", "int4")])})
    diff = compute_diff(cached, live)
    assert len(diff.added_columns) == 1
    add = diff.added_columns[0]
    assert add.schema == "deriva-ml"
    assert add.table == "T"
    assert add.column == "b"
    assert add.type == "int4"


def test_removed_column():
    from deriva_ml.core.schema_diff import compute_diff
    cached = _schema({"T": _table(columns=[_col("a"), _col("b")])})
    live = _schema({"T": _table(columns=[_col("a")])})
    diff = compute_diff(cached, live)
    assert [c.column for c in diff.removed_columns] == ["b"]


def test_column_type_change():
    from deriva_ml.core.schema_diff import compute_diff
    cached = _schema({"T": _table(columns=[_col("a", "text")])})
    live = _schema({"T": _table(columns=[_col("a", "int4")])})
    diff = compute_diff(cached, live)
    assert len(diff.column_type_changes) == 1
    chg = diff.column_type_changes[0]
    assert chg.column == "a"
    assert chg.cached_type == "text"
    assert chg.live_type == "int4"


def test_added_fkey():
    from deriva_ml.core.schema_diff import compute_diff
    cached = _schema({"T": _table(columns=[_col("x")])})
    live = _schema({
        "T": _table(
            columns=[_col("x")],
            fkeys=[_fkey(["x"], "deriva-ml", "Other", ["y"])],
        ),
    })
    diff = compute_diff(cached, live)
    assert len(diff.added_fkeys) == 1
    fk = diff.added_fkeys[0]
    assert fk.columns == ["x"]
    assert fk.referenced_table == "Other"
    assert fk.referenced_columns == ["y"]


def test_removed_fkey():
    from deriva_ml.core.schema_diff import compute_diff
    cached = _schema({
        "T": _table(
            columns=[_col("x")],
            fkeys=[_fkey(["x"], "deriva-ml", "Other", ["y"])],
        ),
    })
    live = _schema({"T": _table(columns=[_col("x")])})
    diff = compute_diff(cached, live)
    assert len(diff.removed_fkeys) == 1


def test_diff_render_produces_human_readable():
    from deriva_ml.core.schema_diff import compute_diff
    cached = _schema({"T": _table(columns=[_col("a")])})
    live = _schema({"T": _table(columns=[_col("a"), _col("b", "int4")])})
    diff = compute_diff(cached, live)
    text = diff.render()
    assert "deriva-ml.T.b" in text
    assert "int4" in text
    assert text == str(diff)
    # Empty diff renders empty-ish, no crash
    empty = compute_diff(cached, cached)
    assert empty.render() == "" or "no changes" in empty.render().lower()


def test_diff_determinism():
    """Two runs over the same inputs produce identical diffs (sorted)."""
    from deriva_ml.core.schema_diff import compute_diff
    cached = _schema({"T": _table(columns=[_col("a")])})
    live = _schema({
        "T": _table(columns=[_col("a"), _col("z"), _col("m"), _col("b")]),
    })
    d1 = compute_diff(cached, live)
    d2 = compute_diff(cached, live)
    assert d1 == d2
    assert [c.column for c in d1.added_columns] == ["b", "m", "z"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_schema_diff.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'deriva_ml.core.schema_diff'`.

- [ ] **Step 3: Implement `schema_diff.py`**

Create `src/deriva_ml/core/schema_diff.py`:

```python
"""Compute structural diffs between two ERMrest /schema payloads.

This module is cache-agnostic. It takes two plain dicts (cached,
live), walks them deterministically, and emits a frozen
:class:`SchemaDiff` Pydantic model.

V1 dimensions: schemas (add/remove), tables (add/remove), columns
(add/remove + type change), foreign keys (add/remove). Out of scope
for V1: non-FK keys, annotations, ACLs, column nullability and
defaults, comments.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AddedTable(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema: str
    table: str


class RemovedTable(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema: str
    table: str


class AddedColumn(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema: str
    table: str
    column: str
    type: str


class RemovedColumn(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema: str
    table: str
    column: str


class ColumnTypeChange(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema: str
    table: str
    column: str
    cached_type: str
    live_type: str


class AddedForeignKey(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema: str
    table: str
    columns: list[str]
    referenced_schema: str
    referenced_table: str
    referenced_columns: list[str]


class RemovedForeignKey(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema: str
    table: str
    columns: list[str]
    referenced_schema: str
    referenced_table: str
    referenced_columns: list[str]


class SchemaDiff(BaseModel):
    """Structured diff between a cached schema and a live schema.

    All list fields are deterministically ordered. An empty
    :class:`SchemaDiff` means the two payloads are structurally
    equivalent under the V1 dimensions.
    """

    model_config = ConfigDict(frozen=True)

    added_schemas: list[str]
    removed_schemas: list[str]
    added_tables: list[AddedTable]
    removed_tables: list[RemovedTable]
    added_columns: list[AddedColumn]
    removed_columns: list[RemovedColumn]
    column_type_changes: list[ColumnTypeChange]
    added_fkeys: list[AddedForeignKey]
    removed_fkeys: list[RemovedForeignKey]

    def is_empty(self) -> bool:
        """True iff no differences were found."""
        return not (
            self.added_schemas
            or self.removed_schemas
            or self.added_tables
            or self.removed_tables
            or self.added_columns
            or self.removed_columns
            or self.column_type_changes
            or self.added_fkeys
            or self.removed_fkeys
        )

    def render(self) -> str:
        """Return a human-readable multi-line summary.

        Empty diff returns the empty string. Useful for log messages
        and paste-into-ticket output.
        """
        if self.is_empty():
            return ""
        lines: list[str] = []
        for s in self.added_schemas:
            lines.append(f"+ schema {s}")
        for s in self.removed_schemas:
            lines.append(f"- schema {s}")
        for t in self.added_tables:
            lines.append(f"+ table {t.schema}.{t.table}")
        for t in self.removed_tables:
            lines.append(f"- table {t.schema}.{t.table}")
        for c in self.added_columns:
            lines.append(f"+ column {c.schema}.{c.table}.{c.column} ({c.type})")
        for c in self.removed_columns:
            lines.append(f"- column {c.schema}.{c.table}.{c.column}")
        for c in self.column_type_changes:
            lines.append(
                f"~ column {c.schema}.{c.table}.{c.column}: "
                f"{c.cached_type} → {c.live_type}"
            )
        for fk in self.added_fkeys:
            lines.append(
                f"+ fkey {fk.schema}.{fk.table}({','.join(fk.columns)}) "
                f"→ {fk.referenced_schema}.{fk.referenced_table}"
                f"({','.join(fk.referenced_columns)})"
            )
        for fk in self.removed_fkeys:
            lines.append(
                f"- fkey {fk.schema}.{fk.table}({','.join(fk.columns)}) "
                f"→ {fk.referenced_schema}.{fk.referenced_table}"
                f"({','.join(fk.referenced_columns)})"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


# --- walker ---------------------------------------------------------------


def _schemas(payload: dict) -> dict:
    return payload.get("schemas", {})


def _tables(schema_payload: dict) -> dict:
    return schema_payload.get("tables", {})


def _col_map(table_payload: dict) -> dict[str, str]:
    """Return {column_name: typename} for a table payload."""
    out: dict[str, str] = {}
    for c in table_payload.get("column_definitions", []):
        out[c["name"]] = c.get("type", {}).get("typename", "")
    return out


def _fkey_key(fk: dict) -> tuple:
    """Stable comparison key for a foreign-key definition.

    (fk columns sorted, ref schema, ref table, ref columns sorted).
    """
    fk_cols = sorted(c["column_name"] for c in fk.get("foreign_key_columns", []))
    ref_cols_raw = fk.get("referenced_columns", [])
    ref_cols = sorted(c["column_name"] for c in ref_cols_raw)
    ref_schema = ref_cols_raw[0]["schema_name"] if ref_cols_raw else ""
    ref_table = ref_cols_raw[0]["table_name"] if ref_cols_raw else ""
    return (tuple(fk_cols), ref_schema, ref_table, tuple(ref_cols))


def _fkey_detail(fk: dict) -> tuple[list[str], str, str, list[str]]:
    fk_cols = sorted(c["column_name"] for c in fk.get("foreign_key_columns", []))
    ref_cols_raw = fk.get("referenced_columns", [])
    ref_cols = sorted(c["column_name"] for c in ref_cols_raw)
    ref_schema = ref_cols_raw[0]["schema_name"] if ref_cols_raw else ""
    ref_table = ref_cols_raw[0]["table_name"] if ref_cols_raw else ""
    return fk_cols, ref_schema, ref_table, ref_cols


def compute_diff(cached: dict, live: dict) -> SchemaDiff:
    """Compare two ERMrest ``/schema`` payloads.

    Args:
        cached: Payload stored in the local schema cache.
        live: Payload fetched from the live catalog.

    Returns:
        A :class:`SchemaDiff`. Empty iff the two payloads are
        structurally equivalent under V1 dimensions.
    """
    cached_schemas = _schemas(cached)
    live_schemas = _schemas(live)

    cached_names = set(cached_schemas)
    live_names = set(live_schemas)
    added_schemas = sorted(live_names - cached_names)
    removed_schemas = sorted(cached_names - live_names)

    added_tables: list[AddedTable] = []
    removed_tables: list[RemovedTable] = []
    added_columns: list[AddedColumn] = []
    removed_columns: list[RemovedColumn] = []
    column_type_changes: list[ColumnTypeChange] = []
    added_fkeys: list[AddedForeignKey] = []
    removed_fkeys: list[RemovedForeignKey] = []

    for schema_name in sorted(cached_names & live_names):
        cached_tables = _tables(cached_schemas[schema_name])
        live_tables = _tables(live_schemas[schema_name])

        for t_name in sorted(set(live_tables) - set(cached_tables)):
            added_tables.append(AddedTable(schema=schema_name, table=t_name))
        for t_name in sorted(set(cached_tables) - set(live_tables)):
            removed_tables.append(RemovedTable(schema=schema_name, table=t_name))

        for t_name in sorted(set(cached_tables) & set(live_tables)):
            cached_cols = _col_map(cached_tables[t_name])
            live_cols = _col_map(live_tables[t_name])
            for col in sorted(set(live_cols) - set(cached_cols)):
                added_columns.append(
                    AddedColumn(
                        schema=schema_name, table=t_name,
                        column=col, type=live_cols[col],
                    )
                )
            for col in sorted(set(cached_cols) - set(live_cols)):
                removed_columns.append(
                    RemovedColumn(schema=schema_name, table=t_name, column=col)
                )
            for col in sorted(set(cached_cols) & set(live_cols)):
                if cached_cols[col] != live_cols[col]:
                    column_type_changes.append(
                        ColumnTypeChange(
                            schema=schema_name, table=t_name, column=col,
                            cached_type=cached_cols[col],
                            live_type=live_cols[col],
                        )
                    )

            # Foreign keys
            cached_fks = cached_tables[t_name].get("foreign_keys", []) or []
            live_fks = live_tables[t_name].get("foreign_keys", []) or []
            cached_keyed = {_fkey_key(fk): fk for fk in cached_fks}
            live_keyed = {_fkey_key(fk): fk for fk in live_fks}
            for k in sorted(set(live_keyed) - set(cached_keyed)):
                cols, rs, rt, rcs = _fkey_detail(live_keyed[k])
                added_fkeys.append(AddedForeignKey(
                    schema=schema_name, table=t_name,
                    columns=cols, referenced_schema=rs,
                    referenced_table=rt, referenced_columns=rcs,
                ))
            for k in sorted(set(cached_keyed) - set(live_keyed)):
                cols, rs, rt, rcs = _fkey_detail(cached_keyed[k])
                removed_fkeys.append(RemovedForeignKey(
                    schema=schema_name, table=t_name,
                    columns=cols, referenced_schema=rs,
                    referenced_table=rt, referenced_columns=rcs,
                ))

    return SchemaDiff(
        added_schemas=added_schemas,
        removed_schemas=removed_schemas,
        added_tables=added_tables,
        removed_tables=removed_tables,
        added_columns=added_columns,
        removed_columns=removed_columns,
        column_type_changes=column_type_changes,
        added_fkeys=added_fkeys,
        removed_fkeys=removed_fkeys,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_schema_diff.py -v`

Expected: all 12 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/core/schema_diff.py tests/core/test_schema_diff.py
git commit -m "feat(schema_diff): add SchemaDiff Pydantic model + compute_diff walker"
```

---

## Task 4: Add `DerivaMLSchemaPinned` exception

**Files:**
- Modify: `src/deriva_ml/core/exceptions.py` (append new class after `DerivaMLSchemaRefreshBlocked`)
- Test: `tests/core/test_exceptions.py` (add 1 test)

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_exceptions.py`:

```python
def test_derivaml_schema_pinned_inherits_configuration_error():
    from deriva_ml.core.exceptions import (
        DerivaMLConfigurationError,
        DerivaMLSchemaPinned,
    )
    err = DerivaMLSchemaPinned("refresh_schema refused: cache is pinned")
    assert isinstance(err, DerivaMLConfigurationError)
    assert "pinned" in str(err)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_exceptions.py::test_derivaml_schema_pinned_inherits_configuration_error -v`

Expected: FAIL with `ImportError: cannot import name 'DerivaMLSchemaPinned'`.

- [ ] **Step 3: Add the exception**

In `src/deriva_ml/core/exceptions.py`, immediately after the existing `class DerivaMLSchemaRefreshBlocked` (around line 121), insert:

```python
class DerivaMLSchemaPinned(DerivaMLConfigurationError):
    """Raised when ``refresh_schema()`` is called on a pinned cache.

    The cache has been explicitly pinned via ``pin_schema()``. Call
    ``unpin_schema()`` first if you really want to refresh. Note:
    ``force=True`` does NOT bypass a pin — it only bypasses the
    pending-rows guard.

    Example:
        >>> raise DerivaMLSchemaPinned(
        ...     "refresh_schema refused: cache is pinned at snapshot s0"
        ... )
    """

    pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_exceptions.py -v`

Expected: all tests in the file pass, including the new one.

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/core/exceptions.py tests/core/test_exceptions.py
git commit -m "feat(exceptions): add DerivaMLSchemaPinned for pinned-cache refresh guard"
```

---

## Task 5: Add `pin_schema`/`unpin_schema`/`pin_status`/`diff_schema` to `DerivaML`

**Files:**
- Modify: `src/deriva_ml/core/base.py` (imports + add 4 new public methods after `refresh_schema`)

Thin wrappers that delegate to `SchemaCache` and `compute_diff`. `pin_schema` is the one that does real work: online it fetches live snapshot + schema, returns a `SchemaDiff` when the cache is behind live content, and logs a warning in that case.

- [ ] **Step 1: Update imports at the top of `base.py`**

In `src/deriva_ml/core/base.py`, modify the exceptions import block (lines 51-56) to add `DerivaMLSchemaPinned`:

```python
from deriva_ml.core.exceptions import (
    DerivaMLConfigurationError,
    DerivaMLException,
    DerivaMLReadOnlyError,
    DerivaMLSchemaPinned,
    DerivaMLSchemaRefreshBlocked,
)
```

And modify the `SchemaCache` import (line 70) so that both `SchemaCache` and `PinStatus` are imported:

```python
from deriva_ml.core.schema_cache import PinStatus, SchemaCache
```

- [ ] **Step 2: Add the four new public methods after `refresh_schema`**

In `src/deriva_ml/core/base.py`, immediately after the current `refresh_schema` method (currently ending around line 564, right before `_get_session_config`), insert:

```python
    def pin_schema(self, reason: str | None = None) -> "SchemaDiff | None":
        """Freeze the local schema cache at its current snapshot.

        While pinned, :meth:`refresh_schema` refuses to update the
        cache (even with ``force=True``). Call :meth:`unpin_schema`
        to clear the pin.

        Online mode additionally checks for structural drift: if the
        live catalog has moved on and its ``/schema`` payload differs
        from the cached one (columns, tables, foreign keys, etc.),
        a :class:`SchemaDiff` describing the drift is returned, and
        a WARNING is logged. The pin is still persisted.

        Offline mode always returns ``None`` — the cache is pinned,
        but no live comparison is possible.

        Args:
            reason: Free-text explanation stored alongside the pin.
                Useful for reporting (``pin_status().pin_reason``).

        Returns:
            A :class:`SchemaDiff` when the pin is applied online and
            the live catalog's schema differs structurally from the
            cache. ``None`` otherwise (offline, no drift, or snapshot
            bumped without schema change).

        Raises:
            FileNotFoundError: If the workspace has no cache yet.
                Run an online ``DerivaML.__init__`` or
                :meth:`refresh_schema` first.
        """
        from deriva_ml.core.schema_diff import compute_diff, SchemaDiff

        cache = SchemaCache(self.working_dir)
        drift: SchemaDiff | None = None
        if self._mode is ConnectionMode.online:
            live_snapshot_id = self.catalog.get("/").json()["snaptime"]
            cached_payload = cache.load()
            if cached_payload["snapshot_id"] != live_snapshot_id:
                live_schema = self.catalog.get("/schema").json()
                diff = compute_diff(cached_payload["schema"], live_schema)
                if not diff.is_empty():
                    logging.getLogger("deriva_ml").warning(
                        "pin_schema: cache at %s, live at %s; "
                        "structural drift detected (see returned SchemaDiff)",
                        cached_payload["snapshot_id"], live_snapshot_id,
                    )
                    drift = diff
        cache.pin(reason=reason)
        return drift

    def unpin_schema(self) -> None:
        """Clear the schema-cache pin. No-op if not pinned.

        Works in any mode. After unpinning, :meth:`refresh_schema`
        is allowed again (subject to the pending-rows guard).

        Raises:
            FileNotFoundError: If the workspace has no cache file.
        """
        SchemaCache(self.working_dir).unpin()

    def pin_status(self) -> "PinStatus":
        """Return the current pin state of the local schema cache.

        Works in any mode.

        Returns:
            A :class:`PinStatus` snapshot: ``pinned`` flag, UTC
            ``pinned_at`` timestamp (or None), caller-supplied
            ``pin_reason`` (or None), and the cache's current
            ``pinned_snapshot_id``.

        Raises:
            FileNotFoundError: If the workspace has no cache file.
        """
        return SchemaCache(self.working_dir).pin_status()

    def diff_schema(self) -> "SchemaDiff":
        """Return the structural diff between the cached and live schemas.

        Online mode only. Fetches the live catalog's ``/schema``
        payload, compares it against the cached copy with
        :func:`~deriva_ml.core.schema_diff.compute_diff`, and returns
        the result. The returned :class:`SchemaDiff` may be empty
        (no drift) — callers should check ``diff.is_empty()`` rather
        than truthiness.

        Unlike :meth:`pin_schema`, this method never modifies the
        cache and never logs a warning; it is a pure inspection
        operation.

        Returns:
            A :class:`SchemaDiff`, possibly empty.

        Raises:
            DerivaMLReadOnlyError: If called in offline mode.
            FileNotFoundError: If the workspace has no cache file.
        """
        from deriva_ml.core.schema_diff import compute_diff

        if self._mode is not ConnectionMode.online:
            raise DerivaMLReadOnlyError("diff_schema requires online mode")
        cache = SchemaCache(self.working_dir)
        cached_payload = cache.load()
        live_schema = self.catalog.get("/schema").json()
        return compute_diff(cached_payload["schema"], live_schema)
```

- [ ] **Step 3: Add `TYPE_CHECKING` import for `SchemaDiff` at the top of `base.py`**

In the existing `if TYPE_CHECKING:` block around line 74 (which already imports `CatalogProvenance`, `Execution`, `DerivaModel`, `SchemaValidationReport`), append:

```python
    from deriva_ml.core.schema_diff import SchemaDiff
```

This keeps the runtime cost zero (import is lazy inside `pin_schema` and `diff_schema`).

- [ ] **Step 4: Sanity-check — import and instantiate signatures**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run python -c "from deriva_ml import DerivaML; assert hasattr(DerivaML, 'pin_schema') and hasattr(DerivaML, 'unpin_schema') and hasattr(DerivaML, 'pin_status') and hasattr(DerivaML, 'diff_schema'); print('ok')"`

Expected: prints `ok`.

- [ ] **Step 5: Run the full fast test tier to confirm nothing broke**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/core/test_schema_cache.py tests/core/test_schema_diff.py tests/core/test_exceptions.py -q`

Expected: all pass (none of these exercise the new methods yet, but they share the import graph).

- [ ] **Step 6: Commit**

```bash
git add src/deriva_ml/core/base.py
git commit -m "feat(base): add pin_schema, unpin_schema, pin_status, diff_schema public methods"
```

---

## Task 6: Modify `refresh_schema` to enforce the pin guard

**Files:**
- Modify: `src/deriva_ml/core/base.py:506-564` (add pin check before pending-rows check)

- [ ] **Step 1: Edit `refresh_schema` body**

In `src/deriva_ml/core/base.py`, find the existing `refresh_schema` method (currently starting at line 506). Replace the body with this version (the change is inserting a new pin-guard block between the offline-mode check and the pending-rows check, and updating the docstring):

```python
    def refresh_schema(self, *, force: bool = False) -> None:
        """Fetch the current catalog schema and overwrite the workspace cache.

        Online mode only. Refuses in two cases:

        1. The cache is pinned (via :meth:`pin_schema`). Raises
           :class:`DerivaMLSchemaPinned`. ``force=True`` does NOT
           bypass a pin — call :meth:`unpin_schema` first.
        2. The workspace has pending rows (staged/leasing/leased/
           uploading/failed). Raises
           :class:`DerivaMLSchemaRefreshBlocked` unless
           ``force=True`` is passed; a forced refresh may leave
           staged rows whose metadata references columns or types
           no longer in the new schema, causing catalog-insert
           failures on the next upload.

        Args:
            force: If True, refresh even when the workspace has
                pending rows. Does NOT bypass a pin.

        Raises:
            DerivaMLReadOnlyError: If called in offline mode.
            DerivaMLSchemaPinned: If the cache is pinned (any
                ``force`` value).
            DerivaMLSchemaRefreshBlocked: If ``force=False`` and the
                workspace has pending rows (and the cache is not
                pinned).
        """
        from deriva_ml.model.catalog import DerivaModel

        if self._mode is not ConnectionMode.online:
            raise DerivaMLReadOnlyError(
                "refresh_schema requires online mode"
            )
        cache = SchemaCache(self.working_dir)
        if cache.exists() and cache.pin_status().pinned:
            pin_info = cache.pin_status()
            raise DerivaMLSchemaPinned(
                f"refresh_schema refused: cache is pinned at snapshot "
                f"{pin_info.pinned_snapshot_id}"
                + (f" (reason: {pin_info.pin_reason})" if pin_info.pin_reason else "")
                + ". Call ml.unpin_schema() first."
            )
        store = self.workspace.execution_state_store()
        count = store.count_pending_rows()
        if count > 0 and not force:
            raise DerivaMLSchemaRefreshBlocked(
                f"refresh_schema requires a drained workspace; "
                f"{count} pending rows. Run ml.upload_pending() first, "
                f"or call refresh_schema(force=True) to discard local "
                f"state (staged rows may become inconsistent with the "
                f"new schema)."
            )
        live_snapshot_id = self.catalog.get("/").json()["snaptime"]
        live_schema = self.catalog.get("/schema").json()
        old_snapshot_id = cache.snapshot_id()
        cache.write(
            snapshot_id=live_snapshot_id,
            hostname=self.host_name,
            catalog_id=str(self.catalog_id),
            ml_schema=self.model.ml_schema,
            schema=live_schema,
        )
        # Reload the in-memory model so this session sees the new schema.
        self.model = DerivaModel.from_cached(
            live_schema,
            catalog=self.catalog,
            ml_schema=self.model.ml_schema,
            domain_schemas=self.model.domain_schemas,
            default_schema=self.model.default_schema,
        )
        logging.getLogger("deriva_ml").info(
            "schema cache refreshed from %s to %s",
            old_snapshot_id, live_snapshot_id,
        )
```

Key diff points:
- Add pin check immediately after offline-mode check (and the cache is constructed earlier so the pin check can use it — keeping the original `cache = SchemaCache(...)` line further down is now redundant, so I fold it into the one at the top)
- Update the docstring to document the new `DerivaMLSchemaPinned` behavior

- [ ] **Step 2: Run the existing refresh-schema test to confirm the unpinned path still works**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_offline_init.py -q`

Expected: all tests pass. (The pin-guard test is in Task 7.)

- [ ] **Step 3: Commit**

```bash
git add src/deriva_ml/core/base.py
git commit -m "feat(base): refresh_schema refuses when cache is pinned"
```

---

## Task 7: Integration tests in `test_schema_pin.py`

**Files:**
- Create: `tests/core/test_schema_pin.py` (10 tests; live-catalog tests gated on `DERIVA_HOST`)

- [ ] **Step 1: Write the tests**

Create `tests/core/test_schema_pin.py`:

```python
"""Integration tests for pin/unpin/diff on DerivaML.

Tests that don't need a live catalog (offline pin/unpin, missing
cache, diff_schema offline) always run. Tests that need a live
catalog are gated on ``DERIVA_HOST``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest


requires_catalog = pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="requires a live catalog at DERIVA_HOST",
)


# ------------------------- offline-capable tests --------------------------

def test_pin_schema_offline_returns_none(tmp_path):
    """Offline mode: pin_schema persists a pin and returns None."""
    from deriva_ml import ConnectionMode, DerivaML
    from deriva_ml.core.schema_cache import SchemaCache

    # Plant a cache so offline init succeeds.
    SchemaCache(tmp_path).write(
        snapshot_id="s0",
        hostname="h",
        catalog_id="1",
        ml_schema="deriva-ml",
        schema={
            "schemas": {
                "deriva-ml": {
                    "schema_name": "deriva-ml",
                    "tables": {},
                    "annotations": {},
                    "comment": None,
                }
            },
            "acls": {},
            "annotations": {},
        },
    )
    ml = DerivaML(
        hostname="h", catalog_id="1",
        mode=ConnectionMode.offline, working_dir=tmp_path,
    )
    result = ml.pin_schema(reason="offline test")
    assert result is None
    assert ml.pin_status().pinned is True
    assert ml.pin_status().pin_reason == "offline test"


def test_unpin_schema_works_offline(tmp_path):
    from deriva_ml import ConnectionMode, DerivaML
    from deriva_ml.core.schema_cache import SchemaCache
    SchemaCache(tmp_path).write(
        snapshot_id="s0", hostname="h", catalog_id="1",
        ml_schema="deriva-ml",
        schema={"schemas": {"deriva-ml": {"schema_name": "deriva-ml", "tables": {}}}},
    )
    ml = DerivaML(hostname="h", catalog_id="1",
                  mode=ConnectionMode.offline, working_dir=tmp_path)
    ml.pin_schema(reason="x")
    assert ml.pin_status().pinned is True
    ml.unpin_schema()
    assert ml.pin_status().pinned is False


def test_pin_status_reflects_cache_state(tmp_path):
    from deriva_ml import ConnectionMode, DerivaML
    from deriva_ml.core.schema_cache import SchemaCache
    SchemaCache(tmp_path).write(
        snapshot_id="s-current", hostname="h", catalog_id="1",
        ml_schema="deriva-ml",
        schema={"schemas": {"deriva-ml": {"schema_name": "deriva-ml", "tables": {}}}},
    )
    ml = DerivaML(hostname="h", catalog_id="1",
                  mode=ConnectionMode.offline, working_dir=tmp_path)
    status = ml.pin_status()
    assert status.pinned is False
    assert status.pinned_snapshot_id == "s-current"


def test_diff_schema_offline_raises(tmp_path):
    from deriva_ml import ConnectionMode, DerivaML
    from deriva_ml.core.exceptions import DerivaMLReadOnlyError
    from deriva_ml.core.schema_cache import SchemaCache
    SchemaCache(tmp_path).write(
        snapshot_id="s0", hostname="h", catalog_id="1",
        ml_schema="deriva-ml",
        schema={"schemas": {"deriva-ml": {"schema_name": "deriva-ml", "tables": {}}}},
    )
    ml = DerivaML(hostname="h", catalog_id="1",
                  mode=ConnectionMode.offline, working_dir=tmp_path)
    with pytest.raises(DerivaMLReadOnlyError):
        ml.diff_schema()


# --------------------------- live-catalog tests ---------------------------

@requires_catalog
def test_pin_schema_online_no_drift_returns_none(test_ml):
    """Freshly-initialized online ml: cache is at live snapshot → no drift."""
    result = test_ml.pin_schema(reason="no-drift test")
    assert result is None
    assert test_ml.pin_status().pinned is True
    test_ml.unpin_schema()  # cleanup


@requires_catalog
def test_pin_schema_online_with_drift_returns_diff_and_logs_warning(
    test_ml, caplog,
):
    """Forge a drift scenario: rewrite the cache with a bogus snapshot
    + schema that differs from live; pin; expect SchemaDiff + warning."""
    from deriva_ml.core.schema_cache import SchemaCache

    # Force a cache with a known-fake snapshot_id and a missing-table
    # payload so compute_diff reports an 'added_tables' result.
    cache = SchemaCache(test_ml.working_dir)
    current = cache.load()
    forged_schema = {
        "schemas": {
            current["ml_schema"]: {
                "schema_name": current["ml_schema"],
                "tables": {},   # live will have many tables → all added
            }
        }
    }
    cache.write(
        snapshot_id="FORGED-SNAPSHOT-00",
        hostname=current["hostname"],
        catalog_id=current["catalog_id"],
        ml_schema=current["ml_schema"],
        schema=forged_schema,
    )

    caplog.set_level(logging.WARNING, logger="deriva_ml")
    diff = test_ml.pin_schema(reason="drift test")
    assert diff is not None
    assert not diff.is_empty()
    assert len(diff.added_tables) > 0  # live has tables our forge didn't
    assert any(
        "drift" in r.message.lower() or "pin_schema" in r.message.lower()
        for r in caplog.records
    )
    # cleanup
    test_ml.unpin_schema()


@requires_catalog
def test_refresh_schema_refuses_when_pinned(test_ml):
    from deriva_ml.core.exceptions import DerivaMLSchemaPinned
    test_ml.pin_schema(reason="refuse test")
    try:
        with pytest.raises(DerivaMLSchemaPinned) as ei:
            test_ml.refresh_schema()
        assert "pinned" in str(ei.value).lower()
    finally:
        test_ml.unpin_schema()


@requires_catalog
def test_refresh_schema_refuses_when_pinned_even_with_force(test_ml):
    from deriva_ml.core.exceptions import DerivaMLSchemaPinned
    test_ml.pin_schema(reason="force doesn't bypass pin")
    try:
        with pytest.raises(DerivaMLSchemaPinned):
            test_ml.refresh_schema(force=True)
    finally:
        test_ml.unpin_schema()


@requires_catalog
def test_unpin_then_refresh_succeeds(test_ml):
    test_ml.pin_schema(reason="transient")
    test_ml.unpin_schema()
    # Should NOT raise now.
    test_ml.refresh_schema()


@requires_catalog
def test_diff_schema_online_returns_diff(test_ml):
    """diff_schema returns a SchemaDiff (possibly empty) online."""
    from deriva_ml.core.schema_diff import SchemaDiff
    diff = test_ml.diff_schema()
    assert isinstance(diff, SchemaDiff)
    # Fresh test_ml: cache IS live, so diff should be empty.
    assert diff.is_empty()
```

- [ ] **Step 2: Run the offline-capable subset (no catalog needed)**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_schema_pin.py -v -k "offline or status or diff_schema_offline"`

Expected: 4 tests pass (`test_pin_schema_offline_returns_none`, `test_unpin_schema_works_offline`, `test_pin_status_reflects_cache_state`, `test_diff_schema_offline_raises`); 6 others skipped (DERIVA_HOST unset).

- [ ] **Step 3: Run live-catalog subset if `DERIVA_HOST` is set**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/test_schema_pin.py -v --timeout=600`

Expected: 10 tests pass.

- [ ] **Step 4: Run full core test suite to confirm no regressions**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/ -v --timeout=600`

Expected: all core tests pass (S4 tests, hygiene-batch tests, new pin tests). If `DERIVA_HOST` is unset, skipped tests are not failures.

- [ ] **Step 5: Commit**

```bash
git add tests/core/test_schema_pin.py
git commit -m "test(schema_pin): integration tests for pin/unpin/diff_schema + refresh-guard"
```

---

## Task 8: CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md` (insert a new top section `## Unreleased — Schema pin + diff`)

- [ ] **Step 1: Prepend a new section to `CHANGELOG.md`**

Open `CHANGELOG.md` and insert immediately after the first line (`# Changelog` + `All notable changes...` preamble) and before the existing `## Unreleased — H3:` section, the new section:

```markdown
## Unreleased — Schema pin + diff

### Added

- **`DerivaML.pin_schema(reason: str | None = None) -> SchemaDiff | None`** — freeze the local schema cache at its current snapshot. While pinned, `refresh_schema()` refuses to update the cache (even with `force=True`). Online-mode pin additionally returns a `SchemaDiff` when the live catalog's schema has drifted structurally from the cached one (and logs a WARNING).
- **`DerivaML.unpin_schema() -> None`** — clear the pin. No-op if not pinned.
- **`DerivaML.pin_status() -> PinStatus`** — return a frozen `PinStatus` snapshot (pinned flag, UTC pinned_at, pin_reason, pinned_snapshot_id).
- **`DerivaML.diff_schema() -> SchemaDiff`** — pure inspection (online only). Returns the structural diff between the cached and live schemas; `SchemaDiff` is a Pydantic model with `.render()` for human-readable output and `.model_dump()` for JSON.
- **`deriva_ml.core.schema_cache.PinStatus`** — Pydantic `BaseModel(frozen=True)`.
- **`deriva_ml.core.schema_diff`** — new module: `SchemaDiff` + 7 fine-grained record types (`AddedTable`, `RemovedTable`, `AddedColumn`, `RemovedColumn`, `ColumnTypeChange`, `AddedForeignKey`, `RemovedForeignKey`) + `compute_diff(cached, live)` walker.
- **`DerivaMLSchemaPinned`** — new `DerivaMLConfigurationError` subclass raised by `refresh_schema()` on a pinned cache.

### Changed

- **Cache file format.** `schema-cache.json` gains an optional top-level `"pin"` object (`{"at": "...", "reason": "..."}`). Presence means pinned. Backward-compatible: unpinned caches written by prior versions remain valid without change.
- **`DerivaML.refresh_schema()`** now raises `DerivaMLSchemaPinned` before its pending-rows check when the cache is pinned. `force=True` does NOT bypass a pin — it only bypasses the pending-rows guard.
- **`SchemaCache.write()`** internals: the tmp + fsync + rename dance was extracted into a private `_write_atomic(payload)` helper so `pin()`/`unpin()` reuse identical on-disk discipline. Public signature and behavior unchanged.

### Migration

No caller changes required. This is strictly additive; existing cache files remain valid.
```

- [ ] **Step 2: Sanity check — the file still parses as Markdown**

Run: `head -40 CHANGELOG.md`

Expected: clean output, the new section appears after the preamble and before the H3 section.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): schema pin + diff"
```

---

## Final verification

- [ ] **Step 1: Run fast tiers — unit tests, local_db, core (offline-capable subset)**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ tests/core/test_schema_cache.py tests/core/test_schema_diff.py tests/core/test_exceptions.py tests/core/test_schema_pin.py -q`

Expected: all pass, no skips beyond the live-catalog ones.

- [ ] **Step 2: Run live-catalog tiers (if `DERIVA_HOST` set)**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/core/ -q --timeout=600`

Expected: all tests pass.

- [ ] **Step 3: Lint**

Run: `uv run ruff check src/ && uv run ruff format --check src/`

Expected: both clean. If `ruff format --check` reports differences, run `uv run ruff format src/` and commit.

- [ ] **Step 4: Demonstrate end-to-end with the smoke script from the spec (optional, paste into PR)**

```python
from deriva_ml import DerivaML, ConnectionMode
from deriva_ml.core.exceptions import DerivaMLSchemaPinned

ml = DerivaML(host, catalog_id, working_dir="/tmp/smoke", mode=ConnectionMode.online)
print(ml.pin_status())                # pinned=False, snapshot_id=<current>
result = ml.pin_schema(reason="paper repro")
print(result)                          # None (no drift on a fresh cache)
print(ml.pin_status())                # pinned=True, pin_reason="paper repro"
try:
    ml.refresh_schema()
except DerivaMLSchemaPinned as e:
    print("correctly refused:", e)
ml.unpin_schema()
print(ml.pin_status())                # pinned=False
ml.refresh_schema()                    # now succeeds
```

---

## Commit summary (expected)

After implementation, the branch `claude/schema-pin` will have these commits on top of `main`:

1. `docs(spec): schema pin + diff design` (already committed, `54835da`)
2. `refactor(schema_cache): extract _write_atomic helper for reuse by pin/unpin`
3. `feat(schema_cache): add pin/unpin/pin_status + PinStatus Pydantic model`
4. `feat(schema_diff): add SchemaDiff Pydantic model + compute_diff walker`
5. `feat(exceptions): add DerivaMLSchemaPinned for pinned-cache refresh guard`
6. `feat(base): add pin_schema, unpin_schema, pin_status, diff_schema public methods`
7. `feat(base): refresh_schema refuses when cache is pinned`
8. `test(schema_pin): integration tests for pin/unpin/diff_schema + refresh-guard`
9. `docs(changelog): schema pin + diff`

Nine commits, each a small independent unit — easy to review, easy to revert individually if something surfaces post-merge.
