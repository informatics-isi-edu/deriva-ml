# Schema-Pin Design

**Status:** Draft · **Date:** 2026-04-22 · **Subsystem:** Schema pinning + diff (built on S4's SchemaCache)

## 1. Goal

Give users a first-class way to freeze the `SchemaCache` at a specific snapshot (pin) and inspect what differs between the cached schema and the live catalog (diff). S4 introduced the cache and a warn-on-drift-but-don't-auto-refresh policy; this subsystem adds explicit pin state and programmatic access to drift information.

**Scope is deliberately narrow: pin and diff only. No migration.** A future subsystem can add map operations that rewrite staged SQLite state against a new schema; this one does not.

## 2. Scope

**In scope:**
- `ml.pin_schema(reason: str | None = None) -> SchemaDiff | None` — freeze the cache; return the diff if the cache is behind live.
- `ml.unpin_schema() -> None` — clear the pin.
- `ml.pin_status() -> PinStatus` — structured read of current pin state.
- `ml.diff_schema() -> SchemaDiff` — show differences between cached and live schemas (online only).
- `refresh_schema()` refuses when the cache is pinned (even with `force=True`).
- New exception `DerivaMLSchemaPinned`.
- New Pydantic models `PinStatus` and `SchemaDiff` (plus fine-grained diff records).
- Extended cache file format: optional `"pin"` object at the top level.
- New helper method `SchemaCache._write_atomic(payload)` extracted from existing `write()` so `pin`/`unpin` can reuse it.

**Explicitly out of scope:**
- **Map operations** (`pin_to_snapshot(snapshot_id)`, `migrate_schema(column_map=...)`, etc.) — deferred to a later subsystem if needed. The diff this subsystem produces is the foundation a future migration subsystem would build on.
- **Staged-data rewriting.** If a user refreshes a schema and their SQLite pending rows reference old columns, the upload will fail — same as S4. This subsystem doesn't touch that.
- **Non-FK keys, annotations, ACLs, column nullability/defaults, comments.** Diff is intentionally narrow: schemas, tables, columns (including type changes), foreign keys. Adding more diff dimensions later is a matter of extending the Pydantic model.
- **CLI exposure.** Python API only; users wrap in their own scripts if they want a CLI.

## 3. Architecture

Three units, same separation-of-concerns boundary the rest of the codebase uses:

**1. `SchemaCache` extensions** (`src/deriva_ml/core/schema_cache.py`)
- New methods: `pin(reason)`, `unpin()`, `pin_status() -> PinStatus`.
- `PinStatus` Pydantic model defined here (it projects the cache file).
- Cache file format extended: optional top-level `"pin"` object.
- Private `_write_atomic(payload)` helper extracted from existing `write()`; `pin`/`unpin` reuse it.

**2. `schema_diff.py`** — new file at `src/deriva_ml/core/schema_diff.py`
- `SchemaDiff` Pydantic model + fine-grained diff records (`AddedTable`, `RemovedTable`, `AddedColumn`, `RemovedColumn`, `ColumnTypeChange`, `AddedForeignKey`, `RemovedForeignKey`).
- `compute_diff(cached: dict, live: dict) -> SchemaDiff` function.

**3. `DerivaML` public methods** (`src/deriva_ml/core/base.py`)
- Four new methods — thin wrappers delegating to `SchemaCache` and `compute_diff`.
- `refresh_schema()` modified to consult `pin_status()` before its existing pending-rows guard.

### 3.1 Idiom rationale (per `CLAUDE.md`)

All new user-facing types are Pydantic `BaseModel` with `ConfigDict(frozen=True)`:
- `PinStatus` — returned from `pin_status()`; user may serialize for logs/reports.
- `SchemaDiff` and all record types — returned from `pin_schema` / `diff_schema`; may be serialized, has a `.render()` method, must be user-inspectable.

Matches the H3 decision to make `ExecutionSnapshot` Pydantic for the same reasons.

## 4. Cache file format

**Before:**
```json
{
  "snapshot_id": "...",
  "hostname": "...",
  "catalog_id": "...",
  "ml_schema": "...",
  "schema": { ... }
}
```

**After:**
```json
{
  "snapshot_id": "...",
  "hostname": "...",
  "catalog_id": "...",
  "ml_schema": "...",
  "schema": { ... },
  "pin": {                             ← optional
    "at": "2026-04-22T20:30:00Z",
    "reason": "reproducing 2025 paper analysis"
  }
}
```

Presence of the `"pin"` key means pinned; absence means not pinned. `reason` may be `null`. No sentinel `pinned: false` field — eliminates a possible inconsistent state (`pinned: true` with `pinned_at: null`).

**Backward compatibility:** unpinned caches written by S4 are valid here without change (no `pin` key; `pin_status()` correctly reports `pinned=False`).

## 5. Component details

### 5.1 `PinStatus` (in `schema_cache.py`)

```python
from pydantic import BaseModel, ConfigDict

class PinStatus(BaseModel):
    """Current pin state of a SchemaCache. Frozen Pydantic snapshot."""

    model_config = ConfigDict(frozen=True)

    pinned: bool
    pinned_at: datetime | None
    pin_reason: str | None
    pinned_snapshot_id: str   # the cache's current snapshot_id; always present
```

### 5.2 `SchemaCache.pin` / `.unpin` / `.pin_status`

```python
def pin(self, reason: str | None = None) -> None:
    """Mark the cache pinned at its current snapshot.

    Idempotent: pinning an already-pinned cache updates ``pinned_at``
    and ``reason`` to reflect the most recent call. Atomic write.

    Raises:
        FileNotFoundError: If the cache doesn't exist.
    """

def unpin(self) -> None:
    """Clear pin state. No-op if already unpinned. Atomic write when needed."""

def pin_status(self) -> PinStatus:
    """Return current pin state.

    Raises:
        FileNotFoundError: If the cache doesn't exist.
    """
```

### 5.3 `SchemaDiff` records (in `schema_diff.py`)

```python
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
    """Structured diff between a cached schema and a live schema."""

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

    def is_empty(self) -> bool: ...
    def render(self) -> str: ...    # human-readable; empty for empty diffs
    def __str__(self) -> str: return self.render()
```

### 5.4 `compute_diff`

```python
def compute_diff(cached: dict, live: dict) -> SchemaDiff:
    """Compare two ERMrest /schema payloads.

    Walks both deterministically (sorted keys) and emits typed records.
    V1 dimensions: schemas (add/remove), tables (add/remove), columns
    (add/remove + type change), foreign keys (add/remove). Out of
    scope for V1: non-FK keys, annotations, ACLs, column nullability
    and defaults, comments.
    """
```

### 5.5 `DerivaML` public methods

Full docstrings are embedded in the plan (§2.3 of the component details section). Brief signatures:

```python
def pin_schema(self, reason: str | None = None) -> "SchemaDiff | None": ...
def unpin_schema(self) -> None: ...
def pin_status(self) -> "PinStatus": ...
def diff_schema(self) -> "SchemaDiff": ...   # raises DerivaMLReadOnlyError offline
def refresh_schema(self, *, force: bool = False) -> None:
    # Extended: raises DerivaMLSchemaPinned if pinned, before pending-rows check.
```

### 5.6 New exception

In `src/deriva_ml/core/exceptions.py`:

```python
class DerivaMLSchemaPinned(DerivaMLConfigurationError):
    """refresh_schema() called on a pinned cache.

    The cache has been explicitly pinned via pin_schema(). Call
    unpin_schema() first if you really want to refresh. ``force=True``
    does NOT bypass a pin — it only bypasses the pending-rows guard.
    """
```

Subclass of `DerivaMLConfigurationError`, parallel to `DerivaMLSchemaRefreshBlocked`.

## 6. Data flow

### 6.1 `pin_schema(reason)` online

```
pin_schema(reason)
  → if offline: skip to 6.2
  → live_snapshot_id = self.catalog.get("/").json()["snaptime"]
  → cached = SchemaCache.load()
  → if cached.snapshot_id != live_snapshot_id:
       → live_schema = self.catalog.get("/schema").json()
       → diff = compute_diff(cached["schema"], live_schema)
       → logger.warning("pinning at %s; live is at %s — see returned SchemaDiff", ...)
       → result = diff if not diff.is_empty() else None
     else:
       → result = None
  → cache.pin(reason)
  → return result
```

**Subtlety:** If snapshot ids differ but content doesn't (rare — snapshot bumped by row-level change), `compute_diff` returns empty and we return `None`. From the user's perspective there's no *schema* drift, which is what they asked about.

### 6.2 `pin_schema(reason)` offline

```
pin_schema(reason)
  → self._mode is offline
  → cache.pin(reason)
  → return None  # no live to compare
```

### 6.3 `refresh_schema` pinned-guard

```
refresh_schema(force=False)
  → if offline: raise DerivaMLReadOnlyError
  → cache = SchemaCache(self.working_dir)
  → if cache.pin_status().pinned:
       → raise DerivaMLSchemaPinned(...)
  → count = store.count_pending_rows()
  → if count > 0 and not force: raise DerivaMLSchemaRefreshBlocked
  → [existing refresh logic unchanged]
```

Pin check is **before** the pending-rows check — pin is the stronger guard.

### 6.4 `diff_schema`

```
diff_schema()
  → if offline: raise DerivaMLReadOnlyError
  → cached = cache.load()
  → live_schema = self.catalog.get("/schema").json()
  → return compute_diff(cached["schema"], live_schema)
```

Always returns a `SchemaDiff`, possibly empty. Consistent with the diff-shape convention.

### 6.5 Pin + pending rows

`pin_schema` and `unpin_schema` ignore the pending-rows count — pin only sets a flag, and a flag doesn't invalidate staged data. In fact, pinning is often the *right* move when pending rows exist: it guarantees the schema can't drift out from under them. `refresh_schema` is the operation that cares about pending rows, and its guard still applies after unpin.

### 6.6 Concurrency / crash safety

Inherits S4's atomic-write guarantee: every `pin` / `unpin` / `refresh_schema` writes a tmp file + fsync + rename. A crash during `pin` leaves the cache file untouched. The `_write_atomic(payload)` helper extracted from `write()` ensures the three write paths share the same on-disk discipline.

No locking. `SchemaCache` has no concurrent-writer protection and this subsystem introduces no concurrent writers.

## 7. Error handling summary

| Scenario | Exception / behavior |
|---|---|
| `pin_schema` in any mode, cache doesn't exist | `FileNotFoundError` |
| `pin_schema` offline, cache exists | Returns `None`, pin persisted |
| `pin_schema` online, no drift | Returns `None`, pin persisted |
| `pin_schema` online, drift detected | Returns `SchemaDiff`, warning logged, pin persisted |
| `unpin_schema` when not pinned | No-op, no error |
| `unpin_schema` when pinned | Pin cleared, cache file rewritten |
| `pin_status` any mode | Returns `PinStatus`; `FileNotFoundError` propagates if no cache |
| `diff_schema` offline | `DerivaMLReadOnlyError` |
| `diff_schema` online | Returns `SchemaDiff` (possibly empty) |
| `refresh_schema` on pinned cache (any `force` value) | `DerivaMLSchemaPinned` |
| `refresh_schema(force=False)` unpinned, with pending rows | `DerivaMLSchemaRefreshBlocked` (S4 existing) |
| `refresh_schema(force=True)` unpinned, with pending rows | Succeeds (S4 existing) |

## 8. Testing plan

### 8.1 Unit tests — no catalog needed

**`tests/core/test_schema_cache.py`** (extend existing file):
1. `test_pin_on_unpinned_cache_sets_fields`
2. `test_pin_without_reason`
3. `test_pin_idempotent_updates_metadata`
4. `test_unpin_clears_fields`
5. `test_unpin_on_unpinned_is_no_op`
6. `test_pin_status_on_missing_cache_raises`
7. `test_pin_persists_across_instances`
8. `test_pin_atomic_write_crash_recovery`
9. `test_cache_file_format_has_nested_pin_object`

**`tests/core/test_schema_diff.py`** (new file):
10. `test_empty_diff_when_schemas_identical`
11. `test_added_schema`
12. `test_removed_schema`
13. `test_added_table`
14. `test_removed_table`
15. `test_added_column`
16. `test_removed_column`
17. `test_column_type_change`
18. `test_added_fkey`
19. `test_removed_fkey`
20. `test_diff_render_produces_human_readable`
21. `test_diff_determinism`

**`tests/core/test_exceptions.py`** (or wherever exception tests live):
22. `test_derivaml_schema_pinned_inherits_configuration_error`

### 8.2 Integration tests — gated on `DERIVA_HOST`

**`tests/core/test_schema_pin.py`** (new file):
23. `test_pin_schema_online_no_drift_returns_none`
24. `test_pin_schema_online_with_drift_returns_diff_and_logs_warning`
25. `test_pin_schema_offline_returns_none`
26. `test_unpin_schema_works_offline`
27. `test_pin_status_reflects_cache_state`
28. `test_refresh_schema_refuses_when_pinned`
29. `test_refresh_schema_refuses_when_pinned_even_with_force`
30. `test_unpin_then_refresh_succeeds`
31. `test_diff_schema_offline_raises`
32. `test_diff_schema_online_returns_diff`

### 8.3 Regression

- Existing S4 tests (`tests/core/test_schema_cache.py`, `tests/core/test_offline_init.py`, `tests/core/test_connection_mode.py`) must still pass — unpinned path behavior is unchanged.
- `refresh_schema` tests from S4 must still pass; the pinned-guard is a pre-check.
- Hygiene-batch's `test_resume_execution_offline_skips_reconcile` / `test_create_execution_offline_raises` / `test_offline_workspace_skips_lease_reconcile` must still pass — none of them pin.

### 8.4 Optional manual smoke

A paste-into-PR demonstration showing the full round-trip:

```python
ml = DerivaML(host, catalog_id, working_dir="/tmp/smoke", mode="online")
print(ml.pin_status())                # pinned=False
result = ml.pin_schema(reason="paper repro")
print(result)                          # None (no drift)
print(ml.pin_status())                # pinned=True, pin_reason="paper repro"
try:
    ml.refresh_schema()
except DerivaMLSchemaPinned as e:
    print("correctly refused:", e)
ml.unpin_schema()
print(ml.pin_status())                # pinned=False
ml.refresh_schema()                    # now succeeds
```

Not required in CI.

## 9. Risks

1. **`compute_diff` determinism.** Walking dicts with non-sorted iteration would produce different diff-list orders between runs, breaking programmatic equality. Mitigation: explicit `sorted()` at every dict-iteration site in the walker. Test `test_diff_determinism` guards against regressions.

2. **ERMrest `/schema` payload shape drift across deriva-py versions.** The diff walker reads specific keys (`schemas`, `tables`, `columns`, `type`, `foreign_keys`, `foreign_key_columns`, `referenced_columns`). If a future deriva-py version reshapes `/schema`, the walker would silently produce wrong diffs. Mitigation: unit tests use hand-constructed dicts with the current shape, so a deriva-py version bump that changes shape would fail our own tests first (before reaching users). Not defending against this more aggressively — it's a standard dependency-pinning risk.

3. **`pin_atomic_write_crash_recovery` test fragility.** Stubbing `os.replace` mid-write to simulate a crash is OS-ish. Mitigation: use `monkeypatch.setattr` with a function that raises `OSError` — same pattern as S4's existing cache atomic-write test, which has been stable.

4. **Pin-state + offline-mode interaction edge cases.** Offline init reads the cache; if the cache is pinned and someone calls `refresh_schema` while offline, the `DerivaMLReadOnlyError` fires first (before the pin check). Intentional: offline is the stronger constraint. Covered by `test_diff_schema_offline_raises` and implicit in the refresh logic.

## 10. Rollout

Single PR against `main`. No feature flag. No breaking change:

- Existing unpinned cache files remain valid (no `"pin"` key → `pin_status()` returns `pinned=False`).
- `refresh_schema` gains a new failure mode (`DerivaMLSchemaPinned`) but only when the user has explicitly pinned, which doesn't happen in pre-S-pin code.
- All four new methods are additive.

CHANGELOG entry under a new "Unreleased — Schema pin + diff" section.
