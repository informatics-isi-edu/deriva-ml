# S4 — Cache-Backed Offline Mode Design

**Status:** Draft · **Date:** 2026-04-21 · **Subsystem:** Phase 2 S4

## 1. Goal

Make `ConnectionMode.offline` actually work without network access. `DerivaML.__init__` currently does `get_authn_session()`, `connect_ermrest()`, and `getCatalogModel()` unconditionally, so offline mode silently requires a reachable catalog. This subsystem introduces a workspace-backed schema cache, a `CatalogStub` that raises `DerivaMLReadOnlyError` on any offline catalog access, and an explicit `refresh_schema()` method with a drift-detection policy.

## 1.1 Scope history note

The initial brainstorming pass also scoped a second hygiene item — finishing the S1a `Status`-enum migration (formerly called "H2"). On starting execution we verified that H2 had already been completed on `main` in commit `8313953` (`refactor(core): migrate base.py + mixins to ExecutionStatus.Aborted`) and that the grep-gate test `tests/test_migration_complete.py` was landed separately. The stale context that suggested otherwise came from a sibling worktree stuck at a pre-migration commit. H2 is therefore dropped from the spec; S4 is scoped to H1 only.

## 2. Scope

**In scope:**
- Cache-backed offline mode: `SchemaCache`, `CatalogStub`, `refresh_schema()`, mode-branched `__init__`.

**Out of scope (tracked in the project todo list):**
- H3 — Legacy Pydantic `ExecutionRecord` class in `execution/execution_record.py` and its 7 importers. Scheduled for a later subsystem.
- Schema-cache data-migration operations (`pin_schema`, `migrate_schema` with table/column maps). Built on top of this subsystem's `SchemaCache` foundation.
- Bug C — asset metadata None-stringification.
- Bug E.2 — deriva-py upserts by MD5+Filename, ignoring pre-leased RIDs.
- H4 — `@pytest.mark.skip` tests in `tests/dataset/test_denormalize.py` and `tests/catalog/test_clone_subset_catalog.py`.

## 3. Architecture

Three new units:

1. `SchemaCache` (`src/deriva_ml/core/schema_cache.py`) — owns one file at `<workspace>/schema-cache.json`. Methods: `exists()`, `load()`, `write(...)`, `snapshot_id()`. Atomic writes (tmp + fsync + rename).
2. `CatalogStub` (`src/deriva_ml/core/catalog_stub.py`) — drop-in replacement for `ErmrestCatalog` in offline mode. `__getattr__` raises `DerivaMLReadOnlyError("catalog access requires online mode")` on any attribute access.
3. `DerivaML.refresh_schema(force=False)` — new method on `DerivaML` (defined in `core/base.py`). Refuses if the workspace has pending rows unless `force=True`.

`DerivaML.__init__` branches on `self._mode` into `_init_online` or `_init_offline` helpers.

## 4. Status-migration hygiene (scoped out; see §1.1)

Originally this section specified the H2 `Status`-enum migration cleanup. On starting execution we discovered the migration was already completed on `main` in commit `8313953`, and the grep-gate test `tests/test_migration_complete.py` was already landed. Nothing to do. Scope section §2 has been updated; the implementation plan will also be trimmed.

## 5. H1 implementation details — cache-backed offline mode

### 5.1 `SchemaCache` (`src/deriva_ml/core/schema_cache.py`)

Owns one file at `<workspace>/schema-cache.json`. Layout:

```json
{
  "snapshot_id": "<ERMrest snapshot id>",
  "hostname": "example.org",
  "catalog_id": "42",
  "ml_schema": "deriva-ml",
  "schema": { ... full ermrest /schema payload ... }
}
```

API:

```python
class SchemaCache:
    def __init__(self, workspace_root: Path):
        self._path = workspace_root / "schema-cache.json"

    def exists(self) -> bool
    def load(self) -> dict                 # raises FileNotFoundError
    def write(
        self, *, snapshot_id: str, hostname: str,
        catalog_id: str, ml_schema: str, schema: dict,
    ) -> None                              # atomic: tmp + fsync + rename
    def snapshot_id(self) -> str | None    # None if cache missing
```

### 5.2 `CatalogStub` (`src/deriva_ml/core/catalog_stub.py`)

```python
class CatalogStub:
    """Placeholder for ErmrestCatalog in offline mode.

    Any attribute access raises DerivaMLReadOnlyError. This makes
    "I hit the catalog while offline" a loud error rather than an
    AttributeError, without requiring every mixin to guard with
    `if self._mode is offline`.
    """

    def __getattr__(self, name: str):
        raise DerivaMLReadOnlyError(
            f"catalog.{name} requires online mode; "
            f"this DerivaML instance was constructed with mode=offline"
        )

    def __repr__(self) -> str:
        return "CatalogStub(offline)"
```

### 5.3 `DerivaML.refresh_schema(force=False)`

Defined in `core/base.py`. Signature:

```python
def refresh_schema(self, *, force: bool = False) -> None:
    """Fetch the current catalog schema and overwrite the workspace cache.

    Args:
        force: If True, refresh even when the workspace has pending
            rows. Those rows may become inconsistent with the new
            schema; caller is responsible for the consequences.
            Default False refuses in that case.

    Raises:
        DerivaMLReadOnlyError: In offline mode.
        DerivaMLSchemaRefreshBlocked: force=False and the workspace
            has pending rows.
    """
```

Body:

1. If `self._mode is ConnectionMode.offline`: raise `DerivaMLReadOnlyError("refresh_schema requires online mode")`.
2. Count pending rows via `self.workspace.execution_state_store().count_pending_rows()`.
3. If count > 0 and not force: raise `DerivaMLSchemaRefreshBlocked`.
4. Fetch live `snapshot_id` and the schema dict from the catalog.
5. `SchemaCache(self.working_dir).write(...)`.
6. Reload `self.model = DerivaModel.from_cached(new_schema, ml_schema, domain_schemas, default_schema)` so the ongoing session sees the new schema.
7. Log: `logger.info("schema cache refreshed from %s to %s", old_id, new_id)`.

### 5.4 New exception `DerivaMLSchemaRefreshBlocked`

In `core/exceptions.py`, a subclass of `DerivaMLConfigurationError`:

```python
class DerivaMLSchemaRefreshBlocked(DerivaMLConfigurationError):
    """refresh_schema was called while the workspace had staged work.

    Drain the workspace first (`ml.upload_pending()`) or call
    `refresh_schema(force=True)` to discard local state.
    """
```

### 5.5 `DerivaML.__init__` changes

Replace the current lines 277–304 block with a mode branch. The `working_dir` setup must move earlier so it's available before the branch.

Mode-dispatched init:

```python
self.credential = credential or get_credential(hostname)  # local file only
cache = SchemaCache(self.working_dir)

if self._mode is ConnectionMode.online:
    self._init_online(
        hostname=hostname, catalog_id=catalog_id, check_auth=check_auth,
        cache=cache, ml_schema=ml_schema,
        domain_schemas=domain_schemas, default_schema=default_schema,
    )
else:
    self._init_offline(
        hostname=hostname, catalog_id=catalog_id,
        cache=cache, ml_schema=ml_schema,
        domain_schemas=domain_schemas, default_schema=default_schema,
    )
```

`_init_online`:

```python
def _init_online(self, *, hostname, catalog_id, check_auth, cache,
                 ml_schema, domain_schemas, default_schema):
    server = DerivaServer("https", hostname, credentials=self.credential,
                          session_config=self._get_session_config())
    if check_auth:
        try:
            server.get_authn_session()
        except Exception:
            raise DerivaMLException("...")
    self.catalog = server.connect_ermrest(catalog_id)
    live_snapshot_id = self._fetch_live_snapshot_id()

    if cache.exists():
        cached = cache.load()
        if cached["snapshot_id"] != live_snapshot_id:
            logger.warning(
                "schema cache is at snapshot %s; live catalog is at %s. "
                "Using cached schema. Call ml.refresh_schema() to update.",
                cached["snapshot_id"], live_snapshot_id,
            )
        self.model = DerivaModel.from_cached(
            cached["schema"], ml_schema=ml_schema,
            domain_schemas=domain_schemas, default_schema=default_schema,
        )
    else:
        # First-time online: fetch live, populate cache.
        live_schema = self._fetch_live_schema()
        cache.write(
            snapshot_id=live_snapshot_id, hostname=hostname,
            catalog_id=str(catalog_id), ml_schema=ml_schema,
            schema=live_schema,
        )
        self.model = DerivaModel.from_cached(
            live_schema, ml_schema=ml_schema,
            domain_schemas=domain_schemas, default_schema=default_schema,
        )
```

**Design choice flagged for user review (§5.5):** When drift is detected in online mode, the implementation **honors the cache** (uses cached schema, discards the live-fetched one). This is consistent with the "drift is a warning, never automatic" rule and leaves room for future pin-to-snapshot workflows. The alternative — use live schema for this session but don't persist to cache — is more useful for first-time users but muddies the offline-cache-is-authoritative model.

`_init_offline`:

```python
def _init_offline(self, *, hostname, catalog_id, cache,
                  ml_schema, domain_schemas, default_schema):
    if not cache.exists():
        raise DerivaMLConfigurationError(
            f"offline mode requires a cached schema at {cache._path}; "
            f"run online once first to populate the cache."
        )
    cached = cache.load()
    if cached["hostname"] != hostname or cached["catalog_id"] != str(catalog_id):
        raise DerivaMLConfigurationError(
            f"cached schema is for {cached['hostname']}/{cached['catalog_id']}, "
            f"but __init__ was called with {hostname}/{catalog_id}. "
            f"Use the matching workspace or run online to refresh."
        )
    self.catalog = CatalogStub()
    self.model = DerivaModel.from_cached(
        cached["schema"], ml_schema=ml_schema,
        domain_schemas=domain_schemas, default_schema=default_schema,
    )
```

### 5.6 `DerivaModel.from_cached` classmethod

Add to `src/deriva_ml/model/catalog.py`:

```python
@classmethod
def from_cached(
    cls, schema_dict: dict, *,
    ml_schema: str, domain_schemas: list[str] | None, default_schema: str | None,
) -> "DerivaModel":
    """Construct a DerivaModel from a cached schema dict (no network).

    Uses deriva-py's `Model.fromcatalog()` pattern in reverse —
    the dict is a captured ermrest /schema response and we
    rehydrate a `Model` object from it.
    """
    from deriva.core.ermrest_model import Model
    model = Model.fromfile_or_dict(schema_dict)  # verify exact API
    return cls(model, ml_schema=ml_schema,
               domain_schemas=domain_schemas, default_schema=default_schema)
```

The exact deriva-py API will be verified by the implementation plan (user confirmed it exists; the plan writer will grep for the precise method name).

### 5.7 Snapshot-id fetch API

`self.catalog.get_server_state()["snaptime"]` or equivalent. Exact call to be verified by the plan. The intent is "what ERMrest snapshot id is the catalog currently at" — a cheap GET that returns snapshot metadata.

### 5.8 `count_pending_rows()` on ExecutionStateStore

If this method doesn't already exist on `ExecutionStateStore`, add it:

```python
def count_pending_rows(self) -> int:
    """Count rows in non-terminal states (staged/leasing/leased/uploading/failed)."""
    with self.engine.begin() as conn:
        return conn.scalar(
            select(func.count()).select_from(self.pending_rows).where(
                self.pending_rows.c.status.in_([
                    str(PendingRowStatus.staged),
                    str(PendingRowStatus.leasing),
                    str(PendingRowStatus.leased),
                    str(PendingRowStatus.uploading),
                    str(PendingRowStatus.failed),
                ])
            )
        ) or 0
```

## 6. Data flow

**Online init — cache hit:**

```
__init__(mode=online)
  → credentials loaded (local file)
  → DerivaServer + connect_ermrest (network: catalog handle)
  → fetch live snapshot_id (network: cheap call)
  → cache.snapshot_id() == live_snapshot_id
  → self.model = DerivaModel.from_cached(cache.load()["schema"])
  → self.catalog = real ErmrestCatalog
```

**Online init — drift detected:**

```
→ cache.snapshot_id() != live_snapshot_id
→ logger.warning(...)
→ self.model = DerivaModel.from_cached(cache.load()["schema"])  # cache still authoritative
→ live schema is discarded
```

**Online init — cache miss (first time):**

```
→ cache.exists() == False
→ fetch live schema (network)
→ cache.write(...)
→ self.model = DerivaModel.from_cached(fresh_schema)
```

**Offline init — happy path:**

```
__init__(mode=offline)
  → credentials loaded
  → cache.exists() == True AND hostname/catalog_id match
  → self.catalog = CatalogStub()
  → self.model = DerivaModel.from_cached(cache.load()["schema"])
  → skip lease reconciliation (online-only block)
```

**Offline init — error paths:**

```
cache missing            → DerivaMLConfigurationError
cache host/catalog mismatch → DerivaMLConfigurationError
```

**refresh_schema():**

```
ml.refresh_schema()
  → if offline: DerivaMLReadOnlyError
  → count_pending_rows() > 0 and not force: DerivaMLSchemaRefreshBlocked
  → fetch live snapshot_id + schema (network)
  → cache.write(...)
  → self.model = DerivaModel.from_cached(new_schema)
  → logger.info(...)
```

## 7. Error handling summary

| Scenario | Exception / behavior |
|---|---|
| Offline init, no cache | `DerivaMLConfigurationError` |
| Offline init, (host,catalog) mismatch | `DerivaMLConfigurationError` |
| Any `self.catalog.X` in offline mode | `DerivaMLReadOnlyError` (via CatalogStub) |
| `refresh_schema()` in offline mode | `DerivaMLReadOnlyError` |
| `refresh_schema(force=False)` with pending rows | `DerivaMLSchemaRefreshBlocked` |
| `refresh_schema(force=True)` with pending rows | Succeeds; warning logged |
| Corrupt cache file (unparseable JSON) | `DerivaMLConfigurationError("cache at <path> is corrupt; delete and re-run online")` |
| Online cache hit + drift | Warning; no exception |
| Legacy `Status` reference in non-exempt src file | `tests/test_migration_complete.py` fails |

## 8. Testing plan

### 8.1 Unit tests (no catalog needed)

- ~~`tests/test_migration_complete.py`~~ — already exists on main from a prior migration; see §4.
- `tests/core/test_schema_cache.py` — `SchemaCache` unit tests: write+load round-trip, atomic-write recovery (tmp exists, original intact on crash), missing file returns `None` from `snapshot_id()`, corrupt JSON raises.
- `tests/core/test_catalog_stub.py` — `CatalogStub`: any attribute access raises `DerivaMLReadOnlyError` with the attribute name in the message.

### 8.2 Integration tests (require catalog, gated on `DERIVA_HOST`)

`tests/core/test_offline_init.py`:

- `test_offline_without_cache_raises` — construct in offline mode without pre-populating → `DerivaMLConfigurationError`.
- `test_online_first_populates_cache` — online init; verify `schema-cache.json` exists in workspace.
- `test_offline_after_online_succeeds` — online once, then offline against same workspace → works; `ml.catalog` is `CatalogStub`.
- `test_offline_hostname_mismatch_raises` — cache for host A; offline init with host B → `DerivaMLConfigurationError`.
- `test_refresh_schema_refuses_with_pending_rows` — stage a pending row; `refresh_schema()` → `DerivaMLSchemaRefreshBlocked`.
- `test_refresh_schema_force_discards` — same setup; `refresh_schema(force=True)` → succeeds; warning in caplog.
- `test_online_drift_warning` — write cache with fake snapshot id; reinit online → warning logged; cache unchanged; `self.model` loaded from cache.

### 8.3 Regression

The existing test suite must pass end-to-end. The `self.catalog = CatalogStub()` change could break tests that currently rely on `self.catalog is None` to detect offline mode. Plan will include an audit task.

## 9. Risks

1. **`DerivaModel.from_cached` / model-from-dict deriva-py API.** User confirmed deriva-py supports this. The plan's first task verifies the exact method name (`Model.fromfile`, `.from_json`, `.fromjson`, etc.) before anything else.
2. **Snapshot-id fetch API.** `catalog.get_server_state()["snaptime"]` or equivalent — verify exact call in the plan.
3. **Existing `self.catalog is None` assumptions.** Plan audit task: `git grep 'catalog is None'` in `src/` and `tests/`; fix any offenders.
4. **Corrupt or partially-written cache files** from earlier crashes. Atomic write + explicit `DerivaMLConfigurationError` on unparseable JSON. User can always delete the file and re-run online.

## 10. Rollout

Single PR against `main`. CHANGELOG entry documents the offline-mode work under one "Unreleased — Phase 2 Subsystem 4" heading. No feature flag. No breaking change to existing API:

- `DerivaML()` default mode is still `online`; behavior unchanged for existing callers.
- `ConnectionMode.offline` now actually works (previously silently network-dependent).
- `refresh_schema()` is net-new, not a replacement for any existing method.

Breaking change only for code that relied on `ml.catalog is None` to detect offline mode. Those callers get `DerivaMLReadOnlyError` instead, which is arguably better than silent `None`-check behavior but requires a code update. Noted in CHANGELOG.
