# Unified Local SQLite Layer — Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two independent denormalization engines (catalog-side and bag-side) with a single unified engine that runs SQL joins against local SQLite, add a cached tabular-read API, and remove Phase 1 compatibility shims.

**Architecture:** The unified denormalizer uses `_prepare_wide_table()` for join planning (unchanged), `PagedFetcher` to populate the working DB with catalog rows, and SQLAlchemy SELECTs against the local SQLite (working DB or attached slice) for the actual joins. The cached tabular-read API stores results in `main.db` with a registry table. The working DB changes from a single file to a per-schema directory layout using `SchemaBuilder`'s ATTACH mechanism. `_LegacyWorkingDataView` and `WorkingDataCache` are deleted.

**Tech Stack:** Python ≥3.12, SQLAlchemy 2.x, SQLite (WAL, ATTACH), deriva-py (ErmrestCatalog, ermrest_model), pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-04-15-unified-local-db-phase2-design.md`

---

## Environment & conventions

- Work in worktree `/Users/carl/GitHub/deriva-ml/.claude/worktrees/compassionate-visvesvaraya/` on branch `claude/compassionate-visvesvaraya`.
- `uv` at `/Users/carl/.local/bin/uv`. Prepend to PATH if not found.
- `DERIVA_ML_ALLOW_DIRTY=true uv run pytest ...` for tests.
- `uv run ruff format src/ tests/` and `uv run ruff check src/` after changes.
- Commit after each task. Messages use `feat(local_db):`, `refactor(local_db):`, `test(local_db):` prefixes.

## File structure

### New files

| Path | Responsibility |
|------|----------------|
| `src/deriva_ml/local_db/denormalize.py` | Unified denormalization engine: `denormalize()` function + `DenormalizeResult` class. |
| `src/deriva_ml/local_db/result_cache.py` | Cached tabular-read API: `ResultCache`, `CachedResult`, `CachedResultMeta` classes. Registry table management, cache key generation, query with sort/filter/pagination. |
| `tests/local_db/test_denormalize.py` | Unit tests for the unified denormalizer against canned bag fixtures. |
| `tests/local_db/test_result_cache.py` | Unit tests for the cached tabular-read API. |
| `tests/local_db/test_denormalize_live.py` | Integration tests for catalog denormalization via PagedFetcher (requires `DERIVA_HOST`). |

### Modified files

| Path | Change |
|------|--------|
| `src/deriva_ml/local_db/paths.py` | Change `working_db_path()` to return a directory path (`working/`) instead of a file path (`working.db`). Add `working_main_db_path()` for the `main.db` inside. |
| `src/deriva_ml/local_db/workspace.py` | Add `local_schema` property (lazy `LocalSchema` build from catalog `Model`). Add `orm_class(name)` convenience. Add `rebuild_schema()`. Update `engine` to use directory layout. Add `cached_table_read`, `cache_denormalized`, `list_cached_results`, `get_cached_result`, `invalidate_cache` methods. Delete `_LegacyWorkingDataView`, `legacy_working_data_view()`, `_RESERVED_TABLES`. Update `attach_slice` for multi-schema slice attachment (prefix aliases). |
| `src/deriva_ml/local_db/schema.py` | No changes needed — `LocalSchema.build()` already handles directory-based multi-schema. |
| `src/deriva_ml/local_db/__init__.py` | Add exports: `CachedResult`, `CachedResultMeta`, `ResultCache`, `DenormalizeResult`, `denormalize`. |
| `src/deriva_ml/core/base.py` | Replace `working_data` property with `workspace` property. Rewire `cache_table`, `cache_features` to use workspace cached-read API. Remove `_LegacyWorkingDataView` usage. |
| `src/deriva_ml/dataset/dataset.py` | `denormalize_as_dataframe`, `denormalize_as_dict` become 3-line delegates to unified `denormalize()`. `cache_denormalized` delegates to `workspace.cache_denormalized()`. Delete `_denormalize_datapath` (~190 lines). |
| `src/deriva_ml/dataset/dataset_bag.py` | `denormalize_as_dataframe`, `denormalize_as_dict` become delegates. Delete `_denormalize` (~100 lines). Add `_workspace` property that wraps the bag's `DatabaseModel` as a lightweight `Workspace`. |
| `src/deriva_ml/execution/execution.py` | Update `_get_manifest` to use `self._ml_object.workspace` instead of `self._ml_object.working_data.workspace`. |
| `src/deriva_ml/core/working_data.py` | Deleted entirely. |
| `tests/test_working_data.py` | Deleted entirely. |

### Unchanged files (verified no impact)

| Path | Why unchanged |
|------|---------------|
| `src/deriva_ml/local_db/sqlite_helpers.py` | Phase 1, no changes needed. |
| `src/deriva_ml/local_db/paged_fetcher.py` | Phase 1, no changes needed. |
| `src/deriva_ml/local_db/paged_fetcher_ermrest.py` | Phase 1, no changes needed. |
| `src/deriva_ml/local_db/manifest_store.py` | Phase 1, no changes needed. |
| `src/deriva_ml/model/catalog.py` | `_prepare_wide_table()`, `_table_relationship()`, `_schema_to_paths()` are consumed as-is. |
| `src/deriva_ml/interfaces.py` | `DatasetLike` protocol methods keep their signatures; implementations change. |

---

## Task 1: Update path helpers for directory-based working DB

The working DB changes from a single file (`working.db`) to a directory (`working/`) with per-schema files.

**Files:**
- Modify: `src/deriva_ml/local_db/paths.py`
- Modify: `tests/local_db/test_paths.py`

- [ ] **Step 1: Write failing tests for the new path helpers.**

Add to `tests/local_db/test_paths.py`:

```python
class TestWorkingDir:
    def test_working_dir_is_directory_not_file(self, tmp_path: Path) -> None:
        d = p.working_db_path(tmp_path, "example.org", "42")
        # Should end with 'working' (a directory), not 'working.db' (a file)
        assert d.name == "working"
        assert not d.name.endswith(".db")

    def test_working_dir_under_workspace_root(self, tmp_path: Path) -> None:
        d = p.working_db_path(tmp_path, "example.org", "42")
        assert d == tmp_path / "catalogs" / "example.org__42" / "working"


class TestWorkingMainDbPath:
    def test_main_db_inside_working_dir(self, tmp_path: Path) -> None:
        main = p.working_main_db_path(tmp_path, "example.org", "42")
        assert main == tmp_path / "catalogs" / "example.org__42" / "working" / "main.db"
```

- [ ] **Step 2: Run tests to confirm failure.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_paths.py -v -k "TestWorkingDir or TestWorkingMainDbPath"
```

Expected: `AttributeError` or assertion failure (old `working_db_path` returns `working.db`).

- [ ] **Step 3: Update `paths.py`.**

Change `working_db_path` to return a directory path, and add `working_main_db_path`:

```python
def working_db_path(working_dir: Path, hostname: str, catalog_id: str | int) -> Path:
    """Return the per-catalog working DB directory path.

    The working DB is a directory containing main.db plus per-schema .db files
    (created by SchemaBuilder's multi-schema ATTACH pattern).

    Layout: {workspace_root}/working/
    """
    return workspace_root(working_dir, hostname, catalog_id) / "working"


def working_main_db_path(working_dir: Path, hostname: str, catalog_id: str | int) -> Path:
    """Return the main.db file inside the working DB directory.

    This is the file the SQLAlchemy engine opens. Per-schema .db files are
    ATTACH'd into connections on this engine.
    """
    return working_db_path(working_dir, hostname, catalog_id) / "main.db"
```

- [ ] **Step 4: Fix the existing `TestWorkingDbPath` test** that asserts `working.db`:

```python
class TestWorkingDbPath:
    def test_under_workspace_root(self, tmp_path: Path) -> None:
        db = p.working_db_path(tmp_path, "example.org", "42")
        assert db == tmp_path / "catalogs" / "example.org__42" / "working"
```

- [ ] **Step 5: Run all path tests — expect pass.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_paths.py -v
```

- [ ] **Step 6: Lint + commit.**

```bash
uv run ruff format src/deriva_ml/local_db/paths.py tests/local_db/test_paths.py
uv run ruff check src/deriva_ml/local_db/paths.py
git add src/deriva_ml/local_db/paths.py tests/local_db/test_paths.py
git commit -m "refactor(local_db): working DB path becomes directory for multi-schema ATTACH"
```

---

## Task 2: Update `Workspace` for directory-based working DB + `local_schema`

The `Workspace` engine now opens `main.db` inside the `working/` directory. Add `local_schema` property (lazy `LocalSchema` build), `orm_class()` convenience, and `rebuild_schema()`.

**Files:**
- Modify: `src/deriva_ml/local_db/workspace.py`
- Modify: `tests/local_db/test_workspace.py`

- [ ] **Step 1: Write failing tests for the new workspace capabilities.**

Add to `tests/local_db/test_workspace.py`:

```python
class TestWorkspaceDirectoryLayout:
    def test_engine_creates_working_directory(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            _ = ws.engine
            working_dir = tmp_path / "catalogs" / "h__1" / "working"
            assert working_dir.is_dir()
            assert (working_dir / "main.db").is_file()
        finally:
            ws.close()


class TestLocalSchema:
    def test_local_schema_is_none_before_build(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            assert ws.local_schema is None
        finally:
            ws.close()

    def test_build_local_schema_from_model(
        self, tmp_path: Path, canned_bag_model
    ) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            ws.build_local_schema(
                model=canned_bag_model,
                schemas=["isa", "deriva-ml"],
            )
            assert ws.local_schema is not None
            # ORM classes should be available
            assert ws.orm_class("Image") is not None
            assert ws.orm_class("Dataset") is not None
        finally:
            ws.close()

    def test_local_schema_creates_per_schema_files(
        self, tmp_path: Path, canned_bag_model
    ) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            ws.build_local_schema(
                model=canned_bag_model,
                schemas=["isa", "deriva-ml"],
            )
            working_dir = tmp_path / "catalogs" / "h__1" / "working"
            assert (working_dir / "isa.db").is_file()
            assert (working_dir / "deriva-ml.db").is_file()
        finally:
            ws.close()

    def test_rebuild_schema_disposes_and_recreates(
        self, tmp_path: Path, canned_bag_model
    ) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            ws.build_local_schema(model=canned_bag_model, schemas=["isa", "deriva-ml"])
            first = ws.local_schema
            ws.rebuild_schema(model=canned_bag_model, schemas=["isa", "deriva-ml"])
            second = ws.local_schema
            assert first is not second
        finally:
            ws.close()

    def test_orm_class_returns_none_for_unknown(
        self, tmp_path: Path, canned_bag_model
    ) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            ws.build_local_schema(model=canned_bag_model, schemas=["isa", "deriva-ml"])
            result = ws.orm_class("NonexistentTable")
            assert result is None
        finally:
            ws.close()
```

- [ ] **Step 2: Run tests to confirm failure.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_workspace.py -v -k "TestWorkspaceDirectoryLayout or TestLocalSchema"
```

- [ ] **Step 3: Update `Workspace` implementation.**

Key changes to `src/deriva_ml/local_db/workspace.py`:

1. Update `working_db_path` property to use `paths.working_main_db_path` (returns `working/main.db`).
2. Update `engine` property to create the engine on `main.db` inside the `working/` directory.
3. Add `_local_schema: LocalSchema | None = None` to `__init__`.
4. Add `local_schema` property (returns `_local_schema`, may be None).
5. Add `build_local_schema(model, schemas)` method that creates a `LocalSchema` using `database_path=working_dir` (the directory, not `main.db` — `SchemaBuilder` handles the per-schema files).
6. Add `rebuild_schema(model, schemas)` that disposes the old schema and builds a new one.
7. Add `orm_class(name)` convenience that delegates to `local_schema.get_orm_class(name)`.
8. Delete `_LegacyWorkingDataView` class, `legacy_working_data_view()` method, and `_RESERVED_TABLES`.

**Important implementation detail:** The `LocalSchema` and the `Workspace` engine must share the same `main.db` file. When `build_local_schema` is called, it creates the `LocalSchema` using `database_path=self.working_db_path` (the `working/` directory). `SchemaBuilder` will create `main.db` there and ATTACH per-schema files. The workspace's own engine (used by `ManifestStore`, result cache) also opens `main.db`. These must be the **same engine** to avoid two-engine conflicts over WAL. Solution: after `LocalSchema.build()`, set `self._engine = self._local_schema.engine`. The workspace engine IS the LocalSchema engine.

- [ ] **Step 4: Run all workspace tests — fix any failures from removed `_LegacyWorkingDataView`.**

Tests that used `legacy_working_data_view()` or `_RESERVED_TABLES` must be removed or rewritten. The new tests from Step 1 must pass. Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_workspace.py -v
```

Fix failures iteratively until all pass.

- [ ] **Step 5: Update `ManifestStore` access path.**

`manifest_store()` on `Workspace` must still work. After the engine unification (Step 3), `ManifestStore(self.engine)` uses the shared engine. Verify manifest tests still pass:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_manifest_store.py tests/local_db/test_manifest_migration.py -v
```

- [ ] **Step 6: Lint + commit.**

```bash
uv run ruff format src/deriva_ml/local_db/ tests/local_db/
uv run ruff check src/deriva_ml/local_db/
git add src/deriva_ml/local_db/workspace.py tests/local_db/test_workspace.py
git commit -m "feat(local_db): directory-based working DB with local_schema and ORM"
```

---

## Task 3: Multi-schema slice attachment

Update `attach_slice` to ATTACH each of the slice's per-schema files under prefixed aliases (`slice_isa`, `slice_deriva-ml`, etc.) so the denormalizer can join across working DB and slice tables in the same connection.

**Files:**
- Modify: `src/deriva_ml/local_db/workspace.py`
- Modify: `tests/local_db/test_workspace.py`

- [ ] **Step 1: Write failing tests.**

```python
class TestMultiSchemaSliceAttach:
    def test_attach_slice_multi_schema(self, tmp_path: Path, canned_bag_model) -> None:
        """Attaching a multi-schema slice makes all schema tables visible."""
        from deriva_ml.local_db.schema import LocalSchema

        # Create a slice with multi-schema data
        slice_dir = tmp_path / "catalogs" / "h__1" / "slices" / "s1"
        slice_dir.mkdir(parents=True)
        ls = LocalSchema.build(
            model=canned_bag_model,
            schemas=["isa", "deriva-ml"],
            database_path=slice_dir,
        )
        # Insert a test row into the slice
        from sqlalchemy import insert, text
        image_t = ls.find_table("isa.Image")
        with ls.engine.begin() as conn:
            conn.execute(insert(image_t).values(RID="SLICE-IMG-1", Filename="test.jpg"))
        ls.dispose()

        # Open workspace and attach the slice
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            ws.build_local_schema(model=canned_bag_model, schemas=["isa", "deriva-ml"])
            with ws.attach_slice("s1") as conn:
                # Slice tables should be visible under prefixed aliases
                result = conn.execute(text("SELECT RID FROM slice_isa.Image")).fetchall()
                assert len(result) == 1
                assert result[0][0] == "SLICE-IMG-1"
        finally:
            ws.close()

    def test_attach_slice_detaches_on_exit(self, tmp_path: Path, canned_bag_model) -> None:
        """After the context manager exits, slice schemas are no longer visible."""
        from deriva_ml.local_db.schema import LocalSchema

        slice_dir = tmp_path / "catalogs" / "h__1" / "slices" / "s1"
        slice_dir.mkdir(parents=True)
        ls = LocalSchema.build(
            model=canned_bag_model, schemas=["isa", "deriva-ml"],
            database_path=slice_dir,
        )
        ls.dispose()

        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            ws.build_local_schema(model=canned_bag_model, schemas=["isa", "deriva-ml"])
            with ws.attach_slice("s1") as conn:
                pass  # Just enter and exit
            # After exit, slice schemas should be detached
            from sqlalchemy import text
            with ws.engine.connect() as conn:
                import pytest
                with pytest.raises(Exception):
                    conn.execute(text("SELECT * FROM slice_isa.Image"))
        finally:
            ws.close()
```

- [ ] **Step 2: Run tests — expect failure.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_workspace.py -v -k "TestMultiSchemaSliceAttach"
```

- [ ] **Step 3: Update `attach_slice` to handle multi-schema.**

The current `attach_slice` does a single ATTACH of `slice.db`. Replace with:

1. Read the slice directory for `*.db` files.
2. For `main.db`, ATTACH as `slice_main`.
3. For each `{schema}.db`, ATTACH as `slice_{schema}`.
4. On context exit, DETACH all of them.

Store the list of attached aliases so detach can clean them all up.

Also update `slice_db_path` — it currently returns a single `slice.db` file. For multi-schema slices, the slice directory contains `main.db` + per-schema files (same layout as the working DB). The path helper should return the directory, not a single file. This parallels the Task 1 change for working DB.

- [ ] **Step 4: Run all workspace tests — expect pass.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_workspace.py -v
```

- [ ] **Step 5: Lint + commit.**

```bash
uv run ruff format src/deriva_ml/local_db/ tests/local_db/
uv run ruff check src/deriva_ml/local_db/
git add src/deriva_ml/local_db/workspace.py src/deriva_ml/local_db/paths.py \
        tests/local_db/test_workspace.py
git commit -m "feat(local_db): multi-schema slice attachment with prefixed ATTACH aliases"
```

---

## Task 4: Cached tabular-read API (`result_cache.py`)

The `ResultCache` manages named cached tables in `main.db` with a registry for metadata, TTL, and re-query support.

**Files:**
- Create: `src/deriva_ml/local_db/result_cache.py`
- Create: `tests/local_db/test_result_cache.py`

- [ ] **Step 1: Write failing tests.**

Write `tests/local_db/test_result_cache.py` covering:

- `ResultCache` construction (creates registry table in engine).
- `cache_key()` is deterministic — same params produce same key, different params produce different keys.
- `store()` + `query()` roundtrip: store rows, query them back, verify column names, row count.
- `query()` with `sort_by` — ascending and descending.
- `query()` with `filter_col` + `filter_val` — substring match.
- `query()` with `limit` + `offset` — pagination.
- `list_cached()` returns metadata for all stored results.
- `invalidate(cache_key=...)` removes one result.
- `invalidate(source=...)` removes all results from a source.
- `invalidate()` with no args removes everything.
- TTL expiry: store with `ttl_seconds=1`, sleep briefly, `has()` returns False.
- `CachedResult.to_dataframe()` returns correct DataFrame.
- `CachedResult.iter_rows()` yields dicts.
- `CachedResult.query()` returns a new `CachedResult` with the query applied.
- `CachedResult.invalidate()` removes the backing table.

Use the workspace engine from a `Workspace` fixture (same as manifest store tests).

- [ ] **Step 2: Run tests — expect failure.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_result_cache.py -v
```

- [ ] **Step 3: Implement `result_cache.py`.**

Write `src/deriva_ml/local_db/result_cache.py` with:

- `CachedResultMeta` dataclass: `cache_key`, `source`, `tool_name`, `params`, `columns`, `row_count`, `created_at`, `ttl_seconds`. Methods: `is_expired()`, `age_seconds()`, `to_summary()`.
- `CachedResult` class: wraps a cache key + engine reference. Properties: `cache_key`, `source`, `row_count`, `columns`, `fetched_at`. Methods: `to_dataframe()`, `iter_rows()`, `query(sort_by, sort_desc, filter_col, filter_val, limit, offset) -> CachedResult`, `invalidate()`.
- `ResultCache` class: manages `cached_results_registry` table in `main.db`. Methods: `ensure_schema()`, `cache_key(**params) -> str`, `has(key) -> bool`, `store(key, columns, rows, meta)`, `query(key, ...) -> CachedResult | None`, `list_cached() -> list[CachedResultMeta]`, `invalidate(cache_key, source) -> int`, `get(key) -> CachedResult | None`.

Key design decisions:
- Use SQLAlchemy for all DB access (not raw sqlite3 like MCP's version).
- Keep original column names (properly quoted via SQLAlchemy) — no `c0/c1` sanitization.
- Cache key generation: `sha256(f"{tool_name}:{json.dumps(sorted_params)}")[:16]` prefixed with `rc_`.
- Each cached result is a separate table named by its cache key.
- `CachedResult.query()` executes SQL with WHERE/ORDER BY/LIMIT/OFFSET against the backing table and returns a new `CachedResult` wrapping the filtered view (as a temp table or in-memory list).

- [ ] **Step 4: Run tests — expect pass.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_result_cache.py -v
```

- [ ] **Step 5: Lint + commit.**

```bash
uv run ruff format src/deriva_ml/local_db/result_cache.py tests/local_db/
uv run ruff check src/deriva_ml/local_db/result_cache.py
git add src/deriva_ml/local_db/result_cache.py tests/local_db/test_result_cache.py
git commit -m "feat(local_db): add ResultCache with CachedResult handle and registry"
```

---

## Task 5: Unified denormalization engine (`denormalize.py`)

The core of Phase 2. One function that plans the join, ensures rows are present, builds the SQL, and executes it.

**Files:**
- Create: `src/deriva_ml/local_db/denormalize.py`
- Create: `tests/local_db/test_denormalize.py`

- [ ] **Step 1: Write failing tests.**

Write `tests/local_db/test_denormalize.py`. Use the `canned_bag_model` fixture from Phase 1's conftest. The test strategy:

1. Create a workspace with a `LocalSchema` built from the canned model.
2. Insert test rows directly into the per-schema SQLite files (via the ORM engine).
3. Call `denormalize(workspace, model, dataset_rid, include_tables, source="catalog")`.
4. Verify the result has the correct columns, row count, and joined values.

Test cases:
- **Simple two-table join** (Image → Subject via FK): insert images with Subject FK values, insert subjects. Denormalize `["Image", "Subject"]`. Verify each output row has both `Image.*` and `Subject.*` columns joined correctly.
- **Multi-hop chain** (Image → Subject, Image → Diagnosis): verify three-table denormalization.
- **LEFT JOIN for nullable FK**: Image with Subject=NULL should still appear in output, with Subject columns as NULL.
- **Multi-schema**: Tables span `isa` and `deriva-ml` schemas. Verify cross-schema join works.
- **Slice source** (`source="slice"`): Create a slice directory with per-schema files, populate it, call with `source="slice"`. Verify it reads from the slice, not the working DB.
- **Empty dataset**: No dataset members. Verify empty result with correct column structure.
- **Ambiguous path detection**: If the model has ambiguous FK paths, verify the denormalizer raises an appropriate error (delegated to `_prepare_wide_table`).

For the canned model fixture: extend `conftest.py` to add `Dataset_Image` association table to the schema (so we can test the full Dataset → Dataset_Image → Image chain). Or use a simpler test that skips the Dataset-membership layer and just tests the join logic with directly-inserted rows.

Note: `_prepare_wide_table()` requires a `DatasetLike` object as its first argument. For unit tests without a real `Dataset` or `DatasetBag`, create a minimal mock that satisfies the protocol — it needs `list_dataset_members()` and `list_dataset_children()`.

- [ ] **Step 2: Run tests — expect failure.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_denormalize.py -v
```

- [ ] **Step 3: Implement `denormalize.py`.**

Write `src/deriva_ml/local_db/denormalize.py` with:

```python
@dataclass
class DenormalizeResult:
    """Result of a denormalization operation."""
    columns: list[tuple[str, str]]   # (name, type) pairs
    cache_key: str | None
    row_count: int
    _rows: list[dict[str, Any]]      # materialized rows
    _engine: Engine

    def to_dataframe(self) -> pd.DataFrame: ...
    def iter_rows(self) -> Generator[dict[str, Any], None, None]: ...


def denormalize(
    workspace: Workspace,
    model: DerivaModel,
    dataset_rid: str,
    include_tables: list[str],
    version: DatasetVersion | str | None = None,
    slice_id: str | None = None,
    source: str = "catalog",
) -> DenormalizeResult: ...
```

The algorithm (five steps from the spec):

1. **Plan:** Call `model._prepare_wide_table(dataset_mock, dataset_rid, include_tables)` where `dataset_mock` is a lightweight wrapper providing `list_dataset_members()`. Get `(element_tables, column_specs, multi_schema)`.

2. **Ensure ORM:** Access `workspace.local_schema` (triggers lazy build if needed).

3. **Populate rows:** For `"catalog"`: use `PagedFetcher` to fetch each table's rows into the working DB's per-schema SQLite tables. For `"slice"`: attach the slice via `workspace.attach_slice(slice_id)`. For `"hybrid"`: attach slice + fetch missing tables.

4. **Build SQL:** From `_prepare_wide_table`'s output, build a SQLAlchemy `select(*labeled_columns).join(...)` chain. Table references resolve to `{schema}.{table}` (working DB) or `slice_{schema}.{table}` (slice) based on source.

5. **Execute:** Run against workspace engine, collect rows, return `DenormalizeResult`.

The key challenge is Step 4: translating `_prepare_wide_table`'s `(path, join_conditions, join_types)` into SQLAlchemy SELECT with proper ATTACH alias references. Today's `DatasetBag._denormalize` already does this for the bag case (references ORM classes from the bag's `DatabaseModel`). The unified version does the same but with alias awareness.

For the `"catalog"` source, a mock `DatasetLike` is needed to pass to `_prepare_wide_table`. This mock's `list_dataset_members()` returns the members fetched from the catalog (or pre-existing in the working DB). The simplest approach: fetch association-table rows first (e.g. `Dataset_Image` where `Dataset=dataset_rid`), then construct the mock from those.

- [ ] **Step 4: Run tests — iterate until all pass.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_denormalize.py -v
```

- [ ] **Step 5: Lint + commit.**

```bash
uv run ruff format src/deriva_ml/local_db/denormalize.py tests/local_db/
uv run ruff check src/deriva_ml/local_db/denormalize.py
git add src/deriva_ml/local_db/denormalize.py tests/local_db/test_denormalize.py
git commit -m "feat(local_db): unified denormalization engine"
```

---

## Task 6: Wire workspace cached-read methods

Add `cached_table_read`, `cache_denormalized`, `list_cached_results`, `get_cached_result`, `invalidate_cache` to `Workspace`. These delegate to the `ResultCache` from Task 4 and the `denormalize()` from Task 5.

**Files:**
- Modify: `src/deriva_ml/local_db/workspace.py`
- Modify: `tests/local_db/test_workspace.py`

- [ ] **Step 1: Write failing tests for the workspace cached-read surface.**

```python
class TestWorkspaceCachedReads:
    def test_cached_table_read_stores_and_returns(
        self, tmp_path: Path, canned_bag_model
    ) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            ws.build_local_schema(model=canned_bag_model, schemas=["isa", "deriva-ml"])
            # Insert rows into the Subject table via ORM
            subject_t = ws.local_schema.find_table("Subject")
            from sqlalchemy import insert
            with ws.engine.begin() as conn:
                conn.execute(insert(subject_t).values(RID="S1", Name="Alice"))
                conn.execute(insert(subject_t).values(RID="S2", Name="Bob"))

            # cached_table_read should store and return
            result = ws.cached_table_read("Subject", source="local")
            assert result.row_count == 2
            df = result.to_dataframe()
            assert len(df) == 2
        finally:
            ws.close()

    def test_list_cached_results(self, tmp_path: Path, canned_bag_model) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            ws.build_local_schema(model=canned_bag_model, schemas=["isa", "deriva-ml"])
            subject_t = ws.local_schema.find_table("Subject")
            from sqlalchemy import insert
            with ws.engine.begin() as conn:
                conn.execute(insert(subject_t).values(RID="S1", Name="Alice"))

            ws.cached_table_read("Subject", source="local")
            results = ws.list_cached_results()
            assert len(results) >= 1
        finally:
            ws.close()

    def test_invalidate_cache(self, tmp_path: Path, canned_bag_model) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            ws.build_local_schema(model=canned_bag_model, schemas=["isa", "deriva-ml"])
            subject_t = ws.local_schema.find_table("Subject")
            from sqlalchemy import insert
            with ws.engine.begin() as conn:
                conn.execute(insert(subject_t).values(RID="S1", Name="Alice"))

            cr = ws.cached_table_read("Subject", source="local")
            ws.invalidate_cache(cache_key=cr.cache_key)
            assert len(ws.list_cached_results()) == 0
        finally:
            ws.close()
```

- [ ] **Step 2: Run tests — expect failure.**

- [ ] **Step 3: Implement the cached-read methods on `Workspace`.**

Each method delegates to a `ResultCache` instance (lazy-created, stored as `self._result_cache`):

- `cached_table_read(table, ...)`: generate cache key from params, check `result_cache.has(key)`, if miss: read rows from the local schema table (via `select(table).fetch()`), store, return `CachedResult`.
- `cache_denormalized(model, dataset_rid, ...)`: generate cache key, check, if miss: call `denormalize(self, model, ...)`, store result, return `CachedResult`.
- `list_cached_results()`: delegate to `result_cache.list_cached()`.
- `get_cached_result(key)`: delegate to `result_cache.get(key)`.
- `invalidate_cache(...)`: delegate to `result_cache.invalidate(...)`.

- [ ] **Step 4: Run tests — expect pass.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_workspace.py -v
```

- [ ] **Step 5: Lint + commit.**

```bash
uv run ruff format src/deriva_ml/local_db/ tests/local_db/
uv run ruff check src/deriva_ml/local_db/
git add src/deriva_ml/local_db/workspace.py src/deriva_ml/local_db/result_cache.py \
        tests/local_db/test_workspace.py
git commit -m "feat(local_db): wire cached-read methods onto Workspace"
```

---

## Task 7: Rewire `Dataset.denormalize_*` and `DatasetBag.denormalize_*`

Replace the two denormalization implementations with delegates to the unified `denormalize()`.

**Files:**
- Modify: `src/deriva_ml/dataset/dataset.py`
- Modify: `src/deriva_ml/dataset/dataset_bag.py`

- [ ] **Step 1: Rewire `Dataset.denormalize_as_dataframe` and `denormalize_as_dict`.**

In `src/deriva_ml/dataset/dataset.py`:

Replace `denormalize_as_dataframe` body (line 974–1044):

```python
def denormalize_as_dataframe(self, include_tables, version=None, **kwargs):
    from deriva_ml.local_db.denormalize import denormalize
    result = denormalize(
        workspace=self._ml_instance.workspace,
        model=self._ml_instance.model,
        dataset_rid=self.rid,
        include_tables=include_tables,
        version=version,
        source="catalog",
    )
    return result.to_dataframe()
```

Replace `denormalize_as_dict` body (line 1091–1147):

```python
def denormalize_as_dict(self, include_tables, version=None, **kwargs):
    from deriva_ml.local_db.denormalize import denormalize
    result = denormalize(
        workspace=self._ml_instance.workspace,
        model=self._ml_instance.model,
        dataset_rid=self.rid,
        include_tables=include_tables,
        version=version,
        source="catalog",
    )
    yield from result.iter_rows()
```

Replace `cache_denormalized` body (line 1047–1089):

```python
def cache_denormalized(self, include_tables, version=None, force=False):
    result = self._ml_instance.workspace.cache_denormalized(
        model=self._ml_instance.model,
        dataset_rid=self.rid,
        include_tables=include_tables,
        version=version,
        source="catalog",
        refresh=force,
    )
    return result.to_dataframe()
```

**Delete** `_denormalize_datapath` method entirely (~190 lines, lines 782–972).

`denormalize_columns` and `denormalize_info` stay unchanged — they use `_prepare_wide_table` directly and don't need the denormalize engine.

- [ ] **Step 2: Rewire `DatasetBag.denormalize_as_dataframe` and `denormalize_as_dict`.**

In `src/deriva_ml/dataset/dataset_bag.py`:

Add a `_workspace` property that wraps the bag's `DatabaseModel`:

```python
@property
def _workspace(self) -> "Workspace":
    """Lightweight workspace wrapping the bag's SQLite files as a slice."""
    if not hasattr(self, "__workspace") or self.__workspace is None:
        from deriva_ml.local_db.workspace import Workspace
        # The bag directory is both the working dir and the slice
        bag_dir = self.model.database_dir
        self.__workspace = Workspace(
            working_dir=bag_dir.parent,
            hostname="bag",
            catalog_id=self.dataset_rid,
        )
        # Build schema from the bag's model
        self.__workspace.build_local_schema(
            model=self.model._model,  # the ERMrest Model
            schemas=self.model.schemas,
        )
    return self.__workspace
```

Replace `denormalize_as_dataframe` body (line 865–930):

```python
def denormalize_as_dataframe(self, include_tables, version=None, **kwargs):
    from deriva_ml.local_db.denormalize import denormalize
    result = denormalize(
        workspace=self._workspace,
        model=self.model,
        dataset_rid=self.dataset_rid,
        include_tables=include_tables,
        source="slice",
    )
    return result.to_dataframe()
```

Replace `denormalize_as_dict` similarly.

**Delete** `_denormalize` method entirely (~100 lines, lines 764–863).

`denormalize_columns` stays unchanged.

- [ ] **Step 3: Run existing denormalization tests.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_denormalize.py -v
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_denormalize_info.py -v
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_composite_fk_denormalize.py -v
```

These are integration tests that require `DERIVA_HOST`. They exercise the full path. Fix any failures iteratively — the most likely issues are:
- `self._ml_instance.workspace` not yet wired (Task 8 adds the `workspace` property to `DerivaML`).
- The `_workspace` property on `DatasetBag` needing refinement to work with the bag's existing `DatabaseModel` / ORM.

**Note:** These tests may not pass until Task 8 (which adds `ml.workspace`). Mark this step as "verify after Task 8 if needed."

- [ ] **Step 4: Lint + commit.**

```bash
uv run ruff format src/deriva_ml/dataset/
uv run ruff check src/deriva_ml/dataset/
git add src/deriva_ml/dataset/dataset.py src/deriva_ml/dataset/dataset_bag.py
git commit -m "refactor(dataset): denormalize_* delegates to unified local_db engine"
```

---

## Task 8: Rewire `DerivaML` — add `workspace` property, update `cache_table`/`cache_features`, clean up

Replace `ml.working_data` with `ml.workspace`. Rewire cache methods. Delete `WorkingDataCache`.

**Files:**
- Modify: `src/deriva_ml/core/base.py`
- Modify: `src/deriva_ml/execution/execution.py`
- Delete: `src/deriva_ml/core/working_data.py`
- Delete: `tests/test_working_data.py`

- [ ] **Step 1: Add `workspace` property to `DerivaML`.**

In `src/deriva_ml/core/base.py`, replace the `working_data` property (around line 437–476) with:

```python
@property
def workspace(self) -> "Workspace":
    """Per-catalog Workspace for local caching, denormalization, and asset manifests.

    Creates the workspace lazily on first access. Runs legacy manifest
    import on creation.
    """
    from deriva_ml.local_db.workspace import Workspace

    if not hasattr(self, "_workspace") or self._workspace is None:
        self._workspace = Workspace(
            working_dir=self.working_dir,
            hostname=self.host_name,
            catalog_id=self.catalog_id,
        )
        try:
            n = self._workspace.import_legacy_manifests()
            if n:
                import logging
                logging.getLogger("deriva_ml").info(
                    "Migrated %d legacy asset manifests into workspace", n,
                )
        except Exception as exc:
            import logging
            logging.getLogger("deriva_ml").warning(
                "Legacy manifest migration failed: %s", exc,
            )
        # Build the local schema so the ORM is available
        self._workspace.build_local_schema(
            model=self.model._model,
            schemas=[self.ml_schema, *self.domain_schemas],
        )
    return self._workspace
```

- [ ] **Step 2: Rewire `cache_table`.**

Replace the body of `cache_table` (around line 478–507):

```python
def cache_table(self, table_name: str, force: bool = False) -> "pd.DataFrame":
    result = self.workspace.cached_table_read(
        table=table_name, source="catalog", refresh=force,
    )
    return result.to_dataframe()
```

- [ ] **Step 3: Rewire `cache_features`.**

Replace the body of `cache_features` (around line 509–545):

```python
def cache_features(self, table_name, feature_name, force=False, **kwargs):
    import pandas as pd
    from deriva_ml.local_db.result_cache import ResultCache

    cache_key = ResultCache.cache_key(
        "features", table=table_name, feature=feature_name, **kwargs
    )
    if not force:
        existing = self.workspace.get_cached_result(cache_key)
        if existing is not None:
            return existing.to_dataframe()

    features = self.fetch_table_features(table_name, feature_name=feature_name, **kwargs)
    records = [r.model_dump(mode="json") for r in features.get(feature_name, [])]
    df = pd.DataFrame(records)
    # Store via workspace result cache
    columns = list(df.columns)
    rows = df.to_dict(orient="records")
    from deriva_ml.local_db.result_cache import CachedResultMeta
    self.workspace._result_cache.store(
        cache_key, columns, rows,
        CachedResultMeta(cache_key=cache_key, source="catalog",
                         tool_name="features", params={"table": table_name, "feature": feature_name},
                         columns=columns, row_count=len(rows)),
    )
    return df
```

- [ ] **Step 4: Update `Execution._get_manifest`.**

In `src/deriva_ml/execution/execution.py`, change (around line 1385–1390):

```python
def _get_manifest(self) -> AssetManifest:
    if not hasattr(self, "_manifest") or self._manifest is None:
        ws = self._ml_object.workspace
        self._manifest = AssetManifest(ws.manifest_store(), self.execution_rid)
    return self._manifest
```

- [ ] **Step 5: Delete `core/working_data.py` and `tests/test_working_data.py`.**

```bash
git rm src/deriva_ml/core/working_data.py
git rm tests/test_working_data.py
```

- [ ] **Step 6: Remove any remaining `working_data` references.**

Search and fix:

```bash
grep -rn "working_data\|WorkingDataCache" src/deriva_ml/ --include="*.py" | grep -v __pycache__
```

Any remaining references must be updated to use `workspace`.

- [ ] **Step 7: Run full test suite.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/test_manifest.py -v
```

Also run any integration tests that are available:

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_denormalize.py -v
```

- [ ] **Step 8: Lint + commit.**

```bash
uv run ruff format src/ tests/
uv run ruff check src/
git add -A
git commit -m "refactor(core): replace working_data with workspace, delete WorkingDataCache"
```

---

## Task 9: Update `__init__.py` exports + full test suite verification

**Files:**
- Modify: `src/deriva_ml/local_db/__init__.py`

- [ ] **Step 1: Update exports.**

```python
"""Unified local SQLite layer for deriva-ml."""

from __future__ import annotations

from deriva_ml.local_db.denormalize import DenormalizeResult, denormalize
from deriva_ml.local_db.manifest_store import ManifestStore
from deriva_ml.local_db.paged_fetcher import PagedFetcher
from deriva_ml.local_db.paged_fetcher_ermrest import ErmrestPagedClient
from deriva_ml.local_db.result_cache import CachedResult, CachedResultMeta, ResultCache
from deriva_ml.local_db.schema import LocalSchema
from deriva_ml.local_db.workspace import Workspace

__all__ = [
    "CachedResult",
    "CachedResultMeta",
    "DenormalizeResult",
    "ErmrestPagedClient",
    "LocalSchema",
    "ManifestStore",
    "PagedFetcher",
    "ResultCache",
    "Workspace",
    "denormalize",
]
```

- [ ] **Step 2: Verify imports work.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run python -c "
from deriva_ml.local_db import (
    Workspace, PagedFetcher, LocalSchema, ManifestStore,
    ErmrestPagedClient, ResultCache, CachedResult, CachedResultMeta,
    DenormalizeResult, denormalize,
)
print('OK')
"
```

- [ ] **Step 3: Run the complete test suite.**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/test_manifest.py -v --tb=short
```

If `DERIVA_HOST=localhost` is available:

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/ -v --tb=short -x
```

- [ ] **Step 4: Lint + commit.**

```bash
uv run ruff format src/deriva_ml/local_db/
uv run ruff check src/deriva_ml/local_db/
git add src/deriva_ml/local_db/__init__.py
git commit -m "docs(local_db): Phase 2 public exports"
```

---

## Verification checklist (run before declaring Phase 2 done)

- [ ] `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ -v` — all pass.
- [ ] `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/asset/test_manifest.py -v` — all pass.
- [ ] `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_denormalize.py -v` — all pass (requires `DERIVA_HOST`).
- [ ] `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_denormalize_info.py -v` — all pass.
- [ ] `grep -rn "working_data\|WorkingDataCache" src/deriva_ml/ --include="*.py"` — zero hits (except comments).
- [ ] `grep -rn "_denormalize_datapath\|def _denormalize" src/deriva_ml/dataset/ --include="*.py"` — zero hits.
- [ ] `uv run ruff check src/` — clean.
- [ ] `from deriva_ml.local_db import ...` — all 10 exports work.
- [ ] `ls src/deriva_ml/core/working_data.py` — file does not exist.
- [ ] `ls tests/test_working_data.py` — file does not exist.

## What this plan does NOT deliver (deferred)

- MCP migration (Phase 3, separate repo).
- Live integration test for full denormalize pipeline (`test_denormalize_live.py`) — noted in Task 5 as a test to add, but full end-to-end requires a populated catalog. The existing `tests/dataset/test_denormalize.py` serves as the live integration test.
- Upload-back from local DB (future spec).
- Cross-workspace slice attachment (future spec).

Phase 2's deliverable is: **one denormalization engine, one cache API, zero legacy code.**
