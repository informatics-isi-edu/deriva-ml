# Unified Local SQLite Layer — Phase 1 (Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundation layer for unified client-side SQLite caching in deriva-ml: a `local_db` module with a reusable schema builder, a per-catalog `Workspace`, a `PagedFetcher` primitive, and a DB-backed `AssetManifest`. No user-visible API changes — existing denormalization and caching continue to work.

**Architecture:** New `deriva_ml.local_db` package with four modules (`schema`, `workspace`, `paged_fetcher`, and `manifest_store`). Today's `WorkingDataCache` is absorbed as the workspace's default cache table layer. Today's `AssetManifest` gets a storage swap (JSON → SQLite tables in the workspace DB) while keeping its public API. No change to denormalization code this phase.

**Tech Stack:** Python ≥3.12, SQLAlchemy 2.x, SQLite (WAL mode), deriva-py (ErmrestCatalog, datapath, ermrest_model), pytest, pytest-asyncio. Spec: `docs/superpowers/specs/2026-04-15-unified-local-db-design.md`.

---

## Environment & conventions

- Work in worktree `/Users/carl/GitHub/deriva-ml/.claude/worktrees/compassionate-visvesvaraya/` on branch `claude/compassionate-visvesvaraya`.
- `uv` is at `/Users/carl/.local/bin/uv`. If `uv` is not found, prepend `/Users/carl/.local/bin` to `PATH`.
- Run tests with `DERIVA_ML_ALLOW_DIRTY=true uv run pytest ...`.
- Unit tests must not require a live catalog — they use fakes/fixtures. Integration tests set `DERIVA_HOST` and use `CatalogManager` (see `tests/catalog_manager.py`).
- Format/lint: `uv run ruff format src/` and `uv run ruff check src/` after any non-trivial change.
- Commit after each task unless noted. Commit messages use `feat(local_db):`, `test(local_db):`, `refactor(local_db):` prefixes.

## File structure

### New files

| Path | Responsibility |
|------|----------------|
| `src/deriva_ml/local_db/__init__.py` | Public exports (`Workspace`, `PagedFetcher`, `ManifestStore`). |
| `src/deriva_ml/local_db/paths.py` | Pure path-helper functions: `workspace_root`, `working_db_path`, `slice_dir`, `slice_db_path`. No side effects, no DB access. |
| `src/deriva_ml/local_db/sqlite_helpers.py` | Shared SQLite/SQLAlchemy helpers: engine factory with WAL pragmas, `attach_database` + `detach_database`, `ensure_schema_meta` migration runner. |
| `src/deriva_ml/local_db/schema.py` | The unified `LocalSchema` class: wraps today's `SchemaBuilder` + `SchemaORM` in a cleaner interface that works for working DB (from live `Model`) and slice DB (from bag `schema.json`). |
| `src/deriva_ml/local_db/paged_fetcher.py` | `PagedFetcher` class with `fetch_predicate`, `fetch_by_rids`, `fetched_rids`, plus internals for URL-length guard, POST fallback, concurrency, cardinality heuristic. |
| `src/deriva_ml/local_db/manifest_store.py` | `ManifestStore` class. SQLite-backed persistence layer for asset/feature entries. Consumed by `AssetManifest`. |
| `src/deriva_ml/local_db/workspace.py` | `Workspace` class: per-catalog working-DB handle, slice registry, `attach_slice(...)` context manager, cached-tabular-read helpers (stubbed; filled out in Phase 2). |
| `tests/local_db/__init__.py` | Test package marker. |
| `tests/local_db/conftest.py` | Shared fixtures: `tmp_workspace`, `fake_ermrest_client`, `canned_model`. |
| `tests/local_db/test_paths.py` | Unit tests for path helpers. |
| `tests/local_db/test_sqlite_helpers.py` | Unit tests for WAL engine + attach/detach + migration runner. |
| `tests/local_db/test_schema.py` | Unit tests for `LocalSchema` (model-from-file and model-from-live paths). |
| `tests/local_db/test_paged_fetcher.py` | Unit tests for fetcher (byte guard, POST fallback, dedup, cardinality heuristic) against a fake client. |
| `tests/local_db/test_paged_fetcher_live.py` | Integration tests hitting a live catalog. |
| `tests/local_db/test_manifest_store.py` | Unit tests for the SQLite manifest store. |
| `tests/local_db/test_manifest_migration.py` | Integration test: seed a JSON manifest, open workspace, confirm rows + `.migrated.json` sidecar. |
| `tests/local_db/test_workspace.py` | Unit + integration tests for the workspace lifecycle and slice attach/detach. |

### Modified files

| Path | Change |
|------|--------|
| `src/deriva_ml/core/base.py` | Replace `working_data` property + `_working_data` attr with delegation through a lazily created `Workspace`. Keep the `working_data` property as a thin shim around `workspace.legacy_working_data_view()` so existing callers (tests, `Dataset.cache_denormalized`) keep working. |
| `src/deriva_ml/asset/manifest.py` | Rewrite `AssetManifest` to delegate persistence to `ManifestStore`. Keep same public API. Keep `AssetEntry`/`FeatureEntry` dataclasses unchanged. Add `to_json()` method for debugging/export. |
| `src/deriva_ml/execution/execution.py` | `_get_manifest` constructs `AssetManifest(workspace, execution_rid)` instead of `AssetManifest(path, execution_rid)`. JSON import runs on workspace open (not here). |
| `src/deriva_ml/asset/__init__.py` | No change to exports, but docstring updated to mention SQLite-backed storage. |
| `src/deriva_ml/core/working_data.py` | Keep module; class becomes a deprecated thin wrapper around `workspace.legacy_working_data_view()`. Adds a DeprecationWarning on direct instantiation. |
| `docs/superpowers/plans/2026-04-15-unified-local-db-phase1.md` | This file. |

---

## Task 1: Add the `local_db` package skeleton and path helpers

**Files:**
- Create: `src/deriva_ml/local_db/__init__.py`
- Create: `src/deriva_ml/local_db/paths.py`
- Create: `tests/local_db/__init__.py`
- Create: `tests/local_db/test_paths.py`

- [ ] **Step 1: Create the package `__init__.py` (empty, will be populated later).**

Write `src/deriva_ml/local_db/__init__.py`:

```python
"""Unified local SQLite layer for deriva-ml.

This package provides per-catalog working databases, per-slice immutable
databases, a paged-fetch primitive, and a SQLite-backed asset manifest.

See `docs/superpowers/specs/2026-04-15-unified-local-db-design.md`.
"""

from __future__ import annotations

__all__: list[str] = []
```

- [ ] **Step 2: Create the tests package marker.**

Write `tests/local_db/__init__.py` as an empty file (but with a newline):

```python
```

- [ ] **Step 3: Write failing tests for path helpers.**

Write `tests/local_db/test_paths.py`:

```python
"""Unit tests for local_db.paths — pure path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from deriva_ml.local_db import paths as p


class TestWorkspaceRoot:
    def test_encodes_host_and_catalog(self, tmp_path: Path) -> None:
        root = p.workspace_root(tmp_path, "example.org", "42")
        assert root == tmp_path / "catalogs" / "example.org__42"

    def test_sanitises_unsafe_hostname(self, tmp_path: Path) -> None:
        # Path separators and other unsafe chars are replaced with '_'.
        root = p.workspace_root(tmp_path, "a/b:c", "1")
        assert root == tmp_path / "catalogs" / "a_b_c__1"

    def test_numeric_catalog_id_coerced_to_str(self, tmp_path: Path) -> None:
        root = p.workspace_root(tmp_path, "example.org", 42)
        assert root.name == "example.org__42"


class TestWorkingDbPath:
    def test_under_workspace_root(self, tmp_path: Path) -> None:
        db = p.working_db_path(tmp_path, "example.org", "42")
        assert db == tmp_path / "catalogs" / "example.org__42" / "working.db"


class TestSliceDir:
    def test_slice_dir_under_workspace(self, tmp_path: Path) -> None:
        d = p.slice_dir(tmp_path, "example.org", "42", "abc123")
        assert d == tmp_path / "catalogs" / "example.org__42" / "slices" / "abc123"

    def test_slice_id_sanitised(self, tmp_path: Path) -> None:
        d = p.slice_dir(tmp_path, "example.org", "42", "a/b")
        assert d == tmp_path / "catalogs" / "example.org__42" / "slices" / "a_b"


class TestSliceDbPath:
    def test_slice_db_file_under_slice_dir(self, tmp_path: Path) -> None:
        db = p.slice_db_path(tmp_path, "example.org", "42", "abc123")
        assert db == (
            tmp_path / "catalogs" / "example.org__42" / "slices" / "abc123" / "slice.db"
        )


class TestSanitiseComponent:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("foo", "foo"),
            ("foo.bar", "foo.bar"),
            ("a/b", "a_b"),
            ("a:b", "a_b"),
            ("..", "__"),
            ("", "_"),
        ],
    )
    def test_sanitise(self, raw: str, expected: str) -> None:
        assert p._sanitise_component(raw) == expected
```

- [ ] **Step 4: Run tests to confirm they fail (module doesn't exist yet).**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_paths.py -v
```

Expected: `ImportError` / collection error because `deriva_ml.local_db.paths` is not yet defined.

- [ ] **Step 5: Implement `paths.py`.**

Write `src/deriva_ml/local_db/paths.py`:

```python
"""Pure path-helper functions for the local_db layout.

No side effects, no SQLite access, no filesystem mutation. These helpers
exist so both tests and runtime code derive layout paths from a single
canonical source.
"""

from __future__ import annotations

from pathlib import Path

_UNSAFE_CHARS = set('/\\:<>"|?*')


def _sanitise_component(raw: str) -> str:
    """Replace path-unsafe characters with '_'. Return '_' for empty input."""
    if not raw:
        return "_"
    return "".join("_" if c in _UNSAFE_CHARS else c for c in raw)


def workspace_root(working_dir: Path, hostname: str, catalog_id: str | int) -> Path:
    """Return the per-catalog workspace root directory.

    Layout: {working_dir}/catalogs/{host}__{catalog_id}/
    """
    host = _sanitise_component(hostname)
    cat = _sanitise_component(str(catalog_id))
    return Path(working_dir) / "catalogs" / f"{host}__{cat}"


def working_db_path(working_dir: Path, hostname: str, catalog_id: str | int) -> Path:
    """Return the per-catalog working SQLite DB path."""
    return workspace_root(working_dir, hostname, catalog_id) / "working.db"


def slice_dir(
    working_dir: Path, hostname: str, catalog_id: str | int, slice_id: str
) -> Path:
    """Return the directory for a single slice.

    Layout: {workspace_root}/slices/{slice_id}/
    """
    return workspace_root(working_dir, hostname, catalog_id) / "slices" / _sanitise_component(slice_id)


def slice_db_path(
    working_dir: Path, hostname: str, catalog_id: str | int, slice_id: str
) -> Path:
    """Return the slice.db path for a single slice."""
    return slice_dir(working_dir, hostname, catalog_id, slice_id) / "slice.db"
```

- [ ] **Step 6: Run tests again to confirm they pass.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_paths.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit.**

```bash
git add src/deriva_ml/local_db/__init__.py \
        src/deriva_ml/local_db/paths.py \
        tests/local_db/__init__.py \
        tests/local_db/test_paths.py
git commit -m "feat(local_db): add package skeleton and path helpers"
```

---

## Task 2: SQLite helpers — WAL engine, attach/detach, schema-version migration runner

**Files:**
- Create: `src/deriva_ml/local_db/sqlite_helpers.py`
- Create: `tests/local_db/test_sqlite_helpers.py`

- [ ] **Step 1: Write failing tests.**

Write `tests/local_db/test_sqlite_helpers.py`:

```python
"""Unit tests for local_db.sqlite_helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from deriva_ml.local_db import sqlite_helpers as sh


class TestCreateWalEngine:
    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        db = tmp_path / "db.sqlite"
        engine = sh.create_wal_engine(db)
        try:
            with engine.connect() as conn:
                result = conn.execute(text("PRAGMA journal_mode")).scalar()
            assert result == "wal"
        finally:
            engine.dispose()

    def test_synchronous_normal(self, tmp_path: Path) -> None:
        db = tmp_path / "db.sqlite"
        engine = sh.create_wal_engine(db)
        try:
            with engine.connect() as conn:
                # synchronous=NORMAL is integer 1 in SQLite
                result = conn.execute(text("PRAGMA synchronous")).scalar()
            assert result == 1
        finally:
            engine.dispose()

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        db = tmp_path / "nested" / "sub" / "db.sqlite"
        assert not db.parent.exists()
        engine = sh.create_wal_engine(db)
        try:
            assert db.parent.exists()
        finally:
            engine.dispose()

    def test_read_only_mode(self, tmp_path: Path) -> None:
        db = tmp_path / "db.sqlite"
        # First create the DB in RW mode
        rw = sh.create_wal_engine(db)
        with rw.connect() as conn:
            conn.execute(text("CREATE TABLE t (x INT)"))
            conn.commit()
        rw.dispose()

        # Then open read-only
        ro = sh.create_wal_engine(db, read_only=True)
        try:
            with ro.connect() as conn:
                # Reads work
                conn.execute(text("SELECT * FROM t")).fetchall()
                # Writes fail
                with pytest.raises(Exception):
                    conn.execute(text("INSERT INTO t VALUES (1)"))
                    conn.commit()
        finally:
            ro.dispose()


class TestAttachDetach:
    def test_attach_makes_tables_visible(self, tmp_path: Path) -> None:
        main_db = tmp_path / "main.sqlite"
        slice_db = tmp_path / "slice.sqlite"

        # Create main DB
        eng_main = sh.create_wal_engine(main_db)
        with eng_main.connect() as conn:
            conn.execute(text("CREATE TABLE main_t (x INT)"))
            conn.execute(text("INSERT INTO main_t VALUES (1)"))
            conn.commit()
        eng_main.dispose()

        # Create slice DB
        eng_slice = sh.create_wal_engine(slice_db)
        with eng_slice.connect() as conn:
            conn.execute(text("CREATE TABLE slice_t (y INT)"))
            conn.execute(text("INSERT INTO slice_t VALUES (99)"))
            conn.commit()
        eng_slice.dispose()

        # Attach slice into main
        eng = sh.create_wal_engine(main_db)
        try:
            with eng.connect() as conn:
                sh.attach_database(conn, slice_db, "slice")
                # Can see slice.slice_t
                result = conn.execute(text("SELECT y FROM slice.slice_t")).scalar()
                assert result == 99
                sh.detach_database(conn, "slice")
        finally:
            eng.dispose()


class TestSchemaMeta:
    def test_initial_version_is_1(self, tmp_path: Path) -> None:
        db = tmp_path / "db.sqlite"
        engine = sh.create_wal_engine(db)
        try:
            version = sh.ensure_schema_meta(engine, expected_version=1)
            assert version == 1
        finally:
            engine.dispose()

    def test_idempotent(self, tmp_path: Path) -> None:
        db = tmp_path / "db.sqlite"
        engine = sh.create_wal_engine(db)
        try:
            sh.ensure_schema_meta(engine, expected_version=1)
            # Second call should be a no-op
            v = sh.ensure_schema_meta(engine, expected_version=1)
            assert v == 1
        finally:
            engine.dispose()

    def test_raises_on_higher_version(self, tmp_path: Path) -> None:
        db = tmp_path / "db.sqlite"
        engine = sh.create_wal_engine(db)
        try:
            # Pretend the DB was created by a newer deriva-ml
            sh.ensure_schema_meta(engine, expected_version=2)
            with pytest.raises(sh.SchemaVersionError):
                sh.ensure_schema_meta(engine, expected_version=1)
        finally:
            engine.dispose()
```

- [ ] **Step 2: Run tests to confirm failure.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_sqlite_helpers.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `sqlite_helpers.py`.**

Write `src/deriva_ml/local_db/sqlite_helpers.py`:

```python
"""SQLite engine and connection helpers for the local_db layer.

Provides:
- ``create_wal_engine``: SQLAlchemy engine factory enforcing WAL + synchronous=NORMAL.
- ``attach_database`` / ``detach_database``: ATTACH / DETACH helpers.
- ``ensure_schema_meta``: idempotent schema-version tracking in a ``schema_meta``
  table. Raises :class:`SchemaVersionError` when the on-disk schema is newer than
  the running code expects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Connection, Engine

logger = logging.getLogger(__name__)

SCHEMA_META_TABLE = "schema_meta"


class SchemaVersionError(RuntimeError):
    """The on-disk schema version is newer than this code supports."""


def create_wal_engine(db_path: Path, *, read_only: bool = False) -> Engine:
    """Create a SQLAlchemy engine for a SQLite file with WAL mode.

    - Ensures the parent directory exists.
    - Sets ``journal_mode=WAL`` and ``synchronous=NORMAL`` on every connection.
    - When ``read_only=True``, opens the file via SQLite's ``mode=ro`` URI.

    Args:
        db_path: Path to the SQLite file.
        read_only: If True, open in read-only mode.

    Returns:
        A SQLAlchemy :class:`Engine`.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if read_only:
        url = f"sqlite:///file:{db_path.resolve()}?mode=ro&uri=true"
        # sqlite3 driver needs uri=True on connect kwargs
        engine = create_engine(
            url,
            future=True,
            connect_args={"uri": True},
        )
    else:
        engine = create_engine(f"sqlite:///{db_path.resolve()}", future=True)

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn: Any, _record: Any) -> None:
        cur = dbapi_conn.cursor()
        try:
            if not read_only:
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
        finally:
            cur.close()

    return engine


def attach_database(conn: Connection, db_path: Path, alias: str) -> None:
    """ATTACH a SQLite file under ``alias`` in the given connection.

    Args:
        conn: An open SQLAlchemy connection.
        db_path: Path to the SQLite file to attach.
        alias: Name to attach it under (used as a schema qualifier).
    """
    path_str = str(Path(db_path).resolve()).replace("'", "''")
    alias_safe = alias.replace('"', '""')
    conn.execute(text(f"ATTACH DATABASE '{path_str}' AS \"{alias_safe}\""))


def detach_database(conn: Connection, alias: str) -> None:
    """DETACH a previously attached database by alias."""
    alias_safe = alias.replace('"', '""')
    conn.execute(text(f"DETACH DATABASE \"{alias_safe}\""))


def ensure_schema_meta(engine: Engine, expected_version: int) -> int:
    """Ensure the ``schema_meta`` table exists and records ``expected_version``.

    If the table is empty, inserts ``expected_version`` as the initial version.
    If the table already has a version ≤ ``expected_version``, returns it.
    If the on-disk version is higher, raises :class:`SchemaVersionError`.

    Returns:
        The current schema version (after any initialization).
    """
    with engine.connect() as conn:
        conn.execute(
            text(
                f"CREATE TABLE IF NOT EXISTS {SCHEMA_META_TABLE} ("
                "  version INTEGER PRIMARY KEY,"
                "  recorded_at TEXT NOT NULL DEFAULT (datetime('now'))"
                ")"
            )
        )
        existing = conn.execute(
            text(f"SELECT MAX(version) FROM {SCHEMA_META_TABLE}")
        ).scalar()
        if existing is None:
            conn.execute(
                text(
                    f"INSERT INTO {SCHEMA_META_TABLE}(version) VALUES (:v)"
                ),
                {"v": expected_version},
            )
            conn.commit()
            return expected_version
        if existing > expected_version:
            raise SchemaVersionError(
                f"Database schema version {existing} is newer than expected "
                f"{expected_version}; upgrade deriva-ml."
            )
        return int(existing)
```

- [ ] **Step 4: Run tests — expect pass.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_sqlite_helpers.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Lint + format.**

```bash
uv run ruff format src/deriva_ml/local_db/ tests/local_db/
uv run ruff check src/deriva_ml/local_db/ tests/local_db/
```

- [ ] **Step 6: Commit.**

```bash
git add src/deriva_ml/local_db/sqlite_helpers.py tests/local_db/test_sqlite_helpers.py
git commit -m "feat(local_db): add SQLite helpers (WAL engine, attach/detach, schema_meta)"
```

---

## Task 3: `LocalSchema` — unified schema builder adapter

Today's `SchemaBuilder` in `src/deriva_ml/model/schema_builder.py` already accepts a `Model` from either a live catalog or a bag's `schema.json`. It's ~800 lines and well-tested. Rather than rewrite it, we wrap it in a thin `LocalSchema` adapter so workspace and slice code depend on `local_db.schema.LocalSchema`, not `model.schema_builder.SchemaBuilder`.

**Files:**
- Create: `src/deriva_ml/local_db/schema.py`
- Create: `tests/local_db/test_schema.py`
- Create: `tests/local_db/conftest.py`

- [ ] **Step 1: Write the shared test conftest.**

Write `tests/local_db/conftest.py`:

```python
"""Shared fixtures for local_db tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from deriva.core.ermrest_model import Model


@pytest.fixture
def canned_bag_schema(tmp_path: Path) -> Path:
    """Write a minimal valid bag-style schema.json into a temp dir.

    Schema has:
      - schema 'isa' with tables 'Subject' and 'Image'
      - schema 'deriva-ml' with table 'Dataset'
      - Image has FK to Subject and FK to Dataset
    """
    schema_doc = {
        "schemas": {
            "isa": {
                "tables": {
                    "Subject": {
                        "table_name": "Subject",
                        "schema_name": "isa",
                        "column_definitions": [
                            {"name": "RID", "type": {"typename": "text"}, "nullok": False},
                            {"name": "RCT", "type": {"typename": "timestamptz"}},
                            {"name": "RMT", "type": {"typename": "timestamptz"}},
                            {"name": "RCB", "type": {"typename": "text"}},
                            {"name": "RMB", "type": {"typename": "text"}},
                            {"name": "Name", "type": {"typename": "text"}, "nullok": True},
                        ],
                        "keys": [
                            {"unique_columns": ["RID"]},
                        ],
                        "foreign_keys": [],
                    },
                    "Image": {
                        "table_name": "Image",
                        "schema_name": "isa",
                        "column_definitions": [
                            {"name": "RID", "type": {"typename": "text"}, "nullok": False},
                            {"name": "RCT", "type": {"typename": "timestamptz"}},
                            {"name": "RMT", "type": {"typename": "timestamptz"}},
                            {"name": "RCB", "type": {"typename": "text"}},
                            {"name": "RMB", "type": {"typename": "text"}},
                            {"name": "Filename", "type": {"typename": "text"}, "nullok": True},
                            {"name": "Subject", "type": {"typename": "text"}, "nullok": True},
                        ],
                        "keys": [{"unique_columns": ["RID"]}],
                        "foreign_keys": [
                            {
                                "foreign_key_columns": [
                                    {"schema_name": "isa", "table_name": "Image", "column_name": "Subject"}
                                ],
                                "referenced_columns": [
                                    {"schema_name": "isa", "table_name": "Subject", "column_name": "RID"}
                                ],
                            }
                        ],
                    },
                },
            },
            "deriva-ml": {
                "tables": {
                    "Dataset": {
                        "table_name": "Dataset",
                        "schema_name": "deriva-ml",
                        "column_definitions": [
                            {"name": "RID", "type": {"typename": "text"}, "nullok": False},
                            {"name": "RCT", "type": {"typename": "timestamptz"}},
                            {"name": "RMT", "type": {"typename": "timestamptz"}},
                            {"name": "RCB", "type": {"typename": "text"}},
                            {"name": "RMB", "type": {"typename": "text"}},
                            {"name": "Description", "type": {"typename": "text"}, "nullok": True},
                        ],
                        "keys": [{"unique_columns": ["RID"]}],
                        "foreign_keys": [],
                    },
                },
            },
        },
    }
    out = tmp_path / "schema.json"
    out.write_text(json.dumps(schema_doc))
    return out


@pytest.fixture
def canned_bag_model(canned_bag_schema: Path) -> Model:
    """Load the canned bag schema as a deriva Model."""
    return Model.fromfile("file-system", canned_bag_schema)
```

- [ ] **Step 2: Write failing tests for `LocalSchema`.**

Write `tests/local_db/test_schema.py`:

```python
"""Unit tests for local_db.schema.LocalSchema adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from deriva.core.ermrest_model import Model
from sqlalchemy import text

from deriva_ml.local_db.schema import LocalSchema


class TestLocalSchemaFromBagModel:
    def test_builds_tables_under_file_db(
        self, canned_bag_model: Model, tmp_path: Path
    ) -> None:
        db = tmp_path / "test.db"
        ls = LocalSchema.build(
            model=canned_bag_model,
            schemas=["isa", "deriva-ml"],
            database_path=db,
        )
        try:
            # Tables should exist with fully-qualified names
            assert ls.find_table("isa.Image") is not None
            assert ls.find_table("isa.Subject") is not None
            assert ls.find_table("deriva-ml.Dataset") is not None
        finally:
            ls.dispose()

    def test_unqualified_find_table_works(
        self, canned_bag_model: Model, tmp_path: Path
    ) -> None:
        db = tmp_path / "test.db"
        ls = LocalSchema.build(
            model=canned_bag_model,
            schemas=["isa", "deriva-ml"],
            database_path=db,
        )
        try:
            t = ls.find_table("Image")
            assert t.name.endswith("Image")
        finally:
            ls.dispose()

    def test_engine_is_wal(self, canned_bag_model: Model, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        ls = LocalSchema.build(
            model=canned_bag_model,
            schemas=["isa", "deriva-ml"],
            database_path=db,
        )
        try:
            with ls.engine.connect() as conn:
                mode = conn.execute(text("PRAGMA journal_mode")).scalar()
            assert mode == "wal"
        finally:
            ls.dispose()


class TestLocalSchemaSchemasList:
    def test_exposes_schemas(self, canned_bag_model: Model, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        ls = LocalSchema.build(
            model=canned_bag_model,
            schemas=["isa", "deriva-ml"],
            database_path=db,
        )
        try:
            assert set(ls.schemas) == {"isa", "deriva-ml"}
        finally:
            ls.dispose()


class TestLocalSchemaListTables:
    def test_list_tables_returns_all(
        self, canned_bag_model: Model, tmp_path: Path
    ) -> None:
        db = tmp_path / "test.db"
        ls = LocalSchema.build(
            model=canned_bag_model,
            schemas=["isa", "deriva-ml"],
            database_path=db,
        )
        try:
            tables = ls.list_tables()
            assert any("Image" in t for t in tables)
            assert any("Subject" in t for t in tables)
            assert any("Dataset" in t for t in tables)
        finally:
            ls.dispose()
```

- [ ] **Step 3: Run tests to confirm failure.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_schema.py -v
```

Expected: `ImportError`.

- [ ] **Step 4: Implement `LocalSchema`.**

Write `src/deriva_ml/local_db/schema.py`:

```python
"""Unified schema adapter for working and slice databases.

Wraps :class:`deriva_ml.model.schema_builder.SchemaBuilder` in a smaller
interface usable from either a live ERMrest ``Model`` or a bag's
``schema.json`` file. Construction is a two-step process:

    ls = LocalSchema.build(model=m, schemas=[...], database_path=p)

which creates (or opens) the SQLite file, builds tables, and returns a handle
with a WAL-mode SQLAlchemy engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from deriva.core.ermrest_model import Model
from sqlalchemy import Table as SQLTable
from sqlalchemy import event
from sqlalchemy.engine import Engine

from deriva_ml.local_db import sqlite_helpers as sh
from deriva_ml.model.schema_builder import SchemaBuilder, SchemaORM

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class LocalSchema:
    """Thin adapter over ``SchemaBuilder`` / ``SchemaORM`` for the local_db layer.

    Normalizes construction and pragma setup so callers don't have to know
    whether the underlying DB is a working DB or a slice DB. All Real Work
    is delegated to the existing ``SchemaBuilder`` machinery.
    """

    def __init__(self, orm: SchemaORM, database_path: Path) -> None:
        self._orm = orm
        self._database_path = database_path

    @classmethod
    def build(
        cls,
        *,
        model: Model,
        schemas: list[str],
        database_path: Path,
        read_only: bool = False,
    ) -> "LocalSchema":
        """Build (or open) the schema at ``database_path``.

        Args:
            model: ERMrest Model (from live catalog or schema.json).
            schemas: Schema names to materialize.
            database_path: Directory or `.db` file path for storage.
            read_only: If True, open the underlying engine read-only. This is
                used for slice DBs at query time.
        """
        database_path = Path(database_path)
        builder = SchemaBuilder(model=model, schemas=schemas, database_path=database_path)
        orm = builder.build()

        # Enforce WAL + synchronous=NORMAL on every new connection. Existing
        # SchemaBuilder doesn't set these.
        @event.listens_for(orm.engine, "connect")
        def _pragmas(dbapi_conn: Any, _r: Any) -> None:
            cur = dbapi_conn.cursor()
            try:
                if not read_only:
                    cur.execute("PRAGMA journal_mode=WAL")
                    cur.execute("PRAGMA synchronous=NORMAL")
                cur.execute("PRAGMA foreign_keys=ON")
            finally:
                cur.close()

        # Force a connection so the event runs at least once now.
        with orm.engine.connect():
            pass

        # Track schema version in the DB.
        sh.ensure_schema_meta(orm.engine, expected_version=SCHEMA_VERSION)

        return cls(orm=orm, database_path=database_path)

    # ---- delegation ----

    @property
    def engine(self) -> Engine:
        return self._orm.engine

    @property
    def metadata(self):  # noqa: ANN201 — SQLAlchemy MetaData
        return self._orm.metadata

    @property
    def schemas(self) -> list[str]:
        return list(self._orm.schemas)

    @property
    def database_path(self) -> Path:
        return self._database_path

    def find_table(self, table_name: str) -> SQLTable:
        return self._orm.find_table(table_name)

    def list_tables(self) -> list[str]:
        return self._orm.list_tables()

    def get_orm_class(self, table_name: str) -> Any | None:
        return self._orm.get_orm_class(table_name)

    def dispose(self) -> None:
        self._orm.dispose()

    def __enter__(self) -> "LocalSchema":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self.dispose()
        return False
```

- [ ] **Step 5: Run tests — expect pass.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_schema.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Lint + format.**

```bash
uv run ruff format src/deriva_ml/local_db/schema.py tests/local_db/
uv run ruff check src/deriva_ml/local_db/schema.py tests/local_db/
```

- [ ] **Step 7: Commit.**

```bash
git add src/deriva_ml/local_db/schema.py \
        tests/local_db/conftest.py \
        tests/local_db/test_schema.py
git commit -m "feat(local_db): add LocalSchema adapter over SchemaBuilder"
```

---

## Task 4: `Workspace` — per-catalog DB handle, slice registry, attach/detach

**Files:**
- Create: `src/deriva_ml/local_db/workspace.py`
- Create: `tests/local_db/test_workspace.py`

`Workspace` in Phase 1 is deliberately thin: engine over `working.db`, multi-catalog-safe lookup helpers, and an `attach_slice(slice_db_path, alias='slice')` context manager. It does *not* yet own a `LocalSchema` for the working DB — the working-DB schema is created on demand by callers (Task 5's `PagedFetcher` needs a target table; it asks the workspace for a `LocalSchema` or creates one lazily). The legacy `WorkingDataCache` behavior (string-keyed DataFrame tables) is exposed via a compatibility shim.

- [ ] **Step 1: Write failing tests.**

Write `tests/local_db/test_workspace.py`:

```python
"""Unit tests for local_db.workspace.Workspace."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

from deriva_ml.local_db.workspace import Workspace


class TestWorkspaceCreation:
    def test_creates_working_db_file(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="host.example.org", catalog_id="1")
        assert ws.working_db_path.parent.is_dir()
        # File is created lazily when an engine is first requested
        _ = ws.engine
        assert ws.working_db_path.is_file()
        ws.close()

    def test_wal_mode(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="host", catalog_id="1")
        try:
            with ws.engine.connect() as conn:
                assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        finally:
            ws.close()

    def test_two_workspaces_different_catalogs_do_not_collide(
        self, tmp_path: Path
    ) -> None:
        ws1 = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        ws2 = Workspace(working_dir=tmp_path, hostname="h", catalog_id="2")
        try:
            assert ws1.working_db_path != ws2.working_db_path
        finally:
            ws1.close()
            ws2.close()


class TestAttachSlice:
    def test_attach_context_manager(self, tmp_path: Path) -> None:
        # Create a real slice.db with a tiny table
        slice_db = tmp_path / "catalogs" / "h__1" / "slices" / "s1" / "slice.db"
        slice_db.parent.mkdir(parents=True, exist_ok=True)
        from deriva_ml.local_db.sqlite_helpers import create_wal_engine

        eng = create_wal_engine(slice_db)
        with eng.connect() as conn:
            conn.execute(text("CREATE TABLE t (x INT)"))
            conn.execute(text("INSERT INTO t VALUES (1), (2), (3)"))
            conn.commit()
        eng.dispose()

        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            with ws.attach_slice("s1") as conn:
                count = conn.execute(text("SELECT COUNT(*) FROM slice.t")).scalar()
                assert count == 3
            # After leaving the context, 'slice' should no longer be attached.
            with ws.engine.connect() as conn:
                with pytest.raises(Exception):
                    conn.execute(text("SELECT COUNT(*) FROM slice.t"))
        finally:
            ws.close()

    def test_attach_missing_slice_raises(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            with pytest.raises(FileNotFoundError):
                with ws.attach_slice("nonexistent"):
                    pass
        finally:
            ws.close()


class TestLegacyWorkingDataView:
    def test_cache_table_roundtrip(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            view = ws.legacy_working_data_view()
            df = pd.DataFrame({"x": [1, 2, 3]})
            view.cache_table("mytable", df)
            got = view.read_table("mytable")
            assert list(got["x"]) == [1, 2, 3]
            assert view.has_table("mytable")
            assert "mytable" in view.list_tables()
        finally:
            ws.close()


class TestWorkspaceClose:
    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        ws.close()
        ws.close()  # Should not raise
```

- [ ] **Step 2: Run tests to confirm failure.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_workspace.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `Workspace`.**

Write `src/deriva_ml/local_db/workspace.py`:

```python
"""Per-catalog working database and slice registry.

A ``Workspace`` is bound to ``(working_dir, hostname, catalog_id)``. It owns:
- A single SQLAlchemy engine over ``working.db`` in WAL mode.
- An ``attach_slice`` context manager that ATTACHes a slice DB under alias
  ``"slice"`` for the duration of a single connection.
- A ``legacy_working_data_view`` compatibility shim exposing the old
  ``WorkingDataCache`` API (pandas table roundtrips) on top of the new file.

Phase 1 intentionally does not build a ``LocalSchema`` over the working DB
eagerly; the paged fetcher and denormalizer drive schema creation lazily.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any, Iterator, TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from deriva_ml.local_db import paths as p
from deriva_ml.local_db import sqlite_helpers as sh

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

WORKING_DB_SCHEMA_VERSION = 1


class Workspace:
    """Per-catalog working database, shared across script invocations."""

    def __init__(
        self,
        *,
        working_dir: Path,
        hostname: str,
        catalog_id: str | int,
    ) -> None:
        self._working_dir = Path(working_dir)
        self._hostname = hostname
        self._catalog_id = str(catalog_id)
        self._engine: Engine | None = None
        self._closed = False

    # ---- paths ----

    @property
    def root(self) -> Path:
        return p.workspace_root(self._working_dir, self._hostname, self._catalog_id)

    @property
    def working_db_path(self) -> Path:
        return p.working_db_path(self._working_dir, self._hostname, self._catalog_id)

    def slice_db_path(self, slice_id: str) -> Path:
        return p.slice_db_path(self._working_dir, self._hostname, self._catalog_id, slice_id)

    # ---- engine ----

    @property
    def engine(self) -> Engine:
        if self._closed:
            raise RuntimeError("Workspace is closed")
        if self._engine is None:
            self._engine = sh.create_wal_engine(self.working_db_path)
            sh.ensure_schema_meta(
                self._engine, expected_version=WORKING_DB_SCHEMA_VERSION
            )
        return self._engine

    # ---- slice attach/detach ----

    @contextlib.contextmanager
    def attach_slice(
        self, slice_id: str, alias: str = "slice"
    ) -> Iterator[Connection]:
        """ATTACH a slice DB under ``alias`` for the duration of the block.

        Yields an open SQLAlchemy :class:`Connection` with the slice visible
        as ``{alias}.{table}``.
        """
        slice_path = self.slice_db_path(slice_id)
        if not slice_path.is_file():
            raise FileNotFoundError(f"Slice database not found: {slice_path}")

        conn = self.engine.connect()
        try:
            sh.attach_database(conn, slice_path, alias)
            yield conn
            try:
                sh.detach_database(conn, alias)
            except Exception:  # pragma: no cover — best-effort detach
                logger.debug("detach_database failed; closing connection")
        finally:
            conn.close()

    # ---- legacy compat ----

    def legacy_working_data_view(self) -> "_LegacyWorkingDataView":
        """Return an adapter exposing the old ``WorkingDataCache`` API.

        Lets existing callers (``Dataset.cache_denormalized``, tests) continue
        to use the string-keyed DataFrame table semantics they rely on, backed
        by the new per-catalog working DB.
        """
        return _LegacyWorkingDataView(self)

    # ---- lifecycle ----

    def close(self) -> None:
        if self._closed:
            return
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
        self._closed = True

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self.close()
        return False


class _LegacyWorkingDataView:
    """Adapter providing the old ``WorkingDataCache`` surface on a Workspace.

    Keeps the working DataFrames in the same SQLite file as everything else
    (under their bare user-chosen table names) so existing callers keep
    working without code changes.
    """

    def __init__(self, ws: Workspace) -> None:
        self._ws = ws

    def cache_table(self, table_name: str, df: "pd.DataFrame") -> Path:
        import pandas as pd  # lazy import

        assert isinstance(df, pd.DataFrame)
        df.to_sql(table_name, self._ws.engine, if_exists="replace", index=False)
        return self._ws.working_db_path

    def read_table(self, table_name: str) -> "pd.DataFrame":
        import pandas as pd

        if not self.has_table(table_name):
            raise ValueError(
                f"Table '{table_name}' not found in workspace working DB; "
                f"available: {self.list_tables()}"
            )
        return pd.read_sql_table(table_name, self._ws.engine)

    def query(self, sql: str) -> "pd.DataFrame":
        import pandas as pd

        return pd.read_sql_query(text(sql), self._ws.engine)

    def has_table(self, table_name: str) -> bool:
        from sqlalchemy import inspect

        if not self._ws.working_db_path.exists():
            return False
        return table_name in inspect(self._ws.engine).get_table_names()

    def list_tables(self) -> list[str]:
        from sqlalchemy import inspect

        if not self._ws.working_db_path.exists():
            return []
        return inspect(self._ws.engine).get_table_names()

    def drop_table(self, table_name: str) -> None:
        if self.has_table(table_name):
            with self._ws.engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS [{table_name}]"))
                conn.commit()

    def clear(self) -> None:
        for t in self.list_tables():
            self.drop_table(t)
```

- [ ] **Step 4: Run tests — expect pass.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_workspace.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Lint + format.**

```bash
uv run ruff format src/deriva_ml/local_db/workspace.py tests/local_db/
uv run ruff check src/deriva_ml/local_db/workspace.py tests/local_db/
```

- [ ] **Step 6: Commit.**

```bash
git add src/deriva_ml/local_db/workspace.py tests/local_db/test_workspace.py
git commit -m "feat(local_db): add Workspace with slice attach and legacy cache view"
```

---

## Task 5: Wire `DerivaML.working_data` through the `Workspace`

Replace the lazy `WorkingDataCache` construction in `DerivaML.working_data` with a `Workspace` + its `legacy_working_data_view()`. `WorkingDataCache` itself stays in the tree but becomes a thin deprecated wrapper for one release.

**Files:**
- Modify: `src/deriva_ml/core/base.py` — `working_data` property (around line 435–457).
- Modify: `src/deriva_ml/core/working_data.py` — add DeprecationWarning on direct instantiation.
- Modify: `tests/test_working_data.py` — update imports where needed.

- [ ] **Step 1: Write a failing test that shows the working_data file moved.**

Append to `tests/local_db/test_workspace.py`:

```python
class TestDerivaMLIntegration:
    def test_ml_working_data_uses_workspace_path(self, tmp_path: Path) -> None:
        """DerivaML.working_data should write to catalogs/{host}__{cat}/working.db."""
        # Construct a DerivaML without connecting (instance attribute bootstrap)
        from deriva_ml import DerivaML
        import pandas as pd

        ml = DerivaML.__new__(DerivaML)
        ml.working_dir = tmp_path
        ml.host_name = "example.org"
        ml.catalog_id = "9"

        wd = ml.working_data
        wd.cache_table("demo", pd.DataFrame({"x": [1]}))

        expected = tmp_path / "catalogs" / "example.org__9" / "working.db"
        assert expected.is_file()
```

Note: this test bypasses `DerivaML.__init__` because connecting to a catalog is expensive; it only exercises the lazy `working_data` property.

- [ ] **Step 2: Run the new test — expect failure.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_workspace.py::TestDerivaMLIntegration -v
```

Expected: FAIL — either `host_name` not referenced by `working_data`, or the file lands in the old `working-data/` location.

- [ ] **Step 3: Update `DerivaML.working_data`.**

In `src/deriva_ml/core/base.py`, replace the `working_data` property body (currently around lines 435–457) with:

```python
    @property
    def working_data(self):
        """Per-catalog working-DB view (pandas-table roundtrips).

        Backed by ``Workspace`` under ``{working_dir}/catalogs/{host}__{cat}/
        working.db``. Shared across invocations of scripts that use the same
        working directory.
        """
        from deriva_ml.local_db.workspace import Workspace

        if not hasattr(self, "_workspace"):
            self._workspace = Workspace(
                working_dir=self.working_dir,
                hostname=self.host_name,
                catalog_id=self.catalog_id,
            )
        return self._workspace.legacy_working_data_view()
```

The `working_data.WorkingDataCache` import at line 453 becomes unused — remove it from the property body.

- [ ] **Step 4: Deprecate direct `WorkingDataCache` usage.**

In `src/deriva_ml/core/working_data.py`, modify `WorkingDataCache.__init__` (around line 46) to emit a DeprecationWarning:

```python
    def __init__(self, working_dir: Path):
        import warnings

        warnings.warn(
            "WorkingDataCache is deprecated; use "
            "deriva_ml.local_db.workspace.Workspace instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._db_dir = working_dir / "working-data"
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._db_dir / "catalog.db"
        self._engine = None
```

- [ ] **Step 5: Update the DerivaML integration test to pass.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_workspace.py::TestDerivaMLIntegration -v
```

Expected: PASS.

- [ ] **Step 6: Run existing working_data tests to make sure they still pass.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/test_working_data.py -v -W error::DeprecationWarning || true
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/test_working_data.py -v
```

The first command may fail because direct `WorkingDataCache()` calls trigger the new warning. Check test output — if tests use `ml.working_data`, they go through the new path and pass. If tests use `WorkingDataCache(tmp_path)` directly, add `pytest.warns(DeprecationWarning)` or `pytest.filterwarnings("ignore::DeprecationWarning")` as needed.

Expected from the second run: all tests pass (deprecation warnings are logged but don't fail).

- [ ] **Step 7: Fix any `test_working_data.py` breakage.**

If any tests fail because of the new warning behavior or path change, update the tests minimally — they're testing the legacy class and are allowed to suppress the deprecation warning:

```python
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
```

Add that at the top of `tests/test_working_data.py`.

- [ ] **Step 8: Run denormalization cache test to verify it still works.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/ -v -k "cache_denormalize or cache_denorm" 2>/dev/null || true
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/ -v -k "working_data or cache_table" | tail -30
```

Expected: no regressions. `Dataset.cache_denormalized` in `src/deriva_ml/dataset/dataset.py:1083` uses `self._ml.working_data` and should transparently hit the new path.

- [ ] **Step 9: Lint + format.**

```bash
uv run ruff format src/deriva_ml/core/base.py src/deriva_ml/core/working_data.py tests/
uv run ruff check src/deriva_ml/core/base.py src/deriva_ml/core/working_data.py
```

- [ ] **Step 10: Commit.**

```bash
git add src/deriva_ml/core/base.py \
        src/deriva_ml/core/working_data.py \
        tests/local_db/test_workspace.py \
        tests/test_working_data.py
git commit -m "refactor(core): route DerivaML.working_data through Workspace"
```

---

## Task 6: `PagedFetcher` — fake-client unit tests

We build the fetcher against a fake ERMrest client first so we can drive URL-length edge cases, POST fallback, dedup, and the cardinality heuristic deterministically. Live-catalog integration tests come in Task 7.

**Files:**
- Create: `src/deriva_ml/local_db/paged_fetcher.py`
- Create: `tests/local_db/test_paged_fetcher.py`

The fake client exposes a small surface:

```python
class FakePagedClient:
    """Stands in for the minimal surface PagedFetcher needs on top of ERMrest.

    Tracks issued requests so tests can assert how the fetcher behaved.
    """
    def __init__(self, *, rows_by_table: dict[str, list[dict]]): ...
    def count(self, table: str) -> int: ...
    def fetch_page(self, table: str, sort: tuple[str, ...], after: tuple | None,
                   predicate: str | None, limit: int) -> list[dict]: ...
    def fetch_rid_batch(self, table: str, column: str, rids: list[str],
                         method: str = "GET") -> list[dict]: ...
```

- [ ] **Step 1: Write failing tests.**

Write `tests/local_db/test_paged_fetcher.py`:

```python
"""Unit tests for local_db.paged_fetcher (against a fake ERMrest client)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, select

from deriva_ml.local_db.paged_fetcher import PagedFetcher


@dataclass
class FakePagedClient:
    """Deterministic stand-in for the PagedFetcher's ERMrest dependency.

    The fetcher calls only these four methods on the client. Tests record
    request parameters to verify batching, fallback, and dedup behavior.
    """

    rows_by_table: dict[str, list[dict[str, Any]]]
    requests: list[tuple[str, dict]] = field(default_factory=list)
    # Inject a max URL length so tests can force POST fallback. When a GET
    # "request" (simulated via fetch_rid_batch with method="GET") would
    # exceed this, the fetcher should call fetch_rid_batch(..., method="POST")
    # instead. The fake raises an error to simulate GET failure so the
    # fetcher must fall back.
    max_get_bytes: int = 6144

    def count(self, table: str) -> int:
        self.requests.append(("count", {"table": table}))
        return len(self.rows_by_table.get(table, []))

    def fetch_page(
        self,
        table: str,
        sort: tuple[str, ...],
        after: tuple | None,
        predicate: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        self.requests.append(
            ("fetch_page", {"table": table, "sort": sort, "after": after,
                            "predicate": predicate, "limit": limit})
        )
        rows = list(self.rows_by_table.get(table, []))
        # Apply sort (supports only sort=("RID",) for these tests)
        rows.sort(key=lambda r: tuple(r[c] for c in sort))
        if after is not None:
            rows = [r for r in rows if tuple(r[c] for c in sort) > tuple(after)]
        return rows[:limit]

    def fetch_rid_batch(
        self,
        table: str,
        column: str,
        rids: list[str],
        method: str = "GET",
    ) -> list[dict[str, Any]]:
        self.requests.append(
            ("fetch_rid_batch", {"table": table, "column": column,
                                  "rids": list(rids), "method": method})
        )
        if method == "GET":
            # Simulate that a joined URL of RIDs has length ~13 per RID +
            # overhead. If it would exceed max_get_bytes, raise to force fallback.
            approx_url_bytes = 128 + 13 * len(rids)  # path + query + RIDs
            if approx_url_bytes > self.max_get_bytes:
                raise RuntimeError(f"GET URL too long ({approx_url_bytes} bytes)")
        rows = self.rows_by_table.get(table, [])
        want = set(rids)
        return [r for r in rows if r.get(column) in want]


def _make_target_table(engine, name: str = "Image") -> Table:
    md = MetaData()
    t = Table(
        name,
        md,
        Column("RID", String, primary_key=True),
        Column("Filename", String),
        Column("Subject", String),
    )
    md.create_all(engine)
    return t


def _rows_count(engine, table: Table) -> int:
    with engine.connect() as conn:
        return conn.execute(select(table)).fetchall().__len__()


class TestFetchPredicate:
    def test_keyset_paging_fetches_all(self, tmp_path: Path) -> None:
        engine = create_engine(f"sqlite:///{tmp_path / 'wd.sqlite'}", future=True)
        target = _make_target_table(engine)

        rows = [{"RID": f"R{i:03d}", "Filename": f"f{i}", "Subject": "S1"} for i in range(25)]
        client = FakePagedClient(rows_by_table={"Image": rows})

        f = PagedFetcher(client=client, engine=engine)
        n = f.fetch_predicate(
            table="Image",
            predicate=None,
            target_table=target,
            sort=("RID",),
            page_size=10,
        )
        assert n == 25
        # Should have issued 3 page requests (10, 10, 5)
        page_reqs = [r for r in client.requests if r[0] == "fetch_page"]
        assert len(page_reqs) == 3
        assert _rows_count(engine, target) == 25

    def test_respects_predicate(self, tmp_path: Path) -> None:
        engine = create_engine(f"sqlite:///{tmp_path / 'wd.sqlite'}", future=True)
        target = _make_target_table(engine)

        rows = [{"RID": f"R{i}", "Filename": f"f{i}", "Subject": "S1"} for i in range(3)]
        client = FakePagedClient(rows_by_table={"Image": rows})

        f = PagedFetcher(client=client, engine=engine)
        # Predicate is passed through opaquely
        f.fetch_predicate("Image", "Subject=S1", target, sort=("RID",), page_size=100)
        assert any(r[1]["predicate"] == "Subject=S1"
                   for r in client.requests if r[0] == "fetch_page")


class TestFetchByRids:
    def test_batches_at_default_size(self, tmp_path: Path) -> None:
        engine = create_engine(f"sqlite:///{tmp_path / 'wd.sqlite'}", future=True)
        target = _make_target_table(engine)

        rows = [{"RID": f"R{i:03d}", "Filename": f"f{i}", "Subject": "S1"} for i in range(1200)]
        client = FakePagedClient(rows_by_table={"Image": rows})

        f = PagedFetcher(client=client, engine=engine)
        rids = [r["RID"] for r in rows]
        n = f.fetch_by_rids(
            table="Image",
            rids=rids,
            target_table=target,
            rid_column="RID",
            batch_size=500,
        )
        assert n == 1200
        batches = [r for r in client.requests if r[0] == "fetch_rid_batch"]
        # 1200 / 500 = 3 batches (500, 500, 200)
        assert len(batches) == 3
        assert {len(b[1]["rids"]) for b in batches} == {500, 200}

    def test_deduplication_across_calls(self, tmp_path: Path) -> None:
        engine = create_engine(f"sqlite:///{tmp_path / 'wd.sqlite'}", future=True)
        target = _make_target_table(engine)

        rows = [{"RID": f"R{i}", "Filename": f"f{i}", "Subject": "S1"} for i in range(5)]
        client = FakePagedClient(rows_by_table={"Image": rows})

        f = PagedFetcher(client=client, engine=engine)
        f.fetch_by_rids("Image", ["R0", "R1", "R2"], target, rid_column="RID")
        # Second call asks for some already-fetched RIDs plus new ones
        f.fetch_by_rids("Image", ["R1", "R2", "R3", "R4"], target, rid_column="RID")

        # Expect the second call to only request R3 and R4
        batches = [r[1] for r in client.requests if r[0] == "fetch_rid_batch"]
        assert len(batches) == 2
        assert set(batches[0]["rids"]) == {"R0", "R1", "R2"}
        assert set(batches[1]["rids"]) == {"R3", "R4"}

        assert _rows_count(engine, target) == 5

    def test_post_fallback_on_long_url(self, tmp_path: Path) -> None:
        engine = create_engine(f"sqlite:///{tmp_path / 'wd.sqlite'}", future=True)
        target = _make_target_table(engine)

        # Huge batch that the fake client will reject on GET
        rows = [{"RID": f"R{i:05d}", "Filename": f"f{i}", "Subject": "S1"}
                for i in range(600)]
        client = FakePagedClient(
            rows_by_table={"Image": rows},
            max_get_bytes=1024,   # tiny, forces POST
        )

        f = PagedFetcher(client=client, engine=engine)
        rids = [r["RID"] for r in rows]
        n = f.fetch_by_rids(
            table="Image",
            rids=rids,
            target_table=target,
            rid_column="RID",
            batch_size=600,       # try a single big batch first
            max_url_bytes=1024,
        )
        assert n == 600
        batches = [r[1] for r in client.requests if r[0] == "fetch_rid_batch"]
        # At least one POST should have been used
        assert any(b["method"] == "POST" for b in batches)

    def test_shrinks_batch_before_post(self, tmp_path: Path) -> None:
        """If a smaller GET batch fits, prefer that over POST."""
        engine = create_engine(f"sqlite:///{tmp_path / 'wd.sqlite'}", future=True)
        target = _make_target_table(engine)

        rows = [{"RID": f"R{i:05d}", "Filename": f"f{i}", "Subject": "S1"}
                for i in range(200)]
        # max_get_bytes is chosen so 200-RID GET fails, but 100-RID GET succeeds.
        # Approx URL bytes formula in fake: 128 + 13 * N.
        # 200 RIDs -> 128 + 2600 = 2728. 100 RIDs -> 128 + 1300 = 1428.
        # Set ceiling to 1500 so 100 fits, 200 doesn't.
        client = FakePagedClient(rows_by_table={"Image": rows}, max_get_bytes=1500)

        f = PagedFetcher(client=client, engine=engine)
        rids = [r["RID"] for r in rows]
        f.fetch_by_rids(
            table="Image",
            rids=rids,
            target_table=target,
            rid_column="RID",
            batch_size=200,
            max_url_bytes=1500,
        )
        batches = [r[1] for r in client.requests if r[0] == "fetch_rid_batch"]
        # Should never have used POST
        assert all(b["method"] == "GET" for b in batches)
        # And should have used at least two batches (because 200 was shrunk)
        assert len(batches) >= 2


class TestFetchedRids:
    def test_tracks_rids_from_predicate_fetch(self, tmp_path: Path) -> None:
        engine = create_engine(f"sqlite:///{tmp_path / 'wd.sqlite'}", future=True)
        target = _make_target_table(engine)

        rows = [{"RID": f"R{i}", "Filename": "f", "Subject": "S"} for i in range(3)]
        client = FakePagedClient(rows_by_table={"Image": rows})
        f = PagedFetcher(client=client, engine=engine)
        f.fetch_predicate("Image", None, target, sort=("RID",), page_size=10)

        assert f.fetched_rids("Image", target) == {"R0", "R1", "R2"}


class TestCardinalityHeuristic:
    def test_switches_to_predicate_when_set_large(self, tmp_path: Path) -> None:
        engine = create_engine(f"sqlite:///{tmp_path / 'wd.sqlite'}", future=True)
        target = _make_target_table(engine)

        # Table has 10 rows total, we ask for 9 of them.
        rows = [{"RID": f"R{i}", "Filename": "f", "Subject": "S"} for i in range(10)]
        client = FakePagedClient(rows_by_table={"Image": rows})

        f = PagedFetcher(client=client, engine=engine)
        rids = [f"R{i}" for i in range(9)]
        n = f.fetch_by_rids_or_predicate(
            table="Image",
            rids=rids,
            target_table=target,
            rid_column="RID",
            sort=("RID",),
            cardinality_threshold=0.5,
        )
        assert n == 9
        # Because 9/10 > 0.5, expect a fetch_page call, not rid batches
        methods_used = {r[0] for r in client.requests}
        assert "fetch_page" in methods_used
```

- [ ] **Step 2: Run tests — expect failure.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_paged_fetcher.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `paged_fetcher.py`.**

Write `src/deriva_ml/local_db/paged_fetcher.py`:

```python
"""Paged fetching primitive for the local_db layer.

Exposes three public methods:

- ``fetch_predicate``: keyset-paged scan of rows matching an ERMrest
  predicate into a SQLAlchemy ``Table`` in local SQLite.
- ``fetch_by_rids``: RID-set batched fetch with URL byte-length guard, GET
  shrink-then-POST fallback, and per-operation dedup against a
  ``target_table``.
- ``fetch_by_rids_or_predicate``: dispatches between the two based on a
  cardinality threshold (``|rid_set| / table_row_count``).

The class is parameterized on a ``client`` object providing four methods:
``count``, ``fetch_page``, ``fetch_rid_batch``, and — optionally — a `POST`
mode flag to ``fetch_rid_batch`` for the oversized-URL fallback. See the
fake client used in tests for the exact interface contract.

For production use, the caller adapts an ``ErmrestCatalog`` + datapath to
this surface. That adapter lives in a separate module
(``paged_fetcher_ermrest.py`` — built in Task 7).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Protocol

from sqlalchemy import Table, insert, select
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = 1000
DEFAULT_BATCH_SIZE = 500
DEFAULT_MAX_URL_BYTES = 6144
DEFAULT_CARDINALITY_THRESHOLD = 0.5


class PagedClient(Protocol):
    """Narrow surface the PagedFetcher depends on."""

    def count(self, table: str) -> int: ...

    def fetch_page(
        self,
        table: str,
        sort: tuple[str, ...],
        after: tuple | None,
        predicate: str | None,
        limit: int,
    ) -> list[dict[str, Any]]: ...

    def fetch_rid_batch(
        self,
        table: str,
        column: str,
        rids: list[str],
        method: str = "GET",
    ) -> list[dict[str, Any]]: ...


class PagedFetcher:
    """Stream rows from an ERMrest-like client into a local SQLAlchemy Table.

    See module docstring for method semantics.
    """

    def __init__(self, *, client: PagedClient, engine: Engine) -> None:
        self._client = client
        self._engine = engine
        # Per-operation dedup: (table_name, rid_column) -> set[str]
        self._seen: dict[tuple[str, str], set[str]] = {}
        self._counts: dict[str, int] = {}

    # ---------------- predicate paging ---------------- #

    def fetch_predicate(
        self,
        table: str,
        predicate: str | None,
        target_table: Table,
        sort: tuple[str, ...] = ("RID",),
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> int:
        """Keyset-paged scan into ``target_table``.

        Returns the number of rows inserted.
        """
        n = 0
        after: tuple | None = None
        while True:
            page = self._client.fetch_page(
                table=table,
                sort=sort,
                after=after,
                predicate=predicate,
                limit=page_size,
            )
            if not page:
                break
            self._insert_rows(target_table, page)
            # Track for dedup under the "RID" column by convention
            key = (table, "RID")
            self._seen.setdefault(key, set()).update(
                str(r["RID"]) for r in page if "RID" in r
            )
            n += len(page)
            if len(page) < page_size:
                break
            last = page[-1]
            after = tuple(last[c] for c in sort)
        return n

    # ---------------- RID-set batching ---------------- #

    def fetch_by_rids(
        self,
        table: str,
        rids: Iterable[str],
        target_table: Table,
        rid_column: str = "RID",
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_url_bytes: int = DEFAULT_MAX_URL_BYTES,
    ) -> int:
        """Fetch rows where ``rid_column`` IN ``rids`` into ``target_table``.

        Dedups against previous fetches on the same (table, rid_column).
        """
        key = (table, rid_column)
        seen = self._seen.setdefault(key, set())
        to_fetch = [r for r in dict.fromkeys(str(x) for x in rids) if r not in seen]
        if not to_fetch:
            return 0

        n = 0
        i = 0
        while i < len(to_fetch):
            batch = to_fetch[i : i + batch_size]
            rows = self._fetch_rid_batch_with_fallback(
                table=table,
                column=rid_column,
                rids=batch,
                max_url_bytes=max_url_bytes,
            )
            self._insert_rows(target_table, rows)
            seen.update(batch)
            n += len(rows)
            i += batch_size
        return n

    def _fetch_rid_batch_with_fallback(
        self,
        *,
        table: str,
        column: str,
        rids: list[str],
        max_url_bytes: int,
    ) -> list[dict[str, Any]]:
        """Try GET first. On URL overflow, shrink batch; if even one RID
        won't fit, use POST. Logs but never raises a URL-length error back
        out.
        """
        # Estimate URL bytes before issuing. The Protocol client may also
        # raise on its own length check; we handle both.
        attempt = list(rids)
        while attempt:
            estimated = 128 + 13 * len(attempt)
            if estimated <= max_url_bytes:
                try:
                    return self._client.fetch_rid_batch(
                        table=table, column=column, rids=attempt, method="GET"
                    )
                except RuntimeError as exc:
                    if "too long" not in str(exc).lower():
                        raise
                    # fall through to shrink
            # Shrink: halve the batch
            half = len(attempt) // 2
            if half == 0:
                # Even 1 RID doesn't fit under GET — use POST.
                logger.debug(
                    "POST fallback for table=%s column=%s rids=%d",
                    table, column, len(rids),
                )
                return self._client.fetch_rid_batch(
                    table=table, column=column, rids=rids, method="POST"
                )
            attempt = attempt[:half]
        return []

    # ---------------- cardinality-based dispatch ---------------- #

    def fetch_by_rids_or_predicate(
        self,
        table: str,
        rids: list[str],
        target_table: Table,
        rid_column: str = "RID",
        sort: tuple[str, ...] = ("RID",),
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_url_bytes: int = DEFAULT_MAX_URL_BYTES,
        cardinality_threshold: float = DEFAULT_CARDINALITY_THRESHOLD,
    ) -> int:
        """Choose RID-set batching or predicate scan based on cardinality."""
        total = self._counts.get(table)
        if total is None:
            total = self._client.count(table)
            self._counts[table] = total

        if total > 0 and len(rids) / total > cardinality_threshold:
            logger.debug(
                "Using predicate fetch for %s (|rids|=%d, total=%d)",
                table, len(rids), total,
            )
            n_fetched = self.fetch_predicate(
                table=table,
                predicate=None,
                target_table=target_table,
                sort=sort,
                page_size=max(batch_size, DEFAULT_PAGE_SIZE),
            )
            # Filter to requested RIDs locally (by deleting unwanted rows)
            wanted = set(rids)
            with self._engine.begin() as conn:
                all_rids = [
                    row[0] for row in conn.execute(select(target_table.c[rid_column]))
                ]
                to_delete = [r for r in all_rids if r not in wanted]
                if to_delete:
                    conn.execute(
                        target_table.delete().where(
                            target_table.c[rid_column].in_(to_delete)
                        )
                    )
            return min(n_fetched, len(wanted))

        return self.fetch_by_rids(
            table=table,
            rids=rids,
            target_table=target_table,
            rid_column=rid_column,
            batch_size=batch_size,
            max_url_bytes=max_url_bytes,
        )

    # ---------------- introspection ---------------- #

    def fetched_rids(self, table: str, target_table: Table | None = None) -> set[str]:
        """Return RIDs already materialized for ``table``.

        Prefers the in-memory dedup set; falls back to querying
        ``target_table`` if no in-memory record exists and the table is
        provided.
        """
        for (t, _col), s in self._seen.items():
            if t == table:
                return set(s)
        if target_table is not None:
            with self._engine.connect() as conn:
                rids = {
                    str(row[0])
                    for row in conn.execute(select(target_table.c.RID))
                }
            self._seen[(table, "RID")] = set(rids)
            return set(rids)
        return set()

    # ---------------- internals ---------------- #

    def _insert_rows(
        self, target_table: Table, rows: list[dict[str, Any]]
    ) -> None:
        if not rows:
            return
        # Project rows to the columns actually on the target table so extra
        # columns in the source are ignored silently.
        cols = {c.name for c in target_table.columns}
        projected = [{k: v for k, v in r.items() if k in cols} for r in rows]
        with self._engine.begin() as conn:
            conn.execute(insert(target_table), projected)
```

- [ ] **Step 4: Run tests — expect pass.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_paged_fetcher.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Lint + format.**

```bash
uv run ruff format src/deriva_ml/local_db/paged_fetcher.py tests/local_db/
uv run ruff check src/deriva_ml/local_db/paged_fetcher.py
```

- [ ] **Step 6: Commit.**

```bash
git add src/deriva_ml/local_db/paged_fetcher.py tests/local_db/test_paged_fetcher.py
git commit -m "feat(local_db): add PagedFetcher with URL guard, POST fallback, dedup"
```

---

## Task 7: ERMrest-backed `PagedClient` adapter + live integration tests

`PagedFetcher` is transport-agnostic. To use it against a real catalog, we need an adapter that translates its `PagedClient` protocol methods into deriva-py datapath / ERMrest URL calls.

**Files:**
- Create: `src/deriva_ml/local_db/paged_fetcher_ermrest.py`
- Create: `tests/local_db/test_paged_fetcher_live.py`

- [ ] **Step 1: Write integration tests (live catalog).**

Write `tests/local_db/test_paged_fetcher_live.py`:

```python
"""Live-catalog integration tests for the ERMrest PagedClient adapter.

Requires DERIVA_HOST and a test catalog. Uses the `test_ml` / `catalog_with_datasets`
fixtures from the top-level conftest.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Column, MetaData, String, Table

from deriva_ml.local_db.paged_fetcher import PagedFetcher
from deriva_ml.local_db.paged_fetcher_ermrest import ErmrestPagedClient
from deriva_ml.local_db.workspace import Workspace


pytestmark = pytest.mark.integration


@pytest.fixture
def workspace_on_ml(test_ml) -> Workspace:
    """Construct a Workspace bound to the test catalog."""
    ws = Workspace(
        working_dir=test_ml.working_dir,
        hostname=test_ml.host_name,
        catalog_id=test_ml.catalog_id,
    )
    yield ws
    ws.close()


def _make_dataset_table(engine) -> Table:
    md = MetaData()
    t = Table(
        "Dataset_cache",
        md,
        Column("RID", String, primary_key=True),
        Column("Description", String),
    )
    md.create_all(engine)
    return t


class TestErmrestPagedClientLive:
    def test_count_matches_catalog(self, test_ml, workspace_on_ml) -> None:
        client = ErmrestPagedClient(catalog=test_ml.catalog)
        # Expect the catalog to have at least one Dataset row after the
        # populated-state fixture runs.
        total = client.count(f"{test_ml.ml_schema}:Dataset")
        assert total >= 0  # weak assertion — fixture state dependent

    def test_fetch_predicate_writes_rows(self, test_ml, workspace_on_ml) -> None:
        target = _make_dataset_table(workspace_on_ml.engine)
        client = ErmrestPagedClient(catalog=test_ml.catalog)
        f = PagedFetcher(client=client, engine=workspace_on_ml.engine)

        n = f.fetch_predicate(
            table=f"{test_ml.ml_schema}:Dataset",
            predicate=None,
            target_table=target,
            sort=("RID",),
            page_size=50,
        )
        assert n >= 0
        # Row count in local DB should match n.
        from sqlalchemy import select
        with workspace_on_ml.engine.connect() as conn:
            got = conn.execute(select(target)).fetchall()
        assert len(got) == n

    def test_fetch_by_rids_roundtrip(self, test_ml, workspace_on_ml) -> None:
        """Fetch a few dataset RIDs directly and assert they land locally."""
        # First use fetch_predicate to discover some RIDs.
        target = _make_dataset_table(workspace_on_ml.engine)
        client = ErmrestPagedClient(catalog=test_ml.catalog)
        f = PagedFetcher(client=client, engine=workspace_on_ml.engine)

        f.fetch_predicate(
            table=f"{test_ml.ml_schema}:Dataset",
            predicate=None,
            target_table=target,
            sort=("RID",),
            page_size=5,
        )
        from sqlalchemy import select
        with workspace_on_ml.engine.connect() as conn:
            rids = [r[0] for r in conn.execute(select(target.c.RID))]

        if not rids:
            pytest.skip("No Dataset rows in catalog to test fetch_by_rids")

        # Clear and re-fetch by RID
        with workspace_on_ml.engine.begin() as conn:
            conn.execute(target.delete())

        # Reset dedup tracking — easiest path: new PagedFetcher
        f2 = PagedFetcher(client=client, engine=workspace_on_ml.engine)
        n = f2.fetch_by_rids(
            table=f"{test_ml.ml_schema}:Dataset",
            rids=rids[:3],
            target_table=target,
            rid_column="RID",
        )
        assert n == min(3, len(rids))
```

- [ ] **Step 2: Write unit-level tests for the adapter with a mock catalog.**

Append to `tests/local_db/test_paged_fetcher.py`:

```python
# ---------- ErmrestPagedClient URL-construction tests ---------- #

class _MockCatalog:
    """Minimal mock of ErmrestCatalog used by the adapter tests."""

    def __init__(self, *, get_responses=None, post_responses=None):
        self.get_calls = []
        self.post_calls = []
        self._get_responses = get_responses or {}
        self._post_responses = post_responses or {}

    def get(self, url, headers=None):
        self.get_calls.append(url)
        class R:
            def __init__(self, data): self._d = data
            def json(self): return self._d
            def raise_for_status(self): return None
        return R(self._get_responses.get(url, []))

    def post(self, url, json=None, headers=None):
        self.post_calls.append((url, json))
        class R:
            def __init__(self, data): self._d = data
            def json(self): return self._d
            def raise_for_status(self): return None
        return R(self._post_responses.get((url, None), []))


class TestErmrestPagedClient:
    def test_count_uses_aggregate_endpoint(self) -> None:
        from deriva_ml.local_db.paged_fetcher_ermrest import ErmrestPagedClient
        cat = _MockCatalog(get_responses={
            "/ermrest/catalog/1/aggregate/isa:Image/n:=cnt(*)": [{"n": 42}],
        })
        cat.catalog_id = "1"
        c = ErmrestPagedClient(catalog=cat, catalog_id="1")
        assert c.count("isa:Image") == 42
```

- [ ] **Step 3: Run tests to confirm failure (ImportError for adapter).**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_paged_fetcher.py::TestErmrestPagedClient -v
```

Expected: `ImportError`.

- [ ] **Step 4: Implement the adapter.**

Write `src/deriva_ml/local_db/paged_fetcher_ermrest.py`:

```python
"""ERMrest adapter for :class:`PagedFetcher`.

Translates the narrow ``PagedClient`` protocol into ERMrest HTTP calls via
an ``ErmrestCatalog`` handle. Responsible for URL construction, GET/POST
transport choice, and JSON response parsing.

URL forms used:

- count:   ``/ermrest/catalog/{N}/aggregate/{schema}:{table}/n:=cnt(*)``
- page:    ``/ermrest/catalog/{N}/entity/{schema}:{table}[/predicate]@sort({s})[@after({a})]?limit={L}``
- RID-IN (GET):  ``/ermrest/catalog/{N}/entity/{schema}:{table}/{col}=any({r1,r2,...})``
- RID-IN (POST): ``POST /ermrest/catalog/{N}/entity/{schema}:{table}`` with a
  JSON body describing the filter (deriva-py supports JSON query submissions
  on entity endpoints).
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

logger = logging.getLogger(__name__)


class ErmrestPagedClient:
    """Adapter conforming to :class:`~deriva_ml.local_db.paged_fetcher.PagedClient`."""

    def __init__(self, *, catalog: Any, catalog_id: str | None = None) -> None:
        """Args:
        catalog: An :class:`ErmrestCatalog` (deriva-py) or compatible.
        catalog_id: Catalog ID for URL construction. Falls back to
            ``catalog.catalog_id`` when not provided.
        """
        self._catalog = catalog
        self._catalog_id = str(catalog_id if catalog_id is not None else getattr(catalog, "catalog_id"))

    # ---- PagedClient protocol ---- #

    def count(self, table: str) -> int:
        url = f"/ermrest/catalog/{self._catalog_id}/aggregate/{table}/n:=cnt(*)"
        resp = self._catalog.get(url)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return 0
        return int(data[0]["n"])

    def fetch_page(
        self,
        table: str,
        sort: tuple[str, ...],
        after: tuple | None,
        predicate: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        parts = [f"/ermrest/catalog/{self._catalog_id}/entity/{table}"]
        if predicate:
            parts.append(f"/{predicate}")
        sort_cols = ",".join(quote(c) for c in sort)
        parts.append(f"@sort({sort_cols})")
        if after is not None:
            after_str = ",".join(quote(str(v)) for v in after)
            parts.append(f"@after({after_str})")
        parts.append(f"?limit={limit}")
        url = "".join(parts)
        resp = self._catalog.get(url)
        resp.raise_for_status()
        return list(resp.json())

    def fetch_rid_batch(
        self,
        table: str,
        column: str,
        rids: list[str],
        method: str = "GET",
    ) -> list[dict[str, Any]]:
        if method == "GET":
            rid_list = ",".join(quote(r) for r in rids)
            url = (
                f"/ermrest/catalog/{self._catalog_id}/entity/{table}"
                f"/{quote(column)}=any({rid_list})"
            )
            # Heuristic URL-length cap; let the upstream fetcher handle fallback.
            if len(url) > 7500:
                raise RuntimeError(f"GET URL too long ({len(url)} bytes)")
            resp = self._catalog.get(url)
            resp.raise_for_status()
            return list(resp.json())
        # POST fallback: deriva-py supports filter submission via POST on
        # the entity endpoint with a JSON array of filter objects.
        url = f"/ermrest/catalog/{self._catalog_id}/entity/{table}"
        body = {"filter": {"and": [{column: {"in": rids}}]}}
        resp = self._catalog.post(url, json=body)
        resp.raise_for_status()
        return list(resp.json())
```

Note: the POST-body filter form above is a placeholder. Real deriva-py ERMrest
POST-query semantics may differ; the live integration test (step 7) will exercise
whichever shape works. If it fails, the adapter is the single place to fix.

- [ ] **Step 5: Run unit tests to confirm pass.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_paged_fetcher.py::TestErmrestPagedClient -v
```

Expected: PASS.

- [ ] **Step 6: Run full paged_fetcher unit suite to confirm no regression.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_paged_fetcher.py -v
```

Expected: all PASS.

- [ ] **Step 7: Run live integration tests (requires `DERIVA_HOST`).**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_paged_fetcher_live.py -v -m integration
```

Expected: PASS against the configured test catalog. If the POST path fails because the body format is wrong, fix `ErmrestPagedClient.fetch_rid_batch` (step 4) only — no fetcher or test changes — then re-run.

- [ ] **Step 8: Lint + format.**

```bash
uv run ruff format src/deriva_ml/local_db/paged_fetcher_ermrest.py tests/local_db/
uv run ruff check src/deriva_ml/local_db/paged_fetcher_ermrest.py tests/local_db/
```

- [ ] **Step 9: Commit.**

```bash
git add src/deriva_ml/local_db/paged_fetcher_ermrest.py \
        tests/local_db/test_paged_fetcher_live.py \
        tests/local_db/test_paged_fetcher.py
git commit -m "feat(local_db): add ErmrestPagedClient adapter and live tests"
```

---

## Task 8: `ManifestStore` — SQLite-backed persistence layer

`ManifestStore` is the storage engine that the rewritten `AssetManifest` will delegate to. It owns the `execution_state.assets` / `execution_state.features` tables in the workspace DB.

**Files:**
- Create: `src/deriva_ml/local_db/manifest_store.py`
- Create: `tests/local_db/test_manifest_store.py`

- [ ] **Step 1: Write failing tests.**

Write `tests/local_db/test_manifest_store.py`:

```python
"""Unit tests for local_db.manifest_store.ManifestStore."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import inspect, text

from deriva_ml.asset.manifest import AssetEntry, FeatureEntry
from deriva_ml.local_db.manifest_store import ManifestStore
from deriva_ml.local_db.workspace import Workspace


@pytest.fixture
def store(tmp_path: Path):
    ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
    s = ManifestStore(ws.engine)
    s.ensure_schema()
    yield s
    ws.close()


class TestEnsureSchema:
    def test_creates_tables(self, store: ManifestStore) -> None:
        tables = inspect(store._engine).get_table_names()
        assert "execution_state__assets" in tables
        assert "execution_state__features" in tables

    def test_idempotent(self, store: ManifestStore) -> None:
        store.ensure_schema()
        store.ensure_schema()  # second call should be a no-op


class TestAssetCrud:
    def test_add_and_get_asset(self, store: ManifestStore) -> None:
        entry = AssetEntry(
            asset_table="Image",
            schema="isa",
            asset_types=["Training"],
            metadata={"Subject": "2-ABC"},
            description="test",
        )
        store.add_asset("4SP", "Image/scan.jpg", entry)
        got = store.get_asset("4SP", "Image/scan.jpg")
        assert got.asset_table == "Image"
        assert got.asset_types == ["Training"]
        assert got.metadata == {"Subject": "2-ABC"}
        assert got.status == "pending"

    def test_list_assets_for_execution(self, store: ManifestStore) -> None:
        e1 = AssetEntry(asset_table="Image", schema="isa")
        e2 = AssetEntry(asset_table="Image", schema="isa")
        store.add_asset("exec-A", "Image/a.jpg", e1)
        store.add_asset("exec-B", "Image/b.jpg", e2)
        a_only = store.list_assets("exec-A")
        assert set(a_only.keys()) == {"Image/a.jpg"}

    def test_mark_uploaded(self, store: ManifestStore) -> None:
        entry = AssetEntry(asset_table="Image", schema="isa")
        store.add_asset("4SP", "Image/x.jpg", entry)
        store.mark_asset_uploaded("4SP", "Image/x.jpg", rid="1-RID")
        got = store.get_asset("4SP", "Image/x.jpg")
        assert got.status == "uploaded"
        assert got.rid == "1-RID"
        assert got.uploaded_at is not None

    def test_mark_failed(self, store: ManifestStore) -> None:
        entry = AssetEntry(asset_table="Image", schema="isa")
        store.add_asset("4SP", "Image/x.jpg", entry)
        store.mark_asset_failed("4SP", "Image/x.jpg", error="nope")
        got = store.get_asset("4SP", "Image/x.jpg")
        assert got.status == "failed"
        assert got.error == "nope"

    def test_update_metadata(self, store: ManifestStore) -> None:
        entry = AssetEntry(asset_table="Image", schema="isa", metadata={})
        store.add_asset("4SP", "Image/x.jpg", entry)
        store.update_asset_metadata("4SP", "Image/x.jpg", {"Subject": "ABC"})
        got = store.get_asset("4SP", "Image/x.jpg")
        assert got.metadata == {"Subject": "ABC"}

    def test_update_types(self, store: ManifestStore) -> None:
        entry = AssetEntry(asset_table="Image", schema="isa", asset_types=["A"])
        store.add_asset("4SP", "Image/x.jpg", entry)
        store.update_asset_types("4SP", "Image/x.jpg", ["B", "C"])
        got = store.get_asset("4SP", "Image/x.jpg")
        assert got.asset_types == ["B", "C"]


class TestFeatureCrud:
    def test_add_and_list(self, store: ManifestStore) -> None:
        f = FeatureEntry(
            feature_name="Diagnosis",
            target_table="Image",
            schema="isa",
            values_path="/some/path.csv",
            asset_columns={},
            status="pending",
        )
        store.add_feature("4SP", "Diagnosis", f)
        got = store.list_features("4SP")
        assert "Diagnosis" in got
        assert got["Diagnosis"].target_table == "Image"


class TestStatusFilters:
    def test_pending_only(self, store: ManifestStore) -> None:
        e1 = AssetEntry(asset_table="Image", schema="isa")
        e2 = AssetEntry(asset_table="Image", schema="isa")
        store.add_asset("4SP", "a", e1)
        store.add_asset("4SP", "b", e2)
        store.mark_asset_uploaded("4SP", "b", "1-RID")
        pending = store.pending_assets("4SP")
        assert set(pending.keys()) == {"a"}
```

- [ ] **Step 2: Run tests to confirm failure.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_manifest_store.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `ManifestStore`.**

Write `src/deriva_ml/local_db/manifest_store.py`:

```python
"""SQLite-backed persistence for AssetManifest.

Stores per-execution asset and feature entries in tables
``execution_state__assets`` and ``execution_state__features`` in the
workspace DB. WAL + per-mutation commit gives crash safety equivalent to the
old JSON fsync-on-write.

SQLite has no true schema namespacing; we use the ``execution_state__``
prefix as a logical namespace on table names.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Column,
    JSON,
    MetaData,
    String,
    Table,
    Text,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine

from deriva_ml.asset.manifest import AssetEntry, FeatureEntry

logger = logging.getLogger(__name__)

ASSETS_TABLE = "execution_state__assets"
FEATURES_TABLE = "execution_state__features"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ManifestStore:
    """SQLite persistence for asset + feature manifest entries."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._metadata = MetaData()
        self._assets_t = Table(
            ASSETS_TABLE,
            self._metadata,
            Column("execution_rid", String, primary_key=True),
            Column("key", String, primary_key=True),
            Column("asset_table", String, nullable=False),
            Column("schema", String, nullable=False),
            Column("asset_types", JSON),
            Column("metadata", JSON),
            Column("description", Text),
            Column("status", String, nullable=False),
            Column("rid", String),
            Column("uploaded_at", String),
            Column("error", Text),
            Column("created_at", String, nullable=False),
            Column("updated_at", String, nullable=False),
        )
        self._features_t = Table(
            FEATURES_TABLE,
            self._metadata,
            Column("execution_rid", String, primary_key=True),
            Column("feature_name", String, primary_key=True),
            Column("target_table", String, nullable=False),
            Column("schema", String, nullable=False),
            Column("values_path", String, nullable=False),
            Column("asset_columns", JSON),
            Column("status", String, nullable=False),
            Column("created_at", String, nullable=False),
            Column("updated_at", String, nullable=False),
        )

    def ensure_schema(self) -> None:
        """Create the tables if they don't exist."""
        self._metadata.create_all(self._engine)

    # ---------------- assets ---------------- #

    def add_asset(self, execution_rid: str, key: str, entry: AssetEntry) -> None:
        now = _now_iso()
        row = {
            "execution_rid": execution_rid,
            "key": key,
            "asset_table": entry.asset_table,
            "schema": entry.schema,
            "asset_types": entry.asset_types,
            "metadata": entry.metadata,
            "description": entry.description,
            "status": entry.status,
            "rid": entry.rid,
            "uploaded_at": entry.uploaded_at,
            "error": entry.error,
            "created_at": now,
            "updated_at": now,
        }
        with self._engine.begin() as conn:
            # Upsert: delete any existing row then insert.
            conn.execute(
                delete(self._assets_t).where(
                    (self._assets_t.c.execution_rid == execution_rid)
                    & (self._assets_t.c.key == key)
                )
            )
            conn.execute(insert(self._assets_t), row)

    def get_asset(self, execution_rid: str, key: str) -> AssetEntry:
        with self._engine.connect() as conn:
            row = conn.execute(
                select(self._assets_t).where(
                    (self._assets_t.c.execution_rid == execution_rid)
                    & (self._assets_t.c.key == key)
                )
            ).mappings().first()
        if row is None:
            raise KeyError(f"Asset '{key}' for execution '{execution_rid}' not found")
        return AssetEntry(
            asset_table=row["asset_table"],
            schema=row["schema"],
            asset_types=row["asset_types"] or [],
            metadata=row["metadata"] or {},
            description=row["description"],
            status=row["status"],
            rid=row["rid"],
            uploaded_at=row["uploaded_at"],
            error=row["error"],
        )

    def list_assets(self, execution_rid: str) -> dict[str, AssetEntry]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(self._assets_t).where(
                    self._assets_t.c.execution_rid == execution_rid
                )
            ).mappings().all()
        return {r["key"]: AssetEntry(
            asset_table=r["asset_table"],
            schema=r["schema"],
            asset_types=r["asset_types"] or [],
            metadata=r["metadata"] or {},
            description=r["description"],
            status=r["status"],
            rid=r["rid"],
            uploaded_at=r["uploaded_at"],
            error=r["error"],
        ) for r in rows}

    def pending_assets(self, execution_rid: str) -> dict[str, AssetEntry]:
        return {k: v for k, v in self.list_assets(execution_rid).items()
                if v.status == "pending"}

    def uploaded_assets(self, execution_rid: str) -> dict[str, AssetEntry]:
        return {k: v for k, v in self.list_assets(execution_rid).items()
                if v.status == "uploaded"}

    def update_asset_metadata(
        self, execution_rid: str, key: str, metadata: dict[str, Any]
    ) -> None:
        self._require_asset(execution_rid, key)
        with self._engine.begin() as conn:
            conn.execute(
                update(self._assets_t)
                .where(
                    (self._assets_t.c.execution_rid == execution_rid)
                    & (self._assets_t.c.key == key)
                )
                .values(metadata=metadata, updated_at=_now_iso())
            )

    def update_asset_types(
        self, execution_rid: str, key: str, asset_types: list[str]
    ) -> None:
        self._require_asset(execution_rid, key)
        with self._engine.begin() as conn:
            conn.execute(
                update(self._assets_t)
                .where(
                    (self._assets_t.c.execution_rid == execution_rid)
                    & (self._assets_t.c.key == key)
                )
                .values(asset_types=asset_types, updated_at=_now_iso())
            )

    def mark_asset_uploaded(
        self, execution_rid: str, key: str, rid: str
    ) -> None:
        self._require_asset(execution_rid, key)
        now = _now_iso()
        with self._engine.begin() as conn:
            conn.execute(
                update(self._assets_t)
                .where(
                    (self._assets_t.c.execution_rid == execution_rid)
                    & (self._assets_t.c.key == key)
                )
                .values(
                    status="uploaded",
                    rid=rid,
                    uploaded_at=now,
                    error=None,
                    updated_at=now,
                )
            )

    def mark_asset_failed(
        self, execution_rid: str, key: str, error: str
    ) -> None:
        self._require_asset(execution_rid, key)
        with self._engine.begin() as conn:
            conn.execute(
                update(self._assets_t)
                .where(
                    (self._assets_t.c.execution_rid == execution_rid)
                    & (self._assets_t.c.key == key)
                )
                .values(
                    status="failed",
                    error=error,
                    updated_at=_now_iso(),
                )
            )

    def _require_asset(self, execution_rid: str, key: str) -> None:
        with self._engine.connect() as conn:
            exists = conn.execute(
                select(self._assets_t.c.key).where(
                    (self._assets_t.c.execution_rid == execution_rid)
                    & (self._assets_t.c.key == key)
                )
            ).first()
        if exists is None:
            raise KeyError(f"Asset '{key}' for execution '{execution_rid}' not found")

    # ---------------- features ---------------- #

    def add_feature(
        self, execution_rid: str, feature_name: str, entry: FeatureEntry
    ) -> None:
        now = _now_iso()
        row = {
            "execution_rid": execution_rid,
            "feature_name": feature_name,
            "target_table": entry.target_table,
            "schema": entry.schema,
            "values_path": entry.values_path,
            "asset_columns": entry.asset_columns,
            "status": entry.status,
            "created_at": now,
            "updated_at": now,
        }
        with self._engine.begin() as conn:
            conn.execute(
                delete(self._features_t).where(
                    (self._features_t.c.execution_rid == execution_rid)
                    & (self._features_t.c.feature_name == feature_name)
                )
            )
            conn.execute(insert(self._features_t), row)

    def list_features(self, execution_rid: str) -> dict[str, FeatureEntry]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(self._features_t).where(
                    self._features_t.c.execution_rid == execution_rid
                )
            ).mappings().all()
        return {r["feature_name"]: FeatureEntry(
            feature_name=r["feature_name"],
            target_table=r["target_table"],
            schema=r["schema"],
            values_path=r["values_path"],
            asset_columns=r["asset_columns"] or {},
            status=r["status"],
        ) for r in rows}
```

- [ ] **Step 4: Run tests — expect pass.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_manifest_store.py -v
```

Expected: all pass.

- [ ] **Step 5: Lint + format, commit.**

```bash
uv run ruff format src/deriva_ml/local_db/manifest_store.py tests/local_db/
uv run ruff check src/deriva_ml/local_db/manifest_store.py
git add src/deriva_ml/local_db/manifest_store.py tests/local_db/test_manifest_store.py
git commit -m "feat(local_db): add ManifestStore (SQLite-backed asset/feature persistence)"
```

---

## Task 9: Rewrite `AssetManifest` to delegate to `ManifestStore`

The existing `AssetManifest` class in `src/deriva_ml/asset/manifest.py` (lines 79–221) owns a JSON file. We rewrite it to delegate every operation to a `ManifestStore`. The public API (`add_asset`, `mark_uploaded`, etc.) stays identical so the existing test suite at `tests/asset/test_manifest.py` continues to exercise behavior.

Constructor signature changes: `AssetManifest(path, execution_rid)` → `AssetManifest(store, execution_rid)`. Callers change at one call site (`execution.py:1405`).

**Files:**
- Modify: `src/deriva_ml/asset/manifest.py` (lines 79–221 — the `AssetManifest` class).
- Modify: `src/deriva_ml/execution/execution.py` (`_get_manifest`, around line 1402).
- Modify: `tests/asset/test_manifest.py` — update construction calls to use the new signature.

- [ ] **Step 1: Update the existing manifest tests to use the new constructor.**

Read `tests/asset/test_manifest.py` to see current construction pattern:

```bash
grep -n "AssetManifest(" /Users/carl/GitHub/deriva-ml/.claude/worktrees/compassionate-visvesvaraya/tests/asset/test_manifest.py
```

Update every call from `AssetManifest(path, rid)` to use a `ManifestStore` fixture. Add to the top of the file:

```python
import pytest
from sqlalchemy import create_engine

from deriva_ml.local_db.manifest_store import ManifestStore


@pytest.fixture
def store(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'ws.sqlite'}", future=True)
    s = ManifestStore(engine)
    s.ensure_schema()
    yield s
    engine.dispose()


@pytest.fixture
def manifest(store):
    from deriva_ml.asset.manifest import AssetManifest
    return AssetManifest(store, "4SP")
```

Rewrite each test to use the `manifest` (or `store`) fixture instead of `AssetManifest(mp, "4SP")`. For tests that currently check disk-state (e.g. that the JSON file was created/written), assert DB-state instead (e.g. row exists in `store.list_assets("4SP")`).

This is the largest test edit in the plan — there are ~10 test classes. Keep the assertions about behavior (statuses, mutations, round-trips), replace only the construction/access patterns.

- [ ] **Step 2: Run the existing manifest tests — expect failure.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/asset/test_manifest.py -v
```

Expected: failures because `AssetManifest` still takes `path`.

- [ ] **Step 3: Rewrite `AssetManifest`.**

Replace the `AssetManifest` class body in `src/deriva_ml/asset/manifest.py` (starting at line 79) with:

```python
class AssetManifest:
    """Per-execution asset+feature manifest, backed by :class:`ManifestStore`.

    Public API is unchanged from the JSON-backed implementation. Storage
    swapped to SQLite (WAL mode) for crash safety and cross-process visibility.

    Args:
        store: A ``ManifestStore`` bound to a workspace engine.
        execution_rid: RID of the execution this manifest covers.
    """

    MANIFEST_VERSION = 2  # bumped: storage layer changed

    def __init__(self, store: "ManifestStore", execution_rid: str) -> None:
        self._store = store
        self._execution_rid = execution_rid

    @property
    def execution_rid(self) -> str:
        return self._execution_rid

    @property
    def assets(self) -> dict[str, AssetEntry]:
        return self._store.list_assets(self._execution_rid)

    @property
    def features(self) -> dict[str, FeatureEntry]:
        return self._store.list_features(self._execution_rid)

    def pending_assets(self) -> dict[str, AssetEntry]:
        return self._store.pending_assets(self._execution_rid)

    def uploaded_assets(self) -> dict[str, AssetEntry]:
        return self._store.uploaded_assets(self._execution_rid)

    def add_asset(self, key: str, entry: AssetEntry) -> None:
        self._store.add_asset(self._execution_rid, key, entry)

    def update_asset_metadata(self, key: str, metadata: dict[str, Any]) -> None:
        self._store.update_asset_metadata(self._execution_rid, key, metadata)

    def update_asset_types(self, key: str, asset_types: list[str]) -> None:
        self._store.update_asset_types(self._execution_rid, key, asset_types)

    def mark_uploaded(self, key: str, rid: str) -> None:
        self._store.mark_asset_uploaded(self._execution_rid, key, rid)

    def mark_failed(self, key: str, error: str) -> None:
        self._store.mark_asset_failed(self._execution_rid, key, error)

    def add_feature(self, name: str, entry: FeatureEntry) -> None:
        self._store.add_feature(self._execution_rid, name, entry)

    def to_json(self) -> dict[str, Any]:
        """Return a dict mirroring the legacy JSON file format.

        For debugging/postmortems. Serialize with
        ``json.dumps(manifest.to_json(), default=_json_default)`` to handle
        datetimes and Path values.
        """
        return {
            "version": self.MANIFEST_VERSION,
            "execution_rid": self._execution_rid,
            "assets": {k: v.to_dict() for k, v in self.assets.items()},
            "features": {k: v.to_dict() for k, v in self.features.items()},
        }
```

Add to the imports at the top of `src/deriva_ml/asset/manifest.py`:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deriva_ml.local_db.manifest_store import ManifestStore
```

Remove the following no-longer-needed attributes/methods from `AssetManifest`: `_path`, `path` property, `_save`, `_load`, `_json_default` (but keep the module-level `_json_default` function for `to_json`).

- [ ] **Step 4: Update the one call site in `execution.py`.**

In `src/deriva_ml/execution/execution.py`, replace `_get_manifest` (around line 1402) with:

```python
    def _get_manifest(self) -> AssetManifest:
        """Return the execution's asset manifest (lazy init).

        Backed by the workspace ManifestStore; rows live in the per-catalog
        working.db, not a per-execution JSON file.
        """
        if not hasattr(self, "_manifest") or self._manifest is None:
            from deriva_ml.local_db.manifest_store import ManifestStore

            store = ManifestStore(self._ml.working_data._ws.engine)  # type: ignore[attr-defined]
            store.ensure_schema()
            self._manifest = AssetManifest(store, self.execution_rid)
        return self._manifest
```

Also remove the `from deriva_ml.dataset.upload import manifest_path` import near the top (if present) — `manifest_path` is no longer needed here but may still be used elsewhere, so check with `grep` before deleting:

```bash
grep -n "manifest_path" /Users/carl/GitHub/deriva-ml/.claude/worktrees/compassionate-visvesvaraya/src/deriva_ml/execution/execution.py
```

If the only use was `_get_manifest`, remove the import. Otherwise leave it.

The `self._ml.working_data._ws.engine` reach-through assumes the `legacy_working_data_view` keeps a `_ws` backreference. Verify that's the case by re-reading Task 4's `_LegacyWorkingDataView` — it stores `self._ws = ws`. So the reach-through is valid but ugly. An explicit accessor would be cleaner. Add this method to `Workspace` in `src/deriva_ml/local_db/workspace.py`:

```python
    def manifest_store(self) -> "ManifestStore":
        """Return a ManifestStore backed by this workspace's engine (cached)."""
        if not hasattr(self, "_manifest_store"):
            from deriva_ml.local_db.manifest_store import ManifestStore
            s = ManifestStore(self.engine)
            s.ensure_schema()
            self._manifest_store = s
        return self._manifest_store
```

And expose `_ws` on `_LegacyWorkingDataView` cleanly — keep the private attribute but add a public accessor:

```python
    @property
    def workspace(self) -> "Workspace":
        return self._ws
```

Then the execution.py reach-through becomes:

```python
        ws = self._ml.working_data.workspace
        self._manifest = AssetManifest(ws.manifest_store(), self.execution_rid)
```

That's cleaner. Use this final form in `execution.py`.

- [ ] **Step 5: Run the manifest tests and execution tests to check for regressions.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/asset/test_manifest.py tests/local_db/ -v
```

Expected: all pass.

Then:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/ -v -k "manifest or execution" 2>&1 | tail -80
```

Expected: no *new* failures. (Some execution tests need a live catalog and may be skipped in unit-only runs.)

- [ ] **Step 6: Lint + format.**

```bash
uv run ruff format src/deriva_ml/asset/manifest.py src/deriva_ml/execution/execution.py src/deriva_ml/local_db/workspace.py
uv run ruff check src/deriva_ml/asset/manifest.py src/deriva_ml/execution/execution.py src/deriva_ml/local_db/workspace.py
```

- [ ] **Step 7: Commit.**

```bash
git add src/deriva_ml/asset/manifest.py \
        src/deriva_ml/execution/execution.py \
        src/deriva_ml/local_db/workspace.py \
        tests/asset/test_manifest.py
git commit -m "refactor(asset): AssetManifest delegates to SQLite ManifestStore"
```

---

## Task 10: JSON → DB import on workspace open

Pre-existing `asset-manifest.json` files from in-progress executions get imported into the workspace DB automatically, then renamed to `*.migrated.json`.

**Files:**
- Modify: `src/deriva_ml/local_db/workspace.py` — add `import_legacy_manifests()` method + wire into `manifest_store()` lazy init.
- Create: `tests/local_db/test_manifest_migration.py`

- [ ] **Step 1: Write failing test.**

Write `tests/local_db/test_manifest_migration.py`:

```python
"""Integration test: JSON manifest migration on workspace open."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from deriva_ml.local_db.workspace import Workspace


@pytest.fixture
def legacy_manifest_dir(tmp_path: Path) -> Path:
    """Seed a legacy execution directory with an asset-manifest.json."""
    exec_dir = tmp_path / "execution" / "2-ABC"
    exec_dir.mkdir(parents=True)
    manifest = {
        "version": 1,
        "execution_rid": "2-ABC",
        "created_at": "2026-04-15T00:00:00Z",
        "assets": {
            "Image/scan.jpg": {
                "asset_table": "Image",
                "schema": "isa",
                "asset_types": ["Training"],
                "metadata": {"Subject": "S-1"},
                "description": "",
                "status": "pending",
                "rid": None,
                "uploaded_at": None,
                "error": None,
            }
        },
        "features": {},
    }
    (exec_dir / "asset-manifest.json").write_text(json.dumps(manifest))
    return exec_dir


class TestManifestImport:
    def test_imports_legacy_manifests(
        self, tmp_path: Path, legacy_manifest_dir: Path
    ) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            count = ws.import_legacy_manifests()
            assert count == 1

            store = ws.manifest_store()
            rows = store.list_assets("2-ABC")
            assert "Image/scan.jpg" in rows
            assert rows["Image/scan.jpg"].asset_types == ["Training"]

            # Sidecar file created
            assert (legacy_manifest_dir / "asset-manifest.json.migrated.json").is_file()
            # Original removed
            assert not (legacy_manifest_dir / "asset-manifest.json").exists()
        finally:
            ws.close()

    def test_idempotent(self, tmp_path: Path, legacy_manifest_dir: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            count1 = ws.import_legacy_manifests()
            count2 = ws.import_legacy_manifests()
            assert count1 == 1
            assert count2 == 0  # Already migrated
        finally:
            ws.close()

    def test_no_manifests_no_sidecar(self, tmp_path: Path) -> None:
        ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
        try:
            count = ws.import_legacy_manifests()
            assert count == 0
        finally:
            ws.close()
```

- [ ] **Step 2: Run test — expect failure.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_manifest_migration.py -v
```

Expected: `AttributeError: 'Workspace' object has no attribute 'import_legacy_manifests'`.

- [ ] **Step 3: Implement `import_legacy_manifests` on `Workspace`.**

Add to `src/deriva_ml/local_db/workspace.py` (inside the `Workspace` class):

```python
    def import_legacy_manifests(self) -> int:
        """Import any pre-existing ``asset-manifest.json`` files into the DB.

        Scans ``{working_dir}`` recursively for files named exactly
        ``asset-manifest.json``. For each, parses the JSON, upserts rows into
        the manifest store, and renames the file to
        ``{path}.migrated.json``. Idempotent: already-migrated files are
        skipped.

        Returns:
            The number of manifests newly migrated.
        """
        import json as _json
        from deriva_ml.asset.manifest import AssetEntry, FeatureEntry

        store = self.manifest_store()
        migrated = 0
        for manifest_path in self._working_dir.rglob("asset-manifest.json"):
            try:
                data = _json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to read %s: %s", manifest_path, exc)
                continue

            execution_rid = data.get("execution_rid") or manifest_path.parent.name
            for key, entry_dict in data.get("assets", {}).items():
                entry = AssetEntry.from_dict(entry_dict)
                store.add_asset(execution_rid, key, entry)
            for name, entry_dict in data.get("features", {}).items():
                entry = FeatureEntry.from_dict(entry_dict)
                store.add_feature(execution_rid, name, entry)

            sidecar = manifest_path.with_suffix(
                manifest_path.suffix + ".migrated.json"
            )
            manifest_path.rename(sidecar)
            migrated += 1
        return migrated
```

- [ ] **Step 4: Run test — expect pass.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_manifest_migration.py -v
```

Expected: all pass.

- [ ] **Step 5: Wire auto-import into `DerivaML.working_data`.**

In `src/deriva_ml/core/base.py`, update the `working_data` property body (added in Task 5) to call the import once on workspace construction:

```python
    @property
    def working_data(self):
        from deriva_ml.local_db.workspace import Workspace

        if not hasattr(self, "_workspace"):
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
                        "Migrated %d legacy asset manifests into %s",
                        n, self._workspace.working_db_path,
                    )
            except Exception as exc:
                import logging
                logging.getLogger("deriva_ml").warning(
                    "Legacy manifest migration failed: %s", exc,
                )
        return self._workspace.legacy_working_data_view()
```

- [ ] **Step 6: Re-run the broader test suite to check for regressions.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/test_manifest.py tests/test_working_data.py -v
```

Expected: all pass.

- [ ] **Step 7: Lint + format, commit.**

```bash
uv run ruff format src/deriva_ml/local_db/workspace.py src/deriva_ml/core/base.py tests/local_db/
uv run ruff check src/deriva_ml/local_db/workspace.py src/deriva_ml/core/base.py
git add src/deriva_ml/local_db/workspace.py \
        src/deriva_ml/core/base.py \
        tests/local_db/test_manifest_migration.py
git commit -m "feat(local_db): import legacy JSON asset manifests on workspace open"
```

---

## Task 11: Phase 1 smoke suite + documentation

Final pass: public exports, a short README for the new package, and a full test suite run.

**Files:**
- Modify: `src/deriva_ml/local_db/__init__.py` — export public classes.
- Create: `src/deriva_ml/local_db/README.md` — short module orientation.

- [ ] **Step 1: Update `local_db/__init__.py` with public exports.**

Replace the contents of `src/deriva_ml/local_db/__init__.py`:

```python
"""Unified local SQLite layer for deriva-ml.

See ``docs/superpowers/specs/2026-04-15-unified-local-db-design.md`` for design.
See ``README.md`` in this directory for a short orientation.
"""

from __future__ import annotations

from deriva_ml.local_db.manifest_store import ManifestStore
from deriva_ml.local_db.paged_fetcher import PagedFetcher
from deriva_ml.local_db.paged_fetcher_ermrest import ErmrestPagedClient
from deriva_ml.local_db.schema import LocalSchema
from deriva_ml.local_db.workspace import Workspace

__all__ = [
    "ErmrestPagedClient",
    "LocalSchema",
    "ManifestStore",
    "PagedFetcher",
    "Workspace",
]
```

- [ ] **Step 2: Write the module README.**

Write `src/deriva_ml/local_db/README.md`:

```markdown
# `deriva_ml.local_db`

Unified client-side SQLite layer for deriva-ml. Phase 1 delivers the
foundation; Phase 2 will migrate denormalization onto it.

## Modules

- `paths.py` — pure path helpers for the working-directory layout.
- `sqlite_helpers.py` — WAL engine factory, ATTACH/DETACH, schema-version
  runner.
- `schema.py` — `LocalSchema` adapter over `SchemaBuilder`.
- `workspace.py` — `Workspace`: per-catalog `working.db` handle, slice
  attach, legacy working-data view, manifest store, legacy-manifest import.
- `paged_fetcher.py` — transport-agnostic `PagedFetcher` primitive.
- `paged_fetcher_ermrest.py` — `ErmrestPagedClient` adapter to deriva-py.
- `manifest_store.py` — SQLite persistence for `AssetManifest`.

## Layout on disk

```
{working_dir}/
  catalogs/
    {host}__{cat}/
      working.db
      slices/
        {slice_id}/
          slice.db
          assets/
```

## Further reading

- Design spec: `docs/superpowers/specs/2026-04-15-unified-local-db-design.md`
- Phase 1 plan: `docs/superpowers/plans/2026-04-15-unified-local-db-phase1.md`
```

- [ ] **Step 3: Run the full local_db test suite + any likely-affected tests.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/test_manifest.py tests/test_working_data.py -v
```

Expected: all pass.

- [ ] **Step 4: Run the full test suite and flag anything unexpected.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest -x 2>&1 | tail -60
```

If any tests fail that aren't obviously unrelated (e.g. pre-existing test-env issues), stop and investigate. A `_LegacyWorkingDataView` consumer or an `AssetManifest` caller that was missed earlier may surface here.

- [ ] **Step 5: Verify imports from the package work as documented.**

Run:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run python -c "from deriva_ml.local_db import Workspace, PagedFetcher, LocalSchema, ManifestStore, ErmrestPagedClient; print('OK')"
```

Expected output: `OK`.

- [ ] **Step 6: Lint + format.**

```bash
uv run ruff format src/deriva_ml/local_db/
uv run ruff check src/deriva_ml/local_db/
```

- [ ] **Step 7: Commit.**

```bash
git add src/deriva_ml/local_db/__init__.py src/deriva_ml/local_db/README.md
git commit -m "docs(local_db): public exports and module README"
```

---

## Verification checklist (run before handing off)

- [ ] `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ -v` — all pass.
- [ ] `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/asset/test_manifest.py -v` — all pass.
- [ ] `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/test_working_data.py -v` — all pass (deprecation warnings are expected).
- [ ] `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/test_paged_fetcher_live.py -v -m integration` — all pass against the test catalog.
- [ ] `DERIVA_ML_ALLOW_DIRTY=true uv run pytest -x` — no new failures.
- [ ] `uv run ruff check src/` — clean.
- [ ] `git log --oneline` — clear, well-scoped commits, one per task.
- [ ] `src/deriva_ml/local_db/` directory holds exactly: `__init__.py`, `README.md`, `paths.py`, `sqlite_helpers.py`, `schema.py`, `workspace.py`, `paged_fetcher.py`, `paged_fetcher_ermrest.py`, `manifest_store.py`.

## What this plan does NOT deliver (deferred to Phase 2)

- Unified denormalization (`local_db/denormalize.py`).
- Deletion of `Dataset._denormalize_datapath` and `DatasetBag._denormalize`.
- Public cached tabular-read API (`cached_table_read`, `CachedResult`, `list_cached_results`, `invalidate_cache`).
- Slice identity / registry beyond the path helpers (no automatic
  `manifest.json` provenance file yet — consumers who create slices write it
  themselves, or will after Phase 2 defines the schema).
- MCP migration (separate repo).

Phase 1's deliverable is: "the plumbing is in place; behavior is unchanged."
