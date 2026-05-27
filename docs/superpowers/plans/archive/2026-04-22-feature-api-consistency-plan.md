# Feature API Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify the feature-access API across `DerivaML`, `Dataset`, and `DatasetBag`; route writes through execution-staged SQLite; retire duplicate and escape-hatch APIs.

**Architecture:** Three symmetric read methods per container (`find_features`, `feature_values`, `lookup_feature`) plus a shared helper (`list_workflow_executions`). `feature_values` returns `Iterable[FeatureRecord]` (pandas-free, stream-friendly). Writes flow exclusively through `exe.add_features` which stages rows to the execution's SQLite and flushes on completion after asset upload. A new selector factory `FeatureRecord.select_by_workflow(workflow, *, container)` replaces the catalog-bound `ml.select_by_workflow` method. The `Denormalizer` subsystem is untouched — feature-wide DataFrames remain its job.

**Tech Stack:** Python 3.12+, Pydantic v2 (`FeatureRecord`, `validate_call`), SQLAlchemy (`execution_state__feature_records` table in execution SQLite), deriva-py datapath (online write paths), pytest + pytest-parametrize (symmetry compliance suite).

**Spec:** `docs/superpowers/specs/2026-04-22-feature-api-consistency-design.md`.

**Dependency ordering:** The §8 SQLite execution-state design provides the staging infrastructure. This plan replaces the existing file-based feature upload (`FeatureEntry` + `FEATURES_TABLE` + per-feature `.jsonl` files + `_update_feature_table`) with row-per-record SQLite staging. Goal is DRY — the file-based path did the same job via filesystem round-trip and is removed in this plan.

**What the old file-based path did that we must preserve:** at flush time, asset-column values that hold local asset filenames must be rewritten to the uploaded asset RIDs (the asset upload happens earlier in `upload_execution_outputs`, so the map from local filename → asset RID is in-memory). This logic lives in `_update_feature_table` today; it moves into `_flush_staged_features` — same contract, different storage.

---

## File Structure

**New files:**
- `src/deriva_ml/dataset/bag_feature_cache.py` — per-feature bag denormalization cache reader (populate-on-first-access, SQLite-backed, read-only).
- `tests/feature/test_feature_values.py` — parametrized symmetry suite (DerivaML / Dataset / DatasetBag).
- `tests/feature/test_select_by_workflow.py` — selector factory tests (online, dataset, bag).
- `tests/feature/test_retired_apis.py` — each retired method raises the expected exception with the expected replacement pointer.
- `tests/execution/test_staged_features.py` — stage, flush, order, resume, error paths.
- `tests/dataset/test_bag_feature_cache.py` — cache population, immutability, corrupt-cache error path.

**Modified files:**
- `src/deriva_ml/feature.py` — add `FeatureRecord.select_by_workflow(workflow, *, container)` classmethod factory; expand docstrings.
- `src/deriva_ml/core/mixins/feature.py` — add `feature_values`, `list_workflow_executions`; remove `add_features`, `fetch_table_features`, `list_feature_values`, `select_by_workflow` (replaced with retired-API error shims).
- `src/deriva_ml/dataset/dataset.py` — add `feature_values`, `lookup_feature`, `list_workflow_executions` (delegate to `ml` where appropriate; filter target RIDs to dataset members where not).
- `src/deriva_ml/dataset/dataset_bag.py` — rewrite feature read path through `bag_feature_cache`; remove `fetch_table_features`, `list_feature_values`; add `lookup_feature`, `list_workflow_executions`.
- `src/deriva_ml/execution/execution.py` — change `add_features` from write-through to stage-to-SQLite; add `_flush_staged_features()`; integrate resume detection with existing staged-asset resume.
- `src/deriva_ml/local_db/manifest_store.py` — add `execution_state__feature_records` table (row-per-record staging); **remove** the now-obsolete `FEATURES_TABLE` (`execution_state__features`), its `FeatureEntry` dataclass in `src/deriva_ml/asset/manifest.py`, `ManifestStore.add_feature` / `ManifestStore.list_features`, and `AssetManifest.add_feature` / `AssetManifest.features` property.
- `src/deriva_ml/asset/manifest.py` — delete `FeatureEntry` dataclass and `AssetManifest.add_feature` / `AssetManifest.features` (no replacement — callers use `exe.add_features` directly).
- `src/deriva_ml/asset/__init__.py` — remove `FeatureEntry` from `__all__`.
- `src/deriva_ml/local_db/workspace.py:363-364` — remove the `FeatureEntry.from_dict` / `store.add_feature` migration path (file-based manifest migration is no longer relevant).
- `src/deriva_ml/dataset/upload.py:22-35, 265` — remove the `.jsonl` per-feature file helpers.
- `src/deriva_ml/execution/execution.py:1200-1207` — remove the `.jsonl` glob and `_update_feature_table` call; the row-per-record flush replaces it. **Preserve the asset-column rewriting logic** (map local filename → uploaded asset RID) by moving it into `_flush_staged_features`.
- `src/deriva_ml/asset/asset.py` — update `list_feature_values` callers if any; retire the method on `Asset` class too (see Task 10).
- `src/deriva_ml/interfaces.py` — update `DatasetLike` and related protocols to the three-method read surface + `lookup_feature` + `list_workflow_executions`; remove retired signatures.
- `src/deriva_ml/model/deriva_ml_database.py` — check `find_features` signature consistency with the protocol update.

**Unchanged (out of scope, do not touch):**
- `src/deriva_ml/local_db/denormalizer.py` and related `Denormalizer` code.
- `src/deriva_ml/feature.py` — `Feature` class, `feature_record_class()`, existing selectors (`select_newest`, `select_first`, `select_latest`, `select_by_execution`, `select_majority_vote`).
- `create_feature` schema operation.
- The pre-existing file-based feature upload path (`FEATURES_TABLE` in `manifest_store.py`, tracked by `values_path`).

---

## Task 1: Add `execution_state__feature_records` SQLite table

**Files:**
- Modify: `src/deriva_ml/local_db/manifest_store.py:89-103` (add new table alongside `_features_t`)
- Test: `tests/execution/test_staged_features.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/execution/test_staged_features.py
"""Unit tests for the row-per-record execution_state__feature_records table.

Stages individual FeatureRecord instances as JSON rows for later batch flush
to ermrest. Replaces the older file-based FEATURES_TABLE / .jsonl path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select

from deriva_ml.local_db.manifest_store import ManifestStore


def test_feature_records_table_created(tmp_path: Path) -> None:
    """ManifestStore.ensure_schema creates the execution_state__feature_records table."""
    engine = create_engine(f"sqlite:///{tmp_path / 'manifest.sqlite'}")
    store = ManifestStore(engine)
    store.ensure_schema()
    # Table is accessible via the store's metadata
    assert "execution_state__feature_records" in store._metadata.tables
    table = store._metadata.tables["execution_state__feature_records"]
    expected_cols = {
        "stage_id", "execution_rid", "feature_table", "feature_name",
        "target_table", "record_json", "created_at", "status",
        "uploaded_at", "error",
    }
    assert set(c.name for c in table.columns) == expected_cols


def test_stage_feature_record_inserts_row(tmp_path: Path) -> None:
    """stage_feature_record writes a Pending row readable by list_feature_records."""
    engine = create_engine(f"sqlite:///{tmp_path / 'manifest.sqlite'}")
    store = ManifestStore(engine)
    store.ensure_schema()
    store.stage_feature_record(
        execution_rid="EXE-1",
        feature_table="domain.Image_Glaucoma",
        feature_name="Glaucoma",
        target_table="Image",
        record_json=json.dumps({"Image": "IMG-1", "Glaucoma": "Normal"}),
    )
    rows = store.list_feature_records("EXE-1")
    assert len(rows) == 1
    row = rows[0]
    assert row.feature_table == "domain.Image_Glaucoma"
    assert row.feature_name == "Glaucoma"
    assert row.status == "Pending"
    assert row.error is None
    assert row.uploaded_at is None
    assert json.loads(row.record_json) == {"Image": "IMG-1", "Glaucoma": "Normal"}


def test_mark_feature_record_uploaded(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'manifest.sqlite'}")
    store = ManifestStore(engine)
    store.ensure_schema()
    store.stage_feature_record(
        execution_rid="EXE-1",
        feature_table="domain.Image_Glaucoma",
        feature_name="Glaucoma",
        target_table="Image",
        record_json=json.dumps({"Image": "IMG-1"}),
    )
    stage_id = store.list_feature_records("EXE-1")[0].stage_id
    store.mark_feature_record_uploaded(stage_id)
    row = store.list_feature_records("EXE-1")[0]
    assert row.status == "Uploaded"
    assert row.uploaded_at is not None


def test_mark_feature_record_failed_records_error(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'manifest.sqlite'}")
    store = ManifestStore(engine)
    store.ensure_schema()
    store.stage_feature_record(
        execution_rid="EXE-1",
        feature_table="domain.Image_Glaucoma",
        feature_name="Glaucoma",
        target_table="Image",
        record_json=json.dumps({"Image": "IMG-1"}),
    )
    stage_id = store.list_feature_records("EXE-1")[0].stage_id
    store.mark_feature_record_failed(stage_id, error="ermrest rejected")
    row = store.list_feature_records("EXE-1")[0]
    assert row.status == "Failed"
    assert row.error == "ermrest rejected"


def test_list_pending_feature_records_filters_by_status(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'manifest.sqlite'}")
    store = ManifestStore(engine)
    store.ensure_schema()
    # Stage three: one pending, one uploaded, one failed
    for i, _ in enumerate(range(3)):
        store.stage_feature_record(
            execution_rid="EXE-1",
            feature_table="domain.Image_Glaucoma",
            feature_name="Glaucoma",
            target_table="Image",
            record_json=json.dumps({"Image": f"IMG-{i}"}),
        )
    ids = [r.stage_id for r in store.list_feature_records("EXE-1")]
    store.mark_feature_record_uploaded(ids[0])
    store.mark_feature_record_failed(ids[1], error="x")
    pending = store.list_pending_feature_records("EXE-1")
    assert len(pending) == 1
    assert pending[0].stage_id == ids[2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_staged_features.py -q`
Expected: FAIL — `ManifestStore` has no `stage_feature_record` / `list_feature_records` / `mark_feature_record_uploaded` / `mark_feature_record_failed` / `list_pending_feature_records` methods, and the `execution_state__feature_records` SQLite table does not exist.

- [ ] **Step 3: Add the SQLite table**

In `src/deriva_ml/local_db/manifest_store.py`, **replace** the existing `_features_t` (around line 89) with the new row-per-record table. Also delete the `FEATURES_TABLE = "execution_state__features"` constant at line 38 and the `Index` definition at line 103 that references `_features_t`:

```python
# Row-per-record staging for feature writes. Replaces the earlier file-based
# FeatureEntry/FEATURES_TABLE mechanism (which wrote values to per-feature .jsonl
# files and tracked them by values_path). One mechanism, one storage layer.
# See docs/superpowers/specs/2026-04-22-feature-api-consistency-design.md §Architecture.
FEATURE_RECORDS_TABLE = "execution_state__feature_records"

self._feature_records_t = Table(
    FEATURE_RECORDS_TABLE,
    self._metadata,
    Column("stage_id", Integer, primary_key=True, autoincrement=True),
    Column("execution_rid", String, nullable=False),
    Column("feature_table", String, nullable=False),   # "schema.Table"
    Column("feature_name", String, nullable=False),
    Column("target_table", String, nullable=False),    # target table name — avoids re-lookup at flush
    Column("record_json", Text, nullable=False),
    Column("created_at", String, nullable=False),
    Column("status", String, nullable=False),          # Pending | Uploaded | Failed
    Column("uploaded_at", String),
    Column("error", Text),
)
Index(
    "ix_execution_state__feature_records_exec_status",
    self._feature_records_t.c.execution_rid,
    self._feature_records_t.c.status,
)
```

Add `Integer` to the `from sqlalchemy import ...` list at the top of the file if not already present.

- [ ] **Step 4: Add the accessor methods**

In the same file, after the existing feature-entry methods (`list_features`, around line 413), add:

```python
@dataclass(frozen=True)
class StagedFeatureRow:
    """One staged feature-record row awaiting flush to the catalog.

    A StagedFeatureRow is one FeatureRecord serialized as JSON, with
    lifecycle status tracked per row. Rows are created by
    ``Execution.add_features()`` and consumed by
    ``Execution._flush_staged_features()``.

    Attributes:
        stage_id: Autoincrement primary key.
        execution_rid: RID of the owning execution.
        feature_table: Qualified feature table name ("schema.Table").
        feature_name: Feature name (redundant with feature_table but kept
            for ergonomic filtering).
        target_table: Target table name (the table the feature is *on*).
            Stored so flush doesn't have to re-resolve it from the model.
        record_json: JSON encoding of ``FeatureRecord.model_dump_json()``.
        created_at: ISO timestamp of staging.
        status: "Pending" | "Uploaded" | "Failed".
        uploaded_at: ISO timestamp of successful flush, or None.
        error: Error message on Failed status, or None.
    """
    stage_id: int
    execution_rid: str
    feature_table: str
    feature_name: str
    target_table: str
    record_json: str
    created_at: str
    status: str
    uploaded_at: str | None
    error: str | None
```

Place `StagedFeatureRow` in the file's top-level declarations — it's an internal value object, `@dataclass` is appropriate per CLAUDE.md guidance (no user-facing serialization).

**Also in the same step**, delete the now-obsolete file-based feature machinery:

```python
# In src/deriva_ml/local_db/manifest_store.py:
#   - DELETE: `FEATURES_TABLE = "execution_state__features"` constant (line 38)
#   - DELETE: `self._features_t = Table(...)` definition (lines 89-101)
#   - DELETE: `Index("ix_features_exec_status", ...)` (line 103)
#   - DELETE: `ManifestStore.add_feature(...)` method (around line 357)
#   - DELETE: `ManifestStore.list_features(...)` method (around line 388)
#   - UPDATE: `from deriva_ml.asset.manifest import AssetEntry, FeatureEntry`
#     becomes `from deriva_ml.asset.manifest import AssetEntry` (remove FeatureEntry)

# In src/deriva_ml/asset/manifest.py:
#   - DELETE: `FeatureEntry` dataclass (around line 63-79)
#   - DELETE: `AssetManifest.features` property (around line 107-109)
#   - DELETE: `AssetManifest.add_feature()` method (around line 149-150)

# In src/deriva_ml/asset/__init__.py:
#   - UPDATE: remove "FeatureEntry" from `__all__` (line 25) and from the
#     `from .manifest import ...` statement (line 15).

# In src/deriva_ml/local_db/workspace.py (around lines 347-366):
#   - DELETE: the FeatureEntry migration import and the `store.add_feature`
#     call that migrates legacy JSON manifests. This is a file-format
#     migration for the old path that no longer exists; without the target,
#     the migration cannot run.

# In src/deriva_ml/dataset/upload.py (around lines 22-35, 265, 713):
#   - DELETE: feature-directory + feature .jsonl helpers that support the
#     file-based path. The asset-side helpers stay (assets still flow
#     through the filesystem for upload).

# In src/deriva_ml/execution/execution.py (around lines 1200-1207):
#   - REPLACE the `.jsonl` glob + `_update_feature_table` loop with a call
#     to `self._flush_staged_features()`. The asset-column rewriting logic
#     inside `_update_feature_table` moves into `_flush_staged_features`
#     (Task 7 implements that part).
#   - DELETE: `_update_feature_table` method after confirming nothing else
#     calls it.
#   - DELETE: `self._feature_root` initialization if it becomes unused
#     after removing the .jsonl path.
```

The deletions are tightly coupled to the additions in Task 1 — do them together in the same commit so tests can never see a half-migrated state.

Then add the methods on `ManifestStore`:

```python
def stage_feature_record(
    self,
    *,
    execution_rid: str,
    feature_table: str,
    feature_name: str,
    target_table: str,
    record_json: str,
) -> int:
    """Stage a single serialized FeatureRecord for later flush.

    Inserts a new row into ``execution_state__feature_records`` with status
    'Pending'. The returned ``stage_id`` is the row's autoincrement primary
    key — use it with :meth:`mark_feature_record_uploaded` or
    :meth:`mark_feature_record_failed` to update lifecycle status.

    Args:
        execution_rid: RID of the owning execution.
        feature_table: Qualified feature table name ("schema.Table").
        feature_name: Feature name.
        target_table: Target table name (the table the feature is defined on).
            Stored so flush doesn't have to re-resolve it from the model.
        record_json: JSON string from ``FeatureRecord.model_dump_json()``.

    Returns:
        The new row's ``stage_id``.
    """
    now = _now_iso()
    with self._engine.begin() as conn:
        result = conn.execute(
            insert(self._feature_records_t),
            {
                "execution_rid": execution_rid,
                "feature_table": feature_table,
                "feature_name": feature_name,
                "target_table": target_table,
                "record_json": record_json,
                "created_at": now,
                "status": "Pending",
                "uploaded_at": None,
                "error": None,
            },
        )
        return int(result.inserted_primary_key[0])

def list_feature_records(self, execution_rid: str) -> list["StagedFeatureRow"]:
    """Return all staged-feature rows for an execution, in stage_id order."""
    with self._engine.connect() as conn:
        rows = conn.execute(
            select(self._feature_records_t)
            .where(self._feature_records_t.c.execution_rid == execution_rid)
            .order_by(self._feature_records_t.c.stage_id)
        ).mappings().all()
    return [StagedFeatureRow(**dict(r)) for r in rows]

def list_pending_feature_records(
    self, execution_rid: str
) -> list["StagedFeatureRow"]:
    """Return Pending staged-feature rows for an execution."""
    with self._engine.connect() as conn:
        rows = conn.execute(
            select(self._feature_records_t)
            .where(
                (self._feature_records_t.c.execution_rid == execution_rid)
                & (self._feature_records_t.c.status == "Pending")
            )
            .order_by(self._feature_records_t.c.stage_id)
        ).mappings().all()
    return [StagedFeatureRow(**dict(r)) for r in rows]

def mark_feature_record_uploaded(self, stage_id: int) -> None:
    """Transition a staged row to Uploaded, setting uploaded_at."""
    with self._engine.begin() as conn:
        conn.execute(
            update(self._feature_records_t)
            .where(self._feature_records_t.c.stage_id == stage_id)
            .values(status="Uploaded", uploaded_at=_now_iso())
        )

def mark_feature_record_failed(self, stage_id: int, *, error: str) -> None:
    """Transition a staged row to Failed, recording the error."""
    with self._engine.begin() as conn:
        conn.execute(
            update(self._feature_records_t)
            .where(self._feature_records_t.c.stage_id == stage_id)
            .values(status="Failed", error=error)
        )
```

Add imports as needed: `from sqlalchemy import insert, update, select, Integer, Text`, `from dataclasses import dataclass`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/execution/test_staged_features.py -q`
Expected: 5 tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/execution/test_staged_features.py src/deriva_ml/local_db/manifest_store.py
git commit -m "feat(staged-features): row-per-record SQLite staging table (S2)"
```

---

## Task 2: Add `FeatureRecord.select_by_workflow` selector factory

**Files:**
- Modify: `src/deriva_ml/feature.py` (add classmethod after existing `select_by_execution`, around line 175)
- Test: `tests/feature/test_select_by_workflow.py` (new)

- [ ] **Step 1: Write the failing test (no catalog needed — use a stub container)**

```python
# tests/feature/test_select_by_workflow.py
"""Unit tests for FeatureRecord.select_by_workflow selector factory.

Uses a stub container that implements list_workflow_executions so the
factory can be tested without a live catalog. Live-catalog integration
coverage is in the parametrized symmetry suite (test_feature_values.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from deriva_ml.core.exceptions import DerivaMLException
from deriva_ml.feature import FeatureRecord


@dataclass
class _StubRecord:
    """Minimal structure matching FeatureRecord attribute access."""
    Execution: str | None
    RCT: str | None


class _StubContainer:
    def __init__(self, *, executions_for_workflow: dict[str, list[str]]):
        self._map = executions_for_workflow

    def list_workflow_executions(self, workflow: str) -> list[str]:
        if workflow not in self._map:
            raise DerivaMLException(f"Unknown workflow '{workflow}'")
        return list(self._map[workflow])


def test_select_by_workflow_requires_container_kwarg() -> None:
    """container is a required keyword-only argument."""
    with pytest.raises(TypeError):
        # positional container → TypeError
        FeatureRecord.select_by_workflow("WF-1", _StubContainer(executions_for_workflow={"WF-1": []}))  # type: ignore[misc]


def test_select_by_workflow_unknown_workflow_raises_at_construction() -> None:
    """Resolution is eager — unknown workflows fail at factory-call time, not at selection time."""
    container = _StubContainer(executions_for_workflow={"WF-known": ["EXE-1"]})
    with pytest.raises(DerivaMLException, match="Unknown workflow"):
        FeatureRecord.select_by_workflow("WF-unknown", container=container)


def test_select_by_workflow_picks_matching_record() -> None:
    container = _StubContainer(executions_for_workflow={"WF-1": ["EXE-A", "EXE-B"]})
    selector = FeatureRecord.select_by_workflow("WF-1", container=container)
    records = [
        _StubRecord(Execution="EXE-A", RCT="2026-01-01T00:00:00Z"),
        _StubRecord(Execution="EXE-Z", RCT="2026-02-01T00:00:00Z"),  # not in workflow
    ]
    chosen = selector(records)
    assert chosen.Execution == "EXE-A"


def test_select_by_workflow_picks_newest_among_matches() -> None:
    container = _StubContainer(executions_for_workflow={"WF-1": ["EXE-A", "EXE-B"]})
    selector = FeatureRecord.select_by_workflow("WF-1", container=container)
    records = [
        _StubRecord(Execution="EXE-A", RCT="2026-01-01T00:00:00Z"),
        _StubRecord(Execution="EXE-B", RCT="2026-03-01T00:00:00Z"),  # newest matching
    ]
    chosen = selector(records)
    assert chosen.Execution == "EXE-B"


def test_select_by_workflow_returns_none_when_no_match() -> None:
    """No record in the group matched the workflow → returns None (feature-absent semantics)."""
    container = _StubContainer(executions_for_workflow={"WF-1": ["EXE-A"]})
    selector = FeatureRecord.select_by_workflow("WF-1", container=container)
    records = [_StubRecord(Execution="EXE-Z", RCT="2026-02-01T00:00:00Z")]
    assert selector(records) is None


def test_select_by_workflow_resolves_executions_once() -> None:
    """list_workflow_executions is called once at construction, not per-group."""
    call_count = {"n": 0}

    class CountingContainer(_StubContainer):
        def list_workflow_executions(self, workflow):  # type: ignore[override]
            call_count["n"] += 1
            return super().list_workflow_executions(workflow)

    container = CountingContainer(executions_for_workflow={"WF-1": ["EXE-A"]})
    selector = FeatureRecord.select_by_workflow("WF-1", container=container)
    # Multiple selection calls
    records = [_StubRecord(Execution="EXE-A", RCT="r1")]
    for _ in range(10):
        selector(records)
    assert call_count["n"] == 1
```

- [ ] **Step 2: Run test to verify failure**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/feature/test_select_by_workflow.py -q`
Expected: FAIL — `FeatureRecord.select_by_workflow` classmethod does not yet exist.

- [ ] **Step 3: Implement the factory**

In `src/deriva_ml/feature.py`, after the `select_by_execution` staticmethod (around line 175) and before `select_first`, add:

```python
@classmethod
def select_by_workflow(
    cls,
    workflow: str,
    *,
    container,
) -> "Callable[[list[FeatureRecord]], Optional[FeatureRecord]]":
    """Return a selector that filters records by the executions of a workflow.

    This is a **factory**: call it once per workflow to get a reusable
    selector closure. The returned closure has signature
    ``(list[FeatureRecord]) -> FeatureRecord | None`` and can be passed
    as the ``selector`` argument to
    :meth:`~deriva_ml.interfaces.DatasetLike.feature_values`.

    **Eager resolution.** The expensive work —
    ``container.list_workflow_executions(workflow)`` — happens once, at
    factory-call time. The returned closure is a pure function over
    records and is cheap to invoke repeatedly. A workflow that doesn't
    exist raises at the factory call, not later at selection time.

    **Behavior when no records match.** If the group contains no record
    whose ``Execution`` RID is in the workflow's execution set, the
    selector returns ``None``. ``feature_values`` treats ``None`` as
    "feature absent for this target RID" and omits the target from its
    iterator. This is consistent with the documented selector contract
    for "no-match" cases.

    **Tie-breaking.** When multiple records match, ``select_newest`` is
    applied — the record with the most recent ``RCT`` wins.

    Args:
        workflow: A Workflow RID (e.g., ``"2-ABC1"``) or Workflow_Type
            name (e.g., ``"Training"``). Resolution is delegated to
            ``container.list_workflow_executions``, which implements the
            two-step lookup (RID first, then type name).
        container: **Required keyword-only.** A container that exposes
            ``list_workflow_executions(workflow) -> list[str]``. All three
            of ``DerivaML``, ``Dataset``, and ``DatasetBag`` qualify.

    Returns:
        A callable ``(list[FeatureRecord]) -> FeatureRecord | None``
        suitable for the ``selector`` kwarg of ``feature_values``.

    Raises:
        DerivaMLException: If ``workflow`` does not resolve in
            ``container`` (unknown RID and unknown Workflow_Type name).
        TypeError: If ``container`` is passed positionally.

    Example:
        Get the newest Glaucoma label per image created by the
        ``Glaucoma_Training_v2`` workflow, works identically online
        and on a downloaded bag::

            >>> from deriva_ml.feature import FeatureRecord
            >>> workflow = ml.lookup_workflow("Glaucoma_Training_v2")
            >>> sel = FeatureRecord.select_by_workflow(workflow, container=bag)
            >>> for rec in bag.feature_values(
            ...     "Image", "Glaucoma", selector=sel,
            ... ):
            ...     print(rec.Image, rec.Glaucoma)
    """
    # Eager resolution: fail-fast on unknown workflow, amortize the lookup
    # across all subsequent selection calls. Convert to a set for O(1)
    # membership testing inside the closure.
    execution_set = set(container.list_workflow_executions(workflow))

    def _selector(records: list["FeatureRecord"]) -> "FeatureRecord | None":
        matching = [r for r in records if r.Execution in execution_set]
        if not matching:
            return None
        return FeatureRecord.select_newest(matching)

    return _selector
```

Note: `Callable` and `Optional` must be importable. Add `from typing import Callable, Optional` if not already in the file's imports. The existing imports at line 19 include `Optional`. Add `Callable`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/feature/test_select_by_workflow.py -q`
Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/feature/test_select_by_workflow.py src/deriva_ml/feature.py
git commit -m "feat(feature): FeatureRecord.select_by_workflow selector factory (S2)"
```

---

## Task 3: Add `DerivaML.list_workflow_executions`

**Files:**
- Modify: `src/deriva_ml/core/mixins/feature.py` (new method, place near end of `FeatureMixin` class)
- Test: extend `tests/feature/test_features.py` with new tests (requires live catalog)

- [ ] **Step 1: Write the failing live-catalog test**

Add to `tests/feature/test_features.py` (existing file):

```python
def test_list_workflow_executions_returns_matching_rids(test_ml) -> None:
    """list_workflow_executions returns all execution RIDs that ran a given workflow."""
    # Fixture: create a workflow, run two executions against it, verify both RIDs returned.
    from deriva_ml.execution import ExecutionConfiguration
    wf = test_ml.add_workflow(
        name="S2_test_wf",
        url="https://example.com/s2_test",
        workflow_type="Test_Workflow",
        description="S2 list_workflow_executions coverage",
        checksum="a" * 64,
    )
    # Two executions against the same workflow
    cfg = ExecutionConfiguration(description="exec 1", workflow=wf)
    with test_ml.create_execution(cfg) as exe1:
        pass
    with test_ml.create_execution(cfg) as exe2:
        pass
    rids = test_ml.list_workflow_executions(wf)
    assert exe1.execution_rid in rids
    assert exe2.execution_rid in rids
    # Unique entries
    assert len(rids) == len(set(rids))


def test_list_workflow_executions_by_workflow_type_name(test_ml) -> None:
    """list_workflow_executions accepts a Workflow_Type name (not just an RID)."""
    wf = test_ml.add_workflow(
        name="S2_type_wf",
        url="https://example.com/s2_type",
        workflow_type="Test_Workflow",
        description="S2 workflow type name coverage",
        checksum="b" * 64,
    )
    from deriva_ml.execution import ExecutionConfiguration
    cfg = ExecutionConfiguration(description="exec", workflow=wf)
    with test_ml.create_execution(cfg) as exe:
        pass
    rids = test_ml.list_workflow_executions("Test_Workflow")
    assert exe.execution_rid in rids


def test_list_workflow_executions_unknown_raises(test_ml) -> None:
    """Unknown workflow → DerivaMLException."""
    from deriva_ml.core.exceptions import DerivaMLException
    with pytest.raises(DerivaMLException):
        test_ml.list_workflow_executions("nonexistent-workflow-xyz")
```

- [ ] **Step 2: Run test to verify it fails (needs DERIVA_HOST)**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/feature/test_features.py::test_list_workflow_executions_returns_matching_rids -q`
Expected: FAIL — method `list_workflow_executions` does not exist on `DerivaML`.

- [ ] **Step 3: Implement `list_workflow_executions` in `FeatureMixin`**

In `src/deriva_ml/core/mixins/feature.py`, replace the existing `select_by_workflow` method (lines 565-664) with `list_workflow_executions`. The new method reuses the existing lookup-by-RID-then-type-name logic but returns the full execution list:

```python
@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def list_workflow_executions(self, workflow: str) -> list[str]:
    """Return execution RIDs that ran the given workflow.

    The ``workflow`` argument resolves in two steps: first as a Workflow
    RID, and if that fails, as a Workflow_Type name. The returned list
    contains every execution RID for every workflow that matches.

    This method is the catalog-backed building block for
    ``FeatureRecord.select_by_workflow(workflow, container=ml)`` — it
    resolves the workflow's execution set once, and the selector closes
    over the result for cheap per-group membership testing.

    Entries are unique by construction (each execution runs one workflow).
    Consumers that need O(1) membership testing convert to ``set`` at the
    call site.

    Args:
        workflow: Workflow RID (e.g., ``"2-ABC1"``) or Workflow_Type name
            (e.g., ``"Training"``).

    Returns:
        List of execution RIDs, in insertion order. May be empty if the
        workflow exists but has no executions yet.

    Raises:
        DerivaMLException: If ``workflow`` does not resolve as a Workflow
            RID nor as a Workflow_Type name.

    Example:
        List all executions of a workflow and count them::

            >>> rids = ml.list_workflow_executions("Glaucoma_Training_v2")
            >>> print(f"{len(rids)} executions of this workflow")

        Use as the catalog-backed resolver for the selector factory::

            >>> from deriva_ml.feature import FeatureRecord
            >>> sel = FeatureRecord.select_by_workflow(
            ...     "Glaucoma_Training_v2", container=ml,
            ... )
    """
    # Try RID first
    try:
        wf = self.lookup_workflow(workflow)
        return [
            exec_record.execution_rid
            for exec_record in self.find_executions(workflow=wf)
        ]
    except DerivaMLException:
        pass

    # Fall back to Workflow_Type name
    pb = self.pathBuilder()
    wt_assoc = pb.schemas[self.ml_schema].Workflow_Workflow_Type
    matching_workflows = {
        row["Workflow"]
        for row in wt_assoc.filter(
            wt_assoc.Workflow_Type == workflow
        ).entities().fetch()
    }
    if not matching_workflows:
        raise DerivaMLException(
            f"No workflow resolved for '{workflow}' — tried as Workflow RID "
            f"and Workflow_Type name."
        )
    return [
        exec_record.execution_rid
        for exec_record in self.find_executions()
        if exec_record.workflow_rid in matching_workflows
    ]
```

Leave `select_by_workflow` in place for now — Task 10 retires it with a clear-error shim. That ordering ensures callers get a helpful message rather than silent breakage during the refactor.

- [ ] **Step 4: Run live-catalog tests**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/feature/test_features.py -k list_workflow -q --timeout=300`
Expected: 3 new tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/feature/test_features.py src/deriva_ml/core/mixins/feature.py
git commit -m "feat(feature): DerivaML.list_workflow_executions (S2)"
```

---

## Task 4: Add `DerivaML.feature_values` (online read, iterator)

**Files:**
- Modify: `src/deriva_ml/core/mixins/feature.py` (new method, place before `fetch_table_features` — which stays until Task 10)
- Test: `tests/feature/test_feature_values.py` (new file, online-only tests in this task; symmetry extension in Task 8)

- [ ] **Step 1: Write the failing test**

Create `tests/feature/test_feature_values.py`:

```python
"""Tests for the unified feature_values read surface.

This task (Task 4) covers DerivaML online reads. The parametrized symmetry
suite across DerivaML / Dataset / DatasetBag is added in Task 8 once all
three containers have the method.
"""
from __future__ import annotations

import pytest

from deriva_ml.core.exceptions import DerivaMLException, DerivaMLNotFoundError
from deriva_ml.feature import FeatureRecord


def test_feature_values_yields_feature_records(test_ml_with_feature) -> None:
    """feature_values yields FeatureRecord instances with the expected attrs."""
    records = list(test_ml_with_feature.ml.feature_values(
        "Image", test_ml_with_feature.feature_name,
    ))
    assert len(records) > 0
    rec = records[0]
    assert isinstance(rec, FeatureRecord)
    # Target RID attribute present and non-empty
    assert getattr(rec, "Image", None)
    # Feature_Name populated
    assert rec.Feature_Name == test_ml_with_feature.feature_name


def test_feature_values_unknown_feature_raises(test_ml) -> None:
    with pytest.raises(DerivaMLException):
        list(test_ml.feature_values("Image", "no_such_feature_xyz"))


def test_feature_values_is_iterable_not_list(test_ml_with_feature) -> None:
    """Return type is an iterator (streaming), not a materialized list."""
    import collections.abc
    result = test_ml_with_feature.ml.feature_values(
        "Image", test_ml_with_feature.feature_name,
    )
    assert isinstance(result, collections.abc.Iterator)


def test_feature_values_with_select_newest(test_ml_with_feature_multi) -> None:
    """Selector reduces multi-value groups to one record per target RID."""
    records = list(test_ml_with_feature_multi.ml.feature_values(
        "Image",
        test_ml_with_feature_multi.feature_name,
        selector=FeatureRecord.select_newest,
    ))
    # After selector: one record per target Image
    rids = [r.Image for r in records]
    assert len(rids) == len(set(rids))


def test_feature_values_selector_returning_none_omits_target(test_ml_with_feature) -> None:
    """Selector returning None removes that target from the iterator."""
    def reject_all(records):
        return None
    records = list(test_ml_with_feature.ml.feature_values(
        "Image", test_ml_with_feature.feature_name, selector=reject_all,
    ))
    assert records == []
```

Two fixtures referenced (`test_ml_with_feature`, `test_ml_with_feature_multi`) must be added to `tests/feature/conftest.py` or `tests/conftest.py` — they produce a catalog with a created feature and one or several feature values. If equivalent fixtures exist in `tests/feature/test_features.py` or `tests/feature/test_fetch_table_features.py`, reuse their setup; otherwise factor them out into a shared conftest. Use real feature creation via `ml.create_feature(...)` + `exe.add_features(...)` — do not mock.

- [ ] **Step 2: Run tests to verify failure**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/feature/test_feature_values.py -q --timeout=300`
Expected: FAIL — `feature_values` method does not exist; fixtures may also be missing.

- [ ] **Step 3: Implement `feature_values` in `FeatureMixin`**

In `src/deriva_ml/core/mixins/feature.py`, add before `fetch_table_features` (around line 382):

```python
@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def feature_values(
    self,
    table: Table | str,
    feature_name: str,
    selector: Callable[[list[FeatureRecord]], FeatureRecord | None] | None = None,
) -> Iterable[FeatureRecord]:
    """Yield feature values for a single feature, one record per target RID.

    Returns an iterator of typed ``FeatureRecord`` instances. Each record is
    wide in shape — target RID, all value columns (vocab terms, asset
    references, metadata columns), and provenance columns (``Execution``,
    ``RCT``) — exposed as typed attributes.

    When a ``selector`` is provided, records are grouped by target RID and
    the selector collapses each group to a single survivor. Target RIDs
    whose group's selector returns ``None`` are omitted. When no selector
    is provided, every raw record is yielded — multiple records per target
    RID are possible.

    This method has identical signatures and semantics across ``DerivaML``,
    ``Dataset``, and ``DatasetBag``. The bag implementation reads from a
    per-feature denormalization cache populated on first access; subsequent
    calls are cheap.

    Args:
        table: Target table the feature is defined on (name or Table).
        feature_name: Name of the feature to read.
        selector: Optional callable
            ``(list[FeatureRecord]) -> FeatureRecord | None`` used to
            reduce multi-value groups. Built-ins include
            ``FeatureRecord.select_newest``,
            ``FeatureRecord.select_first``, and the factory
            ``FeatureRecord.select_by_workflow(workflow, container=...)``.
            Return ``None`` from a selector to omit that target RID.

    Returns:
        Iterator of ``FeatureRecord`` — one record per target RID after
        selector reduction, or all raw records if no selector.

    Raises:
        DerivaMLTableNotFound: ``table`` does not exist.
        DerivaMLException: ``feature_name`` is not a feature on ``table``.

    Example:
        Get the newest Glaucoma label per image::

            >>> from deriva_ml.feature import FeatureRecord
            >>> for rec in ml.feature_values(
            ...     "Image", "Glaucoma", selector=FeatureRecord.select_newest,
            ... ):
            ...     print(f"{rec.Image}: {rec.Glaucoma} (by {rec.Execution})")

        Filter by a specific workflow — works identically on a downloaded bag::

            >>> workflow = ml.lookup_workflow("Glaucoma_Training_v2")
            >>> sel = FeatureRecord.select_by_workflow(workflow, container=ml)
            >>> labels = [r.Glaucoma for r in ml.feature_values(
            ...     "Image", "Glaucoma", selector=sel,
            ... )]

        Convert to a pandas DataFrame when needed::

            >>> import pandas as pd
            >>> df = pd.DataFrame(
            ...     r.model_dump()
            ...     for r in ml.feature_values("Image", "Glaucoma")
            ... )
    """
    table_obj = self.model.name_to_table(table)
    feat = self.lookup_feature(table_obj, feature_name)
    record_class = feat.feature_record_class()
    field_names = set(record_class.model_fields.keys())
    target_col = feat.target_table.name

    # Fetch raw rows via datapath
    pb = self.pathBuilder()
    raw_values = (
        pb.schemas[feat.feature_table.schema.name]
        .tables[feat.feature_table.name]
        .entities()
        .fetch()
    )

    # Materialize to FeatureRecord instances
    records: list[FeatureRecord] = [
        record_class(**{k: v for k, v in raw.items() if k in field_names})
        for raw in raw_values
    ]

    if selector is None:
        # No reduction — yield everything.
        yield from records
        return

    # Group by target RID, apply selector, skip None results.
    grouped: dict[str, list[FeatureRecord]] = defaultdict(list)
    for rec in records:
        target_rid = getattr(rec, target_col, None)
        if target_rid is not None:
            grouped[target_rid].append(rec)

    for group in grouped.values():
        chosen = selector(group) if len(group) > 1 else group[0]
        if chosen is not None:
            yield chosen
```

Note: `defaultdict` is already imported at the top of the file (line 12). `Callable`, `Iterable` are already imported (line 14).

- [ ] **Step 4: Run tests to verify they pass**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/feature/test_feature_values.py -q --timeout=300`
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/feature/test_feature_values.py src/deriva_ml/core/mixins/feature.py
git commit -m "feat(feature): DerivaML.feature_values — iterator read surface (S2)"
```

---

## Task 5: Add `Dataset.feature_values`, `Dataset.lookup_feature`, `Dataset.list_workflow_executions`

**Files:**
- Modify: `src/deriva_ml/dataset/dataset.py` (methods placed near existing `find_features`, around line 455)
- Test: extend `tests/feature/test_feature_values.py` with Dataset-scoped tests

- [ ] **Step 1: Write the failing tests**

Append to `tests/feature/test_feature_values.py`:

```python
def test_dataset_feature_values_filters_to_members(catalog_with_feature_and_dataset) -> None:
    """Dataset.feature_values yields only records whose target RID is in dataset members."""
    fx = catalog_with_feature_and_dataset
    dataset = fx.dataset
    feature_name = fx.feature_name
    member_rids = set(dataset.list_members(fx.target_table))
    records = list(dataset.feature_values(fx.target_table, feature_name))
    assert all(getattr(r, fx.target_table) in member_rids for r in records)


def test_dataset_lookup_feature_delegates_to_ml(catalog_with_feature_and_dataset) -> None:
    """Dataset.lookup_feature returns a Feature object identical to ml.lookup_feature."""
    fx = catalog_with_feature_and_dataset
    ds_feat = fx.dataset.lookup_feature(fx.target_table, fx.feature_name)
    ml_feat = fx.ml.lookup_feature(fx.target_table, fx.feature_name)
    assert ds_feat.feature_name == ml_feat.feature_name
    assert ds_feat.target_table.name == ml_feat.target_table.name


def test_dataset_list_workflow_executions_scopes_to_dataset(
    catalog_with_feature_and_dataset,
) -> None:
    """Dataset.list_workflow_executions returns only executions in the dataset's scope."""
    fx = catalog_with_feature_and_dataset
    ml_rids = set(fx.ml.list_workflow_executions(fx.workflow))
    ds_rids = set(fx.dataset.list_workflow_executions(fx.workflow))
    assert ds_rids.issubset(ml_rids)
```

The fixture `catalog_with_feature_and_dataset` builds a dataset containing a subset of feature-target RIDs. If similar scaffolding exists in `tests/dataset/`, factor it into a shared `tests/feature/conftest.py`. The fixture must expose `ml`, `dataset`, `target_table` (name), `feature_name`, `workflow` (RID) attributes.

- [ ] **Step 2: Run tests to verify failure**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/feature/test_feature_values.py::test_dataset_feature_values_filters_to_members -q --timeout=300`
Expected: FAIL — `Dataset.feature_values`, `Dataset.lookup_feature`, `Dataset.list_workflow_executions` do not exist.

- [ ] **Step 3: Implement the three methods on `Dataset`**

In `src/deriva_ml/dataset/dataset.py`, after the existing `find_features` method (around line 455), add:

```python
def feature_values(
    self,
    table: str | Table,
    feature_name: str,
    selector: Callable[[list[FeatureRecord]], FeatureRecord | None] | None = None,
) -> Iterable[FeatureRecord]:
    """Dataset-scoped feature values — identical signature to ``DerivaML.feature_values``.

    Yields only records whose target RID is a member of this dataset. The
    filtering is applied over the raw feature table query before selector
    reduction — a target RID outside the dataset's member set is never
    presented to the selector.

    See :meth:`deriva_ml.core.mixins.feature.FeatureMixin.feature_values`
    for the full contract (return type, selector semantics, exceptions).

    Example:
        >>> from deriva_ml.feature import FeatureRecord
        >>> for rec in dataset.feature_values(
        ...     "Image", "Glaucoma", selector=FeatureRecord.select_newest,
        ... ):
        ...     ...  # processing only images in this dataset
    """
    members = set(self.list_members(table))
    target_col = (
        table if isinstance(table, str) else table.name
    )  # target attr on each FeatureRecord has this name
    # Delegate to ml for the raw fetch, then filter.
    for rec in self._ml.feature_values(table, feature_name, selector=None):
        if getattr(rec, target_col, None) in members:
            yield rec if selector is None else None  # placeholder — selector applied below

    # NOTE: Implementing in-scope selector application requires grouping
    # post-filter. Replace the body above with the filter-then-group-then-select
    # pattern used in FeatureMixin.feature_values. See the reference
    # implementation below.
```

**Corrected implementation** (use this — the stub above makes the intent clear but is not complete):

```python
def feature_values(
    self,
    table: str | Table,
    feature_name: str,
    selector: Callable[[list[FeatureRecord]], FeatureRecord | None] | None = None,
) -> Iterable[FeatureRecord]:
    """Dataset-scoped feature values — identical signature to DerivaML.feature_values.

    Yields only records whose target RID is a member of this dataset.

    See DerivaML.feature_values docstring for the full contract.

    Example:
        >>> from deriva_ml.feature import FeatureRecord
        >>> records = list(dataset.feature_values(
        ...     "Image", "Glaucoma", selector=FeatureRecord.select_newest,
        ... ))
    """
    from collections import defaultdict

    members = set(self.list_members(table))
    target_col = table if isinstance(table, str) else table.name

    # Filter upstream raw records to dataset members
    raw_in_scope = [
        rec
        for rec in self._ml.feature_values(table, feature_name, selector=None)
        if getattr(rec, target_col, None) in members
    ]

    if selector is None:
        yield from raw_in_scope
        return

    grouped: dict[str, list[FeatureRecord]] = defaultdict(list)
    for rec in raw_in_scope:
        target_rid = getattr(rec, target_col, None)
        if target_rid is not None:
            grouped[target_rid].append(rec)

    for group in grouped.values():
        chosen = selector(group) if len(group) > 1 else group[0]
        if chosen is not None:
            yield chosen


def lookup_feature(self, table: str | Table, feature_name: str) -> "Feature":
    """Look up a Feature definition — delegates to the owning DerivaML.

    Identical signature and return to ``DerivaML.lookup_feature``. Provided
    for API symmetry so dataset-scoped code does not need to reach back
    through ``self._ml``.

    Example:
        >>> feat = dataset.lookup_feature("Image", "Glaucoma")
        >>> RecordClass = feat.feature_record_class()
    """
    return self._ml.lookup_feature(table, feature_name)


def list_workflow_executions(self, workflow: str) -> list[str]:
    """Return execution RIDs in this dataset's scope that ran the given workflow.

    Filters ``DerivaML.list_workflow_executions`` to executions that
    produced data in this dataset. Used by
    ``FeatureRecord.select_by_workflow(workflow, container=dataset)``.

    Args:
        workflow: Workflow RID or Workflow_Type name. See
            ``DerivaML.list_workflow_executions`` for the resolution rules.

    Returns:
        List of execution RIDs in this dataset's scope. May be empty.

    Example:
        >>> rids = dataset.list_workflow_executions("Glaucoma_Training_v2")
        >>> print(f"{len(rids)} training runs touched this dataset")
    """
    all_rids = set(self._ml.list_workflow_executions(workflow))
    # Scope: executions whose target RIDs intersect with dataset members.
    # Concretely: executions that produced any feature value whose target
    # is a member of this dataset.
    in_scope: list[str] = []
    for rid in all_rids:
        # NOTE: "in scope" is defined as "execution touched any dataset
        # member table row". This reuses the existing list_members logic.
        # If the dataset has no direct link to executions, this falls back
        # to the superset (ml_rids) — see the Dataset↔Execution spec for
        # the authoritative scope definition.
        if self._execution_touches_dataset(rid):
            in_scope.append(rid)
    return in_scope
```

**Scope definition caveat.** "Execution is in the dataset's scope" has two reasonable definitions:

1. **Strict:** the execution produced at least one feature value whose target RID is a dataset member (requires a query per execution).
2. **Permissive:** every execution returned by `ml.list_workflow_executions`, with downstream code applying target-RID filtering at selection time.

The implementer should prefer **Definition 2 (permissive)** for this task unless the existing `Dataset` class has a pre-existing helper for "executions that touched this dataset." Definition 2 is cheaper and still correct: the `FeatureRecord.select_by_workflow` selector closes over the full workflow execution set, and the `feature_values` iterator already filters to dataset members at the target-RID level. Any execution RID that lies outside the dataset's member set has no records that pass the member filter, so it never reduces the selector output. Implementing Definition 1 is a performance optimization for very large catalogs, out of scope here.

**Simplification:** replace the `list_workflow_executions` body above with a direct pass-through:

```python
def list_workflow_executions(self, workflow: str) -> list[str]:
    """Dataset-scoped list_workflow_executions — see DerivaML.list_workflow_executions.

    Current implementation returns the full workflow execution list from the
    catalog. Target-RID filtering at selection time ensures feature_values
    only yields records for executions whose output lands on dataset members.
    A stricter scope (executions whose outputs touch dataset members) is a
    performance optimization deferred to a later change.

    Example:
        >>> rids = dataset.list_workflow_executions("Glaucoma_Training_v2")
    """
    return self._ml.list_workflow_executions(workflow)
```

Use the simplified form. Remove the `_execution_touches_dataset` helper scaffolding above — we are not implementing strict scope in this task.

Update the test `test_dataset_list_workflow_executions_scopes_to_dataset` accordingly: it should assert `ds_rids.issubset(ml_rids)`, which the pass-through implementation satisfies trivially (equality). Keep the test as a guard against future divergence.

- [ ] **Step 4: Run tests to verify they pass**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/feature/test_feature_values.py -q --timeout=600`
Expected: all Dataset-scoped tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/feature/test_feature_values.py src/deriva_ml/dataset/dataset.py
git commit -m "feat(dataset): feature_values/lookup_feature/list_workflow_executions (S2)"
```

---

## Task 6: Build bag feature cache reader and wire `DatasetBag.feature_values` through it

**Files:**
- Create: `src/deriva_ml/dataset/bag_feature_cache.py`
- Modify: `src/deriva_ml/dataset/dataset_bag.py` (rewrite internals of `find_features`-adjacent feature read path; add `lookup_feature`, `list_workflow_executions`, `feature_values`; leave `fetch_table_features` / `list_feature_values` for Task 10 retirement)
- Test: `tests/dataset/test_bag_feature_cache.py` (new), extend `tests/feature/test_feature_values.py` with bag-side tests

- [ ] **Step 1: Write the failing tests**

Create `tests/dataset/test_bag_feature_cache.py`:

```python
"""Unit tests for the DatasetBag per-feature denormalization cache."""
from __future__ import annotations

from pathlib import Path

import pytest

from deriva_ml.core.exceptions import DerivaMLDataError
from deriva_ml.dataset.bag_feature_cache import BagFeatureCache


def test_first_access_populates_cache(materialized_bag_with_feature) -> None:
    """First call to fetch_feature_values populates the cache; second call reads it."""
    bag = materialized_bag_with_feature.bag
    cache = BagFeatureCache(bag)
    first = list(cache.fetch_feature_records("Image", materialized_bag_with_feature.feature_name))
    second = list(cache.fetch_feature_records("Image", materialized_bag_with_feature.feature_name))
    assert [r.model_dump() for r in first] == [r.model_dump() for r in second]


def test_cache_immutable_after_population(materialized_bag_with_feature) -> None:
    """Bags are immutable; subsequent reads return identical data."""
    # Assert that mutating the bag's SQLite feature source tables does not affect
    # already-returned cache reads. (Implementation: cache stores records in a new
    # sqlite table populated once; subsequent reads hit that table.)
    bag = materialized_bag_with_feature.bag
    cache = BagFeatureCache(bag)
    first = [r.model_dump() for r in cache.fetch_feature_records("Image", materialized_bag_with_feature.feature_name)]
    # Simulate tampering: drop a row from the source feature table directly.
    # (Omitted here — the guarantee is that the cache table is persistent and
    # not recomputed per call.)
    second = [r.model_dump() for r in cache.fetch_feature_records("Image", materialized_bag_with_feature.feature_name)]
    assert first == second


def test_cache_corrupt_raises_with_recovery_pointer(tmp_path: Path) -> None:
    """If cache init fails (corrupt SQLite, missing source table), raise with bag path."""
    # Build a minimal fake bag that has no feature table to trigger the error path.
    # Use pytest-style monkeypatch or a small helper to construct a BagFeatureCache
    # against a bag missing the requested feature; expect DerivaMLDataError.
    pytest.skip("Requires a fabricated bag-like directory — wire up in implementation.")
```

Append to `tests/feature/test_feature_values.py`:

```python
def test_bag_feature_values_matches_online(
    catalog_with_feature_and_materialized_bag,
) -> None:
    """bag.feature_values yields records whose content matches ml.feature_values."""
    fx = catalog_with_feature_and_materialized_bag
    online = sorted(
        [r.model_dump(exclude={"RCT", "RMT"}) for r in fx.ml.feature_values(
            fx.target_table, fx.feature_name,
        )],
        key=lambda d: d[fx.target_table],
    )
    offline = sorted(
        [r.model_dump(exclude={"RCT", "RMT"}) for r in fx.bag.feature_values(
            fx.target_table, fx.feature_name,
        )],
        key=lambda d: d[fx.target_table],
    )
    assert online == offline


def test_bag_lookup_feature_works_offline(materialized_bag_with_feature) -> None:
    """bag.lookup_feature works without a live catalog connection."""
    bag = materialized_bag_with_feature.bag
    feat = bag.lookup_feature("Image", materialized_bag_with_feature.feature_name)
    assert feat.feature_name == materialized_bag_with_feature.feature_name
    # feature_record_class is usable offline
    RecordClass = feat.feature_record_class()
    instance = RecordClass(Image="IMG-1", Feature_Name=materialized_bag_with_feature.feature_name)
    assert instance.Image == "IMG-1"
```

Fixtures:
- `materialized_bag_with_feature`: produces a downloaded bag containing a feature. Reuse `catalog_with_datasets` + `download_dataset_bag` scaffolding from existing tests.
- `catalog_with_feature_and_materialized_bag`: fixture exposing both an online `ml` and a matching offline `bag` for symmetry comparison.

- [ ] **Step 2: Run tests to verify failure**

Run: `DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/dataset/test_bag_feature_cache.py -q`
Expected: FAIL — `BagFeatureCache` does not exist.

- [ ] **Step 3: Implement `BagFeatureCache`**

Create `src/deriva_ml/dataset/bag_feature_cache.py`:

```python
"""Per-feature denormalization cache for DatasetBag.

Bags are immutable after materialization. This cache populates on first
access per feature and is the read path for ``DatasetBag.feature_values``.
The cache tables live in the bag's existing SQLite database.

See docs/superpowers/specs/2026-04-22-feature-api-consistency-design.md §Architecture.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable

from sqlalchemy import Column, MetaData, String, Table, Text, select
from sqlalchemy.orm import Session

from deriva_ml.core.exceptions import DerivaMLDataError, DerivaMLException
from deriva_ml.feature import FeatureRecord

if TYPE_CHECKING:
    from deriva_ml.dataset.dataset_bag import DatasetBag

logger = logging.getLogger(__name__)

_CACHE_TABLE_PREFIX = "_feature_cache_"


class BagFeatureCache:
    """Read feature records from a bag's per-feature denormalization cache.

    On first access per feature, the cache projects the bag's feature-table
    rows into a cache table keyed by target RID. Subsequent reads hit the
    cache table directly. The cache is bag-local and immutable — bags do not
    mutate after materialization.

    Example:
        >>> cache = BagFeatureCache(bag)
        >>> for rec in cache.fetch_feature_records("Image", "Glaucoma"):
        ...     print(rec.Image, rec.Glaucoma)
    """

    def __init__(self, bag: "DatasetBag") -> None:
        self._bag = bag
        self._engine = bag.engine
        self._metadata = MetaData()

    def fetch_feature_records(
        self, table: str, feature_name: str
    ) -> Iterable[FeatureRecord]:
        """Yield FeatureRecord instances for (table, feature_name) from cache.

        Populates the cache on first access. Subsequent calls read directly
        from the cache table.

        Raises:
            DerivaMLException: Feature not found on table.
            DerivaMLDataError: Cache corruption or missing source data.
        """
        feat = self._bag.model.lookup_feature(table, feature_name)
        record_class = feat.feature_record_class()
        cache_table_name = _CACHE_TABLE_PREFIX + feat.feature_table.name

        cache_table = self._ensure_cache_populated(feat, cache_table_name, record_class)

        # Read from cache
        field_names = set(record_class.model_fields.keys())
        with self._engine.connect() as conn:
            rows = conn.execute(select(cache_table)).mappings().all()
        for row in rows:
            filtered = {k: v for k, v in dict(row).items() if k in field_names}
            yield record_class(**filtered)

    def _ensure_cache_populated(
        self,
        feat,
        cache_table_name: str,
        record_class: type[FeatureRecord],
    ):
        """Ensure the cache table exists and is populated. Returns the table handle."""
        # Reflect existing cache table if present
        from sqlalchemy import inspect

        inspector = inspect(self._engine)
        if cache_table_name in inspector.get_table_names():
            cache_table = Table(
                cache_table_name, self._metadata, autoload_with=self._engine
            )
            return cache_table

        # Build the cache: source table → cache table
        try:
            source_table = self._bag.model.find_table(feat.feature_table.name)
        except DerivaMLException as e:
            raise DerivaMLDataError(
                f"Feature source table '{feat.feature_table.name}' missing from bag "
                f"at {self._bag.path}. Re-extract the bag."
            ) from e

        # Define cache table with columns matching the record_class fields
        cache_columns = [
            Column("_cache_rowid", String, primary_key=True),
        ]
        for name in record_class.model_fields.keys():
            # Store everything as TEXT — FeatureRecord Pydantic validation reifies types.
            cache_columns.append(Column(name, Text))
        cache_table = Table(cache_table_name, self._metadata, *cache_columns)
        cache_table.create(self._engine)

        # Populate from the source table
        with Session(self._engine) as session:
            sql = select(source_table)
            rows = session.execute(sql).mappings().all()
        field_names = set(record_class.model_fields.keys())
        with self._engine.begin() as conn:
            for i, raw in enumerate(rows):
                row_data = {k: v for k, v in dict(raw).items() if k in field_names}
                row_data["_cache_rowid"] = f"{i}"
                conn.execute(cache_table.insert().values(**row_data))
        logger.info(
            "BagFeatureCache: populated %s with %d rows", cache_table_name, len(rows)
        )
        return cache_table
```

- [ ] **Step 4: Wire `DatasetBag` methods**

In `src/deriva_ml/dataset/dataset_bag.py`, add after `find_features` (around line 462):

```python
def feature_values(
    self,
    table: str | Table,
    feature_name: str,
    selector: Callable[[list[FeatureRecord]], FeatureRecord | None] | None = None,
) -> Iterable[FeatureRecord]:
    """Offline feature values — identical signature to DerivaML.feature_values.

    Reads from the bag's per-feature denormalization cache (populated on
    first access). See DerivaML.feature_values for the full contract.

    Example:
        >>> from deriva_ml.feature import FeatureRecord
        >>> records = list(bag.feature_values(
        ...     "Image", "Glaucoma", selector=FeatureRecord.select_newest,
        ... ))
    """
    from collections import defaultdict
    from deriva_ml.dataset.bag_feature_cache import BagFeatureCache

    if not hasattr(self, "_feature_cache"):
        self._feature_cache = BagFeatureCache(self)
    target_col = table if isinstance(table, str) else table.name

    records = list(self._feature_cache.fetch_feature_records(target_col, feature_name))

    if selector is None:
        yield from records
        return

    grouped: dict[str, list[FeatureRecord]] = defaultdict(list)
    for rec in records:
        target_rid = getattr(rec, target_col, None)
        if target_rid is not None:
            grouped[target_rid].append(rec)
    for group in grouped.values():
        chosen = selector(group) if len(group) > 1 else group[0]
        if chosen is not None:
            yield chosen


def lookup_feature(self, table: str | Table, feature_name: str) -> "Feature":
    """Look up a feature definition from bag metadata — works offline.

    Returns a Feature object identical in shape to what
    ``DerivaML.lookup_feature`` returns. The Feature's
    ``feature_record_class()`` method works offline too — use it to
    construct FeatureRecord instances from bag data for later staging
    via ``exe.add_features`` when back online.

    Example:
        >>> feat = bag.lookup_feature("Image", "Glaucoma")
        >>> RecordClass = feat.feature_record_class()
        >>> record = RecordClass(Image="IMG-1", Glaucoma="Normal")
    """
    return self.model.lookup_feature(table, feature_name)


def list_workflow_executions(self, workflow: str) -> list[str]:
    """Offline list_workflow_executions — reads from bag SQLite.

    Returns execution RIDs present in the bag's Execution table that
    match the given workflow (RID or Workflow_Type name). Used by
    ``FeatureRecord.select_by_workflow(workflow, container=bag)``.

    Example:
        >>> rids = bag.list_workflow_executions("Glaucoma_Training_v2")
    """
    from sqlalchemy import select as sa_select, text
    from sqlalchemy.orm import Session

    # Try as Workflow RID first
    workflow_table = self.model.find_table("Workflow")
    execution_table = self.model.find_table("Execution")
    with Session(self.engine) as session:
        wf_rows = session.execute(
            sa_select(workflow_table).where(
                workflow_table.c.RID == workflow
            )
        ).mappings().all()

        if wf_rows:
            # Found as RID → return executions with this workflow
            rows = session.execute(
                sa_select(execution_table.c.RID).where(
                    execution_table.c.Workflow == workflow
                )
            ).all()
            return [r[0] for r in rows]

        # Fall back to Workflow_Type name
        # Workflow_Workflow_Type is the association table; look up all workflows of this type.
        wwt = self.model.find_table("Workflow_Workflow_Type")
        wf_of_type = [
            r[0] for r in session.execute(
                sa_select(wwt.c.Workflow).where(wwt.c.Workflow_Type == workflow)
            ).all()
        ]
        if not wf_of_type:
            from deriva_ml.core.exceptions import DerivaMLException
            raise DerivaMLException(
                f"No workflow resolved for '{workflow}' in bag — tried as "
                f"Workflow RID and Workflow_Type name."
            )
        rows = session.execute(
            sa_select(execution_table.c.RID).where(
                execution_table.c.Workflow.in_(wf_of_type)
            )
        ).all()
        return [r[0] for r in rows]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/test_bag_feature_cache.py tests/feature/test_feature_values.py -q --timeout=600`
Expected: all bag-side tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/deriva_ml/dataset/bag_feature_cache.py src/deriva_ml/dataset/dataset_bag.py tests/dataset/test_bag_feature_cache.py tests/feature/test_feature_values.py
git commit -m "feat(bag): feature_values via per-feature cache, lookup_feature offline (S2)"
```

---

## Task 7: Stage-and-flush `exe.add_features`

**Files:**
- Modify: `src/deriva_ml/execution/execution.py` (replace `add_features` body around line 847; add `_flush_staged_features`, resume detection)
- Test: extend `tests/execution/test_staged_features.py` with integration tests

- [ ] **Step 1: Write the failing tests**

Append to `tests/execution/test_staged_features.py`:

```python
def test_exe_add_features_stages_to_sqlite(test_ml, image_feature) -> None:
    """exe.add_features writes Pending rows to execution_state__feature_records, nothing to ermrest yet."""
    from deriva_ml.execution import ExecutionConfiguration

    cfg = ExecutionConfiguration(description="stage test", workflow=image_feature.workflow)
    with test_ml.create_execution(cfg) as exe:
        RecordClass = image_feature.record_class
        exe.add_features([
            RecordClass(Image=image_feature.image_rids[0], Image_Label="A"),
            RecordClass(Image=image_feature.image_rids[1], Image_Label="B"),
        ])
        # Pending in SQLite
        store = exe._manifest_store
        pending = store.list_pending_feature_records(exe.execution_rid)
        assert len(pending) == 2
        # Not in ermrest yet (query the feature table directly)
        pb = test_ml.pathBuilder()
        rows = pb.schemas[image_feature.schema].tables[image_feature.feature_table_name].entities().fetch()
        # Should have NO rows from this execution yet
        assert all(r.get("Execution") != exe.execution_rid for r in rows)
    # After __exit__: rows Uploaded and in ermrest
    pb = test_ml.pathBuilder()
    rows = list(pb.schemas[image_feature.schema].tables[image_feature.feature_table_name].entities().fetch())
    ours = [r for r in rows if r.get("Execution") == exe.execution_rid]
    assert len(ours) == 2


def test_exe_add_features_auto_fills_execution_rid(test_ml, image_feature) -> None:
    """Records without Execution set get it auto-filled from the execution context."""
    from deriva_ml.execution import ExecutionConfiguration

    cfg = ExecutionConfiguration(description="auto-fill test", workflow=image_feature.workflow)
    with test_ml.create_execution(cfg) as exe:
        RecordClass = image_feature.record_class
        # No Execution set — auto-fill should apply
        exe.add_features([RecordClass(Image=image_feature.image_rids[0], Image_Label="A")])
        pending = exe._manifest_store.list_pending_feature_records(exe.execution_rid)
        import json
        payload = json.loads(pending[0].record_json)
        assert payload["Execution"] == exe.execution_rid


def test_exe_add_features_mixed_feature_defs_raises(test_ml, image_feature, other_feature) -> None:
    """Records from different features → DerivaMLValidationError before staging."""
    from deriva_ml.core.exceptions import DerivaMLValidationError
    from deriva_ml.execution import ExecutionConfiguration

    cfg = ExecutionConfiguration(description="mixed test", workflow=image_feature.workflow)
    with test_ml.create_execution(cfg) as exe:
        mixed = [
            image_feature.record_class(Image=image_feature.image_rids[0], Image_Label="A"),
            other_feature.record_class(Image=other_feature.image_rids[0], Quality=5),
        ]
        with pytest.raises(DerivaMLValidationError):
            exe.add_features(mixed)
        # Nothing staged
        assert exe._manifest_store.list_feature_records(exe.execution_rid) == []


def test_exe_add_features_empty_raises(test_ml, image_feature) -> None:
    from deriva_ml.execution import ExecutionConfiguration

    cfg = ExecutionConfiguration(description="empty test", workflow=image_feature.workflow)
    with test_ml.create_execution(cfg) as exe:
        with pytest.raises(ValueError):
            exe.add_features([])


def test_flush_happens_after_assets(test_ml, image_feature_with_asset_column) -> None:
    """Feature flush order: assets first, then features.

    A feature whose asset column points at a staged asset must see the asset's
    uploaded RID in ermrest when the feature row is inserted. Verify by
    checking that asset upload completes before any feature INSERT attempts
    occur.
    """
    pytest.skip("Requires instrumenting upload_execution_outputs — see test plan")


def test_flush_failure_marks_group_failed_but_continues(
    test_ml, image_feature, failing_feature
) -> None:
    """If one feature group's insert fails, others still flush; DerivaMLUploadError summarizes."""
    pytest.skip("Requires injecting ermrest failure for one group — see test plan")
```

Fixtures needed:
- `image_feature`: object with `.workflow`, `.record_class`, `.schema`, `.feature_table_name`, `.image_rids` (list of target RIDs).
- `other_feature`: different feature on the same target, used for the mixed-definitions test.
- `image_feature_with_asset_column`: like `image_feature` but the feature has an asset column. Used for flush ordering.
- `failing_feature`: a feature rigged to fail ermrest insert (e.g., FK pointing at a non-existent row).

- [ ] **Step 2: Run tests to verify failure**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/test_staged_features.py -q --timeout=600`
Expected: FAIL — `exe.add_features` still writes through (not staging), and `_flush_staged_features` doesn't exist.

- [ ] **Step 3: Rewrite `Execution.add_features` to stage**

In `src/deriva_ml/execution/execution.py`, replace the body of `add_features` (around line 847) with:

```python
def add_features(self, features: list[FeatureRecord]) -> int:
    """Stage feature records for batch insertion on execution completion.

    Writes the records to the execution's SQLite ``execution_state__feature_records`` table
    with status ``Pending``. The records are not sent to ermrest immediately
    — they are flushed in a single batch, **after asset upload**, when the
    execution completes successfully. This integrates with the SQLite
    execution-state design so crash-resume works for feature writes without
    extra plumbing.

    Records with ``Execution`` unset are auto-filled with this execution's
    RID. All records in a single call must share one feature definition;
    mixing features raises ``DerivaMLValidationError`` and nothing is staged.

    **Provenance requirement.** This is the only way to write feature values
    — ``DerivaML.add_features`` is retired (see the retired-API error shims).
    For "admin fixup" cases, create a short-lived execution with an
    appropriate ``Workflow_Type`` (e.g. ``Manual_Correction``) and call
    ``exe.add_features`` inside it. The three-extra-lines give you a real
    audit trail, which is the point.

    Args:
        features: List of FeatureRecord instances to stage. All must share
            the same feature definition. Create instances via
            ``Feature.feature_record_class()``.

    Returns:
        Number of records staged.

    Raises:
        ValueError: features list is empty.
        DerivaMLValidationError: Records do not share a single feature
            definition.
        DerivaMLDataError: SQLite staging write failed.

    Example:
        >>> feature = ml.lookup_feature("Image", "Glaucoma")
        >>> RecordClass = feature.feature_record_class()
        >>> records = [
        ...     RecordClass(Image="IMG-1", Glaucoma="Normal"),
        ...     RecordClass(Image="IMG-2", Glaucoma="Severe"),
        ... ]
        >>> with ml.create_execution(cfg) as exe:
        ...     exe.add_features(records)     # staged, not yet in ermrest
        ...     # ... more work ...
        >>> # on __exit__: staged records flushed to ermrest after assets
    """
    from deriva_ml.core.exceptions import DerivaMLValidationError

    if not features:
        raise ValueError("features list must not be empty")

    # All records must share one feature definition
    feature_defs = {type(f).feature for f in features if type(f).feature is not None}
    if len(feature_defs) > 1:
        raise DerivaMLValidationError(
            f"add_features called with records from {len(feature_defs)} different "
            f"feature definitions; all records must share one feature."
        )

    # Auto-fill Execution RID on records that don't have it
    for f in features:
        if f.Execution is None:
            f.Execution = self.execution_rid

    # Stage to SQLite — durability boundary is the write-through here.
    feat_class = type(features[0])
    feat = feat_class.feature
    schema_name = feat.feature_table.schema.name
    table_name = feat.feature_table.name
    qualified = f"{schema_name}.{table_name}"
    feature_name = feat.feature_name
    target_table_name = feat.target_table.name

    count = 0
    for f in features:
        self._manifest_store.stage_feature_record(
            execution_rid=self.execution_rid,
            feature_table=qualified,
            feature_name=feature_name,
            target_table=target_table_name,
            record_json=f.model_dump_json(),
        )
        count += 1
    return count
```

**Note on `self._manifest_store` access.** The `Execution` class must already carry a reference to its `ManifestStore` (the §8 staged-asset code uses one). If it's named differently, adapt the attribute access. Grep: `manifest_store` in `execution/execution.py` to find the right name.

- [ ] **Step 4: Add `_flush_staged_features`**

In the same file, near the existing `upload_execution_outputs` / asset-flush code (search for `_flush_` or `upload_assets`), add:

```python
def _flush_staged_features(self) -> None:
    """Flush all Pending staged-feature rows to ermrest.

    Called from ``upload_execution_outputs()`` **after** staged-asset
    upload so feature rows referencing assets see their uploaded RIDs
    in ermrest. Groups rows by ``feature_table`` and batch-inserts each
    group via the datapath. Per-group failures mark those rows ``Failed``
    without aborting the other groups; an overall ``DerivaMLUploadError``
    is raised at the end if any failure occurred.

    See docs/superpowers/specs/2026-04-22-feature-api-consistency-design.md
    §Data flow — flush order.
    """
    import json
    from collections import defaultdict

    from deriva_ml.core.exceptions import DerivaMLUploadError

    pending = self._manifest_store.list_pending_feature_records(self.execution_rid)
    if not pending:
        return

    # Group by feature_table for batch insertion
    groups: dict[str, list] = defaultdict(list)
    for row in pending:
        groups[row.feature_table].append(row)

    failures: list[str] = []
    for qualified, rows in groups.items():
        schema_name, table_name = qualified.split(".", 1)
        feature_name = rows[0].feature_name
        target_table = rows[0].target_table   # carried on the staged row — no re-resolve
        try:
            feat = self._ml_object.lookup_feature(target_table, feature_name)
            record_class = feat.feature_record_class()
            payloads = [
                record_class(**json.loads(r.record_json)).model_dump()
                for r in rows
            ]

            # MIGRATED ASSET-COLUMN REWRITE (was in _update_feature_table):
            # Any payload value in an asset column that is a local filename
            # must be rewritten to the uploaded asset's RID before insert.
            # asset_rid_by_filename is populated during the asset flush that
            # runs earlier in upload_execution_outputs (returned by
            # _upload_assets as {asset_table: [AssetFilePath(...)]}).
            # Implementer: pass that map into _flush_staged_features as an
            # argument, or store on self during flush.
            for p in payloads:
                for asset_col in feat.asset_columns:
                    val = p.get(asset_col.name)
                    if isinstance(val, (str, Path)):
                        rid = self._resolve_asset_rid(val, asset_col)
                        if rid is not None:
                            p[asset_col.name] = rid

            pb = self._ml_object.pathBuilder()
            pb.schemas[schema_name].tables[table_name].insert(payloads)
            # Success — mark all rows uploaded
            for r in rows:
                self._manifest_store.mark_feature_record_uploaded(r.stage_id)
        except Exception as e:  # noqa: BLE001 — capture broad, propagate summary
            error_msg = f"{type(e).__name__}: {e}"
            for r in rows:
                self._manifest_store.mark_feature_record_failed(
                    r.stage_id, error=error_msg
                )
            failures.append(f"{qualified} ({len(rows)} records): {error_msg}")

    if failures:
        raise DerivaMLUploadError(
            f"Feature flush failed for {len(failures)} feature table(s): "
            + "; ".join(failures)
        )
```

**Asset-column rewriting — migration from `_update_feature_table`.** The old file-based path had logic in `_update_feature_table` (around `execution.py:1200-1207`) to remap asset-column values from local filenames to uploaded asset RIDs. The concrete mechanism there uses `asset_map` returned from the asset upload. When moving this logic to `_flush_staged_features`:

1. Capture the asset-upload return value in `upload_execution_outputs()` (call it `asset_map`).
2. Pass `asset_map` to `_flush_staged_features(asset_map=asset_map)` — or store on `self` immediately after the asset flush completes.
3. Inside `_flush_staged_features`, `_resolve_asset_rid(val, asset_col)` looks up the filename in `asset_map[asset_col.pk_table.name]` and returns the uploaded `asset_rid`.

The implementer should read the existing `_update_feature_table` body (around `execution.py` line 1100-1200) for the exact remapping contract (handles `Path` objects, bare filenames, and already-RID values) and port it verbatim into `_resolve_asset_rid`.

- [ ] **Step 5: Wire flush into upload path**

Find where staged assets are flushed during `upload_execution_outputs` (or equivalent — search for `_flush_assets` or `upload_directory`). Add a call to `self._flush_staged_features()` **after** the asset flush succeeds.

- [ ] **Step 6: Run tests to verify they pass**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/test_staged_features.py -q --timeout=600`
Expected: the four non-skipped tests pass. Skipped tests (asset ordering, per-group failure) are Task 11.

- [ ] **Step 7: Commit**

```bash
git add src/deriva_ml/execution/execution.py tests/execution/test_staged_features.py src/deriva_ml/local_db/manifest_store.py
git commit -m "feat(execution): stage-and-flush add_features, wired to upload path (S2)"
```

---

## Task 8: Parametrized symmetry compliance suite

**Files:**
- Modify: `tests/feature/test_feature_values.py` (add parametrized test class)
- Create: `tests/feature/conftest.py` (shared fixtures if not already)

- [ ] **Step 1: Write the parametrized tests**

Append to `tests/feature/test_feature_values.py`:

```python
# -----------------------------------------------------------------------------
# Parametrized symmetry suite — runs the same assertions across all three
# container types. Any future container claiming feature capability must pass
# this suite as its acceptance test.
# -----------------------------------------------------------------------------


@pytest.fixture(params=["ml", "dataset", "bag"])
def feature_container(request, feature_symmetry_fixture):
    """Parametrize across DerivaML, Dataset, DatasetBag for symmetry tests."""
    return feature_symmetry_fixture.by_container_kind(request.param)


class TestFeatureValuesSymmetry:
    """Same assertions, three containers. The symmetry contract."""

    def test_find_features_returns_matching_definitions(self, feature_container):
        features = list(feature_container.container.find_features(feature_container.target_table))
        names = {f.feature_name for f in features}
        assert feature_container.feature_name in names

    def test_feature_values_yields_expected_records(self, feature_container):
        records = sorted(
            [
                r.model_dump(exclude={"RCT", "RMT"})
                for r in feature_container.container.feature_values(
                    feature_container.target_table, feature_container.feature_name
                )
            ],
            key=lambda d: d[feature_container.target_table],
        )
        expected = feature_container.expected_records_sorted
        assert records == expected

    def test_feature_values_with_selector_matches(self, feature_container):
        from deriva_ml.feature import FeatureRecord
        records = list(feature_container.container.feature_values(
            feature_container.target_table,
            feature_container.feature_name,
            selector=FeatureRecord.select_newest,
        ))
        rids = [getattr(r, feature_container.target_table) for r in records]
        assert len(rids) == len(set(rids))  # one per target

    def test_lookup_feature_returns_usable_record_class(self, feature_container):
        feat = feature_container.container.lookup_feature(
            feature_container.target_table, feature_container.feature_name
        )
        RecordClass = feat.feature_record_class()
        # Usable constructor
        instance = RecordClass(
            **{feature_container.target_table: "TEST-RID"},
            Feature_Name=feature_container.feature_name,
        )
        assert getattr(instance, feature_container.target_table) == "TEST-RID"

    def test_list_workflow_executions_matches(self, feature_container):
        rids = feature_container.container.list_workflow_executions(
            feature_container.workflow
        )
        # Order-independent comparison
        assert set(rids) == feature_container.expected_workflow_executions
```

The `feature_symmetry_fixture` must set up: online catalog with a feature + values, a dataset scoping a subset of the feature's target RIDs, and a materialized bag downloaded from the catalog. The `by_container_kind("ml"|"dataset"|"bag")` helper returns an object with `.container`, `.target_table`, `.feature_name`, `.workflow`, `.expected_records_sorted`, `.expected_workflow_executions`.

- [ ] **Step 2: Add the shared fixture to `tests/feature/conftest.py`**

Create or extend `tests/feature/conftest.py`:

```python
"""Shared fixtures for feature test suite."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest


@dataclass
class SymmetryContainer:
    container: Any
    target_table: str
    feature_name: str
    workflow: str
    expected_records_sorted: list[dict]
    expected_workflow_executions: set[str]


@dataclass
class FeatureSymmetryFixture:
    ml: Any
    dataset: Any
    bag: Any
    target_table: str
    feature_name: str
    workflow: str
    expected_records_sorted: list[dict]
    expected_workflow_executions: set[str]

    def by_container_kind(self, kind: str) -> SymmetryContainer:
        container = {"ml": self.ml, "dataset": self.dataset, "bag": self.bag}[kind]
        # Datasets filter to members; expected_records must be the dataset-scoped
        # subset when kind == "dataset". If the fixture is built so that ALL
        # feature values are for members of the dataset, the expected records
        # are identical across kinds.
        return SymmetryContainer(
            container=container,
            target_table=self.target_table,
            feature_name=self.feature_name,
            workflow=self.workflow,
            expected_records_sorted=self.expected_records_sorted,
            expected_workflow_executions=self.expected_workflow_executions,
        )


@pytest.fixture
def feature_symmetry_fixture(test_ml, catalog_with_datasets):
    """Build a catalog + dataset + materialized bag with one feature populated.

    All feature values target RIDs that are members of the dataset, so the
    expected records are identical in the online/dataset/bag containers.
    """
    # Implementation: create a feature, add values via exe.add_features,
    # create a dataset containing all target RIDs, materialize a bag.
    # Return a FeatureSymmetryFixture with all handles wired up.
    pytest.skip("Fixture skeleton — fill in with concrete catalog setup")
```

The fixture body is scaffolding for the implementer to fill in against the existing `catalog_with_datasets` infrastructure. Keep the skip until the implementer has a working shape — then remove.

- [ ] **Step 3: Run tests to verify they pass**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/feature/test_feature_values.py::TestFeatureValuesSymmetry -q --timeout=600`
Expected: all 5 tests × 3 parametrizations pass (15 tests).

- [ ] **Step 4: Commit**

```bash
git add tests/feature/conftest.py tests/feature/test_feature_values.py
git commit -m "test(feature): parametrized symmetry compliance suite (S2)"
```

---

## Task 9: Offline-to-online write cycle

**Files:**
- Test: append to `tests/feature/test_feature_values.py`

- [ ] **Step 1: Write the test**

```python
def test_offline_construct_records_online_stage(
    catalog_with_feature_and_materialized_bag, test_ml,
) -> None:
    """Construct FeatureRecord instances from a bag (offline), stage via live exe."""
    fx = catalog_with_feature_and_materialized_bag
    # Offline: use bag.lookup_feature + feature_record_class (no catalog connection here)
    feat = fx.bag.lookup_feature(fx.target_table, fx.feature_name)
    RecordClass = feat.feature_record_class()
    records = [
        RecordClass(**{fx.target_table: rid, fx.value_column: f"val-{rid}"})
        for rid in fx.target_rids[:3]
    ]
    # Online: hand to a live execution
    from deriva_ml.execution import ExecutionConfiguration
    cfg = ExecutionConfiguration(description="offline-to-online", workflow=fx.workflow)
    with test_ml.create_execution(cfg) as exe:
        count = exe.add_features(records)
        assert count == 3
    # Verify round-trip
    written = list(test_ml.feature_values(fx.target_table, fx.feature_name))
    ours = [r for r in written if r.Execution == exe.execution_rid]
    assert len(ours) == 3
```

- [ ] **Step 2: Run test to verify it passes**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/feature/test_feature_values.py::test_offline_construct_records_online_stage -q --timeout=600`
Expected: pass. If it fails, the fixture needs extension — add the `.value_column` and `.target_rids` attributes.

- [ ] **Step 3: Commit**

```bash
git add tests/feature/test_feature_values.py tests/feature/conftest.py
git commit -m "test(feature): offline-to-online write cycle round-trip (S2)"
```

---

## Task 10: Retire old APIs with clear-error shims

**Files:**
- Modify: `src/deriva_ml/core/mixins/feature.py` (replace bodies of `add_features`, `fetch_table_features`, `list_feature_values`, `select_by_workflow` with raising shims; optionally delete instead — see decision below)
- Modify: `src/deriva_ml/dataset/dataset_bag.py` (same for `fetch_table_features`, `list_feature_values`)
- Modify: `src/deriva_ml/asset/asset.py` (retire `Asset.list_feature_values`)
- Modify: `src/deriva_ml/interfaces.py` (update protocols)
- Test: `tests/feature/test_retired_apis.py` (new)

**Decision: delete or raise?** The spec says "no deprecation shims" and "retired APIs raise a clear error pointing at the replacement." That's a raising shim — the method exists, calling it raises `DerivaMLException`. This is kinder than a rename (stack traces show exactly what the caller asked for) and still forces migration.

- [ ] **Step 1: Write the failing test**

Create `tests/feature/test_retired_apis.py`:

```python
"""Retired-API shims — each raises a DerivaMLException with a replacement pointer."""
from __future__ import annotations

import pytest

from deriva_ml.core.exceptions import DerivaMLException


def test_ml_add_features_raises_with_pointer(test_ml) -> None:
    with pytest.raises(DerivaMLException, match=r"exe\.add_features"):
        test_ml.add_features([])


def test_ml_fetch_table_features_raises_with_pointer(test_ml) -> None:
    with pytest.raises(DerivaMLException, match=r"feature_values|Denormalizer"):
        test_ml.fetch_table_features("Image")


def test_ml_list_feature_values_raises_with_pointer(test_ml) -> None:
    with pytest.raises(DerivaMLException, match=r"feature_values"):
        test_ml.list_feature_values("Image", "x")


def test_ml_select_by_workflow_raises_with_pointer(test_ml) -> None:
    with pytest.raises(DerivaMLException, match=r"select_by_workflow.*container"):
        test_ml.select_by_workflow([], "wf")


def test_bag_fetch_table_features_raises(materialized_bag) -> None:
    with pytest.raises(DerivaMLException, match=r"feature_values|Denormalizer"):
        materialized_bag.fetch_table_features("Image")


def test_bag_list_feature_values_raises(materialized_bag) -> None:
    with pytest.raises(DerivaMLException, match=r"feature_values"):
        materialized_bag.list_feature_values("Image", "x")
```

- [ ] **Step 2: Run to verify failure**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/feature/test_retired_apis.py -q --timeout=300`
Expected: FAIL — methods still work or have different signatures.

- [ ] **Step 3: Replace method bodies with raising shims**

In `src/deriva_ml/core/mixins/feature.py`:

```python
def add_features(self, features: list[FeatureRecord]) -> int:
    """Retired in S2 — use ``exe.add_features`` within an execution context.

    All feature writes now require provenance. Create a short-lived execution
    if you are doing admin fixup:

        >>> with ml.create_execution(cfg) as exe:
        ...     exe.add_features(records)

    See docs/superpowers/specs/2026-04-22-feature-api-consistency-design.md.
    """
    raise DerivaMLException(
        "ml.add_features() has been removed. Use exe.add_features() within an "
        "execution context — all feature writes now require provenance. See "
        "docs/superpowers/specs/2026-04-22-feature-api-consistency-design.md."
    )


def fetch_table_features(self, *args, **kwargs) -> "Any":
    """Retired in S2 — use ``feature_values(table, name)`` or ``Denormalizer``."""
    raise DerivaMLException(
        "fetch_table_features() has been removed. Use feature_values(table, feature_name) "
        "for single-feature access, or Denormalizer for multi-table joins. See "
        "docs/superpowers/specs/2026-04-22-feature-api-consistency-design.md."
    )


def list_feature_values(self, *args, **kwargs) -> "Any":
    """Retired in S2 — renamed to ``feature_values`` with the same signature."""
    raise DerivaMLException(
        "list_feature_values() has been renamed to feature_values() with the same "
        "signature. See docs/superpowers/specs/2026-04-22-feature-api-consistency-design.md."
    )


def select_by_workflow(self, *args, **kwargs) -> "Any":
    """Retired in S2 — use ``FeatureRecord.select_by_workflow(workflow, container=...)``."""
    raise DerivaMLException(
        "ml.select_by_workflow() is now a selector factory. Use "
        "FeatureRecord.select_by_workflow(workflow, container=...) and pass as "
        "the selector= argument to feature_values. See "
        "docs/superpowers/specs/2026-04-22-feature-api-consistency-design.md."
    )
```

In `src/deriva_ml/dataset/dataset_bag.py`, replace the bodies of `fetch_table_features` and `list_feature_values` with the same `raise DerivaMLException(...)` shims.

In `src/deriva_ml/asset/asset.py`:
- Find `list_feature_values` around line 227 and apply the same shim.
- Find `find_features` around line 208 — this one stays (it's a real feature-discovery method on `Asset`), but confirm it uses `model.find_features` consistently.

- [ ] **Step 4: Update `interfaces.py`**

In `src/deriva_ml/interfaces.py`:

1. Around line 214 (the `DatasetLike` protocol): keep `find_features(table)`, add `feature_values(table, feature_name, selector=...)`, `lookup_feature(table, feature_name)`, `list_workflow_executions(workflow)`.
2. Around line 543 (`list_feature_values` in `AssetLike` or similar protocol): remove this signature.
3. Around line 896 (`add_features` on some writable protocol): remove from the read-side protocol but keep on the Execution protocol (writes belong to `Execution`, not the container).
4. Update all matching protocols (`DerivaMLCatalogReader`, `DerivaMLCatalog`, `WritableDataset`) for consistency.

- [ ] **Step 5: Run tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/feature/test_retired_apis.py -q --timeout=300
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q
```
Expected: retired-API tests pass; baseline unit tests still pass (modulo the interface updates — any test calling the retired methods needs updating first).

- [ ] **Step 6: Fix any test fallout**

The old `tests/feature/test_fetch_table_features.py` will break because all its tests exercise `fetch_table_features`. Rewrite those tests against `feature_values` + `Denormalizer` as appropriate, or delete tests that become redundant. Do the same sweep for `list_feature_values` callers in the test suite:

```bash
grep -rn "fetch_table_features\|list_feature_values\|select_by_workflow\|ml\.add_features" tests/
```

Update each call site to the new API.

- [ ] **Step 7: Run full fast suite**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q
```
Expected: still green (439+ passing).

- [ ] **Step 8: Commit**

```bash
git add src/deriva_ml/ tests/feature/ tests/
git commit -m "refactor(feature): retire add_features, fetch_table_features, list_feature_values, select_by_workflow (S2)"
```

---

## Task 11: Flush ordering, failure isolation, and crash-resume

**Files:**
- Modify: `src/deriva_ml/execution/execution.py` (ordering, already wired in Task 7; add resume detection)
- Test: extend `tests/execution/test_staged_features.py`

- [ ] **Step 1: Unskip and complete the tests from Task 7**

In `tests/execution/test_staged_features.py`, replace the two skips with real tests:

```python
def test_flush_happens_after_assets(test_ml, image_feature_with_asset_column) -> None:
    """Feature flush order: asset upload completes before any feature INSERT."""
    fx = image_feature_with_asset_column
    # Track ordering via a patched insert hook
    call_log: list[str] = []

    # Monkeypatch the asset upload path to log "asset_done" at completion,
    # and the feature batch insert to log "feature_insert" at call.
    # Then assert call_log is ["asset_done", ..., "feature_insert", ...].
    pytest.skip("Wire monkeypatch against upload_execution_outputs internals")


def test_flush_failure_marks_group_failed_but_continues(
    test_ml, image_feature, failing_feature,
) -> None:
    """One group's flush failure does not abort other groups."""
    # Stage records for both features — failing_feature has an invalid FK.
    # After upload: image_feature's rows are in ermrest; failing_feature's rows are Failed.
    # DerivaMLUploadError was raised.
    pytest.skip("Wire a feature rigged to fail ermrest validation")
```

The implementer writes these against the real execution machinery. They are integration tests — they run live.

- [ ] **Step 2: Add crash-resume test**

```python
def test_crash_before_flush_resumes_without_duplicates(
    test_ml, image_feature,
) -> None:
    """Simulate crash after staging but before flush; resume completes the flush."""
    from deriva_ml.execution import ExecutionConfiguration

    cfg = ExecutionConfiguration(description="crash test", workflow=image_feature.workflow)

    # First run: stage records but exit before the flush path executes
    # (raise inside the `with` block AFTER add_features to skip normal __exit__ flush).
    exe_rid = None
    try:
        with test_ml.create_execution(cfg) as exe:
            exe_rid = exe.execution_rid
            RecordClass = image_feature.record_class
            exe.add_features([RecordClass(Image=image_feature.image_rids[0], Image_Label="A")])
            raise RuntimeError("simulated crash")
    except RuntimeError:
        pass

    # Verify Pending rows exist in SQLite, nothing in ermrest
    # (exact mechanism to retrieve the store after __exit__ depends on execution state design)
    # Resume:
    exe_resumed = test_ml.resume_execution(exe_rid)
    exe_resumed.upload_execution_outputs()

    # Verify ermrest now has exactly one row, no duplicates
    pb = test_ml.pathBuilder()
    rows = [
        r for r in pb.schemas[image_feature.schema].tables[image_feature.feature_table_name].entities().fetch()
        if r.get("Execution") == exe_rid
    ]
    assert len(rows) == 1
```

If `resume_execution` isn't yet wired to flush pending features, this test will guide that implementation. Integration point with §8 execution state is a required prerequisite — if the resume plumbing isn't complete, skip this test with a pointer to the §8 plan and pick it up in a follow-up.

- [ ] **Step 3: Implement ordering + failure isolation + resume**

Open `src/deriva_ml/execution/execution.py`. Verify from Task 7 that `_flush_staged_features` is called after asset flush in `upload_execution_outputs`. If not, move the call.

For **failure isolation**: the `_flush_staged_features` body from Task 7 already continues per-group on exceptions. Verify the test passes.

For **crash-resume**: find the existing `resume_execution` flow (grep `resume_execution`). In the resume path, after the existing staged-asset resume, add a detection step: if `list_pending_feature_records(execution_rid)` returns rows, the execution is in a partially-staged state — the next call to `upload_execution_outputs` will flush them.

- [ ] **Step 4: Run tests**

Run: `DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/test_staged_features.py -q --timeout=600`
Expected: all tests pass (no skips remain).

- [ ] **Step 5: Commit**

```bash
git add src/deriva_ml/execution/execution.py tests/execution/test_staged_features.py
git commit -m "feat(execution): flush ordering, failure isolation, crash-resume for staged features (S2)"
```

---

## Task 12: Full-suite regression + docs refresh

**Files:**
- Grep + fix any remaining callers of retired APIs
- Verify MkDocs build still succeeds
- Run the full live test suite in subsets per CLAUDE.md

- [ ] **Step 1: Full grep for retired-API references**

```bash
grep -rn "fetch_table_features\|list_feature_values\|ml\.add_features\|ml\.select_by_workflow" src/ docs/ tests/ 2>&1 | grep -v "retired\|raise DerivaMLException\|tests/feature/test_retired"
```
Any hit is a call site still using the old name. Fix each — rewrite to `feature_values` / `exe.add_features` / `FeatureRecord.select_by_workflow`.

- [ ] **Step 2: Check MkDocs build**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run mkdocs build 2>&1 | tail -30
```
Expected: clean build, no broken references.

- [ ] **Step 3: Run fast suite**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q
```
Expected: all green.

- [ ] **Step 4: Run feature live suite**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/feature/ -q --timeout=600
```
Expected: all green (includes symmetry, selector factory, retired-API shims, offline-to-online).

- [ ] **Step 5: Run execution live suite**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/execution/ -q --timeout=600
```
Expected: all green (includes staged features, flush ordering, crash-resume).

- [ ] **Step 6: Run dataset live suite**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/dataset/ -q --timeout=600
```
Expected: all green (includes bag feature cache).

- [ ] **Step 7: Coverage check**

```bash
DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST=localhost uv run pytest tests/feature/ tests/execution/test_staged_features.py tests/dataset/test_bag_feature_cache.py --cov=deriva_ml.feature --cov=deriva_ml.core.mixins.feature --cov=deriva_ml.dataset.bag_feature_cache --cov=deriva_ml.execution.execution --cov-report=term-missing --timeout=600
```
Expected: every public method in the new surface has branches for happy path, selector present / absent, and documented error paths covered.

- [ ] **Step 8: Commit any remaining fixes**

```bash
git add -A
git commit -m "chore(feature): fix remaining retired-API callers; docs + coverage green (S2)"
```

- [ ] **Step 9: Use `superpowers:finishing-a-development-branch` for merge/PR**

Per the subagent-driven-development flow, at this point the final code reviewer runs and then `finishing-a-development-branch` presents the merge/PR options.
