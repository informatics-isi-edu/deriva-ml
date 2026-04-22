# H3 — `ExecutionRecord` Disambiguation Design

**Status:** Draft · **Date:** 2026-04-22 · **Subsystem:** H3 (post-Phase-2 hygiene)

## 1. Goal

Eliminate the name collision between two unrelated classes that both ship as `ExecutionRecord`:

- **Live catalog-backed record** (`deriva_ml.execution.execution_record.ExecutionRecord`) — Pydantic `BaseModel`; property setters write to the catalog; returned by `lookup_execution`, `find_executions`.
- **Local SQLite-backed snapshot** (`deriva_ml.execution.execution_record_v2.ExecutionRecord`) — frozen `@dataclass`; value object captured at query time; returned by `list_executions`, `find_incomplete_executions`.

These classes were introduced during Phase 2 S1a as the "v2" replacement was built alongside the live record. They represent legitimately different concepts (live mutable vs. frozen snapshot), not a migration pair, so this subsystem **disambiguates rather than merges them**.

## 2. Scope

**In scope:**
- Rename the V2 class `ExecutionRecord` → `ExecutionSnapshot`.
- Rename the V2 module `execution_record_v2.py` → `execution_snapshot.py` (keeps file-name ↔ class-name symmetry).
- Update every internal importer of V2 (currently all aliased as `_ExecutionRecordV2`) to import `ExecutionSnapshot` directly.
- Update return-type annotations on `list_executions` and `find_incomplete_executions` from `list[_ExecutionRecordV2]` to `list[ExecutionSnapshot]`.
- Rename tests file `tests/execution/test_execution_record_v2.py` → `tests/execution/test_execution_snapshot.py` and update its imports.
- Update `tests/execution/test_execution_registry.py` import.
- Rewrite docstrings on the four execution-query methods (`lookup_execution`, `find_executions`, `list_executions`, `find_incomplete_executions`) so each makes the live-vs-snapshot distinction explicit and cross-references the sibling methods.
- Rewrite the class docstrings on both `ExecutionRecord` (live) and `ExecutionSnapshot` (local) to explicitly state which they are and when to use each.
- CHANGELOG entry.

**Explicitly out of scope:**
- No functional/behavior change. Method signatures, return shapes, and semantics are unchanged.
- No method renames. The audit of the codebase's naming convention (mixins/*, Execution class) showed `list_` / `find_` / `lookup_` / `get_` are not cleanly split by data source — `list_vocabulary_terms` hits the catalog, `list_files` reads the filesystem, `find_incomplete_executions` reads SQLite. Changing method names to encode data source would be a bigger change than the real confusion warrants.
- No merger of the two classes. Their concepts differ (live mutable vs. frozen snapshot) in ways that would make a merged class either lose functionality or grow mode branches everywhere.
- No legacy class rename. `ExecutionRecord` (live) keeps its name — it's the primary concept in the catalog-record mental model and the one most user-facing code will interact with.
- No deprecation shim. The V2 class was aliased internally as `_ExecutionRecordV2`; no user code was importing it under the name `ExecutionRecord`. This is a pure internal rename with no external-caller impact.

## 3. Architecture

No architectural change. This is a rename + docstring cleanup that makes existing behavior legible.

**Before (current `main`):**

```
deriva_ml.execution.execution_record.ExecutionRecord       # live, pydantic, mutable
deriva_ml.execution.execution_record_v2.ExecutionRecord    # snapshot, dataclass, frozen
                                       ↑ aliased as _ExecutionRecordV2 where imported
```

**After:**

```
deriva_ml.execution.execution_record.ExecutionRecord       # unchanged
deriva_ml.execution.execution_snapshot.ExecutionSnapshot   # renamed class + renamed module
```

## 4. Class and module renames

| From | To |
|---|---|
| `src/deriva_ml/execution/execution_record_v2.py` | `src/deriva_ml/execution/execution_snapshot.py` |
| class `ExecutionRecord` (in V2 module) | class `ExecutionSnapshot` |
| `tests/execution/test_execution_record_v2.py` | `tests/execution/test_execution_snapshot.py` |
| import `ExecutionRecord as _ExecutionRecordV2` | import `ExecutionSnapshot` (no alias needed) |

**Internal importer update** (`src/deriva_ml/core/mixins/execution.py`):

```python
# Before (lines 19-20)
from deriva_ml.execution.execution_record_v2 import (
    ExecutionRecord as _ExecutionRecordV2,
)

# After
from deriva_ml.execution.execution_snapshot import ExecutionSnapshot
```

All subsequent references to `_ExecutionRecordV2` (return-type annotations at lines 324, 384; constructor call at line 356) are replaced with `ExecutionSnapshot`.

## 5. Docstring rewrites

**Requirement:** every one of the four execution-query methods must have a docstring that makes these three things clear:

1. Whether it reads from the **live catalog** (ERMrest) or the **local SQLite registry**.
2. What the return type is, in one sentence, including whether it's mutable (live) or frozen (snapshot).
3. A "See also:" or inline cross-reference to the sibling method in the other category, so a reader stumbling on one finds the other.

### 5.1 `lookup_execution(rid)` — live, single item

```python
def lookup_execution(self, execution_rid: RID) -> "ExecutionRecord":
    """Look up a single execution by RID in the live catalog.

    Queries the ERMrest catalog for the Execution row with the given
    RID and returns an ``ExecutionRecord`` — a live, catalog-bound
    value whose mutable properties (``status``, ``description``)
    write through to the catalog on assignment. Online mode only.

    For enumerating executions from the local SQLite registry without
    touching the catalog, see ``list_executions()``. For catalog-side
    filter queries returning live records, see ``find_executions()``.

    Args:
        execution_rid: Resource Identifier (RID) of the execution.

    Returns:
        A live ``ExecutionRecord`` bound to the catalog. Property
        setters (``record.status = ...``) write through.

    Raises:
        DerivaMLException: If execution_rid is not valid or doesn't
            refer to an Execution record.

    Example:
        >>> record = ml.lookup_execution("1-abc123")
        >>> record.status = ExecutionStatus.Uploaded   # writes to catalog
    """
```

### 5.2 `find_executions(...)` — live, filtered

```python
def find_executions(
    self,
    workflow: "Workflow | RID | None" = None,
    workflow_type: str | None = None,
    status: ExecutionStatus | None = None,
) -> Iterable["ExecutionRecord"]:
    """Search the live catalog for executions matching the given filters.

    Queries the ERMrest catalog (online only) and yields live,
    catalog-bound ``ExecutionRecord`` objects for each match. Each
    returned record's mutable properties (``status``, ``description``)
    write through to the catalog on assignment.

    For enumerating locally-known executions from the SQLite registry
    without touching the catalog (works in offline mode), see
    ``list_executions()`` and ``find_incomplete_executions()``.

    Args:
        workflow: Optional Workflow or RID to filter by.
        workflow_type: Optional workflow type name (e.g., "python_script").
        status: Optional ExecutionStatus to filter by.

    Returns:
        Iterable of live ``ExecutionRecord`` objects.

    Example:
        >>> for record in ml.find_executions(status=ExecutionStatus.Uploaded):
        ...     print(record.execution_rid, record.status)
    """
```

### 5.3 `list_executions(...)` — local snapshot, filtered

```python
def list_executions(
    self,
    *,
    status: "ExecutionStatus | list[ExecutionStatus] | None" = None,
    workflow_rid: str | None = None,
    mode: "ConnectionMode | None" = None,
    since: datetime | None = None,
) -> list[ExecutionSnapshot]:
    """Enumerate locally-known executions from the SQLite registry.

    Reads from the workspace SQLite registry — **no server contact**.
    Works in both online and offline mode. Each returned
    ``ExecutionSnapshot`` is a frozen value object captured at query
    time; it cannot mutate the catalog. Pending-row counts are
    included in the same pass.

    For live catalog queries that return mutable
    :class:`~deriva_ml.execution.execution_record.ExecutionRecord`
    objects bound to the catalog, see ``find_executions()`` and
    ``lookup_execution()``.

    Args:
        status: Single ExecutionStatus or list to filter; None = all.
        workflow_rid: Match only executions tagged with this Workflow
            RID; None = all.
        mode: ConnectionMode the execution was last active under;
            None = all.
        since: Return only executions with last_activity >= this
            timestamp (timezone-aware). None = no time filter.

    Returns:
        List of ``ExecutionSnapshot`` dataclasses — one per matching
        row in the registry. Empty list if nothing matches.

    Example:
        >>> from deriva_ml.execution.state_store import ExecutionStatus
        >>> failed = ml.list_executions(status=ExecutionStatus.Failed)
        >>> for snap in failed:
        ...     print(snap.rid, snap.error)
    """
```

### 5.4 `find_incomplete_executions()` — local snapshot, pre-filtered shortcut

```python
def find_incomplete_executions(self) -> list[ExecutionSnapshot]:
    """Sugar over :meth:`list_executions` for everything not terminally done.

    Reads from the workspace SQLite registry — no server contact.
    Returns executions in status in (Created, Running, Stopped, Failed,
    Pending_Upload) — the set of things a user would want to either
    resume, retry, or clean up. Excludes Uploaded (terminal success)
    and Aborted (terminal cleanup).

    For live catalog queries returning mutable
    :class:`~deriva_ml.execution.execution_record.ExecutionRecord`
    objects, see ``find_executions(status=...)``.

    Returns:
        List of ``ExecutionSnapshot`` dataclasses for each incomplete
        execution known to the local registry.

    Example:
        >>> for snap in ml.find_incomplete_executions():
        ...     print(snap.rid, snap.status, snap.pending_rows)
    """
```

### 5.5 Class docstrings

Both classes get explicit disambiguation in their top-of-class docstrings:

**`ExecutionRecord`** (live):

Opens with one paragraph: *"A live, catalog-bound record. Property setters write through to the catalog on assignment; requires online mode for mutations. Returned by :meth:`~deriva_ml.DerivaML.lookup_execution` and :meth:`~deriva_ml.DerivaML.find_executions`. For a frozen snapshot value object that reads from the local SQLite registry and works offline, see :class:`~deriva_ml.execution.execution_snapshot.ExecutionSnapshot`."*

**`ExecutionSnapshot`** (local):

Opens with one paragraph: *"A frozen value object capturing an execution's registry-row fields at query time. Backed by the workspace SQLite registry, not the live catalog — reads in this class do not contact the server, and no property is mutable. Returned by :meth:`~deriva_ml.DerivaML.list_executions` and :meth:`~deriva_ml.DerivaML.find_incomplete_executions`. For a live, mutable catalog record that writes through on property assignment, see :class:`~deriva_ml.execution.execution_record.ExecutionRecord`."*

## 6. Testing plan

No new test cases. The existing tests (`test_execution_record_v2.py`, `test_execution_registry.py`) exercise the V2 class's behavior; renaming imports + the test file itself is the full testing change. All existing tests must pass unchanged post-rename.

**Regression gate:**
- `tests/execution/` unit tests + live-catalog integration tests against `DERIVA_HOST=localhost` must pass with the same counts as on `main` pre-rename.
- `tests/test_migration_complete.py` (the S1a grep-gate) continues to pass — the rename doesn't touch `Status` references.
- `ruff check` on touched files does not regress.

## 7. Risks

1. **Missed call sites.** If I miss an importer or type annotation, a test fails on import. Mitigation: explicit grep for `ExecutionRecord` and `execution_record_v2` at plan time; re-grep after renames to confirm zero hits on `_ExecutionRecordV2` / the old module name.
2. **Downstream users importing `ExecutionRecord` from the V2 module path.** The V2 class wasn't exported from `deriva_ml/__init__.py`, and the only internal users aliased it — but external users could have `from deriva_ml.execution.execution_record_v2 import ExecutionRecord` in their own code. That path disappears after the rename. Mitigation: CHANGELOG documents the rename with the exact replacement path.
3. **Docstring drift.** If I rewrite the four method docstrings but the class docstrings don't mirror the live-vs-snapshot framing, readers arriving from either direction (method signature or class source) see mismatched narrative. Mitigation: write the class docstrings and method docstrings in the same session, review together.

## 8. Rollout

Single PR against `main`. No feature flag. No breaking change to the public `deriva_ml` namespace (neither `ExecutionRecord` flavor was ever re-exported). CHANGELOG entry marked "Renamed" (not "Breaking") with the V2 import path change as the only user-visible delta, and only for callers who reached into the internal `execution_record_v2` module name directly.
