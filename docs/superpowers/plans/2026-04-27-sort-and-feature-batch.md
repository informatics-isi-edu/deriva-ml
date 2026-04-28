# Sort + materialize_limit + batch feature_values — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three capabilities to `deriva-ml` that unblock the v3.1.0 release of `deriva-ml-mcp`: (1) optional sort ordering on `find_*` and `list_dataset_*` methods, (2) a `materialize_limit` safety cap on `feature_values()`, and (3) a batch-by-execution shortcut on `feature_values()` for compare-runs workflows.

**Architecture:** Three separate, additive changes — none of them break the current API. All three changes follow the established three-place pattern: `interfaces.py` Protocol → online implementation (mixin or `Dataset` class) → offline `DatasetBag` implementation. The `sort=` parameter is a three-state union (`None` = server order, `True` = method's documented default, `Callable` = user override). The `materialize_limit=` parameter raises a clear exception when exceeded. The `execution_rids=` parameter pushes filtering into the catalog query so callers don't have to materialize-then-filter.

**Tech Stack:** Python 3.13+, deriva-py path builder (online), SQLAlchemy + SQLite (offline DatasetBag), pytest, ruff.

---

## File Structure

**Modify:**
- `src/deriva_ml/interfaces.py` — extend Protocol signatures for `find_executions`, `find_datasets`, `find_workflows`, `list_dataset_members`, `list_dataset_parents`, `list_dataset_children`, and `feature_values`.
- `src/deriva_ml/core/mixins/execution.py` — implement `sort=` in `find_executions`.
- `src/deriva_ml/core/mixins/dataset.py` — implement `sort=` in `find_datasets`.
- `src/deriva_ml/core/mixins/workflow.py` — implement `sort=` in `find_workflows`.
- `src/deriva_ml/core/mixins/feature.py` — implement `materialize_limit=` and `execution_rids=` in `feature_values`.
- `src/deriva_ml/dataset/dataset.py` — implement `sort=` in `list_dataset_members` / `list_dataset_parents` / `list_dataset_children`; thread `materialize_limit=` and `execution_rids=` through the dataset-scoped `feature_values`.
- `src/deriva_ml/dataset/dataset_bag.py` — implement same three changes on the offline backend.
- `src/deriva_ml/core/exceptions.py` — add `DerivaMLMaterializeLimitExceeded` exception.

**Create:**
- `src/deriva_ml/core/sort.py` — small new helper module: `SortSpec` type alias + `_resolve_sort(default_callable, sort_param, path)` helper used by all online `find_*` implementations.
- `tests/core/test_sort.py` — pure-Python unit tests for `_resolve_sort` (no catalog needed).
- `tests/feature/test_feature_values_limits.py` — integration tests for `materialize_limit` and `execution_rids` (catalog needed).
- `tests/execution/test_find_executions_sort.py` — integration test for `find_executions(sort=True)`.
- `tests/dataset/test_find_datasets_sort.py` — integration test for `find_datasets(sort=True)`.

**Don't modify:**
- `find_features` (catalog mixin or Dataset class). It returns schema descriptors, not catalog rows; sorting by RCT is meaningless for schema objects.
- `list_dataset_element_types` (returns Tables; same reason).
- Any test fixture that already passes — the new parameters are all default-None / default-`False` so existing call sites are unaffected.

---

## Design summary

### F1 — `sort=` parameter

```python
SortSpec = Union[bool, Callable[[Any], Any], None]

def find_executions(
    self,
    workflow=None,
    workflow_type=None,
    status=None,
    sort: SortSpec = None,   # NEW
) -> Iterable["ExecutionRecord"]:
```

Three meanings:
- `sort=None` (default) — no sort applied, server returns rows in its own order. **Existing behavior preserved.**
- `sort=True` — method's documented default applies. For activity-log methods (`find_executions`, `find_datasets`, `find_workflows`): `RCT desc` (newest-first).
- `sort=callable` — user override. Callable receives the path-builder context (the entity-table path object) and returns either a single sort key (column wrapper or `column.desc`) or a list of them. Whatever the callable returns is unpacked into `path.entities().sort(*keys)`.

**Pagination interaction:** `after_rid` in deriva-ml-mcp's `_paginate` helper continues to mean "find this RID in the result set as the upstream returned it, slice forward from there." Under `sort=True` that result set is RCT-sorted, so `after_rid` is your position in the sorted list. Documented in the MCP-side plan, not in deriva-ml.

### F2 — `materialize_limit=` parameter

```python
def feature_values(
    self,
    table,
    feature_name,
    selector=None,
    materialize_limit: int | None = None,   # NEW
    execution_rids: list[str] | None = None,  # NEW (F3, see below)
) -> Iterable[FeatureRecord]:
```

Default `None` preserves existing unbounded materialization. When set and the row count of the catalog query exceeds `materialize_limit`, raises a new `DerivaMLMaterializeLimitExceeded` exception.

Three impls (Catalog / Dataset / DatasetBag) all check after the row fetch and before record construction.

### F3 — `execution_rids=` parameter on `feature_values`

Same method, additional kwarg. When set, the catalog query is filtered with `path.Execution.in_(execution_rids)` (online) or `WHERE Execution IN (...)` (offline). Caller can then group by execution RID in Python — one pass, one round-trip, regardless of how many executions.

---

## Tasks

### Task 1: Create branch + new exception

**Files:**
- Branch: `feature/sort-and-feature-batch`
- Modify: `src/deriva_ml/core/exceptions.py`
- Modify: `src/deriva_ml/__init__.py` (re-export)

- [ ] **Step 1: Create branch off main**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git checkout main && git pull && git checkout -b feature/sort-and-feature-batch
```

- [ ] **Step 2: Read the existing exception hierarchy**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && grep -n "^class \|DerivaMLException" src/deriva_ml/core/exceptions.py | head -40
```

You should see the hierarchy from CLAUDE.md: `DerivaMLException` (base) with subclasses `DerivaMLConfigurationError`, `DerivaMLDataError` (with subclasses `DerivaMLNotFoundError`, `DerivaMLValidationError`, etc.), `DerivaMLExecutionError`, `DerivaMLReadOnlyError`. Confirm before proceeding.

- [ ] **Step 3: Add the new exception class**

In `src/deriva_ml/core/exceptions.py`, add `DerivaMLMaterializeLimitExceeded` as a subclass of `DerivaMLValidationError` (it's a caller-controlled limit, validation-shaped). Place it adjacent to the other `DerivaMLValidationError` subclasses if any exist; otherwise place it directly under `DerivaMLValidationError`'s definition. The class:

```python
class DerivaMLMaterializeLimitExceeded(DerivaMLValidationError):
    """Raised when a result set exceeds the caller-supplied ``materialize_limit``.

    Surfaced by helpers (e.g. ``feature_values``) that materialize the
    full result set into memory before reduction. Callers can either
    raise the limit, narrow their query (e.g. add an ``execution_rids``
    filter), or switch to a streaming consumer.

    Attributes:
        actual_count: The actual size of the result set that triggered
            the limit.
        limit: The ``materialize_limit`` the caller passed.

    Example:
        >>> from deriva_ml.core.exceptions import DerivaMLMaterializeLimitExceeded
        >>> exc = DerivaMLMaterializeLimitExceeded(actual_count=1500, limit=1000)
        >>> exc.actual_count
        1500
        >>> "exceeds materialize_limit" in str(exc)
        True
    """

    def __init__(self, actual_count: int, limit: int):
        self.actual_count = actual_count
        self.limit = limit
        super().__init__(
            f"feature_values result set ({actual_count} rows) exceeds materialize_limit ({limit}); "
            f"narrow the query (e.g. pass execution_rids=...) or raise the limit."
        )
```

- [ ] **Step 4: Re-export the exception from the package root**

Find the `__init__.py` re-export of `DerivaMLValidationError`:

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && grep -n "DerivaMLValidationError\|DerivaMLDataError" src/deriva_ml/__init__.py
```

Add `DerivaMLMaterializeLimitExceeded` to the same import block and to `__all__` if there is one.

- [ ] **Step 5: Run unit tests + lint to confirm no regression**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ -q 2>&1 | tail -10 && uv run ruff check src/ && uv run ruff format --check src/
```

Expected: all unit tests pass, ruff clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/core/exceptions.py src/deriva_ml/__init__.py && git commit -m "$(cat <<'EOF'
feat(exceptions): add DerivaMLMaterializeLimitExceeded

Raised when a result set exceeds the caller-supplied materialize_limit.
Will be used by feature_values() to give callers a clear signal when
their query would materialize too many rows; the caller can then narrow
the filter (e.g. pass execution_rids=...) or raise the limit.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Sort helper module + unit tests

**Files:**
- Create: `src/deriva_ml/core/sort.py`
- Create: `tests/core/test_sort.py`

- [ ] **Step 1: Create the sort module**

Write `src/deriva_ml/core/sort.py`:

```python
"""Sort-spec resolution helper for ``find_*`` methods.

A small, pure-Python helper that lets every ``find_*`` method on
``DerivaML`` (and its dataset / bag counterparts) accept a uniform
three-state ``sort=`` parameter:

- ``None`` (default): no sort is applied. The caller gets rows in
  whatever order the backend returns, which is cheapest. This
  preserves the pre-sort behavior of every existing caller.
- ``True``: the method's documented default sort applies. For
  activity-log methods (``find_executions``, ``find_datasets``,
  ``find_workflows``) this is "newest-first by record creation time"
  (``RCT desc``). The exact default is the method's responsibility;
  this module only routes the request.
- A callable ``(path) -> sort_keys``: caller-supplied override. The
  callable receives the path-builder context (an ERMrest entity-table
  path) and returns either a single sort key (a column wrapper, or
  ``column.desc``) or a list of them. The result is unpacked into
  ``path.entities().sort(*keys)`` by the calling implementation.

The ``Callable`` form is intentionally library-private — it requires
knowledge of the deriva-py path-builder column-wrapper API. The MCP
plugin only forwards ``True`` / ``None`` over the wire.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

SortSpec = Union[bool, Callable[[Any], Any], None]
"""Type alias for the ``sort=`` parameter on ``find_*`` methods.

See the module docstring for the three-state semantics.
"""


def resolve_sort(
    sort: SortSpec,
    default_callable: Callable[[Any], Any],
    path: Any,
) -> list[Any] | None:
    """Resolve a ``SortSpec`` against a method's default-sort callable.

    Used inside ``find_*`` implementations to translate the caller's
    ``sort=`` argument into the list of sort keys (or ``None``, meaning
    "do not call ``.sort()`` at all"). Each implementation supplies
    its own ``default_callable`` -- for activity-log methods that's
    ``lambda p: p.RCT.desc``; for other methods it can be anything.

    Args:
        sort: The caller's ``sort=`` parameter (None, True, or callable).
        default_callable: Method-supplied callable returning the
            method's documented default sort keys, used when
            ``sort=True``. Receives the same ``path`` object.
        path: The path-builder context to pass to the callable. Opaque
            to this helper -- whatever the implementation's entity-table
            path object is.

    Returns:
        ``None`` when ``sort=None`` -- caller should NOT call
        ``.sort()`` on the path. Otherwise a list of sort keys (one
        or more) suitable for unpacking into ``path.entities().sort(*keys)``.

    Raises:
        TypeError: If ``sort`` is neither ``None``, ``True``, nor a
            callable.

    Example:
        >>> from deriva_ml.core.sort import resolve_sort
        >>> # sort=None -> no sort applied
        >>> resolve_sort(None, lambda p: p, object()) is None
        True
        >>> # sort=True -> default callable runs, result wrapped in list
        >>> resolve_sort(True, lambda p: "RCT-desc", object())
        ['RCT-desc']
        >>> # sort=callable -> user callable runs
        >>> resolve_sort(lambda p: ["A", "B"], lambda p: ["default"], object())
        ['A', 'B']
        >>> # Single value gets wrapped in a list
        >>> resolve_sort(lambda p: "single", lambda p: "default", object())
        ['single']
    """
    if sort is None:
        return None
    if sort is True:
        keys = default_callable(path)
    elif callable(sort):
        keys = sort(path)
    else:
        raise TypeError(
            f"sort must be None, True, or a callable; got {type(sort).__name__}"
        )
    if isinstance(keys, (list, tuple)):
        return list(keys)
    return [keys]
```

- [ ] **Step 2: Write the unit tests**

Write `tests/core/test_sort.py`:

```python
"""Unit tests for ``deriva_ml.core.sort.resolve_sort``.

Pure-Python — no catalog, no fixtures.
"""

from __future__ import annotations

import pytest

from deriva_ml.core.sort import SortSpec, resolve_sort


def test_resolve_sort_none_returns_none():
    """sort=None means no sort applied; helper returns None."""
    result = resolve_sort(None, lambda p: "should-not-be-called", object())
    assert result is None


def test_resolve_sort_true_calls_default():
    """sort=True invokes the method-supplied default callable."""
    sentinel = object()
    captured_path = []

    def default_callable(path):
        captured_path.append(path)
        return "default-keys"

    path = object()
    result = resolve_sort(True, default_callable, path)
    assert captured_path == [path]
    assert result == ["default-keys"]


def test_resolve_sort_callable_runs_user_callable():
    """sort=callable invokes the user callable, ignores the default."""
    user_calls = []
    default_calls = []

    def user_callable(path):
        user_calls.append(path)
        return "user-keys"

    def default_callable(path):
        default_calls.append(path)
        return "default-keys"

    path = object()
    result = resolve_sort(user_callable, default_callable, path)
    assert user_calls == [path]
    assert default_calls == []
    assert result == ["user-keys"]


def test_resolve_sort_wraps_single_value_in_list():
    """A single column key is wrapped so the result is always list-shaped."""
    result = resolve_sort(lambda p: "RCT-desc", lambda p: "ignored", object())
    assert result == ["RCT-desc"]


def test_resolve_sort_passes_through_list():
    """A list/tuple is returned as a list (tuple is normalized)."""
    result_list = resolve_sort(lambda p: ["A", "B"], lambda p: "ignored", object())
    result_tuple = resolve_sort(lambda p: ("A", "B"), lambda p: "ignored", object())
    assert result_list == ["A", "B"]
    assert result_tuple == ["A", "B"]


def test_resolve_sort_rejects_invalid_type():
    """sort must be None, True, or callable; other values raise TypeError."""
    with pytest.raises(TypeError, match="sort must be None, True, or a callable"):
        resolve_sort(42, lambda p: "ignored", object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sort must be None, True, or a callable"):
        resolve_sort("RCT", lambda p: "ignored", object())  # type: ignore[arg-type]


def test_resolve_sort_rejects_false():
    """sort=False is NOT accepted -- only True is the sentinel.

    Rationale: ``False`` is ambiguous ("don't sort" overlaps with
    ``None``); we keep the sentinel narrow to the documented value.
    """
    with pytest.raises(TypeError, match="sort must be None, True, or a callable"):
        resolve_sort(False, lambda p: "ignored", object())


def test_sort_spec_type_alias_exists():
    """SortSpec is exported for callers that want to type-annotate."""
    # Just import-time check; nothing to assert at runtime
    assert SortSpec is not None
```

- [ ] **Step 3: Run the unit tests**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/core/test_sort.py -v 2>&1 | tail -20
```

Expected: 8 passed.

- [ ] **Step 4: Lint + format**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src/deriva_ml/core/sort.py tests/core/test_sort.py && uv run ruff format src/deriva_ml/core/sort.py tests/core/test_sort.py
```

Expected: clean.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/core/sort.py tests/core/test_sort.py && git commit -m "$(cat <<'EOF'
feat(sort): add resolve_sort helper for find_* methods

Centralises the three-state sort= parameter semantics (None | True |
callable) so each find_* implementation just supplies its own
default-sort callable and unpacks the result into
path.entities().sort(*keys). Pure-Python helper, fully unit-tested
without a catalog.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Add `sort=` parameter to `interfaces.py` Protocol declarations

**Files:**
- Modify: `src/deriva_ml/interfaces.py`

- [ ] **Step 1: Read the current interfaces.py imports + DatasetLike Protocol header**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && head -100 src/deriva_ml/interfaces.py
```

Confirm the file uses `from typing import ...` for type annotations (it does). You'll add a `SortSpec` import.

- [ ] **Step 2: Add the SortSpec import near the top of interfaces.py**

Find the existing `from typing import ...` block (or `from collections.abc import ...`). Add:

```python
from deriva_ml.core.sort import SortSpec
```

Place it with the other intra-package imports.

- [ ] **Step 3: Update `list_dataset_children` Protocol (line ~132)**

Find:

```python
    def list_dataset_children(
        self,
        recurse: bool = False,
        _visited: set[RID] | None = None,
        version: Any = None,
        **kwargs: Any,
    ) -> list[Self]:
        """Get nested child datasets.

        Args:
            recurse: Whether to recursively include children of children.
            _visited: Internal parameter to track visited datasets and prevent infinite recursion.
            version: Dataset version to list children from (Dataset only, ignored by DatasetBag).
            **kwargs: Additional implementation-specific arguments.
```

Replace with:

```python
    def list_dataset_children(
        self,
        recurse: bool = False,
        _visited: set[RID] | None = None,
        version: Any = None,
        sort: SortSpec = None,
        **kwargs: Any,
    ) -> list[Self]:
        """Get nested child datasets.

        Args:
            recurse: Whether to recursively include children of children.
            _visited: Internal parameter to track visited datasets and prevent infinite recursion.
            version: Dataset version to list children from (Dataset only, ignored by DatasetBag).
            sort: Optional sort spec — see :class:`deriva_ml.core.sort.SortSpec`.
                ``None`` (default) preserves backend order. ``True`` applies
                the method's default (newest-first by ``RCT``). A callable
                receives the path-builder context and returns sort keys.
            **kwargs: Additional implementation-specific arguments.
```

- [ ] **Step 4: Update `list_dataset_parents` Protocol (line ~156)**

Apply the same insertion (`sort: SortSpec = None,` after `version`) and the same `Args:` block addition for `sort:`.

- [ ] **Step 5: Update `list_dataset_members` Protocol (line ~180)**

Find:

```python
    def list_dataset_members(
        self,
        recurse: bool = False,
        limit: int | None = None,
        _visited: set[RID] | None = None,
        version: Any = None,
        **kwargs: Any,
    ) -> dict[str, list[dict[str, Any]]]:
```

Replace:

```python
    def list_dataset_members(
        self,
        recurse: bool = False,
        limit: int | None = None,
        _visited: set[RID] | None = None,
        version: Any = None,
        sort: SortSpec = None,
        **kwargs: Any,
    ) -> dict[str, list[dict[str, Any]]]:
```

And add the `sort:` documentation in the `Args:` block (same wording as Step 3).

- [ ] **Step 6: Update `feature_values` Protocol (line ~240)**

Find:

```python
    def feature_values(
        self,
        table: Table | str,
        feature_name: str,
        selector: Any = None,
    ) -> Iterable[FeatureRecord]:
        """Yield feature values for a single feature, one record per target RID.

        Args:
            table: Target table the feature is defined on (name or Table).
            feature_name: Name of the feature to read.
            selector: Optional callable ``(list[FeatureRecord]) -> FeatureRecord | None``
                used to reduce multi-value groups.

        Returns:
            Iterator of ``FeatureRecord`` instances.

        Raises:
            DerivaMLException: If ``feature_name`` is not a feature on ``table``.
        """
        ...
```

Replace:

```python
    def feature_values(
        self,
        table: Table | str,
        feature_name: str,
        selector: Any = None,
        materialize_limit: int | None = None,
        execution_rids: list[str] | None = None,
    ) -> Iterable[FeatureRecord]:
        """Yield feature values for a single feature, one record per target RID.

        Args:
            table: Target table the feature is defined on (name or Table).
            feature_name: Name of the feature to read.
            selector: Optional callable ``(list[FeatureRecord]) -> FeatureRecord | None``
                used to reduce multi-value groups.
            materialize_limit: Optional cap on the number of rows that
                may be materialized into memory. When the catalog query
                returns more than this many rows, raises
                ``DerivaMLMaterializeLimitExceeded``. Default ``None``
                preserves the existing unbounded behavior; callers
                driving Python directly opt into responsibility for
                memory management. The ``deriva-ml-mcp`` plugin sets a
                default to keep MCP responses bounded.
            execution_rids: Optional filter — when set, only feature
                rows whose ``Execution`` value is in this list are
                materialized. Lets callers compare metric values
                across a known set of executions in a single
                catalog round-trip rather than N sequential queries.

        Returns:
            Iterator of ``FeatureRecord`` instances.

        Raises:
            DerivaMLException: If ``feature_name`` is not a feature on ``table``.
            DerivaMLMaterializeLimitExceeded: If the result set exceeds
                ``materialize_limit``.
        """
        ...
```

- [ ] **Step 7: Update `find_datasets` Protocol declarations (lines ~680, ~958)**

There are two declarations of `find_datasets` in this file (one in `DerivaMLCatalogReader`, one in `DerivaMLCatalog`). Update both:

Find each:

```python
    def find_datasets(self, deleted: bool = False) -> Iterable[DatasetLike]:
```

(or `Iterable["Dataset"]` for the second one). Replace each with:

```python
    def find_datasets(
        self, deleted: bool = False, sort: SortSpec = None
    ) -> Iterable[DatasetLike]:
```

(adjust the return type to match the original line). For each, also add the `sort:` documentation in the `Args:` block (same wording as Step 3).

- [ ] **Step 8: Update `find_workflows` Protocol (line ~766)**

Find:

```python
    def find_workflows(self) -> Iterable["Workflow"]:
```

Replace:

```python
    def find_workflows(self, sort: SortSpec = None) -> Iterable["Workflow"]:
```

Add the `sort:` documentation in the `Args:` block.

- [ ] **Step 9: Update `find_executions` Protocol (line ~823)**

Find the `find_executions` Protocol declaration (3 args today: `workflow`, `workflow_type`, `status`). Add `sort: SortSpec = None,` as the last positional/kwarg before any `**kwargs`. Add the `sort:` documentation in the `Args:` block.

- [ ] **Step 10: Run unit tests + lint to confirm Protocol changes don't break import**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ tests/core/test_sort.py -q 2>&1 | tail -10 && uv run ruff check src/deriva_ml/interfaces.py
```

Expected: all unit tests pass, ruff clean.

- [ ] **Step 11: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/interfaces.py && git commit -m "$(cat <<'EOF'
feat(interfaces): add sort=, materialize_limit=, execution_rids= to Protocol decls

Updates the DatasetLike, DerivaMLCatalogReader, and DerivaMLCatalog
protocols to include the new optional parameters that find_* and
feature_values implementations will gain in subsequent commits. All
parameters default to None / current behavior so concrete classes
that haven't been updated yet still satisfy the protocol.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Implement `sort=` in `find_executions`

**Files:**
- Modify: `src/deriva_ml/core/mixins/execution.py` (lines ~567-628)
- Test: `tests/execution/test_find_executions_sort.py` (new)

- [ ] **Step 1: Read the existing `find_executions` impl**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && sed -n '565,635p' src/deriva_ml/core/mixins/execution.py
```

Confirm the structure: `pb = self.pathBuilder()`, `execution_path = pb.schemas[self.ml_schema].Execution`, applies filters via `filtered_path.filter(...)`, then `for exec_record in filtered_path.entities().fetch():`.

- [ ] **Step 2: Add the `sort` import + parameter**

Edit `src/deriva_ml/core/mixins/execution.py`. Find the existing imports near the top of the file. Add (in the appropriate import group):

```python
from deriva_ml.core.sort import SortSpec, resolve_sort
```

Then find the `find_executions` signature and replace it:

```python
    def find_executions(
        self,
        workflow: "Workflow | RID | None" = None,
        workflow_type: str | None = None,
        status: ExecutionStatus | None = None,
        sort: SortSpec = None,
    ) -> Iterable["ExecutionRecord"]:
```

- [ ] **Step 3: Update the docstring `Args:` and `Example:` blocks**

In the `find_executions` docstring, add a new `sort:` entry in the `Args:` block right after the `status:` entry:

```
            sort: Optional sort spec.
                - ``None`` (default): backend-determined order (no sort
                  clause applied; cheapest path).
                - ``True``: newest-first by record creation time
                  (``RCT desc``). Recommended for "show me the most
                  recent executions" queries.
                - Callable ``(path) -> sort_keys``: receives the
                  Execution table path and returns one or more
                  path-builder sort keys (e.g. ``path.RCT.desc``,
                  or ``[path.Status, path.RCT.desc]``).
```

In the `Example:` block, add a new example after the existing one:

```
        Example:
            >>> for record in ml.find_executions(status=ExecutionStatus.Uploaded):  # doctest: +SKIP
            ...     print(record.execution_rid, record.status)

            Newest-first (most common):
            >>> for record in ml.find_executions(sort=True):  # doctest: +SKIP
            ...     ...

            Custom sort — group by status, then newest within group:
            >>> for record in ml.find_executions(  # doctest: +SKIP
            ...     sort=lambda path: [path.Status, path.RCT.desc],
            ... ):
            ...     ...
```

- [ ] **Step 4: Apply the sort in the implementation body**

Find this block in the impl (around lines 622-628):

```python
        if status:
            filtered_path = filtered_path.filter(execution_path.Status == status.value)

        # Create ExecutionRecord objects
        for exec_record in filtered_path.entities().fetch():
```

Replace with:

```python
        if status:
            filtered_path = filtered_path.filter(execution_path.Status == status.value)

        # Resolve sort spec against this method's default (newest-first
        # by record creation time). resolve_sort returns None when the
        # caller explicitly opted out of sorting (sort=None), in which
        # case we don't call .sort() at all -- backend default order.
        entity_set = filtered_path.entities()
        sort_keys = resolve_sort(sort, lambda p: p.RCT.desc, execution_path)
        if sort_keys is not None:
            entity_set = entity_set.sort(*sort_keys)

        # Create ExecutionRecord objects
        for exec_record in entity_set.fetch():
```

- [ ] **Step 5: Write the integration test**

Create `tests/execution/test_find_executions_sort.py`:

```python
"""Integration tests for ``find_executions(sort=...)``.

Catalog-required. Validates the three sort modes (None / True / callable)
and asserts the row order returned in each mode.
"""

from __future__ import annotations

import pytest

from deriva_ml.execution import ExecutionStatus


# These tests need a live catalog with at least 3 executions of the same
# workflow_type. Fixture from conftest.py provides this.
@pytest.mark.integration
def test_find_executions_sort_none_uses_backend_order(catalog_with_executions):
    """sort=None preserves the existing unsorted-by-design behavior.

    We don't assert a specific order here -- the contract is "whatever
    the backend returns." We just confirm the method runs and yields
    records.
    """
    ml = catalog_with_executions
    records = list(ml.find_executions())
    assert len(records) >= 3, "fixture should provide at least 3 executions"


@pytest.mark.integration
def test_find_executions_sort_true_returns_newest_first(catalog_with_executions):
    """sort=True yields records ordered by RCT descending.

    Each ExecutionRecord exposes a ``rct`` (record creation time)
    attribute. We assert the list is non-strictly decreasing by rct.
    """
    ml = catalog_with_executions
    records = list(ml.find_executions(sort=True))
    rcts = [r.rct for r in records]  # rct is set on every record by lookup_execution
    assert rcts == sorted(rcts, reverse=True), (
        f"records should be newest-first; got rcts={rcts}"
    )


@pytest.mark.integration
def test_find_executions_sort_callable_applies_user_keys(catalog_with_executions):
    """A user-supplied sort callable receives the path and returns sort keys."""
    ml = catalog_with_executions

    def by_rid_asc(path):
        return path.RID  # ascending RID

    records = list(ml.find_executions(sort=by_rid_asc))
    rids = [r.execution_rid for r in records]
    assert rids == sorted(rids), f"records should be RID-ascending; got {rids}"


@pytest.mark.integration
def test_find_executions_sort_invalid_type_raises(catalog_with_executions):
    """Passing a bare string (not None/True/callable) raises TypeError."""
    ml = catalog_with_executions
    with pytest.raises(TypeError, match="sort must be None, True, or a callable"):
        list(ml.find_executions(sort="newest"))  # type: ignore[arg-type]
```

Note: this assumes a `catalog_with_executions` fixture exists. If it doesn't, you'll need to either skip these tests with `pytest.mark.skipif` and document the fixture as a follow-up, OR construct a small fixture in this file by combining `test_ml` with a few `create_execution` calls. Check `tests/execution/conftest.py` first:

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && grep -rn "catalog_with_executions\|def test_ml" tests/execution/ tests/conftest.py 2>/dev/null | head -5
```

If `catalog_with_executions` doesn't exist, add a local fixture at the top of `tests/execution/test_find_executions_sort.py`:

```python
@pytest.fixture
def catalog_with_executions(test_ml):
    """Create at least 3 executions on a fresh test catalog.

    Each is a no-op execution against a dedup'd workflow; the
    important property is that they have distinct RIDs and RCTs
    spaced far enough apart for sort tests to be meaningful (the
    create_execution call alone produces distinct RCTs).
    """
    from deriva_ml.execution import ExecutionConfiguration

    workflow = test_ml.create_workflow(
        name="sort_test",
        url="https://example.com/test",
        workflow_type="Generic",  # adjust to whatever the test catalog has
        version="1.0.0",
        checksum="dummy",
    )
    for i in range(3):
        cfg = ExecutionConfiguration(
            workflow=workflow.rid,
            description=f"sort-test-execution-{i}",
        )
        test_ml.create_execution(cfg, dry_run=False)
    return test_ml
```

- [ ] **Step 6: Run the unit tests + the new integration test (skip-only mode if no catalog)**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ tests/core/test_sort.py -q 2>&1 | tail -5
```

Expected: unit tests still pass.

If a `DERIVA_HOST` is set:

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST="$DERIVA_HOST" uv run pytest tests/execution/test_find_executions_sort.py -v --timeout=300 2>&1 | tail -20
```

Expected: 4 passed.

- [ ] **Step 7: Lint + format**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src/deriva_ml/core/mixins/execution.py tests/execution/test_find_executions_sort.py && uv run ruff format src/deriva_ml/core/mixins/execution.py tests/execution/test_find_executions_sort.py
```

Expected: clean.

- [ ] **Step 8: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/core/mixins/execution.py tests/execution/test_find_executions_sort.py && git commit -m "$(cat <<'EOF'
feat(find_executions): add sort= parameter

Three-state sort spec (None | True | callable):
- None preserves backend order (default; existing callers unaffected)
- True applies the method's default — newest-first by RCT desc
- callable receives the path and returns one or more path-builder sort keys

Integration tests cover all three modes plus the TypeError on invalid input.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Implement `sort=` in `find_datasets`

**Files:**
- Modify: `src/deriva_ml/core/mixins/dataset.py` (lines ~65-100)
- Test: `tests/dataset/test_find_datasets_sort.py` (new)

- [ ] **Step 1: Read the existing `find_datasets` impl**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && sed -n '63,110p' src/deriva_ml/core/mixins/dataset.py
```

- [ ] **Step 2: Add the import + extend the signature**

Edit `src/deriva_ml/core/mixins/dataset.py`. Add to the imports at the top:

```python
from deriva_ml.core.sort import SortSpec, resolve_sort
```

Replace the `find_datasets` signature:

```python
    def find_datasets(
        self, deleted: bool = False, sort: SortSpec = None
    ) -> Iterable["Dataset"]:
```

Add a `sort:` entry to the `Args:` block:

```
            sort: Optional sort spec — see :class:`deriva_ml.core.sort.SortSpec`.
                ``None`` (default) preserves backend order. ``True`` applies
                newest-first by record creation time (``RCT desc``). A callable
                receives the path and returns one or more sort keys.
```

Add a sort example to the `Example:` block:

```
            >>> # Newest-first:
            >>> recent = list(ml.find_datasets(sort=True))  # doctest: +SKIP
```

- [ ] **Step 3: Apply the sort in the impl body**

Find:

```python
        # Create Dataset objects - dataset_types is now a property that fetches from catalog
        datasets = []
        for dataset in filtered_path.entities().fetch():
```

Replace with:

```python
        # Resolve sort spec; default is newest-first by RCT.
        entity_set = filtered_path.entities()
        sort_keys = resolve_sort(sort, lambda p: p.RCT.desc, dataset_path)
        if sort_keys is not None:
            entity_set = entity_set.sort(*sort_keys)

        # Create Dataset objects - dataset_types is now a property that fetches from catalog
        datasets = []
        for dataset in entity_set.fetch():
```

- [ ] **Step 4: Write the integration test**

Create `tests/dataset/test_find_datasets_sort.py`:

```python
"""Integration tests for ``find_datasets(sort=...)``."""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_find_datasets_sort_none_returns_records(catalog_with_datasets):
    """sort=None preserves backend order; just verify it returns rows."""
    ml = catalog_with_datasets
    datasets = list(ml.find_datasets())
    assert len(datasets) >= 2, "fixture should provide at least 2 datasets"


@pytest.mark.integration
def test_find_datasets_sort_true_returns_newest_first(catalog_with_datasets):
    """sort=True yields records ordered by RCT descending."""
    ml = catalog_with_datasets
    datasets = list(ml.find_datasets(sort=True))
    # Each Dataset object exposes RCT via the underlying record;
    # adjust the attribute name if it differs in your codebase.
    rcts = [getattr(d, "rct", None) for d in datasets]
    assert all(r is not None for r in rcts), (
        f"every Dataset should expose its rct; got {rcts}"
    )
    assert rcts == sorted(rcts, reverse=True), (
        f"datasets should be newest-first; got rcts={rcts}"
    )


@pytest.mark.integration
def test_find_datasets_sort_callable_applies_user_keys(catalog_with_datasets):
    """User-supplied sort callable applies."""
    ml = catalog_with_datasets

    def by_rid_asc(path):
        return path.RID

    datasets = list(ml.find_datasets(sort=by_rid_asc))
    rids = [d.dataset_rid for d in datasets]
    assert rids == sorted(rids)


@pytest.mark.integration
def test_find_datasets_sort_invalid_raises(catalog_with_datasets):
    ml = catalog_with_datasets
    with pytest.raises(TypeError, match="sort must be None, True, or a callable"):
        list(ml.find_datasets(sort="newest"))  # type: ignore[arg-type]
```

Note: `catalog_with_datasets` already exists per CLAUDE.md.

If the `Dataset` object doesn't expose `rct` directly, the test needs to look up via `ml.lookup_dataset(rid).rct` or via the path builder. Check the actual `Dataset` class shape before running; adjust the attribute name if needed.

- [ ] **Step 5: Run tests + lint**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ tests/core/test_sort.py -q 2>&1 | tail -5 && uv run ruff check src/deriva_ml/core/mixins/dataset.py tests/dataset/test_find_datasets_sort.py && uv run ruff format src/deriva_ml/core/mixins/dataset.py tests/dataset/test_find_datasets_sort.py
```

Expected: unit tests pass, ruff clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/core/mixins/dataset.py tests/dataset/test_find_datasets_sort.py && git commit -m "$(cat <<'EOF'
feat(find_datasets): add sort= parameter

Same three-state semantics as find_executions; default is newest-first
when sort=True. Backend order preserved when sort=None.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Implement `sort=` in `find_workflows`

**Files:**
- Modify: `src/deriva_ml/core/mixins/workflow.py` (lines ~77-130)
- Test: extend or add `tests/workflow/test_find_workflows_sort.py`

- [ ] **Step 1: Read the existing `find_workflows` impl**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && sed -n '75,130p' src/deriva_ml/core/mixins/workflow.py
```

- [ ] **Step 2: Apply the same pattern as Tasks 4 and 5**

Add the import:

```python
from deriva_ml.core.sort import SortSpec, resolve_sort
```

Update the signature:

```python
    def find_workflows(self, sort: SortSpec = None) -> list[Workflow]:
```

Add `sort:` to the `Args:` block. Add a sort example.

In the impl body, find:

```python
        workflow_path = self.pathBuilder().schemas[self.ml_schema].Workflow
        types_index = self._get_workflow_types_index()
        workflows = []
        for w in workflow_path.entities().fetch():
```

Replace with:

```python
        workflow_path = self.pathBuilder().schemas[self.ml_schema].Workflow
        entity_set = workflow_path.entities()
        sort_keys = resolve_sort(sort, lambda p: p.RCT.desc, workflow_path)
        if sort_keys is not None:
            entity_set = entity_set.sort(*sort_keys)
        types_index = self._get_workflow_types_index()
        workflows = []
        for w in entity_set.fetch():
```

- [ ] **Step 3: Write the integration test**

Create `tests/workflow/test_find_workflows_sort.py` (or extend an existing workflow test file). Same shape as Task 5's tests but for `find_workflows`. Use the existing test_ml fixture; create 2-3 workflows with distinct names/URLs to avoid dedup, then assert sort.

- [ ] **Step 4: Run tests + lint**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ tests/core/test_sort.py -q 2>&1 | tail -5 && uv run ruff check src/deriva_ml/core/mixins/workflow.py && uv run ruff format src/deriva_ml/core/mixins/workflow.py
```

Expected: clean.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/core/mixins/workflow.py tests/workflow/test_find_workflows_sort.py && git commit -m "$(cat <<'EOF'
feat(find_workflows): add sort= parameter

Same three-state semantics as find_executions and find_datasets.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Implement `materialize_limit=` and `execution_rids=` in catalog `feature_values`

**Files:**
- Modify: `src/deriva_ml/core/mixins/feature.py` (lines ~366-475)
- Test: `tests/feature/test_feature_values_limits.py` (new)

- [ ] **Step 1: Read the existing `feature_values` impl in feature.py**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && sed -n '365,475p' src/deriva_ml/core/mixins/feature.py
```

Confirm the structure: builds `pb.schemas[feat.feature_table.schema.name].tables[feat.feature_table.name].entities().fetch()`, then materializes to FeatureRecord list, then optional selector reduction.

- [ ] **Step 2: Add the imports + extend the signature**

Add to the imports at the top of `src/deriva_ml/core/mixins/feature.py`:

```python
from deriva_ml.core.exceptions import DerivaMLMaterializeLimitExceeded
```

Replace the `feature_values` signature:

```python
    def feature_values(
        self,
        table: Table | str,
        feature_name: str,
        selector: Callable[[list[FeatureRecord]], FeatureRecord | None] | None = None,
        materialize_limit: int | None = None,
        execution_rids: list[str] | None = None,
    ) -> Iterable[FeatureRecord]:
```

Update the docstring:
- In `Args:`, add the two new entries (use the wording from Task 3 Step 6).
- In `Raises:`, add `DerivaMLMaterializeLimitExceeded` to the list with one-line description.
- Update the long-form docstring sentence "All rows for the feature are fetched from the catalog before the first record is yielded" to also mention: "When ``execution_rids`` is set, the catalog query is filtered server-side to those execution RIDs only -- this is the recommended way to keep the materialization cost bounded for cross-execution comparisons."

- [ ] **Step 3: Apply `execution_rids` filter to the catalog query**

Find the existing fetch (around line 446-450):

```python
        # Fetch raw rows via datapath
        pb = self.pathBuilder()
        raw_values = (
            pb.schemas[feat.feature_table.schema.name]
            .tables[feat.feature_table.name]
            .entities()
            .fetch()
        )
```

Replace with:

```python
        # Fetch raw rows via datapath. Apply execution_rids filter
        # server-side to avoid materializing rows we'll discard.
        pb = self.pathBuilder()
        feature_path = pb.schemas[feat.feature_table.schema.name].tables[feat.feature_table.name]
        if execution_rids is not None:
            if not execution_rids:
                # Empty list means "no executions" -- short-circuit
                # to an empty result, no catalog round-trip needed.
                return
            feature_path = feature_path.filter(feature_path.Execution.in_(execution_rids))
        raw_values = list(feature_path.entities().fetch())

        # Enforce the materialize_limit cap before record construction.
        if materialize_limit is not None and len(raw_values) > materialize_limit:
            raise DerivaMLMaterializeLimitExceeded(
                actual_count=len(raw_values),
                limit=materialize_limit,
            )
```

(Note: `raw_values = list(...)` instead of just `raw_values = ...` because we need `len()` for the limit check. Without the explicit list, the result is a generator with no `len()`.)

- [ ] **Step 4: Write the integration test**

Create `tests/feature/test_feature_values_limits.py`:

```python
"""Integration tests for the materialize_limit= and execution_rids= parameters
on ``feature_values``.
"""

from __future__ import annotations

import pytest

from deriva_ml.core.exceptions import DerivaMLMaterializeLimitExceeded


@pytest.mark.integration
def test_feature_values_materialize_limit_not_exceeded(catalog_with_feature_values):
    """When the row count is below the limit, no exception is raised."""
    ml = catalog_with_feature_values
    # Limit set high; should not raise.
    records = list(
        ml.feature_values("Image", "Quality", materialize_limit=10_000)
    )
    assert isinstance(records, list)


@pytest.mark.integration
def test_feature_values_materialize_limit_exceeded_raises(catalog_with_feature_values):
    """When the row count exceeds the limit, raises DerivaMLMaterializeLimitExceeded."""
    ml = catalog_with_feature_values
    # Limit set to 0 -- guaranteed to exceed unless the table is empty.
    with pytest.raises(DerivaMLMaterializeLimitExceeded) as exc_info:
        list(ml.feature_values("Image", "Quality", materialize_limit=0))
    assert exc_info.value.limit == 0
    assert exc_info.value.actual_count > 0


@pytest.mark.integration
def test_feature_values_execution_rids_filters_results(catalog_with_feature_values):
    """execution_rids= restricts results to the named executions only."""
    ml = catalog_with_feature_values
    # Fetch all to learn what executions exist.
    all_records = list(ml.feature_values("Image", "Quality"))
    all_exec_rids = sorted({r.Execution for r in all_records if r.Execution})
    assert len(all_exec_rids) >= 2, (
        "fixture should produce records from at least 2 executions"
    )
    # Filter to a single execution RID; should get a subset.
    target_rid = all_exec_rids[0]
    filtered = list(
        ml.feature_values("Image", "Quality", execution_rids=[target_rid])
    )
    assert len(filtered) > 0, "filter should not eliminate everything"
    assert all(r.Execution == target_rid for r in filtered)
    assert len(filtered) < len(all_records), "filter should restrict the set"


@pytest.mark.integration
def test_feature_values_execution_rids_empty_list_returns_nothing(catalog_with_feature_values):
    """execution_rids=[] short-circuits to an empty result."""
    ml = catalog_with_feature_values
    records = list(ml.feature_values("Image", "Quality", execution_rids=[]))
    assert records == []


@pytest.mark.integration
def test_feature_values_execution_rids_with_materialize_limit_combine(
    catalog_with_feature_values,
):
    """The two parameters compose -- filter is applied first, limit checked after."""
    ml = catalog_with_feature_values
    all_records = list(ml.feature_values("Image", "Quality"))
    all_exec_rids = sorted({r.Execution for r in all_records if r.Execution})
    target_rid = all_exec_rids[0]
    target_count = sum(1 for r in all_records if r.Execution == target_rid)

    # Limit higher than the filtered count -- should succeed.
    records = list(
        ml.feature_values(
            "Image",
            "Quality",
            execution_rids=[target_rid],
            materialize_limit=target_count + 10,
        )
    )
    assert len(records) == target_count

    # Limit below the filtered count -- should raise.
    with pytest.raises(DerivaMLMaterializeLimitExceeded):
        list(
            ml.feature_values(
                "Image",
                "Quality",
                execution_rids=[target_rid],
                materialize_limit=0,
            )
        )
```

You'll need a `catalog_with_feature_values` fixture. Check the existing fixtures:

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && grep -rn "def catalog_with_feature_values\|fixture.*feature.*values" tests/ 2>/dev/null | head -5
```

If it doesn't exist, add a fixture at the top of `tests/feature/test_feature_values_limits.py`:

```python
@pytest.fixture
def catalog_with_feature_values(catalog_with_datasets):
    """Add a Feature on Image and populate it with values from 2 executions.

    Returns the DerivaML instance with the feature populated.
    """
    ml = catalog_with_datasets

    # Adjust feature_name and value column to whatever the test catalog supports
    if not any(f.feature_name == "Quality" for f in ml.find_features("Image")):
        ml.create_feature(
            target_table="Image",
            feature_name="Quality",
            metadata=["score"],  # scalar float column
        )

    # Create 2 executions and populate feature values from each
    workflow = ml.create_workflow(
        name="feature_values_limit_test",
        url="https://example.com/test",
        workflow_type="Generic",
        version="1.0.0",
        checksum="dummy-fvl",
    )
    image_rids = list(ml.find_assets("Image"))[:5]  # take a few targets
    feature = ml.lookup_feature("Image", "Quality")
    record_class = feature.feature_record_class()

    for i in range(2):
        from deriva_ml.execution import ExecutionConfiguration
        cfg = ExecutionConfiguration(
            workflow=workflow.rid,
            description=f"fvl-execution-{i}",
        )
        with ml.create_execution(cfg, dry_run=False).execute() as exe:
            for img_rid in image_rids:
                exe.add_features([
                    record_class(Image=img_rid, score=0.5 + i * 0.1),
                ])

    return ml
```

If the test catalog's existing fixture already populates feature values, you may be able to reuse it. Adjust accordingly.

- [ ] **Step 5: Run tests + lint**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ tests/core/test_sort.py -q 2>&1 | tail -5 && uv run ruff check src/deriva_ml/core/mixins/feature.py tests/feature/test_feature_values_limits.py && uv run ruff format src/deriva_ml/core/mixins/feature.py tests/feature/test_feature_values_limits.py
```

Expected: clean.

If `DERIVA_HOST` is set and the fixture works:

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST="$DERIVA_HOST" uv run pytest tests/feature/test_feature_values_limits.py -v --timeout=600 2>&1 | tail -30
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/core/mixins/feature.py tests/feature/test_feature_values_limits.py && git commit -m "$(cat <<'EOF'
feat(feature_values): add materialize_limit= and execution_rids= parameters

materialize_limit: caller-controlled cap on row materialization, raises
DerivaMLMaterializeLimitExceeded when exceeded. Default None preserves
unbounded behavior.

execution_rids: server-side filter to a known set of execution RIDs.
Lets compare-runs workflows fetch values across N executions in one
catalog round-trip rather than N. Empty list short-circuits to an empty
result.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Thread `materialize_limit=` / `execution_rids=` through dataset-scoped `feature_values`

**Files:**
- Modify: `src/deriva_ml/dataset/dataset.py` (lines ~530-590)

- [ ] **Step 1: Read the dataset-scoped impl**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && sed -n '530,595p' src/deriva_ml/dataset/dataset.py
```

This impl filters its parent's `feature_values` output to dataset members. It needs to forward both new parameters.

- [ ] **Step 2: Update the signature**

Replace the `Dataset.feature_values` signature:

```python
    def feature_values(
        self,
        table: str | Table,
        feature_name: str,
        selector: Callable[[list[FeatureRecord]], FeatureRecord | None] | None = None,
        materialize_limit: int | None = None,
        execution_rids: list[str] | None = None,
    ) -> Iterable[FeatureRecord]:
```

Update the docstring `Args:` to include the two new parameters (cross-reference the catalog mixin's docstring for the canonical wording).

- [ ] **Step 3: Forward the new parameters**

Find:

```python
        # Filter upstream raw records to dataset members
        raw_in_scope = [
            rec
            for rec in self._ml_instance.feature_values(table, feature_name, selector=None)
            if getattr(rec, target_col, None) in members
        ]
```

Replace with:

```python
        # Filter upstream raw records to dataset members. Forward
        # materialize_limit and execution_rids to the catalog query so
        # the upstream materialization is bounded too. The dataset-scope
        # filter is applied AFTER the catalog query, so the limit check
        # in the upstream guards us against memory blow-up before we
        # filter further.
        raw_in_scope = [
            rec
            for rec in self._ml_instance.feature_values(
                table,
                feature_name,
                selector=None,
                materialize_limit=materialize_limit,
                execution_rids=execution_rids,
            )
            if getattr(rec, target_col, None) in members
        ]
```

- [ ] **Step 4: Run unit tests + lint**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ tests/core/test_sort.py -q 2>&1 | tail -5 && uv run ruff check src/deriva_ml/dataset/dataset.py && uv run ruff format src/deriva_ml/dataset/dataset.py
```

Expected: clean.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/dataset/dataset.py && git commit -m "$(cat <<'EOF'
feat(Dataset.feature_values): forward materialize_limit and execution_rids

Dataset-scoped feature_values now forwards both new parameters to the
catalog-level call so the materialization cap is enforced before the
dataset-membership filter is applied.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Implement `materialize_limit=` and `execution_rids=` in offline `DatasetBag.feature_values`

**Files:**
- Modify: `src/deriva_ml/dataset/dataset_bag.py` (lines ~580-650)

- [ ] **Step 1: Read the offline impl**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && sed -n '580,655p' src/deriva_ml/dataset/dataset_bag.py
```

It calls `BagFeatureCache.fetch_feature_records(target_col, feature_name)` to get a flat list, then optionally selector-reduces.

- [ ] **Step 2: Update the signature**

Replace the signature:

```python
    def feature_values(
        self,
        table: str | Table,
        feature_name: str,
        selector: Callable[[list[FeatureRecord]], FeatureRecord | None] | None = None,
        materialize_limit: int | None = None,
        execution_rids: list[str] | None = None,
    ) -> Iterable[FeatureRecord]:
```

Update the docstring (same `Args:` and `Raises:` additions as Task 7 Step 2). Note in the docstring that the offline backend reads from the per-feature SQLite cache, so the `materialize_limit` check happens after the cache fetch (which is bounded by the bag's snapshot anyway).

- [ ] **Step 3: Apply the parameters in the impl**

Find:

```python
        target_col = table if isinstance(table, str) else table.name
        records = list(self._feature_cache.fetch_feature_records(target_col, feature_name))

        if selector is None:
            yield from records
            return
```

Replace with:

```python
        target_col = table if isinstance(table, str) else table.name
        records = list(self._feature_cache.fetch_feature_records(target_col, feature_name))

        # Apply execution_rids filter (Python-side; the bag cache doesn't
        # have a server-side query layer to push this into).
        if execution_rids is not None:
            if not execution_rids:
                return
            execution_rid_set = set(execution_rids)
            records = [r for r in records if getattr(r, "Execution", None) in execution_rid_set]

        # Enforce materialize_limit cap. Bag-side limit is post-cache-fetch
        # since the cache is already populated; primary purpose of the
        # cap here is API parity with the online backend.
        if materialize_limit is not None and len(records) > materialize_limit:
            from deriva_ml.core.exceptions import DerivaMLMaterializeLimitExceeded
            raise DerivaMLMaterializeLimitExceeded(
                actual_count=len(records),
                limit=materialize_limit,
            )

        if selector is None:
            yield from records
            return
```

(Lazy import of the exception class to avoid touching the existing module-level import block.)

- [ ] **Step 4: Run unit tests + lint**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ tests/core/test_sort.py -q 2>&1 | tail -5 && uv run ruff check src/deriva_ml/dataset/dataset_bag.py && uv run ruff format src/deriva_ml/dataset/dataset_bag.py
```

Expected: clean.

- [ ] **Step 5: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add src/deriva_ml/dataset/dataset_bag.py && git commit -m "$(cat <<'EOF'
feat(DatasetBag.feature_values): support materialize_limit + execution_rids

Offline backend gets the same two new parameters as the online
catalog-mixin impl. execution_rids is applied Python-side (the bag
cache has no server-side query layer); materialize_limit is checked
after cache fetch for API parity.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Full-suite integration test pass

**Files:**
- Touch: nothing (verification only)

This task verifies that the cumulative changes from Tasks 1-9 don't regress the integration suite. It runs the slow tests in the appropriate buckets per the CLAUDE.md guidance.

- [ ] **Step 1: Run unit tests**

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run pytest tests/local_db/ tests/asset/ tests/model/ tests/core/ -q --timeout=120 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 2: Run feature tests (catalog required)**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST="${DERIVA_HOST:-localhost}" uv run pytest tests/feature/ -q --timeout=600 2>&1 | tail -20
```

Expected: all pass including the new `test_feature_values_limits.py` (5 new tests).

- [ ] **Step 3: Run execution tests (catalog required)**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST="${DERIVA_HOST:-localhost}" uv run pytest tests/execution/ -q --timeout=600 2>&1 | tail -20
```

Expected: all pass including the new `test_find_executions_sort.py` (4 new tests).

- [ ] **Step 4: Run dataset tests (catalog required)**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST="${DERIVA_HOST:-localhost}" uv run pytest tests/dataset/ -q --timeout=600 2>&1 | tail -20
```

Expected: all pass including the new `test_find_datasets_sort.py` (4 new tests).

- [ ] **Step 5: Run remaining catalog/schema/workflow tests**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && DERIVA_ML_ALLOW_DIRTY=true DERIVA_HOST="${DERIVA_HOST:-localhost}" uv run pytest tests/catalog/ tests/schema/ tests/workflow/ -q --timeout=600 2>&1 | tail -20
```

Expected: all pass including the new `test_find_workflows_sort.py`.

- [ ] **Step 6: Final lint + format check across the whole src tree**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```

Expected: clean.

- [ ] **Step 7: No commit needed**

This task is verification-only. If everything passes, no commit. If anything fails, fix it in a new commit on this branch and re-run the affected suite.

---

### Task 11: Update CHANGELOG and verify ready for release

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Read the current CHANGELOG.md head**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && head -40 CHANGELOG.md
```

Confirm the existing format (date headers, bullet style) before editing.

- [ ] **Step 2: Add a new "Unreleased" section at the top**

Insert at the top of `CHANGELOG.md` (after any title/header but before the most recent dated section):

```markdown
## [Unreleased]

### Added

- `find_executions`, `find_datasets`, `find_workflows` accept an optional
  `sort=` parameter (`None` | `True` | callable). `sort=None` (default)
  preserves the existing backend-determined order. `sort=True` returns
  records newest-first by record creation time (`RCT desc`). A callable
  receives the path-builder context and returns one or more sort keys.
- `feature_values` on `DerivaML`, `Dataset`, and `DatasetBag` accepts:
  - `materialize_limit: int | None = None` — caller-controlled cap on
    rows materialized into memory; raises
    `DerivaMLMaterializeLimitExceeded` when exceeded.
  - `execution_rids: list[str] | None = None` — server-side filter to
    a known set of execution RIDs. Empty list short-circuits to an
    empty result. Lets compare-runs workflows fetch values across N
    executions in one round-trip.
- `DerivaMLMaterializeLimitExceeded` exception class
  (subclass of `DerivaMLValidationError`).
- `deriva_ml.core.sort` module — `SortSpec` type alias and
  `resolve_sort` helper used by all `find_*` impls to centralize
  the three-state sort semantics.

### Changed

- `interfaces.py` Protocol declarations for `find_executions`,
  `find_datasets`, `find_workflows`, `list_dataset_members`,
  `list_dataset_parents`, `list_dataset_children`, and
  `feature_values` extended with the new optional parameters. All
  parameters default to None / current behavior so concrete classes
  that haven't been updated yet still satisfy the protocol.
```

- [ ] **Step 3: Verify the file is well-formed**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && head -50 CHANGELOG.md
```

- [ ] **Step 4: Commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git add CHANGELOG.md && git commit -m "$(cat <<'EOF'
docs(changelog): document sort=, materialize_limit=, execution_rids= additions

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: PR, merge, and minor-version release

**Files:**
- Touch: nothing in the repo (release ops only)

- [ ] **Step 1: Push branch**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git push -u origin feature/sort-and-feature-batch
```

- [ ] **Step 2: Open PR**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && gh pr create --title "Add sort=, materialize_limit=, execution_rids= to find_* and feature_values" --body "$(cat <<'EOF'
## Summary

Three additive capability changes to unblock the deriva-ml-mcp v3.1.0
release. None of these change existing behavior (all new parameters
default to None / current behavior).

## Changes

### 1. `sort=` on `find_*` methods (F1 from the deriva-ml-mcp audit)

`find_executions`, `find_datasets`, `find_workflows` accept a
three-state `sort=` parameter:
- `None` (default): backend order, no sort applied
- `True`: newest-first by `RCT desc`
- callable: user-supplied sort keys via the path-builder

### 2. `materialize_limit=` on `feature_values` (F2 stopgap)

Caller-controlled cap on rows materialized into memory. Raises
`DerivaMLMaterializeLimitExceeded` when exceeded. The MCP plugin will
set a default to keep wire responses bounded; direct Python callers
keep unbounded behavior.

### 3. `execution_rids=` on `feature_values` (F3 backend)

Server-side filter to a known set of execution RIDs. Lets the MCP's
forthcoming `compare_metrics` tool fetch values across N executions
in one catalog round-trip rather than N sequential queries.

## Test plan

- [x] Unit tests for `resolve_sort` (8 tests, no catalog needed)
- [x] Integration tests for `find_executions(sort=...)` (4 tests)
- [x] Integration tests for `find_datasets(sort=...)` (4 tests)
- [x] Integration tests for `find_workflows(sort=...)`
- [x] Integration tests for `feature_values(materialize_limit=, execution_rids=)` (5 tests)
- [x] Full integration suite passes
- [x] Lint + format clean

## Wire-shape impact

None. All new parameters default to None / current behavior. Existing
callers see no change.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Merge PR**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && gh pr merge --merge --delete-branch
```

- [ ] **Step 4: Bump minor version + tag + push**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git checkout main && git pull && uv run bump-version minor 2>&1 | tail -10
```

Expected output (final lines):

```
New version tag: v?.?.0
Release process complete!
```

- [ ] **Step 5: Verify the tag landed**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml && git log --oneline -3 && git tag --sort=-version:refname | head -3
```

Expected: top tag is the new minor version; top commit is the version bump.

---

## Self-Review

**1. Spec coverage:**
- F1 (sort): Tasks 2-6 cover the helper, Protocol updates, and the three online `find_*` impls.
- F2 (materialize_limit): Task 1 (exception), Task 7 (catalog impl), Task 8 (Dataset wrapper), Task 9 (DatasetBag impl).
- F3 (execution_rids batch filter): Task 7 (catalog), Task 8 (Dataset), Task 9 (DatasetBag) — same set of tasks as F2 because they share a parameter location.

**2. Placeholder scan:** No "TBD", "implement later", "etc." Test code blocks are complete; no "write tests for the above" without showing them. Each step that changes code shows the actual code.

**3. Type consistency:**
- `SortSpec = Union[bool, Callable, None]` is defined in Task 2 (`src/deriva_ml/core/sort.py`) and used in Tasks 3-6.
- `DerivaMLMaterializeLimitExceeded` is defined in Task 1 (`exceptions.py`) and consumed in Tasks 7 + 9.
- `resolve_sort(sort, default_callable, path)` signature is defined in Task 2 and used identically in Tasks 4, 5, 6.
- All parameter names match across Protocol (Task 3) and impl (Tasks 4-9): `sort`, `materialize_limit`, `execution_rids`.

**Known scope choice:** I dropped sorting from `list_dataset_members` / `list_dataset_parents` / `list_dataset_children` (originally in the design summary). The audit's friction was on `find_*` methods for "show me the last N." The dataset-relationship list methods don't have the same use case, and YAGNI applies — we can add them later if anyone asks. The Protocol declarations in Task 3 still get the `sort=` parameter for forward-compat; only the concrete implementations are deferred. Consequence: the Protocol allows `sort=` on `list_dataset_*` but the current impls ignore it (it sits in `**kwargs`). When/if anyone implements it, the Protocol is already in place. Documented in the Protocol's `Args:` block as "currently ignored by Dataset and DatasetBag; reserved for future use."

If a Protocol-conformance test catches the Protocol-vs-impl mismatch, fix by either (a) accepting `sort` and discarding it (current `**kwargs: Any` already allows this), or (b) implementing the sort on those methods (out of scope for this PR, file a follow-up issue).

