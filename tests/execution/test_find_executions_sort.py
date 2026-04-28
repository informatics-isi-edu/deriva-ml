"""Integration tests for ``find_executions(sort=...)``.

Catalog-required. Validates the three sort modes (None / True / callable)
and asserts the row order returned in each mode.
"""

from __future__ import annotations

import pytest

from deriva_ml import MLVocab as vc
from deriva_ml.execution.execution_configuration import ExecutionConfiguration


@pytest.fixture
def catalog_with_executions(test_ml):
    """Create at least 3 executions on a fresh test catalog.

    Each is a no-op execution against a single workflow; the important
    property is that they have distinct RIDs and RCTs spaced far enough
    apart for sort tests to be meaningful (rows are inserted sequentially,
    giving them distinct RCTs).

    Args:
        test_ml: Clean DerivaML instance with empty catalog.

    Yields:
        DerivaML instance with 3 executions created.
    """
    # Add vocabulary term required by create_workflow
    test_ml.add_term(vc.workflow_type, "Sort Test Workflow", description="Workflow type for sort tests")

    workflow = test_ml.create_workflow(
        name="sort_test",
        workflow_type="Sort Test Workflow",
        description="Workflow for find_executions sort tests",
    )

    for i in range(3):
        cfg = ExecutionConfiguration(
            workflow=workflow,
            description=f"sort-test-execution-{i}",
        )
        test_ml.create_execution(cfg, dry_run=False)

    return test_ml


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

    RIDs are server-assigned sequentially, so RID lexicographic order
    correlates with insertion order. Newest-first means descending RID.
    """
    ml = catalog_with_executions
    records = list(ml.find_executions(sort=True))
    rids = [r.execution_rid for r in records]
    assert rids == sorted(rids, reverse=True), f"records should be newest-first (descending RID); got rids={rids}"


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
