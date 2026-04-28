"""Integration tests for ``find_workflows(sort=...)``.

Catalog-required. Validates the three sort modes (None / True / callable).
``Workflow`` has no public ``rct`` attribute, so the "newest-first" test
asserts on ``rid`` (descending) as a proxy: RIDs are server-assigned
monotonically on a fresh catalog, so the lexicographic order correlates
with insertion order.
"""

from __future__ import annotations

import pytest

from deriva_ml import MLVocab as vc


@pytest.fixture
def catalog_with_workflows(test_ml):
    """Create at least 3 distinct workflows on a fresh test catalog.

    Workflows are deduplicated by checksum, so each must have a unique
    checksum to avoid being collapsed.

    Args:
        test_ml: Clean DerivaML instance with empty catalog.

    Yields:
        DerivaML instance with 3 workflows created.
    """
    # Add vocabulary term required by create_workflow
    test_ml.add_term(vc.workflow_type, "Sort Test Workflow", description="Workflow type for sort tests")

    for i in range(3):
        workflow = test_ml.create_workflow(
            name=f"sort_test_{i}",
            workflow_type="Sort Test Workflow",
            description=f"Workflow for find_workflows sort tests (variant {i})",
        )
        # Assign unique checksums to ensure workflows don't get deduplicated
        workflow.checksum = f"dummy-sort-test-{i}"
        test_ml._add_workflow(workflow)

    return test_ml


@pytest.mark.integration
def test_find_workflows_sort_none_returns_records(catalog_with_workflows):
    """sort=None preserves backend order; just verify it returns rows."""
    ml = catalog_with_workflows
    workflows = list(ml.find_workflows())
    assert len(workflows) >= 3, "fixture should provide at least 3 workflows"


@pytest.mark.integration
def test_find_workflows_sort_true_returns_newest_first(catalog_with_workflows):
    """sort=True yields records ordered by RCT descending.

    Workflow has no public rct attribute, so we use rid as a proxy:
    server-assigned RIDs are monotonic on a fresh catalog, so RID-desc
    correlates with RCT-desc.
    """
    ml = catalog_with_workflows
    workflows = list(ml.find_workflows(sort=True))
    rids = [w.rid for w in workflows]
    assert rids == sorted(rids, reverse=True), f"workflows should be newest-first (RID-desc proxy); got rids={rids}"


@pytest.mark.integration
def test_find_workflows_sort_callable_applies_user_keys(catalog_with_workflows):
    """User-supplied sort callable applies."""
    ml = catalog_with_workflows

    def by_rid_asc(path):
        return path.RID

    workflows = list(ml.find_workflows(sort=by_rid_asc))
    rids = [w.rid for w in workflows]
    assert rids == sorted(rids), f"workflows should be RID-ascending; got {rids}"


@pytest.mark.integration
def test_find_workflows_sort_invalid_type_raises(catalog_with_workflows):
    """Passing a bare string (not None/True/callable) raises TypeError."""
    ml = catalog_with_workflows
    with pytest.raises(TypeError, match="sort must be None, True, or a callable"):
        list(ml.find_workflows(sort="newest"))  # type: ignore[arg-type]
