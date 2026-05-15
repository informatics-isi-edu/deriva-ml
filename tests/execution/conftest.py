"""Test-scoped fixtures for ``tests/execution/``.

The audit at
``docs/design/deriva-ml-audit-2026-05-phase3-execution.md`` §C.S6
flagged a state-leakage risk specific to the execution test suite:

    ``deriva_ml.execution.runner._multirun_state`` is a module-level
    singleton holding the active parent-execution context. The shared
    ``CatalogManager`` reset between tests clears catalog rows but
    does **not** touch deriva-ml module globals. A test that creates
    a parent execution (e.g. ``test_multirun_parent_lifecycle``)
    leaves ``_multirun_state.parent_execution`` populated; the next
    test that invokes any code path consulting that global will see
    a stale parent and silently link a new child to a deleted
    catalog row.

The autouse fixture below closes that hole: before every test in
this directory, ``reset_multirun_state()`` is called so the global
starts blank. The fixture is cheap (one in-process function call),
runs before ``test_ml`` resets the catalog, and adds no
catalog-side cost.
"""

from __future__ import annotations

import pytest

from deriva_ml.execution.runner import reset_multirun_state


@pytest.fixture(autouse=True)
def _reset_multirun_state_between_tests():
    """Clear ``runner._multirun_state`` before every test in this directory.

    Closes the test-isolation hole identified by audit §C.S6 — the
    module-level multirun singleton is **not** reset by the shared
    ``CatalogManager`` fixture. Without this fixture, a test that
    leaves a parent execution dangling pollutes every subsequent
    test that consults the global.

    Yields control to the test, then resets again on teardown so the
    next test's setup doesn't see leftover state if this test
    crashed mid-flow.
    """
    reset_multirun_state()
    yield
    reset_multirun_state()
