"""Test-scoped fixtures for ``tests/core/``.

Closes the state-leakage hole flagged in
``docs/design/deriva-ml-audit-2026-05-phase3-core.md`` §C.S2:

    ``test_vocabulary.py`` creates ~7 vocabularies per test
    (``CV1`` through ``CV_Handle``). The shared ``test_ml``
    fixture's ``catalog_manager.reset()`` clears the ML schema's
    tables, but the existence-guard system in
    ``CatalogManager.ensure_*`` checks only the ML schema's well-
    known tables (Dataset, Workflow, Execution, etc.). If
    ``reset()`` ever fails to clear domain-schema vocabularies,
    the existence guards won't catch it, and the next vocabulary
    test sees leftover terms from the previous run.

The autouse fixture below forces a catalog reset before every
test in ``test_vocabulary.py`` specifically. Other tests in
``tests/core/`` are unaffected — they don't create vocabularies.

For the other state-leakage risks documented in audit §C
(annotation-mutating tests, File-cleanup-on-failure), no
fixture is needed: those are documented as low-risk and tractable
via comments at the call sites.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _vocabulary_test_isolation(request, catalog_manager):
    """Reset the catalog before every test in ``test_vocabulary.py``.

    Per audit §C.S2(a): vocabulary tests create N domain-schema
    vocabularies per test. The shared ``test_ml`` fixture's
    existence-guard pattern checks ML-schema tables; it doesn't
    catch leftover domain-schema vocabularies. Force a real reset
    here to make the isolation guarantee explicit.

    No-op for every other test file in ``tests/core/``.
    """
    if request.node.fspath.basename == "test_vocabulary.py":
        catalog_manager.reset()
    yield
