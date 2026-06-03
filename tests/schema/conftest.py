"""Shared fixtures for ``tests/schema/``.

The schema tests that assert on the *generated* catalog model (FK
conventions, ACL bindings, the Dataset_Execution version FK) need a live
catalog produced by ``create_ml_catalog`` — the same mechanism the existing
integration tests in this directory already use (see
``test_vocab_fk_convention.py``).

These fixtures create that catalog once per session and hand back the
introspected model + the ML schema name, deleting the catalog on teardown.
They are session-scoped because catalog creation is the slow part (~15s) and
the model is read-only for these assertions.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def _schema_catalog():
    """Create one fresh DerivaML catalog for the schema test session.

    Yields the ``ErmrestCatalog`` and deletes it on teardown. Mirrors the
    create/introspect/delete pattern in ``test_vocab_fk_convention.py``.
    """
    from deriva_ml.schema.create_schema import create_ml_catalog

    hostname = os.environ.get("DERIVA_HOST", "localhost")
    catalog = create_ml_catalog(hostname=hostname, project_name="s1b_schema_test")
    try:
        yield catalog
    finally:
        catalog.delete_ermrest_catalog(really=True)


@pytest.fixture(scope="session")
def demo_model(_schema_catalog):
    """The introspected catalog model for the fresh schema-test catalog."""
    return _schema_catalog.getCatalogModel()


@pytest.fixture(scope="session")
def ml_schema() -> str:
    """The name of the deriva-ml schema in the fresh catalog."""
    return "deriva-ml"
