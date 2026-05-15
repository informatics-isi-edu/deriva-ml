"""Integration test for ACL policy application via ``create_ml_catalog``.

Closes audit Phase 3 schema/ §3.2 — ``create_ml_catalog``
shells out to ``deriva.config.acl_config`` with ``policy.json``
to install per-row update/delete protection. The end-to-end
behavior had no test; a regression that silently broke the
``deriva-acl-config`` invocation (e.g., missing policy file,
swallowed CalledProcessError) would leave new catalogs without
``row_owner_guard`` protection.

This test verifies that after ``create_ml_catalog`` returns:

1. The catalog model carries an ``acl_bindings`` annotation
   somewhere in the catalog or schema annotations.
2. The bindings include ``row_owner_guard`` (or an equivalent
   per-row binding).

The exact place ``deriva-acl-config`` writes the bindings can
shift between deriva-py versions; we assert presence at any
ACL-binding-capable level rather than pinning a specific path.

Pattern follows ``test_vocab_fk_convention.py``. Requires
DERIVA_HOST.
"""

from __future__ import annotations

import json
from importlib.resources import files

import pytest


def _find_row_owner_guard(node) -> bool:
    """Recursively search a JSON-ish structure for ``row_owner_guard``."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "row_owner_guard":
                return True
            if _find_row_owner_guard(v):
                return True
    elif isinstance(node, list):
        for item in node:
            if _find_row_owner_guard(item):
                return True
    elif isinstance(node, str):
        if node == "row_owner_guard":
            return True
    return False


@pytest.mark.integration
def test_create_ml_catalog_applies_row_owner_guard() -> None:
    """A freshly-created catalog carries the row_owner_guard ACL binding.

    The binding lives in the policy.json that create_ml_catalog
    passes to deriva-acl-config; verifying it shows up in the
    live catalog model is the end-to-end check that the
    ``subprocess.run(... acl_config ...)`` step actually ran.
    """
    from deriva_ml.schema.create_schema import create_ml_catalog

    # Sanity-check the bundled policy file actually defines
    # row_owner_guard — if it doesn't, the test premise is wrong
    # and we'd assert against thin air.
    policy_file = files("deriva_ml.schema").joinpath("policy.json")
    policy = json.loads(policy_file.read_text())
    assert "row_owner_guard" in policy["acl_bindings"], (
        "policy.json does not declare row_owner_guard — test premise "
        "is broken. Investigate the policy file separately."
    )

    catalog = create_ml_catalog(
        hostname="localhost", project_name="s1b_acl_test",
    )
    try:
        model = catalog.getCatalogModel()

        # deriva-acl-config writes ACL bindings at the catalog or
        # schema level (varies by version). Check both.
        found = _find_row_owner_guard(model.annotations)
        if not found:
            for schema in model.schemas.values():
                if _find_row_owner_guard(schema.annotations):
                    found = True
                    break
                for table in schema.tables.values():
                    if _find_row_owner_guard(table.annotations):
                        found = True
                        break
                if found:
                    break

        assert found, (
            "row_owner_guard not found anywhere in the catalog model "
            "after create_ml_catalog. The deriva-acl-config invocation "
            "in create_ml_catalog may have silently failed."
        )
    finally:
        catalog.delete_ermrest_catalog(really=True)
