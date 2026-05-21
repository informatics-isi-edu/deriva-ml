"""Integration test for ACL policy application via ``create_ml_catalog``.

Closes audit Phase 3 schema/ §3.2 — ``create_ml_catalog``
shells out to ``deriva.config.acl_config`` with ``policy.json``
to install per-row update/delete protection. The end-to-end
behavior had no test; a regression that silently broke the
``deriva-acl-config`` invocation (e.g., missing policy file,
swallowed CalledProcessError, or running it before the
deriva-ml schema exists) would leave new catalogs without
``row_owner_guard`` protection.

This test verifies that after ``create_ml_catalog`` returns,
every table in the ``deriva-ml`` schema carries the
``row_owner_guard`` binding in its ``acl_bindings`` field. The
binding lives as a first-class catalog-model field, not buried
in annotations — see deriva-py's ``Table.acl_bindings`` (a JSON
dict managed independently of ``Table.annotations``).

The original (now-fixed) regression: ``create_ml_catalog``
called ``acl_config`` *before* ``create_ml_schema``, so the
policy's per-table binding rule (which targets all non-public
schemas) matched zero tables. The deriva-ml tables were
silently created later without bindings; non-curator users
hit HTTP 403 on every Execution_Metadata PATCH because
``row_owner_guard`` (the binding that would have authorized
"users can update rows they created") was never applied.

Pattern follows ``test_vocab_fk_convention.py``. Requires
DERIVA_HOST.
"""

from __future__ import annotations

import json
from importlib.resources import files

import pytest


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

        # deriva-acl-config writes per-table bindings into
        # ``table.acl_bindings`` (a first-class catalog-model field,
        # not buried in annotations). Walk the deriva-ml schema and
        # require row_owner_guard on every table — the per-table
        # binding rule in policy.json applies to all non-public
        # schemas, so a single missing table flags either a
        # configure ordering bug (acl_config ran before
        # create_ml_schema) or a policy regression.
        deriva_ml = model.schemas["deriva-ml"]
        missing = [
            tname
            for tname, table in deriva_ml.tables.items()
            if "row_owner_guard" not in (table.acl_bindings or {})
        ]
        assert not missing, (
            f"row_owner_guard binding missing on deriva-ml tables: "
            f"{sorted(missing)}. acl_config must run *after* "
            f"create_ml_schema, otherwise the per-table binding rule "
            f"finds no tables to bind to. See create_schema.py "
            f"ordering of create_ml_schema vs the acl_config "
            f"subprocess."
        )
    finally:
        catalog.delete_ermrest_catalog(really=True)
