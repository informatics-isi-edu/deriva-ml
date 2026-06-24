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
        "policy.json does not declare row_owner_guard — test premise is broken. Investigate the policy file separately."
    )

    catalog = create_ml_catalog(
        hostname="localhost",
        project_name="s1b_acl_test",
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
            tname for tname, table in deriva_ml.tables.items() if "row_owner_guard" not in (table.acl_bindings or {})
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


@pytest.mark.integration
def test_create_ml_catalog_grants_rid_lease_write_policy() -> None:
    """public.ERMrest_RID_Lease gets the SAME write policy as deriva-ml tables.

    Regression: RID_Lease lives in the ``public`` schema, which policy.json sets
    to ``read_only`` (insert/update/delete=empty), and the per-table
    ``row_owner_guard`` rule deliberately excludes ``public``. So a new catalog
    left RID_Lease effectively read-only for regular users — they could not
    create or manage their own RID leases. The fix is an exact-match table_acls
    entry giving RID_Lease the ``self_serve`` ACL + ``row_owner_guard`` binding,
    matching a deriva-ml table (and the ERMrest RID-lease default-policy doc:
    insert allowed, RCB-projection row guard for select/update/delete).
    """
    from deriva_ml.schema.create_schema import create_ml_catalog

    # Premise check: the policy must declare the RID_Lease override.
    policy_file = files("deriva_ml.schema").joinpath("policy.json")
    policy = json.loads(policy_file.read_text())
    rid_lease_specs = [
        e for e in policy["table_acls"] if e.get("schema") == "public" and e.get("table") == "ERMrest_RID_Lease"
    ]
    assert rid_lease_specs, "policy.json has no exact-match table_acls entry for public.ERMrest_RID_Lease."

    catalog = create_ml_catalog(hostname="localhost", project_name="ridlease_acl_test")
    try:
        model = catalog.getCatalogModel()
        rid_lease = model.schemas["public"].tables["ERMrest_RID_Lease"]

        # 1. The row-owner guard is applied (same binding deriva-ml tables get).
        assert "row_owner_guard" in (rid_lease.acl_bindings or {}), (
            "ERMrest_RID_Lease is missing the row_owner_guard binding — a user "
            "cannot update/delete the leases they created."
        )

        # 2. The static ACL grants insert to writers_and_curators (self_serve),
        #    overriding the public schema's read_only. Compare against the
        #    catalog-level self_serve insert group so this stays correct if the
        #    group membership is retuned later.
        insert_acl = (rid_lease.acls or {}).get("insert")
        assert insert_acl, (
            f"ERMrest_RID_Lease has no insert ACL (acls={dict(rid_lease.acls or {})}); it is still inheriting the "
            f"public schema's read_only policy — users cannot create RID leases."
        )
    finally:
        catalog.delete_ermrest_catalog(really=True)
