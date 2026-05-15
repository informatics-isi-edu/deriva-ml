"""Tests for the ACL policy bundled with deriva-ml.

Closes audit Phase 3 schema/ §3.6 — ``policy.json`` had no test
asserting its content. These tests load the JSON resource via
``importlib.resources`` and verify the structural invariants
that ``deriva-acl-config`` will later consume.

They run without a live catalog — pure JSON loading.
"""

from __future__ import annotations

import json
from importlib.resources import files

import pytest

# ---------------------------------------------------------------------------
# Fixture — load policy.json once per test module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def policy() -> dict:
    """Load the bundled ``deriva_ml/schema/policy.json``."""
    policy_file = files("deriva_ml.schema").joinpath("policy.json")
    return json.loads(policy_file.read_text())


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


def test_policy_has_required_top_level_keys(policy: dict) -> None:
    """The policy file has the four sections ``deriva-acl-config`` consumes."""
    required = {"groups", "acl_definitions", "acl_bindings", "catalog_acl"}
    missing = required - set(policy)
    assert not missing, f"policy.json missing required top-level keys: {missing}"


def test_policy_groups_resolve_to_lists(policy: dict) -> None:
    """Every entry in ``groups`` maps to a list (possibly of group names)."""
    for name, members in policy["groups"].items():
        assert isinstance(members, list), (
            f"group '{name}' must map to a list, got {type(members).__name__}"
        )


def test_policy_row_owner_guard_binding_present(policy: dict) -> None:
    """The ``row_owner_guard`` ACL binding controls per-row update/delete.

    Regression guard for audit §3.2: every non-public table is
    expected to carry this binding, and the binding's projection
    must resolve to ``RCB`` (Row Creator's identity) so a row's
    creator owns it.
    """
    bindings = policy["acl_bindings"]
    assert "row_owner_guard" in bindings, (
        "row_owner_guard binding missing — non-public tables would have "
        "no per-row update/delete protection."
    )
    guard = bindings["row_owner_guard"]
    assert guard["types"] == ["update", "delete"]
    assert guard["projection"] == ["RCB"]
    assert guard["projection_type"] == "acl"


def test_policy_table_acls_apply_row_owner_guard_to_non_public(policy: dict) -> None:
    """The ``table_acls`` block applies ``row_owner_guard`` everywhere except ``public``.

    The schema_pattern is a negative-lookahead regex meaning "any
    schema name that is not literally 'public'." A change to that
    pattern that accidentally narrowed it (e.g., to a single
    schema) would leave most catalogs without per-row protection.
    """
    table_acls = policy["table_acls"]
    assert isinstance(table_acls, list) and table_acls, "table_acls must be non-empty list"

    # Find the catch-all entry binding row_owner_guard.
    matching = [
        entry
        for entry in table_acls
        if "row_owner_guard" in entry.get("acl_bindings", [])
    ]
    assert matching, "row_owner_guard not wired through any table_acls entry"

    # The catch-all entry must exclude 'public' (non-public tables
    # get the per-row protection; public.* is read-only via
    # schema_acls).
    entry = matching[0]
    assert "public" in entry["schema_pattern"], (
        "row_owner_guard's schema_pattern must explicitly handle the "
        "public schema (typically via negative lookahead)."
    )
    assert entry["table_pattern"] == ".*", (
        "row_owner_guard should bind every table in matching schemas, "
        "not a subset."
    )


def test_policy_public_schema_is_read_only(policy: dict) -> None:
    """The ``public`` schema must be marked read-only.

    ``public`` carries the catalog-wide reference tables (RID
    leases, group / client registry). Public-schema writes from
    untrusted users would break catalog-wide invariants.
    """
    schema_acls = policy.get("schema_acls", [])
    public_entries = [e for e in schema_acls if e.get("schema") == "public"]
    assert public_entries, "policy.json must declare a schema_acls entry for 'public'"

    entry = public_entries[0]
    assert entry["acl"] == "read_only", (
        f"public schema must be read_only, got '{entry['acl']}'"
    )


def test_policy_acl_definitions_have_required_modes(policy: dict) -> None:
    """Each named ACL definition declares all five CRUD modes.

    ``deriva-acl-config`` expects ``select``, ``enumerate``,
    ``insert``, ``update``, ``delete`` to be present on every
    named ACL. Missing modes silently default to "deny," which
    can cause surprising lockouts.
    """
    required_modes = {"select", "enumerate", "insert", "update", "delete"}
    for name, definition in policy["acl_definitions"].items():
        missing = required_modes - set(definition)
        assert not missing, (
            f"acl_definition '{name}' missing CRUD modes: {missing}"
        )
