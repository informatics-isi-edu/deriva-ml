"""Unit tests for catalog_snapshot() schema reuse (no redundant /schema fetch).

These tests verify the performance fix from
docs/superpowers/specs/2026-06-13-estimate-bag-size-perf-design.md:
a snapshot DerivaML reuses the live instance's already-parsed schema
instead of re-fetching /schema from the server.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def live_ml(catalog_manager, tmp_path):
    """A populated DerivaML instance against the test catalog."""
    catalog_manager.ensure_populated(tmp_path)
    return catalog_manager.get_ml_instance(tmp_path)


def test_live_instance_retains_schema_json(live_ml):
    """_init_online stores the parsed schema dict on the instance for reuse."""
    assert hasattr(live_ml, "_schema_json")
    assert isinstance(live_ml._schema_json, dict)
    # ermrest /schema payloads have a top-level "schemas" key.
    assert "schemas" in live_ml._schema_json
