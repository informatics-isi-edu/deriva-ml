"""Unit tests for catalog_snapshot() schema reuse (no redundant /schema fetch).

These tests verify the performance fix from
docs/superpowers/specs/2026-06-13-estimate-bag-size-perf-design.md:
a snapshot DerivaML reuses the live instance's already-parsed schema
instead of re-fetching /schema from the server.
"""

from __future__ import annotations

import pytest

from deriva_ml import DerivaML


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


def test_init_online_skips_fetch_when_schema_supplied(live_ml, monkeypatch):
    """When reuse_schema_json is supplied, _init_online does not call getCatalogSchema()."""
    # Build a second instance against the same catalog, supplying the
    # already-parsed schema; assert getCatalogSchema is never called.
    from deriva.core.ermrest_catalog import ErmrestCatalog

    calls = {"n": 0}
    real = ErmrestCatalog.getCatalogSchema

    def counting(self, *a, **k):
        calls["n"] += 1
        return real(self, *a, **k)

    monkeypatch.setattr(ErmrestCatalog, "getCatalogSchema", counting)

    DerivaML(
        live_ml.host_name,
        live_ml.catalog_id,
        working_dir=live_ml.working_dir,
        ml_schema=live_ml.ml_schema,
        credential=live_ml.credential,
        reuse_schema_json=live_ml._schema_json,
    )
    assert calls["n"] == 0, "getCatalogSchema must not be called when schema is reused"
