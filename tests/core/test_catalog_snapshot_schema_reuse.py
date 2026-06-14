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


def _a_snapshot_id(live_ml):
    """Resolve a real snapshot id from the live catalog's current snaptime."""
    return live_ml.catalog.get("/").json()["snaptime"]


def test_catalog_snapshot_does_no_schema_fetch(live_ml, monkeypatch):
    """catalog_snapshot() builds the snapshot instance with zero getCatalogSchema calls."""
    from deriva.core.ermrest_catalog import ErmrestCatalog

    calls = {"n": 0}
    real = ErmrestCatalog.getCatalogSchema

    def counting(self, *a, **k):
        calls["n"] += 1
        return real(self, *a, **k)

    monkeypatch.setattr(ErmrestCatalog, "getCatalogSchema", counting)

    snap = live_ml.catalog_snapshot(_a_snapshot_id(live_ml))
    assert snap is not None
    assert calls["n"] == 0, "catalog_snapshot must not fetch /schema"


def test_catalog_snapshot_memoized_per_id(live_ml):
    """Repeated catalog_snapshot() for the same snapshot id returns the same object."""
    sid = _a_snapshot_id(live_ml)
    first = live_ml.catalog_snapshot(sid)
    second = live_ml.catalog_snapshot(sid)
    assert first is second


def test_catalog_snapshot_cache_holds_one_entry_per_id(live_ml):
    """catalog_snapshot caches exactly one instance per snapshot id."""
    sid = _a_snapshot_id(live_ml)
    a = live_ml.catalog_snapshot(sid)
    assert len(live_ml._snapshot_cache) == 1
    b = live_ml.catalog_snapshot(sid)
    assert a is b
    assert len(live_ml._snapshot_cache) == 1


def _model_fingerprint(model) -> dict:
    """A structural fingerprint: {schema: {table: sorted(column names)}}."""
    fp: dict[str, dict[str, list[str]]] = {}
    for sname, schema in model.model.schemas.items():
        fp[sname] = {
            tname: sorted(c.name for c in table.columns)
            for tname, table in schema.tables.items()
        }
    return fp


def test_reused_schema_model_matches_fetched(live_ml):
    """The schema-reusing snapshot model is structurally identical to a fetched one."""
    from deriva.core.ermrest_catalog import ErmrestSnapshot

    raw_snaptime = _a_snapshot_id(live_ml)
    # catalog_snapshot expects the compound "<catalog_id>@<snaptime>" form
    # (what _version_snapshot_catalog_id produces in production); a bare
    # snaptime has no '@' and deriva-py would treat it as a catalog id.
    compound_sid = f"{live_ml.catalog_id}@{raw_snaptime}"

    reused = live_ml.catalog_snapshot(compound_sid)

    # Build the same snapshot WITHOUT reuse — force a real getCatalogSchema.
    fetched = DerivaML(
        live_ml.host_name,
        compound_sid,
        working_dir=live_ml.working_dir,
        ml_schema=live_ml.ml_schema,
        credential=live_ml.credential,
    )

    # Both must be genuinely snapshot-pinned, not live-catalog connections.
    assert isinstance(reused.catalog, ErmrestSnapshot)
    assert isinstance(fetched.catalog, ErmrestSnapshot)

    assert _model_fingerprint(reused.model) == _model_fingerprint(fetched.model)
