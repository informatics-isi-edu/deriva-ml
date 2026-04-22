"""Live smoke test for S4 cache-backed offline mode.

Goes beyond `test_offline_init.py` — that file tests the init mechanics
(cache populated, drift warning, host mismatch). This file tests
real-world usage: online once, then offline, then actually use
``ml.model`` for schema inspection and confirm ``ml.catalog``
properly refuses network access.

Gated on ``DERIVA_HOST`` — skips if unset.
"""
from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="S4 offline-mode smoke test requires DERIVA_HOST",
)
def test_offline_mode_end_to_end_usage(catalog_manager, tmp_path):
    """After going online-then-offline, the instance is usable for
    schema inspection and rejects any catalog access."""
    from deriva_ml import ConnectionMode, DerivaML
    from deriva_ml.core.catalog_stub import CatalogStub
    from deriva_ml.core.exceptions import DerivaMLReadOnlyError

    catalog_manager.reset()

    # --- Phase 1: online init populates the cache and captures the
    # expected schema shape.
    ml_online = DerivaML(
        hostname=catalog_manager.hostname,
        catalog_id=catalog_manager.catalog_id,
        working_dir=tmp_path,
        mode=ConnectionMode.online,
    )
    expected_ml_schema = ml_online.model.ml_schema
    expected_default_schema = ml_online.model.default_schema
    expected_domain_schemas = set(ml_online.model.domain_schemas)

    # --- Phase 2: offline init reads the cache. This should succeed
    # with zero network traffic.
    ml_offline = DerivaML(
        hostname=catalog_manager.hostname,
        catalog_id=catalog_manager.catalog_id,
        working_dir=tmp_path,
        mode=ConnectionMode.offline,
    )

    # --- Contract 1: catalog is a CatalogStub, not a real ErmrestCatalog.
    assert isinstance(ml_offline.catalog, CatalogStub), (
        f"expected CatalogStub, got {type(ml_offline.catalog).__name__}"
    )

    # --- Contract 2: the model is usable for schema inspection.
    # ml_schema / default_schema / domain_schemas should survive the
    # round-trip through the cache.
    assert ml_offline.model.ml_schema == expected_ml_schema
    assert ml_offline.model.default_schema == expected_default_schema
    assert set(ml_offline.model.domain_schemas) == expected_domain_schemas

    # --- Contract 3: model methods that don't touch the catalog work.
    # find_assets and find_vocabularies walk self.model.schemas which
    # was reconstructed from the cached /schema dict.
    offline_asset_names = {t.name for t in ml_offline.model.find_assets()}
    online_asset_names = {t.name for t in ml_online.model.find_assets()}
    assert offline_asset_names == online_asset_names, (
        f"asset discovery differs; online={online_asset_names} "
        f"offline={offline_asset_names}"
    )

    # --- Contract 4: any reach-through to the catalog raises.
    with pytest.raises(DerivaMLReadOnlyError) as ei:
        ml_offline.catalog.getCatalogModel()
    assert "offline" in str(ei.value).lower()
    assert "getCatalogModel" in str(ei.value)

    # --- Contract 5: refresh_schema() refuses in offline mode
    # (no pending rows needed — mode check fires first).
    with pytest.raises(DerivaMLReadOnlyError) as ei:
        ml_offline.refresh_schema()
    assert "online" in str(ei.value).lower()
