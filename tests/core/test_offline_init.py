"""Integration tests for offline-mode ``DerivaML.__init__`` and
``refresh_schema()``. Tests that need a live catalog are gated on
``DERIVA_HOST`` and skip when it's unset. Tests that don't need
a live catalog (e.g., missing-cache error, host/catalog mismatch)
always run.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_offline_without_cache_raises(tmp_path):
    """Offline init against a workspace with no cache file raises
    ``DerivaMLConfigurationError``."""
    from deriva_ml import ConnectionMode, DerivaML
    from deriva_ml.core.exceptions import DerivaMLConfigurationError

    with pytest.raises(DerivaMLConfigurationError) as ei:
        DerivaML(
            hostname="example.org",
            catalog_id="1",
            mode=ConnectionMode.offline,
            working_dir=tmp_path,
        )
    assert "offline" in str(ei.value).lower()
    assert "cache" in str(ei.value).lower()


def test_offline_hostname_mismatch_raises(tmp_path):
    """Cache written for host A; offline init with host B → error."""
    from deriva_ml import ConnectionMode, DerivaML
    from deriva_ml.core.exceptions import DerivaMLConfigurationError
    from deriva_ml.core.schema_cache import SchemaCache

    # Plant a cache file for host A.
    cache = SchemaCache(tmp_path)
    cache.write(
        snapshot_id="fake-snap",
        hostname="host-a.example.org",
        catalog_id="99",
        ml_schema="deriva-ml",
        schema={
            "schemas": {
                "deriva-ml": {
                    "schema_name": "deriva-ml",
                    "tables": {},
                    "annotations": {},
                    "comment": None,
                },
            },
            "acls": {},
            "annotations": {},
        },
    )

    # Offline init with a DIFFERENT host → error.
    with pytest.raises(DerivaMLConfigurationError) as ei:
        DerivaML(
            hostname="host-b.example.org",
            catalog_id="99",
            mode=ConnectionMode.offline,
            working_dir=tmp_path,
        )
    msg = str(ei.value)
    assert "host-a.example.org" in msg
    assert "host-b.example.org" in msg


@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="requires a live catalog at DERIVA_HOST",
)
def test_online_first_populates_cache(test_ml):
    """After an online __init__, schema-cache.json exists in the workspace.

    The ``test_ml`` fixture constructs DerivaML online against
    DERIVA_HOST; its working_dir should contain the cache file.
    """
    cache_file = Path(test_ml.working_dir) / "schema-cache.json"
    assert cache_file.is_file(), (
        f"schema-cache.json not found in {test_ml.working_dir}"
    )
    import json as _json
    data = _json.loads(cache_file.read_text())
    assert "snapshot_id" in data
    assert data["hostname"] == test_ml.host_name
    assert "schema" in data
    assert "schemas" in data["schema"]


@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="requires a live catalog at DERIVA_HOST",
)
def test_offline_after_online_succeeds(catalog_manager, tmp_path):
    """Online once → cache written → offline init against same workspace works."""
    from deriva_ml import ConnectionMode, DerivaML
    from deriva_ml.core.catalog_stub import CatalogStub

    catalog_manager.reset()

    # Online once in tmp_path — populates the cache.
    DerivaML(
        hostname=catalog_manager.hostname,
        catalog_id=catalog_manager.catalog_id,
        mode=ConnectionMode.online,
        working_dir=tmp_path,
    )
    assert (tmp_path / "schema-cache.json").is_file()

    # Offline now against the same tmp_path — should work without network.
    ml_offline = DerivaML(
        hostname=catalog_manager.hostname,
        catalog_id=catalog_manager.catalog_id,
        mode=ConnectionMode.offline,
        working_dir=tmp_path,
    )
    assert isinstance(ml_offline.catalog, CatalogStub)
    assert ml_offline.model is not None


# NOTE: test_refresh_schema_refuses_with_pending_rows was deleted in
# the Phase 3 cleanup (audit §1.5). The pending-rows write surface is
# retired, so ``count_pending_rows()`` always returns 0 and the
# DerivaMLSchemaRefreshBlocked path is no longer reachable through any
# in-tree producer. The schema-refresh guard itself survives (it's
# cheap insurance against future writers), but it cannot be exercised
# from tests.


@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="requires a live catalog at DERIVA_HOST",
)
def test_refresh_schema_force_succeeds(test_ml, caplog):
    """refresh_schema(force=True) populates the schema cache and logs."""
    import logging

    with caplog.at_level(logging.INFO, logger="deriva_ml"):
        test_ml.refresh_schema(force=True)
    # Cache file exists after refresh
    cache_file = Path(test_ml.working_dir) / "schema-cache.json"
    assert cache_file.is_file()
    # Log records the transition
    assert any("refreshed" in r.getMessage().lower() for r in caplog.records), (
        f"expected 'refreshed' in log; got {[r.getMessage() for r in caplog.records]}"
    )


# NOTE: ``test_online_drift_warning`` was removed alongside this commit.
# Its premise -- that online init compares the cached snapshot to the
# live one and emits a warning on mismatch -- belonged to a pre-1f2e7223
# design. Schema-cache freshness is now handled in deriva-py's binding
# layer (auto-invalidation + If-None-Match revalidation), so the
# "cache at X, live at Y" condition can no longer arise at online init
# in a way that requires user action. The disk cache is now solely a
# bootstrap for offline mode; ``_init_online`` writes the live snapshot
# unconditionally, so any stale snapshot on disk is just overwritten.
# The companion 340-line file ``test_refresh_schema_warning.py`` was
# deleted in 1f2e7223; this stub records that this file's drift test
# was a missed cleanup from that commit.
