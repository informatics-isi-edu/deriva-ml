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


@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="requires a live catalog at DERIVA_HOST",
)
def test_refresh_schema_refuses_with_pending_rows(test_ml):
    """Stage a pending row, call refresh_schema() → DerivaMLSchemaRefreshBlocked."""
    from datetime import datetime, timezone
    from deriva_ml import ConnectionMode
    from deriva_ml.core.exceptions import DerivaMLSchemaRefreshBlocked
    from deriva_ml.execution.state_store import (
        ExecutionStatus, PendingRowStatus,
    )

    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)
    # Need an execution row that the pending_row FK can reference.
    store.insert_execution(
        rid="EXE-REFRESH-TEST",
        workflow_rid=None,
        description="refresh test",
        config_json="{}",
        status=ExecutionStatus.Created,
        mode=ConnectionMode.online,
        working_dir_rel="execution/EXE-REFRESH-TEST",
        created_at=now,
        last_activity=now,
    )
    store.insert_pending_row(
        execution_rid="EXE-REFRESH-TEST",
        key="k",
        target_schema="deriva-ml", target_table="Subject",
        metadata_json="{}",
        created_at=now,
        status=PendingRowStatus.staged,
    )

    with pytest.raises(DerivaMLSchemaRefreshBlocked) as ei:
        test_ml.refresh_schema()
    assert "drained" in str(ei.value).lower()


@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="requires a live catalog at DERIVA_HOST",
)
def test_refresh_schema_force_succeeds(test_ml, caplog):
    """refresh_schema(force=True) succeeds even with pending rows and logs."""
    import logging
    from datetime import datetime, timezone
    from deriva_ml import ConnectionMode
    from deriva_ml.execution.state_store import (
        ExecutionStatus, PendingRowStatus,
    )

    store = test_ml.workspace.execution_state_store()
    now = datetime.now(timezone.utc)
    store.insert_execution(
        rid="EXE-REFRESH-FORCE",
        workflow_rid=None,
        description="refresh force test",
        config_json="{}",
        status=ExecutionStatus.Created,
        mode=ConnectionMode.online,
        working_dir_rel="execution/EXE-REFRESH-FORCE",
        created_at=now,
        last_activity=now,
    )
    store.insert_pending_row(
        execution_rid="EXE-REFRESH-FORCE",
        key="k",
        target_schema="deriva-ml", target_table="Subject",
        metadata_json="{}",
        created_at=now,
        status=PendingRowStatus.staged,
    )

    with caplog.at_level(logging.INFO, logger="deriva_ml"):
        test_ml.refresh_schema(force=True)
    # Cache file exists after refresh
    cache_file = Path(test_ml.working_dir) / "schema-cache.json"
    assert cache_file.is_file()
    # Log records the transition
    assert any(
        "refreshed" in r.getMessage().lower()
        for r in caplog.records
    ), f"expected 'refreshed' in log; got {[r.getMessage() for r in caplog.records]}"


@pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="requires a live catalog at DERIVA_HOST",
)
def test_online_drift_warning(catalog_manager, tmp_path, caplog):
    """Cache at stale snapshot id → re-init online → warning logged; cache unchanged."""
    import json as _json
    import logging
    from deriva_ml import ConnectionMode, DerivaML
    from deriva_ml.core.schema_cache import SchemaCache

    catalog_manager.reset()

    # First, online to populate the cache normally.
    DerivaML(
        hostname=catalog_manager.hostname,
        catalog_id=catalog_manager.catalog_id,
        mode=ConnectionMode.online,
        working_dir=tmp_path,
    )

    # Corrupt the snapshot_id to force drift.
    cache = SchemaCache(tmp_path)
    data = cache.load()
    data["snapshot_id"] = "fake-drift-snapshot"
    (tmp_path / "schema-cache.json").write_text(_json.dumps(data))

    # Re-init online → drift warning.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="deriva_ml"):
        DerivaML(
            hostname=catalog_manager.hostname,
            catalog_id=catalog_manager.catalog_id,
            mode=ConnectionMode.online,
            working_dir=tmp_path,
        )
    assert any(
        "snapshot" in r.getMessage().lower()
        for r in caplog.records
    ), f"expected drift-warning log; got {[r.getMessage() for r in caplog.records]}"

    # Cache snapshot_id still says the fake value — we DIDN'T auto-refresh.
    reloaded = _json.loads((tmp_path / "schema-cache.json").read_text())
    assert reloaded["snapshot_id"] == "fake-drift-snapshot"
