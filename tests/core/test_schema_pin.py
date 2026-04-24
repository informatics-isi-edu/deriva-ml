"""Integration tests for pin/unpin/diff on DerivaML.

Tests that don't need a live catalog (offline pin/unpin, missing
cache, diff_schema offline) always run. Tests that need a live
catalog are gated on ``DERIVA_HOST``.
"""
from __future__ import annotations

import logging
import os

import pytest

requires_catalog = pytest.mark.skipif(
    not os.environ.get("DERIVA_HOST"),
    reason="requires a live catalog at DERIVA_HOST",
)


# ------------------------- offline-capable tests --------------------------

def test_pin_schema_offline_returns_none(tmp_path):
    """Offline mode: pin_schema persists a pin and returns None."""
    from deriva_ml import ConnectionMode, DerivaML
    from deriva_ml.core.schema_cache import SchemaCache

    # Plant a cache so offline init succeeds.
    SchemaCache(tmp_path).write(
        snapshot_id="s0",
        hostname="h",
        catalog_id="1",
        ml_schema="deriva-ml",
        schema={
            "schemas": {
                "deriva-ml": {
                    "schema_name": "deriva-ml",
                    "tables": {},
                    "annotations": {},
                    "comment": None,
                }
            },
            "acls": {},
            "annotations": {},
        },
    )
    ml = DerivaML(
        hostname="h", catalog_id="1",
        mode=ConnectionMode.offline, working_dir=tmp_path,
    )
    result = ml.pin_schema(reason="offline test")
    assert result is None
    assert ml.pin_status().pinned is True
    assert ml.pin_status().pin_reason == "offline test"


def test_unpin_schema_works_offline(tmp_path):
    from deriva_ml import ConnectionMode, DerivaML
    from deriva_ml.core.schema_cache import SchemaCache
    SchemaCache(tmp_path).write(
        snapshot_id="s0", hostname="h", catalog_id="1",
        ml_schema="deriva-ml",
        schema={"schemas": {"deriva-ml": {"schema_name": "deriva-ml", "tables": {}}}},
    )
    ml = DerivaML(hostname="h", catalog_id="1",
                  mode=ConnectionMode.offline, working_dir=tmp_path)
    ml.pin_schema(reason="x")
    assert ml.pin_status().pinned is True
    ml.unpin_schema()
    assert ml.pin_status().pinned is False


def test_pin_status_reflects_cache_state(tmp_path):
    from deriva_ml import ConnectionMode, DerivaML
    from deriva_ml.core.schema_cache import SchemaCache
    SchemaCache(tmp_path).write(
        snapshot_id="s-current", hostname="h", catalog_id="1",
        ml_schema="deriva-ml",
        schema={"schemas": {"deriva-ml": {"schema_name": "deriva-ml", "tables": {}}}},
    )
    ml = DerivaML(hostname="h", catalog_id="1",
                  mode=ConnectionMode.offline, working_dir=tmp_path)
    status = ml.pin_status()
    assert status.pinned is False
    assert status.pinned_snapshot_id == "s-current"


def test_diff_schema_offline_raises(tmp_path):
    from deriva_ml import ConnectionMode, DerivaML
    from deriva_ml.core.exceptions import DerivaMLReadOnlyError
    from deriva_ml.core.schema_cache import SchemaCache
    SchemaCache(tmp_path).write(
        snapshot_id="s0", hostname="h", catalog_id="1",
        ml_schema="deriva-ml",
        schema={"schemas": {"deriva-ml": {"schema_name": "deriva-ml", "tables": {}}}},
    )
    ml = DerivaML(hostname="h", catalog_id="1",
                  mode=ConnectionMode.offline, working_dir=tmp_path)
    with pytest.raises(DerivaMLReadOnlyError):
        ml.diff_schema()


# --------------------------- live-catalog tests ---------------------------

@requires_catalog
def test_pin_schema_online_no_drift_returns_none(test_ml):
    """Freshly-initialized online ml: cache is at live snapshot → no drift."""
    result = test_ml.pin_schema(reason="no-drift test")
    try:
        assert result is None
        assert test_ml.pin_status().pinned is True
    finally:
        test_ml.unpin_schema()


@requires_catalog
def test_pin_schema_online_with_drift_returns_diff_and_logs_warning(
    test_ml, caplog,
):
    """Forge a drift scenario: rewrite the cache with a bogus snapshot
    + schema that differs from live; pin; expect SchemaDiff + warning."""
    from deriva_ml.core.schema_cache import SchemaCache

    # Force a cache with a known-fake snapshot_id and a missing-table
    # payload so _compute_diff reports an 'added_tables' result.
    cache = SchemaCache(test_ml.working_dir)
    current = cache.load()
    forged_schema = {
        "schemas": {
            current["ml_schema"]: {
                "schema_name": current["ml_schema"],
                "tables": {},   # live will have many tables → all added
            }
        }
    }
    cache.write(
        snapshot_id="FORGED-SNAPSHOT-00",
        hostname=current["hostname"],
        catalog_id=current["catalog_id"],
        ml_schema=current["ml_schema"],
        schema=forged_schema,
    )

    caplog.set_level(logging.WARNING, logger="deriva_ml")
    diff = test_ml.pin_schema(reason="drift test")
    try:
        assert diff is not None
        assert not diff.is_empty()
        assert len(diff.added_tables) > 0  # live has tables our forge didn't
        assert any(
            "drift" in r.getMessage().lower() or "pin_schema" in r.getMessage().lower()
            for r in caplog.records
        )
    finally:
        test_ml.unpin_schema()


@requires_catalog
def test_refresh_schema_refuses_when_pinned(test_ml):
    from deriva_ml.core.exceptions import DerivaMLSchemaPinned
    test_ml.pin_schema(reason="refuse test")
    try:
        with pytest.raises(DerivaMLSchemaPinned) as ei:
            test_ml.refresh_schema()
        assert "pinned" in str(ei.value).lower()
    finally:
        test_ml.unpin_schema()


@requires_catalog
def test_refresh_schema_refuses_when_pinned_even_with_force(test_ml):
    from deriva_ml.core.exceptions import DerivaMLSchemaPinned
    test_ml.pin_schema(reason="force doesn't bypass pin")
    try:
        with pytest.raises(DerivaMLSchemaPinned):
            test_ml.refresh_schema(force=True)
    finally:
        test_ml.unpin_schema()


@requires_catalog
def test_unpin_then_refresh_succeeds(test_ml):
    test_ml.pin_schema(reason="transient")
    test_ml.unpin_schema()
    # Should NOT raise now.
    test_ml.refresh_schema()


@requires_catalog
def test_diff_schema_online_returns_diff(test_ml):
    """diff_schema returns a SchemaDiff (possibly empty) online."""
    from deriva_ml.core.schema_diff import SchemaDiff
    diff = test_ml.diff_schema()
    assert isinstance(diff, SchemaDiff)
    # Fresh test_ml: cache IS live, so diff should be empty.
    assert diff.is_empty()
