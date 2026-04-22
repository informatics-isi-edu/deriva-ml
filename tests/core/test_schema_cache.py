"""Unit tests for SchemaCache — the workspace-backed schema store."""
from __future__ import annotations

import pytest


def test_exists_false_when_missing(tmp_path):
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    assert cache.exists() is False


def test_snapshot_id_none_when_missing(tmp_path):
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    assert cache.snapshot_id() is None


def test_write_then_load_round_trip(tmp_path):
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    schema = {"schemas": {"deriva-ml": {}}, "acls": {}, "annotations": {}}
    cache.write(
        snapshot_id="2026-04-21T18:00:00Z",
        hostname="example.org",
        catalog_id="42",
        ml_schema="deriva-ml",
        schema=schema,
    )
    assert cache.exists() is True
    assert cache.snapshot_id() == "2026-04-21T18:00:00Z"
    loaded = cache.load()
    assert loaded["snapshot_id"] == "2026-04-21T18:00:00Z"
    assert loaded["hostname"] == "example.org"
    assert loaded["catalog_id"] == "42"
    assert loaded["ml_schema"] == "deriva-ml"
    assert loaded["schema"] == schema


def test_load_missing_raises(tmp_path):
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    with pytest.raises(FileNotFoundError):
        cache.load()


def test_corrupt_cache_raises_configuration_error(tmp_path):
    from deriva_ml.core.exceptions import DerivaMLConfigurationError
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    # Write garbage where the cache file should live
    cache._path.write_text("{ this is not valid json")
    with pytest.raises(DerivaMLConfigurationError) as ei:
        cache.load()
    assert "corrupt" in str(ei.value).lower()


def test_write_is_atomic_no_partial_file_on_crash(tmp_path, monkeypatch):
    """If os.replace fails mid-write, the old cache remains intact."""
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)

    # Populate a valid initial cache
    cache.write(
        snapshot_id="v1", hostname="h", catalog_id="c",
        ml_schema="ml", schema={"a": 1},
    )
    assert cache.snapshot_id() == "v1"

    # Simulate a crash: make os.replace raise after the tmp is written.
    import os as os_mod
    original_replace = os_mod.replace

    def crashing_replace(*args, **kwargs):
        raise OSError("simulated crash during rename")

    monkeypatch.setattr(os_mod, "replace", crashing_replace)

    with pytest.raises(OSError, match="simulated"):
        cache.write(
            snapshot_id="v2", hostname="h", catalog_id="c",
            ml_schema="ml", schema={"a": 2},
        )

    # After restore, the OLD cache is still intact.
    monkeypatch.setattr(os_mod, "replace", original_replace)
    assert cache.snapshot_id() == "v1"
    assert cache.load()["schema"] == {"a": 1}


def test_write_atomic_helper_exists_and_writes_payload(tmp_path):
    """_write_atomic is the private helper pin/unpin will reuse."""
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    payload = {
        "snapshot_id": "s1",
        "hostname": "h",
        "catalog_id": "c",
        "ml_schema": "ml",
        "schema": {"k": "v"},
    }
    cache._write_atomic(payload)
    assert cache.exists()
    import json
    assert json.loads(cache._path.read_text()) == payload


def _populate(cache, schema_payload=None):
    """Helper: write a minimal valid cache for pin tests."""
    cache.write(
        snapshot_id="s0",
        hostname="h",
        catalog_id="c",
        ml_schema="ml",
        schema=schema_payload or {"schemas": {}},
    )


def test_pin_on_unpinned_cache_sets_fields(tmp_path):
    from datetime import datetime, timezone
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    _populate(cache)
    before = datetime.now(timezone.utc)
    cache.pin(reason="paper repro")
    after = datetime.now(timezone.utc)
    status = cache.pin_status()
    assert status.pinned is True
    assert status.pin_reason == "paper repro"
    assert status.pinned_snapshot_id == "s0"
    assert status.pinned_at is not None
    assert before <= status.pinned_at <= after


def test_pin_without_reason(tmp_path):
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    _populate(cache)
    cache.pin()
    status = cache.pin_status()
    assert status.pinned is True
    assert status.pin_reason is None


def test_pin_idempotent_updates_metadata(tmp_path):
    import time
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    _populate(cache)
    cache.pin(reason="first")
    first = cache.pin_status()
    time.sleep(0.01)
    cache.pin(reason="second")
    second = cache.pin_status()
    assert second.pinned is True
    assert second.pin_reason == "second"
    assert second.pinned_at >= first.pinned_at


def test_unpin_clears_fields(tmp_path):
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    _populate(cache)
    cache.pin(reason="r")
    cache.unpin()
    status = cache.pin_status()
    assert status.pinned is False
    assert status.pinned_at is None
    assert status.pin_reason is None
    assert status.pinned_snapshot_id == "s0"


def test_unpin_on_unpinned_is_no_op(tmp_path):
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    _populate(cache)
    cache.unpin()  # should not raise
    status = cache.pin_status()
    assert status.pinned is False


def test_pin_status_on_missing_cache_raises(tmp_path):
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    import pytest
    with pytest.raises(FileNotFoundError):
        cache.pin_status()


def test_pin_persists_across_instances(tmp_path):
    from deriva_ml.core.schema_cache import SchemaCache
    a = SchemaCache(tmp_path)
    _populate(a)
    a.pin(reason="persist me")
    b = SchemaCache(tmp_path)
    status = b.pin_status()
    assert status.pinned is True
    assert status.pin_reason == "persist me"


def test_cache_file_format_has_nested_pin_object(tmp_path):
    """After pin, the JSON has a top-level ``pin`` object; unpin removes it."""
    import json
    from deriva_ml.core.schema_cache import SchemaCache
    cache = SchemaCache(tmp_path)
    _populate(cache)
    cache.pin(reason="x")
    raw = json.loads(cache._path.read_text())
    assert "pin" in raw
    assert raw["pin"]["reason"] == "x"
    assert "at" in raw["pin"]
    cache.unpin()
    raw2 = json.loads(cache._path.read_text())
    assert "pin" not in raw2
