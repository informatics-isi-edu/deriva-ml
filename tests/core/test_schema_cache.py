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
    from deriva_ml.core.schema_cache import SchemaCache
    from deriva_ml.core.exceptions import DerivaMLConfigurationError
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
