"""Unit tests for local_db.result_cache.ResultCache."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from deriva_ml.local_db.workspace import Workspace


@pytest.fixture
def cache(tmp_path: Path):
    ws = Workspace(working_dir=tmp_path, hostname="h", catalog_id="1")
    from deriva_ml.local_db.result_cache import ResultCache

    rc = ResultCache(ws.engine)
    rc.ensure_schema()
    yield rc
    ws.close()


def _make_meta(
    cache_key: str,
    source: str = "catalog",
    tool_name: str = "table_read",
    params: dict | None = None,
    columns: list[str] | None = None,
    row_count: int = 0,
    ttl_seconds: int | None = None,
):
    from deriva_ml.local_db.result_cache import CachedResultMeta

    return CachedResultMeta(
        cache_key=cache_key,
        source=source,
        tool_name=tool_name,
        params=params or {},
        columns=columns or ["id", "name"],
        row_count=row_count,
        ttl_seconds=ttl_seconds,
    )


class TestCacheKey:
    def test_deterministic(self):
        from deriva_ml.local_db.result_cache import ResultCache

        k1 = ResultCache.cache_key("table_read", table="Subject", limit=10)
        k2 = ResultCache.cache_key("table_read", table="Subject", limit=10)
        assert k1 == k2

    def test_different_params_different_key(self):
        from deriva_ml.local_db.result_cache import ResultCache

        k1 = ResultCache.cache_key("table_read", table="Subject")
        k2 = ResultCache.cache_key("table_read", table="Image")
        assert k1 != k2

    def test_different_tool_different_key(self):
        from deriva_ml.local_db.result_cache import ResultCache

        k1 = ResultCache.cache_key("table_read", table="Subject")
        k2 = ResultCache.cache_key("denormalize", table="Subject")
        assert k1 != k2

    def test_sorted_params_same_key(self):
        from deriva_ml.local_db.result_cache import ResultCache

        k1 = ResultCache.cache_key("table_read", table="Subject", limit=10)
        k2 = ResultCache.cache_key("table_read", limit=10, table="Subject")
        assert k1 == k2

    def test_none_params_excluded(self):
        from deriva_ml.local_db.result_cache import ResultCache

        k1 = ResultCache.cache_key("table_read", table="Subject", extra=None)
        k2 = ResultCache.cache_key("table_read", table="Subject")
        assert k1 == k2

    def test_key_prefix(self):
        from deriva_ml.local_db.result_cache import ResultCache

        k = ResultCache.cache_key("tool")
        assert k.startswith("rc_")

    def test_sorted_list_params(self):
        from deriva_ml.local_db.result_cache import ResultCache

        k1 = ResultCache.cache_key("tool", cols=["b", "a"])
        k2 = ResultCache.cache_key("tool", cols=["a", "b"])
        assert k1 == k2


class TestResultCacheStore:
    def test_store_and_get_meta(self, cache):
        rows = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        key = "rc_abc12345678"
        meta = _make_meta(key, columns=["id", "name"], row_count=2)
        cache.store(key, ["id", "name"], rows, meta)

        got = cache.get_meta(key)
        assert got is not None
        assert got.cache_key == key
        assert got.row_count == 2
        assert got.columns == ["id", "name"]
        assert got.source == "catalog"
        assert got.tool_name == "table_read"

    def test_store_replaces_existing(self, cache):
        key = "rc_replace_me0"
        rows1 = [{"id": 1, "name": "Alice"}]
        rows2 = [{"id": 2, "name": "Bob"}, {"id": 3, "name": "Carol"}]
        meta1 = _make_meta(key, row_count=1)
        meta2 = _make_meta(key, row_count=2)
        cache.store(key, ["id", "name"], rows1, meta1)
        cache.store(key, ["id", "name"], rows2, meta2)

        got = cache.get_meta(key)
        assert got is not None
        assert got.row_count == 2

        result = cache.query(key)
        assert result is not None
        assert result.total_count == 2

    def test_get_meta_missing(self, cache):
        assert cache.get_meta("rc_nothere0000") is None

    def test_store_with_dots_and_dashes_in_columns(self, cache):
        key = "rc_dotdash12345"
        rows = [{"schema.name": "isa", "col-name": "val1"}]
        meta = _make_meta(key, columns=["schema.name", "col-name"], row_count=1)
        cache.store(key, ["schema.name", "col-name"], rows, meta)

        result = cache.query(key)
        assert result is not None
        assert result.total_count == 1
        assert "schema.name" in result.columns
        assert "col-name" in result.columns


class TestResultCacheQuery:
    def _setup(self, cache, key="rc_querytest1234"):
        rows = [
            {"id": 1, "name": "Alice", "score": 90},
            {"id": 2, "name": "Bob", "score": 75},
            {"id": 3, "name": "Carol", "score": 85},
            {"id": 4, "name": "Dave", "score": 60},
            {"id": 5, "name": "Eve", "score": 95},
        ]
        meta = _make_meta(key, columns=["id", "name", "score"], row_count=5)
        cache.store(key, ["id", "name", "score"], rows, meta)
        return key

    def test_query_no_filters(self, cache):
        key = self._setup(cache)
        result = cache.query(key)
        assert result is not None
        assert result.total_count == 5
        assert result.count == 5

    def test_query_sort_ascending(self, cache):
        key = self._setup(cache)
        result = cache.query(key, sort_by="score", sort_desc=False)
        scores = [r["score"] for r in result.rows]
        assert scores == sorted(scores)

    def test_query_sort_descending(self, cache):
        key = self._setup(cache)
        result = cache.query(key, sort_by="score", sort_desc=True)
        scores = [r["score"] for r in result.rows]
        assert scores == sorted(scores, reverse=True)

    def test_query_filter_substring(self, cache):
        key = self._setup(cache)
        result = cache.query(key, filter_col="name", filter_val="e")
        # Alice, Dave, Eve contain "e"
        names = {r["name"] for r in result.rows}
        assert "Alice" in names
        assert "Dave" in names
        assert "Eve" in names
        assert "Bob" not in names
        assert "Carol" not in names

    def test_query_limit_offset(self, cache):
        key = self._setup(cache, key="rc_pagination00000")
        result_p1 = cache.query(key, limit=2, offset=0)
        result_p2 = cache.query(key, limit=2, offset=2)

        assert result_p1.count == 2
        assert result_p2.count == 2
        assert result_p1.total_count == 5
        assert result_p2.total_count == 5

        # No overlapping rows
        ids_p1 = {r["id"] for r in result_p1.rows}
        ids_p2 = {r["id"] for r in result_p2.rows}
        assert ids_p1.isdisjoint(ids_p2)

    def test_query_empty_table(self, cache):
        key = "rc_emptytable000"
        meta = _make_meta(key, columns=["id", "name"], row_count=0)
        cache.store(key, ["id", "name"], [], meta)
        result = cache.query(key)
        assert result is not None
        assert result.total_count == 0
        assert result.rows == []

    def test_query_missing_key(self, cache):
        assert cache.query("rc_missing000000") is None

    def test_query_returns_correct_structure(self, cache):
        key = self._setup(cache, key="rc_structure00000")
        result = cache.query(key, limit=3)
        assert isinstance(result.columns, list)
        assert isinstance(result.rows, list)
        assert isinstance(result.count, int)
        assert isinstance(result.total_count, int)
        assert result.cache_key == key
        assert result.source == "catalog"


class TestResultCacheHas:
    def test_has_true_for_stored(self, cache):
        key = "rc_hastruetest0"
        meta = _make_meta(key)
        cache.store(key, ["id", "name"], [{"id": 1, "name": "A"}], meta)
        assert cache.has(key) is True

    def test_has_false_for_missing(self, cache):
        assert cache.has("rc_nothere00000") is False

    def test_has_false_for_expired(self, cache):
        key = "rc_expiredhas000"
        meta = _make_meta(key, ttl_seconds=0)
        cache.store(key, ["id"], [{"id": 1}], meta)
        # ttl=0 means expired immediately
        assert cache.has(key) is False


class TestResultCacheList:
    def test_list_returns_all_entries(self, cache):
        keys = ["rc_list1test0000", "rc_list2test0000", "rc_list3test0000"]
        for key in keys:
            meta = _make_meta(key, source=f"source_{key[-4:]}")
            cache.store(key, ["id"], [{"id": 1}], meta)

        listed = cache.list_cached()
        listed_keys = {m.cache_key for m in listed}
        for k in keys:
            assert k in listed_keys

    def test_list_excludes_expired(self, cache):
        live_key = "rc_listlive00000"
        exp_key = "rc_listexpired00"
        live_meta = _make_meta(live_key)
        exp_meta = _make_meta(exp_key, ttl_seconds=0)

        cache.store(live_key, ["id"], [{"id": 1}], live_meta)
        cache.store(exp_key, ["id"], [{"id": 2}], exp_meta)

        listed_keys = {m.cache_key for m in cache.list_cached()}
        assert live_key in listed_keys
        assert exp_key not in listed_keys

    def test_list_lazy_cleanup(self, cache):
        exp_key = "rc_lazycleanup00"
        exp_meta = _make_meta(exp_key, ttl_seconds=0)
        cache.store(exp_key, ["id"], [{"id": 1}], exp_meta)

        # After listing, expired entry should be cleaned from registry
        cache.list_cached()
        assert cache.get_meta(exp_key) is None


class TestResultCacheInvalidate:
    def test_invalidate_by_key(self, cache):
        key = "rc_invkeytest000"
        meta = _make_meta(key)
        cache.store(key, ["id"], [{"id": 1}], meta)

        count = cache.invalidate(cache_key=key)
        assert count == 1
        assert cache.has(key) is False

    def test_invalidate_by_source(self, cache):
        keys = ["rc_invsrc_a00000", "rc_invsrc_b00000", "rc_invsrc_c00000"]
        for key in keys:
            meta = _make_meta(key, source="feature_values")
            cache.store(key, ["id"], [{"id": 1}], meta)

        other_key = "rc_invsrc_other0"
        other_meta = _make_meta(other_key, source="catalog")
        cache.store(other_key, ["id"], [{"id": 1}], other_meta)

        count = cache.invalidate(source="feature_values")
        assert count == 3
        for key in keys:
            assert cache.has(key) is False
        assert cache.has(other_key) is True

    def test_invalidate_all(self, cache):
        for i in range(3):
            key = f"rc_invall{i}00000"
            meta = _make_meta(key, source=f"src{i}")
            cache.store(key, ["id"], [{"id": 1}], meta)

        count = cache.invalidate()
        assert count >= 3

    def test_invalidate_nonexistent_returns_zero(self, cache):
        count = cache.invalidate(cache_key="rc_nothere00000")
        assert count == 0


class TestCachedResult:
    def _store(self, cache, key="rc_cached_handle0"):
        rows = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        meta = _make_meta(key, columns=["id", "name"], row_count=2)
        cache.store(key, ["id", "name"], rows, meta)
        return key

    def test_to_dataframe(self, cache):
        key = self._store(cache, key="rc_dataframe0000")
        result = cache.get(key)
        assert result is not None
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["id", "name"]

    def test_iter_rows(self, cache):
        key = self._store(cache, key="rc_iterrows00000")
        result = cache.get(key)
        assert result is not None
        rows = list(result.iter_rows())
        assert len(rows) == 2
        assert all(isinstance(r, dict) for r in rows)
        assert {r["name"] for r in rows} == {"Alice", "Bob"}

    def test_query_returns_query_result(self, cache):
        key = self._store(cache, key="rc_queryresult00")
        result = cache.get(key)
        assert result is not None
        qr = result.query(limit=1)
        from deriva_ml.local_db.result_cache import QueryResult

        assert isinstance(qr, QueryResult)
        assert qr.count == 1
        assert qr.total_count == 2

    def test_properties(self, cache):
        key = self._store(cache, key="rc_props00000000")
        result = cache.get(key)
        assert result is not None
        assert result.cache_key == key
        assert result.source == "catalog"
        assert result.row_count == 2
        assert result.columns == ["id", "name"]
        assert isinstance(result.fetched_at, datetime)

    def test_invalidate_removes(self, cache):
        key = self._store(cache, key="rc_invalidate000")
        result = cache.get(key)
        assert result is not None
        result.invalidate()
        assert cache.has(key) is False
        assert cache.get(key) is None

    def test_get_missing_returns_none(self, cache):
        assert cache.get("rc_nothere00000") is None


class TestTTL:
    def test_ttl_none_never_expires(self, cache):
        key = "rc_ttlnever00000"
        meta = _make_meta(key, ttl_seconds=None)
        cache.store(key, ["id"], [{"id": 1}], meta)
        assert cache.has(key) is True

    def test_ttl_zero_expired_immediately(self, cache):
        key = "rc_ttlzero00000"
        meta = _make_meta(key, ttl_seconds=0)
        cache.store(key, ["id"], [{"id": 1}], meta)
        assert cache.has(key) is False

    def test_ttl_expiry_after_sleep(self, cache):
        key = "rc_ttl1sleep0000"
        meta = _make_meta(key, ttl_seconds=1)
        cache.store(key, ["id"], [{"id": 1}], meta)
        # Should be valid immediately
        assert cache.has(key) is True
        time.sleep(1.1)
        assert cache.has(key) is False

    def test_cached_result_meta_is_expired(self):
        meta = _make_meta("rc_metaexpired00", ttl_seconds=0)
        assert meta.is_expired() is True

    def test_cached_result_meta_not_expired(self):
        meta = _make_meta("rc_metaalive0000", ttl_seconds=3600)
        assert meta.is_expired() is False

    def test_cached_result_meta_no_ttl(self):
        meta = _make_meta("rc_metanottl000", ttl_seconds=None)
        assert meta.is_expired() is False

    def test_age_seconds(self):
        meta = _make_meta("rc_age000000000")
        assert meta.age_seconds() >= 0.0
        assert meta.age_seconds() < 5.0

    def test_to_summary(self):
        meta = _make_meta("rc_summary000000", tool_name="table_read", row_count=42)
        s = meta.to_summary()
        assert isinstance(s, dict)
        assert s["cache_key"] == "rc_summary000000"
        assert s["row_count"] == 42
        assert "created_at" in s or "age_seconds" in s
