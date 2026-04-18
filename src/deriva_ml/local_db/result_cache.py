"""Cached tabular-read API for the workspace's main.db.

Each cached result is stored as a separate SQLite table named by its
``cache_key`` (e.g. ``rc_a1b2c3d4e5f67890``).  A registry table
``cached_results_registry`` tracks metadata, TTL, and re-query support.

This is the API that MCP will eventually delegate to for ``preview_table``,
``denormalize``, and similar read operations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generator

import pandas as pd
from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    delete,
    insert,
    select,
    text,
)
from sqlalchemy.engine import Engine

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CachedResultMeta:
    """Metadata stored in the registry for a single cached result.

    One row in ``cached_results_registry`` per cache entry.  The actual data
    is stored in a separate dynamically-named table (e.g., ``rc_a1b2c3d4e5f67890``).

    Attributes:
        cache_key: Unique identifier for this cache entry (``rc_`` + 16 hex chars).
        source: Origin of the data, e.g. ``"catalog"``, ``"bag"``, ``"denormalize"``,
            or ``"feature_values"``.
        tool_name: MCP/API tool that produced this result, e.g. ``"table_read"``
            or ``"denormalize"``.
        params: Query parameters that produced this result (stored for display and
            cache-key derivation).
        columns: Ordered list of column names in the data table.
        row_count: Number of rows stored.
        created_at: Unix timestamp when the entry was created.
        ttl_seconds: Time-to-live in seconds; ``None`` means never expire.
    """

    cache_key: str
    source: str  # "catalog", "bag", "denormalize", "feature_values"
    tool_name: str  # "table_read", "denormalize", etc.
    params: dict  # query parameters (for display/dedup)
    columns: list[str]  # column names in the cached table
    row_count: int = 0
    created_at: float = field(default_factory=time.time)
    ttl_seconds: int | None = None  # None = never expire

    def is_expired(self) -> bool:
        """Return True if this entry has exceeded its TTL."""
        if self.ttl_seconds is None:
            return False
        return self.age_seconds() >= self.ttl_seconds

    def age_seconds(self) -> float:
        """Return the number of seconds since this entry was created."""
        return time.time() - self.created_at

    def to_summary(self) -> dict[str, Any]:
        """Return a human-readable summary dict."""
        return {
            "cache_key": self.cache_key,
            "source": self.source,
            "tool_name": self.tool_name,
            "params": self.params,
            "columns": self.columns,
            "row_count": self.row_count,
            "age_seconds": self.age_seconds(),
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "ttl_seconds": self.ttl_seconds,
            "expired": self.is_expired(),
        }


@dataclass
class QueryResult:
    """Result of querying a cached table with sort/filter/pagination.

    Returned by :meth:`ResultCache.query` and :meth:`CachedResult.query`.

    Attributes:
        columns: Ordered list of column names.
        rows: List of row dicts (after limit/offset applied).
        count: Number of rows in *rows* (``len(rows)``).
        total_count: Total rows matching the filter before pagination.
        cache_key: The cache entry this result came from.
        source: Source tag of the parent cache entry.
    """

    columns: list[str]
    rows: list[dict[str, Any]]
    count: int  # rows returned (after limit/offset)
    total_count: int  # total matching rows (before limit/offset)
    cache_key: str
    source: str


# ---------------------------------------------------------------------------
# CachedResult handle
# ---------------------------------------------------------------------------


class CachedResult:
    """Handle over a cached result stored as a table in ``main.db``.

    Returned by :meth:`ResultCache.get`, :meth:`Workspace.cached_table_read`,
    and :meth:`Workspace.cache_denormalized`.  Provides high-level access to the
    data without exposing SQLite internals.

    Typical usage::

        result = workspace.cached_table_read("Subject")
        df = result.to_dataframe()          # all rows as pandas DataFrame
        for row in result.iter_rows():      # streaming row-by-row
            ...
        paged = result.query(limit=50, offset=0, sort_by="Name")

    Attributes:
        cache_key: Unique ``rc_`` key identifying this cache entry.
        source: Origin tag (e.g., ``"catalog"``).
        row_count: Number of stored rows.
        columns: Ordered column names.
        fetched_at: :class:`datetime` when this entry was created.
    """

    def __init__(self, meta: CachedResultMeta, engine: Engine, result_cache: "ResultCache") -> None:
        self._meta = meta
        self._engine = engine
        self._cache = result_cache

    @property
    def cache_key(self) -> str:
        return self._meta.cache_key

    @property
    def source(self) -> str:
        return self._meta.source

    @property
    def row_count(self) -> int:
        return self._meta.row_count

    @property
    def columns(self) -> list[str]:
        return list(self._meta.columns)

    @property
    def fetched_at(self) -> datetime:
        return datetime.fromtimestamp(self._meta.created_at)

    def to_dataframe(self) -> pd.DataFrame:
        """Read the entire cached result as a DataFrame."""
        with self._engine.connect() as conn:
            quoted = ", ".join(_quote_col(c) for c in self._meta.columns)
            rows = conn.execute(text(f"SELECT {quoted} FROM {self._meta.cache_key}")).mappings().all()
        return pd.DataFrame([dict(r) for r in rows], columns=self._meta.columns)

    def iter_rows(self) -> Generator[dict, None, None]:
        """Yield each row as a dict."""
        with self._engine.connect() as conn:
            quoted = ", ".join(_quote_col(c) for c in self._meta.columns)
            for row in conn.execute(text(f"SELECT {quoted} FROM {self._meta.cache_key}")).mappings():
                yield dict(row)

    def query(
        self,
        sort_by: str | None = None,
        sort_desc: bool = False,
        filter_col: str | None = None,
        filter_val: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> QueryResult:
        """Run SQL against the cached table with sort/filter/pagination."""
        result = self._cache.query(
            self._meta.cache_key,
            sort_by=sort_by,
            sort_desc=sort_desc,
            filter_col=filter_col,
            filter_val=filter_val,
            limit=limit,
            offset=offset,
        )
        # Should never be None since we hold a valid handle, but satisfy type checker
        if result is None:
            return QueryResult(
                columns=self.columns,
                rows=[],
                count=0,
                total_count=0,
                cache_key=self.cache_key,
                source=self.source,
            )
        return result

    def invalidate(self) -> None:
        """Drop this cached result from the registry and DB."""
        self._cache.invalidate(cache_key=self._meta.cache_key)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quote_col(name: str) -> str:
    """Quote a column name with SQLite bracket syntax."""
    # Escape any closing bracket in the name itself
    escaped = name.replace("]", "]]")
    return f"[{escaped}]"


def _infer_col_type(value: Any) -> str:
    """Infer a SQLite affinity string from a Python value."""
    if isinstance(value, bool):
        return "INTEGER"
    if isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float):
        return "REAL"
    return "TEXT"


def _build_create_table(table_name: str, columns: list[str], first_row: dict[str, Any] | None) -> str:
    """Build a CREATE TABLE statement with quoted column names."""
    col_defs = []
    for col in columns:
        if first_row is not None and col in first_row:
            affinity = _infer_col_type(first_row[col])
        else:
            affinity = "TEXT"
        col_defs.append(f"  {_quote_col(col)} {affinity}")
    col_list = ",\n".join(col_defs)
    return f"CREATE TABLE IF NOT EXISTS [{table_name}] (\n{col_list}\n)"


# ---------------------------------------------------------------------------
# ResultCache
# ---------------------------------------------------------------------------

REGISTRY_TABLE = "cached_results_registry"


class ResultCache:
    """Manages cached tabular results in the workspace's ``main.db``.

    Each result is stored as a dedicated SQLite table named by its cache key
    (e.g., ``rc_a1b2c3d4e5f67890``).  A ``cached_results_registry`` table
    tracks metadata (source, columns, TTL, creation time).

    Cache keys are deterministic SHA-256 digests of ``(tool_name, params)``,
    so the same query always maps to the same key.  This enables MCP tools to
    detect hits before re-executing expensive catalog reads.

    Typical usage::

        rc = ResultCache(engine)
        rc.ensure_schema()
        key = ResultCache.cache_key("table_read", table="Subject")
        if not rc.has(key):
            rows = fetch_from_catalog(...)
            rc.store(key, columns, rows, meta)
        result = rc.get(key)
        df = result.to_dataframe()
    """

    REGISTRY_TABLE = REGISTRY_TABLE

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._metadata = MetaData()
        self._registry = Table(
            REGISTRY_TABLE,
            self._metadata,
            Column("cache_key", String, primary_key=True),
            Column("source", String, nullable=False),
            Column("tool_name", String, nullable=False),
            Column("params_json", Text, nullable=False),
            Column("columns_json", Text, nullable=False),
            Column("row_count", Integer, nullable=False, default=0),
            Column("created_at", Float, nullable=False),
            Column("ttl_seconds", Integer, nullable=True),
        )

    # ---- schema ----

    def ensure_schema(self) -> None:
        """Create the registry table if it doesn't exist."""
        self._metadata.create_all(self._engine)

    # ---- cache-key generation ----

    # Cache keys are used as SQLite table names. The generated form is
    # always ``rc_<16 hex chars>`` — safe to concatenate into DDL/DML.
    # Public methods that accept a cache_key validate against this pattern
    # to prevent SQL injection if a caller bypasses ``cache_key()``.
    # The regex allows alphanumeric + underscore after the ``rc_`` prefix
    # to accommodate test fixtures that use human-readable keys; the
    # ``rc_`` prefix and character-class restriction still prevent
    # arbitrary SQL injection (no brackets, quotes, semicolons, etc.).
    _CACHE_KEY_PATTERN = re.compile(r"^rc_[A-Za-z0-9_]{1,128}$")

    @staticmethod
    def cache_key(tool_name: str, **params) -> str:
        """Generate a deterministic cache key: ``rc_{sha256_prefix}``."""
        normalized: dict[str, Any] = {}
        for k, v in sorted(params.items()):
            if v is None:
                continue
            if isinstance(v, list) and all(isinstance(i, str) for i in v):
                normalized[k] = sorted(v)
            else:
                normalized[k] = v
        key_str = f"{tool_name}:{json.dumps(normalized, sort_keys=True)}"
        digest = hashlib.sha256(key_str.encode()).hexdigest()[:16]
        return f"rc_{digest}"

    @classmethod
    def _validate_cache_key(cls, cache_key: str) -> None:
        """Validate that *cache_key* matches the ``rc_<hex>`` format.

        Raises :class:`ValueError` otherwise. This is a defense-in-depth
        check — the :meth:`cache_key` helper always produces conforming
        keys, but public methods accept keys from callers directly (e.g.,
        from a URL parameter), so we validate before interpolating them
        into SQL identifiers.
        """
        if not isinstance(cache_key, str) or not cls._CACHE_KEY_PATTERN.fullmatch(cache_key):
            raise ValueError(
                f"Invalid cache_key {cache_key!r}; must match 'rc_<hex>' "
                f"(use ResultCache.cache_key() to generate valid keys)."
            )

    # ---- metadata helpers ----

    def _row_to_meta(self, row: Any) -> CachedResultMeta:
        return CachedResultMeta(
            cache_key=row["cache_key"],
            source=row["source"],
            tool_name=row["tool_name"],
            params=json.loads(row["params_json"]),
            columns=json.loads(row["columns_json"]),
            row_count=row["row_count"],
            created_at=row["created_at"],
            ttl_seconds=row["ttl_seconds"],
        )

    def get_meta(self, cache_key: str) -> CachedResultMeta | None:
        """Get metadata for a cached entry (regardless of expiry).

        Raises:
            ValueError: If *cache_key* doesn't match the ``rc_<hex>`` format.
        """
        self._validate_cache_key(cache_key)
        with self._engine.connect() as conn:
            row = conn.execute(select(self._registry).where(self._registry.c.cache_key == cache_key)).mappings().first()
        if row is None:
            return None
        return self._row_to_meta(row)

    # ---- has ----

    def has(self, cache_key: str) -> bool:
        """Return True if a non-expired entry exists for *cache_key*.

        Raises:
            ValueError: If *cache_key* doesn't match the ``rc_<hex>`` format.
        """
        self._validate_cache_key(cache_key)
        meta = self.get_meta(cache_key)
        if meta is None:
            return False
        return not meta.is_expired()

    # ---- store ----

    def store(
        self,
        cache_key: str,
        columns: list[str],
        rows: list[dict[str, Any]],
        meta: CachedResultMeta,
    ) -> None:
        """Store *rows* as a new table.  Replaces any existing entry.

        Raises:
            ValueError: If *cache_key* doesn't match the ``rc_<hex>`` format.
        """
        self._validate_cache_key(cache_key)
        first_row = rows[0] if rows else None
        create_sql = _build_create_table(cache_key, columns, first_row)

        with self._engine.begin() as conn:
            # Drop existing data table if present
            conn.execute(text(f"DROP TABLE IF EXISTS [{cache_key}]"))

            # Create the data table
            conn.execute(text(create_sql))

            # Batch insert rows
            if rows:
                quoted_cols = ", ".join(_quote_col(c) for c in columns)
                placeholders = ", ".join(f":col_{i}" for i in range(len(columns)))
                insert_sql = f"INSERT INTO [{cache_key}] ({quoted_cols}) VALUES ({placeholders})"
                bound_rows = [{f"col_{i}": row.get(col) for i, col in enumerate(columns)} for row in rows]
                conn.execute(text(insert_sql), bound_rows)

            # Upsert registry entry
            conn.execute(delete(self._registry).where(self._registry.c.cache_key == cache_key))
            conn.execute(
                insert(self._registry),
                {
                    "cache_key": cache_key,
                    "source": meta.source,
                    "tool_name": meta.tool_name,
                    "params_json": json.dumps(meta.params, sort_keys=True),
                    "columns_json": json.dumps(columns),
                    "row_count": len(rows),
                    "created_at": meta.created_at,
                    "ttl_seconds": meta.ttl_seconds,
                },
            )

    # ---- get ----

    def get(self, cache_key: str) -> CachedResult | None:
        """Get a CachedResult handle, or None if missing/expired."""
        meta = self.get_meta(cache_key)
        if meta is None or meta.is_expired():
            return None
        return CachedResult(meta, self._engine, self)

    # ---- query ----

    def query(
        self,
        cache_key: str,
        sort_by: str | None = None,
        sort_desc: bool = False,
        filter_col: str | None = None,
        filter_val: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> QueryResult | None:
        """Query a cached result with sort/filter/pagination."""
        meta = self.get_meta(cache_key)
        if meta is None or meta.is_expired():
            return None

        columns = meta.columns
        quoted_cols = ", ".join(_quote_col(c) for c in columns)
        table_ref = f"[{cache_key}]"

        # Build WHERE clause
        where_clause = ""
        bind_params: dict[str, Any] = {}
        if filter_col is not None and filter_val is not None and filter_col in columns:
            where_clause = f" WHERE {_quote_col(filter_col)} LIKE :filter_pattern"
            bind_params["filter_pattern"] = f"%{filter_val}%"

        # Count total matching rows
        count_sql = f"SELECT COUNT(*) FROM {table_ref}{where_clause}"
        with self._engine.connect() as conn:
            total_count = conn.execute(text(count_sql), bind_params).scalar() or 0

        # Build ORDER BY
        order_clause = ""
        if sort_by is not None and sort_by in columns:
            direction = "DESC" if sort_desc else "ASC"
            order_clause = f" ORDER BY {_quote_col(sort_by)} {direction}"

        # Full query with pagination
        data_sql = f"SELECT {quoted_cols} FROM {table_ref}{where_clause}{order_clause} LIMIT :limit OFFSET :offset"
        bind_params = dict(bind_params, limit=limit, offset=offset)

        with self._engine.connect() as conn:
            result_rows = conn.execute(text(data_sql), bind_params).mappings().all()

        rows = [dict(r) for r in result_rows]
        return QueryResult(
            columns=columns,
            rows=rows,
            count=len(rows),
            total_count=int(total_count),
            cache_key=cache_key,
            source=meta.source,
        )

    # ---- list ----

    def list_cached(self) -> list[CachedResultMeta]:
        """List all non-expired entries; lazily clean up expired ones."""
        with self._engine.connect() as conn:
            rows = conn.execute(select(self._registry)).mappings().all()

        all_metas = [self._row_to_meta(r) for r in rows]
        live = []
        expired_keys = []
        for m in all_metas:
            if m.is_expired():
                expired_keys.append(m.cache_key)
            else:
                live.append(m)

        if expired_keys:
            with self._engine.begin() as conn:
                for key in expired_keys:
                    conn.execute(delete(self._registry).where(self._registry.c.cache_key == key))
                    conn.execute(text(f"DROP TABLE IF EXISTS [{key}]"))

        return live

    # ---- invalidate ----

    def invalidate(
        self,
        cache_key: str | None = None,
        source: str | None = None,
    ) -> int:
        """Invalidate entries by key, source, or all. Returns count removed."""
        # Gather keys to remove
        if cache_key is not None:
            meta = self.get_meta(cache_key)
            keys_to_remove = [cache_key] if meta is not None else []
        elif source is not None:
            with self._engine.connect() as conn:
                rows = conn.execute(select(self._registry.c.cache_key).where(self._registry.c.source == source)).all()
            keys_to_remove = [r[0] for r in rows]
        else:
            with self._engine.connect() as conn:
                rows = conn.execute(select(self._registry.c.cache_key)).all()
            keys_to_remove = [r[0] for r in rows]

        if not keys_to_remove:
            return 0

        with self._engine.begin() as conn:
            for key in keys_to_remove:
                conn.execute(delete(self._registry).where(self._registry.c.cache_key == key))
                conn.execute(text(f"DROP TABLE IF EXISTS [{key}]"))

        return len(keys_to_remove)
