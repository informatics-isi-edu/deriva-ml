"""Unified local SQLite layer for deriva-ml.

This package provides the persistence and query infrastructure used by
``DerivaML.workspace`` for offline caching, denormalization, and asset-manifest
tracking.  Key public symbols:

- :class:`Workspace`: per-catalog working database.  Entry point for all
  local-DB operations.
- :class:`LocalSchema`: thin adapter over ``SchemaBuilder``/``SchemaORM`` that
  holds the SQLAlchemy ORM for all catalog tables.
- :class:`PagedFetcher` + :class:`ErmrestPagedClient`: stream rows from
  ERMrest into local SQLite with keyset pagination and RID-batch fetching.
- :func:`denormalize` + :class:`DenormalizeResult`: unified denormalization
  engine that builds SQLAlchemy JOINs against local SQLite.
- :class:`ResultCache` + :class:`CachedResult` + :class:`CachedResultMeta`
  + :class:`QueryResult`: result-cache layer storing tabular reads as
  named SQLite tables with TTL and sort/filter/pagination.
- :class:`ManifestStore`: crash-safe SQLite replacement for the old
  ``asset-manifest.json`` file.

See ``docs/superpowers/specs/2026-04-15-unified-local-db-design.md`` for design.
See ``README.md`` in this directory for a short orientation.
"""

from __future__ import annotations

from deriva_ml.local_db.denormalize import DenormalizeResult, denormalize
from deriva_ml.local_db.manifest_store import ManifestStore
from deriva_ml.local_db.paged_fetcher import PagedFetcher
from deriva_ml.local_db.paged_fetcher_ermrest import ErmrestPagedClient
from deriva_ml.local_db.result_cache import CachedResult, CachedResultMeta, QueryResult, ResultCache
from deriva_ml.local_db.schema import LocalSchema
from deriva_ml.local_db.workspace import Workspace

__all__ = [
    "CachedResult",
    "CachedResultMeta",
    "DenormalizeResult",
    "ErmrestPagedClient",
    "LocalSchema",
    "ManifestStore",
    "PagedFetcher",
    "QueryResult",
    "ResultCache",
    "Workspace",
    "denormalize",
]
