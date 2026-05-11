"""Legacy re-export shim. Use :mod:`deriva.bag.sqlite_helpers` instead.

The implementations live upstream at :mod:`deriva.bag.sqlite_helpers`.
This shim preserves the historical
``from deriva_ml.local_db.sqlite_helpers import ...`` import path so
callers inside and outside the project keep working through the
deriva.bag migration.

This module previously held a subtle bug in :func:`ensure_schema_meta`:
the ``INSERT OR IGNORE`` only catches *duplicate-PK* conflicts, so if
the on-disk version was *different* from the expected version (a
``v1`` database opened by ``v2`` code), the INSERT proceeded and
accumulated a second row at the new version, leaving ``MAX(version)``
to report the wrong (newest) value. The upstream version fixes this
by checking for an existing row first and only inserting when the
table is empty. The fix is implicit in this shim — callers picked it
up by switching imports.

New code should import directly from :mod:`deriva.bag.sqlite_helpers`
(or from :mod:`deriva.bag`, where the public surface is exposed at
the package level). This shim is scheduled for removal once all
internal callers have been updated.
"""

from __future__ import annotations

from deriva.bag.sqlite_helpers import (  # noqa: F401  (re-export)
    DEFAULT_BUSY_TIMEOUT_MS,
    SCHEMA_META_TABLE,
    SchemaVersionError,
    attach_database,
    create_wal_engine,
    detach_database,
    ensure_schema_meta,
)

__all__ = [
    "DEFAULT_BUSY_TIMEOUT_MS",
    "SCHEMA_META_TABLE",
    "SchemaVersionError",
    "attach_database",
    "create_wal_engine",
    "detach_database",
    "ensure_schema_meta",
]
