"""Legacy re-export shim. Use :mod:`deriva.bag.loader` instead.

The implementation lives upstream at
:class:`deriva.bag.loader.ForeignKeyOrderer`. This shim preserves the
historical import path
``from deriva_ml.model.fk_orderer import ForeignKeyOrderer`` so callers
inside and outside the project keep working through the deriva.bag
migration. New code should import directly from
:mod:`deriva.bag.loader` (or from :mod:`deriva.bag`, where the symbol
is exposed at the package level).

This shim is scheduled for removal once all internal callers have
been updated and any external code dependent on the legacy path has
had a release cycle to migrate.
"""

from __future__ import annotations

from deriva.bag.loader import ForeignKeyOrderer  # noqa: F401  (re-export)

__all__ = ["ForeignKeyOrderer"]
