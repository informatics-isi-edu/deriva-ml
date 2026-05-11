"""Legacy re-export shim. Use :mod:`deriva_ml.model.deriva_ml_bag_view` instead.

The implementation lives at
:class:`deriva_ml.model.deriva_ml_bag_view.DerivaMLBagView`. This
shim preserves the historical
``from deriva_ml.model.deriva_ml_database import DerivaMLDatabase``
import path so callers inside and outside the project keep working
through the rename. New code should import from
:mod:`deriva_ml.model.deriva_ml_bag_view` (or via
:mod:`deriva_ml.model`, where the lazy import is updated).

The class itself was renamed because ``DerivaMLDatabase`` suggested
a database-backed alternative; the class is more accurately a *view*
over a bag's contents (per ADR-0006's three-class consumer layer:
``BagDatabase`` → ``DatabaseModel`` → ``DerivaMLBagView``). The
``DerivaMLDatabase`` name is aliased here for back-compat.

This shim is scheduled for removal after one release cycle.
"""

from __future__ import annotations

from deriva_ml.model.deriva_ml_bag_view import (  # noqa: F401  (re-export)
    DerivaMLBagView,
)

#: Legacy alias. Use :class:`DerivaMLBagView` in new code.
DerivaMLDatabase = DerivaMLBagView

__all__ = ["DerivaMLBagView", "DerivaMLDatabase"]
