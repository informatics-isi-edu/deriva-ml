"""Catalog management utilities for DerivaML.

After the bag-oriented redesign (ADR-0006), catalog cloning runs
through a single bag-pipeline path:

- :func:`~deriva_ml.catalog.clone_via_bag.clone_via_bag` is the
  canonical clone entry point. ``CatalogBagBuilder`` writes a bag
  from the source; ``BagCatalogLoader`` loads it into the
  destination. On completion the destination's
  :class:`~deriva_ml.catalog.provenance.CatalogProvenance`
  annotation is written with ``creation_method=CLONE`` and
  phase-1 :class:`~deriva_ml.catalog.provenance.CloneDetails`.
- :func:`~deriva_ml.catalog.localize.localize_assets` is the
  phase-2 leg of the split-phase slice copy. It moves asset
  bytes server-to-server (Hatrac has no native cross-server
  copy primitive) and updates the destination's provenance
  annotation with the localization stats.
- :func:`~deriva_ml.catalog.clone.create_ml_workspace` is the
  legacy spelling, reimplemented on top of ``clone_via_bag``.
  Same parameter shape as before; legacy-only parameters
  (``truncate_oversized``, ``prune_hidden_fkeys``, async
  concurrency knobs, etc.) are accepted but no longer load-bearing
  and emit a deprecation warning when set away from default.
- :class:`deriva.bag.traversal.DanglingFKStrategy` is what
  ``clone_via_bag``'s policy actually uses. ``OrphanStrategy``
  (re-exported from ``clone.py``) is a legacy alias of the
  same enum; new code should import ``DanglingFKStrategy``
  directly from ``deriva.bag.traversal``.

The provenance API
(:class:`~deriva_ml.catalog.provenance.CatalogProvenance` etc.)
lives in :mod:`deriva_ml.catalog.provenance` and is re-exported
here for convenience.
"""

from deriva_ml.catalog.clone import (
    OrphanStrategy,
    create_ml_workspace,
)
from deriva_ml.catalog.clone_via_bag import (
    CloneViaBagResult,
    clone_via_bag,
)
from deriva_ml.catalog.localize import (
    LocalizeResult,
    localize_assets,
)
from deriva_ml.catalog.provenance import (
    CATALOG_PROVENANCE_URL,
    CatalogCreationMethod,
    CatalogProvenance,
    CloneDetails,
    get_catalog_provenance,
    set_catalog_provenance,
)

__all__ = [
    # Provenance API
    "CATALOG_PROVENANCE_URL",
    "CatalogCreationMethod",
    "CatalogProvenance",
    "CloneDetails",
    "get_catalog_provenance",
    "set_catalog_provenance",
    # Bag-pipeline clone (preferred)
    "CloneViaBagResult",
    "clone_via_bag",
    # Legacy clone surface (now on top of clone_via_bag)
    "OrphanStrategy",
    "create_ml_workspace",
    # Localize
    "LocalizeResult",
    "localize_assets",
]
