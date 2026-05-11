"""Catalog management utilities for DerivaML.

After the bag-oriented redesign (ADR-0006), catalog cloning runs
through a single bag-pipeline path:

- :func:`~deriva_ml.catalog.clone_via_bag.clone_via_bag` is the
  canonical clone entry point. ``CatalogBagBuilder`` writes a bag
  from the source; ``BagCatalogLoader`` loads it into the
  destination.
- :func:`~deriva_ml.catalog.clone.create_ml_workspace` is the
  legacy spelling, reimplemented on top of ``clone_via_bag``.
  Same parameter shape as before; legacy-only parameters
  (``truncate_oversized``, ``prune_hidden_fkeys``, async
  concurrency knobs, etc.) are accepted but no longer load-bearing
  and emit a deprecation warning when set away from default.
- :data:`~deriva_ml.catalog.clone.OrphanStrategy` is an alias of
  :class:`deriva.bag.traversal.DanglingFKStrategy`;
  :data:`~deriva_ml.catalog.clone.AssetCopyMode` maps its
  legacy values onto :class:`deriva.bag.traversal.AssetMode`.

The provenance API
(:class:`~deriva_ml.catalog.provenance.CatalogProvenance` etc.)
lives in :mod:`deriva_ml.catalog.provenance` and is re-exported
here for back-compat.
"""

from deriva_ml.catalog.clone import (
    AssetCopyMode,
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
    "AssetCopyMode",
    "OrphanStrategy",
    "create_ml_workspace",
    # Localize
    "LocalizeResult",
    "localize_assets",
]
