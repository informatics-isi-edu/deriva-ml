"""Catalog management utilities for DerivaML.

This package exposes two clone paths:

- The **legacy** :func:`create_ml_workspace` flow in
  :mod:`~deriva_ml.catalog.clone` — bespoke three-stage clone
  with orphan handling, oversized-value truncation, and async
  per-table concurrency. Still in place; the production-tested
  path for catalogs that need its full feature set.
- The **bag-based** :func:`clone_via_bag` flow in
  :mod:`~deriva_ml.catalog.clone_via_bag` — the ADR-0006
  pipeline: ``CatalogBagBuilder`` writes a bag from the source,
  ``BagCatalogLoader`` loads it into the destination. The bag in
  the middle is a durable artifact (debuggable, citable via MINID,
  re-loadable on failure).

Prefer ``clone_via_bag`` for new clone work; reach for
``create_ml_workspace`` only when its legacy parameters are
needed.
"""

from deriva_ml.catalog.clone import (
    AssetCopyMode,
    AssetFilter,
    CatalogCreationMethod,
    CatalogProvenance,
    CloneCatalogResult,
    CloneDetails,
    CloneReport,
    CloneReportSummary,
    OrphanStrategy,
    create_ml_workspace,
    get_catalog_provenance,
    set_catalog_provenance,
)
from deriva_ml.catalog.clone_via_bag import (
    CloneViaBagResult,
    clone_via_bag,
)
from deriva_ml.catalog.localize import (
    LocalizeResult,
    localize_assets,
)

__all__ = [
    "AssetCopyMode",
    "AssetFilter",
    "CatalogCreationMethod",
    "CatalogProvenance",
    "CloneCatalogResult",
    "CloneDetails",
    "CloneReport",
    "CloneReportSummary",
    "CloneViaBagResult",
    "LocalizeResult",
    "OrphanStrategy",
    "clone_via_bag",
    "create_ml_workspace",
    "get_catalog_provenance",
    "localize_assets",
    "set_catalog_provenance",
]
