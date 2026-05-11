"""Legacy re-export shim. Use :mod:`deriva.bag.sources` instead.

The implementations live upstream at :mod:`deriva.bag.sources`. This
shim preserves the historical
``from deriva_ml.model.data_sources import ...`` import path so
callers inside and outside the project keep working through the
deriva.bag migration. New code should import directly from
:mod:`deriva.bag.sources` (or from :mod:`deriva.bag`, where the
public surface is exposed at the package level).

This shim is scheduled for removal once all internal callers have
been updated.
"""

from __future__ import annotations

from deriva.bag.sources import (  # noqa: F401  (re-export)
    BagDataSource,
    CatalogDataSource,
    DataFrameDataSource,
    DataSource,
    IterableDataSource,
    LocalDBDataSource,
)

__all__ = [
    "BagDataSource",
    "CatalogDataSource",
    "DataFrameDataSource",
    "DataSource",
    "IterableDataSource",
    "LocalDBDataSource",
]
