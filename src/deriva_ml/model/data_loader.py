"""Legacy re-export shim. Use :mod:`deriva.bag.loader` instead.

The implementation lives upstream at
:class:`deriva.bag.loader.DataLoader`. This shim preserves the
historical ``from deriva_ml.model.data_loader import DataLoader``
import path so callers inside and outside the project keep working
through the deriva.bag migration. New code should import directly
from :mod:`deriva.bag.loader` (or from :mod:`deriva.bag`, where the
symbol is exposed at the package level).

This shim is scheduled for removal once all internal callers have
been updated.
"""

from __future__ import annotations

from deriva.bag.loader import (  # noqa: F401  (re-export)
    CSVSink,
    DataLoader,
    Sink,
    SQLiteSink,
)

__all__ = ["CSVSink", "DataLoader", "Sink", "SQLiteSink"]
