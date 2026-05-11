"""Legacy re-export shim. Use :mod:`deriva.bag.schema` instead.

The implementations live upstream at :mod:`deriva.bag.schema`. This
shim preserves the historical
``from deriva_ml.model.schema_builder import SchemaBuilder, SchemaORM``
import path so callers inside and outside the project keep working
through the deriva.bag migration. New code should import directly
from :mod:`deriva.bag.schema` (or from :mod:`deriva.bag`, where the
public surface is exposed at the package level).

The CSV-to-Python type decorators that used to live in this module
(``ERMRestBoolean``, ``StringToFloat``, etc.) are also re-exported
here for back-compat. They live canonically in
:mod:`deriva.bag.database`.

This shim is scheduled for removal once all internal callers have
been updated.
"""

from __future__ import annotations

from deriva.bag.database import (  # noqa: F401  (re-export)
    ERMRestBoolean,
    StringToDate,
    StringToDateTime,
    StringToFloat,
    StringToInteger,
)
from deriva.bag.schema import SchemaBuilder, SchemaORM  # noqa: F401

__all__ = [
    "ERMRestBoolean",
    "SchemaBuilder",
    "SchemaORM",
    "StringToDate",
    "StringToDateTime",
    "StringToFloat",
    "StringToInteger",
]
