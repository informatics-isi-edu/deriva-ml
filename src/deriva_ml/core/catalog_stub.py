"""Offline-mode stand-in for ErmrestCatalog.

In ``ConnectionMode.offline``, ``DerivaML.__init__`` sets
``self.catalog`` to an instance of this class instead of constructing
a real ``ErmrestCatalog``. Any user code that tries to reach through
``ml.catalog.<something>`` gets a loud ``DerivaMLReadOnlyError``
instead of an ``AttributeError: 'NoneType' object has no attribute
'...'``.

This keeps the mixin code simple: nothing has to check ``if
self._mode is offline`` before using ``self.catalog``; the stub
fails exactly where an online-mode catalog operation would run.

Dunders pass through to the default ``__getattribute__`` path so
``repr()``, ``print()``, copy protocol, etc. still work. Only
attribute names that don't start with ``_`` raise
``DerivaMLReadOnlyError``.
"""
from __future__ import annotations

from deriva_ml.core.exceptions import DerivaMLReadOnlyError


class CatalogStub:
    """Placeholder for ErmrestCatalog in offline mode."""

    def __getattr__(self, name: str):
        # ``__getattr__`` is only called when normal lookup fails, so
        # this fires for user code reaching for real catalog methods.
        # Dunders (e.g., __copy__, __deepcopy__) that don't exist on
        # object trigger this too; raising AttributeError for them is
        # the Python-plumbing-correct behavior.
        if name.startswith("_"):
            raise AttributeError(name)
        raise DerivaMLReadOnlyError(
            f"catalog.{name} requires online mode; "
            f"this DerivaML instance was constructed with mode=offline"
        )

    def __repr__(self) -> str:
        return "CatalogStub(offline)"
