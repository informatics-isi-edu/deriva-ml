"""Unit tests for CatalogStub — the offline-mode ErmrestCatalog stand-in."""
from __future__ import annotations

import pytest

from deriva_ml.core.catalog_stub import CatalogStub
from deriva_ml.core.exceptions import DerivaMLReadOnlyError


def test_attribute_access_raises_read_only_error():
    stub = CatalogStub()
    with pytest.raises(DerivaMLReadOnlyError) as ei:
        stub.getCatalogModel()
    assert "getCatalogModel" in str(ei.value)
    assert "offline" in str(ei.value).lower()


def test_method_call_raises_read_only_error():
    stub = CatalogStub()
    with pytest.raises(DerivaMLReadOnlyError) as ei:
        stub.connect_ermrest("42")
    assert "connect_ermrest" in str(ei.value)


def test_repr_is_readable():
    stub = CatalogStub()
    assert repr(stub) == "CatalogStub(offline)"


def test_dunder_access_does_not_raise():
    """Python internals can probe dunders freely — we only guard real attrs.

    A raise on every dunder would break print(), repr(), copy.copy(), etc.
    The contract is "raise on attribute access that LOOKS like user code
    reaching into the catalog"; dunders are Python plumbing.
    """
    stub = CatalogStub()
    # These must NOT raise
    s = repr(stub)
    d = str(stub)
    assert "CatalogStub" in s
    assert "CatalogStub" in d
