"""Shared fixtures for ``tests/model/``.

Re-exports the ``materialized_bag_with_feature`` fixture from
``tests/dataset/conftest.py`` so the bag-view tests in
``test_catalog.py`` can use it without duplicating the
fixture body. Pytest discovers fixtures by name in the
conftest chain; importing the fixture function from the
neighbouring conftest module + binding it locally makes it
available to all tests in this directory.
"""
from __future__ import annotations

# Re-export the fixture verbatim. ``pytest`` collects fixtures
# by their decorated function in the conftest namespace; the
# import name doesn't matter as long as the original
# ``@pytest.fixture`` decoration is preserved (which it is, on
# the source).
from tests.dataset.conftest import (
    MaterializedBagFixture,
    materialized_bag_with_feature,
)

__all__ = ["MaterializedBagFixture", "materialized_bag_with_feature"]
