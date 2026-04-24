"""Pytest configuration for doctests in deriva_ml.

Adds common symbols to the doctest namespace so Example: blocks in
docstrings don't need boilerplate imports. All catalog-dependent
examples should use ``# doctest: +SKIP`` since there is no live
catalog at doctest-collection time.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _doctest_namespace(doctest_namespace):
    """Populate doctest namespace with commonly-used symbols."""
    # These imports must succeed without a live catalog connection.
    from deriva_ml.feature import FeatureRecord

    doctest_namespace["FeatureRecord"] = FeatureRecord
