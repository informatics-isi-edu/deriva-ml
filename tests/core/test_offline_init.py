"""Integration tests for offline-mode ``DerivaML.__init__`` and
``refresh_schema()``. Tests that need a live catalog are gated on
``DERIVA_HOST`` and skip when it's unset. Tests that don't need
a live catalog (e.g., missing-cache error, host/catalog mismatch)
always run.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_offline_without_cache_raises(tmp_path):
    """Offline init against a workspace with no cache file raises
    ``DerivaMLConfigurationError``."""
    from deriva_ml import ConnectionMode, DerivaML
    from deriva_ml.core.exceptions import DerivaMLConfigurationError

    with pytest.raises(DerivaMLConfigurationError) as ei:
        DerivaML(
            hostname="example.org",
            catalog_id="1",
            mode=ConnectionMode.offline,
            working_dir=tmp_path,
        )
    assert "offline" in str(ei.value).lower()
    assert "cache" in str(ei.value).lower()
